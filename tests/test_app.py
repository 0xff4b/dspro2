import json
from dataclasses import asdict

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly
import pytest
from dash import no_update
from shapely.geometry import box

from rentml import app, geo
from rentml.app_figures import map_figure
from rentml.intervals import (
    CQR,
    MonotoneQuantileLGBM,
    RentModelBundle,
    assign_band,
    fit_band_cutpoints,
    quantile_columns,
)
from rentml.models import make_te_lgbm

E0, N0 = 2_600_000.0, 1_200_000.0
INSIDE = geo.Address("Teststrasse 1, 8000 Aach", E0 + 500, N0 + 500, egid=123)
FEATURES = ["area", "rooms", "area_per_room", "oev"]
TE_COLS = ["re_canton", "re_municipality"]
FAST = {"n_estimators": 40, "learning_rate": 0.1, "min_child_samples": 20}


def _text(component: object) -> str:
    return json.dumps(component, cls=plotly.utils.PlotlyJSONEncoder, ensure_ascii=False)


@pytest.fixture(scope="module")
def layers() -> dict[str, gpd.GeoDataFrame]:
    """Two municipalities (ZH, GE) as one WGS84 layer per level; GE is below MIN_MAP_CELL."""
    units = gpd.GeoDataFrame(
        {
            "municipality_id": [1, 6621],
            "district_id": [101, 2500],
            "canton_id": [1, 25],
            "municipality_name": ["Aach", "Genève"],
            "district_name": ["Nord", "Genève"],
            "canton": ["ZH", "GE"],
            "canton_name": ["Zürich", "Genève"],
            "muni_population": [1000.0, 2000.0],
            "muni_area_km2": [1.0, 1.0],
            "muni_density": [1000.0, 2000.0],
            "lang_region": ["de", "fr"],
            "unit_id": [1, 6621],
            "name": ["Aach", "Genève"],
            "n_objects": [50, 3],
            "chf_per_m2": [25.0, np.nan],
            "rent": [2000.0, np.nan],
        },
        geometry=[box(E0, N0, E0 + 1000, N0 + 1000), box(E0 + 4000, N0, E0 + 5000, N0 + 1000)],
        crs=geo.LV95,
    ).to_crs(geo.WGS84)
    return dict.fromkeys(geo.MAP_LEVELS, units)


@pytest.fixture(scope="module")
def bundle() -> RentModelBundle:
    """A tiny bundle with the real contract: target-encoded keys + monotone quantile models."""
    rng = np.random.default_rng(0)
    n = 1200
    zh = rng.random(n) < 0.6
    raw = pd.DataFrame(
        {
            "area": rng.uniform(30, 150, n),
            "rooms": rng.integers(1, 6, n).astype(float),
            "oev": rng.uniform(0, 100, n),
            "re_canton": np.where(zh, "ZH", "GE"),
            "re_municipality": np.where(zh, "ZH/101/1", "GE/2500/6621"),
            "lang_region": np.where(zh, "de", "fr"),
        }
    )
    raw["area_per_room"] = raw["area"] / raw["rooms"]
    y = 6.0 + 0.01 * raw["area"] + 0.2 * zh + rng.normal(0, 0.1, n)
    pipe = make_te_lgbm(FEATURES, TE_COLS, FAST, monotone_increasing=["area"])
    pipe.fit(raw, y)
    encoder = pipe[:-1]
    X = encoder.transform(raw)
    monotone = [0, 0, 1, 0, 0, 0]
    qmodel = MonotoneQuantileLGBM((0.1, 0.25, 0.5, 0.75, 0.9), FAST, monotone, patience=5)
    qmodel.fit(X, y.to_numpy())
    q_lo, q50, q_hi = quantile_columns(qmodel.predict(X), qmodel.levels_, (0.1, 0.5, 0.9)).T
    cuts = fit_band_cutpoints(q50)
    groups = pd.DataFrame({"lang_region": raw["lang_region"], "band": assign_band(q50, cuts)})
    calibrator = CQR(min_group_size=100).fit(q_lo, q_hi, y.to_numpy(), groups)
    return RentModelBundle(encoder, pipe[-1], qmodel, calibrator, cuts)


@pytest.fixture
def res(bundle: RentModelBundle, layers: dict) -> app.Resources:
    qoli = pd.DataFrame({"qoli": [70.0, 40.0], "transport": [80.0, 30.0]}, index=[1, 6621])
    return app.Resources(bundle, layers, {"min_count": 20}, qoli.rename_axis("municipality_id"))


@pytest.fixture
def payload() -> dict:
    dwellings = [
        geo.Dwelling(1, "101", "EG", 2.5, 61.0),
        geo.Dwelling(2, "201", "1. Stock", None, None),
    ]
    location = geo.LocationData(year_built=1990, apartments=2, oev=50.0, dwellings=dwellings)
    return asdict(location) | {"address": app.encode_address(INSIDE)}


def test_create_app_and_all_pages_render(res: app.Resources) -> None:
    dash_app = app.create_app(res)
    assert "tenant-result.children" in dash_app.callback_map
    assert "..tenant-area.value...tenant-rooms.value.." in dash_app.callback_map
    pages = {path: _text(app.render_page(path, res)) for path in ("/", "/vermieter", "/karte")}
    assert all(f'"tenant-{i}"' in pages["/"] for i in ("address", "dwelling", "rent", "text"))
    assert (
        '"landlord-dwelling"' in pages["/vermieter"] and "landlord-rent" not in pages["/vermieter"]
    )
    assert '"map-level"' in pages["/karte"]


def test_tools_show_notice_without_bundle_but_map_works(layers: dict) -> None:
    res = app.Resources(layers=layers, meta={"min_count": 20})
    assert "Modell nicht geladen" in _text(app.render_page("/mietcheck", res))
    assert '"map-level"' in _text(app.render_page("/karte", res))
    assert "Kartendaten fehlen" in _text(app.render_page("/karte", app.Resources()))


def test_load_resources_without_files(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(app.MODEL_ENV, raising=False)
    res = app.load_resources(tmp_path / "missing.joblib", tmp_path)
    assert res.bundle is None and res.layers is None and res.qoli is None


def test_dwelling_choice_and_prefill(payload: dict) -> None:
    options, hint = app.dwelling_choice(payload)
    assert [o["label"] for o in options] == [
        "EWID 1 · 101 · EG · 2.5 Zi · 61 m²",
        "EWID 2 · 201 · 1. Stock",
    ]
    assert "2 Wohnungen" in hint
    assert app.prefill(0, payload) == (61.0, 2.5)
    assert app.prefill(1, payload) == (no_update, no_update)  # unknown area/rooms stay editable
    assert app.prefill(None, payload) == (no_update, no_update)


def test_building_without_dwellings_keeps_manual_input() -> None:
    options, hint = app.dwelling_choice(asdict(geo.LocationData()))
    assert options == [] and "keine Wohnungen" in hint
    assert app.dwelling_choice(None) == ([], "")


def test_select_address_looks_up_dwellings(res, monkeypatch) -> None:
    calls = []

    def lookup(address: geo.Address, session: object = None) -> geo.LocationData:
        calls.append(address)
        return geo.LocationData(
            year_built=1990, dwellings=[geo.Dwelling(1, "101", "EG", 2.5, 61.0)]
        )

    monkeypatch.setattr(geo, "lookup_location", lookup)
    store, options, _hint = app.select_address(res, app.encode_address(INSIDE))
    assert calls == [INSIDE]
    assert store["dwellings"][0]["area"] == 61.0 and len(options) == 1
    assert store["address"] == app.encode_address(INSIDE)
    assert app.select_address(res, None) == (None, [], "")


def test_prepare_looks_up_again_if_payload_is_stale(res, payload, monkeypatch) -> None:
    other = geo.Address("Teststrasse 2, 8000 Aach", E0 + 400, N0 + 400, egid=124)
    calls = []

    def lookup(address: geo.Address, session: object = None) -> geo.LocationData:
        calls.append(address)
        return geo.LocationData(year_built=2000)

    monkeypatch.setattr(geo, "lookup_location", lookup)
    location, _row = app.prepare(res, app.encode_address(other), payload, 80.0, 3.5)
    assert calls == [other] and location.year_built == 2000
    location, _row = app.prepare(res, app.encode_address(INSIDE), payload, 80.0, 3.5)
    assert calls == [other] and location.dwellings[0].area == 61.0


def test_address_options_keep_selection(res, monkeypatch) -> None:
    other = geo.Address("Andere 2, Bern", E0, N0, None)
    monkeypatch.setattr(geo, "search_addresses", lambda query, session=None: [other])
    selected = app.encode_address(INSIDE)
    options = app.address_options(res, "Ande", selected)
    assert [o["label"] for o in options] == [INSIDE.label, other.label]
    assert app.decode_address(options[1]["value"]) == other


def test_build_row_has_admin_units_keys_and_features(layers: dict) -> None:
    location = geo.LocationData(year_built=1990, apartments=4, land_area=200.0, oev=60.0)
    row = app.build_row(INSIDE, location, 80.0, 3.5, layers["municipality"])
    r = row.iloc[0]
    assert (r["municipality_id"], r["canton"], r["lang_region"]) == (1, "ZH", "de")
    assert r["re_municipality"] == "ZH/101/1"
    assert r["area_per_room"] == pytest.approx(80 / 3.5)
    assert r["land_area_per_apartment"] == 50.0 and np.isnan(r["solar"])


def test_build_row_rejects_bad_input(layers: dict) -> None:
    units = layers["municipality"]
    with pytest.raises(app.InputError, match="Wohnfläche"):
        app.build_row(INSIDE, geo.LocationData(), 5.0, 2.0, units)
    far = geo.Address("Ausland", E0 + 100_000, N0 + 100_000)
    with pytest.raises(app.InputError, match="keiner Schweizer Gemeinde"):
        app.build_row(far, geo.LocationData(), 80.0, 3.0, units)


def test_tenant_result_shows_verdict_drivers_qoli_and_legal(res, payload, monkeypatch) -> None:
    monkeypatch.setattr(geo, "lookup_location", lambda *a, **k: pytest.fail("no lookup needed"))
    address = app.encode_address(INSIDE)
    out = _text(app.tenant_result(res, address, payload, 80.0, 3.5, 99_999.0, None))
    assert "Über der erwarteten Spanne" in out
    assert "am stärksten beeinflusst" in out and "Lebensqualität der Gemeinde: 70" in out
    assert "Schlichtungsbehörde im Kanton ZH" in out and "mieterverband.ch" in out
    low = _text(app.tenant_result(res, address, payload, 80.0, 3.5, 50.0, None))
    assert "Unter der erwarteten Spanne" in low


def test_tenant_result_reports_input_errors(res, payload) -> None:
    out = _text(app.tenant_result(res, None, payload, 80.0, 3.5, 1500.0, None))
    assert "Keine Schätzung möglich" in out and "Bitte Adresse" in out
    out = _text(app.tenant_result(res, app.encode_address(INSIDE), payload, 80, 3.5, None, None))
    assert "positive Nettomiete" in out


def test_attribute_labels_from_extraction_dict() -> None:
    evidence = {"has_lift": "Lift", "view": "Seesicht"}
    found = {"has_lift": True, "has_garden": False, "view": "lake", "parking": "none",
             "renovation_year": 2021, "evidence": evidence}  # fmt: skip
    assert app.attribute_labels(found) == [
        ("Lift", "Lift"),
        ("Seesicht", "Seesicht"),
        ("Renoviert 2021", ""),
    ]
    assert app.attributes_block(None) is None and app.attributes_block("   ") is None


def test_attributes_block_uses_rule_based_extract(monkeypatch) -> None:
    text = pytest.importorskip("rentml.text")
    monkeypatch.setattr(text, "rule_based_extract", lambda s: {"has_lift": True, "evidence": {}})
    assert "Lift" in _text(app.attributes_block("Wohnung mit Lift"))


def test_landlord_result_has_band_and_what_if(res, payload) -> None:
    out = _text(app.landlord_result(res, app.encode_address(INSIDE), payload, 80.0, 3.5))
    assert "Empfohlenes Angebotsband" in out and "What-if: Wohnfläche" in out
    assert "Wohnfläche (m²)" in out and "Ihre Angaben" in out


def test_map_figure_greys_out_small_units(layers: dict) -> None:
    fig = map_figure(layers["canton"], min_count=20)
    grey, values = fig.data
    assert list(grey.locations) == [1, 6621] and list(values.locations) == [1]
    assert "weniger als 20 Inserate" in grey.text[1]
