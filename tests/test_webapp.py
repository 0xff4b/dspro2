"""Tests for the Dash app (state logic, figure, panel, server); skipped without the app extra."""

import pytest

pytest.importorskip("dash")

from rentml.mapbundle import assemble_bundle  # noqa: E402 -- after the optional-dependency skip
from rentml.mapdata import aggregate_listings, build_levels  # noqa: E402
from rentml.webapp import figures, home, panel, state  # noqa: E402
from rentml.webapp.app import create_app  # noqa: E402


@pytest.fixture
def bundle(map_units, map_other, map_listings):
    levels, other = build_levels(map_units, map_other, canton_names={1: "Zürich"})
    stats = aggregate_listings(map_listings, min_count=20)
    meta = {
        "min_count": 20,
        "n_objects": 30,
        "created_at": "2026-09-30T12:00:00+00:00",
        "swiss": {"chf_per_m2": 20.0, "rent": 1600.0},
    }
    return assemble_bundle(levels, other, stats, meta=meta)


@pytest.fixture
def index(bundle):
    return state.UnitIndex(bundle.units)


def _url(level: str) -> str:
    return f"/map-data/{level}.geojson"


def test_unit_keys_roundtrip() -> None:
    assert state.parse_key(state.unit_key("district", 101)) == ("district", 101)
    for bad in ("canton", "street:1", "canton:x"):
        with pytest.raises(ValueError):
            state.parse_key(bad)
    with pytest.raises(ValueError):
        state.unit_key("street", 1)


def test_unit_index_hierarchy(index) -> None:
    assert index.parent_key("municipality:2") == "district:101"
    assert index.ancestors("municipality:2") == ["canton:1", "district:101"]
    assert index.ancestors("canton:1") == []
    assert sorted(index.children("canton:1")["unit_id"]) == [101, 102]
    assert index.children("municipality:1").empty
    assert index.label("municipality:4") == "Genève · Gemeinde, GE"
    assert index.label("canton:1") == "Zürich · Kanton"
    assert index.bounds(None) == index.swiss_bounds
    assert index.bounds("canton:25")[0] == pytest.approx(2_604_000)
    options = index.search_options()
    assert options[0]["value"].startswith("canton:") and len(options) == 9


def test_view_transitions(index) -> None:
    view = dict(state.DEFAULT_VIEW)
    clicked = state.select_on_map(view, index, "municipality:3")
    assert clicked["selected"] == "municipality:3" and clicked["rev"] == 0  # map does not move
    assert state.select_on_map(view, index, "municipality:99")["selected"] is None
    searched = state.select_unit(view, index, "municipality:3")
    assert searched["level"] == "municipality" and searched["focus"] == "district:102"
    assert searched["rev"] == 1
    drilled = state.drill_down(state.select_unit(view, index, "canton:1"), index)
    assert drilled["level"] == "district" and drilled["focus"] == "canton:1"
    zoomed = state.drill_down(searched, index)
    assert zoomed["level"] == "municipality" and zoomed["focus"] == "municipality:3"
    reset = state.reset_view(zoomed)
    assert (
        reset["selected"] is None and reset["focus"] is None and reset["rev"] == zoomed["rev"] + 1
    )
    assert state.set_level(view, "district")["level"] == "district"
    with pytest.raises(ValueError):
        state.set_level(view, "street")


def test_next_view_ignores_recreated_buttons(index) -> None:
    view = dict(state.DEFAULT_VIEW)
    link = {"type": "unit-link", "key": "canton:25"}
    assert home.next_view(index, view, link, False, "canton", None, None) is None
    assert home.next_view(index, view, link, True, "canton", None, None)["selected"] == "canton:25"
    swiss = {"type": "unit-link", "key": panel.SWITZERLAND_KEY}
    assert (
        home.next_view(index, {**view, "selected": "canton:1"}, swiss, True, "canton", None, None)[
            "selected"
        ]
        is None
    )
    click = {"points": [{"customdata": ["municipality:1", "Aach"]}]}
    assert home.next_view(index, view, "map", True, "canton", click, None)["selected"] == (
        "municipality:1"
    )
    assert home.next_view(index, view, "map", True, "canton", {"points": [{}]}, None) is None
    selected = {**view, "selected": "canton:1"}
    assert home.next_view(index, selected, "search", True, "canton", None, None)["selected"] is None
    assert home.next_view(index, selected, "search", True, "canton", None, "canton:1") is None


def test_map_figure_traces(index, bundle) -> None:
    metric = figures.METRICS[0]
    view = {**state.DEFAULT_VIEW, "level": "municipality", "selected": "municipality:1"}
    fig = figures.map_figure(index, view, metric, geojson_url=_url, other_areas=bundle.other_areas)
    names = [t.name for t in fig.data]
    assert names == [
        "Seen und gemeindefreie Gebiete",
        "Gemeinde",
        "Gemeinde",
        "Kantonsgrenzen",
        "Auswahl",
    ]
    missing, valued = fig.data[1], fig.data[2]
    assert list(valued.locations) == ["1"]  # only Aach has >= 20 objects
    assert sorted(missing.locations) == ["2", "3", "4"]
    assert missing.geojson == _url("municipality")  # polygons are referenced, not embedded
    assert valued.customdata[0][0] == "municipality:1"
    assert fig.layout.geo.projection.type == "equirectangular"
    assert fig.layout.geo.projection.rotation.lat == 0


def test_geo_view_zooms_without_clipping(index) -> None:
    swiss = figures.geo_view(index.swiss_bounds, index.swiss_bounds)
    assert swiss["projection"]["scale"] == pytest.approx(1.0)
    zoomed = figures.geo_view(index.swiss_bounds, index.bounds("municipality:4"))
    assert zoomed["lonaxis"] == swiss["lonaxis"]  # ranges always span Switzerland
    assert zoomed["projection"]["scale"] > 1.0
    assert zoomed["center"]["lon"] == pytest.approx(0.045)


def test_log_metric_and_formatting(index) -> None:
    density = next(m for m in figures.METRICS if m.column == "density")
    values = figures.metric_values(index.level_table("municipality"), density)
    assert values.notna().all()
    assert figures.fmt_number(1234567.891, 1) == "1'234'567.9"
    assert density.format(None) == "–"
    extras = figures.available_metrics(
        index.units.assign(qoli=1.0), {"extra_metrics": {"qoli": {"label": "QoLI"}}}
    )
    assert extras[-1].label == "QoLI"


def test_panel_for_unit_and_switzerland(index, bundle) -> None:
    metric = figures.METRICS[0]
    view = {**state.DEFAULT_VIEW, "selected": "municipality:3"}
    text = str(panel.panel_children(index, bundle.meta, view, metric))
    assert "Cham" in text and "Weniger als 20 Inserate" in text and "Heranzoomen" in text
    overview = str(panel.panel_children(index, bundle.meta, dict(state.DEFAULT_VIEW), metric))
    assert "Schweiz" in overview and "Kantone nach" in overview


def test_app_serves_page_and_geojson(bundle) -> None:
    app = create_app(bundle)
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    response = client.get("/map-data/canton.geojson")
    assert response.status_code == 200 and response.json["type"] == "FeatureCollection"
    assert "max-age" in response.headers["Cache-Control"]
    assert client.get("/map-data/street.geojson").status_code == 404
