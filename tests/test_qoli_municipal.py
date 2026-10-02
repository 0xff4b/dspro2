"""Tests for rentml.qoli_municipal and rentml.qoli_sources (no network: HTTP is mocked)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from rentml import qoli_municipal, qoli_sources
from rentml.qoli_municipal import (
    city_statistics,
    fetch_commune_correspondence,
    fetch_sdmx_csv,
    load_tax_burden,
    remap_municipalities,
    vacancy_rates,
)
from rentml.qoli_sources import SOURCES, download, source_table

LWZ_COLS = {
    "unit": "GR_KT_GDE: Cantons, districts and communes",
    "rooms": "WOHN_ANZAHL: Number of rooms",
    "type": "LEERWOHN_TYP: Type of vacant dwelling",
    "measure": "MEASURE_DIMENSION: Measure type",
    "time": "TIME_PERIOD: Time Period",
}


class _Response:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self.content = content
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self.content]

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_source_table_lists_all_sources_with_licence() -> None:
    table = source_table()
    assert set(table.index) == set(SOURCES)
    assert table["licence"].str.len().gt(0).all() and table["reference"].str.len().gt(0).all()
    assert SOURCES["road_day"].filename == "laerm-strassenlaerm_tag_2056.tif"
    with pytest.raises(KeyError):
        source_table(["nope"])


def test_download_writes_once_and_reuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_get(url: str, **kwargs: object) -> _Response:
        calls.append(url)
        return _Response(b"data")

    monkeypatch.setattr(qoli_sources.requests, "get", fake_get)
    path = download("https://example.org/a/file.tif", tmp_path)
    assert path.read_bytes() == b"data"
    download("https://example.org/a/file.tif", tmp_path)
    assert len(calls) == 1
    monkeypatch.setattr(qoli_sources.requests, "get", lambda url, **kw: _Response(b"", 500))
    with pytest.raises(RuntimeError, match="failed"):
        download("https://example.org/a/other.tif", tmp_path)
    assert not (tmp_path / "other.tif").exists() and not (tmp_path / "other.tif.part").exists()


def _lwz_table() -> pd.DataFrame:
    rows = [
        ("261: Zürich", "_T: Total", "_T: All", "RATE: Rate", "2026", "0.1"),
        ("261: Zürich", "_T: Total", "_T: All", "RATE: Rate", "2025", "0.2"),
        ("261: Zürich", "2: 2 rooms", "_T: All", "RATE: Rate", "2026", "9.9"),
        ("261: Zürich", "_T: Total", "_T: All", "OBS: Observed", "2026", "120"),
        ("B_112: Bezirk Zürich", "_T: Total", "_T: All", "RATE: Rate", "2026", "0.3"),
        ("ZH: Zürich", "_T: Total", "_T: All", "RATE: Rate", "2026", "0.5"),
        ("6621: Genève", "_T: Total", "_T: All", "RATE: Rate", "2026", "0.4"),
    ]
    return pd.DataFrame(rows, columns=[*LWZ_COLS.values(), "OBS_VALUE"])


def test_vacancy_rates_keeps_municipal_totals_of_latest_year() -> None:
    rates = vacancy_rates(_lwz_table())
    assert rates.to_dict() == {261: 0.1, 6621: 0.4}
    assert rates.index.name == "municipality_id"


def test_city_statistics_latest_observed_value_per_core_city() -> None:
    cols = [
        "ID_RAUM: Stadt",
        "ID_INDIC_DISPR: Variablen",
        "STATISTICAL_OPERATION: Resultat",
        "TIME_PERIOD: Time",
        "OBS_VALUE",
    ]
    rows = [
        ("261901: Zurich: core city", "EC3063I: Social", "OBS: Observed", "2022", "4.0"),
        ("261901: Zurich: core city", "EC3063I: Social", "OBS: Observed", "2023", "4.5"),
        ("261901: Zurich: core city", "EC3063I: Social", "CI: Confidence", "2024", "0.3"),
        ("261920: Zurich: agglomeration", "EC3063I: Social", "OBS: Observed", "2023", "3.0"),
        ("351901: Bern: core city", "EC3063I: Social", "OBS: Observed", "2024", ""),
        ("351901: Bern: core city", "EC3063I: Social", "OBS: Observed", "2021", "5.0"),
    ]
    out = city_statistics(pd.DataFrame(rows, columns=cols), ["EC3063I", "SA1025I"])
    assert out.loc[261, "EC3063I"] == 4.5 and out.loc[261, "EC3063I_year"] == 2023
    assert out.loc[351, "EC3063I"] == 5.0
    assert out["SA1025I"].isna().all()


def test_fetch_sdmx_csv_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_get(url: str, **kwargs: object) -> _Response:
        calls.append((url, kwargs["params"]))
        return _Response(b"A,OBS_VALUE\nx,1\n")

    monkeypatch.setattr(qoli_municipal.requests, "get", fake_get)
    cache = tmp_path / "flow.csv"
    first = fetch_sdmx_csv("CH1.LWZ,DF_LWZ_1,1.1.0", cache, params={"startPeriod": "2026"})
    second = fetch_sdmx_csv("CH1.LWZ,DF_LWZ_1,1.1.0", cache)
    assert len(calls) == 1 and calls[0][0].endswith("/CH1.LWZ,DF_LWZ_1,1.1.0/all")
    pd.testing.assert_frame_equal(first, second)


def test_load_tax_burden_reads_income_column(tmp_path: Path) -> None:
    sheet = [
        ["Lediger", None, None, None, None],
        [None, None, None, "Bruttoarbeitseinkommen", None],
        [None, None, None, 50000, 80000],
        ["Kanton / canton", "Gemeindenummer / numéro", "Gemeinde", None, None],
        ["ZH", 261, "Zürich", 5.0, 8.5],
        ["ZH", 1, "Aeugst", 4.0, 7.0],
        [None, None, None, None, None],
        ["DBSt", None, None, 0.1, 0.2],
    ]
    xlsx = tmp_path / "tax.xlsx"
    pd.DataFrame(sheet).to_excel(xlsx, sheet_name="Ledig", header=False, index=False)
    tax = load_tax_burden(xlsx, gross_income=80_000)
    assert tax.to_dict() == {261: 8.5, 1: 7.0}
    with pytest.raises(KeyError, match="income"):
        load_tax_burden(xlsx, gross_income=1)


def test_correspondence_and_remap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv = (
        "InitialHistoricalCode,InitialCode,InitialName,InitialParentHistoricalCode,"
        "InitialParentName,InitialStep,TerminalHistoricalCode,TerminalCode,TerminalName,"
        "TerminalParentHistoricalCode,TerminalParentName,TerminalStep\n"
        "1,4122,Villnachern,9,B,1,2,4095,Brugg,9,B,2\n"
        "3,4095,Brugg,9,B,1,2,4095,Brugg,9,B,2\n"
        "4,261,Zürich,9,B,1,4,261,Zürich,9,B,1\n"
    ).encode()
    monkeypatch.setattr(qoli_municipal.requests, "get", lambda url, **kw: _Response(csv))
    corr = fetch_commune_correspondence("01-01-2018", "01-01-2026", tmp_path / "c.csv")
    values = pd.Series({4122: 10.0, 4095: 20.0, 261: 5.0, 9999: 1.0}, name="tax")
    out = remap_municipalities(values, corr)
    assert out.to_dict() == {261: 5.0, 4095: 15.0}  # merger: mean; unmatched dropped
    assert out.name == "tax" and out.index.name == "municipality_id"
    assert np.isfinite(out).all()
