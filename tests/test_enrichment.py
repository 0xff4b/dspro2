"""Tests for rentml.enrichment (pre-filled cache, no network)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rentml import enrichment
from rentml.geo_batch import ResponseCache

LISTINGS = pd.DataFrame(
    {
        "street": ["Bahnhofstrasse", "Bahnhofstr.", "Sonnenweg", "Seeweg"],
        "street_number": ["1", "3", None, "5"],
        "zip": ["8001", "8001", "8002", "8003"],
        "city": ["Zürich", "Zürich", "Zürich", "Zürich"],
        "east": [2683000.0, 2683100.0, 2683300.0, 2690000.0],
        "north": [1247000.0, 1247100.0, 1247300.0, 1250000.0],
        "coord_precise": [True, True, True, False],
        "rooms": [3.5, 2.5, 4.5, np.nan],
        "area": [np.nan, 60.0, np.nan, np.nan],
        "floor": [1.0, np.nan, np.nan, np.nan],
        "property_category": ["apartment", "apartment", "apartment", "house"],
        "year_built": [np.nan, 1990.0, np.nan, 1960.0],
        "land_area": [np.nan, np.nan, np.nan, 800.0],
    },
    index=pd.Index([1, 2, 3, 4], name="listing_id"),
)


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    path = tmp_path / "cache.jsonl"
    cache = ResponseCache(path)
    hit = {"label": "Bahnhofstrasse 1 8001 Zürich", "east": 2683010.0, "north": 1247010.0}
    cache.put("address", "Bahnhofstrasse 1, 8001 Zürich", hit | {"egid": 11})
    wrong = {"label": "Bahnhofstr. 30 8001 Zürich", "east": 2683100.0, "north": 1247100.0}
    cache.put("address", "Bahnhofstr. 3, 8001 Zürich", wrong | {"egid": 99})
    cache.put("address", "Seeweg 5, 8003 Zürich", {"label": "Seeweg 5 8003 Zürich",
                                                    "east": 2690500.0, "north": 1250500.0,
                                                    "egid": 22})  # fmt: skip
    cache.put("buildings", "2683100_1247100", [[33, 2683105.0, 1247100.0, "Bahnhofstrasse 3"]])
    cache.put("buildings", "2683300_1247300", [[44, 2683310.0, 1247300.0, "Feldweg 2"]])
    dwellings = [[3.0, 71.0, 1.0], [3.0, 90.0, 2.0], [2.0, 50.0, 0.0]]
    cache.put("madd", "11", {"year_built": None, "period": 8013, "apartments": 3.0,
                             "land_area": 200.0, "dwellings": dwellings})  # fmt: skip
    cache.put("madd", "33", {"year_built": 1985.0, "period": 8016, "apartments": 1.0,
                             "land_area": 90.0, "dwellings": [[2.0, 61.0, 0.0]]})  # fmt: skip
    cache.put("madd", "22", {"year_built": 1931.0, "period": 8012, "apartments": 1.0,
                             "land_area": 150.0, "dwellings": [[6.0, 180.0, 0.0]]})  # fmt: skip
    for key in ("2683010_1247010", "2683105_1247100", "2683300_1247300", "2690500_1250500"):
        cache.put("location", key, {"oev": 1000.0, "solar": 4.0, "population": 50.0,
                                    "elevation": 410.0})  # fmt: skip
    return path


def test_address_queries_need_number() -> None:
    queries = enrichment.address_queries(LISTINGS)
    assert queries.iloc[0] == "Bahnhofstrasse 1, 8001 Zürich" and pd.isna(queries.iloc[2])


def test_recover_area_rules() -> None:
    dwellings = [[3.0, 71.0, 1.0], [3.0, 90.0, 2.0], [3.0, 72.0, 3.0]]
    assert enrichment.recover_area(3.5, 1.0, "apartment", dwellings) == (71.0, 1)
    assert np.isnan(enrichment.recover_area(3.5, np.nan, "apartment", dwellings)[0])
    assert enrichment.recover_area(np.nan, np.nan, "house", [[5.0, 150.0, 0.0]]) == (150.0, 1)
    assert enrichment.recover_area(np.nan, np.nan, "apartment", dwellings) == (np.nan, 0)
    assert enrichment.recover_area(4.0, np.nan, "apartment", None) == (np.nan, 0)


def test_enrich_from_cache(cache_path: Path) -> None:
    out = enrichment.enrich(LISTINGS, cache_path=cache_path, fetch=False)
    assert out["egid"].tolist()[:2] == [11, 33]  # 2: wrong house number -> nearest building
    assert pd.isna(out.loc[3, "egid"])  # building in another street
    assert out["egid_source"].tolist()[:2] == ["address", "coordinates"]
    assert out.loc[4, "egid"] == 22  # centroid listing, matched by address
    assert out.loc[1, "year_built_gwr"] == 1953 and out.loc[1, "year_built_period"]
    assert out.loc[1, "area_gwr"] == 71.0  # 3.5 rooms -> GWR 3 rooms on floor 1
    assert out.loc[4, "area_gwr"] == 180.0  # single-dwelling house
    assert out.loc[3, "oev"] == 1000.0  # precise point without building still gets location
    assert out.columns.tolist() == enrichment.ENRICHED_COLUMNS


def test_combine_fills_and_flags(cache_path: Path) -> None:
    enriched = enrichment.enrich(LISTINGS, cache_path=cache_path, fetch=False)
    out = enrichment.combine(LISTINGS, enriched)
    assert out["area_source"].tolist()[:2] == ["gwr", "listing"] and pd.isna(out.loc[3, "area"])
    sources = out["year_built_source"]
    assert sources[[1, 2, 4]].tolist() == ["gwr_period", "gwr", "gwr"] and pd.isna(sources[3])
    assert out.loc[4, "plot_area"] == 800.0 and out.loc[4, "land_area"] == 150.0
    assert out.loc[4, "coord_source"] == "building" and out.loc[4, "east"] == 2690500.0
    assert out.loc[1, "coord_source"] == "listing" and out["coord_precise"].all()
