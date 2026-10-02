"""Tests for rentml.amenities (synthetic data, Overpass mocked, no network)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from rentml import amenities
from rentml.amenities import (
    DEFAULT_ACCESS_WEIGHTS,
    OSM_CATEGORIES,
    OSM_COLUMNS,
    OVERPASS_URL,
    accessibility,
    amenity_arrays,
    distance_decay,
    fetch_osm_amenities,
)

ZURICH_HB_WGS84 = (47.37817, 8.54018)  # (lat, lon)
ZURICH_HB_LV95 = np.array([2_683_187.0, 1_248_065.0])


# --- distance decay / accessibility ------------------------------------------------------------


def test_distance_decay_boundaries_and_monotonicity() -> None:
    d = np.array([0.0, 400.0, 1400.0, 2400.0, 5000.0, np.inf])
    np.testing.assert_allclose(distance_decay(d), [1.0, 1.0, 0.5, 0.0, 0.0, 0.0], atol=1e-12)
    grid = distance_decay(np.linspace(0.0, 3000.0, 3001))
    assert np.all(np.diff(grid) <= 1e-12)
    assert np.all((grid >= 0.0) & (grid <= 1.0))
    two_d = distance_decay(np.array([[np.nan, 100.0]]))
    assert two_d.shape == (1, 2) and np.isnan(two_d[0, 0]) and two_d[0, 1] == 1.0
    assert distance_decay(np.array([150.0]), full_until=100.0, zero_at=200.0)[0] == 0.5


def test_distance_decay_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        distance_decay(np.array([-1.0]))
    with pytest.raises(ValueError):
        distance_decay(np.array([1.0]), full_until=500.0, zero_at=500.0)
    with pytest.raises(ValueError):
        distance_decay(np.array([1.0]), full_until=-1.0)


def _points() -> np.ndarray:
    return np.vstack([ZURICH_HB_LV95, ZURICH_HB_LV95 + [5000.0, 0.0]])


def test_accessibility_scores_per_category_and_total() -> None:
    amenities = {
        "supermarket": ZURICH_HB_LV95[None, :],
        "restaurant": np.repeat(ZURICH_HB_LV95[None, :], 10, axis=0),
        "pharmacy": np.empty((0, 2)),
    }
    out = accessibility(_points(), amenities)
    assert list(out.columns) == ["acc_supermarket", "acc_restaurant", "acc_pharmacy", "acc_total"]
    np.testing.assert_allclose(out["acc_supermarket"], [100.0, 0.0])
    np.testing.assert_allclose(out["acc_restaurant"], [100.0, 0.0])
    np.testing.assert_allclose(out["acc_pharmacy"], [0.0, 0.0])
    w_sum = 3.0 + sum(DEFAULT_ACCESS_WEIGHTS["restaurant"]) + 1.0
    expected_total = 100.0 * (3.0 + sum(DEFAULT_ACCESS_WEIGHTS["restaurant"])) / w_sum
    np.testing.assert_allclose(out["acc_total"], [expected_total, 0.0])


def test_accessibility_nearest_weights_and_decay_kwargs() -> None:
    one_restaurant = {"restaurant": ZURICH_HB_LV95[None, :]}
    out = accessibility(_points()[:1], one_restaurant)
    expected = 100.0 * 0.75 / sum(DEFAULT_ACCESS_WEIGHTS["restaurant"])
    assert out["acc_restaurant"].iloc[0] == pytest.approx(expected)
    shop = {"supermarket": (ZURICH_HB_LV95 + [1400.0, 0.0])[None, :]}
    assert accessibility(_points()[:1], shop)["acc_supermarket"].iloc[0] == pytest.approx(50.0)
    wide = {"full_until": 2000.0, "zero_at": 3000.0}
    wide_out = accessibility(_points()[:1], shop, decay_kwargs=wide)
    assert wide_out["acc_supermarket"].iloc[0] == pytest.approx(100.0)
    custom = accessibility(_points()[:1], shop, weights={"supermarket": [1.0, 1.0]})
    assert custom["acc_supermarket"].iloc[0] == pytest.approx(25.0)  # second slot is empty


def test_accessibility_nan_points_and_invalid_input() -> None:
    pts = np.vstack([ZURICH_HB_LV95, [np.nan, np.nan]])
    out = accessibility(pts, {"park": ZURICH_HB_LV95[None, :]})
    assert out["acc_park"].iloc[0] == pytest.approx(100.0)
    assert out.iloc[1].isna().all()
    with pytest.raises(KeyError):
        accessibility(pts, {"park": pts}, weights={"gym": [1.0]})
    with pytest.raises(ValueError):
        accessibility(np.array([1.0, 2.0, 3.0]), {"park": pts})
    with pytest.raises(ValueError):
        accessibility(pts, {"park": pts}, weights={"park": [-1.0]})
    with pytest.raises(ValueError):
        accessibility(pts, {})


def test_accessibility_keeps_dataframe_index_and_nullable_coordinates() -> None:
    idx = pd.Index([101, 205], name="listing_id")
    coords = {"east": [ZURICH_HB_LV95[0], None], "north": [ZURICH_HB_LV95[1], None]}
    pts = pd.DataFrame(coords, index=idx).astype("Float64")
    out = accessibility(pts, {"park": ZURICH_HB_LV95[None, :]})
    assert out.index.equals(idx)
    assert out.loc[101, "acc_park"] == pytest.approx(100.0) and out.loc[205].isna().all()


def test_amenity_arrays_keeps_empty_categories_in_total() -> None:
    east, north = ZURICH_HB_LV95
    osm = pd.DataFrame(
        {"category": ["park", "park", "gym"], "east": [east, 0.0, 1.0], "north": [north, 0.0, 2.0]}
    )
    arrays = amenity_arrays(osm)
    assert list(arrays) == list(OSM_CATEGORIES)
    np.testing.assert_array_equal(arrays["gym"], [[1.0, 2.0]])
    assert arrays["supermarket"].shape == (0, 2) and arrays["park"].shape == (2, 2)
    out = accessibility(ZURICH_HB_LV95[None, :], arrays)
    w_sum = 3.0 + sum(DEFAULT_ACCESS_WEIGHTS["restaurant"]) + 6.0  # 6 categories with [1.0]
    assert out["acc_total"].iloc[0] == pytest.approx(100.0 / w_sum)  # only the park is near
    assert list(amenity_arrays(osm, ["park"])) == ["park"]
    with pytest.raises(KeyError, match="east"):
        amenity_arrays(osm.drop(columns="east"))


# --- OSM download (mocked) ---------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _install_fake_post(
    monkeypatch: pytest.MonkeyPatch, responses: list[_FakeResponse | requests.RequestException]
) -> tuple[list[tuple[str, str, float]], list[float]]:
    calls: list[tuple[str, str, float]] = []  # (url, query, http timeout)
    sleeps: list[float] = []

    def fake_post(
        url: str, *, data: dict[str, str], headers: dict[str, str], timeout: float
    ) -> _FakeResponse:
        assert "User-Agent" in headers
        calls.append((url, data["data"], timeout))
        if isinstance(item := responses.pop(0), requests.RequestException):
            raise item
        return item

    monkeypatch.setattr(amenities.requests, "post", fake_post)
    monkeypatch.setattr(amenities.time, "sleep", sleeps.append)
    return calls, sleeps


def _supermarket_payload() -> dict[str, object]:
    lat, lon = ZURICH_HB_WGS84
    return {
        "elements": [
            {"type": "node", "id": 1, "lat": lat, "lon": lon, "tags": {"name": "Migros"}},
            {"type": "way", "id": 2, "center": {"lat": lat + 0.001, "lon": lon}, "tags": {}},
            {"type": "node", "id": 1, "lat": lat, "lon": lon, "tags": {"name": "Migros"}},
            {"type": "relation", "id": 3, "tags": {"name": "no coordinates"}},
        ]
    }


CATS = {"supermarket": OSM_CATEGORIES["supermarket"], "park": OSM_CATEGORIES["park"]}
OK_PARK = _FakeResponse({"elements": [{"type": "node", "id": 9, "lat": 46.95, "lon": 7.44}]})


def test_fetch_osm_amenities_parses_converts_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = [_FakeResponse(_supermarket_payload()), OK_PARK]
    calls, sleeps = _install_fake_post(monkeypatch, responses)
    cache = tmp_path / "osm" / "amenities.parquet"
    osm = fetch_osm_amenities(cache, categories=CATS)
    assert list(osm.columns) == OSM_COLUMNS
    assert osm["category"].tolist() == ["supermarket", "supermarket", "park"]  # dup + no-coord out
    assert osm["osm_id"].tolist() == [1, 2, 9]
    node = osm.iloc[0][["east", "north"]].to_numpy(dtype=float)
    assert np.linalg.norm(node - ZURICH_HB_LV95) < 500.0
    assert osm.iloc[1]["north"] - osm.iloc[0]["north"] == pytest.approx(111.0, abs=5.0)
    assert cache.is_file() and len(calls) == 2 and sleeps == [2.0]
    url, query, http_timeout = calls[0]
    assert url == OVERPASS_URL and http_timeout == 360
    assert query.startswith("[out:json][timeout:300];")
    assert 'nwr["shop"="supermarket"](45.77,5.9,47.85,10.55);out tags center;' in query
    # Cached: no further request is made.
    monkeypatch.setattr(amenities.requests, "post", _raise_network)
    pd.testing.assert_frame_equal(fetch_osm_amenities(cache, categories=CATS), osm)


def _raise_network(url: str, **kwargs: object) -> _FakeResponse:
    raise AssertionError(f"network must not be used ({url})")


def test_fetch_osm_amenities_overwrite_retries_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "amenities.parquet"
    park_only = {"park": OSM_CATEGORIES["park"]}
    responses = [_FakeResponse({}, 429), _FakeResponse({}, 504), OK_PARK]
    calls, sleeps = _install_fake_post(monkeypatch, responses)
    osm = fetch_osm_amenities(cache, categories=park_only, overwrite=True)
    assert len(osm) == 1 and len(calls) == 3 and sleeps == [30.0, 60.0]
    timed_out = {"elements": [], "remark": 'runtime error: Query timed out in "query"'}
    _install_fake_post(monkeypatch, [_FakeResponse(timed_out)])
    with pytest.raises(RuntimeError, match="incomplete"):
        fetch_osm_amenities(cache, categories=park_only, overwrite=True)
    _install_fake_post(monkeypatch, [_FakeResponse({}, 400)])
    with pytest.raises(requests.HTTPError):
        fetch_osm_amenities(tmp_path / "other.parquet", categories=park_only)
    _install_fake_post(monkeypatch, [_FakeResponse({}, 429)])
    with pytest.raises(requests.HTTPError):
        fetch_osm_amenities(tmp_path / "other.parquet", categories=park_only, max_retries=0)
    assert not (tmp_path / "other.parquet").exists()
    with pytest.raises(ValueError, match="bbox"):
        fetch_osm_amenities(tmp_path / "x.parquet", bbox=(47.0, 8.0, 46.0, 9.0))


def test_fetch_osm_amenities_handles_empty_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_post(monkeypatch, [_FakeResponse({"elements": []})])
    osm = fetch_osm_amenities(tmp_path / "empty.parquet", categories={"gym": OSM_CATEGORIES["gym"]})
    assert osm.empty and list(osm.columns) == OSM_COLUMNS


def test_fetch_osm_amenities_resumes_and_keys_cache_by_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "amenities.parquet"
    reset = requests.ConnectionError("connection reset")
    _install_fake_post(monkeypatch, [_FakeResponse(_supermarket_payload()), reset])
    with pytest.raises(requests.ConnectionError):
        fetch_osm_amenities(cache, categories=CATS, max_retries=0)
    assert not cache.exists()
    calls, sleeps = _install_fake_post(monkeypatch, [OK_PARK])
    osm = fetch_osm_amenities(cache, categories=CATS)  # supermarket comes from its part file
    assert len(calls) == 1 and '["leisure"="park"]' in calls[0][1] and sleeps == []
    assert osm["category"].tolist() == ["supermarket", "supermarket", "park"] and cache.is_file()
    small = (47.3, 8.4, 47.4, 8.6)  # another bbox must not reuse the Swiss-wide cache
    calls, _ = _install_fake_post(monkeypatch, [_FakeResponse({"elements": []}), OK_PARK])
    fetch_osm_amenities(cache, bbox=small, categories=CATS)
    assert len(calls) == 2 and all("(47.3,8.4,47.4,8.6)" in query for _, query, _ in calls)
    monkeypatch.setattr(amenities.requests, "post", _raise_network)
    pd.testing.assert_frame_equal(fetch_osm_amenities(cache, categories=CATS), osm)


def test_fetch_osm_amenities_retries_connection_errors_and_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    park_only = {"park": OSM_CATEGORIES["park"]}
    flaky = [requests.ReadTimeout("slow"), requests.ConnectionError("reset"), OK_PARK]
    calls, sleeps = _install_fake_post(monkeypatch, flaky)
    osm = fetch_osm_amenities(tmp_path / "a.parquet", categories=park_only)
    assert len(osm) == 1 and len(calls) == 3 and sleeps == [30.0, 60.0]
    _install_fake_post(monkeypatch, [requests.ReadTimeout("slow")])
    with pytest.raises(requests.Timeout):
        fetch_osm_amenities(tmp_path / "b.parquet", categories=park_only, max_retries=0)


def test_osm_categories_cover_contract() -> None:
    expected = {"supermarket", "pharmacy", "doctors", "school", "kindergarten", "gym"}
    assert expected | {"restaurant", "park"} == set(OSM_CATEGORIES)
    assert all(f.startswith("[") and f.endswith("]") for f in OSM_CATEGORIES.values())
