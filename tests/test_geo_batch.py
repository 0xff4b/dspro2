"""Tests for rentml.geo_batch (saved responses, no network)."""

import json
from pathlib import Path

import pytest
import requests

from rentml import geo, geo_batch

MADD_XML = """<?xml version="1.0" encoding="utf-8"?>
<maddResponse xmlns="http://www.ech.ch/xmlns/eCH-0206/2"><buildingList><buildingItem>
<EGID>11</EGID><building><dateOfConstruction><periodOfConstruction>8011</periodOfConstruction>
</dateOfConstruction><surfaceAreaOfBuilding>120</surfaceAreaOfBuilding></building>
<buildingEntranceList><buildingEntranceItem><dwellingList>
<dwellingItem><EWID>1</EWID><dwelling><yearOfConstruction>1999</yearOfConstruction>
<noOfHabitableRooms>3</noOfHabitableRooms><floor>3101</floor>
<surfaceAreaOfDwelling>72</surfaceAreaOfDwelling><dwellingStatus>3004</dwellingStatus>
</dwelling></dwellingItem>
<dwellingItem><EWID>2</EWID><dwelling><noOfHabitableRooms>4</noOfHabitableRooms>
<floor>3100</floor><surfaceAreaOfDwelling>95</surfaceAreaOfDwelling>
<dwellingStatus>3004</dwellingStatus></dwelling></dwellingItem>
</dwellingList></buildingEntranceItem></buildingEntranceList>
</buildingItem></buildingList></maddResponse>"""


class FakeResponse:
    def __init__(self, payload: dict | str, status: int = 200) -> None:
        self.payload, self.status_code = payload, status

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> dict:
        if isinstance(self.payload, str):
            raise requests.JSONDecodeError("no json", self.payload, 0)
        return self.payload

    @property
    def content(self) -> bytes:
        return str(self.payload).encode()


class FakeSession:
    """Answers by URL prefix; a list payload is consumed one response per call."""

    def __init__(self, routes: dict[str, object]) -> None:
        self.routes = routes
        self.calls = 0

    def get(self, url: str, params: dict, timeout: float) -> FakeResponse:
        self.calls += 1
        for prefix, payload in self.routes.items():
            if url.startswith(prefix):
                answer = payload.pop(0) if isinstance(payload, list) else payload
                return answer if isinstance(answer, FakeResponse) else FakeResponse(answer)
        raise requests.ConnectionError(url)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(geo_batch.time, "sleep", lambda _: None)


def test_cache_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    cache = geo_batch.ResponseCache(path)
    cache.put("madd", "11", {"year_built": 1974.0})
    cache.put("address", "x", {})
    reloaded = geo_batch.ResponseCache(path)
    assert ("madd", "11") in reloaded and len(reloaded) == 2
    assert reloaded.get("madd", "11") == {"year_built": 1974.0}
    assert reloaded.get("madd", "12") is None
    assert all(json.loads(line)["kind"] for line in path.read_text().splitlines())


def test_request_retries_server_errors() -> None:
    session = FakeSession({"https://x": [FakeResponse({}, 503), FakeResponse({"a": 1})]})
    response = geo_batch._request(session, "https://x", {})
    assert response is not None and response.json() == {"a": 1} and session.calls == 2


def test_request_gives_up_after_retries() -> None:
    session = FakeSession({})
    assert geo_batch._request(session, "https://x", {}) is None
    assert session.calls == geo_batch.RETRIES


def test_fetch_address_parses_first_hit() -> None:
    payload = {
        "results": [
            {
                "attrs": {
                    "label": "Bahnhofstrasse 1 <b>8001 Zürich</b>",
                    "featureId": "123_0",
                    "y": 2683000.5,
                    "x": 1247000.25,
                }
            }
        ]
    }
    hit = geo_batch.fetch_address(FakeSession({geo.SEARCH_URL: payload}), ("q",))
    assert hit == {
        "label": "Bahnhofstrasse 1 8001 Zürich",
        "east": 2683000.5,
        "north": 1247000.25,
        "egid": 123,
    }
    assert geo_batch.fetch_address(FakeSession({geo.SEARCH_URL: {"results": []}}), ("q",)) == {}
    assert geo_batch.fetch_address(FakeSession({}), ("q",)) is None


def test_fetch_buildings_keeps_points_with_egid() -> None:
    payload = {
        "results": [
            {
                "attributes": {"egid": "11", "strname_deinr": "Weg 1"},
                "geometry": {"x": 1.0, "y": 2.0},
            },
            {"attributes": {"egid": None}, "geometry": {"x": 3.0, "y": 4.0}},
        ]
    }
    rows = geo_batch.fetch_buildings(FakeSession({geo.IDENTIFY_URL: payload}), (0.0, 0.0))
    assert rows == [[11, 1.0, 2.0, "Weg 1"]]


def test_fetch_madd_uses_period_not_dwelling_year() -> None:
    value = geo_batch.fetch_madd(FakeSession({geo.MADD_URL: MADD_XML}), (11,))
    assert value["year_built"] is None and value["period"] == 8011
    assert value["apartments"] == 2.0 and value["land_area"] == 120.0
    assert value["dwellings"] == [[4.0, 95.0, 0.0], [3.0, 72.0, 1.0]]


def test_fetch_madd_unknown_and_failed() -> None:
    unknown = FakeSession({geo.MADD_URL: FakeResponse("", 404)})
    assert geo_batch.fetch_madd(unknown, (1,)) == {}
    assert geo_batch.fetch_madd(FakeSession({}), (1,)) is None


def test_fetch_location_needs_both_services() -> None:
    identify = {
        "results": [{"layerBodId": geo.OEV_LAYER, "attributes": {"oev_erreichb_ewap": 500}}]
    }
    routes = {geo.IDENTIFY_URL: identify, geo.HEIGHT_URL: {"height": "433.1"}}
    value = geo_batch.fetch_location(FakeSession(routes), (1.0, 2.0))
    assert value == {"oev": 500.0, "solar": None, "population": None, "elevation": 433.1}
    assert geo_batch.fetch_location(FakeSession({geo.IDENTIFY_URL: identify}), (1, 2)) is None


def test_fetch_all_skips_cached_and_keeps_failures_open(tmp_path: Path) -> None:
    cache = geo_batch.ResponseCache(tmp_path / "c.jsonl")
    cache.put("k", "a", 1)
    seen: list[tuple] = []

    def fetch(session: requests.Session, args: tuple) -> object | None:
        seen.append(args)
        return None if args == ("c",) else args[0]

    failed = geo_batch.fetch_all(cache, "k", {"a": ("a",), "b": ("b",), "c": ("c",)}, fetch, 2)
    assert failed == 1 and sorted(seen) == [("b",), ("c",)]
    assert cache.get("k", "b") == "b" and ("k", "c") not in cache
