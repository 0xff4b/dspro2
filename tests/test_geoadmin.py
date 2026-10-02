"""Tests for rentml.geoadmin with a fake HTTP session (no network)."""

import pytest
import requests

from rentml import geoadmin
from rentml.geoadmin import Address


class _Response:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.payload, self.status = payload, status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise requests.HTTPError(f"status {self.status}")

    def json(self) -> object:
        return self.payload


class _Session:
    """Answers by URL; records the parameters of every call."""

    def __init__(self, routes: dict[str, object]) -> None:
        self.routes, self.calls = routes, []

    def get(self, url: str, params: dict, timeout: float) -> _Response:
        self.calls.append((url, params))
        answer = self.routes.get(url, {})
        if callable(answer):
            answer = answer(params)
        return answer if isinstance(answer, _Response) else _Response(answer)


SEARCH = {
    "results": [
        {
            "attrs": {
                "label": "<b>Pilatusstrasse 20</b> 6003 Luzern",
                "y": 2665893.0,
                "x": 1211217.9,
                "featureId": "213022_0",
            }
        },
        {"attrs": {"label": "<b>Pilatusstrasse 20</b> 6003 Luzern", "y": 1.0, "x": 2.0}},  # dup
        {"attrs": {"label": "Ohne Koordinaten"}},
    ]
}


def _identify(params: dict) -> dict:
    if "einwohner" in params["layers"]:
        return {"results": [{"attributes": {"number": 81}}]}
    return {
        "results": [
            {"layerBodId": geoadmin.OEV_LAYER, "attributes": {"oev_erreichb_ewap": 52009}},
            {"layerBodId": geoadmin.SOLAR_LAYER, "attributes": {"klasse": 3}},
        ]
    }


def _session() -> _Session:
    find = {
        "results": [
            {"attributes": {"egid": "502130226", "gbauj": None, "garea": 7}},  # substring match
            {"attributes": {"egid": "213022", "gbauj": 1889, "ganzwhg": 1, "garea": 184}},
        ]
    }
    return _Session(
        {
            geoadmin.SEARCH_URL: SEARCH,
            geoadmin.FIND_URL: find,
            geoadmin.IDENTIFY_URL: _identify,
            geoadmin.HEIGHT_URL: {"height": "436.2"},
        }
    )


def test_search_addresses_parses_and_deduplicates() -> None:
    session = _session()
    found = geoadmin.search_addresses("Pilatusstrasse 20 Luzern", session=session)
    assert found == [Address("Pilatusstrasse 20 6003 Luzern", 2665893.0, 1211217.9, 213022)]
    assert session.calls[0][1]["sr"] == 2056
    assert geoadmin.search_addresses("ab", session=session) == []  # too short: no request
    assert len(session.calls) == 1


def test_building_attributes_uses_exact_egid() -> None:
    session = _session()
    attrs = geoadmin.building_attributes(213022, session=session)
    assert attrs == {"year_built": 1889.0, "apartments": 1.0, "land_area": 184.0}
    assert session.calls[0][1]["contains"] == "false"
    assert geoadmin.building_attributes(None, session=session)["year_built"] is None


def test_lookup_location_collects_all_inputs() -> None:
    address = Address("Pilatusstrasse 20 6003 Luzern", 2665893.0, 1211217.9, 213022)
    data = geoadmin.lookup_location(address, session=_session())
    assert data.oev == 52009.0 and data.solar == 3.0 and data.population == 81.0
    assert data.elevation == pytest.approx(436.2) and data.missing == []
    assert data.to_features()["year_built"] == 1889.0


def test_lookup_location_survives_failures() -> None:
    session = _Session(
        {
            url: _Response({}, status=503)
            for url in (geoadmin.FIND_URL, geoadmin.IDENTIFY_URL, geoadmin.HEIGHT_URL)
        }
    )
    data = geoadmin.lookup_location(Address("X", 2.6e6, 1.2e6, 1), session=session)
    assert set(data.missing) == set(data.to_features())


def test_address_roundtrip() -> None:
    address = Address("A 1 6000 Luzern", 2.66e6, 1.21e6, None)
    assert Address.from_dict(address.to_dict()) == address
