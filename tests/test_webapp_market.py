"""Tests for the marketplace pages with stub services; skipped without the app extra."""

import pytest

pytest.importorskip("dash")

from rentml.estimate import RentEstimator  # noqa: E402 -- after the optional-dependency skip
from rentml.geoadmin import Address, LocationData  # noqa: E402
from rentml.market.service import Market  # noqa: E402
from rentml.market.store import MarketStore  # noqa: E402
from rentml.webapp import (  # noqa: E402
    market_browse,
    market_detail,
    market_manage,
    market_publish,
)
from rentml.webapp.app import create_app  # noqa: E402
from rentml.webapp.components import encode_address  # noqa: E402
from rentml.webapp.services import Services  # noqa: E402

ADDRESS = encode_address(Address("Teststrasse 1 8000 Aach", 2_600_500.0, 1_200_500.0, 1))
FORM = {
    "title": "Helle Wohnung mit Balkon",
    "description": "Balkon, Lift, Tel. 079 123 45 67",
    "area": 80,
    "rooms": 3.5,
    "rent_net": 1800,
    "contact_name": "Hans Muster",
    "contact_email": "hans@example.ch",
}
APPLICATION = {
    "name": "Eva Beispiel",
    "email": "eva@example.org",
    "message": "Hallo",
    "consent": ["ok"],
}


@pytest.fixture
def services(rent_model, map_bundle) -> Services:
    estimator = RentEstimator(rent_model, map_bundle)
    location = LocationData(year_built=1990.0, missing=["land_area"])
    market = Market(MarketStore.in_memory(), estimator.locator, lambda a: location, estimator)
    return Services(estimator, lambda q: [], lambda a: location, market)


def _publish(services: Services, **changes: object) -> tuple[str, str]:
    view, error = market_publish.publish(
        services.market, {**FORM, **changes}, ADDRESS, ["pets"], None, True, "http://host/"
    )
    assert error is None
    text = str(view)
    token = text.split("http://host/verwalten/")[1].split("'")[0]
    return text, token


def test_publish_flow_shows_secret_link_and_validates(services) -> None:
    text, token = _publish(services)
    assert "Ihr Inserat ist online" in text and "Preis-Check" in text
    assert services.market.store.listing_by_token(token).tags == ("balcony", "lift", "pets")
    market = services.market
    _, error = market_publish.publish(market, FORM, ADDRESS, None, None, False)
    assert "bestätigen" in str(error)
    _, error = market_publish.publish(market, {**FORM, "title": ""}, ADDRESS, None, None, True)
    assert "Titel" in str(error)
    _, error = market_publish.publish(market, FORM, ADDRESS, None, ["kein-bild"], True)
    assert "Foto" in str(error)


def test_price_assistant_needs_no_title_or_contact(services) -> None:
    form = {"area": 80, "rooms": 3.5, "rent_net": 9000, "description": "Balkon"}
    text = str(market_publish.run_assistant(services.market, form, ADDRESS, None))
    assert "Empfohlenes Angebotsband" in text and "Über der erwarteten Spanne" in text
    assert "What-if" in text and "Balkon/Terrasse" in text
    assert "Adresse" in str(market_publish.run_assistant(services.market, form, None, None))
    _publish(services)
    again = str(market_publish.run_assistant(services.market, FORM, ADDRESS, None))
    assert "bereits inseriert" in again


def test_start_and_browse_pages(services) -> None:
    market = services.market
    assert "Noch keine Inserate" in str(market_browse.start_layout(market))
    _publish(services)
    assert "Helle Wohnung" in str(market_browse.start_layout(market))
    query = market_browse.query_from_filters({"text": "aach", "rooms": "3", "tags": ["balcony"]})
    page = str(market_browse.browse_layout(market, query))
    assert "1 Inserat" in page and "Geplant" in page and "Suchabo" in page
    none = market_browse.query_from_filters({"rent": "abc", "canton": "ge", "fair": ["fair"]})
    assert none.rent_max is None and none.canton == "GE" and none.fair_only
    assert "Filter lockern" in str(market_browse.results(market, none))


def test_listing_page_application_and_status(services) -> None:
    market = services.market
    _, token = _publish(services)
    listing = market.store.listing_by_token(token)
    page = str(market_detail.detail_layout(market, str(listing.id)))
    assert "RentLens Preis-Check" in page and "Jetzt bewerben" in page and "079" not in page
    assert "Art. 270 OR" in page and "Besichtigungstermin" in page
    assert "nicht verfügbar" in str(market_detail.detail_layout(market, "abc"))
    _, error = market_detail.submit_application(market, listing.id, {**APPLICATION, "consent": []})
    assert "zustimmen" in str(error)
    confirmation, error = market_detail.submit_application(market, listing.id, APPLICATION)
    assert error is None and "Bewerbung eingereicht" in str(confirmation)
    secret = str(confirmation).split("/bewerbung/")[1].split("'")[0]
    assert "Eingegangen" in str(market_detail.applicant_layout(market, secret))
    assert "nicht gefunden" in str(market_detail.applicant_layout(market, "wrong"))
    assert "gelöscht" in str(market_detail.withdraw(market, secret))
    assert market.store.applications(listing.id) == []


def test_manage_dashboard_actions(services) -> None:
    market = services.market
    _, token = _publish(services)
    listing = market.store.listing_by_token(token)
    application, _ = market.store.add_application(
        listing.id, market_detail.ApplicationDraft.from_form(APPLICATION)
    )
    page = str(market_manage.manage_layout(market, token))
    assert (
        "Eva Beispiel" in page and "eva@example.org" in page and "Reihenfolge nach Eingang" in page
    )
    assert "nicht gefunden" in str(market_manage.manage_layout(market, "wrong"))
    invite = {"type": "manage-app", "app": application.id, "status": "invited"}
    assert "Zur Besichtigung eingeladen" in str(market_manage.run_action(market, token, invite))
    assert "Über der erwarteten Spanne" in str(
        market_manage.run_action(market, token, "manage-reprice", 9000, None)
    )
    assert "Nettomiete" in str(market_manage.run_action(market, token, "manage-reprice", None))
    market_manage.run_action(market, token, "manage-pause")
    assert market.store.listing(listing.id).status == "paused"
    assert "nicht verfügbar" in str(market_detail.detail_layout(market, listing.id))
    assert "Inserat gelöscht" in str(market_manage.run_action(market, token, "manage-delete"))
    assert market.store.listing(listing.id) is None
    assert "nicht gefunden" in str(market_manage.run_action(market, token, "manage-pause"))


def test_app_registers_marketplace(map_bundle, services) -> None:
    app = create_app(map_bundle, services=services)
    outputs = " ".join(app.callback_map)
    for name in ("br-results", "pub-assistant", "pub-page", "apply-box", "manage-body"):
        assert f"{name}.children" in outputs
    client = app.server.test_client()
    for path in ("/", "/inserate", "/inserieren", "/inserat/1", "/verwalten/x", "/karte"):
        assert client.get(path).status_code == 200
    assert client.get("/media/foto/1.jpg").status_code == 404


def test_manage_keeps_every_action_button_in_the_layout(services) -> None:
    # The action callback lists all buttons as inputs; Dash rejects a page that lacks one.
    market = services.market
    _, token = _publish(services)
    for status in ("active", "paused", "rented"):
        market.store.set_status(market.store.listing_by_token(token).id, status)
        page = str(market_manage.manage_layout(market, token))
        for button_id in (*market_manage.LISTING_ACTIONS, "manage-reprice", "manage-delete"):
            assert f"id='{button_id}'" in page
