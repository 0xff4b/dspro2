"""Tests for the marketplace domain layer (in-memory database, stub lookup, no network)."""

import base64
import io

import pandas as pd
import pytest
from PIL import Image

from rentml.estimate import RentEstimator
from rentml.geoadmin import Address, LocationData
from rentml.market import photos
from rentml.market.insight import find_duplicates, merge_tags, public_description, value_scores
from rentml.market.models import (
    ApplicationDraft,
    ListingDraft,
    ListingQuery,
    MarketError,
    RentInsight,
    normalize_search,
)
from rentml.market.quality import QualityLookup
from rentml.market.seed import seed_demo
from rentml.market.service import Market
from rentml.market.store import MarketStore, hash_token

E0, N0 = 2_600_000.0, 1_200_000.0
LOCATION = LocationData(year_built=1990.0, apartments=6.0, missing=["land_area"])
FORM = {
    "title": "Helle Wohnung mit Balkon",
    "description": "Balkon und Lift. Anrufe an 079 123 45 67 oder hans@example.ch. NK CHF 200.",
    "area": 80,
    "rooms": 3.5,
    "rent_net": "1'800",
    "aux_costs": 200,
    "floor": 2,
    "available_from": "2026-12-01",
    "contact_name": "Hans Muster",
    "contact_email": "hans@example.ch",
}
APPLICATION = {
    "name": "Eva Beispiel",
    "email": "eva@example.org",
    "household_size": 2,
    "message": "Guten Tag",
    "consent": ["ok"],
}


def _address(east: float = E0 + 500, egid: int | None = 1) -> Address:
    return Address("Teststrasse 1 8000 Aach", east, N0 + 500, egid)


def _draft(**changes: object) -> ListingDraft:
    return ListingDraft.from_form({**FORM, **changes}, _address())


@pytest.fixture
def quality() -> QualityLookup:
    table = pd.DataFrame(
        {
            "municipality_id": [1, 2, 3, 4],
            "qoli": [40.0, 60.0, 80.0, None],
            "transport": [50.0, 60.0, 70.0, 10.0],
            "noise": [30.0, 60.0, 90.0, 10.0],
        }
    )
    return QualityLookup(table, "qoli-test")


@pytest.fixture
def market(rent_model, map_bundle, quality) -> Market:
    estimator = RentEstimator(rent_model, map_bundle)
    return Market(
        store=MarketStore.in_memory(),
        locator=estimator.locator,
        lookup=lambda address: LOCATION,
        estimator=estimator,
        quality=quality,
    )


def _png(size: tuple[int, int] = (2400, 1200)) -> str:
    buffer = io.BytesIO()
    Image.new("RGBA", size, (200, 30, 30, 128)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_draft_parsing_and_validation() -> None:
    draft = _draft()
    assert draft.rent_net == 1800.0 and draft.floor == 2 and draft.available_from.year == 2026
    assert normalize_search("Genève, Zürich!") == "geneve zurich"
    for changes, message in (
        ({"title": "kurz"}, "Titel"),
        ({"rooms": 3.3}, "halben Schritten"),
        ({"area": 5}, "Wohnfläche"),
        ({"rent_net": None}, "Nettomiete"),
        ({"contact_email": "keine-mail"}, "E-Mail"),
        ({"available_from": "morgen"}, "Datum"),
    ):
        with pytest.raises(MarketError, match=message):
            _draft(**changes)
    with pytest.raises(MarketError, match="Adresse"):
        ListingDraft.from_form(FORM, None)


def test_application_needs_consent_and_valid_contact() -> None:
    assert ApplicationDraft.from_form(APPLICATION).household_size == 2
    with pytest.raises(MarketError, match="zustimmen"):
        ApplicationDraft.from_form({**APPLICATION, "consent": []})
    with pytest.raises(MarketError, match="Personen"):
        ApplicationDraft.from_form({**APPLICATION, "household_size": 40})


def test_public_description_and_tags() -> None:
    text = public_description(FORM["description"])
    assert "079" not in text and "example.ch" not in text and "CHF 200" in text
    assert merge_tags("Balkon mit Seesicht", ["pets", "unknown"]) == ("balcony", "pets", "view")
    assert public_description(None) == ""


def test_publish_stores_snapshot_and_search_filters(market) -> None:
    listing, token = market.publish(_draft(), selected_tags=["pets"])
    insight = listing.insight
    assert insight.verdict in ("below", "within", "above") and insight.lo < insight.hi
    assert RentInsight.from_dict(insight.to_dict()) == insight
    assert listing.place.municipality == "Aach" and listing.location.missing == ["land_area"]
    assert "079" not in listing.description and set(listing.tags) == {"balcony", "lift", "pets"}
    store = market.store
    assert store.listing_by_token(token).id == listing.id
    assert store.listing_by_token("wrong") is None and len(hash_token(token)) == 64

    def ids(**filters: object) -> list[int]:
        return [item.id for item in store.search(ListingQuery(**filters))]

    assert ids(text="AACH teststrasse") == [listing.id] and ids(text="bern") == []
    assert ids(tags=("balcony", "pets")) == [listing.id] and ids(tags=("garden",)) == []
    assert ids(rent_max=1500) == [] and ids(rooms_min=3.5, area_min=80, canton="ZH") == [listing.id]
    assert ids(text="100%") == []  # LIKE wildcards are escaped
    assert store.stats()["active"] == 1 and store.cantons() == ["ZH"]


def test_overpriced_listing_and_reprice(market) -> None:
    listing, _ = market.publish(_draft(rent_net=9000))
    assert listing.insight.verdict == "above"
    assert market.store.search(ListingQuery(fair_only=True)) == []
    cheaper = market.reprice(listing, listing.insight.expected, None)
    assert cheaper.insight.verdict == "within"
    assert market.store.listing(listing.id).rent_net == pytest.approx(listing.insight.expected)
    with pytest.raises(MarketError):
        market.reprice(listing, 5.0, None)


def test_publish_outside_switzerland_or_without_model(market) -> None:
    abroad = ListingDraft.from_form(FORM, _address(east=E0 + 50_000))
    with pytest.raises(MarketError, match="Gemeinde"):
        market.publish(abroad)
    no_model = Market(market.store, market.locator, market.lookup)
    listing, _ = no_model.publish(_draft())
    assert listing.insight is None and no_model.assess(_draft()).what_if is None


def test_duplicates_follow_dedup_rules(market) -> None:
    first, _ = market.publish(_draft())
    assert [d.id for d in market.assess(_draft()).duplicates] == [first.id]
    assert (
        find_duplicates(_draft(area=120, description="Ganz andere Wohnung im Haus"), [first]) == []
    )
    assert find_duplicates(_draft(), []) == []


def test_applications_lifecycle_and_privacy(market) -> None:
    listing, _ = market.publish(_draft())
    store = market.store
    application, token = store.add_application(listing.id, ApplicationDraft.from_form(APPLICATION))
    assert application.status == "new" and store.application_by_token(token).name == "Eva Beispiel"
    assert store.set_application_status(listing.id, application.id, "invited")
    assert not store.set_application_status(listing.id + 1, application.id, "declined")
    with pytest.raises(ValueError):
        store.set_application_status(listing.id, application.id, "ranked")
    store.set_status(listing.id, "paused")
    with pytest.raises(MarketError, match="keine Bewerbungen"):
        store.add_application(listing.id, ApplicationDraft.from_form(APPLICATION))
    assert store.withdraw_application(token) and store.applications(listing.id) == []
    assert not store.withdraw_application(token)
    store.delete_listing(listing.id)
    assert store.listing(listing.id) is None


def test_photos_are_reencoded_and_only_served_while_active(market) -> None:
    images = photos.prepare([_png()])
    with Image.open(io.BytesIO(images[0])) as image:
        assert image.format == "JPEG" and max(image.size) == photos.MAX_SIDE
    listing, _ = market.publish(_draft(), photos=images)
    assert market.store.photo(listing.photo_ids[0]) == images[0]
    market.store.set_status(listing.id, "paused")
    assert market.store.photo(listing.photo_ids[0]) is None
    for bad in ("data:text/plain;base64,aGk=", "data:image/png;base64,aGk=", "no-data-url"):
        with pytest.raises(photos.PhotoError):
            photos.prepare([bad])
    with pytest.raises(photos.PhotoError, match="Höchstens"):
        photos.prepare(["x"] * (photos.MAX_PHOTOS + 1))


def test_quality_profile_and_value_score(market, quality) -> None:
    profile = quality.profile(3)
    assert profile.qoli == 80.0 and profile.percentile == 100.0 and "noise" in profile.dimensions
    assert quality.profile(4) is None and quality.profile(99) is None
    assert QualityLookup.load("missing.parquet") is None
    listings = [market.publish(_draft(rent_net=1500 + 100 * i))[0] for i in range(3)]
    assert value_scores(listings, quality) == {}  # fewer peers than the minimum
    scores = value_scores(listings, quality, min_listings=2)
    assert scores[listings[0].id] > scores[listings[2].id]  # same QoLI, cheaper wins
    assert value_scores(listings, None) == {}
    views = market.browse(ListingQuery(sort="qoli"))
    assert len(views) == 3 and views[0].quality.qoli == 40.0
    assert [v.listing.id for v in market.similar(listings[0])] == [listings[1].id, listings[2].id]


def test_seed_demo_is_offline_and_flagged(market, map_bundle) -> None:
    listings = seed_demo(market, map_bundle, n=5)
    assert len(listings) == 5 and all(item.is_demo and item.insight for item in listings)
    assert "Demo" in listings[0].address.label
    with pytest.raises(ValueError):
        seed_demo(Market(market.store, market.locator, market.lookup), map_bundle, n=1)
