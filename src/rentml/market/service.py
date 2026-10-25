"""The marketplace service: what the pages call to assess, publish, browse and re-price listings.

:class:`Market` ties the store to the models. Publishing looks up the building and location
inputs (geo.admin), rates the asking rent with the rent model and stores the snapshot with the
listing, so a listing page never has to call the network or the model again.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import pandas as pd

from rentml.config import ProjectPaths
from rentml.estimate import Apartment, Estimate, EstimateError, MunicipalityLocator, RentEstimator
from rentml.geoadmin import Address, LocationData
from rentml.mapbundle import MapBundle
from rentml.market.insight import (
    find_duplicates,
    merge_tags,
    public_description,
    rate_listing,
    value_scores,
)
from rentml.market.models import (
    Listing,
    ListingDraft,
    ListingQuery,
    MarketError,
    Place,
    RentInsight,
)
from rentml.market.quality import SPEC_FILE, QualityLookup, QualityProfile, resolve_table
from rentml.market.store import MarketStore, resolve_url

logger = logging.getLogger(__name__)

SIMILAR_ROOMS = 1.0


@dataclass(frozen=True)
class Assessment:
    """What the models say about a draft before it is published (the price assistant).

    Attributes:
        place: Municipality of the address.
        location: Building and location inputs from geo.admin.
        estimate: Full estimate with drivers and text attributes (``None`` without a model).
        insight: Storable snapshot of the estimate (``None`` without a model).
        what_if: Effect of area and room changes (``None`` without a model).
        tags: Feature tags (recognised in the text plus ticked by the landlord).
        duplicates: Active listings that look like the same object.
    """

    place: Place
    location: LocationData
    estimate: Estimate | None
    insight: RentInsight | None
    what_if: pd.DataFrame | None
    tags: tuple[str, ...]
    duplicates: list[Listing]


@dataclass(frozen=True)
class ListingView:
    """A listing with everything a card or the detail page shows next to it.

    Attributes:
        listing: The stored listing.
        quality: QoLI profile of its municipality (``None`` without the QoLI table).
        value: Value score among the active listings (``None`` if not computable).
    """

    listing: Listing
    quality: QualityProfile | None
    value: float | None


@dataclass(frozen=True)
class Market:
    """Marketplace operations on top of the store and the models.

    Attributes:
        store: Listings, photos and applications.
        locator: Municipality lookup on the map bundle.
        lookup: Building and location inputs for an address (geo.admin in production).
        estimator: Rent model; ``None`` = listings are published without a price check.
        quality: QoLI lookup; ``None`` = no quality-of-life block and no value score.
    """

    store: MarketStore
    locator: MunicipalityLocator
    lookup: Callable[[Address], LocationData]
    estimator: RentEstimator | None = None
    quality: QualityLookup | None = None

    def place(self, address: Address) -> Place:
        """Municipality of an address.

        Raises:
            MarketError: If the address lies outside the Swiss municipalities.
        """
        unit = self.locator.locate(address.east, address.north)
        if unit is None:
            raise MarketError("Die Adresse liegt in keiner Schweizer Gemeinde.")
        return Place.from_unit(unit)

    def _rate(
        self, draft: ListingDraft, location: LocationData
    ) -> tuple[Estimate | None, RentInsight | None]:
        if self.estimator is None:
            return None, None
        apartment = Apartment(draft.address, draft.area, draft.rooms, draft.description)
        try:
            return rate_listing(self.estimator, apartment, location, draft.rent_net)
        except EstimateError as exc:
            raise MarketError(str(exc)) from exc

    def assess_rent(self, draft: ListingDraft, location: LocationData) -> float:
        """Model estimate of the net rent for a draft (used to price demo listings).

        Raises:
            MarketError: Without a model or outside its domain.
        """
        _, insight = self._rate(draft, location)
        if insight is None:
            raise MarketError("Kein Mietmodell geladen.")
        return insight.expected

    def assess(self, draft: ListingDraft, selected_tags: Sequence[str] = ()) -> Assessment:
        """Rate a validated draft without storing anything (live price assistant).

        Args:
            draft: Validated draft.
            selected_tags: Feature tags the landlord ticked.

        Returns:
            The assessment.

        Raises:
            MarketError: If the address is outside Switzerland or outside the model's domain.
        """
        place = self.place(draft.address)
        location = self.lookup(draft.address)
        estimate, insight = self._rate(draft, location)
        what_if = None
        if self.estimator is not None:
            apartment = Apartment(draft.address, draft.area, draft.rooms, draft.description)
            what_if = self.estimator.what_if(apartment, location)
        peers = self.store.search(ListingQuery(municipality_id=place.municipality_id))
        return Assessment(
            place=place,
            location=location,
            estimate=estimate,
            insight=insight,
            what_if=what_if,
            tags=merge_tags(draft.description, selected_tags),
            duplicates=find_duplicates(draft, peers),
        )

    def publish(
        self,
        draft: ListingDraft,
        *,
        photos: Sequence[bytes] = (),
        selected_tags: Sequence[str] = (),
        location: LocationData | None = None,
        is_demo: bool = False,
    ) -> tuple[Listing, str]:
        """Rate and store a validated draft.

        Args:
            draft: Validated draft.
            photos: JPEG bytes from :func:`rentml.market.photos.prepare`.
            selected_tags: Feature tags the landlord ticked.
            location: Building and location inputs; ``None`` = look them up.
            is_demo: Mark the listing as synthetic.

        Returns:
            The stored listing and its secret management token.

        Raises:
            MarketError: If the address is outside Switzerland or outside the model's domain.
        """
        place = self.place(draft.address)
        location = location if location is not None else self.lookup(draft.address)
        _, insight = self._rate(draft, location)
        stored = replace(draft, description=public_description(draft.description))
        listing, token = self.store.create_listing(
            stored,
            place,
            location=location,
            insight=insight,
            tags=merge_tags(draft.description, selected_tags),
            photos=photos,
            is_demo=is_demo,
        )
        logger.info("Published listing %d in %s", listing.id, place.municipality)
        return listing, token

    def reprice(self, listing: Listing, rent_net: float, aux_costs: float | None) -> Listing:
        """Change the asking rent and re-rate it with the stored location inputs (no network).

        Raises:
            MarketError: If the new rent is outside the allowed range.
        """
        draft = ListingDraft(
            title=listing.title,
            description=listing.description,
            address=listing.address,
            area=listing.area,
            rooms=listing.rooms,
            rent_net=rent_net,
            aux_costs=aux_costs,
            floor=listing.floor,
            available_from=listing.available_from,
            contact_name=listing.contact_name,
            contact_email=listing.contact_email,
            contact_phone=listing.contact_phone,
        )
        draft.validate()
        _, insight = self._rate(draft, listing.location)
        self.store.update_rent(listing.id, rent_net, aux_costs, insight)
        return replace(listing, rent_net=rent_net, aux_costs=aux_costs, insight=insight)

    def _views(self, listings: Sequence[Listing], values: dict[int, float]) -> list[ListingView]:
        return [
            ListingView(
                listing=item,
                quality=self.quality.profile(item.place.municipality_id) if self.quality else None,
                value=values.get(item.id),
            )
            for item in listings
        ]

    def browse(self, query: ListingQuery | None = None) -> list[ListingView]:
        """Active listings matching the query, with QoLI and value score, in the query's order."""
        query = query or ListingQuery()
        active = self.store.search()  # the peer group of the value score
        found = active if query == ListingQuery() else self.store.search(query)
        views = self._views(found, value_scores(active, self.quality))
        if query.sort == "qoli":
            views.sort(key=lambda v: v.quality.qoli if v.quality else -1.0, reverse=True)
        elif query.sort == "value":
            views.sort(key=lambda v: v.value if v.value is not None else -1.0, reverse=True)
        return views

    def view(self, listing: Listing) -> ListingView:
        """One listing with its QoLI and its value score among the active listings."""
        values = value_scores(self.store.search(), self.quality)
        return self._views([listing], values)[0]

    def similar(self, listing: Listing, limit: int = 3) -> list[ListingView]:
        """Other active listings of the canton with a similar room count, closest rent first."""
        peers = [
            view
            for view in self.browse(ListingQuery(canton=listing.place.canton))
            if view.listing.id != listing.id
            and abs(view.listing.rooms - listing.rooms) <= SIMILAR_ROOMS
        ]
        peers.sort(key=lambda v: abs(v.listing.rent_net - listing.rent_net))
        return peers[:limit]


def default_market(
    bundle: MapBundle,
    estimator: RentEstimator | None,
    lookup: Callable[[Address], LocationData],
    *,
    store: MarketStore | None = None,
) -> Market:
    """Marketplace on the project's default database and QoLI table (see the ``resolve_*`` helpers).

    Args:
        bundle: Map bundle (municipality geometry).
        estimator: Rent model, if a bundle is deployed.
        lookup: Location lookup of the app's services.
        store: Store to use instead of the default database.

    Returns:
        The service; the database is only opened on first use.
    """
    paths = ProjectPaths.discover()
    return Market(
        store=store or MarketStore(resolve_url()),
        locator=estimator.locator if estimator is not None else MunicipalityLocator(bundle),
        lookup=lookup,
        estimator=estimator,
        quality=QualityLookup.load(resolve_table(), paths.models / SPEC_FILE),
    )
