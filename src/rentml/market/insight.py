"""Fusion of the marketplace with the DSPRO2 models.

Everything the platform derives from the models for one listing:

* the rent model's view of the asking rent (estimate, calibrated interval, verdict, market
  percentile, SHAP drivers) as a :class:`rentml.market.models.RentInsight` stored with the listing;
* feature tags from the rule-based attribute extraction of the description;
* a duplicate check with the project's deduplication rules (:mod:`rentml.dedup`);
* the value score (quality of life relative to price, :func:`rentml.qoli.value_score`) among the
  active listings.

Only listings are rated. Applicants are never scored, ranked or filtered by a model.
"""

from collections.abc import Sequence

import pandas as pd

from rentml.dedup import assign_object_ids
from rentml.estimate import Apartment, Estimate, RentEstimator
from rentml.extraction import ExtractedAttributes, rule_based_extract
from rentml.geoadmin import LocationData
from rentml.market.models import Listing, ListingDraft, RentInsight
from rentml.market.quality import QualityLookup
from rentml.qoli import value_score
from rentml.rentcheck import RentModelBundle
from rentml.text import strip_contacts

MIN_VALUE_PEERS = 8
CONTACT_PLACEHOLDER = "[Kontakt über RentLens]"
_DRAFT_ID = 0  # listing ids start at 1

TAG_LABELS: dict[str, str] = {
    "balcony": "Balkon/Terrasse",
    "lift": "Lift",
    "garden": "Garten",
    "parking": "Parkplatz/Garage",
    "washer": "Eigene Waschmaschine",
    "pets": "Haustiere erlaubt",
    "minergie": "Minergie",
    "new_build": "Neubau",
    "attic": "Dachwohnung",
    "view": "Aussicht",
    "furnished": "Möbliert",
    "temporary": "Befristet",
}
_FLAG_TAGS: dict[str, str] = {
    "has_balcony_or_terrace": "balcony",
    "has_lift": "lift",
    "has_garden": "garden",
    "has_own_washer": "washer",
    "pets_allowed": "pets",
    "is_minergie": "minergie",
    "is_new_build": "new_build",
    "is_attic": "attic",
    "is_furnished": "furnished",
    "is_temporary": "temporary",
}


def public_description(text: str | None) -> str:
    """Description as published: e-mail addresses, URLs, IBANs and phone numbers removed.

    Contact runs through the application form, which protects applicants from being lured off
    the platform and keeps personal data out of the public text.
    """
    return strip_contacts(text, CONTACT_PLACEHOLDER)


def tags_from_attributes(attributes: ExtractedAttributes) -> tuple[str, ...]:
    """Feature tags for every attribute the extraction found to be present."""
    tags = [tag for name, tag in _FLAG_TAGS.items() if getattr(attributes, name) is True]
    if attributes.parking != "none":
        tags.append("parking")
    if attributes.view != "none":
        tags.append("view")
    return tuple(tags)


def merge_tags(description: str | None, selected: Sequence[str] | None = None) -> tuple[str, ...]:
    """Tags recognised in the description plus those the landlord ticked (known tags only)."""
    found = set(tags_from_attributes(rule_based_extract(description)))
    found |= {str(tag) for tag in selected or ()}
    return tuple(tag for tag in TAG_LABELS if tag in found)


def insight_from_estimate(estimate: Estimate, model: RentModelBundle) -> RentInsight:
    """Snapshot of an estimate that was computed with an asking rent.

    Args:
        estimate: Result of :meth:`rentml.estimate.RentEstimator.estimate` with ``asking_rent``.
        model: The bundle that produced it (for the provenance fields).

    Returns:
        The snapshot to store with the listing.

    Raises:
        ValueError: If the estimate carries no Fair-Rent Check.
    """
    check = estimate.check
    if check is None:
        raise ValueError("The estimate has no rent check; pass asking_rent to estimate()")
    drivers = tuple(
        (str(row.label), float(row.approx_pct)) for row in estimate.drivers.itertuples()
    )
    return RentInsight(
        expected=check.expected_chf,
        lo=check.lo_chf,
        hi=check.hi_chf,
        q25=check.band25_chf,
        q75=check.band75_chf,
        coverage=check.coverage,
        asking=check.asking_chf,
        verdict=check.verdict,
        percentile=check.market_percentile,
        drivers=drivers,
        missing=tuple(estimate.location.missing),
        model_version=str(model.metadata.get("version", "")),
        model_created=str(model.metadata.get("created", "")),
    )


def rate_listing(
    estimator: RentEstimator, apartment: Apartment, location: LocationData, asking: float
) -> tuple[Estimate, RentInsight]:
    """Estimate one apartment and rate its asking rent.

    Args:
        estimator: Model wrapper of the app.
        apartment: Address, area, rooms and description.
        location: Building and location inputs (geo.admin or a stored snapshot).
        asking: Asking net rent in CHF.

    Returns:
        The full estimate (for the live assistant) and its storable snapshot.

    Raises:
        rentml.estimate.EstimateError: For inputs outside the model's domain.
    """
    estimate = estimator.estimate(apartment, location, asking_rent=asking)
    return estimate, insight_from_estimate(estimate, estimator.model)


def _dedup_row(
    egid: int | None,
    east: float,
    north: float,
    rooms: float,
    area: float,
    price: float,
    description: str,
) -> dict[str, object]:
    return {
        "egid": egid,
        "east": east,
        "north": north,
        "rooms": rooms,
        "area": area,
        "price": price,
        "description": description,
    }


def find_duplicates(draft: ListingDraft, candidates: Sequence[Listing]) -> list[Listing]:
    """Listings that the deduplication rules link to the draft (same object listed twice).

    Uses :func:`rentml.dedup.assign_object_ids` with its default tolerances: same building or
    coordinates with equal rooms, area within 3 m² and rent within 10 %, or an identical
    description. Two identical flats in one building therefore match as well, so treat the result
    as a hint, not as a reason to reject the listing.

    Args:
        draft: The listing about to be published.
        candidates: Stored listings to compare with (e.g. the active ones of the municipality).

    Returns:
        The candidates that share the draft's object id.
    """
    if not candidates:
        return []
    address = draft.address
    rows = {
        _DRAFT_ID: _dedup_row(
            address.egid,
            address.east,
            address.north,
            draft.rooms,
            draft.area,
            draft.rent_net,
            public_description(draft.description),
        )
    }
    for item in candidates:
        rows[item.id] = _dedup_row(
            item.address.egid,
            item.address.east,
            item.address.north,
            item.rooms,
            item.area,
            item.rent_net,
            item.description,
        )
    object_ids = assign_object_ids(pd.DataFrame.from_dict(rows, orient="index"))
    return [item for item in candidates if object_ids[item.id] == object_ids[_DRAFT_ID]]


def value_scores(
    listings: Sequence[Listing],
    quality: QualityLookup | None,
    *,
    min_listings: int = MIN_VALUE_PEERS,
) -> dict[int, float]:
    """Value score (0-100) per listing: QoLI of the municipality relative to net CHF/m².

    The score is a percentile ranking *among the given listings* (the current offers), so it is
    descriptive and needs enough peers.

    Args:
        listings: Peer group, usually all active listings.
        quality: QoLI lookup; ``None`` = no scores.
        min_listings: Minimum number of listings with a QoLI.

    Returns:
        Listing id -> score; empty without a lookup or with fewer peers than ``min_listings``.
    """
    if quality is None:
        return {}
    qoli: dict[int, float] = {}
    for item in listings:
        profile = quality.profile(item.place.municipality_id)
        if profile is not None:
            qoli[item.id] = profile.qoli
    if len(qoli) < min_listings:
        return {}
    price = pd.Series({item.id: item.chf_per_m2 for item in listings if item.id in qoli})
    scores = value_score(pd.Series(qoli), price).dropna()
    return {int(listing_id): float(score) for listing_id, score in scores.items()}
