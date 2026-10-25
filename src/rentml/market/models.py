"""Domain objects of the marketplace and their validation (no Dash, no database).

User-facing validation messages are German like the rest of the app. Everything a form sends is
parsed by ``from_form`` and checked by ``validate``; both raise :class:`MarketError`.
"""

import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from rentml.estimate import AREA_RANGE, ROOMS_RANGE
from rentml.geoadmin import Address, LocationData

LISTING_STATUSES: tuple[str, ...] = ("active", "paused", "rented")
APPLICATION_STATUSES: tuple[str, ...] = ("new", "invited", "accepted", "declined")
VERDICTS: tuple[str, ...] = ("below", "within", "above")
SORTS: tuple[str, ...] = ("newest", "rent_asc", "rent_desc", "deal", "qoli", "value")

TITLE_RANGE: tuple[int, int] = (5, 120)
NAME_RANGE: tuple[int, int] = (2, 120)
MAX_DESCRIPTION = 4000
MAX_MESSAGE = 2000
RENT_RANGE: tuple[float, float] = (200.0, 30_000.0)
AUX_RANGE: tuple[float, float] = (0.0, 5_000.0)
FLOOR_RANGE: tuple[int, int] = (-1, 60)
HOUSEHOLD_RANGE: tuple[int, int] = (1, 12)

_EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[A-Za-z]{2,24}$")
_PHONE_RE = re.compile(r"^[+0-9 ()/.-]{6,24}$")
_COMBINING_MARKS = re.compile("[̀-ͯ]")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


class MarketError(ValueError):
    """Invalid user input; the message is shown to the user (German)."""


def normalize_search(text: object) -> str:
    """Casefold, strip accents and punctuation, so ``"Genève"`` matches ``"geneve"``."""
    folded = unicodedata.normalize("NFKD", str(text or "")).casefold()
    return _NON_ALNUM.sub(" ", _COMBINING_MARKS.sub("", folded)).strip()


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


def _optional_number(value: object, label: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(str(value).replace("'", "").replace(",", "."))
    except ValueError:
        raise MarketError(f"{label}: bitte eine Zahl angeben.") from None
    if not math.isfinite(number):
        raise MarketError(f"{label}: bitte eine Zahl angeben.")
    return number


def _required_number(value: object, label: str) -> float:
    number = _optional_number(value, label)
    if number is None:
        raise MarketError(f"Bitte {label} angeben.")
    return number


def _day(value: object, label: str) -> date | None:
    if value is None or not str(value).strip():
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        raise MarketError(f"{label}: ungültiges Datum.") from None


def _check_range(value: float, bounds: tuple[float, float], label: str, unit: str = "") -> None:
    if not bounds[0] <= value <= bounds[1]:
        suffix = f" {unit}" if unit else ""
        raise MarketError(
            f"{label} muss zwischen {bounds[0]:.0f} und {bounds[1]:.0f}{suffix} liegen."
        )


def _check_contact(name: str, email: str, phone: str | None) -> None:
    if not NAME_RANGE[0] <= len(name) <= NAME_RANGE[1]:
        raise MarketError("Bitte Vor- und Nachnamen angeben.")
    if not _EMAIL_RE.match(email):
        raise MarketError("Bitte eine gültige E-Mail-Adresse angeben.")
    if phone and not _PHONE_RE.match(phone):
        raise MarketError("Telefonnummer: nur Ziffern, Leerzeichen und + ( ) / . - verwenden.")


@dataclass(frozen=True)
class Place:
    """Administrative location of a listing (from the map bundle's municipality row).

    Attributes:
        municipality_id: BFS municipality number.
        municipality: Municipality name.
        district: District name.
        canton: Canton abbreviation, e.g. ``"LU"``.
        lang_region: Language region (``de``, ``fr``, ``it``), the calibration group.
    """

    municipality_id: int
    municipality: str
    district: str
    canton: str
    lang_region: str

    @classmethod
    def from_unit(cls, unit: pd.Series) -> "Place":
        """Build from a municipality row of :class:`rentml.estimate.MunicipalityLocator`."""
        return cls(
            municipality_id=int(unit.name),
            municipality=str(unit["name"]),
            district=str(unit["district_name"]),
            canton=str(unit["canton"]),
            lang_region=str(unit["lang_region"]),
        )


@dataclass(frozen=True)
class RentInsight:
    """Model snapshot stored with a listing: what the rent model says about the asking rent.

    Attributes:
        expected: Point estimate of the net rent in CHF.
        lo: Lower bound of the calibrated interval.
        hi: Upper bound of the calibrated interval.
        q25: 25th percentile (lower end of the recommended listing band).
        q75: 75th percentile (upper end of the recommended listing band).
        coverage: Nominal coverage of the interval, e.g. 0.8.
        asking: Asking net rent that was checked.
        verdict: ``"below"``, ``"within"`` or ``"above"`` the calibrated interval.
        percentile: Approximate market percentile of the asking rent (0-1).
        drivers: Largest SHAP drivers as ``(German label, effect in percent)``.
        missing: Location inputs that geo.admin could not provide.
        model_version: Version of the model bundle.
        model_created: Training date of the model bundle (ISO).
    """

    expected: float
    lo: float
    hi: float
    q25: float
    q75: float
    coverage: float
    asking: float
    verdict: str
    percentile: float
    drivers: tuple[tuple[str, float], ...] = ()
    missing: tuple[str, ...] = ()
    model_version: str = ""
    model_created: str = ""

    @property
    def deviation(self) -> float:
        """Asking rent relative to the estimate, e.g. ``-0.08`` = 8 % below."""
        return self.asking / self.expected - 1.0

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable dict (inverse of :meth:`from_dict`)."""
        return {
            "expected": self.expected,
            "lo": self.lo,
            "hi": self.hi,
            "q25": self.q25,
            "q75": self.q75,
            "coverage": self.coverage,
            "asking": self.asking,
            "verdict": self.verdict,
            "percentile": self.percentile,
            "drivers": [list(d) for d in self.drivers],
            "missing": list(self.missing),
            "model_version": self.model_version,
            "model_created": self.model_created,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RentInsight":
        """Rebuild a snapshot written by :meth:`to_dict`."""
        return cls(
            expected=float(data["expected"]),
            lo=float(data["lo"]),
            hi=float(data["hi"]),
            q25=float(data["q25"]),
            q75=float(data["q75"]),
            coverage=float(data["coverage"]),
            asking=float(data["asking"]),
            verdict=str(data["verdict"]),
            percentile=float(data["percentile"]),
            drivers=tuple((str(d[0]), float(d[1])) for d in data.get("drivers") or ()),
            missing=tuple(str(m) for m in data.get("missing") or ()),
            model_version=str(data.get("model_version") or ""),
            model_created=str(data.get("model_created") or ""),
        )


@dataclass(frozen=True)
class ListingDraft:
    """A listing as entered by the landlord, before it is stored.

    Attributes:
        title: Headline of the listing.
        description: Free text (contact data is stripped before publication).
        address: Geocoded address from geo.admin.
        area: Living area in m².
        rooms: Rooms in half steps.
        rent_net: Monthly net rent in CHF (the target of the rent model).
        aux_costs: Monthly auxiliary costs in CHF, if known.
        floor: Floor (0 = ground floor), if known.
        available_from: First day of the lease; ``None`` = by arrangement.
        contact_name: Name of the landlord or manager.
        contact_email: E-mail address (never shown publicly).
        contact_phone: Optional phone number (never shown publicly).
    """

    title: str
    description: str
    address: Address
    area: float
    rooms: float
    rent_net: float
    aux_costs: float | None
    floor: int | None
    available_from: date | None
    contact_name: str
    contact_email: str
    contact_phone: str | None = None

    @classmethod
    def from_form(cls, form: Mapping[str, object], address: Address | None) -> "ListingDraft":
        """Parse raw form values (strings, numbers or ``None``) and validate them.

        Args:
            form: Values keyed by field name (``title``, ``description``, ``area``, ``rooms``,
                ``rent_net``, ``aux_costs``, ``floor``, ``available_from``, ``contact_name``,
                ``contact_email``, ``contact_phone``).
            address: Selected address; ``None`` if the user picked none.

        Returns:
            The validated draft.

        Raises:
            MarketError: If a value is missing or invalid.
        """
        if address is None:
            raise MarketError("Bitte eine Adresse aus den Vorschlägen wählen.")
        floor = _optional_number(form.get("floor"), "Stockwerk")
        draft = cls(
            title=_text(form.get("title")),
            description=str(form.get("description") or "").strip(),
            address=address,
            area=_required_number(form.get("area"), "die Wohnfläche"),
            rooms=_required_number(form.get("rooms"), "die Zimmerzahl"),
            rent_net=_required_number(form.get("rent_net"), "die Nettomiete"),
            aux_costs=_optional_number(form.get("aux_costs"), "Nebenkosten"),
            floor=None if floor is None else round(floor),
            available_from=_day(form.get("available_from"), "Verfügbar ab"),
            contact_name=_text(form.get("contact_name")),
            contact_email=_text(form.get("contact_email")),
            contact_phone=_text(form.get("contact_phone")) or None,
        )
        draft.validate()
        return draft

    def validate(self) -> None:
        """Check the draft against the marketplace rules and the model's training domain.

        Raises:
            MarketError: With a German message naming the first problem.
        """
        if not TITLE_RANGE[0] <= len(self.title) <= TITLE_RANGE[1]:
            raise MarketError(f"Der Titel braucht {TITLE_RANGE[0]} bis {TITLE_RANGE[1]} Zeichen.")
        if len(self.description) > MAX_DESCRIPTION:
            raise MarketError(f"Der Beschrieb darf höchstens {MAX_DESCRIPTION} Zeichen haben.")
        _check_range(self.area, AREA_RANGE, "Die Wohnfläche", "m²")
        _check_range(self.rooms, ROOMS_RANGE, "Die Zimmerzahl")
        if (self.rooms * 2) % 1:
            raise MarketError("Zimmerzahl bitte in halben Schritten angeben (z. B. 3.5).")
        _check_range(self.rent_net, RENT_RANGE, "Die Nettomiete", "CHF")
        if self.aux_costs is not None:
            _check_range(self.aux_costs, AUX_RANGE, "Die Nebenkosten", "CHF")
        if self.floor is not None:
            _check_range(self.floor, FLOOR_RANGE, "Das Stockwerk")
        _check_contact(self.contact_name, self.contact_email, self.contact_phone)


@dataclass(frozen=True)
class Listing:
    """A stored listing with its model snapshot.

    Attributes:
        id: Database id (used in the public URL).
        status: One of :data:`LISTING_STATUSES`.
        created_at: Publication time (UTC).
        title: Headline.
        description: Public description (contact data removed).
        address: Geocoded address.
        place: Municipality, district, canton and language region.
        area: Living area in m².
        rooms: Rooms.
        rent_net: Monthly net rent in CHF.
        aux_costs: Monthly auxiliary costs in CHF, if known.
        floor: Floor, if known.
        available_from: First day of the lease; ``None`` = by arrangement.
        contact_name: Landlord name (shown as provider).
        contact_email: Landlord e-mail (private).
        contact_phone: Landlord phone (private).
        tags: Feature tags (see :data:`rentml.market.insight.TAG_LABELS`).
        is_demo: Synthetic demo listing.
        location: Building and location inputs used for the estimate (for re-estimation).
        insight: Model snapshot; ``None`` if no model was available at publication.
        photo_ids: Ids of the photos in display order.
    """

    id: int
    status: str
    created_at: datetime
    title: str
    description: str
    address: Address
    place: Place
    area: float
    rooms: float
    rent_net: float
    aux_costs: float | None
    floor: int | None
    available_from: date | None
    contact_name: str
    contact_email: str
    contact_phone: str | None
    tags: tuple[str, ...] = ()
    is_demo: bool = False
    location: LocationData = field(default_factory=LocationData)
    insight: RentInsight | None = None
    photo_ids: tuple[int, ...] = ()

    @property
    def rent_gross(self) -> float | None:
        """Net rent plus auxiliary costs (``None`` if the costs are unknown)."""
        return None if self.aux_costs is None else self.rent_net + self.aux_costs

    @property
    def chf_per_m2(self) -> float:
        """Net rent per m² and month."""
        return self.rent_net / self.area


@dataclass(frozen=True)
class ApplicationDraft:
    """An application as entered by a tenant.

    Attributes:
        name: Full name.
        email: E-mail address.
        phone: Optional phone number.
        move_in: Desired first day of the lease.
        household_size: Number of persons moving in.
        message: Free text to the landlord.
    """

    name: str
    email: str
    phone: str | None
    move_in: date | None
    household_size: int | None
    message: str

    @classmethod
    def from_form(cls, form: Mapping[str, object]) -> "ApplicationDraft":
        """Parse and validate raw form values.

        Args:
            form: Values keyed by ``name``, ``email``, ``phone``, ``move_in``, ``household_size``,
                ``message`` and ``consent`` (truthy = privacy notice accepted).

        Returns:
            The validated draft.

        Raises:
            MarketError: If a value is missing or invalid or the consent is not given.
        """
        if not form.get("consent"):
            raise MarketError("Bitte der Weitergabe Ihrer Angaben an die Vermietenden zustimmen.")
        household = _optional_number(form.get("household_size"), "Personen")
        draft = cls(
            name=_text(form.get("name")),
            email=_text(form.get("email")),
            phone=_text(form.get("phone")) or None,
            move_in=_day(form.get("move_in"), "Einzug"),
            household_size=None if household is None else round(household),
            message=str(form.get("message") or "").strip(),
        )
        draft.validate()
        return draft

    def validate(self) -> None:
        """Check the application.

        Raises:
            MarketError: With a German message naming the first problem.
        """
        _check_contact(self.name, self.email, self.phone)
        if self.household_size is not None:
            _check_range(self.household_size, HOUSEHOLD_RANGE, "Die Anzahl Personen")
        if len(self.message) > MAX_MESSAGE:
            raise MarketError(f"Die Nachricht darf höchstens {MAX_MESSAGE} Zeichen haben.")


@dataclass(frozen=True)
class Application:
    """A stored application.

    Attributes:
        id: Database id.
        listing_id: Listing the tenant applied for.
        created_at: Submission time (UTC).
        status: One of :data:`APPLICATION_STATUSES`.
        name: Applicant name.
        email: Applicant e-mail.
        phone: Applicant phone, if given.
        move_in: Desired first day of the lease.
        household_size: Number of persons.
        message: Message to the landlord.
    """

    id: int
    listing_id: int
    created_at: datetime
    status: str
    name: str
    email: str
    phone: str | None
    move_in: date | None
    household_size: int | None
    message: str


@dataclass(frozen=True)
class ListingQuery:
    """Search filters of the browse page (``None`` or empty = no filter).

    Attributes:
        text: Free text matched against title, address, municipality, district and canton.
        canton: Canton abbreviation.
        municipality_id: BFS municipality number (set by code, not by the search form).
        rooms_min: Minimum number of rooms.
        rent_max: Maximum net rent in CHF.
        area_min: Minimum living area in m².
        tags: Tags a listing must carry (all of them).
        fair_only: Only listings whose asking rent is below or within the expected range.
        sort: One of :data:`SORTS`.
    """

    text: str = ""
    canton: str | None = None
    municipality_id: int | None = None
    rooms_min: float | None = None
    rent_max: float | None = None
    area_min: float | None = None
    tags: tuple[str, ...] = ()
    fair_only: bool = False
    sort: str = "newest"

    @classmethod
    def from_form(cls, form: Mapping[str, object]) -> "ListingQuery":
        """Lenient parser for filter widgets and URL query strings (bad values = no filter)."""

        def number(key: str) -> float | None:
            try:
                return _optional_number(form.get(key), key)
            except MarketError:
                return None

        sort = str(form.get("sort") or "newest")
        tags = form.get("tags") or ()
        return cls(
            text=_text(form.get("text"))[:80],
            canton=_text(form.get("canton")).upper()[:2] or None,
            rooms_min=number("rooms_min"),
            rent_max=number("rent_max"),
            area_min=number("area_min"),
            tags=tuple(str(t) for t in tags) if isinstance(tags, list | tuple) else (),
            fair_only=bool(form.get("fair_only")),
            sort=sort if sort in SORTS else "newest",
        )
