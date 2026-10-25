"""Persistence of the marketplace on SQLAlchemy Core.

SQLite in ``data/app/marketplace.db`` by default (not versioned); set ``RENTML_MARKET_DB`` to a
SQLAlchemy URL (e.g. the Neon PostgreSQL) for a deployment whose container file system is
ephemeral. The tables are created on first use, so building the app never touches the disk.

Access without accounts: a listing gets a secret management token, an application a secret status
token; only their SHA-256 hashes are stored. Withdrawn applications are deleted (revDSG).
"""

import hashlib
import os
import secrets
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.pool import StaticPool

from rentml.config import ProjectPaths
from rentml.geoadmin import Address, LocationData
from rentml.market.models import (
    APPLICATION_STATUSES,
    LISTING_STATUSES,
    Application,
    ApplicationDraft,
    Listing,
    ListingDraft,
    ListingQuery,
    MarketError,
    Place,
    RentInsight,
    normalize_search,
)

DB_ENV = "RENTML_MARKET_DB"
DB_FILE = "marketplace.db"
MEMORY_URL = "sqlite://"
HARD_LIMIT = 500
FAIR_VERDICTS: tuple[str, ...] = ("below", "within")

metadata = sa.MetaData()
listings_t = sa.Table(
    "market_listings",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("status", sa.String(16), nullable=False, index=True),
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Column("title", sa.String(160), nullable=False),
    sa.Column("description", sa.Text, nullable=False),
    sa.Column("search_text", sa.Text, nullable=False),
    sa.Column("address_label", sa.String(240), nullable=False),
    sa.Column("east", sa.Float, nullable=False),
    sa.Column("north", sa.Float, nullable=False),
    sa.Column("egid", sa.BigInteger),
    sa.Column("municipality_id", sa.Integer, nullable=False, index=True),
    sa.Column("municipality", sa.String(120), nullable=False),
    sa.Column("district", sa.String(120), nullable=False),
    sa.Column("canton", sa.String(2), nullable=False, index=True),
    sa.Column("lang_region", sa.String(8), nullable=False),
    sa.Column("area", sa.Float, nullable=False),
    sa.Column("rooms", sa.Float, nullable=False),
    sa.Column("floor", sa.Integer),
    sa.Column("rent_net", sa.Float, nullable=False),
    sa.Column("aux_costs", sa.Float),
    sa.Column("available_from", sa.Date),
    sa.Column("contact_name", sa.String(120), nullable=False),
    sa.Column("contact_email", sa.String(254), nullable=False),
    sa.Column("contact_phone", sa.String(40)),
    sa.Column("tags", sa.Text, nullable=False),
    sa.Column("is_demo", sa.Boolean, nullable=False),
    sa.Column("location", sa.JSON, nullable=False),
    sa.Column("insight", sa.JSON(none_as_null=True)),
    sa.Column("verdict", sa.String(8)),
    sa.Column("expected_rent", sa.Float),
    sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
    sqlite_autoincrement=True,  # never reuse the id of a deleted listing (stale links)
)
photos_t = sa.Table(
    "market_photos",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("listing_id", sa.Integer, nullable=False, index=True),
    sa.Column("position", sa.Integer, nullable=False),
    sa.Column("content", sa.LargeBinary, nullable=False),
    sqlite_autoincrement=True,
)
applications_t = sa.Table(
    "market_applications",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("listing_id", sa.Integer, nullable=False, index=True),
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Column("status", sa.String(16), nullable=False),
    sa.Column("name", sa.String(120), nullable=False),
    sa.Column("email", sa.String(254), nullable=False),
    sa.Column("phone", sa.String(40)),
    sa.Column("move_in", sa.Date),
    sa.Column("household_size", sa.Integer),
    sa.Column("message", sa.Text, nullable=False),
    sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
    sqlite_autoincrement=True,
)
_ORDER: dict[str, tuple[sa.ColumnElement, ...]] = {
    "newest": (listings_t.c.created_at.desc(), listings_t.c.id.desc()),
    "rent_asc": (listings_t.c.rent_net.asc(), listings_t.c.id.desc()),
    "rent_desc": (listings_t.c.rent_net.desc(), listings_t.c.id.desc()),
    "deal": (
        (listings_t.c.rent_net / listings_t.c.expected_rent).asc().nulls_last(),
        listings_t.c.id.desc(),
    ),
}


def hash_token(token: str) -> str:
    """SHA-256 hex digest of a secret token (what the database stores)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def resolve_url(url: str | None = None) -> str:
    """Database URL: argument, then ``$RENTML_MARKET_DB``, then SQLite in ``data/app``."""
    if url or (url := os.environ.get(DB_ENV)):
        return url
    return f"sqlite:///{ProjectPaths.discover().project_root / 'data' / 'app' / DB_FILE}"


def _new_token() -> tuple[str, str]:
    token = secrets.token_urlsafe(24)
    return token, hash_token(token)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)  # naive UTC: portable across SQLite/PostgreSQL


def _create_engine(url: str) -> sa.Engine:
    parsed = sa.engine.make_url(url)
    if parsed.get_backend_name() != "sqlite":
        return sa.create_engine(parsed, pool_pre_ping=True)
    if parsed.database in (None, "", ":memory:"):
        # One shared connection: every new connection would open its own empty database.
        return sa.create_engine(
            parsed, poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
    Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    return sa.create_engine(parsed)


def _tags_text(tags: Sequence[str]) -> str:
    return "," + ",".join(sorted(set(tags))) + "," if tags else ""


def _search_text(draft: ListingDraft, place: Place) -> str:
    parts = (draft.title, draft.address.label, place.municipality, place.district, place.canton)
    return normalize_search(" ".join(parts))


def _location_json(location: LocationData) -> dict[str, object]:
    return {**location.to_features(), "missing": list(location.missing)}


def _to_listing(row: sa.Row, photo_ids: tuple[int, ...]) -> Listing:
    m = row._mapping
    location = dict(m["location"] or {})
    missing = [str(name) for name in location.pop("missing", [])]
    return Listing(
        id=m["id"],
        status=m["status"],
        created_at=m["created_at"].replace(tzinfo=UTC),
        title=m["title"],
        description=m["description"],
        address=Address(m["address_label"], m["east"], m["north"], m["egid"]),
        place=Place(
            m["municipality_id"], m["municipality"], m["district"], m["canton"], m["lang_region"]
        ),
        area=m["area"],
        rooms=m["rooms"],
        rent_net=m["rent_net"],
        aux_costs=m["aux_costs"],
        floor=m["floor"],
        available_from=m["available_from"],
        contact_name=m["contact_name"],
        contact_email=m["contact_email"],
        contact_phone=m["contact_phone"],
        tags=tuple(tag for tag in m["tags"].split(",") if tag),
        is_demo=bool(m["is_demo"]),
        location=LocationData(**location, missing=missing),
        insight=RentInsight.from_dict(m["insight"]) if m["insight"] else None,
        photo_ids=photo_ids,
    )


def _to_application(row: sa.Row) -> Application:
    m = row._mapping
    return Application(
        id=m["id"],
        listing_id=m["listing_id"],
        created_at=m["created_at"].replace(tzinfo=UTC),
        status=m["status"],
        name=m["name"],
        email=m["email"],
        phone=m["phone"],
        move_in=m["move_in"],
        household_size=m["household_size"],
        message=m["message"],
    )


def _insight_columns(insight: RentInsight | None) -> dict[str, object]:
    if insight is None:
        return {"insight": None, "verdict": None, "expected_rent": None}
    return {
        "insight": insight.to_dict(),
        "verdict": insight.verdict,
        "expected_rent": insight.expected,
    }


class MarketStore:
    """Listings, photos and applications in one database (engine created on first use)."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._engine: sa.Engine | None = None
        self._lock = threading.Lock()

    @classmethod
    def in_memory(cls) -> "MarketStore":
        """Throw-away store for tests and for a deployment without a writable disk."""
        return cls(MEMORY_URL)

    @property
    def engine(self) -> sa.Engine:
        """The engine; the first access creates the database file and the tables."""
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    engine = _create_engine(self.url)
                    metadata.create_all(engine)
                    self._engine = engine
        return self._engine

    def create_listing(
        self,
        draft: ListingDraft,
        place: Place,
        *,
        location: LocationData,
        insight: RentInsight | None,
        tags: Sequence[str] = (),
        photos: Sequence[bytes] = (),
        is_demo: bool = False,
    ) -> tuple[Listing, str]:
        """Store a validated listing and return it with its secret management token.

        Args:
            draft: Validated draft; its description is stored as given (strip contact data first).
            place: Municipality of the address.
            location: Building and location inputs used for the estimate.
            insight: Model snapshot (``None`` without a model).
            tags: Feature tags.
            photos: JPEG bytes in display order (see :mod:`rentml.market.photos`).
            is_demo: Mark the listing as synthetic.

        Returns:
            The stored listing and the token (shown to the landlord once, never stored).
        """
        token, token_hash = _new_token()
        values = {
            "status": "active",
            "created_at": _now(),
            "title": draft.title,
            "description": draft.description,
            "search_text": _search_text(draft, place),
            "address_label": draft.address.label,
            "east": draft.address.east,
            "north": draft.address.north,
            "egid": draft.address.egid,
            "municipality_id": place.municipality_id,
            "municipality": place.municipality,
            "district": place.district,
            "canton": place.canton,
            "lang_region": place.lang_region,
            "area": draft.area,
            "rooms": draft.rooms,
            "floor": draft.floor,
            "rent_net": draft.rent_net,
            "aux_costs": draft.aux_costs,
            "available_from": draft.available_from,
            "contact_name": draft.contact_name,
            "contact_email": draft.contact_email,
            "contact_phone": draft.contact_phone,
            "tags": _tags_text(tags),
            "is_demo": is_demo,
            "location": _location_json(location),
            "token_hash": token_hash,
            **_insight_columns(insight),
        }
        with self.engine.begin() as conn:
            result = conn.execute(sa.insert(listings_t).values(**values))
            listing_id = result.inserted_primary_key[0]
            for position, content in enumerate(photos):
                conn.execute(
                    sa.insert(photos_t).values(
                        listing_id=listing_id, position=position, content=content
                    )
                )
            stored = self._load(conn, sa.select(listings_t).where(listings_t.c.id == listing_id))
        return stored[0], token

    def _load(self, conn: sa.Connection, stmt: sa.Select) -> list[Listing]:
        rows = conn.execute(stmt).all()
        ids = [row._mapping["id"] for row in rows]
        photos: dict[int, list[int]] = {listing_id: [] for listing_id in ids}
        if ids:
            photo_rows = conn.execute(
                sa.select(photos_t.c.listing_id, photos_t.c.id)
                .where(photos_t.c.listing_id.in_(ids))
                .order_by(photos_t.c.position, photos_t.c.id)
            )
            for listing_id, photo_id in photo_rows:
                photos[listing_id].append(photo_id)
        return [_to_listing(row, tuple(photos[row._mapping["id"]])) for row in rows]

    def listing(self, listing_id: int) -> Listing | None:
        """One listing by id, whatever its status (``None`` if unknown)."""
        with self.engine.connect() as conn:
            found = self._load(conn, sa.select(listings_t).where(listings_t.c.id == listing_id))
        return found[0] if found else None

    def listing_by_token(self, token: str) -> Listing | None:
        """The listing a management token belongs to (``None`` for a wrong token)."""
        stmt = sa.select(listings_t).where(listings_t.c.token_hash == hash_token(str(token)))
        with self.engine.connect() as conn:
            found = self._load(conn, stmt)
        return found[0] if found else None

    def search(self, query: ListingQuery | None = None) -> list[Listing]:
        """Active listings matching the filters, at most ``HARD_LIMIT``.

        The sorts ``qoli`` and ``value`` need data outside the database; for them the result is
        ordered by date and the caller re-sorts.
        """
        query = query or ListingQuery()
        t = listings_t.c
        where = [t.status == "active"]
        where += [
            t.search_text.contains(tok, autoescape=True)
            for tok in normalize_search(query.text).split()
        ]
        where += [t.tags.contains(f",{tag},", autoescape=True) for tag in query.tags]
        if query.canton:
            where.append(t.canton == query.canton)
        if query.municipality_id is not None:
            where.append(t.municipality_id == query.municipality_id)
        if query.rooms_min is not None:
            where.append(t.rooms >= query.rooms_min)
        if query.rent_max is not None:
            where.append(t.rent_net <= query.rent_max)
        if query.area_min is not None:
            where.append(t.area >= query.area_min)
        if query.fair_only:
            where.append(t.verdict.in_(FAIR_VERDICTS))
        order = _ORDER.get(query.sort, _ORDER["newest"])
        stmt = sa.select(listings_t).where(*where).order_by(*order).limit(HARD_LIMIT)
        with self.engine.connect() as conn:
            return self._load(conn, stmt)

    def stats(self) -> dict[str, int]:
        """Counts for the start page: active listings, fair ones and covered municipalities."""
        t = listings_t.c
        fair = sa.func.sum(sa.case((t.verdict.in_(FAIR_VERDICTS), 1), else_=0))
        stmt = sa.select(sa.func.count(), fair, sa.func.count(t.municipality_id.distinct())).where(
            t.status == "active"
        )
        with self.engine.connect() as conn:
            active, n_fair, municipalities = conn.execute(stmt).one()
        return {"active": active, "fair": int(n_fair or 0), "municipalities": municipalities}

    def cantons(self) -> list[str]:
        """Cantons with at least one active listing, sorted."""
        t = listings_t.c
        stmt = sa.select(t.canton).where(t.status == "active").distinct().order_by(t.canton)
        with self.engine.connect() as conn:
            return [row[0] for row in conn.execute(stmt)]

    def set_status(self, listing_id: int, status: str) -> None:
        """Change the status of a listing.

        Raises:
            ValueError: If ``status`` is not one of ``LISTING_STATUSES``.
        """
        if status not in LISTING_STATUSES:
            raise ValueError(f"Unknown listing status {status!r}")
        stmt = sa.update(listings_t).where(listings_t.c.id == listing_id).values(status=status)
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def update_rent(
        self, listing_id: int, rent_net: float, aux_costs: float | None, insight: RentInsight | None
    ) -> None:
        """Store a new asking rent together with its re-computed model snapshot."""
        values = {"rent_net": rent_net, "aux_costs": aux_costs, **_insight_columns(insight)}
        stmt = sa.update(listings_t).where(listings_t.c.id == listing_id).values(**values)
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def delete_listing(self, listing_id: int) -> None:
        """Remove a listing with its photos and all applications (personal data)."""
        with self.engine.begin() as conn:
            conn.execute(sa.delete(photos_t).where(photos_t.c.listing_id == listing_id))
            conn.execute(sa.delete(applications_t).where(applications_t.c.listing_id == listing_id))
            conn.execute(sa.delete(listings_t).where(listings_t.c.id == listing_id))

    def photo(self, photo_id: int) -> bytes | None:
        """JPEG bytes of a photo of an *active* listing (``None`` otherwise)."""
        stmt = (
            sa.select(photos_t.c.content)
            .join(listings_t, listings_t.c.id == photos_t.c.listing_id)
            .where(photos_t.c.id == photo_id, listings_t.c.status == "active")
        )
        with self.engine.connect() as conn:
            content = conn.execute(stmt).scalar_one_or_none()
        return None if content is None else bytes(content)

    def add_application(self, listing_id: int, draft: ApplicationDraft) -> tuple[Application, str]:
        """Store an application for an active listing and return it with its status token.

        Raises:
            MarketError: If the listing does not exist or is no longer active.
        """
        token, token_hash = _new_token()
        with self.engine.begin() as conn:
            status = conn.execute(
                sa.select(listings_t.c.status).where(listings_t.c.id == listing_id)
            ).scalar_one_or_none()
            if status != "active":
                raise MarketError("Dieses Inserat nimmt keine Bewerbungen mehr entgegen.")
            values = {
                "listing_id": listing_id,
                "created_at": _now(),
                "status": "new",
                "name": draft.name,
                "email": draft.email,
                "phone": draft.phone,
                "move_in": draft.move_in,
                "household_size": draft.household_size,
                "message": draft.message,
                "token_hash": token_hash,
            }
            key = conn.execute(sa.insert(applications_t).values(**values)).inserted_primary_key[0]
            row = conn.execute(sa.select(applications_t).where(applications_t.c.id == key)).one()
        return _to_application(row), token

    def applications(self, listing_id: int) -> list[Application]:
        """Applications of a listing in order of arrival (never ranked)."""
        t = applications_t.c
        stmt = (
            sa.select(applications_t).where(t.listing_id == listing_id).order_by(t.created_at, t.id)
        )
        with self.engine.connect() as conn:
            return [_to_application(row) for row in conn.execute(stmt)]

    def application_by_token(self, token: str) -> Application | None:
        """The application a status token belongs to (``None`` for a wrong token)."""
        stmt = sa.select(applications_t).where(
            applications_t.c.token_hash == hash_token(str(token))
        )
        with self.engine.connect() as conn:
            row = conn.execute(stmt).one_or_none()
        return None if row is None else _to_application(row)

    def set_application_status(self, listing_id: int, application_id: int, status: str) -> bool:
        """Set the status of an application of *this* listing; ``False`` if it is not one.

        Raises:
            ValueError: If ``status`` is not one of ``APPLICATION_STATUSES``.
        """
        if status not in APPLICATION_STATUSES:
            raise ValueError(f"Unknown application status {status!r}")
        t = applications_t.c
        stmt = (
            sa.update(applications_t)
            .where(t.id == application_id, t.listing_id == listing_id)
            .values(status=status)
        )
        with self.engine.begin() as conn:
            return conn.execute(stmt).rowcount == 1

    def withdraw_application(self, token: str) -> bool:
        """Delete the application of a status token; ``False`` for a wrong token."""
        stmt = sa.delete(applications_t).where(
            applications_t.c.token_hash == hash_token(str(token))
        )
        with self.engine.begin() as conn:
            return conn.execute(stmt).rowcount == 1
