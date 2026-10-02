"""Duplicate audit and object identity for rental listings.

Repeated listings of the same apartment (re-posts, cross-portal copies, repeated snapshots)
share one ``object_id`` so that no apartment ends up on both sides of a split or CV fold.
Linkage rules (proposal), applied in the order of :data:`RULES`:

1. ``listing_id``: same index label (e.g. concatenated snapshots of one listing).
2. ``slug``: same non-empty slug / URL (qualified by ``source`` when present).
3. ``description``: same non-empty normalised description text (hash).
4. Building rules within the finest building key that *both* listings carry: ``egid`` (GWR
   building id); else ``coords`` (LV95 points <= ``coord_round_m`` m apart; two different
   egids veto the match); else ``address`` (postcode + normalised street; only used when not
   both listings have coordinates). Attributes match if rooms are equal *or unknown* and
   ``|Δarea| <= area_tol`` and ``|Δprice| / max(price) <= price_rel_tol``, **or** if area and
   price are identical (``|Δarea| <= 0.5`` m², ``|Δprice| <= 1`` CHF) whatever the rooms
   (DSPRO1 stored rooms as integers, so the same flat can appear with 5 and 6 rooms).
5. ``coords_exact``: points <= ``exact_radius_m`` apart (geocoding jitter) with identical area
   and price and equal or unknown rooms; the egid veto applies.

Candidate pairs are generated only inside blocks (sorted area windows per building block,
KD-tree radius queries for coordinates): O(n log n + #candidates) instead of O(n^2). Matches are
merged as connected components, which makes the relation **transitive**: a chain A~B, B~C puts
A and C into one object even if A and C differ by more than the tolerance. This is intended
(conservative grouping for leakage-free splits) but means tolerances must stay tight.
"""

import hashlib
import logging
import re
import unicodedata
from collections.abc import Sequence

import numpy as np
import pandas as pd

from rentml._dedup_rules import EMPTY_PAIRS, Pairs, building_rule_pairs, components, star_edges

logger = logging.getLogger(__name__)

RULES: tuple[str, ...] = (
    "listing_id",
    "slug",
    "description",
    "egid",
    "coords",
    "address",
    "coords_exact",
)
DEFAULT_EXACT_SUBSET: tuple[str, ...] = ("east", "north", "area", "rooms", "price")

# Applied with chained str.replace: much faster than str.translate with multi-char targets.
_TRANSLIT: dict[str, str] = {"ä": "ae", "ö": "oe", "ü": "ue", "æ": "ae", "œ": "oe"}
_HTML_TAG = re.compile(r"<[^>]+>")
_COMBINING_MARKS = re.compile("[\u0300-\u036f]")  # accents after NFKD decomposition
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_POSTCODE_ONLY = re.compile(r"^\d{4}$")
_POSTCODE_LOCALITY = re.compile(r"^(\d{4})\s+\S")
_TRAILING_HOUSE_NUMBER = re.compile(r"(\s\d+[a-z]?)+(\s[a-z])?$")
_STREET_ABBREVIATIONS: dict[str, str] = {
    r"str\b": "strasse",
    r"\brte\b": "route",
    r"\bave?\b": "avenue",
    r"\bch\b": "chemin",
    r"\bbd\b": "boulevard",
    r"\bpl\b": "place",
}


def _is_missing(value: object) -> bool:
    return value is None or (pd.api.types.is_scalar(value) and bool(pd.isna(value)))


def normalize_text(s: str | None) -> str:
    """Normalise free text for exact-match comparisons.

    Removes HTML tags, applies Unicode NFKC + casefolding, transliterates German umlauts
    (``ü`` -> ``ue``, so "Rütistrasse" equals "Ruetistrasse"), strips remaining diacritics and
    collapses every run of non-alphanumeric characters into one space.

    Args:
        s: Raw text; ``None``/NaN/``pd.NA`` are treated as empty.

    Returns:
        The normalised text (``""`` for missing input).
    """
    if _is_missing(s):
        return ""
    text = _HTML_TAG.sub(" ", str(s))
    text = unicodedata.normalize("NFKC", text).casefold()  # casefold also maps ß -> ss
    for umlaut, replacement in _TRANSLIT.items():
        text = text.replace(umlaut, replacement)
    text = _COMBINING_MARKS.sub("", unicodedata.normalize("NFKD", text))
    return _NON_ALNUM.sub(" ", text).strip()


def description_hash(s: pd.Series, *, min_chars: int = 1) -> pd.Series:
    """Hash normalised descriptions so that formatting-only variants collide.

    Args:
        s: Raw description texts.
        min_chars: Normalised texts shorter than this are treated as empty.

    Returns:
        SHA-256 hex digests (``string`` dtype, same index); ``<NA>`` for empty/short texts.
    """
    min_len = max(min_chars, 1)
    normalized = s.map(normalize_text, na_action=None)
    hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest() if len(text) >= min_len else pd.NA
        for text in normalized
    ]
    return pd.Series(hashes, index=s.index, dtype="string", name=s.name)


def exact_duplicate_mask(df: pd.DataFrame, subset: list[str]) -> pd.Series:
    """Flag exact duplicates on a column subset.

    Missing values compare equal (pandas ``duplicated`` semantics).

    Args:
        df: Listings.
        subset: Columns that define an exact duplicate.

    Returns:
        Boolean Series aligned with ``df.index``; True for the 2nd+ occurrence of a key.

    Raises:
        ValueError: If ``subset`` is empty.
        KeyError: If a column of ``subset`` is missing.
    """
    if not subset:
        raise ValueError("subset must contain at least one column")
    missing = [col for col in subset if col not in df.columns]
    if missing:
        raise KeyError(f"Columns not in frame: {missing}")
    return df.duplicated(subset=list(subset), keep="first").rename("exact_duplicate")


def _key_codes(values: pd.Series) -> np.ndarray:
    """Integer codes per distinct non-empty key; -1 for missing/empty keys."""
    clean = values.mask(values.astype("string").str.strip().fillna("") == "")
    return pd.factorize(clean, use_na_sentinel=True)[0].astype(np.int64)


def _numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _split_address(address: object) -> tuple[str, str]:
    """Postcode and street from DSPRO1 address strings (both formats); empty if unknown.

    Handles "8044, Gockhausen-Zürich, Rütistrasse, 1" and "Schellerstrasse 17, 8630 Wetzikon".
    """
    if not isinstance(address, str):
        return "", ""
    parts = [part.strip() for part in address.split(",") if part.strip()]
    if len(parts) >= 3 and _POSTCODE_ONLY.match(parts[0]):
        return parts[0], parts[2]
    for part in parts[1:]:
        match = _POSTCODE_LOCALITY.match(part)
        if match:
            return match.group(1), parts[0]
    return "", ""


def _street_key(street: object) -> str:
    text = _TRAILING_HOUSE_NUMBER.sub(
        "", normalize_text(street if isinstance(street, str) else None)
    )
    for pattern, replacement in _STREET_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
    return text.replace(" ", "")


def _address_keys(df: pd.DataFrame, address_cols: tuple[str, str]) -> pd.Series:
    """``postcode|street`` keys from parsed columns or, as fallback, the raw ``address``."""
    postcode_col, street_col = address_cols
    if df.empty:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    if postcode_col in df.columns and street_col in df.columns:
        postcodes = df[postcode_col].map(normalize_text)
        streets = df[street_col]
    elif "address" in df.columns:
        split = df["address"].map(_split_address)
        postcodes = split.str[0]
        streets = split.str[1]
    else:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    street_keys = streets.map(_street_key)
    keys = postcodes.str.replace(" ", "", regex=False) + "|" + street_keys
    valid = (postcodes.str.len() > 0) & (street_keys.str.len() > 0)
    return keys.where(valid).astype("string")


def _slug_codes(df: pd.DataFrame, col: str) -> np.ndarray:
    slug = df[col].astype("string").str.strip().str.casefold().replace("", pd.NA)
    if "source" in df.columns:
        slug = df["source"].astype("string").str.strip().str.casefold().fillna("") + "|" + slug
    return _key_codes(slug)


def _description_codes(df: pd.DataFrame, min_chars: int, max_group: int | None) -> np.ndarray:
    hashes = description_hash(df["description"], min_chars=min_chars)
    if max_group is not None:
        sizes = hashes.map(hashes.value_counts())
        boilerplate = (sizes > max_group).fillna(False).astype(bool)
        if boilerplate.any():
            logger.info(
                "Ignoring %d boilerplate description groups (> %d listings, %d rows)",
                hashes[boilerplate].nunique(),
                max_group,
                int(boilerplate.sum()),
            )
        hashes = hashes.mask(boilerplate)
    return _key_codes(hashes)


def _rule_pairs(
    df: pd.DataFrame,
    tols: tuple[float, float, float, float | None],
    desc: tuple[int, int | None],
    id_cols: Sequence[str],
    address_cols: tuple[str, str],
) -> dict[str, Pairs]:
    """Matched pairs ``(left, right)`` (positions) per rule of :data:`RULES`."""
    pairs: dict[str, list[Pairs]] = {rule: [] for rule in RULES}
    if not df.index.is_unique:
        label_codes = pd.factorize(df.index, use_na_sentinel=True)[0].astype(np.int64)
        pairs["listing_id"].append(star_edges(label_codes))
    for col in id_cols:
        if col in df.columns:
            pairs["slug"].append(star_edges(_slug_codes(df, col)))
    if "description" in df.columns:
        pairs["description"].append(star_edges(_description_codes(df, *desc)))
    egid = _key_codes(df["egid"]) if "egid" in df.columns else np.full(len(df), -1, np.int64)
    blocks = {"egid": egid, "address": _key_codes(_address_keys(df, address_cols))}
    coords = (_numeric(df, "east"), _numeric(df, "north"))
    attrs = {col: _numeric(df, col) for col in ("rooms", "area", "price")}
    for rule, rule_pairs in building_rule_pairs(blocks, coords, attrs, tols).items():
        pairs[rule].append(rule_pairs)
    return {
        rule: (
            np.concatenate([p[0] for p in found] or [EMPTY_PAIRS[0]]),
            np.concatenate([p[1] for p in found] or [EMPTY_PAIRS[1]]),
        )
        for rule, found in pairs.items()
    }


def assign_object_ids(
    df: pd.DataFrame,
    *,
    area_tol: float = 3.0,
    price_rel_tol: float = 0.10,
    coord_round_m: float = 5.0,
    exact_radius_m: float | None = 30.0,
    desc_min_chars: int = 30,
    desc_max_group: int | None = 10,
    id_cols: Sequence[str] = ("slug", "url"),
    address_cols: tuple[str, str] = ("postcode", "street"),
) -> pd.Series:
    """Assign a stable object id to every listing (connected components over all rules).

    Rules are applied in the order of :data:`RULES`; see the module docstring. Rules whose
    columns are absent are skipped. Rows with missing area or price never match via the
    building rules; missing rooms act as a wildcard. The ids ``obj_000001, ...`` are ordered by
    the smallest index label (``listing_id``) of each object, so they do not depend on the
    row order. Duplicate index labels always end up in the same object.

    Args:
        df: Listings indexed by ``listing_id`` (canonical column names).
        area_tol: Maximum absolute living-area difference in m² (inclusive).
        price_rel_tol: Maximum price difference relative to the larger price (inclusive).
        coord_round_m: Coordinate matching radius in metres (inclusive, KD-tree query).
        exact_radius_m: Radius in metres of the ``coords_exact`` rule (identical area/price);
            ``None`` disables the rule.
        desc_min_chars: Normalised descriptions shorter than this never link (placeholders).
        desc_max_group: Description hashes shared by more listings than this are treated as
            property-manager boilerplate and do not link; ``None`` disables the guard.
        id_cols: Identifier columns (slug/url) that link listings when equal.
        address_cols: Parsed ``(postcode, street)`` columns; falls back to parsing ``address``.

    Returns:
        ``object_id`` strings aligned with ``df.index``. ``attrs["rule_rows"]`` and
        ``attrs["rule_merges"]`` hold per-rule counts (rows with a partner; effective merges,
        order-dependent) for :func:`dedup_report`, valid for exactly these rows (checked
        via ``attrs["index_fingerprint"]``).

    Raises:
        ValueError: If a tolerance or radius is negative.
    """
    tols = (area_tol, price_rel_tol, coord_round_m, exact_radius_m)
    if min(t for t in tols if t is not None) < 0:
        raise ValueError("Tolerances must be non-negative")
    n = len(df)
    pairs = _rule_pairs(df, tols, (desc_min_chars, desc_max_group), id_cols, address_cols)
    rule_rows: dict[str, int] = {}
    rule_merges: dict[str, int] = {}
    edges: list[Pairs] = []
    n_components, labels = n, np.arange(n, dtype=np.int64)
    for rule in RULES:
        left, right = pairs[rule]
        rule_rows[rule] = int(np.unique(np.r_[left, right]).size)
        edges.append((left, right))
        count = n_components
        if left.size:
            all_left, all_right = (np.concatenate(side) for side in zip(*edges, strict=True))
            count, labels = components(n, all_left, all_right)
        rule_merges[rule], n_components = n_components - count, count

    ids = pd.Series(_ids_from_roots(labels, df.index), index=df.index, name="object_id")
    ids.attrs.update(
        rule_rows=rule_rows,
        rule_merges=rule_merges,
        index_fingerprint=_index_fingerprint(df.index),
    )
    logger.info("%d rows -> %d objects; merges per rule %s", n, n_components, rule_merges)
    return ids


def _index_fingerprint(index: pd.Index) -> tuple[int, int]:
    """Order-independent fingerprint (size, wrapped hash sum) of the index labels."""
    hashed = pd.util.hash_pandas_object(index).to_numpy(dtype=np.uint64)
    return len(index), int(hashed.sum(dtype=np.uint64))


def _ids_from_roots(roots: np.ndarray, index: pd.Index) -> list[str]:
    """Deterministic ``obj_%06d`` ids ordered by the smallest index label per component."""
    if roots.size == 0:
        return []
    rank = np.empty(roots.size, dtype=np.int64)
    rank[index.argsort(kind="stable")] = np.arange(roots.size)
    component_min = pd.Series(rank).groupby(roots).transform("min").to_numpy()
    _, dense = np.unique(component_min, return_inverse=True)
    return [f"obj_{k:06d}" for k in (dense + 1).tolist()]


def _check_ids(df: pd.DataFrame, object_ids: pd.Series) -> None:
    if len(object_ids) != len(df) or not object_ids.index.equals(df.index):
        raise ValueError("object_ids must be aligned with df.index")
    if object_ids.isna().any():
        raise ValueError("object_ids contains missing values")


def _split_exact_groups(df: pd.DataFrame, object_ids: pd.Series, subset: list[str]) -> int:
    """Exact-duplicate groups (missing values equal) that span more than one object id."""
    if df.empty:
        return 0
    keys = df.groupby(subset, dropna=False, sort=False).ngroup().to_numpy()
    ids_per_group = pd.Series(object_ids.to_numpy()).groupby(keys).nunique()
    return int((ids_per_group > 1).sum())


def _rule_metrics(object_ids: pd.Series) -> dict[str, float]:
    """``rows_<rule>`` / ``merges_<rule>`` from the attrs, if they belong to these rows."""
    attrs = object_ids.attrs
    if "rule_rows" not in attrs or "rule_merges" not in attrs:
        logger.info("dedup_report: object_ids carry no per-rule counts (no attrs)")
        return {}
    if attrs.get("index_fingerprint") != _index_fingerprint(object_ids.index):
        logger.warning(
            "dedup_report: object_ids were subset or re-indexed after assign_object_ids; "
            "per-rule counts (rows_*/merges_*) are omitted"
        )
        return {}
    metrics: dict[str, float] = {}
    for key, prefix in (("rule_rows", "rows"), ("rule_merges", "merges")):
        for rule, count in attrs[key].items():
            metrics[f"{prefix}_{rule}"] = count
    return metrics


def dedup_report(
    df: pd.DataFrame, object_ids: pd.Series, *, exact_subsets: dict[str, list[str]] | None = None
) -> pd.DataFrame:
    """Summarise the duplicate audit (numbers before and after deduplication).

    Args:
        df: Listings (same index as ``object_ids``).
        object_ids: Output of :func:`assign_object_ids`.
        exact_subsets: Named column subsets; for each, ``exact_dup_rows_<name>`` (2nd+
            occurrences, missing values compare equal) and ``exact_dup_groups_split_<name>``
            (exact-duplicate groups spread over more than one object; should be 0) are added.
            ``None`` uses ``{"coords_attrs": DEFAULT_EXACT_SUBSET}`` when those columns exist.

    Returns:
        DataFrame indexed by ``metric`` with a single float column ``value``. Contains rows,
        unique objects, duplicate rows/share, objects with duplicates, group-size statistics,
        the median relative price range within multi-listing objects, the exact-duplicate
        metrics and, if ``object_ids`` still covers exactly the rows it was computed on,
        ``rows_<rule>`` / ``merges_<rule>`` (otherwise omitted with a warning).

    Raises:
        ValueError: If ``object_ids`` is not aligned with ``df``.
        KeyError: If a column of ``exact_subsets`` is missing.
    """
    _check_ids(df, object_ids)
    sizes = object_ids.value_counts()
    n_rows, n_objects = len(df), int(sizes.size)
    metrics: dict[str, float] = {
        "rows": n_rows,
        "unique_objects": n_objects,
        "duplicate_rows": n_rows - n_objects,
        "duplicate_share_pct": 100.0 * (n_rows - n_objects) / n_rows if n_rows else np.nan,
        "objects_with_duplicates": int((sizes > 1).sum()),
        "objects_size_2": int((sizes == 2).sum()),
        "objects_size_3": int((sizes == 3).sum()),
        "objects_size_4plus": int((sizes >= 4).sum()),
        "max_group_size": int(sizes.max()) if n_objects else 0,
        "mean_group_size": float(sizes.mean()) if n_objects else np.nan,
    }
    if "price" in df.columns and (sizes > 1).any():
        price = pd.to_numeric(df["price"], errors="coerce")
        multi = object_ids.map(sizes) > 1
        grouped = price[multi].groupby(object_ids[multi])
        metrics["median_rel_price_range_dup_objects"] = float(
            ((grouped.max() - grouped.min()) / grouped.max()).median()
        )
    if exact_subsets is None:
        has_all = set(DEFAULT_EXACT_SUBSET) <= set(df.columns)
        exact_subsets = {"coords_attrs": list(DEFAULT_EXACT_SUBSET)} if has_all else {}
    for name, subset in exact_subsets.items():
        metrics[f"exact_dup_rows_{name}"] = int(exact_duplicate_mask(df, subset).sum())
        metrics[f"exact_dup_groups_split_{name}"] = _split_exact_groups(df, object_ids, subset)
    metrics.update(_rule_metrics(object_ids))
    report = pd.DataFrame({"value": pd.Series(metrics, dtype=float)})
    report.index.name = "metric"
    return report


def collapse_objects(
    df: pd.DataFrame,
    object_ids: pd.Series,
    *,
    how: str = "first",
    median_cols: Sequence[str] = ("price",),
) -> pd.DataFrame:
    """Reduce the listings to one row per object.

    The representative is the listing with the smallest index label (``listing_id``), i.e.
    the one that also defines the id order. With ``how="first"`` (default) its row is kept
    unchanged, so every column stays a real, internally consistent observation (text,
    coordinates, ids, flags). With ``how="median"`` only ``median_cols`` are replaced by the
    NaN-skipping median over the object's listings (e.g. a consensus rent); ids and codes are
    never averaged, all other columns keep the representative's values and dtypes, and
    ``log_price`` is recomputed as ``log(price)`` when ``price`` is a median column.

    Args:
        df: Listings indexed by ``listing_id``.
        object_ids: Output of :func:`assign_object_ids` (aligned with ``df``).
        how: ``"first"`` or ``"median"``.
        median_cols: Numeric (non-boolean) columns aggregated by the median when
            ``how="median"``; they become float (``Float64`` for nullable input).

    Returns:
        One row per object, index = representative ``listing_id``, sorted by ``object_id``,
        with added columns ``object_id`` and ``n_listings``.

    Raises:
        ValueError: If ``how`` is unknown, the ids are not aligned with ``df`` or a median
            column is not numeric.
        KeyError: If a median column is missing (``how="median"`` only).
    """
    if how not in {"first", "median"}:
        raise ValueError(f"how must be 'first' or 'median', got {how!r}")
    _check_ids(df, object_ids)
    order = df.index.argsort(kind="stable")
    sorted_ids = object_ids.iloc[order]
    representative = order[~sorted_ids.duplicated(keep="first").to_numpy()]
    result = df.iloc[representative].copy()
    result["object_id"] = object_ids.iloc[representative].to_numpy()
    result["n_listings"] = result["object_id"].map(object_ids.value_counts()).astype(int)
    if how == "median":
        _apply_medians(df, object_ids, result, list(median_cols))
    result = result.sort_values("object_id", kind="stable")
    logger.info("collapse_objects: %d listings -> %d objects (%s)", len(df), len(result), how)
    return result


def _apply_medians(
    df: pd.DataFrame, object_ids: pd.Series, result: pd.DataFrame, cols: list[str]
) -> None:
    """Replace ``cols`` of ``result`` (in place) by per-object medians, column by column."""
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"median_cols not in frame: {missing}")
    bad = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == bool]
    if bad:
        raise ValueError(f"median_cols must be numeric (non-boolean): {bad}")
    medians = df[cols].groupby(object_ids.to_numpy()).median()
    for col in cols:
        result[col] = medians[col].reindex(result["object_id"]).set_axis(result.index)
    if "price" in cols and "log_price" in result.columns:
        price = result["price"].astype("float64")
        result["log_price"] = np.log(price.where(price > 0))
