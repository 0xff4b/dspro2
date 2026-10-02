"""Group-aware data splits for the DSPRO2 validation protocol.

The protocol (project proposal, "Validation protocol") splits the deduplicated data once into
train (60 %), calibration (20 %) and a fixed hold-out test set (20 %), grouped by ``object_id``
and stratified by canton x price band. Model selection uses grouped 5-fold CV on the training
set; robustness checks use a spatially grouped CV (by municipality) and a temporal split at the
data-freeze date.

Conventions: split helpers return index *labels* (``pd.Index``), CV fold helpers return
*positional* numpy arrays (sklearn style). Rows whose group key is missing are treated as
singleton groups. Assignments depend only on the data, never on the row order: group codes are
numbered in sorted label order (missing keys by index label), so re-sorting, merging or
re-loading the same rows reproduces the same pre-registered test set for a given seed.

Implementation note: ``StratifiedGroupKFold(shuffle=True)`` in scikit-learn 1.7 shuffles the
per-group class-count rows but then assigns the *unshuffled* group ids, so the stratification
decisions refer to the wrong groups. We therefore randomise by permuting the group codes with a
seeded RNG and call the splitter with ``shuffle=False``, which keeps the permuted order for
groups with the same class-count spread (stable sort).
"""

import logging
from collections.abc import Sequence
from datetime import tzinfo

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

from rentml.config import RANDOM_STATE

logger = logging.getLogger(__name__)

SPLIT_NAMES: tuple[str, str, str] = ("train", "calib", "test")
_N_BASE_FOLDS = 5
_TOL = 1e-9

Folds = list[tuple[np.ndarray, np.ndarray]]


def make_strata(
    df: pd.DataFrame,
    *,
    canton_col: str = "canton",
    price_col: str = "price",
    n_price_bins: int = 4,
    min_stratum: int = 10,
) -> pd.Series:
    """Build canton x price-band strata for stratified splitting.

    Price bands are quantile bins of ``price_col`` (labels ``P1`` .. ``Pk``, ``PNA`` for missing
    prices). Strata with fewer than ``min_stratum`` rows are merged into a canton-agnostic
    price-band stratum (``ALL|P2``). Rows whose merged stratum is still too small join the
    largest remaining stratum of the same price band (else the largest overall; ``ALL|ALL`` if
    no stratum is large enough), so every stratum has at least ``min_stratum`` rows whenever
    ``len(df) >= min_stratum``.

    Args:
        df: Modelling frame.
        canton_col: Canton column (missing values become ``NA``).
        price_col: Price column in CHF.
        n_price_bins: Number of quantile price bands.
        min_stratum: Minimum stratum size before merging.

    Returns:
        String series ``stratum`` aligned with ``df.index`` (e.g. ``"ZH|P3"``).

    Raises:
        KeyError: If a column is missing.
        ValueError: If ``n_price_bins`` or ``min_stratum`` is smaller than 1.
    """
    missing = [c for c in (canton_col, price_col) if c not in df.columns]
    if missing:
        raise KeyError(f"make_strata: missing columns {missing}")
    if n_price_bins < 1 or min_stratum < 1:
        raise ValueError("n_price_bins and min_stratum must be >= 1")
    price = pd.to_numeric(df[price_col], errors="coerce")
    bins = pd.qcut(price, q=n_price_bins, labels=False, duplicates="drop")
    band = pd.Series("PNA", index=df.index, dtype=object)
    band[bins.notna()] = "P" + (bins[bins.notna()].astype(int) + 1).astype(str)
    canton = df[canton_col].astype("string").fillna("NA").astype(object)
    strata = canton + "|" + band
    rare = strata.map(strata.value_counts()) < min_stratum
    strata[rare] = "ALL|" + band[rare]
    still_rare = strata.map(strata.value_counts()) < min_stratum
    if still_rare.any():
        strata[still_rare] = _absorb_rare(strata[~still_rare], band[still_rare])
    logger.info(
        "Strata: %d distinct, %d rows merged to canton-agnostic bands, %d absorbed by band",
        strata.nunique(),
        int(rare.sum()),
        int(still_rare.sum()),
    )
    return strata.rename("stratum")


def _absorb_rare(kept: pd.Series, rare_band: pd.Series) -> pd.Series:
    """Map rare rows to the largest kept stratum of their price band (fallback: largest)."""
    if kept.empty:
        return pd.Series("ALL|ALL", index=rare_band.index, dtype=object)
    # Sort ties by label so the absorbing stratum does not depend on the row order.
    counts = kept.value_counts().sort_index().sort_values(ascending=False, kind="stable")
    largest_per_band = counts.groupby(counts.index.str.split("|").str[-1]).idxmax()
    return rare_band.map(largest_per_band).fillna(counts.idxmax())


def train_calib_test_split(
    df: pd.DataFrame,
    *,
    group_col: str = "object_id",
    strata: pd.Series | npt.ArrayLike | None = None,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = RANDOM_STATE,
) -> dict[str, pd.Index]:
    """Split once into train / calibration / test, grouped by object and optionally stratified.

    The data is cut into five group-disjoint folds (``StratifiedGroupKFold`` when ``strata`` is
    given, size-balanced group assignment otherwise). The first ``5 * fractions[2]`` folds form
    the test set, the next ``5 * fractions[1]`` the calibration set, the rest the training set.

    Args:
        df: Modelling frame (index = ``listing_id``).
        group_col: Group column; no group appears in two splits.
        strata: Optional stratum labels; a Series is aligned to ``df.index`` by label (so strata
            of the full frame work for a subset), an array positionally (see :func:`make_strata`).
        fractions: Train, calib and test fractions; multiples of 0.2 summing to 1 (calib may
            be 0, train and test must be positive).
        seed: Random seed.

    Returns:
        Dict with keys ``"train"``, ``"calib"``, ``"test"`` mapping to index labels.

    Raises:
        ValueError: If the fractions are invalid, ``df.index`` is not unique or there are fewer
            than five groups.
    """
    n_test, n_calib = _fraction_folds(fractions)
    if not df.index.is_unique:
        raise ValueError("df.index must be unique (listing_id)")
    groups = group_codes(df[group_col])
    strata_values = _aligned_strata(df, strata)
    fold = _fold_ids(groups, _N_BASE_FOLDS, seed, strata_values)
    split_of_fold = np.array(["train"] * _N_BASE_FOLDS, dtype=object)
    split_of_fold[:n_test] = "test"
    split_of_fold[n_test : n_test + n_calib] = "calib"
    assigned = split_of_fold[fold]
    splits = {name: df.index[assigned == name] for name in SPLIT_NAMES}
    assert_no_group_leakage(splits, pd.Series(groups, index=df.index))
    logger.info(
        "Split sizes: %s",
        {name: f"{len(idx)} ({len(idx) / len(df):.1%})" for name, idx in splits.items()},
    )
    return splits


def grouped_cv_folds(
    df: pd.DataFrame,
    *,
    group_col: str = "object_id",
    n_splits: int = 5,
    strata: pd.Series | npt.ArrayLike | None = None,
    seed: int = RANDOM_STATE,
) -> Folds:
    """Grouped (optionally stratified) k-fold CV folds as positional indices.

    Args:
        df: Frame to split (usually the training set).
        group_col: Group column; each group lies in exactly one validation fold.
        n_splits: Number of folds.
        strata: Optional stratum labels; a Series (e.g. computed on the full frame) is aligned
            to ``df.index`` by label, an array positionally.
        seed: Random seed.

    Returns:
        List of ``(train_positions, validation_positions)`` tuples; positions index into
        ``df`` (use ``df.iloc``), not into the frame the subset was taken from.

    Raises:
        ValueError: If ``n_splits`` < 2 or exceeds the number of groups.
    """
    groups = group_codes(df[group_col])
    fold = _fold_ids(groups, n_splits, seed, _aligned_strata(df, strata))
    return _folds_from_ids(fold, n_splits)


def spatial_cv_folds(
    df: pd.DataFrame,
    *,
    spatial_col: str = "municipality_id",
    n_splits: int = 5,
    seed: int = RANDOM_STATE,
    group_col: str | None = "object_id",
) -> Folds:
    """Spatially grouped k-fold CV: whole municipalities are held out together.

    Rows with a missing spatial key fall back to their object group. If ``group_col`` exists,
    spatial units linked by a shared object are merged (connected components), so an object is
    never on both sides even when its listings were geocoded to different municipalities.
    Groups are assigned largest-first to the fold with the fewest rows; ties are broken by a
    seeded permutation.

    Args:
        df: Frame to split.
        spatial_col: Spatial unit column (e.g. ``municipality_id`` or ``district_id``).
        n_splits: Number of folds.
        seed: Random seed.
        group_col: Object column to merge on; ignored if ``None`` or absent from ``df``.

    Returns:
        List of ``(train_positions, validation_positions)`` tuples.

    Raises:
        ValueError: If there are fewer spatial groups than folds.
    """
    has_objects = group_col is not None and group_col in df.columns
    objects = group_codes(df[group_col] if has_objects else df.index.to_series())
    spatial = df[spatial_col]
    n_missing = int(spatial.isna().sum())
    if n_missing:
        logger.warning("%d rows without %s fall back to object groups", n_missing, spatial_col)
    fallback = "obj:" + pd.Series(objects, index=df.index).astype(str)
    keys = ("geo:" + spatial.astype(str)).where(spatial.notna(), fallback)
    groups = group_codes(keys)
    if has_objects:
        groups = _merge_linked_groups(groups, objects)
    fold = _fold_ids(groups, n_splits, seed, None)
    return _folds_from_ids(fold, n_splits)


def temporal_split(
    df: pd.DataFrame,
    *,
    time_col: str = "observed_at",
    freeze_date: str,
    group_col: str | None = "object_id",
) -> tuple[pd.Index, pd.Index]:
    """Split at the data-freeze date into a development and a temporal (drift) test set.

    A date-only (midnight) ``freeze_date`` is inclusive: every observation on that calendar
    day belongs to the development period. Comparisons are on absolute instants: strings are
    parsed with ``utc=True`` (mixed offsets such as ``+01:00``/``+02:00`` across the DST switch
    are fine), naive timestamps are treated as UTC, a naive ``freeze_date`` takes the data's
    time zone (UTC for naive data, so a date-only freeze means UTC midnight). Rows with a
    missing or unparsable timestamp are dropped (with a warning). If ``group_col`` exists,
    objects already observed before the freeze are removed from the post-freeze set, so the
    drift check only contains unseen objects.

    Args:
        df: Frame with an observation timestamp.
        time_col: Timestamp column (anything ``pd.to_datetime`` parses).
        freeze_date: Freeze date, e.g. ``"2026-06-07"``.
        group_col: Object column for the unseen-object filter; ``None`` disables it.

    Returns:
        ``(pre_freeze_index, post_freeze_index)``.

    Raises:
        ValueError: If ``freeze_date`` cannot be parsed.
    """
    observed = df[time_col]
    if not pd.api.types.is_datetime64_any_dtype(observed):
        observed = pd.to_datetime(observed, errors="coerce", format="mixed", utc=True)
    cutoff = _freeze_cutoff(freeze_date, observed.dt.tz)
    n_missing = int(observed.isna().sum())
    if n_missing:
        logger.warning("temporal_split: %d rows without %s are dropped", n_missing, time_col)
    pre_mask = (observed < cutoff).to_numpy()
    post_mask = (observed >= cutoff).to_numpy()
    if group_col is not None and group_col in df.columns:
        seen = df[group_col].isin(df.loc[pre_mask, group_col].dropna()).to_numpy()
        n_seen = int((post_mask & seen).sum())
        if n_seen:
            logger.info("temporal_split: %d post-freeze rows of already seen objects", n_seen)
        post_mask &= ~seen
    return df.index[pre_mask], df.index[post_mask]


def _freeze_cutoff(freeze_date: str, tz: tzinfo | None) -> pd.Timestamp:
    """Exclusive cutoff instant, comparable with timestamps in time zone ``tz`` (None = naive)."""
    try:
        cutoff = pd.Timestamp(freeze_date)
    except ValueError as exc:
        raise ValueError(f"Cannot parse freeze_date {freeze_date!r}") from exc
    if pd.isna(cutoff):
        raise ValueError(f"Cannot parse freeze_date {freeze_date!r}")
    if cutoff == cutoff.normalize():
        cutoff = cutoff + pd.Timedelta(days=1)
    if tz is None:
        return cutoff if cutoff.tzinfo is None else cutoff.tz_convert("UTC").tz_localize(None)
    return cutoff.tz_localize(tz) if cutoff.tzinfo is None else cutoff


def assert_no_group_leakage(splits: dict[str, pd.Index], groups: pd.Series) -> None:
    """Check that splits are disjoint in rows and in groups.

    Missing group values are treated as singleton groups (never counted as leakage).

    Args:
        splits: Mapping split name -> index labels.
        groups: Group labels indexed like the original frame.

    Raises:
        AssertionError: If a row or a group appears in more than one split.
        ValueError: If a split contains labels that are not in ``groups.index``.
    """
    names = list(splits)
    for name in names:
        unknown = splits[name].difference(groups.index)
        if len(unknown):
            raise ValueError(f"Split {name!r} has {len(unknown)} labels not in groups")
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            rows = splits[first].intersection(splits[second])
            if len(rows):
                raise AssertionError(f"{len(rows)} rows in both {first!r} and {second!r}")
            shared = set(groups.loc[splits[first]].dropna()) & set(
                groups.loc[splits[second]].dropna()
            )
            if shared:
                examples = sorted(map(str, shared))[:5]
                raise AssertionError(
                    f"{len(shared)} groups in both {first!r} and {second!r}, e.g. {examples}"
                )


def split_summary(
    df: pd.DataFrame,
    splits: dict[str, pd.Index],
    cols: Sequence[str],
    *,
    group_col: str | None = "object_id",
) -> pd.DataFrame:
    """Describe splits side by side (sizes and column distributions).

    Numeric columns get ``<col>_mean`` and ``<col>_median``; other columns get
    ``<col>_n_unique``, ``<col>_top`` and ``<col>_top_share``. A final ``all`` row describes
    the union of all splits.

    Args:
        df: Frame the splits index into.
        splits: Mapping split name -> index labels.
        cols: Columns to describe.
        group_col: Object column for the ``n_groups`` count; skipped if absent.

    Returns:
        One row per split (plus ``all``) with ``n``, ``share``, optional ``n_groups`` and the
        column statistics.
    """
    union = pd.Index(np.concatenate([idx.to_numpy() for idx in splits.values()])).unique()
    parts = {**splits, "all": union}
    total = max(len(union), 1)
    rows = {name: _describe(df.loc[idx], cols, total, group_col) for name, idx in parts.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def group_codes(values: npt.ArrayLike) -> np.ndarray:
    """Encode group labels as integer codes; missing labels become singleton groups.

    Codes do not depend on the row order: labels are numbered in sorted order, and rows with a
    missing label get the codes after them, ordered by their index label if ``values`` is a
    ``pd.Series`` (else by position).

    Args:
        values: Group label per row (e.g. ``object_id``).

    Returns:
        ``int64`` codes ``0 .. n_groups - 1``.
    """
    labels = pd.Series(np.asarray(values, dtype=object).ravel())
    codes, uniques = pd.factorize(labels, use_na_sentinel=True, sort=True)
    missing = np.flatnonzero(codes < 0)
    if len(missing):
        logger.warning("%d rows with missing group key are singleton groups", len(missing))
        if isinstance(values, pd.Series):
            rank = pd.factorize(values.index[missing], sort=True)[0]
            missing = missing[np.argsort(rank, kind="stable")]
        codes[missing] = len(uniques) + np.arange(len(missing))
    return codes.astype(np.int64)


def _describe(
    part: pd.DataFrame, cols: Sequence[str], total: int, group_col: str | None
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"n": len(part), "share": len(part) / total}
    if group_col is not None and group_col in part.columns:
        row["n_groups"] = int(part[group_col].nunique())
    for col in cols:
        values = part[col]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_median"] = float(values.median())
            continue
        counts = values.value_counts(dropna=True)
        row[f"{col}_n_unique"] = int(values.nunique())
        row[f"{col}_top"] = str(counts.index[0]) if len(counts) else ""
        row[f"{col}_top_share"] = float(counts.iloc[0] / len(part)) if len(counts) else np.nan
    return row


def _fraction_folds(fractions: tuple[float, float, float]) -> tuple[int, int]:
    if len(fractions) != 3:
        raise ValueError("fractions must be (train, calib, test)")
    folds = [f * _N_BASE_FOLDS for f in fractions]
    if any(f < -_TOL or abs(f - round(f)) > 1e-6 for f in folds):
        raise ValueError(f"fractions must be multiples of 0.2, got {fractions}")
    n_train, n_calib, n_test = (round(f) for f in folds)
    if n_train + n_calib + n_test != _N_BASE_FOLDS or n_train < 1 or n_test < 1:
        raise ValueError(f"fractions must sum to 1 with train, test > 0, got {fractions}")
    return n_test, n_calib


def _aligned_strata(
    df: pd.DataFrame, strata: pd.Series | npt.ArrayLike | None
) -> np.ndarray | None:
    if strata is None:
        return None
    if isinstance(strata, pd.Series):
        aligned = strata.reindex(df.index)
    else:
        aligned = pd.Series(np.asarray(strata, dtype=object).ravel())
        if len(aligned) != len(df):
            raise ValueError(f"strata has {len(aligned)} values, expected {len(df)}")
    if aligned.isna().any():
        raise ValueError("strata must have a non-missing value for every row of df")
    return aligned.astype(str).to_numpy()


def _fold_ids(
    groups: np.ndarray, n_splits: int, seed: int, strata: np.ndarray | None
) -> np.ndarray:
    """Assign every row to a fold so that groups never straddle folds."""
    n_groups = int(groups.max()) + 1 if len(groups) else 0
    if n_splits < 2 or n_groups < n_splits:
        raise ValueError(f"Need 2 <= n_splits <= n_groups, got {n_splits=} and {n_groups=}")
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(n_groups)[groups]
    if strata is None:
        return _greedy_group_folds(permuted, n_groups, n_splits)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    fold = np.full(len(groups), -1, dtype=np.int64)
    for k, (_, test_pos) in enumerate(splitter.split(np.zeros(len(groups)), strata, permuted)):
        fold[test_pos] = k
    return fold


def _greedy_group_folds(codes: np.ndarray, n_groups: int, n_splits: int) -> np.ndarray:
    """Largest group first into the lightest fold; ties keep the (random) code order."""
    sizes = np.bincount(codes, minlength=n_groups)
    order = np.argsort(-sizes, kind="stable")
    fold_of_group = np.empty(n_groups, dtype=np.int64)
    load = [0] * n_splits
    for g in order:
        k = min(range(n_splits), key=load.__getitem__)
        fold_of_group[g] = k
        load[k] += int(sizes[g])
    return fold_of_group[codes]


def _folds_from_ids(fold: np.ndarray, n_splits: int) -> Folds:
    folds = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(n_splits)]
    sizes = [len(val) for _, val in folds]
    logger.info("CV fold sizes: %s", sizes)
    return folds


def _merge_linked_groups(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Connected components of the bipartite graph first-code <-> second-code."""
    n_first = int(first.max()) + 1
    n_nodes = n_first + int(second.max()) + 1
    edges = coo_matrix((np.ones(len(first)), (first, n_first + second)), shape=(n_nodes, n_nodes))
    _, labels = connected_components(edges, directed=False)
    merged = np.unique(labels[first], return_inverse=True)[1].astype(np.int64)
    n_merged = n_first - (int(merged.max()) + 1)
    if n_merged > 0:
        logger.info("Merged %d spatial units linked by shared objects", n_merged)
    return merged
