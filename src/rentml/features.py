"""Feature engineering: DSPRO1 features, hierarchical target encoding and kNN price features.

Target-derived features are leakage-aware: the target encoder returns out-of-fold encodings from
``fit_transform``, and the kNN price features have an out-of-fold mode next to the DSPRO1
leave-one-out reproduction.
"""

import logging
from collections.abc import Sequence
from typing import Literal, Self, get_args

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from rentml.config import RANDOM_STATE, REFERENCE_YEAR

logger = logging.getLogger(__name__)

BASE = ["area", "rooms", "area_per_room", "rooms_is_integer"]
BUILDING = ["year_built", "building_age", "apartments", "land_area", "land_area_per_apartment"]
LOCATION = ["population", "oev", "solar", "elevation", "muni_density", "muni_population"]
COORDS = ["east", "north"]
ROT_COORDS = ["rot45_x", "rot45_y"]

# LV95 false origin (Bern): centring keeps the rotated coordinates in a readable range (metres).
_LV95_ORIGIN = (2_600_000.0, 1_200_000.0)
_KEY_SEP = "\x1f"
KnnMode = Literal["loo", "oof"]
FoldSeq = Sequence[tuple[np.ndarray, np.ndarray]]


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        logger.warning("Column %r missing; derived features will be NaN", col)
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den > 0)).astype(float)


def add_engineered_features(
    df: pd.DataFrame,
    reference_year: int = REFERENCE_YEAR,
    *,
    rotation_angles: Sequence[float] = (45.0,),
) -> pd.DataFrame:
    """Add the DSPRO1 engineered features plus rotated coordinates.

    Features: ``area_per_room`` (area / rooms, NaN for rooms <= 0), ``rooms_is_integer`` (only
    computed when ``data.fix_schema`` has not added it yet), ``building_age`` (reference year
    minus ``year_built``, clipped at 0 for buildings still under construction),
    ``land_area_per_apartment`` (``land_area`` / ``apartments``, NaN for apartments <= 0) and
    rotated LV95 coordinates ``rot<angle>_x`` / ``rot<angle>_y`` (centred on the LV95 origin).
    Axis-aligned tree splits can then follow diagonal structures (lake shores, valleys) with
    fewer splits.

    Missing input columns do not raise: the derived feature is NaN and a warning is logged.

    Args:
        df: Modelling frame with canonical column names.
        reference_year: Year used to compute the building age.
        rotation_angles: Rotation angles in degrees (counter-clockwise) for the rotated coordinates.

    Returns:
        A copy of ``df`` with the engineered columns added (existing ones are overwritten).
    """
    out = df.copy()
    area, rooms = _numeric(out, "area"), _numeric(out, "rooms")
    out["area_per_room"] = _safe_ratio(area, rooms)
    if "rooms_is_integer" not in out.columns:
        is_int = np.isclose(rooms, np.round(rooms))
        out["rooms_is_integer"] = pd.Series(is_int, index=out.index, dtype=float).where(
            rooms.notna()
        )
    out["building_age"] = (reference_year - _numeric(out, "year_built")).clip(lower=0)
    out["land_area_per_apartment"] = _safe_ratio(
        _numeric(out, "land_area"), _numeric(out, "apartments")
    )
    east = _numeric(out, "east") - _LV95_ORIGIN[0]
    north = _numeric(out, "north") - _LV95_ORIGIN[1]
    for angle in rotation_angles:
        theta = np.deg2rad(angle)
        name = f"rot{angle:g}"
        out[f"{name}_x"] = east * np.cos(theta) - north * np.sin(theta)
        out[f"{name}_y"] = east * np.sin(theta) + north * np.cos(theta)
    return out


def _as_key(values: pd.Series) -> pd.Series:
    """Stringify codes so 261, 261.0, "261" and object-dtype 1.0/None columns share keys."""
    if not is_numeric_dtype(values) or is_bool_dtype(values):
        numeric = pd.to_numeric(values.astype(object), errors="coerce")
        if numeric.notna().sum() != values.notna().sum():
            return values.astype(str)
        values = numeric
    numeric = values.astype(float)
    finite = numeric.dropna()
    if np.array_equal(finite, np.round(finite)):
        return numeric.astype("Int64").astype(str)
    return numeric.astype(str)


def _path_keys(frame: pd.DataFrame, cols: Sequence[str]) -> list[pd.Series]:
    """Build one path key per hierarchy level; NaN if any level up to it is missing."""
    keys: list[pd.Series] = []
    current = pd.Series("", index=frame.index, dtype=object)
    missing = pd.Series(False, index=frame.index)
    for level, col in enumerate(cols):
        values = frame[col]
        missing = missing | values.isna()
        part = _as_key(values)
        current = part if level == 0 else current + _KEY_SEP + part
        keys.append(current.where(~missing))
    return keys


def _fit_level_stats(
    keys: list[pd.Series], y: np.ndarray, smoothing: float
) -> tuple[float, list[pd.Series]]:
    """Return the global mean and one ``key -> shrunk mean`` table per level."""
    global_mean = float(np.mean(y))
    parent = np.full(len(y), global_mean)
    tables: list[pd.Series] = []
    for key in keys:
        frame = pd.DataFrame({"key": key.to_numpy(), "y": y, "parent": parent})
        stats = (
            frame.dropna(subset=["key"])
            .groupby("key", sort=False)
            .agg(total=("y", "sum"), n=("y", "size"), parent=("parent", "first"))
        )
        table = (stats["total"] + smoothing * stats["parent"]) / (stats["n"] + smoothing)
        tables.append(table)
        parent = _lookup(key, table, parent)
    return global_mean, tables


def _lookup(key: pd.Series, table: pd.Series, parent: np.ndarray) -> np.ndarray:
    mapped = key.map(table).to_numpy(dtype=float)
    return np.where(np.isnan(mapped), parent, mapped)


def _apply_level_stats(
    keys: list[pd.Series], global_mean: float, tables: list[pd.Series]
) -> np.ndarray:
    n_rows = len(keys[0]) if keys else 0
    parent = np.full(n_rows, global_mean)
    columns: list[np.ndarray] = []
    for key, table in zip(keys, tables, strict=True):
        parent = _lookup(key, table, parent)
        columns.append(parent)
    return np.column_stack(columns) if columns else np.empty((n_rows, 0))


class HierarchicalTargetEncoder(TransformerMixin, BaseEstimator):
    """Nested target encoding with shrinkage towards the parent level (empirical Bayes style).

    For level 0 the encoding of a category with ``n`` rows and mean target ``ybar`` is
    ``(n * ybar + m * global_mean) / (n + m)``; each deeper level is shrunk towards the encoding of
    its parent instead of the global mean. Categories are identified by their full path
    (e.g. canton / district / municipality), so non-unique child codes are handled correctly.
    Unseen or missing categories fall back to the parent encoding (global mean for level 0).

    Pass the target on the modelling scale, i.e. ``log_price``.

    Args:
        cols: Hierarchy columns from coarsest to finest.
        smoothing: Pseudo-count ``m`` of the parent prior.
        n_inner: Number of inner folds used by :meth:`fit_transform` for out-of-fold encodings.
        random_state: Seed of the inner fold shuffling.
        group_col: Optional column of ``X`` (e.g. ``object_id``) used as inner-fold groups when
            ``fit_transform`` gets no ``groups``; lets sklearn pipelines stay grouped without
            metadata routing.

    Attributes:
        global_mean_: Mean target of the full fit.
        tables_: One ``pd.Series`` (path key -> encoding) per level from the full fit.
        feature_names_in_: Hierarchy columns seen during fit.
    """

    def __init__(
        self,
        cols: Sequence[str] = ("re_canton", "re_district", "re_municipality"),
        smoothing: float = 20.0,
        n_inner: int = 5,
        random_state: int = RANDOM_STATE,
        group_col: str | None = None,
    ) -> None:
        self.cols = cols
        self.smoothing = smoothing
        self.n_inner = n_inner
        self.random_state = random_state
        self.group_col = group_col

    def _frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        cols = list(self.cols)
        if isinstance(X, pd.DataFrame):
            missing = [c for c in cols if c not in X.columns]
            if missing:
                raise KeyError(f"HierarchicalTargetEncoder: columns missing from X: {missing}")
            return X[cols]
        arr = np.asarray(X, dtype=object)
        if arr.ndim != 2 or arr.shape[1] != len(cols):
            raise ValueError(f"Expected a 2-D array with {len(cols)} columns, got {arr.shape}")
        return pd.DataFrame(arr, columns=cols)

    def _prepare(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> tuple[pd.DataFrame, np.ndarray, list[pd.Series]]:
        if self.smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {self.smoothing}")
        frame = self._frame(X)
        target = np.asarray(y, dtype=float).ravel()
        if len(target) != len(frame):
            raise ValueError(f"X has {len(frame)} rows but y has {len(target)}")
        if np.isnan(target).any():
            raise ValueError("y contains NaN; drop or impute the target first")
        return frame, target, _path_keys(frame, self.cols)

    def _store(self, X: pd.DataFrame | np.ndarray, keys: list[pd.Series], y: np.ndarray) -> None:
        self.global_mean_, self.tables_ = _fit_level_stats(keys, y, float(self.smoothing))
        self.n_features_in_ = int(np.shape(X)[1])
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

    def _wrap(self, values: np.ndarray, index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame(values, index=index, columns=self.get_feature_names_out())

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> Self:
        """Fit the encoding tables on all rows.

        Args:
            X: Frame containing the hierarchy columns (extra columns are ignored).
            y: Target on the modelling scale (log price).

        Returns:
            The fitted encoder.

        Raises:
            KeyError: If a hierarchy column is missing.
            ValueError: If ``y`` has a different length or contains NaN.
        """
        _, target, keys = self._prepare(X, y)
        self._store(X, keys, target)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Encode new rows with the full-fit statistics.

        Args:
            X: Frame containing the hierarchy columns.

        Returns:
            DataFrame with one ``te_<col>`` column per level, index aligned with ``X``.
        """
        check_is_fitted(self, "tables_")
        frame = self._frame(X)
        values = _apply_level_stats(_path_keys(frame, self.cols), self.global_mean_, self.tables_)
        return self._wrap(values, frame.index)

    def fit_transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups: pd.Series | np.ndarray | None = None,
        **fit_params: object,
    ) -> pd.DataFrame:
        """Fit on all rows and return **out-of-fold** encodings for the training rows.

        Each row is encoded with statistics from the other inner folds only (seeded shuffled
        group K-fold when groups are given, e.g. ``object_id``, so duplicates of one object never
        inform each other; shuffled row K-fold otherwise). The full fit is stored for later
        :meth:`transform`.

        Args:
            X: Frame containing the hierarchy columns (and ``group_col`` if set).
            y: Target on the modelling scale (log price).
            groups: Optional group labels for the inner split; overrides ``group_col``.
            **fit_params: Ignored; accepted for sklearn pipeline compatibility.

        Returns:
            DataFrame with one ``te_<col>`` column per level, index aligned with ``X``.

        Raises:
            KeyError: If ``group_col`` is set but missing from ``X``.
            ValueError: If ``y`` is missing or there are fewer rows/groups than inner folds.
        """
        if y is None:
            raise ValueError("HierarchicalTargetEncoder.fit_transform requires y")
        frame, target, keys = self._prepare(X, y)
        self._store(X, keys, target)
        if groups is None and self.group_col is not None:
            if not isinstance(X, pd.DataFrame) or self.group_col not in X.columns:
                raise KeyError(f"group_col {self.group_col!r} not found in X")
            groups = X[self.group_col]
        oof = np.full((len(frame), len(self.cols)), np.nan)
        for train_idx, val_idx in self._inner_splits(len(frame), groups):
            mean, tables = _fit_level_stats(
                [k.iloc[train_idx] for k in keys], target[train_idx], float(self.smoothing)
            )
            oof[val_idx] = _apply_level_stats([k.iloc[val_idx] for k in keys], mean, tables)
        return self._wrap(oof, frame.index)

    def _inner_splits(
        self, n_rows: int, groups: pd.Series | np.ndarray | None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        # Seeded shuffled group K-fold (rows are their own groups without ``groups``), built by
        # hand because GroupKFold(shuffle=True) needs scikit-learn >= 1.6.
        if self.n_inner < 2:
            raise ValueError(f"n_inner must be >= 2, got {self.n_inner}")
        if groups is None:
            codes = np.arange(n_rows)
        else:
            group_arr = np.asarray(groups)
            if len(group_arr) != n_rows:
                raise ValueError(f"groups has {len(group_arr)} entries, expected {n_rows}")
            codes = pd.factorize(group_arr, use_na_sentinel=False)[0]
        n_groups = int(codes.max()) + 1 if n_rows else 0
        if n_groups < self.n_inner:
            raise ValueError(f"{n_groups} rows/groups is fewer than n_inner={self.n_inner}")
        perm = np.random.default_rng(self.random_state).permutation(n_groups)
        fold = (perm % self.n_inner)[codes]
        return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(self.n_inner)]

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return the output column names ``te_<col>``.

        Args:
            input_features: Ignored; present for sklearn API compatibility.

        Returns:
            Array of output feature names.
        """
        return np.asarray([f"te_{c}" for c in self.cols], dtype=object)


def _scaled_coords(
    train_df: pd.DataFrame, frames: list[pd.DataFrame], coord_cols: Sequence[str]
) -> list[np.ndarray]:
    cols = list(coord_cols)
    for frame in (train_df, *frames):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise KeyError(f"knn_price_features: coordinate columns missing: {missing}")
    train_xy = train_df[cols].to_numpy(dtype=float)
    valid = ~np.isnan(train_xy).any(axis=1)
    if not valid.any():
        raise ValueError("knn_price_features: no training row has valid coordinates")
    scaler = StandardScaler().fit(train_xy[valid])
    return [scaler.transform(f[cols].to_numpy(dtype=float)) for f in (train_df, *frames)]


def _neighbour_stats(
    pool_xy: np.ndarray, pool_y: np.ndarray, query_xy: np.ndarray, k: int
) -> np.ndarray:
    """Mean and median target of the k nearest pool points; NaN rows for invalid queries."""
    out = np.full((len(query_xy), 2), np.nan)
    ok = ~np.isnan(query_xy).any(axis=1)
    if not ok.any() or len(pool_xy) == 0:
        return out
    n_nb = min(k, len(pool_xy))
    idx = NearestNeighbors(n_neighbors=n_nb).fit(pool_xy).kneighbors(query_xy[ok])[1]
    vals = pool_y[idx]
    out[ok] = np.column_stack([vals.mean(axis=1), np.median(vals, axis=1)])
    return out


def _loo_stats(xy: np.ndarray, y: np.ndarray, pool: np.ndarray, k: int) -> np.ndarray:
    """LOO neighbour stats for every row with coordinates; self excluded by identity."""
    out = np.full((len(xy), 2), np.nan)
    pool_pos = np.flatnonzero(pool)
    rows = np.flatnonzero(~np.isnan(xy).any(axis=1))
    if len(pool_pos) < 2:
        return out
    n_nb = min(k + 1, len(pool_pos))
    idx = NearestNeighbors(n_neighbors=n_nb).fit(xy[pool_pos]).kneighbors(xy[rows])[1]
    idx = pool_pos[idx]
    # Keep the first k neighbours other than the row itself (distance order). Other rows at
    # identical coordinates (duplicates) stay in and leak their price.
    keep = idx != rows[:, None]
    keep &= np.cumsum(keep, axis=1) <= k
    vals = np.where(keep, y[idx], np.nan)
    out[rows] = np.column_stack([np.nanmean(vals, axis=1), np.nanmedian(vals, axis=1)])
    return out


def _oof_stats(
    xy: np.ndarray, y: np.ndarray, pool: np.ndarray, k: int, folds: FoldSeq
) -> np.ndarray:
    out = np.full((len(xy), 2), np.nan)
    for train_idx, val_idx in folds:
        tr = np.asarray(train_idx)[pool[np.asarray(train_idx)]]
        out[val_idx] = _neighbour_stats(xy[tr], y[tr], xy[np.asarray(val_idx)], k)
    uncovered = int(np.isnan(out[:, 0]).sum())
    if uncovered:
        logger.warning("knn_price_features(oof): %d rows got no OOF value (NaN)", uncovered)
    return out


def knn_price_features(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame | None = None,
    *,
    k: int = 10,
    target: str = "price",
    mode: KnnMode = "loo",
    folds: FoldSeq | None = None,
    coord_cols: Sequence[str] = ("east", "north"),
    prefix: str = "knn_price",
) -> pd.DataFrame:
    """Mean and median target of the k nearest training listings (DSPRO1 location feature).

    Coordinates are standardised with a scaler fitted on the training rows (as in DSPRO1).

    * ``apply_df`` given: every row of ``apply_df`` uses its k nearest training rows (``mode`` and
      ``folds`` are ignored). Use this for calibration/test rows.
    * ``mode="loo"`` (DSPRO1 reproduction): each training row uses the k nearest *other* training
      rows. The row itself is excluded by identity (DSPRO1 simply dropped the first neighbour,
      which is not always the row itself when coordinates tie). Exact duplicates - other
      listings of the same object - are **not** excluded and leak their price into the feature.
    * ``mode="oof"``: each training row uses neighbours from the training part of the fold in
      which it is a validation row; with grouped folds duplicates never inform each other.

    Rows with missing coordinates or target are excluded from the neighbour pool; only rows with
    missing coordinates get NaN features (rows without target are still featurised).

    Args:
        train_df: Training rows with coordinates and the target column.
        apply_df: Optional rows to featurise with all training neighbours.
        k: Number of neighbours.
        target: Target column aggregated over the neighbours (CHF by default).
        mode: ``"loo"`` or ``"oof"`` (only used when ``apply_df`` is None).
        folds: Positional ``(train_idx, val_idx)`` pairs over ``train_df``; required for ``"oof"``.
        coord_cols: Coordinate columns (LV95 metres).
        prefix: Output column prefix.

    Returns:
        DataFrame with ``<prefix>_mean`` and ``<prefix>_median``, indexed like the featurised rows.

    Raises:
        ValueError: For an unknown mode, ``k < 1``, or ``mode="oof"`` without folds.
        KeyError: If coordinate or target columns are missing.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if mode not in get_args(KnnMode):
        raise ValueError(f"mode must be one of {get_args(KnnMode)}, got {mode!r}")
    if target not in train_df.columns:
        raise KeyError(f"knn_price_features: target column {target!r} missing")
    columns = [f"{prefix}_mean", f"{prefix}_median"]
    extra = [] if apply_df is None else [apply_df]
    coords = _scaled_coords(train_df, extra, coord_cols)
    y = pd.to_numeric(train_df[target], errors="coerce").to_numpy(dtype=float)
    pool = ~np.isnan(coords[0]).any(axis=1) & ~np.isnan(y)
    if apply_df is not None:
        stats = _neighbour_stats(coords[0][pool], y[pool], coords[1], k)
        return pd.DataFrame(stats, index=apply_df.index, columns=columns)
    if mode == "loo":
        stats = _loo_stats(coords[0], y, pool, k)
    elif folds is None:
        raise ValueError("mode='oof' requires folds (positional train/val index pairs)")
    else:
        stats = _oof_stats(coords[0], y, pool, k, folds)
    return pd.DataFrame(stats, index=train_df.index, columns=columns)


def monotone_vector(
    feature_cols: list[str],
    increasing: Sequence[str] = ("area",),
    *,
    decreasing: Sequence[str] = (),
) -> list[int]:
    """Build a LightGBM ``monotone_constraints`` vector for the given feature order.

    Args:
        feature_cols: Model features in training order.
        increasing: Features with a non-decreasing effect (+1), e.g. living area.
        decreasing: Features with a non-increasing effect (-1).

    Returns:
        One entry per feature: +1, -1 or 0 (unconstrained).

    Raises:
        ValueError: If a feature is listed as both increasing and decreasing.
    """
    both = set(increasing) & set(decreasing)
    if both:
        raise ValueError(f"Features cannot be both increasing and decreasing: {sorted(both)}")
    absent = [c for c in (*increasing, *decreasing) if c not in feature_cols]
    if absent:
        logger.warning("monotone_vector: constrained features not in feature_cols: %s", absent)
    inc, dec = set(increasing), set(decreasing)
    return [1 if c in inc else -1 if c in dec else 0 for c in feature_cols]
