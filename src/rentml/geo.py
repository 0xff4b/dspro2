"""Swiss administrative geography for the rent models.

Downloads swissBOUNDARIES3D, assigns listings (LV95 points) to municipality, district and
canton, derives the language region and builds the nested random-effect keys
(canton > district > municipality) used by GPBoost and target encoding. Address parsing lives
in :mod:`rentml.address`; :func:`parse_address` is re-exported here.
"""

import logging
import re
import zipfile
import zlib
from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyogrio.errors import DataSourceError

from rentml.address import parse_address as parse_address
from rentml.config import CANTON_ABBR, SB3D_VERSION

logger = logging.getLogger(__name__)

SB3D_BASE_URL = "https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d"
LV95_EPSG = 2056
MUNICIPALITY_LAYER = "tlm_hoheitsgebiet"
DISTRICT_LAYER = "tlm_bezirksgebiet"
CANTON_LAYER = "tlm_kantonsgebiet"
MUNICIPALITY_OBJECT_TYPE = "Gemeindegebiet"

_INT_COLUMNS = ("municipality_id", "district_id", "canton_id")
_NAME_COLUMNS = ("municipality_name", "district_name", "canton")
_FLOAT_COLUMNS = ("muni_population", "muni_area_km2", "muni_density")
UNIT_COLUMNS: tuple[str, ...] = (*_INT_COLUMNS, *_NAME_COLUMNS, *_FLOAT_COLUMNS)
ADMIN_COLUMNS: tuple[str, ...] = (*UNIT_COLUMNS, "lang_region", "admin_match")
_JOIN_COLUMNS = [*UNIT_COLUMNS, "lang_region"]
_ADMIN_DTYPES = {c: "Int64" for c in _INT_COLUMNS} | {c: "float64" for c in _FLOAT_COLUMNS}

# Majority language per canton. Romansh areas (GR) count as "de": the project only
# distinguishes de/fr/it and German is the common written language there.
FRENCH_CANTONS = frozenset({"GE", "VD", "NE", "JU", "FR", "VS"})
ITALIAN_CANTONS = frozenset({"TI"})
_CANTON_LANGUAGE = dict.fromkeys(FRENCH_CANTONS, "fr") | dict.fromkeys(ITALIAN_CANTONS, "it")

# District-level overrides for bilingual cantons (names as in swissBOUNDARIES3D; "Raron"
# covers both half-districts). Biel/Bienne is bilingual with ~70 % German speakers -> "de".
# See/Lac is ~60 % German-speaking; its French-majority municipalities Courtepin,
# Misery-Courtion and Mont-Vully are overrides (Courgevaux is ~60 % German and stays "de").
# Maloja stays "de": the Upper Engadine (most of its population) is German-majority; the
# Italian-speaking Bregaglia is a municipality override.
_OBERWALLIS = ("Goms", "Brig", "Visp", "Raron", "Westlich Raron", "Östlich Raron", "Leuk")
DISTRICT_LANGUAGE_OVERRIDES: dict[tuple[str, str], str] = {
    ("BE", "Jura bernois"): "fr",
    ("BE", "Biel/Bienne"): "de",
    **{("FR", name): "de" for name in ("Sense", "See", "Lac")},
    **{("VS", name): "de" for name in _OBERWALLIS},
    **{("GR", name): "it" for name in ("Moesa", "Bernina")},
}
MUNICIPALITY_LANGUAGE_OVERRIDES: dict[tuple[str, str], str] = {
    ("GR", "Bregaglia"): "it",
    **{("FR", name): "fr" for name in ("Courtepin", "Misery-Courtion", "Mont-Vully")},
}

COUNT_BUCKET_ORDER: tuple[str, ...] = ("<5", "5-20", ">20")

_VERSION_RE = re.compile(r"\d{4}-\d{2}")
_VERSION_MARKER = ".sb3d_version"
_GPKG_PREFIX = "swissboundaries3d"
_CHUNK = 1 << 20


def sb3d_url(version: str) -> str:
    """Build the download URL of the swissBOUNDARIES3D GeoPackage (LV95, LN02).

    Args:
        version: Release ``YYYY-MM`` (e.g. ``"2026-01"``).

    Returns:
        The URL of the zipped GeoPackage on data.geo.admin.ch.

    Raises:
        ValueError: If ``version`` is not of the form ``YYYY-MM``.
    """
    if not _VERSION_RE.fullmatch(version):
        raise ValueError(f"swissBOUNDARIES3D version must look like 'YYYY-MM', got {version!r}")
    name = f"swissboundaries3d_{version}"
    return f"{SB3D_BASE_URL}/{name}/{name}_{LV95_EPSG}_5728.gpkg.zip"


def download_swissboundaries(
    dest_dir: Path, version: str = SB3D_VERSION, *, overwrite: bool = False, timeout: float = 60.0
) -> Path:
    """Download and unzip swissBOUNDARIES3D, reusing a cached GeoPackage.

    Args:
        dest_dir: Target directory (created if needed).
        version: Release ``YYYY-MM``. The file named by the marker of an earlier download of
            this release is reused; without a marker, a readable ``swissBOUNDARIES3D*.gpkg``.
        overwrite: Force a fresh download even if a cached file exists.
        timeout: Connect/read timeout in seconds for the HTTP request.

    Returns:
        Path to the extracted ``.gpkg`` file.

    Raises:
        RuntimeError: If the download fails or the archive is not a valid zip file.
        FileNotFoundError: If the archive contains no ``.gpkg`` file.
    """
    url = sb3d_url(version)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cached = None if overwrite else _cached_gpkg(dest_dir, version)
    if cached is not None:
        logger.info("Using cached swissBOUNDARIES3D file %s", cached)
        return cached
    zip_path = dest_dir / url.rsplit("/", 1)[-1]
    if overwrite or not zip_path.is_file():
        logger.info("Downloading %s", url)
        _stream_download(url, zip_path, timeout=timeout)
    gpkg = _extract_gpkg(zip_path, dest_dir)
    (dest_dir / _VERSION_MARKER).write_text(f"{version}\t{gpkg.name}", encoding="utf-8")
    logger.info("swissBOUNDARIES3D %s extracted to %s", version, gpkg)
    return gpkg


def _cached_gpkg(dest_dir: Path, version: str) -> Path | None:
    marker = dest_dir / _VERSION_MARKER
    text = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
    cached_version, _, name = text.partition("\t")
    if text and cached_version != version:
        return None
    if name:  # the file extracted together with this marker
        return path if (path := dest_dir / Path(name).name).is_file() else None
    # No marker (manual copy) or a legacy one: accept only a readable GeoPackage.
    found = sorted(p for p in dest_dir.glob("*.gpkg") if p.name.lower().startswith(_GPKG_PREFIX))
    return next((p for p in reversed(found) if _has_municipality_layer(p)), None)


def _has_municipality_layer(path: Path) -> bool:
    try:
        return MUNICIPALITY_LAYER in set(gpd.list_layers(path)["name"])
    except DataSourceError:
        return False


def _stream_download(url: str, target: Path, *, timeout: float) -> None:
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            _write_atomic(response.iter_content(chunk_size=_CHUNK), target)
    except requests.RequestException as exc:
        raise RuntimeError(f"Download of {url} failed: {exc}") from exc


def _extract_gpkg(zip_path: Path, dest_dir: Path) -> Path:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = sorted(m for m in archive.namelist() if m.lower().endswith(".gpkg"))
            if not members:
                raise FileNotFoundError(f"No .gpkg file inside {zip_path}")
            # Only the base name is used, so crafted member paths cannot escape dest_dir.
            target = dest_dir / Path(members[0]).name
            with archive.open(members[0]) as src:
                _write_atomic(iter(lambda: src.read(_CHUNK), b""), target)
    except (FileNotFoundError, zipfile.BadZipFile, EOFError, zlib.error) as exc:
        zip_path.unlink(missing_ok=True)  # a broken archive must not block the next attempt
        if isinstance(exc, FileNotFoundError):
            raise
        raise RuntimeError(f"{zip_path} is not a valid zip archive: {exc}") from exc
    return target


def _write_atomic(chunks: Iterable[bytes], target: Path) -> None:
    """Write via ``<target>.part`` so a failed transfer never leaves a truncated target."""
    part = target.with_name(f"{target.name}.part")
    try:
        with part.open("wb") as handle:
            for chunk in chunks:
                handle.write(chunk)
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)  # no-op after a successful rename


def load_admin_units(gpkg: Path, *, exclude_lakes: bool = True) -> gpd.GeoDataFrame:
    """Load Swiss municipality polygons with district, canton and municipal statistics.

    Keeps Swiss ``"Gemeindegebiet"`` polygons (no cantonal lakes, condominiums, LI/DE/IT). Cantons
    without districts get ``district_id = canton_id * 100`` (BFS) and the canton name.

    Args:
        gpkg: Path to the swissBOUNDARIES3D GeoPackage.
        exclude_lakes: ``muni_area_km2`` = land area ``(gem_flaeche - see_flaeche) / 100`` so
            lakeside towns such as Vevey (~72 % lake) are not understated in density (falls
            back to total area with a warning if ``see_flaeche`` is absent); ``False`` = total.

    Returns:
        One row per municipality: ``UNIT_COLUMNS``, ``lang_region``, 2D geometry (EPSG:2056).

    Raises:
        FileNotFoundError: If ``gpkg`` does not exist.
        ValueError: If a canton number is not a BFS canton number (1-26).
    """
    fields = ["objektart", "bfs_nummer", "bezirksnummer", "kantonsnummer", "name"]
    munis = _read_municipalities(gpkg, [*fields, "einwohnerzahl", "gem_flaeche", "see_flaeche"])
    keep = munis["objektart"].eq(MUNICIPALITY_OBJECT_TYPE) & munis["kantonsnummer"].notna()
    logger.info("Dropped %d non-municipal or foreign polygons", int((~keep).sum()))
    munis = munis.loc[keep].copy()
    if exclude_lakes and "see_flaeche" not in munis.columns:
        logger.warning("%s has no see_flaeche field; using total area", gpkg.name)
        exclude_lakes = False
    canton_id = munis["kantonsnummer"].astype(int)
    unknown = sorted(set(canton_id) - set(CANTON_ABBR))
    if unknown:
        raise ValueError(f"Unknown canton numbers in {gpkg.name}: {unknown}")
    district_id = munis["bezirksnummer"].fillna(canton_id * 100).astype(int)
    districts = _name_lookup(gpkg, DISTRICT_LAYER, "bezirksnummer")
    cantons = _name_lookup(gpkg, CANTON_LAYER, "kantonsnummer")
    district_name = district_id.map(districts).fillna(canton_id.map(cantons))
    area_ha = munis["gem_flaeche"] - (munis["see_flaeche"].fillna(0.0) if exclude_lakes else 0.0)
    area_km2 = (area_ha / 100.0).where(area_ha > 0).astype(float)
    population = munis["einwohnerzahl"].astype(float)
    columns = {
        "municipality_id": munis["bfs_nummer"].astype(int),
        "district_id": district_id,
        "canton_id": canton_id,
        "municipality_name": munis["name"].astype(str),
        "district_name": district_name.astype(str),
        "canton": canton_id.map(CANTON_ABBR),
        "muni_population": population,
        "muni_area_km2": area_km2,
        "muni_density": population / area_km2,
    }
    geometry = munis.geometry.force_2d()
    units = _to_lv95(gpd.GeoDataFrame(columns, geometry=geometry, crs=munis.crs))
    units["lang_region"] = language_region(
        units["canton"], units["district_name"], municipality_name=units["municipality_name"]
    )
    logger.info("Loaded %d municipalities from %s", len(units), gpkg.name)
    return units.sort_values("municipality_id").reset_index(drop=True)


def load_foreign_mask(gpkg: Path) -> gpd.GeoSeries:
    """Load foreign municipalities (LI, Büsingen, Campione) as ``assign_admin_units`` mask.

    Args:
        gpkg: Path to the swissBOUNDARIES3D GeoPackage.

    Returns:
        2D polygons (EPSG:2056) of municipalities without a canton number (may be empty);
        pass them as ``exclude`` so points abroad are not snapped to a Swiss municipality.

    Raises:
        FileNotFoundError: If ``gpkg`` does not exist.
    """
    munis = _read_municipalities(gpkg, ["objektart", "kantonsnummer"])
    foreign = munis["objektart"].eq(MUNICIPALITY_OBJECT_TYPE) & munis["kantonsnummer"].isna()
    return _to_lv95(munis.loc[foreign]).geometry.force_2d().reset_index(drop=True)


def _read_municipalities(gpkg: Path, columns: list[str]) -> gpd.GeoDataFrame:
    if not gpkg.is_file():
        raise FileNotFoundError(f"swissBOUNDARIES3D GeoPackage not found: {gpkg}")
    return gpd.read_file(gpkg, layer=MUNICIPALITY_LAYER, columns=columns)


def _name_lookup(gpkg: Path, layer: str, key: str) -> dict[int, str]:
    table = gpd.read_file(gpkg, layer=layer, columns=[key, "name"], ignore_geometry=True)
    return {int(k): str(v) for k, v in zip(table[key], table["name"], strict=True)}


def _to_lv95(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        logger.warning("Admin units have no CRS; assuming EPSG:%d", LV95_EPSG)
        return gdf.set_crs(LV95_EPSG)
    return gdf if gdf.crs.to_epsg() == LV95_EPSG else gdf.to_crs(LV95_EPSG)


def assign_admin_units(
    df: pd.DataFrame,
    units: gpd.GeoDataFrame,
    *,
    east_col: str = "east",
    north_col: str = "north",
    max_snap_m: float = 2000.0,
    exclude: gpd.GeoSeries | None = None,
) -> pd.DataFrame:
    """Attach municipality, district, canton and language region to LV95 points.

    ``admin_match``: "within" a polygon; "nearest" polygon within ``max_snap_m`` for points in
    lakes, on borders or boundary lines; "none" otherwise, without coordinates or inside
    ``exclude``. Ties are resolved by distance, then by the smallest ``municipality_id``.

    Args:
        df: Listings with LV95 (EPSG:2056) coordinates.
        units: Output of :func:`load_admin_units`.
        east_col: Column with the LV95 east coordinate.
        north_col: Column with the LV95 north coordinate.
        max_snap_m: Maximum snapping distance in metres; ``0`` disables snapping.
        exclude: Polygons whose interior points are never snapped, normally
            :func:`load_foreign_mask` (points in Liechtenstein stay "none").

    Returns:
        Copy of ``df`` (same index and order) with ``ADMIN_COLUMNS``; ids as nullable ``Int64``.

    Raises:
        KeyError: If coordinate or unit columns are missing.
        ValueError: If ``max_snap_m`` is negative.
    """
    if max_snap_m < 0:
        raise ValueError(f"max_snap_m must be >= 0, got {max_snap_m}")
    units = _prepare_units(units)
    coords = df[[east_col, north_col]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    pos = np.flatnonzero(np.isfinite(coords).all(axis=1))
    xy = gpd.points_from_xy(coords[pos, 0], coords[pos, 1])
    points = gpd.GeoDataFrame({"_pos": pos}, geometry=xy, crs=units.crs)
    within = _spatial_join(points, units, None)
    outside = _drop_masked(points.loc[~points["_pos"].isin(within.index)], exclude)
    nearest = _spatial_join(outside, units, max_snap_m)
    labelled = [(within, "within"), (nearest, "nearest")]
    frames = [frame.assign(admin_match=label) for frame, label in labelled if not frame.empty]
    attrs = pd.concat(frames) if frames else pd.DataFrame(columns=list(ADMIN_COLUMNS))
    attrs = attrs.reindex(np.arange(len(df))).astype(_ADMIN_DTYPES)
    attrs["admin_match"] = attrs["admin_match"].fillna("none")
    out = df.drop(columns=[c for c in ADMIN_COLUMNS if c in df.columns])
    for col in ADMIN_COLUMNS:
        out[col] = attrs[col].array  # positional, so duplicated index labels are safe
    logger.info("Admin assignment: %s", out["admin_match"].value_counts().to_dict())
    return out


def _prepare_units(units: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    prepared = _to_lv95(units.copy())
    if prepared.geometry.has_z.any():
        prepared = prepared.set_geometry(prepared.geometry.force_2d())
    return prepared[[*_JOIN_COLUMNS, "geometry"]].reset_index(drop=True)


def _drop_masked(points: gpd.GeoDataFrame, mask: gpd.GeoSeries | None) -> gpd.GeoDataFrame:
    if mask is None or mask.empty or points.empty:
        return points
    polygons = _to_lv95(gpd.GeoDataFrame(geometry=gpd.GeoSeries(mask).reset_index(drop=True)))
    hits = gpd.sjoin(points, polygons, predicate="within").index.unique()
    logger.info("%d points inside excluded (foreign) polygons stay unassigned", len(hits))
    return points.drop(index=hits)


def _spatial_join(
    points: gpd.GeoDataFrame, units: gpd.GeoDataFrame, max_m: float | None
) -> pd.DataFrame:
    """Point-in-polygon join (``max_m=None``) or nearest join; one row per ``_pos``."""
    if points.empty or max_m == 0:
        return pd.DataFrame(columns=_JOIN_COLUMNS)
    if max_m is None:
        joined = gpd.sjoin(points, units, predicate="within").assign(_snap_m=0.0)
    else:
        joined = gpd.sjoin_nearest(points, units, max_distance=max_m, distance_col="_snap_m")
    joined = joined.sort_values(["_pos", "_snap_m", "municipality_id"], kind="stable")
    return pd.DataFrame(joined.drop_duplicates("_pos").set_index("_pos")[_JOIN_COLUMNS])


def language_region(
    canton: pd.Series, district_name: pd.Series, *, municipality_name: pd.Series | None = None
) -> pd.Series:
    """Approximate the language region from canton, district and municipality.

    Canton majority language (fr: GE VD NE JU FR VS; it: TI; else de), corrected by the
    ``*_LANGUAGE_OVERRIDES``. Bilingual towns get one label, Romansh areas count as "de".

    Args:
        canton: Canton abbreviations (e.g. ``"ZH"``).
        district_name: swissBOUNDARIES3D district names, aligned by position with ``canton``.
        municipality_name: Optional municipality names, aligned by position.

    Returns:
        Series ``lang_region`` ("de" | "fr" | "it"; NaN where the canton is missing).

    Raises:
        ValueError: If the inputs differ in length or a canton abbreviation is unknown.
    """
    names = [district_name] if municipality_name is None else [district_name, municipality_name]
    if any(len(s) != len(canton) for s in names):
        raise ValueError("canton, district_name and municipality_name must have equal length")
    abbr = pd.Series(canton.to_numpy(), dtype="string").str.strip().str.upper()
    unknown = sorted(set(abbr.dropna()) - set(CANTON_ABBR.values()))
    if unknown:
        raise ValueError(f"Unknown canton abbreviations: {unknown}")
    lang = abbr.map(_CANTON_LANGUAGE).fillna("de").to_numpy(dtype=object)
    tables = (DISTRICT_LANGUAGE_OVERRIDES, MUNICIPALITY_LANGUAGE_OVERRIDES)
    for series, table in zip(names, tables, strict=False):
        lookup = {f"{c}|{n.strip().casefold()}": v for (c, n), v in table.items()}
        norm = pd.Series(series.to_numpy(), dtype="string").str.strip().str.casefold()
        hit = (abbr + "|" + norm).map(lookup).to_numpy(dtype=object, na_value=None)
        lang = np.where(pd.notna(hit), hit, lang)
    lang[abbr.isna().to_numpy(dtype=bool)] = np.nan
    return pd.Series(lang, index=canton.index, name="lang_region", dtype=object)


def nested_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Build globally unique nested random-effect keys.

    Args:
        df: Frame with ``canton``, ``district_id`` and ``municipality_id``.

    Returns:
        ``re_canton`` ("ZH"), ``re_district`` ("ZH/112"), ``re_municipality`` ("ZH/112/261");
        NaN where a level is missing.

    Raises:
        KeyError: If a hierarchy column is missing.
    """
    canton = df["canton"].astype("string").str.strip()
    district = pd.to_numeric(df["district_id"]).astype("Int64").astype("string")
    muni = pd.to_numeric(df["municipality_id"]).astype("Int64").astype("string")
    keys = pd.DataFrame({"re_canton": canton, "re_district": canton + "/" + district})
    keys["re_municipality"] = keys["re_district"] + "/" + muni
    return keys.astype(object).where(keys.notna(), np.nan)


def listings_per_unit(train_df: pd.DataFrame, col: str = "municipality_id") -> pd.Series:
    """Count training listings per administrative unit.

    Args:
        train_df: Training rows only (counts must not use calibration or test rows).
        col: Unit column, e.g. ``"municipality_id"`` or ``"re_district"``.

    Returns:
        Series ``n_listings`` indexed by unit value (sorted); missing units are ignored.

    Raises:
        KeyError: If ``col`` is not a column of ``train_df``.
    """
    return train_df[col].value_counts(dropna=True).sort_index().rename("n_listings")


def count_bucket(counts: pd.Series, *, low: int = 5, high: int = 20) -> pd.Series:
    """Bucket listing counts into ``"<5"``, ``"5-20"`` and ``">20"``.

    Args:
        counts: Listing count per row, e.g. ``df["municipality_id"].map(per_unit)``; missing
            counts (units unseen in training) count as 0. Sort with ``COUNT_BUCKET_ORDER``.
        low: Counts below ``low`` form the first bucket.
        high: Counts up to and including ``high`` form the middle bucket.

    Returns:
        Series ``count_bucket`` of string labels indexed like ``counts``.

    Raises:
        ValueError: If a count is negative or ``low > high``.
    """
    if low > high:
        raise ValueError(f"low ({low}) must not exceed high ({high})")
    values = pd.to_numeric(counts, errors="coerce").fillna(0).to_numpy(dtype=float)
    if (values < 0).any():
        raise ValueError("counts must be non-negative")
    conditions = [values < low, values <= high]
    labels = np.select(conditions, [f"<{low}", f"{low}-{high}"], default=f">{high}")
    return pd.Series(labels, index=counts.index, name="count_bucket", dtype=object)
