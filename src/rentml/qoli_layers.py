"""Point indicators of the Quality-of-Life Index (QoLI) from official geodata, in LV95.

- public transport: ARE quality classes (A-D) and distance to the nearest classified stop;
- noise: BAFU sonBASE road/rail L_r,Tag / L_r,Nacht (façade proxy: the loudest cell within a
  small square, because building footprints carry no value), energetic sum, L_den approximation;
- green: BFS land-use statistics (share of green and wooded hectares within a radius);
- air and sunshine: any LV95 raster sampled at the cell of the point;
- reference grid: inhabited hectares (BFS STATPOP) with BFS distances to services.

All coordinates are LV95 (EPSG:2056) metres; NaN coordinates give NaN indicators.
"""

import logging
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.windows import Window
from scipy import ndimage
from scipy.spatial import cKDTree

from rentml.amenities import accessibility, amenity_arrays

logger = logging.getLogger(__name__)

# ARE public-transport quality classes: A (very good) ... D (low); outside all classes = 0.
OEV_CLASS_SCORE: dict[str, int] = {"A": 4, "B": 3, "C": 2, "D": 1}
OEV_CLASS_LAYER = "OeV_Gueteklassen_ARE"
OEV_STOP_LAYER = "OeV_Haltestellen_ARE"

# BFS land-use statistics, 17 basic categories (AS_17): 5 recreational and green areas,
# 10 forest, 11 brush forest, 12 woods. Water, farmland and alpine pasture are not counted.
GREEN_AS17: frozenset[int] = frozenset({5, 10, 11, 12})
GREEN_RADIUS_M = 300.0  # WHO Europe: green space within 300 m of home (~5 min walk)

# sonBASE: buildings and areas far from any source carry no value, while modelled cells go down
# to about 0 dB. A point takes the value of its own cell; inside a building, the loudest cell of
# the smallest ring that reaches valid cells (the façade); without any valid cell within the
# largest ring it counts as "quiet" (level NaN, best noise score). Taking the maximum of a fixed
# 20 m square instead often picked the road surface and tripled the share above the LSV limits
# compared with the BAFU national statistics.
NOISE_RING_RADII_M: tuple[float, ...] = (0.0, 10.0, 20.0, 50.0, 100.0)
NOISE_FACADE_MAX_M = 20.0  # rings up to this radius count as "facade", larger as "fallback"
# WHO Environmental Noise Guidelines for the European Region (2018), L_den.
WHO_LDEN_DB: dict[str, float] = {"road": 53.0, "rail": 54.0}

# BFS hectare grids give the south-west corner of each 100 m cell.
HECTARE_OFFSET_M = 50.0

# --- raster sampling -------------------------------------------------------------------------


def focal_max(
    path: Path,
    xy: np.ndarray,
    *,
    radii_m: Sequence[float] = (0.0,),
    strip_rows: int = 1024,
) -> np.ndarray:
    """Maximum of the valid raster cells in squares of half-width ``radius`` around points.

    A radius of 0 gives the value of the cell containing the point. The raster is read once, in
    horizontal strips (with a halo) that only span the columns of the points they contain, so
    even 10 m rasters of Switzerland are processed without loading them whole.

    Args:
        path: GeoTIFF in LV95 (any rasterio-readable raster with a nodata value).
        xy: Array (n, 2) of LV95 east/north; NaN rows give NaN.
        radii_m: Half-widths of the square windows in metres (rounded up to whole cells).
        strip_rows: Raster rows per strip (memory vs. number of reads).

    Returns:
        Float array (n, len(radii_m)): the maximum per radius, NaN if the window has no valid
        cell or the point lies outside the raster.

    Raises:
        ValueError: For an empty or negative radius list or wrongly shaped ``xy``.
    """
    xy = _as_xy(xy)
    if not len(radii_m) or min(radii_m) < 0:
        raise ValueError(f"radii_m must be non-empty and >= 0, got {radii_m}")
    out = np.full((len(xy), len(radii_m)), np.nan)
    with rasterio.open(path) as src:
        ks = [int(np.ceil(r / abs(src.res[0]) - 1e-9)) for r in radii_m]
        halo = max(ks)
        finite = np.isfinite(xy).all(axis=1)
        cols = np.full(len(xy), -1, dtype=np.int64)
        rows = np.full(len(xy), -1, dtype=np.int64)
        if finite.any():
            r_f, c_f = rowcol(src.transform, xy[finite, 0], xy[finite, 1], op=np.floor)
            rows[finite], cols[finite] = np.asarray(r_f, np.int64), np.asarray(c_f, np.int64)
        inside = finite & (cols >= 0) & (cols < src.width) & (rows >= 0) & (rows < src.height)
        nodata = src.nodata
        for start in range(0, src.height, strip_rows):
            sel = np.flatnonzero(inside & (rows >= start) & (rows < start + strip_rows))
            if sel.size == 0:
                continue
            r0, r1 = max(start - halo, 0), min(start + strip_rows + halo, src.height)
            c0 = max(int(cols[sel].min()) - halo, 0)
            c1 = min(int(cols[sel].max()) + halo + 1, src.width)
            block = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), out_dtype="float32")
            invalid = ~np.isfinite(block)
            if nodata is not None:
                invalid |= block == np.float32(nodata)
            block[invalid] = -np.inf
            for j, k in enumerate(ks):
                filtered = (
                    block
                    if k == 0
                    else ndimage.maximum_filter(
                        block, size=2 * k + 1, mode="constant", cval=-np.inf
                    )
                )
                values = filtered[rows[sel] - r0, cols[sel] - c0].astype(float)
                out[sel, j] = np.where(np.isfinite(values), values, np.nan)
    return out


def raster_values(path: Path, xy: np.ndarray) -> np.ndarray:
    """Value of the raster cell containing each point (NaN for nodata or outside).

    Args:
        path: GeoTIFF in LV95.
        xy: Array (n, 2) of LV95 east/north.

    Returns:
        Float array (n,).
    """
    return focal_max(path, xy, radii_m=(0.0,))[:, 0]


def noise_levels(
    path: Path, xy: np.ndarray, *, radii_m: Sequence[float] = NOISE_RING_RADII_M
) -> pd.DataFrame:
    """Noise level of a sonBASE raster at points, with a façade search inside buildings.

    Modelled, not measured. The point's own cell is used if it carries a value; otherwise the
    loudest cell of the smallest square ring (``radii_m``, increasing) that contains valid cells,
    i.e. the most exposed façade next to the building footprint. Without any valid cell within
    the largest radius the point is "quiet": no modelled level nearby.

    Args:
        path: sonBASE GeoTIFF (L_r,Tag or L_r,Nacht).
        xy: Array (n, 2) of LV95 east/north.
        radii_m: Strictly increasing search radii in metres (0 = the point's own cell).

    Returns:
        DataFrame with ``level_db`` (NaN if quiet or outside), ``radius_m`` (radius used) and
        ``how``: "point" (own cell), "facade" (ring <= ``NOISE_FACADE_MAX_M``), "fallback"
        (larger ring), "quiet", or "outside" (outside the raster or without coordinates).

    Raises:
        ValueError: If ``radii_m`` is empty or not strictly increasing.
    """
    radii = np.asarray(radii_m, dtype=float)
    if radii.size == 0 or np.any(np.diff(radii) <= 0):
        raise ValueError(f"radii_m must be non-empty and strictly increasing, got {radii_m}")
    xy = _as_xy(xy)
    levels = focal_max(path, xy, radii_m=tuple(radii))
    found = np.isfinite(levels)
    first = np.where(found.any(axis=1), found.argmax(axis=1), -1)
    level = np.where(first >= 0, levels[np.arange(len(xy)), np.maximum(first, 0)], np.nan)
    used = np.where(first >= 0, radii[np.maximum(first, 0)], np.nan)
    how = np.select(
        [first < 0, used == 0, used <= NOISE_FACADE_MAX_M], ["quiet", "point", "facade"], "fallback"
    )
    how = np.where(_inside_raster(path, xy), how, "outside")
    return pd.DataFrame({"level_db": level, "radius_m": used, "how": how})


def apply_quiet_noise(norm: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Give quiet locations (no modelled road or rail level nearby) the best noise score.

    Args:
        norm: Normalised indicators (0-100) with ``noise_day_db``/``noise_night_db`` columns.
        raw: Raw indicators from ``QoliLayers.indicators`` with ``noise_quiet_<period>``.

    Returns:
        Copy of ``norm`` with 100 for quiet locations (same index and attrs).
    """
    out = norm.copy()
    for period in ("day", "night"):
        col, flag = f"noise_{period}_db", f"noise_quiet_{period}"
        if col in out.columns and flag in raw.columns:
            quiet = raw[flag].reindex(out.index).fillna(False).to_numpy(dtype=bool)
            out.loc[quiet, col] = 100.0
    return out


def energetic_sum(*levels_db: np.ndarray | pd.Series) -> np.ndarray:
    """Energetic sum of sound levels: 10·log10(Σ 10^(L/10)); NaN only if all inputs are NaN.

    Used to combine road and rail into one exposure for the index. The LSV limits are assessed
    per source type and never on this sum.

    Args:
        *levels_db: Level arrays of equal length in dB.

    Returns:
        Float array of combined levels.

    Raises:
        ValueError: Without inputs or for unequal lengths.
    """
    if not levels_db:
        raise ValueError("Need at least one level array")
    stack = np.vstack([np.asarray(v, dtype=float) for v in levels_db])
    with np.errstate(divide="ignore"):
        energy = np.nansum(10.0 ** (stack / 10.0), axis=0)
        total = 10.0 * np.log10(energy)
    return np.where(np.isnan(stack).all(axis=0), np.nan, total)


def lden_from_lr(day_db: np.ndarray | pd.Series, night_db: np.ndarray | pd.Series) -> np.ndarray:
    """Approximate L_den from the Swiss rating levels L_r,Tag (06-22 h) and L_r,Nacht (22-06 h).

    L_den = 10·log10[(12·10^(Ld/10) + 4·10^((Le+5)/10) + 8·10^((Ln+10)/10)) / 24] with the
    assumption L_d = L_e = L_r,Tag and L_n = L_r,Nacht. Two simplifications: the Swiss rating
    level contains level corrections K1-K3 (for roads K1 <= 0, so L_r can be below L_eq), and the
    periods differ (L_den: 07-19/19-23/23-07 h). The result is therefore only an approximation
    for a descriptive comparison with the WHO guideline, never a legal assessment.

    Args:
        day_db: L_r,Tag in dB(A).
        night_db: L_r,Nacht in dB(A), same length.

    Returns:
        Approximate L_den in dB (NaN where an input is NaN).
    """
    ld = np.asarray(day_db, dtype=float)
    ln = np.asarray(night_db, dtype=float)
    if ld.shape != ln.shape:
        raise ValueError("day_db and night_db must have the same shape")
    energy = 12 * 10 ** (ld / 10) + 4 * 10 ** ((ld + 5) / 10) + 8 * 10 ** ((ln + 10) / 10)
    return 10.0 * np.log10(energy / 24.0)


# --- public transport, land use, hectare grid ------------------------------------------------


def load_oev(gpkg_zip: Path) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Read the ARE quality-class polygons and the classified stops from the zipped GeoPackage.

    Args:
        gpkg_zip: ``gueteklassen_oev_<year>_2056.gpkg.zip``.

    Returns:
        Tuple of the class polygons (``klasse`` A-D, EPSG:2056) and the stops with a stop
        category 1-5 (``east``, ``north``, ``stop_category``).
    """
    with zipfile.ZipFile(gpkg_zip) as archive:
        member = next(m for m in archive.namelist() if m.lower().endswith(".gpkg"))
        target = gpkg_zip.with_name(Path(member).name)
        if not target.is_file():
            target.write_bytes(archive.read(member))
    classes = gpd.read_file(target, layer=OEV_CLASS_LAYER, columns=["KLASSE"])
    classes = classes.rename(columns={"KLASSE": "klasse"}).to_crs(2056)
    stops = gpd.read_file(target, layer=OEV_STOP_LAYER, columns=["Hst_Kat"]).to_crs(2056)
    stops = stops.loc[stops["Hst_Kat"].between(1, 5)]
    stop_table = pd.DataFrame(
        {
            "east": stops.geometry.x.to_numpy(),
            "north": stops.geometry.y.to_numpy(),
            "stop_category": stops["Hst_Kat"].astype(int).to_numpy(),
        }
    )
    return classes, stop_table


def oev_indicators(xy: np.ndarray, classes: gpd.GeoDataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """Public-transport quality class (0-4) and distance to the nearest classified stop.

    Args:
        xy: Array (n, 2) of LV95 east/north.
        classes: Class polygons from ``load_oev`` (overlaps resolved by the best class).
        stops: Classified stops from ``load_oev``.

    Returns:
        DataFrame with ``oev_class`` (A=4 ... D=1, 0 outside all classes) and ``oev_stop_m``;
        NaN for points without coordinates.
    """
    xy = _as_xy(xy)
    finite = np.isfinite(xy).all(axis=1)
    pos = np.flatnonzero(finite)
    points = gpd.GeoDataFrame(
        {"_pos": pos}, geometry=gpd.points_from_xy(xy[pos, 0], xy[pos, 1]), crs=2056
    )
    scored = classes.assign(_score=classes["klasse"].map(OEV_CLASS_SCORE))
    hits = gpd.sjoin(points, scored[["_score", "geometry"]], predicate="within", how="inner")
    best = hits.groupby("_pos")["_score"].max()
    oev_class = np.full(len(xy), np.nan)
    oev_class[pos] = 0.0
    oev_class[best.index.to_numpy()] = best.to_numpy(dtype=float)
    dist = np.full(len(xy), np.nan)
    if len(stops):
        dist[pos], _ = cKDTree(stops[["east", "north"]].to_numpy(float)).query(xy[pos])
    return pd.DataFrame({"oev_class": oev_class, "oev_stop_m": dist})


def load_landuse(csv_zip: Path) -> pd.DataFrame:
    """Hectare points of the land-use statistics with their AS_17 category.

    Args:
        csv_zip: ``arealstatistik_2056.csv.zip``.

    Returns:
        DataFrame with ``east``/``north`` (hectare centres) and ``as17``.
    """
    raw = pd.read_csv(csv_zip, sep=";", usecols=["E_COORD", "N_COORD", "AS_17"])
    return pd.DataFrame(
        {
            "east": raw["E_COORD"].to_numpy(float) + HECTARE_OFFSET_M,
            "north": raw["N_COORD"].to_numpy(float) + HECTARE_OFFSET_M,
            "as17": raw["AS_17"].astype(int).to_numpy(),
        }
    )


def green_share(
    xy: np.ndarray,
    landuse: pd.DataFrame,
    *,
    radius_m: float = GREEN_RADIUS_M,
    green_classes: Iterable[int] = GREEN_AS17,
) -> np.ndarray:
    """Share of land-use hectares within ``radius_m`` that are green or wooded.

    Args:
        xy: Array (n, 2) of LV95 east/north.
        landuse: Output of ``load_landuse``.
        radius_m: Search radius (hectare centres within the circle count).
        green_classes: AS_17 categories counted as green.

    Returns:
        Float array (n,) in [0, 1]; NaN without coordinates or without any hectare nearby.
    """
    xy = _as_xy(xy)
    finite = np.isfinite(xy).all(axis=1)
    coords = landuse[["east", "north"]].to_numpy(float)
    is_green = landuse["as17"].isin(list(green_classes)).to_numpy()
    out = np.full(len(xy), np.nan)
    if not finite.any():
        return out
    n_all = cKDTree(coords).query_ball_point(xy[finite], radius_m, return_length=True)
    if is_green.any():
        n_green = cKDTree(coords[is_green]).query_ball_point(
            xy[finite], radius_m, return_length=True
        )
    else:
        n_green = np.zeros(len(n_all))
    with np.errstate(invalid="ignore", divide="ignore"):
        out[finite] = np.where(n_all > 0, n_green / n_all, np.nan)
    return out


HECTARE_SERVICE_COLUMNS = [
    "D_GROCERY",
    "D_PHARMA",
    "D_MEDIC",
    "D_SCHOOL_O",
    "D_RESTO",
    "D_STOP_TOT",
]


def load_hectares(csv_path: Path, *, year: int = 2021) -> pd.DataFrame:
    """Inhabited hectares (STATPOP) with BFS distances to services for one year.

    Args:
        csv_path: ``erreichbarkeit_2056.csv`` (semicolon-separated, all years stacked).
        year: Reference year of population and distances.

    Returns:
        DataFrame indexed by ``reli`` with ``east``/``north`` (hectare centres),
        ``population`` and ``HECTARE_SERVICE_COLUMNS`` in metres.

    Raises:
        ValueError: If the year is not in the file.
    """
    cols = ["RELI", "E_COORD", "N_COORD", "YEAR", "POP_TOTAL", *HECTARE_SERVICE_COLUMNS]
    raw = pd.read_csv(csv_path, sep=";", usecols=cols)
    raw = raw.loc[raw["YEAR"].eq(year)]
    if raw.empty:
        raise ValueError(f"No hectares for year {year} in {csv_path.name}")
    out = pd.DataFrame(
        {
            "east": raw["E_COORD"].to_numpy(float) + HECTARE_OFFSET_M,
            "north": raw["N_COORD"].to_numpy(float) + HECTARE_OFFSET_M,
            "population": raw["POP_TOTAL"].to_numpy(float),
        },
        index=pd.Index(raw["RELI"].astype("int64").to_numpy(), name="reli"),
    )
    for col in HECTARE_SERVICE_COLUMNS:
        out[col.lower()] = raw[col].to_numpy(float)
    return out


NOISE_KEYS = ("road_day", "road_night", "rail_day", "rail_night")
RASTER_KEYS = ("no2", "pm25", "sunshine")
LAYER_KEYS = ("oev", *NOISE_KEYS, "landuse", *RASTER_KEYS)


@dataclass
class QoliLayers:
    """Loaded QoLI sources that turn LV95 points into raw (not yet normalised) indicators.

    Attributes:
        paths: Source key -> local file (``LAYER_KEYS``; see ``rentml.qoli_sources.SOURCES``).
        oev_classes: ARE quality-class polygons.
        oev_stops: Classified public-transport stops.
        landuse: Land-use hectare points.
        amenities: OSM category -> LV95 coordinates (empty dict = no accessibility columns).
    """

    paths: Mapping[str, Path]
    oev_classes: gpd.GeoDataFrame
    oev_stops: pd.DataFrame
    landuse: pd.DataFrame
    amenities: dict[str, np.ndarray]

    @classmethod
    def load(cls, paths: Mapping[str, Path], osm: pd.DataFrame | None = None) -> "QoliLayers":
        """Read the vector and tabular sources once (rasters are read lazily per call).

        Args:
            paths: Source key -> local file; must contain every key of ``LAYER_KEYS``.
            osm: OSM amenity table (``rentml.amenities.fetch_osm_amenities``) or ``None``.

        Returns:
            The loaded layers.

        Raises:
            KeyError: If a source path is missing.
        """
        if missing := [k for k in LAYER_KEYS if k not in paths]:
            raise KeyError(f"QoLI source paths missing: {missing}")
        classes, stops = load_oev(paths["oev"])
        amenities = amenity_arrays(osm) if osm is not None else {}
        return cls(dict(paths), classes, stops, load_landuse(paths["landuse"]), amenities)

    def indicators(self, points: pd.DataFrame) -> pd.DataFrame:
        """Raw indicators at LV95 points.

        Columns: ``oev_class``, ``oev_stop_m``; ``<road|rail>_<day|night>_db`` and
        ``noise_<day|night>_db`` (energetic road + rail sum, NaN if quiet), ``noise_quiet_<day|
        night>`` (no modelled road or rail level within the fallback radius) and ``noise_how``
        (road, day);
        ``green_share``; ``acc_<category>`` (OSM), ``acc_total_all`` (all categories) and
        ``acc_total`` (without parks, for indicator sets with a separate park indicator);
        ``no2``, ``pm25``, ``sunshine``.

        Args:
            points: DataFrame with ``east``/``north`` (LV95); its index is kept.

        Returns:
            DataFrame of raw indicators, one row per point.
        """
        xy = points[["east", "north"]].to_numpy(dtype=float, na_value=np.nan)
        out = oev_indicators(xy, self.oev_classes, self.oev_stops)
        quiet = {}
        for key in NOISE_KEYS:
            noise = noise_levels(self.paths[key], xy)
            out[f"{key}_db"] = noise["level_db"].to_numpy()
            quiet[key] = noise["how"].eq("quiet").to_numpy()
            if key == "road_day":
                out["noise_how"] = noise["how"].to_numpy()
        for period in ("day", "night"):
            road, rail = f"road_{period}", f"rail_{period}"
            out[f"noise_{period}_db"] = energetic_sum(out[f"{road}_db"], out[f"{rail}_db"])
            # Outside the rail raster counts as "no rail noise", not as unknown.
            out[f"noise_quiet_{period}"] = quiet[road] & np.isnan(out[f"{rail}_db"].to_numpy())
        out["green_share"] = green_share(xy, self.landuse)
        if self.amenities:
            frame = pd.DataFrame(xy, columns=["east", "north"])
            access = accessibility(frame, self.amenities)  # all categories, incl. parks
            out = out.join(access.rename(columns={"acc_total": "acc_total_all"}))
            no_park = {k: v for k, v in self.amenities.items() if k != "park"}
            out["acc_total"] = accessibility(frame, no_park)["acc_total"]
        for key in RASTER_KEYS:
            out[key] = raster_values(self.paths[key], xy)
        out.index = points.index
        return out


def _inside_raster(path: Path, xy: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as src:
        b = src.bounds
    with np.errstate(invalid="ignore"):
        return (
            (xy[:, 0] >= b.left)
            & (xy[:, 0] < b.right)
            & (xy[:, 1] > b.bottom)
            & (xy[:, 1] <= b.top)
        )


def _as_xy(xy: np.ndarray | pd.DataFrame) -> np.ndarray:
    arr = np.asarray(xy, dtype=float)
    if arr.size and (arr.ndim != 2 or arr.shape[1] != 2):
        raise ValueError(f"Coordinates must have shape (n, 2) (LV95 east/north), got {arr.shape}")
    return arr.reshape(-1, 2)
