"""Map bundle for the Dash app: build, save and load (``uv run python -m rentml.mapbundle``).

The bundle holds only what the public map may show: simplified LV95 boundaries per level
(:mod:`rentml.mapdata`) and aggregates per unit, with medians only above ``MIN_MAP_CELL``
objects. No listing, price or address is stored.
"""

import argparse
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

from rentml import geo
from rentml.cleaning import domain_filter, price_per_sqm_outliers, regime_split
from rentml.config import MIN_MAP_CELL, SB3D_VERSION, SNAPSHOT_DATE, ProjectPaths, load_env
from rentml.data import fix_schema, load_listings
from rentml.dedup import assign_object_ids, collapse_objects
from rentml.mapdata import (
    DEFAULT_TOLERANCE_M,
    LEVELS,
    LV95_ORIGIN,
    PLOT_SCALE_M,
    STAT_COLUMNS,
    UNIT_COLUMNS,
    aggregate_listings,
    build_levels,
    load_canton_names,
    load_other_areas,
    to_plot_geojson,
)

logger = logging.getLogger(__name__)


def default_bundle_dir(paths: ProjectPaths) -> Path:
    """Directory of the app's map bundle (``dspro2/data/app/map``, not versioned)."""
    return paths.project_root / "data" / "app" / "map"


def prepare_listings(
    listings: pd.DataFrame, units: gpd.GeoDataFrame, *, foreign: gpd.GeoSeries | None = None
) -> pd.DataFrame:
    """Clean the listings as in notebook sections 2-6 and attach the admin hierarchy.

    Steps: schema fixes, deduplication to one row per object (median rent), assignment to
    municipality/district/canton (points abroad dropped), market-rent regime (if labelled),
    domain rules and the hierarchical CHF/m² outlier filter.

    Args:
        listings: Output of :func:`rentml.data.load_listings`.
        units: Municipalities from :func:`rentml.geo.load_admin_units`.
        foreign: Foreign municipalities (:func:`rentml.geo.load_foreign_mask`).

    Returns:
        One row per object with ``price``, ``area`` and the ``geo.ADMIN_COLUMNS``.
    """
    df = fix_schema(listings)
    objects = collapse_objects(df, assign_object_ids(df), how="median", median_cols=("price",))
    located = geo.assign_admin_units(objects, units, exclude=foreign)
    located = located.loc[located["admin_match"].ne("none")]
    market, _ = regime_split(located)
    kept, _ = domain_filter(market)
    outlier = price_per_sqm_outliers(kept, object_ids=kept["object_id"])
    logger.info("prepare_listings: %d listings -> %d objects on the map", len(df), (~outlier).sum())
    return kept.loc[~outlier].copy()


@dataclass
class MapBundle:
    """Everything the map page needs, without geopandas at runtime.

    Attributes:
        units: One row per unit and level (``UNIT_COLUMNS`` plus statistics), bounds in LV95.
        geojson: Level -> FeatureCollection in plot degrees, feature ``id`` = ``unit_id``.
        other_areas: FeatureCollection of lakes and other non-municipal areas (``kind``).
        meta: Provenance (sources, versions, minimum cell size, Swiss reference values).
    """

    units: pd.DataFrame
    geojson: dict[str, dict]
    other_areas: dict
    meta: dict[str, object] = field(default_factory=dict)

    def save(self, out_dir: Path) -> Path:
        """Write ``units.parquet``, one GeoJSON per level, the other areas and ``meta.json``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.units.to_parquet(out_dir / "units.parquet", index=False)
        for level, collection in self.geojson.items():
            _write_json(out_dir / f"{level}.geojson", collection)
        _write_json(out_dir / "other_areas.geojson", self.other_areas)
        (out_dir / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        logger.info("Saved map bundle to %s", out_dir)
        return out_dir

    @classmethod
    def load(cls, bundle_dir: Path) -> "MapBundle":
        """Read a bundle written by :meth:`save`.

        Raises:
            FileNotFoundError: If the directory or one of its files is missing.
        """
        bundle_dir = Path(bundle_dir)
        if not (bundle_dir / "meta.json").is_file():
            raise FileNotFoundError(
                f"No map bundle in {bundle_dir}; build it with `uv run python -m rentml.mapbundle`"
            )
        geojson = {lvl: _read_json(bundle_dir / f"{lvl}.geojson") for lvl in LEVELS}
        return cls(
            units=pd.read_parquet(bundle_dir / "units.parquet"),
            geojson=geojson,
            other_areas=_read_json(bundle_dir / "other_areas.geojson"),
            meta=json.loads((bundle_dir / "meta.json").read_text(encoding="utf-8")),
        )


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assemble_bundle(
    levels: dict[str, gpd.GeoDataFrame],
    other: gpd.GeoDataFrame,
    stats: dict[str, pd.DataFrame] | None = None,
    *,
    meta: dict[str, object] | None = None,
) -> MapBundle:
    """Join the statistics to the level geometries and convert them for the app.

    Args:
        levels: Output of :func:`build_levels`.
        other: Simplified non-municipal areas from :func:`build_levels`.
        stats: Output of :func:`aggregate_listings`; ``None`` = geometry only (counts 0).
        meta: Provenance written to ``meta.json``.

    Returns:
        The bundle; units without objects get ``n_objects = 0``.
    """
    frames = []
    for level in LEVELS:
        table = pd.DataFrame(levels[level].drop(columns="geometry"))
        level_stats = (stats or {}).get(level, pd.DataFrame(columns=list(STAT_COLUMNS)))
        table = table.join(level_stats, on="unit_id")
        table["n_objects"] = table["n_objects"].fillna(0).astype(int)
        frames.append(table.assign(level=level))
    units = pd.concat(frames, ignore_index=True)
    units = units[[*UNIT_COLUMNS, *[c for c in units.columns if c not in UNIT_COLUMNS]]]
    geojson = {level: to_plot_geojson(levels[level]) for level in LEVELS}
    other_json = to_plot_geojson(other.reset_index(drop=True), id_col=None, props=("name", "kind"))
    return MapBundle(units=units, geojson=geojson, other_areas=other_json, meta=dict(meta or {}))


def build_map_bundle(
    paths: ProjectPaths,
    *,
    source: str = "auto",
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    min_count: int = MIN_MAP_CELL,
) -> MapBundle:
    """Build the bundle from swissBOUNDARIES3D and the listings (database or DSPRO1 CSVs).

    Args:
        paths: Project layout.
        source: Listing source for :func:`rentml.data.load_listings`.
        tolerance_m: Boundary simplification tolerance in metres.
        min_count: Minimum objects per unit for published medians.

    Returns:
        The assembled :class:`MapBundle`.
    """
    load_env(paths)
    gpkg = geo.download_swissboundaries(paths.data_external, SB3D_VERSION)
    units = geo.load_admin_units(gpkg)
    levels, other = build_levels(
        units,
        load_other_areas(gpkg),
        canton_names=load_canton_names(gpkg),
        tolerance_m=tolerance_m,
    )
    listings, used = load_listings(
        source,
        csv_dir=paths.dspro1_csv_dir,
        database_url=os.environ.get("DATABASE_URL") if source != "csv" else None,
        cache_path=paths.data_interim / "listings_db.parquet",
    )
    objects = prepare_listings(listings, units, foreign=geo.load_foreign_mask(gpkg))
    ppsqm = objects["price"] / objects["area"]
    meta = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "boundaries": f"swissBOUNDARIES3D {SB3D_VERSION} (swisstopo, OGD)",
        "listings": f"rentumo.ch snapshot {SNAPSHOT_DATE} ({used})",
        "n_objects": len(objects),
        "min_count": min_count,
        "tolerance_m": tolerance_m,
        "plot_scale_m": PLOT_SCALE_M,
        "lv95_origin": list(LV95_ORIGIN),
        "swiss": {
            "chf_per_m2": float(ppsqm.median()),
            "chf_per_m2_p25": float(ppsqm.quantile(0.25)),
            "chf_per_m2_p75": float(ppsqm.quantile(0.75)),
            "rent": float(objects["price"].median()),
        },
    }
    stats = aggregate_listings(objects, min_count=min_count)
    return assemble_bundle(levels, other, stats, meta=meta)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: build the map bundle for the Dash app."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="auto", choices=["auto", "db", "csv"])
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_M, help="metres")
    parser.add_argument("--min-count", type=int, default=MIN_MAP_CELL)
    parser.add_argument("--out", type=Path, default=None, help="default: data/app/map")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = ProjectPaths.discover()
    bundle = build_map_bundle(
        paths, source=args.source, tolerance_m=args.tolerance, min_count=args.min_count
    )
    out = bundle.save(args.out or default_bundle_dir(paths))
    shown = bundle.units.groupby("level")["chf_per_m2"].count().to_dict()
    logger.info(
        "Map bundle in %s: %d objects; units with a published median: %s",
        out,
        bundle.meta["n_objects"],
        shown,
    )


if __name__ == "__main__":
    main()
