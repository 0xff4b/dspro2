"""Unit lookup and the map view state (pure functions, no Dash imports).

A unit is addressed by the key ``"<level>:<unit_id>"`` (e.g. ``"municipality:1061"``). The view
state lives in a ``dcc.Store`` as a plain dict:

* ``level``: level drawn on the map (``"canton"``, ``"district"``, ``"municipality"``),
* ``focus``: unit key the map is zoomed to (``None`` = all of Switzerland),
* ``selected``: highlighted unit key shown in the detail panel (``None`` = Switzerland),
* ``rev``: counter that changes whenever the map should jump to the ``focus`` extent.
"""

from dataclasses import dataclass

import pandas as pd

from rentml.mapdata import LEVELS

LEVEL_LABELS: dict[str, str] = {
    "canton": "Kanton",
    "district": "Bezirk",
    "municipality": "Gemeinde",
}
LEVEL_PLURALS: dict[str, str] = {
    "canton": "Kantone",
    "district": "Bezirke",
    "municipality": "Gemeinden",
}
CHILD_LEVEL: dict[str, str] = {"canton": "district", "district": "municipality"}
DEFAULT_VIEW: dict[str, object] = {
    "level": "canton",
    "focus": None,
    "selected": None,
    "rev": 0,
}


def unit_key(level: str, unit_id: int) -> str:
    """Build the key of a unit, e.g. ``unit_key("canton", 3) == "canton:3"``."""
    if level not in LEVELS:
        raise ValueError(f"Unknown level {level!r}")
    return f"{level}:{int(unit_id)}"


def parse_key(key: str) -> tuple[str, int]:
    """Split a unit key into level and id.

    Raises:
        ValueError: If the key is malformed or the level is unknown.
    """
    level, sep, raw_id = str(key).partition(":")
    if not sep or level not in LEVELS or not raw_id.lstrip("-").isdigit():
        raise ValueError(f"Invalid unit key {key!r}")
    return level, int(raw_id)


@dataclass
class UnitIndex:
    """Fast lookups on the ``units`` table of a :class:`rentml.mapbundle.MapBundle`."""

    units: pd.DataFrame

    def __post_init__(self) -> None:
        pairs = zip(self.units["level"], self.units["unit_id"], strict=True)
        keys = [unit_key(level, unit_id) for level, unit_id in pairs]
        self._table = self.units.set_index(pd.Index(keys, name="key"))
        if not self._table.index.is_unique:
            raise ValueError("Unit ids must be unique within each level")
        swiss = self.units.loc[self.units["level"].eq("canton")]
        self.swiss_bounds = (
            float(swiss["minx"].min()),
            float(swiss["miny"].min()),
            float(swiss["maxx"].max()),
            float(swiss["maxy"].max()),
        )

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._table.index

    def row(self, key: str) -> pd.Series:
        """Table row of a unit (raises ``KeyError`` if unknown)."""
        return self._table.loc[key]

    def level_table(self, level: str) -> pd.DataFrame:
        """All units of one level, indexed by key."""
        return self._table.loc[self._table["level"].eq(level)]

    def parent_key(self, key: str) -> str | None:
        """Key of the next coarser unit (``None`` for cantons)."""
        level, _ = parse_key(key)
        row = self.row(key)
        if level == "municipality":
            return unit_key("district", row["district_id"])
        if level == "district":
            return unit_key("canton", row["canton_id"])
        return None

    def ancestors(self, key: str) -> list[str]:
        """Keys from the canton down to the parent of ``key``."""
        chain: list[str] = []
        parent = self.parent_key(key)
        while parent is not None:
            chain.insert(0, parent)
            parent = self.parent_key(parent)
        return chain

    def children(self, key: str) -> pd.DataFrame:
        """Units one level below ``key`` (empty for municipalities)."""
        level, unit_id = parse_key(key)
        if level not in CHILD_LEVEL:
            return self._table.iloc[0:0]
        table = self.level_table(CHILD_LEVEL[level])
        return table.loc[table[f"{level}_id"].eq(unit_id)]

    def bounds(self, key: str | None) -> tuple[float, float, float, float]:
        """LV95 bounding box of a unit, or of Switzerland for ``None``."""
        if key is None or key not in self:
            return self.swiss_bounds
        row = self.row(key)
        return float(row["minx"]), float(row["miny"]), float(row["maxx"]), float(row["maxy"])

    def label(self, key: str) -> str:
        """Search label such as ``"Luzern · Gemeinde, LU"``."""
        level, _ = parse_key(key)
        row = self.row(key)
        suffix = "" if level == "canton" else f", {row['canton']}"
        return f"{row['name']} · {LEVEL_LABELS[level]}{suffix}"

    def search_options(self) -> list[dict[str, str]]:
        """Dropdown options for all units, cantons first, then alphabetically."""
        order = {level: i for i, level in enumerate(LEVELS)}
        keys = sorted(
            self._table.index,
            key=lambda k: (order[parse_key(k)[0]], str(self._table.at[k, "name"]).casefold()),
        )
        return [{"label": self.label(k), "value": k} for k in keys]


def _next(view: dict, **changes: object) -> dict:
    return {**DEFAULT_VIEW, **view, **changes}


def set_level(view: dict, level: str) -> dict:
    """Draw another level; selection and zoom stay."""
    if level not in LEVELS:
        raise ValueError(f"Unknown level {level!r}")
    return _next(view, level=level)


def select_on_map(view: dict, index: UnitIndex, key: str) -> dict:
    """A click on the map selects the unit without moving the map."""
    return _next(view, selected=key) if key in index else _next(view)


def select_unit(view: dict, index: UnitIndex, key: str) -> dict:
    """Select a unit from search or a link: draw its level, zoom to its parent for context."""
    if key not in index:
        return _next(view)
    level, _ = parse_key(key)
    rev = int(view.get("rev", 0)) + 1
    return _next(view, level=level, selected=key, focus=index.parent_key(key), rev=rev)


def drill_down(view: dict, index: UnitIndex) -> dict:
    """Zoom to the selected unit and draw its subdivisions (municipalities: zoom only)."""
    key = view.get("selected")
    if key not in index:
        return _next(view)
    level, _ = parse_key(key)
    rev = int(view.get("rev", 0)) + 1
    return _next(view, level=CHILD_LEVEL.get(level, level), focus=key, rev=rev)


def reset_view(view: dict) -> dict:
    """Back to all of Switzerland, cantons drawn, nothing selected."""
    return _next(DEFAULT_VIEW, rev=int(view.get("rev", 0)) + 1)
