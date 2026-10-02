"""Candidate generation and attribute matching for :mod:`rentml.dedup` (private helpers).

Everything here works on positional numpy arrays: block codes (``-1`` = no key), LV95
coordinates and the ``rooms`` / ``area`` / ``price`` attributes. Pairs are returned as two
aligned position arrays ``(left, right)``.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

EXACT_AREA_TOL = 0.5  # m²: "identical" living area (portal rounding)
EXACT_PRICE_TOL = 1.0  # CHF: "identical" rent
EMPTY_PAIRS = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

Pairs = tuple[np.ndarray, np.ndarray]


def components(n: int, left: np.ndarray, right: np.ndarray) -> tuple[int, np.ndarray]:
    """Number of connected components and a component label per node (undirected edges)."""
    graph = coo_matrix((np.ones(left.size), (left, right)), shape=(n, n))
    count, labels = connected_components(graph, directed=False)
    return int(count), labels.astype(np.int64)


def star_edges(codes: np.ndarray) -> Pairs:
    """Link every member of a key group to the group's first member (spanning star)."""
    pos = np.flatnonzero(codes >= 0)
    if pos.size < 2:
        return EMPTY_PAIRS
    order = pos[np.argsort(codes[pos], kind="stable")]
    sorted_codes = codes[order]
    first = np.r_[True, sorted_codes[1:] != sorted_codes[:-1]]
    group_root = order[first][np.cumsum(first) - 1]
    return group_root[~first], order[~first]


def window_pairs(codes: np.ndarray, area: np.ndarray, area_tol: float) -> Pairs:
    """Candidate pairs sharing a block code with ``|d area| <= area_tol``.

    Rows are sorted by (block, area); each row is paired with the following rows of its block
    whose area lies within the tolerance (vectorised ``searchsorted`` window). Rooms are not
    used for blocking because they may be missing or truncated (see :func:`attribute_match`).
    """
    pos = np.flatnonzero((codes >= 0) & np.isfinite(area))
    if pos.size < 2:
        return EMPTY_PAIRS
    code, size = codes[pos], area[pos]
    order = np.lexsort((size, code))
    code, size, pos = code[order], size[order], pos[order]
    group = np.cumsum(np.r_[True, code[1:] != code[:-1]]) - 1
    span = float(size.max() - size.min()) + area_tol + 1.0
    key = group * span + (size - size.min())
    upper = np.searchsorted(key, key + area_tol + 1e-6, side="right")
    counts = upper - np.arange(key.size) - 1
    left = np.repeat(np.arange(key.size), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    return pos[left], pos[left + 1 + offsets]


def coord_pairs(east: np.ndarray, north: np.ndarray, radius: float) -> Pairs:
    """All pairs of points at most ``radius`` metres apart (KD-tree, no grid artefacts)."""
    pos = np.flatnonzero(np.isfinite(east) & np.isfinite(north))
    if pos.size < 2 or radius < 0:
        return EMPTY_PAIRS
    tree = cKDTree(np.column_stack([east[pos], north[pos]]))
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    return pos[pairs[:, 0]], pos[pairs[:, 1]]


def rooms_compatible(i: np.ndarray, j: np.ndarray, rooms: np.ndarray) -> np.ndarray:
    """Equal room counts (to 0.1) or at least one unknown room count (wildcard)."""
    with np.errstate(invalid="ignore"):
        same = np.round(rooms[i] * 10) == np.round(rooms[j] * 10)
    return same | np.isnan(rooms[i]) | np.isnan(rooms[j])


def exact_match(i: np.ndarray, j: np.ndarray, attrs: dict[str, np.ndarray]) -> np.ndarray:
    """Identical area and price (up to :data:`EXACT_AREA_TOL` / :data:`EXACT_PRICE_TOL`)."""
    area, price = attrs["area"], attrs["price"]
    with np.errstate(invalid="ignore"):
        area_ok = np.abs(area[i] - area[j]) <= EXACT_AREA_TOL + 1e-9
        return area_ok & (np.abs(price[i] - price[j]) <= EXACT_PRICE_TOL + 1e-6)


def attribute_match(
    i: np.ndarray, j: np.ndarray, attrs: dict[str, np.ndarray], area_tol: float, rel_tol: float
) -> np.ndarray:
    """Tolerance match with compatible rooms, or identical area and price (any rooms).

    DSPRO1 stored rooms as integers (half rooms rounded either way), so identical area and
    price outweigh a room difference; missing values of area or price never match.
    """
    area, price = attrs["area"], attrs["price"]
    with np.errstate(invalid="ignore"):
        area_ok = np.abs(area[i] - area[j]) <= area_tol + 1e-9
        price_max = np.maximum(price[i], price[j])
        price_ok = np.abs(price[i] - price[j]) <= rel_tol * price_max + 1e-6
    tolerant = rooms_compatible(i, j, attrs["rooms"]) & area_ok & price_ok
    return tolerant | exact_match(i, j, attrs)


def building_rule_pairs(
    blocks: dict[str, np.ndarray],
    coords: tuple[np.ndarray, np.ndarray],
    attrs: dict[str, np.ndarray],
    tols: tuple[float, float, float, float | None],
) -> dict[str, Pairs]:
    """Matched pairs for the egid, coords, address and coords_exact rules.

    Args:
        blocks: ``"egid"`` and ``"address"`` block codes (``-1`` = no key).
        coords: LV95 ``(east, north)`` arrays (NaN = unknown).
        attrs: ``"rooms"``, ``"area"`` and ``"price"`` arrays.
        tols: ``(area_tol, price_rel_tol, coord_round_m, exact_radius_m)``; an
            ``exact_radius_m`` of ``None`` disables the ``coords_exact`` rule.

    Returns:
        ``{rule: (left, right)}``. Two known, different egids veto every non-egid rule; the
        address rule only applies if not both listings have coordinates.
    """
    area_tol, price_rel_tol, coord_round_m, exact_radius_m = tols
    has_egid = blocks["egid"] >= 0
    east, north = coords
    has_coords = np.isfinite(east) & np.isfinite(north)
    window = max(area_tol, EXACT_AREA_TOL)  # exact matches must be candidates, too
    candidates = {
        "egid": window_pairs(blocks["egid"], attrs["area"], window),
        "coords": coord_pairs(east, north, coord_round_m),
        "address": window_pairs(blocks["address"], attrs["area"], window),
        "coords_exact": (
            EMPTY_PAIRS if exact_radius_m is None else coord_pairs(east, north, exact_radius_m)
        ),
    }
    result: dict[str, Pairs] = {}
    for rule, (i, j) in candidates.items():
        if rule == "coords_exact":
            keep = exact_match(i, j, attrs) & rooms_compatible(i, j, attrs["rooms"])
        else:
            keep = attribute_match(i, j, attrs, area_tol, price_rel_tol)
        if rule != "egid":
            keep &= ~(has_egid[i] & has_egid[j])
        if rule == "address":
            keep &= ~(has_coords[i] & has_coords[j])
        result[rule] = (i[keep], j[keep])
    return result
