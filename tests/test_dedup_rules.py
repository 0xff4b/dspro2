"""Tests for rentml._dedup_rules (candidate generation and matching behind rentml.dedup)."""

import itertools

import numpy as np
import pandas as pd

from rentml._dedup_rules import (
    EMPTY_PAIRS,
    attribute_match,
    components,
    coord_pairs,
    exact_match,
    rooms_compatible,
    star_edges,
    window_pairs,
)
from rentml.config import RANDOM_STATE
from rentml.dedup import assign_object_ids


def _components(ids: pd.Series) -> set[frozenset[int]]:
    return {frozenset(group.index) for _, group in ids.groupby(ids)}


def _pairs(left: np.ndarray, right: np.ndarray) -> set[tuple[int, int]]:
    return {(min(i, j), max(i, j)) for i, j in zip(left.tolist(), right.tolist(), strict=True)}


def test_window_pairs_links_rows_of_a_block_within_the_area_window() -> None:
    codes = np.array([0, 0, 0, 1, 1, -1, 0])
    area = np.array([70.0, 73.0, 76.5, 70.0, 70.0, 70.0, np.nan])
    assert _pairs(*window_pairs(codes, area, 3.0)) == {(0, 1), (3, 4)}


def test_window_pairs_and_coord_pairs_return_empty_without_candidates() -> None:
    one = np.array([0, -1, -1])
    area = np.array([70.0, 70.0, 70.0])
    assert window_pairs(one, area, 3.0)[0].size == 0
    east, north = np.array([0.0, 1.0, np.nan]), np.array([0.0, 0.0, 0.0])
    assert _pairs(*coord_pairs(east, north, 1.0)) == {(0, 1)}
    assert coord_pairs(east, north, -1.0)[0].size == 0
    assert coord_pairs(east[:1], north[:1], 5.0)[0].size == 0


def test_matching_rules_treat_missing_rooms_as_wildcard_and_missing_area_as_no_match() -> None:
    attrs = {
        "rooms": np.array([3.5, 3.5, np.nan, 3.0, 3.5]),
        "area": np.array([70.0, 72.0, 71.0, 70.0, np.nan]),
        "price": np.array([2000.0, 2100.0, 1950.0, 2000.0, 2000.0]),
    }
    i, j = np.array([0, 0, 0, 1, 0]), np.array([1, 2, 3, 3, 4])
    assert rooms_compatible(i, j, attrs["rooms"]).tolist() == [True, True, False, False, True]
    assert exact_match(i, j, attrs).tolist() == [False, False, True, False, False]
    # (0, 3): rooms differ but area and price are identical -> match.
    assert attribute_match(i, j, attrs, 3.0, 0.10).tolist() == [True, True, True, False, False]


def test_star_edges_and_components() -> None:
    left, right = star_edges(np.array([5, -1, 5, 7, 5, 7]))
    assert _pairs(left, right) == {(0, 2), (0, 4), (3, 5)}
    assert star_edges(np.array([-1, 3]))[0].size == 0
    count, labels = components(6, left, right)
    assert count == 3 and labels[0] == labels[2] == labels[4] and labels[1] != labels[0]
    assert components(2, *EMPTY_PAIRS)[0] == 2


def _brute_force_components(df: pd.DataFrame, area_tol: float, rel_tol: float) -> set:
    """O(n^2) reference implementation of the egid, coords and coords_exact rules."""
    rows = df.reset_index().to_dict("records")
    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    for a, b in itertools.combinations(range(len(rows)), 2):
        ra, rb = rows[a], rows[b]
        both_egid = not (np.isnan(ra["egid"]) or np.isnan(rb["egid"]))
        dist = np.hypot(ra["east"] - rb["east"], ra["north"] - rb["north"])
        same_block = ra["egid"] == rb["egid"] if both_egid else dist <= 5.0
        rooms_ok = ra["rooms"] == rb["rooms"] or np.isnan(ra["rooms"]) or np.isnan(rb["rooms"])
        d_area, d_price = abs(ra["area"] - rb["area"]), abs(ra["price"] - rb["price"])
        exact = d_area <= 0.5 and d_price <= 1.0
        tolerant = (
            rooms_ok and d_area <= area_tol and d_price <= rel_tol * max(ra["price"], rb["price"])
        )
        near_exact = not both_egid and dist <= 30.0 and exact and rooms_ok
        if (same_block and (tolerant or exact)) or near_exact:
            parent[find(a)] = find(b)
    groups: dict[int, set[int]] = {}
    for k, row in enumerate(rows):
        groups.setdefault(find(k), set()).add(int(row["listing_id"]))
    return {frozenset(g) for g in groups.values()}


def test_assign_object_ids_matches_brute_force_reference() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    n = 300
    sites = rng.integers(0, 25, n)
    df = pd.DataFrame(
        {
            "listing_id": rng.permutation(np.arange(1, n + 1)),
            "area": rng.integers(40, 50, n).astype(float),
            "rooms": rng.choice([2.5, 3.5, np.nan], n, p=[0.45, 0.45, 0.1]),
            "price": (rng.integers(15, 19, n) * 100 + rng.integers(0, 3, n)).astype(float),
            "east": 2.6e6 + sites * 25.0 + rng.integers(0, 3, n) * 3.0,
            "north": np.full(n, 1.2e6),
            "egid": np.where(rng.random(n) < 0.5, sites % 7, np.nan),
        }
    ).set_index("listing_id")
    ids = assign_object_ids(df, area_tol=2.0, price_rel_tol=0.05)
    assert _components(ids) == _brute_force_components(df, 2.0, 0.05)
    merges = ids.attrs["rule_merges"]
    assert min(merges["egid"], merges["coords"], merges["coords_exact"]) > 0  # all rules used


def test_candidate_generation_stays_linear_on_large_frames() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    n = 40_000
    df = pd.DataFrame(
        {
            "area": rng.integers(30, 150, n).astype(float),
            "rooms": rng.choice([1.5, 2.5, 3.5, 4.5], n),
            "price": rng.integers(800, 4000, n).astype(float),
            "east": 2.5e6 + rng.integers(0, 20_000, n) * 10.0,
            "north": 1.1e6 + rng.integers(0, 50, n) * 10.0,
            "egid": np.where(rng.random(n) < 0.4, rng.integers(0, 5_000, n), np.nan),
        },
        index=pd.RangeIndex(1, n + 1, name="listing_id"),
    )
    codes = pd.factorize(df["egid"], use_na_sentinel=True)[0].astype(np.int64)
    left, right = window_pairs(codes, df["area"].to_numpy(), 3.0)
    # Reference: self-join on egid, count pairs within the area window.
    blocks = df.reset_index()[["listing_id", "egid", "area"]].dropna()
    joined = blocks.merge(blocks, on="egid")
    within = (joined["listing_id_x"] < joined["listing_id_y"]) & (
        (joined["area_x"] - joined["area_y"]).abs() <= 3.0
    )
    assert left.size == int(within.sum())
    assert (codes[left] == codes[right]).all()
    near_left, _ = coord_pairs(df["east"].to_numpy(), df["north"].to_numpy(), 30.0)
    assert left.size + near_left.size < 10 * n  # far below n^2 / 2 = 8e8 brute-force pairs
    ids = assign_object_ids(df)
    assert ids.notna().all() and len(ids) == n
