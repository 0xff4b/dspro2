"""Tests for rentml.dedup."""

import logging

import numpy as np
import pandas as pd
import pytest

from rentml.config import RANDOM_STATE
from rentml.dedup import (
    assign_object_ids,
    collapse_objects,
    dedup_report,
    description_hash,
    exact_duplicate_mask,
    normalize_text,
)

LONG_TEXT = "Helle 3.5-Zimmer-Wohnung mit Balkon und Seesicht, ruhige Lage, Lift vorhanden."


def _listings(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a canonical listings frame; defaults describe one generic apartment."""
    defaults: dict[str, object] = {
        "area": 70.0,
        "rooms": 3.5,
        "price": 2000.0,
        "east": 2_600_000.0,
        "north": 1_200_000.0,
        "egid": np.nan,
        "description": pd.NA,
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    frame.index = pd.Index(frame.pop("listing_id"), name="listing_id")
    return frame


def _components(ids: pd.Series) -> set[frozenset[int]]:
    return {frozenset(group.index) for _, group in ids.groupby(ids)}


def test_normalize_text_folds_case_umlauts_accents_and_punctuation() -> None:
    assert normalize_text("Rütistrasse 1, ZÜRICH!") == "ruetistrasse 1 zuerich"
    assert normalize_text("Café <b>Crème</b>\n\n  Grüße") == "cafe creme gruesse"


@pytest.mark.parametrize("value", [None, np.nan, pd.NA, "", "  -- "])
def test_normalize_text_returns_empty_for_missing_or_blank(value: object) -> None:
    assert normalize_text(value) == ""  # type: ignore[arg-type]


def test_description_hash_matches_formatting_variants_and_is_na_for_empty() -> None:
    texts = pd.Series(["Schöne Wohnung!", "schöne   WOHNUNG", "", None, np.nan, "Andere"])
    hashes = description_hash(texts)
    assert hashes.iloc[0] == hashes.iloc[1]
    assert hashes.iloc[0] != hashes.iloc[5]
    assert hashes.iloc[2:5].isna().all()
    assert hashes.index.equals(texts.index)


def test_description_hash_respects_min_chars() -> None:
    hashes = description_hash(pd.Series(["kurz", LONG_TEXT]), min_chars=30)
    assert pd.isna(hashes.iloc[0])
    assert isinstance(hashes.iloc[1], str) and len(hashes.iloc[1]) == 64


def test_exact_duplicate_mask_flags_second_and_later_occurrences() -> None:
    df = pd.DataFrame({"a": [1, 1, 2, 1, np.nan, np.nan], "b": [5, 5, 5, 6, 7, 7]})
    mask = exact_duplicate_mask(df, ["a", "b"])
    assert mask.tolist() == [False, True, False, False, False, True]


def test_exact_duplicate_mask_rejects_bad_subset() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(KeyError):
        exact_duplicate_mask(df, ["a", "missing"])
    with pytest.raises(ValueError):
        exact_duplicate_mask(df, [])


def test_assign_object_ids_links_chains_transitively() -> None:
    # A~B and B~C are within 3 m², A and C are 5 m² apart: components still merge all three.
    df = _listings(
        [
            {"listing_id": 1, "area": 70.0},
            {"listing_id": 2, "area": 72.5},
            {"listing_id": 3, "area": 75.0},
            {"listing_id": 4, "area": 90.0},
        ]
    )
    ids = assign_object_ids(df)
    assert _components(ids) == {frozenset({1, 2, 3}), frozenset({4})}


def test_assign_object_ids_tolerance_boundaries_are_inclusive() -> None:
    df = _listings(
        [
            {"listing_id": 1, "area": 70.0, "price": 1000.0},
            {"listing_id": 2, "area": 73.0, "price": 1000.0},  # |d area| = 3.0 -> match
            {"listing_id": 3, "area": 70.0, "price": 1000.0, "east": 2_610_000.0},
            {"listing_id": 4, "area": 73.01, "price": 1000.0, "east": 2_610_000.0},  # 3.01
            {"listing_id": 5, "area": 70.0, "price": 900.0, "east": 2_620_000.0},
            {"listing_id": 6, "area": 70.0, "price": 1000.0, "east": 2_620_000.0},  # 10.0 %
            {"listing_id": 7, "area": 70.0, "price": 895.0, "east": 2_630_000.0},
            {"listing_id": 8, "area": 70.0, "price": 1000.0, "east": 2_630_000.0},  # 10.5 %
        ]
    )
    comps = _components(assign_object_ids(df))
    assert frozenset({1, 2}) in comps
    assert frozenset({5, 6}) in comps
    assert {frozenset({3}), frozenset({4}), frozenset({7}), frozenset({8})} <= comps


def test_assign_object_ids_rooms_are_wildcard_if_missing_and_ignored_for_exact_matches() -> None:
    df = _listings(
        [
            {"listing_id": 1, "rooms": 3.5},
            {"listing_id": 2, "rooms": 3.0, "area": 72.0},  # other rooms, not identical
            # Identical rows with unknown rooms (real case 2960/7677).
            {"listing_id": 3, "rooms": np.nan, "east": 2.61e6},
            {"listing_id": 4, "rooms": np.nan, "east": 2.61e6},
            # Identical area and price, rooms 5 vs 6 (integer rooms in DSPRO1; case 139/7641).
            {"listing_id": 5, "rooms": 5.0, "area": 140.0, "price": 4330.0, "egid": 7.0},
            {"listing_id": 6, "rooms": 6.0, "area": 140.0, "price": 4330.0, "egid": 7.0},
            # Unknown rooms within the tolerances.
            {"listing_id": 7, "rooms": np.nan, "area": 71.5, "price": 2080.0, "east": 2.63e6},
            {"listing_id": 8, "rooms": 2.5, "area": 70.0, "price": 2000.0, "east": 2.63e6},
            # Other rooms and area 1 m² apart: neither tolerant nor exact.
            {"listing_id": 9, "rooms": 2.0, "area": 50.0, "east": 2.64e6},
            {"listing_id": 10, "rooms": 3.0, "area": 51.0, "east": 2.64e6},
        ]
    )
    comps = _components(assign_object_ids(df))
    singletons = {frozenset({k}) for k in (1, 2, 9, 10)}
    assert comps == singletons | {frozenset({3, 4}), frozenset({5, 6}), frozenset({7, 8})}


def test_assign_object_ids_egid_block_beats_coordinates() -> None:
    df = _listings(
        [
            # Same building id, geocodes 50 m apart -> same object via egid.
            {"listing_id": 1, "egid": 100.0},
            {"listing_id": 2, "egid": 100.0, "east": 2_600_050.0},
            # Same coordinates but different buildings -> egid veto.
            {"listing_id": 3, "egid": 200.0, "east": 2_700_000.0},
            {"listing_id": 4, "egid": 201.0, "east": 2_700_000.0},
            # One egid missing -> coordinate rule applies.
            {"listing_id": 5, "egid": 300.0, "east": 2_800_000.0},
            {"listing_id": 6, "east": 2_800_000.0},
        ]
    )
    ids = assign_object_ids(df)
    assert _components(ids) == {
        frozenset({1, 2}),
        frozenset({3}),
        frozenset({4}),
        frozenset({5, 6}),
    }
    assert ids.attrs["rule_merges"]["egid"] == 1
    assert ids.attrs["rule_merges"]["coords"] == 1


def test_assign_object_ids_coordinate_radius() -> None:
    # Prices differ (within tolerance, not identical) so that only the 5 m coords rule applies.
    df = _listings(
        [
            {"listing_id": 1},
            {"listing_id": 2, "east": 2_600_004.0},  # 4 m away -> match
            {"listing_id": 3, "east": 2_600_020.0, "price": 2040.0},
            {"listing_id": 4, "east": 2_600_026.0, "price": 2080.0},  # 6 m away -> no match
        ]
    )
    comps = _components(assign_object_ids(df, coord_round_m=5.0))
    assert comps == {frozenset({1, 2}), frozenset({3}), frozenset({4})}


def test_assign_object_ids_coords_exact_rule_links_identical_listings_nearby() -> None:
    df = _listings(
        [
            {"listing_id": 1},
            {"listing_id": 2, "east": 2_600_012.6},  # identical, 12.6 m (geocoding jitter)
            {"listing_id": 3, "east": 2_600_020.0, "price": 2100.0},  # not identical
            {"listing_id": 4, "north": 1_200_031.0},  # identical but 31 m away
            {"listing_id": 5, "egid": 1.0, "east": 2_700_000.0},
            {"listing_id": 6, "egid": 2.0, "east": 2_700_010.0},  # identical, other building
        ]
    )
    ids = assign_object_ids(df)
    singletons = {frozenset({k}) for k in (3, 4, 5, 6)}
    assert _components(ids) == singletons | {frozenset({1, 2})}
    assert ids.attrs["rule_merges"]["coords_exact"] == 1
    assert assign_object_ids(df, exact_radius_m=None).nunique() == 6
    assert assign_object_ids(df, exact_radius_m=35.0)[4] == ids[1]
    with pytest.raises(ValueError):
        assign_object_ids(df, exact_radius_m=-1.0)


def test_assign_object_ids_links_duplicate_listing_ids() -> None:
    # Concatenated snapshots: the same listing_id with a price change > 10 %.
    df = _listings(
        [
            {"listing_id": 5, "price": 1000.0},
            {"listing_id": 5, "price": 1300.0},
            {"listing_id": 6, "east": 2.7e6},
        ]
    )
    ids = assign_object_ids(df)
    assert ids.iloc[0] == ids.iloc[1] != ids.iloc[2]
    assert ids.attrs["rule_merges"]["listing_id"] == 1
    assert ids.attrs["rule_rows"]["listing_id"] == 2


def test_assign_object_ids_description_rule_and_na_descriptions() -> None:
    boiler = "Kontaktieren Sie unsere Verwaltung fuer eine Besichtigung dieser Wohnung."
    df = _listings(
        [
            {"listing_id": 1, "description": LONG_TEXT, "area": 50.0},
            {"listing_id": 2, "description": LONG_TEXT.upper(), "area": 99.0, "east": 2.7e6},
            {"listing_id": 3, "description": "Wohnung", "east": 2.71e6},
            {"listing_id": 4, "description": "Wohnung", "east": 2.72e6, "area": 20.0},
            {"listing_id": 5, "east": 2.73e6},
            {"listing_id": 6, "east": 2.74e6, "area": 30.0},
            *[
                {"listing_id": 10 + k, "description": boiler, "east": 2.8e6 + 100 * k}
                for k in range(4)
            ],
        ]
    )
    ids = assign_object_ids(df, desc_max_group=3)
    assert ids[1] == ids[2]  # long identical description (case-insensitive)
    assert ids[3] != ids[4]  # too short to be evidence
    assert ids[5] != ids[6]  # NA descriptions never link
    assert ids.loc[[10, 11, 12, 13]].nunique() == 4  # boilerplate group > desc_max_group
    assert assign_object_ids(df, desc_max_group=None).loc[[10, 11, 12, 13]].nunique() == 1


def test_assign_object_ids_slug_rule_is_source_qualified() -> None:
    df = _listings(
        [
            {"listing_id": 1, "slug": "abc-123", "source": "rentumo", "area": 40.0},
            {"listing_id": 2, "slug": "ABC-123 ", "source": "rentumo", "east": 2.7e6},
            {"listing_id": 3, "slug": "abc-123", "source": "other", "east": 2.8e6},
            {"listing_id": 4, "slug": "", "source": "rentumo", "east": 2.9e6},
            {"listing_id": 5, "slug": "", "source": "rentumo", "east": 3.0e6},
        ]
    )
    comps = _components(assign_object_ids(df))
    assert comps == {frozenset({1, 2}), frozenset({3}), frozenset({4}), frozenset({5})}


def test_assign_object_ids_address_fallback_without_coordinates() -> None:
    df = _listings(
        [
            {"listing_id": 1, "address": "8044, Zürich, Rütistrasse, 1"},
            {"listing_id": 2, "address": "Ruetistr. 3, 8044 Zürich"},
            {"listing_id": 3, "address": "8044, Zürich"},  # no street -> no address key
            {"listing_id": 4, "address": "8044, Zürich, Rütistrasse, 9", "east": 2.7e6},
        ]
    ).assign(east=[np.nan, np.nan, np.nan, 2.7e6], north=[np.nan, np.nan, np.nan, 1.2e6])
    comps = _components(assign_object_ids(df))
    # Rows 1, 2 and 4 share the address key; 4 has coordinates but 1/2 do not -> rule applies.
    assert comps == {frozenset({1, 2, 4}), frozenset({3})}


def test_assign_object_ids_ignores_address_when_both_have_coordinates() -> None:
    df = _listings(
        [
            {"listing_id": 1, "address": "Bahnhofstrasse 1, 8001 Zürich"},
            {"listing_id": 2, "address": "Bahnhofstrasse 90, 8001 Zürich", "east": 2.6005e6},
        ]
    )
    assert assign_object_ids(df).nunique() == 2


def test_assign_object_ids_is_deterministic_and_ordered_by_listing_id() -> None:
    df = _listings(
        [
            {"listing_id": 30, "area": 80.0},
            {"listing_id": 10, "area": 81.0},
            {"listing_id": 20, "area": 40.0, "rooms": 1.5},
        ]
    )
    ids = assign_object_ids(df)
    assert ids[10] == ids[30] == "obj_000001"
    assert ids[20] == "obj_000002"
    shuffled = df.sample(frac=1.0, random_state=RANDOM_STATE)
    pd.testing.assert_series_equal(assign_object_ids(shuffled).sort_index(), ids.sort_index())


def test_assign_object_ids_handles_empty_and_rejects_negative_tolerance() -> None:
    empty = _listings([{"listing_id": 1}]).iloc[0:0]
    assert assign_object_ids(empty).empty
    with pytest.raises(ValueError):
        assign_object_ids(empty, area_tol=-1.0)


def test_dedup_report_counts_objects_rules_and_exact_duplicates() -> None:
    df = _listings(
        [
            {"listing_id": 1},
            {"listing_id": 2},
            {"listing_id": 3, "price": 2100.0},
            {"listing_id": 4, "east": 2.7e6},
        ]
    )
    ids = assign_object_ids(df)
    report = dedup_report(df, ids, exact_subsets={"all_attrs": ["area", "rooms", "price"]})
    value = report["value"]
    assert report.index.name == "metric"
    assert value["rows"] == 4
    assert value["unique_objects"] == 2
    assert value["duplicate_rows"] == 2
    assert value["max_group_size"] == 3
    assert value["objects_size_3"] == 1
    assert value["exact_dup_rows_all_attrs"] == 2
    assert value["merges_coords"] == 2
    assert value["median_rel_price_range_dup_objects"] == pytest.approx(100 / 2100)


def test_dedup_report_exact_duplicates_never_span_objects() -> None:
    df = _listings(
        [
            {"listing_id": 2960, "rooms": np.nan, "area": 112.0, "price": 1830.0},
            {"listing_id": 7677, "rooms": np.nan, "area": 112.0, "price": 1830.0},
            {"listing_id": 139, "rooms": 5.0, "area": 140.0, "price": 4330.0, "east": 2.5e6},
            {"listing_id": 7641, "rooms": 6.0, "area": 140.0, "price": 4330.0, "east": 2.5e6},
        ]
    )
    ids = assign_object_ids(df)
    value = dedup_report(df, ids)["value"]  # default subset (east, north, area, rooms, price)
    assert value["exact_dup_rows_coords_attrs"] == 1
    assert value["exact_dup_groups_split_coords_attrs"] == 0
    custom = dedup_report(df, ids, exact_subsets={"no_rooms": ["east", "north", "area", "price"]})
    assert custom["value"]["exact_dup_rows_no_rooms"] == 2
    assert custom["value"]["exact_dup_groups_split_no_rooms"] == 0


def test_dedup_report_detects_exact_duplicates_split_over_objects() -> None:
    # Without coordinates, egid or address there is no building key: identical attributes
    # alone are no evidence, but the audit must show the split group.
    df = _listings([{"listing_id": 1}, {"listing_id": 2}]).assign(east=np.nan, north=np.nan)
    ids = assign_object_ids(df)
    value = dedup_report(df, ids)["value"]
    assert ids.nunique() == 2
    assert value["exact_dup_groups_split_coords_attrs"] == 1
    no_default = dedup_report(df.drop(columns="rooms"), ids)["value"]
    assert not any(m.startswith("exact_dup") for m in no_default.index)


def test_dedup_report_omits_stale_rule_counts(caplog: pytest.LogCaptureFixture) -> None:
    df = _listings(
        [
            {"listing_id": 1},
            {"listing_id": 2},
            {"listing_id": 3, "east": 2.7e6},
            {"listing_id": 4, "east": 2.8e6},
        ]
    )
    ids = assign_object_ids(df)
    full = dedup_report(df, ids)["value"]
    assert full["merges_coords"] == 1 and full["rows_coords"] == 2
    reordered = dedup_report(df.iloc[::-1], ids.iloc[::-1])["value"]  # same rows -> still valid
    assert reordered["merges_coords"] == 1
    subset = [3, 4]
    with caplog.at_level(logging.WARNING, logger="rentml.dedup"):
        report = dedup_report(df.loc[subset], ids.loc[subset])
    assert report["value"]["duplicate_rows"] == 0
    assert not any(m.startswith(("rows_", "merges_")) for m in report.index)
    assert "per-rule counts" in caplog.text
    as_column = df.assign(object_id=ids)["object_id"]  # attrs do not survive a column
    assert not any(m.startswith("merges_") for m in dedup_report(df, as_column).index)


def test_dedup_report_rejects_misaligned_ids() -> None:
    df = _listings([{"listing_id": 1}, {"listing_id": 2}])
    ids = assign_object_ids(df)
    with pytest.raises(ValueError):
        dedup_report(df, ids.iloc[:1])
    with pytest.raises(ValueError):
        dedup_report(df, ids.set_axis([7, 8]))


def test_collapse_objects_keeps_representative_with_smallest_listing_id() -> None:
    df = _listings(
        [
            {"listing_id": 5, "price": 2100.0},
            {"listing_id": 2, "price": 2000.0},
            {"listing_id": 9, "price": 1950.0},
            {"listing_id": 7, "area": 30.0, "rooms": 1.0},
        ]
    )
    ids = assign_object_ids(df)
    collapsed = collapse_objects(df, ids)
    assert collapsed.index.tolist() == [2, 7]
    assert collapsed["n_listings"].tolist() == [3, 1]
    assert collapsed.loc[2, "price"] == 2000.0
    assert collapsed["object_id"].is_unique
    medians = collapse_objects(df, ids, how="median")
    assert medians.loc[2, "price"] == 2000.0
    assert medians.loc[7, "area"] == 30.0


def test_collapse_objects_median_keeps_dtypes_ids_and_log_price() -> None:
    df = _listings(
        [
            {"listing_id": 1, "price": 2000.0},
            {"listing_id": 2, "price": 2100.0},
            {"listing_id": 3, "price": 2150.0, "area": 71.0},
            {"listing_id": 4, "price": 2200.0},
            {"listing_id": 9, "area": 30.0, "rooms": 1.0, "price": 900.0},
        ]
    )
    df["egid"] = pd.array([100, 100, 100, 100, 200], dtype="Int64")
    # One object geocoded on a municipality border: the median id 261.5 must never appear.
    df["municipality_id"] = pd.array([261, 261, 262, 262, 230], dtype="Int64")
    df["rooms_is_integer"] = False
    df["log_price"] = np.log(df["price"])
    ids = assign_object_ids(df)
    first = collapse_objects(df, ids)
    median = collapse_objects(df, ids, how="median")
    pd.testing.assert_series_equal(median.dtypes, first.dtypes)
    assert median.loc[1, "price"] == 2125.0
    np.testing.assert_allclose(median["log_price"], np.log(median["price"]))
    assert median["municipality_id"].tolist() == [261, 230]
    assert median["egid"].tolist() == [100, 200]
    assert median.loc[1, "area"] == 70.0  # not a median column by default
    both = collapse_objects(df, ids, how="median", median_cols=("price", "area"))
    assert both.loc[1, "area"] == 70.0 and both.loc[9, "area"] == 30.0


def test_collapse_objects_median_rejects_bad_columns() -> None:
    df = _listings([{"listing_id": 1}, {"listing_id": 2}]).assign(flag=True, text="a")
    ids = assign_object_ids(df)
    with pytest.raises(KeyError):
        collapse_objects(df, ids, how="median", median_cols=("missing",))
    with pytest.raises(ValueError):
        collapse_objects(df, ids, how="median", median_cols=("text",))
    with pytest.raises(ValueError):
        collapse_objects(df, ids, how="median", median_cols=("flag",))


def test_collapse_objects_rejects_unknown_how() -> None:
    df = _listings([{"listing_id": 1}])
    with pytest.raises(ValueError):
        collapse_objects(df, assign_object_ids(df), how="mean")
