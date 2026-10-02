"""Tests for rentml.rentcheck (fake models; one smoke test with the real quantile/CQR stack)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from rentml.config import QUANTILE_LEVELS, RANDOM_STATE
from rentml.conformal import CQRCalibrator, MondrianCQR, assign_band, fit_band_cutpoints
from rentml.features import add_engineered_features
from rentml.quantile import MonotoneQuantileLGBM
from rentml.rentcheck import (
    RentCheckResult,
    RentModelBundle,
    fair_rent_check,
    predict_frame,
    verdict,
    what_if,
)

SIGMA = 0.2


class FakePoint:
    """log(rent) = log(500 + 20 * area)."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.log(500.0 + 20.0 * X["area"].to_numpy(dtype=float))


class FakeQuantile:
    """Normal quantiles around the fake point prediction (returned unsorted on purpose)."""

    levels_ = np.asarray(QUANTILE_LEVELS)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        mu = FakePoint().predict(X)
        q = mu[:, None] + SIGMA * norm.ppf(self.levels_)[None, :]
        return q[:, ::-1]


class FakeGroupCalibrator:
    """Widens by 0.05 log units and reports the region as cell."""

    alpha = 0.2

    def predict(
        self, q_lo: np.ndarray, q_hi: np.ndarray, groups: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, pd.Series]:
        cell = "lang_region=" + groups["lang_region"].astype(str)
        if "band" in groups:
            cell = cell + "|band=" + groups["band"].astype(str)
        return q_lo - 0.05, q_hi + 0.05, cell


class FakeMarginalCalibrator:
    alpha = 0.2

    def predict(self, q_lo: np.ndarray, q_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return q_lo - 0.1, q_hi + 0.1


def _bundle(calibrator: object = None, **kwargs: object) -> RentModelBundle:
    defaults: dict[str, object] = {"group_cols": ["lang_region"], "metadata": {"version": 1}}
    defaults.update(kwargs)
    return RentModelBundle(
        feature_cols=["area", "rooms"],
        point_model=FakePoint(),
        quantile_model=FakeQuantile(),
        calibrator=FakeGroupCalibrator() if calibrator is None else calibrator,
        **defaults,
    )


def _row(area: float = 80.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_row = pd.DataFrame({"area": [area], "rooms": [3.5], "extra": ["x"]}, index=[123])
    return x_row, pd.DataFrame({"lang_region": ["de"]}, index=[123])


def test_verdict_classifies_asking_rent() -> None:
    assert verdict(900.0, 1000.0, 2000.0) == "below"
    assert verdict(1000.0, 1000.0, 2000.0) == "within"
    assert verdict(2500.0, 1000.0, 2000.0) == "above"


def test_verdict_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        verdict(float("nan"), 1.0, 2.0)
    with pytest.raises(ValueError):
        verdict(1.5, 2.0, 1.0)


def test_predict_frame_uses_interval_levels_and_calibrator() -> None:
    X = pd.DataFrame({"area": [50.0, 80.0, 120.0], "rooms": [2.0, 3.5, 4.5]}, index=[7, 8, 9])
    groups = pd.DataFrame({"lang_region": ["fr", "de", "it"]}, index=[9, 8, 7])  # reordered
    frame = predict_frame(_bundle(), X, groups)
    mu = np.log(500 + 20 * X["area"].to_numpy())
    z = norm.ppf(0.9)
    assert list(frame.index) == [7, 8, 9]
    np.testing.assert_allclose(frame["expected"], np.exp(mu))
    np.testing.assert_allclose(frame["lo"], np.exp(mu - SIGMA * z - 0.05))
    np.testing.assert_allclose(frame["hi"], np.exp(mu + SIGMA * z + 0.05))
    np.testing.assert_allclose(frame["q25"], np.exp(mu + SIGMA * norm.ppf(0.25)))
    np.testing.assert_allclose(frame["q75"], np.exp(mu + SIGMA * norm.ppf(0.75)))
    assert list(frame["cell"]) == ["lang_region=it", "lang_region=de", "lang_region=fr"]


def test_predict_frame_assigns_bands_from_cutpoints() -> None:
    X = pd.DataFrame({"area": [30.0, 60.0, 90.0, 150.0], "rooms": [1.0, 2.0, 3.0, 5.0]})
    cuts = fit_band_cutpoints(np.log(500 + 20 * np.array([30.0, 60.0, 90.0, 150.0])))
    groups = pd.DataFrame({"lang_region": ["de"] * 4})
    frame = predict_frame(_bundle(band_cutpoints=cuts), X, groups)
    assert list(frame["band"]) == ["B1", "B2", "B3", "B4"]
    assert frame["cell"].str.endswith(tuple(frame["band"])).all()


def test_predict_frame_with_marginal_or_no_calibrator() -> None:
    X = pd.DataFrame({"area": [80.0], "rooms": [3.0]})
    marginal = predict_frame(_bundle(FakeMarginalCalibrator(), group_cols=[]), X)
    raw = RentModelBundle(["area", "rooms"], FakePoint(), FakeQuantile(), None)
    uncalibrated = predict_frame(raw, X)
    assert "cell" not in marginal.columns
    np.testing.assert_allclose(marginal["lo"], uncalibrated["lo"] * np.exp(-0.1))
    np.testing.assert_allclose(marginal["hi"], uncalibrated["hi"] * np.exp(0.1))


def test_predict_frame_rejects_missing_columns_and_levels() -> None:
    X = pd.DataFrame({"area": [80.0]})
    with pytest.raises(ValueError, match="feature columns"):
        predict_frame(_bundle(), X, pd.DataFrame({"lang_region": ["de"]}))
    X = pd.DataFrame({"area": [80.0], "rooms": [3.0]})
    with pytest.raises(ValueError, match="groups"):
        predict_frame(_bundle(), X, None)
    wide = _bundle(FakeGroupCalibrator())
    wide.calibrator.alpha = 0.5  # needs levels 0.25/0.75 -> present
    assert np.isfinite(predict_frame(wide, X, pd.DataFrame({"lang_region": ["de"]}))["lo"]).all()
    odd = _bundle(FakeMarginalCalibrator(), group_cols=[])
    odd.calibrator.alpha = 0.3  # needs 0.15/0.85 -> not in the grid
    with pytest.raises(ValueError, match="not in the quantile grid"):
        predict_frame(odd, X)


@pytest.mark.parametrize(
    ("asking", "expected"), [(1000.0, "below"), (2100.0, "within"), (4000.0, "above")]
)
def test_fair_rent_check_verdicts(asking: float, expected: str) -> None:
    x_row, groups_row = _row(80.0)
    drivers = pd.DataFrame({"feature": ["area"], "shap_log": [0.1]})
    result = fair_rent_check(_bundle(), x_row, groups_row, asking, drivers=drivers)
    assert isinstance(result, RentCheckResult)
    assert result.verdict == expected
    assert result.expected_chf == pytest.approx(2100.0)
    assert (
        result.lo_chf < result.band25_chf < result.expected_chf < result.band75_chf < result.hi_chf
    )
    assert result.coverage == pytest.approx(0.8)
    assert result.cell == "lang_region=de"
    assert result.drivers is drivers
    assert expected in result.summary()


def test_fair_rent_check_market_percentile() -> None:
    x_row, groups_row = _row(80.0)
    at_median = fair_rent_check(_bundle(), x_row, groups_row, 2100.0)
    assert at_median.market_percentile == pytest.approx(0.5)
    at_q75 = fair_rent_check(_bundle(), x_row, groups_row, 2100.0 * np.exp(SIGMA * norm.ppf(0.75)))
    assert at_q75.market_percentile == pytest.approx(0.75)
    extreme = fair_rent_check(_bundle(), x_row, groups_row, 100_000.0)
    assert extreme.market_percentile == pytest.approx(0.99)


def test_fair_rent_check_rejects_bad_input() -> None:
    x_row, groups_row = _row()
    with pytest.raises(ValueError, match="positive"):
        fair_rent_check(_bundle(), x_row, groups_row, 0.0)
    with pytest.raises(ValueError, match="exactly one row"):
        fair_rent_check(_bundle(), pd.concat([x_row, x_row]), groups_row, 2000.0)


def test_what_if_applies_each_change_to_a_copy() -> None:
    x_row, groups_row = _row(80.0)
    table = what_if(_bundle(), x_row, groups_row, {"area": [60.0, 100.0], "rooms": [2.5]})
    assert list(table["feature"]) == ["baseline", "area", "area", "rooms"]
    assert table.loc[0, "delta_chf"] == 0.0
    np.testing.assert_allclose(table["expected_chf"], [2100.0, 1700.0, 2500.0, 2100.0])
    np.testing.assert_allclose(table["delta_pct"], 100 * (table["expected_chf"] / 2100.0 - 1))
    assert table.loc[1, "original"] == 80.0 and table.loc[1, "value"] == 60.0
    assert (table["lo_chf"] < table["expected_chf"]).all()
    assert x_row.loc[123, "area"] == 80.0  # input untouched


def test_what_if_transform_recomputes_derived_features() -> None:
    x_row, _ = _row(80.0)
    bundle = RentModelBundle(["area"], FakePoint(), FakeQuantile(), None)

    def double_area(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(area=frame["area"] * 2)

    table = what_if(bundle, x_row, None, {"area": [50.0]}, transform=double_area)
    np.testing.assert_allclose(table["expected_chf"], [500 + 20 * 160, 500 + 20 * 100])


def test_what_if_rejects_invalid_changes() -> None:
    x_row, groups_row = _row()
    with pytest.raises(ValueError, match="at least one"):
        what_if(_bundle(), x_row, groups_row, {})
    with pytest.raises(ValueError, match="not in x_row"):
        what_if(_bundle(), x_row, groups_row, {"balcony": [1.0]})
    with pytest.raises(ValueError, match="No values"):
        what_if(_bundle(), x_row, groups_row, {"area": []})


def test_bundle_save_load_roundtrip(tmp_path: Path) -> None:
    bundle = _bundle(band_cutpoints=np.array([7.0, 7.5, 8.0]))
    path = bundle.save(tmp_path / "nested" / "bundle.joblib")
    restored = RentModelBundle.load(path)
    x_row, groups_row = _row()
    pd.testing.assert_frame_equal(
        predict_frame(restored, x_row, groups_row), predict_frame(bundle, x_row, groups_row)
    )
    assert restored.metadata == {"version": 1}


def test_bundle_load_rejects_other_objects(tmp_path: Path) -> None:
    path = tmp_path / "other.joblib"
    joblib.dump({"not": "a bundle"}, path)
    with pytest.raises(TypeError):
        RentModelBundle.load(path)


def test_real_quantile_and_mondrian_stack_end_to_end(tmp_path: Path) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    n = 1600
    area = rng.uniform(30, 150, n)
    region = rng.choice(["de", "fr"], n)
    y = np.log(500 + 20 * area) + rng.normal(0, np.where(region == "de", 0.1, 0.2))
    X = pd.DataFrame({"area": area, "rooms": np.round(area / 25) + 0.5})
    groups = pd.DataFrame({"lang_region": region})
    train, calib = np.arange(1000), np.arange(1000, n)
    qmodel = MonotoneQuantileLGBM(monotone=[1, 0]).fit(X.iloc[train], y[train])
    q_cal = qmodel.predict(X.iloc[calib])
    cuts = fit_band_cutpoints(q_cal[:, 3], n_bands=2)
    g_cal = groups.iloc[calib].assign(band=assign_band(q_cal[:, 3], cuts))
    mondrian = MondrianCQR(0.2, min_group_size=150).fit(q_cal[:, 1], q_cal[:, 5], y[calib], g_cal)
    bundle = RentModelBundle(
        ["area", "rooms"], FakePoint(), qmodel, mondrian, cuts, {}, ["lang_region"]
    )
    restored = RentModelBundle.load(bundle.save(tmp_path / "b.joblib"))
    frame = predict_frame(restored, X.iloc[calib], groups.iloc[calib])
    # a negative qhat may shrink the 80 % interval inside the raw 25-75 band, so only check order
    assert (frame["lo"] < frame["hi"]).all() and (frame["q25"] <= frame["q75"]).all()
    assert frame["cell"].notna().all()
    # calibrated bounds may jump between bands; the uncalibrated band is monotone in area
    grid = what_if(restored, X.iloc[[0]], groups.iloc[[0]], {"area": [40.0, 80.0, 120.0]})
    assert np.all(np.diff(grid["q25_chf"].to_numpy()[1:]) >= 0)
    assert np.all(np.diff(grid["q75_chf"].to_numpy()[1:]) >= 0)
    marginal = RentModelBundle(
        ["area", "rooms"],
        FakePoint(),
        qmodel,
        CQRCalibrator().fit(q_cal[:, 1], q_cal[:, 5], y[calib]),
    )
    frame_m = predict_frame(marginal, X.iloc[calib])
    assert "cell" not in frame_m.columns
    assert (frame_m["lo"] < frame_m["hi"]).all()


class LevelledCalibrator(FakeGroupCalibrator):
    """Calibrated on the 5 % / 95 % quantiles although alpha is 0.2."""

    interval_levels_ = (0.05, 0.95)


class MarginalWithOption:
    """Marginal calibrator whose third parameter is not ``groups``."""

    alpha = 0.2

    def predict(
        self, q_lo: np.ndarray, q_hi: np.ndarray, clip: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        assert clip is None, "groups must not be passed positionally"
        return q_lo - 0.1, q_hi + 0.1


class OrderSensitivePoint:
    """Fitted on columns [rooms, area] and reads them by position like LightGBM does."""

    feature_names_in_ = np.array(["rooms", "area"])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.log(500.0 + 20.0 * X.iloc[:, 1].to_numpy(dtype=float))


class IntegerRoomsPoint:
    """Adds CHF 100 for whole room counts."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        flag = X["rooms_is_integer"].to_numpy(dtype=float)
        return np.log(500.0 + 20.0 * X["area"].to_numpy(dtype=float) + 100.0 * flag)


def test_predict_frame_uses_recorded_interval_levels() -> None:
    X, groups = _row(80.0)
    mu = np.log(2100.0)
    frame = predict_frame(_bundle(LevelledCalibrator()), X, groups)
    np.testing.assert_allclose(frame["lo"], np.exp(mu + SIGMA * norm.ppf(0.05) - 0.05))
    np.testing.assert_allclose(frame["hi"], np.exp(mu + SIGMA * norm.ppf(0.95) + 0.05))
    from_meta = _bundle(metadata={"interval_levels": (0.05, 0.95)})
    pd.testing.assert_frame_equal(predict_frame(from_meta, X, groups), frame)
    assert fair_rent_check(_bundle(LevelledCalibrator()), X, groups, 2100.0).coverage == 0.8


def test_predict_frame_rejects_conflicting_or_missing_interval_levels() -> None:
    X, groups = _row(80.0)
    conflict = _bundle(LevelledCalibrator(), metadata={"interval_levels": (0.1, 0.9)})
    with pytest.raises(ValueError, match="interval_levels"):
        predict_frame(conflict, X, groups)
    off_grid = _bundle(metadata={"interval_levels": (0.15, 0.85)})
    with pytest.raises(ValueError, match="not in the quantile grid"):
        predict_frame(off_grid, X, groups)


def test_calibrator_dispatch_uses_groups_parameter_not_arity() -> None:
    X = pd.DataFrame({"area": [80.0], "rooms": [3.0]})
    frame = predict_frame(_bundle(MarginalWithOption(), group_cols=[]), X)
    reference = predict_frame(_bundle(FakeMarginalCalibrator(), group_cols=[]), X)
    pd.testing.assert_frame_equal(frame, reference)


def test_groups_must_share_labels_or_have_default_index() -> None:
    X = pd.DataFrame({"area": [50.0, 80.0, 120.0], "rooms": [2.0, 3.5, 4.5]}, index=[7, 8, 9])
    fresh = pd.DataFrame({"lang_region": ["fr", "de", "it"]})  # RangeIndex: row by row
    frame = predict_frame(_bundle(), X, fresh)
    assert list(frame["cell"]) == ["lang_region=fr", "lang_region=de", "lang_region=it"]
    partial = fresh.set_axis([8, 9, 10])
    with pytest.raises(ValueError, match="2 of 3 shared"):
        predict_frame(_bundle(), X, partial)
    with pytest.raises(ValueError, match="reset_index"):
        predict_frame(_bundle(), X, fresh.set_axis([1, 2, 3]))


def test_point_model_input_follows_its_feature_names() -> None:
    X = pd.DataFrame({"area": [50.0, 120.0], "rooms": [2.0, 4.5]})
    bundle = _bundle(FakeMarginalCalibrator(), group_cols=[])
    bundle.point_model = OrderSensitivePoint()
    np.testing.assert_allclose(predict_frame(bundle, X)["expected"], 500 + 20 * X["area"])


def test_what_if_recomputes_rooms_is_integer_with_feature_transform() -> None:
    x_row = pd.DataFrame(
        {"area": [80.0], "rooms": [3.5], "rooms_is_integer": [False], "area_per_room": [80 / 3.5]}
    )
    bundle = RentModelBundle(
        ["area", "rooms", "area_per_room", "rooms_is_integer"],
        IntegerRoomsPoint(),
        FakeQuantile(),
        None,
    )
    changes = {"rooms": [3.0, 4.5]}
    table = what_if(bundle, x_row, None, changes, transform=add_engineered_features)
    np.testing.assert_allclose(table["expected_chf"], [2100.0, 2200.0, 2100.0])
    stale = what_if(
        bundle, x_row, None, changes, transform=add_engineered_features, derived_cols=()
    )
    np.testing.assert_allclose(stale["expected_chf"], [2100.0, 2100.0, 2100.0])
    assert not x_row.loc[0, "rooms_is_integer"]  # input untouched
    explicit = what_if(
        bundle, x_row, None, {"rooms_is_integer": [1.0]}, transform=add_engineered_features
    )
    np.testing.assert_allclose(explicit["expected_chf"], [2100.0, 2200.0])


def test_real_stack_with_reordered_features_and_asymmetric_mondrian() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    n = 1600
    area = rng.uniform(30, 150, n)
    region = rng.choice(["de", "fr"], n)
    y = np.log(500 + 20 * area) + rng.normal(0, np.where(region == "de", 0.1, 0.2))
    X = pd.DataFrame({"area": area, "rooms": np.round(area / 25) + 0.5})
    groups = pd.DataFrame({"lang_region": region})
    train, calib = np.arange(1000), np.arange(1000, n)
    qmodel = MonotoneQuantileLGBM(monotone=[1, 0]).fit(X.iloc[train], y[train])
    q_cal = qmodel.predict(X.iloc[calib])
    mondrian = MondrianCQR(0.2, min_group_size=150, hierarchy=(("lang_region",),), symmetric=False)
    mondrian.fit(
        q_cal[:, 0], q_cal[:, 6], y[calib], groups.iloc[calib], interval_levels=(0.05, 0.95)
    )
    bundle = RentModelBundle(
        ["rooms", "area"], FakePoint(), qmodel, mondrian, None, {}, ["lang_region"]
    )
    frame = predict_frame(bundle, X.iloc[calib], groups.iloc[calib])
    cell_lo = frame["cell"].map(mondrian.qhat_lo_).to_numpy()
    np.testing.assert_allclose(np.log(frame["lo"]), q_cal[:, 0] - cell_lo)
    covered = (y[calib] >= np.log(frame["lo"])) & (y[calib] <= np.log(frame["hi"]))
    assert covered.mean() >= 0.8
