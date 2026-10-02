"""Tests for rentml._gpboost_cov (stage 1 of the two-stage GPBoost fit)."""

import numpy as np
import pytest

from rentml._gpboost_cov import expected_names, initial_values, linear_design, uses_aux_error


@pytest.mark.parametrize(
    ("n_groups", "use_gp", "approx", "names"),
    [
        (3, True, "vecchia", ("Group_1", "Group_2", "Group_3", "GP_var", "GP_range")),
        (2, False, "vecchia", ("Error_var", "Group_1", "Group_2")),
        (0, True, "vecchia", ("Error_var", "GP_var", "GP_range")),
        (1, True, "none", ("Error_var", "Group_1", "GP_var", "GP_range")),
    ],
)
def test_expected_names_follow_gpboost_order(
    n_groups: int, use_gp: bool, approx: str, names: tuple[str, ...]
) -> None:
    assert expected_names(n_groups, use_gp, approx) == names
    assert uses_aux_error(n_groups, use_gp, approx) == ("Error_var" not in names)


def test_linear_design_drops_collinear_columns_and_keeps_intercept() -> None:
    rng = np.random.default_rng(42)
    year = rng.integers(1900, 2020, 50).astype(float)
    x = np.column_stack([year, 2026.0 - year, rng.normal(size=50), np.full(50, 3.0)])
    x[::7, 2] = np.nan
    design = linear_design(x)
    assert design.shape == (50, 3)  # intercept + one of year/age + noise column
    np.testing.assert_allclose(design[:, 0], 1.0)
    assert np.linalg.matrix_rank(design) == design.shape[1]
    assert np.isfinite(design).all()


def test_linear_design_all_nan_column_is_dropped() -> None:
    x = np.column_stack([np.full(10, np.nan), np.arange(10.0)])
    design = linear_design(x)
    assert design.shape == (10, 2)


def test_initial_values_scale_with_residual_variance() -> None:
    rng = np.random.default_rng(0)
    design = np.column_stack([np.ones(200), rng.normal(size=200)])
    y = 7.0 + 0.5 * design[:, 1] + rng.normal(0.0, 0.2, 200)
    xy = rng.uniform(0.0, 100.0, (200, 2))
    cov, aux = initial_values(y, design, xy, n_groups=3, aux=True)
    assert cov.shape == (5,) and aux is not None and aux.shape == (1,)
    assert aux[0] == pytest.approx(0.5 * 0.04, rel=0.3)
    assert cov[-1] == pytest.approx(0.03 * np.ptp(xy, axis=0).max())
    cov2, aux2 = initial_values(y, design, None, n_groups=2, aux=False)
    assert aux2 is None and cov2.shape == (3,)
    assert cov2[0] > cov2[1]  # error variance first, then the group variances
