"""Tests for population shift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poolguard.config import PoolGuardConfig
from poolguard.shift import (
    detect_shift,
    kolmogorov_smirnov_test,
    maximum_mean_discrepancy,
    standardized_mean_difference,
)


@pytest.fixture
def shift_data() -> tuple[pd.DataFrame, pd.Series[str]]:
    rng = np.random.default_rng(0)
    n = 200
    site_a = pd.DataFrame({"age": rng.normal(60, 5, n // 2), "score": rng.normal(0, 1, n // 2)})
    site_b = pd.DataFrame({"age": rng.normal(70, 5, n // 2), "score": rng.normal(1, 1, n // 2)})
    X = pd.concat([site_a, site_b], ignore_index=True)
    pools = pd.Series(["site_a"] * (n // 2) + ["site_b"] * (n // 2))
    return X, pools


def test_standardized_mean_difference_detects_shift() -> None:
    ref = np.array([0.0, 0.1, 0.2, 0.0, -0.1])
    shifted = np.array([2.0, 2.1, 1.9, 2.2, 2.0])
    smd = standardized_mean_difference(ref, shifted)
    assert abs(smd) > 1.0


def test_ks_test_returns_valid_statistics() -> None:
    ref = np.linspace(0, 1, 50)
    cmp = np.linspace(0.5, 1.5, 50)
    stat, pvalue = kolmogorov_smirnov_test(ref, cmp)
    assert 0.0 <= stat <= 1.0
    assert 0.0 <= pvalue <= 1.0


def test_mmd_is_non_negative() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, (30, 2))
    cmp = rng.normal(0.5, 1, (30, 2))
    mmd = maximum_mean_discrepancy(ref, cmp)
    assert mmd >= 0.0


def test_detect_shift_flags_shifted_pool(shift_data: tuple[pd.DataFrame, pd.Series[str]]) -> None:
    X, pools = shift_data
    config = PoolGuardConfig(smd_threshold=0.1, ks_alpha=0.05, bootstrap_n=100)
    report = detect_shift(X, pools, reference_pool="site_a", config=config)
    assert report.reference_pool == "site_a"
    assert "site_b" in report.mmd_scores
    assert len(report.feature_results) > 0
    assert report.summary().startswith("ShiftReport")


def test_detect_shift_requires_two_pools() -> None:
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    pools = pd.Series(["only"] * 3)
    with pytest.raises(ValueError, match="At least two distinct pools"):
        detect_shift(X, pools)


def test_detect_shift_invalid_reference(shift_data: tuple[pd.DataFrame, pd.Series[str]]) -> None:
    X, pools = shift_data
    with pytest.raises(ValueError, match="reference_pool"):
        detect_shift(X, pools, reference_pool="missing")
