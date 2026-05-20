"""Tests for PoolGuard orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from poolguard import PoolGuard, PoolGuardConfig


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series[int], pd.Series[str]]:
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame({"age": rng.normal(60, 10, n), "score": rng.normal(0, 1, n)})
    y = pd.Series((X["age"] > 60).astype(int))
    pools = pd.Series(rng.choice(["site_a", "site_b", "site_c"], n))
    return X, y, pools


@pytest.fixture
def fitted_model(
    classification_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> LogisticRegression:
    X, y, _ = classification_data
    model = LogisticRegression(max_iter=500, random_state=0)
    model.fit(X, y)
    return model


def test_fit_reference_requires_pools(
    classification_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> None:
    X, y, _ = classification_data
    pg = PoolGuard()
    with pytest.raises(ValueError, match="pools must be provided"):
        pg.fit_reference(X, y)


def test_pool_column_in_dataframe(
    fitted_model: LogisticRegression,
    classification_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> None:
    X, y, pools = classification_data
    df = X.copy()
    df["site"] = pools
    config = PoolGuardConfig(bootstrap_n=200, audit_enabled=True)
    pg = PoolGuard(model=fitted_model, pool_column="site", config=config)
    ref = pg.fit_reference(df, y, reference_pool="site_a")
    assert ref == "site_a"
    shift = pg.detect_shift(df)
    assert shift.reference_pool == "site_a"
    assert len(pg.audit) >= 2


def test_run_full_analysis_requires_model(
    classification_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> None:
    X, y, pools = classification_data
    pg = PoolGuard()
    pg.fit_reference(X, y, pools)
    with pytest.raises(ValueError, match="fitted model"):
        pg.run_full_analysis(X, y, pools)


def test_detect_shift_without_reference_still_runs(
    fitted_model: LogisticRegression,
    classification_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> None:
    X, _y, pools = classification_data
    pg = PoolGuard(model=fitted_model, config=PoolGuardConfig(bootstrap_n=200, audit_enabled=False))
    report = pg.detect_shift(X, pools, reference_pool="site_b")
    assert report.reference_pool == "site_b"
