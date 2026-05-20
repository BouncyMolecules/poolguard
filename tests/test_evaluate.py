"""Tests for per-pool model evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from poolguard.config import PoolGuardConfig
from poolguard.core import PoolGuard
from poolguard.evaluate import evaluate_pools


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


def test_evaluate_pools_returns_metrics(
    fitted_model: LogisticRegression,
    classification_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> None:
    X, y, pools = classification_data
    config = PoolGuardConfig(bootstrap_n=200, random_state=0)
    report = evaluate_pools(fitted_model, X, y, pools, config=config)
    assert "accuracy" in report.overall.metrics
    assert "auc" in report.overall.metrics
    assert len(report.by_pool) == 3
    for pool_result in report.by_pool:
        assert pool_result.n > 0
        lower, upper = pool_result.ci["accuracy"]
        assert lower <= pool_result.metrics["accuracy"] <= upper or lower <= upper


def test_poolguard_evaluate_with_audit(
    fitted_model: LogisticRegression,
    classification_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> None:
    X, y, pools = classification_data
    pg = PoolGuard(model=fitted_model, config=PoolGuardConfig(bootstrap_n=200))
    pg.fit_reference(X, y, pools)
    report = pg.evaluate(X, y, pools)
    assert report.run_id
    assert len(pg.audit) >= 2
    assert pg.audit.verify()


def test_evaluate_requires_model(
    classification_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> None:
    X, y, pools = classification_data
    pg = PoolGuard()
    with pytest.raises(ValueError, match="fitted model"):
        pg.evaluate(X, y, pools)
