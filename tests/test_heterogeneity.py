"""Tests for heterogeneity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from poolguard import PoolGuard, PoolGuardConfig
from poolguard.heterogeneity import (
    analyze_heterogeneity,
    calculate_i_squared,
    cochran_q_test,
    forest_plot_summary,
    pool_interaction_test,
)
from poolguard.utils import predict_proba_positive


@pytest.fixture
def multi_pool_data() -> tuple[pd.DataFrame, pd.Series[int], pd.Series[str]]:
    rng = np.random.default_rng(7)
    n = 450
    X = pd.DataFrame({"age": rng.normal(60, 10, n), "score": rng.normal(0, 1, n)})
    y = pd.Series((X["age"] > 58).astype(int))
    pools = pd.Series(rng.choice(["site_a", "site_b", "site_c"], n))
    return X, y, pools


@pytest.fixture
def fitted_model(
    multi_pool_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> LogisticRegression:
    X, y, _ = multi_pool_data
    model = LogisticRegression(max_iter=500, random_state=0)
    model.fit(X, y)
    return model


def test_cochran_q_test() -> None:
    q, p, i2, pooled = cochran_q_test(
        {"a": 0.8, "b": 0.7, "c": 0.75},
        {"a": 0.01, "b": 0.01, "c": 0.01},
    )
    assert q >= 0.0
    assert 0.0 <= p <= 1.0
    assert i2 >= 0.0
    assert 0.0 < pooled < 1.0


def test_calculate_i_squared() -> None:
    assert calculate_i_squared(10.0, 2) >= 0.0
    assert calculate_i_squared(0.0, 2) == 0.0


def test_cochran_q_requires_two_pools() -> None:
    with pytest.raises(ValueError, match="At least two pools"):
        cochran_q_test({"a": 0.8}, {"a": 0.01})


def test_forest_plot_summary() -> None:
    entries = forest_plot_summary(
        {"a": 0.8, "b": 0.7},
        {"a": 0.01, "b": 0.01},
    )
    assert len(entries) == 3
    assert entries[-1].pool == "__pooled__"


def test_pool_interaction_test(
    fitted_model: LogisticRegression,
    multi_pool_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> None:
    X, y, pools = multi_pool_data
    scores = predict_proba_positive(fitted_model, X)
    lr, p = pool_interaction_test(scores, y.to_numpy(), pools)
    assert lr >= 0.0
    assert 0.0 <= p <= 1.0


def test_heterogeneity_report(
    fitted_model: LogisticRegression,
    multi_pool_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> None:
    X, y, pools = multi_pool_data
    config = PoolGuardConfig(bootstrap_n=200, random_state=0)
    report = analyze_heterogeneity(fitted_model, X, y, pools, config=config)
    assert report.metric == "auc"
    assert report.forest_plot
    assert report.pooled_estimate == report.forest_plot[-1].estimate
    assert report.summary().startswith("HeterogeneityReport")


def test_poolguard_heterogeneity_workflow(
    fitted_model: LogisticRegression,
    multi_pool_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> None:
    X, y, pools = multi_pool_data
    pg = PoolGuard(model=fitted_model, config=PoolGuardConfig(bootstrap_n=200, random_state=0))
    pg.fit_reference(X, y, pools)
    het = pg.test_heterogeneity(X, y, pools)
    assert het.run_id
    assert pg.audit.verify()
