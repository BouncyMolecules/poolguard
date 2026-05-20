"""Tests for transportability analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from poolguard import PoolGuard, PoolGuardConfig
from poolguard.transport import (
    assess_transportability,
    g_computation_transport,
    inverse_probability_weight,
    transportability_score,
)


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


def test_transportability_score_bounds() -> None:
    score = transportability_score(0.1, 0.2)
    assert 0.0 <= score <= 1.0


def test_inverse_probability_weight(
    fitted_model: LogisticRegression,
    multi_pool_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> None:
    X, y, pools = multi_pool_data
    ref = str(pools.value_counts().index[0])
    ipw = inverse_probability_weight(
        fitted_model,
        X,
        y,
        pools,
        reference_pool=ref,
        metric="auc",
    )
    assert ipw
    for value in ipw.values():
        assert 0.0 <= value <= 1.0 or np.isnan(value)


def test_g_computation_transport(
    fitted_model: LogisticRegression,
    multi_pool_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> None:
    X, y, pools = multi_pool_data
    ref = str(pools.value_counts().index[0])
    gcomp = g_computation_transport(
        fitted_model,
        X,
        y,
        pools,
        reference_pool=ref,
        metric="auc",
    )
    assert gcomp
    for value in gcomp.values():
        assert 0.0 <= value <= 1.0 or np.isnan(value)


def test_assess_transportability(
    fitted_model: LogisticRegression,
    multi_pool_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> None:
    X, y, pools = multi_pool_data
    config = PoolGuardConfig(bootstrap_n=200, random_state=0)
    report = assess_transportability(fitted_model, X, y, pools, config=config)
    assert report.reference_pool in pools.unique()
    assert report.ipw_estimates
    assert report.gcomp_estimates
    assert report.summary().startswith("TransportReport")


def test_poolguard_full_workflow(
    fitted_model: LogisticRegression,
    multi_pool_data: tuple[pd.DataFrame, pd.Series[int], pd.Series[str]],
) -> None:
    X, y, pools = multi_pool_data
    config = PoolGuardConfig(bootstrap_n=200, random_state=0)
    pg = PoolGuard(model=fitted_model, config=config)
    pg.fit_reference(X, y, pools)
    transport = pg.assess_transportability(X, y, pools)
    analysis = pg.run_full_analysis(X, y, pools)
    assert transport.run_id
    assert analysis.shift.run_id
    assert analysis.evaluation.run_id
    assert analysis.heterogeneity.run_id
    assert analysis.transport.run_id
    assert pg.audit.verify()
