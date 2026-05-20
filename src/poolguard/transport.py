"""Transportability quantification across patient pools."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from poolguard.config import PoolGuardConfig
from poolguard.evaluate import evaluate_pools
from poolguard.report import TransportReport
from poolguard.shift import detect_shift
from poolguard.utils import (
    compute_binary_metrics,
    ensure_dataframe,
    ensure_series,
    predict_proba_positive,
    utc_now_iso,
    validate_aligned_lengths,
)


def transportability_score(
    performance_gap: float,
    shift_mmd: float,
    *,
    performance_weight: float = 0.6,
) -> float:
    """Combine performance gap and covariate shift into a transport score.

    Higher scores indicate better transportability (closer to the reference pool).

    Args:
        performance_gap: Absolute metric difference vs the reference pool.
        shift_mmd: MMD covariate-shift magnitude toward the reference.
        performance_weight: Weight on performance vs shift in [0, 1].

    Returns:
        Score in [0, 1].
    """
    shift_weight = 1.0 - performance_weight
    penalty = performance_weight * min(abs(performance_gap), 1.0) + shift_weight * min(
        shift_mmd, 1.0
    )
    return float(max(0.0, 1.0 - penalty))


def _propensity_reference(
    X: pd.DataFrame,
    pools: pd.Series[Any],
    reference_pool: str,
) -> npt.NDArray[np.float64]:
    """Estimate P(reference_pool | X) via logistic propensity scores."""
    pool_values = pools.astype(str)
    y_ref = (pool_values == reference_pool).astype(int).to_numpy()
    if len(np.unique(y_ref)) < 2:
        msg = "Reference pool must share covariates with at least one other pool"
        raise ValueError(msg)
    prop_model = LogisticRegression(max_iter=500, random_state=0)
    prop_model.fit(X, y_ref)
    return np.asarray(prop_model.predict_proba(X)[:, 1], dtype=np.float64)


def inverse_probability_weight(
    model: Any,
    X: pd.DataFrame | npt.NDArray[np.floating[Any]],
    y: pd.Series[Any] | npt.NDArray[Any],
    pools: pd.Series[Any] | npt.NDArray[Any],
    *,
    reference_pool: str,
    metric: str = "auc",
) -> dict[str, float]:
    """IPW transport estimates of a metric toward the reference population.

    For each non-reference pool, reweights samples with
    ``P(reference | X) / P(pool | X)`` and computes the weighted metric.

    Args:
        model: Fitted predictor.
        X: Feature matrix.
        y: Binary labels.
        pools: Pool labels.
        reference_pool: Target population for transport.
        metric: Metric to transport (default ``"auc"``).

    Returns:
        IPW-adjusted metric estimate per non-reference pool label.

    Raises:
        ValueError: If inputs are misaligned or the reference pool is degenerate.
    """
    frame = ensure_dataframe(X)
    labels = ensure_series(y, name="y")
    pool_series = ensure_series(pools, name="pool")
    validate_aligned_lengths(frame, labels, pool_series)

    p_ref = _propensity_reference(frame, pool_series, reference_pool)
    pool_values = pool_series.astype(str)
    scores = predict_proba_positive(model, frame)
    preds = model.predict(frame)
    y_array = labels.to_numpy()

    estimates: dict[str, float] = {}
    for pool in sorted(pool_values.unique()):
        if pool == reference_pool:
            continue
        mask = (pool_values == pool).to_numpy()
        p_pool = np.clip(1.0 - p_ref[mask], 1e-6, 1.0)
        weights = p_ref[mask] / p_pool
        weights = weights / np.mean(weights)

        weighted_scores = scores[mask]
        weighted_preds = preds[mask]
        weighted_y = y_array[mask]
        if metric == "auc":
            if len(np.unique(weighted_y)) < 2:
                estimates[str(pool)] = float("nan")
                continue
            estimates[str(pool)] = float(
                roc_auc_score(weighted_y, weighted_scores, sample_weight=weights)
            )
        else:
            boot_metrics = compute_binary_metrics(weighted_y, weighted_preds, weighted_scores)
            estimates[str(pool)] = boot_metrics.get(metric, float("nan"))

    return estimates


def g_computation_transport(
    model: Any,
    X: pd.DataFrame | npt.NDArray[np.floating[Any]],
    y: pd.Series[Any] | npt.NDArray[Any],
    pools: pd.Series[Any] | npt.NDArray[Any],
    *,
    reference_pool: str,
    metric: str = "auc",
) -> dict[str, float]:
    """G-computation transport under the reference covariate distribution.

    Fits a pool-specific outcome model for each source pool, evaluates
    counterfactual predictions on reference-pool covariates, and reports the metric.

    Args:
        model: Fitted predictor (must expose ``predict``).
        X: Feature matrix.
        y: Binary labels.
        pools: Pool labels.
        reference_pool: Target covariate distribution.
        metric: Metric to estimate (default ``"auc"``).

    Returns:
        G-computation metric estimate per non-reference pool label.

    Raises:
        TypeError: If ``model`` lacks ``predict``.
        ValueError: If ``reference_pool`` is missing or inputs are misaligned.
    """
    if not hasattr(model, "predict"):
        msg = "model must expose a predict method"
        raise TypeError(msg)

    frame = ensure_dataframe(X)
    labels = ensure_series(y, name="y")
    pool_series = ensure_series(pools, name="pool")
    validate_aligned_lengths(frame, labels, pool_series)

    ref_mask = pool_series.astype(str) == reference_pool
    if not ref_mask.any():
        msg = f"reference_pool '{reference_pool}' not found in pools"
        raise ValueError(msg)

    pool_values = pool_series.astype(str)
    y_ref = labels.loc[ref_mask].to_numpy()
    ref_frame = frame.loc[ref_mask]
    estimates: dict[str, float] = {}

    for pool in sorted(pool_values.unique()):
        if pool == reference_pool:
            continue
        pool_mask = pool_values == pool
        outcome_model = LogisticRegression(max_iter=500, random_state=0)
        outcome_model.fit(frame.loc[pool_mask], labels.loc[pool_mask].to_numpy())
        mu_ref = np.asarray(outcome_model.predict_proba(ref_frame)[:, 1], dtype=np.float64)
        if metric == "auc":
            if len(np.unique(y_ref)) < 2:
                estimates[str(pool)] = float("nan")
                continue
            estimates[str(pool)] = float(roc_auc_score(y_ref, mu_ref))
        else:
            pred_ref = (mu_ref >= 0.5).astype(int)
            estimates[str(pool)] = compute_binary_metrics(y_ref, pred_ref, mu_ref).get(
                metric, float("nan")
            )

    return estimates


def assess_transportability(
    model: Any,
    X: pd.DataFrame | np.ndarray[Any, Any],
    y: pd.Series[Any] | np.ndarray[Any, Any],
    pools: pd.Series[Any] | np.ndarray[Any, Any],
    *,
    metric: str = "auc",
    reference_pool: str | None = None,
    config: PoolGuardConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> TransportReport:
    """Quantify model transportability from a reference pool to others.

    Combines performance gaps, covariate shift (MMD), IPW, and g-computation.

    Args:
        model: Fitted predictor.
        X: Feature matrix.
        y: Binary labels.
        pools: Pool labels.
        metric: Performance metric for gap calculation (default ``"auc"``).
        reference_pool: Reference label; defaults to the largest pool.
        config: Analysis configuration.
        metadata: Extra report metadata.

    Returns:
        A :class:`~poolguard.report.TransportReport`.

    Raises:
        ValueError: If no non-reference pools exist for scoring.
    """
    cfg = config or PoolGuardConfig()
    shift_report = detect_shift(
        X,
        pools,
        reference_pool=reference_pool,
        config=cfg,
    )
    eval_report = evaluate_pools(model, X, y, pools, config=cfg)
    ref_pool = shift_report.reference_pool

    ref_metrics = next(p for p in eval_report.by_pool if p.pool == ref_pool).metrics
    ref_value = ref_metrics.get(metric, float("nan"))

    ipw_estimates = inverse_probability_weight(
        model,
        X,
        y,
        pools,
        reference_pool=ref_pool,
        metric=metric,
    )
    gcomp_estimates = g_computation_transport(
        model,
        X,
        y,
        pools,
        reference_pool=ref_pool,
        metric=metric,
    )

    transport_scores: dict[str, float] = {}
    for pool_result in eval_report.by_pool:
        pool = pool_result.pool
        if pool in ("__overall__", ref_pool):
            continue
        gap = abs(pool_result.metrics.get(metric, float("nan")) - ref_value)
        mmd = shift_report.mmd_scores.get(pool, 0.0)
        transport_scores[pool] = transportability_score(gap, mmd)

    if not transport_scores:
        msg = "No non-reference pools available for transportability assessment"
        raise ValueError(msg)

    best_pool = max(transport_scores, key=lambda pool: transport_scores[pool])
    worst_pool = min(transport_scores, key=lambda pool: transport_scores[pool])
    mean_score = float(np.mean(list(transport_scores.values())))

    run_id = str(uuid.uuid4())
    return TransportReport(
        timestamp=utc_now_iso(),
        run_id=run_id,
        metadata={**(cfg.metadata), **(metadata or {}), "metric": metric},
        reference_pool=ref_pool,
        transport_scores=transport_scores,
        worst_pool=worst_pool,
        best_pool=best_pool,
        mean_transport_score=mean_score,
        ipw_estimates=ipw_estimates,
        gcomp_estimates=gcomp_estimates,
    )
