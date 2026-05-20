"""Per-pool model performance evaluation with bootstrap confidence intervals."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from poolguard.config import PoolGuardConfig
from poolguard.report import EvaluateReport, PoolMetricResult
from poolguard.utils import (
    bootstrap_ci,
    compute_binary_metrics,
    ensure_dataframe,
    ensure_series,
    predict_proba_positive,
    utc_now_iso,
    validate_aligned_lengths,
)

PredictFn = Callable[[pd.DataFrame], npt.NDArray[Any]]


def evaluate_pools(
    model: Any,
    X: pd.DataFrame | npt.NDArray[np.floating[Any]],
    y: pd.Series[Any] | npt.NDArray[Any],
    pools: pd.Series[Any] | npt.NDArray[Any],
    *,
    config: PoolGuardConfig | None = None,
    metadata: dict[str, Any] | None = None,
    predict_fn: PredictFn | None = None,
) -> EvaluateReport:
    """Evaluate model performance overall and within each pool.

    Computes accuracy, F1, Brier score, and AUC (when both classes are present)
    with percentile bootstrap confidence intervals per metric.

    Args:
        model: Fitted model with ``predict`` and optionally ``predict_proba``.
        X: Feature matrix.
        y: Binary labels aligned with ``X``.
        pools: Pool labels aligned with ``X`` and ``y``.
        config: Bootstrap count, CI level, and metadata.
        metadata: Extra report metadata.
        predict_fn: Optional ``(X) -> y_pred`` override for non-sklearn models.

    Returns:
        An :class:`~poolguard.report.EvaluateReport`.

    Raises:
        ValueError: If ``X``, ``y``, and ``pools`` lengths differ.
    """
    cfg = config or PoolGuardConfig()
    frame = ensure_dataframe(X)
    labels = ensure_series(y, name="y")
    pool_series = ensure_series(pools, name="pool")
    validate_aligned_lengths(frame, labels, pool_series)

    y_array = labels.to_numpy()
    pool_values = pool_series.astype(str)
    rng = np.random.default_rng(cfg.random_state)

    scores = predict_proba_positive(model, frame)
    preds = predict_fn(frame) if predict_fn is not None else model.predict(frame)
    preds_array = np.asarray(preds)

    overall = _evaluate_subset(
        y_array,
        preds_array,
        scores,
        pool_label="__overall__",
        rng=rng,
        bootstrap_n=cfg.bootstrap_n,
        ci_level=cfg.bootstrap_ci,
    )

    by_pool: list[PoolMetricResult] = []
    for pool in sorted(pool_values.unique()):
        mask = pool_values == pool
        by_pool.append(
            _evaluate_subset(
                y_array[mask.to_numpy()],
                preds_array[mask.to_numpy()],
                scores[mask.to_numpy()],
                pool_label=str(pool),
                rng=rng,
                bootstrap_n=cfg.bootstrap_n,
                ci_level=cfg.bootstrap_ci,
            )
        )

    metric_names = tuple(overall.metrics.keys())
    run_id = str(uuid.uuid4())
    return EvaluateReport(
        timestamp=utc_now_iso(),
        run_id=run_id,
        metadata={**(cfg.metadata), **(metadata or {})},
        overall=overall,
        by_pool=tuple(by_pool),
        metric_names=metric_names,
    )


def _evaluate_subset(
    y_true: npt.NDArray[Any],
    y_pred: npt.NDArray[Any],
    y_score: npt.NDArray[np.float64],
    *,
    pool_label: str,
    rng: np.random.Generator,
    bootstrap_n: int,
    ci_level: float,
) -> PoolMetricResult:
    metrics = compute_binary_metrics(y_true, y_pred, y_score)
    ci: dict[str, tuple[float, float]] = {}
    n = len(y_true)

    for name in metrics:
        if n == 0:
            ci[name] = (float("nan"), float("nan"))
            continue
        boot_values = np.empty(bootstrap_n, dtype=np.float64)
        for i in range(bootstrap_n):
            idx = rng.integers(0, n, size=n)
            boot_metrics = compute_binary_metrics(y_true[idx], y_pred[idx], y_score[idx])
            boot_values[i] = boot_metrics[name]
        ci[name] = bootstrap_ci(boot_values, ci=ci_level)

    return PoolMetricResult(pool=pool_label, n=n, metrics=metrics, ci=ci)
