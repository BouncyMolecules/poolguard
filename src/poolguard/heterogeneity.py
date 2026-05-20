"""Cross-pool heterogeneity testing."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from statsmodels.tools.tools import add_constant

from poolguard.config import PoolGuardConfig
from poolguard.evaluate import evaluate_pools
from poolguard.report import ForestPlotEntry, HeterogeneityReport
from poolguard.utils import ensure_dataframe, ensure_series, predict_proba_positive, utc_now_iso


def calculate_i_squared(q_stat: float, df: int) -> float:
    """Derive I² heterogeneity percentage from Cochran's Q.

    Args:
        q_stat: Cochran's Q statistic.
        df: Degrees of freedom (number of pools minus one).

    Returns:
        I² in [0, 100].
    """
    if q_stat <= 0.0:
        return 0.0
    return max(0.0, ((q_stat - df) / q_stat) * 100.0)


def cochran_q_test(
    estimates: dict[str, float],
    variances: dict[str, float],
) -> tuple[float, float, float, float]:
    """Cochran's Q test for heterogeneity across pool-level estimates.

    Args:
        estimates: Point estimates per pool (e.g. AUC).
        variances: Sampling variances per pool (same keys as ``estimates``).

    Returns:
        Tuple of (Q statistic, p-value, I² percentage, inverse-variance pooled estimate).

    Raises:
        ValueError: If fewer than two pools are provided.
    """
    pools = sorted(estimates.keys())
    if len(pools) < 2:
        msg = "At least two pools are required for heterogeneity testing"
        raise ValueError(msg)

    thetas = np.array([estimates[p] for p in pools], dtype=np.float64)
    vars_ = np.array([max(variances[p], 1e-12) for p in pools], dtype=np.float64)
    weights = 1.0 / vars_
    pooled = float(np.sum(weights * thetas) / np.sum(weights))
    q_stat = float(np.sum(weights * (thetas - pooled) ** 2))
    df = len(pools) - 1
    p_value = float(1.0 - stats.chi2.cdf(q_stat, df))
    i_squared = calculate_i_squared(q_stat, df)
    return q_stat, p_value, i_squared, pooled


def pool_interaction_test(
    scores: npt.NDArray[np.float64],
    y: npt.NDArray[Any],
    pools: pd.Series[Any],
    *,
    reference_pool: str | None = None,
) -> tuple[float, float]:
    """Likelihood-ratio test for pool-by-score interaction on outcomes.

    Fits logistic models with and without ``score:pool`` interaction terms.
    Falls back to a binned chi-square test when maximum likelihood fails to converge.

    Args:
        scores: Model risk scores or probabilities.
        y: Binary outcomes.
        pools: Pool labels aligned with ``scores`` and ``y``.
        reference_pool: Reference pool omitted from dummy encoding.

    Returns:
        Tuple of (LR chi-square statistic, p-value).

    Raises:
        ValueError: If fewer than two pools are present.
    """
    import statsmodels.api as sm

    pool_values = pools.astype(str)
    ref = reference_pool or str(pool_values.value_counts().index[0])
    unique_pools = sorted(pool_values.unique())
    if len(unique_pools) < 2:
        msg = "At least two pools are required for interaction testing"
        raise ValueError(msg)

    y_array = np.asarray(y, dtype=np.float64)
    score_col = np.asarray(scores, dtype=np.float64)
    score_std = float(np.std(score_col))
    if score_std > 0.0:
        score_col = (score_col - float(np.mean(score_col))) / score_std

    design = pd.DataFrame({"score": score_col})
    for pool in unique_pools:
        if pool == ref:
            continue
        design[f"pool_{pool}"] = (pool_values == pool).astype(np.float64)
        design[f"score_x_{pool}"] = design["score"] * design[f"pool_{pool}"]

    reduced = add_constant(design[["score"] + [c for c in design.columns if c.startswith("pool_")]])
    full = add_constant(design)

    try:
        reduced_model = sm.Logit(y_array, reduced).fit(disp=False, maxiter=300, method="bfgs")
        full_model = sm.Logit(y_array, full).fit(disp=False, maxiter=300, method="bfgs")
        if not reduced_model.mle_retvals["converged"] or not full_model.mle_retvals["converged"]:
            return _fallback_interaction_test(score_col, y_array, pool_values)
    except Exception:
        return _fallback_interaction_test(score_col, y_array, pool_values)

    lr_stat = float(max(2.0 * (full_model.llf - reduced_model.llf), 0.0))
    n_interactions = len(unique_pools) - 1
    p_value = float(1.0 - stats.chi2.cdf(lr_stat, n_interactions))
    return lr_stat, p_value


def _fallback_interaction_test(
    scores: npt.NDArray[np.float64],
    y: npt.NDArray[Any],
    pools: pd.Series[Any],
) -> tuple[float, float]:
    """Binned score-by-pool contingency fallback when logistic LR fails."""
    pool_values = pools.astype(str)
    bins = pd.qcut(scores, q=4, duplicates="drop")
    contingency = pd.crosstab([pool_values, bins], y)
    if contingency.size == 0 or contingency.shape[0] < 2:
        return 0.0, 1.0
    chi2, p_value, _, _ = stats.chi2_contingency(contingency.to_numpy())
    return float(chi2), float(p_value)


def forest_plot_summary(
    estimates: dict[str, float],
    variances: dict[str, float],
    *,
    pooled_estimate: float | None = None,
) -> tuple[ForestPlotEntry, ...]:
    """Build forest-plot rows with inverse-variance weights.

    Args:
        estimates: Point estimates per pool.
        variances: Sampling variances per pool.
        pooled_estimate: Pooled line; computed via IVW when omitted.

    Returns:
        Ordered :class:`~poolguard.report.ForestPlotEntry` rows including ``__pooled__``.
    """
    pools = sorted(estimates.keys())
    if not pools:
        return ()

    vars_ = np.array([max(variances[p], 1e-12) for p in pools], dtype=np.float64)
    weights = 1.0 / vars_
    weight_share = weights / np.sum(weights)
    if pooled_estimate is None:
        pooled_estimate = float(np.sum(weights * [estimates[p] for p in pools]) / np.sum(weights))

    entries: list[ForestPlotEntry] = []
    for pool, w in zip(pools, weight_share, strict=True):
        se = float(np.sqrt(max(variances[pool], 1e-12)))
        estimate = estimates[pool]
        entries.append(
            ForestPlotEntry(
                pool=pool,
                estimate=estimate,
                ci_lower=estimate - 1.96 * se,
                ci_upper=estimate + 1.96 * se,
                weight=float(w),
            )
        )

    entries.append(
        ForestPlotEntry(
            pool="__pooled__",
            estimate=pooled_estimate,
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            weight=1.0,
        )
    )
    return tuple(entries)


def analyze_heterogeneity(
    model: Any,
    X: pd.DataFrame | npt.NDArray[np.floating[Any]],
    y: pd.Series[Any] | npt.NDArray[Any],
    pools: pd.Series[Any] | npt.NDArray[Any],
    *,
    metric: str = "auc",
    config: PoolGuardConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> HeterogeneityReport:
    """Test whether a performance metric varies significantly across pools.

    Combines bootstrap variance estimates, Cochran's Q, a pool-by-score interaction
    test, and an inverse-variance forest-plot summary.

    Args:
        model: Fitted predictor.
        X: Feature matrix.
        y: Binary labels.
        pools: Pool labels.
        metric: Metric from evaluation (default ``"auc"``).
        config: Bootstrap and significance settings.
        metadata: Extra report metadata.

    Returns:
        A :class:`~poolguard.report.HeterogeneityReport`.
    """
    cfg = config or PoolGuardConfig()
    frame = ensure_dataframe(X)
    labels = ensure_series(y, name="y")
    pool_series = ensure_series(pools, name="pool")

    eval_report = evaluate_pools(model, frame, labels, pool_series, config=cfg)
    pool_estimates: dict[str, float] = {}
    pool_variances: dict[str, float] = {}

    for pool_result in eval_report.by_pool:
        if pool_result.pool == "__overall__":
            continue
        value = pool_result.metrics.get(metric, float("nan"))
        ci = pool_result.ci.get(metric, (float("nan"), float("nan")))
        lower, upper = ci
        se = (upper - lower) / (2.0 * 1.96) if upper > lower else 1e-6
        pool_estimates[pool_result.pool] = value
        pool_variances[pool_result.pool] = se**2

    q_stat, p_value, i_squared, pooled = cochran_q_test(pool_estimates, pool_variances)
    forest = forest_plot_summary(pool_estimates, pool_variances, pooled_estimate=pooled)

    scores = predict_proba_positive(model, frame)
    ref_pool = str(pool_series.astype(str).value_counts().index[0])
    lr_stat, lr_p = pool_interaction_test(
        scores,
        labels.to_numpy(),
        pool_series,
        reference_pool=ref_pool,
    )

    run_id = str(uuid.uuid4())
    return HeterogeneityReport(
        timestamp=utc_now_iso(),
        run_id=run_id,
        metadata={**(cfg.metadata), **(metadata or {}), "metric": metric},
        metric=metric,
        cochran_q=q_stat,
        cochran_pvalue=p_value,
        i_squared=i_squared,
        pool_estimates=pool_estimates,
        heterogeneous=p_value < cfg.ks_alpha,
        pooled_estimate=pooled,
        interaction_stat=lr_stat,
        interaction_pvalue=lr_p,
        forest_plot=forest,
    )
