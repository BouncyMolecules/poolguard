"""Population shift detection methods."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from poolguard.config import PoolGuardConfig
from poolguard.report import ShiftFeatureResult, ShiftReport
from poolguard.utils import ensure_dataframe, ensure_series, numeric_columns, utc_now_iso


def standardized_mean_difference(
    reference: npt.NDArray[np.float64],
    comparison: npt.NDArray[np.float64],
) -> float:
    """Compute pooled standardized mean difference (SMD) between two samples.

    Args:
        reference: Reference-sample values.
        comparison: Comparison-sample values.

    Returns:
        SMD using pooled variance; ``0.0`` when both samples are constant and equal,
        or ``inf`` when means differ but variance is zero.
    """
    ref = reference.astype(np.float64, copy=False)
    cmp = comparison.astype(np.float64, copy=False)
    mean_diff = float(np.mean(ref) - np.mean(cmp))
    var_ref = float(np.var(ref, ddof=1)) if len(ref) > 1 else 0.0
    var_cmp = float(np.var(cmp, ddof=1)) if len(cmp) > 1 else 0.0
    pooled_var = (var_ref + var_cmp) / 2.0
    if pooled_var <= 0.0:
        return 0.0 if mean_diff == 0.0 else float("inf")
    return float(mean_diff / np.sqrt(pooled_var))


def kolmogorov_smirnov_test(
    reference: npt.NDArray[np.float64],
    comparison: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Run a two-sample Kolmogorov-Smirnov test.

    Args:
        reference: Reference-sample values.
        comparison: Comparison-sample values.

    Returns:
        Tuple of (statistic, two-sided p-value).
    """
    result = stats.ks_2samp(reference, comparison, method="auto")
    return float(result.statistic), float(result.pvalue)


def maximum_mean_discrepancy(
    reference: npt.NDArray[np.float64],
    comparison: npt.NDArray[np.float64],
    *,
    gamma: float | None = None,
) -> float:
    """Compute RBF-kernel maximum mean discrepancy (MMD) between two samples.

    Args:
        reference: Reference observations (n x d or length n).
        comparison: Comparison observations, same dimensionality as ``reference``.
        gamma: RBF bandwidth; median heuristic when ``None``.

    Returns:
        Non-negative MMD estimate.
    """
    x = reference.astype(np.float64, copy=False)
    y = comparison.astype(np.float64, copy=False)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    bandwidth = gamma if gamma is not None else _median_bandwidth(np.vstack([x, y]))
    k_xx = _rbf_kernel(x, x, bandwidth)
    k_yy = _rbf_kernel(y, y, bandwidth)
    k_xy = _rbf_kernel(x, y, bandwidth)
    m = x.shape[0]
    n = y.shape[0]
    term_xx = (np.sum(k_xx) - np.trace(k_xx)) / (m * (m - 1)) if m > 1 else 0.0
    term_yy = (np.sum(k_yy) - np.trace(k_yy)) / (n * (n - 1)) if n > 1 else 0.0
    term_xy = np.mean(k_xy)
    return float(max(term_xx + term_yy - 2.0 * term_xy, 0.0))


def _median_bandwidth(data: npt.NDArray[np.float64]) -> float:
    if data.shape[0] < 2:
        return 1.0
    pairwise = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
    upper = pairwise[np.triu_indices_from(pairwise, k=1)]
    if upper.size == 0:
        return 1.0
    median = float(np.median(upper))
    return median if median > 0.0 else 1.0


def _rbf_kernel(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    gamma: float,
) -> npt.NDArray[np.float64]:
    sq_dist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-gamma * sq_dist)
    return np.asarray(kernel, dtype=np.float64)


def detect_shift(
    X: pd.DataFrame | npt.NDArray[np.floating[Any]],
    pools: pd.Series[Any] | npt.NDArray[Any],
    *,
    reference_pool: str | None = None,
    config: PoolGuardConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> ShiftReport:
    """Detect covariate shift between a reference pool and all other pools.

    Per numeric covariate, computes SMD and KS tests; per comparison pool, computes
    multivariate MMD. Pools are flagged when any covariate exceeds configured
    thresholds.

    Args:
        X: Feature matrix (numeric columns only).
        pools: Pool labels aligned with ``X``.
        reference_pool: Reference label; defaults to the largest pool.
        config: SMD, KS, and MMD settings.
        metadata: Extra key-value pairs stored on the report.

    Returns:
        A :class:`~poolguard.report.ShiftReport`.

    Raises:
        ValueError: If inputs are misaligned, lack numeric columns, or contain
            fewer than two pools; if ``reference_pool`` is absent from ``pools``.
    """
    cfg = config or PoolGuardConfig()
    frame = ensure_dataframe(X)
    pool_series = ensure_series(pools, name="pool")
    if len(frame) != len(pool_series):
        msg = "X and pools must have the same number of rows"
        raise ValueError(msg)

    columns = numeric_columns(frame)
    if not columns:
        msg = "X must contain at least one numeric column"
        raise ValueError(msg)

    numeric_frame = frame[columns].astype(np.float64)
    pool_values = pool_series.astype(str)
    unique_pools = pool_values.unique().tolist()
    if len(unique_pools) < 2:
        msg = "At least two distinct pools are required"
        raise ValueError(msg)

    ref_pool = reference_pool or _largest_pool(pool_values)
    if ref_pool not in unique_pools:
        msg = f"reference_pool '{ref_pool}' not found in pools"
        raise ValueError(msg)

    ref_mask = pool_values == ref_pool
    ref_data = numeric_frame.loc[ref_mask].to_numpy()

    comparisons: dict[str, dict[str, Any]] = {}
    mmd_scores: dict[str, float] = {}
    feature_results: list[ShiftFeatureResult] = []
    flagged: set[str] = set()

    other_pools = [p for p in unique_pools if p != ref_pool]
    for pool in other_pools:
        cmp_mask = pool_values == pool
        cmp_data = numeric_frame.loc[cmp_mask].to_numpy()
        pool_feature_flags: list[bool] = []

        for col in columns:
            ref_col = numeric_frame.loc[ref_mask, col].to_numpy()
            cmp_col = numeric_frame.loc[cmp_mask, col].to_numpy()
            smd = standardized_mean_difference(ref_col, cmp_col)
            ks_stat, ks_p = kolmogorov_smirnov_test(ref_col, cmp_col)
            smd_flagged = abs(smd) >= cfg.smd_threshold
            ks_flagged = ks_p < cfg.ks_alpha
            pool_feature_flags.extend([smd_flagged, ks_flagged])
            feature_results.append(
                ShiftFeatureResult(
                    feature=f"{col}@{pool}",
                    smd=float(smd),
                    smd_flagged=smd_flagged,
                    ks_statistic=ks_stat,
                    ks_pvalue=ks_p,
                    ks_flagged=ks_flagged,
                )
            )

        mmd = maximum_mean_discrepancy(ref_data, cmp_data, gamma=cfg.mmd_gamma)
        mmd_scores[pool] = mmd
        comparisons[pool] = {
            "n_reference": int(ref_mask.sum()),
            "n_comparison": int(cmp_mask.sum()),
            "mmd": mmd,
        }
        if any(pool_feature_flags):
            flagged.add(pool)

    run_id = str(uuid.uuid4())
    return ShiftReport(
        timestamp=utc_now_iso(),
        run_id=run_id,
        metadata={**(cfg.metadata), **(metadata or {})},
        reference_pool=ref_pool,
        comparisons=comparisons,
        mmd_scores=mmd_scores,
        flagged_pools=tuple(sorted(flagged)),
        feature_results=tuple(feature_results),
    )


def _largest_pool(pools: pd.Series[Any]) -> str:
    counts = pools.value_counts()
    return str(counts.index[0])
