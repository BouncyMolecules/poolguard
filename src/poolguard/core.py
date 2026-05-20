"""PoolGuard orchestrator — model-agnostic end-to-end patient-pool workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from poolguard.audit import AuditTrail
from poolguard.config import PoolGuardConfig
from poolguard.evaluate import evaluate_pools
from poolguard.heterogeneity import analyze_heterogeneity
from poolguard.report import (
    HeterogeneityReport,
    PoolEvaluationReport,
    ReportBase,
    ShiftReport,
    TransportReport,
    report_digest,
)
from poolguard.shift import detect_shift
from poolguard.transport import assess_transportability
from poolguard.utils import ensure_dataframe, ensure_series, validate_aligned_lengths


@dataclass(frozen=True)
class PoolGuardAnalysis:
    """Reports from a full :meth:`PoolGuard.run_full_analysis` run.

    Attributes:
        shift: Covariate shift vs the reference pool.
        evaluation: Per-pool performance with bootstrap CIs.
        heterogeneity: Cochran's Q, I², and interaction tests.
        transport: IPW, g-computation, and transportability scores.
    """

    shift: ShiftReport
    evaluation: PoolEvaluationReport
    heterogeneity: HeterogeneityReport
    transport: TransportReport


class PoolGuard:
    """Model-agnostic patient-pool analysis for clinical ML.

    Wraps shift detection, per-pool evaluation, heterogeneity testing, and
    transportability assessment with optional hash-chained audit logging.

    Args:
        model: Fitted predictor; may be supplied per method instead.
        pool_column: Column name for pool labels when ``X`` is a DataFrame.
        config: Runtime thresholds and bootstrap settings.
        study_id: Identifier stored in the audit trail.

    Example:
        >>> pg = PoolGuard(model=clf, pool_column="site")
        >>> pg.fit_reference(X_train, y_train, pools_train)
        >>> analysis = pg.run_full_analysis(X_test, y_test, pools_test)
    """

    def __init__(
        self,
        model: Any | None = None,
        *,
        pool_column: str | None = None,
        config: PoolGuardConfig | None = None,
        study_id: str = "default-study",
    ) -> None:
        """Initialize PoolGuard with optional model, pool column, and config."""
        self.model = model
        self.pool_column = pool_column
        self.config = config or PoolGuardConfig()
        self.audit = AuditTrail(study_id=study_id)
        self._reference_pool: str | None = None
        self._reference_X: pd.DataFrame | None = None
        self._reference_pools: pd.Series[Any] | None = None

    def fit_reference(
        self,
        X: pd.DataFrame | npt.NDArray[np.floating[Any]],
        y: pd.Series[Any] | npt.NDArray[Any] | None = None,
        pools: pd.Series[Any] | npt.NDArray[Any] | None = None,
        *,
        reference_pool: str | None = None,
    ) -> str:
        """Store the reference cohort for subsequent shift analyses.

        Args:
            X: Reference feature matrix.
            y: Reference labels (recorded in audit metadata when provided).
            pools: Pool labels; read from ``pool_column`` when omitted and present in ``X``.
            reference_pool: Explicit reference label; defaults to the largest pool.

        Returns:
            The selected reference pool name.

        Raises:
            ValueError: If ``pools`` is missing and ``pool_column`` is not in ``X``.
        """
        frame = ensure_dataframe(X)
        if pools is None and self.pool_column and self.pool_column in frame.columns:
            pool_series = ensure_series(frame[self.pool_column], name=self.pool_column)
            feature_frame = frame.drop(columns=[self.pool_column])
        else:
            if pools is None:
                msg = "pools must be provided when pool_column is not in X"
                raise ValueError(msg)
            pool_series = ensure_series(pools, name="pool")
            feature_frame = frame

        if reference_pool is None:
            counts = pool_series.astype(str).value_counts()
            reference_pool = str(counts.index[0])

        self._reference_pool = reference_pool
        self._reference_X = feature_frame
        self._reference_pools = pool_series

        payload: dict[str, Any] = {
            "reference_pool": reference_pool,
            "n_samples": len(feature_frame),
            "n_pools": int(pool_series.nunique()),
            "feature_columns": list(feature_frame.columns),
        }
        if y is not None:
            payload["n_labels"] = len(y)
        self._audit("fit_reference", payload)
        return reference_pool

    def detect_shift(
        self,
        X: pd.DataFrame | npt.NDArray[np.floating[Any]],
        pools: pd.Series[Any] | npt.NDArray[Any] | None = None,
        *,
        reference_pool: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ShiftReport:
        """Detect covariate shift relative to the reference cohort.

        Args:
            X: Feature matrix for the comparison cohort.
            pools: Pool labels aligned with ``X``.
            reference_pool: Override the stored reference pool.
            metadata: Extra key-value pairs attached to the report.

        Returns:
            A :class:`~poolguard.report.ShiftReport`.
        """
        frame, pool_series = self._resolve_inputs(X, pools)
        ref_pool = reference_pool or self._reference_pool
        report = detect_shift(
            frame,
            pool_series,
            reference_pool=ref_pool,
            config=self.config,
            metadata=metadata,
        )
        self._audit_report("detect_shift", report)
        return report

    def evaluate(
        self,
        X: pd.DataFrame | npt.NDArray[np.floating[Any]],
        y: pd.Series[Any] | npt.NDArray[Any],
        pools: pd.Series[Any] | npt.NDArray[Any] | None = None,
        *,
        model: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PoolEvaluationReport:
        """Evaluate model performance overall and within each pool.

        Args:
            X: Feature matrix.
            y: Binary outcome labels.
            pools: Pool labels aligned with ``X`` and ``y``.
            model: Override the instance model.
            metadata: Extra report metadata.

        Returns:
            An :class:`~poolguard.report.EvaluateReport`.

        Raises:
            ValueError: If no fitted model is available.
        """
        predictor = model or self.model
        if predictor is None:
            msg = "A fitted model must be provided via constructor or evaluate(model=...)"
            raise ValueError(msg)

        frame, pool_series = self._resolve_inputs(X, pools)
        labels = ensure_series(y, name="y")
        validate_aligned_lengths(frame, labels, pool_series)

        report = evaluate_pools(
            predictor,
            frame,
            labels,
            pool_series,
            config=self.config,
            metadata=metadata,
        )
        self._audit_report("evaluate", report)
        return report

    def test_heterogeneity(
        self,
        X: pd.DataFrame | npt.NDArray[np.floating[Any]],
        y: pd.Series[Any] | npt.NDArray[Any],
        pools: pd.Series[Any] | npt.NDArray[Any] | None = None,
        *,
        metric: str = "auc",
        model: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HeterogeneityReport:
        """Test whether a performance metric differs across pools.

        Args:
            X: Feature matrix.
            y: Binary outcome labels.
            pools: Pool labels.
            metric: Metric name from evaluation (default ``"auc"``).
            model: Override the instance model.
            metadata: Extra report metadata.

        Returns:
            A :class:`~poolguard.report.HeterogeneityReport`.

        Raises:
            ValueError: If no fitted model is available.
        """
        predictor = model or self.model
        if predictor is None:
            msg = "A fitted model must be provided"
            raise ValueError(msg)

        frame, pool_series = self._resolve_inputs(X, pools)
        labels = ensure_series(y, name="y")
        report = analyze_heterogeneity(
            predictor,
            frame,
            labels,
            pool_series,
            metric=metric,
            config=self.config,
            metadata=metadata,
        )
        self._audit_report("test_heterogeneity", report)
        return report

    def assess_transportability(
        self,
        X: pd.DataFrame | npt.NDArray[np.floating[Any]],
        y: pd.Series[Any] | npt.NDArray[Any],
        pools: pd.Series[Any] | npt.NDArray[Any] | None = None,
        *,
        metric: str = "auc",
        reference_pool: str | None = None,
        model: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TransportReport:
        """Quantify transportability from the reference pool to others.

        Args:
            X: Feature matrix.
            y: Binary outcome labels.
            pools: Pool labels.
            metric: Performance metric for gap calculation (default ``"auc"``).
            reference_pool: Override the stored reference pool.
            model: Override the instance model.
            metadata: Extra report metadata.

        Returns:
            A :class:`~poolguard.report.TransportReport`.

        Raises:
            ValueError: If no fitted model is available.
        """
        predictor = model or self.model
        if predictor is None:
            msg = "A fitted model must be provided"
            raise ValueError(msg)

        frame, pool_series = self._resolve_inputs(X, pools)
        labels = ensure_series(y, name="y")
        report = assess_transportability(
            predictor,
            frame,
            labels,
            pool_series,
            metric=metric,
            reference_pool=reference_pool or self._reference_pool,
            config=self.config,
            metadata=metadata,
        )
        self._audit_report("assess_transportability", report)
        return report

    def run_full_analysis(
        self,
        X: pd.DataFrame | npt.NDArray[np.floating[Any]],
        y: pd.Series[Any] | npt.NDArray[Any],
        pools: pd.Series[Any] | npt.NDArray[Any] | None = None,
        *,
        metric: str = "auc",
        model: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PoolGuardAnalysis:
        """Run shift, evaluation, heterogeneity, and transport in one pass.

        Args:
            X: Feature matrix.
            y: Binary outcome labels.
            pools: Pool labels.
            metric: Metric for heterogeneity and transport (default ``"auc"``).
            model: Override the instance model.
            metadata: Shared metadata for all reports.

        Returns:
            A :class:`PoolGuardAnalysis` bundle.

        Raises:
            ValueError: If no fitted model is available.
        """
        predictor = model or self.model
        if predictor is None:
            msg = "A fitted model must be provided"
            raise ValueError(msg)

        shared_meta = {**(metadata or {})}
        shift = self.detect_shift(X, pools, metadata=shared_meta)
        evaluation = self.evaluate(X, y, pools, model=predictor, metadata=shared_meta)
        heterogeneity = self.test_heterogeneity(
            X,
            y,
            pools,
            metric=metric,
            model=predictor,
            metadata=shared_meta,
        )
        transport = self.assess_transportability(
            X,
            y,
            pools,
            metric=metric,
            model=predictor,
            metadata=shared_meta,
        )
        self._audit(
            "run_full_analysis",
            {
                "metric": metric,
                "shift_run_id": shift.run_id,
                "evaluation_run_id": evaluation.run_id,
                "heterogeneity_run_id": heterogeneity.run_id,
                "transport_run_id": transport.run_id,
            },
        )
        return PoolGuardAnalysis(
            shift=shift,
            evaluation=evaluation,
            heterogeneity=heterogeneity,
            transport=transport,
        )

    def _resolve_inputs(
        self,
        X: pd.DataFrame | npt.NDArray[np.floating[Any]],
        pools: pd.Series[Any] | npt.NDArray[Any] | None,
    ) -> tuple[pd.DataFrame, pd.Series[Any]]:
        frame = ensure_dataframe(X)
        if pools is None and self.pool_column and self.pool_column in frame.columns:
            pool_series = ensure_series(frame[self.pool_column], name=self.pool_column)
            feature_frame = frame.drop(columns=[self.pool_column])
            return feature_frame, pool_series
        if pools is None:
            msg = "pools must be provided when pool_column is not in X"
            raise ValueError(msg)
        pool_series = ensure_series(pools, name="pool")
        return frame, pool_series

    def _audit(self, action: str, payload: dict[str, Any]) -> None:
        if self.config.audit_enabled:
            self.audit.record(action, payload)

    def _audit_report(self, action: str, report: ReportBase) -> None:
        if self.config.audit_enabled:
            self.audit.record(
                action,
                {
                    "run_id": report.run_id,
                    "report_type": report.__class__.__name__,
                    "report_digest": report_digest(report),
                },
            )
