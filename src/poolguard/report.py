"""Structured report objects returned by poolguard analyses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

from poolguard.utils import stable_json_dumps


@dataclass(frozen=True, kw_only=True)
class ReportBase:
    """Base class for immutable analysis reports."""

    timestamp: str
    run_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-compatible dictionary."""
        data: dict[str, Any] = {
            "report_type": self.__class__.__name__,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "metadata": self.metadata,
        }
        data.update(self._payload())
        return data

    def _payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def summary(self) -> str:
        """Return a human-readable summary."""
        raise NotImplementedError

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialize the report to JSON, optionally writing to ``path``."""
        payload = json.dumps(self.to_dict(), indent=indent, default=_json_fallback)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_html(self, path: str | Path | None = None) -> str:
        """Render the report as a lightweight HTML document."""
        rows = _dict_to_rows(self.to_dict())
        body_rows = "".join(
            f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
            for key, value in rows
        )
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{escape(self.__class__.__name__)}</title>"
            "<style>body{font-family:system-ui,sans-serif;margin:2rem;}"
            "table{border-collapse:collapse;width:100%;max-width:960px;}"
            "th,td{border:1px solid #ccc;padding:0.5rem;text-align:left;}"
            "th{background:#f5f5f5;width:30%;}</style></head><body>"
            f"<h1>{escape(self.__class__.__name__)}</h1>"
            f"<pre>{escape(self.summary())}</pre>"
            f"<table>{body_rows}</table></body></html>"
        )
        if path is not None:
            Path(path).write_text(html, encoding="utf-8")
        return html

    def to_pdf(self, path: str | Path) -> None:
        """Write a PDF summary using matplotlib (optional dependency)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            msg = "PDF export requires matplotlib: pip install matplotlib"
            raise ImportError(msg) from exc

        lines = self.summary().splitlines()
        fig_height = max(4.0, 0.35 * len(lines) + 1.5)
        fig, ax = plt.subplots(figsize=(8.5, fig_height))
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            wrap=True,
        )
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)


@dataclass(frozen=True, kw_only=True)
class ShiftFeatureResult:
    """Shift statistics for a single covariate."""

    feature: str
    smd: float
    smd_flagged: bool
    ks_statistic: float
    ks_pvalue: float
    ks_flagged: bool


@dataclass(frozen=True, kw_only=True)
class ShiftReport(ReportBase):
    """Population shift detection results."""

    reference_pool: str
    comparisons: dict[str, dict[str, Any]]
    mmd_scores: dict[str, float]
    flagged_pools: tuple[str, ...]
    feature_results: tuple[ShiftFeatureResult, ...]

    def _payload(self) -> dict[str, Any]:
        return {
            "reference_pool": self.reference_pool,
            "comparisons": self.comparisons,
            "mmd_scores": self.mmd_scores,
            "flagged_pools": list(self.flagged_pools),
            "feature_results": [
                {
                    "feature": r.feature,
                    "smd": r.smd,
                    "smd_flagged": r.smd_flagged,
                    "ks_statistic": r.ks_statistic,
                    "ks_pvalue": r.ks_pvalue,
                    "ks_flagged": r.ks_flagged,
                }
                for r in self.feature_results
            ],
        }

    def summary(self) -> str:
        """Return a human-readable shift summary."""
        lines = [
            f"ShiftReport run_id={self.run_id}",
            f"  reference_pool: {self.reference_pool}",
            f"  flagged_pools: {', '.join(self.flagged_pools) or 'none'}",
        ]
        for pool, score in self.mmd_scores.items():
            lines.append(f"  MMD({self.reference_pool} vs {pool}): {score:.4f}")
        return "\n".join(lines)


@dataclass(frozen=True)
class PoolMetricResult:
    """Performance metrics for a single pool."""

    pool: str
    n: int
    metrics: dict[str, float]
    ci: dict[str, tuple[float, float]]


@dataclass(frozen=True, kw_only=True)
class EvaluateReport(ReportBase):
    """Per-pool model performance evaluation."""

    overall: PoolMetricResult
    by_pool: tuple[PoolMetricResult, ...]
    metric_names: tuple[str, ...]

    def _payload(self) -> dict[str, Any]:
        return {
            "overall": _pool_metric_to_dict(self.overall),
            "by_pool": [_pool_metric_to_dict(p) for p in self.by_pool],
            "metric_names": list(self.metric_names),
        }

    def summary(self) -> str:
        """Return a human-readable evaluation summary."""
        lines = [f"EvaluateReport run_id={self.run_id}", "  overall:"]
        for name in self.metric_names:
            value = self.overall.metrics[name]
            lower, upper = self.overall.ci[name]
            lines.append(f"    {name}: {value:.4f} [{lower:.4f}, {upper:.4f}]")
        for pool_result in self.by_pool:
            lines.append(f"  pool={pool_result.pool} (n={pool_result.n}):")
            for name in self.metric_names:
                value = pool_result.metrics[name]
                lines.append(f"    {name}: {value:.4f}")
        return "\n".join(lines)


PoolEvaluationReport = EvaluateReport


@dataclass(frozen=True)
class ForestPlotEntry:
    """Single row for a forest-plot style heterogeneity summary."""

    pool: str
    estimate: float
    ci_lower: float
    ci_upper: float
    weight: float


@dataclass(frozen=True, kw_only=True)
class HeterogeneityReport(ReportBase):
    """Cross-pool heterogeneity test results."""

    metric: str
    cochran_q: float
    cochran_pvalue: float
    i_squared: float
    pool_estimates: dict[str, float]
    heterogeneous: bool
    pooled_estimate: float
    interaction_stat: float
    interaction_pvalue: float
    forest_plot: tuple[ForestPlotEntry, ...]

    def _payload(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "cochran_q": self.cochran_q,
            "cochran_pvalue": self.cochran_pvalue,
            "i_squared": self.i_squared,
            "pool_estimates": self.pool_estimates,
            "heterogeneous": self.heterogeneous,
            "pooled_estimate": self.pooled_estimate,
            "interaction_stat": self.interaction_stat,
            "interaction_pvalue": self.interaction_pvalue,
            "forest_plot": [
                {
                    "pool": e.pool,
                    "estimate": e.estimate,
                    "ci_lower": e.ci_lower,
                    "ci_upper": e.ci_upper,
                    "weight": e.weight,
                }
                for e in self.forest_plot
            ],
        }

    def summary(self) -> str:
        """Return a human-readable heterogeneity summary."""
        status = "heterogeneous" if self.heterogeneous else "homogeneous"
        lines = [
            f"HeterogeneityReport run_id={self.run_id}",
            f"  metric={self.metric}: Q={self.cochran_q:.4f}, "
            f"p={self.cochran_pvalue:.4g}, I²={self.i_squared:.1f}% ({status})",
            f"  pooled_estimate={self.pooled_estimate:.4f}",
            f"  interaction: LR={self.interaction_stat:.4f}, p={self.interaction_pvalue:.4g}",
            "  forest_plot:",
        ]
        for entry in self.forest_plot:
            lines.append(
                f"    {entry.pool}: {entry.estimate:.4f} "
                f"[{entry.ci_lower:.4f}, {entry.ci_upper:.4f}] w={entry.weight:.3f}"
            )
        return "\n".join(lines)


@dataclass(frozen=True, kw_only=True)
class TransportReport(ReportBase):
    """Transportability assessment across pools."""

    reference_pool: str
    transport_scores: dict[str, float]
    worst_pool: str
    best_pool: str
    mean_transport_score: float
    ipw_estimates: dict[str, float]
    gcomp_estimates: dict[str, float]

    def _payload(self) -> dict[str, Any]:
        return {
            "reference_pool": self.reference_pool,
            "transport_scores": self.transport_scores,
            "worst_pool": self.worst_pool,
            "best_pool": self.best_pool,
            "mean_transport_score": self.mean_transport_score,
            "ipw_estimates": self.ipw_estimates,
            "gcomp_estimates": self.gcomp_estimates,
        }

    def summary(self) -> str:
        """Return a human-readable transportability summary."""
        lines = [
            f"TransportReport run_id={self.run_id}",
            f"  reference_pool: {self.reference_pool}",
            f"  mean_transport_score: {self.mean_transport_score:.4f}",
            f"  best_pool: {self.best_pool}",
            f"  worst_pool: {self.worst_pool}",
        ]
        for pool in sorted(self.transport_scores):
            lines.append(f"    {pool}: score={self.transport_scores[pool]:.4f}")
            if pool in self.ipw_estimates:
                lines.append(f"      ipw_{self.metric_suffix()}: {self.ipw_estimates[pool]:.4f}")
            if pool in self.gcomp_estimates:
                lines.append(
                    f"      gcomp_{self.metric_suffix()}: {self.gcomp_estimates[pool]:.4f}"
                )
        return "\n".join(lines)

    def metric_suffix(self) -> str:
        """Return metric label stored in transport metadata when available."""
        metric = self.metadata.get("metric", "metric")
        return str(metric)


def _pool_metric_to_dict(result: PoolMetricResult) -> dict[str, Any]:
    return {
        "pool": result.pool,
        "n": result.n,
        "metrics": result.metrics,
        "ci": {k: [v[0], v[1]] for k, v in result.ci.items()},
    }


def _dict_to_rows(data: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, value in data.items():
        label = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            rows.extend(_dict_to_rows(value, label))
        elif isinstance(value, list):
            rows.append((label, json.dumps(value, default=_json_fallback)))
        else:
            rows.append((label, str(value)))
    return rows


def _json_fallback(obj: object) -> object:
    if isinstance(obj, tuple):
        return list(obj)
    msg = f"Object of type {type(obj)!r} is not JSON serializable"
    raise TypeError(msg)


def report_digest(report: ReportBase) -> str:
    """Return SHA-256 digest of a report's canonical JSON payload.

    Args:
        report: Any :class:`ReportBase` subclass instance.

    Returns:
        Lowercase hex SHA-256 digest suitable for audit cross-referencing.
    """
    from poolguard.utils import sha256_hex

    return sha256_hex(stable_json_dumps(report.to_dict()))
