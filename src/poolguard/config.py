"""Configuration objects for PoolGuard analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PoolGuardConfig:
    """Runtime configuration for PoolGuard analyses.

    Attributes:
        smd_threshold: Absolute SMD threshold for covariate shift flagging.
        ks_alpha: Significance level for Kolmogorov-Smirnov tests.
        bootstrap_n: Bootstrap resamples for confidence intervals (>= 100).
        bootstrap_ci: Confidence level for intervals (e.g. 0.95).
        mmd_gamma: RBF bandwidth for MMD; ``None`` uses the median heuristic.
        random_state: Seed for reproducible bootstrap resampling.
        audit_enabled: Whether :class:`~poolguard.core.PoolGuard` records audit events.
        metadata: Study-level metadata merged into every report.

    Raises:
        ValueError: If any threshold or interval setting is out of range.
    """

    smd_threshold: float = 0.1
    ks_alpha: float = 0.05
    bootstrap_n: int = 1000
    bootstrap_ci: float = 0.95
    mmd_gamma: float | None = None
    random_state: int = 42
    audit_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.smd_threshold <= 0.0:
            msg = "smd_threshold must be positive"
            raise ValueError(msg)
        if not 0.0 < self.ks_alpha < 1.0:
            msg = "ks_alpha must be in (0, 1)"
            raise ValueError(msg)
        if self.bootstrap_n < 100:
            msg = "bootstrap_n must be at least 100 for stable CIs"
            raise ValueError(msg)
        if not 0.0 < self.bootstrap_ci < 1.0:
            msg = "bootstrap_ci must be in (0, 1)"
            raise ValueError(msg)
