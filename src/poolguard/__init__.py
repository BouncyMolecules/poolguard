"""Statistically rigorous patient-pool analysis for clinical ML.

Each pool (site, region, cohort, time window) is treated as its own population.
:class:`PoolGuard` runs shift detection, per-pool evaluation, heterogeneity
testing, and transportability assessment with hash-chained audit logging suitable
for ALCOA+ traceability.

Example:
    >>> from poolguard import PoolGuard, PoolGuardConfig
    >>> pg = PoolGuard(model=clf, config=PoolGuardConfig(bootstrap_n=500))
    >>> pg.fit_reference(X_train, y_train, pools_train, reference_pool="site_a")
    >>> analysis = pg.run_full_analysis(X_test, y_test, pools_test)
    >>> print(analysis.shift.summary())
"""

from poolguard.__version__ import __version__
from poolguard.audit import AuditEvent, AuditTrail
from poolguard.config import PoolGuardConfig
from poolguard.core import PoolGuard, PoolGuardAnalysis
from poolguard.evaluate import evaluate_pools
from poolguard.heterogeneity import (
    analyze_heterogeneity,
    calculate_i_squared,
    cochran_q_test,
    forest_plot_summary,
    pool_interaction_test,
)
from poolguard.report import (
    EvaluateReport,
    ForestPlotEntry,
    HeterogeneityReport,
    PoolEvaluationReport,
    PoolMetricResult,
    ReportBase,
    ShiftFeatureResult,
    ShiftReport,
    TransportReport,
    report_digest,
)
from poolguard.shift import (
    detect_shift,
    kolmogorov_smirnov_test,
    maximum_mean_discrepancy,
    standardized_mean_difference,
)
from poolguard.transport import (
    assess_transportability,
    g_computation_transport,
    inverse_probability_weight,
    transportability_score,
)

__all__ = [
    "AuditEvent",
    "AuditTrail",
    "EvaluateReport",
    "ForestPlotEntry",
    "HeterogeneityReport",
    "PoolEvaluationReport",
    "PoolGuard",
    "PoolGuardAnalysis",
    "PoolGuardConfig",
    "PoolMetricResult",
    "ReportBase",
    "ShiftFeatureResult",
    "ShiftReport",
    "TransportReport",
    "__version__",
    "analyze_heterogeneity",
    "assess_transportability",
    "calculate_i_squared",
    "cochran_q_test",
    "detect_shift",
    "evaluate_pools",
    "forest_plot_summary",
    "g_computation_transport",
    "inverse_probability_weight",
    "kolmogorov_smirnov_test",
    "maximum_mean_discrepancy",
    "pool_interaction_test",
    "report_digest",
    "standardized_mean_difference",
    "transportability_score",
]
