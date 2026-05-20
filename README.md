# poolguard

**A practical tool to detect, measure, and manage patient pool differences in clinical AI.**

---

### A common challenge in clinical AI

You train a model on patients from one hospital, one trial, one region, or one time period.  
It performs well on that data.

When you apply the same model to a new patient pool, performance often drops — sometimes significantly — because the populations differ in important ways.

`poolguard` helps you see these differences clearly, quantify them statistically, and produce auditable results that support better decision-making and regulatory readiness.

---

### What poolguard does

It treats each patient pool (hospital, site, region, time window, demographic group, etc.) as its own distinct statistical population and provides:

- **Population shift detection** — clear statistical metrics (SMD, MMD, KS-test, etc.) with p-values  
- **Per-pool performance evaluation** — AUC, calibration, sensitivity, specificity, and confidence intervals for every pool  
- **Heterogeneity testing** — statistical assessment of whether model performance differs meaningfully across pools  
- **Transportability estimates** — practical prediction of expected performance drop when moving to a new pool  
- **Immutable audit trail** — hash-chained, inspection-ready provenance for every analysis run  

All outputs are designed to be clear, statistically sound, and easy to include in regulatory or quality documentation.

---

### Installation

```bash
pip install poolguard
```

### Quickstart

```python
from poolguard import PoolGuard
import pandas as pd

# 1. Initialize
guard = PoolGuard(pool_column="site_id")   # or "hospital", "region", etc.

# 2. Learn pool fingerprints from your training data
guard.fit(X_train, y_train)

# 3. Evaluate any model on new data
report = guard.evaluate(
    model=your_model,
    X=X_test,
    y=y_test
)

# 4. View results
print(report.summary())

# 5. Export for review or submission
report.to_pdf("pool_analysis_for_submission.pdf")
report.to_json("audit_bundle.json")
```

### When to use poolguard

Use it whenever you need to:

- Move a model from one patient population to another
- Demonstrate to regulators or QA teams that your model is robust across different sites or cohorts
- Understand exactly where and why model performance changes
- Generate auditable, inspection-ready documentation of model behavior across populations

### How it works

You specify which column defines the patient pool (e.g. site_id, hospital, region).

It automatically fingerprints each pool and measures how different they are using established statistical methods.

It evaluates your model separately on every pool and returns clear performance metrics with uncertainty.

It tells you whether the differences are statistically meaningful.

It logs everything with full provenance so the results are transparent and defensible.

No black boxes. Just practical, defensible statistics.

### Why this matters

Most clinical AI models are evaluated on random splits from the same dataset.

poolguard helps you account for real-world differences between patient populations in a structured and auditable way.

It bridges the gap between promising prototypes and tools that can be trusted in clinical and regulatory settings.

Repository: github.com/BouncyMolecules/poolguard  
Author: Ela Halilovic
