"""Shared utilities for poolguard."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sized
from datetime import datetime, timezone
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

MetricName = str


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(tz=timezone.utc).isoformat()


def stable_json_dumps(payload: dict[str, Any]) -> str:
    """Serialize a payload to deterministic JSON for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(obj: object) -> object:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    msg = f"Object of type {type(obj)!r} is not JSON serializable"
    raise TypeError(msg)


def sha256_hex(data: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def ensure_dataframe(data: pd.DataFrame | npt.NDArray[np.floating[Any]]) -> pd.DataFrame:
    """Coerce array-like input to a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def ensure_series(
    values: pd.Series[Any] | npt.NDArray[Any],
    *,
    name: str = "value",
) -> pd.Series[Any]:
    """Coerce array-like input to a Series."""
    if isinstance(values, pd.Series):
        return values.copy()
    return pd.Series(values, name=name)


def validate_aligned_lengths(*arrays: Sized) -> None:
    """Raise if inputs do not share the same length."""
    lengths = {len(arr) for arr in arrays}
    if len(lengths) != 1:
        msg = f"All inputs must share the same length; got {lengths}"
        raise ValueError(msg)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric column names."""
    return [str(col) for col in frame.select_dtypes(include=[np.number]).columns]


def predict_proba_positive(model: Any, X: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Obtain positive-class probabilities from a fitted model."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba[:, 1], dtype=np.float64)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return _sigmoid(np.asarray(scores, dtype=np.float64))
    preds = model.predict(X)
    return np.asarray(preds, dtype=np.float64)


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))


def compute_binary_metrics(
    y_true: npt.NDArray[Any],
    y_pred: npt.NDArray[Any],
    y_score: npt.NDArray[np.float64],
) -> dict[MetricName, float]:
    """Compute standard binary classification metrics."""
    metrics: dict[MetricName, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0.0)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["auc"] = float("nan")
    return metrics


def bootstrap_ci(
    values: npt.NDArray[np.float64],
    *,
    ci: float,
) -> tuple[float, float]:
    """Return percentile bootstrap confidence interval bounds."""
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(values, alpha))
    upper = float(np.quantile(values, 1.0 - alpha))
    return lower, upper
