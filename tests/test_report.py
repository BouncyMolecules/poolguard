"""Tests for report serialization and summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from poolguard.config import PoolGuardConfig
from poolguard.evaluate import evaluate_pools
from poolguard.report import PoolEvaluationReport, report_digest
from poolguard.shift import detect_shift
from poolguard.utils import ensure_dataframe, ensure_series, predict_proba_positive


def test_report_to_dict_and_digest() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(0, 1, 100)})
    pools = pd.Series(["a"] * 50 + ["b"] * 50)
    report = detect_shift(X, pools, reference_pool="a")
    payload = report.to_dict()
    assert payload["report_type"] == "ShiftReport"
    digest = report_digest(report)
    assert len(digest) == 64


def test_evaluate_report_summary() -> None:
    rng = np.random.default_rng(1)
    n = 120
    X = pd.DataFrame({"x": rng.normal(0, 1, n)})
    y = pd.Series((X["x"] > 0).astype(int))
    pools = pd.Series(rng.choice(["a", "b"], n))
    model = LogisticRegression(max_iter=200).fit(X, y)
    report = evaluate_pools(model, X, y, pools, config=PoolGuardConfig(bootstrap_n=100))
    summary = report.summary()
    assert "EvaluateReport" in summary
    assert "accuracy" in summary
    assert isinstance(report, PoolEvaluationReport)


def test_report_to_json_and_html(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"x": rng.normal(0, 1, 100)})
    pools = pd.Series(["a"] * 50 + ["b"] * 50)
    report = detect_shift(X, pools, reference_pool="a")
    json_path = tmp_path / "shift.json"
    html_path = tmp_path / "shift.html"
    payload = report.to_json(json_path)
    assert json_path.exists()
    assert json.loads(payload)["report_type"] == "ShiftReport"
    html = report.to_html(html_path)
    assert html_path.exists()
    assert "<html>" in html
    assert "ShiftReport" in html


def test_report_to_pdf(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"x": rng.normal(0, 1, 100)})
    pools = pd.Series(["a"] * 50 + ["b"] * 50)
    report = detect_shift(X, pools, reference_pool="a")
    pdf_path = tmp_path / "shift.pdf"
    report.to_pdf(pdf_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_utils_helpers_and_decision_function_path() -> None:
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"x": rng.normal(0, 1, 40)})
    y = pd.Series((X["x"] > 0).astype(int))
    from sklearn.svm import LinearSVC

    model = LinearSVC(dual="auto", random_state=0).fit(X, y)
    scores = predict_proba_positive(model, X)
    assert len(scores) == len(X)
    frame = ensure_dataframe(X.to_numpy())
    series = ensure_series(y.to_numpy(), name="y")
    assert len(frame) == len(series)
