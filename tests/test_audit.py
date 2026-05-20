"""Tests for hash-chained audit trail."""

from __future__ import annotations

from poolguard.audit import AuditTrail
from poolguard.utils import sha256_hex, stable_json_dumps


def test_audit_trail_records_and_verifies() -> None:
    trail = AuditTrail(study_id="study-001")
    trail.record("init", {"version": "0.1.0"})
    trail.record("analyze", {"step": "shift"})
    assert len(trail) == 2
    assert trail.verify()
    assert trail.head_hash() != "GENESIS"


def test_audit_tamper_detection() -> None:
    trail = AuditTrail(study_id="study-002")
    trail.record("action", {"ok": True})
    trail.events[0].payload = {"ok": False}
    assert not trail.verify()


def test_audit_previous_hash_mismatch() -> None:
    trail = AuditTrail(study_id="study-004")
    trail.record("first", {"step": 1})
    trail.record("second", {"step": 2})
    trail.events[1].previous_hash = "tampered"
    assert not trail.verify()


def test_stable_json_is_deterministic() -> None:
    payload = {"b": 2, "a": 1, "nested": {"z": 3, "y": 2}}
    expected = stable_json_dumps({"a": 1, "b": 2, "nested": {"y": 2, "z": 3}})
    assert stable_json_dumps(payload) == expected


def test_sha256_hex() -> None:
    digest = sha256_hex("poolguard")
    assert len(digest) == 64
    assert digest == sha256_hex("poolguard")


def test_export_returns_serializable_events() -> None:
    trail = AuditTrail(study_id="study-003")
    trail.record("run", {"metric": "auc", "value": 0.91})
    exported = trail.export()
    assert isinstance(exported, list)
    assert exported[0]["action"] == "run"
    assert "event_hash" in exported[0]
