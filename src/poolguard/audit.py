"""Immutable hash-chained audit trail for ALCOA+ compliance."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from poolguard.utils import sha256_hex, stable_json_dumps, utc_now_iso


@dataclass
class AuditEvent:
    """Single tamper-evident audit record in a hash chain.

    Attributes:
        event_id: Unique event identifier.
        timestamp: UTC ISO-8601 timestamp.
        action: Semantic action name (e.g. ``"evaluate"``).
        payload: JSON-serializable event body.
        previous_hash: Hash of the preceding event (``"GENESIS"`` for the first).
        event_hash: SHA-256 digest of this event's canonical JSON body.
    """

    event_id: str
    timestamp: str
    action: str
    payload: dict[str, Any]
    previous_hash: str
    event_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
        }


@dataclass
class AuditTrail:
    """Append-only, hash-chained audit log for ALCOA+ traceability.

    Args:
        study_id: Study or deployment identifier included in exports.
    """

    study_id: str
    events: list[AuditEvent] = field(default_factory=list)
    _last_hash: str = field(default="GENESIS", repr=False)

    def record(self, action: str, payload: dict[str, Any]) -> AuditEvent:
        """Append a new audit event and return it.

        Args:
            action: Semantic action label.
            payload: JSON-serializable event details.

        Returns:
            The newly created :class:`AuditEvent`.
        """
        event_id = str(uuid.uuid4())
        timestamp = utc_now_iso()
        body = {
            "event_id": event_id,
            "timestamp": timestamp,
            "action": action,
            "payload": payload,
            "previous_hash": self._last_hash,
        }
        event_hash = sha256_hex(stable_json_dumps(body))
        event = AuditEvent(
            event_id=event_id,
            timestamp=timestamp,
            action=action,
            payload=payload,
            previous_hash=self._last_hash,
            event_hash=event_hash,
        )
        self.events.append(event)
        self._last_hash = event_hash
        return event

    def verify(self) -> bool:
        """Verify hash-chain integrity across all recorded events.

        Returns:
            ``True`` if every ``previous_hash`` link and ``event_hash`` matches.
        """
        previous = "GENESIS"
        for event in self.events:
            if event.previous_hash != previous:
                return False
            body = {
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "action": event.action,
                "payload": event.payload,
                "previous_hash": event.previous_hash,
            }
            expected = sha256_hex(stable_json_dumps(body))
            if event.event_hash != expected:
                return False
            previous = event.event_hash
        return True

    def export(self) -> list[dict[str, Any]]:
        """Export all events as JSON-serializable dictionaries."""
        return [event.to_dict() for event in self.events]

    def head_hash(self) -> str:
        """Return the hash of the most recent event."""
        return self._last_hash

    def __len__(self) -> int:
        return len(self.events)
