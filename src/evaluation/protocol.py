"""Typed, JSON-serializable protocol for optional scientific metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class MetricStatus(str, Enum):
    OK = "ok"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True)
class MetricResult:
    """One metric result that never hides missing tools behind a numeric zero."""

    name: str
    status: MetricStatus
    value: float | None = None
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    implementation: str = ""

    @classmethod
    def ok(
        cls,
        name: str,
        value: float,
        *,
        details: Mapping[str, Any] | None = None,
        implementation: str = "",
    ) -> "MetricResult":
        return cls(
            name=name,
            status=MetricStatus.OK,
            value=float(value),
            details=dict(details or {}),
            implementation=implementation,
        )

    @classmethod
    def skipped(
        cls,
        name: str,
        reason: str,
        *,
        details: Mapping[str, Any] | None = None,
        implementation: str = "",
    ) -> "MetricResult":
        return cls(
            name=name,
            status=MetricStatus.SKIPPED,
            reason=reason,
            details=dict(details or {}),
            implementation=implementation,
        )

    @classmethod
    def error(
        cls,
        name: str,
        reason: str,
        *,
        details: Mapping[str, Any] | None = None,
        implementation: str = "",
    ) -> "MetricResult":
        return cls(
            name=name,
            status=MetricStatus.ERROR,
            reason=reason,
            details=dict(details or {}),
            implementation=implementation,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload
