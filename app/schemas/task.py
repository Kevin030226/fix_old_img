"""Task-related schemas (plan sections 12 and 14)."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskCreateRequest:
    """Payload for POST /api/v1/tasks (plan section 12)."""

    type: str
    image_path: str
    user: str = "unknown"
    options: dict = field(default_factory=dict)


@dataclass
class TaskView:
    """Read model returned by the task API."""

    id: str
    task_type: str
    status: str
    progress: int = 0
    current_stage: str | None = None
    error_message: str | None = None
    result_path: str | None = None
    evaluation_text: str | None = None
    duration_ms: int | None = None
    stages: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.id,
            "status": self.status,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "stages": self.stages,
            "metrics": self.metrics,
            "result_path": self.result_path,
            "evaluation_text": self.evaluation_text,
        }


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str):
        import json

        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value or {}


def task_row_to_view(row: dict) -> TaskView:
    """Convert a raw tasks-row dict (joined with stages/metrics) into a TaskView."""
    stages = _coerce_json(row.get("stages"))
    metrics = _coerce_json(row.get("metrics"))
    return TaskView(
        id=row.get("id", ""),
        task_type=row.get("task_type", ""),
        status=row.get("status", "queued"),
        progress=int(row.get("progress") or 0),
        current_stage=row.get("current_stage"),
        error_message=row.get("error_message"),
        result_path=row.get("result_path"),
        evaluation_text=row.get("evaluation_text"),
        duration_ms=row.get("duration_ms"),
        stages=stages if isinstance(stages, list) else [],
        metrics=metrics if isinstance(metrics, dict) else {},
    )
