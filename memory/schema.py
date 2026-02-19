from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    """
    A single tutoring 'case' stored in memory.
    Keep it structured: it powers retrieval + personalization.
    """
    memory_id: str
    student_id: str
    topic: str
    problem: str
    student_attempt: str
    error_type: str
    teacher_move: str  # e.g., prompting / hint / error_signaling / explanation
    assistant_response: str
    verified: bool
    tags: List[str]
    created_at: str  # ISO timestamp

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MemoryItem":
        # basic backward/forward compatibility
        return MemoryItem(
            memory_id=str(d.get("memory_id", "")),
            student_id=str(d.get("student_id", "anon")),
            topic=str(d.get("topic", "unknown")),
            problem=str(d.get("problem", "")),
            student_attempt=str(d.get("student_attempt", "")),
            error_type=str(d.get("error_type", "unknown")),
            teacher_move=str(d.get("teacher_move", "hint")),
            assistant_response=str(d.get("assistant_response", "")),
            verified=bool(d.get("verified", False)),
            tags=list(d.get("tags", [])),
            created_at=str(d.get("created_at", MemoryItem.now_iso())),
        )
