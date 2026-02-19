from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class Exercise:
    ex_id: str
    grade: str            # metadata.niveau
    section: str          # metadata.section
    topic: str            # type
    enonce: str
    solution: str
    metadata: Dict[str, Any]


def load_exercises(json_path: str) -> List[Exercise]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    exercises: List[Exercise] = []
    for i, item in enumerate(raw):
        meta = item.get("metadata", {}) or {}
        ex = Exercise(
            ex_id=str(item.get("id", f"ex_{i}")),
            grade=str(meta.get("niveau", "")).strip(),
            section=str(meta.get("section", "")).strip(),
            topic=str(item.get("type", "")).strip(),
            enonce=str(item.get("enonce", "")).strip(),
            solution=str(item.get("solution", "")).strip(),
            metadata=meta,
        )
        # keep only valid records
        if ex.grade and ex.section and ex.topic and ex.enonce and ex.solution:
            exercises.append(ex)

    return exercises


def unique_values(exercises: List[Exercise]):
    grades = sorted(set(e.grade for e in exercises))
    sections = sorted(set(e.section for e in exercises))
    topics = sorted(set(e.topic for e in exercises))
    return grades, sections, topics


def add_generated_exercise(db_path: str, item: dict):
    p = Path(db_path)
    data = []
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))

    item = dict(item)
    item["created_at"] = datetime.utcnow().isoformat() + "Z"
    data.append(item)

    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")