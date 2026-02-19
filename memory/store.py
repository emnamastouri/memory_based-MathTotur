from __future__ import annotations

import json
import os
from typing import List, Iterable

from memory.schema import MemoryItem


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def append_memory(path: str, item: MemoryItem) -> None:
    """
    Append one MemoryItem to a JSONL file.
    """
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def load_memory(path: str) -> List[MemoryItem]:
    """
    Load all MemoryItems from a JSONL file.
    """
    if not os.path.exists(path):
        return []

    items: List[MemoryItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                items.append(MemoryItem.from_dict(d))
            except json.JSONDecodeError:
                # skip bad lines rather than crashing
                continue
    return items


def overwrite_memory(path: str, items: Iterable[MemoryItem]) -> None:
    """
    Rewrite the JSONL file from scratch (useful after cleanup).
    """
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")