from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class CheckItem:
    name: str
    ok: bool
    message: str

@dataclass
class VerifyReport:
    ok: bool
    kind: str                    # "symbolic", "numeric", "structural", "mixed"
    summary: str
    items: List[CheckItem] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, ok: bool, message: str):
        self.items.append(CheckItem(name=name, ok=ok, message=message))
        if not ok:
            self.ok = False