from __future__ import annotations
from abc import ABC, abstractmethod
from ..verify_report import VerifyReport

class BaseVerifier(ABC):
    name: str = "base"

    @abstractmethod
    def can_handle(self, topic: str, enonce: str, solution: str, final_answer: str | None, check: str | None) -> bool:
        ...

    @abstractmethod
    def verify(self, topic: str, enonce: str, solution: str, final_answer: str | None, check: str | None) -> VerifyReport:
        ...