
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import sympy as sp


@dataclass
class VerificationResult:
    ok: bool
    message: str
    details: Optional[str] = None


def verify_equivalence(expr1: str, expr2: str) -> VerificationResult:
    """
    Verify if two expressions are symbolically equivalent (best-effort).
    Example: "sin(x)**2 + cos(x)**2" vs "1"
    """
    try:
        e1 = sp.sympify(expr1)
        e2 = sp.sympify(expr2)
        diff = sp.simplify(e1 - e2)
        ok = (diff == 0)
        return VerificationResult(ok=ok, message=("Equivalent" if ok else "Not equivalent"), details=str(diff))
    except Exception as e:
        return VerificationResult(ok=False, message="Verification failed (parse/symbolic error)", details=str(e))


def verify_solution_by_substitution(equation: str, solution: str, var: str = "x") -> VerificationResult:
    """
    Verify if `solution` satisfies `equation` by substitution.
    equation example: "x**2 - 4 = 0"
    solution example: "2"
    """
    try:
        x = sp.Symbol(var)
        eq = sp.sympify(equation)
        sol = sp.sympify(solution)
        val = sp.simplify(eq.subs({x: sol}))
        ok = (val == 0)
        return VerificationResult(ok=ok, message=("Solution satisfies equation" if ok else "Solution does NOT satisfy equation"), details=str(val))
    except Exception as e:
        return VerificationResult(ok=False, message="Verification failed (parse/substitution error)", details=str(e))