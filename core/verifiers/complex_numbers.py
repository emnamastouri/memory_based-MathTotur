from __future__ import annotations

import sympy as sp
from core.verify_report import VerifyReport
from core.verifiers.base import BaseVerifier


def _normalize_math(s: str) -> str:
    """
    Make user-friendly math safer for sympy:
      - '^' -> '**'  (common in student answers)
      - strip spaces
    """
    if s is None:
        return ""
    return str(s).replace("^", "**").strip()


class ComplexNumbersVerifier(BaseVerifier):
    name = "complex_numbers"

    def can_handle(self, topic, enonce, solution, final_answer, check):
        t = (topic or "").lower()
        # accept both "complex" and French bac wording
        return any(k in t for k in ["complex", "complexe", "affixe", "imaginaire"]) and (final_answer is not None)

    def verify(self, topic, enonce, solution, final_answer, check):
        rep = VerifyReport(ok=True, kind="mixed", summary="Vérification Nombres complexes")

        # ---------- Parse FINAL_ANSWER ----------
        try:
            ans = sp.sympify(
                _normalize_math(final_answer),
                locals={"I": sp.I, "i": sp.I},  # accept I or i
            )
            rep.add("parse.final_answer", True, "FINAL_ANSWER parsé")
        except Exception as e:
            rep.add("parse.final_answer", False, f"FINAL_ANSWER non parsable: {e}")
            rep.ok = False
            return rep

        # ---------- Parse + verify CHECK ----------
        if not check:
            rep.add("parse.check", True, "CHECK absent (OK)")
            rep.ok = all(item.ok for item in rep.items)
            return rep

        try:
            ck = sp.sympify(
                _normalize_math(check),
                locals={"Eq": sp.Eq, "I": sp.I, "i": sp.I},
            )
            rep.add("parse.check", True, "CHECK parsé")
        except Exception as e:
            rep.add("parse.check", False, f"CHECK non parsable: {e}")
            rep.ok = False
            return rep

        # Case 1) Eq(...) survived as an Equality
        if isinstance(ck, sp.Equality):
            ok = sp.simplify(ck.lhs - ck.rhs) == 0
            rep.add("symbolic.check_eq", ok, "CHECK Eq vérifiée")

        # Case 2) SymPy evaluated Eq(...) directly to True/False
        elif ck == sp.S.true:
            rep.add("symbolic.check_bool", True, "CHECK évalué à True par SymPy")
        elif ck == sp.S.false:
            rep.add("symbolic.check_bool", False, "CHECK évalué à False par SymPy")

        # Case 3) Expression supposed to be 0 (e.g. I**2 + 1)
        else:
            ok = sp.simplify(ck) == 0
            rep.add("symbolic.check_expr", ok, "CHECK expression == 0")

        rep.ok = all(item.ok for item in rep.items)
        return rep