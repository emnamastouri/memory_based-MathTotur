from __future__ import annotations
import sympy as sp
from core.verify_report import VerifyReport
from core.verifiers.base import BaseVerifier


class LinearAlgebraVerifier(BaseVerifier):
    name = "linear_algebra"

    def can_handle(self, topic, enonce, solution, final_answer, check):
        t = (topic or "").lower()
        return any(
            k in t
            for k in ["matrice", "matrix", "déterminant", "determinant", "vecteur", "espace"]
        ) and (final_answer is not None)

    def verify(self, topic, enonce, solution, final_answer, check):
        rep = VerifyReport(ok=True, kind="symbolic", summary="Vérification Algèbre linéaire")

        # Parse FINAL_ANSWER
        try:
            ans = sp.sympify(final_answer, locals={"Matrix": sp.Matrix})
            rep.add("parse.final_answer", True, "FINAL_ANSWER parsé")
        except Exception as e:
            rep.add("parse.final_answer", False, f"FINAL_ANSWER non parsable: {e}")
            rep.ok = False
            return rep

        if check:
            try:
                ck = sp.sympify(check, locals={"Eq": sp.Eq, "Matrix": sp.Matrix})
                rep.add("parse.check", True, "CHECK parsé")

                if isinstance(ck, sp.Equality):
                    ok = sp.simplify(ck.lhs - ck.rhs) == 0
                    rep.add("symbolic.check_eq", ok, "CHECK Eq vérifiée symboliquement")
                else:
                    # If CHECK is expression expected to be zero
                    ok = sp.simplify(ck) == 0
                    rep.add("symbolic.check_expr", ok, "CHECK expression == 0")

            except Exception as e:
                rep.add("parse.check", False, f"CHECK non parsable: {e}")
        else:
            rep.add("parse.check", True, "CHECK absent (OK)")

        rep.ok = all(item.ok for item in rep.items)
        return rep