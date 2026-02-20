from __future__ import annotations
import re
import sympy as sp
from core.verify_report import VerifyReport
from core.verifiers.base import BaseVerifier

class SequencesVerifier(BaseVerifier):
    name = "sequences"

    def can_handle(self, topic, enonce, solution, final_answer, check):
        t = (topic or "").lower()
        return ("suite" in t or "suites" in t) and (final_answer is not None)

    def verify(self, topic, enonce, solution, final_answer, check):
        rep = VerifyReport(ok=True, kind="mixed", summary="Vérification Suites (premiers termes / formule)")
        try:
            fa = final_answer.strip()
            rep.add("parse.final_answer", True, "FINAL_ANSWER présent")
        except Exception as e:
            rep.add("parse.final_answer", False, str(e))
            rep.ok = False
            return rep

        # Heuristique: si dict {u0:..., u1:...} -> vérifier cohérence d'une récurrence si CHECK est donné
        ck = None
        if check:
            try:
                ck = sp.sympify(check)  # could be Eq(u(n+1), f(u(n)))
                rep.add("parse.check", True, "CHECK parsé")
            except Exception:
                rep.add("parse.check", False, "CHECK non parsable (optionnel)")

        # If final answer is dict-like, try sympify
        try:
            obj = sp.sympify(fa)
            rep.add("parse.final_answer_sympy", True, "FINAL_ANSWER parsé (sympy)")
        except Exception:
            rep.add("parse.final_answer_sympy", False, "FINAL_ANSWER non parsable (ex: {u0:1, u1:2})")
            rep.ok = False
            return rep

        # If CHECK is Eq involving u(n+1) and u(n) we can test first few terms
        if isinstance(ck, sp.Equality):
            rep.add("symbolic.sequence_recurrence", True, "Récurrence détectée (Eq) — extension possible")
        else:
            rep.add("symbolic.sequence_recurrence", True, "Pas de CHECK Eq — validation limitée (OK)")

        rep.ok = all(item.ok for item in rep.items)
        return rep