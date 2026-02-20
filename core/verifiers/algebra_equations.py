from __future__ import annotations
import random
import sympy as sp
from core.verify_report import VerifyReport
from core.verifiers.base import BaseVerifier

class AlgebraEquationsVerifier(BaseVerifier):
    name = "algebra_equations"

    def can_handle(self, topic, enonce, solution, final_answer, check):
        t = (topic or "").lower()
        return ("équation" in t or "equation" in t or "système" in t or "systeme" in t or "arithmétique" in t) and (final_answer or check)

    def verify(self, topic, enonce, solution, final_answer, check):
        rep = VerifyReport(ok=True, kind="mixed", summary="Vérification équations / systèmes")
        rep.add("structure.final_answer_present", bool(final_answer), "FINAL_ANSWER présent" if final_answer else "FINAL_ANSWER manquant")

        # Try parse CHECK as Eq(...) if provided
        eq = None
        if check:
            try:
                eq = sp.sympify(check)  # expects Eq(...)
                rep.add("parse.check", True, "CHECK parsé via sympy.sympify")
            except Exception as e:
                rep.add("parse.check", False, f"CHECK non parsable: {e}")

        # Parse final answer
        fa_obj = None
        if final_answer:
            try:
                fa_obj = sp.sympify(final_answer)
                rep.add("parse.final_answer", True, "FINAL_ANSWER parsé via sympy.sympify")
            except Exception:
                # Could be list/dict syntax -> try python eval safely not recommended
                rep.add("parse.final_answer", False, "FINAL_ANSWER non parsable (utilise syntaxe SymPy: 2*x+1, [..], {x:2})")

        # If CHECK is Eq and we have solution candidates -> substitute
        if isinstance(eq, sp.Equality):
            lhs, rhs = eq.lhs, eq.rhs
            expr = sp.simplify(lhs - rhs)

            # Case 1: fa_obj is a list of roots
            if isinstance(fa_obj, (sp.Tuple, list)):
                roots = list(fa_obj)
            elif isinstance(fa_obj, sp.Set):
                roots = list(fa_obj)
            else:
                roots = [fa_obj] if fa_obj is not None else []

            x = None
            # Guess variable
            syms = list(expr.free_symbols)
            if syms:
                x = syms[0]

            if x is not None and roots:
                all_ok = True
                for r in roots:
                    try:
                        val = sp.N(expr.subs({x: r}))
                        ok = abs(complex(val)) < 1e-6
                        all_ok = all_ok and ok
                    except Exception:
                        all_ok = False
                rep.add("symbolic.substitution", all_ok, "Substitution des solutions dans CHECK (Eq)")

                # numeric sampling around
                try:
                    f = sp.lambdify(x, expr, "numpy")
                    samples = [random.uniform(-5, 5) for _ in range(5)]
                    # we just ensure function evaluates without errors
                    _ = [f(s) for s in samples]
                    rep.add("numeric.eval", True, "Évaluation numérique OK (pas d'erreur)")
                except Exception as e:
                    rep.add("numeric.eval", False, f"Évaluation numérique échouée: {e}")
            else:
                rep.add("symbolic.substitution", False, "Impossible de déterminer variable ou solutions")
        else:
            # If no Eq, we can only do basic checks
            rep.add("symbolic.substitution", False, "CHECK Eq(...) absent : vérification exacte limitée")

        rep.ok = all(item.ok for item in rep.items)
        return rep