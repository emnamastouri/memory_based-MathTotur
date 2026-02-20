from __future__ import annotations
import random
import sympy as sp
from core.verify_report import VerifyReport
from core.verifiers.base import BaseVerifier

class CalculusVerifier(BaseVerifier):
    name = "calculus"

    def can_handle(self, topic, enonce, solution, final_answer, check):
        t = (topic or "").lower()
        return any(k in t for k in ["dériv", "derive", "intégr", "integr", "limite", "limit"]) and (final_answer is not None)

    def verify(self, topic, enonce, solution, final_answer, check):
        rep = VerifyReport(ok=True, kind="mixed", summary="Vérification Calcul (dérivée/intégrale/limite)")
        rep.add("structure.final_answer_present", True, "FINAL_ANSWER présent")

        t = (topic or "").lower()
        x = sp.Symbol("x", real=True)

        # Parse answer expression
        try:
            ans = sp.sympify(final_answer)
            rep.add("parse.final_answer", True, "FINAL_ANSWER parsé")
        except Exception as e:
            rep.add("parse.final_answer", False, f"FINAL_ANSWER non parsable: {e}")
            rep.ok = False
            return rep

        # If CHECK provided, it can guide (e.g., derivative of f(x) or integral of g(x))
        ck = None
        if check:
            try:
                ck = sp.sympify(check)
                rep.add("parse.check", True, "CHECK parsé")
            except Exception:
                rep.add("parse.check", False, "CHECK non parsable (optionnel)")

        # Derivative verification:
        if "dériv" in t or "derive" in t:
            # Expect CHECK to contain original function f(x) as expression, e.g. "x**3 + 2*x"
            if ck is None or isinstance(ck, sp.Equality):
                rep.add("symbolic.derivative", False, "CHECK doit être une expression f(x) (pas Eq) pour vérifier la dérivée")
            else:
                try:
                    target = sp.diff(ck, x)
                    ok_sym = sp.simplify(target - ans) == 0
                    rep.add("symbolic.derivative", ok_sym, f"Symbolique: d/dx f(x) == FINAL_ANSWER")
                    # numeric sampling
                    f1 = sp.lambdify(x, sp.simplify(target - ans), "numpy")
                    samples = [random.uniform(-3, 3) for _ in range(6)]
                    ok_num = all(abs(complex(f1(s))) < 1e-5 for s in samples)
                    rep.add("numeric.derivative", ok_num, "Numérique: diff ~ 0 sur échantillons")
                except Exception as e:
                    rep.add("symbolic.derivative", False, f"Erreur dérivée: {e}")

        # Integral verification:
        if "intégr" in t or "integr" in t:
            # Expect CHECK to be integrand g(x)
            if ck is None or isinstance(ck, sp.Equality):
                rep.add("symbolic.integral", False, "CHECK doit être une expression g(x) (pas Eq) pour vérifier l'intégrale")
            else:
                try:
                    # verify derivative of answer equals integrand
                    der = sp.diff(ans, x)
                    ok_sym = sp.simplify(der - ck) == 0
                    rep.add("symbolic.integral", ok_sym, "Symbolique: d/dx(FINAL_ANSWER) == integrand")
                    f1 = sp.lambdify(x, sp.simplify(der - ck), "numpy")
                    samples = [random.uniform(-2, 2) for _ in range(6)]
                    ok_num = all(abs(complex(f1(s))) < 1e-5 for s in samples)
                    rep.add("numeric.integral", ok_num, "Numérique: diff ~ 0 sur échantillons")
                except Exception as e:
                    rep.add("symbolic.integral", False, f"Erreur intégrale: {e}")

        # Limit verification (mostly numeric unless CHECK is given)
        if "limite" in t or "limit" in t:
            # CHECK can be like "limit(x*sin(x), x, 0)" or "x*sin(x),0"
            try:
                if ck is not None and hasattr(sp, "limit"):
                    # if check itself is limit(...) return compare
                    if ck.func == sp.limit:
                        lim_val = ck
                    else:
                        lim_val = None
                    if lim_val is not None:
                        ok_sym = sp.simplify(lim_val - ans) == 0
                        rep.add("symbolic.limit", ok_sym, "Symbolique: limite == FINAL_ANSWER")
                else:
                    rep.add("symbolic.limit", True, "Limite: pas de CHECK exploitable (skip symbolique)")
            except Exception:
                rep.add("symbolic.limit", False, "Erreur calcul limite")

        rep.ok = all(item.ok for item in rep.items)
        return rep