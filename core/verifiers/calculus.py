from __future__ import annotations

import random
import sympy as sp

from core.verify_report import VerifyReport
from core.verifiers.base import BaseVerifier


def _parse_directive(check: str):
    """
    Expected directives:
      DERIVATIVE; var=x; func=x**3+2*x
      INTEGRAL; var=x; integrand=x**2+1
      LIMIT; var=x; expr=sin(x)/x; point=0
    """
    if not check:
        return None
    parts = [p.strip() for p in check.split(";")]
    head = parts[0].upper()
    kv = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k.strip().lower()] = v.strip()
    return head, kv


def _sympify_expr(expr: str, x: sp.Symbol):
    # Parse expression using the *same* Symbol instance used by the verifier
    return sp.sympify(expr, locals={str(x): x, "log": sp.log, "sqrt": sp.sqrt})


def _safe_samples_for_expr(expr_str: str, n: int = 6):
    """
    Choose numeric samples that avoid domain errors for common functions.
    - log(x), sqrt(x): sample x > 0
    """
    s = (expr_str or "").lower()
    if "log(" in s or "sqrt(" in s:
        return [random.uniform(0.2, 3.0) for _ in range(n)]
    return [random.uniform(-3, 3) for _ in range(n)]


class CalculusVerifier(BaseVerifier):
    name = "calculus"

    def can_handle(self, topic_or_directive, enonce, solution, final_answer, check):
        # Accept both topic-based and directive-based
        t = (topic_or_directive or "").lower()
        if check:
            head = check.strip().split(";", 1)[0].strip().upper()
            if head in {"DERIVATIVE", "INTEGRAL", "LIMIT"}:
                return final_answer is not None
        return (
            any(k in t for k in ["dériv", "derive", "intégr", "integr", "limite", "limit"])
            and (final_answer is not None)
        )

    def verify(self, topic, enonce, solution, final_answer, check):
        rep = VerifyReport(ok=True, kind="mixed", summary="Vérification Calcul (dérivée/intégrale/limite)")
        rep.add(
            "structure.final_answer_present",
            bool(final_answer),
            "FINAL_ANSWER présent" if final_answer else "FINAL_ANSWER manquant",
        )
        if not final_answer:
            rep.ok = False
            return rep

        # 1) Parse directive first (so we know which variable to use)
        directive = None
        kv = {}
        if check:
            parsed = _parse_directive(check)
            if parsed:
                directive, kv = parsed
                rep.add(
                    "parse.check_directive",
                    directive in {"DERIVATIVE", "INTEGRAL", "LIMIT"},
                    f"Directive CHECK: {directive}",
                )
            else:
                rep.add("parse.check_directive", True, "CHECK non-directive (mode ancien)")

        # 2) Create the variable symbol (must exist BEFORE parsing FINAL_ANSWER)
        var_name = kv.get("var", "x")
        x = sp.Symbol(var_name, real=True)

        # Infer derivative order from enonce (f'(x) vs f''(x))
        en = (enonce or "").replace(" ", "").lower()
        deriv_order = 2 if ("f''" in en or "f’’" in en or "secondederivee" in en or "seconddérivée" in en) else 1

        # 3) Parse FINAL_ANSWER with SAME symbol x
        try:
            ans = sp.sympify(final_answer, locals={str(x): x, "log": sp.log, "sqrt": sp.sqrt})
            rep.add("parse.final_answer", True, "FINAL_ANSWER parsé")
        except Exception as e:
            rep.add("parse.final_answer", False, f"FINAL_ANSWER non parsable: {e}")
            rep.ok = False
            return rep

        # ---------------- DERIVATIVE ----------------
        if directive == "DERIVATIVE" or ("dériv" in (topic or "").lower()) or ("derive" in (topic or "").lower()):
            func_str = kv.get("func")

            # Preferred directive mode: DERIVATIVE; var=..; func=..
            if func_str:
                try:
                    f = _sympify_expr(func_str, x)
                    target = sp.diff(f, x, deriv_order)

                    ok_sym = sp.simplify(target - ans) == 0
                    rep.add("symbolic.derivative", ok_sym, "Symbolique: d^n/dx^n(func) == FINAL_ANSWER (directive)")

                    try:
                        f1 = sp.lambdify(x, sp.simplify(target - ans), "numpy")
                        samples = _safe_samples_for_expr(func_str, 6)
                        ok_num = all(abs(complex(f1(s))) < 1e-5 for s in samples)
                        rep.add("numeric.derivative", ok_num, "Numérique: diff ~ 0 sur échantillons")
                    except Exception as e:
                        rep.add("numeric.derivative", False, f"Erreur numeric: {e}")

                except Exception as e:
                    rep.add("symbolic.derivative", False, f"Erreur DERIVATIVE: {e}")

            # Old mode: CHECK should be a function f(x) (NOT Eq(...))
            else:
                try:
                    ck = None
                    if check:
                        s = check.strip()
                        # if someone mistakenly wrote "DERIVATIVE; ..." but without func=, treat as invalid old-mode
                        if s.upper().startswith("DERIVATIVE;"):
                            ck = None
                        else:
                            ck = sp.sympify(s, locals={str(x): x, "log": sp.log, "sqrt": sp.sqrt})

                    if ck is None or isinstance(ck, sp.Equality):
                        rep.add("symbolic.derivative", False, "CHECK doit contenir f(x) ou directive DERIVATIVE; var=..; func=..")
                    else:
                        target = sp.diff(ck, x, deriv_order)
                        ok_sym = sp.simplify(target - ans) == 0
                        rep.add("symbolic.derivative", ok_sym, "Symbolique: d^n/dx^n(CHECK) == FINAL_ANSWER (old mode)")

                        try:
                            f1 = sp.lambdify(x, sp.simplify(target - ans), "numpy")
                            samples = _safe_samples_for_expr(str(ck), 6)
                            ok_num = all(abs(complex(f1(s))) < 1e-5 for s in samples)
                            rep.add("numeric.derivative", ok_num, "Numérique: diff ~ 0 sur échantillons")
                        except Exception as e:
                            rep.add("numeric.derivative", False, f"Erreur numeric: {e}")

                except Exception as e:
                    rep.add("symbolic.derivative", False, f"Erreur DERIVATIVE old mode: {e}")

        # ---------------- INTEGRAL ----------------
        if directive == "INTEGRAL" or ("intégr" in (topic or "").lower()) or ("integr" in (topic or "").lower()):
            integrand_str = kv.get("integrand")

            # Preferred directive mode: INTEGRAL; var=..; integrand=..
            if integrand_str:
                try:
                    g = _sympify_expr(integrand_str, x)
                    der = sp.diff(ans, x)

                    ok_sym = sp.simplify(der - g) == 0
                    rep.add("symbolic.integral", ok_sym, "Symbolique: d/dx(FINAL_ANSWER) == integrand (directive)")

                    try:
                        f1 = sp.lambdify(x, sp.simplify(der - g), "numpy")
                        samples = _safe_samples_for_expr(integrand_str, 6)
                        ok_num = all(abs(complex(f1(s))) < 1e-5 for s in samples)
                        rep.add("numeric.integral", ok_num, "Numérique: diff ~ 0 sur échantillons")
                    except Exception as e:
                        rep.add("numeric.integral", False, f"Erreur numeric: {e}")

                except Exception as e:
                    rep.add("symbolic.integral", False, f"Erreur INTEGRAL: {e}")

            # Old mode: CHECK should be integrand g(x) (NOT Eq(...))
            else:
                try:
                    ck = None
                    if check:
                        s = check.strip()
                        if s.upper().startswith("INTEGRAL;"):
                            ck = None
                        else:
                            ck = sp.sympify(s, locals={str(x): x, "log": sp.log, "sqrt": sp.sqrt})

                    if ck is None or isinstance(ck, sp.Equality):
                        rep.add("symbolic.integral", False, "CHECK doit contenir integrand ou directive INTEGRAL; var=..; integrand=..")
                    else:
                        der = sp.diff(ans, x)
                        ok_sym = sp.simplify(der - ck) == 0
                        rep.add("symbolic.integral", ok_sym, "Symbolique: d/dx(FINAL_ANSWER) == CHECK (old mode)")

                        try:
                            f1 = sp.lambdify(x, sp.simplify(der - ck), "numpy")
                            samples = _safe_samples_for_expr(str(ck), 6)
                            ok_num = all(abs(complex(f1(s))) < 1e-5 for s in samples)
                            rep.add("numeric.integral", ok_num, "Numérique: diff ~ 0 sur échantillons")
                        except Exception as e:
                            rep.add("numeric.integral", False, f"Erreur numeric: {e}")

                except Exception as e:
                    rep.add("symbolic.integral", False, f"Erreur INTEGRAL old mode: {e}")

        # ---------------- LIMIT ----------------
        if directive == "LIMIT" or ("limite" in (topic or "").lower()) or ("limit" in (topic or "").lower()):
            expr_str = kv.get("expr")
            point_str = kv.get("point")

            # Preferred directive mode: LIMIT; var=..; expr=..; point=..
            if expr_str and (point_str is not None):
                try:
                    expr = _sympify_expr(expr_str, x)
                    point = sp.sympify(point_str, locals={str(x): x, "log": sp.log, "sqrt": sp.sqrt})
                    lim = sp.limit(expr, x, point)

                    ok_sym = sp.simplify(lim - ans) == 0
                    rep.add("symbolic.limit", ok_sym, f"Symbolique: limit(expr,x->{point}) == FINAL_ANSWER")

                    # numeric approach from both sides
                    try:
                        f_num = sp.lambdify(x, expr, "numpy")
                        p = float(sp.N(point))
                        eps_list = [1e-2, 1e-3, 1e-4]
                        vals = []
                        for eps in eps_list:
                            vals.append(f_num(p + eps))
                            vals.append(f_num(p - eps))
                        target = float(sp.N(ans))
                        ok_num = all(abs(complex(v - target)) < 1e-2 for v in vals)
                        rep.add("numeric.limit", ok_num, "Numérique: approche ±ε cohérente")
                    except Exception as e:
                        rep.add("numeric.limit", False, f"Erreur numeric limit: {e}")

                except Exception as e:
                    rep.add("symbolic.limit", False, f"Erreur LIMIT: {e}")

            # Old mode: CHECK itself is "limit(expr, x, point)" OR "LIMIT; limit(expr, x, point)"
            else:
                try:
                    ck = None
                    if check:
                        s = check.strip()
                        if s.upper().startswith("LIMIT;"):
                            s = s.split(";", 1)[1].strip()
                        ck = sp.sympify(s, locals={str(x): x, "limit": sp.limit, "log": sp.log, "sqrt": sp.sqrt}) if s else None

                    if ck is None:
                       rep.add("symbolic.limit", False, "CHECK limit(...) ou directive LIMIT manquante")
                    else:
    # ck may be:
    # - a Limit object (Limit(expr, x, point))
    # - a call result (e.g., 1) if sympy evaluated limit(...)
    # - something else
                      try:
                       if isinstance(ck, sp.Limit):
                         ck_val = ck.doit()
                       else:
                           ck_val = ck
                           ok_sym = sp.simplify(ck_val - ans) == 0
                           rep.add("symbolic.limit", ok_sym, "Symbolique: limite == FINAL_ANSWER (old mode)")
                      except Exception as e:
                       rep.add("symbolic.limit", False, f"Erreur old limit: {e}")
                except Exception as e:
                    rep.add("symbolic.limit", False, f"Erreur old limit: {e}")

        rep.ok = all(item.ok for item in rep.items)
        return rep

