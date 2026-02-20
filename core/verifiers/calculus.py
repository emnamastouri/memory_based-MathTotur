from __future__ import annotations
import random
import re
import sympy as sp
from core.verify_report import VerifyReport
from core.verifiers.base import BaseVerifier

def _parse_directive(check: str):
    """
    Expected:
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

def _sympify_expr(expr: str, var: sp.Symbol):
    # allow using the variable name in sympify locals
    return sp.sympify(expr, locals={str(var): var})

class CalculusVerifier(BaseVerifier):
    name = "calculus"

    def can_handle(self, topic_or_directive, enonce, solution, final_answer, check):
        # Accept both topic-based and directive-based
        t = (topic_or_directive or "").lower()
        if check:
            head = check.strip().split(";", 1)[0].strip().upper()
            if head in {"DERIVATIVE", "INTEGRAL", "LIMIT"}:
                return final_answer is not None
        return any(k in t for k in ["dériv", "derive", "intégr", "integr", "limite", "limit"]) and (final_answer is not None)

    def verify(self, topic, enonce, solution, final_answer, check):
        rep = VerifyReport(ok=True, kind="mixed", summary="Vérification Calcul (dérivée/intégrale/limite)")
        rep.add("structure.final_answer_present", bool(final_answer), "FINAL_ANSWER présent" if final_answer else "FINAL_ANSWER manquant")
        if not final_answer:
            rep.ok = False
            return rep

        # Parse FINAL_ANSWER
        try:
            ans = sp.sympify(final_answer)
            rep.add("parse.final_answer", True, "FINAL_ANSWER parsé")
        except Exception as e:
            rep.add("parse.final_answer", False, f"FINAL_ANSWER non parsable: {e}")
            rep.ok = False
            return rep

        # If directive provided, use it; else try old style: CHECK is raw expression
        directive = None
        kv = {}
        if check:
            parsed = _parse_directive(check)
            if parsed:
                directive, kv = parsed
                rep.add("parse.check_directive", directive in {"DERIVATIVE","INTEGRAL","LIMIT"}, f"Directive CHECK: {directive}")
            else:
                rep.add("parse.check_directive", True, "CHECK non-directive (mode ancien)")

        # Determine variable
        var_name = kv.get("var", "x")
        x = sp.Symbol(var_name, real=True)

        # ---------------- DERIVATIVE ----------------
        if directive == "DERIVATIVE" or ("dériv" in (topic or "").lower()) or ("derive" in (topic or "").lower()):
            func_str = kv.get("func", None)
            if func_str is None:
                # old mode: CHECK should be expression f(x)
                try:
                    ck = sp.sympify(check) if check else None
                except Exception:
                    ck = None
                if ck is None or isinstance(ck, sp.Equality):
                    rep.add("symbolic.derivative", False, "CHECK doit contenir f(x) ou directive DERIVATIVE; var=..; func=..")
                else:
                    target = sp.diff(ck, x)
                    ok_sym = sp.simplify(target - ans) == 0
                    rep.add("symbolic.derivative", ok_sym, "Symbolique: d/dx f(x) == FINAL_ANSWER")
                    try:
                        f1 = sp.lambdify(x, sp.simplify(target - ans), "numpy")
                        samples = [random.uniform(-3, 3) for _ in range(6)]
                        ok_num = all(abs(complex(f1(s))) < 1e-5 for s in samples)
                        rep.add("numeric.derivative", ok_num, "Numérique: diff ~ 0 sur échantillons")
                    except Exception as e:
                        rep.add("numeric.derivative", False, f"Erreur numeric: {e}")
            else:
                try:
                    f = _sympify_expr(func_str, x)
                    target = sp.diff(f, x)
                    ok_sym = sp.simplify(target - ans) == 0
                    rep.add("symbolic.derivative", ok_sym, "Symbolique: d/dx(func) == FINAL_ANSWER (directive)")
                    f1 = sp.lambdify(x, sp.simplify(target - ans), "numpy")
                    samples = [random.uniform(-3, 3) for _ in range(6)]
                    ok_num = all(abs(complex(f1(s))) < 1e-5 for s in samples)
                    rep.add("numeric.derivative", ok_num, "Numérique: diff ~ 0 sur échantillons")
                except Exception as e:
                    rep.add("symbolic.derivative", False, f"Erreur DERIVATIVE: {e}")

        # ---------------- INTEGRAL ----------------
        if directive == "INTEGRAL" or ("intégr" in (topic or "").lower()) or ("integr" in (topic or "").lower()):
            integrand_str = kv.get("integrand", None)
            if integrand_str is None:
                # old mode: CHECK should be integrand g(x)
                try:
                    ck = sp.sympify(check) if check else None
                except Exception:
                    ck = None
                if ck is None or isinstance(ck, sp.Equality):
                    rep.add("symbolic.integral", False, "CHECK doit contenir integrand ou directive INTEGRAL; var=..; integrand=..")
                else:
                    der = sp.diff(ans, x)
                    ok_sym = sp.simplify(der - ck) == 0
                    rep.add("symbolic.integral", ok_sym, "Symbolique: d/dx(FINAL_ANSWER) == integrand")
                    try:
                        f1 = sp.lambdify(x, sp.simplify(der - ck), "numpy")
                        samples = [random.uniform(-2, 2) for _ in range(6)]
                        ok_num = all(abs(complex(f1(s))) < 1e-5 for s in samples)
                        rep.add("numeric.integral", ok_num, "Numérique: diff ~ 0 sur échantillons")
                    except Exception as e:
                        rep.add("numeric.integral", False, f"Erreur numeric: {e}")
            else:
                try:
                    g = _sympify_expr(integrand_str, x)
                    der = sp.diff(ans, x)
                    ok_sym = sp.simplify(der - g) == 0
                    rep.add("symbolic.integral", ok_sym, "Symbolique: d/dx(FINAL_ANSWER) == integrand (directive)")
                    f1 = sp.lambdify(x, sp.simplify(der - g), "numpy")
                    samples = [random.uniform(-2, 2) for _ in range(6)]
                    ok_num = all(abs(complex(f1(s))) < 1e-5 for s in samples)
                    rep.add("numeric.integral", ok_num, "Numérique: diff ~ 0 sur échantillons")
                except Exception as e:
                    rep.add("symbolic.integral", False, f"Erreur INTEGRAL: {e}")

        # ---------------- LIMIT ----------------
        if directive == "LIMIT" or ("limite" in (topic or "").lower()) or ("limit" in (topic or "").lower()):
            expr_str = kv.get("expr", None)
            point_str = kv.get("point", None)

            if expr_str and point_str is not None:
                try:
                    expr = _sympify_expr(expr_str, x)
                    point = sp.sympify(point_str)
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
            else:
                # Old mode: if CHECK itself is limit(...)
                try:
                    ck = sp.sympify(check) if check else None
                    if ck is not None and hasattr(sp, "limit") and ck.func == sp.limit:
                        ok_sym = sp.simplify(ck - ans) == 0
                        rep.add("symbolic.limit", ok_sym, "Symbolique: limite == FINAL_ANSWER (old mode)")
                    else:
                        rep.add("symbolic.limit", False, "CHECK limit(...) ou directive LIMIT manquante")
                except Exception as e:
                    rep.add("symbolic.limit", False, f"Erreur old limit: {e}")

        rep.ok = all(item.ok for item in rep.items)
        return rep