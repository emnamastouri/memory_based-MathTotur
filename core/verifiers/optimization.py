from __future__ import annotations

import ast
import re
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import sympy as sp

from core.verify_report import VerifyReport
from core.verifiers.base import BaseVerifier


# -----------------------------
# Parsing helpers
# -----------------------------

def _smart_parse_dict(text: str) -> Optional[dict]:
    """
    Parse FINAL_ANSWER robustly:
    Accepts:
      {x_star: 3, f_star: 2}
      {"x_star": 3, "f_star": 2}
      {x: 3, y: 4, z: 10, f_star: 12}
    Also tolerates extra spaces/newlines.
    """
    if not text:
        return None
    s = text.strip()

    # 1) try sympy
    try:
        obj = sp.sympify(s)
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "items"):
            return dict(obj.items())
    except Exception:
        pass

    # 2) try python literal eval (safe)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3) last chance: extract {...} block inside text
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        block = m.group(0)
        try:
            obj = ast.literal_eval(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = sp.sympify(block)
            if hasattr(obj, "items"):
                return dict(obj.items())
        except Exception:
            pass

    return None


def _get_key(d: dict, *names: str):
    """
    keys may be:
      Symbol('x_star') or 'x_star'
    """
    for n in names:
        for k in d.keys():
            if str(k) == n:
                return d[k]
    return None


def _parse_optimize_check(check: str):
    """
    Supports:
    OPTIMIZE; var=z; func=...; domain=[0,20]; goal=max
    OPTIMIZE; var=[x,y,z]; func=...; domain=[2x+3y+z<=100, x+2y+z<=70, x>=0, ...]; goal=min
    """
    parts = [p.strip() for p in check.split(";")]
    if not parts or not parts[0].upper().startswith("OPTIMIZE"):
        return None

    kv: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k.strip().lower()] = v.strip()

    var = kv.get("var", "x")
    func = kv.get("func", None)
    goal = kv.get("goal", "max").lower()
    dom = kv.get("domain", None)
    return var, func, dom, goal


def _parse_var_list(var_field: str) -> List[str]:
    """
    var can be:
      x
      [x,y,z]
    """
    v = (var_field or "x").strip()
    if v.startswith("[") and v.endswith("]"):
        inside = v[1:-1].strip()
        if not inside:
            return ["x"]
        return [a.strip() for a in inside.split(",") if a.strip()]
    return [v]


def _parse_domain_1d(dom: str) -> Optional[Tuple[float, float]]:
    # domain like [0,20] or [-1, 2]
    m = re.match(r"^\[\s*([-+]?\d+(\.\d+)?)\s*,\s*([-+]?\d+(\.\d+)?)\s*\]$", dom.strip())
    if not m:
        return None
    a = float(m.group(1))
    b = float(m.group(3))
    return (min(a, b), max(a, b))


def _parse_domain_constraints(dom: str) -> Optional[List[str]]:
    """
    domain like:
    [2x+3y+z<=100, x+2y+z<=70, x>=0, y>=0, z>=10, x<=20]
    Returns list of constraint strings.
    """
    s = dom.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    inside = s[1:-1].strip()
    if not inside:
        return []
    # split by comma, but keep it simple (your constraints are simple)
    return [c.strip() for c in inside.split(",") if c.strip()]


def _constraint_to_expr(c: str, locals_map: dict) -> Optional[sp.Expr]:
    """
    Convert constraint like:
      2*x+3*y+z<=100  ->  2*x+3*y+z-100 <= 0  => expr = 2*x+3*y+z-100
      x>=0           ->  0-x <= 0           => expr = -x
      x<=20          ->  x-20 <= 0          => expr = x-20
    Returns expr such that expr <= 0 must hold.
    """
    c = c.replace("≤", "<=").replace("≥", ">=")
    if "<=" in c:
        left, right = c.split("<=", 1)
        try:
            return sp.sympify(left, locals=locals_map) - sp.sympify(right, locals=locals_map)
        except Exception:
            return None
    if ">=" in c:
        left, right = c.split(">=", 1)
        try:
            # left >= right  <=>  right - left <= 0
            return sp.sympify(right, locals=locals_map) - sp.sympify(left, locals=locals_map)
        except Exception:
            return None
    if "=" in c:
        left, right = c.split("=", 1)
        try:
            return sp.Abs(sp.sympify(left, locals=locals_map) - sp.sympify(right, locals=locals_map))
        except Exception:
            return None
    return None


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(sp.N(x))
    except Exception:
        try:
            return float(x)
        except Exception:
            return None


# -----------------------------
# Verifier
# -----------------------------

class OptimizationVerifier(BaseVerifier):
    name = "optimization"

    def can_handle(self, topic, enonce, solution, final_answer, check):
        if check and isinstance(check, str) and check.strip().upper().startswith("OPTIMIZE"):
            return True
        t = (topic or "").lower()
        return any(k in t for k in ["maximum", "minimum", "optim", "extrem", "variation"])

    def verify(self, topic, enonce, solution, final_answer, check):
        rep = VerifyReport(ok=True, kind="mixed", summary="Vérification Optimisation (symbolique + numérique)")

        if not check:
            rep.add("opt.check_present", False, "CHECK manquant (OPTIMIZE; ... requis)")
            rep.ok = False
            return rep

        parsed = _parse_optimize_check(check.strip())
        if not parsed:
            rep.add("opt.check_format", False, "CHECK invalide. Attendu: OPTIMIZE; var=...; func=...; domain=...; goal=max|min")
            rep.ok = False
            return rep

        var_field, func_str, dom_str, goal = parsed
        rep.add("opt.check_format", True, "CHECK OPTIMIZE parsé")

        if not func_str:
            rep.add("opt.func_present", False, "func=... manquant dans CHECK")
            rep.ok = False
            return rep

        var_names = _parse_var_list(var_field)
        symbols = [sp.Symbol(v, real=True) for v in var_names]
        locals_map = {v: s for v, s in zip(var_names, symbols)}

        # Parse function
        try:
            f = sp.sympify(func_str, locals=locals_map)
            rep.add("opt.func_parse", True, "Fonction parsée par SymPy")
        except Exception as e:
            rep.add("opt.func_parse", False, f"Impossible de parser func: {e}")
            rep.ok = False
            return rep

        rep.details["vars"] = var_names
        rep.details["f"] = str(f)
        rep.details["goal"] = goal

        # Parse FINAL_ANSWER dict
        if not final_answer:
            rep.add("opt.final_answer_present", False, "FINAL_ANSWER manquant")
            rep.ok = False
            return rep

        d = _smart_parse_dict(final_answer)
        if d is None:
            rep.add("opt.final_answer_dict", False, "FINAL_ANSWER doit être un dict (ex: {x_star:..., f_star:...})")
            rep.ok = False
            return rep
        rep.add("opt.final_answer_dict", True, "FINAL_ANSWER dict OK")

        # Extract claimed x*/y*/z* (support both *_star and plain var keys)
        claimed_point: Dict[sp.Symbol, Any] = {}
        for v in var_names:
            s = locals_map[v]
            val = _get_key(d, f"{v}_star", v, f"arg{v}", f"{v}*")
            if val is not None:
                claimed_point[s] = val

        # Also accept single key x_star for 1D
        if len(var_names) == 1 and locals_map[var_names[0]] not in claimed_point:
            x_star_claimed = _get_key(d, "x_star", "argmax", "argmin", "x")
            if x_star_claimed is not None:
                claimed_point[locals_map[var_names[0]]] = x_star_claimed

        f_star_claimed = _get_key(d, "f_star", "max", "min", "f_max", "f_min", "value")

        rep.add("opt.keys_point", len(claimed_point) == len(var_names), f"Point détecté: { {str(k): str(v) for k,v in claimed_point.items()} }")
        rep.add("opt.keys_f_star", f_star_claimed is not None, "Clé f_star/max/min détectée" if f_star_claimed is not None else "Manque f_star")

        # -----------------------------
        # Parse domain (1D interval or constraints list)
        # -----------------------------
        dom_interval = None
        dom_constraints = None

        if dom_str:
            # try 1D interval first
            dom_interval = _parse_domain_1d(dom_str) if len(var_names) == 1 else None
            if dom_interval is not None:
                rep.add("opt.domain_parse", True, f"Domaine intervalle: {dom_interval}")
            else:
                dom_constraints = _parse_domain_constraints(dom_str)
                rep.add(
                    "opt.domain_parse",
                    dom_constraints is not None,
                    f"Domaine contraintes: {len(dom_constraints)} contraintes" if dom_constraints is not None else "Domaine illisible",
                )
        else:
            rep.add("opt.domain_parse", True, "Pas de domaine fourni")

        # -----------------------------
        # Feasibility check (constraints)
        # -----------------------------
        if dom_constraints is not None and claimed_point:
            ok_all = True
            parsed_count = 0
            failed = []
            for c in dom_constraints:
                expr = _constraint_to_expr(c, locals_map)
                if expr is None:
                    continue
                parsed_count += 1
                try:
                    val = sp.N(expr.subs(claimed_point))
                    v = complex(val)
                    ok = (v.real <= 1e-6)  # expr <= 0
                    if not ok:
                        ok_all = False
                        failed.append((c, float(v.real)))
                except Exception:
                    ok_all = False
                    failed.append((c, None))
            rep.add("opt.constraints_parsed", parsed_count > 0, f"{parsed_count} contraintes parsées")
            rep.add("opt.feasible", ok_all, f"Faisable" if ok_all else f"Non faisable: {failed[:3]}")
        else:
            rep.add("opt.feasible", True, "Pas de contraintes à vérifier (ou point manquant)")

        # -----------------------------
        # Evaluate f at claimed point (numeric)
        # -----------------------------
        f_claim_val = None
        if claimed_point:
            try:
                f_claim_val = sp.N(f.subs(claimed_point))
                rep.add("opt.eval_f_claimed", True, f"f(point)= {f_claim_val}")
            except Exception as e:
                rep.add("opt.eval_f_claimed", False, f"Impossible d'évaluer f au point: {e}")
        else:
            rep.add("opt.eval_f_claimed", False, "Point (x*/y*/z*) manquant dans FINAL_ANSWER")

        # Compare f_star if provided
        if f_star_claimed is not None and f_claim_val is not None:
            fc = _to_float(f_claim_val)
            fs = _to_float(f_star_claimed)
            if fc is not None and fs is not None:
                rep.add("opt.compare_f_star", abs(fc - fs) < 1e-2, f"f_star: claimed={fs}, computed={fc}")
            else:
                rep.add("opt.compare_f_star", False, "Comparaison f_star impossible (non numérique)")
        else:
            rep.add("opt.compare_f_star", True, "Comparaison f_star ignorée (manquant)")

        # -----------------------------
        # 1D exact check (critical points + endpoints)
        # -----------------------------
        if len(var_names) == 1:
            x = symbols[0]
            try:
                fp = sp.diff(f, x)
                crit = sp.solve(sp.Eq(fp, 0), x)
                rep.add("opt.crit_points", True, f"Points critiques: {crit}")
            except Exception as e:
                rep.add("opt.crit_points", False, f"Erreur dérivée/solve: {e}")
                crit = []

            candidates = []
            if dom_interval is not None:
                a, b = dom_interval
                candidates.extend([sp.nsimplify(a), sp.nsimplify(b)])
            candidates.extend(crit)

            vals = []
            for c in candidates:
                try:
                    vals.append((c, sp.N(f.subs({x: c}))))
                except Exception:
                    pass

            if vals:
                if goal == "min":
                    x_star_true, f_star_true = min(vals, key=lambda t: float(t[1]))
                else:
                    x_star_true, f_star_true = max(vals, key=lambda t: float(t[1]))
                rep.details["x_star_true"] = str(x_star_true)
                rep.details["f_star_true"] = str(f_star_true)
                rep.add("opt.best_candidate", True, f"Best candidate: x*={x_star_true}, f*={f_star_true}")

                # compare x if provided
                if claimed_point and x in claimed_point:
                    xc = _to_float(claimed_point[x])
                    xt = _to_float(x_star_true)
                    if xc is not None and xt is not None:
                        rep.add("opt.compare_x", abs(xc - xt) < 1e-3, f"x*: claimed={xc}, true={xt}")
                    else:
                        rep.add("opt.compare_x", False, "Comparaison x* impossible")
                else:
                    rep.add("opt.compare_x", True, "Comparaison x* ignorée (non fourni)")
            else:
                rep.add("opt.best_candidate", False, "Impossible d'évaluer les candidats 1D")

        # -----------------------------
        # Multi-var heuristic local optimality check (numeric)
        # -----------------------------
        if len(var_names) > 1 and claimed_point and f_claim_val is not None:
            # small random neighborhood checks (NOT a proof, but useful)
            base = {s: _to_float(v) for s, v in claimed_point.items()}
            if all(v is not None for v in base.values()):
                base_f = _to_float(f_claim_val)
                improved = 0
                tested = 0

                # build feasibility function if constraints exist
                constraint_exprs = []
                if dom_constraints is not None:
                    for c in dom_constraints:
                        expr = _constraint_to_expr(c, locals_map)
                        if expr is not None:
                            constraint_exprs.append(expr)

                def feasible(pt: Dict[sp.Symbol, float]) -> bool:
                    if not constraint_exprs:
                        return True
                    for expr in constraint_exprs:
                        try:
                            v = float(sp.N(expr.subs(pt)))
                            if v > 1e-6:
                                return False
                        except Exception:
                            return False
                    return True

                for _ in range(40):
                    pt = {}
                    for s in symbols:
                        pt[s] = base[s] + random.uniform(-0.5, 0.5)
                    if not feasible(pt):
                        continue
                    tested += 1
                    try:
                        fv = float(sp.N(f.subs(pt)))
                    except Exception:
                        continue
                    if goal == "min" and fv < base_f - 1e-3:
                        improved += 1
                    if goal == "max" and fv > base_f + 1e-3:
                        improved += 1

                rep.add(
                    "opt.local_check",
                    improved == 0,
                    f"Test local: {tested} voisins testés, améliorations trouvées={improved} (0 attendu)" if tested else "Test local ignoré (aucun voisin faisable)",
                )
            else:
                rep.add("opt.local_check", True, "Test local ignoré (point non numérique)")

        rep.ok = all(it.ok for it in rep.items)
        return rep