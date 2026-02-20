from __future__ import annotations

import ast
import random
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp

from core.verify_report import VerifyReport
from core.verifiers.base import BaseVerifier


def _smart_parse(value: Optional[str]) -> Any:
    """
    Robust parser for FINAL_ANSWER:
    - Tries sympy.sympify first (handles 2*x+1, [1,2], {x:2})
    - If it fails, tries ast.literal_eval (handles {"x":2} JSON-like dict/list)
    """
    if not value:
        return None
    s = value.strip()
    if not s:
        return None

    # 1) sympy parse
    try:
        return sp.sympify(s)
    except Exception:
        pass

    # 2) python literal parse (safe)
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _is_eq_string(s: str) -> bool:
    s = (s or "").strip()
    return s.startswith("Eq(") or s.startswith("Eq ")


def _extract_system_equations(check: str) -> List[str]:
    """
    CHECK example:
      SYSTEM; Eq(x+y,1); Eq(x-y,3)
    Returns: ["Eq(x+y,1)", "Eq(x-y,3)"]
    """
    parts = [p.strip() for p in (check or "").split(";")]
    # first is "SYSTEM"
    return [p for p in parts[1:] if p]


def _collect_symbols_from_eqs(eqs: List[sp.Equality]) -> List[sp.Symbol]:
    syms = set()
    for e in eqs:
        syms |= set(e.free_symbols)
    # stable order
    return sorted(list(syms), key=lambda x: x.name)


def _final_answer_to_subs_dict(fa_obj: Any) -> Optional[Dict[sp.Symbol, Any]]:
    """
    Convert FINAL_ANSWER to a substitution dict {Symbol: value}.
    Accepts:
      - sympy Dict / python dict with keys as Symbols or strings
      - list/tuple of (var, val) pairs
    """
    if fa_obj is None:
        return None

    # SymPy Dict behaves like dict
    if isinstance(fa_obj, dict):
        out: Dict[sp.Symbol, Any] = {}
        for k, v in fa_obj.items():
            key = sp.Symbol(k) if isinstance(k, str) else k
            if isinstance(key, sp.Basic):
                out[key] = v
        return out if out else None

    # SymPy Dict type
    if hasattr(fa_obj, "items"):
        try:
            items = list(fa_obj.items())
            out: Dict[sp.Symbol, Any] = {}
            for k, v in items:
                key = sp.Symbol(str(k)) if isinstance(k, str) else k
                if isinstance(key, sp.Basic):
                    out[key] = v
            return out if out else None
        except Exception:
            pass

    # list of pairs
    if isinstance(fa_obj, (list, tuple)):
        try:
            out: Dict[sp.Symbol, Any] = {}
            for pair in fa_obj:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    k, v = pair
                    key = sp.Symbol(k) if isinstance(k, str) else k
                    if isinstance(key, sp.Basic):
                        out[key] = v
            return out if out else None
        except Exception:
            return None

    return None


class AlgebraEquationsVerifier(BaseVerifier):
    """
    Full SymPy integration for:
      - CHECK = Eq(...)
      - CHECK = SYSTEM; Eq(...); Eq(...); ...
    Supports FINAL_ANSWER as:
      - expression / list of roots / set
      - dict (SymPy dict {x:2} or JSON-like {"x":2})
    """

    name = "algebra_equations"

    def can_handle(self, topic_or_directive, enonce, solution, final_answer, check):
        # Prefer directive-driven activation
        if check:
            head = check.strip().split(";", 1)[0].strip().upper()
            if head == "SYSTEM":
                return True
            if _is_eq_string(check.strip()):
                return True

        # Fallback topic keywords
        t = (topic_or_directive or "").lower()
        return any(k in t for k in ["équation", "equation", "système", "systeme", "arithmétique", "arithmetique"])

    def verify(self, topic, enonce, solution, final_answer, check):
        rep = VerifyReport(ok=True, kind="mixed", summary="Vérification Algèbre (Eq / Système)")

        rep.add(
            "structure.final_answer_present",
            bool(final_answer),
            "FINAL_ANSWER présent" if final_answer else "FINAL_ANSWER manquant",
        )
        if not check:
            rep.add("structure.check_present", False, "CHECK manquant (Eq(...) ou SYSTEM; Eq(...); ...)")
            rep.ok = False
            return rep

        # Parse FINAL_ANSWER robustly
        fa_obj = _smart_parse(final_answer)
        rep.add("parse.final_answer", fa_obj is not None, "FINAL_ANSWER parsé" if fa_obj is not None else "FINAL_ANSWER non parsable")

        # --------------------------
        # CASE A: SYSTEM; Eq(...); ...
        # --------------------------
        if check.strip().upper().startswith("SYSTEM"):
            eq_strs = _extract_system_equations(check)
            if not eq_strs:
                rep.add("parse.system", False, "Aucune équation trouvée après SYSTEM;")
                rep.ok = False
                return rep

            eqs: List[sp.Equality] = []
            try:
                for s in eq_strs:
                    e = sp.sympify(s)
                    if not isinstance(e, sp.Equality):
                        raise ValueError(f"Pas une égalité: {s}")
                    eqs.append(e)
                rep.add("parse.system", True, f"{len(eqs)} équations parsées")
            except Exception as e:
                rep.add("parse.system", False, f"Erreur parsing SYSTEM: {e}")
                rep.ok = False
                return rep

            syms = _collect_symbols_from_eqs(eqs)
            rep.details["system_symbols"] = [str(s) for s in syms]
            if not syms:
                rep.add("symbolic.system_symbols", False, "Aucune variable détectée dans le système")
                rep.ok = False
                return rep
            rep.add("symbolic.system_symbols", True, f"Variables: {', '.join([str(s) for s in syms])}")

            # Solve system
            try:
                sols = sp.solve(eqs, syms, dict=True)
                rep.details["system_solutions"] = [str(s) for s in sols]
                rep.add("symbolic.system_solve", bool(sols), f"Solutions: {sols}" if sols else "Aucune solution trouvée")
            except Exception as e:
                rep.add("symbolic.system_solve", False, f"Erreur solve(): {e}")
                rep.ok = False
                return rep

            # Compare with FINAL_ANSWER if it is a dict-like mapping
            subs = _final_answer_to_subs_dict(fa_obj)
            if subs is None:
                rep.add("compare.final_answer_mapping", False, "FINAL_ANSWER attendu comme dict pour un système (ex: {x:2, y:3})")
                # Still can do numeric sanity checks below
            else:
                # Check if provided assignment satisfies all equations
                all_ok = True
                try:
                    for e in eqs:
                        val = sp.simplify(e.lhs.subs(subs) - e.rhs.subs(subs))
                        all_ok = all_ok and (sp.simplify(val) == 0)
                    rep.add("symbolic.system_substitution", all_ok, "Substitution: FINAL_ANSWER satisfait le système" if all_ok else "Substitution: FINAL_ANSWER ne satisfait pas le système")
                except Exception as e:
                    rep.add("symbolic.system_substitution", False, f"Erreur substitution: {e}")

            # Numeric evaluation sanity (optional)
            try:
                # pick first equation residual
                e0 = eqs[0]
                res = sp.simplify(e0.lhs - e0.rhs)
                x0 = syms[0]
                f = sp.lambdify(x0, res.subs({s: 1 for s in syms if s != x0}), "numpy")
                _ = [f(random.uniform(-5, 5)) for _ in range(4)]
                rep.add("numeric.eval", True, "Évaluation numérique OK (sanity)")
            except Exception:
                rep.add("numeric.eval", True, "Évaluation numérique ignorée (pas nécessaire)")

            rep.ok = all(item.ok for item in rep.items)
            return rep

        # --------------------------
        # CASE B: Eq(...)
        # --------------------------
        # Parse CHECK as Eq
        try:
            eq = sp.sympify(check)
            rep.add("parse.check", isinstance(eq, sp.Equality), "CHECK parsé (Eq)" if isinstance(eq, sp.Equality) else "CHECK n'est pas Eq(...)")
        except Exception as e:
            rep.add("parse.check", False, f"CHECK non parsable: {e}")
            rep.ok = False
            return rep

        if not isinstance(eq, sp.Equality):
            rep.add("symbolic.substitution", False, "CHECK doit être Eq(...) ou SYSTEM; ...")
            rep.ok = False
            return rep

        expr = sp.simplify(eq.lhs - eq.rhs)
        syms = sorted(list(expr.free_symbols), key=lambda s: s.name)
        rep.details["eq_symbols"] = [str(s) for s in syms]

        if not syms:
            # Equation without variable: check if it is true/false
            rep.add("symbolic.eq_constant", sp.simplify(expr) == 0, "Eq constante vraie" if sp.simplify(expr) == 0 else "Eq constante fausse")
            rep.ok = all(item.ok for item in rep.items)
            return rep

        x = syms[0]  # choose first symbol
        rep.add("symbolic.variable_guess", True, f"Variable utilisée: {x}")

        # Determine roots from FINAL_ANSWER
        roots: List[Any] = []
        if fa_obj is None:
            roots = []
        elif isinstance(fa_obj, (list, tuple, sp.Tuple)):
            roots = list(fa_obj)
        elif isinstance(fa_obj, sp.Set):
            roots = list(fa_obj)
        elif isinstance(fa_obj, dict):
            # If dict is given, attempt to use x key
            subs = _final_answer_to_subs_dict(fa_obj)
            if subs and x in subs:
                roots = [subs[x]]
        else:
            roots = [fa_obj]

        if not roots:
            rep.add("symbolic.substitution", False, "Aucune solution exploitable dans FINAL_ANSWER")
            rep.ok = False
            return rep

        # Substitute each root into equation residual
        all_ok = True
        for r in roots:
            try:
                val = sp.N(expr.subs({x: r}))
                ok = abs(complex(val)) < 1e-6
                all_ok = all_ok and ok
            except Exception:
                all_ok = False

        rep.add("symbolic.substitution", all_ok, "Substitution: solutions satisfont Eq(...)" if all_ok else "Substitution: au moins une solution ne satisfait pas Eq(...)")

        # Numeric evaluation sanity (not a proof, just ensures expression is evaluable)
        try:
            f = sp.lambdify(x, expr, "numpy")
            samples = [random.uniform(-5, 5) for _ in range(5)]
            _ = [f(s) for s in samples]
            rep.add("numeric.eval", True, "Évaluation numérique OK (pas d'erreur)")
        except Exception as e:
            rep.add("numeric.eval", False, f"Évaluation numérique échouée: {e}")

        rep.ok = all(item.ok for item in rep.items)
        return rep