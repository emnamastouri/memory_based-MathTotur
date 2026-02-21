import re
from typing import Tuple, Dict, Optional, Any

import sympy as sp

from core.parse_blocks import extract_blocks

HEADINGS = ("EXERCICE", "SOLUTION", "FINAL_ANSWER", "CHECK")


def _force_heading_on_own_line(text: str) -> str:
    if not text:
        return text
    t = text.replace("\r\n", "\n")
    for h in HEADINGS:
        t = re.sub(rf"(?m)^{h}\s*:\s*(.+)$", rf"{h}:\n\1", t)
    return t


def _strip_ellipsis(text: str) -> str:
    return (text or "").replace("...", "").strip()


def _rebuild_blocks(blocks: Dict[str, str]) -> str:
    parts = []
    for h in HEADINGS:
        if h in blocks and blocks[h].strip():
            parts.append(f"{h}:\n{blocks[h].strip()}")
    return "\n\n".join(parts).strip()


def _extract_var_from_derivative(expr: Any) -> Optional[str]:
    try:
        if getattr(expr, "func", None) == sp.Derivative and getattr(expr, "variables", None):
            v = expr.variables[0]
            if isinstance(v, sp.Symbol):
                return str(v)
    except Exception:
        pass
    return None


def _fix_check_derivative_eq_to_directive(check: str) -> str:
    if not check:
        return check
    s = check.strip()

    if re.search(r"(?i)\bDERIVATIVE\s*;.*\bfunc\s*=", s):
        return s

    m = re.match(r"(?is)^\s*DERIVATIVE\s*;\s*(.+)$", s)
    if not m:
        return s

    payload = m.group(1).strip()
    try:
        obj = sp.sympify(payload)
    except Exception:
        return s

    if not isinstance(obj, sp.Equality):
        return s

    diff_expr = None
    if obj.lhs.has(sp.Derivative) or (getattr(obj.lhs, "func", None) == sp.Derivative):
        diff_expr = obj.lhs
    elif obj.rhs.has(sp.Derivative) or (getattr(obj.rhs, "func", None) == sp.Derivative):
        diff_expr = obj.rhs

    if diff_expr is None:
        return s

    try:
        func = getattr(diff_expr, "expr", None)
        var = _extract_var_from_derivative(diff_expr) or "x"
        if func is None:
            return s
        return f"DERIVATIVE; var={var}; func={str(func)}"
    except Exception:
        return s


def _wrap_scalar_final_answer_for_system_if_needed(final_answer: str, check: str) -> str:
    """
    If CHECK is SYSTEM; Eq(...) and FINAL_ANSWER is scalar like "2",
    convert FINAL_ANSWER -> "{x:2}" when system has exactly one symbol.
    This matches your engine requirement: mapping dict for system.
    """
    if not final_answer or not check:
        return final_answer

    if not check.strip().upper().startswith("SYSTEM;"):
        return final_answer

    # If it already looks like a dict, keep
    fa = final_answer.strip()
    if fa.startswith("{") and fa.endswith("}"):
        return fa

    # Extract Eq(...) parts after "SYSTEM;"
    payload = check.split(";", 1)[1].strip()
    # The engine usually supports multiple Eq(...) separated by ';' or newlines,
    # but for this fix we parse the first Eq(...)
    eq_text = payload.split("\n")[0].strip()
    try:
        eq_obj = sp.sympify(eq_text)
    except Exception:
        return fa

    if not isinstance(eq_obj, sp.Equality):
        return fa

    syms = sorted(list(eq_obj.free_symbols), key=lambda s: s.name)
    if len(syms) != 1:
        return fa

    x = syms[0]
    # Try parse scalar as sympy
    try:
        val = sp.sympify(fa)
    except Exception:
        return fa

    return "{" + f"{x}:{val}" + "}"


def auto_fix_solution_for_verify(enonce: str, solution_text: str) -> Tuple[str, str]:
    fixed_enonce = _strip_ellipsis((enonce or "").strip())
    if len(fixed_enonce) < 45:
        fixed_enonce = fixed_enonce + " (Donner la réponse finale et vérifier.)"

    t = _strip_ellipsis(solution_text or "")
    t = _force_heading_on_own_line(t)

    blocks = extract_blocks(t)
    if not blocks:
        blocks = {
            "EXERCICE": fixed_enonce,
            "SOLUTION": t.strip(),
            "FINAL_ANSWER": "",
            "CHECK": "",
        }

    fa = (blocks.get("FINAL_ANSWER") or "").strip()
    ck = (blocks.get("CHECK") or "").strip()

    # Move mistaken DERIVATIVE directive from FINAL_ANSWER to CHECK
    if fa.upper().startswith("DERIVATIVE;") and not ck:
        blocks["CHECK"] = fa
        blocks["FINAL_ANSWER"] = ""
        fa = ""
        ck = blocks["CHECK"]

    if ck:
        blocks["CHECK"] = _fix_check_derivative_eq_to_directive(ck)

    # ✅ Fix SYSTEM: scalar -> {x:scalar} when needed
    fa2 = (blocks.get("FINAL_ANSWER") or "").strip()
    ck2 = (blocks.get("CHECK") or "").strip()
    if fa2 and ck2:
        blocks["FINAL_ANSWER"] = _wrap_scalar_final_answer_for_system_if_needed(fa2, ck2)

    blocks["EXERCICE"] = fixed_enonce
    fixed_solution = _rebuild_blocks(blocks)

    return fixed_enonce, fixed_solution