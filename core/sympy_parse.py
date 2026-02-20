import ast
import sympy as sp

def smart_parse(value: str):
    value = (value or "").strip()
    if not value:
        return None

    # 1) direct sympy parse
    try:
        return sp.sympify(value)
    except Exception:
        pass

    # 2) python literal parse (handles {"x":2} / [1,2] etc.)
    try:
        obj = ast.literal_eval(value)
        return obj
    except Exception:
        return None