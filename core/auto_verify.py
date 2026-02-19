from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import re
import numpy as np

@dataclass
class VerifyReport:
    ok: bool
    kind: str
    message: str

def verify_by_topic(topic: str, enonce: str, solution: str) -> Optional[VerifyReport]:
    t = topic.lower()

    # Example 1: Statistics regression – numeric recompute if table is present
    if "stat" in t or "régression" in t or "correlation" in t:
        rep = verify_regression_if_possible(enonce, solution)
        return rep

    # Example 2: Differential equation – verify a claimed solution y(x) if present (basic)
    if "équations différentielles" in t or "equations différentielles" in t:
        rep = verify_diff_eq_if_possible(enonce, solution)
        return rep

    return None

def verify_regression_if_possible(enonce: str, solution: str) -> Optional[VerifyReport]:
    # Try to extract x and y lists from the ENONCE table (very simple heuristic)
    # If not found, skip.
    # You can improve this later with stricter parsing.
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", enonce)
    if len(nums) < 10:
        return None

    # heuristic: last 5 numbers could be y-values in your typical tables
    # (because your generated examples often include 5 years)
    try:
        y = np.array(list(map(float, nums[-5:])))
        x = np.arange(len(y), dtype=float)

        # compute correlation and regression y = ax + b
        x_mean, y_mean = x.mean(), y.mean()
        cov = ((x-x_mean)*(y-y_mean)).mean()
        var = ((x-x_mean)**2).mean()
        a = cov/var if var != 0 else 0.0
        b = y_mean - a*x_mean
        r = cov/(x.std()*y.std()) if x.std() != 0 and y.std() != 0 else 0.0

        # Check if solution contains roughly these numbers
        sol_nums = list(map(float, re.findall(r"-?\d+(?:\.\d+)?", solution)))
        if not sol_nums:
            return VerifyReport(False, "numeric", "Could not find any numeric values in solution.")

        # loose match: does solution contain something close to r or a?
        def close_to(val, arr, tol=0.05):
            return any(abs(v - val) <= tol for v in arr)

        ok = close_to(r, sol_nums, 0.05) or close_to(a, sol_nums, 0.2)
        msg = f"Numeric check (rough): computed r≈{r:.3f}, a≈{a:.3f}, b≈{b:.3f}. " \
              f"{'Looks consistent.' if ok else 'Could be inconsistent (heuristic check).'}"
        return VerifyReport(ok, "numeric", msg)
    except Exception:
        return None

def verify_diff_eq_if_possible(enonce: str, solution: str) -> Optional[VerifyReport]:
    # Placeholder: you can expand later using sympy dsolve/checkodesol if you standardize format.
    # For now we just ensure the solution is not empty and mentions a general form.
    if "C" in solution or "constante" in solution.lower():
        return VerifyReport(True, "heuristic", "Heuristic check: solution seems to include integration constants.")
    return VerifyReport(False, "heuristic", "Heuristic check failed: no obvious general-solution structure found.")
