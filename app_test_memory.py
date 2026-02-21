# test_verifiers.py
# Run a small regression test suite for the SymPy plugin verification engine.

from __future__ import annotations

from typing import List, Dict, Any, Tuple

from core.verify_engine import verify

# If you added the auto-fix helper (recommended)
try:
    from core.verify_fix import auto_fix_solution_for_verify
    HAS_FIX = True
except Exception:
    HAS_FIX = False


def run_case(case: Dict[str, Any]) -> Tuple[bool, str, List[Tuple[str, bool, str]]]:
    topic = case["topic"]
    enonce = case["enonce"]
    sol = case["solution"]

    if HAS_FIX:
        enonce, sol = auto_fix_solution_for_verify(enonce, sol)

    rep = verify(topic, enonce, sol)

    items = []
    if hasattr(rep, "items"):
        for it in rep.items:
            items.append((it.name, it.ok, it.message))
    return bool(getattr(rep, "ok", False)), str(getattr(rep, "summary", "")), items


def print_report(title: str, ok: bool, summary: str, items: List[Tuple[str, bool, str]]):
    status = "✅ OK" if ok else "❌ FAIL"
    print(f"\n{'='*90}\n{status}  —  {title}\nSummary: {summary}\n{'-'*90}")
    for name, iok, msg in items:
        if not iok:
            print(f"  - [FAIL] {name}: {msg}")
    if ok:
        # If OK, show only a couple of positives (optional)
        shown = 0
        for name, iok, msg in items:
            if iok:
                print(f"  - [OK]   {name}: {msg}")
                shown += 1
            if shown >= 3:
                break


def main():
    cases: List[Dict[str, Any]] = []

    # 1) Algebra/system: engine expects dict for SYSTEM solutions; our auto-fix can wrap scalar -> {x:2}
    cases.append({
        "title": "Algebra SYSTEM (2x+1=5)",
        "topic": "Equations",
        "enonce": "Résoudre dans R l’équation suivante : 2x + 1 = 5. Donner la solution et vérifier.",
        "solution": (
            "EXERCICE:\nRésoudre dans R l’équation suivante : 2x + 1 = 5. Donner la solution et vérifier.\n\n"
            "SOLUTION:\nOn résout 2x+1=5.\n\n"
            "FINAL_ANSWER:\n2\n\n"
            "CHECK:\nSYSTEM; Eq(2*x+1,5)\n"
        )
    })

    # 2) Derivative directive (log domain sensitive) — should pass after calculus sampling fix and correct answer
    # NOTE: This checks f'(x) not f''(x)
    cases.append({
        "title": "Calculus DERIVATIVE (f(x)=log(x)/x)",
        "topic": "Analyse",
        "enonce": "Soit f définie sur ]0,+∞[ par f(x)=log(x)/x. Calculer f'(x). Donner la réponse finale et vérifier.",
        "solution": (
            "EXERCICE:\nSoit f définie sur ]0,+∞[ par f(x)=log(x)/x. Calculer f'(x). Donner la réponse finale et vérifier.\n\n"
            "SOLUTION:\nOn dérive f(x)=log(x)/x.\n\n"
            "FINAL_ANSWER:\n(1 - log(x))/x**2\n\n"
            "CHECK:\nDERIVATIVE; var=x; func=log(x)/x\n"
        )
    })

    # 3) Limit example (if your calculus verifier supports LIMIT directives)
    cases.append({
        "title": "Calculus LIMIT (sin(x)/x at 0)",
        "topic": "Analyse",
        "enonce": "Calculer la limite suivante : lim_{x->0} sin(x)/x. Donner la réponse finale et vérifier.",
        "solution": (
            "EXERCICE:\nCalculer la limite suivante : lim_{x->0} sin(x)/x. Donner la réponse finale et vérifier.\n\n"
            "SOLUTION:\nLimite classique.\n\n"
            "FINAL_ANSWER:\n1\n\n"
            "CHECK:\nLIMIT; limit(sin(x)/x, x, 0)\n"
        )
    })

    # 4) Complex numbers (if complex verifier checks Eq simplification)
    cases.append({
        "title": "Complex (i^2 = -1)",
        "topic": "Complexes",
        "enonce": "Dans C, simplifier i^2. Donner la réponse finale et vérifier.",
        "solution": (
            "EXERCICE:\nDans C, simplifier i^2. Donner la réponse finale et vérifier.\n\n"
            "SOLUTION:\nOn sait que i^2 = -1.\n\n"
            "FINAL_ANSWER:\n-1\n\n"
            "CHECK:\nEq(I**2, -1)\n"
        )
    })

    # 5) Linear algebra (basic equality)
    cases.append({
        "title": "Linear algebra (det([[1,2],[3,4]]) = -2)",
        "topic": "Algebre lineaire",
        "enonce": "Calculer le déterminant de la matrice [[1,2],[3,4]]. Donner la réponse finale et vérifier.",
        "solution": (
            "EXERCICE:\nCalculer le déterminant de la matrice [[1,2],[3,4]]. Donner la réponse finale et vérifier.\n\n"
            "SOLUTION:\nDet = 1*4 - 2*3 = -2.\n\n"
            "FINAL_ANSWER:\n-2\n\n"
            "CHECK:\nEq(Matrix([[1,2],[3,4]]).det(), -2)\n"
        )
    })

    total = len(cases)
    passed = 0

    print(f"\nRunning {total} verification tests. auto-fix={'ON' if HAS_FIX else 'OFF'}")
    print("Tip: If many FAIL with parsing, ensure headings are on their own lines and CHECK matches verifier format.\n")

    for c in cases:
        ok, summary, items = run_case(c)
        if ok:
            passed += 1
        print_report(c["title"], ok, summary, items)

    print(f"\n{'='*90}\nRESULT: {passed}/{total} passed")
    if passed != total:
        print("Some tests failed. Fix the first failing structural/parsing item; it often cascades.")


if __name__ == "__main__":
    main()