from __future__ import annotations
from core.parse_blocks import get_final_answer, get_check
from core.verify_report import VerifyReport
from core.verifiers import ALL_VERIFIERS

def structural_checks(enonce: str, solution: str) -> VerifyReport:
    rep = VerifyReport(ok=True, kind="structural", summary="Contrôles structurels")
    if not enonce or len(enonce.strip()) < 40:
        rep.add("structure.enonce", False, "Énoncé trop court / vide")
    else:
        rep.add("structure.enonce", True, "Énoncé OK")
    if "..." in (enonce or "") or "..." in (solution or ""):
        rep.add("structure.ellipsis", False, "Présence de '...' (énoncé ou solution)")
    else:
        rep.add("structure.ellipsis", True, "Pas de '...'")
    return rep

def verify(topic: str, enonce: str, solution_text: str) -> VerifyReport:
    # Extract blocks from solution_text
    final_answer = get_final_answer(solution_text)
    check = get_check(solution_text)

    # Structural checks
    srep = structural_checks(enonce, solution_text)

    # Run applicable verifiers
    chosen = [v for v in ALL_VERIFIERS if v.can_handle(topic, enonce, solution_text, final_answer, check)]
    if not chosen:
        # fallback report
        rep = VerifyReport(ok=srep.ok, kind="structural", summary="Aucun plugin applicable — seulement checks structurels")
        rep.items.extend(srep.items)
        return rep

    # Merge reports
    rep = VerifyReport(ok=srep.ok, kind="mixed", summary="Vérification complète (plugins)")
    rep.items.extend(srep.items)

    for v in chosen:
        vrep = v.verify(topic, enonce, solution_text, final_answer, check)
        rep.items.extend(vrep.items)

    rep.ok = all(item.ok for item in rep.items)
    return rep