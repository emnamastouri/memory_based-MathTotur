from __future__ import annotations
import re
import numpy as np
from core.verify_report import VerifyReport
from core.verifiers.base import BaseVerifier

class StatsVerifier(BaseVerifier):
    name = "stats"

    def can_handle(self, topic, enonce, solution, final_answer, check):
        t = (topic or "").lower()
        return ("stat" in t or "régression" in t or "regression" in t or "corrél" in t or "correl" in t)

    def verify(self, topic, enonce, solution, final_answer, check):
        rep = VerifyReport(ok=True, kind="numeric", summary="Vérification Stats/Régression (numérique)")

        nums = re.findall(r"\b\d+(?:\.\d+)?\b", enonce.replace(",", "."))
        if len(nums) < 8:
            rep.add("extract.table", False, "Impossible d'extraire assez de nombres depuis l'énoncé")
            rep.ok = False
            return rep

        # Heuristique: on prend les 5 derniers comme y
        try:
            y = np.array(list(map(float, nums[-5:])))
            x = np.arange(len(y), dtype=float)

            x_mean, y_mean = x.mean(), y.mean()
            cov = ((x-x_mean)*(y-y_mean)).mean()
            var = ((x-x_mean)**2).mean()
            a = cov/var if var != 0 else 0.0
            b = y_mean - a*x_mean
            r = cov/(x.std()*y.std()) if x.std() != 0 and y.std() != 0 else 0.0

            sol_nums = list(map(float, re.findall(r"-?\d+(?:\.\d+)?", (solution or "").replace(",", "."))))
            rep.add("extract.solution_numbers", bool(sol_nums), "Nombres détectés dans la solution")
            if sol_nums:
                ok = any(abs(v - r) < 0.05 for v in sol_nums) or any(abs(v - a) < 0.2 for v in sol_nums)
                rep.add("numeric.consistency", ok, f"Calcul interne: r≈{r:.3f}, a≈{a:.3f}, b≈{b:.3f}")
            else:
                rep.add("numeric.consistency", False, "Aucun nombre trouvé dans la solution pour comparer")
        except Exception as e:
            rep.add("numeric.compute", False, f"Erreur recalcul stats: {e}")

        rep.ok = all(item.ok for item in rep.items)
        return rep