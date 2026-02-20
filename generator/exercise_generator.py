
from __future__ import annotations

from typing import List

from generator.retriever import RetrievalResult
from core.llm_client import LLMClient


def build_generation_prompt(section: str, topic: str, retrieved: List[RetrievalResult]) -> str:

    examples = []
    for r in retrieved:
        e = r.exercise
        # keep examples compact to avoid huge tokens
        examples.append(
            f"EXEMPLE:\n"
            f"- Enoncé:\n{e.enonce}\n\n"
            f"- Correction (résumé):\n{e.solution}\n"
        )

    examples_block = "\n\n---\n\n".join(examples) if examples else "Aucun exemple disponible."

    return (
    "Tu es un générateur d'exercices officiels pour le baccalauréat tunisien.\n\n"

    "Objectif : générer UN SEUL exercice cohérent, clair, académique, niveau bac.\n\n"

    "Contraintes :\n"
    f"- Section : {section}\n"
    f"- Thème : {topic}\n"
    "- Style bac tunisien, français académique.\n"
    "- Exercice mathématiquement cohérent (aucune contradiction).\n"
    "- Un seul sous-thème principal (pas mélange excessif).\n\n"

    "RÈGLES STRICTES :\n"
    "1) L'énoncé doit être COMPLET et autonome.\n"
    "2) Aucune incohérence logique.\n"
    "3) Ne jamais corriger l'exercice dans la solution.\n"
    "4) Ne jamais écrire 'supposons', 'il y a incohérence', 'correction'.\n"
    "5) Pas de répétitions inutiles.\n"
    "6) Pas de LaTeX (pas de \\lim, \\to, etc.).\n"
    "7) Utiliser * pour la multiplication (ex: 2*x, pas 2x).\n\n"

    "SOLUTION — RÈGLES DE LONGUEUR :\n"
    "- Maximum 25 lignes.\n"
    "- Étapes courtes et claires.\n"
    "- Pas de paragraphes longs.\n"
    "- Pas d'explications théoriques inutiles.\n"
    "- Aller directement aux calculs.\n\n"

    "FORMAT STRICT (à respecter EXACTEMENT) :\n"
    "EXERCICE:\n"
    "<énoncé complet>\n\n"

    "SOLUTION:\n"
    "1) Idée / méthode (1-2 lignes maximum)\n"
    "2) Calculs clairs et structurés\n"
    "3) Résultat final\n\n"

    "FINAL_ANSWER:\n"
    "<réponse machine-readable SymPy uniquement. "
    "Pas de texte. Pas de phrases. "
    "Exemples valides : 2*x+1, [1,2], {x:2, y:3}, {x_star:11, f_star:4.25}>\n\n"

    "CHECK (OBLIGATOIRE, UNE SEULE LIGNE) :\n"
    "Choisir UNE directive adaptée :\n"
    "- Eq(x**2-4,0)\n"
    "- SYSTEM; Eq(x+y,1); Eq(x-y,3)\n"
    "- OPTIMIZE; var=x; func=-2*x**2+8*x-3; domain=[0,5]; goal=max\n"
    "- DERIVATIVE; var=x; func=x**3+2*x\n"
    "- INTEGRAL; var=x; integrand=x**2+1\n"
    "- LIMIT; var=x; expr=sin(x)/x; point=0\n\n"

    "RÈGLES SYMPY OBLIGATOIRES :\n"
    "- Utiliser ** pour les puissances.\n"
    "- Utiliser exp(), log(), sqrt(), sin(), cos().\n"
    "- Toujours écrire 2*x et jamais 2x.\n"
    "- CHECK doit être valide SymPy.\n"
    "- FINAL_ANSWER doit être cohérent avec CHECK.\n\n"

    "IMPORTANT :\n"
    "- Si l'exercice est une optimisation : FINAL_ANSWER doit être {x_star:..., f_star:...}.\n"
    "- Si système : FINAL_ANSWER doit être {x:..., y:...}.\n"
    "- Si l'équation n'admet pas de solution évidente, calculer une approximation numérique avec 3 décimales."
     "- Toujours vérifier la solution numériquement avant d'écrire FINAL_ANSWER."
    "- Si équation simple : FINAL_ANSWER doit être une liste ou valeur.\n\n"

    "Voici des exemples de style à imiter :\n"
    f"{examples_block}\n"
)
def generate_exercise(llm: LLMClient, system_prompt: str, section: str, topic: str, retrieved: List[RetrievalResult]) -> str:
    user_prompt = build_generation_prompt(section=section, topic=topic, retrieved=retrieved)

    out = llm.generate(system_prompt=system_prompt, context="", user_prompt=user_prompt)

    # IMPORTANT: selon ton llm_client, out peut être un objet ou une string
    # Si c'est un objet: out.text ; si c'est une string: out
    text = getattr(out, "text", out)
    return (text or "").strip()

