
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
        f"Tu es un générateur d'exercices officiels pour le baccalauréat tunisien.\n\n"
        f"Contraintes:\n"
        f"- Section: {section}\n"
        f"- Thème: {topic}\n"
        f"- Style bac tunisien\n"
        f"- Exercice complet, cohérent, mathématiquement bien défini\n\n"
        f"Règles OBLIGATOIRES:\n"
        f"1) L'énoncé doit être COMPLET (aucune formule inachevée).\n"
        f"2) Toutes les définitions doivent être explicites.\n"
        f"3) Si c'est une suite, donner clairement la relation de récurrence complète.\n"
        f"4) Inclure au moins 2 ou 3 questions numérotées.\n"
        f"5) Après l'exercice, donner une solution complète et rigoureuse.\n"
        f"6) Ne jamais écrire '...' ou laisser une expression incomplète.\n\n"
        f"Format STRICT:\n"
        f"EXERCICE:\n"
        f"<énoncé complet>\n\n"
        f"SOLUTION:\n"
        f"<solution détaillée étape par étape>\n\n"
        f"Voici des exemples de style à imiter:\n"
        f"{examples_block}\n"
        f"FORMAT STRICT (obligatoire):\n"
        f"EXERCICE:\n<énoncé complet>\n\n"
        f"SOLUTION:\n<solution détaillée étape par étape>\n\n"
        f"FINAL_ANSWER:\n<réponse finale machine-readable (SymPy), ex: 2*x+1, {{x:2, y:3}}, [1,2,3]>\n"
        f"CHECK:\n<OPTIONNEL: une relation à vérifier si pertinent, ex: Eq(x**2-4,0)>\n\n"
        f"Règles pour FINAL_ANSWER:\n"
        f"- Utiliser la syntaxe SymPy: **, sqrt(), exp(), log(), sin(), cos(), pi\n"
        f"- Pas de phrases, pas de LaTeX, uniquement expression/valeurs.\n"
        f"- Si plusieurs réponses: liste Python [..] ou dict {{variable: valeur}}.\n\n"

    )


def generate_exercise(llm: LLMClient, system_prompt: str, section: str, topic: str, retrieved: List[RetrievalResult]) -> str:
    user_prompt = build_generation_prompt(section=section, topic=topic, retrieved=retrieved)

    out = llm.generate(system_prompt=system_prompt, context="", user_prompt=user_prompt)

    # IMPORTANT: selon ton llm_client, out peut être un objet ou une string
    # Si c'est un objet: out.text ; si c'est une string: out
    text = getattr(out, "text", out)
    return (text or "").strip()

