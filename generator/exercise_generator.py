
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
    )


def generate_exercise(llm, system_prompt: str, section: str, topic: str, retrieved):
    # Build your user prompt the same way you already do
    examples_txt = ""
    if retrieved:
        # If your retrieved items are objects with .exercise.enonce/solution
        # adjust if your structure is different
        for r in retrieved:
            ex = getattr(r.exercise, "enonce", "")
            sol = getattr(r.exercise, "solution", "")
            examples_txt += f"\n---\nEXEMPLE:\n{ex}\nSOLUTION:\n{sol}\n"

    user_prompt = (
        f"SECTION: {section}\n"
        f"TOPIC: {topic}\n\n"
        "Generate ONE new exercise for the Tunisian Baccalaureate.\n"
        "Use the style of the examples if provided.\n\n"
        "Return EXACTLY this format:\n"
        "EXERCICE:\n<statement>\n\n"
        "SOLUTION:\n<detailed solution>\n\n"
        "EXAMPLES (if any):"
        f"{examples_txt}\n"
    )

    # IMPORTANT: llm.generate returns a STRING
    out_text = llm.generate(system_prompt=system_prompt, context="", user_prompt=user_prompt)
    return (out_text or "").strip()
