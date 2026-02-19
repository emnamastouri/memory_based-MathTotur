
from __future__ import annotations

from typing import List, Tuple


TEACHER_MOVES = [
    "prompting",          # ask guiding questions
    "error_signaling",    # indicate incorrectness without giving answer
    "hint",               # give partial information
    "concept_explanation",# explain rule/principle
    "partial_solution",   # reveal next step
    "full_solution",      # only when appropriate
]


def build_system_prompt() -> str:
    return (
        "You are MathTutorAI, a math tutor for Tunisian Bac/L1.\n"
        "Pedagogical rules:\n"
        "1) Use scaffolding: start with prompting or hints before giving full solution.\n"
        "2) Be step-by-step, but do not dump everything at once.\n"
        "3) Diagnose common misconceptions (sign errors, wrong rule, domain issues).\n"
        "4) If the student is wrong, signal the error and guide them to self-correct.\n"
        "5) If asked explicitly for the full solution, you may provide it, still step-by-step.\n"
        "6) Keep it clear and concise; prefer math notation.\n"
    )


def format_retrieved_memories(memories: List[Tuple[str, str, str]]) -> str:
    """
    memories: list of (error_type, teacher_move, assistant_response)
    Keep it short: top-3 max.
    """
    if not memories:
        return "No prior similar cases found."

    lines = ["Similar past tutoring cases (use them to personalize):"]
    for i, (error_type, teacher_move, response) in enumerate(memories, start=1):
        # truncate response to keep context small
        short = response.strip()
        if len(short) > 400:
            short = short[:400] + "..."
        lines.append(f"{i}) error_type={error_type} | teacher_move={teacher_move} | example_response={short}")
    return "\n".join(lines)


def build_user_prompt(topic: str, problem: str, student_attempt: str) -> str:
    return (
        f"Topic: {topic}\n"
        f"Problem: {problem}\n"
        f"Student attempt: {student_attempt}\n\n"
        "Task: Tutor the student. Start with a helpful teacher move (prompting/hint/error_signaling).\n"
        "Do NOT reveal the final answer immediately unless the student explicitly asks for it.\n"
    )
