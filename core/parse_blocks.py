from __future__ import annotations
import re
from typing import Optional, Dict

_BLOCKS = ["EXERCICE", "SOLUTION", "FINAL_ANSWER", "CHECK"]

def extract_blocks(text: str) -> Dict[str, str]:
    """
    Returns dict with keys in _BLOCKS if found.
    Handles headings like 'FINAL_ANSWER:' 'FINAL_ANSWER :' etc.
    """
    if not text:
        return {}

    t = text.replace("\r\n", "\n")
    # Build regex to split on headings
    # Captures heading name
    pattern = r"(?m)^(EXERCICE|SOLUTION|FINAL_ANSWER|CHECK)\s*:\s*$"
    parts = re.split(pattern, t)
    # re.split returns: [before, H1, afterH1, H2, afterH2, ...]
    if len(parts) < 3:
        return {}

    blocks = {}
    # parts[0] is before first heading (ignore)
    i = 1
    while i < len(parts) - 1:
        heading = parts[i].strip()
        content = parts[i+1].strip()
        blocks[heading] = content
        i += 2
    return blocks

def get_final_answer(text: str) -> Optional[str]:
    blocks = extract_blocks(text)
    fa = blocks.get("FINAL_ANSWER")
    if fa:
        # take only first non-empty line if someone writes long paragraphs
        # but keep JSON/dict/list form across lines
        return fa.strip()
    return None

def get_check(text: str) -> Optional[str]:
    blocks = extract_blocks(text)
    ck = blocks.get("CHECK")
    return ck.strip() if ck else None