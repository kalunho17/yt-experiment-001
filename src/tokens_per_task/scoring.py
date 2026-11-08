"""Exact-match scoring against answer keys. No LLM judge is used anywhere.

* T1: normalised label equality.
* T2, T4: the first number after the last ``Answer:`` marker (fallback: the last number in
  the reply) must equal the key.
* T3: numbers within 1 % of the key; supplier names must match exactly one supplier.
"""
from __future__ import annotations

import math
import re

_NUM_TOKEN = re.compile(r"[-−]?\d[\d,.]*")
_ANSWER_MARK = re.compile(r"answer\s*\**\s*[:：]", re.IGNORECASE)


def parse_number(tok: str) -> float | None:
    """Parse a number written with either thousands or decimal commas/points."""
    s = tok.replace("−", "-").strip().rstrip(".,")
    neg = s.startswith("-")
    s = s.lstrip("-")
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):          # 1.234,56
            s = s.replace(".", "").replace(",", ".")
        else:                                     # 1,234.56
            s = s.replace(",", "")
    elif "," in s:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", s):  # 1,234 / 1,234,567
            s = s.replace(",", "")
        elif s.count(",") == 1:                   # 12,5 (decimal comma)
            s = s.replace(",", ".")
        else:
            return None
    elif s.count(".") > 1:
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
            s = s.replace(".", "")
        else:
            return None
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUM_TOKEN.finditer(text):
        v = parse_number(m.group(0))
        if v is not None:
            out.append(v)
    return out


def answer_segment(text: str) -> str | None:
    """The first non-empty line after the last 'Answer:' marker, or None if there is no marker."""
    marks = list(_ANSWER_MARK.finditer(text))
    if not marks:
        return None
    rest = text[marks[-1].end():]
    for line in rest.splitlines():
        line = line.strip().strip("*").strip()
        if line:
            return line
    return ""


def extract_number(text: str) -> float | None:
    """First number after the last 'Answer:'; if there is none, the last number in the reply."""
    seg = answer_segment(text)
    if seg:
        nums = numbers_in(seg.replace("$", "").replace("€", ""))
        if nums:
            return nums[0]
    nums = numbers_in(text.replace("$", "").replace("€", ""))
    return nums[-1] if nums else None


def norm_label(s: str) -> str:
    s = s.strip().strip("`*\"'").strip().rstrip(".").strip().strip("`*\"'").strip()
    return re.sub(r"\s+", " ", s).lower()


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def fmt_num(v: float) -> str:
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.6g}" if abs(v) >= 1e6 else str(round(v, 6))


def score_t1(content: str, expected: str) -> tuple[bool, str]:
    pred = norm_label(content)
    return pred == norm_label(expected), pred


def score_numeric(content: str, expected: str) -> tuple[bool, str]:
    """T2 and T4: exact numeric match."""
    v = extract_number(content)
    if v is None:
        return False, ""
    key = float(expected)
    return math.isclose(v, key, rel_tol=1e-9, abs_tol=1e-6), fmt_num(v)


def score_t3(content: str, question: dict, suppliers: list[str]) -> tuple[bool, str]:
    seg = answer_segment(content)
    if question["answer_type"] == "supplier":
        hay = norm_name(seg if seg is not None else content)
        found = [s for s in suppliers if norm_name(s) in hay]
        # a name contained in a longer supplier name must not count as a second hit
        found = [s for s in found if not any(s != o and norm_name(s) in norm_name(o) for o in found)]
        if len(found) != 1:
            return False, (seg or "")[:120]
        return found[0] == question["answer"], found[0]
    v = extract_number(content)
    if v is None:
        return False, ""
    key = float(question["answer"])
    ok = abs(v - key) <= 0.01 * abs(key) if key != 0 else abs(v) <= 0.01
    return ok, fmt_num(v)
