import re
from collections import Counter


FINAL_ANSWER_PATTERNS = [
    re.compile(r"final answer\s*[:=-]\s*([^\n.]+)", re.IGNORECASE),
    re.compile(r"answer\s*[:=-]\s*([^\n.]+)", re.IGNORECASE),
    re.compile(r"=\s*([^\n.]+)\s*$", re.IGNORECASE),
]


def _extract_final_answer(reasoning):
    text = reasoning.strip()

    for pattern in FINAL_ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:50] if lines else ""


def _build_summary(score):
    if score >= 9:
        return "Strong and well-structured reasoning"
    if score >= 8:
        return "Clear answer with good structure"
    if score >= 7:
        return "Reasonable path with minor gaps"
    if score >= 6:
        return "Usable answer but could be clearer"
    return "Weak reasoning or low agreement"


def score_reasonings(paths):
    if not paths:
        return []

    extracted_answers = [_extract_final_answer(path) for path in paths]
    consensus = Counter(answer.lower() for answer in extracted_answers if answer)

    results = []
    for index, reasoning in enumerate(paths, start=1):
        answer = extracted_answers[index - 1]
        score = 6.0
        lower_reasoning = reasoning.lower()

        if len(reasoning) > 120:
            score += 0.5
        if len(reasoning) > 220:
            score += 0.3
        if any(token in lower_reasoning for token in ["step", "because", "therefore", "so "]):
            score += 0.8
        if "final answer" in lower_reasoning or "answer:" in lower_reasoning:
            score += 0.8
        if answer and consensus.get(answer.lower(), 0) > 1:
            score += 1.2
        if answer and any(char.isdigit() for char in answer):
            score += 0.4
        if len(reasoning.split()) < 18:
            score -= 0.8

        score = max(1.0, min(10.0, round(score, 1)))
        results.append(
            {
                "path": index,
                "score": score,
                "summary": _build_summary(score),
            }
        )

    return results