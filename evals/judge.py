"""LLM-as-judge scoring for generated answers.

Written by hand rather than using RAGAS. Two reasons:

1. ragas 0.4.x imports `langchain_community.chat_models.vertexai`, which no
   longer exists - langchain-community is being sunset. Pinning an old version
   would fight the project's LangChain 1.x install.
2. RAGAS prompts ask for structured JSON and are tuned for frontier models. A
   9B judge often returns something unparseable, and RAGAS records NaN for that
   row - producing a plausible-looking average computed from partial data.

Design for a small local judge:
  - one question per call, never batched
  - ask for a single word or single number, never JSON
  - parse leniently, retry once
  - COUNT PARSE FAILURES as a metric, so a struggling judge is visible rather
    than silently averaged away
"""

import re

from rag.config import settings

# Sentences per answer to check for faithfulness. Caps runtime on rambling
# answers; 8 covers everything the guides realistically produce.
MAX_CLAIMS = 8

_judge = None


def get_judge():
    """One ChatOllama instance, temperature 0 so scores are reproducible."""
    global _judge
    if _judge is None:
        from rag.llm import get_chat_model

        _judge = get_chat_model(
            settings.judge_model, temperature=settings.judge_temperature
        )
    return _judge


def ask(prompt: str) -> str:
    """One judge call, returning raw text."""
    return get_judge().invoke(prompt).content.strip()


def _yes_no(prompt: str, stats: dict) -> bool | None:
    """Ask a yes/no question. Returns None if the judge's answer is unparseable.

    Retries once - small models sometimes preamble on the first attempt and
    comply when asked again.
    """
    for attempt in range(2):
        reply = ask(prompt).lower()
        if re.search(r"\byes\b", reply):
            return True
        if re.search(r"\bno\b", reply):
            return False
        stats["retries"] = stats.get("retries", 0) + 1
    stats["parse_failures"] = stats.get("parse_failures", 0) + 1
    return None


def split_claims(answer: str) -> list[str]:
    """Split an answer into checkable statements.

    Sentence-level rather than LLM claim-extraction: extraction needs the judge
    to emit a structured list, which is exactly what small models fail at.
    Splitting locally is deterministic and free.

    The closing-punctuation class matters. A naive `(?<=[.!?])\\s+` misses
    sentences that end inside a quotation - `... in hard mode." Additionally,`
    - because the character before the space is a quote, not a period. That
    produced one compound claim per answer, and a single unsupported fragment
    then scored the whole answer 0.0. Found by spot-checking a 0.00 that
    should have been 1.00.
    """
    parts = re.split(r"""(?<=[.!?])["')\]]*\s+|\n+""", answer)
    claims = [p.strip(" -•\"'") for p in parts if len(p.strip(" -•\"'")) > 15]
    return claims[:MAX_CLAIMS]


def faithfulness(context: str, answer: str, stats: dict) -> float | None:
    """Fraction of the answer's statements that the context supports.

    This is the hallucination measure: 1.0 means everything said is grounded in
    what was retrieved, 0.0 means none of it is.
    """
    claims = split_claims(answer)
    if not claims:
        return None

    supported = 0
    counted = 0
    for claim in claims:
        verdict = _yes_no(
            "You are checking whether a statement is supported by a reference text.\n\n"
            f"REFERENCE TEXT:\n{context}\n\n"
            f"STATEMENT:\n{claim}\n\n"
            "Is the statement supported by the reference text?\n"
            "Rewording, paraphrasing and summarising all count as supported - the\n"
            "statement does not need to appear word for word. Answer no only if\n"
            "the statement contradicts the reference text or adds facts that are\n"
            "not in it.\n"
            "Reply with exactly one word: yes or no.",
            stats,
        )
        if verdict is None:
            continue
        counted += 1
        supported += int(verdict)

    return supported / counted if counted else None


def relevancy(question: str, answer: str, stats: dict) -> float | None:
    """Does the answer actually address the question that was asked?

    Separate from faithfulness: an answer can be perfectly grounded in the
    context and still not answer the question.
    """
    for attempt in range(2):
        reply = ask(
            "Rate how well the ANSWER addresses the QUESTION.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:\n{answer}\n\n"
            "Reply with a single number from 0 to 10, nothing else. "
            "0 = does not address the question at all. "
            "10 = fully and directly answers it."
        )
        match = re.search(r"\d+(?:\.\d+)?", reply)
        if match:
            return min(float(match.group()) / 10.0, 1.0)
        stats["retries"] = stats.get("retries", 0) + 1

    stats["parse_failures"] = stats.get("parse_failures", 0) + 1
    return None


def refused(answer: str, stats: dict) -> bool | None:
    """Did the model decline to answer?

    Cheap string check first - the prompt asks for a specific phrase, so most
    refusals are caught for free. The judge is only consulted for phrasings the
    string check misses.
    """
    lowered = answer.lower()
    for marker in (
        "don't have enough information",
        "do not have enough information",
        "not enough information",
        "cannot answer",
        "can't answer",
        "does not contain",
        "doesn't contain",
        "no information",
    ):
        if marker in lowered:
            return True

    return _yes_no(
        "Read this response and decide whether it declines to answer "
        "(says it lacks the information) or whether it gives a real answer.\n\n"
        f"RESPONSE:\n{answer}\n\n"
        "Does it decline to answer? Reply with exactly one word: yes or no.",
        stats,
    )
