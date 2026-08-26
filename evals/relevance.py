"""Shared relevance logic for the eval scripts.

A chunk is relevant to a question when ALL of these hold:
  - it comes from expected_source (if the question names one)
  - its deepest header matches expected_section
  - expected_phrase appears in its text

Source matters since the three Arch-Glacor guides answer the same questions
differently, and both hybrid guides have a section called 'Melee Phase'.

Section alone was too coarse (7 sections, so 'did we hit the right section'
was nearly free). Phrase alone would be unfair across collections, since
phrases must exist in both the raw and preprocessed text.
"""

import re

from rag.retrieve import TEXT_KEY

# collection name -> list of (chunk_id, section, text)
# Scrolling the whole collection once per question was wasteful; cache it.
_CHUNK_CACHE: dict[str, list[tuple[str, str, str]]] = {}


def normalize(text):
    """Lowercase, keep only letters and digits."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def section_of(payload):
    """The most specific header on a chunk.

    Only the deepest: MarkdownHeaderTextSplitter carries parent headers
    forward, so every ### chunk also inherits '__Presets and Relics__' as H2.
    """
    metadata = payload.get("metadata", payload)
    headers = [str(v) for k, v in sorted(metadata.items()) if k.startswith("Header")]
    return headers[-1] if headers else ""


def text_of(payload):
    return payload.get(TEXT_KEY) or payload.get("page_content") or ""


def source_of(payload):
    """Which guide file this chunk came from."""
    metadata = payload.get("metadata", payload)
    return str(metadata.get("source", ""))


def load_chunks(client, collection, limit=10_000):
    """Every chunk as (id, source, normalized_section, text). Cached."""
    if collection not in _CHUNK_CACHE:
        points, _ = client.scroll(collection, limit=limit, with_payload=True)
        _CHUNK_CACHE[collection] = [
            (
                str(p.id),
                source_of(p.payload or {}),
                normalize(section_of(p.payload or {})),
                text_of(p.payload or {}),
            )
            for p in points
        ]
    return _CHUNK_CACHE[collection]


def wanted_sources(question):
    """expected_source may be null (any), one filename, or a list of them.

    A list is needed because the two hybrid guides share whole sections
    verbatim - the Opener, Melee Phase and Defensive Usage text is identical -
    so for those questions either guide is a correct answer.
    """
    value = question.get("expected_source")
    if not value:
        return None                      # any source is acceptable
    return {value} if isinstance(value, str) else set(value)


def relevant_ids(client, collection, question):
    """Chunk IDs that correctly answer this question."""
    want_sources = wanted_sources(question)
    want_section = normalize(question["expected_section"] or "")
    want_phrase = question.get("expected_phrase") or ""
    if not want_section or not want_phrase:
        return set()

    return {
        chunk_id
        for chunk_id, source, section, text in load_chunks(client, collection)
        if (want_sources is None or source in want_sources)
        and want_section in section
        and want_phrase in text
    }
