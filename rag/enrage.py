"""Make enrage bands matchable by BM25.

Measured problem (scripts/check_bm25_tokens.py): BM25 tokenizes '2500%+' to the
single term '2500', and '0-2500%' to '0' and '2500'. The '%' and '+' are
stripped, so both rotation chunks contain '2500' and a query about either band
matches both equally. BM25 keeps the number and loses the direction.

Fix: annotate every enrage expression with the 500-point bands it covers, so
    '0-2500%'  -> enrageband0 ... enrageband2000
    '2500%+'   -> enrageband2500 ... enrageband5000
    'at 3000%' -> enrageband3000

Band tokens are used rather than direction tokens ('upto'/'from') because a
query naming a value *inside* a range ("arms rotation at 3000%") must match the
chunk describing that range. Band membership handles both cases with one scheme.

Tokens are APPENDED, never substituted, so the original text stays intact and
golden-set phrases like '0-2500%' still match.

The same annotation must be applied to the query, or the two sides speak
different languages - see annotate_query.
"""

import re

BAND_STEP = 500
MAX_ENRAGE = 5000

# "0-2500%", "3000-4000%", "2500%-4000%"
RANGE_RE = re.compile(r"\b(\d{1,5})\s*%?\s*[-–]\s*(\d{1,5})\s*%")
# "2500%+", "3500%+ enrage"
FROM_RE = re.compile(r"\b(\d{1,5})\s*%\s*\+")
# "below 2500%", "under 3500%", "<3500%"
BELOW_RE = re.compile(r"(?:below|under|<)\s*(\d{1,5})\s*%", re.IGNORECASE)
# "above 3000%", "over 3000%"
ABOVE_RE = re.compile(r"(?:above|over)\s*(\d{1,5})\s*%", re.IGNORECASE)
# "at 3000% enrage" / "3000% enrage" - requires the word 'enrage' so that
# unrelated percentages ("100% adrenaline") are never annotated
AT_RE = re.compile(r"\b(\d{1,5})\s*%\s*(?=\w*\s*enrage)", re.IGNORECASE)

TOKEN = "enrageband{}"


def _bands(low, high):
    """Band tokens covering [low, high], clamped to the ladder."""
    low = max(0, min(low, MAX_ENRAGE))
    high = max(0, min(high, MAX_ENRAGE))
    start = (low // BAND_STEP) * BAND_STEP
    return [TOKEN.format(b) for b in range(start, high + 1, BAND_STEP)]


def bands_for(text):
    """Every band token implied by the enrage expressions in `text`.

    Patterns are applied most-specific first, and each match is blanked out
    before the next pattern runs. Without that, 'below 2500%' matches BOTH the
    below-rule and the at-rule, emitting a spurious enrageband2500 that then
    collides with the '2500%+' chunk - the exact confusion this module exists
    to remove.
    """
    found = []
    remaining = text

    def consume(pattern, handler):
        nonlocal remaining
        for match in list(pattern.finditer(remaining)):
            found.extend(handler(*(int(g) for g in match.groups())))
        remaining = pattern.sub(lambda m: " " * len(m.group(0)), remaining)

    consume(RANGE_RE, lambda lo, hi: _bands(lo, hi) if lo <= hi else [])
    consume(FROM_RE, lambda v: _bands(v, MAX_ENRAGE))
    consume(BELOW_RE, lambda v: _bands(0, max(0, v - BAND_STEP)))
    consume(ABOVE_RE, lambda v: _bands(v, MAX_ENRAGE))
    consume(AT_RE, lambda v: _bands(v, v))

    # Preserve order, drop duplicates
    return list(dict.fromkeys(found))


def annotate(text):
    """Append band tokens after each line containing an enrage expression.

    Appended per line rather than per document so the tokens survive chunking
    and stay attached to the rotation they describe.
    """
    out = []
    for line in text.split("\n"):
        tokens = bands_for(line)
        out.append(f"{line} [{' '.join(tokens)}]" if tokens else line)
    return "\n".join(out)


def annotate_query(query):
    """Append band tokens to a query, for the BM25 side only.

    Not applied to the dense side: 'enrageband2500' is meaningless to a
    sentence-transformer and would only add noise to the embedding.
    """
    tokens = bands_for(query)
    return f"{query} {' '.join(tokens)}" if tokens else query
