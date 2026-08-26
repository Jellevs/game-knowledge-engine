"""Test the hypothesis that BM25 cannot separate enrage bands.

Claim: '0-2500%' and '2500%+' both tokenize to include '2500', and the
discriminating characters ('0-' prefix, '+' suffix) are stripped. If so, BM25
sees the two rotation chunks as containing the same term and cannot tell them
apart - which would explain why hybrid search barely moved the
enrage-conditional tag (0.339 -> 0.435).

    uv run python scripts/check_bm25_tokens.py
"""

from rag import sparse


def indices(text):
    """The set of term ids BM25 assigns to this text."""
    return set(sparse.embed_documents([text])[0].indices)


def report(label, a, b):
    ia, ib = indices(a), indices(b)
    shared = ia & ib
    print(f"\n{label}")
    print(f"  {a!r:24} -> {len(ia)} terms")
    print(f"  {b!r:24} -> {len(ib)} terms")
    print(f"  shared terms: {len(shared)}")
    if ia == ib and ia:
        print("  >>> IDENTICAL token sets - BM25 cannot distinguish these at all")
    elif shared:
        print(f"  >>> overlap, unique to first: {len(ia - ib)}, "
              f"unique to second: {len(ib - ia)}")
    else:
        print("  >>> no overlap - BM25 CAN distinguish them")


def main():
    print("Does BM25 see any difference between the enrage bands?")

    report("bare band markers", "0-2500%", "2500%+")
    report("in a sentence", "arms rotation 0-2500%", "arms rotation 2500%+")
    report("other band pair", "below 2500%", "3500%+ enrage")

    # The queries that actually fail, against the text they should match
    print("\n" + "=" * 60)
    print("Query vs the two competing chunks (real golden-set case)")
    query = "What is the arms rotation below 2500% enrage with Necromancy?"
    low = "0-2500%: Invoke Death then Soul Strike (flanking) then Soul Strike (flanking)"
    high = "2500%+: Invoke Death then Split Soul then Vulnerability bomb and Bloat"

    q = set(sparse.embed_query(query).indices)
    for name, text in (("LOW  (correct)", low), ("HIGH (distractor)", high)):
        overlap = q & indices(text)
        print(f"  {name}: {len(overlap)} query terms matched")

    print("\nIf both chunks match the same number of query terms, BM25 is blind "
          "to the band and the enrage-band normalisation is justified.")


if __name__ == "__main__":
    main()
