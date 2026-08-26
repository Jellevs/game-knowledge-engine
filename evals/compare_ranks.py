"""Per-question rank comparison between two collections.

Aggregate metrics can hide equal-and-opposite movements. This shows where
each question actually landed, so a suspicious tie can be checked.

    uv run python evals/compare_ranks.py
"""

import json

from evals.relevance import relevant_ids
from rag.config import settings
from rag.search import get_client, get_embeddings

GOLDEN_PATH = settings.data_dir / "golden" / "arch_glacor.json"
COLLECTIONS = ["arch_glacor_raw", "arch_glacor_clean"]
K = 10


def rank_and_score(client, collection, embeddings, question, vector):
    """(rank of first relevant chunk, top-1 similarity score)."""
    relevant = relevant_ids(client, collection, question)
    hits = client.query_points(collection, query=vector, limit=K).points
    rank = next((i for i, h in enumerate(hits, 1) if str(h.id) in relevant), None)
    return rank, (hits[0].score if hits else 0.0)


def main():
    questions = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["questions"]
    client = get_client()
    embeddings = get_embeddings()

    raw_name, clean_name = COLLECTIONS
    print(f"{'question':34}{'raw':>16}{'clean':>16}   diff")
    print("-" * 82)

    differing = 0
    better = 0
    worse = 0

    for q in questions:
        if not q.get("expected_phrase"):
            continue

        # Same query, so embed once and reuse for both collections
        vector = embeddings.embed_query(q["question"])

        r_rank, r_score = rank_and_score(client, raw_name, embeddings, q, vector)
        c_rank, c_score = rank_and_score(client, clean_name, embeddings, q, vector)

        mark = ""
        if r_rank != c_rank:
            differing += 1
            # None means "not found in top K" - treat as worst possible
            r_val = r_rank if r_rank else K + 1
            c_val = c_rank if c_rank else K + 1
            if c_val < r_val:
                better += 1
                mark = "  <-- clean better"
            else:
                worse += 1
                mark = "  <-- clean worse"

        print(
            f"{q['id'][:33]:34}"
            f"{str(r_rank):>5} ({r_score:.3f})"
            f"{str(c_rank):>9} ({c_score:.3f})"
            f"{mark}"
        )

    print(f"\n{differing} questions ranked differently "
          f"({better} clean better, {worse} clean worse)")


if __name__ == "__main__":
    main()
