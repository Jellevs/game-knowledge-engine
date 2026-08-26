"""Retrieval evaluation using ranx.

Measures only whether the right chunk comes back - no LLM involved.

    uv run python evals/run_retrieval.py
"""

import json

from ranx import Qrels, Run, evaluate
from scipy.stats import ttest_rel

from evals.relevance import relevant_ids
from evals.tracking import log_comparison, log_run
from rag import registry
from rag.config import settings
from rag.retrieve import is_hybrid_collection, search
from rag.search import get_client, get_embeddings

GOLDEN_PATH = settings.data_dir / "golden" / "arch_glacor.json"
COLLECTIONS = [
    "arch_glacor_raw",     # no preprocessing (baseline)
    "arch_glacor_both",    # preprocessed, dense only
    "arch_glacor_hybrid",  # preprocessed, dense + BM25 fused with RRF
    "arch_glacor_bands",   # hybrid + enrage band tokens
]
METRICS = ["hit_rate@1", "hit_rate@3", "hit_rate@5", "hit_rate@10", "mrr@10"]
K = 10


def build_qrels(client, collection, questions):
    """Which chunk IDs are correct answers for each question."""
    qrels = {}
    for q in questions:
        if not q.get("expected_phrase"):
            continue  # unanswerable questions have no correct chunk

        relevant = relevant_ids(client, collection, q)
        if not relevant:
            # Label problem or a phrase split across a chunk boundary -
            # either way, not a retrieval failure. Don't score it.
            print(f"   !! {q['id']}: no chunk has {q['expected_phrase']!r} "
                  f"in section {q['expected_section']!r}")
            continue

        # ranx wants {doc_id: relevance_grade}, not a bare set of ids
        qrels[q["id"]] = dict.fromkeys(relevant, 1)
        print(f"   {q['id']}: {len(relevant)} relevant chunks")
    return qrels


def build_run(client, collection, embeddings, questions, qrels):
    """What the retriever actually returned, with scores.

    Uses hybrid search automatically if the collection has sparse vectors, so
    dense-only and hybrid collections can be compared in the same run.
    """
    hybrid = is_hybrid_collection(client, collection)
    if hybrid:
        from rag import sparse

    run = {}
    for q in questions:
        if q["id"] not in qrels:
            continue
        dense_vector = embeddings.embed_query(q["question"])
        sparse_vector = sparse.embed_query(q["question"]) if hybrid else None
        hits = search(client, collection, dense_vector, sparse_vector, limit=K)
        run[q["id"]] = {str(h.id): float(h.score) for h in hits}
    return run


def reciprocal_ranks(qrels, run):
    """Per-question 1/rank of the first relevant chunk, 0 if none in top K.

    Computed here rather than pulled from ranx so the ordering is under our
    control - a paired test silently gives nonsense if the two score vectors
    are not aligned question-for-question.
    """
    scores = {}
    for qid, relevant in qrels.items():
        ranked = sorted(run.get(qid, {}).items(), key=lambda kv: kv[1], reverse=True)
        scores[qid] = 0.0
        for position, (doc_id, _) in enumerate(ranked, start=1):
            if doc_id in relevant:
                scores[qid] = 1.0 / position
                break
    return scores


def report_by_tag(per_question, questions, collections, min_size=4):
    """Mean reciprocal rank sliced by question tag.

    Shows which *kinds* of question are hard. 'shared-text' is the one to watch:
    those target sections the two hybrid guides contain verbatim, so two
    near-identical chunks compete for the same query.
    """
    by_tag = {}
    for q in questions:
        for tag in q.get("tags", []):
            by_tag.setdefault(tag, []).append(q["id"])

    width = max(len(c) for c in collections) + 3
    print("\nMean reciprocal rank by tag:")
    print("  " + "tag".ljust(22) + "n".ljust(5) +
          "".join(c.ljust(width) for c in collections))
    print("  " + "-" * (27 + width * len(collections)))

    per_collection_tags = {c: {} for c in collections}
    for tag, qids in sorted(by_tag.items()):
        scored = [q for q in qids if q in per_question[collections[0]]]
        if len(scored) < min_size:
            continue
        row = "  " + tag.ljust(22) + str(len(scored)).ljust(5)
        for c in collections:
            mean = sum(per_question[c].get(q, 0.0) for q in scored) / len(scored)
            per_collection_tags[c][tag] = mean
            row += f"{mean:.3f}".ljust(width)
        print(row)
    return per_collection_tags


def report_significance(per_question, collections):
    """Paired t-test of each collection against the first (the baseline)."""
    baseline = collections[0]
    shared = set(per_question[baseline])
    for c in collections[1:]:
        shared &= set(per_question[c])
    shared = sorted(shared)

    print(f"\nPaired t-test on reciprocal rank vs '{baseline}' (n={len(shared)}):")
    base_scores = [per_question[baseline][q] for q in shared]
    results = {}
    for c in collections[1:]:
        scores = [per_question[c][q] for q in shared]
        diff = sum(s - b for s, b in zip(scores, base_scores)) / len(shared)
        result = ttest_rel(scores, base_scores)
        verdict = "significant" if result.pvalue < 0.05 else "NOT significant"
        print(f"   {c:24} mean diff {diff:+.3f}   "
              f"t={result.statistic:+.2f}  p={result.pvalue:.4f}  ({verdict} at 0.05)")
        results[c] = (diff, result.statistic, result.pvalue)
    return results


def main():
    questions = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["questions"]

    client = get_client()          # one client: embedded Qdrant allows only one
    embeddings = get_embeddings()

    scores = {}
    scored = {}
    per_question = {}
    for collection in COLLECTIONS:
        print(f"Evaluating {collection}...")
        registry.check_embedding_model(collection)
        # Each collection needs its OWN labels - chunk IDs differ between them
        qrels = build_qrels(client, collection, questions)
        run = build_run(client, collection, embeddings, questions, qrels)
        scores[collection] = evaluate(Qrels(qrels), Run(run), METRICS)
        scored[collection] = len(qrels)
        per_question[collection] = reciprocal_ranks(qrels, run)

    width = max(len(c) for c in COLLECTIONS) + 3
    print()
    print("metric".ljust(14) + "".join(c.ljust(width) for c in COLLECTIONS))
    print("-" * (14 + width * len(COLLECTIONS)))
    for metric in METRICS:
        row = metric.ljust(14)
        for collection in COLLECTIONS:
            row += f"{scores[collection][metric]:.3f}".ljust(width)
        print(row)

    print()
    for collection in COLLECTIONS:
        print(f"{collection}: {scored[collection]} questions scored")

    tag_metrics = report_by_tag(per_question, questions, COLLECTIONS)
    comparisons = report_significance(per_question, COLLECTIONS)

    for collection in COLLECTIONS:
        with log_run(collection, scores[collection],
                     tag_metrics[collection], scored[collection]):
            pass
    log_comparison(COLLECTIONS[0], comparisons)

    if settings.mlflow_enabled:
        print(f"\nLogged {len(COLLECTIONS)} runs to MLflow "
              f"(experiment: {settings.mlflow_experiment})")


if __name__ == "__main__":
    main()
