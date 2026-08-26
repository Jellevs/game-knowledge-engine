"""Phase 1 of generation eval: answer every golden question, save to disk.

Split from judging on purpose:

  - VRAM. The answerer and the judge would otherwise be loaded alternately for
    every question. Even when they are the same model, keeping the phases apart
    means one load and no swapping.
  - Re-judging. Prompts and metrics change more often than answers do. Cached
    answers mean you can re-score in minutes without regenerating anything.
  - Inspection. The saved file is the raw material for spot-checking the judge
    against your own reading, which is the only way to know if the scores mean
    anything.

    uv run python evals/generate_answers.py
    uv run python evals/generate_answers.py --limit 10        # smoke test
"""

import argparse
import json
import time

from rag.config import settings

GOLDEN_PATH = settings.data_dir / "golden" / "arch_glacor.json"
ANSWERS_DIR = settings.data_dir / "eval_runs"


def answers_path(collection: str):
    return ANSWERS_DIR / f"{collection}_answers.json"


def main():
    parser = argparse.ArgumentParser(description="Generate answers for the golden set")
    parser.add_argument("--collection", default="arch_glacor_bands")
    parser.add_argument("--limit", type=int, help="Only the first N questions")
    parser.add_argument("-k", type=int, default=settings.top_k)
    args = parser.parse_args()

    settings.collection_name = args.collection

    questions = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["questions"]
    if args.limit:
        # Interleave so a small limit still includes unanswerable questions
        answerable = [q for q in questions if q["expected_section"]]
        unanswerable = [q for q in questions if not q["expected_section"]]
        take = max(1, args.limit // 4)
        questions = answerable[: args.limit - take] + unanswerable[:take]

    from rag.pipeline import build_rag_chain
    from rag.search import build_retriever

    print(f"collection : {args.collection}")
    print(f"answerer   : {settings.llm_model}")
    print(f"questions  : {len(questions)}   top_k={args.k}\n")

    # One retriever, shared: the contexts recorded below must be exactly the
    # ones the chain used, or faithfulness is scored against the wrong text
    retriever = build_retriever(k=args.k)
    chain = build_rag_chain(k=args.k, retriever=retriever)

    records = []
    started = time.time()
    for i, q in enumerate(questions, start=1):
        t0 = time.time()

        # Retrieve separately as well, so the judge can see exactly what the
        # model was given. Without this we could only score the answer, not
        # whether it was faithful to its actual context.
        docs = retriever.invoke(q["question"])
        answer = chain.invoke(q["question"])
        elapsed = time.time() - t0

        records.append({
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "contexts": [d.page_content for d in docs],
            "context_sources": [d.metadata.get("source", "") for d in docs],
            "expected_answer": q["expected_answer"],
            "tags": q.get("tags", []),
            "answerable": bool(q["expected_section"]),
            "seconds": round(elapsed, 2),
        })

        print(f"[{i:>3}/{len(questions)}] {elapsed:5.1f}s  {q['id'][:44]:44} "
              f"{answer[:50].replace(chr(10), ' ')}")

    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    out = answers_path(args.collection)
    out.write_text(
        json.dumps(
            {
                "collection": args.collection,
                "llm_model": settings.llm_model,
                "top_k": args.k,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    total = time.time() - started
    print(f"\n{len(records)} answers in {total/60:.1f} min "
          f"({total/len(records):.1f}s each)")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
