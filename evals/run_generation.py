"""Phase 2 of generation eval: judge the cached answers.

Four metrics, each answering a different question:

  faithfulness    is what the model said grounded in what it retrieved?
                  (the hallucination measure)
  relevancy       does the answer address the question asked?
                  (an answer can be perfectly grounded and still off-topic)
  refusal_rate    on questions no guide can answer, did it decline?
                  (measured on the 12 unanswerable questions)
  false_refusal   on answerable questions, did it wrongly decline?
                  (the counterpart - a model that refuses everything would
                  score a perfect refusal_rate, which would be meaningless)

    uv run python evals/run_generation.py
    uv run python evals/run_generation.py --collection arch_glacor_bands
"""

import argparse
import json
import time

from evals import judge
from evals.generate_answers import answers_path
from rag.config import settings


def score_record(record, stats):
    """Judge one answer. Returns a dict of metric -> value (or None)."""
    context = "\n\n".join(record["contexts"])
    answer = record["answer"]

    result = {"id": record["id"], "answerable": record["answerable"]}

    # An empty answer is a generation failure, not a refusal and not an answer.
    # Scoring it as either would quietly corrupt the metrics, so it gets its
    # own bucket - this is what reasoning-mode overrun produced.
    result["empty"] = not answer.strip()
    if result["empty"]:
        result["refused"] = None
        result["faithfulness"] = None
        result["relevancy"] = None
        return result

    result["refused"] = judge.refused(answer, stats)

    if record["answerable"]:
        # Only score content quality on answers that tried to answer -
        # a refusal has nothing to be faithful to.
        if result["refused"]:
            result["faithfulness"] = None
            result["relevancy"] = None
        else:
            result["faithfulness"] = judge.faithfulness(context, answer, stats)
            result["relevancy"] = judge.relevancy(record["question"], answer, stats)
    else:
        result["faithfulness"] = None
        result["relevancy"] = None

    return result


def summarise(results, stats, total_calls):
    """Aggregate per-question results into the four headline metrics."""
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    def mean(values):
        clean = [v for v in values if v is not None]
        return sum(clean) / len(clean) if clean else None

    metrics = {
        "faithfulness": mean([r["faithfulness"] for r in answerable]),
        "relevancy": mean([r["relevancy"] for r in answerable]),
        # Correct behaviour on questions the corpus cannot answer
        "refusal_rate": mean([
            1.0 if r["refused"] else 0.0
            for r in unanswerable if r["refused"] is not None
        ]),
        # Wrongly refusing something the corpus *can* answer
        "false_refusal_rate": mean([
            1.0 if r["refused"] else 0.0
            for r in answerable if r["refused"] is not None
        ]),
    }

    # Generation failures: empty output. Kept separate from refusals.
    metrics["empty_answer_rate"] = mean([
        1.0 if r.get("empty") else 0.0 for r in results
    ])

    # If the judge could not be parsed often, the numbers above are unreliable
    metrics["judge_parse_failure_rate"] = (
        stats.get("parse_failures", 0) / total_calls if total_calls else 0.0
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Judge cached answers")
    parser.add_argument("--collection", default="arch_glacor_bands")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    path = answers_path(args.collection)
    if not path.exists():
        raise SystemExit(
            f"No cached answers at {path}\n"
            f"Run: uv run python evals/generate_answers.py --collection {args.collection}"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"][: args.limit] if args.limit else data["records"]

    print(f"collection : {data['collection']}")
    print(f"answerer   : {data['llm_model']}")
    print(f"judge      : {settings.judge_model} (temp {settings.judge_temperature})")
    if settings.judge_model == data["llm_model"]:
        print("             NOTE: judge == answerer, so faithfulness is")
        print("             optimistic (self-preference bias)")
    print(f"answers    : {len(records)}\n")

    stats = {}
    results = []
    started = time.time()
    for i, record in enumerate(records, start=1):
        result = score_record(record, stats)
        results.append(result)

        flag = "EMPTY" if result.get("empty") else ("REFUSED" if result["refused"] else "")
        faith = f"{result['faithfulness']:.2f}" if result["faithfulness"] is not None else "  - "
        rel = f"{result['relevancy']:.2f}" if result["relevancy"] is not None else "  - "
        print(f"[{i:>3}/{len(records)}] faith={faith} rel={rel} "
              f"{'ANS' if result['answerable'] else 'UNA'} {flag:8} {result['id'][:40]}")

    # Rough call count for the parse-failure rate denominator
    total_calls = sum(
        (1 if not r["answerable"] else 1 + judge.MAX_CLAIMS + 1) for r in records
    )
    metrics = summarise(results, stats, total_calls)

    print(f"\n{'-' * 56}")
    for name, value in metrics.items():
        print(f"  {name:26} {value:.3f}" if value is not None else f"  {name:26}    -")
    print(f"{'-' * 56}")
    print(f"  judge retries: {stats.get('retries', 0)}, "
          f"parse failures: {stats.get('parse_failures', 0)}")
    print(f"  elapsed: {(time.time() - started) / 60:.1f} min")

    if metrics["judge_parse_failure_rate"] > 0.05:
        print("\n  !! Judge failed to parse on >5% of calls. Treat these")
        print("     numbers as unreliable and consider a larger judge model.")

    out = path.with_name(path.stem.replace("_answers", "_scores") + ".json")
    out.write_text(
        json.dumps({"metrics": metrics, "per_question": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nper-question scores: {out}")

    if not args.no_mlflow and settings.mlflow_enabled:
        from evals.tracking import log_generation_run

        log_generation_run(args.collection, data, metrics, len(records))
        print(f"logged to MLflow ({settings.mlflow_experiment})")


if __name__ == "__main__":
    main()
