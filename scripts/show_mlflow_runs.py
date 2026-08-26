"""Print logged MLflow runs without the web UI.

Useful when the UI misbehaves, and as a quick check that the eval actually
logged what you think it did.

    uv run python scripts/show_mlflow_runs.py
"""

import mlflow

from rag.config import settings

KEY_METRICS = ["mrr_at_10", "hit_rate_at_1", "hit_rate_at_3"]
KEY_PARAMS = ["preprocess_guide", "emoji_mode", "hybrid", "enrage_bands", "chunk_count"]


def main():
    print(f"tracking uri : {mlflow.get_tracking_uri()}")
    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment)
    if experiment is None:
        print(f"No experiment named {settings.mlflow_experiment!r}. Run the eval first.")
        return

    print(f"experiment   : {experiment.name} (id {experiment.experiment_id})")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.mrr_at_10 DESC"],
    )
    if runs.empty:
        print("No runs found.")
        return

    print(f"runs         : {len(runs)}\n")
    header = f"{'run':26}" + "".join(m.replace('_at_', '@').ljust(14) for m in KEY_METRICS)
    print(header)
    print("-" * len(header))
    for _, row in runs.iterrows():
        name = str(row.get("tags.mlflow.runName", "?"))[:25]
        line = f"{name:26}"
        for metric in KEY_METRICS:
            value = row.get(f"metrics.{metric}")
            line += (f"{value:.3f}" if value == value and value is not None else "-").ljust(14)
        print(line)

    print("\nparams per run:")
    for _, row in runs.iterrows():
        name = str(row.get("tags.mlflow.runName", "?"))
        parts = [f"{p}={row.get(f'params.{p}')}" for p in KEY_PARAMS
                 if row.get(f"params.{p}") is not None]
        print(f"  {name:26} {'  '.join(parts) if parts else '(no build params)'}")


if __name__ == "__main__":
    main()
