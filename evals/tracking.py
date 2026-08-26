"""MLflow logging for eval runs.

One MLflow run per collection. Params come from the *collection registry*, not
from current settings, so each run records the config that actually produced
the index it scored.

Local file store by default (./mlruns). Set MLFLOW_TRACKING_URI, or
DAGSHUB_REPO_OWNER + DAGSHUB_REPO_NAME, to push to DagsHub instead.
"""

import contextlib
import os

from rag import registry
from rag.config import settings

_dagshub_ready = False


def _metric_name(name: str) -> str:
    """MLflow metric names allow only alphanumerics, _ - . space and /.

    Our ranx metrics are 'hit_rate@1', 'mrr@10' - the '@' is rejected, so it
    becomes '_at_': hit_rate_at_1, mrr_at_10.
    """
    return name.replace("@", "_at_")


def _setup():
    """Point MLflow at DagsHub if configured, otherwise the local file store."""
    global _dagshub_ready
    import mlflow

    owner = os.getenv("DAGSHUB_REPO_OWNER")
    name = os.getenv("DAGSHUB_REPO_NAME")

    if owner and name and not _dagshub_ready:
        import dagshub

        dagshub.init(repo_owner=owner, repo_name=name, mlflow=True)
        _dagshub_ready = True
    elif uri := os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(uri)

    mlflow.set_experiment(settings.mlflow_experiment)
    return mlflow


@contextlib.contextmanager
def log_run(collection, metrics, tag_metrics, question_count):
    """Log one collection's eval as an MLflow run.

    Silently does nothing when mlflow_enabled is False, so the eval still runs
    on a machine without tracking configured.
    """
    if not settings.mlflow_enabled:
        yield None
        return

    mlflow = _setup()
    build = registry.get(collection)

    with mlflow.start_run(run_name=collection) as run:
        mlflow.set_tag("collection", collection)
        mlflow.set_tag("built_at", build.get("built_at", "unknown"))

        if build:
            mlflow.log_params({k: v for k, v in build.items() if k != "guides"})
            mlflow.log_param("guide_count", len(build.get("guides", [])))
        else:
            # Collection predates the registry - record that rather than
            # logging today's settings and pretending they were used
            mlflow.log_param("build_config", "unknown (built before registry)")

        mlflow.log_param("questions_scored", question_count)
        mlflow.log_metrics({_metric_name(k): float(v) for k, v in metrics.items()})
        mlflow.log_metrics(
            {_metric_name(f"tag_{k}"): float(v) for k, v in tag_metrics.items()}
        )

        yield run


def log_generation_run(collection, answer_data, metrics, question_count):
    """Log a generation eval into the same experiment as the retrieval runs.

    Same experiment on purpose: retrieval and generation quality for one config
    belong side by side, since a faithfulness drop is often a retrieval problem.
    """
    if not settings.mlflow_enabled:
        return

    mlflow = _setup()
    build = registry.get(collection)

    with mlflow.start_run(run_name=f"generation_{collection}"):
        mlflow.set_tag("kind", "generation")
        mlflow.set_tag("collection", collection)

        if build:
            mlflow.log_params({k: v for k, v in build.items() if k != "guides"})
        mlflow.log_param("llm_model", answer_data["llm_model"])
        mlflow.log_param("judge_model", settings.judge_model)
        mlflow.log_param("judge_independent",
                         settings.judge_model != answer_data["llm_model"])
        mlflow.log_param("top_k", answer_data["top_k"])
        mlflow.log_param("questions_scored", question_count)

        mlflow.log_metrics({
            _metric_name(k): float(v) for k, v in metrics.items() if v is not None
        })


def log_comparison(baseline, comparisons):
    """Log the paired significance tests as one summary run."""
    if not settings.mlflow_enabled:
        return

    mlflow = _setup()
    with mlflow.start_run(run_name=f"comparison_vs_{baseline}"):
        mlflow.set_tag("kind", "significance")
        mlflow.log_param("baseline", baseline)
        for name, (diff, t, p) in comparisons.items():
            mlflow.log_metric(_metric_name(f"{name}_mrr_diff"), float(diff))
            mlflow.log_metric(_metric_name(f"{name}_t"), float(t))
            mlflow.log_metric(_metric_name(f"{name}_p"), float(p))
