"""Record how each Qdrant collection was built.

Without this the eval can only log the settings it happens to be running with,
which are not the settings the collection was *ingested* with. Logging those as
MLflow params would be quietly wrong - the classic mistake of recording the
config you have rather than the config that produced the artifact.

Also gives the embedding-model consistency guard something to check: querying a
collection with a different model than it was built with produces no crash, just
silently garbage retrieval.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from rag.config import settings

REGISTRY_PATH = settings.data_dir / "collections.json"

# Settings that change what ends up in the index
TRACKED = [
    "embedding_model",
    "chunk_size",
    "chunk_overlap",
    "preprocess_guide",
    "emoji_mode",
    "enrage_bands",
    "hybrid",
]


def _load_all() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def record(collection: str, chunk_count: int, guides: list[str]) -> None:
    """Save the build config for a collection."""
    entries = _load_all()
    entries[collection] = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_count": chunk_count,
        "guides": sorted(guides),
        **{key: getattr(settings, key) for key in TRACKED},
    }
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(entries, indent=2, default=str) + "\n", encoding="utf-8"
    )


def get(collection: str) -> dict:
    """Build config for a collection, or {} if it was built before this existed."""
    return _load_all().get(collection, {})


def check_embedding_model(collection: str) -> None:
    """Fail loudly if the collection was built with a different embedding model.

    A mismatch does not crash - both models produce vectors of plausible shape
    in incompatible spaces - so retrieval silently degrades to noise.
    """
    recorded = get(collection).get("embedding_model")
    if recorded and recorded != settings.embedding_model:
        raise RuntimeError(
            f"Collection '{collection}' was built with embedding model "
            f"{recorded!r} but settings say {settings.embedding_model!r}. "
            f"Re-ingest, or set EMBEDDING_MODEL={recorded}."
        )
