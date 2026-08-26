"""Every tunable value in one place.

The point is experiments, no hardcoded variables.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    # All *.txt directly inside guide_dir are ingested as separate sources.
    guide_dir: Path = PROJECT_ROOT / "data"
    qdrant_path: Path = PROJECT_ROOT / "data" / "qdrant_db"

    # Qdrant server mode (takes precedence over qdrant_path when set)
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    collection_name: str = "arch_glacor"

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "qwen3.5:9b"
    # Judge for RAGAS. Same model as the answerer for now: it fits 8GB VRAM in
    # one load, so no swapping. Caveat: a model grading its own output shows
    # self-preference bias, so absolute faithfulness is flattering. Separate
    # setting so an independent judge is a one-line change.
    judge_model: str = "qwen3.5:9b"
    # Judging must be reproducible - a regression should be a real regression
    judge_temperature: float = 0.0
    # Cap generated length. Qwen3 reasoning is disabled in rag/llm.py;
    # without both, answers came back empty or truncated mid-sentence.
    max_output_tokens: int = 512

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Preprocessing
    preprocess_guide: bool = True

    # How to render abbreviated shortcodes:
    # full  -> "Fractured Staff of Armadyl"
    # alias -> "fsoa"
    # both  -> "Fractured Staff of Armadyl (fsoa)"
    emoji_mode: Literal["full", "alias", "both"] = "both"

    # Append band tokens so BM25 can tell '0-2500%' from '2500%+'
    enrage_bands: bool = True

    # Experiment tracking
    mlflow_enabled: bool = True
    mlflow_experiment: str = "pvme-retrieval"

    # Retrieval
    top_k: int = 3
    hybrid: bool = True

    # How many candidates each retriever contributes before RRF fusion
    hybrid_prefetch: int = 40


settings = Settings()
