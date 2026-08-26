"""One place where chat models are constructed.

Two jobs:

1. Turn OFF reasoning mode. Qwen3-family models emit a long internal <think>
   block by default. Measured cost of leaving it on: ~98s per answer, two of
   eight answers empty (the whole output budget spent thinking) and two more
   truncated mid-sentence. RAG answering is extraction from supplied context,
   not a reasoning task - the thinking is pure overhead.

2. Be the single seam for swapping providers. Moving from Ollama to a hosted
   API later means changing this function, not every call site.
"""

from rag.config import settings


def get_chat_model(model: str | None = None, temperature: float = 0.0):
    """A chat model with reasoning disabled and a bounded output length."""
    from langchain_ollama import ChatOllama

    kwargs = {
        "model": model or settings.llm_model,
        "temperature": temperature,
        # Cap output so a rambling answer cannot run indefinitely. Generous
        # enough for a full rotation listing.
        "num_predict": settings.max_output_tokens,
    }

    try:
        # langchain-ollama >= 0.3 exposes this; passes think=false to Ollama
        return ChatOllama(reasoning=False, **kwargs)
    except TypeError:
        # Older versions have no such parameter - fall back rather than crash
        return ChatOllama(**kwargs)
