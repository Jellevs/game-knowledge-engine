import re

from langchain_core.prompts import PromptTemplate

from rag.config import settings
from rag.llm import get_chat_model


# '[enrageband0 enrageband500 ...]' appended at ingest for BM25 only
BAND_TOKENS_RE = re.compile(r"\s*\[(?:enrageband\d+\s*)+\]")

TEMPLATE = """
You are an expert RuneScape 3 assistant. Answer the user's question using ONLY the
provided context. If the context does not contain the answer, say:
"I don't have enough information to answer that."

Answer directly. Do not describe the context, do not mention what the context does
or does not contain, and do not comment on whether links were available.

If a relevant link appears in the context (for example an example kill video),
include the full URL exactly as written. Never invent or modify a URL.

Context:
{context}

Question:
{question}
"""


def build_generator():
    llm = get_chat_model(settings.llm_model)
    prompt = PromptTemplate.from_template(TEMPLATE)
    return prompt, llm


STYLE_LABELS = {
    "necromancy": "Necromancy",
    "melee_magic": "Hybrid Melee/Magic",
    "melee_ranged": "Hybrid Melee/Ranged",
}


def format_docs(docs) -> str:
    """Chunk text with its source guide labelled, retrieval-only tokens removed.

    The label matters. Retrieval returns chunks from all three guides, and the
    text alone gives no clue which is which - an example-kill line reads
    'Example kill video, Arch-Glacor: ...' whether it came from the Necromancy
    guide or a hybrid one. Without the label the model cannot answer 'is there
    a video for the Melee/Magic hybrid?' even holding the right chunk, and
    correctly refuses instead. Measured: 8 of 10 false refusals had the answer
    in context.

    Enrage band tokens are stripped here rather than at ingest, because the
    index still needs them for BM25.
    """
    parts = []
    for doc in docs:
        style = doc.metadata.get("style", "")
        label = STYLE_LABELS.get(style, style or "guide")
        text = strip_retrieval_tokens(doc.page_content)
        parts.append(f"[From the {label} guide]\n{text}")
    return "\n\n".join(parts)


def strip_retrieval_tokens(text: str) -> str:
    return BAND_TOKENS_RE.sub("", text)
