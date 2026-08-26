"""BM25 sparse vectors, the keyword half of hybrid search.

Dense embeddings compress meaning into 384 numbers, which is exactly why they
lose small exact details - '0-2500%' and '2500%+' end up nearly identical.
BM25 does the opposite: it understands nothing, but scores rare exact tokens
enormously. Each covers the other's blind spot.

fastembed's "Qdrant/bm25" produces the sparse vectors; Qdrant applies the IDF
weighting server-side (see Modifier.IDF in ingest.py), so corpus statistics stay
correct as documents are added.
"""

from fastembed import SparseTextEmbedding
from qdrant_client import models

from rag import enrage

MODEL_NAME = "Qdrant/bm25"

_model = None


def get_sparse_model():
    """Loaded once and reused - instantiating it per call is slow."""
    global _model
    if _model is None:
        _model = SparseTextEmbedding(model_name=MODEL_NAME)
    return _model


def _to_qdrant(embedding):
    return models.SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )


def embed_documents(texts):
    """Sparse vectors for chunks being indexed."""
    return [_to_qdrant(e) for e in get_sparse_model().embed(texts)]


def embed_query(text):
    """Sparse vector for a query.

    query_embed differs from embed: for BM25 the query side skips term-frequency
    weighting, since a query mentioning a word twice does not make it twice as
    important.

    Enrage band tokens are appended here so the query and the documents speak
    the same language. Applied to the sparse side only - 'enrageband3000' is
    meaningless to a sentence-transformer and would just add noise to the dense
    vector. If the collection was built without band annotation these extra
    terms simply match nothing and contribute zero.
    """
    text = enrage.annotate_query(text)
    return _to_qdrant(next(iter(get_sparse_model().query_embed(text))))
