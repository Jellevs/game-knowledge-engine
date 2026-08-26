"""Low-level retrieval, shared by the app and the eval harness.

Kept separate from search.py (which builds LangChain objects) so the eval can
call it without any LangChain wrapping - it needs raw chunk IDs and scores.
"""

from qdrant_client import models

from rag.config import settings

# Qdrant schema. Fixed by how ingest.py writes collections - not configurable,
# because changing one of these without re-ingesting silently returns nothing.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
TEXT_KEY = "text"      # payload key holding the chunk text (LangChain's default)


def is_hybrid_collection(client, collection) -> bool:
    """True if the collection was built with named dense + sparse vectors.

    Lets one eval run compare dense-only and hybrid collections side by side.
    """
    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    sparse = info.config.params.sparse_vectors
    return bool(sparse) and isinstance(vectors, dict) and DENSE_VECTOR in vectors


def search(client, collection, dense_vector, sparse_vector=None, limit=10):
    """Return scored points, hybrid if the collection supports it.

    Hybrid runs both retrievers and merges with Reciprocal Rank Fusion: each
    result scores 1/(60 + rank) from each retriever, summed. Positions are used
    rather than scores because a cosine similarity and a BM25 score are not on
    comparable scales.
    """
    if sparse_vector is not None and is_hybrid_collection(client, collection):
        return client.query_points(
            collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR,
                    limit=settings.hybrid_prefetch,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using=SPARSE_VECTOR,
                    limit=settings.hybrid_prefetch,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        ).points

    return client.query_points(collection, query=dense_vector, limit=limit).points
