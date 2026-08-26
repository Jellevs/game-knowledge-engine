from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag import retrieve, sparse
from rag.config import settings


_client = None
_embeddings = None


def get_client() -> QdrantClient:
    """One Qdrant client per process, reused.

    Embedded Qdrant takes an exclusive lock on its storage folder, so a second
    client in the same process fails with 'already accessed by another
    instance'. Caching makes that impossible rather than relying on every
    caller to pass one around. Goes away entirely in server mode.
    """
    global _client
    if _client is None:
        if settings.qdrant_url:
            _client = QdrantClient(
                url=settings.qdrant_url, api_key=settings.qdrant_api_key
            )
        else:
            _client = QdrantClient(path=str(settings.qdrant_path))
    return _client


def get_embeddings() -> HuggingFaceEmbeddings:
    """Loaded once - instantiating re-reads the model from disk each time."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return _embeddings


class HybridRetriever(BaseRetriever):
    """Dense + BM25 retrieval fused with RRF, as a LangChain retriever.

    Hand-written because LangChain's Qdrant hybrid support hides the vector
    names and fusion settings; the eval queries the same collection directly and
    the two must agree exactly.
    """

    client: Any
    collection_name: str
    embeddings: Any
    k: int = 3

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        dense_vector = self.embeddings.embed_query(query)
        sparse_vector = sparse.embed_query(query)
        hits = retrieve.search(
            self.client, self.collection_name, dense_vector, sparse_vector, limit=self.k
        )
        docs = []
        for hit in hits:
            payload = hit.payload or {}
            docs.append(
                Document(
                    page_content=payload.get(retrieve.TEXT_KEY, ""),
                    metadata=payload.get("metadata", {}),
                )
            )
        return docs


def build_retriever(k: int | None = None):
    """Build a retriever over the existing Qdrant collection.

    Returns a hybrid retriever when the collection carries sparse vectors,
    otherwise the plain dense one.
    """
    client = get_client()
    k = k or settings.top_k

    if retrieve.is_hybrid_collection(client, settings.collection_name):
        return HybridRetriever(
            client=client,
            collection_name=settings.collection_name,
            embeddings=get_embeddings(),
            k=k,
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=get_embeddings(),
        content_payload_key=retrieve.TEXT_KEY,
    )
    return vector_store.as_retriever(search_kwargs={"k": k})
