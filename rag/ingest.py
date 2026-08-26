from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag import registry
from rag.config import settings
from rag.preprocess import preprocess
from rag.retrieve import DENSE_VECTOR, SPARSE_VECTOR, TEXT_KEY
from rag.sources import boss_of, guide_paths, style_of

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def load_file(path: Path):
    """Read one guide and split it into chunks tagged with their source."""
    markdown_document = path.read_text(encoding="utf-8")

    if settings.preprocess_guide:
        before = len(markdown_document)
        markdown_document = preprocess(
            markdown_document,
            mode=settings.emoji_mode,
            enrage_bands=settings.enrage_bands,
        )
        pct = 100 * (before - len(markdown_document)) / before
        print(f"   preprocessed: {before} -> {len(markdown_document)} chars ({pct:.1f}% removed)")

    # Split text into sections according to their markdown headers "#, ##, ###"
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    sections = markdown_splitter.split_text(markdown_document)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = text_splitter.split_documents(sections)

    # Stamp every chunk so relevance labels and metadata filters can tell the
    # three guides apart - both hybrids have a section called 'Melee Phase'.
    for chunk in chunks:
        chunk.metadata["source"] = path.name
        chunk.metadata["style"] = style_of(path)
        chunk.metadata["boss"] = boss_of(path)

    print(f"   {len(sections)} sections -> {len(chunks)} chunks")
    return chunks


def load_documents(guide_dir: Path | None = None):
    """Read every guide in the directory."""
    paths = guide_paths(guide_dir or settings.guide_dir)
    if not paths:
        raise FileNotFoundError(f"No .txt guides in {guide_dir or settings.guide_dir}")

    all_chunks = []
    for path in paths:
        print(f"-> {path.name}  (style={style_of(path)})")
        all_chunks.extend(load_file(path))

    print(f"\nTotal: {len(all_chunks)} chunks from {len(paths)} guides")
    return all_chunks


def ingest_dense(chunks, embeddings) -> None:
    """Dense-only collection, via LangChain's helper."""
    if settings.qdrant_url:
        kwargs = {"url": settings.qdrant_url, "api_key": settings.qdrant_api_key}
    else:
        settings.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"path": str(settings.qdrant_path)}

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.collection_name,
        content_payload_key=TEXT_KEY,
        **kwargs,
    )


def ingest_hybrid(chunks, embeddings) -> None:
    """Dense + BM25 sparse in one collection, using named vectors.

    Built with the raw client rather than LangChain so the vector names and
    payload shape are explicit - the eval reads these directly.
    """
    from qdrant_client import models

    from rag import sparse
    from rag.search import get_client

    texts = [c.page_content for c in chunks]

    print("   embedding dense...")
    dense_vectors = embeddings.embed_documents(texts)
    print("   embedding sparse (BM25)...")
    sparse_vectors = sparse.embed_documents(texts)

    client = get_client()
    if client.collection_exists(settings.collection_name):
        client.delete_collection(settings.collection_name)

    client.create_collection(
        settings.collection_name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=len(dense_vectors[0]), distance=models.Distance.COSINE
            )
        },
        # IDF is applied server-side so corpus statistics stay correct
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        },
    )

    client.upsert(
        settings.collection_name,
        points=[
            models.PointStruct(
                id=i,
                vector={
                    DENSE_VECTOR: dense,
                    SPARSE_VECTOR: sp,
                },
                # Same payload shape LangChain writes, so the eval needs no branch
                payload={
                    TEXT_KEY: chunk.page_content,
                    "metadata": chunk.metadata,
                },
            )
            for i, (chunk, dense, sp) in enumerate(
                zip(chunks, dense_vectors, sparse_vectors)
            )
        ],
    )


def main() -> None:
    print(f"Loading guides from {settings.guide_dir}\n")
    chunks = load_documents()

    print("\nInitializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    mode = "hybrid (dense + BM25)" if settings.hybrid else "dense only"
    print(f"Upserting into '{settings.collection_name}' [{mode}]...")

    if settings.hybrid:
        ingest_hybrid(chunks, embeddings)
    else:
        ingest_dense(chunks, embeddings)

    # Record how this collection was built, so the eval logs the config that
    # actually produced it rather than whatever settings it runs with later
    registry.record(
        settings.collection_name,
        chunk_count=len(chunks),
        guides=[p.name for p in guide_paths(settings.guide_dir)],
    )

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
