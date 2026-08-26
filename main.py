"""CLI entrypoint. Run from the project root:

    uv run python main.py "What is the best necromancy armor for Arch-Glacor?"
    uv run python main.py --ingest
"""

import argparse

from rag.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="RS3 PvME RAG assistant")

    # Ask the question
    parser.add_argument("question", nargs="?", help="Question to ask the assistant")

    # Toggle to rebuild the vectore database
    parser.add_argument("--ingest", action="store_true", help="Rebuild the vector store")

    # Top chunks to retrieve
    parser.add_argument("-k", type=int, default=settings.top_k, help="Chunks to retrieve")

    # Set the collection name
    parser.add_argument("--collection", help="Override the Qdrant collection name")

    # Add pre-processing yes or no
    parser.add_argument("--preprocess", action=argparse.BooleanOptionalAction,
                        default=None, help="Clean the guide markup (default: on)")

    # How to abbreviate the emojis
    parser.add_argument("--emoji-mode", choices=["full", "alias", "both"],
                        help="How to render abbreviations (default: both)")

    # Add information to denote different enrages
    parser.add_argument("--enrage-bands", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Append band tokens so BM25 can separate enrage ranges")

    # Toggle hybrid search
    parser.add_argument("--hybrid", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Index dense + BM25 sparse vectors, fused with RRF")
    args = parser.parse_args()

    if args.collection:
        settings.collection_name = args.collection
    if args.preprocess is not None:
        settings.preprocess_guide = args.preprocess
    if args.emoji_mode:
        settings.emoji_mode = args.emoji_mode
    if args.enrage_bands is not None:
        settings.enrage_bands = args.enrage_bands
    if args.hybrid is not None:
        settings.hybrid = args.hybrid

    if args.ingest:
        # Add late import to speed up questions
        from rag.ingest import main as ingest_main

        # Build the vector database
        ingest_main()
        return

    if not args.question:
        parser.error("provide a question, or use --ingest")


    # Build the whole pipeline
    from rag.pipeline import build_rag_chain

    # build_rag_chain: connect to  Qdrant -> load embedding model -> create receiver -> load prompt -> connect ollama -> return chain object
    print(build_rag_chain(k=args.k).invoke(args.question))


if __name__ == "__main__":
    main()
