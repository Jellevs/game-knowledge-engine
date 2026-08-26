from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag.generator import build_generator, format_docs
from rag.search import build_retriever


def build_rag_chain(k: int | None = None, retriever=None):
    """Assemble retriever -> prompt -> LLM -> string.

    An existing retriever can be passed in. The eval needs to record the exact
    chunks the model saw, so it retrieves separately and hands the same
    retriever here - otherwise the saved contexts are only *probably* what the
    chain used, and faithfulness would be scored against the wrong text.
    """
    retriever = retriever or build_retriever(k=k)
    prompt, llm = build_generator()

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
