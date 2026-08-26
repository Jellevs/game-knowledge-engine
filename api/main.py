from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

from rag.config import settings


_state =  {}


async def lifespan(app: FastAPI):
    from rag.pipeline import build_rag_chain
    from rag.search import build_retriever, get_client


    _state['retriever'] = build_retriever()
    _state['chain'] = build_rag_chain(retriever=_state['retriever'])
    _state['client'] = get_client()

    yield

    _state.clear()


app = FastAPI(title='PvME RAG', lifespan=lifespan)


class Query(BaseModel):
    question: str


@app.get('/health')
def health():
    try:
        chunks =  _state['client'].count(settings.collection_name).count
    except Exception as exc:
        raise HTTPException(503, f"Qdrant unavailable: {exc}")

    return {
        'status': 'ok',
        'collection': settings.collection_name,
        'chunks': chunks,
        'llm_model': settings.llm_model
    }


@app.post('/query')
def query(request: Query):
    started = time.perf_counter()
    
    docs = _state['retriever'].invoke(request.question)
    answer = _state['chain'].invoke(request.question)

    return {
        'question': request.question,
        'answer': answer,
        'sources': [
            {
                'guide': doc.metadata.get('style', ''),
                'section': doc.metadata.get('Header 3') or doc.metadata.get('Header 2'),
            }
            for doc in docs
        ],
        'latency_ms': int((time.perf_counter() -  started) * 1000)
    }






