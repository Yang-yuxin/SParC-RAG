"""Dense retriever server: BAAI/bge-large-en-v1.5 query embeddings, exact inner-product search (FAISS FlatIP).

Run from the repository root:
    uvicorn serve_beir:app --port 8000 --app-dir retriever_server

At startup it loads, for every corpus in SPARC_RAG_DENSE_CORPORA (comma-separated; default
hotpotqa,2wikimultihopqa,musique,wiki), the passages (dataloaders.py) and the index built by
build_beir_index.py. Serving all four needs roughly 140 GB of RAM, most of it for `wiki`.

POST /retrieve/ with {"retrieval_method": "vector_based_similarity_search", "corpus_name",
"query_text", "max_hits_count"} returns {"retrieval": [{"corpus_name", "title",
"paragraph_text", "score"}, ...]}, best match first. A corpus that is not loaded gives HTTP 404.
"""

import os
from typing import Dict

from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from fastapi import FastAPI, HTTPException, Request

from dataloaders import DENSE_CORPORA, Corpus, load_corpus
from utils import EMBEDDING_MODEL, INDEX_EXT, INDEX_PREFIX, SCORE_FUNCTION, ConcreteFlatIPFaissSearch, index_directory

CORPORA = [name.strip() for name in os.environ.get("SPARC_RAG_DENSE_CORPORA", ",".join(DENSE_CORPORA)).split(",")]

model = models.SentenceBERT(EMBEDDING_MODEL)
corpora: Dict[str, Corpus] = {}
retrievers: Dict[str, EvaluateRetrieval] = {}

for corpus_name in CORPORA:
    print(f"Loading corpus: {corpus_name}")
    corpora[corpus_name] = load_corpus(corpus_name)

    index_dir = index_directory(corpus_name)
    index_file = os.path.join(index_dir, f"{INDEX_PREFIX}.{INDEX_EXT}.faiss")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"{index_file} not found. Build it with retriever_server/build_beir_index.py {corpus_name}.")
    print(f"Loading FAISS index: {index_file}")
    faiss_search = ConcreteFlatIPFaissSearch(model, batch_size=128)
    faiss_search.load(input_dir=index_dir, prefix=INDEX_PREFIX, ext=INDEX_EXT)
    retrievers[corpus_name] = EvaluateRetrieval(faiss_search, score_function=SCORE_FUNCTION)

print(f"Serving corpora: {list(retrievers)}")

app = FastAPI()


@app.get("/")
async def index_route():
    return {"message": f"Dense retriever serving: {list(retrievers)}"}


@app.post("/retrieve/")
async def retrieve(request: Request):
    body = await request.json()
    if body.get("retrieval_method") != "vector_based_similarity_search":
        raise HTTPException(status_code=400, detail="retrieval_method must be 'vector_based_similarity_search'.")
    corpus_name = body.get("corpus_name")
    if corpus_name not in retrievers:
        raise HTTPException(
            status_code=404, detail=f"Corpus {corpus_name!r} is not loaded; loaded corpora: {list(retrievers)}."
        )
    if not isinstance(body.get("query_text"), str):
        raise HTTPException(status_code=400, detail="query_text must be a string.")
    max_hits_count = body.get("max_hits_count", 5)

    corpus = corpora[corpus_name]
    # EvaluateRetrieval searches the top max(k_values) = 1000 passages; keep the best max_hits_count.
    scores = retrievers[corpus_name].retrieve(corpus, {0: body["query_text"]})[0]
    top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max_hits_count]
    return {
        "retrieval": [
            {
                "corpus_name": corpus_name,
                "title": corpus[doc_id].get("title"),
                "paragraph_text": corpus[doc_id].get("text"),
                "score": score,
            }
            for doc_id, score in top_hits
        ]
    }


@app.get("/corpus_stats")
async def corpus_stats():
    return {"corpus_sizes": {corpus_name: len(corpora[corpus_name]) for corpus_name in retrievers}}
