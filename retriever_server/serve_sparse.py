"""BM25 retriever server backed by Elasticsearch 7.10.

Run from the repository root, with the indices built by build_index.py:
    uvicorn serve_sparse:app --port 8020 --app-dir retriever_server
Elasticsearch is reached at ELASTICSEARCH_HOST:ELASTICSEARCH_PORT (default localhost:9200).

POST /retrieve/ with {"retrieval_method": "retrieve_from_elasticsearch", "corpus_name",
"query_text", "max_hits_count", ...} (the other arguments of
UnifiedRetriever.retrieve_from_elasticsearch) returns {"retrieval": [...], "time_in_seconds"}.
A corpus without an Elasticsearch index gives HTTP 404.
"""

import os
from time import perf_counter

from elasticsearch.exceptions import NotFoundError
from fastapi import FastAPI, HTTPException, Request

from unified_retriever import UnifiedRetriever

retriever = UnifiedRetriever(
    host=os.environ.get("ELASTICSEARCH_HOST", "localhost"),
    port=int(os.environ.get("ELASTICSEARCH_PORT", "9200")),
)

app = FastAPI()


@app.get("/")
async def index():
    return {"message": "Hello! This is a retriever server."}


@app.post("/retrieve/")
async def retrieve(request: Request):
    arguments = await request.json()
    retrieval_method = arguments.pop("retrieval_method", None)
    if retrieval_method != "retrieve_from_elasticsearch":
        raise HTTPException(status_code=400, detail="retrieval_method must be 'retrieve_from_elasticsearch'.")
    start_time = perf_counter()
    try:
        retrieval = retriever.retrieve_from_elasticsearch(**arguments)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"No Elasticsearch index {arguments.get('corpus_name')!r}.")
    time_in_seconds = round(perf_counter() - start_time, 1)
    return {"retrieval": retrieval, "time_in_seconds": time_in_seconds}
