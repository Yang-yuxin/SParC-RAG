# Querying the retriever servers

Both servers are started from the repository root (see the README for building the indices):

```bash
uvicorn serve_sparse:app --port 8020 --app-dir retriever_server   # BM25 (needs Elasticsearch on :9200)
uvicorn serve_beir:app   --port 8000 --app-dir retriever_server   # dense (BGE-large + FAISS)
```

Corpus names: `hotpotqa`, `2wikimultihopqa`, `musique` and `wiki` (the DPR Wikipedia
passages, used for NQ and Bamboogle). A corpus that is not loaded gives HTTP 404.

## Dense retriever (port 8000)

```bash
curl -X POST http://localhost:8000/retrieve/ \
  -H "Content-Type: application/json" \
  -d '{
    "retrieval_method": "vector_based_similarity_search",
    "corpus_name": "hotpotqa",
    "query_text": "What year was the director of the film Katyn born",
    "max_hits_count": 6
  }'
```

`curl http://localhost:8000/` lists the loaded corpora and `curl http://localhost:8000/corpus_stats`
their sizes in passages.

## Sparse retriever (port 8020)

```bash
curl -X POST http://localhost:8020/retrieve/ \
  -H "Content-Type: application/json" \
  -d '{
    "retrieval_method": "retrieve_from_elasticsearch",
    "corpus_name": "wiki",
    "query_text": "when did the eagles win the super bowl",
    "max_hits_count": 6,
    "document_type": "title_paragraph_text"
  }'
```

`document_type` is `paragraph_text` (default), `title_paragraph_text` (also match titles; this
is what the SPARC-RAG configs use) or `title`. `curl localhost:9200/_cat/indices` lists the
Elasticsearch indices.

## Response

Both servers return the hits best first:

```json
{
  "retrieval": [
    {"corpus_name": "hotpotqa", "title": "...", "paragraph_text": "...", "score": 0.83}
  ]
}
```

BM25 hits also carry the stored `id`, `url`, `is_abstract` and `paragraph_index` fields, and the
response has a `time_in_seconds` field.
