from collections import OrderedDict
from typing import Dict, List

from elasticsearch import Elasticsearch


class ElasticsearchRetriever:
    """BM25 paragraph and title retrieval from the Elasticsearch index named after the corpus.

    In the bool queries below, `must` acts as AND, `should` as OR, and `filter` restricts the
    matches without affecting the score.
    """

    def __init__(self, host: str = "localhost", port: int = 9200):
        self._es = Elasticsearch([host], scheme="http", port=port, timeout=30)

    def retrieve_paragraphs(
        self,
        corpus_name: str,
        query_text: str = None,
        is_abstract: bool = None,
        query_title_field_too: bool = False,
        max_buffer_count: int = 100,
        max_hits_count: int = 10,
    ) -> List[Dict]:

        query = {
            "size": max_buffer_count,
            # what records are needed in result
            "_source": ["id", "title", "paragraph_text", "url", "is_abstract", "paragraph_index"],
            "query": {
                "bool": {
                    "should": [],
                }
            },
        }

        if query_text is not None:
            # must is too strict for this:
            query["query"]["bool"]["should"].append({"match": {"paragraph_text": query_text}})

        if query_title_field_too:
            query["query"]["bool"]["should"].append({"match": {"title": query_text}})

        if is_abstract is not None:
            query["query"]["bool"]["filter"] = [{"match": {"is_abstract": is_abstract}}]

        assert query["query"]["bool"]["should"], "Paragraph retrieval needs a query_text."

        result = self._es.search(index=corpus_name, body=query)

        retrieval = []
        if result.get("hits") is not None and result["hits"].get("hits") is not None:
            retrieval = result["hits"]["hits"]
            text2retrieval = OrderedDict()
            for item in retrieval:
                text = item["_source"]["paragraph_text"].strip().lower()
                text2retrieval[text] = item
            retrieval = list(text2retrieval.values())

        retrieval = sorted(retrieval, key=lambda e: e["_score"], reverse=True)
        retrieval = retrieval[:max_hits_count]
        for retrieval_ in retrieval:
            retrieval_["_source"]["score"] = retrieval_["_score"]
        retrieval = [e["_source"] for e in retrieval]

        for retrieval_ in retrieval:
            retrieval_["corpus_name"] = corpus_name

        return retrieval

    def retrieve_titles(
        self,
        query_text: str,
        corpus_name: str,
        max_buffer_count: int = 100,
        max_hits_count: int = 10,
    ) -> List[Dict]:

        query = {
            "size": max_buffer_count,
            # what records are needed in the result.
            "_source": ["id", "title", "paragraph_text", "url", "is_abstract", "paragraph_index"],
            "query": {
                "bool": {
                    "must": [
                        {"match": {"title": query_text}},
                    ],
                    "filter": [
                        {"match": {"is_abstract": True}},  # so that same title doesn't show up many times.
                    ],
                }
            },
        }

        result = self._es.search(index=corpus_name, body=query)

        retrieval = []
        if result.get("hits") is not None and result["hits"].get("hits") is not None:
            retrieval = result["hits"]["hits"]
            text2retrieval = OrderedDict()
            for item in retrieval:
                text = item["_source"]["title"].strip().lower()
                text2retrieval[text] = item
            retrieval = list(text2retrieval.values())[:max_hits_count]

        retrieval = [e["_source"] for e in retrieval]

        for retrieval_ in retrieval:
            retrieval_["corpus_name"] = corpus_name

        return retrieval
