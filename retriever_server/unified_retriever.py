from typing import Dict, List

from elasticsearch_retriever import ElasticsearchRetriever


class UnifiedRetriever:
    """Request-level interface of the BM25 server (serve_sparse.py) over ElasticsearchRetriever."""

    def __init__(self, host: str = "localhost", port: int = 9200):
        self._elasticsearch_retriever = ElasticsearchRetriever(host=host, port=port)

    def retrieve_from_elasticsearch(
        self,
        query_text: str,
        max_hits_count: int = 3,
        max_buffer_count: int = 100,
        document_type: str = "paragraph_text",
        corpus_name: str = None,
    ) -> List[Dict]:
        """Retrieve paragraphs (document_type "paragraph_text" or "title_paragraph_text") or titles.

        "title_paragraph_text" matches the query against the title field as well. For the
        hotpotqa index, paragraph retrieval is restricted to abstracts (is_abstract=True).
        """
        assert document_type in ("title", "paragraph_text", "title_paragraph_text")

        if document_type in ("paragraph_text", "title_paragraph_text"):
            is_abstract = True if corpus_name == "hotpotqa" else None  # None: no is_abstract filter
            query_title_field_too = document_type == "title_paragraph_text"
            return self._elasticsearch_retriever.retrieve_paragraphs(
                query_text=query_text,
                is_abstract=is_abstract,
                max_hits_count=max_hits_count,
                corpus_name=corpus_name,
                query_title_field_too=query_title_field_too,
                max_buffer_count=max_buffer_count,
            )

        return self._elasticsearch_retriever.retrieve_titles(
            query_text=query_text, max_hits_count=max_hits_count, corpus_name=corpus_name
        )
