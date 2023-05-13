from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever
from typing import List

class BM25PassageRetriever:
    def __init__(self, document_store: ElasticsearchDocumentStore):
        self.retriever = BM25Retriever(document_store=document_store)

    def retrieve_passages(self, query: str, top_k: int = 10) -> List[str]:
        results = self.retriever.retrieve(query=query, top_k=top_k)
        
        # Preprocess the output by removing results with score lower than n
        # retrieve text, score, and source, split only?
        return results
