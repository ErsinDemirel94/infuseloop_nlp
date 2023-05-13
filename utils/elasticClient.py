from elasticsearch import Elasticsearch
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.schema import Document
from typing import List

class ElasticClient:
    
    
    def __init__(self, host: str, port: int, username: str, password: str, use_ssl: bool = True):
        
        self.es_client = Elasticsearch(hosts=[{"host": host, "port": port}], 
                                       http_auth=(username, password), 
                                       use_ssl=use_ssl, 
                                       verify_certs=True, 
                                       scheme="https")
        
        self.document_store = ElasticsearchDocumentStore(host=host, 
                                                         port=port, 
                                                         scheme='https', 
                                                         username=username, 
                                                         password=password, 
                                                         index="document",  
                                                         embedding_field="embedding", 
                                                         embedding_dim=768)

    def test_connection(self) -> bool:
        return self.es_client.ping()
    
    def get_document_store(self) -> ElasticsearchDocumentStore:
        # Get DocumentStore object for further operations like retrieval, update or write
        return self.document_store
    
    def search_with_query(self, index_name: str, query: str, top_k: int = 10) -> List[dict]:
        search_body = {
            "query": {
                "query_string": {
                    "query": query
                }
            },
            "size": top_k,
        }

        result = self.es_client.search(index=index_name, body=search_body)
        return [hit["_source"] for hit in result["hits"]["hits"]]
   
    def delete_documents(self, index_name: str, ids: List[str]):
        for doc_id in ids:
            self.es_client.delete(index=index_name, id=doc_id)    
    
    def get_document_by_id(self, index_name: str, doc_id: str):
        return self.es_client.get(id=doc_id, index=index_name)

    def get_all_ids(self, index_name: str):
        # Get ids until 1000th record for testing.
        # For better option, use scroll.
        
        res = self.es_client.search(index=index_name, body={"query": {"match_all": {}}, "size": 1000, "fields": ["_id"]})
        ids = [d['_id'] for d in res['hits']['hits']]
        
        return ids
    
    def create_index(self, index_name: str):
        self.es_client.create_index(index=index_name)


# TO DO:

# Test class.
# How to handle authentication?
