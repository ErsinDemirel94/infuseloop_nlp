# API_KEY: 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

# HF endpoint kullanılabilir mi?? 

from typing import List
from haystack.pipelines import Pipeline
from haystack.nodes import Shaper, PromptNode, PromptTemplate
from haystack.schema import Document
from haystack.nodes import BM25Retriever
from haystack.nodes import TransformersReader

class AnswerGeneratorReader:
    def __init__(self, document_store, top_k: int = 3):
        
        retriever = BM25Retriever(document_store, top_k=top_k)
        reader = TransformersReader(model_name_or_path="savasy/bert-base-turkish-squad", use_gpu=False)
        
        # add query classifier?
        
        self.pipe = Pipeline()
        self.pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
        self.pipe.add_node(component=reader, name="Reader", inputs=["retriever"])

    def generate_answer(self, query: str) -> str:
        output = self.pipe.run(query=query)
        return output
    
    
        # Need to process the output.
        # This gave better output then gpt based QA
        

# Usage example
"""
document_store = ...
api_key = "your_api_key"
top_k = 3
answer_generator = AnswerGenerator(document_store=document_store, api_key=api_key, top_k=top_k)

query = "Gülyağının en büyük satın alıcısı hangi ülkelerdir?"
answer = answer_generator.generate_answer(query=query)
"""