# API_KEY: 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

from typing import List
from haystack.pipelines import Pipeline
from haystack.nodes import Shaper, PromptNode, PromptTemplate
from haystack.schema import Document
from haystack.nodes import BM25Retriever


## models from least to most powerful: text-ada-001, text-babbage-001, text-davinci-003

class AnswerGenerator:
    def __init__(self, model, document_store, api_key: str, top_k: int = 3):
        self.api_key = api_key
        lfqa_prompt = PromptTemplate(name="lfqa",
                                     prompt_text="""Synthesize a comprehensive answer from the following top_k most 
                                     relevant paragraphs and the given question. 
                                     Provide a clear and concise response that summarizes the key points and information presented in the paragraphs. 
                                     Your answer should be in your own words and be no longer than 20 words.
                                     Always finish the output with an end of a sentence.
                                     \n\n Paragraphs: $documents \n\n Question: $query \n\n Answer:""") 
        
        retriever = BM25Retriever(document_store, top_k=top_k)
        shaper = Shaper(func="join_documents", inputs={"documents": "documents"}, outputs=["documents"])
        node = PromptNode(model, default_prompt_template=lfqa_prompt, api_key=api_key)
        
        # # add query classifier?
        self.pipe = Pipeline()
        self.pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
        self.pipe.add_node(component=shaper, name="shaper", inputs=["retriever"])
        self.pipe.add_node(component=node, name="prompt_node", inputs=["shaper"])

    def generate_answer(self, query: str) -> str:
        output = self.pipe.run(query=query)
        return output

# Usage example
"""
document_store = ...
api_key = "your_api_key"
top_k = 3
answer_generator = AnswerGenerator(document_store=document_store, api_key=api_key, top_k=top_k)

query = "Gülyağının en büyük satın alıcısı hangi ülkelerdir?"
answer = answer_generator.generate_answer(query=query)
"""