# QA-READER ve QA-GPT nin birleştirilmiş hali, hata verebilir.
# QGEN de buraya alınabilir?


from typing import List
from haystack.pipelines import Pipeline
from haystack.nodes import Shaper, PromptNode, PromptTemplate
from haystack.schema import Document
from haystack.nodes import BM25Retriever
from haystack.nodes import TransformersReader

class AnswerGeneratorUnified:
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
        prompt_node = PromptNode(model, default_prompt_template=lfqa_prompt, api_key=api_key)
        reader = TransformersReader(model_name_or_path="savasy/bert-base-turkish-squad", use_gpu=False)

        self.pipe_prompt = Pipeline()
        self.pipe_prompt.add_node(component=retriever, name="retriever", inputs=["Query"])
        self.pipe_prompt.add_node(component=shaper, name="shaper", inputs=["retriever"])
        self.pipe_prompt.add_node(component=prompt_node, name="prompt_node", inputs=["shaper"])

        self.pipe_reader = Pipeline()
        self.pipe_reader.add_node(component=retriever, name="retriever", inputs=["Query"])
        self.pipe_reader.add_node(component=reader, name="Reader", inputs=["retriever"])

    def generate_answer_prompt(self, query: str) -> str:
        output = self.pipe_prompt.run(query=query)
        return output

    def generate_answer_reader(self, query: str) -> str:
        output = self.pipe_reader.run(query=query)
        return output
