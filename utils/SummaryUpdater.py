from haystack.pipelines import Pipeline
from haystack.nodes import TransformersSummarizer
from typing import List

class DocumentSummarizer:
    
    def __init__(self, summarizer_model_english: str, summarizer_model_turkish: str):
        
        self.summarizer_model_english = summarizer_model_english
        self.summarizer_model_turkish = summarizer_model_turkish

    def update_documents(self, document):
        
        
        # MAX LENGTH is 784, SUM PER PAGE OLABİLİR?
        
        
        content = document["content"]
        language = document["language"]
        
        if language == "english":
            model_name = self.summarizer_model_english
        elif language == "turkish":
            model_name = self.summarizer_model_turkish        
        
        else:
                raise ValueError(f"Unsupported language: {language}")

        summarizer = Summarizer(model_name_or_path=model_name)
        summary = summarizer.summarize([document.text])[0] 
        
        return summary

        # Can optionally use openai or huggingface