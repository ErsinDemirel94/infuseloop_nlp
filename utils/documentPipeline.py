import os
from typing import List, Tuple, Optional
from haystack.pipelines import Pipeline
from haystack.utils import convert_files_to_docs
from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor, FileTypeClassifier
from haystack.nodes.base import BaseComponent
from langdetect import detect
import re
from haystack.schema import Document

class LanguageDetector(BaseComponent):
    # Custom haystack node to detect and update meta.language of document objects.
    
    outgoing_edges = 1

    def run(self, documents: List[Document]) -> Tuple:
        for document in documents:
            detected_language = detect(document.content)
            document.meta["language"] = detected_language

        output = {
            "documents": documents,
        }
        return output, "output_1"

    def run_batch(self, documents: List[Document]) -> Tuple:
        return self.run(documents=documents)


class Postprocessor(BaseComponent):
    # Custom haystack node to clean text.
    
    outgoing_edges = 1

    def run(self, documents: List[Document]) -> Tuple:
        for document in documents:
            
            # Remove consecutive new lines
            content = re.sub(r'\n+', '\n', document.content)

            # Remove excessive spaces
            content = re.sub(r'\s+', ' ', content).strip()

            # Remove useless punctuation
            content = re.sub(r'([.!?]){2,}', r'\1', content)

            document.content = content

        output = {
            "documents": documents,
        }
        return output, "output_1"

    def run_batch(self, documents: List[Document]) -> Tuple:
        return self.run(documents=documents)

       

class DocumentPipeline:
    # Main class to process documents.
    def __init__(self, valid_languages, remove_numeric_tables):

        self.valid_languages = valid_languages # list of languages
        self.remove_numeric_tables = remove_numeric_tables # bool

    def process_txt(self, text_paths): 
        
        file_type_classifier = FileTypeClassifier()

        # converters
        text_converter = TextConverter(
          remove_numeric_tables = self.remove_numeric_tables,
          valid_languages = self.valid_languages
          )

        docx_converter = DocxToTextConverter(
          valid_languages = self.valid_languages
          )

        pdf_converter = PDFToTextConverter()

        # preprocessor
        preprocessor = PreProcessor(clean_empty_lines=True,
                                    clean_whitespace=True,
                                    clean_header_footer=True,
                                    split_by="word",
                                    split_length=100,
                                    split_respect_sentence_boundary=True,
                                    split_overlap=5)
        
        
        # language detector and postprocessor custom nodes
        
        lang_detector = LanguageDetector()
        postprocessor = Postprocessor()

        # init pipeline
        p = Pipeline()

        p.add_node(component=file_type_classifier, name="FileTypeClassifier", inputs=["File"])
        p.add_node(component=text_converter, name="TextConverter", inputs=["FileTypeClassifier.output_1"])
        p.add_node(component=pdf_converter, name="PdfConverter", inputs=["FileTypeClassifier.output_2"])
        p.add_node(component=docx_converter, name="DocxConverter", inputs=["FileTypeClassifier.output_4"])
        p.add_node(component= preprocessor,name="Preprocessor", inputs=["TextConverter","PdfConverter","DocxConverter"])
        p.add_node(component= postprocessor,name="Postprocessor", inputs=["Preprocessor"])
        p.add_node(component=lang_detector, name="lang_detector", inputs=["Postprocessor"])

        output = []
        for item in text_paths:

            docs = p.run(file_paths=item)
            output.append(docs)

        return output

    

# TO DO:

# Frontends will remove unsupported file types. Too large files etc.
# Exception handling. empty file input etc, language different than eng or tr?
# Handle scanned pdf files? (images) DO not process them here. Needs additional class.
# Do some tests with different length files.
