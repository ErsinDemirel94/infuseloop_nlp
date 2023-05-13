from haystack.pipelines import Pipeline
from haystack.nodes import EntityExtractor
from typing import List


"""
from fuzzywuzzy import fuzz

def deduplicate(lst):
    "Remove duplicates from a list of dictionaries using fuzzy string matching"
    unique_dicts = []
    for d in lst:
        # Convert the dictionary values to strings
        values_str = ' '.join(str(v) for v in d.values())
        # Check if a similar dictionary is already in the list
        found = False
        for ud in unique_dicts:
            ud_values_str = ' '.join(str(v) for v in ud.values())
            if fuzz.token_set_ratio(values_str, ud_values_str) >= 90:
                found = True
                break
        # Add the dictionary to the list if it's unique
        if not found:
            unique_dicts.append(d)
    return unique_dicts
"""


class NamedEntityUpdater:
    def __init__(self, ner_model_english: str, ner_model_turkish: str):
        self.ner_model_english = ner_model_english
        self.ner_model_turkish = ner_model_turkish

    def update_documents(self, documents):
        
        for doc in documents:
            
            doc_id = doc["_id"]
            content = doc["_source"]["content"]
            language = doc["_source"]["language"]
            
            #document = document_store.get_document_by_id(index_name=index_name, doc_id=doc_id)

            if language == "eng":
                model_name = self.ner_model_english
                
            elif language == "tr":
                model_name = self.ner_model_turkish
                
            else:
                raise ValueError(f"Unsupported language: {language}")

            extractor = EntityExtractor(model_name_or_path=model_name)
            
            entities = extractor.extract(content)
            ## Postprocess: Filter entities by score, Deduplicate records.
            
            output = [x for x in entities if x["score"]>0.8]
            
            # dedup_out = deduplicate(output)
            
            return output

            

# TO DO:
# # connect to es, document stores.
# # İnitializationu çöz, 
# # Postprocess entities
# # Update records.

            
