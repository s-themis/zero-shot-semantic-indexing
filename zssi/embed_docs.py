import spacy
import time

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

class DocParser:
    def __init__(self, doc_segmentation: bool) -> None:
        self.doc_segmentation = doc_segmentation
        self.nlp = spacy.load('en_core_web_sm')

    def add_parsed_text(self, doc: dict) -> dict:
        parsed_text = []
        if self.doc_segmentation:
            parsed_text.append(doc['title'])
            parsed_text.extend(
                [str(sent) for sent in self.nlp(doc['abstractText']).sents])
        else:
            parsed_text.append(doc['title'] + " " + doc['abstractText'])
        doc["parsed_text"] = parsed_text
        return doc

class DocEmbedder:
    def __init__(self, modelhub_model: str, pooling_mode: str, do_lower_case: bool) -> None:
        token_embedding_model = Transformer(modelhub_model, do_lower_case=do_lower_case)
        pooling_model = Pooling(
            token_embedding_model.get_word_embedding_dimension(),
            pooling_mode)
        model = SentenceTransformer(
            modules=[token_embedding_model, pooling_model])
        self.model = model
        
    def add_embeddings(self, doc: dict) -> dict:
        embeddings = self.model.encode(sentences=doc["parsed_text"]).tolist()
        doc["embeddings"] = embeddings
        return doc

class ProgressLogger:
    def __init__(self, logging_interval: int) -> None:
        self.count = 0
        self.logging_interval = logging_interval
        current_time = time.time()
        self.start_time = current_time
        self.last_print_time = current_time
    
    def log(self, doc: dict) -> dict:
        self._log()
        return doc
    
    def _log(self):
        self.count += 1
        current_time = time.time()
        if self.count % self.logging_interval == 0:
            total_time = current_time - self.start_time
            current_rate = self.logging_interval / (current_time - self.last_print_time)
            print(f"total progress: {self.count} | total time: {total_time:.2f} s | current rate {current_rate} 1/s")
            self.last_print_time = current_time

class DocFilter:
    def __init__(self, keys_to_keep) -> None:
        self.keys_to_keep = keys_to_keep
    
    def filter(self, doc: dict) -> dict:
        filtered_doc = dict()
        for key in self.keys_to_keep:
            filtered_doc[key] = doc[key]
        return filtered_doc

if __name__ == "__main__":

    import argparse
    import json
    import lzma

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_docs_jsonl", type=str)
    parser.add_argument("--modelhub_model", type=str)
    parser.add_argument("--pooling_mode", type=str)
    parser.add_argument("--doc_segmentation", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--dest_jsonl_xz", type=str)
    parser.add_argument("--logging_interval", type=int)
    args = parser.parse_args()
  
    doc_parser = DocParser(args.doc_segmentation)
    doc_embedder = DocEmbedder(args.modelhub_model, args.pooling_mode, args.do_lower_case)
    progress_logger = ProgressLogger(args.logging_interval)
    doc_filter = DocFilter(keys_to_keep=["pmid", "embeddings", "Descriptor_UIs", "newFGDescriptors"])
    
    with lzma.open(args.dest_jsonl_xz, "w") as f_out:
        with open(args.test_docs_jsonl) as f_in:
            for line in f_in:
                doc = json.loads(line)
                doc = doc_parser.add_parsed_text(doc)
                doc = doc_embedder.add_embeddings(doc)
                doc = doc_filter.filter(doc)
                doc = progress_logger.log(doc)
                f_out.write(json.dumps(doc))
