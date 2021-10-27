import jsonlines as jsonl

from collections import deque


class DocGenerator:
    """Defines the interface that any subclass should implement.

    """
    def __init__(self):
        raise NotImplementedError

    def generate(self, batch_size):
        """Generates examples from the document collection.

        Args:
            batch_size (int): The number of examples to generate for each batch.
        
        Yields:
            list(str): A single batch of examples.
        """
        raise NotImplementedError


class JsonlDocGenerator(DocGenerator):
    """A class to generate examples from a collection of documents in jsonlines format."
    
    """
    def __init__(self, path_to_jsonl, doc_parser):
        """Initializes the generator with a document collection and a parser class.

        Args:
            path_to_jsonl (str): The path to a collection of documents in jsonlines format.
            doc_parser (obj): An instance of a document parser class.
        """
        self.path_to_jsonl = path_to_jsonl
        self.doc_parser = doc_parser

    def generate(self, batch_size):
        with jsonl.open(self.path_to_jsonl) as f:
            gen = iter(f)
            parsed_docs = deque()
            batch = []
            exhausted = False
            while not exhausted:
                try:
                    doc = next(gen)
                    parsed_docs.extend(self.doc_parser.parse(doc))
                    while len(batch) < batch_size and parsed_docs:
                        batch.append(parsed_docs.popleft())
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
                except StopIteration:
                    exhausted = True
            if len(batch) > 0:
                yield batch
