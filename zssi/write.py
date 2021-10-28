import jsonlines as jsonl

from collections import deque


class EmbeddingWriter:
    """Defines the interface that any subclass should implement.

    """
    def __init__(self):
        raise NotImplementedError

    def start(self):
        """Conducts any necessary initialization (e.g. open file) before writing can begin.

        """

    def write(self, embeddings, doc_ids):
        """Writes embeddings of correspoding documents.

        Args:
            embeddings (list(list(int))): The extracted embeddings.
            doc_ids (list(int)): The ids of documents that correspond to the embeddings.
        
        Returns:
            None
        """
        raise NotImplementedError

    def log_errors(self, doc_ids):
        """Log document ids for which an error occurred in embedding extraction in order to handle them.

        Returns:
            None
        """

    def finish(self):
        """Finish by writing any remaining data.

        Returns:
            None
        """
        raise NotImplementedError


class JsonlEmbeddingWriter(EmbeddingWriter):
    """A class to write embeddings of documents in jsonlines format."
    
    """
    def __init__(self, path_to_jsonl, n_objs_per_write):
        """Initializes the writer with a path to a jsonlines file.

        Args:
            path_to_jsonl (str): The path to a jsonlines file in which to write the embeddings.
        """
        self.path_to_jsonl = path_to_jsonl
        self.n_objs_per_write = n_objs_per_write

        self.received_embeddings = deque()
        self.received_doc_ids = deque()
        self.current_embeddings = []
        self.current_doc_id = None
        self.objs_to_write = []
        self.error_doc_ids = set()

    def _write(self):
        self.writer.write_all(self.objs_to_write)
        self.objs_to_write = []

    def start(self):
        self.f = open(self.path_to_jsonl, mode='w')
        self.writer = jsonl.Writer(self.f)

    def write(self, embeddings, doc_ids):

        self.received_embeddings.extend(embeddings)
        self.received_doc_ids.extend(doc_ids)

        while self.received_doc_ids:
            if self.current_doc_id is None:
                self.current_doc_id = self.received_doc_ids.popleft()
                self.current_embeddings.append(
                    self.received_embeddings.popleft().tolist())
            elif self.received_doc_ids[0] == self.current_doc_id:
                self.current_embeddings.append(
                    self.received_embeddings.popleft().tolist())
                self.received_doc_ids.popleft()
            else:
                if self.current_doc_id in self.error_doc_ids:
                    self.current_embeddings = None
                    self.error_doc_ids.remove(self.current_doc_id)

                self.objs_to_write.append({
                    "pmid": self.current_doc_id,
                    "embeddings": self.current_embeddings
                })

                self.current_doc_id = None
                self.current_embeddings = []
                if len(self.objs_to_write) == self.n_objs_per_write:
                    self._write()

    def log_errors(self, doc_ids):
        self.error_doc_ids.add(doc_ids)

    def finish(self):
        if self.current_doc_id is not None:
            if self.current_doc_id in self.error_doc_ids:
                self.current_embeddings = None
                self.error_doc_ids.remove(self.current_doc_id)

            self.objs_to_write.append({
                "pmid": self.current_doc_id,
                "embeddings": self.current_embeddings
            })
            self.current_doc_id = None
            self.current_embeddings = []

        if self.objs_to_write:
            self._write()

        self.writer.close()
        self.f.close()
