import time

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

from zssi.datagen import JsonlDocGenerator
from zssi.write import JsonlEmbeddingWriter


class Embedder:
    def __init__(self, modelhub_path, pooling_mode):
        self.modelhub_path = modelhub_path
        self.pooling_mode = pooling_mode

        token_embedding_model = Transformer(modelhub_path)

        pooling_model = Pooling(word_embedding_dimension=token_embedding_model.
                                get_word_embedding_dimension(),
                                pooling_mode=pooling_mode)

        self.model = SentenceTransformer(
            modules=[token_embedding_model, pooling_model])

    def embed(self, datagen, batch_size, writer):
        doc_generator = datagen.generate(batch_size=batch_size)
        exhausted = False
        writer.start()
        start = time.time()
        while not exhausted:
            try:
                batch, batch_doc_ids = next(doc_generator)
                embeddings = self.model.encode(sentences=batch,
                                               batch_size=len(batch))
                writer.write(embeddings, batch_doc_ids)
            except IndexError as e:
                print("Caught error!")
            except StopIteration:
                exhausted = True
        writer.finish()
        stop = time.time()
        print(
            f'\n\nFinished embedding "{datagen.path_to_jsonl}" using "{self.modelhub_path}" with "{self.pooling_mode}" pooling in: {stop - start} seconds.\n\n'
        )
