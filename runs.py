from zssi.datagen import JsonlDocGenerator
from zssi.embed import Embedder
from zssi.parse import SentenceSegmentationDocParser, WholeTextDocParser
from zssi.write import JsonlEmbeddingWriter, PickleEmbeddingWriter


def run_1():

    modelhub_model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    pooling_mode = 'cls'
    path_to_jsonl_data = 'data/test_2006_text_only.jsonl'
    path_to_jsonl_output = 'data/test_2006_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_cls_whole_text.jsonl'
    batch_size = 10

    doc_parser = WholeTextDocParser()
    datagen = JsonlDocGenerator(path_to_jsonl=path_to_jsonl_data,
                                doc_parser=doc_parser)
    embedder = Embedder(modelhub_path=modelhub_model,
                        pooling_mode=pooling_mode)
    writer = JsonlEmbeddingWriter(path_to_jsonl=path_to_jsonl_output,
                                  n_objs_per_write=1000)
    embedder.embed(datagen=datagen, batch_size=batch_size, writer=writer)


def run_2():

    modelhub_model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    pooling_mode = 'cls'
    path_to_jsonl_data = 'data/test_2006_text_only.jsonl'
    path_to_pkl_output = 'data/test_2006_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_cls_whole_text.pkl'
    batch_size = 10

    doc_parser = WholeTextDocParser()
    datagen = JsonlDocGenerator(path_to_jsonl=path_to_jsonl_data,
                                doc_parser=doc_parser)
    embedder = Embedder(modelhub_path=modelhub_model,
                        pooling_mode=pooling_mode)
    writer = PickleEmbeddingWriter(path_to_pkl=path_to_pkl_output,
                                   n_objs_per_file=50000)
    embedder.embed(datagen=datagen, batch_size=batch_size, writer=writer)


if __name__ == "__main__":

    run_2()
