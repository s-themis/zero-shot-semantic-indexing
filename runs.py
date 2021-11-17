import glob

from zssi.datagen import JsonlDocGenerator
from zssi.descriptor import DescriptorAugmenter
from zssi.embed import Embedder
from zssi.parse import SentenceSegmentationDocParser, WholeTextDocParser, AugmentedDescriptorParser
from zssi.write import JsonlEmbeddingWriter, PickleEmbeddingWriter, JsonlLabelEmbeddingWriter

from descriptors.test_2006_emerging_fine_grained_descriptors import descriptors


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


def run_3():

    modelhub_model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    pooling_mode = 'cls'
    path_to_jsonl_data = 'data/test_2006_text_only.jsonl'
    path_to_pkl_output = 'data/test_2006_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_cls_sentence_segmentation.pkl'
    batch_size = 10

    doc_parser = SentenceSegmentationDocParser()
    datagen = JsonlDocGenerator(path_to_jsonl=path_to_jsonl_data,
                                doc_parser=doc_parser)
    embedder = Embedder(modelhub_path=modelhub_model,
                        pooling_mode=pooling_mode)
    writer = PickleEmbeddingWriter(path_to_pkl=path_to_pkl_output,
                                   n_objs_per_file=20000)
    embedder.embed(datagen=datagen, batch_size=batch_size, writer=writer)


def run_4():

    daug = DescriptorAugmenter(descriptor_objs=descriptors,
                               output_name="test_2006_emerging_fine_grained")
    daug.save_all_flavors()


def run_5():

    modelhub_model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    pooling_mode = 'cls'
    batch_size = 10

    for path_to_jsonl_data in glob.glob("augmented_descriptors/*.jsonl"):
        temp = path_to_jsonl_data.split("/")[-1].split(".")[0]
        path_to_jsonl_output = 'test_data/' + temp + "_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_cls.jsonl"

        doc_parser = AugmentedDescriptorParser()
        datagen = JsonlDocGenerator(path_to_jsonl=path_to_jsonl_data,
                                    doc_parser=doc_parser)
        embedder = Embedder(modelhub_path=modelhub_model,
                            pooling_mode=pooling_mode)
        writer = JsonlLabelEmbeddingWriter(path_to_jsonl=path_to_jsonl_output)
        embedder.embed(datagen=datagen, batch_size=batch_size, writer=writer)


if __name__ == "__main__":

    run_5()
