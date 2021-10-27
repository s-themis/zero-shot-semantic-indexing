import json
import spacy
import time
import tqdm

import jsonlines as jsonl

from sentence_transformers import models, SentenceTransformer

from zssi.datagen import JsonlDocGenerator
from zssi.parse import SentenceSegmentationDocParser, WholeTextDocParser


class EmbeddingExtractionSpeed:
    def __init__(self):
        self.TRANSFORMER_MODELS = [
            'dmis-lab/biobert-base-cased-v1.2', 'allenai/biomed_roberta_base',
            'sultan/BioM-ELECTRA-Base-Discriminator',
            'sultan/BioM-ELECTRA-Base-Generator',
            'sultan/BioM-ELECTRA-Large-Generator',
            'sultan/BioM-ALBERT-xxlarge', 'sultan/BioM-ALBERT-xxlarge-PMC',
            'pretrained_models/BioMegatron_bert_345mUncased',
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        ]

        self.nlp = spacy.load('en_core_web_sm')

        with open('data/test_2006.json') as f:
            self.data = json.load(f)

    def test_model(self, modelhub_path, num_abstracts):
        token_embedding_model = models.Transformer(modelhub_path)

        pooling_model = models.Pooling(
            word_embedding_dimension=token_embedding_model.
            get_word_embedding_dimension(),
            pooling_mode='cls')

        model = SentenceTransformer(
            modules=[token_embedding_model, pooling_model])

        print(
            f'\n\n\nTesting {modelhub_path} on whole text of {num_abstracts} abstracts...'
        )
        for i in tqdm.tqdm(range(num_abstracts)):
            whole_text = self.data['documents'][i]['abstractText']
            try:
                embeddings = model.encode(sentences=[whole_text])
            except IndexError as e:
                print("Caught error!")

        print(
            f'\n\n\nTesting {modelhub_path} on all segmented sentenceses of {num_abstracts} abstracts...'
        )
        for i in tqdm.tqdm(range(num_abstracts)):
            doc = self.nlp(self.data['documents'][i]['abstractText'])
            sents = list(doc.sents)
            try:
                embeddings = model.encode(sentences=sents)
            except IndexError as e:
                print("Caught error!")

        print("\n\n\n")

    def test_all_models(self, num_abstracts):
        for modelhub_path in self.TRANSFORMER_MODELS:
            self.test_model(modelhub_path, num_abstracts)


class EmbeddingExtractionSpeedWithDatagen:
    def __init__(self, path_to_jsonl):
        self.TRANSFORMER_MODELS = [
            'dmis-lab/biobert-base-cased-v1.2', 'allenai/biomed_roberta_base',
            'sultan/BioM-ELECTRA-Base-Discriminator',
            'sultan/BioM-ELECTRA-Base-Generator',
            'sultan/BioM-ELECTRA-Large-Generator',
            'sultan/BioM-ALBERT-xxlarge', 'sultan/BioM-ALBERT-xxlarge-PMC',
            'pretrained_models/BioMegatron_bert_345mUncased',
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        ]

        self.whole_text_datagen = JsonlDocGenerator(
            path_to_jsonl=path_to_jsonl, doc_parser=WholeTextDocParser())

        self.segmented_sentences_datagen = JsonlDocGenerator(
            path_to_jsonl=path_to_jsonl,
            doc_parser=SentenceSegmentationDocParser())

        with jsonl.open(path_to_jsonl) as f:
            gen = iter(f)
            num_docs = 0
            exhausted = False
            while not exhausted:
                try:
                    doc = next(gen)
                    num_docs += 1
                except StopIteration:
                    exhausted = True
        self.num_docs = num_docs

    def test_model_on_whole_text(self, modelhub_path, batch_size):
        token_embedding_model = models.Transformer(modelhub_path)

        pooling_model = models.Pooling(
            word_embedding_dimension=token_embedding_model.
            get_word_embedding_dimension(),
            pooling_mode='cls')

        model = SentenceTransformer(
            modules=[token_embedding_model, pooling_model])

        print(
            f'\n\n\nTesting {modelhub_path} on concatenation of text and abstract for {self.num_docs} documents...'
        )
        doc_gen = self.whole_text_datagen.generate(batch_size)
        exhausted = False
        start = time.time()
        while not exhausted:
            try:
                whole_text_batch = next(doc_gen)
                embeddings = model.encode(sentences=whole_text_batch,
                                          batch_size=batch_size)
            except IndexError as e:
                print("Caught error!")
            except StopIteration:
                exhausted = True
        stop = time.time()
        print(f'Finished in: {stop - start} seconds.')

    def test_model_on_segmented_sentences(self, modelhub_path, batch_size):
        token_embedding_model = models.Transformer(modelhub_path)

        pooling_model = models.Pooling(
            word_embedding_dimension=token_embedding_model.
            get_word_embedding_dimension(),
            pooling_mode='cls')

        model = SentenceTransformer(
            modules=[token_embedding_model, pooling_model])

        print(
            f'\n\n\nTesting {modelhub_path} on title and segmented sentenceses for {self.num_docs} documents...'
        )
        doc_gen = self.segmented_sentences_datagen.generate(batch_size)
        exhausted = False
        start = time.time()
        while not exhausted:
            try:
                segmented_sentences_batch = next(doc_gen)
                embeddings = model.encode(sentences=segmented_sentences_batch,
                                          batch_size=batch_size)
            except IndexError as e:
                print("Caught error!")
            except StopIteration:
                exhausted = True
        stop = time.time()
        print(f'Finished in: {stop - start} seconds.')

        print("\n\n\n")

    def test_all_models_on_whole_text(self, batch_size):
        for modelhub_path in self.TRANSFORMER_MODELS:
            self.test_model_on_whole_text(modelhub_path, batch_size)

    def test_all_models_on_segmented_sentences(self, batch_size):
        for modelhub_path in self.TRANSFORMER_MODELS:
            self.test_model_on_segmented_sentences(modelhub_path, batch_size)


if __name__ == '__main__':

    speed_test = EmbeddingExtractionSpeed()
    speed_test.test_all_models(num_abstracts=100)

    speed_test_with_datagen = EmbeddingExtractionSpeedWithDatagen(
        'data/test_2006_text_only_100_docs.jsonl')
    speed_test_with_datagen.test_all_models_on_whole_text(batch_size=1)
    speed_test_with_datagen.test_all_models_on_segmented_sentences(
        batch_size=10)
