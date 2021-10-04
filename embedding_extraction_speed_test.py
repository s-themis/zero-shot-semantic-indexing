import json
import spacy
import tqdm

from sentence_transformers import models, SentenceTransformer


class EmbeddingExtractionSpeedTest:
    def __init__(self):
        self.TRANSFORMER_MODELS = [
            'dmis-lab/biobert-base-cased-v1.2',
            'allenai/biomed_roberta_base',
            'sultan/BioM-ELECTRA-Base-Discriminator',
            'sultan/BioM-ELECTRA-Base-Generator',
            'sultan/BioM-ELECTRA-Large-Generator',
            'sultan/BioM-ALBERT-xxlarge',
            'sultan/BioM-ALBERT-xxlarge-PMC',
            'pretrained_models/BioMegatron_bert_345mUncased',
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


if __name__ == '__main__':

    speed_test = EmbeddingExtractionSpeedTest()
    speed_test.test_all_models(num_abstracts=100)
