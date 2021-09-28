import json
import spacy

from sentence_transformers import models, SentenceTransformer

nlp = spacy.load('en_core_web_sm')

with open('data/test_2006.json') as f:
    data = json.load(f)

doc = nlp(data['documents'][0]['abstractText'])

token_embedding_model = models.Transformer('dmis-lab/biobert-base-cased-v1.2')

pooling_model = models.Pooling(word_embedding_dimension=token_embedding_model.
                               get_word_embedding_dimension(),
                               pooling_mode='cls')

model = SentenceTransformer(modules=[token_embedding_model, pooling_model])

embeddings = model.encode(list(doc.sents))
