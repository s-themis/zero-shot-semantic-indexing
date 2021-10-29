from zssi.datagen import JsonlDocGenerator
from zssi.embed import Embedder
from zssi.parse import SentenceSegmentationDocParser, WholeTextDocParser
from zssi.write import JsonlEmbeddingWriter

modelhub_models = [
    'dmis-lab/biobert-base-cased-v1.2',
    #'allenai/biomed_roberta_base',
    #'sultan/BioM-ELECTRA-Base-Discriminator',
    #'sultan/BioM-ELECTRA-Base-Generator',
    #'sultan/BioM-ELECTRA-Large-Generator',
    #'sultan/BioM-ALBERT-xxlarge',
    #'sultan/BioM-ALBERT-xxlarge-PMC',
    #'pretrained_models/BioMegatron_bert_345mUncased',
    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
]

pooling_mode = 'cls'

path_to_jsonl_data = 'data/test_2006_text_only_500_docs.jsonl'

#for model in modelhub_models:
#    for batch_size in [10, 15, 20, 25, 30, 40, 50]:
#        doc_parser = SentenceSegmentationDocParser()
#        datagen = JsonlDocGenerator(path_to_jsonl_data, doc_parser)
#        embedder = Embedder(model, pooling_mode)
#        writer = JsonlEmbeddingWriter(
#            path_to_jsonl_data.split(".")[0] + "_segmented_sentences_" +
#            model.replace("/", "--") + "_" + pooling_mode + ".jsonl", 500)
#        embedder.embed(datagen, batch_size, writer)

for model in modelhub_models:
    for batch_size in [5, 10, 15, 20, 25, 30, 40, 50]:
        doc_parser = WholeTextDocParser()
        datagen = JsonlDocGenerator(path_to_jsonl_data, doc_parser)
        embedder = Embedder(model, pooling_mode)
        writer = JsonlEmbeddingWriter(
            path_to_jsonl_data.split(".")[0] + "_whole_text_" +
            model.replace("/", "--") + "_" + pooling_mode + ".jsonl", 500)
        embedder.embed(datagen, batch_size, writer)
