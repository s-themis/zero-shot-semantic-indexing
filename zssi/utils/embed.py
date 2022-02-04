if __name__ == "__main__":

    import json

    import dask.bag as db
    import spacy
    from dask.distributed import Client
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer

    class DocParser:

        def __init__(self) -> None:
            self.nlp = spacy.load('en_core_web_sm')

        def add_parsed_text(self, doc: dict, segment: bool) -> dict:
            parsed_text = []
            if segment:
                parsed_text.append(doc['title'])
                parsed_text.extend(
                    [str(sent) for sent in self.nlp(doc['abstractText']).sents])
            else:
                parsed_text.append(doc['title'] + " " + doc['abstractText'])
            doc["parsed_text"] = parsed_text
            return doc

    class DocEmbedder:

        def __init__(self, modelhub_model: str, pooling_mode: str) -> None:
            token_embedding_model = Transformer(modelhub_model)
            pooling_model = Pooling(
                token_embedding_model.get_word_embedding_dimension(),
                pooling_mode)
            model = SentenceTransformer(
                modules=[token_embedding_model, pooling_model])
            self.model = model

        def add_embeddings(self, doc: dict) -> dict:
            embeddings = self.model.encode(sentences=doc["parsed_text"]).tolist()
            keys = list(range(len(embeddings)))
            doc["embeddings"] = dict(zip(keys, embeddings))
            return doc

    def to_output_doc(doc):
        output_doc = dict()
        output_doc["pmid"] = doc["pmid"]
        output_doc["embeddings"] = doc["embeddings"]
        return output_doc

    def embed(docs_file: str, destination_file: str, segment: bool, modelhub_model: str, pooling_mode: str):
        """
        Calculate embeddings from input file and save to output file.

        Args:
            docs_file (str): file or glob pattern for input docs
            destination_file (str): file or glob pattern for output
            modelhub_model (str): huggingface model name
            pooling_mode (str): s-bert pooling mode

        Return: None
        """
        client = Client(n_workers=1)    # noqa: F841
        doc_parser = DocParser()
        doc_embedder = DocEmbedder(modelhub_model, pooling_mode)
        db.read_text(docs_file).map(json.loads).map(lambda doc: doc_parser.add_parsed_text(doc, segment)).map(doc_embedder.add_embeddings).map(to_output_doc).map(json.dumps).to_textfiles(destination_file)

    docs_file = "data/experimental/test_docs_2006_limit_100.jsonl"
    destination_file = ["data/experimental/test_docs_2006_limit_100_embeddings.jsonl.xz"]
    segment = False
    modelhub_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    pooling_mode = "cls"
    embed(docs_file, destination_file, segment, modelhub_model, pooling_mode)
