if __name__ == "__main__":

    import json
    from typing import List

    import dask.bag as db
    from dask.distributed import Client
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer

    class DescriptorEmbedder:

        def __init__(self, modelhub_model: str, pooling_mode: str) -> None:
            token_embedding_model = Transformer(modelhub_model)
            pooling_model = Pooling(
                token_embedding_model.get_word_embedding_dimension(),
                pooling_mode)
            model = SentenceTransformer(
                modules=[token_embedding_model, pooling_model])
            self.model = model

        def add_embedding(self, doc: dict) -> dict:
            embedding = self.model.encode(sentences=[doc["label"]]).tolist()
            doc["embedding"] = embedding[0]
            return doc

    def to_output_doc(doc):
        output_doc = dict()
        output_doc["name"] = doc["name"]
        output_doc["UI"] = doc["UI"]
        output_doc["PHex_UI"] = doc["PHex_UI"]
        output_doc["embedding"] = doc["embedding"]
        return output_doc

    def embed_descriptors(descriptors_file: str, destination_files: List[str], modelhub_model: str, pooling_mode: str):
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
        descriptor_embedder = DescriptorEmbedder(modelhub_model, pooling_mode)
        db.read_text(descriptors_file).map(json.loads).map(descriptor_embedder.add_embedding).map(to_output_doc).map(json.dumps).to_textfiles(destination_files)

    modelhub_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    pooling_mode = "cls"
    for descriptors_file, destination_files in zip(
        ["data/external-transformed/emerging_descriptors_2006_name.jsonl",
            "data/external-transformed/emerging_descriptors_2006_name_entryterms.jsonl",
            "data/external-transformed/emerging_descriptors_2006_name_scopenote.jsonl",
            "data/external-transformed/emerging_descriptors_2006_name_entryterms_scopenote.jsonl"],
        [["data/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_cls/embeddings/emerging_descriptors_2006_name.jsonl"],
            ["data/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_cls/embeddings/emerging_descriptors_2006_name_entryterms.jsonl"],
            ["data/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_cls/embeddings/emerging_descriptors_2006_name_scopenote.jsonl"],
            ["data/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_cls/embeddings/emerging_descriptors_2006_name_entryterms_scopenote.jsonl"]]):
        embed_descriptors(descriptors_file, destination_files, modelhub_model, pooling_mode)
