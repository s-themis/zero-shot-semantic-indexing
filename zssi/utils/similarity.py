if __name__ == "__main__":    # noqa: C901

    import json
    from typing import Callable

    import dask.bag as db
    import numpy as np
    from dask.distributed import Client

    def load_descriptors(descriptors_embedding_file):
        descriptors = []
        with open(descriptors_embedding_file) as f:
            for line in f:
                descriptors.append(json.loads(line))
        for descriptor in descriptors:
            descriptor["embedding"] = np.array(descriptor["embeddings"])
            descriptor["PHex_UI"] = set(descriptor["PHex_UI"])
        return descriptors

    def preprocess_doc(doc):
        doc["embeddings"] = list(map(np.array, doc["embeddings"]))
        doc["Descriptor_UIs"] = set(doc["Descriptor_UIs"])
        doc["newFGDescriptors"] = set(doc["newFGDescriptors"])
        return doc

    def cosine_similarity(embedding_1: np.array, embedding_2: np.array) -> float:
        return np.inner(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))

    def add_true_label_and_similarity_for_valid_descriptors(doc, descriptors, similarity_func):
        for descriptor in descriptors:
            if descriptor["PHex_UI"].intersection(doc["Descriptor_UIs"]):
                descriptor_UI = descriptor["UI"]
                doc[f"{descriptor_UI}_true"] = descriptor_UI in doc["newFGDescriptors"]
                doc[f"{descriptor_UI}_sims"] = list(map(lambda doc_embedding: similarity_func(doc_embedding, descriptor["embedding"]), doc["embeddings"]))
                doc[f"{descriptor_UI}_sim_min"] = min(doc[f"{descriptor_UI}_sims"])
                doc[f"{descriptor_UI}_sim_mean"] = sum(doc[f"{descriptor_UI}_sims"]) / len(doc[f"{descriptor_UI}_sims"])
                doc[f"{descriptor_UI}_sim_max"] = max(doc[f"{descriptor_UI}_sims"])
        return doc

    def to_output_doc(doc):
        del doc["Descriptor_UIs"]
        del doc["newFGDescriptors"]
        del doc["embeddings"]
        return doc

    def similarity(docs_embeddings_file: str, descriptors_embedding_file: str, destination_file: str, similarity_func: Callable):
        """
        Calculate the similarities between document and descriptor embeddings.

        Args:
            docs_embeddings_file (str): file or glob pattern for embeddings of docs
            descriptors_embeddings_file (str): file or glob pattern for embeddings of descriptors
            destination_file (str): file or glob pattern for output
            similarity_func (callable): function that calculates the similarity between two embeddings
        """
        client = Client(n_workers=1)    # noqa: F841
        descriptors = load_descriptors(descriptors_embedding_file)
        db.read_text(docs_embeddings_file).map(json.loads).map(preprocess_doc).map(lambda doc: add_true_label_and_similarity_for_valid_descriptors(doc, descriptors, similarity_func)).map(to_output_doc).map(json.dumps).to_textfiles(destination_file)

    docs_embeddings_file = "data/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_cls/embeddings/test_docs_2006_wholetext.jsonl.xz"
    descriptors_embedding_file = "data/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_cls/embeddings/descriptors_2006_name.jsonl"
    destination_file = ["data/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_cls/similarities/wholetext_name_2006.jsonl.xz"]
    similarity(docs_embeddings_file, descriptors_embedding_file, destination_file, cosine_similarity)
