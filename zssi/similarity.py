import glob
import pickle
import tqdm
import jsonlines as jsonl
import numpy as np


def cosine_similarity(embedding_a, embedding_b):
    return np.inner(embedding_a, embedding_b) / (np.linalg.norm(embedding_a) *
                                                 np.linalg.norm(embedding_b))


def calculate_similarities(path_to_doc_embeddigs_dir,
                           path_to_descr_embeddings_dir,
                           path_to_similarities_dir):

    #doc_parsing_type = path_to_doc_embeddigs_dir.split("/")[-1]
    #descriptors_flavor = path_to_descr_embeddings_dir.split("/")[-1]

    descriptors = []
    descriptor_files = glob.glob(path_to_descr_embeddings_dir + "/*.jsonl")
    for file in descriptor_files:
        with jsonl.open(file) as f:
            for descr in f:
                descr["embedding"] = np.array(descr["embedding"])
                descriptors.append(descr)

    doc_files = glob.glob(path_to_doc_embeddigs_dir + "/*.pkl")
    for file in doc_files:
        with open(file, "rb") as f:
            documents = pickle.load(f)
            for doc in tqdm.tqdm(documents):
                doc["embeddings"] = list(
                    map(lambda x: np.array(x), doc["embeddings"]))
                for descr in descriptors:
                    doc[descr["UI"]] = list(
                        map(lambda x: cosine_similarity(x, descr["embedding"]),
                            doc["embeddings"]))
                del doc["embeddings"]
        file_no = file.split("/")[-1].split(".")[0]
        with open(path_to_similarities_dir + f"/{file_no}.pkl", "wb") as f:
            pickle.dump(documents, f)
        with jsonl.open(path_to_similarities_dir + f"/{file_no}.jsonl",
                        "w") as f:
            f.write_all(documents)


if __name__ == "__main__":

    path_to_doc_embeddings_dir = "data/2006/embeddings/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_cls/docs/sentence_segmentation"
    path_to_descr_embeddings_dir = "data/2006/embeddings/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_cls/descriptors/name"
    path_to_similarities_dir = "data/2006/similarities/sentence_segmentation_name"
    calculate_similarities(path_to_doc_embeddings_dir,
                           path_to_descr_embeddings_dir,
                           path_to_similarities_dir)
