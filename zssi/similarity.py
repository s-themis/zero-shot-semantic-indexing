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

    # Load all emerging-fine grained descriptors (embeddings) in memory
    descriptors = []
    descriptor_files = glob.glob(path_to_descr_embeddings_dir + "/*.jsonl")

    for file in descriptor_files:

        with jsonl.open(file) as f:

            for descr in f:
                # covnert embedding to np.array
                descr["embedding"] = np.array(descr["embedding"])
                descriptors.append(descr)

    # Load documents (embeddings) file by file and calculate similarities with every descriptor
    doc_files = glob.glob(path_to_doc_embeddigs_dir + "/*.pkl")

    for file in doc_files:

        with open(file, "rb") as f:
            # Load only one file at a time in memory
            documents = pickle.load(f)

            for doc in tqdm.tqdm(documents):
                # convert embeddings to np.array
                doc["embeddings"] = list(
                    map(lambda x: np.array(x), doc["embeddings"]))

                # calculate similarities of one document with every descriptor
                for descr in descriptors:
                    # similarities of each sentence
                    doc[descr["UI"]] = list(
                        map(lambda x: cosine_similarity(x, descr["embedding"]),
                            doc["embeddings"]))
                    # min similarity
                    doc[descr["UI"] + "_min"] = min(doc[descr["UI"]])
                    # mean similarity
                    doc[descr["UI"] + "_mean"] = sum(doc[descr["UI"]]) / len(
                        doc[descr["UI"]])
                    # max similarity
                    doc[descr["UI"] + "_max"] = max(doc[descr["UI"]])

                # delete embeddings to save only the similarities
                del doc["embeddings"]

        # save similarities
        file_no = file.split("/")[-1].split(".")[0]

        # with open(path_to_similarities_dir + f"/{file_no}.pkl", "wb") as f:
        #    pickle.dump(documents, f)

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
