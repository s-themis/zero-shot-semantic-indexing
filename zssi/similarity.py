
import json
import numpy as np
import tqdm

from more_itertools import chunked

def load_descriptors(descriptors_embedding_file):
    descriptors = []
    with open(descriptors_embedding_file) as f:
        for line in f:
            descriptors.append(json.loads(line))
    for descriptor in descriptors:
        descriptor["embedding"] = np.array(descriptor["embedding"])
        descriptor["PHex UIs"] = set(descriptor["PHex UIs"])
    return descriptors

def preprocess_doc(doc):
    doc["embeddings"] = list(map(np.array, doc["embeddings"]))
    doc["Descriptor_UIs_set"] = set(doc["Descriptor_UIs"])
    doc["newFGDescriptors_set"] = set(doc["newFGDescriptors"])
    return doc

def cos_sim(embedding_1: np.array, embedding_2: np.array) -> float:
    return np.inner(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))

def add_true_label_and_similarity_for_valid_descriptors(doc, descriptors, sim_func, valid_docs_only):
    doc["similarities"] = []
    doc["true"] = []
    for descr in descriptors:
        if valid_docs_only:
            if descr["PHex UIs"].intersection(doc["Descriptor_UIs_set"]):
                doc["similarities"].append(list(map(lambda doc_embedding: sim_func(doc_embedding, descr["embedding"]), doc["embeddings"])))
                doc["true"].append(int(descr["Descr. UI"] in doc["Descriptor_UIs_set"]))
            else:
                doc["similarities"].append([0] * len(doc["embeddings"]))
                doc["true"].append(0)
        else:
            doc["similarities"].append(list(map(lambda doc_embedding: sim_func(doc_embedding, descr["embedding"]), doc["embeddings"])))
            doc["true"].append(int(descr["Descr. UI"] in doc["Descriptor_UIs_set"]))
    doc["max_similarities"] = list(map(max, doc["similarities"]))
    doc["avg_similarities"] = list(map(lambda x: sum(x) / len(x), doc["similarities"]))
    doc["min_similarities"] = list(map(min, doc["similarities"]))
    del doc["embeddings"]
    del doc["Descriptor_UIs_set"]
    del doc["newFGDescriptors_set"]
    return doc

def calculate_similarities(docs_embeddings_jsonl, descrs_embeddings_jsonl, dest_jsonl, valid_docs_only):
    descriptors = load_descriptors(descrs_embeddings_jsonl)
    with open(dest_jsonl, "w") as f_out:
        with open(docs_embeddings_jsonl, "r") as f_in:
            with tqdm.tqdm(total=500000) as pr_bar:
                for lines_in in chunked(f_in, 1000):
                    lines_out = []
                    for line_in in lines_in:
                        doc = json.loads(line_in)
                        doc = preprocess_doc(doc)
                        doc = add_true_label_and_similarity_for_valid_descriptors(doc, descriptors, cos_sim, valid_docs_only)
                        line_out = json.dumps(doc) + "\n"
                        lines_out.append(line_out)
                    f_out.writelines(lines_out)
                    pr_bar.update(1000)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_embeddings_jsonl", type=str)
    parser.add_argument("--descrs_embeddings_jsonl", type=str)
    parser.add_argument("--dest_jsonl", type=str)
    parser.add_argument("--valid_docs_only", action="store_true")
    args = parser.parse_args()

    calculate_similarities(
        args.docs_embeddings_jsonl,
        args.descrs_embeddings_jsonl,
        args.dest_jsonl,
        args.valid_docs_only)
