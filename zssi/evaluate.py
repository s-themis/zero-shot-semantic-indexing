from descriptors.test_2006_emerging_fine_grained_descriptors import descriptors

import glob
import jsonlines as jsonl
import pickle

# load docs and turn descriptors into set
path_to_jsonl = "data/test_2006_descriptors_only.jsonl"
docs_with_descriptors = []
with jsonl.open(path_to_jsonl) as f:
    for doc in f:
        doc["Descriptor_UIs"] = set(doc["Descriptor_UIs"])
        docs_with_descriptors.append(doc)

# load descriptors and turn PHex into set
for i in range(len(descriptors)):
    descriptors[i]["PHex_UI"] = set(descriptors[i]["PHex_UI"])

similarities_path = "data/2006/similarities/"

similarities_flavors = [
    "sentence_segmentation_name",
    "sentence_segmentation_name_entry_terms",
    "sentence_segmentation_name_scope_note",
    "sentence_segmentation_name_entry_terms_scope_note",
    "whole_text_name",
    "whole_text_name_entry_terms",
    "whole_text_name_scope_note",
    "whole_text_name_entry_terms_scope_note",
]

predictions = {
    "pmid": [],
    "UI": [],
    "y_true": [],
    "y_score_min": [],
    "y_score_mean": [],
    "y_score_max": [],
    "method": []
}

for similarities_flavor in similarities_flavors:

    sim_files = glob.glob(similarities_path + similarities_flavor + "/*.jsonl")

    doc_idx = 0

    for sim_file in sim_files:

        with jsonl.open(sim_file) as f:

            docs_with_similarities = []
            for doc in f:
                docs_with_similarities.append(doc)

            # Iterate through docs with similarities of the currently open file
            # and get corresponding doc with descriptors
            for doc_sim in docs_with_similarities:
                doc_descr = docs_with_descriptors[doc_idx]
                doc_idx += 1

                for descr in descriptors:

                    # check if the document is valid for the specific descriptor
                    if set.intersection(doc_descr["Descriptor_UIs"],
                                        descr["PHex_UI"]):

                        predictions["pmid"].append(doc_descr["pmid"])
                        predictions["UI"].append(descr["UI"])
                        predictions["y_true"] = descr["UI"] in doc_descr[
                            "Descriptor_UIs"]
                        predictions["y_score_min"].append(doc_sim[descr["UI"] +
                                                                  "_min"])
                        predictions["y_score_mean"].append(
                            doc_sim[descr["UI"] + "_mean"])
                        predictions["y_score_max"].append(doc_sim[descr["UI"] +
                                                                  "_max"])
                        predictions["method"].append(similarities_flavor)

with open("data/2006/predictions/scores.pkl", "wb") as f:
    pickle.dump(predictions, f)
