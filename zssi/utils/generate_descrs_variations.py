import pathlib

import jsonlines as jsonl


def generate_descrs_variations(descrs_jsonl: str) -> None:
    descrs_jsonl = pathlib.PurePath(descrs_jsonl)

    variations = {
        "name": [],
        "name_entry_terms": [],
        "name_scope_note": [],
        "name_entry_terms_scope_note": []
    }
    with jsonl.open(descrs_jsonl, "r") as f:
        for descr in f:

            if descr["name"]:
                d = dict(descr)
                d["label"] = d["name"]
                variations["name"].append(d)

            if descr["name"] and descr["all_entry_terms"]:
                d = dict(descr)
                d["label"] = d["name"] + " " + " ".join(d["all_entry_terms"])
                variations["name_entry_terms"].append(d)

            if descr["name"] and descr["scope_note"]:
                d = dict(descr)
                d["label"] = d["name"] + " " + d["scope_note"]
                variations["name_scope_note"].append(d)

            if descr["name"] and descr["all_entry_terms"] and descr["scope_note"]:
                d = dict(descr)
                d["label"] = d["name"] + " " + " ".join(d["all_entry_terms"]) + " " + d["scope_note"]
                variations["name_entry_terms_scope_note"].append(d)
    
    for variation, descriptors in variations.items():
        variation_jsonl = descrs_jsonl.with_name(descrs_jsonl.with_suffix("").name + f"_{variation}").with_suffix(".jsonl")
        with jsonl.open(variation_jsonl, "w") as f:
            f.write_all(descriptors)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--descrs_jsonl", type=str)
    args = parser.parse_args()

    generate_descrs_variations(args.descrs_jsonl)
