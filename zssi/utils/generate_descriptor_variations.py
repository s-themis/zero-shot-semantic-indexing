import pathlib

import jsonlines as jsonl


def generate_descriptor_variations(descriptors_file: str) -> None:
    name_variation = []
    name_entryterms_variation = []
    name_scopenote_variation = []
    name_entryterms_scopenote_variation = []
    dfile = pathlib.Path(descriptors_file)
    with jsonl.open(descriptors_file, "r") as f:
        for descriptor in f:

            # name
            if descriptor["name"]:
                d = dict(descriptor)
                d["label"] = d["name"]
                name_variation.append(d)

            # name_entryterms
            if descriptor["name"] and descriptor["entry_terms"]:
                d = dict(descriptor)
                d["label"] = d["name"] + " " + " ".join(d["entry_terms"])
                name_entryterms_variation.append(d)

            # name_scopenote
            if descriptor["name"] and descriptor["scope_note"]:
                d = dict(descriptor)
                d["label"] = d["name"] + " " + d["scope_note"]
                name_scopenote_variation.append(d)

            # name_entryterms_scopenote
            if descriptor["name"] and descriptor["entry_terms"] and descriptor["scope_note"]:
                d = dict(descriptor)
                d["label"] = d["name"] + " " + " ".join(d["entry_terms"]) + " " + d["scope_note"]
                name_entryterms_scopenote_variation.append(d)

    with jsonl.open(dfile.with_stem(dfile.stem + "_name"), "w") as f:
        f.write_all(name_variation)

    with jsonl.open(dfile.with_stem(dfile.stem + "_name_entryterms"), "w") as f:
        f.write_all(name_entryterms_variation)

    with jsonl.open(dfile.with_stem(dfile.stem + "_name_scopenote"), "w") as f:
        f.write_all(name_scopenote_variation)

    with jsonl.open(dfile.with_stem(dfile.stem + "_name_entryterms_scopenote"), "w") as f:
        f.write_all(name_entryterms_scopenote_variation)
