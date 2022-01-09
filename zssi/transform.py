import json

import jsonlines as jsonl


def json_to_jsonl(path_to_json, path_to_jsonl, fields=None):

    with open(path_to_json, "r") as f:
        data = json.load(f)

    for i in range(len(data["documents"])):
        doc = data["documents"][i]
        filtered_doc = {}
        for field in fields:
            filtered_doc[field] = doc[field]
        data["documents"][i] = filtered_doc

    with jsonl.open(path_to_jsonl, "w") as f:
        f.write_all(data["documents"])
