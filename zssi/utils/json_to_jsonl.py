import json

import jsonlines as jsonl


def json_to_jsonl(input_json: str, output_jsonl: str, objects_field: str, limit: int):
    with open(input_json, "r") as f:
        data = json.load(f)
    objects = data[objects_field]
    if limit:
        objects = objects[:limit]
    with jsonl.open(output_jsonl, "w") as f:
        f.write_all(objects)

if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str)
    parser.add_argument("--output_jsonl", type=str)
    parser.add_argument("--objects_field", type=str)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    json_to_jsonl(args.input_json, args.output_jsonl, args.objects_field, args.limit)
