import json
import pathlib

import jsonlines as jsonl


def json_to_jsonl(json_file: str, destination: str, data_field: str, limit: int):
    """
    Transform json file to jsonl file.

    Args:
        json_file (str): file to be transformed
        destination (str): directory or .jsonl file
        limit (int): number of objects to keep

    Return: None
    """
    json_file = pathlib.Path(json_file)
    destination = pathlib.Path(destination)
    if destination.is_dir():
        jsonl_file = destination.joinpath(json_file.with_suffix(".jsonl").name)
    elif destination.suffix == ".jsonl":
        jsonl_file = destination
    else:
        raise ValueError("Destination must be a directory or .jsonl file.")
    with open(json_file, "r") as f:
        data = json.load(f)
    documents = data[data_field]
    if limit:
        documents = documents[:limit]
    with jsonl.open(jsonl_file, "w") as f:
        f.write_all(documents)
