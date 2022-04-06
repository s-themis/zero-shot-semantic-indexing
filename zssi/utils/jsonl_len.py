import lzma
import pathlib


def jsonl_len(jsonl: str) -> None:
    """
    Return number of lines/objects in jsonl file.

    Args:
        jsonl (str): file to be transformed

    Return: int
    """
    jsonl = pathlib.Path(jsonl)
    res = 0
    if jsonl.suffix == ".xz":
        with lzma.open(jsonl, "r") as f:
            for _ in f:
                res += 1
    else:
        with open(jsonl, "r") as f:
            for _ in f:
                res += 1
    print(f"This file has {res} lines!")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str)
    args = parser.parse_args()

    jsonl_len(args.jsonl)