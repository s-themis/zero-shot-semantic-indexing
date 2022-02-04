import lzma
import pathlib


def jsonl_len(json_file: str) -> None:
    """
    Return number of lines/objects in jsonl file.

    Args:
        json_file (str): file to be transformed

    Return: int
    """
    json_file = pathlib.Path(json_file)
    res = 0
    if json_file.suffix == ".xz":
        with lzma.open(json_file, "r") as f:
            for _ in f:
                res += 1
    else:
        with open(json_file, "r") as f:
            for _ in f:
                res += 1
    print(res)
