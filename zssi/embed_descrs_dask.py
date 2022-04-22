from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

class DescrEmbedder:
    def __init__(self, modelhub_model: str, pooling_mode: str, do_lower_case: bool) -> None:
        token_embedding_model = Transformer(modelhub_model, do_lower_case=do_lower_case)
        pooling_model = Pooling(
            token_embedding_model.get_word_embedding_dimension(),
            pooling_mode)
        model = SentenceTransformer(
            modules=[token_embedding_model, pooling_model])
        self.model = model

    def add_embedding(self, descr: dict) -> dict:
        embedding = self.model.encode(sentences=[descr["label"]]).tolist()
        descr["embedding"] = embedding[0]
        return descr

if __name__ == "__main__":

    import argparse
    import json
    import dask.bag as db

    from dask.distributed import Client

    parser = argparse.ArgumentParser()
    parser.add_argument("--descrs_jsonl", type=str)
    parser.add_argument("--modelhub_model", type=str)
    parser.add_argument("--pooling_mode", type=str)
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--dest_jsonl", type=str)
    args = parser.parse_args()
    
    client = Client(n_workers=1)    # noqa: F841
    descriptor_embedder = DescrEmbedder(args.modelhub_model, args.pooling_mode, args.do_lower_case)
    db.read_text(args.descrs_jsonl).map(json.loads).map(descriptor_embedder.add_embedding).map(json.dumps).to_textfiles([args.dest_jsonl])
