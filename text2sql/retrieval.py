"""Dynamic few-shot retrieval.

Embed the question (via Ollama's nomic-embed-text) and fetch the most similar
(question -> gold SQL) pairs from Spider's train split. Retrieved examples
replace the fixed toy examples -- the standard way to beat static few-shot.

Embeddings for the train set are computed once and cached next to the train file.
"""
import json
import os

from . import ollama_client


def _load_train(train_file, limit):
    with open(train_file, encoding="utf-8") as f:
        data = json.load(f)
    items = [(d["question"], d["query"]) for d in data if d.get("question") and d.get("query")]
    return items[:limit] if limit else items


class ExampleStore:
    def __init__(self, items, vectors):
        self.items = items          # list of (question, sql)
        self.vectors = vectors      # np.ndarray (n, d), L2-normalized

    def retrieve(self, question, k=5):
        import numpy as np

        v = np.asarray(ollama_client.embed(question), dtype="float32")
        v /= (np.linalg.norm(v) or 1.0)
        sims = self.vectors @ v
        return [self.items[i] for i in np.argsort(-sims)[:k]]


def build_from_train(train_file, cache_path=None, limit=None):
    """Build (and cache) an ExampleStore from a Spider train.json."""
    import numpy as np

    items = _load_train(train_file, limit)
    cache_path = cache_path or f"{train_file}.emb{limit or 'all'}.npz"
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return ExampleStore([tuple(x) for x in data["items"]], data["vectors"])

    vecs = []
    for i, (q, _) in enumerate(items):
        e = np.asarray(ollama_client.embed(q), dtype="float32")
        vecs.append(e / (np.linalg.norm(e) or 1.0))
        if (i + 1) % 200 == 0:
            print(f"  embedded {i + 1}/{len(items)} train questions")
    vectors = np.vstack(vecs).astype("float32")
    np.savez(cache_path, items=np.array(items, dtype=object), vectors=vectors)
    return ExampleStore(items, vectors)
