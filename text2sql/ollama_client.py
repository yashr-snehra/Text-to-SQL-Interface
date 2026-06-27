"""Thin client for a locally running Ollama model."""
import functools
import os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
# qwen2.5-coder:14b is SQL/code-tuned and fits an Arc 140V (32GB) at Q4 (~9GB);
# it beats general 8B models on multi-hop joins. Set OLLAMA_MODEL=qwen2.5-coder:7b
# (or llama3.1) for lower latency.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Keep the model resident between calls so we don't pay model-load latency on
# every request -- a big win when the evaluator fires hundreds of generations.
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
# 8192 avoids silently truncating big schemas + few-shot + value hints past the
# context window (see schema._relevant_tables); 32GB RAM absorbs the KV cache.
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

# Stop as soon as the model drifts past the SQL into a new turn or prose block.
# NOTE: do not stop on ``` -- SQL/code-tuned models (qwen2.5-coder) often lead
# with a ```sql fence, which would halt generation immediately and yield an empty
# completion. extract_sql() strips fences from the full text instead.
_STOP = ["\nQuestion:", "\nQ:", "\n\n\n"]
_EMBED_URL = OLLAMA_URL.replace("/api/generate", "/api/embeddings")


def generate_full(prompt, *, temperature=0.0, num_predict=256, timeout=180):
    """Call Ollama and return the full JSON (response text + token/timing stats)."""
    import requests  # lazy: keeps schema/prompt/exec usable without the dep installed

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": NUM_CTX,
                "stop": _STOP,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def generate(prompt, **kwargs):
    """Return just the completion text. temperature=0 keeps eval reproducible."""
    return generate_full(prompt, **kwargs)["response"]


@functools.lru_cache(maxsize=2048)
def embed(text, *, timeout=60):
    """Return an embedding vector for `text` (dynamic few-shot + schema linking).

    Cached: the same text (table descriptions, repeated questions) embeds to the
    same vector, so we don't re-hit Ollama on every request. Callers treat the
    result as read-only.
    """
    import requests

    resp = requests.post(
        _EMBED_URL,
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]
