"""FastAPI backend: question + schema -> SQL (and run it against a database).

Hardening: request-size limits, optional API key (TEXT2SQL_API_KEY), simple
per-IP rate limiting (TEXT2SQL_RATE_LIMIT/min), and append-only request logging
(logs/requests.jsonl).
"""
import json
import os
import sqlite3
import tempfile
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from text2sql import pipeline
from text2sql import schema as schema_mod

DEFAULT_DB = os.environ.get("TEXT2SQL_DB", "sample.db")
# db_path from a request is restricted to this directory (no arbitrary file read).
_ALLOWED_DIR = os.path.realpath(
    os.environ.get("TEXT2SQL_DB_DIR") or os.path.dirname(os.path.abspath(DEFAULT_DB)) or "."
)
_API_KEY = os.environ.get("TEXT2SQL_API_KEY")           # if set, required on /generate
_RATE = int(os.environ.get("TEXT2SQL_RATE_LIMIT", "60"))  # requests/min/IP; 0 disables
_LOG_PATH = os.environ.get("TEXT2SQL_LOG", os.path.join("logs", "requests.jsonl"))
_MAX_QUESTION = 2000
_MAX_SCHEMA = 20000
# Dynamic few-shot: point at a Spider-style train.json ([{question, query}, ...])
# to retrieve semantically-similar examples per request instead of fixed ones.
_EXAMPLES_FILE = os.environ.get("TEXT2SQL_EXAMPLES")
_RETRIEVE_K = int(os.environ.get("TEXT2SQL_RETRIEVE_K", "5"))

app = FastAPI(title="Text-to-SQL Interface")
_hits = defaultdict(deque)
_hits_lock = threading.Lock()
_example_store = None
_store_lock = threading.Lock()
_store_tried = False


def _get_examples(question):
    """Retrieve (question, sql) few-shot examples, or None if not configured /
    unavailable. The store is built once (lazily) and cached."""
    global _example_store, _store_tried
    if not _EXAMPLES_FILE:
        return None
    if _example_store is None and not _store_tried:
        with _store_lock:
            if _example_store is None and not _store_tried:
                _store_tried = True
                try:
                    from text2sql import retrieval
                    _example_store = retrieval.build_from_train(_EXAMPLES_FILE)
                except Exception:
                    _example_store = None
    if _example_store is None:
        return None
    try:
        return _example_store.retrieve(question, _RETRIEVE_K)
    except Exception:
        return None


class GenerateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=_MAX_QUESTION)
    schema_sql: Optional[str] = Field(None, max_length=_MAX_SCHEMA)
    db_path: Optional[str] = None
    mode: str = "few_shot_retry"
    samples: int = Field(1, ge=1, le=10)


class GenerateResponse(BaseModel):
    sql: str
    mode: str
    retried: bool
    first_error: Optional[str] = None
    categories: list = []
    value_hints: Optional[list] = None
    prompt: Optional[str] = None
    retry_prompt: Optional[str] = None
    executed: bool = False
    structure_only: bool = False
    error: Optional[str] = None
    note: Optional[str] = None
    rows: Optional[list] = None
    columns: Optional[list] = None
    stats: Optional[dict] = None


def require_key(x_api_key: Optional[str] = Header(None)):
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def rate_limit(request: Request):
    if _RATE <= 0:
        return
    ip = request.client.host if request.client else "?"
    now = time.time()
    with _hits_lock:
        dq = _hits[ip]
        while dq and dq[0] < now - 60:
            dq.popleft()
        if len(dq) >= _RATE:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        dq.append(now)


def _log(entry):
    try:
        os.makedirs(os.path.dirname(_LOG_PATH) or ".", exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def _safe_db_path(path):
    try:
        rp = os.path.realpath(path)
        if os.path.commonpath([rp, _ALLOWED_DIR]) == _ALLOWED_DIR and os.path.exists(rp):
            return rp
    except (ValueError, OSError):
        pass
    return None


def _materialize(schema_sql):
    """Empty SQLite db from pasted CREATE TABLE statements (CREATE only)."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    ddl = ";\n".join(s for s in schema_sql.split(";") if s.strip().upper().startswith("CREATE"))
    conn = sqlite3.connect(path)
    try:
        conn.executescript(ddl)
        conn.commit()
    finally:
        conn.close()
    return path


@app.get("/")
def index():
    """The root is the JSON API; the web UI is the Streamlit app (port 8501)."""
    return {
        "service": "Text-to-SQL API",
        "ui": "run `streamlit run app.py` -> http://localhost:8501",
        "docs": "/docs",
        "endpoints": ["/health", "/schema", "/generate"],
    }


@app.get("/health")
def health():
    return {"ok": True, "model": os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b")}


@app.get("/schema")
def get_schema(db_path: Optional[str] = None):
    path = _safe_db_path(db_path or DEFAULT_DB)
    if not path:
        return {"db_path": db_path, "description": None, "error": "path not allowed or missing"}
    return {"db_path": path, "description": schema_mod.describe_db(path)}


@app.post("/generate", response_model=GenerateResponse,
          dependencies=[Depends(require_key), Depends(rate_limit)])
def generate(req: GenerateRequest, request: Request):
    structure_only, temp_db = False, None

    if req.schema_sql:
        schema = schema_mod.from_create_statements(req.schema_sql)
        temp_db = _materialize(req.schema_sql)
        exec_db = temp_db
        structure_only = True
    else:
        exec_db = _safe_db_path(req.db_path or DEFAULT_DB)
        if not exec_db:
            raise HTTPException(status_code=400,
                                detail=f"db_path must be inside {_ALLOWED_DIR} and exist")
        schema = schema_mod.from_db(exec_db)

    try:
        result = pipeline.generate_sql(
            req.question, schema, mode=req.mode, db_path=exec_db, samples=req.samples,
            examples=_get_examples(req.question))
        resp = GenerateResponse(**result, structure_only=structure_only)
        run = pipeline.run_query(exec_db, result["sql"])
        resp.executed, resp.error = run["ok"], run["error"]
        resp.rows, resp.columns = run["rows"], run["columns"]
        if structure_only:
            resp.note = ("Executed against an empty database built from the pasted "
                         "schema -- this validates tables/joins/columns, not values.")
        _log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "ip": request.client.host if request.client else None,
            "mode": req.mode, "question": req.question, "sql": resp.sql,
            "executed": resp.executed, "error": resp.error,
            "latency_ms": (result.get("stats") or {}).get("latency_ms"),
        })
        return resp
    finally:
        if temp_db and os.path.exists(temp_db):
            os.remove(temp_db)
