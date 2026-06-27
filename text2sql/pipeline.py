"""Generate SQL, execute it, and repair it.

few_shot_retry does several bounded things beyond plain generation:
  1. value linking  -- ground WHERE literals in real cell values (linking.py)
  2. recall-safe schema linking -- if a pruned-away table is referenced, retry
                       with the full schema so big schemas don't lose tables
  3. repair         -- if the SQL errors / is invalid, feed the error back
                       (an empty result is NOT a failure -- many correct queries
                       return no rows, so retrying on empty would regress them)
  4. self-verify    -- if it's still broken, ask the model to repair once more
  5. voting         -- (optional, samples>1) sample several and pick the result
                       the majority agree on -- the lever against wrong-but-valid

The LLM call is injectable (`generate_fn`) so the loop is unit-testable without a
running model, and every call's latency/token counts are accumulated into stats.
"""
import pathlib
import re
import sqlite3
import time

import sqlglot
from sqlglot import exp

from . import linking, ollama_client
from .matching import canonicalize
from .prompts import (build_prompt, build_retry_prompt, build_verify_prompt,
                      detect_categories, _resolve_desc)

MODES = ("zero_shot", "few_shot", "few_shot_retry")
_MAX_TABLES = 10
_FETCH_CAP = 100_000      # cap runaway result materialization (e.g. cross joins)
_VOTE_TEMP = 0.6          # diversity when self-consistency sampling
_SQL_START = re.compile(r"\b(select|with|insert|update|delete)\b", re.IGNORECASE)


def extract_sql(text):
    """Pull a single SQL statement out of a model completion."""
    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start = _SQL_START.search(text)  # drop any "Sure, here is:" preamble
    if start:
        text = text[start.start():]
    if ";" in text:
        text = text.split(";")[0]
    return text.strip()


class _Aborter:
    """Abort a runaway model-generated query after a fixed number of VM steps."""

    def __init__(self, limit):
        self.limit = limit
        self.n = 0

    def __call__(self):
        self.n += 1
        return 1 if self.n > self.limit else 0  # non-zero -> SQLite aborts


def _connect(db_path):
    try:
        uri = pathlib.Path(db_path).resolve().as_uri() + "?mode=ro"  # read-only
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        conn = sqlite3.connect(db_path)
    conn.set_progress_handler(_Aborter(5_000_000), 1000)  # cap pathological queries
    return conn


def try_execute(db_path, sql):
    """Run sql read-only. Returns (ok, rows, error_message). Rows are capped."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(sql).fetchmany(_FETCH_CAP)
        return True, rows, None
    except Exception as exc:
        return False, None, str(exc)
    finally:
        conn.close()


def run_query(db_path, sql, limit=200):
    """Execute for display: returns {ok, rows, columns, error} with column names."""
    conn = _connect(db_path)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [list(r) for r in cur.fetchmany(limit)]
        return {"ok": True, "rows": rows, "columns": cols, "error": None}
    except Exception as exc:
        return {"ok": False, "rows": None, "columns": None, "error": str(exc)}
    finally:
        conn.close()


def _check(db_path, sql):
    """(ok, error, empty) -- execution if a db is available, else a syntax parse."""
    if db_path:
        ok, rows, err = try_execute(db_path, sql)
        return ok, err, (ok and not rows)
    try:
        sqlglot.parse_one(sql, dialect="sqlite")
        return True, None, False
    except Exception as exc:
        return False, str(exc), False


def _result_key(db_path, sql):
    """A hashable key for voting: the result set, 'EMPTY', or None (errored)."""
    if not db_path:
        return canonicalize(sql)
    ok, rows, _ = try_execute(db_path, sql)
    if not ok:
        return None
    if not rows:
        return "EMPTY"
    return tuple(sorted(tuple(str(c) for c in r) for r in rows))


def _referenced_tables(sql):
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
        return {t.name.lower() for t in tree.find_all(exp.Table)} if tree else set()
    except Exception:
        return set()


def _make_caller(stats, generate_fn, default_temp):
    """Return call(prompt, temperature=None) -> sql, accumulating latency/tokens."""
    def call(prompt, temperature=None):
        temp = default_temp if temperature is None else temperature
        t0 = time.perf_counter()
        out = generate_fn(prompt, temperature=temp)
        stats["llm_calls"] += 1
        stats["latency_ms"] += (time.perf_counter() - t0) * 1000.0
        stats["completion_tokens"] += out.get("eval_count") or 0
        stats["prompt_tokens"] += out.get("prompt_eval_count") or 0
        return extract_sql(out.get("response", ""))
    return call


def _generate(call, prompt, db_path, samples):
    if samples <= 1:
        return call(prompt)
    cands = [call(prompt, temperature=_VOTE_TEMP) for _ in range(samples)]
    buckets = {}
    for sql in cands:
        buckets.setdefault(_result_key(db_path, sql), []).append(sql)
    # Majority vote over result sets. EMPTY is a legitimate consensus; only
    # errored candidates (None) are excluded.
    valid = {k: v for k, v in buckets.items() if k is not None}
    if valid:
        return max(valid.values(), key=len)[0]
    return cands[0]


def _repair_and_verify(call, question, desc, sql, db_path, max_repairs, self_verify):
    retried, first_error, retry_prompt = False, None, None

    for _ in range(max_repairs):
        ok, err, _ = _check(db_path, sql)
        if ok:  # executes (empty or not) -> accept; empty is not a failure
            break
        retry_prompt = build_retry_prompt(question, desc, sql, err)
        sql = call(retry_prompt)
        retried = True
        first_error = first_error or err

    if self_verify:
        ok, err, _ = _check(db_path, sql)
        if not ok:  # last-resort rescue for still-broken queries
            revised = call(build_verify_prompt(question, desc, sql))
            if _check(db_path, revised)[0]:
                sql = revised
                retried = True
                first_error = first_error or err

    return sql, retried, first_error, retry_prompt


def generate_sql(question, schema, mode="few_shot_retry", db_path=None,
                 samples=1, temperature=0.0, max_repairs=2, self_verify=True,
                 examples=None, value_linking=True, generate_fn=None):
    """Generate SQL under one of MODES.

    `schema` may be a Schema (preferred) or a pre-rendered description string.
    `examples` (retrieved (question, sql) pairs) enables dynamic few-shot.
    `samples>1` enables self-consistency voting. `generate_fn` is injectable for
    tests and must return Ollama-style dicts ({"response", "eval_count", ...}).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    stats = {"llm_calls": 0, "latency_ms": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
    call = _make_caller(stats, generate_fn or ollama_client.generate_full, temperature)

    few_shot = mode != "zero_shot"
    structured = hasattr(schema, "tables")
    desc = _resolve_desc(schema, question, _MAX_TABLES)
    hints = (linking.value_hints(db_path, schema, question)
             if value_linking and db_path and structured else None)
    categories = detect_categories(question, schema) if few_shot else []

    prompt = build_prompt(question, schema, few_shot, _MAX_TABLES,
                          examples=examples, value_hints=hints, schema_desc=desc)
    sql = _generate(call, prompt, db_path, samples)

    # Recall-safe schema linking: if pruning dropped a table the model went on to
    # reference, regenerate once with the full schema.
    if few_shot and structured and len(schema.tables) > _MAX_TABLES:
        pruned = {t.name.lower() for t in schema._relevant_tables(question, _MAX_TABLES)}
        all_tables = {t.name.lower() for t in schema.tables}
        if _referenced_tables(sql) & (all_tables - pruned):
            desc = schema.describe(question=question, max_tables=10_000)
            prompt = build_prompt(question, schema, few_shot, 10_000,
                                  examples=examples, value_hints=hints, schema_desc=desc)
            sql = _generate(call, prompt, db_path, samples)

    retried, first_error, retry_prompt = False, None, None
    if mode == "few_shot_retry":
        sql, retried, first_error, retry_prompt = _repair_and_verify(
            call, question, desc, sql, db_path, max_repairs, self_verify)

    return {
        "sql": sql,
        "mode": mode,
        "retried": retried,
        "first_error": first_error,
        "categories": categories,
        "value_hints": hints,
        "prompt": prompt,
        "retry_prompt": retry_prompt,
        "stats": stats,
    }
