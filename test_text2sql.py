"""Self-check for the deterministic parts (no Ollama needed).

Run: python test_text2sql.py
Covers schema parsing/sampling/pruning/plurals, Spider FK loading, schema-aware
detection, component (incl. self-join) + execution matching, value linking,
hardness, the retry/self-verify/voting loop (via an injected fake LLM), the eval
aggregator, and the API safety helpers.
"""
import os
import sqlite3
import tempfile

import api
import evaluate
from text2sql import linking, matching, pipeline, prompts, spider
from text2sql import schema as schema_mod
from text2sql.hardness import hardness
from text2sql.schema import Column, Schema, Table, _loose_match

DDL = """
CREATE TABLE department (id INTEGER PRIMARY KEY, name TEXT, budget INTEGER);
CREATE TABLE student (id INTEGER PRIMARY KEY, name TEXT, age INTEGER,
                      dept_id INTEGER REFERENCES department(id));
"""


def _temp_db(ddl, rows=None):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.executescript(ddl)
    for sql, params in (rows or []):
        conn.executemany(sql, params)
    conn.commit()
    conn.close()
    return path


class FakeLLM:
    """Injectable generate_fn: maps (prompt, call_index) -> SQL, Ollama-shaped."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, prompt, temperature=0.0):
        sql = self.fn(prompt, self.calls)
        self.calls += 1
        return {"response": sql, "eval_count": 5, "prompt_eval_count": 10}


# ---- schema / parsing ----

def test_schema_fk_and_description():
    s = schema_mod.from_create_statements(DDL)
    student = next(t for t in s.tables if t.name == "student")
    assert ("dept_id", "department", "id") in student.fks
    assert "Column dept_id references department.id" in s.describe()


def test_value_sampling_skips_pii():
    path = _temp_db(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, country TEXT);",
        [("INSERT INTO users VALUES (?,?,?,?)",
          [(1, "Alice", "a@x.com", "UK"), (2, "Bob", "b@x.com", "US")])],
    )
    try:
        cols = {c.name: c for c in schema_mod.from_db(path).tables[0].columns}
        assert cols["name"].samples and cols["country"].samples
        assert cols["email"].samples == []
    finally:
        os.remove(path)


def test_schema_pruning_keeps_relevant_plus_fk_neighbour():
    tables = [Table(f"tbl{i}", [Column(f"c{i}", "INT")]) for i in range(12)]
    tables[5].columns.append(Column("movie", "TEXT"))
    tables[5].fks.append(("c5", "tbl6", "c6"))
    out = Schema(tables).describe(question="list the movies", max_tables=3)
    assert "tbl5" in out and "tbl6" in out and "tbl0" not in out


def test_plural_matching():
    assert _loose_match("track", {"tracks"})
    assert _loose_match("category", {"categories"})
    assert _loose_match("person", {"people"})
    assert not _loose_match("track", {"album"})


def test_spider_schema_from_entry():
    entry = {
        "db_id": "t",
        "table_names_original": ["singer", "concert"],
        "column_names_original": [[-1, "*"], [0, "Singer_ID"], [0, "Name"],
                                  [1, "Concert_ID"], [1, "Singer_ID"]],
        "column_types": ["text", "number", "text", "number", "number"],
        "primary_keys": [1, 3],
        "foreign_keys": [[4, 1]],
    }
    s = spider.schema_from_entry(entry)
    concert = next(t for t in s.tables if t.name == "concert")
    assert ("Singer_ID", "singer", "Singer_ID") in concert.fks


def test_detect_categories_schema_aware():
    s = schema_mod.from_create_statements(DDL)
    assert "join" in prompts.detect_categories("students in the department named physics", s)
    assert prompts.detect_categories("How many students are there?") == ["aggregation"]


# ---- matching ----

def test_component_match_alias_and_order_insensitive():
    gold = "SELECT T1.name FROM student AS T1 JOIN department AS T2 ON T1.dept_id = T2.id"
    pred = "select x.name from student as x join department as y on x.dept_id = y.id"
    assert matching.component_match(pred, gold)
    assert matching.component_match("SELECT a, b FROM t", "SELECT b, a FROM t")
    assert matching.component_match("SELECT a FROM t WHERE x=1 AND y=2",
                                    "SELECT a FROM t WHERE y=2 AND x=1")
    assert not matching.component_match("SELECT a FROM t", "SELECT c FROM t")


def test_component_match_self_join():
    gold = ("SELECT a.name FROM employee AS a JOIN employee AS b "
            "ON a.mgr_id = b.id WHERE b.name = 'X'")
    pred = ("select x.name from employee as x join employee as y "
            "on x.mgr_id = y.id where y.name = 'X'")
    assert matching.component_match(pred, gold)               # self-join, renamed aliases
    swapped = ("SELECT b.name FROM employee AS a JOIN employee AS b "
               "ON a.mgr_id = b.id WHERE a.name = 'X'")
    assert not matching.component_match(swapped, gold)        # genuinely different query


def test_exec_match_modes():
    assert matching.is_ordered("SELECT a FROM t ORDER BY a")
    assert matching.exec_match([(1,), (2,)], [(2,), (1,)], ordered=False) is True
    assert matching.exec_match([(1,), (2,)], [(2,), (1,)], ordered=True) is False
    assert matching.exec_match([(2.0,)], [(2,)], ordered=False) is True
    assert matching.exec_match(None, [(1,)], ordered=False) is None
    assert matching.exec_match([(1,)], None, ordered=False) is False


def test_hardness():
    assert hardness("SELECT name FROM t") == "easy"
    assert hardness("SELECT a FROM t WHERE b > (SELECT max(b) FROM t)") in ("hard", "extra")
    assert hardness("SELECT count(*) FROM a JOIN b ON a.id=b.a GROUP BY a.x") != "easy"


# ---- value linking ----

def test_value_hints():
    path = _temp_db(
        "CREATE TABLE artist (id INTEGER, name TEXT); CREATE TABLE track (id INTEGER, title TEXT);",
        [("INSERT INTO artist VALUES (?,?)", [(1, "Radiohead")])],
    )
    try:
        s = schema_mod.from_db(path)
        hints = linking.value_hints(path, s, "list tracks by Radiohead")
        assert any("Radiohead" in h and "artist.name" in h for h in hints)
    finally:
        os.remove(path)


# ---- pipeline loop (fake LLM) ----

def test_retry_repairs_bad_sql():
    path = _temp_db(DDL, [("INSERT INTO student VALUES (?,?,?,?)", [(1, "Alice", 22, 1)])])
    try:
        s = schema_mod.from_db(path)
        llm = FakeLLM(lambda p, i: "SELECT name FROM student" if "Problem:" in p
                      else "SELECT FROMM student")  # first errors, retry fixes
        res = pipeline.generate_sql("names", s, mode="few_shot_retry", db_path=path,
                                    generate_fn=llm, value_linking=False)
        assert res["retried"] and res["sql"] == "SELECT name FROM student"
        assert res["stats"]["llm_calls"] == 2
    finally:
        os.remove(path)


def test_self_verify_rescue():
    path = _temp_db(DDL)
    try:
        s = schema_mod.from_db(path)
        llm = FakeLLM(lambda p, i: "SELECT id FROM student" if "candidate" in p.lower()
                      else "SELECT FROMM student")
        res = pipeline.generate_sql("ids", s, mode="few_shot_retry", db_path=path,
                                    max_repairs=0, self_verify=True,
                                    generate_fn=llm, value_linking=False)
        assert res["sql"] == "SELECT id FROM student" and res["retried"]
    finally:
        os.remove(path)


def test_self_consistency_voting():
    path = _temp_db(DDL)
    try:
        s = schema_mod.from_db(path)
        seq = ["SELECT 1", "SELECT 1", "SELECT 2"]
        llm = FakeLLM(lambda p, i: seq[i])
        res = pipeline.generate_sql("x", s, mode="few_shot", db_path=path, samples=3,
                                    generate_fn=llm, value_linking=False)
        assert res["sql"] == "SELECT 1" and llm.calls == 3   # majority result wins
    finally:
        os.remove(path)


# ---- evaluate aggregation ----

def test_aggregate():
    modes = pipeline.MODES

    def mode_row(exact, exec_m, valid, retried):
        return {"sql": "x", "exact_match": exact, "exec_match": exec_m,
                "valid": valid, "retried": retried, "latency_ms": 10.0, "tokens": 7}

    r1 = {"ungradable": False, "cats": ["join"], "hardness": "medium",
          "modes": {m: mode_row(False, m == "few_shot_retry", True, m == "few_shot_retry")
                    for m in modes}}
    r2 = {"ungradable": True, "cats": ["subquery"], "hardness": "extra",
          "modes": {m: mode_row(False, None, False, False) for m in modes}}
    summary = evaluate.aggregate([r1, r2], modes)
    assert summary["gradable"] == 1 and summary["ungradable"] == 1
    assert summary["exec_match"]["few_shot_retry"] == 1.0
    assert summary["exec_match"]["zero_shot"] == 0.0
    assert summary["by_category_exec_match"]["few_shot_retry"]["join"] == 1.0
    assert summary["retry"]["fixed_by_retry"] == 1


# ---- API safety ----

def test_api_safe_db_path_and_materialize():
    inside = _temp_db("CREATE TABLE x (id INT);")
    try:
        assert api._safe_db_path(inside) is None
    finally:
        os.remove(inside)
    mat = api._materialize("CREATE TABLE a (id INT); DROP TABLE a; CREATE TABLE b (id INT);")
    try:
        conn = sqlite3.connect(mat)
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        assert names == {"a", "b"}
    finally:
        os.remove(mat)


def test_extract_sql():
    assert pipeline.extract_sql("```sql\nSELECT * FROM t;\n```") == "SELECT * FROM t"
    assert pipeline.extract_sql("Sure! SELECT name FROM student") == "SELECT name FROM student"


# ---- API HTTP layer (TestClient, model stubbed so no Ollama needed) ----

from fastapi.testclient import TestClient  # noqa: E402  (httpx-backed)


def _stub_llm(sql):
    """Replace the Ollama call with a fixed completion; returns the original."""
    orig = api.pipeline.ollama_client.generate_full
    api.pipeline.ollama_client.generate_full = (
        lambda prompt, temperature=0.0: {"response": sql, "eval_count": 3,
                                         "prompt_eval_count": 5})
    return orig


def test_api_root_points_to_ui():
    r = TestClient(api.app).get("/")
    assert r.status_code == 200
    body = r.json()
    assert "8501" in body["ui"] and "/generate" in body["endpoints"]


def test_api_health_and_schema():
    c = TestClient(api.app)
    assert c.get("/health").json()["ok"] is True
    body = c.get("/schema").json()                 # defaults to sample.db
    assert body.get("description") and "Table" in body["description"]


def test_api_generate_executes_against_sample_db():
    orig = _stub_llm("SELECT name FROM artist")
    fd, logp = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    old_log, api._LOG_PATH = api._LOG_PATH, logp
    try:
        r = TestClient(api.app).post(
            "/generate", json={"question": "list artist names", "mode": "few_shot_retry"})
        assert r.status_code == 200
        data = r.json()
        assert data["sql"] == "SELECT name FROM artist"
        assert data["executed"] is True and data["rows"]      # ran against real data
        assert os.path.getsize(logp) > 0                       # request was logged
    finally:
        api.pipeline.ollama_client.generate_full = orig
        api._LOG_PATH = old_log
        os.remove(logp)


def test_api_generate_rejects_db_path_outside_allowlist():
    orig = _stub_llm("SELECT 1")
    try:
        r = TestClient(api.app).post(
            "/generate", json={"question": "x", "db_path": "C:/Windows/win.ini"})
        assert r.status_code == 400                            # allow-list guard
    finally:
        api.pipeline.ollama_client.generate_full = orig


def test_api_generate_validates_question_size():
    c = TestClient(api.app)
    assert c.post("/generate", json={"question": ""}).status_code == 422
    assert c.post("/generate",
                  json={"question": "a" * (api._MAX_QUESTION + 1)}).status_code == 422


def test_api_generate_structure_only_from_schema_sql():
    orig = _stub_llm("SELECT name FROM department")
    try:
        r = TestClient(api.app).post("/generate", json={
            "question": "names",
            "schema_sql": "CREATE TABLE department (id INTEGER, name TEXT);"})
        assert r.status_code == 200
        data = r.json()
        assert data["structure_only"] is True and data["executed"] is True
        assert "empty database" in (data.get("note") or "")
    finally:
        api.pipeline.ollama_client.generate_full = orig


def test_api_key_required_when_set():
    old, api._API_KEY = api._API_KEY, "secret"
    try:
        api.require_key("secret")                              # correct key -> no raise
        try:
            api.require_key(None)
            assert False, "missing key should be rejected"
        except api.HTTPException as exc:
            assert exc.status_code == 401
    finally:
        api._API_KEY = old


def test_rate_limit_blocks_after_threshold():
    import types
    old_rate, api._RATE = api._RATE, 2
    api._hits.clear()
    req = types.SimpleNamespace(client=types.SimpleNamespace(host="9.9.9.9"))
    try:
        api.rate_limit(req)
        api.rate_limit(req)                                    # 2 within the window: ok
        try:
            api.rate_limit(req)
            assert False, "third call should exceed the limit"
        except api.HTTPException as exc:
            assert exc.status_code == 429
    finally:
        api._RATE = old_rate
        api._hits.clear()


# ---- accuracy-improvement features ----

def test_value_hints_fuzzy_substring():
    path = _temp_db(
        "CREATE TABLE artist (id INTEGER, name TEXT);",
        [("INSERT INTO artist VALUES (?,?)", [(1, "Radiohead")])],
    )
    try:
        s = schema_mod.from_db(path)
        hints = linking.value_hints(path, s, "show me radio tracks")
        assert any("Radiohead" in h for h in hints)    # 'radio' -> 'Radiohead' via LIKE
    finally:
        os.remove(path)


def test_embed_rank_tables_orders_by_similarity():
    tables = [Table("movies", [Column("title", "TEXT")]),
              Table("weather", [Column("temp", "INT")]),
              Table("staff", [Column("name", "TEXT")])]

    def fake_embed(text):
        return [1.0, 0.0] if "movie" in text.lower() else [0.0, 1.0]

    ranked = schema_mod.embed_rank_tables("list the movies", tables, 1, fake_embed)
    assert ranked and ranked[0].name == "movies"


def test_api_uses_retrieved_examples_when_configured():
    orig = _stub_llm("SELECT name FROM artist")

    class _FakeStore:
        def retrieve(self, q, k):
            return [("List the names of all artists.", "SELECT name FROM artist")]

    saved = (api._EXAMPLES_FILE, api._example_store, api._store_tried)
    api._EXAMPLES_FILE, api._example_store, api._store_tried = "dummy", _FakeStore(), True
    try:
        r = TestClient(api.app).post("/generate", json={"question": "names", "mode": "few_shot"})
        assert r.status_code == 200
        assert "Examples from other databases:" in (r.json().get("prompt") or "")
    finally:
        api._EXAMPLES_FILE, api._example_store, api._store_tried = saved
        api.pipeline.ollama_client.generate_full = orig


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nAll {len(fns)} checks passed.")
