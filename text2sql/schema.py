"""Structured database schema -> natural-language description.

A `Schema` carries the structure the model needs (tables, columns, primary keys,
**foreign keys**, and optional sampled values) and renders it to the NL string
fed into the prompt. Foreign keys drive JOINs and value samples drive WHERE
literals -- the two things that most move text-to-SQL accuracy.

Schemas can be built from a live SQLite file (PRAGMA), from raw CREATE TABLE
statements (sandboxed in-memory), or from a Spider tables.json entry (see
spider.py) -- the last is important because many Spider SQLite files omit FK
constraints, so PRAGMA alone would hand the model a schema with no joins.
"""
import functools
import os
import re
import sqlite3
from dataclasses import dataclass, field

_SAMPLE_VALUES = 3     # distinct example values shown per text column
_SAMPLE_MAX_LEN = 40   # skip long text blobs -- they bloat the prompt
_TEXTISH = {"TEXT", "VARCHAR", "CHAR", "CLOB", "STRING", "NVARCHAR", ""}

# Columns whose values must never be sampled into a prompt (PII / secrets).
_SENSITIVE = re.compile(
    r"pass|pwd|secret|token|api[_-]?key|ssn|social|credit|card|cvv|"
    r"email|e-mail|phone|mobile|dob|birth|salary|income|address|postcode|zip",
    re.IGNORECASE,
)
# Global off-switch: TEXT2SQL_SAMPLE_VALUES=0 disables value sampling entirely.
_SAMPLING_ENABLED = os.environ.get("TEXT2SQL_SAMPLE_VALUES", "1") not in ("0", "false", "False")
# Opt-in: rank tables for pruning by embedding similarity instead of lexical
# token overlap (needs an Ollama embed model). Falls back to lexical on failure.
_EMBED_LINKING = os.environ.get("TEXT2SQL_EMBED_LINKING", "0") in ("1", "true", "True")


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


_IRREGULAR = {"people": "person", "children": "child", "men": "man",
              "women": "woman", "feet": "foot", "teeth": "tooth",
              "mice": "mouse", "geese": "goose"}


def _singular(word):
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _loose_match(name, qtokens):
    """True if a schema name plausibly appears in the question tokens (matching
    singular/plural forms: 'tracks' -> 'track', 'categories' -> 'category')."""
    n = name.lower()
    if len(n) <= 2:
        return False
    ns = _singular(n)
    for tok in qtokens:
        if tok == n or _singular(tok) == ns:
            return True
        if len(tok) > 3 and (tok.startswith(n) or n.startswith(tok)):
            return True
    return False


def embed_rank_tables(question, tables, max_tables, embed_fn):
    """Rank tables by cosine similarity of their (name + columns) text to the
    question, returning the top-`max_tables` Table objects. Returns None on any
    failure (missing numpy, embed model down, ...) so the caller can fall back
    to lexical ranking."""
    try:
        import numpy as np
    except Exception:
        return None
    try:
        qv = np.asarray(embed_fn(question), dtype="float32")
        qv /= (np.linalg.norm(qv) or 1.0)
        scored = []
        for t in tables:
            text = t.name + " " + " ".join(c.name for c in t.columns)
            tv = np.asarray(embed_fn(text), dtype="float32")
            tv /= (np.linalg.norm(tv) or 1.0)
            scored.append((float(qv @ tv), t))
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:max_tables]]
    except Exception:
        return None


@dataclass
class Column:
    name: str
    type: str = "TEXT"
    samples: list = field(default_factory=list)

    @property
    def is_textish(self):
        return self.type.upper().split("(")[0] in _TEXTISH


@dataclass
class Table:
    name: str
    columns: list = field(default_factory=list)
    pks: list = field(default_factory=list)
    fks: list = field(default_factory=list)  # (from_col, ref_table, ref_col)

    def line(self):
        parts = []
        for col in self.columns:
            desc = f"{col.name} ({col.type or 'TEXT'})"
            if col.samples:
                desc += " e.g. " + ", ".join(repr(v) for v in col.samples)
            parts.append(desc)
        line = f'Table "{self.name}" has columns: ' + ", ".join(parts) + "."
        if self.pks:
            line += f" Primary key: {', '.join(self.pks)}."
        for from_col, ref_table, ref_col in self.fks:
            line += f" Column {from_col} references {ref_table}.{ref_col}."
        return line


@dataclass
class Schema:
    tables: list = field(default_factory=list)

    def table_names(self):
        return [t.name for t in self.tables]

    def fill_samples(self, db_path):
        """Populate column sample values from a SQLite file (skips PII columns)."""
        if not _SAMPLING_ENABLED or not db_path or not os.path.exists(db_path):
            return self
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            for table in self.tables:
                for col in table.columns:
                    if col.is_textish and not _SENSITIVE.search(col.name):
                        col.samples = _sample_values(cur, table.name, col.name)
        except sqlite3.Error:
            pass
        finally:
            conn.close()
        return self

    def _expand_fk(self, chosen):
        """Add FK-connected neighbours of `chosen` so joins stay expressible."""
        by_name = {t.name: t for t in self.tables}
        for name in list(chosen):
            for _, ref_table, _ in by_name[name].fks:
                chosen.add(ref_table)
            for t in self.tables:
                if any(ref == name for _, ref, _ in t.fks):
                    chosen.add(t.name)
        return chosen

    def _relevant_tables(self, question, max_tables):
        """Pick the tables most relevant to the question, then pull in their
        FK neighbours so JOINs stay expressible. Avoids dumping a huge schema
        (which Ollama would silently truncate past num_ctx)."""
        if len(self.tables) <= max_tables:
            return self.tables

        if _EMBED_LINKING:  # semantic ranking; falls back to lexical on failure
            from . import ollama_client
            ranked = embed_rank_tables(question, self.tables, max_tables, ollama_client.embed)
            if ranked:
                chosen = self._expand_fk({t.name for t in ranked})
                return [t for t in self.tables if t.name in chosen]

        qtokens = _tokens(question)
        ql = question.lower()

        def score(table):
            s = 2 if _loose_match(table.name, qtokens) else 0
            for col in table.columns:
                if _loose_match(col.name, qtokens):
                    s += 1
                for v in col.samples:
                    if isinstance(v, str) and v.lower() in ql:
                        s += 2
            return s

        ranked = sorted(self.tables, key=score, reverse=True)
        chosen = {t.name for t in ranked[:max_tables] if score(t) > 0}
        if not chosen:  # nothing matched -- fall back to the first few tables
            chosen = {t.name for t in ranked[:max_tables]}
        chosen = self._expand_fk(chosen)
        return [t for t in self.tables if t.name in chosen]

    def describe(self, question=None, max_tables=10):
        tables = self.tables
        if question is not None:
            tables = self._relevant_tables(question, max_tables)
        return "\n".join(t.line() for t in tables)


def _sample_values(cur, table, col):
    try:
        rows = cur.execute(
            f'SELECT DISTINCT "{col}" FROM "{table}" '
            f'WHERE "{col}" IS NOT NULL AND length("{col}") <= ? LIMIT ?',
            (_SAMPLE_MAX_LEN, _SAMPLE_VALUES),
        ).fetchall()
    except sqlite3.Error:
        return []
    return [r[0] for r in rows]


def _schema_from_conn(conn):
    cur = conn.cursor()
    names = [
        r[0]
        for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    tables = []
    for name in names:
        cols, pks, fks = [], [], []
        for _, cname, ctype, _, _, pk in cur.execute(f'PRAGMA table_info("{name}")'):
            cols.append(Column(cname, ctype or "TEXT"))
            if pk:
                pks.append(cname)
        for fk in cur.execute(f'PRAGMA foreign_key_list("{name}")'):
            fks.append((fk[3], fk[2], fk[4]))  # (from, ref_table, ref_col)
        tables.append(Table(name, cols, pks, fks))
    return Schema(tables)


@functools.lru_cache(maxsize=256)
def _describe_db_cached(path, with_samples):
    conn = sqlite3.connect(path)
    try:
        schema = _schema_from_conn(conn)
    finally:
        conn.close()
    if with_samples:
        schema.fill_samples(path)
    return schema


def from_db(path, with_samples=True):
    """Build a Schema from an existing SQLite file (cached)."""
    return _describe_db_cached(path, with_samples)


def from_create_statements(sql):
    """Build a Schema from raw CREATE TABLE statements (DDL only, sandboxed)."""
    ddl = ";\n".join(s for s in sql.split(";") if s.strip().upper().startswith("CREATE"))
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(ddl)
        return _schema_from_conn(conn)  # fresh db -> no rows to sample
    finally:
        conn.close()


# Backwards-compatible string helpers (used by the API / older callers).
def describe_db(path, with_samples=True):
    return from_db(path, with_samples).describe()


def describe_create_statements(sql):
    return from_create_statements(sql).describe()
