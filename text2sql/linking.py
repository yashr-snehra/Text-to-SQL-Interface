"""Value linking: ground WHERE-clause literals in real cell values.

Given a question and a schema, find actual values in the database that match
salient question tokens (e.g. question "tracks by Radiohead" -> the value
'Radiohead' lives in artist.name). Those become explicit hints in the prompt so
the model writes the right literal instead of guessing. PII columns are skipped.
"""
import pathlib
import re
import sqlite3

from .schema import _SENSITIVE

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "by", "with",
    "all", "list", "show", "find", "what", "which", "who", "how", "many", "is",
    "are", "was", "were", "that", "have", "has", "had", "their", "there", "each",
    "name", "names", "number", "count", "average", "total", "sum", "from", "where",
}


def _connect_ro(db_path):
    try:
        uri = pathlib.Path(db_path).resolve().as_uri() + "?mode=ro"
        return sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        return sqlite3.connect(db_path)


def _salient_tokens(question):
    tokens = list(re.findall(r"['\"]([^'\"]{2,40})['\"]", question))  # quoted phrases first
    for word in re.findall(r"[A-Za-z][A-Za-z0-9]+", question):
        if len(word) >= 3 and word.lower() not in _STOP:
            tokens.append(word)
    seen, out = set(), []
    for tok in tokens:
        if tok.lower() not in seen:
            seen.add(tok.lower())
            out.append(tok)
    return out[:12]


def value_hints(db_path, schema, question, max_hints=8, max_queries=80):
    """Return ["table.col = 'value'", ...] for cell values that match the
    question. Exact, case-insensitive matches only -- precise and cheap."""
    tokens = _salient_tokens(question)
    if not tokens or not getattr(schema, "tables", None):
        return []
    conn = _connect_ro(db_path)
    hints, seen, queries = [], set(), 0
    try:
        cur = conn.cursor()
        for table in schema.tables:
            for col in table.columns:
                if not col.is_textish or _SENSITIVE.search(col.name):
                    continue
                for tok in tokens:
                    if queries >= max_queries or len(hints) >= max_hints:
                        return hints
                    val = None
                    try:
                        queries += 1
                        row = cur.execute(
                            f'SELECT "{col.name}" FROM "{table.name}" '
                            f'WHERE "{col.name}" = ? COLLATE NOCASE LIMIT 1',
                            (tok,),
                        ).fetchone()
                        if row and row[0] is not None:
                            val = row[0]
                        elif len(tok) >= 4 and queries < max_queries:
                            # Fuzzy fallback: a salient token often appears inside a
                            # longer cell value ("radio" -> "Radiohead", a misspelling,
                            # or a multi-word name). Substring match catches those.
                            queries += 1
                            row = cur.execute(
                                f'SELECT "{col.name}" FROM "{table.name}" '
                                f'WHERE "{col.name}" LIKE ? COLLATE NOCASE LIMIT 1',
                                (f"%{tok}%",),
                            ).fetchone()
                            if row and row[0] is not None:
                                val = row[0]
                    except sqlite3.Error:
                        continue
                    if val is not None:
                        key = (table.name, col.name, str(val))
                        if key not in seen:
                            seen.add(key)
                            hints.append(f"{table.name}.{col.name} = {val!r}")
    finally:
        conn.close()
    return hints
