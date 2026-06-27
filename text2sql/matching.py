"""SQL comparison for evaluation: component match + execution match.

This is an *approximation* of Spider's official exact-set match (which uses
their component matcher over a parsed SQL structure). It is alias/format/order
insensitive, which is far more honest than raw string equality, but it is not
byte-identical to the official metric -- execution match remains the headline.
"""
import re
from collections import Counter
from functools import reduce

import sqlglot
from sqlglot import exp


def _string_norm(sql):
    s = sql.strip().rstrip(";").lower().replace('"', "'")
    return re.sub(r"\s+", " ", s).strip()


def _dealias(tree):
    """Normalize table aliases to a canonical `table__occurrence` name so alias
    *names* don't matter (`T1` vs `x`) while self-joins stay distinct.

    Each table reference gets `name__k` where k counts occurrences of that table.
    A self-join (`person AS a JOIN person AS b`) becomes `person__1`/`person__2`
    instead of collapsing both to `person` -- the old version made `a.x`/`b.x`
    indistinguishable and mis-scored every self-join."""
    counts = {}
    alias_map = {}                       # original alias -> canonical
    for t in tree.find_all(exp.Table):
        base = (t.name or "").lower()
        counts[base] = counts.get(base, 0) + 1
        canon = f"{base}__{counts[base]}"
        if t.alias:
            alias_map[t.alias] = canon
        t.set("alias", exp.to_identifier(canon))
    single = {b for b, n in counts.items() if n == 1}
    for c in tree.find_all(exp.Column):
        tbl = c.table
        if not tbl:
            continue
        if tbl in alias_map:                       # qualified by an alias
            c.set("table", exp.to_identifier(alias_map[tbl]))
        elif tbl.lower() in single:                # qualified by the table name
            c.set("table", exp.to_identifier(f"{tbl.lower()}__1"))


def _sort_unordered(tree):
    """Make set-like clauses order-insensitive (SELECT items, WHERE AND-terms,
    GROUP BY). ORDER BY is left alone -- its order is semantic."""
    for sel in tree.find_all(exp.Select):
        if sel.expressions:
            sel.set("expressions", sorted(sel.expressions, key=lambda e: e.sql()))
        grp = sel.args.get("group")
        if grp and grp.expressions:
            grp.set("expressions", sorted(grp.expressions, key=lambda e: e.sql()))
    for where in tree.find_all(exp.Where):
        conds = list(where.this.flatten()) if isinstance(where.this, exp.And) else [where.this]
        if len(conds) > 1:
            conds = sorted(conds, key=lambda e: e.sql())
            where.set("this", reduce(lambda a, b: exp.And(this=a, expression=b), conds))


def canonicalize(sql, dialect="sqlite"):
    """Structural canonical form for component (approximate exact-set) matching."""
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return _string_norm(sql)
    if tree is None:
        return _string_norm(sql)
    try:
        _dealias(tree)
        _sort_unordered(tree)
        out = tree.sql(dialect=dialect, normalize=True)
    except Exception:
        return _string_norm(sql)
    return re.sub(r"\s+", " ", out).strip().lower()


def component_match(pred_sql, gold_sql):
    return canonicalize(pred_sql) == canonicalize(gold_sql)


def is_ordered(sql):
    """True when the query's result order is meaningful (has ORDER BY)."""
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
        return tree is not None and tree.find(exp.Order) is not None
    except Exception:
        return "order by" in sql.lower()


def _cell(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return round(float(v), 6)  # 2 vs 2.0, float rounding noise
    return str(v)


def _rows(result):
    return [tuple(_cell(c) for c in row) for row in result]


def exec_match(gold_rows, pred_rows, ordered):
    """Compare result sets.

    Returns None  -> ungradable (the *gold* query failed to run; exclude it),
            False -> prediction errored or produced a different result,
            True  -> match (sequence equality if ordered, else multiset).
    """
    if gold_rows is None:
        return None
    if pred_rows is None:
        return False
    g, p = _rows(gold_rows), _rows(pred_rows)
    if ordered:
        return g == p
    return Counter(g) == Counter(p)
