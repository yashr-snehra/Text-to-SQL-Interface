"""Spider-style query difficulty: easy / medium / hard / extra.

This is an approximation of Spider's official `eval_hardness` (which counts SQL
components over their own parse). We count comparable components with sqlglot.
For the exact published buckets, use evaluate.py --official-eval-dir.
"""
import sqlglot
from sqlglot import exp


def hardness(sql):
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return "medium"
    if tree is None:
        return "medium"

    nested = max(0, len(list(tree.find_all(exp.Select))) - 1)
    setops = len(list(tree.find_all(exp.Union, exp.Intersect, exp.Except)))
    aggs = len(list(tree.find_all(exp.AggFunc)))
    joins = len(list(tree.find_all(exp.Join)))
    has_group = tree.find(exp.Group) is not None
    has_order = tree.find(exp.Order) is not None
    has_having = tree.find(exp.Having) is not None

    where = tree.find(exp.Where)
    n_where = 0
    if where is not None:
        n_where = 1 + len(list(where.find_all(exp.And))) + len(list(where.find_all(exp.Or)))

    comp1 = aggs + n_where + has_group + has_order + has_having  # "where/agg/clause" load
    comp2 = nested + setops                                      # "nesting" load

    if comp2 == 0 and comp1 <= 1 and joins == 0:
        return "easy"
    if comp2 == 0 and comp1 <= 2 and joins <= 1:
        return "medium"
    if comp2 <= 1 and comp1 <= 4:
        return "hard"
    return "extra"
