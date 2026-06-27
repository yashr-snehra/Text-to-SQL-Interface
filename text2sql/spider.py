"""Build schemas from Spider's tables.json.

Many Spider SQLite files declare no FOREIGN KEY constraints, so PRAGMA-based
introspection alone yields a schema with no joins. The authoritative FK graph
lives in tables.json -- we parse it here, then merge in value samples from the
actual SQLite file.

tables.json entry shape (the fields we use):
    db_id                  : "concert_singer"
    table_names_original   : ["stadium", "singer", ...]
    column_names_original  : [[-1,"*"], [0,"Stadium_ID"], [0,"Location"], ...]
    column_types           : ["text", "number", ...]   # parallel to columns
    primary_keys           : [1, 8, ...]                # indices into columns
    foreign_keys           : [[12, 1], ...]             # [from_col_idx, ref_col_idx]
"""
import functools
import json

from .schema import Column, Schema, Table


@functools.lru_cache(maxsize=8)
def load_tables(tables_json_path):
    """Return {db_id: entry} from a Spider tables.json file."""
    with open(tables_json_path, encoding="utf-8") as f:
        return {entry["db_id"]: entry for entry in json.load(f)}


def schema_from_entry(entry, db_path=None, with_samples=True):
    """Convert one tables.json entry into a Schema (with FKs), optionally
    sampling values from the matching SQLite file."""
    table_names = entry["table_names_original"]
    columns = entry["column_names_original"]          # [[tbl_idx, name], ...]
    col_types = entry.get("column_types", ["text"] * len(columns))

    tables = [Table(name) for name in table_names]
    for ci, (tbl_idx, col_name) in enumerate(columns):
        if tbl_idx < 0:  # the synthetic "*" column
            continue
        tables[tbl_idx].columns.append(Column(col_name, _sql_type(col_types[ci])))

    for ci in entry.get("primary_keys", []):
        # primary_keys may be a single index or a composite list of indices
        for idx in (ci if isinstance(ci, list) else [ci]):
            tbl_idx, col_name = columns[idx]
            tables[tbl_idx].pks.append(col_name)

    for from_ci, ref_ci in entry.get("foreign_keys", []):
        from_tbl, from_col = columns[from_ci]
        ref_tbl, ref_col = columns[ref_ci]
        tables[from_tbl].fks.append((from_col, table_names[ref_tbl], ref_col))

    schema = Schema(tables)
    if with_samples and db_path:
        schema.fill_samples(db_path)
    return schema


def _sql_type(spider_type):
    # Spider collapses types to "text"/"number"/"time"/"boolean"/"others".
    return {"number": "INTEGER", "time": "DATETIME", "boolean": "BOOLEAN"}.get(
        spider_type, "TEXT"
    )
