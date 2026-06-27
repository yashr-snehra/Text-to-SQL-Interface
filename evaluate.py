"""Evaluate on Spider dev: zero-shot vs few-shot vs few-shot+retry.

Per condition it reports exact (component) match, execution match, valid-SQL
rate, average latency/tokens, and a breakdown by category and by Spider-style
hardness (easy/medium/hard/extra). Questions whose *gold* query fails to run are
counted as ungradable and excluded from execution accuracy.

Schemas use Spider tables.json (authoritative foreign keys) + value samples.

Usage:
    python evaluate.py --spider-dir DIR --n 100
    python evaluate.py --spider-dir DIR --all --workers 2
    python evaluate.py --spider-dir DIR --n 120 --stratified --self-consistency 3
    python evaluate.py --spider-dir DIR --train-file DIR/train_spider.json   # dynamic few-shot
    python evaluate.py --spider-dir DIR --official-eval-dir path/to/spider-eval-repo
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from text2sql import pipeline, retrieval
from text2sql import schema as schema_mod
from text2sql import spider
from text2sql.hardness import hardness
from text2sql.matching import canonicalize, exec_match, is_ordered

MODES = list(pipeline.MODES)


def categorize(gold_sql):
    s = gold_sql.lower()
    cats = []
    if " join " in s:
        cats.append("join")
    if "group by" in s or re.search(r"\b(count|avg|sum|max|min)\s*\(", s):
        cats.append("aggregation")
    if s.count("select") > 1:
        cats.append("subquery")
    return cats or ["simple"]


def db_path_for(spider_dir, db_id):
    return os.path.join(spider_dir, "database", db_id, db_id + ".sqlite")


def _sample(data, n, seed, stratified):
    random.seed(seed)
    if n >= len(data):
        return list(data)
    if not stratified:
        return random.sample(data, n)
    buckets = defaultdict(list)
    for item in data:
        buckets[categorize(item["query"])[0]].append(item)
    for b in buckets.values():
        random.shuffle(b)
    chosen, order = [], sorted(buckets)
    while len(chosen) < n and any(buckets[c] for c in order):
        for c in order:
            if buckets[c] and len(chosen) < n:
                chosen.append(buckets[c].pop())
    return chosen


def _grade_one(item, spider_dir, schema, samples, examples, value_linking):
    db_id, question, gold = item["db_id"], item["question"], item["query"]
    dbp = db_path_for(spider_dir, db_id)

    g_ok, g_rows, _ = pipeline.try_execute(dbp, gold)
    gold_exec = g_rows if g_ok else None
    ordered = is_ordered(gold)
    gold_canon = canonicalize(gold)

    rec = {"db_id": db_id, "question": question, "gold": gold,
           "cats": categorize(gold), "hardness": hardness(gold),
           "ungradable": gold_exec is None, "modes": {}}
    for m in MODES:
        res = pipeline.generate_sql(question, schema, mode=m, db_path=dbp,
                                    samples=samples, examples=examples,
                                    value_linking=value_linking)
        pred = res["sql"]
        p_ok, p_rows, _ = pipeline.try_execute(dbp, pred)
        rec["modes"][m] = {
            "sql": pred,
            "exact_match": canonicalize(pred) == gold_canon,
            "exec_match": exec_match(gold_exec, p_rows if p_ok else None, ordered),
            "valid": p_ok,
            "retried": res["retried"],
            "latency_ms": round(res["stats"]["latency_ms"], 1),
            "tokens": res["stats"]["completion_tokens"],
        }
    return rec


def aggregate(records, modes):
    """Pure aggregation over graded records -> summary dict (unit-testable)."""
    n = len(records)
    ungradable = sum(r["ungradable"] for r in records)
    gradable = n - ungradable

    em = {m: 0 for m in modes}
    ex = {m: 0 for m in modes}
    valid = {m: 0 for m in modes}
    lat = {m: 0.0 for m in modes}
    toks = {m: 0 for m in modes}
    cat_total, hard_total = defaultdict(int), defaultdict(int)
    cat_correct = {m: defaultdict(int) for m in modes}
    hard_correct = {m: defaultdict(int) for m in modes}
    retried_total = retried_fixed = 0

    for r in records:
        if not r["ungradable"]:
            for c in r["cats"]:
                cat_total[c] += 1
            hard_total[r["hardness"]] += 1
        for m in modes:
            md = r["modes"][m]
            em[m] += int(md["exact_match"])
            valid[m] += int(md["valid"])
            lat[m] += md.get("latency_ms", 0.0)
            toks[m] += md.get("tokens", 0)
            if md["exec_match"] is True:
                ex[m] += 1
                for c in r["cats"]:
                    cat_correct[m][c] += 1
                hard_correct[m][r["hardness"]] += 1
            if md["retried"]:
                retried_total += 1
                retried_fixed += int(md["exec_match"] is True)

    def rate(num, den):
        return round(num / den, 3) if den else None

    return {
        "n": n, "gradable": gradable, "ungradable": ungradable,
        "exact_match": {m: rate(em[m], n) for m in modes},
        "exec_match": {m: rate(ex[m], gradable) for m in modes},
        "valid_sql": {m: rate(valid[m], n) for m in modes},
        "avg_latency_ms": {m: rate(lat[m], n) for m in modes},
        "avg_completion_tokens": {m: rate(toks[m], n) for m in modes},
        "by_category_exec_match": {
            m: {c: rate(cat_correct[m][c], cat_total[c]) for c in sorted(cat_total)}
            for m in modes},
        "by_hardness_exec_match": {
            m: {h: rate(hard_correct[m][h], hard_total[h]) for h in sorted(hard_total)}
            for m in modes},
        "category_counts": dict(cat_total),
        "hardness_counts": dict(hard_total),
        "retry": {"retried": retried_total, "fixed_by_retry": retried_fixed},
    }


def run_official(records, spider_dir, tables_file, official_dir, mode):
    """Best-effort: run Spider's official evaluation.py on one mode's predictions."""
    script = os.path.join(official_dir, "evaluation.py")
    if not os.path.exists(script):
        print(f"[official] no evaluation.py in {official_dir}; skipping")
        return
    gold_path, pred_path = "official_gold.txt", "official_pred.txt"
    with open(gold_path, "w", encoding="utf-8") as gf, open(pred_path, "w", encoding="utf-8") as pf:
        for r in records:
            gf.write(f"{r['gold']}\t{r['db_id']}\n")
            pf.write(f"{r['modes'][mode]['sql'] or 'SELECT 1'}\n")
    cmd = [sys.executable, script, "--gold", gold_path, "--pred", pred_path,
           "--db", os.path.join(spider_dir, "database"), "--table", tables_file,
           "--etype", "all"]
    print(f"[official] running Spider evaluator on '{mode}' predictions...")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        print(out.stdout[-4000:] or out.stderr[-2000:])
    except Exception as exc:
        print(f"[official] failed: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider-dir", required=True)
    ap.add_argument("--dev-file", default=None)
    ap.add_argument("--tables-file", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--self-consistency", type=int, default=3, dest="samples",
                    help="samples voted by result set; 3 smooths temp-0 nondeterminism (1 = greedy)")
    ap.add_argument("--train-file", default=None, help="Spider train.json for dynamic few-shot")
    ap.add_argument("--retrieve-k", type=int, default=5)
    ap.add_argument("--train-limit", type=int, default=2000, help="cap train examples embedded")
    ap.add_argument("--no-value-linking", action="store_true")
    ap.add_argument("--official-eval-dir", default=None, help="path to a Spider eval repo")
    ap.add_argument("--official-mode", default="few_shot_retry")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    dev_file = args.dev_file or os.path.join(args.spider_dir, "dev.json")
    tables_file = args.tables_file or os.path.join(args.spider_dir, "tables.json")
    with open(dev_file, encoding="utf-8") as f:
        data = json.load(f)
    tables = spider.load_tables(tables_file) if os.path.exists(tables_file) else {}
    sample = data if args.all else _sample(data, args.n, args.seed, args.stratified)

    store = None
    if args.train_file:
        print(f"Building dynamic few-shot store from {args.train_file} ...")
        store = retrieval.build_from_train(args.train_file, limit=args.train_limit)

    schemas = {}
    for item in sample:
        db_id = item["db_id"]
        if db_id not in schemas:
            dbp = db_path_for(args.spider_dir, db_id)
            schemas[db_id] = (spider.schema_from_entry(tables[db_id], dbp)
                              if db_id in tables else schema_mod.from_db(dbp))

    value_linking = not args.no_value_linking
    records = [None] * len(sample)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for i, item in enumerate(sample):
            ex = store.retrieve(item["question"], args.retrieve_k) if store else None
            futs[pool.submit(_grade_one, item, args.spider_dir,
                             schemas[item["db_id"]], args.samples, ex, value_linking)] = i
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            records[i] = fut.result()
            done += 1
            print(f"[{done}/{len(sample)}] {records[i]['db_id']}")

    summary = aggregate(records, MODES)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    print(f"\n=== SUMMARY (n={summary['n']}, gradable={summary['gradable']}, "
          f"ungradable={summary['ungradable']}) ===")
    print("Exact-match (component):", summary["exact_match"])
    print("Execution accuracy:     ", summary["exec_match"])
    print("Valid-SQL rate:         ", summary["valid_sql"])
    print("Avg latency (ms):       ", summary["avg_latency_ms"])
    print("Execution accuracy by category:")
    for m in MODES:
        print(f"  {m}: {summary['by_category_exec_match'][m]}")
    print("Execution accuracy by hardness:")
    for m in MODES:
        print(f"  {m}: {summary['by_hardness_exec_match'][m]}")
    print(f"Category counts: {summary['category_counts']}")
    print(f"Hardness counts: {summary['hardness_counts']}")
    print(f"Retries fired: {summary['retry']['retried']}, fixed: {summary['retry']['fixed_by_retry']}")
    print(f"Wrote {args.out}")

    if args.official_eval_dir:
        run_official(records, args.spider_dir, tables_file, args.official_eval_dir, args.official_mode)


if __name__ == "__main__":
    main()
