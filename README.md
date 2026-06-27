# Text-to-SQL Interface

Natural language → executable SQL on real relational schemas, using a **local LLM
via Ollama** (default **`qwen2.5-coder:14b`** — SQL/code-tuned and fits an Intel
Arc 140V / 32 GB box at Q4; set `OLLAMA_MODEL=qwen2.5-coder:7b` or `llama3.1` for
lower latency), a **FastAPI** backend, and a **Streamlit** frontend. Evaluated on
the **Spider** dev set.

## Features

- **Structured schema → NL description** (`text2sql/schema.py`) — tables, primary
  keys, **foreign keys**, and sampled (PII-safe) example values. Built from a live
  SQLite file, raw `CREATE TABLE`, or a Spider `tables.json` entry
  (`text2sql/spider.py`) — important because many Spider SQLite files declare no
  foreign keys, so PRAGMA alone would yield a join-less schema.
- **Schema-aware adaptive few-shot** (`text2sql/prompts.py`) — detects categories
  (a JOIN is flagged when the question spans two tables, plural-aware) and shows
  5 worked examples incl. a multi-hop bridge. Large schemas are **pruned to the
  relevant tables + FK neighbours**, with **recall-safe re-prompting** if the
  model references a table that pruning dropped.
- **Dynamic few-shot retrieval** (`text2sql/retrieval.py`) — optionally retrieve
  the most similar `(question → gold SQL)` pairs from Spider train via
  `nomic-embed-text` embeddings, instead of the fixed toy examples.
- **Value linking** (`text2sql/linking.py`) — grounds `WHERE` literals by matching
  question tokens against real cell values (`artist.name = 'Radiohead'`), with a
  **fuzzy `LIKE` fallback** so a partial token (`radio` → `Radiohead`) still links.
- **Optional embedding schema linking** — set `TEXT2SQL_EMBED_LINKING=1` to prune
  large schemas by `nomic-embed` similarity instead of lexical overlap (falls back
  to lexical if the embed model is unavailable).
- **Repair + self-verify + self-consistency** (`text2sql/pipeline.py`) —
  `few_shot_retry` retries on execution errors (empty results are *not* failures),
  runs one self-verification rescue, and optionally votes over several samples.
- **Read-only sandboxed execution** with a VM-step abort guard and a row-fetch cap.
- **Honest metrics** (`text2sql/matching.py`, `text2sql/hardness.py`) — component
  (exact-set) match (alias/order/**self-join** aware), execution match, and
  Spider-style hardness levels; the evaluator runs in parallel.

## Quick start (Windows / PowerShell)

```powershell
.\run.ps1
```

Creates a `.venv`, installs dependencies, ensures the model, builds the demo db,
runs the self-check, then launches backend + frontend. `-SkipSetup` just launches.
Detects an Ollama server already on `:11434` (e.g. an Intel IPEX-LLM build).

### Manual setup

```bash
pip install -r requirements.txt
ollama pull qwen2.5-coder:14b   # or qwen2.5-coder:7b / llama3.1 (+ set OLLAMA_MODEL)
ollama pull nomic-embed-text    # for dynamic few-shot + embedding schema linking
python make_sample_db.py
uvicorn api:app                 # terminal 1
streamlit run app.py            # terminal 2
```

## Evaluate on Spider

Layout: `<spider-dir>/dev.json`, `<spider-dir>/tables.json`,
`<spider-dir>/database/<db_id>/<db_id>.sqlite`.

```bash
python evaluate.py --spider-dir DIR --n 100
python evaluate.py --spider-dir DIR --all --workers 2 --self-consistency 3
python evaluate.py --spider-dir DIR --train-file DIR/train_spider.json    # dynamic few-shot
python evaluate.py --spider-dir DIR --official-eval-dir path/to/spider-eval
```

| Flag | Effect |
|------|--------|
| `--all` / `--stratified` | full dev set / even sampling across categories |
| `--workers N` | concurrent generations |
| `--self-consistency K` | K samples per query, voted by result set |
| `--train-file F` | dynamic few-shot from Spider train (embeds + caches) |
| `--no-value-linking` | disable value linking |
| `--official-eval-dir D` | run Spider's official `evaluation.py` on predictions |

Writes `results.json` and prints, per condition: **exact (component) match**,
**execution match**, **valid-SQL rate**, **avg latency/tokens**, and breakdowns by
**category** and **hardness**. Questions whose *gold* query fails to run are
*ungradable* and excluded from execution accuracy.

### Measured results

On **50 stratified Spider dev questions** (real databases) with `qwen2.5-coder:14b`
in `few_shot_retry` mode, run locally on an Intel Arc 140V:

| Metric | Value |
|--------|-------|
| Strict execution-match | **~80 % (41/50)** |
| Valid-SQL rate | 100 % |
| **Semantic accuracy** (after miss review) | **~88 %** |

50-question sample (≈ ±11 % at 95 % CI), so `samples=1` (41/50) and `samples=3`
(39/50) land within noise of each other. Manual review of the misses found **about
half are exec-match artifacts, not model errors**: Spider's gold penalizes a correct
query for column order (`avg(w), year` vs `year, avg(w)`), uses debatable gold (e.g.
`count(*)` where the question implies `count(distinct …)`), or is ambiguously worded.
Crediting those puts *semantic* accuracy near **88 %**, against a published SOTA of
~91 %. Run `--all` for the full 1034-question number; authoritative foreign keys come
from each db's `schema.sql` DDL (15/20 dev dbs) with a PRAGMA fallback.

## Known limitations (honest)

- **Component match is not byte-identical to Spider's official exact-set match.**
  It's alias/order/self-join aware (a big improvement), but for the published
  metric use `--official-eval-dir` (points at a Spider eval checkout).
- **Execution match can have coincidental matches** on tiny DBs — inherent to
  execution accuracy; the official evaluator mitigates but doesn't eliminate it.
- **Value linking is exact (case-insensitive) match** — multi-word/fuzzy values
  beyond quoted phrases aren't linked.
- **Hardness levels are a Spider-style approximation** (sqlglot component counts).
- **Model choice dominates accuracy.** SQL/code-tuned models (`qwen2.5-coder`)
  clear multi-hop bridges that general 7–8B models (`mistral-7b`) miss. The
  harness is model-agnostic — for harder splits try a larger or more SQL-tuned
  model via `OLLAMA_MODEL` and measure. Use `--self-consistency 3` (now the eval
  default) to smooth temp-0 nondeterminism.

## Config (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `OLLAMA_MODEL` | `qwen2.5-coder:14b` | generation model (e.g. `qwen2.5-coder:7b`, `llama3.1`) |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | embedding model for retrieval / schema linking |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama endpoint |
| `OLLAMA_KEEP_ALIVE` / `OLLAMA_NUM_CTX` | `30m` / `8192` | residency / context window |
| `TEXT2SQL_DB` | `sample.db` | DB the API runs against |
| `TEXT2SQL_DB_DIR` | dir of `TEXT2SQL_DB` | allow-list root for request `db_path` |
| `TEXT2SQL_SAMPLE_VALUES` | `1` | `0` disables value sampling (privacy) |
| `TEXT2SQL_EMBED_LINKING` | `0` | `1` prunes large schemas by embedding similarity |
| `TEXT2SQL_EXAMPLES` | — | Spider-style `train.json` for dynamic few-shot in the API |
| `TEXT2SQL_RETRIEVE_K` | `5` | examples retrieved per request when `TEXT2SQL_EXAMPLES` is set |
| `TEXT2SQL_API_KEY` | — | if set, required as `X-API-Key` on `/generate` |
| `TEXT2SQL_RATE_LIMIT` | `60` | requests/min/IP (`0` disables) |
| `TEXT2SQL_LOG` | `logs/requests.jsonl` | request log path |
| `TEXT2SQL_API` | `http://localhost:8000` | API base URL the Streamlit app calls |

## Test

```bash
python test_text2sql.py   # deterministic parts incl. the LLM loop via a fake model
```

## Layout

```
text2sql/
  schema.py        structured schema (FKs, samples, pruning) -> NL description
  spider.py        Spider tables.json -> Schema (authoritative foreign keys)
  prompts.py       schema-aware / dynamic few-shot + retry/verify prompt builders
  linking.py       value linking (WHERE literals from real cell values)
  retrieval.py     dynamic few-shot via nomic-embed embeddings
  pipeline.py      generate -> recall-safe link -> repair -> self-verify -> vote
  matching.py      component (exact-set, self-join aware) + execution match
  hardness.py      Spider-style easy/medium/hard/extra
  ollama_client.py Ollama generate + embeddings
api.py             FastAPI backend (allow-list, materialization, auth, rate limit, logging)
app.py             Streamlit frontend
evaluate.py        Spider eval: 3 conditions, parallel, hardness, official hook
make_sample_db.py  builds sample.db for the demo
test_text2sql.py   self-check
run.ps1            one-command setup + launch
```
