# benchmarking-LLMs
Reproducible entity-linking benchmarks with LamAPI retrieval and multiple runners.

## Quickstart

1) Configure `.env`:
```
ENTITY_RETRIEVAL_ENDPOINT=...
ENTITY_RETRIEVAL_TOKEN=...
```

2) Build datasets:
```bash
make build-datasets
```

3) Run a smoke test:
```bash
make run-editsim DATASET=mv MAX_ROWS=5 NIL_THRESHOLD=0.2 FORCE_GT=1
```

4) Evaluate:
```bash
make eval PRED=outputs/mv/editsim/<hash>/predictions.csv GT=data/mv/gt.csv
```

## Runners

```bash
make run-llm DATASET=mv MAX_ROWS=5 MODEL=openai/gpt-oss-120b CONCURRENCY=16
make run-llm DATASET=semtab:2T_2020 MAX_ROWS=200   # any SemTab folder under DATASETS_ROOT
make run-crocodile DATASET=mv MAX_ROWS=5
make run-alligator DATASET=mv MAX_ROWS=5
make run-editsim DATASET=mv MAX_ROWS=5 NIL_THRESHOLD=0.2
```

Note: Crocodile and Alligator require MongoDB to run their native pipelines.
Use the included `docker-compose.yml` to start MongoDB:

```bash
docker compose up -d
```

Common flags (all runners):
- `--max-rows` for smoke tests
- `--force-gt-candidate` to force GT ids into candidate sets (always on for `run-llm`; opt out with `--no-force-gt`)
- `--force-id Qxxxx` (repeatable) to add extra forced ids

Forced ids are passed to LamAPI as `ids=Q1 Q2` (space-separated; every GT id of the cell), so the
correct entity is always among the candidates. Runs report how many cells still miss it (ids
LamAPI does not know, or non-Wikidata GT such as `cp_crunchbase`).

Makefile equivalents:
- `FORCE_GT=1`
- `FORCE_ID="Q1 Q2"`

Outputs:
- `outputs/{dataset}/{method}/{settings_hash}/predictions.csv`
- `outputs/{dataset}/{method}/{settings_hash}/report.json`

## Crocodile and Alligator

The classic table linkers [Crocodile](https://github.com/enRichMyData/crocodile) (`crocodile-linker`)
and [Alligator](https://github.com/enRichMyData/alligator) (`alligator-linker`) run from the same
console (Runs page, **Methods**) on the same datasets, LamAPI candidates and GT forcing, and are
scored with the same P/R/F1. They pin `tensorflow==2.18` and `numpy<2` (Python <= 3.12), so they
run in their own image and worker: the `linkers` service of `docker compose` (`Dockerfile.linkers`,
~1.8 GB, built once), which uses the `mongo` service as their working store.

Each table is linked by one instance: rows up to the last target cell, the library's column
classifier with every target column forced to NE, the cell's GT ids as `correct_qids`, and the
top candidate it stores in MongoDB as the prediction. Crocodile also gets LamAPI's bag-of-words
endpoint (`ENTITY_BOW_ENDPOINT`, default `<lamapi>/entity/bow`; `--no-bow` / Runs page checkbox to
turn it off): it does not read that variable itself, and without it its `bow_similarity` ranking
feature is 0 for every candidate (on 100 `mv` rows: F1 0.93 without, 1.00 with). CLI (inside the image):

```bash
make run-crocodile DATASET=mv MAX_ROWS=100
docker compose run --rm linkers python -m experiments.runners.run_alligator --dataset semtab:2t-red --max-rows 500
python -m experiments.runs submit --method crocodile --dataset mv --limit 100   # queued for the linkers worker
```

Earlier versions of `run_crocodile`/`run_alligator` silently fell back to LamAPI's top candidate
when the libraries were missing; outputs in `outputs/*/crocodile|alligator` produced before this
change are that fallback, not the linkers.

## Running and monitoring LLM runs

The console (`make up` or `make ui`) has three pages:

- **Runs** - queue dataset × prompt × model runs with a duration estimate, then watch them live:
  progress, throughput, ETA, cost so far / projected, requests in flight, errors and 429s, and the
  share of cells whose GT id is among the candidates. Cancel, resume or retry failed cells.
- **Estimate cost** - tokens, USD and duration before running (see below).
- **Cache** - size of the LamAPI / OpenRouter caches and run outputs, and how many of a dataset's
  LamAPI lookups are already cached.

Runs are executed by a **worker**, not by the console, so they survive browser refreshes and
console restarts. `docker compose up -d` starts one; without Docker run `make worker` next to
`make ui`. The run store is `outputs/runs.db` (SQLite; override with `RUNS_DB`).

```bash
python -m experiments.runs submit --dataset mv --model openai/gpt-oss-120b --concurrency 32
python -m experiments.runs list          # or: make runs
python -m experiments.runs cancel <id>
python -m experiments.runs resume <id>   # continues where it stopped; failed cells are retried
```

How a run works (`experiments/runs/executor.py`): prompts are built with LionLinker's own prompt
generator (byte-identical, see `tests/test_cost_estimator.py`), but requests are sent in parallel
(`--concurrency` LLM requests, optional `--max-rpm`; LamAPI lookups separately with
`--lamapi-concurrency`, default 8, on their own thread and connection pool) instead of
LionLinker's one-at-a-time loop. A 429 pauses
all requests (honouring `Retry-After`) before retrying; other transient errors are retried with
backoff; failed cells are recorded as errors instead of silently becoming NIL. Every cell is
appended to `results.jsonl` in the run folder, so runs resume. Cost comes from OpenRouter's usage
report, including the upstream provider cost for BYOK keys.

Provider routing (`--provider-policy`, Runs page "Provider"): by default runs go only to the
providers you have a BYOK key for in OpenRouter (`provider.order` = those providers, no fallbacks),
so a run stays on one provider/quantization and a 429 makes it back off instead of moving to
shared capacity. BYOK providers come from `OPENROUTER_BYOK_PROVIDERS=cerebras,...` in `.env`, the
BYOK list API (`OPENROUTER_MANAGEMENT_KEY`), or are learned from `is_byok` in earlier runs (the
very first run uses OpenRouter routing). OpenRouter can still serve some calls from the same
provider's shared capacity when your key is throttled, and no request option forbids it: under
"BYOK only" such answers (`is_byok: false`) are discarded, the run backs off and retries, and a run
with no known BYOK provider refuses to start. Discarded answers are still billed (shown as
`shared_rejected_usd`); set the key to "Never use shared capacity" in OpenRouter so it answers
429 instead.

Duration estimate: throughput = min(parallel requests ÷ seconds per call, requests/min cap ÷ 60);
uncached LamAPI lookups run alongside. Seconds per call come from calibration, else the median of
earlier runs of the same model, else an assumption. During a run the ETA uses the throughput
measured over the last 90 s.

## Cost estimation (before running LLMs)

With Docker (recommended):

```bash
make up     # = docker compose up -d --build  ->  http://localhost:8501 (+ LLM worker, linkers worker, MongoDB)
make logs   # follow the console logs
make down
```

The repo is bind-mounted into the container (code edits reload live; `outputs/` and caches stay
on the host) and `DATASETS_ROOT` from `.env` is mounted read-only at `/datasets`.

Without Docker:

```bash
pip install -r requirements.txt   # adds streamlit + tiktoken
make ui                           # opens http://localhost:8501
```

Pick datasets × prompt schemas × models and get LLM calls, tokens and USD. The estimator
rebuilds the exact LionLinker prompts (checked by `tests/test_cost_estimator.py`), counts tokens
with tiktoken and prices them with live OpenRouter rates. No LLM is called.

CLI equivalent:

```bash
make estimate DATASET=mv MODEL=openai/gpt-oss-120b
python -m experiments.cost.estimate --dataset mv --dataset HardTablesR2 \
    --schema base --schema fewshot --model openai/gpt-oss-120b --json estimate.json

# prices differ per OpenRouter provider: default is the cheapest healthy one
python -m experiments.cost.estimate --dataset mv --model openai/gpt-oss-120b --provider groq

# exact numbers: send 20 fully filled prompts per run (real candidates + real table summary)
python -m experiments.cost.estimate --dataset mv --model openai/gpt-oss-120b --calibrate 20
```

Prompts are always filled with real LamAPI candidates (free; it also warms `data/lamapi_cache`
for the real run). Calibration replaces the remaining assumptions — table summary length,
output/reasoning tokens and the provider's tokenizer — with values measured from real calls.

SemTab datasets are discovered from `DATASETS_ROOT` (set in `.env`, `~` allowed, e.g.
`DATASETS_ROOT=~/Documents/Datasets`): every sub-folder with `tables/*.csv` and a `gt/cea*.csv`
file (`table,row,col,uri`, row 1 = first data row; several URIs separated by spaces are all
correct and all forced into the candidates). Docker mounts the same folder read-only at `/datasets`.

## Data

Frozen datasets live in:
- `data/mv/`
- `data/cp/`
- `data/sn/`
