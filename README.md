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
make run-llm DATASET=mv MAX_ROWS=5 MODEL=gpt-oss-120b
make run-crocodile DATASET=mv MAX_ROWS=5
make run-alligator DATASET=mv MAX_ROWS=5
make run-editsim DATASET=mv MAX_ROWS=5 NIL_THRESHOLD=0.2
```

Common flags (all runners):
- `--max-rows` for smoke tests
- `--force-gt-candidate` to force GT ids into candidate sets
- `--force-id Qxxxx` (repeatable) to add extra forced ids

Makefile equivalents:
- `FORCE_GT=1`
- `FORCE_ID="Q1 Q2"`

Outputs:
- `outputs/{dataset}/{method}/{settings_hash}/predictions.csv`
- `outputs/{dataset}/{method}/{settings_hash}/report.json`

## Data

Frozen datasets live in:
- `data/mv/`
- `data/cp/`
- `data/sn/`
