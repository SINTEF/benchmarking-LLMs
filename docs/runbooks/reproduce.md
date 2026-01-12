# Reproduction runbook

## Build datasets

```bash
make build-datasets
```

This populates `data/mv/`, `data/cp/`, and `data/sn/` with frozen tables, GT, and metadata.

## Run experiments

All runners share the same interface.

```bash
make run-llm DATASET=mv MAX_ROWS=5 MODEL=gpt-oss-120b
make run-crocodile DATASET=mv MAX_ROWS=5
make run-alligator DATASET=mv MAX_ROWS=5
make run-editsim DATASET=mv MAX_ROWS=5 NIL_THRESHOLD=0.2
```

Outputs follow:

```
outputs/{dataset}/{method}/{settings_hash}/predictions.csv
outputs/{dataset}/{method}/{settings_hash}/report.json
```

## Evaluate

```bash
make eval PRED=outputs/mv/editsim/<hash>/predictions.csv GT=data/mv/gt.csv
```
