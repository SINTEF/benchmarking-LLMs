PYTHON ?= python3
PYTHONPATH ?= .
DATASET ?= mv
MAX_ROWS ?= 0
ifneq ($(strip $(MAXROWS)),)
MAX_ROWS := $(MAXROWS)
endif
OUTPUT_DIR ?= outputs
MODEL ?= openai/gpt-oss-120b
PROVIDER ?= openrouter
ENDPOINT ?=
PROMPT_SCHEMA ?= base
CONCURRENCY ?= 16
MAX_RPM ?= 0
LAMAPI_CONCURRENCY ?= 16
NIL_THRESHOLD ?= 0.0
FORCE_GT ?= 0
FORCE_ID ?=
PRED ?=
GT ?=

.PHONY: build-datasets run-llm run-crocodile run-alligator run-editsim eval ui worker runs estimate test up down logs mongo-debug mongo-debug-stop

build-datasets:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) datasets/builders/build_mv.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) datasets/builders/build_cp.py --include-crunchbase
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) datasets/builders/build_sn.py

run-llm:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m experiments.runners.run_llm \
		--dataset $(DATASET) \
		--max-rows $(MAX_ROWS) \
		--output-dir $(OUTPUT_DIR) \
		--model $(MODEL) \
		--provider $(PROVIDER) \
		--prompt-schema $(PROMPT_SCHEMA) \
		--max-concurrency $(CONCURRENCY) \
		--max-rpm $(MAX_RPM) \
		--lamapi-concurrency $(LAMAPI_CONCURRENCY) \
		$(if $(ENDPOINT),--endpoint $(ENDPOINT))

# Needs the linkers image (tensorflow 2.18, Python <= 3.12) and MongoDB: runs in docker compose.
run-crocodile:
	docker compose run --rm linkers python -m experiments.runners.run_crocodile \
		--dataset $(DATASET) \
		--max-rows $(MAX_ROWS) \
		--output-dir $(OUTPUT_DIR)

# Needs the linkers image (tensorflow 2.18, Python <= 3.12) and MongoDB: runs in docker compose.
run-alligator:
	docker compose run --rm linkers python -m experiments.runners.run_alligator \
		--dataset $(DATASET) \
		--max-rows $(MAX_ROWS) \
		--output-dir $(OUTPUT_DIR)

run-editsim:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m experiments.runners.run_editsim \
		--dataset $(DATASET) \
		--max-rows $(MAX_ROWS) \
		--output-dir $(OUTPUT_DIR) \
		--nil-threshold $(NIL_THRESHOLD) \
		$(if $(filter 1 true TRUE yes YES,$(FORCE_GT)),--force-gt-candidate,) \
		$(foreach id,$(FORCE_ID),--force-id $(id))

eval:
	@if [ -z "$(PRED)" ] || [ -z "$(GT)" ]; then \
		echo "Usage: make eval PRED=path/to/predictions.csv GT=path/to/gt.csv"; \
		exit 1; \
	fi
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m experiments.evaluation.evaluate --pred $(PRED) --gt $(GT)

ui:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m streamlit run ui/app.py

# Executes runs queued from the console; keep it running next to `make ui`.
worker:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m experiments.runs worker

runs:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m experiments.runs list

estimate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m experiments.cost.estimate \
		--dataset $(DATASET) \
		--model $(MODEL) \
		--schema $(PROMPT_SCHEMA) \
		--limit $(MAX_ROWS)

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

# Docker: console on http://localhost:$CONSOLE_PORT (default 8501), run workers, MongoDB (internal only)
up:
	docker compose up -d --build
	@port=$$(grep -s '^CONSOLE_PORT=' .env | cut -d= -f2); echo "Console: http://localhost:$${port:-8501}"

down:
	docker compose down

logs:
	docker compose logs -f ui worker

# Debugging only: MongoDB on 127.0.0.1:$MONGO_DEBUG_PORT (default 27018) without restarting anything.
mongo-debug:
	docker compose --profile debug up -d mongo-debug
	@port=$$(grep -s '^MONGO_DEBUG_PORT=' .env | cut -d= -f2); echo "MongoDB: mongodb://localhost:$${port:-27018} (stop with: make mongo-debug-stop)"

mongo-debug-stop:
	docker compose --profile debug rm -sf mongo-debug
