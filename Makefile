PYTHON ?= python3
PYTHONPATH ?= .
DATASET ?= mv
MAX_ROWS ?= 0
ifneq ($(strip $(MAXROWS)),)
MAX_ROWS := $(MAXROWS)
endif
OUTPUT_DIR ?= outputs
MODEL ?= gpt-oss-120b
PROVIDER ?= openrouter
PROMPT_SCHEMA ?= base
NIL_THRESHOLD ?= 0.0
FORCE_GT ?= 0
FORCE_ID ?=
PRED ?=
GT ?=

.PHONY: build-datasets run-llm run-crocodile run-alligator run-editsim eval

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
		--prompt-schema $(PROMPT_SCHEMA)

run-crocodile:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m experiments.runners.run_crocodile \
		--dataset $(DATASET) \
		--max-rows $(MAX_ROWS) \
		--output-dir $(OUTPUT_DIR)

run-alligator:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m experiments.runners.run_alligator \
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
