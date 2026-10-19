"""What a run is: one dataset x prompt schema x model (x provider), plus how fast to send it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from experiments.common.hashing import stable_hash
from experiments.cost.tasks import REPO_ROOT, DatasetSource, discover_sources

SCHEMAS = ("base", "structured", "fewshot")
METHODS = {"llm": "LLM (OpenRouter)", "crocodile": "Crocodile", "alligator": "Alligator"}
LINKER_METHODS = ("crocodile", "alligator")  # classic linkers: own worker image (Dockerfile.linkers)
# Fields that only matter to LLM runs; left out of a classic linker run's folder hash and label.
LLM_FIELDS = {"model", "schema", "provider", "provider_policy", "max_concurrency", "max_rpm", "max_retries"}

# Fields that do not change the results: excluded from the run-directory hash so a resumed or
# re-tuned run (other concurrency) keeps writing to the same folder.
SPEED_FIELDS = {"max_concurrency", "max_rpm", "lamapi_concurrency", "max_retries", "output_dir", "datasets_root"}


@dataclass
class RunConfig:
    dataset: str  # DatasetSource key ("spec:mv", "semtab:HardTablesR2") or a spec name ("mv")
    model: str = ""  # OpenRouter model id (LLM runs)
    method: str = "llm"  # llm | crocodile | alligator
    schema: str = "base"
    provider: str | None = None  # OpenRouter endpoint tag, used when provider_policy == "pin"
    provider_policy: str = "byok"  # byok | byok_first | routing | pin (see experiments/runs/byok.py)
    limit: int = 0  # first N rows (spec) / cells (SemTab); 0 = all
    lamapi_limit: int = 10
    force_gt: bool = True  # put every GT id of the cell into its candidate list
    force_ids: list[str] = field(default_factory=list)  # extra ids forced into every candidate list
    max_concurrency: int = 16  # parallel LLM requests
    max_rpm: int = 0  # requests per minute cap; 0 = none
    lamapi_concurrency: int = 8
    linker_workers: int = 4  # Crocodile/Alligator worker processes per table
    max_retries: int = 6
    seed: int = 13
    output_dir: str = "outputs"
    datasets_root: str | None = None

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown method '{self.method}'. Use one of {list(METHODS)}.")
        if self.method == "llm" and not self.model:
            raise ValueError("LLM runs need a model.")
        if self.schema not in SCHEMAS:
            raise ValueError(f"Unknown prompt schema '{self.schema}'. Use one of {SCHEMAS}.")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.provider and self.provider_policy != "pin":
            self.provider_policy = "pin"
        if self.provider_policy not in ("byok", "byok_first", "routing", "pin"):
            raise ValueError(f"Unknown provider_policy '{self.provider_policy}'.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        known = {f.name for f in fields(cls)}
        data = dict(data)
        # Runs queued before provider policies existed used OpenRouter routing (or a pinned tag).
        data.setdefault("provider_policy", "pin" if data.get("provider") else "routing")
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_source(self) -> DatasetSource:
        root = Path(self.datasets_root).expanduser() if self.datasets_root else None
        sources = discover_sources(root)
        for s in sources:
            if self.dataset in (s.key, s.name):
                return s
        raise ValueError(f"Unknown dataset '{self.dataset}'. Available: {', '.join(s.key for s in sources)}")

    def run_dir(self, source: DatasetSource) -> Path:
        settings = {k: v for k, v in self.to_dict().items() if k not in SPEED_FIELDS | {"linker_workers"}}
        if settings["method"] == "llm":
            del settings["method"]  # same folder as runs made before other methods existed
            if settings["provider_policy"] in ("routing", "pin"):
                del settings["provider_policy"]  # same folder as runs made before policies existed
        else:
            settings = {k: v for k, v in settings.items() if k not in LLM_FIELDS}
        settings["dataset"] = source.key
        base = Path(self.output_dir)
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base / source.name / self.method / stable_hash(settings)

    def label(self) -> str:
        name = self.dataset.split(":", 1)[-1]
        if self.method != "llm":
            return f"{name} · {METHODS[self.method]}"
        where = {"pin": f" @ {self.provider}", "byok": " @ BYOK", "byok_first": " @ BYOK first"}.get(self.provider_policy, "")
        return f"{name} · {self.schema} · {self.model}{where}"
