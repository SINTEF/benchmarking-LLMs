"""What a run is: one dataset x prompt schema x model (x provider), plus how fast to send it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from experiments.common.hashing import stable_hash
from experiments.cost.tasks import REPO_ROOT, DatasetSource, discover_sources

SCHEMAS = ("base", "structured", "fewshot")

# Fields that do not change the results: excluded from the run-directory hash so a resumed or
# re-tuned run (other concurrency) keeps writing to the same folder.
SPEED_FIELDS = {"max_concurrency", "max_rpm", "lamapi_concurrency", "max_retries", "output_dir", "datasets_root"}


@dataclass
class RunConfig:
    dataset: str  # DatasetSource key ("spec:mv", "semtab:HardTablesR2") or a spec name ("mv")
    model: str  # OpenRouter model id
    schema: str = "base"
    provider: str | None = None  # OpenRouter endpoint tag to pin (e.g. "groq"); None = OpenRouter routing
    limit: int = 0  # first N rows (spec) / cells (SemTab); 0 = all
    lamapi_limit: int = 10
    force_gt: bool = True  # put every GT id of the cell into its candidate list
    force_ids: list[str] = field(default_factory=list)  # extra ids forced into every candidate list
    max_concurrency: int = 16  # parallel LLM requests
    max_rpm: int = 0  # requests per minute cap; 0 = none
    lamapi_concurrency: int = 8
    max_retries: int = 6
    seed: int = 13
    output_dir: str = "outputs"
    datasets_root: str | None = None

    def __post_init__(self) -> None:
        if self.schema not in SCHEMAS:
            raise ValueError(f"Unknown prompt schema '{self.schema}'. Use one of {SCHEMAS}.")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        known = {f.name for f in fields(cls)}
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
        settings = {k: v for k, v in self.to_dict().items() if k not in SPEED_FIELDS}
        settings["dataset"] = source.key
        base = Path(self.output_dir)
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base / source.name / "llm" / stable_hash(settings)

    def label(self) -> str:
        name = self.dataset.split(":", 1)[-1]
        return f"{name} · {self.schema} · {self.model}" + (f" @ {self.provider}" if self.provider else "")
