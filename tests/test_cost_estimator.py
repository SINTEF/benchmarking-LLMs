"""Tests for experiments.cost (token & cost estimation)."""

import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd

from experiments.cost.estimate import ProfileConfig, price_profile, profile_prompts
from experiments.cost.pricing import ProviderList, manual_price, parse_endpoints, parse_models, with_provider
from experiments.cost.prompts import ANSWER_FORMAT, build_prompt
from experiments.cost.tasks import DatasetSource, load_taskset

FIXTURES = Path(__file__).parent / "fixtures"
CANDIDATES = [
    {"id": "Q1", "label": "Alpha", "description": "a film", "types": [{"id": "Q11424", "name": "film"}]},
    {"id": "Q2", "label": "Alpha 2", "description": None, "types": []},
]


def _stub_heavy_modules() -> None:
    """LionLinker imports torch/ollama/openai/tqdm at module level; stub any that are missing."""

    def stub(name: str, **attrs) -> None:
        try:
            __import__(name)
        except Exception:
            mod = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(mod, key, value)
            sys.modules[name] = mod

    class _Any:
        def __init__(self, *a, **k):
            pass

    stub("torch")
    stub("ollama", Client=_Any)
    stub("openai", OpenAI=_Any, api_key=None)
    stub("aiohttp", ClientSession=_Any, ClientResponseError=Exception, ClientConnectionError=Exception)
    try:
        import tqdm.asyncio  # noqa: F401
    except Exception:
        class _Bar:
            def __init__(self, *a, **k):
                self.n = self.total = 0

            def update(self, *_):
                pass

            refresh = close = update

        sys.modules["tqdm"] = types.ModuleType("tqdm")
        sys.modules["tqdm.asyncio"] = types.ModuleType("tqdm.asyncio")
        sys.modules["tqdm.asyncio"].tqdm = _Bar


def _spec_source(table_path: Path, mention_columns=("Title",)) -> DatasetSource:
    return DatasetSource(
        key="spec:test",
        name="test",
        kind="spec",
        description="",
        target_kg="wikidata",
        table_path=table_path,
        mention_columns=tuple(mention_columns),
    )


class TestPromptParity(unittest.TestCase):
    """The estimator must rebuild byte-identical prompts to what LionLinker sends."""

    @classmethod
    def setUpClass(cls) -> None:
        _stub_heavy_modules()
        try:
            from lion_linker.lion_linker import LionLinker
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"lion_linker not importable: {exc}")
        cls.LionLinker = LionLinker

    def test_answer_format_in_sync(self) -> None:
        self.assertEqual(ANSWER_FORMAT, " ".join(self.LionLinker.DEFAULT_ANSWER_FORMAT.split()))

    def test_prompts_identical(self) -> None:
        class FakeRetriever:
            async def fetch_multiple_entities(self, mentions, **kwargs):
                from experiments.cost.prompts import to_lionlinker_candidates

                return {m: to_lionlinker_candidates(CANDIDATES) for m in mentions}

        with tempfile.TemporaryDirectory() as tmp:
            # 150 rows -> crosses LionLinker's 64-row chunk boundaries; includes a NaN cell.
            df = pd.DataFrame(
                {
                    "Title": [f"Movie {i}" if i != 70 else None for i in range(150)],
                    "Year": list(range(1900, 2050)),
                    "Director": [f"Person {i % 7}" for i in range(150)],
                }
            )
            csv = Path(tmp) / "table.csv"
            df.to_csv(csv, index=False)

            for schema, template in [("base", "base"), ("structured", "detailed"), ("fewshot", "few_shot")]:
                sent: list[str] = []

                def fake_chat(prompt, *a, **k):
                    if prompt.startswith("Provide a high-level summary"):
                        return "A table of movies."
                    sent.append(prompt)
                    return "ANSWER:NIL"

                linker = self.LionLinker(
                    input_csv=str(csv),
                    model_name="test/model",
                    retriever=FakeRetriever(),
                    output_csv=str(Path(tmp) / f"out_{schema}.csv"),
                    mention_columns=["Title", "Director"],
                    model_api_provider="openrouter",
                    model_api_key="x",
                    prompt_template=template,
                )
                linker.llm_interaction.chat = fake_chat
                asyncio.run(linker.run())

                src = _spec_source(csv, ("Title", "Director"))
                ts = load_taskset(src)
                table = ts.load_table("__spec__")
                expected = []
                for t in ts.tasks.itertuples(index=False):
                    mention = table.iloc[t.row, t.col]
                    cands = [] if pd.isna(mention) else CANDIDATES
                    expected.append(build_prompt(schema, table, t.row, t.col, cands, "A table of movies."))
                self.assertEqual(len(sent), len(expected))
                for i, (got, exp) in enumerate(zip(sent, expected)):
                    self.assertEqual(got, exp, f"{schema}: prompt {i} differs")


class TestTasksAndProfile(unittest.TestCase):
    def test_semtab_taskset_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "MySemTab"
            (root / "tables").mkdir(parents=True)
            (root / "gt").mkdir()
            pd.DataFrame({"col0": ["Rome", "Oslo", "Lima"], "col1": [1, 2, 3]}).to_csv(
                root / "tables" / "T1.csv", index=False
            )
            pd.DataFrame({"col0": ["Paris"]}).to_csv(root / "tables" / "T2.csv", index=False)
            (root / "gt" / "cea_gt.csv").write_text(
                "T1,1,0,http://www.wikidata.org/entity/Q220\n"
                "T1,3,0,http://www.wikidata.org/entity/Q2868\n"
                "T1,3,0,http://www.wikidata.org/entity/Q2868\n"  # duplicate
                "T2,1,0,http://www.wikidata.org/entity/Q90\n"
            )
            from experiments.cost.tasks import discover_sources

            src = next(s for s in discover_sources(Path(tmp)) if s.kind == "semtab")
            ts = load_taskset(src)
            self.assertEqual(ts.n_tasks, 3)
            self.assertEqual(ts.n_tables, 2)
            self.assertEqual(ts.load_table("T1").iloc[ts.tasks.iloc[1].row, 0], "Lima")

            prof = profile_prompts(ts, ProfileConfig(sample_size=10))
            self.assertTrue(prof.exact)
            self.assertEqual(prof.summary_calls, 2)
            self.assertEqual(prof.cache_misses, 3)  # no LamAPI client -> candidate tokens estimated
            self.assertGreater(prof.mean_input, 100)

    def test_cache_hits_from_fixture(self) -> None:
        from experiments.retrieval.lamapi_client import LamapiClient

        client = LamapiClient(
            endpoint="http://lamapi.test", num_candidates=3, cache_dir=FIXTURES / "lamapi_cache", offline=True
        )
        ts = load_taskset(_spec_source(FIXTURES / "mv_table.csv"))
        prof = profile_prompts(ts, ProfileConfig(lamapi_limit=3), (client, None))
        self.assertGreater(prof.cache_hits, 0)
        self.assertGreater(prof.filled_rate, 0)


class TestPricing(unittest.TestCase):
    def test_parse_and_price(self) -> None:
        payload = {
            "data": [
                {
                    "id": "openai/gpt-oss-120b",
                    "name": "gpt-oss-120b",
                    "pricing": {"prompt": "0.0000001", "completion": "0.0000005", "request": "0"},
                    "supported_parameters": ["reasoning"],
                },
                {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
            ]
        }
        models = parse_models(payload)
        self.assertTrue(models["openai/gpt-oss-120b"].reasoning)
        self.assertEqual(models["openrouter/auto"].input_per_token, 0.0)

        ts = load_taskset(_spec_source(FIXTURES / "mv_table.csv"))
        prof = profile_prompts(ts, ProfileConfig())
        price = manual_price("m", input_per_mtok=1.0, output_per_mtok=2.0, reasoning=False)
        row = price_profile(prof, price, output_tokens_per_call=10, summary_tokens=60)
        link_in = prof.input_tokens + ts.n_tasks * (60 - prof.na_summary_tokens)
        expected = (link_in + prof.summary_input_tokens) * 1e-6 + (ts.n_tasks * 10 + prof.summary_calls * 60) * 2e-6
        self.assertAlmostEqual(row["cost_usd"], expected)
        self.assertEqual(row["llm_calls"], ts.n_tasks + 1)
        self.assertFalse(row["calibrated"])


    def test_providers(self) -> None:
        import json

        payload = json.loads((FIXTURES / "openrouter_endpoints_gpt_oss_120b.json").read_text())
        providers = parse_endpoints(payload)
        self.assertEqual([p.tag for p in providers], ["near-ai/fp4", "google-vertex/global", "groq", "cerebras/fp16"])
        eps = ProviderList("openai/gpt-oss-120b", providers, "cache")
        # The degraded Google endpoint is cheaper but must not be picked as "cheapest".
        self.assertEqual(eps.cheapest(700, 10).tag, "near-ai/fp4")
        self.assertEqual(eps.cheapest(700, 10, healthy_only=False).tag, "google-vertex/global")
        self.assertEqual(eps.by_tag("Groq").tag, "groq")

        base = parse_models({"data": [{"id": "openai/gpt-oss-120b", "pricing": {"prompt": "1", "completion": "1"}}]})
        priced = with_provider(base["openai/gpt-oss-120b"], eps.by_tag("cerebras/fp16"))
        self.assertEqual(priced.provider, "cerebras/fp16")
        self.assertAlmostEqual(priced.input_per_mtok, 0.35)


class TestCalibration(unittest.TestCase):
    def test_calibrate_with_fake_openrouter(self) -> None:
        import experiments.cost.calibrate as cal_mod

        sent = []

        def fake_chat(model, content, provider, api_key, timeout=180):
            sent.append((model, provider, content))
            if content.startswith("Provide a high-level summary"):
                return "Movies with release dates.", {"prompt_tokens": 50, "completion_tokens": 120,
                                                      "completion_tokens_details": {"reasoning_tokens": 110}}
            return "ANSWER:Q1", {"prompt_tokens": 400, "completion_tokens": 250, "cost": 0.0001,
                                 "completion_tokens_details": {"reasoning_tokens": 245}}

        original = cal_mod.openrouter_chat
        cal_mod.openrouter_chat = fake_chat
        try:
            ts = load_taskset(_spec_source(FIXTURES / "mv_table.csv"))
            price = manual_price("openai/gpt-oss-120b", 0.1, 0.5, reasoning=True)
            cal = cal_mod.calibrate(ts, "base", price, (None, None), n_prompts=3, api_key="k")
        finally:
            cal_mod.openrouter_chat = original

        self.assertEqual(cal.n_summary_calls, 1)
        self.assertEqual(cal.n_link_calls, 3)
        self.assertEqual(cal.output_tokens_per_call, 250)
        self.assertEqual(cal.reasoning_tokens_per_call, 245)
        self.assertEqual(cal.summary_call_output_tokens, 120)
        # Linking prompts are fully filled: they carry the real summary from the first call.
        self.assertTrue(all("Movies with release dates." in c for _, _, c in sent[1:]))
        link_counted = [r.counted_prompt_tokens for r in cal.records if r.kind == "link"]
        self.assertAlmostEqual(cal.prompt_token_ratio, 400 * 3 / sum(link_counted))

        prof = profile_prompts(ts, ProfileConfig())
        row = price_profile(prof, price, output_tokens_per_call=400, calibration=cal)
        self.assertTrue(row["calibrated"])
        self.assertEqual(row["output_tokens"], round(ts.n_tasks * 250 + prof.summary_calls * 120))


if __name__ == "__main__":
    unittest.main()
