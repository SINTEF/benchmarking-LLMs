"""Tests for experiments.runs (parallel executor, run store, ETA) and LamAPI id forcing."""

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import httpx
import pandas as pd

from experiments.cost.tasks import DatasetSource
from experiments.forcing import load_gt_ids
from experiments.runs.config import RunConfig
from experiments.runs.eta import estimate_duration, live_eta
from experiments.runs.executor import RunExecutor, extract_answer
from experiments.runs.store import CANCELLED, DONE, FAILED, INTERRUPTED, QUEUED, RUNNING, RunStore

FIXTURES = Path(__file__).parent / "fixtures"


def _source(gt_path=FIXTURES / "mv_gt.csv") -> DatasetSource:
    return DatasetSource(
        key="spec:test", name="test", kind="spec", description="", target_kg="wikidata",
        table_path=FIXTURES / "mv_table.csv", mention_columns=("Title",), gt_path=gt_path,
    )


class FakeLamapi:
    """Returns two fixed candidates and appends the forced ids, like LamAPI does."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def is_cached(self, name, ids=None, target_kg=None) -> bool:
        return False

    def fetch_entities(self, name, ids=None, target_kg=None):
        self.calls.append((name, ids))
        cands = [{"id": "Q1", "label": "One", "description": None, "types": []},
                 {"id": "Q2", "label": "Two", "description": None, "types": []}]
        for i in ids or []:
            cands.append({"id": i, "label": f"forced {i}", "description": None, "types": []})
        return cands


class FakeOpenRouter:
    """httpx transport: answers with the last forced id in the prompt, tracks peak concurrency."""

    def __init__(self, fail_first_with: int | None = None, always: int | None = None, delay: float = 0.02,
                 shared_first: int = 0):
        self.in_flight = self.peak = self.requests = 0
        self.bodies: list[dict] = []
        self.shared_first = shared_first  # first N answers come from shared capacity (is_byok false)
        self.fail_first_with = fail_first_with
        self.always = always
        self.delay = delay

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        if self.always:
            return httpx.Response(self.always, json={"error": {"message": "nope"}})
        if self.fail_first_with and self.requests == 1:
            return httpx.Response(self.fail_first_with, headers={"retry-after": "0.05"}, json={})
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(self.delay)
        self.in_flight -= 1
        body = json.loads(request.content)
        self.bodies.append(body)
        prompt = body["messages"][0]["content"]
        answer = "A table about movies." if "summary" in prompt.lower() and "ANSWER" not in prompt else "ANSWER:Q2"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": answer}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5, "cost": 0.0001,
                      "is_byok": len(self.bodies) > self.shared_first},
            "provider": "Fake",
        })

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


class TestLamapiIds(unittest.TestCase):
    def _client(self, tmp):
        from experiments.retrieval.lamapi_client import LamapiClient

        return LamapiClient(endpoint="http://lamapi.test/lookup/entity-retrieval", num_candidates=5, cache_dir=tmp)

    def test_ids_are_space_separated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client(tmp)
            resp = mock.Mock(status_code=200)
            resp.json.return_value = []
            with mock.patch.object(client.session, "get", return_value=resp) as get:
                client.fetch_entities("Batman Begins", ids=["Q2", "Q1"])
                client.fetch_entities("Oslo", ids="Q3,Q4 Q5")
            self.assertEqual(get.call_args_list[0].kwargs["params"]["ids"], "Q1 Q2")
            self.assertEqual(get.call_args_list[1].kwargs["params"]["ids"], "Q3 Q4 Q5")

    def test_cache_keys_kept_for_single_id(self) -> None:
        from experiments.retrieval.lamapi_client import _cache_key

        with tempfile.TemporaryDirectory() as tmp:
            client = self._client(tmp)
            old = {"endpoint": client.endpoint, "name": "x", "limit": 5, "ids": ["Q1"], "target_kg": None}
            self.assertEqual(_cache_key(client._request_payload("x", 5, ["Q1"], None)), _cache_key(old))
            old_multi = {**old, "ids": ["Q1", "Q2"]}  # cached while ids were comma-joined: must not be reused
            self.assertNotEqual(_cache_key(client._request_payload("x", 5, ["Q1", "Q2"], None)), _cache_key(old_multi))


class TestGtIds(unittest.TestCase):
    def test_spec(self) -> None:
        gt = load_gt_ids(_source())
        self.assertTrue(gt)
        self.assertTrue(all(k[0] == "__spec__" and v for k, v in gt.items()))

    def test_semtab_multiple_uris(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cea = Path(tmp) / "cea.csv"
            cea.write_text(
                "T1,1,0,http://www.wikidata.org/entity/Q1 http://www.wikidata.org/entity/Q9\n"
                "T1,2,0,http://www.wikidata.org/entity/Q2\n"
            )
            src = DatasetSource(key="semtab:x", name="x", kind="semtab", description="", target_kg="wikidata",
                                tables_dir=Path(tmp), cea_path=cea)
            self.assertEqual(load_gt_ids(src), {("T1", 0, 0): ["Q1", "Q9"], ("T1", 1, 0): ["Q2"]})


class TestExecutor(unittest.TestCase):
    def _run(self, tmp, fake: FakeOpenRouter, store=None, run_id=None, **cfg):
        cfg.setdefault("provider_policy", "routing")
        config = RunConfig(dataset="spec:test", model="m/x", limit=6, output_dir=tmp, max_concurrency=4, **cfg)
        lamapi = FakeLamapi()
        ex = RunExecutor(config, store, run_id, api_key="k", lamapi_client=lamapi,
                         llm_transport=fake.transport(), source=_source())
        return ex, ex.run(), lamapi

    def test_parallel_run_with_gt_forcing_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeOpenRouter(fail_first_with=429)
            ex, manifest, lamapi = self._run(tmp, fake)
            p = manifest["progress"]
            self.assertEqual(manifest["status"], DONE)
            self.assertEqual(p["n_tasks"], 5)
            self.assertEqual(p["n_errors"], 0)
            self.assertEqual(p["rate_limited"], 1)
            self.assertGreater(fake.peak, 1, "requests should run in parallel")
            self.assertEqual(p["gt_present"], p["gt_cells"])
            gt = load_gt_ids(_source())
            forced = {name: ids for name, ids in lamapi.calls}
            table = pd.read_csv(FIXTURES / "mv_table.csv")
            for row in range(5):
                if ("__spec__", row, 0) in gt:
                    self.assertEqual(forced[table.iloc[row, 0]], gt[("__spec__", row, 0)])
            preds = pd.read_csv(ex.run_dir / "predictions.csv")
            self.assertEqual(len(preds), 5)
            self.assertTrue((preds["pred_id"] == "Q2").all())
            self.assertAlmostEqual(p["cost_usd"], 0.0001 * fake.requests - 0.0001, places=6)  # 429 is free

            # Second run: everything is resumed, nothing is sent.
            fake2 = FakeOpenRouter()
            _, manifest2, _ = self._run(tmp, fake2)
            self.assertEqual(fake2.requests, 0)
            self.assertEqual(manifest2["progress"]["n_resumed"], 5)

    def test_byok_only_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"OPENROUTER_BYOK_PROVIDERS": "cerebras"}), \
                mock.patch("experiments.cost.pricing.load_endpoints", return_value=None):
            fake = FakeOpenRouter()
            _, manifest, _ = self._run(tmp, fake, provider_policy="byok")
            self.assertTrue(all(b["provider"] == {"order": ["cerebras"], "allow_fallbacks": False} for b in fake.bodies))
            p = manifest["progress"]
            self.assertEqual(p["byok_calls"], fake.requests)
            self.assertEqual(p["shared_calls"], 0)
            self.assertIn("BYOK only", p["provider_route"])

    def test_byok_only_rejects_shared_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"OPENROUTER_BYOK_PROVIDERS": "cerebras"}), \
                mock.patch("experiments.cost.pricing.load_endpoints", return_value=None), \
                mock.patch("experiments.runs.llm._backoff", return_value=0.01):
            fake = FakeOpenRouter(shared_first=2)
            _, manifest, _ = self._run(tmp, fake, provider_policy="byok")
            p = manifest["progress"]
            self.assertEqual(p["shared_rejected"], 2)
            self.assertEqual(p["shared_calls"], 0)  # nothing accepted from shared capacity
            self.assertEqual(p["n_ok"], 5)
            self.assertAlmostEqual(p["cost_usd"], 0.0001 * fake.requests)  # rejected answers were paid too

    def test_byok_only_without_byok_provider_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"OPENROUTER_BYOK_PROVIDERS": ""}), \
                mock.patch("experiments.cost.pricing.load_endpoints", return_value=None):
            fake = FakeOpenRouter()
            _, manifest, _ = self._run(tmp, fake, provider_policy="byok")
            self.assertEqual(manifest["status"], FAILED)
            self.assertIn("BYOK only", manifest["error"])
            self.assertEqual(fake.requests, 0)

    def test_bad_key_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs.db")
            run_id = store.submit({}, "x")
            _, manifest, _ = self._run(tmp, FakeOpenRouter(always=401), store, run_id)
            self.assertEqual(manifest["status"], FAILED)
            self.assertIn("401", manifest["error"])
            self.assertEqual(store.get(run_id).status, FAILED)

    def test_extract_answer(self) -> None:
        self.assertEqual(extract_answer("bla ANSWER: Q5 ... ANSWER:Q42"), "Q42")
        self.assertEqual(extract_answer(""), "No Identifier")


class TestStore(unittest.TestCase):
    def test_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs.db")
            a = store.submit({"dataset": "mv"}, "a")
            b = store.submit({"dataset": "mv"}, "b")
            self.assertEqual(store.claim_next("w").id, a)
            self.assertEqual(store.get(a).status, RUNNING)
            store.cancel(b)
            self.assertEqual(store.get(b).status, CANCELLED)
            self.assertIsNone(store.claim_next("w"))
            store.update_progress(a, {"n_done": 3})
            self.assertEqual(len(store.ticks(a)), 1)
            with store._conn() as conn:
                conn.execute("UPDATE runs SET heartbeat = 0 WHERE id = ?", (a,))
            self.assertEqual(store.mark_stale(), 1)
            self.assertEqual(store.get(a).status, INTERRUPTED)
            store.requeue(a)
            self.assertEqual(store.get(a).status, QUEUED)


class TestEta(unittest.TestCase):
    def test_bottlenecks(self) -> None:
        # 16 parallel / 2 s = 8 calls/s -> 1000 calls in 125 s (+ one call latency)
        e = estimate_duration(1000, seconds_per_call=2.0, max_concurrency=16)
        self.assertEqual(e.bottleneck, "parallelism")
        self.assertAlmostEqual(e.seconds, 127.0)
        e = estimate_duration(1000, seconds_per_call=2.0, max_concurrency=16, max_rpm=120)
        self.assertEqual(e.bottleneck, "rpm cap")
        self.assertAlmostEqual(e.seconds, 502.0)
        e = estimate_duration(1000, 2.0, 16, lookups_to_fetch=10000, seconds_per_lookup=1.0, lamapi_concurrency=4)
        self.assertEqual(e.bottleneck, "LamAPI")

    def test_live(self) -> None:
        now = time.time()
        ticks = [{"t": now - 60, "done": 100}, {"t": now, "done": 400}]
        eta = live_eta({"n_tasks": 1000, "n_done": 400, "n_ok": 400, "cost_usd": 0.4}, ticks)
        self.assertAlmostEqual(eta["rate_per_s"], 5.0)
        self.assertAlmostEqual(eta["eta_s"], 120.0)
        self.assertAlmostEqual(eta["projected_cost_usd"], 1.0)


if __name__ == "__main__":
    unittest.main()


class TestMetrics(unittest.TestCase):
    def test_prf_with_nil_and_alternatives(self) -> None:
        from experiments.evaluation.metrics import evaluate

        gt = pd.DataFrame({
            "key": ["a", "b", "c", "d", "e", "e"],
            "gt_id": ["Q1", "Q2", "NIL", "Q4", "Q5", "Q55"],  # e: two valid ids
        })
        pred = pd.DataFrame({
            "key": ["a", "b", "c", "d", "e", "z"],  # z has no GT row: ignored
            "pred_id": ["Q1", "Q9", "NIL", "No Identifier", "Q55", "Q7"],
        })
        r = evaluate(pred, gt)
        # entity predictions: a, b, e (d's "No Identifier" counts as NIL) -> 2 correct (a, e)
        self.assertEqual((r["tp"], r["predicted_entities"], r["gold_entities"]), (2, 3, 4))
        self.assertAlmostEqual(r["precision"], 2 / 3)
        self.assertAlmostEqual(r["recall"], 2 / 4)
        self.assertAlmostEqual(r["f1"], 2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
        self.assertAlmostEqual(r["accuracy"], 3 / 5)  # a, c (NIL), e
        self.assertEqual((r["nil_correct"], r["nil_pred_count"], r["nil_gt_count"]), (1, 2, 1))

    def test_run_metrics_partial(self) -> None:
        from experiments.runs.metrics import run_metrics

        with tempfile.TemporaryDirectory() as tmp:
            gt = load_gt_ids(_source())
            first = sorted(gt)[:2]
            lines = [
                {"key": f"{r}-{c}", "pred_id": gt[(t, r, c)][0], "status": "ok"} for t, r, c in first
            ]
            lines[1]["pred_id"] = "NIL"  # an errored cell counts as no prediction
            Path(tmp, "results.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
            r = run_metrics(tmp, _source())
            self.assertEqual(r["evaluated"], 2)  # only cells finished so far
            self.assertEqual((r["precision"], r["recall"]), (1.0, 0.5))


class TestByok(unittest.TestCase):
    def test_preferences(self) -> None:
        from experiments.cost.pricing import ProviderList, parse_endpoints
        from experiments.runs.byok import provider_preferences

        eps = ProviderList("m", parse_endpoints(json.loads((FIXTURES / "openrouter_endpoints_gpt_oss_120b.json").read_text())), "cache")
        prefs, _ = provider_preferences("byok", None, eps, {"cerebras", "openai"})
        self.assertEqual(prefs, {"order": ["cerebras"], "allow_fallbacks": False})  # openai does not serve it
        prefs, _ = provider_preferences("byok_first", None, eps, {"cerebras"})
        self.assertTrue(prefs["allow_fallbacks"])
        prefs, desc = provider_preferences("byok", None, eps, {"openai"})
        self.assertIsNone(prefs)
        self.assertIn("routing", desc)
        self.assertEqual(provider_preferences("pin", "groq", eps, set())[0]["order"], ["groq"])

    def test_old_configs_keep_routing_and_folder(self) -> None:
        old = {"dataset": "mv", "model": "openai/gpt-oss-120b"}
        cfg = RunConfig.from_dict(old)
        self.assertEqual(cfg.provider_policy, "routing")
        self.assertNotEqual(cfg.run_dir(_source()), RunConfig(**old).run_dir(_source()))  # new default: byok


class TestDatasetsRoot(unittest.TestCase):
    def test_problem_points_to_parent_with_datasets(self) -> None:
        from experiments.cost.tasks import datasets_root_problem

        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp, "Round9")
            (ds / "tables").mkdir(parents=True)
            (ds / "gt").mkdir()
            (ds / "gt" / "cea.csv").write_text("T1,1,0,http://www.wikidata.org/entity/Q1\n")
            Path(ds / "tables" / "T1.csv").write_text("name\nRome\n")
            empty = Path(tmp, "entity-linking")
            empty.mkdir()
            self.assertIsNone(datasets_root_problem(tmp))
            self.assertIn("parent folder", datasets_root_problem(empty))
            with mock.patch.dict("os.environ", {"HOME": tmp}):
                self.assertIsNone(datasets_root_problem("~"))  # "~" is expanded


class TestLinkers(unittest.TestCase):
    def test_store_claims_by_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs.db")
            old = store.submit({"dataset": "mv", "model": "m"}, "old llm run")  # no method key: LLM
            croc = store.submit({"dataset": "mv", "method": "crocodile"}, "croc")
            self.assertEqual(store.claim_next("w", ("crocodile", "alligator")).id, croc)
            self.assertIsNone(store.claim_next("w", ("crocodile", "alligator")))
            self.assertEqual(store.claim_next("w", ("llm",)).id, old)

    def test_config_folders(self) -> None:
        llm = RunConfig(dataset="mv", model="m")
        croc = RunConfig(dataset="mv", method="crocodile")
        self.assertEqual(llm.run_dir(_source()).parent.name, "llm")
        self.assertEqual(croc.run_dir(_source()).parent.name, "crocodile")
        self.assertEqual(croc.label(), "mv · Crocodile + BoW")
        # LLM-only settings do not change a linker run's folder
        self.assertEqual(croc.run_dir(_source()), RunConfig(dataset="mv", method="crocodile", schema="fewshot").run_dir(_source()))
        with self.assertRaises(ValueError):
            RunConfig(dataset="mv")  # LLM runs need a model

    def test_pipeline_records_and_scores(self) -> None:
        from experiments.runs.linkers import LinkerExecutor

        gt = load_gt_ids(_source())
        calls = []

        def fake_link_table(self, taskset, table_id, cells, gt_ids):
            calls.append([self._forced(t, gt_ids) for t in cells])
            df = taskset.load_table(table_id)
            preds = {"__mentions__": {(int(t.row), int(t.col)): str(df.iloc[int(t.row), int(t.col)]) for t in cells}}
            for t in cells:  # right on every cell but the first
                gid = gt_ids[(t.table_id, int(t.row), int(t.col))][0]
                preds[(int(t.row), int(t.col))] = {"id": "Q0" if int(t.row) == 0 else gid, "score": 0.9,
                                                   "candidates": [{"id": gid}]}
            return preds

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(LinkerExecutor, "_link_table", fake_link_table):
            cfg = RunConfig(dataset="spec:test", method="crocodile", limit=5, output_dir=tmp)
            manifest = LinkerExecutor(cfg, source=_source()).run()
            p = manifest["progress"]
            self.assertEqual((p["n_ok"], p["tables_done"], p["gt_present"]), (5, 1, 5))
            self.assertEqual(calls[0][0], gt[("__spec__", 0, 0)])  # GT ids forced
            self.assertAlmostEqual(manifest["report"]["precision"], 4 / 5)
            self.assertTrue(Path(tmp, "test", "crocodile").is_dir())


class TestCrocodileBow(unittest.TestCase):
    def test_bow_setting(self) -> None:
        from experiments.runs.linkers import LinkerExecutor

        old = {"dataset": "mv", "method": "crocodile"}  # queued before the setting existed: ran without BoW
        legacy = RunConfig.from_dict(old)
        self.assertFalse(legacy.bow)
        on = RunConfig(dataset="mv", method="crocodile")
        self.assertTrue(on.bow)
        self.assertNotEqual(on.run_dir(_source()), legacy.run_dir(_source()))
        self.assertEqual(on.label(), "mv · Crocodile + BoW")
        gator = RunConfig(dataset="mv", method="alligator")
        self.assertEqual(gator.run_dir(_source()), RunConfig(dataset="mv", method="alligator", bow=False).run_dir(_source()))

        with mock.patch.dict("os.environ", {"ENTITY_BOW_ENDPOINT": "https://lamapi.test/entity/bow"}):
            ex = LinkerExecutor(on, source=_source())
            self.assertEqual(ex._bow_endpoint("https://lamapi.test/lookup/entity-retrieval"), "https://lamapi.test/entity/bow")
            self.assertIsNone(LinkerExecutor(legacy, source=_source())._bow_endpoint("x"))
        with mock.patch.dict("os.environ", {"ENTITY_BOW_ENDPOINT": ""}):
            self.assertEqual(ex._bow_endpoint("https://h/lookup/entity-retrieval"), "https://h/entity/bow")


class TestPromptsContent(unittest.TestCase):
    def test_fewshot_has_real_examples(self) -> None:
        from experiments.cost.prompts import FEWSHOT_EXAMPLES_PATH, build_prompt

        df = pd.read_csv(FIXTURES / "mv_table.csv")
        prompt = build_prompt("fewshot", df, 0, 0, [], "A table of films.")
        examples = FEWSHOT_EXAMPLES_PATH.read_text()
        self.assertIn(examples, prompt)
        self.assertNotIn("Examples:\nN.A.", prompt)
        self.assertEqual(examples.count("ANSWER:"), 3)
        self.assertNotIn(examples, build_prompt("base", df, 0, 0, [], None))
        # few-shot runs (and only they) depend on the examples file
        a = RunConfig(dataset="mv", model="m", schema="fewshot").run_dir(_source())
        with mock.patch("experiments.cost.prompts.fewshot_examples_version", return_value="other"):
            self.assertNotEqual(a, RunConfig(dataset="mv", model="m", schema="fewshot").run_dir(_source()))
            base = RunConfig(dataset="mv", model="m").run_dir(_source())
        self.assertEqual(base, RunConfig(dataset="mv", model="m").run_dir(_source()))

    def test_empty_summary_is_retried_and_prompts_saved(self) -> None:
        class EmptyFirstSummary(FakeOpenRouter):
            summaries = 0

            async def handler(self, request):
                body = json.loads(request.content)
                if "summary" in body["messages"][0]["content"].lower() and "ANSWER" not in body["messages"][0]["content"]:
                    self.summaries += 1
                    if self.summaries == 1:  # a reasoning model that answered nothing
                        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}],
                                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
                return await super().handler(request)

        with tempfile.TemporaryDirectory() as tmp:
            fake = EmptyFirstSummary()
            ex, manifest, _ = TestExecutor()._run(tmp, fake)
            self.assertEqual(fake.summaries, 2)
            self.assertEqual(manifest["progress"]["summary_errors"], 0)
            samples = [json.loads(l) for l in (ex.run_dir / "sample_prompts.jsonl").read_text().splitlines()]
            self.assertEqual(len(samples), 5)
            self.assertIn("A table about movies.", samples[0]["prompt"])  # the real summary, not N.A.


class TestDeleteRuns(unittest.TestCase):
    def _store_with_runs(self, tmp):
        store = RunStore(Path(tmp) / "runs.db")
        out = Path(tmp) / "outputs"

        def finished(run_dir: Path, status=DONE):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "results.jsonl").write_text("{}\n")
            rid = store.submit({"dataset": "mv", "model": "m", "output_dir": str(out)}, "r")
            store.update_progress(rid, {"n_done": 1}, run_dir=str(run_dir))
            store.finish(rid, status)
            return rid

        return store, out, finished

    def test_delete_history_and_outputs(self) -> None:
        from experiments.runs.manage import delete_runs

        with tempfile.TemporaryDirectory() as tmp:
            store, out, finished = self._store_with_runs(tmp)
            a = finished(out / "mv" / "llm" / "aaa")
            shared1 = finished(out / "mv" / "llm" / "shared")
            shared2 = finished(out / "mv" / "llm" / "shared")  # same settings -> same folder
            active = store.submit({"dataset": "mv", "model": "m"}, "queued")

            rep = delete_runs(store, [a, shared1, active], outputs=True)
            self.assertEqual(sorted(rep.deleted), sorted([a, shared1]))
            self.assertEqual(rep.skipped_active, [active])
            self.assertFalse((out / "mv" / "llm" / "aaa").exists())
            self.assertTrue((out / "mv" / "llm" / "shared").exists())  # still used by shared2
            self.assertIn(shared2, next(iter(rep.folders_kept.values())))
            self.assertIsNone(store.get(a))
            self.assertEqual(store.ticks(a), [])
            self.assertIsNotNone(store.get(active))

            rep = delete_runs(store, [shared2])  # history only
            self.assertEqual(rep.folders_removed, [])
            self.assertTrue((out / "mv" / "llm" / "shared").exists())

            last = finished(out / "mv" / "llm" / "shared")  # same folder, now used by this run only
            delete_runs(store, [last], outputs=True)
            self.assertFalse((out / "mv").exists())  # emptied dataset/method folders are pruned
            self.assertTrue(out.exists())

    def test_never_removes_outside_outputs(self) -> None:
        from experiments.runs.manage import delete_runs

        with tempfile.TemporaryDirectory() as tmp:
            store, out, finished = self._store_with_runs(tmp)
            elsewhere = Path(tmp) / "precious"
            rid = finished(elsewhere)
            rep = delete_runs(store, [rid], outputs=True)
            self.assertTrue(elsewhere.exists())
            self.assertIn("not inside the outputs folder", rep.folders_kept[str(elsewhere)])

    def test_container_paths(self) -> None:
        from experiments.runs.manage import local_run_dir
        from experiments.runs.store import Run

        run = Run("x", 0, 0, DONE, None, {}, "/app/outputs/mv/llm/abc", None, None, None, None, None)
        from experiments.cost.tasks import REPO_ROOT

        self.assertEqual(local_run_dir(run), REPO_ROOT / "outputs" / "mv" / "llm" / "abc")


class TestDurationFromHistory(unittest.TestCase):
    def _run(self, store, dataset, model, cells, seconds, conc=16, schema="base"):
        rid = store.submit({"dataset": dataset, "model": model, "schema": schema, "max_concurrency": conc}, "r")
        now = time.time()
        store.update_progress(rid, {"n_done": cells, "n_tasks": cells, "session_started_at": now - seconds,
                                    "session_done_at_start": 0, "latency_p50_s": 0.5})
        store.finish(rid, DONE)
        with store._conn() as conn:  # finished exactly ``seconds`` after the start
            conn.execute("UPDATE runs SET started_at = ?, finished_at = ? WHERE id = ?", (now - seconds, now, rid))
        return rid

    def test_same_dataset_wins_and_uses_wall_clock(self) -> None:
        from experiments.runs.eta import seconds_per_call_for

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs.db")
            self._run(store, "spec:mv", "m", cells=1000, seconds=180)  # 16*180/1000 = 2.88 s/call
            self._run(store, "semtab:2t-red", "m", cells=2400, seconds=900)  # 16*900/2400 = 6.0 s/call
            spc, src = seconds_per_call_for(store, "m", None, True, dataset="2t-red", schema="base")
            self.assertAlmostEqual(spc, 6.0, places=1)  # not the 0.5 s median latency
            self.assertIn("this dataset", src)
            spc, src = seconds_per_call_for(store, "m", None, True, dataset="mv")  # "mv" == "spec:mv"
            self.assertAlmostEqual(spc, 2.88, places=1)
            # calibration beats other datasets, but not this dataset's own runs
            spc, src = seconds_per_call_for(store, "m", None, True, calibrated=1.0, dataset="semtab:Round4_2020")
            self.assertEqual((spc, "calibration" in src), (1.0, True))
            spc, src = seconds_per_call_for(store, "m", None, True, dataset="semtab:Round4_2020")
            self.assertIn("other data", src)
            self.assertAlmostEqual(spc, (2.88 * 1000 + 6.0 * 2400) / 3400, places=1)

    def test_live_eta_range(self) -> None:
        now = time.time()
        # fast at first, slow in the last 5 minutes
        ticks = [{"t": now - 300, "done": 2000}, {"t": now, "done": 2500}]  # 100/min recently
        p = {"n_tasks": 4500, "n_done": 2500, "session_started_at": now - 1000, "session_done_at_start": 0}  # 150/min avg
        eta = live_eta(p, ticks)
        self.assertAlmostEqual(eta["eta_s"], 2000 / (500 / 300), delta=1)
        self.assertLess(eta["eta_low_s"], eta["eta_high_s"])
        self.assertAlmostEqual(eta["eta_low_s"], 2000 / (2500 / 1000), delta=5)
        short = live_eta(p, [{"t": now - 20, "done": 2490}, {"t": now, "done": 2500}])  # <1 min of ticks
        self.assertAlmostEqual(short["rate_per_s"], 2.5, places=2)  # falls back to the session average
