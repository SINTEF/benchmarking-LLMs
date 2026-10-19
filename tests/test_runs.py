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

    def __init__(self, fail_first_with: int | None = None, always: int | None = None, delay: float = 0.02):
        self.in_flight = self.peak = self.requests = 0
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
        prompt = json.loads(request.content)["messages"][0]["content"]
        answer = "A table about movies." if "summary" in prompt.lower() and "ANSWER" not in prompt else "ANSWER:Q2"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": answer}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5, "cost": 0.0001},
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
