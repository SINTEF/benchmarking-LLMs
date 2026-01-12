import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestEditSimRunner(unittest.TestCase):
    def test_editsim_runner_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            cache_dir = Path("tests/fixtures/lamapi_cache")
            table_path = Path("tests/fixtures/mv_table.csv")
            gt_path = Path("tests/fixtures/mv_gt.csv")

            cmd = [
                sys.executable,
                "-m",
                "experiments.runners.run_editsim",
                "--dataset",
                "mv",
                "--table-path",
                str(table_path),
                "--gt-path",
                str(gt_path),
                "--max-rows",
                "5",
                "--output-dir",
                str(output_dir),
                "--lamapi-endpoint",
                "http://lamapi.test",
                "--lamapi-limit",
                "3",
                "--lamapi-cache-dir",
                str(cache_dir),
                "--lamapi-offline",
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(".").resolve())
            env["LAMAPI_OFFLINE"] = "1"

            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)

            preds = list(output_dir.glob("mv/editsim/*/predictions.csv"))
            self.assertEqual(len(preds), 1)
            pred_path = preds[0]

            eval_cmd = [
                sys.executable,
                "-m",
                "experiments.evaluation.evaluate",
                "--pred",
                str(pred_path),
                "--gt",
                str(gt_path),
            ]
            eval_result = subprocess.run(
                eval_cmd, env=env, check=True, capture_output=True, text=True
            )
            self.assertEqual(eval_result.returncode, 0)

            report_path = pred_path.parent / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
