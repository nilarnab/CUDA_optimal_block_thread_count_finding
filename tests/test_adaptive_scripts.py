import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).parents[1]


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class AdaptiveScriptsTest(unittest.TestCase):
    def test_train_log_lengths_times_reload_and_compare(self):
        import torch
        from data.one_d_adapter import prepare_one_d_dataset
        from test_one_d_adapter import fixture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _, _ = fixture(root)
            (run / "metadata.json").write_text(json.dumps({"device": {"gpu_name": "Test GPU", "compute_capability": "7.5"}}))
            processed = root / "processed"
            prepare_one_d_dataset(run, processed)
            groups = json.loads((processed / "split_groups.json").read_text())
            token_path, matrix_path = root / "tokens.json", root / "embeddings.pt"
            token_path.write_text(json.dumps({kernel: list(range(1, index + 3)) for index, kernel in enumerate(groups)}))
            torch.save(torch.randn(12, 5), matrix_path)

            def command(script, *args, succeeds=True):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                    cwd=ROOT, capture_output=True, text=True, timeout=90,
                    env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                )
                if succeeds:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    return json.loads(result.stdout)
                self.assertNotEqual(result.returncode, 0)
                return result.stderr

            inputs = ["--runtimes", processed / "runtimes.csv", "--token-ids", token_path]
            common = [*inputs, "--embedding-matrix", matrix_path, "--problem-columns", "m", "n", "k",
                      "--split-groups", processed / "split_groups.json", "--validation-groups", 1,
                      "--seed", 1, "--epochs", 2, "--batch-size", 2,
                      "--latency-warmup", 0, "--latency-repetitions", 1]
            trained = {}
            for architecture in ("baseline", "adaptive"):
                trained[architecture] = command(
                    "train_adaptive.py", *common, "--architecture", architecture,
                    "--checkpoint", root / f"{architecture}.pt", "--dataset-run-dir", run,
                    "--length-penalty", 0.01,
                )
                result = trained[architecture]
                self.assertEqual(result["timing_environment"]["same_gpu_model"], False)
                self.assertEqual(result["compression"]["kernel_count"], 4)
                self.assertGreater(result["validation"]["mean_dense_ms"], 0)
                with Path(result["cases_csv"]).open() as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    self.assertAlmostEqual(float(row["selected_kernel_ms"]), int(row["selected_threads"]) / 1000)
                    self.assertAlmostEqual(float(row["estimated_total_ms"]),
                                           float(row["predictor_ms"]) + float(row["selected_kernel_ms"]))
                    self.assertAlmostEqual(float(row["dense_plus_kernel_ms"]),
                                           float(row["dense_ms"]) + float(row["selected_kernel_ms"]))
                    self.assertLessEqual(int(row["learned_h"]), int(row["input_token_count"]))
                    self.assertGreaterEqual(int(row["learned_h"]), 1)
                with Path(result["lengths_csv"]).open() as handle:
                    lengths = list(csv.DictReader(handle))
                self.assertEqual({row["kernel_id"] for row in lengths}, set(groups))
                self.assertTrue({"m", "n", "k", "learned_h", "expected_h"}.issubset(lengths[0]))
            evaluated = command(
                "evaluate_adaptive.py", *inputs, "--checkpoint", root / "adaptive.pt",
                "--compare", root / "baseline.pt", "--output", root / "comparison.json",
                "--latency-warmup", 0, "--latency-repetitions", 1,
            )
            self.assertGreater(evaluated["speedup_comparison_over_model"]["estimated_total_ms"], 0)
            self.assertAlmostEqual(evaluated["model"]["summary"]["validation_task_loss"],
                                   trained["adaptive"]["validation"]["validation_task_loss"], places=6)
            final = json.loads(Path(trained["adaptive"]["report"]).read_text())
            self.assertEqual([(r["kernel_id"], r["learned_h"], r["selected_action"]) for r in final["cases"]],
                             [(r["kernel_id"], r["learned_h"], r["selected_action"]) for r in evaluated["model"]["cases"]])
            # Existing sequence-runner baseline checkpoints remain usable for comparison.
            command("train_sequence.py", *common, "--architecture", "baseline", "--checkpoint", root / "old.pt")
            command("evaluate_adaptive.py", *inputs, "--checkpoint", root / "adaptive.pt",
                    "--compare", root / "old.pt", "--output", root / "old-comparison.json",
                    "--latency-warmup", 0, "--latency-repetitions", 1)
            error = command("train_adaptive.py", *common, "--length-penalty", "nan", succeeds=False)
            self.assertIn("length penalty must be finite", error)
            token_path.write_text(token_path.read_text() + "\n")
            error = command("evaluate_adaptive.py", *inputs, "--checkpoint", root / "adaptive.pt",
                            "--output", root / "bad.json", succeeds=False)
            self.assertIn("differ from the checkpoint", error)

    def test_single_dimension_csv_has_no_duplicate_problem_size_column(self):
        from training.adaptive_evaluation import write_rows

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lengths.csv"
            write_rows(path, [{"kernel_id": "a", "problem_size": [1024.0], "learned_h": 2}], ["problem_size"])
            with path.open() as handle:
                reader = csv.DictReader(handle)
                row = next(reader)
                self.assertEqual(len(reader.fieldnames), len(set(reader.fieldnames)))
                self.assertEqual(json.loads(row["problem_size"]), [1024.0])


if __name__ == "__main__":
    unittest.main()
