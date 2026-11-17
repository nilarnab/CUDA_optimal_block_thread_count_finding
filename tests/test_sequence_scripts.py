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
class SequenceScriptsTest(unittest.TestCase):
    def test_train_reload_compare_and_reject_changed_data(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_path, token_path = root / "runtimes.csv", root / "tokens.json"
            with runtime_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["kernel_id", "problem_size", "block_x", "block_y", "block_z", "runtime_us"])
                for kernel in range(4):
                    for size in (128, 1024):
                        for action in range(16):
                            writer.writerow([f"k{kernel}", size, 64 * (action + 1), 1, 1,
                                             10 + abs(action - kernel - size // 128)])
            token_path.write_text(json.dumps({f"k{k}": list(range(1, k + 3)) for k in range(4)}))
            matrix_path = root / "embeddings.pt"
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
                return result

            inputs = ["--runtimes", runtime_path, "--token-ids", token_path]
            for architecture, loss, width in (("baseline", "bce", 8), ("sequence", "bce", 4), ("sequence", "regret", 2)):
                checkpoint = root / f"{architecture}-{loss}.pt"
                trained = command(
                    "train_sequence.py", *inputs, "--embedding-matrix", matrix_path,
                    "--checkpoint", checkpoint, "--architecture", architecture,
                    "--lstm-hidden-size", width,
                    "--loss", loss, "--epochs", 2, "--batch-size", 3,
                    "--validation-kernels", 1, "--latency-warmup", 0, "--latency-repetitions", 1,
                )
                evaluated = command(
                    "evaluate_sequence.py", *inputs, "--checkpoint", checkpoint, "--batch-size", 1
                )["model"]
                self.assertEqual(trained["lstm_hidden_size"], width)
                self.assertEqual(evaluated["lstm_hidden_size"], width)
                for name in ("performance_score", "mean_regret", "exact_optimal_rate", "validation_loss"):
                    self.assertAlmostEqual(trained["final_epoch"][name], evaluated[name], places=6)
                self.assertEqual(len(evaluated["selections"]), 2)
                self.assertTrue(checkpoint.with_suffix(".metrics.json").is_file())
            output_path = root / "comparison.json"
            compared = command(
                "evaluate_sequence.py", *inputs, "--checkpoint", root / "sequence-bce.pt",
                "--compare", root / "baseline-bce.pt", "--output", output_path,
            )
            self.assertEqual(json.loads(output_path.read_text()), compared)
            self.assertGreater(compared["selected_runtime_sum_speedup"], 0)
            # Evaluation uses saved kernel IDs, never a newly randomized split.
            self.assertEqual(compared["model"]["validation_kernels"], compared["comparison"]["validation_kernels"])
            token_path.write_text(token_path.read_text() + "\n")
            failed = command(
                "evaluate_sequence.py", *inputs, "--checkpoint", root / "sequence-bce.pt", succeeds=False
            )
            self.assertIn("differ from the checkpoint", failed.stderr)


if __name__ == "__main__":
    unittest.main()
