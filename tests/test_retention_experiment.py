import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[1]


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class RetentionEvidenceTest(unittest.TestCase):
    def row(self, kernel, h, n=4, epoch=1):
        return {"kernel_id": kernel, "learned_h": h, "input_token_count": n,
                "hidden_width": 8, "epoch": epoch, "split": "validation"}

    def test_signal_excludes_single_state_full_input_and_constant_lengths(self):
        from training.retention_evaluation import length_evidence

        for rows in ([self.row("a", 1), self.row("b", 1)],
                     [self.row("a", 3, 3), self.row("b", 4, 4)],
                     [self.row("a", 2), self.row("b", 2, 6)]):
            self.assertFalse(length_evidence(rows)["nontrivial_variable_length"])
        rows = [self.row("a", 2), self.row("a", 2), self.row("b", 1)]
        evidence = length_evidence(rows)
        self.assertTrue(evidence["nontrivial_variable_length"])
        self.assertTrue(evidence["equal_input_length_different_h"])
        self.assertEqual(evidence["kernel_count"], 2)
        self.assertEqual(evidence["mean_hidden_values"], 12)
        self.assertTrue(length_evidence([self.row("a", 1), self.row("b", 4)])["nontrivial_variable_length"])
        with self.assertRaisesRegex(ValueError, "changed across contexts"):
            length_evidence([self.row("a", 1), self.row("a", 2)])

    def test_persistence_does_not_reward_transient_variation(self):
        from training.retention_evaluation import persistent_length_evidence

        rows = [self.row(kernel, h, epoch=epoch) for epoch in range(1, 6)
                for kernel, h in (("a", 2), ("b", 1))]
        evidence = persistent_length_evidence(rows)
        self.assertTrue(evidence["persistent_nontrivial_variation"])
        self.assertTrue(evidence["stable_kernel_counts"])
        self.assertFalse(persistent_length_evidence(rows, through_epoch=4)["window_complete"])
        rows.extend([self.row("a", 1, epoch=6), self.row("b", 1, epoch=6)])
        self.assertFalse(persistent_length_evidence(rows)["persistent_nontrivial_variation"])
        self.assertTrue(persistent_length_evidence(rows, through_epoch=5)["persistent_nontrivial_variation"])

    def test_final_only_ablation_measures_actual_task_and_runtime_effect(self):
        import torch
        from model.adaptive_config_scorer import AdaptiveKernelSequence
        from launch.configuration_catalog import Configuration, ConfigurationCatalog
        from training.retention_evaluation import retention_diagnostics

        class ProbeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_network = torch.nn.Identity()

            def encode_kernel(self, tokens, lengths):
                count = 2 if int(tokens[0, 0]) == 1 else 1
                return AdaptiveKernelSequence(torch.ones(1, count, 2), torch.tensor([count]),
                                              torch.ones(1, count), torch.tensor([float(count)]))

            def score_cached(self, cached, context):
                return torch.tensor([[0., 4.]]) if int(cached.lengths[0]) > 1 else torch.tensor([[4., 0.]])

        batch = {"kernel_ids": ["a", "b"], "tokens": torch.tensor([[1, 2, 3], [2, 3, 4]]),
                 "lengths": torch.tensor([3, 3]), "context": torch.ones(2, 1),
                 "runtimes": torch.tensor([[10., 1.], [1., 10.]]),
                 "targets": torch.tensor([[0., 1.], [1., 0.]])}
        catalog = ConfigurationCatalog([Configuration(64), Configuration(256)])
        result = retention_diagnostics(ProbeModel(), [batch], catalog, "cpu")
        self.assertTrue(result["lengths"]["nontrivial_variable_length"])
        self.assertEqual(result["selected"]["mean_regret"], 0)
        self.assertEqual(result["final_state_only"]["mean_regret"], 4.5)
        self.assertEqual(result["fixed_256"]["mean_regret"], 4.5)
        self.assertTrue(result["extra_states_improve_regret"])
        self.assertTrue(result["extra_states_improve_task_loss"])
        batch["tokens"][0, 0] = 2
        collapsed = retention_diagnostics(ProbeModel(), [batch], catalog, "cpu")
        self.assertEqual(collapsed["extra_state_regret_gain"], 0)
        self.assertFalse(collapsed["lengths"]["nontrivial_variable_length"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class RetentionExperimentIntegrationTest(unittest.TestCase):
    def prepared(self, root):
        import torch
        from data.one_d_adapter import prepare_one_d_dataset
        from test_one_d_adapter import fixture

        run, _, _ = fixture(root)
        output = root / "data"
        prepare_one_d_dataset(run, output)
        groups = json.loads((output / "split_groups.json").read_text())
        (output / "token_ids.json").write_text(json.dumps({kernel: [1, 2, 3] for kernel in groups}))
        torch.save(torch.randn(6, 5), output / "fasttext_embeddings.pt")
        return output

    def test_best_checkpoint_keeps_earlier_weights_by_regret_not_loss(self):
        import torch
        from training.adaptive_trainer import AdaptiveTrainer

        with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
            import train_adaptive

        def controlled_fit(trainer, train, validation, *, epoch_callback):
            history = []
            for epoch, regret, task in ((1, .3, .7), (2, .1, .6), (3, .2, .5)):
                with torch.no_grad():
                    parameter = next(p for p in trainer.model.parameters() if p.requires_grad)
                    parameter.fill_(epoch / 100)
                values = {"epoch": epoch, "mean_regret": regret, "validation_task_loss": task,
                          "training_loss": task, "validation_loss": task,
                          "performance_score": 1 / (1 + regret), "exact_optimal_rate": 0,
                          "validation_nontrivial_variable_length": False}
                history.append(values)
                epoch_callback(values)
            return history

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.prepared(root)
            checkpoint = root / "run.pt"
            argv = ["train_adaptive.py", "--runtimes", str(data / "runtimes.csv"),
                    "--token-ids", str(data / "token_ids.json"),
                    "--embedding-matrix", str(data / "fasttext_embeddings.pt"),
                    "--problem-columns", "m", "n", "k", "--split-groups", str(data / "split_groups.json"),
                    "--validation-groups", "1", "--seed", "7", "--split-seed", "1",
                    "--epochs", "3", "--lstm-hidden-size", "2", "--save-best",
                    "--checkpoint", str(checkpoint), "--latency-warmup", "0", "--latency-repetitions", "1"]
            with patch.object(sys, "argv", argv), patch.object(AdaptiveTrainer, "fit", controlled_fit), contextlib.redirect_stdout(io.StringIO()):
                train_adaptive.main()
            best = torch.load(checkpoint.with_suffix(".best.pt"), weights_only=True)
            final = torch.load(checkpoint, weights_only=True)
            self.assertEqual(best["metadata"]["selected_epoch"], 2)
            self.assertEqual(final["metadata"]["selected_epoch"], 3)
            self.assertEqual(best["metadata"]["seed"], 7)
            self.assertEqual(best["metadata"]["split_seed"], 1)
            torch.testing.assert_close(best["model_state"]["kernel_encoder.weight_ih_l0"],
                                       torch.full_like(best["model_state"]["kernel_encoder.weight_ih_l0"], .02))
            torch.testing.assert_close(final["model_state"]["kernel_encoder.weight_ih_l0"],
                                       torch.full_like(final["model_state"]["kernel_encoder.weight_ih_l0"], .03))

    def test_matched_experiment_runs_controls_and_writes_negative_signal_honestly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.prepared(root)
            output = root / "experiment"
            command = [sys.executable, str(ROOT / "scripts" / "run_retention_experiment.py"),
                       "--data-dir", str(data), "--output-dir", str(output), "--device", "cpu",
                       "--epochs", "2", "--batch-size", "2", "--validation-groups", "1",
                       "--lstm-hidden-size", "2", "--seeds", "7", "--split-seed", "1",
                       "--latency-warmup", "0", "--latency-repetitions", "1"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180,
                                    env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "experiment.json").read_text())
            self.assertEqual(len(report["runs"]), 4)
            self.assertEqual({(r["architecture"], r["length_penalty"]) for r in report["runs"]},
                             {("baseline", 0), ("adaptive", 0), ("adaptive", 1e-5), ("adaptive", 1e-4)})
            self.assertTrue(all(r["seed"] == 7 and r["split_seed"] == 1 for r in report["runs"]))
            self.assertTrue(all(not r["best_variable_output_signal"] for r in report["runs"]))
            self.assertTrue(all(r["replicated_useful_signal"] is None for r in report["replication"]))
            for row in report["runs"]:
                self.assertTrue(Path(row["best_checkpoint"]).is_file())
                evaluated = json.loads(Path(row["best_evaluation"]).read_text())["model"]
                self.assertEqual(evaluated["epoch"], row["best_epoch"])
            with (output / "experiment.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertIn("best_distinct_h", rows[0])
            repeated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30,
                                      env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("new or empty", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
