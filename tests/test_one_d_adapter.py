import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from data.one_d_adapter import prepare_one_d_dataset
from data.runtime_table import RuntimeKey, RuntimeTable
from data.splits import split_grouped_kernel_ids
from launch.configuration_catalog import Configuration


ROOT = Path(__file__).parents[1]


def fixture(root):
    run = root / "run"
    (run / "source" / "kernels").mkdir(parents=True)
    contexts, rows = [], []
    for name, group, suffix in (("alpha", "shared", ".cu"), ("beta", "shared", ".cu"),
                                ("gamma", "group_c", ".cuh"), ("delta", "group_d", ".cu")):
        path = f"kernels/{name}{suffix}"
        source = f'// {name}\n__global__ void {name}() {{}}\n'.encode()
        (run / "source" / path).write_bytes(source)
        for k in ((1024, 4096) if name == "alpha" else (1024,)):
            contexts.append({
                "kernel_id": f"{name}__k{k}", "base_kernel_id": name,
                "m": 32, "n": 1, "k": k, "input_seed_index": 0,
                "split_group": group, "grid_policy": "coverage",
                "source_path": path, "source_sha256": hashlib.sha256(source).hexdigest(),
            })
            for block in range(64, 1025, 64):
                rows.append({
                    "kernel_id": name, "m": 32, "n": 1, "k": k, "input_seed_index": 0,
                    "split_group": group, "grid_policy": "coverage",
                    "block_x": block, "block_y": 1, "median_ms": block / 1000,
                    "valid": "True", "repeats_ok": 3, "repeats_planned": 3,
                })
    (run / "contexts.json").write_text(json.dumps(contexts))
    write_actions(run, rows)
    return run, contexts, rows


def write_actions(run, rows):
    with (run / "heavy_action_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class OneDAdapterTest(unittest.TestCase):
    def test_preserves_mnk_base_sources_groups_and_microseconds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _, _ = fixture(root)
            before = (run / "heavy_action_table.csv").read_bytes()
            output = root / "processed"
            summary = prepare_one_d_dataset(run, output)
            self.assertEqual((summary["kernels"], summary["contexts"], summary["runtime_rows"]), (4, 5, 80))
            table = RuntimeTable.from_csv(output / "runtimes.csv", problem_columns=("m", "n", "k"))
            for k in (1024, 4096):
                self.assertEqual(table.runtime(RuntimeKey("alpha", (32, 1, k)), Configuration(64)), 64)
            self.assertEqual(len(table.contexts()), 5)
            self.assertEqual((output / "cuda" / "alpha.cu").read_bytes(),
                             (run / "source" / "kernels" / "alpha.cu").read_bytes())
            self.assertIn('#include "gamma.cuh"', (output / "cuda" / "gamma.cu").read_text())
            self.assertEqual((output / "cuda" / "gamma.cuh").read_bytes(),
                             (run / "source" / "kernels" / "gamma.cuh").read_bytes())
            groups = json.loads((output / "split_groups.json").read_text())
            self.assertEqual(groups["alpha"], groups["beta"])
            self.assertEqual((run / "heavy_action_table.csv").read_bytes(), before)
            with self.assertRaises(FileExistsError):
                prepare_one_d_dataset(run, output)

    def test_rejects_incomplete_invalid_and_duplicate_actions_before_writing(self):
        for kind in ("missing", "invalid", "duplicate", "repeats", "nan"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, _, rows = fixture(root)
                if kind == "missing":
                    rows.pop()
                elif kind == "duplicate":
                    rows.append(rows[0])
                else:
                    field, value = {"invalid": ("valid", "False"), "repeats": ("repeats_ok", 2),
                                    "nan": ("median_ms", "nan")}[kind]
                    rows[0][field] = value
                write_actions(run, rows)
                with self.assertRaises(ValueError):
                    prepare_one_d_dataset(run, root / "output")
                self.assertFalse((root / "output").exists())

    def test_rejects_changed_sources_and_source_path_escape(self):
        for kind in ("hash", "path"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, contexts, _ = fixture(root)
                if kind == "hash":
                    contexts[0]["source_sha256"] = "0" * 64
                else:
                    contexts[0]["source_path"] = "../outside.cu"
                (run / "contexts.json").write_text(json.dumps(contexts))
                with self.assertRaisesRegex(ValueError, "hash mismatch|escapes"):
                    prepare_one_d_dataset(run, root / "output")

    def test_group_split_keeps_related_kernels_together(self):
        groups = {"a": "shared", "b": "shared", "c": "other", "d": "third"}
        for seed in range(3):
            first = split_grouped_kernel_ids(groups, groups, validation_groups=1, seed=seed)
            second = split_grouped_kernel_ids(groups, groups, validation_groups=1, seed=seed)
            self.assertEqual(first, second)
            self.assertEqual("a" in first.train, "b" in first.train)
            self.assertFalse({groups[k] for k in first.train} & {groups[k] for k in first.validation})
        with self.assertRaisesRegex(ValueError, "map every kernel"):
            split_grouped_kernel_ids(["a", "b"], {"a": "shared"})

    def test_group_split_can_select_fixed_training_and_validation_group_counts(self):
        groups = {f"kernel_{index}": f"group_{index // 2}" for index in range(20)}
        split = split_grouped_kernel_ids(
            groups, groups, training_groups=4, validation_groups=2, seed=7
        )
        train_groups = {groups[kernel] for kernel in split.train}
        validation_groups = {groups[kernel] for kernel in split.validation}
        self.assertEqual(len(train_groups), 4)
        self.assertEqual(len(validation_groups), 2)
        self.assertFalse(train_groups & validation_groups)
        self.assertEqual(len(split.train), 8)
        self.assertEqual(len(split.validation), 4)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class OneDTrainingTest(unittest.TestCase):
    def test_train_reload_compare_mnk_and_reject_group_leakage(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _, _ = fixture(root)
            output = root / "processed"
            prepare_one_d_dataset(run, output)
            groups = json.loads((output / "split_groups.json").read_text())
            token_path, matrix_path = root / "tokens.json", root / "embeddings.pt"
            token_path.write_text(json.dumps({kernel: [1, 2, 3] for kernel in groups}))
            torch.save(torch.randn(8, 5), matrix_path)

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

            inputs = ["--runtimes", output / "runtimes.csv", "--token-ids", token_path]
            error = command(
                "train_sequence.py", *inputs, "--embedding-matrix", matrix_path,
                "--problem-columns", "m", "n", "k", "--epochs", 1,
                "--checkpoint", root / "missing-groups.pt", succeeds=False,
            )
            self.assertIn("matching --split-groups JSON", error)
            self.assertFalse((root / "missing-groups.pt").exists())
            for architecture in ("baseline", "sequence"):
                command(
                    "train_sequence.py", *inputs, "--embedding-matrix", matrix_path,
                    "--checkpoint", root / f"{architecture}.pt", "--architecture", architecture,
                    "--problem-columns", "m", "n", "k", "--split-groups", output / "split_groups.json",
                    "--validation-groups", 1, "--seed", 1, "--epochs", 1, "--batch-size", 2,
                    "--latency-warmup", 0, "--latency-repetitions", 1,
                )
            report = command(
                "evaluate_sequence.py", *inputs, "--checkpoint", root / "sequence.pt",
                "--compare", root / "baseline.pt",
            )
            self.assertEqual(report["model"]["problem_columns"], ["m", "n", "k"])
            self.assertTrue(all(len(row["problem_size"]) == 3 for row in report["model"]["selections"]))
            self.assertEqual(report["model"]["validation_groups"], report["comparison"]["validation_groups"])
            saved = torch.load(root / "sequence.pt", weights_only=True)
            saved["metadata"]["split"] = {"train": ["beta", "gamma", "delta"], "validation": ["alpha"]}
            torch.save(saved, root / "leaky.pt")
            error = command("evaluate_sequence.py", *inputs, "--checkpoint", root / "leaky.pt", succeeds=False)
            self.assertIn("source group occurs in both", error)


if __name__ == "__main__":
    unittest.main()
