import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest
from unittest.mock import patch

import collect


class CollectorTests(unittest.TestCase):
    trial = dict(kernel_id="vector_add", size=240, block_x=64, block_y=1, round=0, seed=7)

    @staticmethod
    def result(**changes):
        return dict(status="ok", correctness="passed", kernel_ms=0.01, batch_ms=1.0,
                    iterations=100, grid_x=900, grid_y=1, grid_z=1, input_elements=57600,
                    output_elements=57600, max_abs_error=0.0, warmup_launches=3, **changes)

    def measure_result(self, result):
        response = subprocess.CompletedProcess([], 0, json.dumps(result), "")
        with patch.object(collect.subprocess, "run", return_value=response):
            return collect.measure(Path("benchmark"), self.trial, 0)

    def test_valid_result(self):
        row = self.measure_result(self.result())
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["kernel_ms"], 0.01)

    def test_bad_results_do_not_create_successful_timing(self):
        for field, value in (("correctness", "failed"), ("kernel_ms", float("nan")),
                             ("kernel_ms", True), ("batch_ms", 2), ("iterations", 0),
                             ("max_abs_error", -1), ("status", "error")):
            result = self.result()
            result[field] = value
            with self.subTest(field=field, value=value):
                row = self.measure_result(result)
                self.assertEqual(row["status"], "protocol_error")
                self.assertNotIn("kernel_ms", row)

    def test_invalid_json(self):
        response = subprocess.CompletedProcess([], 0, "not JSON", "")
        with patch.object(collect.subprocess, "run", return_value=response):
            self.assertEqual(collect.measure(Path("benchmark"), self.trial, 0)["status"], "protocol_error")

    def test_process_error(self):
        response = subprocess.CompletedProcess([], 1, "", "illegal memory access")
        with patch.object(collect.subprocess, "run", return_value=response):
            row = collect.measure(Path("benchmark"), self.trial, 0)
        self.assertEqual(row["status"], "process_error")
        self.assertIn("illegal memory", row["error"])

    def test_timeout(self):
        with patch.object(collect.subprocess, "run", side_effect=subprocess.TimeoutExpired("benchmark", 1)):
            self.assertEqual(collect.measure(Path("benchmark"), self.trial, 0)["status"], "timeout")

    @staticmethod
    def device():
        return dict(max_threads_per_block=1024, max_block_dim_x=1024, max_block_dim_y=1024,
                    kernels={name: dict(max_threads_per_block=1024) for name in ("vector_add", "matrix_add")})

    def test_plan_shapes_seed_and_rounds(self):
        manifest = {"kernels": [{"id": "vector_add"}, {"id": "matrix_add"}]}
        plan = collect.make_plan(manifest, self.device(), "quick", 7)
        self.assertEqual(len(plan), 36 * 3 * 3)
        self.assertEqual(plan, collect.make_plan(manifest, self.device(), "quick", 7))
        self.assertEqual({t["block_y"] for t in plan if t["kernel_id"] == "vector_add"}, {1})
        self.assertIn((1, 1024), {(t["block_x"], t["block_y"]) for t in plan if t["kernel_id"] == "matrix_add"})
        self.assertEqual(len({t["seed"] for t in plan if t["kernel_id"] == "vector_add" and t["size"] == 240}), 1)
        self.assertEqual({t["round"] for t in plan}, {0, 1, 2})

    def test_plan_filters_device_and_kernel_limits(self):
        device = self.device()
        device["kernels"]["vector_add"]["max_threads_per_block"] = 256
        plan = collect.make_plan({"kernels": [{"id": "vector_add"}]}, device, "smoke", 1)
        self.assertEqual({t["block_x"] for t in plan}, {64, 128, 192, 256})

    def test_source_hash_and_path_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            shutil.copytree(Path(__file__).resolve().parent / "kernels", source / "kernels")
            path = source / "kernels/manifest.json"
            manifest = json.loads(path.read_text())
            kernel = source / "kernels" / manifest["kernels"][0]["path"]
            collect.validate_sources(source)
            kernel.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                collect.validate_sources(source)
            manifest["kernels"][0]["path"] = "../../outside"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "escapes"):
                collect.validate_sources(source)

    def test_changed_include_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            shutil.copytree(Path(__file__).resolve().parent / "kernels", source / "kernels")
            (source / "kernels/10047/0/includes.h").write_text("#define altered 1\n")
            with self.assertRaisesRegex(ValueError, "remain empty"):
                collect.validate_sources(source)

    def test_collector_to_analyzer_integration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            shutil.copytree(Path(__file__).resolve().parent / "kernels", source / "kernels")
            metadata = self.device()
            metadata["kernels"] = {name: {"max_threads_per_block": 1024} for name in collect.SOURCE_FILES}
            def fake_run(args, **kwargs):
                if "--describe" in args:
                    return subprocess.CompletedProcess(args, 0, json.dumps(metadata), "")
                return subprocess.CompletedProcess(args, 0, json.dumps(self.result()), "")
            with patch.object(collect.subprocess, "run", side_effect=fake_run):
                with patch("builtins.print"):
                    self.assertEqual(collect.collect(root, "smoke", 0), 0)
            self.assertTrue((root / "COMPLETE").is_file())
            self.assertEqual(json.loads((root / "summary.json").read_text())["ok_trials"], 100)

    def test_bundled_sources_match_manifest(self):
        manifest = collect.validate_sources(Path(__file__).resolve().parent)
        self.assertEqual(len(manifest["kernels"]), 6)


if __name__ == "__main__":
    unittest.main()
