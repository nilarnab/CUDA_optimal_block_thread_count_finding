import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).parents[1]


class PrepareFrontendFromIRTest(unittest.TestCase):
    def test_builds_artifacts_and_filters_failed_kernels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ir = root / "ir"
            (ir / "repo" / "0").mkdir(parents=True)
            (ir / "repo" / "0" / "kept.ll").write_text(
                "define void @kept() { ret void }\n", encoding="utf-8"
            )
            groups = root / "groups.json"
            groups.write_text(json.dumps({"repo/0/kept": "repo", "repo/1/failed": "repo"}))
            runtimes = root / "runtimes.csv"
            with runtimes.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("kernel_id", "m", "n", "runtime_us"))
                writer.writeheader()
                writer.writerow({"kernel_id": "repo/0/kept", "m": 1, "n": 1, "runtime_us": 2})
                writer.writerow({"kernel_id": "repo/1/failed", "m": 1, "n": 1, "runtime_us": 3})
            output = root / "output"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "prepare_frontend_from_ir.py"),
                 "--ir-dir", str(ir), "--output-dir", str(output),
                 "--runtimes", str(runtimes), "--split-groups", str(groups)],
                cwd=ROOT, capture_output=True, text=True,
                env={"PYTHONPATH": str(ROOT / "src")},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(set(json.loads((output / "token_ids.json").read_text())), {"repo/0/kept"})
            self.assertEqual(json.loads((output / "split_groups_frontend.json").read_text()),
                             {"repo/0/kept": "repo"})
            with (output / "runtimes_frontend.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["kernel_id"] for row in rows], ["repo/0/kept"])


if __name__ == "__main__":
    unittest.main()
