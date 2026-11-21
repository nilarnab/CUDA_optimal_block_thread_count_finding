import csv
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_launch_benchmarks.py"


class CollectLaunchBenchmarksTest(unittest.TestCase):
    def test_default_sweep_runs_once_with_finer_steps_near_each_prediction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            benchmark.write_text(
                '#!/bin/sh\n'
                'if [ "$#" -eq 2 ]; then\n'
                '  mode=model; block=576\n'
                '  if [ "$1" -gt 150051 ]; then block=1024; fi\n'
                'else mode=fixed; block="$3"; fi\n'
                'printf "%s trials=51 M=%s N=%s grid=261 block=%s verified\\n" "$mode" "$1" "$2" "$block"\n'
                'printf "gpu_us=8.25 selection_us=1.75 total_us=12.5\\n"\n',
                encoding="utf-8",
            )
            benchmark.chmod(0o755)
            manifest = root / "build.json"
            manifest.write_text('{"retained_states": 1}', encoding="utf-8")
            (root / "kernel.ll").write_text("ret i32 0 ; comments are ignored", encoding="utf-8")
            output = root / "output.csv"
            completed = subprocess.run(
                [
                    sys.executable, str(SCRIPT), "--size", "150051", "1500",
                    "--size", "180051", "1500",
                    "--fixed-bin", str(benchmark), "--model-bin", str(benchmark),
                    "--build-json", str(manifest), "--csv", str(output),
                ],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("model picked 576", completed.stdout)
            self.assertIn("model picked 1024", completed.stdout)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            coarse = set(range(64, 1025, 64))
            for size, predicted, extra in (
                ("150051x1500", 576, {480, 544, 608, 672}),
                ("180051x1500", 1024, {928, 992}),
            ):
                runs = [(row["mode"], int(row["block_size"])) for row in rows if row["problem_size"] == size]
                self.assertEqual(runs, [("model", predicted)] + [("fixed", block) for block in sorted(coarse | extra)])
            for row in rows:
                self.assertEqual(row["repeat"], "1")
                self.assertEqual(row["trials"], "51")
                self.assertEqual(row["total_us"], "12.5")
                self.assertEqual(row["retained_states"], "1")
                self.assertEqual(row["full_states"], "3")
                self.assertEqual(row["state_reduction_pct"], "66.667")


if __name__ == "__main__":
    unittest.main()
