import csv
import json
import tempfile
import unittest
from pathlib import Path

from data.pilot_adapter import NLP_THREAD_COUNTS, prepare_pilot_dataset


class PilotAdapterTest(unittest.TestCase):
    def test_normalizes_orientation_and_skips_square_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kernel_root = root / "run" / "source" / "kernels"
            source_dir = kernel_root / "42" / "0"
            source_dir.mkdir(parents=True)
            (source_dir / "kernel.cu").write_text(
                '#include "includes.h"\n__global__ void kernel() {}\n',
                encoding="utf-8",
            )
            (kernel_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "kernels": [
                            {
                                "id": "kernel_a",
                                "path": "42/0/kernel.cu",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with (root / "run" / "action_table.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "kernel_id",
                        "size",
                        "block_x",
                        "block_y",
                        "median_ms",
                    ),
                )
                writer.writeheader()
                for threads in NLP_THREAD_COUNTS:
                    writer.writerow(
                        {
                            "kernel_id": "kernel_a",
                            "size": 240,
                            "block_x": 1,
                            "block_y": threads,
                            "median_ms": threads / 1000,
                        }
                    )
                writer.writerow(
                    {
                        "kernel_id": "kernel_a",
                        "size": 240,
                        "block_x": 8,
                        "block_y": 8,
                        "median_ms": 1,
                    }
                )

            output = root / "processed"
            summary = prepare_pilot_dataset(root / "run", output)
            self.assertEqual(summary.runtime_rows, 16)
            self.assertEqual(summary.skipped_non_1d_rows, 1)
            with (output / "runtimes.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["block_x"], "64")
            self.assertEqual(rows[0]["block_y"], "1")
            self.assertEqual(float(rows[0]["runtime_us"]), 64.0)
            mappings = json.loads(
                (output / "launch_mappings.json").read_text(encoding="utf-8")
            )
            self.assertEqual(mappings["kernel_a"]["64"], [1, 64, 1])
            self.assertTrue((output / "cuda" / "kernel_a.cu").is_file())


if __name__ == "__main__":
    unittest.main()
