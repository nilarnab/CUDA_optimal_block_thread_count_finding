import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from data.lscat_adapter import import_lscat_1d
from data.runtime_table import RuntimeKey, RuntimeTable
from launch.configuration_catalog import Configuration, ConfigurationCatalog


class LSCATAdapterTest(unittest.TestCase):
    def test_imports_complete_1d_contexts_and_averages_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "LS-CAT"
            kernels = repository / "data" / "kernels"
            rows = []
            for repo, child, function, matrix in (
                ("10", "0", "alpha", "(240, 240)"),
                ("10", "1", "beta", "(496, 496)"),
            ):
                source = kernels / repo / child / f"{function}.cu"
                source.parent.mkdir(parents=True)
                source.write_text(f"__global__ void {function}() {{}}\n")
                for threads in range(64, 1025, 64):
                    repeats = (1.0, 3.0) if function == "alpha" else (2.0,)
                    for runtime in repeats:
                        rows.append({
                            "path": f"step2/{repo}/{child}/main.cu", "function": function,
                            "blocks": f"(1, {threads})", "matrix": matrix,
                            "time": runtime * threads,
                        })
                rows.append({
                    "path": f"step2/{repo}/{child}/main.cu", "function": function,
                    "blocks": "(8, 8)", "matrix": matrix, "time": 1,
                })
            # Make beta incomplete; it must not enter the training table.
            rows = [row for row in rows if not (
                row["function"] == "beta" and row["blocks"] == "(1, 1024)"
            )]
            rows.append({
                "path": "step2/10/1/main.cu", "function": "beta",
                "blocks": "0.0", "matrix": "0.0", "time": 0,
            })
            archive = repository / "results" / "archive" / "T4_18-6-3_sm52.zip"
            archive.parent.mkdir(parents=True)
            csv_path = root / "measurements.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.write(csv_path, "T4.csv")

            output = root / "processed"
            summary = import_lscat_1d(repository, output)
            self.assertEqual((summary["kernels"], summary["complete_contexts"], summary["runtime_rows"]),
                             (1, 1, 16))
            self.assertEqual(summary["rejected_nonpositive_or_nonfinite_rows"], 1)
            table = RuntimeTable.from_csv(
                output / "runtimes.csv", problem_columns=("m", "n")
            )
            key = RuntimeKey("10/0/alpha", (240, 240))
            self.assertTrue(table.has_complete_context(key, ConfigurationCatalog.lscat()))
            self.assertEqual(table.runtime(key, Configuration(64)), 128.0)
            self.assertEqual(json.loads((output / "split_groups.json").read_text()),
                             {"10/0/alpha": "10"})
            self.assertEqual(json.loads((output / "sources.json").read_text()),
                             {"10/0/alpha": "data/kernels/10/0/alpha.cu"})


if __name__ == "__main__":
    unittest.main()
