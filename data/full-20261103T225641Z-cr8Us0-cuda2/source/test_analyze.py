import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import analyze


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = [dict(kernel_id="vector_add", size=1024, block_x=block, block_y=1,
                          round=round_id, status="ok", kernel_ms=time, error="")
                     for block, times in ((64, (2, 3, 4)), (128, (5, 6, 7)))
                     for round_id, time in enumerate(times)]
        self.plan = [{field: row[field] for field in analyze.KEY_FIELDS} for row in self.rows]

    def write(self):
        (self.root / "planned.json").write_text(json.dumps(self.plan))
        with (self.root / "trials.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=analyze.TRIAL_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def output(self, filename):
        with (self.root / filename).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_aggregation_and_descriptive_rewards(self):
        self.write()
        result = analyze.analyze(self.root)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["full_groups"], 2)
        measurements = self.output("measurements.csv")
        self.assertEqual(float(measurements[0]["median_ms"]), 3)
        self.assertEqual(float(measurements[0]["mad_ms"]), 1)
        self.assertAlmostEqual(float(measurements[0]["relative_mad"]), 1 / 3)
        actions = self.output("action_table.csv")
        self.assertAlmostEqual(float(actions[0]["reward_log_vs_default"]), math.log(2))
        self.assertEqual(float(actions[0]["regret"]), 0)
        self.assertEqual(float(actions[1]["regret"]), 1)
        self.assertEqual(actions[0]["baseline_block_x"], "128")
        self.assertEqual(result["contexts"][0]["default_to_best_speedup"], 2)
        self.assertTrue((self.root / "COMPLETE").is_file())
        self.assertIn("not a reproduction", (self.root / "summary.md").read_text())

    def test_failures_are_preserved_and_never_zero_latency(self):
        for row in self.rows[:3]:
            row.update(status="timeout", kernel_ms="", error="deadline")
        self.write()
        result = analyze.analyze(self.root)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["failed_trials"], 3)
        first = self.output("measurements.csv")[0]
        self.assertEqual(first["repeats_ok"], "0")
        self.assertEqual(first["repeats_failed"], "3")
        self.assertEqual(first["median_ms"], "")
        self.assertEqual(len(self.output("action_table.csv")), 1)
        self.assertTrue((self.root / "INCOMPLETE").exists())

    def test_partial_failure_excluded_from_actions(self):
        self.rows[0].update(status="incorrect", kernel_ms="")
        self.write()
        result = analyze.analyze(self.root)
        self.assertEqual(result["failed_groups"], 1)
        self.assertEqual(self.output("measurements.csv")[0]["median_ms"], "3.5")
        self.assertEqual(len(self.output("action_table.csv")), 1)

    def test_missing_trial_is_incomplete_and_reports_written(self):
        self.rows.pop()
        self.write()
        self.assertEqual(analyze.main([str(self.root)]), 2)
        result = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(result["missing_trials"], 1)
        self.assertEqual(result["incomplete_groups"], 1)
        self.assertFalse((self.root / "COMPLETE").exists())

    def test_entire_unobserved_group_remains_in_measurements(self):
        self.rows = self.rows[3:]
        self.write()
        result = analyze.analyze(self.root)
        self.assertEqual(result["missing_trials"], 3)
        self.assertEqual(len(self.output("measurements.csv")), 2)
        self.assertEqual(self.output("measurements.csv")[0]["median_ms"], "")

    def test_duplicate_csv_key_rejected(self):
        self.rows.append(dict(self.rows[0]))
        self.write()
        result = analyze.analyze(self.root)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["duplicate_observed_keys"], 1)
        self.assertEqual(result["extra_trials"], 1)
        self.assertEqual(len(self.output("action_table.csv")), 1)

    def test_matching_duplicate_plan_and_csv_never_complete(self):
        self.plan.append(dict(self.plan[0]))
        self.rows.append(dict(self.rows[0]))
        self.write()
        result = analyze.analyze(self.root)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["duplicate_planned_keys"], 1)
        self.assertEqual(result["missing_trials"], 0)
        self.assertEqual(result["extra_trials"], 0)

    def test_unplanned_round_excludes_that_action(self):
        self.rows.append(dict(self.rows[0], round=10))
        self.write()
        result = analyze.analyze(self.root)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["extra_trials"], 1)
        self.assertEqual(result["incomplete_groups"], 1)
        self.assertEqual(len(self.output("action_table.csv")), 1)

    def test_bad_ok_timings_rejected(self):
        for value in ("nan", "inf", "-inf", "0", "-1", "", "oops"):
            with self.subTest(value=value):
                self.rows[0]["kernel_ms"] = value
                self.write()
                result = analyze.analyze(self.root)
                self.assertEqual(result["status"], "INCOMPLETE")
                self.assertEqual(result["failed_trials"], 1)
                self.assertTrue(result["validation_errors"])

    def test_stale_complete_removed_after_failed_reanalysis(self):
        self.write()
        analyze.analyze(self.root)
        self.rows.pop()
        self.write()
        analyze.analyze(self.root)
        self.assertFalse((self.root / "COMPLETE").exists())
        self.assertTrue((self.root / "INCOMPLETE").exists())

    def test_malformed_manifest_never_complete(self):
        self.write()
        (self.root / "planned.json").write_text('{"not": "a plan"}')
        result = analyze.analyze(self.root)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["full_groups"], 0)
        self.assertEqual(self.output("action_table.csv"), [])

    def test_truncated_csv_is_rejected_with_reports(self):
        self.write()
        with (self.root / "trials.csv").open("a") as stream:
            stream.write("vector_add,1024,64\n")
        result = analyze.analyze(self.root)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertTrue(any("truncated" in error for error in result["validation_errors"]))
        self.assertTrue((self.root / "summary.md").is_file())

    def test_output_symlink_is_replaced_without_touching_target(self):
        self.write()
        target = self.root / "unrelated.txt"
        target.write_text("keep me")
        (self.root / "summary.json").symlink_to(target)
        self.assertEqual(analyze.analyze(self.root)["status"], "COMPLETE")
        self.assertEqual(target.read_text(), "keep me")
        self.assertFalse((self.root / "summary.json").is_symlink())

    def test_manifest_dimensions_are_not_paths_or_floats(self):
        for field, bad in (("kernel_id", ""), ("size", True), ("size", 1.5), ("block_x", 0), ("round", -1)):
            with self.subTest(field=field, bad=bad):
                saved = self.plan[0][field]
                self.plan[0][field] = bad
                self.write()
                self.assertEqual(analyze.analyze(self.root)["status"], "INCOMPLETE")
                self.plan[0][field] = saved

    def test_default_tie_prefers_larger_x(self):
        self.rows[3]["block_x"], self.rows[3]["block_y"] = 8, 16
        self.plan[3]["block_x"], self.plan[3]["block_y"] = 8, 16
        self.write()
        result = analyze.analyze(self.root)
        self.assertEqual(result["contexts"][0]["default_block_x"], 128)


if __name__ == "__main__":
    unittest.main()
