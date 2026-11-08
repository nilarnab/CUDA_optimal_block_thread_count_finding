import tempfile
import unittest
from pathlib import Path

from data.dataset import KernelRuntimeDataset
from data.runtime_table import RuntimeKey, RuntimeTable
from data.splits import split_kernel_ids, split_train_validation_kernel_ids
from launch.configuration_catalog import Configuration, ConfigurationCatalog


class RuntimeDataTest(unittest.TestCase):
    def test_csv_medians_and_soft_targets(self):
        catalog = ConfigurationCatalog([Configuration(64), Configuration(128)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtimes.csv"
            path.write_text(
                "kernel_id,problem_size,block_x,block_y,runtime_us\n"
                "k1,1024,64,1,10\n"
                "k1,1024,64,1,14\n"
                "k1,1024,128,1,8\n",
                encoding="utf-8",
            )
            table = RuntimeTable.from_csv(path)

        key = RuntimeKey("k1", (1024,))
        self.assertEqual(table.runtime(key, Configuration(64)), 12.0)
        dataset = KernelRuntimeDataset(
            table, catalog, {"k1": [2, 3]}, target_exponent=2
        )
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].optimal_action, 1)
        self.assertAlmostEqual(dataset[0].target_quality[0], (8 / 12) ** 2)
        self.assertEqual(dataset[0].target_quality[1], 1.0)

        with tempfile.TemporaryDirectory() as directory:
            normalized = Path(directory) / "normalized.csv"
            table.write_medians(normalized)
            round_trip = RuntimeTable.from_csv(normalized)
        self.assertEqual(round_trip.runtime(key, Configuration(64)), 12.0)

    def test_rejects_empty_token_sequence(self):
        catalog = ConfigurationCatalog([Configuration(64)])
        table = RuntimeTable()
        table.add(RuntimeKey("k1", (128,)), Configuration(64), 1.0)
        with self.assertRaisesRegex(ValueError, "empty token sequence"):
            KernelRuntimeDataset(table, catalog, {"k1": []})

    def test_split_is_deterministic_and_disjoint(self):
        first = split_kernel_ids((f"k{index}" for index in range(20)), seed=7)
        second = split_kernel_ids((f"k{index}" for index in range(20)), seed=7)
        self.assertEqual(first, second)
        groups = [set(first.train), set(first.validation), set(first.test)]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

    def test_two_way_split_keeps_kernels_disjoint(self):
        split = split_train_validation_kernel_ids(
            (f"k{index}" for index in range(6)), validation_kernels=2, seed=4
        )
        self.assertEqual(len(split.train), 4)
        self.assertEqual(len(split.validation), 2)
        self.assertFalse(set(split.train) & set(split.validation))


if __name__ == "__main__":
    unittest.main()

