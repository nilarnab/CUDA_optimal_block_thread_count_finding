import argparse
import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_supervised.py"
SPEC = importlib.util.spec_from_file_location("train_supervised_script", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TrainScriptTest(unittest.TestCase):
    def test_wandb_name_contains_hyperparameters(self):
        args = argparse.Namespace(
            wandb_run_name="pilot baseline",
            epochs=20,
            batch_size=256,
            loss="bce",
            target_exponent=12.0,
            seed=7,
            device="cuda:0",
            validation_kernels=2,
            latency_warmup=3,
            latency_repetitions=5,
        )
        self.assertEqual(
            MODULE.wandb_run_name(args),
            "pilot_baseline-ep20-bs256-loss-bce-exp12-seed7-dev-cuda_0-"
            "valk2-lw3-lr5",
        )


if __name__ == "__main__":
    unittest.main()
