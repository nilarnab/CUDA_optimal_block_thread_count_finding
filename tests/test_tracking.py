import unittest

from training.tracking import (
    WandbEpochLogger,
    wandb_epoch_metrics,
    wandb_inference_metrics,
)


class FakeRun:
    def __init__(self):
        self.calls = []

    def log(self, metrics, *, step):
        self.calls.append((metrics, step))


class TrackingTest(unittest.TestCase):
    def test_maps_and_logs_epoch_metrics(self):
        result = {
            "epoch": 3,
            "training_loss": 0.4,
            "validation_loss": 0.5,
            "performance_score": 0.97,
            "mean_regret": 0.03,
            "exact_optimal_rate": 0.25,
        }
        metrics = wandb_epoch_metrics(result)
        self.assertEqual(metrics["train/loss"], 0.4)
        self.assertEqual(metrics["validation/performance_score"], 0.97)
        run = FakeRun()
        WandbEpochLogger(run)(result)
        self.assertEqual(run.calls, [(metrics, 3)])

    def test_maps_inference_latency(self):
        class Latency:
            mean_ms = 1.2
            median_ms = 1.0
            p95_ms = 1.8
            samples_per_second = 833.33

        metrics = wandb_inference_metrics(Latency())
        self.assertEqual(
            metrics["inference/runtime_predictor_latency_median_ms"], 1.0
        )
        self.assertEqual(
            metrics["inference/runtime_predictor_throughput_per_second"], 833.33
        )
