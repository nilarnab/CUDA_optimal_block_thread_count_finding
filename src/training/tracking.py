def wandb_epoch_metrics(epoch_result):
    return {
        "epoch": int(epoch_result["epoch"]),
        "train/loss": float(epoch_result["training_loss"]),
        "validation/loss": float(epoch_result["validation_loss"]),
        "validation/performance_score": float(epoch_result["performance_score"]),
        "validation/mean_regret": float(epoch_result["mean_regret"]),
        "validation/exact_optimal_rate": float(
            epoch_result["exact_optimal_rate"]
        ),
    }


class WandbEpochLogger:
    def __init__(self, run):
        self.run = run

    def __call__(self, epoch_result):
        metrics = wandb_epoch_metrics(epoch_result)
        self.run.log(metrics, step=metrics["epoch"])


def wandb_inference_metrics(latency):
    return {
        "inference/runtime_predictor_latency_mean_ms": float(latency.mean_ms),
        "inference/runtime_predictor_latency_median_ms": float(latency.median_ms),
        "inference/runtime_predictor_latency_p95_ms": float(latency.p95_ms),
        "inference/runtime_predictor_throughput_per_second": float(
            latency.samples_per_second
        ),
    }
