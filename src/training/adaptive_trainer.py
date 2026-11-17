"""Train a learned representation budget with the existing action objectives."""

from __future__ import annotations

import math

import torch

from training.metrics import RuntimeMetrics, runtime_metrics
from training.sequence_trainer import SequenceTrainer
from training.retention_evaluation import length_evidence


class AdaptiveTrainer(SequenceTrainer):
    def __init__(self, model, config, *, length_penalty: float = 0.0):
        if not math.isfinite(length_penalty) or length_penalty < 0:
            raise ValueError("length penalty must be finite and nonnegative")
        super().__init__(model, config)
        self.length_penalty = length_penalty
        self.length_history = []

    def _epoch(self, batches, *, training, epoch):
        self.model.train(training)
        count = 0
        total_task = total_objective = total_expected = 0.0
        logits_all, runtimes_all, rows = [], [], []
        with torch.set_grad_enabled(training):
            for batch in batches:
                batch = self._move(batch)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                cached = self.model.encode_kernel(batch["tokens"], batch["lengths"])
                logits = self.model.score_cached(cached, batch["context"])
                actual = getattr(cached, "lengths", torch.ones_like(batch["lengths"]))
                expected = getattr(cached, "expected_lengths", actual.float())
                task_loss = self.loss(logits, batch["targets"])
                loss = task_loss + self.length_penalty * expected.mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite task/length objective")
                if training:
                    loss.backward()
                    if self.config.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                size = logits.size(0)
                count += size
                total_task += float(task_loss.detach()) * size
                total_objective += float(loss.detach()) * size
                total_expected += float(expected.detach().sum())
                if not training:
                    logits_all.append(logits.cpu())
                    runtimes_all.append(batch["runtimes"].cpu())
                    for index, kernel in enumerate(batch["kernel_ids"]):
                        rows.append({
                            "epoch": epoch, "split": "validation", "kernel_id": kernel,
                            "problem_size": batch["context"][index].tolist(),
                            "input_token_count": int(batch["lengths"][index]),
                            "learned_h": int(actual[index]), "expected_h": float(expected[index]),
                            "hidden_width": cached.states.size(-1) if hasattr(cached, "states") else cached.size(-1),
                        })
        if not count:
            raise ValueError("loader produced no batches")
        result = {"loss": total_objective / count, "task_loss": total_task / count,
                  "expected_h": total_expected / count}
        if not training:
            result.update(runtime_metrics(torch.cat(logits_all), torch.cat(runtimes_all)).__dict__)
            result["h_mean"] = sum(row["learned_h"] for row in rows) / len(rows)
            result["h_min"] = min(row["learned_h"] for row in rows)
            result["h_max"] = max(row["learned_h"] for row in rows)
            evidence = length_evidence(rows)
            result["distinct_h"] = evidence["distinct_h_count"]
            result["nontrivial_variable_length"] = evidence["nontrivial_variable_length"]
        return result, rows

    def evaluate(self, batches):
        values, _ = self._epoch(batches, training=False, epoch=0)
        return values["loss"], RuntimeMetrics(
            values["performance_score"], values["mean_regret"], values["exact_optimal_rate"]
        )

    def fit(self, train_batches, validation_batches, *, epoch_callback=None):
        history = []
        for epoch in range(1, self.config.epochs + 1):
            train, _ = self._epoch(train_batches, training=True, epoch=epoch)
            validation, rows = self._epoch(validation_batches, training=False, epoch=epoch)
            values = {
                "epoch": epoch, "training_loss": train["loss"],
                "training_task_loss": train["task_loss"], "training_expected_h": train["expected_h"],
                "validation_loss": validation["loss"], "validation_task_loss": validation["task_loss"],
                "validation_expected_h": validation["expected_h"],
                "validation_h_mean": validation["h_mean"], "validation_h_min": validation["h_min"],
                "validation_h_max": validation["h_max"],
                "validation_distinct_h": validation["distinct_h"],
                "validation_nontrivial_variable_length": validation["nontrivial_variable_length"],
                **{name: validation[name] for name in ("performance_score", "mean_regret", "exact_optimal_rate")},
            }
            history.append(values)
            self.length_history.extend(rows)
            if self.config.verbose:
                print(f"[adaptive:epoch] {epoch}/{self.config.epochs} "
                      f"task={train['task_loss']:.6f} expected_h={train['expected_h']:.2f} "
                      f"validation_task={validation['task_loss']:.6f} "
                      f"h={validation['h_mean']:.2f} [{validation['h_min']},{validation['h_max']}] "
                      f"performance={validation['performance_score']:.6f}")
            if epoch_callback:
                epoch_callback(values)
        return history
