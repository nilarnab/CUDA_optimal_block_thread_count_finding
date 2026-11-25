import copy
import math
import statistics

import torch
from torch.nn import functional as F

from training.metrics import RuntimeMetrics, runtime_metrics
from training.sequence_trainer import SequenceTrainer
from training.retention_evaluation import length_evidence


class AdaptiveTrainer(SequenceTrainer):
    def __init__(
        self,
        model,
        config,
        *,
        length_penalty=0.0,
        kl_weight=0.0,
        kl_temperature=2.0,
        kl_ema_decay=0.99,
    ):
        super().__init__(model, config)
        self.length_penalty = length_penalty
        self.kl_weight = kl_weight
        self.kl_temperature = kl_temperature
        self.kl_ema_decay = kl_ema_decay
        self.ema_teacher = None
        if kl_weight:
            self.ema_teacher = copy.deepcopy(self.model).requires_grad_(False).eval()
        self.length_history = []

    def _teacher_logits(self, batch):
        if self.ema_teacher is None:
            return None
        with torch.no_grad():
            cached = self.ema_teacher.encode_kernel(batch["tokens"], batch["lengths"])
            return self.ema_teacher.score_cached(cached, batch["context"])

    @torch.no_grad()
    def _update_ema_teacher(self):
        if self.ema_teacher is None:
            return
        decay = self.kl_ema_decay
        for teacher, current in zip(self.ema_teacher.parameters(), self.model.parameters()):
            teacher.mul_(decay).add_(current.detach(), alpha=1 - decay)
        for teacher, current in zip(self.ema_teacher.buffers(), self.model.buffers()):
            teacher.copy_(current)

    def _kl_loss(self, logits, teacher_logits):
        if teacher_logits is None:
            return logits.new_zeros(())
        temperature = self.kl_temperature
        return F.kl_div(
            F.log_softmax(logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature**2

    def _epoch(self, batches, *, training, epoch):
        self.model.train(training)
        count = 0
        total_task = total_objective = total_expected = total_kl = 0.0
        logits_all, runtimes_all, rows = [], [], []
        with torch.set_grad_enabled(training):
            for batch in batches:
                batch = self._move(batch)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                cached = self.model.encode_kernel(batch["tokens"], batch["lengths"])
                logits = self.model.score_cached(cached, batch["context"])
                context_only = (
                    isinstance(cached, torch.Tensor)
                    and cached.ndim == 2
                    and cached.size(-1) == 0
                )
                actual = getattr(
                    cached, "lengths",
                    torch.zeros_like(batch["lengths"])
                    if context_only else torch.ones_like(batch["lengths"]),
                )
                expected = getattr(cached, "expected_lengths", actual.float())
                task_loss = self.loss(logits, batch["targets"])
                teacher_logits = self._teacher_logits(batch) if training else None
                kl_loss = self._kl_loss(logits, teacher_logits)
                loss = (
                    task_loss
                    + self.length_penalty * expected.mean()
                    + self.kl_weight * kl_loss
                )
                if training:
                    loss.backward()
                    if self.config.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                    self._update_ema_teacher()
                size = logits.size(0)
                count += size
                total_task += float(task_loss.detach()) * size
                total_objective += float(loss.detach()) * size
                total_expected += float(expected.detach().sum())
                total_kl += float(kl_loss.detach()) * size
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
        result = {"loss": total_objective / count, "task_loss": total_task / count,
                  "expected_h": total_expected / count, "kl_loss": total_kl / count}
        if not training:
            result.update(runtime_metrics(torch.cat(logits_all), torch.cat(runtimes_all)).__dict__)
            retained_lengths = [row["learned_h"] for row in rows]
            result["h_mean"] = statistics.fmean(retained_lengths)
            result["h_std"] = statistics.pstdev(retained_lengths)
            result["h_min"] = min(retained_lengths)
            result["h_max"] = max(retained_lengths)
            optional_gate_count = sum(row["input_token_count"] - 1 for row in rows)
            open_optional_gates = sum(max(0, row["learned_h"] - 1) for row in rows)
            result["gate_open_fraction"] = (
                open_optional_gates / optional_gate_count
                if optional_gate_count else 0.0
            )
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
            lengths_by_kernel = {}
            for row in rows:
                lengths_by_kernel.setdefault(row["kernel_id"], []).append(row["learned_h"])
            validation_h_by_kernel = {
                kernel: statistics.fmean(lengths)
                for kernel, lengths in sorted(lengths_by_kernel.items())
            }
            values = {
                "epoch": epoch, "training_loss": train["loss"],
                "training_task_loss": train["task_loss"], "training_expected_h": train["expected_h"],
                "training_kl_loss": train["kl_loss"],
                "validation_loss": validation["loss"], "validation_task_loss": validation["task_loss"],
                "validation_expected_h": validation["expected_h"],
                "validation_h_mean": validation["h_mean"], "validation_h_std": validation["h_std"],
                "validation_h_min": validation["h_min"],
                "validation_h_max": validation["h_max"],
                "validation_gate_open_fraction": validation["gate_open_fraction"],
                "validation_distinct_h": validation["distinct_h"],
                "validation_nontrivial_variable_length": validation["nontrivial_variable_length"],
                **{name: validation[name] for name in ("performance_score", "mean_regret", "exact_optimal_rate")},
            }
            history.append(values)
            self.length_history.extend(rows)
            if self.config.verbose:
                per_kernel = ", ".join(
                    f"{kernel}={retained:.2f}"
                    for kernel, retained in validation_h_by_kernel.items()
                )
                print(f"[adaptive:epoch] {epoch}/{self.config.epochs} "
                      f"task={train['task_loss']:.6f} expected_h={train['expected_h']:.2f} "
                      f"kl={train['kl_loss']:.6f} "
                      f"validation_task={validation['task_loss']:.6f} "
                      f"h={validation['h_mean']:.2f}±{validation['h_std']:.2f} "
                      f"[{validation['h_min']},{validation['h_max']}] "
                      f"gates_open={validation['gate_open_fraction']:.2%} "
                      f"val_open_gates=[{per_kernel}] "
                      f"performance={validation['performance_score']:.6f}")
            if epoch_callback:
                epoch_callback(values)
        return history
