import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class SequenceModelTest(unittest.TestCase):
    def setUp(self):
        import torch
        from model.sequence_config_scorer import SequenceConfigurationModel

        torch.manual_seed(4)
        self.torch = torch
        self.model = SequenceConfigurationModel(
            torch.randn(12, 5), 1, 3, hidden_size=7, context_features=4,
            hidden_sizes=(9,), dropout=0,
        ).eval()
        self.tokens = torch.tensor([[2, 3, 4, 5], [6, 7, 0, 0], [8, 0, 0, 0]])
        self.lengths = torch.tensor([4, 2, 1])
        self.context = torch.tensor([[128.0], [512.0], [1024.0]])

    def test_dense_receives_all_steps_and_padding_does_not_change_scores(self):
        torch = self.torch
        shapes = []
        hook = self.model.scorer.register_forward_pre_hook(
            lambda module, inputs: shapes.append(tuple(inputs[0].shape))
        )
        together = self.model(self.tokens, self.lengths, self.context)
        hook.remove()
        self.assertEqual(shapes, [(3, 4, 11)])
        self.assertEqual(tuple(together.shape), (3, 3))
        for index, length in enumerate(self.lengths.tolist()):
            alone = self.model(
                self.tokens[index:index + 1, :length], self.lengths[index:index + 1],
                self.context[index:index + 1],
            )
            torch.testing.assert_close(together[index:index + 1], alone)
        padded = torch.cat((self.tokens, torch.full((3, 5), 9)), dim=1)
        torch.testing.assert_close(together, self.model(padded, self.lengths, self.context))
        cache = self.model.encode_kernel(self.tokens, self.lengths)
        self.assertEqual(tuple(cache[1:2].states.shape), (1, 2, 7))
        torch.testing.assert_close(together, self.model.score_cached(cache, self.context))

    def test_final_state_model_always_returns_one_state(self):
        from model.sequence_config_scorer import (
            FinalStateConfigurationModel,
            SequenceConfigurationModel,
        )

        model = FinalStateConfigurationModel(
            self.model.token_encoder.embedding.weight.detach(), 1, 3,
            hidden_size=7, context_features=4, hidden_sizes=(9,), dropout=0,
        ).eval()
        full = SequenceConfigurationModel.encode_kernel(model, self.tokens, self.lengths)
        retained = model.encode_kernel(self.tokens, self.lengths)
        rows = self.torch.arange(len(self.lengths))
        self.assertEqual(retained.lengths.tolist(), [1, 1, 1])
        self.assertEqual(tuple(retained.states.shape), (3, 1, 7))
        self.torch.testing.assert_close(
            retained.states[:, 0], full.states[rows, self.lengths - 1]
        )

    def test_context_only_model_ignores_kernel_tokens(self):
        from model.config_scorer import ContextOnlyConfigurationModel

        model = ContextOnlyConfigurationModel(
            self.model.token_encoder.embedding.weight.detach(), 1, 3,
            context_features=4, hidden_sizes=(9,), dropout=0,
        ).eval()
        cache = model.encode_kernel(self.tokens, self.lengths)
        self.assertEqual(tuple(cache.shape), (3, 0))
        logits = model(self.tokens, self.lengths, self.context)
        changed_tokens = self.tokens.roll(1, dims=1)
        self.torch.testing.assert_close(
            logits, model(changed_tokens, self.lengths, self.context)
        )
        self.assertEqual(model.scorer.network[0].in_features, 4)

    def test_gradients_reach_all_valid_timesteps_and_lstm(self):
        torch = self.torch
        from training.losses import PerformanceTargetLoss

        cache = self.model.encode_kernel(self.tokens, self.lengths)
        cache.states.retain_grad()
        logits = self.model.score_cached(cache, self.context)
        PerformanceTargetLoss()(logits, torch.zeros_like(logits)).backward()
        gradients = cache.states.grad.abs().sum(dim=-1)
        valid = torch.arange(4)[None, :] < self.lengths[:, None]
        self.assertTrue(bool((gradients[valid] > 0).all()))
        self.assertTrue(bool((gradients[~valid] == 0).all()))
        self.assertGreater(float(self.model.kernel_encoder.weight_ih_l0.grad.abs().sum()), 0)
        self.assertIsNone(self.model.token_encoder.embedding.weight.grad)

    def test_runtime_regret_value_direction_ties_and_invalid_labels(self):
        torch = self.torch
        from training.sequence_trainer import ExpectedRuntimeRegretLoss

        loss_fn = ExpectedRuntimeRegretLoss()
        logits = torch.zeros(1, 2, requires_grad=True)
        loss = loss_fn(logits, torch.tensor([[10.0, 20.0]]))
        self.assertAlmostEqual(float(loss.detach()), 0.5)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[-0.25, 0.25]]))
        self.assertEqual(float(loss_fn(logits, torch.ones(1, 2)).detach()), 0)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                loss_fn(logits, torch.tensor([[1.0, bad]]))
        with self.assertRaisesRegex(ValueError, "matching"):
            loss_fn(logits, torch.ones(2, 2))

    def test_losses_train_evaluate_and_benchmark_cached_sequence(self):
        torch = self.torch
        from training.sequence_trainer import SequenceTrainer
        from training.supervised_trainer import TrainerConfig

        batch = {
            "tokens": self.tokens, "lengths": self.lengths, "context": self.context,
            "runtimes": torch.tensor([[1., 2., 3.], [3., 1., 2.], [2., 3., 1.]]),
            "targets": torch.tensor([[1., .1, .01], [.01, 1., .1], [.1, .01, 1.]]),
        }
        for kind in ("bce", "mse", "regret"):
            trainer = SequenceTrainer(self.model, TrainerConfig(epochs=1, loss=kind))
            before = self.model.kernel_encoder.weight_ih_l0.detach().clone()
            history = trainer.fit([batch], [batch])
            self.assertTrue(torch.isfinite(torch.tensor(history[0]["validation_loss"])))
            self.assertFalse(torch.equal(before, self.model.kernel_encoder.weight_ih_l0))
            # Changing evaluation batch sizes must not change its sample-weighted loss.
            separate = [
                {name: value[index:index + 1] for name, value in batch.items()}
                for index in range(3)
            ]
            self.assertAlmostEqual(trainer.evaluate([batch])[0], trainer.evaluate(separate)[0], places=6)
        latency = trainer.benchmark_inference([batch], warmup=0, repetitions=1)
        self.assertGreater(latency.mean_ms, 0)

    def test_invalid_lengths_are_rejected(self):
        torch = self.torch
        for lengths in (torch.tensor([4, 0, 1]), torch.tensor([5, 2, 1]), self.lengths.float()):
            with self.assertRaisesRegex(ValueError, "lengths"):
                self.model(self.tokens, lengths, self.context)


if __name__ == "__main__":
    unittest.main()
