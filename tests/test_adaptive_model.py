import importlib.util
import math
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class AdaptiveModelTest(unittest.TestCase):
    def setUp(self):
        import torch
        from model.adaptive_config_scorer import AdaptiveConfigurationModel

        torch.manual_seed(5)
        self.torch = torch
        self.model = AdaptiveConfigurationModel(
            torch.randn(12, 5), 1, 3, hidden_size=7, context_features=4,
            hidden_sizes=(9,), dropout=0,
        )
        self.tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
        self.lengths = torch.tensor([4, 2])
        self.context = torch.tensor([[128.0], [1024.0]])

    def test_gate_output_layer_starts_near_shared_prior_without_zero_weights(self):
        torch = self.torch
        output = self.model.gate_network[-1]
        torch.testing.assert_close(output.bias, torch.full_like(output.bias, -2.0))
        self.assertGreater(float(output.weight.detach().abs().sum()), 0)
        self.assertLess(float(output.weight.detach().std()), 0.03)

    def test_compaction_final_state_padding_and_cached_scoring(self):
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            self.model.gate_network[-1].bias.fill_(-100)
        from model.sequence_config_scorer import SequenceConfigurationModel
        full = SequenceConfigurationModel.encode_kernel(self.model, self.tokens, self.lengths)
        cached = self.model.encode_kernel(self.tokens, self.lengths)
        self.assertEqual(cached.lengths.tolist(), [1, 1])
        self.assertEqual(tuple(cached.states.shape), (2, 1, 7))
        torch.testing.assert_close(cached.states[:, 0], full.states[torch.arange(2), self.lengths - 1])
        self.assertEqual(cached.gates.tolist(), [[1.0], [1.0]])
        shapes = []
        hook = self.model.scorer.register_forward_pre_hook(
            lambda module, args: shapes.append(tuple(args[0].shape))
        )
        logits = self.model.score_cached(cached, self.context)
        hook.remove()
        self.assertEqual(shapes, [(2, 1, 11)])
        self.assertTrue(bool(torch.isfinite(logits).all()))
        padded = torch.cat((self.tokens, torch.full((2, 8), 9)), dim=1)
        torch.testing.assert_close(logits, self.model(padded, self.lengths, self.context))
        torch.testing.assert_close(logits, self.model(self.tokens, self.lengths, self.context))
        torch.testing.assert_close(logits[1:2], self.model.score_cached(cached[1:2], self.context[1:2]))

    def test_content_can_select_different_h_for_equal_token_lengths(self):
        torch = self.torch
        from model.adaptive_config_scorer import AdaptiveConfigurationModel
        from model.sequence_config_scorer import KernelSequence

        model = AdaptiveConfigurationModel(torch.randn(4, 3), 1, 2, hidden_size=2,
                                            context_features=2, hidden_sizes=(4,), dropout=0).eval()
        with torch.no_grad():
            for parameter in model.gate_network.parameters():
                parameter.zero_()
            model.gate_network[0].weight[0, 0] = 1
            model.gate_network[-1].weight[0, 0] = 10
            model.gate_network[-1].bias.fill_(-2.5)
        states = torch.tensor([[[-1., 0.], [-1., 0.], [1., 0.]],
                               [[1., 0.], [-1., 0.], [1., 0.]]])
        cached = model.compress(KernelSequence(states, torch.tensor([3, 3])))
        self.assertEqual(cached.lengths.tolist(), [1, 2])
        self.assertEqual(tuple(cached[0:1].states.shape), (1, 1, 2))

    def test_expected_count_matches_sampling_and_has_gate_gradient(self):
        torch = self.torch
        import math

        self.model.train()
        alpha = torch.full((20000,), -2.0, requires_grad=True)
        gates, probability = self.model._gate_values(alpha)
        expected = torch.sigmoid(torch.tensor(-2.0 - (2 / 3) * math.log(0.1 / 1.1)))
        torch.testing.assert_close(probability[0], expected)
        self.assertAlmostEqual(float((gates > 0).float().mean()), float(expected), delta=0.02)
        probability.mean().backward()
        self.assertTrue(bool((alpha.grad > 0).all()))

    def test_length_objective_can_learn_to_close_gates(self):
        torch = self.torch
        from model.sequence_config_scorer import KernelSequence

        self.model.eval()
        sequence = KernelSequence(torch.zeros(2, 4, 7), torch.tensor([4, 4]))
        before = self.model.compress(sequence)
        optimizer = torch.optim.SGD(self.model.gate_network.parameters(), lr=0.5)
        for _ in range(12):
            optimizer.zero_grad()
            self.model.compress(sequence).expected_lengths.mean().backward()
            optimizer.step()
        after = self.model.compress(sequence)
        self.assertLess(float(after.expected_lengths.detach().mean()), float(before.expected_lengths.detach().mean()))
        self.assertEqual(after.lengths.tolist(), [1, 1])
        self.assertGreater(int(before.lengths.max()), 1)

    def test_prediction_gradients_reach_gates_and_lstm(self):
        torch = self.torch
        from training.losses import PerformanceTargetLoss

        self.model.eval()
        with torch.no_grad():
            self.model.gate_network[-1].bias.zero_()
        logits = self.model(self.tokens, self.lengths, self.context)
        PerformanceTargetLoss()(logits, torch.zeros_like(logits)).backward()
        self.assertGreater(float(self.model.gate_network[-1].bias.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.kernel_encoder.weight_ih_l0.grad.abs().sum()), 0)
        self.assertIsNone(self.model.token_encoder.embedding.weight.grad)

    def test_three_objectives_and_expected_length_reduction(self):
        torch = self.torch
        from training.adaptive_trainer import AdaptiveTrainer
        from training.supervised_trainer import TrainerConfig

        batch = {"tokens": self.tokens, "lengths": self.lengths, "context": self.context,
                 "kernel_ids": ["a", "b"], "runtimes": torch.tensor([[1., 2., 3.], [3., 1., 2.]]),
                 "targets": torch.tensor([[1., .1, .01], [.01, 1., .1]])}
        for kind in ("bce", "mse", "regret"):
            trainer = AdaptiveTrainer(self.model, TrainerConfig(epochs=1, loss=kind), length_penalty=0.1)
            history = trainer.fit([batch], [batch])
            final = history[-1]
            self.assertIn("validation_h_std", final)
            self.assertGreaterEqual(final["validation_h_std"], 0)
            self.assertGreaterEqual(final["validation_gate_open_fraction"], 0)
            self.assertLessEqual(final["validation_gate_open_fraction"], 1)
            self.assertAlmostEqual(final["validation_loss"],
                                   final["validation_task_loss"] + .1 * final["validation_expected_h"], places=6)
            self.assertAlmostEqual(trainer.evaluate([batch])[0], final["validation_loss"], places=6)
            self.assertEqual(len(trainer.length_history), 2)
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                AdaptiveTrainer(self.model, TrainerConfig(), length_penalty=value)

    def test_optional_ema_kl_regularization(self):
        torch = self.torch
        from training.adaptive_trainer import AdaptiveTrainer
        from training.supervised_trainer import TrainerConfig

        batch = {"tokens": self.tokens, "lengths": self.lengths, "context": self.context,
                 "kernel_ids": ["a", "b"], "runtimes": torch.tensor([[1., 2., 3.], [3., 1., 2.]]),
                 "targets": torch.tensor([[1., .1, .01], [.01, 1., .1]])}
        trainer = AdaptiveTrainer(
            self.model, TrainerConfig(epochs=1), kl_weight=0.01,
            kl_temperature=2.0, kl_ema_decay=0.9,
        )
        history = trainer.fit([batch], [batch])
        self.assertIsNotNone(trainer.ema_teacher)
        self.assertGreaterEqual(history[-1]["training_kl_loss"], 0)
        self.assertTrue(math.isfinite(history[-1]["training_kl_loss"]))
        self.assertFalse(any(parameter.requires_grad for parameter in trainer.ema_teacher.parameters()))
        for keyword, value in (("kl_weight", -1), ("kl_temperature", 0), ("kl_ema_decay", 1)):
            with self.assertRaises(ValueError):
                AdaptiveTrainer(self.model, TrainerConfig(), **{keyword: value})


if __name__ == "__main__":
    unittest.main()
