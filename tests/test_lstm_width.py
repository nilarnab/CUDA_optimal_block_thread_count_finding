import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[1]


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class LSTMWidthTest(unittest.TestCase):
    def test_factories_resize_lstm_dense_and_gates_with_prediction_gradients(self):
        import torch

        with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
            from train_adaptive import build_model

        torch.manual_seed(3)
        matrix = torch.randn(10, 5)
        tokens = torch.tensor([[1, 2, 3], [4, 5, 0]])
        lengths = torch.tensor([3, 2])
        context = torch.tensor([[32., 1., 1024.], [64., 1., 4096.]])
        for width in (2, 4, 8):
            for architecture in ("baseline", "sequence", "adaptive"):
                with self.subTest(width=width, architecture=architecture):
                    model = build_model(matrix, 3, 16, architecture, lstm_hidden_size=width).eval()
                    lstm = getattr(model.kernel_encoder, "lstm", model.kernel_encoder)
                    self.assertEqual(lstm.hidden_size, width)
                    self.assertEqual(model.scorer.network[0].in_features, width + 32)
                    self.assertEqual(model.token_encoder.output_size, 5)
                    cached = model.encode_kernel(tokens, lengths)
                    states = cached.states if hasattr(cached, "states") else cached
                    self.assertEqual(states.size(-1), width)
                    if architecture == "adaptive":
                        self.assertEqual(model.gate_network[0].in_features, 2 * width)
                    logits = model.score_cached(cached, context)
                    self.assertEqual(tuple(logits.shape), (2, 16))
                    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
                    self.assertTrue(bool(torch.isfinite(lstm.weight_ih_l0.grad).all()))
                    self.assertGreater(float(lstm.weight_ih_l0.grad.abs().sum()), 0)
                    self.assertIsNone(model.token_encoder.embedding.weight.grad)
        default = build_model(matrix, 3, 16, "adaptive")
        self.assertEqual(default.kernel_encoder.hidden_size, 128)

    def test_small_width_does_not_force_more_than_one_retained_state(self):
        import torch
        from model.adaptive_config_scorer import AdaptiveConfigurationModel

        tokens, lengths = torch.tensor([[1, 2, 3]]), torch.tensor([3])
        for width in (2, 4, 8):
            model = AdaptiveConfigurationModel(torch.randn(5, 3), 1, 16, hidden_size=width).eval()
            with torch.no_grad():
                model.gate_network[-1].bias.fill_(-100)
            cached = model.encode_kernel(tokens, lengths)
            self.assertEqual(cached.lengths.tolist(), [1])
            self.assertEqual(tuple(cached.states.shape), (1, 1, width))


if __name__ == "__main__":
    unittest.main()
