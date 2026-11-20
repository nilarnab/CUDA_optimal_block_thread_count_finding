import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class RuntimeInferenceTest(unittest.TestCase):
    def test_reload_checkpoint_and_rank_only_permitted_blocks(self):
        import torch

        from frontend.tokenizer import Vocabulary
        from inference import InferenceEngine
        from launch.configuration_catalog import Configuration
        from model.config_scorer import build_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocabulary = root / "vocabulary.json"
            Vocabulary({"<PAD>": 0, "<UNK>": 1, "define": 2}).save(vocabulary)
            actions = [Configuration(256), Configuration(64), Configuration(128)]
            for architecture in ("baseline", "sequence", "final", "context", "adaptive"):
                with self.subTest(architecture=architecture):
                    model = build_model(torch.randn(3, 5), 3, len(actions), architecture, 8).eval()
                    with torch.no_grad():
                        model.scorer.network[-1].weight.zero_()
                        model.scorer.network[-1].bias.copy_(torch.tensor([3., 1., 2.]))
                    checkpoint = root / "model.pt"
                    torch.save({
                        "metadata": {"format_version": 1, "adaptive_experiment_version": 1,
                                     "architecture": architecture, "lstm_hidden_size": 8,
                                     "context_size": 3, "problem_columns": ["m", "n", "k"],
                                     "action_catalog": [config.__dict__ for config in actions]},
                        "model_state": model.state_dict(),
                    }, checkpoint)
                    engine = InferenceEngine.from_checkpoint(checkpoint, vocabulary)
                    self.assertEqual(engine.problem_columns, ("m", "n", "k"))
                    cached = engine.prepare_kernel("define void @vector_add()")
                    states = getattr(cached.representation, "states", cached.representation)
                    self.assertEqual(states.size(-1), 0 if architecture == "context" else 8)
                    self.assertEqual(engine.predict_cached(cached, (64, 1, 1)).configuration.x, 256)
                    prediction = engine.predict_cached(cached, (128, 1, 1), allowed=actions[1:])
                    self.assertEqual(prediction.configuration.x, 128)
                    with self.assertRaisesRegex(ValueError, "no permitted"):
                        engine.predict_cached(cached, (64, 1, 1), allowed=[Configuration(32)])
                    with self.assertRaisesRegex(ValueError, "expected problem dimensions"):
                        engine.predict_cached(cached, (64,))

            Vocabulary({"<PAD>": 0, "<UNK>": 1}).save(vocabulary)
            with self.assertRaisesRegex(ValueError, "vocabulary IDs"):
                InferenceEngine.from_checkpoint(checkpoint, vocabulary)


if __name__ == "__main__":
    unittest.main()
