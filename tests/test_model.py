import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class ModelSmokeTest(unittest.TestCase):
    def test_forward_loss_metrics_and_inference(self):
        import torch

        from encoders.context_encoder import ContextEncoder
        from encoders.fasttext_encoder import FrozenFastTextEncoder
        from encoders.lstm_encoder import LSTMEncoder
        from frontend.tokenizer import LLVMTokenizer, Vocabulary
        from inference import InferenceEngine
        from launch.configuration_catalog import ConfigurationCatalog
        from model.config_scorer import ConfigurationModel, ConfigurationScorer
        from training.losses import PerformanceTargetLoss
        from training.metrics import runtime_metrics
        from training.supervised_trainer import SupervisedTrainer, TrainerConfig

        torch.manual_seed(0)
        catalog = ConfigurationCatalog.lscat()
        matrix = torch.randn(8, 12)
        matrix[0].zero_()
        token_encoder = FrozenFastTextEncoder(matrix)
        kernel_encoder = LSTMEncoder(12, 10, attention=True)
        context_encoder = ContextEncoder(1, 4)
        scorer = ConfigurationScorer(14, len(catalog), hidden_sizes=(16,), dropout=0)
        model = ConfigurationModel(
            token_encoder, kernel_encoder, context_encoder, scorer
        )

        tokens = torch.tensor([[2, 3, 4], [5, 6, 0]])
        lengths = torch.tensor([3, 2])
        context = torch.tensor([[1024.0], [2048.0]])
        logits = model(tokens, lengths, context)
        self.assertEqual(tuple(logits.shape), (2, 16))
        cached_logits = model.score_cached(
            model.encode_kernel(tokens, lengths), context
        )
        self.assertTrue(torch.equal(logits, cached_logits))
        targets = torch.zeros_like(logits)
        targets[:, 0] = 1
        self.assertTrue(torch.isfinite(PerformanceTargetLoss()(logits, targets)))
        runtimes = torch.arange(1, 17, dtype=torch.float32).repeat(2, 1)
        self.assertGreater(runtime_metrics(logits, runtimes).performance_score, 0)

        latency = SupervisedTrainer(
            model, TrainerConfig(epochs=1)
        ).benchmark_inference(
            [{"tokens": tokens, "lengths": lengths, "context": context}],
            warmup=1,
            repetitions=1,
        )
        self.assertGreater(latency.mean_ms, 0)
        self.assertGreater(latency.samples_per_second, 0)

        vocabulary = Vocabulary(
            {Vocabulary.PAD: 0, Vocabulary.UNK: 1, "define": 2, "void": 3}
        )
        engine = InferenceEngine(
            model, LLVMTokenizer(), vocabulary, catalog
        )
        prediction = engine.predict("define void @kernel()", (1024,))
        cached = engine.prepare_kernel("define void @kernel()")
        cached_prediction = engine.predict_cached(cached, (1024,))
        self.assertEqual(len(prediction.scores), 16)
        self.assertEqual(catalog.from_id(prediction.action_id), prediction.configuration)
        self.assertEqual(prediction.action_id, cached_prediction.action_id)


if __name__ == "__main__":
    unittest.main()
