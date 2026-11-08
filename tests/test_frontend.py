import unittest

from frontend.tokenizer import LLVMTokenizer, Vocabulary


class LLVMTokenizerTest(unittest.TestCase):
    def test_tokenizes_and_normalizes_llvm_ir(self):
        tokenizer = LLVMTokenizer()
        tokens = tokenizer.tokenize(
            "%7 = load float, ptr %input ; comment\n%8 = fadd float %7, 1.5"
        )
        self.assertEqual(
            tokens,
            [
                "%ID",
                "=",
                "load",
                "float",
                ",",
                "ptr",
                "%ID",
                "%ID",
                "=",
                "fadd",
                "float",
                "%ID",
                ",",
                "<NUM>",
            ],
        )

    def test_vocabulary_is_deterministic(self):
        vocabulary = Vocabulary.build([["load", "float"], ["load", "ptr"]])
        self.assertEqual(vocabulary.token_to_id["load"], 2)
        self.assertEqual(vocabulary.encode(["missing"]), (1,))


if __name__ == "__main__":
    unittest.main()

