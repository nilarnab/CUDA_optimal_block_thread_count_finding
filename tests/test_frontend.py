import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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


class LLVMCompilerTest(unittest.TestCase):
    def test_adds_conda_cuda_target_include(self):
        from frontend.llvm_compiler import LLVMCompiler

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "kernel.cu"
            source.write_text("__global__ void k() {}", encoding="utf-8")
            include = root / "targets" / "x86_64-linux" / "include"
            include.mkdir(parents=True)
            (include / "cuda.h").touch()
            completed = type(
                "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
            )()
            with patch(
                "frontend.llvm_compiler.subprocess.run", return_value=completed
            ) as run:
                LLVMCompiler(cuda_path=str(root)).compile(
                    source, root / "kernel.ll"
                )

            command = run.call_args.args[0]
            self.assertIn(f"--cuda-path={root}", command)
            self.assertEqual(command[command.index("-I") + 1], str(include))


if __name__ == "__main__":
    unittest.main()

