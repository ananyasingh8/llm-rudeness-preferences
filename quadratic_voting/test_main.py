import argparse
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import torch
from huggingface_hub.errors import LocalEntryNotFoundError
from httpx import ProxyError
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from quadratic_voting import main


class GemmaRunnerTests(unittest.TestCase):
    def test_help_describes_python_runtime(self) -> None:
        help_text = main.build_parser().format_help()

        self.assertIn("Transformers and PyTorch", help_text)
        self.assertNotIn("llama.cpp", help_text)

    @patch(
        "quadratic_voting.main.download_model",
        return_value=Path("/cache/snapshot"),
    )
    def test_cli_download_dispatches_with_selected_cache(
        self, download: MagicMock
    ) -> None:
        with patch("builtins.print") as output:
            result = main.main(["--cache-dir", "/cache", "download"])

        self.assertEqual(result, 0)
        download.assert_called_once_with(Path("/cache"))
        output.assert_called_once_with(
            "Downloaded pinned Gemma checkpoint to: /cache/snapshot"
        )

    @patch("quadratic_voting.main.run_chat")
    def test_cli_chat_dispatches_runtime_options(self, run_chat: MagicMock) -> None:
        result = main.main(
            [
                "--cache-dir",
                "/cache",
                "chat",
                "--device",
                "cpu",
                "--max-new-tokens",
                "64",
            ]
        )

        self.assertEqual(result, 0)
        run_chat.assert_called_once_with(Path("/cache"), main.Device.CPU, 64)

    def test_cli_rejects_invalid_max_new_tokens(self) -> None:
        with self.assertRaises(SystemExit):
            main.main(["chat", "--max-new-tokens", "0"])

    def test_integer_parser_reports_nonnumeric_values(self) -> None:
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "'many' is not an integer; expected a positive integer",
        ):
            main.positive_integer("many")

    @patch("quadratic_voting.main.snapshot_download")
    def test_download_uses_pinned_official_artifact(self, download: MagicMock) -> None:
        download.return_value = "/cache/snapshot"

        result = main.download_model(Path("/cache"))

        self.assertEqual(result, Path("/cache/snapshot"))
        download.assert_called_once_with(
            repo_id="google/gemma-4-E2B-it-qat-q4_0-unquantized",
            revision="6befbaca7398925921802abd1f277b495b78b738",
            cache_dir=Path("/cache"),
        )

    @patch("quadratic_voting.main.snapshot_download")
    def test_download_reports_proxy_failure(self, download: MagicMock) -> None:
        download.side_effect = ProxyError("proxy unavailable")

        with self.assertRaisesRegex(main.RunnerError, "proxy unavailable"):
            main.download_model(Path("/cache"))

    @patch("quadratic_voting.main.snapshot_download")
    def test_cached_model_requires_complete_local_snapshot(
        self, download: MagicMock
    ) -> None:
        download.side_effect = LocalEntryNotFoundError("not cached")

        with self.assertRaisesRegex(main.RunnerError, "uv run python -m"):
            main.cached_model(Path("/cache"))

        download.assert_called_once_with(
            repo_id=main.GEMMA_IT.repository,
            revision=main.GEMMA_IT.revision,
            cache_dir=Path("/cache"),
            local_files_only=True,
        )

    def test_empty_real_hf_cache_reports_local_only_failure(self) -> None:
        with (
            tempfile.TemporaryDirectory() as cache_dir,
            self.assertRaisesRegex(
                main.RunnerError,
                "uv run python -m quadratic_voting.main download",
            ),
        ):
            main.cached_model(Path(cache_dir))

    def test_module_help_runs_with_active_interpreter(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "quadratic_voting.main", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Transformers and PyTorch", result.stdout)

    def test_auto_device_uses_accelerate_placement(self) -> None:
        self.assertIs(main.resolve_device(main.Device.AUTO), main.Device.AUTO)

    @patch("quadratic_voting.main.torch.cuda.is_available", return_value=False)
    def test_explicit_cuda_requires_available_device(
        self, is_available: MagicMock
    ) -> None:
        with self.assertRaisesRegex(main.RunnerError, "--device cpu"):
            main.resolve_device(main.Device.CUDA)

    @patch(
        "quadratic_voting.main.torch.cuda.mem_get_info",
        return_value=(8_000_000_000, 24_000_000_000),
    )
    @patch("quadratic_voting.main.torch.cuda.is_available", return_value=True)
    def test_explicit_cuda_requires_enough_free_memory(
        self, is_available: MagicMock, mem_get_info: MagicMock
    ) -> None:
        with self.assertRaisesRegex(main.RunnerError, "8.0 GB.*12.0 GB.*--device auto"):
            main.resolve_device(main.Device.CUDA)

    @patch("quadratic_voting.main.AutoModelForCausalLM.from_pretrained")
    @patch("quadratic_voting.main.AutoTokenizer.from_pretrained")
    @patch("quadratic_voting.main.resolve_device", return_value=main.Device.AUTO)
    def test_runtime_loads_local_bf16_checkpoint(
        self,
        resolve: MagicMock,
        load_tokenizer: MagicMock,
        load_model: MagicMock,
    ) -> None:
        model = load_model.return_value

        result = main.load_runtime(Path("/cache/snapshot"), main.Device.AUTO)

        self.assertEqual(result, (model, load_tokenizer.return_value))
        load_tokenizer.assert_called_once_with(
            Path("/cache/snapshot"), local_files_only=True, padding_side="left"
        )
        load_model.assert_called_once_with(
            Path("/cache/snapshot"),
            local_files_only=True,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.eval.assert_called_once_with()

    @unittest.skipUnless(
        os.environ.get("RUN_HF_INTEGRATION") == "1",
        "set RUN_HF_INTEGRATION=1 to verify pinned Hub metadata",
    )
    def test_pinned_transformers_metadata_and_chat_template(self) -> None:
        config = AutoConfig.from_pretrained(
            main.GEMMA_IT.repository,
            revision=main.GEMMA_IT.revision,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            main.GEMMA_IT.repository,
            revision=main.GEMMA_IT.revision,
            padding_side="left",
        )

        self.assertEqual(type(config).__name__, "Gemma4Config")
        model_class = cast(
            type[object], AutoModelForCausalLM._model_mapping[type(config)]
        )
        self.assertEqual(model_class.__name__, "Gemma4ForConditionalGeneration")
        rendered = cast(
            str,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hello"}],
                tokenize=False,
                add_generation_prompt=True,
            ),
        )
        self.assertIn("<|turn>user\nHello<turn|>", rendered)
        self.assertTrue(rendered.endswith("<|turn>model\n"))

    @patch("quadratic_voting.main.load_runtime")
    @patch("quadratic_voting.main.cached_model", return_value=Path("/cache/snapshot"))
    def test_chat_preserves_conversation_and_generates_response(
        self, cached: MagicMock, load_runtime: MagicMock
    ) -> None:
        model = MagicMock()
        model.device = torch.device("cpu")
        model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        tokenizer = MagicMock()
        inputs = {"input_ids": torch.tensor([[1, 2]])}
        batch = MagicMock()
        batch.to.return_value = inputs
        tokenizer.apply_chat_template.return_value = batch
        tokenizer.decode.return_value = "Hello"
        load_runtime.return_value = (model, tokenizer)
        prompts = iter(["Hi", "/exit"])
        output: list[str] = []

        main.run_chat(
            Path("/cache"),
            main.Device.CPU,
            16,
            read_input=lambda _: next(prompts),
            write_output=output.append,
        )

        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Hi"}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        model.generate.assert_called_once_with(
            **inputs, max_new_tokens=16, do_sample=False
        )
        tokenizer.decode.assert_called_once()
        self.assertEqual(output, ["Gemma: Hello"])

    @patch("quadratic_voting.main.load_runtime")
    @patch("quadratic_voting.main.cached_model", return_value=Path("/cache/snapshot"))
    def test_chat_reports_generation_failure(
        self, cached: MagicMock, load_runtime: MagicMock
    ) -> None:
        model = MagicMock()
        model.device = torch.device("cpu")
        model.generate.side_effect = RuntimeError("out of memory")
        tokenizer = MagicMock()
        batch = MagicMock()
        batch.to.return_value = {"input_ids": torch.tensor([[1, 2]])}
        tokenizer.apply_chat_template.return_value = batch
        load_runtime.return_value = (model, tokenizer)

        with self.assertRaisesRegex(main.RunnerError, "out of memory"):
            main.run_chat(
                Path("/cache"),
                main.Device.CPU,
                16,
                read_input=lambda _: "Hi",
            )

    @patch("quadratic_voting.main.load_runtime")
    @patch("quadratic_voting.main.cached_model", return_value=Path("/cache/snapshot"))
    def test_chat_rejects_conversation_beyond_model_context(
        self, cached: MagicMock, load_runtime: MagicMock
    ) -> None:
        model = MagicMock()
        model.device = torch.device("cpu")
        tokenizer = MagicMock()
        batch = MagicMock()
        batch.to.return_value = {
            "input_ids": torch.zeros(
                (1, main.MAX_CONTEXT_TOKENS - 8), dtype=torch.int64
            )
        }
        tokenizer.apply_chat_template.return_value = batch
        load_runtime.return_value = (model, tokenizer)

        with self.assertRaisesRegex(
            main.RunnerError, "exceeding the 131072-token model context.*Restart"
        ):
            main.run_chat(
                Path("/cache"),
                main.Device.CPU,
                16,
                read_input=lambda _: "Hi",
            )

        model.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
