import argparse
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from huggingface_hub.errors import LocalEntryNotFoundError

from quadratic_voting import main


class GemmaRunnerTests(unittest.TestCase):
    def test_help_marks_base_support_as_deferred(self) -> None:
        help_text = main.build_parser().format_help()

        self.assertIn("Base/non-instruction-tuned support is deferred", help_text)

    @patch(
        "quadratic_voting.main.download_model",
        return_value=Path("/cache/model.gguf"),
    )
    def test_cli_download_dispatches_with_selected_cache(
        self, download: object
    ) -> None:
        with patch("builtins.print") as output:
            result = main.main(["--cache-dir", "/cache", "download"])

        self.assertEqual(result, 0)
        download.assert_called_once_with(Path("/cache"))  # type: ignore[attr-defined]
        output.assert_called_once_with(
            "Downloaded pinned Gemma model to: /cache/model.gguf"
        )

    @patch("quadratic_voting.main.run_chat")
    def test_cli_chat_dispatches_runtime_options(self, run_chat: object) -> None:
        result = main.main(
            [
                "--cache-dir",
                "/cache",
                "chat",
                "--context-size",
                "4096",
                "--gpu-layers",
                "12",
            ]
        )

        self.assertEqual(result, 0)
        run_chat.assert_called_once_with(  # type: ignore[attr-defined]
            Path("/cache"), 4096, 12
        )

    def test_cli_rejects_negative_gpu_layers(self) -> None:
        with self.assertRaises(SystemExit):
            main.main(["chat", "--gpu-layers", "-1"])

    def test_integer_parsers_report_nonnumeric_values(self) -> None:
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "'many' is not an integer; expected a positive integer",
        ):
            main.positive_integer("many")
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "'some' is not an integer; expected zero or a positive integer",
        ):
            main.nonnegative_integer("some")

    @patch("quadratic_voting.main.hf_hub_download")
    def test_download_uses_pinned_official_artifact(self, download: object) -> None:
        download.return_value = "/cache/model.gguf"  # type: ignore[attr-defined]

        result = main.download_model(Path("/cache"))

        self.assertEqual(result, Path("/cache/model.gguf"))
        download.assert_called_once_with(  # type: ignore[attr-defined]
            repo_id="google/gemma-4-E2B-it-qat-q4_0-gguf",
            filename="gemma-4-E2B_q4_0-it.gguf",
            revision="675cff42a74c774d6cb76f76d8eacb49b48c9b93",
            cache_dir=Path("/cache"),
        )

    @patch("quadratic_voting.main.hf_hub_download")
    def test_cached_model_reports_download_command(self, download: object) -> None:
        download.side_effect = LocalEntryNotFoundError("not cached")  # type: ignore[attr-defined]

        with self.assertRaisesRegex(main.RunnerError, "uv run python -m"):
            main.cached_model(Path("/cache"))

    def test_empty_real_hf_cache_reports_local_only_failure(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            with self.assertRaisesRegex(
                main.RunnerError,
                "uv run python -m quadratic_voting.main download",
            ):
                main.cached_model(Path(cache_dir))

    def test_module_help_runs_with_active_interpreter(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "quadratic_voting.main", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("official instruction-tuned Gemma 4", result.stdout)
        self.assertIn("Base/non-instruction-tuned support is deferred", result.stdout)

    def test_nix_llama_cli_starts(self) -> None:
        if "IN_NIX_SHELL" not in os.environ:
            self.skipTest("outside nix develop; Nix-provided llama-cli unavailable")

        llama_cli = shutil.which("llama-cli")
        self.assertIsNotNone(
            llama_cli, "llama-cli must be available inside the Nix development shell"
        )
        assert llama_cli is not None

        result = subprocess.run(
            [llama_cli, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("version:", result.stdout + result.stderr)

    @patch("quadratic_voting.main.subprocess.run")
    @patch("quadratic_voting.main.cached_model", return_value=Path("/cache/model.gguf"))
    @patch("quadratic_voting.main.shutil.which", return_value="/bin/llama-cli")
    def test_chat_uses_conversation_mode_and_embedded_template(
        self, which: object, cached: object, run: object
    ) -> None:
        main.run_chat(Path("/cache"), context_size=4096, gpu_layers=12)

        run.assert_called_once_with(  # type: ignore[attr-defined]
            [
                "/bin/llama-cli",
                "--model",
                "/cache/model.gguf",
                "--conversation",
                "--ctx-size",
                "4096",
                "--gpu-layers",
                "12",
            ],
            check=True,
        )

    @patch("quadratic_voting.main.shutil.which", return_value=None)
    def test_chat_reports_missing_llama_cli(self, which: object) -> None:
        with self.assertRaisesRegex(main.RunnerError, "nix develop"):
            main.run_chat(Path("/cache"), context_size=4096, gpu_layers=0)

    @patch("quadratic_voting.main.subprocess.run")
    @patch("quadratic_voting.main.cached_model", return_value=Path("/cache/model.gguf"))
    @patch("quadratic_voting.main.shutil.which", return_value="/bin/llama-cli")
    def test_chat_reports_subprocess_failure(
        self, which: object, cached: object, run: object
    ) -> None:
        run.side_effect = subprocess.CalledProcessError(7, ["llama-cli"])  # type: ignore[attr-defined]

        with self.assertRaisesRegex(
            main.RunnerError, "status 7.*uv run python -m quadratic_voting.main chat"
        ):
            main.run_chat(Path("/cache"), context_size=4096, gpu_layers=0)

    @patch("quadratic_voting.main.subprocess.run")
    @patch("quadratic_voting.main.cached_model", return_value=Path("/cache/model.gguf"))
    @patch(
        "quadratic_voting.main.shutil.which", return_value="/nix/store/bin/llama-cli"
    )
    def test_chat_reports_subprocess_startup_failure(
        self, which: object, cached: object, run: object
    ) -> None:
        run.side_effect = OSError("permission denied")  # type: ignore[attr-defined]

        with self.assertRaisesRegex(
            main.RunnerError,
            "/nix/store/bin/llama-cli.*permission denied.*executable.*uv run python -m",
        ):
            main.run_chat(Path("/cache"), context_size=4096, gpu_layers=0)


if __name__ == "__main__":
    unittest.main()
