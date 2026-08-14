from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from huggingface_hub.errors import LocalEntryNotFoundError

from quadratic_voting import main


class GemmaRunnerTests(unittest.TestCase):
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

        with self.assertRaisesRegex(main.RunnerError, "main download"):
            main.cached_model(Path("/cache"))

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

        with self.assertRaisesRegex(main.RunnerError, "status 7"):
            main.run_chat(Path("/cache"), context_size=4096, gpu_layers=0)


if __name__ == "__main__":
    unittest.main()
