import os
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import ANY, MagicMock, patch

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llm_runtime import (
    ChatMessage,
    GenerationSettings,
    LocalTransformersRoute,
    MessageRole,
    ModelId,
    ProviderId,
    QuantizationId,
    resolve_route,
)
from llm_runtime.transformers import (
    Device,
    create_transformers_runtime,
    download_transformers_artifact,
)


class TransformersRuntimeTests(unittest.TestCase):
    @patch("llm_runtime.transformers.AutoModelForCausalLM.from_pretrained")
    @patch("llm_runtime.transformers.AutoTokenizer.from_pretrained")
    @patch("llm_runtime.transformers.snapshot_download")
    def test_pinned_factory_delegates_generation_and_exposes_activations(
        self,
        snapshot_download: MagicMock,
        load_tokenizer: MagicMock,
        load_model: MagicMock,
    ) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_E2B_IT,
                ProviderId.LOCAL,
                QuantizationId.BF16,
            ),
        )
        snapshot_download.return_value = "/cache/snapshot"
        model = load_model.return_value
        model.device = torch.device("cpu")
        model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        tokenizer = load_tokenizer.return_value
        batch = MagicMock()
        batch.to.return_value = {"input_ids": torch.tensor([[1, 2]])}
        tokenizer.apply_chat_template.return_value = batch
        tokenizer.decode.return_value = " hello "

        downloaded = download_transformers_artifact(route, Path("/cache"))
        runtime = create_transformers_runtime(
            route, cache_dir=Path("/cache"), device=Device.AUTO
        )
        response = runtime.generate(
            [ChatMessage(MessageRole.USER, "Hi")],
            GenerationSettings(max_new_tokens=16),
        )

        self.assertEqual(downloaded, Path("/cache/snapshot"))
        self.assertEqual(response, "hello")
        self.assertIs(runtime.model, model)
        self.assertIs(runtime.tokenizer, tokenizer)
        self.assertEqual(
            snapshot_download.call_args_list[0].kwargs,
            {
                "repo_id": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
                "revision": "6befbaca7398925921802abd1f277b495b78b738",
                "cache_dir": Path("/cache"),
            },
        )
        self.assertEqual(
            snapshot_download.call_args_list[1].kwargs["revision"],
            route.artifact.revision,
        )
        load_model.assert_called_once_with(
            Path("/cache/snapshot"),
            local_files_only=True,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.generate.assert_called_once_with(
            input_ids=ANY,
            max_new_tokens=16,
            do_sample=False,
        )

    @unittest.skipUnless(
        os.environ.get("RUN_HF_INTEGRATION") == "1",
        "set RUN_HF_INTEGRATION=1 to verify pinned Hub metadata",
    )
    def test_pinned_bf16_metadata_and_text_chat_template(self) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_E2B_IT,
                ProviderId.LOCAL,
                QuantizationId.BF16,
            ),
        )
        config = AutoConfig.from_pretrained(
            route.artifact.repository, revision=route.artifact.revision
        )
        tokenizer = AutoTokenizer.from_pretrained(
            route.artifact.repository,
            revision=route.artifact.revision,
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
        self.assertIn("Hello", rendered)


if __name__ == "__main__":
    unittest.main()
