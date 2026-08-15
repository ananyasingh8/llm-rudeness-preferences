import os
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import ANY, MagicMock, call, patch

import torch
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llm_runtime import (
    ChatMessage,
    GenerationSettings,
    LocalTransformersRoute,
    MessageRole,
    ModelId,
    ProviderId,
    QuantizationId,
    TorchDTypeId,
    resolve_route,
)
from llm_runtime.transformers import (
    Device,
    TransformersRuntimeError,
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

    @patch("llm_runtime.transformers._torch_dtype")
    @patch("llm_runtime.transformers.BitsAndBytesConfig")
    @patch("llm_runtime.transformers.torch.cuda.is_available", return_value=True)
    @patch("llm_runtime.transformers.importlib.util.find_spec", return_value=object())
    @patch("llm_runtime.transformers.AutoModelForCausalLM.from_pretrained")
    @patch("llm_runtime.transformers.AutoTokenizer.from_pretrained")
    @patch("llm_runtime.transformers.snapshot_download", return_value="/cache/snapshot")
    def test_bitsandbytes_factory_uses_exact_recipe_and_cuda_placement(
        self,
        snapshot_download: MagicMock,
        load_tokenizer: MagicMock,
        load_model: MagicMock,
        _find_spec: MagicMock,
        _cuda_available: MagicMock,
        bnb_config: MagicMock,
        torch_dtype: MagicMock,
    ) -> None:
        torch_dtype.side_effect = {
            TorchDTypeId.BFLOAT16: torch.bfloat16,
            TorchDTypeId.UINT8: torch.uint8,
        }.__getitem__
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_31B_IT,
                ProviderId.LOCAL,
                QuantizationId.BITSANDBYTES_FP4,
            ),
        )
        model = load_model.return_value
        model.hf_device_map = {"": 0}

        runtime = create_transformers_runtime(
            route, cache_dir=Path("/cache"), device=Device.AUTO
        )

        bnb_config.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8,
            bnb_4bit_use_double_quant=False,
        )
        self.assertEqual(
            torch_dtype.call_args_list,
            [call(TorchDTypeId.BFLOAT16), call(TorchDTypeId.UINT8)],
        )
        load_model.assert_called_once_with(
            Path("/cache/snapshot"),
            local_files_only=True,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            quantization_config=bnb_config.return_value,
        )
        self.assertEqual(runtime.placement.resolved_device_map, (("", "0"),))
        self.assertFalse(runtime.placement.has_cpu_or_offload)
        self.assertEqual(
            snapshot_download.call_args.kwargs["revision"],
            "842da3794eaa0b77d5f08bae87a17459d91ff475",
        )

    @patch("llm_runtime.transformers.AutoModelForCausalLM.from_pretrained")
    @patch("llm_runtime.transformers.snapshot_download")
    @patch("llm_runtime.transformers.importlib.util.find_spec", return_value=None)
    def test_missing_bitsandbytes_fails_before_cache_or_model_load(
        self,
        _find_spec: MagicMock,
        snapshot_download: MagicMock,
        load_model: MagicMock,
    ) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_31B_IT,
                ProviderId.LOCAL,
                QuantizationId.BITSANDBYTES_FP4,
            ),
        )
        with self.assertRaisesRegex(
            TransformersRuntimeError,
            "direct bitsandbytes dependency is absent.*no precision",
        ):
            create_transformers_runtime(route, cache_dir=Path("/cache"))
        snapshot_download.assert_not_called()
        load_model.assert_not_called()

    @patch("llm_runtime.transformers.BitsAndBytesConfig")
    @patch("llm_runtime.transformers.torch.cuda.is_available", return_value=True)
    @patch("llm_runtime.transformers.importlib.util.find_spec", return_value=object())
    @patch("llm_runtime.transformers.AutoModelForCausalLM.from_pretrained")
    @patch("llm_runtime.transformers.AutoTokenizer.from_pretrained")
    @patch("llm_runtime.transformers.snapshot_download", return_value="/cache/snapshot")
    def test_bitsandbytes_cpu_offload_is_rejected_after_load(
        self,
        _snapshot: MagicMock,
        _tokenizer: MagicMock,
        load_model: MagicMock,
        _find_spec: MagicMock,
        _cuda_available: MagicMock,
        _bnb_config: MagicMock,
    ) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_31B_IT,
                ProviderId.LOCAL,
                QuantizationId.BITSANDBYTES_FP4,
            ),
        )
        load_model.return_value.hf_device_map = {"model.layers.40": "cpu"}
        with self.assertRaisesRegex(
            TransformersRuntimeError, "non-CUDA placement.*model.layers.40=cpu"
        ):
            create_transformers_runtime(route, cache_dir=Path("/cache"))

    @patch("llm_runtime.transformers.BitsAndBytesConfig")
    @patch("llm_runtime.transformers.torch.cuda.is_available", return_value=True)
    @patch("llm_runtime.transformers.importlib.util.find_spec", return_value=object())
    @patch("llm_runtime.transformers.AutoModelForCausalLM.from_pretrained")
    @patch("llm_runtime.transformers.AutoTokenizer.from_pretrained")
    @patch("llm_runtime.transformers.snapshot_download", return_value="/cache/snapshot")
    def test_bitsandbytes_missing_device_map_verifies_named_tensors(
        self,
        _snapshot: MagicMock,
        _tokenizer: MagicMock,
        load_model: MagicMock,
        _find_spec: MagicMock,
        _cuda_available: MagicMock,
        _bnb_config: MagicMock,
    ) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_31B_IT,
                ProviderId.LOCAL,
                QuantizationId.BITSANDBYTES_FP4,
            ),
        )
        model = load_model.return_value
        model.hf_device_map = {}
        parameter = MagicMock()
        parameter.device = torch.device("cuda:0")
        buffer = MagicMock()
        buffer.device = torch.device("cuda:0")
        model.named_parameters.return_value = [("model.layers.40.weight", parameter)]
        model.named_buffers.return_value = [("model.rotary.inv_freq", buffer)]

        runtime = create_transformers_runtime(route, cache_dir=Path("/cache"))

        self.assertEqual(
            runtime.placement.resolved_device_map,
            (("<all-parameters-and-buffers>", "cuda:0"),),
        )
        self.assertFalse(runtime.placement.has_cpu_or_offload)

    @patch("llm_runtime.transformers.BitsAndBytesConfig")
    @patch("llm_runtime.transformers.torch.cuda.is_available", return_value=True)
    @patch("llm_runtime.transformers.importlib.util.find_spec", return_value=object())
    @patch("llm_runtime.transformers.AutoModelForCausalLM.from_pretrained")
    @patch("llm_runtime.transformers.AutoTokenizer.from_pretrained")
    @patch("llm_runtime.transformers.snapshot_download", return_value="/cache/snapshot")
    def test_bitsandbytes_missing_device_map_rejects_cpu_tensor(
        self,
        _snapshot: MagicMock,
        _tokenizer: MagicMock,
        load_model: MagicMock,
        _find_spec: MagicMock,
        _cuda_available: MagicMock,
        _bnb_config: MagicMock,
    ) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_31B_IT,
                ProviderId.LOCAL,
                QuantizationId.BITSANDBYTES_FP4,
            ),
        )
        model = load_model.return_value
        model.hf_device_map = None
        parameter = MagicMock()
        parameter.device = torch.device("cpu")
        model.named_parameters.return_value = [("model.layers.40.weight", parameter)]
        model.named_buffers.return_value = []

        with self.assertRaisesRegex(
            TransformersRuntimeError,
            "direct tensor inspection found non-CUDA placement.*parameter:model.layers.40.weight=cpu",
        ):
            create_transformers_runtime(route, cache_dir=Path("/cache"))

    @patch("llm_runtime.transformers.snapshot_download")
    @patch("llm_runtime.transformers.torch.cuda.is_available", return_value=False)
    @patch("llm_runtime.transformers.importlib.util.find_spec", return_value=object())
    def test_bitsandbytes_auto_without_cuda_fails_before_cache_load(
        self,
        _find_spec: MagicMock,
        _cuda_available: MagicMock,
        snapshot_download: MagicMock,
    ) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_31B_IT,
                ProviderId.LOCAL,
                QuantizationId.BITSANDBYTES_FP4,
            ),
        )
        with self.assertRaisesRegex(
            TransformersRuntimeError, "automatic placement.*did not detect CUDA.*no CPU"
        ):
            create_transformers_runtime(route, cache_dir=Path("/cache"))
        snapshot_download.assert_not_called()

    @patch("llm_runtime.transformers.snapshot_download")
    @patch("llm_runtime.transformers.torch.cuda.is_available", return_value=False)
    def test_bitsandbytes_explicit_cuda_unavailable_has_cuda_only_remedy(
        self, _cuda_available: MagicMock, snapshot_download: MagicMock
    ) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_31B_IT,
                ProviderId.LOCAL,
                QuantizationId.BITSANDBYTES_FP4,
            ),
        )
        with self.assertRaises(TransformersRuntimeError) as raised:
            create_transformers_runtime(
                route, cache_dir=Path("/cache"), device=Device.CUDA
            )
        message = str(raised.exception)
        self.assertIn("retry with `--device cuda`", message)
        self.assertIn("does not support CPU, automatic", message)
        self.assertNotIn("select `cpu`", message)
        self.assertNotIn("select `auto`", message)
        snapshot_download.assert_not_called()

    @patch("llm_runtime.transformers.snapshot_download")
    @patch(
        "llm_runtime.transformers.torch.cuda.mem_get_info",
        return_value=(1, 24_000_000_000),
    )
    @patch("llm_runtime.transformers.torch.cuda.is_available", return_value=True)
    def test_bitsandbytes_low_cuda_memory_has_cuda_only_remedy(
        self,
        _cuda_available: MagicMock,
        _mem_get_info: MagicMock,
        snapshot_download: MagicMock,
    ) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_31B_IT,
                ProviderId.LOCAL,
                QuantizationId.BITSANDBYTES_FP4,
            ),
        )
        with self.assertRaises(TransformersRuntimeError) as raised:
            create_transformers_runtime(
                route, cache_dir=Path("/cache"), device=Device.CUDA
            )
        message = str(raised.exception)
        self.assertIn("required CUDA budget", message)
        self.assertIn("retry with `--device cuda`", message)
        self.assertIn("does not support CPU, automatic", message)
        self.assertNotIn("select `auto`", message)
        self.assertNotIn("select `cpu`", message)
        snapshot_download.assert_not_called()

    @patch("llm_runtime.transformers.snapshot_download")
    @patch("llm_runtime.transformers.importlib.util.find_spec", return_value=object())
    def test_bitsandbytes_explicit_cpu_fails_before_cache_load(
        self, _find_spec: MagicMock, snapshot_download: MagicMock
    ) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_31B_IT,
                ProviderId.LOCAL,
                QuantizationId.BITSANDBYTES_FP4,
            ),
        )
        with self.assertRaisesRegex(
            TransformersRuntimeError, "rejected CPU placement.*no CPU.*offload"
        ):
            create_transformers_runtime(
                route, cache_dir=Path("/cache"), device=Device.CPU
            )
        snapshot_download.assert_not_called()

    @patch("llm_runtime.transformers.AutoModelForCausalLM.from_pretrained")
    @patch(
        "llm_runtime.transformers.snapshot_download",
        side_effect=LocalEntryNotFoundError("missing"),
    )
    @patch("llm_runtime.transformers.torch.cuda.is_available", return_value=True)
    @patch("llm_runtime.transformers.importlib.util.find_spec", return_value=object())
    def test_bitsandbytes_missing_cache_reports_exact_convabuse_download_command(
        self,
        _find_spec: MagicMock,
        _cuda_available: MagicMock,
        _snapshot_download: MagicMock,
        load_model: MagicMock,
    ) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_31B_IT,
                ProviderId.LOCAL,
                QuantizationId.BITSANDBYTES_FP4,
            ),
        )
        with self.assertRaisesRegex(
            TransformersRuntimeError,
            "emotion_probing.main --experiment convabuse-31b-local-quant --cache-dir "
            "/cache/models download.*same run command",
        ):
            create_transformers_runtime(
                route, cache_dir=Path("/cache/models"), device=Device.AUTO
            )
        load_model.assert_not_called()

    @patch("llm_runtime.transformers.AutoModelForCausalLM.from_pretrained")
    @patch(
        "llm_runtime.transformers.snapshot_download",
        side_effect=LocalEntryNotFoundError("missing"),
    )
    def test_non_bitsandbytes_cache_remedies_select_the_exact_registered_route(
        self, _snapshot_download: MagicMock, load_model: MagicMock
    ) -> None:
        cases = (
            (
                ModelId.GEMMA_2_2B_IT,
                QuantizationId.BF16,
                "emotion_probing.main --experiment bailbench-2b --cache-dir "
                "'/cache with space' download",
            ),
            (
                ModelId.GEMMA_4_E2B_IT,
                QuantizationId.BF16,
                "quadratic_voting.main --model gemma-4-e2b-it --provider local "
                "--quantization bf16 --cache-dir '/cache with space' download",
            ),
        )
        for model_id, quantization_id, expected in cases:
            with self.subTest(model_id=model_id):
                route = cast(
                    LocalTransformersRoute,
                    resolve_route(model_id, ProviderId.LOCAL, quantization_id),
                )
                with self.assertRaises(TransformersRuntimeError) as raised:
                    create_transformers_runtime(
                        route, cache_dir=Path("/cache with space"), device=Device.AUTO
                    )
                self.assertIn(expected, str(raised.exception))
        load_model.assert_not_called()

    @patch(
        "llm_runtime.transformers.snapshot_download",
        side_effect=LocalEntryNotFoundError("download failed"),
    )
    def test_bitsandbytes_download_failure_reports_exact_experiment_command(
        self, _snapshot_download: MagicMock
    ) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_31B_IT,
                ProviderId.LOCAL,
                QuantizationId.BITSANDBYTES_FP4,
            ),
        )
        with self.assertRaisesRegex(
            TransformersRuntimeError,
            "download_transformers_artifact.*google/gemma-4-31B-it.*"
            "emotion_probing.main --experiment convabuse-31b-local-quant --cache-dir "
            "/cache/models download",
        ):
            download_transformers_artifact(route, Path("/cache/models"))

    @patch(
        "llm_runtime.transformers.snapshot_download",
        side_effect=LocalEntryNotFoundError("download failed"),
    )
    def test_compressed_download_failure_reports_exact_qv_route(
        self, _snapshot_download: MagicMock
    ) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_31B_IT,
                ProviderId.LOCAL,
                QuantizationId.W4A16_COMPRESSED_TENSORS,
            ),
        )
        with self.assertRaises(TransformersRuntimeError) as raised:
            download_transformers_artifact(route, Path("/cache with space"))
        self.assertIn(
            "quadratic_voting.main --model gemma-4-31b-it --provider local "
            "--quantization w4a16-compressed-tensors --cache-dir "
            "'/cache with space' download",
            str(raised.exception),
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
