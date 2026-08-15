import os
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import ANY, MagicMock, call, patch

import torch
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from llm_runtime import (
    ChatMessage,
    FinishReason,
    GenerationResult,
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
    TransformersRuntime,
    TransformersRuntimeError,
    create_transformers_runtime,
    download_transformers_artifact,
)


class TinyTokenizer:
    """Concrete no-download tokenizer exercising the production runtime boundary."""

    eos_token_id = 7

    def apply_chat_template(self, *_args: object, **_kwargs: object) -> BatchEncoding:
        return BatchEncoding({"input_ids": torch.tensor([[1, 2]], dtype=torch.long)})

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class WhitespaceTokenizer(TinyTokenizer):
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert token_ids
        assert skip_special_tokens
        return " \n exact response \t"


class TinySamplingModel(torch.nn.Module):
    """Concrete stochastic torch model using the global RNG like HF generation."""

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def generate(self, **kwargs: object) -> torch.Tensor:
        input_ids = cast(torch.Tensor, kwargs["input_ids"])
        max_new_tokens = cast(int, kwargs["max_new_tokens"])
        probabilities = torch.tensor([0.03, 0.07, 0.11, 0.13, 0.17, 0.19, 0.23, 0.07])
        generated = torch.multinomial(probabilities, max_new_tokens, replacement=True)
        return torch.cat((input_ids, generated.unsqueeze(0)), dim=1)


class FailingSamplingModel(TinySamplingModel):
    def generate(self, **_: object) -> torch.Tensor:
        torch.rand(4)
        if torch.cuda.is_available():
            torch.rand(4, device="cuda")
        raise RuntimeError("sk-secret fixture generation failure")


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
        self.assertEqual(response.raw_text, " hello ")
        self.assertEqual(response.prompt_token_count, 2)
        self.assertEqual(response.completion_token_ids, (3, 4))
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

    def test_generation_settings_reject_invalid_sampling_values(self) -> None:
        invalid_settings = (
            ({"max_new_tokens": True}, "max_new_tokens"),
            ({"max_new_tokens": 1.5}, "max_new_tokens"),
            ({"temperature": True}, "temperature"),
            ({"temperature": "0.5"}, "temperature"),
            ({"top_p": 0.0}, "top_p"),
            ({"top_p": 1.01}, "top_p"),
            ({"top_p": float("inf")}, "top_p"),
            ({"top_p": float("nan")}, "top_p"),
            ({"top_k": 0}, "top_k"),
            ({"top_k": -1}, "top_k"),
            ({"top_k": 1.5}, "top_k"),
            ({"top_k": True}, "top_k"),
            ({"seed": -1}, "seed"),
            ({"seed": 1.5}, "seed"),
            ({"seed": True}, "seed"),
            ({"seed": 2**64}, "seed"),
        )

        for overrides, field_name in invalid_settings:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, field_name):
                    GenerationSettings(**({"max_new_tokens": 16} | overrides))  # type: ignore[arg-type]

    def test_generation_result_rejects_coercion_and_unsafe_metadata(self) -> None:
        valid = {
            "raw_text": "response",
            "prompt_token_count": 1,
            "completion_token_count": 1,
            "completion_token_ids": (3,),
            "finish_reason": FinishReason.EOS,
            "duration_ms": 1,
            "diagnostics": {"status": "ok"},
        }
        for override, field_name in (
            ({"prompt_token_count": True}, "prompt_token_count"),
            ({"completion_token_count": -1}, "completion_token_count"),
            ({"completion_token_ids": (False,)}, "completion_token_ids"),
            ({"duration_ms": 1.5}, "duration_ms"),
            ({"diagnostics": {"status": float("nan")}}, "diagnostics"),
            ({"diagnostics": {"authorization": "Bearer secret"}}, "diagnostics"),
        ):
            with self.subTest(field=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    GenerationResult(**(valid | override))  # type: ignore[arg-type]

        diagnostics = {"status": "ok"}
        result = GenerationResult(**(valid | {"diagnostics": diagnostics}))  # type: ignore[arg-type]
        diagnostics["status"] = "mutated"
        self.assertEqual(result.diagnostics, {"status": "ok"})
        with self.assertRaises(TypeError):
            result.diagnostics["status"] = "mutated"  # type: ignore[index]

    def test_sampling_options_and_seed_are_forwarded(self) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_E2B_IT,
                ProviderId.LOCAL,
                QuantizationId.BF16,
            ),
        )
        model = MagicMock()
        model.device = torch.device("cpu")
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        tokenizer = MagicMock()
        batch = MagicMock()
        batch.to.return_value = {"input_ids": torch.tensor([[1, 2]])}
        tokenizer.apply_chat_template.return_value = batch
        tokenizer.decode.return_value = "sampled"
        runtime = TransformersRuntime(route, model, tokenizer)

        response = runtime.generate(
            [ChatMessage(MessageRole.USER, "Hi")],
            GenerationSettings(
                max_new_tokens=16,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                seed=123,
            ),
        )

        self.assertEqual(response.raw_text, "sampled")
        generation_kwargs = model.generate.call_args.kwargs
        self.assertEqual(generation_kwargs["input_ids"].shape, (1, 2))
        self.assertEqual(generation_kwargs["max_new_tokens"], 16)
        self.assertIs(generation_kwargs["do_sample"], True)
        self.assertEqual(generation_kwargs["temperature"], 0.7)
        self.assertEqual(generation_kwargs["top_p"], 0.9)
        self.assertEqual(generation_kwargs["top_k"], 40)
        self.assertNotIn("generator", generation_kwargs)

    def _concrete_runtime(
        self,
        model: TinySamplingModel | None = None,
        tokenizer: TinyTokenizer | None = None,
    ) -> TransformersRuntime:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_E2B_IT, ProviderId.LOCAL, QuantizationId.BF16
            ),
        )
        return TransformersRuntime(
            route,
            model or TinySamplingModel(),
            cast(PreTrainedTokenizerBase, tokenizer or TinyTokenizer()),
        )

    def _cuda_states(self) -> tuple[torch.Tensor, ...]:
        if not torch.cuda.is_available():
            return ()
        return tuple(
            torch.cuda.get_rng_state(index).clone()
            for index in range(torch.cuda.device_count())
        )

    def test_concrete_runtime_seed_reproducibility_and_metadata(self) -> None:
        runtime = self._concrete_runtime()
        settings = GenerationSettings(max_new_tokens=12, temperature=1.0, seed=481)

        first = runtime.generate([ChatMessage(MessageRole.USER, "Hi")], settings)
        second = runtime.generate([ChatMessage(MessageRole.USER, "Hi")], settings)
        different = runtime.generate(
            [ChatMessage(MessageRole.USER, "Hi")],
            GenerationSettings(max_new_tokens=12, temperature=1.0, seed=482),
        )

        self.assertEqual(first.completion_token_ids, second.completion_token_ids)
        self.assertEqual(first.raw_text, second.raw_text)
        self.assertNotEqual(first.completion_token_ids, different.completion_token_ids)
        self.assertEqual(first.prompt_token_count, 2)
        self.assertEqual(first.completion_token_count, 12)
        self.assertEqual(len(first.completion_token_ids or ()), 12)
        expected_finish = (
            FinishReason.EOS
            if first.completion_token_ids and first.completion_token_ids[-1] == 7
            else FinishReason.LENGTH
        )
        self.assertIs(first.finish_reason, expected_finish)
        self.assertGreaterEqual(first.duration_ms, 0)
        self.assertEqual(first.diagnostics, {})

    def test_concrete_runtime_preserves_exact_decoded_text_and_max_uint64_seed(
        self,
    ) -> None:
        runtime = self._concrete_runtime(tokenizer=WhitespaceTokenizer())
        result = runtime.generate(
            [ChatMessage(MessageRole.USER, "Hi")],
            GenerationSettings(
                max_new_tokens=4,
                temperature=1.0,
                seed=2**64 - 1,
            ),
        )

        self.assertEqual(result.raw_text, " \n exact response \t")
        self.assertEqual(result.completion_token_count, 4)

    def test_concrete_runtime_restores_ambient_rng_on_success_and_exception(
        self,
    ) -> None:
        messages = [ChatMessage(MessageRole.USER, "Hi")]
        settings = GenerationSettings(max_new_tokens=6, temperature=1.0, seed=99)
        for model, should_raise in (
            (TinySamplingModel(), False),
            (FailingSamplingModel(), True),
        ):
            with self.subTest(should_raise=should_raise):
                torch.manual_seed(123456)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(654321)
                cpu_before = torch.random.get_rng_state().clone()
                cuda_before = self._cuda_states()
                if should_raise:
                    with self.assertRaisesRegex(
                        RuntimeError, "Local generation failed.*no response"
                    ) as rejected:
                        self._concrete_runtime(model).generate(messages, settings)
                    self.assertNotIn("sk-secret", str(rejected.exception))
                    self.assertIsNone(rejected.exception.__cause__)
                else:
                    self._concrete_runtime(model).generate(messages, settings)
                self.assertTrue(torch.equal(cpu_before, torch.random.get_rng_state()))
                cuda_after = self._cuda_states()
                self.assertEqual(len(cuda_before), len(cuda_after))
                for before, after in zip(cuda_before, cuda_after, strict=True):
                    self.assertTrue(torch.equal(before, after))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_concrete_runtime_restores_every_cuda_rng_on_success_and_exception(
        self,
    ) -> None:
        messages = [ChatMessage(MessageRole.USER, "Hi")]
        settings = GenerationSettings(max_new_tokens=6, temperature=1.0, seed=99)
        for model, should_raise in (
            (TinySamplingModel(), False),
            (FailingSamplingModel(), True),
        ):
            with self.subTest(should_raise=should_raise):
                torch.cuda.manual_seed_all(987654)
                before = self._cuda_states()
                if should_raise:
                    with self.assertRaisesRegex(
                        RuntimeError, "Local generation failed.*no response"
                    ) as rejected:
                        self._concrete_runtime(model).generate(messages, settings)
                    self.assertNotIn("sk-secret", str(rejected.exception))
                    self.assertIsNone(rejected.exception.__cause__)
                else:
                    self._concrete_runtime(model).generate(messages, settings)
                after = self._cuda_states()
                self.assertGreater(len(before), 0)
                self.assertEqual(len(before), len(after))
                for expected, actual in zip(before, after, strict=True):
                    self.assertTrue(torch.equal(expected, actual))

    def test_sampling_filters_are_omitted_for_greedy_generation(self) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_E2B_IT,
                ProviderId.LOCAL,
                QuantizationId.BF16,
            ),
        )
        model = MagicMock()
        model.device = torch.device("cpu")
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        tokenizer = MagicMock()
        batch = MagicMock()
        batch.to.return_value = {"input_ids": torch.tensor([[1, 2]])}
        tokenizer.apply_chat_template.return_value = batch
        tokenizer.decode.return_value = "greedy"
        runtime = TransformersRuntime(route, model, tokenizer)

        runtime.generate(
            [ChatMessage(MessageRole.USER, "Hi")],
            GenerationSettings(max_new_tokens=16, top_p=0.9, top_k=40),
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
