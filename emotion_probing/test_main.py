import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from emotion_probing.main import (
    DEFAULT_EXPERIMENT,
    EXPERIMENTS,
    ExperimentId,
    ProbeError,
    _require_resume_compatible,
    _run_writer_lock,
    _stage_new_run,
    build_parser,
    build_input_fingerprints,
    build_run_provenance,
    emotion_scores,
    persist_cuda_peaks,
    prepare_run_dir,
    reset_cuda_peaks,
    resolve_probe_route,
    run_probe,
    validate_prompt_bounds_before_iteration,
    validate_probe_setup,
)
from llm_runtime import QuantizationId
from llm_runtime.transformers import Device, DevicePlacement


class FakeBatch(dict[str, torch.Tensor]):
    def to(self, _device: torch.device) -> "FakeBatch":
        return self


class FakeTokenizer:
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens

    def apply_chat_template(self, *_args: object, **_kwargs: object) -> FakeBatch:
        return FakeBatch(input_ids=torch.zeros((1, self.tokens), dtype=torch.long))


class TrackingHandle:
    def __init__(self, handle: object) -> None:
        self.handle = handle
        self.remove_count = 0

    def remove(self) -> None:
        self.remove_count += 1
        getattr(self.handle, "remove")()


class ProbeBlock(torch.nn.Module):
    def __init__(self, index: int, width: int) -> None:
        super().__init__()
        self.index = index
        self.width = width
        self.last_handle: TrackingHandle | None = None
        self.inference_mode_seen = False

    def register_forward_hook(  # type: ignore[override]
        self, hook: object, *args: object, **kwargs: object
    ) -> TrackingHandle:
        raw = super().register_forward_hook(hook, *args, **kwargs)  # type: ignore[arg-type]
        self.last_handle = TrackingHandle(raw)
        return self.last_handle

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.index == 40:
            self.inference_mode_seen = torch.is_inference_mode_enabled()
            result = torch.zeros_like(value)
            result[:, 0, 0] = 1
            result[:, -1, 1] = 1
            return result
        result = torch.zeros_like(value)
        result[:, -1, 0] = 1
        return result


class FakeProbeModel(torch.nn.Module):
    def __init__(self, width: int = 5_376, *, fail: bool = False) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.blocks = torch.nn.ModuleList([ProbeBlock(i, width) for i in range(42)])
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=self.blocks))
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=width))
        self.width = width
        self.fail = fail
        self.calls = 0
        self.last_use_cache: bool | None = None

    def forward(self, input_ids: torch.Tensor, *, use_cache: bool) -> torch.Tensor:
        self.calls += 1
        self.last_use_cache = use_cache
        value = torch.zeros((1, input_ids.shape[-1], self.width))
        for block in self.blocks:
            value = block(value)
            if self.fail and block.index == 40:
                raise RuntimeError("synthetic model failure")
        return value


def fake_runtime(tokens: int, *, width: int = 5_376, fail: bool = False) -> Any:
    return SimpleNamespace(
        model=FakeProbeModel(width, fail=fail),
        tokenizer=FakeTokenizer(tokens),
        placement=DevicePlacement(Device.CUDA, (("", "0"),), (), ()),
    )


class EmotionProbeTests(unittest.TestCase):
    def test_convabuse_flags_keep_prequantized_and_local_routes_distinct(self) -> None:
        self.assertIs(DEFAULT_EXPERIMENT, ExperimentId.CONVABUSE_31B)
        self.assertIs(
            EXPERIMENTS[ExperimentId.CONVABUSE_31B].quantization_id,
            QuantizationId.W4A16_COMPRESSED_TENSORS,
        )
        self.assertIs(
            EXPERIMENTS[ExperimentId.CONVABUSE_31B_LOCAL_QUANT].quantization_id,
            QuantizationId.BITSANDBYTES_FP4,
        )

    def _fingerprints(self) -> dict[str, object]:
        return {
            "dataset_manifest_sha256": "dataset-hash",
            "vector_source_path": "emotion_probing/vectors.npz",
            "vector_source_sha256": "vector-hash",
            "cluster_analysis_path": "emotion_probing/clusters.json",
            "cluster_analysis_sha256": "cluster-hash",
            "cluster_snapshot_expected": False,
            "cluster_snapshot_sha256": None,
            "probe_implementation_source": "emotion_probing/main.py",
            "probe_implementation_sha256": "probe-hash",
            "probe_implementation_revision": "sha256:probe-hash",
            "historical_extraction_model_revision": "unknown",
        }

    def _provenance(self, runtime: Any, limit: int | None = 1) -> dict[str, object]:
        config = EXPERIMENTS[ExperimentId.CONVABUSE_31B_LOCAL_QUANT]
        with patch(
            "emotion_probing.main.importlib.metadata.version", return_value="locked"
        ):
            return build_run_provenance(
                config,
                resolve_probe_route(config),
                runtime,
                limit,
                self._fingerprints(),
            )

    def test_exact_block_40_final_token_inference_and_bounds(self) -> None:
        runtime = fake_runtime(512)
        vectors = torch.zeros((2, 5_376))
        vectors[0, 0] = 1
        vectors[1, 1] = 1

        scores, token_count = emotion_scores(
            runtime,  # type: ignore[arg-type]
            [{"role": "user", "content": "hello"}],
            vectors,
            40,
            expected_width=5_376,
            token_limit=512,
        )

        model = runtime.model
        self.assertEqual(token_count, 512)
        self.assertEqual(scores, [0.0, 1.0])
        self.assertFalse(model.last_use_cache)
        self.assertTrue(model.blocks[40].inference_mode_seen)
        self.assertIsNone(model.blocks[39].last_handle)
        self.assertIsNone(model.blocks[41].last_handle)
        self.assertEqual(model.blocks[40].last_handle.remove_count, 1)
        self.assertEqual(len(model.blocks[40]._forward_hooks), 0)

    def test_513_tokens_rejected_before_model_call(self) -> None:
        runtime = fake_runtime(513)
        with self.assertRaisesRegex(ProbeError, "513 tokens exceed.*512-token"):
            emotion_scores(
                runtime,  # type: ignore[arg-type]
                [{"role": "user", "content": "too long"}],
                torch.zeros((1, 5_376)),
                40,
                expected_width=5_376,
                token_limit=512,
            )
        self.assertEqual(runtime.model.calls, 0)

    def test_all_prompt_bounds_are_preflighted_before_iteration(self) -> None:
        runtime = fake_runtime(2)
        runtime.tokenizer.apply_chat_template = MagicMock(
            side_effect=[
                FakeBatch(input_ids=torch.zeros((1, 512), dtype=torch.long)),
                FakeBatch(input_ids=torch.zeros((1, 513), dtype=torch.long)),
            ]
        )
        with self.assertRaisesRegex(
            ProbeError, "preflight failed.*task 1.*513 tokens exceed.*512-token"
        ):
            validate_prompt_bounds_before_iteration(
                runtime,
                [
                    [{"role": "user", "content": "accepted"}],
                    [{"role": "user", "content": "rejected"}],
                ],
                512,
            )
        self.assertEqual(runtime.model.calls, 0)

    def test_invalid_layer_and_batch_fail_before_model_call(self) -> None:
        runtime = fake_runtime(2)
        with self.assertRaisesRegex(ProbeError, "layer 42 is outside"):
            emotion_scores(
                runtime,  # type: ignore[arg-type]
                [],
                torch.zeros((1, 5_376)),
                42,
                expected_width=5_376,
                token_limit=512,
            )
        self.assertEqual(runtime.model.calls, 0)

        runtime.tokenizer.apply_chat_template = MagicMock(
            return_value=FakeBatch(input_ids=torch.zeros((2, 2), dtype=torch.long))
        )
        with self.assertRaisesRegex(ProbeError, "instead of batch one"):
            emotion_scores(
                runtime,  # type: ignore[arg-type]
                [],
                torch.zeros((1, 5_376)),
                40,
                expected_width=5_376,
                token_limit=512,
            )
        self.assertEqual(runtime.model.calls, 0)

    def test_hook_removed_once_on_model_and_shape_errors(self) -> None:
        for runtime, pattern in (
            (fake_runtime(2, fail=True), "synthetic model failure"),
            (fake_runtime(2, width=5_375), "observed.*5375.*expected.*5376"),
        ):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ProbeError, pattern):
                    emotion_scores(
                        runtime,  # type: ignore[arg-type]
                        [{"role": "user", "content": "hello"}],
                        torch.zeros((1, 5_376)),
                        40,
                        expected_width=5_376,
                        token_limit=512,
                    )
                handle = runtime.model.blocks[40].last_handle
                self.assertIsNotNone(handle)
                assert handle is not None
                self.assertEqual(handle.remove_count, 1)
                self.assertEqual(len(runtime.model.blocks[40]._forward_hooks), 0)

    def test_setup_and_provenance_include_route_placement_and_quantization(
        self,
    ) -> None:
        config = EXPERIMENTS[ExperimentId.CONVABUSE_31B_LOCAL_QUANT]
        route = resolve_probe_route(config)
        runtime = fake_runtime(2)
        validate_probe_setup(runtime, config, torch.zeros((171, 5_376)))  # type: ignore[arg-type]
        with patch(
            "emotion_probing.main.importlib.metadata.version", return_value="locked"
        ):
            provenance = build_run_provenance(
                config,
                route,
                runtime,
                1,  # type: ignore[arg-type]
                self._fingerprints(),
            )
        self.assertEqual(provenance["repository"], "google/gemma-4-31B-it")
        self.assertEqual(
            provenance["revision"], "842da3794eaa0b77d5f08bae87a17459d91ff475"
        )
        self.assertEqual(provenance["resolved_device_map"], {"": "0"})
        self.assertFalse(provenance["has_cpu_or_offload"])
        self.assertEqual(
            provenance["quantization_settings"],
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "fp4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_storage": "uint8",
                "bnb_4bit_use_double_quant": False,
            },
        )
        self.assertEqual(provenance["token_limit"], 512)
        self.assertEqual(provenance["batch_size"], 1)
        self.assertFalse(provenance["use_cache"])
        self.assertTrue(provenance["inference_mode"])

    def test_setup_rejects_model_or_vector_width_before_iteration(self) -> None:
        config = EXPERIMENTS[ExperimentId.CONVABUSE_31B_LOCAL_QUANT]
        with self.assertRaisesRegex(ProbeError, "loaded model reports width 5375"):
            validate_probe_setup(
                fake_runtime(2, width=5_375),  # type: ignore[arg-type]
                config,
                torch.zeros((171, 5_376)),
            )
        with self.assertRaisesRegex(ProbeError, "vectors have shape.*5375"):
            validate_probe_setup(
                fake_runtime(2),  # type: ignore[arg-type]
                config,
                torch.zeros((171, 5_375)),
            )

    def test_compatible_resume_preserves_original_provenance_and_peaks(self) -> None:
        config = EXPERIMENTS[ExperimentId.CONVABUSE_31B_LOCAL_QUANT]
        runtime = fake_runtime(2)
        provenance = self._provenance(runtime)
        provenance |= {
            "started": "original",
            "cuda_peak_memory_measured": True,
            "cuda_peak_allocated_bytes": 111,
            "cuda_peak_reserved_bytes": 222,
        }
        columns = ["example_id", "n_tokens", "score_emotion"]
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            run_dir = results / "2026-01-01_000000_convabuse-31b-local-quant"
            run_dir.mkdir()
            info_path = run_dir / "run_info.json"
            scores_path = run_dir / "scores.csv"
            info_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            scores_path.write_text(
                "example_id,n_tokens,score_emotion\n1,2,0.5\n", encoding="utf-8"
            )
            before_info = info_path.read_bytes()
            before_scores = scores_path.read_bytes()
            current = self._provenance(runtime)
            with patch("emotion_probing.main.RESULTS_DIR", results):
                selected = prepare_run_dir(config, True)
                with _run_writer_lock(run_dir):
                    keys = _require_resume_compatible(
                        run_dir, current, columns, ["example_id"]
                    )

            self.assertEqual(selected, run_dir)
            self.assertEqual(keys, {("1",)})
            self.assertEqual(info_path.read_bytes(), before_info)
            self.assertEqual(scores_path.read_bytes(), before_scores)

    def test_resume_mismatch_is_actionable_and_leaves_files_unchanged(self) -> None:
        config = EXPERIMENTS[ExperimentId.CONVABUSE_31B_LOCAL_QUANT]
        runtime = fake_runtime(2)
        previous = self._provenance(runtime)
        columns = ["example_id", "n_tokens", "score_emotion"]
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            run_dir = results / "2026-01-01_000000_convabuse-31b-local-quant"
            run_dir.mkdir()
            info_path = run_dir / "run_info.json"
            scores_path = run_dir / "scores.csv"
            info_path.write_text(json.dumps(previous, indent=2), encoding="utf-8")
            scores_path.write_text(
                "example_id,n_tokens,score_other\n1,2,0.5\n", encoding="utf-8"
            )
            before_info = info_path.read_bytes()
            before_scores = scores_path.read_bytes()
            current = self._provenance(runtime)
            current["revision"] = "different"
            with (
                patch("emotion_probing.main.RESULTS_DIR", results),
                self.assertRaisesRegex(
                    ProbeError,
                    "before any run file was modified.*revision.*scores.csv columns.*left unchanged",
                ),
            ):
                selected = prepare_run_dir(config, True)
                with _run_writer_lock(selected):
                    _require_resume_compatible(
                        selected, current, columns, ["example_id"]
                    )

            self.assertEqual(info_path.read_bytes(), before_info)
            self.assertEqual(scores_path.read_bytes(), before_scores)

    def test_noop_resume_orchestration_preserves_peaks_and_rows(self) -> None:
        runtime = fake_runtime(2)
        provenance = self._provenance(runtime)
        provenance |= {
            "started": "original",
            "cuda_peak_memory_measured": True,
            "cuda_peak_allocated_bytes": 333,
            "cuda_peak_reserved_bytes": 444,
        }
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            run_dir = results / "2026-01-01_000000_convabuse-31b-local-quant"
            run_dir.mkdir()
            info_path = run_dir / "run_info.json"
            scores_path = run_dir / "scores.csv"
            info_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            scores_path.write_text(
                "example_id,n_tokens,score_emotion\n1,2,0.5\n", encoding="utf-8"
            )
            before_info = info_path.read_bytes()
            before_scores = scores_path.read_bytes()
            with (
                patch("emotion_probing.main.RESULTS_DIR", results),
                patch(
                    "emotion_probing.main.load_vectors",
                    return_value=(["emotion"], torch.zeros((1, 5_376))),
                ),
                patch(
                    "emotion_probing.main.create_transformers_runtime",
                    return_value=runtime,
                ),
                patch(
                    "emotion_probing.main.load_dataset",
                    return_value=(
                        ["example_id"],
                        ["example_id"],
                        [
                            {
                                "row": {"example_id": "1"},
                                "messages": [{"role": "user", "content": "hi"}],
                            }
                        ],
                    ),
                ),
                patch(
                    "emotion_probing.main.importlib.metadata.version",
                    return_value="locked",
                ),
                patch(
                    "emotion_probing.main.build_input_fingerprints",
                    return_value=self._fingerprints(),
                ),
                patch("emotion_probing.main.reset_cuda_peaks") as reset,
            ):
                run_probe(
                    ExperimentId.CONVABUSE_31B_LOCAL_QUANT,
                    Path("/cache"),
                    Device.CUDA,
                    1,
                    True,
                )

            reset.assert_not_called()
            self.assertEqual(runtime.model.calls, 0)
            self.assertEqual(info_path.read_bytes(), before_info)
            self.assertEqual(scores_path.read_bytes(), before_scores)

    def test_partial_resume_appends_one_real_probe_row(self) -> None:
        runtime = fake_runtime(2)
        provenance = self._provenance(runtime, 2)
        existing_row = b"example_id,n_tokens,score_first,score_last\n1,2,0.25,0.75\n"
        vectors = torch.zeros((2, 5_376))
        vectors[0, 0] = 1
        vectors[1, 1] = 1
        tasks = [
            {"row": {"example_id": "1"}, "messages": []},
            {"row": {"example_id": "2"}, "messages": []},
        ]
        events: list[str] = []
        original_forward = runtime.model.forward

        def tracked_forward(*args: object, **kwargs: object) -> torch.Tensor:
            events.append("forward")
            return original_forward(*args, **kwargs)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            run_dir = results / "2026-01-01_000000_convabuse-31b-local-quant"
            run_dir.mkdir()
            info_path = run_dir / "run_info.json"
            scores_path = run_dir / "scores.csv"
            info_path.write_text(json.dumps(provenance), encoding="utf-8")
            scores_path.write_bytes(existing_row)
            with (
                patch("emotion_probing.main.RESULTS_DIR", results),
                patch(
                    "emotion_probing.main.load_vectors",
                    return_value=(["first", "last"], vectors),
                ),
                patch(
                    "emotion_probing.main.create_transformers_runtime",
                    return_value=runtime,
                ),
                patch(
                    "emotion_probing.main.load_dataset",
                    return_value=(["example_id"], ["example_id"], tasks),
                ),
                patch(
                    "emotion_probing.main.importlib.metadata.version",
                    return_value="locked",
                ),
                patch(
                    "emotion_probing.main.build_input_fingerprints",
                    return_value=self._fingerprints(),
                ),
                patch.object(runtime.model, "forward", side_effect=tracked_forward),
                patch("emotion_probing.main.torch.cuda.synchronize"),
                patch(
                    "emotion_probing.main.torch.cuda.reset_peak_memory_stats",
                    side_effect=lambda: events.append("reset"),
                ) as reset,
                patch(
                    "emotion_probing.main.torch.cuda.max_memory_allocated",
                    return_value=111,
                ),
                patch(
                    "emotion_probing.main.torch.cuda.max_memory_reserved",
                    return_value=222,
                ),
            ):
                run_probe(
                    ExperimentId.CONVABUSE_31B_LOCAL_QUANT,
                    Path("/cache"),
                    Device.CUDA,
                    2,
                    True,
                )

            self.assertEqual(runtime.model.calls, 1)
            self.assertEqual(events, ["reset", "forward"])
            reset.assert_called_once_with()
            written = scores_path.read_bytes()
            self.assertTrue(written.startswith(existing_row))
            rows = list(csv.DictReader(written.decode().splitlines()))
            self.assertEqual([row["example_id"] for row in rows], ["1", "2"])
            self.assertEqual(rows[1]["n_tokens"], "2")
            self.assertEqual(rows[1]["score_first"], "0.0")
            self.assertEqual(rows[1]["score_last"], "1.0")
            saved = json.loads(info_path.read_text())
            self.assertTrue(saved["cuda_peak_memory_measured"])
            self.assertEqual(saved["cuda_peak_allocated_bytes"], 111)
            self.assertEqual(saved["cuda_peak_reserved_bytes"], 222)

    def test_resume_rejects_changed_task_vector_and_probe_fingerprints(self) -> None:
        runtime = fake_runtime(2)
        previous = self._provenance(runtime)
        columns = ["example_id", "n_tokens", "score_emotion"]
        for field in (
            "dataset_manifest_sha256",
            "vector_source_sha256",
            "probe_implementation_sha256",
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                info_path = run_dir / "run_info.json"
                scores_path = run_dir / "scores.csv"
                info_path.write_text(json.dumps(previous, indent=2), encoding="utf-8")
                scores_path.write_text(
                    "example_id,n_tokens,score_emotion\n1,2,0.5\n",
                    encoding="utf-8",
                )
                before_info = info_path.read_bytes()
                before_scores = scores_path.read_bytes()
                current = self._provenance(runtime)
                current[field] = "changed"
                with (
                    self.assertRaisesRegex(ProbeError, field),
                    _run_writer_lock(run_dir),
                ):
                    _require_resume_compatible(
                        run_dir, current, columns, ["example_id"]
                    )
                self.assertEqual(info_path.read_bytes(), before_info)
                self.assertEqual(scores_path.read_bytes(), before_scores)

    def test_input_fingerprints_follow_actual_task_vector_and_probe_bytes(self) -> None:
        config = EXPERIMENTS[ExperimentId.CONVABUSE_31B_LOCAL_QUANT]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "emotion_probing"
            results = package / "gemotions" / "results" / "gemma4-31b"
            analysis_dir = results / "analysis"
            analysis_dir.mkdir(parents=True)
            vector_path = results / "emotion_vectors_layer40.npz"
            analysis_path = analysis_dir / "analysis_results.json"
            probe_path = package / "main.py"
            vector_path.write_bytes(b"vector-one")
            analysis_path.write_text(
                json.dumps({"40": {"clusters": {"0": ["joy"]}}}),
                encoding="utf-8",
            )
            probe_path.write_bytes(b"probe-one")
            tasks: list[dict[str, object]] = [
                {
                    "row": {"example_id": "1"},
                    "messages": [{"role": "user", "content": "one"}],
                }
            ]
            patches = (
                patch("emotion_probing.main.PACKAGE_DIR", package),
                patch("emotion_probing.main.GEMOTIONS_DIR", package / "gemotions"),
                patch("emotion_probing.main.GEMOTIONS_ANALYSIS_FILE", analysis_path),
                patch("emotion_probing.main.PROBE_SOURCE_FILE", probe_path),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                original = build_input_fingerprints(
                    config, ["example_id"], ["example_id"], tasks
                )
                changed_tasks: list[dict[str, object]] = [
                    {
                        "row": {"example_id": "1"},
                        "messages": [{"role": "user", "content": "changed"}],
                    }
                ]
                changed_task = build_input_fingerprints(
                    config,
                    ["example_id"],
                    ["example_id"],
                    changed_tasks,
                )
                vector_path.write_bytes(b"vector-two")
                changed_vector = build_input_fingerprints(
                    config, ["example_id"], ["example_id"], tasks
                )
                probe_path.write_bytes(b"probe-two")
                changed_probe = build_input_fingerprints(
                    config, ["example_id"], ["example_id"], tasks
                )

            self.assertNotEqual(
                original["dataset_manifest_sha256"],
                changed_task["dataset_manifest_sha256"],
            )
            self.assertNotEqual(
                original["vector_source_sha256"],
                changed_vector["vector_source_sha256"],
            )
            self.assertNotEqual(
                changed_vector["probe_implementation_sha256"],
                changed_probe["probe_implementation_sha256"],
            )
            self.assertEqual(
                original["historical_extraction_model_revision"], "unknown"
            )

    def test_concurrent_writer_fails_fast_and_lock_cleans_up_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            lock_path = run_dir / ".writer.lock"
            lock_path.write_text("other writer", encoding="utf-8")
            with self.assertRaisesRegex(
                ProbeError,
                "failed immediately.*Another process.*remove that exact lock",
            ):
                with _run_writer_lock(run_dir):
                    self.fail("existing writer lock must not be acquired")
            self.assertEqual(lock_path.read_text(encoding="utf-8"), "other writer")

            lock_path.unlink()
            with self.assertRaisesRegex(RuntimeError, "synthetic writer error"):
                with _run_writer_lock(run_dir):
                    raise RuntimeError("synthetic writer error")
            self.assertFalse(lock_path.exists())

    def test_cli_limit_requires_positive_integer(self) -> None:
        parser = build_parser()
        for value in ("0", "-1"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args(["run", "--limit", value])
        parsed = parser.parse_args(["run", "--limit", "1"])
        self.assertEqual(parsed.limit, 1)

    def test_resume_limit_must_match_original_limit(self) -> None:
        runtime = fake_runtime(2)
        columns = ["example_id", "n_tokens", "score_emotion"]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            info_path = run_dir / "run_info.json"
            scores_path = run_dir / "scores.csv"
            info_path.write_text(
                json.dumps(self._provenance(runtime, 1)), encoding="utf-8"
            )
            scores_path.write_text(
                "example_id,n_tokens,score_emotion\n1,2,0.5\n", encoding="utf-8"
            )
            _require_resume_compatible(
                run_dir, self._provenance(runtime, 1), columns, ["example_id"]
            )
            before = (info_path.read_bytes(), scores_path.read_bytes())
            with self.assertRaisesRegex(ProbeError, "invariants differ: limit"):
                _require_resume_compatible(
                    run_dir, self._provenance(runtime, 2), columns, ["example_id"]
                )
            self.assertEqual(before, (info_path.read_bytes(), scores_path.read_bytes()))

    def test_resume_rejects_missing_or_tampered_cluster_snapshot(self) -> None:
        runtime = fake_runtime(2)
        columns = ["example_id", "n_tokens", "score_emotion"]
        canonical = b'{"clusters":{"0":["joy"]}}'
        fingerprints = self._fingerprints()
        fingerprints["cluster_snapshot_expected"] = True
        fingerprints["cluster_snapshot_sha256"] = hashlib.sha256(canonical).hexdigest()
        with patch(
            "emotion_probing.main.importlib.metadata.version", return_value="locked"
        ):
            provenance = build_run_provenance(
                EXPERIMENTS[ExperimentId.CONVABUSE_31B_LOCAL_QUANT],
                resolve_probe_route(
                    EXPERIMENTS[ExperimentId.CONVABUSE_31B_LOCAL_QUANT]
                ),
                runtime,
                1,
                fingerprints,
            )
        for cluster_bytes, pattern in (
            (None, "missing"),
            (b'{"clusters":{}}', "digest"),
        ):
            with (
                self.subTest(pattern=pattern),
                tempfile.TemporaryDirectory() as directory,
            ):
                run_dir = Path(directory)
                info_path = run_dir / "run_info.json"
                scores_path = run_dir / "scores.csv"
                cluster_path = run_dir / "clusters.json"
                info_path.write_text(json.dumps(provenance), encoding="utf-8")
                scores_path.write_text(
                    "example_id,n_tokens,score_emotion\n1,2,0.5\n", encoding="utf-8"
                )
                if cluster_bytes is not None:
                    cluster_path.write_bytes(cluster_bytes)
                before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
                with self.assertRaisesRegex(ProbeError, pattern):
                    _require_resume_compatible(
                        run_dir, provenance, columns, ["example_id"]
                    )
                self.assertEqual(
                    before, {path.name: path.read_bytes() for path in run_dir.iterdir()}
                )

    def test_new_run_writes_exact_canonical_cluster_snapshot(self) -> None:
        payload = {"clusters": {"1": ["sad"], "0": ["joy"]}}
        canonical = b'{"clusters":{"0":["joy"],"1":["sad"]}}'
        provenance = {
            "cluster_snapshot_expected": True,
            "cluster_snapshot_sha256": hashlib.sha256(canonical).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "2026-01-01_000000_convabuse-31b-local-quant"
            run_dir.mkdir()
            with patch("emotion_probing.main.load_clusters", return_value=payload):
                _stage_new_run(
                    EXPERIMENTS[ExperimentId.CONVABUSE_31B_LOCAL_QUANT],
                    run_dir,
                    provenance,
                )
            self.assertEqual((run_dir / "clusters.json").read_bytes(), canonical)
            saved = json.loads((run_dir / "run_info.json").read_text())
            self.assertEqual(
                saved["cluster_snapshot_sha256"], hashlib.sha256(canonical).hexdigest()
            )

    def test_run_probe_rejects_corrupt_resume_csv_without_mutation(self) -> None:
        runtime = fake_runtime(2)
        provenance = self._provenance(runtime)
        malformed = {
            "truncated": "example_id,n_tokens,score_emotion\n1,2\n",
            "extra-field": "example_id,n_tokens,score_emotion\n1,2,0.5,extra\n",
            "blank-key": "example_id,n_tokens,score_emotion\n,2,0.5\n",
            "duplicate": ("example_id,n_tokens,score_emotion\n1,2,0.5\n1,2,0.6\n"),
            "invalid-token-count": "example_id,n_tokens,score_emotion\n1,nope,0.5\n",
            "non-positive-token-count": "example_id,n_tokens,score_emotion\n1,0,0.5\n",
            "non-numeric-score": "example_id,n_tokens,score_emotion\n1,2,nope\n",
            "non-finite": "example_id,n_tokens,score_emotion\n1,2,nan\n",
        }
        for label, csv_text in malformed.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                results = Path(directory)
                run_dir = results / "2026-01-01_000000_convabuse-31b-local-quant"
                run_dir.mkdir()
                info_path = run_dir / "run_info.json"
                scores_path = run_dir / "scores.csv"
                info_path.write_text(json.dumps(provenance), encoding="utf-8")
                scores_path.write_text(csv_text, encoding="utf-8")
                before = (info_path.read_bytes(), scores_path.read_bytes())
                with (
                    patch("emotion_probing.main.RESULTS_DIR", results),
                    patch(
                        "emotion_probing.main.load_vectors",
                        return_value=(["emotion"], torch.zeros((1, 5_376))),
                    ),
                    patch(
                        "emotion_probing.main.create_transformers_runtime",
                        return_value=runtime,
                    ),
                    patch(
                        "emotion_probing.main.load_dataset",
                        return_value=(
                            ["example_id"],
                            ["example_id"],
                            [{"row": {"example_id": "1"}, "messages": []}],
                        ),
                    ),
                    patch(
                        "emotion_probing.main.importlib.metadata.version",
                        return_value="locked",
                    ),
                    patch(
                        "emotion_probing.main.build_input_fingerprints",
                        return_value=self._fingerprints(),
                    ),
                    self.assertRaisesRegex(ProbeError, "Resume CSV validation failed"),
                ):
                    run_probe(
                        ExperimentId.CONVABUSE_31B_LOCAL_QUANT,
                        Path("/cache"),
                        Device.CUDA,
                        1,
                        True,
                    )
                self.assertEqual(
                    before, (info_path.read_bytes(), scores_path.read_bytes())
                )

    @patch("emotion_probing.main.torch.cuda.max_memory_reserved", return_value=222)
    @patch("emotion_probing.main.torch.cuda.max_memory_allocated", return_value=111)
    @patch("emotion_probing.main.torch.cuda.reset_peak_memory_stats")
    @patch("emotion_probing.main.torch.cuda.synchronize")
    def test_cuda_peaks_are_reset_synchronized_and_persisted(
        self,
        synchronize: MagicMock,
        reset: MagicMock,
        _allocated: MagicMock,
        _reserved: MagicMock,
    ) -> None:
        runtime = fake_runtime(2)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "run_info.json").write_text(
                json.dumps(
                    {
                        "cuda_peak_memory_measured": True,
                        "cuda_peak_allocated_bytes": 333,
                        "cuda_peak_reserved_bytes": 100,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(reset_cuda_peaks(runtime))  # type: ignore[arg-type]
            persist_cuda_peaks(run_dir, True)
            saved = json.loads((run_dir / "run_info.json").read_text())
        self.assertEqual(synchronize.call_count, 2)
        reset.assert_called_once_with()
        self.assertTrue(saved["cuda_peak_memory_measured"])
        self.assertEqual(saved["cuda_peak_allocated_bytes"], 333)
        self.assertEqual(saved["cuda_peak_reserved_bytes"], 222)

    @patch("emotion_probing.main.torch.cuda.max_memory_reserved", return_value=222)
    @patch("emotion_probing.main.torch.cuda.max_memory_allocated", return_value=111)
    @patch("emotion_probing.main.torch.cuda.synchronize")
    @patch("emotion_probing.main.os.replace", side_effect=OSError("replace failed"))
    def test_atomic_peak_replacement_failure_preserves_original_provenance(
        self,
        _replace: MagicMock,
        _synchronize: MagicMock,
        _allocated: MagicMock,
        _reserved: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            info_path = run_dir / "run_info.json"
            original = json.dumps(
                {
                    "started": "original",
                    "cuda_peak_memory_measured": True,
                    "cuda_peak_allocated_bytes": 333,
                    "cuda_peak_reserved_bytes": 444,
                },
                indent=2,
            ).encode()
            info_path.write_bytes(original)

            with self.assertRaisesRegex(
                ProbeError,
                "Atomic provenance update failed.*persisting synchronized CUDA.*"
                "previous run_info.json remains authoritative",
            ):
                persist_cuda_peaks(run_dir, True)

            self.assertEqual(info_path.read_bytes(), original)
            self.assertEqual(
                [path.name for path in run_dir.iterdir()], ["run_info.json"]
            )


if __name__ == "__main__":
    unittest.main()
