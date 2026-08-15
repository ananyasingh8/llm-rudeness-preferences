#!/usr/bin/env python3
"""Validate emotion vectors against external corpora.

Projects activations from external text onto emotion vectors to verify
they activate on emotionally matching content.

Run:
    python -m full_replication.validate_external --model e4b
    python -m full_replication.validate_external --model 31b
"""

import argparse
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from full_replication.config import MODELS, START_TOKEN, get_results_dir

warnings.filterwarnings("ignore")

# Datasets to validate against (HuggingFace dataset IDs)
DATASETS = {
    "pile_subset": {
        "path": "monology/pile-uncopyrighted",
        "split": "train",
        "text_field": "text",
        "n_samples": 5000,
    },
    "lmsys_chat": {
        "path": "lmsys/lmsys-chat-1m",
        "split": "train",
        "text_field": "conversation",
        "n_samples": 5000,
    },
}


def load_emotion_vectors(results_dir, layer):
    path = os.path.join(results_dir, f"emotion_vectors_layer{layer}.npz")
    data = np.load(path)
    return {name: data[name] for name in data.files}


def get_hooks_and_layers(model):
    activations = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            activations[name] = hidden.detach().cpu().float()
        return hook_fn

    if hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers
    elif hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise RuntimeError("Cannot find model layers")

    hooks = []
    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(make_hook(f"layer_{i}"))
        hooks.append(h)

    return activations, hooks


def extract_activation(model, tokenizer, text, activations_dict, target_layer):
    """Extract mean activation at target layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        model(**inputs)

    key = f"layer_{target_layer}"
    if key not in activations_dict:
        return None

    hidden = activations_dict[key]
    seq_len = hidden.shape[1]

    if seq_len <= START_TOKEN:
        vec = hidden[0].mean(dim=0).numpy()
    else:
        vec = hidden[0, START_TOKEN:].mean(dim=0).numpy()

    activations_dict.clear()
    return vec


def project_onto_emotions(activation, emotion_vectors):
    """Project activation onto each emotion vector, return cosine similarities."""
    results = {}
    act_norm = np.linalg.norm(activation) + 1e-8
    for emotion, vec in emotion_vectors.items():
        vec_norm = np.linalg.norm(vec) + 1e-8
        results[emotion] = float(np.dot(activation, vec) / (act_norm * vec_norm))
    return results


def validate_dataset(model, tokenizer, emotion_vectors, target_layer,
                     activations_dict, dataset_cfg, results_dir):
    """Run validation on one dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' library not installed. Run: pip install datasets")
        return None

    dataset_name = dataset_cfg["path"]
    print(f"\n  Loading dataset: {dataset_name}...")

    try:
        ds = load_dataset(
            dataset_cfg["path"],
            split=dataset_cfg["split"],
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        return None

    n_samples = dataset_cfg["n_samples"]
    text_field = dataset_cfg["text_field"]

    # Incremental save file for projections
    incremental_file = os.path.join(results_dir, "validation",
                                     f"_{dataset_name}_layer{target_layer}_progress.jsonl")
    os.makedirs(os.path.dirname(incremental_file), exist_ok=True)

    # Resume from existing progress
    projections = []
    emotion_activation_sums = defaultdict(float)
    emotion_activation_counts = defaultdict(int)
    count = 0

    if os.path.exists(incremental_file):
        with open(incremental_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                projections.append(record)
                for emotion, score in record["top_emotions"]:
                    emotion_activation_sums[emotion] += score
                    emotion_activation_counts[emotion] += 1
                count += 1
        print(f"  Resuming from {count} cached samples...")

    if count >= n_samples:
        print(f"  Already complete ({count} samples).")
    else:
        print(f"  Processing {n_samples - count} remaining samples...")
        skip = count
        with open(incremental_file, "a", encoding="utf-8") as f:
            for item in ds:
                if count >= n_samples:
                    break

                if skip > 0:
                    skip -= 1
                    continue

                # Extract text
                if isinstance(item.get(text_field), list):
                    text = " ".join(str(turn) for turn in item[text_field][:3])
                else:
                    text = str(item.get(text_field, ""))

                if len(text) < 50:
                    continue

                activation = extract_activation(model, tokenizer, text, activations_dict, target_layer)
                if activation is None:
                    continue

                projs = project_onto_emotions(activation, emotion_vectors)

                for emotion, score in projs.items():
                    emotion_activation_sums[emotion] += score
                    emotion_activation_counts[emotion] += 1

                top_5 = sorted(projs.items(), key=lambda x: -x[1])[:5]
                record = {"text_preview": text[:100], "top_emotions": top_5}
                projections.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                count += 1
                if count % 500 == 0:
                    f.flush()
                    print(f"    [{count}/{n_samples}]")

    # Compute statistics
    emotion_stats = {}
    for emotion in emotion_vectors:
        n = emotion_activation_counts.get(emotion, 0)
        if n > 0:
            mean = emotion_activation_sums[emotion] / n
            emotion_stats[emotion] = {"mean_projection": float(mean), "n_samples": n}

    sorted_emotions = sorted(emotion_stats.items(), key=lambda x: -x[1]["mean_projection"])

    print(f"  Top 10 most activated emotions across dataset:")
    for emotion, stats in sorted_emotions[:10]:
        print(f"    {emotion}: mean projection = {stats['mean_projection']:.4f}")

    return {
        "dataset": dataset_name,
        "n_samples": count,
        "emotion_stats": emotion_stats,
        "sample_projections": projections[:100],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["e4b", "31b"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default=None,
                        help="Run on specific dataset (default: all)")
    args = parser.parse_args()

    model_cfg = MODELS[args.model]
    results_dir = get_results_dir(args.model)

    target_layer = args.layer or int(model_cfg["num_layers"] * 2 / 3)

    # Check vectors exist
    vec_path = os.path.join(results_dir, f"emotion_vectors_layer{target_layer}.npz")
    if not os.path.exists(vec_path):
        print(f"ERROR: No vectors at {vec_path}. Run extract_vectors.py first.")
        return

    emotion_vectors = load_emotion_vectors(results_dir, target_layer)
    print(f"Loaded {len(emotion_vectors)} emotion vectors from layer {target_layer}")

    # Load model
    print(f"Loading model {model_cfg['model_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])

    load_kwargs = {"device_map": "auto"}
    if model_cfg["quantization"] == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
        )
    else:
        load_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_cfg["model_id"], **load_kwargs)
    model.eval()

    activations_dict, hooks = get_hooks_and_layers(model)

    # Run validation
    datasets_to_run = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
    validation_dir = os.path.join(results_dir, "validation")
    os.makedirs(validation_dir, exist_ok=True)

    for ds_name, ds_cfg in datasets_to_run.items():
        result = validate_dataset(
            model, tokenizer, emotion_vectors, target_layer,
            activations_dict, ds_cfg, results_dir
        )
        if result:
            out_file = os.path.join(validation_dir, f"{ds_name}_layer{target_layer}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {out_file}")

    for h in hooks:
        h.remove()

    print("\n=== VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()
