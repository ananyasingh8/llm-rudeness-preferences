#!/usr/bin/env python3
"""Extract emotion vectors from multiple layers of Gemma4 models.

Supports both E4B (bfloat16) and 31B (4-bit quantized).
Extracts from multiple layers, performs centering, denoising, logit lens, and PCA.

Run:
    python -m full_replication.extract_vectors --model e4b
    python -m full_replication.extract_vectors --model 31b
"""

import argparse
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from full_replication.config import (
    DATA_DIR, START_TOKEN, DENOISING_VARIANCE_THRESHOLD,
    MODELS, get_extraction_layers, get_results_dir
)

warnings.filterwarnings("ignore")


def load_stories():
    """Load stories from SQLite DB, fall back to JSONL."""
    import sqlite3
    db_path = os.path.join(DATA_DIR, "stories.db")
    stories = defaultdict(list)

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path, timeout=30)
        rows = conn.execute("SELECT emotion, text FROM stories_clean ORDER BY emotion, story_idx").fetchall()
        conn.close()
        for emotion, text in rows:
            stories[emotion].append(text)
        return stories

    # Fallback to JSONL
    path = os.path.join(DATA_DIR, "emotion_stories.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            stories[d["emotion"]].append(d["text"])
    return stories


def load_neutral_dialogues():
    """Load neutral dialogues from SQLite DB, fall back to JSONL, then built-in."""
    import sqlite3
    db_path = os.path.join(DATA_DIR, "neutral.db")

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path, timeout=30)
        rows = conn.execute("SELECT text FROM dialogues ORDER BY topic_idx, dialogue_idx").fetchall()
        conn.close()
        if rows:
            return [r[0] for r in rows]

    path = os.path.join(DATA_DIR, "neutral_dialogues.jsonl")
    if os.path.exists(path):
        dialogues = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                dialogues.append(d["text"])
        if dialogues:
            return dialogues

    print("  WARNING: no neutral dialogues found, using built-in neutral texts")
    return _fallback_neutral_texts()


def _fallback_neutral_texts():
    """Minimal neutral texts if dialogue file doesn't exist yet."""
    texts = [
        "The weather report indicates rain tomorrow with temperatures around 15 degrees.",
        "The meeting is scheduled for 3 PM in conference room B.",
        "The document contains 45 pages of technical specifications.",
        "The train departs from platform 7 at 10:30 AM.",
        "The library closes at 9 PM on weekdays and 5 PM on weekends.",
        "The recipe calls for 200 grams of flour and two eggs.",
        "The software update includes bug fixes and performance improvements.",
        "The population of the city is approximately 500,000.",
        "The bridge was constructed in 1965 and spans 400 meters.",
        "The report summarizes quarterly financial data from three divisions.",
    ]
    return texts * 20  # 200 samples


def get_residual_stream_hooks(model):
    """Attach hooks to capture residual stream activations at all layers."""
    activations = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            activations[name] = hidden_states.detach().cpu().float()
        return hook_fn

    hooks = []
    if hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers
    elif hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise RuntimeError("Cannot find model layers")

    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(make_hook(f"layer_{i}"))
        hooks.append(h)

    return activations, hooks


def extract_activations(model, tokenizer, text, activations_dict, target_layer):
    """Extract residual stream activation at target layer for a single text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        model(**inputs)

    key = f"layer_{target_layer}"
    if key not in activations_dict:
        return None

    hidden = activations_dict[key]  # (1, seq_len, hidden_dim)
    seq_len = hidden.shape[1]

    if seq_len <= START_TOKEN:
        vec = hidden[0].mean(dim=0).numpy()
    else:
        vec = hidden[0, START_TOKEN:].mean(dim=0).numpy()

    activations_dict.clear()
    return vec


def compute_emotion_vectors(emotion_activations):
    """Compute centered emotion vectors: emotion_mean - global_mean."""
    all_vecs = []
    for vecs in emotion_activations.values():
        all_vecs.extend(vecs)
    global_mean = np.mean(all_vecs, axis=0)

    emotion_vectors = {}
    for emotion, vecs in emotion_activations.items():
        emotion_mean = np.mean(vecs, axis=0)
        emotion_vectors[emotion] = emotion_mean - global_mean

    return emotion_vectors, global_mean


def denoise_vectors(emotion_vectors, neutral_activations, variance_threshold=0.5):
    """Project out top PCs from neutral activations explaining threshold variance."""
    neutral_matrix = np.stack(neutral_activations)
    neutral_centered = neutral_matrix - neutral_matrix.mean(axis=0)

    U, S, Vt = np.linalg.svd(neutral_centered, full_matrices=False)

    total_var = (S ** 2).sum()
    cumvar = np.cumsum(S ** 2) / total_var
    n_components = np.searchsorted(cumvar, variance_threshold) + 1

    V_noise = Vt[:n_components].T  # (hidden_dim, n_components)

    denoised = {}
    for emotion, vec in emotion_vectors.items():
        projection = V_noise @ (V_noise.T @ vec)
        denoised[emotion] = vec - projection

    return denoised, n_components, cumvar[n_components - 1]


def logit_lens(model, tokenizer, emotion_vectors, top_k=10):
    """Project emotion vectors through unembedding matrix."""
    # Get unembedding weights
    if hasattr(model, 'lm_head'):
        W = model.lm_head.weight.detach().cpu().float().numpy()
    elif hasattr(model.model, 'language_model'):
        W = model.model.language_model.embed_tokens.weight.detach().cpu().float().numpy()
    else:
        W = model.model.embed_tokens.weight.detach().cpu().float().numpy()

    results = {}
    for emotion, vec in emotion_vectors.items():
        logits = W @ vec
        top_idx = np.argsort(logits)[-top_k:][::-1]
        bot_idx = np.argsort(logits)[:top_k]

        top_tokens = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top_idx]
        bot_tokens = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in bot_idx]
        results[emotion] = {"top": top_tokens, "bottom": bot_tokens}

    return results


def pca_analysis(emotion_vectors):
    """PCA on emotion vectors, return projections and explained variance."""
    emotions = sorted(emotion_vectors.keys())
    matrix = np.stack([emotion_vectors[e] for e in emotions])
    matrix_centered = matrix - matrix.mean(axis=0)

    U, S, Vt = np.linalg.svd(matrix_centered, full_matrices=False)

    n_pcs = min(5, len(S))
    projections = matrix_centered @ Vt[:n_pcs].T
    explained = (S[:n_pcs] ** 2) / (S ** 2).sum()

    return {
        "emotions": emotions,
        "projections": {f"pc{i+1}": projections[:, i].tolist() for i in range(n_pcs)},
        "explained_variance": {f"pc{i+1}": float(explained[i]) for i in range(n_pcs)},
    }


def process_layer(model, tokenizer, stories, neutral_texts, target_layer,
                  activations_dict, results_dir):
    """Full extraction pipeline for a single layer."""
    print(f"\n--- Layer {target_layer} ---")

    # Check if already done
    vec_file = os.path.join(results_dir, f"emotion_vectors_layer{target_layer}.npz")
    res_file = os.path.join(results_dir, f"experiment_results_layer{target_layer}.json")
    if os.path.exists(vec_file) and os.path.exists(res_file):
        print(f"  Already extracted, skipping.")
        return

    # Raw activations cache — save per-emotion so crashes don't lose work
    raw_cache_dir = os.path.join(results_dir, f"_raw_cache_layer{target_layer}")
    os.makedirs(raw_cache_dir, exist_ok=True)

    # Extract emotion activations (with per-emotion checkpointing)
    print(f"  Extracting emotion activations...")
    emotion_activations = defaultdict(list)
    total = sum(len(v) for v in stories.values())
    done = 0
    for emotion, texts in stories.items():
        cache_file = os.path.join(raw_cache_dir, f"{emotion}.npy")
        if os.path.exists(cache_file):
            emotion_activations[emotion] = list(np.load(cache_file))
            done += len(texts)
            if done % 5000 == 0:
                print(f"    [{done}/{total}] (cached)")
            continue

        vecs = []
        for text in texts:
            vec = extract_activations(model, tokenizer, text, activations_dict, target_layer)
            if vec is not None:
                vecs.append(vec)
            done += 1
            if done % 500 == 0:
                print(f"    [{done}/{total}]")

        if vecs:
            np.save(cache_file, np.stack(vecs))
            emotion_activations[emotion] = vecs
    print(f"  {len(emotion_activations)} emotions extracted")

    # Extract neutral activations (with checkpointing)
    neutral_cache = os.path.join(raw_cache_dir, "_neutral.npy")
    if os.path.exists(neutral_cache):
        neutral_activations = list(np.load(neutral_cache))
        print(f"  {len(neutral_activations)} neutral activations (cached)")
    else:
        print(f"  Extracting neutral activations...")
        neutral_activations = []
        for text in neutral_texts:
            vec = extract_activations(model, tokenizer, text, activations_dict, target_layer)
            if vec is not None:
                neutral_activations.append(vec)
        if neutral_activations:
            np.save(neutral_cache, np.stack(neutral_activations))
        print(f"  {len(neutral_activations)} neutral activations")

    # Compute and denoise
    print(f"  Computing emotion vectors...")
    raw_vectors, global_mean = compute_emotion_vectors(emotion_activations)
    print(f"  {len(raw_vectors)} raw vectors computed")

    print(f"  Denoising...")
    vectors, n_comp, var_explained = denoise_vectors(
        raw_vectors, neutral_activations, DENOISING_VARIANCE_THRESHOLD
    )
    print(f"  Projected out {n_comp} components ({var_explained*100:.1f}% variance)")

    # Logit lens
    print(f"  Running logit lens...")
    ll_results = logit_lens(model, tokenizer, vectors, top_k=5)

    # PCA
    print(f"  Running PCA...")
    pca = pca_analysis(vectors)
    for pc, var in pca["explained_variance"].items():
        print(f"    {pc.upper()} explains {var*100:.1f}%")

    # Save vectors
    np.savez(vec_file, **vectors)
    print(f"  Vectors saved: {vec_file}")

    # Save results
    results = {
        "target_layer": target_layer,
        "num_emotions": len(vectors),
        "stories_per_emotion": {e: int(len(v)) for e, v in stories.items()},
        "denoising_components": int(n_comp),
        "denoising_variance": float(var_explained),
        "logit_lens": ll_results,
        "pca": pca,
    }
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved: {res_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["e4b", "31b"],
                        help="Model to extract from")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer numbers (default: auto)")
    args = parser.parse_args()

    model_cfg = MODELS[args.model]
    results_dir = get_results_dir(args.model)
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print(f"=== Emotion Vector Extraction: {model_cfg['model_id']} ===\n")
    stories = load_stories()
    total_stories = sum(len(v) for v in stories.values())
    print(f"Loaded {total_stories} stories across {len(stories)} emotions")

    neutral_texts = load_neutral_dialogues()
    print(f"Loaded {len(neutral_texts)} neutral texts")

    # Determine layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = get_extraction_layers(args.model)
    print(f"Target layers: {layers}")

    # Load model
    print(f"\nLoading model {model_cfg['model_id']}...")
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

    # Detect layers
    if hasattr(model.model, 'language_model'):
        num_layers = len(model.model.language_model.layers)
    elif hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    else:
        raise RuntimeError("Cannot detect model layers")
    print(f"Model loaded. {num_layers} layers.\n")

    # Attach hooks
    activations_dict, hooks = get_residual_stream_hooks(model)

    # Process each layer
    for layer in layers:
        if layer >= num_layers:
            print(f"Skipping layer {layer} (model has {num_layers} layers)")
            continue
        process_layer(model, tokenizer, stories, neutral_texts, layer,
                      activations_dict, results_dir)

    # Cleanup
    for h in hooks:
        h.remove()

    print(f"\n=== EXTRACTION COMPLETE ===")
    print(f"Results in: {results_dir}")


if __name__ == "__main__":
    main()
