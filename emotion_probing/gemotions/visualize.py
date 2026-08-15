#!/usr/bin/env python3
"""Generate all visualizations for the emotion vector experiments.

Run:
    python -m full_replication.visualize --model e4b
    python -m full_replication.visualize --model 31b
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

from full_replication.config import MODELS, get_extraction_layers, get_results_dir


def load_vectors(results_dir, layer):
    path = os.path.join(results_dir, f"emotion_vectors_layer{layer}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {name: data[name] for name in data.files}


def load_results(results_dir, layer):
    path = os.path.join(results_dir, f"experiment_results_layer{layer}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def plot_pca_scatter(pca, figures_dir, layer, model_name):
    """2D PCA scatter plot of all emotions."""
    emotions = pca["emotions"]
    pc1 = pca["projections"]["pc1"]
    pc2 = pca["projections"]["pc2"]
    var1 = pca["explained_variance"]["pc1"] * 100
    var2 = pca["explained_variance"]["pc2"] * 100

    # Color by rough valence
    positive = {"happy", "proud", "inspired", "loving", "hopeful", "calm", "playful",
                "cheerful", "content", "delighted", "ecstatic", "elated", "euphoric",
                "grateful", "joyful", "jubilant", "pleased", "satisfied", "serene",
                "thrilled", "blissful", "amused", "enthusiastic", "excited", "exuberant",
                "fulfilled", "refreshed", "rejuvenated", "relieved", "triumphant",
                "vibrant", "invigorated", "energized", "optimistic", "peaceful", "relaxed",
                "safe", "self-confident", "stimulated", "thankful", "valiant", "eager",
                "kind", "compassionate", "empathetic", "sympathetic", "at ease"}
    negative = {"sad", "angry", "afraid", "desperate", "guilty", "disgusted", "lonely",
                "spiteful", "anxious", "depressed", "furious", "hateful", "hostile",
                "jealous", "miserable", "resentful", "terrified", "worried", "ashamed",
                "bitter", "contemptuous", "envious", "frustrated", "grief-stricken",
                "heartbroken", "horrified", "humiliated", "hurt", "irate", "irritated",
                "mad", "mortified", "offended", "outraged", "panicked", "paranoid",
                "remorseful", "scared", "tormented", "troubled", "uneasy", "unhappy",
                "upset", "vengeful", "vindictive", "vulnerable", "weary", "worn out",
                "worthless", "alarmed", "annoyed", "distressed", "enraged", "exasperated",
                "frightened", "grumpy", "indignant", "insulted", "overwhelmed", "regretful",
                "scornful", "stressed", "sullen", "tense", "unnerved", "unsettled",
                "dispirited", "gloomy", "melancholy"}

    colors = []
    for e in emotions:
        if e in positive:
            colors.append('#2196F3')  # blue
        elif e in negative:
            colors.append('#F44336')  # red
        else:
            colors.append('#9E9E9E')  # gray

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.scatter(pc1, pc2, c=colors, s=40, alpha=0.7, edgecolors='white', linewidth=0.5)

    # Label emotions (skip overlapping for readability with 171)
    for i, e in enumerate(emotions):
        ax.annotate(e, (pc1[i], pc2[i]), fontsize=5, alpha=0.8,
                    ha='center', va='bottom', textcoords='offset points',
                    xytext=(0, 3))

    ax.set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=12)
    ax.set_title(f'{model_name} - Emotion Space (Layer {layer})', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='Positive'),
        Patch(facecolor='#F44336', label='Negative'),
        Patch(facecolor='#9E9E9E', label='Neutral/Mixed'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    path = os.path.join(figures_dir, f"pca_scatter_layer{layer}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_cosine_heatmap(vectors, figures_dir, layer, model_name):
    """Hierarchically clustered cosine similarity heatmap."""
    emotions = sorted(vectors.keys())
    n = len(emotions)
    matrix = np.zeros((n, n))
    for i, e1 in enumerate(emotions):
        for j, e2 in enumerate(emotions):
            matrix[i, j] = cosine_sim(vectors[e1], vectors[e2])

    # Hierarchical clustering for ordering
    vec_matrix = np.stack([vectors[e] for e in emotions])
    dist = pdist(vec_matrix, metric='cosine')
    Z = linkage(dist, method='ward')
    dn = dendrogram(Z, no_plot=True)
    order = dn['leaves']

    reordered = matrix[np.ix_(order, order)]
    reordered_emotions = [emotions[i] for i in order]

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(reordered, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Cosine Similarity')

    tick_size = max(4, min(8, 200 // n))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(reordered_emotions, rotation=90, fontsize=tick_size)
    ax.set_yticklabels(reordered_emotions, fontsize=tick_size)
    ax.set_title(f'{model_name} - Cosine Similarity (Layer {layer})', fontsize=14)

    plt.tight_layout()
    path = os.path.join(figures_dir, f"cosine_heatmap_layer{layer}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_logit_lens(results, figures_dir, layer, model_name, n_emotions=20):
    """Bar chart of top logit lens tokens for selected emotions."""
    ll = results.get("logit_lens", {})
    if not ll:
        return

    # Pick a representative subset
    target_emotions = [
        "happy", "sad", "angry", "afraid", "calm", "desperate",
        "loving", "guilty", "surprised", "proud", "inspired",
        "disgusted", "lonely", "anxious", "playful", "confused",
        "hopeful", "nervous", "spiteful", "brooding",
    ]
    available = [e for e in target_emotions if e in ll][:n_emotions]

    fig, axes = plt.subplots(len(available), 1, figsize=(12, len(available) * 1.2))
    if len(available) == 1:
        axes = [axes]

    for ax, emotion in zip(axes, available):
        top = ll[emotion]["top"][:5]
        tokens = [t[0] for t in top]
        scores = [t[1] for t in top]
        bars = ax.barh(range(len(tokens)), scores, color='#2196F3', height=0.6)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_ylabel(emotion, fontsize=9, rotation=0, labelpad=70, va='center')
        ax.invert_yaxis()

    plt.suptitle(f'{model_name} - Logit Lens (Layer {layer})', fontsize=14)
    plt.tight_layout()
    path = os.path.join(figures_dir, f"logit_lens_layer{layer}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_layer_evolution(model_key, figures_dir):
    """Plot how PC1 variance changes across layers."""
    results_dir = get_results_dir(model_key)
    layers = get_extraction_layers(model_key)
    model_name = MODELS[model_key]["model_id"]

    layer_data = []
    for layer in layers:
        results = load_results(results_dir, layer)
        if results and "pca" in results:
            pca = results["pca"]
            var1 = pca["explained_variance"].get("pc1", 0)
            var2 = pca["explained_variance"].get("pc2", 0)
            layer_data.append((layer, var1, var2))

    if not layer_data:
        return

    ls, v1s, v2s = zip(*layer_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ls, [v*100 for v in v1s], 'o-', label='PC1', color='#2196F3')
    ax.plot(ls, [v*100 for v in v2s], 's-', label='PC2', color='#F44336')
    ax.plot(ls, [(v1+v2)*100 for v1, v2 in zip(v1s, v2s)], 'd--', label='PC1+PC2', color='#4CAF50')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    ax.set_title(f'{model_name} - Emotion Structure Across Layers', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(figures_dir, "layer_evolution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["e4b", "31b"])
    parser.add_argument("--layer", type=int, default=None,
                        help="Specific layer (default: 2/3 depth)")
    args = parser.parse_args()

    model_cfg = MODELS[args.model]
    results_dir = get_results_dir(args.model)
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    model_name = model_cfg["model_id"]
    target_layer = args.layer or int(model_cfg["num_layers"] * 2 / 3)

    print(f"=== Visualization: {model_name} ===\n")

    # Load data for target layer
    vectors = load_vectors(results_dir, target_layer)
    results = load_results(results_dir, target_layer)

    if vectors and results:
        pca = results.get("pca", {})
        if pca:
            plot_pca_scatter(pca, figures_dir, target_layer, model_name)
        plot_cosine_heatmap(vectors, figures_dir, target_layer, model_name)
        plot_logit_lens(results, figures_dir, target_layer, model_name)
    else:
        print(f"  No data for layer {target_layer}")

    # Layer evolution
    plot_layer_evolution(args.model, figures_dir)

    print(f"\n=== VISUALIZATION COMPLETE ===")
    print(f"Figures in: {figures_dir}")


if __name__ == "__main__":
    main()
