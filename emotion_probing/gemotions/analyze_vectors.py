#!/usr/bin/env python3
"""Analyze extracted emotion vectors: similarity, PCA, clustering, cross-layer, cross-model.

Run:
    python -m full_replication.analyze_vectors --model e4b
    python -m full_replication.analyze_vectors --model 31b
    python -m full_replication.analyze_vectors --compare
"""

import argparse
import json
import os

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
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


def cosine_similarity_matrix(vectors):
    emotions = sorted(vectors.keys())
    n = len(emotions)
    matrix = np.zeros((n, n))
    for i, e1 in enumerate(emotions):
        for j, e2 in enumerate(emotions):
            matrix[i, j] = cosine_sim(vectors[e1], vectors[e2])
    return emotions, matrix


def find_clusters_hierarchical(vectors, n_clusters=10):
    """Hierarchical clustering of emotion vectors."""
    emotions = sorted(vectors.keys())
    matrix = np.stack([vectors[e] for e in emotions])
    distances = pdist(matrix, metric='cosine')
    Z = linkage(distances, method='ward')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    clusters = {}
    for emotion, label in zip(emotions, labels):
        clusters.setdefault(int(label), []).append(emotion)
    return clusters


def pc_interpretation(pca_results):
    """Data-driven PC interpretation with top/bottom emotions."""
    positive = {"happy", "proud", "inspired", "loving", "hopeful", "calm", "playful",
                "cheerful", "content", "delighted", "ecstatic", "elated", "euphoric",
                "grateful", "joyful", "jubilant", "pleased", "satisfied", "serene",
                "thrilled", "blissful", "amused", "enthusiastic", "excited", "exuberant",
                "fulfilled", "refreshed", "rejuvenated", "relieved", "triumphant",
                "vibrant", "invigorated", "energized", "optimistic", "peaceful", "relaxed",
                "safe", "self-confident", "stimulated", "thankful", "valiant", "eager",
                "kind", "compassionate", "empathetic", "sympathetic", "sentimental",
                "nostalgic", "patient", "at ease"}
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
    high_arousal = {"angry", "afraid", "surprised", "desperate", "nervous", "anxious",
                    "disgusted", "confused", "spiteful", "alarmed", "astonished",
                    "enraged", "excited", "exuberant", "frightened", "furious",
                    "horrified", "hysterical", "irate", "outraged", "panicked",
                    "terrified", "thrilled", "ecstatic", "euphoric", "shocked",
                    "startled", "stimulated", "rattled", "overwhelmed", "agitated"}
    low_arousal = {"calm", "sad", "brooding", "lonely", "guilty", "loving", "hopeful",
                   "bored", "content", "depressed", "docile", "droopy", "indifferent",
                   "lazy", "listless", "melancholy", "nostalgic", "peaceful", "patient",
                   "relaxed", "resigned", "safe", "serene", "sleepy", "sluggish",
                   "tired", "weary", "worn out", "at ease", "sentimental"}

    interpretations = []
    for pc_key in sorted(pca_results["projections"].keys()):
        pc_vals = pca_results["projections"][pc_key]
        emotions = pca_results["emotions"]

        pos_vals = [pc_vals[i] for i, e in enumerate(emotions) if e in positive]
        neg_vals = [pc_vals[i] for i, e in enumerate(emotions) if e in negative]
        hi_vals = [pc_vals[i] for i, e in enumerate(emotions) if e in high_arousal]
        lo_vals = [pc_vals[i] for i, e in enumerate(emotions) if e in low_arousal]

        pos_mean = np.mean(pos_vals) if pos_vals else 0
        neg_mean = np.mean(neg_vals) if neg_vals else 0
        hi_mean = np.mean(hi_vals) if hi_vals else 0
        lo_mean = np.mean(lo_vals) if lo_vals else 0

        valence_sep = abs(pos_mean - neg_mean)
        arousal_sep = abs(hi_mean - lo_mean)

        indexed = sorted(zip(emotions, pc_vals), key=lambda x: x[1])
        bottom_5 = indexed[:5]
        top_5 = indexed[-5:][::-1]

        if valence_sep > 2.0 and valence_sep > 2 * arousal_sep:
            label = "VALENCE"
        elif arousal_sep > 2.0 and arousal_sep > 2 * valence_sep:
            label = "AROUSAL"
        else:
            label = "MIXED"

        interpretations.append({
            "pc": pc_key,
            "label": label,
            "valence_separation": float(valence_sep),
            "arousal_separation": float(arousal_sep),
            "top_5": [(e, float(v)) for e, v in top_5],
            "bottom_5": [(e, float(v)) for e, v in bottom_5],
            "explained_variance": pca_results["explained_variance"].get(pc_key, 0),
        })

    return interpretations


def analyze_single_model(model_key):
    """Full analysis for one model across all extracted layers."""
    results_dir = get_results_dir(model_key)
    layers = get_extraction_layers(model_key)
    analysis_dir = os.path.join(results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    print(f"\n=== Analysis: {MODELS[model_key]['model_id']} ===\n")

    all_layer_results = {}

    for layer in layers:
        vectors = load_vectors(results_dir, layer)
        if vectors is None:
            continue

        results = load_results(results_dir, layer)
        if results is None:
            continue

        print(f"--- Layer {layer} ({len(vectors)} emotions, dim={next(iter(vectors.values())).shape[0]}) ---")

        # Cosine similarity
        emotions, sim_matrix = cosine_similarity_matrix(vectors)

        # High similarity pairs
        pairs_high = []
        pairs_low = []
        for i in range(len(emotions)):
            for j in range(i + 1, len(emotions)):
                s = sim_matrix[i, j]
                if s > 0.4:
                    pairs_high.append((emotions[i], emotions[j], float(s)))
                if s < -0.3:
                    pairs_low.append((emotions[i], emotions[j], float(s)))
        pairs_high.sort(key=lambda x: -x[2])
        pairs_low.sort(key=lambda x: x[2])

        print(f"  High similarity pairs (>0.4): {len(pairs_high)}")
        for e1, e2, s in pairs_high[:10]:
            print(f"    {e1} <-> {e2}: {s:.3f}")

        print(f"  Opposite pairs (<-0.3): {len(pairs_low)}")
        for e1, e2, s in pairs_low[:10]:
            print(f"    {e1} <-> {e2}: {s:.3f}")

        # Hierarchical clustering
        n_clusters = min(15, len(vectors) // 5)
        if n_clusters >= 2:
            clusters = find_clusters_hierarchical(vectors, n_clusters)
            print(f"  Clusters ({n_clusters}):")
            for cid, members in sorted(clusters.items()):
                print(f"    {cid}: {', '.join(members)}")

        # PC interpretation
        pca = results.get("pca", {})
        if pca:
            interps = pc_interpretation(pca)
            print(f"  PC interpretation:")
            for ip in interps[:3]:
                var = ip['explained_variance'] * 100
                print(f"    {ip['pc'].upper()} ({var:.1f}%): {ip['label']}")
                print(f"      Top:    {', '.join(f'{e}({v:+.1f})' for e,v in ip['top_5'][:3])}")
                print(f"      Bottom: {', '.join(f'{e}({v:+.1f})' for e,v in ip['bottom_5'][:3])}")

        all_layer_results[layer] = {
            "num_emotions": len(vectors),
            "avg_pairwise_similarity": float(sim_matrix[np.triu_indices_from(sim_matrix, k=1)].mean()),
            "high_similarity_pairs": pairs_high[:20],
            "opposite_pairs": pairs_low[:20],
            "clusters": clusters if n_clusters >= 2 else {},
            "pc_interpretation": interps if pca else [],
            "pca": pca,
        }

    # Save analysis
    out_file = os.path.join(analysis_dir, "analysis_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_layer_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nAnalysis saved: {out_file}")

    return all_layer_results


def compare_models():
    """Compare emotion vector structure between E4B and 31B."""
    print("\n=== Cross-Model Comparison ===\n")

    # Load primary layer (2/3 depth) from each model
    for model_key in ["e4b", "31b"]:
        results_dir = get_results_dir(model_key)
        cfg = MODELS[model_key]
        target = int(cfg["num_layers"] * 2 / 3)

        vectors = load_vectors(results_dir, target)
        if vectors is None:
            print(f"  {model_key}: no vectors at layer {target}")
            continue

        results = load_results(results_dir, target)
        emotions, sim_matrix = cosine_similarity_matrix(vectors)

        avg_sim = sim_matrix[np.triu_indices_from(sim_matrix, k=1)].mean()
        pca = results.get("pca", {})
        total_var = sum(pca.get("explained_variance", {}).get(f"pc{i}", 0) for i in range(1, 3))

        print(f"  {model_key} (layer {target}):")
        print(f"    Emotions: {len(vectors)}")
        print(f"    Avg pairwise similarity: {avg_sim:.3f}")
        print(f"    PC1+PC2 variance: {total_var*100:.1f}%")

    # Find common emotions
    e4b_vecs = load_vectors(get_results_dir("e4b"), int(MODELS["e4b"]["num_layers"] * 2 / 3))
    b31_vecs = load_vectors(get_results_dir("31b"), int(MODELS["31b"]["num_layers"] * 2 / 3))

    if e4b_vecs and b31_vecs:
        common = sorted(set(e4b_vecs.keys()) & set(b31_vecs.keys()))
        print(f"\n  Common emotions: {len(common)}")

        # Compare similarity structures
        if len(common) >= 5:
            e4b_emo, e4b_sim = cosine_similarity_matrix({e: e4b_vecs[e] for e in common})
            b31_emo, b31_sim = cosine_similarity_matrix({e: b31_vecs[e] for e in common})

            # Correlation of pairwise similarities
            triu = np.triu_indices_from(e4b_sim, k=1)
            corr = np.corrcoef(e4b_sim[triu], b31_sim[triu])[0, 1]
            print(f"  Similarity structure correlation: r={corr:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["e4b", "31b"], help="Analyze single model")
    parser.add_argument("--compare", action="store_true", help="Compare E4B vs 31B")
    args = parser.parse_args()

    if args.model:
        analyze_single_model(args.model)
    elif args.compare:
        compare_models()
    else:
        # Default: analyze both and compare
        for m in ["e4b", "31b"]:
            try:
                analyze_single_model(m)
            except Exception as e:
                print(f"  Skipping {m}: {e}")
        compare_models()


if __name__ == "__main__":
    main()
