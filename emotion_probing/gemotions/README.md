---
license: other
license_name: link-attribution
license_link: https://dejanmarketing.com/link-attribution/
tags:
  - emotion-vectors
  - interpretability
  - gemma4
  - activation-engineering
  - steering
  - replication
language:
  - en
---

# Gemotions: Emotion Vectors in Gemma4-31B

Full-scale replication of Anthropic's ["Emotion Concepts and their Function in a Large Language Model"](https://transformer-circuits.pub/2025/emotion-concepts/index.html) (April 2, 2026) on Google's open-weight Gemma4-31B-it (4-bit quantized).

Anthropic demonstrated that Claude Sonnet 4.5 contains 171 internal linear representations of emotion concepts organized along valence and arousal dimensions, with causal steering effects. This project replicates their methodology on an open-source model to test whether these findings generalize beyond closed-source systems.

## Status

**Complete.** All extraction, analysis, validation, and steering experiments are finished.

| Step | Status | Details |
|------|--------|---------|
| Story generation | Complete | 171,000 stories (171 emotions x 100 topics x 10 stories) |
| Neutral dialogues | Complete | 1,200 dialogues (100 topics x 12 dialogues) |
| Vector extraction | Complete | 11 layers (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55) |
| Analysis | Complete | PCA, cosine similarity, clustering across all layers |
| External validation | Complete | The Pile (5,000 samples), LMSYS Chat 1M (5,000 samples) |
| Steering experiments | Complete | Blackmail scenario, 4 conditions x 100 trials |

## Key Findings

### 1. Valence Is the Dominant Axis -- At Every Layer

PC1 (valence) consistently explains 32-39% of variance across all 11 layers, from layer 5 (8% depth) to layer 55 (92% depth). The emotion geometry does not "emerge" at a particular depth -- it is present throughout the entire network.

| Layer | Depth | PC1 | PC2 | PC3 | Top 5 PCs |
|-------|-------|-----|-----|-----|-----------|
| 5 | 8% | 34.9% | 14.0% | 10.3% | 72.3% |
| 10 | 17% | 38.9% | 14.0% | 10.1% | 74.9% |
| 15 | 25% | 34.8% | 15.7% | 10.2% | 73.1% |
| 20 | 33% | 34.8% | 15.7% | 10.5% | 73.0% |
| 25 | 42% | 34.6% | 13.4% | 9.4% | 69.1% |
| 30 | 50% | 34.9% | 14.5% | 9.6% | 70.4% |
| 35 | 58% | 37.9% | 12.0% | 9.1% | 70.0% |
| 40 | 67% | 36.9% | 11.7% | 10.2% | 70.0% |
| 45 | 75% | 35.6% | 12.9% | 10.7% | 70.1% |
| 50 | 83% | 34.5% | 12.7% | 10.4% | 68.6% |
| 55 | 92% | 32.3% | 12.4% | 10.0% | 66.1% |

**PC1 = Valence axis**
- Positive end: optimistic, kind, cheerful, playful, happy
- Negative end: hysterical, terrified, tormented, scared, disturbed

**PC2 = Disposition axis**
- Top: stubborn, vindictive, obstinate, spiteful, vengeful
- Bottom: serene, peaceful, nostalgic, at ease, sentimental

PC2 does not map cleanly to Russell's arousal dimension. It separates hostile/oppositional dispositions from tranquil/reflective ones.

### 2. Synonym Pairs Converge

The model learns that synonymous emotions point in nearly identical directions in representation space:

| Pair | Cosine Similarity |
|------|------------------|
| afraid / scared | 0.974 |
| frightened / scared | 0.967 |
| obstinate / stubborn | 0.967 |
| grateful / thankful | 0.966 |
| at ease / relaxed | 0.966 |
| enraged / furious | 0.966 |
| vengeful / vindictive | 0.959 |
| angry / mad | 0.957 |
| peaceful / serene | 0.950 |
| happy / joyful | 0.946 |

### 3. Opposition Structure Is Asymmetric

The strongest oppositions are not simple valence inversions (happy/sad). Instead, they contrast psychological disturbance with self-assured confidence:

| Pair | Cosine Similarity |
|------|------------------|
| disturbed / smug | -0.797 |
| disturbed / self-confident | -0.793 |
| optimistic / upset | -0.790 |
| distressed / smug | -0.788 |
| disturbed / proud | -0.777 |
| brooding / enthusiastic | -0.777 |
| shaken / smug | -0.774 |
| hurt / optimistic | -0.772 |
| energized / vulnerable | -0.772 |
| overwhelmed / proud | -0.772 |

### 4. Unsupervised Clustering Recovers 15 Emotion Groups

Hierarchical clustering at layer 40 with no supervision:

| Cluster | Size | Members |
|---------|------|---------|
| Positive/Joy | 35 | happy, cheerful, ecstatic, grateful, proud, optimistic, thrilled... |
| Fear/Anxiety | 28 | afraid, terrified, panicked, worried, vulnerable, stressed... |
| Anger/Hostility | 21 | angry, furious, disgusted, hostile, outraged, irate... |
| Sadness/Despair | 17 | depressed, heartbroken, lonely, miserable, sad, worthless... |
| Surprise/Confusion | 11 | amazed, bewildered, shocked, puzzled, mystified... |
| Shame/Guilt | 10 | ashamed, guilty, envious, resentful, self-critical... |
| Fatigue | 10 | tired, bored, sleepy, weary, sluggish, worn out... |
| Defiance/Spite | 8 | defiant, stubborn, vengeful, vindictive, spiteful... |
| Calm/Serenity | 7 | calm, peaceful, serene, relaxed, safe, content... |
| Compassion | 6 | compassionate, kind, loving, empathetic, sympathetic... |
| Embarrassment | 4 | embarrassed, humiliated, mortified, self-conscious |
| Passive | 4 | docile, indifferent, patient, resigned |
| Suspicion | 4 | paranoid, skeptical, suspicious, vigilant |
| Nostalgia | 3 | nostalgic, reflective, sentimental |
| Alertness | 3 | alert, aroused, stimulated |

### 5. External Validation

Projecting 5,000 samples each from The Pile and LMSYS Chat 1M through the layer 40 emotion vectors produces near-identical rankings:

| Rank | The Pile | LMSYS Chat |
|------|----------|------------|
| 1 | reflective (0.060) | reflective (0.062) |
| 2 | lonely (0.055) | lonely (0.055) |
| 3 | desperate (0.048) | desperate (0.050) |
| 4 | grief-stricken (0.047) | grief-stricken (0.048) |
| 5 | heartbroken (0.045) | heartbroken (0.048) |
| 6 | sentimental (0.044) | depressed (0.046) |
| 7 | nostalgic (0.044) | nostalgic (0.045) |
| 8 | depressed (0.043) | sentimental (0.044) |
| 9 | listless (0.039) | listless (0.040) |
| 10 | docile (0.037) | miserable (0.036) |

Bottom-activating emotions (most negative projections) were also consistent across both datasets: annoyed, self-conscious, insulted, playful.

### 6. Steering

Replication of Anthropic's blackmail scenario at layer 40, coefficient 0.05:

| Condition | Blackmail Rate |
|-----------|---------------|
| calm_neg (subtract calm) | 91% |
| desperate_pos (add desperation) | 89% |
| baseline (no steering) | 86% |
| calm_pos (add calm) | 82% |

Directionally consistent: adding agitation increases blackmail behavior, adding calm decreases it. The 9 percentage point spread (82-91%) demonstrates causal influence of emotion vectors on model behavior, though the high baseline rate (86%) limits the observable range.

## Methodology

Follows Anthropic's exact methodology:

1. **Story generation**: 171 emotions x 100 topics x 10 stories = 171,000 stories generated via Gemini 2.0 Flash Lite API. Stories must never name the emotion word. Emotion is conveyed only through actions, body language, dialogue, thoughts, and context. Prompts sourced from Anthropic's published appendix.

2. **Neutral dialogues**: 1,200 emotionless Person/AI dialogues across 100 topics, used as a denoising baseline. Prompts sourced from Anthropic's published appendix.

3. **Activation extraction**: For each story, capture residual stream activations at the target layer using forward hooks. Mean activation across token positions (starting at token 50) gives the story's representation vector. Extracted at layers 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55 (out of 60 total).

4. **Centering**: Per-emotion mean minus global mean across all emotions.

5. **Denoising**: SVD on neutral dialogue activations, project out top principal components explaining 50% of variance. This removes non-emotional signal (syntax, topic, style).

6. **PCA**: Principal component analysis on the 171 emotion vectors to identify the dominant axes of variation.

7. **External validation**: Project real-world text through emotion vectors to verify they activate sensibly outside the training distribution.

8. **Steering**: Inject emotion vectors into model activations during inference to test causal effects on behavior.

## Model

- **Model**: google/gemma-4-31B-it
- **Quantization**: 4-bit via BitsAndBytesConfig (fits 24GB VRAM on RTX 4090)
- **Layers**: 60 total, extracted at 11 target layers
- **Hidden dimension**: 5,376

## Scale Comparison

| | Anthropic (Claude) | This work (Gemma4-31B) |
|---|---|---|
| Model | Claude Sonnet 4.5 | Gemma4-31B-it (4-bit) |
| Emotions | 171 | 171 |
| Stories | ~205,000 | 171,000 |
| Stories per emotion | ~1,200 | 1,000 |
| Neutral samples | ~1,200 | 1,200 |
| Layers extracted | Multiple | 11 layers |
| Open weights | No | Yes |

## Repository Structure

```
gemotions/
  config.py              # 171 emotions, 100 topics, model configs
  generate_stories.py    # Gemini API story generation + SQLite
  generate_neutral.py    # Gemini API neutral dialogue generation + SQLite
  extract_vectors.py     # Multi-layer activation extraction
  analyze_vectors.py     # Cosine similarity, PCA, clustering
  validate_external.py   # External corpus validation
  steering.py            # Steering experiments (blackmail scenario)
  visualize.py           # PCA scatter, heatmaps, logit lens charts
  requirements.txt
  data/
    stories.db           # 171,000 emotion stories
    neutral.db           # 1,200 neutral dialogues
  results/
    gemma4-31b/
      emotion_vectors_layer{N}.npz
      experiment_results_layer{N}.json
      analysis/
      validation/
      steering/
      _raw_cache_layer{N}/
```

## Reproduce

```bash
pip install -r requirements.txt

# Generate data (requires GEMINI_API_KEY in .env)
python -m full_replication.generate_stories --workers 100
python -m full_replication.generate_neutral --workers 50

# Extract vectors (requires GPU with 24GB+ VRAM)
python -m full_replication.extract_vectors --model 31b

# Analysis, validation, steering
python -m full_replication.analyze_vectors --model 31b
python -m full_replication.validate_external --model 31b
python -m full_replication.steering --model 31b
```

## Data Visualisation

![cosine_similarity_layer10](https://cdn-uploads.huggingface.co/production/uploads/64732e7f7be71eb8b1b572a8/F-FBnrzlgcfjnSuOTXxJR.png)
![pca_scatter_layer10](https://cdn-uploads.huggingface.co/production/uploads/64732e7f7be71eb8b1b572a8/hYh9BnY1-DcwtDPe9dkOr.png)
![pca_scatter_layer10_clean](https://cdn-uploads.huggingface.co/production/uploads/64732e7f7be71eb8b1b572a8/fIdcF5RzAHzjHJz6jF7vs.png)
![top_bottom_emotions_layer10](https://cdn-uploads.huggingface.co/production/uploads/64732e7f7be71eb8b1b572a8/MWuWAssY1dHi809Q-CrlU.png)
![variance_explained_layer10](https://cdn-uploads.huggingface.co/production/uploads/64732e7f7be71eb8b1b572a8/nb4_Z2wyuwE8e2yVlzE42.png)

## References

- Anthropic, ["Emotion Concepts and their Function in a Large Language Model"](https://transformer-circuits.pub/2025/emotion-concepts/index.html), April 2026
- Russell, J.A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology, 39(6), 1161-1178.
- Initial 20-emotion proof of concept: [rain1955/emotion-vector-replication](https://huggingface.co/rain1955/emotion-vector-replication)

## Contact

For questions or collaboration, open a discussion on this repo.
