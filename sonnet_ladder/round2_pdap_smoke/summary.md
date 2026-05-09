# PDAP outlier-Jaccard probe — Qwen/Qwen3-1.7B-Base

Date: 2026-05-08T18:58:04.909505+00:00
Prompts: 20, total tokens: 525
Sampled layers (7 of 28): [2, 6, 10, 14, 19, 23, 27]
K fractions tested: ['0.1%', '1.0%', '5.0%', '10.0%']
Massive-activation classifier: max(|h|) > 5.0× batch median

## Verdict: **GO (token-dynamic)**

## Headline numbers (k = 1% of d)

- Mean Jaccard (consecutive normal-token pairs): **0.347**  (n=3373)
- Mean Jaccard (massive-activation pairs): **0.260**  (n=162)

If the *normal* mean is high (≥0.85) and the *massive* mean is low, this is the PrefixQuant pattern — channel-static for the bulk of tokens, with sparse token-wise outliers driving the apparent dynamism.

## Multi-K Jaccard (normal stratum, mean across layers)

| K | Jaccard mean |
|---|---|
| 0.1% | 0.718 |
| 1.0% | 0.347 |
| 5.0% | 0.296 |
| 10.0% | 0.282 |

## Channel concentration (top-K=1%)

| L | frac channels for 90% events | top-1%-channels capture |
|---|---|---|
| 2 | 0.056 | 0.456 |
| 6 | 0.057 | 0.452 |
| 10 | 0.069 | 0.479 |
| 14 | 0.069 | 0.453 |
| 19 | 0.080 | 0.414 |
| 23 | 0.141 | 0.422 |
| 27 | 0.145 | 0.452 |

## Reference

- **PrefixQuant** (arXiv:2410.05265): channel-wise vs token-wise outlier decomposition. We stratify per their pattern.
- LLM.int8() (arXiv:2208.07339): claimed channel-static qualitatively; no formal Jaccard.
- SmoothQuant (arXiv:2211.10438): assumed channel-static.
- Quaff (arXiv:2505.14742): Outlier Spatial Stability Hypothesis (fine-tuning, not inference).