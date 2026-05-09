# F6-WALIGN — Per-Neuron W_gate/W_up Row-Cosine Alignment Test

Date: 2026-05-09
Model: Qwen3-1.7B-Base (n_layers=28, d_ffn=6144, d_model=2048)
Source: ICLR 2025 Yadav noted FP8-training-context alignment; this test measures the same
object as a post-training compression primitive (no published variant found).

## Verdict: NO_GO

frac_99=0.0000%, frac_90=0.0006%, random_baseline_frac_90=0.0000%. Insufficient alignment for exact-fold compression. Per-neuron alignment claim killed.

## Aggregate statistics (all 28 layers, 172032 total neurons)

- median |cos|: **0.0708**
- mean |cos|: 0.0918
- max |cos|: 0.9761

| Threshold | Fraction (model) | Fraction (random baseline d=2048) | Ratio |
|---|---|---|---|
| |cos| >= 0.99 | 0.0000% | 0.000000% | 0.0× |
| |cos| >= 0.95 | 0.0006% | 0.000000% | 5812872.0× |
| |cos| >= 0.90 | 0.0006% | 0.000000% | 5812872.0× |
| |cos| >= 0.80 | 0.0122% | 0.000000% | 122070312.5× |
| |cos| >= 0.50 | 0.4354% | 0.000000% | 4353841145.8× |

## Per-layer fractions (|cos| >= 0.99 / 0.95 / 0.90)

| Layer | >=0.99 | >=0.95 | >=0.90 | median |
|---|---|---|---|---|
| L00 | 0.000% | 0.000% | 0.000% | 0.051 |
| L01 | 0.000% | 0.000% | 0.000% | 0.059 |
| L02 | 0.000% | 0.016% | 0.016% | 0.031 |
| L03 | 0.000% | 0.000% | 0.000% | 0.038 |
| L04 | 0.000% | 0.000% | 0.000% | 0.066 |
| L05 | 0.000% | 0.000% | 0.000% | 0.042 |
| L06 | 0.000% | 0.000% | 0.000% | 0.074 |
| L07 | 0.000% | 0.000% | 0.000% | 0.068 |
| L08 | 0.000% | 0.000% | 0.000% | 0.074 |
| L09 | 0.000% | 0.000% | 0.000% | 0.080 |
| L10 | 0.000% | 0.000% | 0.000% | 0.093 |
| L11 | 0.000% | 0.000% | 0.000% | 0.083 |
| L12 | 0.000% | 0.000% | 0.000% | 0.088 |
| L13 | 0.000% | 0.000% | 0.000% | 0.086 |
| L14 | 0.000% | 0.000% | 0.000% | 0.082 |
| L15 | 0.000% | 0.000% | 0.000% | 0.082 |
| L16 | 0.000% | 0.000% | 0.000% | 0.079 |
| L17 | 0.000% | 0.000% | 0.000% | 0.092 |
| L18 | 0.000% | 0.000% | 0.000% | 0.087 |
| L19 | 0.000% | 0.000% | 0.000% | 0.100 |
| L20 | 0.000% | 0.000% | 0.000% | 0.083 |
| L21 | 0.000% | 0.000% | 0.000% | 0.080 |
| L22 | 0.000% | 0.000% | 0.000% | 0.078 |
| L23 | 0.000% | 0.000% | 0.000% | 0.079 |
| L24 | 0.000% | 0.000% | 0.000% | 0.063 |
| L25 | 0.000% | 0.000% | 0.000% | 0.051 |
| L26 | 0.000% | 0.000% | 0.000% | 0.049 |
| L27 | 0.000% | 0.000% | 0.000% | 0.103 |

## Compression-viability implication

Insufficient |cos| >= 0.99 fraction for exact-fold compression. The mechanism is dead.

## Files

- `scripts/f6_walign.py`
- `experiments/stage0/ladder_v2/round1_walign/f6_walign_results.json`
