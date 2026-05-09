# SDZC gate-variance probe — Qwen/Qwen3-1.7B-Base

Date: 2026-05-08T20:38:43.033087+00:00
Calibration: 566 tokens across 10 passages
n_layers=28, d=2048, intermediate=6144, total neurons=172032

## Verdict: **NO-GO (<5% neurons foldable)**

## Global statistics (all neurons across all layers)

- **Foldable** (std < 0.05): **1.5%**
  - dead (also |μ|<0.05): 1.3%
  - const (|μ|≥0.05): 0.3%
- Loose (0.05 ≤ std < 0.1): 13.3%
- Live (std ≥ 0.1): 85.1%
- median std: 0.3156
- p10 std: 0.0821
- p90 std: 1.0336

## Per-layer breakdown

| L | foldable | dead | const | live | median std |
|---|---|---|---|---|---|
| 0 | 0.4% | 0.0% | 0.4% | 96.6% | 0.1558 |
| 1 | 36.1% | 30.8% | 5.3% | 27.9% | 0.0611 |
| 2 | 2.0% | 0.8% | 1.2% | 44.4% | 0.0907 |
| 3 | 0.8% | 0.5% | 0.3% | 44.0% | 0.0929 |
| 4 | 0.2% | 0.1% | 0.1% | 50.4% | 0.1006 |
| 5 | 0.0% | 0.0% | 0.0% | 88.2% | 0.1875 |
| 6 | 0.0% | 0.0% | 0.0% | 75.9% | 0.1763 |
| 7 | 0.4% | 0.0% | 0.4% | 65.9% | 0.1490 |
| 8 | 0.0% | 0.0% | 0.0% | 81.4% | 0.2029 |
| 9 | 0.0% | 0.0% | 0.0% | 83.1% | 0.2290 |
| 10 | 0.0% | 0.0% | 0.0% | 88.8% | 0.2846 |
| 11 | 0.0% | 0.0% | 0.0% | 92.0% | 0.3150 |
| 12 | 0.0% | 0.0% | 0.0% | 94.3% | 0.3428 |
| 13 | 0.0% | 0.0% | 0.0% | 96.1% | 0.3666 |
| 14 | 0.0% | 0.0% | 0.0% | 97.6% | 0.4104 |
| 15 | 0.0% | 0.0% | 0.0% | 97.1% | 0.4398 |
| 16 | 0.0% | 0.0% | 0.0% | 98.6% | 0.5482 |
| 17 | 0.0% | 0.0% | 0.0% | 97.2% | 0.5381 |
| 18 | 0.1% | 0.1% | 0.0% | 95.4% | 0.5642 |
| 19 | 0.3% | 0.2% | 0.0% | 95.2% | 0.6173 |
| 20 | 0.5% | 0.5% | 0.0% | 93.5% | 0.5513 |
| 21 | 0.3% | 0.3% | 0.0% | 93.3% | 0.5173 |
| 22 | 0.3% | 0.3% | 0.0% | 96.0% | 0.5367 |
| 23 | 0.2% | 0.2% | 0.0% | 97.3% | 0.5557 |
| 24 | 0.1% | 0.1% | 0.0% | 98.9% | 0.6820 |
| 25 | 0.1% | 0.1% | 0.0% | 98.9% | 0.8208 |
| 26 | 0.1% | 0.1% | 0.0% | 99.3% | 0.9057 |
| 27 | 1.1% | 1.1% | 0.0% | 96.7% | 0.6264 |

## Class definitions

- **dead**: std < 0.05 AND |mean| < 0.05 — gate output is near-zero on all tokens. The neuron contributes near-nothing; can be ELIMINATED entirely (drop both W_gate row and W_up row).
- **const**: std < 0.05 AND |mean| ≥ 0.05 — gate output is near-constant nonzero c. FOLDABLE: rescale W_up[i,:] ← c · W_up[i,:] and drop W_gate row.
- **loose**: 0.05 ≤ std < 0.1 — gate has small but non-trivial variation. Borderline; might admit polynomial fit per neuron or stay as is.
- **live**: std ≥ 0.1 — gate genuinely depends on input. Keep full SwiGLU path.

## References

- SCAP (arXiv:2412.07174): GLU activations are 'centered'.
- Round 1 finding: W_up dominates SwiGLU firing-rank ranking.
- Round 3 finding: W_gate is full-rank in linear-algebra sense; SDZC tests whether the *function output* is near-constant on the actual data manifold (compatible with full rank).