# W_gate rank reduction sweep — Qwen/Qwen3-1.7B-Base

Date: 2026-05-08T19:43:57.492277+00:00
Eval: 455 tokens; n_layers=28, d_hidden=2048, intermediate=6144
Baseline: NLL/tok = 2.4668, PPL = 11.784

## Verdict: **MID (layer-sensitivity follow-up needed; per-layer adaptive K)**

## Results

| K | K/d | PPL | ΔPPL | ΔNLL (nats) | bf16 W_gate (MB/layer) | rank-K (MB/layer) | compress |
|---|---|---|---|---|---|---|---|
| 16 | 0.008 | 14789.126 | +14777.342 | +7.1349 | 25.2 | 0.26 | 96.00x |
| 128 | 0.062 | 19249.136 | +19237.352 | +7.3984 | 25.2 | 2.10 | 12.00x |

## Notes

- W_up and W_down kept at bf16 throughout (testing W_gate sensitivity in isolation).
- All MLP layers receive rank-K replacement simultaneously (deployment-relevant case).
- SVD computed in fp32 for stability, cast to bf16 for inference.

## References

- AERO (arXiv:2410.13060): activation removal + FFN folding (different mechanism, same goal of eliminating gate-side weight).
- Round 1 finding: W_up dominates SwiGLU top-K firing → W_gate carries less load-bearing information → should compress more aggressively. This experiment tests that hypothesis directly.