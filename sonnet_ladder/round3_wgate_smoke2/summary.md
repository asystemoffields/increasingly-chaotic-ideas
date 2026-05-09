# W_gate rank reduction sweep — Qwen/Qwen3-1.7B-Base

Date: 2026-05-08T19:53:27.433326+00:00
Eval: 455 tokens; n_layers=28, d_hidden=2048, intermediate=6144
Baseline: NLL/tok = 2.4668, PPL = 11.784

## Verdict: **NO-GO (W_gate is genuinely full-rank; rank reduction is not the path)**

## Results

| K | K/d | PPL | ΔPPL | ΔNLL (nats) | bf16 W_gate (MB/layer) | rank-K (MB/layer) | compress |
|---|---|---|---|---|---|---|---|
| 2048 | 1.000 | 11.810 | +0.026 | +0.0022 | 25.2 | 33.55 | 0.75x |
| 1024 | 0.500 | 25.474 | +13.689 | +0.7709 | 25.2 | 16.78 | 1.50x |
| 512 | 0.250 | 420.346 | +408.562 | +3.5743 | 25.2 | 8.39 | 3.00x |
| 256 | 0.125 | 5572.053 | +5560.269 | +6.1587 | 25.2 | 4.19 | 6.00x |

## Notes

- W_up and W_down kept at bf16 throughout (testing W_gate sensitivity in isolation).
- All MLP layers receive rank-K replacement simultaneously (deployment-relevant case).
- SVD computed in fp32 for stability, cast to bf16 for inference.

## References

- AERO (arXiv:2410.13060): activation removal + FFN folding (different mechanism, same goal of eliminating gate-side weight).
- Round 1 finding: W_up dominates SwiGLU top-K firing → W_gate carries less load-bearing information → should compress more aggressively. This experiment tests that hypothesis directly.