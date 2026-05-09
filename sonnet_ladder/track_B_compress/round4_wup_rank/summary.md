# W_up rank reduction sweep — Qwen/Qwen3-1.7B-Base

Date: 2026-05-08T20:35:48.853947+00:00
Eval: 455 tokens; n_layers=28, d_hidden=2048, intermediate=6144
Baseline: NLL/tok = 2.4668, PPL = 11.784

## Verdict: **NO-GO (W_up is genuinely full-rank; rank reduction is not the path)**

## Results

| K | K/d | PPL | ΔPPL | ΔNLL (nats) | bf16 W_up (MB/layer) | rank-K (MB/layer) | compress |
|---|---|---|---|---|---|---|---|
| 4 | 0.002 | 15441193.324 | +15441181.539 | +14.0858 | 25.2 | 0.07 | 384.00x |
| 8 | 0.004 | 28876940.070 | +28876928.286 | +14.7118 | 25.2 | 0.13 | 192.00x |
| 16 | 0.008 | 2124853.314 | +2124841.530 | +12.1024 | 25.2 | 0.26 | 96.00x |
| 32 | 0.016 | 316815.813 | +316804.029 | +10.1993 | 25.2 | 0.52 | 48.00x |
| 64 | 0.031 | 13451.540 | +13439.755 | +7.0401 | 25.2 | 1.05 | 24.00x |
| 128 | 0.062 | 80037.789 | +80026.005 | +8.8235 | 25.2 | 2.10 | 12.00x |
| 256 | 0.125 | 52246.188 | +52234.403 | +8.3969 | 25.2 | 4.19 | 6.00x |
| 512 | 0.250 | 51666.000 | +51654.215 | +8.3858 | 25.2 | 8.39 | 3.00x |
| 1024 | 0.500 | 122.584 | +110.799 | +2.3420 | 25.2 | 16.78 | 1.50x |

## Notes

- W_up and W_down kept at bf16 throughout (testing W_up sensitivity in isolation).
- All MLP layers receive rank-K replacement simultaneously (deployment-relevant case).
- SVD computed in fp32 for stability, cast to bf16 for inference.

## References

- AERO (arXiv:2410.13060): activation removal + FFN folding (different mechanism, same goal of eliminating gate-side weight).
- Round 1 finding: W_up dominates SwiGLU top-K firing → W_up carries less load-bearing information → should compress more aggressively. This experiment tests that hypothesis directly.