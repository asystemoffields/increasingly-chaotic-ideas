# LHQD First-Pass — lm_head SVD spectrum — Qwen/Qwen3-1.7B-Base

Date: 2026-05-08T22:38:04.315766+00:00
Eval: 399 primary tokens; vocab=151936, d=2048, n_layers=28
Baseline: NLL/tok = 2.0590, PPL = 7.838

## Verdict: **EMBED-TIED — measured embed_tokens spectrum, not lm_head; reframe**

- tie_word_embeddings (config): True
- storage tied (data_ptr): True
- sanity gate r=2048 ΔNLL ≈ 0: PASS
- dead-row confound (>20% near-zero): False

## Spectrum

- rank for 99% variance: **1992** (of 2048)
- rank for 95% variance: 1840
- rank for 90% variance: 1675
- variance at r=128: 0.1912
- variance at r=256: 0.2817
- variance at r=512: 0.4268

## Per-row norm distribution

- median row norm: 1.6223
- near-zero count (<0.01×median): 0 (0.00% of 151936)
- bottom-decile norm: 1.3555

## Sweep results

| r | ΔNLL (nats) | PPL | top1 ret | top5 ret | KL/tok | comp (full→mixed) |
|---|---|---|---|---|---|---|
| 2048 | -0.0024 | 7.819 | 0.967 | 0.981 | 0.0016 | 1.95x |
| 1024 | +19.9568 | 3641751222.592 | 0.005 | 0.008 | 20.2165 | 3.89x |
| 512 | +21.3357 | 14459885265.498 | 0.000 | 0.000 | 21.5996 | 7.79x |
| 256 | +22.4159 | 42590683208.647 | 0.000 | 0.001 | 22.8524 | 15.58x |
| 128 | +25.6241 | 1053363796785.239 | 0.000 | 0.000 | 26.1521 | 31.16x |
| 64 | +26.0294 | 1579897930489.409 | 0.000 | 0.000 | 26.5858 | 62.32x |

### Multi-passage robustness (ΔNLL nats per passage)

| r | nav | bio | tech |
|---|---|---|---|
| 2048 | +0.0020 | -0.0056 | -0.0121 |
| 1024 | +21.1245 | +17.0009 | +17.6379 |
| 512 | +22.9715 | +18.4684 | +20.7587 |
| 256 | +24.4482 | +22.2984 | +23.4267 |
| 128 | +26.9593 | +25.3523 | +26.6257 |
| 64 | +28.6481 | +27.4274 | +28.8551 |

## Notes

- LHQD-mixed bytes assume INT8 U + BF16 V^T (the LHQD deployment format).
- arXiv:2603.10145 (Mar 2026) predicts lm_head's training dynamics suppress gradient rank; a concentrated spectrum here is consistent with that paper's mechanistic implication on the trained weight.
- arXiv:2510.24966 is about logit-matrix rank, NOT weight rank — the precondition is that we measure W's spectrum directly (CF9).

## References

- arXiv:2603.10145 (Godey & Artzi, Mar 2026) — Lost in Backpropagation
- arXiv:2510.24966 — Sequences of Logits / low logit-matrix rank
- SVD-LLM (ICLR 2025), ARA (arXiv:2510.19389) — global SVD baselines