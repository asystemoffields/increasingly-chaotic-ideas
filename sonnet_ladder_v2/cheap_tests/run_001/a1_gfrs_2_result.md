# A1-GFRS-2 — Z_2^d Sign-Entropy Gauge on W_up

Date: 2026-05-09
Model: Qwen3-1.7B-Base bf16, 28 layers x d_ffn=6144

## Verdict: NO_GO

H_per_row(trained)=1.000, H_per_row(random)=1.000. Trained sign distribution is essentially random; canonical-gauge sign-codebook compression infeasible.

## Numbers

| Metric | Trained | Random-init |
|---|---|---|
| H(global p_pos) bits | 1.0000 | 1.0000 |
| H(per-row mean) bits | 0.9996 | 0.9996 |
| p(sign=+) global mean | 0.5002 | 0.4973 |
| p(sign=+) per-row std | 0.0111 | 0.0114 |
| Frac rows H<0.8 | 0.00% | 0.00% |
| Frac rows H<0.5 | 0.00% | 0.00% |
| W_gate/W_up max-abs sign-agreement | 0.5001 | 0.5006 |

## Compression-viability implication

Sign-bit Huffman per row would save ~0.0% of sign-bit storage (vs uniform 1 bit/sign).
Total weight-storage savings: ~0.00% on the model (since each weight has 1 sign bit out of typically 8-16 stored bits at quantized precision).

## Files
- `scripts/a1_gfrs_2.py`
- `experiments/stage0/ladder_v2/round1_gfrs2/a1_gfrs_2_results.json`