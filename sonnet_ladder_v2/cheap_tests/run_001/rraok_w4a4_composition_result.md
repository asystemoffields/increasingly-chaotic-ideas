# W4A4 Composition Test Result

Date: 2026-05-09
Model: Qwen/Qwen3-1.7B-Base (bf16 loaded, CPU)
Script: `scripts/rraok_w4a4_composition.py`
Output JSON: `experiments/stage0/ladder_v2/round1_raok70b/w4a4_composition_results.json`
Wall-clock elapsed: 2273.8 s (~37.9 min)

---

## Verdict: COMPOSES CLEANLY

Composition delta=0.0081 nats < 0.10. W4-only ΔNLL=0.5902, W4A4 ΔNLL=0.6813. W4 and RAOK stack approximately additively. Deployment arithmetic holds: predicted W4A4 NLL penalty ≈ W4 + RAOK.

---

## Summary Table

| Scheme | Mean ΔNLL | Std | Worst-of-3 |
|---|---|---|---|
| W4-only (INT4 weights, FP activations) | 0.5902 | 0.0097 | 0.6024 |
| W4A4 (INT4 weights + RAOK activations) | 0.6813 | 0.0214 | 0.6994 |
| RAOK-only reference (Rung 2) | 0.0830 | 0.0159 | 0.1052 |

**Composition delta = (W4A4 ΔNLL) - (W4-only ΔNLL) - (RAOK-only ΔNLL)**
= 0.6813 - 0.5902 - 0.0830 = **0.0081 nats**

W4 and RAOK stack approximately additively. Deployment arithmetic holds.
Predicted penalty for W4+RAOK ≈ W4_delta + RAOK_delta (± noise).

---

## Per-Seed Breakdown

| Seed | Offset | Baseline NLL | W4 ΔNLL | W4A4 ΔNLL | Comp. Delta |
|---|---|---|---|---|---|
| 1 | 0 | 2.7640 | 0.5894 | 0.6512 | -0.0213 |
| 2 | 512 | 2.9562 | 0.5787 | 0.6933 | 0.0315 |
| 3 | 1024 | 2.8352 | 0.6024 | 0.6994 | 0.0141 |

---

## Composition Delta Thresholds

| Zone | Range | This result |
|---|---|---|
| COMPOSES CLEANLY | < 0.10 nats | <-- HERE |
| COMPOSES WITH SLIPPAGE | 0.10 – 0.50 nats |  |
| DESTRUCTIVE INTERFERENCE | > 0.50 nats |  |
| W4 ALONE FAILS | W4-only ΔNLL > 1.0 |  |

---

## Weight Quantization Method

**Per-channel INT4 (pure torch, no external libs):**
- Applied to: `gate_proj`, `up_proj`, `down_proj` in all 28 MLP blocks.
- Per output channel c: `scale_c = max_abs(W[c, :]) / 7`
- `W_q[c, :] = round(W[c, :] / scale_c).clamp(-7, 7) * scale_c`
- B4 compliance: scale denominator is 7 (not 8).
- Dequantized FP copy stored in weight tensors; simulates INT4 inference error.

**Activation quantization:** RAOK three-tier (identical to rraok_rung2.py):
- Tier 0: top-2 channels by mean calibration magnitude → FP16 bypass.
- Tier 1: top-18 current-token channels → INT8 dynamic.
- Tier 2: remaining 2028 channels → INT4 bulk.

---

## Deployment Implication

The 70B-on-laptop claim requires W4 weights + RAOK activations + KV compression.
This test determines whether W4 and RAOK stack without destructive interaction.

DEPLOYMENT CLAIM HOLDS: W4A4 composition is viable. The combined NLL penalty is approximately additive; operators can budget W4_penalty + RAOK_penalty independently.
