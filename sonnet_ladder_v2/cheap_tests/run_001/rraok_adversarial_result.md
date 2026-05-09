# R-RAOK-70B Adversarial Baseline Results

Date: 2026-05-09
Model: Qwen/Qwen3-1.7B-Base (bf16, CPU)
Script: `scripts/rraok_adversarial.py`
RAOK Rung 2 reference mean ΔNLL: **0.0830 nats**

---

## Aggregate Verdict: PARTIAL

Mixed verdict. KILLED: Tier-0 channel selection (random works equally). SURVIVING: tier-structure, Tier-1 dynamicity.

---

## Adversarial 1 — Flat INT4 (no tier structure)

**Verdict: SURVIVES**

Flat INT4 ΔNLL=5.7500 nats, gap=+5.6670 over RAOK (0.0830). Gap >= 0.20 nats. Tier-0/Tier-1 shell is load-bearing. Three-tier design has structural merit.

Mean ΔNLL: **5.7500 ± 0.1279 nats**
Gap vs RAOK: **+5.6670 nats**

| Seed | WikiText-2 offset | Baseline NLL | Quantized NLL | ΔNLL |
|---|---|---|---|---|
| 1 | 0 | 2.7640 | 8.3380 | 5.5739 |
| 2 | 512 | 2.9562 | 8.8302 | 5.8740 |
| 3 | 1024 | 2.8352 | 8.6373 | 5.8020 |


---

## Adversarial 2 — Random Tier-0 channels

**Verdict: KILL**

Random Tier-0 ΔNLL=0.0571 nats, gap=0.0258 nats from RAOK (0.0830). Gap <= 0.05 nats. Any 2 FP16 channels perform equally well. CF3-derived Tier-0 channel identification is NOT load-bearing. The 'calibration-free CF3-derived Tier-0 selection' novelty COLLAPSES.

Mean ΔNLL: **0.0571 ± 0.0169 nats**
Gap vs RAOK: **-0.0258 nats**

Random Tier-0 seed: 999 (numpy default_rng). Channels drawn without replacement, sorted per layer.

| Seed | WikiText-2 offset | Baseline NLL | Quantized NLL | ΔNLL |
|---|---|---|---|---|
| 1 | 0 | 2.7640 | 2.8450 | 0.0810 |
| 2 | 512 | 2.9562 | 3.0014 | 0.0453 |
| 3 | 1024 | 2.8352 | 2.8804 | 0.0452 |


---

## Adversarial 3 — Static Tier-1 (no per-token rotation)

**Verdict: SURVIVES**

Static Tier-1 ΔNLL=0.4733 nats, gap=0.3904 nats from RAOK (0.0830). Gap >= 0.10 nats. Per-token dynamic rotation IS load-bearing. CF3 dynamicity (K=1% Jaccard) is the genuine moat. Dynamic shell claim survives.

Mean ΔNLL: **0.4733 ± 0.0628 nats**
Gap vs RAOK: **+0.3904 nats**

Static Tier-1 = top-18 channels by corpus mean magnitude (same as calibration pass), fixed across all tokens.

| Seed | WikiText-2 offset | Baseline NLL | Quantized NLL | ΔNLL |
|---|---|---|---|---|
| 1 | 0 | 2.7640 | 3.2939 | 0.5299 |
| 2 | 512 | 2.9562 | 3.4605 | 0.5043 |
| 3 | 1024 | 2.8352 | 3.2210 | 0.3858 |


---

## Summary Table

| Baseline | Mean ΔNLL | Std | Gap vs RAOK | Verdict |
|---|---|---|---|---|
| RAOK three-tier (Rung 2) | 0.0830 | — | 0.0000 | reference |
| Flat INT4 (Adv-1) | 5.7500 | 0.1279 | +5.6670 | SURVIVES |
| Random Tier-0 (Adv-2) | 0.0571 | 0.0169 | -0.0258 | KILL |
| Static Tier-1 (Adv-3) | 0.4733 | 0.0628 | +0.3904 | SURVIVES |

---

## Interpretation Guide

- **KILL**: The attacked component is NOT load-bearing. RAOK's claim for that component fails.
- **MARGINAL**: Component provides minimal benefit (gap < threshold). Inconclusive.
- **SURVIVES**: Component IS load-bearing. Gap clearly exceeds threshold.

Kill thresholds:
- Adv-1 (flat INT4): ΔNLL <= 0.10 nats → KILL; gap >= 0.20 nats → SURVIVES
- Adv-2 (random Tier-0): gap <= 0.05 nats → KILL; gap >= 0.10 nats → SURVIVES
- Adv-3 (static Tier-1): gap <= 0.05 nats → KILL; gap >= 0.10 nats → SURVIVES

If Adv-1 KILLS: RAOK reduces to flat INT4 activation quantization (LLM-FP4, SmoothQuant).
If Adv-2 KILLS: CF3-derived calibration-free channel identification is not the moat.
If Adv-3 KILLS: CF3 K=1% dynamicity claim is not exploited by RAOK's Tier-1 design.
