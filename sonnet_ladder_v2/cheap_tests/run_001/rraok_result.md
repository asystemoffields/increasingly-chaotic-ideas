# R-RAOK-70B Rung 1 Result

Date: 2026-05-09
Model: Qwen/Qwen3-1.7B-Base (bf16, CPU)
Script: `scripts/rraok_rung1.py`
Output JSON: `experiments/stage0/ladder_v2/round1_raok70b/tier0_stability_per_layer.json`
Wall-clock elapsed: 1643.9 s (~27.4 min; delayed ~17 min waiting for RAM from competing m_strat job; compute-only time ~18 min)

---

## B4 Unit Test: PASS (CRITICAL)

INT4 scale factor unit test (`scripts/rraok_int4_unit_test.py`) verified before running:
- `scale = max_abs(x) / 7` produces zero reconstruction error on integer inputs
- `scale = max_abs(x) / 8` (wrong) produces 0.875 reconstruction error at x=-7 (clipping artifact confirmed)
- Random-tensor max error: 0.5358 <= 0.5 * scale (0.5378) — within expected quantization bound

Not blocking. Proceeding to Rung 1 is valid.

---

## Rung 1 Verdict: GO

**Jaccard@K=1% < 0.50 in 27/28 layers (96.4%, threshold 80%). CF3 K-dependent dynamicity generalizes to all 28 layers.**

GO threshold: Jaccard@K=1% < 0.50 in >=80% of 28 layers — PASSED (96.4%)
NO-GO threshold: Jaccard@K=1% > 0.65 in >=30% of 28 layers — NOT MET (0/28 layers above 0.65)

---

## Per-Layer Jaccard at K=1%

| Layer | Jaccard@K=1% | Status |
|---|---|---|
| L0  | 0.2211 | < 0.50 |
| L1  | 0.4108 | < 0.50 |
| L2  | 0.3255 | < 0.50 |
| L3  | 0.3500 | < 0.50 |
| L4  | 0.3808 | < 0.50 |
| L5  | 0.3697 | < 0.50 |
| L6  | 0.3970 | < 0.50 |
| L7  | 0.4377 | < 0.50 |
| L8  | 0.4145 | < 0.50 |
| L9  | 0.4356 | < 0.50 |
| L10 | 0.4304 | < 0.50 |
| L11 | 0.3913 | < 0.50 |
| L12 | 0.3931 | < 0.50 |
| L13 | 0.4100 | < 0.50 |
| L14 | 0.4209 | < 0.50 |
| L15 | 0.4397 | < 0.50 |
| L16 | 0.4207 | < 0.50 |
| L17 | 0.4081 | < 0.50 |
| L18 | 0.3994 | < 0.50 |
| L19 | 0.3741 | < 0.50 |
| L20 | 0.3423 | < 0.50 |
| L21 | 0.3492 | < 0.50 |
| L22 | 0.3477 | < 0.50 |
| L23 | 0.3404 | < 0.50 |
| L24 | 0.3442 | < 0.50 |
| L25 | 0.3653 | < 0.50 |
| L26 | 0.4162 | < 0.50 |
| L27 | 0.5311 | > 0.50 (lone outlier) |

Mean (all 28 layers): **0.3881**
Only L27 exceeds 0.50; none exceed 0.65.

---

## Multi-K Jaccard Summary (mean across all 28 layers, normal-token stratum)

| K | Mean Jaccard |
|---|---|
| 0.1% (K=2) | 0.4919 |
| 1.0% (K=21) | 0.3881 |
| 5.0% (K=102) | 0.2236 |

K-dependent monotonic decrease confirmed across all 28 layers (consistent with CF3 phase transition finding). Larger K yields lower Jaccard, consistent with the Tier-0/Tier-1 boundary design.

---

## Controls

| Control | Result | Pass |
|---|---|---|
| Permutation control (layer 14, K=1%) | Jaccard = 0.0053 (<0.10 threshold) | PASS |
| Random-init mean (all 28 layers, K=1%) | 0.2095 | noted |
| Gap trained vs permuted | 0.3828 (>=0.30 required) | PASS |
| Gap trained vs random-init | 0.1786 (gap > 0.05 required) | PASS |

Note on random-init control: B1 amendment applied (torch.nn.init.normal_(std=0.02)). The random-init Jaccard at 0.21 is notably non-zero — this reflects the fact that normal_(std=0.02) initialization still produces some channel magnitude structure (not fully uniform). The trained model mean (0.39) exceeds random-init (0.21) by 0.18. The gap satisfies the >0.05 threshold but is modest, suggesting some fraction of the observed trained-model Jaccard reflects architectural activation patterns rather than purely learned weight structure. This does not invalidate the GO verdict (the trained channels are more stable than random-init, and the permutation control confirms the measurement is real), but should be noted for Rung 2 interpretation.

---

## Structural Finding: CF3 Generalizes to 28 Layers

CF3 found mean Jaccard = 0.31 at K=1% across 7 sampled layers. This Rung 1 measurement (amendment B2: first full 28-layer measurement) finds mean Jaccard = 0.388 across all 28 layers.

The 28-layer mean is 0.078 higher than the CF3 7-layer mean. This reflects the inclusion of mid-depth layers (L7-L16) which have slightly higher Jaccard (0.41-0.44) than the deeper layers measured in CF3. The shape is consistent: outlier sets are moderately dynamic (not static) throughout the network, with early layers (L0: 0.22) and deep layers (L20-L24: ~0.34) showing lower Jaccard (more dynamic) than mid-depth layers.

**Key structural note (B3):** The hook captures the MLP block input before the W_gate/W_up split. In deployment, the same quantized activation feeds both W_gate and W_up. The K-dependent dynamicity measured here applies to both projections simultaneously.

**L27 anomaly (0.531):** Layer 27 (the final layer) exceeds 0.50 but is still well below the 0.65 NO-GO threshold. The final layer's output feeds the LM head rather than another MLP block, potentially explaining its slightly more static outlier pattern. This layer warrants a Tier-0/Tier-1 split review in Rung 2 design.

---

## Recommendation: Proceed to Rung 2

Rung 1 GO. Proceed to Rung 2 (three-tier ΔNLL evaluation, ~30-60 min).

**Rung 2 design notes from this measurement:**
1. Tier-0 (FP16 bypass) channel indices: identify per-layer from the mean-magnitude rank on the 200-prompt corpus (already collected during this run — extend Rung 1 hook data collection to capture mean magnitudes in a Rung 2 re-run).
2. L27 threshold review: consider widening Tier-1 shell for L27 (Jaccard 0.531 vs 0.34-0.44 elsewhere), or applying INT8 bulk instead of INT4 for L27 only.
3. Rung 2 B3 scope: quantize the shared MLP block input (before split); both W_gate and W_up receive the quantized tensor. This is already the correct deployment interpretation per B3.
4. Rung 2 multi-seed: 3 corpus orderings for Tier-1 channel identification; ΔNLL GO requires mean < 0.30 nats, worst-of-3 < 0.50 nats.

**If Rung 2 GO:** Cluster C2 (R-RAOK-70B, F-OCSSQ, C-RAOK-Grounded) advances. The calibration-free Jaccard-derived tier boundary is structurally validated. Next step: S4 VPTK (AVX2 kernel for Tier-2 INT4 bulk) or C-ABAR extension.

**If Rung 2 NO-GO (ΔNLL > 1.0 nats):** The CF3 Jaccard phase transition thresholds are not valid quantization-tier oracles. Class-level kill: C2 + all RAOK derivatives. Structural finding: "activation quantization sensitivity does not align with per-token outlier-set stability; different metric required."
