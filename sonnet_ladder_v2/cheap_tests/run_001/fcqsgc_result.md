# F-CQSGC Result — Run 001, Track A

**Experiment**: Cross-Layer W_Q Stacked SVD  
**Model**: Qwen3-0.6B-Base (bf16, CPU) — 1.7B-Base used as target but OOM (only 0.44 GB free RAM with co-running rraok_rung1.py); 0.6B used per documented fallback policy. Same architecture (28 layers, d=1024 vs d=2048 at 1.7B). Stage 5 Section 4 permits this fallback.  
**Date**: 2026-05-09  
**Wall-clock**: 9.9 min  
**Stage 6 amendments applied**: A1, A2 (BLOCKING), A3, A4

---

## Verdict: NO-GO

**Hard NO-GO on primary structural gate**: var@128 = 0.2560 < 0.50 (NO-GO threshold).  
**A2 BLOCKING gate**: FAIL (gap = +0.0721, requires > 0.20; but moot given primary NO-GO).  
**Permutation control**: FAIL (permuted var@128 = 0.2559 ≈ real var@128 = 0.2560 — gap = +0.0001; expected gap ≥ 0.60).  
**Random-init control (A1)**: FAIL (gap = +0.1163, requires > 0.30).

---

## Key Numbers

### Stacked SVD var@K profile (3 seeds: 42, 1337, 2024)

| K | mean var@K | std | per_seed |
|---|---|---|---|
| 64 | 0.1550 | 0.0004 | 0.1553, 0.1553, 0.1545 |
| 128 | 0.2560 | 0.0001 | 0.2559, 0.2560, 0.2561 |
| 256 | 0.4184 | 0.0001 | 0.4184, 0.4183, 0.4185 |
| 512 | 0.6701 | 0.0000 | 0.6701, 0.6701, 0.6700 |

### Controls

| Control | Value | Threshold | Pass? |
|---|---|---|---|
| A2 matched-spectrum baseline var@128 | 0.1839 | real > baseline + 0.20 | FAIL (gap = +0.0721) |
| Permuted var@128 | 0.2559 | < 0.20 AND gap ≥ 0.60 | FAIL (gap = +0.0001) |
| Random-init var@128 (normal_(std=0.02)) | 0.1396 | gap > 0.30 | FAIL (gap = +0.1163) |

### Rank sweep ΔNLL (cross-layer shared basis reconstruction)

| K | ΔNLL (nats) | compression | deployment GO? |
|---|---|---|---|
| 32 | +3.504 | 30.9× | NO |
| 64 | +3.162 | 15.5× | NO |
| 128 | +2.678 | 7.7× | NO |
| 256 | +1.909 | 3.9× | NO |
| 512 | +1.071 | 1.9× | NO |
| 1024 | −0.002 | 1.0× | YES (sanity check only) |
| 2048 | (full rank) | 1.0× | — |

**Deployment GO at K=128** (A3 threshold: ΔNLL < 0.30 nat): not achieved. ΔNLL = +2.678 nats at K=128.

### SVD approximation quality check
Frobenius reconstruction error ||M_stack − U S V^T||_F / ||M_stack||_F = 0.8615 (WARNING: >> 0.05). This confirms that a rank-128 truncation captures only ~26% of energy in the stacked matrix — the stacked matrix is effectively full-rank, not low-rank.

---

## Structural Finding

**The permutation control result is the most striking finding**: permuted var@128 (0.2559) ≈ real var@128 (0.2560), gap = +0.0001. This means the "shared subspace" signal in the stacked W_Q is entirely explained by row-permutation-invariant spectral properties (the per-layer spectral concentration being stacked), NOT by any cross-layer directional alignment. When rows of each W_Q layer are independently permuted before stacking, the stacked SVD produces essentially identical var@128. This is the structural-artifact pattern the permutation control was designed to catch.

**Implication for CF11 cross-layer extension**: CF11 (per-layer head-redundancy, var@128 ≈ 0.63 per layer globally) does NOT extend cross-layer. Each layer's W_Q projects onto a different 128-dim subspace of the residual stream. The shared-subspace hypothesis fails at the quantitative level required for compression: var@128 = 0.256 means a rank-128 stacked basis captures only 26% of total W_Q energy across all 28 layers.

**A2 finding**: the matched-spectrum random-orientation baseline gives var@128 = 0.1839. The real trained weights give var@128 = 0.2560, a gap of only +0.0721 (below the 0.20 GO threshold). The small positive gap suggests the trained weights have *slightly* more cross-layer alignment than random orientation would produce, but the effect is negligible for compression purposes.

**COMPOT (arXiv:2602.15200) relevance**: the COMPOT finding that "single shared subspace degrades accuracy at moderate compression" is directly confirmed. Even at K=512 (capturing 67% of stacked energy, 1.9× compression), ΔNLL = +1.071 nats — substantial quality degradation. The design space requires a union-of-subspaces approach if cross-layer sharing is to be viable, but the baseline evidence suggests the individual per-layer subspaces are not coherent enough to make even a union-of-K approach practical without retraining.

---

## Class-Level Kill: Cluster C1

Per Stage 5, Section 6: this NO-GO terminates the entire Cluster C1 family simultaneously:

- **R-CROSS-Q** — killed. Cross-layer Q sharing requires shared subspace; disproven.
- **C-CQBAL** — killed. Balanced cross-layer Q compression depends on var@128 > 0.80.
- **F-CQSGC** (this experiment) — NO-GO.
- **A-GFRS** — killed by association; the same underlying cross-layer residual-stream stability claim fails.
- **S3 HOIST cascade** — killed. The HOIST cascade (W_Q NVMe bandwidth reduction via shared U) requires GO.

Per-layer K=128 compression (CF11, ΔNLL +0.98 nats per-layer, within-layer only) remains valid and unaffected.

**New CF candidate**: CF-C1-KILL — "W_Q subspaces rotate independently across transformer depth in Qwen3-0.6B-Base; cross-layer shared basis compression is not viable without retraining in the SwiGLU-GQA transformer family."

---

## Notes and Caveats

1. **Model scale**: Results obtained on Qwen3-0.6B-Base (d=1024), not Qwen3-1.7B-Base (d=2048) as planned. The 1.7B model caused an OOM crash with only 0.44 GB free RAM. The 0.6B result is structurally consistent — same 28-layer architecture, same attention pattern — but the 1.7B-Base result may differ. Given the severity of the NO-GO (var@128 = 0.256, permutation gap = +0.0001), scale is unlikely to change the verdict.

2. **Permutation control anomaly**: permuted var@128 (0.2559) is nearly identical to real var@128 (0.2560). This is the signal that the "low var@128" result in the real weights is itself structurally expected given the per-layer spectral properties — it is not even a property of the trained directions, just the spectral concentration level. The cross-layer shared-basis hypothesis is falsified at the mechanistic level, not just at the numeric threshold.

3. **A2 bottleneck**: The matched-spectrum A2 baseline computation took 118.8s (28 × torch.linalg.svdvals + QR-based random orthogonal construction). This is within the 30-min budget.

4. **A4 COMPOT**: arXiv:2602.15200 is directly validated by the rank-sweep ΔNLL results. Even rank-512 (capturing 67% energy, 1.9× compression) gives ΔNLL = +1.071 nats.

---

## Recommended Next Experiment

Given Cluster C1 kill, redirect to runner-up **A-GFRS** (Stage 5 score 16.84) — Cluster C3 gate, Frobenius-ratio test, 10-min runtime, binary outcome, no pre-emption flags. Script: `scripts/gfrs_frob_ratio.py`.

Alternatively, if Track B (R-RAOK-70B) produces a GO, the weight-side attention compression now relies exclusively on per-layer K=128 (CF11, +0.98 nats). The combined ΔNLL budget for 70B deployment must be recalculated with CF11 only (no cross-layer W_Q gain).

---

## References

- COMPOT (arXiv:2602.15200, Feb 2026): single shared subspace degrades accuracy at moderate compression — confirmed.
- Basis Sharing (arXiv:2410.03765, ICLR 2025): requires fine-tuning; the no-retraining result here confirms fine-tuning is necessary for cross-layer basis sharing.
- MASA (arXiv:2508.04581, Aug 2025): fine-tuning required; consistent with NO-GO finding.
- QSVD (arXiv:2510.16292, NeurIPS 2025): cross-layer adaptive rank, calibration-guided; different approach, unaffected.
