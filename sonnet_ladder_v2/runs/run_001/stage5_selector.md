# Stage 5 — Selector — Run 001 (v2 Sonnet Ladder)

Selector: Sonnet 4.6 (single-agent pass, both tracks).
Date: 2026-05-09.
Inputs read: STAGE5_SELECTOR.md, stage2_judge.md, stage3_deep_research.md, stage4_skeptic.md, SUMMARY.md, KILL_LIST.md.

---

## Scoring Summary (all candidates)

### Track A

| ID | S2 Total | S3 Conf | Pre-Mult | Conv × | Elegance × | Frame | NO-GO | Final |
|---|---|---|---|---|---|---|---|---|
| F-CQSGC | 10 | +1 | 11 | 1.5 (C1, 4-orient) | 1.2 (gauge-exploit, constructive) | 0 | +2 (CF-candidate kill of C1) | **21.8** |
| A-GFRS | 10 | +1 | 11 | 1.2 (C3, 2-orient) | 1.2 (gauge-exploit) | 0 | +1 (C3 gate) | 16.84 |
| F-SGFVO | 12 | +1 | 13 | 1.0 | 1.2 (constructive A4=3) | 0 | +1 (W_V/W_O closure) | 16.6 |
| C-JQSOH | 10 | +2 | 12 | 1.0 | 1.2 (subspace-alignment) | 0 | +1 (deployment consequence) | 15.4 |
| R-AERO-PROBE | 10 | +2 | 12 | 1.0 | 1.0 (wildcard, no stack) | 0 | +2 (AERO SwiGLU CF-candidate) | 14.0 |
| C-LGHF | 10 | +2 | 12 | 1.0 | 1.0 | 0 | 0 | 12.0 |
| R-ACWF | 9 | +1 | 10 | 1.0 | 1.0 | 0 | +1 | 11.0 |
| F-RSCLE | 10 | 0 (DOWNGRADE) | — | — | — | — | — | demoted |
| S3 HOIST | — | — | — | — | — | — | — | dependent (ineligible) |

**Track A pick: F-CQSGC**

### Track B

| ID | S2 Total | S3 Conf | Pre-Mult | Conv × | Elegance × | Frame | NO-GO | Final |
|---|---|---|---|---|---|---|---|---|
| R-RAOK-70B | 11 | +2 | 13 | 1.5 (C2, 3-orient) | 1.1 (conserved-quantity) | 0 | +1 | **22.45** |
| F-OCSSQ | 11 | +2 | 13 | 1.5 (C2) | 1.1 | 0 | +1 | 22.45 |
| C-RAOK-Grounded | 11 | +2 | 13 | 1.5 (C2) | 1.1 | 0 | +1 | 22.45 |
| C-CTERA | 12 | +2 | 14 | 1.0 | 1.2 (subspace-align) | +1 (A2=3) | +1 | 19.8 |
| F-AJDGF | 10 | +1 | 11 | 1.2 (C3) | 1.2 (gauge-exploit) | 0 | +1 | 16.84 |
| R-W_down-SPEC | 11 | +2 | 13 | 1.0 | 1.0 | 0 | +2 (CF-candidate) | 15.0 |
| C-ABAR | 11 | +2 | 13 | 1.0 | 1.0 | 0 | +1 | 14.0 |
| F-ASIT | 10 | +2 | 12 | 1.0 | 1.0 (wildcard) | 0 | +2 | 14.0 |
| R-MCQKV | 10 | +2 | 12 | 1.0 | 1.0 | 0 | +1 | 13.0 |

**Three-way C2 tie (R-RAOK-70B / F-OCSSQ / C-RAOK-Grounded all score 22.45).** Tiebreak on lower runtime then cleaner threshold: R-RAOK-70B and C-RAOK-Grounded share the same Rung 1 (10min). R-RAOK-70B is the stated strongest Cluster C2 representative per Stage 2 (best A1 for the 70B cascade argument; F-OCSSQ is the algebraic framing rep; C-RAOK-Grounded has the best composition argument). R-RAOK-70B wins on the cascade completeness and the 70B deployment arithmetic being self-contained in a single document.

**Track B pick: R-RAOK-70B**

---

## Track A Pick

### 1. Title and one-line description

**Round 1 / Track A — F-CQSGC: Cross-Layer W_Q Subspace Gauge Collapse**

Post-training cross-layer W_Q basis sharing on Qwen3-1.7B-Base: does stacking all 28 W_Q matrices reveal a shared dominant right singular subspace that compresses the entire W_Q family from 457 MB to ~15 MB without retraining?

### 2. Class tags

`compression-lr` / `arch-transposition` (shared-basis inference path changes the computation graph).
Elegance class: `gauge-exploitation` (the shared U is the gauge-fixed coordinate for cross-layer residual-stream attention computation).

### 3. Hypothesis

**H1 (load-bearing)**: The 28 W_Q matrices in Qwen3-1.7B-Base collectively share a dominant right singular subspace U ∈ R^{2048 × 128} that captures > 80% of total variance across the stacked matrix (var@128 > 0.80). This shared subspace exists because, across all transformer layers, query attention reads from the same residual-stream directions that carry the stable semantic signal established in the first few layers; the gauge freedom (any invertible Σ leaves scores M = W_Q W_K^T invariant) allows U to be chosen as the shared coordinate.

**H2 (downstream)**: Given H1, each layer's W_Q can be written W_Q^(ℓ) ≈ C_ℓ U^T where C_ℓ ∈ R^{128 × 2048}, giving 15× storage compression (457 MB → ~15 MB for all 28 layers). This prediction connects to CF11 (within-layer head-redundancy collapses 16 heads to 128 dims) and extends it cross-layer: the same 128 directions that suffice per-layer also suffice across layers.

H1 depends on: CF11 (within-layer head-redundancy), no prior CF entry that disconfirms cross-layer coherence. H1 falsifiability: run the 30-min stacked SVD; measure var@128. H2 is downstream of H1 GO; if H1 is NO-GO, H2 is not tested.

This is related to but not pre-empted by Basis Sharing (arXiv:2410.03765, ICLR 2025) or MASA (arXiv:2508.04581, Aug 2025): both require fine-tuning to recover quality; F-CQSGC's claim is that the shared subspace exists in the pre-trained weights WITHOUT retraining. The no-retraining moat is the differentiating structural claim. As of 2026-05-09, no paper demonstrates weight-only (no-retraining, no calibration data) cross-layer W_Q basis sharing on any LLM family with a measured var@128 result.

### 4. Smallest test

**Model**: Qwen3-1.7B-Base, bf16 (the model for which CF11 holds; no precision conversion required).

**Hardware**: Ryzen 5 7530U, 7.28 GiB RAM. The stacked matrix (57344 × 2048, ~110 MB in bf16) fits in RAM.

**No calibration corpus needed** — purely weight-based operation.

**Eval corpus**: WikiText-2, 512 tokens held-out (same corpus used for all v2 ladder evaluations; reproducible as `datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")[:512]`).

**Script**: `scripts/cqsgc_stacked_svd.py` (new script needed).

Description: Load all 28 W_Q weight tensors (each 2048×2048); normalize each by its Frobenius norm (to prevent large-Frobenius layers dominating the stack); row-reshape each to (2048×2048) and stack into a (57344×2048) matrix M_stack. Compute randomized truncated SVD of M_stack keeping D* ∈ {64, 128, 256, 512} singular values (use `torch.svd_lowrank` or `sklearn.utils.extmath.randomized_svd`). Record `var@D*` = (sum of top-D* squared singular values) / (total Frobenius energy). For the D*=128 case, form U ∈ R^{2048×128} (right singular vectors); for each layer ℓ, reconstruct C_ℓ = W_Q^(ℓ) @ U (shape 2048×128) and W_Q^(ℓ)_approx = C_ℓ @ U^T. Replace all 28 W_Q matrices with the reconstructed approximations. Run forward pass on eval corpus; record ΔNLL vs original.

**Layers/matrices touched**: All 28 W_Q matrices (W_K, W_V, W_O untouched).

**Sweep parameters**:
- D* ∈ {64, 128, 256, 512} for the var@D* profile (pure measurement, no model modification).
- D*=128 for the reconstruction + ΔNLL test (one eval pass).
- With/without Frobenius normalization before stacking (report both; primary result is with normalization).

**Output path**: `experiments/stage0/ladder_v2/round1_cqsgc/`
- `stacked_wq_var_profile.json` — var@{64,128,256,512} with and without normalization.
- `reconstruction_nll_d128.json` — ΔNLL at D*=128, per-layer reconstruction error.

**Wall-clock estimate**:
- Load + normalize + stack: ~2 min.
- Randomized SVD at D*=128 of 57344×2048: ~15 min on Ryzen 5 7530U (uses LAPACK partial SVD; the 128 components are cheap via `svd_lowrank`).
- Reconstruction + eval: ~10 min.
- Total: ≤30 min. **Passes ≤8h gate with large margin.**

**Per-head condition check (added per R6-B LHQD lesson)**: After stacking with Frobenius normalization, verify no individual layer's W_Q is a near-zero-norm outlier that could suppress its contribution. Flag any layer with ‖W_Q^(ℓ)‖_F < 0.1 × median_Frobenius; treat those layers separately.

### 5. Go threshold

**Numerical primary GO**: var@128 > 0.80 AND per-layer reconstruction ΔNLL at D*=128 < 1.0 nat across all 28 layers.

**Soft GO**: var@128 ∈ [0.65, 0.80] AND ΔNLL < 1.5 nats — cross-layer basis sharing requires D*=256 instead of 128; still a 4× W_Q compression (down from 15×). Report D*=256 var and ΔNLL as supplemental.

**CF tether**: GO motivates the S3 HOIST execution-schedule claim (shared (x U) precomputed once per token) and directly advances the GGUF repack variant. The HOIST runtime saving (W_Q NVMe reads: 224 MB → 14.5 MB per token) materializes only on GO.

### 6. No-go threshold

**Numerical NO-GO**: var@128 < 0.50 across all normalization variants.

**Class-level kill**: NO-GO terminates the entire Cluster C1 family simultaneously — R-CROSS-Q, C-CQBAL, and the S3 HOIST cascade all die. Per-layer K=128 compression (CF11 confirmed, +0.98 nats) remains valid under NO-GO; this path is not killed. The NO-GO structural finding is: "W_Q subspaces rotate independently across layers; the residual stream's dominant semantic directions are not stable across transformer depth in Qwen3-1.7B." This is a new CF candidate (CF-C1-KILL) that constrains all future cross-layer W_Q sharing proposals in the SwiGLU-GQA transformer family. **This qualifies for the +2 NO-GO-finding bonus as a CF-entry candidate.**

### 7. Ambiguous-zone follow-up

**Gray zone**: var@128 ∈ [0.50, 0.80].

Follow-up (additional ~15 min, can run in same session): measure var@256 and var@512 to find the D* elbow. If var@256 > 0.80, cross-layer sharing is valid at D*=256 — still a 4× W_Q compression and motivates a widened HOIST cascade at D*=256. Report ΔNLL at D*=256.

If the elbow is at D*=512 (var@512 > 0.80 but var@256 < 0.80): cross-layer basis sharing is marginal (only 2× compression vs per-layer K=128 which already achieves 8×). In this case, DOWNGRADE to "cross-layer sharing adds marginal gain over per-layer compression." The structural finding (W_Q subspace dimension across 28 layers ≈ 512) is still reportable as a CF measurement.

Gray zone resolution runtime: ≤15 min additional.

### 8. Kill criteria (Stage 6 amendment slot)

Stage 6 should reject this pick outright if:

1. **Basis Sharing (arXiv:2410.03765) or MASA (arXiv:2508.04581) explicitly demonstrate weight-only (no fine-tuning) cross-layer W_Q basis sharing on any LLaMA/Qwen family model with positive results.** If Stage 6 reads these papers and finds the no-retraining claim is already there, F-CQSGC is PRE-EMPTED on its structural moat. Stage 6 must read Basis Sharing Section 4 (experiments table, specifically W_Q column) and MASA Section 3 (whether they test without fine-tuning). If pre-empted, escalate to runner-up A-GFRS.
2. **Stacked matrix RAM overflow**: if the 57344×2048 bf16 matrix (~224 MB) plus PyTorch overhead exceeds available RAM, abort and use chunked streaming SVD.
3. **Frobenius norms wildly heterogeneous** (layer 1 norm > 10× other layers per CF6 layer-1 anomaly): re-run without layer 1 and report Layer-1-excluded result separately.

### 9. Skeptic-controls declaration

**Claim under test**: "The stacked W_Q matrices share a common dominant right singular subspace (var@128 > 0.80)" — this is a "X is consistent across layers" claim.

**Permutation control**: Run the same stacked SVD with the rows of each W_Q^(ℓ) randomly permuted before stacking (permutation independently per layer). The permuted stack should have near-uniform singular value energy (var@128 ≈ 128/2048 ≈ 0.063). **GO requires: real var@128 > 0.80 AND permuted var@128 < 0.20 AND gap ≥ 0.60.** Add ~2 min to runtime.

**Random-init control**: Run the stacked SVD on a randomly initialized Qwen3-1.7B (no trained weights — use `AutoConfig` + random `nn.Linear` initialization). The var@128 for a random-init model should be low (near-uniform singular spectrum after normalization, var@128 ≈ 0.063–0.10 depending on initialization). **GO requires: trained var@128 substantially exceeds random-init var@128 by > 0.30.** This catches the case where the shared subspace is a structural property of the architecture (e.g., forced by RMSNorm or GQA weight shapes) rather than a property of the learned weights. Add ~5 min to runtime.

**Multi-seed**: The stacked SVD is deterministic (no randomness in the algorithm beyond randomized SVD). However, randomized SVD has a random projection seed. Run with 3 random seeds (seeds 42, 1337, 2024) for the `svd_lowrank` call; report mean ± std of var@128. The std should be < 0.02 for a stable result. **GO requires: mean var@128 > 0.80 AND worst-of-3 > 0.75 (not below the soft-GO lower bound of 0.65).** Add ~6 min total.

**Full controlled GO condition**: real var@128 (mean over 3 seeds) > 0.80 AND ≥ 0.60 above permuted control AND ≥ 0.30 above random-init control AND worst-of-3 seeds > 0.75.

**Why these controls matter**: The CF11 within-layer head-redundancy finding was NOT subjected to these controls (it was a v1 result). If the cross-layer coherence were an architectural artifact (e.g., all 28 layers happen to have similar W_Q initialization due to weight decay or the GQA head-tying), the permuted and random-init controls would expose this. The pipeline's v2 mandate is to catch false-positive structural findings.

### 10. Honest reach assessment

**P(end-to-end success — var@128 > 0.80 AND ΔNLL < 1.0 nat)**: ~0.45.

Reasons for skepticism: MASA (arXiv:2508.04581) shows that cross-layer attention weight sharing of any kind requires fine-tuning to recover quality on existing models; the no-retraining var@128 > 0.80 threshold is aggressive. The within-layer CF11 result (16 heads span 128 dims) does not mechanistically guarantee cross-layer coherence (the per-layer 128-dim subspace could rotate across depth). However, the four-orientation convergence on this measurement is the strongest independent signal in the run, and the CF2 finding (cos ≈ 0.99 between adjacent residual states) is weak but consistent with stable residual-stream semantics.

**Partial success (GO on var@128 but ΔNLL > 1.0 nat)**: P ≈ 0.25. This would indicate the shared subspace exists but reconstructing W_Q from it introduces non-negligible error. In this case, the structural finding (var@128 value) is reportable as a new CF measurement; the compression application requires calibration-assisted correction (analogous to A3's approach) rather than pure weight-only.

**What NO-GO (var@128 < 0.50) leaves us with**: A new CF entry confirming that per-layer K=128 compression (CF11) does NOT extend to a cross-layer basis. This redirects Stage 6 to: (a) per-layer K=128 deployment as the current best attention compression, (b) S6 LCSO (layer-covariance-similarity operator sharing) as an alternative cross-layer structure hypothesis, (c) the cascade focusing on Track B RAOK for compression gains.

**Cascade implications**: GO unlocks S3 HOIST (execution schedule + GGUF repack for 16× W_Q NVMe bandwidth reduction), and in combination with R-RAOK-70B's activation compression, completes the weight-side + activation-side compression picture for a 70B DRAM-resident attention tier.

### 11. Cascade context

F-CQSGC is **rung 1 of a 3-rung cascade**:

- **Rung 0 (already confirmed)**: CF11 — within-layer W_Q K=128 head-redundancy (16:1, ΔNLL = +0.98 nats).
- **Rung 1 (this experiment)**: F-CQSGC — cross-layer shared U, var@128, D*=128 ΔNLL.
- **Rung 2 (if GO)**: S3 HOIST — implement (x U) precomputation, measure W_Q GEMV time reduction + GGUF repack. Runtime ~2h.
- **Rung 3 (if Rung 2 GO)**: 70B deployment arithmetic — project W_Q NVMe savings from 8.6 GB to ~0.5 GB at D*=128; measure cross-scale spectrum extrapolation validity.

Below this cascade: F-CQSGC NO-GO forces reliance on per-layer K=128 (rung 0) only, with attention weight saving of ~329 MB on 1.7B and ~8.1 GB on 70B (W_Q only). Cross-layer sharing's additional ~7 GB advantage at 70B is forfeit.

### 12. Selection algorithm trace

`(Stage 2 total + Stage 3 confidence) × convergence multiplier × elegance multiplier + frame-novelty bonus + NO-GO-finding bonus`

- Stage 2 total: 10
- Stage 3 confidence: +1 (clean smallest test ≤30min, binary settler at var@128 = single float)
- Pre-multiplier score: 11
- Convergence multiplier: **1.5** (Cluster C1, 4 orientations: R-CROSS-Q, C-CQBAL, F-CQSGC, A-GFRS all converge on the same measurement)
- Elegant-equivalence multiplier: **1.2** (tagged `gauge-exploitation`; the shared U is the gauge-fixed coordinate — an exact algebraic identity, not an approximation; constructive, auditable end-to-end; pre-multiplier score 11 ≥ 9)
- Combined: 11 × 1.5 × 1.2 = **19.8**
- Frame-novelty bonus: **0** (A2 = 2; frame is not saturated but not +1 threshold)
- NO-GO-finding bonus: **+2** (NO-GO terminates all four Cluster C1 ideas simultaneously; the structural finding — cross-layer W_Q subspaces rotate independently — qualifies as a CF-entry candidate constraining the entire cross-layer shared-basis family across SwiGLU-GQA models)
- **Final score: 21.8**

Nearest competitor: A-GFRS (16.84) — Cluster C3, 1.2×, gauge-exploit 1.2×, +1 NO-GO.

### 13. Hand-off note for Stage 6

**Math-correctness checks Stage 6 must re-verify**:

1. **Randomized SVD correctness**: `torch.svd_lowrank` returns approximate singular vectors; verify that ‖M_stack − U_D* S_D* V_D*^T‖_F / ‖M_stack‖_F < 0.05 for D*=128. If the approximation error is large, fall back to `scipy.sparse.linalg.svds`.
2. **Reconstruction formula**: W_Q^(ℓ)_approx = C_ℓ U^T where C_ℓ = W_Q^(ℓ) @ U. Verify that the forward pass uses the reconstructed W_Q^(ℓ) (not the stacked U directly) — the reconstruction is per-layer.
3. **Frobenius normalization reversibility**: each W_Q^(ℓ) is normalized by its Frobenius norm before stacking; U is then computed in normalized space; the per-layer C_ℓ = W_Q^(ℓ) @ U must use the ORIGINAL (un-normalized) W_Q^(ℓ) to produce the correct residency result. Verify the normalization is applied only to the stacked matrix for the SVD, not to the weight reconstruction.
4. **Basis Sharing W_Q results**: Stage 6 must read arXiv:2410.03765 Section 4 (experiment tables) and identify whether W_Q was tested with cross-layer basis sharing WITHOUT fine-tuning. If yes, report whether their var@K metric matches F-CQSGC's var@128 threshold. If the paper shows var@128 > 0.80 for LLaMA-family W_Q, the mechanism is confirmed but the Qwen3-specific result remains novel.

**Blind spots**:
- Stage 3 flagged that MASA uses dictionary learning (a softer form of basis sharing) and achieves 66.7% attention parameter reduction WITH fine-tuning. Stage 6 should note whether the F-CQSGC var@128 result, if GO, corresponds to a similar parameter-reduction ratio as MASA's fine-tuned result. If F-CQSGC weight-only achieves comparable compression to MASA fine-tuned, this is a strong standalone finding; if dramatically worse, it quantifies the "cost of no retraining."
- The stacking treats all 28 layers equally. Stage 6 should check whether early layers (1–4) or late layers (24–28) dominate the variance. If the shared subspace is dominated by early-layer W_Q matrices, the cascade should weight those layers' C_ℓ matrices more carefully.
- **Runner-up for Stage 6 amendment**: If Stage 6 kills F-CQSGC due to Basis Sharing pre-emption, the pre-primed runner-up is **A-GFRS** (score 16.84): Cluster C3 gate, 10-min Frobenius ratio, binary outcome, no pre-emption flags. Script: `scripts/gfrs_frob_ratio.py`.

---

## Track B Pick

### 1. Title and one-line description

**Round 1 / Track B — R-RAOK-70B: R2-Grounded Outlier-Keyed Activation Cascade**

Three-tier CF3-Jaccard-derived calibration-free activation quantization on Qwen3-1.7B-Base: does the K=0.1% static channel stability (Tier 0: 2 channels FP16) plus K=1% dynamic shell (Tier 1: 18 channels INT8) plus bulk INT4 (Tier 2: 2028 channels) achieve ΔNLL < 0.30 nats while delivering 4× KV cache compression?

### 2. Class tags

`compression-quant` / `kv-side`.
Elegance class: `conserved-quantity` (the Tier-0 static channels are a conserved identity: same 2 channels dominant across ≥90% of tokens in ≥20/28 layers — a stable fixed set in the activation distribution).

### 3. Hypothesis

**H1 (upstream gate — Tier-0 stability)**: The top-2 channels by mean magnitude rank in Qwen3-1.7B-Base activations are stable (same channels appear in ≥90% of tokens) in ≥20/28 layers. This follows from CF3 (Jaccard = 0.718 at K=0.1%), which says 71.8% of the top-2-channel set is conserved between consecutive tokens — implying the set is nearly constant. H1 extends this to cross-token stability on diverse prompts.

**H2 (three-tier ΔNLL)**: Given H1, applying the three-tier scheme (2×FP16 / 18×INT8 / 2028×INT4) to all W_up GEMV inputs across all 28 layers achieves ΔNLL < 0.30 nats on WikiText-2 held-out. This depends on: (a) Tier-0 FP16 bypass being lossless for the static channels, (b) Tier-1 INT8 being sufficient for the 18-channel dynamic shell, (c) Tier-2 INT4 bulk introducing bounded error on the non-outlier majority.

H2 depends on: CF3 (K-dependent Jaccard, confirmed), no CF entry that disconfirms INT4 tolerance for the non-outlier bulk. The closest disconfirmation risk is CF5 (W_up is more rank-sensitive than W_gate) — but rank-sensitivity of the WEIGHT and quantization-sensitivity of the ACTIVATION are different objects.

Structural novelty: the tier boundaries (0.1%/1%) are derived from the CF3 Jaccard phase transition, not from calibration sensitivity or Hessian-diagonal estimates (as in LLM.int8(), SmoothQuant, LAMPQ). This calibration-free derivation is the load-bearing structural claim as of 2026-05-09.

### 4. Smallest test

**Two-rung pipeline** (Rung 1 gates Rung 2):

**Rung 1 — Tier-0 stability (upstream gate)**:
- Script: `scripts/raok_tier0_stability.py` (new script).
- Data: R2 PDAP dataset (200 diverse prompts on Qwen3-1.7B-Base; already collected).
- Operation: For each of 200 prompts × 28 layers, extract the top-2 channels by abs magnitude rank from the W_up GEMV input activations. Compute, per layer, the fraction of prompt-token pairs where the same 2 channels appear. Report mean stability per layer and fraction of layers above 90%.
- Runtime: ~10 min (forward pass data already in PDAP; re-extract top-2 per layer).
- Output: `experiments/stage0/ladder_v2/round1_raok70b/tier0_stability_per_layer.json`
- Rung 1 GO: ≥90% same-channel stability in ≥20/28 layers.

**Rung 2 — Three-tier ΔNLL (if Rung 1 GO)**:
- Script: `scripts/raok_three_tier_nll.py` (new script).
- Model: Qwen3-1.7B-Base, bf16.
- Cal corpus: R2 PDAP dataset (200 prompts; identify Tier-0 channels per layer from Rung 1 results; identify Tier-1 channels as next-18 by mean magnitude rank).
- Eval corpus: WikiText-2, 512 tokens held-out (same as all v2 ladder evaluations).
- Layers: all 28.
- Matrices: W_up GEMV inputs quantized at every layer.
- Operation: (1) From Rung 1 output, extract the stable top-2 channel indices per layer (Tier-0). (2) From the 200-prompt PDAP data, identify the next-18 channels by mean magnitude rank per layer (Tier-1). (3) Implement activation quantization: apply FP16 passthrough to Tier-0 channels, INT8 quantization to Tier-1 channels, INT4 quantization to remaining 2028 channels. (4) Run forward pass on eval corpus with quantized activations; measure ΔNLL vs baseline.
- Runtime: Cal setup 10 min + eval 20 min = ~30 min.
- Output: `experiments/stage0/ladder_v2/round1_raok70b/three_tier_nll.json`
- Total pipeline runtime: ~40 min. **Passes ≤8h gate.**

**Per-layer ΔNLL breakdown** (added here vs Stage 3, per R6-B lesson): record per-layer NLL contribution to identify whether deep layers (23-27, wider outlier spread per CF3) are the quality bottleneck. If deep-layer ΔNLL > 3× average, flag for targeted Tier-1 shell widening in those layers.

**Quantization implementation note**: INT4 quantization of activations uses symmetric per-tensor INT4 (not per-row, to keep the overhead negligible). Tier-1 INT8 uses per-tensor symmetric. FP16 passthrough is exact. The quantization step mimics the inference-path quantization that would occur in a production deployment.

### 5. Go threshold

**Numerical primary GO**: Rung 1 stability ≥90% in ≥20/28 layers AND Rung 2 ΔNLL < 0.30 nats.

**Soft GO**: Rung 1 stability ≥70% in ≥20/28 layers AND ΔNLL < 0.50 nats — Tier-0 bypass is valid for 70% of tokens; the other 30% use a per-token fallback (detect and upgrade the channels that shifted). This is a deployment variant with runtime overhead but still a structural finding.

**CF tether**: ΔNLL < 0.30 nats at 4× activation compression (1036 bytes vs 4096 bytes per token per layer) advances the KV cache savings argument: at 4K context, KV reduces from 471 MB to 118 MB on Qwen3-1.7B, and from ~4 GB to ~1 GB on 70B.

### 6. No-go threshold

**Numerical primary NO-GO on Rung 1**: < 70% Tier-0 stability in more than 8/28 layers.

**Class-level kill from Rung 1 NO-GO**: the calibration-free FP16 bypass claim dies. Tier-0 must be determined per-token (per-token detection, not static assignment), which changes the entire tier architecture from calibration-free to per-token-dynamic. This demotes the scheme from "static three-tier" to "per-token detection + INT8 outlier bypass" — a different idea (closer to SmoothQuant's approach) and not novel.

**Numerical primary NO-GO on Rung 2**: ΔNLL > 1.0 nats.

**Class-level kill from Rung 2 NO-GO**: The CF3 Jaccard phase transition thresholds (K=0.1% and K=1%) are NOT the correct tier boundaries for quantization sensitivity. The Jaccard-to-tier derivation is invalid as stated. This constrains all future proposals that use CF3 Jaccard thresholds as quantization-tier oracles. The structural finding is: "activation quantization sensitivity does not align with per-token outlier-set stability; different metric required." This constrains a class of future ideas (+1 bonus).

### 7. Ambiguous-zone follow-up

**Gray zone on Rung 1**: 70–90% Tier-0 stability.
Follow-up: implement adaptive bypass — use static Tier-0 assignment for the 70% stable case, and a 2-cycle per-layer check for the mobile 30%. Measure ΔNLL with and without the adaptive component. Runtime: ~15 min additional. Resolves whether the mobile fraction is worth the per-token detection overhead.

**Gray zone on Rung 2**: ΔNLL 0.30–1.0 nats.
Follow-up: measure per-layer ΔNLL profile (from the Rung 2 output already captured). Identify the worst 5 layers. Widen Tier-1 to INT8 for those 5 layers only (layers 23-27 are the predicted candidates per CF3's wider deep-layer outlier spread). Re-run eval with targeted widening. Runtime: ~10 min. This disambiguates whether the gray-zone ΔNLL is concentrated in a few deep layers (fixable) or distributed uniformly (deeper problem).

### 8. Kill criteria (Stage 6 amendment slot)

Stage 6 should reject this pick outright if:

1. **LLM.int8() or SmoothQuant already implement a three-tier (FP16 / INT8 / INT4) static/dynamic/bulk scheme with the same 0.1%/1% channel boundaries, derived from any method.** Stage 6 must verify: do any published activation quantization papers use three tiers with the specific boundary at K=0.1% (static) and K=1% (dynamic shell)? If so, the scheme is prior art even if the Jaccard motivation is ours.
2. **CF3's Jaccard measurement was run on only 7 of 28 layers** (per KILL_LIST Round 2 finding: "32425 normal-token consecutive-pairs at K=1%, across 7 sampled layers"). If Stage 6 determines that the 7 sampled layers are not representative of the remaining 21 layers (e.g., sampled only from the middle), the Tier-0 stability claim for "≥20/28 layers" requires extending the measurement to all 28 layers. This should be done as part of Rung 1 in the smallest test above (the script already specifies all 28 layers).
3. **Skeptic-controls failure** (see Section 9).

### 9. Skeptic-controls declaration

**Claim under test (H1)**: "Top-2 channels are stable across diverse prompts in ≥20/28 layers (≥90% same-channel stability)" — this is a "X is consistent across prompts / layers" claim.

**Permutation control**: Take the PDAP activation data for one layer (e.g., layer 14); randomly permute the channel indices for each token's activation vector before computing the top-2 channels. The permuted stability should be ~1/2048² × 2! ≈ near zero (random chance of the same permuted channels appearing in top-2 is negligible). **GO requires: real stability ≥90% AND permuted stability < 10% AND gap ≥ 80 percentage points.** Runtime: ~1 min. This verifies the stability is not a measurement artifact.

**Random-init control**: Run the Tier-0 stability measurement on a randomly initialized Qwen3-1.7B (random weights, same architecture). A random-init model will produce random activation distributions with no systematic channel dominance; expected stability for random-init ≈ 2/2048 ≈ 0.1%. **GO requires: trained model stability ≥90% substantially exceeds random-init stability by > 70 percentage points.** This checks that the channel stability is a property of the learned weight structure (CF3 outlier-forming channels), not an architectural artifact. Runtime: ~5 min.

**Multi-seed for Rung 2 (ΔNLL)**: Run Rung 2 three times with different random subsets of the calibration corpus (seeds: first 200 PDAP prompts, random 200 PDAP prompts, last 200 PDAP prompts — if PDAP has >200 prompts; otherwise use 3 random orderings of the 200 prompts for Tier-1 channel identification). Report mean ± std ΔNLL. **GO requires: mean ΔNLL < 0.30 nats AND worst-of-3 ΔNLL < 0.50 nats.** This checks that the tier channel assignment is not overfitted to a specific subset of the calibration corpus. Runtime: ~10 min additional (Rung 2 is cheap; re-running 3× adds 30 min but is within budget).

**Why these controls matter**: CF3 Jaccard was measured on a specific PDAP dataset (200 diverse prompts). If the Tier-0 stable channels are overfitted to that dataset distribution (e.g., they are stable for the specific token types in PDAP but not for general generation), the random-seed multi-run will expose this. The permutation control catches any measurement artifact in the top-2 channel identification algorithm itself.

### 10. Honest reach assessment

**P(end-to-end success — Rung 1 GO AND ΔNLL < 0.30 nats)**: ~0.55.

This is modestly above 50% because: (a) CF3 Jaccard = 0.718 at K=0.1% provides strong prior evidence for Tier-0 stability; (b) LLM.int8() demonstrates binary static/dynamic split works in practice; (c) INT4 on the bulk 99% of channels is the weakest link, but modern quantization literature suggests per-tensor INT4 on non-outlier activations is generally tolerable at < 0.5 nats. The main uncertainty is whether the CF3 7-layer Jaccard measurement generalizes to all 28 layers.

**Partial success path (Rung 1 GO but ΔNLL 0.30–1.0 nats)**: P ≈ 0.25. The structural finding — calibration-free tier boundaries from Jaccard transitions PARTIALLY work — is reportable and refines the tier design. The follow-up (widen Tier-1 for deep layers) is a clear next step.

**What NO-GO (Rung 1 < 70% stability) leaves us with**: A new structural finding that the CF3 K=0.1% Jaccard stability does not hold uniformly across all 28 layers. This reshapes the activation quantization design space: the two static channels are not reliably static in all layers, and the scheme must use per-token outlier detection throughout. Per-token detection is more expensive but is the fallback used by existing methods. The finding constrains Cluster C2 ideas (F-OCSSQ and C-RAOK-Grounded both depend on Tier-0 stability being layer-universal). This NO-GO constrains a class of future calibration-free tier-oracle proposals, earning the +1 NO-GO bonus.

**Cascade implications**: GO at both rungs advances the 70B arithmetic: combined with F-CQSGC GO (Track A), the cascade achieves compressed attention (W_Q + shared basis) + compressed activations (4× KV), projecting to 70B with DRAM-resident attention tier (~3.4 GB) and INT4-activation GEMV throughout. ΔNLL budget: ~1.28 nats from W_Q K=128 (CF11) + 0.30 nats from RAOK ≈ 1.58 nats total — above the 1.0-nat practical threshold; Stage 6 should note this and plan for a fine-tuning recovery step if both compress simultaneously.

### 11. Cascade context

R-RAOK-70B is **rung 1 of a 2-rung Track B cascade**:

- **Rung 0 (already confirmed)**: CF3 — K-dependent Jaccard phase transition at K=0.1% (0.718) and K=1% (0.308). This is the empirical motivation.
- **Rung 1 (this experiment)**: R-RAOK-70B — Tier-0 stability verification + three-tier ΔNLL on Qwen3-1.7B.
- **Rung 2 (if GO)**: S4 VPTK — VPSHUFB integer-only kernel for Tier-2 bulk (AVX2, ~3h; compute complement to the scheme); OR C-ABAR extension (asymmetric bpw allocation oracle using W_Q + W_K spectral hierarchy as the weight-side complement).

Above this cascade: RAOK is a KV-cache and activation-quantization rung. The weight-side compression (W_Q from Track A, W_K from C-ABAR) is orthogonal and stacks additively assuming independence of weight and activation errors.

### 12. Selection algorithm trace

`(Stage 2 total + Stage 3 confidence) × convergence multiplier × elegance multiplier + frame-novelty bonus + NO-GO-finding bonus`

- Stage 2 total: 11
- Stage 3 confidence: +2 (two-rung pipeline ≤40 min total; Rung 1 is a clean binary settler in ≤10 min from existing data; binary class-level kill on NO-GO)
- Pre-multiplier score: 13
- Convergence multiplier: **1.5** (Cluster C2, 3 orientations: R-RAOK-70B, F-OCSSQ, C-RAOK-Grounded all converge on the same Tier-0 stability measurement from different derivation angles)
- Elegant-equivalence multiplier: **1.1** (tagged `conserved-quantity`; Tier-0 channel set is conserved across tokens — an empirically observed fixed point, auditable, but the "conservation" is statistical rather than algebraically exact; 1.1 rather than 1.2 per the calibration-fit quality rule; pre-multiplier score 13 ≥ 9)
- Combined: 13 × 1.5 × 1.1 = **21.45**
- Frame-novelty bonus: **0** (A2=2; Jaccard-to-tier framing is fresh but A2=2 not A2=3)
- NO-GO-finding bonus: **+1** (NO-GO on Rung 1 constrains the calibration-free tier-oracle class; not a CF-candidate in the same sense as F-CQSGC — it is mechanism-class kill, not CF-entry-level kill; +1 rather than +2)
- **Final score: 22.45**

Three-way tie with F-OCSSQ and C-RAOK-Grounded (both score identically). Tiebreak: R-RAOK-70B is the strongest Cluster C2 representative per Stage 2 (A1 = highest for 70B cascade argument) and has the most complete 70B deployment arithmetic (CF13-CF15 independent). F-OCSSQ is the algebraic derivation rep (valuable if the composition angle is needed later); C-RAOK-Grounded is the composition angle rep. All three advance to Stage 3; only one runs first.

Nearest non-C2-cluster competitor: C-CTERA (19.8) — frame-novelty +1, 30-second runtime, but lower A1 (quality-diagnostic, not direct compression).

### 13. Hand-off note for Stage 6

**Math-correctness checks Stage 6 must re-verify**:

1. **CF3 layer coverage**: confirm that the R2 PDAP Jaccard measurement covered all 28 layers, not just 7. The KILL_LIST entry says "7 sampled layers." If only 7 were measured, Rung 1 of this experiment provides the first full-28-layer Tier-0 stability measurement — this is a feature, not a bug, but Stage 6 should not assume the 7-layer result generalizes without running Rung 1.
2. **Tier-1 set identification**: the 18-channel Tier-1 set is defined as "next-18 by mean magnitude rank." Stage 6 should verify that: (a) mean magnitude rank is computed across all 200 PDAP prompts jointly (not per-prompt), and (b) the top-2 (Tier-0) channels are excluded from the Tier-1 set before ranking the remaining 2046.
3. **INT4 quantization symmetry**: ensure the INT4 quantization of Tier-2 activations uses the correct scale factor (max abs value / 7 for symmetric INT4, not max abs / 8 which would create clipping at negative saturation). A sign error here could inflate ΔNLL artifactually.
4. **LLM.int8() three-tier claim**: Stage 6 must verify whether any LLM.int8() variant, SmoothQuant, PrefixQuant, or QuaFF implementation already uses three tiers with these specific boundaries. The Stage 3 prior-art check found no matching paper; Stage 6 re-verifies with the specific query: "three-tier activation quantization FP16 INT8 INT4 static dynamic bulk 0.1% 1% channel thresholds."

**Blind spots**:
- The activation quantization is applied to W_up GEMV inputs only. Stage 3 specifies "all W_up GEMV inputs across all 28 layers." Stage 6 should verify: do W_gate and W_down GEMV inputs also get quantized, or only W_up? The CF3 Jaccard measurement was conducted on the residual-stream activations entering the MLP block, which feed both W_gate and W_up. If only W_up is quantized, the ΔNLL impact may differ from production (where all three MLP projections would use the same quantized input). Stage 6 should run a quick supplemental: RAOK on W_gate and W_down inputs simultaneously and compare ΔNLL.
- **Runner-up for Stage 6 amendment**: If R-RAOK-70B is killed by Stage 6 (e.g., a three-tier prior-art paper is found), the pre-primed runner-up is **C-CTERA** (score 19.8): 30-second experiment, frame-novelty, clean binary outcome, no prior-art threat found. Script: `scripts/ctera_align_angle.py`.

---

## Appendix: Full Pool Summary

### Track A — Final Scores

| ID | Formula | Final Score | Status |
|---|---|---|---|
| F-CQSGC | (10+1)×1.5×1.2 + 0 + 2 = 19.8 + 2 | **21.8** | **SELECTED** |
| A-GFRS | (10+1)×1.2×1.2 + 0 + 1 = 15.84 + 1 | 16.84 | Runner-up |
| F-SGFVO | (12+1)×1.0×1.2 + 0 + 1 = 15.6 + 1 | 16.6 | 3rd |
| C-JQSOH | (10+2)×1.0×1.2 + 0 + 1 = 14.4 + 1 | 15.4 | 4th |
| R-AERO-PROBE | (10+2)×1.0×1.0 + 0 + 2 = 12 + 2 | 14.0 | 5th |
| C-LGHF | (10+2)×1.0×1.0 + 0 + 0 | 12.0 | 6th |
| R-ACWF | (9+1)×1.0×1.0 + 0 + 1 | 11.0 | 7th |
| F-RSCLE | demoted (DOWNGRADE, arXiv:2605.03109) | — | ineligible |
| S3 HOIST | dependent on C1 GO (not independently testable) | — | ineligible |
| S6 LCSO | Stage 4, estimated ~9; not competitive | ~9 | below floor |
| S7 QKIDX | Stage 4, A1~1 (compute not residency) | ~9 | below floor |
| A-PWGI | OUT-OF-ORIENTATION; total 10 but A1=1 | ~11 | below floor |

### Track B — Final Scores

| ID | Formula | Final Score | Status |
|---|---|---|---|
| R-RAOK-70B | (11+2)×1.5×1.1 + 0 + 1 = 21.45 + 1 | **22.45** | **SELECTED** (tiebreak) |
| F-OCSSQ | (11+2)×1.5×1.1 + 0 + 1 | 22.45 | Tied; algebraic-framing runner-up |
| C-RAOK-Grounded | (11+2)×1.5×1.1 + 0 + 1 | 22.45 | Tied; composition-framing runner-up |
| C-CTERA | (12+2)×1.0×1.2 + 1 + 1 = 16.8 + 2 | 19.8 | 4th; Stage-6 amendment runner-up |
| F-AJDGF | (10+1)×1.2×1.2 + 0 + 1 = 15.84 + 1 | 16.84 | 5th |
| R-W_down-SPEC | (11+2)×1.0×1.0 + 0 + 2 = 13 + 2 | 15.0 | 6th |
| C-ABAR | (11+2)×1.0×1.0 + 0 + 1 = 13 + 1 | 14.0 | 7th |
| F-ASIT | (10+2)×1.0×1.0 + 0 + 2 = 12 + 2 | 14.0 | 7th (tied) |
| R-MCQKV | (10+2)×1.0×1.0 + 0 + 1 = 12 + 1 | 13.0 | 9th |
| S1 WDSG | Stage 4; conditional on F-RSCLE data | ~10 | below floor |
| S2 ACES | Stage 4; compute-side, A1~1 at 70B | ~9 | below floor |
| S4 VPTK | Stage 4; dependent on C2 GO | conditional | ineligible as standalone |
| S5 WLHD | Stage 4; A1~1 at 1.7B scale | ~10 | below floor |
| S8 VOMS | Stage 4; dependent on F-SGFVO GO | conditional | ineligible as standalone |
