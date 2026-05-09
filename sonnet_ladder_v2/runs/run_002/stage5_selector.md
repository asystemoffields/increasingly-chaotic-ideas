# Stage 5 — Selector — Run 002
Run: 002 | Date: 2026-05-09 | Selector: Sonnet claude-sonnet-4-6

---

## Selection Algorithm Trace

### Step 1 — Re-verification of novelty (as of 2026-05-09)

Top-5 Track A candidates by raw Stage 2/3 scoring: R3-A/XLQB, A1/GHPI, F3/RMGF, C5/ROPE, F4/SSIF.
Top-5 Track B candidates: F2/CQBS, F1/JVOC, C2/CQST, R6-B/ATTN-SPECTRUM, C1/OHAA.

Re-checked all ten against the publication landscape as of 2026-05-09:

**R3-A/XLQB / F2/CQBS** (same measurement, two tracks): Stage 3 found MASA (arXiv:2508.04581, Aug 2025) as the closest cousin — training-time dictionary weight sharing. The post-training stacked-SVD measurement asking whether TRAINED Qwen3 W_Q matrices accidentally share a cross-layer basis is NOVEL as of 2026-05-09. No paper in the prior-art search found post-hoc stacked-SVD measurement on a deployed transformer. Status: **NOVEL — confirmed.**

**A1/GHPI**: "Quantifying LLM Attention-Head Stability" (arXiv:2602.16740, Feb 2026) measures head stability for circuit analysis, NOT head-permutation-gauge-fixing for compression. Status: **NOVEL — confirmed.**

**F3/RMGF**: "An Extra RMSNorm is All You Need for Fine Tuning to 1.58 Bits" (arXiv:2505.08823) uses RMSNorm for quantization stability without folding gains into weight spectra. SmoothQuant/QuaRot confirmed as ADJACENT not PRE-EMPTING. Status: **NOVEL for spectrum-concentration question — confirmed.** Note: Stage 4 flagged the gauge-exploitation orientation as "heavily colonized" (SE-4) with 5 ideas. Frame-novelty bonus reduced from +2 to +1.

**F4/SSIF**: arXiv:2605.02907 (April 2026) formalizes the softmax energy field invariant. SSIF's quantization application survives as NOVEL but the frame is now PARTIAL per Stage 3 assessment. Frame-novelty bonus: +1 (quantization application novel; frame occupied).

**F1/JVOC**: The fused operator ∑_h W_O^h W_V^h is NOVEL as post-training measurement. "Predicting LLM Compression Degradation from Spectral Statistics" (arXiv:2604.18085) discusses per-matrix spectra but NOT the fused sum. MLA (training-time design) is ADJACENT. Status: **NOVEL — confirmed.**

**F2/CQBS, C2/CQST, R6-B/ATTN-SPECTRUM, C1/OHAA**: All confirmed NOVEL per Stage 3 prior-art checks; no new papers found in the 24h window since Stage 3 ran (same-day run).

### Step 2 — Convergence multipliers applied

- C1 (cross-layer W_Q subspace, 4-orient: XLQB/R, CQBS/F, CQST/C, GHPI/A): **1.5×**
- C2 (W_V/W_O characterization, 3-orient: ATTN-SPECTRUM/R, JVOC/F, QKJB/R): **1.5×**
- C3 (CF3×CF11 coupling, 2-orient: OHAA/C, LGOP/C): **1.2×**

### Step 3 — Frame-novelty bonuses

Track A:
- R3-A/XLQB: 0 (not A2=3; frame-novelty via convergence weight, not frame score)
- A1/GHPI: +1 (A2=3, head-permutation gauge-fixing for compression is unpublished)
- F3/RMGF: +1 (A2=3 but SE-4 flags gauge orientation as colonized; spectrum-concentration subframe survives)
- F4/SSIF: +1 (A2=3 but frame partially occupied by arXiv:2605.02907; quantization application novel)
- C5/ROPE: 0 (A2=2; frame is fresh but not A2=3)
- A2/ASAKS: +1 (A2=3; append-only framing for KV is genuine transplant)

Track B:
- F2/CQBS: 0 (convergence cluster carries the weight; the measurement is novel but A2=2 in Stage2)
- F1/JVOC: 0 (A2=2 in Stage2; novelty is algebraic-exact construction, not frame-transplant)
- C2/CQST: 0 (A2=2)
- R6-B/ATTN-SPECTRUM: 0 (A2=1)
- C1/OHAA: 0 (A2=2)

### Step 3b — Elegant-equivalence multipliers

Track A:
- F3/RMGF: elegance-class `gauge-exploitation` tag from Stage3. The fold W_Q_gauged = W_Q × diag(g) is algebraically exact (constructive). Raw pre-multiplier score = 12+1=13 ≥ 9 floor → **1.2× applied.**
- F6/RSGO: elegance-class `gauge-exploitation`, constructive but narrower scope than RGPS. Raw=11+1=12 ≥ 9 → **1.1×.**
- All others in Track A: 1.0×.

Track B:
- F2/CQBS: elegance-class `subspace-alignment` from Stage3 M2 tag. The shared-basis construction is algebraically exact (SVD is deterministic). Raw pre-mult=15 ≥ 9 → **1.2×.**
- F1/JVOC: elegance-class `algebraic-identity` from Stage3 M4. The head-summation ∑_h W_O^h W_V^h is an exact algebraic identity for the attention output. Raw pre-mult=14 ≥ 9. Part of a 3-orient convergence cluster (C2). Constructive + convergence: **1.5 × 1.2 = 1.8 effective mult.** (Capped by "cannot single-handedly flip a weak proposal" — but JVOC is not weak; raw is 14. Apply 1.2× elegance after 1.5× convergence.)
- A6/RGPS: elegance-class `gauge-exploitation`, constructive. Raw=10 ≥ 9 → **1.2×.** Not in a convergence cluster → 1.0× conv.

### Step 3c — Wildcard non-penalization

C5/ROPE, F6/RSGO, A4/RSLK: CF-tether suspension confirmed per their FREE SWING tags. No penalty for solo advancer status. Structural floor verified (A4≥1, smallest test ≤ 8h, primitive on actual stack). All three pass.

### Step 4 — NO-GO-finding bonuses

Track A:
- R3-A/XLQB: NO-GO (cumvar < 0.70) ⇒ kills all cross-layer W_Q sharing variants = class-level kill. +1.
- F3/RMGF: NO-GO (r_99 ≥ 0.60 post-gauging) ⇒ closes norm-gain-spectrum concentration route. +1.
- F4/SSIF: NO-GO (mean offset small) ⇒ closes logit-offset-quantization class. +1.
- C5/ROPE: NO-GO (CV > 0.3) ⇒ confirms head-outlier independence = structural finding. +1.
- R5-A/WDLA-RESCUE: NO-GO (R² < 0.5) ⇒ permanently closes inference-time cross-scale affine surgery. CF candidate. +2.
- S5/ZKAM: NO-GO (skip fraction < 30%) ⇒ closes zone-map block-skip for this attention density regime. CF candidate. +2.

Track B:
- F2/CQBS: NO-GO (cumvar < 0.70) ⇒ class-level kill on cross-layer W_Q sharing. +1.
- F1/JVOC: NO-GO (r_99 ≥ 0.90) ⇒ extends CF8 to W_V/W_O class; closes attention-weight compression direction cleanly. CF candidate. **+2.**
- R6-B/ATTN-SPECTRUM: NO-GO ⇒ closes per-matrix V/O compression. +1.
- C1/OHAA: NO-GO ⇒ closes CF3×CF11 coupling hypothesis. +0 (structural finding, but not class-level kill).
- R2-B/RAOK-SCALE: NO-GO ⇒ closes CF3-partition benefit over naive INT4. +1.

### Step 5 — Final scores

**Track A final scores:**

| ID | (S2+S3conf) | ×conv | ×eleg | =sub | +FN | +NO-GO | =TOTAL |
|----|-------------|-------|-------|------|-----|--------|--------|
| R3-A/XLQB | 13+1=14 | 1.5 | 1.0 | 21.0 | 0 | +1 | **22.0** |
| A1/GHPI | 11+1=12 | 1.5 | 1.0 | 18.0 | +1 | 0 | **19.0** |
| F3/RMGF | 11+1=12 | 1.0 | 1.2 | 14.4 | +1 | +1 | **16.4** |
| R1-A/QKJB | 9+1=10 | 1.5 | 1.0 | 15.0 | 0 | 0 | **15.0** |
| C5/ROPE | 11+2=13 | 1.0 | 1.0 | 13.0 | 0 | +1 | **14.0** |
| F4/SSIF | 11+1=12 | 1.0 | 1.0 | 12.0 | +1 | +1 | **14.0** |
| S5/ZKAM | ~9+1=10 | 1.0 | 1.0 | 10.0 | +2 | +2 | **14.0** |
| A2/ASAKS | 11+1=12 | 1.0 | 1.0 | 12.0 | +1 | 0 | **13.0** |
| S6/RCPS | ~9+1=10 | 1.0 | 1.0 | 10.0 | +2 | +1 | **13.0** |
| F6/RSGO | 10+1=11 | 1.0 | 1.1 | 12.1 | +1 | 0 | **13.1** |
| R5-A/WDLA | 9+1=10 | 1.0 | 1.0 | 10.0 | 0 | +2 | **12.0** |
| A5/CAWD | 9+1=10 | 1.0 | 1.0 | 10.0 | 0 | 0 | **10.0** |

**Track A pick: R3-A/XLQB (22.0).** Clear winner. Runner-up: A1/GHPI (19.0).

**Track B final scores:**

| ID | (S2+S3conf) | ×conv | ×eleg | =sub | +FN | +NO-GO | =TOTAL |
|----|-------------|-------|-------|------|-----|--------|--------|
| F2/CQBS | 13+2=15 | 1.5 | 1.2 | 27.0 | 0 | +1 | **28.0** |
| F1/JVOC | 12+2=14 | 1.5 | 1.2 | 25.2 | 0 | +2 | **27.2** |
| C2/CQST | 13+1=14 | 1.5 | 1.0 | 21.0 | 0 | +1 | **22.0** |
| R6-B/ATTN-SPECTRUM | 11+2=13 | 1.5 | 1.0 | 19.5 | 0 | +1 | **20.5** |
| C1/OHAA | 10+2=12 | 1.2 | 1.2 | 17.3 | 0 | 0 | **17.3** |
| R2-B/RAOK-SCALE | 10+1=11 | 1.2 | 1.0 | 13.2 | 0 | +1 | **14.2** |
| A6/RGPS | 9+1=10 | 1.0 | 1.2 | 12.0 | 0 | +1 | **13.0** |
| S2/PASS | ~9+1=10 | 1.0 | 1.0 | 10.0 | +1 | +1 | **12.0** |
| S4/CDCA | ~9+1=10 | 1.0 | 1.0 | 10.0 | 0 | +1 | **11.0** |
| S7/MPRW | ~8+1=9 | 1.0 | 1.0 | 9.0 | 0 | +1 | **10.0** |

**Track B pick: F2/CQBS (28.0).** Clear winner. Runner-up: F1/JVOC (27.2).

**Important structural note on Track A and B shared experiment**: R3-A/XLQB (Track A) and F2/CQBS (Track B) resolve the SAME stacked-SVD measurement. Stage 3 explicitly states: "XLQB's first rung is IDENTICAL to CQBS's first rung. They should be run as a single experiment." The two plans use the same 20-minute script with different threshold interpretations and diverging cascade frameworks. The runner executes ONE combined experiment (scripts/cqbs_xlqb_combined.py) and reports results against both threshold sets. This is intentional and optimal: the convergence cluster is so strong that both tracks are pointing at the same empirical question.

---

---

# TRACK A SELECTION

---

## Round 002 / Track A — R3-A/XLQB: Cross-Layer W_Q Basis Sharing (Reach)

### 1. Title and one-line description

Round 002 / Track A — R3-A/XLQB: Does stacking all 28 W_Q matrices reveal a single shared subspace in R^{2048}, enabling a 70B cascade deployment under sub-8-GB DRAM constraints?

### 2. Class tags

`compression-lr` | `convergence-cluster-C1` | `measurement-structural`

### 3. Hypothesis (1–2 paragraphs)

The 28 W_Q matrices in Qwen3-1.7B-Base (each 2048×2048) were independently trained but may converge on a shared low-dimensional subspace due to the functional pressure of multi-head attention: CF11 shows that within each layer, all 16 query heads collectively span only ~128 dimensions. If this within-layer head-redundancy is stable across layers — driven by CF2's near-parallel residual stream forcing similar inputs to all layers' W_Q matrices — then the 28 per-layer 128-dim subspaces should be mutually aligned. A stacked SVD on the 57344×2048 matrix measures this alignment directly. If cumvar(K=128) ≥ 0.85, a single shared basis U ∈ R^{2048×128} plus 28 thin coefficient matrices C_L ∈ R^{128×2048} reconstructs all W_Q layers, giving a 15.6× W_Q storage reduction and opening the 70B cascade path.

This hypothesis depends on CF11 (W_Q global K=128 GO, r_99/d ≈ 0.63) as the within-layer anchor and CF2 (cos(h_L, h_{L+1}) ≈ 0.99) as the cross-layer forcing argument. Both are confirmed v2 internal findings. The experiment is the direct falsification: either the 28 within-layer subspaces are mutually aligned (shared basis) or they rotate across layers (independent). The result is decisive with no ambiguity about the prior-art: MASA (arXiv:2508.04581) uses training-time weight sharing; this is a post-hoc measurement on an already-trained model.

### 4. Smallest test

**Model**: Qwen3-1.7B-Base bf16 (already on system, no additional download needed)
**Precision**: bf16 throughout (no calibration corpus for this rung)
**Calibration corpus**: Not required — this is a pure weight-space measurement
**Eval corpus**: Not required for first rung (threshold is on the SVD, not NLL)

**Protocol** (combined with Track B CQBS; single script, two threshold interpretations):
1. Load W_Q for all 28 layers: 28 × 2048×2048 bf16 tensors → 28 × 8 MB = 224 MB I/O
2. Stack into a single 57344×2048 matrix (concatenate along axis 0)
3. Run `torch.linalg.svd(W_stack, full_matrices=False)` → singular values S (shape 2048), right vectors V^T (shape 2048×2048). Cast to float32 before SVD for numerical stability.
4. Compute cumvar(K) = sum(S[:K]²) / sum(S²) for K ∈ {32, 64, 96, 128, 192, 256, 384, 512}
5. Secondary metrics (CQST free rider, same pass):
   - Per-layer cosine similarity: for each layer ℓ, extract top-128 right singular vectors V_Q^ℓ (from per-layer SVD, 28 separate SVDs); compute mean cosine sim between V_Q^ℓ and V_Q^{ℓ'} for all pairs.
   - Spearman ρ between per-layer effective rank r̂_Q^ℓ and layer index ℓ (for QCDS).
   - Projection of CF3 static outlier channels onto V_Q^ℓ (for OHAA, free after per-layer SVDs).
6. Report: cumvar table, per-layer-pair cosine similarity heatmap, depth-rank Spearman, outlier-alignment scores.

**Layers/matrices touched**: W_Q only (primary). W_K as secondary in same pass (~5 min additional).
**Output path**: `experiments/stage0/ladder_v2/round2_XLQB_CQBS/`
**Script**: `scripts/cqbs_xlqb_combined.py` — inputs: model path, output dir; loads all W_Q, stacks, runs SVD, computes all secondary metrics; writes CSV tables and PNG cumvar plot.

**Wall-clock estimate**: 
- Model load (W_Q only): ~3 min (224 MB at NVMe 3 GB/s)
- Stacked SVD on 57344×2048 float32: ~12 min on Ryzen 5 7530U (numpy/scipy LAPACK; dgemm-dominated)
- Per-layer SVDs (28 × 2048×2048): ~5 min
- Secondary metrics: ~2 min
- **Total: ~22 minutes**

**8h gate**: PASSES easily (22 min << 8h).

### 5. Go threshold (Track A / XLQB)

**GO**: cumvar(K=128) ≥ 0.50 on the 57344×2048 stacked W_Q matrix.

- At 0.50, a shared basis at K=128 compresses W_Q by ~2× (128-dim shared + 128-dim per-layer residual). Not the 15.6× of the 0.85 case, but the cascade arithmetic for 70B deployment at K=256 is still material: 10.7 GB W_Q → ~2.6 GB.
- XLQB's lower threshold (0.50 vs CQBS's 0.85) reflects the Reach orientation's focus on whether ANY shared structure exists at the 70B scale, with per-layer-pair cosine similarity providing the follow-up.
- Tied to CF11 (K=128 head-redundancy finding): the same 128 dimensions that constitute one head's worth of subspace within a layer should appear in the shared basis.

**SOFT-GO**: cumvar(K=256) ≥ 0.70. A broader shared basis at K=256 gives 4× compression. Motivates deeper cascade investigation at 256-dim shared + 256-dim per-layer = K_eff=512. Still actionable for 70B deployment.

### 6. No-go threshold (Track A / XLQB)

**NO-GO**: cumvar(K=128) < 0.30.

- Class-level kill: cross-layer W_Q basis sharing as a compression primitive is dead. The 28 W_Q matrices span independent subspaces despite within-layer head-redundancy. This closes the C1 convergence cluster's hypothesis (XLQB, CQBS, CQST, GHPI all share this fate). The per-layer CF11 result (K=128 global GO) survives; it is a within-layer finding unaffected by the cross-layer null result.
- Downstream constraint: GHPI's gauge-fixing framing loses its main motivation. CQBS's 15.6× compression claim is dead. CQST's CF2+CF11 composition prediction is falsified. S3/CQBS-IO (sequential IO optimizer) is moot.
- This is a high-value NO-GO: it closes the largest single compression lever in the pool and redirects attention to JVOC/RAOK/WDLA-RESCUE.

### 7. Ambiguous-zone follow-up

**GRAY (0.30 ≤ cumvar(K=128) < 0.50)**:

Run per-layer-pair cosine similarity heatmap (free, already computed in Step 5 of the protocol). Check whether any connected component spanning ≥14 of 28 layers has mean cosine similarity ≥ 0.7 between their top-128 right singular vectors. If such a component exists, partial layer-cluster sharing survives: the basis is shared within the cluster but not globally.

Follow-up decision tree:
- If cluster of ≥14 layers with cosine ≥ 0.7: SOFT-GO for cluster-specific sharing. Follow up with 30-min per-cluster cumvar computation. Advance to CQBS (Track B) for the full cluster analysis.
- If no cluster with ≥14 layers: NO-GO overall.

Runtime: 30 additional minutes (cosine heatmap computation already in the script). Resolves the "partial sharing" hypothesis cleanly.

### 8. Kill criteria (Stage 6 amendment slot)

Stage 6 should reject this pick if:
- **CF9 failure**: the stacked SVD precondition (weight matrices have compatible shapes and the stacking is meaningful) is trivially satisfied by construction. No CF9 risk.
- **CF10 failure**: no calibration fitting in this rung. No CF10 risk.
- **Frame mismatch**: if the "same subspace" claim is reframed as "shared trained initialization" (which would require evidence from weight initialization analysis, not SVD), Stage 6 should note this is a different hypothesis than the post-hoc measurement XLQB tests.
- **Hidden prior art**: if a paper published between Stage 3 (2026-05-09) and Stage 6 review explicitly measures the stacked-SVD cumvar on a transformer's W_Q matrices and reports the result, Stage 6 should demote to REPLICATION. The 24h publication-velocity window has been respected; Stage 6 must re-check.
- **Skeptic-controls deficiency**: if the permutation control or random-init control (Section 9) is not included in the actual run, Stage 6 must reject and require the controls before interpreting the cumvar result.

### 9. Skeptic-controls declaration

The experiment claims: "cross-layer W_Q basis sharing is a trained-model structural property, not an architectural artifact."

This claim triggers all three v2 skeptic controls:

**Permutation control**: After computing cumvar(K=128) on the original stacked matrix, run the same computation on a row-permuted version: shuffle the 57344 rows of the stacked matrix randomly. Expected outcome: cumvar(K=128) of the permuted matrix should be approximately K/d = 128/2048 ≈ 0.063 (uniform distribution of variance). The GO threshold (cumvar ≥ 0.50) must be met by the original AND must exceed the permuted baseline by ≥ 0.40. If the original cumvar is 0.52 and the permuted cumvar is 0.48 (gap of 0.04), this is NOT a GO — the concentration is architectural, not trained.

Rationale for non-trivial permuted baseline: the stacked matrix contains 28 copies of W_Q matrices that all have head-redundancy (CF11). Even after row-permutation across layers, per-layer head-redundancy is preserved in subblocks. Therefore the permuted-matrix cumvar may be higher than true random. The gap criterion (≥ 0.40) is conservative to account for this.

**Random-init control**: Generate a random-initialized Qwen3-1.7B-architecture model with the same weight shapes (using `torch.nn.init.kaiming_normal_` or the default Hugging Face init). Stack its 28 W_Q matrices identically and compute cumvar(K=128). Expected outcome: random-init cumvar ≈ 0.063 (random Gaussian matrix spectra, no structure). The trained-model GO threshold must be met AND must exceed the random-init baseline by ≥ 0.40. This controls for "head-redundancy is architectural (heads are initialized to be redundant) rather than trained."

**Multi-seed**: The experiment has no randomness in the primary measurement (SVD is deterministic). However, the calibration corpus (used in secondary metrics only; not in the primary SVD) is sampled. For the primary measurement (cumvar on weight matrices), multi-seed is not applicable: the weight matrices are deterministic. Document: "Multi-seed control not applicable to the primary SVD measurement because the stacked matrix is a deterministic function of the model weights. For calibration-dependent secondary metrics (OHAA alignment scores), run with 3 random seeds for calibration-token sampling and report mean ± std."

Note: "Not applicable" for multi-seed on the primary measurement is justified because there is no stochastic element in SVD of a fixed weight matrix. Stage 6 should accept this justification.

### 10. Runtime estimate

| Phase | Time |
|---|---|
| Model load (W_Q only, 224 MB bf16) | 3 min |
| Stacked SVD (57344×2048 float32, LAPACK) | 12 min |
| Per-layer SVDs (28 × 2048×2048) | 5 min |
| Secondary metrics (cosine, Spearman, OHAA) | 2 min |
| Permutation control SVD | 10 min |
| Random-init control SVD | 5 min (init is fast; SVD dominates) |
| Post-processing, CSV/PNG output | 1 min |
| **Total** | **~38 min** |

Well within 8h gate.

### 11. Script identification

`scripts/cqbs_xlqb_combined.py` — new script needed. Description:

Inputs: `--model_path` (path to Qwen3-1.7B-Base), `--output_dir`, `--controls` (boolean flag for permutation/random-init controls), `--calibration_path` (WikiText-2 for secondary metrics).

Outputs:
- `cumvar_table.csv`: cumvar(K) for K ∈ {32,64,96,128,192,256,384,512} for (a) original, (b) row-permuted, (c) random-init stacked matrices.
- `perlayer_cosine_heatmap.png`: 28×28 cosine similarity matrix between per-layer top-128 right singular vectors.
- `depth_rank_spearman.csv`: per-layer effective rank r̂_Q^ℓ vs layer index, Spearman ρ.
- `ohaa_alignment.csv`: per-layer projection norms of CF3 static outlier channels onto V_Q^ℓ.

Load-bearing computation: `torch.linalg.svd(W_stack.to(torch.float32), full_matrices=False)` where W_stack is shape (57344, 2048). The SVD is the only expensive step; all secondary metrics are matrix products on already-computed singular vectors.

### 12. Downstream implications

**If GO (cumvar ≥ 0.50)**:
- Unlocks the 70B cascade: W_Q compression of ≥2× (0.50) to ≥15.6× (0.85 CQBS threshold) is now motivated.
- Unlocks CQBS-IO (S3/Track B): the GGUF file re-layout for sequential IO of the shared basis and thin coefficients is now actionable.
- Unlocks GHPI: the gauge-framing for head-permutation is validated as a compression-relevant structure.
- Motivates CF13 (v2): "cross-layer W_Q basis is shared" as a confirmed structural finding.
- Enables R3-A cascade Rung 2: per-layer coefficient quality check (C_L condition number, ΔNLL at K=128 reconstruction).

**If NO-GO (cumvar < 0.30)**:
- Class-level kill on cross-layer W_Q basis sharing.
- CQBS (Track B), CQST, GHPI, XLQB all close simultaneously — the entire C1 cluster is resolved negative.
- Redirects Track A attention to: WDLA-RESCUE (audacious cascade, independent of W_Q sharing), ZKAM (KV-compute side), ASAKS (KV dedup).
- Does NOT kill CF11's within-layer finding (K=128 global GO remains valid).
- Does NOT kill JVOC/ATTN-SPECTRUM (W_V/W_O characterization is independent).

Runner-up for Stage 6 escalation if Stage 6 rejects this pick: **A1/GHPI** (19.0 points). GHPI is structurally distinct — it asks whether gauge-fixed head coordinates reveal cross-layer lanes — and does not require the stacked-SVD cumvar threshold. GHPI can run independently with the same SVD outputs.

### 13. Provenance

- **Originating orientation**: R (Reach) via R3-A/XLQB. Stage 2 total: 13 (A1=2, A2=2, A3=3, A4=3, A5=3).
- **Convergence cluster**: C1 — Cross-layer W_Q subspace alignment. Four-orientation convergence: R (XLQB), F (CQBS), C (CQST), A (GHPI). Strongest representative with the 70B cascade arithmetic: XLQB. Cheapest measurement representative: CQBS (Track B). Selected candidates represent the full cluster.
- **Stage 4 gap-idea**: Not Stage 4 originated; Stage 4 declared this a "saturation-exhausted frame" for GENERATING new variants (SE-1) but confirmed the experiment is unmeasured and valuable. The Stage 4 framing supports running XLQB/CQBS rather than generating more variants.
- **Frame-novelty bonus path**: Not applicable (convergence weight carries XLQB; frame bonus not the primary driver).
- **Runner-up**: A1/GHPI (19.0).

---

---

# TRACK B SELECTION

---

## Round 002 / Track B — F2/CQBS: Cross-Layer W_Q Basis Sharing (First-Principles)

### 1. Title and one-line description

Round 002 / Track B — F2/CQBS: Does the 57344×2048 stacked W_Q matrix have cumvar(K=128) ≥ 0.85, confirming a single shared subspace enabling 15.6× W_Q storage reduction?

### 2. Class tags

`compression-lr` | `convergence-cluster-C1` | `measurement-structural` | `elegance:subspace-alignment`

### 3. Hypothesis (1–2 paragraphs)

The 28 W_Q matrices in Qwen3-1.7B-Base each project the 2048-dim residual stream into queries for 16 attention heads. CF11 confirmed that globally (across all heads within a layer), K=128 suffices to capture enough variance to hold ΔNLL ≤ +0.98 nats — meaning all 16 heads collectively collapse to ~one head's worth of query subspace. The cross-layer question: do all 28 layers share that SAME 128-dimensional subspace? If so, the model learns a single approximately 128-dim "universal query direction space" that each layer samples via a thin coefficient matrix. This is more compressive than CF11: rather than 28 × 128-dim per-layer subspaces, there would be 1 × 128-dim global basis shared by all layers. The test is a single SVD on the 57344×2048 stacked matrix; cumvar(K=128) ≥ 0.85 confirms the shared structure.

The claim is falsifiable in ~20 minutes with no calibration data and no approximation in the test: SVD is exact, cumvar is deterministic, and the threshold is hard. If confirmed, the 15.6× W_Q storage reduction (from 235 MB to 15.1 MB, comprising 0.52 MB shared U and 14.6 MB for all 28 C_L) is a large single-matrix residency lever. The inference-time computation is also improved: W_Q x = U (C_L x) allows C_L (128×2048) to be applied per-layer while U (2048×128) is applied once, reducing per-layer W_Q GEMV cost from 2048×2048 to 128×2048 for the variable part.

### 4. Smallest test

**Model**: Qwen3-1.7B-Base bf16
**Calibration corpus**: Not required — pure weight-space SVD measurement
**Eval corpus**: Not required for this rung

**Protocol** (shared with Track A XLQB; single combined script):

1. Load W_Q for all 28 layers from the Qwen3-1.7B-Base bf16 checkpoint.
2. Stack into a 57344×2048 matrix in float32.
3. Compute SVD: `torch.linalg.svd(W_stack, full_matrices=False)` → S (2048,), V^T (2048×2048).
4. Compute cumvar(K) for K ∈ {32, 64, 96, 128, 192, 256, 384, 512}.
5. Secondary: per-layer SVDs (28 separate 2048×2048 SVDs) for the CQST α_cross metric, the QCDS Spearman, and OHAA alignment check.
6. Permutation control: row-shuffle the 57344×2048 matrix, re-compute cumvar(K=128).
7. Random-init control: initialize a fresh Qwen3-1.7B architecture with default init, stack its W_Q, compute cumvar(K=128).
8. Multi-seed for secondary calibration metrics: 3 random seeds for the 200-token calibration sample used in OHAA.

**Output path**: `experiments/stage0/ladder_v2/round2_XLQB_CQBS/` (shared with Track A)
**Script**: `scripts/cqbs_xlqb_combined.py` (same script as Track A)

**Wall-clock**: ~38 minutes total including controls (same as Track A estimate above).
**8h gate**: PASSES (38 min << 8h).

Track A and Track B share the same execution. The single run produces outputs for both tracks; threshold interpretation diverges:
- Track A GO: cumvar(K=128) ≥ 0.50
- Track B GO: cumvar(K=128) ≥ 0.85

### 5. Go threshold (Track B / CQBS)

**GO**: cumvar(K=128) ≥ 0.85 on the 57344×2048 stacked W_Q matrix.

At cumvar ≥ 0.85:
- A single shared basis U ∈ R^{2048×128} reconstructs ≥85% of the total squared Frobenius mass of all 28 W_Q matrices.
- Per-layer coefficient matrices C_L ∈ R^{128×2048} reconstruct the per-layer residuals.
- Storage: 0.52 MB (U) + 28 × 0.52 MB (C_L) = 15.1 MB total, vs 28 × 8.39 MB = 235 MB baseline. **15.6× reduction on W_Q.**
- Inference bandwidth: per-layer cost drops from 2048×2048 multiply (4 MB read) to 128×2048 multiply (0.5 MB read) for C_L after U is applied once. ~16× per-layer bandwidth reduction on W_Q.
- Advance to: coefficient reconstruction quality check (Rung 2, ~1h), then NLL evaluation at K=128 (Rung 3, ~2h).

Tied to CF11: the within-layer 128-dim finding must compose with the cross-layer alignment for GO to be meaningful.

**SOFT-GO**: cumvar(K=256) ≥ 0.85. The shared basis requires K=256 instead of 128. 8× compression (vs 15.6×). Still highly material for 70B deployment.

### 6. No-go threshold (Track B / CQBS)

**NO-GO**: cumvar(K=128) < 0.70.

Class-level kill: "cross-layer W_Q basis sharing" as a post-training compression primitive is dead at any practical K. The 28 W_Q matrices have rotating per-layer subspaces despite within-layer head-redundancy. This is a decisive experiment: the result applies to the entire C1 convergence cluster (XLQB, CQBS, CQST, GHPI, and Stage 4's VQWQ all depend on the shared basis existing).

Structural finding produced regardless of outcome: the cumvar curve across K values provides the first direct measurement of the cross-layer W_Q spectral distribution for Qwen3-1.7B-Base. This is an empirical substrate contribution regardless of whether GO or NO-GO — it joins CF11 as a precision measurement of the attention-weight spectrum.

CQBS-IO (S3) is moot on NO-GO. GHPI's cross-layer lane claim loses motivation. VQWQ (S1) loses its CF11+CQBS composition motivation but the codebook coverage question is independently valuable.

### 7. Ambiguous-zone follow-up

**GRAY (0.70 ≤ cumvar(K=128) < 0.85)**:

Run the per-layer-pair cosine similarity heatmap (already computed in the combined script). Determine whether:
(a) A connected subgraph of ≥14 layers has mean cross-layer cosine similarity ≥ 0.7 among their top-128 singular vectors. If yes: partial-layer-cluster sharing survives. The shared basis applies within the cluster; layers outside the cluster maintain independent subspaces. Track B advances as SOFT-GO with the cluster-specific shared basis.

(b) The cumvar at K=256 is ≥ 0.85. If yes: SOFT-GO with larger K. Advance to coefficient reconstruction at K=256.

Follow-up runtime: The cosine heatmap is already in the combined script output. Reading it is 5 minutes. If a 30-min per-cluster cumvar recompute is needed, add 30 min. Total disambiguation: ≤ 35 min.

### 8. Kill criteria (Stage 6 amendment slot)

Stage 6 should reject this pick if:
- **Hidden prior art**: a paper published between Stage 3 (2026-05-09) and Stage 6 that explicitly runs a stacked-SVD on a deployed transformer's W_Q and reports cumvar results. The 24h window has been respected; Stage 6 must re-check before the experiment runs.
- **Skeptic-controls gap**: if the permutation and random-init controls are missing from the actual run (see Section 9), Stage 6 must reject the cumvar GO as potentially architectural.
- **Numerical precision error**: the stacked SVD must be run in float32 (not bf16); bf16 accumulation errors in SVD can produce spuriously high cumvar at small K. If the script uses bf16 throughout, Stage 6 should require float32 rerun before accepting the result.
- **C_L condition number failure**: if, on GO, the per-layer coefficient matrices C_L have condition number κ > 1000, the reconstruction is numerically ill-conditioned even at low K. Stage 6 should check this before advancing to Rung 2.

### 9. Skeptic-controls declaration

The experiment claims: "W_Q cross-layer basis alignment is a TRAINED-MODEL structural property, not an artifact of (a) row-permutation invariance of SVD, (b) random-init architecture behavior, or (c) within-layer head-redundancy structure alone."

All three v2 skeptic controls apply:

**Permutation control**: Row-permute the 57344-row stacked matrix randomly (shuffling rows scrambles the cross-layer grouping while preserving within-layer Frobenius norm and per-matrix statistics). Compute cumvar(K=128) of the permuted matrix.

Expected permuted cumvar: higher than true random (0.063) because within-layer head-redundancy is preserved in per-layer blocks of 2048 rows, even after cross-layer row shuffle. A reasonable estimate: ~0.35–0.45 (cumvar from 28 independent 2048×2048 blocks with within-layer K=128 concentration, each contributing roughly K/d = 128/2048 of global variance but spread across different singular vector directions).

GO criterion with gap: real cumvar ≥ 0.85 AND real cumvar ≥ permuted cumvar + 0.35. If real is 0.87 and permuted is 0.40, the gap is 0.47 — clear GO. If real is 0.87 and permuted is 0.82, the gap is 0.05 — NOT a GO, because the concentration is explained by within-layer block structure, not cross-layer alignment.

**Random-init control**: Generate a fresh Qwen3-1.7B-Base with identical architecture but random weights (kaiming_normal initialization or Hugging Face default). Stack its 28 W_Q matrices. Compute cumvar(K=128).

Expected random-init cumvar: close to the permuted-matrix estimate (~0.35–0.45) because random-init W_Q matrices have Gaussian-like spectra with no cross-layer alignment. If random-init cumvar is ≥ 0.50 (same as Track A's GO threshold), the GO interpretation must shift: the structure is architectural, not trained.

GO criterion with gap: real cumvar ≥ 0.85 AND real cumvar ≥ random-init cumvar + 0.35.

**Multi-seed**: The primary SVD measurement has no stochastic element — it is deterministic. Multi-seed is documented as not applicable to the primary measurement. For calibration-dependent secondary metrics (OHAA alignment projection, which uses 200 calibration tokens), run with 3 random seeds for the token-sample selection and report mean ± std. The OHAA alignment score must hold (mean ≥ 0.30 in >50% layers) across all 3 seeds.

Justification for N/A on primary: SVD of a fixed weight matrix is a deterministic computation with no random element. The model weights are a fixed artifact; no sampling is involved in the stacked SVD. "Not applicable" is justified per the STAGE5_SELECTOR.md criterion.

### 10. Runtime estimate

| Phase | Time |
|---|---|
| Model load (W_Q, 224 MB) | 3 min |
| Stacked SVD (float32, LAPACK) | 12 min |
| Per-layer SVDs (28 ×) | 5 min |
| Secondary metrics | 2 min |
| Permutation control SVD | 10 min |
| Random-init control (generate weights + SVD) | 8 min |
| Multi-seed calibration for OHAA (3 seeds × 200 tokens) | 3 min |
| Post-processing, CSV/PNG | 1 min |
| **Total** | **~44 min** |

Well within 8h gate. Track A and Track B share this single run.

### 11. Script identification

`scripts/cqbs_xlqb_combined.py` — same script as Track A. No separate script needed for Track B.

The script outputs a `cumvar_table.csv` with three columns: `original`, `permuted`, `random_init` — each containing cumvar(K) for the relevant matrix. Track B GO condition reads the `original` value at K=128 and checks the gap against the `permuted` and `random_init` columns. The script does not need modification to serve both tracks.

### 12. Downstream implications

**If GO (cumvar ≥ 0.85)**:
- Unlocks 15.6× W_Q storage reduction for Qwen3-1.7B. At Qwen3-72B (extrapolation, NOT load-bearing for this rung): W_Q saving of ~10 GB.
- CQBS-IO (S3): GGUF file re-layout for the compressed W_Q format is immediately actionable after GO.
- VQWQ (S1): the VQ-codebook coverage experiment in the CQBS-projected 128-dim space is now well-motivated.
- S2/PASS: the W_Q class entry in the per-class precision schedule can be set with confidence to INT4 global + basis sharing, rather than INT4 global alone.
- Motivates CF13 (v2): "Qwen3-1.7B W_Q matrices share a cross-layer 128-dim basis" as a confirmed structural finding.
- Rung 2 (1h): Compute C_L condition numbers and test coefficient reconstruction quality (Frobenius error of W_Q^ℓ ≈ U C_L).
- Rung 3 (2h): End-to-end NLL evaluation at K=128 shared basis.

**If NO-GO (cumvar < 0.70)**:
- Class-level kill on cross-layer W_Q basis sharing.
- CQBS, XLQB, CQST, GHPI all resolve negative simultaneously.
- CQBS-IO is moot.
- Track B redirects cleanly to F1/JVOC (runner-up, 27.2 points): the W_V/W_O fused-operator measurement is independent of this result and proceeds immediately.
- Track A redirects to A1/GHPI (runner-up, 19.0 points): GHPI's gauge-fixing framing can still measure per-head cross-layer alignment in sorted coordinates, which is a different question from the stacked-SVD cumvar.
- CF11's within-layer K=128 finding is not affected by this NO-GO.

**Runner-up for Stage 6 escalation**: F1/JVOC (27.2 points). JVOC is independent of the stacked-SVD measurement and can run immediately after CQBS. Stage 6 should prime JVOC as the next experiment if CQBS GO motivates the Track B cascade, OR if CQBS NO-GO redirects Track B to the W_V/W_O characterization path.

### 13. Provenance

- **Originating orientation**: F (First-Principles) via F2/CQBS. Stage 2 total: 13 (A1=2, A2=2, A3=3, A4=3, A5=3). Stage 2 judge noted "strongest A4 of the cluster" — the construction is auditable in <10 lines: reshape 28 W_Q tensors, concatenate, SVD, read cumvar.
- **Convergence cluster**: C1 — Cross-layer W_Q subspace alignment as a universal compression primitive. Four-orientation convergence: F (CQBS), R (XLQB), C (CQST), A (GHPI). CQBS carries A4=3 (most explicit construction, lowest probability of hidden precondition) and Stage 3 confidence +2 (binary settler in 20 min, no ambiguity in the measurement protocol).
- **Stage 4 gap-idea**: SE-1 in Stage 4 confirmed the C1 cluster frame is "saturation-exhausted for generation" (all orientations have already converged on the same experiment) but is NOT saturation-exhausted for execution (the measurement has never been run). Stage 4 explicitly recommends running CQBS as the settlement experiment.
- **Elegant-equivalence multiplier**: Applied (1.2×). The `subspace-alignment` tag from Stage 3 M2 reflects that the shared-basis construction is an exact algebraic reduction: W_Q^ℓ x ≈ U (C_L x) where the approximation error is precisely 1 - cumvar(K). The elegance is constructive.
- **Runner-up**: F1/JVOC (27.2 points).

---

*End of Stage 5 Selector — Run 002*
