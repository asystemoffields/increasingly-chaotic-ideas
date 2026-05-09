# Stage 3 — Deep Research — Run 002
Run: 002 | Date: 2026-05-09 | Researcher: Sonnet claude-sonnet-4-6

---

## Preamble

This document pressure-tests every Stage 2 advancer from run_002 using the 11-section per-idea structure from STAGE3_DEEP_RESEARCHER.md. Depth allocation per the task brief:

- **Deep treatment (~800–1200 words)**: F2/CQBS (C1 convergence rep), F1/JVOC (C2 convergence rep), C1/OHAA (C3 convergence rep), A6/RGPS (Path 4 elegant-equivalence), R5-A/WDLA-RESCUE (audacious cascade).
- **Standard treatment (~400–600 words)**: remaining Path 1 advancers (R3-A/XLQB, R6-B/ATTN-SPECTRUM, F3/RMGF, F4/SSIF, A1/GHPI, A2/ASAKS, R2-B/RAOK-SCALE, C2/CQST, C3/LGOP, C4/QCDS, F5/WDCA, C6/RKGP, R1-A/QKJB).
- **Light treatment (~250–400 words)**: wildcards (C5/ROPE, A4/RSLK, F6/RSGO), substrate cluster (U1–U5, A5/CAWD).

---

---

## DEEP TREATMENT

---

## F2 / CQBS — Cross-Layer W_Q Basis Sharing (First-Principles)

**Convergence C1 strongest representative | Track B | Path 1 + Path 2**

### 1. Mechanism decomposition into load-bearing claims

- **M1**: The 28 W_Q matrices in Qwen3-1.7B (each 2048×2048) stack into a 57344×2048 matrix whose singular value decomposition measures cross-layer basis concentration.
- **M2**: If the cumulative variance at K=128 of the stacked matrix is ≥ 0.85, a single shared basis U ∈ R^{2048×128} plus 28 small coefficient matrices C_L ∈ R^{128×2048} reconstructs all W_Q layers — a 16× reduction on W_Q storage.
- **M3**: At inference, computing W_Q x = U (C_L x) allows the shared basis to be applied once; per-layer cost reduces from (2048×2048)·x to (2048×128)·(C_L x) — cheap serial composition.
- **M4**: The within-layer head-redundancy finding (CF11: 16 heads → ~128-dim subspace per layer) is the within-layer anchor; the cross-layer question is whether those 28 within-layer 128-dim subspaces are mutually aligned.

### 2. Per-claim prior-art status

**M1 (stacked SVD measurement protocol)**
Status: **NOVEL** — as of 2026-05-09. "Share Your Attention" (arXiv:2508.04581, Aug 2025) proposes MASA: structured weight sharing decomposing Q/K/V/O matrices via shared "dictionary atoms," trained from scratch. This is a TRAINING-TIME weight sharing design. It does NOT measure whether pre-trained weights accidentally share a basis via a post-hoc stacked SVD. TensorLLM (arXiv:2501.15674, IJCNN 2025) applies Tucker decomposition within a single layer across heads, not across layers. "Cross-layer Attention Sharing" (arXiv:2408.01890, Aug 2024) shares KV but not Q across layers. The CQBS measurement — SVD of the stacked post-hoc matrix on a trained model — is NOVEL.
Closest cousin pair: MASA (arXiv:2508.04581) + CF11 (AQFKV, v2 run_002 internal). Value-add: MASA is training-time and uses dictionary decomposition; CQBS asks whether the emergent post-training basis is accidentally shared without any architectural mandate.

**M2 (16× residency reduction from shared basis)**
Status: **NOVEL** — the residency arithmetic and storage format (single shared U + thin per-layer C_L) follows directly from M1 but has no published precedent for post-training deployment.

**M3 (inference efficiency from shared basis amortization)**
Status: **ADJACENT** — ALBERT (Lan et al., 2020) ties all layer weights identically during training; this is a special case K=1 (single coefficient matrix, identity). CQBS has K=128 per-layer residuals, which is structurally different and less restrictive.

**M4 (within-layer CF11 anchor for cross-layer hypothesis)**
Status: **PARTIAL** — CF11 is an internal v2 empirical finding (not published). The cross-layer extension is unpublished.

**Elegance-class tag on M2:** `subspace-alignment` — the mechanism exploits an accidental subspace alignment across 28 layers to fold storage.

### 3. Frame-mismatch check (CF9)

No imported theorem with non-trivial preconditions. The stacked SVD is a standard linear algebra procedure. No precondition requires verification. CF9: CLEAR.

### 4. Calibration ill-conditioning pre-flight (CF10)

CQBS involves NO calibration fitting. The stacked SVD is a deterministic computation on the weight matrices themselves. No least-squares fit; no calibration data. CF10: NOT APPLICABLE.

### 5. Residency arithmetic in detail

Qwen3-1.7B bf16:
- Baseline W_Q per layer: 2048×2048×2 = 8.39 MB. All 28 layers: 235 MB.
- Shared basis U: 2048×128×2 = 0.52 MB (stored once).
- Per-layer coefficient C_L: 128×2048×2 = 0.52 MB × 28 = 14.6 MB.
- Total CQBS W_Q storage: 15.1 MB vs 235 MB baseline = 15.6× reduction.
- KV cache at 4K context, INT8: 28 × 2 × 4096 × 1024 = 234 MB (unchanged by CQBS).
- Activation footprint: batch=1, d=2048 at bf16 = 4 KB (unchanged).
- Attention bandwidth saving per token: (235 MB - 15 MB) / 11.5 GB/s ≈ 19 ms/token → ~52 tok/s headroom on 1.7B ceiling.

Qwen3-72B projection (hypothetical, CF13/CF15 NOT load-bearing):
- W_Q all layers: 80 × 8192×8192×2 = 10.7 GB.
- Under CQBS: 0.52 MB (shared) + 80 × 512×8192×2 = ~655 MB total.
- Saving: 10.0 GB on W_Q alone.
- Conservative branch (if cumvar is 0.80 not 0.85, shared K required = 256): storage doubles to ~1.3 GB, saving halves to ~9.4 GB — still material.

### 6. Smallest-test sharpening

- Script: `scripts/cqbs_stack_svd.py`
- Model: Qwen3-1.7B-Base bf16
- No calibration corpus needed (pure weight measurement)
- Protocol: load W_Q per layer (28 layers × 2048×2048 bf16), stack rows into 57344×2048 matrix, run `torch.linalg.svd` (full_matrices=False), compute cumvar at K=64,128,256.
- Layers/matrices: W_Q only; W_K, W_V, W_O as secondary measurements in same pass.
- Output: `experiments/stage0/ladder_v2/round1_CQBS/cumvar_K128.txt`
- Wall-clock estimate: ~20 minutes on Ryzen 5 7530U (57344×2048 SVD via scipy/numpy; dominated by 800 MB I/O).
- Verdict on 8h gate: PASSES easily at ~20 min.

### 7. Refined go/no-go thresholds

- **GO**: cumvar(K=128) ≥ 0.85. Advance to coefficient reconstruction and per-token NLL evaluation at K=128 shared basis.
- **NO-GO**: cumvar(K=128) ≤ 0.70. Layers span independent subspaces; cross-layer sharing is infeasible. Class-level kill: "cross-layer W_Q basis sharing" as a compression primitive. Per-layer CF11 result survives.
- **GRAY (0.70 < cumvar < 0.85)**: Run per-layer-pair cosine similarity heatmap (secondary metric: mean cosine sim of top-128 singular vectors between layer i and j). If cosine > 0.7 for any connected component spanning ≥14 of 28 layers, partial sharing on that subgraph survives. Follow-up: 30-min cosine similarity scan; resolves whether layers cluster into sharing groups.

### 8. Updated risk profile

| Risk | Mitigation |
|------|-----------|
| Subspaces rotate across layers despite CF11 within-layer concentration | The PCA stacking catches any rotation: if bases rotate, cumvar < 0.70 and the experiment is decisive |
| Coefficient matrices C_L are not well-conditioned at K=128 | Measure condition number of each C_L; flag if kappa > 1000; fallback to K=256 |
| Cross-layer sharing degrades deep-layer performance disproportionately | Measure ΔNLL per layer independently; weight cost by layer index in go/no-go |

Cheapest falsification: measure cumvar(K=128) on the stacked matrix. If ≤ 0.70, the entire cascade dies in 20 minutes.

### 9. Two-paper composition flag

Closest cousin pair: **CF11 (AQFKV internal measurement) + MASA (arXiv:2508.04581)**. MASA shows that training-time dictionary sharing of Q/K/V/O at 66.7% parameter reduction works. CF11 shows within-layer head-redundancy collapses W_Q to 128 dims. CQBS's value-add: post-hoc measurement asking whether the EMERGENT basis is cross-layer aligned — a structural measurement, not a training-time design choice. The composition is not trivially obvious because MASA requires training and CQBS requires only SVD; they are complementary data points, not redundant.

### 10. Verdict

**REFINE** — cumvar measurement is the load-bearing go/no-go. No CF9/CF10 failure modes. Prior art is ADJACENT (training-time sharing) not PRE-EMPTING (post-training measurement). Smallest test is 20 minutes. Stage 5 should weight this heavily: it is the fastest decisor for the entire Convergence C1 cluster.

---

## F1 / JVOC — Joint W_V/W_O Spectral Collapse

**Convergence C2 strongest representative | Track B | Path 1 + Path 2**

### 1. Mechanism decomposition into load-bearing claims

- **M1**: The per-layer fused operator M_L = ∑_h W_O^h W_V^h (shape 2048×2048) captures the joint spectral structure of the V-O composition independently of per-matrix per-head decomposition.
- **M2**: If r_99(M_L)/d ≤ 0.75 in ≥20 of 28 layers, the fused operator is compressible without retraining, enabling a truncated SVD approximation M_L ≈ U_K Σ_K V_K^T.
- **M3**: At inference, head-concatenated attention output A = (∑_h softmax(Q_h K_h^T) V_h) W_O can be rewritten using the fused low-rank M_L to reduce total compute and storage.
- **M4**: The fused-operator approach avoids the per-head per-matrix failure mode (CF11 kills per-head W_Q rank truncation at K=64); by working on the summed object, head-level redundancy is automatically aggregated.

### 2. Per-claim prior-art status

**M1 (fused operator construction)**
Status: **NOVEL** — as of 2026-05-09. "Predicting LLM Compression Degradation from Spectral Statistics" (arXiv:2604.18085, April 2026) discusses spectral rank of W_Q/W_K/W_V/W_O independently and finds rank×compression-ratio predicts degradation. It does NOT measure the fused sum ∑_h W_O^h W_V^h. Eigen Attention (arXiv:2408.05646) performs SVD of KV projections in a joint low-rank space but requires training-time modification. MLA (DeepSeek-V2) shares KV via a training-time design; JVOC is post-hoc. The fused post-training SVD measurement on ∑_h W_O^h W_V^h is NOVEL.

**M2 (compressibility threshold)**
Status: **NOVEL** — subject to empirical confirmation. The r_99/d ≤ 0.75 threshold is a falsifiable hypothesis grounded in CF11's W_K result (r_99/d ≈ 0.79) as the upper bound.

**M3 (inference-time factored computation)**
Status: **ADJACENT** — MLA uses a related factored computation but at training time. Low-Rank KV Attention (arXiv:2601.11471, Jan 2026) preserves full-rank shared projection with low-rank head-specific residuals. JVOC's post-training fused-operator version is distinct.

**M4 (head aggregation avoids per-head kill)**
Status: **NOVEL** — the specific argument that summing over heads before SVD escapes the CF11 per-head NO-GO is an original structural insight. `elegance-class: algebraic-identity` — summing over heads is an exact algebraic identity (the attention output is exactly the sum) that folds head-level redundancy into the SVD.

### 3. Frame-mismatch check (CF9)

No external theorem imported. SVD applied to the fused operator is standard. The only precondition is that W_O and W_V have compatible shapes for head summation — true by construction (W_O^h ∈ R^{d_model × d_head}, W_V^h ∈ R^{d_head × d_model}, product ∈ R^{d_model × d_model}). CF9: CLEAR.

### 4. Calibration ill-conditioning pre-flight (CF10)

No calibration fitting. Pure weight SVD computation. CF10: NOT APPLICABLE.

### 5. Residency arithmetic in detail

Qwen3-1.7B:
- W_V all layers: 28 × 2048×2048×2 = 235 MB.
- W_O all layers: 28 × 2048×2048×2 = 235 MB.
- Fused compressed (K=512): U ∈ R^{2048×512} + V^T ∈ R^{512×2048} per layer = 28 × (2 × 2048×512×2) = 28 × 4 MB = 112 MB for both U and V factors.
- Net saving: 470 MB → 112 MB = 4.2× compression on V+O combined.
- DRAM bandwidth freed: 358 MB / 11.5 GB/s ≈ 31 ms/token → ~32 tok/s headroom uplift on 1.7B.

CF11 risk carried forward: per-matrix compression at K=512 for W_Q gives ΔNLL=+0.20. W_V/W_O may be less concentrated (W_O writes to residual stream, analogous to W_down which CF8 says is full-rank). Conservative branch: if W_O is full-rank (r_99/d ≈ 1.0), only W_V is compressible. In that case: 235 MB W_V at 4× = ~59 MB; W_O stays at 235 MB; total saving ~176 MB vs 470 MB — still worthwhile but not the full joint case.

### 6. Smallest-test sharpening

- Script: `scripts/jvoc_fused_svd.py`
- Model: Qwen3-1.7B-Base bf16
- No calibration corpus
- Protocol: for each layer, load W_V^h (shape 128×2048) and W_O^h (shape 2048×128) for h=0..15, compute M_L = ∑_h matmul(W_O^h, W_V^h) (shape 2048×2048), run SVD, measure r_99/d and cumvar(K=512). Secondary: measure per-matrix W_V SVD and W_O SVD for comparison.
- Output: `experiments/stage0/ladder_v2/round1_JVOC/fused_r99.csv`
- Wall-clock: ~25 min on Ryzen 5 7530U (28 × 2048×2048 SVDs).
- 8h gate: PASSES at ~25 min.

### 7. Refined go/no-go thresholds

- **GO**: r_99(M_L)/d ≤ 0.75 in ≥20 of 28 layers AND ΔNLL ≤ +0.40 nats at K=512 truncated reconstruction.
- **NO-GO**: r_99/d ≥ 0.90 in ≥20 layers. Extends CF8 to include the W_V/W_O class; closes the last known potential attention compression direction. High-value negative.
- **GRAY (0.75 < r_99/d < 0.90)**: Fused operator is partially concentrated. Sweep K from 256 to 1024, find K* where ΔNLL ≤ +0.50 nats; compute residency savings at K*. If K* > 1024 (>50% rank), storage saving < 2× — marginal.

### 8. Updated risk profile

| Risk | Mitigation |
|------|-----------|
| W_O writes to residual stream → full-rank (CF8 analogy with W_down) | Fused operator measurement bypasses per-matrix issue; even if W_O alone is full-rank, M_L = ∑_h W_O^h W_V^h may be concentrated |
| Per-head summation washes out head-specific information | This is fine for compression — we only need to reconstruct the aggregate output accurately |
| ΔNLL assumes per-layer independence; cross-layer accumulation may amplify error | Measure joint W_V+W_O compression at K=512 simultaneously across all 28 layers as single-pass test |

Cheapest falsification: measure r_99(M_L)/d. If ≥ 0.90 in majority of layers, closes the attention-weight compression path cleanly.

### 9. Two-paper composition flag

Closest cousin pair: **CF11 (AQFKV) + MLA (DeepSeek-V2, arXiv:2405.04434)**. CF11 shows W_Q/W_K are compressible post-training. MLA shows joint V-O can share a single low-rank projection at training time. JVOC's value-add: post-hoc measurement on a trained non-MLA model (Qwen3) to determine if the fused V-O emergently has low-rank structure — bridging the training-time (MLA) insight to post-training deployment.

### 10. Verdict

**REFINE** — algebraically exact construction, no CF9/CF10 issues, ≤25 min experiment, closes the last unmeasured attention-weight class. High structural value regardless of outcome. Stage 5 should weight this as the second most decisive experiment after CQBS.

---

## C1 / OHAA — Outlier-Highway Attention Alignment

**Convergence C3 representative | Track B | Path 1**

### 1. Mechanism decomposition into load-bearing claims

- **M1**: The 2 static outlier channels (CF3 K=0.1%, Jaccard=0.718) correspond to specific standard basis vectors e_{i1}, e_{i2} ∈ R^{2048}.
- **M2**: For each layer ℓ, the projection ‖e_i^T V_Q^ℓ_{K=128}‖₂ ≥ 0.3 tests whether the outlier channel direction lies within the W_Q dominant subspace (top-128 right singular vectors of W_Q^ℓ).
- **M3**: If alignment ≥ 0.3 in >50% of layers, a single quantization exception basis covers both the outlier channel and the W_Q residual, eliminating separate bookkeeping.
- **M4**: The alignment is a structural property of how training placed the persistent activation spike directions relative to the dominant query-projection directions.

### 2. Per-claim prior-art status

**M1 (outlier channel identity)**
Status: **PARTIAL** — CF3 measures Jaccard dynamicity (internal v2 finding). Published: LLM.int8() identifies channel-static outliers; SmoothQuant uses them; PrefixQuant (arXiv:2410.05265) separates channel-wise from token-wise. But the specific K=0.1% vs K=1% threshold finding is v2-internal.

**M2 (alignment projection test)**
Status: **NOVEL** — no published method projects activation outlier channels onto W_Q right-singular-vector subspace to test dimensional overlap. SmoothQuant migrates scale; OHAA tests dimensional alignment. Structurally distinct object.

**M3 (unified exception basis)**
Status: **NOVEL** — the collapsed bookkeeping argument is novel. As of 2026-05-09 no paper proposes unifying the activation quantization exception mask with the attention weight low-rank basis via alignment measurement.

**M4 (training-dynamics interpretation)**
Status: **NOVEL** — no published measurement of whether training places outlier channels inside or orthogonal to the W_Q head-shared subspace.

`elegance-class: subspace-alignment` on M3 — the mechanism exploits an accidental subspace alignment between activation-space outlier channels and weight-space dominant directions.

### 3. Frame-mismatch check (CF9)

No external theorem. Dot-product projection is elementary. CF9: CLEAR.

### 4. Calibration ill-conditioning pre-flight (CF10)

No calibration fitting. The outlier channel identity and the W_Q SVD are both derived from existing measurements (CF3 and the CQBS experiment). CF10: NOT APPLICABLE.

### 5. Residency arithmetic in detail

OHAA does not reduce residency per se — it eliminates bookkeeping overlap. The practical payoff is a quantization-quality improvement: if the outlier channels are inside the W_Q subspace, the 2-channel FP16 sidecar required by RAOK is already within the W_Q low-rank projection, meaning those 2 channels need no separate handling during W_Q application. For RAOK-SCALE (R2-B), this means the sidecar overhead of 18 × 2 bytes/token/layer (36 bytes/token/layer × 28 = 1008 bytes/token) can potentially be reduced: the static tier is already represented.

Residency impact: small on 1.7B (elimination of a ~1 KB/token sidecar). Structural impact: determines whether RAOK and W_Q compression can share infrastructure.

### 6. Smallest-test sharpening

- Script: `scripts/ohaa_alignment.py`
- Model: Qwen3-1.7B-Base bf16
- Protocol: (a) identify i1, i2 from CF3 (already known — 2 static outlier channels at K=0.1%); (b) load W_Q per layer, compute top-128 right SVD (V_Q, shape 2048×128); (c) compute ‖e_{i}^T V_Q‖₂ for both outlier channels; (d) count fraction of layers with both ≥ 0.3.
- Free to run if CQBS runs first (shares the 28-SVD pass; V_Q already computed).
- Output: `experiments/stage0/ladder_v2/round1_OHAA/alignment_scores.csv`
- Wall-clock: ~2 min additional after CQBS (dot-product check is negligible given V_Q already computed).
- 8h gate: PASSES trivially.

### 7. Refined go/no-go thresholds

- **GO**: ‖projection‖₂ ≥ 0.3 for both outlier channels in >50% of layers (14 of 28). Advance: RAOK and W_Q compression can share the same 2-channel exception basis.
- **NO-GO**: ‖projection‖₂ < 0.10 across all layers. Outlier channels are orthogonal to W_Q dominant subspace — two independent bookkeeping structures required. RAOK proceeds as designed; W_Q compression proceeds separately. Not a kill for either; closes the composition hypothesis.
- **GRAY (0.10–0.30)**: Partial alignment. Report the mean projection norm; if > 0.20 in majority of layers, partial shared basis may still reduce sidecar cost.

### 8. Updated risk profile

| Risk | Mitigation |
|------|-----------|
| Outlier channels are activation-space (residual stream); W_Q right-singular vectors are weight-space — alignment may be coincidental | The measurement directly tests the coincidence claim; no assumed mechanism |
| K=0.1% gives only 2 channels — insufficient statistical power | Test at K=1% (20 channels) as secondary measurement; if any of the top-20 align, weaker but real coupling |
| CF3 outlier identity is for Qwen3-1.7B-Base; may not generalize to larger models | Generalization is a follow-up question; 1.7B measurement is the primary gate |

Cheapest falsification: the 2-channel projection check is 2 minutes of computation after the CQBS SVD pass.

### 9. Two-paper composition flag

Closest cousin pair: **SmoothQuant (arXiv:2211.10438) + CF11 (AQFKV)**. SmoothQuant uses outlier channels for activation-weight scale migration. CF11 finds W_Q has a 128-dim head-shared subspace. OHAA value-add: tests whether the outlier channel DIRECTIONS (not just scales) lie within the W_Q subspace — a dimensional alignment question that neither paper addresses.

### 10. Verdict

**REFINE** — free experiment (piggybacked on CQBS), no CF9/CF10 risks, purely structural measurement. Even on NO-GO the result closes a structural hypothesis and cleanly decouples RAOK from W_Q infrastructure.

---

## A6 / RGPS — Residual-Stream Gauge-Fixed Precision Scheduling

**Path 4 elegant-equivalence | Track B | gauge-exploitation sub-class**

### 1. Mechanism decomposition into load-bearing claims

- **M1**: The residual stream h_L has an orthogonal gauge symmetry O(d): any rotation P_L applied to h_L and propagated through all reading/writing weight matrices leaves outputs unchanged.
- **M2**: The PCA of the calibration residual-stream covariance C_L = E[h_L h_L^T] yields a canonical gauge P_L where h_L has unit covariance in the most-active directions.
- **M3**: In the PCA-gauge basis, W_gate^L P_L has column mass more concentrated in the top-k columns than raw W_gate^L — enabling higher-precision bits for those columns without additional parameters.
- **M4**: The top-25% PCA columns of W_gate^L P_L account for ≥80% of column L2-norm mass (the concentration hypothesis to test).
- **M5**: Mixed-precision INT8 on top-25% + INT4 on bottom-75% in PCA-gauge matches INT8 quality at ~0.625 bpw effective weight — a quality-per-bit improvement over both INT4 and INT8 uniform.

### 2. Per-claim prior-art status

**M1 (orthogonal gauge symmetry)**
Status: **PARTIAL** — QuaRot (arXiv:2404.00456, NeurIPS 2024) uses Hadamard rotations applied to the residual stream for outlier suppression; SpinQuant (arXiv:2405.16406) learns rotations; paroquant (arXiv:2511.10645, ICLR 2026) uses pairwise rotations. All of these apply random or learned orthogonal transforms to the residual stream. The existence of the O(d) gauge symmetry in this context is well-established by these papers.

**M2 (PCA as canonical gauge choice)**
Status: **ADJACENT** — QuaRot uses Hadamard (a fixed random orthogonal). SpinQuant learns the rotation. RGPS proposes using the data-driven PCA of the residual-stream covariance as the canonical choice. No paper proposes this specific gauge choice. However, SpinQuant searches a broader space and might discover a PCA-like solution. The differentiation: RGPS's PCA is PRINCIPLED (the residual-stream covariance eigenbasis maximally decorrelates the activation distribution) rather than random or learned.

**M3 (column mass concentration in PCA-gauge)**
Status: **NOVEL** — the question "does the PCA gauge concentrate W_gate column mass relative to the raw-basis W_gate?" has not been measured. QuaRot measures activation-quantization noise reduction, not weight column-mass concentration.

**M4 (80% mass in top-25% threshold)**
Status: **NOVEL** — specific empirical threshold on Qwen3; not measured in any published work.

**M5 (0.625 bpw effective quality)**
Status: **ADJACENT** — mixed-precision quantization is published (QuaRot produces mixed-precision weight+activation quantization). RGPS's differentiation is the PCA-gauge motivation for the split. Whether this motivation produces better results than QuaRot's random Hadamard split is the empirical question Stage 2 flagged.

`elegance-class: gauge-exploitation` on M1–M3 — the gauge symmetry is real (O(d) acting on residual stream), and PCA is the canonical gauge-fixing choice for this symmetry group.

**Stage 2 pressure-test question**: does the PCA gauge choice produce measurably different (higher) spectral concentration than a random Hadamard rotation? If not, RGPS is a principled VARIANT of QuaRot, not a distinct primitive.

### 3. Frame-mismatch check (CF9)

No imported theorem with non-trivial preconditions. PCA of a covariance matrix is always well-defined. The O(d) gauge symmetry holds for any orthogonal transform — the RMSNorm operation does constrain this symmetry (it's O(d) not GL(d)), but PCA is orthogonal by construction. CF9: CLEAR.

**One subtle check**: RGPS claims "RMSNorm is approximately the identity in PCA-gauge." This is not exact — RMSNorm normalizes the VECTOR NORM, not the coordinate-wise variance. In PCA-gauge, the covariance is diagonal, but the per-sample norm ‖h_L‖ is not unit. The claim should be softened: "in PCA-gauge, the PCA columns with large eigenvalues dominate the norm, so the RMSNorm implicitly weighs them heavily." This is approximately true but not exact. Stage 4 should note this approximation.

### 4. Calibration ill-conditioning pre-flight (CF10)

RGPS requires computing the calibration covariance C_L = E[h_L h_L^T] from calibration data.
- n_params_to_fit for the covariance: d×d = 2048×2048 = 4.2M elements (symmetric = 2.1M independent). But this is NOT a least-squares fit — it is a sample mean. The eigendecomposition of a sample covariance is well-defined at any N.
- For reliable eigenvector estimation of a 2048×2048 covariance matrix, standard rules of thumb suggest N ≥ 10×d = 20,480 samples. At 200 tokens × 2048 positions = 409,600 hidden states — more than sufficient.
- CF10: PASSES with even 200 calibration tokens.

### 5. Residency arithmetic in detail

Qwen3-1.7B, W_gate:
- Baseline W_gate INT4: 28 × 6144×2048 × 0.5 bytes = 172 MB.
- RGPS (25% INT8 + 75% INT4): 28 × (1536×2048×1 + 4608×2048×0.5) = 28 × (3.15 + 4.72) MB = 28 × 7.87 MB = 220 MB.
- RGPS costs +48 MB vs INT4 but claims to match INT8 quality (346 MB). Net saving vs INT8: 126 MB on W_gate.
- Rotation storage: 28 × 2048×2048×2 = 235 MB — this overhead DOMINATES and would negate all savings if stored per layer. Must amortize: store P_L as a product of givens rotations or use a single global PCA (one P shared across all layers, 8 MB). If cross-layer PCA is stable (CQBS's underlying question), a single P suffices.
- Total model impact: if rotation is global (one P), overhead = 8 MB. Net RGPS 1.7B: 220 MB W_gate + 8 MB rotation = 228 MB vs 172 MB INT4.

**Critical dependency**: RGPS is residency-competitive with INT4 ONLY if a global rotation can replace per-layer rotations. This ties RGPS to CQBS's finding — if the cross-layer W_Q subspace is stable, the residual-stream PCA is also likely stable across layers.

### 6. Smallest-test sharpening

- Script: `scripts/rgps_pca_concentration.py`
- Model: Qwen3-1.7B-Base bf16
- Calibration: 200 tokens from WikiText-2 train split
- Protocol: (a) run 200-token forward pass, record {h_L} at all 28 layers; (b) compute C_L = h_L^T h_L / n_tokens; (c) eigendecompose C_L → P_L (2048×2048); (d) compute W_gate^L P_L; (e) sort columns by L2 norm; (f) compute fraction of total column mass in top-512 columns (top-25%).
- Secondary: compare to random Hadamard rotation (Q from scipy.stats.ortho_group) to test QuaRot baseline.
- Output: `experiments/stage0/ladder_v2/round1_RGPS/pca_concentration.csv`
- Wall-clock: ~40 min (200-token forward + 28 × 2048×2048 eigendecompositions).
- 8h gate: PASSES at ~40 min.

### 7. Refined go/no-go thresholds

- **GO**: top-25% PCA columns account for ≥80% of total column mass AND PCA concentration > Hadamard concentration by ≥10%. RGPS is doing principled work; advance to mixed-precision quantization prototype.
- **NO-GO**: top-25% PCA columns account for ≤50% of total mass. PCA gauge does not concentrate W_gate. The gauge is already PCA-like (well-aligned weight and activation spaces), meaning no rotation can improve concentration. KILL RGPS; confirms QuaRot's Hadamard approach is already near-optimal for this model.
- **GRAY (50–80%)**: PCA concentrates somewhat. Compare to Hadamard baseline: if PCA is measurably better, continue with REFINE. If not, DOWNGRADE to "principled QuaRot variant" (ENGINEERING-INTEGRATION).

### 8. Updated risk profile

| Risk | Mitigation |
|------|-----------|
| Per-layer PCA costs 235 MB rotation storage, negating gains | Use global PCA (single P); test whether cross-layer covariance PCA is stable (CQBS finding resolves this) |
| PCA concentration is no better than Hadamard (QuaRot's approach) | Include Hadamard comparison in smallest test; if equal, DOWNGRADE |
| ΔNLL from INT8/INT4 mixed in PCA-gauge may be worse than INT4 uniform | Test ΔNLL on 1-layer prototype before full model; cheap smoke test |

Cheapest falsification: if top-25% column mass < 50% after PCA rotation, entire cascade dies at 40 minutes.

### 9. Two-paper composition flag

Closest cousin pair: **QuaRot (arXiv:2404.00456) + SpinQuant (arXiv:2405.16406)**. QuaRot uses fixed Hadamard rotation; SpinQuant learns the rotation. RGPS's value-add: proposes PCA of residual-stream covariance as the CANONICAL gauge-fixing choice, motivated by the orthogonal gauge symmetry argument — principled rather than random or trained. Whether this produces empirically better column-mass concentration is the experiment's question. If it does not, RGPS is an ENGINEERING-INTEGRATION of QuaRot+covariance-PCA without structural value-add.

### 10. Verdict

**REFINE** — conditional. The gauge-exploitation framing is genuine and the PCA canonical choice is principled. The experiment directly falsifies or confirms the QuaRot differentiation claim. If the falsification test in section 8 passes (PCA > Hadamard), RGPS is a structural advance; if not, DOWNGRADE to engineering-integration. Stage 5 should weight this as a high-elegance idea with a clean binary experiment.

---

## R5-A / WDLA-RESCUE — Cross-Scale Affine Surgery (Rescoped)

**Track A | Path 1 | Audacious cascade**

### 1. Mechanism decomposition into load-bearing claims

- **M1**: A rank-64 affine map A_L: R^{1024} → R^{2048} with n_params = 1024×64 + 64×2048 ≈ 196K can be fit with 10K calibration tokens × 2048 output dims = 20.5M values → well-conditioned (ratio 0.0096 << 1).
- **M2**: The fitted map, when applied layer-by-layer to Qwen3-0.6B hidden states, produces hidden states whose held-out R²_eval > 0.85 vs Qwen3-1.7B target (the CF10 fix addresses WDLA's failure).
- **M3**: IF R²_eval > 0.85, then the corrected 0.6B forward pass approximates the 1.7B forward pass closely enough to reduce PPL.
- **M4**: The mechanism scales to 7B → 72B with rank-256 corrections at 220 MB overhead — a 2.85 GB total runtime footprint.

### 2. Per-claim prior-art status

**M1 (well-conditioned rank-64 affine)**
Status: **NOVEL** — inference-time cross-scale affine surgery with forced low-rank conditioning. Model Stitching (arXiv:2506.06609, NeurIPS 2025 spotlight) transfers SAE features BETWEEN models via affine maps but is training-time (stitching layer trained) and uses full-rank affine. WDLA-RESCUE's forced rank-64 at inference time with no additional training is structurally distinct.

**M2 (R²_eval > 0.85 on held-out)**
Status: **NOVEL** — the specific prediction that rank-64 conditioning fixes WDLA's failure is a direct replication of CF10's lesson. No published paper has attempted inference-time cross-scale hidden-state correction at rank-64 with this conditioning check.

**M3 (PPL quality transfer)**
Status: **NOVEL but highly uncertain** — the claim that corrected 0.6B approximates 1.7B quality is speculative. Even with R²_eval > 0.85 in hidden state space, the downstream impact on PPL is non-trivial. A high R² in hidden state space does not guarantee quality preservation; it means the affine map is not overfitting, but quality loss from the dimension mismatch (1024 → 2048) is a separate question.

**M4 (7B → 72B scaling)**
Status: **SPECULATIVE** — no evidence that rank-256 corrections scale across size classes with quality gap recovery. CF10 gives no support for or against this.

### 3. Frame-mismatch check (CF9)

No external theorem imported. Ridge regression is appropriate for a well-conditioned linear system. The key precondition: the 0.6B → 1.7B hidden state mapping must be APPROXIMATELY affine for the rank-64 map to capture meaningful signal. This is the load-bearing precondition that is NOT guaranteed:
- 0.6B has d=1024; 1.7B has d=2048. The affine map is NOT a square rotation — it projects from a lower-dimensional to a higher-dimensional space. The additional 1024 dimensions in the 1.7B space cannot be recovered from the 0.6B signal — they must be predicted from the 0.6B directions. Whether those predicted dimensions are meaningful depends on whether the information is genuinely encoded in the 0.6B representation.
- This is the "dimensionality blocker" flagged in Stage 1. It is a structural concern, not a calibration concern. If the additional 1024 dimensions carry independent information (not representable from the 0.6B space), the rank-64 map will produce those dimensions as near-zero — which the R²_eval > 0.85 threshold will STILL PASS if the target 1.7B dimensions are also near-zero in that direction.

**CF9 verdict**: the frame is NOT obviously broken, but the precondition (affine relationship between model spaces) is empirically uncertain. The R²_eval check is the right gate for this specific concern. CF9: CONDITIONAL.

### 4. Calibration ill-conditioning pre-flight (CF10)

Explicit CF10 analysis:
- n_params_to_fit: rank-64 A_L = 1024×64 + 64×2048 = 65,536 + 131,072 = 196,608.
- n_independent_samples: 10,000 tokens (stated).
- n_output_dims: 2048.
- n_independent_samples × n_output_dims: 10,000 × 2048 = 20,480,000.
- Ratio: 196,608 / 20,480,000 = 0.0096. Well below the 0.1 underdetermination threshold.
- Ridge=0.1 specified: appropriate (not needed for conditioning but provides additional regularization).
- CF10: PASSES. The CF10 mitigation from Stage 6 amendment is correctly applied.

**However**: the calibration must actually contain 10K DISTINCT token positions, not 10K tokens from a single short passage (which would be highly autocorrelated). State corpus: 10K tokens from WikiText-2 train split, sampled across 50 passages of 200 tokens each.

### 5. Residency arithmetic in detail

Qwen3-0.6B at 3 bpw: ~225 MB DRAM.
Rank-64 affine corrections: 28 layers × 196,608 params × 2 bytes = ~11 MB.
Total: ~236 MB DRAM. Compare to Qwen3-1.7B at 3 bpw: ~637 MB.
If quality is comparable, this is a 2.7× residency reduction for the "inference body."

For 7B → 72B projection: NOT supportable with current evidence. That cascade rung requires separate empirical validation (XLQB, CQBS, JVOC findings are preconditions). The conservative branch: 0.6B → 1.7B is the testable unit; do not assume scaling.

### 6. Smallest-test sharpening

- Script: `scripts/wdla_rescue_rank64.py`
- Model pair: Qwen3-0.6B-Base bf16 + Qwen3-1.7B-Base bf16 (same 28-layer depth)
- Calibration corpus: WikiText-2 train, 10K tokens sampled from 50 diverse 200-token passages
- Eval corpus: WikiText-2 test, 1K tokens (held-out)
- Protocol: (a) run both models on calibration; (b) fit rank-64 affine A_L for each layer using ridge=0.1; (c) run eval pass; (d) measure R²_eval per layer; (e) gate: R²_eval > 0.85 in layers 10–28 (as specified in Stage 6 amendment).
- Output: `experiments/stage0/ladder_v2/round1_WDLA_RESCUE/r2_eval_per_layer.csv`
- Wall-clock: ~3 hours (two model loads + 10K-token calibration pass + fit + eval).
- 8h gate: PASSES at ~3 hours.

### 7. Refined go/no-go thresholds

- **GO**: mean R²_eval > 0.85 across layers 10–28. Advance to PPL measurement (Rung 2).
- **NO-GO**: mean R²_eval < 0.5 in most layers (catastrophic; worse than zero-mean predictor). Frame is structurally broken; the dimensional information content of the 0.6B representation is insufficient to predict the 1.7B hidden states. KILL WDLA frame.
- **GRAY (0.50 < R²_eval < 0.85)**: Affine relationship exists but reconstruction is imperfect. Report per-layer R² distribution; if early layers (0–5) have R²_eval > 0.85 but deep layers fall below, the frame survives for shallow correction only (modest PPL impact).

### 8. Updated risk profile

| Risk | Mitigation |
|------|-----------|
| Dimensionality mismatch (1024→2048) means additional 1024 dims cannot be predicted | Check per-direction R² in the 2048-dim output space; if the extra 1024 dims have R²≈0, the map is only recovering the within-manifold component |
| 10K tokens may not be diverse enough (all WikiText-2 = single domain) | Sample across 5 domains: wiki, code, math, news, fiction (2K tokens each) |
| R²_eval > 0.85 does not guarantee PPL improvement | Run PPL comparison directly in Rung 2 (the real gate) |

Cheapest falsification: per-layer R²_eval measurement. If R² < 0.5 in deep layers, frame is broken. Runtime: 3 hours.

### 9. Two-paper composition flag

Closest cousin pair: **WDLA (internal v2 failure) + Model Stitching (arXiv:2506.06609)**. Model Stitching shows affine maps can transfer features between models at training time. WDLA-RESCUE applies this at inference time with proper rank and conditioning. Value-add: (1) inference-time deployment (no additional training), (2) explicit CF10 conditioning fix (rank-64 forced), (3) quality-gap recovery as the metric rather than feature-transfer accuracy. The proposal is NOT an obvious composition of the two — the inference-time application and the conditioning fix are the novel structural moves.

### 10. Verdict

**REFINE** — CF10 is correctly addressed. CF9 precondition (affine relationship in hidden state space) is testable and not obviously broken. Prior art is ADJACENT, not PRE-EMPTING. Smallest test at 3 hours passes the 8h gate. The cascade's audacity (7B→72B deployment) is speculative but the first rung is tractable and informative regardless of outcome.

---

---

## STANDARD TREATMENT

---

## R3-A / XLQB — Cross-Layer W_Q Basis Sharing (Reach)

**Convergence C1 rep (Reach angle) | Track A | Path 1**

### 1–2. Claims and prior-art

Essentially the same measurement as CQBS (F2) but framed as a cascade toward 70B deployment. Load-bearing claims identical to CQBS M1–M4. Prior-art status: same as CQBS. XLQB adds the go threshold at cumvar ≥ 0.50 (vs CQBS's 0.85) and the fallback (per-layer-pair cosine similarity as a softer signal). The lower threshold makes XLQB's GO condition easier to trigger but also less structurally compelling — a shared basis at cumvar=0.50 gives only ~2× compression, not the 16× CQBS claims at 0.85.

### 3. CF9 check

No imported theorem. CF9: CLEAR.

### 4. CF10 check

No calibration fitting. CF10: NOT APPLICABLE.

### 5. Residency arithmetic

Same as CQBS at its conservative branch (cumvar = 0.50–0.70). At 70B: W_Q at 2× compression (not 16×) = 5.35 GB vs 10.7 GB. Still meaningful. The 70B cascade arithmetic is conditional on 5+ additional rungs, all unverified.

### 6. Smallest test

Identical to CQBS. Shared script. XLQB's added metric: Spearman correlation between cumvar and layer-pair distance.

### 7. Thresholds

- **GO**: cumvar(K=128) ≥ 0.50 (weaker than CQBS; motivates the per-layer-pair analysis).
- **NO-GO**: cumvar(K=128) < 0.30 (no shared structure at any level).
- **GRAY**: 0.30–0.50. Run layer-pair analysis.

### 9. Two-paper composition flag

Same as CQBS. XLQB is largely subsumed by CQBS once the shared experiment runs. Its additional value is the 70B arithmetic cascade — the per-layer-pair fallback is unique. Not a trivial composition.

### 10. Verdict

**REFINE** — but XLQB's first rung is IDENTICAL to CQBS's first rung. They should be run as a single experiment. XLQB advances primarily for its 70B cascade architecture and per-layer-pair fallback analysis; the measurement itself deduplicates with CQBS. Stage 5 should run CQBS as the settlement experiment and use XLQB's cascade arithmetic as the context for interpreting results.

---

## R6-B / ATTN-SPECTRUM — W_V/W_O Rank Cascade

**Convergence C2 rep (Reach angle) | Track B | Path 1**

### 1–2. Claims and prior-art

Targets per-matrix W_V and W_O spectra (as distinct from JVOC's fused operator). Load-bearing claims:
- **M1**: W_V at K=512 global achieves ΔNLL ≤ +0.50 nats.
- **M2**: W_O at K=512 may be full-rank (analogy with W_down).
- Prior art: "Predicting LLM Compression Degradation from Spectral Statistics" (arXiv:2604.18085) discusses W_Q/W_K/W_V/W_O independently but does not provide per-matrix empirical results for Qwen3. Status: **NOVEL** for Qwen3-specific measurements. `elegance-class: none` (pure measurement).

### 3. CF9 check

No imported theorem. CF9: CLEAR.

### 4. CF10 check

Pure SVD truncation, no calibration fitting. CF10: NOT APPLICABLE.

### 5. Residency arithmetic

See JVOC §5. If W_V compresses at K=512 but W_O does not: 235 MB → ~118 MB for W_V, W_O stays at 235 MB. Total attention saving: 235 MB vs JVOC's 358 MB. Still valuable.

### 6. Smallest test

20-min experiment (same script as AQFKV with W_V/W_O substituted). Output: per-layer r_99/d and ΔNLL at K=512.

### 7. Thresholds

- **GO**: W_V ΔNLL ≤ +0.50 nats at K=512. W_O ΔNLL ≤ +1.00 nats at K=512 (conditional).
- **NO-GO**: Both W_V and W_O r_99/d ≥ 0.90. Extends CF8 to full attention class; no compression possible on V/O side.

### 9. Two-paper composition

Closest cousins: **CF11 (AQFKV) + arXiv:2604.18085**. Value-add: empirical Qwen3-specific measurements that published spectral statistics papers don't provide.

### 10. Verdict

**REFINE** — ATTN-SPECTRUM is the per-matrix measurement complement to JVOC's fused-operator measurement. Both should run. ATTN-SPECTRUM's W_O full-rank prediction (if confirmed) constrains JVOC's interpretation. 20-min experiment, clean go/no-go.

---

## F3 / RMGF — RMSNorm Gauge Freedom Exploitation

**Track A | Path 3 frame-novelty | Path 1**

### 1–2. Claims and prior-art

- **M1**: W_Q_gauged = W_Q × diag(g_RMSNorm) is an algebraically exact fold (no approximation).
- **M2**: r_99(W_Q_gauged)/d ≤ 0.55 in majority of layers (spectrum concentration hypothesis).
- Prior art check: "An Extra RMSNorm is All You Need for Fine Tuning to 1.58 Bits" (arXiv:2505.08823) uses RMSNorm for quantization stability but does NOT fold g into downstream weights or measure spectrum change. SmoothQuant folds activation scale into weights for activation range; RMGF folds the NORM GAIN g (a weight tensor) into the downstream weight spectrum. **M1: PARTIAL** (SmoothQuant absorbs scale but for activation range, not spectrum concentration). **M2: NOVEL** — spectrum-concentration consequence of the fold has not been measured.
- `elegance-class: gauge-exploitation` on M1 — the fold is an exact gauge transformation of the norm gain parameter.

### 3. CF9 check

No imported theorem. The algebraic fold `W_Q_gauged = W_Q diag(g)` is exact. The spectrum concentration claim is empirical. CF9: CLEAR.

### 4. CF10 check

No calibration fitting. CF10: NOT APPLICABLE.

### 5. Residency arithmetic

Direct saving from eliminating g: 28 × 2048 × 2 bytes = 115 KB — negligible. Payoff is downstream compression quality if spectrum concentrates. If r_99/d drops from 0.63 to 0.50, K=96 achieves the same quality as K=128 ungauged — 25% fewer coefficients in the shared basis (from CQBS).

### 6. Smallest test

Pre-check: g_L variance statistics (5 min). Main: 28-SVD pass on W_Q_gauged vs CF11 baseline (~20 min). Total: ~25 min.

### 7. Thresholds

- **GO**: median r_99(W_Q_gauged)/d ≤ 0.55 AND improvement over CF11's 0.63 is statistically significant (paired t-test p < 0.05 across 28 layers).
- **NO-GO**: median ≥ 0.60. g absorption does not reshape the spectrum. Closes the "norm gain as spectrum shaper" route.

### 9. Two-paper composition

Closest cousins: **SmoothQuant (arXiv:2211.10438) + CF11 (AQFKV)**. SmoothQuant absorbs activation scale for quantization range; CF11 shows W_Q spectrum is concentrated. RMGF asks whether folding the norm gain (different from activation scale) further concentrates the already-concentrated W_Q spectrum. Value-add: new object measured (g_L fold, not activation scale fold); new question (spectrum concentration rather than activation range).

### 10. Verdict

**REFINE** — clean algebraic operation, no CF9/CF10 risks, 25-min experiment, novel spectrum-concentration question. Pre-check on g_L variance is essential (if g ≈ 1, fold is no-op; kills in 5 minutes). Path 3 frame-novelty qualifier is intact; no published analog of spectrum-concentration-via-norm-gain-fold.

---

## F4 / SSIF — Softmax Shift Invariance Folding

**Track A | Path 3 frame-novelty | Path 1**

### 1–2. Claims and prior-art

- **M1**: Softmax(x + c·1) = softmax(x) for any scalar c is an algebraic identity — exact.
- **M2**: Per-head mean logit offset μ_Q (computed over calibration distribution) has norm ≥ 5% of row std in ≥15 of 28 layers.
- **M3**: Removing μ_Q reduces Q/K dynamic range and enables tighter activation quantization.
- Prior art: "On the Invariants of Softmax Attention" (arXiv:2605.02907, April 2026) defines the "energy field" as the row-centered attention logit and shows it exhibits invariant properties. This is DIRECTLY RELEVANT: this paper formalizes the shift invariance identity and characterizes the energy field. The paper focuses on mechanism-level invariants for interpretability, not for quantization dynamic-range reduction. **M1: PRE-EMPTED conceptually** (the shift invariance is formalized in this paper), **but the QUANTIZATION APPLICATION (M3) is NOVEL**. Also found: "Intrinsic and Extrinsic Organized Attention: Softmax Invariance and Network Sparsity" (arXiv:2506.15541) explores softmax invariance for attention sparsity, not quantization.
- The RoPE concern (position-dependent offset) is real: for RoPE-based models, Q and K are rotated per position, meaning the "per-head mean offset" varies with position. arXiv:2605.02907 explicitly characterizes this structure. SSIF must be reframed: not a global per-head offset but a per-head per-frequency-band offset (extractable from the energy field's low-rank structure, since arXiv:2605.02907 finds 20 SVD components capture >90% of the energy field variance).

### 3. CF9 check

The shift invariance identity (M1) is exact — no precondition. But the QUANTIZATION application (M3) has a precondition: the per-head mean logit must have STABLE, LARGE norm across calibration tokens. If the mean logit varies substantially per position (due to RoPE), the offset is not a stable removable constant. CF9: CONDITIONAL — requires M2 measurement to confirm the precondition holds.

### 4. CF10 check

SSIF is a measurement-based proposal (measure the mean logit offset). No calibration fitting. CF10: NOT APPLICABLE.

### 5. Residency arithmetic

The shift removal stores a per-head bias (one scalar per head per layer): 28 × 16 × 4 bytes = 1.8 KB. Negligible storage. The payoff is activation quantization quality. If Q/K dynamic range after shift removal is 10% smaller, INT8 quantization error decreases proportionally — affecting KV cache quantization quality at long context.

### 6. Smallest test

30-min forward pass on 200 calibration tokens, record attention logit matrices, compute per-head per-position mean and variance. The key measurement: does the mean logit have a STABLE component across positions (low-rank structure, as arXiv:2605.02907 suggests), or is it purely position-dependent (RoPE-dominated)?

### 7. Thresholds

- **GO**: median ‖mean_head‖/‖row_std_head‖ ≥ 0.05 in ≥15 layers AND the mean has a stable cross-position component (first PC of the energy field accounts for ≥30% of mean-logit variance).
- **NO-GO**: ‖mean‖/‖row_std‖ < 0.02 in all layers. No exploitable offset. Close the "softmax symmetry for quantization range" family.
- **GRAY**: Offset is stable but small (0.02–0.05). Possible benefit for INT4 KV but not INT8. Reframe as long-context-only optimization.

### 9. Two-paper composition

Closest cousins: **arXiv:2605.02907 (Invariants of Softmax Attention) + PrefixQuant (arXiv:2410.05265)**. arXiv:2605.02907 formalizes the energy field invariant; PrefixQuant handles outlier tokens in the KV domain. SSIF's value-add: apply the energy field's stable offset structure to DYNAMIC-RANGE reduction for Q/K activation quantization — a quantization application not addressed in either paper.

### 10. Verdict

**REFINE** — but prior-art landscape is more crowded than Stage 2 assessed. arXiv:2605.02907 (May 2026!) directly formalizes the shift invariance property for this class of models. SSIF must explicitly position against this paper: the quantization application remains NOVEL (the invariants paper focuses on interpretability, not compression). Smallest test: 30 min. RoPE concern requires the measurement to stratify by position-band (not just global mean). Path 3 frame-novelty qualifier narrows: SSIF is now "PARTIAL novel" — the frame is published (arXiv:2605.02907) but the quantization application is unpublished.

**Flag for Stage 4**: revise SSIF to explicitly cite arXiv:2605.02907 and frame the quantization application as the novel contribution beyond the invariants paper.

---

## A1 / GHPI — Gauge-Fixed Head-Permutation Inference

**Convergence C1 rep (Constraint-Alien angle) | Track A | Path 1**

### 1–2. Claims and prior-art

- **M1**: Head-permutation is a Z_{H!} gauge symmetry of multi-head attention (bit-for-bit invariant under head sort + W_O column repermutation). **NOVEL** — while equivariant network literature knows permutation symmetry of independent operations, its use as a gauge-fixing coordinate for LLM attention compression has no published form. "Quantifying LLM Attention-Head Stability" (arXiv:2602.16740, Feb 2026) proposes permutation-invariant head stability metrics but for circuit universality analysis, not for compression.
- **M2**: In gauge-fixed (sorted) coordinates, cross-layer head alignment reveals lane structure. **NOVEL** — the specific claim that sorting by principal direction reveals cross-layer lanes is distinct from existing head-pruning methods (which prune in raw coordinates).
- **M3**: Cross-layer head lanes in gauge-fixed coordinates enable compression of the HEAD dimension (not the weight matrix rank). **ADJACENT** — TensorLLM (arXiv:2501.15674, IJCNN 2025) uses Tucker decomposition across heads in a single layer. GHPI extends this idea across layers via gauge-fixing. Differentiation: Tucker in raw head-permutation-unfixed coordinates may miss cross-layer alignment; GHPI's gauge-fixing is the novel preprocessing.

### 3. CF9 check

No external theorem with non-trivial preconditions. CF9: CLEAR.

### 4. CF10 check

No calibration fitting. Pure weight SVD. CF10: NOT APPLICABLE.

### 5–7. Arithmetic, smallest test, thresholds

- Smallest test: 20-min SVD on all 28 layers × 16 heads × W_Q^h (shape 128×2048). Extract top singular vector per head. Argsort per layer. Compute cross-layer cosine similarity in sorted basis.
- GO: effective rank ≤ 8 in the 28×16 similarity matrix (16× redundancy across the collection).
- NO-GO: effective rank > 64. Heads are genuinely diverse; gauge-fixing adds no leverage.

### 9. Two-paper composition

Closest cousins: **TensorLLM (arXiv:2501.15674) + CF11 (AQFKV)**. TensorLLM applies Tucker decomposition within-layer across heads. CF11 finds within-layer head redundancy. GHPI's value-add: gauge-fixing the head-permutation before measuring cross-LAYER head alignment — a preprocessing step that TensorLLM lacks.

### 10. Verdict

**REFINE** — the gauge-framing is principled and prior art is ADJACENT not PRE-EMPTING. Measurement piggybacks on the CQBS SVD pass. The convergence C1 angle (gauge-fixed coordinates may reveal the same shared subspace as CQBS's stacked SVD) is theoretically interesting. If CQBS confirms a shared cross-layer basis, GHPI's gauge-fixing framing becomes the principled explanation for WHY it exists.

---

## A2 / ASAKS — All-State Append-Only KV Store

**Track A | Path 1**

### 1–2. Claims and prior-art

- **M1**: KV cache is effectively append-only during autoregressive generation (no position is overwritten). **PARTIAL** — this is well-known; all KV cache implementations exploit it for sequential writes.
- **M2**: Content-addressed dedup of near-duplicate K-vectors (cosine ≥ 0.90) reduces KV cache size by ≥25% at 512-token context. **NOVEL** — no published method uses SimHash/cosine-based dedup on K-vectors within the KV cache during inference.
- **M3**: LSM-tree compaction of K,V entries with near-duplicate pointers enables sublinear KV storage in repetitive contexts. **ADJACENT** — prefix caching (in vLLM and PagedAttention) reuses KV entries for shared prefix tokens, which is a form of deduplication. However, prefix caching deduplicates identical token sequences, not semantically similar K-vectors. ASAKS's cosine-threshold dedup is strictly broader. "RadixMLP - Intra-batch Deduplication" (arXiv:2601.15013, Jan 2026) does prefix dedup within a batch, not cosine-threshold dedup across dissimilar positions.
- K-vector dedup rate is the critical unknown. If K-vectors are as token-dynamic as CF3 suggests for activations, dedup rate may be low.

### 3. CF9 check

SimHash collision analysis: for d=128-dim K-vectors with K=64 probe vectors, the false positive rate (two dissimilar vectors sharing 64-bit SimHash) is 2^{-64} per pair — negligible. The true positive rate (cosine ≥ 0.90 maps to hash collision probability ~1-arccos(0.90)/π ≈ 0.72) — adequate for dedup. CF9: CLEAR.

### 4. CF10 check

No calibration fitting. CF10: NOT APPLICABLE.

### 5. Residency arithmetic

At 512-token context, 28 layers × 16 heads × 128 dim × 2 bytes × 512 tokens = 117 MB for K-cache. At 25% dedup rate: ~88 MB. Modest on 1.7B. At 128K context: ~30 GB → ~22.5 GB. The mechanism scales with context length.

### 6. Smallest test

15-min forward pass + pairwise cosine computation on 512-token WikiText-2 prefix. Measure dedup rate per layer.

### 7. Thresholds

- **GO**: ≥25% dedup rate in ≥10 of 28 layers at cosine ≥ 0.90.
- **NO-GO**: <5% dedup rate globally. K-vectors are diverse; append-only framing adds no compression leverage (tiered storage survives independently).

### 9. Two-paper composition

Closest cousins: **KVQuant (arXiv:2401.18079) + DiskANN**. KVQuant quantizes/compresses KV entries; DiskANN indexes embeddings by approximate similarity. ASAKS value-add: applies LSM-tree compaction with content-addressed dedup to KV entries within a single inference session — a paradigm transplant that neither paper considers.

### 10. Verdict

**REFINE** — prior art is ADJACENT on the dedup framing (prefix caching), but cosine-threshold dedup of K-vectors within-session has no direct prior. Smallest test is 15 min. Dedup rate measurement is the binary go/no-go. CF3's token-dynamic activation finding is a risk signal (if activations rotate, K-vectors may too), making the dedup rate measurement essential before investing in the LSM-tree infrastructure.

---

## R2-B / RAOK-SCALE — Activation-Outlier Tiered-Quant Cascade

**Convergence C3 adj. | Track B | Path 1**

### 1–2. Claims and prior-art

- **M1**: K=0.1% static (2 channels, Jaccard=0.718) + K=1% dynamic (18 channels, INT8) + bulk INT4 is the correct tier partition derived from CF3.
- **M2**: This achieves ΔNLL ≤ +0.10 nats vs INT4 activation baseline.

Prior art: "Rethinking the Outlier Distribution in Large Language Models" (arXiv:2505.21670, May 2025) distinguishes massive activations and channel-wise outliers and notes massive activations persist via residual connections. "Outlier-Safe Pre-Training" (arXiv:2506.19697) addresses pre-training. Neither directly measures the K=0.1% vs K=1% Jaccard partition as CF3 does. PrefixQuant (arXiv:2410.05265) handles token-wise vs channel-wise but at a coarser granularity. **M1: PARTIAL** (CF3's K-dependent Jaccard measurement is v2-internal; the tier partition derivation from it is NOVEL). **M2: NOVEL** — empirical quality measurement with this specific partition on Qwen3.

### 3. CF9 check

No imported theorem. The tier boundaries come from measured Jaccard values (CF3). CF9: CLEAR.

### 4. CF10 check

RAOK-SCALE involves activation quantization (per-token dynamic assignment), NOT weight calibration fitting. CF10 applies to least-squares parameter fits. CF10: NOT APPLICABLE here.

### 5. Residency arithmetic

Overhead: 28 × 18 INT8 values/token = 504 bytes/token/layer context overhead. Negligible. Payoff: quality preservation at INT4-equivalent weight residency.

### 6. Smallest test

2-hour ΔNLL measurement on Qwen3-1.7B with instrumented activation quantization using PDAP infrastructure (already built in R2).

### 7. Thresholds

- **GO**: ΔNLL ≤ +0.10 nats vs INT4 activation baseline.
- **NO-GO**: ΔNLL ≥ +0.30 nats (no improvement over naive INT4). CF3's K-dependent partition adds no quality benefit; static channel handling in INT4 already handles the dominant outlier structure.

### 9. Two-paper composition

Closest cousins: **CF3 (v2 internal PDAP finding) + PrefixQuant (arXiv:2410.05265)**. PrefixQuant separates token-wise from channel-wise. RAOK adds a THIRD tier (token-dynamic at K=1%) keyed on the CF3 Jaccard threshold — a finer partition than PrefixQuant's binary split.

### 10. Verdict

**REFINE** — the CF3 Jaccard threshold is a v2-internal measurement with genuine novelty over the published landscape. The PDAP infrastructure from R2 is already built. 2-hour experiment, clean ΔNLL threshold.

---

## C2 / CQST — Cross-Layer W_Q Subspace Telescoping

**Convergence C1 rep (Composition angle) | Track B | Path 1**

### 1–2. Claims and prior-art

CQST adds the CF2+CF11 composition argument: near-parallel residuals (CF2) force similar inputs to W_Q, which should force similar W_Q basis directions across layers. Load-bearing claim: α_cross ≥ 0.7 (mean cross-layer basis alignment). Prior art status same as CQBS — NOVEL for post-training measurement, ADJACENT to MASA. The α_cross metric is more nuanced than CQBS's cumvar and tests directional alignment rather than variance concentration. These are complementary metrics.

### CF9–CF10

No imported theorem. No calibration fit. CLEAR for both.

### Smallest test

Shares the 28-SVD pass with CQBS; only adds pairwise cosine computation (< 2 min additional). Total: ~22 min.

### Verdict

**REFINE** — free rider on the CQBS experiment. The CF2+CF11 composition argument is theoretically the strongest of the C1 convergence cluster. Stage 5 should run CQBS and CQST simultaneously (same script, two output metrics). If α_cross ≥ 0.7 AND cumvar(K=128) ≥ 0.85, both confirming, the cross-layer sharing mechanism is triply validated (XLQB, CQBS, CQST).

---

## C3 / LGOP — Layer-1 Gate-Fold × Outlier-Static Pinning

**Convergence C3 adj. | Track B | Path 1**

### 1–2. Claims and prior-art

- **M1**: Foldable neurons F_1 (|F_1|≈2218, from CF6) have W_up^1[F_1, :] concentrated on the 2 static outlier columns (CF3). Test: ‖W_up^1[F_1, :]_{:, O_1}‖_F / ‖W_up^1[F_1, :]‖_F ≥ 0.5.
- Prior art: No published paper couples CF6's Layer-1 gate-variance anomaly with CF3's static outlier channels. SDZC (internal killed idea) tested only global gate variance. **NOVEL** for the coupling hypothesis. The 2-channel dominance is statistically unlikely to explain 36% gate near-constancy by chance (2/2048 = 0.1% of dimensions), so the coupling is mechanistically specific if confirmed.

### CF9 check

The coupling claim requires that W_up^1[F_1, :]_{:, O_1} (projection of weight onto 2 outlier columns) accounts for gate near-constancy. The precondition is that the static outlier channels have LARGE NORM in h at all times (near-constant input), which drives near-constant gate output. CF3's Jaccard=0.718 at K=0.1% confirms these channels have persistently high-magnitude activations. The precondition is approximately satisfied by CF3. CF9: CONDITIONAL but plausible.

### CF10, residency, smallest test

No calibration fit. 10-min column norm extraction. Free if CQBS runs first (the W_Q SVD doesn't overlap but the model is already loaded). GO: ratio ≥ 0.5. NO-GO: ratio < 0.1.

### Verdict

**REFINE** — mechanistically specific hypothesis, cheap test, novel composition of two v2 internal findings. Even NO-GO is informative (the Layer-1 anomaly has a different cause than outlier channel saturation).

---

## C4 / QCDS — W_Q Cross-Layer Drift Quantization Schedule

**Track B | Path 1**

### 1–2. Claims and prior-art

**M1**: r̂_Q^ℓ (W_Q rank capturing 95% of Frobenius norm) increases monotonically with layer depth. Spearman ρ ≥ 0.7. Prior art: "Predicting LLM Compression Degradation from Spectral Statistics" (arXiv:2604.18085) examines per-layer spectral statistics. If this paper reports per-layer W_Q rank vs depth, QCDS may be ADJACENT. The specific prediction that r̂_Q increases monotonically because CF2's near-parallel residuals force simpler early-layer queries is the novel compositional move.

### CF9, CF10 check

No external theorem. No calibration fit. CLEAR.

### Smallest test

Free: runs in the same 28-SVD pass as CQBS and CQST. Spearman correlation of r̂_Q^ℓ vs ℓ. Total additional cost: negligible.

### Verdict

**REFINE** — free rider on the CQBS experiment. Low cost, high value if the depth-monotone pattern holds (enables per-layer bpw schedule without calibration fitting).

---

## F5 / WDCA — W_down Spectral Audit / Output Subspace

**Track B | Path 1**

### 1–2. Claims and prior-art

- **M1**: The activation-weighted output matrix X_out_L = W_down^L × A_L (where A_L is the post-SwiGLU activation matrix) has cumvar(K=512) ≥ 0.90 in ≥15 layers.
- Prior art: GPTVQ (arXiv:2402.15319) uses Hessian-weighted EM on codebooks; Hessian = X^T X = activation second moment. WDCA targets the OUTPUT subspace (right singular vectors of X_out_L) rather than the input Hessian. These are dual perspectives: GPTVQ minimizes input-weighted error; WDCA minimizes output-subspace approximation error. **PARTIAL** — the concept of output-subspace truncation is adjacent to GPTVQ but the specific measurement (SVD of W_down × A_calibration) is distinct.
- **CF10 explicit check**: F5 itself notes the CF10 risk. The author's mitigation (pure SVD truncation, no least-squares fit) is correct. IF the output-subspace measurement uses SVD truncation only (not a calibration-fitted projection), then CF10 is not triggered. Confirm: the proposal is to truncate W_down to the top-K output directions measured from calibration. This is NOT a fit — it is a deterministic projection of W_down onto its empirically dominant output subspace. CF10: NOT APPLICABLE if implemented as pure SVD truncation.

### CF9 check

No imported theorem with non-trivial preconditions. CF9: CLEAR.

### Smallest test

40-min calibration forward pass + per-layer SVD. GO: cumvar(K=512) ≥ 0.90 in ≥15 layers AND ΔNLL ≤ +0.25 nats at K=512 reconstruction.

### Verdict

**REFINE** — WDCA measures a distinct object (activation-weighted output subspace) from CF8's weight SVD. The CF10 risk is correctly mitigated by pure SVD truncation. Prior art is ADJACENT but not PRE-EMPTING. 40-min experiment, clean threshold.

---

## C6 / RKGP — Residual-Cosine Flatness × W_K Depth-Gradient

**Track B | Path 1**

### 1–2. Claims and prior-art

The KV layer-skip recomputation mechanism (K_{ℓ+1}(t) ≈ W_K^{ℓ+1} h_ℓ(t)) is motivated by CF2's near-parallel residuals (δ_ℓ ≈ 0.14 × ‖h_ℓ‖). Prior art: "Layer-Condensed KV Cache" (arXiv:2405.10637) computes KV at one training-time-designated layer and reuses across layers. RKGP extends this post-hoc via CF2's measured delta norm. Structural difference: Layer-Condensed KV uses training-time coupling; RKGP uses the post-hoc delta norm measurement to determine which layer pairs allow skip recomputation. **M1: ADJACENT** — the layer-skip idea is published; the CF2-grounded delta-norm approximation bound is the novel differential.

### CF9 check

The key claim is ‖W_K^{ℓ+1} δ_ℓ‖ / ‖K_{ℓ+1}‖ ≤ 0.15. This follows from CF2's cos ≈ 0.99 → ‖δ_ℓ‖ ≈ 0.14‖h_ℓ‖ AND ‖W_K^{ℓ+1} δ_ℓ‖ ≤ ‖W_K^{ℓ+1}‖ × 0.14‖h_ℓ‖. The spectral norm ‖W_K^{ℓ+1}‖ = σ_max(W_K^{ℓ+1}) determines whether the delta amplification is small. CF11 shows W_K r_99/d ≈ 0.79, suggesting σ_max is not catastrophically large. But the bound is only tight if σ_max(W_K) × 0.14 << ‖K_{ℓ+1}‖/‖h_ℓ‖. This is not verified. CF9: CONDITIONAL — spectral norm check is needed. The smallest test directly measures this.

### CF10 check

No calibration fitting. CF10: NOT APPLICABLE.

### Smallest test

20-min forward pass, measure relative error ‖K_{ℓ+1}^{approx} - K_{ℓ+1}^{true}‖ / ‖K_{ℓ+1}^{true}‖ per layer pair. This directly tests the CF9 conditional.

### Verdict

**REFINE** — the CF9 conditional is resolved by the smallest test itself. Prior art is ADJACENT (Layer-Condensed KV) but the CF2-grounded post-hoc framing is distinct. 20-min test, long-context payoff motivates the investment.

---

## R1-A / QKJB — Joint Q/K Low-Rank Folding Cascade

**Track A | Path 1**

### 1–2. Claims and prior-art

R1-A's load-bearing claim is that W_V/W_O spectra extend CF11's concentration pattern to all 4 attention matrices, enabling a 4-matrix joint low-rank cascade to 70B deployment. The first rung is identical to ATTN-SPECTRUM (R6-B). The cascade's novel piece is the per-layer ΔNLL independence assumption — the claim that 80 layers of W_V at K=512 give cumulative ΔNLL = 80 × +0.20 ≈ +16 nats is acknowledged as a concern (layers may absorb errors via the residual stream, making the actual cumulative ΔNLL sublinear). **The per-layer independence assumption is the key CF9-adjacent concern**: does the residual skip connection absorb cross-layer rank deficiency errors?

Prior art: same as JVOC + ATTN-SPECTRUM.

### CF9 check

The per-layer ΔNLL independence assumption is a mathematical simplification. The residual stream's skip connection means that an error in one layer's W_V approximation is added to the residual stream and passed to the next layer WITH the skip connection intact. In a well-functioning residual network, later layers can partially compensate for earlier approximation errors. This is NOT a theorem-import problem (CF9 is about imported theory preconditions), but it IS a modeling assumption worth flagging. The safest approach: measure ΔNLL at 28-layer SIMULTANEOUS truncation on 1.7B, then compare to 28 × per-layer individual truncation ΔNLL. If the simultaneous truncation is < 50% of the sum, the independence assumption is conservative.

### CF10 check

No calibration fitting in the first rung. CF10: NOT APPLICABLE for first rung. (Later cascade rungs involving cross-layer basis fitting would require CF10 analysis.)

### Smallest test

Same as ATTN-SPECTRUM (20 min). First rung is purely structural measurement.

### Verdict

**REFINE** — first rung deduplicates with ATTN-SPECTRUM and JVOC. The cascade architecture and 70B arithmetic are the distinctive contributions. Stage 5 should interpret R1-A primarily as a cascade framework that ATTN-SPECTRUM and JVOC measurements feed into; not as an independent experiment.

---

---

## LIGHT TREATMENT

---

## C5 / ROPE — Residual Stream Outlier-Head Excision [FREE SWING]

**Track A | Wildcard | CF-tether suspended**

Load-bearing claim: per-head β_k values (projection of attention output onto outlier channel directions) have CV ≤ 0.3 in >50% of layers — i.e., all heads produce similar outlier-channel projections, enabling factored "outlier attention" + "residual attention" decomposition.

Prior art check: "Quantifying LLM Attention-Head Stability" (arXiv:2602.16740, Feb 2026) measures head stability but NOT head output decomposition by outlier channels. A3 (arXiv:2505.12942) targets attention approximation via logit error minimization. Neither proposes the outlier-channel symmetric β_k decomposition. **Status: NOVEL** for the specific β_k symmetry hypothesis.

CF9: The mechanism uses CF3's static outlier channels as O_1. The precondition: the outlier channels in the RESIDUAL STREAM are relevant after W_O projection. This requires that the static outlier channels in h_L (the input to the attention block) remain identifiable in h_L^{attn} (the output of W_O). This is testable (β_k measurement) but not derived. CF9: CONDITIONAL.

CF10: No calibration fitting. NOT APPLICABLE.

Smallest test: 7-min forward pass + β_k computation. Well within 8h gate.

Verdict: **REFINE — wildcard, CF-tether-suspended, structural floor met.** The β_k symmetry claim is specific and falsifiable. Prior art does not PRE-EMPT. Stage 5 note: if β_k CV ≤ 0.3 is confirmed, the outlier-attention factoring becomes a genuine arch-transposition candidate.

---

## A4 / RSLK — Register-Only Streaming Layer Kernel [FREE SWING]

**Track B | Wildcard | CF-tether suspended**

Load-bearing claim: fused RMSNorm + GEMV streaming kernel achieves ≥1.3× throughput on single MLP layer vs unfused baseline (measured via L1 cache-miss reduction ≥40%).

Stack primitive: L1/L2/L3 cache specs (Ryzen 5 7530U: L1=32KB, L2=512KB, L3=16MB — all public). Hidden state at bf16 = 4 KB fits in L1.

CF9: The prefetcher concern is real — if the Zen3 hardware prefetcher already achieves L1-residency for sequential GEMV access, the kernel adds nothing. The smallest test directly measures this via `perf stat` cache-misses.

CF10: No calibration fitting. NOT APPLICABLE.

Smallest test: 3-hour Python prototype + benchmark. Within 8h gate.

The performance claim (~1.3× throughput) is modest and the mechanism is incremental over existing llama.cpp GEMV. No residency improvement. Payoff is throughput, not residency.

Verdict: **REFINE — wildcard, CF-tether-suspended, structural floor met.** The primitive (L1/L2/L3 cache hierarchy) is real and the kernel design is coherent. However, payoff is likely small (prefetcher may already be doing the work). Low priority for Stage 5 selection.

---

## F6 / RSGO — Residual-Stream Gauge Orbit Folding [FREE SWING]

**Track A | Wildcard | CF-tether suspended**

Load-bearing claim: the optimal orthogonal G^* (sum of layer covariances' eigenbasis) reduces median r_99/d of W_Q^gauged below 0.55 vs CF11's 0.63.

This is RGPS (A6) with a different gauge-fixing procedure: RSGO uses the JOINT eigenbasis of ∑_L C_L (sum of per-layer covariances), while RGPS uses per-layer PCA. The structural difference: RSGO finds a global G^* applied uniformly across all layers; RGPS applies per-layer P_L. If CQBS confirms a stable cross-layer basis, RSGO's global G^* is the natural companion.

Prior art: Same as RGPS (QuaRot, SpinQuant). RSGO's global rotation is closer to QuaRot's single Hadamard rotation than RGPS's per-layer PCA. Differentiation over QuaRot: RSGO uses the empirically principled G^* = eigenvectors of ∑_L C_L vs QuaRot's random Hadamard.

CF9: Same analysis as RGPS. CLEAR.
CF10: Same as RGPS. NOT APPLICABLE.
Smallest test: 45 min. Within 8h gate.

Verdict: **REFINE — wildcard, CF-tether-suspended, structural floor met.** RSGO and RGPS are complementary approaches to the same gauge-fixing problem (global vs per-layer). If both advance, Stage 5 should select ONE for the experiment and use the other as a follow-up. RSGO has a narrower scope than RGPS (no precision scheduling claim; purely spectrum concentration).

---

## Substrate Cluster: U1–U5 (Light Treatment)

All five substrate ideas advance on substrate-specific primitives. Brief assessments:

**U1 / PrefetchVirtualMemory**: prior art closest = Apple LLM-in-a-Flash (arXiv:2312.11514). Differentiation: Win32-specific `PrefetchVirtualMemory` call with application-level scheduling. CF9/CF10: not applicable (no theorem, no calibration fit). Key pre-check: does ik_llama.cpp already overlap I/O? Smallest test: 2 hours. GO: ≥10% latency reduction under NVMe-forced inference. **REFINE.**

**U2 / VirtualLock**: prior art: none in LLM literature. CF9/CF10: not applicable. Key constraint: `SE_LOCK_MEMORY_NAME` privilege. Test: 1 hour. GO: latency CV drops by ≥2× under memory pressure. **REFINE** — most useful as a jitter-reduction primitive for production deployment.

**U4 / NTFS-sparse**: prior art: none in LLM literature. CF9: precondition is that Q2 quantization produces systematically zero blocks. CF8 says MLP weights are full-rank — this CONTRADICTS the zero-block hypothesis. At Q2, blocks normalize per-group, making exactly zero blocks rare. Risk: **HIGH** that GO threshold (≥5% zero blocks) fails. Smallest test: 1 hour. If zero-block fraction < 1%, KILL. **REFINE with LOW confidence** — likely a quick NO-GO that cleanly extends CF8 to Q2 quantization behavior.

**U5 / MEM_LARGE_PAGES**: prior art: none in LLM inference literature. CF9/CF10: not applicable. TLB analysis is sound (Zen3 stlb: 1024 × 2 MB entries = 2 GB coverage). Key constraint: `SeLockMemoryPrivilege`. Test: 2 hours. GO: ≥3% latency reduction on DRAM-resident inference. **REFINE.**

**A5 / CAWD**: prior art: VPTQ (positional codebook) + DiskANN. The SimHash-based content-addressed dedup is ADJACENT. CF8 (MLP weights full-rank) is a direct risk: full-rank means W_gate rows are geometrically diverse, reducing dedup rate to near zero for MLP. W_Q rows (from CF11: 16 heads collapse to ~8 functional types) are more likely to dedup. Smallest test: 10 minutes (SimHash computation on W_Q rows). GO: ≥15% collision rate in W_Q. **REFINE** — pivot scope to W_Q only (not W_gate, which CF8 predicts has negligible dedup rate).

---

---

## VERDICT SUMMARY TABLE

| ID | Title | Track | Verdict | Key Kill Class / Condition |
|----|-------|-------|---------|---------------------------|
| F2/CQBS | Cross-Layer W_Q Basis Sharing (FP) | B | **REFINE** | — |
| F1/JVOC | Joint W_V/W_O Spectral Collapse | B | **REFINE** | — |
| C1/OHAA | Outlier-Highway Attention Alignment | B | **REFINE** | — |
| A6/RGPS | Residual-Stream Gauge-Fixed Precision | B | **REFINE** | Conditional: if PCA concentration ≤ Hadamard → DOWNGRADE |
| R5-A/WDLA-RESCUE | Cross-Scale Affine Surgery (Rescoped) | A | **REFINE** | CF9 conditional (affine precondition) |
| R3-A/XLQB | Cross-Layer W_Q Basis (Reach) | A | **REFINE** | Deduplicates with CQBS; run as cascade context |
| R6-B/ATTN-SPECTRUM | W_V/W_O Rank Cascade | B | **REFINE** | — |
| F3/RMGF | RMSNorm Gauge Freedom | A | **REFINE** | Pre-check: g_L variance (5 min) |
| F4/SSIF | Softmax Shift Invariance Folding | A | **REFINE** | arXiv:2605.02907 partially crowds frame; quantization application remains novel |
| A1/GHPI | Gauge-Fixed Head-Permutation | A | **REFINE** | — |
| A2/ASAKS | All-State Append-Only KV | A | **REFINE** | Dedup rate may be near-zero if K-vectors are as token-dynamic as activations |
| R2-B/RAOK-SCALE | Activation-Outlier Tiered-Quant | B | **REFINE** | — |
| C2/CQST | Cross-Layer W_Q Subspace Telescoping | B | **REFINE** | Free rider on CQBS |
| C3/LGOP | Layer-1 Gate-Fold × Outlier-Static | B | **REFINE** | CF9 conditional (2-channel dominance) |
| C4/QCDS | W_Q Cross-Layer Drift Quant Schedule | B | **REFINE** | Free rider on CQBS |
| F5/WDCA | W_down Spectral Audit / Output Subspace | B | **REFINE** | CF10 NOT triggered if pure SVD truncation |
| C6/RKGP | Residual-Cosine × W_K Depth-Gradient | B | **REFINE** | CF9 conditional: spectral norm amplification |
| R1-A/QKJB | Joint Q/K Low-Rank Folding Cascade | A | **REFINE** | First rung deduplicates with ATTN-SPECTRUM; per-layer independence assumption |
| C5/ROPE | Outlier-Head Excision [FREE SWING] | A | **REFINE** | Wildcard; β_k symmetry is the gate |
| A4/RSLK | Register-Only Streaming Layer [FREE SWING] | B | **REFINE** | Wildcard; likely quick NO-GO if prefetcher already handles |
| F6/RSGO | Residual-Stream Gauge Orbit [FREE SWING] | A | **REFINE** | Wildcard; companion to RGPS |
| U1/PrefetchVM | GGUF Extent Coalescing / PrefetchVirtualMemory | B | **REFINE** | Pre-check: ik_llama.cpp existing I/O overlap |
| U2/VirtualLock | VirtualLock Hot-Tier KV Pinning | B | **REFINE** | Privilege gate |
| U4/NTFS-sparse | NTFS Sparse File Sentinel | B | **REFINE** | Low confidence; CF8 predicts near-zero zero-block fraction |
| U5/MEM_LARGE_PAGES | HugePage Promotion | B | **REFINE** | Privilege gate |
| A5/CAWD | Content-Addressed Weight Dispatch | A | **REFINE** | Pivot scope to W_Q only |

**KILL count: 0** (all prior-art searches found ADJACENT, not PRE-EMPTING, for the load-bearing novel claims; no CF9/CF10 hard failures found at Stage 3 analysis depth).

**REGENERATE count: 0.**

**DOWNGRADE risk: 1 conditional** (A6/RGPS if PCA concentration is not measurably better than Hadamard rotation baseline).

**Total REFINE: 26**

---

## Stage 5 Priority Weighting

**Weight most heavily for Stage 5 selection:**

1. **F2/CQBS** — settles all of Convergence C1 (4 orientations) in 20 minutes; highest information-per-minute of any experiment in the pool. Binary go/no-go with enormous downside value. The C1 cluster holds the largest single residency lever (10+ GB on 70B-class if confirmed).

2. **F1/JVOC** — settles Convergence C2 (3 orientations, W_V/W_O class) in 25 minutes; closes the last unmeasured attention-weight class. Algebraic exactness with `algebraic-identity` elegance class. If W_O is full-rank (NO-GO), cleanly closes the attention-weight-class compression direction and redirects to RAOK/quantization.

3. **R5-A/WDLA-RESCUE** — highest-A1 audacious cascade. The 3-hour R²_eval measurement directly tests whether the entire cross-scale inference-time surgery frame lives or dies. If GO, opens a path to sub-3 GB inference on a 70B-quality signal. If NO-GO (R²_eval < 0.5), permanently closes the WDLA frame.

**Note on prior-art publication velocity (Track A policy)**: SSIF (F4) faces increased prior-art pressure from arXiv:2605.02907 (April 2026). The softmax invariants paper was published AFTER Stage 2 review and materially changes SSIF's novelty position from "frame-novel" to "application-novel." Stage 5 should note this velocity signal and explicitly position SSIF against arXiv:2605.02907 in any experimental write-up.

---

*End of Stage 3 Deep Research — Run 002*
