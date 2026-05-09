# Stage 5 Selector — Run 008 (v2 Sonnet Ladder)

Completed: 2026-05-09. ONE pick per track. Selection algorithm applied: novelty re-verification, convergence weighting, frame-novelty weighting, elegant-equivalence multiplier, runtime/structural-finding weighting.

Inputs: Stage 3 top 3 (C-UFDM 5min, F3-WO-HEADBLOCK-RANK 5min, R8-WVWO-FOLD 20min); Stage 4 top 2 (S5 10min SINQ-balance SVD-rank, S2 30min SwiGLU gate-predicate W_up skip); KILL_LIST (v2-S3-R008-001 F4-SVAL-CONSERVED killed by SINQ; A-INTG downgraded by I-LLM).

---

## Pre-Selection: Re-Verified Novelty (2026-05-09)

**Track A candidates:**

- **C-UFDM**: Stage 3 confirmed NOVEL. Re-check: searched "W_up firing subspace W_Q head-shared subspace principal angle overlap joint input basis LLM compression 2026." No new paper as of 2026-05-09. arXiv:2604.22778 (Spectral Lifecycle, April 2026) and arXiv:2603.13314 (Linear Predictability, March 2026) both confirmed prior art at Stage 3; neither measures the CF1×CF11 cross-module principal-angle overlap. Status: **NOVEL** (confirmed at Stage 5, 2026-05-09).

- **R8-WVWO-FOLD**: Stage 3 confirmed NOVEL. Re-check: searched "W_V W_O product rank SVD Qwen3 attention compression post-training 2026." arXiv:2604.22778 mentions V/O "uniform compression" in training but does not measure post-training product rank or propose W_VO folding. A3 (arXiv:2505.12942) fuses W_VO for activation-aware SVD compression at 70B scale — **ADJACENT**. A3 applies activation-aware Frobenius error minimization on the full fused product; WVWO-FOLD measures pure weight-matrix product rank and tests whether the fold saves storage. Structural distinction: A3 compresses W_VO using Hessian-weighted loss; WVWO-FOLD tests whether r_99(W_VO)/d < 0.75 without calibration. These are structurally different objects. Status: **ADJACENT but not pre-empted** (2026-05-09). Stage 6 must verify A3 coverage explicitly.

- **S5 (SINQ-balance SVD-rank)**: Stage 4 origin. Re-check: SINQ (arXiv:2509.22944) covers quantization-scale optimization; SVD-rank prediction from Sinkhorn balance is explicitly NOT in SINQ. Status: **NOVEL** (confirmed at Stage 5, 2026-05-09).

**Track B candidates:**

- **F3-WO-HEADBLOCK-RANK**: Stage 3 confirmed NOVEL. Re-check: searched "W_O per-head chunk rank Qwen3 GQA output projection SVD low-rank compression 2026." arXiv:2604.22778 covers global V/O spectrum, not per-head-chunk W_O rank. Status: **NOVEL** (confirmed, 2026-05-09).

- **R8-RAOK-70B**: Stage 3 confirmed NOVEL. Re-check: searched "three-tier activation quantization K=0.1% static K=1% dynamic Jaccard stratification 70B CPU inference 2026." No new paper. Status: **NOVEL** (confirmed, 2026-05-09).

- **S2 (SwiGLU gate-predicate W_up row skip)**: Stage 4 origin. Re-check: searched "SwiGLU gate predicate per-token W_up row skip NVMe offload 2026." SABLE-v1 (pre-pipeline kill) used historical frequency tables, not per-token gate output. Status: **NOVEL** (confirmed, 2026-05-09).

---

## Scoring Summary

### Track A

| Candidate | S2+S3conf | Conv mult | Elegance mult | Frame-novelty | NO-GO bonus | Final |
|---|---|---|---|---|---|---|
| C-UFDM | 13+2=15 | 1.0 | 1.2 (constructive subspace-alignment) | +2 (A2=3, fresh frame) | +1 | **21.0** |
| R8-WVWO-FOLD | 12+2=14 | 1.0 | 1.2 (constructive algebraic-identity) | +1 (A2=2) | +2 (NO-GO extends CF8 boundary to W_VO, potential CF entry) | **19.8** |
| S5 (SINQ SVD-rank) | 10+2=12 | 1.0 | 1.2 (constructive gauge-exploitation) | +1 | +1 | **15.4** |

Notes: C-UFDM S3 confidence = +2 (5-min binary settler; cleanest smallest-test in run). WVWO-FOLD S3 confidence = +2 (20-min binary settler, binary r_99 threshold). S5 S3 confidence = +2 (10-min, single binary bit). C-UFDM: convergence multiplier 1.0 (solo orientation, C). WVWO-FOLD: 1.0 (R solo; R8-WVOS-SPEC is convergence backup but WVWO-FOLD is Track A, not B). WVWO-FOLD NO-GO bonus = +2 because the NO-GO (W_VO full-rank ≥ 20/28 layers) is a candidate CF entry — extends the CF8 family to the V-O composition, a class-level structural finding. C-UFDM NO-GO bonus = +1 (class-level finding that MLP and attention input spaces are orthogonally optimized — informative but narrower scope than a CF extension).

**Track A pick: C-UFDM.** Score 21.0 vs WVWO-FOLD 19.8. Primary driver: C-UFDM is a genuine two-finding coupling (CF1×CF11) with no published cross-measurement analog; 5-min runtime; NO-GO is structurally informative; AERO and A3 do not cover the input-side subspace overlap claim. Runner-up: R8-WVWO-FOLD (named for Stage 6 kill-and-escalate; if Stage 6 finds A3 coverage invalidates WVWO-FOLD, escalate to WVWO-FOLD's Stage 5 plan directly — its structural finding case is independent of A3's Hessian-weighted framing).

### Track B

| Candidate | S2+S3conf | Conv mult | Elegance mult | Frame-novelty | NO-GO bonus | Final |
|---|---|---|---|---|---|---|
| F3-WO-HEADBLOCK-RANK | 12+2=14 | 1.5 (C1 3-orientation convergence: R8-WVOS-SPEC, F1-WVWO-SPECTRUM, F3) | 1.2 (constructive algebraic-identity) | +1 | +1 | **26.2** |
| R8-RAOK-70B | 12+2=14 | 1.0 | 1.1 | +1 | +2 (NO-GO kills 3-tier design, constrains activation quantization class) | **18.4** |
| S2 (gate-predicate W_up skip) | 10+2=12 | 1.0 | 1.1 (conserved-quantity) | +1 | +2 | **17.2** |

Notes: F3-WO-HEADBLOCK-RANK convergence multiplier = 1.5 (Cluster C1 is a 3-orientation cluster: R8-WVOS-SPEC from R orientation, F1-WVWO-SPECTRUM from F orientation, F3-WO-HEADBLOCK-RANK from F orientation — three independent orientations converged on W_V/W_O characterization). The convergence multiplier is the dominant scoring factor. F3 NO-GO = +1 (all heads r_99 ≥ 100 extends the CF12/CF8 full-rank pattern to per-head W_O chunks — informative but narrower than a standalone CF entry without broader NLL measurement). R8-RAOK-70B NO-GO = +2 (kills the 3-tier design; collapses to LLM.int8() 2-tier territory — class-constraining for all dynamic-tier activation quantization). S2 NO-GO = +2 (skip rate < 1% means SwiGLU gating is not token-sparsifying in trained Qwen3 — confirms the full-rank/dense-firing theme established by CF7/CF8; candidate CF entry).

**Track B pick: F3-WO-HEADBLOCK-RANK.** Score 26.2, driven entirely by the 3-orientation convergence on C1. The 5-min runtime, constructive algebraic identity, and direct connection to CF11 head-redundancy are secondary factors. This is the pipeline's loudest signal (3-orientation convergence) with the cheapest test. Runner-up: R8-RAOK-70B (named for Stage 6 kill-and-escalate; if F3 is killed at Stage 6 for CF8-extension reasons, escalate RAOK-70B directly — it has no prior-art risk and is independently load-bearing).

---

## PICK 1 — Track A: C-UFDM

### 1. Title and One-Line Description

**Run 008 / Track A — C-UFDM: W_up Firing-Rank Direction × W_Q Head-Shared Subspace Principal-Angle Overlap**

Measure whether the empirical firing subspace of W_up (top-128 right singular vectors of W_up · H_calib) and the CF11 head-shared W_Q subspace (V_Q^128) share ≥ 30% of principal angles below 30°, establishing whether MLP gating and attention querying are co-optimized on the same residual-stream coordinate frame.

### 2. Class Tags

`arch-transpose`, `subspace-alignment`, `compression-lr`, `frame-novelty`

### 3. Hypothesis

CF1 (W_up firing-rank dominance) establishes that SwiGLU post-activation magnitude is dominated by the W_up · x factor. The directions in the residual stream h that most strongly excite W_up — the empirical firing subspace V_up^128 = top-128 right singular vectors of the matrix [W_up · h_1 | W_up · h_2 | ... | W_up · h_N] — represent the residual-stream coordinate frame that MLP gating selects for.

CF11 (W_Q K=128 global GO) establishes that 16 attention heads collectively span a ~128-dim W_Q subspace of the residual stream. The top-128 right singular vectors V_Q^128 represent the coordinate frame that attention querying selects for.

The hypothesis: these two independently measured subspaces — derived from independent experiments on independent objectives — overlap substantially (principal-angle fraction α = #{angles < 30°} / 128 ≥ 0.3 at layer 14). If true, the residual stream allocates a shared coordinate frame to both MLP firing and attention querying. This motivates a joint input projection P ∈ R^{d×256} that replaces both objects, reducing per-layer GEMV count by ~1.

The hypothesis depends on CF1 (A4, load-bearing) and CF11 (A4, confirmed). It does not depend on any rank structure in W_up's weight matrix (CF8 is not violated — the firing subspace is an activation-statistic object, not a weight-rank object).

### 4. Smallest Test

**Model**: Qwen3-1.7B-Base, bf16.

**Calibration corpus**: PDAP R2 200-prompt activation dump (H_calib = residual stream activations at each layer, already collected from round 2 experiment). Load from `experiments/stage0/ladder_v2/round2_pdap/activations/`.

**Eval corpus**: N/A — this is a measurement on fixed weights + fixed calibration activations. No held-out eval needed for the primary GO/NO-GO measurement.

**Procedure**:
1. Load W_up[14] (shape 6144 × 2048, bf16 → float32 for SVD).
2. Load H_calib[14] from PDAP activation dump (shape 2048 × N_tokens, N ≈ 10K tokens from 200 prompts).
3. Compute M_up = W_up[14] @ H_calib[14] (shape 6144 × N). Run SVD; take top-128 right singular vectors → V_up^128 ∈ R^{2048 × 128}.
4. Load V_Q^128 from AQFKV run SVD data at L14 (or recompute from W_Q[14], ~1 min).
5. Compute `scipy.linalg.subspace_angles(V_up^128, V_Q^128)` → 128 principal angles θ_1 ≤ θ_2 ≤ ... ≤ θ_128.
6. Compute α = #{i : θ_i < 30°} / 128.
7. Repeat at L0 and L27 for consistency check.

**Permutation control** (Section 9): shuffle the rows of H_calib[14] (break token-sequence structure) and recompute V_up^128_perm. Compute α_perm using V_up^128_perm vs V_Q^128. GO requires α ≥ 0.3 AND α − α_perm ≥ 0.15 (real signal must substantially exceed permuted baseline).

**Random-init control** (Section 9): load a random-initialized Qwen3-1.7B-Base (torch.manual_seed(42), no trained weights). Compute V_up^128_rand on random W_up[14] and H_calib from random-init forward pass. Compute α_rand. GO requires α ≥ 0.3 AND α − α_rand ≥ 0.15.

**Multi-seed** (Section 9): recompute V_up^128 with 3 different calibration sample subsets (sample 3 × 3K-token subsets from the 200-prompt corpus at different random seeds). Report mean α ± std. GO requires mean α ≥ 0.3 AND worst-of-3 α ≥ 0.2 (above NO-GO boundary).

**Sweep parameters**: K ∈ {64, 128, 256} for both V_up^K and V_Q^K at L14. Primary GO/NO-GO at K=128; K=64 and K=256 for GRAY disambiguation.

**Output path**: `experiments/stage0/ladder_v2/round8_ufdm/`

**Script**: `scripts/ufdm_principal_angle.py` — inputs: W_up[L], H_calib[L], W_Q[L] SVD data; outputs: principal angles per (L, K), α values, permutation-control α_perm, random-init α_rand, multi-seed stats.

**Wall-clock**: < 5 min post-data-load (SVDs already computed at L14 for AQFKV; H_calib already collected for PDAP R2). Total wall-clock including controls: ~30 min.

### 5. Go Threshold

**GO**: α ≥ 0.3 at L14 (primary), AND α − α_perm ≥ 0.15 (permutation control), AND α − α_rand ≥ 0.15 (random-init control), AND mean multi-seed α ≥ 0.3 with worst-of-3 ≥ 0.2.

**Interpretation**: The firing subspace and W_Q subspace share ≥ 30% of their principal angles; the overlap is structure from trained weights, not from architecture.

### 6. No-Go Threshold

**NO-GO**: α < 0.1 at all three layers (L0, L14, L27), OR α_perm ≥ α − 0.05 (permutation destroys the signal — the overlap is a within-token artifact of H_calib distribution, not a cross-module alignment), OR α_rand ≥ α − 0.05 (the overlap is architectural, not trained).

**NO-GO structural finding**: CF1 (W_up firing dominance) and CF11 (W_Q head-sharing) operate on orthogonal residual-stream coordinate systems. The residual stream is not co-optimized across MLP and attention heads; each module extracts information from an independent subspace of h. Class-level implication: joint input projection P across MLP and attention is not motivated; the two subspaces must be treated independently. This kills the C-UFDM joint-compression cascade and any future proposal that assumes MLP firing direction and attention query direction overlap in the input space.

### 7. Ambiguous-Zone Follow-Up

**GRAY**: 0.1 ≤ α < 0.3 at L14 but ≥ 0.15 at L14 (partial overlap, controls pass).

Follow-up: sweep K ∈ {64, 128, 256, 512} for both V_up^K and V_Q^K at L14. If overlap stabilizes at α ≥ 0.3 for K ≥ 256 but not K = 128: the shared subspace is larger than 128 dimensions; the joint projection is viable at dim 512, not 256. Adjust joint P to dim=512; redo residency arithmetic. Runtime: ~20 min additional.

If overlap is layer-specific (α ≥ 0.3 at L0 or L27 but not L14): layer-dependent co-optimization; the joint projection is a per-stratum primitive, not a global one. Runtime: already collected at L0 and L27 in primary run; no additional work.

### 8. Kill Criteria (Stage 6)

- A3 (arXiv:2505.12942) covers the input-side W_up × W_Q joint-basis measurement (Stage 6 must verify explicitly; if A3's activation-aware SVD on W_VO implies input-side principal-angle measurement, escalate to R8-WVWO-FOLD runner-up immediately).
- Calibration ill-conditioning introduced if Stage 6 proposes fitting a joint P by regression (do not fit P in this experiment — measurement only; CF10 does not apply as stated, but Stage 6 should not expand scope to include a regression step).
- permutation-control failure at Stage 6 (α_perm ≥ α − 0.05): the overlap is H_calib-distribution artifact, not structural; kill and escalate to WVWO-FOLD.

### 9. Skeptic-Controls Declaration

This experiment claims "the W_up empirical firing subspace and W_Q head-shared subspace overlap in the trained model" — a "X is consistent / X transfers from CF1's measurement to CF11's measurement" claim. All three controls required.

**Permutation control**: included in Section 4. Shuffle H_calib rows to break token-sequence correlations; recompute V_up^128_perm; measure α_perm. GO threshold includes α − α_perm ≥ 0.15 gap requirement. Catches false-positive overlap from H_calib having a dominant direction that projects onto both subspaces regardless of token identity.

**Random-init control**: included in Section 4. Random-initialized Qwen3-1.7B-Base (seed 42); compute V_up^128_rand on random W_up with random H_calib. Measure α_rand. GO threshold includes α − α_rand ≥ 0.15 gap requirement. Catches structural overlap that is an architectural artifact of the Qwen3 weight-init distribution rather than a trained property.

**Multi-seed**: included in Section 4. Three calibration subsets (3K tokens each, different random draw seeds). Report mean α ± std. GO requires mean ≥ 0.3 AND worst-of-3 ≥ 0.2. Catches sensitivity to calibration corpus composition.

### 10. Runtime Estimate

- Data load (W_up[14], H_calib, V_Q^128 from AQFKV): ~5 min
- V_up^128 SVD (6144 × N matrix, N~10K, top-128 only via `scipy.sparse.linalg.svds`): ~3 min
- Principal-angle computation: < 1 min
- Permutation control: ~3 min (re-run V_up^128 on shuffled H_calib)
- Random-init control: ~8 min (random model forward pass + V_up^128)
- Multi-seed (×3 calibration subsets): ~9 min
- L0, L27 repeat: ~5 min
- **Total: ~34 min** — well within 8-hour budget.

### 11. Script Identification

`scripts/ufdm_principal_angle.py` — new script. Inputs: layer index L, K (singular vector count), PDAP activation dump path, AQFKV SVD data path or recompute flag. Outputs: JSON with principal angles, α values, permutation/random-init/multi-seed controls, and per-K sweep results. The load-bearing computation is `scipy.linalg.subspace_angles(V_up, V_Q)` on (2048 × K, 2048 × K) matrices.

### 12. Downstream Implications

**GO**: Joint input projection P ∈ R^{d × 256} replaces both W_Q's firing subspace and W_up's input query-projection. At 70B: ~2.6 GB reduction in joint-projection bandwidth. Stage 4's S8 (low-rank-by-construction affine correction) becomes relevant to the cross-scale generalization of P. Also motivates measuring whether the same overlap exists in W_K's subspace (composable with C-WKOL).

**NO-GO**: Kills the C-UFDM compression cascade. CF1 and CF11 are independent coordinate-frame findings. Any future "joint MLP-attention input basis" proposal is also killed at the class level. The pipeline's compression path is confirmed to be purely per-module (MLP quantization via RAOK-70B; attention via AQFKV-derived compression; no cross-module coupling at the input).

### 13. Provenance

- Originating orientation: C (Composition).
- Convergence cluster: none (solo advancer; no other orientation proposed the CF1×CF11 principal-angle coupling independently).
- Stage 4 gap-idea slot: N/A (C-UFDM is a Stage 2 advancer).
- Frame-novelty bonus path: A2=3 (cross-module joint input basis is not represented in any prior-round saturated frame); +2 frame-novelty bonus applied.
- Runner-up: R8-WVWO-FOLD (Stage 6 kill-and-escalate target).

---

## PICK 2 — Track B: F3-WO-HEADBLOCK-RANK

### 1. Title and One-Line Description

**Run 008 / Track B — F3-WO-HEADBLOCK-RANK: W_O Per-Head-Chunk Rank Measurement**

Measure whether per-head-chunk W_O submatrices (each 2048×128 for 16 heads) have r_99 ≤ 64, enabling joint W_V/W_O compression via the algebraic identity W_O^{(i)} W_V^{(i)} → U_i Σ_i (V_i^T W_V^{(i)}) at up to 3.6× per eligible head.

### 2. Class Tags

`compression-lr`, `arch-fold`, `algebraic-identity`, `convergence-C1`

### 3. Hypothesis

CF11 established that 16 query heads collectively span a ~128-dim subspace (head-redundancy finding). The value/output projection path (W_V → attention-weighted values → W_O) follows CF11's head-sharing premise: if the query subspace is 128-dim across 16 heads, the value subspace that each head accesses should also be lower-dimensional than the per-head dimension (128). Specifically, each head's W_O chunk (shape 2048×128 — the output rows corresponding to head i) maps head i's 128-dim value output to the 2048-dim residual stream. If this chunk has r_99 ≤ 64 (50% of head_dim), the head is effectively writing into a ≤64-dim subspace of the residual stream.

The algebraic identity is exact: W_O^{(i)} W_V^{(i)} = (U_i Σ_i V_i^T) W_V^{(i)}, where U_i (2048×k_i), Σ_i (k_i×k_i), V_i^T (k_i×128) is the SVD of W_O^{(i)}, and k_i = r_99(W_O^{(i)}). This collapses two matrices (128-dim head output projection + d-dim full output) into a product of three smaller factors. At k_i = 64: per-head storage drops from 2048×128 + 2048×128 = 524K to 2048×64 + 64×128 = 139K weights — 3.8× compression on the V-O chain for eligible heads.

The structural argument distinguishing W_O from W_down (which is CF8-full-rank): W_down receives SwiGLU's nonlinear-shaped input that drives all d_int = 6144 directions; W_O^{(i)} receives the attention-weighted value of a single head, whose effective output dimensionality is bounded by the head's attention pattern concentration. CF11's head-redundancy (16 heads → 128-dim global query subspace) is the empirical prior that the per-head output direction count is similarly constrained.

### 4. Smallest Test

**Model**: Qwen3-1.7B-Base, bf16.

**Calibration corpus**: N/A for the primary r_99 measurement (weight-only computation). WikiText-2 validation set (512 tokens) for ΔNLL evaluation if GO on r_99.

**Procedure**:
1. Load W_O[14] (shape 2048 × 2048, bf16 → float32).
2. Reshape to 16 head-chunks: W_O_heads[i] = W_O[14][i*128:(i+1)*128, :] (shape 128 × 2048) — NOTE: W_O maps from head-concatenated values (n_heads × head_dim = 2048) to residual stream (2048). Per-head chunk W_O^{(i)} is the 128 × 2048 block for head i.
   - Alternatively, consider the transpose: W_O^{(i)} as 2048 × 128 (residual stream output for head i's contribution). Check Qwen3-1.7B weight layout in config to confirm shape convention.
3. For each of 16 head-chunks: compute full SVD; report r_99 (fraction of variance at r singular vectors).
4. Identify eligible heads: those with r_99 ≤ 64. Count eligible_heads.
5. If eligible_heads ≥ 4: extend measurement to all 28 layers (spectrum only, no ΔNLL needed for kill/go decision).
6. If GO on r_99: implement joint W_V/W_O compression for eligible heads at k_i = r_99; evaluate ΔNLL on 512-token WikiText-2 held-out.

**Permutation control** (Section 9): The r_99 measurement is on weight matrices, not activations — no token-sequence structure to permute. Applicable control: shuffle the rows of W_O^{(i)} (destroy head-chunk spatial structure) and re-measure r_99_perm. GO requires r_99 ≤ 64 AND r_99 substantially lower than r_99_perm (the head-chunk structure, not random weight organization, produces the low rank). Specifically: GO requires r_99(W_O^{(i)}) / r_99_perm(W_O^{(i)}) ≤ 0.75 for eligible heads.

**Random-init control** (Section 9): Compute r_99 on the same-shaped weight matrix from a random-initialized Qwen3-1.7B-Base (seed 42). GO requires trained r_99 ≤ 64 AND r_99_rand > 100 (the low rank is trained, not architectural). For a 128×2048 random Gaussian matrix, expected r_99 ≈ 126 (near-full); any trained r_99 ≤ 64 would be strongly below random baseline.

**Multi-seed** (Section 9): For the ΔNLL evaluation (if GO on r_99): evaluate with 3 different 512-token WikiText-2 random splits (different token start offsets). Report mean ΔNLL ± std. GO requires mean ΔNLL < 0.2 nats AND worst-of-3 < 0.4 nats.

**Sweep parameters**: k_i ∈ {32, 48, 64, 96} for eligible heads. Primary GO/NO-GO at k_i = 64.

**Output path**: `experiments/stage0/ladder_v2/round8_woheadblock/`

**Script**: `scripts/f3_wo_headblock.py` — new script. Inputs: W_O[L], layer index, head_dim=128, k_sweep list. Outputs: per-head r_99, eligible head list, permutation/random-init controls, ΔNLL at each k_i if GO.

**Wall-clock**: 5 min for r_99 measurement. Add ~30 min for ΔNLL if GO on r_99. Add ~15 min for controls. **Total: ~50 min if GO on r_99; ~20 min if NO-GO.**

### 5. Go Threshold

**GO**: ≥ 4 of 16 heads at L14 have r_99(W_O^{(i)}) ≤ 64, AND r_99_trained / r_99_rand ≤ 0.75 (random-init control passes), AND r_99_trained / r_99_perm ≤ 0.75 (permutation control passes). If GO on r_99: ΔNLL < 0.2 nats at k_i = 64 for eligible heads (mean of 3 seeds).

**Interpretation**: Per-head W_O output direction count is bounded by training to ≤ 64, consistent with CF11 head-redundancy. The joint W_V/W_O compression is empirically viable.

### 6. No-Go Threshold

**NO-GO**: All 16 heads have r_99(W_O^{(i)}) ≥ 100 at L14 AND ≥ 20/28 layers.

**NO-GO structural finding**: W_O head-chunks are full-rank despite CF11's global head-redundancy. The head-redundancy finding (16 heads → 128-dim global query subspace) does NOT extend to the value/output path — each head writes into the full residual-stream dimensionality independently. Class-level kill: any per-head W_O compression without retraining. Extends the CF8/CF12 full-rank boundary to the per-head attention output projection. The attention-side compression ceiling remains W_Q + W_K global (CF11); W_V and W_O are full-rank at per-head granularity.

### 7. Ambiguous-Zone Follow-Up

**GRAY**: ≥ 4 heads at L14 with 64 < r_99 < 100 (partial concentration; below full-rank but above k_i = 64 GO).

Follow-up: sweep k_i ∈ {80, 96} for the GRAY heads. If ΔNLL < 0.2 nats at k_i = 80: practical compression at 1.9× (less than the 3.6× at k_i = 64, but still material at 70B). Also: check whether r_99 is layer-stratified (early layers lower rank, deep layers higher rank) — if so, per-layer k_i schedule is the follow-up. Runtime: ~20 min additional.

### 8. Kill Criteria (Stage 6)

- A3 (arXiv:2505.12942) explicitly tests per-head W_O rank compression and finds it NO-GO for Qwen3-class models: escalate to R8-RAOK-70B runner-up immediately.
- CF9 frame-mismatch: if Stage 6 finds the algebraic identity W_O^{(i)} W_V^{(i)} → U_i Σ_i (V_i^T W_V^{(i)}) requires a non-trivial approximation step (e.g., the attention-averaged value input changes the effective rank): kill and escalate.
- Permutation control failure: r_99_trained / r_99_perm > 0.90 (head-chunk structure not responsible for low rank; random row arrangement produces similar r_99) — kill measurement as inconclusive.

### 9. Skeptic-Controls Declaration

The experiment claims "W_O per-head-chunk rank is low in trained Qwen3-1.7B" — a "X is consistent across heads / X is a trained-model property" claim. All three controls required and included in Section 4.

**Permutation control**: Shuffle W_O^{(i)} rows; re-measure r_99_perm. GO requires r_99_trained/r_99_perm ≤ 0.75. Catches: random organization of weight rows could produce apparent low rank from numerical SVD behavior.

**Random-init control**: Random Qwen3-1.7B-Base (seed 42) W_O head-chunks. For a 128×2048 random Gaussian matrix, r_99 ≈ 126 (near-full rank). Any trained r_99 ≤ 64 substantially exceeds the random baseline (ratio ≤ 0.51). Catches: architectural constraint forcing low per-head rank regardless of training.

**Multi-seed**: For ΔNLL evaluation (GO path only): 3 WikiText-2 token subsets. Mean ΔNLL ± std. GO requires mean < 0.2 nats AND worst-of-3 < 0.4 nats. Catches: ΔNLL sensitivity to eval corpus composition.

### 10. Runtime Estimate

- Load W_O[14] (bf16 → float32): ~1 min
- 16 head-chunk SVDs (128×2048 each, full SVD): ~3 min
- Permutation control SVDs: ~3 min
- Random-init control: ~5 min (load random model, extract W_O, 16 SVDs)
- L0, L27 extension (spectrum only): ~3 min
- All-28-layer extension (spectrum only, if GO on L14): ~8 min
- ΔNLL evaluation at k sweep (if GO): ~20 min (forward pass with modified W_O for each k_i, 3 seeds)
- **Total (GO path): ~43 min. Total (NO-GO path): ~15 min.** Well within 8-hour budget.

### 11. Script Identification

`scripts/f3_wo_headblock.py` — new script. Inputs: W_O[L] path, head_dim, k_sweep, permutation and random-init flags, eval corpus path. Outputs: JSON with per-head r_99, permutation/random-init controls, eligible head list, ΔNLL at each k_i (if GO). The load-bearing computation: reshape W_O to head-chunks, `numpy.linalg.svd` per chunk (full SVD, no approximation), cumulative variance fraction to find r_99.

### 12. Downstream Implications

**GO**: Joint W_V/W_O compression at k_i = 64 for eligible heads unlocks Track B compression of the full attention-weight class (W_Q via CF11, W_K via CF11, W_O via this experiment). At 70B: W_V + W_O compression adds ~160 MB saving on top of CF11's W_Q + W_K saving. Composable with RAOK-70B's activation quantization (independent weight residency path). Also motivates measuring W_V head-chunk rank (symmetric to W_O; expected similar behavior). Unblocks Cluster C1 entirely — the four C1 advancers (WVOS-SPEC, F1-WVWO-SPECTRUM, F3, WVWO-FOLD) converged on this measurement; GO cascades through all four.

**NO-GO**: Per-head W_O is full-rank. Closes the per-head V-O compression path. Track B attention-side compression ceiling remains W_Q K=128 + W_K K=512 (global, not per-head). Kills F3's cascade AND the F1-WVWO-SPECTRUM cluster (all four C1 advancers share this load-bearing premise). Track B falls back to RAOK-70B as primary next experiment.

### 13. Provenance

- Originating orientation: F (First-Principles) — F3-WO-HEADBLOCK-RANK.
- Convergence cluster: **C1 (3-orientation: R-orientation R8-WVOS-SPEC, F-orientation F1-WVWO-SPECTRUM + F3-WO-HEADBLOCK-RANK, and implicitly R8-WVWO-FOLD's product-rank premise)** — the pipeline's strongest convergence signal this round. Convergence multiplier 1.5 applied.
- Stage 4 gap-idea slot: N/A (F3 is Stage 2/3 advancer).
- Frame-novelty bonus path: A2=2; +1 frame-novelty bonus applied (per-head-chunk W_O rank measurement absent from published attention compression literature).
- Runner-up: R8-RAOK-70B (Stage 6 kill-and-escalate target; independently load-bearing, no prior-art risk, most CF-robust proposal in the pool).
