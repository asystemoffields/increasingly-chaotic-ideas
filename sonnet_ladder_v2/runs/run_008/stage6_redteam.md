# Stage 6 Red Team — Run 008

**Date**: 2026-05-09
**Tracks**: A — C-UFDM (score 21.0) | B — F3-WO-HEADBLOCK-RANK (score 26.2)
**Runner-ups**: A — R8-WVWO-FOLD | B — R8-RAOK-70B
**Known kill**: v2-S3-R008-001 F4-SVAL-CONSERVED (SINQ, arXiv:2509.22944)

---

## Track A — C-UFDM

### 1. Frame-Mismatch Re-Check (CF9)

C-UFDM imports one non-trivial primitive: `scipy.linalg.subspace_angles` for principal-angle measurement between two real subspaces. Precondition: both input matrices must span well-defined subspaces of the same ambient dimension d=2048. The inputs are V_up^128 (top-128 right singular vectors of W_up[14] @ H_calib) and V_Q^128 (top-128 right singular vectors of W_Q[14]).

**Independent verification**: both objects are real matrices with orthonormal columns in R^{2048 × 128}. `subspace_angles` computes SVD of V_Q^T V_up; precondition (orthonormality, consistent ambient dimension) is trivially satisfied for the described construction. No precondition failure.

One structural subtlety Stage 3 noted but did not fully pin: V_up^128 depends on the token distribution of H_calib. If H_calib has a dominant low-rank structure (e.g., nearly all tokens produce similar residual-stream activations), then W_up @ H_calib is also near-low-rank, and V_up^128 picks up H_calib's dominant directions regardless of W_up's structure. The permutation control (shuffle H_calib rows) does NOT catch this: shuffling rows of H_calib does not change its column space. **The correct permutation control is to shuffle the columns of H_calib** (break the token-identity while preserving each token's activation vector) — but that also does not change the column-span. The actual control needed is to replace H_calib with a random Gaussian matrix of the same shape and verify V_up^128_rand is uniformly spread. This is the random-init control (separate W_up is random), not the permutation control.

**Flag**: Section 9 states "shuffle the rows of H_calib" as permutation control. Shuffling rows of a matrix X of shape (N_tokens × d) changes the token ordering — but V_up^128 is the top-128 right singular vectors of W_up @ X^T, i.e., a (6144 × N) matrix. Shuffling the N columns of H_calib^T (= rows of H_calib in the plan's shape convention: H_calib is d × N_tokens) DOES change the product only if W_up has per-token dependence — but W_up is a fixed weight matrix. The SVD of W_up @ H_calib depends on the column space of H_calib, not the column ordering. **Row-shuffle of H_calib is a no-op for the column space of the product.** The permutation control as written does not probe what it claims to probe. This is a structural defect: α_perm will equal α exactly (column space unchanged), the gap α − α_perm ≈ 0, and the GO criterion α − α_perm ≥ 0.15 will always fail.

**Severity**: Medium-high. The permutation control as written is broken; it will always produce α_perm ≈ α, causing every result to fail the GO criterion even when the signal is real. Stage 5 must amend the permutation control before running. Correct permutation: replace H_calib with a random Gaussian matrix N(0,1) of the same shape (d × N_tokens), recompute V_up^128_rand_H, measure α. This tests whether the overlap is driven by H_calib's specific geometry or by W_up's intrinsic structure.

No CF9 kill. Math is otherwise correct.

### 2. Calibration Ill-Conditioning (CF10)

No regression fit in the measurement stage. Both V_up^128 and V_Q^128 are computed via deterministic SVD on fixed matrices. CF10 not applicable to this experiment as designed. Stage 5's kill criterion correctly flags CF10 as a risk only if Stage 4 proposes a regression-fit joint P — not applicable here.

### 3. Skeptic-Controls Check

Hypothesis shape: "the W_up empirical firing subspace and W_Q head-shared subspace overlap in the trained model" — a "trained structure exists" claim. Controls required.

**Permutation control**: present but broken as analyzed in Section 1. The row-shuffle of H_calib does not permute the column space of H_calib; α_perm ≈ α trivially. Amendment required: replace permutation control with random-H_calib control (sample fresh Gaussian matrix, compute V_up^128 on it, measure α).

**Random-init control**: present and correctly specified. Random Qwen3-1.7B-Base (seed 42) gives an independent check on whether the overlap is architectural vs trained. GO requires α − α_rand ≥ 0.15. Structurally sound.

**Multi-seed**: present. Three calibration subsets of 3K tokens each. Mean α ± std, worst-of-3 ≥ 0.2. Sound.

**Amendment A1**: Replace permutation control "shuffle rows of H_calib" with "sample fresh Gaussian H_calib of same shape (d × N_tokens), compute V_up^128_gaussH, report α_gaussH." GO requires α − α_gaussH ≥ 0.10 (testing that W_up's weight structure, not just H_calib's input geometry, drives the subspace selection). Relax gap to 0.10 vs 0.15 because some overlap with Gaussian H_calib is expected (W_up itself has structure) — the question is whether trained H_calib geometry adds additional overlap.

Status: ACCEPT-WITH-AMENDMENTS (permutation control fix required).

### 4. Hidden Prior-Art Search (Third Pass)

Searches executed 2026-05-09:

1. "W_up firing subspace W_Q principal angle overlap joint input projection LLM 2026" — no paper found measuring CF1×CF11 cross-module principal angle.

2. **LatentLLM (arXiv:2505.18413, May 2025)**: "Attention-Aware Joint Tensor Compression." This paper jointly decomposes pairs of (Q, V) projections, (V, O) projections, and (U, D) MLP projections. The MLP compression addresses the (W_up, W_down) pair — not W_up×W_Q cross-module. LatentLLM's compression is output-side product folding (same object as A3/WVWO-FOLD), not input-side joint basis from measured subspace overlap. **C-UFDM's claim is input-side**: do the residual-stream directions that excite W_up overlap with the directions that W_Q projects onto? LatentLLM does not measure this geometric question. Not pre-empted.

3. **A3 (arXiv:2505.12942)**: compresses W_Q via activation-aware SVD on the OV and QK components separately. No cross-MLP measurement. Not pre-empted.

4. **Spectral Lifecycle (arXiv:2604.22778)**: documents Q/K–V functional asymmetry during training. Discusses W_Q/W_K subspace dynamics. Does not measure overlap between W_up firing subspace and W_Q input subspace. Not pre-empted.

**C-UFDM novelty holds.** The specific geometric measurement — principal-angle overlap between the activation-statistics firing subspace of W_up and the weight-derived head-shared subspace of W_Q — is not in any found paper.

### 5. Biased-Framing Audit

- **GO threshold (α ≥ 0.3)**: 30% of 128 principal angles below 30°. This is a non-trivial threshold — for two random 128-dim subspaces of R^{2048}, the expected fraction of principal angles below 30° is approximately sin²(30°) = 25% under a uniform-subspace distribution, which means the GO threshold is only modestly above chance. The threshold should be raised to α ≥ 0.35 to provide clearer separation from a random-subspace baseline.

- **NO-GO threshold (α < 0.1)**: The GRAY zone (0.1 ≤ α < 0.3) is 20 percentage points wide with no automatic escalation path — Stage 5 defers to a K-sweep follow-up. This is acceptable given the follow-up is specified and bounded (~20 min).

- **NO-GO structural claim**: "CF1 and CF11 operate on orthogonal coordinate systems" is a legitimate class-level kill. Load-bearing and specific.

- **Amendment A2**: Raise primary GO threshold to α ≥ 0.35 and add an explicit random-subspace baseline to the report (analytically derived for 128-dim subspaces in R^{2048}; expected α_random ≈ 0.22–0.26). GO condition should require α − α_random_theoretical ≥ 0.10.

### 6. Runtime/Scope Sanity

Section 10 estimates ~34 min total. The random-init control requires a full forward pass through a random-initialized Qwen3-1.7B-Base — this is ~8 min as stated, reasonable. Data load path from PDAP R2 activation dump is a dependency; if the dump is not at the stated path, the experiment blocks. Stage 5 should verify the path `experiments/stage0/ladder_v2/round2_pdap/activations/` exists before scheduling.

Eval corpus: N/A (measurement only). Cal/eval disjointness: not applicable — no regression fit.

Section 4 plan is correctly scoped to a measurement experiment, not a regression. Scope is appropriate for the 5-min (primary) / 34-min (with controls) runtime claim.

### 6c. Elegant-Equivalence Check

elegance-class: `subspace-alignment`. The claim is not a commutation-identity claim; it is a geometric measurement claim. No commuting-operators precondition applies. 6c does not trigger for C-UFDM.

### 7. CF13–CF15 Verification Check

No CF13–CF15 numbers cited in the experiment plan. CF9/CF10 quarantine compliant.

### Track A Verdict

**ACCEPT-WITH-AMENDMENTS**

Amendments:
1. Fix permutation control: replace "shuffle rows of H_calib" with "sample fresh Gaussian H_calib of same shape; compute V_up^128_gaussH; report α_gaussH; GO requires α − α_gaussH ≥ 0.10."
2. Raise GO threshold from α ≥ 0.30 to α ≥ 0.35; add analytical random-subspace baseline α_random ≈ 0.22–0.26 to report.
3. Verify path `experiments/stage0/ladder_v2/round2_pdap/activations/` exists before run.

---

## Track B — F3-WO-HEADBLOCK-RANK

### 1. Frame-Mismatch Re-Check (CF9)

Load-bearing import: algebraic identity W_O^{(i)} W_V^{(i)} = (U_i Σ_i V_i^T) W_V^{(i)}, where U_i Σ_i V_i^T is the SVD of W_O^{(i)}.

**Independent verification**: the identity is exact linear algebra. No precondition beyond W_O^{(i)} being a real matrix. SVD always exists. The identity holds trivially: W_O^{(i)} = U_i Σ_i V_i^T by SVD definition; therefore W_O^{(i)} W_V^{(i)} = U_i Σ_i (V_i^T W_V^{(i)}) exactly. This is not an approximation — the storage saving comes from truncating Σ_i to rank k_i at the ΔNLL measurement stage, which introduces approximation only then. CF9 is clean.

**Shape-convention flag**: Section 4 step 2 describes W_O_heads[i] = W_O[14][i*128:(i+1)*128, :] (shape 128 × 2048). But the experiment plan's hypothesis section describes W_O^{(i)} as "the 128 × 2048 block for head i" mapping head i's 128-dim output to the 2048-dim residual stream. In the standard Qwen3 implementation, W_O (o_proj) has shape (d_model, n_heads × head_dim) = (2048, 2048). Head i's contribution is W_O[:, i*128:(i+1)*128] — shape (2048, 128), NOT (128, 2048). The plan's indexing `W_O[14][i*128:(i+1)*128, :]` slices rows, not columns. **This is likely a shape-convention error**: if W_O is (2048, 2048), then `W_O[i*128:(i+1)*128, :]` gives (128, 2048) slicing the output dimension, not the head dimension. The correct per-head chunk is the (2048, 128) column-slice.

**Severity**: Medium. The r_99 of a (128 × 2048) matrix and its transpose (2048 × 128) are identical (rank is invariant to transpose). The r_99 measurement result is unaffected by this confusion. However, the algebraic identity step requires the correct shape orientation: W_O^{(i)} @ V_i = U_i Σ_i (V_i^T @ V_i) = U_i Σ_i only if V_i^T @ W_V^{(i)} is the correct product. Script `f3_wo_headblock.py` must use the column-slice convention (2048 × 128 per head) to ensure the downstream compression identity is correctly dimensioned.

**Amendment A3**: Add explicit shape assertion to the script: `assert W_O.shape == (2048, 2048)` and `W_O_head_i = W_O[:, i*128:(i+1)*128]` (shape 2048×128). Document the convention explicitly so the compression identity step uses the right orientation.

No CF9 kill.

### 2. Calibration Ill-Conditioning (CF10)

Primary measurement is weight-only SVD — no calibration fit. CF10 not applicable to the r_99 measurement.

ΔNLL evaluation (if GO): this is a forward-pass evaluation, not a regression fit. CF10 does not apply.

### 3. Skeptic-Controls Check

Hypothesis shape: "W_O per-head-chunk rank is low in trained Qwen3-1.7B" — a trained-model structural property claim. Controls required.

**Permutation control**: Section 4 proposes "shuffle the rows of W_O^{(i)}, re-measure r_99_perm." For a weight matrix (not activation-statistic), shuffling rows of W_O^{(i)} changes the matrix's structure. The SVD singular values are invariant to row permutations only if the rows are orthonormal (which they are not for arbitrary trained weights). In general, row permutation of W changes its SVD. **However**: the expected behavior of a random row-permutation of a 128×2048 matrix is near-preservation of the singular value spectrum (random permutations preserve the Frobenius norm and approximately preserve the singular value distribution for large matrices). The control as stated is marginally useful but not strong. A more diagnostic control: randomly sign-flip columns of W_O^{(i)} (preserves Frobenius norm, maximally disrupts row-structure). Either way, the control's primary purpose is to confirm r_99(trained) << r_99(random), which is better captured by the random-init control.

**Random-init control**: correctly specified. For a 128×2048 (or 2048×128) random Gaussian matrix, r_99 ≈ min(128, 2048) = 128 (near-full rank). Any trained r_99 ≤ 64 would be ratio ≤ 0.51 — strongly below random. This is the primary structural control.

**Multi-seed**: correctly specified for the ΔNLL path. N/A for the r_99 measurement (deterministic SVD on fixed weights — multi-seed adds no information). The plan correctly marks multi-seed as applicable to ΔNLL only.

Controls are sound. Minor: permutation control could be strengthened (see above) but is not broken in the same way as C-UFDM's.

### 4. Hidden Prior-Art Search (Third Pass)

1. **A3 (arXiv:2505.12942)**: Stage 3 flagged A3 as a kill criterion. Fresh verification: A3 performs joint SVD on the OV component by "concatenating the scaled error matrices along the column dimension" — this is a cross-head stacked OV decomposition, not per-head W_O chunk rank measurement. A3's OV compression minimizes activation-reconstruction error on the fused OV product; it does NOT separately measure r_99(W_O^{(i)}) per head-chunk, nor does it test the per-head-chunk rank threshold criterion. **A3 does not pre-empt the per-head-chunk rank measurement.** The Stage 5 kill criterion (if A3 "explicitly tests per-head W_O rank compression and finds it NO-GO") is not triggered by A3's actual content.

2. **LatentLLM (arXiv:2505.18413)**: compresses (V, O) pairs jointly using high-order tensor decomposition. Does not measure per-head-chunk r_99 as a structural finding. Not pre-empted.

3. **Spectral Lifecycle (arXiv:2604.22778)**: documents V/O "uniform compression" during training — favorable prior. Does not measure post-training per-head W_O chunk rank. Not pre-empted.

**No prior-art kill found for F3-WO-HEADBLOCK-RANK.** The per-head-chunk r_99 measurement on Qwen3-1.7B post-training is not in any found paper.

### 5. Biased-Framing Audit

- **GO threshold (≥ 4 of 16 heads at r_99 ≤ 64)**: 25% of heads eligible. For a 128×2048 matrix, r_99 ≤ 64 requires ≥ 50% of variance in the top-64 singular values — a moderately concentrated spectrum. The threshold is not gerrymandered: r_99 ≤ 64 = 50% of head_dim is not a trivially easy bar for trained matrices.

- **NO-GO threshold (all 16 heads r_99 ≥ 100 at L14 AND ≥ 20/28 layers)**: The AND condition across both layer 14 and 20/28 layers is reasonable. The NO-GO structural claim (extends CF8 to per-head W_O) is specific and load-bearing — it explicitly kills the entire C1 cluster cascade, which is stated in Section 12.

- **GRAY zone**: 64 < r_99 < 100 for ≥ 4 heads. The follow-up (sweep k_i ∈ {80, 96}) is bounded and specific. Not rhetorical.

No biased-framing flags.

### 6. Runtime/Scope Sanity

Section 10: ~43 min (GO path), ~15 min (NO-GO path). This is consistent with the work described: 16 head-chunk SVDs on 128×2048 matrices (~3 min) + random-init control + optional ΔNLL forward passes. Runtime is well within the 5-min label applied by Stage 5 (the "5-min" label in the Stage 5 scoring refers to the r_99 decision gate before ΔNLL, which is correct — Section 4 step 5 gates ΔNLL on GO at L14).

Cal/eval disjointness: WikiText-2 validation set is disjoint from the weight matrices (no calibration fit). Sound.

### 6c. Elegant-Equivalence Check

elegance-class: `algebraic-identity`. The identity W_O^{(i)} W_V^{(i)} = U_i Σ_i (V_i^T W_V^{(i)}) is exact. The precondition for the compression application is that rank(W_O^{(i)}) << head_dim = 128, which is exactly what the measurement tests. No hidden commutation requirement. The elegance argument does not depend on any failing precondition. 6c is clean.

### 7. CF13–CF15 Verification Check

No CF13–CF15 numbers cited. Compliant.

### Track B Verdict

**ACCEPT-WITH-AMENDMENTS**

Amendments:
1. Fix shape convention in script: use `W_O_head_i = W_O[:, i*128:(i+1)*128]` (2048×128 column-slice), add explicit shape assertion. Add convention note to output JSON.
2. In the script, verify W_O.shape[0] == d_model and W_O.shape[1] == n_heads × head_dim before slicing — catch config-mismatch early.
3. Strengthen permutation control: use column sign-flip (multiply each column of W_O^{(i)} by a random ±1) rather than row shuffle; this disrupts structure while preserving norm. Report ratio r_99_trained / r_99_signflip.

---

## Overall Verdict Summary

| Track | Pick | Verdict | Key Amendments |
|---|---|---|---|
| A | C-UFDM | ACCEPT-WITH-AMENDMENTS | Fix broken permutation control; raise GO threshold to α≥0.35; verify H_calib path |
| B | F3-WO-HEADBLOCK-RANK | ACCEPT-WITH-AMENDMENTS | Fix W_O shape-slice convention (columns not rows); strengthen permutation control |

No runner-up escalation triggered. Neither pick has fatal math errors, pre-empting prior art, CF10 failures, or missing skeptic controls (after amendments). Both experiments proceed with the stated amendments incorporated before execution.
