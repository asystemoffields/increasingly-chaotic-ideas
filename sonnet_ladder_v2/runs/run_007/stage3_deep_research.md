# Stage 3 — Deep Research — Run 007

Date: 2026-05-09
Researcher: Sonnet 4.6 (claude-sonnet-4-6)
Inputs: stage1_R, stage1_F, stage1_C, stage1_A (run_007); stage2_judge (run_007); SUMMARY.md (CF1–CF12, v2-CF1, v2-CF2); KILL_LIST.md; WebSearch live.

---

## Table of Contents

1. Path 1 Track A: F7-1 WOVR, R7-R-001 ATRC, C8-WDOS, F7-5 WDCA
2. Path 1 Track B: F7-2 IDSM, C10-FRCF
3. Path 2 Convergence: Cluster C1 (W_V/W_O), Cluster C2 (Jaccard × W_Q), Cluster C3 (W_down output)
4. Path 3 Frame-Novelty: F7-4 KSATT, U4-A IOPB
5. Path 4 Elegant-Equivalence: A2-PERM
6. Wildcards: R7-R-005 QAWKC, C12-RUNE, A6-TBLK

---

## 1. F7-1 WOVR — W_O Left-Singular / W_V Right-Singular Spectral Coupling

**Stage 2 score:** 13 (highest). Path 1 Track A, convergence C1 anchor. Stage 2 pressure-test: does the U_O ≈ V_V alignment survive all 28 layers and multiple heads, or is it confined to a few layers? Check A3 for accidental W_{OV} rank measurement.

### 1.1 Mechanism decomposition

- M1: The composition W_V_h W_O_h ∈ R^{d_model × d_model} is already a natural product; the proposal claims its effective rank r_eff is well below d_head = 128 due to principal-angle alignment between U_O and V_V.
- M2: Principal-angle alignment test (U_O^T V_V) determines whether the fold is algebraic-identity (angles near 0) or approximate (moderate angles).
- M3: If M1 holds, storage replaces two d_model × d_head matrices with a single rank-r_eff factored form; the saving scales with r_eff/d_head.
- M4: Residency cascade to 70B: W_V + W_O ≈ 8 GB at 70B; 4× fold → 6 GB freed.

### 1.2 Per-claim prior-art status

**M1 (W_{OV} product rank as compression target):** **PRE-EMPTED — PARTIAL.**

A3 (arXiv:2505.12942, May 2025) explicitly targets the W_{OV} component for joint post-training low-rank compression. A3 defines W_vo,i = W_v,i · W_o,i ∈ R^{d_m × d_m} as a fused weight matrix, then applies activation-aware SVD minimizing per-head attention-output error. A3 applies this to LLaMA 3.1-70B, achieving perplexity 4.69 on WikiText-2, outperforming prior methods by 3.18 ppl. A3 supports GQA and RoPE (Qwen3-relevant).

**Differentiation:** A3 uses an activation-weighted SVD criterion (input: post-softmax attention weights 𝒑_i), not a pure weight-spectrum criterion. WOVR's mechanism M1 claims the fold is grounded in a principal-angle alignment between left/right singular bases of W_O and W_V — which is a DIFFERENT structural claim. A3 does not test whether U_O ≈ V_V holds (it never does a principal-angle test); it simply takes the product and compresses it. The WOVR structural claim (the alignment is a post-training SGD attractor, not just a compression that can be done) is **not tested in A3**. However, A3's effective compression of W_{OV} already delivers the engineering outcome WOVR targets.

**Status for M1:** ADJACENT. A3 pre-empts the compression application but not the structural alignment measurement as a scientific question.

**M2 (principal-angle alignment U_O ≈ V_V as distinct from W_OV SVD):** **NOVEL.** No paper tests whether this alignment exists in trained transformers as of 2026-05-09. A3 goes straight to the compressed form without asking whether SGD induced the alignment. elegance-class: `algebraic-identity` — if the alignment is near-zero, the fold is exact, not approximate.

**M3 (storage saving via factored form):** PRE-EMPTED by A3 for the compression application. But WOVR's framing as a structural question (not just compression) survives.

**M4 (70B residency):** ADJACENT. A3 demonstrates this at 70B scale; residency arithmetic is confirmed in principle.

**Two-paper composition flag:** A3 (arXiv:2505.12942) + AQFKV (v2 R3-A) = WOVR's compression application. Value-add beyond composition: the principal-angle test (does SGD produce this alignment?) is the structural question A3 skips. Engineering value is gone; scientific novelty survives in M2.

### 1.3 Frame-mismatch check (CF9)

M2 imports no named theorem; it is a standard SVD + cosine-angle measurement. No precondition to verify. Frame is clean.

### 1.4 Calibration ill-conditioning (CF10)

No calibration fit involved. Not applicable.

### 1.5 Residency arithmetic

Per WOVR stage1: Qwen3-1.7B, W_V + W_O = 16 MB/layer × 28 = 448 MB. At r_eff=32 (4×): 4 MB/layer → 112 MB total. Combined with CF11 W_Q (28 MB) and W_K savings: total attention at ~140 MB vs ~560 MB bf16 baseline, ≈4× attention compression. This arithmetic is consistent with A3's empirical results on similar-size models.

**What-if CF13–15 absent:** residency arithmetic uses only CF11 (confirmed). No dependence.

### 1.6 Smallest test sharpening

Script: `scripts/wovr_principal_angle_sweep.py`
- Model: Qwen3-1.7B-Base, bf16
- Layers: all 28, heads 0 and 7 (diverse)
- Operation: SVD of W_V_h (d_head × d_model, top-64 left singular vectors = U_V), SVD of W_O_h (d_model × d_head, top-64 right singular vectors = V_O); canonical angles via QR + SVD of (U_V^T V_O)
- Eval: mean canonical angle across layers and both heads; separately: SVD of W_{OV} product per layer per head to measure r_99 of the product
- Cal corpus: N/A (pure weight analysis)
- Wall-clock: ~8 min (two SVDs per head per layer, 56 SVDs of size 2048×128 on Ryzen 5 7530U at ~0.05s each)
- Output path: `experiments/stage0/ladder_v2/round7_wovr/`
- **Does NOT exceed 8h gate: ~8 min**

**Go:** mean canonical angle ≤ 30° at ≥ 18 of 28 layers AND r_99(W_OV) / d_head ≤ 0.50 at ≥ 18 layers.
**No-go:** mean canonical angle ≥ 60° at majority of layers — W_V and W_O are trained in orthogonal coordinate frames; A3's compression works but the WOVR alignment story is wrong.
**Gray:** angles 30–60°, r_99 0.5–0.8 — some alignment but not a dominant algebraic identity; use A3's activation-weighted criterion instead of pure Frobenius.

### 1.7 Risk table

| Risk | Mitigation | Falsifier |
|---|---|---|
| A3 pre-empts engineering application | Pivot WOVR to the structural science question | Run principal-angle test; if alignment absent, the structural claim dies |
| Alignment is head/layer heterogeneous | Report per-head-layer distribution; Stage 5 can target best-aligned heads | Per-head canonical angle histogram |
| W_OV full rank despite product form | A3 still works via activation-weighted SVD; WOVR framing dies | r_99(W_OV)/d_head > 0.9 at median layer |

### 1.8 Verdict

**DOWNGRADE.** The compression application of WOVR is pre-empted by A3 (arXiv:2505.12942), which already delivers activation-aware joint W_V·W_O low-rank compression at 70B scale with SOTA results. The residual scientific value — does SGD produce U_O ≈ V_V alignment as a structural attractor? — is a NOVEL measurement (M2) but does not constitute a Stage 5 compression experiment in its own right. Recommend: if the principal-angle measurement runs as a cheap 8-min smoke, the result enriches the C1 convergence cluster's understanding of why A3 works. But WOVR is not a Stage 5 selection target this round. The structural question is a side-measurement of the W_V spectrum experiment (ATRC rung 1).

**Engineering value lost to A3. Scientific value survives but is below Stage 5 selection threshold.**

---

## 2. R7-R-001 ATRC — Attention-Weight Tiered Residency Cascade

**Stage 2 score:** 12. Track B/A cascade. Stage 2 pressure-test: is W_V spectrum class closer to W_K (r_99/d ≈ 0.79) or MLP (r_99/d ≈ 1.0)? Check arXiv for W_V/W_O spectral concentration measurements.

### 2.1 Mechanism decomposition

- M1: W_V and W_O have r_99/d < 0.80 (CF11-class spectrum) — the load-bearing empirical claim.
- M2: Global SVD truncation of W_V + W_O at K matching quality budget gives 2–4× V+O compression without retraining.
- M3: Combined with CF11 W_Q (8×) and W_K (4×), total attention weight compression is 4–8× → ≈6 GB freed at 70B.
- M4: Combined with INT4 MLP, total 70B residency approaches DRAM-resident target.

### 2.2 Per-claim prior-art status

**M1 (W_V spectrum concentration):** **NOVEL as a measurement; ADJACENT in principle.** A3 (arXiv:2505.12942) jointly compresses W_V + W_O without reporting r_99/d measurements for either matrix. SVD-LLM (arXiv:2403.07378, ICLR 2025) applies truncation-aware SVD to all weight matrices including attention but does not report the CF11-style r_99 metric. ARA (arXiv:2510.19389) proposes Adaptive Rank Allocation for LLM compression and studies spectral statistics. No paper reports the CF11 metric (r_99/d = σ_{99th_percentile_of_cumulative_variance} / d) for W_V or W_O separately on Qwen3-family or equivalent post-training. The AQFKV finding (v2 R3-A, CF11) used the same metric and showed W_Q at 0.63, W_K at 0.79; W_V/W_O are untested on this ladder.

**Status M1:** NOVEL on the v2 measurement frame. The CF11-aligned r_99/d metric applied to W_V/W_O is not in the literature as of 2026-05-09. elegance-class: `conserved-quantity` (r_99/d as a measured structural invariant separating attention from MLP weight classes).

**M2 (global SVD truncation at K):** PRE-EMPTED for compression application. SVD-LLM, ARA, A3 all cover this. The contribution of ATRC is in the K-value and quality-vs-compression characterization *under the CF11 measurement frame*, not the compression method itself.

**M3 (combined attention compression 4–8×):** ADJACENT. A3 achieves comparable compression at 70B scale. ATRC's cascade framing (sequential confirmation from W_V, then W_O, then joint) adds structure not in A3.

**M4 (70B residency):** ADJACENT.

**Two-paper composition flag:** A3 (2505.12942) + AQFKV (v2) = ATRC's compression cascade. Value-add: the per-matrix CF11-style r_99/d measurement on W_V and W_O, which A3 skips in favor of activation-aware joint compression. If W_V r_99/d ≈ 0.79 (matching W_K), this directly extends CF11's structural finding and sets the v2 empirical moat for attention-weight characterization. That moat has independent value regardless of A3.

### 2.3 Frame-mismatch (CF9)

All operations are standard SVD; no imported theorem. Clean.

### 2.4 Calibration ill-conditioning (CF10)

No calibration fit. Not applicable.

### 2.5 Residency arithmetic

Full arithmetic per stage1. Confirmed consistent with A3's empirical results. Conservative branch (W_V is MLP-class, r_99/d ≈ 1.0): W_V cannot be truncated; attention compression is limited to W_Q + W_K savings (~12% total model, 70B attention weight savings ≈ 2 GB). Optimistic branch (W_V class matches W_K): additional 2–4× on W_V+W_O → ~6 GB additional freed.

### 2.6 Smallest test

Script: `scripts/atrc_wv_wo_spectrum.py`
- Model: Qwen3-1.7B-Base, bf16
- Layers: all 28; matrices: W_V (per GQA config, 8 heads × 128 × 2048), W_O (16 heads × 2048 × 128)
- Operation: SVD per layer per matrix; compute r_99; ΔNLL at K ∈ {512, 256, 128} for W_V (global, analogous to AQFKV W_K measurement)
- Calibration/eval: WikiText-2 (500 tokens eval)
- Wall-clock: ~35 min (same pattern as AQFKV W_K sweep; two more matrices)
- Output: `experiments/stage0/ladder_v2/round7_atrc/`
- **Within 8h gate**

**Go:** W_V r_99/d < 0.80 AND ΔNLL(K=256) < 0.50 nats — W_V is CF11-class; cascade proceeds to W_O and joint measurement.
**No-go:** W_V r_99/d ≥ 0.95 — MLP-class; kills the attention-tiered cascade for W_V component; extends CF8 boundary to V matrix.
**Gray:** r_99/d = 0.80–0.95, ΔNLL(K=256) = 0.50–1.50 — partial concentration; adaptive K per-layer needed.

### 2.7 Risk table

| Risk | Mitigation | Falsifier |
|---|---|---|
| W_V MLP-class (r_99/d ≈ 1.0) | Fall back to W_Q + W_K savings only; ATRC still a structural finding | W_V ΔNLL at K=256 > 1.50 nats |
| A3 already delivers W_V+W_O compression | A3 is unpublished in v2 context; CF11-metric characterization is the v2 moat | Any paper reporting CF11-style r_99/d for W_V on Qwen3 |
| GQA W_V shape mismatch | Verify Qwen3-1.7B GQA: 8 KV heads × 128 × 2048; adjust K accordingly | Config file check |

### 2.8 Verdict

**REFINE.** M1 (W_V spectrum measurement) is NOVEL on the v2 measurement frame and is the cheapest rung in the convergence C1 cluster (35 min, resolves ATRC + VJOFR + WOVR + JAQV + JVSTAB simultaneously). A3 pre-empts the engineering compression application but not the CF11-metric characterization. The 35-min experiment produces a structural finding (r_99/d for W_V and W_O) that either (a) extends the CF11 class boundary to all attention matrices, or (b) reveals W_V is MLP-class, a major architectural finding. Either outcome is load-bearing for Stage 5 selection. Priority: run this before any other advancer experiment this round.

---

## 3. C8-WDOS — W_down Output-Space × Tied-Embed Span Disjointness

**Stage 2 score:** 12. Track B. Stage 2 pressure-test: CF12 established E is nearly isotropic (r_99=1992/2048). If E^T has no dominant directions, the "E^T-invisible subspace" for W_down is essentially zero — does the disjointness claim survive isotropic E^T?

### 3.1 Mechanism decomposition

- M1: colspan(W_down, top-128) and colspan(E^T, top-128) are geometrically disjoint (mean principal angle > 60° for ≥ 70% of layers).
- M2: If M1 holds, W_down quantization error at low precision lands in an E^T-invisible subspace → near-zero logit degradation from W_down precision reduction.
- M3: At 70B, W_down ≈ 37.7 GB; INT2 W_down (if GO) is the largest single compression lever on the table.

### 3.2 Prior-art status

**M1 (E^T-invisible W_down subspace):** **NOVEL — but Stage 2's isotropic-E^T concern is fatal to the payoff.**

No paper explicitly tests the principal angle between W_down output column space and tied lm_head principal directions. However, the Stage 2 concern is critical: CF12 shows E has r_99 = 1992/2048 and **no row norms near zero**. This means E^T is NOT a concentrated matrix with a few "dominant directions" — it is nearly isotropic. The "E^T top-128 directions" are not meaningfully more important than any other 128 directions; the logit variance is spread uniformly across all 2048.

**Consequence:** The condition `dim(R_down ∩ colspan(E^T at r=128)) / 128 < 0.5` (Eq. C8) is asking whether W_down avoids 128 out of 2048 directions in an isotropic space. Even if W_down's output is confined to a 1600-dimensional subspace (the F7-5 WDCA hypothesis), the probability that 128 randomly oriented directions (from E^T isotropic) are disjoint from a 1600/2048 = 78% filling of R^{2048} is approximately (1 - 1600/2048) × 128 = 28 dimensions expected in intersection — not disjoint. The "E^T-invisible" subspace has measure proportional to 2048 - 1600 = 448 directions, and E^T's "top" 128 directions are not preferentially in this 448-dimensional space when E^T is isotropic.

**CF9 check:** No imported theorem. The principal-angle measurement itself is valid. The problem is the PAYOFF argument depends on E^T having concentrated "important" directions, which CF12 refutes.

**Status M1:** NOVEL as a measurement but the compressed-precision payoff (M2) is likely null under isotropic E^T. The Stage 2 concern is confirmed as fatal to the practical application of M1.

**M2 (low-precision W_down with near-zero logit degradation):** PRE-EMPTED by the CF12 isotropy finding. The only way to compress W_down without logit degradation is if W_down's output lands in a true null space of E^T, but E^T has no null space (r_99 = 1992). The residual 56 directions (2048 - 1992) are essentially zero-logit-weight, but 56 directions is too thin for a compression story.

**M3 (37.7 GB W_down at 70B as the lever):** The residency arithmetic is correct; the mechanism enabling it (disjointness) is broken by CF12.

**Two-paper composition flag:** There is no published cousin pair this precisely approximates, but the mechanism is equivalent to "compression-safe subspace rotation" ideas that rely on identifying logit-invisible directions — a class where CF12's isotropic finding kills all variants.

### 3.3 Calibration ill-conditioning (CF10)

No calibration fit. Not applicable.

### 3.4 Residency arithmetic

The arithmetic (706 MB at 1.7B; 37.7 GB at 70B) is correct. Under the isotropic-E^T constraint: the actual fraction of W_down's output energy that is E^T-invisible is bounded by (2048 - 1992)/2048 ≈ 2.7%. At most 2.7% of W_down's output energy is in truly E^T-invisible directions. That would allow 2.7% of W_down's weights to be quantized more aggressively — essentially zero practical gain. The residency lever vanishes.

### 3.5 Verdict

**KILL.** CF12's near-isotropic E^T (r_99 = 1992/2048) kills the payoff argument of WDOS. With E^T nearly uniformly covering all of R^{2048}, there is no meaningful "E^T-invisible subspace" larger than ~56 directions. Compressing W_down based on disjointness from those 56 directions gives at most 2.7% of W_down output energy in the safe zone — negligible. The principal-angle measurement could still run (it would reveal the actual intersection dimension), but the structural claim's payoff is null by CF12 arithmetic. **Appended to KILL_LIST below.**

Kill-list one-liner: *v2-S3-R007-001 / C8-WDOS — W_down output-space × tied-embed disjointness — CF12 establishes E^T is nearly isotropic (r_99=1992/2048), so the "E^T-invisible subspace" is ≤ 56/2048 dimensions (2.7%) — payoff null; mechanism premise refuted by CF12 arithmetic.*

---

## 4. F7-5 WDCA — W_down Column-Space Alignment with Residual Stream

**Stage 2 score:** 12. Track B. Stage 2 pressure-test: is the stacked W_down column-space analysis distinguishable from the v2-CHEAP-TEST-001 pattern? Apply skeptic-controls discipline: matched-spectrum random-orientation baseline required.

### 4.1 Mechanism decomposition

- M1: The stacked W_down matrices [W_down^1 ... W_down^28]^T have a right singular value space that covers < 1600 of 2048 residual-stream dimensions (var@1600 ≥ 0.99).
- M2: The remaining ≥ 448 residual dimensions are MLP-inert — MLP never writes to them.
- M3: MLP-inert dimensions can be stored at lower precision (INT2) in W_Q, W_K projection rows corresponding to those dimensions.
- M4: If M1 holds, a new CF emerges: "MLP output subspace is bounded in dimension," analogous to how CF11 bounded W_Q's effective rank.

### 4.2 Prior-art status

**M1 (stacked W_down column space dimension):** **NOVEL as a measurement, but v2-CHEAP-TEST-001 alarm triggered.**

v2-CHEAP-TEST-001 (killed) found that stacked W_Q var@128 = 0.2560 vs permuted-rows var@128 = 0.2559 — the cross-layer W_Q coherence was entirely a summation artifact. WDCA's experiment stacks W_down columns in R^{2048} and computes their joint PCA. The Stage 2 directive requires the same skeptic-controls discipline.

**Skeptic-controls analysis for WDCA:**
- W_down^L has shape d_ffn × d_model = 6144 × 2048. The column vectors are 2048-dimensional.
- Stacking 28 layers gives 28 × 6144 = 172K column vectors in R^{2048}.
- PCA of 172K vectors in R^{2048}: this is asking what fraction of R^{2048} is spanned by the 172K column vectors of W_down.
- Baseline question: if 172K random vectors in R^{2048} are drawn, do they span R^{2048}? Yes — trivially: with 172K >> 2048 vectors, random vectors span R^{2048} completely. var@2048 = 1.0 by the law of large numbers.
- **The key question is whether the STRUCTURE of W_down columns (not their quantity) concentrates in a subspace.** This is different from the W_Q cross-layer case: there, the claim was "W_Q subspaces are coherent across layers" — here, the claim is "W_down column vectors are confined to a subspace despite there being 172K of them."

**CF8 note:** W_down is untested on the v2 ladder (explicitly listed as open in SUMMARY.md), but CF4/CF5/CF8 established W_gate and W_up are full-rank. W_down is the output matrix and could in principle have a different structure.

**Matched-spectrum baseline:** The correct baseline for WDCA is: draw 172K random vectors with the same per-layer column-norm distribution as W_down's columns, and compute var@K. If the actual var@K significantly exceeds the random baseline (gap analogous to v2-CF2: +0.072 is insufficient), then the W_down column space IS genuinely concentrated. The Stage 1 proposal does not include this baseline — it is required here.

**Status M1:** PARTIAL. The measurement is worth running, but ONLY with the matched-spectrum random baseline. Without the baseline, the result is uninterpretable (same failure mode as v2-CHEAP-TEST-001).

**M3 (INT2 for MLP-inert W_Q rows):** NOVEL if M1 holds and the baseline passes. No paper assigns mixed-precision to W_Q rows based on MLP output subspace membership.

**Two-paper composition flag:** There is no direct cousin pair for WDCA's specific claim. Closest: FLAT-LLM (activation-subspace projection for weight compression) uses a similar "project into activation subspace" logic but does not measure W_down's column space jointly across layers. Value-add: the joint MLP output subspace characterization as a structural finding (new CF).

### 4.3 CF9 check

PCA of W_down column vectors: no imported theorem. Standard SVD. No precondition to verify. Clean.

### 4.4 Residency arithmetic

If M1 holds (r_col = 1600): 448 MLP-inert residual dimensions. W_Q rows corresponding to those 448 dims at INT2 vs BF16 saves: 448/2048 × W_Q per layer × 28 = 0.219 × 8 MB × 28 = ~49 MB. Modest. At 70B: 0.219 × (global W_Q 14 GB) = 3.1 GB. Non-trivial.

**Conservative branch (CF8-by-analogy; W_down is also full-rank):** var@2048 = 1.0, r_col = 2048, no MLP-inert subspace. Payoff = 0. This is the expected outcome given CF4/CF5/CF8 pattern.

### 4.5 Smallest test (with required baseline)

Script: `scripts/wdca_column_space_sweep.py` (requires adding the baseline)
- Model: Qwen3-1.7B-Base, bf16
- Operation: (a) load all 28 W_down; (b) flatten column vectors to 28×6144×2048; (c) compute sample covariance of the 172K column vectors in R^{2048}; (d) eigendecompose; record var@K for K ∈ {1024, 1400, 1600, 1800, 2048}; (e) MATCHED BASELINE: generate 172K random vectors with same per-column-norm distribution; compute var@K for baseline; (f) report real − baseline gap
- Wall-clock: ~15 min (covariance of 172K × 2048 matrix + PCA) + ~5 min for baseline = ~20 min total
- Output: `experiments/stage0/ladder_v2/round7_wdca/`
- **Within 8h gate**

**Go:** real var@1600 - baseline var@1600 ≥ 0.05 (analogous to CF2 gap criterion) — genuine structural concentration.
**No-go:** gap < 0.01 — W_down columns are uniformly distributed; MLP output subspace fills R^{2048}; no MLP-inert directions; extends CF8 to W_down.
**Gray:** gap 0.01–0.05 — mild concentration; follow-up with layer-by-layer var@K to identify which layers contribute the structure.

### 4.6 Verdict

**REFINE — requires skeptic-controls baseline added to experimental plan.** The experiment is valid and fast (20 min), but the Stage 1 plan lacks the matched-spectrum random baseline that the v2-CHEAP-TEST-001 history makes mandatory. Adding the baseline converts WDCA from a false-positive risk to a properly calibrated measurement. Run after ATRC (W_V/W_O spectrum) since both together characterize the full residual-stream output geometry of the attention + MLP system.

---

## 5. F7-2 IDSM — Intrinsic Dimension of the Attention Score Manifold

**Stage 2 score:** 12. Track B. First-principles KV compression bound.

### 5.1 Mechanism decomposition

- M1: The attention score matrix A ∈ R^{T×T} lives on a low-dimensional manifold (participation ratio PR/T² ≤ 0.20 at layer 14).
- M2: If M1 holds, a structured KV compression scheme can retain ≤ 20% of KV dimensions without fidelity loss.
- M3: This provides a first-principles bound distinct from engineering heuristics (H2O, ScissorHands) — it grounds KV compression in the intrinsic geometry of attention outputs.

### 5.2 Prior-art status

**M1 (attention score covariance participation ratio):** **ADJACENT — ClusterAttn is the critical cousin.**

ClusterAttn (ACL 2025, "KV Cache Compression under Intrinsic Attention Clustering") shows that attention heads consistently focus on specific clusters of the prompt during decoding — a pattern detectable from an observation window at the prompt's end. ClusterAttn uses a density-based clustering algorithm on the attention distribution. This is adjacent to IDSM in that it confirms attention scores have "intrinsic structure," but it uses clustering (not PCA/participation ratio) and focuses on which *tokens* matter (not the intrinsic dimensionality of the score *matrix*).

Expected Attention (arXiv:2510.00636, Oct 2025) estimates future query attention distributions using distributional properties of LLM activations. This is adjacent to IDSM but focuses on query prediction, not score-matrix intrinsic dimension.

Neither paper computes the participation ratio of the attention-score covariance matrix — that specific measurement is NOVEL as of 2026-05-09.

**Status M1:** NOVEL. No paper measures PR of the attention-score distribution in the PCA covariance sense as of 2026-05-09.

**M2 (KV compression bound from PR):** NOVEL — this is the first-principles derivation that IDSM's value-add over engineering heuristics.

**elegance-class for M1:** `conserved-quantity` — PR as a measured invariant characterizing information content of the attention distribution.

**Two-paper composition flag:** ClusterAttn (ACL 2025) + AQFKV (v2) ≈ IDSM's framing. Value-add: PR is a single number quantifying attention score dimensionality, whereas ClusterAttn requires fitting clusters and Expected Attention requires distributional modeling. IDSM's contribution is the cleaner scalar bound.

### 5.3 CF9 check

M1 imports PCA / participation ratio — standard covariance eigendecomposition. Precondition: the covariance of attention score matrices exists and is well-conditioned given 200 calibration sequences. With T=128 and 200 sequences, the sample matrix has 200 rows and T² = 16384 columns — underdetermined (200 << 16384). This is a significant concern.

**CF9 flag:** The participation ratio of a 200 × 16384 sample covariance is unreliable when n_samples << n_dimensions. With 200 samples and 16384-dim observations, the sample covariance is rank-200 by construction — the participation ratio will appear artificially low (PR/T² ≈ 200/16384 ≈ 0.012) simply because there are only 200 samples. This is NOT the same as the underlying distribution having low PR; it is a sampling artifact.

**Mitigation:** Use a MUCH smaller context T (e.g., T=32), giving T²=1024-dim observations vs 200 samples — marginally underdetermined (200/1024 ≈ 0.20). Or increase n_samples to 2000+. The Stage 1 smallest-test plan (T=128, n=200) will almost certainly show artificially low PR regardless of the true distribution. This must be fixed before the experiment is meaningful.

**Revised smallest test:** T=32 (PR/T²=1024 dims), n_sequences=500 (giving 500 × 1024, marginally overdetermined). Alternatively, T=64, n=1000.

### 5.4 Calibration ill-conditioning (CF10)

No calibration fit. But the sample-size issue (section 5.3) is analogous to ill-conditioning — same symptom (apparent structure from insufficient data).

### 5.5 Residency arithmetic

Speculative until M1 is confirmed. KV at T=4096, 7.28 GiB system: KV cache ≈ 56 MB at T=4096 (from stage1). This is not the binding constraint at 1.7B — weights dominate. At 70B with T=32K: KV ≈ 20 GB — binding. IDSM's payoff is long-context inference; at the current 4K inference window it has no residency lever.

### 5.6 Verdict

**REFINE — requires CF9 mitigation (increase samples or reduce T) before run.** The measurement is novel and clean if the sampling issue is resolved. The 8-min estimate from Stage 1 becomes 20 min at T=64, n=1000. The mechanism is valid; the experimental design requires a fix to be interpretable. Lower priority than ATRC/WDCA in the current round (KV is not the binding constraint at 1.7B), but advances for Stage 4/5 consideration for long-context use cases.

---

## 6. C10-FRCF — Firing-Rank × Embed Span

**Stage 2 score:** 11. Track B. Stage 2 pressure-test: verify memory feasibility of c_E(j) computation; check whether SpQR already covers this.

### 6.1 Mechanism decomposition

- M1: Spearman ρ(F_j, c_E(j)) > 0.15 at ≥ 20/28 layers — firing rank correlates with logit sensitivity at the neuron level.
- M2: High-F_j × c_E(j) neurons can be stored at BF16; others at INT4 — a per-neuron mixed-precision assignment grounded in the dual signal.
- M3: At 70B, the 5% BF16 sidecar + 95% INT4 reduces W_up+W_down from 75.5 GB to ~21.7 GB.

### 6.2 Prior-art status

**M1 (W_up firing rank × lm_head logit sensitivity joint signal):** **NOVEL.**

SpQR (arXiv:2306.03078, ICLR 2024) identifies outlier weights using the Hessian (activation covariance-based sensitivity). SpQR assigns precision per weight group, not per neuron, and uses Hessian as the sole signal. FRCF's dual signal (W_up firing magnitude × W_down column sensitivity through E) is a cross-matrix coupling not available to Hessian-only methods. SpQR does not compose CF1 (W_up firing rank) with CF12 (lm_head full-rank structure) to identify a joint "fast path" neuron class.

GPTVQ (arXiv:2402.15319) and AQLM use clustering-based codebook methods without neuron-level dual signals.

**Status M1:** NOVEL. No published method uses both the upstream (W_up firing magnitude) and downstream (E · W_down column norm) signals simultaneously for neuron-level precision assignment. elegance-class: `subspace-alignment` — the coupling exploits the alignment between the MLP firing path and the lm_head read path.

**M2 (per-neuron mixed-precision from dual signal):** PARTIAL — per-neuron precision exists (SpQR, unstructured mode), but the specific dual-signal derivation is NOVEL.

**Two-paper composition flag:** CF1 (v2 measurement) + CF12 (v2 measurement) ≈ FRCF's structural coupling. There is no published cousin pair — FRCF is uniquely grounded in two v2-internal findings. Value-add is the cross-matrix coupling itself.

### 6.3 CF9 check

No imported theorem. F_j is a standard forward-pass statistic; c_E(j) is a matrix-vector norm. No preconditions to verify. Clean.

### 6.4 Calibration ill-conditioning (CF10)

FRCF performs no calibration fit (F_j and c_E(j) are statistics, not fitted parameters). CF10 not applicable. **Safe.**

### 6.5 Memory feasibility verification

Stage 2 concern: E (590 MB bf16) + W_down (25 MB per layer) must fit in 7.28 GiB.

Verification: E is 151936 × 2048 × 2B = 589 MB. W_down[L] is 6144 × 2048 × 2B = 25.2 MB. Per-layer computation: E loaded once (~600 MB), W_down[L] loaded per layer (~25 MB). Total peak: 600 + 25 + Python overhead ≈ 1.5 GB. Well within 7.28 GiB. **Memory feasibility confirmed.**

### 6.6 Residency arithmetic

Stage 1 arithmetic confirmed: 5% BF16 + 95% INT4 → W_up + W_down per layer = 14.5 MB vs 25.2 MB BF16 (−42%). At 70B: 21.7 GB vs 75.5 GB. This is a large lever IF the correlation holds and IF 5% of neurons by the joint score are genuinely the right threshold.

**Quality caveat:** The INT4 assignment for the 95% low-signal neurons assumes they do not have hidden outlier behavior. The SpQR Hessian is a better individual-weight sensitivity metric than neuron-level averaging; the dual signal may over-promote neurons that are W_up-dominant but W_down-small (or vice versa), missing Hessian-detected outliers. The CF10-style conditioning check here is: F_j is computed from 200 forward passes (200 × 6144 = 1.2M samples for d_ffn=6144 per layer) — adequately sampled.

### 6.7 Smallest test

Script: `scripts/frcf_firing_embed_correlation.py`
- Model: Qwen3-1.7B-Base, bf16
- Cal corpus: PDAP calibration (200 prompts, existing)
- Operations: (a) compute F_j = mean |silu(W_gate[:,j]·x) × W_up[:,j]·x| per neuron per layer; (b) compute c_E(j) = ‖E · W_down[:,j]‖_2 for all j per layer; (c) Spearman ρ per layer; (d) permutation control: shuffle F_j across j, recompute ρ
- Wall-clock: ~45 min (28 layers × per-layer forward pass stat + column dot computation)
- Output: `experiments/stage0/ladder_v2/round7_frcf/`
- **Within 8h gate**

**Go:** ρ > 0.15 at ≥ 20/28 layers AND permutation-control ρ < 0.05.
**No-go:** ρ ≤ 0.05 at majority of layers — CF1 and CF12 are decoupled; per-neuron Hessian methods (SpQR) remain the best approach.
**Gray:** ρ = 0.05–0.15, inconsistent across layers — signal exists but weak; increase calibration corpus to 2000 tokens and recheck.

### 6.8 Verdict

**REFINE.** FRCF has a genuinely novel dual-signal mechanism, clean CF9, safe CF10 (no calibration fit), confirmed memory feasibility, and a 45-min decisive experiment. The primary risk is that the correlation is weak (ρ < 0.15) because the firing path (W_up → W_down → E) is long enough that the signal decorrelates. But this is an empirical question with a fast answer. Second-highest priority for Stage 5 after ATRC.

---

## 7. Path 2 Convergence Analysis

### Cluster C1 — W_V and W_O spectral structure

**Representatives:** F7-1 WOVR (DOWNGRADED — A3 pre-emption), R7-R-001 ATRC (REFINE), R7-R-003 VJOFR (convergence rep), C9-JAQV (convergence rep), C11-JVSTAB (convergence rep).

**Strongest representative:** ATRC (M1 is NOVEL on v2 frame; 35-min settler; A3 pre-empts engineering application but not CF11-metric characterization).

**C1 cheapest settler:** W_V and W_O SVD sweep (35 min). Resolves ATRC, VJOFR, JAQV, JVSTAB simultaneously. WOVR's engineering value is pre-empted by A3, but the principal-angle measurement is a 3-min add-on that enriches the result.

**Prior-art context for C1 as a cluster:** A3 (2505.12942) is the key paper — it has already compressed W_V + W_O at 70B scale with SOTA results. This does NOT kill the v2 CF11-metric characterization (ATRC), because A3 uses an activation-weighted criterion without reporting r_99/d per matrix. The v2 moat is the CF11-style structural measurement, not the compression application itself.

### Cluster C2 — Per-layer Jaccard × W_Q structural behavior

**Representatives:** C7-JDWQ (convergence anchor), R7-R-004 ASOR (Track A), C9-JAQV (Track B joint budget).

**Stage 3 evaluation:** JDWQ proposes a Spearman correlation between the v2-CF1 per-layer Jaccard profile (J_ℓ) and the per-layer W_Q r_99/d. This requires running the per-layer W_Q SVD (all 28 layers). Two prior-art concerns:

1. v2-CF2 killed cross-layer W_Q stacking. JDWQ is NOT cross-layer stacking — it measures per-layer W_Q independently, then correlates with J_ℓ. This is safe under v2-CF2.
2. The correlation hypothesis (higher J = more W_Q rank) is a falsifiable prediction. No published paper uses per-token outlier Jaccard as a predictor of W_Q spectral rank.

**Prior-art status for JDWQ M1:** NOVEL. The v2-CF1 Jaccard is the v2 moat; no paper has correlated it with W_Q spectral structure.

**Smallest settler for C2:** Per-layer W_Q SVD (all 28 layers, top-256 singular values), ~30 min. Spearman ρ with 28 data points. If ρ is insignificant, the budget-allocation idea (JAQV) is purely data-driven rather than principle-grounded, but JAQV survives as an adaptive compression idea regardless.

**C7-JDWQ verdict: REFINE.** Cheap, novel measurement, decisive go/no-go.

### Cluster C3 — W_down output subspace as a compression object

**Representatives:** C8-WDOS (KILLED — CF12 isotropic E^T), F7-5 WDCA (REFINE with baseline).

The WDOS kill does not affect WDCA: WDCA asks about r_col of W_down's output space (does MLP write to all of R^{2048}?), not about disjointness from E^T's top directions. These are orthogonal measurements. F7-5 WDCA proceeds with the skeptic-controls baseline.

---

## 8. F7-4 KSATT — Kronecker Structure in Attention Weight Matrices

**Stage 2 score:** 12. Track B. Frame-novelty slot 1 (A2=3). Stage 2 directive: search "Kronecker product test transformer weights" and "post-training structured matrix decomposition" 2024–2026.

### 8.1 Mechanism decomposition

- M1: The attention weight matrices W_Q, W_K, W_V have implicit Kronecker product structure in trained weights (≥ 50% Frobenius fraction captured by rank-1 Kronecker at p=32, q=64 partition).
- M2: If M1 holds, compression via r Kronecker factors (r × (p² + q²) ≪ d_model²) is substantially better than SVD at the same parameter count.
- M3: The structure is an SGD-implicit attractor, not a training-time constraint — this is a structural characterization question, not a compression method proposal.

### 8.2 Prior-art status

**M1 (post-training implicit Kronecker structure in trained LLM weights):** **NOVEL — no published paper tests this as a structural characterization question as of 2026-05-09.**

Search results found:
- "Identifying Kronecker product factorizations" (arXiv:2510.25292, Oct 2025): focuses on detecting Kronecker structure in binary/sparse matrices via sparsity patterns. Does NOT test trained LLM attention weights.
- Higher-Order Transformers with Kronecker-Structured Attention (arXiv:2412.02919, Dec 2024): proposes Kronecker-structured attention as a training-time architectural constraint for multiway (not standard language model) attention. Different application domain.
- SoKA (arXiv:2506.15251): combines Kronecker-product tensor factorization with SVD for PEFT (fine-tuning). Not post-training structural characterization.
- Monarch matrices / Kaleidoscope: training-time structured matrices. Not post-training characterization.

**No paper tests whether trained standard LLM attention weights (W_Q, W_K) have emergent Kronecker structure as a consequence of SGD dynamics. This is NOVEL as of 2026-05-09.**

**M2 (Kronecker compression vs SVD at same parameter count):** ADJACENT. The Monarch/Kaleidoscope family uses Kronecker-structured training to achieve better compression than SVD; the post-training version of this claim would need to be demonstrated empirically. The compression arithmetic is sound.

**M3 (structural characterization not compression method):** NOVEL — the frame of asking "what structure did SGD converge to?" rather than "what structure can we impose?" is the distinctive contribution.

**elegance-class M1:** `algebraic-identity` — if Kronecker structure exists in trained weights, it would be an exact algebraic factorization (not an approximation) that reduces storage and compute simultaneously.

**Two-paper composition flag:** Monarch (arXiv:2204.00595) + AQFKV (v2) ≈ KSATT's compression application. Value-add: KSATT tests the question retroactively (does the structure already exist?) rather than imposing it at training time. No cousin pair covers this retroactive structural question.

### 8.3 CF9 check

The Kronecker reshuffling test (reshape W_Q as p×q × p×q block matrix M, take SVD) is standard linear algebra with no preconditions. The test is: "does the reshuffled matrix M have a leading singular value that captures ≥ 50% of Frobenius?" — purely checkable. No imported theorem.

**One concern:** the partition sensitivity. If only 3 partitions are tested (p=32,q=64; p=64,q=32; p=128,q=16) and all three fail, the claim "W_Q has no Kronecker structure" is valid only for those factorization shapes. A different partition (p=4, q=512 or p=16, q=128) might show structure. The Stage 1 proposal acknowledges this; 10 min allows testing 3–5 partitions. This is a minor limitation, not a CF9 failure.

### 8.4 Residency arithmetic

If Kronecker rank r=4 (p=32,q=64): 4 × (32²+64²) = 4 × 5120 = 20K params vs W_Q = 2048² = 4M → 200× compression. Even r=64: 328K vs 4M → 12×. This would be the most aggressive attention compression achievable, far exceeding CF11's 8× at K=128. At 70B: W_Q 14 GB × 12× = 1.2 GB. Transformative if it holds. The probability it holds at r=4 is low (trained weights rarely exhibit exact Kronecker structure), but at r=64 the probability is moderate.

### 8.5 Smallest test

Script: `scripts/ksatt_kronecker_test.py`
- Model: Qwen3-1.7B-Base, bf16
- Layers/heads: Layer 14, heads 0 and 7; test W_Q (concatenated 16×128×2048 → reshaped)
- Partitions: (p=32,q=64), (p=64,q=32), (p=128,q=16)
- Operation: reshape W_Q head block to p×q × p×q, SVD of the 4D reshuffling M, report ||rank-1 Kronecker||_F / ||W_Q_head||_F
- Wall-clock: ~5 min (three SVDs of 2048×2048 matrices)
- Output: `experiments/stage0/ladder_v2/round7_ksatt/`
- **Within 8h gate**

**Go:** ≥ 50% Frobenius at any tested partition → Kronecker structure exists; extend to all 28 layers and measure ΔNLL.
**No-go:** < 20% at all tested partitions → no Kronecker structure; confirms SGD does NOT converge to Kronecker organization of attention weights.
**Gray:** 20–50% — partial structure; test more partition shapes and larger r.

### 8.6 Verdict

**REFINE.** KSATT has a genuinely novel frame (retroactive Kronecker characterization), clean CF9 (no imported preconditions to worry about), no calibration fit (CF10 safe), a 5-min decisive experiment, and no prior-art pre-emption. The frame-novelty (A2=3) is confirmed: no published paper asks "did SGD converge to Kronecker structure?" for standard trained LLM attention weights. The risk (no structure found) produces a negative structural finding that closes this class permanently. Either outcome is load-bearing. **Strong Stage 5 candidate.**

---

## 9. U4-A IOPB — Windows IOCP Token Budget as Attention-Head Bandwidth Regulator

**Stage 2 score:** 11. Track A. Frame-novelty slot 2 (A2=3). Stage 2 pressure-test: does Windows IOCP `NumberOfConcurrentThreads` actually throttle I/O completions vs just limiting thread wakeups?

### 9.1 Mechanism decomposition

- M1: IOCP `NumberOfConcurrentThreads` can be configured to throttle the rate at which NVMe read completions trigger computation threads — acting as a bandwidth governor for concurrent weight-streaming operations.
- M2: By mapping attention heads to separate IOCP completion queues with per-head concurrency limits, head-level bandwidth allocation becomes controllable.
- M3: This enables per-head attention-weight streaming at regulated bandwidth, analogous to QoS scheduling for weight access.

### 9.2 Prior-art status

**M1 (IOCP concurrency as bandwidth governor):** **FRAME-NOVEL — no prior LLM paper, but the mechanism semantics are ambiguous.**

From the search results (Wikipedia, Microsoft Learn docs): IOCP's `NumberOfConcurrentThreads` limits the number of *threads* that can SIMULTANEOUSLY process completions — it is a thread concurrency limiter, not a throughput limiter on the I/O requests themselves. The OS NVMe driver submits I/O requests independently of IOCP thread scheduling. Throttling `NumberOfConcurrentThreads` dequeues completions more slowly (threads are bottlenecked waiting for IOCP) but does NOT reduce the rate at which the NVMe SSD services I/O requests — those run at hardware speed regardless of IOCP concurrency.

**CF9 check — imported object:** The proposal imports IOCP's `NumberOfConcurrentThreads` as a "bandwidth governor." The precondition required: "throttling completion threads limits I/O throughput." Does this hold?

From Windows internals knowledge: IOCP dequeues completions only when a thread calls `GetQueuedCompletionStatus()`. If there are N active completion threads and `NumberOfConcurrentThreads = K < N`, only K threads process completions simultaneously. However, the NVMe I/O requests remain in flight at hardware speed — the SSD processes them regardless. What throttling `NumberOfConcurrentThreads` does is:
- Limit how quickly COMPLETED I/Os are consumed by the processing pipeline (completion processing rate = min(K threads, I/O throughput / completion size)).
- Does NOT limit how fast I/Os are ISSUED (that depends on `ReadFileEx` calls, not IOCP completion handling).

Therefore, the "bandwidth governor" claim is partially incorrect: IOCP throttling gates *computation* triggered by completions (i.e., how fast data is processed after being read), not the I/O bandwidth to NVMe itself. The mechanism can still be useful as a **computation regulator** (limit how many attention head weight buffers are simultaneously ready for computation), but it is NOT a bandwidth governor in the NVMe throughput sense.

**CF9 verdict:** The imported mechanism (IOCP `NumberOfConcurrentThreads` as bandwidth governor) is ADJACENT but not exact. The precondition (throttling completions reduces NVMe throughput) does NOT hold — the NVMe still reads at full speed. What remains after stripping the incorrect precondition: IOCP as a computation-scheduling primitive that serializes weight processing across heads, which is ADJACENT to standard thread-pool management (not novel). However, using IOCP completion ordering to serialize *which head's weights* are processed when (by assigning different heads to different IOCP ports with sequential ordering) is a valid and novel scheduling primitive for attention parallelism management.

**Revised mechanism (surviving CF9 strip):** Use IOCP ports as head-level computation serialization locks — head k's computation begins only when head k's read completion is dequeued (no thread-parallel computation for different heads). This is structurally similar to a semaphore-gated computation pipeline, not a bandwidth governor, but it could reduce peak DRAM pressure from multiple head computations overlapping.

**Status M1 (revised):** ADJACENT. The frame is novel (IOCP as LLM head scheduler), but the "bandwidth governor" claim is CF9-broken. The surviving primitive is a computation serializer, not a bandwidth limiter.

### 9.3 Revised smallest test

What the experiment should actually test: with ik_llama.cpp, does serializing attention head computation via IOCP (one completion port per head, sequential dequeuing) reduce peak RAM utilization vs parallel head computation, and does this improve tok/s on the Ryzen 5 7530U (where DRAM bandwidth, not NVMe, is the attention-compute bottleneck)?

Wall-clock: 3-hour experiment (implementing IOCP-based serialization in a C test harness for Qwen3 attention layer). **Still within 8h gate.**

**Go:** ≥ 5% tok/s improvement from reduced DRAM-bandwidth contention.
**No-go:** No improvement — DRAM contention between heads is negligible at 1.7B scale (only 28 MB attention weights, bandwidth cost is small).

### 9.4 Verdict

**REGENERATE — CF9 partial failure.** The "bandwidth governor" claim is broken by IOCP's actual semantics (completion throttling ≠ NVMe throughput throttling). The surviving primitive (IOCP as computation serializer for per-head scheduling) is real but ADJACENT to standard semaphore-based computation scheduling — it is the same mechanism. The frame novelty is reduced from A2=3 to approximately A2=2 after the strip. The surviving question (does head-serialized computation reduce DRAM contention?) is worth asking but is a much narrower engineering question than the original frame-novel proposal.

**Surviving primitive:** IOCP completion-port ordering as a per-head computation serialization primitive. Pass to Stage 4 as a narrow engineering experiment (3-hour harness, go = 5% speedup).

---

## 10. A2-PERM — Permutation-Only Gauge for Codebook Alignment

**Stage 2 score:** 10+. Track B. Elegant-equivalence, Path 4. gauge-exploitation sub-class.

### 10.1 Mechanism decomposition

- M1: The trained transformer has a gauge freedom under S_{16} × S_{128} (head permutation × within-head dimension permutation) that commutes exactly with RMSNorm.
- M2: A calibration-derived permutation sorts heads by utilization (W_Q row norm as proxy), placing the four least-used heads at positions 12–15.
- M3: A structured mixed-precision codebook assigns 4-bit to heads 0–11 and 2-bit to heads 12–15, grounded in the permutation alignment.
- M4: This is the only gauge freedom in RMSNorm-normalized transformers that is both exact and composable with any downstream quantization scheme.

### 10.2 Prior-art status

**M1 (head permutation gauge commuting with RMSNorm):** **NOVEL as a compression primitive; ADJACENT in the optimization literature.**

DuQuant (from search results) uses zigzag permutation + rotation for mixed-precision quantization — it combines permutations with rotations (which break RMSNorm), making it CF9-violating on the rotation component. DuQuant's permutation component alone (reordering channels to even out outliers) is adjacent to PERM.

CVPR 2024 "Permutation Equivariance of Transformers and Its Applications" addresses permutation symmetries of transformers generally but does not apply this to mixed-precision codebook alignment as a compression primitive.

SpAtten prunes heads by importance score without the permutation-alignment story.

**The specific mechanism of using the exact S_{16} × S_{128} permutation gauge to ALIGN a mixed-precision codebook structure (not just to prune) has no named entry as of 2026-05-09.**

**Status M1:** NOVEL. The elegance-class `gauge-exploitation` tag is confirmed — this is using a genuine group action (the permutation group S_{16}) that the loss is invariant to, in order to force a structured precision assignment. The "invariance" is what makes the fold lossless.

**M2 (W_Q row norm as head utilization proxy):** ADJACENT. Many methods rank heads by importance (SpAtten, A3), but none use the specific permutation-alignment framing. The Stage 2 note that "calibration-based head importance ranking is not already addressed by SpAtten, A3, or head pruning" is confirmed by search — those methods prune, not permute-align.

**elegance-class confirmed:** `gauge-exploitation` — S_{16} × S_{128} permutation symmetry group acts on the attention block and is invariant under the loss; exploiting this to force structured codebook alignment is exact and lossless.

**Two-paper composition flag:** DuQuant (permutation component) + CF11 (attention compressibility) ≈ PERM. Value-add: PERM's permutation is EXACT (not approximate rotation), uses a group invariance argument (not a heuristic), and the codebook alignment is grounded in the gauge structure rather than sensitivity.

### 10.3 CF9 check

M1 imports no theorem; the permutation-commutes-with-RMSNorm claim is verifiable: RMSNorm applies a per-channel scale γ and normalization. A permutation π applied consistently to the attention block (reordering rows of W_Q, W_K, W_V and columns of W_O) maps γ → π(γ) — since RMSNorm's per-channel scale γ is permuted consistently, the function is preserved exactly. This is not an approximation; it is a symmetry.

**Precondition check:** Does RMSNorm's per-channel γ commute with attention head reordering? Yes — if the permutation is applied consistently to both the head dimension and the corresponding W_Q rows, W_K rows, W_V rows, W_O columns, AND the next layer's input dimensions. In practice, reordering heads requires reordering 128-wide blocks consistently through W_Q, W_K, W_V, W_O, and the input of the next RMSNorm. This is a bookkeeping operation, not a transformation that changes function. **Precondition holds.**

### 10.4 Residency arithmetic

Stage 1 arithmetic: at Qwen3-1.7B (1.7B × 4bpw = 850 MB), attention weights = 112 MB at 4bpw. PERM mixed-precision (4-bit heads 0–11, 2-bit heads 12–15 = 25% at 2bpw): 98 MB, saving 14 MB. Modest at 1.7B.

**Value-add beyond residency:** The permutation alignment makes the mixed-precision assignment **auditable and reproducible** — any quantization method (GPTQ, AWQ) can be applied on top of the permuted representation and the low-precision assignment is structural rather than heuristic. At 70B, if head-norm ratio > 2× holds for all layers, 25% of attention weights at 2bpw saves: 30% of total params × 25% = 7.5% of total, or ~10.5 GB × 0.25 × (4bpw→2bpw relative saving) ≈ 2.6 GB. Useful but not transformative alone.

### 10.5 Smallest test

Script: `scripts/perm_head_norm_ratio.py`
- Model: Qwen3-1.7B-Base, bf16
- Operation: load W_Q for all 28 layers; per layer, compute per-head mean row L2 norm (16 heads × 128 rows each); sort heads by norm; compute ratio = mean(top-4 norms) / mean(bottom-4 norms)
- Wall-clock: < 10 min (pure weight load + norm computation)
- Output: `experiments/stage0/ladder_v2/round7_perm/`
- **Within 8h gate**

**Go:** mean ratio across layers > 2.0 — structurally diverse head utilization; permutation alignment creates meaningful codebook separation.
**No-go:** ratio ≤ 1.2 across most layers — heads have uniform utilization; permutation-based mixed-precision has no structural motivation; fall back to calibration-entropy-based sorting.
**Gray:** ratio 1.2–2.0 — use attention entropy as sorting key instead of W_Q norm.

### 10.6 Verdict

**REFINE.** Confirmed `gauge-exploitation` elegant-equivalence class. M1 is NOVEL (no published paper uses the exact S_{16} × S_{128} gauge for structured codebook alignment). CF9 precondition verified. The 10-min experiment is decisive. The mechanism is elegant and lossless. Residency gain is modest in isolation but composes cleanly with CF11 W_Q truncation. **Strong Stage 5 candidate for Path 4.**

---

## 11. R7-R-005 QAWKC — Quantization-Aware W_Q/W_K Cascade for 70B [FREE SWING]

**Stage 2 score:** 11. Wildcard. CF-tether suspended.

### 11.1 Mechanism decomposition

- M1: The K/n_heads ≈ 8 head-redundancy invariant from CF11 (1.7B: K=128, n_heads=16) scales to larger models (4B: K=256?, 70B: K=512?).
- M2: If the scaling invariant holds across model sizes, 70B W_Q compression is empirically motivated without directly running 70B experiments.

### 11.2 Prior-art status

**M1 (K/n_heads invariant across model scales):** **NOVEL.** No paper systematically measures the effective subspace dimensionality per head across model sizes as a scale-invariant structural property.

Search results for W_Q scaling law: no paper found that tests this specific invariant. The closest: ARA (arXiv:2510.19389) does adaptive rank allocation but does not test scale invariance of the per-head redundancy ratio. SVD-LLM V2 (NAACL 2025) ranks layers but not per-head across scales.

**Status M1:** NOVEL. Wildcard structural-floor met.

### 11.3 CF9 check (wildcards still bound)

No imported theorem. The SVD measurement on Qwen3-4B is standard. Preconditions: Qwen3-4B fits in 7.28 GiB for blockwise SVD. Qwen3-4B has 4B × 2 bytes = 8 GB (just over limit) — but blockwise streaming SVD processes layer-by-layer, never holding the full model. Memory per layer for W_Q (4B): d_model=2560, W_Q = 2560×2560×2B = 13 MB per layer. Plus model architecture (activations, etc.): ~2 GB for model overhead. Feasible with blockwise streaming. CF9 clean.

### 11.4 Smallest test

Script: `scripts/qawkc_scale_invariant_test.py`
- Model: Qwen3-4B-Instruct (or 4B-Base if available) OR fallback to Qwen3-1.7B vs comparison with published scaling results
- Operation: W_Q SVD on all layers; ΔNLL at K = 32×8 = 256 (the invariant prediction); compare to K=128 result (expected ~2× worse or better based on scaling)
- Wall-clock: ~50 min (SVD on 36 layers × 2560×2560 matrices; ΔNLL eval)
- Output: `experiments/stage0/ladder_v2/round7_qawkc/`
- **Within 8h gate**

**Go:** ΔNLL(4B, K=256) < 1.0 nats AND K/n_heads ≈ 8 within 50% → scaling invariant holds at two data points.
**No-go:** ΔNLL(4B, K=256) > 2.0 → subspace does not scale linearly with n_heads; 70B attention compression extrapolation is not justified.

### 11.5 Verdict

**REFINE — wildcard, CF-tether-suspended, structural floor met.** Novel scaling-invariant hypothesis with a 50-min decisive test. Advances as wildcard to Stage 5 candidate pool.

---

## 12. C12-RUNE — Residual Near-Unity × Embed Logit Factorization [FREE SWING]

**Stage 2 score:** 11. Wildcard. CF-tether suspended.

### 12.1 Mechanism decomposition

- M1: The residual increment Δh = h_L − h_0 has r_99 ≤ 128 (PCA of 200 calibration token Δh vectors achieves 99% variance at k=128).
- M2: If M1 holds, the lm_head computation can be factored into a prompt-amortized base term + cheap rank-k correction.
- M3: The lm_head bandwidth per token drops from 590 MB (E^T read) to 18.9 MB (E^T_k at k=64).

### 12.2 Prior-art status

**M1 (Δh intrinsic dimension):** **NOVEL** — the Stage 1 explicitly notes this is different from CF13-15 (unconfirmed residual intrinsic dimension). Δh = h_L − h_0 is the *increment* from input to output, not the total residual h_L. The measurement is: does the RESIDUAL UPDATE (not the residual state) live in a low-dimensional subspace?

No paper measures the intrinsic dimension of {h_L(t) − h_0(t)} across tokens. This is a new measurement.

**Risk from Stage 1 (confirmed):** CF2 establishes cos(h_L, h_{L+1}) ≈ 0.99 for ADJACENT layers, not cos(h_L, h_0). The actual cos(h_L, h_0) across 28 layers of transformation may be much smaller — potentially 0.6–0.7, meaning Δh ≈ 0.4–0.5 × h_0 in magnitude. If Δh is large, the factored lm_head's "base + correction" decomposition is still valid mathematically, but the correction term dominates and the savings disappear (the cheap E^T_k term carries most of the logit variance). The r_99(Δh) measurement determines whether M1 holds regardless of the magnitude issue.

**Two-paper composition flag:** CF2 (residual additivity) + CF12 (lm_head full-rank) ≈ RUNE's motivation. Value-add: the Δh intrinsic dimension measurement as a NEW CF. No published method exploits the specific factorization.

### 12.3 CF9 check (wildcards still bound)

No imported theorem. PCA of Δh is standard; preconditions: n_samples >> d for sample covariance. With 200 tokens and d=2048, n/d = 0.098 — severely underdetermined (same issue as IDSM section 5.3). The measured r_99 will be ≤ 200 by construction, not ≤ 128 from true structure.

**CF9 flag:** Sample-size insufficiency. With 200 samples in R^{2048}, the sample covariance is rank-200 — any claimed r_99 ≤ 128 is impossible to distinguish from sampling artifact. Need n_samples >> 2048 for reliable r_99 estimation.

**Mitigation:** Increase calibration to 2000+ tokens (20 min instead of 5 min). At n=2000, sample covariance is rank-2000 > 2048 — overdetermined. With n=5000 (50 min for 200 prompts × 25 tokens each, or 50 min for 50 prompts × 100 tokens), the covariance is better conditioned. This is within the 8h gate.

### 12.4 Verdict

**REFINE — requires sample-size mitigation before run is interpretable.** CF-tether suspended (wildcard). M1 is NOVEL and potentially valuable (lm_head bandwidth reduction by 31×). The CF9 flag (sample-size insufficiency at n=200) must be addressed: increase calibration to n=2000 tokens. With this fix, the experiment is 30 min and decisive. The lm_head bandwidth saving (590 MB → 18.9 MB per token) is a meaningful speedup on the Ryzen (11.5 GB/s DRAM) if M1 holds.

---

## 13. A6-TBLK — Table-Driven Block Decode [FREE SWING]

**Stage 2 score:** 11. Wildcard. CF-tether suspended.

### 13.1 Mechanism decomposition

- M1: WikiText-2 (or narrow domain) 4-gram coverage from a 1M-entry table > 20%.
- M2: If M1 holds, a B-tree NVMe table replaces neural inference for covered N-grams at 175K tok/s effective throughput.

### 13.2 Prior-art status

**M1 (offline N-gram table replacing neural inference):** **ADJACENT — Infini-Gram is the critical cousin.**

Infini-Gram (arXiv:2401.17377, COLM 2024) is a large-scale N-gram engine using a suffix array for fast N-gram counting on 1.4 trillion tokens (7 bytes/token overhead). Infini-Gram supports N-gram lookup at low latency but focuses on counts/probability estimation, not as a neural-inference replacement.

However, the Stage 2 note correctly distinguished: TBLK inverts the framing (neural model → table offline; table → online inference), which N-Gram Trie Speculative Decoding does NOT do (it uses N-grams to SPEED UP neural inference, not replace it). This inversion is the structural distinction.

**No published paper uses offline neural inference to populate an NVMe B-tree that replaces neural inference at test time (with fallback). NOVEL as a system framing.**

**M1 status:** NOVEL on the framing. The 4-gram coverage measurement is technically trivial to run.

### 13.3 CF9 / structural floor

No imported theorem. Stack primitives: BLAKE3, B-tree, NVMe binary search. All confirmed primitives. Structural floor met.

### 13.4 Smallest test

Script: `scripts/tblk_4gram_coverage.py`
- Operation: (a) build Python dict of 4-grams from WikiText-103 first 100M tokens; (b) for each 4-gram in WikiText-2 10K-token test set, check dict membership; (c) report coverage
- Wall-clock: < 30 min
- Output: `experiments/stage0/ladder_v2/round7_tblk/`
- **Within 8h gate**

**Go:** 4-gram coverage > 20% on WikiText-2 test.
**No-go:** Coverage < 5% → natural language 4-gram distribution is too sparse for this domain.

### 13.5 Verdict

**REFINE — wildcard, CF-tether-suspended, structural floor met.** The cheapest experiment in this round (< 30 min), no prior-art pre-emption for the specific system framing, and the coverage measurement is decisive. Note: even if WikiText-2 coverage is > 20%, the deployment scope is narrow domains (code, templates). That is fine for the wildcard framing; the claim is not general LLM inference replacement.

---

## 14. Summary Table and Verdicts

| ID | Score | Verdict | Reason class |
|---|---|---|---|
| F7-1 WOVR | 13 | **DOWNGRADE** | A3 (2505.12942) pre-empts engineering application; structural science (M2 principal-angle) is below Stage 5 threshold; cheap 8-min add-on to ATRC |
| R7-R-001 ATRC | 12 | **REFINE** | M1 NOVEL on v2 CF11 frame; 35-min C1 cluster settler; A3 does not report r_99/d metrics |
| C8-WDOS | 12 | **KILL** | CF12 isotropic E^T kills payoff; ≤2.7% of W_down output energy in E^T-invisible directions |
| F7-5 WDCA | 12 | **REFINE** | M1 NOVEL; requires matched-spectrum baseline (v2-CHEAP-TEST-001 discipline); 20-min experiment |
| F7-2 IDSM | 12 | **REFINE** | M1 NOVEL; CF9 sample-size fix required (T=32 or n=1000); long-context use case |
| C10-FRCF | 11 | **REFINE** | M1 NOVEL (dual signal CF1×CF12); CF10 safe; memory feasibility confirmed; 45-min experiment |
| F7-4 KSATT | 12 | **REFINE** | M1 NOVEL (no post-training Kronecker test published); CF9 clean; 5-min experiment; strong Stage 5 candidate |
| U4-A IOPB | 11 | **REGENERATE** | CF9 partial failure: IOCP throttling ≠ NVMe throughput throttling; surviving primitive is computation serialization (ADJACENT to standard semaphore scheduling); pass to Stage 4 as narrow engineering experiment |
| A2-PERM | 10+ | **REFINE** | gauge-exploitation `elegance-class` confirmed; CF9 precondition verified; 10-min experiment; Path 4 candidate |
| R7-R-005 QAWKC | 11 | **REFINE (wildcard)** | NOVEL scaling invariant; CF-tether suspended; structural floor met; 50-min |
| C12-RUNE | 11 | **REFINE (wildcard)** | M1 NOVEL; CF9 sample-size fix required (n=2000+); 30-min with fix; lm_head 31× bandwidth reduction |
| A6-TBLK | 11 | **REFINE (wildcard)** | NOVEL framing (offline neural→table); CF-tether suspended; 30-min trivial coverage check |

**Convergence pre-program order for Stage 5:**
1. C1 settler: ATRC W_V/W_O SVD sweep (35 min) — resolves ATRC, VJOFR, WOVR add-on, JAQV, JVSTAB
2. C2 settler: per-layer W_Q SVD all 28 layers (30 min) — resolves JDWQ correlation
3. C3 settler: WDCA stacked column PCA + baseline (20 min) — resolves WDCA
4. KSATT (5 min) — frame-novelty, rapid signal
5. PERM (10 min) — gauge-exploitation check
6. FRCF (45 min) — dual-signal correlation
7. RUNE (30 min with sample fix) — Δh intrinsic dimension
8. TBLK (30 min) — 4-gram coverage
9. QAWKC (50 min) — scaling invariant on Qwen3-4B

---

## 15. Appendix: KILL_LIST Additions

```
v2-S3-R007-001 / C8-WDOS — W_down output-space × tied-embed disjointness (compression-subspace) — CF12 establishes E^T is nearly isotropic (r_99=1992/2048), so the E^T-invisible subspace is ≤56/2048 dimensions (2.7%) — payoff null; mechanism premise refuted by CF12 arithmetic. 2026-05-09.

v2-S3-R007-002 / F7-1 WOVR — W_O left-singular / W_V right-singular spectral coupling (compression-lr, arch-fold) — DOWNGRADED (not killed): engineering compression application pre-empted by A3 (arXiv:2505.12942, May 2025) which jointly compresses W_V·W_O at 70B scale with SOTA results; structural science (principal-angle alignment as SGD attractor) survives as side-measurement but not Stage 5 selection target. 2026-05-09.

v2-S3-R007-003 / U4-A IOPB — IOCP token budget as attention-head bandwidth regulator (systems-os, arch-transpose) — CF9 partial failure: IOCP NumberOfConcurrentThreads throttles completion processing threads, not NVMe I/O throughput; surviving mechanism is computation serializer adjacent to standard semaphore scheduling; pass to Stage 4 as narrow engineering experiment. 2026-05-09.
```

---

## Sources consulted

- [A3: Analytical Low-Rank Approximation Framework for Attention (arXiv:2505.12942)](https://arxiv.org/abs/2505.12942)
- [QKV Projections Require a Fraction of Their Memory (arXiv:2506.02939)](https://arxiv.org/abs/2506.02939)
- [SVD-LLM: ICLR 2025 (arXiv:2403.07378)](https://arxiv.org/pdf/2403.07378)
- [ARA: Adaptive Rank Allocation (arXiv:2510.19389)](https://arxiv.org/html/2510.19389v1)
- [Identifying Kronecker product factorizations (arXiv:2510.25292)](https://arxiv.org/abs/2510.25292)
- [Higher-Order Transformers with Kronecker-Structured Attention (arXiv:2412.02919)](https://arxiv.org/abs/2412.02919)
- [SpQR: Sparse-Quantized LLM Compression, ICLR 2024 (arXiv:2306.03078)](https://arxiv.org/abs/2306.03078v1)
- [ClusterAttn: KV Cache Compression under Intrinsic Attention Clustering (ACL 2025)](https://aclanthology.org/2025.acl-long.703/)
- [Expected Attention: KV Cache Compression (arXiv:2510.00636)](https://arxiv.org/abs/2510.00636)
- [I/O Completion Ports documentation (Microsoft Learn)](https://learn.microsoft.com/en-us/windows/win32/fileio/i-o-completion-ports)
- [Infini-Gram N-gram Engine (arXiv:2401.17377)](https://arxiv.org/pdf/2401.17377)
- [Mixed-Precision Quantization for LLMs (arXiv:2510.16805)](https://arxiv.org/html/2510.16805v1)
