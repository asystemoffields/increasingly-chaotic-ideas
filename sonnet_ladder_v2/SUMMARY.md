# Pipeline Status Summary (v2)

This file is the load-bearing empirical-substrate input read by every Stage 1 ideator. Every mechanism a Stage 1 proposal cites must connect to a number on this page or to a primitive on the actual stack.

## Provenance and confirmation status

The v1 ladder produced findings CF1-CF12. Those are USER-CONFIRMED outputs of the Sonnet ladder own experiments and are the load-bearing inputs for v2 Stage 1.

The in-flight Opus pipeline produced PRE-1 / PRE-2 measurements that became CF13-CF15 in `sonnet_ladder/SUMMARY.md`. As of the v2 cutover those measurements are NOT user-confirmed and are NOT carried forward as feed-forward inputs to v2 Stage 1. If v2 ladder rounds re-derive any of them via a v2 experiment, those re-derivations enter v2 SUMMARY as CF13''..CF15'' under a v2 numbering, with explicit citation of the v2 round that produced them.

This means: v2 Stage 1 ideators reason from CF1-CF12 only. They may not cite "the residual stream has a 2-channel highway" or "Windows mmap page-fault is 137us / 1.9us follow-on" or "stratified residual intrinsic dim is ~223" as load-bearing inputs. Those numbers exist; they are not yet confirmed by the v2 substrate.

If a v2 Stage 1 proposal NEEDS one of those numbers to close, the proposal must include a v2 measurement as its first cascade rung that re-derives the number. This is a feature: it forces v2 empirical map to be self-consistent rather than inheriting unconfirmed inputs from the parallel pipeline.

When in doubt: load-bearing for v2 = CF1..CF12. Everything else is reference-only.

---

## v2-confirmed findings (added during 50-run pass)

- **v2-CF1 — CF3 generalizes to all 28 layers (Qwen3-1.7B-Base, 2026-05-09)**: K-dependent outlier-channel-set Jaccard at K=1% measured across all 28 layers (CF3 only sampled 7). Mean = 0.39, range [0.22, 0.531]. **27 of 28 layers below 0.50**; only L27 marginal at 0.531. Mid-depth L7-L15 cluster around 0.41-0.44 (least dynamic). Permutation control: 0.0053 (validates measurement). Random-init: mean 0.21, trained gap +0.18 (learned structure real). Source: `sonnet_ladder_v2/cheap_tests/run_001/rraok_result.md`. Implication: per-token outlier handling is needed across the full layer stack at operational K, not just sampled deep layers.

- **v2-CF2 — Cluster C1 cross-layer W_Q subspace sharing CLASS-KILLED (Qwen3-0.6B-Base, 2026-05-09)**: stacked SVD of W_Q across all 28 layers gives var@128 = 0.2560. Matched-spectrum random-orientation baseline gives var@128 = 0.1839 (gap = +0.072, requires > 0.20 — FAIL). Permutation control gives var@128 = 0.2559 (gap = +0.0001 — FAIL). The "cross-layer subspace coherence" was entirely the within-layer CF11 concentration trivially summed in the stack. **W_Q subspaces rotate independently across all 28 transformer layers without retraining.** ΔNLL at K=128 = +2.678 nats. Per-layer K=128 (CF11) is the post-training W_Q ceiling. Source: `sonnet_ladder_v2/cheap_tests/run_001/fcqsgc_result.md`. Implication: any cross-layer W_Q stacking / shared-basis / joint-diagonalization variant inherits this kill. KILL_LIST entry v2-CHEAP-TEST-001.

- **v2-CF3 — CF8 closes to all 3 MLP matrices (Qwen3-1.7B-Base, 2026-05-09)**: pure-weight SVD of W_down across all 28 layers. r_99 median = 1889/2048 = 0.922 of max rank. var@1024 mean = 0.785. **0/28 layers reach 99% variance at K ≤ 1024.** W_down is full-rank like W_gate (CF4) and W_up (CF5) in trained Qwen3-1.7B-Base. Source: `sonnet_ladder_v2/cheap_tests/run_001/f_wnorm_result.md`. **Class-level finding**: no MLP weight matrix admits low-rank decomposition without retraining. The only MLP-weight compression path is per-element quantization (Q4_K_M et al.).

- **v2-CF4 — Per-neuron W_gate / W_up cosine alignment is distributed, not localized (Qwen3-1.7B-Base, 2026-05-09)**: across 172,032 neurons (28 × 6144), median |cos(W_gate[i,:], W_up[i,:])| = 0.071, mean = 0.092, max = 0.976. Random baseline d=2048 gives median 0.015 — so trained weights have ~5× higher median alignment than random, BUT 0/172k neurons cross |cos| ≥ 0.99 and only 0.0006% cross |cos| ≥ 0.90 (essentially random-baseline level). Source: `f6_walign_result.md`. **Implication**: ICLR 2025 Yadav's gate/up alignment observation is a soft aggregate statistic, not a per-neuron exact-fold property. Algebraic-identity per-neuron MLP fold is dead.

- **v2-CF5 — W_up canonical-gauge sign distribution is uniform random (Qwen3-1.7B-Base, 2026-05-09)**: after canonical-gauge sign flip (max-abs element positive per row), Shannon entropy H_per_row = **0.9996 bits/weight** in trained model AND 0.9996 in random-init same architecture. p(sign=+) global mean = 0.5002 trained vs 0.4973 random-init (within sampling noise). W_gate / W_up max-abs sign agreement = 0.5001 (50/50 random). **Sign-codebook compression of trained weights is infeasible.** Source: `a1_gfrs_2_result.md`. **Class-level finding**: training does not impose sign-correlation structure on MLP weights at the row level.

---

## Confirmed empirical structural findings (CF1-CF12)

These are the v1 ladder own outputs. Each was produced by a Sonnet-ladder-selected experiment, run end-to-end, with a result the user confirmed.

1. **CF1 — SwiGLU W_up firing-rank dominance (Qwen3-1.7B)**: SwiGLU top-K firing dominated by `W_up·x`, not `W_gate·x`. Predictors using gate-side only fail in deep Qwen3 layers (precision below random). Generalizes to all SwiGLU/SiLU-gated families.

2. **CF2 — Residual additivity uninformative**: cos(h_L, h_{L+1}) ~ 0.99. Layer-pair angle predictor based on raw residual addition is uninformative.

3. **CF3 — K-dependent activation outlier dynamicity (Qwen3-1.7B normal stratum)**: per-token top-K outlier-channel-set Jaccard:
   - K=0.1% (top ~2 channels): Jaccard **0.718** — channel-static at top tier
   - K=1% (~20 channels): Jaccard **0.31** — token-dynamic
   - K=5% (~100 channels): Jaccard **0.26** — token-dynamic
   - K=10% (~200 channels): Jaccard **0.24** — token-dynamic
   LLM.int8() channel-static finding holds **only at K=0.1%**. At any operational K for quantization, outliers rotate per token.

4. **CF4 — W_gate full-rank in trained Qwen3-1.7B**: PPL doubles at 50% rank truncation. r_99/d ~ 1.0.

5. **CF5 — W_up MORE rank-sensitive than W_gate**: dNLL +2.34 vs +0.77 at K=1024 (50% rank). The asymmetry is consistent with CF1: W_up carries the firing-rank signal, so dropping its top directions costs more.

6. **CF6 — SDZC gate near-constancy 1.5% globally; Layer 1 anomaly at 36%**: SwiGLU dead-zone gate variance is uniformly tiny across layers EXCEPT layer 1, where 36% of gate channels have variance below the foldable threshold. Layer 1 admits structural simplification that the rest of the stack does not.

7. **CF7 — Compound: firing-rank dominance does NOT translate to no-retraining structural compressibility (in any tested form)**: this is a class-level kill. The intuition "the firing-rank signal lives in a low-rank subspace, so we can compress" was tested across multiple variants (gate-only low-rank, joint low-rank, calibration-fitted projection); all failed.

8. **CF8 — Compound (formalized): trained Qwen3 MLP weights (W_gate AND W_up) full-rank; class-level kill on "find low-rank in MLP weights for compression"**: r_99/d ~ 1.0 for both W_gate and W_up. Any proposal that relies on rank-truncating MLP weights without retraining is dead at the class level.

9. **CF9 — Compound: theoretical frame mismatch**: when an ideator imports machinery from another field, Stage 2/3 must verify the imported theory preconditions hold for the target object. Three R6 kills exemplified the failure pattern (Count-Sketch on dense INT4 residuals; Bochner on weight matrices; tabulation hashing on c=4 with 16 entries). Before scoring an idea that imports a named theorem, verify the structural precondition (sparsity, shift-invariance, collision-resistance, etc.). If the precondition does not hold, the imported machinery is decorative; what is left is usually well-known prior art under a different name.

10. **CF10 — WDLA failure mode: calibration ill-conditioning**: when Stage 5/6 selection involves least-squares fitting on calibration data, the selector and red-team MUST explicitly check `n_params_to_fit` vs `n_independent_samples * n_output_dims_per_sample`. WDLA failed because A had 2.1M parameters fit on ~2M calibration values with ridge=1e-3, producing R^2=1.0 on calibration and R^2=-118000 on held-out. Mitigations: N >> n_params/d_output, strong ridge when underdetermined, force low-rank by construction, explicitly report cal-vs-eval R^2.

11. **CF11 — CF8 boundary delineation (R3-A AQFKV)**: CF8 applies to MLP weights ONLY. Attention weight matrices have structurally distinct, more concentrated spectra. W_Q r_99/d ~ 0.63 vs W_gate/W_up r_99/d ~ 1.0. W_Q at K=512 (50% rank) gives dNLL=+0.20; W_K at K=512 gives dNLL=+0.29. Global K=128 GO is a head-sharing finding (16 heads collapse to ~1 head worth of subspace) NOT a per-head rank reduction (per-head K=64 NO-GO at +1.53). MLA-style joint Q-K projection lines now empirically motivated for Qwen3 post-training.

12. **CF12 — Tied embedding/lm_head full-rank (R6-B LHQD)**: in Qwen3-1.7B-Base where `tie_word_embeddings=True`, the joint embed/lm_head matrix has r_99=1992/2048 — flat spectrum despite V>>d shape. Functional collapse below r=2048 (dNLL=+19.96 at r=1024, top-1 retention 0.005). Per-row norms full (no dead rows). Gradient flows through both input embedding lookup AND output projection, keeping every direction load-bearing. Untied lm_head (Qwen3-8B+) remains untested — the question is genuinely different there.

---

## Class-level kills (carried forward as flavor anti-frames)

These are not single-idea kills; they are flavor-level kills that bind future generation:

- **No-SGD low-rank decomposition of MLP weights** (CF7, CF8). Dead at the class level. Variants: SVD truncation, NMF, NMF-on-residuals, Tucker on MLP weight tensor, basis sharing across MLP, calibration-fitted MLP projection. All dead.
- **Calibration-fitted parameters with n_params >> n_independent_samples** (CF10). Any proposal that fits a high-parameter affine / projection / codebook must explicitly state n_params and n_samples and pass the conditioning check.
- **Imported theorem without precondition verification** (CF9). Bochner on non-shift-invariant matrices; Count-Sketch on dense signals; tabulation hashing on small alphabets; sparsity-assuming algorithms on dense residual streams.
- **Tied lm_head SVD truncation** (CF12). Dead at any r < d_model with catastrophic dNLL. Variants on tied configurations are dead. Untied configurations remain open.
- **Per-head W_Q rank truncation** (CF11). NO-GO at K=64 (+1.53). Class kill on per-head decomposition; head-shared decomposition lives.

---

## What is NOT carried forward into v2

The following measurements exist in `sonnet_ladder/SUMMARY.md` as CF13-CF15. They were produced by the Opus pipeline pre-program measurements. They are NOT confirmed by the user as of the v2 cutover. They are NOT load-bearing inputs for v2 Stage 1.

If a v2 proposal needs the substance of any of these, the proposal must include a v2 measurement that re-derives the number from scratch as the first cascade rung. That re-derivation enters v2 SUMMARY only after the v2 round confirming it.

For traceability, the in-flight measurements are:

- (in-flight) Windows 11 NTFS mmap random 4 KB read latency: see `sonnet_ladder/SUMMARY.md#13`. Not a v2 input.
- (in-flight) Massive-activation channel dominance in residual stream: see `sonnet_ladder/SUMMARY.md#14`. Not a v2 input.
- (in-flight) Stratified residual-stream intrinsic dimension after BOS-stripping: see `sonnet_ladder/SUMMARY.md#15`. Not a v2 input.

This is a deliberate firewall. The v1 confirmed findings are the substrate v2 reasons from. Mixing in unconfirmed parallel-pipeline measurements would erode the ladder epistemic moat (the moat being: every v2 finding traces to a v2-selected and v2-run experiment).

---

## Killed / pre-empted ideas — count

~50 across all v1 rounds and tracks. See `sonnet_ladder/KILL_LIST.md` for the full ledger; it is read-only for v2 reference. v2 own KILL_LIST.md lives at `sonnet_ladder_v2/KILL_LIST.md` and starts empty.

The v1 kill list is referenced by v2 Stage 1 ideators as a hard kill list — any v1-killed idea is also v2-killed. The v2 kill list grows monotonically on top of it.

---

## Round-over-round insights carried forward

These process insights from v1 are load-bearing for v2 and are restated here so the v2 Stage 1 ideators have them in their context window:

- **Mass pre-emption is signal, not failure.** When all advancing ideas in a round get pre-empted by recent papers, the search has reached a publishing-saturated region. Genuinely novel space lives just past the saturation boundary. v2 Stage 1 ideators must push past, not retreat.
- **Frame-novelty matters more than mechanism-novelty.** Ideators converge on saturated regions because that is their training distribution. Cross-orientation parallelism (the v2 structural change) is the v2 mechanism for breaking the convergence. Within an orientation, frame-orthogonal generation is the within-orientation mechanism.
- **Empirical findings the pipeline produces itself are the most pre-empt-resistant.** CF1, CF3, CF6, CF11, CF12 — these are not on arXiv yet, so ideas grounded in them have natural moats. v2 Stage 1 ideators should preferentially ground load-bearing claims in v2-confirmed CF_i rather than in published-literature numbers.

---

## When you return

1. Check this file first.
2. Read newest entries in v2 `KILL_LIST.md` (bottom of file).
3. v2 `SELECTED.md` has the queued experiments + results from completed v2 rounds.
4. v2 `ROUND_<N>_TRANSCRIPT.md` per-round summaries.
5. `prompts/ROUND_RUNBOOK.md` has the v2 runbook.
6. `prompts/STAGE1_IDEATOR.md` and `prompts/ORIENTATIONS.md` are the Stage 1 contract.
