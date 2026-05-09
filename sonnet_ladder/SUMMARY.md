# Pipeline Status Summary (auto-updated by ladder loop)

This file is updated after each round closes. Look here first to see where the pipeline is.

## Current state

**Active mode**: long-autonomous (running rounds until user returns)

**Rounds completed**:
- R1 unified — selected experiment ran (cross-layer row predictor; NO-GO; structural finding 1: SwiGLU W_up firing-rank dominance)
- R2 unified — selected experiment ran (PDAP outlier-Jaccard; structural finding 3: K-dependent outlier dynamicity)
- R3 unified — selected experiment ran (W_gate rank sweep; NO-GO; structural finding 4: W_gate full-rank in Qwen3-1.7B)
- Track A R1 — selected SDZC ran (NO-GO globally; Layer 1 anomaly: 36% foldable)
- Track B R4 — selected W_up rank sweep ran (NO-GO; structural finding 5: W_up MORE rank-sensitive than W_gate)
- Track A R2 — WDLA NO-GO (calibration ill-conditioning; structural finding 8 added)
- Track B R5 — SPADC selected (Stage 6 weakened; bimodal claim unsupported by R4 SDZC)
- Track A R3 — AQFKV ran. **GO globally at K=128 (ΔNLL=+0.98); GRAY per-head; W_K K=512 ΔNLL=+0.29.** Structural finding 11: CF8 does NOT extend to attention. W_Q compressibility is head-redundancy, not per-head weight low-rank.
- Track B R6 — LHQD ran. **EMBED-TIED in Qwen3-1.7B-Base (Stage 6 gate caught it). Embedding/lm_head matrix spectrum is FLAT (r_99=1992/2048; var@r=256=0.28). Catastrophic functional collapse below r=2048 (ΔNLL=+19.96 at r=1024).** Structural finding 12: in tied configs, embed/lm_head joint matrix is full-rank — gradient comes through both input and output paths, keeping every direction load-bearing.

**Empirical structural findings to date**:

1. SwiGLU top-K firing dominated by W_up·x, not W_gate·x (Qwen3-1.7B)
2. Residual additivity gives cos(h_L, h_{L+1}) ≈ 0.99 (uninformative)
3. Activation outlier dynamicity is K-dependent (K=0.1% Jaccard 0.72; K=1-10% Jaccard 0.24-0.31)
4. W_gate full-rank in trained Qwen3-1.7B (PPL doubles at 50% rank)
5. W_up MORE rank-sensitive than W_gate (ΔNLL +2.34 vs +0.77 at K=1024)
6. SDZC gate near-constancy 1.5% globally; Layer 1 anomaly at 36%
7. **Compound**: R1 firing-rank dominance does NOT translate to no-retraining structural compressibility (in any tested form)
8. **Compound (formalized)**: Trained Qwen3 MLP weights (W_gate AND W_up) full-rank; class-level kill on "find low-rank in MLP weights for compression"
9. **Compound (CF9)**: Theoretical frame mismatch — imported theory's preconditions must be verified for the target object before adopting the mechanism
10. **WDLA failure mode**: calibration fits with n_params > n_independent_samples → memorization with garbage held-out (R²=1.0 cal / R²=-118000 eval). Now a Stage-5 check item in PIPELINE.md.
11. **CF8 boundary delineation (R3-A AQFKV)**: CF8 applies to MLP weights ONLY. Attention weight matrices have structurally distinct, more concentrated spectra. W_Q r_99/d ≈ 0.63 vs W_gate/W_up r_99/d ≈ 1.0. Direct measurement: W_Q at K=512 (50% rank) gives ΔNLL=+0.20; W_K at K=512 gives ΔNLL=+0.29. Global K=128 GO is a head-sharing finding (16 heads collapse to ~1 head's worth of subspace) NOT a per-head rank reduction (per-head K=64 NO-GO at +1.53). MLA-style joint Q-K projection lines now empirically motivated for Qwen3 post-training.
12. **Tied embedding/lm_head full-rank (R6-B LHQD)**: in Qwen3-1.7B-Base where `tie_word_embeddings=True`, the joint embed/lm_head matrix has r_99=1992/2048 — flat spectrum despite V>>d shape. Functional collapse below r=2048 (ΔNLL=+19.96 at r=1024, top-1 retention 0.005). Per-row norms full (no dead rows). Refutes the Godey&Artzi (2603.10145) implication for tied configurations: gradient flows through both input embedding lookup AND output projection, keeping every direction load-bearing. Untied lm_head (Qwen3-8B+) remains untested — the question is genuinely different there.
13. **Windows 11 NTFS NVMe page-fault profile (Opus pipeline R1 PRE-1, 2026-05-08)**: single-thread random 4 KB read on mmapped file — **median 137μs** (p10 122μs / p90 158μs / p99 201μs). This is 24× the optimistic 5.6μs assumption. **Kills E0a-optimistic branches of Reach and Unconventional v2 cascades.** BUT: NEXTENT prefetch is empirically confirmed engaged — same-extent follow-on reads at median 1.9μs (72× faster than first-page). NTFS Cache Manager pulls rest of 1 MB extent for ~free after first fault. The Unconventional v2 mechanism (NTFS extent prefetch as routing primitive) is empirically validated on Windows 11. Implication: PFOR + NEXTENT layout + ZIPFLOCK is the deployable architecture for NVMe-streamed 70B; Reach v2 must use overlapped ReadFileEx queued prefetch for sub-matrix block streaming.
14. **Massive-activation channel dominance (Opus R1 PRE-2 M-Strat, 2026-05-08)**: residual-stream variance in Qwen3-1.7B-Base is catastrophically dominated by ~1-2 channels in middle layers. Per-layer SVD of residual-stream activations across 7809 tokens shows participation ratio ≈ 1.0 and k_99 = 1 in layers 4-16 (one channel explains 99% of variance). Layer 0 has PR=58 / k_99=276; deep layers (24-27) recover to PR=2-4 / k_99 ≈ 280-309. This is consistent with massive activations (Sun et al. 2024) at extreme scale — the BOS-sink and similar attention-sink positions have magnitudes ~100-1000× normal tokens, dominating naive SVD. **Implication**: the First-Principles E2 MLP-saturation inequality test as currently designed measures k_residual contaminated by massive activations and would falsely refute the inequality. A stratified version (project out top-2 magnitude channels before measuring) is needed.

15. **Stratified residual-stream intrinsic dimension (Opus R1 PRE-2 v2, 2026-05-08)**: when massive activations are stripped (top-2 magnitude channels projected out per layer; first 4 BOS/attention-sink tokens excluded), the typical-token residual stream lives in **~223 dimensions out of 2048 (k_99/d ≈ 0.11)** in middle layers. **First-Principles geometric-fingerprint hypothesis SUPPORTED, with much stronger stratification than the original 1400-1700 prediction.** The ~10× tighter actual intrinsic dimension implies joint Q* construction can compress weights more aggressively than previously projected; r=256 lm_head truncation in Q* basis may genuinely dissolve the +19.96 catastrophe (FP E4 binary settling experiment). Independently: top-2 magnitude channels are **CONSISTENT across layers 2-27** at indices [1999, 1793] (layer 0 differs at [784, 1371]). This empirically confirms Composition v2's M5 claim (L1-resident persistent residual highway) — the "highway" of static outlier channels really is two specific channels persisting across nearly all layers. Combined finding: the residual stream of Qwen3-1.7B-Base has a 2-dim "highway" of extreme-magnitude persistent channels + a ~220-dim typical-token operating subspace + a ~1825-dim near-noise null space. This is the structural fingerprint that both First-Principles (Q* on stratified subspace) and Composition (highway/bus/bulk decomposition) need.

## Killed / pre-empted ideas — count

~50 across all rounds and tracks. See KILL_LIST.md for full ledger.

## Active selected experiments queue

See SELECTED.md.

## Round-over-round insights

- **Mass pre-emption is signal, not failure**: Track A R2 had ALL 4 advancing ideas pre-empted by 2025-2026 papers Stage 1 hadn't found. This validated the publication-velocity policy and led to PIPELINE.md updates.
- **Frame-novelty matters more than mechanism-novelty**: ideators tend to converge on saturated regions (ML training distribution); Stage 4 skeptic explorers with cross-domain framing prompts produce more pre-empt-resistant ideas.
- **Empirical findings the pipeline itself produces are the most pre-empt-resistant**: SDZC Layer-1 anomaly, R4 W_up sensitivity, R2 K-outlier — these aren't on arXiv yet, so ideas grounded in them have natural moats.

## When you return

1. Check this file first
2. Read newest entries in KILL_LIST.md (bottom of file)
3. SELECTED.md has the queued experiments + results from completed runs
4. ROUND_*_TRANSCRIPT.md per-round summaries
5. PIPELINE.md has the runbook + accumulated process insights
