# Stage 5 — Selector — Run 007

Date: 2026-05-09
Model: claude-sonnet-4-6
Inputs: stage2_judge, stage3_deep_research, stage4_skeptic (run_007); SUMMARY.md (CF1–CF12, v2-CF1, v2-CF2); KILL_LIST.md.

---

## Selection Summary

**Track A pick: S3 — ATRC W_V/W_O Spectrum Sweep (35 min)**
**Track B pick: KSATT + S7 compound — Kronecker Smoke + Follow-on (5 min gate + ≤30 min follow-on)**

---

## Scoring Traces

### Track A — S3 (ATRC W_V/W_O Spectrum Sweep)

Stage 2 total for ATRC: 12. Stage 3 confidence: +2 (clean binary settler in ≤35 min, GO/NO-GO both load-bearing). Subtotal: 14.

Convergence multiplier: 1.5 (C1 is a 5-orientation cluster: R7-R-001 ATRC, R7-R-003 VJOFR, F7-1 WOVR, C9-JAQV, C11-JVSTAB all converge on W_V/W_O as the first rung). 14 × 1.5 = 21.

Frame-novelty bonus: +1. CF11-style r_99/d applied to W_V/W_O is confirmed NOVEL by Stage 3 (no paper reports this metric for these matrices on Qwen3-family or equivalent as of 2026-05-09). Not in the round's saturated-frame list.

NO-GO-finding bonus: +2. NO-GO (W_V MLP-class, r_99/d ≥ 0.95) is a CF entry candidate: extends CF8 boundary inward, closes the attention-tiered cascade class, and kills all five C1 advancers simultaneously. This is exactly the kind of NO-GO that earns +2 under the Stage 5 algorithm.

**Track A adjusted score: 21 + 1 + 2 = 24.**

Re-verification of novelty (2026-05-09): Stage 3 searched for W_V/W_O spectral concentration measurements on Qwen3-family models. A3 (arXiv:2505.12942) performs activation-aware joint W_V·W_O compression at 70B without reporting CF11-style r_99/d per matrix. SVD-LLM (arXiv:2403.07378), ARA (arXiv:2510.19389) apply SVD broadly without reporting this specific metric. No paper found reporting CF11-style r_99/d for W_V or W_O on Qwen3-family as of 2026-05-09.

Runners-up considered: S4 (Profile-Guided Dead-Neuron Elimination, A-track, 20 min add-on) is gated on FRCF running first and has an expected 1–5% neuron-fraction payoff — structural finding is much narrower than C1's class-level outcome. C12-RUNE (FREE SWING, A-track, 20 min) is a measurement with high uncertainty and no CF tether. Neither approaches Track A score 24.

### Track B — KSATT → S7 compound

KSATT Stage 2 total: 12. Stage 3 confidence: +2 (5-min binary settler, clear GO/NO-GO). Subtotal: 14.

Convergence multiplier: 1.0 (no convergence cluster; KSATT is a solo frame-novelty advancer; wildcard non-penalization applies: solo status does not incur a convergence-deficit penalty for frame-novelty picks).

Frame-novelty bonus: +2. F7-4 KSATT holds the highest frame-novelty score in this round (A2=3). Stage 3 confirmed: no paper tests post-training implicit Kronecker structure in trained standard LLM attention weights as of 2026-05-09. The frame ("did SGD converge to Kronecker organization?") is not present in the saturated-frame list and is not a compression-method proposal — it is a structural characterization question. Listed explicitly as an under-represented frame in Track B's framing diversity. Bonus = +2.

NO-GO-finding bonus: +2. KSATT NO-GO (< 20% Frobenius at all tested partitions) closes the entire class of "post-training implicit Kronecker compression of LLM attention weights" permanently — candidate CF entry. GO gates S7 (30–45 min additional experiment); the compound runtime remains within the 8-hour ceiling (5 + 45 = 50 min).

Elegant-equivalence multiplier: 1.2. KSATT carries elegance-class `algebraic-identity` (exact Kronecker factorization if structure exists); pre-multiplier raw score 14 ≥ 9 floor; multiplier is constructive. 14 × 1.0 × 1.2 = 16.8.

**Track B adjusted score: 16.8 + 2 + 2 = 20.8.**

Re-verification of novelty (2026-05-09): Stage 3 found no paper testing post-training implicit Kronecker structure in trained standard LLM attention weights. Closest: "Identifying Kronecker product factorizations" (arXiv:2510.25292, Oct 2025) targets binary/sparse matrices, not trained LLM weights. SoKA (arXiv:2506.15251) uses Kronecker for PEFT fine-tuning, not post-training characterization. Monarch/Kaleidoscope are training-time constraints. No pre-emption found as of 2026-05-09.

Runners-up considered: C10-FRCF (45 min, score ≈ 11 × 1.0 + 1 + 2 = 14 effective) is strong but trails KSATT+S7 on frame-novelty bonus and NO-GO finding class. KSATT+S7 is selected as the compound; if KSATT NO-GOs at 5 min, the slot falls to FRCF (45 min, explicitly named as runner-up).

---

## Track A Experiment Plan

### 1. Title and one-line description

Round 7 / Track A — S3 (ATRC): W_V and W_O Spectrum Sweep.  
Measure CF11-style r_99/d for W_V and W_O across all 28 layers of Qwen3-1.7B-Base; determine whether the attention-tiered residency cascade can extend to V+O matrices.

### 2. Class tags

`compression-lr`, `arch-fold`

### 3. Hypothesis

CF11 showed W_Q (r_99/d ≈ 0.63) and W_K (r_99/d ≈ 0.79) have substantially more concentrated spectra than MLP weights (r_99/d ≈ 1.0). W_V and W_O are the last attention weight matrices unmeasured on the v2 ladder. The hypothesis: W_V and W_O are also CF11-class (r_99/d < 0.80), enabling global SVD truncation at K ≤ 256 with ΔNLL < 0.50 nats — extending the attention-tiered cascade to the full attention-weight family and opening 4–8× total attention compression at 70B. The structural claim depends on CF11 (confirmed W_Q/W_K) and CF8 (MLP full-rank boundary); no other CF numbers.

### 4. Smallest test

- Model: Qwen3-1.7B-Base, bf16, tokenizer `Qwen/Qwen3-1.7B`
- Calibration corpus: N/A (pure weight-spectrum measurement; no calibration data used)
- Eval corpus: WikiText-2, 500 held-out tokens (same split as AQFKV v2 R3-A)
- Matrices touched:
  - W_V: GQA-config 8 KV heads × d_head × d_model = 8 × 128 × 2048. Verify exact shape from config.json before running. At default Qwen3-1.7B: d_model=2048, num_key_value_heads=8, head_dim=128.
  - W_O: 16 heads × d_head × d_model = 16 × 128 × 2048 (standard). Verify exact shape.
  - All 28 layers for both matrices.
- SVD sweep: compute r_99/d per layer per matrix. Measure ΔNLL at K ∈ {512, 256, 128} for W_V; K ∈ {256, 128} for W_O (global simultaneous truncation, analogous to AQFKV W_K measurement).
- WOVR add-on (3 min, zero additional experiment cost): for layers 0, 7, 14, 21 and head 0, compute canonical angles between U_O (top-64 left singular vectors of W_O_h) and V_V (top-64 right singular vectors of W_V_h) via QR + SVD of (U_V^T V_O). Report mean canonical angle; adds structural finding about A3's compression basis.
- Wall-clock: ~35 min total (28 layers × 2 matrices × SVD; eval passes at K ∈ {512,256,128,full}). WOVR add-on: +3 min.
- Output path: `experiments/stage0/ladder_v2/round7_atrc/`
- Script: `scripts/atrc_wv_wo_spectrum.py` (new script needed). Script inputs: model path, layer range 0–27, eval tokens 500. Script computes: (a) SVD per layer per matrix, records cumulative variance fraction at K=64,128,256,512; (b) forward-pass ΔNLL at each K for W_V and W_O; (c) r_99 per layer; (d) WOVR canonical-angle add-on for sampled layers/heads. Outputs: per-layer r_99/d table, ΔNLL table, WOVR angle table, summary statistics.

### 5. Go threshold

W_V r_99/d < 0.80 at ≥ 22 of 28 layers AND ΔNLL(W_V, K=256) < 0.50 nats (global simultaneous). Both must hold.  
SOFT-GO: r_99/d = 0.80–0.90 AND ΔNLL(K=256) = 0.50–1.50 nats — partial CF11-class; per-layer adaptive K required (follow-up: 2-hour per-layer K sweep).

### 6. No-go threshold

W_V r_99/d ≥ 0.95 at majority of layers OR ΔNLL(W_V, K=256) ≥ 1.50 nats.  
NO-GO ⇒ W_V is MLP-class; CF8 boundary extends to V matrix; attention-tiered cascade for W_V+W_O is closed. Class-level kill: "post-training global SVD truncation of W_V without retraining" joins the CF8 family. All five C1 advancers (ATRC, VJOFR, WOVR, JAQV, JVSTAB) are killed simultaneously.

### 7. Ambiguous-zone follow-up

SOFT-GO (r_99/d = 0.80–0.90): Run per-layer K sweep (28 layers × 5 K-values) to find the per-layer GO threshold. Runtime: ~2 hours. Resolves: whether adaptive-K per-layer assignment can achieve ΔNLL < 0.50 nats. If per-layer K also fails, full NO-GO applies.

### 8. Kill criteria (Stage 6 amendment slot)

- GQA shape mismatch: Qwen3-1.7B has 8 KV heads for W_V vs 16 Q heads. Script must handle the GQA asymmetry explicitly. If shape loading fails, Stage 6 amends the shape handling before execution.
- A3 engineering pre-emption does NOT kill this experiment — Stage 3 confirmed the CF11-metric characterization is distinct from A3's activation-weighted compression.
- If a paper reporting CF11-style r_99/d for W_V on Qwen3-family appears between now and execution, this experiment is pre-empted. Stage 6 reviewer checks arXiv on execution day.

### 9. Skeptic-controls declaration

The experiment's primary claim is "W_V is CF11-class" — a structural characterization claim ("X is consistent across layers"). Required controls:

- **Permutation control:** run the same r_99/d measurement on W_V where each row of W_V has been randomly permuted (destroying singular value structure while preserving row-norm distribution). The GO threshold must be met by the real signal AND real r_99/d substantially below the permuted baseline. Expected: permuted W_V has r_99/d ≈ 1.0 (no concentration) — the gap between real and permuted is the load-bearing signal. State gap explicitly in results: "real r_99/d = X, permuted r_99/d = Y, gap = Z."
- **Random-init control:** run the same r_99/d measurement on a randomly initialized Qwen3-1.7B model (weights drawn from uniform/normal, same dtype). GO requires trained model r_99/d substantially below random-init r_99/d. This catches architectural artifacts. v2-CF2 baseline for W_Q: trained r_99/d ≈ 0.63, random-init r_99/d ≈ 1.0 — the same pattern is expected here. If random-init shows r_99/d < 0.80 for W_V, the concentration is architectural, not a trained-model property.
- **Multi-seed:** The experiment involves no stochastic calibration (pure weight analysis). The only randomness is the permutation control's permutation draw. Run permutation control with 3 distinct random seeds for the permutation; report mean ± std of permuted r_99/d. Confirms the permuted baseline is stable.

### 10. Runtime estimate

- Setup (model load, shape verification): 3 min
- SVD per layer per matrix (28 × 2 × ~30s): 28 min
- Eval forward passes at each K: 5 min (4 K-values × ~1 min each)
- WOVR add-on: 3 min
- Permutation + random-init controls: 5 min (single-layer spot checks; full 28-layer if time permits)
- Post-processing and table generation: 2 min

**Total: ~38 min. Well within 8-hour ceiling.**

### 11. Script identification

New script needed: `scripts/atrc_wv_wo_spectrum.py`  
Inputs: `--model-path`, `--layers 0-27`, `--eval-tokens 500`, `--output-dir experiments/stage0/ladder_v2/round7_atrc/`, `--add-wovr-angles` flag.  
Key computations: (1) load each layer's W_V and W_O, reshape per GQA config; (2) numpy SVD truncation and cumulative variance computation; (3) forward-pass NLL at each K via patched model weights; (4) WOVR canonical angle computation for sampled layers; (5) permutation and random-init baselines for skeptic controls.  
Output files: `wv_r99_table.csv`, `wo_r99_table.csv`, `delta_nll_table.csv`, `wovr_angles.csv`, `summary.md`.

### 12. Downstream implications

GO: Extends CF11 to all four attention weight matrices (W_Q, W_K, W_V, W_O). Opens 70B attention-tiered cascade: combined W_Q (K=128) + W_K (K=256) + W_V (K≤256, pending) + W_O = 4–8× total attention compression. Kills C1 pending-advancers (VJOFR, WOVR compression application) as now resolved, not killed. Motivates next round: W_Q + W_V + W_O joint low-rank (MLA-style), practical DRAM-resident attention at 70B.

NO-GO: Extends CF8 to W_V (and potentially W_O). Class-level kill on "attention-tiered residency cascade via W_V/W_O SVD without retraining." Kills C1 cluster (ATRC, VJOFR, WOVR, JAQV, JVSTAB). Leaves CF11 boundary at {W_Q ≈ 0.63, W_K ≈ 0.79, W_V = full-rank}. RAOK (Track B deferred) is unaffected. Motivates pivot: Track A R4 must find a non-SVD attention compression path (MLA-style joint training? KV-delta coding from S5?).

### 13. Provenance

- Originating orientations: R (Reach, ATRC), F (First-Principles, WOVR), C (Composition, JAQV, JVSTAB), R (VJOFR)
- Convergence cluster: C1 (5-orientation — highest signal in this round)
- Stage 4 gap-idea slot: S3 (convergence enforcer)
- Frame-novelty bonus path: CF11-metric extension to W_V/W_O — confirmed NOVEL by Stage 3 as of 2026-05-09
- Runner-up if Stage 6 kills this pick: S4 (Profile-Guided Dead-Neuron Elimination, 20 min add-on to FRCF)

---

## Track B Experiment Plan

### 1. Title and one-line description

Round 7 / Track B — KSATT → S7: Kronecker Structure Smoke Test then Comparative Compression.  
5-minute Kronecker structure probe on W_Q (layer 14); if GO, 30-minute matched-parameter comparison of Kronecker rank-r vs SVD rank-K for W_Q compression quality.

### 2. Class tags

`compression-lr`, `arch-fold`, `frame-novelty`

### 3. Hypothesis

SGD training of transformer attention weights may implicitly induce Kronecker product structure — a natural attractor for any linear map that processes inputs with block-separable structure. The hypothesis: W_Q at layer 14, when reshaped as a p×q × p×q block matrix (p=32, q=64), has a rank-1 Kronecker decomposition capturing ≥ 50% of Frobenius norm. If confirmed, Kronecker rank-r compression at equal parameter count to SVD rank-K achieves lower ΔNLL (S7), because the Kronecker basis is better-aligned with the natural structure. The structural claim depends on CF11 (W_Q is compressible) but asks a different question: is SVD the best basis for that compression? No additional CF numbers are required for the smoke test; S7 requires CF11's SVD-K data for the matched-parameter comparison.

### 4. Smallest test

**Phase 1 — KSATT smoke test (5 min)**

- Model: Qwen3-1.7B-Base, bf16
- Matrices: W_Q concatenated global matrix (16 × 128 × 2048 → 2048 × 2048 for global; or per-head 128 × 2048)
- Layer: 14 (mid-depth, representative); heads 0 and 7
- Partitions tested: (p=32, q=64), (p=64, q=32), (p=128, q=16), (p=16, q=128)
- Operation: reshape W_Q head block to (p × q) × (p × q) block matrix M; SVD of M; compute ‖rank-1 Kronecker‖_F / ‖W_Q_head‖_F
- Wall-clock: ~5 min (four SVDs of at most 2048×2048 matrices on Ryzen 5 7530U)
- Output path: `experiments/stage0/ladder_v2/round7_ksatt/`
- Script: `scripts/ksatt_kronecker_test.py` (described in Stage 3 §8.5)

**Phase 2 — S7 follow-on (if KSATT GO, 30–45 min)**

- Condition: any tested partition achieves ≥ 50% Frobenius at rank-1 Kronecker
- Best partition from Phase 1 used for Phase 2
- Kronecker ranks: r ∈ {4, 8, 16, 32, 64}
- For each r: decompose W_Q (layer 14, all heads) into sum of r Kronecker factors (A_i ⊗ B_i); reconstruct; measure ΔNLL
- Matched SVD comparison: for each r, compute equivalent SVD rank K = r × (p² + q²) / (2 × d_model) (matched total parameter count); measure ΔNLL(SVD at K) from CF11's existing data or recompute
- Calibration/eval corpus: WikiText-2, 500 held-out tokens
- Wall-clock: ~30 min (Kronecker optimization at r=4,8,16,32,64; ΔNLL eval passes)
- Output: `experiments/stage0/ladder_v2/round7_s7/`
- Script: `scripts/s7_kronecker_vs_svd.py` (new script; inputs: W_Q path, partition shape, r-values, eval tokens; outputs: Kronecker ΔNLL table, SVD matched-param ΔNLL, parameter efficiency gain)

### 5. Go threshold

**KSATT GO:** ≥ 50% Frobenius at rank-1 Kronecker for any tested partition at layer 14, head 0 OR head 7. Both heads must be tested; GO requires at least one partition succeeding on at least one head.

**S7 GO (conditional on KSATT GO):** Kronecker ΔNLL < SVD ΔNLL at matched parameter count for ≥ 3 of 5 tested r values AND the gap ≥ 0.05 nats (non-trivially better). Confirms Kronecker is a structurally superior compression basis for W_Q.

### 6. No-go threshold

**KSATT NO-GO:** < 20% Frobenius at all tested partitions for both heads at layer 14.  
NO-GO ⇒ SGD does NOT converge to Kronecker organization of attention weights. Class-level kill: "post-training implicit Kronecker compression of LLM attention weights." Slot falls to FRCF (45 min, dual-signal neuron precision, named runner-up).

**S7 NO-GO (conditional on KSATT GO):** Kronecker ΔNLL ≥ SVD ΔNLL at matched parameters. Partial Frobenius structure (KSATT GO) does not improve reconstruction quality; Kronecker is "decorative." Class narrowing: Kronecker structure exists in W_Q but does not align with the compression-relevant singular subspace; SVD remains optimal.

### 7. Ambiguous-zone follow-up

**KSATT Gray (20–50% Frobenius):** Test 4 additional partition shapes (p=4, q=512; p=8, q=256; p=16, q=128; p=256, q=8) and higher ranks r=2,4. Runtime: +10 min. If any partition at r=2 reaches ≥ 50%, proceed to Phase 2 with that partition.

**S7 Gray (Kronecker ΔNLL within 0.05 nats of SVD ΔNLL):** Run all 28 layers (not just layer 14) to test whether the benefit is layer-dependent. Runtime: +45 min. Resolves: whether particular layers with higher Kronecker structure show a larger benefit.

### 8. Kill criteria (Stage 6 amendment slot)

- Reshape ambiguity: the Kronecker test requires careful reshaping of the GQA W_Q matrix (concatenated heads vs per-head). Stage 6 must verify the script handles both the per-head and global reshape correctly before execution.
- If a paper reporting post-training Kronecker characterization of standard LLM attention weights appears before execution, Stage 6 checks and may pre-empt.
- KSATT NO-GO at Stage 6 review triggers runner-up (FRCF) without further discussion.

### 9. Skeptic-controls declaration

The experiment's claims: "Kronecker structure is consistent across heads" (after extension beyond layer 14) and "Kronecker compression works on this stack."

- **Permutation control (KSATT Phase 1):** Run the same Frobenius fraction measurement on a randomly row-permuted W_Q_head (destroys potential Kronecker structure while preserving row-norm distribution). The GO threshold (≥ 50%) must be met by the real W_Q AND substantially exceed the permuted baseline. Expected permuted baseline: < 10% (random matrices have no Kronecker structure). State the gap explicitly.
- **Random-init control:** Run the same Kronecker test on a randomly initialized W_Q of the same shape (weights drawn from Gaussian with matched variance). GO requires trained W_Q Frobenius fraction substantially above random-init. This catches "architectural Kronecker" (structure from matrix shape alone, not training). For Kronecker, this is crucial: a 2048×2048 random matrix reshaped as 32×64 × 32×64 has a near-zero rank-1 Kronecker fraction by construction; if trained W_Q shows the same, training added no structure.
- **Multi-seed for S7 (if Phase 2 runs):** Kronecker factorization optimization (fitting A_i, B_i for each r) has a non-trivial initialization dependence. Run with 3 random initializations for the Kronecker factor optimization; report mean ± std ΔNLL. GO requires mean ΔNLL < SVD ΔNLL and worst-of-3 not worse than SVD by more than 0.10 nats.

For KSATT Phase 1 alone: permutation and random-init controls add < 5 min and are mandatory per v2 protocol.

### 10. Runtime estimate

- Phase 1 setup + SVD: 5 min
- Permutation + random-init controls for Phase 1: 4 min
- Total Phase 1: 9 min

If KSATT GO:
- Phase 2 Kronecker optimization at r=4,8,16,32,64: 20 min
- Phase 2 ΔNLL eval passes: 10 min
- Multi-seed (3 seeds for Phase 2): 15 min
- Total Phase 2: 45 min

**Total compound: 9 + 45 = 54 min. Within 8-hour ceiling.**

If KSATT NO-GO (5–9 min): slot falls to runner-up FRCF (45 min).

### 11. Script identification

Phase 1: `scripts/ksatt_kronecker_test.py` — inputs: model path, layer index, head indices, partition shapes; outputs: per-partition Frobenius fraction table, permutation-control baseline, random-init baseline.

Phase 2 (if needed): `scripts/s7_kronecker_vs_svd.py` — inputs: model path, layer 14 (extendable), best partition from Phase 1, r-values, eval tokens, n_seeds; outputs: per-r Kronecker ΔNLL, matched SVD ΔNLL, parameter efficiency ratio, mean ± std over seeds.

### 12. Downstream implications

KSATT + S7 GO: Kronecker structure is a natural W_Q compression basis superior to SVD at matched parameters. Opens: Kronecker-based attention compression as a new class (parameter efficiency potentially 2–12× better than CF11's current K=128 bound). Motivates: full-layer Kronecker sweep, W_K Kronecker test. Constrains: CF11's K=128 boundary may be pessimistic if Kronecker achieves equal ΔNLL at lower parameter count.

KSATT NO-GO: Closes the entire post-training implicit Kronecker attention-weight class. Slot falls to FRCF. FRCF GO would open per-neuron dual-signal precision assignment; FRCF NO-GO would confirm CF1 and CF12 are structurally decoupled at the neuron level, constraining SpQR-style extensions.

### 13. Provenance

- Originating orientation: F (First-Principles, F7-4 KSATT), Stage 4 gap idea S7
- Convergence cluster: none (solo frame-novelty advancer; wildcard non-penalization applies)
- Stage 4 gap-idea slot: S7 (Kronecker vs SVD at matched parameters)
- Frame-novelty bonus path: A2=3 confirmed by Stage 3 (no prior paper tests retroactive Kronecker characterization of trained standard LLM attention weights as of 2026-05-09)
- Runner-up if KSATT NO-GO or Stage 6 kills this pick: C10-FRCF (45 min, Track B, dual-signal neuron precision, Stage 3 REFINE verdict, CF9 clean, CF10 safe, memory feasibility confirmed)

---

## Hand-off to Stage 6

Track A: run `scripts/atrc_wv_wo_spectrum.py`. Verify GQA W_V shape before execution. Include WOVR canonical-angle add-on (3 min). Permutation and random-init controls are mandatory (5 min additional). Report r_99/d table, ΔNLL table, and control baselines. If SOFT-GO, proceed to per-layer K sweep (2 hours additional).

Track B: run `scripts/ksatt_kronecker_test.py` with permutation and random-init controls. If Frobenius fraction ≥ 50% at any partition: immediately proceed to `scripts/s7_kronecker_vs_svd.py` with 3-seed multi-seed requirement. If Frobenius fraction < 20% at all partitions: record as KSATT NO-GO, add kill-list entry, and run FRCF runner-up (`scripts/frcf_firing_embed_correlation.py`, 45 min).

Both tracks can run sequentially within a single session (~100 min for full compound, assuming KSATT GO path).
