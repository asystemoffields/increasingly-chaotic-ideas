# Stage 5 — Selector — Run 006

**Agent**: Sonnet claude-sonnet-4-6 | **Date**: 2026-05-09
**Inputs read**: stage2_judge.md, stage3_deep_research.md, stage4_skeptic.md, SUMMARY.md, KILL_LIST.md, prompts/STAGE5_SELECTOR.md
**Kill-list enforced**: v2-CHEAP-TEST-001, v2-S3-R004-001, v2-S3-R007-002 (W_VO fold pre-empted by A3 at engineering level).

---

## Selection Trace

### Step 1 — Novelty re-verification (2026-05-09)

**Track A candidates re-checked:**

**C-JOINT-DELNLL**: Stage 3 found LatentLLM (MERL TR2025-075) for joint QK Tucker decomposition, but does NOT measure the interaction coefficient δ = ΔNLL_joint − (ΔNLL_Q + ΔNLL_K). No paper found that predicts δ from subspace_overlap(V_Q^K, V_K^K). NOVEL for the structural prediction. Date-stamped 2026-05-09.

**R6-WVOP**: Stage 4 (S7 note) flagged arXiv:2505.12942 (A3) as pre-empting the W_VO = W_V @ W_O fold at the engineering/application level. Stage 4 explicitly demotes WVOP's "fold and compress" claim. Per Stage 5 policy: **DEMOTED**. The Qwen3-specific spectral measurement (r_99(W_VO)/d, whether product rank is tighter than either factor) retains novelty, but the core compression application is no longer novel. This is below the selection floor as a standalone Track A pick.

**F6-WALIGN**: No paper reports per-neuron cosine histogram between W_gate and W_up rows. AERO folds globally after activation removal; WALIGN folds per-neuron without removing activation. NOVEL for the per-neuron granularity and the cos distribution measurement. Date-stamped 2026-05-09.

**Track A Stage 4 candidates:**
**S2 (QKPART)**: No published paper applies CF11's W_Q basis as a database partition-join key for attention bucketing. LSH-attention (Reformer, arXiv:2001.04451) uses random hashing; QKPART uses the trained W_Q eigenbasis — a post-training structural argument, not a random projection. NOVEL for the CF11-grounded partition-join framing. Date-stamped 2026-05-09.

**Track B candidates re-checked:**

**S1 (FNDP-WMCE composition)**: Stage 3 (U2-FNDP) found MemAscend adjacent on Linux/CUDA; Windows FILE_FLAG_NO_BUFFERING for ik_llama.cpp weight loading remains novel. The FNDP+WMCE anonymous-memory composition is not in MemAscend or any paper found. NOVEL for the Windows-specific double-buffer anonymous loading + OS Compression Store composition. Date-stamped 2026-05-09.

**C-WVOK / R6-ATTNQ-FULL**: Both gate on W_V spectrum. WVOP A3 pre-emption affects R6-ATTNQ-FULL (one of its cascade rungs is the V-O fold); C-WVOK's direct claim (W_V rank from CF11 coupling) is intact. The four-matrix joint ΔNLL interaction coefficient (M2 of ATTNQ-FULL) has no published precedent. PARTIAL demotion of ATTNQ-FULL's V-O fold rung; the joint interaction coefficient claim survives. C-WVOK advances cleanly.

---

### Step 2 — Convergence multipliers

| Candidate | Cluster | Multiplier |
|---|---|---|
| C-JOINT-DELNLL | None (solo; C is only orientation) | 1.0 |
| F6-WALIGN | None (solo orientation F) | 1.0 |
| S2 QKPART | None (solo; Stage 4 origin) | 1.0 (wildcard non-penalization applies) |
| S1 FNDP-WMCE | None (solo Stage 4) | 1.0 (wildcard) |
| C-WVOK | Convergence C2 (3-orientation: R, C, R-ATTNQ-FULL) | **1.5** |
| F6-SINKCONST | None ([FREE SWING]) | 1.0 |
| A6-LLMR | C3 (1-orientation convergence handle) | 1.0 |

---

### Step 3 — Frame-novelty bonuses

| Candidate | A2 | Frame status | Bonus |
|---|---|---|---|
| C-JOINT-DELNLL | 2 | δ prediction from subspace_overlap — NOVEL, not listed in any over-represented frame | +1 |
| F6-WALIGN | 2 | Per-neuron SwiGLU row-alignment — NOVEL, under-represented in run_006 framing targets | +2 (algebraic-identity + under-represented) |
| S2 QKPART | 3 (Stage 4) | Database partition-join on attention — NOVEL frame, under-represented (no Stage 1-3 coverage) | +2 |
| S1 FNDP-WMCE | 2 (Stage 4) | Anonymous weight loading + OS tier — ADJACENT frame (MemAscend nearby), not under-represented | +0 |
| C-WVOK | 2 | W_V rank from CF11 coupling — NOVEL for Qwen3 specific number, but W_V compression is a fairly direct extension | +1 |
| F6-SINKCONST | 3 | Algebraic sink origin from W_K·e_BOS × W_Q spectrum — NOVEL frame, under-represented | +2 |
| A6-LLMR | 2 | Monotone-entropy early exit — ADJACENT to SimLens/ConfLayers | +0 |

---

### Step 3b — Elegant-equivalence multipliers

| Candidate | Pre-multiplier raw score (S2-total + S3-confidence) | Elegance tag | Multiplier |
|---|---|---|---|
| C-JOINT-DELNLL | 13 + 2 = 15 | None | 1.0 |
| F6-WALIGN | 12 + 2 = 14 | algebraic-identity, constructive | **1.2** |
| S2 QKPART | ~10 + 2 = 12 | algebraic-identity (subspace-alignment), constructive, Stage 4 origin | 1.2 |
| S1 FNDP-WMCE | ~10 + 1 = 11 | None | 1.0 |
| C-WVOK | 12 + 2 = 14 | None explicit | 1.0 |
| F6-SINKCONST | 11 + 1 = 12 | None constructive (measurement, not fold) | 1.0 |

---

### Step 4 — NO-GO-finding bonuses

| Candidate | NO-GO structural finding | Bonus |
|---|---|---|
| C-JOINT-DELNLL | δ interaction coefficient table — constrains all joint Q+K compression proposals. CF-candidate for joint-compression interaction class. | **+2** |
| F6-WALIGN | SGD allocates gate/up DoF to maximize angular diversity — constrains all per-neuron alignment compression ideas | **+1** (class-level constraint, not CF-candidate yet) |
| S2 QKPART | Attention mass not bucketed → CF11 128-dim subspace does not separate key-query distances → constraints on any partition-attention scheme | +1 |
| S1 FNDP-WMCE | v2-CF13'' NVMe sequential throughput confirmed → constrains all NVMe-path bandwidth arithmetic | +1 |
| C-WVOK | W_V full-rank → CF11 head-redundancy is query-only → kills 3-proposal cascade at once | **+2** (class-level + CF11 extension + gates 3 proposals simultaneously) |
| F6-WALIGN (GO path) | Distribution of per-neuron alignment is a novel structural primitive regardless of mean level | included in +1 above |

---

### Step 5 — Final weighted scores

Formula: `(S2-total + S3-confidence) × convergence × elegance + frame-novelty + NO-GO`

**Track A candidates:**

| Candidate | S2+S3 | Conv | Eleg | Frame | NOGO | Final |
|---|---|---|---|---|---|---|
| C-JOINT-DELNLL | 15 | 1.0 | 1.0 | +1 | +2 | **18** |
| F6-WALIGN | 14 | 1.0 | 1.2 | +2 | +1 | **19.8** |
| S2 QKPART | 12 | 1.0 | 1.2 | +2 | +1 | **17.4** |
| F6-SINKCONST | 12 | 1.0 | 1.0 | +2 | +0 | **14** |
| R6-WVOP | DEMOTED (A3 pre-emption) | — | — | — | — | not eligible |

**Track A winner: F6-WALIGN (19.8) over C-JOINT-DELNLL (18.0)**

Tie-break check not needed — WALIGN leads by 1.8 points. WALIGN's elegant-equivalence multiplier (constructive algebraic identity) plus its fresh under-represented frame (per-neuron SwiGLU row-alignment) lifts it above C-JOINT-DELNLL. Its 5-min smallest test is also shorter than DELNLL's 12-min. All tie-break criteria favor WALIGN.

Runner-up: C-JOINT-DELNLL (18.0). If Stage 6 kills WALIGN (e.g., finds a per-neuron cosine distribution paper not surfaced at Stage 3), escalate to C-JOINT-DELNLL.

**Track B candidates:**

| Candidate | S2+S3 | Conv | Eleg | Frame | NOGO | Final |
|---|---|---|---|---|---|---|
| C-WVOK | 14 | **1.5** | 1.0 | +1 | +2 | **24** |
| S1 FNDP-WMCE | 11 | 1.0 | 1.0 | +0 | +1 | **12** |
| U2-FNDP (standalone) | 13 | 1.0 | 1.0 | +0 | +1 | **14** |
| C-DEEP-SPREAD-QBUD | 11 | 1.2 | 1.0 | +1 | +1 | **15.2** |

**Track B winner: C-WVOK (24.0)**

Convergence multiplier 1.5 (C2 three-orientation cluster) dominates. Its NO-GO is the single most informative outcome in the run: a full-rank W_V finding simultaneously gates and constrains R6-ATTNQ-FULL and WVOP, sharply constrains CF11's scope, and resolves the 70B residency arithmetic. Its smallest test (20 min) is cheaper than FNDP-WMCE (45 min + 1 hr).

Runner-up: C-DEEP-SPREAD-QBUD (15.2). If Stage 6 kills C-WVOK, escalate.

---

## Picks

**Track A: F6-WALIGN** — W_up/W_gate Row Alignment Fraction, 5-min histogram.
**Track B: C-WVOK** — W_V Subspace from CF11 Head-Redundancy Coupling, 20-min spectrum.

---

---

# Experiment Plan A: F6-WALIGN

## 1. Title and One-Line Description

**Run 006 / Track A — F6-WALIGN: Per-Neuron W_gate/W_up Row Alignment Fraction in Trained SwiGLU**

Measure the fraction of SwiGLU neurons where cos(W_gate[i,:], W_up[i,:]) ≥ 0.90 across all 172K neurons in Qwen3-1.7B; for aligned neurons, the exact algebraic identity y_i = λ_i · silu(z) · z collapses W_up[i,:] to a scalar multiple of W_gate[i,:], enabling lossless per-neuron row-pair compression.

## 2. Class Tags

`compression-algebraic-identity`, `mlp-weight`, `elegant-equivalence`

## 3. Hypothesis

**H1 (algebraic identity, exact):** For any neuron where cos(W_gate[i,:], W_up[i,:]) = 1 exactly, W_up[i,:] = λ_i · W_gate[i,:] for some scalar λ_i > 0. The SwiGLU output for that neuron is y_i = silu(W_gate[i,:]·x) · (W_up[i,:]·x) = silu(z_i) · λ_i z_i, where z_i = W_gate[i,:]·x. W_up[i,:] is algebraically redundant and can be replaced by the scalar λ_i at load time — no approximation, no retraining, no quality cost for exactly-aligned neurons (CF9: no imported theorem; direct trigonometric identity).

**H2 (empirical claim):** At least 5% of the 172,032 neurons in trained Qwen3-1.7B-Base have |cos(W_gate[i,:], W_up[i,:])| ≥ 0.90. This threshold is the Stage 5 GO gate. The empirical question is whether gradient descent, which is free to allocate both row vectors independently, systematically produces near-collinear (gate, up) pairs for a measurable neuron fraction.

H2 depends on CF8 (MLP W_gate and W_up are full-rank globally), which does NOT rule out alignment at the individual neuron level. WALIGN does not truncate any weight matrix — it folds individual row-pairs where the algebra is near-exact. CF8's full-rank finding applies to the matrix as a whole; per-neuron cosine statistics are orthogonal to global rank.

## 4. Smallest Test

**Model**: Qwen3-1.7B-Base BF16 (`Qwen/Qwen3-1.7B-Base`), full-precision weights, CPU-loaded.

**Calibration corpus**: None required. This is a pure weight-arithmetic measurement.

**Eval corpus**: WikiText-2 (held-out 512-token slice, same as AQFKV standard) — used only if Step 2 is reached (GO path follow-up).

**Matrices touched**: W_gate and W_up at all 28 layers. Shape: 6144 × 2048 each, BF16. Total: 28 × 2 × 6144 × 2048 × 2B = ~1.4 GB loaded.

**Step 1 — Histogram (~5 min):**
```
Script: experiments/stage0/ladder_v2/round6_walign/run_walign_hist.py
- Load W_gate[L] and W_up[L] for L in 0..27.
- For each layer: row_gate = W_gate[L] / ||W_gate[L]||_row  (shape 6144×2048, normalize each row)
- row_up   = W_up[L]   / ||W_up[L]||_row
- cos_sim[L] = einsum('id,id->i', row_gate, row_up)  (shape 6144,)
- Histogram bins: [-1.0, -0.99, -0.90, -0.70, -0.30, 0.30, 0.70, 0.90, 0.99, 1.0]
- Record: fraction with |cos| >= 0.90 per layer; global fraction; mean |cos|; median |cos|.
- Also record: for aligned neurons (|cos| >= 0.90), compute lambda_i = ||W_up[i,:]|| / ||W_gate[i,:]||.
```
Wall-clock: ~5 min (pure tensor arithmetic, no forward pass).

**Step 2 — Single-layer fold ΔNLL (~10 min, only if Step 1 GO):**
For layer 14 (standard probe layer), replace every W_up[14][i,:] where |cos[14][i]| ≥ 0.90 with lambda_i * W_gate[14][i,:]. Measure ΔNLL on 512-token WikiText-2 held-out vs unmodified baseline. Expected: ΔNLL < 0.01 nats for exactly-aligned; ΔNLL < 0.05 nats for near-aligned (cos ≥ 0.90).

**Sweep parameters**: No sweep. Single threshold (|cos| ≥ 0.90) is the primary gate; secondary at |cos| ≥ 0.70 for the near-aligned population.

**Output path**: `experiments/stage0/ladder_v2/round6_walign/`

**Wall-clock total**: ~5 min (Step 1 only; Step 2 adds ~10 min if GO).

**Script**: New script required — `scripts/walign_hist.py`. Inputs: GGUF model path, threshold (default 0.90). Outputs: per-layer histogram CSV, global fraction, lambda distribution per aligned neuron, histogram plot PNG.

## 5. Go Threshold

**GO**: ≥ 5% of all 172,032 neurons (≥ 8,601 neurons) have |cos(W_gate[i,:], W_up[i,:])| ≥ 0.90 globally.

Secondary GO condition (for reporting, not for advancement): at least one layer has ≥ 10% alignment fraction — indicating non-uniform depth distribution.

This threshold is grounded in residency arithmetic: 5% × 172,032 neurons × 2,048-dim row × 2B = ~34 MB saved from W_up on Qwen3-1.7B at ZERO quality cost for exactly-aligned neurons. At 10%: ~68 MB. At 70B (2.3M neurons): 5% alignment → ~920 MB saved from W_up at BF16; ~460 MB at INT4.

## 6. No-Go Threshold

**NO-GO**: < 1% of neurons with |cos| ≥ 0.90 AND mean |cos| < 0.10 across all neurons (distribution peaked near zero, consistent with SGD maximizing angular diversity between gate and up projections).

**Class-level structural finding on NO-GO:** Trained Qwen3 SwiGLU layers systematically maintain near-orthogonality between corresponding W_gate and W_up row pairs. The two-matrix degree of freedom in SwiGLU is used by gradient descent to encode independent information in both projections simultaneously (angular diversity maximization). The per-neuron alignment compression is unavailable post-training on Qwen3 family. This constrains a sub-class of algebraic-identity compression ideas: any scheme that tries to exploit gate/up collinearity (under any name) will find near-zero collinear fraction on Qwen3.

This is a structural finding about training dynamics regardless of the GO/NO-GO outcome — it directly characterizes how SwiGLU representation capacity is allocated in a trained model.

## 7. Ambiguous-Zone Follow-Up

**GRAY**: 1–5% with |cos| ≥ 0.90 OR mean |cos| ≈ 0.20–0.40 (moderate alignment throughout, not peaked alignment).

Follow-up: per-layer alignment profile (~10 min). Check if Layer 1 has a higher alignment fraction (consistent with CF6's Layer-1 anomaly — 36% gate near-constancy in Layer 1 may correlate with higher gate/up alignment in Layer 1 specifically). If Layer 1 has ≥ 10% alignment while other layers are < 1%, the WALIGN mechanism applies as a Layer-1-only micro-optimization (analogous to CF6's Layer-1 SDZC fold — small absolute saving but consistent with Layer-1's structural anomaly pattern).

This follow-up takes 10 min and produces a per-layer histogram. It does NOT require a Stage 6 amendment — it is a natural extension of the primary script's output.

## 8. Kill Criteria (Stage 6 Amendment Slot)

Stage 6 should reject this selection if:

1. **Prior art found**: A paper reporting per-neuron cosine(W_gate, W_up) distribution statistics in a trained SwiGLU model (at any depth or threshold) is found at Stage 6 that was missed at Stage 3 and Stage 5 re-verification. This would make M2's empirical distribution claim ADJACENT rather than NOVEL.

2. **CF9 frame-mismatch**: If any Stage 6 reviewer identifies a theoretical precondition in the algebraic identity (Section 3, H1) that fails for a reason not caught in Stage 3's frame-mismatch check. (Expected: none — the identity is a trigonometric fact with no external preconditions.)

3. **Skeptic-controls deficiency**: If Section 9's controls are incomplete or unjustified.

4. **Smallest test too large**: If the 5-min wall-clock estimate is wrong by more than 4× (unlikely; pure tensor arithmetic on 1.4 GB). If actual runtime > 8 hours, return to Stage 3 for a single-layer smoke version.

## 9. Skeptic-Controls Declaration

The hypothesis claims: **"SwiGLU neurons have measurable alignment structure in trained models"** — this is a "structure is present" claim that requires controls to distinguish learned structure from architectural artifact.

**Permutation control**: After computing per-row cosines on the trained model, recompute on W_gate[L] with rows randomly permuted within the layer (rows shuffled, but each row vector unchanged). The global cos distribution should be identical if alignment is a per-row property; but the per-neuron paired cosine (cos between row i of W_gate and row i of W_up) should drop toward the mean if the alignment is random. GO requires: fraction with |cos_real| ≥ 0.90 ≥ 0.20 above the fraction with |cos_permuted| ≥ 0.90. This rules out "W_gate rows happen to be similar to W_up rows in general" (which would persist under permutation).

**Random-init control**: Run the same histogram on a randomly initialized Qwen3-1.7B-Base (weights drawn from the initialization distribution, no training). For a random-init model, the per-neuron cosine distribution should be approximately uniform or Gaussian-near-zero (no training signal to produce alignment). GO requires: fraction with |cos_trained| ≥ 0.90 substantially (≥ 0.10 absolute) above fraction with |cos_random_init| ≥ 0.90. This rules out "the per-neuron cosine structure is an architectural property of the initialization, not a learned feature."

Implementation: Use `torch.manual_seed(42)` to generate random BF16 weights of the same shape as W_gate and W_up; compute the same histogram. This takes ~2 min additional (same computation, different weights).

**Multi-seed**: The measurement is deterministic (no randomness in the weight-arithmetic measurement itself). Multi-seed is not applicable here — the trained weights are fixed and the cosine computation is deterministic. Documented: "Multi-seed not applicable — measurement is deterministic weight arithmetic with no stochastic components."

For the Step 2 ΔNLL measurement (if reached on GO path): calibration corpus slice is drawn with seed 42; the measurement itself is deterministic (no dropout in inference). Multi-seed at ΔNLL step: sample 3 different 512-token WikiText-2 slices and report mean ± std ΔNLL. GO requires mean ΔNLL < 0.05 nats AND worst-of-3 < 0.10 nats.

## 10. Runtime Estimate

| Step | Description | Time |
|---|---|---|
| Setup | Load model weights (BF16, 1.7B) | ~30 s |
| Step 1 | Row-normalize + einsum cos for 28×6144 pairs | ~4 min |
| Step 1 | Permutation control | ~1 min |
| Step 1 | Random-init control | ~2 min |
| Histogram + logging | Write CSV, plot | ~30 s |
| **Step 2 (if GO)** | Single-layer fold + ΔNLL (512 tokens) | ~10 min |
| **Step 2** | Multi-seed ΔNLL (3 slices) | ~20 min |

**Total end-to-end (GO path)**: ~38 min. Step 1 only: ~8 min.
**Status**: well within 8-hour ceiling.

## 11. Script Identification

**New script required**: `scripts/walign_hist.py`

The script loads Qwen3-1.7B-Base BF16 via safetensors. For each layer, it computes row-normalized cosines between W_gate[L] and W_up[L] (einsum over the d_model dimension), bins the results into the 9 histogram bins above, and writes a per-layer CSV and a global summary. A permutation control recomputes cosines with W_gate rows permuted (random seed 42). A random-init control regenerates W_gate and W_up at the same shape from the model's init distribution and computes the same histogram. Outputs: `cosine_histogram_by_layer.csv`, `global_summary.json`, `permutation_control.json`, `random_init_control.json`, `histogram.png`. If `--run_fold` flag is set and any layer has fraction ≥ 0.05, proceeds to Step 2 (single-layer fold ΔNLL measurement on WikiText-2).

## 12. Downstream Implications

**GO (≥ 5% alignment):**
- Enables lossless per-neuron W_up row elimination for aligned neurons. Stackable on top of CF8 (MLP full-rank globally — WALIGN does not conflict with CF8 because it does not truncate; it folds).
- Composable with CF6 (Layer-1 has 36% gate near-constancy; if Layer-1 also has high alignment, the two findings compose for a stronger Layer-1 simplification).
- Directly feeds into F-orientation's next round: "what other per-neuron algebraic identities exist in SwiGLU layers?" (e.g., W_down row alignment with W_up output direction).
- Unlocks: F6-GQARED (which also exploits discrete algebraic structure — GQA group elimination). The two experiments are independent but thematically coherent for a Track A round.

**NO-GO (< 1% alignment):**
- Kills: any future proposal that hypothesizes "SwiGLU gate and up row vectors are approximately collinear in a measurable fraction of neurons in trained models." This is a class-level behavioral fact about Qwen3 gradient descent.
- Constrains: AERO (arXiv:2410.13060) claimed activation-removal-based fold works; WALIGN's NO-GO would reveal that the structural reason AERO requires training-time activation removal is precisely because post-training alignment is not achievable — a mechanistic explanation for AERO's training-time requirement.
- Does NOT kill: AERO itself (training-time fold is unaffected by post-training alignment statistics), or any CF8-grounded MLP quantization path (RAOK, PDAP extensions).

## 13. Provenance

- **Originating orientation**: F (First-Principles) — F6-WALIGN
- **Stage 2 score**: 12/15, Path 4 elegant-equivalence [algebraic-identity]
- **Stage 2 convergence cluster**: None (solo; no cross-orientation convergence)
- **Stage 3 verdict**: REFINE. Key insight: per-neuron granularity is structurally distinct from AERO's global fold; neither AERO nor 2405.01943 measures per-neuron cosine(W_gate, W_up).
- **Stage 4 note**: No Stage 4 amendment. WALIGN was not discussed in the Stage 4 gap-finder — it was Stage 3's top-5 Track A without requiring a Stage 4 reframe.
- **Frame-novelty path**: A2=2, A5=3 (NOVEL for per-neuron distribution), under-represented frame in run_006. Frame-novelty bonus +2.
- **Elegant-equivalence**: algebraic-identity, constructive (identity is exact at cos=1; near-exact at cos≥0.90). Multiplier 1.2.
- **Pre-emption check (2026-05-09)**: No paper found reporting per-neuron cos(W_gate[i,:], W_up[i,:]) distribution in a trained SwiGLU network.
- **Runner-up**: C-JOINT-DELNLL (score 18.0). If Stage 6 kills WALIGN, advance C-JOINT-DELNLL using Stage 3 section A's plan (12-min experiment, script `round6_joint_delnll/run_joint_qk.py`).

---

---

# Experiment Plan B: C-WVOK

## 1. Title and One-Line Description

**Run 006 / Track B — C-WVOK: W_V Subspace Concentration from CF11 Head-Redundancy Coupling**

Measure the spectrum of W_V at layer 14 in Qwen3-1.7B-Base; if r_99(W_V)/d < 0.80 and ΔNLL(K_V=512) < 0.40 nats, CF11's head-redundancy extends to the value projection, unlocking W_V compression and gating the full four-matrix attention compression cascade to 70B.

## 2. Class Tags

`compression-lr`, `attention-weight`, `convergence-gate`, `cf11-extension`

## 3. Hypothesis

**H1 (coupling equation from CF11):** CF11 established that 16 query heads span a ~128-dim subspace (head-redundancy). The value-routing argument: value projection W_V produces output v = W_V · h for each head's KV group; if attention weights are dominated by ~128 effective query directions (CF11), then the effective value outputs accessed per token lie in the span of at most 128 token-positions × W_V, which compresses under SVD if W_V itself has concentrated spectrum. The specific quantitative prediction: dim(effective_value_subspace) ≤ 128 (coupling from CF11). This predicts r_99(W_V)/d < 0.80 (less than MLP matrices, which are at ~1.0; more compressed than W_K's 0.79 per CF11 measurement). Cite: CF11 (AQFKV result, run_006's ladder finding).

**H2 (spectral uniformity prior):** arXiv:2604.22778 (Spectral Lifecycle, April 2026) observes that V/O projections develop more uniform, concentrated spectra during training compared to Q/K projections. This independently supports H1's prediction for Qwen3 post-training. The 2604.22778 measurement is on GPT-2/Pythia during training; the Qwen3 post-training measurement is the experiment.

**H3 (ΔNLL cascade):** If H1 holds (W_V concentrated), then W_V at K=512 (50% rank truncation) achieves ΔNLL < 0.40 nats — better than W_K at K=512 (+0.29 nats) because the coupling predicts W_V is more concentrated than W_K (the CF11 head-redundancy applies to the value channel's effective output subspace, which is further collapsed by the attention weight concentration).

This is a Convergence C2 representative: three orientations (R6-WVOP, C-WVOK, R6-ATTNQ-FULL) independently converge on the W_V spectrum as the measurement gate for the 70B attention compression target. Running C-WVOK once resolves all three proposals.

## 4. Smallest Test

**Model**: Qwen3-1.7B-Base BF16 (`Qwen/Qwen3-1.7B-Base`), full-precision weights, CPU-loaded.

**Calibration corpus**: None for Step 1 (spectrum measurement). WikiText-2 512-token held-out for Step 2 (ΔNLL measurement).

**Eval corpus**: WikiText-2 held-out (same 512-token slice used in all run_006 experiments for comparability with CF11 AQFKV numbers).

**Matrices touched**: W_V[14] (shape 1024×2048, GQA, n_kv_heads=8, d_head=128, so shape is n_kv_heads × d_head × d_model = 8×128×2048 reshaped to 1024×2048). W_O[14] optionally as a bonus measurement (shape 2048×2048). Explicit indices: layer 14 (standard mid-depth probe).

**Step 1 — Spectrum measurement (~5 min):**
```
Script: experiments/stage0/ladder_v2/round6_cwvok/run_wv_spectrum.py
- Load W_V[14] (1024×2048, BF16). Cast to float32 for SVD stability.
- torch.linalg.svd(W_V[14], full_matrices=False) → singular values s (shape 1024,).
- Compute:
  - r_99: smallest r such that sum(s[:r]**2) / sum(s**2) >= 0.99.
  - var@128: sum(s[:128]**2) / sum(s**2)
  - var@256: sum(s[:256]**2) / sum(s**2)
  - var@512: sum(s[:512]**2) / sum(s**2)
- Bonus: per-head SVD for one GQA head (W_V[14][0:128, :], shape 128×2048) to check
  per-head vs full-matrix behavior.
- Bonus: W_O[14] SVD (2048×2048) for free (same infrastructure). Record r_99(W_O), var@{256,512,1024}.
```

**Step 2 — ΔNLL measurement (~15 min, only if Step 1 confirms r_99 < 0.90):**
```
- Replace W_V[14] with its rank-K_V SVD reconstruction at K_V ∈ {256, 512}.
- Hold all other layers' W_V at BF16; hold W_Q, W_K, W_O at BF16.
- Run forward pass on WikiText-2 512-token held-out.
- Record ΔNLL at K_V=512 (primary gate) and K_V=256 (secondary, aggressive).
```

**Wall-clock**: Step 1: ~5 min. Step 2: ~15 min. Total GO path: ~20 min.

**Output path**: `experiments/stage0/ladder_v2/round6_cwvok/`

**Script**: New script required — `scripts/wv_spectrum.py`. Inputs: GGUF model path, layer index (default 14), eval corpus path. Outputs: SVD spectrum CSV, var@K values, per-head SVD summary, optional W_O spectrum, optional ΔNLL at K ∈ {256, 512}.

## 5. Go Threshold

**GO**: r_99(W_V[14])/d < 0.80 AND ΔNLL(K_V=512) < 0.40 nats.

This is numerically grounded: W_K achieved r_99/d = 0.79 and ΔNLL(K_K=512) = +0.29 nats (CF11 AQFKV data). H1 predicts W_V should be MORE concentrated than W_K (the value-routing argument couples W_V concentration to the CF11 head-redundancy). GO threshold is conservative: set at W_K's observed boundary (0.79) with ΔNLL tolerance slightly higher than W_K (0.40 vs 0.29 to account for model-specific variance).

**GO-strong**: r_99 < 0.70 AND ΔNLL(K_V=512) < 0.20 nats. Proceed directly to full 28-layer W_V compression + W_O spectrum + cascade planning for R6-ATTNQ-FULL.

## 6. No-Go Threshold

**NO-GO**: r_99(W_V[14])/d > 0.90 (W_V approaches full-rank like MLP matrices at ~1.0).

**Class-level structural finding on NO-GO:** CF11's head-redundancy is a query-side phenomenon only. Value projection W_V encodes content-specific routing that is independent of the attention weight subspace concentration. The 70B attention compression target falls back to W_Q + W_K only (~2.04 GB saving at 70B from CF11 data vs the ~11.24 GB potential with W_VO included). All three Convergence C2 proposals (R6-WVOP, C-WVOK, R6-ATTNQ-FULL) are simultaneously demoted or killed. The 70B DRAM-residency target becomes harder to achieve via attention compression alone; focus shifts to substrate path (Track B S1 FNDP-WMCE or RAOK quantization).

This is the most information-dense NO-GO in the run: resolves three simultaneously-open questions with one 5-minute SVD.

## 7. Ambiguous-Zone Follow-Up

**GRAY**: 0.80 ≤ r_99(W_V) ≤ 0.90 AND ΔNLL(K_V=512) between 0.40 and 1.00 nats.

Follow-up (30 min): Measure ΔNLL at K_V=1024 (conservative 50% of 1024-dim GQA W_V). If ΔNLL(K_V=1024) < 0.20 nats, the knee of acceptable compression is at K_V≈1024 rather than K_V=512; smaller savings (25%) but still meaningful for 70B. Compute: W_V residency saving at K_V=1024 vs K_V=512 at 70B.

Also: measure W_O spectrum as bonus in the same Gray-resolution experiment (W_O may be more concentrated than W_V even if W_V is moderate, providing an alternative V-O fold entry point).

The Gray follow-up takes 30 min (ΔNLL at K_V=1024 + W_O spectrum) and does NOT require a Stage 6 amendment.

## 8. Kill Criteria (Stage 6 Amendment Slot)

Stage 6 should reject this selection if:

1. **A3 prior art conflict**: arXiv:2505.12942 (A3) explicitly reports W_V spectrum measurements on Qwen3 family (not just the W_VO fold). If A3 includes Qwen3-specific r_99(W_V) numbers that directly answer H1, the experiment becomes a replication rather than a novel measurement. Stage 6 must check A3's appendix/tables for W_V spectral data on Qwen3.

2. **GQA shape error**: The W_V shape in Qwen3-1.7B-Base with GQA (n_kv_heads=8, d_head=128) is 1024×2048, NOT 2048×2048. If the script uses the wrong shape (e.g., loading the weight as a full 2048×2048 matrix by mistake), the SVD result is meaningless. Stage 6 should verify the GQA shape in the script's weight loading code.

3. **Per-head vs full-matrix discrepancy**: If the Stage 6 reviewer notes that the coupling argument (H1) operates at the per-head level (each GQA KV group's W_V^g is 128×2048), but the experiment tests the full concatenated 1024×2048 matrix, there is a structural mismatch. The full-matrix SVD captures cross-head structure; the per-head SVD tests the actual compression unit. If the Stage 6 review identifies this as a load-bearing distinction, escalate to per-head SVD (5 additional min).

4. **Missing skeptic-controls declaration** (see Section 9).

## 9. Skeptic-Controls Declaration

The hypothesis claims: **"W_V has concentrated spectrum (trained model has structural property that differs from untrained)"** — a "structure exists / X works on this stack" claim requiring controls.

**Permutation control**: After computing the singular values of the trained W_V[14], recompute the SVD on W_V[14] with rows randomly permuted (random seed 42). For a permuted matrix, the singular value distribution should be essentially unchanged (row permutation preserves the Frobenius norm and the set of singular values — permutation does NOT change singular values, only singular vectors). Therefore permutation control is NOT the right control for SVD spectrum. The correct control for "W_V spectrum is concentrated" is the random-init control (see below). Permutation control is documented as not applicable to SVD spectrum measurement: "Permuting rows of W_V does not change the singular value distribution; the permutation control is geometrically equivalent to the original measurement. Not applicable to spectrum concentration claims."

**Random-init control**: Run the same SVD on a randomly initialized W_V[14] (same shape 1024×2048, weights from Qwen3's initialization distribution, `torch.manual_seed(42)`). For a random matrix of shape m×n with m<n, the singular value distribution follows the Marchenko-Pastur law, giving a relatively flat distribution. GO requires: r_99(W_V_trained)/d substantially less than r_99(W_V_random_init)/d — specifically, trained r_99 < 0.80 AND random-init r_99 ≥ 0.90. This distinguishes "W_V is concentrated because it is trained" from "1024×2048 matrices are inherently concentrated by dimension ratio." The Marchenko-Pastur bulk edge for aspect ratio 1024/2048 = 0.5 predicts r_99 ≈ 0.95–0.98 for random matrices, so a trained r_99 < 0.80 would be a clear separation.

**Multi-seed**: The spectrum measurement is deterministic (no stochastic component). Multi-seed is not applicable. Documented: "SVD computation on BF16 weight tensors is deterministic; no seed-varying component exists. Multi-seed not applicable."

For the ΔNLL Step 2 (if reached): calibration slice is deterministic (fixed WikiText-2 token slice). Report ΔNLL on 3 different 512-token WikiText-2 slices (seeds 0, 1, 2) as mean ± std. GO requires mean ΔNLL(K_V=512) < 0.40 nats AND worst-of-3 < 0.60 nats.

## 10. Runtime Estimate

| Step | Description | Time |
|---|---|---|
| Setup | Load model weights (BF16, 1.7B) to CPU | ~30 s |
| Step 1 | SVD of W_V[14] (1024×2048, float32) | ~2 min |
| Step 1 | Per-head bonus SVD (128×2048) | ~30 s |
| Step 1 | W_O[14] bonus SVD (2048×2048) | ~3 min |
| Step 1 | Random-init control SVD (same shapes) | ~3 min |
| Logging | Write CSVs + summary | ~30 s |
| **Step 2 (if GO)** | ΔNLL at K_V=256 and K_V=512 (512 tokens) | ~12 min |
| **Step 2** | ΔNLL multi-seed (3 slices × 2 K values) | ~20 min |

**Total end-to-end (GO path, including bonus W_O measurement)**: ~41 min.
**Step 1 only (spectrum gate)**: ~9 min.
**Status**: well within 8-hour ceiling.

## 11. Script Identification

**New script required**: `scripts/wv_spectrum.py`

The script loads Qwen3-1.7B-Base BF16 weights via safetensors. For the specified layer (default 14), it computes SVD of W_V[L] (GQA-shaped 1024×2048), records r_99/d, var@{128,256,512}, and singular value decay curve. Bonus: per-head SVD for GQA head 0 (128×2048). Bonus: W_O[L] SVD (2048×2048). Random-init control: regenerates W_V and W_O at same shapes from initialization distribution (seed 42) and runs the same SVD. If `--run_delta_nll` flag and r_99 < 0.90, proceeds to Step 2: reconstructs W_V at K ∈ {256, 512}, runs forward pass on WikiText-2 slices, records ΔNLL. Outputs: `wv_spectrum.csv`, `wo_spectrum.csv`, `random_init_control.csv`, `delta_nll_results.json` (if Step 2 run), `spectrum_plot.png`.

## 12. Downstream Implications

**GO (r_99 < 0.80):**
- Gates and advances R6-ATTNQ-FULL (four-matrix attention compression cascade). The 70B combined attention residency target (~1.17 GB after W_Q + W_K + W_V + W_O at respective K values) becomes achievable. With NanoQuant MLP (~5.35 GB), total 70B model ~6.5 GB — fits in 7.28 GiB with margin.
- R6-WVOP (W_VO fold) gets the Qwen3-specific W_V spectrum anchor it needed for the product-rank argument. Even though the W_VO fold as a compression application is A3-adjacent, the Frobenius-SVD product rank comparison (WVOP's surviving novelty claim) is now grounded.
- Directly enables the next Track B round: full 28-layer W_V + W_O compression sweep with go/no-go per layer.
- Structural finding: CF11 head-redundancy extends to the value projection → the attention mechanism's compression budget is bilateral (Q+K AND V+O), not unilateral.

**NO-GO (r_99 > 0.90):**
- Class-level kill: Convergence C2 collapses. R6-WVOP, C-WVOK, R6-ATTNQ-FULL all demoted or killed simultaneously. The attention compression ceiling is W_Q + W_K only.
- 70B plan: falls back to CF11's W_Q(K=128) + W_K(K=512) = ~2.04 GB saving from ~12.08 GB Q+K pair. With NanoQuant ~5.35 GB, total ~10.29 GB — does NOT fit in 7.28 GiB DRAM. 70B remains NVMe-resident; substrate path (FNDP-WMCE composition) becomes the primary Track B direction.
- Enriches CF11: "head-redundancy is query-side only" is a precise structural delineation of what CF11's finding means architecturally.

## 13. Provenance

- **Originating orientation**: C (Composition) — C-WVOK, coupling CF11 × W_V subspace
- **Stage 2 score**: 12/15, Path 1 standard advancer
- **Stage 2 convergence cluster**: C2 (3-orientation: R6-WVOP, C-WVOK, R6-ATTNQ-FULL). Strongest convergence cluster in run_006. Convergence multiplier 1.5.
- **Stage 3 verdict**: REFINE. "Cheapest experiment in the run (20 min). Gates three proposals. 2604.22778 provides supportive prior." Stage 3 section E confirmed M1 NOVEL, M2/M3 NOVEL for Qwen3-specific numbers.
- **Stage 4 note**: Stage 4 (S7) flagged A3 pre-emption for R6-WVOP's engineering application. C-WVOK's coupling-equation novelty (M1: CF11 derives W_V rank bound) is intact — A3 does not derive W_V rank from CF11 head-redundancy.
- **Frame-novelty path**: A2=2, novel coupling equation (CF11 → W_V rank). Frame-novelty bonus +1 (novel but not explicitly under-represented in framing targets).
- **Convergence bonus**: +1.5× (3-orientation C2 cluster).
- **NO-GO-finding bonus**: +2 (class-level kill on C2 all three proposals; sharpens CF11 scope; candidate CF entry for "head-redundancy is query-side" finding).
- **Pre-emption check (2026-05-09)**: A3 (arXiv:2505.12942) performs W_VO fold + activation-aware SVD. Stage 6 must verify A3 does NOT report Qwen3-specific r_99(W_V) numbers in its paper body or appendix. If found, Step 1's spectrum measurement becomes a replication and the coupling-equation framing (M1) remains novel but the empirical measurement does not.
- **Runner-up**: C-DEEP-SPREAD-QBUD (score 15.2). If Stage 6 kills C-WVOK (e.g., A3 reports W_V spectrum numbers), advance C-DEEP-SPREAD-QBUD using Stage 3 section I's plan (35-min Spearman ρ experiment, script `round6_qbud/run_perlay_wq_rank.py`).
