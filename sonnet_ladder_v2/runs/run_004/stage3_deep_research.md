# Stage 3 — Deep Research — Run 004
Date: 2026-05-09

Inputs: stage1_{R,C,F,U,A}.md, stage2_judge.md, SUMMARY.md (CF1-CF12, v2-CF1, v2-CF2), KILL_LIST.md, live WebSearch.
Advancer count entering Stage 3: 24 (Path 1: 12 Track-B + 5 Track-A; convergence reps: 7; frame-novelty: 2; wildcards: 3).

---

## PATH 1 — Track B Weight-Compression Proposals

---

### AWCJR — Attention-Weight Cascade: W_Q+W_K+W_V+W_O Joint

#### 1. Load-bearing claims

M1: W_V[L] r_99/d < 0.80 in Qwen3-1.7B (spectrum concentrated like W_K, not like W_gate/W_up).
M2: W_O[L] r_99/d < 0.80 (concentrated, not full-rank output projection).
M3: Joint ΔNLL of W_Q@K=128 + W_K@K=512 + W_V@K=256 + W_O@K=256 < 1.5 nats (sub-additive composition).
M4: Compounding residency savings are arithmetic (no weight-matrix interaction effects).

#### 2. Per-claim prior-art status

**M1 (W_V spectrum):** PARTIAL. The Spectral Lifecycle paper (arXiv:2604.22778, Apr 2026) finds "value/output projections compress uniformly" — a strong supporting signal that W_V is spectral-compressible. SVD-LLM (arXiv:2403.07378, ICLR 2025) applies SVD to attention weight matrices including W_V in practice. LatentLLM (MERL TR2026-018) performs joint V+O decomposition. However, none measure W_V r_99/d specifically on Qwen3-1.7B — the private measurement remains NOVEL as of 2026-05-09.
elegance-class: `subspace-alignment` (concentrated spectrum enabling low-rank fold).

**M2 (W_O spectrum):** PARTIAL. Same spectral-lifecycle paper confirms V/O compress uniformly. MHA2MLA (arXiv:2502.14837) performs joint KV decomposition post-training with fine-tuning; LatentLLM does joint V+O zero-shot. Neither reports Qwen3-1.7B W_O r_99/d. NOVEL measurement.

**M3 (sub-additive composition):** NOVEL. No published work measures interaction coefficient of simultaneous 4-matrix SVD truncation on Qwen3 family. SVD-LLM V2 (NAACL 2025) does global rank allocation but doesn't report additive/sub-additive interaction structure. As of 2026-05-09: no paper found doing joint W_Q+W_K+W_V+W_O simultaneous truncation NLL measurement on SwiGLU decoder model.

**M4 (arithmetic compounding):** ADJACENT. SVD-LLM V2's global allocation shows truncation effects can compound non-trivially; ASVD (arXiv:2312.05821) shows activation-aware weighting changes interaction. Pipeline assumption of additive needs empirical verification.

#### 3. Frame-mismatch check (CF9)

No imported theorem. Pure SVD truncation + arithmetic. CF9 clean.

#### 4. Calibration ill-conditioning (CF10)

No calibration fit. SVD is computed analytically. CF10 not triggered.

#### 5. Residency arithmetic

Per Stage 1 arithmetic, confirmed: W_V+W_O savings at K=256 each: 448 MB on 1.7B. Combined with W_Q@K=128 (196 MB) + W_K@K=512 (~56 MB) = ~700 MB from 896 MB total attention weights. Conditional on M1/M2 holding. CF13-15 not cited.

#### 6. Smallest-test sharpening

Script: `scripts/awcjr_wv_wo_spectrum.py`
Model: Qwen3-1.7B-Base, bf16, transformers tokenizer.
Step 1: SVD of W_V[all 28 layers] + W_O[all 28 layers], record r_99/d and var@K for K ∈ {64,128,256,512}. ~25 min.
Step 2 (conditional on M1/M2 GO): construct rank-256 folds for W_V, W_O; measure NLL on 512-token WikiText-2. ~15 min.
Output: `experiments/stage0/ladder_v2/round4_awcjr/`
Wall-clock: 40 min total. **Within 8h gate.**

#### 7. Go/no-go thresholds

GO: mean W_V r_99/d ≤ 0.80 AND mean W_O r_99/d ≤ 0.80 across 28 layers → proceed to NLL sweep.
NLL GO: joint W_Q+W_K+W_V+W_O ΔNLL ≤ 1.5 nats.
NO-GO: W_V or W_O r_99/d > 0.90 → CF8 extends partially into attention; kill joint cascade, keep per-matrix where concentrated.
GRAY: r_99/d in 0.80–0.90 → ACTIVQ analog: try activation-weighted SVD (weight by post-attention activation magnitude) as mitigation.

#### 8. Risk profile

R1: W_V or W_O full-rank. Cheapest falsifier: 5-min SVD of W_V[L14] r_99 check.
R2: Superadditive ΔNLL from joint truncation. Mitigation: apply cascades sequentially (W_Q first, then others on top) to isolate interaction.
R3: LatentLLM/MHA2MLA pre-emption extends to W_V fold (they do V+O joint, zero-shot). Differentiation: they use fine-tuning or training-time conversion; this is pure SVD-based post-training without ANY fine-tuning on Qwen3-specific spectrum measurement.

#### 9. Two-paper composition flag

Closest pair: SVD-LLM (ICLR 2025, W_V/W_O truncation) + Spectral Lifecycle (arXiv:2604.22778, V/O compress uniformly).
Value-add: Qwen3-specific empirical measurement + 4-matrix simultaneous NLL interaction coefficient + CF11-calibrated threshold selection. The Spectral Lifecycle paper validates W_V/W_O compressibility in general; AWCJR provides Qwen3 quantification and the interaction structure.

**PARTIAL pre-emption alert:** LatentLLM (MERL TR2026-018, zero-shot V+O joint decomposition) is structurally close to M1+M2. If LatentLLM includes Qwen3-class benchmarks showing no retraining needed for V+O fold, M1+M2 are ADJACENT (different loss function: LatentLLM optimizes activation-aware tensor decomposition vs pure SVD). Need to verify LatentLLM's Qwen3 coverage. For now: PARTIAL.

#### 10. Verdict

**REFINE** — M1/M2 gate is cheap (25 min), either outcome is structurally valuable, M3 is novel. Conditional advance to Stage 4/5 after W_V[L14] binary 5-min check. LatentLLM adjacency does not pre-empt because LatentLLM targets KV-cache compression via training, not pure weight-rank fold quality measurement on Qwen3.

---

### RATJ — RAOK + Attention-Tier Joint Cascade

#### 1. Load-bearing claims

M1: RAOK 3-tier activation codebook on W_up outputs is viable (grounded in CF3 K-dependent Jaccard; independently tested in pipeline).
M2: CF11 W_Q K=128 ΔNLL (+0.98 nats) is stable under RAOK's activation modification (interaction ΔNLL < 1.5 nats).
M3: Joint cascade produces sub-additive (or at most additive) combined ΔNLL.

#### 2. Per-claim prior-art status

**M1 (RAOK viability):** PARTIAL. RAOK is internally developed and grounded in CF3 — private Qwen3-specific numbers. Closest published: PrefixQuant (arXiv:2410.05265) decomposes static/token-wise outliers. SmoothQuant addresses static. RAOK's CF3-calibrated 3-tier (K=0.1% FP16/K=1% INT8/rest INT4) is NOVEL in its grounding.

**M2 (interaction stability):** NOVEL. No published paper measures the interaction coefficient between activation-tier outlier handling and attention weight SVD truncation on the same model. As of 2026-05-09: no paper found doing this on SwiGLU decoder model.

**M3 (sub-additive composition):** NOVEL. Same as M3 in AWCJR.

#### 3. Frame-mismatch check (CF9)

No imported theorem. Empirical composition of two CF-grounded primitives. CF9 clean.

#### 4. Calibration ill-conditioning (CF10)

RAOK's 3-tier codebook: centroids are fixed by calibration Jaccard thresholds (threshold lookup, not a fit). n_params: 3 tier boundaries per layer = trivial. CF10 not triggered.

#### 5. Residency arithmetic

Combined: RAOK MLP at ~2.2 bpw (13 GB for 70B) + CF11 attention at ~5 GB. Conditional on RAOK viability (still untested). Arithmetic is sound conditional on both being independently confirmed.

#### 6. Smallest-test sharpening

Script: `scripts/ratj_interaction_check.py`
Model: Qwen3-1.7B-Base.
Step 1: RAOK-only NLL (3-tier activation on W_up output). ~20 min.
Step 2: W_Q-K128-only NLL (already in AQFKV, reuse). ~0 min.
Step 3: Joint NLL. ~20 min.
Output: `experiments/stage0/ladder_v2/round4_ratj/`
Wall-clock: 45 min. **Within 8h gate.**

#### 7. Go/no-go thresholds

GO: joint ΔNLL ≤ RAOK-only + W_Q-only (sub-additive or additive).
NO-GO: joint ΔNLL > sum by > 0.3 nats (destructive interaction from activation distortion + low-rank Q subspace mismatch).
GRAY: interaction within 0.3 nats → decompose per-layer; identify which layers show destructive interaction.

#### 8. Risk profile

R1: RAOK not independently validated yet (deferred). Gate RATJ on RAOK viability first.
R2: Activation distribution entering W_Q changes under RAOK → W_Q SVD computed on original distribution becomes suboptimal. Mitigation: re-run AQFKV after RAOK activation modification to verify subspace stability.
R3: Cheapest falsifier: joint NLL measurement (45 min).

#### 9. Two-paper composition flag

Closest pair: PrefixQuant (activation outlier stratification) + SVD-LLM (attention weight SVD).
Value-add: CF3 × CF11 grounding is Qwen3-specific; the interaction coefficient between the two mechanisms is not measured in either paper or their composition; the private empirical moat makes pre-emption harder.

#### 10. Verdict

**REFINE** — but sequence-gated: run RAOK viability first, then RATJ interaction check. RATJ's Stage 5 value is the composition structure, not independent novelty.

---

### ULHC — Untied lm_head Spectrum + SVLD Revival

#### 1. Load-bearing claims

M1: Qwen3-8B has `tie_word_embeddings=False` (verified from config).
M2: Untied lm_head 151936×4096 has r_99/d < 0.40 (Godey & Artzi prediction applies).
M3: Rank-256 truncation gives ΔNLL < 0.30 nats.
M4: lm_head.weight (1.24 GB) can be loaded into 7.28 GiB RAM for standalone SVD.

#### 2. Per-claim prior-art status

**M1 (config verification):** NOVEL. Factual check, not a research claim.

**M2 (untied spectrum concentration):** ADJACENT. Godey & Artzi (arXiv:2510.24966) predicts this theoretically. SVD-LLM / LHQD work confirms tied configuration is full-rank (CF12). No paper reports Qwen3-8B untied lm_head r_99/d measurement. NOVEL measurement.

**M3 (ΔNLL at r=256):** NOVEL. No published result on Qwen3-8B untied lm_head quality at SVD truncation.

**M4 (RAM feasibility):** NOVEL. Verifiable from model config.

#### 3. Frame-mismatch check (CF9)

Godey & Artzi theorem: predicts concentrated spectrum in untied lm_head due to gradient bottleneck (single gradient path). Precondition: untied configuration (verified for Qwen3-8B). CF9 clean — the precondition is met unlike CF12's tied configuration where the theorem failed.

#### 4. Calibration ill-conditioning (CF10)

No calibration fit. Pure SVD. CF10 not triggered.

#### 5. Residency arithmetic

lm_head at bf16: 1.24 GB. At r=256: ~79 MB. 15.6× reduction on lm_head. Model impact: 15.2 GB → 14.0 GB. Modest. The strategic value is closing CF12 boundary to all configurations and enabling SVLD revival.

#### 6. Smallest-test sharpening

Script: `scripts/ulhc_lmhead_spectrum.py` (extends existing `scripts/lhqd_lmhead_spectrum.py`)
Model: Qwen3-8B, bf16. Load lm_head.weight only (1.24 GB).
Blockwise SVD: 37 blocks of 4096×4096. ~45 min.
NLL sweep at r ∈ {64,128,256,512,1024}: ~15 min on WikiText-2 512 tokens.
Output: `experiments/stage0/ladder_v2/round4_ulhc/`
Wall-clock: 60 min. **Within 8h gate.**
Note: requires Qwen3-8B download (15.2 GB). Verify model availability before scheduling.

#### 7. Go/no-go thresholds

GO: r_99/d < 0.40 AND ΔNLL@r=256 < 0.30 nats → SVLD revival + streaming SVD deployment.
NO-GO: r_99/d > 0.80 → Godey & Artzi prediction fails for Qwen3-8B; CF12 extends to untied; permanent kill of lm_head SVD line.
GRAY: r_99/d in 0.40–0.80 → ΔNLL sweep determines acceptable r; streaming SVD may still be viable at higher r.

#### 8. Risk profile

R1: Qwen3-8B may not be downloadable on this system (15.2 GB model, 7.28 GiB RAM). Mitigation: use `--load-in-8bit` or stream lm_head only without loading full model.
R2: Godey & Artzi prediction may fail due to architectural differences (GQA in 8B, different gradient dynamics). Mitigation: the measurement settles this directly.
Cheapest falsifier: check Qwen3-8B model config for tie_word_embeddings first (1 min).

#### 9. Two-paper composition flag

Closest pair: Godey & Artzi (arXiv:2510.24966, lm_head spectrum prediction) + CF12 (tied lm_head full-rank confirmation, this pipeline).
Value-add: First empirical test of Godey & Artzi on Qwen3-class SwiGLU decoder. The prediction is clean; this proposal operationalizes it. Not ENGINEERING-INTEGRATION because the prediction hasn't been tested on this architecture.

#### 10. Verdict

**REFINE** — clean proposal, cheap test, decisive either way. Model availability is the only operational risk.

---

### WDRS / F-WNORM — W_down Rank Sweep (CF8 Closure)

(Treated as a single entry; F-WNORM is the canonical representative per Stage 2.)

#### 1. Load-bearing claims

M1: W_down in Qwen3-1.7B has r_99/d_out > 0.90 (full-rank per CF8 prediction, expected outcome).
M2: The d_ffn > d_model shape (6144→2048) means maximum rank is d_model=2048, not d_ffn — the "r_99/d_out" framing is structurally different from W_gate/W_up.
M3 (WDRS secondary): Activation-weighted SVD of W_down (using CF3's 3-tier structure as weighting kernel) gives different effective rank than Frobenius-norm SVD.

#### 2. Per-claim prior-art status

**M1 (W_down full-rank):** PARTIAL. CF8 predicts it; ASVD (arXiv:2312.05821) applies activation-weighted SVD to all weight matrices including W_down but doesn't report Qwen3 specifically. Stage 2 notes "W_down almost certainly full-rank, untested." NOVEL measurement on Qwen3. The Spectral Lifecycle paper (arXiv:2604.22778) does NOT address MLP W_down specifically (focuses on Q/K/V/O and training dynamics on smaller models).

**M2 (d_ffn > d_model structural framing):** NOVEL framing. No paper explicitly calls out this asymmetric rank bound as a distinct structural question. The fact that W_down's max rank is d_out=2048 (not d_in=6144) means "full-rank" is different here.

**M3 (activation-weighted SVD vs Frobenius, WDRS):** ADJACENT. ASVD (2023) and SVD-LLM (2025) both implement activation-weighted SVD. GPTVQ (arXiv:2402.15319) uses Hessian = X^TX = activation second moment for codebook fitting — the activation-weighted SVD with CF3's 3-tier structure is close to GPTVQ's Hessian weighting but uses a different kernel (CF3's outlier tier structure). Stage 2 correctly identified this ADJACENT relationship. M3 is NOT fully novel — ACTIVQ is engineering differentiation over ASVD/GPTVQ, not a structural claim.

#### 3. Frame-mismatch check (CF9)

No imported theorem for M1/M2. M3 (ACTIVQ variant): the CF3 3-tier weighting is an activation-statistics-derived weight, same class as ASVD/GPTVQ. Precondition: CF3's tier structure must correspond to functionally different activation groups. CF9 concern: CF3 measures RESIDUAL STREAM outliers (W_up input channels), but ACTIVQ weights W_down SVD by POST-SwiGLU activation (W_down input channels, different object). This is a mild CF9 concern — the tier structure from residual stream outliers may not directly apply to post-SwiGLU activation distribution. Mitigation: use post-SwiGLU activation magnitudes directly (not CF3 residual-stream tiers) for the weighting kernel. ASVD-style activation-weighted SVD with the actual W_down input distribution.

#### 4. Calibration ill-conditioning (CF10)

No calibration fit for M1/M2. M3 ACTIVQ: activation-weighted SVD is analytically computed from X^TX; not a fitted regression. CF10 not triggered.

#### 5. Residency arithmetic

If W_down full-rank (expected): no compression win from rank reduction. Pipeline value: closes CF8 to all three MLP matrices. If W_down concentrated (surprise at r=256): 555 MB saving on 1.7B.

#### 6. Smallest-test sharpening

Script: `scripts/fwnorm_wdown_spectrum.py`
Model: Qwen3-1.7B-Base, bf16.
SVD of all 28 W_down layers (2048×6144 each). ~20 min.
NLL sweep at K ∈ {128,256,512,1024,2048} on WikiText-2 512 tokens. ~15 min.
Optional M3: activation-weighted variant (reuse PDAP activation stats). ~15 min additional.
Output: `experiments/stage0/ladder_v2/round4_fwnorm/`
Wall-clock: 35 min (50 min with M3). **Within 8h gate.**

#### 7. Go/no-go thresholds

GO (expected null): var@256 ≥ 0.99 in ≥ 20/28 layers → CF8 now covers all MLP matrices. Pipeline kill-list entry.
GO (surprise): var@256 ≤ 0.85 in ≥ 15/28 layers → W_down compression path opens.
NO-GO: doesn't apply here — any result is structurally informative.
GRAY: var@256 in 0.85–0.99 → per-layer adaptive rank sweep.

#### 8. Risk profile

R1: Result confirms CF8 (highest probability). This is structurally useful closure, not a compression win.
R2: M3 ACTIVQ is adjacent to ASVD/GPTVQ — limit M3 to a secondary diagnostic, not a primary contribution.
Cheapest falsifier: W_down[L14] SVD (3 min) gives the go/no-go directional signal.

#### 9. Two-paper composition flag

Closest pair: ASVD (activation-aware SVD for MLP) + CF8 (Qwen3 MLP full-rank, this pipeline).
Value-add: Qwen3-specific closure of CF8 for W_down, including the asymmetric d_ffn>d_model structural observation. The d_out-relative rank framing is a novel observation.

#### 10. Verdict

**REFINE** — pure structural closure experiment, high scientific value regardless of outcome, 35 min runtime. M3 (ACTIVQ) should be labeled as ADJACENT to ASVD/GPTVQ in the experiment plan.

---

### L1AW — Layer-1 SDZC + Global Attention Compression

#### 1. Load-bearing claims

M1: W_Q spectrum at layer 1 has r_99/d materially lower than L14 (CF11 reference: 0.63), specifically < 0.50.
M2: CF6's layer-1 MLP softness (36% foldable gates) and attention-side softness are coupled (same architectural specialization).

#### 2. Per-claim prior-art status

**M1:** PARTIAL. AQFKV measured L14; per-layer variation is unpublished on Qwen3. The "layer-1 anomaly extends to attention" is NOVEL empirically. The general principle of per-layer sensitivity variation is well-known (SVD-LLM V2 does non-uniform rank allocation based on sensitivity). The Qwen3-specific coupling of CF6 (MLP softness) × W_Q spectrum is NOVEL.

**M2:** NOVEL. No paper couples MLP gate variance at layer 1 with attention spectrum concentration at the same layer. As of 2026-05-09: no paper found doing this cross-modality coupling.

#### 3. Frame-mismatch check (CF9)

No imported theorem. CF9 clean.

#### 4. Calibration ill-conditioning (CF10)

No calibration fit. CF10 clean.

#### 5. Smallest-test sharpening

Script: extend `scripts/awcjr_wv_wo_spectrum.py` to include per-layer W_Q r_99 sweep.
Layer-1 W_Q SVD: 5 min. Decision gate before full per-layer sweep.
Output: `experiments/stage0/ladder_v2/round4_l1aw/`
Wall-clock: 5 min gate + 30 min full sweep. **Within 8h gate.**

#### 6. Go/no-go thresholds

GO: r_99/d(W_Q[L1]) < 0.50 → per-layer budget motivated.
NO-GO: r_99/d(W_Q[L1]) ≥ 0.60 → depth-invariant, flat K=128 is already near-optimal.
GRAY: 0.50–0.60 → modest per-layer budget advantage; combine with ODWQ's Spearman correlation.

#### 7. Risk profile

R1: Layer-1 W_Q may not be more concentrated; CF6 MLP anomaly and attention anomaly may decouple.
R2: If W_Q[L1] r_99 is actually HIGHER than L14 (more distributed attention in early layers), the per-layer schedule goes the other direction — still useful.

#### 8. Two-paper composition flag

Closest pair: SVD-LLM V2 (per-layer rank allocation) + CF6 (this pipeline, layer-1 anomaly).
Value-add: grounding per-layer attention compression budget in MLP gate variance structure (CF6) rather than Hessian/sensitivity — a different, spectrum-native budget allocation argument.

#### 10. Verdict

**REFINE** — 5-min gate check before investing. Shares infrastructure with AWCJR/ODWQ. Run as a precondition sweep.

---

## PATH 2 — Convergence Representatives

---

### VOKV — W_V/W_O Spectrum Joint + KV-Cache Compression (C1 rep)

#### 1. Load-bearing claims

M1: W_V[L] r_99/d < 0.75 (concentrated spectrum).
M2: W_V low-rank basis enables KV-cache storage compression algebraically (KV stored in W_V's 256-dim basis instead of 2048-dim).
M3: W_V + W_O joint compression at K=256 each is viable with ΔNLL < 0.5 nats.

#### 2. Per-claim prior-art status

**M1:** PARTIAL. Spectral Lifecycle (arXiv:2604.22778): "value/output projections compress uniformly." This is strong adjacent evidence. LatentLLM performs joint V+O decomposition zero-shot. However, Qwen3-specific W_V r_99/d measurement: NOVEL.

**M2 (KV-cache algebraic compression via W_V basis):** ADJACENT. MLA (DeepSeek-V2) stores KV in compressed latent space via training. MHA2MLA (arXiv:2502.14837) converts MHA to MLA post-training with minimal fine-tuning. KVCrush (2025) uses head-behavior similarity. SVDq achieves 410× key cache compression. The NOVEL element: exploiting W_V's own spectral structure (SVD basis) as the KV storage coordinate WITHOUT any fine-tuning AND deriving the algebraic equivalence (not learned compression). The algebraic identity V_compressed = V^T h, then KV = (V_V^T h) stored, recovered via W_O · U_V · Σ_V is structurally distinct from MLA/MHA2MLA which use trained projection layers.

**M3 (ΔNLL):** NOVEL. Qwen3-1.7B no-retraining W_V+W_O SVD ΔNLL: unpublished.

#### 3. Frame-mismatch check (CF9)

M2 algebraic identity claim: precondition is W_V spectral concentration (M1). If W_V is full-rank, M2 reduces to trivial identity. The identity itself is mathematically exact given any low-rank approximation — no theorem precondition issue. CF9 clean.

#### 4. Calibration ill-conditioning (CF10)

No calibration fit. CF10 clean.

#### 5. Residency arithmetic (detail)

KV-cache Qwen3-1.7B at 4K context: 28 × 2 × 8 KV-heads × 128 dim × 4096 tokens × 2B = 944 MB (GQA: 8 KV heads × 128 dim). At K=256 W_V basis: 944 MB × (256/2048) = 118 MB. 8× KV reduction contingent on M1.
W_V+W_O weight compression: 448 MB → ~100 MB at K=256.
Total attention weight: 896 MB → ~200 MB.
CF13-15 not cited.

#### 6. Smallest-test sharpening

Script: `scripts/vokv_wv_wo_spectrum.py` (share with AWCJR)
Model: Qwen3-1.7B-Base.
W_V[L14] and W_O[L14] SVD: 10 min.
GO → full 28-layer sweep + NLL: 30 min additional.
Output: `experiments/stage0/ladder_v2/round4_vokv/`
Wall-clock: 40 min. **Within 8h gate.**

#### 7. Go/no-go thresholds

GO: mean W_V r_99/d ≤ 0.75 → KV-algebraic compression viable.
NO-GO: W_V r_99/d > 0.90 → Spectral Lifecycle's "uniform compression" may apply to the training dynamics view but Qwen3's post-training W_V doesn't compress post-hoc; kill KV-algebraic path.
GRAY: 0.75–0.90 → higher K (e.g., K=512) may still enable 4× KV compression at lower quality cost.

#### 8. Risk profile

R1: MHA2MLA/LatentLLM pre-emption risk on KV compression (they do it too but require fine-tuning). Differentiation: no fine-tuning + Qwen3-specific spectrum measurement.
R2: GQA structure in Qwen3-1.7B (8 KV heads, not 16) means W_V is already smaller; the per-KV-head W_V is 128×2048, not 2048×2048 for W_Q. W_V is already at a smaller shape; r_99/d question is per KV-head matrix.
Cheapest falsifier: W_V[L14] SVD (5 min).

#### 9. Two-paper composition flag

Closest pair: MHA2MLA (arXiv:2502.14837, MHA→MLA post-training) + Spectral Lifecycle (arXiv:2604.22778, V/O uniform compression).
Value-add: zero-shot (no fine-tuning), algebraic identity derivation, Qwen3 GQA-specific measurement, CF11-calibrated threshold. Not purely ENGINEERING-INTEGRATION because the zero-shot algebraic identity is structurally different from MHA2MLA's retraining approach.

#### 10. Verdict

**REFINE** — shares spectrum measurement with AWCJR (run once, credit both). KV algebraic compression claim is novel enough to merit Stage 4/5 attention. MHA2MLA adjacency is noted but not pre-emptive.

---

### C-AQOD — Attention-Outlier Disjoint Partition (C2/C5 rep)

#### 1. Load-bearing claims

M1: Static outlier channels (K=0.1%, ~2 channels per CF3) are nearly orthogonal to W_Q's top-128 right singular subspace (cos² < 0.05).
M2: This disjointness allows separate FP16 handling of outlier channels + compressed W_Q without quality loss.

#### 2. Per-claim prior-art status

**M1 (outlier ⊥ W_Q subspace):** NOVEL. CF3 × CF11 composition measuring orthogonality between activation outlier channels and W_Q projection subspace: no published paper found doing this as of 2026-05-09. SmoothQuant handles outliers in activation quantization, not in W_Q projection accuracy. PrefixQuant separates channel-wise/token-wise outliers but doesn't intersect with attention weight subspace.
elegance-class: `subspace-alignment` (disjointness of two independently measured structures).

**M2 (practical decomposition):** NOVEL in coupling. Engineering is straightforward; the structural claim (disjointness) is what's novel.

#### 3. Frame-mismatch check (CF9)

No imported theorem. Pure inner-product measurement. CF9 clean.

#### 4. Calibration ill-conditioning (CF10)

No fit. CF10 clean.

#### 5. Smallest-test sharpening

Script: `scripts/caqod_outlier_wq_alignment.py` (extends `scripts/outlier_wq_alignment.py`)
Uses PDAP result (static outlier channel IDs) + AQFKV result (W_Q SVD top-128 singular vectors).
Compute cos² alignment for all 28 layers. Runtime: ~20 min.
Output: `experiments/stage0/ladder_v2/round4_caqod/`
Wall-clock: 20 min. **Within 8h gate.**

#### 7. Go/no-go thresholds

GO: mean cos² < 0.05 across layers → disjoint → partition compresses W_Q without outlier interference.
NO-GO: cos² ≥ 0.20 → outlier channels encode high-attention-score directions; W_Q subspace IS the outlier subspace → different structural finding (W_Q latches onto outlier channels for computation).
GRAY: cos² in 0.05–0.20 → partial overlap; per-layer adaptive partition.

#### 10. Verdict

**REFINE** — 20-min measurement on existing data. High information-to-cost ratio. NOVEL first-order claim.

---

### C-VKSD — W_K/W_V Spectral Delta Composition (C1 rep)

#### 1. Load-bearing claims

M1: Within-layer subspace gap ‖P_{K,256} − P_{V,256}‖_F < 0.3 for ≥ 20/28 layers.
M2: Shared W_K/W_V basis at K=256 compresses both at 5× with ΔNLL < 0.5 nats.

#### 2. Per-claim prior-art status

**M1:** NOVEL. No published paper measures within-layer W_K/W_V subspace alignment on standard MHA Qwen3. GQA research focuses on head-count sharing, not per-layer within-matrix subspace alignment between K and V projections.

**M2:** NOVEL for Qwen3 post-training. MLA (training-time) does something structurally similar but requires training-time reparameterization.

#### 3. Frame-mismatch check (CF9)

No imported theorem. CF9 clean.

#### 4. Calibration ill-conditioning (CF10)

Shared basis is computed analytically via SVD of stacked [W_K; W_V]. Not a fit. CF10 clean.

#### 5. Smallest-test sharpening

Script: `scripts/cvksd_kv_subspace_gap.py`
Load W_K, W_V per layer (use GQA-adjusted shapes). SVD each, measure F-norm projection difference. ~30 min.
Output: `experiments/stage0/ladder_v2/round4_cvksd/`
Wall-clock: 30 min. **Within 8h gate.**
Note: Qwen3-1.7B uses GQA with 8 KV heads. W_K and W_V are (8 × 128) × 2048 matrices, not 2048×2048. Subspace gap measurement must account for this shape difference. Per-KV-head gap is the relevant measurement.

#### 7. Go/no-go thresholds

GO: mean per-head subspace gap < 0.3 → joint KV projection viable.
NO-GO: gap ≥ 0.5 → W_K and W_V span independent subspaces post-training → closes spontaneous-MLA question.

#### 10. Verdict

**REFINE** — 30-min measurement, novel per-head subspace alignment claim, NOVEL on Qwen3.

---

### C-RBHD — Residual-Band Head-Diversity Decoupling (C4 rep)

#### 1. Load-bearing claims

M1: Per-head outlier-routing Jaccard |D_h ∩ D_{h'}| / |D_h ∪ D_{h'}| < 0.3 for ≥ 10/28 layers (heads are functionally diverse at outlier level despite weight-level redundancy per CF11).

#### 2. Per-claim prior-art status

**M1:** NOVEL. CF11 × CF3 coupling measuring per-head attention-routing diversity over activation outlier channels: no published paper found doing this as of 2026-05-09. "Share Your Attention" (CoSpaDi) focuses on weight-space head sharing; doesn't measure functional routing diversity over outlier channels.

#### 3. Frame-mismatch check (CF9)

Jaccard is a standard set-intersection measure. Precondition: D_h must be well-defined as a stable set. CF3 shows K=1% is token-dynamic (Jaccard=0.31) — so per-head outlier routing may be context-dependent, making D_h noisy. CF9 concern: the "per-head stable routing set" may not exist if routing is highly context-dependent. Mitigation in smallest test: compute mean Jaccard across prompts with variance; if variance > mean, the routing set is not stable and M1 is unmeasurable.

#### 4. Calibration ill-conditioning (CF10)

No fit. CF10 clean.

#### 5. Smallest-test sharpening

Script: `scripts/crbhd_head_diversity.py`
60 min: reuse PDAP forward-pass (200 prompts), extract per-head Q activations, compute per-head top-5 channel routing, compute pairwise Jaccard.
Output: `experiments/stage0/ladder_v2/round4_crbhd/`
Wall-clock: 60 min. **Within 8h gate.**

#### 7. Go/no-go thresholds

GO: mean pairwise Jaccard < 0.3 across 28 layers → per-head K budget assignment live.
NO-GO: Jaccard ≥ 0.5 → heads are functionally indistinct in outlier routing.

#### 10. Verdict

**REFINE** — novel composition, but flag the CF9-adjacent concern about routing set stability. Include variance reporting in experiment plan.

---

### C-ODWQ — Outlier-Drift Weighted W_Q Layer Budget (C2 rep)

#### 1. Load-bearing claims

M1: Spearman ρ(outlier_spread_L, r_99(W_Q^L)) > 0.3 across 28 layers.
M2: Non-uniform K schedule (e.g., K=64 early layers, K=256 deep layers) achieves same or better ΔNLL as flat K=128.

#### 2. Per-claim prior-art status

**M1:** NOVEL. No published paper correlates per-layer activation outlier spread with per-layer W_Q effective rank on any model. This is a pure CF3 × CF11 composition.

**M2:** ADJACENT. SVD-LLM V2's non-uniform rank allocation optimizes per-layer rank based on sensitivity (Hessian-based), not spectrum correlation. The mechanism is different (Hessian vs outlier-spread proxy), but the outcome (non-uniform K schedule) is similar. Differentiation: CF3-based budget is cheaper to compute than Hessian and grounded in activation dynamics, not weight sensitivity.

#### 3. Frame-mismatch check (CF9)

No imported theorem. Spearman correlation is a non-parametric test with no precondition beyond ordinal data. CF9 clean.

#### 4. Calibration ill-conditioning (CF10)

No fit. Spearman correlation is analytical. CF10 clean.

#### 5. Smallest-test sharpening

Script: `scripts/codwq_outlier_rank_correlation.py`
All data in hand from PDAP (per-layer outlier_spread_L) and AQFKV (per-layer r_99(W_Q^L)).
Spearman ρ computation over 28 points. Runtime: ~10 min.
Output: `experiments/stage0/ladder_v2/round4_codwq/`
Wall-clock: 10 min. **Within 8h gate.**
Note: Verify that AQFKV result file contains per-layer r_99, not just L14. If only L14, add 15-min per-layer W_Q SVD sweep as first step.

#### 7. Go/no-go thresholds

GO: ρ > 0.3, p < 0.05 → budget stratification motivated.
NO-GO: ρ < 0.1 → W_Q rank and outlier dynamics are independent → flat K=128 is already near-optimal.
GRAY: ρ in 0.1–0.3 → marginal; check whether the deep-layer group (L22-L27) shows higher ρ.

#### 10. Verdict

**REFINE** — 10-min measurement using existing pipeline data. Highest-value cheapest experiment in this cluster. Run this first.

---

### F-SPECA — Spectral Equivalence W_K via W_Q Alignment (C1 rep)

#### 1. Load-bearing claims

M1: Within-layer [W_Q; W_K] stacked var@256 ≥ 0.85 in ≥ 20/28 layers (high Q/K subspace overlap per layer).
M2: Shared basis at r=256 gives ΔNLL ≤ 0.5 nats.

#### 2. Prior-art status

**M1:** NOVEL. Per-layer within-layer W_Q/W_K joint basis measurement: unpublished on Qwen3. GQA / MLA share K/V heads (not W_Q+W_K basis) during training. LatentLLM joints Q and V, not Q and K.

**M2:** NOVEL on Qwen3 post-training.

#### 3. Frame-mismatch (CF9)

CF9 concern: GQA changes W_K shape (8 KV-head × 128 × 2048) vs W_Q (16 Q-head × 128 × 2048). Stacking [W_Q; W_K] has a 2:1 Q-vs-K shape ratio; SVD of the stack is dominated by the numerically larger W_Q component. Mitigation: normalize W_Q and W_K contributions before stacking, or measure principal angle between subspaces directly rather than variance of the stack. The var@256 measurement on a non-normalized stack may be biased. Flag this for the experiment plan.

#### 4. Smallest-test sharpening

Script: `scripts/fspeca_qk_joint_basis.py`
~45 min: 28 SVDs of shape-normalized [W_Q; W_K] stacked matrices.
Output: `experiments/stage0/ladder_v2/round4_fspeca/`
Wall-clock: 45 min. **Within 8h gate.**
Add normalization correction to address GQA shape asymmetry.

#### 7. Verdict

**REFINE** — with GQA-normalization correction added to the experiment plan. CF9 concern is addressable but must be documented.

---

### F-WNORM (covered under WDRS above) — Verdict: REFINE

---

### A-GFWV — Gauge-Fixed W_V Folding (C4 rep)

#### 1. Load-bearing claims

M1: Per-head W_V·W_O product (128×2048) has r_99/d ≤ 0.70 (concentrated, enabling product fold).
M2: Gauge-fix to W_V·W_O product's principal basis is algebraically valid within-layer.

#### 2. Prior-art status

**M1:** PARTIAL. Spectral Lifecycle (arXiv:2604.22778) confirms V/O projections compress uniformly. LatentLLM explicitly performs joint V+O decomposition. However: W_V·W_O PRODUCT spectrum (vs independent spectra) is NOT the same object. The product can have lower rank than either individual matrix (rank collapse through multiplication). This is an ADJACENT measurement to LatentLLM's joint decomposition: LatentLLM decomposes V and O independently; A-GFWV decomposes their PRODUCT.
elegance-class: `algebraic-identity` (exact fold of product operator).

**M2:** NOVEL. Gauge-fixing using product matrix's principal basis is a different operation from LatentLLM's paired decomposition.

#### 3. Frame-mismatch (CF9)

Gauge symmetry claim: the residual-stream rotation symmetry used to "optimally align" coordinates is EXACT only if RMSNorm is not present. Qwen3 uses RMSNorm before each block. The same CF9 concern as F-GFQO: RMSNorm is not rotationally invariant. However, for A-GFWV, the gauge-fix is WITHIN the attention block (before the softmax and after the value projection), and RMSNorm operates on the residual stream (before the block, not within it). The W_V·W_O fold is purely within the attention output computation: Out = A · (h W_V) · W_O. The fold is exact regardless of RMSNorm because it operates on the output of softmax(QK^T)·V, not on the residual stream input. CF9 clean for the within-attention fold.

#### 4. Calibration ill-conditioning (CF10)

No fit. SVD of product matrix. CF10 clean.

#### 5. Smallest-test sharpening

Script: `scripts/agfwv_product_spectrum.py`
Compute W_V·W_O per-head (128×2048 product per head). SVD each. 196 products + SVDs. ~30 min.
Output: `experiments/stage0/ladder_v2/round4_agfwv/`
Wall-clock: 30 min. **Within 8h gate.**

#### 7. Go/no-go thresholds

GO: mean r_99/d ≤ 0.70 across heads/layers → product fold viable.
NO-GO: r_99/d ≥ 0.90 → product is full-rank even if individual matrices compress. Closes this path; Spectral Lifecycle's V/O claim may be training-dynamics rather than weight-structure.

#### 10. Verdict

**REFINE** — distinct from LatentLLM (product spectrum ≠ individual spectra), algebraic-identity elegance class, 30-min check.

---

## PATH 3 — Frame-Novelty Proposals

---

### F-GFQO — Gauge-Fixing Residual Stream via Q-K Subspace

#### 1. Load-bearing claims

M1: Joint rowspan(W_Q) ∪ rowspan(W_V) ∪ rowspan(W_K) per layer has effective dimension ≤ 1600 (var@1600 ≥ 0.99).
M2: Gauge rotation exposing this block-sparse structure is compatible with RMSNorm (rotation R commutes with per-channel scale up to < 1% distortion).
M3: Block-sparse structure reduces attention-side GEMV cost.

#### 2. Prior-art status

**M1 (joint subspace dimension):** NOVEL. No published paper measures the joint attention input subspace dimension as a residual-stream coordinate property on Qwen3 or similar SwiGLU decoder. The Sparse Frontier (arXiv:2504.17768) analyzes sparse attention score patterns but not residual-stream coordinate geometry. QuIP/QuIP# use rotation for incoherence (opposite goal). As of 2026-05-09: no paper found applying gauge-freedom frame to expose attention-invisible residual dimensions.

**M2 (RMSNorm compatibility):** NOVEL. The specific question of which rotation classes commute with RMSNorm per-channel scale is unaddressed in the literature. QuIP's random Hadamard rotation is indifferent to this; this paper requires a structured R.

**M3 (cost reduction):** ADJACENT. Efficient attention with sparse/structured coordinate alignment is well-studied, but not from the gauge-fixing angle.

#### 3. Frame-mismatch check (CF9)

**CF9 risk (flagged in Stage 2):** The gauge rotation R must preserve RMSNorm behavior. RMSNorm normalizes the residual stream by its Euclidean norm, then scales by per-channel γ. The operation y = γ ⊙ (x / ‖x‖) is NOT invariant under arbitrary rotations: ‖Rx‖ = ‖x‖ (preserved) but γ ⊙ (Rx / ‖Rx‖) ≠ R · (γ ⊙ x / ‖x‖) because the per-channel scale γ applies AFTER rotation, breaking the equivariance. This is the CF9 precondition failure: the gauge symmetry is broken by RMSNorm's per-channel scale unless R is restricted to the class that commutes with diag(γ). Such an R must be a product of permutations and sign flips (the group of matrices that commute with any diagonal matrix is the group of generalized permutation matrices) — this is highly restrictive and eliminates general orthogonal R, reducing the gauge freedom from O(d) to just {sign flips × permutations}. **The block-sparse structure exposed by general orthogonal R is NOT achievable while preserving RMSNorm's semantic behavior.** This is a material CF9 failure for the general gauge-freedom claim. What survives: restricted gauge (permutations + sign flips) can still permute channel order to expose the attention subspace structure, but cannot rotate to a continuously optimal basis.

Mitigation in proposal: Stage 2 noted "RMSNorm-compatible rotation class" as mitigation. The surviving mechanism is: (1) measure which residual-stream channels are NOT read by W_Q/W_K/W_V (i.e., have low projection magnitude onto all attention weight rows), (2) permute those channels to a separate coordinate block, (3) store that block at lower precision. This is a valid mechanism but is substantially weaker than the original gauge-freedom claim — it reduces to channel importance ordering, which is adjacent to SmoothQuant/Activation-Aware Weight Quantization.

**CF9 verdict: PARTIAL FAILURE.** The elegant gauge-freedom framing survives only in restricted form; the residual-stream channel-importance measurement is ADJACENT to published activation-aware methods.

#### 4. Calibration ill-conditioning (CF10)

No calibration fit in the measurement phase. CF10 clean.

#### 5. Smallest-test sharpening

Script: `scripts/fgfqo_joint_subspace.py`
Measure joint rowspan of [W_Q; W_K; W_V] per layer; record var@K for K ∈ {1024,1280,1536,1600,1800,2048}. ~25 min.
ALSO: compute ‖γ-distortion‖ for RMSNorm scale under random R (CF9 mitigation check). ~5 min.
Output: `experiments/stage0/ladder_v2/round4_fgfqo/`
Wall-clock: 30 min. **Within 8h gate.**

#### 7. Verdict

**DOWNGRADE** — CF9 partially fails: general gauge rotation is broken by RMSNorm per-channel scale. The surviving mechanism (channel-importance permutation for lower-precision assignment) is ADJACENT to SmoothQuant / activation-aware quantization. The frame is genuinely novel (no published paper applies gauge-freedom language here) but the mechanism it exposes is not novel after the CF9 correction. Downgrade to an **engineering measurement** (channel-importance ordering for mixed-precision assignment); not a Stage 5 candidate this round unless CF9 mitigation produces a strictly stronger mechanism than SmoothQuant. Park for Stage 4 reframe: can the channel-importance structure discovered here motivate a mixed-precision W_Q block structure?

---

### WSFI — Windows SuperFetch Footprint Inversion

#### 1. Load-bearing claims (wildcard)

M1: SysMain/SuperFetch learns sequential GGUF slab access pattern within 2-3 inference runs.
M2: After learning, prefetch overlap reduces per-layer NVMe wait by ≥ 30% (run-10 latency ≤ 70% of run-3).

#### 2. Prior-art status

**M1 (SuperFetch learning of GGUF slab pattern):** NOVEL. No published LLM inference paper exploits SuperFetch/SysMain's CLR/PC-based prefetch learning. Apple LLM-in-a-Flash uses mmap demand paging; FlexGen uses CPU-GPU offload scheduling; none use OS-level prefetch histogram learning. Confirmed: no published paper found invoking Windows SysMain prefetcher for LLM inference as of 2026-05-09. NOVEL.

**M2 (latency reduction measurement):** NOVEL. The specific quantification of SuperFetch learning loop benefit for sequential weight-slab access has no published measurement.

#### 3. Frame-mismatch check (CF9)

Imported machinery: SysMain's CLR/PC-based prefetcher (Windows Internals).
Precondition: sequential, consistent access pattern that the prefetcher can learn. Qwen3 inference in llama.cpp with `--no-mmap` and sequential layer loading: access pattern IS sequential and consistent across inference passes on the same prompt. Precondition met. CF9 clean.
However: a real-world concern is that llama.cpp's GGUF access pattern may not be purely sequential if different tensor types within a layer are accessed in different orders. The GGUF offset-table reordering step is critical to make the pattern consistent. The WSFI proposal correctly requires a slab-ordered GGUF as a precondition; this must be an explicit experimental step.

#### 4. Calibration ill-conditioning (CF10)

No calibration fit. CF10 clean.

#### 5. Smallest-test sharpening

Script: Measurement harness using ETW (Event Tracing for Windows) or Process Monitor for per-layer NVMe read timing.
Model: Qwen3-1.7B-Base, IQ4_XS GGUF.
Step 1: Verify SuperFetch is enabled (1 min: registry check `HKLM\...\EnableSuperfetch`).
Step 2: Lay out slab-ordered GGUF (sort tensors by layer index). ~20 min preprocessing.
Step 3: 10 inference passes on each (standard + slab-ordered GGUF), record per-layer read timing with procmon/ETW.
Step 4: Compare run-3 vs run-10 latency on slab-ordered GGUF.
Wall-clock: ~2 hours. **Within 8h gate.**

#### 7. Go/no-go thresholds

GO: run-10 per-layer NVMe latency ≤ 70% of run-3 on slab-ordered GGUF.
NO-GO: No measurable reduction → SuperFetch does not engage on GGUF access patterns (documents real Windows 11 NVMe behavior for all future proposals).
GRAY: 70-85% reduction → partial learning; check if SuperFetch registry shows PF trace growth.

#### 8. Risk profile

R1: SuperFetch disabled or suppressed for NVMe SSDs (some Windows 11 configurations auto-disable it on fast SSDs). Mitigation: check registry first; if disabled, fallback to `FILE_FLAG_SEQUENTIAL_SCAN` CreateFile hint as alternative primitive.
R2: llama.cpp's mmap mode bypasses CreateFile, making sequential-scan hints ineffective; must use `--no-mmap`.

#### 9. Two-paper composition flag

Closest pair: Apple LLM-in-a-Flash (mmap demand paging) + Windows Internals SuperFetch documentation.
Value-add: exploits the LEARNING aspect of SysMain (dynamic adaptation over runs) vs static layout tricks. Research value: first empirical measurement of SuperFetch behavior on LLM inference workload. The learning-loop dynamic is the structural novelty.

#### 10. Verdict

**REFINE — wildcard, CF-tether-suspended, structural floor met.** CF9 clean (precondition holds for sequential GGUF access). CF10 clean. Smallest test ≤ 8h. Genuine substrate novelty. No published prior.

---

## PATH 4 — Algebraic-Identity Pair

---

### F-WVOS — W_V/W_O Spectral Structure + Algebraic Fold

#### 1. Load-bearing claims

M1: W_V has concentrated spectrum (r_99/d ≤ 0.80) in Qwen3-1.7B.
M2: The fold V_folded = V_V^T, O_folded = W_O · U_V · Σ_V is algebraically exact.
M3: Rank-256 fold gives ΔNLL ≤ 0.5 nats.

#### 2. Prior-art status

**M1:** PARTIAL. Spectral Lifecycle (2604.22778) supports W_V compressibility. LatentLLM/MHA2MLA do joint V+O decomposition. Qwen3 private measurement: NOVEL.
elegance-class: `algebraic-identity` (exact computation-graph reduction, no approximation in the fold operator itself — only approximation is in the SVD truncation at rank r).

**M2 (algebraic identity):** ADJACENT. The identity itself is mathematically standard (low-rank matrix factorization of the value projection). The Stage 2 audit flagged that the fold produces two matrices (V_folded ∈ R^{r×d_model} and O_folded ∈ R^{d_model×r}) — this is a standard SVD decomposition, NOT further simplification. The claim is NOT that the two matrices become one; it is that the combined (r×d_model) + (d_model×r) footprint is smaller than the original (d_k×d_model) + (d_model×d_model) footprint when r < d_k. At r=128 vs d_k=128 per head, the per-head V is already 128-dim — no compression unless the GQA structure allows r < 128. For Qwen3-1.7B with 8 KV heads × 128 dim: W_V per KV head is 128×2048; the fold at r=64 produces (64×2048) + (2048×64) = 262K params vs 128×2048 = 262K params. **There is NO parameter reduction at the per-head level if we fold W_V (128×2048) per KV-head into two factors.** The algebraic fold only produces compression if it folds ACROSS heads.

**CF9 ALERT for F-WVOS:** The Stage 2 audit requested verification of the fold construction. The original Stage 1 description treats W_V as the full 2048×2048 matrix (multi-head stacked). In Qwen3-1.7B with GQA (8 KV heads), W_V is NOT 2048×2048; it is shape (8×128)×2048 = 1024×2048. The fold applies to the full 1024×2048 matrix: rank-256 fold gives (256×2048) + (2048×256) = 1M params vs 2×2048×1024 = 4.2M params — 4× reduction. This IS meaningful. But the Stage 1 description used d_model×d_model = 2048×2048 for W_V, which is incorrect for GQA. **Must verify Qwen3-1.7B's exact W_V shape before arithmetic.** If GQA gives W_V as (1024×2048), the compression math holds.

**M3 (ΔNLL):** NOVEL on Qwen3.

#### 3. Frame-mismatch check (CF9)

GQA shape mismatch (flagged above): the arithmetic must use the actual W_V tensor shape. Verify with `model.layers[0].self_attn.v_proj.weight.shape` before finalizing residency claims.

#### 4. Calibration ill-conditioning (CF10)

No calibration fit. SVD + product. CF10 clean.

#### 5. Smallest-test sharpening

Script: `scripts/fwvos_wv_fold.py`
Step 1: Verify W_V shape for Qwen3-1.7B GQA. 2 min.
Step 2: SVD of W_V[all layers]. ~15 min.
Step 3: Construct fold at K=256, measure NLL on 512-token WikiText-2. ~15 min.
Output: `experiments/stage0/ladder_v2/round4_fwvos/`
Wall-clock: 30 min. **Within 8h gate.**

#### 7. Go/no-go thresholds

GO: r_99/d ≤ 0.80 AND ΔNLL ≤ 0.5 nats at K=256.
NO-GO: W_V spectrum flat (r_99/d > 0.90) → CF8 extends to W_V; fold gives no compression.
GRAY: r_99/d in 0.80–0.90 → try K=512 or activation-weighted SVD variant.

#### 10. Verdict

**REFINE** — top score in run, cheapest definitive test (30 min), algebraic-identity elegance. Must fix GQA shape arithmetic before Stage 5. LatentLLM/MHA2MLA do similar but require fine-tuning; this is zero-shot.

---

### F-AERO-Q3 — AERO-Style W_gate Elimination Feasibility Probe

#### 1. Load-bearing claims

M1: On Qwen3-1.7B, replacing silu(W_gate·x) with W_gate·x globally (silu→identity) gives ΔNLL ≤ +0.5 nats.
M2: Gate pre-activations are predominantly in the linear region of silu on Qwen3's calibration distribution.
M3: If M1 holds, W_gate and W_up algebraically merge (AERO fold reduces FFN from 3 matrices to effectively 2 with lower residency).

#### 2. Prior-art status

**M1 (silu→identity quality cost on Qwen3):** NOVEL. AERO paper (arXiv:2410.13060) was applied to GELU/softmax models for private inference. The KILL_LIST explicitly records: "AERO-style activation removal for SwiGLU on Qwen3 specifically: untested." Confirmed by search: AERO paper text uses GELU and LayerNorm removal for private inference context; it does NOT test post-training removal on SwiGLU without retraining on Qwen3-family models. No paper found doing silu→identity on Qwen3 post-training for compression. NOVEL.

**M2 (gate pre-activation linear region fraction):** NOVEL. CF1 established W_up dominates firing-rank; CF6 established gate output near-constancy is rare (1.5% globally). Neither measures what fraction of gate PRE-ACTIVATIONS fall in silu's linear region (|silu(z)−z|/|z| < 0.05), which requires z >> 0 (large positive values). These are different questions. NOVEL measurement.

**M3 (fold validity after M1):** ADJACENT. AERO explicitly derives this fold for activation-free FFN layers. The novel claim for Qwen3 is M1's ΔNLL.
elegance-class: `algebraic-identity` (W_gate + W_up → single matrix after activation removal, exact fold).

#### 3. Frame-mismatch check (CF9)

AERO paper context: designed for PRIVATE INFERENCE (two-party computation where nonlinearities are expensive). The private-inference motivation doesn't apply here. However, the MECHANISM (activation removal + fold) is independent of the motivation. The precondition that matters here: does silu→identity on Qwen3 preserve quality? This is an EMPIRICAL precondition, not a theorem precondition. CF9 doesn't apply to empirical tests — it applies to imported theoretical machinery. The silu→identity substitution is straightforward; the quality question is purely empirical.

CF6 concern: 1.5% globally near-constant gate OUTPUTS suggests the gate IS actively used for routing (output variability is high). But AERO's fold requires low ACTIVATION ERROR (silu(z) ≈ z for the actual z values in Qwen3), not near-constant outputs. These are different. If z is consistently large and positive, silu(z) ≈ z even when silu(z) is large and variable. The fold is plausible if gate pre-activations are predominantly positive.

**Stage 2's aggressive check:** "does AERO cite Qwen3-family results? If not, the experiment is the first such test." Confirmed: AERO paper focuses on private inference for GPT-2/BERT family; no Qwen3/SwiGLU results. This is the first such test.

#### 4. Calibration ill-conditioning (CF10)

No calibration fit. Forward pass modification only. CF10 clean.

#### 5. Smallest-test sharpening

Script: `scripts/faeroq3_silu_identity.py`
Model: Qwen3-1.7B-Base, bf16.
Step 1: Intercept SwiGLU forward pass; replace silu with identity (lambda x: x) for all 28 layers.
Step 2: Measure NLL on 512-token WikiText-2.
Step 3: Measure gate pre-activation distribution (mean, std, fraction with z > 0 and |silu(z)−z|/|z| < 0.05).
Runtime: ~10 min.
Output: `experiments/stage0/ladder_v2/round4_faeroq3/`
Wall-clock: 10 min. **Within 8h gate.** Fastest decisive test in this entire run.

#### 7. Go/no-go thresholds

GO: ΔNLL ≤ 0.5 nats → AERO fold viable on Qwen3, W_gate elimination confirmed; cascade to merged-matrix weight storage.
NO-GO: ΔNLL > 2 nats → silu is load-bearing; gate nonlinearity is actively used; AERO-style fold requires retraining on SwiGLU models. Class kill for "post-training silu→identity on SwiGLU without retraining."
GRAY: 0.5–2 nats → per-layer replacement (identify which layers tolerate it, likely layer 1 per CF6's 36% near-constant neurons).

#### 8. Risk profile

R1: silu IS load-bearing in Qwen3 (CF6's 1.5% near-constant outputs suggests most neurons actively use the nonlinearity). Expected NO-GO. But the experiment is 10 min and either outcome is structurally valuable.
R2: If silu is layer-1-tolerant only (CF6's 36% anomaly), the W_gate elimination applies only to layer 1 — modest saving (~24 MB) but confirms a layer-specific fold mechanism.
Cheapest falsifier: this IS the cheapest test (10 min).

#### 9. Two-paper composition flag

Closest pair: AERO (arXiv:2410.13060, activation removal fold) + CSMF kill (R3, SwiGLU activation manipulation fails on Qwen3).
Value-add: CSMF killed polynomial substitution; this is identity substitution (different mechanism). AERO tested on GELU/softmax; this is the first SwiGLU+Qwen3 test. The value-add is empirical completeness of the gate-manipulation class characterization.

#### 10. Verdict

**REFINE** — top-2 score, fastest test (10 min), most decisive binary in this run. Expected NO-GO is fine — either outcome closes an important question. Stage 5 should run this first.

---

## WILDCARDS

---

### HRQK — Head-Redundancy Routing: MLA-Style Q-K Post-Training [FREE SWING]

#### 1. Load-bearing claims (wildcard)

M1: PCA over 16 W_Q heads of a single layer yields K_basis=64 capturing ≥ 90% head variance.
M2: Reconstruction of W_Q[L14]·x via K_basis=64 head-PCA gives ΔNLL < 1.5 nats.

#### 2. Prior-art status

**M1:** NOVEL. Within-layer head-PCA of W_Q (16 heads × 128 × 2048, treating heads as samples): no published paper found. MLA/GQA share heads during training; this is post-training within-layer head-PCA.

**CRITICAL reconciliation with CF11:** CF11 says per-head K=64 is NO-GO (+1.53 nats). HRQK claims within-layer head-PCA at K_basis=64 captures ≥ 90% head variance. These are DIFFERENT OPERATIONS:
- CF11 per-head K=64: truncate EACH head's 128×2048 weight matrix to rank-64 independently → each head loses its lower 64 singular values.
- HRQK K_basis=64: compute PCA over the 16 heads' weight matrices (treating the head index as the sample dimension), find the 64-dim subspace shared across heads → replace 16 × (128 × 2048) with 1 × (64 × 2048) basis + 16 × (128 × 64) readout coefficients.
These ARE different operations and could have different quality costs. The CF11 NO-GO applies to per-head rank truncation; the HRQK operation pools across heads. However: if CF11's global K=128 meaning is "the 16 heads span a 128-dim subspace but each head is diverse within that subspace," then HRQK's head-PCA basis = 64 captures only HALF of that shared subspace — and the readout coefficients (128 × 64 per head) may not reconstruct the individual head's function. Stage 2 identified this reconciliation as the pressure-test target.

**M3:** The Stage 2 note "K_basis=64 ≥ 90% head variance" check is the gate. If the 16 heads are NOT similar (each spans a different part of the 128-dim per-head space), PCA across head matrices will not concentrate variance at K_basis=64 at all.

#### 3. Frame-mismatch (CF9)

No imported theorem. CF9 clean for wildcard.

#### 4. Calibration ill-conditioning (CF10)

No calibration fit. CF10 clean.

#### 5. Smallest-test sharpening

Script: `scripts/hrqk_head_pca.py`
Model: Qwen3-1.7B.
Reshape W_Q[L14] to (16 × 128 × 2048) → flatten to (16 × 262144); PCA; var@K_basis for K_basis ∈ {16,32,64,128}. 5 min.
Reconstruction NLL check: 10 min.
Output: `experiments/stage0/ladder_v2/round4_hrqk/`
Wall-clock: 15 min. **Within 8h gate.**

#### 7. Go/no-go thresholds

GO: K_basis=64 captures ≥ 90% head variance AND ΔNLL < 1.5 nats.
NO-GO: K_basis=64 captures < 60% head variance → heads are diverse, not redundant at head-space level; CF11's global finding arises from a different structure.

#### 10. Verdict

**REFINE — wildcard, CF-tether-suspended, structural floor met.** The CF11 reconciliation concern is addressable by the 5-min variance check. LatentLLM/TransMLA do post-training MHA→MLA with fine-tuning; HRQK is zero-shot with spectrum-grounded operation. Differentiated enough to advance.

---

### C-WVKQ — Spectral Ladder Joint KVQ Operator [FREE SWING]

#### 1. Load-bearing claims (wildcard)

M1: Joint rowspan of [W_Q; W_K; W_V] per layer concentrates at K=256 (F-norm ratio < 0.2).
M2: Single shared GEMV z=U^T h + three cheap readouts replaces three separate GEMVs.

#### 2. Prior-art status

**M1:** ADJACENT. LatentLLM (MERL TR2026-018) performs joint Q+V decomposition. TransMLA performs joint KV. A3 (arXiv:2505.12942) uses joint Q/K compression. HOWEVER: joint Q+K+V simultaneously (all three) with post-training no-retraining measurement: no paper found. PARTIAL pre-emption from LatentLLM's Q+V and TransMLA's K+V, but not Q+K+V joint.

**M2:** ADJACENT. MLA does this at training time. No post-training Q+K+V joint GEMV on standard MHA found.

#### 3. Frame-mismatch (CF9)

GQA concern: in Qwen3-1.7B, W_Q has 16 heads but W_K/W_V have 8 KV heads. Stacking [W_Q; W_K; W_V] has shape (16×128 + 8×128 + 8×128) × 2048 = 4096×2048 in local-head space, but the Q-heads and KV-heads have different input subspace structures. A single shared U^T basis for all three may bias toward W_Q's larger head count. CF9 concern: the GQA architectural asymmetry means the "single shared GEMV" requires GQA-aware treatment to avoid biasing U toward W_Q's structure. Mitigation: normalize by head count before stacking, or compute principal angles separately.

#### 4. Smallest-test sharpening

Script: `scripts/cwvkq_joint_svd.py`
Joint SVD of GQA-normalized [W_Q; W_K; W_V] stacked matrices at layers {5,14,25}. ~40 min.
Output: `experiments/stage0/ladder_v2/round4_cwvkq/`
Wall-clock: 40 min. **Within 8h gate.**
Add GQA normalization correction to experiment plan.

#### 7. Go/no-go thresholds

GO: F-norm ratio < 0.2 → joint basis well-motivated.
NO-GO: ratio > 0.5 → W_V (if full-rank per CF8 extension possibility) pollutes joint basis.

#### 10. Verdict

**REFINE — wildcard, CF-tether-suspended, structural floor met** — with GQA normalization correction added. LatentLLM/TransMLA partial pre-emption noted but joint Q+K+V is still unpublished. C-WVKQ should be treated as a lower priority than the pure W_V/W_O proposals (its novelty depends on W_V concentration, which is ungated).

---

### NNPA — NVMe Namespace Partitioning as Weight Tier [FREE SWING]

#### 1. Load-bearing claims (wildcard)

M1: GC stalls ≥ 50 ms occur during Qwen3-1.7B inference with concurrent write activity.
M2: NVMe namespace separation eliminates cross-contamination between read-only weight-layer NS and write-active NS.

#### 2. Prior-art status

**M1 (GC stall measurement):** NOVEL. Published work confirms GC stalls exist (Samsung AN1527); no published LLM inference paper quantifies the GC stall fraction of total inference wall time on consumer NVMe. NOVEL measurement.

**M2 (namespace separation):** ADJACENT. Nvidia ICMSP (CES 2026) uses NVMe namespace management for KV cache offload, but at datacenter scale with BlueField DPUs. Database literature (RocksDB NVMe tiering) uses namespace partitioning. No consumer-hardware LLM inference work uses it. NOVEL in the consumer/local-inference context.

Search confirmed: Dual-Blade (arXiv:2604.26557) uses NVMe-direct KV cache offloading but doesn't address GC stall isolation via namespace partitioning for weight layers.

#### 3. Frame-mismatch (CF9)

Substrate primitive: NVMe namespace management via `IOCTL_STORAGE_MANAGE_DATA_SET_ATTRIBUTES`. Precondition: consumer NVMe drive must support multiple namespace creation. Samsung 980/990 Pro, WD SN850 support NVMe 1.2+ namespace commands. Not guaranteed for all drives. Precondition: verify drive support before full experiment. Smallest test (GC stall measurement) does NOT require namespace support.

#### 4. Calibration ill-conditioning (CF10)

No fit. CF10 clean.

#### 5. Smallest-test sharpening

Script: measurement harness (Python + subprocess, simulate write load)
Step 1: Run 200 single-token inference passes on Qwen3-1.7B GGUF; simultaneously write 1 GB random data.
Step 2: Record per-token latency distribution.
Step 3: Compare p95 latency with vs without background writes.
Runtime: ~2 hours. **Within 8h gate.**

#### 7. Go/no-go thresholds

GO: p95 latency with writes ≥ 1.3× p95 without writes → GC stalls measurable.
NO-GO: No measurable difference → SSD's internal write buffer absorbs GC completely at this write rate. Documents GC interference floor.

#### 10. Verdict

**REFINE — wildcard, CF-tether-suspended, structural floor met.** GC stall measurement is the necessary first step; namespace partitioning is the solution only if M1 holds. Sequence: run M1 measurement first.

---

## SYSTEMS PROPOSALS (Track B, U orientation)

---

### NCZC — NTFS Compressed Clusters as Zero-Copy Decode Cache

**Summary treatment (~400w):**

Claims: row-sorted GGUF achieves ≥ 1.15× NTFS compression ratio. No imported theorem. CF9 clean. CF10 clean (no fit). Smallest test: 30 min (compact/c on two GGUFs). Wall-clock within 8h gate.

Prior-art: No published LLM work uses NTFS cluster compression with layout optimization (search confirmed, no results). The nearest neighbor is GGUF layout research for MMAP sequential access, which targets different goal (page-fault sequencing, not compression ratio). NOVEL in combination.

Two-paper pair: GGUF layout optimization (llama.cpp internal) + NTFS transparent compression (Windows Internals).
Value-add: LZ77 compression ratio improvement from magnitude-sorted row layout is a novel substrate observation.

**Risk:** NTFS random-access decompression penalty (64 KB unit decompression for 4 KB reads) is the critical concern under mmap. Mitigation: test with `--no-mmap` (sequential read mode). If sequential scan compresses 1.25× with kernel-transparent decompression hiding I/O overhead, this is a free 25% bandwidth improvement.

**Verdict: REFINE** — 30-min binary test, substrate-novel, no published prior.

---

### WWTE — Windows Working Set Trimmer as Eviction Oracle

**Summary treatment (~400w):**

Claims: ≥ 15% of GGUF pages persistently resident across 10 inference passes (measurable via QueryWorkingSetEx). CF9 clean (QueryWorkingSetEx is a standard Win32 API, no theorem preconditions). CF10 clean. Smallest test: 1 hour. Wall-clock within 8h gate.

Prior-art: Apple LLM-in-a-Flash uses activation statistics for hot-page selection; WWTE uses OS LRU state (different oracle, no ML predictor). No published paper uses QueryWorkingSetEx + VirtualLock for LLM inference hot-page selection. NOVEL substrate usage.

Two-paper pair: Apple LLM-in-a-Flash (hot-page selection for NVMe inference) + Windows working-set management (OS LRU).
Value-add: zero-cost oracle (OS already computed it) vs activation-statistics oracle (requires forward passes). Structural novelty is the "use the OS's own already-computed LRU state" inversion.

**Risk:** VirtualLock limit (192 KB default per process). The Stage 1 proposal correctly notes the `--mlock` workaround in llama.cpp. No CF9/CF10 issues.

**Verdict: REFINE** — 1-hour measurement, substrate-novel, novel oracle principle.

---

### GTBP — GGUF Tensor Offset B-Tree as Page-Fault Sequencer

**Summary treatment (~400w):**

Claims: IOCP 2-layer lookahead achieves ≥ 1.3× layer-delivery latency reduction vs synchronous reads. CF9 clean (IOCP is a documented Win32 primitive, preconditions met). CF10 clean. Smallest test: 2 hours (50-line C harness). Wall-clock within 8h gate.

Prior-art: Search confirmed no IOCP lookahead in mainline llama.cpp or ik_llama.cpp. Recent NVMe prefetch research (Prima.cpp, 2025) uses layer-level prefetch in heterogeneous clusters but not IOCP-based explicit async I/O for single-host consumer NVMe. NOVEL implementation.

**CF9 check:** IOCP precondition: single-file sequential reads with aligned offsets. GGUF tensors ARE stored at sequential file offsets per the format spec; IOCP reads can be issued as deterministic (offset, length) pairs. Precondition fully met.

Two-paper pair: llama.cpp GGUF mmap access + Win32 IOCP documentation.
Value-add: B-tree schedule derived from GGUF offset table enables deterministic 2-layer lookahead with zero misprediction cost (unlike branch-predictor-style methods).

**Risk:** ik_llama.cpp mmap integration requires bypass; harness-only test is sufficient for go/no-go. Integration only if GO.

**Verdict: REFINE** — 2-hour implementation, deterministic lookahead, substrate-novel in LLM inference context.

---

## TRACK A ARCHITECTURE PROPOSALS

---

### A-TROP — Tropical Weight Dispatch

**Summary (~400w):**

Claims: ≥ 10% of W_down output rows have stable dominant input channel (argmax stability ≥ 0.85 across calibration tokens). No imported theorem — pure combinatorial structure (tropical algebra = max+addition). CF9 clean. CF10 clean. Smallest test: 30 min.

Prior-art (Stage 2 pressure-test): Is tropical max-dominance a re-description of LLM.int8() / SmoothQuant? **Stage 2 correctly identifies the distinction:** LLM.int8() identifies outlier channels in the ACTIVATION OUTPUT (which output channels of h_mlp are large); A-TROP identifies stable dominant INPUT channels for each W_down OUTPUT ROW (which input j dominates W_down[i,j] × h_mlp[j] for row i). These are different axes: A-TROP is about W_down row-input structure, LLM.int8() is about W_down output structure. The distinction is real. Search found no paper using tropical semiring for LLM weight-channel dispatch. NOVEL framing.

Two-paper pair: LLM.int8() (activation outlier isolation) + Tropical Semiring (mathematical structure).
Value-add: tropical max applied to W_down INPUT-channel stability per output row — orthogonal axis from LLM.int8()'s output-channel focus. First use of tropical algebra for weight dispatch in LLM inference.

**Risk:** Post-SwiGLU h_mlp is token-dynamic (CF3 analogy), so stable dominant rows may be rare. If stable fraction < 5%, the mechanism provides negligible residency gain. But even the measurement (30 min) characterizes W_down's activation-input structure for the first time on Qwen3, which informs RAOK's tier-2 design.

**Verdict: REFINE** — novel framing, cheap measurement, either outcome is informative.

---

### A-CADH — Content-Addressable Attention Head Dispatch

#### 1. Load-bearing claims

M1: Within-layer pairwise W_Q head cos-similarity ≥ 0.80 for ≥ 50% of pairs at L14.
M2: ≥ 4 heads form a cluster with a single representative (≥ 4× W_Q fetch reduction for those heads).

#### 2. Prior-art status

**M1 (within-layer head similarity):** ADJACENT. "Head-wise Shareable Attention" (arXiv:2402.11819) measures head similarity for KV sharing. CoSpaDi / "Share Your Attention" (killed at R2 for cross-head basis sharing) is ADJACENT. However: post-training per-head W_Q matrix similarity (cosine of full weight matrices, not just KV) for content-addressable dispatch has no direct prior. The combination of BLAKE3-keyed dispatch + SVD-top-32 content hash is novel engineering. **ADJACENT overall.**

**M2:** PARTIAL. CF11's global K=128 finding is ambiguous — it could arise from diverse-but-projecting-to-same-subspace heads rather than similar heads. M2's value-add depends entirely on M1's measurement.

#### 3. Frame-mismatch (CF9)

BLAKE3 on SVD top-32 as a content key: precondition is that SVD top-32 is a stable identifier of a matrix's "action" and that matrices with similar top-32 singular structures produce similar outputs. In attention, head function depends on fine-grained W_Q structure, not just top-32 singular values. A matrix with similar top-32 but different lower singular structure may produce qualitatively different attention patterns. CF9 concern: the BLAKE3 content-hash-as-key is imprecise as a functional equivalence certificate. This does NOT kill the measurement (M1), but does weaken the content-addressable dispatch claim (M2). Mitigation: use full matrix cosine similarity (not BLAKE3) for the measurement; use BLAKE3 only for the dispatch engineering step.

#### 4. Smallest-test sharpening

Script: `scripts/acadh_head_cosim.py`
Extract W_Q[L14] head submatrices (16 × 128 × 2048); compute 16×16 pairwise cos-similarity matrix. ~10 min.
Output: `experiments/stage0/ladder_v2/round4_acadh/`
Wall-clock: 10 min. **Within 8h gate.** Fastest binary settler in this run.

#### 7. Go/no-go thresholds

GO: ≥ 50% of pairs with cos-sim ≥ 0.80 AND ≥ 1 cluster of ≥ 4 heads.
NO-GO: mean cos-sim < 0.30 → heads are diverse within-layer → CF11's global K=128 arises from joint subspace, not head similarity → closes within-layer sharing axis.
GRAY: clusters exist but cos-sim < 0.80 → weaker sharing threshold → ≥ 2× (not 4×) fetch reduction.

#### 8. Risk profile

R1: CF11's global K=128 may arise from 16 orthogonal 8-dim heads (diverse, not similar). The 10-min measurement settles this definitively.
R2: "Head-wise Shareable Attention" (2402.11819) may pre-empt the head-similarity measurement for KV cache. For W_Q weight matrix similarity specifically (not KV), it's less clear. Distinguish: CADH measures W_Q weight matrix similarity, not KV value similarity.

#### 10. Verdict

**REFINE** — 10-min measurement, decisive disambiguation of CF11, content-addressable dispatch is a genuinely novel engineering mechanism even if head-similarity is ADJACENT. CF9 concern noted but doesn't kill the measurement.

---

### A-ALOG — Append-Only Log Attention

#### 1. Load-bearing claims

M1: RoPE attention weights decay geometrically as λ^d for measurable λ, R² ≥ 0.80 on Qwen3 long-context.
M2: Cold-journal tier approximation gives ΔNLL ≤ 0.3 nats vs full-precision attention at 4K context.

#### 2. Prior-art status

**M1 (geometric decay measurement):** NOVEL empirically on Qwen3. Search found: "Rope to Nope and Back Again" (arXiv:2501.18795) discusses RoPE and NoPE hybrid attention; "RoPE does not ensure attention decay with relative distance" per search results. This PARTIALLY challenges M1's premise. If RoPE attention does NOT reliably decay geometrically (depends on learned Q/K alignment), the λ^d model may have low R², and M2 fails. However: the empirical measurement on Qwen3 is still NOVEL regardless of the R² outcome.

**M2 (cold-journal tier):** ADJACENT. StreamingLLM retains attention sinks for long context; H2O / SnapKV perform KV eviction based on attention scores; ClusterAttn uses 1024 tokens with 10-65% memory reduction. A-ALOG's append-only structure and hierarchical tier with running summary is different from eviction-based methods but is in a crowded long-context KV compression field. The append-only monotonicity constraint is a novel framing device but doesn't create a structurally new mechanism beyond hierarchical KV compression.

**Two-paper composition flag:** Closest pair: StreamingLLM (attention-sink KV retention) + KV-Latent (hierarchical KV compression, arXiv:2507.11273). Value-add: append-only constraint eliminates eviction, forcing the geometric-decay summary which is derivable from RoPE structure. The constraint-derived structure is the value-add; if M1 doesn't hold (RoPE doesn't produce geometric decay), the cold-journal summary degrades to a heuristic, which is just StreamingLLM + compression.

#### 3. Frame-mismatch (CF9)

M1's geometric-decay model: the theoretical basis is RoPE position encoding → increasing rotation angle → decreasing Q·K dot product for distant tokens. This is a reasonable prior but empirically is NOT guaranteed (search confirmed RoPE does not ensure decay when Q/K are not random). CF9 concern: the theoretical decay argument imports the "random Q/K vectors" assumption, which doesn't hold for trained models. The measurement (R² fit) resolves this empirically.

#### 4. Calibration ill-conditioning (CF10)

M1 fitting a decay curve λ: 1 parameter fitted to 100 attention entropy curves. n_params = 1, n_samples = 100 × context_length >> 1. CF10 clean.

#### 5. Smallest-test sharpening

Script: `scripts/aaLog_attention_decay.py`
100 long-context forward passes at 4K context. Extract attention weights per layer. Fit λ per layer. ~3 hours.
Output: `experiments/stage0/ladder_v2/round4_aaLog/`
Wall-clock: 3 hours. **Within 8h gate, but 3-hour investment.** Pre-check: compute R² for 5 prompts (20 min) before committing to 100.

#### 7. Go/no-go thresholds

GO: mean R² ≥ 0.80 across layers → geometric decay model sufficient for cold-journal summary.
NO-GO: R² < 0.50 → attention patterns are NOT geometrically decaying in Qwen3 (content-dependent, non-monotone); cold-journal summary will be lossy → revise M2 to content-based compression.

#### 10. Verdict

**REFINE** — but lower priority than algebraic proposals given 3-hour investment and crowded long-context KV field. Stage 5 should slot this after F-AERO-Q3, F-WVOS, WDRS, and A-CADH.

---

### A-REGM — Register-Only Residual Decode

#### 1. Load-bearing claims

M1: Qwen3-1.7B inference can complete with < 512 MB peak RAM while producing correct output.
M2: The experiment re-derives v2's CF13 (NVMe read bytes per token) as a v2-native measurement.

#### 2. Prior-art status

**M1:** ADJACENT. Apple LLM-in-a-Flash (arXiv:2312.11514) uses NVMe offload with RAM staging buffer. REGM eliminates the RAM staging buffer. No published paper derives the minimum RAM floor for transformer inference by making RAM=0 a design constraint. The analysis object (compute schedule at NVMe bandwidth floor) is NOVEL.

**M2:** NOVEL in v2 context. Re-derives CF13 from scratch, which is valuable for the pipeline's epistemic moat.

#### 3. Frame-mismatch (CF9)

No imported theorem. CF9 clean.

#### 4. Calibration ill-conditioning (CF10)

No fit. CF10 clean.

#### 5. Smallest-test sharpening

Script: Standalone C binary (linked against ggml directly).
Single-layer L3-mlock GEMV streaming test. Peak RSS via Windows Working Set API. ~4 hours.
Output: `experiments/stage0/ladder_v2/round4_aregm/`
Wall-clock: 4 hours. **Within 8h gate, but 4-hour implementation investment.** Pre-check: implement 1-layer test in 1 hour before full cascade.

#### 7. Go/no-go thresholds

GO: peak RSS ≤ 512 MB AND output matches.
NO-GO: minimum practical RAM > 512 MB (OS, Python, or ggml allocator requirements) → documents hardware floor.

#### 10. Verdict

**REFINE** — structural investigation value (v2-native CF13 re-derivation) justifies the 4-hour investment. Not the highest-priority experiment but provides unique bottleneck characterization.

---

### A-SEED — 10 MB Seed Weight Synthesis

#### 1. Load-bearing claims

M1: A 256-dim per-matrix θ vector + 2-layer MLP F achieves cos-sim ≥ 0.85 on W_Q for Qwen3-1.7B.
M2: The synthesis generalizes to held-out W_Q from Qwen3-0.6B (cos-sim ≥ 0.75).

#### 2. Prior-art status

**M1/M2:** NOVEL frame. HyperNetworks (Ha et al. 2016) generate weights for small networks; inverse-direction synthesis of billion-parameter LLMs from a 10 MB seed has no positive published result. This is in speculative territory. A4=1 in Stage 2 (quality bound not analytically derivable).

#### 3. Frame-mismatch (CF9)

The "hypernetwork can synthesize trained weights" assumption has no theoretical grounding. The synthesis quality depends entirely on the empirical structure of trained Qwen3 weights. CF8/CF9 don't apply formally, but the mechanism is ungrounded until the experiment runs.

#### 4. Calibration ill-conditioning (CF10)

Fitting F on 28 × W_Q matrices (28 × (128 × 2048) = 7.5M parameters in the targets) with a 2-layer MLP (width 512 ≈ 2M parameters). n_params_to_fit ≈ 2M. n_independent_samples = 28 matrices × 2048 output dims = 57K. With 256 θ dims per matrix: 28 × 256 floats = 7168 floats (θ space) + 2M MLP params. The fit has MLP_params=2M >> n_matrices=28 in the θ-input regime. This is effectively an overfitting scenario — the MLP can memorize 28 matrices exactly and fail to generalize. **CF10 concern: n_params (2M MLP) >> effective n_independent_matrices (28)**. Mitigation: use a much smaller F (e.g., width 64, ~100K params), ensuring n_params << n_matrices × d_out = 28 × 2048 = 57K. Force low-rank F by construction.

#### 5. Smallest-test sharpening

Script: `scripts/aseed_hypernetwork.py`
Fit F_fixed (smaller MLP, width 64, ~100K params) on W_Q[0..27]. Evaluate on held-out Qwen3-0.6B W_Q. ~2 hours.
Output: `experiments/stage0/ladder_v2/round4_aseed/`
Wall-clock: 2 hours. **Within 8h gate.**
**REQUIRES-MITIGATION:** Force F to width 64 (not 512) before running to satisfy CF10. Stage 1 proposal's width-512 F is CF10-underdetermined.

#### 7. Go/no-go thresholds

GO: held-out cos-sim ≥ 0.75 → hypernetwork is generative-capable, pipeline value high.
NO-GO: cos-sim < 0.50 → confirms trained weights are not hypernetwork-synthesizable at this dim count.

#### 10. Verdict

**REFINE — REQUIRES-MITIGATION** on CF10 (reduce F from width-512 to width-64). The frame is genuinely novel; the experiment is informative either way. Flag A4=1 risk and report cal-vs-eval cos-sim gap.

---

## KILL_LIST ENTRIES FROM THIS STAGE

---

**v2-S3-R004-001 — F-GFQO (gauge-freedom general rotation):** DOWNGRADE. General orthogonal R is incompatible with RMSNorm per-channel scale (CF9 partial failure). Surviving mechanism (channel-importance permutation) is ADJACENT to SmoothQuant / activation-aware quantization. Not a Stage 5 target. Stripped primitive: per-channel attention-subspace importance ordering for mixed-precision W_Q block assignment → pass to Stage 4 as a quantization-alignment mechanism.

---

## SUMMARY TABLE

| ID | Verdict | Reason | Priority for Stage 5 |
|---|---|---|---|
| F-AERO-Q3 | REFINE | NOVEL, 10 min, decisive binary, algebraic-identity | Top-1 |
| F-WVOS | REFINE | NOVEL, 30 min, algebraic-identity, top score | Top-2 |
| A-CADH | REFINE | NOVEL, 10 min, disambiguates CF11 | Top-3 |
| WDRS/F-WNORM | REFINE | CF8 closure, 35 min, ADJACENT M3 noted | Top-4 |
| C-ODWQ | REFINE | NOVEL, 10 min, uses existing data | Top-5 |
| AWCJR | REFINE | PARTIAL M1/M2 (LatentLLM adj.), 40 min | High |
| VOKV | REFINE | PARTIAL M1 (Spectral-Lifecycle adj.), KV algebraic novel | High |
| A-TROP | REFINE | NOVEL framing, 30 min, tropical algebra first-use | High |
| A-GFWV | REFINE | PARTIAL (LatentLLM adj.), product-spectrum distinct | High |
| C-AQOD | REFINE | NOVEL, 20 min, CF3×CF11 | High |
| C-VKSD | REFINE | NOVEL, 30 min | High |
| C-RBHD | REFINE | NOVEL, CF9 routing-stability concern flagged | Medium |
| F-SPECA | REFINE | NOVEL, GQA normalization correction required | Medium |
| ULHC | REFINE | NOVEL (Qwen3-8B), 60 min, model download needed | Medium |
| RATJ | REFINE | NOVEL interaction coeff., sequence-gated on RAOK | Medium |
| L1AW | REFINE | NOVEL coupling, 5-min gate, shares infra | Medium |
| HRQK | REFINE (wildcard) | NOVEL zero-shot, CF11 reconciliation in plan | Medium |
| WSFI | REFINE (wildcard) | NOVEL substrate, 2h, SuperFetch registry check first | Medium |
| NNPA | REFINE (wildcard) | NOVEL consumer substrate, 2h, drive support conditional | Medium |
| C-WVKQ | REFINE (wildcard) | PARTIAL (LatentLLM adj.), GQA correction needed | Lower |
| NCZC | REFINE | NOVEL substrate, 30 min | Medium |
| WWTE | REFINE | NOVEL oracle principle, 1h | Medium |
| GTBP | REFINE | NOVEL IOCP implementation, 2h | Medium |
| A-ALOG | REFINE | ADJACENT (crowded KV field), 3h, M1 not guaranteed | Lower |
| A-REGM | REFINE | ADJACENT Apple Flash, v2-CF13 re-derivation value | Lower |
| A-SEED | REFINE (REQUIRES-MITIGATION) | CF10 fix required (reduce F width), NOVEL frame | Lower |
| F-GFQO | DOWNGRADE | CF9 partial fail (RMSNorm breaks general rotation) | — |

**Totals:**
- REFINE: 25 (including 3 wildcards, 1 REQUIRES-MITIGATION)
- DOWNGRADE: 1 (F-GFQO)
- REGENERATE: 0
- KILL: 0

**KILL_LIST additions from Stage 3: 1 downgrade (F-GFQO general gauge rotation).**

---

## TOP CANDIDATES FOR STAGE 5

**Top-1: F-AERO-Q3** — Cheapest decisive binary (10 min), fully NOVEL on Qwen3, algebraic-identity elegance class, closes a known open question from the kill list, high strategic value either outcome.

**Top-2: F-WVOS** — Top run score (12), algebraic-identity fold, 30-min test, NOVEL Qwen3-specific measurement, differentiates from LatentLLM/MHA2MLA by being zero-shot pure-SVD.

**Top-3: A-CADH** — 10-min decisive binary, disambiguates CF11 structure, content-addressable dispatch is novel engineering mechanism, high residency payoff if GO (14 GB at 70B).

**Secondary cluster:** C-ODWQ (10 min, existing data), C-AQOD (20 min, existing data), WDRS/F-WNORM (35 min, pipeline closure) — these three are cheapest structural findings and should be co-scheduled.
