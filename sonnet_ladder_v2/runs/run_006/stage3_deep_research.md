# Stage 3 — Deep Research — Run 006
**Researcher**: Sonnet claude-sonnet-4-6 | **Date**: 2026-05-09
**Kill-list enforced**: v2-CHEAP-TEST-001 (cross-layer W_Q class); v2-S3-R004-001 (arbitrary O(d) rotation / RMSNorm-gauge break). CF9 hard gate applied throughout. All searches date-stamped 2026-05-09.

---

## A. Path 1 Track A — C-JOINT-DELNLL

### 1. Load-bearing claims

M1: W_Q and W_K truncation errors interact via a cross-term ΔW_Q · ΔW_K^T in attention logit space; the cross-term is second-order.
M2: The principal input subspaces of W_Q and W_K share high-variance directions (because both operate on the same residual stream h), so truncation residuals ΔW_Q and ΔW_K may be correlated.
M3: subspace_overlap(V_Q^K, V_K^K) measured in weight space predicts the sign of the interaction term δ = ΔNLL_joint − (ΔNLL_Q + ΔNLL_K).
M4: The activation-weighted subspace overlap (W_Q·H and W_K·H against calibration activations H) is more diagnostically valid than the pure weight-space overlap when weights operate on a structured input distribution.

### 2. Per-claim prior-art status

**M1 — ADJACENT**. LatentLLM (MERL, 2025, TR2025-075) performs Tucker decomposition for joint QK compression, addressing multi-matrix interaction in attention compression. However it does not specifically derive or measure the interaction coefficient δ = ΔNLL_joint − (ΔNLL_Q + ΔNLL_K) as a structural prediction from subspace overlap. No paper found that predicts δ from V_Q^T V_K subspace overlap and then verifies the prediction empirically. Date-stamped: no exact coverage as of 2026-05-09.

**M2 — NOVEL**. The claim that both W_Q and W_K operate on the same residual stream and therefore share high-variance input directions — making truncation errors of the two matrices correlated — is unverified in published literature. Closest cousin: the Q/K functional asymmetry finding in arXiv:2604.22778 (Spectral Lifecycle, April 2026), which shows Q and K develop different spectral profiles during training, implying they may specialize on different subspaces — which would predict low subspace_overlap and hence δ ≈ 0 (constructive for the proposal). Date-stamped: subspace overlap correlation prediction untested as of 2026-05-09. `elegance-class: subspace-alignment`

**M3 — NOVEL** (with caveat). The prediction "subspace overlap predicts sign of δ" is a specific testable claim not found in literature. The caveat (per Stage 2): weight-space overlap is a proxy for the activation-space overlap that actually matters. Date-stamped: not found as of 2026-05-09.

**M4 — ADJACENT**. Activation-weighted subspace overlap (measuring W·H rather than W alone) is the activation-aware variant implicit in Hessian-weighted methods (GPTQ, AWQ). Those methods use H^T H as a weighting matrix for quantization, not for predicting cross-matrix truncation interactions. The proposal adds the interaction-prediction framing on top. ADJACENT to GPTQ but with a distinct structural purpose.

**Two-paper composition flag**: LatentLLM (joint QK Tucker decomposition) + AQFKV (CF11, our own per-layer W_Q/W_K rank finding) ≈ "jointly compress W_Q and W_K." Value-add: the interaction coefficient δ prediction from subspace_overlap is not in either paper; LatentLLM minimizes joint reconstruction error without predicting its decomposition into independent+interaction terms; CF11 measures individual ranks but not joint interaction. The structural insight — that the sign of the interaction is predictable from the weight-space geometry before running the joint experiment — is the novel primitive.

### 3. Frame-mismatch check

No theorem imported with preconditions. The cross-term ΔA = (ΔW_Q h)(W_K h)^T + (W_Q^K h)(ΔW_K h)^T is a first-order Taylor expansion of the attention logit error; the second-order cross-term (ΔW_Q h)(ΔW_K h)^T is small when individual truncation errors are small. This is not an imported theorem — it is direct algebra. Preconditions: individual truncation errors small relative to signal (satisfied at K_Q=512: ΔNLL=+0.20 nats). No CF9 risk.

**Stage 2 pressure-test resolution**: The flagged concern is whether weight-space subspace_overlap matches the activation-space alignment that determines actual logit-error correlation. The proposal already identifies this as the primary risk (M4) and proposes the activation-weighted variant. Resolution: run both. The activation-weighted version (W_Q H)^T (W_K H) is the correct predictor; if it disagrees with the weight-space version, the activation-weighted result governs.

### 4. Calibration ill-conditioning

No calibration fitting; measurement only (SVD of stored factors from AQFKV + forward pass with joint truncation). CF10 not applicable.

### 5. Residency arithmetic

Qwen3-1.7B: joint truncation at K_Q=K_K=512 saves (via CF11 data):
- W_Q: 2048×2048 → 2048×512+512×2048 = 4.19 MB/layer × 28 = 117 MB saved (vs 235 MB baseline)
- W_K: 2048×512 → (GQA shape 1024×2048 in Qwen3-1.7B) → 1024×512+512×2048 = 1.57 MB/layer × 28 = 44 MB saved (vs 57.4 MB baseline W_K)
- Total W_Q+W_K saving: ~161 MB on Qwen3-1.7B
- Quality cost if additive: ΔNLL_Q(K=512) + ΔNLL_K(K=512) ≈ 0.20 + 0.29 = 0.49 nats
- Quality cost if super-additive (δ > 0): up to 0.49 + δ nats; the measurement determines δ
- 70B projection: W_Q (10.74 GB) + W_K (1.34 GB) combined saving at K=512: W_Q → 2.7 GB + W_K → 0.34 GB = 3.04 GB from ~12 GB total Q+K; combined Q+K residency 3.04 GB + other = enables DRAM-residency at 7.28 GiB when stacked with NanoQuant MLP

### 6. Smallest test (refined)

**Script**: `experiments/stage0/ladder_v2/round6_joint_delnll/run_joint_qk.py`
- Model: Qwen3-1.7B-Base BF16
- Calibration: None (weight-only measurement for subspace overlap; WikiText-2 512 tokens for ΔNLL)
- Step 1 (~2 min): Load AQFKV factors V_Q^512, V_K^512 at L14. Compute ‖V_Q^T V_K‖_F² / 512 (weight overlap). Compute (W_Q H)^T (W_K H) normalized (activation overlap).
- Step 2 (~10 min): Apply joint K_Q=K_K=512 truncation all 28 layers, compute ΔNLL on 512-token WikiText-2 held-out. Compute δ = ΔNLL_joint − (0.20 + 0.29).
- Output: activation-weighted subspace_overlap, weight-space overlap, δ.
- Wall-clock: ~12 min total on Ryzen 5 7530U (SVD already done in AQFKV).
- Output path: `experiments/stage0/ladder_v2/round6_joint_delnll/`

### 7. Go/no-go thresholds

**GO**: |δ| ≤ 0.10 nats AND activation-weighted overlap-sign matches δ sign. Advance to full 28-layer joint cascade at K_Q=512, K_K=512 as a free compression step on top of CF11.
**NO-GO**: δ > 0.20 nats (super-additive destructive coupling). Class-level finding: W_Q and W_K share principal input directions; joint compression requires re-budgeting (K_Q must be raised when W_K is simultaneously compressed). Report interaction coefficient per-layer as a structural table.
**GRAY**: 0.10 < δ ≤ 0.20 nats. Follow-up: per-layer overlap vs per-layer δ plot — determine if specific depth ranges drive the interaction. ~20 min additional sweep at K ∈ {256, 512, 1024} per layer.

### 8. Risk profile

| Risk | Mitigation |
|---|---|
| Weight-space subspace_overlap ≠ activation-space overlap | Run both; activation-weighted version governs if they disagree |
| δ is super-additive, forcing quality re-budget | Structural finding; report interaction coefficient table; adjust K upward for W_K when W_Q is simultaneously compressed |
| AQFKV factors are from Qwen3-1.7B only; 70B interaction coefficient could differ | Note as a generalization caveat; 1.7B result is a directional signal for 70B scheduling |

**Cheapest falsifier**: measure δ at K_Q=K_K=512; if δ > 0.30 nats at single-layer level the cascade must redesign the joint budget.

### 9. Verdict: **REFINE**

Strong CF11 grounding, cheap test, additive outcome is a free win, super-additive outcome is a useful structural constraint. Advance to Stage 4/5. Top-2 candidate.

---

## B. Path 1 Track A — R6-WVOP

### 1. Load-bearing claims

M1: The product W_VO = W_V @ W_O is computable once at model load and replaces two GEMVs with one at inference (AERO-style fold on V-O chain).
M2: rank(W_VO) ≤ min(rank(W_V), rank(W_O)) — the product's rank is bounded by the weaker factor.
M3: W_V and/or W_O have r_99/d substantially below 1.0 (like W_Q/W_K, not like MLP weights), enabling low-rank W_VO storage with acceptable ΔNLL.

### 2. Per-claim prior-art status

**M1 — ADJACENT**. AERO (arXiv:2410.13060) folds W_gate into W_up on the MLP side by removing the activation. The attention V-O fold is structurally analogous but has a different algebraic basis (no activation removal needed — the V-O chain is linear). No paper found that explicitly computes W_VO = W_V @ W_O and stores the product for inference on any attention-based architecture. Date-stamped: 2026-05-09. `elegance-class: algebraic-identity`

**M2 — NOVEL**. The product rank bound is a matrix-algebra fact. No paper applies this specifically to W_V @ W_O. Date-stamped: 2026-05-09.

**M3 — PARTIAL** with new prior art. arXiv:2604.22778 (Spectral Lifecycle, April 2026) presents a significant finding directly bearing on M3: "Value/output projections compress uniformly while query/key projections carry the full depth-dependent dynamics. This asymmetry suggests that attention selection is the adaptive computation while value transformation is more generic." This paper measures the spectral dynamics of V and O projections during training and finds they compress UNIFORMLY — meaning they have MORE concentrated spectra than Q/K during training. This is evidence that W_V and W_O may be MORE compressible than W_Q/W_K, not less. This upgrades M3's probability substantially. However 2604.22778 does not measure W_V and W_O spectra in Qwen3 post-training; it analyzes training dynamics on GPT-2/Pythia. The Qwen3-specific measurement remains open. Status: PARTIAL — uniform spectral compression during training is precedented for V/O; Qwen3-specific r_99/d measurement is novel.

**Two-paper composition flag**: AERO (V-O analogy) + arXiv:2604.22778 (V/O spectral uniformity) ≈ "fold V-O chain exploiting V/O spectral concentration." Value-add: the product-rank argument (M2) means W_VO can be further compressed below either factor alone; the fold + rank compression combination is not in 2604.22778. Structural insight beyond composition: rank(W_VO) ≤ min(rank(W_V), rank(W_O)) means even if W_V has moderate rank, W_O may impose a tighter bound. The fold provides a compute reduction (2 GEMVs → 1) independent of the rank argument.

### 3. Frame-mismatch check

No theorem imported. W_VO = W_V @ W_O is exact linear algebra. The GEMV fold works because the attention output h = A(XW_V)W_O = A · X · (W_V W_O) for fixed W. No precondition issues.

**Critical note on GQA shape**: In Qwen3-1.7B with GQA, W_V has shape (n_kv_heads × d_head) × d_model = 1024 × 2048, and W_O has shape d_model × (n_heads × d_head) = 2048 × 2048. The product W_V @ W_O has shape 1024 × 2048 (same as W_V). At 70B: W_V shape is 1024 × 8192, W_O shape is 8192 × 8192; product shape 1024 × 8192. The fold reduces W_V + W_O storage (1024×2048 + 2048×2048 = 6.3 MB/layer) to W_VO storage (1024×2048 = 4 MB/layer) plus the SVD factorization benefit. The shape arithmetic holds; no CF9 issue.

### 4. Calibration ill-conditioning

No calibration fitting. Pure weight arithmetic + ΔNLL measurement. CF10 not applicable.

### 5. Residency arithmetic

Qwen3-1.7B: W_V (1024×2048) + W_O (2048×2048) = 4 MB + 8 MB = 12 MB/layer × 28 = 336 MB.
W_VO at r_99/d = 0.60 (optimistic, similar to W_Q): K=512: (1024×512 + 512×2048) × 2B = 3.15 MB/layer × 28 = 88 MB. Save 248 MB.
W_VO at r_99/d = 0.79 (conservative, like W_K): K=810 captures 99%: ≈ (1024×810 + 810×2048) × 2B = 5.0 MB/layer × 28 = 140 MB. Save 196 MB.
Even in the conservative case: 196 MB saving on a 1.86 GB model = 10.5%.
70B: W_V (GQA 1024×8192) + W_O (8192×8192) = 1.34 GB + 10.74 GB = 12.08 GB → W_VO compressed: 0.84 GB (from Stage 2 arithmetic). Save 11.24 GB — this is pivotal.
Compute benefit: 2 GEMV per layer → 1 GEMV per layer for V-O chain. ~10% inference speed gain at 1.7B where attention is ~20% of bandwidth.

**CF13/CF15 branch**: No CF13-15 dependence. Only CF11 boundary (attention matrices more compressible) grounded the prior probability; 2604.22778 independently confirms V/O spectral uniformity as an additional anchor.

### 6. Smallest test (refined)

**Script**: `experiments/stage0/ladder_v2/round6_wvop/run_wvop.py`
- Model: Qwen3-1.7B-Base BF16
- Step 1 (~5 min): Compute W_VO[14] = W_V[14] @ W_O[14]. SVD. Record r_99/d, var@{256,512,810}.
- Step 2 (~20 min): Replace W_V[14] and W_O[14] with W_VO[14] SVD at K=256/512; run ΔNLL on 512-token WikiText-2.
- Wall-clock: ~25 min on Ryzen 5 7530U.
- Output path: `experiments/stage0/ladder_v2/round6_wvop/`

### 7. Go/no-go thresholds

**GO**: r_99(W_VO[14])/d < 0.75 AND ΔNLL(K=512) < 0.80 nats. Proceed to full 28-layer W_VO sweep.
**NO-GO**: r_99 > 0.90 (W_O full-rank as feared). Class-level finding: attention compression ceiling is W_Q + W_K only; W_V/W_O resist post-training SVD truncation. 70B residency plan falls back to CF11-only attention compression.
**GRAY**: 0.75 ≤ r_99 ≤ 0.90 AND ΔNLL(K=512) 0.80–1.50 nats. Investigate whether the W_VO product has a tighter spectrum than W_V alone (rank may compound, compressing beyond either individual factor). Test K=256 (aggressive) vs K=512 vs K=810 to find the acceptable knee.

### 8. Risk profile

| Risk | Mitigation |
|---|---|
| W_O is full-rank (output projection must span full residual stream from 16 heads) | 25-min spectrum measurement resolves before any implementation; NO-GO is a useful structural finding |
| arXiv:2604.22778 used GPT-2/Pythia, not Qwen3 — spectral patterns may differ | Qwen3-specific measurement is the experiment; 2604.22778 provides prior probability only |
| W_VO product matrix may degrade more at truncation than either factor alone | Measure at multiple K values; the product rank argument predicts it should be BETTER, not worse |

**Cheapest falsifier**: SVD of W_VO[14]; if spectrum is flat (var@512/total < 0.60), the fold-and-compress strategy is dead at this model.

### 9. Verdict: **REFINE**

Elevated prior probability from arXiv:2604.22778 (V/O compress uniformly). Cheap test. 70B prize if W_VO is concentrated. Top-2 candidate.

---

## C. Path 1 Track A — F6-WALIGN

**Path 4 elegant-equivalence, sub-class: algebraic-identity**

### 1. Load-bearing claims

M1: For neurons where cos(W_gate[i,:], W_up[i,:]) ≈ 1, the SwiGLU output is a scalar function of a single direction; W_up[i,:] is redundant.
M2: The fraction of neurons with |cos| ≥ 0.90 is ≥ 5% across 172K neurons in trained Qwen3-1.7B.
M3: For neurons where the alignment holds exactly, the identity y_i = silu(w^T x) · λ_i (w^T x) = λ_i z · σ(z) is algebraically exact with no approximation.

### 2. Per-claim prior-art status

**M1 — ADJACENT**. AERO (arXiv:2410.13060) globally removes the SwiGLU gating activation and folds W_gate into W_up at the matrix level. WALIGN operates per-neuron and KEEPS the SwiGLU activation (it folds only when the two row vectors are collinear). The structural difference is precise: AERO eliminates the activation function globally (changing the computation graph everywhere); WALIGN folds individual neuron row-pairs without touching the activation function for any neuron. The comp. payoff is different: AERO saves one entire weight matrix; WALIGN saves individual rows where the alignment is exact. The two approaches compose (AERO could be viewed as the extreme limit of WALIGN applied when ALL neurons are aligned), but WALIGN's per-neuron granularity is finer and not explored in AERO.

Dependency-Aware Semi-Structured Sparsity of GLU Variants (arXiv:2405.01943) groups W_gate and W_up for structured pruning but does not test cosine alignment between corresponding row pairs. Status: ADJACENT — neither AERO nor 2405.01943 measures per-neuron cos(W_gate[i], W_up[i]) or exploits exact-alignment folds without activation removal. Date-stamped: 2026-05-09. `elegance-class: algebraic-identity` (NOVEL for the per-neuron granularity).

**M2 — NOVEL**. No published paper reports the distribution of per-neuron cosine similarity between W_gate and W_up row pairs in a trained SwiGLU network. Date-stamped: 2026-05-09.

**M3 — NOVEL** (algebraic identity). This is a first-principles algebraic fact; the identity holds exactly when cos = 1. No CF9 risk (no imported theorem). `elegance-class: algebraic-identity`

**Stage 2 pressure-test**: Does SGD favor near-orthogonal (W_gate, W_up) pairs as a feature-maximizing strategy? arXiv:2604.22778 provides indirect evidence on this question: value/output projections (which are roughly analogous to the "combined" gate+up computation) compress uniformly, suggesting gradient descent does not deliberately maximize angular diversity in the gate/up pair. However this is indirect. The PolyGLU paper (arXiv:2603.13347, March 2026) finds "emergent routing behavior — without any explicit sparsity loss, the routing mechanism converges to near-deterministic activation selections." This means neurons DO specialize to specific activation functions, which is consistent with some neurons having near-aligned (gate, up) pairs. The measurement settles this.

**Two-paper composition flag**: AERO (activation-removal fold) + CF8 (MLP weights full-rank, ruling out global SVD compression) ≈ "find per-neuron structure in SwiGLU beyond global rank." Value-add: per-neuron cosine alignment is a structurally different object from global rank (CF8 says rank-truncation of W_gate or W_up fails; WALIGN does not truncate any matrix). The elegant identity is the value-add.

### 3. Frame-mismatch check

No imported theorem. Pure trigonometric identity. No CF9 risk.

### 4. Calibration ill-conditioning

No calibration fitting. Row-wise cosine computation is deterministic. CF10 not applicable.

### 5. Residency arithmetic

Qwen3-1.7B, 172K neurons, 28 layers × 6144 neurons:
- At 5% aligned (|cos| ≥ 0.90): 8600 neurons save one 4 KB row each = 34 MB from W_up
- At 10% aligned: 68 MB
- At 20% aligned: 137 MB
- Full W_gate + W_up: 1.34 GiB; 20% alignment saves ~10% of MLP pair weight storage
- 70B (28672 neurons/layer × 80 layers = 2.3M neurons): at 10% aligned: 230K × 4 KB × 2 = 920 MB saved from W_up; at INT4: 460 MB
- ΔNLL for exactly aligned neurons: 0.00 nats (algebraic identity). For near-aligned (cos ≥ 0.90): small residual norm; expected ΔNLL < 0.05 nats per aligned neuron substitution.

### 6. Smallest test (refined)

**Script**: `experiments/stage0/ladder_v2/round6_walign/run_walign_hist.py`
- Model: Qwen3-1.7B-Base BF16
- Step 1 (~5 min): Load all 28 layers' W_gate and W_up (6144×2048 each). Row-normalize. Compute einsum cosines for all 28×6144 pairs. Histogram with bins [-1, -0.99, -0.90, -0.70, 0, 0.70, 0.90, 0.99, 1.0].
- Output: distribution histogram, fraction with |cos| ≥ 0.90, fraction with |cos| ≥ 0.70.
- Wall-clock: ~5 min (pure tensor arithmetic).
- Output path: `experiments/stage0/ladder_v2/round6_walign/`

### 7. Go/no-go thresholds

**GO**: ≥ 5% of neurons with |cos| ≥ 0.90. Proceed to per-neuron fold for aligned neurons + single-layer ΔNLL measurement.
**NO-GO**: < 1% with |cos| ≥ 0.90 AND mean |cos| ≈ 0 (distribution peaked near 0, consistent with orthogonality-maximizing SGD). Structural finding: SGD allocates the two-matrix degree of freedom in SwiGLU to maximize angular diversity between (gate, up) row pairs. The per-neuron fold is unavailable post-training; compression must come from quantization not from algebraic row-alignment.
**GRAY**: 1–5% with |cos| ≥ 0.90 OR mean |cos| ≈ 0.3–0.5 (moderate alignment throughout). Investigate whether alignment is depth-stratified (could layers 0 or 27 be more aligned than mid-depth?). 10-min per-layer alignment profile.

### 8. Risk profile

| Risk | Mitigation |
|---|---|
| Mean |cos| ≈ 0 (SGD orthogonality) | 5-min measurement settles; even if alignment is near zero, the distribution shape is a structural finding |
| AERO may be considered prior art if Stage 4/5 views WALIGN as "AERO + per-neuron granularity" | Explicitly position WALIGN's value-add: (1) it does not remove the activation function, (2) it composes with non-aligned neurons unchanged, (3) per-neuron scope vs global scope is a structurally distinct operation |
| Near-aligned neurons (cos ≈ 0.95) may have a residual that matters at scale | Measure per-neuron ΔNLL for the residual approximation separately from the exact-fold neurons |

**Cheapest falsifier**: the 5-min histogram. If distribution is peaked near |cos| = 0, the mechanism provides no compression payoff.

### 9. Verdict: **REFINE** (pending histogram)

Path 4 elegant-equivalence claim is structurally sound (algebraic identity is exact). 5-min measurement is the first rung. High information content regardless of outcome. Top-3 candidate.

---

## D. Path 1 Track A — F6-GQARED

### 1. Load-bearing claims

M1: In GQA, the outer sum over KV groups is algebraically discrete (8 independent groups), making per-group elimination exact for groups with zero attention weight.
M2: At least one KV group across any layer has mean total attention weight < 0.05 on a 200-token calibration corpus.
M3: Per-group attention weight is low-variance across tokens (low mean AND low variance makes a group safe to drop).

### 2. Per-claim prior-art status

**M1 — PARTIAL**. Michel et al. (2019, "Are Sixteen Heads Really Better Than One?") proved head pruning in standard MHA. No paper found that exploits GQA's SPECIFIC discrete-sum structure (8 groups sharing KV matrices, each group serving 2 Q heads) for group-level elimination. The distinction: in MHA, heads share W_O rows (each head contributes to all output channels); in GQA, each KV group serves a specific subset of Q heads via the same W_K^g, W_V^g. Dropping a GQA group eliminates W_K^g AND W_V^g cleanly. No MHA head-pruning paper addresses this algebraic structure. Status: PARTIAL — head pruning precedented for MHA; GQA group structure is novel territory. Date-stamped: 2026-05-09.

**M2 — NOVEL** (empirical claim, no published measurement on Qwen3 GQA groups). Date-stamped: 2026-05-09.

**M3 — NOVEL** (per-group variance measurement not found in literature). Date-stamped: 2026-05-09.

**Two-paper composition flag**: Michel et al. 2019 (MHA head pruning) + GQA Ainslie et al. 2023 (GQA architecture) ≈ "prune GQA groups." Value-add: the GQA-specific algebraic structure makes group-dropping EXACT (not approximate as in MHA head pruning, where the W_O contribution is distributed). The M3 variance gate (not just mean weight) is also absent from Michel et al.

### 3. Frame-mismatch check

No imported theorem. M1 is direct algebraic observation about the GQA sum structure. No CF9 risk.

### 4. Calibration ill-conditioning

No calibration fitting. Only measurement of attention weights and ΔNLL. CF10 not applicable.

### 5. Residency arithmetic

Qwen3-1.7B: 8 KV groups, W_K^g (128×2048) + W_V^g (128×2048) per group = 1 MB/group × 8 = 8 MB/layer × 28 = 224 MB total.
Dropping 2 of 8 groups: save 56 MB.
Dropping 1 group: 28 MB.
At 70B (Qwen3-72B, 8 KV heads): W_K + W_V ≈ 1.34 GB + 1.34 GB = 2.68 GB. Dropping 2/8 groups: 670 MB.
Quality cost: exactly 0 for groups with mean weight = 0; near-0 for groups below ε. The empirical question is whether any group clears the threshold.

### 6. Smallest test (refined)

**Script**: `experiments/stage0/ladder_v2/round6_gqared/run_gqa_group_importance.py`
- Model: Qwen3-1.7B-Base BF16
- Calibration: WikiText-2 200 tokens (forward pass with attention weight logging)
- Measure: per-head attention weights per layer → group means and variance (2 Q heads per KV group, 8 KV groups, 28 layers, 200 tokens)
- Sweep: identify any (group, layer) pair with mean weight < 0.05 AND std < 0.10
- Wall-clock: ~15 min (forward pass 200 tokens with attention logging)
- Output path: `experiments/stage0/ladder_v2/round6_gqared/`

### 7. Go/no-go thresholds

**GO**: Any (group, layer) pair with mean weight < 0.05 AND std < 0.10. Zero W_K^g, W_V^g for that group+layer; measure ΔNLL.
**NO-GO**: All groups equipotent (each ≈ 0.125 ± 0.05) across all layers. Class-level finding: Qwen3's GQA groups are uniformly utilized — either training penalizes group-weight collapse, or all 8 groups encode distinct routing specializations. Rules out post-training GQA group elimination on Qwen3.
**GRAY**: Low-mean groups exist but with high variance (mean < 0.05, std > 0.10). Context-dependent groups — safe to drop on most inputs but critical on specific inputs. Follow-up: evaluate on adversarial inputs that would activate the low-mean group; report worst-case ΔNLL. ~30 min.

### 8. Risk profile

| Risk | Mitigation |
|---|---|
| All GQA groups equipotent — training optimizes for balanced group utilization | 15-min measurement settles; finding is structurally valuable regardless |
| Input-dependent variance: dropped group critical on specific inputs | Measure variance; only drop group if std < ε |
| 200-token calibration corpus may under-sample the full distribution | Also test on code vs prose vs instruction-following passages to check cross-domain stability |

**Cheapest falsifier**: 15-min forward pass; if all groups cluster near 0.125, the mechanism is dead on Qwen3.

### 9. Verdict: **REFINE**

Clean algebraic structure, fast measurement, 70B payoff if confirmed.

---

## E. Path 1 Track B — C-WVOK

### 1. Load-bearing claims

M1: CF11's head-redundancy finding (16 Q heads → 128-dim joint subspace) implies corresponding compression of the value subspace: dim(effective_value_subspace) ≤ 128.
M2: r_99(W_V) / d ≤ 0.75 at L14 (more concentrated than W_K).
M3: ΔNLL at K_V=512 < 0.40 nats.

### 2. Per-claim prior-art status

**M1 — NOVEL** (coupling equation derived from CF11; not in published literature). The argument that W_Q head-redundancy implies W_V subspace collapse because value routing tracks query routing is a pipeline-internal deduction from CF11. No published paper derives W_V rank from W_Q head-redundancy. Date-stamped: 2026-05-09.

**M2 — PARTIAL**. arXiv:2604.22778 (Spectral Lifecycle) provides indirect support: V/O projections compress uniformly during training, suggesting r_99/d may be lower for W_V than for W_Q/W_K. But this is training-time spectral behavior, not post-training measurement. PARTIAL — training dynamics support the prediction; post-training Qwen3-specific measurement is novel. Date-stamped: 2026-05-09.

**M3 — NOVEL** (no Qwen3 W_V rank vs ΔNLL published). Date-stamped: 2026-05-09.

**Per-head vs full-matrix concern from Stage 2**: If per-head W_V blocks (128×2048 each) are individually full-rank but their union is concentrated, the full-matrix result governs. The 2604.22778 finding (V/O compress uniformly, i.e., they have lower variance-weighted spectral energy at any given rank) supports the full-matrix spectrum being concentrated. The per-head block test is worth running as a sub-check within the same experiment.

**Two-paper composition flag**: CF11 (head-redundancy, our own) + arXiv:2604.22778 (V/O spectral uniformity) ≈ "W_V should be compressible." Value-add: the coupling equation M1 derives a specific quantitative prediction for W_V's rank from CF11's K_Q=128 measurement; 2604.22778 provides only a qualitative prediction from training dynamics. The quantitative prediction (dim(effective_value_subspace) ≤ 128) is novel.

### 3. Frame-mismatch check

No imported theorem. Coupling equation is a derived inequality from head-redundancy; derivation assumes query routing determines value routing, which is a model of attention dynamics, not an imported theorem. The assumption may fail if W_V encodes content-specific routing independent of W_Q. This is the primary risk and is tested empirically by M2/M3.

### 4. Calibration ill-conditioning

No calibration fitting. Pure SVD + ΔNLL measurement. CF10 not applicable.

### 5. Residency arithmetic

Qwen3-1.7B: W_V (1024×2048, GQA) = 4 MB/layer × 28 = 112 MB.
At K=512: (1024×512 + 512×2048) × 2B = 3.15 MB/layer × 28 = 88 MB. Save 24 MB (small on 1.7B).
The primary value is 70B: W_V_GQA at 70B = 1.34 GB → K=256: (1024×256 + 256×8192) × 2B × 80 = 0.37 GB. Save 0.97 GB.
Convergence C2: this measurement simultaneously gates R6-ATTNQ-FULL and R6-WVOP. Running it once resolves three proposals.

### 6. Smallest test (refined)

**Script**: `experiments/stage0/ladder_v2/round6_cwvok/run_wv_spectrum.py`
- Model: Qwen3-1.7B-Base BF16
- Step 1 (~5 min): SVD of W_V[14] (1024×2048). Record r_99/d, var@{128,256,512}.
- Step 2 (~10 min): ΔNLL at K=512 for W_V[14] only.
- Bonus (~5 min): per-head W_V block SVD for one head (128×2048) to check per-head vs full behavior.
- Wall-clock: ~20 min.
- Output path: `experiments/stage0/ladder_v2/round6_cwvok/`

### 7. Go/no-go thresholds

**GO**: r_99(W_V)/d < 0.80 AND ΔNLL(K=512) < 0.40 nats. Proceed to W_O spectrum and full cascade (ATTNQ-FULL path).
**NO-GO**: r_99 > 0.90 (W_V full-rank like MLP). Sharpens CF11: head-redundancy is query-side only; value projection decouples from query routing. W_O also likely full-rank; attention compression ceiling is W_Q + W_K only. Note: even then, 70B residency plan remains viable via W_Q(K=128) + W_K(K=512) saving.
**GRAY**: 0.80 ≤ r_99 ≤ 0.90 AND ΔNLL(K=512) 0.40–1.00 nats. Test K=1024 (conservative truncation) for ΔNLL < 0.20.

### 8. Risk profile

| Risk | Mitigation |
|---|---|
| Per-head W_V blocks individually full-rank even if full-matrix is concentrated | Measure per-head SVD as sub-experiment; governs which approach to use |
| 1.7B spectral findings may not generalize to 70B GQA structure | Flag as generalization caveat; the cheap 1.7B measurement is a directional signal |
| 2604.22778 based on GPT-2/Pythia — different architecture family | Qwen3 measurement is the definitive one |

**Cheapest falsifier**: W_V[14] SVD (~5 min); if flat spectrum, the coupling equation M1 is wrong.

### 9. Verdict: **REFINE**

Cheapest experiment in the run (20 min). Gates three proposals. 2604.22778 provides supportive prior. Advance.

---

## F. Path 1 Track B — R6-ATTNQ-FULL

### 1. Load-bearing claims

M1: W_V and W_O each have r_99/d < 0.85 (gate condition for the cascade).
M2: Joint SVD truncation of all four attention matrices (W_Q, W_K, W_V, W_O) produces at most additive ΔNLL (no super-additive interaction between V/O errors and Q/K errors).
M3: Combined 70B attention residency at K_Q=128, K_K=256, K_V=256, K_O=256 ≈ 1.17 GB (Stage 2 arithmetic).
M4: Total 70B model fits in 7.28 GiB when combined with NanoQuant MLP (~5.35 GB).

### 2. Per-claim prior-art status

**M1 — PARTIAL** (see C-WVOK section above; gated by experiment).

**M2 — NOVEL**. Joint four-matrix interaction coefficient for simultaneous SVD truncation not found. LatentLLM addresses joint QK compression only; no paper measures the Q+K+V+O joint ΔNLL interaction. Date-stamped: 2026-05-09.

**M3/M4 — NOVEL** (70B combined residency argument depends on confirmed W_V/W_O spectra; no published work combines per-matrix SVD truncation for all four attention matrices at 70B scale). Date-stamped: 2026-05-09.

**Two-paper composition flag**: GPTQ/QuIP (quantization of all matrices) + arXiv:2604.22778 (V/O spectral uniformity) ≈ "compress all attention matrices post-training." Value-add: SVD truncation exploits the specific spectral structure (not quantization grid); the four-matrix interaction coefficient is the novel structural primitive not in either paper.

### 3. Frame-mismatch check

No imported theorem. Purely empirical cascade. The super-additivity risk (M2) is an empirical question, not a theoretical one; the experiment directly measures it.

### 4. Calibration ill-conditioning

No calibration fitting. CF10 not applicable.

### 5. Residency arithmetic

See Stage 2 rationale (already detailed). Key: M1 must be confirmed by C-WVOK + WVOP experiments. Conservative branch (W_V/W_O full-rank): falls back to W_Q(K=128)+W_K(K=512) attention saving (~1.3 GB at 70B) — still enough for 7.28 GiB combined with NanoQuant.

### 6. Smallest test (refined)

This proposal is GATED on W_V spectrum from C-WVOK. Its own smallest test (after W_V gate): measure joint ΔNLL for all four matrices simultaneously truncated at (K_Q=128, K_K=512, K_V=512, K_O=512) on Qwen3-1.7B to check super-additivity. ~20 min after gating experiments complete.

### 7. Go/no-go thresholds

**GO**: W_V gate passes (r_99 < 0.85) AND joint 4-matrix ΔNLL < 2.5 nats (sum of individual estimates ≈ 0.98 + 0.29 + ~0.30 + ~0.50 = 2.07 nats; tolerance 20% for interaction).
**NO-GO (gate fail)**: W_V/W_O full-rank. Cascade falls back to 2-matrix (Q+K) compression.
**NO-GO (interaction)**: joint ΔNLL > 4.0 nats (super-additive, unusable). Investigate per-pair interaction first.

### 8. Risk profile

| Risk | Mitigation |
|---|---|
| W_V/W_O full-rank — cascade gate never opens | C-WVOK measurement resolves in 20 min |
| Joint four-matrix ΔNLL super-additive | Measure joint ΔNLL as first cascade step; if super-additive, back off on weakest matrix K first |
| 70B generalization | 70B arithmetic is dependent on 1.7B spectrum profile; if r_99 differs at scale, re-estimate |

### 9. Verdict: **REFINE** (gated on C-WVOK)

The cascade is structurally sound; it advances contingent on W_V gate. Run C-WVOK first.

---

## G. Path 1 Track B — U2-FNDP

### 1. Load-bearing claims

M1: `FILE_FLAG_NO_BUFFERING` bypasses Windows Cache Manager double-copy, eliminating one DRAM-bandwidth pass per byte on sequential weight reads.
M2: The bandwidth saving is ≥15% on 200 MB sequential weight reads (the smallest-test claim).
M3: The integration path into ik_llama.cpp requires switching from mmap to ReadFile with sector-aligned buffers (engineering cost, not a fundamental block).

### 2. Per-claim prior-art status

**M1 — ADJACENT**. MemAscend (arXiv:2505.23254, May 2025) introduces a "Direct NVMe Engine" for LLM fine-tuning on SSDs that bypasses page cache. The fine-tuning use case (MemAscend) and the inference use case (FNDP) are different in workload pattern but the same kernel-bypass mechanism is adjacent. CHEOPS '25 I/O characterization study also examines direct NVMe offloading for LLM inference. However no paper applies `FILE_FLAG_NO_BUFFERING` specifically on Windows for ik_llama.cpp-style inference. The mechanism is ADJACENT — the O_DIRECT / kernel bypass principle is established in the database literature and now in LLM systems; the Windows-specific `FILE_FLAG_NO_BUFFERING` implementation for ik_llama.cpp weight loading appears novel. Date-stamped: 2026-05-09.

**M2/M3 — NOVEL** (specific throughput gain on Windows NVMe path, Qwen3-class weight reads). Date-stamped: 2026-05-09.

**Two-paper composition flag**: MemAscend (direct NVMe for LLM) + llama.cpp mmap path (standard) ≈ "bypass page cache for LLM weights." Value-add: Windows-specific `FILE_FLAG_NO_BUFFERING` implementation details (sector alignment, double-buffer scheme for ik_llama.cpp integration) are not in MemAscend (which targets Linux/CUDA). The engineering specification for Windows + ik_llama.cpp is the novel primitive.

### 3. Frame-mismatch check

No imported theorem. Mechanism is purely Win32 API behavior. No CF9 risk.

### 4. Calibration ill-conditioning

No calibration fitting. CF10 not applicable.

### 5. Residency arithmetic

70B at IQ4_XS: 437 MB/layer. With Cache Manager: NVMe read (3 GB/s, 146 ms) + kernel copy (~38 ms). With FNDP: 146 ms only. ~20% per-layer time reduction. 28 layers: 28 × 38 ms = 1.06 s/token saved. At a 5 s/token baseline for 70B NVMe: ~20% speedup, quality cost 0 nats.
CF13/CF15 note: the 3 GB/s NVMe number is unconfirmed (not v2-confirmed). The experiment itself produces a v2-CF13'' candidate. The NO-GO outcome is that the Cache Manager copy overhead is negligible, which is itself a structural finding.

### 6. Smallest test (refined)

**Script**: standalone C harness, not part of ik_llama.cpp
- Read 200 MB slab from Qwen3-1.7B GGUF in two modes: standard ReadFile (Cache Manager path) and ReadFile with FILE_FLAG_NO_BUFFERING + sector-aligned buffer (direct I/O path).
- Page cache cleared between runs (`NtSetSystemInformation(SystemMemoryListCommand, ...)` or restart).
- Measure wall-clock time for each mode, 5 repetitions each.
- Wall-clock estimate: ~1 hour (harness implementation ~45 min + runs ~15 min on Ryzen 5 7530U).
- Output path: `experiments/stage0/ladder_v2/round6_fndp/`

### 7. Go/no-go thresholds

**GO**: ≥15% throughput improvement (direct vs buffered). Proceed to ik_llama.cpp integration scope assessment.
**NO-GO**: < 5% improvement. Cache Manager overhead is negligible; bottleneck is NVMe latency/bandwidth, not the kernel copy. Establish confirmed NVMe sequential read throughput as v2-CF13'' candidate.
**GRAY**: 5–15% improvement. Context-dependent — may depend on system memory pressure; test under both low-pressure (model only in memory) and high-pressure (model + 4 GB background allocation) conditions.

### 8. Risk profile

| Risk | Mitigation |
|---|---|
| FILE_FLAG_NO_BUFFERING incompatible with mmap; ik_llama.cpp refactor needed | Smallest test is standalone; integration is gated on this result |
| ik_llama.cpp uses mmap by default; architecture change required | Stage 2 rationale: integration cost must be scoped; smallest test establishes whether the gain justifies the engineering cost |
| NVMe bandwidth may already saturate DRAM; copy overhead negligible at NVMe << DRAM bandwidth | Measured by the experiment; NVMe = 3 GB/s < DRAM = 11.5 GB/s means the copy is non-negligible in theory |

**Cheapest falsifier**: 1-hour C harness; if ≥15%, the mechanism is real.

### 9. Verdict: **REFINE**

Clean engineering mechanism, no math risk, no CF9/CF10 issues. Adjacent to now-established precedent (MemAscend) but on a distinct platform (Windows/ik_llama.cpp). Confirmed v2-CF13'' candidate on NO-GO path.

---

## H. Path 1 Track B — U3-WMCE

### 1. Load-bearing claims

M1: Windows Memory Compression Store engages for file-backed mmap pages (GGUF tensors), not only anonymous pages.
M2: Compressed weight pages in the Store are accessible at < 5 ms latency (DRAM-resident compressed) vs 146 ms NVMe read.
M3: Compression Store achieves ≥1.5× on INT4 weight data.

### 2. Per-claim prior-art status

**M1 — NOVEL** (OS behavior query; no published verification for mmap GGUF pages). This is the binary gate question from Stage 2.

**M2/M3 — ADJACENT**. Windows documentation confirms the Compression Store is in-DRAM. Cloudflare's Unweight blog (2026) reports 22% lossless model weight compression, showing weight data is compressible but uses a different mechanism (network-level, not OS memory compression). The DRAM tier between DRAM and NVMe is unique to Windows; no published LLM paper exploits it.

**Critical Stage 2 risk (primary question for Stage 3)**: Does Windows Compression Store engage for file-backed mmap pages?

Based on Windows documentation and the Hacker News discussion on mmap in llama.cpp: file-backed mmap pages are typically paged back to the backing file (GGUF on NVMe) when evicted under memory pressure, NOT compressed in the Store. The Compression Store is designed primarily for anonymous (privately allocated) pages. This is a near-CF9 precondition failure for M1.

**Resolution**: If M1 fails (mmap pages excluded from Store), the mechanism REQUIRES switching to anonymous-memory loading (VirtualAlloc + ReadFile), which is exactly the FNDP mechanism (U2). The two proposals COMPOSE: use FILE_FLAG_NO_BUFFERING + anonymous memory → pages are now anonymous → eligible for Compression Store. The composed mechanism is stronger than either alone.

### 3. Frame-mismatch check

**PARTIAL CF9 RISK on M1**. The Compression Store operates on anonymous pages; file-backed mmap pages are evicted to the backing file. The precondition (M1: Store engages for mmap pages) likely FAILS based on Windows memory management documentation. What survives after stripping M1: the anonymous-memory loading path + Compression Store on anonymous pages. This transforms WMCE from a standalone mechanism into a COMPOSITION with FNDP. Stripping the failed precondition leaves: anonymous weight loading (achievable via FNDP's double-buffer scheme) + OS-automatic Compression Store compression of cold anonymous weight pages.

### 4. Calibration ill-conditioning

No calibration fitting. CF10 not applicable.

### 5. Residency arithmetic (revised for anonymous-memory path)

70B at IQ4_XS: ~35 GB. After FNDP-style anonymous loading: weights in anonymous DRAM pages. Under memory pressure (7.28 GiB system, 35 GB model): Windows compresses cold anonymous pages at ~40–50% size reduction (Windows documentation). 7.28 GiB × 1.5× compressed = effectively ~10.9 GiB of weight accessible in DRAM (compressed). Remaining 35 - 10.9 = ~24 GB on NVMe. Per-token NVMe: ~24/35 × 437 MB/layer × 28 layers ≈ 8.46 GB NVMe I/O vs ~16.4 GB without Store. Speedup: ~1.5-1.9× on the NVMe-resident fraction of layers. Quality cost: 0 (decompression is lossless).

CF13'' note: the 3 GB/s NVMe and 7.28 GiB DRAM numbers are from FNDP experiment targets; WMCE depends on the same measurements.

### 6. Smallest test (refined)

**Primary test**: RAMMap.exe inspection
- Load Qwen3-1.7B into memory via mmap (standard ik_llama.cpp). Wait 5 minutes idle. Run RAMMap. Note: Compressed category rows.
- Then: load via anonymous memory (VirtualAlloc + ReadFile). Trigger memory pressure (4 GB background allocation). Wait 5 minutes. Run RAMMap again. Note: Compressed category rows.
- Measure re-access latency of compressed pages (time a sequential weight read after compression occurs).
- Wall-clock: ~1 hour.

**GO**: ≥20% of anonymous-loaded weight pages appear in Compressed category AND re-access latency ≤ 5 ms (confirming in-DRAM, not NVMe-evicted).
**NO-GO (M1 fail)**: mmap pages evicted to NVMe, not compressed. WMCE requires anonymous loading. Mechanism transforms to FNDP-WMCE composition.
**NO-GO (compression ratio low)**: Compressed pages only achieve ≤1.1× compression on INT4 data (INT4 weight data may be near-random after quantization, making it incompressible). This would kill the capacity-expansion claim.

### 7. Go/no-go thresholds

As above. The primary failure mode is M1 (mmap pages excluded). Secondary failure is low compression ratio on INT4-quantized weight data.

### 8. Risk profile

| Risk | Mitigation |
|---|---|
| M1 likely fails (mmap pages excluded from Store) | RAMMap test resolves in 1 hour; if fails, route to FNDP+WMCE composition |
| INT4 weight data near-incompressible (high entropy after quantization) | Measure compression ratio on actual GGUF tensor data; if < 1.1×, mechanism is dead even with anonymous loading |
| Compression Store CPU overhead during decompression | Measure re-access latency; if decompression is visible in per-token timing, overhead may cancel the gain |

### 9. Verdict: **REFINE — REQUIRES-MITIGATION**

The primary precondition (M1: Compression Store for mmap pages) likely fails. The surviving mechanism — anonymous-memory loading + automatic Compression Store on cold anonymous pages — requires FNDP's loading path as a prerequisite. WMCE advances only as a COMPOSITION with FNDP, not standalone. Mark as REFINE-COMPOSITION: advance to Stage 4 with the explicit framing that WMCE is the second layer of a FNDP+WMCE combined mechanism. Stage 4 should evaluate the composed mechanism as a single proposal.

---

## I. Path 2 Convergence Rep — C-DEEP-SPREAD-QBUD

### 1. Load-bearing claims

M1: Per-layer channel_coverage_90% from v2-CF1 (measured at layer OUTPUT) is a valid proxy for W_Q rank sensitivity at that layer's INPUT.
M2: Spearman ρ(channel_coverage_90%, per-layer ΔNLL at K=128) ≥ 0.70 across 5 probe layers.
M3: A depth-stratified rank schedule (K_Q(ℓ) from outlier spread) achieves better ΔNLL than uniform K=128 at matched bytes.

### 2. Per-claim prior-art status

**M1 — NOVEL** (coupling from activation-outlier spread to W_Q rank sensitivity). No published paper uses per-layer activation spread to schedule post-training weight-rank truncation. All prior work (SmoothQuant, OS+, PrefixQuant) treats layers as interchangeable for outlier handling. Date-stamped: 2026-05-09.

**M2 — NOVEL** (empirical Spearman ρ between these two measurements). Date-stamped: 2026-05-09.

**M3 — NOVEL as a quality-at-fixed-bytes improvement** (free improvement layered on CF11 path). Date-stamped: 2026-05-09.

**Direction ambiguity from Stage 2**: v2-CF1 measures outlier spread at LAYER OUTPUT (post-attention residual), but W_Q reads from the PREVIOUS layer's output (which is this layer's INPUT). These are the same object: W_Q at layer L reads from h_L, which is the output of layer L-1. The v2-CF1 measurement of layer L's output IS what W_Q at layer L+1 encounters. The coupling is: spread_L (CF1 output measurement at layer L) predicts the rank sensitivity of W_Q at layer L+1. The indexing is off by one, but the correlation may still hold if spread evolves smoothly across layers. The Spearman ρ test accounts for this implicitly.

**Two-paper composition flag**: v2-CF1 (outlier depth profile) + CF11 (per-layer W_Q rank from AQFKV) ≈ "use depth-varying outlier spread to schedule W_Q rank." Value-add: the coupling equation (spread predicts rank sensitivity) is the novel structural primitive; neither v2-CF1 nor CF11 alone derives the schedule; their composition is the insight.

### 3. Frame-mismatch check

No imported theorem. Spearman correlation is a nonparametric test requiring no distributional assumptions. No CF9 risk.

### 4. Calibration ill-conditioning

No calibration fitting per se; the rank schedule is set by the ρ-weighted spread, not fitted to data. However, using 5 probe layers for a Spearman ρ is a very small n; a ρ ≥ 0.70 at n=5 requires t-statistic confirmation (t = ρ√(n-2)/√(1-ρ²) ≈ 2.0 for ρ=0.70 at n=5, p < 0.10 one-sided). This is marginal statistical evidence. Mitigation: extend to 10 probe layers for ρ significance (add 5 more layers to the ΔNLL sweep, ~15 min additional).

### 5. Residency arithmetic

At zero residency cost vs CF11 path: quality improvement only. A schedule that drops K from 128 to 64 for shallow layers (L0–L7) and raises to 256 for deep layers (L20–L27) uses the same total W_Q bytes (within ~5%) as uniform K=128. ΔNLL improvement: estimated 0.10–0.25 nats vs uniform K=128. This is a free quality improvement layered on top of any residency-reducing path — it does not itself reduce residency.

### 6. Smallest test (refined)

**Script**: `experiments/stage0/ladder_v2/round6_qbud/run_perlay_wq_rank.py`
- Model: Qwen3-1.7B-Base BF16
- Layers: L0, L7, L14, L21, L27 (expand to 10 layers if budget allows)
- For each layer: truncate ONLY that layer's W_Q to K=128; hold all others BF16; compute ΔNLL on 512-token WikiText-2.
- Cross-reference per-layer ΔNLL against v2-CF1 channel_coverage_90% values.
- Compute Spearman ρ.
- Wall-clock: ~35 min (5 probe layers × 7 min each).
- Output path: `experiments/stage0/ladder_v2/round6_qbud/`

### 7. Go/no-go thresholds

**GO**: ρ ≥ 0.70 (strong monotone coupling). Proceed to schedule design and full-28-layer per-layer ΔNLL sweep for schedule calibration.
**NO-GO**: ρ < 0.30 (no coupling; activation spread and W_Q rank sensitivity are independent). Structural finding: depth-varying outlier spread is NOT a predictor of W_Q rank compression sensitivity. Direct per-layer profiling required.
**GRAY**: 0.30 ≤ ρ < 0.70. Weak coupling. Follow-up: layer-input vs layer-output spread distinction (rerun v2-CF1 with output of layer L−1 instead of layer L). ~1 hour.

### 8. Risk profile

| Risk | Mitigation |
|---|---|
| v2-CF1 measures output spread; W_Q reads input spread (off-by-one) | Explicitly compute input-side spread at L0, L7, L14, L21, L27 for comparison |
| n=5 Spearman ρ is low statistical power | Extend to 10 probe layers; report t-statistic and p-value |
| The coupling is non-monotone (spread and rank may have a non-linear relationship) | Also test Pearson correlation and visually inspect the scatter; non-linear but monotone is fine for scheduling purposes |

**Cheapest falsifier**: ρ < 0.30 at the 5-probe test; kills the shortcut and requires direct per-layer profiling.

### 9. Verdict: **REFINE**

Convergence C1 representative. 35-min test. Zero residency cost if confirmed (free quality improvement). Statistical rigor requirement noted (extend to 10 probe layers). Advance.

---

## J. Path 3 Frame-Novelty — F6-SINKCONST [FREE SWING]

### 1. Load-bearing claims

M1: k_sink^{L,g} = W_K^g · e_BOS is a fixed vector fully precomputed at model-load time from static weights.
M2: cos(k_sink^{L,g}, u_1^{L,g}) ≥ 0.5 for ≥ 20/28 layers and ≥ 4/8 KV groups, where u_1 is the top-1 right singular vector of W_Q per head group.
M3: High alignment explains why sinks are stable: BOS key aligns with the dominant W_Q projection direction, making ALL queries assign high logit to BOS regardless of query content.

### 2. Per-claim prior-art status

**M1 — NOVEL** (precomputed k_sink at model load time). The algebraic fact that BOS produces a fixed key vector is trivially true but has not been exploited as a precomputed primitive. Date-stamped: 2026-05-09. `elegance-class: conserved-quantity`

**M2 — PARTIAL with adjacent prior art**. Critical finding from Web search: arXiv:2508.02546 ("What are you sinking? A geometric approach on attention sink," August 2025) provides a geometric approach to attention sinks using spectral graph analysis and identifies that dominant eigenvectors align with reference tokens (including BOS). The paper finds "centralized reference frame signature" and "BOS specialization." However it uses a graph-theoretic (Fiedler eigenvalue / centralization) framework, not the direct W_K × W_Q spectral alignment claim of SINKCONST.

arXiv:2402.09221 ("Spectral Filters, Dark Signals, and Attention Sinks," Cancedda, 2024) presents a spectral filter interpretation of attention sinks but does not derive the sink from W_K · e_BOS alignment with W_Q singular vectors.

arXiv:2603.11487 ("Attention Sinks Are Provably Necessary," March 2026) provides a mathematical proof that sinks must exist for certain function classes under softmax normalization. This does not address the specific W_K × W_Q alignment claim.

The specific claim (k_sink = W_K · e_BOS aligns with top W_Q singular vector, AND this is the algebraic explanation of sink stability) has no direct published form. 2508.02546 is the closest but uses different geometric machinery. Status: PARTIAL — the alignment connection is not directly stated in 2508.02546; the SINKCONST claim is more specific and testable. Date-stamped: 2026-05-09.

**M3 — ADJACENT**. 2508.02546 and 2603.11487 both provide explanations of why sinks are stable/necessary, but from different angles. The W_K singular-vector alignment explanation is a novel alternative to these existing explanations. ADJACENT (the phenomenon is explained, but not via this specific algebraic route). Date-stamped: 2026-05-09.

**Two-paper composition flag**: arXiv:2508.02546 (geometric sink explanation via eigenvectors) + CF11 (W_Q singular structure in Qwen3) ≈ "BOS key aligns with W_Q principal direction." Value-add: 2508.02546 does not use the specific W_K · e_BOS = k_sink construction; SINKCONST's cos(k_sink, u_1) measurement gives a quantitative, testable number (the cosine) from static weights alone, without requiring a forward pass or graph analysis. The no-forward-pass structural test is the novel primitive — it predicts sink behavior entirely from weight geometry.

### 3. Frame-mismatch check

[FREE SWING] — CF-tether requirement suspended. Stack primitive: W_K and W_Q weight matrices (load-time accessible); BOS token embedding e_BOS (accessible from embedding table). The imported computation is SVD of W_Q (already done in AQFKV). No theorem imported with preconditions.

One potential CF9 concern: the alignment claim assumes e_BOS has a distinctive structure (high norm or specific direction) that makes W_K · e_BOS align with W_Q's principal direction. If e_BOS is similar to all other high-frequency token embeddings, the alignment may be trivially true for many tokens (not a BOS-specific phenomenon). The Stage 1 proposal already addresses this: "also test for the top-3 highest-frequency tokens." This is the right mitigation. No CF9 kill.

### 4. Calibration ill-conditioning

No calibration fitting. CF10 not applicable.

### 5. Residency arithmetic

Precomputed k_sink storage: 28 × 8 × 128 × 2B = 458 KB — negligible. Direct residency savings are minor (BOS KV cache entry: 1/32K of KV cache at 32K context). Mechanism value is structural (algebraic explanation of sink stability) and potential for architectural reuse (if all queries project similarly onto u_1, the u_1 component can be separated from the query residual for further compression of the K attention path).

### 6. Smallest test (refined)

**Script**: `experiments/stage0/ladder_v2/round6_sinkconst/run_sinkconst.py`
- Model: Qwen3-1.7B-Base BF16
- Step 1 (~5 min): Load W_K (8 groups × 128 × 2048), W_Q (16 heads × 128 × 2048), e_BOS from embedding table (tokenizer).
- Step 2 (~5 min): Compute k_sink^{L,g} = W_K^g[L] @ e_BOS for all 28L × 8G. Load u_1^{L,g} from AQFKV SVD factors (per-head).
- Step 3 (~2 min): Compute cos(k_sink, u_1) for all 28×8=224 pairs. Histogram and per-layer heatmap.
- Also compute for top-3 most-frequent tokens (to check BOS-specificity).
- Wall-clock: ~12 min.
- Output path: `experiments/stage0/ladder_v2/round6_sinkconst/`

### 7. Go/no-go thresholds

**GO**: cos ≥ 0.5 in ≥ 20 layers AND ≥ 4/8 groups AND cos significantly higher for BOS than for random tokens. Proceed to k_sink projection compression proposal.
**NO-GO**: k_sink random relative to W_Q singular structure (mean cos ≈ 0). The sink phenomenon cannot be explained by static W_K/W_Q weight geometry alone; it requires dynamic attention patterns or training-dynamics explanations (consistent with 2508.02546's graph-based approach). Structural finding: W_K·e_BOS alignment with W_Q singular vectors is not the algebraic origin of sink stability.
**GRAY (cosmopolitan alignment)**: High cos for BOS AND for many other tokens. The alignment is a general property of common tokens, not BOS-specific. Finding: W_K maps high-frequency tokens into alignment with W_Q's dominant direction — a generalization of the sink phenomenon. Report as a structural finding.

### 8. Risk profile

| Risk | Mitigation |
|---|---|
| 2508.02546 already provides a comparable geometric explanation, diminishing novelty | SINKCONST's no-forward-pass, weight-only, quantitative cosine test is complementary to 2508.02546's training-dynamics analysis; position as corroborating evidence with a distinct, testable claim |
| BOS embedding e_BOS may not be high-norm (embedding norms vary) | Report ‖e_BOS‖ vs median embedding norm; if e_BOS is high-norm, the alignment may be norm-driven, not directionally specific |
| The alignment may be trivially explained by the dominant embedding PC axis aligning with W_K and W_Q (not BOS-specific at all) | Run PCA on the embedding matrix; check if e_BOS aligns with PC1 of embeddings; if so, the mechanism is about embedding PC1, not BOS-specifically |

### 9. Verdict: **REFINE — wildcard, CF-tether-suspended, structural floor met**

Prior art 2508.02546 is adjacent but uses different geometric machinery. The specific cos(W_K·e_BOS, W_Q-singular-vector) test is not in any paper found. 12-min test, 0 quality risk. High information content on sink theory.

---

## K. Path 3 Frame-Novelty — A1-PEER

### 1. Load-bearing claims

M1: The constraint "the model cannot hold its own weights" forces inference to run entirely in a peer (smaller) model's weight coordinate system.
M2: A calibration-fit rank-64 projector P: R^{1024} → R^{2048} (Qwen3-0.6B to 1.7B dimensional alignment) achieves R²_eval > 0.70 at mid-depth layer 14.
M3: CF10 conditioning is satisfied: rank-64 projector has ~200K parameters; calibration at 10K tokens × 2048 dims = 20M values. Well-conditioned (100:1 ratio).

### 2. Per-claim prior-art status

**M1 — NOVEL** (constraint framing). The "model cannot hold its own weights" constraint is not in any published paper. Model Stitching (arXiv:2506.06609, NeurIPS 2025) is training-time only; distillation is training-time; knowledge transfer requires retraining. Inference-time peer-swap with no model surgery is outside the published compression vocabulary. Date-stamped: 2026-05-09.

**M2 — NOVEL** (empirical claim; cross-scale hidden-state alignment at inference time without training has not been measured). Date-stamped: 2026-05-09.

**M3 — PARTIAL**. The CF10 conditioning check passes for rank-64 (Stage 1 arithmetic confirmed, and restated in Stage 2). The CF10 failure mode (2.1M params on ~2M samples) is avoided. WDLA died from this; A1-PEER addresses it by construction. Rank-64 is well-conditioned.

**Frobenius lower bound on reconstruction error** (from Stage 2 pressure-test): d=1024 (Qwen3-0.6B) cannot represent all of d=2048's degrees of freedom without loss. The theoretical minimum reconstruction error under any rank-64 projector is bounded below by the discarded singular value energy of the activation matrix cross-correlation: min_P ‖H_{1.7B} - P H_{0.6B}‖_F² ≥ Σ_{k>64} σ_k²(C) where C = H_{1.7B}^T H_{0.6B} is the cross-scale covariance matrix. If the cross-scale covariance has rank ≤ 64 (concentrated), R²_eval can approach 1. If it is full-rank, R²_eval is bounded by the fraction of cross-scale variance captured by rank-64. This bound must be computed from the calibration data in step 0 of the experiment before fitting the projector.

**Two-paper composition flag**: WDLA (our own CF10-killed predecessor, rank-64 variant) + Model Stitching (cross-scale representation alignment) ≈ "calibration-fit cross-scale projector for inference." Value-add: A1-PEER uses rank-64 by construction (CF10-safe); the constraint framing ("cannot hold own weights") is structurally alien; the R²_eval measurement at mid-depth will determine whether cross-scale alignment is achievable post-training without any gradient signal. The constraint-alien framing is the value-add beyond obvious composition.

### 3. Frame-mismatch check

No theorem imported. The rank-64 projector is a standard least-squares fit (Moore-Penrose pseudoinverse or ridge regression). CF10 is satisfied by construction (rank-64 parameter count ~200K << calibration samples × dims).

**Edge case**: if Qwen3-0.6B and Qwen3-1.7B use different tokenizers or positional encoding schemes, the activation alignment may be degraded. Both are Qwen3-family models with identical tokenizers and RoPE configuration. No frame-mismatch.

### 4. Calibration ill-conditioning

**CF10 pre-flight (explicit)**:
- `n_params_to_fit`: rank-64 projector P: R^{1024} → R^{2048} = 1024 × 64 + 64 × 2048 = 197,632 parameters (low-rank: P = U × V)
- `n_independent_samples`: 10K tokens (Stage 1 specification)
- `n_output_dims_per_sample`: 2048
- `n_independent_samples × n_output_dims_per_sample` = 10K × 2048 = 20.5M
- Ratio: 20.5M / 197.6K ≈ 103 — well-conditioned (≥ 10:1).
- Ridge specification: ridge=1e-2 or higher; report cal R² and eval R² separately.
- **CF10: PASS**.

### 5. Residency arithmetic

The mechanism does not reduce weight storage — it substitutes one model's weights for another's. The peer (0.6B) is 4× smaller than the target (1.7B); running inference in the peer's coordinate system reduces the model loaded from 1.7B to 0.6B plus the projector overhead (200 KB per layer). For 70B: the peer would be Qwen3-4B (the same 80-layer structure), loading 4B weights instead of 70B — a 17× reduction in model storage. However this is a speculative 70B projection; the experiment at 1.7B is the structural signal.

### 6. Smallest test (refined)

**Script**: `experiments/stage0/ladder_v2/round6_peer/run_peer_proj.py`
- Models: Qwen3-0.6B-Base BF16 (peer) and Qwen3-1.7B-Base BF16 (target)
- Step 0 (~15 min): Run both models on 10K WikiText-2 tokens; cache hidden states at L14. Compute cross-scale covariance C = H_{1.7B}^T H_{0.6B}. Compute rank-64 truncation of C via SVD. Report theoretical R²_upper_bound.
- Step 1 (~30 min): Fit rank-64 projector P on calibration 9K tokens (ridge=1e-2). Evaluate on held-out 1K tokens. Report cal R² and eval R².
- Step 2 (optional, ~1.5 hr): If R²_eval > 0.70, measure perplexity using 0.6B hidden states + P projector into 1.7B computation graph.
- Wall-clock: ~3 hr total (or ~45 min for steps 0+1 alone as the gate measurement).
- Output path: `experiments/stage0/ladder_v2/round6_peer/`

### 7. Go/no-go thresholds

**GO**: R²_eval > 0.70 AND cal/eval R² gap < 0.10 (confirming good generalization). Proceed to perplexity measurement and scope the full cascade.
**NO-GO**: R²_eval < 0.40 or cal/eval gap > 0.30. The cross-scale activation alignment is insufficient for useful inference-time substitution; fundamental information is lost in the dimensional change from 1024 to 2048. The Frobenius lower bound from Step 0 would explain why.
**GRAY**: R²_eval = 0.40–0.70. Alignment is partial. Investigate whether the poorly-aligned dimensions correspond to late-layer deep-context representations vs early-layer syntactic representations. May still be useful for the first N layers (where representations are more syntactic and cross-scale alignment may be higher).

### 8. Risk profile

| Risk | Mitigation |
|---|---|
| Cross-scale activation spaces may not align post-training (different feature scales despite same family) | Theoretical lower bound from Step 0 provides ceiling before fitting; if ceiling < 0.50, mechanism is dead before Stage 5 |
| 3-hour wall-clock exceeds 2-hour smoke threshold | Gate on Step 0+1 alone (~45 min); perplexity measurement is only run if R² > 0.70 |
| Frame novelty may not translate to practical compression at 70B | The mechanism's value is structural (proves whether cross-scale alignment is achievable post-training); 70B application is speculative without 1.7B confirmation |

**Cheapest falsifier**: compute theoretical R²_upper_bound from cross-scale covariance SVD at L14 (~15 min, step 0). If the top-64 singular values of the cross-scale covariance capture < 50% of the total cross-scale variance, the mechanism cannot achieve R²_eval > 0.50 at rank 64, and the experiment should not proceed.

### 9. Verdict: **REFINE**

CF10-safe by construction. Frame-novelty is genuine (no published inference-time peer-swap without retraining). Theoretical R²_upper_bound as a pre-experiment cheapest-falsifier gate. 3-hour experiment gated on 15-minute upper-bound check. Top-3 candidate for frame-novelty prize.

---

## L. Path 4 Elegant-Equivalence — R6-WDOWN-CONDP [FREE SWING, algebraic-identity]

### 1. Load-bearing claims

M1: y = W_down·a = W_down[:,H]·a_H + W_down[:,C]·a_C (exact column-partition identity, no approximation).
M2: Equal-residency formulation: top-17% neurons (H) at INT8 and bottom-83% (C) at INT2 matches total bytes of uniform INT4 across all neurons.
M3: CF1's firing-rank dominance implies that H neurons contribute most of the W_down output signal; INT8 on H reduces the high-consequence quantization error while INT2 on C's small-magnitude neurons is acceptable.
M4: ΔNLL improvement vs uniform INT4 on W_down[14] ≥ 0.05 nats (the go threshold).

### 2. Per-claim prior-art status

**M1 — NOVEL** (exact algebraic identity applied as a precision-routing primitive). MoEfication (Zhang et al., ICLR 2022) clusters FFN neurons into experts; Deja Vu (ICML 2023) predicts sparse attention/FFN from prior context. Neither applies post-activation magnitude-partitioned precision routing on W_down as described. The algebraic partition (column split + precision assignment) with a FIXED PARTITION (calibration-mean-derived, not predicted) is distinct from both. Date-stamped: 2026-05-09. `elegance-class: algebraic-identity`

**M2 — NOVEL** (equal-residency formulation deriving the H/C split as a constraint). Date-stamped: 2026-05-09.

**M3 — PARTIAL**. CF1's firing-rank dominance is the premise. The specific claim that W_down quantization error is dominated by H neurons (not just the forward-pass output) requires the additional assumption that H neurons are those where quantization error costs most. This is not guaranteed by CF1 alone: CF1 says H neurons produce the largest-magnitude output, but quantization error is proportional to the weight value magnitudes AND the activation magnitudes. The relevant object is ‖ΔW_down[:,H] a_H‖₂ (quantization error contribution from H) vs ‖ΔW_down[:,C] a_C‖₂ (from C). PARTIAL — CF1 supports M3 but does not prove it.

**M4 — stage-gate claim (to be measured)**.

**INT2 tail risk (from Stage 2)**: Stage 2 explicitly flagged this as the pre-experiment: measure INT2 tail-only ΔNLL before full cascade. PolyGLU (arXiv:2603.13347) shows that neurons specialize to near-deterministic activation functions — consistent with the idea that C neurons are consistently small in activation magnitude. This provides additional support for INT2 on C being acceptable.

**Two-paper composition flag**: AQLM / GPTQ (mixed-precision quantization) + CF1 (W_up firing-rank dominance) ≈ "use firing-rank to assign precision to W_down columns." Value-add: the exact algebraic column partition identity (M1) means the precision assignment can be derived from a simple post-activation sort at each token rather than from a learned codebook. The equal-residency formulation (M2) makes the precision assignment bytes-neutral. Neither AQLM nor GPTQ uses CF1 as a column-partition oracle for W_down.

### 3. Frame-mismatch check

[FREE SWING] — CF-tether requirement suspended. Stack primitive: INT8 and INT2 quantization are natively supported in ggml/ik_llama.cpp (Q8_0 and IQ2_S kernels exist). The column partition is a storage-layout decision. The algebraic identity M1 is unconditionally exact. No CF9 risk.

**INT2 precondition**: INT2 (4 levels per weight) imposes large quantization error if cold-neuron weights span a wide dynamic range. CF1/CF3 confirm that C neurons have small ACTIVATION magnitudes; but W_down columns corresponding to C neurons still have full dynamic range in the weights themselves. The quantization error for W_down[:,C] at INT2 is ‖ΔW_down[:,C]‖_F (weight-domain error) × ‖a_C‖ (activation magnitude). The activation magnitude ‖a_C‖ is small (CF1); the weight error is not. So the error contribution is: (large weight error) × (small activation) — the product may still be acceptable. This is the empirical question for M4.

### 4. Calibration ill-conditioning

No calibration fitting per se. The H/C partition boundary (17% split) is derived analytically from the equal-residency constraint. The partition indices (which specific neurons are in H) use calibration-mean activation magnitudes — but this is a rank-ordering, not a parameter fit. CF10 not applicable in the standard sense. The distribution-specificity risk is analogous to CF10 (the calibration-derived partition may not generalize to OOD inputs). Mitigation: measure ΔNLL on OOD corpus (code vs prose) in addition to WikiText-2.

### 5. Residency arithmetic

Equal-residency formulation (from Stage 1): top-17% H at INT8 (1 byte/weight) + bottom-83% C at INT2 (0.25 byte/weight) = 0.17 × 1 + 0.83 × 0.25 = 0.17 + 0.2075 = 0.3775 bytes/weight average = 3.02 bpw. Baseline uniform INT4: 4 bpw. WDOWN-CONDP is actually WORSE in bpw (3.02 bpw) than uniform INT4 (4 bpw)... Wait: INT4 = 0.5 bytes/weight = 4 bpw. CONDP: 17% × 8 bpw + 83% × 2 bpw = 1.36 + 1.66 = 3.02 bpw. So CONDP is BETTER than INT4 at 3.02 bpw. The correct equal-bpw formulation: x × 8 + (1-x) × 2 = 4 → 8x + 2 - 2x = 4 → 6x = 2 → x = 1/3. So top-33% H at INT8, bottom-67% C at INT2, average 4 bpw — equal to INT4. At top-17% INT8 + bottom-83% INT2: average bpw = 17%×8 + 83%×2 = 3.02 bpw — actually LOWER (better) than INT4. The Stage 1 arithmetic used 1/6 ≈ 17% for matching INT4 total bytes, which yields ~3.02 bpw — this is cheaper than INT4 while concentrating precision on H neurons. This is favorable for residency, not neutral.

Correction to Stage 1 residency arithmetic: WDOWN-CONDP at the 17% split achieves ~3.0 bpw on W_down (better than INT4's 4 bpw) AND concentrates precision on the high-firing neurons. Total W_down residency at 70B: 28672 × 8192 × 80 × 3.0/8 = 8.65 GB → vs INT4 at 11.53 GB. Saves 2.88 GB. This is a significant residency improvement at 70B, not just a quality improvement.

### 6. Smallest test (refined)

**Script**: `experiments/stage0/ladder_v2/round6_wdown_condp/run_condp.py`
- Model: Qwen3-1.7B-Base BF16
- Step 0 (~10 min): Quantize ONLY W_down[14]'s bottom-83% columns (C partition) at INT2; leave H partition at BF16. Measure ΔNLL on 512-token WikiText-2. This is the "INT2 tail-only" pre-check.
- **If INT2 tail ΔNLL > 0.50 nats**: STOP. INT2 on cold neurons is catastrophic; report KILL for the precision-routing idea.
- Step 1 (~20 min): If INT2 tail passes: apply full CONDP (H at INT8, C at INT2). Measure ΔNLL vs uniform INT4 W_down[14].
- Wall-clock: ~30 min total.
- Output path: `experiments/stage0/ladder_v2/round6_wdown_condp/`

### 7. Go/no-go thresholds

**GO**: INT2 tail ΔNLL ≤ 0.50 nats AND ΔNLL(CONDP vs INT4) ≤ 0 nats (CONDP is at least as good as INT4 at lower average bpw). The key point: at 17% H, CONDP averages ~3 bpw, which is better than INT4 (4 bpw). So CONDP should be compared to uniform 3-bpw quantization (INT3), not INT4. GO means CONDP outperforms uniform INT3 on W_down at matched bpw.
**NO-GO**: INT2 tail ΔNLL > 0.50 nats. Dead before cascade.
**GRAY**: INT2 tail ≤ 0.50 but CONDP no better than uniform INT3. Structural finding: precision concentration at H neurons does not outperform uniform quantization at the same average bpw. CF1's firing-rank asymmetry does not translate to quantization-error asymmetry for W_down columns.

### 8. Risk profile

| Risk | Mitigation |
|---|---|
| INT2 tail quantization catastrophic (large weights × small activations still large error) | INT2-tail-only pre-check (Step 0); if fails, no further investment |
| Activation-mean-derived H/C partition may not reflect OOD importance | Test on code corpus vs WikiText-2 after GO |
| Column-interleaved storage (H and C columns non-contiguous) may cause NVMe random-read overhead | Layout: store H columns contiguously in first block, C in second; sequential read within each block |

**Cheapest falsifier**: 10-min INT2 tail ΔNLL pre-check (Step 0).

### 9. Verdict: **REFINE — wildcard, CF-tether-suspended, structural floor met**

Algebraic-identity elegance class confirmed. Residency arithmetic corrected upward (3 bpw, not 4 bpw, at the 17% split — actually better than INT4). INT2 tail pre-check is the decisive gate. Top candidate for elegant-equivalence path.

---

## M. A6-LLMR (Convergence Representative C3)

### 1. Load-bearing claims

M1: Per-token log-likelihood under lm_head (computed at each intermediate layer L) is monotone-decreasing in L for most (≥ 60%) tokens in normal Qwen3 inference.
M2: A hard-monotone confidence constraint (exit at first layer where per-token entropy increases) achieves early exit on ≥ 40% of tokens in practice.
M3: The first-layer entropy increase is a reliable signal: once entropy increases, it does not decrease again in subsequent layers (for the exiting tokens).

### 2. Per-claim prior-art status

**M1 — PARTIAL**. "The Remarkable Robustness of LLMs: Stages of Inference?" (arXiv:2406.19384) characterizes per-layer confidence evolution, noting phases of prediction consolidation. However it does not establish a monotone-confidence hard constraint for early exit. SimLens (arXiv:2507.17618, July 2025) uses entropy-based confidence estimation in a hybrid early-exit mechanism "SimExit" — adjacent but uses a THRESHOLD (not a monotone constraint). ConfLayers (arXiv:2604.14612, April 2026) adapts layer skipping based on confidence scores — adjacent but uses learned thresholds. QuickSilver (arXiv:2506.22396, June 2025) uses entropy-based gating for token-level halting — structurally closest to LLMR.

**Stage 2 pressure-test (from Stage 2 hand-off)**: "SimLens (March 2026) and ConfLayers (April 2026) — do they enforce a monotone-confidence hard constraint, or only a learned threshold?" Based on the search results: **both SimLens and ConfLayers use learned/adaptive thresholds, NOT a hard monotone constraint**. SimLens uses "entropy-based confidence estimation" with a threshold. ConfLayers "iteratively selects layers to skip based on an adaptive threshold." Neither enforces the specific structural constraint: "exit at the first layer where per-token entropy INCREASES." LLMR's hard-monotone constraint (no threshold, no learning, fires at first non-monotone point) is structurally distinct from all found papers. PARTIAL — early exit via per-layer entropy exists; the hard-monotone (no learned threshold) variant is novel. `elegance-class: conserved-quantity` (entropy as a monotone convergence diagnostic). Date-stamped: 2026-05-09.

**M2/M3 — NOVEL** (no paper measures the fraction of tokens for which Qwen3 per-layer entropy is monotone). Date-stamped: 2026-05-09.

**Two-paper composition flag**: SimLens/ConfLayers (per-layer entropy early exit) + CF11's attention concentration finding (layers have different information content) ≈ "use per-layer lm_head entropy to gate exit." Value-add: the HARD MONOTONE constraint (exit at entropy increase, no trained threshold) is the novel primitive. All published methods require a trained threshold or calibration; LLMR's constraint is parameter-free and interpretable as a monotone convergence criterion.

### 3. Frame-mismatch check

No theorem imported. Entropy is computed directly from lm_head logits at each layer. The monotone constraint is a structural observation, not a mathematical import. No CF9 risk.

### 4. Calibration ill-conditioning

No calibration fitting. CF10 not applicable.

### 5. Residency arithmetic

If 40% of tokens exit at L < 28: mean compute depth = 0.4 × E[exit layer] + 0.6 × 28. If mean exit at L=14 for exiting tokens: mean depth = 0.4×14 + 0.6×28 = 5.6 + 16.8 = 22.4 layers. Compute reduction: 22.4/28 ≈ 0.80 of full compute. At 70B NVMe: 20% compute saving → ~20% token latency improvement. At 1.7B BF16: ~25% speedup on exiting tokens.

**Important note**: The per-layer entropy profile measurement (30-min, 200 tokens) produces data useful for multiple downstream proposals (C-DEEP-SPREAD-QBUD, ATTNQ-FULL scheduling, any early-exit scheme). This is its primary value as a Convergence C3 representative.

### 6. Smallest test (refined)

**Script**: `experiments/stage0/ladder_v2/round6_llmr/run_entropy_profile.py`
- Model: Qwen3-1.7B-Base BF16
- Calibration: WikiText-2 200 tokens (greedy decoding)
- For each token: compute lm_head(h_L) for L ∈ {0, 4, 7, 10, 14, 18, 21, 24, 27} (sparse first pass).
- Compute per-layer entropy H(softmax(lm_head(h_L) / T)) for T=1.
- For each token, check if entropy is monotone (non-increasing) across the 9 sampled layers.
- Report fraction monotone, mean exit layer under hard-monotone policy.
- Wall-clock: ~30 min.
- Output path: `experiments/stage0/ladder_v2/round6_llmr/`

### 7. Go/no-go thresholds

**GO**: ≥ 60% non-monotone tokens (first entropy increase occurs before L=28), AND mean first-increase layer ≤ 21. Validates that hard-monotone early exit is applicable on ≥ 60% of tokens with 25%+ depth reduction on average.
**NO-GO**: Entropy is naturally monotone-decreasing for ≥ 80% of tokens across all layers. This is a class-level settlement: Qwen3's inference naturally consolidates confidently across layers, no early exit is available. Structural finding: Qwen3's intermediate representations converge monotonically to the final prediction, suggesting strong per-layer progress in prediction consolidation.
**GRAY**: 40–60% non-monotone. Follow-up: token-type stratification (content words vs function words vs numerical tokens). Some token types may have reliably early exit while others require full depth. ~15 min additional analysis on existing profile data.

### 8. Risk profile

| Risk | Mitigation |
|---|---|
| Entropy is monotone for most tokens (NO-GO class finding) | NO-GO is a high-value structural finding about Qwen3 inference dynamics |
| Per-layer entropy depends on how lm_head is applied at intermediate layers (the head may need adaptation for intermediate states) | Use the tied lm_head directly (Qwen3-1.7B has tied embeddings); no adaptation needed; intermediate logits are raw |
| 9 sparse layers insufficient to detect entropy increase precisely | If GO signal found, run dense (all 28 layers) profile on a subset of 50 tokens to characterize the exit distribution more precisely |

**Cheapest falsifier**: 30-min entropy profile on 200 tokens.

### 9. Verdict: **REFINE**

Prior art (SimLens, ConfLayers, QuickSilver) all use learned thresholds; the hard-monotone no-threshold constraint is structurally distinct. High convergence value as C3 representative (entropy profile data useful for multiple downstream proposals).

---

## N. Summary of verdicts

| ID | Verdict | Reason |
|---|---|---|
| C-JOINT-DELNLL | **REFINE** | Clean CF11 coupling, novel interaction-prediction framing, 12-min test |
| R6-WVOP | **REFINE** | arXiv:2604.22778 elevates prior probability on V/O spectra; 25-min test; 70B prize |
| F6-WALIGN | **REFINE** | Algebraic identity exact; 5-min histogram; top-3 candidate |
| F6-GQARED | **REFINE** | GQA-specific discrete-sum structure is novel; 15-min test |
| C-WVOK | **REFINE** | 20-min test; gates three proposals (C2 convergence); 2604.22778 support |
| R6-ATTNQ-FULL | **REFINE** (gated on C-WVOK) | Cascade contingent on W_V gate; proceed after C-WVOK |
| U2-FNDP | **REFINE** | MemAscend precedent ADJACENT; Windows-specific implementation novel; 1-hr harness |
| U3-WMCE | **REFINE—COMPOSITION** | M1 likely fails (mmap exclusion); advance only as FNDP+WMCE composition |
| C-DEEP-SPREAD-QBUD | **REFINE** | Convergence C1 rep; 35-min test; zero residency cost if GO |
| F6-SINKCONST | **REFINE** | 2508.02546 adjacent but different machinery; 12-min test; structural finding regardless |
| A1-PEER | **REFINE** | CF10-safe; frame-novelty genuine; 15-min upper-bound pre-check gates 3-hr experiment |
| R6-WDOWN-CONDP | **REFINE** | Residency corrected (3 bpw, better than INT4); INT2 tail pre-check is decisive gate |
| A6-LLMR | **REFINE** | SimLens/ConfLayers use trained thresholds; hard-monotone is distinct; C3 convergence value |

**Verdict counts**: REFINE: 12 (one as COMPOSITION); REGENERATE: 0; KILL: 0; DOWNGRADE: 0.

**U3-WMCE KILL_LIST note**: Not killed outright, but the standalone mechanism (mmap pages in Compression Store) is REQUIRES-MITIGATION. If the RAMMap pre-check confirms mmap pages are excluded from the Store, WMCE is DOWNGRADED to an engineering sub-component of the FNDP+WMCE composition.

---

## O. Experiment queue (priority order)

1. **C-WVOK / R6-WVOP** (20-25 min, same measurement): W_V[14] and W_O[14] SVD. Gates three proposals (C-WVOK, R6-WVOP, R6-ATTNQ-FULL). Highest leverage per minute.
2. **F6-WALIGN** (5 min): W_gate/W_up row cosine histogram. Cheapest non-trivial measurement.
3. **F6-SINKCONST** (12 min): W_K·e_BOS alignment. Leverages AQFKV factors already computed.
4. **C-JOINT-DELNLL** (12 min): W_Q/W_K joint truncation δ. Leverages AQFKV factors.
5. **A6-LLMR** (30 min): Per-layer entropy profile. High-multiplier for downstream proposals.
6. **F6-GQARED** (15 min): GQA group importance sweep.
7. **C-DEEP-SPREAD-QBUD** (35 min): Per-layer W_Q rank sensitivity Spearman ρ.
8. **U2-FNDP** (1 hr): FILE_FLAG_NO_BUFFERING harness.
9. **U3-WMCE** (1 hr): RAMMap mmap vs anonymous Compression Store check.
10. **R6-WDOWN-CONDP** (30 min): INT2 tail pre-check → CONDP measurement.
11. **A1-PEER** (45 min gates → 3 hr full): Cross-scale R² upper-bound → projector fit.

**Stage 5 top-3 candidates**:
1. **R6-WVOP** — highest residency prize at 70B (~11 GB saved if W_VO concentrated); now elevated by arXiv:2604.22778; cheapest test relative to prize size.
2. **C-JOINT-DELNLL** — measurement produces a concrete δ that enables free joint compression if additive; gates the entire attention compression cascade.
3. **F6-WALIGN** — 5-min histogram, elegant-equivalence class, exact algebraic identity; if ≥5% aligned neurons the payoff is structurally novel relative to everything published.

---

*Sources consulted*:
- arXiv:2604.22778 (Spectral Lifecycle of Transformer Training, April 2026) — V/O spectral asymmetry
- arXiv:2508.02546 (Geometric approach to attention sinks, August 2025) — attention sink geometry
- arXiv:2604.14612 (ConfLayers, April 2026) — per-layer confidence early exit
- arXiv:2507.17618 (SimLens, July 2025) — entropy-based early exit
- arXiv:2506.22396 (QuickSilver, June 2025) — entropy-based token halting
- arXiv:2505.23254 (MemAscend, May 2025) — direct NVMe engine for LLM
- arXiv:2603.13347 (PolyGLU, March 2026) — SwiGLU activation routing
- arXiv:2502.07864 (TransMLA, February 2026) — post-training MLA conversion
- arXiv:2407.21118 (PALU, 2024) — KV low-rank decomposition
- MERL TR2025-075 (LatentLLM, 2025) — joint QK Tucker compression
- arXiv:2410.13060 (AERO, 2024) — SwiGLU fold
- arXiv:2405.01943 (GLU sparsity, 2024) — SwiGLU structured pruning
- arXiv:2603.11487 (Attention Sinks Provably Necessary, March 2026) — sink necessity proof
