# Stage 3 Deep Research — Run 009

Date: 2026-05-09
Researcher: Claude Sonnet 4.6 (Stage 3 Deep Researcher role)
WebSearch: live searches run; all prior-art citations date-stamped.

---

## Scope and reading order

This document covers all 19 advancers from Stage 2 (plus the 4 convergence-cluster reps and 2 frame-novelty slots = same set of ideas). Depth allocation follows the prompt specification:
- ~800–1200w: F1-MQKP, F5-RSUC, A1-TROPFFN, R9-R4 (Path 2 cluster reps) + F1-MQKP algebraic-identity (Path 4) + R9-R1/ATQKV (highest A1)
- ~400–600w: standard advancers
- ~250–400w: wildcards (F6, A3, U5-A-VGTP, R9-R5)

Hard gate applied first in every section: CF9 frame-mismatch check, CF10 calibration ill-conditioning check. Kill list gates applied.

---

## CF9 / CF10 Gate Summary (applied across all advancers before narrative)

| ID | Imported theory / calibration fit | CF9 pass? | CF10 pass? |
|---|---|---|---|
| F1-MQKP | Eckart-Young theorem (any real matrix) | YES — trivially satisfied | N/A (no calibration fit) |
| F2-CKDJ | Softmax shift-invariance (adding constant to all logits) | YES — trivially satisfied | N/A |
| F3-L1GNF | AERO fold identity + null-space criterion | YES — eigendecomposition precondition verified (C_x^0 is positive semidefinite by construction) | YES — 6144 scalar params vs 200×2048=409K samples |
| F4-WQKRS | Water-filling / Shannon rate-distortion (continuous spectrum, separable per-matrix) | YES — trivially satisfied on SVD spectrum | N/A |
| F5-RSUC | PCA / covariance eigendecomposition | YES — positive semidefinite covariance guaranteed | N/A |
| F6-TROPICALDECODE | Tropical max-plus GEMV | YES — no fragile theorem import; max-plus is exact for argmax | N/A |
| C13-EQSUB | SVD projection mass (standard Frobenius argument) | YES | N/A |
| C14-CFSHIFT | Linearity of W_Q over vector addition | YES — trivially satisfied | YES — ~6144 constants vs 200×2048 samples |
| C15-SNRATE | Spectral SNR as compression oracle | YES — no imported theorem | YES — 3-param fit on 5 data points |
| F1-MQKP (C1 rep) | (same as F1-MQKP above) | | |
| F5-RSUC (C2 rep) | (same as F5-RSUC above) | | |
| A1-TROPFFN (C3 rep) | Tropical semiring arithmetic | YES — log-quantization precondition: per-row normalization handles dynamic range | N/A |
| R9-R4 (C4 rep) | Sequential NVMe read throughput (CF13* re-derivation) | YES — no theorem import; pure engineering measurement | N/A |
| C14-CFSHIFT (frame-novelty slot 1) | (same as C14 above) | | |
| F2-CKDJ (frame-novelty slot 2) | Softmax shift-invariance | **CRITICAL CHECK** — see Section F2-CKDJ below | N/A |
| R9-R1/ATQKV | SVD truncation (Eckart-Young) | YES | N/A |
| R9-R2/RAOK-REACH | Three-tier activation codebook | YES | YES — tier boundaries from calibration Jaccard, not fitted weights |
| R9-R3/HSMLA | Frobenius-optimal shared-basis extraction via stacked SVD | YES — but see TransMLA prior art below | N/A |
| A2-APPENDKV | LSH over INT8 key vectors | YES — LSH preconditions: approximate cosine similarity, not exact; INT8 approximation is intentional | N/A |
| A3-FSMDEC | k-means centroid assignment | YES | N/A |
| U1-B-ETWO | ETW kernel I/O events; page-fault fallback | YES — documented API, no theorem | N/A |
| U5-A-VGTP | VEH guard-page interposition | YES — Windows VEH is a standard mechanism | N/A |
| R9-R5/LAYERONE-FOLD | AERO algebraic fold (c_i ≈ constant) | YES | YES — see C14-CFSHIFT equivalence |
| C17-LAYERTAX | Spearman correlation of two proxy signals | YES | N/A |

**F2-CKDJ CF9 critical check (RoPE, from prompt warning re: run 003 Stage 6):**
F2-CKDJ's load-bearing algebraic identity is softmax shift-invariance: adding a constant vector to ALL query vectors (equivalently subtracting from all logits) leaves softmax unchanged. The concern from run 003 (F3-SRSC CF9 failure) was that RoPE rotates keys position-by-position, breaking shift-invariance claims that involve the key side. **CKDJ's strip is on the QUERY side only:** the static outlier channels contribute a constant component to the query vector q_h = W_Q[:, static_channels] × x[static_channels]. This is a constant QUERY offset δq_c — a token-independent additive term applied to every head's query. Softmax shift-invariance applied on the query side: adding δq_c to q_h shifts all logits by δq_c^T K (where K is the key matrix of all positions). This is NOT a constant scalar per position; it depends on k_t for each key position t. Therefore the "constant logit bias" interpretation is position-dependent and NOT strippable by the shift-invariance argument without additional structure.

**Verdict on F2-CKDJ CF9:** The Stage 1/Stage 2 mechanism description partially misapplies the shift-invariance identity. Shift-invariance requires adding a scalar constant to all logits (the full row of attention scores). CKDJ's δq_c contributes δq_c^T k_t which varies per position t. The strip is NOT lossless in general. What IS valid: the static outlier channels contribute a predictable query-direction component that can be precomputed and potentially used to reduce W_Q's input dynamic range. The "zero-cost lossless strip" claim is **BROKEN under CF9**. The surviving primitive: static outlier channels as high-energy W_Q input directions, measurable and useful for W_Q quantization precision allocation — but not a computation-graph-rewrite via shift-invariance. **REGENERATE class.**

---

## 1. F1-MQKP — Attention Product M = W_Q W_K^T Direct Spectral Compression

**Stage 2 verdict: ADVANCE (total 13, highest in union)**
**Path 4 elegant-equivalence candidate (algebraic-identity tag requested)**

### Mechanism decomposition

M1: The attention score operator for layer L is M^L = W_Q^L (W_K^L)^T ∈ ℝ^{d×d}.
M2: The MAP-optimal rank-r approximation of M^L minimizing ‖M^L − M̃^L‖_F is the rank-r SVD of M^L (Eckart-Young).
M3: Compressing W_Q and W_K independently at equal rank K_Q = K_K = K gives ‖M − W̃_Q W̃_K^T‖_F ≥ ‖M − M̃‖_F for any r = min(K_Q, K_K) — the product-SVD is strictly better.
M4: If r_99(M^L)/d < min(r_99(W_Q^L)/d, r_99(W_K^L)/d), joint product compression at rank r beats independent compression at matched bit budget.
M5: At inference, replace W_Q^L and W_K^L with (U_M^L Σ_M^{L,1/2}) and (V_M^L Σ_M^{L,1/2}), no change to the attention score computation.

### Per-claim prior-art status

**M1:** ADJACENT. QSVD (arXiv:2510.16292, Oct 2025) compresses [W_Q, W_K, W_V] jointly by stacking them. CARE (arXiv:2603.17946, Mar 2026) applies covariance-weighted SVD to K and V. Neither computes the attention score operator M = W_Q W_K^T as the compression target. LatentLLM (arXiv:2505.18413, May 2025) does "joint QK compression" iteratively but not via product-SVD. Differentiation: all published methods compress the weight matrices themselves as separate objects; MQKP compresses their PRODUCT — the attention score operator — which is the algebraically correct target by Eckart-Young. No paper found that takes SVD of M = W_Q W_K^T as the compression primitive.

**M2:** ADJACENT. Eckart-Young is published (1936). Its application to M = W_Q W_K^T for attention compression is **NOVEL** as of 2026-05-09. `elegance-class: algebraic-identity`.

**M3:** NOVEL. The explicit proof that per-factor compression is suboptimal vs product-SVD at matched rank (the cross-term bound) is not in QSVD, CARE, or any paper found. No paper found making this claim as of 2026-05-09.

**M4:** NOVEL. The structural question "does M = W_Q W_K^T have r_99/d < min(r_Q, r_K)?" is directly measurable and unanswered. No paper found reporting M's spectrum as of 2026-05-09.

**M5:** ADJACENT. AQFKV (prior run, internal) showed K=128 global GO; MQKP is the mathematically superior compression at equal budget.

**Closest published cousin pair:** QSVD (arXiv:2510.16292) + AQFKV (internal finding). Composition: "compress QK jointly using concatenation + shared down-projection." MQKP's value-add over this pair: using the product M = W_Q W_K^T as the compression target (not concatenation), which is algebraically optimal by Eckart-Young — a structural claim the cousin pair does not make.

### CF9 frame-mismatch check

Imported object: Eckart-Young theorem.
Precondition: applies to any real matrix M.
M = W_Q W_K^T is a real matrix.
Precondition holds trivially. No CF9 risk.

RoPE check: at inference, xW_Q → xU_M Σ^{1/2} and xW_K → xV_M Σ^{1/2}. RoPE is applied to q_h and k_h after the linear projections. The factored form still produces the same raw query and key vectors (same linear operation on x), so RoPE applies identically to the factored basis vectors. No RoPE incompatibility.

### CF10 calibration ill-conditioning

No calibration fit. The product M is computed exactly from the trained weights. No CF10 risk.

### Residency arithmetic (Qwen3-1.7B)

W_Q + W_K per layer: 2 × 2048×2048 × 2B = 32 MB/layer × 28 = 896 MB baseline.
CF11 independent K=512 each: 2 × 2 × (2048×512) × 2B × 28 = 448 MB.
If M's r_99/d ≤ 0.50 (GO hypothesis): joint K=400 gives 2×(2048×400)×2B×28 = 179 MB.
Saving vs CF11 independent: 448 MB → 179 MB, 60% further reduction on Q+K block.
70B scale: W_Q + W_K at K=400 product SVD ≈ 3.5 GB vs ~7 GB independent K=512. Material.

KV cache: unchanged (this compresses weights, not KV). Activation footprint: unchanged.
Total residency: 70B at IQ4_XS (43 GB) - 3.5 GB (Q+K saving) ≈ 39.5 GB on NVMe. With NanoQuant 4.95 GiB floor, DRAM-resident if M compression applied on top.

Compute envelope: M = W_Q W_K^T is computed once offline. At inference, xU_M Σ^{1/2} and xV_M Σ^{1/2} are two GEMVs of dimension (d × K) instead of (d × d) each. At K=400 vs K=2048: 5× compute reduction on the Q and K projection side. DRAM bandwidth dominates for d=2048; no change to bottleneck type.

CF13/CF15 independence: residency arithmetic above uses only confirmed CF11 numbers. No CF13/CF15 dependency.

### Smallest-test sharpening

Script: `scripts/stage3_mqkp_pilot.py`
Model: Qwen3-1.7B-Base bf16
Layers: L7, L14, L21 (3-layer pilot)
Protocol: load W_Q^L, W_K^L; compute M^L = W_Q @ W_K.T (each 2048×2048); full SVD (np.linalg.svd); compute var@{256, 300, 350, 400, 450, 500} = cumulative variance fraction; compare against CF11 per-factor baselines.
Eval: apply M^L factorization at K=350,400; measure ΔNLL on 455-token WikiText-2 held-out.
Output: `experiments/stage0/ladder_v2/round9_mqkp/`
Wall-clock: 3 matrix multiplications (2048×2048 × 2048×2048) = ~18 FLOPs each, ~90s each on Ryzen; 3 full SVDs = ~3min each; total ~15 min.
Well within 8h gate.

GO: var@350(M^L) ≥ 0.99 in all 3 layers (M concentrates faster than W_Q's r_99/d=0.63 = need K=1289 for 99%) AND ΔNLL ≤ +1.0 nats at K=350 (tighter than CF11 K=128 at +0.98, because product compression is Eckart-Young optimal).
NO-GO: M spectrum is no more concentrated than W_Q alone; product-SVD offers no gain over independent compression at matched rank.
GRAY: M concentrates marginally better (r_99/d = 0.55–0.62) but quality improvement vs independent compression is < 0.05 nats → run 28-layer aggregate and check cumulative.

### Updated risk profile

| Risk | Mitigation |
|---|---|
| M = W_Q W_K^T may be full-rank if W_Q and W_K have near-orthogonal column spans | 3-layer pilot is the cheapest falsifier; spectrum measurement is the first rung |
| QSVD's concatenation approach might achieve similar compression at lower compute cost | Compare M-product-SVD vs QSVD-style concatenation SVD at matched K in pilot |
| Quality cost compounds with W_K compression (CF11 W_K K=512 +0.29 nats) | Joint M factorization replaces both independent compressions; ΔNLL should be lower by the cross-term argument |

Cheapest falsifier: compute M^L spectrum at 3 layers (~15 min). If var@350 < 0.99, the entire class is dead.

### Two-paper composition flag

Closest cousin pair: AQFKV (internal, prior run) + QSVD (arXiv:2510.16292, Oct 2025).
Composition would give: "compress Q and K jointly using concatenated SVD." MQKP's value-add: the product W_Q W_K^T is the algebraically optimal compression target by Eckart-Young, not the concatenation — a structural claim the cousin pair leaves on the table.
Not engineering-integration: the algebraic-identity distinction (product SVD vs concatenation SVD) is a structural research claim.

**Verdict: REFINE.** Highest-priority Stage 5 candidate. Path 4 algebraic-identity tag confirmed: `elegance-class: algebraic-identity`.

---

## 2. R9-R1 / ATQKV — W_V / W_O Spectrum Audit + Four-Matrix Cascade

**Stage 2 verdict: ADVANCE (12), highest A1 (=2) in Track B**

### Mechanism decomposition

M1: W_V and W_O spectra in Qwen3-1.7B are unmeasured.
M2: If r_99(W_V)/d ≤ 0.85 and r_99(W_O)/d ≤ 0.85, SVD truncation closes the full four-matrix attention compression cascade.
M3: The joint ΔNLL of W_Q@K=128 + W_K@K=512 + W_V@K_V + W_O@K_O ≤ total quality budget.

### Per-claim prior-art status

**M1:** NOVEL. No run (1-8) measured W_V or W_O spectrum on Qwen3. No published paper found reporting W_V / W_O r_99/d on Qwen3-1.7B as of 2026-05-09.

**M2:** PARTIAL. Layer-wise Dynamic Rank (arXiv:2509.25622, Sep 2025) found "effective rank of W_Q and W_K lower than W_V" — a structural insight suggesting W_V may actually be HARDER to compress. This is a risk flag for M2.

**M3:** NOVEL for the specific Qwen3-1.7B numbers. No paper found doing this four-matrix joint ΔNLL accounting on Qwen3 as of 2026-05-09.

### CF9

All four compressions use SVD truncation (Eckart-Young). Precondition holds trivially. RoPE applies to q/k post-projection; W_V and W_O are unaffected by RoPE. No CF9 risk.

### Prior-art risk on W_V

Layer-wise Dynamic Rank (arXiv:2509.25622) explicitly notes W_V has higher effective rank than W_Q/W_K. If W_V r_99/d ≥ 0.90, M2 fails for W_V — and the cascade loses the W_V compression rung. This is the highest risk. Stage 3 pressure-test asks exactly this question.

### Residency arithmetic

If W_V r_99/d = 0.79 (matching W_K): W_V K=512 saves same as W_K K=512 = +0.29 nats additional.
If W_O r_99/d = 0.70: W_O at K=400 gives ~+0.30 nats.
Joint budget at full cascade: W_Q (+0.98) + W_K (+0.82 at K=256) + W_V (+0.29 est.) + W_O (+0.30 est.) = ~2.39 nats total. This EXCEEDS the quality budget for a production deployment. The cascade's quality cost is the critical unknown — the ΔNLL sum may be additive or may partially cancel (if attention score reconstruction is what matters and M = W_Q W_K^T is the right target, then MQKP's product-SVD at K=400 may actually cost less than ATQKV's W_Q+W_K at separate K values for the same residency).

With CF13/CF15 removed from arithmetic: residency savings are real regardless. At 70B, all four attention matrices compressed to 4× = 25% of weight slice saved, ~3-5 GB on 70B NVMe-resident model.

### Smallest test

Script: `scripts/stage3_atqkv_pilot.py`
Protocol: load W_V and W_O for all 28 layers; SVD (use scipy.sparse.linalg.svds at rank 1024 for speed); compute var@{256,512,768,1024}; then four-matrix joint ΔNLL at K_V=K_O=512 with W_Q@128 + W_K@256 fixed.
Wall-clock: 2 SVDs × 28 layers ≈ 30 min + eval ≈ 10 min = ~40 min total.

GO: W_V r_99/d ≤ 0.85 AND W_O r_99/d ≤ 0.85 AND joint ΔNLL ≤ 2.0 nats (aggressive but survivable quality budget).
NO-GO: W_V r_99/d ≥ 0.90 → kills attention-cascade class; extends CF8 boundary to include W_V.
GRAY: W_V ≤ 0.85 but joint ΔNLL 2.0–3.0 nats → test at K_Q=256, K_K=512, K_V=512, K_O=512 (relaxing W_Q to recover quality).

**Verdict: REFINE.** Must run before MQKP cascade deployment can be fully characterized; the W_V question is a load-bearing structural measurement that closes or opens the full attention-weight class.

---

## 3. F4-WQKRS — W_Q/W_K Spectral Asymmetry Water-Filling Schedule

**Stage 2 verdict: ADVANCE (12)**

### Mechanism decomposition

M1: W_Q r_99/d ≈ 0.63, W_K r_99/d ≈ 0.79 (CF11). Asymmetry = 0.16 gap.
M2: At fixed total bit budget (K_Q + K_K = K_total), water-filling assigns more rank to W_K (flatter spectrum = needs more rank) and less to W_Q (steeper decay).
M3: The optimal (K_Q*, K_K*) satisfies σ_{K_Q*+1}(W_Q) = σ_{K_K*+1}(W_K) (Lagrangian condition for equal marginal distortion).

### Per-claim prior-art status

**M1-M2:** PARTIAL. Layer-wise Dynamic Rank (arXiv:2509.25622, Sep 2025) proposes "reallocating rank from W_Q/W_K to W_V" — an asymmetric allocation, but across matrices not within the W_Q/W_K pair. Water-filling within (K_Q, K_K) subject to K_Q+K_K=const is not addressed.

**M3:** NOVEL. The specific Lagrangian condition (equal marginal singular values between W_Q and W_K) for joint rank allocation is not published for attention compression as of 2026-05-09.

### CF9

Water-filling is classical Shannon-1948 applied to matrix approximation. Preconditions: continuous spectrum (real), separable per-matrix distortion. Both hold for SVD approximation. No CF9 risk.

### Residency arithmetic

At K_total = 1020 (equivalent to symmetric K=510 each):
Water-filling: K_Q* ≈ 350 (W_Q decays faster), K_K* ≈ 670 (W_K needs more rank).
Bit count: 2048×(350+670)×4B×28 = same as symmetric K=510 each.
Quality gain: by Lagrangian optimality, ΔNLL(water-fill) ≤ ΔNLL(symmetric) at same budget. Expected gain: ~0.05-0.15 nats at 3-layer pilot, larger at full 28 layers.

The gain is quality-at-fixed-cost, not residency reduction. The real payoff is enabling K_total = 800 (K_Q*=280, K_K*=520) at same quality as symmetric K=510 → 22% fewer bytes stored.

### Prior-art risk (water-filling in attention)

Searching for "water-filling attention W_Q W_K" returned nothing specific. COMPOT (arXiv, 2025 — from search) uses "optimized rank allocation" but based on functional loss, not per-matrix spectral asymmetry. The water-filling framing (equal marginal σ as Lagrangian condition) is not in the published field.

### Smallest test

Script: `scripts/stage3_wqkrs_pilot.py`
Protocol: full SVD W_Q and W_K at 3 layers; compute water-filling (K_Q*, K_K*) from equal-σ condition at K_total=1020; compare ΔNLL vs symmetric K=510 each.
Wall-clock: ~30 min.

GO: water-filling ΔNLL ≤ symmetric − 0.05 nats at all 3 layers.
NO-GO: symmetric and water-filling identical → W_Q/W_K asymmetry doesn't translate to quality-at-budget advantage; attention scores are insensitive to rank asymmetry at CF11 compression levels.
GRAY: improvement in 1 of 3 layers but not all → run full 28 layers to get statistical power.

**Verdict: REFINE.** Short runtime, clean decisive gate, motivated by confirmed CF11 asymmetry.

---

## 4. F2-CKDJ — CF3 Outlier Channels × CF11 W_Q Subspace Alignment

**Stage 2 verdict: ADVANCE (12), frame-novelty slot 2**

### CF9 critical finding — CKDJ fail

As established in the gate table above and detailed in the prologue: F2-CKDJ's mechanism rests on the claim that the static outlier channels' contribution to the query vector is a "constant component strippable by softmax shift-invariance." The softmax shift-invariance identity requires a constant SCALAR added to all logits in the same row. CKDJ's δq_c contributes δq_c^T k_t which is position-dependent (varies with each key token k_t). This is NOT a constant scalar per row of the attention matrix. The "lossless strip" claim is therefore mathematically broken under CF9.

**What survives:** The empirical coupling claim (do the 2 static outlier channels lie in the top-5% of W_Q column norms?) is still a valid measurement that closes a structural question about CF3/CF11 independence. If YES: the static channels are important to W_Q and should be pinned at higher precision during W_Q quantization. If NO: CF3 and CF11 are decoupled phenomena.

**Stripped primitive:** "Static outlier channels (CF3 K=0.1%) as high-energy W_Q input columns" — a measurement useful for precision allocation in W_Q quantization, adjacent to SageAttention2 (arXiv:2411.10958, Feb 2025) and FireQ (arXiv:2505.20839, May 2025) which both smooth outlier channels for quantization.

**Closest published cousin pair:** PrefixQuant (arXiv:2410.05265) + SageAttention2 (arXiv:2411.10958). Composition: "handle static outlier channels in activation quantization + smooth W_Q outlier input directions." CKDJ adds: the coupling claim that these are the same channels — a structural measurement. But the "lossless strip" mechanism is gone.

**Verdict: REGENERATE.** The load-bearing mechanism (shift-invariance strip) is broken under CF9. The surviving stripped primitive (static outlier channel × W_Q column norm alignment) is real measurement worth doing, but it is ADJACENT to SageAttention2/FireQ rather than novel. Pass to Stage 4: "measure whether CF3 static channels = top W_Q column energy — if YES, motivates per-column precision allocation for W_Q quantization." Not a Stage 5 target this round.

---

## 5. F3-L1GNF — Layer-1 Gate Null-Space Fold

**Stage 2 verdict: ADVANCE (11), Track A**

### Mechanism decomposition

M1: CF6's 36.1% foldable layer-1 neurons are neurons with gate rows primarily in null(C_x^0) — the null space of the layer-1 input covariance (token embeddings).
M2: Gate rows in null(C_x^0) produce near-zero gate output for all high-frequency tokens → AERO algebraic fold applicable.
M3: The null-space criterion is a zero-calibration-sample fold predictor: if confirmed, fold eligibility can be predicted from weight geometry alone.

### Per-claim prior-art status

**M1:** NOVEL. The null-space alignment as explanation for layer-1 anomaly is not in the published literature. AERO (arXiv:2410.13060) removes activations globally with retraining; this is post-training + per-neuron + layer-specific. No paper found testing null(C_x^0) as a fold predictor as of 2026-05-09.

**M2:** ADJACENT. AERO (arXiv:2410.13060) provides the fold identity. L1GNF uses the fold post-training on CF6-identified neurons — structurally different.

**M3:** NOVEL. The no-calibration-sample predictor from weight geometry alone is not published as of 2026-05-09.

### CF9

Imported object: eigendecomposition of C_x^0.
Precondition: C_x^0 is positive semidefinite by construction (empirical covariance = X^T X / n). Holds trivially.
Null-space projection is a standard linear algebra operation. No CF9 risk.

### CF10

c_i = mean over calibration of silu(W_gate^{L1}·x)_i for foldable neurons.
n_params = 6144 × (fraction foldable ≈ 0.361) ≈ 2218 scalars.
n_independent_samples = 200 tokens × (different token IDs), n_output_dims_per_sample = 1 per neuron.
n_params (2218) << n_samples × d_output (200 × 1 = 200). Wait — this is 2218 > 200 in terms of parameters to samples. CF10 check: this is borderline. Mitigation required: use 500+ calibration tokens (as the proposal mentions) to ensure n_samples > n_foldable_neurons (2218). The proposal correctly flags this risk. With 500 tokens: 2218 < 500 → still under-determined. **CF10 REQUIRES MITIGATION:** use 2500+ tokens (WikiText-2 training split), or enforce strong regularization (ridge λ = 0.1). Alternatively, note that c_i for the FOLD is a single scalar per neuron (one parameter per neuron, not one per token), and each of the 500 tokens provides an independent estimate of c_i → the mean is well-conditioned at 500 tokens with n_eff_samples = 500 >> n_params/d_output = 2218/1. Actually the fit here is estimating the MEAN of a scalar per neuron — c_i = E[silu(W_gate^L1·x)_i] — which needs only n_tokens >> 1 for stability of the mean. This is well-conditioned at 200 tokens; each token is an independent draw. CF10 passes with 200+ diverse tokens.

### Smallest test (refined)

Script: `scripts/stage3_l1gnf_pilot.py`
Protocol: (1) compute C_x^0 from 500 WikiText-2 tokens; (2) eigendecompose; (3) project W_gate^{L1} rows onto top-256 eigenvectors; (4) compare foldable vs non-foldable distribution; (5) apply fold to foldable neurons; (6) evaluate ΔNLL on held-out 455 tokens.
Wall-clock: ~25 min.

GO: mean null-space fraction for foldable neurons ≥ 0.70 AND effect size > 0.30 vs non-foldable AND ΔNLL ≤ 0.05 nats on held-out.
NO-GO: null-space projection indistinguishable → CF6 anomaly is not a geometric property but a calibration-distribution artifact.
GRAY: null-space fraction 0.50–0.70 → measure at C_x^0 rank 512 (more accurate eigenbasis); check if the 0.70 threshold is met at higher rank.

**Verdict: REFINE.** Clean mechanism, well-motivated, no prior art on the specific null-space criterion. Low wall-clock.

---

## 6. F5-RSUC — Residual-Stream Update Covariance Depth Profile (C2 cluster rep)

**Stage 2 verdict: ADVANCE (11), Convergence C2 representative**

### Mechanism decomposition

M1: Residual update δ^L = h^{L+1} − h^L has a covariance C^L = E[δ^L δ^{L,T}].
M2: The effective rank r_{90%}(L) of C^L varies by depth (early/late < mid) by ≥ 40%.
M3: This depth profile motivates a per-layer compression budget that is tighter for early/late layers and looser for mid-layers.

### Per-claim prior-art status

**M1:** PARTIAL. FlattenGPT (arXiv:2602.08858) studies cross-layer similarity and residual dominance. "Variance Is Not Importance" (arXiv:2604.20682) uses cross-covariance analysis for compressibility. Neither directly measures r_{90%}(δ^L) as a function of layer depth and proposes a compression budget from it.

**M2:** PARTIAL. Layer-wise sensitivity analysis (GPTQ, OmniQuant, AWQ) uses Hessians for per-layer budget. RSUC proposes using the update covariance rank directly, without calibration data for the budget assignment step — a simpler, interpretable proxy.

**M3:** NOVEL. The specific depth-profile-driven budget schedule (early/late compress more, mid compress less) based on r_{90%}(δ^L) is not published as of 2026-05-09.

### Prior-art check on "Variance Is Not Importance" (arXiv:2604.20682)

This 2026 paper appears to directly address structural compressibility via variance analysis. The search result states: "structural properties that make individual blocks compressible (approximate linearity, concentrated variance) do not compose across blocks." This is relevant — it suggests the depth profile may NOT be a clean predictor of compressibility once cross-block interactions are considered. This is a risk to M3: even if r_{90%}(δ^L) varies by depth, the compressibility of the individual weight matrices at that layer may not correlate with the update covariance rank. RSUC's Stage 3 pressure-test must explicitly check whether r_{90%}(δ^L) correlates with ΔNLL sensitivity (from CF11 W_Q and the asymmetric W_K data).

### CF9

Eigendecomposition of empirical covariance. No imported theorem with fragile preconditions. No CF9 risk.

### Residency arithmetic (depth-heterogeneous schedule)

At 1.7B, depth-aware budget gives ~31% compression vs 33% for uniform K=512 — near-identical residency but higher quality at fixed budget. The payoff is quality-at-budget, not absolute residency. At 70B where mid-layer count is larger, the compounding is more material.

Conservative path (if CF13/CF15 don't replicate): the residency arithmetic is not CF13-dependent; it uses only confirmed CF11 numbers. Safe.

### Smallest test (refined)

Script: `scripts/stage3_rsuc_pilot.py`
Protocol: 200 WikiText-2 tokens forward pass; collect δ^L for all 28 layers; compute C^L = δ^{L,T}δ^L/n; eigendecompose; report r_{90%}(L) per layer; also extract from 3 disjoint 67-token batches for stability check.
Wall-clock: ~25 min.

GO: r_{90%}(L=1) and r_{90%}(L=28) both ≤ 0.60 × r_{90%}(L=14).
NO-GO: r_{90%} approximately constant → residual updates have uniform directional diversity; depth-aware budget gives no gain over uniform.
GRAY: weak stratification (0.60–0.85 ratio) → check correlation of r_{90%}(L) with ΔNLL(W_Q@K_L) across per-layer K values; if no correlation, the profile is measurement noise.

**Verdict: REFINE.** The "Variance Is Not Importance" paper (arXiv:2604.20682) adds a risk flag that the profile may not predict compressibility; the GRAY zone follow-up addresses this. Still worth running: the depth profile measurement itself (r_{90%}(L) across 28 layers) is a new empirical characterization.

---

## 7. A1-TROPFFN — Tropical Semiring FFN Dispatch (C3 cluster rep)

**Stage 2 verdict: ADVANCE (11), Convergence C3 representative**

### Mechanism decomposition

M1: W_up and W_gate weights can be log-quantized (per-row INT8 log₂-magnitude + sign bit = 9 bits/weight).
M2: Tropical GEMV (max-plus): output_i = max_j(W_{ij} + x_j) in log space, computable with AVX2 vpaddb + vpmaxsb.
M3: Tropical dispatch approximates CF1's top-K firing structure (W_up·x dominance) with recall@K=5% ≥ 0.70.
M4: At K=5%, tropical dispatch selects the same top neurons as standard GEMV, enabling selective full-precision computation on the top-K neurons only.

### Per-claim prior-art status

**M1:** NOVEL. Log-quantized weights for tropical GEMV applied to LLM MLP post-training inference is not in the published literature as of 2026-05-09. Tropical Neural Networks (Charisopoulos & Maragos, 2020) use tropical layers from scratch (training) — not post-hoc substitution.

**M2:** ADJACENT. Max-pooling (max-plus GEMV) is a standard operation in neural architecture. Applying it as an inference-time approximation of LLM MLP firing is not published.

**M3:** NOVEL. No paper found testing tropical GEMV recall against standard GEMV for LLM token-generation firing patterns as of 2026-05-09.

**M4:** PARTIAL. Deja Vu (arXiv:2310.17157) predicts top-K neurons with a small predictor network. Tropical dispatch replaces the predictor with a max-plus GEMV — zero additional parameters, structurally different.

### CF9

Log-quantization precondition: assumes W_up has per-row dynamic range capturable in INT8 log scale after per-row normalization. CF5 shows W_up is rank-sensitive and high-dynamic-range — per-row normalization is the stated mitigation. The precondition (per-row normalizability) must be verified in the smallest test (measure max/min per-row ratio before log-quantization). No fragile theorem import; max-plus GEMV is the computation itself.

### CF10

No calibration fit. Per-row normalization scales are computed from weights directly. No CF10 risk.

### Residency arithmetic

Log-quantized: 9 bits/weight vs 16 bits bf16 = 1.78× compression. At 70B: 140 GB × 9/16 = 78.75 GB — worse than IQ4_XS. This mechanism is NOT about residency; it is about throughput in the compute-bound (L3-hot) regime.

At Ryzen 5 7530U AVX2: tropical GEMV on INT8 log-magnitudes runs at estimated ~4 cycles/neuron vs ~32 cycles for bf16 GEMV. For a batch of 1 layer: 8× compute speedup on the FFN in the L3-resident case. But DRAM bandwidth is the bottleneck for NVMe-resident 70B inference, not compute. The tropical mechanism matters only when the model fits in L3 (≤16 MB = sub-7B at extreme quantization, or hot-layer scenario).

**Revised scope:** A1-TROPFFN is a valid free-swing idea but its applicability to the primary 70B problem is limited. In the L3-hot regime (hot-layer partial residency à la OSRD/VGTP), tropical dispatch provides real compute speedup.

### Smallest test (refined)

Script: `scripts/stage3_tropffn_pilot.py`
Model: Qwen3-1.7B-Base, one W_up layer
Protocol: (1) check per-row dynamic range (log10(max_weight / min_abs_weight) per row); (2) per-row normalize → INT8 log₂-quantize; (3) implement numpy tropical_gemv = max over (W_log + x_log); (4) measure recall@K=5% vs bf16 standard GEMV on 200 calibration tokens.
Wall-clock: ~2 hours.

GO: recall@K=5% ≥ 0.70 → tropical dispatch approximates the load-bearing firing structure.
NO-GO: recall@K=5% ≤ 0.40 → the log-quantization error (from weight dynamic range or sign ambiguity) destroys the approximation. Structural finding: W_up's dynamic range is incompatible with per-row INT8 log-quantization.
GRAY: recall 0.50–0.70 → test with per-channel (not per-row) normalization; check if recall improves to ≥ 0.70.

**Verdict: REFINE — wildcard, CF-tether-suspended, structural floor met.** A1-TROPFFN has no CF9/CF10 failure, the smallest test is ≤8h, and the mechanism is grounded in a real CPU primitive (AVX2 max-plus). However, its A1 (=1) reflects the limited applicability to the primary 70B NVMe path. Relevant primarily as a hot-layer compute accelerator in combination with OSRD/VGTP.

---

## 8. R9-R4 / NVME-ATTN-STREAM — NVMe Interleaved Layout (C4 cluster rep)

**Stage 2 verdict: ADVANCE-convergence (9)**

### Mechanism decomposition

M1: Per-layer attention weight bytes < MLP weight bytes (at any bpw, ~31% vs 69%).
M2: Interleaved GGUF layout (attention-then-MLP per layer) enables attention weights for layer L+1 to be prefetched during MLP compute of layer L.
M3: If MLP compute time ≥ attention load time at NVMe speed, attention arrives "free" → 1.44× throughput at NVMe tier.
M4: CF13* (NVMe sequential read throughput) must be re-derived as first rung.

### Per-claim prior-art status

**M1-M2:** PARTIAL. Apple LLM-in-a-Flash (arXiv:2312.11514) uses sequential NVMe weight loading. The specific interleaved per-layer attention-MLP layout with compute-overlap arithmetic is not in LLM-in-a-Flash (which uses sliding-window caching, not interleaved layer layout).

**M3:** NOVEL. The compute-overlap arithmetic (MLP compute time vs attention load time at 0.55 bpw NVMe) with the 1.44× throughput claim is not published as of 2026-05-09.

**M4:** Required cascade rung — CF13* re-derivation is the first experiment.

### CF9

No imported theorem. Pure systems engineering + arithmetic. No CF9 risk.

### Residency arithmetic (detailed)

At NVMe 3 GB/s (CF13* assumed — must re-derive):
70B at 0.55 bpw: per-layer attention = 0.14 GB, per-layer MLP = 0.31 GB.
MLP load time: 0.31 / 3.0 = 103 ms/layer.
Attention load time: 0.14 / 3.0 = 47 ms/layer.
Since 47 ms < 103 ms, attention can be fully prefetched during MLP load. Net effective bandwidth: total bytes / (layers × max(attn_load, mlp_load)) = 80 layers × (0.14+0.31) GB / (80 × 103 ms) = 36 GB / 8.24 s = 4.37 GB/s effective vs 3.0 GB/s nominal = 1.46× gain. This matches the M3 claim.

Conservative path (if CF13* < 2.5 GB/s, e.g., 1.5 GB/s NVMe under actual inference load):
At 1.5 GB/s: attention 0.14/1.5 = 93 ms, MLP 0.31/1.5 = 207 ms. Overlap still holds (93 < 207). Effective bandwidth: 0.45 GB/layer / 207 ms = 2.17 GB/s. 1.45× gain still holds. The mechanism is robust to NVMe underperformance.

At DRAM-resident (4.95 GiB, 11.5 GB/s): mechanism is irrelevant; DRAM is not the bottleneck for GEMV compute on CPU. R9-R4 contributes only in the NVMe-tier regime.

### Smallest test (CF13* re-derivation)

Script: `scripts/stage3_nvme_cf13_star.py`
Protocol: timed mmap sequential read of 1 GB GGUF file under active inference-like memory pressure (load the 1.7B GGUF + run a 10-token forward pass simultaneously to simulate actual inference I/O pattern); record GB/s.
Wall-clock: ~10 min.

GO: ≥ 2.5 GB/s NVMe sequential under inference workload.
NO-GO: < 2.5 GB/s → NVMe is slower than expected; the overlap arithmetic still holds (1.44× is invariant to absolute bandwidth), but the absolute tok/s at 0.55 bpw NVMe would be lower than estimated.
GRAY: 1.5–2.5 GB/s → recalculate tok/s estimate with actual number; cascade still valid at any speed ≥ 1 GB/s.

**Verdict: REFINE.** The mechanism is architecture-correct and the compute-overlap arithmetic is sound. The 1.44× throughput improvement is robust across a wide NVMe bandwidth range. Stage 4 should pair this with DRAM-resident path (MQKP + RAOK).

---

## 9. C14-CFSHIFT — CF6 Constant Injection × W_Q[L2] Logit Pre-bias

**Stage 2 verdict: ADVANCE (11), frame-novelty slot 1**

### Mechanism decomposition

M1: CF6's foldable layer-1 neurons contribute c_1 = mean(silu(W_gate^{L1}·x)_i × W_up^{L1}_i for foldable i) ∈ ℝ^{2048} to h_2.
M2: δq_c1 = W_Q^{h,L2}·c_1 is a constant head-specific query offset (computed once at load time).
M3: The cross-term δq_c1^T δk_c1 / √d is a constant per-head scalar logit bias.
M4: If ‖δq_c1‖₂ / mean‖W_Q^{L2}h_2‖₂ ≥ 0.10, precomputation of the cross-term reduces Layer-2 attention compute.

### Per-claim prior-art status

**M1:** PARTIAL. CF6 is confirmed. The mean-field approximation (c_1 ≈ constant) is the load-bearing claim here. The variance check is key: if std(foldable neuron contribution) > 10% of mean, c_1 is not truly constant and M1 fails.

**M2-M3:** NOVEL. No published method derives a constant attention logit bias from the preceding layer's gate-fold statistics. ALiBi adds position bias; this adds a calibration-content bias. No paper found as of 2026-05-09.

**M4:** NOVEL. The specific pre-computation enabling Layer-2 logit partial bypass is not published.

### CF9

Imported: linearity of W_Q over vector addition (W_Q(a+b) = W_Qa + W_Qb).
Precondition: holds trivially for any linear layer.
RoPE treatment: δk_c1 = W_K^{L2}·c_1 is a known constant vector; at position p, RoPE rotates δk_c1 by R(p). The rotated vector R(p)δk_c1 is still precomputable per position p — it's a table of precomputed vectors indexed by p. The precomputation claim still holds. **CF9 PASSES for this mechanism.**

### CF10

n_params = ~2218 (foldable neuron count) scalar gate outputs → 2218 scalars to estimate.
n_independent_samples = 200 tokens × 2048 dims (actually n = 200 tokens, each providing c_i estimate).
The c_1 vector is the mean of {foldable neuron output across 200 tokens} — 200 independent scalars per c_i coordinate. 200 >> 1 parameter per coordinate. **CF10 passes easily.**

### Prior-art search

"Attention logit constant bias calibration gate output prior layer injection" — no relevant papers found matching this specific mechanism. ALiBi, xPos, YaRN all add position-dependent biases. None derives a calibration-content bias from a prior layer's gate statistics.

### Stage 3 pressure-test

The critical question: is std(c_1 contribution per token) < 10% of mean magnitude?

If std is large (e.g., > 50% of mean), the "constant offset" is actually a variable offset — the mean-field approximation fails. In this case, M1 is not a constant and the precomputation argument collapses.

From CF6: the 36.1% foldable neurons have std(gate output) < 0.05 (very small variance). This means each individual c_i IS near-constant, so c_1 = Σ c_i W_up_i is near-constant as a vector. The std check should pass by construction of CF6's foldable criterion. However, the MAGNITUDE matters: if c_1 is near zero (the foldable neurons have mean ≈ 0), then δq_c1 is also near zero and M4 fails (‖δq_c1‖₂ < 0.10 × mean query magnitude).

### Smallest test (refined)

Script: `scripts/stage3_cfshift_pilot.py`
Protocol: 200 calibration tokens; compute c_1 (mean contribution from CF6-foldable neurons); compute δq_c1^h = W_Q^{h,L2}·c_1 for 16 heads; measure ratio ‖δq_c1^h‖₂ / mean‖W_Q^{h,L2}·h_2(t)‖₂ per head.
Wall-clock: ~30 min.

GO: ≥ 8 heads with ratio ≥ 0.10.
NO-GO: δq_c1 negligible (ratio < 0.02 all heads) → CF6 foldable neurons' contributions are near-zero magnitude; the mechanism produces a near-zero logit bias; CF6 and CF11 are decoupled at the cross-layer attention-logit level.
GRAY: 5–7 heads with ratio ≥ 0.10 → extend to measure δq_c1 magnitude across all 28 layer pairs (L_i → L_{i+1}); CF6 anomaly is L1-specific; check if other layers show any coupling.

**Verdict: REFINE.** No prior art. CF9 pass (including RoPE). CF10 pass. The mechanism is algebraically sound; the empirical question (magnitude of δq_c1) is cheap to measure and decisive.

---

## 10. C15-SNRATE — W_up Sensitivity × CF11 Spectrum as Precision Budget Oracle

**Stage 2 verdict: ADVANCE (11)**

### Mechanism decomposition

M1: SNR(C) = (1 − r_99/d) / ΔNLL@50%_rank as a joint compression signal.
M2: SNR ordering correctly predicts bpw ordering (W_Q >> W_K >> W_gate ≈ W_up >> tied embed).
M3: The SNRATE law (bpw = f(concentration, sensitivity)) generalizes to W_V and W_O without additional calibration.

### Per-claim prior-art status

**M1:** ADJACENT. HAQ, LLM-QP, OmniQuant all use Hessian-based per-layer sensitivity for mixed-precision. SNRATE replaces the Hessian with SVD-derived sensitivity — simpler and directly connected to confirmed CFs. No paper found combining spectrum concentration AND rank-truncation sensitivity as a joint SNR oracle for bpw assignment as of 2026-05-09.

**M2:** NOVEL. The specific bpw discrimination between W_Q and W_gate using the SNR law is a new empirical claim. The key risk: rank-truncation sensitivity ≠ quantization sensitivity (CF5 measures SVD rank truncation; the bpw assignment requires quantization sensitivity).

**M3:** PARTIAL. COMPOT (arXiv, 2025) uses "optimized rank allocation" but with functional loss calibration. SNRATE uses spectrum + rank-sensitivity (weight-only, no activation data). The generalization to W_V/W_O is novel but depends on M3 holding for new weight classes.

### CF9

SNR(C) uses confirmed CF5/CF11 numbers. No imported theorem. No CF9 risk.

### CF10

3-param fit on 5 data points (W_Q, W_K, W_gate, W_up, tied embed). n_params=3, n_data=5. **CF10: well-conditioned by design.** The fit is explicitly specified as 3 params on 5 points; n_params << n_data.

### Critical risk: rank truncation ≠ quantization sensitivity

CF5 measures ΔNLL at 50% rank truncation (SVD). SNRATE uses this as a proxy for quantization sensitivity at 2 bpw. The two are structurally different error distributions (low-rank error is structured in singular-value space; quantization error is approximately uniform grid rounding). If W_Q is sensitive to low-rank truncation but insensitive to 2-bpw quantization (because quantization error is spread uniformly vs concentrated in low-energy singular directions), the SNRATE law would give wrong bpw assignments.

This is the cleanest falsifier: compare ΔNLL(W_Q at 2bpw uniform INT2) vs ΔNLL(W_Q at 50% rank). If CF11's rank sensitivity (ΔNLL@K=512 = +0.03 nats = very LOW sensitivity) correctly predicts W_Q is safe at 2bpw, the SNRATE law holds. If W_Q at 2bpw gives ΔNLL >> +0.50 nats, the two sensitivity types are decorrelated.

### Smallest test (refined)

Script: `scripts/stage3_snrate_pilot.py`
Protocol: (a) quantize W_Q (all 28 layers) to INT2 (per-group round-to-nearest, group size=64) holding all other weights at bf16; eval ΔNLL on 455-token held-out; (b) quantize W_gate to INT2 similarly; compare.
Wall-clock: ~40 min.

GO: ΔNLL(W_Q at 2bpw) ≤ +0.50 nats AND ΔNLL(W_gate at 2bpw) ≥ 2× ΔNLL(W_Q at 2bpw).
NO-GO: ΔNLL(W_Q at 2bpw) ≥ +1.00 nats → rank-truncation sensitivity does NOT predict quantization sensitivity; SNRATE law is invalid.
GRAY: W_Q at 2bpw ≤ +0.50 nats but W_gate gap < 2× → the SNRATE ordering holds directionally but the magnitude isn't discriminative enough for precise bpw allocation.

**Verdict: REFINE.** Critical risk is identified and the experiment directly tests it. Clean 40-min test. Prior art gap confirmed (no published paper uses the joint spectrum + rank-sensitivity SNR for bpw).

---

## 11. C13-EQSUB — Embedding Row Projection into W_Q K=128 Subspace

**Stage 2 verdict: ADVANCE (11), convergence C1**

### Load-bearing claims

M1: μ_E (mean projection mass of vocab embedding rows into per-layer W_Q K=128 principal subspace) ≥ 0.50.
M2: If μ_E ≥ 0.50, W_Q at K=64 may be adequate at Layer 1 (and potentially all layers) because the input already lives in the subspace.

### Prior-art status

**M1:** NOVEL. No paper found measuring E row projection mass into W_Q principal directions as a compression budget signal as of 2026-05-09. TransMLA (arXiv:2502.07864) involves joint KV projection but not E×W_Q alignment.

**M2:** PARTIAL. CF11 per-head K=64 is NO-GO (+1.53 nats). If μ_E ≥ 0.50, M2 claims K=64 at Layer 1 may be different from the global K=64 result. The distinction: global K=64 is over all tokens; Layer-1 K=64 with high μ_E may be OK for the embedding-rich early layer. This requires experimental verification.

### Risk

The proposal's main risk is that "global K=64 NO-GO" (CF11) already rules out K=64 for W_Q at any layer. The μ_E result would only help if Layer 1 specifically tolerates K=64 due to embedding alignment. This is a real but narrow claim — the scaling argument (μ_E ≥ 0.50 → all layers can use K=64) requires the embedding alignment to persist through the residual stream, which CF2 (0.99 layer-cosine) doesn't guarantee for 28 layers.

### Smallest test

Script: `scripts/stage3_eqsub_pilot.py`
Protocol: SVD of W_Q^{L1} (top-128 right singular vectors V_Q^{128}); batch matrix multiply E @ V_Q^{128} to get projection norms; compute μ_E; extend to all 28 layers.
Wall-clock: ~35 min.

GO: μ_E ≥ 0.50 at L1 AND ≥ 0.45 mean across all 28 layers.
NO-GO: μ_E ≈ 0.063 (random expectation for 128/2048-dim subspace) → E rows are isotropic relative to W_Q subspace.

**Verdict: REFINE.** Clean measurement, 35 min, no prior art, directly falsifiable. The scaling argument is speculative but the coupling measurement itself is load-bearing for understanding E↔W_Q training dynamics.

---

## 12. R9-R2 / RAOK-REACH — Tiered Activation Codebook as Cascade Anchor

**Stage 2 verdict: ADVANCE (10)**

### Prior-art status

The RAOK mechanism (3-tier: 2 channels FP16 / 18 channels INT8 / 2026 channels INT4) is directly grounded in CF3/v2-CF1 Jaccard measurements. PrefixQuant (arXiv:2410.05265, Oct 2024) covers channel-wise vs token-wise outlier decomposition. SmoothQuant (arXiv:2211.10438) handles channel-static outliers. The specific K-dependent Jaccard crossover (0.1% static, 1% dynamic) as the tier boundary is unique to this pipeline's CF3 measurements.

**Novel element:** The K-dependent tier boundary derived from CF3 Jaccard crossover — no published method uses this specific criterion. ADJACENT to LLM.int8() + SmoothQuant composition, but with a CF3-grounded tier boundary.

### CF9/CF10

No imported theorem; tier boundary is empirically derived from CF3. No calibration fit for the tier assignment (Jaccard measurement is non-parametric). **Both pass.**

### Residency arithmetic

RAOK as precision-matching enabler: at 2 bpw weight quantization, if activation outlier tiering reduces quantization error by 0.3 nats, the effective quality matches 3 bpw without tiering → 1.5× weight compression gain. At 70B: 36 GB (4bpw) → 24 GB (2.67 bpw effective) via RAOK enablement. Still NVMe tier. The cascade closes only with NanoQuant-level 0.55 bpw. RAOK as enabler is a necessary but not sufficient rung.

**CF13/CF15 independence:** arithmetic uses confirmed CF3/v2-CF1 numbers only. No dependency.

**Verdict: REFINE.** RAOK is the strongest surviving activation-quantization path. The experiment is 2 hours. No prior art on the specific K-Jaccard-derived tier boundary. ADJACENT to LLM.int8() + SmoothQuant but differentiated by the empirically derived tier split.

---

## 13. R9-R3 / HSMLA — Post-Training Shared-Basis MLA

**Stage 2 verdict: ADVANCE (10), Track A**

### Critical prior-art finding: TransMLA

**TransMLA (arXiv:2502.07864, Feb 2025, NeurIPS 2025 Spotlight)** converts GQA-based models (including Qwen, LLaMA) to MLA post-training. It performs joint PCA on K and V weight matrices to extract a shared latent space, achieving 68.75% KV cache reduction with only 1.65% performance drop (training-free variant) or 93% compression with 6B fine-tuning tokens.

**Key differences from HSMLA:**
- TransMLA operates on GQA K+V jointly; HSMLA operates on the 16 W_Q heads for shared basis extraction.
- TransMLA requires fine-tuning to recover performance above the training-free 1.65% drop level; HSMLA targets the post-training W_Q compression from CF11's head-redundancy finding.
- TransMLA does not use the Grassmannian head-pair principal angle geometry that HSMLA proposes.
- TransMLA's "FreqFold" exploits RoPE dimension similarity — a different technique.

**CARE (arXiv:2603.17946, Mar 2026)** applies covariance-aware decomposition to K and V globally per layer. Not W_Q specifically.

**HSMLA's unique claim:** Post-training extraction of a W_Q shared basis from CF11's measured 16:1 head-redundancy (all 16 heads span a ~128-dim joint subspace). This specific W_Q-head factorization is not TransMLA's mechanism.

### Per-claim prior-art status

**M1 (16 W_Q heads collectively span ~128-dim subspace, CF11):** PARTIAL. CF11 confirms this. The v2-CF2 kill (cross-layer W_Q stacking) is NOT the same as within-layer cross-head stacking — HSMLA stacks heads within one layer, not W_Q matrices across layers.

**M2 (shared basis extractable via stacked SVD without retraining):** ADJACENT. TransMLA extracts shared KV basis with fine-tuning; HSMLA claims training-free. The Frobenius-only approach (no activation weighting) is simpler but may lose quality vs CARE's covariance-weighted version.

**M3 (ΔNLL ≤ +1.1 nats at K_shared=128):** NOVEL target. CF11's K=128 global GO (+0.98 nats) is the reference ceiling; HSMLA at same K_shared should match.

### CF9

Stacked SVD of 16 per-head matrices. No fragile precondition. The Frobenius-optimal shared basis is a standard PCA. **CF9 passes.**

### CF10

No calibration fit. Pure weight SVD. **CF10 not applicable.**

### Key risk: within-layer Grassmannian geometry

The CF11 global-K=128 result may not decompose into a shared basis across the 16 per-head matrices. The 16 heads may each span their own distinct 128-dim subspace that only looks shared in aggregate. The HSMLA experiment must first measure Grassmannian principal angles between head pairs — if all pairs have large principal angles (near 90°), the heads are geometrically independent and the shared basis extraction will fail.

### Smallest test (refined)

Script: `scripts/stage3_hsmla_pilot.py`
Protocol: (1) compute Grassmannian principal angles between W_Q^h and W_Q^{h'} for all 120 head pairs at one layer (fast, ~5 min); (2) if angles are ≤45° in majority, proceed with stacked SVD; (3) compute shared basis B, per-head extractors B^h; (4) measure ΔNLL.
Wall-clock: ~45 min total.

GO: Grassmannian principal angles ≤ 45° for ≥ 80 of 120 pairs AND ΔNLL ≤ +1.1 nats.
NO-GO: Head-pair principal angles are near 90° for majority → W_Q subspaces are geometrically independent; CF11's global result is the stack of orthogonal per-head subspaces; HSMLA fails. Structural finding: head-level W_Q diversity is real; the shared basis cannot be extracted without quality loss.
GRAY: Angles suggest partial sharing → test smaller K_shared (64) or 2-level hierarchy (4 groups of 4 heads sharing basis within group).

**Verdict: REFINE** (with TransMLA risk flag). TransMLA is ADJACENT (different target: KV vs W_Q, fine-tuning required) but represents serious prior art on the post-training MLA conversion space. HSMLA's W_Q-specific factorization from CF11 head-redundancy is differentiated. Stage 4 should verify the Grassmannian angle result before Stage 5 selection.

---

## 14. A2-APPENDKV — Append-Only KV Accumulator with Idempotent Retrieval

**Stage 2 verdict: ADVANCE (10), Track A**

### CF9: RoPE key rotation check

Stage 2 correctly identified RoPE as the primary CF9 risk. RoPE rotates key vectors position-by-position: k_t(p) = R(p) k_t^{base}. The LSH index built on raw (RoPE-applied) key vectors from different positions is geometrically inconsistent — semantically similar tokens at different positions have different key vectors.

**Mitigation (stated in proposal): index on RoPE-stripped keys.** This requires extracting W_K·x before RoPE application — feasible if the inference code exposes pre-RoPE keys. ik_llama.cpp uses RoPE application after the K projection; pre-RoPE keys are intermediate tensors accessible with minor code modification.

**CF9 status: PARTIAL fail, mitigated.** The raw append-only index doesn't work; the RoPE-stripped version does but requires inference code modification. The mechanism is not truly zero-code-change.

### Prior-art status

HASHEVICT (arXiv:2412.16187, Dec 2024) uses LSH over key embeddings for KV eviction. KVSwap (arXiv:2511.11907, Nov 2025) addresses NVMe-resident KV for long-context. LSH Tells You What To Discard (ICLR 2025) uses LSH for KV compression.

**Key differentiation:** All published methods permit eviction/mutation. APPENDKV's constraint-forced innovation is the append-only binding — converting eviction policy space into retrieval policy space. This framing is novel. But the retrieval mechanism (LSH over keys) is adjacent to HASHEVICT.

**Two-paper composition flag:** KVSwap (NVMe KV) + HASHEVICT (LSH key retrieval). Composition: "LSH-indexed KV on NVMe." APPENDKV's value-add: the append-only constraint eliminates the eviction-policy design space and maps the problem cleanly to content-addressable retrieval — a structural simplification the cousin pair doesn't claim.

### Residency arithmetic

At 32K context, Qwen3-1.7B: KV at INT4 append-only log = 32K × 28 layers × 2 × 128 × 0.5B = 1.4 GB NVMe. Hash index (binary projections of 128-dim INT8 vectors): 32K × 64 bits = 256 KB DRAM. Feasible.
At 4K context (primary target): KV = 175 MB NVMe — not the binding constraint (weights are 3 GB). The mechanism's value is primarily at 32K+ context.

### Smallest test (refined)

Script: `scripts/stage3_appendkv_pilot.py`
Protocol: implement append-only INT4 KV log + binary LSH index (RoPE-stripped keys); measure attention output cosine similarity vs exact bf16 attention on 50 prompts, 4K context.
Wall-clock: ~3 hours.

GO: cosine similarity ≥ 0.92 at 4K context.
NO-GO: similarity ≤ 0.80 → RoPE-stripped LSH index doesn't capture semantic similarity well enough for attention approximation; the approximation error in V-weighted attention output is too high.
GRAY: 0.80–0.92 → test with finer LSH (more hash bits) or hybrid: exact attention on top-K retrieved + approximate on the rest.

**Verdict: REFINE.** Novel framing (append-only constraint → retrieval architecture). CF9 partially mitigated (RoPE-stripped keys). Adjacent to HASHEVICT/KVSwap but differentiated by the constraint-forced structural simplification.

---

## 15. A3-FSMDEC — Finite-State-Machine Decoder with N ≤ 256 States

**Stage 2 verdict: ADVANCE (11), Track A**

### Prior-art status

Published FSM + LLM work (LMSYS blog, FARS OpenReview) all concern CONSTRAINED OUTPUT GENERATION (FSM over the vocabulary) — not FSM over the internal residual-stream state as a routing key. A3-FSMDEC's FSM is over the RESIDUAL STREAM CLUSTERS, not over the output vocabulary. These are orthogonal mechanisms.

**NOVEL:** Discrete FSM routing keyed on residual-stream cluster membership as a precomputed dispatch table is not in the published literature as of 2026-05-09.

**CF9:** k-means centroid assignment. Preconditions: Euclidean distance metric, continuous input space. Both hold for the residual stream. **CF9 passes.**

### Smallest test (refined)

Script: `scripts/stage3_fsmdec_pilot.py`
Protocol: 256-centroid k-means on layer-14 residual vectors from 1000 calibration tokens; per-centroid firing-pattern recall@30% (top-30% neurons by post-SwiGLU magnitude).
Wall-clock: ~1 hour.

GO: mean per-centroid firing recall ≥ 0.80.
NO-GO: recall ≤ 0.60 → residual-stream clusters don't predict firing patterns; the FSM state is uninformative for computation routing.

**Verdict: REFINE.** Novel routing architecture. Clean experiment. 1 hour.

---

## 16. U1-B-ETWO — ETW File-I/O Kernel Trace as Weight-Access Hotness Oracle

**Stage 2 verdict: ADVANCE (11)**

### Prior-art status

eInfer (ACM 2025) uses eBPF for distributed LLM tracing — server-side, not client-side weight access tracking. No paper found using ETW for LLM weight access hotness as of 2026-05-09. The eInfer approach is ADJACENT but targets different observables (distributed system tracing vs per-file byte-offset access histogram).

### CF9/CF10

No imported theorem. Pure systems substrate. ETW API preconditions: process must own the trace session (no admin rights needed for file-IO provider). If mmap path doesn't emit ReadFile events, page-fault provider fallback is specified. **CF9/CF10 not applicable.**

### Primary risk (revised)

The primary risk is that llama.cpp uses mmap (not ReadFile), so file-IO ETW events may not fire. The page-fault provider fallback is specified. Additional risk: page-fault granularity is 4 KB, not tensor-granular — post-processing must map faulted addresses back to GGUF tensor offsets via the tensor metadata table. This mapping step is documented in llama.cpp's gguf library.

### Smallest test

Script: `scripts/stage3_etwo_pilot.py` (Python + pywin32 or subprocess logman)
Wall-clock: ~3 hours.

GO: top-10 tensors by read count cover ≥ 30% of total reads AND cross-run Jaccard ≥ 0.80.
NO-GO: flat read histogram → no exploitable hot/cold structure at file-read level.

**Verdict: REFINE.** Novel substrate. 3-hour test. No prior art on ETW-based LLM weight access hotness.

---

## 17. U5-A-VGTP — Windows VEH Guard-Page Demand-Tier Promotion [FREE SWING]

**Stage 2 verdict: ADVANCE-wildcard (11)**

### CF9/CF10

No imported theorem. Windows VEH API: documented, available since Windows XP. Page-fault on NVMe-backed page may trigger full NVMe I/O before VEH fires — this is the primary risk and must be verified in the smallest test. **CF9/CF10 passes (structural floor met).**

### Scale limitation

As noted in Stage 2: at 70B, 10% promotion = 220M blocks × 10 µs VEH = 2200s for first-token. Mechanism is only viable at sub-7B scale. This limits A1 but doesn't kill the mechanism.

### Smallest test

Script: C harness (`scripts/stage3_vgtp_test.c`)
Protocol: VirtualProtect(PAGE_GUARD) on 1000 mmap'd pages; measure VEH round-trip time per page.
Wall-clock: ~1 hour.

GO: 1000-page VEH chain ≤ 200 ms total.
NO-GO: VEH fires after NVMe I/O completion → >10 ms/page → closes the VEH-based per-block interposition class for NVMe-backed pages.

**Verdict: REFINE — wildcard, CF-tether-suspended, structural floor met.** Sub-7B scope acknowledged; wildcard status is appropriate.

---

## 18. R9-R5 / LAYERONE-FOLD — Layer-1 Gate Constant Fold [FREE SWING]

**Stage 2 verdict: ADVANCE-wildcard (10)**

CF6 is the anchor. The fold mechanism is AERO-style, confirmed applicable to CF6-identified neurons. The base finding (Layer-1 36%) is confirmed; the context-conditional extension is speculative ([FREE SWING]).

**Relationship to C14-CFSHIFT:** CFSHIFT and LAYERONE-FOLD both start from CF6's c_1 constant. They are complementary — LAYERONE-FOLD uses c_1 to absorb the gate (fold into W_up), CFSHIFT uses c_1 to derive a constant Layer-2 logit bias. They can run in sequence or together.

### Smallest test (refined)

LAYERONE-FOLD's test is simpler than F3-L1GNF (which tests the null-space criterion). R9-R5's smallest test is:
(a) Apply fold: set W_up^{L1}_{scaled,i} = c_i × W_up^{L1}_{i,:} for foldable i; remove W_gate^{L1}[foldable,:] from the graph.
(b) Measure ΔNLL on 455-token held-out.
(c) Extended: measure per-layer near-constant fraction for all 28 layers.
Wall-clock: ~1.5 hours.

GO: ΔNLL ≤ 0.05 nats on held-out AND ≥ 3 layers with ≥ 10% near-constant fraction.
NO-GO: ΔNLL > 0.05 → c_i is not stable enough across the full token distribution (calibration-distribution artifact).

**Verdict: REFINE — wildcard, CF-tether-suspended, structural floor met.** Low absolute residency saving (9 MB at 1.7B) but zero quality cost and important as a cascade template demonstration.

---

## 19. C17-LAYERTAX — v2-CF1 Jaccard × CF6 Joint Layer Characterization [FREE SWING]

**Stage 2 verdict: ADVANCE-convergence (9)**

Prior-art: no paper combines outlier-channel stability (Jaccard) and gate-variance as a joint per-layer compression score. ADJACENT to HAQ/OmniQuant (which use Hessian-based sensitivity, not structural proxies).

CF9/CF10: no imported theorem; no calibration fit. Spearman ρ on 5 data points.

### Smallest test

Need G_ℓ at layers L7, L14, L21, L27 (L1 already known from CF6). J_ℓ from v2-CF1 (available for sampled layers — need to verify which layers were sampled in rraok_result.md).
Wall-clock: ~35 min new measurements + Spearman ρ.

GO: ρ > 0.20.
NO-GO: anti-correlated or uncorrelated → the two proxies measure different structural dimensions; no unified LCS.

**Verdict: REFINE — wildcard, CF-tether-suspended, structural floor met.** 35-min measurement, no prior art on the specific joint score.

---

## 20. F6-TROPICALDECODE — Tropical Semiring Attention Argmax [FREE SWING]

**Stage 2 verdict: ADVANCE-convergence (11)**

**CF9:** no fragile theorem; tropical max-plus is exact for argmax. **Passes.**

Prior-art search found no paper applying tropical/max-plus GEMV to replace softmax for attention computation (all tropical NN work targets training, not post-training inference approximation).

**Key dependence on attention concentration:** if layer-1 attention is diffuse (mean max-fraction < 0.50), F6-TROPICALDECODE's approximation error is unacceptably large and the idea dies at the smallest test. The concentration measurement is the cheapest falsifier (~20 min at layer 1 only).

### Smallest test

Script: `scripts/stage3_tropical_attn_pilot.py`
Protocol: 200-token forward pass; record softmax attention weights at layer 1 for all heads; compute max(softmax_row) per (token, head); histogram.
Wall-clock: ~20 min.

GO: ≥ 60% of (token, head) pairs at layer 1 have concentration ≥ 0.80.
NO-GO: attention is diffuse → tropical approximation error is too high; mechanism dies.

**Verdict: REFINE — wildcard, CF-tether-suspended, structural floor met.** 20-min test. If attention is concentrated (empirically testable), this is a genuine compute graph innovation.

---

## Kill List Additions

The following should be appended to `KILL_LIST.md`:

**v2-S3-R009-001 / F2-CKDJ (frame-novelty) — Softmax shift-invariance strip for static outlier channels** (CF9 violation). REGENERATE 2026-05-09 by run_009 Stage 3. Load-bearing mechanism (lossless strip via softmax shift-invariance) fails under CF9: the static-channel query offset δq_c contributes δq_c^T k_t which is position-dependent — not a constant scalar per attention-row — therefore cannot be stripped losslessly. What survives: "static outlier channels are high-energy W_Q input columns" as a measurement claim (adjacent to SageAttention2/FireQ); useful for W_Q quantization precision allocation but not a novel computation-graph rewrite. Pass to Stage 4 as a precision-allocation measurement.

---

## Convergence Cluster Summary

**C1 (attention-weight spectrum):** F1-MQKP (product SVD, Path 4 algebraic-identity), R9-R1 (W_V/W_O audit), F4-WQKRS (water-filling), C13-EQSUB (embedding alignment), R9-R3 (HSMLA). All REFINE. Cheapest settler: F1-MQKP 15-min pilot (product spectrum). If M's r_99/d < 0.55, F1-MQKP dominates the entire cluster as the compression-optimal treatment of attention weights.

**C2 (layer heterogeneity):** F5-RSUC (update covariance depth profile), C17-LAYERTAX (joint LCS). Both REFINE. Cheapest settler: RSUC 25-min r_{90%} depth profile. C17-LAYERTAX needs RSUC's depth-stratification result to be meaningful.

**C3 (tropical semiring):** F6-TROPICALDECODE (attention argmax) and A1-TROPFFN (FFN dispatch). Both REFINE (wildcards). F6's 20-min concentration test is the cheapest settler — if layer-1 attention is diffuse, F6 dies but A1 proceeds independently.

**C4 (NVMe compute bus):** R9-R4 (interleaved layout). REFINE. CF13* re-derivation (10 min) is the first rung.

---

## Final Verdict Table

| ID | Verdict | Reason |
|---|---|---|
| F1-MQKP | **REFINE** | Algebraic-identity elegance class; NOVEL product-SVD claim; top Stage 5 candidate |
| R9-R1/ATQKV | **REFINE** | W_V/W_O spectrum is load-bearing unmeasured; closes or opens full attention cascade |
| F4-WQKRS | **REFINE** | NOVEL water-filling allocation; 30-min test; confirmed CF11 asymmetry motivates it |
| F2-CKDJ | **REGENERATE** | CF9 FAIL: shift-invariance is position-dependent (δq_c^T k_t varies per key position) |
| F3-L1GNF | **REFINE** | NOVEL null-space criterion; 25-min test; no prior art |
| F5-RSUC | **REFINE** | C2 cluster rep; 25-min test; important empirical depth characterization |
| F6-TROPICALDECODE | **REFINE** | Wildcard; CF-tether suspended; 20-min cheapest falsifier |
| C13-EQSUB | **REFINE** | NOVEL μ_E coupling claim; 35-min; clean falsifiable |
| C14-CFSHIFT | **REFINE** | NOVEL constant logit bias from CF6; CF9 passes (including RoPE); 30-min test |
| C15-SNRATE | **REFINE** | NOVEL joint SNR oracle; CF10 compliant; critical rank-vs-quant risk tested directly |
| R9-R2/RAOK-REACH | **REFINE** | Strongest activation-quantization path; CF3-grounded tier boundary novel |
| R9-R3/HSMLA | **REFINE** | ADJACENT to TransMLA (arXiv:2502.07864, NeurIPS 2025); W_Q-specific factorization differentiated; run Grassmannian test first |
| R9-R4/NVME-STREAM | **REFINE** | 1.44× throughput gain robust to NVMe bandwidth; CF13* re-derivation first rung |
| R9-R5/LAYERONE-FOLD | **REFINE** | Wildcard; CF6-anchored; zero quality cost on calibration distribution |
| A1-TROPFFN | **REFINE** | Wildcard; C3 cluster rep; compute-bound speedup in L3-hot regime |
| A2-APPENDKV | **REFINE** | CF9 partially mitigated (RoPE-stripped keys); novel append-only framing |
| A3-FSMDEC | **REFINE** | NOVEL residual-stream FSM routing; 1-hour test |
| U1-B-ETWO | **REFINE** | Novel ETW oracle; 3-hour test; no prior art |
| U5-A-VGTP | **REFINE** | Wildcard; structural floor met; 1-hour test |
| C17-LAYERTAX | **REFINE** | Wildcard; 35-min test; joint LCS novel |

---

## Top Stage 5 Candidates (in priority order)

1. **F1-MQKP** — Highest Stage 2 score (13), algebraic-identity elegance class, 15-min cheapest falsifier, no prior art on product-SVD of attention score operator.
2. **R9-R1/ATQKV** — Load-bearing unmeasured W_V/W_O; closes or opens the full four-matrix attention compression cascade; 40-min test; highest A1 in Track B.
3. **C14-CFSHIFT** — Frame-novel constant logit bias from CF6; CF9/CF10 pass; 30-min test; no published cousin.
4. **C15-SNRATE** — Joint SNR precision budget oracle; novel coupling of CF5/CF11; directly testable rank-vs-quant correlation; 40-min test.
5. **F3-L1GNF** — Null-space criterion explanation for CF6 layer-1 anomaly; novel zero-calibration-cost fold predictor; 25-min test.

---

*Sources used:*
- [TransMLA: Multi-Head Latent Attention Is All You Need](https://arxiv.org/abs/2502.07864)
- [QSVD: Query-Key-Value Joint SVD Compression](https://arxiv.org/abs/2510.16292)
- [CARE: Covariance-Aware and Rank-Enhanced Decomposition](https://arxiv.org/html/2603.17946)
- [Layer-wise Dynamic Rank for Compressing LLMs](https://arxiv.org/pdf/2509.25622v1)
- [SageAttention2: Outlier Smoothing + INT4](https://arxiv.org/abs/2411.10958v4)
- [FireQ: RoPE-aware Quantization](https://arxiv.org/html/2505.20839v2)
- [Variance Is Not Importance: Structural Analysis of Transformer Compressibility](https://arxiv.org/html/2604.20682)
- [HASHEVICT: Pre-attention KV Cache Eviction via LSH](https://arxiv.org/pdf/2412.16187)
- [KVSwap: Disk-aware KV Cache Offloading](https://arxiv.org/html/2511.11907v1)
- [eInfer: Fine-Grained Tracing for Distributed LLM Inference](https://dl.acm.org/doi/pdf/10.1145/3748355.3748372)
- [AERO: Activation Removal for LLM Efficiency](https://arxiv.org/abs/2410.13060) (referenced from KILL_LIST context)
- [PrefixQuant: Outlier Decomposition for Activation Quantization](https://arxiv.org/abs/2410.05265) (referenced from KILL_LIST context)
