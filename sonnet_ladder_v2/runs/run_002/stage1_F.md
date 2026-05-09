# Stage 1 — Orientation F (First-Principles) — Run 002

Orientation: F — First-Principles
Run: 002 (independent cold-start, no inheritance from run_001)

---

## F1 — Joint W_V / W_O Spectral Collapse JVOC
**F / Track B**

### Mechanism
Track B — compression. CF11 showed W_Q has r_99/d ≈ 0.63 and W_K ≈ 0.79; MLP weights have r_99/d ≈ 1.0. The entire MLP class is dead for rank reduction (CF7, CF8). The attention weight class is demonstrably compressible at global scope. W_V and W_O are the last two unmeasured attention matrices — their spectra have been explicitly flagged as "cheap follow-up" in CF11's post-experiment notes but never measured. The first-principles move: IF W_V (shape n_heads × d_v × d_model) and W_O (shape n_heads × d_model × d_v) share a joint low-rank structure — plausible because W_V · W_O is an n_heads × d_model × d_model composition and the residual output that matters is the sum across heads — THEN the composition can be written as W_V W_O ≈ U Σ V^T with U, V shared across heads, collapsing storage. The algebraic identity: ∑_h W_O^h W_V^h = U (∑_h Σ_h) V^T is exact when all heads share singular vectors. Even a partial alignment (a few top singular directions aligned) produces a non-trivial residency reduction. No SGD; no retraining. AERO-archetype: find the mathematical identity the trained weights already satisfy, then exploit it.

### Residency arithmetic
Qwen3-1.7B bf16: attention weights (W_Q, W_K, W_V, W_O) per layer each have shape (n_heads × d_head × d_model) = (16 × 128 × 2048) = 4,194,304 params × 28 layers × 4 matrices × 2 bytes = ~1.88 GiB total attention weight storage. CF11 showed global W_Q at K=512 gives ΔNLL=+0.20 nats (4× residency reduction on W_Q). If W_V and W_O compress to similar K, that is another 2 matrices × 28 layers × (d × d × 2 bytes) at 4× reduction = ~0.47 GiB saved. Combined with W_Q+W_K (CF11's tested regime at K=512), total attention residency could drop from 1.88 GiB to ~0.60 GiB. On Qwen3-72B-scale extrapolation: attention weights are roughly proportional. At ~11 GB DRAM for 72B-4bit-weights, attention is ~1.5 GB; attention compression at 4× saves ~1.1 GB and directly enables higher DRAM-resident fraction. Tok/s uplift: each GB saved at 11.5 GB/s DRAM bandwidth frees ~87 ms/token → ~+0.3 tok/s at baseline of 1.5 tok/s. Modest but cumulative.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: per-head W_Q rank truncation (CF11 NO-GO). This is NOT per-head — it is global, head-shared, and on W_V/W_O matrices not yet measured. Closest published method: MLA (DeepSeek-V2) compresses KV by joint low-rank projection. JVOC differs: it applies post-training, uses no retraining, and targets W_V/W_O as a fused operator rather than a KV-cache side channel. A3 (arXiv:2505.12942) targets attention logit error; JVOC targets weight Frobenius error on the V·O composition, a structurally distinct object. The composition W_O^h W_V^h has never been measured in isolation for Qwen3 — this is genuinely unmeasured territory on this stack.

### Smallest experiment
**Claim**: The joint operator ∑_h W_O^h W_V^h (shape 2048×2048 per layer) has r_99/d ≤ 0.70 in at least 20 of 28 layers of Qwen3-1.7B-Base.
**Protocol**: load Qwen3-1.7B bf16, extract W_V and W_O tensors per layer, compute M_L = ∑_h W_O_h W_V_h, run SVD on each M_L (shape 2048×2048), measure r_99/d and var@K=512. Runtime: ~25 min on Ryzen.
**Go threshold**: r_99/d ≤ 0.75 in ≥20 layers AND ΔNLL at K=512 global reconstruction ≤ +0.40 nats.
**NO-GO finding**: if r_99/d ≈ 1.0, extends CF8 to W_V/W_O and closes the last known compressible direction in individual weight matrices. That is a high-value structural finding even on failure.

### Primary risk
W_V and W_O may have r_99/d ≈ 0.9 (slightly more concentrated than MLP but far below W_Q's 0.63), making the compression marginal vs the ΔNLL penalty. Mitigation: test the fused M_L rather than per-matrix — the fusion removes the head-decomposition identity and may reveal more concentration than per-matrix spectra.

---

## F2 — Cross-Layer W_Q Basis Sharing CQBS
**F / Track B**

### Mechanism
Track B — compression. CF11 established that W_Q globally across 16 heads spans only a K=128-dimensional subspace (r_99/d ≈ 0.63). The natural first-principles question is: do all 28 layers share that 128-dim subspace, or is each layer's subspace distinct? If the 28 layer-level W_Q subspaces are nearly aligned, a single cross-layer basis U ∈ R^{2048×128} describes all of them; each layer's W_Q stores only a 128×128 coefficient matrix instead of a 2048×128 weight matrix — a 16× storage reduction on W_Q across all layers. The algebraic identity: if W_Q^{(L)} ≈ U C_L for all L, then at inference W_Q^{(L)} x = U (C_L x) — compute C_L x first (cheap: 128-dim), apply U once globally (amortized across layers if sequential). No retraining; the identity holds by construction if the alignment angle is small. CF11 EXPLICITLY flagged "cross-layer Q basis" as open: "do all 28 W_Q share a common subspace? PCA-stack experiment." This is that experiment with a computable payoff.

### Residency arithmetic
W_Q per layer: 2048×2048 = 4M params × 28 layers × 2 bytes = 448 MB. At 16× reduction (store 128×128 per layer + one 2048×128 shared basis): 28 × 128×128 × 2 + 2048×128 × 2 = 917 KB + 512 KB = 1.4 MB. Savings: 448 MB → 1.4 MB, essentially free W_Q storage. On Qwen3-72B: W_Q is ~7× larger → ~3.1 GB → 10 MB. At 11.5 GB/s DRAM, 3.1 GB saves ~270 ms/token → meaningful at 1-3 tok/s regime. The aligned-subspace assumption is binary: either the alignment is there or it isn't, and the smallest experiment settles it in 20 min.

### Novelty gloss vs the kill list and the published landscape
Closest kill: "per-head W_Q rank truncation" (CF11 NO-GO). CQBS is cross-layer and cross-head sharing; per-layer per-head reduction is the dead variant. Closest published method: cross-layer weight sharing in ALBERT uses tied parameters by construction (training-time decision); CQBS is post-training discovery of accidentally shared structure. Head-sharing across layers (not heads within a layer) has no published measurement on Qwen3 or any SwiGLU-gated GQA family. The proposal lives in unmeasured territory opened by CF11.

### Smallest experiment
**Claim**: When all 28 layers' W_Q matrices (shape 2048×2048 each) are stacked and PCA'd, the top-128 principal components capture ≥ 85% of total variance.
**Protocol**: stack all W_Q matrices row-wise (shape 28×2048² or equivalently reshape to 28×2048×2048, flatten to 57344×2048), run SVD on the 57344×2048 matrix, measure cumulative variance at K=128 and K=256. Runtime: ~20 min on Ryzen.
**Go threshold**: cumvar(K=128) ≥ 0.85.
**NO-GO finding**: if cumvar(K=128) ≤ 0.70, layers have independent subspaces — valuable negative: each layer's W_Q contributes distinct information, which refines CF11's head-redundancy finding to be within-layer only.

### Primary risk
The 28 subspaces may be modestly aligned (cumvar 75-85%) but the reconstruction error may accumulate across layers non-linearly, making ΔNLL worse than expected from the per-layer residuals. Mitigation: measure both alignment AND per-token NLL on a held-out set after coefficient reconstruction, as two separate go/no-go gates.

---

## F3 — RMSNorm Gauge Freedom Exploitation RMGF
**F / Track A**

### Mechanism
Track A — arch-transposition. RMSNorm at layer L computes `h_L_normed = h_L / ‖h_L‖ × g_L`, where g_L is a learned diagonal gain vector (shape d_model). The subsequent linear projection W × h_L_normed includes an implicit left-multiplication by diag(g_L). The first-principles observation: g_L and the first weight matrix of every downstream block (W_Q, W_K, W_V, W_gate, W_up) form a commuting pair — `W × diag(g_L) × (h_L / ‖h_L‖)` can be rewritten as `(W × diag(g_L)) × (h_L / ‖h_L‖)`, absorbing g_L into W permanently and setting g_L to 1. This is a gauge-fixing identity: the norm-output space has a residual scale freedom that SGD picks up as a learned g_L, but the freedom can be gauged away by folding g_L into the downstream weights, producing a structurally equivalent computation graph with one fewer element-wise multiply per layer. The saving is small per layer (one d_model multiply avoided) but the more interesting downstream question is whether folding g_L into W changes the effective spectrum of W in a way that makes W MORE compressible — i.e., does the trained g_L contain scale information that, when folded in, concentrates the spectrum of W? This has never been measured. The algebraic identity `W_gauged = W diag(g_L)` is exact; no calibration needed. The hypothesis is that Frobenius-energy concentration of W_gauged is HIGHER than W alone — a spectrum-reshaping move without any retraining.

### Residency arithmetic
Direct saving from eliminating g_L: 28 layers × 1 RMSNorm per residual block × d_model × 2 bytes = 28 × 2048 × 2 = 115 KB. Negligible. The payoff is NOT in g_L removal but in downstream compressibility: IF W_Q_gauged has r_99/d ≤ 0.50 (vs 0.63 ungauged, CF11), then K=128 global SVD quality improves, and the compression chain from F1/F2 works at lower K. This is a no-cost residency-shaping move that makes other proposals more effective. On their own terms: measuring the spectrum shift is a 15-min computation; it either opens the door wider or closes it.

### Novelty gloss vs the kill list and the published landscape
Closest kill: CRANK (γ-inert RMSNorm, Track A R2) — KILLED as pre-empted by SmoothQuant. RMGF is structurally different: SmoothQuant migrates scale from activations to weights for quantization purposes; RMGF folds the norm gain g into downstream weight spectra to test whether the spectrum becomes more concentrated. The move is not for activation quantization — it is a spectrum-shaping pre-processing step for rank-reduction experiments. SmoothQuant and CRANK have no bearing on the spectral-concentration question. No published measurement of "does RMSNorm gain absorption change the effective spectrum of Q/K attention projections" exists in the literature.

### Smallest experiment
**Claim**: For Qwen3-1.7B-Base, W_Q_gauged = W_Q × diag(g_RMSNorm) has r_99/d ≤ 0.55 (vs CF11's 0.63 ungauged) in at least half the layers.
**Protocol**: load bf16 model, extract W_Q and g per layer, compute W_Q_gauged = W_Q × diag(g), run SVD, measure r_99/d and cumvar@128 vs the CF11 baseline. Runtime: ~20 min.
**Go threshold**: median r_99/d ≤ 0.55 across 28 layers.
**NO-GO finding**: spectrum is not reshaped by gauge absorption — g_L is scale-neutral for the attention weight spectrum. This constrains the family of spectrum-shaping moves available post-training.

### Primary risk
The gain vector g_L may be predominantly near-constant (all values ≈ 1) after training, making the fold a no-op spectrally. Mitigation: check g_L statistics (mean, variance, max/min) before the full SVD run; if g_L variance is < 0.01, stop immediately (5-min pre-check).

---

## F4 — Softmax Shift Invariance Folding SSIF
**F / Track A**

### Mechanism
Track A — arch-transposition. Softmax(QK^T / √d) is invariant to constant shifts in the row of QK^T: softmax(x + c·1) = softmax(x) for any scalar c. This shift-invariance is an algebraic identity — it is exactly true, not approximate. The consequence for inference: any per-query additive constant in the attention logit matrix can be stripped before softmax without changing attention output. The question is whether trained Qwen3 W_Q and W_K have persistent per-channel offsets that, when removed, reduce the effective numerical range and make Q and K representable at lower precision. Specifically, if Q = W_Q x has a channel mean μ_Q (across the calibration distribution) and K has a corresponding μ_K, the per-query logit offset is μ_Q^T μ_K (scalar per head) — exactly absorbed by softmax invariance. Removing this offset from Q before the inner product is algebraically lossless. More interesting: if there is a cross-head persistent structure in the attention pattern (a few "sink tokens" that receive near-uniform attention mass regardless of query), that structure is a near-constant additive in QK^T — it can be represented by a bias correction term (a few hundred bytes) rather than full-precision Q and K, sharpening the information in Q and K per step. This connects to PrefixQuant's token-wise outlier analysis (R2 stage-6 notes) via a different route: instead of handling outlier channels in activation quantization, handle the predictable additive component in the softmax domain.

### Residency arithmetic
Direct: removing per-head constant from QK^T costs no storage but allows Q,K to be represented at lower effective dynamic range → tighter quantization. If Q activations compress to INT8 at +0.05 nats with offset removal vs INT8 at +0.20 nats without, activation quantization of Q/K becomes viable. For Qwen3-1.7B at 4K context, KV cache is small; but for Qwen3-72B at 32K context, KV cache is the bottleneck. The mechanism's value is in the quantization-quality tradeoff on the KV path, not in weight residency. Estimated improvement: if per-head Q offset μ_Q has norm ~10% of Q's total norm (to be measured), the dynamic range after removal is 10% tighter → ~0.15 bit savings per element → at FP16 → INT8 KV compression, this is the difference between +0.10 nats and +0.05 nats.

### Novelty gloss vs the kill list and the published landscape
Closest kill: KV Temporal Differencing (R1/S2) — KILLED because RoPE rotations make deltas non-small. SSIF is different: it uses softmax shift-invariance in the logit domain (not the key-value domain) and does not touch the KV values at all. Closest published method: ASCQ (Attention-Sink Channel Quantization, Track A R3 deferred) addresses sink tokens in KV; SSIF addresses the additive constant in the pre-softmax logit, a different layer. The softmax shift-invariance identity is elementary and well-known in attention literature, but its use as a KV-precision-sharpening tool in post-training inference (without retraining) is not published in any paper we have encountered. The hook is the algebraic identity being load-bearing, not the quantization.

### Smallest experiment
**Claim**: For Qwen3-1.7B, per-head per-query softmax logit offset (mean over calibration tokens) has norm ≥ 5% of the softmax input's per-row norm in at least 15 of 28 layers.
**Protocol**: run forward passes on 200 calibration tokens, record attention logit matrices (QK^T / √d), compute per-head mean logit vector per layer, compute ratio ‖mean‖ / ‖row_std‖. Runtime: ~30 min.
**Go threshold**: median ‖mean‖/‖row_std‖ ≥ 0.05 across 28 layers AND at least 5 layers with ratio ≥ 0.10.
**NO-GO finding**: logit offsets are negligible — softmax shift invariance has no practical activation-range payoff on Qwen3. This rules out the entire family of "exploit softmax symmetry for quantization-range reduction" ideas.

### Primary risk
RoPE rotations change Q and K per position, meaning the "per-query mean" is not a single offset but position-dependent — the invariance argument may not reduce to a simple constant offset per head. Mitigation: measure the variance of the per-position mean (not just global mean); if per-position mean has low variance across tokens at fixed position, the offset is real; if per-position mean varies as much as the signal, the mechanism fails.

---

## F5 — W_down Spectral Audit for Cross-Layer Basis WDCA
**F / Track B**

### Mechanism
Track B — compression. CF7/CF8 killed low-rank compression of W_gate and W_up. CF5 showed W_up is MORE rank-sensitive than W_gate. Both are full-rank. But W_down has never been measured. W_down (shape d_model × d_intermediate = 2048 × 6144) is the projection that maps the post-SwiGLU hidden vector back to the residual stream. Its role is fundamentally different from W_gate and W_up: it does not participate in the gating nonlinearity (it operates on the OUTPUT of the SwiGLU, not on x). The SwiGLU output is sparse-ish in a soft sense (CF1 — W_up dominates firing), and W_down accumulates contributions from a relatively concentrated set of neurons per forward pass. The hypothesis: the ROW-span of W_down (dimension d_model=2048) may be more concentrated than its full-rank column count (6144) suggests — specifically, the image of W_down under typical activation patterns may occupy a low-dimensional subspace of the residual stream, even if the weight matrix itself is full-rank. This is NOT a rank reduction of W_down itself (which CF8's spirit would kill). It is a measurement of the EFFECTIVE output subspace under calibration activations. The tool is not SVD of W_down alone but SVD of the activation-weighted output: X_out = W_down × diag(silu(W_gate·x) ⊙ (W_up·x)) over the calibration set, which measures the actual residual-stream contribution. If this output lives in a K-dim subspace, we can project W_down to that subspace without changing calibration behavior.

### Residency arithmetic
W_down storage: 28 layers × 2048 × 6144 × 2 bytes = 1.34 GiB. At 4× output-subspace compression (K=512 of 2048): store U ∈ R^{2048×512} + W_down_compressed ∈ R^{512×6144} per layer: 28 × (2048×512 + 512×6144) × 2 = 28 × (2M + 3.1M) × 2 = ~280 MB. Savings: 1.34 GiB → 0.27 GiB = 1.07 GiB saved. On 7.28 GiB target: meaningful. ΔNLL depends entirely on whether the output subspace is actually concentrated; minimum threshold is ≤ +0.20 nats at K=512 to be worth the engineering complexity.

### Novelty gloss vs the kill list and the published landscape
Closest kill: low-rank MLP weight compression of any form (CF7/CF8). WDCA explicitly DOES NOT propose W_down weight-rank reduction; it proposes output-subspace measurement under actual calibration activations — a different object. The weight's singular values may be flat while the activation-weighted output is concentrated. Closest published method: activation-weighted projection follows the logic of GPTQ/GPTVQ (Hessian = X^T X as second moment of activations); WDCA targets the output rather than the input Hessian, which selects COLUMNS of W_down via output importance rather than rows via input importance. This inversion has not been measured for W_down on Qwen3. CF10 warning: the calibration-fitted step (projecting W_down to output subspace) must satisfy n_samples >> n_params. Here n_params = 2048×512 = 1M; n_samples = 1000 tokens × 2048 dims = 2M. Borderline — use ridge ≥ 0.1 and measure cal-vs-eval R² explicitly.

### Smallest experiment
**Claim**: The matrix X_out_L = W_down^L × A_L (where A_L is the post-SwiGLU activations on 200 calibration tokens) has 90% of its variance concentrated in ≤ 512 output directions in at least 15 of 28 layers.
**Protocol**: run calibration forward pass (200 tokens), record post-SwiGLU activations A_L per layer, compute X_out_L = W_down^L × A_L (shape 2048×n_tokens), SVD of X_out_L, measure cumvar(K=512). Runtime: ~40 min (includes forward pass + per-layer SVD).
**Go threshold**: cumvar(K=512) ≥ 0.90 in ≥ 15 layers AND ΔNLL at reconstructed K=512 W_down ≤ +0.25 nats.
**NO-GO finding**: output distribution of W_down is full-rank under actual activations — the SwiGLU diffuseness (CF3 K-dependent dynamicity) spreads output directions enough to fill the residual-stream space. This would imply that the "soft sparsity" of SwiGLU does not concentrate W_down's effective output, a useful structural fact.

### Primary risk
The calibration-fitted output projection is the WDLA failure mode (CF10) in disguise — even if the output subspace is concentrated, fitting W_down to that subspace on calibration data and evaluating on held-out may exhibit the same R²=-118000 catastrophe. Mitigation: enforce low-rank by SVD truncation (not by least-squares fit) — no fitting step at all; truncate and measure the lossless component, making this a purely algebraic compression. The CF10 risk only triggers if we fit parameters to calibration; pure SVD truncation bypasses it entirely.

---

## F6 — Residual-Stream Gauge Orbit Folding RSGO [FREE SWING]
**F / Track A**

### Mechanism
Track A — arch-transposition. The residual stream in a transformer has a gauge freedom: any invertible transform G applied to h_L can be absorbed by simultaneously applying G to all weight matrices that read from h_L and G^{-1} to all weight matrices that write to h_L, leaving the final output unchanged. In an L-layer transformer with N blocks, this is an O(d)^L symmetry group acting on the residual stream (a different G per layer). Trained SGD breaks this symmetry only partially — the loss function is input/output symmetric under the gauge group, so the trained weights settle into a random gauge. The first-principles question: is there a canonical gauge (a choice of G^* per layer) that simultaneously diagonalizes or near-diagonalizes the weight matrices that matter most — specifically W_Q W_K^T (the attention kernel per layer) AND reduces the condition number of W_V W_O (the V-O product)? If such G^* exists, it can be found by solving a joint diagonalization problem on calibration covariance matrices. The algebraic content: find G^* = argmin ∑_L ‖diag(G W_Q^L) - D_L‖_F subject to G G^T = I (orthogonal gauge fixing). This is a simultaneous block-diagonalization problem over the 28-layer collection. If the gauge G^* concentrates the W_Q spectrum (lowers r_99/d from 0.63 to below 0.50), it enables higher-K compression on all attention matrices simultaneously. No retraining; G^* is applied once as a basis change. The mechanism is orthogonal to every existing compression scheme because it works in the coordinates of the residual stream, not on individual weight matrices.

### Residency arithmetic
If gauge fixing reduces effective rank of all attention weight matrices by 20% (from r_99/d≈0.63 to ≈0.50), and the cross-layer sharing from F2 applies in the gauged coordinates, total attention weight residency drops: W_Q + W_K at 2× current compression (K=512 each, ΔNLL < 0.30 nats per CF11) → K=256 in gauged coordinates → 8× compression of each. 4 matrices × 28 layers × 2048×2048 × 2 bytes × (1/8) = 470 MB total attention → 59 MB. Storage of G^*: d × d × 2 bytes = 8 MB. Net: from 1.88 GiB to 0.06 GiB for attention. That is 1.82 GiB saved — meaningful on the 7.28 GiB target. These numbers are speculative (the gauge concentration claim is the falsifiable part). The go/no-go comes from the spectrum measurement.

### Novelty gloss vs the kill list and the published landscape
No item on the kill list addresses residual-stream gauge-orbit analysis. Closest published method: rotation-based quantization (QuIP, QuaRot, SpinQuant) uses random orthogonal rotations to smooth activation outliers. RSGO differs: it is looking for a structured non-random rotation that simultaneously diagonalizes W_Q W_K^T, motivated by representation theory (the gauge group is the "symmetry that the trained net didn't break"). QuIP/QuaRot don't target attention-weight spectral concentration; they target activation-quantization grid uniformity. The simultaneous-diagonalization setup (joint eigenvalue problem on the calibration covariance and the attention kernel) is a distinct mathematical object not in the published attention-compression literature.

### Smallest experiment
**Claim**: The residual-stream gauge orbit (orthogonal G applied simultaneously to residual stream and all reading/writing weight matrices) has a non-trivial concentration minimum — i.e., the optimal orthogonal G^* reduces median r_99/d of W_Q^gauged below 0.55 vs CF11's 0.63 baseline.
**Protocol**: (1) compute attention covariance C_L = E[h_L h_L^T] on 200 calibration tokens. (2) Find G^* via simultaneous diagonalization of {C_L}_L (truncated eigenbasis of ∑_L C_L as first approximation, ~10 min). (3) Compute W_Q^gauged_L = W_Q_L G^{*T} and measure r_99/d. Runtime: ~45 min total.
**Go threshold**: median r_99/d of W_Q^gauged ≤ 0.55 across 28 layers.
**NO-GO finding**: the trained gauge cannot be profitably fixed to improve spectral concentration — gauge-orbit analysis on trained transformer weights produces no useful compression primitive. This closes the gauge-freedom route, which has never been directly empirically tested on Qwen3-class weights.

### Primary risk
Simultaneous diagonalization of 28 non-commuting covariance matrices is ill-posed in general; the "best G^*" may leave substantial residual off-diagonal terms in every layer, and the resulting W_Q^gauged has no better spectrum than the original. Mitigation: use the simpler approximation G^* = top-d eigenvectors of ∑_L C_L (a sum of covariances is at least positive semi-definite, and its eigenbasis is well-defined) as the first pass; this is guaranteed to be a valid orthogonal transform and the runtime drops to a single PCA.

---

## Convergence handles

- W_V and W_O spectra (unmeasured; directly opens or closes F1)
- Cross-layer W_Q subspace alignment (CF11 opened; F2 is the measurement)
- RMSNorm gain g_L as a spectrum-shaping operator (F3; interacts with any attention-compression chain)
- Per-head softmax logit offset ‖mean‖/‖row_std‖ across calibration (F4; KV-quantization precondition)
- Activation-weighted W_down output-subspace rank (F5; different from weight rank)
- Orthogonal gauge orbit of the residual stream (F6; jointly relevant to any rotation-based compression)
