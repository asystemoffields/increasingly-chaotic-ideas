# Stage 1 — Orientation F (First-Principles) — Run 4

**Orientation:** F — First-Principles
**Track classification note:** CF8 applies to MLP weights only. Attention W_Q (r_99/d ≈ 0.63) and W_K (r_99/d ≈ 0.79) are the live algebraic-identity territory. MLP weights are full-rank; proposals targeting MLP must come through activation/routing structure (AERO-style fold), not rank reduction. Cross-layer W_Q basis sharing is KILLED (v2-CHEAP-TEST-001). Per-layer K=128 is the W_Q ceiling.

---

## F-WVOS — W_V / W_O Spectral Structure + Attention-Side Algebraic Fold

**F, Track A/B (load-bearing novelty: Track A fold; compression is consequence)**

### Mechanism

Track A. The AQFKV experiment (CF11) measured W_Q and W_K spectra but left W_V and W_O unmeasured. CF11 established a sharp spectral boundary: MLP weights (r_99/d ≈ 1.0) vs attention weights (W_Q r_99/d ≈ 0.63, W_K r_99/d ≈ 0.79). If W_V has similarly concentrated spectrum (r_99/d ≪ 1), a post-training algebraic fold becomes available: absorb the low-rank W_V truncation into W_O without retraining. Specifically, let W_V ≈ U_V Σ_V V_V^T at rank r. The attention output for head h is: head_h(Q, K, V) = softmax(Q_h K_h^T / √d_k) · V_h · W_O_h. Substituting V_h = U_V Σ_V V_V^T: the value projection becomes (X W_V_h) = X (U_V Σ_V V_V^T), and the output projection is then W_O_h (X W_V_h) = W_O_h U_V Σ_V V_V^T X^T ... rewritten as (W_O_h U_V Σ_V)(V_V^T x). The fold is: define W_V_folded = V_V^T (d_k × d_model; rank r), W_O_folded = W_O_h U_V Σ_V (d_model × r). This replaces one d_model×d_k + d_k×d_model pair with one r×d_model + d_model×r pair. No retraining; the equivalence holds exactly at rank r. Not CF8 MLP rank reduction; exploits CF11 attention-spectrum concentration, which is the regime CF8 excludes. Does not require cross-layer sharing (KILLED by v2-CHEAP-TEST-001).

### Residency arithmetic

W_V and W_O per layer in Qwen3-1.7B: each is (2048 × 2048) at bf16 = 8 MB per matrix, 16 MB per layer, 28 layers = 448 MB. At rank r=128 (matching CF11's W_Q GO threshold): W_V_folded (128×2048) + W_O_folded (2048×128) = 2×128×2048 × 2 bytes = 1 MB per layer vs 16 MB → 15× per-layer reduction on V+O alone. Total attention weight saving: 448 MB × (14/16) ≈ 420 MB. On a 1.7B model at bf16 (3.4 GB total), that's ~12% reduction — modest but mechanically clean. At 70B scale (Qwen3-72B / Llama-3-70B): W_V+W_O ≈ 4 GB of the ~140 GB bf16 total; at r=512 (analog of CF11's W_K boundary) that's ~75% of W_V+W_O saved ≈ 3 GB, not a residency breakthrough alone but an additive contribution to the attention-weight compression line (stacks with W_Q K=128 saving from CF11). Combined W_Q+W_V+W_O at appropriate ranks: ~5-6 GB saving on 70B, enough to matter when NVMe-resident. ΔNLL cost: unknown until measured; the W_V fold is algebraically exact at rank r, so the quality cost is precisely the W_V truncation error at that rank, predicted by the spectrum.

### Novelty gloss vs kill list and published landscape

Closest kill-list entry: CF11 per-head W_Q NO-GO (K=64 at +1.53 nats) — but this is W_V/W_O, not W_Q, and the fold mechanism is algebraic not per-head-only. The proposal does NOT require per-head decomposition; the fold is global across heads. Closest published method: MLA (DeepSeek-V2) absorbs W_V into W_O at training time. This is a post-training, no-retraining analog — the structural question of whether Qwen3's W_V admits this fold is empirically open. GQA/MQA also reduce V-head count but don't fold; they reduce the number of V matrices, not their rank. The novelty is: measuring whether W_V (untested object) has the same concentrated-spectrum property that CF11 found in W_Q/W_K, and if so whether the algebraic fold is lossless at r=128 or r=256.

### Smallest experiment

**Claim:** Qwen3-1.7B W_V matrices have r_99/d ≤ 0.80 and rank-256 SVD fold gives ΔNLL ≤ +0.5 nats.

Compute: load W_V[0..27] (28 matrices, 2048×2048 bf16 each), compute SVD for each, record r_99, var@K for K ∈ {64, 128, 256, 512}. Then construct W_V_folded = V_V^T and W_O_folded = W_O · U_V · Σ_V for K=256, measure NLL on 512 WikiText-2 tokens. Runtime: SVD on 28 matrices (2048²) ≈ 15 min; NLL eval ≈ 10 min. Total ≤ 30 min.

**Go threshold:** r_99/d ≤ 0.80 AND ΔNLL ≤ +0.5 nats at r=256. **No-go structural finding:** extends CF8/CF11 boundary classification to W_V/W_O — if flat spectrum, closes the attention-weight compression class.

### Primary risk

W_V may be full-rank like MLP weights (CF8), in which case the fold adds no residency gain and the O absorption is pointless. Mitigation: the SVD spectrum measurement (30 min, above) settles this before any engineering investment.

---

## F-GFQO — Gauge-Fixing the Residual Stream via Q-K Subspace Alignment

**F, Track A**

### Mechanism

Track A. The residual stream x ∈ R^{d_model} is invariant under a class of rotations: if we simultaneously transform x → Rx and transform every weight matrix that projects from or into the residual by R^{-1} or R respectively, the model's output is unchanged. This is a gauge freedom: the trained model's computation is defined up to a residual-stream rotation R ∈ O(d_model). CF11 measured that W_Q has r_99/d ≈ 0.63, meaning its row space occupies ~1293/2048 of the residual stream. Gauge-fixing argument: choose R so that the active W_Q subspace aligns with a standard coordinate block (first 1293 dimensions). Under this R, 755 residual-stream dimensions are orthogonal to ALL W_Q projections — those dimensions contribute nothing to attention query computation. If W_V's subspace (unmeasured) is also concentrated, gauge-fixing to a joint W_Q + W_V subspace may expose a block of residual dimensions that attention ignores entirely. Those dimensions could be allocated to a narrower "attention-invisible" computation (e.g., MLP routing). The algebraic identity: for any orthogonal R, f_attention(Rx; R·W_Q, R·W_K, R·W_V, R^{-1}·W_O) = f_attention(x; W_Q, W_K, W_V, W_O). The question is whether the calibration-optimal R (the rotation that concentrates joint attention subspace into minimum coordinates) produces a block-sparse structure tight enough to reduce computation or storage. This does NOT require cross-layer sharing (killed); it operates per-layer on the residual-stream coordinates.

### Residency arithmetic

This is a structural-finding idea rather than a direct residency win. The payoff is: if gauge-fixed coordinates expose M ≪ d_model "attention-invisible" residual dimensions per layer, those can be computed at lower precision or skipped for attention-side operations. Upper bound: CF11 says W_Q spans ~1293 dims; if W_V also spans ~1293 dims and they overlap substantially (a 45-degree angle between subspaces leaves ~half overlap), gauge-fixed attention uses only ~1600 of 2048 residual dimensions → 22% reduction in attention-side GEMV cost. At 70B with attention weights ~30% of total computation: ~7% overall compute reduction. Not a residency breakthrough; a compute and potentially precision-assignment insight. Residency upside: lower-precision storage for attention-invisible block.

### Novelty gloss vs kill list and published landscape

Closest kill-list entry: cross-layer W_Q subspace sharing (KILLED by v2-CHEAP-TEST-001) — this is different: per-layer gauge fixing, not cross-layer stacking. Closest published: QuIP / QuIP# (rotation-based quantization) uses random orthogonal R to spread weight energy uniformly before quantizing. This is the opposite operation — we want R that concentrates attention energy, not spreads it. SpAtten (attention sparsity) and H2O prune by attention-score magnitude, not by residual-coordinate geometry. The gauge-freedom argument for transformers appears in theoretical work (e.g., Baydin et al. on neural network symmetry) but has not, to my knowledge, been applied to expose coordinate-level block-sparsity for inference cost reduction. The structural question — does the joint W_Q + W_V + W_K subspace concentrate into < d_model dimensions tightly enough to matter — is empirically open on Qwen3.

### Smallest experiment

**Claim:** In Qwen3-1.7B, the joint rowspan(W_Q) ∪ rowspan(W_V) ∪ rowspan(W_K) per layer has effective dimension ≤ 1600 (78% of d_model), measured by stacking W_Q/W_V/W_K matrices and computing singular value decay.

Compute: stack [W_Q; W_K; W_V] ∈ R^{6144×2048} per layer (3 attention projections × 2048), run SVD, record var@K for K ∈ {1024, 1280, 1536, 1600, 1800, 2048}. 28 layers, each a 6144×2048 SVD. Runtime: ~25 min.

**Go threshold:** var@1600 ≥ 0.99 for all layers. **No-go finding:** joint subspace fills d_model — gauge-freedom argument gives no block-sparse structure, attention cannot be reduced in input dimensionality.

### Primary risk

CF9 precondition failure: the gauge-fixing argument assumes exact orthogonal symmetry, but RMSNorm (applied to residual stream before each block) breaks this symmetry — RMSNorm is not invariant under arbitrary rotations (only under norm-preserving ones where the scale factors still match). Mitigation: verify that the rotation R preserves the RMSNorm folding structure, or restrict R to the class that commutes with the per-channel scale (i.e., R must be an orthonormal matrix that leaves the channel-wise norm distribution unchanged — this is more restrictive than O(d)). Check: compute the norm-distribution mismatch ‖σ(Rx) - σ(x)‖ for 100 calibration tokens; if < 1% distortion the symmetry is approximately usable.

---

## F-AERO-Q3 — AERO-Style W_gate Elimination: SwiGLU Activation-Removal Feasibility Probe for Qwen3

**F, Track A**

### Mechanism

Track A. AERO (arXiv:2410.13060) removes the activation function from FFN layers, causing W_gate and W_up to algebraically merge into a single matrix: without silu(·), the computation becomes (W_up · x) ⊙ (W_gate · x) → (W_up ⊙ W_gate) · x (a Hadamard product of rows, not the same as a single linear op). Wait — correction: with activation *removal* (silu → identity), the gate output silu(W_gate·x) becomes W_gate·x, so the full SwiGLU output is (W_gate·x) ⊙ (W_up·x). This is NOT a single linear map unless we allow a different parameterization. AERO's actual trick: when f(·)=identity, SwiGLU(W_gate,W_up,x) = (W_gate·x)⊙(W_up·x) = diag(W_gate·x)·W_up·x. A valid fold: pre-multiply W_up by W_gate^T element-wise (this only works exactly if W_gate = W_up, which won't hold). The correct statement from AERO: in a *trained* model with f→identity, one can write the output as a single d_model→d_ffn→d_model operator (size d_model×d_ffn×d_model) that is separable as (W_gate ∘ W_up)·W_down where ∘ is Hadamard-outer. The key question for Qwen3 (raised explicitly in KILL_LIST.md R3): does activation removal without retraining catastrophically hurt Qwen3? This is EMPIRICALLY UNTESTED on Qwen3 post-training. If silu(z) ≈ z on the Qwen3 calibration distribution for most neurons — i.e., the gate pre-activations are predominantly positive and large — then the approximation error is small and the fold is low-loss. CF6 (SDZC) found only 1.5% of neurons are near-constant globally, but that was asking whether the *output* is near-constant, not whether the *gate pre-activation* is predominantly in the linear region of silu. These are distinct questions.

Define: linear-region fraction = fraction of (token, neuron) pairs where |silu(z) - z| / |z| < 0.05. If this is ≥ 80% globally (a plausible claim for large positive z values that dominate in trained LLMs), the fold is approximately valid with bounded error.

### Residency arithmetic

If W_gate is eliminatable without retraining: W_gate is (d_ffn × d_model) = (6144 × 2048) bf16 = 24 MB per layer. 28 layers = 672 MB of W_gate eliminated from Qwen3-1.7B (out of 3.4 GB total = 20% of model). At 70B scale, W_gate ≈ 7 GB of ~140 GB (5%). On the 7.28 GiB Ryzen target with Qwen3-1.7B at bf16 → this fold alone could push 1.7B from ~3.4 GB to ~2.7 GB. Main payoff: enables 70B on NVMe at lower effective bpw by folding one of the three MLP weight matrices.

### Novelty gloss vs kill list and published landscape

KILL_LIST.md R3 explicitly opened "AERO-style activation removal for SwiGLU specifically" as untested. No kill entry for Qwen3 AERO. Published: AERO (arXiv:2410.13060) is the parent; it was applied on ReLU/GELU models and its Qwen3/SwiGLU applicability is explicitly flagged as open. CSMF (killed R3) died on a different error: polynomial substitution of silu doesn't eliminate W_gate (the linear part remains). AERO is different: it replaces silu with identity entirely, then the computation changes (linear gate instead of nonlinear gate). The structural question is entirely about the quality cost of this specific approximation on Qwen3's trained weights.

### Smallest experiment

**Claim:** On Qwen3-1.7B, replacing silu(W_gate·x) with W_gate·x globally (silu→identity) gives ΔNLL ≤ +0.5 nats, confirming that gate pre-activations are predominantly in the linear region.

Implement: intercept the SwiGLU forward pass, replace silu(·) with identity for all layers simultaneously. Measure NLL on 512-token WikiText-2 slice. Runtime: ~10 min (no weight modification — pure forward-pass substitution).

**Go threshold:** ΔNLL ≤ +0.5 nats. **No-go finding:** silu is load-bearing in Qwen3; the gate's nonlinearity is used, not a near-linear side channel. Confirms AERO-style fold requires retraining for SwiGLU Qwen3.

### Primary risk

The kill from CSMF (R3) established that polynomial approximation of silu fails for W_gate elimination; this idea is NOT that — it tests whether identity replacement of silu (i.e., exact linear gate) is close enough that quality survives. Risk: the 1.5% near-constant neurons (CF6) suggest the gate IS actively used for routing, in which case ΔNLL will be large. Mitigation: if global replacement fails, test layer-by-layer replacement to identify which layers tolerate it (CF6's layer-1 anomaly at 36% near-constant may tolerate activation removal locally even if global is impossible).

---

## F-SPECA — Spectral Equivalence of Attention W_K via W_Q Alignment: Shared-Basis Compression Without Cross-Layer Sharing

**F, Track B**

### Mechanism

Track B. CF11 established: W_Q r_99/d ≈ 0.63 (1293/2048), W_K r_99/d ≈ 0.79 (1618/2048). Per-layer, NOT cross-layer (v2-CHEAP-TEST-001 killed cross-layer sharing). The question: within a single layer, do the W_Q and W_K row-spaces substantially overlap? If rowspan(W_Q^L) and rowspan(W_K^L) share a common subspace S^L with dim(S^L) ≥ r_min, then a per-layer joint SVD of [W_Q^L; W_K^L] ∈ R^{4096×2048} (stacked) captures both projections in a shared basis without using any cross-layer information. Algebraic identity: W_Q^L ≈ A_Q^L B^L and W_K^L ≈ A_K^L B^L where B^L ∈ R^{r×2048} is the shared per-layer basis, A_Q^L ∈ R^{d_head_total × r}, A_K^L ∈ R^{d_head_total × r}. Storage: original W_Q + W_K = 2 × 2048² × 2 bytes = 16 MB/layer. Shared basis + small factors at r=256: B^L (256×2048) + A_Q (2048×256) + A_K (2048×256) = 256×2048 + 2×2048×256 = 3 × 256 × 2048 × 2 bytes = 3 MB/layer. 16 MB → 3 MB = 5.3× reduction on W_Q + W_K. This is per-layer joint W_Q/W_K decomposition, NOT cross-layer (KILLED). Precondition: the within-layer W_Q/W_K subspace overlap must be high enough that a shared basis is a good approximation of both.

### Residency arithmetic

Qwen3-1.7B: W_Q + W_K per layer = 16 MB. 28 layers = 448 MB. At r=256 with shared basis: ~3 MB/layer × 28 = 84 MB. Saving: 364 MB from attention Q/K projections (a 10% reduction on total 3.4 GB model). At 70B scale: W_Q + W_K ≈ 14 GB of ~140 GB. At r=512 (matching CF11's W_K GO boundary): ~18% of W_Q+W_K saved ≈ 2.5 GB, pushing toward DRAM-resident-at-lower-bpw feasibility when combined with W_V fold (F-WVOS). ΔNLL: predicted by the combined approximation error; if the within-layer Q/K subspace overlap is ≥ 0.85 (var@256 ≥ 0.85 for stacked [W_Q;W_K]), the quality cost is bounded by the residual of the truncation. If overlap is poor, shared basis degrades independently of the rank.

### Novelty gloss vs kill list and published landscape

Closest kill: v2-CHEAP-TEST-001 (cross-layer W_Q subspace sharing, KILLED). This is within-layer only — the basis B^L is per-layer, never shared across layers. Closest published: GQA shares K/V heads across Q groups (reduces head count), not the same as within-layer joint SVD of W_Q and W_K. MLA (DeepSeek-V2) does cross-head joint K/Q low-rank projection at training time; this is post-training per-layer joint decomposition. The specifically novel claim: within a single Qwen3 layer, W_Q and W_K both project from the same residual stream into the same attention head dimension; their row-spaces should share structure forced by the same residual-stream input distribution. Whether this holds in trained Qwen3 is empirically open.

### Smallest experiment

**Claim:** In Qwen3-1.7B, stacked [W_Q^L; W_K^L] has var@256 ≥ 0.85 in more than 20 of 28 layers, and rank-256 shared-basis approximation of W_Q+W_K gives ΔNLL ≤ +0.5 nats.

Compute: for each of 28 layers, stack [W_Q; W_K] (4096×2048), SVD, record var@K for K ∈ {128, 256, 512}. Then construct the A_Q/A_K/B factorization at r=256 for all layers, run forward pass, measure NLL. Runtime: 28 SVDs on 4096×2048 matrices ≈ 30 min; NLL eval ≈ 10 min. Total ≤ 45 min.

**Go threshold:** mean var@256 ≥ 0.85 across layers AND ΔNLL ≤ +0.5 nats. **No-go finding:** W_Q and W_K occupy independent row-spaces within each layer — per-layer joint compression is not possible; attention-side compression is limited to separate per-matrix rank reduction at the CF11 boundaries.

### Primary risk

The within-layer Q/K subspace overlap may be near-zero (W_Q and W_K could be learning complementary projections, not aligned ones). CF11 measured them independently and found W_Q more concentrated than W_K; this doesn't tell us about their mutual alignment. If overlap < 0.5, shared basis performs worse than separate truncation. Mitigation: the measurement (30 min) runs before any engineering; the var@256 diagnostic gives a clean yes/no before building the factorization.

---

## F-WNORM — W_down Spectral Probe: Closing the MLP Weight Class Characterization

**F, Track B**

### Mechanism

Track B. CF8 established that W_gate and W_up are full-rank in Qwen3-1.7B (r_99/d ≈ 1.0 for both). W_down (d_model × d_ffn = 2048 × 6144) has NOT been measured. KILL_LIST.md R4 explicitly notes: "W_down full-rank finding applies to W_gate (R3) and W_up (R4) — almost certainly applies to W_down too (untested but predictable)." This is a first-principles probe to close the characterization. The mechanism is simple: SVD of W_down across all 28 layers, plot singular value decay, measure var@K for K ∈ {512, 1024, 2048}. If full-rank confirmed: CF8 now covers all three MLP matrices with data. If W_down has concentrated spectrum (less likely but possible — W_down is the output projection and sees different gradient signal from W_gate/W_up during training): opens a compression path that differs from the already-dead W_gate/W_up rank reduction. The first-principles motivation: W_down maps from activation space (post-SwiGLU, dimension d_ffn=6144) to residual stream (dimension d_model=2048). Because d_ffn > d_model and W_down is not square, the maximum rank is d_model=2048 — meaning even a "full-rank" W_down has r_99 ≤ 2048, i.e., r_99/d_out ≤ 1.0 but r_99/d_in ≤ 0.33. The relevant question is: what is var@K in the output dimension? At r=256 (vs d_out=2048, 12.5%), is the spectrum concentrated enough to truncate cleanly? W_down is structurally different from W_gate/W_up because it projects a SPARSER (post-SwiGLU) input into a dense residual stream — the firing pattern sparsity (CF1/CF3) may induce a concentrated column structure in W_down.

### Residency arithmetic

W_down per layer: 2048 × 6144 × 2 bytes bf16 = 24 MB. 28 layers = 672 MB. If r=256 (12.5% of 2048) gives ΔNLL ≤ +0.5 nats: storage → 256 × (2048 + 6144) × 2 bytes = ~4.2 MB/layer, 28 layers = ~117 MB. Saving: 555 MB on Qwen3-1.7B (16% of total model). At 70B: W_down ≈ 7 GB of 140 GB; at r=512 (25% of d_out): ~75% of W_down saved ≈ 5 GB. This would be significant — but only if the spectrum is concentrated, which CF8's pattern strongly argues against.

### Novelty gloss vs kill list and published landscape

Closest kill: CF8 (MLP weights full-rank) — this is explicitly testing whether CF8 extends to W_down, the untested third MLP matrix. The CF8 class kill covers W_gate and W_up explicitly; W_down is noted as "predictable" but NOT measured. No kill entry covers W_down. Published: GPTQ / AWQ / SqueezeLLM target W_down with weight quantization (not rank reduction); none specifically reports W_down singular value decay for Qwen3. The structural novelty is the d_ffn > d_model shape: W_down's max rank is d_model, not d_ffn, so the spectrum question is: does the actual r_99 fall significantly below d_model due to the SwiGLU activation sparsity feeding into it?

### Smallest experiment

**Claim:** W_down in Qwen3-1.7B has r_99/d_out > 0.90 (full-rank, following CF8 pattern), measured by SVD of W_down across all 28 layers.

Compute: SVD of 28 matrices each 2048×6144 (or equivalently 6144×2048^T). Record r_99, var@K for K ∈ {64, 128, 256, 512, 1024, 2048}. Runtime: ~20 min (larger matrices than W_gate/W_up but same operation).

**Go threshold:** var@256 ≥ 0.99 in ≥ 20/28 layers = FULL-RANK (confirms CF8). If var@256 ≤ 0.85 in ≥ 15/28 layers = CONCENTRATED (opens compression path not blocked by CF8 kills). **No-go finding:** CF8 is now fully characterized across all MLP matrices — closes the MLP compression class for Qwen3 1.7B, redirects attention to attention weights and quantization.

### Primary risk

Result is almost certainly a CF8 extension (full-rank), meaning the experiment confirms a kill rather than opening a path. The payoff is the structural completeness of the CF8 characterization, not a new compression win. Mitigation: if W_down is full-rank, the experiment took 20 min and definitively closes a question that multiple future ideators would re-open. Worth running for map value alone.

---

## F-RMNQ — RMSNorm Scale Absorption: Post-Training Gauge-Fix That Eliminates Scale Parameters

**F, Track A/B — algebraic identity**

### Mechanism

Track A (computation graph changes: RMSNorm scale γ eliminated by absorption). RMSNorm before attention in Qwen3: y = γ ⊙ (x / ‖x‖) where γ ∈ R^d is a per-channel learned scale. The subsequent linear projection W_Q, W_K, W_V projects y: W · y = W · diag(γ) · (x/‖x‖) = (W · diag(γ)) · (x/‖x‖). The algebraic identity: W_absorbed = W · diag(γ). This is exact — no approximation. The RMSNorm scale is absorbed into the projection matrix, and the scale parameters γ can be set to all-ones (identity, effectively removed from the computation). This reduces the computation graph: instead of (1) compute RMSNorm, (2) scale by γ, (3) project by W, the new graph is (1) compute RMSNorm (norm only, no scale multiply), (2) project by W_absorbed. This saves the per-channel scale multiply (d-dimensional element-wise multiply) — a small compute saving. The storage saving: γ ∈ R^d (one per RMSNorm layer) is small (2048 × 2 bytes per norm layer × 56 norm layers ≈ 0.23 MB). Compute saving: eliminates 56 element-wise multiplies of size 2048 per token per forward pass. The real value: if the absorbed W_absorbed = W · diag(γ) has different spectral properties than W (because γ reweights rows), the SVD of W_absorbed may differ from SVD of W alone — this is the potential second-order effect. If γ is highly non-uniform, absorbing it into W may improve or worsen the low-rank structure of the resulting matrix. First-principles question: does RMSNorm scale absorption change the effective rank of W_Q enough to affect CF11's K=128 threshold? Does W_Q_absorbed = W_Q · diag(γ_attn) have the same r_99/d as W_Q alone?

### Residency arithmetic

Direct saving: 0.23 MB — negligible. Indirect saving: if the absorbed W_Q_absorbed has better (more concentrated) spectrum than W_Q, then the K=128 GO threshold from CF11 improves, meaning a lower K becomes viable. At K=64 (per-head NO-GO in CF11 at +1.53 nats): if absorption reweights spectrum to shift the GO boundary down, K=64 global could become viable. W_Q at K=64 global: d_model × 64 × 2 bytes × 28 layers = 7 MB (vs 224 MB at full W_Q storage). This is the scenario where the indirect effect dominates.

### Novelty gloss vs kill list and published landscape

Closest published: QuIP / QuIP# and SpinQuant perform orthogonal transformations on residual stream before quantization to improve quantization properties. This is a simpler, per-channel scale absorption that is algebraically exact (not an approximation). SmoothQuant absorbs activation outlier scales into weights for quantization; this is structurally similar but applied to RMSNorm scales (not activation outliers) for the purpose of graph simplification + potential spectral improvement. No kill-list entry covers this. It's a clean first-principles move: the scale is absorbed, the computation graph is simplified, and the question is whether the spectral properties of the absorbed matrix differ enough to improve CF11 compression.

### Smallest experiment

**Claim:** W_Q_absorbed = W_Q · diag(γ_pre_attn) has var@128 ≥ var@128(W_Q_unabsorbed) (spectrum is at least as concentrated, enabling same or better CF11 compression threshold).

Compute: load W_Q[0..27] and γ[0..27] for Qwen3-1.7B, compute W_Q_absorbed = W_Q · diag(γ), SVD both, compare var@K for K ∈ {64, 128, 256}. Runtime: ~10 min.

**Go threshold:** var@128(W_Q_absorbed) ≥ var@128(W_Q) + 0.02 in ≥ 20/28 layers AND K=64 absorbed gives ΔNLL ≤ +0.5 nats. **No-go finding:** RMSNorm absorption does not improve spectral concentration; CF11's K=128 floor is robust to this gauge-fixing operation.

### Primary risk

The RMSNorm scale γ may not be significantly non-uniform in Qwen3 (if γ ≈ 1 everywhere, absorption changes nothing). Check: std(γ) across channels. If std(γ)/mean(γ) < 0.01, skip — the absorption is numerically trivial. Mitigation: log std(γ) first (< 1 min), abort if trivial.

---

## F-TIED — Tied Embedding Gauge Rotation: Exploiting the Dual-Path Gradient for Factored Storage [FREE SWING]

**F, Track B — [FREE SWING]**

### Mechanism

Track B. CF12 established: the tied embed/lm_head in Qwen3-1.7B-Base is full-rank (r_99=1992/2048) with catastrophic ΔNLL at r=1024 (+19.96 nats). The standard SVD truncation is dead. However, CF12 also characterizes WHY the matrix is full-rank: gradient flows through TWO paths (input embedding lookup + output projection), keeping every direction load-bearing. First-principles observation: while NEITHER path alone can be truncated (both need the full matrix), the gradient paths constrain different DIRECTIONS to be large. Path 1 (embedding lookup) demands that token-discriminative directions are non-zero. Path 2 (lm_head output) demands that vocabulary-prediction directions are non-zero. These two demands are imposed on the SAME matrix by the tied configuration. This is a tension: the matrix is full-rank partly because it must simultaneously serve two functional roles that might prefer different coordinate systems.

Gauge-rotation question: is there an orthogonal R such that R^T · W_E · R (where W_E is the 151936×2048 tied matrix) separates into a block structure that serves path 1 better in one block and path 2 better in another? If such R exists (calibration-derivable: minimize off-block-diagonal energy under some functional partitioning), storage would remain the same size but the block structure could enable lower-precision storage for the off-diagonal cross-blocks. This does NOT reduce r_99 (the matrix stays full-rank under rotation), but may improve quantization properties (aligned-axis quantization hits codebook boundaries better when the active subspace is coordinate-aligned). The "free swing" is that this hasn't been tried and may not work — but the structural argument from CF12's dual-path analysis makes it a principled question rather than random search.

### Residency arithmetic

Storage of W_E does not change under rotation. Payoff is indirect: if block-diagonal structure improves quantization from 4-bit to 2-bit on the off-diagonal blocks (20-40% of the matrix), residency reduces by ~0.3-0.6 bpw × 300 MB = 90-180 MB on Qwen3-1.7B. Moderate upside. At 70B scale: tied embedding weight is ~3 GB (150K × 8K × 2 bytes); similar arithmetic gives 400-900 MB saving if the block structure enables lower-bit storage for cross-blocks.

### Novelty gloss vs kill list and published landscape

Closest kill: CF12 / LHQD (tied lm_head SVD truncation) — KILLED. This is NOT truncation — it is block-diagonal rotation for quantization alignment. Different mechanism: the matrix remains full-rank, the rotation changes the coordinate frame for quantization. Closest published: QuIP# (Hadamard incoherence for quantization) applies a random rotation before quantization to reduce max-magnitude outliers. This is a calibration-derived structured rotation targeting dual-path functional separation, not a random incoherence rotation. No published work specifically targets tied-embedding block-diagonal rotation derived from the dual-path gradient structure.

### Smallest experiment

**Claim:** A calibration-derived block-diagonal rotation of W_E (derived by k-means clustering of token embedding rows into k=16 semantic clusters) reduces quantization error (INT4 MSE) by ≥ 10% on the off-cluster-diagonal blocks vs unrotated storage.

Compute: k-means (k=16) on W_E rows (151936 × 2048), compute block permutation, measure INT4 MSE on permuted vs unpermuted matrix. Runtime: k-means on 151K points × 2048 dims ≈ 30 min with sklearn mini-batch.

**Go threshold:** INT4 MSE reduction ≥ 10% on off-diagonal blocks. **No-go finding:** tied embedding rows have no cluster structure separating the two gradient paths; the dual-path tension is resolved by the full-rank structure, not by an exploitable partition.

### Primary risk

k-means on 151K × 2048 is clustering the token vocabulary, which has well-known structure (semantic clusters exist in embedding space) — so clusters will form. The question is whether the clusters correspond to the "path 1 vs path 2" functional split, which is unlikely to be the dominant factor in semantic embedding geometry. The free-swing risk is that the rotation finds semantic structure but not the dual-path structure. Mitigation: also test a functional-gradient-derived rotation (permute by frequency bin, since BOS/high-freq tokens are the path-2 outliers per CF12's PrefixQuant reference) and compare.

---

## Convergence handles

1. **W_V / W_O spectral concentration** — untested attention weight class; multiple ideas (F-WVOS, F-GFQO) converge on needing this measurement; cf. CF11's W_Q/W_K pattern
2. **Within-layer joint W_Q/W_K subspace overlap** — per-layer (not cross-layer) alignment claim; convergence point for F-SPECA and F-GFQO
3. **AERO-style silu→identity substitution quality cost on Qwen3** — opened explicitly in KILL_LIST.md R3; F-AERO-Q3 is the settler
4. **RMSNorm scale non-uniformity** — appears in F-RMNQ; cheap precondition check that multiple future ideas may depend on
5. **W_down spectral class completion** — F-WNORM closes the CF8 boundary; any future MLP-side idea referencing "the third MLP matrix" needs this number
6. **Gauge-rotation compatibility with RMSNorm (does orthogonal R commute with per-channel scale?)** — F-GFQO and F-RMNQ both depend on the RMSNorm symmetry structure; this is a shared precondition
