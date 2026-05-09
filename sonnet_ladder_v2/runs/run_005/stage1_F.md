# Stage 1 — Orientation F (First-Principles) — Run 005

**Kill acknowledgment**: v2-CHEAP-TEST-001 killed the entire cross-layer W_Q subspace sharing class. Per-layer K=128 (CF11) is the W_Q ceiling. No idea below touches cross-layer W_Q stacked SVD, shared basis, or joint diagonalization.

---

## F1-WQWK-JOINT — Joint W_Q / W_K Simultaneous Low-Rank (JQKLR)
**F / Track B**

### Mechanism
Track B compression. CF11 establishes W_Q r_99/d ≈ 0.63 and W_K r_99/d ≈ 0.79 for Qwen3-1.7B. These were measured independently. The first-principles question: do W_Q and W_K in the same layer share singular vectors? If U_Q (left singular vectors of W_Q, shape d×K_Q) and U_K (left singular vectors of W_K, shape d×K_K) have substantial subspace overlap — measured by the canonical correlation (principal angles between their column spans) — then a single joint projector P ∈ R^{d×K_joint} reconstructs both simultaneously, buying compression beyond what either alone achieves. Concretely: for each layer L, stack [W_Q^L; W_K^L] ∈ R^{2d×d}, compute SVD, find the top-K joint left singular vectors P_L ∈ R^{d×K_joint}. The compressed representations are Q̃ = P_L^T W_Q^L and K̃ = W_K^L P_L (rank K_joint each). If canonical correlation is high, K_joint < K_Q + K_K with similar reconstruction error. This is NOT the cross-layer kill: P_L is computed per-layer. It is a within-layer joint compression of two attention matrices, exploiting the geometric fact (if it holds) that the input directions W_Q and W_K attend to are correlated because they are trained on the same residual stream. Does not collide with CF8 (MLP matrices are untested here). Not CF9: precondition is just subspace overlap, measurable by canonical correlation angle θ_1 between top-K_Q left-singular vectors of W_Q and top-K_K of W_K.

### Residency arithmetic
Qwen3-1.7B: 28 layers, d=2048, num_heads=16, head_dim=128. W_Q and W_K each: 2048×2048 = 4M params × 2 bytes (bf16) = 8 MB per matrix, 16 MB per layer, 448 MB total for 28 layers.

CF11 baseline: W_Q at K=512 (ΔNLL=+0.20), W_K at K=512 (ΔNLL=+0.29). Current: store two rank-512 decompositions per layer. Factor count: W_Q → U_Q (2048×512) + V_Q (512×2048) = 2M + 1M = 3M; same for W_K. Total: 6M params × 28 layers = 168M × 2 bytes = 336 MB for the attention {Q,K} block. Original: 448 MB. Ratio so far: 1.33×.

Joint compression hypothesis: if canonical correlation between U_Q[:, :512] and U_K[:, :512] has top-r angles near zero for r=256, then P ∈ R^{2048×256} approximates both. Residuals handled separately at lower rank (~128 each). Total: P (2048×256) + W_Q·P thin (256×2048) + W_K·P thin (256×2048) + two residuals (2048×128 each) ≈ 0.5M + 0.5M + 0.5M + 0.25M × 2 = 2.0M per layer. Reduction: 4M → 2M (2× per layer) vs independent compression giving 3M. Gain over independent CF11: 1.5×. Combined with CF11 gain: 28 layers × 2.0M × 2 bytes = 112 MB for Q+K. On 70B-scale (same head structure scaled), proportional.

At DRAM 11.5 GB/s, 112 MB Q+K block loads in 9.7 ms/layer vs 16 MB/layer baseline = 0.7× of Q+K bandwidth. Marginal at 4K context (Q+K is small fraction of total 70B weight traffic) but non-trivial as part of a composite strategy.

### Novelty gloss
Closest kill list item: v2-CHEAP-TEST-001 (cross-layer W_Q sharing). This is within-layer joint Q+K compression — orthogonal to the cross-layer kill. Closest published method: MLA (DeepSeek-V2) does training-time joint Q-K projection. CF11 note mentions post-training MLA-style as motivated. This proposal operationalizes that motivation as a measurable first step: canonical correlation angle determines whether joint compression beats independent compression, without any retraining. Not on arXiv as a post-hoc measurement to establish the canonical angle first, then apply it.

### Smallest experiment
**Claim**: the top-256 left singular vectors of W_Q^L and top-256 of W_K^L at a representative layer (L=14) have at least 30 principal angles θ_i < 10°.

Measurement: load W_Q and W_K for layer 14 in bf16, compute top-256 SVD for each (scipy.sparse.linalg.svds or torch.linalg.svd truncated), compute canonical correlation via QR+SVD of (U_Q[:, :256])^T (U_K[:, :256]). Count angles below 10°.

Runtime: ~8 min on Ryzen 5 7530U (two 2048×2048 SVDs + small matrix multiply).

**Go threshold**: ≥30 angles < 10° → joint subspace is real; proceed to joint-compression reconstruction test.
**No-go**: U_Q and U_K are nearly orthogonal → independent CF11 compressions do not compose; joint compressor buys nothing. Structural finding: input-attending and key-attending subspaces are geometrically decoupled in this layer.

### Primary risk
W_Q and W_K left singular vectors are trained to span orthogonal subspaces (queries and keys index different representation directions). Mitigation: smallest experiment settles this in 8 min; if orthogonal, the idea dies cheaply without cascade investment.

---

## F2-WQWV-SYMM — Head-Permutation Gauge Fixing on W_Q / W_V (HPGF)
**F / Track A**

### Mechanism
Track A (computation-graph change). Qwen3-1.7B has num_heads=16, GQA ratio 1:1 (W_Q and W_V both 16 heads). The trained model's loss is invariant under simultaneous permutation of heads in W_Q, W_O (the output projection absorbs the permutation), because head identity is not fixed by any architectural constraint — the loss only cares about the sum over heads. This is a gauge freedom: any permutation σ ∈ S_16 acting simultaneously on W_Q (permute rows of each 128-d head block) and W_O (permute corresponding columns of the head-aggregation axis) leaves the function identical. Under this gauge freedom, choose σ to sort heads by their W_Q spectral energy (Frobenius norm of the head-block). Now the top heads are geometrically first; the bottom heads (low-energy) are last. CF11 showed that 16 heads collectively span a ~128-dim subspace (one head's worth). If after permutation the bottom M heads have nearly zero energy relative to the top, those heads can be dropped without retraining — not rank reduction within a head (per-head K=64 is NO-GO per CF11), but whole-head elimination after gauge-fixing to energy order. The algebraic identity: for any permutation σ, f_{attn}(x) = f_{attn,σ}(x) exactly. The empirical question: does the head energy spectrum (W_Q^L head-block Frobenius norms squared, sorted descending) have a cliff at some M < 16? If 4 heads carry 90% of the total head-energy, the remaining 12 can be zeroed at inference, saving 12/16 = 75% of Q+K+V overhead with zero algebraic error in the gauge.

### Residency arithmetic
If M=4 heads are kept (hypothesis), Q+K+V for those layers: 4/16 × 16 MB = 4 MB/layer vs 16 MB baseline. Savings: 12 MB/layer × 28 layers = 336 MB on Qwen3-1.7B. At 70B scale (32 layers, 64 heads, d=8192), proportional to ~2.1 GB saved on Q+K+V alone. Attention weights are ~15% of 70B total weight footprint (~5.5 GB at bf16, ~0.4 GB at IQ4_XS). Not the dominant residency lever, but provides a clean ΔNLL-measurable block that composes with MLP quantization strategies (PDAP, RAOK).

Quality cost: if head energy cliff is real, the dropped heads contribute near-zero to the output already — algebraic identity up to the threshold; soft quality cost estimated <0.1 nats for M≥4.

### Novelty gloss
This is not head pruning via Taylor expansion or gradient-based importance (standard literature). The gauge-fixing move — that permutation σ is algebraically exact before any quality evaluation begins — is what makes it first-principles rather than heuristic. Closest kill: no explicit head-permutation gauge-fix idea appears in the kill list. Closest published: "Head pruning" literature (Michel et al., 2019) uses gradient-based importance; this uses weight-space energy under the permutation gauge. The distinction: gauge-fixed energy is an exact statement about the trained weights' structure, not a proxy correlated with importance on a calibration set.

### Smallest experiment
**Claim**: in Qwen3-1.7B, at least one layer has head Frobenius-norm distribution where the bottom 4 heads contribute <5% of total Q+K head energy.

Measurement: load W_Q for all 28 layers; for each layer compute the 16 head-block Frobenius norms (each block is 128 rows of W_Q); sort descending; report cumulative energy at M=4, 8, 12.

Runtime: 5 min (pure tensor slicing and norm computation on the loaded model).

**Go threshold**: any layer with bottom-4 head energy <5% of total → head elimination is plausible; proceed to zero-head-block inference and measure ΔNLL.
**No-go**: all layers have near-uniform head energy → CF11's "16 heads collapse to 1 head worth of subspace" is a global spectral property not exploitable per-head in the energy gauge. Structural finding: head energy is equidistributed despite subspace concentration.

### Primary risk
Head energy may be equidistributed even if the joint subspace is concentrated (a known pathology of attention training: all heads are trained to carry similar energy). Mitigation: 5-min experiment settles it; equidistribution is itself a structural finding (CF11 addendum).

---

## F3-WDOWN-WUP-FACTOR — W_down / W_up Algebraic Fold via AERO Precondition Test (AEROT)
**F / Track A**

### Mechanism
Track A. AERO (arXiv:2410.13060) removes the SiLU activation entirely and folds W_gate into W_up algebraically (W_fold = W_up × diag(silu_linearized) × W_gate, approximately). CF6 (SDZC) showed that globally only 1.5% of neurons have near-constant gate output — AERO's "activation removal" requires the activation to be approximately linear or constant on the calibration distribution, not just that the gate is near-zero. The first-principles question this idea tests: on the calibration distribution, what fraction of neurons have gate output concentrated in a linear regime (i.e., silu(W_gate·x) ≈ α_i + β_i·(W_gate·x) for α_i, β_i derived per-neuron from calibration data), such that the fold W_fold_i = β_i · W_up[:, i] · W_gate[i, :] is accurate to within δ nats? CF6 measures gate variance; this idea measures gate linearity on the operational distribution. The mechanism: for neurons where the linearization error is <ε, replace silu(W_gate·x)·W_up[:, i] with (α_i·W_up[:, i] + β_i·W_fold_i·x), reducing two projections to one for those neurons. The fold is exact in the limit ε→0 (AERO limit). At finite ε, it is a calibration-derived closed-form substitution — in-orientation for F.

Key distinction from CF7 (no-retraining structural compression killed class): this is NOT rank reduction of W_gate or W_up; it is function-output substitution for a neuron subset where the substitution is provably accurate. The mechanism targets the activation function's regime, not the weight matrix's rank.

### Residency arithmetic
If 20% of neurons across all layers admit linearization with ΔNLL contribution < 0.01 nats (hypothesis), then 20% of W_gate rows can be folded into W_up: W_fold_i (row-vector, length d) replaces both W_gate[i, :] (length d) and the silu application. Storage: W_gate has 172K rows × 2048 cols (intermediate 6144 × d 2048) — 28 layers × 6144 × 2048 × 2 bytes ≈ 700 MB for W_gate. 20% reduction: 140 MB saved. Modest on its own. Composes with PDAP (quantized W_up codebook): the folded neurons reduce PDAP's input size by 20%, improving codebook fit quality on the remaining 80%.

### Novelty gloss
Closest kill: CF7/CF8 class kill is for W_gate rank reduction without retraining. This is per-neuron linearization testing — different object (the function silu not the matrix rank). Closest published: AERO itself (arXiv:2410.13060). Difference: AERO does global activation removal as a training step. This proposal tests whether a post-hoc, per-neuron, calibration-fitted linearization achieves the AERO fold for a subset of neurons without retraining. If the linearization error is acceptable for N% of neurons, those neurons are foldable post-hoc. That is a post-training subset-AERO, not in the AERO paper.

### Smallest experiment
**Claim**: for at least one layer, ≥10% of neurons have calibration-distribution silu gate linearization error (max |silu(W_gate·x_t) − α_i − β_i·(W_gate·x_t)| over 500 calibration tokens) below 0.05.

Measurement: run 500 calibration tokens forward; for each layer collect silu(W_gate·x) per neuron; fit (α_i, β_i) by 1D linear regression; compute max absolute residual per neuron; count neurons below 0.05 threshold.

Runtime: ~30 min (forward pass on 500 tokens, per-neuron regression is O(N_tokens × d_intermediate) = trivial).

**Go threshold**: ≥10% of neurons in any layer linearizable with max residual < 0.05 → proceed to fold those neurons and measure ΔNLL.
**No-go**: all neurons have highly nonlinear gate output → silu is uniformly nonlinear on the calibration distribution; post-hoc AERO fold is infeasible. Structural finding: CF6's 1.5% dead-zone boundary is also a linearizability boundary (nonlinear everywhere except the dead-zone).

### Primary risk
SwiGLU neurons may have highly variable gate output precisely because they are trained to be nonlinear (the gate is the expressivity). Max-residual threshold may exclude all but the near-dead neurons. Mitigation: experiment directly measures this; if only the CF6 dead-zone neurons pass, the fold is only slightly beyond what CF6 already predicts.

---

## F4-WQWK-COMMUTE — Attention Score Commutativity Under RoPE Gauge (ASCR)
**F / Track A**

### Mechanism
Track A. RoPE rotates Q and K identically per position: q_pos = R_pos · q_raw, k_pos = R_pos · k_raw, where R_pos is the rotation matrix for position pos. The attention score is q_pos^T k_pos = (R_pos q_raw)^T (R_pos k_raw) = q_raw^T (R_pos^T R_pos) k_raw = q_raw^T k_raw when R is orthogonal (as standard RoPE is). Wait — this is already an identity and the field knows it. The first-principles question to push past the known: does the trained W_Q for Qwen3 have an additional residual symmetry under the RoPE rotation group? Specifically: R_pos W_Q ≈ W_Q R_pos^{(d_head)} for typical pos offsets? If W_Q is (approximately) equivariant under the RoPE rotation group on its input, then certain W_Q directions are irreducible representations of the RoPE rotation group (2D rotation irreps). Under this equivariance, W_Q decomposes into a block-diagonal structure where each block corresponds to one RoPE frequency. This decomposition is NOT rank reduction — it is a coordinate change that reveals the learned structure. The payoff: if W_Q is block-diagonalizable in the RoPE frequency basis, the GEMV W_Q·h decomposes into 1024 independent 2×2 rotations plus a residual, reducing the full 2048×128 head GEMV to 1024 independent 2D projections with amortized cache reuse.

This is a genuinely novel first-principles question: nobody has measured whether trained W_Q matrices at Qwen3 scale are approximately RoPE-equivariant.

### Residency arithmetic
If W_Q is block-diagonal in the 2D RoPE-frequency basis (1024 2×2 blocks per head), storage does not decrease (same number of params). Payoff is compute: GEMV becomes 1024 × (2×2 matmul) instead of 1 × (128×2048 matmul). For CPU inference, 2×2 matmuls can be vectorized as pairs and are cache-resident; the 128×2048 matmul is bandwidth-bound. On Ryzen 5 7530U with 256 KB L2, a 2×2 tile is always in register; the large matmul is always bandwidth-bound. Theoretical speedup on Q-projection: 1024 × (2 FMAs) vs 128×2048 = 262K FMAs — same count, but 2×2 tiles have zero L2 miss while 128×2048 is L2-miss-dominated. Speedup estimate: 1.5–2× for W_Q GEMV, which is 1 of ~28 matmuls per layer, so ~1–3% end-to-end speedup. Small but a structural finding.

### Novelty gloss
No kill list item is adjacent. Closest published: RoPE literature (Su et al. 2023) does not analyze whether trained W_Q matrices are equivariant under their own rotation group — they are designed so the rotation cancels out in the dot-product, not to be equivariant internally. This is a structural question about the trained model that is absent from the published RoPE analysis. If equivariance fails, the structural finding (W_Q does NOT inherit RoPE symmetry) is itself informative for understanding attention learning dynamics.

### Smallest experiment
**Claim**: for W_Q at layer 14, head 0, the off-block-diagonal norm (in the 2D RoPE frequency block basis) is ≤10% of the total Frobenius norm.

Measurement: load W_Q head-0 (128×2048 matrix, viewing as d_head × d_model). Reshape d_model=2048 into 1024 frequency-pairs (RoPE groups consecutive dims). Measure sum of (2×2 block Frobenius norms on diagonal) vs total Frobenius. Ratio < 90% diagonal → off-block significant.

Runtime: 5 min (one matrix load + reshape + norm computation).

**Go threshold**: ≥90% of Frobenius in diagonal blocks → W_Q inherits approximate RoPE equivariance; proceed to block-diagonal GEMV benchmark.
**No-go**: <50% diagonal → W_Q does not inherit RoPE structure; trained Q-projections mix RoPE frequencies. Structural finding: SGD breaks RoPE equivariance systematically.

### Primary risk
RoPE equivariance is a structure that would emerge only if training regularized W_Q toward the frequency subspaces — no mechanism forces it during standard training. Most likely outcome is partial equivariance (~50–70% diagonal), producing a nuanced structural finding rather than a binary result. Mitigation: the experiment produces the diagonal fraction directly; any value is informative.

---

## F5-SPEC-ENTROPY — Spectral Entropy as Calibration-Free Rank Proxy (SECP)
**F / Track B**

### Mechanism
Track B. CF11 measured W_Q r_99/d ≈ 0.63 (via SVD). The spectral entropy H_s = −Σ_i p_i log p_i where p_i = σ_i² / Σ_j σ_j² is a continuous, convex functional of the singular value distribution. H_s is maximized for flat spectrum (H_max = log d, full rank) and minimized for rank-1 (H_s = 0). For CF11's W_Q: if r_99/d ≈ 0.63, the spectrum has substantially sub-maximal entropy. The first-principles question: does W_Q spectral entropy predict the calibration-free NLL cost of truncation at K better than the r_99 threshold alone? Specifically, is the K that achieves ΔNLL < 0.20 nats (the CF11 K=512 result) predictable from H_s computed without any calibration tokens? If spectral entropy is a calibration-free proxy for safe truncation rank, then the optimal K for attention compression (across W_Q, W_K, across layers, across models) can be determined from weight statistics alone — no forward pass needed. This generalizes CF11 to untested model families and matrix types (W_V, W_O, untied lm_head in Qwen3-8B) without running full truncation sweeps. The mechanism: compress at K = argmin_K {K : H_s(W_trunc_K) / H_s(W_full) > τ} for empirically calibrated τ, run one forward-pass NLL check to confirm.

Not CF8 MLP: MLP matrices have near-flat spectrum (H_s ≈ H_max), so spectral entropy would correctly predict high truncation cost for them (consistent with CF7/CF8). This is the unifying-object direction: spectral entropy as the underlying object that explains CF11's W_Q compressibility AND CF8's MLP incompressibility.

### Residency arithmetic
Payoff is not direct residency reduction; it is a calibration-free model-characterization tool. Enables: given a new 70B model (e.g., Llama-3-70B, Qwen3-72B), determine in <5 min (weight statistics only, no forward pass) which attention matrices are safely compressible at what rank. This unlocks CF11-style compression (attention weight residency) on untested model families. For Qwen3-72B attention weights: W_Q (8192×8192, 64 heads) at predicted-safe rank K would save ~64 heads × (8192 − K) × 8192 × 2 bytes. At K=4096 (50%), 8 GB → 4 GB on attention alone. With spectral-entropy-guided compression applied to W_V and W_O as well (untested, likely similarly concentrated), total attention-block residency reduction 2–4×. At 70B IQ4_XS baseline of 5.35 GiB, attention is ~15% = 0.8 GiB; 4× reduction saves 0.6 GiB. Combines with PDAP/RAOK on MLP to reach toward DRAM-resident target.

### Novelty gloss
No kill list item targets spectral entropy as a calibration-free rank proxy. Closest published: GPTQ / SparseGPT use Hessian-based importance, which requires calibration data. This uses weight-only spectral entropy — cheaper and generalizable. Spectral entropy of weight matrices appears in some matrix sketching literature but not as an LLM compression guidance tool. The CF11/CF8 anchor makes this grounded: spectral entropy is not being imported as a decorated claim but as an object that should retrodict the measured findings.

### Smallest experiment
**Claim**: spectral entropy H_s(W_Q^L) predicts the CF11 K=512 safe-truncation result better than chance, defined as: the K* = argmin_K {H_s(W_Q_K) / H_s(W_Q_full) > 0.99} gives ΔNLL within 0.10 nats of the CF11 measured threshold (K=512, ΔNLL=+0.20).

Measurement: compute full SVD of W_Q for layers 14, 20, 27 (3 layers); compute H_s curve vs K; apply τ=0.99 threshold to predict K*; run inference at that K* and measure ΔNLL.

Runtime: ~20 min (3 full SVDs + inference runs).

**Go threshold**: |ΔNLL(K*) − 0.20| < 0.10 nats across all 3 layers → spectral entropy is a calibration-free rank proxy.
**No-go**: K* from spectral entropy diverges from CF11-measured K → spectral entropy does not predict truncation safety; the CF11 finding is not explained by spectral concentration alone. Structural finding: the threshold for safe truncation is a function of more than the singular value distribution (e.g., alignment of singular vectors with calibration activations matters).

### Primary risk
The τ threshold is a free parameter; post-hoc tuning to match CF11 would be circular. Mitigation: use τ=0.99 (derived from information-theoretic first principles: preserve 99% of spectral information) WITHOUT fitting to CF11's numbers. The experiment is a prediction, not a fit.

---

## F6-GAUGE-NORM — RMSNorm Gauge Freedom for W_Q / W_K Scale Absorption (GFNM) [FREE SWING]
**F / Track A** [FREE SWING]

### Mechanism
Track A. Qwen3 uses RMSNorm before Q and K projection: q = W_Q · RMSNorm(h), k = W_K · RMSNorm(h). RMSNorm has learnable gain γ ∈ R^d. The composition W_Q · RMSNorm is: W_Q · (γ ⊙ h / ||h||). The learned γ and the W_Q matrix are co-trained — there is a gauge freedom: scaling γ_i by α and W_Q[:, i] by 1/α leaves the output identical. This is the standard SmoothQuant / LLM-in-a-flash "per-channel scale absorption" observation, BUT the first-principles angle is different: in the gauge where each input channel i has unit contribution to W_Q (||W_Q[:, i]|| = 1), the effective singular value distribution of W_Q becomes the one that reflects only the learned directional structure, not the scale entanglement with γ. The question: does gauge-fixing (absorbing γ into W_Q so that γ_absorbed ≡ 1 and W_Q_gauged = W_Q · diag(γ)) produce a W_Q_gauged with strictly MORE concentrated spectrum than the un-gauged W_Q? If yes, the "true" rank of W_Q (in the gauge-invariant sense) is lower than CF11's measured K=128 ceiling, and the compression could be tighter. The algebraic identity: W_Q · diag(γ) ≡ W_Q_gauged, and the spectrum of W_Q_gauged may differ substantially from W_Q when γ has large dynamic range.

### Residency arithmetic
If W_Q_gauged has r_99/d ≈ 0.40 (vs measured 0.63 for W_Q), then safe truncation K drops from 512 to 320 (roughly). Per-layer W_Q storage: 2048×2048 × 2 bytes = 8 MB. At rank 320: U (2048×320) + V (320×2048) ≈ 2.6 MB. Per-layer savings vs CF11 K=512 (3 MB): ~0.4 MB × 28 layers = 11 MB — marginal on Qwen3-1.7B. On 70B-scale (W_Q 8192×8192): proportionally ~150 MB additional savings. Small incremental benefit on its own; meaningful if stacked with F1/F2.

### Novelty gloss
Closest published: SmoothQuant absorbs scale from activations into weights for quantization. This applies the same absorption to W_Q before SVD, asking whether the spectral concentration is masked by scale entanglement. Not on arXiv as an SVD-compression precondition check. Kill list: SmoothQuant-adjacent mechanisms have appeared (CRANK killed in Track A R2), but CRANK targeted RMSNorm scale for quantization, not for SVD concentration measurement. The structural question (does gauge-fixing change the effective rank of W_Q?) is novel as a first-principles investigation.

### Smallest experiment
**Claim**: ||W_Q_gauged||_spectral_entropy < ||W_Q_original||_spectral_entropy for at least 10 of 28 layers, where gauging = right-multiply by diag(RMSNorm_gamma_L).

Measurement: load W_Q and gamma for all 28 layers; compute W_Q_gauged = W_Q @ diag(gamma) for each layer; compute spectral entropy of both (top-256 singular values sufficient as proxy); compare.

Runtime: 10 min.

**Go threshold**: H_s(W_Q_gauged) < 0.95 × H_s(W_Q_original) on ≥10 layers → gauge absorption increases concentration; K can be tightened further. Proceed to NLL sweep on W_Q_gauged at K=320.
**No-go**: H_s unchanged → γ has near-unit dynamic range in Qwen3 (consistent with SmoothQuant literature finding that Qwen-family does not have extreme outlier scales). Structural finding: Qwen3's RMSNorm γ does not entangle scale into W_Q spectra.

### Primary risk
Qwen3 RMSNorm γ may have small dynamic range (all γ_i ≈ 1), in which case the gauge absorption is a no-op. This is actually the most likely outcome given that Qwen3 is a well-regularized model. Mitigation: 10-min measurement before any further cascade investment.

---

## Convergence handles

1. **Canonical correlation / principal angle between W_Q and W_K left singular subspaces** (F1, F2 both rely on inter-matrix geometric relationships within one layer — cross-orientation F/C convergence possible here)
2. **Per-head Frobenius energy distribution under the head-permutation gauge** (F2's binary settler; relevant to any head-redundancy idea in other orientations)
3. **Per-neuron gate linearizability fraction on calibration distribution** (F3; any AERO-adjacent idea across orientations will need this number)
4. **Spectral entropy as calibration-free rank proxy** (F5; if confirmed, unlocks compression prediction for W_V, W_O, untied lm_head — direct feed into Reach orientation cascades)
5. **RoPE frequency block-diagonality of W_Q** (F4; convergence signal if any Composition or Constraint-Alien idea independently surfaces RoPE-frequency coordinates)
6. **Gauge-fixed W_Q spectrum concentration** (F6; convergence with any SmoothQuant-ancestry idea from C or U orientations)
