# Stage 4 — Skeptic Explorer Report — Run 003
Date: 2026-05-09

Substrate: 26 REFINE, 0 KILL from Stage 3. Five orientations (R, C, F, U, A). Kill-list includes v2-CHEAP-TEST-001, v2-S3-R004-001. Stage 3 noted REFINE for every advancer; no KILLs available for cross-pollination (section 3 will draw on DOWNGRADE candidates from prior rounds instead).

---

## 1. Frame-Exhaustion Map

### Saturation-exhausted frames

**Per-matrix individual attention weight SVD.** SVD-LLM (Mar 2024) applies SVD independently to W_V and W_O. The per-matrix approach is now published prior art. Any proposal that decomposes W_V or W_O individually without forming the fused M_L = Σ_h W_O^h W_V^h falls within SVD-LLM's coverage. *Stripped primitive*: what remains is the fused operator M_L itself — not yet published. F1-VOFR correctly identifies this object; the saturation boundary is the per-matrix frame, not the fused-operator frame.

**Activation-scale-weighted INPUT-side SVD of MLP weights.** ASVD and ARHQ (arXiv:2605.00140) both apply activation-scale weighting to the input channels of W before decomposing. This frame is saturated — any proposal that rescales W's input directions by calibration activation magnitude before SVD is within this family. *Stripped primitive*: the OUTPUT-side projection (Delta_L = W_down · A_L, measuring the actual span of W_down's outputs under calibration). F4-WAOS correctly targets this stripped primitive.

**Per-channel scale+sign absorption from SmoothQuant/OS+ family.** The continuous scale migration (SmoothQuant) and combined scale+sign transform (OS+) together saturate the per-matrix, per-channel transform frame. Any proposal that applies per-channel affine to individual weight matrices without cross-layer consistency enforcement falls into the OS+ saturation zone. *Stripped primitive*: the residual-stream gauge enforcement — applying the same sign convention across all layers sharing channel d, not per-matrix independently. A2-RTGF's distinguishing claim is exactly this stripped primitive.

**Cross-layer W_Q stacked-basis / joint-diagonalization.** Hard-killed by v2-CHEAP-TEST-001. The permuted-rows baseline gap was 0.0001, proving the "coherence" was an artifact of within-layer CF11 concentration trivially summed. *Stripped primitive*: per-layer K=128 W_Q compression (CF11). This is the ceiling; cross-layer stacking adds nothing.

**ReLU-sparsity-dependent skipping methods (Deja Vu family).** Deja Vu (ICML 2023) and its trained-predictor variants are killed. *Stripped primitive*: zero-overhead top-K cache from previous token's actual top-K firing set, keyed on W_up magnitudes (CF1-grounded). A5-TCRD correctly identifies this as the live stripped primitive.

### CF9-exhausted frames

**Convolution-analog weight tiling (Winograd, FFT-fused matmul).** Killed in R3/S2. Winograd requires sliding-kernel spatial reuse; standard GEMV has no such structure. Any proposal importing convolution algebra for transformer GEMV fails CF9. *Stripped primitive*: none useful — the precondition (spatial signal structure) simply does not hold.

**Random Fourier Features on weight matrices.** Killed in R6-B (RFSK). Bochner's theorem requires the kernel being approximated to be positive-definite and shift-invariant. Weight matrices are not kernels. *Stripped primitive*: JL-style random projection as a dimensionality reduction, but SVD dominates JL on full-rank matrices (SVD minimizes Frobenius error; JL doesn't). Dead in the current context.

**Sketches on dense signals (Count-Sketch, TensorSketch).** Killed in R6-B (CSKQ). L1 error bound is 5470× signal magnitude at INT4 resolution; Count-Sketch requires sparse signals. *Stripped primitive*: frequency-domain approximation of attention scores (not weights) — open if attention patterns are sparse, but CF11 shows W_Q is global, not attention-pattern-sparse.

### CF10-exhausted frames

**Full-rank affine correction across model-scale boundary (WDLA shape).** Killed empirically (Track A R2). Fitting A_L: R^{1024} → R^{2048} requires 2.1M parameters vs 2.05M calibration values — underdetermined. *Stripped primitive*: low-rank-by-construction cross-scale affine at rank ≤ 64 (reduces to 200K parameters, well-conditioned at 1K tokens). This stripped primitive is open and not yet explored.

**High-parameter codebook fits with small calibration sets.** The general pattern: if n_params_to_fit > n_tokens × d_output, ridge must be extremely aggressive or the fit must be low-rank by construction. Any proposal fitting > d² parameters on < 1000 tokens is in the CF10 danger zone. *Stripped primitive*: measurement-only approaches (SVD of measured objects, principal-angle computations, row-mean statistics) — all pass CF10 because they do not fit parameters.

---

## 2. Orientation-Saturation Diagnostics

**Orientation R (Reach).** Advanced 6 ideas to Stage 3; all REFINE. R's convergence with F on the W_V/W_O spectrum cluster (WAVQ + F1-VOFR) is the round's strongest multi-orientation signal. R's natural style (ambitious cascade, residency arithmetic, hardware-grounded numbers) is well-utilized in this round — RAOK-72 and SOBIB-V are the strongest Track B survivors from this orientation. Anti-frame addition recommendation: add "per-matrix attention weight SVD without fused composition" to R's anti-frame list (SVD-LLM saturation; the fused M_L operator is the live frame). KEEP.

**Orientation C (Composition).** Advanced 5 ideas; all REFINE. C1-QAOC (score 12, convergence cluster 3) and C5-WUPDOWN (score 12, convergence cluster 2) are the round's strongest Composition outputs. C6-SPECDRIFT advanced as a free-rider. The C4-GEMBED rejection (A1=0, negligible residency) shows C's anti-frame list should explicitly add "coupling claims whose residency payoff is < 1% of model size" — mathematically correct coupling is not enough if the payoff cannot move the deployment arithmetic. Anti-frame addition recommendation: add "coupling claims with confirmed residency < 1% of 70B target" to C's anti-frame list. KEEP, TIGHTEN.

**Orientation F (First-Principles).** The round's highest-scorer with F1-VOFR (14) and F4-WAOS (14). F also contributed F5-HPGO (frame-novelty, score 12), F2-RGAS and F3-SRSC (score 12 each). F's algebraic-identity derivation style is this ladder's strongest source of elegant-equivalence ideas. F7-TEGRA advanced out-of-orientation. Anti-frame addition recommendation: add "importing theorem families (wavelets, RFF, sketches) on non-smooth objects without CF9 precondition audit" to F's anti-frame list. KEEP.

**Orientation U (Unconventional Substrate).** Advanced 5 ideas. U6-NQDS (score 12, convergence) and U1-GECP (score 12) are strong. U5-GTBP was rejected (A1=0, 28 µs/token overhead negligible at 70B). The U orientation's ideas are consistently Track B and systems-level — no convergence with F/C's attention-spectrum work. This is expected; U's substrate primitives are orthogonal to weight-structure findings. Anti-frame addition recommendation: add "metadata-overhead optimizations where total gain < 0.1% of per-token cost" to U's anti-frame list (generalizing U5-GTBP's reject reason). KEEP.

**Orientation A (Constraint-Alien).** Advanced 5 ideas. A2-RTGF (score 13, elegant-equivalence) is the round's top Path 4 idea. A1-GFSM was rejected (A1=0, sub-0.5% residency gain). The rejection pattern for A's ideas shows a recurring structural risk: the constraint forces a mathematically correct mechanism whose residency payoff is negligible at 70B scale. A5-TCRD (score 10) and A4-APPA (score 9) are borderline. Anti-frame addition recommendation: add "constraint-forced mechanisms with ≤1% residency payoff at 70B — even if mathematically elegant, they do not move the deployment floor" to A's anti-frame list. KEEP, TIGHTEN.

---

## 3. Cross-Pollination Opportunities

Run 003 produced 26 REFINE and 0 KILL, so the standard cross-pollination pipeline (REGENERATE/KILL → rescue in another orientation) is thin. Draw instead on the prior-round DOWNGRADE and deferred-from-Stage-4 ideas that carry stripped primitives worth transposing.

**Cross-pollination candidate 1 — v2-S3-R004-001 (DOWNGRADED: gauge-freedom general residual-stream rotation).**
Kill reason: CF9 partial failure — general O(d) rotation is incompatible with RMSNorm per-channel scale (only the generalized permutation group survives). Stripped primitive: per-channel attention-subspace importance ordering. In Orientation F's pose: the magnitude of W_Q right singular vectors per channel is a lossless coordinate-change signal — measuring which channels contribute most to W_Q's concentration could ground a MIXED-PRECISION W_Q layout (high-precision for channels that span the K=128 subspace, lower-precision for the orthogonal complement). This is not per-channel scale (SmoothQuant) — it is per-channel subspace-participation weighting, derived purely from W_Q's measured spectrum. Transposition: **F-pose → mixed-precision W_Q allocation keyed on per-channel V_Q participation**. → Expanded in S3.

**Cross-pollination candidate 2 — R3/S4-deferred CALFAV (Calibration-Fitted Diagonal+LR Attention Kernel).**
Deferred because "most ambitious arch-transpose." The CF10 ill-conditioning risk was the blocker at full affine scale. Stripped primitive: diagonal + low-rank calibration fit on attention, where the diagonal component passes CF10 trivially (d parameters vs d×N samples, always conditioned) and the low-rank component is bounded by construction (rank K = 16 → 2048×16 + 2048×16 = 65K parameters, well-conditioned at 200 tokens). The Composition orientation never targeted attention diagonal fits directly. Transposition: **C-pose → CF11 W_Q spectrum × calibration-diagonal coupling**. → Expanded in S5.

**Cross-pollination candidate 3 — R2/S4-deferred LSEI (Logit Shortlist via LSH on Vocab).**
Deferred as "a refinement; deferred." The stripped primitive: using LSH or approximate nearest-neighbor over lm_head rows to shortlist the top-K probable output tokens without full matmul. CF12 killed tied lm_head SVD; LHQD empirically confirmed tied embedding is full-rank. But CF12/LHQD was about WEIGHT-LEVEL decomposition — the logit shortlist mechanism requires only approximate nearest-neighbor in the output distribution space, not weight compression. On UNTIED lm_head (Qwen3-8B, Llama-3 70B), CF12's kill does not apply. The Constraint-Alien orientation never posed the "content-addressable output token retrieval" frame. Transposition: **A-pose → content-addressable vocab retrieval on untied lm_head**. → Expanded in S7.

---

## 4. Cross-Domain Seed Exploration

For this round, given the kill list and the frame-exhaustion map, two professional priors are highest-leverage.

### Professional prior 1: Compiler writer

*Elegant-equivalence prompt: what algebraic identity does a compiler writer's view of the transformer forward pass reveal — specifically, which computations are dead code that can be eliminated or hoisted because their output is already present in a more compact form?*

The compiler's view: a compiler identifies common subexpressions (CSE), hoists loop-invariants out of inner loops, and eliminates dead stores (values written but never read). Applied to transformer inference:

- **CSE on W_Q / W_K per-token cost**: RoPE applies a deterministic per-position rotation to both Q and K after the projection. But the W_Q row mean μ_row is invariant to position-dependent rotation applied multiplicatively. The compiler's insight: the position-independent component of W_Q output is a loop-invariant relative to the position variable — it can be computed once and cached. F3-SRSC already captures the row-mean case. The compiler view extends it: what is the maximal position-invariant subspace of W_Q's output? That subspace's contribution is CSE across positions and can be collapsed into a static bias.

- **Dead-code elimination on W_down output directions**: CF1 shows W_up firing dominates the post-SwiGLU output. W_down maps these activations to residual updates. A compiler would ask: after identifying the dead-code neurons (those rarely firing across all tokens), are there output directions of W_down that are written only by those neurons? Those output directions are dead stores — W_down rows that only dead-code neurons write to. If the top-K "live neurons" (CF1 firing-rank dominance) produce outputs that concentrate in a subspace of W_down's range, the complement directions are dead code. This is precisely F4-WAOS's claim — but the compiler framing adds a direction: **the dead-output-direction test is orthogonal to the live-output-subspace test**. Measuring r_eff via Delta_L (F4-WAOS) tests the live subspace; measuring the complement tests the dead-code fraction. These are dual.

- **Register allocation for the residual stream**: the compiler's register allocator minimizes live-variable pressure by reusing registers when values are dead. Applied to the residual stream: after each MLP layer writes δh_L, what fraction of the d_model=2048 channels become "dead" (not read by subsequent layers' dominant subspace)? If residual stream channels accumulate dead directions across layers, they are eligible for lower-precision storage in the "register" of the DRAM-resident state buffer. The CF11 finding (W_Q spans only K=128 dimensions globally) suggests 1920 of 2048 channels are outside the Q-routing subspace — but they may still be live for other weight matrices. The question: is there a cross-layer dead-channel structure where certain channels are not load-bearing for ANY downstream matrix in layers L through 28?

→ **Highest-leverage elegant equivalence from compiler prior**: the position-invariant subspace of W_Q output as a CSE-hoistable component, computable once per model rather than once per token per position. This is a strict generalization of F3-SRSC, and may admit a more aggressive compression (not just the row-mean scalar, but the full low-rank position-invariant component). → Generates S1.

### Professional prior 2: Control theorist / Kalman-filter engineer

*Elegant-equivalence prompt: what conserved quantity or state-observer identity does a control theorist see in the residual stream that would let a layer be replaced by a fixed-form operator on the conserved quantity alone?*

The control theorist views the residual stream as a state trajectory: h_0 → h_1 → ... → h_L where each transition h_{L+1} = h_L + δh_L is a small-perturbation update (CF2: cos ≈ 0.99, the "small perturbation" regime). A Kalman filter tracks a state estimate and updates it with measurements. The analogy:

- **h_L is the state estimate**: it accumulates all information from the input sequence through layer L.
- **Each attention + MLP block is a measurement update**: it incorporates the "measurement" (query-key attention scores + firing neurons) to update the state.
- **The process noise covariance** (covariance of δh_L across tokens) determines how much information each layer injects into the stream.

Control-theorist insight: in a Kalman filter, layers (measurement updates) where the innovation (new information) is small relative to prior uncertainty are low-information updates — they can be approximated by a lighter operator. The "information content" of a layer's update is measured by the KL divergence between the posterior (after the update) and the prior (before). If CF2 (cos ≈ 0.99) holds uniformly, then most layers inject very little information per token — the posterior is barely different from the prior. This suggests a layer-importance schedule grounded in the ACTUAL information injected (measured empirically per layer) rather than uniform treatment.

The Kalman insight that is NOT in the current pipeline: the **Kalman gain** K_L = P_L H_L^T (H_L P_L H_L^T + R_L)^{-1} can be approximated when R_L >> H_L P_L H_L^T — i.e., when the measurement noise (activation quantization error) dominates the signal. At low-bit quantization (INT2/INT4), R_L grows; there exists a layer-specific crossover point where the quantization-noisy update is worth less than the residency cost. This is not standard layer-pruning (which removes layers entirely) but **layer-precision-scheduling by information-injection estimation**.

→ **Highest-leverage elegant equivalence from control-theory prior**: per-layer information-injection measurement (KL divergence between h_L and h_{L+1} distributions across calibration tokens) as a layer-specific quantization budget. Layers with low KL (small injection, close to a no-op) can tolerate more aggressive quantization (INT2 or even binary) because their quantization error is small relative to the information they add. Layers with high KL (the attention-heavy or early-transformation layers) need higher precision. This is distinct from Depth-Quantile Precision Map (DQPM, rejected for no smallest test) because it is grounded in a specific measurable quantity (per-layer KL) not a vague depth-stratification heuristic. → Generates S2.

---

## 5. Constraint-Driven Reframing

From the STAGE4 constraint menu, the following two are least represented in the current kill list:

### Constraint 1: No RAM during decode — only registers, L1/L2/L3, NVMe

The kill list has no idea targeting the L3-as-DRAM-substitute frame. On Ryzen 5 7530U (Zen3): L3 cache = 16 MB. This is a severe constraint — the model cannot hold even a single layer's weights in L3 for the 70B target. But it forces a mechanism: **the computation must be structured so that each L3-resident chunk is fully processed before the next chunk is loaded.** This is a stream-computation constraint.

What mechanisms does this force that the standard pipeline has not considered?

- **Cache-resident codebook inference**: if W_Q and W_K are compressed to fit in L3 (16 MB), the attention mechanism runs entirely from L3 during the QK^T pass, then the V-weighted output from NVMe. At K=128, W_Q = 28 × 2048 × 128 × 2 bytes = 14.7 MB — fits in L3. W_K similarly at K=512 = 28 × 2048 × 512 × 2 bytes = 58.7 MB — does not fit. But at K=128 for W_K too (K=128 is +0.82 nats, borderline): 14.7 MB — fits jointly with W_Q.

If both W_Q and W_K are in L3 at K=128, the query and key projections can run at L3 bandwidth (~400 GB/s on Zen3's L3) rather than DRAM bandwidth (11.5 GB/s). Per layer, QK^T computation using L3-resident W_Q and W_K at K=128: 14.7 MB / 400 GB/s = 37 µs — negligible. The attention score computation is essentially free compared to the MLP GEMV, which remains DRAM-bound. This is a meaningful repartitioning of the computation: run attention heads at L3 speed, stream MLP weights from DRAM. → Generates S4.

### Constraint 2: Content-addressable — every weight chunk retrievable by hash of its content

The kill list has no content-addressable weight store idea except NTFS-Reflink Codebook Deduplication (R1/S4 deferred, Windows 11 Home limitation). The constraint forces: if two weight chunks have the same content (within a tolerance), they map to the same address. This enables deduplication across layers.

What is the actual deduplication potential in Qwen3-1.7B?

- Q4_K_M GGUF quantization bins weights into 4-bit values with per-block scales. Across 28 layers and multiple weight matrices, the distribution of quantized INT4 nibble blocks is shaped by the empirical weight distribution. CF3 (K-dependent Jaccard ~0.31 at K=1%) shows that activation outlier channels are partially shared across tokens — but do WEIGHT blocks show similar inter-layer similarity?

- The constraint forces a different question: not "are individual weights identical" but "are 32-byte or 64-byte blocks of quantized weights hash-colliding across layers?" Even a 5% collision rate across 28 layers saves significant storage and could enable a compact write-once block store for GGUF files.

- The mechanism is not per-weight comparison (too fine) but **block-level content-addressable GGUF store**: quantized 32-weight blocks are hashed (BLAKE3 128-bit hash) and stored in a content-addressable index. On load, blocks are looked up by hash; duplicate blocks are stored once. This is CoW reflink semantics without requiring filesystem support.

→ The content-addressable constraint generates a weight-deduplication mechanism that bypasses the Windows 11 Home reflink limitation (no filesystem dependency). → Generates S8.

---

## 6. Generated Ideas (S1–S8)

---

### S1 — QKPI: Q/K Position-Invariant Subspace Hoisting

**Ambition**: CF11 established that W_Q spans a K=128 global subspace (all 16 heads share ~128 effective dimensions). F3-SRSC found the W_Q row-mean scalar case. S1 generalizes: extract the full position-invariant subspace of W_Q output (not just the scalar mean) and hoist it as a static computation-time constant, reducing per-token Q computation cost. The position-invariant directions of W_Q x are exactly those directions invariant under RoPE rotation — they form the orthogonal complement of the RoPE-rotated subspace. If a significant fraction of W_Q's K=128 concentrated directions is position-invariant, those directions can be precomputed once per input token (not once per position) and cached in the K-cache without the per-position RoPE update.

**Mechanism**: Track A. Define V_inv ⊆ R^{K=128} as the subspace of W_Q's top-K singular vectors that is invariant under the RoPE rotation for any position (i.e., rows of U_Q such that RoPE(U_Q e) ≈ U_Q e for any rotation angle). RoPE applies a rotation R(θ_p) in d_head / 2 = 64 2D planes per head. If U_Q's top singular vectors are aligned with the RoPE-invariant directions (the real and imaginary parts of frequency-zero modes), then those directions experience no rotation under RoPE and are truly position-invariant. The elegant equivalence: Q_invariant(x) = V_inv^T x is a static computation per input vector x; Q_variant(x, p) = V_var^T (W_Q x) uses the remaining directions with per-position rotation. Separate the two components at model-preparation time. The invariant component is computed once per token (amortized over all sequence positions that query this token), not re-rotated per position.

**Why our findings make this work**: CF11 (K=128 global W_Q compression at +0.98 nats) establishes that W_Q's relevant subspace is low-dimensional. F3-SRSC (W_Q row-mean magnitude as a Q dynamic-range reduction mechanism) establishes that per-row static components of W_Q output exist and may be non-trivial. RoPE's frequency structure (each head applies a fixed set of 64 rotation frequencies) is deterministic — the invariant subspace is computable from the model weights alone without any calibration data.

**What would have to be true**: RoPE's rotation spectrum must include a non-trivial fraction of near-zero frequencies (or zero-frequency modes) that span a significant portion of W_Q's K=128 concentrated subspace. If RoPE uses only high frequencies (the standard RoPE formulation uses frequencies 10000^{-2i/d_head} for i=0..63, ranging from 1.0 to 10000^{-2}), the lowest-frequency modes have rotation angles close to 0 for nearby positions — partially invariant at short range. The claim is empirically testable.

**Experiment cascade**: E1 (35 min): compute the principal angles between W_Q's top-128 singular vectors U_Q and the RoPE zero-frequency subspace (the identity-rotation eigenspace) for each of 28 layers. Measure: what fraction of U_Q's singular values are in near-zero-frequency directions? GO if > 20% of K=128 directions are position-invariant. E2 (20 min): if GO, split W_Q output into invariant + variant components; verify ΔNLL = 0 on the split. E3 (30 min): implement K-cache with invariant component precomputed; measure K-cache write bandwidth reduction.

**What you're NOT proposing**: Not per-position Q caching (trivial with attention masking). Not RoPE-free attention. Not position encoding ablation.

**Track**: A. Anti-frame: sits outside F's anti-frame (not plain quantization), outside C's anti-frame (not a gestured composition without math), outside A's anti-frame (mechanism is checkable, not a slogan). Orientation white space: this is a First-Principles idea that requires knowledge of both CF11 (W_Q spectrum) and RoPE's frequency structure — it sits in the F/C white space that neither orientation generated in this round.

**CF tether**: CF11 (W_Q global K=128 spectrum). **Elegance class**: `algebraic-identity` — the position-invariant subspace is a structural property of (W_Q, RoPE) joint action; extracting it is a computational equivalence, not an approximation. **Smallest test**: 35 min.

---

### S2 — LQIS: Layer-Quantile Information-Injection Scheduling

**Ambition**: Use per-layer KL divergence of residual stream (KL(h_{L+1} || h_L) measured across calibration tokens) as a grounded precision budget. Layers with small KL injection (close to no-op) are allocated lower bpw; layers with high KL are protected at higher precision. This is the CF2 (residual additivity cos ≈ 0.99) finding converted from a static observation into a dynamic precision schedule, grounded in the control-theory measurement of "innovation" rather than the standard depth-quartile heuristic.

**Mechanism**: Track B. Compute per-layer innovation score: s_L = mean_{t in calibration} KL(P(h_{L+1}|x_t) || P(h_L|x_t)) approximated as 0.5 × ||h_{L+1}^{(t)} - h_L^{(t)}||² / Var(h_L) (the Gaussian-approximation KL). This is the squared increment normalized by prior variance — exactly the Kalman-gain scalar in a uniform-Gaussian model. Rank layers by s_L; assign INT2 to the bottom quartile, INT4 to the middle two quartiles, INT8 to the top quartile. No calibration-fitting of parameters: the assignment is a lookup against the measured s_L values.

**Why our findings make this work**: CF2 (cos(h_L, h_{L+1}) ≈ 0.99) establishes that residual increments are universally small, but does not distinguish which layers add MORE information than others. v2-CF1 (K=1% Jaccard mean=0.39 across all 28 layers) shows that the outlier-channel dynamics vary by depth (mid-depth layers cluster at 0.41-0.44; deep layers spread more). This depth-varying activation pattern suggests layer-varying information injection. The KL measurement quantifies this directly.

**What would have to be true**: The per-layer s_L values must have sufficient variance across layers that the bpw schedule produces meaningfully different budgets. If all s_L are within 10% of each other (all layers inject roughly equal information), the schedule degenerates to uniform precision. The measurement settles this.

**Experiment cascade**: E1 (30 min): forward pass on 200 calibration tokens, cache h_L per layer, compute ||h_{L+1} - h_L||² / Var(h_L) for all 28 layers. Plot s_L distribution. GO if std(s_L) / mean(s_L) > 0.30 (30% coefficient of variation — enough spread for a meaningful schedule). E2 (45 min): if GO, assign bpw schedule (INT2/INT4/INT8) by quartile of s_L; measure ΔNLL vs uniform INT4 at matched mean bpw. E3: 70B residency arithmetic if E2 shows ≤+0.20 nats degradation.

**What you're NOT proposing**: Not depth-quartile pruning (which removes layers). Not DQPM (rejected for no smallest test — S2 has a 30-min smallest test). Not standard mixed-precision quantization without a grounded criterion.

**Track**: B. Anti-frame: sits outside U's anti-frame (not standard SIMD optimization), outside R's anti-frame (this is a single-measurement binary settler, R's style, but it feeds Track B residency math). Orientation white space: C orientation never composed CF2's numerical value with the per-layer precision budget question. The coupling (CF2 cos ≈ 0.99 as prior + per-layer s_L as innovation) is a C-shaped idea generated from a control-theory prior.

**CF tether**: CF2 (residual additivity), v2-CF1 (depth-varying Jaccard). **Elegance class**: `calibration-fit` (measurement-derived schedule, no optimization). **Smallest test**: 30 min.

---

### S3 — QCMI: W_Q Channel Subspace Mixed-Precision Allocation

**Ambition** (Cross-pollination from v2-S3-R004-001 stripped primitive): The failed gauge rotation (O(d) incompatible with RMSNorm) left a stripped primitive: per-channel W_Q subspace-participation weighting. S3 transposes this primitive into the F/C white space. Each of the 2048 residual channels has a different "participation weight" in W_Q's K=128 concentrated subspace — measured by the L2 norm of the corresponding column of V_Q^T (the right singular vectors of W_Q). Channels with high V_Q participation are load-bearing for the K=128 compression; channels with low participation can be stored at lower precision without affecting the compressed W_Q's quality. This is a per-channel storage schedule derived from the weight's own spectrum.

**Mechanism**: Track B. For W_Q at the CF11-grounded K=128, compute the column norms of V_Q^T[:,0:128] (the top-128 right singular vectors). Each column i gives the L2 participation of input channel i in the K=128 subspace. Sort channels by participation; partition into top-20% (high participation, store at BF16) and bottom-80% (low participation, store at INT4). Apply this partition to W_Q only (not W_K or W_V — those are different objects). The losslessness claim: the compressed W_Q produces the SAME K=128 approximation as if stored at full BF16, because the low-participation channels contribute negligibly to the top-128 singular directions. No calibration data needed — the partition is derived purely from V_Q.

**Why our findings make this work**: CF11 (W_Q K=128 at +0.98 nats) establishes the target subspace. The per-channel participation weight is a direct geometric consequence of the SVD measurement already required for CF11-based compression. The insight: the CF11 compression preserves the output subspace regardless of input precision in the low-participation channels.

**What would have to be true**: Low-participation channels must not contribute significantly to the K=128 approximation's accuracy. This holds if the V_Q column norms are sufficiently heterogeneous (high variance across channels). If all channels contribute equally (uniform V_Q column norms), the partition is vacuous.

**Experiment cascade**: E1 (10 min): compute V_Q column norms for all 28 layers; measure coefficient of variation. GO if CoV > 0.5. E2 (20 min): apply mixed-precision W_Q per the partition; verify ΔNLL at K=128 vs uniform-BF16-K=128 baseline. GO threshold: ΔNLL degradation ≤ +0.02 nats (the precision savings should be near-lossless relative to the K=128 approximation). E3: compute residency savings at 70B.

**What you're NOT proposing**: Not per-channel activation scale migration (SmoothQuant). Not arbitrary per-channel mixed-precision (which requires calibration). Not a new SVD decomposition beyond what CF11 already requires.

**Track**: B. Anti-frame: sits outside F's anti-frame (the move is a structural consequence of the spectrum, not plain quantization). Outside A's anti-frame (residency payoff depends on CoV — if CoV > 0.5 and 80% of channels are stored at INT4, the W_Q residency reduction is 80% × (0.5 byte - 2 bytes) per channel = meaningful). Orientation white space: this idea sits between F (algebraic identity from spectrum) and C (composition of CF11 + per-channel weight structure) — neither orientation proposed per-channel W_Q mixed-precision from V_Q column norms.

**CF tether**: CF11 (W_Q K=128 global spectrum). **Elegance class**: `subspace-alignment` — the partition is derived from the subspace that defines the compression, not from an external calibration. **Smallest test**: 10 min.

---

### S4 — ATCL: Attention-Tier Cache-Line Partitioning (L3-resident W_Q/W_K)

**Ambition**: If both W_Q and W_K are compressed to K=128 (CF11: +0.98 and +0.82 nats respectively), their combined footprint is 28 × 2 × 2048 × 128 × 2 bytes ≈ 29.4 MB — too large for L3 (16 MB). But at K=64 with acceptable quality (not yet measured — this is the open question), W_Q + W_K at K=64 = 28 × 2 × 2048 × 64 × 2 bytes ≈ 14.7 MB — fits in L3 cache. If attention projection weights reside in L3, per-token QK^T computation runs at L3 bandwidth (~400 GB/s) vs DRAM bandwidth (11.5 GB/s). The speedup on the attention projection: 35× for those operations. The bottleneck becomes MLP, not attention, at 70B.

**Mechanism**: Track B. Compress W_Q and W_K to K=64 (not yet measured; K=64 per-head was NO-GO at +1.53 nats but K=64 GLOBAL is untested — this is a critical measurement gap). If K=64 global W_Q passes quality (analogy: K=128 was +0.98 nats; the K=64 quality point is between +0.98 and +1.53 nats, depending on whether the spectrum decay is monotone), pack the K=64 compressed W_Q and W_K into a consecutive 14.7 MB GGUF section. Use VirtualLock or mlock to pin this section in L3 (fits comfortably). All other weights stream from DRAM. The W_Q/W_K attention projection at 400 GB/s L3 bandwidth saves ~35× compute time vs DRAM-bandwidth-bound access.

**Why our findings make this work**: CF11 establishes W_Q's compressed representation is viable at K=128. The L3 bandwidth advantage on Zen3 is a documented hardware property. The attention projection is naturally separable from the MLP projection, enabling tier-based residency assignment. v2-CF2 (cross-layer W_Q subspace independent across layers) is irrelevant — S4 does NOT attempt cross-layer sharing, only per-layer K=64 global compression.

**What would have to be true**: K=64 global W_Q must give ΔNLL ≤ +1.5 nats (tolerable degradation). This is not yet measured. The 14.7 MB must actually pin to L3 under VirtualLock (on Windows 11, VirtualLock requires SeImpersonatePrivilege but is available to elevated processes). The L3 bandwidth advantage must be realized for sequential W_Q column accesses (the Zen3 L3 serves sequential read well at measured 400 GB/s).

**Experiment cascade**: E1 (15 min): measure ΔNLL at K=64 global W_Q (not per-head) and W_K. GO if ≤+1.5 nats combined. E2 (30 min): pack K=64 W_Q + W_K into a 14.7 MB pinned buffer; run forward pass with L3-resident attention projections vs DRAM-resident; measure tokens/sec improvement on Qwen3-1.7B. E3: scale arithmetic to 70B.

**What you're NOT proposing**: Not cross-layer W_Q basis sharing (killed). Not per-head K=64 compression (killed at +1.53 nats). Not attention sink pinning (separate mechanism).

**Track**: B. Anti-frame: outside U's anti-frame (VirtualLock is an existing OS API, not a custom kernel). Outside A's anti-frame (mechanism produces real throughput numbers, not a constraint-exercise). Orientation white space: U orientation proposed VirtualLock (U2-VLPE) for general working-set pinning but did not combine it with CF11 attention-weight compression to target the L3 fit threshold. This idea sits in the U/F white space.

**CF tether**: CF11 (W_Q/W_K spectrum). **Elegance class**: `conserved-quantity` — the L3-resident subspace is a conserved computational resource across all token positions; attention projections in that subspace are always L3-hit. **Smallest test**: 15 min.

---

### S5 — CQDA: Calibration-Diagonal-Only Attention Kernel (CF10-safe CALFAV rescue)

**Ambition** (Cross-pollination from R3/S4-deferred CALFAV stripped primitive): CALFAV was "most ambitious arch-transpose" and was deferred. The CF10 ill-conditioning risk (full affine A: R^{d×K} → R^{d×K}) was the structural blocker at the sizes CALFAV proposed. S5 restricts to diagonal-only calibration: fit a per-channel scale d_L ∈ R^d on W_Q's input channels using the empirical activation covariance (d parameters, trivially well-conditioned). This is the diagonal component of what CALFAV attempted — but the diagonal alone may capture enough of the calibration benefit to be worth measuring, without the CF10 risk.

**Mechanism**: Track A. The calibration-diagonal transform is: for each layer L, compute d_L^h = sqrt(diag(Cov(W_Q^h h_normed))) across calibration tokens — the per-channel standard deviation of the W_Q-projected normalized residual stream. Apply a diagonal rescaling to W_Q's input: W_Q_cal = W_Q diag(1/d_L). This is an input-whitening step. Whitened W_Q may have more concentrated spectrum than the raw W_Q (if the input covariance is heteroscedastic, it inflates some singular values artificially). Post-whitening, the K=128 approximation captures more variance per singular direction.

The CF10 check: d parameters = d_model = 2048 per layer. n_independent_samples = n_calibration_tokens × 1 (scalar per channel). At 200 calibration tokens, n = 200 >> 2048 only if the activation covariance is estimated per-channel (not jointly) — which it is in the diagonal case. Effectively well-conditioned: 200 >> 1 per channel. CF10 PASS.

**Why our findings make this work**: F2-RGAS (run 003) tests whether absorbing the RMSNorm gain g improves W_Q spectrum. CQDA tests whether absorbing the INPUT ACTIVATION variance (empirical covariance) improves it further. These are different objects: g is a model parameter (learned scale); d_L is a calibration-derived statistic (observed input variance). CQDA is the activation-side analog of RGAS's weight-side gain absorption.

**What would have to be true**: The residual stream h_normed must have heteroscedastic channel variance at the input to W_Q (some channels contributing more variance than others). CF3 (K-dependent Jaccard) suggests this: certain channels are consistently higher-activation (the static outlier channels at K=0.1%). If the diagonal covariance is dominated by the outlier channels, whitening by d_L will concentrate the W_Q input distribution, potentially improving the spectral concentration of W_Q_cal.

**Experiment cascade**: E1 (20 min): compute per-channel variance of W_Q input (h_normed) across 200 calibration tokens; measure CoV of the diagonal. GO if CoV > 0.5 (heteroscedastic enough to matter). E2 (20 min): compute W_Q_cal = W_Q diag(1/d_L); measure r_99/d of W_Q_cal vs W_Q raw. GO threshold: r_99/d of W_Q_cal ≤ 0.55 (vs CF11's 0.63 baseline). E3: if GO on E2, measure ΔNLL at K=128 with W_Q_cal vs W_Q raw.

**What you're NOT proposing**: Not the full affine CALFAV (CF10 risk). Not SmoothQuant activation migration (that targets W_gate/W_up input, not W_Q input, and uses output-channel migration, not input whitening). Not RMSNorm fusion (RGAS is the run-003 advancer for that).

**Track**: A. Anti-frame: outside F's anti-frame (diagonal fitting is not "calibration-fitted parameters with n_params >> n_samples" — 2048 diagonal parameters on 200 × 2048-output measurements is well-conditioned). Outside C's anti-frame (the composition with CF3 outlier statistics is the coupling). Orientation white space: C orientation never composed the per-channel activation variance (related to CF3) with the W_Q spectral concentration question (CF11). This coupling is a C-shaped idea in the F/C white space.

**CF tether**: CF11 (W_Q K=128 spectrum), CF3 (K-dependent Jaccard, outlier channels have higher activation variance). **Elegance class**: `calibration-fit` (diagonal whitening is a calibration-derived closed-form substitution). **Smallest test**: 20 min.

---

### S6 — WDKV: W_down Compressed KV Residual Injection [FREE SWING]

**Ambition**: F4-WAOS establishes that W_down's output subspace may concentrate in a rank-r_eff subspace. If this holds, each MLP layer's contribution to the residual stream lies in a low-dimensional subspace. The attention mechanism reads from the residual stream, which means the KV computation for subsequent layers is projecting into a residual stream that is structured. S6 asks: if W_down's output is concentrated in U_L (d_model × r_eff), does the KV cache gain anything by storing the residual stream increment in compressed form (U_L^T δh_L at rank r_eff) rather than the full δh_L? This is a KV-side compute reduction (computes the attention over the MLP contribution in compressed form) that only becomes available IF F4-WAOS succeeds.

**Mechanism**: Track A. If F4-WAOS gives GO (r_eff ≤ 512 for W_down output), then each MLP layer writes δh_L = W_down A_L ≈ U_L (U_L^T W_down A_L) to the residual stream. The V-cache for layer L+1's attention computation reads h_{L+1} = h_L + δh_L. If δh_L lives in the U_L subspace, the W_V^{L+1} · h_{L+1} inner product can be decomposed: W_V^{L+1} h_{L+1} = W_V^{L+1} h_L + W_V^{L+1} δh_L. The second term W_V^{L+1} U_L (U_L^T W_down A_L) can be precomputed if W_V^{L+1} U_L (a d_head × r_eff matrix) is stored at model-prep time. This is a "cross-layer product precomputation" — (W_V^{L+1})(W_down^L) cross-layer composed, only feasible if W_down^L's output lives in a low-rank subspace.

**Why our findings make this work**: This idea is explicitly conditional on F4-WAOS's GO. CF1 (W_up firing dominance → δh_L has soft-sparsity structure) motivates why W_down's output might concentrate. The KV-side compute reduction is a second-pass consequence of the weight-side finding, not a standalone assumption.

**What would have to be true**: F4-WAOS must give GO (r_eff ≤ 512). The cross-layer product W_V^{L+1} U_L must fit in memory (2048 × 512 × 2 bytes = 2 MB per layer × 28 = 56 MB total) — fits in DRAM. The decomposition must not introduce additional quantization error relative to the direct computation.

**Experiment cascade**: E0 (gated on F4-WAOS GO): compute W_V^{L+1} U_L^L for all 28 layer pairs; measure deviation from direct W_V^{L+1} h_{L+1} computation. E1 (30 min): measure KL divergence between direct V-projection and decomposed V-projection. GO if KL ≤ 0.01 per head per token. E2: compute per-token compute cost reduction for the decomposed path.

**What you're NOT proposing**: Not KV cache quantization. Not KV cache eviction. Not a standalone W_down compression (that is F4-WAOS). This is a KV-SIDE compute reduction enabled by weight-side finding.

**Track**: A. Anti-frame: outside A's anti-frame (residency payoff is conditional on F4-WAOS; the compute payoff — avoiding part of the V projection by decomposing it — is a runtime gain). Outside R's anti-frame (this is a single-measurement binary settler, gated on another idea's GO, not a standalone cascade). This is a **[FREE SWING]** — it connects to CF1 indirectly through F4-WAOS but has no direct CF tether without F4-WAOS's GO. Orientation white space: this sits in the KV-side compute gap (not KV storage, which is bottleneck-mismatched at 4K context, but KV-side compute — a gap explicitly identified in the task brief).

**[FREE SWING] rationale**: KV-side compute (not storage) is an explicitly underexplored gap noted in the brief. No run-003 advancer targets KV-side compute directly. The mechanism is grounded (cross-layer product precomputation is a standard compiler optimization — section 4's compiler prior), falsifiable (30-min measurement gated on F4-WAOS), and passes CF9 (no imported theorem) and CF10 (no parameter fitting). **Elegance class**: `algebraic-identity` — the decomposition is exact if W_down's output is in U_L exactly. **Smallest test**: gated, 30 min once F4-WAOS returns GO.

---

### S7 — CAVR: Content-Addressable Vocab Retrieval on Untied lm_head [FREE SWING]

**Ambition** (Cross-pollination from R2/S4-deferred LSEI stripped primitive): CF12 killed tied lm_head SVD — the tied embedding is full-rank, gradient-bound through both paths. But LSEI was about approximate nearest-neighbor for logit shortlisting, not weight compression. And CF12 applies only to tied configurations. On untied lm_head models (Qwen3-8B, Llama-3-70B), the Godey & Artzi gradient-bottleneck argument suggests concentration at training time — the untied spectrum may be more concentrated than the tied case. S7 proposes: build a content-addressable approximate nearest-neighbor index over the lm_head rows (vocab size × d_model) using product quantization. At inference, the final residual stream vector h_final is used to query the top-K most probable next tokens by approximate inner product, avoiding the full 151936 × 2048 matmul.

**Mechanism**: Track A. Use product quantization (PQ, standard FAISS primitive) to index the lm_head rows. At inference, h_final is the query; approximate inner product search returns top-128 token candidates in O(d × m) cost where m is the number of subspaces in PQ (typically 8–16 for d=2048). The full matmul cost is O(V × d) = 151936 × 2048 ≈ 311M MACs; PQ approximate search is O(d × m) ≈ 16K MACs — 20,000× cheaper. Quality cost: PQ approximation error on inner product. The untied lm_head spectrum (not yet measured but the Godey & Artzi gradient-bottleneck prediction applies) should show concentration that makes PQ reconstruction accurate enough for top-1 token recovery.

**Why our findings make this work**: CF12 (tied lm_head full-rank) does NOT apply to untied lm_head. The arXiv:2510.24966 logit low-rank theorem motivates that the FUNCTION-level rank of the lm_head computation is lower than the weight-level rank — PQ exploits this by approximating the inner product structure, not the weight structure. The key insight: PQ is a KV-store approximation for inner products, not a weight decomposition. CF12 killed weight decomposition; PQ is a query-time approximation that requires no weight modification.

**What would have to be true**: On Qwen3-8B (untied lm_head), PQ with m=16 subspaces must achieve top-1 token recall ≥ 0.95 (95% of the time, the exact argmax token is in the PQ top-16 candidates). This is an empirically testable claim on the untied model.

**Experiment cascade**: E1 (30 min): on Qwen3-8B, build FAISS flat index over lm_head rows; measure exact top-1 recall rate vs PQ approximation at m=8, 16, 32 subspaces. GO if top-1 recall ≥ 0.95 at m=16. E2 (30 min): measure inference throughput with PQ index vs full matmul on Qwen3-8B. E3: scale to Llama-3-70B.

**What you're NOT proposing**: Not lm_head SVD weight compression (CF12-killed for tied). Not vocabulary pruning (which changes the model). Not output sparsification (which changes token probabilities). This is a query-time ANN index that returns exact logits for a shortlist, not an approximation of the logits themselves.

**Track**: A. Anti-frame: outside A's anti-frame (residency payoff is real: at 70B, lm_head = 151936 × 8192 × 0.5 bytes ≈ 591 MB; PQ index = m × (V/k × d/m) ≈ much smaller; compute payoff is the dominant gain). Outside F's anti-frame (this imports FAISS product quantization — CF9 check: PQ requires that the vector distribution is well-approximated by independent subspace quantization. For lm_head rows, which are trained vocabulary embeddings in R^{8192}, the subspace-independence assumption may fail for nearby-meaning tokens. The CF9 precondition is "subspace quantization error is small for the query distribution." This is empirically tested in E1. PASS with caveat). **[FREE SWING]** — connects to CF12's finding (lm_head full-rank in TIED config) as motivation for the UNTIED open question, but no direct CF tether for the untied case. **Elegance class**: `no-SGD-reformulation` — builds an approximate inner product structure from existing weights without any retraining. **Smallest test**: 30 min on Qwen3-8B (untied lm_head).

---

### S8 — BAGW: Block-Addressed GGUF Weight Store (Content-Addressable Dedup)

**Ambition**: Inspired by the content-addressable constraint (section 5). Across 28 layers of Qwen3-1.7B-Base at Q4_K_M quantization, the distribution of 32-weight INT4 blocks (the native Q4_K_M superblock unit) may exhibit inter-layer repetition. Duplicate blocks appear whenever two layers happen to quantize similar weight values into the same block representation. A GGUF-level content-addressable index (BLAKE3-hashed 64-byte blocks) deduplicates repeated blocks at rest, reducing NVMe reads for duplicate blocks to a single fetch. Unique benefit over reflink: requires no filesystem support, works on Windows 11 Home.

**Mechanism**: Track B. Postprocess any existing quantized GGUF: scan all 64-byte quantized blocks; build a hash → offset index. Write a compact GGUF variant where: (1) a hash index table maps BLAKE3-hash to unique block offset; (2) each weight tensor is stored as a sequence of hash-indices, not raw blocks. At inference, the ik_llama.cpp loader resolves hash-indices → raw blocks via the index table. Duplicate blocks (same hash) are stored once. Per-token throughput improvement: if D% of blocks are duplicates, NVMe reads are reduced by D% (for the duplicate blocks, the cache hit from the first read is reused). CF9 check: BLAKE3 hash collision probability for 64-byte blocks is negligible (128-bit hash, 10^8 blocks → 10^{-20} expected collision rate). No imported theorem with preconditions. PASS.

**Why our findings make this work**: The quantization process (Q4_K_M) bins weights into INT4 values with per-superblock scales. Similar weights across layers (which share architectural patterns) may produce identical quantized blocks. The actual deduplication rate is an empirical measurement. If even 10% of blocks are duplicated, the NVMe read reduction is 10% — equivalent to a 1.11× read throughput improvement for free. At 70B NVMe streaming, 10% fewer reads directly translates to +10% tok/s.

**What would have to be true**: At least 5% of 64-byte Q4_K_M blocks must be exact hash-duplicates across the GGUF file. The hash-index table overhead must be smaller than the dedup savings. For Qwen3-1.7B-Base at Q4_K_M ≈ 1.1 GB: 1.1 GB / 64 bytes ≈ 17M blocks. Index table at 20 bytes/entry (16-byte hash + 4-byte offset) = 340 MB — comparable to the model size! The index table must be compressed or the block size must be larger. Rescope: 4 KB blocks (standard page size) → 1.1 GB / 4096 = 270K blocks; index = 270K × 20 bytes = 5.4 MB overhead. Dedup at 4 KB granularity: quantization blocks of 32 weights × 0.5 bytes = 16 bytes are too small; 4 KB spans 256 weights (4 superblocks). Inter-layer 4 KB block duplication is unlikely to be high. Use 32-KB blocks (2048 weights = 32 superblocks): 1.1 GB / 32 KB = 34K blocks; index = 34K × 20 bytes = 680 KB. Dedup at 32-KB granularity may find similar "template" blocks across layers for matrices like W_Q, W_K, W_V that share architectural patterns.

**Experiment cascade**: E1 (20 min): scan Qwen3-1.7B-Base Q4_K_M GGUF file; compute BLAKE3 hashes of all 32-KB blocks; measure duplication rate. GO if ≥ 3% of blocks are exact duplicates. E2 (30 min): implement the hash-index reader (Python prototype); verify that the deduplicated GGUF produces ΔNLL = 0 (lossless check). E3: project throughput improvement at 70B based on dedup rate.

**What you're NOT proposing**: Not NTFS reflink (Windows 11 Home limitation, R1 deferred). Not semantic deduplication (fuzzy blocks — only exact hashes, lossless). Not quantization scheme changes.

**Track**: B. Anti-frame: outside U's anti-frame (BLAKE3 hashing is a standard algorithm, no custom kernel). Outside R's anti-frame (this is a single-measurement go/no-go at E1, R's style but feeding a Track B chain). Orientation white space: U orientation never targeted hash-based block deduplication within a GGUF file as a NVMe-read-reduction mechanism. This sits in the U/A white space (substrate primitive + content-addressable constraint).

**CF tether**: None required (this is a systems mechanism with no weight-structure dependency). Closest CF: CF8 (MLP weights full-rank) and CF11 (attention weights concentrated) together suggest that different weight matrix families have different quantization patterns — which could either increase or decrease cross-layer block duplication. The empirical measurement in E1 is the load-bearing test. **Elegance class**: none — this is a substrate primitive, not an algebraic identity. **Smallest test**: 20 min.

---

## 7. Output Handoff

### Stage 4 ideas summary

| ID | One-line description | Motivating section | Fills gap |
|---|---|---|---|
| S1-QKPI | RoPE position-invariant subspace of W_Q output, hoistable as per-token CSE | Sec 4 (compiler prior) | KV-side compute; output-side compute reduction |
| S2-LQIS | Per-layer KL-injection schedule for bpw assignment, CF2-grounded | Sec 4 (control theory prior) | Training-dynamics-derived precision schedule |
| S3-QCMI | Per-channel W_Q mixed-precision from V_Q column norms (participation weight) | Sec 3 (cross-pollination: v2-S3-R004-001 stripped primitive) | Output-side + weight mixed-precision without calibration fitting |
| S4-ATCL | K=64 global W_Q+W_K packed to L3 with VirtualLock; attention projections at L3 bandwidth | Sec 5 (constraint: no RAM) | Kernel-fusion gap; L3-resident compute tier |
| S5-CQDA | Diagonal input-whitening calibration for W_Q from activation covariance; CF10-safe CALFAV rescue | Sec 3 (cross-pollination: R3/S4-deferred CALFAV) | Calibration-fit in the CF10-safe diagonal regime |
| S6-WDKV [FREE SWING] | KV-side compute reduction via cross-layer W_V × W_down^output-subspace precomputation | Sec 4 (compiler prior, CSE across layers) + F4-WAOS GO gate | KV-side compute (not storage) — explicitly flagged gap |
| S7-CAVR [FREE SWING] | FAISS product-quantization ANN index over untied lm_head rows; avoids full V×d matmul | Sec 3 (cross-pollination: LSEI primitive) | Output-side matmul gap; untied lm_head open question from CF12 |
| S8-BAGW | BLAKE3-hash content-addressable 32-KB block dedup in GGUF; NVMe read reduction | Sec 5 (constraint: content-addressable) | Substrate NVMe read optimization without filesystem dependency |

**Two ideas to weight at Stage 5**: **S6-WDKV** (if F4-WAOS gives GO, this is the highest-leverage KV-side compute idea in the pipeline — fills the explicitly flagged KV-side compute gap with a cross-layer algebraic identity, elegance class `algebraic-identity`) and **S3-QCMI** (smallest test is 10 min, no calibration needed, directly extends CF11 compression with a per-channel mixed-precision allocation that is a structural consequence of the same SVD already required — minimum additional experiment cost for a potential free residency gain).

### Orientation rotation recommendations

- **R**: KEEP. Strong round; anti-frame tightening: add "per-matrix W_V/W_O SVD without forming fused operator M_L."
- **C**: KEEP, TIGHTEN. Add anti-frame: "coupling claims with residency payoff < 1% of 70B target at confirmed scale."
- **F**: KEEP. Strongest orientation by score in run 003. Add anti-frame: "theorem imports (RFF, sketches, wavelets) on non-smooth objects without CF9 precondition audit."
- **U**: KEEP. Add anti-frame: "metadata-overhead optimizations where gain < 0.1% of per-token cost."
- **A**: KEEP, TIGHTEN. Add anti-frame: "constraint-forced mechanisms with ≤1% residency payoff at 70B scale even if mathematically elegant."

### Anti-frame additions

| Orientation | Addition | Rationale |
|---|---|---|
| R | Per-matrix W_V/W_O SVD without fused composition | SVD-LLM saturation; M_L = Σ_h W_O W_V is the live frame |
| C | Coupling claims with confirmed residency < 1% of 70B target | C4-GEMBED reject reason generalized |
| F | Theorem-family imports (wavelets, RFF, Count-Sketch) on non-smooth LLM objects without CF9 audit | R6-B kills (RFSK, CSKQ, THWR) generalized |
| U | Metadata-overhead optimizations with < 0.1% per-token gain | U5-GTBP reject reason generalized |
| A | Constraint-forced mechanisms with ≤1% residency payoff at 70B regardless of elegance | A1-GFSM reject reason generalized |
