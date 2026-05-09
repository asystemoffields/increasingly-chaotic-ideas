# Stage 4 — Skeptic Explorer — Run 005 (v2 Sonnet Ladder)

Completed: 2026-05-09. Substrate: 32 Stage 1 proposals across 5 orientations; 18 REFINE + 4 DOWNGRADE (WVWK-MLA and F1-JQKLR pre-empted by PALU+TransMLA; F5-SECP pre-empted by arXiv:2604.18085; R5-AWQKV ADJACENT to A3+PALU). 5 outright REJECT (low-score). Cross-orientation white-space search below.

---

## 1. Frame-Exhaustion Map

### Saturation-Exhausted Frames (as of 2026-05-09)

**Post-training KV projection compression via GQA stacking (PALU + TransMLA + ReCalKV).**
Any proposal of the form "reshape W_K across GQA heads, find low-rank, cache compressed KV" is published three times over at ICLR 2025, NeurIPS 2025 Spotlight, and May 2026. WVWK-MLA and F1-JQKLR both hit this. Stripped primitive: the within-layer GQA-stacked W_K r_99 measurement itself (15 min, not the compression downstream) remains structurally informative for model characterization — it feeds the VOWN product-rank diagnostic and A3's functional error bound.

**Calibration-free spectral statistics predicting SVD compression degradation on Qwen3 (arXiv:2604.18085).**
F5-SECP's core motivation — spectral entropy as a calibration-free rank proxy — is covered for Qwen3 attention layers specifically. Stable rank ρ=0.890 Pearson for attention layers. Stripped primitive: the comparison of spectral entropy H_s vs stable rank as alternative functionals is an incremental engineering question. What is NOT saturated: using spectral statistics to predict ACTIVATION compression degradation (as opposed to weight compression) — no paper covers this.

**Analytical low-rank Q-K and O-V decomposition (A3, arXiv:2505.12942).**
A3 closes the "post-training attention weight compression via functional-error minimization" frame for Q-K and O-V jointly. AWQKV (extend CF11 to all 4 matrices) is engineering-integration of A3 + PALU. Stripped primitive: the product W_V × W_O as a distinct mathematical object (not the O-V functional pair A3 treats) — VOWN's claim that the product rank < factor rank is empirically separate from A3.

### CF9-Exhausted Frames

**Shift-invariant kernel approximation (Bochner's theorem) on weight matrices** — RFSK killed this in Track B R6. The precondition (shift-invariance of the target function) is not met by weight matrices. Any weight-domain compression importing random Fourier features, Bochner-based approximations, or shift-invariant spectral methods inherits this kill. Stripped primitive: random projections with L2-preserving bounds (JL lemma) do hold for weight matrices — but JL compression is strictly worse than SVD (SVD minimizes truncation error for a given projection dimension), and SVD fails on MLP weights (CF7/CF8). JL is exhausted too.

**Count-Sketch on dense signals** — CSKQ killed this in Track B R6. L1 error bound is proportional to signal L1 norm / budget width; for dense INT4 residuals the bound is catastrophically large. Stripped primitive: sketching approaches where the signal is genuinely sparse (NOT MLP weights or dense activations). The K=0.1% static outlier channels ARE sparse (2 of 2048 nonzero) — a Count-Sketch on this sparse representation could work as an approximate identity lookup, but the savings are trivially zero (the static pinning scheme already handles this exactly and cheaply).

**Discrete Z_2 symmetry in the published layer (v2-S3-R004-001 GFQO):** general orthogonal rotation is incompatible with RMSNorm per-channel scale (the commuting group reduces to generalized permutations only). The O(d) gauge freedom is exhausted. Stripped primitive: the Z_2^d sign-flip subgroup (permutation matrices with ±1 diagonal) survives as A1-GFRS-2's alive target.

### CF10-Exhausted Frames

**Full-matrix affine calibration fitting (WDLA failure class):** fitting an affine A: ℝ^d → ℝ^{d'} on typical calibration budgets without heavy regularization. CF10 kills any proposal that fits more than ~200K parameters on 1K calibration tokens. Stripped primitive: low-rank-by-construction calibration fits (rank ≤ 64) at the WDLA failure point have only 200K parameters — well-conditioned at 1K tokens. The low-rank-by-construction variant is alive.

**Calibration-fitted KV projection at full d_model rank (WVWK-MLA, PALU's configuration):** independently killed by pre-emption (PALU), but the CF10 conditioning analysis confirms: even the correct r_K=128 version (n_params=394K, n_samples=10M) is well-conditioned. Only rank choices that push n_params > n_samples / d_output are CF10-dangerous.

---

## 2. Orientation-Saturation Diagnostics

**Orientation R (Reach):** Advanced 5 ideas; 2 DOWNGRADE (AWQKV pre-empted, WVWK-MLA pre-empted by PALU+TransMLA), 4 REFINE (RAOK-FULL, VOWN, LAYERGATE, ATTNBAND). One elegant-equivalence advancer (ATTNBAND). Convergence C1 (within-layer attention joint subspace) originated here. Orientation is productive but converged heavily on the attention compression region that is now partially saturated by PALU/A3. Anti-frame addition: add "post-training KV projection via GQA-head stacking or MLA-style decomposition" to R anti-frame list. The orientation should pivot toward training-dynamics and compute/bandwidth-fusion ideas for run 006.

**Orientation C (Composition):** Strongest run. All 6 ideas advanced; REFINE on all 6. Two convergence representatives (C2, C6 for depth-stratified scheduling; C5 for CF2 delta amplitude). Zero pre-emptions. The composition primitive (CF3 × CF11, CF2 × CF11, CF12 × CF3) is a productive seam. Anti-frame addition: add "depth-stratified scheduling derived from activation statistics" as a KNOWN frame for run 006 (ODSP and DSAF now own this seam; new composition ideas should compose different CF pairs). Recommendation: KEEP + TIGHTEN on the CF-cross-product generation strategy.

**Orientation F (First-Principles):** 4 REFINE, 2 DOWNGRADE (SECP pre-empted by 2604.18085; F1-JQKLR pre-empted by A3+PALU). One frame-novelty advancer (F2-HPGF). The SECP kill is instructive: F orientation converged on spectral-statistics as the natural first-principles object, and that frame is saturating. Anti-frame addition: add "calibration-free spectral statistics as rank/compression predictor for weight matrices" to F anti-frame list. Recommendation: TIGHTEN; push F orientation toward training-dynamics explanations (why do attention weights concentrate?) and symmetry-exploitation (A1-GFRS-2 showed the constraint-alien orientation beats F at symmetry exploitation — F should own the MECHANISM derivation, not the gauge-exploitation frame).

**Orientation U (Unconventional Substrate):** 5 REFINE, 1 REJECT (U4-SBDS, A1=0). All substrate ideas are Windows-specific I/O primitives with no published analog. Zero pre-emptions. The substrate frame is genuinely underexplored in the LLM literature. Anti-frame addition: none needed — the substrate orientation's ideas are orthogonal to the saturation region. Recommendation: KEEP.

**Orientation A (Constraint-Alien):** 4 REFINE (A1-GFRS-2 frame-novelty, A6-CNTR frame-novelty, A2-TROP, A4-APND), 3 REJECT (A3-SEED A4=1, A5-FSMI A4=1 FREE SWING, A7-NRAM pre-empted). Two of the most novel ideas in the run (GFRS-2, CNTR) originated here. The constraint-forcing mechanism is working. Anti-frame addition: add "10 MB seed Kolmogorov-complexity framing" (A3-SEED killed) and "finite-state machine Myhill-Nerode framing for LLMs" (A5-FSMI killed, CF4/CF8 contra) to A anti-frame list. Recommendation: KEEP; the constraint-alien orientation is the least saturated frame source in the ladder.

---

## 3. Cross-Pollination Opportunities

**CP1: F5-SECP downgraded (pre-empted by arXiv:2604.18085 for weight compression prediction) → transpose to ACTIVATION compression sensitivity prediction.**
The kill reason is saturation for weight-matrix spectral statistics as predictors of weight compression degradation. arXiv:2604.18085 covers weight matrices. It does NOT cover whether spectral statistics of activation distributions predict per-layer ACTIVATION quantization sensitivity (RAOK's T1/T2/T3 boundaries, or ODSP's bpw schedule). Stripping the pre-empted weight-side machinery: the primitive is "spectral summary statistic of a weight/activation matrix predicts compression degradation." In the activation domain (e.g., spectral entropy of the per-layer activation covariance), no paper covers this. Proposed cross-pollination: S3-SPECACT (see Section 6).

**CP2: R5-WVWK-MLA downgraded (M2 pre-empted by PALU+TransMLA for KV weight projection) → transpose to training-dynamics question about WHY attention weights concentrate.**
The kill reason is that PALU/TransMLA cover the post-training KV projection compression application. The stripped primitive from this kill: the GQA kv-head stacking rank measurement. Why do W_K and W_Q have concentrated spectra while W_gate/W_up do not? This is a training-dynamics question no paper addresses. The mechanism forcing spectral concentration in attention vs MLP is structurally different gradient signal. Proposed cross-pollination: S4-GRADDYN (see Section 6).

**CP3: A5-FSMI rejected (A4=1, state-space cardinality claim has no constructive argument) → the distinct-int8-quantized-states diagnostic.**
Stage 2 noted: "the distinct-int8-quantized-states measurement is a cheap (~20 min) structural diagnostic for residual-stream compressibility." CF4/CF8 (full-rank weights) makes large effective state space likely, but Stage 4's job is to look for the white space. The stripped primitive: an empirical lower bound on the residual stream's effective state-space cardinality. If measured, it directly constrains ALL lossless residual-stream compression ideas. Proposed cross-pollination: run the FSMI diagnostic not as an FSM-compression proposal but as a structural measurement primitive (tie to S3-SPECACT or provide as a cheap addon to any run-006 Stage 1 experiment).

---

## 4. Cross-Domain Seed Exploration

### Compiler Writer's View

A compiler writer looking at the inference loop asks: "which computations produce results that are not consumed before the buffer is rewritten?" This is the dead-code elimination / live-variable analysis view. Applied to transformer inference: at decode step t, the W_Q GEMV produces query q_t which is used ONLY for the score computation at the current step — it is not stored or reused. The W_K GEMV produces k_t which IS stored in the KV cache. The W_V GEMV produces v_t which IS stored. So q_t is a pure ephemeral computation.

**Elegant-equivalence prompt for compiler view:** What computational equivalence does a compiler's live-variable analysis reveal about the query projection? The query is always multiplied by all stored keys: q_t · k_s for all s ≤ t. If the query lives in a low-rank subspace (CF11: K=128), the score q_t · k_s = (P^T q_raw_t) · k_s — but the key k_s was computed with the full W_K and cached at full rank. This means the score computation can be reformulated: compute (W_Q P) ∈ R^{d_head × 128} once at load time, then at decode compute scores as (W_Q P)^T h_t · k_s_stored — the stored keys k_s do NOT need recompression. The keys and queries live in different subspaces; the compiler view reveals that the "live" computation chain is q_t → scores, and the dead-variable is q_raw in favor of (W_Q P)^T h_t. This is the ZOIA idea from a different angle — but the compiler framing suggests a fusion: fuse W_Q rank reduction into the key-score product at compute time (no storage change for k), saving W_Q GEMV bandwidth without touching the KV cache format. This produces idea S1-QFUSE.

### FPGA Designer's View

An FPGA designer sees the weight matrix W_up (2048 × 6144) as a spatial dataflow problem: data flows from the input vector through the matrix in a systolic pattern. The question is not "what rank is W_up" but "what is the arithmetic precision required for each bit of the 4-bit weight value?" In a bit-serial view, the 4-bit weight W_{ij} = w_3 · 2^3 + w_2 · 2^2 + w_1 · 2 + w_0. The sign bit (w_3) carries the most arithmetic significance but the LEAST entropy (in gauge-fixed coordinates per A1-GFRS-2, H(sign) may be < 0.8 bits). If the sign plane is nearly constant after gauge-fixing, the sign bit can be stored at lower bandwidth than the magnitude bits, and the 4-bit GEMV decomposes into a 1-bit sign GEMV (cheap: XOR + popcount) plus a 3-bit magnitude GEMV (standard). This is not the same as A1-GFRS-2's sign-entropy measurement — GFRS-2 asks if sign entropy is low; the FPGA view says if low, decompose the GEMV itself into sign-path + magnitude-path with different execution units. The elegant-equivalence: a 4-bit GEMV = 1-bit sign GEMV (AVX2 VPCMPEQB + POPCNTQ, ~1/4 cost) + 3-bit magnitude GEMV (shift-add, ~3/4 cost), saving ~1/4 of GEMV compute IFF the sign plane is precomputed and cached. This extends A1-GFRS-2's measurement into a compute decomposition. Produces idea S2-BITSER.

### Control Theorist's View

A control theorist looks at the residual stream as a dynamical system: h_{L+1} = h_L + f_L(h_L) where f_L is the transformer layer's perturbation. The CF2 finding (cos(h_L, h_{L+1}) ≈ 0.99) implies the system operates in a near-identity regime — the perturbation f_L is small relative to the state. Control theorists handle near-identity systems with state observers: estimate the state at layer L from partial observations at layer L-2 using the known (estimated) system matrix. The Kalman filter formulation: predict h_{L+1} from h_L using the layer-average dynamics, then correct with the W_Q/W_K attention output. For compression: if the "prediction step" (using a stationary approximation to f_L) is accurate enough, the correction step (the actual layer computation) needs only to represent the RESIDUAL from the prediction, not the full state. This bounds the needed precision of any skip-layer or recompute scheme.

**Elegant-equivalence prompt:** What conservation law does a control theorist see in the residual stream that would let a layer be replaced by a fixed-form operator on the conserved quantity alone? The near-conservation of h_L norm across layers (residual additivity with small perturbations) suggests that ‖h_L‖ is a near-conserved quantity (RMSNorm explicitly normalizes it at each layer). The CONSERVED QUANTITY is the RMSNorm-normalized direction of h_L. This doesn't immediately produce a compression, but it reframes the RKDB proposal: the CF2 cosine bound is a statement about the conserved direction's rotation rate, not about the absolute magnitude. For the purpose of key recomputation error bounds (RKDB), the operator-norm of W_K acts on the CHANGE in direction, not the change in magnitude. This reframing tightens RKDB's error bound in a way the original proposal didn't exploit. This cross-pollinates into S5-KALMAN below.

**Selected two professional priors for idea generation: Compiler (query GEMV fusion) and FPGA bit-serial decomposition.** These produce the cleanest elegant-equivalence mechanisms grounded in the actual Qwen3 stack.

---

## 5. Constraint-Driven Reframing

### "No continuous numbers" constraint → ternary/binary

Applied to attention scores: if attention scores S_{ij} are ternary {-1, 0, +1} after centering (a far stronger version of ATTNBAND's centered-INT4), the softmax denominator collapses to a sum of exponentials of -1, 0, +1 — exactly three values. This is not lossless at general precision, but for sparse-attention approximations it motivates an algebraic structure: the ternary score matrix has at most 3 distinct row distributions, and the softmax outputs can be precomputed for those distributions. No CF tether is needed — this is a constraint-derived mechanism. The cleanest application: at K=1 (top-1 attention, A2-TROP's limit), the score matrix is all-zero except one +1 per row. If we can quantize to 3 levels post-centering without quality loss, the attention score tensor compresses 8× vs INT4. Experiment: measure ΔNLL at ternary centered scores vs INT4 centered scores (ATTNBAND extension). Not enough novelty separation from ATTNBAND to merit a full new slot.

### "Storage is sequential-only" constraint → stream-oriented decode

This constraint is already partially explored by the U-orientation's NVMe ideas. What is NOT explored: the sequential-only constraint applied to the WEIGHT GENERATION ORDER during a matrix multiply. Standard GEMV reads W_up rows in sequence (input-vector-inner-product order), which matches sequential storage. But for the 3-tier RAOK scheme, the T2 scatter (18 dynamic channels) requires non-sequential reads to pull the dynamic-channel rows first. Under a sequential-only constraint, the T2 channels must be identified by their position in the sequential scan, not by a scatter index. This reframes RAOK's T2 dispatch as a sequential-scan filter: "as we scan W_up rows, flag rows whose outlier-channel projection (20 precomputed inner products) exceeds a threshold." This eliminates the scatter gather overhead A6-CNTR and RAOK both assume, producing idea S6-SEQFILT.

### "No persistent state" constraint → weight-from-activation derivation

The kill is near-certain (CF7/CF8 show weights are full-rank, high entropy), but the failure shape is itself a structural finding: on the Ryzen, during the NVMe stall while waiting for a layer's weights to stream in, is there enough compute budget to partially reconstruct the layer from the activation stream? For a 70B model at 3 GB/s NVMe, one W_up layer takes ~8 ms. The Ryzen at 3 GHz × 16 FP32 ops/cycle = 48 GFLOP/s × 8 ms = 384 MFLOPs available during that stall. A rank-16 reconstruction of W_up (2048 × 6144) requires 2 × 2048 × 6144 × 16 ≈ 402 MFLOPs — right at the boundary. This is not a viable production scheme (reconstruction error would be huge), but the boundary itself — NVMe-stall-budget ≈ rank-16-reconstruction-budget — is a structural finding about the compute/I/O overlap regime. Not proposed as a new idea but as a constraint-finding.

---

## 6. Generated Ideas (8 New)

### S1-QFUSE — Query Projection Fusion into Score Computation (Compiler View)

**Track A — computation-graph change.**

**Ambition.** CF11 shows that W_Q global K=128 achieves ΔNLL=+0.98 nats — the 16 heads collectively span a 128-dim subspace. At decode time, the query q_t = W_Q h_t is computed using the full 2048-dim W_Q and then multiplied against all stored keys k_s. The elegant reformulation: fuse W_Q rank reduction into the score computation algebraically so that the W_Q GEMV is replaced by two smaller GEMVs, and the score becomes q_fused_t · k_stored = (P^T W_Q h_t)_fused · k_stored where the key cache is NEVER reformatted. No KV cache change; no quality regression beyond CF11's +0.98; pure compute savings.

**Mechanism.** Factor W_Q = U_Q Σ_Q V_Q^T (128-rank truncation). At load time, precompute M = (U_Q Σ_Q)^T W_Q ∈ R^{128 × d_model} — this is the "query subspace projector fused with W_Q" stored as a single smaller matrix. At decode: instead of W_Q (2048×2048) applied to h_t (2048), apply M (128×2048) to get q̃_t (128), then the score q̃_t · k_s where k_s is still d_head-dimensional — the score is now a 128-dim inner product instead of d_head=128 per head (16 heads × 128 = 2048 effective dot products with full W_Q). Wait — the score is q_head · k_head (128 × 128 per head). With per-head query from M: q̃_t has 128 total dims (one-head equivalent), not 16×128. The trick: M_h = U_Q[:, h×8:(h+1)×8] Σ_Q[h×8:(h+1)×8] V_Q^T (8 singular vectors per head) distributes the 128-dim global subspace evenly (8 per head of 16 heads). Each head now uses an 8-dim projection instead of 128-dim; score per head is an 8-dim dot product with k_head (128-dim) — the attention score is W_Q_fused_h · h_t · k_h ≈ (V_Q_h^T h_t) · W_K^T h_s, and the rank-8 head-level query is multiplied against the full-rank key. This avoids per-head K=64 NO-GO (which required full 128-dim per head) by distributing the global 128-dim budget across heads unequally — heads with higher W_Q energy get more of the 128-dim global budget.

**Why our findings make this work.** CF11 confirms global K=128 is GO (+0.98); per-head K=64 NO-GO (+1.53). The fusion avoids the per-head failure by keeping the global-rank projector and distributing it per-head based on head energy (F2-HPGF's energy-sorted gauge is the natural complement — sort heads by energy first, allocate more global-subspace dims to high-energy heads). Sits outside all current anti-frames: not cross-layer W_Q stacking (killed), not per-head truncation (killed), not GQA kv-head stacking (pre-empted by PALU).

**What would have to be true.** (a) Head-energy allocation of 128 global dims: a low-energy head getting 2 dims vs a high-energy head getting 20 dims needs ΔNLL better than uniform 8 per head. (b) The fused M matrix (128×2048) costs the same GEMV bandwidth as a rank-128 factorization — it's the same arithmetic.

**Experiment cascade.** First: F2-HPGF's 5-min head-energy distribution measurement. If there's an energy cliff, allocate the 128 global dims proportionally to head energy. Then: measure ΔNLL with energy-allocated vs uniform allocation at total rank 128. Runtime ≤ 15 min. Go threshold: energy-allocated ΔNLL < +0.98 nats (i.e., ≤ current CF11 result at full-rank-128). Structural finding on NO-GO: head-energy-weighted subspace allocation buys nothing over uniform allocation — F2-HPGF's energy cliff is a rank distribution artifact, not a quality-relevant feature.

**What you're NOT proposing.** Not per-head truncation (NO-GO). Not cross-layer W_Q sharing (killed). Not a change to K storage or KV format.

**Elegance class: gauge-exploitation + subspace-alignment.** Self-classify: Track A. Sits outside F anti-frame (not per-head truncation), C anti-frame (not depth-stratified scheduling), A anti-frame (not RoPE-equivariance which ASCR failed), U anti-frame (not I/O scheduling), R anti-frame (not GQA kv-head stacking).

---

### S2-BITSER — Bit-Serial Sign/Magnitude Decomposed W_up GEMV (FPGA View)

**Track B — compute/bandwidth.**

**Ambition.** If A1-GFRS-2 confirms H(sign) < 0.8 bits/weight in gauge-fixed W_up, the sign plane is nearly constant — meaning in each gauge-fixed row, the signs follow a structured pattern. A bit-serial GEMV decomposes the 4-bit weight into a 1-bit sign GEMV (cheap: AVX2 VPCMPGTB comparison + popcount accumulation, ~4× cheaper than INT4 GEMV) and a 3-bit magnitude GEMV (shift-add, ~3/4 of INT4 cost). Total cost: 1/4 + 3/4 = 1× (same). BUT: if the sign bits are nearly constant after gauge-fixing, we can PRECOMPUTE the sign-GEMV output once (it's the same for all inputs approximately) and store a bias vector b_L = W_up_sign · E[x], where E[x] is the mean activation. At runtime, the sign GEMV is replaced by a table lookup into b_L (one 2048-dim vector per layer, 28 KB total) plus a correction for the per-token sign deviation. This saves the sign GEMV entirely for the ~(1 - H(sign)) fraction of neurons whose sign is near-constant.

**Mechanism.** At load time: gauge-fix W_up (flip rows so max-abs element is positive). Precompute mean-activation sign response b_L = W_up_sign_gauged · μ_x where μ_x is the mean residual-stream vector over calibration corpus (a 2048-dim vector available from R2 PDAP data). At runtime: GEMV = W_up_magnitude_gauged · x + sign_correction(x) + b_L where sign_correction(x) = (W_up_sign_gauged · (x − μ_x)) only for the fraction of neurons whose sign deviates from the mean. If H(sign) ≈ 0.5 bits/weight (expected if gauge-fixed signs cluster), ~90% of neurons have the same sign for all inputs, meaning sign_correction(x) involves only ~10% of W_up rows (scatter operation, ~10% overhead). Net GEMV cost: ~0.75× of standard INT4 GEMV (magnitude only at INT3 + 10% scatter correction). CF3 grounding: the K=0.1% static outlier channels are prime candidates for sign correction, since their activations deviate from mean most.

**Why our findings make this work.** CF1 (W_up firing dominance) makes the magnitude GEMV the quality-critical path. A1-GFRS-2's sign-entropy measurement is the gate: if H(sign) < 0.8, the sign precomputation saves real bandwidth. CF3 (K=0.1% static outliers) provides the mean-activation μ_x needed for b_L precomputation.

**What would have to be true.** H(sign) < 0.8 bits/weight (A1-GFRS-2's GO threshold). Mean-activation correction error < 0.5 nats on held-out (μ_x from calibration generalizes to eval). AVX2 VPCMPGTB + POPCNTQ for the magnitude GEMV achieves ≥ 1.2× speedup vs standard INT4 decode.

**Experiment cascade.** Gate: A1-GFRS-2's 10-min sign entropy measurement. If GO: implement b_L precomputation (Python, ~10 min); test mean-activation correction ΔNLL on WikiText-2 512 tokens. Go threshold: ΔNLL < 0.5 nats with sign precomputation. Runtime ≤ 30 min total. NO-GO: H(sign) ≈ 1.0 bit → gauge-fixing doesn't produce structured sign plane → bit-serial decomposition has no sign-precompute shortcut; falls back to standard INT4.

**What you're NOT proposing.** Not per-row sign normalization as standalone compression (A1-GFRS-2 does the entropy measurement; this proposes GEMV decomposition IF entropy is low). Not per-channel SmoothQuant-style scale absorption (different axis: sign plane vs magnitude scaling). Not training-time ternary quantization (no retraining).

**Elegance class: algebraic-identity** (4-bit GEMV = sign GEMV + magnitude GEMV, exactly). Track B. Sits outside all five anti-frames.

---

### S3-SPECACT — Spectral Entropy of Activation Covariance as Per-Layer bpw Proxy

**Track B — compression (calibration-free activation quantization schedule).**

**Ambition.** arXiv:2604.18085 covers spectral statistics predicting WEIGHT compression degradation for Qwen3. What it does NOT cover: using spectral statistics of the per-layer ACTIVATION COVARIANCE MATRIX (the Hessian proxy Σ_L = E[x_L x_L^T]) as a calibration-free predictor of per-layer activation quantization sensitivity. This is a transposition of the SECP primitive from weight space to activation space — a cross-pollination from CP1.

**Mechanism.** For each layer L, the activation covariance Σ_L ∈ R^{d×d} (computed from the PDAP calibration corpus, already collected in R2 data) has a spectral entropy H_s(Σ_L). The claim: layers where H_s(Σ_L) is low (activations are concentrated in a few directions) tolerate lower bpw quantization with smaller ΔNLL, because the quantization error is dominated by the few dominant directions and can be predicted from the spectral structure. Layers where H_s(Σ_L) is high (activations are spread across many directions) require higher bpw to maintain quality. This motivates a per-layer activation bpw schedule H_s(Σ_L) → bpw_L analogously to how CF3 spread_ℓ → bpw_L in C1-ODSP — but using the activation covariance spectrum as the input rather than the Jaccard-based outlier spread.

**Why our findings make this work.** The R2 PDAP calibration corpus already contains activation samples at all sampled layers. Computing Σ_L from those samples is a single matrix multiply per layer (negligible). arXiv:2604.18085's weight-side coverage actually OPENS this question by demonstrating the spectral-statistics-to-degradation link for weight matrices and leaving the activation side unmapped. C1-ODSP's NOVEL status (no prior uses Jaccard spread as bpw proxy) confirms the per-layer activation-statistics-driven scheduling frame is alive.

**What would have to be true.** H_s(Σ_L) is monotonically correlated with depth-stratified quantization sensitivity (Spearman ρ ≥ 0.6, analogous to ODSP's threshold). The correlation must be different from and better than the raw outlier-spread proxy from C1-ODSP — if both proxies give the same schedule, SPECACT adds nothing.

**Experiment cascade.** From the PDAP R2 dataset: compute Σ_L = (1/n) X_L^T X_L for 7 sampled layers; compute H_s(Σ_L); compare to C1-ODSP's spread_ℓ schedule and to measured INT4 quantization ΔNLL per layer. Spearman ρ between H_s(Σ_L) and ΔNLL. Runtime: ≤ 1 hour (pure computation from existing data). Go threshold: Spearman ρ ≥ 0.6 AND H_s(Σ_L) schedule outperforms spread_ℓ schedule by ≥ 0.1 nats at matched average bpw. NO-GO: both proxies predict the same schedule → H_s(Σ_L) and spread_ℓ are measuring the same structural property; no information gain from covariance-spectrum view.

**What you're NOT proposing.** Not arXiv:2604.18085 (weight-side spectral statistics). Not C1-ODSP (Jaccard-spread depth gradient). Not Hessian-based calibrated mixed-precision (no fitting, pure statistics).

**CF11 grounding.** The activation covariance Σ_L has a dual relationship to W_Q's spectrum (CF11): W_Q's left singular vectors are learned to attend to the directions that dominate Σ_L. If H_s(Σ_L) is low (activations concentrated), W_Q's compression at K=128 is doubly justified — the input is low-dimensional too. This is a composition primitive (CF11 × SPECACT) that C-orientation could develop in run 006.

**Elegance class: conserved-quantity** (activation covariance spectrum as the scheduling invariant). Track B. Sits outside F anti-frame (not weight-side spectral statistics, which F-orientation owned and lost to 2604.18085).

---

### S4-GRADDYN — Training-Dynamics Explanation for Attention vs MLP Spectral Divergence

**Track A — arch-transpose / structural finding. [FREE SWING]**

**Ambition.** CF8/CF11 reveal a sharp structural boundary: MLP weights (W_gate, W_up) are full-rank (r_99/d ≈ 1.0), while attention weights (W_Q) are concentrated (r_99/d ≈ 0.63). No published paper explains WHY this boundary exists. The hypothesis: attention weights concentrate because they are updated by gradient signal that is RANK-DEFICIENT in the head-identity dimension (the head-permutation symmetry, F2-HPGF's gauge). Under the S_16 head-permutation symmetry, the gradient of the loss with respect to W_Q rows is permutation-invariant across heads — gradient descent therefore SHOULD drive W_Q toward a representation where head identity is redundant (low rank in the head-identity dimension). MLP weights (W_gate, W_up) have no analogous symmetry — the intermediate neurons are not permutation-invariant with respect to each other (ReLU/SwiGLU non-linearity breaks the symmetry) — so gradient descent preserves full rank.

**Mechanism.** Not a compression mechanism — a structural understanding. The arch-transposition implication: if head-permutation symmetry is WHY W_Q concentrates, then any mechanism that breaks head-permutation symmetry (e.g., head-specialized adapters, head-specific gating) would increase W_Q's effective rank during retraining, DECREASING post-training compressibility. Conversely, any architecture that preserves head-permutation symmetry (MLA, which ties Q and K projections across heads via a shared latent) would maximize W_Q concentration in untied models. The deployment implication: post-training, GQA models with fewer unique KV heads (Qwen3-1.7B has 8 KV heads vs 16 Q heads) should have MORE concentrated W_K than models with Q-count = KV-count, because the GQA constraint reduces the effective gradient diversity.

**Why our findings make this work.** CF11 (W_Q concentration) + F2-HPGF (head permutation gauge) + v2-CHEAP-TEST-001 (cross-layer W_Q subspace rotation independence). The three together are consistent with: within a layer, W_Q concentrates (head symmetry → gradient reduces rank); across layers, W_Q rotates (each layer's head-assignment is independent). This is a structural narrative that EXPLAINS the CF findings rather than just reporting them.

**Smallest experiment.** Claim: in a randomly initialized (untrained) Qwen3-1.7B, W_Q has r_99/d ≈ 1.0 (random initialization). After one epoch of training on a 1GB corpus, W_Q has r_99/d < 0.90 (training concentrates the spectrum). This is a direct test of the gradient-dynamics hypothesis. Runtime: 2-4 hours (one training epoch + two SVD measurements). This experiment produces a v2-confirmed structural finding if the spectrum trajectory is measured. Go threshold: r_99/d after training < 0.90 AND training loss decreases (confirming genuine learning, not random-init collapse). NO-GO: spectrum stays at ~1.0 after training → the head-permutation-gradient hypothesis is wrong; W_Q concentration is a consequence of large-scale training, not a property of single-epoch gradient dynamics.

**What you're NOT proposing.** Not a compression mechanism. Not a new arch (MLA already exploits this). This is a MEASUREMENT that would make CF11 EXPLAINABLE, and that explanation would propagate to: which future architectures will be compressible, which won't, and why.

**CF tether.** CF11 (W_Q concentration) and F2-HPGF (head permutation gauge) are the load-bearing structural findings. This is a [FREE SWING] in the sense that the mechanism (gradient dynamics explanation) has no CF1-CF12 grounding, but it is grounded in two within-run validated findings. FREE SWING budget: 1 of 2.

**Elegance class: conserved-quantity** (head-permutation symmetry as a training-dynamics conserved structure that manifests as spectral concentration). Track A. Sits outside all orientation anti-frames — none of the orientations generate training-dynamics explanations.

---

### S5-KALMAN — Kalman-Observer KV Recompute with Tighter Residual-Stream Bound

**Track B — compression (KV cache).**

**Ambition.** C5-RKDB's error bound for KV recomputation from adjacent-layer residuals uses CF2 (cosine ≈ 0.99) and CF11 (W_K K=512 operator-norm reduction). The control theorist's cross-domain seed (Section 4) reveals a tighter bound by exploiting the DIRECTION of the residual-stream perturbation. The perturbation h_{L+1} - h_L has a specific directional structure: it is the output of MLP_L + Attention_L, which is known to lie in a low-dimensional subspace (CF11: attention output lives in the W_O column span, dimension ≤ 2048 but with concentrated spectrum). A Kalman-observer view: the "process noise" (the perturbation) has a STRUCTURED covariance, not a worst-case bounded one. If the perturbation lives in a K_A=128-dimensional subspace (W_O's column span concentrated), then the operator-norm of W_K applied to the perturbation is not ‖W_K‖_op × ‖Δ_L‖ but rather ‖W_K P_A‖_op × ‖Δ_L‖, where P_A is the projection onto W_O's column span — a tighter bound by the factor (spectral norm of W_K restricted to W_O's column span) / (spectral norm of W_K overall).

**Mechanism.** At load time: compute P_A = top-128 right singular vectors of W_O (the column span of W_O, a 2048×128 matrix). Compute W_K · P_A (the restriction of W_K to W_O's column span) and measure its spectral norm. This restricted spectral norm bounds the W_K error propagation from Δ_L more tightly than ‖W_K‖_op. The Kalman observer: instead of assuming worst-case Δ_L direction, use the KNOWN structure (Δ_L ∈ W_O column span + MLP_L output span) to tighten the recompute error bound by a factor that depends on the spectral alignment of W_K with those spans.

**Why our findings make this work.** W_O spectrum is untested but expected to be concentrated (AWQKV hypothesis; W_Q and W_K are concentrated, W_O as the output projection likely is too). CF2 provides the cosine bound. CF11 provides the W_K compressed operator norm. The Kalman view adds the STRUCTURE of the perturbation direction as the tightening ingredient.

**Smallest experiment.** Claim: ‖W_K^{L+1, K=512} · P_{W_O^L}‖_op < 0.8 × ‖W_K^{L+1, K=512}‖_op at layer 14 (the restricted spectral norm is meaningfully smaller than the unrestricted). Measurement: W_O SVD (top-128 vectors), project W_K^{512} onto those vectors, measure the restricted singular values. Runtime ≤ 10 min (two SVDs and a matrix multiply). Go threshold: restricted spectral norm / unrestricted < 0.8. NO-GO: W_O column span is not aligned with low-spectral-norm directions of W_K → Kalman tightening is negligible; the original CF2+CF11 bound is already tight.

**CF connection.** CF11 (W_K K=512 GO, ΔNLL=+0.29 nats). Prediction: if the restricted spectral norm is 60% of the unrestricted, the RKDB recompute error bound tightens by 40%, allowing every-4th-layer storage skip instead of every-2nd-layer. Track B. Sits outside C anti-frame (not depth-stratified scheduling), F anti-frame (not spectral entropy prediction for weight compression), R anti-frame (not KV projection cascade).

**Elegance class: conserved-quantity** (Δ_L lies in a structured subspace — W_O's column span — whose alignment with W_K provides a tighter bound than worst-case).

---

### S6-SEQFILT — Sequential-Scan T2 Channel Filter for RAOK (No Scatter Overhead)

**Track B — compression + runtime fusion.**

**Ambition.** RAOK-FULL's T2 tier (18 INT8 dynamic channels) requires per-token scatter-gather to load only those 18 channels' rows from W_up at runtime. The scatter-gather adds ~18 random reads per row-scan, which breaks the CPU stride-prefetcher pattern (U5-HPST). The sequential-only constraint (Section 5) reframes the T2 dispatch: instead of scatter-gathering the 18 dynamic channels, scan W_up sequentially and FLAG each row inline as T1/T2/T3 based on its outlier-channel projection inner product (a 20-element dot product, computable in 2-3 AVX2 ops per row). The decision — T2 or T3 for this row — is made AS the row is read sequentially, with no additional memory access. The flag result gates whether to process the row through INT8 or INT4 decode.

**Mechanism.** Precompute the outlier-channel projection vector p = W_outlier_channels (20 × 2048 matrix storing the current token's top-20 outlier-channel activations from CF3). For each W_up row w_i scanned sequentially: dot product ⟨w_i[outlier_indices], p⟩ (20 multiplies, AVX2 in 1 cycle). If |dot product| > threshold → T2 (INT8 decode this row). Else → T3 (INT4 decode). This replaces RAOK's precomputed T2 row indices (which require a scatter load) with an inline classification that costs 1 AVX2 instruction per row and maintains sequential access. The sequential-filter also allows the Zen 3 stride prefetcher (U5-HPST) to fire at full bandwidth, since the access pattern is perfectly sequential.

**Why our findings make this work.** CF3 (K=1% outlier channels Jaccard=0.308 — dynamic) provides the outlier channel identity. A6-CNTR's precision@20% experiment is a precondition check: if the outlier channels predict W_up firing at precision > 0.5, the sequential filter is well-motivated. U5-HPST's split-field GGUF repack (metadata first, nibbles second) enables sequential scanning of the metadata to get the 20 outlier-channel projections in one sequential pass before the nibble pass — a natural pipeline.

**What would have to be true.** A6-CNTR's precision@20% > 0.5 (outlier channels predict W_up rows). The inline 20-dot-product classification fits within the compute-hide window of the sequential NVMe/DRAM read (requires ≤ 1 ns per row classification, which 1 AVX2 instruction achieves at 3 GHz). T2 fraction (18 of 2048 channels → ~0.9% of rows classified T2) adds only 0.9% overhead to the T3 bulk INT4 scan.

**Smallest experiment.** Claim: sequential-filter T2 classification (inline 20-dot-product per W_up row) achieves the same precision@T2 as RAOK's precomputed scatter-gather T2 index, measured by comparing classification accuracy on 200 calibration tokens. Runtime: Python simulation, ≤ 30 min (reuses PDAP data + A6-CNTR's precision experiment). Go threshold: sequential-filter precision ≥ 0.90 × scatter-gather precision (within 10% of the precomputed scatter method). NO-GO: the sequential classifier mismatch is > 10% → the 20-dot-product threshold is too coarse; the T2 channel selection requires per-token outlier identity that can't be approximated inline.

**CF connection.** CF3 (K-dependent Jaccard), CF1 (W_up firing dominance). Track B. Sits outside all orientation anti-frames — no prior idea proposes inline row classification during sequential scan as a substitute for scatter-gather T2 dispatch.

**Elegance class: algebraic-identity** (the scatter-gather index is equivalent to an inline dot-product threshold test on sequentially read data — same result, different access pattern).

---

### S7-TRAINCONC — Training-Time Gradient-Variance Schedule as Post-Training Compression Signal

**Track B — compression (meta-signal for bpw scheduling).**

**Ambition.** The existing bpw scheduling proposals (C1-ODSP, C6-DSAF) use activation statistics (CF3 Jaccard spread) as the scheduling signal. A different scheduling signal: gradient variance during training. Layers where the gradient variance is high (the loss is sensitive to small weight perturbations) should use higher bpw post-training; layers where gradient variance is low (the loss is insensitive) can use lower bpw. This is not a Hessian-based approach (which requires calibration data) — it is a statistical property of the training process that can be ESTIMATED from the model's weight statistics alone, post-training, without any calibration forward pass. The estimator: gradient variance ∝ ‖W_L‖_F² / (layer width × depth scaling), derived from the neural network Gaussian process approximation (valid in the lazy-training regime). For transformer layers, ‖W_up‖_F² / (d_in × d_out) is a proxy for gradient variance (normalized weight scale). Layers with high normalized weight Frobenius norm had higher gradient variance during training.

**Mechanism.** For each layer L, compute normalized Frobenius norm proxy g_L = ‖W_up^L‖_F / √(d_in × d_out). The bpw schedule: bpw_L ∝ g_L (higher Frobenius → higher gradient variance → higher bpw needed). This is calibration-free (computable from the loaded model weights in O(d²) per layer, ~1 min total). The composition with RAOK: RAOK's T3 INT4 tier could be adjusted per layer based on g_L — layers with high g_L use T3 at 5 bpw, layers with low g_L use T3 at 3 bpw.

**Why our findings make this work.** CF8 boundary: MLP weights full-rank (r_99/d ≈ 1.0) means ‖W_up‖_F is proportional to √(d_in × d_out) uniformly — the normalized Frobenius norm g_L may be approximately constant across layers. If so, TRAINCONC would produce a flat schedule (same as uniform bpw), and the idea is dead. The most interesting case: if g_L varies significantly across layers (e.g., deep layers have higher g_L due to larger weight magnitudes), the schedule is useful. The CF3 depth-gradient finding (deep layers have wider outlier spread) is suggestive evidence that deep layers do behave differently. Whether g_L tracks outlier spread or provides an orthogonal signal is the empirical question.

**Smallest experiment.** Claim: g_L = ‖W_up^L‖_F / √(d_in × d_out) has Spearman ρ ≥ 0.4 with layer depth across all 28 layers of Qwen3-1.7B-Base, AND ρ ≥ 0.3 with INT4 quantization ΔNLL per layer (partial information about per-layer quantization sensitivity available from the track B R4 W_up rank sweep). Runtime: 5 min (Frobenius norms of all W_up layers + correlation computation from existing data). Go threshold: Spearman ρ(g_L, depth) ≥ 0.4 AND ρ(g_L, ΔNLL) ≥ 0.3. NO-GO: g_L is flat across layers → normalized Frobenius norm is approximately constant; the gradient-variance proxy adds no scheduling information beyond layer depth alone.

**CF connection.** Not directly CF1-CF12 grounded — the normalized Frobenius proxy is derived from a theoretical approximation to gradient variance. HOWEVER: if NO-GO, the finding that g_L is flat across MLP layers (despite full-rank structure, CF8) is a structural measurement that constrains any future gradient-variance-based scheduling idea. Track B. Sits outside all orientation anti-frames (calibration-free weight-norm scheduling with a training-dynamics interpretation is orthogonal to all five orientation anti-frames).

**Elegance class: none.** This is an empirical scheduling probe, not an algebraic identity.

---

### S8-NTSTORE — Non-Temporal Store for SwiGLU Intermediate Activations (Revised U4-SBDS)

**Track B — compute / memory bandwidth.**

**Ambition.** U4-SBDS (NT-store for RMSNorm scale vectors) was rejected (A1=0) because the RMSNorm scale vector IS reread within the same layer (by the GEMV output scaling), making NT stores counterproductive. Stage 2 noted the surviving structure: NT stores applied to the SwiGLU INTERMEDIATE activations (post-gate, pre-multiply, and post-multiply), which are written once then read once before being discarded. The SwiGLU intermediate a_L = silu(W_gate · h_L) ∈ R^{d_int=6144} is written to DRAM after W_gate GEMV and read once by the elementwise W_up multiply. For W_up GEMV size (2048 × 6144), the 6144-element intermediate is read in full once and never again. This matches the NT-store requirement exactly.

**Mechanism.** In ik_llama.cpp's SwiGLU kernel, replace the store of the gate-activation intermediate (silu(W_gate·h) buffer, 6144 × bf16 = 12 KB) with `_mm_stream_si128` (non-temporal store, bypasses L3 write-allocate). When the W_up GEMV reads the intermediate, it reads from DRAM via a sequential load (which IS prefetch-friendly, unlike a write-allocate eviction path). At 70B (d_int=28672 per layer, 80 layers): 80 × 28672 × 2B = 4.6 MB of gate-intermediate write-allocate pollution eliminated per token. L3 bandwidth freed: 4.6 MB × (L3-hit fraction of weight reads). At 70B DRAM-resident, weight reads are already DRAM-bound, so the freed L3 capacity enables higher weight prefetch hit rate.

**Why our findings make this work.** CF1 (W_up GEMV is load-bearing — the gate intermediate feeds the W_up elementwise multiply, making the intermediate a hot path). CF6 (Layer-1 SDZC: 36% of Layer-1 gate neurons are near-constant — their intermediates are also near-constant and could be stored in a constant-overhead register rather than the full buffer). The NT-store optimization is especially leveraged at Layer 1 where 36% of the 6144-element intermediate can be precomputed.

**Smallest experiment.** Claim: replacing gate-intermediate writeback with NT-stores (`_mm_stream_si128`) in ik_llama.cpp reduces L3 write-allocate pollution (measured as L3 write miss count per token) by ≥ 20% on DRAM-resident Qwen3-1.7B. Measurement: AMD uProf or Linux perf stat `-e cache-misses` on 50-token inference, with and without NT-store patch. Runtime: 2-hour code change + rebuild + measurement. Go threshold: ≥ 20% reduction in L3 write misses AND ≥ 1% end-to-end speedup (at 70B, where write-allocate pressure is higher, expected gain scales). NO-GO: the gate-intermediate buffer is too small (12 KB at 1.7B) to cause measurable L3 pollution; NT-store benefit materializes only at 70B scale where d_int = 28672.

**What you're NOT proposing.** Not NT-store for RMSNorm scale (U4-SBDS, rejected A1=0 — reread within layer). Not kernel vectorization. Not layout repack (U5-HPST). This is a cache-bypass primitive for the strictly write-once gate-intermediate buffer.

**CF connection.** CF6 (Layer-1 anomaly enables partial precomputation of the gate intermediate). CF1 (W_up dominance makes the intermediate's quality important). Track B. Sits outside all orientation anti-frames. The natural smallest test is a code change in ik_llama.cpp, grounding it firmly in the actual stack.

**Elegance class: none.** Microarchitectural optimization.

---

## 7. Output Handoff

### Stage 4 Ideas S1–S8

| ID | Name | One-line | Motivating Section | Track | CF Connection |
|---|---|---|---|---|---|
| S1-QFUSE | Query Projection Fusion | Distribute global W_Q rank-128 budget across heads by energy, fuse with score compute | §3 CP1 + §4 compiler | A | CF11 |
| S2-BITSER | Bit-Serial W_up GEMV | Decompose 4-bit GEMV into sign precompute + magnitude sweep if sign entropy low (A1-GFRS-2 gate) | §4 FPGA | B | CF1, CF3 |
| S3-SPECACT | Activation Covariance Spectral Entropy | H_s(Σ_L) as calibration-free per-layer activation bpw proxy (transposition of SECP to activation space) | §3 CP1 + §4 | B | CF3, CF11 |
| S4-GRADDYN | Training-Dynamics Head-Symmetry Explanation | One training-epoch spectrum measurement to confirm head-permutation gradient forces W_Q concentration | §3 CP2 + §5 | A [FREE SWING] | CF11 + F2-HPGF |
| S5-KALMAN | Kalman-Observer KV Recompute Tightening | Tighten RKDB error bound by restricting W_K spectral norm to W_O column span | §4 control theory | B | CF11 |
| S6-SEQFILT | Sequential-Scan T2 Filter for RAOK | Inline 20-dot-product row classification during sequential scan replaces scatter-gather T2 | §5 constraint | B | CF3, CF1 |
| S7-TRAINCONC | Normalized Frobenius as Gradient-Variance bpw Proxy | Weight-norm proxy for per-layer gradient variance as calibration-free bpw signal | §5 constraint | B | CF8 |
| S8-NTSTORE | NT-Store for SwiGLU Gate Intermediate | Non-temporal store for write-once gate buffer eliminates L3 write-allocate pollution | §2 U4-SBDS rescue | B | CF6, CF1 |

### Recommendations for Orientation Rotation

- **R — TIGHTEN.** Add anti-frame: "post-training KV projection via GQA kv-head stacking or MLA-style decomposition" (pre-empted by PALU+TransMLA). Pivot toward runtime fusion and training-dynamics framing.
- **C — KEEP + TIGHTEN.** Add anti-frame: "depth-stratified activation-statistics scheduling" (ODSP/DSAF now own this seam). Force composition to draw on CF pairs not yet used (e.g., CF6 × CF12, CF2 × CF3).
- **F — TIGHTEN.** Add anti-frame: "calibration-free spectral statistics predicting weight SVD compression degradation on Qwen3" (arXiv:2604.18085 now covers this). Pivot toward training-dynamics and symmetry-mechanism derivation rather than spectral-proxy measurement.
- **U — KEEP.** No anti-frame additions needed. Substrate orientation remains the least saturated frame source.
- **A — KEEP.** Add anti-frames: "10 MB seed Kolmogorov-complexity framing" (A3-SEED); "Myhill-Nerode FSM approximation of transformer state space" (A5-FSMI). Retain constraint-forcing approach.

### Anti-Frame Additions (One-Line Rationale)

- R anti-frame: "GQA kv-head stacked W_K low-rank projection without retraining" — PALU + TransMLA pre-empt (2024–2025).
- F anti-frame: "spectral entropy / stable-rank as calibration-free predictor of weight SVD compression degradation on Qwen3" — arXiv:2604.18085 (April 2026).
- C anti-frame: "Jaccard-spread depth gradient as per-layer bpw or rank schedule" — ODSP/DSAF own this primitive; run 006 C-orientation must compose different CF pairs.
- All orientations: "within-layer Q/K joint subspace measurement as a compression primitive" — A3 (arXiv:2505.12942) covers functional-error minimization for Q-K and O-V jointly; this frame is PUBLISHED for the compression application (structural measurement for model characterization remains open but is not a pipeline-level idea).

### Top 1–2 Candidates for Stage 5

**Strongest: S6-SEQFILT.** If A6-CNTR's precision@20% GO confirms, S6-SEQFILT converts RAOK from scatter-gather to sequential-scan, composing directly with U5-HPST's stride-prefetcher optimization for a compounded bandwidth win. The experiment cost is ≤ 30 min and uses existing data. The elegant-equivalence (scatter-gather = inline threshold test) is clean and auditable.

**Second: S3-SPECACT.** The transposition of SECP from weight-space to activation-space opens a novel calibration-free scheduling signal that arXiv:2604.18085 explicitly does NOT cover. The experiment reuses the R2 PDAP data (already collected) and costs ≤ 1 hour. If H_s(Σ_L) and outlier-spread give different schedules, SPECACT wins an independent scheduling primitive.

**For Stage 5 alt: S2-BITSER** gates on A1-GFRS-2's sign-entropy result (already top-3 in Stage 3). If H(sign) < 0.8 bits/weight confirms, S2-BITSER's bit-decomposed GEMV is a direct compute win that stacks with RAOK's quantization scheme.
