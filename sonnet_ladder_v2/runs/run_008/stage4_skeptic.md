# Stage 4 — Skeptic Explorer — Run 008

**Date**: 2026-05-09
**Substrate**: 5 orientations (A, C, F, R, U), 25 Stage 1 proposals, 14 advancers, 12 REFINE / 1 DOWNGRADE (A-INTG, pre-empted by I-LLM) / 1 KILL (F4-SVAL-CONSERVED, pre-empted by SINQ)
**Top 3 from Stage 3**: C-UFDM, F3-WO-HEADBLOCK-RANK, R8-WVWO-FOLD

---

## 1. Frame-Exhaustion Map

### 1.1 Saturation-exhausted frames

**Cross-layer W_Q stacked basis** — dense prior art (v2-CHEAP-TEST-001 measured it directly; class-killed empirically). Stripped primitive: per-layer W_Q spectrum concentration (CF11, alive).

**γ-absorption / integer RMSNorm** — I-LLM (arXiv:2405.17849, May 2024) publishes the identical DI-Normalization mechanism. Stripped primitive: AVX2 INT32 isqrt kernel as a free CPU performance primitive for ik_llama.cpp (not a research proposal, an engineering add-on).

**Sinkhorn gauge for weight balancing** — SINQ (arXiv:2509.22944, Sep 2025) covers the Sinkhorn-Knopp normalization for LLM quantization on exactly the Qwen3 family. Stripped primitive after killing F4: **Sinkhorn balance applied to SVD-rank prediction rather than quantization scale** is explicitly not in SINQ. SINQ predicts quantization quality from balance; predicting the safe SVD truncation rank K from the Sinkhorn-balanced spectrum — without any quantization — is a structurally distinct application. This stripped primitive is a legitimate cross-pollination target (see Section 3).

**Semantic-similarity layer caching** — LLMCache (arXiv:2512.16843, Dec 2025) covers learned-encoder-based cache with 3.1× speedup. Stripped primitive from A-CLOG adjacency: exact-hash without a learned encoder (BLAKE3 on INT8), within-run collision rate as a structural measurement. Not killed — REFINE — but the learned-encoder branch of the frame is saturated.

**Arbitrary O(d) residual stream rotation** — killed by v2-S3-R004-001 (RMSNorm breaks O(d)). Stripped primitive: permutation symmetry S_{n_heads} and sign-flip Z_2 — both alive, both used in run_008 (A-PERM partially, A-SYMX).

### 1.2 CF9-exhausted frames

**Shift-invariant kernel approximation on weight matrices** — Bochner's theorem requires shift-invariant kernels; weight matrices are not kernels. Killed by RFSK in run_007 Track B. Stripped primitive: random projection (JL) on activations, where the target space is vector-valued and the precondition is just dimensionality reduction, not kernel approximation — this is alive.

**Count-Sketch on dense residuals** — CSKQ killed (Run 7). L1 error bound 5470× signal. Stripped primitive from CSKQ: symmetric quantization on residuals — dead in compression-quant (well-known), but as a measurement of residual-stream quantization error distribution across layers, it is alive as an activation-quantization supporting primitive (relevant to RAOK-70B T2 tier design).

**Softmax shift-invariance + RoPE** — precondition fails because RoPE rotations are position-dependent (logit δq_c^T k_t varies per key position t, not a constant scalar). Killed by v2-S3-R009-001. Stripped primitive: static outlier channels (CF3 K=0.1%) as high-energy W_Q input columns → precision allocation (pass to C-WKOL and RAOK-70B, already in pipeline).

### 1.3 CF10-exhausted frames

**Full-affine calibration-fit cross-scale correction (WDLA class)** — CF10 killed WDLA empirically (R^2_eval = -118000 at 1000 calibration tokens for 2.1M-parameter affine map). Stripped primitive: **low-rank-by-construction affine correction at r ≤ 64** is explicitly alive per WDLA post-mortem. This is a legitimate cross-pollination target for Stage 4 (see Section 3).

**Calibration-fitted per-layer W_down projection at K ≈ d_int/6** — F5-WDOWN-SPECTRUM was rejected with A4=1 because the activation-covariance alignment premise is ungrounded. Not a CF10 kill in isolation, but the calibration-fit concern applies if K is large. Stripped primitive: W_down spectrum measurement (20 min, not killed). Alive as a measurement.

---

## 2. Orientation-Saturation Diagnostics

**Orientation A (Constraint-Alien)**: Advanced 2 to Stage 3 (A-INTG, A-CLOG). A-INTG DOWNGRADEd by I-LLM pre-emption. A-CLOG REFINE (partial adjacency to LLMCache but survives). No convergence cluster participation. **Anti-frame addition**: add "integer-only normalization as a research target" — engineering integration of I-LLM DI-Normalization is already published; future A proposals involving integer normalization must offer structural value beyond CPU implementation. Recommendation: **TIGHTEN** — the integer-constraint frame produced one pre-empted idea; the append-only / content-addressable constraint produced a survivor. Focus next round's A generation on the content-addressable and distributed-state constraint families, not integer arithmetic (saturated by I-LLM family).

**Orientation C (Composition)**: Advanced 3 to Stage 3 with one of the top-3 (C-UFDM, score 13). C-WKOL and C-RKSC also REFINE. C-SPECQ advanced as FREE SWING. C-LDOC rejected at Stage 2 but its measurement flows into C-UFDM as a supporting rung. Strong round; convergence participation in C2. **Anti-frame addition**: add "per-layer W_Q rank-as-Jaccard-function as a standalone proposal" — the slope measurement (C-LDOC premise) is not novel enough at Stage 2; it is a supporting measurement, not a ranked compression primitive. Recommendation: **KEEP**.

**Orientation F (First-Principles)**: Advanced 4 to Stage 3 (F1-WVWO-SPECTRUM convergence rep, F2-UNTIED-LMHEAD, F3-WO-HEADBLOCK-RANK, F4-SVAL-CONSERVED). F4 killed by SINQ. F3 in top-3. Strong orientation; provides the load-bearing C1 cluster settled by F3. **Anti-frame addition**: add "Sinkhorn balance as a compression-gauge predictor for LLM quantization" — SINQ owns this; any future F proposal invoking Sinkhorn must differentiate from SINQ explicitly (the SVD-rank application is the narrow surviving edge). Recommendation: **KEEP**.

**Orientation R (Reach)**: Advanced 3 to Stage 3, including R8-WVWO-FOLD (top-3) and R8-RAOK-70B (strong). R8-GUCB-SHARED and R8-DELTAROPE rejected. Convergence C1 participation (R8-WVOS-SPEC). **Anti-frame addition**: add "KV cache compression at ≤4K context" as low-priority (C-WKOL + RAOK-70B cover the KV sidecar efficiently; DELTAROPE was the KV-RoPE angle and scored too low). The reach frame should focus on 70B deployment arithmetic — RAOK-70B and WVWO-FOLD are the natural survivors. Recommendation: **KEEP**.

**Orientation U (Unconventional Substrate)**: Advanced 3 to Stage 3 (U-NVME-QD, U-MMAPF, U-PGWT). All REFINE. No convergence with ML-weight ideas (correct for this orientation). U-WSTB, U-ZSTD-DICT, U-FSCTL-PRIO all rejected but with cheap surviving measurements. The orientation is performing well: three genuine substrate-recognition proposals in the REFINE pool. **Anti-frame addition**: add "zstd-dict compression of bf16 weight tiles without prior quantization" as low-priority — the compressibility pre-check (1.1× ratio expected) argues against this branch; direct the U orientation toward OS-scheduling and prefetch-engine primitives rather than codec-based compression. Recommendation: **KEEP**.

---

## 3. Cross-Pollination Opportunities

### CP-1: F4-SVAL-CONSERVED stripped primitive → SVD-rank prediction via SINQ balance (not quantization scale)

SINQ kills F4's "Sinkhorn gauge for quantization-scale optimization" but does NOT kill "Sinkhorn gauge for predicting safe SVD truncation rank K." SINQ optimizes the weight scale for quantization rounding error; the SVD-rank prediction question asks: does Sinkhorn-balancing W_Q reduce the r_99 of the resulting matrix (tighter safe K for compression)? SINQ does not measure or claim this. The stripped primitive is: **apply SINQ's published Sinkhorn balance (not re-derive it) and measure whether r_99/d drops**, grounding the SVD-rank interpretation as an extension of published work. This is a Track B measurement with a 10-min test, no imported theorem, no calibration fit, and a clean CF9 precondition (W_Q W_Q^T > 0, verified per CF11).

Cross-pollination target: F orientation's first-principles frame. The stripped primitive belongs in the F orientation (Sinkhorn balance is a known published tool; asking what it does to the SVD spectrum rather than the quantization scale is a F-style "what does this reveal about the weight structure" question). Expanded into S5 below.

### CP-2: A-INTG stripped primitive → AVX2 INT32 isqrt as a free I-LLM integration for ik_llama.cpp

I-LLM kills A-INTG as a research proposal. The surviving primitive is the CPU-specific AVX2 INT32 implementation detail. Cross-pollination: the U orientation (Unconventional Substrate) naturally handles "implement a published mechanism in the specific substrate." A Stage 4 idea that implements I-LLM's DI-Normalization specifically for ik_llama.cpp on the Ryzen 5 7530U AVX2 path, measuring actual throughput improvement in the CPU inference context, is an engineering primitive that no published paper measures on this hardware. This does not warrant a Stage 5 research slot, but it is a free optimization that Stage 4 can note as an add-on to NQDL/MMAPF implementation work.

### CP-3: WDLA stripped primitive → low-rank-by-construction affine correction at r ≤ 64

CF10 killed the full-affine WDLA (2.1M parameters, 1000 calibration tokens, underdetermined). The surviving primitive is the **force-low-rank-by-construction variant**: fit A_L of rank r ≤ 64 (parameter count ≈ 2048×64 + 1024×64 ≈ 200K — well-conditioned at 1K calibration tokens). This is explicitly named in WDLA's post-mortem as the rescue path. No orientation in run_008 has picked this up. The A orientation (constraint-alien) could frame it under the "no continuous numbers" or "10 MB seed reconstruction" constraints; the C orientation (composition) could frame it as CF10-conditioned affine surgery. Expanding into S8 below.

---

## 4. Cross-Domain Seed Exploration

### 4.1 Compiler writer's view: Profile-Guided Optimization applied to weight residency

A compiler writer sees the inference compute graph as a static DAG with profiling data (CF1 firing-rank dominance, CF3 outlier-channel frequencies, CF6 gate variance by layer). The elegant-equivalence prompt: **what algebraic identity does profile-guided optimization reveal in the transformer forward pass that would let some weight matrix be replaced by a fixed-form operator compiled from the profiling data?**

The CF6 Layer-1 gate-variance finding (36% near-constant gate neurons) IS this: the foldable neurons are the "dead code" that PGO eliminates at compile time. C-RKSC is the instance. But the compiler framing extends further: CF3 shows that K=0.1% channels are static across tokens (Jaccard=0.718). A PGO-like compiler would fold these static channels into a pre-computed bias vector at model-load time, eliminating 2 channels from every W_Q / W_K / W_V / W_up / W_gate computation at runtime. This is a static-channel fusion that no orientation in run_008 has targeted for ALL weight classes simultaneously. The load-bearing question: if 2 channels are static across all tokens, can all weight matrices be restructured so those 2 channels produce a constant additive offset computed once per prompt, collapsing 2/2048 = 0.1% of each GEMV into an initialization step? At 70B, 0.1% of 35 GB × 11.5 GB/s DRAM = 0.035 GB × 11.5 = 400 MB/s saved per token — marginal, but the structural finding (can you provably fold all static-channel contributions at once?) is a genuine arch-transpose contribution. Expanded into S1 below.

### 4.2 Database engineer's view: Zone maps and late materialization

A database engineer sees weight tensors as column stores and activation vectors as row queries. The elegant-equivalence prompt: **what algebraic identity does the database engineer's view of W_up · h reveal — is the matrix-vector product expressible as a join over a pre-computed index that makes it cheaper without changing it?**

Zone maps in columnar databases index the min/max of each column chunk to skip chunks that cannot satisfy a predicate. Applied to the GEMV W_up · h: if the calibration distribution shows that rows i..j of W_up have near-zero dot product with h for most tokens (low energy rows), a zone-map index precomputed on W_up's row norms would let the GEMV skip those rows. This is activation-sparsity skipping, already explored and killed (SABLE-v1, SABLE-v2 from pre-pipeline kills). But the database framing reveals a different object: **late materialization of the post-SwiGLU output**. In columnar databases, late materialization defers column-store decompression to after predicate evaluation. In GEMV terms: compute the gate-side silu(W_gate · h) first (cheaper, per CF1 it is less informative but smaller in magnitude-variance), use its sign as a bitmask to skip W_up · h for the neurons where silu(W_gate · h) ≈ 0, and only materialize W_up · h for the neurons that pass the gate. This is a token-side version of SABLE but using the gate output as a predicate rather than a historical frequency table. The precondition: silu(W_gate · h) ≈ 0 for a meaningful fraction of neurons per token. CF6 says 1.5% globally near-constant (insufficient), but the token-wise distribution of small silu outputs (not near-constant across tokens, but small for THIS token) is unmeasured. Expanded into S2 below.

### 4.3 Control theorist's view: State observer for activation prediction

A control theorist sees the residual stream as a state vector evolving under a sequence of operators (transformer layers). The elegant-equivalence prompt: **what conservation law does a control theorist see in the residual stream that would let a layer be replaced by a fixed-form operator on the conserved quantity?**

CF2 (cos(h_L, h_{L+1}) ≈ 0.99) is the near-conservation law: the residual stream changes slowly, meaning the layer operator is close to the identity. A Kalman filter would predict h_{L+1} ≈ h_L + K × (measured_innovation), where K is the Kalman gain. For the transformer: the "innovation" is the MLP+attention output Δh_L; the "measurement" is h_L itself; the "state" is h_L. If Δh_L is predictable from h_L without loading W_gate/W_up/W_down (the expensive DRAM reads), a lightweight linear prediction of Δh_L reduces the number of full GEMV passes. This is adjacent to the killed WDLA (cross-scale affine map), but the control-theorist framing points at a different object: an **intra-scale layer-skip oracle** (predict whether Δh_L is small enough that the layer can be skipped with low NLL cost). The structural condition: if ‖Δh_L‖₂ / ‖h_L‖₂ < threshold for a given token at a given layer (CF2 bounds this ratio globally at 0.141), the layer contribution is below the quantization noise floor. The oracle needed: predict ‖Δh_L‖₂ from h_L alone, without loading MLP weights. If the prediction is cheap (a dot product of h_L with a pre-computed vector v_L calibrated as the mean Δh_L direction), and the skip threshold is ≈ 0.05, then 5-10% of (token, layer) pairs may be skippable — relevant at 70B NVMe-offload. This is the adaptive-computation / early-exit family, adjacent to CVEEKS (run_008 Track A R3 S4 deferred). The control-framing novelty is using CF2's ‖Δh‖ bound as the conservation law that motivates the oracle's precision requirement. Expanded into S3 below.

---

## 5. Constraint-Driven Reframing

### 5.1 No continuous numbers (active)

The constraint forces binary / ternary / lookup mechanisms. Run_008's kill list excludes I-LLM (saturated). What is NOT saturated: **weight-matrix entries expressed as a sum of lookup-table outputs indexed by activation quantile**. This is not integer arithmetic applied to existing floats — it is a fundamentally lookup-table-organized matrix multiply where the table entries are pre-computed for each quantile bucket of h. At d_model=2048 with 16 activation quantile buckets: the table for W_Q has 16 × 2048 = 32K entries (64 KB at FP16). The GEMV becomes: for each row of W_Q, look up the table entry for the quantile of h[column], accumulate. This is an FPGA-style lookup-table matmul, and its precondition (h must be quantile-bucketed, W_Q rows must be representable by 16 centroid values) is a calibration-conditioned claim. CF9 check: no imported theorem; the quantile-bucket GEMV is a direct computation. CF10: the 16 centroids per row have 16 × 2048 = 32K parameters, fit on 512-token calibration corpus (512 × 2048 = 1M values >> 32K parameters). Well-conditioned. Expanded into S4 below.

### 5.2 Content-addressable (partially active via A-CLOG)

A-CLOG is REFINE, so this constraint is live. The stripped primitive from A-CLOG that has NOT been exploited: **content-addressable weight deduplication at the page level**. If two weight pages (4 KB each) have the same quantized content (e.g., after INT4 quantization, two pages produce identical nibble sequences), the OS can alias them to the same physical page. This is the NTFS-Reflink idea (killed for Windows Home), BUT at the logical-content level without reflinks: detect identical INT4-quantized weight pages in the GGUF file at load time, build a pointer table, and at inference time serve the same physical page for all aliases. On Windows Home, this requires a custom user-space page-alias layer (a read-only mmap substitution where aliased offsets share a single physical buffer). Different mechanism from PGWT (which punches zeros) and from reflinks (which require NTFS Home feature). Expanded into S6 below.

### 5.3 No floating point at inference (partially covered)

Integer arithmetic is saturated (I-LLM). What is NOT saturated: **bit-serial arithmetic**. An FPGA designer would implement matrix multiply as a series of 1-bit shifts and POPCNT accumulations. On AVX2 with VPAND / VPOPCNT, bit-serial W_Q × h can process 256 bits per instruction. The structural condition: h must be binarized or ternarized; W_Q must be stored as packed bits. This is BitNet territory (pre-empted by BitNet b1.58 for training-time quantization) BUT the post-training binarization of W_Q specifically — given CF11's concentrated spectrum — has not been killed. CF11 says W_Q at K=128 incurs only +0.98 nats ΔNLL; if binarizing W_Q on top of K=128 truncation adds less than 0.5 nats, the total attention weight bitwidth for W_Q is effectively 1 bit. Expanded into S7 below.

### 5.4 Storage sequential-only (active for NVMe-offload scenario)

The NVMe constraint forces stream-oriented decode. U orientation's ideas (NQDL, MMAPF) already address prefetch scheduling. The unexplored angle: **GGUF tensor layout re-packing for optimal NVMe sequential throughput**. Current GGUF stores tensors in type-order (all Q4_0 tensors together). For sequential layer-by-layer inference, optimal layout packs all of layer L's tensors together, so a single sequential NVMe read of one layer retrieves all its weight pages without seeks. INPOLT (Track A R3 S4, deferred) proposes this, but was deferred for the NVMe-bottleneck precondition check. Run_008's NQDL/MMAPF measure the NVMe throughput gap; if GO, INPOLT's GGUF re-packing becomes the structural optimization. Stage 4 should note this dependency rather than generate a new independent idea — the dependency structure means INPOLT should follow NQDL/MMAPF GO, not be re-proposed independently. Not generating a new idea for this constraint; noting the cascade dependency instead.

---

## 6. Generated Ideas (8)

### S1 — Static-Channel Universal GEMV Fold [Track B / F orientation's white space]

**Elegance class**: `algebraic-identity`
**Track**: B
**CF anchor**: CF3 (K=0.1% static outlier channels, Jaccard=0.718), CF11 (W_Q subspace)
**Outside anti-frames of**: F orientation (does not use Sinkhorn/gauge, does not use cross-layer stacking, does not use per-head W_Q truncation); A orientation (does not rotate residual stream); C orientation (does not require subspace overlap measurement first)

**Ambition**: Fold the 2 static outlier channels' contribution out of ALL weight GEMVs as a constant offset computed once per prompt, reducing all subsequent per-token GEMVs from d=2048 to d=2046 input columns.

**Mechanism**: CF3 K=0.1% channels (2 of 2048) have Jaccard=0.718 — 71.8% of consecutive token pairs share the same top-2 channels. These 2 channels are effectively static per prompt prefix. Define: for each weight matrix W ∈ R^{d_out × d_model}, extract the 2 static-channel columns W[:,O] ∈ R^{d_out × 2}. Pre-compute the static contribution bias_L = W[:,O] · h[O] once when h[O] first converges (after the prompt prefix stabilizes). At inference, GEMV becomes: W[:,~O] · h[~O] + bias_L (cached). The bias_L is d_out-dimensional (e.g., 2048 FP16 = 4 KB per matrix), computed once per prompt prefix per matrix. For 28 layers × ~8 matrices per layer: 28 × 8 × 4 KB = 896 KB bias cache — trivial. Each GEMV shrinks: 2048 → 2046 input columns (0.1% cheaper). At 70B: not the savings mechanism; the savings mechanism is **eliminating a GEMV entirely for layers where the static-channel contribution dominates h's output**. If at some layers the static-channel bias_L > 50% of the full GEMV output for typical tokens, the dynamic portion (W[:,~O] · h[~O]) can be quantized more aggressively at the same total output quality (because the dominant contribution is known exactly from the bias).

**Why our findings make this work**: CF3's Jaccard=0.718 at K=0.1% means 2 channels are genuinely nearly static. The universal fold does not require knowing which 2 channels in advance (they are measured from CF3 PDAP data and fixed). No calibration fit (the static-channel identity is a measurement, not a regression). No CF10 risk.

**What would have to be true**: At ≥ 14/28 layers, the 2-column contribution W[:,O] · h[O] contributes ≥ 5% of the total GEMV output energy. If < 1%, the bias-caching is free but useless. The activation-magnitude argument (static channels have high magnitude per CF3) suggests this is plausible.

**Experiment cascade**: (1) Load W_up[14], extract columns O (2 static outlier channels from CF3 PDAP). For 200 calibration prompts, compute ‖W_up[:,O] · h[O]‖₂ / ‖W_up · h‖₂ per token. Runtime: ~15 min. GO: mean ratio ≥ 0.05 across tokens and layers. (2) If GO: implement fold, measure ΔNLL on WikiText-2 held-out. Runtime: ~30 min.

**What you're NOT proposing**: Not proposing to remove the 2 columns from W (they are still used in the fold-initialization step). Not proposing to approximate h[O] — it is read exactly.

**Smallest test**: ≤ 45 min. Class-level: whether K=0.1% static channels carry ≥ 5% of GEMV output energy.

---

### S2 — SwiGLU Gate-Predicate Late Materialization [Track B / DB-engineer frame]

**Elegance class**: `conserved-quantity`
**Track**: B
**CF anchor**: CF1 (W_up firing-rank dominance), CF6 (gate near-constancy Layer 1 at 36%)
**Outside anti-frames of**: All orientations. No orientation in run_008 targeted a gate-predicate skip on per-token W_up materialization. F orientation's anti-frame includes "polynomial activation substitution" (CSMF-kill) but NOT "predicate-gated W_up load."

**Ambition**: Skip the W_up GEMV for neurons where silu(W_gate · h) is provably small this token, skipping those rows' NVMe reads entirely at 70B offload.

**Mechanism**: At each transformer layer, compute silu(W_gate · h) first (W_gate is loaded anyway). For neurons where |silu(W_gate · h)| < ε, the contribution of neuron i to the MLP output is W_down[:,i] × silu(W_gate[i,:] · h) × (W_up[i,:] · h). Since silu(W_gate · h)[i] ≈ 0, this contribution is negligible regardless of W_up[i,:] · h. Therefore W_up rows corresponding to these neurons need not be loaded. At 70B NVMe-offload: W_up rows not loaded = rows not read from NVMe. If 10% of neurons have |silu(W_gate · h)| < ε at a given layer/token, 10% of W_up (per layer at 70B: ~24 MB row-chunk) is skipped. At NVMe 3 GB/s, skipping 2.4 MB/layer = 0.8 ms saved per layer per token at 10% skip rate. Across 80 layers: 64 ms saved per token — not decisive alone, but composable with RAOK-70B.

CF1 says W_up dominates firing-rank, which means W_up rows ARE load-bearing. But the predicate skip only applies when the gate pre-screen filters a neuron OUT — for those neurons, the W_up load is genuinely unnecessary. CF1 does not contradict this: CF1 says W_up dominates the neurons that DO fire. Neurons that are pre-filtered by the gate are irrelevant to both firing rank and quality.

CF10 check: no calibration fit. Gate output computed at runtime. CF9 check: no imported theorem; the gate-predicate skip is a direct consequence of SwiGLU's multiplicative structure.

**What would have to be true**: At least 5-10% of neurons per layer per token have |silu(W_gate · h)| < ε (say ε = 0.01). This is the empirically unmeasured skip rate. CF6 says 1.5% are near-constant across ALL tokens, but many more may be small for a specific token at a given layer.

**Experiment cascade**: (1) Collect silu(W_gate · h) values for 200 calibration prompts, all 28 layers; compute fraction of neurons with |silu| < 0.01, 0.05, 0.1 per token-layer pair. Runtime: ~30 min. GO: mean skip-fraction ≥ 0.05 across token-layer pairs at ε = 0.05. (2) If GO: implement row-skip in forward pass, measure ΔNLL impact at chosen ε.

**Smallest test**: ≤ 30 min. One histogram computation on existing forward pass infrastructure.

**What you're NOT proposing**: Not approximating the gate (it is computed exactly). Not predicting skip from prior tokens (purely reactive to this token's gate output). Not a structured-sparsity scheme (the skip pattern is token-dynamic per CF3 — no static mask).

---

### S3 — Intra-Scale Layer-Skip Oracle via CF2 ‖Δh‖ Bound [Track A / control-theory frame]

**Elegance class**: `conserved-quantity`
**Track**: A
**CF anchor**: CF2 (cos ≈ 0.99, ‖Δh_L‖₂ ≈ 0.141 × ‖h_L‖₂), CF8 (MLP full-rank)
**Outside anti-frames of**: A orientation (no O(d) rotation), C orientation (no cross-layer subspace sharing), F orientation (no Sinkhorn/gauge), R orientation (no W_VO product in this proposal)

**Ambition**: Predict ‖Δh_L‖₂ from h_L alone (without loading MLP weights) to identify per-token-per-layer pairs where the layer contribution is below the INT4 quantization noise floor, enabling those layers to be skipped at 70B NVMe-offload.

**Mechanism**: CF2 establishes that ‖Δh_L‖₂ ≤ 0.141 × ‖h_L‖₂ globally. The per-token-per-layer ‖Δh_L‖₂ varies: some tokens at some layers change the residual more, others less. The oracle: a learned linear predictor f_L(h_L) = v_L^T · h_L where v_L ∈ R^d is a calibration-derived weight vector predicting ‖Δh_L‖₂. If f_L(h_L) < threshold, skip layer L for this token (use h_L as the "output" without computing Δh_L). The predictor is lightweight (one dot product, negligible FLOP vs GEMV), calibration-derived (v_L fit on 1K calibration prompts: 2048 parameters, 1K × 1 output values, well-conditioned per CF10).

CF10 check: n_params = 2048, n_independent_samples = 1000 × 1 = 1000. Ratio n_params/n_samples = 2.048 — borderline but per CF10 guidance, strong ridge (1e-1) makes this well-conditioned at 2048 parameters (it is a ridge regression on h_L ∈ R^{2048} to predict one scalar). Using ridge regularization properly, this is a manageable fit. Alternative: use ‖h_L‖₂ itself as a coarse proxy (no fit required — just a threshold on residual-stream norm).

The arch-transposition: the layer IS executed but the weights are not loaded from NVMe. The oracle fires before the NVMe read is issued. If oracle fires, skip the NVMe read for that layer. At 70B, one layer = ~125 MB NVMe read at 3 GB/s = 41 ms. Skipping 5% of layers = 5% of NVMe reads = 0.05 × 80 layers × 41 ms = 164 ms saved per token → +6.7% tok/s at 1.0 tok/s baseline.

**What would have to be true**: The calibration-derived linear predictor achieves R² > 0.5 on held-out ‖Δh_L‖₂ across diverse tokens. If ‖Δh_L‖₂ is determined purely by h_L's norm (not its direction), the oracle reduces to a norm threshold — a zero-parameter, zero-calibration mechanism.

**Experiment cascade**: (1) Collect (h_L, ‖Δh_L‖₂) pairs from 200 calibration prompts across all 28 layers. Fit v_L by ridge regression. Measure held-out R². Runtime: ~35 min. (2) If R² > 0.3: implement oracle in forward pass, measure ΔNLL as a function of skip threshold.

**Smallest test**: ≤ 35 min. GO: held-out R² ≥ 0.3 at any layer. NO-GO: ‖Δh_L‖₂ is not predictable from h_L — the per-layer contribution is randomly distributed, making adaptive skipping no better than uniform random skipping.

**What you're NOT proposing**: Not proposing layer removal (skipped layers use h_L unchanged, not a trained smaller model). Not proposing cross-layer prediction (the oracle only uses the current h_L).

---

### S4 — Quantile-Bucket Lookup-Table GEMV for W_Q [Track B / FPGA-engineer frame / bit-serial]

**Elegance class**: `no-SGD-reformulation`
**Track**: B
**CF anchor**: CF11 (W_Q r_99/d ≈ 0.63, head-shared K=128 subspace), CF3 (outlier channel structure)
**Outside anti-frames of**: F orientation (no Sinkhorn), A orientation (no rotation), C orientation (no cross-layer sharing). Explicitly different from INT4 quantization (which uses global absmax) and from per-head W_Q truncation (killed at CF11 per-head K=64).

**Ambition**: Replace W_Q GEMV with a lookup-table accumulation indexed by h's activation quantile, targeting the pure-integer / lookup-table constraint.

**Mechanism**: Partition the residual stream's 2048 dimensions into Q quantile buckets (Q = 16) based on the empirical activation distribution from calibration. Pre-compute for each weight matrix W_Q[l] and each (row, quantile_bucket): W_Q[l,row,q] = mean column weight over the d/Q dimensions in bucket q. At inference: quantize h into Q-bucket indices (one integer per bucket, 2048 / 16 = 128 dimensions per bucket, encoded as the leading principal component within each bucket). GEMV: sum over Q buckets of W_Q[l,row,q] × scalar_projection_of_h_into_bucket_q. This reduces the 2048-dimensional GEMV to a 16-element lookup accumulation per row.

CF10 check: table has Q × d_out = 16 × 2048 = 32K entries per weight matrix. Calibration: 1K tokens × 2048 output dims = 2M values >> 32K parameters. Well-conditioned with standard ridge.

CF9 check: the quantile-bucket decomposition is a calibration-conditioned approximation of the GEMV; no imported theorem beyond the law of total expectation. Precondition: the within-bucket variation of column weights is smaller than the between-bucket variation. This is the load-bearing empirical question.

**What would have to be true**: The between-quantile column weight variation (mean W_Q column value differs across quantile buckets) is ≥ 5× the within-bucket variation for a majority of rows. This is a structural claim about W_Q's column organization relative to the activation distribution.

**Experiment cascade**: (1) Load W_Q[14], H_calib from PDAP. Compute quantile bucket assignments of h columns. For each row of W_Q, compute between-bucket vs within-bucket column weight variance. Runtime: ~15 min. GO: mean between/within ratio ≥ 5 across rows at ≥ 10/28 layers. (2) If GO: build the lookup table, measure ΔNLL.

**Smallest test**: ≤ 15 min variance ratio computation. NO-GO kills the frame: W_Q column weights are isotropic w.r.t. activation quantile — no quantile structure exists, and the lookup table is no better than global mean.

**What you're NOT proposing**: Not a global compression of W_Q (the lookup table is LARGER than W_Q in the worst case). The payoff is compute: the 16-bucket lookup is faster than 2048-multiply GEMV on constrained hardware (8-bit SIMD, embedded inference, where lookup tables outperform GEMV). This is a CPU throughput optimization at the cost of a slightly larger table, not a weight residency reduction.

---

### S5 — SINQ-Balance SVD-Rank Prediction (F4 stripped primitive, now grounded in SINQ) [Track B / F white space]

**Elegance class**: `gauge-exploitation`
**Track**: B
**CF anchor**: CF11 (W_Q r_99/d ≈ 0.63 at baseline), CF11 Sinkhorn precondition (W_Q W_Q^T > 0)
**Outside anti-frames of**: F orientation (anti-frame added this round: "Sinkhorn balance for quantization scale" — this targets SVD rank, not quantization scale); all other orientations

**Ambition**: Use SINQ's published Sinkhorn balance (arXiv:2509.22944) as a free preprocessing step to discover whether W_Q's r_99/d drops below 0.63 in the balanced coordinate, enabling more aggressive SVD truncation than CF11's K=128 baseline.

**Mechanism**: Apply SINQ's Sinkhorn-Knopp iterations to W_Q[14] (50 iterations, already available as a published algorithm). Compute r_99 of the balanced W_Q. If r_99/d_balanced < 0.63 × 0.90 (10%+ reduction), the Sinkhorn-balanced form supports a lower K than the training-time form, and the compression savings compound on CF11's existing K=128 GO. The scale factors D_L, D_R produced by Sinkhorn are absorbed into adjacent RMSNorm (input side: diagonal scaling of γ) and W_O (output side), producing an algebraically equivalent network with a more compressible W_Q. This is NOT the F4-killed proposal: F4 re-derived the theory from first principles. S5 uses SINQ's published algorithm directly as a tool and tests a new application (SVD rank) that SINQ does not claim or test.

CF10 check: no calibration fit. Sinkhorn is calibration-free (only weight statistics needed). CF9 check: Sinkhorn convergence theorem (Sinkhorn-Knopp), precondition W_Q W_Q^T > 0 verified by CF11. The absorption of D_L into RMSNorm γ requires per-channel scaling of γ — this is the same γ-absorption as SmoothQuant, permissible (diagonal, not O(d) rotation per killed GFQO). No kill-list conflict.

**What would have to be true**: W_Q is NOT already near-Sinkhorn-balanced from SGD training. F4's Stage 2 analysis noted that weight decay may produce near-balanced weights spontaneously; if W_Q is already balanced, r_99/d will not decrease and this is a clean NO-GO finding with structural value (SGD + weight decay = spontaneous Sinkhorn balance).

**Experiment cascade**: (1) Load W_Q[14]; run 50 Sinkhorn iterations; compute r_99 before and after. Runtime: ~10 min. GO: r_99/d drops ≥ 10% after balancing. (2) If GO: run NLL sweep at new K on balanced W_Q; confirm ΔNLL improvement vs CF11 baseline.

**Smallest test**: ≤ 10 min. Clean binary: r_99/d drops or stays flat.

**What you're NOT proposing**: Not re-deriving Sinkhorn theory (cite SINQ directly). Not applying SINQ's quantization output (this is pure SVD-rank). Not rotating the residual stream (Sinkhorn produces diagonal D_L, D_R, absorbed as per-channel scales — safe per the run_004 kill's anti-frame which banned O(d) rotations, not diagonal scalings).

---

### S6 — User-Space INT4-Alias Page Dedup for GGUF [Track B / content-addressable frame]

**Elegance class**: `no-SGD-reformulation`
**Track**: B
**CF anchor**: Actual stack (GGUF format, ik_llama.cpp, INT4 weights, NVMe offload)
**Outside anti-frames of**: U orientation (doesn't require VirtualLock or PrefetchVirtualMemory); R orientation (not activation quantization); all others (pure systems primitive)

**[FREE SWING]** — no CF anchor beyond actual stack

**Ambition**: Detect identical INT4-quantized weight pages in GGUF at model load time; serve aliased pages from a single physical memory buffer, reducing DRAM working set without changing model format or ML code.

**Mechanism**: At GGUF model load time, compute a fast (xxHash3) hash of each 4 KB aligned INT4-quantized weight page. Build a dedup table: if two pages have identical hashes (and a byte-by-byte verification), register them as aliases. At inference, the alias table maps any aliased page offset to the first occurrence's physical buffer. Read requests for aliased pages return the physical buffer without NVMe reads after the first occurrence. This is user-space reflink semantics, not NTFS-reflink: a pointer indirection in the GGUF mmap layer, not a filesystem feature.

Implementation: modify ik_llama.cpp's weight loading to maintain an alias table (hash → physical offset). At GEMV time, the tensor address lookup goes through the alias table. Overhead: one hash-lookup per tensor per token (O(1), ~0.1 μs).

The dedup rate at INT4 quantization: with a finite codebook, duplicate pages exist when two weight tiles happen to map to the same set of INT4 codes. For random-looking INT4 tiles at 4 KB (8192 nibbles), the probability of a random collision is negligible. But the K=0.1% static-channel + median-channel structure (CF3) suggests some weight pages are near-identical (the same 2 static channels appear in every layer's W_Q, W_K, W_V, W_up; if those 2 channels dominate a 4 KB page's content, many pages across layers may produce identical nibble patterns after INT4). This is the structural argument, not a random-collision argument.

**What would have to be true**: The dedup rate at INT4, 4 KB pages, across all weight tensors is ≥ 1% (at least 1 in 100 pages is a duplicate). At 35 GB model (70B): 1% dedup = 350 MB DRAM savings.

**Experiment**: (1) INT4-quantize Qwen3-1.7B; compute xxHash3 of all 4 KB pages; count duplicates. Runtime: ~20 min. GO: dedup rate ≥ 0.5%. NO-GO: < 0.1% — INT4 quantization produces near-uniform nibble distributions per page and no structural dedup exists at page granularity.

**Smallest test**: ≤ 20 min. Pure measurement, no ML code modification.

---

### S7 — Post-Training W_Q Binarization on CF11 K=128 Truncated Basis [Track B / bit-serial frame]

**Elegance class**: `subspace-alignment`
**Track**: B
**CF anchor**: CF11 (W_Q K=128 global GO at +0.98 nats; per-head K=64 NO-GO)
**Outside anti-frames of**: F orientation (no Sinkhorn, no cross-layer basis); A orientation (no rotation); C orientation (no composition coupling); the per-head truncation kill (CF11 per-head K=64 NO-GO) does not apply because this is post-truncation binarization of the GLOBAL K=128 projection matrices, not per-head truncation

**Ambition**: After CF11 K=128 global W_Q truncation, binarize the 2048×128 and 128×2048 factor matrices, achieving effective 1-bit storage for W_Q at < 2 nats ΔNLL.

**Mechanism**: CF11 establishes that W_Q compressed to K=128 (global, not per-head) costs +0.98 nats. The factor matrices are U ∈ R^{2048×128} and V^T ∈ R^{128×2048}. After the K=128 truncation, binarize: U_binary = sign(U), V_binary = sign(V^T). The binarized GEMV: W_Q_binary × h ≈ U_binary × diag(Σ_K) × (V_binary × h). This is a BitLinear-style computation on the truncated factors. The CF11 K=128 truncation already discarded 87.5% of W_Q's dimensions; the remaining 128-dimensional representation is binarized, introducing additional quantization error. The load-bearing question: does binarizing the K=128 factors add < 0.5 nats ΔNLL on top of the K=128 truncation cost?

The structural argument: the top-128 singular vectors of W_Q span the "head-shared subspace" which CF11 identified as the query-space attractor. Within this 128-dimensional subspace, the singular values Σ_K have relatively flat spectrum (all 128 kept — no further truncation). Binarizing within this subspace should have less ΔNLL cost than binarizing the full W_Q, because the K=128 subspace is maximally information-dense.

CF10 check: no calibration fit (binarization is deterministic: sign function). CF9 check: no imported theorem; standard sign binarization.

**What would have to be true**: The mean binarization error ‖U_binary - U‖_F / ‖U‖_F is < 0.5 for U ∈ R^{2048×128} (typical normalized random matrix: expected error ~0.36 per SVD vector direction). At the K=128 subspace, the binarization noise adds in quadrature with K=128 truncation noise. If binarization adds < 0.5 nats, the total W_Q storage drops from 4.2 MB/layer (K=512, CF11 GO at +0.20 nats) to 128/8 × 2048 × 2 / 8 KB/layer ≈ 0.065 MB/layer — 64× reduction.

**Experiment**: (1) Compute U, V^T from W_Q[14] SVD K=128; binarize; measure ΔNLL vs K=128-only. Runtime: ~15 min. GO: total ΔNLL (K=128 + binarize) < 2.0 nats. NO-GO: binarization adds > 1.5 nats on top of K=128 → binarization at this quantization level is not viable post-truncation.

**Smallest test**: ≤ 15 min. Binary outcome (binarization within the K=128 subspace is tolerable or not).

---

### S8 — Low-Rank-by-Construction Cross-Scale Affine Correction (WDLA Rescue) [Track A / CF10-conditioned]

**Elegance class**: `calibration-fit` (conditioned, r ≤ 64)
**Track**: A
**CF anchor**: CF10 (WDLA killed at r=full due to calibration ill-conditioning; explicitly names r ≤ 64 as well-conditioned rescue), CF11 (W_Q concentrated attention weights as the natural target for cross-scale affine surgery)
**Outside anti-frames of**: A orientation's current anti-frames (no O(d) rotation; this uses affine, not rotation); C orientation (no composition coupling needed); the WDLA kill is for full-affine, not low-rank

**Ambition**: Resurrect the WDLA cross-scale compression frame (Qwen3-0.6B weights + rank-64 affine correction → approximate Qwen3-1.7B inference quality) under CF10-safe conditioning, applying it specifically to W_Q where CF11 establishes the most concentrated weight class.

**Mechanism**: Fit a rank-r affine correction A_Q: R^{d×d} for W_Q only (not full cross-scale map for all matrices). A_Q at r=64: 2048×64 + 64×2048 = 262K parameters. Calibration: 1K tokens × 2048 output dims = 2M values >> 262K parameters — well-conditioned with ridge=1e-1. The correction: W_Q^{1.7B} ≈ W_Q^{0.6B} + U_A Σ_A V_A^T where U_A, V_A are r=64 matrices fit by least squares on calibration. The structural argument: CF11 shows W_Q has r_99/d ≈ 0.63, meaning only ~1290 of 2048 dimensions carry 99% of the spectrum. If the cross-scale delta W_Q^{1.7B} - W_Q^{0.6B} is concentrated in these same 1290 dimensions, a rank-64 correction captures a significant fraction. The A-PEER idea tested whether the raw delta is low-rank; S8 fits the correction explicitly under CF10-safe conditioning rather than relying on the raw delta being low-rank.

CF10 check: r=64 → 262K parameters, 1K calibration tokens × 2048 output dims = 2M values. 2M >> 262K. With ridge=1e-1: expected R²_eval > 0.8 (per CF10 guidance for properly conditioned fits). This is the explicitly named CF10-safe rescue from WDLA's post-mortem.

CF9 check: no imported theorem; ridge least squares, standard. Precondition: the training-conditioned mean of (W_Q^{1.7B} - W_Q^{0.6B}) applied to the calibration distribution is learnable at r=64. This is falsifiable by the R²_eval gate.

**What would have to be true**: The rank-64 correction of W_Q achieves held-out R²_eval > 0.5, implying that running Qwen3-0.6B with the W_Q correction approximates Qwen3-1.7B W_Q quality. The residency saving: store only W_Q corrections (28 × 262K × 2B = 14.7 MB) instead of W_Q^{1.7B} (28 × 8 MB = 224 MB), IF W_Q^{0.6B} is freely available (pre-downloaded). Saving: 209 MB on W_Q, keeping W_Q^{0.6B} as a shared base.

**Experiment**: (1) Fit A_Q^{r=64} for all 28 layers via ridge regression (1K calibration tokens, W_Q^{0.6B} activations as X, W_Q^{1.7B} activations as Y). Compute train R² and held-out R² on separate 200-token set. Runtime: ~45 min. GO: held-out R²_eval > 0.5 AND ΔNLL < 1.0 nats vs baseline W_Q^{1.7B} on WikiText-2 using the corrected W_Q.

**Smallest test**: ≤ 45 min. The R²_eval gate is the load-bearing check from CF10.

**What you're NOT proposing**: Not fitting corrections for all weight matrices (only W_Q, where CF11 gives the most concentrated target). Not using the full affine map (rank-64 by construction from the start, not rank-truncated after the fact).

---

## 7. Output Handoff

### Stage 4 ideas summary

| ID | Track | Section motivation | Description |
|---|---|---|---|
| S1 | B | §4.1 compiler PGO, §5.1 static channels | Static-channel GEMV fold across all weight matrices; CF3 K=0.1% contribution pre-computed as bias |
| S2 | B | §4.2 DB late materialization | SwiGLU gate-predicate W_up row skip; token-wise skip where |silu(W_gate·h)| < ε |
| S3 | A | §4.3 control theory Kalman observer | CF2-bounded layer-skip oracle: linear predictor f_L(h_L) forecasts ‖Δh_L‖₂ without loading MLP weights |
| S4 | B | §5.1 no-continuous-numbers + FPGA lookup | Quantile-bucket lookup-table GEMV for W_Q; replaces 2048-multiply GEMV with 16-bucket accumulation |
| S5 | B | §3 CP-1 F4 stripped primitive | Apply SINQ's published Sinkhorn balance to measure whether W_Q r_99/d decreases; SVD-rank application SINQ does not claim |
| S6 | B [FREE SWING] | §5.2 content-addressable | User-space INT4-alias page dedup for GGUF; xxHash3 at load time, pointer indirection at runtime |
| S7 | B | §5.3 bit-serial / integer-only | Post-training W_Q binarization on CF11 K=128 truncated factors; tests whether binarization within the head-shared subspace is tolerable |
| S8 | A | §3 CP-3 WDLA stripped primitive | Low-rank-by-construction r=64 cross-scale W_Q correction (WDLA rescue); CF10-conditioned, 262K parameters, 2M calibration values |

### Orientation rotation recommendations

- **A — TIGHTEN**: integer-constraint frame saturated by I-LLM; add anti-frame "integer normalization as research target." Focus A on content-addressable and distributed-state constraints next round.
- **C — KEEP**: top Stage 2 score (C-UFDM, 13); Cluster C1/C2 participation.
- **F — KEEP**: two top-3 proposals (F3-WO-HEADBLOCK-RANK, F2-UNTIED-LMHEAD); add anti-frame "Sinkhorn balance for quantization scale optimization (SINQ space)."
- **R — KEEP**: WVWO-FOLD top-3; RAOK-70B strong; add anti-frame "KV compression at ≤4K context" (covered by C-WKOL sidecar; diminishing return for standalone KV ideas at short context).
- **U — KEEP**: three genuine substrate-primitive REFINEs; add anti-frame "zstd-dict or entropy-coding of bf16 weight tiles before quantization" (compressibility check expected to fail per §U-ZSTD-DICT analysis).

### Anti-frame additions

| Orientation | Addition | Rationale |
|---|---|---|
| A | "integer normalization folding as a research contribution" | I-LLM (arXiv:2405.17849) publishes DI-Normalization; engineering port is valid, research claim is dead |
| F | "Sinkhorn balance applied to LLM weight quantization scale" | SINQ (arXiv:2509.22944) owns this; F orientation should target Sinkhorn's SVD-rank application (S5 is the surviving edge) |
| R | "KV cache compression at ≤4K context as primary lever" | C-WKOL sidecar + RAOK Tier structure handles short-context KV adequately; standalone KV compression at 4K is low-leverage per existing residency arithmetic |
| U | "zstd or arithmetic entropy coding of bf16 weight tiles without prior quantization" | Near-uniform entropy expected; compressibility check (U-ZSTD-DICT 30-min test) should confirm NO-GO before investing further |

### Stage 5 recommendations

**Primary recommendation**: **S5 (SINQ-Balance SVD-Rank)** — 10-min test, calibration-free, directly extends F4's stripped primitive into the only non-SINQ-covered territory. If W_Q r_99/d drops ≥ 10% under Sinkhorn balance, the safe K for CF11 compression drops from 128 to ~115, saving an additional ~10% on attention weights without any new mechanism beyond applying a published algorithm (SINQ) to a new output (SVD rank rather than quantization scale).

**Secondary recommendation**: **S2 (SwiGLU Gate-Predicate Late Materialization)** — 30-min measurement test, connects directly to CF1's firing-rank dominance finding in a way that run_008's advancing ideas have not exploited (CF1 is the input to C-UFDM and WVWO-FOLD but neither targets the per-token gate-predicate skip), and produces a load-side saving at 70B NVMe-offload that is composable with RAOK-70B.
