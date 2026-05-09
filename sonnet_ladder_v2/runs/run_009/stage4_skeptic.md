# Stage 4 — Skeptic Explorer — Run 009

Date: 2026-05-09
Agent: Claude Sonnet 4.6 (Stage 4 Skeptic Explorer role)

---

## 1. Frame-Exhaustion Map

### Saturation-exhausted frames

**SE-1: Per-factor independent attention weight SVD compression.** The convergence cluster C1 has five ideas (MQKP, ATQKV, WQKRS, EQSUB, HSMLA) all attacking from independent angles, and Stage 3 found QSVD (arXiv:2510.16292), CARE (arXiv:2603.17946), TransMLA (arXiv:2502.07864), and Layer-wise Dynamic Rank (arXiv:2509.25622) in the vicinity. Specifically, "compress W_Q and W_K as independent matrices with per-matrix SVD" is saturated by AQFKV (internal, previous round), QSVD, and the general MLA literature. F1-MQKP survives because it asks a structurally different question (the product M = W_Q W_K^T), but the underlying "compress each attention matrix via low-rank" frame is deeply crowded. **Stripped primitive:** The product M^L = W_Q^L (W_K^L)^T as the algebraically correct compression target — this is what survives and what F1-MQKP exploits.

**SE-2: Hessian-calibrated per-layer mixed-precision quantization.** HAQ, OmniQuant, AWQ, GPTQ, LLM-QP are saturated in this space. C15-SNRATE survives by replacing the Hessian with a spectrum-derived SNR, but any proposal that reimplements the Hessian approach in disguise is dead. **Stripped primitive:** Sensitivity order derivable from weight-only SVD (no activation data, no Hessian computation) as a fast approximation to Hessian-based sensitivity.

**SE-3: Global gate folding / activation removal for MLP.** AERO (arXiv:2410.13060) is the published anchor. SDZC was killed globally (R4). CSMF was killed (R3). The frame "fold the gate globally without retraining" is saturated either by AERO (with retraining) or by the empirical R4 result (1.5% global fold rate, negligible). **Stripped primitive:** Layer-specific gate folding where the input distribution has a structural null-space property — what F3-L1GNF exploits.

**SE-4: KV cache quantization and eviction at short context (<32K tokens).** KVQuant, MiniKV, H2O, KITTY, KVSwap, HASHEVICT — the field is thoroughly covered. At 4K context, KV is a negligible 670 MB vs 40 GB weight bytes. Stage 3 correctly flagged that A2-APPENDKV's payoff is long-context (32K+). Any proposal attacking KV size at 4K context is dead. **Stripped primitive:** Content-addressable KV retrieval as a policy framing, applicable specifically to the ≥32K context case where KV becomes the dominant residency term.

### CF9-exhausted frames

**CF9-1: Shift-invariance strip for any query/key offset that is token-dependent.** The F2-CKDJ REGENERATE this run (and F3-SRSC in run_003) both died on the same failure: "static" channels contribute to query or key vectors in a way that is constant on the query side but position-dependent on the key side. Any mechanism that tries to strip a position-varying term via shift-invariance is dead. **Stripped primitive:** Static outlier channels (CF3 K=0.1%) as high-energy W_Q column directions — the coupling claim (are these the same channels?) is a valid measurement, but the strip mechanism is dead.

**CF9-2: Random Fourier Feature / Bochner's theorem applied to non-shift-invariant weight matrices.** RFSK (Track B R6) was killed on this. Any proposal importing Bochner's theorem for weight-matrix kernel approximation is dead. The precondition (shift-invariance of the covariance kernel) does not hold for weight matrices. **Stripped primitive:** Random low-rank projections as approximations — but this reduces to JL projection, which reduces to SVD, which we already know is the right answer (CF8 says MLP is full-rank so even SVD fails there).

**CF9-3: Count-Sketch / sparse-signal algorithms on dense residual streams.** CSKQ (Track B R6) was killed because INT4 residuals are dense (L1 error bound = 5470× signal). Any proposal requiring sparsity to get polynomial-time approximation guarantees is dead for LLM activation streams. **Stripped primitive:** Dense-signal algorithms only — any idea importing a sparse-signal primitive must verify at the outset that the relevant signal is actually sparse.

### CF10-exhausted frames

**CF10-1: Calibration-fitted full-rank affine maps between hidden states of different models.** WDLA died on this (2.1M params, ~2M calibration samples, R² = -118000 on held-out). Any proposal that fits a d_model × d_model (or similar) linear map on calibration data without explicit low-rank-by-construction or strong ridge is dead. **Stripped primitive:** Low-rank-by-construction calibration fits (e.g., fit a rank-64 map instead of a full-rank d × d map) — the conditioning check requires n_params << n_samples × d_output_per_sample.

---

## 2. Orientation-Saturation Diagnostics

**Orientation F (First-Principles):** Advanced 5 ideas to Stage 3, all REFINE — the strongest performance of any orientation this round. F1-MQKP (score 13, highest in union) is the top Stage 5 candidate. F produced both the top convergence-cluster representative (C1: MQKP) and the most elegant mechanism (product-SVD as Eckart-Young). Anti-frame addition: append "per-factor independent W_Q / W_K SVD at matched rank (without forming the product M = W_Q W_K^T)" to F's anti-frame list — this frame is now saturated by QSVD/CARE/AQFKV. Recommendation: **KEEP, TIGHTEN.** The product-SVD framing is the right direction; F should now push into the product's spectrum on other attention matrices (M_VO = W_V W_O^T? M_QK for all 28 layers jointly?).

**Orientation R (Reach):** Advanced 5 ideas; 4 REFINE, 1 conditionally (NVME-ATTN-STREAM). R9-R1 (ATQKV) and R9-R2 (RAOK-REACH) are the key cascade anchors. R produced the clearest path to a practical 70B deployment improvement. Anti-frame addition: append "standalone KV management at ≤4K context" and "NVMe prefetch schemes that don't re-derive CF13* as cascade rung 1" to R's anti-frame list. Recommendation: **KEEP.**

**Orientation C (Composition):** Advanced 4 of 5 ideas; F2-CKDJ was REGENERATE (CF9). C's remaining advancers (CFSHIFT, SNRATE, EQSUB, LAYERTAX) are all REFINE. C14-CFSHIFT is a frame-novel idea with no prior art. Anti-frame addition: append "softmax shift-invariance applied to any term that is query-offset (not query-independent scalar constant)" — same failure mode as F2-CKDJ and F3-SRSC. Recommendation: **KEEP, TIGHTEN** on the CF9 shift-invariance trap.

**Orientation A (Constraint-Alien):** Advanced 3 of 5 ideas (APPENDKV, FSMDEC, TROPFFN). A4-SEEDNET and A5-REGONLY were rejected. TROPFFN is a wildcard (C3 cluster). The constraint-alien frame is producing ideas in unexplored computational models (tropical algebra, FSM dispatch, append-only KV). Anti-frame addition: append "constraint-alien ideas that re-derive Apple LLM-in-a-Flash architecture without adding a new primitive" (A5-REGONLY killed for this reason). Recommendation: **KEEP.** The tropical semiring + FSM dispatch ideas are genuinely novel and belong in the pipeline.

**Orientation U (Unconventional Substrate):** Advanced 2 of 5 (ETWO, VGTP); 3 rejected on A1 scores. U is producing substrate-novel ideas but the residency/throughput payoffs are secondary (6% mean improvement for FIOP, 30% for OSRD on 8B only). U's surviving ideas (ETWO, VGTP) are wildcards. Anti-frame additions: append "OS I/O priority elevation for mean throughput improvement < 10%" (FIOP killed reason) and "OS RAM disk partial layer residency without quantification at 70B scale" (OSRD killed reason). Recommendation: **TIGHTEN.** U should focus on substrate primitives that produce structural findings (ETWO's access histogram, VGTP's VEH latency measurement) rather than engineering optimizations with marginal impact.

---

## 3. Cross-Pollination Opportunities

**XP-1: F2-CKDJ stripped primitive → Orientation C or R (quantization precision allocation).** The stripped primitive from F2-CKDJ REGENERATE is: "static outlier channels (CF3 K=0.1%) as high-energy W_Q input columns — if confirmed, motivates per-column W_Q precision allocation." The CF9 failure was in Orientation F (First-Principles, which tried to derive a lossless strip via shift-invariance). Orientation C's SNRATE (C15) is already building a precision-allocation oracle from spectrum + sensitivity. The stripped primitive from CKDJ would add a THIRD axis to the SNRATE oracle: column-specific energy within W_Q (not just global concentration). The cross-pollination: CKDJ's measurement (are CF3 static channels = top W_Q column energy channels?) feeds into a PER-COLUMN precision allocation for W_Q, tightening SNRATE's per-class bpw assignment to a per-column assignment. This is a Track B idea grounded in the CKDJ stripped primitive and lives outside F's "algebraic-identity" frame and outside C's "class-level SNR" frame.

**XP-2: WDLA stripped primitive → Orientation F (low-rank-by-construction cross-scale map).** WDLA died on CF10 (underdetermined fit). The stripped primitive was: "inference-time cross-scale affine surgery." Orientation F's RSUC (residual update covariance depth profile) opens a closely related door: if r_90%(δ^L) varies by depth, the active subspace dimensions are knowable per layer. A rank-64-by-construction affine map from 0.6B residual stream to 1.7B residual stream (forcing rank ≤ 64 from the start, ~200K parameters, well-conditioned at 200 tokens) is CF10-safe. This cross-pollination: WDLA's surviving mechanism + RSUC's depth-profile = a per-layer rank-scheduled cross-scale alignment. The cross-pollination produces idea S3 below.

**XP-3: HSMLA (R9-R3) warning from TransMLA → Track A (post-training head-sharing on KV, not Q).** TransMLA (NeurIPS 2025 Spotlight) covers the W_Q head-sharing territory at training time. HSMLA's W_Q factorization is differentiated but adjacent. The safer cross-pollination: apply the same stacked-SVD approach to W_V and W_O (the untested matrices from ATQKV). If TransMLA's KV-latent approach is training-time, the SAME algebraic question asked post-training for W_V and W_O (do they share a cross-head basis?) is unstudied and directly parallel to HSMLA but outside HSMLA's W_Q scope.

---

## 4. Cross-Domain Seed Exploration

### Database engineer's view

A database engineer sees attention as a **join operation**: Q is one table, K is another; the inner product Q K^T is a self-join on vector similarity; softmax is a weighted aggregation over matching rows. The database view immediately suggests **index structures** — B-tree-style range coding over attention logits, or zone maps (min/max per layer) that skip blocks of keys with no chance of being selected.

**Elegant-equivalence prompt:** What algebraic identity does a database engineer's view of the attention score reveal? The argmax(Q K^T) is a **nearest-neighbor query** in the Euclidean metric (after appropriate transformation). The nearest-neighbor query on a product M = W_Q W_K^T is exactly what MQKP computes — but a database engineer would additionally ask: can the index be maintained **incrementally** as the KV cache grows? An append-only index (A2-APPENDKV style) supports incremental NN queries without recomputation, which is the database-natural extension of MQKP's product-SVD insight to the runtime KV problem.

**Cross-domain idea trigger:** Zone maps over pre-projected keys. At inference, after computing M^L = U_M Σ V^T (MQKP's product), the projected keys V^T K_{cache} can have per-block min/max bounding boxes precomputed. For each new query q = x U_M Σ^{1/2}, the dot product with K_block is bounded by ||q|| × ||K_block|| ≤ max_bound. Blocks where the upper bound is below the top-K threshold can be skipped — a ZONE MAP SCAN over the projected key cache, requiring no approximation for the top-K result. This is a Track A idea grounded in MQKP's product SVD and database zone-map indexing. Selected as idea S1 below.

### Compiler writer's view

A compiler writer sees the autoregressive decode loop as a **profile-guided optimization target**: the same 80 layers execute every token, and the compiler should dead-code-eliminate layers (or sub-computations) that are invariant across tokens, specialize hot paths for common token types, and fuse computations that share intermediate values.

**Elegant-equivalence prompt:** What dead-code elimination does a compiler writer see in the residual stream? CF6's 36.1% layer-1 foldable neurons are exactly the compiler's dead code: they compute a constant, so the W_gate row is dead code for those neurons. F3-L1GNF and R9-R5 already exploit this. The compiler view adds one more: **common subexpression elimination (CSE)**. If two heads' W_Q factored projections share a substantial subspace (CF11's head-redundancy), the projection x W_Q^h for all 16 heads contains repeated computation — the shared-basis component x B (in HSMLA's notation) is computed 16 times but is the same tensor. CSE eliminates 15 of those 16 computations. HSMLA already exploits this structurally, but the compiler view frames it as a standard optimization pass — and asks: which OTHER computations in the transformer forward pass have the same CSE structure? W_K, W_V, W_O share the same residual stream input per layer; if their input projections share common subspaces (a question complementary to MQKP's), CSE applies similarly. This motivates a multi-matrix shared-basis idea for all four attention matrices simultaneously, selected as idea S4 below.

### Control theorist's view

A control theorist sees the residual stream as a **dynamical system**: h_{L+1} = h_L + f_L(h_L) where f_L is the transformer block's contribution. The CF2 finding (cos(h_L, h_{L+1}) ≈ 0.99) means the system is near-linear and slowly varying. A **Kalman filter** applied to the residual stream would model h_L as a hidden state with additive noise and predict h_{L+1} from h_L with a state-transition matrix close to identity. The Kalman gain would weight the new observation (layer output) against the prior (previous residual state) — exactly the structure of the residual update.

**Elegant-equivalence prompt:** What conservation law does a control theorist see in the residual stream? The near-unity cosine (CF2) implies that the residual stream's norm is approximately conserved (it can only change by 14% per layer). This is a near-conservation law. A control theorist would immediately ask: is there a GAUGE where the conservation is exact? The RMSNorm per layer normalizes the residual to unit norm — RMSNorm IS the conservation-enforcing operation. The control-theoretic view then asks: given that h_L / ||h_L|| lies on the unit sphere, and each layer adds a small tangent-space perturbation, can the total perturbation be bounded by a LYAPUNOV FUNCTION? If yes, the model is robustly stable and layers whose Lyapunov contribution is below a threshold can be pruned. This is different from any current kill-list mechanism (it targets layer pruning via control-theoretic stability, not weight-level compression). However, this re-derives layer-drop pruning (already killed in "pre" as "works K=1-2 layers but doesn't help residency at frontier scale"). The control view is interesting but converges on a known-dead frame. Not proposing an idea here; noting the convergence.

**Pivot:** The control theorist's view of ACTIVATION QUANTIZATION is more productive. Kalman-style state estimation for the activation outlier channels: instead of tier-switching per token (RAOK), use a Kalman update — the static-channel prior (Jaccard 0.718 at K=0.1%) is the process model, the per-token deviation is the observation noise, and the Kalman gain produces an optimal posterior estimate of the outlier channel set. This motivates a Bayesian-flavored activation quantization scheme that is lighter than RAOK's full dynamic per-token overhead but heavier than fully static LLM.int8(). Selected as idea S5 below.

---

## 5. Constraint-Driven Reframing

### Constraint: No persistent state — weights reconstructed from residual stream alone

This constraint is almost certainly impossible (CF8: MLP weights are full-rank, completely disconnected from the residual stream's current token-specific content). The failure shape IS the finding: the residual stream (2048 dims) cannot encode the information in a 6144 × 2048 matrix (12.6M params per layer). The only surviving fragment: for CF6's 36% layer-1 foldable neurons, the gate contribution IS derivable from calibration data + model structure (the c_i values). A narrow "no-persistent-state" design: at layer 1, the foldable-neuron gate contributions are derived from the input embedding (no stored state needed — the embedding IS the state). This reframes R9-R5/LAYERONE-FOLD as a "reconstruct 36% of layer-1 W_gate from the embedding input" rather than "fold constants." Not generating a new idea here; noting the conceptual connection.

### Constraint: Content-addressable — every weight chunk retrievable by hash of content

This constraint forces deduplication across weight blocks. If two weight blocks (e.g., W_up rows at different layers) have the same content hash, they share a storage cell — one copy serves both. The question: how many Qwen3-1.7B weight rows are near-duplicate across layers? CF8 (MLP full-rank per layer) plus the KILL of cross-layer W_Q sharing (v2-CHEAP-TEST-001) suggests minimal duplication. The content-addressable constraint does NOT require exact duplication — near-duplicate blocks (||W_i - W_j||_F / ||W_i||_F ≤ 0.01) can also share storage with a residual correction. This is adjacent to DeltaLLM (R1/S2 killed: "DeltaLLM/ADAMIX prior art; ceiling below NanoQuant"). Dead frame for MLP weights; possibly alive for attention weights if W_Q subspace sharing within a layer (HSMLA) produces near-duplicate head weight structures. Not generating a separate idea; noting the connection to HSMLA.

### Constraint: Storage is sequential-only (no random reads)

This constraint directly motivates R9-R4 (NVME-ATTN-STREAM), which is already in the pipeline. The constraint produces a natural extension: if weights must be read sequentially, the GGUF file layout is the only free variable. A **streaming decode** that fuses weight loading with the GEMV computation — reading a block of weight rows and immediately computing the dot product with the activation, without buffering the full weight matrix — is a pure sequential-read design. This is the Apple LLM-in-a-Flash architecture, already noted as adjacent to A5-REGONLY (killed). The only novel variant: **adaptive fused-decode block size** — rather than fixed-block streaming, the block size is chosen per layer based on F5-RSUC's residual update covariance depth profile (larger blocks for mid-layers where the update uses more directions, smaller blocks for early/late layers). This connects RSUC's empirical finding to the streaming architecture. Selected as idea S6 below.

### Constraint: No floating point at inference (integer arithmetic + lookup tables only)

This constraint is directly addressed by A1-TROPFFN (tropical semiring, INT8 max-plus GEMV) — already in the pipeline. The orthogonal slot this opens: **pure-integer attention score computation**. The attention score Q K^T involves FP16 dot products. For a DRAM-resident model (7B or smaller), the attention score computation is compute-bound rather than bandwidth-bound. An integer-only Q K^T using INT8 or INT4 Q and K vectors (post MQKP's factored form: U_M Σ^{1/2} and V_M Σ^{1/2}) combined with a 16-bit accumulator and a lookup table for the softmax exponential is feasible. The specific constraint: INT8 Q and INT8 K → INT16 accumulate → lookup-table softmax on INT16 scores (4096-entry table, 8 KB). No FP16 involved. The lookup-table softmax was explored in BinaryBERT-style work but not for autoregressive LLMs with RoPE. Selected as idea S7 below.

---

## 6. Generated Ideas (8)

### S1-MQKP-ZONEMAP — Zone-Map-Indexed KV Scan over Product-SVD Projected Cache

**Orientation:** Stage 4 cross-domain seed (database/MQKP cross-pollination)
**Track:** A (arch-transposition — changes the attention computation graph)
**Elegance-class:** algebraic-identity (product SVD projection + zone-map scan is an exact bound, not an approximation)
**CF tether:** CF11 (W_Q concentration), F1-MQKP (product SVD of M = W_Q W_K^T), A2-APPENDKV (KV-log structure)
**Sits outside:** F's anti-frame (this uses the product SVD FROM F1-MQKP as a prerequisite, not as the idea — the zone-map indexing is a Track A compute-graph change F hasn't proposed); C's anti-frame (this is not a composition of CF findings but a cross-domain database primitive applied to a pipeline-internal result); R's anti-frame (R doesn't touch zone maps on projected caches)

**Ambition.** After computing M^L = U_M Σ_M V_M^T (MQKP's product-SVD decomposition of the attention score operator), the KV cache is projected into the product-SVD basis: K̃_{cached} = K_{cached} V_M Σ_M^{1/2} ∈ ℝ^{T × K} (where K is the compressed rank, say 400). For each new query projection q̃ = x U_M Σ_M^{1/2} ∈ ℝ^K, the attention score is q̃^T K̃_{cached,t} for each cached position t. The zone-map insight: partition the projected KV cache into blocks of B=64 positions; for each block, precompute [min_b (K̃_t), max_b (K̃_t)] per projected dimension — a K-dimensional bounding box (zone map, 2 × K × 4 bytes per block). At query time, for each block, upper-bound the max dot product as ||q̃|| × ||K̃_block_max|| ≤ (element-wise max bound). Blocks where this upper bound is below the current top-K threshold in the attention logit heap are **skipped exactly** — no approximation, just early termination. This reduces the expected attention GEMV cost from O(T × K) to O(T_top × K + T × K / B) in the best case (most blocks pruned after zone-map check).

**Why our findings make this work.** MQKP's product-SVD is the enabling step — without the compact K-dimensional projection, zone-map blocks would span the full d=2048 dimensions (impractical). CF11's K=128-400 concentration makes the projection tight. A2-APPENDKV's append-only KV structure is the ideal substrate: the zone-map can be maintained incrementally (one append per B tokens, O(K) cost).

**What would have to be true.** (a) MQKP GO (M's r_99/d ≤ 0.55 at K≤400). (b) The projected KV cache has exploitable block-level non-uniformity (not all blocks have equal norm — if all K̃_t are equal-norm, zone maps give no pruning). (c) The top-K attention mass concentrates in few blocks (empirically, attention sinks and recent tokens typically dominate).

**Experiment cascade.** Rung 1: MQKP pilot (F1-MQKP, 15 min) — must GO before this idea proceeds. Rung 2: measure per-block projected-key block-norm variance and estimate pruning rate (10 min on calibration set). Rung 3: implement zone-map attention scan (Python numpy, ~4 hours) and measure realized FLOPs reduction vs exact attention.

**Smallest test.** Rung 2 above: after computing K̃ = K V_M Σ^{1/2} for all 4096 calibration-set positions at layer L14, partition into B=64 blocks and measure what fraction of blocks can be pruned at top-K threshold covering 90% of softmax mass. Runtime: ~45 min total (MQKP Rung 1 + zone-map measurement). GO threshold: ≥ 40% of blocks pruned at 90% mass retention. **NOT proposing:** KV eviction, KV quantization, or any approximate attention mechanism. The zone map produces exact top-K selection.

**What you're NOT proposing.** This is not locality-sensitive hashing (approximate) and not KV eviction. The zone-map bound is exact; the only cost is the O(K/B) bound computation per block.

---

### S2-WQCOL-PREC — Static Outlier Channel × W_Q Column-Energy Per-Column Precision Allocation

**Orientation:** Stage 4 cross-pollination from F2-CKDJ stripped primitive
**Track:** B (compression / quantization scheme)
**Elegance-class:** calibration-fit (but CF10-compliant: one scalar per column, ~2048 scalars, trivially conditioned at 200+ tokens)
**CF tether:** CF3 (K=0.1% static channels, Jaccard=0.718), CF11 (W_Q global K=128 GO, column-energy concentration)
**Sits outside:** F's anti-frame (F killed the "lossless strip" frame; this proposes per-column precision allocation — a Track B quantization schedule, not a computation-graph rewrite); all orientations' anti-frame for "shift-invariance strip" (this idea does not attempt a strip)

**Ambition.** F2-CKDJ's stripped primitive: measure whether the CF3 K=0.1% static outlier channels (the ~2 channels stable across tokens with Jaccard=0.718) have high W_Q column energy (top-5% of ‖W_Q[:, c]‖ for those 2 columns c). If YES, then for W_Q quantization at any bpw, those 2 columns should be pinned at higher precision (FP16 or INT8) while the remaining 2046 columns use the uniform grid. This is a per-column mixed-precision quantization of W_Q, motivated by the column-energy × static-channel coupling. Unlike per-row (SmoothQuant) or per-layer (OmniQuant) schemes, per-column W_Q precision allocation has not been published.

**Mechanism.** For each of the 28 W_Q layers: (1) identify CF3 static channels (K=0.1%, 1-2 channels per layer from calibration); (2) measure ‖W_Q[:, c]‖ for all 2048 columns; (3) check if static channels are in top-5% by column norm; (4) if YES, pin those columns at FP16 in the INT4 quantized W_Q (2 × 2048 × 2B FP16 = 8 KB per layer, negligible overhead); quantize remaining 2046 columns uniformly. The ΔNLL saving comes from: W_Q columns with high activation are the ones most degraded by uniform quantization (large-magnitude × quantization error = large contribution to output error). Pinning them at FP16 eliminates this contribution.

**CF9 precondition check.** No imported theorem. Per-column norm is a standard measurement. CF3 Jaccard is a measurement. The coupling claim (static channels = high W_Q column norm channels) is empirical, not theoretical. CF9 passes.

**CF10 check.** Per-column assignment: 2 columns identified as "high-precision" out of 2048. No fit; purely measurement-based identification. CF10 not applicable.

**Residency arithmetic.** Pinning 2 columns at FP16 per layer per head: 2 × 128 × 2B = 512 bytes per head, 16 heads × 28 layers = 229 KB overhead vs ~29 MB W_Q at INT4 = 0.8% overhead. Negligible. Quality gain: if the static channels are indeed top-5% column norm AND quantizing them uniformly is the dominant error source, ΔNLL improvement at INT4 W_Q could be 0.1–0.3 nats (based on typical per-column error concentrations in activation-aware quantization literature).

**What would have to be true.** (a) CF3 static channels at K=0.1% are in top-5% W_Q column norms in ≥20/28 layers (the F2-CKDJ experiment, not yet run). (b) The uniform INT4 quantization error on W_Q is dominated by those columns' quantization error. (c) Pinning 2 columns at FP16 measurably reduces ΔNLL at INT4 W_Q.

**Experiment cascade.** Rung 1: F2-CKDJ stripped measurement (35 min, calibration forward pass + column norm check). Rung 2 (if Rung 1 GO): quantize W_Q to INT4 with vs without FP16-pinned static columns; measure ΔNLL difference on 455-token held-out (~30 min). Total: ~65 min.

**Smallest test.** Rung 1 described above. GO threshold: static channels in top-5% W_Q column norms in ≥20/28 layers. **NOT proposing:** a shift-invariance strip (CF9-killed), per-row activation migration (SmoothQuant), or Hessian-based per-column sensitivity. This is purely measurement-driven column identification.

---

### S3-LRCS-WDLA — Low-Rank-by-Construction Cross-Scale Surgery (WDLA Rescued)

**Orientation:** Stage 4 cross-pollination from WDLA stripped primitive
**Track:** A (arch-transposition — inference-time cross-scale model alignment)
**Elegance-class:** calibration-fit (but CF10-compliant by construction: rank ≤ 64 forces parameter count <<< calibration samples)
**CF tether:** CF10 (calibration conditioning lesson from WDLA), F5-RSUC (depth profile motivates per-layer rank assignment), CF11 (W_Q concentrated subspace motivates the low-rank structure for alignment)
**Sits outside:** all prior orientations' anti-frames for WDLA (the original died on CF10; this variant is CF10-safe by construction)

**Ambition.** WDLA attempted to fit a full-rank (d × d) affine map from 0.6B hidden states to 1.7B hidden states and died on CF10 (2.1M parameters, ~2M calibration samples, R²=1.0 on calibration and R²=-118000 on held-out). The stripped primitive: a cross-scale alignment map that is **low-rank by construction** — rank ≤ 64 forced from the start, parameter count ~200K, well-conditioned at even 200 calibration tokens.

**Mechanism.** Replace each 0.6B residual stream vector h^{0.6B}_L ∈ ℝ^{1024} with a rank-64 linear projection to ℝ^{2048}: h^{1.7B est}_L = A_L h^{0.6B}_L + b_L where A_L = U_L (64 × 1024) and V_L (2048 × 64) decomposed as A_L = V_L U_L ∈ ℝ^{2048 × 1024}, with V_L and U_L fitting only 64 × (1024 + 2048) = 196,608 parameters per layer. At 200 calibration tokens × 2048 output dims = 409,600 independent samples vs 196,608 parameters → n_samples > n_params by 2×. Ridge λ = 0.1 ensures no memorization. CF10 is satisfied. The goal: h^{1.7B est}_L replaces h^{1.7B}_L at inference; if the alignment is good (R² > 0.70 on held-out), the 0.6B model generates the 1.7B model's per-layer hidden states, enabling a cheaper speculative decoder that runs the 0.6B model but generates 1.7B-quality activations.

**Why RSUC's depth profile matters.** If r_90%(δ^L) varies by depth (F5-RSUC hypothesis), early and late layers have lower-dimensional update directions — the rank-64 alignment map may be tighter at those layers (rank 32 sufficient) and looser at mid-layers (rank 128 needed). A depth-stratified rank assignment (LRCS-WDLA + RSUC) reduces parameter count further while matching calibration quality.

**CF9 precondition check.** No imported theorem. Rank-64 linear projection is a standard matrix operation. CF9 passes.

**CF10 check.** Explicitly: n_params = 196,608 per layer; n_independent_samples = 200 tokens × 2048 output dims = 409,600. Ratio = 2.08× oversampling. With ridge λ = 0.1: well-conditioned. CF10 passes.

**Residency arithmetic.** The 0.6B model runs as a "shadow" predictor. Its residency: ~1.2 GB at bf16. Per-layer alignment maps: 28 layers × 196,608 params × 2B = 11 MB total. At inference: run 0.6B, project to 1.7B space, use as speculative decode draft. Accept rate target: > 0.40 (viable speculative decode). If accept rate = 0.40, effective throughput = 1.7B quality at ~1.4× speed. Quality cost: ΔNLL from the projected alignment vs true 1.7B hidden states — measured as ΔNLL of decoding with aligned 0.6B hidden states vs true 1.7B at full precision.

**What would have to be true.** (a) F5-RSUC GO: depth-stratified rank assignments reduce per-layer alignment error. (b) Rank-64 cross-scale alignment achieves R² > 0.70 on held-out (unlike WDLA's R² = -118000). (c) Aligned hidden states produce speculative decode accept rate > 0.30.

**Experiment cascade.** Rung 1: fit rank-64 alignment at one layer (L14) on 200 tokens; measure R²_calib vs R²_eval. Runtime: ~15 min. Rung 2 (if Rung 1 GO): fit all 28 layers; measure ΔNLL of full-model decode with aligned 0.6B states.

**Smallest test.** R²_eval ≥ 0.60 at L14. **NOT proposing:** full-rank affine surgery (CF10-killed), weight-space interpolation between models, or retraining.

---

### S4-ATTN-CSE — Compiler-Motivated Common-Subexpression Elimination Across All Four Attention Matrices

**Orientation:** Stage 4 cross-domain seed (compiler writer's view)
**Track:** B (compression — storage reduction for attention weights)
**Elegance-class:** subspace-alignment (the shared input projection is an algebraic identity when the four attention matrices share input directions)
**CF tether:** CF11 (W_Q head-redundancy motivates CSE), R9-R1/ATQKV (all four attention matrices being measured), F4-WQKRS (asymmetric rank allocation motivates non-uniform CSE)
**Sits outside:** F's "compute product M = W_Q W_K^T as compression target" frame (ATTN-CSE compresses the INPUT projection, not the product); R's anti-frame (R proposed the four-matrix cascade for RESIDENCY, this is for COMPUTE CSE); C's anti-frame (not a CF coupling chain)

**Ambition.** All four attention projection matrices (W_Q, W_K, W_V, W_O at layer L) share the same input: the residual stream vector h_L ∈ ℝ^{2048}. From a compiler's CSE perspective: if W_Q, W_K, W_V share a common set of "important input directions" (a shared subspace in ℝ^{2048}), then the projection of h_L onto that shared subspace can be computed ONCE and reused for all three projections. This is a standard CSE transformation: compute z = B^T h_L (one small projection) once, then compute q = W_Q_red z, k = W_K_red z, v = W_V_red z where W_Q_red, W_K_red, W_V_red are the residual projections mapping from the shared basis to each matrix's output space. The COMPUTE saving: one B^T h_L (d × K_shared) instead of three separate (d_model × d_KV) projections = reducing 3d² MACs to d × K_shared + 3 × K_shared × d_KV MACs.

**CF9 precondition check.** The CSE transformation is valid if and only if the shared basis B spans the UNION of the important input subspaces of W_Q, W_K, and W_V. This is a standard subspace-union argument, not an imported theorem. The precondition: the union of the top-K_Q, top-K_K, top-K_V right singular subspaces of W_Q, W_K, W_V has dimension ≤ K_shared (< d). Given CF11 (W_Q r_99/d ≈ 0.63 → K_Q ≈ 1289) and the expected W_K (r_99/d ≈ 0.79 → K_K ≈ 1618) and unknown W_V, the union might span up to K_Q + K_K + K_V dimensions (no sharing). But if W_Q, W_K, W_V share substantial input-subspace overlap (which EQSUB's hypothesis suggests is possible via embedding alignment), K_shared < K_Q + K_K + K_V. CF9 passes if we measure the union subspace dimension before claiming CSE applies.

**CF10 check.** No calibration fit. Pure weight SVD + subspace union measurement. CF10 not applicable.

**Residency arithmetic.** The shared basis B ∈ ℝ^{d × K_shared} replaces separate Q/K/V input projections. Storage for B: K_shared × 2048 × 2B. If K_shared = 1700 (generous union): B = 1700 × 2048 × 2 = 6.97 MB vs W_Q + W_K + W_V at K=512 each = 3 × 2 × (2048 × 512) × 2 = 12.6 MB — saves 45% on the Q+K+V input projection storage. Compute saving: one K_shared-dim projection vs three separate d-dim projections: 3 × 2048 × d_head × n_heads MACs reduced to K_shared × 2048 + 3 × K_shared × d_head × n_heads. At K_shared = 1700 vs 3 × 2048 = 6144: ~3.6× compute reduction on the input projection step.

**What would have to be true.** (a) The union of W_Q, W_K, W_V right singular subspaces at the same rank budget fits in a K_shared < K_Q + K_K + K_V dimensional basis. (b) The CSE-factored projections (via B + separate W_red matrices) produce the same ΔNLL as separate compressions at matched budget. (c) ATQKV GO (W_V and W_O spectrum measured, not full-rank).

**Experiment cascade.** Rung 1: ATQKV pilot (R9-R1, 40 min) — must GO. Rung 2: compute the union subspace of W_Q, W_K, W_V right singular vectors at K=512 each via SVD of [W_Q^T; W_K^T; W_V^T] (stacked 3 × 2048 × 2048 → top-K_shared singular vectors); measure K_shared needed for 95% subspace coverage. ~20 min. Rung 3: CSE factorization + ΔNLL measurement.

**Smallest test.** Rung 2 described above. GO threshold: K_shared ≤ 0.75 × (K_Q + K_K + K_V) at 95% coverage. **NOT proposing:** cross-layer sharing (v2-CHEAP-TEST-001 killed), rotation-based transforms (v2-S3-R004-001 killed), MLA-style training-time compression (TransMLA).

---

### S5-KALMAN-TIER — Kalman-Filtered Activation Outlier Tier Predictor

**Orientation:** Stage 4 cross-domain seed (control theorist's view)
**Track:** B (compression — activation quantization with temporal prediction)
**Elegance-class:** conserved-quantity (the static-channel prior is a near-conservation law exploited by the Kalman gain)
**CF tether:** CF3 (K-dependent Jaccard — process model: top-0.1% static with Jaccard=0.718, top-1% dynamic with Jaccard=0.31), v2-CF1 (28-layer generalization of CF3), R9-R2/RAOK-REACH (the same tiering problem from a different angle)
**Sits outside:** R's anti-frame (RAOK-REACH is three tiers with per-token full sweep; KALMAN-TIER uses the temporal prediction to SKIP the full dynamic sweep on most tokens); C's composition frame (this is a control-theory import, not a CF-coupling)

**Ambition.** RAOK's three-tier scheme requires, at each token: (1) always use FP16 for the 2 static channels (cheap), (2) compute the dynamic-channel set for the 18 K=1%-tier channels via per-token measurement (expensive: requires an activation check on 18 channels per token per layer). The Kalman insight: the dynamic-channel set at token t+1 is PREDICTABLE from the dynamic-channel set at token t and the static-channel prior (Jaccard = 0.31 means ~31% of the same 18 channels are retained token-to-token). A Kalman filter on the binary channel-membership vector (which channels are "active" in the dynamic tier) uses the Jaccard persistence as the process model and the actual per-token activation as the observation. The Kalman gain produces a posterior estimate of the channel set for the NEXT token without requiring a full per-token sweep of all 18 channels. The control-theoretic view: the Jaccard persistence (0.31) IS the process noise covariance; the observation noise is the within-token measurement error. At steady state, the Kalman filter converges to a fixed-lag predictor: "predict the next token's dynamic channel set from the current set, correcting only the channels where the prediction confidence is below threshold."

**Concrete mechanism.** For each layer, maintain a 18-entry "dynamic tier confidence" vector c_t ∈ [0,1] (probability that channel j is in the dynamic tier at token t). Process update: c_{t+1|t} = 0.31 × c_t + (1 − 0.31) × base_prior (Jaccard = 0.31 is the retention rate). Observation: after computing the per-token activation, for the top-18 channels, update c_{t+1|t+1} via Bayes: channels that are indeed active get c raised toward 1; channels that were predicted but absent get c lowered. For channels with c_{t+1|t} > 0.85 (high confidence they WILL be in the dynamic tier), pre-allocate INT8 precision BEFORE computing the activation (saving the tier-switch cost). For channels with c_{t+1|t} < 0.15 (high confidence NOT in dynamic tier), use INT4 preemptively without verification. Only channels with c ∈ [0.15, 0.85] require full per-token measurement.

**CF9 precondition check.** Kalman filter precondition: linear process model, Gaussian noise. The binary channel-membership process is NOT Gaussian. Mitigation: use a Beta-Bernoulli model (conjugate prior for binary processes) rather than the standard Gaussian Kalman filter — this is a Bayesian analog that maintains the same recursive structure but is correctly specified for binary observations. Alternatively, use the Kalman-inspired structure as a heuristic predictor (not a formal Kalman filter) with empirical validation. The CF9 risk is the Gaussian assumption; the mitigation is either Beta-Bernoulli or purely empirical. **CF9 PASSES with mitigation.**

**CF10 check.** The process model has 2 parameters (retention rate = 0.31 from CF3, base prior from calibration). Trivially well-conditioned. CF10 passes.

**What would have to be true.** (a) The Kalman predictor correctly identifies ≥70% of the K=1% dynamic channel set with only ≥60% of tokens requiring full per-token measurement (30% savings in tier-checking overhead). (b) The mis-predicted channels (false negatives: channels that ARE dynamic but predicted as INT4) do not cause more than 0.05 nats ΔNLL per mis-prediction event. (c) Implementation is compatible with RAOK-REACH's activation quantization harness.

**Experiment cascade.** Rung 1: measure per-layer Jaccard persistence from token t to t+1 WITHIN a sequence (CF3 measured across random token pairs; within-sequence persistence may be higher). Runtime: ~20 min using PDAP calibration set. Rung 2: implement Beta-Bernoulli predictor; measure prediction accuracy on held-out token sequences. Rung 3: measure ΔNLL with Kalman-predicted tier vs full-measurement tier.

**Smallest test.** Within-sequence token-to-token Jaccard (K=1%) at 5 representative layers. GO threshold: mean within-sequence Jaccard ≥ 0.40 (higher than the 0.31 cross-token baseline, indicating temporal predictability WITHIN a sequence). **NOT proposing:** a global static channel assignment (LLM.int8(), killed) or a full per-token sweep (RAOK's baseline). This idea reduces the OVERHEAD of RAOK without changing the quality of its output.

---

### S6-STREAMING-DEPTH — Depth-Stratified Streaming Block Size for Sequential-Only NVMe Decode

**Orientation:** Stage 4 constraint reframing (sequential-only storage) × F5-RSUC depth profile
**Track:** B (systems + compression — streaming weight load scheduling)
**Elegance-class:** conserved-quantity (the update covariance rank as a conserved quantity per layer depth band)
**CF tether:** F5-RSUC (r_90%(δ^L) depth profile), R9-R4/NVME-ATTN-STREAM (NVMe streaming architecture), CF2 (adjacent-layer cosine ≈ 0.99 motivating depth stratification)
**Sits outside:** R's anti-frame (R9-R4 was layer-interleaved attention/MLP; this is WITHIN-LAYER block size adaptation based on depth); U's anti-frame (U's NVMe ideas were about I/O priority and RAM disk, not block-size scheduling); F's anti-frame (F doesn't touch streaming schedules)

**Ambition.** RSUC's hypothesis: early and late layers have lower-dimensional residual update covariances (r_90% is lower). If confirmed, these layers' GEMV computations touch fewer independent weight directions per token. A streaming decoder that reads weight blocks sequentially can exploit this: **use smaller streaming blocks for early/late layers** (fewer weight rows needed per token, faster per-layer decode without buffering) **and larger blocks for mid-layers** (higher-dimensional updates require more weight rows per step). The adaptive block size reduces the working-set size (DRAM buffer needed) for early/late layers, freeing DRAM for the mid-layer weight blocks that matter most.

**Mechanism.** For each layer L, measure r_90%(δ^L) from RSUC. Define streaming block size B_L = ceil(r_90%(δ^L) / d × d_model × batch_size). Early/late layers (r_90% ≈ 0.3 × d): B_L ≈ 0.3 × 2048 × batch_size = small. Mid-layers (r_90% ≈ 0.8 × d): B_L ≈ 0.8 × 2048 × batch_size = large. The streaming decoder reads B_L weight rows per step, fuses the GEMV computation with the load (no buffering beyond the current block), and proceeds. The DRAM buffer required: max_L (B_L × d × bpw) = mid-layer buffer size. The saving: early/late layers use smaller DRAM buffers, leaving more headroom for the mid-layer blocks. At 70B: if r_90% varies by 40% (RSUC GO threshold), early/late blocks are 60% of mid-layer size → total DRAM buffer peak ≈ 0.8 × uniform-block DRAM requirement.

**CF9 precondition check.** No imported theorem. r_90%(δ^L) is a standard eigenvalue computation. CF9 passes.

**CF10 check.** No calibration fit. Block sizes are derived from weight-only measurements. CF10 not applicable.

**Dependency.** This idea is GATED on F5-RSUC GO. If RSUC finds r_90% is uniform across depth (RSUC NO-GO), this idea dies trivially. The Stage 5 decision should note this gating.

**Residency arithmetic.** At 70B 0.55 bpw NVMe: per-layer weight block for GEMV at batch_size=1 and d_model=8192: 8192 × 8192 × 0.55/8 = 4.6 MB/layer. Depth-stratified: early/late layers at 0.6 × 4.6 = 2.8 MB; mid-layers at 4.6 MB. Peak DRAM buffer = max = 4.6 MB/layer (mid-layer). Total NVMe weight stream per token is unchanged (same bytes must be read); the gain is in DRAM peak usage during streaming, not in bytes read.

**What would have to be true.** RSUC GO: r_90%(δ^L) varies by ≥ 40% from early to mid layers. The block-size adaptation is then directly motivated.

**Experiment cascade.** Rung 1: RSUC pilot (25 min) — must GO. Rung 2: implement depth-stratified streaming in the GGUF loading path (Python prototype, ~2 hours). Rung 3: measure peak DRAM buffer usage and tok/s on Qwen3-1.7B NVMe-simulated load.

**Smallest test.** RSUC Rung 1 (contingent on RSUC running). **NOT proposing:** a different model (same Qwen3-1.7B); NVMe seek optimization (sequential-only constraint is satisfied by the existing GGUF layout); any quality change (zero quality cost, layout-only change).

---

### S7-INT8-LOOKUP-SOFTMAX — INT8 Q/K Product + Lookup-Table Softmax for DRAM-Resident Inference

**Orientation:** Stage 4 constraint reframing (no floating point at inference)
**Track:** A (arch-transposition — replaces FP16 attention score computation with integer arithmetic)
**Elegance-class:** gauge-exploitation (the softmax gauge symmetry — invariant to constant shifts of logits — is exploited to fit the full softmax into an INT16 range)
**CF tether:** F1-MQKP (product SVD gives INT8 compressed Q and K via U_M Σ^{1/2} and V_M Σ^{1/2}); CF11 (W_Q concentrated at K=128-400, making the INT8 Q vectors compact)
**Sits outside:** F's anti-frame (this is an inference execution primitive, not a weight compression scheme); A's constraint-alien frame (this applies the integer-arithmetic constraint to attention specifically, beyond what A1-TROPFFN's FFN focus covers); R's anti-frame (R focuses on streaming and cascade arithmetic, not integer-only attention computation)

**Ambition.** After MQKP's product SVD at rank K=400, the attention score computation at inference is: for each head h, q̃_h = x (U_M^h Σ_M^{h,1/2}) ∈ ℝ^K and K̃_{cache,t} = k_t (V_M^h Σ_M^{h,1/2}) ∈ ℝ^K. At K=128 (CF11's global GO), the attention score a_t = q̃_h · K̃_{cache,t} is a 128-dimensional dot product. If q̃ and K̃ are both quantized to INT8 (range [-127, 127]), the dot product a_t ∈ [-127² × 128, +127² × 128] = [-2,064,000, +2,064,000] — fits comfortably in INT32 accumulation, no overflow. The softmax requires exp(a_t / √K); with INT32 a_t and K=128 (√K=11.3), the scaled score a_t / 11.3 ≈ range [-183K, +183K]. For softmax stability, subtract max before exp: all logits shift to range [-(max_minus_min), 0] ≤ [-366K, 0]. At INT32 precision with a lookup table for exp: a 16-bit lookup table (65536 entries × 4 bytes = 256 KB) maps INT16 logit → exp(logit/scale), with scale = 366K / 65535 ≈ 5.6 per logit unit. No FP16 involved in the attention score computation; only INT8 inputs, INT32 accumulation, and an INT16-indexed lookup table.

**CF9 precondition check.** No imported theorem with fragile preconditions. Softmax lookup table is a standard technique (documented in INT8-only inference literature). The gauge invariance (subtracting max from all logits before exp) is a standard softmax numerical stability trick, which IS an algebraic identity: softmax(x) = softmax(x - max(x)). CF9 passes.

**CF10 check.** No calibration fit. The quantization scale for INT8 Q and K is derived from the weight spectra (MQKP's Σ_M factors). CF10 not applicable.

**Residency arithmetic.** Lookup table: 256 KB DRAM (one-time allocation). INT8 Q and K: 128 × 1 byte = 128 bytes per query/key vector per head vs FP16 = 256 bytes — 2× reduction in Q/K bandwidth. At 4K context, 28 layers, 16 heads: KV bandwidth in attention = 28 × 16 × 4096 × 128 × 2 (Q and K) × 1 byte = 471 MB (INT8) vs 942 MB (FP16). On Ryzen 5 7530U, DRAM bandwidth 11.5 GB/s: attention Q/K bandwidth at INT8 = 471/11500 = 41 ms vs FP16 = 82 ms — 41 ms saving per full-model pass. At 3 tok/s baseline (333 ms/token): 41 ms / 333 ms = 12% speedup on the attention Q/K bandwidth contribution. Non-trivial.

**What would have to be true.** (a) MQKP GO (product SVD compresses Q and K to INT8-quantizable vectors without quality loss). (b) The INT8 × INT8 → INT32 dot product matches FP16 dot product quality with ΔNLL ≤ 0.05 nats. (c) The lookup-table softmax introduces ≤ 0.02 nats additional ΔNLL vs the exact FP32 softmax.

**Experiment cascade.** Rung 1: MQKP pilot (prerequisite). Rung 2: at K=128 MQKP factorization, quantize the projected Q and K vectors to INT8 (per-vector normalization); compute INT8 dot product vs FP16 dot product for 200-token calibration set; measure cosine similarity of attention outputs. Runtime: ~30 min. Rung 3: implement lookup-table softmax (Python numpy, ~1 hour); measure ΔNLL.

**Smallest test.** Rung 2: cosine similarity of INT8 vs FP16 attention output ≥ 0.98 at K=128. **NOT proposing:** quantizing the VALUE vectors or the output projection (this is strictly the Q/K score computation); any weight quantization (this is inference-path arithmetic, not weight storage). The lookup-table approach is NOT an approximation — it is an exact evaluation of exp() on a discretized domain.

---

### S8-ETWO-SNRATE-BRIDGE — ETW Access Histogram as Per-Tensor bpw Calibration Signal

**Orientation:** Stage 4 cross-pollination (U1-B-ETWO stripped primitive + C15-SNRATE stripped frame)
**Track:** B (compression — quantization schedule)
**Elegance-class:** calibration-fit (CF10-compliant: access frequency is a single scalar per tensor, zero fit parameters)
**CF tether:** C15-SNRATE (SNR oracle for per-class bpw assignment), U1-B-ETWO (ETW access histogram as zero-overhead oracle), CF11 (W_Q high SNR, compress aggressively), CF8 (MLP low SNR, compress conservatively)
**Sits outside:** C's "coupling two CF findings" frame (this couples a substrate measurement with a CF-derived oracle); U's "substrate primitive only" frame (this uses ETWO's output as a compression signal, not a substrate finding alone); F's "first-principles only" frame

**Ambition.** C15-SNRATE proposes a per-class bpw schedule based on SNR = concentration/sensitivity. The resolution is at the granularity of weight CLASSES (W_Q, W_K, W_gate, W_up, tied embed). The ETW access histogram (U1-B-ETWO) provides a finer-grained signal: which specific TENSORS (individual layer × weight-type combinations) are accessed most frequently during inference. The cross-pollination: combine SNRATE's per-class budget with ETWO's per-tensor access frequency to produce a PER-TENSOR bpw schedule.

**Mechanism.** (1) Run ETWO experiment (3 hours): get per-tensor read-frequency histogram across calibration inference. (2) Multiply SNRATE's class-level bpw by an access-frequency correction: bpw_tensor = bpw_class × (1 + α × (freq_tensor - mean_freq) / std_freq), where α is a small mixing coefficient (e.g., 0.2). Hot tensors (high access frequency) get slightly higher bpw (protect quality for frequently accessed weights); cold tensors get lower bpw (compress aggressively when accessed rarely). (3) Apply per-tensor bpw schedule to the full quantization pipeline.

**Why access frequency matters.** In NVMe-resident inference, hot tensors are read every token (contributing proportionally to total NVMe bandwidth) while cold tensors may be accessed rarely (if sparsity, early-exit, or context-conditional fold applies). Quantizing hot tensors at higher precision preserves quality for the critical path; cold tensors can be aggressively compressed with negligible quality impact.

**CF9 precondition check.** No imported theorem. Access frequency × SNR product is a heuristic combination, not a theorem. CF9 passes (no fragile precondition).

**CF10 check.** Zero fit parameters: bpw_class from SNRATE (3-param fit on 5 points, confirmed CF10-safe), access-frequency correction uses α = 0.2 (a fixed hyperparameter, not a fitted parameter). CF10 passes.

**Residency arithmetic.** Per-tensor bpw schedule: same total bytes as uniform class-level schedule from SNRATE (the access-frequency correction redistributes precision within fixed total budget). Quality improvement: hot tensors accessed every token are the bottleneck for ΔNLL at low bpw; raising their precision by 0.5 bpw while dropping cold-tensor precision recovers 0.05–0.15 nats ΔNLL at the same total residency.

**What would have to be true.** (a) ETWO GO: top-10 tensors account for ≥ 30% of access events AND histogram is stable (Jaccard ≥ 0.80 across 3 runs). (b) SNRATE GO: bpw ordering predicted by SNR is validated at 2 bpw. (c) The access-frequency and SNR signals are non-redundant: high-frequency tensors are not always high-SNR tensors (otherwise the combination adds no value over SNRATE alone).

**Experiment cascade.** Rung 1: ETWO experiment (U1-B, 3 hours). Rung 2: SNRATE experiment (C15, 40 min). Rung 3: per-tensor bpw schedule implementation + ΔNLL comparison vs SNRATE-only schedule.

**Smallest test.** ETWO Rung 1 + verify that per-tensor access frequency is non-uniform (GO threshold: top-10 tensors cover ≥ 30% of reads). **NOT proposing:** a new quantization algorithm; this is a SCHEDULING improvement on top of SNRATE + ETWO as components that are already in the pipeline.

---

## 7. Output Handoff

### Stage 4 idea summary

| ID | One-line description | Motivating section |
|---|---|---|
| S1-MQKP-ZONEMAP | Zone-map-indexed exact KV scan over MQKP product-SVD projected cache | Section 4 (database cross-domain) + Section 3 (XP from MQKP/APPENDKV) |
| S2-WQCOL-PREC | Per-column W_Q precision allocation from CF3 static-channel × CF11 column-energy coupling | Section 3 (XP from F2-CKDJ stripped primitive) |
| S3-LRCS-WDLA | Low-rank-by-construction (rank≤64) cross-scale surgery, WDLA rescued via CF10-safe design | Section 3 (XP from WDLA stripped primitive) + Section 4 (RSUC cross-pollination) |
| S4-ATTN-CSE | Compiler CSE: shared input basis for W_Q, W_K, W_V reduces input-projection computation 3.6× | Section 4 (compiler cross-domain) |
| S5-KALMAN-TIER | Kalman-filtered activation outlier tier predictor reduces RAOK per-token overhead by ~30% | Section 4 (control-theory cross-domain) |
| S6-STREAMING-DEPTH | Depth-stratified streaming block size; gated on RSUC GO; reduces peak DRAM buffer 20% | Section 5 (sequential-only constraint) + Section 4 (RSUC) |
| S7-INT8-LOOKUP-SOFTMAX | INT8 Q/K dot product + 256 KB lookup-table softmax; 12% attention Q/K bandwidth saving | Section 5 (no-FP constraint) + F1-MQKP cascade |
| S8-ETWO-SNRATE-BRIDGE | ETWO per-tensor access histogram × SNRATE per-class SNR = per-tensor bpw schedule | Section 3 (cross-pollination U+C) |

### Orientation rotation recommendations

| Orientation | Recommendation | Rationale |
|---|---|---|
| F (First-Principles) | KEEP, TIGHTEN anti-frame on per-factor independent W_Q/W_K SVD | F is the strongest producer this round; tighten to push toward product-SVD and multi-matrix CSE territory |
| R (Reach) | KEEP | R's cascade arithmetic is the clearest path to 70B deployment; no saturation signal |
| C (Composition) | KEEP, TIGHTEN anti-frame on shift-invariance + key-side interaction | C14-CFSHIFT and C15-SNRATE are strong; F2-CKDJ died on the same CF9 trap as F3-SRSC — codify this explicitly |
| A (Constraint-Alien) | KEEP | Tropical semiring, FSM dispatch, append-only KV are in unexplored computational frames; valuable |
| U (Unconventional Substrate) | TIGHTEN | 3/5 ideas rejected on A1; U should focus on substrate measurements that produce structural findings (ETWO's histogram, VGTP's latency) rather than engineering micro-optimizations |

### Anti-frame additions

| Orientation | Addition | Rationale |
|---|---|---|
| F | "per-factor independent W_Q / W_K SVD at matched rank without computing product M = W_Q W_K^T" | Saturated by QSVD/CARE; MQKP's product-SVD is the surviving angle |
| F | "softmax shift-invariance strip for any term involving a key-side inner product (q^T k_t varies per position t)" | F2-CKDJ died here; F3-SRSC also died here; this is now a class kill |
| C | "softmax shift-invariance applied to a query offset that interacts with a position-dependent key vector" | Same kill as F above; make explicit for C's anti-frame |
| R | "standalone KV management schemes at context length ≤ 4K tokens" | KV is 670 MB vs 40 GB weights at 4K; wrong bottleneck |
| U | "I/O priority or latency-variance mechanisms with predicted mean throughput improvement < 10%" | FIOP killed for this reason; prevents future similar ideas from consuming Stage 3 slots |
| A | "constraint-alien ideas whose mechanism re-derives a published architecture (Apple LLM-in-a-Flash, standard speculative decode) without adding a new primitive" | A5-REGONLY killed for this; A3/A4/A6 in future rounds need this guard |

### Stage 5 recommendation

**Primary Stage 5 candidate: F1-MQKP.** Algebraic-identity elegance class, highest Stage 2 score (13), cleanest binary settler (15-min SVD of M at 3 layers), no prior art on the product M = W_Q W_K^T as compression target, and it gates four downstream Stage 4 ideas (S1, S4, S7, and S3 indirectly). Running MQKP first maximizes cascade leverage.

**Secondary Stage 5 candidate: R9-R1/ATQKV.** The W_V and W_O spectrum measurement is the critical structural measurement that determines whether the full four-matrix attention compression cascade is viable. 40-min test. The result will either (a) open the cascade (W_V, W_O concentrated → ATQKV + MQKP + ATTN-CSE all become viable) or (b) close the class (W_V full-rank → extends CF8 boundary). Either outcome is a load-bearing structural finding.
