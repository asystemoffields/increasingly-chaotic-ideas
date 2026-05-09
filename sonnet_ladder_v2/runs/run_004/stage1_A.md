# Stage 1 — Orientation A (Constraint-Alien) — Run 004

Orientation: A — Constraint-Alien
Run: 004
Ideator: Sonnet 4.6

KILL_LIST note for this run: v2-CHEAP-TEST-001 kills the entire cross-layer W_Q shared-subspace class. Ideas below that use gauge-symmetry / symmetry-group constraints must produce mechanisms whose residency claim does NOT depend on cross-layer basis sharing. Each uses a distinct constraint.

---

## A-TROP — Tropical Weight Dispatch (A-004-1)

**Track B — compression (mechanism) / Track A cross-domain framing**

### Mechanism

**Constraint: No continuous arithmetic during weight selection and routing; only max and addition (tropical semiring) are permitted for dispatch decisions.** This rules out softmax, dot-product sorting, and any floating-point comparison in the selection phase — roughly 80% of standard dynamic-sparsity and predictor mechanisms become inadmissible.

In the tropical semiring (ℝ ∪ {-∞}, max, +), matrix-vector products compute argmax rather than sums. The proposal: represent the pre-quantized W_down (8192 × 2048 on a 70B-class model) as a tropical matrix. During inference, instead of computing the full W_down · h_mlp (a dense GEMV), compute the tropical product (W_down ⊗_trop h_mlp)[i] = max_j(W_down[i,j] + h_mlp[j]) for each output row i. The argmax over j identifies the dominant input channel for row i with no multiply. Use this dominant-channel index as a routing key: for rows whose dominant-channel Jaccard stays stable across calibration tokens (analogous to CF3's K=0.1% stable tier), pre-cache that row with its top-k additive correction. The result is a two-tier scheme: (tier-1) stable-dominant rows stored as a single channel index + scalar coefficient; (tier-2) dynamic rows stored at full INT4. The tropical product itself requires only integer add + max, executable entirely in SWAR (SIMD within a register) on Zen-3 without FP pipeline pressure.

This is distinct from CLUBS/RSTD: it exploits channel dominance in the activation distribution, not rank structure in the weight matrix. CF8 (MLP weights full-rank) is not violated — the mechanism operates on the activation channel identity, not on a rank-truncated weight matrix.

### Residency arithmetic

Qwen3-1.7B-Base: W_down is 28 × (6144 × 2048) = 353M parameters. At bf16: 706 MB. At INT4: 176 MB. Tier-1 stable rows: if 5% of W_down rows have a dominant channel (conservative, given CF3 K=0.1% Jaccard=0.718 as an analogy): 0.05 × 353M × 4 bits + overhead ≈ 22 MB (index + one INT8 scalar per row). Tier-2 remaining 95%: INT4 as baseline. Net saving over flat INT4: the tier-1 rows saved (176 MB × 0.05 = 8.8 MB). Modest on 1.7B but at 70B scale (W_down ≈ 14.4 GB at INT4, 28×64): tier-1 stable fraction × 14.4 GB potential. The load-bearing experiment is verifying whether stable-dominant rows exist in W_down's activation distribution at a useful fraction. If the stable fraction reaches 20%, saving is ~2.9 GB at 70B scale.

### Novelty gloss

Closest kill-list item: CLUBS (cross-layer basis on W_up) — structurally different; TROP operates on W_down activation-side channel stability, not weight-matrix rank. Closest published method: LLM.int8() per-channel outlier isolation — different; that method identifies outlier activation channels statically; TROP uses tropical max-dominance to identify per-row stable input channels dynamically at calibration time, then stores only the dominant index for stable rows. Tropical algebra applied to weight-selection dispatch has no direct prior in the LLM compression literature we've encountered.

### Smallest experiment

**Claim**: In Qwen3-1.7B, at least 10% of W_down output rows have a dominant input channel j* such that across calibration tokens, argmax_j |h_mlp[j]| = j* with frequency ≥ 0.85 (tropical-stability threshold).

- Measure: run 200 calibration tokens through Qwen3-1.7B, record h_mlp (post-SwiGLU) per layer. For each W_down output row i, compute argmax_j(W_down[i,j] × h_mlp[j]) (standard product argmax, not tropical, since h_mlp can be negative) across tokens. Compute frequency of modal argmax.
- Runtime: ~30 min on Ryzen 5 7530U (matrix-stats pass, no grad).
- GO threshold: ≥10% of rows stable at ≥0.85 token frequency in ≥3 layers.
- NO-GO structural finding: establishes that W_down row outputs are fully distributed across input channels (no concentrated activations), closing this axis and informing RAOK's three-tier codebook design.

### Primary risk

h_mlp (post-SwiGLU) is token-dynamic at K=1% (CF3 analogy), so the dominant channel may rotate per token — making stable-dominant rows rare. Mitigation: run the measurement first; if stable fraction < 5%, pivot to using tropical max as a cheap approximation kernel for RAOK's tier-2 rows (complementary role rather than primary mechanism).

---

## A-SEED — 10 MB Seed Weight Synthesis (A-004-2)

**Track A — arch-transposition**

### Mechanism

**Constraint: The model's weights must be derivable from a 10 MB seed procedure — no weight matrix may be stored at its full trained size.** This constraint eliminates every standard compression scheme below ~30× (Qwen3-1.7B at bf16 is 3.4 GB; 10 MB is 340× compression), forcing a generative-synthesis frame the literature does not cover.

The mechanism: treat each weight matrix W as a deterministic function of (layer index ℓ, matrix role r, a low-dimensional parameter vector θ_{ℓ,r}). The seed stores only the parameter vectors and a universal synthesis function F(ℓ, r, θ). F is a structured hypernetwork: it must reproduce W to within a quality bound δ (ΔNLL ≤ 0.5 nats). The structural grounding: CF11 and the AQFKV results show that W_Q matrices have r_99/d ≈ 0.63 and head-redundancy at 16:1. This establishes that at least one weight class is not randomly structured — it has regularity that a small generative procedure might encode. W_gate and W_up are full-rank (CF7/CF8/CF5), but the SDZC finding (CF6) shows layer-1 has anomalous structure (36% foldable), suggesting layer-indexed structure does exist. The synthesis function is fit on a per-model basis (offline) using a nested loop: minimize ‖F(ℓ, r, θ_{ℓ,r}) − W‖_F over θ with dim(θ) = 256 per matrix. At 28 layers × 7 matrix roles × 256 floats × 4 bytes = 200 KB for θ vectors; remaining 9.8 MB budget for the fixed F parameters (a tiny MLP). This is a genuinely radical frame: F is a neural network that knows "what a transformer weight matrix should look like" as a function of its position, and θ is what distinguishes this specific trained model from a random one.

Does not depend on cross-layer W_Q subspace sharing (killed by v2-CHEAP-TEST-001). Instead exploits the POSITIONAL structure (ℓ, r) as the coordinate, not shared-subspace alignment across layers.

### Residency arithmetic

10 MB total. If synthesis can be run in 4 ms per matrix on Ryzen (tiny MLP forward pass), and Qwen3-1.7B has 28×7 = 196 matrices, total synthesis at model load = 196 × 4 ms ≈ 0.8 s (one-time overhead). At inference: regenerate matrix on demand (streaming); F is 10 MB, fits in L3 cache. This eliminates the 17-40 GB NVMe residency problem entirely — the loaded model is 10 MB + activations. The audacity: if synthesis fidelity within 0.5 nats is achievable, this is the 340× residency reduction that makes the floor disappear. P(end-to-end success) is low (~5%); the structural finding from the experiment (how well can a hypernetwork synthesize Qwen3-family weights?) is itself novel and informative regardless.

### Novelty gloss

No direct kill-list neighbor. Closest published: LoRA / adapter parameterization — different; LoRA is a low-rank residual on a frozen pretrained matrix, requiring the full original matrix as the base. The closest conceptual neighbor is HyperNetworks (Ha et al., 2016) which generates weights for a small target network from a larger generator; the inverse direction (small generator synthesizing large target without access to the original) is much harder and, for billion-parameter language models, has no published positive result we know of. The 10 MB constraint is what makes this a novel experiment: it forces derivation of the synthesis quality bound.

### Smallest experiment

**Claim**: A 256-parameter θ vector per matrix, fit by least-squares minimization of ‖F_fixed(ℓ, r, θ) − W‖_F on a single layer of Qwen3-1.7B, achieves cos-similarity ≥ 0.85 between the synthesized and actual W_Q matrix when F_fixed is a two-layer MLP of width 512 (≈ 2 MB).

- Runtime: train F on W_Q for all 28 layers of Qwen3-1.7B, ~2 hours on Ryzen (offline fitting, CPU-only, einsum).
- GO threshold: held-out W_Q (from Qwen3-0.6B, not used in fitting) cos-sim ≥ 0.75 AND within-training cos-sim ≥ 0.90.
- NO-GO structural finding: confirms Qwen3 weight matrices are not hypernetwork-synthesizable at 256 dims, establishing a lower bound on the effective Kolmogorov complexity of trained transformer weights — itself a first in this empirical pipeline.

### Primary risk

Trained weights may be effectively incompressible at 256 dims (consistent with CF8 full-rank findings on MLP matrices). Mitigation: separate synthesis experiments per matrix role; if W_Q synthesizes well (lower r_99/d per CF11) and MLP weights do not, the hypernetwork is still useful as a W_Q-specific compressor within a broader architecture.

---

## A-ALOG — Append-Only Log Attention (A-004-3)

**Track A — arch-transposition**

### Mechanism

**Constraint: All state during inference is append-only; no weight or activation value may be mutated in place.** This rules out in-place KV cache update, weight-swapping, and all standard KV eviction / compression policies (eviction is a mutation/deletion). Approximately 70% of standard KV-compression and speculative-decode mechanisms become inadmissible.

The mechanism exploits what append-only forces. Standard KV caches grow and get evicted. Under append-only, they grow only. The constraint turns attention into a log-structured read: queries are issued against a monotonically growing immutable log of (K, V) pairs. The structural insight: if the attention softmax weight for old entries decays geometrically (as it does in position-encoded models like Qwen3 with RoPE), then old entries contribute exponentially less. Rather than evicting, store the KV log as a compressed ANS-encoded stream (keys quantized to INT4, values to INT4, accumulated as a write-once block journal). At query time, decompress the last W entries (window), compute attention over the window, and retrieve the contribution from older entries via a coarse summary (a running sum-V weighted by approximate geometric decay). This is structurally a hierarchical log: hot window (last 512 tokens, INT8 full precision), warm block (512–4096 tokens, INT4), cold journal (4096+, single summarized vector per 512-token block). The cold journal is idempotent and never mutated. Crucially, all three tiers are write-once — the constraint is maintained.

Not a KV eviction scheme. Not standard sliding-window attention. The structural novelty: the append-only constraint forces derivation of a geometric-decay attention approximation that is provably correct on the append-only model. The cold journal approximation's error is bounded by the RoPE decay rate (empirically measurable).

### Residency arithmetic

At 4K context, Qwen3-1.7B: KV cache = 28 layers × 2 × 16 heads × 128 dim × 4096 tokens × bf16 = 28 × 2 × 16 × 128 × 4096 × 2 bytes ≈ 1.5 GB. Under hot/warm/cold: hot (512 tokens, INT8): 28 × 2 × 16 × 128 × 512 × 1 = 59 MB. Warm (3584 tokens, INT4): 28 × 2 × 16 × 128 × 3584 × 0.5 = 205 MB. Cold (0 tokens at 4K context): 0. Total ≈ 264 MB vs 1.5 GB baseline — 5.7× KV compression. At 128K context (the long-context target where KV compression matters per CF): hot 59 MB + warm 720 MB (3584 at INT4) + cold summary 28 MB (one 2048-dim vector per 512-token block × 251 blocks × 28 layers × 2 × 2 bytes) = 807 MB vs 48 GB baseline — 60× reduction. Total residency at 128K context: weights (17 GB at INT4) + KV (807 MB) = 17.8 GB. Still over target, but pairs with weight compression.

### Novelty gloss

Closest kill-list item: CLASE (cross-layer KV aliasing INT4 deltas, R3-A deferred) — different; CLASE uses delta compression between layers; ALOG uses tier-partitioned append-only log with geometric-decay cold summary. Closest published: StreamingLLM / KV eviction policies — fundamentally different: ALOG never evicts, relies on append-only monotonic growth with a coarse cold summary. Log-structured storage system design (LFS, RocksDB) applied to KV cache attention is not in the published LLM inference literature.

### Smallest experiment

**Claim**: Geometric-decay attention (RoPE position encoding implies that attention to tokens at distance d decays as λ^d for measurable λ) is a sufficient approximation for the cold-journal tier at 128K context, with NLL degradation < 0.3 nats vs full-precision attention.

- Measure: on Qwen3-1.7B, run 100 long-context completions (4K context window), compute attention entropy at each layer as a function of relative position distance. Fit decay curve λ. Then simulate cold-journal tier: replace attention to tokens >2048 positions back with single summary vector; measure NLL delta.
- Runtime: ~3 hours on Ryzen (100 × 4K context forward passes).
- GO threshold: mean NLL delta ≤ 0.3 nats; λ fit R² ≥ 0.80.
- NO-GO finding: establishes whether RoPE attention is actually geometric-decay structured (empirically unknown for Qwen3 family); if not, surfaces the actual long-range attention distribution shape as a novel CF for this pipeline.

### Primary risk

Qwen3's RoPE attention may not be well-approximated by geometric decay — the attention pattern may be content-dependent and non-monotone in distance, making the cold-journal summary lossy beyond the quality budget. Mitigation: the decay-curve measurement (smallest experiment) is the gate; if R² < 0.80, reframe as a hybrid where cold journal uses an actual compressed sparse matrix rather than a summary vector (still append-only, higher residency but correct).

---

## A-REGM — Register-Only Residual Decode (A-004-4)

**Track A — arch-transposition**

### Mechanism

**Constraint: No RAM during the decode phase — only registers, L1 (32 KB), L2 (512 KB), L3 (16 MB), and NVMe may hold state.** RAM allocation during the forward pass is prohibited; activations must fit in the CPU cache hierarchy, and weights are streamed from NVMe. This constraint eliminates all standard in-RAM buffering of activations (~90% of standard inference paths).

The mechanism: Qwen3-1.7B residual stream at inference is d_model = 2048 floats × 4 bytes = 8 KB per token. At INT8, 2 KB. This fits in L1 (32 KB has room for one full residual state plus two layer-norm scales). The forced design: each transformer layer operates as a streaming kernel where the residual vector lives in L1, weight columns of each matrix are streamed from NVMe one column at a time (GEMV outer product), and the partial sum accumulates in registers. This is an extreme form of NVMe-streaming inference: the architectural invariant is that the residual state is the ONLY non-NVMe state. All KV cache, all weight buffers, all activations are streamed from NVMe or computed on the fly.

Feasibility anchor: Zen-3 Ryzen 5 7530U has 16 MB L3 per CCD. At INT8, a single attention head's W_Q (128 × 2048 = 256 KB) fits in L3. A single MLP row (8192 floats INT4 = 4 KB) fits in L1. The constraint forces a specific compute-access schedule that can be analyzed as an I/O-optimal streaming schedule: total weight bytes / NVMe bandwidth = minimum token latency. At 70B INT4 (17 GB) and 3 GB/s NVMe: 17 GB / 3 GB/s = 5.7 s/token. This is the hard floor — the constraint makes it explicit and forces engineering at that floor, not above it.

Not CLUBS/RSTD (weight-matrix structure). Not ALOG (KV cache). This is a scheduling constraint: forces derivation of the minimum-residency compute schedule for transformer inference, which is a novel analysis object.

### Residency arithmetic

RAM residency = 0 by constraint (only L3 and below). NVMe residency = full model file (no compression assumed; compression is orthogonal and multiplicative). Token latency = weight bytes / NVMe bandwidth. At 70B INT4: 17 GB / 3 GB/s = 5.7 s/token (NVMe floor), or at 700 MB/s cold: 24 s/token. The engineering payoff is not residency reduction but **clarity of the bottleneck** — by making RAM = 0 a design constraint, the NVMe bandwidth becomes the only variable, and all other optimizations are exactly priced against it. The structural finding from the smallest experiment is the measured per-token NVMe read bytes on actual Qwen3 inference, which is v2's re-derivation of CF13 (in-flight, currently unverified).

### Novelty gloss

Closest kill-list item: INPOLT (NVMe prefetch with phase-ordered layer tiling, Track A R3 deferred) — different; INPOLT reorders the weight layout to improve prefetch; REGM eliminates the RAM budget entirely and derives the minimum-latency schedule from first principles. Closest published: Apple "LLM in a Flash" (arXiv:2312.11514) uses NVMe offload with predictive prefetch. Different: Apple still uses RAM as a staging buffer; REGM eliminates RAM and asks what happens when the cache hierarchy is the only staging. The analysis of register-only residual streaming for transformer inference has no published instance we know of.

### Smallest experiment

**Claim**: Qwen3-1.7B inference (single token, greedy) can be executed with < 512 MB peak RAM allocation while achieving correct output (NLL within 0.05 nats of unconstrained baseline) by routing all activation temporaries through L3-resident buffers.

- Measure: implement a minimal single-layer test: pre-load one W_Q matrix (256 KB INT8) into L3 via mlock, implement GEMV streaming one column at a time, measure peak RSS via /proc/self/status or Windows working set API.
- Runtime: ~4 hours to implement + measure on Ryzen.
- GO threshold: peak RSS ≤ 512 MB AND output matches.
- NO-GO finding: establishes the minimum practical RAM floor for transformer inference, producing a precise hardware-constraint characterization that informs the NVMe-offload line (INPOLT, NVPF, and related).

### Primary risk

OS memory allocator may not permit sub-512 MB RAM operations for a Python/C process running a 1.7B model (Python interpreter alone uses ~100 MB; PyTorch initialization uses more). Mitigation: implement as a standalone C binary that links against ggml directly, bypassing Python allocator. Already feasible in ik_llama.cpp build environment.

---

## A-FSMT — Finite-State Decode Machine (A-004-5) [FREE SWING]

**Track A — arch-transposition**

### Mechanism

**Constraint: The inference loop's internal state must be representable as a finite automaton with N ≤ 65536 states (16-bit state word), excluding the KV cache and position counter.** This rules out continuous residual-stream representations — the effective dimensionality of the decoder's INTERNAL STATE must be discretized to fit in 2^16 entries. Approximately 95% of standard transformer decode mechanisms become inadmissible (the residual stream is 2048-dim continuous; any parameterization of that space in 65K states is lossy).

The mechanism exploits the following: at each decode step, the residual stream h_L ∈ ℝ^2048 is quantized to a discrete state s_L ∈ {0, ..., 65535} via a learned quantizer Q (calibration-fitted, 65K centroids of dim 2048). The next-state function T(s_L, token) = s_{L+1} is implemented as a look-up table: 65K states × 50K token IDs × 2 bytes = 6.5 GB table (too large). Instead, factorize: T(s, tok) = Q(Embed[tok] ⊕ Centroid[s]) where ⊕ is element-wise addition and Q is the quantizer. This reduces the full transformer to: token embedding lookup → add to current centroid → quantize → next state. No matrix multiplications during inference (only 2048-dim addition + nearest-centroid lookup). At 65K centroids × 2048 dim × bf16: 268 MB for the centroid table. Decode cost: 2048-dim add + ANN lookup in 268 MB index — approximately 1 ms/token on Ryzen.

This is a radical arch-transpose: the transformer layers are replaced by a learned FSM transition function. The learned function is trained offline by running the actual transformer and recording (residual state quantized, token, next residual state quantized) transitions. The FSM is the distillation target. It cannot match the full transformer's quality, but the experiment measures HOW CLOSE it can get, which is a novel measurement.

[FREE SWING] — no CF anchor required; this is a speculative structural probe.

### Residency arithmetic

Centroid table: 65K × 2048 × 2 bytes = 268 MB (fits in RAM on 7.28 GiB target). No weight matrices at inference. Token embedding: 50K × 2048 × 2 bytes = 205 MB (static, load once). Total model: 268 + 205 ≈ 473 MB. A 70B-class FSM distillation would require more centroids (N scales with model capacity), but even at N = 1M centroids: 1M × 2048 × 2 bytes = 4.1 GB — still fits in RAM. Decode: O(log N) ANN search per token ≈ sub-ms. This would be the residency argument if quality is within budget.

### Novelty gloss

No kill-list neighbor. Published closest: LUT-based neural networks (LUTNet, PolyLUT) — different; those replace activation functions with LUTs, not the entire transformer state with an FSM transition. VQ-VAE-style discrete state representations exist in generative modeling but not as inference-time FSM decoders for LLMs. The experiment is a first: can a 65K-state FSM approximation of Qwen3-1.7B produce coherent text, and if not, at what N does it approach acceptable quality?

### Smallest experiment

**Claim**: A 4096-centroid FSM approximation of Qwen3-1.7B, trained on 1K calibration sequences, achieves NLL ≤ 2× baseline (perplexity ≤ 23 vs 11.78 baseline) on held-out WikiText.

- Measure: run 200 calibration sequences through Qwen3-1.7B; record residual states at layer 14 (midpoint); k-means cluster to 4096 centroids; build transition matrix from (centroid_t, token_t) → centroid_{t+1}; evaluate approximate NLL on 100 held-out sequences by stepping the centroid chain.
- Runtime: ~6 hours (200 forward passes + k-means on 200K × 2048 activation matrix on Ryzen).
- GO threshold: held-out NLL ≤ 23 (2× baseline 11.78).
- NO-GO finding: establishes a lower bound on the Kolmogorov complexity of the Qwen3-1.7B residual stream — how many distinct residual states exist in practice. This is a novel empirical measurement even on NO-GO.

### Primary risk

4096 centroids is far too few to capture the continuous residual stream distribution — expected NLL >> 2× baseline on the smallest experiment. Mitigation: the centroid count vs NLL scaling curve is itself the load-bearing output; cascade rung 2 measures N=65536 and N=1M to find the knee.

---

## A-GFWV — Gauge-Fixed W_V Folding (A-004-6)

**Track A — arch-transposition (algebraic fold)**

### Mechanism

**Constraint: The computation graph must algebraically simplify under the residual-stream gauge rotation symmetry, and the simplified form must be used at inference.** Specifically: the residual stream has a gauge freedom — multiplying all residual-stream activations by an invertible matrix G and all weight matrices by G^{-1} (from the appropriate side) leaves model outputs unchanged. The constraint requires picking G to expose a structural simplification, then deploying in G-coordinates.

This is distinct from the killed C1 gauge-symmetry class (cross-layer W_Q subspace sharing). That class looked for shared subspaces across layers; this class uses gauge freedom to fold W_V into W_O within a single head-layer pair. The derivation: for a single attention head, the output is Out = (softmax(QK^T / √d) · V) · W_O = A · (h W_V) · W_O where A is the attention weight matrix (a function of Q, K). The product W_V · W_O ∈ ℝ^{d × d} is a composed linear map. With d=128 per head and d_model=2048, this is 128×2048. AQFKV measured W_Q at r_99/d ≈ 0.63; W_V and W_O spectra are UNTESTED (noted in CF11 follow-up list). If W_V · W_O is low-rank (say r_99 ≈ 0.5 × d), the composed operator can be stored as two smaller matrices with no separate W_O — one multiply instead of two, and lower residency. The gauge-fixing step: choose G to diagonalize the joint W_V · W_O spectrum, then store in G-basis where the joint operator is minimal. This is NOT the same as joint low-rank decomposition of W_V and W_O independently (that would inherit CF11's per-head rank limit); it's a decomposition of their PRODUCT.

Anti-kill-list distinction: v2-CHEAP-TEST-001 kills cross-layer W_Q stacking. This idea is within-layer W_V × W_O product folding with gauge-fix. Single-layer, single-head, different matrices entirely.

### Residency arithmetic

Qwen3-1.7B: per-layer W_V is 16 heads × 128 × 2048 (in GQA: 8 KV heads × 128 × 2048) = 4 MB at bf16. W_O is d_model × (n_heads × d_head) = 2048 × 2048 = 8 MB. Per-layer total: 12 MB. 28 layers: 336 MB for {W_V, W_O} across the model. If joint W_V·W_O product is rank-k with k = 0.5 × d_head = 64 per head: store as two factors of shape 128 × 64 and 64 × 2048 per head. Memory per head: (128×64 + 64×2048) × 2 bytes = 277 KB vs 128×2048 + 2048×2048/16_heads = 656 KB. Per-layer saving: (16 heads) × (656 - 277) KB = 6 MB saved out of 12 MB = 50%. Model-wide: 168 MB saved out of 336 MB. Total model impact: modest on 1.7B, but scales to 70B where {W_V, W_O} is 10+ GB.

### Novelty gloss

Closest kill-list item: R3-A AQFKV (W_Q rank sweep) — different matrix family. AQFKV measured W_Q and W_K; W_V × W_O folding is explicitly listed as "untested, cheap parallel measurement" in the AQFKV follow-up list. Closest published method: MLA (DeepSeek-V2) decomposes W_KV jointly with a learned low-rank; different — MLA requires training, and decomposes into separate K/V matrices, not a product fold. This idea decomposes the W_V·W_O PRODUCT without retraining, using the gauge freedom to choose optimal coordinates. No direct prior in the post-training compression literature.

### Smallest experiment

**Claim**: In Qwen3-1.7B, the per-head W_V · W_O product matrix (128 × 2048) has r_99/d ≤ 0.7 (vs W_Q r_99/d = 0.63 and W_gate r_99/d ≈ 1.0), i.e., it falls in the compressed-spectrum regime, not the full-rank regime.

- Measure: compute W_V · W_O for all 16 heads × 28 layers; run SVD; plot singular value decay; measure r_99 for each head.
- Runtime: ~30 min (196 matrix products + SVDs on 2048 × 2048 matrices on Ryzen, scipy).
- GO threshold: mean r_99/d ≤ 0.7 across heads and layers.
- NO-GO finding: if W_V·W_O is full-rank (r_99/d ≈ 1.0), extends CF8 to the attention-value pathway; closes this route and provides the first complete characterization of all weight-matrix rank structure in Qwen3-1.7B.

### Primary risk

W_V·W_O product may be full-rank even if W_V and W_O individually have concentrated spectra (because their product can fill rank via composition). Mitigation: check W_V and W_O independently first (2 × 15 min runs); if both have concentrated spectra, the product is more likely to as well.

---

## A-CADH — Content-Addressable Attention Head Dispatch (A-004-7)

**Track A — arch-transposition**

### Mechanism

**Constraint: Every weight matrix must be content-addressable by its action — retrieved by a hash of what it does to a representative input, not by spatial position in memory.** This eliminates all layout-dependent access patterns (sequential layer stride, tensor offset tables, fixed memory maps). Approximately 60% of standard weight-streaming and prefetch mechanisms (including INPOLT, NVPF, and layout-reorder methods) become inadmissible.

The mechanism: precompute a "signature" for each weight matrix W as sig(W) = BLAKE3(top-32 singular values of W as INT16 vector) — a 32-byte content hash of the matrix's most load-bearing structure. During inference, instead of loading matrices by position, maintain a content-addressable cache keyed on sig(W). When the computation graph requests "the W_Q for layer 14, head 3," the scheduler hashes the expected signature (precomputed at model load) and looks up the cache. If the signature matches a cached entry, the matrix is served from cache (L3 or RAM). If not, it's fetched from NVMe.

The structural payoff: two matrices with identical or near-identical signatures (by SVD top-32 overlap) are functionally interchangeable at inference time. CF11 found that all 16 W_Q heads collapse to a ~128-dim shared subspace — meaning the 16 per-head W_Q matrices have HIGHLY similar SVD signatures. Under content-addressing, these 16 matrices hash to the same bucket, and one fetch serves all 16 heads' W_Q needs. This is NOT the killed cross-layer sharing (v2-CHEAP-TEST-001 killed cross-LAYER; this is cross-HEAD within-layer, which the CF11 global K=128 finding directly supports). The fetch savings: 16 heads share 1 NVMe fetch instead of 16, for W_Q specifically — 16× bandwidth reduction on the most concentrated-spectrum weight class.

### Residency arithmetic

W_Q cross-head sharing within a layer: 16 heads × 128 × 2048 × bf16 = 8 MB per layer; if all 16 share 1 fetch: 0.5 MB per layer. 28 layers: 224 MB → 14 MB for W_Q alone. At 70B (32 heads × 128 dim × 8192 × 28 layers = 14.7 GB for all W_Q heads): if cross-head sharing applies to all layers: 14.7 GB → 0.46 GB. Model-wide savings at 70B: W_Q is ~14 GB of 17 GB INT4 total; rest is ~3 GB. With W_Q sharing: total reduces to ~3.5 GB. Plus W_K sharing by same logic (CF11 shows W_K r_99/d ≈ 0.79): further reduction. This is the mechanism that could reach DRAM-resident 70B, contingent on within-layer cross-head W_Q identity holding.

Note: content-addressable dispatch is the MECHANISM; the payoff depends on within-layer head-signature concentration being high (which CF11 supports for W_Q globally but not per-head individually — the 8× global compression means cross-head OVERLAP exists, which is exactly the sharing structure CADH exploits).

### Novelty gloss

Closest kill-list item: v2-CHEAP-TEST-001 (cross-LAYER W_Q stacking) — structurally different. CHEAP-TEST-001 tested whether W_Q subspaces are coherent ACROSS LAYERS (they are not); CADH tests whether W_Q matrices are coherent ACROSS HEADS WITHIN A LAYER (which CF11's 16-head-to-128-dim finding directly supports). Closest published: MQA / GQA (grouped-query attention) — different; GQA shares K/V heads during training; CADH shares all-head W_Q matrices POST-TRAINING via content-addressable dispatch without any architectural change. The content-addressable dispatch mechanism (BLAKE3-keyed NVMe fetch) applied to LLM attention matrices has no published prior.

### Smallest experiment

**Claim**: In Qwen3-1.7B, within a single layer, the 16 W_Q head matrices (each 128 × 2048 in local head coordinates) have pairwise cos-similarity ≥ 0.80 for at least 50% of pairs, confirming that cross-head CADH dispatch reduces effective W_Q fetches by ≥ 4× on that layer.

- Measure: extract all 16 W_Q head submatrices from layer 14 of Qwen3-1.7B; compute all 16×16 pairwise cos-similarities; report fraction above 0.80 threshold.
- Runtime: ~10 min (one-time extraction + 256 dot-products).
- GO threshold: ≥50% of pairs with cos-sim ≥ 0.80 AND ≥ 1 cluster of ≥ 4 heads sharing a representative.
- NO-GO finding: establishes that W_Q heads are NOT interchangeable within a layer, closing the within-layer sharing axis and sharpening the interpretation of CF11's global-K=128 finding (which must then arise from a different geometric structure than within-layer head-similarity).

### Primary risk

CF11's global K=128 finding may arise from all 16 heads spanning a 128-dim JOINT subspace without any two individual heads being similar — e.g., 16 orthogonal 8-dim heads that together span 128 dims. In that case, pairwise cos-sim would be low. Mitigation: the smallest experiment directly tests this and disambiguates; it runs in 10 min and costs nothing on NO-GO except interpretive clarity.

---

## Convergence handles

Primitives and measurements that 3+ ideas in this output share or depend on:

1. **W_V × W_O product spectrum** — A-GFWV and A-CADH both need the W_V/W_O rank characterization (noted as untested in CF11 follow-up); ~45 min measurement that gates both ideas.
2. **Within-layer cross-head W_Q similarity structure** — A-CADH directly; also relevant to A-GFWV's gauge-fix basis choice. Disambiguates CF11's global-K=128 finding.
3. **Post-SwiGLU activation dominant-channel stability** — A-TROP's smallest experiment; also informs RAOK (Track B R6 deferred) and PDAP-V (W_down extension).
4. **NVMe read bytes per token (v2 re-derivation of CF13)** — A-REGM's smallest experiment re-derives this number cleanly; gates INPOLT, NVPF, and any NVMe-bottleneck-conditional idea.
5. **Residual-stream state cardinality** — A-FSMT's centroid scaling curve; relevant to RSIDC (deferred) and any intrinsic-dimension-based bpw scheduling.
6. **Geometric decay rate λ of RoPE attention over position distance** — A-ALOG; gates all long-context KV compression schemes (CLASE, KHQL, ASCQ deferred).
