# Stage 1 — Orientation A (Constraint-Alien) — Run 002

Orientation: **A — Constraint-Alien**
Run: 002 (independent cold start)

---

## Idea A-1: Gauge-Fixed Head-Permutation Inference (GHPI)
**Orientation A / Track A**

### Name
Gauge-Fixed Head-Permutation Inference — GHPI — A / Track A

### Mechanism
**Track A.** Constraint: *the computation graph must algebraically simplify under a symmetry the trained weights respect — specifically, the multi-head attention permutation symmetry.*

Standard multi-head attention computes H independent heads and concatenates them. The output is invariant under any permutation of the head index, because concatenation followed by W_O mixes them linearly anyway — only the column-partition of W_O cares about which head is "head 3" versus "head 7." This is a genuine Z_{H!} gauge symmetry of the inference computation: permute the heads, repermute the corresponding columns of W_O, and the output is identical bit-for-bit.

The constraint — "express the inference computation in coordinates where this symmetry collapses out" — forces gauge-fixing: choose one canonical representative from each equivalence class. The load-bearing consequence is NOT the permutation itself, but what gauge-fixing reveals: if we sort heads at each layer by their principal direction (e.g., top singular vector of W_Q^{(h)}) and then look at the resulting sorted head sequence across layers, structurally similar heads from different layers will align into "lanes." Once aligned, the multi-layer head sequence becomes a matrix (layers × heads) with potential low-rank block structure that is invisible in the raw head-permuted basis.

CF11 is directly relevant: global W_Q at K=128 (8× compression) implies 16 heads collectively span ~128 dims, meaning many heads are near-duplicates. In gauge-fixed coordinates where heads are sorted by principal direction, the near-duplicate heads will cluster at the top of the sorted order and the tail heads will be near-zero projections. This gives a principled way to identify which heads to merge or drop WITHOUT per-head rank truncation (which CF11 kills) — instead, we merge ACROSS heads in gauge-fixed coordinates.

Stack primitive: torch.linalg.svd on W_Q^{(h)} for each head h across all 28 layers; numpy argsort to fix the gauge; matrix rank measurement in the aligned basis. No retraining, no new model weights — pure post-hoc measurement on the bf16 checkpoint.

Constraint difficulty calibration: ~85% of the standard "per-head compression" literature (per-head SVD, per-head pruning, per-head quantization) becomes inadmissible because those methods all operate in gauge-unfixed coordinates. The methods that survive are those that measure head similarity in gauge-fixed coordinates — a small minority.

### Residency arithmetic
Qwen3-1.7B: 28 layers × 16 heads × 4 matrices (W_Q, W_K, W_V, W_O). Attention weight total ≈ 28 × 4 × 2048² × 2 bytes = 940 MB (rough, ignoring GQA details). CF11 says global W_Q at K=128 gives ΔNLL=+0.98 nats at 8× compression — that's 940/8 ≈ 118 MB saved just from W_Q if the global subspace is stored once. If W_K compresses similarly (CF11 GO at K=512, 2×) that's another 235 MB. Together: ~350 MB reduction on a model that is ~3.4 GB at bf16. Not a floor-breaking compression on 1.7B, but scales: at 70B attention weights are ~20 GB of a ~140 GB model. At 4× attention compression: ~15 GB saved. That brings a 70B Q4_K_M (≈40 GB) to ≈37 GB — modest. The real payoff is at sub-2-bpw total (the 7.28 GiB wall): if the gauge-fixing reveals cross-layer lane structure with 4× compression on all attention matrices, and MLP stays at 4-bit, total model approaches 7.28 GiB for a 1.7B / 7B class model. For 70B the path requires stacking with MLP quantization.

The actual payoff of this idea is structural: the gauge-fixed basis is the correct coordinate system for measuring which compression methods to apply to attention. The residency win from GHPI is a second-order effect; the first-order win is a map of which heads are truly redundant that CF11's global-K result already hints at.

DRAM bandwidth at 4 tok/s (7530U ~11.5 GB/s): 1.7B at 4-bit = ~0.85 GB; 0.85/11.5 × 1000 = 74 ms/tok → ~14 tok/s unconstrained. Compression of attention weights further reduces effective bandwidth per token.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: CF11's per-head rank truncation kill. GHPI does NOT do per-head rank truncation — it fixes the head-permutation gauge and looks for inter-head redundancy in the aligned basis. Structural difference: per-head SVD (killed) operates on one head's weight matrix in isolation; GHPI operates on the cross-head alignment after gauge-fixing, finding redundancy that is invisible to per-head methods.

Closest published method: "Share Your Attention" (Aug 2025) and A3 (arXiv:2505.12942) share some attention-compression motivation. Structural difference: neither enforces the gauge-fixing symmetry argument; they both operate on attention scores or weight matrices in raw coordinates. GHPI's contribution is the gauge-fixing framing itself, which gives a principled reason (symmetry collapse) rather than a heuristic similarity metric.

No published paper explicitly uses head-permutation gauge-fixing as the coordinate system for attention compression — the concept exists in the equivariant networks literature but has not been applied here.

### Smallest experiment
**Claim**: after gauge-fixing (sort 16 heads per layer by top singular vector of W_Q^{(h)}), the sorted-head weight matrices across 28 layers show rank < 4 in the pairwise cosine-similarity matrix at similarity threshold ≥ 0.85.

- Instrument: load Qwen3-1.7B bf16; for each layer, SVD each W_Q^{(h)} (shape 128×2048), extract top singular vector; argsort heads by projection onto first principal direction; compute pairwise cosine similarity in the sorted basis across heads and layers; measure effective rank of the 28×16 = 448-dim similarity matrix.
- Wall-clock: ~20 minutes on Ryzen 5 7530U (SVD on 128×2048 matrices is fast).
- Go threshold: effective rank ≤ 8 in the cross-layer sorted-head similarity matrix (i.e., ~16× redundancy). This would indicate that the 448-head collection spanning 28 layers has only 8 distinct head "types" in gauge-fixed coordinates.
- NO-GO structural finding: if effective rank > 64, the heads are genuinely diverse across layers and gauge-fixing adds no compression leverage. This closes the "cross-layer head lane" path but leaves CF11's within-layer head redundancy intact as the operative finding.

### Primary risk
The sorted heads may not align across layers — gauge-fixing at each layer independently does not guarantee cross-layer alignment, because the residual stream rotation between layers scrambles the head directions. Mitigation: run the SVD on the jointly stacked cross-layer W_Q matrix (shape 28×16 × 128 × 2048 → reshape to (28×16×128) × 2048 and take SVD of that) to find a single global gauge; this is a 57344×2048 matrix SVD, still feasible in ~10 minutes on CPU.

---

## Idea A-2: All-State Append-Only KV Store (ASAKS)
**Orientation A / Track A**

### Name
All-State Append-Only KV Store — ASAKS — A / Track A

### Mechanism
**Track A.** Constraint: *all state is append-only — no mutation during inference.*

Standard KV cache mutates the KV store at every token: it writes new K, V vectors at position t, updates the buffer. The constraint eliminates all in-place mutation. Every write is a new log entry. Reads retrieve the latest version by content-addressing.

What this forces: the KV store becomes a log-structured store (like LSM-trees in LevelDB/RocksDB) where each attention key-value pair is an immutable record keyed by (layer_id, head_id, token_position, content_hash). "Updating" the KV cache at time t means appending a new record, not overwriting position t.

The structural consequence: once all K,V writes are immutable log entries, the attention computation at token t+n can be reformulated as a range-scan over the log. This is exactly what log-structured storage engines do for reads. The insight is that for *autoregressive generation*, the KV cache is already effectively append-only (you never overwrite a past K,V, you only ADD new ones) — the constraint makes this explicit and opens up LSM-tree-style techniques:

1. **Compaction passes** between generation batches: merge K,V entries that are near-duplicates (Jaccard similarity in K-space above threshold τ) into a single shared entry with a fanout pointer. This is the dedup step.
2. **Bloom filter per level** for fast "is this key present at this position?" lookup during beam search or speculative decode reuse.
3. **Tiered storage**: hot K,V (recent 128 tokens) in L3 cache (96 × 2048 × 2 bytes × 28 layers = ~10 MB at bf16); warm K,V (128–4096 tokens) in DRAM; cold K,V (4096+) on NVMe. The append-only constraint makes tiering clean — cold entries are never invalidated.

Stack primitives: Python `sortedcontainers.SortedList` for the log; xxHash for content-addressing; existing llama.cpp KV cache struct as the mutation layer being replaced. All primitives exist on the actual stack.

The constraint difficulty: ~75% of standard KV cache optimization literature (KV eviction by magnitude, KV in-place quantization, dynamic KV buffer reallocation) becomes inadmissible because all of these mutate existing entries. Surviving methods: append-only quantization (write new quantized copy, discard raw), Bloom filter routing, LSM-style compaction.

CF11 motivates the dedup step: if 16 heads collapse to ~8 head-type equivalence classes (as GHPI above suggests), then K,V entries from structurally similar heads across tokens may hash to nearby buckets, making content-addressed dedup effective.

### Residency arithmetic
Qwen3-1.7B, 4K context: KV cache at bf16 = 28 layers × 2 × 16 heads × 128 dims × 4096 tokens × 2 bytes ≈ 28×2×16×128×4096×2 = ~930 MB. This is ~27% of the 3.4 GB model. At 128K context it dominates.

Compaction dedup gain depends on K near-duplicate rate. If 30% of K vectors at each position hash to within cosine-sim ≥ 0.95 of a prior vector (plausible at longer contexts with repetitive token patterns), dedup reduces KV by ~30% → 651 MB at 4K. More aggressive at 128K where repetition is higher.

Tiered storage to NVMe for cold tokens: tokens >4K go to NVMe (3 GB/s sequential). At 128K context, ~96% of KV goes cold. Access latency is the bottleneck; Bloom filter pre-screen eliminates NVMe lookups for keys not in KV (O(1) per probe). Net: KV DRAM footprint from 128K context reduces from ~28 GB to ~2 GB (hot+warm tiers) + NVMe for cold.

For DRAM-resident 70B scenario: 70B weights at 0.55 bpw ≈ 4.8 GB. With 7.28 GiB total, only ~2 GB remains for KV at 4K context. ASAKS tiered storage enables 70B inference with any context length — NVMe cold tier is unbounded.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: R1/S2 KV Temporal Differencing (killed because RoPE rotates keys, deltas not small). ASAKS does NOT delta-encode; it content-addresses and deduplicates by absolute cosine similarity. Structural difference: ASAKS never diffs K vectors; it identifies near-duplicates and aliases their pointers.

Closest published method: KVQuant, MiniKV, KITTY (Track B R5 killed set) — all operate by quantization or eviction of KV entries (mutation-based). ASAKS operates in the append-only regime which those methods never consider. The closest structural analog is DiskANN's graph-indexed similarity search, but applied to KV cache entries rather than embedding retrieval.

The log-structured append-only framing for KV cache has not appeared in the literature (as of August 2025 knowledge cutoff) — it is a genuine framing transplant from storage engines.

### Smallest experiment
**Claim**: in a 512-token prefix on Qwen3-1.7B, at least 25% of K-vectors at any single layer share cosine similarity ≥ 0.90 with a prior K-vector in the same layer (content-addressable dedup rate ≥ 25%).

- Instrument: forward pass on Qwen3-1.7B with 512-token WikiText-2 prefix; extract K cache at all 28 layers (shape 512 × 16 heads × 128 dims); for each layer, compute pairwise cosine similarity matrix; count fraction above 0.90 threshold.
- Wall-clock: ~15 minutes (one forward pass + pairwise cosine on 512 × 16 × 128 K-vectors).
- Go threshold: ≥25% dedup rate in at least 10 of 28 layers.
- NO-GO structural finding: if dedup rate < 5% globally, K-vectors are diverse enough that content-addressed dedup does not help at typical context lengths. The tiered storage idea (ASAKS-cold tier) survives even on NO-GO because it requires only append-only semantics, not dedup.

### Primary risk
Content-addressed K lookup requires a hash function that is fast yet collision-resistant at the bit level; xxHash at 128-bit keys over 256-byte K-vector blobs has known performance but the bucket collision rate under real K-vector distributions is unmeasured on this workload. Mitigation: in the smallest experiment, measure xxHash throughput and collision rate on the extracted K-vectors before committing to it as the dispatch primitive.

---

## Idea A-3: Tropical-Algebra Attention Routing (TAAR)
**Orientation A / Track A**

### Name
Tropical-Algebra Attention Routing — TAAR — A / Track A

### Mechanism
**Track A.** Constraint: *no continuous numbers during inference — everything integer / tropical / table-lookup.*

Standard softmax attention: Q·K^T / √d → softmax → weighted sum over V. The softmax is a continuous operation producing floating-point weights. The constraint forbids this. Force the computation into max-plus tropical algebra.

In the max-plus semiring (ℝ ∪ {-∞}, ⊕=max, ⊗=+), the "softmax" of a score vector s simplifies to the argmax: the tropical softmax of s is the indicator 1_{argmax(s)}. The attention computation in the tropical limit is: for each query q, find the key k* = argmax_{j} q·K_j^T and return V_{k*} directly — i.e., hard attention / nearest-neighbor retrieval.

The constraint "no continuous numbers" here forces: all intermediate attention scores must be INTEGER. This is mechanizable by: (a) quantize Q, K to 8-bit integers; (b) compute Q·K^T as integer dot product (VPDPBUSD / VPDPWSSD on AVX2 — these are existing CPU instructions); (c) take argmax over the integer scores — no division by √d, no softmax exponential, no floating-point accumulation; (d) retrieve the corresponding V vector.

What does this buy? Two things:
1. **Hard attention eliminates the V weighted-sum** — instead of summing all V_j scaled by float attention weights, you retrieve exactly one V_{k*} per query head. This collapses the attention O(n·d_v) weighted-sum into O(d_v) for each head at inference time (only the winning V is loaded). For long contexts (n > 1000), this is a large bandwidth saving.
2. **Integer Q·K^T inner products are cheap on AVX2**: VPDPWSSD computes 8 int16 dot products per cycle; for d_k=128, one Q·K_j inner product = 8 VPDPWSSD cycles. At n=4096, full Q·K^T for one head: 4096 × 8 cycles ≈ 32K cycles ≈ 10 µs on 3.2 GHz Zen3. That's much faster than bf16 matmul.

The algebraic argument for approximation quality: tropical softmax is NOT equivalent to standard softmax (it is a 0-temperature limit). This is NOT a mathematically exact substitution. However, under the hypothesis that attention is approximately sparse (a few keys dominate per query), the approximation error is bounded: ‖softmax(s) − tropical(s)‖_1 ≤ H(s)/log(n) where H(s) is the entropy of the softmax distribution. CF11 (head-shared subspace structure) suggests many heads may have effectively sparse attention (low entropy), making the tropical approximation better on those heads.

Stack primitives: VPDPWSSD (AVX2 integer dot product, available on Zen3), numpy argmax, existing Q/K quantization in ggml (INT8 activation quantization pipeline).

Constraint difficulty: ~70% of the published attention optimization literature (linear attention, performer, flash attention, sparse attention with continuous weights) is inadmissible because it requires continuous attention coefficients. Surviving methods: hard attention (SMYRF adjacent, but SMYRF still uses random continuous projections), LSH nearest-neighbor retrieval (discrete but still uses Hamming distance not max-plus algebra).

### Residency arithmetic
Per-token attention bandwidth savings from hard attention: instead of loading ALL V vectors for weighted sum (n × d_v × 2 bytes per layer per head), only ONE V is loaded (d_v × 2 bytes). At n=4096 tokens: 4096× reduction in V-load bandwidth. For 28 layers × 16 heads × 128 dims × 2 bytes = 28 × 16 × 128 × 2 ≈ 115 KB per token (full V weighted sum at 4K context). With hard attention: 28 × 16 × 128 × 2 / 4096 ≈ 28 bytes per token (only winning V loaded). This is dramatic at long context. At short context (n=128), the gain is 128× — still material.

DRAM bandwidth at 11.5 GB/s: standard attention at 4K context loads 115 KB of V per token = 115/11500 × 1000 = ~10 ms for V loading alone. Hard attention: 28 bytes → ~2.5 µs for V loading. At 70B scale, V loading is an even larger fraction of per-token latency.

Quality cost: at temperature T→0, attention entropy collapses. Perplexity impact depends on how often "the right answer" is dominated by a single key. Estimating: +1.5 to +5 nats on WikiText-2 for a 70-temperature-like hard-attention approximation (inference prior; varies by task).

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: none directly. The integer-only constraint and tropical-algebra framing are absent from the v2 kill list.
Closest published method: hard attention (Graves, 2013), SMYRF (approximate nearest neighbor attention). Structural difference: TAAR's load-bearing claim is NOT "sparse attention is good" (well-known) but "integer Q·K^T + argmax on AVX2 is a FASTER and CHEAPER computation than float softmax at long context, and the constraint enforces it as the ONLY computation." The constraint is doing work: it rules out soft-attention fallback.

The tropical-algebra framing transplanted onto transformer attention has no direct published antecedent (checking against published survey as of August 2025).

### Smallest experiment
**Claim**: on Qwen3-1.7B generating 128 tokens from a 512-token prefix, tropical (hard) attention (argmax per query, single V retrieved) produces ΔNLL ≤ 3.0 nats vs standard softmax attention.

- Instrument: patch llama.cpp or Python transformers to swap standard attention for argmax(Q·K^T) + V[argmax] (single-V retrieval); run on WikiText-2 512-token inputs; measure NLL.
- Wall-clock: ~2 hours (Python prototype suffices; no need for AVX2 kernel for the quality measurement).
- Go threshold: ΔNLL ≤ 3.0 nats (within tolerable quality degradation for a stripped approximation).
- NO-GO structural finding: if ΔNLL > 5.0 nats, hard attention catastrophically fails on this model. This closes the "no continuous numbers in attention" path but tells us that attention weights are high-entropy on Qwen3 (attention is NOT sparse), which is itself a structural finding (contradicting sparse-attention literature claims for this model family).

### Primary risk
Attention entropy on Qwen3 is high (not sparse), making the tropical approximation grossly inaccurate and ΔNLL >> threshold. This would mean the "no continuous numbers" constraint, as applied to attention, produces an unusably bad model. Mitigation: run a quick entropy histogram of the softmax attention distribution on Qwen3-1.7B across 100 tokens before committing to the full experiment — if mean attention entropy > 4 bits per head, upgrade the constraint to "top-k attention" (k=8, still integer-addressable) rather than tropical argmax.

---

## Idea A-4: Register-Only Streaming Layer Kernel (RSLK) [FREE SWING]
**Orientation A / Track B**

### Name
Register-Only Streaming Layer Kernel — RSLK [FREE SWING] — A / Track B

### Mechanism
**Track B.** Constraint: *no RAM during decode — only registers, L1, L2, L3, NVMe.*

The Ryzen 5 7530U has: 32 AVX2 YMM registers (256-bit each, 32 bytes each = 1024 bytes total register file for float); 32 KB L1 data cache per core; 512 KB L2 per core; 16 MB L3 shared. "No RAM" during decode means: all active compute tensors must reside in register + L1 + L2 + L3. RAM is only for weight *storage* (pre-loaded buffer), not for activation *computation*.

What this forces: reformulate each transformer layer as a streaming kernel where activations at each step are maintained ONLY in cache/registers. The hidden state h (shape 2048 × 2 bytes = 4 KB at bf16) must stay in L1 during the entire layer forward pass. This is achievable if the GEMV for each weight matrix streams weights from RAM/NVMe to cache line by line and accumulates the result in registers, never materializing a full intermediate tensor.

Standard llama.cpp already does something like this via its GEMV path (weight rows loaded one by one, dot product accumulated), but the *activation residuals* between sublayers are often stored in a preallocated DRAM buffer. The constraint tightens this: every intermediate activation tensor must either (a) be a register variable (scalar or YMM register), (b) fit in L1/L2/L3 at its live point, or (c) not exist (fused away).

Concrete mechanism:
1. **Fused RMSNorm + GEMV**: compute RMSNorm in one pass over h (streaming from L1 to register), never writing the normed h back to RAM; feed directly into the GEMV dot product.
2. **SwiGLU activation fusion**: compute silu(W_gate·x) ⊙ (W_up·x) without materializing either intermediate. x lives in registers/L1; W_gate and W_up stream from weight buffer (NVMe or DRAM) into cache lines; the product is accumulated in registers.
3. **KV write is the ONLY RAM write**: at the end of each attention sublayer, write the new K,V pair to the DRAM KV cache. This is the single allowed mutation under the constraint.

Stack primitives: L1/L2/L3 cache specs (measured on this CPU — L1=32KB, L2=512KB, L3=16MB; all public); VFMADD231PS for FP32 accumulation; existing ggml kernel structure (k_quants.c GEMV loop). The hidden state at bf16 = 4 KB fits comfortably in L1 (32 KB). W_gate row is 2048 × 2 bytes = 4 KB — one row fits in L1 alongside h.

Constraint difficulty: ~60% of standard inference optimization (batched matrix multiply, activation checkpointing, KV materialization strategies, tensor parallelism across RAM regions) is inadmissible because it relies on intermediate tensor storage in RAM. Surviving: streaming GEMV with register accumulation (what llama.cpp already partially does), KV in DRAM (required), weight paging from NVMe.

This idea does NOT change model quality at all (it's a realization of the same computation graph in a more cache-local layout). The Track B payoff is: by keeping activations in L1/L2/L3, DRAM bandwidth is used ONLY for weight streaming and KV writes — not for intermediate activation reads. On the Ryzen 5 7530U, DRAM bandwidth is 11.5 GB/s but L3 bandwidth is ~200 GB/s. If activations stay in L3, the effective per-operation bandwidth budget is 17× higher.

### Residency arithmetic
This idea is not about bytes-per-weight (the model file stays the same size). The payoff is in tok/s.

Current bottleneck at 1.7B: model is 3.4 GB at bf16. DRAM bandwidth = 11.5 GB/s. If all activations stay in L3, then the only DRAM traffic is weight loading. At 3.4 GB per forward pass, throughput ceiling = 11.5 GB/s / 3.4 GB = ~3.4 tok/s. Standard llama.cpp achieves ~6 tok/s on Qwen3-1.7B at Q4 (weights = 0.85 GB; ceiling = 11.5/0.85 = 13.5 tok/s), so it is already near the bandwidth ceiling.

The RSLK payoff is at LARGER models where activations are currently spilling to RAM. At 70B at Q4_K_M (~40 GB): DRAM bandwidth ceiling = 11.5/40 = 0.29 tok/s. If intermediate activations (h=8192 dims × 2 bytes = 16 KB at bf16 — still fits in L1) stay in L3 AND weights stream from NVMe to L3: effective throughput depends on NVMe bandwidth (3 GB/s sequential). 40 GB / 3 GB/s = 13 sec/tok — catastrophically slow. This confirms that for 70B, NVMe streaming is too slow regardless of activation locality. RSLK is most useful for 7B-class models where the full model fits in DRAM (7B at Q4 ≈ 4 GB, within 7.28 GiB).

At 7B Q4 (4 GB) with RSLK: if intermediate activations stay in L3, all DRAM bandwidth goes to weight streaming. Baseline 7530U: L3 is 16 MB, h is 4096×2=8 KB (7B model), W_gate row is 4096×2=8 KB → both fit in L1. RSLK fused streaming could approach the DRAM bandwidth ceiling of 11.5/4 = ~2.9 tok/s for 7B at bf16. At Q4: ~11.5/1 = 11.5 tok/s ceiling. Measured llama.cpp on 7B Q4 on comparable hardware: ~6-8 tok/s, suggesting current code leaves 1.5-2× on the table relative to the DRAM bandwidth ceiling.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: none directly. RSLK is orthogonal to all MLP-weight compression kills (it doesn't change weights). Closest is "OS-Paging-Aware Weight Layout" (R1/S2, killed for not reducing bpw) — but RSLK doesn't depend on paging; it targets L1/L2/L3 residency of activations, not weight layout.

Closest published method: Apple's LLM-in-a-Flash (flash attention streaming from SSD) and DejaVu's lazy FFN (skip zero activations). RSLK differs: it does not require activation sparsity (no constraint on which neurons fire), it does not offload weights to flash (weights can be anywhere), and it explicitly targets L1-residency of the hidden state as the invariant. No published paper names "hidden state must never leave L1 during the layer forward pass" as the organizing constraint.

This is marked [FREE SWING] because it has no CF connection and is a structural/systems idea rather than a compression idea.

### Smallest experiment
**Claim**: a Python prototype that fuses RMSNorm + GEMV (streaming W_gate rows one at a time, accumulating in a pre-allocated 4-KB buffer that stays in L1 across the entire GEMV) achieves ≥ 1.3× throughput vs the unfused baseline for a single MLP layer forward pass on Qwen3-1.7B-class hidden dimensions (d=2048).

- Instrument: write a NumPy / Cython inner loop that performs one MLP GEMV streaming row by row with explicit row-pinning via np.empty(4096, dtype=np.float32) reuse; benchmark with timeit vs standard np.dot; confirm cache residency with perf stat cache-misses.
- Wall-clock: ~3 hours to implement and benchmark.
- Go threshold: ≥1.3× throughput improvement on single-layer GEMV (measured by L1 cache-miss rate reduction ≥ 40%).
- NO-GO structural finding: if cache-miss rate is already low in the unfused baseline, the Ryzen 5 7530U's prefetcher is already doing the streaming work. That is itself useful information — it means llama.cpp's existing GEMV loop is already near L1-optimal for 2048-dim hidden states, and the constraint-forced reformulation adds nothing for 1.7B-class models.

### Primary risk
The Ryzen 5 7530U's hardware prefetcher may already be achieving effective L1-residency for the activation vector, making the constraint-forced kernel equivalent to the existing GEMV loop. Mitigation: check the hardware prefetcher's stride detection capability (AMD Zen3 has a stride prefetcher with up to 2KB lookahead) — if the hidden state access pattern is regular, the prefetcher handles it; the constraint-forced kernel then only wins for irregular access patterns (sparse activations, non-uniform stride), which is a testable hypothesis.

---

## Idea A-5: Content-Addressed Weight Dispatch (CAWD)
**Orientation A / Track A**

### Name
Content-Addressed Weight Dispatch — CAWD — A / Track A

### Mechanism
**Track A.** Constraint: *every weight must be content-addressable by its action — no spatial layout; weights retrieved by hash of "what they do."*

Standard transformer inference uses spatially-indexed weights: W_gate[layer_id][row_id] is loaded because we know the spatial address. The constraint forces a different regime: weights are stored in a content-addressable store (CAS) keyed by a hash of their *functional signature* — a compact descriptor of what the weight vector does to a typical input distribution.

Define the functional signature of a weight row w_i ∈ ℝ^d as: fs(w_i) = sign(w_i · e_1, w_i · e_2, ..., w_i · e_K) where e_1,...,e_K are K fixed random probe vectors (K=64). This is a binary hash of length 64 bits — SimHash / LSH locality-sensitive hash of the weight's action on the probe distribution. Two weight rows with identical functional signatures have cosine similarity ≥ cos(θ_{64}) ≈ 0.87 with high probability (standard SimHash analysis).

The constraint-forced inference procedure:
1. At load time, compute fs(w_i) for every weight row in every matrix; store rows in a CAS keyed by fs.
2. Identical or near-identical functional signatures → DEDUPLICATE: store one canonical row, many pointers.
3. At inference time, identify the functional signature needed for each row of the current forward pass, look up in the CAS, retrieve the canonical row. Rows with identical signatures are the same canonical row loaded once and broadcast.

The deduplication rate depends on how many weight rows have similar functional signatures. This is a directly testable question on the actual checkpoint. CF11 (W_Q head-shared 8× compression) suggests that at least for W_Q, many rows in different heads share functional structure. CF3 (outlier dynamicity) does NOT apply here (this is about weights, not activations). CF8 (MLP weights full-rank) suggests that W_gate rows are diverse enough that dedup rate in W_gate might be low — but W_O and W_Q may fare better.

Stack primitives: BLAKE3 or xxHash for the CAS key function; numpy for the SimHash computation; Python dict for the CAS store; GGUF format for the deduplicated weight store (each tensor stores CAS pointers + canonical rows rather than full row-indexed weights).

Constraint difficulty: ~80% of standard weight storage (GGUF format, quantization to INT4/INT8, per-row scale factors, spatial locality assumptions in memory layout) is inadmissible because all of those assume spatial indexing. Content-addressable weight storage has no concept of "row 42 of W_gate" — only "the row whose functional signature is 0xABCD1234."

### Residency arithmetic
Dedup ratio depends on fraction of weight rows with duplicate functional signatures. OPTIMISTIC: if 30% of W_gate rows (6144 intermediate dim, each 2048 wide) share signatures, dedup reduces W_gate storage from 6144×2048×2 = 25.2 MB to 0.7 × 25.2 = 17.6 MB, saving 7.6 MB per layer across 28 layers = 213 MB total on 1.7B. At bf16, full 1.7B model is ~3.4 GB. 213 MB is 6% — not a game-changer.

PESSIMISTIC (given CF8): W_gate is full-rank, implying rows are diverse. Dedup rate may be < 5%. In that case CAWD only helps for W_Q (which CF11 says has 8× head-shared structure). W_Q dedup: 28 layers × 16 heads × 128 dim rows = 57344 rows at 2048 dim; if 12/16 heads per layer are near-duplicates (from CF11), 75% dedup rate → 57344 → 14336 canonical rows. W_Q storage: 57344×2048×2 = 235 MB → 59 MB. Saving: 176 MB = ~5% of 1.7B model. Still modest.

At 70B: W_Q storage is ~4 GB. 75% dedup → saves 3 GB. With total model at 140 GB (bf16), this is again modest. CAWD is not a primary residency lever; its value is architectural (CAS enables weight sharing across layers, inference caching, and runtime deduplication).

The real payoff is for weight *streaming* scenarios: a CAS allows the same canonical row to be loaded ONCE into L3 and reused for all duplicate hits — the bandwidth per token decreases proportionally to the dedup rate, not the storage reduction.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: NTFS-Reflink Codebook Deduplication (R1/S4 deferred, limited by Windows 11 Home not having reflinks). CAWD differs: it is not a filesystem-level dedup (no reflinks needed); it is an application-level CAS with a semantic key (functional signature), not a byte-level hash. Two weight rows with the same functional signature but different bytes (e.g., different scales but same direction) would deduplicate under CAWD but not under reflink dedup.

Closest published method: VPTQ (per-position quantization with codebooks) and GUCB (deferred Track B R5). CAWD differs: those methods are spatial codebooks; CAWD uses content-addressing keyed on the weight's FUNCTIONAL action on a fixed probe distribution, which is a different key function. The hash function is semantic (what the row DOES), not positional (where the row IS).

No paper uses SimHash of weight row action-on-probes as the deduplication key for transformer weight storage.

### Smallest experiment
**Claim**: in Qwen3-1.7B bf16, at least 15% of W_Q rows across all 28 layers have SimHash functional signature (K=64 probes) identical to at least one other W_Q row in the same or adjacent layer.

- Instrument: load W_Q matrices (28 × 16 heads × 128 rows × 2048 cols); sample K=64 random probe vectors (fixed seed); compute SimHash for each row (64-bit binary signature); count collisions; report dedup rate.
- Wall-clock: ~10 minutes (64 matrix-vector products for 57344 rows × 2048 dims = small computation).
- Go threshold: ≥15% collision rate in W_Q (i.e., ≥15% of W_Q rows share their SimHash with at least one other W_Q row across layers).
- NO-GO structural finding: if dedup rate < 3%, the functional-signature approach yields no residency benefit. However, this measurement itself is a structural finding about W_Q row diversity that has not been measured on Qwen3 — the NO-GO outcome closes CAWD but opens a "W_Q rows are geometrically diverse" confirmation.

### Primary risk
SimHash collision rate may be entirely explained by chance collisions (false positives) rather than genuine functional similarity, yielding garbage dedup that changes model outputs. Mitigation: for any two rows flagged as duplicates by SimHash, measure actual cosine similarity; require cosine ≥ 0.85 as a secondary filter (this eliminates SimHash false positives at the cost of some true positives). The smallest experiment should include this secondary filter.

---

## Idea A-6: Residual-Stream Gauge-Fixed Precision Scheduling (RGPS)
**Orientation A / Track B**

### Name
Residual-Stream Gauge-Fixed Precision Scheduling — RGPS — A / Track B

### Mechanism
**Track B.** Constraint: *the computation graph must algebraically simplify under the residual-stream gauge rotation symmetry — and inference must be expressed in coordinates where that symmetry is fixed.*

The residual stream h_L has a continuous rotation symmetry: for any orthogonal matrix R, replacing h_L → R·h_L and simultaneously rotating all weight matrices that read/write h_L (W_Q, W_K, W_V, W_O, W_gate, W_up, W_down, RMSNorm γ) by R^{-1}/R produces an IDENTICAL computation with IDENTICAL outputs. This is a GL(d) gauge symmetry of the inference graph (more precisely: the RMSNorm means the symmetry is the orthogonal group O(d), not full GL(d)).

The constraint forces: choose a gauge (a canonical basis for h_L at each layer) that algebraically simplifies storage and computation. The natural gauge is the PCA basis of the residual stream covariance: rotate all weight matrices such that h_L is white (unit covariance in calibration distribution). In this gauge, the RMSNorm operation is approximately the identity (covariance is already unit), and the weight matrices W absorb the normalization.

What this buys for compression: in the PCA gauge, singular values of weight matrices are re-ordered by the empirical importance of each residual-stream direction (the most-used directions are sorted to the top). This makes the spectrum of W in PCA-gauge basis MORE concentrated than in the raw basis — low singular values correspond to directions the model rarely uses. Standard quantization then assigns more bits to high-singular-value directions and fewer to low — per-dimension precision scheduling in a principled coordinate system.

This is NOT low-rank decomposition (which CF8 kills for MLP). It is a PRECISION SCHEDULING operation: same weight matrix, same rank, but different bit allocation per direction in the gauge-fixed basis. The total bits are reduced by allocating fewer bits to low-importance directions.

Concretely:
1. Compute PCA of {h_L}_{calibration} for each layer L (calibration covariance estimation — 2048×2048 matrix, feasible in <1 hour).
2. Rotate W_gate^{(L)} → W_gate^{(L)} · P_L (where P_L is the eigenvector matrix of the layer-L residual-stream covariance).
3. In the rotated basis, the column singular values of W_gate^{(L)} are ordered by importance. Quantize: top-512 dims at INT8, remaining 1536 dims at INT4.
4. Equivalent to a "budget-aware" column-quantization, but derived PRINCIPALLY from the gauge symmetry argument, not from a heuristic magnitude measure.

Stack primitives: numpy.linalg.eigh for calibration covariance; the rotation is a matrix multiply (W_gate → W_gate · P); existing ggml quantization pipeline for INT8/INT4 mixed-precision.

CF11 is ancillary: W_Q row-space is 8× compressible. In the PCA gauge, the W_Q column structure aligns with the residual-stream PCA — the gauge-fixed W_Q should inherit the same head-sharing redundancy but in coordinates where it is more explicit.

Constraint difficulty: ~65% of the standard quantization literature (fixed-precision per-tensor, per-row scale factors, INT4 uniform grids) is inadmissible because it does NOT fix the residual-stream gauge. The gauge-fixing is what makes the precision assignment principled rather than heuristic. Surviving methods: those that rotate weights before quantization (QuIP#, QuaRot, GPTQ with rotation) — but these typically rotate to eliminate outliers (the SmoothQuant motivation), not to PCA-align the residual-stream distribution.

### Residency arithmetic
Qwen3-1.7B W_gate: 28 layers × 6144 × 2048 = 346 M params. At INT4 (0.5 bytes): 173 MB. At the proposed 25%/75% INT8/INT4 split: 25% × 346M × 1 + 75% × 346M × 0.5 = 86.5 + 129.75 = 216 MB. Compared to uniform INT4 (173 MB): RGPS costs 43 MB MORE than INT4 for W_gate in absolute terms. But the claim is that RGPS INT8+INT4 matches the QUALITY of INT8 uniform (with the gauge-fixed allocation buying quality on the critical directions). The comparison is: vs INT8 uniform (346 MB), RGPS saves 346-216 = 130 MB = 37% for W_gate.

At 70B scale: W_gate alone is ~35 GB at bf16. At INT4 uniform = 8.75 GB. At RGPS 25/75 INT8+INT4 mix = 10.8 GB. The quality-per-bit improvement of RGPS is that INT8 on top 25% of dimensions makes RGPS match INT8 quality while using 37% less storage than INT8. Whether this quality-per-bit is actually better than INT4 uniform + a calibration correction is the testable question.

If RGPS matches INT8 quality at 0.625 bpw effective (weighted average of INT8 on 25% + INT4 on 75%): then 70B total at 0.625 bpw = ~5.8 GB DRAM residency — within the 7.28 GiB target.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: "MDL-Selected Per-Layer bpw" (R2/S2 killed as converging to mixed-precision territory). RGPS differs: MDL selection assigns precision per LAYER; RGPS assigns precision per COLUMN DIRECTION within each layer, guided by a gauge-fixing argument. The unit of precision variation is different (direction vs layer), and the motivation is principled (gauge symmetry) rather than MDL-heuristic.

Closest published methods: QuaRot (arXiv:2404.00456) rotates weight matrices for outlier suppression; GPTQ applies Hessian-weighted quantization per column. RGPS differs from QuaRot in that the rotation is the PCA of the residual-stream covariance (a principled gauge choice), not a random Hadamard. It differs from GPTQ in that the precision allocation follows the gauge-fixed singular value ordering, not the Hessian trace.

The specific framing "residual-stream gauge rotation symmetry → PCA gauge → per-direction precision scheduling" is new. QuaRot is closest but uses Hadamard (random orthogonal) for numerical reasons, not PCA for principled-gauge reasons.

### Smallest experiment
**Claim**: in Qwen3-1.7B, the gauge-fixed (PCA-rotated) W_gate columns have a more concentrated singular-value spectrum than raw W_gate — specifically, top-25% of PCA-rotated columns account for ≥ 80% of the column L2-norm mass.

- Instrument: (1) run 200-token calibration forward pass, collect {h_L} for each layer; (2) compute covariance C_L = (1/n) Σ h_L h_L^T; (3) eigendecompose C_L → P_L; (4) compute W_gate^{(L)} · P_L; (5) measure fraction of column L2-norm mass in top-25% of columns (after sorting by norm).
- Wall-clock: ~40 minutes (200-token forward pass + per-layer 2048×2048 eigendecompose + matmul).
- Go threshold: top-25% of PCA-gauge columns account for ≥ 80% of total column mass (i.e., gauge-fixing IS concentrating the column spectrum).
- NO-GO structural finding: if top-25% accounts for < 50% of column mass, the residual-stream covariance PCA does not concentrate the weight spectrum. This tells us that the "natural" weight directions and the "natural" activation directions are already aligned (the gauge is already PCA-like), meaning no gauge-fixing benefit exists. This is also a structural finding about how well-aligned Qwen3's weight initialization is with the data distribution.

### Primary risk
The rotation W_gate · P_L is a bijective change of basis — it does NOT change the model's outputs. But the quantization AFTER rotation (INT8 on top 25%) does change outputs. The risk is that the performance of INT8 on top-25% + INT4 on bottom-75% in PCA-gauge is no better than INT4 uniform without any rotation, because the INT4 noise on 75% of directions is larger than the quality improvement from INT8 on 25%. Mitigation: in the smallest experiment, run both (a) INT4 uniform and (b) RGPS INT8+INT4 on a 1-layer prototype; compare ΔNLL. If RGPS is not better, the gauge-fixing is not doing useful work for quantization precision (it may still be useful for other purposes).

---

## Idea A-7: Finite-State-Machine Token Decoder (FSMD)
**Orientation A / Track A**

### Name
Finite-State-Machine Token Decoder — FSMD — A / Track A

### Mechanism
**Track A.** Constraint: *the inference loop must fit in a state machine of N states for small N (target N ≤ 65536 = 16-bit).*

Standard autoregressive decoding maintains a continuous hidden state h ∈ ℝ^{2048} — effectively infinite distinct states. The constraint requires collapsing the decoder to a finite-state machine with at most 2^16 states.

The constraint-forced reformulation: discretize the residual stream h_L into one of N discrete symbols. Define the discrete residual stream as s_L = VQ(h_L) where VQ is a learned vector quantizer with K = 65536 codewords (16-bit codebook). The inference computation then becomes:

1. Tokenize input → sequence of token IDs.
2. Initialize FSM state s_0 = VQ(embedding(token_0)).
3. At each step: look up transition table T[s_L][token_id] → s_{L+1}. If the transition table is precomputed for all (s_L, token_id) pairs and all L layers, the per-token computation is a TABLE LOOKUP, not a matrix multiply.

The catch: T is a table of size K × V × L = 65536 × 151936 × 28 entries, each a 16-bit state ID. Size: 65536 × 151936 × 28 × 2 bytes = 552 GB. Utterly infeasible to precompute.

However, the constraint can be relaxed to a *local* FSM: only precompute T for the most frequent (s_L, token_id) pairs. Under a power-law token distribution (Zipf), the top-10K (s_L, token_id) pairs cover the majority of transitions. For those transitions, decode via table lookup; for rare pairs, decode via the standard matmul path. This gives a CONDITIONAL FSM where common paths are O(1) table lookup and rare paths fall back to O(d²) matmul.

Stack primitives: Python dict or numpy array for the transition table; xxHash for fast (s_L, token_id) key lookup; existing VQ implementation (faiss.IndexFlatL2 for codebook assignment, or scipy.cluster.vq for smaller codebooks). The table caches in L3 for hot entries.

Constraint difficulty: ~90% of the standard inference optimization literature assumes continuous hidden states and is inadmissible here. Surviving: cached computation (speculative decoding is adjacent, but FSMD precomputes transitions rather than speculatively running the full model). Myhill-Nerode theorem gives the minimum DFA for any regular language — FSMD can be seen as a neural approximate DFA where the accept condition is "output next token."

### Residency arithmetic
Transition table for N=65536 states, V=151936 vocab, L=28 layers would be 552 GB — infeasible. However, a PARTIAL table for top-P% of transitions:

If top-10K (state, token) pairs cover 80% of transitions (plausible under power-law), and each entry is 16 bits (2 bytes): 10000 × 2 bytes = 20 KB. Fits in L1 cache. Cache hit rate: 80%. For those 80% of tokens, per-token latency = L3/L1 lookup latency (~4 cycles = ~1.3 ns) vs standard matmul (~70 ms). At 80% cache hit rate, effective per-token latency ≈ 0.8 × 1.3ns + 0.2 × 70ms ≈ 14 ms → ~70 tok/s.

Whether 10K entries actually cover 80% of transitions depends on the actual distribution. If the distribution is more uniform (coverage falls to 20% at 10K entries), the effective latency is 0.2 × 1.3ns + 0.8 × 70ms = 56 ms → 17 tok/s (marginal improvement). The power-law coverage rate is testable.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: R1/S4 deferred Count-Min Sketch Activation Filter (prediction-historical). FSMD differs: CMS is a frequency table over activations; FSMD is a full state-machine over the discretized hidden state. CMS still runs the matmul; FSMD replaces it entirely for cached transitions.

Closest published method: Speculative decoding (draft model generates candidates, target model verifies); N-gram Trie Speculative Decoding (EMNLP 2025, on kill list for Branch-Predictor Token Trie). FSMD differs fundamentally: it does NOT run a draft model. It precomputes transitions during a warm-up phase and caches them. The closest analog is KV caching + prefix sharing, but those operate on continuous states; FSMD operates on discrete states.

The Myhill-Nerode / FSM framing for transformer inference is genuinely novel. No paper precomputes (discrete-residual-state, token-id) → next-discrete-state transitions as a lookup table. The constraint is doing real work: the continuous hidden state must be discretized because the FSM constraint requires finite states.

### Smallest experiment
**Claim**: a codebook of K=1024 codewords (VQ fit on 200-token calibration) assigns >50% of forward passes at a single layer (layer 14) to codewords such that the top-100 most-frequent (codeword_id, next_token_id) pairs cover ≥40% of all (codeword, token) transition pairs in a 1000-token generation.

- Instrument: VQ-quantize h_14 activations from 200-token calibration pass (K=1024, scipy.cluster.vq); run 1000-token generation on a different passage; record (h_14_codeword, next_token) pairs; count coverage of top-100 pairs.
- Wall-clock: ~30 minutes (one calibration forward pass + one generation + k-means fit).
- Go threshold: top-100 pairs cover ≥40% of transitions at K=1024 (indicating high concentration).
- NO-GO structural finding: if coverage < 20%, the joint (state, token) distribution is flat — FSMD cannot achieve significant cache hit rate at tractable table sizes. This tells us the residual stream at layer 14 is not compressible to a small number of discrete states without quality loss — a structural finding about the intrinsic dimension of transformer computation.

### Primary risk
VQ quantization of the hidden state introduces representation error: ΔNLL from discretizing h to K=65536 codewords may be catastrophic (similar to the CF12 embedding catastrophe at truncated rank). The CF12 finding that tied embedding is catastrophically sensitive to truncation at r < d is a warning signal. Mitigation: in the smallest experiment, measure ΔNLL from VQ-substituting h_{14} at K=1024 before investing in the full table-construction; if ΔNLL > 2 nats even at K=1024, the constraint is too tight and must be relaxed to K=2^20 or abandoned.

---

## Convergence handles

Primitives / measurements that this output's ideas depend on or produce:

1. **W_Q cross-head and cross-layer principal direction alignment** — GHPI and CAWD both depend on it; load-bearing for CF11 extension beyond per-layer measurement.
2. **K-vector cosine dedup rate at long context** — ASAKS depends on it; partially implied by CF3 per-token Jaccard dynamicity (if K-vectors rotate per token, dedup rate is low).
3. **Residual-stream covariance PCA concentration** — RGPS depends on it; if top-25% of PCA columns account for ≥80% of column mass, the gauge-fixing is paying for precision scheduling.
4. **Discrete residual-stream transition distribution** — FSMD depends on it; coverage of top-K (state, token) pairs under VQ quantization determines whether FSM table dispatch is feasible.
5. **Attention entropy distribution per head on Qwen3** — TAAR depends on it; if mean entropy > 4 bits, tropical (hard) attention is unusable; this is an unmeasured quantity on the v2 stack.
6. **Per-layer cache-miss rate for activation buffers in existing llama.cpp** — RSLK depends on whether the current implementation already achieves L1-residency for the hidden state; perf stat measurement resolves this in <1 hour.
