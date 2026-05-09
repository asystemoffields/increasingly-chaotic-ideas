# Stage 1 — Constraint-Alien (A) — Run 1

Orientation: A — Constraint-Alien
Run: 1 of 50
Ideator: Sonnet 4.6

---

## A1 — Tropical-Semiring Weight Dispatch (TSWD) [Track A]

**Constraint**: No continuous arithmetic during the weight-routing phase of inference. Weight selection and mixing must operate in the (max, +) tropical semiring; floating-point multiply-accumulate is forbidden until the routing decision is made.

**Constraint difficulty**: ~70% of standard mechanisms are inadmissible. Standard continuous attention (softmax over dot-products), continuous mixture-of-experts gating, learned router logits with sigmoid/softmax, and all gradient-based calibration-fit routing are ruled out. Admissible: integer indices, max operations, addition operations, table-dispatch, threshold comparisons. Inadmissible: softmax, weighted averages of weight matrices, continuous interpolation between experts, NNLS routing, NMF routing.

### Mechanism

Track A. The standard Qwen3 MLP computes `output = (silu(W_gate · x)) ⊙ (W_up · x); W_down · output`. The routing question — which neurons fire — is currently a continuous product of two full-rank matrices (CF1: W_up dominates firing rank; CF8: both matrices full-rank). The tropical constraint says: compute `i* = argmax_i { W_up[i,:] · x + b_i }` where the argmax and addition operate in (max, +) and `b_i` is a learned integer bias stored as an INT8 offset; no multiplication is permitted in the routing layer itself. The output is then computed ONLY for the top-K neurons selected by the tropical argmax — the routing becomes a deterministic lookup-and-execute rather than a dense matrix multiply. Concretely: quantize the inner-product `W_up[i,:] · x` to an integer rank score via INT8 GEMV (this step IS a matmul, but of the routing projection, not the full W_up), apply the max-plus priority `rank[i] + b_i` (INT16 addition), threshold at τ, execute W_down GEMV only for the surviving K neurons. The `b_i` offsets are calibrated once on 200-token calibration data: set `b_i = median(rank[i])` so the threshold τ=0 gives the desired sparsity. This is NOT the standard playbook: it imposes a non-differentiable integer gate as the architectural primitive, motivated by the constraint, not by a neural training objective. The novelty is that the routing is done in tropical coordinates, making the sparsity deterministic and cache-friendly (the executing neurons form a contiguous INT16-sorted list). Relies on CF1 (W_up·x dominates firing rank) as motivation for routing on W_up inner-product; CF8 ensures W_up is full-rank so no low-rank substitute for the routing matmul exists — the routing matmul is unavoidable, but the downstream W_down execution is sparse.

### Residency arithmetic

Baseline: Qwen3-1.7B at BF16, 1.54 GiB. For a 70B-class model at 4-bit (Q4_K_M ~ 4.5 bpw), 70B × 4.5 / 8 = 39.4 GiB — NVMe resident. Per-token bandwidth: at 39.4 GB read per 2-token batch / 11.5 GB/s = 3.4 s/token on DRAM-bound path.

Tropical routing win: if the integer gate selects K=20% of neurons, the W_down GEMV cost drops to 20% of the full MLP. Bandwidth: W_down is 70B × (intermediate/hidden) weight fraction; MLP is ~2/3 of parameters. 20% W_down sparsity → ~13% total bandwidth reduction. Effective tok/s improvement: modest alone. Stacked with a 4-bit baseline: negligible residency change but compute-bound improvement on the CPU matmul side. However, the tropical-gate insight is NOT primarily about bandwidth — it is about deterministic sparsity structure enabling cache-line-aligned weight access (the surviving neurons can be pre-sorted to contiguous INT16 index lists, giving sequential cache-line reads rather than scatter-gather). On the Ryzen 5 7530U, scattered non-sequential reads for the sparse neurons would thrash L2; the sorted-index pre-layout converts this to a sequential read burst. At K=20% neurons, if the index-sorted layout achieves 80% of sequential bandwidth vs 30% of sequential bandwidth for naive scatter, the effective throughput gain on W_down is 80/30 ≈ 2.7× for the sparse MLP portion. Overall: ~40% total inference speedup on MLP token generation is the ceiling if K=20% is achievable without quality loss. Quality cost: if the K=20% gate loses the neurons that contribute the tail of W_up firing distribution, dNLL degradation is unquantified here — this is the primary risk.

### Novelty gloss vs kill list

Closest kill list item: SABLE-v1 (Top-K scalar + Horvitz-Thompson) — killed due to diffuseness. Structural difference: TSWD does not use statistical sampling or unbiasedness corrections; it uses a deterministic tropical-argmax gate with calibrated INT8 offsets. The sparsity comes from thresholding a max-plus priority score, not from a stochastic estimator. Closest published: Deja Vu (ICML 2023) uses cross-layer cosine similarity as a predictor; TSWD uses within-layer tropical argmax of W_up·x (the CF1-motivated predictor). Deja Vu is killed for ReLU/OPT models; CF1 says gate-side prediction fails for SwiGLU, but W_up-side prediction is the actual target here, which Deja Vu did not test. The "tropical semiring routing" framing has no vocabulary in the published LLM compression literature.

### Smallest experiment

Claim: on Qwen3-1.7B-Base, a tropical argmax gate on W_up inner-products with K=20% neuron retention achieves ΔNLL < 0.5 nats vs full-dense MLP, measured on 200-token WikiText-2 held-out. Runtime: ~45 min on Ryzen 5 7530U (INT8 GEMV of W_up routing projection + mask application + masked W_down GEMV in BF16 PyTorch). Go/No-go: ΔNLL < 0.5 nats at K=20% → GO; ΔNLL > 1.0 nats at K=20% → NO-GO. On NO-GO: measures the effective firing sparsity of Qwen3 MLP under CF1-motivated routing, producing a structural finding about W_up sparsity that bounds all future sparse-dispatch ideas.

### Primary risk

CF1 says W_up·x predicts firing rank but the absolute sparsity level (what fraction of neurons can be dropped at K=20% without quality loss) is unmeasured — if Qwen3 MLP is functionally dense at K=20%, quality degrades catastrophically. Mitigation: run the smallest experiment at K=50%, 30%, 20%, 10% in sequence (4 thresholds, 45 min each); kill if K=50% already shows ΔNLL > 1.0.

---

## A2 — Content-Addressable Weight Shards (CAWS) [Track B]

**Constraint**: Every weight shard must be retrievable solely by the hash of its action — not by its spatial position in the tensor. No weight is stored at a fixed byte offset derived from layer index or row index; instead, all weights are stored in a content-addressable object store keyed by BLAKE3 hash of a compact action descriptor (the top-K singular value fingerprint of the shard, quantized to INT8).

**Constraint difficulty**: ~55% of standard mechanisms inadmissible. Ruled out: fixed-offset tensor layout (GGUF row-major), layer-sequential loading, spatially-indexed quantization codebooks, any scheme that relies on knowing "this is W_gate layer 12 row 47" at decode time. Admissible: hash-keyed dispatch, deduplication, fingerprint matching, lookup-by-action-signature, shared-weight detection. The constraint is mechanizable: the BLAKE3 hash of a shard's INT8-quantized top-64 singular vectors serves as the content key.

### Mechanism

Track B. The standard GGUF layout stores weights sequentially: bytes at offset `f(layer, matrix_name, row)`. The content-addressable constraint replaces spatial layout with BLAKE3-keyed object storage: each weight shard (e.g., a 64-row block of W_Q for layer L) is stored once under `BLAKE3(INT8_quantize(SVD_top64(shard)))`. Two shards with similar action signatures — same top-64 singular vectors within INT8 tolerance — collide to the same key and are stored exactly once. At inference, the lookup is: compute the action key for the requested shard from a small index table (key → byte offset in the object file), fetch the shard, execute. This is NOT deduplication by weight value (which would trivially fail for full-rank matrices); it is deduplication by action signature — two shards that perform the same linear map within INT8 approximation are the same shard. The empirical question is: do any weight shards in trained Qwen3 have sufficiently similar action signatures to collide? CF11 provides motivation: W_Q has r_99/d ≈ 0.63 and shows head-redundancy (16 heads collapse to ~128-dim subspace). If the top-64 singular vectors of different W_Q heads (treated as 64×128 shards) are similar, they will hash to the same key and compress. This is a structural deduplication on the attention weight side. CF8 says MLP weights are full-rank — MLP shard collisions are unlikely, so CAWS gains come primarily from attention. The object-store format can be written as a simple flat file with a prefix index (key → offset), readable by any process that can compute BLAKE3 and look up an offset — satisfying the "debuggable to a step level by reading bytes" constraint as a side effect.

### Residency arithmetic

Attention weights in Qwen3-1.7B: 28 layers × (W_Q + W_K + W_V + W_O) × d² = 28 × 4 × 2048² × 2 bytes ≈ 0.97 GiB at BF16. If head-redundancy (CF11: 16 heads → ~128-dim subspace) means that all 16 W_Q heads per layer have similar top-64 action signatures → 1 stored shard per layer for W_Q (16× dedup factor within a layer). Residency win: W_Q contributes 28 × 2048 × 2048 × 2 bytes × (1/16 dedup) at the extreme → 0.97/4 × (1/16) ≈ 15 MB. Even at 4× dedup (4 heads de-duplicated per 16): W_Q residency from 240 MB → 60 MB. For a 70B model: attention weights ~16 GiB at BF16 × 4-bit quantization → 4 GiB; if 4× head dedup applies: 1 GiB. Total model at 4-bit: ~40 GiB; attention dedup: saves ~3 GiB → 37 GiB. Modest at 70B but the mechanism is a pure structural finding — the dedup factor measures the actual head-redundancy, providing a number not yet measured. At the 7.28 GiB constraint with a 7B model (Q4 ~3.8 GiB), CAWS could in principle bring it to ~3.2 GiB if 4× attention dedup holds. The quality cost is zero by construction (deduplication stores one reference, all reads return the same bytes).

### Novelty gloss vs kill list

Closest kill list: NTFS-Reflink Codebook Deduplication (R1/S4-deferred, Idea I) — killed because Windows 11 Home lacks reflinks. Structural difference: CAWS does not rely on any filesystem feature; the object store is a flat file with a prefix index, self-contained and portable. The content-addressable key is the action signature (singular vectors), not the weight values — this is a different dedup criterion. Closest published: Git/IPFS content-addressable storage applied to ML weights is not published in the LLM inference literature; GGUF stores weights sequentially by layout. The closest ML mechanism is weight sharing / head-tying, but those are typically trained-in; CAWS discovers post-hoc sharing by action-signature comparison. "Content-addressable weight dispatch" has no vocabulary in the LLM deployment literature.

### Smallest experiment

Claim: in Qwen3-1.7B-Base, at least 2 of 16 W_Q heads per layer have top-64 singular-vector similarity cos(U_i^{(64)}, U_j^{(64)}) > 0.85, measured on 5 representative layers (layers 7, 14, 21, 27, 28). Runtime: ~15 min (compute SVD of 16 × 2048×128 slices of W_Q per layer, compute pairwise cosine similarity of top-64 left singular vectors, histogram). Go/no-go: ≥2 head pairs per layer with cosine similarity > 0.85 → GO for content-addressable dedup; < 1 head pair at > 0.70 → NO-GO. NO-GO finding: measures the actual angular diversity of W_Q heads in trained Qwen3, adding a new structural number to the CF11 boundary (head-redundancy in Q-subspace vs head-redundancy in individual-head SVD).

### Primary risk

The INT8 quantization of the action signature may introduce enough noise that genuinely similar shards hash differently (collision-miss) while unrelated shards hash similarly (false collision). Mitigation: use the cosine similarity measurement as the gate; only declare dedup when cosine similarity > 0.85 in the smallest experiment. Hash the full top-64 singular vectors at BF16, not INT8, for the lookup key — use the INT8 approximation only to estimate the collision rate.

---

## A3 — Append-Only Log-Structured Attention (ALSA) [Track A]

**Constraint**: All state during decode is append-only. No tensor is mutated in place. Every operation writes new bytes; no overwrite is permitted. The KV cache is a write-once append log; attention reads from the full log at every step; the residual stream is an accumulation of immutable per-layer records.

**Constraint difficulty**: ~40% of standard mechanisms inadmissible. Ruled out: in-place KV eviction (sliding-window KV, H2O, ScissorHands, SnapKV), KV in-place quantization-after-write, residual-stream overwrite, any attention mechanism that modifies previously written bytes. Admissible: log-structured append (new tokens add new KV records), monotonic KV growth, retrieval-only attention over the full log, content-addressable KV (keyed by token hash rather than position), idempotent reads.

### Mechanism

Track A. The append-only constraint forces the KV cache to behave as a log-structured storage engine (like an LSM-tree write path or LMDB append log). Each decode step appends a new (k, v) pair to a flat ring buffer on NVMe; no position is ever overwritten. Attention then retrieves the subset of KV records needed for the query. The payoff is NOT in residency (KV grows monotonically) but in access pattern: an append-only log on NVMe has sequential write (near-saturation bandwidth) and the read pattern is a sequential scan of the log segment in context window. This maps perfectly to NVMe sequential read (3 GB/s) vs random-read (0.7 GB/s). For a 70B-class model at 4K context, KV at INT4 is: 70B with d_kv=1024 (MLA-style), 80 layers, INT4 → 80 × 2 × 4K × 1024 × 0.5 bytes = 320 MB. At 8K context: 640 MB. At 32K context: 2.56 GB. The append-only log makes the 32K context case tractable on NVMe at sequential bandwidth: 2.56 GB / 3 GB/s = 0.85 s to scan once. Combined with CF11 (W_Q head-sharing, global K=128 compresses W_Q by 8×): the attention weight residency drops, and the KV cache becomes the dominant residency item at 32K+. The append-only constraint also enables a structural simplification: since no KV is ever overwritten, the KV log can be content-addressed by token hash — if the same token appears at the same position in two different decodes, the KV record is identical and can be looked up without recomputation (a natural prefix-cache). The primitive is the OS page cache + NVMe sequential read path (an existing OS subsystem); no new data structure is required — a flat file extended with write() syscalls + mmap for reading is the full implementation.

### Residency arithmetic

Target: Qwen3-72B (closest public model in 70B class) at 4-bit. Weights: 72B × 4.5 bpw / 8 = 40.5 GiB — NVMe resident. KV cache at INT4, 4K context, 64 layers, d_kv=128 per head, 64 heads (GQA=8, 8 KV heads): 64 layers × 8 KV heads × 2 × 4096 tokens × 128 dims × 0.5 bytes = 268 MB — DRAM resident. Append-only NVMe log win: at 4K context, KV fits in DRAM regardless; the append-only structure gives zero residency reduction at 4K but sequential NVMe access at 32K+ context. At 32K context on NVMe: 2.15 GB KV at INT4; sequential read → 0.72 s scan. Total inference at 32K: weight loading (40.5 GB / 3 GB/s = 13.5 s/token, NVMe sequential) + KV scan (0.72 s) ≈ 14.2 s/token. Not usable at 32K without aggressive weight compression, but this is the floor measurement. Quality cost of INT4 KV: uncalibrated; likely < 0.3 nats at 4K based on KVQuant literature.

### Novelty gloss vs kill list

Closest kill list: KV Temporal Differencing (R1/S2 — killed because RoPE rotates keys, making deltas non-small). Structural difference: ALSA does not differentiate or modify KV; it appends and reads whole records. The append-only constraint is the opposite of delta compression. Closest published: Flash-Decoding / PagedAttention — both manage KV block allocation but allow in-place eviction; ALSA forbids eviction entirely. The content-addressable prefix-cache aspect is adjacent to RadixAttention (SGLang), but RadixAttention allows eviction; ALSA keeps all records and exploits sequential NVMe scan instead. "Log-structured append-only KV with NVMe sequential-scan attention" has no published analog in the LLM inference literature.

### Smallest experiment

Claim: a naive append-only KV store in Python (write each (k, v) pair as a BF16 blob to a memory-mapped flat file, read the full log for attention via sequential mmap scan) achieves NVMe read bandwidth ≥ 2.5 GB/s on Qwen3-1.7B at 2K context on the Ryzen 5 7530U. Runtime: ~2 hours (implement the mmap KV log, run 100-prompt inference with Qwen3-1.7B in PyTorch with custom attention reading from the log, measure NVMe read bandwidth via perfmon). Go/no-go: sequential mmap throughput ≥ 2.5 GB/s → GO (sequential path engages); < 1.0 GB/s → NO-GO (random-page-fault path dominates, mmap doesn't give sequential benefit). NO-GO finding: measures the OS mmap sequential-vs-random read threshold on this hardware for KV-shaped access patterns (a structural number useful for all NVMe-offload ideas).

### Primary risk

The RoPE rotation of keys means that "same token at same position" does not produce identical KV records across different sequence contexts (RoPE encodes absolute position, so the same token at position 47 vs position 1047 gets different K). This breaks the content-addressable prefix-cache aspect. Mitigation: separate the sequential-scan storage benefit (which does not require KV identity) from the prefix-cache benefit; the smallest experiment tests only the sequential-scan bandwidth claim.

---

## A4 — Register-Only Residual Streaming (RORS) [FREE SWING] [Track A]

**Constraint**: No RAM during decode — only registers, L1, L2, L3, NVMe. The residual stream vector `h_t` (d_model floats) must fit entirely within L3 cache during a decode step; it is never written to DRAM. Weights stream from NVMe through L3 and are not resident in DRAM at all.

**Constraint difficulty**: ~80% of standard mechanisms inadmissible. Ruled out: any scheme that pages weights through DRAM (standard llama.cpp NVMe path), any scheme relying on DRAM bandwidth (11.5 GB/s path), full-model DRAM residency, KV cache in DRAM at 32K+ context, batch sizes > 1 that require DRAM buffering. Admissible: streaming weights from NVMe directly to CPU registers + L1/L2/L3 via DMA + sequential read; residual stream maintained in L3 (Ryzen 5 7530U has 16 MB L3 — fits d_model=4096 floats = 16 KB easily, fits d_model=8192 = 32 KB easily, fits even d_model=32768 = 128 KB); computed output written back to NVMe for the next layer's weights to be streamed over.

### Mechanism

Track A. The Ryzen 5 7530U has 16 MB L3 cache. A Qwen3-72B residual stream at d_model=8192, BF16: 8192 × 2 bytes = 16 KB — this fits in L1. The hidden activations during an FFN forward pass (intermediate=49152 for 72B) require more scratch space: 49152 × 4 bytes = 196 KB — fits in L2 (512 KB). The constraint forces us to ask: if the residual stream fits in L3 and the scratch activations fit in L2, what is the bottleneck? It is exclusively NVMe-to-CPU weight streaming bandwidth. The mechanism is: pre-sort GGUF weight layout so that each layer's matrices (W_Q, W_K, W_V, W_O, W_gate, W_up, W_down) are stored contiguously and aligned to 4K NVMe sector boundaries; initiate a direct sequential NVMe read (via ReadFile with non-buffered, direct I/O on Windows — `FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN`) so the OS page cache is bypassed and bytes flow NVMe → DMA → L3 → CPU registers directly. At NVMe sequential bandwidth of 3 GB/s, a single 72B-class layer's weights (72B × (1/64 layers) × 4.5 bpw / 8 ≈ 632 MB per layer — too large) ... recalibrate: a 7B model layer: 7B × (1/32 layers) × 4.5 bpw / 8 ≈ 12 MB per layer; at 3 GB/s sequential: 4 ms/layer, 32 layers = 128 ms/token ≈ 7.8 tok/s. This is the direct-NVMe streaming floor for a 7B model with NO DRAM buffer. For 13B: 25 MB/layer × 32 layers = 800 MB, 267 ms/token = 3.7 tok/s. The "no DRAM" constraint forces the NVMe-direct-stream path, which may outperform the llama.cpp mmap path on DRAM-starved hardware because it eliminates the mmap page-fault overhead and DRAM round-trip. The actual primitive is `FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN` on Windows, which engages the NVMe controller's sequential prefetch hardware prefetcher and bypasses the NTFS Cache Manager (avoiding DRAM residency). This is a real Windows API available on the Ryzen target; no kernel modifications needed.

### Residency arithmetic

7B model at Q4_K_M (4.5 bpw): 7B × 4.5 / 8 = 3.94 GiB on NVMe. DRAM used: d_model scratch only = 16 KB (residual) + 196 KB (FFN activations) + KV at 4K context INT4 (268 MB → too large for no-DRAM). Correction: KV at INT4 for 7B at 4K context, d_kv=128, 32 heads (GQA=4, 4 KV heads), 32 layers: 32 × 4 × 2 × 4096 × 128 × 0.5 = 67 MB — this must also be NVMe-resident under the constraint. Two-tier: hot KV (last 128 tokens) in L3 (2 MB), cold KV (4K-128) on NVMe (67 MB - 2 MB = 65 MB; at 3 GB/s sequential scan for each attention: 22 ms/layer attention scan). Total per-token: weight streaming 128 ms + attention KV scan 32 layers × 22 ms = 704 ms + 128 ms = 832 ms → 1.2 tok/s at 7B, NVMe-only. This is below the 3 tok/s target but the mechanism is new: TRUE zero-DRAM inference at 7B. At NVMe 3 GB/s, 7B NVMe-only floor is ~1 tok/s — competitive with current llama.cpp NVMe paths on the user's 7530U (reference: ~6.2 tok/s at 4B Qwen3 IQ4_XS implies bandwidth is not NVMe-limited for 4B; DRAM path is used there). The constraint's value is NOT higher tok/s — it's eliminating DRAM as a bottleneck entirely, enabling 70B on a 4 GB DRAM system.

### Novelty gloss vs kill list

Closest kill list: OS-Paging-Aware Weight Layout (R1/S2, killed for not reducing bpw). Structural difference: RORS does not just reorganize paging; it eliminates DRAM residency entirely by driving direct-NVMe-I/O with `FILE_FLAG_NO_BUFFERING`, bypassing the Windows page cache. The load-bearing novelty is the bypass — the OS page cache is explicitly not engaged, so DRAM is not touched. Closest published: Apple LLM-in-a-Flash streams weights from SSD to DRAM to GPU; RORS eliminates the DRAM stage. The "NVMe → L3 → registers" path with explicit no-buffer direct I/O has no published LLM-inference analog. This is the `[FREE SWING]` slot.

### Smallest experiment

Claim: `ReadFile` with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN` on the Ryzen 5 7530U achieves ≥ 2.5 GB/s sequential read throughput on a 100 MB test file without involving DRAM (measurable via perfmon `Memory\Available MBytes` staying flat during the read). Runtime: 20 min (write 100 MB test file, read it in non-buffered mode in a tight loop, measure throughput and RAM usage). Go/no-go: ≥ 2.5 GB/s with flat Available MBytes → GO (direct NVMe-to-cache path confirmed); < 1.5 GB/s or Available MBytes drops proportionally → NO-GO (page cache unavoidable on this Windows version). NO-GO finding: measures the minimum achievable NVMe streaming bandwidth on this hardware for non-buffered direct I/O, a number needed for all NVMe-offload ideas.

### Primary risk

Windows 11 Home may not honor `FILE_FLAG_NO_BUFFERING` for the user-space read path in a way that bypasses DRAM entirely — the DMA may still write to pinned DRAM pages before L3 prefetch. Mitigation: measure perfmon `Memory\Available MBytes` and `NVMe\Read Bytes/sec` simultaneously; if Available MBytes drops proportionally to read bytes, DRAM is being used and the constraint is not achievable on this hardware. If so, the experiment still produces the NVMe bandwidth floor, which is a useful structural finding.

---

## A5 — Gauge-Fixed Residual-Stream Inference (GFRS) [Track A]

**Constraint**: The computation graph must algebraically simplify under residual-stream gauge rotation. Specifically: the residual stream `h_t` is invariant under global orthogonal rotation `h_t → O h_t` (for any O ∈ O(d_model)) because every layer applies `h_t ← h_t + f(h_t)` and the norms are RMSNorm-invariant. The constraint: fix a canonical gauge — a specific O* that jointly diagonalizes some pair of weight matrices — and express all inference in that gauge. Any weight matrix that becomes diagonal or block-diagonal in the canonical gauge contributes zero off-diagonal bandwidth and can be stored and computed more cheaply.

**Constraint difficulty**: ~50% of standard mechanisms inadmissible. Standard weight layout (row-major, no coordinate dependence) is admissible only if the gauge is trivially identity. Inadmissible: any quantization scheme that treats weight rows as independent (standard per-row quantization ignores the global coordinate frame, leaving gauge-exposed redundancy on the table). Admissible: joint diagonalization of multiple matrices, rotation-then-quantize (GPTQ-style but with gauge-motivated rotation), any scheme that exploits the gauge symmetry to merge or eliminate operations.

### Mechanism

Track A. RMSNorm in Qwen3 computes `RMSNorm(h) = h / ‖h‖₂ · γ` where γ is a learned scale vector and the norm is coordinate-independent. The residual stream therefore has a gauge freedom: any global orthogonal rotation O of `h_t` — if applied consistently to all downstream weight matrices `W → WO^T` — leaves the final output invariant. The question is: which O* makes the most downstream weight matrices diagonal or block-diagonal? If W_Q and W_K jointly diagonalize under O* (i.e., `O* W_Q^T W_Q O*^T` and `O* W_K^T W_K O*^T` are both diagonal), then in gauge-fixed coordinates the attention weight products are diagonal operators — an inner-product becomes a pointwise scale instead of a dense dot-product. From CF11: W_Q r_99/d ≈ 0.63, W_K r_99/d ≈ 0.79. Joint eigenvectors of W_Q and W_K may exist if the two operators nearly commute: `‖W_Q W_Q^T - W_K W_K^T‖_F` is small. If small, a single O* jointly diagonalizes both, and the QK attention inner-product becomes O*(q) · O*(k) = (O* h_q) · (O* h_k) — a pointwise product in the rotated frame, cheap to compute. The gain is not compression per se but elimination of the full-rank dense matmul in the QK inner product (replacing it with a structured inner product in the gauge-fixed frame). The concrete implementation: compute O* = joint eigenvectors of (W_Q^T W_Q + W_K^T W_K), apply O* to all weight matrices at initialization (offline, once), store the rotated weights. At inference, no explicit rotation is needed because all matrices are pre-rotated. The primitive is `scipy.linalg.eig` / `numpy.linalg.eigh` for the joint eigendecomposition (standard library).

### Residency arithmetic

The gauge-fixing itself does not change residency in bits — all weight matrices still have the same number of parameters. The payoff is computational: in gauge-fixed coordinates, the QK inner product has a structured form. If W_Q becomes rank-K in the rotated frame (K < d_model), then the attention GEMV reduces from O(d²) to O(Kd). From CF11: W_Q r_99=1293/2048 at global K=128 gives ΔNLL=0.98 nats; at K=256 gives 0.51 nats. In the gauge-fixed frame, if the rotation concentrates the spectrum further (joint diagonalization pushes more variance into fewer dimensions), K may be smaller. Quantitative estimate: if O* reduces W_Q effective rank from r_99=1293 to r_99=600 (speculative — needs measurement), then the QK matmul cost drops by 53%, and since attention is ~20% of total FLOPs, total FLOPs drop by ~10%. Not a residency win, but a compute win on the matmul bottleneck. The residency-saving version: store the gauge-fixed W_Q and W_K at their effective reduced rank (project to K=256 dimensions), saving 2×(d² - Kd) parameters per layer per matrix: 2 × (2048² - 256×2048) × 28 layers × 2 bytes = 2 × (4M - 512K) × 28 × 2 = 2 × 3.5M × 28 × 2 ≈ 392 MB savings from W_Q + W_K alone at BF16.

### Novelty gloss vs kill list

Closest kill list: Sparse PCA Cross-Head Basis (R2/S2, killed — CoSpaDi and Share Your Attention pre-empt). Structural difference: GFRS is not a cross-head shared basis; it is a global gauge transformation on the residual stream that makes individual weight matrices diagonalizable. The gauge-fixing is algebraically different from basis sharing. Closest published: QuaRot / SpinQuant apply orthogonal rotations to residual streams for quantization purposes (outlier mitigation). GFRS applies the rotation specifically to jointly diagonalize the QK operators — a different target. The "joint diagonalization of W_Q and W_K as a residual-stream gauge fix" has no published analog.

### Smallest experiment

Claim: on Qwen3-1.7B-Base, the Frobenius distance ‖W_Q W_Q^T - W_K W_K^T‖_F / ‖W_Q W_Q^T‖_F < 0.30 for at least 50% of layers (measuring approximate joint diagonalizability). Runtime: ~10 min (compute W_Q W_Q^T and W_K W_K^T for all 28 layers in PyTorch BF16, compute Frobenius ratio). Go/no-go: ratio < 0.30 for ≥ 14 of 28 layers → GO (joint diagonalization is approximate and useful); ratio > 0.50 for all layers → NO-GO. NO-GO finding: measures the angular relationship between W_Q and W_K operators in trained Qwen3, providing a structural number about attention-weight co-structure not yet in the CF record.

### Primary risk

Joint diagonalization of W_Q W_Q^T and W_K W_K^T produces a gauge rotation O* that simplifies the QK inner product, but may make W_V and W_O less structured (gauge rotations don't commute with the RMSNorm scale vectors γ, so the simplification for QK may increase the effective rank of the FFN weights in the same gauge). Mitigation: measure the effective rank of W_gate and W_up in the O*-rotated frame; if they become more concentrated (or no worse), the gauge fix is cost-free.

---

## A6 — FSM-Bounded Vocabulary State Machine (FBVS) [Track A]

**Constraint**: The decoder's internal state, as it pertains to next-token selection, must be expressible as a finite state machine of N ≤ 512 states. No continuous latent state may influence the output distribution; only the current FSM state index (9 bits) and the last-emitted token (encoding the token ID) are permitted to influence which vocabulary entries are sampled.

**Constraint difficulty**: ~90% of standard mechanisms inadmissible. Ruled out: softmax over the full 151936-token vocabulary using the full residual stream (requires continuous latent state), attention over past context (requires context-length continuous state), any scheme where the output distribution is a continuous function of the hidden state. Admissible: lookup-table from (FSM state, last-token) → next-token distribution; learned transitions; Myhill-Nerode minimal FSM for the output distribution; integer-coded state transitions. This is not a language model in the standard sense — it is the extreme constraint that forces a Myhill-Nerode framing.

### Mechanism

Track A. The FSM constraint is a hard inversion: instead of computing `P(next_token | context)` via a continuous transformer, we ask: which part of the output distribution can be captured by an N-state FSM over the token sequence? The Myhill-Nerode theorem says the minimum FSM has states = equivalence classes of right-contexts. For structured outputs (JSON, code, constrained natural language), the equivalence classes are small — a JSON FSM for `{"key": value}` structures has O(50) states. For free natural language, the equivalence classes are theoretically infinite but empirically concentrated: the top-K most likely next tokens depend on the last 1-3 tokens, not the full context, for the majority of positions. The mechanism: train (or calibrate, no retraining) a 512-state FSM using the model's calibration outputs as supervision. The FSM is a 512 × vocab-size probability table (512 × 151936 × 2 bytes = 155 MB at BF16, or 512 × 151936 × 1 byte = 78 MB at INT8). At inference, the FSM runs as the primary decoder; the full transformer is invoked only when the FSM state entropy exceeds a threshold (per-token fallback). The FSM table is read sequentially from NVMe (512 states × next-token lookup = 512 × 1 random read, but at 4K block size = 1 read per state transition). The full transformer is the fallback for high-entropy states. This is the "O(1) tokens regardless of context" constraint in a different guise: the FSM produces the answer in O(1) from the table. The primitive: the Myhill-Nerode minimal automaton can be computed from transition-frequency data using Hopcroft's algorithm (`automata-lib` or `greenery` Python library, or custom implementation — all standard). The calibration FSM is built from 200 tokens of calibration text; each unique (state, token) pair is a transition; state merging uses Hopcroft minimization.

### Residency arithmetic

FSM table at INT8: 512 states × 151936 tokens × 1 byte = 78 MB. This fits in DRAM (7.28 GiB target). At each decode step: 1 table lookup (< 1 µs, fully DRAM-resident). Fallback transformer (Qwen3-1.7B at Q4): invoked at p% of steps. If p=20% (80% of steps handled by FSM), effective throughput: 80% × (1 µs lookup) + 20% × (full transformer at ~300 ms for 1.7B NVMe) = ~60 ms/token average → 16 tok/s. If p=50%: 150 ms average → 6.7 tok/s. The key question is what value of p achieves acceptable quality. On structured output tasks (code generation, JSON, constrained text), p may be < 10%, giving near-pure-FSM throughput. On free text, p may be > 80%, making the FSM a minor optimization. Quality cost: zero when fallback triggers; calibration FSM accuracy vs transformer depends on the task distribution.

### Novelty gloss vs kill list

No kill list item covers FSM-bounded decoders. Closest published: constrained decoding (LMQL, Outlines, Guidance) uses FSMs to constrain the output to a grammar, but the FULL transformer still runs — the FSM is a constraint mask, not a replacement decoder. FBVS proposes the FSM as the PRIMARY decoder with transformer fallback, inverting the standard relationship. The "FSM as primary decoder with transformer as fallback" framing has no published analog for general-purpose LLM inference. Myhill-Nerode applied to LLM next-token distributions is not in the published LLM literature.

### Smallest experiment

Claim: a 512-state Hopcroft-minimized FSM trained on 200-token calibration bigrams from Qwen3-1.7B achieves top-1 accuracy ≥ 0.45 on 200-token WikiText-2 held-out (where top-1 accuracy = fraction of positions where the FSM's argmax state matches the transformer's argmax token). Runtime: ~30 min (run Qwen3-1.7B on 400 tokens, extract bigram transition counts, run Hopcroft minimization, evaluate top-1 on held-out). Go/no-go: top-1 accuracy ≥ 0.45 → GO (FSM captures meaningful predictive signal); < 0.20 → NO-GO (FSM is near-random, fallback rate too high to be useful). NO-GO finding: measures the bigram predictive content of Qwen3-1.7B outputs — a structural number that bounds all FSM/n-gram acceleration schemes.

### Primary risk

The 512-state FSM cannot represent context-sensitive output distributions (grammar agreements, coreference, long-range dependencies); fallback rate p may be too high for the scheme to be cost-effective on any real task except heavily structured output. Mitigation: limit scope to structured output tasks (JSON, Python, YAML) in the smallest experiment; if fallback rate > 50% even on structured tasks, the idea is scoped to grammar-constrained decoding only (still useful, but narrower than proposed).

---

## A7 — Peer-Weight Gossip Inference (PWGI) [OUT-OF-ORIENTATION] [Track A]

**Note**: This idea exceeds the A orientation into distributed systems territory. Flagged `[OUT-OF-ORIENTATION]` because the mechanism is more naturally a Composition or Reach idea; included because the constraint (model cannot hold its own weights) forces a genuinely novel framing.

**Constraint**: The model cannot hold its own weights. Each inference step requires fetching the weight for layer L from a "peer" — a second copy of the model on the same machine stored in a different on-disk format. The primary copy is Q2_K (2.1 bpw); the peer copy is Q8_0 (8 bpw). The constraint forces a gossip-style weight reconciliation where the compute unit receives its weights from one of two peers at each step, not from a self-resident store.

**Constraint difficulty**: ~35% of standard mechanisms inadmissible. Ruled out: single-buffer weight loading (must manage two sources), stateless inference (must track which peer's weight was used), any scheme that requires a single consistent weight view. Admissible: per-layer source selection, quality-adaptive switching, dual-buffer prefetch, CRDT-style weight merging (average of Q2_K and Q8_0).

### Mechanism

Track A. The constraint forces a question the published literature has not asked: if two versions of the same model (Q2_K at 2.1 bpw and Q8_0 at 8 bpw) are resident on the same NVMe, can a per-layer source-selection policy achieve quality closer to Q8_0 at bandwidth closer to Q2_K? This is a CRDT-inspired framing: the "peer weights" are two consistent views of the same model at different fidelity. The policy: at each layer, observe the calibration-derived layer sensitivity (from CF5: W_up more rank-sensitive; from CF11: W_Q compressed at K=128 with ΔNLL=0.98; deeper layers spread outliers per CF3). Use a per-layer binary gate to select Q2_K (cheap, noisy) vs Q8_0 (expensive, precise) for each weight matrix. The gate is static (determined by calibration, no per-token computation) — this is NOT dynamic activation prediction (killed by CF1 failure). The effective bpw is `p × 2.1 + (1-p) × 8` where p is the fraction of layers using Q2_K. If p=0.6: effective bpw = 0.6×2.1 + 0.4×8 = 4.46 bpw — comparable to Q4 but with a quality profile determined by which layers get the high-fidelity treatment. CF11 says attention W_Q is more compressible than MLP; CF5 says W_up is more rank-sensitive. Policy: use Q2_K for W_Q (CF11 low-rank) and Q8_0 for W_up (CF5 high sensitivity). The primitive: standard llama.cpp model loading modified to open two GGUF files and select per-tensor. No new math required; `llama_tensor_load` extended with a source-selection table.

### Residency arithmetic

Qwen3-7B. Q2_K: 7B × 2.1 / 8 = 1.84 GiB. Q8_0: 7B × 8 / 8 = 7.0 GiB (exceeds 7.28 GiB RAM). So Q8_0 must be NVMe-resident. Policy: W_Q, W_K (attention, CF11-motivated compressed) from Q2_K (cheap); W_up, W_down, W_gate (MLP, CF5/CF8-motivated sensitive) from Q8_0. MLP parameters ≈ 2/3 of total (7B × 2/3 = 4.7B at Q8_0 = 4.7 GiB) + attention W_Q, W_K from Q2_K (7B × 1/3 × 2.1/8 = 0.61 GiB). Total: 5.3 GiB. Fits in 7.28 GiB with KV headroom. Quality: MLP at Q8_0 (lossless) + attention at Q2_K (degraded but CF11 says W_Q is more compressible). Expected dNLL: < 0.5 nats for W_Q at Q2_K per CF11 extrapolation (K=128 gives +0.98 nats for pure SVD; Q2_K random quantization with 2.1 bpw is a different degradation mode — estimate < 0.5 nats for attention weights). Total effective bpw: 5.3/7 × 8 = 6.0 bpw for this mixed-precision variant. Not 2.1 bpw, but quality-adaptive: the sensitive weights (MLP) are at Q8_0, the compressible weights (attention) at Q2_K.

### Novelty gloss vs kill list

Closest kill list: Per-Token Multi-Level bpw (R1/S2, killed — multiple-version storage breaks residency math). Structural difference: PWGI stores two fixed-precision models (not per-token variable versions) and uses a static per-layer policy to select which to read — not a per-token policy. The static policy does not break residency math. Closest published: mixed-precision quantization (LLM.int8(), GPTQ with per-layer bpw schedules) — but these are STORED as one mixed-precision model. PWGI stores two separate full-precision models and selects at inference time, enabling A/B testing and graceful degradation. The "dual-model peer selection at inference" framing is not in the published literature.

### Smallest experiment

Claim: loading W_Q from Q2_K and all other weights from Q4_K_M on Qwen3-1.7B achieves ΔNLL < 0.5 nats vs full Q4_K_M, measurable in PyTorch by patch-loading the W_Q tensors from a second GGUF file. Runtime: ~20 min. Go/no-go: ΔNLL < 0.5 nats → GO; > 1.0 nats → NO-GO. NO-GO finding: measures the quality penalty of Q2_K on W_Q specifically, bounding the mixed-precision attention-compression benefit.

### Primary risk

The dual-file I/O pattern may thrash NVMe I/O scheduling (interleaved reads from two large files defeats sequential prefetch). Mitigation: pre-sort both GGUF files in the same layer-sequential order; read Q2_K for layer L attention weights, then Q8_0 for layer L MLP weights, advancing both file pointers sequentially.

---

## Convergence handles

1. **W_up inner-product as firing-rank gate** (CF1 — used by A1 tropical dispatch, motivates routing on up-projection side rather than gate-projection side)
2. **Head-redundancy in W_Q: 16 heads → ~128-dim subspace** (CF11 — used by A2 content-addressable dedup, A5 gauge-fixing, A7 peer-weight attention compression)
3. **NVMe sequential-vs-random read bandwidth gap** (3 GB/s vs 0.7 GB/s — used by A3 append-only log, A4 direct-NVMe streaming; actual CF13* — verify in smallest experiments for A3 and A4)
4. **INT8-quantized action signature as deduplication key** (A2 primitive — measures head angular diversity in W_Q; converges with CF11 if any pair of heads has cos > 0.85)
5. **Frobenius ratio ‖W_Q W_Q^T − W_K W_K^T‖_F / ‖W_Q W_Q^T‖_F** (A5 gauge-fixing primitive — not yet in CF record; measures whether joint diagonalization of Q and K is tractable; links to CF11 head-sharing structure)
6. **Static per-layer calibration gate on weight precision** (A7 primitive — calibration-derived binary gate selecting per-layer quality; links to CF5 W_up sensitivity and CF11 W_Q compressibility; converges with any orientation-C composition of CF5 × CF11)
