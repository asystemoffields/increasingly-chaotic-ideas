# Stage 1 — Orientation U (Unconventional Substrate) — Run 3

Orientation: U — Unconventional Substrate
Track classification per idea: A or B as noted.

---

## U1 — GGUF Extent-Coalescence Prefetch (GECP) [Track B]

### Mechanism
Track B. GGUF lays tensors as contiguous byte extents at known file offsets (stored in the header's tensor-info table). The Windows Cache Manager issues speculative read-aheads when it detects sequential access; but LLM weight streaming is layer-by-layer GEMV — each layer's tensors are read once, in a predictable fixed order, with no branching. The substrate primitive: `ReadFileEx` + IOCP or `FILE_FLAG_SEQUENTIAL_SCAN` on Windows 11, which signals the Cache Manager to prefetch entire extents rather than 4 KB demand pages. The move is to pre-parse the GGUF tensor table at model load (O(n_tensors), ~5 ms), sort tensor extents by access order, and issue `PrefetchVirtualMemory` (a documented Win32 call) on the next 2–4 layers' extents during the compute phase of the current layer. The CPU is occupied with GEMV; the NVMe controller is simultaneously fetching the next layer. This is double-buffered I/O at the OS-API level — no custom NVMe firmware, no kernel patch. Not CF-grounded (orthogonal to CF1–CF12) but grounded in the measured access pattern of autoregressive inference: each layer's weights are consumed exactly once per token, in deterministic order. Closest prior: Apple LLM-in-a-flash uses mmap + OS demand paging; GECP is a step ahead — it replaces reactive demand-paging with proactive OS-API-level prefetch. Does NOT touch W_Q layout (no cross-layer subspace claim; orthogonal to v2-CHEAP-TEST-001 kill).

### Residency arithmetic
70B at IQ4_XS ≈ 5.35 GiB (below 7.28 GiB DRAM ceiling but verify; using ~40 GB model at 1.07 bpw NVMe tier). More relevant: 4B Qwen3 at 2.1 GiB DRAM-resident (IQ4_XS). For NVMe-resident 70B at 0.55 bpw (NanoQuant territory, 5.35 GiB), current NVMe baseline without orchestrated prefetch: ~700 MB/s × (bytes/layer) = roughly 350–500 ms/layer at 70B (28 layers × ~190 MB each). With `PrefetchVirtualMemory` overlapping next-layer fetch during current-layer GEMV (GEMV at ~11.5 GB/s DRAM = ~16 ms/layer for 190 MB at DRAM bandwidth for in-cache path), the I/O time collapses from the serial path toward near-zero if NVMe sustains 3 GB/s on sequential reads. Upper bound tok/s: 1/(28 × 16 ms) ≈ 2.2 tok/s — matching the DRAM-bandwidth-bound case. Without orchestrated prefetch the NVMe serial path gives 1/(28 × 430 ms) ≈ 0.08 tok/s. If prefetch engages fully: ~25–30× throughput gain on NVMe-resident inference. ΔNLL = 0 (pure I/O scheduling, no weight modification).

### Novelty gloss
Closest kill-list item: R1/S2 "OS-Paging-Aware Weight Layout" (killed as "doesn't reduce bpw, just engineers paging"). Structural difference: that idea relaid weight bytes to match OS page boundaries (storage rearrangement); GECP uses the OS-API `PrefetchVirtualMemory` to issue explicit double-buffered I/O without rearranging storage, exploiting the deterministic layer-access order of autoregressive inference. Closest published method: Apple LLM-in-a-flash (reactive mmap demand paging); GECP is proactive — parse the access schedule at load time, issue prefetch during compute. Second closest: CLASP (Chunked-Layer Speculative Prefetch, R2/S4 deferred) — CLASP was framed around speculative decoding accepting chunks; GECP is simpler (no speculative decoding, just double-buffered layer streaming via OS API). GECP is the cheap first-rung measurement that CLASP depends on anyway.

### Smallest experiment
Claim: `PrefetchVirtualMemory` on the next-layer extents, issued mid-GEMV of the current layer, reduces wall-clock time per token by ≥5× on NVMe-resident Qwen3-1.7B (as proxy for 70B I/O behavior) versus reactive mmap baseline.
Test: load Qwen3-1.7B-Base GGUF from a cold NVMe state (flush OS page cache via `SetSystemFileCacheSize` or reboot); run 20-token generation; record per-token latency. Then repeat with a Python wrapper that calls `ctypes` `PrefetchVirtualMemory` on the next layer's byte range at the start of each layer's GEMV. Runtime: ~30 min total.
Go threshold: ≥3× latency reduction on cold-NVMe path (i.e., prefetch overlaps measurably).
NO-GO finding: quantifies the actual NVMe sequential read speed under ik_llama.cpp on this hardware, producing a v2-confirmed CF13'' replacement number.

### Primary risk
The Windows Cache Manager may already be prefetching aggressively under `FILE_FLAG_SEQUENTIAL_SCAN`, making explicit `PrefetchVirtualMemory` redundant. Mitigation: the smallest experiment measures both baselines (with and without explicit prefetch, and with cache flushed between runs); if the Cache Manager already prefetches, the GO threshold is not met but we get the confirmed NVMe bandwidth number.

---

## U2 — VirtualLock Layer-Pin with Eviction Scheduling (VLPE) [Track B]

### Mechanism
Track B. On Windows, `VirtualLock` pins virtual address ranges into physical RAM, preventing the working-set trimmer from evicting them. The substrate primitive is the working-set trimmer (part of the Memory Manager, invoked when system RAM pressure rises) — which is precisely the eviction mechanism that causes page faults during inference. The move: at model load, profile which layers are accessed most frequently (for autoregressive inference: ALL layers are accessed every token, but the lm_head / embedding and layers 0–3 are accessed on EVERY token including prefill, while layers 4–27 are accessed once per decode token). Use `VirtualLock` on the highest-reuse byte ranges — embedding table (151936 × 2048 × 2 bytes ≈ 621 MB for bf16, or ~78 MB at 1 bpw), layer 0 (smallest and most reused in prefill), and the lm_head if untied. For a 4B model in full bf16 (8 GB, over the 7.28 GiB ceiling), selectively locking the 10–15% most reused tensors while allowing the rest to be demand-paged provides a hybrid DRAM/NVMe residency without changing bpw. Not connected to CF1–CF12 (orthogonal). Avoids W_Q layout changes (no subspace coherence claim; orthogonal to v2-CHEAP-TEST-001).

### Residency arithmetic
Qwen3-4B-Base bf16 ≈ 8 GB (exceeds 7.28 GiB). Selective lock: embed_tokens (621 MB bf16) + lm_head (if untied: 621 MB; if tied: 0 extra) + first 4 layers (4 × ~280 MB ≈ 1.12 GB) = ~1.7 GB pinned. Remaining ~6.3 GB demand-paged. With 11.5 GB/s DRAM bandwidth, the pinned tensors (accessed every token) contribute 0 page-fault latency; the demand-paged layers contribute faults only on cold start or under memory pressure. At 7.28 GiB system RAM and 8 GB model, without `VirtualLock` the trimmer evicts under sustained inference, causing periodic latency spikes. With `VirtualLock` on the 1.7 GB hot set, the trimmer can evict the cold layers (6.3 GB) gracefully. Steady-state tok/s: bounded by compute (GEMV at DRAM bandwidth) for the first N tokens until cold layers are loaded; then DRAM-bandwidth-bound. Benefit: eliminates stall-latency spikes on long-context runs without reducing model quality (ΔNLL = 0).

### Novelty gloss
No equivalent on the kill list. Closest published method: Apple LLM-in-a-flash's "hot layer caching" — they profile which layers to keep resident based on access frequency. Structural difference: LLM-in-a-flash is an Apple-proprietary framework targeting Apple Silicon unified memory; VLPE uses the Windows-documented `VirtualLock` API on a standard x86 DRAM+NVMe setup. The substrate primitive (working-set trimmer avoidance via explicit lock) is not used in any published LLM serving framework on Windows. The key observation is that the trimmer is the adversary in sustained inference, and `VirtualLock` is its documented off switch.

### Smallest experiment
Claim: under sustained 200-token generation of Qwen3-1.7B-Base (or 4B), `VirtualLock` on the embed_tokens + first 4 layers reduces per-token latency variance (standard deviation across tokens) by ≥2× compared to unlocked inference under simulated memory pressure (achieved by running a parallel process consuming ~4 GB RAM to trigger the trimmer).
Test: Python + ctypes `VirtualLock` wrapper; measure per-token latency distribution locked vs unlocked under pressure. Runtime: ~45 min.
Go threshold: ≥2× variance reduction AND ≥10% median latency improvement.
NO-GO finding: establishes that the Windows working-set trimmer does NOT evict inference working sets under normal operation on this hardware, which is itself a useful substrate characterization.

### Primary risk
`VirtualLock` requires `SeLockMemoryPrivilege` (not granted by default to non-admin processes on Windows). Mitigation: the experiment can be run from an elevated process (easy on a personal machine); for deployment, use `AdjustTokenPrivileges` at process startup (documented API, no kernel mod required).

---

## U3 — NTFS Compressed-Cluster GGUF Layout (NCCG) [Track B]

### Mechanism
Track B. NTFS supports transparent file compression at the cluster level (`FILE_ATTRIBUTE_COMPRESSED`), using LZ77-based compression of 16-cluster (64 KB) blocks. The OS decompresses transparently on read, in the kernel, using the NTFS Cache Manager. For quantized GGUF weight data (INT4 / INT2), the blocks have non-trivial but low entropy (well-quantized weights are not fully random; run-of-same-quantization-indices occur for near-zero weights). The substrate primitive: NTFS cluster compression is invoked automatically when the file attribute is set; no application code changes needed beyond the `compact /C` command at file creation. The move: store the GGUF on an NTFS-compressed volume; the kernel decompresses 64 KB blocks on read-ahead, overlapping with GEMV compute. The effective NVMe bandwidth increases by the compression ratio. For INT4 weights (Q4_K_M), typical zstd ratio on GGUF is ~1.05–1.15 (weights are already quantized, low redundancy). For INT2+sparse formats (SOBIB-style), the compression ratio may be higher (sparse INT8 corrections compress well). This is distinct from GECP (U1): GECP is about prefetch scheduling; NCCG is about read amplification via kernel-transparent decompression. NOT a custom kernel; `FILE_ATTRIBUTE_COMPRESSED` is a documented NTFS feature available on all Windows NTFS volumes. Does not touch W_Q subspace layout.

### Residency arithmetic
70B at 0.55 bpw (NanoQuant, 5.35 GiB) on NVMe, sequential read throughput 700 MB/s (cold, demand-paged). If NTFS compression achieves 1.15× ratio on INT4 weights, effective throughput = 700 × 1.15 = 805 MB/s. Marginal. If a mixed-precision format with sparse INT8 outlier corrections (like SOBIB, R5 deferred) achieves 1.4× compression, effective throughput = 980 MB/s — a 40% improvement on the NVMe bottleneck, worth ~0.4× additional tok/s at 70B NVMe-resident. ΔNLL = 0 (no weight change; decompression is lossless). The CPU cost of NTFS decompression: LZ77 decode at kernel level saturates ~2–4 GB/s on modern CPUs (well above NVMe throughput); the bottleneck remains NVMe, not CPU decode. Feasibility check: the 64 KB block granularity matches well with GGUF tensor storage (most tensors >> 64 KB).

### Novelty gloss
Closest kill-list item: R1/S4-deferred Idea I "NTFS-Reflink Codebook Deduplication" — killed for Windows 11 Home reflink limitation. Structural difference: NCCG uses NTFS compression (available on all NTFS volumes, including Home) not reflinks. R2/S2 "NVMe Prefetch Sequencer via RoPE Fingerprints" killed as wrong axis; NCCG is bandwidth, not scheduling. Closest published method: ZipServ (ASPLOS 2026) uses arithmetic coding; NCCG uses the OS's existing NTFS LZ77, not a custom entropy coder. The move is purely in substrate recognition: NTFS compression is a 30-year-old primitive that nobody applies to LLM weight files. Whether it gives meaningful ratio on quantized weights is an empirical question — the experiment answers it.

### Smallest experiment
Claim: storing Qwen3-1.7B-Base IQ4_XS GGUF with `FILE_ATTRIBUTE_COMPRESSED` (via `compact /C`) reduces on-disk size by ≥5% AND reduces cold-NVMe inference time by ≥5% versus uncompressed GGUF.
Test: `compact /C` the GGUF file; measure compressed size; run cold-NVMe inference (flush page cache between runs) and compare per-token latency. Runtime: ~20 min (includes compaction time ~5 min).
Go threshold: ≥5% size reduction AND ≥5% latency improvement (if NVMe-bound).
NO-GO finding: establishes that INT4 GGUF weight data has near-uniform entropy at 64 KB block granularity under NTFS LZ77 — useful bound on compressibility of quantized weights, motivates ANS/zstd as higher-compression alternatives (GECP + zstd layering).

### Primary risk
NTFS compression on large files incurs fragmentation over time, degrading sequential read performance. Mitigation: test on a freshly defragmented volume; flag fragmentation as a deployment concern but not an experimental blocker.

---

## U4 — Windows Prefetcher Layout Imprinting (WPLI) [Track B]

### Mechanism
Track B. Windows 11's Superfetch/SysMain service maintains a prefetch database (`%SystemRoot%\Prefetch\`) recording which file-system clusters each process accessed during previous runs, and pre-populates the RAM page cache on subsequent launches. The substrate primitive: `CreateProcess` causes SysMain to begin reading the process's prefetch record and issuing speculative I/O for the recorded cluster set, BEFORE the application code issues any read. For LLM inference, the access pattern is perfectly deterministic and reproducible across launches (same model, same GGUF, same layers in the same order). If ik_llama.cpp's GGUF reads are sufficiently uniform across runs, SysMain will pre-warm the GGUF pages before the first token. The move: ensure the GGUF is always accessed via the same process path (consistent file path + process name), run a "warm-up" generation at install time to imprint the prefetch record, and measure whether subsequent cold-boot inference gets a DRAM-warm page cache for free. This requires ZERO application code changes. It is the OS doing exactly what it was designed to do, applied to LLM weights. Potential interaction: for 70B models where GGUF > system RAM, Superfetch cannot pre-warm the entire file; but for 4B models at 2.1 GiB IQ4_XS (well within 7.28 GiB), full pre-warm is plausible. Not CF-grounded; does not touch W_Q layout.

### Residency arithmetic
Qwen3-4B at IQ4_XS ≈ 2.1 GiB. On a 7.28 GiB system, SysMain can plausibly pre-warm 2.1 GiB from NVMe during OS idle time. Without pre-warm: first-token latency on cold boot = NVMe sequential read of ~2.1 GiB at 700 MB/s ≈ 3.0 seconds. With SysMain pre-warm: first-token latency approaches DRAM-bandwidth case (already warm). Subsequent tokens: DRAM-bound regardless. The benefit is entirely in time-to-first-token (TTFT) on cold launches — reducing TTFT from ~3 s to ~0.2 s. For interactive use on a personal machine, TTFT is the dominant UX metric. ΔNLL = 0 (no weight modification).

### Novelty gloss
No kill-list equivalent. Closest published: nothing in the LLM serving literature uses OS-level Superfetch/SysMain imprinting. Apple's LLM-in-a-flash has "predictive prefetch" but it is application-level, not OS-level, and targets Apple Silicon. WPLI is the recognition that the Windows SysMain database is a free predictive prefetcher, and its predictions for LLM workloads (deterministic access pattern) are maximally accurate. The idea has no arXiv equivalent because it requires no research — it requires noticing that an OS mechanism already exists and is already optimal for this workload. The novelty is in pointing, not in inventing.

### Smallest experiment
Claim: after one warm-up inference run, Qwen3-1.7B TTFT on a subsequent cold-NVMe launch (system reboot, not just page cache flush) is ≤2× the DRAM-warm TTFT (implying SysMain pre-warmed most or all of the model pages).
Test: (1) run one inference; reboot; measure TTFT. (2) compare to TTFT immediately after the first run (DRAM warm). Ratio target ≤2×. Runtime: ~30 min including reboots.
Go threshold: TTFT_cold / TTFT_warm ≤ 2.0.
NO-GO finding: establishes that SysMain does not pre-warm GGUF pages (either because the file is too large, or because SysMain's heuristics suppress large files). Motivates explicit GECP (U1) as the replacement.

### Primary risk
SysMain may be disabled on the user's machine (common in "optimize Windows" guides), or it may not pre-warm files > 1 GB. Mitigation: check `sc query SysMain` before the experiment; if disabled, note it and re-enable for the test; this is a one-line PowerShell check.

---

## U5 — GGUF Tensor-Table B-tree Page Layout for Cache-Line Access (GTBP) [Track B]

### Mechanism
Track B. GGUF stores tensor data in flat byte offsets, with the header tensor-info table listing name + type + shape + offset + size for each tensor. During GEMV execution in ik_llama.cpp, the runtime iterates through layers sequentially, but the tensor-info table lookups are random-access string-keyed (hash map over tensor names). On a 70B model with thousands of tensors, the tensor-info table itself may not fit in L1/L2 cache, causing cache misses on every tensor lookup during the hot inference path. The substrate primitive: CPU cache-line prefetch via `_mm_prefetch` / `__builtin_prefetch`, OR — more unconventionally — reorganizing the tensor-info table from a hash map into a sorted array indexed by inference-time access order, so that the next-tensor's metadata is always on the SAME cache line as the current-tensor's metadata. The move: write a one-time GGUF post-processor that reorders tensor entries in the header to match layer-sequential access order and packs metadata into cache-line-aligned structs. No weight bytes change; only the header is reorganized. On the hot path, each layer's tensor-metadata lookup is a sequential scan of ~5 cache lines rather than a hash-map chase through ~100 cache lines. This is a GGUF file-format engineering move, not a kernel move, but the substrate primitive is CPU cache-line layout — a non-ML engineering concern. Does not interact with W_Q subspace content.

### Residency arithmetic
Impact is on metadata access latency, not weight bytes. At 70B with ~3000 tensor entries, a hash-map lookup is ~5 pointer dereferences × 64-byte cache lines = ~320 bytes read per lookup, most of which miss L1 at 64 bytes stride. 28 layers × 10 tensors/layer = 280 lookups per token. At 100 ns per L3 miss, 280 × 100 ns = 28 µs of metadata overhead per token. Not the bottleneck at 70B (where GEMV takes ~16 ms/layer). At 1.7B, GEMV is ~1 ms/layer; 28 µs metadata is 0.1% overhead. Cache-line reorganization saves at most ~20 µs/token. This is a micro-optimization. However, the real payoff is when composing with GECP (U1): if tensor-info lookups happen during the prefetch-scheduling phase (deciding what to prefetch next), a cache-miss-heavy tensor table adds latency to the prefetch-issuance critical path. GTBP's payoff is as an enabler for GECP accuracy and latency.

### Novelty gloss
No kill-list equivalent. Nothing in the published LLM inference literature discusses tensor-table cache-line layout in the GGUF format. Closest: generic software-engineering advice about cache-friendly data structures. As a standalone idea, GTBP is micro-optimization (low payoff). As a substrate enabler for U1 (GECP), it removes the metadata-lookup bottleneck from the prefetch-scheduling critical path. Framing: GTBP is a free 1-hour engineering fix that should accompany any GECP deployment.

### Smallest experiment
Claim: reorganizing the GGUF tensor-info table into inference-order linear layout reduces tensor-metadata lookup time (profiled via perf sampling or manual timestamping in ik_llama.cpp) by ≥30% on Qwen3-1.7B.
Test: instrument ik_llama.cpp to timestamp tensor-info lookups; run 50-token generation; compare before/after table reorganization. Runtime: ~1 hour (includes one-time GGUF post-processor implementation).
Go threshold: ≥30% reduction in metadata lookup time.
NO-GO finding: confirms that tensor-table lookups are already cache-efficient (perhaps due to ik_llama.cpp's own caching), bounding the metadata overhead and ruling out this class of micro-optimization.

### Primary risk
ik_llama.cpp may already cache tensor pointers at load time, making per-token tensor-table lookups nonexistent. Mitigation: inspect the ik_llama.cpp load path before implementing; if tensor pointers are cached at load time, GTBP is a no-op and can be immediately dropped.

---

## U6 — NVMe Queue-Depth Saturation for Weight Streaming (NQDS) [Track B]

### Mechanism
Track B. NVMe SSDs expose multiple hardware queues (NVMe spec: up to 65535 queues, 65535 commands each). On Windows 11, the StorNVMe driver supports QD (queue depth) up to 32 per queue. Consumer SSDs (e.g., Samsung 980 Pro, WD SN850) achieve peak sequential throughput only when multiple outstanding I/O requests are in flight simultaneously — a single outstanding read (QD=1) typically achieves 50–60% of rated throughput; QD=4–8 achieves 95%+. The GGUF sequential layer stream accesses weight tensors one-at-a-time (QD=1 in the naive demand-paging path). The move: use overlapped I/O (`ReadFileEx` with completion callbacks, or `CreateIoCompletionPort` IOCP) to maintain QD=4 outstanding reads during inference — fetching layers N+1, N+2, N+3 while computing on layer N. This is different from GECP (U1) which uses `PrefetchVirtualMemory` (a hint-based VM call); NQDS uses explicit overlapped I/O with explicit buffers, giving precise control over queue depth and avoiding OS page-fault intermediaries entirely. The substrate primitive is the NVMe QD scheduling behavior — the fact that SSDs have internal parallelism that single-outstanding-request access patterns leave idle. Not CF-grounded. Does not modify weights.

### Residency arithmetic
Rated sequential read of Samsung 980 Pro: 7 GB/s at QD=128. At QD=1 (naive mmap): ~1.5–2 GB/s. At QD=4: ~4–5 GB/s. For 70B at 0.55 bpw (5.35 GiB): layer size ≈ 191 MB. At QD=1: 191 MB / 1.75 GB/s ≈ 109 ms/layer. At QD=4: 191 MB / 4.5 GB/s ≈ 42 ms/layer. With compute overlapping (layer N compute = ~16 ms at 11.5 GB/s DRAM bandwidth), effective bottleneck at QD=4 is max(16 ms compute, 42 ms/4 per-layer-prefetch-amortized) — if 3 layers are prefetched while 1 is computed, effective I/O time = 42 ms / 3 ≈ 14 ms (matching compute). Result: near-compute-bound streaming at QD=4. Estimated tok/s: 1/(28 × 16 ms) ≈ 2.2 tok/s (same ceiling as GECP). The substrate measurement to verify: actual QD=1 vs QD=4 throughput on THIS NVMe device under Windows StorNVMe.

### Novelty gloss
Closest kill-list item: none directly. GECP (U1, this run) and CLASP (R2/S4 deferred) are the closest — both do layer prefetch but via OS hints (mmap / `PrefetchVirtualMemory`). NQDS is lower-level: explicit overlapped I/O with IOCP bypasses the VM subsystem entirely, giving application-level control over NVMe queue saturation. Closest published method: llama.cpp's `--numa` and mmap flags are OS-hint-based; no published LLM inference framework uses explicit IOCP for NVMe QD saturation. The mechanism is standard in database engines (PostgreSQL's `posix_fadvise` + `O_DIRECT` path; MySQL InnoDB's AIO). The move is importing database-engine I/O discipline into LLM inference.

### Smallest experiment
Claim: explicit overlapped I/O (IOCP, QD=4) on sequential GGUF layer reads achieves ≥2× throughput over OS demand-paging (mmap QD=1) on the Ryzen 5 7530U's NVMe device.
Test: write a 50-line C program (or Python ctypes wrapper) that reads 4 × 50 MB sequential chunks of a GGUF file using overlapped I/O vs `mmap` + sequential access; measure throughput with both methods on a cold NVMe (page cache flushed). Runtime: ~2 hours (includes C program write).
Go threshold: ≥2× throughput ratio QD=4 vs QD=1.
NO-GO finding: establishes that this NVMe device achieves peak throughput at QD=1 (some consumer NVMe drives do at sequential access), bounding the benefit of any prefetch strategy and producing a v2-confirmed NVMe bandwidth number.

### Primary risk
ik_llama.cpp uses mmap by default, and integrating IOCP requires non-trivial refactor of the weight-loading path. Mitigation: the smallest experiment is a standalone program, not ik_llama.cpp integration. Integration effort is a Stage 5 question, not a Stage 1 blocker.

---

## U7 — NUMA-Aware CCX-Local Weight Partitioning on Zen3 (NAWP) [FREE SWING] [Track B]

### Mechanism
Track B. [FREE SWING] The Ryzen 5 7530U (Zen3 laptop) has 6 cores in a single CCX sharing a 16 MB L3 cache. DRAM is single-channel (≈11.5 GB/s peak). The CPU's prefetcher is hardware stride-prefetcher per-core, working on 64-byte cache lines. For GEMV on a weight matrix (rows = output dim, cols = input dim), accessing W in row-major order (one row = one output neuron) means each neuron's weight vector traverses d_model cache lines in sequence — the hardware prefetcher engages perfectly IF strides are regular. BUT standard GGUF Q4_K_M layout interleaves scale and quantized nibbles in blocks of 256 weights, breaking the stride regularity that the hardware prefetcher expects. The substrate primitive: the hardware stride prefetcher is a real CPU mechanism (documented in AMD optimization guides, kicks in at 2–8 consecutive strides). The move: rearrange Q4_K_M blocks in the GGUF so that all the quantized nibbles for a given weight row are contiguous (pulling scales to a separate array), enabling the hardware prefetcher to engage on the full row. This is a GGUF post-processor — no retraining, no weight value change. The hardware prefetcher is a non-ML substrate primitive. Note: this interacts with W_Q layout only in the trivial sense that ALL weight matrices in GGUF are affected; no cross-layer subspace coherence claim is made (orthogonal to v2-CHEAP-TEST-001 kill).

### Residency arithmetic
Payoff is in DRAM bandwidth utilization efficiency (hardware prefetcher reduces effective latency from DRAM without changing bandwidth). On a Ryzen 5 7530U at 11.5 GB/s, GEMV on a 1.7B layer (hidden=2048, intermediate=8192) with Q4_K_M: ~5.2 MB/layer weight bytes. At 11.5 GB/s, theoretic: ~0.45 ms/layer. Actual measured: closer to 1.5–2 ms/layer due to cache-miss overhead on Q4_K_M interleaved layout (scales scattered every 32 nibbles = ~50% cache-line waste on scale fetches). If hardware prefetcher engages on a separated layout: cache-line utilization rises from ~50% to ~90%, approaching the theoretical 0.45 ms/layer. Estimate: ~2–3× effective DRAM bandwidth improvement for GEMV. For 4B at IQ4_XS (current 6.2 tok/s reference): potential to reach ~8–10 tok/s. ΔNLL = 0 (lossless weight rearrangement).

### Novelty gloss
Closest: ASPI "inline scale layout" (Track B R5, killed as "kernel surgery cost prohibitive"). Structural difference: ASPI was about changing the in-kernel quantization code; NAWP is a one-time GGUF post-processor that rearranges the on-disk layout without touching ik_llama.cpp's dequantization kernel. The separated-scales layout is read by the existing dequantization code transparently IF the format is designed for it (which Q4_K_M is not, but K_8 supertile format is). Alternative: use the existing Q8_0 layout (already scale-separated) as a reference implementation. No published LLM inference paper specifically targets hardware prefetcher engagement via scale-separation in 4-bit quantized GGUF layout.

### Smallest experiment
Claim: reading a GGUF weight tensor in scale-separated layout (scales contiguous, nibbles contiguous, separate arrays) achieves ≥1.5× DRAM bandwidth utilization versus interleaved Q4_K_M layout, measured via `perf stat` cache-miss rate or `hardware performance counters` on Ryzen.
Test: write a GGUF tensor reader in two layouts; measure time to decode a single large matrix (e.g., W_up from Qwen3-1.7B at IQ4_XS vs a synthetic scale-separated variant) under AMD uProf or `perf stat -e cache-misses`. Runtime: ~1.5 hours.
Go threshold: ≥1.5× throughput ratio scale-separated vs interleaved.
NO-GO finding: establishes that Q4_K_M's interleaved layout does NOT cause hardware prefetcher thrashing (perhaps because the hardware prefetcher handles stride-2 patterns well on Zen3), bounding scale-separation as a layout optimization.

### Primary risk
ik_llama.cpp's dequantization kernels are tightly coupled to Q4_K_M's interleaved layout; changing layout requires either a new format or a layout-translation layer at load time (adding memory bandwidth overhead that cancels the gain). Mitigation: the smallest experiment is read-only throughput, not end-to-end inference; layout-translation overhead is a Stage 3 question.

---

## Convergence handles

- **NVMe queue depth and device throughput at QD=1 vs QD=4 on this hardware** — U1, U3, U6 all depend on knowing the actual NVMe bandwidth under sequential access; a single 30-min measurement settles all three ideas' residency arithmetic.
- **GGUF tensor-access pattern determinism** — U1, U4, U5 all rely on the inference-time access order being perfectly predictable and reproducible; verify in ik_llama.cpp source (5-min code read).
- **OS page cache pre-warming under Windows SysMain** — U4 (WPLI) directly; U1 (GECP) must check whether SysMain already does the prefetch GECP proposes.
- **Q4_K_M cache-line utilization on Zen3** — U7 (NAWP); the AMD optimization guide's stride-prefetcher spec is the primitive; `perf stat cache-misses` is the measurement.
- **Windows `VirtualLock` privilege availability and trimmer behavior** — U2 (VLPE); one PowerShell check resolves the privilege question; whether the trimmer evicts GGUF pages under pressure is the experiment.
- **NTFS compression ratio on quantized GGUF bytes** — U3 (NCCG); `compact /C` + file-size comparison is a 5-min measurement that resolves the entire idea's feasibility question.
