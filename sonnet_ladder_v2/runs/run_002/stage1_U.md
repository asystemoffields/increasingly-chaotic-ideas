# Stage 1 — Orientation U (Unconventional Substrate) — Run 002

Orientation: U — Unconventional Substrate
Run: 002 (independent cold start)

---

## U-1: GGUF Extent Coalescing via Windows Cache Manager Prefetch Hint

**U / Track B**

### Mechanism

Track B (compression via I/O scheduling). The Windows Cache Manager (NT kernel, `CcCopyRead` / `CcMdlRead` path) coalesces read requests that fall within the same 64 KB "coherent range" into a single read-ahead issue. GGUF tensor offsets are currently laid out in generation order (embedding, attention, MLP per layer), with tensor headers at file head and data blocks scattered. The insight: repack the GGUF so that the *per-token decode critical path* — at 70B NVMe-resident, only ONE layer's W_Q + W_K + W_V + W_O + W_gate + W_up + W_down is needed at a time — occupies a single contiguous 512 KB–2 MB NTFS extent per layer. Windows Cache Manager's `ReadAheadGranularity` defaults to 64 KB; if the hot decode path is in a single extent, the first read issues prefetch for the remainder of that extent "for free" with zero ML-layer changes. The substrate primitive: `NtReadFile` with FILE_FLAG_SEQUENTIAL_SCAN hint OR manual `PrefetchVirtualMemory` (Win32, available since Windows 8) called on the mmap range of the next layer's extent just after starting the current layer's matmul. No model changes. No quantization changes. The bytes-per-weight does not change; the latency-per-layer drops because NVMe queue fill begins earlier.

### Residency arithmetic

70B at Q4_K_M ≈ 40 GB on NVMe. Per-token decode at 70B: each layer ≈ 40 GB / 80 layers = ~500 MB (Llama-3-70B layer count). Sequential read throughput on this NVMe: ~3 GB/s spec. Latency per layer if sequential: 500 MB / 3 GB/s = 167 ms/layer, 80 layers = 13.3 s/token — obviously too slow. The win here is different: OVERLAP, not bandwidth. If layer L's matmul on CPU takes T_compute, and extent-layout ensures layer L+1's weights are in the prefetch queue before layer L completes, the effective latency per token is max(T_compute, T_io) rather than T_compute + T_io. At 70B with AVX2 fp32, T_compute per layer is roughly: 500 MB × (FLOPS/byte ratio) / (AVX2 throughput) — estimate ~50–80 ms / layer. If T_io = 167 ms and T_compute = 60 ms, naive is 227 ms/layer; with full overlap it collapses to 167 ms/layer = 1.5× speedup. At lower quantization (1-bit: 70B ≈ 8–10 GB, T_io = 10 GB / 80 / 3 GB/s = 42 ms/layer), T_io becomes smaller than T_compute and the overlap mechanism becomes the bottleneck solver. The mechanism doesn't reduce bpw; it reduces wall-clock latency by issuing NVMe I/O earlier via the substrate's own prefetch path.

### Novelty gloss vs kill list and published landscape

Closest kill list item: R1/S2 "OS-Paging-Aware Weight Layout" (killed: "doesn't reduce bpw, just engineers paging"). This proposal differs structurally: the mechanism here is not page-layout for the OS demand-pager but explicit `PrefetchVirtualMemory` calls into the mmap window, issuing the I/O at the application level with Windows-native async semantics. The OS proposal in R1 was killed for wrong reasons in hindsight — it genuinely can't reduce bpw, but the kill reason should have been "I/O overlap already handled by ik_llama.cpp's async layer loading." Apple LLM-in-a-Flash (arXiv:2312.11514) is the closest published method; it uses mmap with sequential access patterns but is on Apple NVMe with different queue-depth characteristics. The structural difference: this proposal uses `PrefetchVirtualMemory` (a Win32-specific async prefetch primitive) rather than mmap sequential-read semantics, allowing application-level scheduling of the prefetch relative to the compute timeline.

Precondition to verify: does ik_llama.cpp already issue overlapped I/O for 70B NVMe inference? If yes, the mechanism is pre-empted by the existing implementation. That is the go/no-go.

### Smallest experiment

**Testable claim**: on Qwen3-1.7B loaded from NVMe (not RAM-pinned), calling `PrefetchVirtualMemory` on layer L+1's mmap region during layer L's GEMV reduces per-token wall-clock latency by ≥15% vs the default mmap demand-paging baseline.

**Runtime**: measure with Process Monitor or ETW tracing + `llama-bench` on 1.7B NVMe-forced. ~2 hours.

**Go threshold**: ≥10% latency reduction on per-token decode time with NVMe active (not DRAM-resident baseline).

**NO-GO structural finding**: ik_llama.cpp already overlaps I/O, in which case the finding is "ik_llama.cpp's existing prefetch mechanism is correct and extent layout doesn't add marginal gain on Windows" — structural constraint on future I/O-overlap ideas.

### Primary risk

ik_llama.cpp already performs overlapped layer loading via its own async I/O path, making this a no-op. Mitigation: run `strace`-equivalent (ETW + xperf) to profile actual NVMe queue utilization during inference before implementing anything.

---

## U-2: Working-Set Trimmer Avoidance via VirtualLock for Hot-Tier KV + Attention Weights

**U / Track B**

### Mechanism

Track B (compression-adjacent: residency stability, not bpw reduction). Windows 11's Working-Set Trimmer (WSMT) — triggered when the OS is under memory pressure — silently evicts pages from process working sets using `MmTrimWorkingSet`. On an 8 GB system running a 7 GB model, WSMT fires during inference and causes demand-paging faults mid-decode that are invisible in profiling but catastrophically slow. The substrate primitive: `VirtualLock` (Win32) pins pages to physical RAM, exempting them from WSMT. The mechanism: given 7.28 GiB RAM and ~6 GB model DRAM-resident, LOCK the highest-bandwidth-critical matrices (attention W_Q, W_K, W_V based on CF11: W_Q r_99/d ≈ 0.63, actually the most computationally critical per-token objects) and allow WSMT to page-swap the less critical, pre-identifiable cold segments. Specifically: (a) quantize to target bpw; (b) DRAM-load all layers; (c) `VirtualLock` the attention weight tensors (W_Q, W_K, W_V, W_O across all layers) — approximately 4 × 4 × d_model × d_head × num_heads × layers bytes = for 70B: 4 × 4 × 8192 × 128 × 64 × 80 ≈ 8.6 GB — too large alone, but at 2-4 bit quantization this drops to ~0.5–1 GB; (d) allow MLP weights (much larger, already loaded) to remain unlocked. When WSMT fires, it will evict MLP pages not KV-path pages, so attention-side decode does not fault. The substrate primitive is `VirtualLock`; it does not require kernel modification; it is standard Win32.

### Residency arithmetic

7.28 GiB system RAM. Qwen3-1.7B at IQ4_XS: ~1.1 GB total. Attention weights (W_Q + W_K + W_V + W_O, 28 layers, d_model=2048, num_heads=16, d_head=128): 4 matrices × 28 × 2048 × 2048 × 2 bytes (bf16) = 4 × 28 × 8 MB = 896 MB bf16; at IQ4_XS (~0.5 bpw equivalent for QK weights) ≈ 224 MB. `VirtualLock` of 224 MB is within the default process lock quota (typically 64 MB but adjustable via `AdjustTokenPrivileges(SE_LOCK_MEMORY_NAME)`). For 70B at 1-2 bpw: attention weights ≈ 8.6 GB × (1 bpw / 16 bpw) = 0.54 GB — lockable. The remaining ~5 GB MLP weights remain under WSMT control; on demand-page fault they reload from NVMe at ~3 GB/s worst case. The gain is in INFERENCE STABILITY: attention-path latency has zero variance; MLP path has occasional fault cost. This is a jitter-reduction win, not a throughput win.

### Novelty gloss

No kill-list item covers this. LLM-in-a-Flash and Apple iGPU approaches address NVMe-resident weights; this proposal addresses the complementary problem of RAM-resident weights being evicted mid-inference by the OS memory manager. No published LLM inference paper known to me explicitly invokes `VirtualLock` for attention-tier pinning. Closest published analogy is database buffer pool management (Oracle pin buffers, PostgreSQL shared_buffers with `mlock`), but not in the LLM inference literature.

### Smallest experiment

**Testable claim**: on the Ryzen 5 7530U with 8 GB RAM, running Qwen3-1.7B inference under simulated memory pressure (a background process holding ~1 GB RAM), `VirtualLock`-pinned attention weights produce ≤5% variance in per-token latency vs baseline >30% variance.

**Runtime**: ~1 hour (llama-bench with and without background memory pressure + lock/unlock comparison).

**Go threshold**: latency coefficient of variation drops by ≥2× with VirtualLock active under pressure.

**NO-GO structural finding**: if WSMT doesn't fire during inference on this hardware (8 GB sufficient headroom), the mechanism is unnecessary — but the finding establishes that memory-pressure-induced eviction is not a factor on this tier, which rules out the entire class of OS-memory-pressure ideas.

### Primary risk

`AdjustTokenPrivileges(SE_LOCK_MEMORY_NAME)` requires SeLockeMemory privilege, which standard user accounts don't hold by default — requires admin rights or policy change. Mitigation: test under admin account first; if privilege gate is real, the mechanism is only usable by admins and the idea's practical deployment is narrower.

---

## U-3: GGUF ANS-Dict Pre-Warmed zstd Dictionary on Weight Tensors as NVMe Bandwidth Multiplier

**U / Track B**

### Mechanism

Track B (compression + I/O bandwidth). zstd's `ZSTD_DCtx` with a pre-trained dictionary can decompress at ~1 GB/s on the Ryzen 5 7530U. NVMe sequential read: ~3 GB/s. The key question: can quantized weight tensors (Q4, Q2) be zstd-compressed with a pre-trained dictionary at >3:1 ratio such that the decompression throughput bottleneck (1 GB/s) × (compression ratio) exceeds the raw NVMe read throughput (3 GB/s)? If ratio > 3.0×, the effective bandwidth becomes decompression-bound at 1 GB/s × ratio = >3 GB/s effective. The substrate primitive: `zstd --train` (dictionary training on weight quantile buckets) + `ZSTD_DCtx` streaming decompression. The non-obvious move: use a GLOBAL zstd dictionary trained on ALL quantized weight tensors in the GGUF file (not per-layer, not per-type). INT4 quantized weights have ~16 possible nibble values; the nibble pair distribution across all Q4_0 blocks is highly non-uniform and near-identical across the model. A global dictionary trained on a sample of all blocks should compress the rest efficiently. The kill condition from R1's "Arithmetic-Coded Weight Streams + Lazy Decode" (killed for ZipServ ASPLOS 2026 + AVX2 serial decode bottleneck): that idea used ANS/entropy coding which has serial decode constraints. zstd with dictionary uses a DIFFERENT primitive: LZ77 backreferences into the dictionary, which parallelizes better and doesn't require bit-serial decode. The actual bottleneck test is decompression throughput vs NVMe read throughput.

### Residency arithmetic

Qwen3-1.7B-Base Q4_0: ~1.1 GB on disk. zstd -19 with trained dictionary on INT4 nibble pairs (empirically similar to bzip2 on uniform noise: ~1.05–1.10× for truly uniform, but Q4 nibbles are bimodal): expected ratio 1.1–1.4×. At 1.4×: file = 786 MB, NVMe read time = 786 MB / 3 GB/s = 0.26 s, decompression time = 1.1 GB / 1 GB/s = 1.1 s. RATIO < 3, so decompression is bottleneck — NET LOSS for 1.7B. For a model with more structure (70B at Q4): the Q4 weight nibble distribution may be more bimodal (trained Qwen3 has specific outlier structure per CF3). Test on actual GGUF first. GO threshold: ratio ≥ 3.0× with decompression throughput ≥ 1.5 GB/s (possible with zstd multithreaded or dict tuned to nibble structure) or ratio ≥ 2.0× when combined with U-1's overlap (NVMe read of smaller file overlaps with decompression of prior chunk). The smallest experiment resolves this immediately.

### Novelty gloss

Closest kill-list item: R1/S2 "Arithmetic-Coded Weight Streams + Lazy Decode" (killed for ZipServ prior art and serial decode). Structural difference: zstd dictionary compression uses LZ77 backreferences, not arithmetic coding — the decode operation parallelizes across chunks using `ZSTD_decompressStream` with separate `ZSTD_DCtx` per thread. ZipServ stores entropy-coded weights requiring bit-serial decode at bottleneck; this stores zstd-with-dict blocks requiring byte-addressable LZ77 decode. R1's Stage-2-deferred Idea E (ANS/tANS 8-Way Interleaved) was estimated at ~12% bandwidth gain ceiling; that estimate used tANS throughput. zstd dict may have a different ceiling (test required). No published LLM paper uses zstd dictionary compression of quantized weights as a bandwidth multiplier in this specific way.

### Smallest experiment

**Testable claim**: `zstd --train` on a 10 MB sample of Q4_0 weight nibbles from Qwen3-1.7B, then `zstd -d` on the full Q4_0 tensor file, achieves compression ratio ≥ 2.0× AND `ZSTD_DCtx` decompression throughput ≥ 1.5 GB/s on the Ryzen 5 7530U.

**Runtime**: 30 minutes.

**Go threshold**: ratio ≥ 2.0 AND throughput ≥ 1.5 GB/s (implying effective bandwidth ≥ 3.0 GB/s equivalent NVMe, breaking even with raw NVMe).

**NO-GO structural finding**: Q4 nibble entropy is near-maximal (as expected for well-trained uniform quantization), zstd compression ratio is ≤1.1×. This confirms the R1/S2 kill was correct at the root: quantized weight bytes are near-incompressible by dictionary methods. Structural finding: weight nibbles post-quantization are effectively i.i.d. — no locality zstd can exploit.

### Primary risk

Q4_0 nibbles may be near-uniformly distributed after GGUF quantization (well-calibrated quantizer forces entropy ≈ log2(16) = 4 bits per nibble), making the compression ratio ≈ 1.0×. Mitigation: test first on raw bf16 weights vs Q4_0; if bf16 compresses but Q4 doesn't, the finding is that quantization itself removes the compressible structure (which is valuable to know).

---

## U-4: NTFS Sparse File Sentinel for Zero-Block Skip in NVMe-Resident Structured-Sparse Weights

**U / Track B**

### Mechanism

Track B (storage + I/O reduction). Windows NTFS supports sparse files (`DeviceIoControl(FSCTL_SET_SPARSE)` + `DeviceIoControl(FSCTL_SET_ZERO_DATA)`): ranges declared zero occupy NO disk space and read back as zeroes without issuing NVMe I/O. If a weight tensor can be represented with a sparse component whose zero-valued blocks are NTFS-sparse-declared, the NVMe I/O for those blocks is PHYSICALLY ABSENT — the kernel returns zero-fill from a page of zeros cached in the kernel pool, with no NVMe queue transaction. The substrate primitive: NTFS sparse files, a standard Win32 feature available on any NTFS volume. The connection to the inference problem: after post-training quantization to low bit-widths, are there zero-valued INT4 blocks? At Q2 or lower, yes — particularly for layers with near-dead neurons (CF6: Layer 1 has 36% gate neurons near-constant; their W_gate rows contributing to a Q2-quantized block that rounds to zero). The mechanism: (1) quantize to Q2 or Q1.5; (2) identify GGUF tensor blocks whose quantized value is identically zero after rounding; (3) FSCTL_SET_ZERO_DATA those byte ranges in the GGUF file; (4) ik_llama.cpp reads GGUF normally — zero-blocks get zero-fill from kernel, non-zero blocks read from NVMe. No model code changes needed.

### Residency arithmetic

At Q2 (≈2 bpw): 70B model ≈ 17.5 GB. If 5–15% of quantized blocks round to zero (plausible at Q2 with global zero-padding: 0-mean init + small calibration dist → near-zero blocks), sparse file removes 5–15% of I/O = 0.88–2.6 GB not read from NVMe. At 3 GB/s sequential NVMe: saves 0.3–0.9 s per full model scan. More significant: 1.7B at Q2 ≈ 0.43 GB; 5% sparse = 21 MB saved; at DRAM-resident speeds, negligible. The mechanism is most useful at 70B NVMe-resident Q2, where even 10% I/O reduction = ~0.6 s/token improvement.

CF6 motivation: Layer 1 has 36% gate neurons foldable (near-constant output). At Q2, those neurons' weight rows may quantize to zero. This is a testable hypothesis — if CF6's near-constant neurons correspond to near-zero W_gate rows, NTFS sparse marking those rows eliminates their I/O cost with no NVMe reads.

### Novelty gloss

No kill-list item covers this. R1/S4-deferred Idea I "NTFS-Reflink Codebook Deduplication" was killed for Windows 11 Home limitation (Home lacks ReFS reflinking). This proposal uses NTFS sparse files, which IS supported on Home. Structural difference: sparse files mark actual physical zero-blocks as absent from disk; reflinks share physical storage for duplicate data. Different mechanism, different primitive. No published LLM inference paper uses NTFS sparse-file semantics for weight tensor storage.

### Smallest experiment

**Testable claim**: at Q2 quantization of Qwen3-1.7B, ≥5% of quantized weight blocks (GGUF Q2_K blocks) are identically zero after rounding; marking these sparse reduces mmap-read NVMe I/O proportionally, measurable via ETW storage tracing or `iostat`-equivalent.

**Runtime**: 1 hour (quantize to Q2 with llama.cpp tools, scan blocks for zero count, measure actual NVMe reads with and without FSCTL).

**Go threshold**: ≥5% zero blocks AND measurable I/O reduction (physical reads drop by ≥5% per ETW trace).

**NO-GO structural finding**: well-calibrated Q2 quantizer normalizes blocks so zero-blocks are rare (<1%). Structural finding: quantization normalization prevents systematic zero blocks — sparse-file approach requires pre-quantization structural sparsity, which CF8 says trained Qwen3 weights don't have.

### Primary risk

NTFS sparse-file zero-ranges only help if ik_llama.cpp reads the GGUF via memory-mapped I/O AND the OS honors the sparse-zero path without issuing actual I/O. If ik_llama.cpp opens with `CreateFile` and `ReadFile` (non-mmap), the kernel still issues a read, returns zeros — but from the kernel zero-page, not NVMe. Need to verify ik_llama.cpp's I/O path. Mitigation: check source before investing; if mmap is used, the sparse path is zero-cost.

---

## U-5: Page-Table Walk Elimination via HugePage Promotion for Weight Tensor DRAM-Resident Inference

**U / Track B**

### Mechanism

Track B (latency/bandwidth). Windows 10/11 supports Large Page Memory (2 MB pages, `VirtualAlloc` with `MEM_LARGE_PAGES` flag, requires `SeLockMemoryPrivilege`). Standard page table walks for a 4 KB page require 4 levels (PML4→PDPT→PD→PT) per TLB miss. At 2 MB large pages, the PT level is eliminated: 3-level walk on miss, and TLB coverage per entry is 512× larger (one entry covers 2 MB vs 4 KB). For DRAM-resident inference: W_Q for 1.7B is 2048×2048×2 bytes = 8 MB ≈ 4 large pages; W_up is 2048×6144×2 = 24 MB ≈ 12 large pages. If the TLB covers the working set in large-page entries, per-element access has zero walk overhead vs up to 4 DRAM-latency walks per page fault on a cold 4 KB walk. The substrate primitive: `MEM_LARGE_PAGES` on Windows 11, or `madvise(MADV_HUGEPAGE)` on Linux (transparent huge pages). The critical parameter: Ryzen 5 7530U TLB sizes. Zen 3 L1-DTLB: 64 entries (4 KB pages) and 32 entries (2 MB pages). L2-TLB (stlb): 2048 entries (4 KB) and 1024 entries (2 MB). At 4 KB pages: 2048-entry stlb covers 2048 × 4 KB = 8 MB; Qwen3-1.7B weights are ~1 GB → at most 8/1000 = 0.8% of working set covered → high TLB-miss rate. At 2 MB pages: 1024-entry stlb covers 1024 × 2 MB = 2 GB → all of Qwen3-1.7B weight tensors fit in stlb. This is not a custom kernel; it is a standard API invoked with the right privilege.

### Residency arithmetic

TLB miss cost on Zen 3: page-walk time for a 4-level walk hits L2 cache for each level at ~5 ns each × 4 = 20 ns per miss. At 4 KB pages and 1 GB weight tensor: GEMV over 1 GB of weights with stride-1 access issues ~262K 4 KB page accesses. If cache-miss rate is low (weights already in L3 / DRAM, sequential access → hardware prefetcher fires), TLB misses are the bottleneck. With 2 MB pages: same 1 GB → 512 large pages → all fit in stlb (1024 entries). Zero stlb misses during sequential GEMV. Speedup on sequential GEMV: conservatively 5–15% for TLB-miss-dominated workloads (measured in database / in-memory analytics benchmarks on similar hardware). For per-token decode at 6 GB/s DRAM bandwidth (Ryzen DDR4 single-channel): 1 GB weight scan ≈ 166 ms. TLB miss overhead at ~5% = 8 ms/token saved if all misses eliminated. At 1.7B: ~0.8 tok/s → 8 ms per token ≈ 1–2% improvement. At 70B with 40 GB weights and much larger TLB miss contribution (DRAM bandwidth already saturated), the gain fraction is similar but may stack with other improvements.

### Novelty gloss

No kill-list item covers this. No published LLM inference paper explicitly reports large-page (MEM_LARGE_PAGES / MADV_HUGEPAGE) allocation as a primary optimization for weight tensor access in CPU-only inference. The closest adjacent work is database in-memory analytics (SAP HANA, PostgreSQL huge_pages = on), but not LLM. ik_llama.cpp does not allocate with `MEM_LARGE_PAGES` by default; its `ggml_malloc` uses standard `malloc` or `mmap` without large-page promotion. The mechanism is purely substrate-level.

### Smallest experiment

**Testable claim**: allocating Qwen3-1.7B weight tensors with `VirtualAlloc(MEM_LARGE_PAGES)` (or equivalently via hugepages-backed mmap on Linux) reduces per-token decode wall-clock time by ≥3% vs baseline `malloc` allocation on the Ryzen 5 7530U.

**Runtime**: 2 hours (patch ggml_malloc to use large-page allocation, run llama-bench comparison, requires admin for SeLockMemoryPrivilege).

**Go threshold**: ≥3% latency reduction on per-token decode (DRAM-resident, not NVMe).

**NO-GO structural finding**: hardware prefetcher eliminates TLB-miss cost by keeping translations warm; large pages confer no benefit. Finding: sequential GEMV on Zen 3 is purely bandwidth-bound, not TLB-bound, at 1 GB working set. This rules out the TLB-miss hypothesis entirely.

### Primary risk

`SeLockMemoryPrivilege` required for `MEM_LARGE_PAGES`; not held by standard user accounts. On Linux, transparent huge pages (THP) are enabled by default and ik_llama.cpp may already benefit. Mitigation: test on Linux first (THP = always, no privilege needed); if Linux shows no gain, Windows large-page test is moot.

---

## U-6: Named Pipe Zero-Copy Between llama.cpp Worker Threads via Loopback IOCP as a Latency Smoother [FREE SWING]

**U / Track A**

### Mechanism

Track A (computation graph routing). Windows IO Completion Ports (IOCP) over loopback named pipes provide async zero-copy message passing between threads at ~microsecond latency with kernel-backed flow control. Standard ik_llama.cpp thread dispatch uses a condition-variable/mutex pattern per-layer; this is fine but introduces:
(a) thundering-herd wakeup on layer completion,
(b) no backpressure mechanism if one layer's decode stalls.

The substrate primitive: `CreateNamedPipe` + `ConnectNamedPipe` + `WriteFile`/`ReadFile` with overlapped I/O on an IOCP. The mechanism: model each layer's output (the residual stream tensor) as a "message" pushed through a named pipe to the next layer's worker thread. IOCP delivers the completion on the consumer thread with zero-copy semantics (the pipe buffer IS the residual stream tensor's memory, with the producer and consumer sharing the mapped view). The non-obvious inference payoff: IOCP naturally implements a bounded-buffer with OS-backed flow control. If later layers are faster than early layers (possible with the CF11 finding that attention layers are more compressible), the IOCP backpressure prevents a long-context prefill from flooding the decode pipeline. More importantly, IOCP allows speculative `GetQueuedCompletionStatus` with a timeout — a worker thread can sleep waiting for the next layer's input, and the OS schedules it exactly when data arrives. No spin-wait. This is a latency-smoothing / scheduling primitive, not a throughput multiplier.

This is a [FREE SWING] because it has no connection to CF1–CF12 and the payoff is scheduling quality, not tok/s. Its structural argument is that ik_llama.cpp's threading model leaves OS scheduling efficiency on the table, and IOCP is the right substrate primitive to recover it.

### Residency arithmetic

Payoff is not in residency or bpw — it is in decode latency variance. The claim: per-token decode latency coefficient of variation drops by ≥20% under IOCP-mediated layer dispatch vs mutex/condvar dispatch. This is relevant for interactive use (P99 latency matters as much as median). No residency change.

### Novelty gloss

No kill-list or published LLM paper uses IOCP for layer dispatch in CPU-only inference. Closest analog: producer-consumer threading in database engines (PostgreSQL parallel query uses IOCP-analog). The structural difference from all existing LLM threading work: IOCP gives the OS the scheduling contract, not the user-space condition variable. The OS honors QOS priorities and CPU affinity during IOCP dispatch — on a Ryzen mobile with power gating, this matters.

### Smallest experiment

**Testable claim**: replacing ggml_threadpool's condition-variable layer dispatch with a named-pipe IOCP loop reduces per-token latency P99 by ≥20% on Qwen3-1.7B with 4 threads (T=4) on the Ryzen 5 7530U.

**Runtime**: 1 day (implementation is ~200 lines wrapping the IOCP dispatch around ggml_graph_compute).

**Go threshold**: P99 latency drops ≥20%; median unchanged.

**NO-GO structural finding**: IOCP overhead (kernel transitions per layer dispatch) exceeds the spin-wait cost on short layers. Structural finding: per-layer compute time on 1.7B is too short (<<100 µs) for OS-level scheduling overhead to matter; only at 70B NVMe-stall timescales does OS scheduling quality matter.

### Primary risk

Windows loopback named-pipe overhead (~2–5 µs per message) may dominate per-layer dispatch for small models (1.7B has layers computing in <100 µs at 4 KB context). Mitigation: measure at 70B on NVMe where per-layer time is >100 ms — overhead is negligible there.

---

## U-7: ReadDirectoryChangesW Trigger for Weight Hot-Swap Without Process Restart

**U / Track B**

### Mechanism

Track B (operational / deployment). Windows `ReadDirectoryChangesW` (kernel filesystem watcher API) delivers completion notifications when any file in a watched directory changes. The substrate move: watch the GGUF weight file directory; when a new GGUF is atomically swapped in (rename over old file — NTFS atomic rename guarantees), ik_llama.cpp's main loop receives the notification and re-mmaps the new tensor data on the next inference call without process restart. This is relevant for the pipeline use case: round-trip between quantization pass (Stage 5 of this ladder) and inference testing. Currently each experiment requires stopping and restarting llama-server with a new model file. With `ReadDirectoryChangesW` + mmap re-bind, the hot-swap is <100 ms (mmap re-open, no heap realloc for same-size model). The substrate primitive: `ReadDirectoryChangesW` with `FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_LAST_WRITE`, standard Win32, no kernel modification.

This is less about improving tok/s and more about reducing the experiment loop latency for this exact ladder pipeline. The audacity is "the substrate already has the primitive; the ML experiment loop has not noticed."

### Residency arithmetic

Payoff: experiment iteration time reduction. Current round-trip (stop server → copy new GGUF → restart → warm cache): ~30–60 s per experiment iteration. With hot-swap: <1 s. For a 200-prompt calibration run needing 5 re-iterations, this saves 200–300 s per round. Over 50 runs of this ladder, that is a 2–3 hour wall-clock saving.

### Novelty gloss

No kill-list item touches this. Not a published LLM inference technique. Closest analogy: Webpack's hot-module-replacement (HMR) for web development, using OS filesystem notifications to reload changed modules without restart. The structural difference from all published work: this targets the research iteration loop, not end-user serving — and it uses NTFS atomic rename + mmap semantics to achieve zero-gap swap.

### Smallest experiment

**Testable claim**: `ReadDirectoryChangesW` delivers notification within 50 ms of an `MoveFileEx` rename of a GGUF file, and ik_llama.cpp can re-mmap the file and resume inference within 200 ms, without process restart, on the Ryzen 5 7530U.

**Runtime**: 2 hours (implement the watcher loop + mmap-rebind in ik_llama.cpp, test with 1.7B GGUF rename).

**Go threshold**: end-to-end swap latency ≤200 ms with no inference quality degradation.

**NO-GO structural finding**: ggml's internal tensor metadata is heap-allocated with internal pointers, making mmap re-bind require a full model reload anyway. Finding: ggml's memory model is not designed for live mmap rebinding.

### Primary risk

ggml stores tensor shape/stride metadata in heap-allocated `ggml_tensor` structs that reference the mmap base pointer internally. Renaming the mmap file without updating all struct pointers causes UAF. Mitigation: implement at the ik_llama.cpp server level (re-issue `llama_load_model_from_file` on the same ggml_context, replacing internal references) — this IS a process-level reload, just without TCP connection teardown.

---

## Convergence handles

- **NVMe I/O overlap with compute** — U-1 and potentially others; convergence with any Reach idea that cites overlapped layer streaming as a cascade rung
- **Windows privilege gate (SeLockMemoryPrivilege / SE_LOCK_MEMORY_NAME)** — U-2 and U-5 both gated on same privilege; any other idea invoking VirtualLock or MEM_LARGE_PAGES shares this precondition
- **TLB working-set residency at weight-tensor scale** — U-5; convergence with any F or C idea quantifying attention vs MLP access-pattern divergence
- **NTFS file semantics (sparse, rename atomicity, directory watcher)** — U-4 and U-7; convergence with any U or A idea that touches on-disk GGUF layout
- **ik_llama.cpp I/O path (mmap vs ReadFile vs async)** — shared precondition across U-1, U-4, U-7; a single measurement resolves the precondition for all three
- **CF6 Layer-1 gate near-constancy as structural sparsity source** — U-4 references CF6's 36% foldable neurons as the only confirmed hook from SUMMARY.md; any idea using CF6 shares this convergence handle
