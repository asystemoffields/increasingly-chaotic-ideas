# Stage 1 — Unconventional Substrate — Run 004

Orientation U. 6 ideas. Not CF8 MLP rank reduction in any idea. KILL_LIST v2-CHEAP-TEST-001 (cross-layer W_Q subspace sharing) noted — no idea here touches that class.

---

## U1 — Windows SuperFetch Footprint Inversion (WSFI) — U/B

### Mechanism
**Track B.** Windows 11's SuperFetch / SysMain service (Host Process for Windows Services, svchost -k NetworkService) maintains a working-set prefetcher that learns per-process access histograms and preloads pages before a process needs them. The standard LLM inference process treats this as noise — random access patterns confuse the prefetcher, suppressing it. Inversion: if the GGUF weight file is laid out such that inference always scans it in a predictable, process-lifetime-consistent order (a fixed rotation schedule per layer-index), SuperFetch will learn the schedule within 2–3 cold runs and begin preloading the next stratum of weight pages while the CPU works on the current stratum. No API call required — only layout discipline. The substrate primitive is SysMain's CLR/PC-based prefetcher (documented in Windows Internals 7e, Ch.10 "Prefetcher"), which operates on page-level access traces stored in C:\Windows\Prefetch\*.pf. For an 8B Qwen3 GGUF at IQ4_XS (~4.7 GB), the access pattern over 28 layers is 28 sequential slabs of ~168 MB each; if the GGUF offset table is reordered to match this sequential slab order, the PF trace converges to a strided linear scan — the pattern SysMain's prefetcher handles best. Does not rely on CF13* NVMe prefetch numbers; relies on NTFS page cache and SuperFetch learning, both measurable on this stack independently.

### Residency arithmetic
- 8B at IQ4_XS: ~4.7 GB on NVMe; fits in 7.28 GiB DRAM on a cold layout if fully pinned (but we are NOT pinning — we are layer-streaming).
- Current random-access NVMe throughput for GGUF weight fetch: ~700 MB/s (measured behavior, NVMe random 4K baseline; CF13* is the prefetch-engaged variant we do NOT cite as load-bearing).
- If SuperFetch achieves prefetch overlap on ≥50% of layer-fetch intervals, the effective I/O latency per layer drops from serial (~0.24 s per 168 MB at 700 MB/s) toward overlap with CPU matmul (~0.09 s at 11.5 GB/s DRAM bandwidth for the in-cache portion).
- Conservative estimate: 1.3–1.8× throughput improvement for NVMe-resident 8B, from ~2.5 tok/s to ~3.5–4.5 tok/s. This is not residency reduction — it is latency hiding via learned prefetch.
- Quality cost: 0 nats (layout permutation changes no bytes, only offsets).

### Novelty gloss
Closest KILL_LIST item: R1/S2 "OS-Paging-Aware Weight Layout" (killed as "doesn't reduce bpw, just engineers paging"). Structural difference: that proposal was passive layout-for-DRAM; this proposal actively exploits SuperFetch's *learning* loop — a dynamic system that improves over successive inference sessions, not a one-shot layout trick. Closest published method: Apple LLM-in-a-flash uses mmap-based demand paging but does not exploit OS-level prefetch learning. SuperFetch is a Windows-specific primitive not invoked by any published LLM inference system identified.

### Smallest experiment
Claim: On Qwen3-1.7B-GGUF at IQ4_XS with sequential-slab GGUF layout, SuperFetch shows measurable reduction in NVMe read latency after ≥3 inference runs vs a randomly-addressed GGUF. Test: (1) Lay out a GGUF with llama-gguf-split-style sequential slab order. (2) Run 10 identical inference prompts on the standard GGUF; record per-layer NVMe read time (etw trace or procmon). (3) Repeat on slab-ordered GGUF for 10 runs. (4) Compare per-layer NVMe latency between run 3 and run 10 on slab layout. Runtime: ~2 hrs. Go threshold: run-10 layer-NVMe latency ≤70% of run-3 on slab layout (SuperFetch learned). No-go finding: SuperFetch does not engage on GGUF access patterns → substrate primitive does not fire on this workload → documents a real stack behavior for all future NVMe-path proposals.

### Primary risk
SuperFetch may be disabled or throttled on this workstation (registry: `HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PrefetchParameters\EnableSuperfetch`). Mitigation: Check registry value first (1 minute); if 0, the experiment is moot; try `ReadyBoost` as fallback or pivot to `FILE_FLAG_SEQUENTIAL_SCAN` CreateFile flag (a weaker but deterministic prefetch hint requiring 1 ioctl call, no OS learning loop needed).

---

## U2 — NTFS Compressed Clusters as Zero-Copy Decode Cache (NCZC) — U/B

### Mechanism
**Track B.** NTFS supports transparent cluster-level LZ77 compression (enabled per-file via `compact.exe /c`). On Ryzen 5 7530U with NVMe, reading a compressed NTFS file triggers in-kernel LZ77 decompression; the decompressed bytes land in the NTFS Cache Manager's working set. For a Q4_0 GGUF (low entropy per 4-bit nibble run, but weight tensors have ~3.5 bits/symbol after nibble packing), NTFS LZ77 achieves roughly 1.05–1.10× size reduction — not enough to justify compression alone. However: if the GGUF is reordered so that weight rows with similar magnitude profiles are stored adjacently, the local run-length / LZ77 match distance drops, and NTFS LZ77 compression ratio improves toward 1.2–1.3×. The substrate primitive is `NtfsCompressionUnit` (4 KB clusters grouped into 64 KB compression units, documented in NTFS internals). The payoff is dual: (a) reduced NVMe read bytes (1.2–1.3× after layout optimization), and (b) zero decompression cost (kernel transparent, happens during Cache Manager fill, not on the inference thread's CPU time — the decompression runs on a separate pool thread). This is not a custom kernel; it is using NTFS compression as designed, applied to a GGUF with a layout that makes it compress better.

### Residency arithmetic
- 8B at Q4_K_M: ~4.7 GB. With NTFS compression at 1.25×: ~3.76 GB on disk (NVMe read bytes reduced).
- NVMe reads at 700 MB/s random: current ~6.7 s cold load of 4.7 GB → 5.4 s for 3.76 GB (1.25× speedup on I/O).
- Decompression cost (kernel LZ77 at ~1.5 GB/s single-thread) runs parallel on pool thread; effective decompression rate ~1.5 GB/s. For 3.76 GB: ~2.5 s decompression time. I/O dominates (5.4 s > 2.5 s) so decompression does not add to wall time — it is hidden in I/O.
- Net: ~1.25× throughput improvement on NVMe-bound cold paths.
- Quality cost: 0 nats (no weight modification, only storage format).

### Novelty gloss
Closest KILL_LIST item: R2/S2 "Arithmetic-Coded Weight Streams + Lazy Decode" (killed for AVX2 serial decode bottleneck). Structural difference: NCZC uses kernel transparent NTFS LZ77 (no inference-thread decode cost) and targets the NVMe-read-bytes bottleneck, not the compute bottleneck. Closest published method: GPTQ / LLM.int8() store weights in standard formats; no published work uses NTFS cluster compression with layout optimization for LLM inference. The nearest substrate-level work is llama.cpp GGUF layout research (layer order, tensor packing), but that work targets MMAP sequential access, not NTFS compression ratio.

### Smallest experiment
Claim: Row-sorted Q4_K_M GGUF achieves ≥1.15× NTFS compression ratio over default GGUF layout on Qwen3-1.7B. Test: (1) Sort W_up rows by L2-norm buckets (10 magnitude tiers). (2) Write two GGUFs (standard and sorted). (3) Apply `compact /c` to both. (4) Measure compressed sizes. Runtime: ~30 min. Go threshold: sorted-layout compressed size ≤ 87% of standard-layout compressed size. No-go finding: NTFS LZ77 achieves no benefit from weight-row sorting → weight tensors are effectively random at the 64 KB compression-unit granularity → documents the entropy floor of Q4_K_M weights for future codec proposals.

### Primary risk
NTFS transparent compression has a known penalty for random-access reads: the Cache Manager must decompress the entire 64 KB compression unit to serve a 4 KB read. If llama.cpp uses random-access MMAP patterns, the decompression overhead could exceed the NVMe savings. Mitigation: test with sequential-scan inference only (disable mmap, use `--no-mmap` flag in llama.cpp); sequential access decompresses compression units serially, matching NTFS's intended usage.

---

## U3 — Windows Working Set Trimmer as Eviction Oracle (WWTE) — U/B

### Mechanism
**Track B.** Windows kernel's Working Set Manager (wsmgr, runs as PASSIVE_LEVEL kernel thread, documented in Windows Internals 7e Ch.10 "Working Set Management") evicts memory-mapped pages from process working sets under memory pressure using a modified CLOCK algorithm. For a GGUF loaded via mmap, wsmgr evicts pages it determines to be "standby" (not recently accessed). This eviction decision is information: the pages wsmgr retained in the working set after `MmWorkingSetTrim` cycles are exactly the frequently-accessed weight pages. Proposal: after a warm-up inference pass (e.g., 10 prompts), query which GGUF pages remain resident (via `VirtualQuery` + `VirtualLock` testing, or via `QueryWorkingSet` API — both are standard Win32 documented APIs) and use that residency map as a calibration-derived static pin-map. Pages identified as persistently-resident across all 10 prompts are "hot weights" — candidate for explicit `VirtualLock` (locks pages into physical RAM, prevents future trim). The inference loop then only waits for NVMe fetches on the "cold" pages. The oracle is the OS LRU state, which encodes the access history the OS measured, not the model's internal statistics. No ML training required; no CF-derived knowledge required. The substrate primitive: `QueryWorkingSetEx` (documented in MSDN, available since Windows Vista) returns a per-page `PSAPI_WORKING_SET_EX_INFORMATION` struct with age and share-count fields.

### Residency arithmetic
- 8B at IQ4_XS: ~4.7 GB. If hot pages (persistently resident) are 20% of weights = 940 MB → these are VirtualLock'd in 7.28 GiB DRAM (~940 MB pinned).
- Cold pages (3.76 GB) are fetched from NVMe as needed at 700 MB/s.
- Before WWTE: every layer fetch hits NVMe (~0.24 s per 168 MB layer). After WWTE: 20% of accesses hit DRAM (~0.015 s per pinned 168 MB chunk at 11.5 GB/s). Net: ~0.19 s per layer average → ~1.25× throughput improvement.
- If hot pages are 40%: 1.88 GB pinned, 2.82 GB cold → ~1.5× throughput improvement. This approaches usable tok/s on 8B.
- Quality cost: 0 nats (no weight modification).

### Novelty gloss
Closest KILL_LIST item: R1/SELECTED "Shadow-Model NVMe Prefetch Oracle" (killed because SwiGLU density caps skip rate and NanoQuant binary storage hostile to selective access). Structural difference: WWTE does not predict which weights to skip — it measures which weights the OS already chose to retain, using the OS's own LRU state as the oracle. No ML predictor required. No skip logic required. The oracle is free (OS computed it already); the action is VirtualLock (one syscall). Closest published method: Apple LLM-in-a-flash's "prefetch clustering" identifies frequently-used weight pages via activation statistics. WWTE uses the OS working-set age instead of activation statistics — a different, cheaper, substrate-level signal.

### Smallest experiment
Claim: After 10 warm-up inference passes on Qwen3-1.7B GGUF (mmap), `QueryWorkingSetEx` identifies a consistent subset (≥15% of GGUF pages) that remains resident across all 10 passes. Test: (1) Run 10 identical prompts on Qwen3-1.7B mmap'd GGUF. (2) After each pass, call `QueryWorkingSetEx` on the GGUF mapping and record per-page residency. (3) Compute intersection of resident pages across all 10 passes. Runtime: ~1 hr. Go threshold: ≥15% of GGUF pages persistently resident. No-go finding: OS working-set manager evicts all GGUF pages between passes → no stable hot-page set exists → the OS sees the GGUF as fully cold → documents mmap behavior of llama.cpp under Windows 11 for future proposals.

### Primary risk
`VirtualLock` has a per-process limit (default: minimum working set size, typically 192 KB on consumer Windows — configurable via `SetProcessWorkingSetSizeEx` but requires SE_INCREASE_WORKING_SET_NAME privilege). If the hot-page set is large (e.g., 1 GB), locking it may require privilege elevation. Mitigation: First measure the hot-page set size (go/no-go step); if >192 KB, check if llama.cpp already calls `SetProcessWorkingSetSizeEx` (it does for `--mlock` flag) — reuse that codepath.

---

## U4 — GGUF Tensor Offset B-Tree as a Page-Fault Sequencer (GTBP) — U/B

### Mechanism
**Track B.** GGUF's tensor offset table (the metadata section at file head, listing byte offsets + sizes for every tensor) is read once at load time and then effectively abandoned — llama.cpp resolves tensor pointers at model load, then uses mmap'd virtual addresses directly. Proposal: read the offset table to construct a B-tree keyed by (layer_index, tensor_type, row_index) with leaf values = (file_offset, length). Feed this B-tree to a dedicated background thread that issues sequential `ReadFileEx` async I/O requests 2–3 layers ahead of the inference thread's current position (using `OVERLAPPED` + completion-port, standard Win32 async I/O, no custom driver). The inference thread signals the prefetch thread via a lightweight `SetEvent` each time it begins a new layer. The prefetch thread's B-tree walk is deterministic and aligned to NTFS cluster boundaries (4 KB), matching the NTFS Cache Manager's internal fetch unit. The substrate primitive: Win32 `ReadFileEx` with `OVERLAPPED` and I/O completion ports (IOCP), the standard Windows high-performance async I/O path documented in MSDN since Windows NT 3.5. This is not a new kernel primitive — it is a client-side access-pattern scheduler that feeds the existing NTFS async pipeline in the pattern it was designed for (large, sequential, aligned reads). Does not rely on CF13* NVMe prefetch numbers; the underlying NTFS IOCP path is independently verifiable.

### Residency arithmetic
- 8B at IQ4_XS, 28 layers, ~168 MB per layer: serial NVMe fetch at 700 MB/s = 0.24 s/layer.
- With 2-layer lookahead: while inference thread processes layer L (CPU at ~11.5 GB/s on DRAM), prefetch thread issues IOCP reads for layers L+1 and L+2 asynchronously. If prefetch completes before inference finishes L, effective NVMe wait = 0.
- CPU time per layer (8B, matmul on Ryzen 5 7530U): ~0.20–0.30 s (at ~11.5 GB/s DRAM for 168 MB weights). NVMe read time: 0.24 s. These are comparable → overlap eliminates ~80% of NVMe wait.
- Conservative estimate: 1.5–1.8× throughput improvement from 2-layer IOCP lookahead.
- Quality cost: 0 nats.

### Novelty gloss
Closest KILL_LIST item: R3/S2 "Branch-Predictor Token Trie" (killed: pre-empted by N-Gram Trie Speculative Decoding + bandwidth contention on misprediction). Structural difference: GTBP does NOT predict token sequences; it prefetches weight pages deterministically using the GGUF offset table as a static schedule. No prediction, no misprediction penalty. Closest published method: llama.cpp's `--mmap` mode uses demand paging (passive); llama.cpp `--no-mmap` reads weights synchronously (serial). IOCP lookahead is not implemented in mainline llama.cpp or ik_llama.cpp — confirmed by absence in source (no `OVERLAPPED` calls in llama.cpp weight loading path as of 2026-05).

### Smallest experiment
Claim: Win32 IOCP async reads of GGUF tensors 2 layers ahead of inference achieve ≥1.3× layer-fetch latency reduction vs synchronous reads on Qwen3-1.7B on this Ryzen 5 7530U NVMe path. Test: (1) Write a 50-line C measurement harness: read Qwen3-1.7B GGUF layer by layer, alternating synchronous ReadFile vs 2-layer IOCP lookahead. (2) Time per-layer delivery. Runtime: ~2 hrs (write + run). Go threshold: IOCP mean layer-delivery latency ≤ 70% of synchronous. No-go finding: NVMe controller serializes requests regardless of IOCP concurrency → Windows storage stack does not parallelize within single-SSD queue → documents actual NVMe QD behavior on this hardware.

### Primary risk
ik_llama.cpp uses mmap for weight access; inserting IOCP requires bypassing mmap and using explicit ReadFile calls, which conflicts with the current weight-pointer model. Mitigation: Test with a standalone harness that does not modify ik_llama.cpp first (go/no-go is the latency measurement, not the integration). Integration only if go.

---

## U5 — Denormal Float Sentinel as Quantization Tier Marker (DFSQ) — U/B

### Mechanism
**Track B.** IEEE 754 denormal (subnormal) floats have hardware-detected bit patterns (exponent field = 0x00, mantissa ≠ 0). On x86-64 with DAZ (Denormals Are Zero) mode enabled — the default in \_MM\_SET\_FLUSH\_ZERO\_MODE + \_MM\_SET\_DENORMALS\_ZERO\_MODE, both present in every SIMD-enabled C runtime — denormals are silently flushed to zero in SSE/AVX FPUs. Proposal: in a mixed-precision inference system (e.g., 3-tier: FP16 hot / INT8 warm / INT4 cold), encode the tier tag for each weight block not in a separate metadata array but directly as sentinel values in the existing weight file. Specifically: the "cold" (INT4-resident) blocks are encoded with a denormal-exponent marker in the block header float (the per-block scale factor, which is always FP16 in GGUF). A denormal scale value (e.g., 5e-45f) signals "this block is INT4, decompress before use" and is never a valid scale in practice (model weights have scales >> 1e-38). The inference runtime detects tier by testing `if (scale < 1e-40f)` — one comparison, no metadata array, no separate tier-lookup table. The substrate primitive: IEEE 754 denormal encoding (hardware-guaranteed bit pattern) + x86 FPU DAZ flag behavior (hardware-guaranteed zero-flush). This eliminates the per-block metadata overhead of a tier-map (typically 1 byte/block = 96 KB for 8B model at Q4_K_M block granularity) and replaces it with in-band signaling at zero storage overhead.

### Residency arithmetic
- 8B at Q4_K_M: ~4.7 GB. The tier map for a 3-tier system (FP16/INT8/INT4 per block) needs 2 bits/block = 48 KB overhead → negligible in isolation.
- DFSQ's direct residency savings: ~48 KB tier map eliminated. Negligible.
- The real payoff is correctness + latency: eliminating a separate tier-lookup pointer-chase (saves ~1 cache miss per block access = ~80 ns × 96K blocks = ~7.7 ms/pass). Small but nonzero.
- Indirect payoff: enables the 3-tier codebook scheme (RAOK / PDAP-style) to operate at finer block granularity without scaling the metadata cost — the sentinel encoding scales to 1-block granularity free.

### Novelty gloss
Closest KILL_LIST item: none directly — DFSQ is a storage-encoding mechanism, not a compression algorithm. Closest published method: GGUF uses separate metadata for per-block quantization type (the `ggml_type` enum per tensor). DFSQ proposes in-band signaling using the IEEE 754 denormal encoding of the scale factor, which no published quantization format uses. The nearest concept is NaN-boxing in dynamic languages (using NaN bit patterns as type tags) — applied here to denormals for quantization-tier tagging in weight files.

### Smallest experiment
Claim: A denormal-scale sentinel (5e-45f) is unambiguously distinguishable from all valid GGUF Q4_K_M block scales, and the `if (scale < 1e-40f)` branch predicts correctly on 100% of blocks in Qwen3-1.7B GGUF. Test: (1) Read all block scale values from Qwen3-1.7B Q4_K_M GGUF. (2) Compute min/max/histogram of scale values. (3) Verify no valid scale falls below 1e-40f. Runtime: ~15 min. Go threshold: min valid scale > 1e-38f (leaves 18-order gap for sentinel encoding). No-go finding: some valid GGUF scale falls below 1e-38f → in-band denormal sentinel is not safe → documents the valid-scale dynamic range of Q4_K_M for all future mixed-precision tier-encoding proposals.

### Primary risk
GGUF may store scales in a different precision or with a different bit representation (e.g., E4M3 or block-float) in future GGUF versions. Mitigation: DFSQ is a storage-format patch, not an algorithm; it can version-pin to the GGUF v3 scale encoding, with a version check at load time.

---

## U6 — NVMe Namespace Partitioning as Weight Tier Allocator (NNPA) — U/B [FREE SWING]

### Mechanism
**Track B.** NVMe SSDs expose multiple namespaces (NVMe 1.2+ spec, `Identify Namespace` command, NVMe multiple-namespace feature). On consumer SSDs, a single namespace is standard; but Windows 11 supports multiple namespace creation via `StorNVMe.sys` and the `IOCTL_STORAGE_MANAGE_DATA_SET_ATTRIBUTES` interface. NVMe controller internals assign physical flash dies to namespaces, with each namespace having independent garbage collection (GC) cycles, write buffers, and read queues. Proposal: partition the SSD into 2 namespaces — NS1 (small, e.g., 10 GB) for hot weight layers assigned to the OS partition, NS2 (remainder) for cold weight layers. The OS stores Qwen3 8B's "hot-layer" GGUF slab (layers 0–13, the first half of the network) in NS1 and the "cold-layer" slab (layers 14–27) in NS2. With independent GC per namespace, NS1 (hot, frequently read, rarely written) avoids GC stalls triggered by NS2's write activity (llama.cpp KV cache writes and prompt log writes). The GC stall on a single-namespace SSD can spike read latency by 50–200 ms; namespace separation eliminates this cross-contamination. The substrate primitive: NVMe namespace management, available on most consumer NVMe SSDs ≥2021 vintage (Samsung 980, WD SN850, etc.), configurable via Windows StorNVMe driver without firmware modification.

### Residency arithmetic
- GC stall magnitude on consumer NVMe: 50–200 ms per event (documented in SSD endurance literature; Samsung AN1527 white paper).
- Inference at 2.5 tok/s for 8B: ~400 ms/token. GC stalls (50–200 ms) appear as 12–50% latency spikes on affected tokens.
- With NS1/NS2 partition: NS1 (hot layers) is read-only after model load → GC frequency approaches zero → stall probability → 0 on hot-layer reads.
- Effect: eliminates GC-induced latency spikes on ~50% of weight accesses. Throughput improvement is stochastic but measurable as reduced p95/p99 latency rather than mean throughput.
- Quality cost: 0 nats.

### Novelty gloss
No KILL_LIST item is close: all prior NVMe proposals targeted bandwidth or prefetch, not GC interference. Closest published method: database literature uses NVMe namespace partitioning for log-structured storage (RocksDB NVMe tiering), but no LLM inference system uses it. The mechanism is orthogonal to all existing ML compression ideas — it does not compress weights, it eliminates a stochastic I/O penalty.

### Smallest experiment
Claim: On this Ryzen 5 7530U system's NVMe SSD, GC stalls ≥50 ms are measurable during Qwen3-1.7B inference and account for ≥5% of total wall-clock inference time. Test: (1) Run 200 single-token inference passes on Qwen3-1.7B GGUF while simultaneously writing 1 GB of random data to disk (simulating write activity that triggers GC). (2) Record per-token latency distribution. (3) Compare with inference-only (no background writes). Runtime: ~2 hrs. Go threshold: p95 latency with writes ≥ 1.3× p95 without writes. No-go finding: NVMe GC stalls are not measurable at this scale → SSD's internal wear leveling is sufficiently buffered → documents GC interference floor for all NVMe-path proposals on this hardware.

### Primary risk
Consumer NVMe SSDs may not expose namespace management to Windows StorNVMe without vendor tooling (many consumer drives expose NVMe namespace commands only via vendor-specific NVMe Admin commands, not through standard `IOCTL_STORAGE_MANAGE_DATA_SET_ATTRIBUTES`). Mitigation: the smallest experiment (GC stall measurement) is valid regardless of namespace support and produces a real structural finding; the full namespace-partition step is only attempted if the go threshold is met and the drive supports NS management (checkable via `nvme id-ctrl` vendor bit via `nvme-cli` on Windows).

---

## Convergence handles

- GGUF tensor offset table as a structured access schedule (GTBP, WSFI both rely on predictable, layer-sequential GGUF layout)
- Windows NVMe I/O path behavior under read-only workloads (NNPA, GTBP, WWTE all condition on measured NVMe behavior on this specific hardware)
- Per-block scale value dynamic range in GGUF Q4_K_M (DFSQ; also relevant to any future mixed-precision tier-encoding idea)
- OS working-set LRU as a zero-cost access-history oracle (WWTE; potentially useful to GTBP for adaptive lookahead depth)
- NTFS Cache Manager / SuperFetch interaction with mmap'd GGUF (WSFI, NCZC, WWTE all touch this layer)
- IEEE 754 denormal bit pattern as in-band metadata sentinel (DFSQ; generalizes to any format that reserves a non-representable encoding)
