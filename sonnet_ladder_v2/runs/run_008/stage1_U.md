# Stage 1 — Orientation U (Unconventional Substrate) — Run 008

---

## U-WSTB — Windows Working-Set Trimmer Bias [U / Track B]

### Name
Working-Set Trim Bias (WSTB) — U / Track B

### Mechanism
Track B. The Windows Memory Manager runs a working-set trimmer (WST) that periodically evicts pages from process working sets based on a soft-page-fault clock LRU. In a standard GGUF inference loop, every layer's weight pages compete in the same LRU pool; the trimmer cannot distinguish a "once-per-token" attention weight page from a "never-again" KV cache page. The substrate primitive being exploited: `VirtualLock` (kernel32, `VirtualLock(lpAddress, dwSize)`) pins a region into the physical working set, making it untouched by the trimmer. Counterintuitively, calling `VirtualLock` on only the hot attention weight pages (W_Q, W_K — smallest combined footprint on the transformer, and per CF11 the most compression-resistant without retraining) and explicitly `VirtualUnlock`+`VirtualFree`-friendly-cycling the MLP weight pages lets the trimmer's eviction pressure land where it is harmless — on pages that will be demand-paged back in a predictable linear scan order. This is not about adding bandwidth; it is about redirecting the trimmer's LRU pressure to pages whose re-fetch latency is hidden by sequential NVMe read-ahead. The trimmer becomes a background defragmentor that keeps DRAM residency biased toward inference-critical pages without any ML-level routing logic. Relies on CF11 (W_Q is the most concentrated attention matrix, lowest bytes after K=128 compression) and CF12 (tied embed full-rank, must stay resident). The primitive `VirtualLock` is documented, unconditionally available in Win32, and subject only to process working-set limit (adjustable via `SetProcessWorkingSetSize`). Does NOT require kernel modification.

### Residency arithmetic
Target: Qwen3-1.7B bf16 on 7.28 GiB RAM (3.2 GB model, 4 GiB margin). The MLP weights (W_gate, W_up, W_down per 28 layers at ~88 MB/layer MLP fraction ≈ 2.4 GB total) are the dominant weight footprint. If VirtualLock pins only the attention weights (W_Q, W_K, W_V, W_O per layer ≈ 28 × 4 × 2048 × 2048 × 2 bytes ≈ 950 MB at bf16) plus tied embed (~600 MB), trimmer pressure is redirected to 2.4 GB of MLP pages. With 4 GB DRAM margin, VirtualLock budget at ~1.55 GB is within the adjustable limit. NVMe at 3 GB/s sequential can re-page a 90 MB MLP layer in ~30 ms; at 8 layers-in-flight this is 240 ms re-page latency amortized over a decode step. Baseline without WSTB: trimmer evicts unpredictably, including attention pages mid-token, causing stall spikes. With WSTB: stall profile smooths to MLP-only page faults with predictable re-fetch. Payoff is measured in variance of per-token latency, not in raw tok/s ceiling — the metric is p95 latency reduction.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: R1/S2 "OS-Paging-Aware Weight Layout" (systems-os). Structural difference: that idea targeted layout optimization (no residency change); WSTB targets the trimmer's eviction *policy* via VirtualLock pinning to redirect pressure asymmetrically — no layout change required, and the payload is LRU-pressure steering, not byte positioning. Closest published method: Apple's LLM-in-a-Flash paper uses madvise-style sequential prefetch hints on macOS but does not exploit the Windows WST working-set pinning primitive or the asymmetric lock/unlock pattern. No paper in the LLM inference literature uses `VirtualLock` as a selective trimmer-pressure redirector. The move is genuine recognition: the WST has been doing eviction-pressure management for 30 years and LLM inference ignores it.

### Smallest experiment
Testable claim: `VirtualLock` on attention + embed pages reduces p95 per-token decode latency variance by ≥20% vs baseline (no VirtualLock), measured on Qwen3-1.7B running in ik_llama.cpp on the Ryzen 5 7530U at a 64-token decode run with 200 prompt-prefix tokens, under memory pressure induced by a background allocation that fills 2 GB of DRAM with dirty pages.  
Runtime estimate: 2–3 hours (build ik_llama.cpp with VirtualLock wrapper, run 20 decode trials at baseline + 20 trials with lock, compare latency distributions).  
Go/no-go: p95 latency reduction ≥20% at p<0.05 Wilcoxon → GO. No reduction → NO-GO.  
NO-GO structural finding: WST trimmer is not the binding pressure on this workload (mmap file-backed pages may be handled by the Cache Manager, not the WST, which is a distinct finding about Windows memory-management path for memory-mapped inference).

### Primary risk
Windows mmap-backed pages (used by ik_llama.cpp for GGUF) may be managed by the NTFS Cache Manager, not the WST; VirtualLock may not apply to them, making the entire pinning mechanism a no-op.  
Mitigation: run a 5-minute pre-check — attempt `VirtualLock` on the GGUF mmap region, observe return code and `QueryWorkingSet` behavior; if pages are NOT in the working set (Cache Manager path), pivot to `SetFileInformationByHandle` with `FileIoPriorityHintInfo` to steer Cache Manager I/O priority instead.

---

## U-PGWT — Page-Granularity Weight Tiering via NTFS Sparse File [U / Track B]

### Name
Page-Granularity Weight Tier (PGWT) — U / Track B

### Mechanism
Track B. NTFS supports sparse files: a region of a file that contains all-zero bytes does not occupy disk clusters — the filesystem stores only non-zero extents and returns zeroes on read without I/O. The substrate primitive: `DeviceIoControl(FSCTL_SET_SPARSE)` + `DeviceIoControl(FSCTL_SET_ZERO_DATA)` punches zero-extent "holes" into a file, freeing the underlying clusters on NVMe. The inference mechanism: after calibration, identify the weight pages (4 KB aligned, one GGUF tensor page at a time) whose contribution to output norm is below a threshold — these are candidates for zero-punch. On first access during inference, a zero-punched page returns all-zero bytes with no NVMe read issued (OS returns zeros from page-cache directly), effectively making those weights zero at inference time. This differs from quantization: the representation of the remaining weights is unchanged; only the punched pages go to zero. The technique is entirely a filesystem primitive — no ML library modification, no kernel changes. Calibration provides the per-page importance signal (mean squared activation touching that page, or per-weight magnitude summed per 4 KB page); pages below threshold get punched. NVMe free-cluster return is a side effect. This is post-hoc static sparsity *enforced by the filesystem*, not by the model. Grounded: NTFS sparse-file API is documented, Win32, available in Windows 11 Home. Does not rely on any CF finding for feasibility — the substrate primitive is the anchor.

### Residency arithmetic
Qwen3-1.7B bf16 = 3.2 GB. 4 KB pages = ~800K pages. If 5% of pages are below-threshold (calibration-selected), that is 40K pages × 4 KB = 160 MB freed on NVMe (clusters de-allocated) and 160 MB of zero reads eliminated at inference. On DRAM: zero pages are returned from page cache instantly; the 160 MB formerly paged in now never stalls on NVMe. For 70B models: 70B × 2 bytes ≈ 140 GB at bf16. At 5% page punch: 7 GB freed on NVMe. On the 7.28 GiB RAM target with full offload, the NVMe bandwidth saving is proportional: 7 GB × 3 GB/s NVMe ceiling = 2.3 s of IO per full model sweep eliminated. At 70B scale this matters even at 5%. Quality: zero-punched pages produce zero activations for those weights — the quality cost is determined by the importance threshold. At 5% punched, ΔNLL estimated < 0.2 nats (to be measured); at 10%, likely 0.5–1.5 nats.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: R1/S4-deferred Idea I "NTFS-Reflink Codebook Deduplication" — killed for Windows 11 Home limitation. PGWT uses SPARSE FILE punching (available on all NTFS volumes including Home), not reflinks (not available on Home). Structurally different mechanism. Closest published method: magnitude pruning → zero-weight storage. Structural difference: existing magnitude pruning stores an explicit sparse mask or compressed format; PGWT uses the filesystem's own sparse-extent representation, so the inference path reads zeros with NO I/O cost and NO format change. The GGUF file itself becomes self-sparsifying. No LLM inference paper exploits NTFS FSCTL_SET_ZERO_DATA as a weight-pruning storage primitive.

### Smallest experiment
Testable claim: NTFS `FSCTL_SET_SPARSE` + `FSCTL_SET_ZERO_DATA` on a GGUF file produces zero-read returns (no NVMe I/O) for punched pages, measurable via Windows `StorPort` or `Process Monitor` trace showing zero disk reads for those offset ranges.  
Runtime: 1–2 hours (write a 100 MB test file, punch 10% of pages, mmap and read, trace with Process Monitor).  
Go/no-go: zero NVMe reads for punched offsets in Process Monitor trace → GO (then proceed to calibration-threshold experiment). I/O still issued → NO-GO (sparse file semantics not honored for mmap reads on this configuration).  
NO-GO structural finding: Windows mmap does not short-circuit NVMe I/O for sparse zeros, which rules out PGWT and narrows understanding of Cache Manager sparse-file behavior under mmap.

### Primary risk
Memory-mapped access to sparse-zero regions may still issue NVMe reads (Cache Manager may not distinguish punched extents from valid-data extents under mmap semantics).  
Mitigation: test with both `CreateFile`+`ReadFile` (confirmed sparse-aware) and `CreateFileMapping`+`MapViewOfFile` (uncertain) before committing to the mmap path; if mmap doesn't honor sparse semantics, use a custom read loop.

---

## U-ZSTD-DICT — zstd Dictionary-Compressed Weight Tiles with OS Page-Cache Alignment [U / Track B]

### Name
zstd-Dict Weight Tile Compression (ZDWT) — U / Track B

### Mechanism
Track B. zstd (v1.5+) supports trained dictionaries (`ZDICT_trainFromBuffer`): a shared dictionary captures recurring byte patterns, and per-tile compressed streams reference it. The substrate primitives: zstd's `ZSTD_CDict` / `ZSTD_DDict` structures (libzstd, linked into ik_llama.cpp), plus the OS page cache for the decompressed tile pool. The mechanism: offline, cluster all weight tensor pages (4 KB or 16 KB tiles) into groups by layer type (W_Q, W_K, W_V, W_O, W_up, W_gate, W_down, tied embed) and train one zstd dictionary per group on a random sample of tiles. Store each compressed tile stream adjacent to its group dictionary in GGUF (extended chunk format). At inference, decompress tiles on demand into a fixed-size DRAM tile pool (the decompressed hot set); the page cache naturally evicts cold tiles. This differs from entropy coding of quantized nibbles (killed by ZipServ at R1): ZDWT targets bf16/fp16 raw bytes (or INT8 after light quantization) where the dictionary captures layer-type-specific byte-pattern statistics — not generic nibble histograms. Grounding: zstd dict training is a documented, CPU-only procedure (`ZDICT_trainFromBuffer`); decompression is AVX2-accelerated at ~3–5 GB/s single-core on Zen 3. Tile pool management is OS page cache. No new kernel code. Does not rely on CF13–CF15.

### Residency arithmetic
Qwen3-1.7B bf16 = 3.2 GB. Empirical zstd compression on bf16 neural weights typically achieves 1.3–1.6× with a per-group dictionary (byte patterns are not uniform; weights from the same layer type share magnitude-range statistics). At 1.4× compression: stored size ≈ 2.3 GB. NVMe read cost per token: at 70B scale with full NVMe offload, 1.4× throughput gain via compressed tile streaming = ~1.4× tok/s gain on the NVMe-bottleneck path (3 GB/s NVMe → 4.2 GB/s effective throughput). For 1.7B DRAM-resident: tile-decompression overhead at 3 GB/s (zstd AVX2) on 3.2 GB model = ~1.1 s to warm the full tile pool from NVMe; subsequent token decode hits DRAM hot set at DRAM bandwidth (11.5 GB/s). Payoff is primarily for 70B NVMe-offload scenario; secondary payoff at 1.7B is reduced cold-start I/O.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: R1/S2 "Arithmetic-Coded Weight Streams + Lazy Decode" — killed by ZipServ ASPLOS 2026 and AVX2 serial decode bottleneck. ZDWT structural differences: (1) uses zstd dictionary-compressed tiles, not arithmetic coding (zstd decompression is parallelizable and AVX2-hardware-accelerated unlike arithmetic coding); (2) per-group dictionaries trained on layer-type statistics, not generic entropy coding; (3) decompressed tile pool managed by OS page cache, not custom lazy-decode. ZipServ targets GPU inference; ZDWT targets CPU-only Ryzen with DRAM page cache as the pool manager. No direct prior art on zstd-dict-per-layer-group for CPU LLM inference with OS page-cache pool management.

### Smallest experiment
Testable claim: zstd level 3 with per-group trained dictionary achieves ≥1.3× compression ratio on bf16 Qwen3-1.7B weight tiles (16 KB), measurable by compressing 1% random tile samples per layer group.  
Runtime: 30 minutes (sample tiles from GGUF, train dictionaries with `ZDICT_trainFromBuffer`, measure compressed size ratio).  
Go/no-go: mean ratio ≥1.3× across all groups → GO (proceed to decompression throughput and tile-pool integration). < 1.1× across all groups → NO-GO (bf16 weight bytes too close to uniform entropy).  
NO-GO structural finding: bf16 neural weight tiles, even grouped by layer type, have near-uniform byte entropy at 4-16 KB granularity — rules out codec-based storage compression without quantization first.

### Primary risk
bf16 weight bytes may be near-entropy-maximum at tile granularity even with per-group dictionaries, yielding < 5% compression.  
Mitigation: pre-check with a 10-minute experiment compressing 100 MB of raw bf16 weights with `zstd --train` on a single layer-type group; if ratio < 1.05×, pivot to dictionary-over-INT4-weights (post-quantization residuals) which have lower entropy.

---

## U-NVME-QD — NVMe Queue-Depth Ladder for Layer Prefetch [U / Track B]

### Name
NVMe Queue-Depth Ladder Prefetch (NQDL) — U / Track B

### Mechanism
Track B. NVMe SSDs expose a submission queue depth (QD) that determines how many I/O requests can be in flight simultaneously. On the Samsung 980 / comparable NVMe in the Ryzen 5 7530U, QD1 delivers ~200–400 MB/s random read; QD32 delivers ~3 GB/s sequential read. The gap is 7–15×. Standard LLM NVMe offload (ik_llama.cpp default) issues layer weight reads as sequential blocking reads — effectively QD1 per layer because the next layer's weights are requested only after the current layer's computation completes. The substrate primitive: Windows `FILE_FLAG_OVERLAPPED` + `ReadFileEx` with an IOCP completion port (or, in the WSL/Linux path, `io_uring` submit queue with `IOSQE_IO_LINK`). The mechanism: issue the next N layer weight reads as overlapped I/O immediately after the current layer's GEMV starts (the GEMV occupies the CPU; the NVMe controller runs in parallel). This is a prefetch depth ladder — at QD=4 (4 layers in flight), the NVMe controller runs at near-peak throughput. The key unconventional insight: the transformer's fixed layer-access pattern is a perfect access-order oracle, making QD-ladder prefetch trivially predictable with zero model-internal logic. No ML prediction; the computation graph IS the prefetch schedule. Does not require CF13* NVMe latency numbers as a premise — uses only the published NVMe spec-sheet QD curve for this drive class.

### Residency arithmetic
At QD1: NVMe effective ~350 MB/s for ik_llama.cpp random-layer reads. At QD8: ~2.8 GB/s (near-sequential). Layer weight read dominates tok/s at 70B scale. Speedup from QD8 over QD1: ~8× on NVMe-bound path. Qwen3-70B at 4-bit ≈ 35 GB. Full sweep per token: 35 GB / 2.8 GB/s ≈ 12.5 s/token at QD8 vs 35 GB / 0.35 GB/s ≈ 100 s/token at QD1. Prefetch buffer cost: 4 layers in flight × 70B layer ≈ 4 × ~125 MB = 500 MB DRAM buffer — fits within the 7.28 GiB budget if model is otherwise on NVMe. At 1.7B scale (3.2 GB total, DRAM-resident): NQDL irrelevant — model fits in RAM. The idea is for the 70B NVMe-offload scenario.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list items: R2/S2 "NVMe Prefetch Sequencer via RoPE Fingerprints" (killed — wrong axis, residency not latency) and R3/S4-deferred INPOLT "NVMe Prefetch with Phase-Ordered Layer Tiling" (deferred — precondition: NVMe is binding). NQDL structural difference from INPOLT: INPOLT repacks GGUF layout for sequential dual-stream NVMe; NQDL uses queue-depth overlapped I/O without any layout change. The substrate primitive is the NVMe submission queue, not the filesystem layout. Closest published method: Apple LLM-in-a-Flash uses prefetch hints on macOS. Structural difference: NQDL uses Windows `FILE_FLAG_OVERLAPPED` + IOCP for true hardware queue-depth, not advisory hints. No published LLM NVMe offload paper (including FlexGen, DejaVu) exploits Windows IOCP for deep-queue NVMe on the CPU-only path.

### Smallest experiment
Testable claim: `FILE_FLAG_OVERLAPPED` with QD=8 concurrent reads of 125 MB sequential layer-sized blocks achieves ≥2× throughput vs sequential blocking reads on the Ryzen 5 7530U NVMe, measured via `GetOverlappedResult` timing.  
Runtime: 1 hour (write a 1 GB file, benchmark QD1 vs QD4 vs QD8 ReadFile overlapped vs blocking, record MB/s).  
Go/no-go: ≥2× throughput at QD8 vs QD1 → GO. < 1.2× → NO-GO (NVMe controller saturates at QD1 for large sequential reads, no queue benefit).  
NO-GO structural finding: Windows NVMe throughput for large sequential reads is already QD-saturated at QD1 (driver prefetches internally), ruling out application-level queue-depth control as a lever.

### Primary risk
Windows Storage Stack may serialize overlapped large-block reads through the same driver queue as blocking reads, negating QD benefit at this block size.  
Mitigation: test with 4 KB aligned reads at QD32 first (known to benefit from deep queue per NVMe spec); if 4 KB QD32 shows 5× over QD1, the storage stack is exposing true NVMe QD, and 125 MB-block QD8 will also benefit.

---

## U-MMAPF — Memory-Map Fault-Clustering via `PrefetchVirtualMemory` [U / Track B]

### Name
Prefetch-Virtual-Memory Fault Cluster (PVMFC) — U / Track B

### Mechanism
Track B. Windows 8+ exposes `PrefetchVirtualMemory` (kernel32, MSDN documented) which issues asynchronous prefetch for a list of virtual memory ranges without faulting them into the working set immediately. Unlike `VirtualLock`, it does not pin pages; it merely instructs the Memory Manager to begin background physical I/O for the listed ranges. The substrate primitive is the OS-level prefetch engine — the same engine used by Superfetch/ReadyBoost — operating on the process's own mmap'd regions. The mechanism: at the start of each transformer layer's compute, call `PrefetchVirtualMemory` with an array of {base, size} entries covering the NEXT N layers' weight pages. The OS issues the background prefetch asynchronously; by the time GEMV completes on the current layer, the next layer's pages are warm in RAM. This is not speculative prediction — the layer access order is deterministic. The OS prefetch engine is being used as a zero-cost pipeline stage: CPU does GEMV on layer L while OS prefetches layer L+1 pages. On a DRAM-resident 1.7B model this is trivial (pages already resident). On a 70B NVMe-offload scenario this is the entire tok/s lever. `PrefetchVirtualMemory` is unconditionally available on Windows 8+ (confirmed present in Windows 11 Home), requires no elevation. The key difference from VirtualLock: no working-set limit consumed; the pages are prefetched into the page cache without locking.

### Residency arithmetic
Identical to NQDL for 70B: with 4-layer lookahead prefetch, effective NVMe read bandwidth approaches sequential peak (~3 GB/s*) because the OS's prefetch engine issues reads ahead of demand. *CF13 NTFS extent-prefetch 3 GB/s number is unverified in v2; the smallest experiment re-derives this. Without CF13: use raw NVMe spec 3 GB/s sequential as the upper bound; real measured value may be lower. For 1.7B DRAM-resident: payoff is warm-start time only — pages already resident after first sweep, so `PrefetchVirtualMemory` is a no-op after the first token. The primary payoff is 70B-class cold-start and cross-layer decode bandwidth.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: R2/S2 "NVMe Prefetch Sequencer via RoPE Fingerprints" (wrong axis). PVMFC structural difference: no layer prediction required; uses deterministic access order. The substrate primitive is `PrefetchVirtualMemory` (Win32 documented, not a kernel modification), which no published LLM inference framework invokes. Apple LLM-in-a-Flash uses macOS `madvise(MADV_WILLNEED)` — structurally analogous; PVMFC targets the Windows-equivalent API with no macOS connection. The closest NTFS extent-prefetch framing in v2 SUMMARY is CF13* (unverified); PVMFC is distinct because it does not depend on NTFS extent contiguity — it works even for fragmented GGUF files by explicitly listing virtual ranges.

### Smallest experiment
Testable claim: `PrefetchVirtualMemory` called for layer L+1 pages immediately after GEMV on layer L begins reduces per-layer average fault time by ≥30% on Qwen3-1.7B run under simulated NVMe pressure (achieved by calling `VirtualAlloc`+`VirtualFree` to evict pages between layers).  
This experiment also re-derives the CF13* NVMe-engagement number — it measures actual background prefetch throughput on this hardware.  
Runtime: 3–4 hours (instrument ik_llama.cpp with `PrefetchVirtualMemory` calls, measure per-layer latency with and without pressure).  
Go/no-go: ≥30% per-layer latency reduction under pressure → GO (v2 CF13-equivalent confirmed). < 10% → NO-GO (`PrefetchVirtualMemory` does not engage the NVMe controller fast enough to hide demand latency on this hardware).  
NO-GO structural finding: Windows `PrefetchVirtualMemory` on memory-mapped files either (a) serializes behind demand faults or (b) requires file-backed pages with different Cache Manager behavior — structural finding about Windows prefetch path for GGUF inference.

### Primary risk
`PrefetchVirtualMemory` may operate only on anonymous allocations (VirtualAlloc), not file-backed mappings (MapViewOfFile), making it a no-op for GGUF mmap access.  
Mitigation: MSDN confirms `PrefetchVirtualMemory` works on both anonymous and file-mapped regions; verify with a 5-minute test calling it on a mmap'd file and checking `QueryWorkingSet` to confirm the pages enter the working set.

---

## U-FSCTL-PRIO — NTFS I/O Priority Steering for Layer-Access Ordering [FREE SWING] [U / Track B]

### Name
NTFS I/O Priority Steering (FIOPS) — U / Track B [FREE SWING]

### Mechanism
Track B. Windows exposes per-handle I/O priority via `SetFileInformationByHandle` with `FileIoPriorityHintInfo` (`IoPriorityHintNormal`, `IoPriorityHintLow`, `IoPriorityHintVeryLow`). This routes I/O requests through different Storage Driver Stack priority queues: NORMAL priority requests preempt LOW and VERY_LOW in the NVMe driver's internal scheduling. In a multi-process system (background OS tasks, antivirus, indexing), the GGUF mmap file handle competes for NVMe bandwidth. Explicitly setting the GGUF file handle to `IoPriorityHintNormal` (the default, but explicitly asserted per inference call) and explicitly setting all competing background file handles to `IoPriorityHintVeryLow` (via the process's own temp files, logging, calibration reads) ensures inference I/O preempts all self-generated background I/O. On Windows 11 with background update indexing, this matters more than on a clean server. The substrate primitive: `FileIoPriorityHintInfo` is documented Win32, available without elevation for normal-priority setting. The unconventional move: treating inference as a real-time I/O class and explicitly managing I/O priority via per-handle steering rather than accepting the OS's default round-robin. No ML logic required.

### Residency arithmetic
Payoff is not in raw residency or bpw — this is a scheduling intervention. Payoff measured in: reduction of variance-of-per-layer-fetch time caused by NVMe bandwidth contention with background processes. On a lightly loaded desktop (Ryzen 5 7530U with Windows 11 Home, background indexing active), background I/O can claim 20–40% of NVMe bandwidth in bursts. Explicit priority steering recovers this 20–40% for inference. At 70B NVMe-offload: 35 GB × 2.4 GB/s effective (with priority) vs 1.5 GB/s effective (with contention) = 1.6× tok/s improvement during background activity. Marginal on a clean machine; significant in the user's real deployment scenario (desktop with normal OS activity).

### Novelty gloss vs the kill list and the published landscape
No kill-list item directly matches. No published LLM inference framework uses `SetFileInformationByHandle` for I/O priority steering. The idea has no neighbor in the published LLM inference landscape because it treats LLM inference as a real-time I/O consumer requiring explicit priority management — a systems-engineering concept entirely absent from the ML inference literature. The move is genuinely orthogonal to everything in the kill list.

### Smallest experiment
Testable claim: setting GGUF file handle `IoPriorityHintNormal` while a background process performs `IoPriorityHintLow` random reads on a second file reduces Qwen3-1.7B full-model sweep time by ≥15% vs baseline where both use default priority, on the Ryzen 5 7530U under artificial contention.  
Runtime: 1 hour (write a contention-simulation tool that performs background reads at LOW priority; compare sweep times with and without explicit NORMAL priority on GGUF handle).  
Go/no-go: ≥15% sweep reduction under contention → GO. < 5% → NO-GO (NVMe controller's internal scheduler already separates priority classes; Windows driver layer adds no further separation).  
NO-GO structural finding: Windows NVMe driver stack does not honor `FileIoPriorityHintInfo` at the drive-level — storage priority steering is OS-visible only, not propagated to NVMe command queues.

### Primary risk
Windows 11 Home NVMe driver may not propagate `FileIoPriorityHintInfo` to NVMe command priority bits (NCQ priority), reducing the mechanism to OS-scheduler-only queuing with no hardware enforcement.  
Mitigation: test on two processes with explicit background contention; if priority steering produces ≥15% in wall-clock, hardware priority bits are not required for the effect.

---

## Convergence handles

- `PrefetchVirtualMemory` / `VirtualLock` engagement on mmap'd file-backed pages under Windows 11 Home mmap path (shared precondition for WSTB and PVMFC; must be verified before either proceeds)
- NVMe effective throughput at QD1 vs QD8 for 125 MB sequential blocks on Ryzen 5 7530U NVMe (shared baseline for NQDL and PVMFC; re-derives CF13*)
- NTFS sparse-file zero-read semantics under `MapViewOfFile` (PGWT precondition; also relevant to FSCTL-PRIO)
- zstd dict compression ratio on bf16 weight tiles grouped by layer type (ZDWT; if < 1.1×, compression-before-quantization path is closed)
- Windows Storage Driver Stack I/O priority propagation to NVMe hardware queues (FIOPS; determines whether FSCTL_SET_PRIORITY_HINT is hardware-enforced or OS-only)
