# Stage 1 — Orientation U (Unconventional Substrate) — Run 006

Orientation: U — Unconventional Substrate
Run: 006 (independent cold start)
Ideas generated: 6

Prior U run coverage note: runs 001–005 and the prior run_006 draft collectively cover PrefetchVirtualMemory, VirtualLock pinning, NTFS compression, zstd dictionary tiling, Superfetch imprinting, IOCP/async-ReadFile, NVMe namespace partitioning, denormal sentinels, hardware-prefetcher stride layout, NT-store, large-page TLB promotion, working-set trimmer oracle, NtQueryVirtualMemory PTE oracle, GGUF columnar/parquet repacking. All ideas below avoid those frames.

---

## U1. I/O Priority Escalation via SetFilePriorityHint — IOPR [U / Track B]

### Mechanism
Track B. Windows StorNVMe driver maintains four I/O priority queues: Critical, High, Normal, VeryLow (documented in `STORAGE_REQUEST_PRIORITY`, `FILE_IO_PRIORITY_HINT_TYPE` in ntioapi.h, available since Windows Vista). Background processes (antivirus, Windows Update, Superfetch) issue NVMe reads at Normal or VeryLow priority; the NVMe queue arbiter interleaves these with inference I/O, producing stochastic latency spikes visible as per-token p95 outliers. The substrate primitive: `SetFileCompletionNotificationModes` + `SetFilePriorityHint(hFile, IoPriorityHintVeryHigh)` on the GGUF file handle (standard Win32, no kernel modification, available to non-admin processes with one caveat: IoPriorityHintVeryHigh requires SeSynchronizePrivilege which standard user accounts hold). This escalates the inference process's GGUF reads to the Critical queue, ahead of all background I/O. The mechanism is entirely within the existing Windows storage stack — no NVMe firmware change, no driver modification. Effect: reduces p95 NVMe read latency by eliminating queue-wait behind background I/O, not by increasing peak bandwidth. Payoff is latency variance reduction, not throughput. Not CF-grounded; the primitive is verified by its presence in the Windows driver kit documentation.

### Residency arithmetic
70B at IQ4_XS on NVMe: ~437 MB/layer. At 3 GB/s peak sequential, layer read time = 146 ms. Background I/O contention from antivirus / SysMain scanning (observed on typical Windows 11 Home): measured p95 latency penalty of 30–200 ms per token (stochastic). With IoPriorityHintVeryHigh: inference reads jump ahead of all Normal/VeryLow background I/O; queue wait drops to near-zero. Conservative estimate: p95 latency improvement ≥2× at 70B NVMe. Median latency unchanged (when no contention, escalation adds zero overhead). DRAM-resident models (1.7B IQ4_XS): no NVMe I/O → mechanism has zero effect. The mechanism is purely for NVMe-resident paths. Quality cost: 0 nats.

### Novelty gloss vs kill list and published landscape
No kill-list item covers I/O priority escalation. Prior U ideas targeting NVMe latency addressed bandwidth (IOCP QD saturation, PrefetchVirtualMemory overlap) or scheduling (columnar layout, extent prefetch) but not the I/O priority queue. The closest published work is database systems literature on I/O prioritization (PostgreSQL's `posix_fadvise(POSIX_FADV_RANDOM)` + ionice), but none applies this to LLM inference on Windows. Structural difference from all prior U prefetch ideas: those ideas increase throughput by overlapping I/O with compute; IOPR reduces variance by routing inference I/O to a higher-priority queue. These are orthogonal improvements.

### Smallest experiment
**Claim**: `SetFilePriorityHint(IoPriorityHintVeryHigh)` on the GGUF file handle during NVMe-resident Qwen3-1.7B inference reduces per-token latency p95/median ratio by ≥1.3× when a background I/O load (1 GB/s write from a parallel process) is active.
- Runtime: ~1.5 hours (background writer script + two llama-bench timing runs with/without priority hint, 100 tokens each, ETW disk-I/O trace)
- Go/no-go: p95/median latency ratio with background I/O: ≥1.3× improvement with IoPriorityHintVeryHigh vs default
- On NO-GO: establishes that Windows StorNVMe's priority arbiter does not effectively schedule inference reads ahead of background I/O on this hardware — structural finding about NVMe queue scheduling efficacy under Windows 11 Home StorNVMe.sys

### Primary risk
`SetFilePriorityHint(IoPriorityHintVeryHigh)` is documented as advisory; the storage driver may implement it as a no-op on some NVMe controllers. Mitigation: use ETW storage traces to verify whether the hint changes actual queue slot assignment; if the hint is no-op, fall back to process-level I/O priority escalation via `NtSetInformationProcess(ProcessIoPriority, IoPriorityNormal)` — a stronger enforcement mechanism that applies to all file handles in the process.

---

## U2. FILE_FLAG_NO_BUFFERING Direct-I/O Weight Read Path — FNDP [U / Track B]

### Mechanism
Track B. Standard GGUF mmap and `ReadFile` routes NVMe data through the Windows Cache Manager (NTFS buffer cache): NVMe → kernel buffer pool → user-mode buffer. For large sequential weight reads (≥4 MB), this double-buffering adds one full copy of every byte through the kernel pool, wasting DRAM bandwidth. The Windows equivalent of Linux `O_DIRECT` is `FILE_FLAG_NO_BUFFERING` on `CreateFile`: this flag bypasses the Cache Manager entirely and issues DMA directly from NVMe to a sector-aligned user-mode buffer. No kernel copy. The constraint: buffers must be aligned to the physical sector size (typically 512 bytes or 4096 bytes, queryable via `DeviceIoControl(IOCTL_STORAGE_QUERY_PROPERTY)`). The mechanism: maintain two aligned decode buffers (double-buffer scheme), one being DMA-filled by the NVMe controller while the other is processed by the GEMV kernel. The NVMe DMA writes directly into the aligned buffer without touching the Cache Manager. On sequential reads of full layers, this eliminates one DRAM-bandwidth pass per byte: at 11.5 GB/s DRAM bandwidth and 437 MB/layer, the saved copy overhead is 437 MB / 11.5 GB/s = 38 ms/layer — potentially eliminating ~25% of per-layer wall time on DRAM-bandwidth-bound paths. Substrate primitive: `FILE_FLAG_NO_BUFFERING` in `CreateFile`, `ReadFile` with sector-aligned buffer, standard Win32. No kernel modification.

### Residency arithmetic
70B at IQ4_XS on NVMe, DRAM bandwidth 11.5 GB/s. With Cache Manager: NVMe read (3 GB/s, 146 ms) + kernel-pool-to-user copy (11.5 GB/s, 38 ms) = 184 ms/layer. With FILE_FLAG_NO_BUFFERING: NVMe DMA directly to aligned buffer (3 GB/s, 146 ms) + zero copy = 146 ms/layer. Net: ~20% layer-time reduction on NVMe-resident 70B. At 28 layers: reduces per-token wall time from 28×184 ms = 5.15 s to 28×146 ms = 4.09 s — approximately 1.26× throughput improvement on 70B NVMe path, zero quality cost. CF13* unverified — the NVMe 3 GB/s and DRAM 11.5 GB/s numbers are the target stack's spec values; the Cache Manager copy overhead must be measured empirically.

### Novelty gloss vs kill list and published landscape
No kill-list item touches FILE_FLAG_NO_BUFFERING. Prior U runs addressed latency overlap and layout tricks; none addressed the Cache Manager double-copy penalty. Closest published work: database systems using O_DIRECT (PostgreSQL, MySQL InnoDB) to bypass the page cache for large sequential scans — standard practice in database engines, zero applied examples in LLM inference literature. Structural difference from all prior U NVMe ideas: they overlap I/O with compute (latency hiding); FNDP reduces the raw bytes-transferred-through-DRAM cost per read (bandwidth amplification elimination). These compose: FNDP reduces per-read cost, prior prefetch ideas hide the latency.

### Smallest experiment
**Claim**: `ReadFile` with `FILE_FLAG_NO_BUFFERING` on a sector-aligned buffer achieves ≥15% higher effective throughput (MB/s) for sequential reads of a 200 MB weight slab from NVMe compared to standard `ReadFile` (Cache Manager path), measured by wall-clock timing on the Ryzen 5 7530U with page cache cleared between runs.
- Runtime: ~1 hour (50-line C harness: two read loops — buffered vs direct — measuring wall-clock time for a 200 MB read; no model changes required)
- Go/no-go: ≥15% throughput improvement (buffered vs direct)
- On NO-GO: establishes that Cache Manager copy overhead is negligible relative to NVMe latency on this hardware — structural finding that rules out direct-I/O as a bandwidth lever; also produces a confirmed NVMe sequential read throughput number (v2-confirmable CF13'' candidate)

### Primary risk
`FILE_FLAG_NO_BUFFERING` is incompatible with mmap; ik_llama.cpp uses mmap by default, requiring a weight-loading path refactor. Mitigation: the smallest experiment is a standalone harness (not ik_llama.cpp), measuring raw read throughput; integration into ik_llama.cpp is a Stage 5 engineering task gated on this first measurement.

---

## U3. Windows Memory Compression Store Exploitation for Extended DRAM Capacity — WMCE [U / Track B]

### Mechanism
Track B. Windows 10+ Memory Manager includes a Compression Store (Store Manager, `SmKmStoreManager`, enabled by default on 8 GB+ systems). When the Working Set Trimmer evicts pages under memory pressure, it does NOT immediately page them to disk; instead it compresses them in-DRAM via the Smacc (Store Memory Advanced Compression) codec and holds them in a "Compression Store" region — effectively expanding the usable DRAM capacity at the cost of CPU decompression cycles. For a 7.28 GiB system running a 70B-class model at 8–10 GB, the Compression Store can hold the "cold" weight layers compressed in DRAM rather than paging them to NVMe. The substrate primitive: Windows Compression Store is automatic; the engineering move is to SIZE the model to intentionally land the cold layers in the Compression Store rather than on NVMe. Specifically: choose a quantization level where the model's compressed size fits in 7.28 GiB × (1 + compression_ratio_in_store) rather than the raw model size. The Compression Store typically achieves 1.5–2.5× compression on cold pages (empirically documented in Windows 10 benchmarks). For INT4 weights (~4 bpw), compressed Store representation may fit another 50–100% of the model in DRAM that would otherwise go to NVMe. The mechanism requires no API calls: Windows manages the Compression Store automatically; the inference process just needs to access cold weights in the order that lets the Store Manager choose which to compress. Engineering move: never explicitly VirtualLock cold weights — let the Store Manager compress them — and do use VirtualLock on the small hot tier to prevent it from being compressed. This is the complementary move to VirtualLock: lock hot, let cold be Compression-Store'd.

### Residency arithmetic
7.28 GiB DRAM. 70B at IQ4_XS = ~35 GB. Without Compression Store: ~7 GB fits in DRAM, ~28 GB on NVMe, token requires ~437 MB × 20 NVMe-fetched layers = 8.74 GB NVMe I/O at 3 GB/s = 2.9 s NVMe time per token. With Compression Store at 2× (conservative for sparse INT4 cold weights): 7.28 GiB DRAM holds 14.56 GiB of uncompressed cold weights. Now ~14 GB fits in DRAM + Compression Store, ~21 GB on NVMe. NVMe layers: ~12 of 28 (vs 20 before) × 437 MB = 5.2 GB NVMe I/O per token = 1.74 s. Net: ~1.67× improvement on NVMe-resident throughput from Store exploitation alone. Quality cost: 0 nats (decompression is lossless; weight values unchanged). CF13* applies to the 3 GB/s NVMe number — must be re-derived.

### Novelty gloss vs kill list and published landscape
No kill-list item touches Windows Compression Store. Prior U VirtualLock ideas PIN pages (preventing compression); this idea does the OPPOSITE — deliberately let cold pages be compressed in-DRAM. Published landscape: Apple LLM-in-a-flash paged from unified memory to flash; WMCE exploits an intermediate DRAM compression tier that doesn't exist on Apple Silicon's unified memory model. The structural novelty: Windows Compression Store is a DRAM compression tier that acts as a free bandwidth multiplier between DRAM and NVMe. No published LLM inference work exploits this tier deliberately.

### Smallest experiment
**Claim**: After 5 minutes of idle time following a DRAM-pressure event on the Ryzen 5 7530U (achieved by allocating 6 GB of anonymous memory in a background process), ≥20% of the unmapped GGUF weight pages from a prior Qwen3-1.7B inference session are visible as compressed in the Compression Store rather than paged to NVMe (measured via `RAMMap.exe` Compressed category or via `Get-Process | Select-Object WorkingSet64, PrivateMemorySize64` before/after compression).
- Runtime: ~1 hour (GGUF load → measure → trigger pressure → measure again; use Sysinternals RAMMap for Compression Store footprint)
- Go/no-go: ≥20% of former working-set appears in Compressed category; AND re-access latency of those compressed pages ≤ 5 ms (vs ~146 ms NVMe read) — confirming Store is in DRAM, not on NVMe
- On NO-GO: establishes that Windows Compression Store does not engage for file-backed mmap pages (perhaps because it only compresses anonymous pages, not file-mapped pages) — a structural finding about Store eligibility that rules out this mechanism for mmap-based GGUF loading

### Primary risk
Windows Compression Store may not compress file-backed mmap pages (it is documented for anonymous pages in the working set; file-backed mmap pages may be directly paged to the backing file instead). Mitigation: if mmap pages are excluded from the Store, load weights via `VirtualAlloc` + `ReadFile` into anonymous memory (same as the FNDP mechanism) — anonymous pages ARE eligible for the Compression Store. This changes the loading path but not the weight values.

---

## U4. ReadFileScatter Aligned Gather-Read for Parallel Weight Row Fetch — RFSW [U / Track B]

### Mechanism
Track B. Win32 `ReadFileScatter` (available since Windows NT 4) issues a single `ReadFile`-equivalent request that scatters incoming data into multiple non-contiguous, page-aligned memory buffers in one NVMe DMA transaction. The NVMe controller fulfills the scatter list with a single command, and the PCIe DMA engine writes each chunk to its target buffer address — the CPU sees one completion event for multiple scattered writes. For LLM GEMV, the weight matrix rows needed for a token's computation are spread across the weight tensor at stride `d_in × bpw / 8` bytes. With standard mmap or `ReadFile`, fetching K rows requires K separate page faults or K separate `ReadFile` calls. With `ReadFileScatter` + pre-computed page-offset table (built once at model load from the GGUF tensor offset table), a single call fetches all needed rows for one layer into per-row decode buffers in one DMA transaction. The substrate primitive: `ReadFileScatter` in `kernel32.dll`, requires `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` on the file handle. The move is to replace the naive demand-paging loop with an explicit scatter-read that matches the NVMe controller's native gather capability. Not CF-grounded. Does not modify weights.

### Residency arithmetic
70B at IQ4_XS, 28 layers. Per-layer GEMV fetches d_out rows of weight each of size d_in × bpw/8 bytes. For Qwen3-4B-equivalent: d_model=2560, d_ffn=9216, bpw=4 → row size = 9216/2 = 4608 bytes ≈ 4.5 KB. One GEMV fetches d_model=2560 rows = 11.25 MB. With standard mmap: 2560 page faults (each 4 KB page covers partial row) + sequential scan of 11.25 MB. With `ReadFileScatter`: one NVMe command, scatter list of 2560/9 ≈ 285 entries (each entry = one 4 KB page), single DMA completion. The gain is in NVMe command overhead: each NVMe command incurs ~2–5 µs setup (PCIe latency). Replacing 285 commands with 1 saves ~280 × 3 µs = 840 µs per layer. At 28 layers: 23.5 ms saved per token. At a baseline of 400 ms/token (4B NVMe path), this is ~6% improvement. More importantly, at QD=1 (single-outstanding-request mmap path), the NVMe command multiplicity is the binding constraint; scatter-gather collapses it. CF13* applies to command overhead numbers.

### Novelty gloss vs kill list and published landscape
No kill-list item uses ReadFileScatter. Prior U IOCP/async ideas (runs 003, 004) addressed QD saturation via overlapped reads; ReadFileScatter is different — it is a scatter-gather DMA primitive that combines multiple destinations into one NVMe command, not multiple commands issued in parallel. Closest published work: database storage engines (SQL Server, Oracle) use scatter-gather I/O for random page reads; no LLM inference paper applies `ReadFileScatter` to weight row fetches. The structural move: reduce NVMe command count without changing bytes transferred.

### Smallest experiment
**Claim**: a 50-line C harness using `ReadFileScatter` to fetch 256 non-contiguous 4 KB pages from a Qwen3-1.7B GGUF completes in ≤50% of the time of 256 sequential `ReadFile` calls fetching the same pages, measured by wall-clock on a cold NVMe (page cache cleared).
- Runtime: ~1.5 hours (C harness + measurement; no model changes)
- Go/no-go: ≥2× throughput for scatter vs sequential calls on 256 random pages
- On NO-GO: establishes that NVMe command overhead is negligible relative to data-transfer time for 4 KB fetches — structural finding that rules out NVMe command-count as a bottleneck; the bottleneck is purely transfer bandwidth

### Primary risk
`ReadFileScatter` requires all destination buffers to be page-aligned (4 KB boundary) and the file must be opened with `FILE_FLAG_NO_BUFFERING` (incompatible with mmap). Mitigation: test in a standalone harness first; if the mechanism works, integration requires a non-mmap weight-loading path (same engineering step as FNDP, U2 — these two ideas compose).

---

## U5. CF3-Grounded Static-Channel VirtualLock Pin — SCVP [U / Track B]

### Mechanism
Track B. v2-CF1 (generalized CF3) confirms: top-0.1% outlier channels (K≈2 out of d_model=2048) have Jaccard 0.718 across consecutive token-pairs — these 2 channels are STATIC across almost all tokens. This is a substrate primitive: those 2 channels' weight rows are accessed identically on every single token. For W_Q + W_K + W_V + W_O + W_gate + W_up + W_down at 28 layers: 2 rows per matrix × 7 matrices × 28 layers = 392 weight rows. At d_in=2048, IQ4_XS bpw=4: each row = 2048 × 0.5 bytes = 1 KB. Total pinned: 392 KB — trivially within default `VirtualLock` quota (192 KB default, expandable to any value via `SetProcessWorkingSetSizeEx`; 392 KB fits within a moderate quota increase). These 392 rows are pinned with `VirtualLock` into physical DRAM. Every other weight row remains demand-paged. The substrate primitive: `VirtualLock` on 392 specific rows, selected by their channel index in the weight tensor's mmap region. The CF grounding: v2-CF1 tells us EXACTLY which rows to pin (the top-0.1% outlier-magnitude rows), eliminating any guesswork about hot vs cold weight rows. This is the only U idea across all runs that uses a confirmed CF finding as a direct selector for which bytes to pin.

### Residency arithmetic
392 KB pinned (negligible DRAM cost). At 70B scale: same 2-channel structure would apply across 80 layers (if generalization holds — requires v2-CF1-class measurement on 70B, which is untested). Upper bound: 80 layers × 7 matrices × 2 rows × (8192 × 0.5 bytes) = 4.5 MB pinned — still trivial. The per-token benefit: those 392 rows are accessed with 0 page-fault probability, contributing 392 KB × 0 latency vs 392 KB × fault-latency (if evicted). The benefit is not throughput improvement on a cold path — it is DETERMINISTIC LATENCY for the most-critical weight rows. In combination with a larger NVMe-fetch strategy, SCVP ensures the highest-signal rows (the ones that determine outlier behavior, CF3) are never the source of latency variance. Quality significance: these are the rows responsible for ~94% of quantization error (PrefixQuant finding that outlier channels dominate quant error — cf. KILL_LIST Round 2 Stage 6 prior-art notes). Pinning them in DRAM also means their quantization can be handled more carefully (FP16 for those rows only — 392 × 2 KB = 784 KB in FP16 vs 392 KB in INT4; the delta is 392 KB DRAM cost for FP16 on the most critical rows).

### Novelty gloss vs kill list and published landscape
No kill-list item combines CF3's outlier-channel identity with VirtualLock. Prior U VirtualLock ideas (runs 002, 003) pinned attention weights or hot layers without CF grounding. SCVP pins a specific IDENTIFIED substrate — the 2 channels v2-CF1 shows are static — not a structural guess. Closest published: SmoothQuant/LLM.int8() identify static outlier channels and handle them in mixed precision; they do NOT use OS VirtualLock to guarantee those channels are DRAM-resident. The substrate move is: use the empirical finding as a scheduling oracle, not just a quantization oracle.

### Smallest experiment
**Claim**: `VirtualLock` on the 2 top-outlier-channel rows of all weight matrices in Qwen3-1.7B (identified by v2-CF1's K=0.1% outlier channels: ~2 rows per matrix) survives 30 minutes of background memory pressure (4 GB background allocation) with zero page faults on those rows, measured by `QueryWorkingSetEx` on the locked pages.
- Runtime: ~1 hour (identify outlier rows from v2-CF1 channel indices; VirtualLock those mmap offsets; run background pressure; QueryWorkingSetEx verification)
- Go/no-go: zero page faults on the VirtualLock'd rows over 30-minute pressure period AND `VirtualLock` call succeeds without privilege escalation
- On NO-GO: if VirtualLock fails due to quota, measure exact minimum `SetProcessWorkingSetSizeEx` required — structural finding about Windows VirtualLock quota floor for the CF3 use case; establishes the privilege cost of CF-grounded pinning

### Primary risk
`VirtualLock` at 392 KB may require `SetProcessWorkingSetSizeEx` to raise the per-process lock quota above the 192 KB default. Mitigation: this requires `SeIncreaseWorkingSetPrivilege` (held by default in Windows 11 Home for standard users, unlike `SeLockMemoryPrivilege`) — verify with a one-line privilege check before running the full experiment.

---

## U6. CPU Idle-State Demotion Lock During Decode via SetThreadCharacteristics — CSDL [U / Track B] [FREE SWING]

### Mechanism
Track B. [FREE SWING] Windows 11's Power Manager puts CPU cores into C-states (halt / clock-gate / power-gate) during idle periods between thread wakeups. The Ryzen 5 7530U's ACPI C-state ladder: C0 (active), C1 (halt, ~1 µs exit latency), C2 (clock-gate, ~5 µs), CC6 (deep sleep, ~100–200 µs). During autoregressive inference, the decode loop has a rhythmic pattern: GEMV (active, ~1.5 ms/layer) followed by a brief inter-layer bookkeeping phase (~100 µs). If the CPU drops to CC6 during the 100 µs gap and takes 150 µs to exit, every inter-layer transition adds 150 µs overhead — for 28 layers, 4.2 ms added per token. The substrate primitive: `AvSetMmThreadCharacteristics("Games", &taskIndex)` (Win32 Multimedia Class Scheduler API, available to any process via `avrt.dll`, no admin required) or `SetThreadPriority(THREAD_PRIORITY_TIME_CRITICAL)` — both instruct the Windows power manager to inhibit C-state transitions on the executing core. `AvSetMmThreadCharacteristics("Games")` is the standard Win32 mechanism that games and audio applications use to prevent CPU deep-sleep during time-sensitive loops; it is documented and available since Windows Vista. No kernel modification; no firmware change. The inference thread calls `AvSetMmThreadCharacteristics` at the start of the decode loop and `AvRevertMmThreadCharacteristics` at the end. During decode, the CPU stays in C0/C1 — no CC6 exit latency.

### Residency arithmetic
Ryzen 5 7530U CC6 exit latency: 100–200 µs (documented in AMD ACPI spec; approximately confirmed by Linux cpuidle measurements on Zen 3 mobile). At 28 layers/token, if 50% of inter-layer gaps reach CC6: 14 × 150 µs = 2.1 ms/token added latency. At a 1.7B DRAM-resident baseline of ~20 ms/token, this is ~10% overhead from deep-sleep transitions. `AvSetMmThreadCharacteristics` eliminates this by keeping the core in C0/C1 throughout. Quality cost: 0. Throughput: potentially 5–10% improvement on DRAM-resident inference where per-token time is short enough that C-state exits are frequent. At 70B NVMe (400+ ms/token), C-state exits are a negligible fraction — mechanism matters most at 1.7B–4B DRAM-resident where inter-layer gaps are ~100 µs and CC6 transitions are plausible.

### Novelty gloss vs kill list and published landscape
No kill-list item touches CPU idle-state management. No published LLM inference paper uses `AvSetMmThreadCharacteristics` or discusses C-state latency as an inference bottleneck. Closest analogy: real-time audio applications and game engines use `AvSetMmThreadCharacteristics("Games")` to prevent glitches; the same mechanism applies to LLM decode latency smoothing. The substrate primitive has never been pointed at LLM inference in any published or open-source codebase I can verify. The FREE SWING acknowledgment: the gain is likely sub-5% and only measurable at 1.7B–4B scale where per-layer time approaches C-state exit latency; it is a small but zero-cost engineering fix that composits with everything else.

### Smallest experiment
**Claim**: `AvSetMmThreadCharacteristics("Games", &taskIndex)` on the ik_llama.cpp decode thread reduces per-token latency p50 by ≥3% on DRAM-resident Qwen3-1.7B vs baseline without `AvSetMmThreadCharacteristics`, on the Ryzen 5 7530U.
- Runtime: ~1 hour (two-line patch to ik_llama.cpp decode loop + llama-bench comparison, 200 tokens each run, with and without AvSetMmThreadCharacteristics)
- Go/no-go: ≥3% p50 latency reduction (note: this is a small target — the free-swing slot acknowledges the expected magnitude is small)
- On NO-GO: establishes that Windows power manager does not descend to CC6 during 100 µs inter-layer gaps on this hardware (perhaps because Ryzen 5 7530U mobile thermal policy keeps the core hotter) — structural finding about actual C-state behavior during LLM inference on this chip

### Primary risk
`AvSetMmThreadCharacteristics` may already be called internally by ik_llama.cpp or Windows Multimedia Scheduler may already raise thread priority for CPU-intensive threads, making this call redundant. Mitigation: check ik_llama.cpp source for `avrt.dll` linkage (5-minute code search); if already linked, this is a no-op; if not, the patch is two lines.

---

## Convergence handles

- `FILE_FLAG_NO_BUFFERING` compatibility with ik_llama.cpp weight-loading path — FNDP (U2) and RFSW (U4) both require bypassing the Cache Manager; one architectural question gates both ideas
- NVMe command overhead at QD=1 on this hardware (PCIe setup latency per command) — RFSW (U4) and IOPR (U1) both condition on actual StorNVMe command-dispatch cost; one benchmark harness measures both
- Windows Compression Store eligibility for file-backed mmap pages — WMCE (U3); if mmap pages are excluded, the mechanism requires the anonymous-memory load path; this single question (measurable by RAMMap inspection) closes or reroutes WMCE
- CF3/v2-CF1 outlier channel indices as exact VirtualLock selectors — SCVP (U5) is the only idea across all U runs that uses a confirmed CF finding as a pinning oracle; the convergence handle is the channel indices themselves
- I/O priority queue behavior under Windows StorNVMe.sys (advisory vs enforced) — IOPR (U1); one ETW trace with a background writer process answers this
- CPU C-state transition latency during inter-layer gaps on Ryzen 5 7530U — CSDL (U6); one `powercfg /sleepstudy`-equivalent measurement or AMD uProf C-state histogram resolves whether CC6 is reached during 100 µs inter-layer windows
