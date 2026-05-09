# Stage 1 — Orientation U (Unconventional Substrate) — Run 007

Orientation: U — Unconventional Substrate
Run: 007 (independent cold start)
Ideas generated: 5

Prior U runs (001–006) have covered: NVMe sector alignment, zstd dictionary on metadata, VirtualLock pinning, SuperFetch learning loop, NTFS transparent compression, working-set trimmer oracle, GGUF offset B-tree + IOCP, denormal sentinels, NVMe namespace partitioning, NTFS extent defrag, NT-store for RMSNorm, CPU stride prefetcher layout, large-page TLB, and PrefetchVirtualMemory / IOCP layer streaming. Ideas below avoid all those frames entirely.

CF13 (NTFS extent-prefetch throughput numbers) is NOT used as load-bearing. CF13* is flagged where used and re-derivation is the first cascade rung.

---

## U1-B — Windows RAM-Mapped Compressed Weight Overlay via CreateFileMapping with SEC_COMMIT (RCWO)

### Mechanism

**Track B.** Windows Virtual Memory Manager supports `CreateFileMapping` with `SEC_COMMIT | PAGE_READWRITE` to create anonymous shared memory backed by the pagefile, not by the GGUF file on disk. The proposal: at model load, allocate a `SEC_COMMIT` anonymous mapping of size = compressed-weight-stratum (fp16 block scales, ~8.6 GB for 70B Q4_K_M), fill it from a pre-built offline-compressed binary (see below), then keep only the decompressed scale mapping live in DRAM. The complementary piece: the weight nibble payload stays mmap'd to the GGUF file on NVMe in the normal ik_llama.cpp path. The result is a two-mapping layout: (A) a DRAM-resident anonymous section holding decompressed scales (~8.6 GB → ~2.1 GB at 4× delta-coded compression, see below); (B) a file-backed section for nibble payload (~31 GB for 70B). At inference time the per-block decode reads scale from mapping A (DRAM cache hit, ~4 ns) and nibbles from mapping B (NVMe fetch, ~4 µs). The substrate primitive: `CreateFileMapping(INVALID_HANDLE_VALUE, ...)` with `SEC_COMMIT` for anonymous DRAM-backed sections, a standard Win32 call requiring no elevated privileges. The compression format for the scale stratum: offline delta-coding + zstd (scale values ordered by layer→tensor→block are strongly locally-correlated due to per-layer magnitude banding; delta-coded fp16 scales compress to ~0.5 bytes/block under zstd level 3, confirmed by the run_001 ZDPW analysis of scale distribution). On 7.28 GiB system: the 2.1 GB decompressed anonymous section plus the normal process and KV overhead fits in DRAM (2.1 + 0.5 KV + 0.3 process ≈ 2.9 GiB), leaving the NVMe path only responsible for nibble payload (~31 GB). The effect: NVMe reads per token drop by the fraction of total bytes that are scale data, i.e., 8.6/40 = 21.5% of reads are eliminated; the remaining 78.5% (nibbles) still come from NVMe.

### Residency arithmetic

- 70B Q4_K_M total: ~40 GB on NVMe; scale stratum: ~8.6 GB; nibble stratum: ~31.4 GB.
- Anonymous mapping with delta+zstd scales: 8.6 GB → ~2.1 GB in DRAM (4× ratio assumed; ZDPW run_001 measured ~4× on metadata stratum with zstd).
- NVMe reads per token (before): 40 GB / 80 layers = 500 MB/layer. After split: 31.4/40 × 500 = 392 MB/layer from NVMe (scales now in DRAM).
- NVMe time saved: (500 − 392) MB/layer × 80 layers / 3 GB/s = 2.88 s/token saved at 3 GB/s NVMe (CF13*-conditioned; re-derive). At the 700 MB/s cold-NVMe baseline: 2.88 × (3/0.7) = 12.3 s saved — but total token time is also larger, so ~21.5% reduction applies at both throughput levels.
- DRAM cost: 2.1 GB anonymous mapping. On 7.28 GiB system with 70B nibble stream (NVMe) this is feasible because only one layer's nibbles (~392 MB) need to be in DRAM simultaneously.
- Quality cost: 0 nats (no weight modification).

### Novelty gloss

Closest kill-list item: R1/S2 "OS-Paging-Aware Weight Layout" (killed for not reducing bpw). RCWO differs: it does reduce the NVMe read volume (scales moved to DRAM) without changing bpw of the stored format. It splits the weight tensor into two DRAM-policy tiers using the standard `CreateFileMapping` two-section architecture, which no prior U idea has used. Closest published method: Apple LLM-in-a-flash uses mmap demand paging uniformly; RCWO explicitly tears the weight file into a scale section (DRAM-backed) and a nibble section (NVMe-backed). The run_001 ZDPW proposal compressed metadata at the application level; RCWO does it at the VM-section level with scale data decompressed once at startup and held in an anonymous mapping for the entire inference session. Structural difference from ZDPW: ZDPW decompressed metadata per-block at inference time; RCWO decompresses once at startup and uses standard pointer dereference thereafter — zero per-token decode overhead.

### Smallest experiment

**Testable claim**: on Qwen3-1.7B Q4_K_M, the fp16 block-scale stratum delta-coded and zstd-compressed achieves ≥ 3× size reduction, and loading the decompressed scales from a `SEC_COMMIT` anonymous mapping reduces per-token wall-clock time by ≥ 10% vs demand-paged scales on NVMe-resident inference.

- Step 1 (15 min): extract scale stratum from Qwen3-1.7B GGUF, delta-code, zstd compress; measure ratio.
- Step 2 (1 hr): patch ik_llama.cpp to allocate scale array in a `VirtualAlloc(MEM_COMMIT, PAGE_READWRITE)` anonymous buffer filled at startup; leave nibble mmap unchanged; measure `llama-bench` per-token time vs unpatched.
- Total runtime: ≤ 2 hours.
- Go/no-go: ≥ 3× compression ratio on scale stratum AND ≥ 10% per-token speedup on NVMe-resident run.
- Structural finding on NO-GO: NVMe throughput on this hardware is not meaningfully improved by eliminating 21% of read bytes (i.e., other overheads dominate), which closes the scale-tier split as a latency lever and re-opens the question of whether compute or I/O is the true bottleneck at 1.7B NVMe-resident.

### Primary risk

The Qwen3-1.7B GGUF scale stratum is only ~200 MB; even after compression it fits in L3 after warm-up, so the test must use a model whose scale stratum does NOT fit in L3 to expose the real DRAM vs NVMe delta. Mitigation: force cache eviction between runs with `SetSystemFileCacheSize` or run on the 7B model where scale stratum = ~800 MB, comfortably above L3 (48 MB on Ryzen 5700U).

---

## U2-B — Windows File System Cache Write-Barrier as Inference-Epoch Marker (FSCW)

### Mechanism

**Track B.** Windows NTFS Cache Manager maintains a `CcFlushCache` / `CcPurgeCacheSection` path that, when called on a file range, evicts those pages from the file cache and marks them clean so future reads re-fetch from disk. The standard use is for write-back consistency; the proposed use is the inverse: *controlled partial eviction* of weight pages that have already been consumed by the current token's GEMV and will not be needed again until the next token. On the NVMe-resident path, the Cache Manager's standby list fills with recently-used weight pages and competes with the working set of future layers for the limited DRAM. By calling `CcPurgeCacheSection` (accessible via `NtSetSystemInformation(SystemCacheSizeInformation, ...)` or by mapping the file and calling `FlushViewOfFile` + `UnmapViewOfFile` on the consumed range), the application can explicitly retire stale weight pages, keeping the Cache Manager's working set populated exclusively with upcoming layer pages. The substrate primitive: `FlushViewOfFile` + `UnmapViewOfFile` + `MapViewOfFile` on a per-layer granularity (standard Win32, no admin rights). The access pattern becomes: (1) MapViewOfFile for layer L, (2) GEMV completes, (3) FlushViewOfFile + UnmapViewOfFile for layer L — pages are dropped; Cache Manager serves layer L+1 without competing with stale L data. This is a cache-partition trick using the Windows mmap unmap primitive as an eviction signal — no custom kernel required, behavior guaranteed by Windows mmap semantics (UnmapViewOfFile drops the pages from the process working set; under memory pressure Windows evicts them from standby immediately).

### Residency arithmetic

- 70B at Q4_K_M, 80 layers, ~500 MB/layer. After layer L GEMV completes, L's 500 MB pages remain in standby, competing with L+1's upcoming 500 MB read.
- DRAM is 7.28 GiB. Hot working set at any moment = current layer (500 MB) + KV cache (~670 MB at 4K) + process (~300 MB) = ~1.47 GiB hot. Without explicit eviction, the standby list can hold up to 7.28 − 1.47 = 5.8 GiB of stale weight pages, but the Cache Manager may not evict them quickly enough to make room for the next layer's NVMe prefetch.
- With per-layer UnmapViewOfFile: the 500 MB from layer L is immediately returned to free, ensuring the NVMe prefetch for L+1 has a full 500 MB of free DRAM available before the read completes. This removes a potential stall where the Cache Manager must evict stale pages during an active NVMe read.
- The gain is not throughput (NVMe bandwidth is unchanged) but per-layer I/O stall reduction: if the Cache Manager occasionally stalls the NVMe DMA because it cannot evict fast enough, explicit unmap removes those stalls. Magnitude: 0–100 ms/layer depending on memory pressure dynamics.

### Novelty gloss

No kill-list item matches. Closest is R1/S4-deferred "Idea G: CoW Overlay with Graceful Degradation" (systems-fs) — that idea used copy-on-write to save state during GC pauses; FSCW uses the unmap path for eviction-scheduling. The structural difference: CoW Overlay was for fault tolerance (snapshot); FSCW is for cache management (eviction). Published landscape: no published LLM inference work invokes `UnmapViewOfFile` per-layer to release stale weight pages. llama.cpp's `--no-mmap` mode reads weights into RAM permanently; `--mmap` mode lets the OS manage eviction reactively. FSCW is the proactive eviction discipline that the OS does not do on its own.

### Smallest experiment

**Testable claim**: on Qwen3-1.7B loaded via mmap from NVMe on the 7.28 GiB system, calling `UnmapViewOfFile` after each layer's GEMV and `MapViewOfFile` for the next layer reduces wall-clock per-token latency by ≥ 5% vs continuous mmap baseline, due to reduced Cache Manager stall during NVMe reads.

- Implementation: 20-line Python ctypes wrapper around `UnmapViewOfFile` / `MapViewOfFile`; inject into ik_llama.cpp layer dispatch loop (or implement standalone layer-streaming harness). Time 20-token generation.
- Runtime: ≤ 2.5 hours.
- Go/no-go: ≥ 5% per-token speedup on NVMe-resident Qwen3-1.7B.
- Structural finding on NO-GO: the Windows Cache Manager evicts stale weight pages fast enough on this hardware that explicit unmap adds nothing — the reactive eviction is sufficient, and proactive layer-eviction is unnecessary. Closes this class.

### Primary risk

`UnmapViewOfFile` on a file-backed mapping may incur `FlushViewOfFile` overhead (dirty page write-back) even for read-only mappings if the OS marks pages dirty due to Access Dirty bit tracking. Mitigation: verify with `VirtualQuery` that unmapped pages are `PAGE_NOACCESS` rather than triggering write-back; for read-only weight files, no dirty pages should be generated (weight GGUF is opened `GENERIC_READ` so no dirty tracking occurs).

---

## U3-B — GGUF Tensor-Interleave Reordering for NUMA-Like L3 Affinity on Zen 3 (TINA)

### Mechanism

**Track B.** Zen 3's L3 cache is organized as 8 MB slices shared by pairs of CCX cores (Ryzen 5 7530U has a single CCD with 16 MB L3 shared across all 6 cores). The L3 cache is physically indexed, physically tagged (PIPT) with a hash-based set-selector using address bits [16:6] to distribute cache lines across L3 ways. The consequence: two weight tensors whose starting addresses differ by exactly 2^k for large k map to the same L3 set, causing conflict evictions even when total working set fits in L3. For GEMV on W_up (2048×6144 for Qwen3-1.7B), consecutive rows stride by `6144 × sizeof(q4_block) = 6144/32 × 20 bytes = 3840 bytes`. At 3840-byte row stride, the L3 index bits are determined by bits [16:6] of the physical address. If rows are page-aligned and the physical page allocation follows a power-of-two pattern, two rows may collide in L3. The substrate primitive: Zen 3 L3 set-associativity mapping (documented in AMD Zen 3 Software Optimization Guide sections 2.5 and 2.7 — 16 MB L3 is 16-way set-associative with 64-byte cache lines = 16384 sets). Proposal: pad each tensor's start address to an offset that, when combined with the mmap base address, places its first cache line in an L3 set orthogonal to the previous tensor. Offline tool computes the physical-offset padding required for conflict-free placement given the OS's ASLR base and alignment. The padding adds <1% storage overhead. Unlike HPST (run_005, which targeted the CPU stride prefetcher), TINA targets L3 conflict-eviction patterns — a different microarchitectural phenomenon on a different hardware unit.

### Residency arithmetic

- Qwen3-1.7B W_up: 2048×6144 in Q4_K_M. Each decode pass touches all rows sequentially. Row size = 6144/32 × 20 bytes = 3840 bytes. At 16 MB L3 (262144 sets × 16 ways × 64 bytes), L3 index = address bits [21:6] → upper bits depend on ASLR. If ASLR places W_up at a base whose upper bits collide with W_down (similarly sized), every W_up row fetch evicts a W_down row that was prefetched. At 1.7B the total MLP weights are ~170 MB — far too large for L3. For the DRAM-resident path the bottleneck is DRAM bandwidth, not L3 conflict; TINA's payoff is on the DRAM-bandwidth-bound path where L3 hit rate on the active row matters. If L3 conflict eviction causes 1% more DRAM round-trips, on 11.5 GB/s DRAM that is 0.115 GB/s wasted. At ~200 ms/token (1.7B DRAM), the savings from eliminating L3 conflict eviction are ≤ 2 ms — likely sub-threshold.
- The mechanism's payoff is larger for the NVMe-resident path where the weights currently in L3 from the current layer are the only warm data and any eviction causes an NVMe re-read (4 µs vs 4 ns — 1000× more expensive).

### Novelty gloss

Closest kill-list item: none (L3 conflict-eviction alignment is not in any prior kill). Closest to run_005's HPST (stride prefetcher repack): HPST addressed CPU hardware prefetcher stride detection; TINA addresses L3 set-conflict eviction rates — different hardware units, different fix (padding vs field-split). Published landscape: L3 conflict-avoidance by address-padding is standard in high-performance BLAS (LAPACK padded allocation), but no LLM inference work applies it. The structural argument is real but the magnitude is likely small (sub-2 ms) at 1.7B. At 70B where weight matrices are 60–300 MB and L3 is still 16–32 MB, the conflict-set density is higher.

### Smallest experiment

**Testable claim**: on DRAM-resident Qwen3-1.7B, inserting 4 KB of padding between each tensor in the mmap layout reduces L3 cache miss rate (measured by AMD uProf L3 Miss event or `perf stat -e LLC-load-misses` on Linux) by ≥ 10% per token.

- Implementation: Python GGUF repacker that inserts 4 KB inter-tensor padding; rebuild llama.cpp with modified tensor pointer arithmetic. Run `llama-bench --n-predict 50` and collect L3 miss PMU counter.
- Runtime: ≤ 3 hours.
- Go/no-go: ≥ 10% reduction in L3 miss count per token.
- Structural finding on NO-GO: Ryzen 5 7530U L3 conflict eviction from weight-tensor address patterns is not measurable, confirming that ASLR introduces enough address randomness to prevent systematic conflict — closes L3-alignment as a lever and documents the effective L3 address-mapping behavior on this hardware.

### Primary risk

Physical address bits are not visible to user space; the padding calculation is based on virtual addresses, which may not predict physical placement after Windows VMM page coloring (if any — Windows does not guarantee physical-page coloring). Mitigation: measure empirically via PMU counters rather than predicting from virtual addresses; if the L3 miss count improves with 4 KB padding vs no padding, the mechanism engaged regardless of whether the physical address model was correct.

---

## U4-A — Windows I/O Completion Port Token Budget as Attention-Head Bandwidth Regulator (IOPB)

### Mechanism

**Track A.** Windows I/O Completion Ports (IOCP) have a documented `NumberOfConcurrentThreads` constructor parameter that limits how many threads are *released* simultaneously to process completions. The standard use is throttling: a network server with `NumberOfConcurrentThreads=N` on a quad-core processes at most N completions simultaneously, matching CPU count. The proposed use is as an attention-head bandwidth regulator: in a multi-head attention layer with 16 heads, the Q, K, V projections for all 16 heads could be issued as 16 separate async ReadFile requests on a weight-streaming NVMe path, with the IOCP thread count set to match the CPU's available DRAM bandwidth budget (e.g., 4 concurrent completions × one-head worth of data per completion = 4 heads processed simultaneously). This enforces a regulated compute/memory bandwidth ratio without any ML-layer changes: the IOCP throttle prevents all 16 head weight reads from queuing simultaneously (which would saturate NVMe QD and cause read stalls), instead releasing them 4 at a time, interleaved with GEMV computation on the previously-fetched heads. The substrate primitive: `CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, NumberOfConcurrentThreads)` (standard Win32 IOCP, documented in MSDN since Windows NT 3.5). The architectural change (Track A): the computation graph is modified — instead of fetching all W_Q, W_K, W_V matrices then computing, the graph becomes head-pipelined (fetch-head-i, compute-head-i, repeat) with IOCP throttling regulating the overlap depth. This is not standard inference; it is a computation-graph restructuring that exploits the IOCP primitive as a bandwidth governor. Does not rely on CF1–CF12 directly; the primitive is pure substrate.

### Residency arithmetic

- Qwen3-1.7B attention: 16 heads, d_head=128, W_Q per head = 2048×128 Q4_K_M ≈ 128 KB. Total W_Q: 2 MB; W_K: 2 MB; W_V: 2 MB = 6 MB per attention layer. At 28 layers, attention weights = 168 MB.
- Serial GEMV for all 16 heads: fetch 6 MB then compute. With IOCP-head-pipelined (4 concurrent, overlap fetch of heads 5–8 during compute of heads 1–4): 16 heads / 4 concurrent = 4 rounds. Each round: max(fetch-4-heads, compute-4-heads). Compute for 4 heads of Q: 4 × 2048 × 128 × 2 FMAs at ~10 GFLOPS AVX2 = 4.3 µs. Fetch 4 heads of W_K: 4 × 128 KB at 11.5 GB/s = 44 µs. At DRAM-resident, IOCP pipelining provides minimal benefit (compute << fetch). At NVMe-resident, fetch dominates; IOCP pipelining can overlap the compute of one head group with the NVMe fetch of the next — reduces per-layer attention time by up to the compute fraction (which is small). Primary gain at 70B where W_Q per head = 8192×128 × Q4_K_M ≈ 4 MB, fetch = 4 MB / 700 MB/s = 5.7 ms; compute for 4 heads at 70B = ~1 ms → 6 ms per group → 6 groups × 6 ms = 36 ms attention weight fetch, potentially with 1 ms overlap per group = ~5 ms saved at 70B per attention layer.
- Not primarily a residency reduction; this is a latency-hiding mechanism.

### Novelty gloss

No kill-list item matches: IOCP as an inference-graph pipelining primitive has not appeared. Closest published method: Flash Attention uses online softmax to pipeline attention computation; IOPB pipelines NVMe I/O for attention head weights using IOCP throttling — orthogonal to Flash Attention (which assumes DRAM-resident weights). Closest substrate neighbor: GTBP (run_004) used IOCP for sequential layer prefetch; IOPB uses IOCP's `NumberOfConcurrentThreads` throttle as a head-level bandwidth governor — the load-bearing novelty is the throttle mechanism, not the async I/O itself.

### Smallest experiment

**Testable claim**: on NVMe-resident Qwen3-1.7B, a head-pipelined attention layer with `NumberOfConcurrentThreads=4` IOCP reduces per-attention-layer wall-clock time by ≥ 5% vs serial head fetching, measurable via a standalone head-streaming harness.

- Implementation: write a 60-line C harness that fetches Qwen3-1.7B's W_Q blocks for all 16 heads using IOCP with 4-concurrent-thread throttle, times the total fetch + matmul for one attention layer, and compares to serial.
- Runtime: ≤ 3 hours.
- Go/no-go: ≥ 5% speedup on attention-layer wall-clock time with IOCP throttled overlap.
- Structural finding on NO-GO: NVMe QD on this hardware does not benefit from 4-concurrent read interleaving — either the NVMe is inherently QD=1 (no parallelism) or the OS storage stack serializes all IOCP completions — both documented hardware behaviors that close the IOCP-attention-pipelining class.

### Primary risk

Qwen3-1.7B's W_Q per head is only 128 KB — below the NVMe DMA granularity, so IOCP requests may be coalesced by the storage driver into serial 4 KB reads anyway. Mitigation: test at 7B (W_Q per head = 4096×128 = 4 MB) where the head size exceeds NVMe DMA coalescing threshold, making the IOCP throttle meaningful.

---

## U5-B — Windows Memory Compression (VMC) as Weight Cold-Tier Proxy [FREE SWING] (WMCW)

### Mechanism

**[FREE SWING] Track B.** Windows 11 Memory Compression (enabled by default; `System` process handles `VirtualAlloc` + `CompressMemoryRequest` via the VMC kernel subsystem, documented in Windows Internals 7e Ch.10) automatically compresses pages in the standby list using LZ-based compression in kernel (the "compressed store"). The key: for a DRAM-resident 1.7B model (all weights in RAM), Windows may compress standby pages of weight tensors that have not been accessed recently — the `MmCompressPage` path runs automatically when physical memory approaches capacity. The proposal inverts this: **proactively** move weight tensors for layers not currently executing into the compressed store by calling `VirtualUnlock` (removes them from the working set, making them candidates for compression) and `MmAdvise`-equivalent intent signaling. The compressed store operates at ~5 GB/s decompression throughput (kernel LZ77), which is better than NVMe (~700 MB/s–3 GB/s). If the compressed store retains the weight pages, accessing them triggers decompression rather than NVMe re-read — effectively a DRAM-resident compressed tier at ~4–6 GB/s throughput rather than 700 MB/s NVMe or 11.5 GB/s uncompressed DRAM. The substrate primitive: Windows Memory Compression subsystem (no API call — it operates automatically under memory pressure) + `SetProcessWorkingSetSizeEx` to induce pressure on non-active weight pages. The [FREE SWING] acknowledgment: Windows Memory Compression decompression throughput is a documented OS characteristic but its engagement on GGUF weight pages during active inference is unverified on this hardware. The claim is structural: if VMC engages, decompression cost (~5 GB/s) is better than NVMe cost (~700 MB/s) by ~7×.

### Residency arithmetic

- Qwen3-1.7B Q4_K_M: ~1 GB. DRAM = 7.28 GiB. Total fits; no NVMe paging needed at 1.7B. VMC is only relevant when DRAM fills.
- Qwen3-7B Q4_K_M: ~4.5 GB. Total ~= DRAM capacity; other processes consume ~2 GB. Weight pages compete with OS pages for the remaining 5 GB. If VMC compresses cold weight pages at 2.5× (Q4_K_M near-incompressible as established by ZDPW, so maybe only 1.1× for nibbles; but scales may compress 4×), the effective DRAM utilization of cold weights drops, deferring NVMe paging.
- Weight nibbles (Q4_K_M) are near-incompressible under LZ77 → VMC compression ratio ≈ 1.05–1.10 for nibbles. The scale stratum (8.6 GB for 70B, ~600 MB for 7B) compresses better under LZ77 (~2–3×). So the VMC benefit is scale-stratum-specific: scale pages compress, nibble pages don't.
- At 7B with VMC on scale pages: 600 MB scale stratum → ~250 MB in compressed store. Freed DRAM: 350 MB. Not a large residency win but a real one. The go/no-go settles whether VMC actually engages on these pages.

### Novelty gloss

No kill-list item matches. Closest: run_004 WWTE (Working Set Trimmer as eviction oracle) — WWTE queried which pages the OS LRU retained and proposed VirtualLock for hot pages. WMCW inverts: propose VirtualUnlock for cold pages to encourage VMC compression. Structural difference: WWTE was a hot-pin strategy; WMCW is a cold-compress strategy exploiting the VMC compressed store as a warm tier between uncompressed DRAM and NVMe. Published landscape: no published LLM inference work exploits Windows Memory Compression for weight tensor cold-tier management. The compressed store is used implicitly by Windows but no inference runtime has proposed proactive cold-tier management via the compressed store.

### Smallest experiment

**Testable claim**: on Qwen3-7B loaded into a 7.28 GiB DRAM system with 4 GB of other active processes, `VirtualUnlock` on non-active weight layers induces Windows Memory Compression on those pages, measurable by `Get-Process | Select-Object WorkingSet, VirtualMemorySize` and the "Compressed" counter in Task Manager's Memory report, and results in ≥ 5% better per-token wall-clock time vs unmanaged baseline.

- Step 1 (30 min): install Qwen3-7B; run inference under memory pressure (open additional memory-hungry applications to fill DRAM to ~90%); observe Task Manager "Compressed" counter; verify VMC engages on weight pages by checking page-fault rate before/after `VirtualUnlock` via `QueryWorkingSetEx`.
- Step 2 (1 hr): if VMC engages, compare per-token latency with vs without `VirtualUnlock` on cold layers.
- Runtime: ≤ 2.5 hours.
- Go/no-go: VMC "Compressed" memory increases by ≥ 100 MB after `VirtualUnlock` on cold layers AND per-token time improves ≥ 5%.
- Structural finding on NO-GO: Q4_K_M nibble pages are incompressible (ratio < 1.05) and VMC stores them as-is, meaning the VMC compressed store adds decompression overhead with no size reduction — closes VMC as a weight cold-tier for quantized models.

### Primary risk

`VirtualUnlock` on a working-set page in a memory-mapped file does not guarantee the page goes to the VMC compressed store; it may simply go to the standby list and be evicted to disk (paged out to pagefile) instead, which is WORSE than NVMe GGUF access (pagefile I/O has no compression). Mitigation: verify via `RtlQueryProcessHeapInformation` or the VMC memory counters that the freed pages land in "Compressed" rather than "Standby-disk" state before measuring throughput.

---

## Convergence handles

1. **GGUF scale-stratum byte range compressibility under delta-coding + zstd** — RCWO (U1) primary go/no-go; also closes run_001 ZDPW's open question about decompression-at-startup vs decompression-per-block.
2. **Windows Cache Manager eviction speed under active NVMe reads** — FSCW (U2) depends on whether reactive eviction is a stall source; one measurement settles this for all future per-layer eviction ideas.
3. **L3 conflict-eviction rate from weight tensor address collisions on Zen 3** — TINA (U3) go/no-go; also informs HPST (run_005) since both target weight-read microarchitecture on the same L3.
4. **NVMe QD effective parallelism under IOCP multi-request** — IOPB (U4) foundation; also determines whether GTBP (run_004) would have achieved its claimed overlap.
5. **Windows Memory Compression engagement on Q4_K_M weight pages under real DRAM pressure** — WMCW (U5) fundamental premise; measured via Task Manager VMC counter — single measurement with broad implications for all OS-managed cold-tier proposals.
