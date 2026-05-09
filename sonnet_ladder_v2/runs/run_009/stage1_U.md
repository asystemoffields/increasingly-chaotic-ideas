# Stage 1 — Orientation U (Unconventional Substrate) — Run 009

Orientation: U — Unconventional Substrate
Run: 009 (independent cold start)
Ideas generated: 5

Prior-run avoidance: Runs 001-005 covered NVMe sector alignment, NTFS compression, VirtualLock, branch-predictor dispatch, BLAKE3 dedup, async IOCP prefetch, SuperFetch, working-set trimmer oracle, denormal sentinels, NVMe namespace partition, large-page TLB, NUMA/CCX, NT-store drain, stride prefetcher. None of those frames reappear below. CF13 UNVERIFIED — any idea touching NVMe prefetch numbers must re-derive in cascade rung 1.

---

## U1-B — ETW File-I/O Kernel Trace as Zero-Overhead Weight-Access Hotness Oracle (ETWO)

### Mechanism

**Track B.** Windows ETW (Event Tracing for Windows) emits a kernel event for every file read via the `Microsoft-Windows-Kernel-FileIO` provider (`{EDD08927-9CC4-4E65-B970-C2560FB5C289}`, event ID 0x0A = ReadFile). These events fire in kernel space, are logged to a circular in-RAM buffer, and can be consumed by a user-mode controller via `OpenTrace` + `ProcessTrace` with zero modification to the traced process — no inference code change, no instrumentation hooks. Proposal: run a short (e.g., 20-token) calibration inference pass under ETW collection on the GGUF mmap path. The trace records which byte offsets of the GGUF file were read, in what order, and at what wall-clock times. Post-process the trace offline to build a per-tensor, per-layer read-frequency histogram. Tensors read on every token (hot) vs tensors read only on specific prompt types (cold) are now labeled from hardware-observed behavior, not from model-internal statistics. The histogram feeds two downstream uses: (a) static GGUF repack so hot tensors are at file head for Cache Manager sequential prefetch (no inference code change), or (b) a VirtualLock pin list for the hot tensors only (requires the inference process's `--mlock` flag, already present in llama.cpp). The substrate primitives: `StartTrace` / `EnableTraceEx2` / `ProcessTrace` APIs (advapi32.dll, fully documented in MSDN, available since Windows Vista, no admin rights for kernel-file-io if the process owns its own session). The key insight: ETW already observes LLM inference at zero marginal cost; the access histogram is a free measurement, not a model-derived one.

**CF anchor**: not CF-grounded (substrate-orthogonal). No kill-list collision. Does not use CF13*.

### Residency arithmetic

Qwen3-1.7B at Q4_K_M, NVMe-resident: ~1 GB, 28 layers. The ETW trace for a 20-token run produces ~56 kernel ReadFile events (2 per layer per token pass) totaling ~1 KB of trace data. Post-processing time: ~1 s in Python. Output: a ranked list of tensor offsets by read frequency. If the top 20% of reads concentrate on 10% of tensors (a common locality pattern), VirtualLock'ing 100 MB covers the hot zone and fits in 7.28 GiB with ~7.2 GiB headroom for the 1.7B case. At 70B NVMe-resident: same 20% hot / 10% mass claim implies ~4 GB hot zone — within 7.28 GiB budget if the cold 36 GB stays on NVMe. Hot-zone DRAM reads at 11.5 GB/s vs cold NVMe at ~700 MB/s: for 70B, if 10% of tensors are hot and account for 40% of reads, average effective bandwidth = 0.4 × 11.5 + 0.6 × 0.7 ≈ 5.0 GB/s — ~7× speedup on the hot fraction, ~2.5× on the cold fraction, net ~3× throughput at 70B. The 10%/40% locality claim is the go/no-go pivot; the ETW experiment itself measures this directly.

### Novelty gloss

Closest kill-list item: run_004 U3-B WWTE (working-set trimmer as eviction oracle). Structural difference: WWTE queries the OS LRU state *after* inference runs; ETWO captures the actual byte-offset read stream *during* inference, producing a tensor-resolution (not page-resolution) histogram. WWTE is page-granular and passive; ETWO is tensor-granular and active (an explicit trace session). The distinction matters: WWTE cannot tell you *which tensor* is hot, only which 4 KB pages are retained; ETWO gives exact byte-range read counts mapped back to GGUF tensor names. Closest published method: Apple LLM-in-a-Flash's activation-statistics-based prefetch clustering (arXiv:2312.11514). Structural difference: that method requires running the full model with instrumented activations; ETWO requires zero model instrumentation and captures the actual OS I/O behavior. No published LLM inference paper uses ETW file I/O traces as the hotness oracle.

### Smallest experiment

**Testable claim**: a 20-token calibration run of Qwen3-1.7B under ETW `Microsoft-Windows-Kernel-FileIO` trace identifies ≥3 tensors that account for ≥30% of total ReadFile events, with the histogram stable across 3 independent calibration runs (Jaccard of top-10 tensors ≥ 0.80).

**Runtime**: ~3 hours total (ETW controller script in Python using `pywin32` or `subprocess` with `logman`; post-processing; stability check across 3 runs).

**Go/no-go threshold**: top-10 tensors by read count cover ≥30% of total reads AND cross-run Jaccard ≥ 0.80.

**Structural finding on NO-GO**: GGUF mmap access under llama.cpp produces a flat read histogram (all tensors accessed at equal frequency), meaning there is no exploitable hot/cold structure at the file-read level — this closes the ETW-oracle class and narrows the access-locality question to within-layer activation sparsity (a different axis).

### Primary risk

`mmap`-based weight access in llama.cpp may not emit `ReadFile` ETW events — the kernel emits `PageFault` events for mmap reads, not `ReadFile` events. The `Microsoft-Windows-Kernel-MemoryManager` provider (`{D1D93EF7-E1F2-4F45-9943-03D245FE6C00}`) emits HardPageFault events that include the faulting address, allowing the same tensor-granular histogram from page-fault traces instead of ReadFile traces.

**Mitigation**: if ReadFile events are absent (mmap path confirmed), switch ETW provider to `Microsoft-Windows-Kernel-MemoryManager` and record hard page-fault addresses instead. The experiment structure is identical; the provider string changes.

---

## U2-B — NTFS Alternate Data Stream Tier Sidecar for Mixed-Precision GGUF (ADST)

### Mechanism

**Track B.** NTFS supports Alternate Data Streams (ADS): named byte streams attached to a file inode, accessed via `filename:streamname` syntax (e.g., `model.gguf:tier_map`). An ADS stream is stored in the MFT entry or in a separate cluster run, but is part of the same inode — no separate file, no directory entry. The substrate primitive: `CreateFile("model.gguf:tier_map", ...)` (documented Win32 API, available since Windows NT 4.0; no admin rights, no kernel modification). Proposal: for a mixed-precision inference scheme (e.g., RAOK-style 3-tier: FP16 / INT8 / INT4 per block), store the per-block tier assignments as a compact binary array in a `:tier_map` ADS stream rather than embedding them in the GGUF tensor data or maintaining a separate sidecar file. The ADS stream is co-resident with the GGUF inode — the kernel's MFT cache keeps it warm alongside the file's $INDEX entries. For Qwen3-1.7B at Q4_K_M block granularity: ~48K blocks × 2 bits/block = 12 KB tier map. A 12 KB ADS stream fits entirely within the NTFS MFT inline attribute limit (968 bytes per attribute run in base MFT record; overflow goes to extension records). The inference runtime reads the `:tier_map` stream once at load time (12 KB read, ~4 µs at NVMe speed), caches it in RAM, and uses it to dispatch block reads to the appropriate decode path. Secondary benefit: the `:tier_map` stream can be NTFS-compressed independently (via `compact /c model.gguf:tier_map`) achieving near-2× compression on the tier-assignment bit array (runs of identical tier assignments are common in layer heads), dropping it from 12 KB to ~7 KB — trivial in absolute terms but establishes the ADS-as-metadata-sidecar pattern for arbitrary per-block auxiliary data.

**CF anchor**: loosely RAOK / PDAP-compatible (per-block tier metadata); no specific CF number required. Does not use CF13*.

### Residency arithmetic

Direct residency impact: negligible (12 KB ADS vs 1 GB model). The payoff is architectural: the ADS sidecar pattern enables fine-grained per-block metadata without modifying the GGUF format. At 70B (Q4_K_M, ~2.2B blocks): tier map = 2.2B × 2 bits = 550 MB — no longer fits in MFT inline, becomes a regular ADS cluster run; 550 MB ADS reads at NVMe speed = 0.18 s one-time load. At this scale, NTFS compression on the tier map (achieving ~2× on run-length-coded tier assignments) reduces the one-time load to 0.09 s. The real value at 70B is not the size but the decoupling: the tier map can be updated (re-calibrated) without touching the 40 GB weight file, rewriting only the 550 MB ADS stream.

### Novelty gloss

No kill-list item covers ADS-based metadata sidecar. Closest: run_004 U5-B DFSQ (denormal sentinel in-band tier marking). Structural difference: DFSQ encodes tier in the scale float value (in-band, read during decode), adding a comparison on the hot decode path; ADST stores tier in a separate ADS stream (out-of-band, read once at load), adding zero overhead on the hot decode path. The two mechanisms are complementary but structurally distinct. Published landscape: GGUF metadata stores per-tensor quantization type in the header, not per-block; no published inference framework uses NTFS ADS for quantization metadata. The nearest concept is PostgreSQL TOAST (out-of-line storage for large attributes) — ADST applies the out-of-line metadata pattern from DB engines to the inference file format.

### Smallest experiment

**Testable claim**: `CreateFile("qwen3-1.7b.gguf:tier_map", GENERIC_WRITE, ...)` succeeds on the Ryzen 5 7530U Windows 11 system's NTFS volume, a 12 KB binary write to the ADS stream succeeds, and a subsequent `CreateFile("qwen3-1.7b.gguf:tier_map", GENERIC_READ, ...)` reads back the same bytes — confirming ADS is available and readable on this filesystem with zero inference code change.

**Runtime**: ~15 minutes (20-line Python test using `open("model.gguf:tier_map", "wb")`).

**Go/no-go threshold**: ADS read/write succeeds AND ADS survives a file copy (`copy model.gguf model2.gguf` — NTFS preserves ADS in copies within the same volume; copies across volumes strip ADS, a known behavior to document).

**Structural finding on NO-GO**: ADS is unavailable or stripped on this NTFS configuration — closes the NTFS-ADS-as-metadata class; forces fallback to a `.tiers` sidecar file (same content, less elegant, cross-volume safe).

### Primary risk

File copying or backup tools may silently strip ADS streams (robocopy preserves ADS; xcopy does not). If the GGUF is distributed without ADS, the tier map is lost. Mitigation: embed a SHA-256 hash of the GGUF in the ADS stream; at load time, verify the hash matches the GGUF content; if ADS is missing, fall back to regenerating the tier map from calibration (tier map is derivable from model weights + calibration data, not hardcoded).

---

## U3-B — SetFileInformationByHandle IoPriority Elevation for GGUF Handle (FIOP)

### Mechanism

**Track B.** Windows I/O Manager assigns an I/O priority to each I/O request, ranging from `IoPriorityVeryLow` (background, throttled) to `IoPriorityVeryHigh` (real-time, not throttled). The inference process's weight-file reads compete for NVMe queue slots against: (a) Windows Update background downloads, (b) Antivirus live-scan I/O, (c) SysMain prefetch I/O, (d) llama.cpp's own KV cache writes. By default, all I/O from a user-mode process runs at `IoPriorityNormal`. Windows 11 exposes per-file-handle I/O priority via `SetFileInformationByHandle` with `FileIoPriorityHintInfo` (documented in MSDN; the `IO_PRIORITY_HINT` enum includes `IoPriorityHintVeryHigh`). Elevating the GGUF file handle to `IoPriorityHintVeryHigh` causes the Storage port driver to drain the NVMe submission queue of all GGUF read requests before servicing lower-priority requests from competing processes. For NVMe inference, the bottleneck is NVMe queue latency at QD=1 (one outstanding request at a time from llama.cpp). Background I/O from Antivirus or Windows Update at `IoPriorityNormal` can preempt a GGUF read request for 50–200 ms (documented I/O priority contention latency in Windows storage stack internals). At `IoPriorityHintVeryHigh`, GGUF reads are never preempted by these background tasks. Substrate primitive: `SetFileInformationByHandle(hGguf, FileIoPriorityHintInfo, &hint, sizeof(hint))` — one Win32 call, no admin rights, no kernel modification, available since Windows Vista.

**CF anchor**: orthogonal to all CFs. Does not use CF13*.

### Residency arithmetic

Qwen3-70B NVMe-resident at Q4_K_M: 40 GB, ~13 s/token nominal. If background I/O contention causes 200 ms stalls on 5% of layer reads (conservative estimate for a Windows 11 system with AV live-scan enabled): 80 layers × 0.05 × 0.2 s = 0.8 s/token stall overhead. With `IoPriorityHintVeryHigh`: background stalls drop to near zero — saving ~0.8 s/token = ~6% throughput improvement. The mechanism is more impactful on p95/p99 latency than on mean throughput: stalls are stochastic (AV scan triggers on writes, not reads, and is sporadic), so the mean improvement is small but variance drops sharply. For interactive use, p95 latency improvement may be more valuable than mean tok/s.

### Novelty gloss

No kill-list item covers per-handle I/O priority elevation. Closest run-prior: run_004 U6-B NNPA (NVMe namespace partition to avoid GC contention). Structural difference: NNPA addresses NVMe controller-internal GC stalls (firmware-level); FIOP addresses OS-level I/O scheduler preemption (OS storage stack level) — orthogonal mechanisms at different layers of the stack. Published landscape: no published LLM inference paper uses `SetFileInformationByHandle` I/O priority; the `--mlock` flag in llama.cpp addresses residency (keeping weights in RAM) but not I/O scheduler priority for the weight reads. The HPC literature uses real-time I/O priorities for latency-sensitive storage, but this application to LLM inference is novel.

### Smallest experiment

**Testable claim**: on this Ryzen 5 7530U Windows 11 system, running Qwen3-1.7B inference simultaneously with a background I/O stress (sequential write of 500 MB at `IoPriorityNormal`) produces measurably higher p95 per-token latency than the same run with GGUF handle elevated to `IoPriorityHintVeryHigh`.

**Runtime**: ~2 hours (modify llama.cpp to add one `SetFileInformationByHandle` call + write a background I/O stress script + measure 100-token run latency distribution in both conditions).

**Go/no-go threshold**: p95 per-token latency with background I/O stress ≥ 1.20× baseline (background-I/O-free); AND `IoPriorityHintVeryHigh` reduces p95 latency to within 1.05× baseline (≥ 75% stall elimination).

**Structural finding on NO-GO**: Windows 11 storage stack does not honor `IoPriorityHintVeryHigh` from user-mode processes for NVMe I/O, or the NVMe driver (StorNVMe.sys) does not propagate I/O priority to the NVMe submission queue — documents the storage-stack priority-honoring behavior on this hardware.

### Primary risk

Windows 11 Home may not honor `IoPriorityHintVeryHigh` for user processes without a Multimedia Class Scheduler registration (`AvSetMmThreadCharacteristics`). Mitigation: also call `AvSetMmThreadCharacteristics("Games", &taskIndex)` on the inference thread before issuing the I/O priority hint; the multimedia class scheduler elevates the thread's I/O class alongside its CPU priority.

---

## U4-B — GGUF Hot-Layer Partial Residency via OS RAM Disk (OSRD)

### Mechanism

**Track B.** Windows supports RAM disks via built-in ImDisk kernel driver (available in Windows 11 via `imdisk.exe`, an in-box driver for virtual disk devices since Windows Vista) or the third-party OSFMount (a standard user-mode disk driver, no kernel modification). A RAM disk appears as a block device (`\\.\E:`) backed by a contiguous physical RAM allocation; files stored on it are read by the inference process via ordinary NTFS semantics (mmap, ReadFile) but the backing storage is RAM, not NVMe. The OS kernel routes reads to the virtual block device at DRAM bandwidth (~11.5 GB/s on this system) vs NVMe (~700 MB/s random). Proposal: at startup, create a 2 GB RAM disk consuming DRAM headroom, copy the first 14 layers of the Qwen3-8B GGUF to the RAM disk, and open those layers' mmap from the RAM disk path while opening the remaining layers from NVMe. The inference code sees two mmap handles, both presenting the same file API — no inference logic change. The hot layers (early attention + MLP) run at DRAM bandwidth; the cold layers (deep MLP, often less frequently accessed in prefill-dominant workloads) stream from NVMe. The substrate primitive: `imdisk -a -s 2g -m E: -o rem` (creates a RAM disk using the ImDisk driver, standard Windows utility). Not a custom kernel; ImDisk is an in-box Windows kernel module (`imdisk.sys`).

**CF anchor**: v2-CF1 (K-dependent outlier dynamicity is layer-uniform) confirms no layer is structurally safe to drop in quality; OSRD does not drop any layer — it moves layers between storage tiers, zero quality cost. Does not use CF13*.

### Residency arithmetic

Qwen3-8B at IQ4_XS: ~4.7 GB total. RAM budget: 7.28 GiB available; 8B model needs at minimum ~1 GB for KV cache and inference buffers. Available for RAM disk: 7.28 − 1 (KV+runtime) − 4.7 (model, if all-NVMe) ≈ 1.58 GB. Allocate 1.5 GB RAM disk: copies layers 0–8 (9 × ~168 MB ≈ 1.5 GB). Those 9 layers read at 11.5 GB/s: 1.5 GB / 11.5 GB/s = 0.13 s latency if cold. Remaining 19 layers on NVMe: 3.2 GB at 700 MB/s = 4.6 s/token pass. Without RAM disk: all 28 layers at 700 MB/s = 4.7/0.7 = 6.7 s/token. With 9 layers on RAM disk: (0.13 + 4.6) = 4.73 s/token. Savings: ~30%. If overlap between RAM-disk read and NVMe read is achievable (both issued in parallel): further reduction possible. At 70B: 40 GB total; RAM disk can hold at most 3-4 GB of hot layers (leaving headroom for KV cache); ~7–10% of layers at DRAM speed, modest absolute gain but confirms the mechanism.

### Novelty gloss

No kill-list item covers RAM disk partial residency. Closest: run_001 U3-A VLAP (VirtualLock for attention pinning). Structural difference: VLAP pins pages in the process's working set via VirtualLock (a virtual memory mechanism, constrained by working set limits); OSRD copies layers to a RAM disk (a virtual block device), bypassing working-set constraints entirely — the OS allocates the RAM disk as a device, not as process private memory, so it is not subject to per-process VirtualLock limits. The RAM disk is accessible to any process; VLAP is process-private. Published landscape: no published LLM inference paper uses a RAM disk for partial layer residency (the standard approach is OS page cache, which is subject to eviction; RAM disk is not evicted). The nearest concept is Apple M-series unified memory where the OS manages CPU/GPU transfers, but RAM disk is a fully general Windows primitive.

### Smallest experiment

**Testable claim**: 9 layers of Qwen3-1.7B mmap'd from an ImDisk RAM disk deliver ≥ 10× lower layer-read latency than the same layers mmap'd from NVMe, measured by wall-clock time for the 9-layer forward pass.

**Runtime**: ~2 hours (install ImDisk if not present, copy 9 layers, patch llama.cpp to open from two paths, run llama-bench with per-layer timing).

**Go/no-go threshold**: RAM-disk layer reads ≥ 10× faster than NVMe layer reads (11.5 / 0.7 ≈ 16× theoretical; expect ≥ 10× measured due to overhead).

**Structural finding on NO-GO**: mmap from a RAM disk does not achieve DRAM bandwidth via the Windows virtual disk driver (ImDisk driver adds an intermediate copy through the block device stack, adding latency) — documents the actual throughput of the Windows RAM disk mmap path vs direct VirtualLock, helping future proposals choose between the two mechanisms.

### Primary risk

ImDisk requires installation (not in-box on all Windows 11 configurations). Mitigation: OSFMount is a zero-install portable alternative (runs as user-mode service via `VirtualBox VBoxDrv.sys` kernel driver, already present if the Ryzen system runs any VM software); alternatively, the Windows Subystem for Linux 2 (WSL2) virtual disk driver provides an analogous mechanism on Linux-path inference.

---

## U5-A — Windows VEH Guard-Page Demand-Tier Promotion for GGUF Mixed-Precision (VGTP) [FREE SWING]

**[FREE SWING] Track A (arch/residency hybrid — changes how the inference computation accesses weight tiers at runtime).** Windows supports guard pages via `VirtualAlloc` with `PAGE_GUARD` modifier. When a guard page is touched, the hardware raises a STATUS_GUARD_PAGE_VIOLATION exception before the access completes; a Vectored Exception Handler (VEH), registered via `AddVectoredExceptionHandler` (Win32, documented MSDN, available since Windows XP), intercepts the exception before any frame unwind. The guard page is automatically demoted to its underlying access rights after the first touch. Proposal: at model load, partition the mmap'd GGUF into tier regions. The "cold" tier (INT4 blocks, lowest quality) is initially mapped without guard pages — reads proceed normally at INT4 precision. A "promotion list" is maintained: blocks in the cold tier that, based on ETW or working-set history (using ideas U1/U3 as predecessors), are candidates for promotion to INT8 or FP16 at inference time. Before the inference loop begins, these candidate blocks' mmap pages are remapped with `PAGE_GUARD` via `VirtualProtect`. During the forward pass, when the inference thread first reads a guarded page, the VEH fires, synchronously dequantizes the page's weights into a higher-precision buffer already allocated in DRAM, removes the guard (`VirtualProtect` back to `PAGE_READONLY`), and resumes execution. The result: blocks that were predicted to be hot are transparently upgraded to higher precision on the first touch with zero branching cost in the hot decode path — the branch cost is paid at the page-fault handler, which runs once per promoted block per inference session. The substrate primitives: `VirtualProtect` with `PAGE_GUARD` + `AddVectoredExceptionHandler` — both standard Win32, no kernel modification, no admin rights.

**Residency arithmetic.** Qwen3-1.7B, 48K blocks: promote top 10% (4800 blocks × 32 bytes/block INT8 = 150 KB) vs keep at INT4 (90% = 48K × 16 bytes = 768 KB). Promotion overhead: 4800 page-fault exceptions at ~10 µs/VEH invocation (Windows VEH round-trip time on Zen 3, documented benchmark ~5-15 µs) = 48 ms one-time per session. After first-token: all promotions complete, subsequent tokens run at INT8 on the hot blocks with zero extra overhead. Quality impact: +dNLL on the promoted blocks = 0 (upgrading precision, not degrading). The mechanism is novel in the direction of upgrade rather than compression — it enables a quality floor without increasing average residency. At 70B: 10% promotion = 2.2B × 0.1 = 220M blocks; 220M × 10 µs VEH = 2200 s for the first token only — catastrophically slow. The mechanism is only viable at 1.7B–7B scale where 10% promotion is ≤50K blocks.

**Novelty gloss.** No kill-list item covers VEH-based demand tier promotion. Closest: run_004 U3-B WWTE (working-set trimmer as eviction oracle) — that idea identifies hot pages passively; VGTP uses the guard-page mechanism to interpose on the hot-page access and perform tier upgrade at that moment. WWTE is read-only observation; VGTP is read-triggered action. Published landscape: VEH guard-page interposition is used in memory forensics (hooking heap allocations) and copy-on-write VM implementations; no published LLM inference paper uses it for per-block precision promotion. The free-swing justification: the mechanism is genuinely orthogonal to any existing ML idea and exploits a CPU/OS primitive (the hardware page-fault → VEH chain) that ML inference has never invoked. The scale limitation (sub-7B only) makes it a niche rather than a flagship, hence free-swing rather than primary.

**Smallest experiment.** Testable claim: `AddVectoredExceptionHandler` + `VirtualProtect(PAGE_GUARD)` on 1000 pages of Qwen3-1.7B mmap region fires VEH for each page on first read access within 200 µs per page on Zen 3 (target: total 1000-page first-touch cost ≤ 200 ms). Runtime: ~1 hour (standalone C harness, 50 lines). Go/no-go: 1000-page VEH chain completes in ≤ 200 ms AND VEH correctly restores page access rights (single-shot per page). Structural finding on NO-GO: Windows VEH invocation from page-fault path on mmap'd NVMe-backed pages takes >200 µs/page (hardware contention or TLB shootdown overhead) — closes VEH-based per-block interposition and documents the VEH latency floor on this hardware.

**Primary risk.** mmap pages backed by NVMe may not be touchable in VEH context (the page may be not-present, requiring a kernel I/O to fulfill before VEH runs, adding the full NVMe latency to every VEH invocation). Mitigation: verify by examining whether the VEH fires on a page-present guard page vs a page-absent (NVMe-backed) page — the former is a hardware-only exception; the latter involves the Memory Manager's I/O completion path. If the latter adds >10 ms, restrict promotion to RAM-resident pages only (copy-in the block before arming the guard).

---

## Convergence handles

1. **ETW kernel file-I/O provider engagement on mmap-based GGUF access** — ETWO's go/no-go pivot; also gates any future kernel-trace-based access histogram proposal (mmap emits page-fault events, not ReadFile events).
2. **Windows storage stack I/O priority propagation from FileIoPriorityHintInfo to NVMe submission queue** — FIOP mechanism; one measurement closes or opens the entire OS-I/O-priority class for inference.
3. **ImDisk / OSFMount RAM disk mmap throughput vs VirtualLock throughput** — OSRD vs VLAP structural comparison; the first measurement on this hardware characterizes both paths simultaneously.
4. **NTFS ADS persistence across copy operations (robocopy vs xcopy vs cloud sync)** — ADST distribution risk; one copy test on this filesystem closes the ADS-stripping question.
5. **VEH invocation latency on mmap'd NVMe-backed pages** — VGTP free-swing go/no-go; also characterizes the Windows page-fault handler latency floor, which is a primitive several other substrate ideas (OSRD, ETWO) touch.
