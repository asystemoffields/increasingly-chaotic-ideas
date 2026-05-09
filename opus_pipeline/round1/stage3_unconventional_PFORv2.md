# Stage 3 Unconventional v2 — PFOR + NEXTENT + ZIPFLOCK + iGPU Overlap

## Path A — refine, significant restructure.

CF9 latency risk real but doesn't kill the bet. Changes which mechanism carries the contribution. WSTLO loses zero-overhead lead. PFOR survives and gets stronger. Unconventional spine — *the OS is a better predictor than us* — survives by relocating its predictor from working-set trimmer (only works if faults cheap) to **NTFS extents + Cache Manager read-ahead** (work even when individual faults expensive, cost amortized across extent).

**New lead**: information-theoretically optimal layout makes OS demand-paging efficient even when OS demand-paging is slow. Each fault drags 4 KB regardless of how slow — PFOR makes those 4 KB worth more than any other 4 KB.

## Mechanism (revised)

### Primary: PFOR — sharpened in claim
Build weight co-access graph G=(V,E) where V = 4 KB-sized weight tiles, E weighted by Jaccard co-occurrence within forward-pass token. Permutation π via **spectral ordering (Fiedler vector of normalized Laplacian)** with RCM as fallback. Materialize GGUF in π order, **aligned to NTFS extent boundaries (1-2 MB)**.

Generalizes Apple's two-tensor "row-column bundling" to globally-optimal graph permutation at page granularity. Different problem class than NeurIPS 2022 Coleman et al. (vector index graphs).

### Secondary: NEXTENT — NEW, replaces WSTLO as lead
NTFS allocates files in **extents** (contiguous runs of clusters). Windows Cache Manager's read-ahead detects sequential/strided patterns and **prefetches entire extents** when one page in extent faulted. Extent size on defragmented file = 256 KB-16 MB.

**Align PFOR-permuted co-access groups to NTFS extent boundaries.** When token T touches one weight in extent E, Cache Manager pulls entire extent (~1 MB = 256 pages) at near-sequential NVMe throughput (3-7 GB/s), not random-4K throughput. Sequential 1 MB on Samsung 990 Pro ~150 μs. Random 4 KB on Windows mmap ~150 μs. **Extent-aligned read pulls 256× more useful bytes per wall clock.**

### Tertiary: WSTLO — demoted to corollary
Working-set trimmer's clock-LRU still eviction oracle, but no longer load-time mechanism. Eviction at 3.75-15s clock intervals fine; **load governed by extent prefetch, not page fault.** CF9 trim-mid-forward-pass risk reduced by (a) extent-granular working set, (b) next section's VirtualLock removes hot tail from trimmer.

### Quaternary: ZIPFLOCK — NEW
**VirtualLock top-30% Zipf-ranked weight pages** at process start — forcibly resident, never fault, never trim. Long 70% tail goes through PFOR/NEXTENT/WSTLO path. Bypasses demand-paging for majority of access by token-frequency-weighted volume.

VirtualLock requires SeLockMemoryPrivilege (administrative on Windows but available). Pinned working set quota raised via SetProcessWorkingSetSize.

### Quintary: NQRA — retained, downgraded
NVMe queue-depth bursting still useful for explicit-prefetch path under realistic-E0 branch (PrefetchVirtualMemory issues batched async faults at QD=8-16). No longer primary throughput claim.

### Hexary: Vega 7 iGPU overlap — NEW load-bearing role
CPU waits on fault ≠ GPU waits on fault. Vega 7 has zero-copy access via UMA. **While CPU blocked on next page-fault round trip, dispatch already-resident matmul work to iGPU.** Critical-path compute proceeds during I/O blocking. Conventional async-overlap dressed up — but **using iGPU's separate execution context to keep arithmetic intensity high during page-fault stalls** without explicit threading.

## Hardware/OS primitives used

| Mechanism | Primitive | API/structure |
|---|---|---|
| PFOR | NVMe + ext4/NTFS layout | mmap MAP_PRIVATE, fallocate-extent control |
| NEXTENT | NTFS extents + Cache Manager read-ahead | FSCTL_QUERY_FILE_LAYOUT, FSCTL_MOVE_FILE, GetFileInformationByHandleEx |
| WSTLO | Memory Manager working-set trimmer | NtSetSystemInformation, PsSetWorkingSetEvictionPolicy |
| ZIPFLOCK | Pinned non-pageable pages | VirtualLock + SeLockMemoryPrivilege + SetProcessWorkingSetSize |
| NQRA | NVMe SQ/CQ queue depth | PrefetchVirtualMemory (Win8.1+), ReadFileEx async |
| iGPU overlap | Vega 7 UMA via OpenCL/Vulkan | clCreateBufferWithProperties CL_MEM_USE_HOST_PTR, VK_EXT_external_memory_host |

Every mechanism resolves to documented OS API or hardware feature. No simulator-only claims.

## Falsification (E0-gated cascade)

**E0 (HARD PREREQUISITE, 1 hr): Windows 11 NVMe mmap demand-page round-trip benchmark.** 8 GiB sparse file, mmap, randomly touch 100k 4 KB pages cold (drop standby cache via EmptyStandbyList), measure per-fault latency at thread counts 1, 4, 8.

**Branch decision:**
- **E0a (optimistic, ≤30 μs/fault)**: Original PFOR + WSTLO cascade proceeds. Single-fault latency low enough that 50-page-per-token unique fault count fits in 1.5 ms.
- **E0b (realistic, 100-300 μs/fault)**: Pivot to PFOR + NEXTENT + ZIPFLOCK + iGPU overlap cascade. WSTLO becomes corollary.

**Conditional research plan**, not single linear cascade.

### Under E0b (realistic branch):

**E1' NTFS extent prefetch latency.** Defragment file to large extents (1-8 MB), measure first-page-of-extent fault vs same-extent subsequent-page latency. Hypothesis: first triggers full extent read-ahead; subsequent same-extent hit standby cache at ~5 μs.

**E2' PFOR vs sequential layout.** Build co-access graph from one model + 100 tokens of WikiText. Compare default sequential vs PFOR-spectral GGUF. Measure unique-extents-touched per token. Target: ≥4× reduction in extent count per token.

**E3' ZIPFLOCK residency.** Identify top 30% by access frequency, VirtualLock, measure per-token fault count for non-locked weights. Target: ≥70% of access volume served from locked pool.

**E4' iGPU overlap.** Decompose forward into (matmul-on-resident-tile, fault-blocking-load-next-tile) pairs. Measure CPU stall fraction with vs without Vulkan-dispatched iGPU concurrent matmul. Target: ≥40% stall coverage.

**E5 Composite tok/s.** End-to-end on 70B-class (Qwen3-72B IQ3_XXS or similar fitting ~30 GB on disk, ~7 GB resident). Target: **≥1.5 tok/s under E0b cascade, ≥2.5 tok/s under E0a cascade.**

### Pre-registered failures
1. E0 > 300 μs/fault → entire approach fails
2. E1' shows extent prefetch doesn't engage on mmap → NEXTENT fails; pivot to explicit PrefetchVirtualMemory
3. E2' PFOR ≤2× over sequential → layout marginal; "negative result on weight permutation"
4. E3' top-30% covers <50% volume → not Zipfian enough; ZIPFLOCK fails
5. E4' iGPU overlap <20% → either compute dominates or thread sync overhead exceeds benefit
6. E5 < 0.5 tok/s under any branch → hard fail

## Sharpened Novelty Claim

**Novel contribution is the conditional architecture, not any single mechanism.** Specifically: information-theoretically optimal storage layout (PFOR-spectral) combined with extent-aligned read-ahead (NEXTENT) makes OS demand-paging competitive with explicit predict-then-fetch even on consumer Windows hardware where individual fault latency is 30-60× the synthetic IOPS-implied number.

Distinct from:
- **Apple LLM in a Flash**: DISABLES OS caching, uses direct I/O with explicit predicted prefetch. We do opposite: ENABLE OS caching, arrange weights so OS's existing prefetch heuristic is correct without prediction.
- **llama.cpp mmap**: default sequential GGUF layout. We replace with co-access-graph permutation.
- **NeurIPS 2022 Coleman et al.**: graph reordering for vector index structures, not LLM weight tensors.
- **Speculative decoding / KV-cache offload**: orthogonal; PFOR/NEXTENT operate at storage layer beneath them.

**Single-sentence pitch**: *We make 4 KB the unit of meaning by making each 4 KB page a near-clique in the weight co-access graph; the OS's existing prefetcher and trimmer then do inference-time routing without us writing a router.*

## The Unconventional Move That Survives Empirical Reality

**Reified claim: NTFS extents are a Bloom filter for "what weights are needed next."**

Lay out PFOR-permuted weights so each 1 MB extent is a co-access clique. When model touches any weight in clique, NTFS Cache Manager's read-ahead pulls whole extent. The act of faulting on weight w *answers the question* "what other weights does w predict?" — answer: "the rest of this extent." **The disk-format encodes the predictor.**

Collapses three things conventionally separate:
1. Weight storage layout (offline)
2. Inference-time prefetch prediction (online)
3. Cache admission policy (online)

Into one offline computation: spectral permutation. After permutation, **prediction is geometry, not inference.** OS's read-ahead is correct because layout makes it impossible to be wrong by more than spectral cut value of boundary between extents.

The bet survives by being restated: *OS is a perfect predictor when the layout makes prediction trivial.* Work shifts from runtime to layout time — exactly the move that makes sense given empirical fault-latency reality.

## User Context

Ryzen 5 7530U, Vega 7 iGPU, 16 GB DDR4-3200 (7.28 GiB usable), Samsung-class consumer NVMe, Windows 11. Target ≥2 tok/s on 70B-class is ~100× the working set laptop can hold resident.

Realistic-branch math: 70B at IQ3_XXS = ~30 GB. ZIPFLOCK top 30% = 9 GB pinned (exceeds 7.28 GiB — reduce to top ~20% = 6 GB pinned, 1.28 GiB free for tail + KV + activations). At 0.5 ms compute per token of resident work + iGPU-overlapped tail-fault work, **theoretical ceiling ~2 tok/s under E0b**. Under E0a, ~3-4 tok/s. Both branches clear 2 tok/s ambition under at least one realistic empirical outcome.

## Path / Changes

1. E0 added as hard prerequisite
2. Conditional cascade (E0a/E0b both legitimate research plans)
3. WSTLO demoted from lead to corollary
4. NEXTENT added as new primary mechanism under E0b
5. ZIPFLOCK added
6. iGPU overlap promoted to load-bearing under E0b
7. Spectral ordering promoted over RCM
8. NQRA downgraded
9. Single-sentence pitch: layout makes prediction geometry, not inference
10. CF9 trim-mid-forward-pass risk addressed via extent-granular working set + ZIPFLOCK pinning
11. Acknowledgements: Apple LLM in a Flash (architectural opposite), llama.cpp mmap (accidental prototype), NeurIPS 2022 Coleman et al. (technique precedent)

Core bet survives — relocated from working-set trimmer to NTFS extents + Cache Manager. PFOR is the contribution; everything else is plumbing chosen empirically.
