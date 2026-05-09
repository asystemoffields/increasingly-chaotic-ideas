# Stage 2 Prior-Art Filter — Unconventional Angles / PFOR + WSTLO

## Verdict: PARTIALLY NOVEL with critical empirical risk

## M1: Cuthill-McKee/spectral graph ordering of weight co-access graph for storage layout
Status: PARTIAL (closest: Apple LLM in a Flash row-column bundling, Graph Reordering for Cache-Efficient NN Search NeurIPS 2022)
What survives: Generalizing row-column bundling (two-tensor heuristic) to globally-optimal graph-theoretic permutation at 4 KB page granularity. Spectral ordering (Fiedler vector) likely dominates RCM for dense irregular co-access graphs.
Sharpen: Lead with "we generalize bundling from two-tensor heuristic to globally-optimal graph permutation"; cite NeurIPS 2022 for the technique's prior application to vector index graphs.

## M2: OS page-fault handler + working-set trimmer as inference-time weight router
Status: PARTIAL (llama.cpp mmap is the accidental prototype; LLM in a Flash explicitly disables OS caching; opposite bet)
What survives: Architectural distinction predict-then-fetch (LLM in a Flash) vs layout-then-fault (PFOR). Opposite bet on whether OS caching is feature.
Sharpen: This architectural split is the cleanest differentiator and should be the paper's lead.

## M3: Memory-mapped LLM weights with offline-computed page layout
Status: NOVEL combination (mmap is prior; offline co-access-graph permutation is novel)
What survives: llama.cpp uses mmap with default sequential GGUF layout. PFOR replaces with co-access-graph-optimal permutation.
CF9: layer-sequential GGUF is layer-sequential not co-access-optimal. Real advance over llama.cpp baseline.

## M4: Memory pressure as LRU eviction oracle
Status: NOVEL in LLM inference literature
Closest: General OS literature on clock-LRU; no LLM paper uses trimmer as design component.
CF9: Trimmer interval 3.75-15s under memory pressure means pages for current tokens may evict mid-forward-pass under heavy pressure from other processes. Failure mode unaddressed.

## M5: AQFKV head-redundancy as redundant-page-copy detection
Status: NOVEL
Closest: None directly. PowerInfer routes hot/cold but doesn't detect redundancy in attention weights.

## M6: iGPU shared-memory zero-copy on faulted-in pages
Status: NOVEL (within LLM inference)
Closest: OpenUMA (recent open-source effort).

## CRITICAL CF9 risks

### EMPIRICAL RISK (the 5.6 μs assumption)
Consumer NVMe at QD=32 random 4K achieves 400k-900k IOPS in synthetic tests. **However on Windows NTFS with standard storage stack, page faults on mmapped files do NOT generate QD=32 requests.** Windows issues demand-page faults serially through Memory Manager, effective queue depth QD=1-4 per thread. Sustained single-thread random 4K on consumer NVMe on Windows = 20-100k IOPS, latency 10-50 μs/page, not 5.6 μs.

Further, Windows 11 NTFS overhead (metadata locking, PFN database lock contention) adds latency. Realistic demand-page fault round-trip on Windows 11 with modern NVMe = **100-300 μs per fault**, not 5.6 μs.

**The 5.6μs/180k IOPS assumption is optimistic by 3-9×.**

At 50 pages/token × 50-300 μs = 2.5-15 ms fault latency per token = 67-400 tok/s if compute-free, but realistic compute is 110ms+ per token, so faults dominate at 50 pages and yield 0.07-0.2 tok/s, NOT 2.5 tok/s.

**MUST measure E0 (page-fault round-trip benchmark) BEFORE any other experiment.**

### Trimmer behavior under heavy pressure
Trim interval 3.75-15s; pages for CURRENT tokens may evict mid-forward-pass under pressure from other processes.

### Spectral vs RCM
For dense irregular co-access graphs (plausible given attention's global mixing), RCM may converge to poor solution. Spectral ordering (Fiedler vector) should be primary recommendation.

## Apple LLM in a Flash (arXiv:2312.11514) — the dangerous adjacency
- Storage layout rewrite: row-column bundling (FFN only)
- Predict-then-fetch (sparsity predictor for FFN neurons, app-level direct I/O bypassing OS cache)
- Windowing (inter-token reuse)

PFOR contributions vs LLM in a Flash:
- Generalizes bundling to full-graph globally-optimal permutation (M1)
- Architectural inversion: layout-then-fault vs predict-then-fetch (M2)
- 4 KB page granularity matching NVMe atomic units (vs row/column granularity)
- Targets Windows + NTFS (not iOS); different OS machinery considerations

## Required amendments
1. Add E0 page-fault round-trip benchmark BEFORE everything else. If 100-300 μs confirmed, redesign target tok/s downward.
2. If E0 confirms high latency, consider hybrid with async prefetch (architecture moves toward LLM in a Flash, eroding novelty — but still novel via co-access graph layout).
3. Position spectral ordering as primary, RCM as fallback.
4. Address trim-interval failure mode under heavy memory pressure (other processes consuming pages mid-forward-pass).
5. Acknowledge llama.cpp mmap as accidental prototype; PFOR's contribution is the LAYOUT optimization not the mmap mechanism.
