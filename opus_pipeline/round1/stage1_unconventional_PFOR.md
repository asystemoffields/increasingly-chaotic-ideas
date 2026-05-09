# Stage 1 Unconventional Angles — PFOR + WSTLO

## Ambition

Run a 70B-class model on the 7.28 GiB Ryzen laptop at ≥2 tok/s by NEVER deciding which weights to load. The OS virtual memory machinery — page fault handler, working-set trimmer, NVMe read-ahead, prefetcher — is already a high-quality predictor and cache. Expose model weights as memory-mapped object whose byte layout is a learned function of forward-pass access locality. Forward pass dereferences pointers; OS produces (or doesn't produce) bytes.

Aspiration: 70B at 0.55 bpw NanoQuant = 5.35 GB on disk. Working set per token ≤400 MB resident. NVMe sequential reads dominated by hardware prefetcher. **Target 2.5 tok/s, beating 1.8 s/token wall by ~4.5×** by exploiting only-subset-of-weights-matter-per-token + measurable spatial coherence.

The radical move: **the model's weight file is a self-organizing storage object, not a serialized tensor dump.** Layout is a function of co-fault probability, recomputed offline from a profile run.

## Mechanism

**(a) PFOR — Page-Fault-Ordered Routing.** A 4 KB page holds ~512 IQ4_XS weights or ~7000 NanoQuant 0.55 bpw weights. Forward pass at token t touches sparse subset with probability p_ij. Place w_i, w_j on same page when p_ij high. **Graph bandwidth minimization** on weight co-access graph, solved offline by Cuthill-McKee or spectral ordering. Output: permutation π : logical_index → file_offset such that each page's weights have maximal mutual co-fault probability.

Forward pass issues loads through MapViewOfFile-backed pointers. Fault on w_i pulls in 511 (or 7000) co-resident neighbors chosen by offline solver. **Kernel demand-paging IS the routing decision.**

**(b) WSTLO — Working-Set Trimmer as LRU Oracle.** Windows trims working sets under pressure using clock-style approximate-LRU. Set model mmap larger than RAM (5.35 GB on 7.28 GiB = pressured but not OOM). Trimmer evicts pages not touched recently. With PFOR layout, evicted pages = exactly the ones whose contents weren't needed.

LRU caching with **zero software bookkeeping** — no expert-tracking, no LFU counters. OS does it in kernel mode with hardware-assisted PTE accessed/dirty bits.

**(c) NQRA — NVMe Queue-Depth Reverse Allocation.** NVMe at QD=32 random 4K = ~700 MB/s ~180k IOPS, ~5.6μs/page. Kernel batches faults; we want **bursty access patterns hitting same page repeatedly while resident, then moving on cleanly.** PFOR layout produces this naturally: token's working set = K pages, all resident together for forward pass duration, then evicted as next token's set faults in.

Position: **not quantization, not sparsity, not MoE.** Layout technique. Compatible with everything below (any quantization format) and replaces nothing above (no architectural change). Class: "storage substrate."

Math: minimize Σ_pages variance of access-time across w in P, subject to |P| = page_size / weight_size. Dual: maximize co-access probability for weights co-located on a page. Equivalent to graph partitioning with pages as parts. Approximate solution: spectral ordering of co-access graph, chunk in page-sized contiguous blocks.

## Why findings make this work

- R2 K=1% Jaccard 0.31 = ~31% page reuse between adjacent tokens
- R2 K=5%, K=10% Jaccard 0.26, 0.24 vs random K/d = 0.05, 0.10 → co-access 5× and 2.4× above random. **That ratio is exactly the lift PFOR exploits.**
- AQFKV W_Q effectively rank ~128 globally → most of W_Q on disk is redundant restatement; trimmer evicts redundant copies first
- CF12 tied embed/lm_head full-rank K=50% catastrophic +19.96 → embed/lm_head pages must NEVER evict. PFOR pins via VirtualLock on Zipf-frequent token range (~5% of vocab covers ~95% of tokens), cold tail faults on demand.
- Layers 23-27 14-18% vs 2-19 5-8% → late layers need bigger working set per token. PFOR layout per-layer; offline solver allocates more page budget where structure demands.
- **Vega 7 iGPU shared-memory quirk**: iGPU memory IS system memory. iGPU computes on pages CPU just faulted in, zero copy. Free FLOPs.

## Preconditions

P1: Co-access graph has measurable cluster structure. Test: instrument forward pass on Qwen3-1.7B, log every weight-byte access at 4 KB granularity, build co-access matrix on first 10k tokens, compute spectral gap of Laplacian. λ_2/λ_n > 0.1 → exploitable. ~6 hrs instrumented + 30 min spectral solve.

P2: Page-fault latency amortizable. Per-token compute budget exceeds per-token fault budget. At 70B 0.55 bpw, ~300 MB working set per token of new pages = ~75k pages = ~420ms at 180k IOPS QD≥8. Compute per token = ~5.35 GFLOPs at AVX2 int8 ~50 GFLOPs/s = ~110ms. **Faults dominate.** Mitigation: PFOR's ≥70% page reuse from Jaccard collapses 75k → 22k → ~125ms. Go: measured page reuse ≥60% across consecutive tokens.

P3: Windows working-set trimmer evicts in PFOR-friendly order. Test: 1 GB mmap with synthetic access pattern matching Qwen3-1.7B profile, monitor working set via GetProcessMemoryInfo over 10 min, fraction of evictions hitting pages we'd evict. Go: trimmer agreement ≥85%.

P4: Cuthill-McKee/spectral ordering stable across calibration corpora. Test: compute layout on each of {C4, Wiki, code, dialog}, compare via Kendall tau on page-permutation. Go: τ ≥ 0.7.

P5: NQRA bursting actually happens. NVMe QD has to rise during fault burst. Test: ETW trace during forward pass, histogram of queue depth. Go: median QD ≥ 8 during fault bursts.

## Cascade

**E1 (4 hrs): Co-access profile on Qwen3-1.7B.** Modify ik_llama.cpp ggml backend to log weight-byte ranges accessed per token at 4 KB granularity. Run 10k tokens. Build sparse co-access matrix M (N×N, N=pages_in_model ~1M, manageable). Spectral gap.
- Go: gap > 0.1
- No-go: weights are randomly accessed → PFOR can't help; structural finding about transformer access patterns

**E2 (8 hrs): PFOR layout build + smoke test.** Cuthill-McKee on M produces π. Rewrite model file with weights reordered (record π for addressing). Run 1k tokens with thin shim translating logical → physical offsets.
- Go: bit-exact output vs original. Then peak RSS over run < 80% of original.
- No-go: layout doesn't reduce RSS → page selection isn't the bottleneck; pivot to different layout

**E3 (4 hrs): Working-set trimmer agreement.** Force memory pressure (allocate dummy 2 GB), monitor evictions during forward pass via ETW PageFault events. Compute trimmer-vs-oracle agreement.
- Go: ≥85%
- No-go: trimmer's clock-LRU misaligned with our access pattern; need explicit hint via VirtualLock/PrefetchVirtualMemory hybrid

**E4 (12 hrs): Throughput at scale on Qwen3-8B with PFOR.** Build PFOR layout from Qwen3-8B profile (~5× runtime). Run on actual hardware under deliberate pressure (cap RSS at 4 GB via Job Object). Measure tok/s, page-fault rate, NVMe queue depth.
- Go: tok/s within 15% of unconstrained baseline despite 4 GB cap
- No-go: pressure breaks throughput → finding tells us at what residency level OS paging stops working

**E5 (24 hrs, contingent on E4): 70B NanoQuant feasibility.** No 70B Qwen3 in stack yet. Build simulated 70B by tiling Qwen3-8B 9× with offset weights, mmap as 5.35 GB, run PFOR on simulated profile.
- Go: ≥1.5 tok/s on laptop (2.0 success target)
- No-go: ≤1.0 tok/s → PFOR not enough; need hybrid with explicit prefetch

Total: ~52 hrs compute, ~3 days wall.

## What I'm NOT proposing

- Not MoE (no expert routing in software; decision implicit in pointer dereference + kernel paging)
- Not quantization (PFOR runs on top of NanoQuant/IQ4_XS unchanged; bytes reordered not recoded)
- Not prefetch hints (OS hardware prefetcher does sequential well; structure we exploit is spatial; manual prefetch can't help on cold pages; trimmer would fight hints)
- Not RAM-disk caching (Windows already does this via standby list; PFOR exploits standby list)
- Not custom block I/O (bypassing page cache with FILE_FLAG_NO_BUFFERING; killed because trimmer-as-oracle gain depends on going through the cache; INPOLT remains valid alternative)
- Not iGPU compute (noted as free side-channel but not built around)
- Not denormal sentinels, NaN payloads, branch predictor as hash, PSHUFB tricks (would require deep ggml surgery for smaller wins)
- Not category theory or sheaf theory on residual stream (math is gorgeous but no falsifiable measurement in <50 hrs compute)

The bet: **OS is a better predictor than us, and we should let it do the work.** Unconventional move: recognize Windows kernel — written by people who've never thought about transformers — has been optimizing locality of reference for 30 years and beats anything we'd hand-roll. We just lay out bytes so its existing heuristics align with our access pattern.

## CRITICAL CF9 RISKS (per Stage 2 review)

1. **Apple LLM in a Flash (arXiv:2312.11514) does row-column bundling** — special case of PFOR. PFOR generalizes bundling to full-graph Cuthill-McKee at 4 KB page granularity, which is real technical advance, but must be framed as "we generalize row-column bundling from two-tensor heuristic to globally-optimal graph-theoretic permutation."
2. **Architectural distinction from LLM in a Flash**: predict-then-fetch (their app-managed direct I/O) vs layout-then-fault (PFOR's OS demand-paging). Opposite bets on whether OS caching is a feature.
3. **CRITICAL empirical risk**: 5.6μs/180k IOPS Windows page-fault assumption is optimistic by 3-9×. Realistic Windows 11 demand-paging on mmapped NVMe = 100-300μs per fault. At K=50 pages/token = 5-15 ms = 0.07-0.2 tok/s, NOT 2.5 tok/s. **MUST measure E0 (page-fault round-trip benchmark) BEFORE E1.** If measurement confirms high latency, hybrid with async prefetch needed (architecture moves toward LLM in a Flash, eroding novelty).
4. **Working-set trimmer interval (3.75-15s under memory pressure)** means pages for current tokens may evict mid-forward-pass under heavy pressure from other processes. Failure mode unaddressed.
5. **NTFS/PFN database lock contention** adds latency to demand-page faults.

**E0 added**: page-fault round-trip benchmark on Windows 11 NVMe mmapped file. 1 hr. Hard prerequisite for entire cascade. If 100-300μs confirmed, redesign target tok/s downward and add async prefetch.
