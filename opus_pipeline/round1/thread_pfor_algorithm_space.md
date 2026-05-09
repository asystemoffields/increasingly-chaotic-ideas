# Side Thread: Optimal Weight Layout Algorithm with Empirical NEXTENT

Investigative thread opened 2026-05-08 after M-NVMe empirical confirmation of NEXTENT (1.9 μs same-extent prefetch vs 137 μs first-page fault, 72× ratio).

## Key insight

**Spectral ordering is the wrong algorithm.** Spectral minimizes cut weight. Cuthill-McKee minimizes bandwidth. Our actual objective is **λ−1 (connectivity-1) hypergraph partitioning**: minimize Σ_e (number of partitions a hyperedge crosses) − 1, summed over edges weighted by frequency.

Mathematically:
  C(t, π) = |{π(v) : v ∈ t}|  (# distinct extents touched by token t under layout π)
  π* = argmin_π E_t[C(t, π)] subject to |π⁻¹(k)| = 256 ∀ k

This is **VLSI placement** in disguise. KaHyPar / hMETIS solve it directly. 5,000 extents for 70B IQ4 model is well within solver tractability.

## Architecture (NAHP — NEXTENT-Aware Hypergraph Partition)

1. **Calibration trace**: 1000 tokens; record per-token block-IDs touched. Build hypergraph H = (V=blocks, E={S_t per token}, weighted by frequency).
2. **Coarsen** by heavy-edge matching (collapse always-co-firing block pairs).
3. **Initial partition**: KaHyPar with λ−1 objective, k=5000, balance ε=0 (hard).
4. **V-cycle refinement**: FM gain bucket on λ−1 metric.
5. **Within-extent ordering**: spectral on residual induced subgraph (helps L1/L2 prefetch streams within an opened extent).
6. **Meta-extent layout**: Cuthill-McKee on partition-quotient graph so prefetched-cold extents are physically nearby (NTFS extent merging gives further amortization).

**Multi-level**: λ−1 partitioning at 1 MB (NVMe-tier objective), spectral within-extent (L2/L3 prefetch streams), Cuthill-McKee at meta-extent (NTFS extent-merge friendliness). Each level optimizes the metric matched to its hardware tier.

## Hybrid with explicit prefetch (LLM-in-a-Flash hybridization)

- **Tier A (predicted-hot)**: small extent-predictor MLP (~10K params, prev K tokens' attention pattern + layer index → top-N extent IDs). PrefetchVirtualMemory ~5-10 ms ahead of FFN.
- **Tier B (cold tail)**: demand-fault + NEXTENT for the 30% unpredictable activations.

Layout absorbs structural locality; prediction absorbs semantic locality. At 80% Tier A recall, residual cold ≈10 extents/token = ~1.5 ms IO floor — under compute.

## Quantitative ambition

- 70B IQ4 = ~5 GB residency = 5,000 extents
- Naive sequential layout: ~800 unique extents/token
- Spectral baseline (educated guess): ~200 extents/token
- NAHP (λ−1 + hybrid): target ≤50 extents/token
- Saved IO: (250 − 50) × 137 μs = 27 ms/token
- Compute: ~110 ms/token at IQ4 on Ryzen
- Net: 5.7 tok/s → 8.2 tok/s = **44% throughput gain from layout alone**
- 8B class: extrapolate to 12-15 tok/s

## Cascade

E1 (4 hr) — Access trace + entropy probe; spectral baseline. Go: ≥3:1 compressibility AND spectral >100 extents/token.
E2 (2 hr + 30 min KaHyPar) — λ−1 partitioning. Go: NAHP ≤50 extents/token median.
E3 (6 hr) — End-to-end NTFS deployment + tok/s measurement. Go: ≥7 tok/s on Llama-70B-IQ4.
E4 (8 hr) — Hybrid prefetch overlay. Go: ≥15% additional gain.
E5 (1 day, stretch) — Adaptive layout (background defragmenter on sliding 10K-token window). Go: ≥10% improvement at steady state.

## Cross-domain analog

The right analog is **VLSI placement / FPGA logic packing**, NOT databases. "Minimize signals crossing tile boundaries" = "minimize extents crossed per token" mathematically identical. hMETIS / KaHyPar are VLSI tools. Borrow VLSI evaluation methodology too: wirelength histograms map to per-token extent-count histograms; report p50/p95/p99 not just mean.

## What this kills/distinguishes

- KILLS spectral as a weight-layout primitive (wrong objective for this problem).
- Distinguishes from Apple LLM in a Flash: their row-column bundling is a 2-tensor heuristic; NAHP is globally optimal at 1 MB granularity.
- Distinguishes from PowerInfer: predict-then-fetch routes by neuron sparsity; NAHP routes by co-access graph at storage layer.

## Headline pitch

**Treat 70B-weight layout as a VLSI placement problem. Solve it with 30-year-mature hypergraph partitioning tools. Let the OS handle paging.** No new algorithms; just the right algorithm against the right objective, with M-NVMe showing the OS substrate cooperates.

Source: Opus agent run 2026-05-08, full reply at task af663c5c403329402's successor.
