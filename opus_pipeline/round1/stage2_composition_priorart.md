# Stage 2 Prior-Art Filter — Composition Agent / Outlier Highway

## Verdict: SURVIVES WITH AMENDMENTS

## M1: Activation outlier decomposition into static/dynamic/bulk subspaces aligned to memory hierarchy
Status: PARTIAL
Closest: LLM.int8() (NeurIPS 2022), Systematic Outliers (arXiv:2502.06415), Rethinking the Outlier Distribution (arXiv:2505.21670, May 2025), PrefixQuant (arXiv:2410.05265).
What survives: No prior uses decomposition as STORAGE ROUTING PROTOCOL. LLM.int8 routes to FP16/INT8 kernels; PrefixQuant routes to KV cache vs dynamic quant. Three-way split as STORAGE-TIER assignment (L1d/DRAM/NVMe) not published.
Sharpen: Position vs Rethinking the Outlier Distribution — that paper observes two populations but doesn't exploit Jaccard-cliff as basis-selection signal. Our contribution: cliff sharp enough to define hardware-stable basis; basis stability measurable by inter-session Jaccard.

## M2: Outlier-channel-to-W_Q-head-shared-subspace alignment as routing structure
Status: NOVEL
Closest: DeepSeek-V2 MLA (arXiv:2405.04434), TransMLA (arXiv:2502.07864), Linear Predictability of Attention Heads (arXiv:2603.13314).
What survives: dim(D^1% ∩ rowspan(W_Q^shared)) ≥ 0.6 × |D^1%| — outlier channels and W_Q head-shared subspace co-aligned — unpublished. Prior notes head-redundancy and outlier persistence SEPARATELY.
CF9: Highest-risk quantitative claim. 0.6 figure asserted without measurement. Realistic prior 0.3-0.7. Test: principal angles between top-1% outlier channel span and W_Q rowspan across heads.

## M3: AVX2 PEXT/PDEP for sparse outlier channel gather
Status: NOVEL (within ML inference)
Closest: DeepSparse (deprecated June 2025), generic SIMD PDEP guides.
What survives: PEXT/PDEP for ~20-channel gather out of d~3000 in transformer forward, CPU-only deployment. PDEP re-scatter into 128-dim W_Q subspace projection.
Sharpen: At K=1%, 20 channels from d=3072 is small gather; pre-sorted index loop may beat PEXT/PDEP chaining. Benchmark both.

## M4: NVMe-resident MLP weights with bus-signal-driven prefetch
Status: ADJACENT
Closest: FlexGen (ICML 2023), PowerInfer (SOSP 2024), Pre-gated MoE (ISCA 2024), MoE-SpeQ (arXiv:2511.14102).
What survives: Signal used for prefetch is novel. PowerInfer trains predictor; Pre-gated MoE trains gate; MoE-SpeQ uses spec decoding. Bus signal is ZERO-COST — emerges naturally from highway decomposition.
Sharpen: Formalize prefetch lead time. NVMe ReadFileEx ~100μs; per-token at 3-5 tok/s = 200-333ms. Prove h_bus computed early enough in each layer to issue prefetch before W_down needed.

## M5: L1-resident persistent residual highway via static outlier channels
Status: NOVEL
Closest: Massive Activations (Sun et al., 2024), FairyFuse (arXiv:2604.20913, Apr 2026).
What survives: Identifying ~2 static outlier channels by Jaccard stability and pinning projection in L1d as first-class architectural primitive — not published. Reframes massive activations from "phenomenon to manage" to "communication medium to exploit."
CF9: "Static" claim depends on Jaccard 0.72 reproducible across sessions and model instances. Massive activations may be position-fixed (BOS, delimiters); CHANNEL identity may not be fully input-independent. Direct measurement on Qwen3-4B needed.

## M6: Information-theoretic framing
Status: NOVEL framing; HIGH CF9 RISK on quantitative claims
Closest: PRAC (arXiv:2602.23111, Feb 2026), KVQuant (NeurIPS 2024).
What survives: Bounding I(h_hwy+h_bus;y) as fraction of I(h;y) and using bound to justify NVMe residency of h_bulk — unpublished.
CF9: 0.85 claim is most dangerous single number. KVQuant inverse evidence (1% outlier removal → <0.1 ppl) doesn't directly constrain because different axes (KV cache vs residual stream). MEASURE DIRECTLY: zero h_bulk, measure perplexity increase.

## Two-paper compositions to flag
1. Systematic Outliers + FlexGen ≈ "use persistent outlier channels to route to NVMe-resident weights" — value-add: signal specificity (Jaccard-cliff + W_Q alignment) + overlapped prefetch
2. PowerInfer + Massive Activations ≈ "outlier channels as hot/cold predictor" — value-add: outlier projection IS the predictor at zero marginal cost
3. PrefixQuant + MLA ≈ "static channel outliers offline, dynamic token outliers through shared low-rank subspace" — value-add: storage-routing framing as distinct contribution

## Required amendments
1. Measure 0.6 intersection fraction (W_Q rowspan ∩ D^1% basis) before publishing
2. Measure 0.85 information concentration claim empirically (zero h_bulk, record perplexity)
3. Sharpen distinction from PowerInfer: quantify h_bus available X μs before W_down needed at zero overhead
4. Acknowledge "Rethinking the Outlier Distribution" (2505.21670, May 2025) as closest observational precursor
