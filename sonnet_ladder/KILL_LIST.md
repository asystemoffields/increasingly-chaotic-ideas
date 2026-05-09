# Cumulative Kill List

One-line per killed idea: `Round R / Stage S — name (class[,class]) — reason`

Maintained by the Ladder pipeline. Future ideators read this to avoid
re-proposing exhausted ideas. Do not delete entries.

## Pre-pipeline (already-killed paths from main project history)

- pre / — Layer-drop / Block influence pruning (sparsity-static) — works K=1-2 layers but doesn't help residency at frontier scale
- pre / — Speculative decoding with off-the-shelf drafters (prediction-shadow) — 20-26% accept on Qwen3 family, below break-even in both ik_llama.cpp and mainline llama.cpp
- pre / — WrapSketch attention capsules (compression-lr) — page-capsule and per-token phase index both died on autoregressive long-prefix
- pre / — SABLE-v1 scalar Top-K + Horvitz-Thompson (sparsity-dynamic) — diffuseness intrinsic in Qwen3 early/mid layers
- pre / — SABLE-v2 polynomial control variate via TensorSketch (compression-lr,sparsity-dynamic) — variance d^{m+2}/D infeasible at transformer activation scale
- pre / — Activation-weighted low-rank correction objective (compression-lr) — equals QERA Theorem 1 (Zhang et al. arXiv:2410.06040), already published

## Round 1

### Stage 2 cuts

- R1/S2 — Inter-Layer Delta Quantization with Anchor (compression-quant) — DeltaLLM/ADAMIX prior art; ceiling below NanoQuant
- R1/S2 — Arithmetic-Coded Weight Streams + Lazy Decode (compression-entropy,runtime-fused) — ZipServ ASPLOS 2026; AVX2 serial decode
- R1/S2 — OS-Paging-Aware Weight Layout (systems-os) — doesn't reduce bpw, just engineers paging
- R1/S2 — Signed-Activation Weight Skipping (sparsity-dynamic) — SiLU is dense, cold fraction <15%
- R1/S2 — Cascaded Residual Codebook + NVMe Entropy Gate (compression-quant,offload-disk) — QDEC overlap
- R1/S2 — Per-Token Multi-Level bpw (compression-quant,offload-disk) — multiple-version storage breaks residency math
- R1/S2 — KV Temporal Differencing (kv-side) — RoPE rotates keys, deltas not small
- R1/S2 — DCT Weight Compression (compression-entropy) — weight matrices have no spatial structure

### Stage 3 conditional kills (after deeper analysis)

- R1/S3 — Vocabulary-Conditioned Weight Residency (sparsity-dynamic,prediction-historical) — 110ms/cluster-swap × frequent transitions makes 1 tok/s impossible; deep-layer activations are context-shaped not token-shaped
- R1/S3 — Columnar NVMe Retirement with Predictor (sparsity-dynamic,offload-disk) — SwiGLU 5% natural sparsity; full-hidden-dim predictor adds 576ms/token

### Stage 3 surviving (conditional advance — listed for completeness, NOT killed)

- R1/S3 — Shadow-Model NVMe Prefetch Oracle (prediction-shadow,offload-disk) — ADVANCE conditional: novelty holds (no cross-scale model-to-model prefetch); risks: NanoQuant binary storage hostile to selective access; SwiGLU density caps skip rate
- R1/S3 — Global SVD Basis + Binary Coefficients (compression-lr,compression-quant) — DOWNGRADE conditional: Basis Sharing covers per-group; mixed matrix shapes need multiple bases; only worth experiment if global K=256 captures 90% energy

### Stage 6 amendment to Stage 5 selection (note for Round 2 ideators)

- R1/S6 — Cross-layer cosine sim ≥ 0.70 as a meaningful gate (prediction-internal) — KILLED: Deja Vu (arXiv:2310.17157, ICML 2023) already reports ≈0.99 in OPT-175B; structurally guaranteed by residual additivity. Do not re-propose cosine-similarity-as-novelty for transformer adjacent layers.
- R1/S6 — Trained cross-layer MLP activation predictor for ReLU (prediction-internal) — PRIOR ART: Deja Vu owns this. Do not re-propose. The open-question variant is zero-overhead dot-product predictor on SwiGLU (which is what R1's selected experiment tests).

### Round 1 selected (queued in SELECTED.md, do NOT re-propose unless first run is run and result motivates a refinement)

- R1/SELECTED — Zero-Overhead Cross-Layer Row-Herding Predictor on SwiGLU (prediction-internal, offload-disk) — see SELECTED.md for full plan. After running: kill or advance based on result; if killed, add post-hoc kill entry citing the empirical numbers.

### Round 1 — empirical findings from selected experiment (smoke, 5 prompts on Qwen3-0.6B-Base)

These are PROVISIONAL pending the full 200-prompt run, but the qualitative pattern is already clear enough to record as a structural insight for Round 2's ideator.

**Structural finding 1 — SwiGLU top-K firing is dominated by up_proj, not gate**
- Even the within-layer baseline `|⟨h_L, W_gate[L]_row⟩|` (perfect within-layer information) achieves precision@30% only 0.40-0.50 in early-mid layers, and **drops below random (<0.30) in layers 18-27** of Qwen3-0.6B.
- The "drop below random" is the load-bearing finding: gate-magnitude is *anti-correlated* with post-SwiGLU magnitude in deep layers. The post-SwiGLU output `silu(W_gate·x) ⊙ (W_up·x)` has top-K firing dominated by the `W_up·x` factor in trained Qwen3.
- **Class implication**: any predictor that operates only on gate-side information (or activation magnitude into gate_proj only) will fail to predict top-K firing without access to up_proj. Affected ideas:
  - R1/SELECTED — Idea D itself: CONFIRMED HEADING TO NO-GO
  - R1/S4-deferred Idea J (Count-Min Sketch on gate activations) — STRUCTURALLY WEAKENED if frequency table is keyed on gate-firing patterns; salvageable only if keyed on combined (gate, up) firing
  - R1/S3 conditional Idea 3 (Shadow Oracle) — STRUCTURALLY WEAKENED if oracle predicts via gate-side only; needs to be reframed to predict via post-SwiGLU magnitude (which requires gate AND up at runtime, partially defeating the savings)
  - Generalizes: SiLU/GELU + element-wise gating models (Llama-2/3, Qwen, Mistral, Gemma) inherit this. Pure-ReLU models (older OPT, GPT-2) do NOT — gate-only prediction works there because there's no up-gating.

**Structural finding 2 — residual additivity confirmed** (already known from Deja Vu, reconfirmed here)
- cos(h_L, h_{L+1}) = 0.88-0.97 across all 28 layers of Qwen3-0.6B. Adjacent residual states are nearly parallel. Useless as a discriminating signal for prediction; structurally guaranteed by the residual stream architecture.
- **Class implication**: do not propose ideas that rely on cosine-similarity-of-adjacent-residual-states as a meaningful gate or feature.

### Round 1 deferred (Stage 4 ideas not selected — eligible for Round 2 ideator to consider, but they were already generated and judged)

- R1/S4-deferred — Idea A: Exponent-Palette Fused Decode (compression-entropy,runtime-fused) — generated but not picked; Stage 5 noted marginal gain over Q4 baseline
- R1/S4-deferred — Idea B: Tensor-Ring Weight Factorization with NVMe Core Streaming (compression-tensor,offload-disk) — generated but not picked; risk that r >> 128 needed
- R1/S4-deferred — Idea C: Roaring Bitmap Residual Sign (compression-entropy,runtime-fused) — generated but not picked; ≤Q4.5 ceiling
- R1/S4-deferred — Idea E: ANS/tANS Fused Decode 8-Way Interleaved (compression-entropy,runtime-fused) — generated but not picked; cleanest go/no-go test (zstd a GGUF) but lowest impact ceiling (~12% bandwidth gain)
- R1/S4-deferred — Idea F: Succinct Rank-Select Structured Sparsity (sparsity-static,runtime-fused) — generated but not picked; highest ceiling (70B in 24.5 GB) but historically fragile no-retrain pruning
- R1/S4-deferred — Idea G: CoW Overlay with Graceful Degradation (systems-fs,offload-disk) — generated but not picked
- R1/S4-deferred — Idea H: Krylov Subspace Iterative Approximation (compression-lr) — generated but not picked; reduces to SVD/Basis Sharing variant
- R1/S4-deferred — Idea I: NTFS-Reflink Codebook Deduplication (systems-fs) — generated but not picked; Windows 11 Home limitation
- R1/S4-deferred — Idea J: Count-Min Sketch Activation Filter (prediction-historical) — generated but not picked; Stage 6 noted this is independent of D and survives even if D fails

## Round 2

### Stage 2 cuts
- R2/S2 — Haar Wavelet Tucker Core (compression-tensor) — neural weight matrices have no spatial structure; Haar coefficients dense
- R2/S2 — Bloom-Filter KV Retirement per Head (kv-side, prediction-historical) — wrong bottleneck for short context (KV ~670MB at 4K vs weights at 17-40GB)
- R2/S2 — MDL-Selected Per-Layer bpw (compression-quant, compression-entropy) — Compressibility Measures Complexity (Oct 2025) covers this; converges to mixed-precision territory
- R2/S2 — Trie/FSST Codebook over Q4_0 Nibbles (compression-entropy, systems-fs) — FSST is prior; well-quantized weight nibbles have near-uniform entropy
- R2/S2 — Sparse PCA Cross-Head Basis (compression-tensor, compression-lr) — CoSpaDi (Sep 2025) and "Share Your Attention" (Aug 2025) cover this
- R2/S2 — NVMe Prefetch Sequencer via RoPE Fingerprints (offload-disk) — wrong axis (residency not latency); Windows storage stack already prefetches

### Stage 3 kills/downgrades
- R2/S3 — NMF Codebook with Sign-Factored Residual (compression-tensor, compression-quant) — KILLED: sign bitmap dominates (1 bit/weight); 6-day NMF runtime on CPU; no advantage over PTQTP/NanoQuant/LittleBit
- R2/S3 — Cross-Head Variance KV Pinning (kv-side) — DOWNGRADE/PARK: bottleneck mismatch confirmed; only relevant for >32K context
- R2/S3 — Dual-ternary post-hoc decomposition core mechanism — PRE-EMPTED by PTQTP (arXiv:2509.16989, Sep 2025); only Zen-3-AVX2-specific kernel + sparse correction is salvageable

### Stage 3 surviving (rescoped, do NOT re-propose bare)
- R2/S3 — ANS-Fused Sparse Residual Byte-Stream (A) — RESCOPED to 7B-13B DRAM-resident; CPU fused entropy+matmul kernel is genuinely novel
- R2/S3 — AVX2 VPSHUFB Ternary + Sparse Correction (H, downgraded) — Zen-3-AVX2 kernel + sparse correction wedge

### Stage 4 not selected (deferred)
- R2/S4 — W4A8-Stream — modest 1.5x gain, similar lift to CLASP without leverage
- R2/S4 — CLASP Chunked-Layer Speculative Prefetch — highest Framing-I leverage but 1-2 week implementation lift; deferred for Round 3+
- R2/S4 — LSEI Logit Shortlist via LSH on Vocab — promising but it's a refinement; deferred
- R2/S4 — AVX2 W(ternary)A8 GEMV — extension of H; would chain after PDAP validates
- R2/S4 — FCWR Fountain-Code Weight Reconstruction — highly speculative
- R2/S4 — UCLOCK Zen 3 µOp-Cache Decode Kernel — pure micro-arch, low cross-idea info
- R2/S4 — MFAP Mean-Field Activation Propagation — precondition (repetitive context) too narrow

### Round 2 selected (queued in SELECTED.md after Stage 6 amendments)
- R2/SELECTED — PDAP Per-Token Dynamic Activation Outlier Pinning (with Stage 6 amendments) — symmetric to Round 1's W_up dominance finding; tests activation-side outlier dynamicity

### Round 2 — Stage 6 prior-art notes (FOR ROUND 3 IDEATORS)

Critical: PrefixQuant (arXiv:2410.05265, Oct 2024) explicitly decomposes activation outliers into:
- **Channel-wise outliers**: stable across tokens, handled by SmoothQuant/OS+/etc.
- **Token-wise outliers**: sparse, concentrated on BOS / delimiters / specific high-frequency tokens; ~2 per 2048 tokens contribute 94.7% of quant error.

Any future activation-side experiment must stratify by token type to avoid conflating these two phenomena. LLM.int8() (arXiv:2208.07339) and SmoothQuant (arXiv:2211.10438) did NOT formally measure per-token Jaccard — that's open. Quaff (arXiv:2505.14742, ACL 2025) introduced "Outlier Spatial Stability Hypothesis" for fine-tuning. NVFP4 Pretraining outlier dynamics (arXiv:2602.02047) addresses training-time, not inference-time per-token structure.

### Round 2 — empirical findings from PDAP (full run, 200 prompts on Qwen3-1.7B-Base)

CONFIRMED at full sample size: 32425 normal-token consecutive-pairs at K=1%, across 7 sampled layers of 28-layer Qwen3-1.7B-Base.

**Structural finding 3 — outlier dynamicity is K-dependent on Qwen3 family**

Per-token consecutive-pair Jaccard at K = top-fraction-of-channels by abs magnitude (normal-token stratum, mean across layers):

| K (% of d=2048) | Mean Jaccard |
|---|---|
| 0.1% (~2 chans) | **0.718** — channel-static at top tier (LLM.int8() pattern HOLDS for the very top) |
| 1.0% (~20 chans) | **0.308** — token-dynamic |
| 5.0% (~102 chans) | 0.257 — token-dynamic |
| 10.0% (~205 chans) | 0.244 — token-dynamic |

Massive-activation pairs (4.1% of total) at K=1%: Jaccard = 0.182 — lower than normal (PrefixQuant pattern partially supported), but normal is already firmly token-dynamic. The bimodal massive-vs-normal split is NOT the dominant signal.

**Class implication**: LLM.int8() and SmoothQuant's "channel-static outlier" claim is **correct only for the very top handful of channels** (K≈0.1%). At any operationally interesting K for activation quantization (1-10%), the outlier *set* rotates substantially per token. This means:
- Pure channel-static masking (LLM.int8()) is sub-optimal for any quantization scheme pinning >0.1% of channels.
- Pure per-token dynamic is overkill at K=0.1% (the 2 stable channels can be pinned statically).
- **Hybrid scheme indicated**: pin top-0.1% statically (cheap, lossless) + per-token dynamic for the 1-10% range.
- The W(ternary)A8 stack (Round 2 ideas H+4) needs per-token outlier handling at operational K.

Channel concentration evolves across layers: layers 2-19 have 5-8% of channels covering 90% of outlier events; layers 23-27 have 14-18%. **Deep layers spread outliers across more channels** (more diversity, harder to predict).

## Round 3

### Stage 2 cuts
- R3/S2 — Winograd-Tiled MLP (compression-quant, runtime-fused) — Winograd is for convolution (sliding kernel reuse); standard GEMV has no spatial reuse. Mechanism does not apply. Confirmed by ACM/Springer Winograd literature.
- R3/S2 — Belief Propagation Quant Annealing (compression-entropy, physics-inspired) — loopy BP on dense weight-correlation factor graphs has no convergence guarantee; falsifiability paradoxically low.
- R3/S2 — Renormalization-Group Layer Fusion (physics-inspired) — RG handles multiplicative coupling; transformer residual connections are additive. Mechanism does not apply. Reduces to known low-rank attention.
- R3/S2 — Logit Skirt Vocab Clustering (output-side) — lower ceiling than the 12+2 line; lm_head is not the dominant 70B bottleneck.
- R3/S2 — Sparse-Dictionary lm_head with K-SVD (output-side, cross-domain numerical) — subsumed by Idea 12; modest residency vs MLP-side compression.
- R3/S2 — KC-Guided Layer Scheduling (information-theoretic) — quantized weights have near-uniform compressed entropy; modest impact estimate.
- R3/S2 — Branch-Predictor Token Trie (offload-disk, prediction-historical) — pre-empted by N-Gram Trie Speculative Decoding (EMNLP 2025) and SuffixDecoding (NeurIPS 2025); per-token prefetch saves only 1.4% per layer; bandwidth contention on misprediction.

### Stage 3 kills/downgrades
- R3/S3 — Idea 1 W_up Draft Oracle (prediction-shadow) — DOWNGRADE: literal "zero W_gate" interpretation kills the layer (silu(0)=0 → MLP output zero). Corrected mechanism (full small model as draft) reduces to standard staged speculative decoding; accept rate marginal at ~25% on cross-size draft. NVMe K-batch amortization insight remains real but unproven.

### Stage 6 fatal math flaw (CSMF)
- R3/S6 — CSMF Polynomial SiLU substitution claiming W_gate elimination — KILLED: math flaw confirmed. silu(W_gate·x) ≈ α + β·(W_gate·x) + γ·(W_gate·x)² leaves β·(W_gate·x)·(W_up·x) and γ·(W_gate·x)²·(W_up·x) terms that still require W_gate at inference. Polynomial substitution replaces the activation function (saves ~3 FMAs/neuron, 0.018% of layer compute), NOT the linear projection. Add to general kill class: "polynomial activation substitution as a mechanism for weight elimination" — it does not work. The published adjacent technique that DOES work is AERO (arXiv:2410.13060) — remove activation entirely, then W_gate and W_up algebraically fold into one matrix. AERO is real and published; if Round 4 revisits "eliminate a projection weight," it must engage AERO.

### Stage 4 (arch-transposition) deferred
- R3/S4-deferred — Idea A CALFAV (Calibration-Fitted Diagonal+LR Attention Kernel) — generated but not selected; the most ambitious arch-transpose; Round 4 candidate
- R3/S4-deferred — Idea B CHASM (Per-Head MoE Calibration Clustering) — Round 4 candidate; stacks multiplicatively with 12+2
- R3/S4-deferred — Idea D BCPD Block-Circulant Permuted W_down — high reconstruction error risk
- R3/S4-deferred — Idea E JLVH (JL-Sketched Vocabulary Head) — cheap output-side win, deferred
- R3/S4-deferred — Idea F LSDT Zen-3 LSD-Tiled GEMV — pure micro-arch optimization, low cross-idea info
- R3/S4-deferred — Idea G RSTD Residual-Stream Tucker Cross-Layer — convergence signal from R1's Idea 5 (deferred there too); strongest dominance candidate; should be Round 4 priority if Round 3's Idea 12 is positive

### Round 3 selected (escalated from Stage 5; ESCALATE verdict by Stage 6)
- R3/SELECTED — Idea 12 Asymmetric W_gate Rank Reduction — see SELECTED.md

### Class-level kill (for Round 4+ ideators)
- "Polynomial activation substitution to eliminate weight matrices" is mathematically broken. Polynomial substitution of f(Wx) changes the cost of evaluating f, not the cost of storing or applying W. Round 4 ideators must explicitly identify which object in `f(Wx)` a proposed compression targets — W (residency), the GEMV (compute), or f (nonlinearity cost) — before claiming residency savings.

### Round 3 — empirical findings from Idea 12 W_gate rank sweep

CONFIRMED: W_gate has essentially no low-rank structure in Qwen3-1.7B-Base bf16. PPL impact at all-layer simultaneous rank reduction:

| K (rank) | K/d_in | ΔPPL vs baseline (11.78) |
|---|---|---|
| 2048 (full, sanity) | 1.000 | +0.03 |
| 1024 | 0.500 | +13.7 |
| 512 | 0.250 | +408.6 |
| 256 | 0.125 | +5560 |
| 128 | 0.063 | +19237 |
| 16 | 0.008 | +14777 |

Even at 50% rank retention, PPL doubles. At 25%, PPL is 35× baseline.

**Structural finding 4 — Round 1's W_up dominance is about firing-magnitude RANK, NOT W_gate's linear-algebra RANK**

These are different questions:
- R1: which neurons FIRE STRONGLY post-SwiGLU? Answer dominated by W_up·x.
- R3: is W_gate LINEARLY low-rank? Answer: NO. All singular values of W_gate carry meaningful information for capability preservation.

**Class implication for Round 4 ideators**: do NOT propose ideas of the form "R1 implies W_gate carries less load-bearing information, so W_gate should compress more aggressively than W_up." This was Idea 12's premise and it is empirically false. The two notions of "importance" (firing magnitude vs linear-algebra dimensionality) are decoupled.

**What's still in scope**:
- **Idea G (RSTD cross-layer Tucker on W_up)**: tests W_up compressibility, the matrix that R1 says IS load-bearing. The cross-layer shared basis is a different mechanism than per-layer rank reduction. Convergence signal from R1's deferred Idea 5.
- **Idea 2 (PDAP three-tier codebook on W_up)**: tests R2 finding directly on the matrix R1 identified as critical.
- **AERO-style FFN folding** (arXiv:2410.13060): activation removal + algebraic merge of W_gate and W_up. Only known mechanism to TRULY eliminate W_gate. Round 4 should consider whether this transposition works on Qwen3 family.
- **Idea A (CALFAV)**: attention-side arch-transpose; orthogonal to MLP findings.
- **Idea E (JLVH)**: output-side LSH lm_head shortlist; orthogonal.

**The 12+2 line is killed.** Round 4 must find a different path to DRAM-resident 70B.

## Track B Round 4 — empirical findings from W_up rank sweep (CLUBS gating experiment)

**Structural finding 5 — W_up is ALSO full-rank in trained Qwen3, MORE rank-sensitive than W_gate**

Qwen3-1.7B-Base bf16, 28 layers, intermediate=6144, baseline PPL=11.78. All-layer simultaneous rank reduction of W_up (W_gate and W_down kept at bf16):

| K | W_up ΔNLL | W_gate ΔNLL (R3 reference) |
|---|---:|---:|
| 2048 (full) | (sanity, ~0) | +0.002 |
| 1024 (50%) | +2.34 | +0.77 |
| 512 (25%) | +8.39 | +3.57 |
| 256 (12.5%) | +8.40 | +6.16 |
| 128 (6.3%) | +8.82 | +7.40 |
| 64 (3.1%) | +7.04 | (similar) |

W_up is *more* rank-sensitive than W_gate at matched K fractions. PPL doubles already at K=1024, becomes catastrophic at K≤512.

**Class implication — generalization of R3 finding**: "R1 says W_up matters more (firing-rank), so W_up should be the compressible matrix" was the CLUBS premise. It is empirically false. Firing-rank dominance and linear-algebra rank are independent properties.

**KILLED ideas**:
- R4-B/CLUBS (cross-layer basis + tiered coefficients on W_up) — built on the premise W_up is low-rank
- R4-B/RSTD (cross-layer Tucker on W_up) — same premise
- Generalized: any compression scheme premised on "W_up should compress more aggressively than W_gate because R1." Both matrices are full-rank in trained Qwen3.

**For Round 5+ ideators**: low-rank decomposition of MLP weight matrices in trained Qwen3 family does NOT work without retraining. Compression must come from quantization (PDAP-style tiering, sparse outliers, codebooks) or arch-transposition (SDZC-style function-output folding, MoE clustering, etc.). The full-rank finding applies to W_gate (R3) and W_up (R4) — almost certainly applies to W_down too (untested but predictable).

The bandwidth wall remains intact: 70B at 0.55 bpw on NVMe is still the floor. DRAM-resident 70B requires sub-1-bit quantization (NanoQuant territory at 5.35 GB) plus aggressive KV management — no low-rank shortcut exists.

## Track A Round 1 — empirical findings from SDZC (gate-variance probe)

**Structural finding 6 — In trained Qwen3-1.7B, gate output is overwhelmingly NOT near-constant**

Per-neuron statistics of silu(W_gate·x) across 566 calibration tokens × 10 diverse passages on Qwen3-1.7B-Base bf16:

- Globally: only **1.5% of 172K neurons foldable** (std < 0.05). Below the 5% kill threshold.
- Layer-1 anomaly: 36.1% foldable in layer 1 alone (likely early-layer specialization). All other layers <2.5%.
- Deep layers (5-27): 75-99% "live" (std ≥ 0.1). Median std at layer 27 = 0.91.

**Class implication — yet another failed proxy from R1**:

Round 1's "W_up dominates firing-rank" finding has now generated three derived hypotheses:
1. R3 (W_gate rank reduction): "W_gate matters less, so it's low-rank" — FALSE
2. R4-B (W_up rank reduction): "W_up matters more, so SHOULD be compressible because R1" — FALSE; W_up is *more* rank-sensitive than W_gate
3. R4-A (SDZC gate folding): "Gate carries less firing-information, so its output may be near-constant" — FALSE; gate output is overwhelmingly variable

Three different proxies for "compressible because R1 says less load-bearing" have all failed. **R1's firing-rank dominance is NOT a route to no-retraining structural compression.**

### KILLED ideas (R4 cumulative)

- R4-A/SDZC (SwiGLU Dead-Zone Gate Collapse) as a global compression scheme — KILLED. Layer-1 anomaly (36% foldable) is interesting but isolated; can't anchor 70B deployment.
- R4-B/CLUBS (cross-layer basis + tiered W_up) — KILLED (W_up full-rank).
- R4-B/RSTD (cross-layer Tucker on W_up) — KILLED (same).

### Compounded structural finding (R3 + R4)

**Finding 7 (super-structure)**: Trained Qwen3-family weights resist no-retraining structural compression. Three independent measurements (W_gate rank, W_up rank, gate-output variance) all show full-rank or full-variance behavior. The R1 firing-rank dominance is real but does NOT translate to ANY of:
- linear-algebra rank deficiency of either matrix
- function-output near-constancy of the gate
- token-level sparsity of firings (R2 already showed K=1-10% outliers are token-dynamic)

**Implication for Round 5+ ideators**: every compression idea premised on "R1 says X is less important so X is compressible" must be empirically pre-validated before pipeline investment. The pattern is now established: firing-rank dominance is a stochastic statistic on the forward pass, NOT a property of the underlying weights or activation distributions. Avoid this entire reasoning template.

### What's actually still alive

**Track B (compression)**:
- **PDAP three-tier on W_up** (R2-grounded; quantization, not rank reduction; UNTESTED) — strongest survivor
- **PDAP-V on W_down** (extension) — UNTESTED
- **SOBIB INT2+sparse outliers** (calibration-sensitivity selection; ~2.2 bpw) — UNTESTED

**Track A (arch-transpose)**:
- **AERO-style activation removal + W_gate/W_up algebraic folding** (the only known mechanism to truly eliminate W_gate; activation removal hasn't been tested for SwiGLU specifically without retraining)
- **CFPK linear-attention** (NNLS feasibility was bad at r=64; r=8 shallow-only might salvage)
- **KV-side methods** at long context (CFPK, PHKS, etc.) for >32K context use cases

**Cross-track**:
- Layer 1 SDZC for free — 36% saving on layer 1 alone is real but tiny in absolute terms
- Combining quantization with KV-side methods for total residency

## Track A R2 closure (Stage 6 amendments)

### Track A R2 selected: WDLA (amended)

WDLA — Weight-Delta Layer Accumulation. Cross-scale reduced-order inference via calibration-fit affine corrections.

**Amendments per Stage 6**:
- Use Qwen3-0.6B → Qwen3-1.7B (NOT Qwen3-7B which doesn't exist; not 1.7B → 4B which has 28→36 layer mismatch + memory issues at 8 GB bf16 on 7.28 GiB system).
- Same 28-layer depth eliminates alignment ambiguity.
- Add Phase 2.5: gate Phase 3 perplexity test on mean R² > 0.85 across layers 10-28.

**Frame novelty**: narrowly survives. Model Stitching (NeurIPS 2025, arXiv:2506.06609) confirmed training-time only. No inference-time cross-scale affine surgery found in 2025-2026 lit.

### Track A R2 killed by Stage 3
- CRANK (γ-inert RMSNorm) — pre-empted by SmoothQuant; 0% residency benefit on CPU-NVMe
- TRED (frozen lm_head early exit) — pre-empted by SimLens (March 2026), ConfLayers (April 2026)
- DLRA (deep-layer linear block) — pre-empted by source paper itself; rank-32 was fabricated
- RECS (EMA recurrent KV) — pre-empted by RetNet (2023), RADLADS (May 2025)

### Track A R2 killed by Stage 2
- RELU-FOLD — same category error as CSMF (silu→ReLU is lossy substitution)
- IOFC — out of scope (systems engineering, not arch-transpose)
- PAVK — KV-codebook crowded field
- SPRAT — content-routed attention well-published
- RMFA — attention high-frequency, DFT compression doesn't apply

### Track A R2 Stage 4 (NEW skeptic ideas, deferred for now)
- LLSC (lm_head SVD with RAM-pin smaller factor)
- DSOP (depth-stratified operator portfolio)
- APCI (adjacent-layer delta compression during NVMe stall)
- FSKA (formal sparse kernel attention)
- LCAB (Layer-1 specific gate folding)
- KSVA (rank-r_K KV projection)
- XPOW (delta-logit update)

## Track B R5 closure (Stage 6 amendments)

### Track B R5 selected: SPADC (amended, weakened)

SPADC — SwiGLU Post-Activation Distribution Codebook for W_down.

**Amendments per Stage 6**:
- **Severe novelty erosion**: GPTVQ (arXiv:2402.15319, Feb 2024) does Hessian-weighted EM on codebooks. Hessian = X^T X = activation second moment. SPADC is incremental over GPTVQ, not a frame-novel idea.
- **Bimodal claim is likely overstated**: only 1.5% gates near-constant per R4 SDZC. Distribution is more likely skewed-unimodal than bimodal.
- **k-means weighting was dimensionally ambiguous** as written. Coherent version (column-weighted) reduces to GPTVQ.
- **Reframe**: from "bimodal-specific" to "SwiGLU-specific activation-skew-aware codebook vs GPTVQ Hessian baseline."
- Test at 2 bpw primary (where centroid placement matters most), 3 bpw secondary.
- Raise threshold to 0.5 nats.
- W_up comparison is PRIMARY falsifiability (if equal gain, replicates AQLM/GPTVQ rather than novel).

**Status**: SPADC remains queued but heavily weakened. The amended experiment is closer to a replication-extension study than a novel mechanism.

### Track B R5 killed by Stage 3 / earlier
- GRAEC (GPTQ residual ANS coding) — partially pre-empted by WaterSIC + act-order structural threat
- CAPS (calibrated Huffman on INT2) — PTQTP at 1.58 bpw dominates
- PHSKVO (per-head KV INT2/INT8 split) — KVQuant + MiniKV + KITTY occupy
- SSCC (SwiGLU codebook) — marginal, k-means on Gaussian collapses to uniform
- GLDF (GGUF dedup) — storage-only, no residency benefit
- SLHP — out of scope (Track A)
- ASPI (inline scale layout) — kernel surgery cost prohibitive
- SOBIB (INT2 + sparse INT8 outliers) — calibration-only sensitivity may underperform Hessian-based; AVX2 scatter overhead

### Track B R5 Stage 4 (NEW skeptic ideas, deferred for now)
- RSIDC (intrinsic-dimension-based bpw schedule)
- GUCB (within-layer cross-tensor codebook sharing)
- DAPCS (prompt-vs-continuation phase codebooks)
- CIVSS (block-local input variance scheduling)
- ENVC (lm_head null-token pointer aliasing)
- LHQM (nested lattice + sparse residual hybrid)
- FWSC (NVMe-DMA-stall SVD subspace error correction)

### R2-A and R5-B compounded learning

**Saturation manifested in selection too**: Track A's Stage 5 picked WDLA (least pre-empted) but used a non-existent model name. Track B's Stage 5 picked SPADC, which Stage 6 found is closely subsumed by GPTVQ. Selectors are biased toward "narrative novelty" which doesn't always match technical novelty in publication-saturated regions.

**The pipeline is approaching the publication frontier**. The kill list now includes ~50 ideas, and more recent rounds increasingly identify ideas pre-empted by 2025-2026 papers. Genuinely novel space exists past this boundary but requires more aggressive frame-orthogonal generation.

## Track A R2 — empirical findings from WDLA experiment

WDLA ran on Qwen3-0.6B → Qwen3-1.7B (Stage 6 amended pair). Result: **NO-GO** with R² catastrophically negative on held-out.

### Structural finding 8 — calibration data ill-conditioning

The affine map A_L: R^1024 → R^2048 has 2048×1024+2048 ≈ 2.1M parameters. Calibration provided ~1000 tokens × 2048 target dims = 2.05M output values. **The fit is underdetermined**: ridge=1e-3 is insufficient to prevent perfect calibration memorization with garbage held-out behavior.

| Layer | R²_calib | R²_eval | cos_eval |
|---|---:|---:|---:|
| 25 | 1.000 | -12329 | -0.03 |
| 26 | 1.000 | -23708 | -0.03 |
| 27 | 1.000 | -118121 | +0.21 |

**Class implication**: cross-scale model surgery via least-squares-fit affine correction REQUIRES calibration data N >> d_target × d_source / regularization_strength. For 1.7B-class targets, this means ~10K-100K calibration tokens AND aggressive ridge (1e-1 or higher). The Stage 5 plan's 1000-token budget was insufficient by 1-2 orders of magnitude.

**WDLA does NOT die from this**; the frame is still potentially viable with:
- Larger calibration corpus (10K-100K tokens)
- Stronger ridge (1e-1 or higher)
- Lower-rank A_L by construction (force rank ≤ 64 from the start, parameter count drops to 2048×64+1024×64 ≈ 200K — well-conditioned at 1K tokens)

**Lesson for Round 4+**: any calibration-fit linear surgery on transformer hidden states must explicitly compute `n_params_to_fit` vs `n_independent_samples × output_dim` and choose regularization accordingly. The Stage 5 plan didn't, and Stage 6 didn't catch it. **Add this to PIPELINE.md as a check item for selectors.**

### What's now still open in Track A

- Track A R2 Stage 4 deferred ideas (LLSC, DSOP, APCI, FSKA, LCAB, KSVA, XPOW)
- Track A R3 ideas (Parity-Shard FFN, Predicate-Gated Materialization, Carry-Lookahead, Interval-Shrink, Epidemic Residual, Lapped-Transform, Spiking-Burst)
- A rescue WDLA with proper calibration scale + low-rank construction

## Track A R3 Stage 3 verdicts

- **Interval-Shrink Gate Absorber** — DOWNGRADE: math correct, novel framing, but 1.5% global prune rate (R4 SDZC empirical) saves only ~30-45 MB on a 3 GB model. Micro-optimization, not residency lever. Marginal.
- **Parity-Shard FFN** — DOWNGRADE conditional: algebraically equivalent to rank-k SVD with GF-shaped basis. Fails on same R4 W_up rank-sensitivity evidence that killed CLUBS. **Run W_up SVD spectrum first; if rank-4480 captures <99.5% variance, this is dead.**
- **Lapped-Transform FFN Tiling** — KILLED: three independent blockers — Windows NTFS NVMe doesn't deliver QD32 parallelism in ik_llama.cpp; 50% storage bloat guaranteed; LOT's locality assumption has no basis in the neuron weight domain.

## Track B R6 Stage 3 verdicts

- **THWR (Tabulation-Hash)** — KILLED: reduces to VPTQ (per-position non-uniform quant). Tabulation hashing machinery contributes nothing technically beyond naming.
- **CSKQ (Count-Sketch)** — KILLED: L1 error bound is 5470× signal magnitude. Count-Sketch needs sparse signals; INT4 residuals are dense. Math is broken; 784 ms/token compute overhead.
- **RFSK (Random Fourier Features)** — KILLED: Bochner's theorem doesn't apply to weight matrices (not kernels). Mechanism reduces to JL projection, strictly worse than SVD which CLUBS proved fails on full-rank Qwen3 W_up.

## Cross-stage-3 structural finding (Compound finding 8)

After R3 (W_gate full-rank), R4 (W_up MORE rank-sensitive), and now Track A R3 + Track B R6 results: **the compression and arch-transpose space for trained Qwen3 MLP weights is fundamentally bounded by their empirical full-rank-ness.**

Multiple independent rounds attempting structural-compression of W_gate / W_up have died on this same wall:
- Idea 12 (W_gate rank reduction) — R3 empirical kill
- CLUBS / RSTD (cross-layer Tucker on W_up) — R4 empirical kill
- Parity-Shard FFN — R3-A Stage 3 conditional kill on same evidence
- RFSK — R6-B Stage 3 kill on same evidence

**Class-level kill**: "find a low-rank or low-effective-rank structure in trained Qwen3 MLP weight matrices to enable structural compression" is empirically dead. Round 4+ ideators MUST NOT propose ideas that depend on this assumption.

**What's still open**:
1. **Attention weights** — never directly tested for rank structure on Qwen3 family
2. **KV cache** — not the weight matrices; different object
3. **I/O scheduling** without changing the bytes — accept full-rank weights, optimize how they're streamed
4. **Empirical-structure-driven** — R2 outlier dynamics, R4 SDZC Layer-1 anomaly, residual-stream intrinsic dimension (RSIDC was deferred), inter-layer weight similarity (DeltaLLM-related but post-hoc)
5. **Output-side / lm_head** — Logit-Layer Spectral Collapse (LLSC) was a Track A R2 deferred candidate; lm_head IS approximately low-rank per arXiv:2510.24966

**Priority next experiment for resolving the W_up question definitively**: SVD spectrum of W_up across all layers of Qwen3-1.7B. Plot singular value decay curves. If 99% of variance is captured at rank << d (say, < d/4), the rank-reduction ideas have a chance. If the spectrum is flat (which R4 strongly suggests), the entire family is dead.

## Track A R3 Stage 4 (frame-orthogonal skeptic; CF8-compliant — deferred for now)

Ideas all pass CF8 by targeting NON-MLP-weight objects (attention, KV, lm_head, I/O):

- SGHH — Sink-Gated Head Hibernation (calibrated bias replaces sink-routed heads; 15-17% bandwidth saving claim) — DEFER pending AQFKV outcome
- CVEEKS — Compression-Valley Early Exit + KV Staleness (composition: dynamic exit + stale-KV fill) — DEFER; precondition (compression valley locatable in Qwen3 decoder) unverified
- CLASE — Cross-Layer KV Aliasing INT4 deltas (~37% KV reduction on aliased pairs) — DEFER; bottleneck mismatch at <32K context (KV is ~670 MB on 4B at 4K; weights dominate)
- LRALP — Logit-Rank Adaptive lm_head Projection (dynamic top-3K vocab per step; arXiv:2510.24966 grounded) — DEFER; FR-Spec / arXiv:2509.18362 closely adjacent (frame partially pre-empted)
- INPOLT — NVMe Prefetch with Phase-Ordered Layer Tiling (re-pack GGUF for sequential dual-stream NVMe) — DEFER pending NVMe-bottleneck profile (precondition: NVMe is binding, not RAM-bandwidth or matmul on this hardware)
- ASERSF — Attention-Score Entropy Routing for Sparse KV Fetch — DEFER (long-context only; not load-bearing at 4K)
- RSDIPS — Residual-stream Intrinsic-Dim Conditioned Precision Switching — DEFER (modest gain; closely related to deferred RSIDC from Track B R5)

### Track A R3 selected: AQFKV (pre-Stage 6)

AQFKV — Q-Head SVD with KV Preserved. Replace W_Q per-layer with rank-K SVD reconstruction (K ∈ {16,32,64,128,256,512,1024}); W_K/W_V/W_O held at bf16. Tests whether CF8 (MLP weights full-rank) extends to attention W_Q — the LAST untested weight class on Qwen3. Either outcome produces structural finding; ~50 min runtime; surgical.

## Track B R6 Stage 4 (frame-orthogonal skeptic; CF8/CF9-compliant — deferred for now)

Six ideas; one selected. All target NON-MLP-weight objects (KV cache, lm_head, activations, NVMe I/O):

- KHQL — KV Head Quantization with Layer Sensitivity Ladder (per-head static INT8/INT4 mix; calibration-derived attention-logit-mass; 40% KV reduction at 20/80 split) — DEFER; closely adjacent to KVmix/PM-KVQ literature; precondition (within-layer head heterogeneity on Qwen3) untested
- RAOK — R2-Aware Outlier-Keyed Activation Codebook (3-tier: 2 chans FP16 static / 18 chans INT8 dynamic / 2026 chans INT4 static, grounded in our R2 PDAP K-dependent Jaccard finding; 1.5-2.5× AVX2 GEMV speedup claim) — DEFER (most CF9-robust idea, no imported theory; runs after lm_head spectrum question is resolved)
- NVPF — NVMe Layer Prefetch via Static Access Fingerprint (deterministic two-slot weight buffer with overlapped I/O; baseline measurement at 70B-NVMe scenario) — DEFER; produces single scalar baseline rather than structural finding
- ASCQ — Attention-Sink Channel Quantization for KV (INT2 sink-V / INT4 sink-K / INT8 non-sink) — DEFER; weak in isolation at 4K context; scaffolds for KHQL composition at 128K
- SVLD — Singular Value Ladder Deployment (progressive SVD lm_head with confidence-gated early exit; r=64 if argmax-confidence-mass>0.9 else expand) — CONDITIONAL ON LHQD; shares first experiment; activates if lm_head spectrum is concentrated

### Track B R6 selected: LHQD first-pass (pre-Stage 6)

LHQD first-pass — SVD spectrum of lm_head.weight (151936×2048) on Qwen3-1.7B-Base bf16. Plot singular value decay; reconstruct at r ∈ {64,128,256,512,1024,2048}; measure NLL on 512-token WikiText-2 held-out. STRONG GO if 99% variance at r≤256 AND ΔNLL<0.10. NO-GO extends CF8 to all weight matrices (quantization-only path). ~35 min runtime. Shares experiment with SVLD.

## Round 3-A AQFKV — empirical findings (CF8 boundary delineation)

**Verdict: Global GO at K=128 (ΔNLL=+0.98) AND Per-head GRAY (sanity-pass at K=128 but K=64 NO-GO at +1.53).** This is the modal Stage-6-anticipated outcome for the disambiguation question. W_K bonus measurement: GO-adjacent at K=512 (ΔNLL=+0.29 nats).

### Quantitative results (Qwen3-1.7B-Base bf16, 455 eval tokens)

| matrix | K | K/d | ΔNLL (nats) | compress |
|---|---|---|---|---|
| W_Q global | 1024 | 0.50 | +0.03 | 1.0× |
| W_Q global | 512 | 0.25 | +0.20 | 2.0× |
| W_Q global | 256 | 0.125 | +0.51 | 4.0× |
| **W_Q global** | **128** | **0.062** | **+0.98** | **8.0×** |
| W_Q per-head | 96/128 | 0.75 | +0.94 | 1.25× |
| W_Q per-head | 64/128 | 0.50 | +1.53 | 1.88× |
| W_K global | 512 | 0.50 | +0.29 | 2.0× |
| W_K global | 256 | 0.25 | +0.82 | 4.0× |

Spectrum diagnostic: W_Q L14 r_99=1293, var@256=0.64; W_K L14 r_99=814, var@128=0.53. **MLP comparators (W_gate, W_up): r_99 ≈ d (essentially full-rank).**

### Compound finding 10 (formalized)

CF8 (originally: "trained Qwen3 weights are full-rank") must be restricted: **CF8 applies to MLP weights only.** Attention weights (W_Q, W_K) have substantially more concentrated spectra. The structural class boundary is sharp: r_99/d ≈ 1.0 for MLP, ≈ 0.63 for W_Q, ≈ 0.79 for W_K.

### Structural finding 11

The 8× compression at global W_Q K=128 is NOT a per-head weight-rank finding. It is a **head-redundancy** finding: 16 query heads collectively span a ~128-dim subspace (one head's worth of capacity). Per-head SVD at K_per_head=64 is NO-GO (+1.53 nats). Per-head K_per_head=96 is borderline (+0.94 nats). Each head's natural 128-dim cannot be reduced cleanly.

**Implication**: MLA-style joint Q-K low-rank projection in head space is empirically motivated for Qwen3 post-training (as opposed to the training-time MLA in DeepSeek-V2). The closest published prior art is A3 (arXiv:2505.12942) which uses attention-logit-error minimization; the AQFKV measurement uses pure weight Frobenius error and reaches a structurally consistent conclusion.

### What's now open in Track A

- **MLA-style head-sharing** for Qwen3 attention: post-training joint Q-K low-rank (no retraining)
- **W_V and W_O spectra**: untested; cheap parallel measurement (5-10 min each)
- **Cross-layer Q basis**: do all 28 W_Q share a common subspace? PCA-stack experiment
- **Direct deployment win**: W_Q at K=512 + W_K at K=512 = 2× attention weight residency reduction at <0.5 nats. Engineering follow-up worth doing.

### What's killed by R3-A AQFKV

- "CF8 extends to all weight matrices in Qwen3" — REFUTED. Class boundary is sharp.
- "Isolated W_Q low-rank reduction" as a per-head compression — KILLED. Per-head K_per_head=64 NO-GO.
- "MLP rank reduction with attention left alone" as the compression strategy — REVERSED. Attention is more compressible than MLP. The natural strategy is the opposite.

## Round 6-B LHQD — empirical findings (tied-embedding hard-gate triggered)

**Verdict: EMBED-TIED. Spectrum FLAT.** In Qwen3-1.7B-Base, `tie_word_embeddings=True` and storage is shared (data_ptr equality). Stage 6's hard gate caught it cleanly. The measured matrix is the joint embed_tokens/lm_head, not an independent lm_head.

### Quantitative results (Qwen3-1.7B-Base bf16, 399 primary + 3 multi-passages)

| r | ΔNLL (nats) | top1 ret | top5 ret | KL/tok |
|---|---|---|---|---|
| 2048 (full) | -0.0024 | 0.967 | 0.981 | 0.0016 |
| **1024** | **+19.96** | **0.005** | 0.008 | 20.22 |
| 512 | +21.34 | 0.000 | 0.000 | 21.60 |
| 256 | +22.42 | 0.000 | 0.001 | 22.85 |
| 128 | +25.62 | 0.000 | 0.000 | 26.15 |
| 64 | +26.03 | 0.000 | 0.000 | 26.59 |

Spectrum: r_99=1992/2048 (97% of available), var@r=256=0.28, var@r=512=0.43.
Per-row norms: median 1.62, bottom-decile 1.36, near-zero count = 0 (NOT a dead-row artifact).
Multi-passage robustness: consistent across 3 disjoint passages (max ΔNLL +28.85 at r=64).

### Compound finding 12

In **tied embedding configurations**, the joint embed/lm_head matrix is full-rank because gradient signal flows through TWO paths during training:
1. Input embedding lookup gradient (from every token in every sequence)
2. Output projection gradient (Godey & Artzi's "lm_head as gradient bottleneck")

The Godey & Artzi argument applies only to path 2; path 1 ensures every direction stays load-bearing. **The mechanism predicting concentrated weight spectrum is REFUTED in tied configurations.**

### Refinement to CF8 boundary classification

Adding the tied embed/lm_head data point reshapes the picture:

| matrix family | r_99/d | K=50% rank ΔNLL | structural notes |
|---|---|---|---|
| MLP W_gate, W_up | ≈1.0 | +0.77 to +2.34 | full rank, dense nonlinear routing |
| **Tied embed/lm_head** | **0.97** | **+19.96 (catastrophic)** | **full rank with extreme functional sensitivity to truncation** |
| Attention W_K | 0.79 | +0.29 | concentrated, moderate compression possible |
| Attention W_Q | 0.63 | +0.03 | most concentrated, head-redundancy at 16:1 ratio |

### What's killed by R6-B LHQD

- LHQD (INT8 U + BF16 V^T at r=256) for Qwen3-1.7B-Base — KILLED in tied config
- SVLD (progressive SVD with confidence-gated early exit) on tied lm_head — KILLED
- "Concentrated lm_head weight from gradient bottleneck training dynamics" — REFUTED in tied config
- ENVC (lm_head null-token pointer aliasing, Track B R5 deferred) — KILLED by same evidence

### What's still open

- **Untied lm_head spectrum** (Qwen3-8B has `tie_word_embeddings=False`; GPT-NeoX, Llama-3 8B/70B all untied). The Godey & Artzi prediction may hold there. Streaming SVD on Qwen3-8B feasible despite RAM constraints (decompose blockwise, never materialize full U).
- **arXiv:2510.24966 logit low-rank theorem** — survives. Function-level rank ≠ weight-level rank. Moves to RSIDC (residual stream intrinsic dimension) measurement.
- **RAOK** — strongest surviving Track B path. R2-grounded, no imported theory.
- **MLA-style joint Q-K projection** — opened by AQFKV.
- **W_V / W_O spectra** — cheap follow-up to AQFKV; close out the attention-weight class characterization.
- **Cross-layer Q basis sharing** — does the head-redundancy pattern in W_Q extend across layers?

### Honest framing note (per feedback_no_saturation_claims.md)

The LHQD NO-GO did NOT close the search space; it sharpened the CF8 boundary, refuted a specific theoretical prediction in a specific configuration (tied), and surfaced the untied case as a genuinely different open question. Track B's compression line moves cleanly to RAOK; Track A's attention line was already opened wide by AQFKV.
