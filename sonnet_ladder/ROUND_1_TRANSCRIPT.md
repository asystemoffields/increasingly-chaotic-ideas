# Round 1 — Transcript

Date: 2026-05-08

## Pipeline state at start

- Cumulative kill list seeded with the pre-pipeline killed paths (layer-drop, off-shelf spec-dec, WrapSketch, SABLE-v1, SABLE-v2 sketch, plain QERA-style)
- Published landscape brief: NanoQuant, BiLLM, PTQ1.61, QDEC, LLM in a Flash, PowerInfer, FlexGen, Basis Sharing, DeltaLLM, ZipServ, ShadowLLM
- Hardware target: Ryzen 5 7530U, 7.28 GiB RAM, no GPU, NVMe SSD

## Stage 1 — Ideator output

12 candidate ideas generated. Self-pick top 3: #3, #8, #6.

## Stage 2 — Judge output

Scored each idea 0-15 across 5 dimensions. Top 4 advance:

| # | Idea | Score |
|---|------|-------|
| 3 | Shadow-Model NVMe Prefetch Oracle | 15 |
| 12 | Vocabulary-Conditioned Weight Residency | 14 |
| 5 | Post-Hoc Global SVD Basis + Binary Coefficients | 13 |
| 2 | Columnar NVMe Retirement | 12 (conditional) |

Cuts: #1 (DeltaLLM/ADAMIX), #4 (ZipServ ASPLOS 2026 overlap), #6 (not residency, just paging mechanics), #7 (SiLU dense), #8 (QDEC overlap), #9 (multi-version storage), #10 (RoPE kills KV deltas), #11 (no spatial structure).

## Stage 3 — Deep researcher output

- **Idea 3 — DOWNGRADE conditional**: novelty unmatched by ShadowLLM/SVD-CSP intra-model literature. NanoQuant binary storage hostile to selective NVMe access. SwiGLU 5% natural sparsity caps skip rate at 1 tok/s.
- **Idea 12 — KILL**: 110 ms/cluster-swap × frequent transitions blocks 1 tok/s; deep-layer activations are context-shaped per probing literature.
- **Idea 5 — DOWNGRADE conditional**: Basis Sharing (ICLR 2025) covers per-group; mixed matrix shapes need multiple bases.
- **Idea 2 — KILL**: SwiGLU 5% sparsity + full-hidden-dim predictor 576 ms/token closes the door.

Synthesis insight: **Hybrid 3+5** (RAM-resident shared basis + NVMe-resident layer A_l blocks + shadow oracle predicts which A_l to prefetch) resolves Idea 3's storage-format issue.

## Stage 4 — Skeptic explorer output

10 new ideas (A–J) from frames Stage 1 didn't use. Self-pick top 3: D, E, F. Meta-hybrid suggestion: F (sparsity) + D (prefetch) + E (storage), three independently testable components.

| # | Name | Class |
|---|------|-------|
| A | Exponent-Palette Fused Decode | compression-entropy, runtime-fused |
| B | Tensor-Ring + NVMe Core Streaming | compression-tensor, offload-disk |
| C | Roaring Bitmap Residual Sign | compression-entropy, runtime-fused |
| D | Cross-Layer Activation Correlation Predictor | prediction-internal, sparsity-dynamic |
| E | ANS/tANS Fused Decode (8-way interleaved) | compression-entropy, runtime-fused |
| F | Succinct Rank-Select Structured Sparsity | sparsity-static, runtime-fused |
| G | CoW Overlay + Graceful Degradation | systems-fs, offload-disk |
| H | Krylov Subspace Iterative Approximation | compression-lr |
| I | NTFS-Reflink Codebook Deduplication | systems-fs |
| J | Count-Min Sketch Activation Filter | prediction-historical |

## Stage 5 — Final selector output

Picked Idea D over the alternatives on grounds of "primitive value" — D's result branches to multiple downstream ideas (3, J, meta-hybrid). Cosine ≥ 0.70 + precision@30% ≥ 0.60 thresholds.

## Stage 6 — Red team output

**PROCEED WITH AMENDMENT.** Findings:

1. **Deja Vu (arXiv:2310.17157, ICML 2023)** already reports cross-layer cosine sim ≈ 0.99 in OPT-175B and implements a trained cross-layer MLP predictor (93-99% precision on ReLU). Stage 5 missed this prior art entirely.
2. The cosine ≥ 0.70 gate is structurally guaranteed by residual additivity — passes trivially regardless of any useful predictive property. Should not be a go-gate.
3. precision@30% ≥ 0.60 is only 2× random — Deja Vu needed 93-99% for usable prefetch. Threshold too low to validate downstream use.
4. SwiGLU post-activation depends on both gate and up; raw `|h · W_gate row|` doesn't cleanly capture top-K firing.
5. Downstream implication map overclaims: positive D doesn't validate cross-model Idea 3; negative D doesn't kill count-min Idea J.

**Amendments applied to SELECTED.md**:
- Drop cosine gate (already known)
- Raise precision threshold to ≥ 0.80 (no-go < 0.55)
- Add within-layer baseline (h_L predicts W_gate[L], same layer) to isolate cross-layer claim
- Reframe hypothesis as "does Deja Vu's trained-predictor result transfer to zero-overhead dot-product on SwiGLU?"
- Narrow downstream implications

## Round 1 final state

**Selected experiment**: Zero-Overhead Cross-Layer Row-Herding Predictor on SwiGLU (Idea D, amended). Queued in SELECTED.md.

**Class taxonomy used in Round 1**:
- Heavy: compression-quant, compression-lr, compression-entropy, sparsity-dynamic, prediction-shadow, prediction-internal, offload-disk
- Light: compression-tensor, prediction-historical, systems-fs, runtime-fused, kv-side
- Untouched: arch-modify (out-of-scope), train-required (out-of-scope)

**Round 2 framing-diversity hint**: Round 1 generated heavily in compression-quant / compression-lr / sparsity / offload-disk frames. Underexplored in Round 1: pure runtime/kernel-level tricks, KV-side methods, cross-domain analogies (LSH, Bayesian, neuroscience), market-style allocation.

## Pipeline state at end

- KILL_LIST.md updated with Stage 2 cuts, Stage 3 kills, Stage 6 amendment notes, Round 1 selection lock
- SELECTED.md has 1 entry (queued)
- 9 deferred Stage 4 ideas eligible for Round 2 (or future rounds) re-proposal
- No experiments run yet; selected experiment remains queued pending user go/no-go on a/b execution mode
