# Round 3 Stage 1 brief (Stage 1 has run; this is now the Round 3 reference doc)

## Constraint expansion (2026-05-08, after Stage 1 completed)

**Architectural transposition is now in scope.** The "no retraining" constraint forbids SGD against a loss function — it does NOT forbid changing the architecture if the trained weights can be transposed into the new shape via a non-gradient procedure (closed-form fitting, calibration-driven clustering, mathematical equivalences, tensor reshapes).

This widens Round 3 Stage 4's brief (skeptic explorer) and Round 4+ ideator briefs to admit:
- FFN → MoE via expert-clustering + calibration-fit router (no SGD)
- Softmax attention → linear attention via Performer-style positive random features (exact in expectation)
- Dense attention → structured-sparse attention with the pattern derived from calibration
- Nonlinearity replacement: silu → polynomial / piecewise-linear / LUT fit on calibration
- Mamba-style / SSM operator fit to trained MLP behavior on calibration (no SGD, just regression)
- Kronecker / Tucker / hierarchical decomposition reshapes
- Layer merging beyond Basis Sharing — algebraic equivalences

The user noted: "I'm not saying I expect to find that just lying around" — these are hard but the search must include them.



This is the brief for Round 3's ideator agent. Key updates since Round 2:

## Empirical findings to feed forward

### Round 1 (W_up dominance)
SwiGLU top-K firing dominated by `W_up·x`, not `W_gate·x`. Predictors using gate-side only fail in deep Qwen3 layers (precision below random). Generalizes to all SwiGLU/SiLU-gated families.

### Round 2 (K-dependent outlier dynamicity)
Per-token outlier-channel-set Jaccard on Qwen3-1.7B normal stratum:
- K=0.1% (top ~2 channels): Jaccard **0.718** — channel-static at top tier
- K=1-10% (operational range): Jaccard **0.28-0.35** — token-dynamic

LLM.int8()'s channel-static finding holds **only at K=0.1%**. At any operational K for quantization, outliers rotate per token. PrefixQuant's bimodal pattern (massive vs normal tokens) is partially supported but not dominant — only 4.6% of pairs were massive in our sample.

### Bandwidth wall (Round 2 Stage 3)
70B-on-NVMe at any bpw is fundamentally bandwidth-limited at ~0.04 tok/s. No compression beats it; only architectural moves do.

## Round 3 framing-diversity targets

Heavy-used in Rounds 1+2: prediction-internal, compression-quant, compression-lr, offload-disk, runtime-fused, compression-tensor, compression-entropy, kv-side, prediction-shadow, prediction-historical, systems-fs.

**Genuinely under-explored**:
- **output-side** (lm_head, vocab embedding, sampling) — only LSEI proposed, not selected
- **approximate-compute** (not compression — actual operation replacement) — only MFAP proposed
- **coding-theory** — only FCWR fountain codes; polar codes / LDPC / network coding open
- **cross-domain numerical** — NMF and Sparse PCA tried; ICA / sparse coding / dictionary learning still untouched
- **physics-inspired** beyond MFAP — renormalization group / belief propagation / message-passing
- **speculative-execution patterns from CPU branch prediction** adapted to LLM
- **distributed-systems patterns** (gossip, vector clocks, CRDTs) — wholly untouched

## Surviving from Round 2 (should NOT be re-proposed bare; can be combined with new ideas)

- **A** — ANS-Fused Sparse Residual Byte-Stream (rescoped to 7B-13B DRAM)
- **H** — AVX2 VPSHUFB Ternary + Sparse Correction (Zen-3-AVX2 wedge)
- **CLASP** — Chunked-Layer Speculative Prefetch (deferred, high implementation lift)
- **LSEI** — Logit Shortlist via LSH on Vocab (deferred refinement)
- **PDAP** — Per-Token Outlier Pinning (Round 2 selected, with K-dependent finding)

## Key constraint that Round 3 should respect

The bandwidth wall finding suggests "best 70B-runnable" requires *architectural* moves (CLASP-like batching) rather than compression. "Best DRAM-resident" (7B-13B class) is the conventional sweet spot.

Round 3 should aggressively probe:
1. Architectural moves that break the bandwidth wall in Framing I
2. Mid-scale DRAM-resident ideas in untouched frames (output-side, approx-compute, coding-theory)
3. Compositions of Round 1+2 surviving primitives with new mechanisms

The K-dependent outlier finding (R2) opens specific design space: **schemes that pin top ~0.1% of channels statically AND handle the per-token rotation in the 1-10% range dynamically** — neither LLM.int8() (pure static) nor pure dynamic per-token approaches suffice.
