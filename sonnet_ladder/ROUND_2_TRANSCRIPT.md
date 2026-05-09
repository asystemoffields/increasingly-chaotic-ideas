# Round 2 — Transcript

Date: 2026-05-08

## Pipeline state at start

- KILL_LIST.md from Round 1 (~25 entries) + Round 1 empirical finding: SwiGLU top-K firing dominated by W_up·x, not W_gate·x.
- SELECTED.md: 1 entry (Round 1, done with structural-finding result)

## Stage 1 — Ideator output

10 candidate ideas in frames Round 1 under-used (compression-tensor, compression-entropy, runtime-fused, kv-side, prediction-historical, systems-fs).

| # | Name | Class |
|---|------|-------|
| A | ANS-Fused Sparse Residual Byte-Stream | compression-entropy, runtime-fused |
| B | Haar Wavelet Tucker Core + NVMe Streaming | compression-tensor, offload-disk |
| C | Bloom-Filter Row Retirement per Head | kv-side, prediction-historical |
| D | NMF Codebook + Sign-Factored Residual | compression-tensor, compression-quant |
| E | Per-Layer Entropy-Adaptive bpw / MDL | compression-entropy, compression-quant |
| F | Trie/FSST Codebook over Q4_0 Nibbles | compression-entropy, systems-fs |
| G | Cross-Head Attention Variance KV Pinning | kv-side, systems-os |
| H | SIMD Ternary Basis + Sparse Correction | runtime-fused, sparsity-static |
| I | Sparse PCA Cross-Head Basis | compression-tensor, compression-lr |
| J | NVMe Prefetch Sequencer via RoPE | offload-disk, prediction-historical |

Self-pick top 3: A, H, G.

## Stage 2 — Judge output

Top 4 advance:

| # | Idea | Score |
|---|------|-------|
| H | SIMD Ternary Basis + Sparse Correction | 15 |
| A | ANS-Fused Sparse Residual | 13 |
| G | Cross-Head Variance KV Pinning | 12 |
| D | NMF Sign-Factored Codebook | 11 (conditional) |

Cuts: B (no spatial structure), C (wrong bottleneck), E (MDL overlap), F (FSST + uniform nibble entropy), I (CoSpaDi + Share Your Attention pre-empt), J (Windows already prefetches).

## Stage 3 — Deep researcher output

- **A — ADVANCE (rescoped)**: novelty confirmed — no CPU fused entropy+matmul exists. Rescope from 70B to 7B-13B DRAM-resident; rANS software decode (~1 GB/s) is slower than NVMe sequential (3-5 GB/s) so 70B framing is wrong.
- **H — DOWNGRADE**: PTQTP (arXiv:2509.16989, Sep 2025) pre-empts dual-ternary post-hoc decomposition. Salvageable: Zen-3-AVX2-specific kernel (Litespark targets AVX-512/NEON only) + sparse correction.
- **G — DOWNGRADE**: bottleneck mismatch confirmed. KV at 4K context = 670 MB; weights at 17-40 GB. KV optimization irrelevant unless context > 32K.
- **D — KILL**: sign bitmap dominates storage (1 bit/weight). 6+ days NMF runtime on CPU. No advantage over PTQTP/NanoQuant/LittleBit.

**Critical reframing**: 70B-on-NVMe at any bpw is bandwidth-limited at ~0.04 tok/s. Architecture moves needed, not better compression. Two valid framings emerged:
- (I) "Best 70B-runnable": amortize/batch/shortcut NVMe loads
- (II) "Best DRAM-resident" (7B-13B): extreme compression + fast decode

## Stage 4 — Skeptic explorer output

8 new ideas across both framings:

| # | Name | Framing | Class |
|---|------|---------|-------|
| 1 | W4A8-Stream Activation Quant | I | runtime-fused, compression-quant |
| 2 | CLASP Chunked-Layer Speculative Prefetch | I | offload-disk, prediction-shadow |
| 3 | LSEI Logit Shortlist via LSH on Vocab | I, II | output-side, runtime-fused |
| 4 | AVX2 W(ternary)A8 GEMV via VPSHUFB | II | runtime-fused, compression-quant |
| 5 | PDAP Per-Token Dynamic Outlier Pinning | II | compression-quant, runtime-fused |
| 6 | FCWR Fountain-Code Weight Reconstruction | II, I | compression-coding-theory |
| 7 | UCLOCK Zen 3 µOp-Cache Decode Kernel | II | runtime-fused, hardware-specific |
| 8 | MFAP Mean-Field Activation Propagation | II | approximate-compute, kv-side |

Self-pick top 3: CLASP, LSEI, PDAP.

## Stage 5 — Final selector output

Picked **PDAP (Idea 5)** over alternatives:
- CLASP: 1-2 week implementation lift; cheap measurement only bounds CLASP's ceiling without informing other ideas
- LSEI: refinement, not building block; saves logit-step bandwidth but doesn't change the layer-streaming problem
- UCLOCK: low cross-idea info value
- A/H/4: weight-side; PDAP is the activation-side prerequisite for stack H+4+5

PDAP is the symmetric counterpart to Round 1's weight-side finding. 2-hour experiment, structural finding either way.

## Stage 6 — Red team output

**PROCEED WITH AMENDMENT.** Three critical findings:

1. **PrefixQuant (arXiv:2410.05265, Oct 2024) decomposition blindspot**: explicit channel-wise (stable) vs token-wise (BOS/delimiters/sparse) outlier categories. ~5-15 massive-activation tokens in 4000 will contaminate aggregate Jaccard. Stage 5's go/no-go fails to stratify.
2. **Layer sampling too coarse**: {4, 8, 16, 24} on 36-layer Qwen3-4B misses final third.
3. **Single K=1% cutoff arbitrary**: should report Jaccard at multiple K (0.1%, 1%, 5%, 10%).

Plus: LLM.int8() and SmoothQuant did NOT formally measure per-token Jaccard. The measurement is novel even on previously-studied models.

**Amendments applied to SELECTED.md**:
- Stratify by token type (massive-activation vs normal)
- Layers {4, 8, 12, 16, 20, 24, 28, 32}
- Multi-K Jaccard reporting
- Most likely outcome: gray-zone bimodal (PrefixQuant pattern). Implies hybrid scheme.

## Round 2 final state

**Selected experiment**: PDAP Per-Token Dynamic Activation Outlier Pinning (amended). Queued in SELECTED.md.

**Class taxonomy after Round 2**:
- Heavy-used: prediction-internal (R1 + R2), compression-quant, compression-lr, offload-disk, runtime-fused
- Moderate: compression-tensor, compression-entropy, kv-side, prediction-shadow, prediction-historical, systems-fs
- Untouched: arch-modify (out-of-scope), train-required (out-of-scope), output-side (only LSEI), approximate-compute (only MFAP), coding-theory (only FCWR)

**Round 3 framing-diversity hint**:
- Output-side (lm_head, vocab, sampling) is genuinely under-explored — only LSEI proposed it
- Approximate-compute as a category is open
- Coding-theory beyond fountain codes is open
- Cross-domain (neuroscience, physics-inspired, distributed systems) is barely touched

**Pipeline state at end**:
- KILL_LIST.md grew by ~15 entries; total ~40 entries
- SELECTED.md has 2 entries: R1 (done with structural finding), R2 (queued, amended)
- Next: implement amended PDAP, run, then start Round 3
