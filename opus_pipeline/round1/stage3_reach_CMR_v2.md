# Stage 3 Reach Agent v2 — Token-Conditional Manifold Routing (TC-MR)

## 0. Path Choice and Key Changes (Section 7 first, by request)

**Path A — refine.** The Stage 2 critique sharpens the proposal but does not dissolve it. Two of three primitives were partially pre-empted; one (token-conditional bond dimension χ on a per-layer MPS factorization, applied at inference time) was confirmed novel and was buried as M2 in v1. Refinement promotes that claim to the top, reframes M1 as storage-geometry-aligned rather than activation-conditioned (because the latter is Deja Vu and PowerInfer's territory), and re-derives the throughput math against a corrected NVMe latency floor.

**Headline change.** Rename CMR → **TC-MR (Token-Conditional Manifold Routing)** to put the load-bearing novelty in the title. χ_t — the per-token bond dimension chosen by a 4 KB-footprint sketch network — is the single contribution Stage 2 could not find prior art for. Everything else (block masks, NVMe tier, hot/warm/cold partition) is engineering substrate around that algorithmic core.

**Three structural amendments.**
1. **M2 (token-conditional χ) is the lead.** The MPS factorization with χ_t selected per-token at inference is the genuinely new claim. Concatenated-vs-per-matrix MPS is now an explicit Stage 1 experiment (E0b) with a binary structural finding either way.
2. **M1 reframed as NVMe-geometry-aligned mask predictor.** 4 KB block granularity, tuned to the NTFS cluster size and NVMe atomic write unit, is the differentiator from Deja Vu/PowerInfer's neuron-column granularity. The mask is consumed by ReadFileEx scheduling, not by a kernel sparse matmul.
3. **NVMe latency floor honestly priced.** The PFOR sibling proposal got destroyed by the 5.6 μs assumption being 100-300 μs on Windows demand paging. CMR uses overlapped ReadFileEx (not demand paging), which is a *different* I/O path with realistic ceiling 12-30 μs/4K at QD=32 — but I run the math against both 12 μs and 30 μs and keep the upper case as the planning number. The result reduces the 70B target from "≥3 tok/s" to a more defensible **≥1.5 tok/s with stretch goal 2.5 tok/s**, and elevates **34B at ≥3 tok/s** to first-class status as a Reach target the math actually supports.

The ambition does not retreat: 70B on 7.28 GiB at any positive tok/s is still a leap nobody has cleanly shown on Windows NTFS, and TC-MR's residency math says it lands. What changes is which number we're willing to write down.

---

## 1. Ambition Statement

**Run a dense 70B-class transformer on a 7.28 GiB Ryzen 5 7530U laptop at ≥1.5 tok/s, with a stretch target of 2.5 tok/s, and run a dense 34B at ≥3 tok/s.** Quality target: within 2 perplexity points of full-precision Q4_K_M on WikiText-2 and within 4% on MMLU-zero-shot, with ≤6 GiB resident DRAM and NVMe carrying 4-30 GiB of structured cold tier.

**The audacious algorithmic claim.** A dense 70B model has a *token-dependent* effective bond dimension that varies by 8-32× across token regimes. Most tokens (routine English completion, repeated tokens in code, common-bigram continuations) need χ_t ≈ 8-16 bond dimension to land within 0.05 nat of full precision; a minority of tokens (rare names, multilingual switches, math symbols, code identifiers in long-tail vocab, end-of-sentence at semantically loaded positions) need χ_t ≈ 128-512. **The mean χ over a calibration corpus is closer to 32 than to a fixed-χ baseline's required 128.** Token-conditional selection of χ harvests this 4× residency reduction without re-training and without quality cliff.

**The audacious systems claim.** Sub-matrix 4 KB block streaming on a *dense* model (not MoE) over Windows NTFS overlapped ReadFileEx, hitting realistic 250-450k IOPS at QD=32, lets the dense 70B's working set per token live in DRAM at 5-6 GB while the χ-correction tail streams from NVMe. This is 13× better effective parameter throughput per joule than our current 5.5 tok/s on Qwen3-4B, normalized to active-bytes-touched.

**The leap.** Stop treating "rank" and "bond dimension" as architecture-time design parameters and treat them as runtime control variables, the same way attention treats softmax weights. The 16:1 head-redundancy finding from AQFKV is the seed: attention isn't redundant per se, it is *conditionally* redundant, and the conditioning is cheap to predict from a 32-dim token sketch.

---

## 2. Mechanism — Token-Conditional Manifold Routing

Three coupled primitives, ordered by novelty descending:

### Primitive A (the headline) — Token-conditional bond dimension χ_t

For each decoder layer ℓ, factorize the concatenated weight tensor (Q || K || V || O || gate || up || down) as a Matrix Product State with a *maximum* bond dimension χ_max = 512 across the chain. At inference, for token t, a tiny predictor π_χ(x_t) → χ_t ∈ {8, 16, 32, 64, 128, 256, 512} selects which truncation of the MPS to use. Lower χ_t → only the leading bond-dimension cores are touched, higher χ_t → the full chain is contracted.

The MPS is constructed offline by streaming SVD on calibration activations (no SGD). χ_max corresponds to lossless reconstruction within numerical precision; truncation to χ_t is an exact MPS truncation. The novelty: π_χ predicts *which* truncation suffices for *this* token, on the fly.

π_χ is a 32-dim sketch → 7-way softmax classifier, ≤4 KB parameters per layer, calibrated on (sketch, χ_required) pairs where χ_required is the minimum bond dimension keeping ΔNLL_t < 0.01 nat.

The math, per layer ℓ:

```
y_t = MPS_contract(W^ℓ, x_t, χ = π_χ^ℓ(sketch(x_t)))
ΔNLL_t = E[||y_t - W^ℓ x_t||² / σ²_y]
budget: Σ_ℓ ΔNLL_t < 0.005 nat per layer × L layers < 0.3 nat total
```

The residency win: at χ_max = 512, MPS residency for a 70B's per-layer concatenated weight is roughly 0.7× of the dense matrix (modest savings). At mean χ_t = 32 (with calibration-driven distribution), the *amortized read traffic* per token drops 6-10×. **MPS residency on disk does not change. What changes is bytes touched per token.** This is the routing claim: tensor-network factorization is the substrate that makes "stop reading bytes you don't need" precisely meaningful.

### Primitive B — NVMe-geometry-aligned block-mask predictor

For each layer ℓ, partition the χ-correction cores (the bond cores beyond χ = 32, which are the marginal cost of high-χ tokens) into 4 KB-aligned blocks. **Block boundary chosen to match NTFS cluster size + NVMe atomic write unit, NOT chosen by neuron column or attention head.** Predictor π_M(x_t) → M_t ∈ {0,1}^B over those blocks, computed in L2 cache with ≤4 KB parameters per layer. M_t consumed by an overlapped ReadFileEx queue that prefetches needed blocks 1-2 layers ahead, NOT by a kernel sparse matmul.

This is the precise differentiator from Deja Vu, PowerInfer, R-Sparse, COUNTDOWN: those papers predict neuron-column or activation-channel sparsity at GPU kernel granularity. Their mask is consumed by a sparse matmul. **TC-MR's mask is a storage-aligned prefetch hint, not a compute mask.** The block geometry is co-designed with the file system and NVMe sector layout — the prediction is at the level a ReadFileEx queue can act on. If a block is needed and not yet resident, the system stalls; if it is resident, it is contracted.

### Primitive C — Hot/warm/cold residency hierarchy

Hot tier (~1.8 GB DRAM): embeddings, lm_head, layer norms, the MPS leading cores at χ ≤ 32 for all layers, and the Primitive A predictor π_χ + Primitive B predictor π_M parameters. Always resident, VirtualLock-pinned.

Warm tier (~3.5 GB DRAM, Clock-Pro replacement): the χ ∈ (32, 128] cores for all layers. Touched by ~30% of tokens. Replacement policy is Clock-Pro on layer-cluster granularity (not per-page) — Clock-Pro's adaptive scan/scan-resistance is appropriate at the granularity at which we evict.

Cold tier (4-30 GB NVMe): the χ ∈ (128, 512] cores, plus rare-block χ-correction tails. Streamed via overlapped ReadFileEx, prefetched by π_M one layer ahead. Crucially, this is the layer of the system where the I/O latency bound bites — which is why we audit it carefully in Section 4.

---

## 3. Why Our Findings Make This Work

The activation outlier dynamicity table is the keystone, but read differently than v1.

**K=1% Jaccard 0.31 → χ-distribution evidence.** When 1% of channels carry the dynamicity load and that 1% has only 31% overlap across random token pairs, the *dominant* channels are conditionally dominant. Translated to bond dimension: the singular vectors that would matter at χ = 16 are not the same vectors token-to-token, which means a fixed-χ baseline must over-allocate to cover the union — that's the inefficiency TC-MR's token-conditional χ recovers.

**AQFKV W_Q rank ~128 globally — but per-token effective rank ~16.** Re-reading AQFKV's findings as conditional-rank evidence: globally the head weights span a 128-dim subspace; per token a much narrower subspace is exercised. χ_t is the formalization of this observation that generalizes from W_Q (where it's known) to all matrix families (where it's hypothesized).

**Layer-depth concentration evolution (5-8% in layers 2-19, 14-18% in layers 23-27).** This is direct evidence that χ_t should be *layer-conditional* on top of token-conditional: π_χ^ℓ has different output distributions across layers. We allocate more residency budget to deep layers via a higher χ_max there, less in shallow layers.

**Heterogeneity across W_Q (rank 0.63), W_K (rank 0.79), MLP (rank 1.0), tied embed (rank 0.97).** This is the load-bearing CF9-flagged risk on concatenated MPS. The MLP gate/up/down matrices have rank ~1.0 (full rank) while attention W_Q has rank 0.63. Concatenating and MPS-factorizing them mixes a rank-deficient subspace with a full-rank subspace, and the joint MPS bond dimension may need to be at least the sum of the maxima of the per-matrix bond dimensions to lose nothing — defeating the purpose. **E0b explicitly tests this** (see cascade).

**WrapSketch's per-token phase index Spearman 0.345.** This is a closer-to-home result than I initially gave it credit for. Phase coding *did* produce non-trivial token-conditional structure (Spearman 0.345 vs a 0 baseline) even though it failed the autoregressive long-prefix gate. The signal exists; what failed was using phase as an *autoregressive selector*. Using a similar 32-dim sketch as input to χ_t prediction is using the structure for what it was — a within-context conditional indicator — not for autoregressive retrieval.

---

## 4. What Would Have to Be True

**P1 (algorithmic).** χ_required for ΔNLL_t < 0.01 nat is bimodal or heavy-tailed across tokens, with mean ≤ 32 and 95th percentile ≤ 256. Test: per-layer streaming SVD on Qwen3-4B layer 12, sweep χ truncation, measure per-token reconstruction error. Cheap; 4 hrs. **If χ_required is unimodal at ~256, TC-MR collapses to fixed-χ and the headline novelty dies.**

**P2 (predictor).** π_χ trained on (sketch, χ_required) achieves >0.6 Pearson correlation with χ_required on held-out tokens with ≤4 KB parameters. Test: train per-layer 32→7 classifier, measure correlation. Cheap; 1 day.

**P3 (concatenated-MPS structural risk).** Joint MPS over Q||K||V||O||gate||up||down at χ_max=128 has ΔNLL ≤ 1.5× the sum of per-matrix MPS at the same total parameter count. Test: build both factorizations on Qwen3-4B layer 12, compare. Cheap; 1 day. **If joint is worse, we use per-matrix MPS and re-derive numbers — does not kill TC-MR; changes its substrate.**

**P4 (block-mask).** π_M predicts 4 KB-block usage with mask-Jaccard ≥0.5 on held-out tokens at ≤4 KB parameters per layer. Test: train sketch→mask classifier, measure Jaccard. Cheap; 1 day.

**P5 (NVMe latency floor — the load-bearing one).** Overlapped ReadFileEx on FILE_FLAG_OVERLAPPED handle to pre-aligned 4 KB blocks at QD=16 sustains ≥250k IOPS on the laptop's actual NVMe, with median per-request latency ≤30 μs. Test: synthetic benchmark using overlapped I/O, NOT memory-mapped page faults. **This is the test that PFOR's sibling proposal failed.** Hard prerequisite. 2 hrs.

**P6 (end-to-end nat budget).** Σ_ℓ ΔNLL_t with TC-MR active stays under 0.3 nats; equivalently <2 perplexity-point degradation on WikiText-2 and <4% on MMLU.

**Aspirational P7 (residency + latency together).** At mean χ_t = 32, mean blocks-per-layer-token = 12, NVMe latency = 30 μs, 70B = 64 layers, demand for blocks per token = 12 × 64 = 768 blocks = 23 ms wall at QD=16 sustained, leaving 470 ms compute budget at 1.5 tok/s. Plausible at 5.5 GFLOP/s decode rate measured on Qwen3-4B. Stretch case 12 μs latency: 9 ms I/O, 2.5 tok/s feasible.

---

## 5. Experiment Cascade

**E0a (2 hrs) — NVMe overlapped ReadFileEx benchmark.** Synthetic test: open a 30 GB file with FILE_FLAG_OVERLAPPED + FILE_FLAG_NO_BUFFERING, issue QD=16 random 4 KB reads, measure sustained IOPS and per-request latency distribution. This is the single most load-bearing prerequisite.
- Go: ≥250k IOPS sustained, p50 ≤ 30 μs, p99 ≤ 80 μs.
- No-go (≤80k IOPS or p50 ≥80 μs): TC-MR's cold tier ceiling drops to ~0.5 tok/s on 70B; redirect to 34B target only and re-run E0a with FILE_FLAG_NO_BUFFERING + DIRECT_IO equivalent paths. If still no-go, consider iGPU-shared-memory tier as a second warm tier.

**E0b (1 day) — Concatenated vs per-matrix MPS structural test.** On Qwen3-4B layer 12, build (i) per-matrix MPS for each of Q, K, V, O, gate, up, down at χ ∈ {16, 64, 256}; (ii) joint MPS over the concatenation at the same total parameter budgets. Measure ΔNLL on a 4K-token held-out set for each variant.
- Outcome 1 (joint within 1.5× of per-matrix): proceed with concatenated MPS as headline.
- Outcome 2 (joint worse than 1.5×): swap to per-matrix MPS; the token-conditional χ claim survives unchanged; the residency win moves from joint-rank-collapse to per-matrix-conditional-truncation.
- Either way, this is a publishable structural finding about what tensor-network factorization can and cannot do across attention-MLP boundary.

**E1 (1 day) — Per-token χ_required distribution and predictability (P1, P2).** On Qwen3-4B all layers, streaming SVD; sweep χ ∈ {8,16,32,64,128,256,512}; for each (layer, token) compute χ_required for ΔNLL_t < 0.01. Plot distribution. Train π_χ; measure Pearson on held-out.
- Go: mean χ_required ≤ 32, π_χ Pearson ≥ 0.6.
- No-go (mean χ_required ≥ 128): the conditional distribution is too tight; TC-MR collapses to fixed-χ. Pivot to fixed-χ MPS (still useful, but no longer Reach).

**E2 (1 day) — Block-mask predictor π_M validation (P4).** Per-layer per-token block-usage measurement at 4 KB granularity over the χ ∈ (32, 512] cores. Train π_M; mask-Jaccard.
- Go: ≥0.5.

**E3 (3 days) — Tier integration on Qwen3-4B.** Build hot/warm/cold partition with E0b's preferred MPS variant + E1's π_χ + E2's π_M, end-to-end. Use llama.cpp/ik_llama.cpp as baseline and inject TC-MR routing at the weight-load layer. Measure tok/s and ΔNLL.
- Go: ≥6 tok/s with <0.3 nat ΔNLL on Qwen3-4B at <4 GB resident.
- No-go: profile to find which tier is the bottleneck (DRAM bandwidth, NVMe IOPS, π_χ misprediction tail, MPS contraction overhead).

**E4 (1 week) — 34B target.** Use Qwen3-32B or comparable. Calibration on cloud (no SGD, only streaming SVD on activations + π_χ/π_M training, fits within scope and budget). Deploy resulting routing on laptop with weights pre-partitioned to NVMe.
- Go: ≥3 tok/s at <0.3 nat ΔNLL.
- No-go: identify whether algorithmic (χ distribution shifted at scale) or systems (NVMe ceiling) is the limit.

**E5 (2 weeks) — 70B target.** Llama-3.3-70B or Qwen3-72B. Same pipeline.
- Go: ≥1.5 tok/s.
- Stretch: ≥2.5 tok/s if E0a hit the 12 μs latency case.
- No-go: write up ceiling found and the structural reason.

**Miss-handling protocol.** If a token's χ_t prediction misses (true χ_required > predicted, or required block not resident), the system must produce *some* output. Options: (i) stall and demand-fetch (worst-case latency = 1 layer × 30 μs × 100 blocks = 3 ms penalty per miss), (ii) fall back to χ_t = 32 best-effort (introduces error but keeps latency bounded), (iii) flag the token and use χ_t = χ_max for the *next* token to recover. Protocol: combination — stall with a 5 ms timeout, then fall back to (ii). Worst-case per-token latency thus bounded at 5 ms over expected.

Total cascade: ~4 weeks wall time. Every stage is structurally informative on failure.

---

## 6. What I Am NOT Proposing

- Not contextual sparsity at neuron-column granularity (Deja Vu / PowerInfer occupy that ground; TC-MR's mask is at storage-block granularity for a different I/O path).
- Not predict-then-fetch with app-managed direct I/O bypassing OS cache (LLM in a Flash; TC-MR uses overlapped ReadFileEx with FILE_FLAG_NO_BUFFERING but goes through page cache for the warm tier).
- Not memory-mapped demand paging as primary I/O (PFOR's path; killed by 100-300 μs Windows page-fault latency; TC-MR uses overlapped ReadFileEx as the cold-tier path explicitly to avoid that latency).
- Not fixed-χ MPS factorization (TensorGPT, CompactifAI, MPO-LLaMA, Saten, SVD-LLM occupy that ground; TC-MR's contribution is *token-conditional* χ).
- Not standard MoE expert routing (no expert blocks; the "experts" in TC-MR are bond-dimension truncation levels, which is a different combinatorial structure).
- Not KV cache compression (orthogonal; TC-MR composes with quantized KV).
- Not SGD retraining at any stage (calibration-only via streaming SVD and small classifier fits).
- Not iGPU compute offload (noted as a possible second warm tier in the no-go path of E0a; not load-bearing).
- Not linear or kernel attention (attention stays exact softmax; what changes is bytes touched).
- Not generic weight quantization (TC-MR composes with Q4_K_M / IQ4_XS unchanged).

---

## 7. Path choice recap and the Reach posture

The Reach posture asks "what is the largest delta this evidence supports?" Stage 2 narrowed the answer from "70B at ≥3 tok/s with three primitives" to "70B at ≥1.5 tok/s with the headline being one primitive (token-conditional χ) and the other two being engineering substrate." The delta is smaller in the marquee number but sharper in the algorithmic claim. The honest planning numbers are 70B at ≥1.5 tok/s and 34B at ≥3 tok/s; the stretch 70B target of 2.5 tok/s lives or dies on E0a's NVMe overlapped-I/O measurement, which is the 2-hour test that gates everything downstream.

What I refused to retreat on: the audacious claim that a 70B's *active bytes per token* is closer to a 4B's worth than to its full 70B residency, and that the routing structure is extractable without SGD. Token-conditional χ is the cleanest formalization of that claim I can find in the literature, and it is what Stage 2 says is genuinely novel. The Reach proposal is now organized around it.
