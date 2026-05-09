# Stage 2 Prior-Art Filter — Reach Agent / CMR

## Verdict: SURVIVES WITH AMENDMENTS

## Mechanism 1: Cheap learned predictor produces per-token weight-access masks at inference time, no SGD

Status: PARTIAL PRE-EMPTION (heavy)

Closest prior art:
- Deja Vu (Liu et al., ICML 2023, arXiv:2310.17157): small MLP predictor on hidden state predicts contextual sparsity set per token, no retraining. Architecturally identical to π(x_t) → M_t.
- PowerInfer (SOSP 2024, arXiv:2312.12456): offline profiling labels neurons hot/cold; online predictor decides cold neuron activation per token.
- PowerInfer-2 (arXiv:2406.06282): neuron-cluster prediction with overlapped I/O on smartphones. Maps directly to "stream W^ℓ[M_t]".
- R-Sparse, COUNTDOWN: training-free block-sparse activation prediction, no retraining.

What survives: Cache-line-granularity block level (≤4 KB, 32-dim sketch) targeting NVMe alignment rather than GPU kernel sparsity. Deja Vu/PowerInfer operate on neuron columns inside GPU; do not treat mask as I/O prefetch hint for cold storage tier.

How to sharpen: Reframe as "storage-aligned block-mask predictor for NVMe prefetch scheduling." Don't claim novelty in per-token activation-conditioned gating itself — pre-empted. Novel claim: mask granularity co-designed with NVMe sector geometry (4 KB), output drives ReadFileEx call not kernel sparse-matmul.

CF9 check: Predictor assumes activation patterns at ℓ are predictable from ℓ-1. Deja Vu validates on OPT/LLaMA. R2 Jaccard cliff (K=0.1%, J=0.72) consistent. Holds for MLPs; weaker for attention.

## Mechanism 2: Per-layer tensor-network factorization with token-conditional bond dimension χ

Status: ADJACENT BUT DISTINCT (narrow novel claim)

Closest prior art:
- TensorGPT (arXiv:2307.00526): TT-decomposition of embedding layers post-training. Fixed bond dimension.
- CompactifAI / MPO-LLaMA (Multiverse 2024): MPO factorization of full weight matrices in LLaMA-2 7B, χ≈100 fixed per-layer.
- Saten (EMNLP 2025): sparse-augmented tensor networks for post-training compression. Fixed rank per layer.
- SVD-LLM (ICLR 2025, arXiv:2403.07378): truncation-aware SVD with calibration whitening; heterogeneous rank per layer.

What survives: All existing MPS/MPO/TT work uses FIXED bond dimension. No published paper makes χ TOKEN-CONDITIONAL at inference time. The genuinely new claim.

How to sharpen: Separate (i) MPS factorization of concatenated Q+K+V+O+gate+up+down weights — incrementally novel but not cleanly differentiated; (ii) inference-time selection of χ per token — genuinely new, headline claim.

CF9 check: MPS assumes bounded entanglement. AQFKV head-redundancy applies to Q only, not gate/up/down. Concatenating before factorizing mixes low-entanglement (Q) with high-entanglement (down) tensors; could inflate χ_required. "Joint MPS more efficient than per-matrix" is unverified.

## Mechanism 3: Three-tier RAM/RAM/NVMe weight residency hierarchy with predictive prefetch

Status: PARTIAL PRE-EMPTION (moderate)

Closest prior art:
- FlexGen (ICML 2023, arXiv:2303.06865): GPU/CPU/disk three-tier hierarchy with LP scheduling.
- ZeRO-Inference: layer-by-layer NVMe-to-CPU prefetch.
- HOBBIT (arXiv:2411.01433): token-level dynamic expert loading from SSD+CPU, three-level hierarchy with predictive fetch on MoE.
- DAOP (arXiv:2501.10375): data-aware offloading + predictive pre-calculation for MoE.
- DALI (arXiv:2602.03495): workload-aware expert cache on local PCs.

What survives: All prior three-tier work either targets MoE (cold tier = non-activated experts) or treats layers as indivisible (FlexGen loads entire matrices). CMR's distinguishing claim: SUB-LAYER cold tier on a DENSE model — only mask-selected blocks fetched from NVMe, not full matrix. Combined with Clock-Pro warm-tier policy for dense-model block streaming = not directly published.

How to sharpen: Lead with "sub-matrix block streaming for dense models" as key differentiator. Demote Clock-Pro to engineering choice. Headline: predictor M_t directly indexes NVMe at 4 KB granularity, random-access streaming of needed blocks only. Quantify I/O reduction vs full-matrix offload (FlexGen baseline).

CF9 check: Prefetch assumes π's prediction accurate enough that 1-2-layer-ahead reads cover actual M_t. If accuracy drops (R2 K=1% Jaccard 0.31), miss handling? Not specified. Precondition Jaccard ≥0.6 at block level for 3 tok/s — unverified for 70B.

## Mechanism 4: Integrated system on CPU-only 7.28 GiB

Status: ADJACENT BUT DISTINCT

Composition crowding: PowerInfer + FlexGen + CompactifAI ≈ 80% surface area.

What survives: CPU-only, NVMe-backed, dense 70B, sub-matrix block streaming on Ryzen with 6 GiB DRAM is an underserved niche. None of those systems run without GPU. ≥3 tok/s on this configuration is a systems contribution regardless of individual mechanism novelty.

## Cross-cutting CF9 risks

1. Streaming SVD calibration for MPS: validated for SVD/pruning on LLaMA-class. NOT well-validated for MLP gate/up/down at χ=16. Expect significant accuracy loss at low χ for MLP unless χ allowed larger there.
2. 0.3 nats target needs calibration against perplexity/downstream task degradation.
3. Clock-Pro derived from OS workloads, not neural weight access. Long-tail reuse may break it; LFU might be better.
4. Windows ReadFileEx overlapped I/O: random 4 KB reads bandwidth-limited differently. May saturate per-queue IOPS before bandwidth.

## Required amendments

1. Reframe M1 as "storage-geometry-aligned block-mask predictor for NVMe prefetch" (not contextual sparsity).
2. Lead with M2's token-conditional χ as primary algorithmic novelty.
3. Separate concatenated-weight MPS from per-matrix; test if joint is actually better.
4. Quantify I/O reduction over FlexGen baseline.
5. Add miss-handling protocol bounding worst-case latency.
6. Replace 0.3 nats with perplexity/downstream numbers comparable to prior work.
