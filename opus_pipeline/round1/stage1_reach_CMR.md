# Stage 1 Reach Agent — Conditional Manifold Routing (CMR)

## 1. Ambition Statement

**Run a 70B-class model at ≥ 3 tok/s on the 7.28 GiB Ryzen 5 7530U laptop.**

Concretely: target Qwen3-72B (or Llama-3.3-70B) decoding at 3-5 tok/s with quality within 0.3 nats of full-precision, using ≤ 6 GiB resident RAM and NVMe as a routed storage tier for weights that are *provably not needed this token*. This is a 13× improvement in effective parameter throughput per joule over our current 5.5 tok/s on Qwen3-4B.

The audacious claim: a 70B model has an intrinsic dimension of active computation per token that is closer to a 4B model's worth of bytes than to its 70B residency, and that this intrinsic dimension is measurable from the activation outlier dynamicity we have already characterized.

The leap: stop treating the model as a function over weights and start treating it as a routing problem over a learned conditional manifold whose chart we can extract without SGD.

## 2. Mechanism — Conditional Manifold Routing (CMR)

Three coupled primitives:

**(a) Activation-conditioned weight gating.** For input x_t arriving at layer ℓ, compute a cheap predictor π(x_t) → support mask M_t over W^ℓ's rows/columns/blocks. Stream only W^ℓ[M_t] from the storage tier. The predictor π is a tiny learned function (post-hoc, calibration-only) — a hash-codebook-like map from a 32-dim sketch of x_t to a sparse {0,1}^B block mask, where B is the block count of W^ℓ partitioned at cache-line granularity. The mask is computed *while the previous layer's output is still in L2* — π's footprint is ≤ 4 KB per layer.

**(b) Tensor-network factorization at the layer level, not the matrix level.** Replace the layer's full weight tensor (Q, K, V, O, gate, up, down concatenated) with a low-bond-dimension MPS whose bond dimension χ is *conditional on x_t* via π. When π says "this token is in routine-text regime," χ = 16; when π says "this token is in code/math/multilingual outlier regime," χ = 256. Average bond dimension dominates residency; peak bond dimension dominates quality. The decomposition is constructed by streaming SVD on calibration activations, never via SGD.

**(c) NVMe as the cold tier of a structured cache hierarchy.** Hot weights (say 1.5 GB of MPS cores at low χ + always-on layers like embeddings, lm_head, layer 1) live in DRAM. Warm weights (2 GB of medium-χ correction tensors) live in DRAM under a clock-pro replacement policy. Cold weights (3-4 GB of high-χ outlier-regime corrections) live on NVMe and are prefetched by π's mask 1-2 layers ahead via overlapped ReadFileEx I/O. Page-fault-rate becomes a *measured tok/s cost* with a known floor.

The math. For decoder layer ℓ acting on x_t ∈ ℝ^d, the standard computation is y = W x where W ∈ ℝ^(d×d). CMR computes:

  y ≈ W_hot x + Σ_{i ∈ M_t} W_warm[i] x + Σ_{j ∈ N_t} W_cold[j] x

where (M_t, N_t) = π(x_t). The approximation error budget is tracked per layer per token; π is calibrated so expected ε per layer × layers L = total nats < 0.3.

**Position relative to class boundary.** This is *not* CF8 MLP rank reduction — full-rank MLPs stay full-rank in CMR; what changes is which rows of W_up are touched per token, not the rank of W_up itself. This is *not* tied-lm_head SVD (CF11) — embeddings stay hot, full-rank, in DRAM. This is closest to CF1 (W_Q head-redundancy) generalized: the 16:1 head-redundancy finding is the seed observation that *redundancy in attention is conditional structure*, not weight-level low-rank, and CMR extends that intuition to the entire model by routing rather than reducing.

## 3. Why Our Findings Make This Work

Activation outlier dynamicity table is the keystone. K=0.1% (channel-static) Jaccard 0.72 says ~2 channels per layer are always important — these are the hot weights. K=1% Jaccard 0.31 says ~20 channels per layer are important for some tokens with > 30% overlap across random pairs — these are the warm/cold weights, and 0.31 is the lower bound on how much π can hit per token. K=10% Jaccard 0.24 means even at 10% of channels, dynamicity dominates — uniform-everything-every-token streaming pays for ~76% wasted bytes.

Class boundary CF10. Heterogeneity across W_Q (rank 0.63), W_K (rank 0.79), MLP (rank 1.0), tied embed (rank 0.97) means CMR's bond dimension χ should be class-conditional: attention sub-blocks compress aggressively, MLP sub-blocks compress only via routing, tied embed never compresses.

Layer-depth concentration evolution. Layers 2-19 have 5-8% channels covering 90% of outlier mass; layers 23-27 spread to 14-18%. π's job gets harder in deep layers — CMR can spend more residency budget there, less in shallow layers.

W_Q head redundancy (16 heads → ~128 effective dims). Canonical CMR success case in miniature: 16× residency reduction by recognizing head structure IS routing structure.

## 4. What Would Have to Be True

**P1: π is learnable from O(10K) calibration tokens with prediction quality > 0.5 mask-Jaccard.** Cheap test: train a 32-dim → block-mask MLP on Qwen3-4B layer 12, measure mask-Jaccard on held-out tokens. Runtime: 30 min CPU.

**P2: Conditional ε-support has block locality at 64-byte cache-line / 4 KB page granularity.** Cheap test: spatial autocorrelation of |W^ℓ x_t|_i across i, averaged over tokens. Runtime: 1 hour.

**P3: Streaming SVD on calibration activations produces bimodal MPS bond dimension χ.** Cheap test: per-layer SVD on 32K activations, plot singular value spectrum conditional on token regime. Runtime: 4 hours.

**P4: NVMe random-block read at 4 KB sustains ≥ 800 MB/s under overlapped I/O.** Cheap test: synthetic ReadFileEx benchmark. Runtime: 1 hour.

**P5: Total nat budget across all layers' CMR approximation stays under 0.3 nats vs full-precision baseline.**

## 5. Experiment Cascade

**E1 — Conditional support measurement (1 day).** Extend K-sweep dynamicity from activations to weight contributions. For Qwen3-4B layers 1, 12, 24, 32 compute per-token |W^ℓ x_t|_i over 32K tokens. Output: support-size distribution + spatial autocorrelation. Go: support occupies ≤ 30% of W elements at ε giving < 0.05 nat/layer cost.

**E2 — Predictor π training and Jaccard validation (1 day).** Train 32-dim sketch → block-mask predictor on E1's per-token support data, per layer. Held-out mask-Jaccard ≥ 0.5 with < 4 KB per-layer parameter budget.

**E3 — Per-layer conditional MPS construction (3 days).** Streaming SVD on calibration activations to produce variable-χ MPS factorization per matrix family per layer. Measure ΔNLL per layer per regime. Go: ΔNLL < 0.05 nat/layer at residency reduction ≥ 5×.

**E4 — Storage tier integration on Qwen3-4B (4 days).** Build hot/warm/cold partition with E2 predictor + E3 factorizations, end-to-end. Measure tok/s and ΔNLL. Go: ≥ 8 tok/s at < 0.3 nat ΔNLL with < 4 GB DRAM residency.

**E5 — 70B scale-up (1 week).** Run calibration pipeline on Qwen3-72B / Llama-3.3-70B using temporary cloud instance for calibration only (no SGD, just SVD on activations — fits within scope). Deploy resulting CMR routing on Ryzen laptop with model weights pre-partitioned to NVMe. Target: ≥ 3 tok/s.

Total cascade: ~3 weeks of clock time, every stage produces a structural finding even on failure.

## 6. What I Am NOT Proposing

- Not CF8 MLP rank reduction in any form
- Not tied-lm_head SVD (CF11)
- Not standard KV cache compression (orthogonal)
- Not SGD retraining (π calibration-fit, MPS streaming-SVD-fit)
- Not iGPU compute offload (separate engineering track)
- Not generic mixture-of-experts (MoE is router + expert blocks at training time; CMR is router + structured weight access at inference time)
- Not naïve weight sparsification (block-locality and conditional support are load-bearing claims)
- Not linear or kernel attention (attention stays exact softmax; bytes touched per attention call change)

The audacious claim: 70B's "true working set per token" is closer to 4B than to 70B, and we can extract the routing structure to prove it without retraining. The 13× ambition is bandwidth math: if 70B residency drops from 35 GB (Q4) to 5 GB hot + routed cold, with cold tier hit rate matching π's Jaccard, effective per-token bandwidth = 5 GB hot DRAM + 1-2 GB NVMe, at measured 11.5 GB/s DRAM + 3 GB/s NVMe lands in 3 tok/s neighborhood.
