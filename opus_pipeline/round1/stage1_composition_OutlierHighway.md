# Stage 1 Composition Agent — The Outlier Highway

## Ambition
70B-class model at 3-5 tok/s on 7530U laptop by treating activation outlier dynamics as routing protocol over storage hierarchy. K-dependent Jaccard cliff (K=0.1% J=0.72 vs K=1% J=0.31) is the model telling us where to draw storage/compute boundary. Static ~2 channels/layer = persistent residual highway in L2 cache; dynamic ~20-200 channels = streamed packet bus through head-shared subspace exposed by AQFKV; full-rank MLP bulk lives on NVMe, fetched only when packets demand.

Quantitative target: 70B at 0.55 bpw NanoQuant = 5.35 GB residency. Naive sequential streaming gives ~1.8 s/token (NVMe-bound). By collapsing 90% of MLP fetches into head-shared subspace and serving channel-static highway from L3, target effective bandwidth amplification of 4-6×, putting us at 0.3-0.5 s/token = 2-3 tok/s baseline, stretch goal 5 tok/s.

## Mechanism

Decompose residual stream:
  h_ℓ = h_hwy + h_bus + h_bulk

- h_hwy: static outlier basis S_ℓ^0.1% (~2 channels, J=0.72). Token-invariant. Persistent, layer-local, fits L1d (32 KB/core).
- h_bus: dynamic outlier basis D_ℓ^1% (~20 channels, J=0.31). Token-dynamic *which* but bounded *how many*.
- h_bulk: residual ~2026-channel subspace, low-magnitude, full-rank in MLP weights but low information rate per token.

**Coupling claim (the composition)**:
  dim(D_ℓ^1% ∩ rowspan(W_Q^shared)) ≥ 0.6 × |D_ℓ^1%|

If both R2 and AQFKV findings hold on same model, head-shared subspace IS natural codomain for outlier routing — outliers carry cross-head signal that motivates head-redundancy.

**Information-theoretic frame**:
  I(h_hwy;y) + I(h_bus;y) ≥ 0.85 × I(h;y)
with cardinality ≤ 0.011 × d. Rate-distortion: 1.1% of channels carry 85% predictive information.

**Primitives**:
1. Static highway: AVX2 broadcast-multiply-accumulate on 2 channels in L1d. ~16 bytes/layer × 28 layers = 448 bytes — single cache line group.
2. Dynamic bus: PEXT/PDEP-based sparse gather over 20 channels per token, routed through W_Q^shared (128-dim) before scatter into rest of model. 2-cycle PEXT on Zen 3.
3. Bulk: NVMe-resident, fetched only when bus signals demand. ReadFileEx overlapped I/O, 4 KB pages aligned to channel groups.

**Position**: Not weight compression (CF12, R3, R4 forbid SGD-free aggressive weight rank reduction). Activation routing AS storage layout. Lives at intersection of activation compression, attention head sharing, OS-level memory hierarchy.

## Why findings make this work
- R2 K-dependent cliff (0.72 → 0.31): qualitatively distinct populations, not continuum. Cliff implies two regimes with different physics → discrete tiers, not gradients.
- R2 layer-2-19 vs 23-27 (5-8% vs 14-18%): highway wider in late layers. Bus traffic anti-correlated across layer stack, smoothing aggregate L3/NVMe queue depth.
- AQFKV W_Q K=128 ΔNLL+0.98 at 8×: 16 heads → 128 dims = exact dimensionality budget for dynamic bus (~20 outlier channels with 6× ECC redundancy).
- CF12 tied-embed catastrophic at K=50%: cannot compress at WEIGHT level. But if outliers carry 85% of I(h;y), embedding's full-rank-ness is driven by outlier-channel sensitivity. Embedding writes preferentially into static highway. Testable.
- R3/R4 MLP full-rank: forbids low-rank approximation of MLP WEIGHTS. Permits low-rank approximation of MLP ACTIVATIONS entering W_down (firing-rank dominance is in activations, not weights).
- R6 SDZC layer-1 anomaly (36% foldable): replace layer 1 with structurally simpler operator writing into static highway initialization.

## Preconditions

P1: Static-dynamic alignment. dim(D_ℓ^1% ∩ rowspan(W_Q^shared)) ≥ 0.6 × |D_ℓ^1%|. Test: outlier-channel indicator vector per token (top-1% by |activation|) for 1000 tokens; project W_Q rows onto SVD of indicator matrix; ≥ 60% energy in head-shared subspace. ~10 min.

P2: Information concentration. I(h_hwy ∪ h_bus; y) / I(h; y) ≥ 0.85. Test: zero non-highway/non-bus channels, measure ΔNLL on 5K-token validation. If ΔNLL < +0.5, threshold cleared. ~30 min.

P3: Cross-token bus stationarity. PEXT mask updates not bottleneck. Test: per-token Hamming distance between consecutive bus masks. Median ≤ 4 channel flips. ~5 min.

P4: Layer-bus coherence. Bus channels at ℓ predict bus channels at ℓ+1 for speculative prefetch. Test: cross-layer Jaccard of bus masks ≥ 0.4. ~15 min.

P5: NVMe queue depth tractability. Bulk fetches per token fit Ryzen NVMe random-read IOPS at QD32. Test: count weight-page touches under simulated bus routing on CPU profiler trace; multiply by token rate. <50K random 4KB reads/sec safe. ~1 hour.

## Experiment cascade

**E1 (2 hrs): Outlier-AQFKV alignment.** Compute static outlier set (J ≥ 0.7 across tokens) and dynamic outlier set (top-1% per token) for layers 2-19 and 23-27 of Qwen3-1.7B. Compute alignment with W_Q head-shared subspace via SVD overlap.
- Go: alignment ≥ 0.6 at majority of layers → E2
- No-go: alignment < 0.3 → outlier basis and head-shared basis ORTHOGONAL — clean decomposition theorem ("attention shared subspace and activation outliers are independent axes"), pivots to E2' decomposed routing.

**E2 (4 hrs): Three-tier ablation.** Implement highway/bus/bulk decomposition in ik_llama.cpp inference with static set fixed, dynamic set top-1% per token, bulk zeroed in attention input only (keep MLP intact for clean signal). ΔNLL on 5K wiki tokens.
- Go: ΔNLL < +0.5 → P2 cleared, E3
- No-go: ΔNLL > +1.5 → information more spread than R2 implies. Finding: "outlier-channel mutual information saturates at K=X%, not K=1%" — calibrates K for downstream NanoQuant work.

**E3 (1 day): Layer-1 SDZC + highway bootstrap.** Replace Qwen3-1.7B layer 1 with learned-free folded operator writing 36% foldable component into static highway slots. End-to-end NLL.
- Go: ΔNLL ≤ +0.3 → R6 confirmed compatible
- No-go: ΔNLL > +1.0 → layer 1 anomaly real but not highway-aligned

**E4 (2-3 days): NVMe-resident bulk prototype.** Bus-driven NVMe layer prefetch for MLP weights. Bus mask at ℓ triggers async fetch of layer ℓ+1 MLP weight pages. Run Qwen3-4B with simulated 70B residency profile.
- Go: ≥ 2 tok/s with ΔNLL ≤ +0.5 vs full-RAM baseline
- No-go: NVMe queue saturates → P5 violated; finding tells us exact bus hit-rate needed

**E5 (1 wk, conditional on E1-E4): 8B + Untied lm_head extension.** Repeat E1 measurements on Qwen3-8B (untied lm_head per CF12). Test whether untied lm_head's "concentrated weights" prediction (Godey & Artzi) gives stronger highway signal.
- Go: highway/bus info concentration ≥ 0.85 at 8B trends upward → 70B extrapolation confidence
- No-go: degrades with scale → small-model-specific; useful for Qwen3-4B deployment, kills 70B aspiration

## What I am NOT proposing
- Not weight low-rank for MLP (R3/R4 forbid)
- Not lm_head/embedding compression (CF12)
- Not per-head W_Q K=64 (AQFKV NO-GO at +1.53)
- Not SGD retraining
- Not page-capsule revival (dead per WrapSketch state)
- Not generic linear attention or softmax replacement
- Not RAOK collision (RAOK = value quantization; highway = channel-set routing; compatible distinct, compose multiplicatively)

The mechanism's value: nobody who measured only R2 would predict AQFKV alignment; nobody who measured only AQFKV would expect K-dependent cliff to be storage boundary; nobody looking at CF12 would see embedding's full-rank-ness as evidence FOR routing scheme. Three findings on same model on same hardware jointly imply the model is telling us how to lay it out.
