# Side Thread: HoBRA — Holographic Boundary Reconstruction Attention

Investigative thread on attention reformulation from genuinely abstract angles. Opus agent landed 2026-05-08.

## Framing choice

Considered all four families (quantum/tensor networks, statistical mechanics, information geometry, coding theory). Quantum/tensor-networks dominates because:

**AQFKV's 16/128 ratio is the EXACT signature of a tensor network with low cross-head bond dimension and full local Hilbert dimension per head.** Not metaphor — structural match to MERA/MPS bond-cut behavior. The 128/16 ratio, R2's K-dependent layered Jaccard, and M-NVMe's 1 MB extent become THE SAME PARAMETER in the math: bond dimension, RG-layer index, and bulk-tile size.

## Mechanism: HoBRA (Holographic Boundary Reconstruction Attention)

Replace `softmax(QK^T/√d_h)V` per layer with two-tier tensor network contraction:
- **Shared cross-head bulk tensor** of bond dimension χ ≈ 128 (AQFKV-confirmed) carrying inter-head correlations
- **Per-head boundary tensors** of full local rank 64-128 carrying head-specific computation

Standard attention factorizes per-head: heads never talk during attention, only at output projection W_O. AQFKV says: across heads, union of Q subspaces has rank ≈ d_h, not H · d_h. So there exists global basis B ∈ R^{(H·d_h)×χ} with χ ≈ 128 such that:

  Q[h,t,i] ≈ Σ_α U[h,i,α] · Z[t,α]

where Z is **boundary tensor** (token × bond) and U[h,:,:] is per-head **bulk leg**. Same for K with χ_K ≈ 256 (asymmetric bonds — W_K has higher r_99/d than W_Q, K=512 ΔNLL=+0.29 anchors this).

Attention as bulk-boundary contraction:

  A[h,t,t'] = softmax_{t'}(1/√d_h · U[h] · G[t,t'] · V[h]^T)

where **bulk Gram tensor** G[t,t'] = Z[t,:] · W[t',:]^T ∈ R^{χ × χ_K}.

This is the MERA pattern: coarse-grained boundary-to-boundary correlation G computed ONCE per layer, then each head reconstructs its own attention pattern from G via cheap per-head bulk legs U, V. Holographic aspect: G lives in bulk and is shared; U[h], V[h] are boundary disentanglers recovering per-head detail.

**Why not just low-rank Q/K**: low-rank alone is the linear-attention disaster. Crucial difference: **softmax happens on reconstructed per-head A[h,t,t'], not on the bulk G**. Bulk is linear; boundary is non-linear. Matches MERA exactly (linear isometries in bulk, non-linear "physical" layer at boundary). And matches AQFKV: subspace is shared, but per-head computation needs full softmax expressivity in 64-128 dims.

## Extent-aligned bulk storage (M-NVMe hook)

Bulk tensors Z, W, Y are token-major. Pack as: stripe across tokens such that one 1 MB extent holds exactly 1MB / (χ·2 bytes) = 4096 tokens of one component for one layer. Result: **single first-page fault (137 μs) prefetches 4096 tokens of bulk state**, then 1.9 μs per subsequent token-page. For 4096-token chunked-prefill: ~12 extents per layer total.

## R2 PDAP integration

Top-0.1% channels (Jaccard 0.72) are channel-static = the channels that should live in bulk basis (slow modes shared across heads/tokens). Top-1% to 10% (Jaccard 0.31 → 0.24) are head-specific boundary detail. Bond χ should cover static-channel subspace; everything else stays in per-head residual d_head_residual ≈ 32.

## Quantitative target on Qwen3-1.7B

- KV-cache: from `2·L·n_heads_kv·d_head·n_tok` to `L·χ·n_tok + L·n_heads_kv·d_head_residual·n_tok` with χ=128, d_head_residual=32 → **~2.0-2.4× KV reduction at ΔNLL ≤+0.10**
- Compute: bulk contraction O(χ²·n_tok) shared across heads; per-head residual O(d_head_residual·n_tok). Per-layer attention FLOPs: **~0.55× baseline**
- I/O: bulk extent-aligned → ~3.5-4× effective prefetch amortization

Combined: **~2× KV memory + ~1.8× attention wall-clock** on Ryzen 5 7530U at ΔNLL ≤+0.10. 70B extrapolation: χ=128 vs d_model=8192 = 64× cross-head compression of dominant memory term.

## Cascade

E1 (1 hr, go/no-go): bond-spectrum confirmation. SVD of stacked Q tensor across 16 heads on 1024-token prefix, layer-by-layer. Go: knee ≤144 in ≥24/28 layers. No-go: any layer needs χ>192 → bond hypothesis dead.
E2 (2 hr): static R2 channels = bulk basis. Cosine of static-channel indicators with top-χ singular vectors from E1. Go: mean cosine ≥0.4.
E3 (4 hr): replace one layer with HoBRA, measure ΔNLL. χ_Q=128, χ_K=256, χ_V=128, d_head_residual=32. Go: ΔNLL ≤+0.05.
E4 (8 hr): all-layer HoBRA + extent-aligned storage. Go: KV ≤0.55× AND wall-clock ≤0.65× AND ΔNLL ≤+0.10.
E5 (3 hr): 70B extrapolation feasibility check. Go: χ ≈ d_head holds for 7B; projected 70B KV savings ≥6×.

Total: ~18 hours. Ryzen-feasible.

## Headline pitch

**The AQFKV 128/16 ratio is structurally identical to a MERA bond dimension χ for the cross-head leg.** That fact predicts χ_K=256 from W_K's r_99/d=0.79 — a number we haven't yet measured but can in 15 minutes. If E1 finds χ ≤144 across most layers, we have a tensor-network reformulation of attention that wasn't on anyone's radar. If it finds χ >192, the framing is dead and we walk away in an afternoon.
