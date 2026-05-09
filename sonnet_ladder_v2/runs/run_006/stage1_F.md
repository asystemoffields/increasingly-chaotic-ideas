# Stage 1 — Orientation F (First-Principles) — Run 006

**Kill acknowledgments**: v2-CHEAP-TEST-001 (cross-layer W_Q stacking dead at class level); v2-S3-R004-001 (arbitrary O(d) rotation breaks RMSNorm γ — not O(d), only signed-permutation survives); within-layer Q/K joint subspace (PALU+TransMLA pre-empted; run 005 F1-WQWK-JOINT proposed canonical correlation; F-SPECA run 004 already proposed within-layer shared basis — both proposed, results pending, do not re-propose); softmax shift-invariance × RoPE (run 003 CF9 kill).

No idea below touches: cross-layer W_Q basis sharing, arbitrary residual-stream rotation, within-layer W_Q/W_K joint SVD, softmax row-shift under RoPE.

---

## F6-WALIGN — W_up / W_gate Row Alignment Fraction (WALIGN)
**F / Track A**

### Mechanism
Track A — arch-transposition. For each neuron i in an FFN layer, the SwiGLU contribution is silu(W_gate[i,:]·x) · (W_up[i,:]·x). The two row-vectors W_gate[i,:] and W_up[i,:] are independent d_model-dimensional vectors in general. First-principles question: for what fraction of neurons i is W_up[i,:] ≈ λ_i · W_gate[i,:] (i.e., they project from the same residual-stream direction, differing only by a scalar scale)? When alignment holds exactly, the neuron's output collapses: silu(z) · λ_i z = λ_i · z · σ(z) · z (a scalar function of a single linear projection z = W_gate[i,:]·x). Such an aligned neuron requires only ONE row vector (not two) at inference — the pair (W_gate[i,:], W_up[i,:]) is replaced by (w_i, λ_i) where w_i = W_gate[i,:] and λ_i is a stored scalar. The algebraic identity is exact for perfectly aligned neurons: no approximation, no retraining needed. The compression payoff is one row of W_up (2048 params × 2 bytes = 4 KB) replaced by one scalar per aligned neuron. For non-perfectly-aligned neurons, the residual (W_up[i,:] − λ_i W_gate[i,:]) measures the deviation; if this residual is small in norm, a low-rank correction suffices.

Alignment is measured by cos(W_gate[i,:], W_up[i,:]): values near ±1 indicate alignment. The distributional question is whether this cosine is concentrated at high values across the neuron population — which is a structural property of the trained weights, directly measurable. Not CF8 MLP rank reduction: this exploits within-neuron row alignment between paired matrices, not singular value structure of either matrix individually.

### Residency arithmetic
Qwen3-1.7B: d_ffn = 6144 neurons per layer, 28 layers. W_gate and W_up each: 6144 × 2048 × 2 bytes = 24 MB per layer; total pair = 48 MB/layer × 28 = 1.34 GiB. Per aligned neuron savings: replace one 4 KB row of W_up with one scalar (8 bytes). If 20% of neurons are aligned (cos ≥ 0.95): 6144 × 0.20 = 1228 aligned neurons/layer × 28 layers × 4 KB saved = ~137 MB. At 50% alignment: ~343 MB. On Qwen3-72B: W_gate + W_up ≈ 28 GiB; 20% alignment saves ~2.8 GiB from W_up alone. ΔNLL for exact-alignment cases: 0.00 nats (algebraic identity). ΔNLL for near-alignment (cos ≥ 0.90 threshold, small residual): bounded by residual norm; estimated from calibration distribution.

### Novelty gloss vs kill list and published landscape
Closest kill: CF7/CF8 (no-retrain MLP compression via low-rank). WALIGN does not decompose either W_gate or W_up individually; it tests pairwise row alignment between the two paired matrices — a cross-matrix structural property absent from all kill-list entries. Closest published: SVD-based MLP compression (LASER, AERO). AERO merges W_gate and W_up only after activation removal (changing the function); WALIGN merges INDIVIDUAL aligned neurons while keeping the activation function exact. No paper reports per-neuron cosine alignment between W_gate and W_up rows in a trained SwiGLU network. The measurement is cheap (cosine of pairs of rows) and produces either a compression path or a structural finding about how SGD chooses to allocate the two-matrix degree of freedom.

### Smallest experiment
**Claim**: In Qwen3-1.7B, the distribution of per-neuron cos(W_gate[i,:], W_up[i,:]) across all 28 × 6144 = 172K neurons has ≥ 5% of neurons with |cos| ≥ 0.90.

Measurement: load W_gate and W_up for all 28 layers; for each layer compute row-wise cosine similarity between corresponding rows of W_gate and W_up (matrix shape 6144; einsum normalization); histogram the distribution. Runtime: ~5 min (pure tensor arithmetic, no forward pass).

**Go threshold**: ≥ 5% of neurons with |cos| ≥ 0.90 → proceed to replace those rows with scalar + single direction, measure ΔNLL.
**No-go**: distribution is uniform or bimodal near ±1 without a spike at high values → W_gate and W_up rows are trained to be geometrically independent for all neurons; the two-matrix degree of freedom is fully used. Structural finding: SwiGLU's expressivity is encoded in the angular diversity of (gate, up) row pairs, not collapsed to aligned pairs.

### Primary risk
SGD may favor near-orthogonal (gate, up) pairs as the optimal use of two projection vectors per neuron, since orthogonality maximizes information per projection. If mean |cos| ≈ 0.0, the alignment fraction is near zero and the idea collapses. Mitigation: the 5-min measurement settles it. Even if alignment is zero, the distribution shape (uniform, bimodal, concentrated near 0, etc.) is itself a structural finding about how trained SwiGLU uses its two-matrix degrees of freedom.

---

## F6-DHSUB — Residual-Stream Update Subspace Concentration (DHSUB)
**F / Track B**

### Mechanism
Track B — compression. CF2 confirmed cos(h_L, h_{L+1}) ≈ 0.99 in Qwen3-1.7B, meaning the per-layer update Δh_L = h_{L+1} − h_L is small relative to h_L. The first-principles question is not about the magnitude of Δh but its dimensionality: does Δh_L live in a consistent low-dimensional subspace across tokens and inputs? Define the calibration update matrix ΔH_L = [Δh_L^{(1)}, ..., Δh_L^{(N)}] ∈ R^{d × N} (updates across N calibration tokens). SVD of ΔH_L measures the effective rank of the per-layer residual stream update — an entirely different object from the weight matrices (CF8 killed weight rank reduction) or the residual stream itself.

If ΔH_L has r_eff ≪ d (say r_eff ≤ 256 for d=2048), then the per-layer residual-stream coordinates that are NOT in the Δh subspace are pass-through: they enter the layer unchanged and exit unchanged. These pass-through coordinates carry no information through this layer and could be stored at lower precision (e.g., INT4 vs BF16 for the pass-through block). This is a mixed-precision assignment derived from a structural measurement, not a heuristic — the pass-through coordinate indices are exactly those outside the top-r_eff singular vectors of ΔH_L.

CF2 makes this non-trivial: if Δh were large (cos < 0.99), the update subspace could fill d_model. The small Δh norm (CF2) is a necessary condition for Δh to have low effective rank, but not sufficient — the update could be small and isotropic (spread over all d_model dimensions). The empirical question is whether it is small AND directionally concentrated.

No CF9 risk: SVD of activation matrices has no theorem-precondition issues. No CF10 risk: pure measurement, no parameter fitting.

### Residency arithmetic
Payoff is not reduced weight storage but reduced ACTIVATION precision during forward pass — mixed precision of the residual stream. Pass-through residual dimensions at INT4 vs BF16: if r_eff = 256 (12.5% of d=2048), then 1875 pass-through dimensions out of 2048 can be held at INT4 rather than BF16 at 50% of arithmetic cost. At DRAM bandwidth 11.5 GB/s and residual stream traffic of d × n_layers × 2 bytes × n_tokens = 2048 × 28 × 2 × T bytes per sequence: halving the precision of 92% of residual coordinates saves ~92% × 50% = 46% of residual stream bandwidth. Per token this is: 2048 × 28 × 2 bytes = 115 KB full-precision vs 115 KB × 54% ≈ 62 KB at mixed precision. On the Ryzen 5 7530U at 11.5 GB/s: 115 KB → 62 KB per token = 5 μs saved per token. Marginal per-token but cumulative at inference scale. Primary value: feeds into RAOK (Track B R6 deferred, strongest survivor per SUMMARY.md) by identifying which residual coordinates are active.

### Novelty gloss vs kill list and published landscape
No kill-list entry targets ΔH_L (per-layer residual update matrix). CF2 only measured the magnitude (cosine angle). The dimensionality of the update has not been proposed as an F-orientation idea in any of runs 001–005. Closest published: LLM Pruning via activation analysis (Sun et al. "Wanda," 2023) uses activation magnitude per weight; this uses the residual update geometry, a different object. Closest kill: RSIDC (residual-stream intrinsic-dimension bpw schedule, Track B R5 Stage 4 deferred) targets the residual stream STATE's intrinsic dimension, not the per-layer UPDATE's dimension — distinct objects (state vs delta). Not pre-empted by that deferral.

### Smallest experiment
**Claim**: ΔH_L for at least 15 of 28 layers has var@256 ≥ 0.90 (90% of update energy in top-256 of 2048 directions) on 500-token calibration corpus.

Measurement: forward pass on 500 tokens, cache h_L and h_{L+1} per layer, compute ΔH_L = H_{L+1} − H_L (d × 500 matrix), SVD, record var@K for K ∈ {64, 128, 256, 512}. Runtime: ~35 min (forward pass 500 tokens + 28 SVDs on 2048×500 matrices).

**Go threshold**: var@256 ≥ 0.90 in ≥ 15/28 layers → pass-through coordinate identification is viable; proceed to mixed-precision assignment and ΔNLL measurement.
**No-go**: ΔH_L is approximately rank-d for all layers → per-layer updates fill all residual-stream dimensions; CF2's near-parallel residual states are consistent with isotropic small updates. Structural finding: despite the small magnitude of Δh (CF2), the update is geometrically non-concentrated — no pass-through coordinate structure exploitable for mixed precision.

### Primary risk
The 500-token sample may underestimate the update subspace (true r_eff > 500 tokens if contexts are diverse). Mitigation: run with two disjoint 250-token subsets, compare var@256 between them; if they agree within 5% the 500-token estimate is stable; if not, extend to 2000 tokens before drawing conclusions.

---

## F6-GQARED — GQA Head-Group Importance Asymmetry (GQARED)
**F / Track A**

### Mechanism
Track A — arch-transposition. Qwen3-1.7B uses GQA with 16 Q heads and 8 KV head groups (2 Q heads per KV group). The algebraic consequence: the attention output is ∑_{g=1}^{8} ∑_{q∈group_g} softmax(Q_q K_g^T / √d) V_g. The outer sum over KV groups is an 8-term sum. First-principles question: do all 8 KV groups contribute equally to the output, or is the 8-term sum dominated by a subset? If KV group g has zero or negligible attention weight across all queries (i.e., ∀ q, ∑_t softmax_t ≈ 0 for group g), that group's W_K^g and W_V^g matrices can be dropped at inference without algebraic approximation — the drop is exact for groups with exactly-zero attention weight, and approximately correct for groups with near-zero attention weight.

The group-importance measurement: on a calibration corpus, compute the mean total attention weight assigned to each KV group g (sum over positions and query heads in the group). Groups with mean weight < ε are candidates for elimination. The algebraic identity: dropping group g from the sum is equivalent to setting W_K^g = 0 and W_V^g = 0 — the computation graph changes (Track A) but the remaining groups' computation is unchanged.

This is NOT the same as the head-permutation gauge proposals (runs 003/005) which targeted head energy as a proxy for whole-head elimination within standard MHA; GQA's group structure creates a discrete algebraic factorization (the outer sum over groups) that makes group-dropping an exact operation rather than an approximation.

### Residency arithmetic
Qwen3-1.7B: 8 KV groups, W_K and W_V per group: 128 × 2048 × 2 bytes = 0.5 MB each, 1 MB per group, 8 MB per layer, 224 MB total W_K + W_V. If 2 of 8 groups have negligible attention weight (25% of groups): 56 MB saved. At 70B scale (Qwen3-72B has 8 KV heads per 64 Q heads, same GQA ratio): W_K + W_V ≈ 7 GB; dropping 2/8 groups saves ~1.75 GB. ΔNLL: exactly zero for groups with weight < ε on the calibration set; off-distribution cost requires measurement. The small savings on 1.7B makes this most compelling as a structural finding + 70B projection.

### Novelty gloss vs kill list and published landscape
No kill-list entry for GQA group-importance measurement. Closest published: GQA (Ainslie et al., 2023) introduces the architecture; "H2O" (Zhang et al., 2023) and KVQuant target KV cache eviction at position level, not at KV-group level. Per-KV-group elimination based on attention-weight mass is a different granularity. The algebraic fact that GQA's group structure is a discrete sum (unlike standard MHA where head outputs are summed in a single continuous sum) makes per-group dropping algebraically cleaner than per-head dropping in standard MHA. No paper exploits this for post-training group elimination; the closest is head-pruning in standard MHA which has been published extensively (Michel et al., 2019) but without the GQA-specific algebraic structure.

### Smallest experiment
**Claim**: In Qwen3-1.7B, at least one KV group across any layer has mean total attention weight < 0.05 (summed over all tokens and query heads in the group) on a 200-token calibration corpus.

Measurement: forward pass on 200 tokens, extract attention weights per head per layer, average over tokens and group them by KV group membership (groups 0–7 with 2 Q heads per group), compute mean weight per KV group per layer. Runtime: ~15 min (forward pass with attention weight logging).

**Go threshold**: any KV group with mean weight < 0.05 in ≥ 1 layer → identify group, zero W_K^g and W_V^g for that group in that layer, measure ΔNLL.
**No-go**: all KV groups have comparable attention weight (each near 1/8 = 0.125) across all layers → GQA groups are uniformly utilized; no group-level redundancy exists. Structural finding: Qwen3's GQA groups are trained to be equipotent (no group is significantly more used than others), which is itself a constraint on the training dynamics and rules out GQA-group compression as a post-training route.

### Primary risk
Attention weights may be highly input-dependent: a KV group with low mean weight may be critical on specific inputs (e.g., specific query types). Mitigation: measure not just mean but variance across tokens; high-variance groups are risky even if low-mean; only low-mean, low-variance groups are safe to drop.

---

## F6-WDCOL — W_down Column Effective-Rank via SwiGLU Activation Covariance (WDCOL)
**F / Track B**

### Mechanism
Track B — compression. W_down (shape d_model × d_ffn = 2048 × 6144) maps the post-SwiGLU activation vector a_L ∈ R^{6144} to the residual-stream update δh_L ∈ R^{2048}. CF8 showed W_down's weight matrix is almost certainly full-rank (explicitly listed as "predicted but untested" in SUMMARY.md; F-WNORM run 004 proposed measuring it — that result may be pending). The first-principles angle orthogonal to weight rank: what is the effective rank of W_down's output under the TRUE activation distribution? Even if W_down is full-rank as a matrix, if the activation vector a_L is concentrated in a low-dimensional subspace (because only K neurons fire, per CF1/CF3), then W_down restricted to the active directions is effectively a low-column-rank operator on the active subspace.

Define: C_L = E[a_L a_L^T] ∈ R^{d_ffn × d_ffn} (activation covariance, d_ffn = 6144). The effective operator is W_down C_L^{1/2} (shape d_model × d_ffn). SVD of this product gives the principal output directions of W_down weighted by activation variance. If the top-K right singular values of W_down C_L^{1/2} capture 90% of the Frobenius, the effective dimensionality of W_down's output is K on the calibration distribution — regardless of W_down's own rank. Compression: project W_down from the right by the top-K right singular vectors of W_down C_L^{1/2}: W_down_compressed = W_down V_K (shape d_model × K). At inference, compute a_L_projected = V_K^T a_L ∈ R^K (costs K × d_ffn FMAs), then W_down_compressed a_L_projected (costs d_model × K FMAs). Storage: V_K (d_ffn × K) + W_down_compressed (d_model × K) = K(d_ffn + d_model). At K=128: 128 × (6144 + 2048) × 2 bytes ≈ 2.1 MB/layer vs 24 MB/layer for full W_down. 28 layers: ~59 MB vs 672 MB. Savings: 613 MB if K=128 is adequate.

CRITICAL: this is NOT the same as low-rank W_down itself (CF8 class kill) because the right projection V_K comes from the activation covariance, not from the weight SVD. The projected matrix W_down V_K is NOT a rank-K approximation of W_down in Frobenius norm — it is the dominant output directions of W_down under the EMPIRICAL activation distribution. On OOD inputs, this approximation degrades — exactly the CF10 analog for subspace truncation rather than parameter fitting. The go/no-go threshold must include an OOD sanity check.

### Residency arithmetic
See Mechanism above. K=128 scenario: 672 MB → 59 MB (11.3× on W_down alone). At 70B scale (W_down ≈ 28 GiB): K=512 (still aggressive), storage = 512 × (8192 + 28672) × 2 × 32 layers ≈ 1.2 GiB vs 28 GiB baseline. If K=512 at ≤ +0.30 nats ΔNLL, this alone pushes 70B residency below 7 GiB combined with attention compression from CF11. At DRAM 11.5 GB/s: reading 1.2 GiB W_down at 70B = 104 ms vs 2435 ms for full W_down. This is the most arithmetically aggressive F-orientation idea in this run.

### Novelty gloss vs kill list and published landscape
Closest kill: CF8/CF7 class kill (low-rank W_down). WDCOL uses activation-covariance-weighted right projection — the mathematical object is W_down C_L^{1/2}, not W_down. The singular values of W_down C_L^{1/2} are NOT the singular values of W_down; they are the principal output directions weighted by how much each input direction is actually used. Closest published: GPTQ uses the input Hessian H = X^T X (second moment = activation covariance) to weight quantization error, not to find a right-projection subspace. The algebraic move here is: truncate in the activation-covariance right-singular basis, not in the weight-SVD right-singular basis. This produces a fundamentally different compression: GPTQ is uniform with Hessian-weighted error; WDCOL is a basis change followed by weight-matrix truncation. No paper reports this specific move for W_down in SwiGLU models.

### Smallest experiment
**Claim**: The activation-covariance-weighted W_down C_L^{1/2} has var@128 ≥ 0.85 in at least 20 of 28 layers (90% of W_down effective output variance in top-128 of 6144 directions) on 500-token calibration corpus.

Measurement:
1. Forward pass on 500 tokens, cache post-SwiGLU activations a_L per layer (shape d_ffn × 500).
2. Compute activation covariance (or just the singular values of a_L directly: C_L^{1/2} columns = a_L / √N).
3. For each layer: compute W_down @ a_L (shape d_model × 500); SVD of this product to get var@K.
4. Compare to W_down SVD alone (pure weight rank).

Runtime: forward pass ~20 min + 28 SVDs on d_model × 500 matrices ~5 min. Total ~25 min.

**Go threshold**: var@128 ≥ 0.85 in ≥ 20/28 layers on calibration corpus AND var@128 ≥ 0.70 on 100-token OOD held-out (guards against distribution-specific over-compression, CF10 analog).
**No-go**: activation distribution fills all 6144 input directions of W_down despite CF1/CF3 soft sparsity; the many-but-weak firing neurons collectively span the full space. Structural finding: SwiGLU soft sparsity (few neurons firing strongly) does not concentrate the activation subspace — the weak-firing tail collectively distributes activation energy across all d_ffn dimensions.

### Primary risk
CF10 analog: the activation covariance may be dramatically different on OOD inputs vs calibration. The OOD check in the go threshold addresses this. If OOD var@128 drops below 0.70 while calibration exceeds 0.85, the compression is distribution-specific and fragile. Mitigation: explicitly include OOD in the go/no-go threshold as written; if OOD fails, report calibration-only effective rank as a structural measurement anyway.

---

## F6-SINKCONST — Attention Sink Key-Vector Constancy (SINKCONST) [FREE SWING]
**F / Track A** [FREE SWING]

### Mechanism
Track A — arch-transposition. Attention sinks (BOS / delimiter tokens) receive near-constant high attention weight regardless of query. The first-principles question: is this because the SINK KEY VECTOR K_sink is near-constant across contexts (it is the W_K projection of a fixed BOS embedding, so K_sink = W_K · e_BOS is a FIXED VECTOR independent of context), and the QUERY distribution has high overlap with this fixed vector?

The algebraic identity: K_sink = W_K^L · e_BOS is fully precomputed at model-load time (no forward pass needed). For each layer L and head group g, compute k_sink^{L,g} = W_K^g · e_BOS ∈ R^{d_head}. This is a fixed vector stored alongside model weights (28 × 8 × 128 × 2 bytes = 458 KB per k_sink set; trivial).

Now, for any query q: the attention logit to the sink position is q^T k_sink^{L,g}. The first-principles structural question: is this logit ANOMALOUSLY LARGE compared to other logits because k_sink^{L,g} aligns with the dominant singular direction of W_Q^{L,g}? If k_sink^{L,g} lies in the dominant singular direction of W_Q (the direction W_Q maps most strongly into), then k_sink captures large logits by design regardless of query content. This would explain the sink phenomenon algebraically: the network has learned W_K e_BOS to align with the top W_Q output direction, guaranteeing large logits for BOS-type queries universally.

The measurable claim: cos(k_sink^{L,g}, u_1^{L,g}) where u_1^{L,g} is the top-1 left singular vector of W_Q^{L,g} (shape d_head). If this cosine is high (> 0.8) across heads and layers, the sink attention is algebraically explained by a structural relationship between W_K and W_Q. The arch-transposition: replace the BOS key with its projection onto u_1^{L,g} (a rank-1 approximation) stored as a scalar, eliminating the BOS token from the KV cache and computing its attention score via the precomputed alignment.

This is strictly more grounded than run 003 F6-ASFA (attention sink fixed-point absorption via calibration mean): SINKCONST derives the sink mechanism from the algebraic structure of W_K and W_Q, not from a calibrated mean — no calibration data needed, no CF10 risk.

### Residency arithmetic
Direct storage for k_sink precomputed vectors: 28 layers × 8 KV groups × 128 × 2 bytes = 458 KB — negligible. Payoff: at long context (32K tokens), the BOS token occupies 1/32000 of the KV cache. At Qwen3-72B with 128K context and KV cache ~50 GB: removing BOS saves 0.05% — negligible in absolute terms. The mechanism's value is not residency but structural insight: if cos(k_sink, u_1) is high across all heads/layers, this IS the explanation of why attention sinks emerge in transformer training. This is a structural finding that motivates the NEXT compression step: if all queries project similarly onto u_1, and u_1 is shared with k_sink, then the query's u_1 component is a fixed contribution to BOS attention — and that component can be separated from the query residual for further compression.

### Novelty gloss vs kill list and published landscape
No kill-list entry for W_K e_BOS alignment with W_Q singular structure. Run 003 F6-ASFA proposed a calibration-mean absorption of BOS contribution (different mechanism: calibration-data-dependent). SINKCONST is purely weight-analytic: it requires no calibration data and makes a structural prediction (k_sink aligns with top-W_Q direction) that has never been tested. StreamingLLM/H2O/SnapKV identify sinks empirically but do not derive their origin from W_K × W_Q spectral structure. If the alignment prediction holds, this is a first-principles explanation of the attention sink phenomenon — a structural finding regardless of compression payoff.

### Smallest experiment
**Claim**: For at least 20 of 28 layers and at least 4 of 8 KV groups, cos(k_sink^{L,g}, u_1^{L,g}) ≥ 0.5 (where k_sink = W_K^g · e_BOS and u_1 = top-1 right singular vector of W_Q^L restricted to head group g's query heads).

Measurement: load W_K (8 groups × 128 × 2048) and W_Q (16 heads × 128 × 2048); extract e_BOS from embedding table; compute k_sink = W_K^g @ e_BOS for each group; compute SVD of W_Q per head (or stacked per group); extract u_1; compute cosines. Runtime: ~10 min (no forward pass; pure weight arithmetic).

**Go threshold**: cos ≥ 0.5 in ≥ 20 layers, ≥ 4/8 groups → sink mechanism is W_K/W_Q-aligned; proceed to k_sink projection compression.
**No-go**: k_sink is random relative to W_Q singular structure → sinks emerge from training dynamics not captured by static weight alignment; the BOS-key mechanism is more subtle. Structural finding: attention sinks cannot be explained purely by W_K × W_Q alignment; empirical calibration approaches (ASFA) are necessary.

### Primary risk
The BOS token embedding e_BOS is high-dimensional and the embedding varies across model families — the alignment may hold for Qwen3's specific BOS token but not generalize. Mitigation: also test for the top-3 highest-frequency tokens (delimiter, EOS, common punctuation) to check whether the alignment is BOS-specific or a general property of common tokens. If only BOS shows high alignment, the mechanism is a BOS-specific artifact; if multiple high-frequency tokens share alignment with u_1, it is a structural property.

---

## Convergence handles

1. **Per-neuron cos(W_gate[i,:], W_up[i,:]) distribution** — WALIGN's binary settler; any idea in any orientation exploiting within-neuron row structure of SwiGLU pairs needs this number.
2. **var@K of per-layer residual-stream update ΔH_L** — DHSUB's measurement; relevant to any Reach or Composition idea that asks "which residual coordinates carry information between layers."
3. **Per-KV-group mean attention weight in Qwen3 GQA (8 groups)** — GQARED's measurement; convergence signal if any U or A orientation independently targets GQA structure.
4. **var@128 of W_down @ a_L (activation-covariance-weighted column subspace)** — WDCOL's key number; structurally different from W_down weight rank; determines whether SwiGLU soft-sparsity concentrates the effective W_down output subspace.
5. **cos(W_K^g · e_BOS, top-right-singular-vector of W_Q)** — SINKCONST's key measurement; convergence handle for any attention-sink mechanism in C or R orientations.
