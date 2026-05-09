# Stage 1 — Orientation C (Composition) — Run 002

Orientation: C — Composition  
Run: 002 (independent cold start; does not inherit from run_001)  
Ideas produced: 6

---

## C1 — Outlier-Highway Attention Alignment (OHAA)

**Tags:** C, Track B

### Mechanism

**Track B — compression.** CF3 establishes that at K=0.1% (≈2 channels of d=2048), outlier-channel Jaccard is 0.718 — nearly static across tokens. CF11 establishes that W_Q's global 16-head redundancy collapses to a ≈128-dim head-shared subspace (r_99/d ≈ 0.63, with K=128 global GO at ΔNLL=+0.98 nats). The coupling claim:

**Equation (C1):** Let D_ℓ^{0.1%} ⊂ ℝ^d be the 2-channel static outlier set at layer ℓ. Let S_Q^ℓ = rowspan(W_Q^ℓ, K=128) ⊂ ℝ^d be the head-shared W_Q subspace. The composition hypothesis is:

    dim(D_ℓ^{0.1%} ∩ S_Q^ℓ) ≥ 1    (for >50% of layers)

If the static outlier channels persistently fall inside the W_Q dominant subspace, then the per-token quantization mask for activation outliers and the W_Q low-rank projection can share the same K=2-channel "excision + reinsertion" basis — a single lightweight INT16 sidecar covers both the quantization exception path and the Q-projection residual, without separate bookkeeping for each.

**Why non-obvious:** CF3's measurement is over activations (residual stream magnitudes); CF11's measurement is over weight spectra. That an activation outlier's direction aligns with the W_Q head-redundancy subspace is not entailed by either finding alone — it would require that the model's training dynamics placed the dominant query-projection axes along the same channels as the persistent activation spikes.

**Primitive:** dot-product projection ‖e_i^T V_Q^{K=128}‖₂ where e_i is the i-th standard basis vector (the outlier channel) and V_Q is the top-128 right-singular vectors of W_Q stacked across heads.

**Does not rely on:** MLP rank structure (CF8 class kill), per-head W_Q rank reduction (CF11 per-head NO-GO), or any CF13–CF15 number.

### Residency arithmetic

Target: Qwen3-1.7B-Base (or 7B as proxy for 70B scaling study).  
Baseline 1.7B at Q4_K_M: ≈1.1 GiB. Primary relevance is the design principle for 70B-class deployment.

For 70B at Q4_K_M (≈4.0 bpw): ~35 GiB. At DRAM bandwidth 11.5 GB/s, decode at ≈0.33 tok/s — NVMe-assisted scenario.

If CF3's 2 static channels + CF11's 128-dim W_Q subspace share a joint excision basis: the per-token overhead of tracking outliers in W_Q quantization drops from a separate 2-channel INT16 exception list to zero additional overhead (it falls inside the already-tracked W_Q subspace). This is not a residency reduction per se — it is a **quantization-error reduction at fixed bpw**: the 2 outlier channels are handled for free by the W_Q projection basis, meaning W_Q can tolerate 0.3–0.5 bit lower quantization on those columns without quality loss.

For W_Q at 70B: ≈2.56 GiB of attention weights. If W_Q outlier columns tolerate 0.5 bpw reduction: 0.5/4 × 2.56 = 320 MB saved.  
More important path: **enables lowering W_Q precision from Q4 to Q3 on outlier columns without quality degradation**, which scales proportionally with model size.

Quality cost: if alignment holds (equation C1 holds for >50% of layers), zero additional ΔNLL — the outlier handling is already implicit in the W_Q projection. If alignment fails, this composition does not apply.

### Novelty gloss

Kill list closest: RAOK (R2-B Track B deferred) proposes 3-tier activation codebook from CF3. Structural difference: OHAA composes CF3 with CF11 specifically to ask whether the two bookkeeping structures collapse to one — the coupling claim is absent from RAOK. Published closest: SmoothQuant + A3 (arXiv:2505.12942) independently handle activation outliers and W_Q low-rank respectively, but no published method couples the static outlier channel identity with the W_Q head-shared subspace to test alignment. The alignment equation is the un-arXiv'd move.

### Smallest experiment

**Claim:** For Qwen3-1.7B-Base, at each layer ℓ, the 2 top-magnitude outlier channels (measured over 200 tokens) project non-trivially onto the top-128 right-singular vectors of stacked W_Q^ℓ: ‖e_{i(ℓ)}^T V_Q^ℓ_{K=128}‖₂ ≥ 0.3 for both outlier channels i(ℓ), in >50% of layers.

Runtime: SVD of W_Q per layer (128 right singular vectors × 28 layers) ≈ 5–10 min on Ryzen 5 7530U. Outlier channel identity from CF3 is already measured. Dot-product check: negligible.  
**GO threshold:** ‖projection‖₂ ≥ 0.3 in >50% of layers for both outlier channels.  
**NO-GO structural finding:** outlier channels and W_Q subspace are orthogonal — the two bookkeeping structures are independent, implying they cannot be unified; RAOK's three-tier quantization scheme stands without the W_Q coupling.

### Primary risk

Outlier channels may be dimension-wise extremes that are perpendicular to the query subspace (the query machinery actively suppresses the outlier direction). Mitigation: run the smallest experiment first as a 10-minute computation; the alignment check costs nothing if it fails early.

---

## C2 — Cross-Layer W_Q Subspace Telescoping (CQST)

**Tags:** C, Track B

### Mechanism

**Track B — compression.** CF11 establishes that the 16-head W_Q at each layer collapses to a ≈128-dim global subspace (head-redundancy). The unused finding is: do all 28 layers share a common W_Q dominant subspace? If yes, a cross-layer basis B ∈ ℝ^{d×K_cross} (K_cross ≤ 128) can encode every layer's W_Q as a thin coefficient matrix, yielding a larger residency reduction.

**Equation (C2):** Let V_Q^ℓ ∈ ℝ^{d×128} be the top-128 right-singular basis of W_Q^ℓ (established as the per-layer GO threshold by CF11). Define the cross-layer alignment:

    α_cross = (1/L) Σ_ℓ ‖proj_{V_Q^1}(V_Q^ℓ)‖_F² / 128

α_cross → 1 means all layers share layer-1's basis. α_cross → 0 means bases are independent. The composition claim (from CF11 applied to layer pairs) is:

    If α_cross ≥ 0.7, then W_Q^ℓ ≈ B · C_ℓ^T    for a shared B ∈ ℝ^{d×128}, layer-specific C_ℓ ∈ ℝ^{d_head_total×128}

Storage of W_Q across all 28 layers: normally 28 × d × d_head_total × bpw. With shared B: B (d×128 at bf16) + 28 × C_ℓ (d_head_total×128 at Q4) replaces 28 full W_Q matrices.

**Why this is a composition:** CF11's per-layer finding (head-redundancy within one layer) is a well-defined result. The cross-layer extension is NOT entailed by CF11 — layers could have head-redundancy at 128 dims yet the 128-dim bases could rotate arbitrarily across layers. The interaction between CF11's spectral concentration AND residual-stream near-parallelism (CF2: adjacent cos ≈ 0.99) is the composition: CF2 says inputs to each layer are nearly the same residual; if similar inputs drive similar query directions, the W_Q bases may be aligned cross-layer. Two findings composing to force a third structure.

**Does not rely on:** MLP weight structure (CF8), per-head W_Q rank reduction (CF11 per-head NO-GO), CF13–CF15.

### Residency arithmetic

Qwen3-1.7B: W_Q per layer = 2048 × (128-dim GQA, 16 heads) = 2048 × 2048 = 4M params × 28 = 112M params total for W_Q.  
At bf16: 224 MB. At Q4: 56 MB.

If α_cross ≥ 0.7 with K_cross = 128:  
- Shared B: 2048 × 128 × 2 bytes = 0.5 MB  
- Per-layer C_ℓ (d_head_total × 128 at Q4): 2048 × 128 × 0.5 bytes × 28 = 3.7 MB  
- Total W_Q footprint: ~4.2 MB vs 56 MB at Q4 — **13×** residency reduction for attention Q weights alone.

At 70B scale (W_Q ≈ 2.5 GiB at Q4): 13× → ~190 MB. Net savings: ~2.3 GiB of W_Q footprint.

Total attention weight saving at 70B (if W_K shows similar cross-layer structure, W_V/W_O untested): up to 8 GiB potential, though W_K's r_99/d=0.79 is less concentrated — conservatively 3–4 GiB saving on W_Q + W_K cross-layer sharing.

This is the most aggressive known path to DRAM-resident 70B without MLP weight rank reduction. Combined with Q4_K_M on MLP: 70B MLP weights ~32 GiB at Q4 + 4 GiB cross-layer-shared attention ≈ 36 GiB — still NVMe territory. But it points at the right lever.

Quality cost: at K_cross=128 per layer, we already know per-layer GO at ΔNLL=+0.98 nats. Cross-layer sharing with coefficient quantization adds an additional approximation step; expected additional ΔNLL ≤ 0.5 nats if α_cross ≥ 0.7 (i.e., the cross-layer approximation is tight).

### Novelty gloss

Kill list: R2/S2 cut "Sparse PCA Cross-Head Basis" (CoSpaDi + Share Your Attention cover this). Structural difference: CQST composes CF11's head-within-layer result with CF2's cross-layer near-parallel residuals to predict cross-layer W_Q basis alignment. The kill list item was proposing to find a basis; CQST is predicting the alignment will hold because of CF2's cross-layer constraint, and the equation gives a numerical test for that specific prediction. "Share Your Attention" (Aug 2025) shares KV heads; CQST shares Q projection bases across layers — a different matrix, and the mechanistic argument is different (residual near-parallelism as forcing function).

### Smallest experiment

**Claim:** α_cross ≥ 0.7 in Qwen3-1.7B-Base — i.e., the top-128 W_Q subspace is stable across the 28 layers.

Procedure: compute V_Q^ℓ for ℓ = 1..28 (each: SVD of 2048×2048 W_Q, take top-128 right singular vectors). Compute pairwise alignment ‖proj_{V_Q^1}(V_Q^ℓ)‖_F² / 128. Mean across all ℓ = α_cross.  
Runtime: 28 SVDs of 2048×2048 ≈ 8–15 min on Ryzen 5 7530U (numpy/scipy SVD, bf16 loaded).  
**GO threshold:** α_cross ≥ 0.7.  
**NO-GO structural finding:** W_Q bases rotate significantly across layers despite CF2's near-parallel residuals; coupling of CF2 + CF11 does not force cross-layer basis alignment; cross-layer sharing is not feasible; AQFKV's per-layer finding is a per-layer fact, not a global structure.

### Primary risk

GQA (grouped query attention) in Qwen3 may use fewer unique W_Q matrices than assumed; if GQA ties Q-heads per group, the effective W_Q may already be smaller than 2048×2048. Mitigation: inspect model config before running SVD; if GQA groups are 2–4, the W_Q storage is already reduced and the composition may be less impactful (but still valid).

---

## C3 — Layer-1 Gate-Fold × Outlier-Static Pinning (LGOP)

**Tags:** C, Track B

### Mechanism

**Track B — compression.** CF6 establishes that Layer 1 of Qwen3-1.7B has 36.1% of gate channels with variance below the foldable threshold (std < 0.05) — a structural anomaly absent in all other layers (<2.5%). CF3 establishes that at K=0.1% (≈2 channels), the top outlier channels are static (Jaccard=0.718) across tokens. The composition:

**Equation (C3):** Let F_1 ⊂ [d_ffn] be the foldable-gate neuron set in layer 1 (|F_1| ≈ 0.361 × d_ffn = 0.361 × 6144 ≈ 2218 neurons). Let O_1 ⊂ [d] be the 2 static outlier channels in the residual stream entering layer 1. The coupling hypothesis is:

    The rows of W_up^1[F_1, :] that project onto O_1 are the dominant contributors to |F_1|'s activation variance.

Formally: if ‖W_up^1[F_1, :]_{:, O_1}‖_F / ‖W_up^1[F_1, :]‖_F ≥ 0.5, then the foldable neurons in layer 1 are foldable precisely because they depend on the two static-outlier input channels. The static channels suppress within-F_1 variation because the input is near-constant in those dimensions. The FOLD then becomes algebraically cleaner: the mean activation μ_{F_1} = silu(W_gate^1_{F_1} · x̄) where x̄ is the mean input vector (dominated by the static outlier channels), and the correction for token-specific variation is second-order.

**Why non-obvious:** CF6 measures variance of silu(W_gate·x) as the output of the gate; CF3 measures variance of the input residual stream. Connecting them requires asking whether the input static channels cause the gate output to be static. Neither finding alone implies the other; the composition predicts a structural identity between the two measurements.

**Does not rely on:** MLP weight rank structure, CF13–CF15.

### Residency arithmetic

Layer 1 of Qwen3-1.7B: W_gate^1, W_up^1, W_down^1, each of size 2048 × 6144.  
|F_1| ≈ 2218 neurons foldable. Folding means replacing silu(W_gate^1_{F_1}·x) with a scalar mean μ_{F_1} — W_gate^1's rows indexed by F_1 become a fixed scalar per neuron, storable as 2218 × 4 bytes = 8.9 KB (bf16 scalars) instead of 2218 × 2048 × 2 bytes = 9.1 MB.

**Saving at layer 1:** ~9 MB for W_gate^1[F_1, :] folded to scalars. Tiny at 1.7B scale.

At 70B: if layer-1 anomaly scales (36% foldable gate neurons × 1 layer of ~80K × 8192 W_gate), saving ≈ 0.36 × 80K × 8192 × 2 bytes ≈ 470 MB. Still marginal.

**The composition's payoff is not residency; it is quantization tolerance.** If the foldable neurons' gate outputs are driven by static outlier channels, then: during quantization of the remaining (non-folded) W_gate^1 rows, the outlier channels are the dominant source of quantization error. Excising the outlier columns from W_gate^1[F_1^c, :] before quantization (handling them via INT16 sidecar) reduces quantization error for those rows specifically. Combined with RAOK-style tiering, this reduces Layer-1 W_gate quantization error from the dominant outlier source.

Quality gain: difficult to estimate without the coupling measurement; the smallest experiment reveals the quantitative answer.

### Novelty gloss

Kill list closest: R4-A SDZC (SwiGLU Dead-Zone Gate Collapse) — killed as a global scheme due to 1.5% global rate. Structural difference: LGOP is NOT a global scheme; it composes the Layer-1 anomaly (CF6) with the input outlier structure (CF3) to ask whether the anomaly is CAUSED by the outlier's near-constant input. This is a mechanistic composition that SDZC never tested. SDZC was killed for wrong reasons (global rate too low), but the composition with CF3 opens a new mechanistic question.

### Smallest experiment

**Claim:** The foldable neurons in layer 1 (F_1, |F_1| ≈ 2218) have W_up^1[F_1, :] significantly concentrated in the 2 static outlier channels: ‖W_up^1[F_1, :]_{:, O_1}‖_F / ‖W_up^1[F_1, :]‖_F ≥ 0.5.

Procedure: (a) identify F_1 — run the SDZC probe on layer 1 (2218-neuron set is already known from CF6); (b) identify O_1 — the 2 static outlier channels from CF3 (already known); (c) compute column-norm of W_up^1[F_1, :]_{:, O_1} vs total W_up^1[F_1, :].  
Runtime: ≤ 10 minutes (W_up^1 is 2048×6144 × 2 bytes = 24 MB; extraction trivial).  
**GO threshold:** ratio ≥ 0.5.  
**NO-GO structural finding:** foldable gate neurons are NOT caused by static outlier input; the layer-1 anomaly has a different cause (e.g., initialization artifact, specific training dynamics); the two findings are independent coincidences rather than mechanistically coupled.

### Primary risk

The static outlier channels (K=0.1%) may be too few (2 out of 2048) to explain 36% gate near-constancy — the mechanism may be statistically implausible (2 channels cannot drive 2218 neurons to near-constant output if W_up has full rank per CF8). Mitigation: test at K=1% (20 channels) in addition to K=0.1%; if the ratio requires 20 static channels to hold, the composition weakens but doesn't die.

---

## C4 — W_Q Cross-Layer Drift Informs Quantization Schedule (QCDS)

**Tags:** C, Track B

### Mechanism

**Track B — compression.** CF11 gives per-layer W_Q spectra at K=128 (GO, ΔNLL=+0.98) and K=512 (near-lossless, ΔNLL=+0.20). CF2 gives cos(h_L, h_{L+1}) ≈ 0.99 — residual stream near-parallelism across layers. The composition:

**Equation (C4):** For layer ℓ, define the effective W_Q rank as r̂_Q^ℓ — the number of singular values capturing 95% of ‖W_Q^ℓ‖_F. CF11 establishes r̂_Q varies across layers (r_99/d ≈ 0.63 on average, but per-layer variance is unknown). CF2's near-parallelism means the input distribution x_ℓ changes slowly across layers. The composition hypothesis:

    r̂_Q^ℓ is monotonically non-decreasing with layer depth ℓ.

If residual streams are near-parallel (CF2), early layers encounter nearly identical distributions, and W_Q compression at those layers costs less (smaller effective dimension of the input manifold). Deep layers where the residual stream has accumulated more specific context should show higher W_Q rank. **The coupling then forces a per-layer bpw schedule on W_Q:** low bpw at early layers (lower effective rank), higher bpw at deep layers (higher effective rank), without calibration fitting (avoiding CF10 risk).

The schedule is derived entirely from W_Q singular value spectra — a cheap offline measurement over the weight matrices, no calibration data required.

**Concretely:** if r̂_Q^{early} / r̂_Q^{deep} ≤ 0.5, then early-layer W_Q tolerates Q2 or Q3 while deep-layer W_Q requires Q4 or Q5. Net bpw on W_Q: weighted mean across 28 layers reduces from uniform Q4 toward ≈3.2 bpw, saving ≈20% W_Q bytes.

**Does not rely on:** MLP structure, CF13–CF15.

### Residency arithmetic

Qwen3-1.7B W_Q total: ~224 MB at bf16, ~56 MB at Q4.  
If per-layer schedule drops average from Q4 (4 bpw) to Q3.2 bpw: 0.8/4 × 56 MB = 11.2 MB saved on W_Q. Modest for 1.7B.

At 70B: W_Q ≈ 2.5 GiB at Q4. 20% reduction → 500 MB saved.  
More important: if the depth-stratified schedule reveals W_Q at layers 1–7 tolerates Q2 (1.5× more compression than Q4): (7/28) × 2.5 GiB × (1 - 2/4) = 312 MB additional savings.

Composable with CQST (C2): if cross-layer subspace is shared AND per-layer bpw schedules by depth, the two interact — CQST already compresses via shared basis, QCDS provides the depth-schedule for per-layer coefficient quantization. They are not independently additive but they are orthogonal mechanisms.

Quality cost: depth-stratified schedule at Q3.2 average is expected ΔNLL ≤ 0.3 nats on W_Q (well within 1-nat budget given CF11's per-layer baseline at K=128 was +0.98 at full-layer SVD approximation).

### Novelty gloss

Kill list: R2/S2 "MDL-Selected Per-Layer bpw" killed as converging to mixed-precision territory. Structural difference: QCDS derives the layer schedule directly from the W_Q singular value decay profile — no MDL fitting, no calibration data (avoiding CF10), no mixed-precision arithmetic for MLP. QCDS is W_Q-specific and derives from the composition of CF11's spectral data with CF2's near-parallelism. "Mixed-precision quantization" as a general class is published (LLM-QP, etc.); the specific argument that W_Q rank should monotonically increase with depth because of near-parallel residuals is the un-arXiv'd compositional move. The empirical test (does r̂_Q^ℓ increase with depth?) is cheap and not in the published record for Qwen3.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base, r̂_Q^ℓ (rank capturing 95% of ‖W_Q^ℓ‖_F) increases monotonically (or near-monotonically) with layer depth ℓ.

Procedure: compute r̂_Q^ℓ for ℓ = 1..28 (same SVD pass as CQST C2 — free composition of experiments). Fit a Spearman rank-correlation coefficient between ℓ and r̂_Q^ℓ.  
Runtime: 8–15 min (shared with C2's 28-SVD pass — both experiments run in the same script).  
**GO threshold:** Spearman ρ ≥ 0.7.  
**NO-GO structural finding:** W_Q rank does NOT increase with depth; depth-stratified bpw scheduling is not directly motivated by the CF2+CF11 composition; the residual near-parallelism does not propagate to W_Q spectral depth-dependence; uniform-bpw W_Q quantization is already as good as one can do without calibration.

### Primary risk

If W_Q rank does increase with depth but the magnitude of variation is small (say, r̂_Q varies from 1200 to 1600 across 28 layers), the bpw schedule gain is negligible. Mitigation: report both the monotonicity p-value AND the range ratio r̂_Q^{max} / r̂_Q^{min}; GO requires ρ ≥ 0.7 AND range ratio ≥ 1.5.

---

## C5 — Residual Stream Outlier Pinning for Attention-Head Excision (ROPE) [FREE SWING]

**Tags:** C, Track A

### Mechanism

**Track A — arch-transposition.** CF11 established that 16 query heads in Qwen3-1.7B collapse to a ≈128-dim subspace — head redundancy at roughly 16:1 ratio. CF3 established that the residual stream has 2 static outlier channels (K=0.1%, Jaccard=0.718). The composition asks a different question than C1: instead of quantization bookkeeping, the question is **whether the static outlier channels serve as the routing key for head differentiation**.

**Equation (C5):** Let h_ℓ^{(k)}(x) = softmax((W_Q^{(k)} x)(W_K^{(k)} C)^T / √d) W_V^{(k)} C be the output of head k in layer ℓ for context C. Define the head-output alignment:

    β_k = ‖e_{O_1}^T h_ℓ^{(k)}(x) ‖₂ / ‖h_ℓ^{(k)}(x)‖₂

where O_1 is the 2-channel static outlier set. Composition hypothesis:

    If β_{k1} ≠ β_{k2} for most head pairs (k1, k2), heads differ primarily in how they route through the outlier channels.
    If β_{k1} ≈ β_{k2} for most head pairs, the outlier projection is symmetric across heads and heads DO NOT differentiate via outlier channels.

The second case (symmetric β) is the interesting compression case: if all heads produce similar projections onto the outlier channels, then the outlier channel component of the attention output is independent of which head computed it — it can be factored out as a shared "outlier attention" component, and the remaining per-head residual is lower-magnitude and more compressible.

**Track A move:** Replace the 16-head computation with a 1-head "outlier attention" (full-precision on the 2 static channels) + a 4–8 head "residual attention" (lower-precision on the remaining 2046 channels). This restructures the computation graph, not just the storage.

**Does not rely on:** MLP structure, CF13–CF15. CF3's static outlier channels are the only measurement needed for the O_1 definition; CF11's head-redundancy motivates the 4–8 head residual.

### Residency arithmetic

Qwen3-1.7B attention weights W_Q + W_K + W_V + W_O: ~896 MB at bf16, ~224 MB at Q4.

Under ROPE: 1 outlier-attention head (2-channel subspace) + 4–8 residual heads (128-dim subspace each).  
Outlier head: effectively 2×d Q/K and 2×d V/O matrices ≈ 4 × 2048 × 2 × 2 bytes = 32 KB — negligible.  
4 residual heads at K_per_head=128: 4 × 128 × 2048 × 4 matrices × 2 bytes = 8.4 MB.

Total ROPE attention storage: ~8.5 MB vs 224 MB at Q4 for full 16-head Q4 — **26× compression on attention weights** if β alignment holds. ΔNLL budget: CF11's head-redundancy at K=128 global was +0.98 nats; ROPE's architecture is more aggressive (structured factoring), so expected ΔNLL ≈ 1.5–2.5 nats — borderline but potentially acceptable for throughput-quality tradeoff.

At 70B scale: attention weight savings ~2× the 1.7B ratio (different d, n_heads); directionally 10–15 GiB of attention weights → 1–2 GiB.

### Novelty gloss

Kill list: no direct ROPE analog. SGHH (Track A R3 deferred) targets "sink-gated head hibernation" via calibrated bias — operates via attention-logit sinks, not outlier-channel alignment. Published: MLA (DeepSeek-V2) does joint Q-K low-rank projection during training. A3 (arXiv:2505.12942) does post-training attention approximation via attention-logit error minimization. None compose the static-outlier-channel structure (CF3) with head-redundancy (CF11) to propose an architecture split into outlier-attention + residual-attention components. The "outlier-channel as routing separator" is the novel structural claim.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base, the per-head β_k values (projection onto outlier channels) are approximately equal across heads: std(β_k) / mean(β_k) ≤ 0.3 for >50% of layers.

Procedure: (a) run a 50-token forward pass on Qwen3-1.7B, capturing per-head attention outputs at each layer; (b) compute β_k for each head and each layer; (c) measure coefficient of variation of β_k across heads per layer.  
Runtime: forward pass capture ≈ 5 min; β_k computation ≈ 2 min.  
**GO threshold:** CV(β_k) ≤ 0.3 in >50% of layers.  
**NO-GO structural finding:** heads strongly differentiate via outlier-channel projections; the outlier channel is a head-specific routing mechanism (not a shared background); ROPE's factoring is invalid; outlier-channel structure is head-private not head-shared; RAOK's activation-quantization path handles outliers without head-architecture changes.

### Primary risk

CF3's static channels are measured at K=0.1% of d_residual; in the attention output space (dimension d_out_head = d/n_heads = 128), there may be no equivalent concept of "outlier channel" — the measurement domain changes. Mitigation: measure β_k in the residual stream output of the attention block (after W_O projection back to d=2048), not in the per-head intermediate space.

---

## C6 — Residual-Cosine Flatness × W_Q Depth-Gradient as Per-Layer KV Precision (RKGP)

**Tags:** C, Track B

### Mechanism

**Track B — compression.** CF2 establishes cos(h_L, h_{L+1}) ≈ 0.99 across all 28 layers of Qwen3-0.6B (and by replication, 1.7B family). CF11 establishes W_K at K=512 is GO (ΔNLL=+0.29 nats), and W_K at K=256 is ΔNLL=+0.82 nats, with r_99/d ≈ 0.79 for W_K. The composition:

**Equation (C6):** The KV cache at token position t stores K_ℓ(t) = W_K^ℓ · h_ℓ(t) and V_ℓ(t) = W_V^ℓ · h_ℓ(t). Since cos(h_ℓ, h_{ℓ+1}) ≈ 0.99 (CF2), consecutive residual stream states differ by a small delta:

    h_{ℓ+1}(t) = h_ℓ(t) + δ_ℓ(t),    ‖δ_ℓ(t)‖ / ‖h_ℓ(t)‖ ≈ √(2(1 - 0.99)) ≈ 0.14

So K_{ℓ+1}(t) = W_K^{ℓ+1} · h_{ℓ+1}(t) = W_K^{ℓ+1} · h_ℓ(t) + W_K^{ℓ+1} · δ_ℓ(t).

The coupling claim: **the cross-layer delta W_K^{ℓ+1} · δ_ℓ(t) is small relative to the base W_K^{ℓ+1} · h_ℓ(t), and the magnitude of the delta is predictable from ‖W_K^{ℓ+1}‖ × 0.14.** This means the KV cache for adjacent layers is correlated — a stored K_ℓ(t) can be used to predict K_{ℓ+1}(t) with correction cost ≈ 0.14 × ‖W_K‖. If the correction falls below the KV quantization noise floor, the correction can be omitted — K_{ℓ+1}(t) ≈ W_K^{ℓ+1} · h_ℓ(t) (recomputed from the previous layer's residual rather than stored).

**Practical mechanism:** for long-context inference (>8K tokens) where KV cache is the bottleneck, store K/V at every other layer and recompute the missing layer's KV on-the-fly from the stored adjacent layer's residual. The computation cost (one W_K GEMV per missing layer per token) is smaller than loading from NVMe; the storage cost halves.

**Does not rely on:** MLP structure, CF13–CF15. CF2 and CF11 are both v2-confirmed.

### Residency arithmetic

KV cache at 1.7B, context=4K: ≈ (28 layers × 2 × 4K × d_kv × bpw) / 8. d_kv for GQA Qwen3-1.7B: 8 KV heads × 128-dim = 1024. At INT8: 28 × 2 × 4096 × 1024 = 234 MB.

For long context (32K tokens): ~1.8 GiB. Halving the KV cache by storing every other layer: ~0.9 GiB saved.

At 70B, context=32K: KV cache ≈ 64 KV heads × 128 dim × 32K tokens × 80 layers × 2 (K+V) × 1 byte ≈ 41 GiB at INT8. That exceeds DRAM. At FP8: ~20 GiB. Storing every other layer: 10–20 GiB saved — the KV cache becomes the dominant leverage point for long-context 70B inference.

ΔNLL cost: recomputing K_{ℓ+1} from h_ℓ rather than h_{ℓ+1} introduces an approximation error proportional to ‖W_K^{ℓ+1} · δ_ℓ‖. CF2 gives the delta norm ≈ 0.14 × ‖h_ℓ‖. The quality impact depends on ‖W_K^{ℓ+1}‖ — measured as part of the smallest experiment.

### Novelty gloss

Kill list: CLASE (Track A R3 deferred — Cross-Layer KV Aliasing INT4 deltas, deferred for bottleneck mismatch at <32K context). Structural difference: RKGP does not use delta-coding for storage (that is CLASE); it proposes on-the-fly recomputation of every-other-layer KV from the prior layer's residual, using CF2's near-parallelism as the justification that the approximation error is small. The key distinction: CLASE stores compressed deltas; RKGP stores nothing for the skipped layers and recomputes at inference time. The recompute cost (1 GEMV vs 1 NVMe read) is the go/no-go trade-off. Published closest: "Layer-Condensed KV Cache" (arXiv:2405.10637) computes KV at one layer and reuses across layers — a related idea but premised on training-time coupling; RKGP exploits the no-training CF2 measurement and tests whether the approximation quality holds post-hoc without retraining.

### Smallest experiment

**Claim:** The approximation error ‖K_{ℓ+1}(t) - W_K^{ℓ+1} · h_ℓ(t)‖₂ / ‖K_{ℓ+1}(t)‖₂ ≤ 0.15 for >60% of layer pairs in Qwen3-1.7B-Base.

Procedure: forward pass 100 tokens, capture h_ℓ and h_{ℓ+1} at all layers. Compute K_{ℓ+1}^{approx} = W_K^{ℓ+1} · h_ℓ and compare to K_{ℓ+1}^{true} = W_K^{ℓ+1} · h_{ℓ+1}. Measure relative error per layer pair.  
Runtime: ≤ 20 min.  
**GO threshold:** mean relative error ≤ 0.15 for >60% of layer-pairs.  
**NO-GO structural finding:** the delta δ_ℓ, though small in cosine angle, drives large magnitude variations in W_K^{ℓ+1} · δ_ℓ because W_K amplifies the delta direction; CF2's near-parallel angle is a poor proxy for KV recomputation error; the layer-skip approach requires per-token correction that costs more than NVMe reads.

### Primary risk

CF2's cos ≈ 0.99 measurement was on Qwen3-0.6B, confirmed at 1.7B (KILL_LIST mentions residual additivity reconfirmation). However, for 70B the residual stream and W_K dimensions scale differently. The 0.14 delta norm estimate may not hold at larger scale. Mitigation: the smallest experiment runs on 1.7B and provides the empirical rather than theoretical delta norm; scale-up is a follow-on.

---

## Convergence handles

These are the primitives multiple ideas above depend on — cross-orientation convergence signal if any other ideator surfaces the same primitive independently:

1. **Top-128 right-singular basis of W_Q stacked across layers** — shared measurement primitive for C1 (alignment with outlier channels), C2 (cross-layer α_cross), and C4 (depth-monotone rank schedule). A single 28-SVD computation script resolves C1, C2, and C4 simultaneously.

2. **Static outlier channel identity (K=0.1%, Jaccard=0.718, 2 channels)** — CF3 measurement used as input to C1, C3, and C5. Any idea from other orientations that also uses the K=0.1% outlier-channel identity as a routing key or alignment anchor is a convergence signal.

3. **Cross-layer residual delta norm from CF2** — ‖δ_ℓ‖ / ‖h_ℓ‖ ≈ 0.14, derived from cos ≈ 0.99. Used in C4 (input distribution near-parallelism as forcing function for monotone rank) and C6 (KV layer-skip approximation bound). Both ideas would fail if CF2's near-parallelism does not translate to small magnitude deltas.

4. **β_k symmetry across heads** — C5's core measurement (per-head outlier-channel projection). If the Unconventional Substrate or Reach orientations surface attention-head differentiation through a different lens, comparing their mechanism against β_k symmetry is the convergence check.

5. **W_K relative error under layer-skip recomputation** — C6's go/no-go number. If any other orientation proposes KV-cache reduction via cross-layer structure, the relative error bound ≤ 0.15 is the shared falsifiability anchor.
