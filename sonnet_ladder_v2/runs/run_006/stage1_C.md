# Stage 1 — Orientation C (Composition) — Run 006 (Independent)

---

## C-WVOK — W_V / W_O Rank Ordering Composes into Attention-Side Residency Budget

**C / Track B**

### Mechanism

Track B — compression. **Coupling claim**: CF11 established a rank-concentration ordering W_Q (r_99/d ≈ 0.63) < W_K (r_99/d ≈ 0.79) < MLP (≈ 1.0), measured from AQFKV. W_V and W_O were explicitly flagged as unmeasured. CF12 established that tied lm_head is full-rank-and-catastrophically-sensitive; its mechanism is dual gradient flow. W_V does NOT receive dual gradient flow (it projects values, not token identities) — so the CF12 mechanism does not apply to W_V. W_O aggregates multi-head outputs into d_model; it is structurally analogous to a wide mixing matrix, which could be either concentrated (if heads are redundant, which CF11 already implies: 16 heads → ~1 effective subspace) or flat (if W_O must route each head to different output channels). **Coupling equation**: CF11's head-redundancy finding (16 query heads collapse to ~128-dim joint subspace) implies that the 16 head VALUE streams also carry redundant information. If query heads are redundant, key-value matching routes most tokens to the same ~1/16th of the value space. Therefore:

    dim(effective_value_subspace) ≤ dim(W_Q^{shared}) × (d_head / d_model) × n_heads
                                  ≈ 128 × (128/2048) × 16 = 128

The same head-redundancy structure that forces W_Q's 128-dim collapse FORCES a corresponding W_V collapse if value routing tracks query routing. Testable: r_99(W_V) ≤ r_99(W_Q) + σ for modest σ. If this holds, the total four-matrix attention weight compression budget (K_Q=128, K_K=512, K_V=K_Q-derived, K_O=K_V-derived) is determined by the CF11 head-redundancy measurement alone — no additional measurements needed. This is a composition because CF11 alone tells you about W_Q; it takes the additional coupling (W_V tracks W_Q via value-routing) to close the W_V budget.

### Residency arithmetic

Qwen3-1.7B-Base. All four attention matrices: 2048² × 4 × 28 × 2 bytes = 938 MB. CF11 K_Q=128: saves 938/4 × (1 - 128/2048) = 212 MB. CF11 K_K=512: saves 938/4 × (1 - 512/2048) = 163 MB. If coupling holds and K_V ≈ K_Q = 128 (value subspace as compact as query): saves another 212 MB. If K_O ≈ K_K = 512 (output projection a bit wider than value): saves 163 MB. Total attention savings: 750 MB out of 938 MB (80%). Remaining attention residency: 188 MB. Quality cost: CF11 budgets +0.98 (K_Q=128) + 0.29 (K_K=512). If coupling holds at same ΔNLL/matrix rates: +0.98 + 0.29 + 0.98 + 0.29 ≈ 2.54 nats — too high for deployment. But K_Q=512 gives only +0.20; raising K_Q to 512 and targeting K_V=512 as well: total ≈ 0.20 + 0.29 + 0.20 + 0.29 = 0.98 nats — acceptable for lossy-but-usable regime. At 70B scale (W_Q alone ~10.7 GB): 80% attention weight reduction ≈ 8 GB freed, shifting the total model from ~38 GB to ~30 GB at the 4-bit quantization floor.

### Novelty gloss vs kill list and published landscape

Closest kill: v2-CHEAP-TEST-001 (cross-layer W_Q stacking) is orthogonal — this is within-layer value-matrix spectrum, not cross-layer stacking. Closest published: GQA (Ainslie et al., 2023) groups K/V heads at training time; MLA (DeepSeek) uses a joint low-rank KV projection training-time. This composition is post-training, no-retraining, and derives the W_V budget from the CF11 head-redundancy number rather than training-time choices. The coupling equation (head-redundancy forcing value subspace collapse) is not in the published landscape for Qwen3.

### Smallest experiment

**Claim**: r_99(W_V) / d ≤ 0.75 at layer L14 of Qwen3-1.7B-Base (i.e., more concentrated than W_K's 0.79, consistent with coupling to W_Q's 0.63).

- SVD of W_V (2048×2048) at L14. Compute cumulative variance at K ∈ {128, 256, 512, 1024}. Measure ΔNLL at K=512 (hold all other weights at bf16).
- Runtime: ~15 min (one SVD + 2 PPL evals on 455 tokens).
- **GO**: r_99(W_V)/d < 0.80 AND ΔNLL(K=512) < +0.40 nats.
- **NO-GO finding**: W_V is full-rank (r_99/d ≈ 1.0), structurally like MLP weights. This would mean value computation is not downstream of the query head-redundancy — the routing mechanism for W_Q and the value projection decouple. That sharpens CF11: head-redundancy is a QUERY-side phenomenon, not an attention-wide phenomenon.

### Primary risk

W_V operates per head before W_O aggregates; the full 2048×2048 W_V matrix merges all heads implicitly in storage. If per-head W_V blocks (128×2048 each) are individually concentrated but their union is full-rank (because different heads load on different output channels), the full-matrix SVD will show flat spectrum. **Mitigation**: measure SVD per head block (128×2048 shape); if any head block shows r_99 < 0.70×128, the per-head structure survives independent of the global finding.

---

## C-KSTAT-QDYN — Static K=0.1% Outlier Channel Identity Composes with W_Q Singular Vector Alignment

**C / Track B**

### Mechanism

Track B — compression. **Coupling claim**: v2-CF1 establishes two tiers — K=0.1% channels have Jaccard=0.718 (nearly channel-static) while K=1% channels have Jaccard~0.31–0.39 (token-dynamic). CF11 establishes that W_Q global K=128 singular vectors capture 99% of W_Q functional variance. The static-tier channels (top ~2 channels at d=2048) are persistently dominant in activation magnitude. Those ~2 channels either (a) align with the top W_Q singular vectors — in which case W_Q's dominant computation already handles them, and no special static-channel bookkeeping is needed at inference; or (b) are orthogonal to W_Q's principal subspace — in which case they represent activation content that W_Q does NOT attend to, meaning they can be stored at INT4 without worrying about query-matching quality. Both outcomes reduce inference complexity. **Coupling equation**:

    σ_static = max_i { |⟨e_{c_static}, v_i(W_Q)⟩| : i = 1,...,128 }

where e_{c_static} is the indicator of the 2 static outlier channels and v_i(W_Q) are the top-128 right singular vectors of the global W_Q factor. If σ_static ≥ 0.70 → case (a): static channels ARE the W_Q subspace, fold bookkeeping into W_Q projection. If σ_static ≤ 0.20 → case (b): static channels are W_Q-invisible, INT4 them unconditionally. Either branch closes the static outlier question structurally — no calibration-dependent threshold needed at inference time.

### Residency arithmetic

The residency payoff is operational rather than byte-count. At Qwen3-1.7B at 200 tok/s, 28 layers: the per-token outlier-channel dynamic detection adds ~28 compare-and-scatter operations per token. If case (a) holds (static channels inside W_Q subspace): no separate detection needed — W_Q projection already handles them. If case (b) holds (static channels outside W_Q subspace): static channels can be permanently quantized to INT4 with 0 marginal cost. Either way, the 2-channel static outlier handling reduces from a per-token runtime decision to a one-time weight-preparation step. Bandwidth saving: ~2 × 2 × 28 × 2 bytes = 224 bytes/token shifted from FP16 to INT4 (×4 compression on those bytes). Small in isolation; the composition value is eliminating a code-path branch in the per-token kernel, which enables RAOK's 3-tier scheme to simplify its tier-1 detection logic.

### Novelty gloss vs kill list and published landscape

No kill-list item covers this coupling. Closest published: SmoothQuant migrates outlier magnitude into weight matrices but does not use W_Q spectral structure to identify outlier alignment. RAOK (deferred R6-B) proposes the 3-tier quantization scheme independently; this composition supplies the structural argument for why the static tier boundary is a consequence of W_Q's spectral structure rather than an ad hoc calibration threshold. LLM.int8() treats static channels as mixed-precision without the subspace-alignment test.

### Smallest experiment

**Claim**: σ_static (defined above) is either ≥ 0.70 or ≤ 0.20 at L14 — i.e., the coupling has a clean verdict rather than landing in the ambiguous middle.

- Load W_Q K=128 global SVD factors from AQFKV. Load PDAP activations at L14. Identify top-2 channels by 200-prompt mean magnitude. Compute σ_static as the max dot-product of e_{c_static} against V's columns.
- Runtime: ~5 min (all stored data; no new forward passes required).
- **GO (case a)**: σ_static ≥ 0.70 — static outlier channels lie in W_Q principal subspace.
- **GO (case b)**: σ_static ≤ 0.20 — static outlier channels are W_Q-invisible, unconditional INT4 safe.
- **Ambiguous** (0.20 < σ_static < 0.70): partial alignment, requires hybrid handling. Still informative — gives the alignment angle as a graded input to RAOK's tier design.

### Primary risk

The W_Q SVD was computed in Frobenius norm over weight matrices; the outlier channels were identified by activation magnitude over the calibration set. These two statistics operate on structurally different objects (weights vs inputs) and their alignment has no theoretical guarantee. **Mitigation**: compute σ_static using BOTH (a) the Frobenius-SVD right singular vectors of W_Q and (b) the PCA right singular vectors of W_Q · H (W_Q applied to calibration activations H), which directly captures what W_Q does to the activation distribution. If both agree on the branch (both ≥ 0.70 or both ≤ 0.20), the result is robust.

---

## C-JOINT-DELNLL — Joint W_Q + W_K Truncation Interaction Coefficient

**C / Track A**

### Mechanism

Track A — arch-transposition. **Coupling claim**: CF11 measured W_Q truncation and W_K truncation INDEPENDENTLY. The joint experiment (both truncated simultaneously) has not been run. The coupling is algebraic: softmax attention logits are A = (W_Q h)(W_K h)^T / √d_head. After rank-K_Q truncation, W_Q → W_Q^{K_Q} = U_Q Σ_Q^{K_Q} V_Q^{K_Q T}. After rank-K_K truncation, W_K → W_K^{K_K}. The joint logit error is:

    ΔA = A - A_{joint} = (ΔW_Q h)(W_K h)^T + (W_Q^{K_Q} h)(ΔW_K h)^T
    where ΔW_Q = W_Q - W_Q^{K_Q}, ΔW_K = W_K - W_K^{K_K}

The cross-term (ΔW_Q)(ΔW_K)^T is second-order in the truncation errors. The key question: are the truncation errors ΔW_Q and ΔW_K correlated across directions? If the principal subspaces of W_Q and W_K are aligned (attending to similar input channels), their truncation errors subtract from the same activation directions — destructive interference. If orthogonal — independent errors, additive ΔNLL. CF11 found both W_Q and W_K have concentrated spectra operating on the SAME residual stream h; if they share high-variance input directions, destructive coupling is likely. **Coupling equation**:

    subspace_overlap = ‖V_Q^{K_Q T} V_K^{K_K}‖_F² / min(K_Q, K_K)

If subspace_overlap ≥ 0.5 at K_Q=K_K=512: expect δ = ΔNLL_joint − (ΔNLL_Q + ΔNLL_K) > 0 (destructive). If overlap ≤ 0.2: expect δ ≈ 0 (independent). This gives a PREDICTION for δ before running the joint experiment — a testable cross-finding coupling.

### Residency arithmetic

At K_Q=512 + K_K=512 (the "safe" operating point from CF11): saving is W_Q from 2048² to 2048×512 per layer (75% W_Q reduction) plus W_K similar. Across 28 layers of Qwen3-1.7B: 28 × 2 × (2048²−2048×512) × 2 bytes = 28 × 2 × 3.15 MB ≈ 177 MB W_Q saving + proportional W_K saving ≈ 177 MB. Total: ~354 MB on a 1.86 GB model = 19% residency reduction at ≤0.49 nats ΔNLL (if independent). If δ < 0 (constructive): ΔNLL_joint < 0.49 — the joint is a free improvement. At 70B scale: W_Q (all heads) ≈ 10.7 GB → 2.7 GB at K=512; W_K (GQA 8 groups) ≈ 1.34 GB → 0.34 GB. Total ~9.4 GB saving on Q+K — pivotal for 70B in DRAM.

### Novelty gloss vs kill list and published landscape

Closest kill: v2-CHEAP-TEST-001 kills cross-LAYER W_Q stacking — this is within-layer JOINT truncation interaction, a different question. Per-head K=64 NO-GO (CF11) eliminated per-head decomposition; this is global K=512 joint. Closest published: GQA / MLA do training-time joint reduction. A3 (arXiv:2505.12942) measures attention-logit-error from W_Q perturbations only. No published work measures the INTERACTION COEFFICIENT δ between simultaneous W_Q and W_K post-training truncation on Qwen3; the subspace-overlap prediction makes this composition mechanistically novel, not just an engineering test.

### Smallest experiment

**Claim**: subspace_overlap(V_Q^{512}, V_K^{512}) at L14 predicts the sign of δ (δ > 0 if overlap ≥ 0.5; δ ≈ 0 if overlap ≤ 0.2).

- Step 1: compute ‖V_Q^{512 T} V_K^{512}‖_F² / 512 at L14 using stored AQFKV factors. Runtime: ~2 min.
- Step 2: run joint W_Q@K=512 + W_K@K=512 truncation across all 28 layers; compute ΔNLL on 455-token eval. Runtime: ~10 min.
- Compare δ sign to overlap prediction.
- **GO**: |δ| ≤ 0.10 nats AND overlap-prediction matches sign of δ — composition is additive and the subspace-overlap is the right predictor.
- **NO-GO finding**: δ > 0.20 nats — destructive coupling is large enough to force re-budgeting (raise K_Q to 1024 when W_K is also truncated). Structural finding: W_Q and W_K share principal input directions, so their errors compound.

### Primary risk

The subspace_overlap metric (Frobenius norm of V_Q^T V_K) measures weight-space alignment, not activation-space alignment at runtime. W_Q and W_K may be weight-space aligned but activation-space orthogonal if the input distribution h is structured. **Mitigation**: compute the activation-weighted subspace overlap — replace V_Q^T V_K with (W_Q H)^T (W_K H) for calibration activations H; this directly measures what W_Q and W_K do to the same distribution, not their abstract weight alignment.

---

## C-L1-ATTEN-GATE — Layer-1 Anomaly Composes: Gate Fold + Tighter Attention + Cheaper Layer [FREE SWING]

**C / Track A**

### Mechanism

Track A — arch-transposition. **Coupling claim**: CF6 establishes layer-1 as structurally anomalous with 36% gate neurons near-constant. CF11 establishes global W_Q K=128 captures 99% variance across all 28 layers. These two anomalies were measured independently. The composition: if layer 1 is anomalous in the MLP pathway, what is its attention pathway doing? Layer 1 processes the token embedding (nearly raw, before context integration) — at this stage queries attend to positional/lexical patterns, not contextual patterns. This suggests W_Q at L1 may be MORE concentrated (queries encode simpler patterns) than the global K=128 average. The coupling equation:

    effective_rank_L1 = argmin_K { ΔNLL_L1(K) ≤ ε }   vs.   effective_rank_global = 128

If effective_rank_L1 ≤ 64 at ε = 0.20 nats, layer 1 is anomalous in BOTH pathways. The composition: drop L1 to (a) fold 36% of gate neurons (CF6), (b) use K_Q=64 for W_Q at L1 (tighter than global), (c) use K_K=256 for W_K at L1 (tighter than global K=512 at comparable ΔNLL). Total L1 simplification produces a layer that runs at ~40% of the compute of a standard layer, with no quality cost (within threshold ε for both components). Mark [FREE SWING] because the per-layer attention rank has not been measured — the coupling is compositionally motivated but empirically unverified at the per-layer level.

### Residency arithmetic

Qwen3-1.7B-Base, Layer 1 only. MLP gate fold (CF6): 36% of W_gate L1 = 36% × 2048×6144×2 bytes = 9.0 MB saved. W_Q at K=64 instead of K=128 for L1: 2048×(128−64)×2 = 0.26 MB. W_K at K=256 instead of K=512: 2048×(512−256)×2 = 1.0 MB. Total L1 additional saving: ~10.3 MB over CF11's global K=128 W_Q budget. Absolute residency: small (10 MB on a 1.86 GB model). The composition value is QUALITY-ADJUSTED: the L1 fold is zero-quality-cost (CF6 explicitly verified), and if L1 W_Q is more concentrated, tightening K_Q at L1 also has below-average quality cost. This creates a "free first layer" that reduces model residency without touching the quality budget for the remaining 27 layers.

### Novelty gloss vs kill list and published landscape

No kill-list item targets layer-1 attention specifically. LCAB (deferred Track A R2 Stage 4 idea) proposed layer-1 gate folding alone. This composition adds the per-layer attention spectrum measurement at L1, motivated by the functional argument (layer 1 handles pre-context token representations, simpler routing). Closest published: early-exit and layer-drop literature treats layers as interchangeable; the layer-1-specific anomaly composition is not in the published landscape for Qwen3. AERO (arXiv:2410.13060) does activation removal globally, not layer-selectively.

### Smallest experiment

**Claim**: ΔNLL for W_Q at L1 only truncated to K=64 (all other layers at bf16) is ≤ 0.20 nats — tighter than the global K=128 budget would predict for a random layer.

- Run: truncate only W_Q[L1] to K=64; hold all other layers at bf16. Compute ΔNLL on 455-token eval.
- Runtime: ~5 min (single-layer modification, reuse AQFKV infrastructure).
- **GO**: ΔNLL ≤ 0.20 — layer 1 W_Q is more concentrated than global average; L1 anomaly extends to attention.
- **NO-GO finding**: ΔNLL > 0.50 — layer 1 W_Q is NOT more concentrated; the MLP and attention anomalies at layer 1 are independent (different structural origin). This sharpens the CF6 anomaly: it is exclusive to the MLP gate pathway, not a layer-wide simplification.

### Primary risk

CF6's 36% gate-foldable measurement used 566 tokens on Qwen3-1.7B; if the constant-gate fraction is calibration-set-sensitive (e.g., certain token types drive those gates live), deployment diversity may shrink the foldable fraction below the threshold where folding is safe. **Mitigation**: re-measure CF6 gate variance at layer 1 across 5 structurally different passage types (code, math, natural language, instruction following, multilingual) and report min across types as the conservative floor; accept fold only for neurons with min-variance < 0.05.

---

## C-DEEP-SPREAD-QBUD — Deep-Layer Outlier Spread Implies Per-Layer W_Q Rank Schedule

**C / Track B**

### Mechanism

Track B — compression. **Coupling claim**: v2-CF1 reports that deep layers (L23–L27) of Qwen3-1.7B need 14–18% of d=2048 channels to cover 90% of outlier events, versus 5–8% for layers L2–L19. The outlier spread is a measure of how many dimensions are ACTIVATED in each layer's residual stream. CF11 measured GLOBAL W_Q K=128 (stacking all 28 layers). The coupling: if deep layers activate more channels (higher effective activation dimension), their attention computation must READ from more dimensions — i.e., W_Q at deep layers should require a HIGHER rank to capture the same fraction of the attention logit variance as shallow layers. **Coupling equation**:

    predicted_K_Q(ℓ) ∝ channel_coverage_90%(ℓ)  (from v2-CF1 per-layer data)
    ratio(L23−L27) / ratio(L2−L19) = (14–18%) / (5–8%) ≈ 2×–3×

If this coupling holds, a rank schedule K_Q(ℓ) = 64 for L0–L7, K_Q(ℓ) = 128 for L8–L19, K_Q(ℓ) = 256 for L20–L27 would match activation dimensionality to attention rank budget. Total W_Q bytes: 2048 × [8×64 + 12×128 + 8×256] × 28 is NOT right — each layer is separate. Total W_Q: (8 layers × 2048×64 + 12 layers × 2048×128 + 8 layers × 2048×256) × 2 = (1.05M + 3.15M + 4.19M) × 2 = 16.8 MB vs uniform K=128: 2048×128×28×2 = 14.7 MB. Counterintuitively, this schedule uses MORE storage (because deep layers get more K) but provides BETTER quality than uniform K=128 (because shallow layers that currently use K=128 can drop to K=64 without penalty, offsetting the deep-layer increase). The composition is about the QUALITY-STORAGE PARETO FRONTIER, not just byte reduction.

### Residency arithmetic

Compared to uniform K=128, the schedule adds 2 MB (deep layers) and subtracts 1.05 MB (shallow layers, K=64 instead of K=128), net +1 MB on W_Q storage. But ΔNLL improves: instead of imposing K=128 quality cost equally across all layers, shallow layers (where K=64 suffices) incur less loss, and deep layers (where K=128 is insufficient for the wider activation space) avoid over-compression. Total ΔNLL under the schedule may be materially lower than uniform K=128 at the same or slightly higher bytes. The engineering value: a 70B model where shallow-layer W_Q can be cut to K=64/d_head-equivalent saves ~3× in those layers vs uniform — on 70B, where W_Q total at K=128 would be ~1.3 GB (scaled), a 2× compression on the first 8 layers adds ~300 MB additional saving.

### Novelty gloss vs kill list and published landscape

Closest kill: v2-CHEAP-TEST-001 kills CROSS-LAYER stacking of W_Q — this is per-layer independent rank budgets, no cross-layer stacking. Closest published: mixed-precision quantization schedules (GPTQ per-layer sensitivity) choose per-layer bpw from gradient-based sensitivity; this uses activation-side outlier spread (v2-CF1) as the sensitivity predictor for RANK (not bits). The coupling (outlier spread predicts rank sensitivity) is a novel cross-domain pairing of two pipeline-internal measurements. No published work uses per-token activation dimensionality to schedule post-training weight-rank truncation.

### Smallest experiment

**Claim**: Spearman rank correlation between (per-layer ΔNLL at K=128, measuring 5 probe layers) and (per-layer channel_coverage_90% from v2-CF1) is ρ ≥ 0.70.

- Run: per-layer ΔNLL at K=128 for W_Q at layers L0, L7, L14, L21, L27 individually (hold all other layers at bf16). Runtime: ~35 min (5 × 2 PPL evals). Cross-reference against v2-CF1's channel_coverage_90% per-layer data. Compute Spearman ρ.
- **GO**: ρ ≥ 0.70 — activation dimensionality predicts W_Q rank sensitivity; schedule is principled.
- **NO-GO finding**: ρ < 0.30 — the two measurements are structurally independent. W_Q rank sensitivity must be measured directly per layer (weight-side only), not inferred from activation statistics. That is a crisp finding: activation spread and weight rank occupy different design dimensions.

### Primary risk

v2-CF1's channel_coverage_90% was measured at LAYER OUTPUT (post-attention residual stream), not at layer input. W_Q reads from layer INPUT (before attention). If the layer's own attention and MLP reshape the activation space significantly, the output spread may not predict what W_Q encounters. **Mitigation**: the experiment's ΔNLL measurement is the ground truth — even if the mechanistic prediction is wrong, the rank-per-layer profile is the correct input to the schedule. The Spearman test tells whether the activation-spread shortcut is valid; failure just means direct per-layer rank profiling is needed (which is what the experiment produces anyway).

---

## Convergence handles

- `W_V / W_O rank spectra (r_99/d, per-layer)` — C-WVOK depends on this directly; C-WQKV-BWFR from the prior ideator round also flagged it. If W_V/W_O are concentrated, opens ~400 MB additional compression on Qwen3-1.7B and ~8 GB on 70B.
- `subspace_overlap(V_Q^K, V_K^K)` at representative layers — C-JOINT-DELNLL needs this to predict the interaction coefficient δ before running the joint experiment.
- `per-layer W_Q rank sensitivity (ΔNLL at K=64, 128, 256 individually)` — C-L1-ATTEN-GATE and C-DEEP-SPREAD-QBUD both need this 5-probe-layer measurement; ~35 min on the Ryzen, produces a per-layer profile that multiple future compositions can use.
- `static outlier channel identity vs W_Q right singular vector alignment (σ_static)` — C-KSTAT-QDYN's 5-min measurement; resolves whether RAOK's tier-1 static detection is structurally redundant with W_Q compression already in CF11.
- `v2-CF1 channel_coverage_90% at LAYER INPUT (pre-attention)` — C-DEEP-SPREAD-QBUD and C-KSTAT-QDYN both benefit from input-side rather than output-side outlier characterization; one re-instrumentation pass covers both.
