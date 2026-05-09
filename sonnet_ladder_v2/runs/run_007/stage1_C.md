# Stage 1 — Orientation C (Composition) — Run 007

Orientation: C — Composition
Run: 007 (independent cold start)
Kill compliance: v2-CHEAP-TEST-001 kills all cross-layer W_Q stacked-SVD / shared-basis variants. Per-layer K=128 is the W_Q ceiling; no cross-layer stacking. Killed in runs 001-005 (and therefore avoided here): JQSOH, OHAA, QAOC, AQOD, ZOIA, KDJB, RBHD, RKDB, TESOC, WUPDOWN, ODSP/DSAF/ODWQ, LAYERFOLD/SDLC/GCQI, VKSD/WVKQ, GEMBED, SPECDRIFT.
Run-level kill: cross-layer W_Q, arbitrary O(d) rotation, within-layer Q/K joint softmax × RoPE.
Ideas: 6

---

## C7-JDWQ — v2-CF1 Layer-Jaccard × W_Q Per-Layer r_99 Correlation Oracle (JDWQ)

**C / Track B**

### Mechanism

Track B — compression. v2-CF1 (confirmed 2026-05-09) measured K=1% outlier-channel Jaccard across all 28 layers of Qwen3-1.7B-Base: mean 0.39, range [0.22, 0.531]. The within-layers pattern is structured: mid-depth layers (L7-L15) cluster around 0.41-0.44, only L27 is marginal at 0.531. CF11 found W_Q effective rank r_99/d ≈ 0.63 at L14 (one layer). The composition claim:

**Equation (C7):** Let J_ℓ = K=1% per-layer Jaccard at layer ℓ (v2-CF1 full sweep). Let ρ_ℓ = r_99(W_Q^ℓ) / d from a per-layer W_Q SVD sweep (CF11 opened this as an untested extension). The coupling:

    Spearman_ρ(J_ℓ, ρ_ℓ) > 0  across all 28 layers

Rationale: layers where the top-1% outlier channel set is more token-dynamic (low Jaccard) are processing more context-dependent information — those layers require W_Q to span more directions to route attention to the varying outlier-driven tokens. Layers where Jaccard is higher (more static) can afford a lower-rank W_Q because the dominant query directions are stable and predictable. The coupling converts v2-CF1's per-layer Jaccard profile into a principled, closed-form W_Q per-layer rank budget: K(ℓ) = K_base × J_ℓ / mean(J), normalized so the total rank budget equals 28 × K_base (same total storage as uniform K=128).

This differs structurally from run_004's ODWQ (which correlated outlier *spread* from CF3 with r_99 from CF11 but both from the original 7-sampled-layer PDAP data). JDWQ uses the full 28-layer v2-CF1 Jaccard sweep, which is a different quantity (consecutive-pair Jaccard of the outlier *set* vs the channel *fraction covering 90% of events*), and the per-layer W_Q SVD across all 28 layers has not been run. The empirical anchor is v2-CF1, a v2-confirmed finding not used in any prior run's C proposals.

Does not rely on: cross-layer W_Q (killed), MLP rank (CF8), CF13-15.

### Residency arithmetic

Uniform K=128 for 28 layers: 28 × 2 × 2048 × 128 × 2B = 29.4 MB W_Q factors.

JDWQ Jaccard-proportional budget: J ranges [0.22, 0.531], mean ≈ 0.39. Normalized per-layer K(ℓ) = 128 × J_ℓ / 0.39. Range: K_min = 128 × 0.22/0.39 ≈ 72, K_max = 128 × 0.531/0.39 ≈ 174. Total: same 29.4 MB (by construction — the budget is redistributed, not reduced). Quality benefit: layers where uniform K=128 was insufficient (high-Jaccard layers = context-dynamic, need more rank) get K>128; layers over-provisioned get K<128. Expected ΔNLL at fixed budget: improvement from +0.98 nats (uniform K=128, CF11) toward +0.6-0.8 nats by allocating rank where it matters.

At 70B (80 layers, d_model=8192): W_Q at K_base=512 → 80 × 2 × 8192 × 512 × 2B = 1.34 GB. Jaccard-proportional redistribution within same budget: if per-layer W_Q ΔNLL variance is large, budget-neutral redistribution saves quality cost, enabling use of K_base=256 total instead of K_base=512 for the same ΔNLL — halving W_Q to 0.67 GB.

Quality cost: depends on correlation sign and strength. Experiment-determined.

### Novelty gloss

Closest prior-run Composition idea: ODWQ (run_004) — correlated PDAP spread with W_Q r_99 from AQFKV. Structural differences: (1) v2-CF1 Jaccard is a different measure than CF3 spread (consecutive-pair outlier-SET Jaccard vs channel-fraction), (2) v2-CF1 covers all 28 layers with permutation control confirming the signal is learned (not random), (3) the K budget allocation is derived from Jaccard, not spread. Closest published method: LLM-MQ / OmniQuant use per-layer sensitivity for bpw assignment; none use the activation outlier-set stability (Jaccard) as the sensitivity proxy. The v2-CF1 anchor is the distinguishing feature — that data was produced by this ladder and is not in the literature.

### Smallest experiment

**Claim:** Spearman ρ(J_ℓ, r_99(W_Q^ℓ)/d) has the same sign (positive) across all 28 layers of Qwen3-1.7B-Base.

Procedure: (a) v2-CF1 J_ℓ is already measured (from rraok_result.md); (b) run per-layer W_Q SVD for all 28 layers — the top-256 singular values per layer — ~30 min on Ryzen 5 7530U; (c) compute r_99 per layer; (d) compute Spearman ρ.
Runtime: 30–35 min.
GO threshold: ρ > 0.20 AND p < 0.10 (28 data points; moderate threshold given noise).
NO-GO structural finding: W_Q effective rank is uniform across layers and uncorrelated with outlier Jaccard — per-layer rank budget allocation buys nothing; uniform K=128 is near-optimal; the per-layer Jaccard variation from v2-CF1 is an activation property orthogonal to the weight structure.

### Primary risk

The per-layer SVD may show that r_99(W_Q) is approximately uniform across all 28 layers (CF11 measured L14 = 0.63; other layers could be similar). Mitigation: the 30-min experiment settles this before any implementation; even NO-GO confirms that CF11's L14 measurement is representative.

---

## C8-WDOS — W_down Output-Space × Tied-Embed Span Disjointness (WDOS)

**C / Track B**

### Mechanism

Track B — compression. CF12 (tied embed/lm_head E ∈ ℝ^{151936×2048}, r_99=1992/2048, catastrophic ΔNLL=+19.96 at r=1024) establishes that the embed/lm_head is full-rank and cannot be truncated. The mechanism is: gradient flows through TWO paths (input lookup + output projection), keeping all 2048 directions load-bearing. CF5 (W_up is MORE rank-sensitive than W_gate at matched K fractions) and CF8 (W_down untested for rank structure per SUMMARY "what's still open" §3) together motivate a spectral measurement of W_down. The composition claim:

**Equation (C8):** Let R_down = colspan(W_down) ⊂ ℝ^{2048} (the output subspace of W_down, shape d_FFN×d_model). Let R_E = colspan(E^T) = ℝ^{2048} (full span, since CF12 r_99=1992≈d). The composition prediction:

    dim(R_down ∩ colspan(E^T at r=128)) / 128  <  0.5

i.e., the top-128 principal directions of the tied embed matrix are DISJOINT from the principal output directions of W_down. Rationale: W_down maps from the intermediate SwiGLU space to the residual, and the residual is what gets looked up by E^T at the lm_head step. If E^T's top directions are orthogonal to W_down's output principal directions, then compressing E^T (if it were possible) would not harm the W_down signal. More directly: the disjointness tells us whether the tied embed's high-rank structure is "wasted" on directions W_down never writes to — a gauge freedom in the span allocation between the residual-writing weights (W_down, W_O) and the residual-reading output (E^T).

If Eq. (C8) holds: the tied embed's top-128 principal directions (the "most-read" directions by E^T for logit computation) are not the directions W_down writes to. This means quantizing W_down to lower precision has zero first-order impact on the logit distribution — the quantization error lands in a subspace E^T does not read. This would allow low-precision W_down without logit degradation, orthogonal to the CF8 full-rank finding (which is about rank, not about output-subspace alignment with the reader).

Does not require: MLP rank reduction (CF8 — this is a different axis: residual-subspace alignment, not rank), cross-layer W_Q (killed), CF13-15.

### Residency arithmetic

W_down per layer: 6144×2048×2B = 25.2 MB × 28 layers = 706 MB for Qwen3-1.7B-Base.

If the disjointness holds: W_down can be quantized to INT4 or INT2 without logit degradation (because its output subspace is E^T-invisible). At INT4: 706/4 = 176 MB saving = 530 MB. At INT2: 706/8 = 88 MB, saving 618 MB. This is the single-largest MLP-side compression lever if the coupling holds — W_down has resisted all rank-based approaches (CF8), but quantization precision is a different question.

Quality cost: E(ε²_quant × ‖W_down_output‖²) lands in the invisible subspace, so ΔNLL ≈ 0 for the disjoint component. The non-disjoint component carries the quality cost. If 80% of W_down's output energy is in E^T-invisible directions, effective INT2 W_down ΔNLL ≈ 0.2 × (INT2 noise on aligned component) — potentially << 0.5 nats.

At 70B: W_down = 28672×8192×2B×80 layers ≈ 37.7 GB. INT4: 9.4 GB. The saving (28 GB) is the largest single lever toward the 7.28 GiB target.

### Novelty gloss

Closest kill-list item: C5-WUPDOWN (run_003) — predicted W_down is rank-concentrated via CF2 × CF5 argument. WDOS is structurally different: it measures W_down's output subspace alignment with the tied embed's principal directions, not W_down's own rank. The coupling is CF12 × CF8-extension: the tied embed's full-rank structure defines the "reader subspace," and W_down's disjointness from it determines quantization safety. No prior run has asked whether W_down's output signal is visible to E^T. Closest published method: SmoothQuant handles activation outliers between W_down and the next layer's input; no method asks whether W_down's output is geometrically invisible to the tied lm_head projection.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base, the principal angle between colspan(W_down[L] at r=128) and colspan(E^T at r=128) is > 60° for ≥ 20 of 28 layers.

Procedure: (a) W_down SVD per layer: top-128 right singular vectors of each W_down[L] (shape 6144×2048); (b) E^T top-128 right singular vectors (already computable from CF12 LHQD data: E = embed/lm_head matrix at 151936×2048, top-128 cols of E); (c) compute principal angles between the two 128-dim subspaces per layer via SVD of the cross-Gram matrix.
Runtime: ~30 min (W_down SVD for 28 layers, reuse CF12 E data).
GO threshold: mean principal angle > 60° AND fraction of layers > 60° exceeds 0.70.
NO-GO structural finding: W_down output subspace and tied embed principal directions are ALIGNED — W_down preferentially writes to the directions E^T reads most; quantization error in W_down DOES land where the lm_head can see it; the disjointness argument fails.

### Primary risk

CF12 established E is full-rank (r_99=1992/2048), so E^T's top-128 directions are not especially dominant in logit computation — the logit variance is spread nearly uniformly across all 2048 directions. If E^T is nearly isotropic, the "E^T-invisible subspace" barely exists. Mitigation: before running the full experiment, compute the fraction of logit variance concentrated in E^T's top-128 directions on the PDAP calibration corpus — if this fraction < 0.20, the disjointness claim's payoff is small.

---

## C9-JAQV — Jaccard-Layer Gradient × W_Q / W_V Rank Budget Joint Allocation (JAQV)

**C / Track B**

### Mechanism

Track B — compression. v2-CF1 measured the per-layer K=1% Jaccard depth profile across all 28 layers. CF11 established W_Q r_99/d ≈ 0.63, W_K r_99/d ≈ 0.79. W_V was flagged as untested. The composition claim is a joint budget allocation for W_Q AND W_V simultaneously, driven by a single per-layer signal:

**Equation (C9):** Define the attention-weight compression budget at layer ℓ as B(ℓ) = K_Q(ℓ) + K_V(ℓ), total rank budget split across W_Q and W_V. The claim: given a fixed total budget B_total = 28 × (K_Q + K_V), the OPTIMAL split allocates:

    K_Q(ℓ) = K_Q_base × J_ℓ^{α},   K_V(ℓ) = K_V_base × (1 − J_ℓ)^{β}

with α, β > 0. The intuition: when J_ℓ (token-dynamic outlier Jaccard) is LOW (tokens are diverse; high context-dependency), W_Q needs more rank (more diverse queries) — hence K_Q scales UP with J. When J_ℓ is HIGH (tokens are stable; low context-dependency), W_V needs less rank (values are predictable from context) — hence K_V scales DOWN with (1−J). The two adjustments are anticorrelated and partially offset, keeping total budget fixed.

This differs from JDWQ (C7) in two ways: (1) it allocates the budget JOINTLY across W_Q and W_V per layer, not just W_Q, using the complementary nature of query (routing) and value (content) roles; (2) it introduces the hypothesis that W_V and W_Q have OPPOSITE responses to the Jaccard signal, making the budget trade productive rather than uniform.

Does not require: W_V spectrum confirmed (the allocation can be run as a search over α,β via calibration with n_params = 2 scalars and n_samples >> 2); cross-layer W_Q (killed); CF13-15.

### Residency arithmetic

W_Q + W_V per layer: (2048×2048 + 2048×128) × 2B × 28 — note Qwen3-1.7B GQA: n_q_heads=16, n_kv_heads=8, d_head=128. W_V shape: 8×128×2048 = 2097152 params per layer × 2B = 4 MB per layer. W_Q: 2048×2048×2B = 8 MB per layer. Total W_Q+W_V: 12 MB × 28 = 336 MB.

At K_Q=128 (CF11 GO), K_V=64 (untested, plausible if W_V r_99/d < 0.50): compressed W_Q+W_V per layer = 2×2048×128×2 + 2×2048×64×2 B = 1.05+0.52 = 1.57 MB. Total: 44 MB vs 336 MB — 7.6× compression.

Jaccard-allocation gain over uniform K_Q=128, K_V=64: if high-J layers (L7-L15, mid-depth, Jaccard 0.41-0.44) tolerate K_Q=96 and K_V=80, and low-J layers (L27, Jaccard=0.531) use K_Q=160 and K_V=32, same total budget but better quality.

### Novelty gloss

Closest prior-run idea: C7 (JDWQ) — this extends JDWQ to the W_V dimension with a complementary anti-correlation hypothesis. Also related to run_004's VKSD (W_V spectrum measurement), which measured the W_K/W_V subspace gap but didn't involve the Jaccard signal. The anti-correlation hypothesis (W_Q and W_V budget should move in OPPOSITE directions with J) is the novel coupling. No prior Composition run proposed this anticorrelation. Closest published: PALU / TransMLA (mentioned in run_005 kills as pre-empting WVWK-MLA) — those are training-time. Post-training per-layer W_Q + W_V joint Jaccard-driven allocation is not in the literature.

### Smallest experiment

**Claim:** K=64 on W_V for all layers of Qwen3-1.7B-Base gives ΔNLL < +0.50 nats (i.e., W_V is at least as compressible as W_K at K=512 per CF11's W_K measurement).

Procedure: (a) SVD of all W_V per layer; eval ΔNLL at K=64, K=128; (b) check whether per-layer W_V r_99/d < 0.50 holds; (c) spot-check anticorrelation: layers L7 (J≈0.42) vs L27 (J≈0.531) — does W_Q ΔNLL at K=96 track the Jaccard order?
Runtime: 40 min (W_V SVD 28 layers + 3 eval passes).
GO threshold: K=64 W_V ΔNLL < +0.50 nats AND r_99(W_V)/d_kv < 0.70.
NO-GO structural finding: W_V is full-rank (like W_gate/W_up — CF8 territory may extend to V); the anticorrelation hypothesis has no traction; W_V compression must use quantization not rank reduction.

### Primary risk

GQA in Qwen3-1.7B: W_V has shape 8×128×2048 = 1 MB per layer (8 KV heads × d_head × d_model), not 2048×2048. The rank ceiling is min(1024, 2048) = 1024, not 2048. This makes W_V inherently more rank-concentrated per dimension than W_Q. Mitigation: verify GQA config before running; if GQA is active, adjust K_V accordingly (K_V=64 out of 1024 possible rank is 6.25% — aggressively low but potentially fine if W_V has concentrated spectrum like W_Q).

---

## C10-FRCF — Firing-Rank × Embed Span: W_up Row-Norm Distribution Predicts Tied-Embed Column Dominance (FRCF)

**C / Track B**

### Mechanism

Track B — compression. CF1 (SwiGLU top-K firing dominated by W_up·x, not W_gate·x) and CF12 (tied embed/lm_head E full-rank, catastrophic ΔNLL at truncation, per-row norms full — "per-row norms: median 1.62, bottom-decile 1.36, near-zero count = 0"). The composition claim: W_up determines which neurons fire post-SwiGLU (CF1). The post-SwiGLU activation vector is then multiplied by W_down to produce the MLP residual update. The residual update reaches E^T at the lm_head. Therefore the column importance in E (the columns that matter most for logit production) should correlate with the rows of W_up that fire most (the neurons with highest W_up·x magnitude across calibration tokens).

**Equation (C10):** Let F_j = mean firing magnitude of neuron j across calibration tokens: F_j = mean_t(|silu(W_gate[j]·x_t) × W_up[j]·x_t|). Let c_E(j) = the E column importance of neuron j's downstream contribution: c_E(j) = ‖E · W_down[:,j]‖_2 (the logit sensitivity to neuron j's output). The coupling:

    Spearman_ρ(F_j, c_E(j)) > 0.30  across j = 1..d_FFN at representative layers

Rationale: neurons that fire strongly (high F_j per CF1's W_up dominance) should produce large W_down[:,j]·x_FFN contributions, which then hit E^T. If those contributions land in the high-logit-sensitivity columns of E, the firing rank is doubly important — not just for intermediate activations but for the output distribution. If ρ > 0.30: we can identify a subset of d_FFN neurons (the high-F_j tier) whose W_up rows and W_down columns are simultaneously prioritized. These form a "fast path" from input residual to output logit via the fired neurons — a structural identity that reduces the effective MLP cost without touching the weight rank (CF8).

Practical consequence: the high-F_j neurons can be stored at BF16 while low-F_j neurons are stored at INT4 — not as rank reduction but as a per-neuron precision assignment grounded in the joint firing-rank × logit-sensitivity coupling. Unlike PDAP / RAOK (which handle activation channels), this assigns precision to MLP neurons (rows of W_up, columns of W_down) rather than to residual-stream channels.

Does not require: MLP rank reduction (CF8 — this is precision assignment within the full-rank matrix), cross-layer W_Q (killed), CF13-15.

### Residency arithmetic

d_FFN = 6144, 28 layers. If top-5% of neurons by F_j × c_E(j) stay at BF16 and the remaining 95% at INT4: per layer, 0.05 × 6144 × (2+2) × 2048 × 2B (W_up row + W_down col at BF16) + 0.95 × 6144 × (0.5+0.5) × 2048 = 0.05 × 50.3 MB + 0.95 × 12.6 MB = 2.5 + 12.0 = 14.5 MB/layer vs 25.2 MB BF16. Saving per layer: 10.7 MB. Total: 10.7 × 28 = 300 MB saving on W_up+W_down. Model size reduction: 300 MB out of ~3 GB model ≈ 10% at 1.7B. At 70B: W_up+W_down ≈ 2 × 28672 × 8192 × 80 × 2B = 75.5 GB. 5% BF16 sidecar + 95% INT4 = 0.05 × 75.5 + 0.95 × (75.5/4) = 3.8 + 17.9 = 21.7 GB vs 75.5 GB — saving 53 GB. Combined with W_Q compression (CF11): could bring 70B under 7.28 GiB if both apply.

### Novelty gloss

Closest prior-run idea: none in runs 001-005 directly compose CF1 with CF12 on the neuron-dimension (d_FFN). Prior runs either treated the MLP (CF1, CF4, CF5, CF8) and the embed (CF12) separately. Closest kill-list item: SPADC (Track B R5 — W_down activation-skew codebook, heavily weakened by GPTVQ overlap). FRCF differs: it identifies NEURONS (not activations) as the compression unit, using the dual signal from W_up firing rank AND embed logit sensitivity rather than any calibration-fitted codebook. No calibration fitting involved (CF10 safe: F_j is a statistic over forward passes, not a fitted parameter). Closest published: SpQR (per-weight outlier isolation based on Hessian sensitivity); FRCF uses a cross-matrix coupling (W_up firing rank × lm_head projection sensitivity) not available to Hessian-only methods.

### Smallest experiment

**Claim:** Spearman ρ(F_j, c_E(j)) > 0.15 at ≥ 20 of 28 layers of Qwen3-1.7B-Base.

Procedure: (a) compute F_j from PDAP calibration corpus (200 prompts): mean |silu(W_gate[j]·x) × W_up[j]·x| per neuron per layer; (b) compute c_E(j) = ‖E · W_down[:,j]‖_2 for each neuron j (dot each W_down column into E — 6144 column-matrix products, 30 min); (c) Spearman ρ per layer.
Runtime: ~45 min.
GO threshold: ρ > 0.15 (small but positive, 6144 data points → statistically meaningful at p < 0.001).
NO-GO structural finding: W_up firing dominance (CF1) does NOT propagate to lm_head logit sensitivity; the MLP-to-output path decorrelates the two signals; CF1 and CF12 are structurally independent; per-neuron precision assignment must use Hessian-based methods rather than firing-rank × embed-sensitivity.

### Primary risk

The c_E(j) computation — ‖E · W_down[:,j]‖_2 for 6144 neurons — requires loading E (590 MB bf16) and W_down (25 MB per layer) into memory simultaneously. At 7.28 GiB system RAM, 28 layers × (590+25) MB would overflow. Mitigation: compute layer-by-layer, releasing E from cache and reloading (or memory-mapping E with mmap so the OS handles eviction); 1 layer at a time fits in <1 GB.

---

## C11-JVSTAB — Jaccard Stability Tier × W_V Subspace Reuse Rate (JVSTAB)

**C / Track A**

### Mechanism

Track A — arch-transposition. v2-CF1 established that mid-depth layers (L7-L15) have Jaccard clustering around 0.41-0.44 (the most stable tier), while L27 is marginal at 0.531. CF11 establishes W_V is untested but W_K at K=512 has ΔNLL=+0.29 nats, and W_Q at K=128 has ΔNLL=+0.98. The composition:

At layers where J_ℓ is high (token-stable), the residual stream changes little per token (CF2: cos≈0.99) AND the dominant input channels are stable (v2-CF1). Therefore the VALUE vectors V_ℓ = W_V^ℓ · h_ℓ change slowly across consecutive tokens. When V_ℓ changes slowly, the KV cache for those layers can be recomputed instead of stored — the recompute cost (one W_V GEMV) is bounded by the stability of W_V's input, and the storage cost (d_kv × seq_len bytes per layer) can be avoided.

**Equation (C11):** Define the per-layer value-recompute error as:

    ε_V(ℓ, t) = ‖W_V^ℓ · h_ℓ(t) − W_V^ℓ · h_ℓ(t-1)‖_2 / ‖W_V^ℓ · h_ℓ(t)‖_2

The stability claim: ε_V(ℓ, t) < 0.10 for layers ℓ where J_ℓ > 0.45 (high Jaccard). If this holds: for those stable layers, the previous token's V is a good approximation of the current token's V — we can "reuse" the stored V from the previous step without re-reading W_V from DRAM. This is a compute-reuse arch-transposition: the V GEMV is replaced by a staleness check (compare input residual norm change to ε_V threshold) + conditional reuse. On tokens where the check passes, W_V is not read.

The arch change: instead of always computing V_ℓ = W_V^ℓ · h_ℓ, compute only if ‖Δh_ℓ‖ > threshold (derived from J_ℓ); otherwise reuse V_ℓ(t-1). The threshold is the v2-CF1 Jaccard signal converted to a per-layer input-stability bound.

Does not require: cross-layer W_Q (killed), softmax × RoPE changes (killed), MLP rank (CF8), CF13-15.

### Residency arithmetic

W_V at Qwen3-1.7B (GQA, 8 KV heads): W_V shape 8×128×2048 = 2M params × 2B = 4 MB per layer. At 28 layers: 112 MB. The arch-transposition does not change residency of W_V itself; it reduces the bandwidth consumed reading W_V per token at stable layers.

Bandwidth saving: at J_ℓ > 0.45, if 60% of tokens trigger V-reuse (i.e., ‖Δh_ℓ‖ < threshold), bandwidth per token for W_V at those layers drops by 60%. For layers L7-L15 (8 layers, mid-depth): 8 × 4 MB × 0.60 = 19 MB saved per token. At 11.5 GB/s DRAM: 19 MB / 11500 MB/s = 1.65 ms per token saved from W_V reads alone.

At 70B (GQA-8, W_V = 8192×1024×2B×80 = 1.34 GB): if 20 stable layers save 60% bandwidth: 20 × (1.34/80) × 0.60 = 0.20 GB per token bandwidth saved. At 11.5 GB/s: 17 ms per token = 0.30× speedup on V-bandwidth. Total model bandwidth ≈ 18 GB/token; W_V read ≈ 0.5 GB × 0.60 = 0.30 GB saved; total speedup ≈ 1.7%.

Quality cost: ε_V < 0.10 at stable layers means the V approximation introduces < 10% relative error in V at each step; cumulative attention output error across stable layers is bounded by the product of per-step errors.

### Novelty gloss

Closest prior-run Composition idea: RKDB (run_005) — KV recompute from adjacent layer's residual using CF2 + CF11 W_K operator norm. JVSTAB differs: (1) reuses the SAME layer's V from the previous TOKEN (not the adjacent layer's residual); (2) uses v2-CF1's per-layer Jaccard as the stability signal for the skip condition (RKDB used CF2's per-layer cosine + W_K operator norm); (3) targets W_V not W_K. The temporal-reuse (same layer, consecutive tokens) vs cross-layer recompute (adjacent layer, same token) is a structural distinction. Closest published: Scissorhands / SnapKV / StreamingLLM manage KV cache eviction; none gate V-reuse on per-layer Jaccard from the activation outlier stream.

### Smallest experiment

**Claim:** At layers where v2-CF1 J_ℓ ≥ 0.45 (target: L7-L15), the per-token relative value change ε_V(ℓ, t) < 0.10 for ≥ 60% of consecutive token pairs in the PDAP calibration corpus.

Procedure: forward pass 200 PDAP prompts through Qwen3-1.7B-Base; at each stable-J layer, compute V_ℓ(t) = W_V^ℓ · h_ℓ(t) for each token t; compute relative change ε_V(ℓ, t) across consecutive token pairs; histogram.
Runtime: ~40 min (forward pass + V vector computation per token).
GO threshold: ε_V < 0.10 for ≥ 60% of pairs in stable-J layers.
NO-GO structural finding: token-to-token value-vector change is NOT predicted by Jaccard stability; the residual-stream Jaccard (outlier-channel identity stability) does not translate to value-vector stability; v2-CF1 and V-vector dynamics are decoupled.

### Primary risk

V-reuse introduces error that compounds over multi-token generation spans. Even if per-step ε_V < 0.10 holds, 100 consecutive reused steps produce ε ≈ 1 − (0.90)^100 = near-total corruption. Mitigation: cap reuse at N consecutive tokens (N=5); run the smallest experiment with staleness analysis showing how ε_V compounds with reuse depth; set N empirically.

---

## C12-RUNE — Residual Near-Unity × Embed Logit Sensitivity as Low-Rank Projection Bottleneck [FREE SWING]

**C / Track A**

### Mechanism

Track A — arch-transposition. [FREE SWING]. CF2 (residual additivity: cos(h_L, h_{L+1}) ≈ 0.99 — all hidden states live near the initial embedding direction) and CF12 (tied embed E is full-rank, catastrophic at truncation, but this confirms gradient flows through E^T as output projection). The composition is not about compression of E; it is about using the near-unity cosine to derive a cheap approximate lm_head computation.

**Equation (C12):** cos(h_L, h_0) ≈ 0.99 implies ‖h_L − h_0‖ ≈ 0.141 × ‖h_0‖. If h_0 = E[token_id] (the input embedding), then at inference time h_L ≈ h_0 + Δh where ‖Δh‖/‖h_0‖ ≈ 0.141. The lm_head logit is E^T · h_L = E^T · h_0 + E^T · Δh. The first term E^T · h_0 = E^T · E[token_id] — the dot product of the embedding row with all other embedding rows, which can be precomputed at model-load time as a lookup table of shape 151936×151936×2B (catastrophically large — 46 GB). So the naive precompute is infeasible.

However: E^T · Δh is a correction term where ‖Δh‖ is small. The ARCH-TRANSPOSITION: replace the full lm_head GEMV (E^T · h_L, cost = 151936 × 2048 MACs = 311M MACs) with the decomposition:

    logit ≈ E^T · h_0 + E^T · P_k · Δh

where P_k = V_k V_k^T is the top-k projection onto the k principal directions of Δh variation (i.e., the directions in which h changes most across the forward pass). The claim: Δh lives in a lower-dimensional subspace than h itself — if PCA of the residual increments {Δh_t = h_L(t) - h_0(t)} across calibration tokens gives 99% variance at k << 2048, then E^T · P_k · Δh is a cheap rank-k correction to the precomputed base.

**Coupling:** CF2 (Δh is small) + CF12 (E^T is full-rank, so E^T · h_0 spans 151936 logit dims but must be computed exactly) + the new hypothesis that **Δh has lower intrinsic dimension than h_0**. This is NOT CF13-CF15 (unconfirmed residual intrinsic dimension) — it is a new v2 measurement of the INCREMENTAL residual Δh = h_L - h_0, not the total h_L.

CF9 check: no imported theorem; the PCA decomposition is a standard SVD; preconditions are just "Δh vectors have low-rank covariance," which is testable.

The arch-transposition: compute E^T · h_0 from a precomputed 151936-dimensional lookup (one row of E^T E per token — the inner product of E[token_id] with all vocab embeddings, 311M MACs precomputed once at model-load time then indexed as a 151936×151936 half-precision table? No — still 46 GB). Revised: do NOT precompute; instead, factor the lm_head as E^T · (h_0 + Δh) = E^T · h_0 + E^T_k · Δh_k, where E^T_k = E^T · V_k (shape 151936×k, k << 2048), computed once at model-load from the PCA of Δh. At k=64: E^T_k = 151936×64×2B = 18.9 MB (fits in DRAM), Δh_k = V_k^T · (h_L - h_0) (k=64 projections, trivial). Full lm_head is now: E^T · h_0 (full 311M MACs) + E^T_k · Δh_k (151936×64 = 9.7M MACs). The SECOND term is 32× cheaper. The FIRST term (E^T · h_0 = the contribution from the base embedding) is amortized across decode steps because h_0 changes only when the prompt changes — for autoregressive decode, h_0 is fixed and E^T · h_0 can be precomputed once per PROMPT (151936×2048 MACs per prompt position, cached). This converts the per-token lm_head from 311M MACs to: (precomputed E^T · h_0 amortized over T tokens) + 9.7M MACs Δh correction per token.

Quality cost: the approximation E^T · h_L ≈ E^T · h_0 + E^T_k · Δh_k loses the E^T · (I - P_k) · Δh component, which is the incremental residual's high-k directions. If PCA of Δh has 99% variance at k=64, this component is tiny.

Does not rely on: CF13/14/15 (the "stratified intrinsic dimension" is CF15, unverified; this measures Δh directly). Cross-layer W_Q (killed). CF8 MLP rank.

### Residency arithmetic

The arch-transposition adds a precomputed correction matrix E^T_k ∈ ℝ^{151936×64}: 18.9 MB overhead.

Per-token cost: E^T · h_L (311M MACs) → E^T · h_0 (precomputed, ~0 per token after prompt) + E^T_k · Δh_k (9.7M MACs). At 3 tok/s on Ryzen (NVMe-bound), lm_head is ~2% of decode time per token (E^T is 590 MB, read once per token at 11.5 GB/s ≈ 51 ms → at 3 tok/s token period ≈ 333 ms, lm_head is 15%). The arch-transposition replaces 590 MB read (lm_head weight) per token with 18.9 MB read (E^T_k) + O(1) precomputed lookup per token — a 31× bandwidth reduction on lm_head. At 11.5 GB/s: from 51 ms to 1.6 ms per token on lm_head alone → saves 49 ms per token → ~5% speedup at 3 tok/s.

At 70B / NVMe scenario: lm_head read (590 MB even at 70B since vocab is fixed) amortized over NVMe total = 35 GB per token; lm_head = 1.7% of bandwidth. The 31× saving on lm_head saves 0.017 × 0.97 × 35 GB ≈ 0.56 GB/token → ~1.6% speedup.

### Novelty gloss

Closest prior-run Composition idea: GEMBED (run_003) — fold W_Q[0] into embedding via alignment test. RUNE is structurally different: it decomposes the OUTPUT lm_head into a base term + low-rank correction, using CF2's small-Δh argument. The [FREE SWING] tag reflects that the Δh intrinsic dimension is unconfirmed. Closest published method: "Speculative Decoding" reuses the draft model's logit; "Flash-Decoding" optimizes attention; "MagicPIG" uses locality-sensitive hashing on lm_head. RUNE uses a per-token δh-projection (based on residual increment structure) to separate the lm_head into a prompt-amortized base and a cheap per-token correction — no existing method uses CF2's near-unity residual cosine to structure this decomposition.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base, the PCA of Δh_t = h_L(t) - h_0(t) across 200 calibration tokens achieves 99% variance at k ≤ 128 (i.e., the residual increments live in a significantly lower-dimensional subspace than the 2048-dim ambient space).

Procedure: (a) forward pass 200 PDAP prompts; capture h_0(t) = E[token_id_t] and h_L(t) = final residual; (b) compute Δh_t = h_L(t) - h_0(t) for each token; (c) PCA of the 200×2048 matrix of Δh vectors; record r_99. (d) verify ‖E^T_k · Δh_k − E^T · Δh‖_1 / ‖E^T · Δh‖_1 < 0.05 at k=r_99.
Runtime: ~20 min.
GO threshold: r_99(Δh) ≤ 128 AND logit-error < 5%.
NO-GO structural finding: Δh_t has full intrinsic dimension (~2048) — CF2's near-unity cosine is a consequence of scale, not of dimensionality; the incremental residual is not low-rank; the lm_head factorization does not compress; CF2 and CF12 are independent along the Δh-dimension axis.

### Primary risk

CF2 establishes cos(h_L, h_{L+1}) ≈ 0.99, but this is a statement about the cosine between ADJACENT layers, not between h_L and h_0. The cosine between the FINAL hidden state h_L and the initial embedding h_0 = E[token_id] may be much smaller (e.g., ≈ 0.60) across 28 layers of transformation — meaning Δh = h_L - h_0 is NOT small but is on the order of h_0. Mitigation: check cos(h_L, h_0) empirically across the PDAP corpus before assuming Δh is small; if cos(h_L, h_0) < 0.80, the Δh-small argument collapses and the idea must be reformulated as a standard PCA decomposition without the CF2 grounding.

---

## Convergence handles

1. **v2-CF1 per-layer K=1% Jaccard profile** (J_ℓ, all 28 layers, confirmed 2026-05-09) — load-bearing in C7 (JDWQ rank correlation), C9 (JAQV joint W_Q+W_V allocation), C11 (JVSTAB value-reuse gate). Single sweep, reuse across all three.

2. **Per-layer W_Q r_99 full sweep (all 28 layers)** — load-bearing in C7 (JDWQ correlation test). CF11 measured only L14; 30-min extension resolves C7 and provides the per-layer distribution for C9's anticorrelation test.

3. **W_V SVD spectrum (all 28 layers)** — load-bearing in C9 (JAQV W_V rank budget), C11 (JVSTAB V-vector stability). Single W_V SVD sweep resolves C9's GO/NO-GO and provides W_V r_99 for C11's reuse threshold.

4. **Δh = h_L − h_0 intrinsic dimension** (new measurement, not CF13/14/15) — load-bearing in C12 (RUNE lm_head factorization). 20-min PCA of residual increments. Converges if Reach or First-Principles orientations independently surface the same Δh low-rank claim.

5. **W_down output subspace vs tied-embed column alignment** — load-bearing in C8 (WDOS). Requires W_down SVD (all 28 layers) + CF12 E^T column projection. Converges if another orientation surfaces "W_down output subspace is E^T-invisible" independently.

6. **Neuron-level firing rank × logit sensitivity (F_j × c_E(j))** — load-bearing in C10 (FRCF). A single forward-pass + column-norm computation resolves C10's go/no-go. Converges if RAOK / PDAP-style orientations surface per-neuron precision assignment from firing statistics.
