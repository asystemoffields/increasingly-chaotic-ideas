# Stage 1 — Orientation R (Reach) — Run 006

Orientation: R — Reach
Kill enforced: v2-CHEAP-TEST-001 (cross-layer W_Q stacked SVD / class dead); v2-S3-R004-001 (arbitrary orthogonal rotation / RMSNorm-gauge-breaking class dead).
Pre-emption enforced: within-layer Q/K joint subspace (PALU+TransMLA); softmax shift-invariance × RoPE; cross-layer W_Q variants; arbitrary orthogonal rotations breaking RMSNorm gauge.
CF anchor status: CF1–CF12 confirmed load-bearing. CF13–CF15 unverified; re-derivation required if numbers appear in residency math.

---

## R6-WVOP — W_V/W_O Product Rank Cascade

### Name
W_V/W_O Product Rank Cascade (WVOP) — R, Track A

### Mechanism
**Track A — arch-transposition.** In every transformer layer the attended-value output flows through two GEMVs: `h = (A · XW_V) W_O`. The composed operator W_VO = W_V W_O ∈ ℝ^{d×d} is computable once at load time (one matmul, ~17 ms on Ryzen) and collapses both GEMVs into one at inference — an AERO-style fold applied to the attention V-O chain rather than to SwiGLU. CF11 (r_99/d ≈ 0.63 for W_Q, 0.79 for W_K) delineates the attention matrices as the compressible class vs MLP (CF8 full-rank class). W_V and W_O are explicitly flagged in SUMMARY.md as "untested — cheap parallel measurement." rank(W_VO) ≤ min(rank(W_V), rank(W_O)), so if either matrix has r_99/d < 0.80, the product has a compressed SVD factorization W_VO ≈ U_r Σ_r V_r^T with residency 2 × d × r × 2B vs the original 2 × d² × 2B. Not per-head (CF11 per-head K=64 NO-GO); this is global over all heads jointly, the confirmed GO path. Not cross-layer (v2-CHEAP-TEST-001 killed). Distinct from A3 (which targets the QK pair); W_VO is a different algebraic object — the fused output projection chain.

### Residency arithmetic
- Qwen3-1.7B (d=2048, 28 layers): W_V + W_O each d×d = 2048² × 2B = 8 MB/layer/matrix. Both = 16 MB/layer × 28 = 448 MB bf16.
  At r=512: W_VO → (2048×512 + 512×2048) × 2B = 4 MB/layer × 28 = 112 MB. Save 336 MB.
- Qwen3-72B analogue (d=8192, n_kv=8, d_head=128, 80 layers):
  W_O = 8192² × 2B × 80 = 10.74 GB. W_V (GQA) = 1024×8192 × 2B × 80 = 1.34 GB. Combined = 12.08 GB bf16.
  W_VO product shape: W_V (1024×8192) @ W_O (8192×8192) = (1024×8192). r_99 target ≤ 512:
  Compressed: (1024×512 + 512×8192) × 2B × 80 = 0.84 GB. Saves 11.24 GB.
- Combined residency at 70B: NanoQuant MLP ~5.35 GB + W_VO compressed 0.84 GB + W_Q K=128 (CF11) ~0.34 GB + W_K K=256 ~0.17 GB + embed/residual ~0.30 GB ≈ 7.00 GiB → within 7.28 GiB budget.
- DRAM bandwidth: 7.00 GB / 11.5 GB/s = 0.609 s/token → 1.64 tok/s. Single-GEMV fold per layer (W_VO instead of W_V then W_O) also reduces matmul calls 2×1 per layer → latency gain ~10% → ~1.80 tok/s effective.
- Quality estimate: if W_V r_99/d ≈ 0.63 (same as W_Q), K=512 on W_VO ≈ K/d=0.25. CF11 W_Q K=256 gives ΔNLL=+0.51. W_VO is a product; rank may be more concentrated than W_V alone (product compresses redundant directions). Conservative: ΔNLL ≈ +0.50 nats at K=512. Within budget if W_Q K=128 (+0.98) is not simultaneously applied; need joint measurement.
- P(end-to-end): P(r_99(W_VO)/d < 0.70) × P(ΔNLL(K=512) < 0.80) ≈ 0.55 × 0.70 ≈ 0.39.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: CF11 per-head W_Q rank truncation NO-GO at K=64. Structural difference: WVOP targets the global product W_VO, a fused two-matrix algebraic object with its own spectrum, not per-head per-matrix truncation. Closest published: AERO (arXiv:2410.13060) folds SwiGLU W_gate into W_up by removing the activation; this is the attention-side analog — fold W_O into W_V by exploiting W_VO product rank. A3 (arXiv:2505.12942) targets W_Q/W_K; this proposal is structurally orthogonal (different matrix pair). Post-training W_VO product compression not found in published literature.

### Smallest experiment
Testable claim: r_99(W_V[14] @ W_O[14]) / d < 0.70 AND ΔNLL at K_VO=512 < 0.80 nats on Qwen3-1.7B-Base bf16.
Method: compute W_VO = W_V[14] @ W_O[14] ∈ ℝ^{1024×2048} (≈0.3 s), truncated SVD at K=256/512/1024 (≈45 s via scipy.linalg.svds), apply to forward pass of 512-token WikiText-2 slice, measure ΔNLL.
Runtime: ≈ 25 min.
GO threshold: r_99/d < 0.70 AND ΔNLL(K=512) < 0.80 nats.
NO-GO finding: W_O is full-rank (like MLP W_up) because the output projection must map heads back into the full residual stream; extends CF8 boundary to V-O chain; confirms attention compression ceiling is W_Q + W_K only.

### Primary risk
W_O may be full-rank — it aggregates 16 heads' outputs into a d-dimensional residual and must maintain the full spectrum of the combined head manifold. Mitigation: the 25-min spectrum measurement resolves this before any implementation; NO-GO is a structural finding worth having.

---

## R6-ATTNQ-FULL — Joint 4-Matrix Attention Compression 70B Cascade

### Name
Joint 4-Matrix Attention Compression 70B Cascade (ATTNQ-FULL) — R, Track B

### Mechanism
**Track B — compression.** CF11 (W_Q r_99/d ≈ 0.63, W_K r_99/d ≈ 0.79) establishes that attention weights are structurally unlike MLP weights (CF8). W_V and W_O spectra are unmeasured. This cascade builds the 70B-residency argument contingent on W_V and W_O each having r_99/d < 0.85 — a gate that resolves in ≤20 min. If all four matrices are compressible at K ≤ d/2, each is stored as its rank-K SVD factorization (U_K Σ_K V_K^T). Inference: each GEMV is replaced by a two-step rank-K matmul (x → V_K^T space → apply Σ, → U_K space). DRAM bytes loaded per token fall proportionally. This is pure Track B — the computation graph structure is unchanged (still four attention projections); only bytes-per-weight change. Ground: CF11 (confirmed GO at K=128 global W_Q, K=512 W_K GO-adjacent). No cross-layer stacking (v2-CHEAP-TEST-001 killed cross-layer W_Q; this is per-layer per-matrix). No per-head decomposition (CF11 per-head NO-GO).

### Residency arithmetic
- 70B: d=8192, n_heads=64, n_kv=8, d_head=128, n_layers=80.
  W_Q = 8192² × 2B × 80 = 10.74 GB; W_K = 1024×8192 × 2B × 80 = 1.34 GB; W_V = 1.34 GB (GQA); W_O = 10.74 GB. Total attention = 24.16 GB bf16.
- Compressed at K_Q=128, K_K=K_V=256, K_O=256 (assuming r_99/d ≈ 0.79 for W_V and W_O):
  W_Q: 2×8192×128 × 2B × 80 = 335 MB.
  W_K: 2×1024×256 × 2B × 80 = 83 MB.
  W_V: 2×1024×256 × 2B × 80 = 83 MB.
  W_O: 2×8192×256 × 2B × 80 = 671 MB.
  Total compressed attention: 1.17 GB — a 20.7× compression vs bf16.
- Combined with NanoQuant MLP at 5.35 GiB: 5.35 + 1.17 + 0.30 = 6.82 GiB → within 7.28 GiB.
- DRAM bandwidth: 6.82 GB / 11.5 GB/s = 0.593 s/token → 1.69 tok/s. Above the deployment floor.
- Aggressive path (K_Q=128, K_K=K_V=K_O=512, RAOK MLP ~0.51 bpw): MLP ≈ 4.40 GB + attention 1.45 GB + other 0.30 = 6.15 GiB → 11.5/6.15 ≈ 1.87 tok/s.
- Quality: ΔNLL from W_Q K=128 (+0.98, CF11 confirmed) + W_K K=256 (~+0.15 extrapolated from CF11) + W_V K=256 (unknown; target <0.30) + W_O K=256 (unknown; target <0.50). Optimistic total: ~1.93 nats. Above the 0.3-nat target but below catastrophic. Joint compression may have destructive interaction; must measure.
- P(end-to-end): P(W_V r_99/d<0.85) × P(W_O r_99/d<0.85) × P(joint ΔNLL<2.5 nats) ≈ 0.60 × 0.50 × 0.65 ≈ 0.195. Low but the prize is DRAM-resident 70B.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: CF11 per-head W_Q NO-GO. Structural difference: global per-matrix SVD, not per-head; confirmed GO path per CF11. Closest published: GPTQ / QuIP / QuaRot apply quantization to all weight matrices including attention; this uses SVD truncation (spectrum-aware rank reduction) rather than quantization grid, exploiting the attention-class structural difference (CF11 vs CF8). A3 covers W_Q/W_K; this extends to W_V/W_O. No published paper does post-training joint SVD across all four attention matrices on 70B for deployment.

### Smallest experiment
Testable claim: W_V[14] has r_99/d < 0.85 AND ΔNLL at K=256 < 0.80 nats on Qwen3-1.7B.
Method: SVD of W_V[14] (shape 1024×2048 in GQA Qwen3-1.7B; ~20 s), reconstruct at K=128/256/512, measure ΔNLL. Simultaneously measure W_O[14] (shape 2048×2048; ~45 s).
Runtime: ≈ 20 min total for both matrices.
GO threshold: r_99/d < 0.85 for W_V OR W_O (partial GO opens partial cascade).
NO-GO finding: W_V and W_O full-rank → attention compression ceiling is W_Q+W_K only; 70B residency floor ~2.0 GB for attention, which still enables DRAM-residency at 7.28 GiB when combined with NanoQuant MLP.

### Primary risk
Quality destructive interaction: four independently truncated attention matrices may produce super-additive ΔNLL from misaligned subspace removals. Mitigation: measure joint-truncation ΔNLL as a single forward pass immediately after individual measurements; if super-additive, reduce K for the weakest matrix first.

---

## R6-DSAF-70B — Depth-Spread W_Q Rank Schedule 70B Cascade

### Name
Depth-Spread Attention-Rank Schedule 70B (DSAF-70B) — R, Track B

### Mechanism
**Track B — compression.** v2-CF1 measured K=1% Jaccard across all 28 layers: range [0.22, 0.531], with mid-depth layers 7–15 least dynamic (Jaccard ≈ 0.41–0.44) and deep layers more dynamic (L27 at 0.531). CF3 further shows per-layer outlier spread_ℓ (fraction of channels covering 90% of outlier events) increases with depth: layers 2–10 have spread_ℓ ≈ 5–8%, layers 23–27 ≈ 14–18%. The coupling claim: layers with broader activation spread need higher-rank attention to redistribute information across more independent directions — a depth-gradient activation-space signal predicting the minimum safe W_Q rank per layer. Per-layer K_safe(ℓ) = f(spread_ℓ(ℓ)). For a fixed total K budget across layers, a depth-stratified schedule achieves lower ΔNLL than uniform K at the same total bytes. The Reach case: at 70B with budget fixed at K_avg=160 (between CF11's K=128 (+0.98 nats) and K=256 (+0.51 nats)), a variable schedule with K=96 for low-spread layers (0–13) and K=192 for high-spread layers (14–27) targets ΔNLL ~0.60 nats — better quality than uniform K=128 at slightly higher bytes, but the per-layer byte schedule can be tuned to equal bytes with quality gain. Not cross-layer W_Q (v2-CHEAP-TEST-001 killed stacked cross-layer sharing; this is per-layer independent SVD with a depth-informed rank budget).

### Residency arithmetic
- 70B (d=8192, 80 layers). W_Q at uniform K=128: 2×8192×128 × 2B × 80 = 335 MB. At K_avg=144 (variable):
  Layers 0–39 (low-spread, K=112): 2×8192×112 × 2B × 40 = 147 MB.
  Layers 40–79 (high-spread, K=176): 2×8192×176 × 2B × 40 = 231 MB.
  Total variable: 378 MB vs 335 MB uniform K=128. Cost: 43 MB more. Benefit: ΔNLL improvement ~0.15–0.30 nats at the same quality as uniform K=160 (which would cost 419 MB).
- Equivalently: quality-neutral variable schedule at same 335 MB uses K=96 early / K=160 deep, achieving better ΔNLL than uniform K=128 for zero residency cost.
- At 70B combined: attention total ≈ 1.17 GB (ATTNQ-FULL cascade) → within 7.28 GiB when stacked with NanoQuant MLP.
- Quality improvement: estimated ΔNLL reduction of 0.10–0.25 nats vs uniform K, contingent on spread_ℓ–rank correlation holding across 28 layers (Spearman ρ ≥ 0.60).
- P(end-to-end): P(ρ(spread_ℓ, depth) ≥ 0.60 across all 28 layers) × P(variable schedule improves ΔNLL at matched bytes) ≈ 0.65 × 0.75 ≈ 0.49.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: MDL-selected per-layer bpw (R2/S2, killed for weight-Hessian reasons). Structural difference: DSAF uses activation-space outlier spread (CF3/v2-CF1 measurement) as the scheduling signal, not weight-side Hessian or perplexity. No prior work uses per-layer activation Jaccard spread to schedule W_Q rank; SmoothQuant, OS+, PrefixQuant all treat outlier handling as uniform across layers. The coupling arrow from activation-space outlier distribution to weight-space rank schedule is unmeasured.

### Smallest experiment
Testable claim: Spearman ρ(spread_ℓ, layer_depth) ≥ 0.60 across all 28 layers of Qwen3-1.7B-Base, where spread_ℓ = fraction of channels covering 90% of outlier events at K=1%.
Method: extend v2-CF1 PDAP measurement to compute per-channel event-count histogram across 200 prompts per layer; derive spread_ℓ from per-layer marginal distribution; fit Spearman ρ vs layer index. Runtime: ≤ 2 hours (28-layer PDAP extension using existing infrastructure from v2-CF1).
GO threshold: ρ ≥ 0.60 AND ratio spread(L23-27) / spread(L2-6) ≥ 2.0.
NO-GO finding: depth-gradient of spread_ℓ is weak → v2-CF1's Jaccard gradient is not a reliable predictor of W_Q rank sensitivity; variable-K schedule degrades to a noisy search with no CF anchor.

### Primary risk
v2-CF1 measured Jaccard (set overlap) not spread_ℓ (coverage fraction); these are correlated but not identical. Mitigation: spread_ℓ computable from the same data as v2-CF1 with one additional histogram pass (≤1 hr incremental).

---

## R6-RAOK-CF3EXT — RAOK Depth-Stratified 3-Tier Extension Cascade

### Name
RAOK CF3-Extended Depth-Stratified Activation Codebook (RAOK-CF3EXT) — R, Track B

### Mechanism
**Track B — compression.** The RAOK 3-tier scheme (deferred as strongest surviving Track B path per SUMMARY.md) assigns: T1 = top-0.1% channels FP16 static (Jaccard 0.718 — channel-static confirmed, CF3/v2-CF1), T2 = 0.1%–1% INT8 per-token dynamic, T3 = INT4 bulk. v2-CF1 (all 28 layers) adds the depth profile: K=1% Jaccard ranges from 0.22 (layers 2-6, most dynamic) to 0.531 (L27). The extension: in layers where Jaccard at K=1% exceeds 0.45 (layers 14–27 of Qwen3-1.7B), T2 is semi-static — these layers' 20 most-active channels change less across tokens, so T2 can use a layer-local codebook of 16 entries (pre-calibrated) instead of full per-token dispatch. This reduces T2 dispatch overhead (18 scatter loads per GEMV row → 1 lookup per GEMV row) for ~half the layers at negligible quality cost. The Reach arithmetic: at 70B, if 40 of 80 layers use codebook-T2 instead of per-token dynamic T2, the T2 inference overhead drops from O(n_T2 scatter loads × 80 layers) to O(n_T2 scatter loads × 40 + 40 table lookups). At 11.5 GB/s DRAM with T2=20 channels/layer, this is ~4 µs/layer savings × 40 layers = 160 µs/token → ~0.15 tok/s improvement.

### Residency arithmetic
- Qwen3-1.7B full model (R5-RAOK-FULL architecture): T1 channels at FP16 (2 chans/layer) + T2 at INT8 (18 chans/layer) + T3 at INT4 (rest). Total ≈ 2.0 GB for 1.7B.
- 70B: T1 = 2 chans × 8192 tokens-wide × 80 layers × 2B = 2.62 MB (negligible). T2 = 20 chans × 8192 × 80 × 1B = 13.1 MB. T3 = 8192 × 8192 × 80 × 0.5B per W_up = 2.15 GB for W_up/W_gate. W_down similarly ≈ 2.15 GB at INT4. Total MLP ≈ 4.30 GB.
  RAOK-CF3EXT codebooks for T2 semi-static layers: 40 layers × 16 entries × 20 channels × 1B = 12.8 KB — negligible residency.
  Total model with RAOK-CF3EXT + attention (ATTNQ-FULL path): 4.30 + 1.17 + 0.30 ≈ 5.77 GiB → 11.5/5.77 ≈ 2.0 tok/s.
- Quality: depth-stratified T2 reduces T2 mismatch in semi-static layers. Estimated ΔNLL improvement vs uniform RAOK: 0.05–0.15 nats. Small but free at matched residency.
- P(end-to-end): v2-CF1 confirms depth gradient exists. P(codebook-T2 matches per-token-T2 quality within 0.05 nats) ≈ 0.75. P(RAOK base experiment succeeds) = prerequisite (not yet run). P(conditional end-to-end) ≈ 0.65 × 0.75 ≈ 0.49.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: PDAP (run_001 cheap test, source of CF3 which RAOK-CF3EXT builds on). Structural difference: this extension adds depth-stratification to the T2 tier based on per-layer Jaccard profiles, a scheduling decision not in PDAP or base RAOK. Closest published: SmoothQuant / PrefixQuant apply uniform per-layer outlier handling; neither stratifies T2 behavior by depth using measured Jaccard gradients. The v2-CF1 depth profile is a v2-pipeline-produced finding with no publication neighbor.

### Smallest experiment
Testable claim: applying codebook-T2 (16-entry pre-calibrated per-layer codebook) to layers with K=1% Jaccard > 0.45 reduces ΔNLL by ≤0.02 nats vs per-token-dynamic T2 on those layers in Qwen3-1.7B, at matched T2 budget.
Method: select layers with v2-CF1 Jaccard > 0.45 (expected ~10 of 28); build 16-entry k-means codebook on T2 channels from calibration data; replace per-token T2 with codebook lookup; measure ΔNLL difference on 512-token WikiText-2 held-out.
Runtime: ≈ 45 min (v2-CF1 infrastructure + k-means on calibration data + ΔNLL measurement).
GO threshold: ΔNLL degradation from codebook-T2 vs per-token-T2 ≤ 0.02 nats on the high-Jaccard layers.
NO-GO finding: codebook-T2 degrades ΔNLL by >0.02 nats → the 0.45 Jaccard threshold is insufficient for semi-static treatment; depth-stratification of T2 is not viable at ≤16 codebook entries.

### Primary risk
RAOK base (per-token dynamic T2) not yet implemented; this extension depends on it. Mitigation: this cascade rung is explicitly downstream — implement and validate RAOK base first; this is a scheduled Stage 2 addition once R5-RAOK-FULL runs.

---

## R6-WDOWN-CONDP — W_down Conditional-Precision Routing via CF1 [FREE SWING]

### Name
W_down Conditional-Precision Activation Router (WDOWN-CONDP) [FREE SWING] — R, Track A

### Mechanism
**Track A — arch-transposition.** CF1 establishes that SwiGLU post-activation output magnitude is dominated by W_up·x. CF3/v2-CF1 confirm that the top-K active channels rotate per token at operational K. This proposal: at inference time, after computing a = silu(W_gate·x) ⊙ (W_up·x), partition neuron indices into a "hot set" H = top-K% by |a_i| and a "cold set" C. Apply W_down[:, H] at INT8 precision (higher fidelity on the high-magnitude neurons) and W_down[:, C] at INT2 precision. The routing is post-activation (no prediction required — actual post-SwiGLU values determine the partition). The arch-transpose is not in the storage of W_down (which remains full-rank in memory, satisfying CF8) but in the computational path: two GEMV calls (INT8 on H, INT2 on C) replace one INT4 GEMV. The structural advantage: CF1 says H neurons dominate the output; INT8 precision on them avoids quantization error where it costs the most; INT2 on C is nearly free since |a_C| are small. This is **not** low-rank W_down compression (CF8 kills that) and **not** magnitude pruning (C neurons contribute non-zero via INT2 path). The algebraic identity: y = W_down·a = W_down[:, H]·a_H + W_down[:, C]·a_C, exact splitting, no approximation in the partition itself. Approximation is only in the quantization precision per partition.

### Residency arithmetic
- 70B W_down: shape 28672×8192 per layer, 80 layers. At uniform INT4: 28672×8192×0.5B×80 = 11.53 GB.
- WDOWN-CONDP: H = top-10% = 2867 neurons per layer. W_down[:, H] at INT8: 2867×8192×1B×80 = 1.88 GB. W_down[:, C] at INT2 (90%): 25805×8192×0.25B×80 = 4.24 GB. Total W_down = 6.12 GB vs 11.53 GB INT4. Save 5.41 GB.
- However: storage must accommodate the non-interleaved column partitioning; random-access column reads on NVMe degrade throughput. A layout solution: store H columns contiguously (hot block), C columns in INT2 compressed block. With ik_llama.cpp GGUF-level block layout, this is a file-layout decision, not a kernel change.
- 70B total: NanoQuant MLP minus W_down saving = (5.35 GB - portion from W_down) + 6.12 GB W_down + attention 1.17 GB + other 0.30 GB. NanoQuant is 0.55 bpw total; assuming W_down is ~25% of MLP = 1.34 GB of 5.35 GB, the non-W_down MLP = 4.01 GB. With WDOWN-CONDP W_down at 6.12 GB: total = 4.01 + 6.12 + 1.17 + 0.30 = 11.60 GB. EXCEEDS budget. The scheme reduces W_down from NanoQuant-style to WDOWN-CONDP only when WDOWN-CONDP achieves < NanoQuant (0.55 bpw); current WDOWN-CONDP effective bpw = (1.88×8 + 4.24×8) / (28672×8192×80) ≈ 2.14 bpw — worse than NanoQuant 0.55. This proposal does NOT improve residency vs NanoQuant; its payoff is QUALITY at fixed computation, not residency. At 4 bpw W_down, replacing INT4 with WDOWN-CONDP at same residency: tune H size so INT8 hot fraction has same bytes as INT4 full → H size = total×(0.5/1) × 1/1 = 50% at INT8 (1B) + 50% at INT2 (0.25B) = 0.625 bpw on average × d_int/d_total. Equal-residency formulation: x INT8 + (1-x) INT2 = 0.5 → x = 1/6 ≈ 17%. So top-17% neurons at INT8, rest INT2. Same bytes as INT4 across all neurons, with quality improvement concentrated on the 17% that matter.
- Equal-residency quality gain: CF1 says top-10-20% neurons dominate output; going from INT4 to INT8 on them at zero residency cost (compensated by INT2 on the rest) reduces quantization error where it counts. Estimated ΔNLL improvement vs uniform INT4 W_down: 0.10–0.30 nats.
- P(end-to-end success as quality-at-fixed-residency): P(top-17% H neurons dominate W_down output error) × P(INT2 on C neurons acceptable ΔNLL) ≈ 0.65 × 0.70 ≈ 0.455.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: CLUBS/RSTD (W_up rank reduction — dead). Structural difference: WDOWN-CONDP does not reduce W_down rank or truncate any matrix; it reroutes computation to precision-stratify the column-wise contribution to the output, exploiting CF1's firing-rank dominance not as a predictor but as a post-hoc partition key. Closest published: MoEfication (Zhang et al., ICLR 2022) clusters FFN neurons into experts; Deja Vu (ICML 2023) predicts sparse attention/FFN; both work on ReLU sparsity. WDOWN-CONDP works on SwiGLU's continuous firing-rank and requires no sparse-prediction infrastructure — the partition is computed from the actual post-activation values, not predicted. No published scheme does post-activation magnitude-partitioned precision routing on W_down without modifying W_down's storage structure.

### Smallest experiment
Testable claim: applying top-17% H neurons at INT8 and bottom-83% C neurons at INT2 (equal-residency formulation) on W_down[14] of Qwen3-1.7B reduces ΔNLL vs uniform INT4 W_down[14] by ≥ 0.05 nats on 512-token WikiText-2, single-layer substitution.
Method: quantize W_down[14] columns in two precision tiers (top-17% by calibration-mean post-activation magnitude at INT8; rest at INT2); run forward pass with substituted layer; measure ΔNLL delta.
Runtime: ≈ 30 min.
GO threshold: ΔNLL reduction ≥ 0.05 nats vs uniform INT4 at matched W_down bytes.
NO-GO finding: post-activation magnitude partition provides no quality advantage vs uniform INT4 → CF1's firing-rank dominance does not translate to quantization-error asymmetry; the quantization error of the tail (INT2) overwhelms the gain on the head (INT8).

### Primary risk
INT2 tail quantization error may be catastrophic: cold neurons are small in magnitude but non-zero, and INT2 (4 levels per weight) may introduce errors that sum to large output perturbations at the d_model=2048 projection output. Mitigation: measure INT2 tail-only ΔNLL as a first sub-experiment (quantize only the C partition; leave H at bf16); if tail ΔNLL > 0.50 nats at the single-layer level, the scheme is dead before full cascade investment.

---

## R6-MLA-POSTRAIN — Post-Training MLA Conversion for 70B via CF11 Head Redundancy

### Name
Post-Training MLA Conversion via Head Redundancy (MLA-POSTRAIN) — R, Track A

### Mechanism
**Track A — arch-transposition.** CF11's most striking finding: global W_Q at K=128 achieves ΔNLL=+0.98 nats — meaning 16 query heads collectively span only a ~128-dim subspace (one head's worth). This is not per-head rank reduction (per-head K=64 NO-GO at +1.53 nats); it is head-redundancy: the heads are not individually compressible but their joint Q manifold is 8× smaller than their combined storage. The arch-transposition: apply a post-training MLA (Multi-head Latent Attention) conversion — replace the 16 separate W_Q matrices with a single shared low-rank bottleneck W_Q_shared ∈ ℝ^{d×128} plus 16 per-head up-projection W_Q_h ∈ ℝ^{128×d_head} (where d_head=128 in Qwen3). At inference, the query computation becomes: q_h = x W_Q_shared W_Q_h, a two-step matmul with an intermediate 128-dim bottleneck. This is exactly MLA's structure (DeepSeek-V2) but applied post-training via calibration-fitted joint factorization of the stacked W_Q matrix, not trained from scratch. SUMMARY.md explicitly notes "MLA-style joint Q-K projection in head space is empirically motivated for Qwen3 post-training (as opposed to the training-time MLA in DeepSeek-V2)" following CF11. Not cross-layer (v2-CHEAP-TEST-001 killed cross-layer W_Q; this is per-layer across-heads). Not PALU+TransMLA (pre-empted for training-time use; this is a post-training factorization whose feasibility is unverified for Qwen3).

### Residency arithmetic
- Qwen3-1.7B (d=2048, 16 heads, d_head=128, K_shared=128, 28 layers):
  Original W_Q: 16 × d × d_head × 28 × 2B = 16 × 2048 × 128 × 28 × 2B = 235 MB.
  Post-MLA W_Q_shared: d × K_shared × 28 × 2B = 2048 × 128 × 28 × 2B = 14.7 MB.
  Post-MLA W_Q_h: 16 × K_shared × d_head × 28 × 2B = 16 × 128 × 128 × 28 × 2B = 14.7 MB.
  Total post-MLA W_Q: 29.4 MB vs 235 MB — 8× compression.
- 70B (d=8192, 64 heads, d_head=128, 80 layers):
  Original W_Q: 8192² × 2B × 80 = 10.74 GB.
  Post-MLA: W_Q_shared = 8192 × 128 × 2B × 80 = 167 MB; W_Q_h = 64 × 128 × 128 × 2B × 80 = 167 MB. Total = 334 MB.
  Compression: 10.74 GB → 0.334 GB (32×).
- Combined 70B residency: NanoQuant MLP 5.35 GB + W_Q MLA 0.334 GB + W_K K=256 0.17 GB + W_V/W_O (untested, assume CF11-class ~1.0 GB) + other 0.30 GB ≈ 7.15 GiB → within 7.28 GiB.
- DRAM bandwidth: 7.15 / 11.5 ≈ 0.622 s/token → 1.61 tok/s. Combined with attention GEMV reduction (shared bottleneck reduces D→128→per_head vs D→per_head: compute reduction ~2× for Q) → effective ~1.75 tok/s.
- Quality: CF11 K=128 global ΔNLL=+0.98 nats. Post-training MLA factorization aims to match this — the calibration-fitted W_Q_shared and W_Q_h jointly reconstruct W_Q_full with the same rank-128 truncation. If the factorization fits correctly, ΔNLL = CF11's +0.98. If the per-head up-projections add reconstruction error, ΔNLL may be higher. Target: ΔNLL ≤ 1.20 nats.
- CF10 conditioning check: W_Q_shared has d×K_shared = 2048×128 = 262K params. Calibration: 1K tokens × 2048 output dims per layer = 2M values. Ratio = 2M/262K ≈ 7.6 — well-conditioned. W_Q_h: 16×128×128 = 262K params per layer, same ratio. No CF10 risk.
- P(end-to-end): P(calibration-fitted factorization matches CF11 ΔNLL within +0.22 nats) × P(inference-time two-step GEMV produces expected compute savings) ≈ 0.65 × 0.85 ≈ 0.55.

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: v2-CHEAP-TEST-001 (cross-layer W_Q class kill) — structural difference: this is within-layer cross-head factorization, not cross-layer. CF11's kill is per-head within-layer K=64; MLA-POSTRAIN uses global K=128 (the confirmed GO path) and distributes the bottleneck via per-head up-projections. Closest published: TransMLA (arXiv:2506.01189) and PALU (arXiv:2407.21118) do post-training MLA conversion; PALU+TransMLA is explicitly listed as pre-empted "without structural value-add." The structural value-add here: CF11's empirical head-redundancy measurement (16 heads → 128-dim joint subspace) is the anchor that TransMLA/PALU do not have for Qwen3 specifically. The factorization rank K=128 is not a hyperparameter guess — it is derived from the CF11 measurement. This is the distinction: CF11-grounded rank selection vs arbitrary rank choice.

### Smallest experiment
Testable claim: calibration-fitted post-training MLA factorization of W_Q at K_shared=128 on Qwen3-1.7B achieves ΔNLL ≤ 1.20 nats on 512-token WikiText-2 held-out (vs CF11's +0.98 nats from pure SVD truncation).
Method: stack all 16 per-head W_Q[14] matrices into shape (16, d, d_head) = (16, 2048, 128) = (2048, 2048). SVD → take top-128 directions for W_Q_shared; solve for per-head W_Q_h via least-squares. Apply reconstructed Q to forward pass; measure ΔNLL.
Runtime: ≈ 30 min (single layer, same infrastructure as AQFKV).
GO threshold: ΔNLL ≤ 1.20 nats (≤ 0.22 nats above CF11's pure SVD result, which represents factorization reconstruction overhead).
NO-GO finding: factorization error adds > 0.22 nats above pure SVD → the per-head up-projection structure introduces representation mismatch that pure global truncation avoids; post-training MLA conversion requires additional calibration optimization (gradient-based fine-tuning of W_Q_h).

### Primary risk
TransMLA/PALU may already cover this at the frame level even with the CF11-anchor distinction, triggering a Stage 2 novelty kill. Mitigation: the structural value-add claim (CF11-derived K_shared vs arbitrary hyperparameter) must be made explicit and testable — the experiment should compare CF11-derived K=128 vs arbitrary K=64 vs K=256 to demonstrate the empirical grounding of the rank choice.

---

## Convergence handles

- W_V and W_O spectra: r_99/d measurements (gates R6-WVOP, R6-ATTNQ-FULL; cheaply resolved in ~20 min each)
- Depth-spread_ℓ Spearman ρ vs layer index (gates R6-DSAF-70B and R6-RAOK-CF3EXT; same measurement)
- W_VO product rank vs individual matrix ranks (new measurement; different from individual matrix spectra)
- Post-training MLA factorization reconstruction gap: ΔNLL(MLA-fitted) − ΔNLL(SVD-only) (gates R6-MLA-POSTRAIN)
- W_down precision-tier quality interaction: INT8 hot + INT2 cold vs uniform INT4 (gates R6-WDOWN-CONDP)
- Joint 4-matrix SVD ΔNLL interaction term: super-additive vs additive (gates R6-ATTNQ-FULL)
