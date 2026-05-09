# Stage 1 — Orientation R (Reach) — Run 005

Orientation: R — Reach
Independent of runs 001–004.
Kill enforced: v2-CHEAP-TEST-001 (cross-layer W_Q stacked SVD / shared basis — class dead).

---

## R5-AWQKV — Attention Weight Joint Quartet Compression

**Track B — compression.**

CF11 establishes W_Q r_99/d ≈ 0.63 and W_K r_99/d ≈ 0.79 with a global GO at K=128 (+0.98 nats) and K=512 (+0.20 nats) respectively. W_V and W_O spectra are explicitly noted in SUMMARY.md as "untested — cheap parallel measurement." The Reach claim: all four attention weight matrices (W_Q, W_K, W_V, W_O) per layer are jointly compressible at 2–4× per matrix — not by sharing subspaces across layers (KILLED, v2-CHEAP-TEST-001) but by per-layer SVD truncation applied to each of the four matrices independently, stacked multiplicatively. If W_V and W_O have spectra similar to W_K (r_99/d ≈ 0.63–0.79), the four-matrix joint compression delivers 6–8× attention-weight residency reduction. For Qwen3-1.7B with 28 layers × 4 matrices × 2048² × 2 bytes ≈ 1.88 GB attention weights: 6× reduction saves ~1.57 GB. For a 70B analogue (scaling ~ (70/1.7)² ≈ 1695× by parameter count in attention): attention weights ≈ 7.7 GB at bf16; 6× reduction frees ~6.4 GB. Combined with GGUF Q4 for MLP weights, total 70B residency ≈ (70B × 4-bit / 8) - 6.4 GB ≈ 4.1 GiB — within 7.28 GiB budget at ≥3 tok/s on DRAM.

**Residency arithmetic (7.28 GiB Ryzen target).**
- 70B baseline at Q4_0 (≈4.5 bpw): ~39.4 GB → NVMe-resident, ~3 GB/s stream → ~0.4 tok/s (too slow).
- 70B baseline at NanoQuant 0.55 bpw: ~5.35 GiB → DRAM-resident if it fits. DRAM bandwidth 11.5 GB/s → ~2.15 tok/s.
- Attention weights at bf16 in 70B (28 heads × 8 GQA groups × 4 matrices × 128 head-dim × 8192 d_model): W_Q = 8192×8192×28×2B ≈ 3.86 GB; W_K, W_V = GQA shared across 8 groups → 1024×8192×28×2B ≈ 0.48 GB each; W_O = 8192×8192×28×2B ≈ 3.86 GB. Total attention ≈ 8.7 GB at bf16.
- At K=256 for W_Q/W_O (≈ 3.1× compression) and K=512 for W_K/W_V (≈ 2× compression, roughly matching CF11 W_K GO threshold), attention block ≈ 8.7 / 2.5 ≈ 3.5 GB — saves ~5.2 GB vs bf16 attention. If MLP weights already at 4.5 bpw (~0.56 bpw effective after NanoQuant-class scheme), the combined DRAM-resident 70B = 5.35 GiB (MLP NanoQuant) + 3.5 GB (SVD attention) = ~8.85 GB — just over budget. However, at K=128 for W_Q/W_O (CF11 GO threshold +0.98 nats) the attention block ≈ 8.7 / 4 ≈ 2.2 GB → total ≈ 7.55 GiB. Close. Quality: +0.98 nats W_Q + untested W_V/W_O — cascade rung 1 (W_V/W_O spectrum measurement) is the critical gate.
- This is a 2-rung cascade: (1) measure W_V / W_O spectra → if r_99/d < 0.80, the 4-matrix joint compression holds; (2) combine with NanoQuant-class MLP compression for DRAM residency.

**Novelty gloss.**
Closest kill-list item: CF11 per-head W_Q rank truncation NO-GO (killed). Structural difference: this proposal uses per-layer *global* SVD (all heads jointly, not per-head), which is the GO path per CF11. No kill-list item covers W_V or W_O spectra — SUMMARY.md explicitly calls them untested. Closest published method: A3 (arXiv:2505.12942) compresses W_Q/W_K via attention-logit error minimization; does not address W_V/W_O or 70B deployment arithmetic. AQFKV measured W_Q and W_K; W_V/W_O are a natural extension not yet on the map.

**Smallest experiment.**
Claim: W_V and W_O in Qwen3-1.7B have r_99/d < 0.85 (implying ≥1.18× compression at K matching CF11 W_Q thresholds).
Test: SVD spectrum of W_V[L14] and W_O[L14] on Qwen3-1.7B-Base bf16; plot singular value decay; compute r_99 and ΔNLL at K ∈ {128, 256, 512} with W_Q/W_K/MLP held at bf16.
Runtime: ≈15–20 min on Ryzen 5 7530U (two matrices, same size as W_Q).
GO threshold: r_99/d < 0.85 for W_V OR W_O AND ΔNLL < 1.5 nats at K=256.
NO-GO finding: W_V/W_O are full-rank like MLP weights → CF8 boundary sharpening; extends the spectrum taxonomy; attention compression ceiling is lower than hoped but still better than MLP.

**Primary risk.**
W_V and W_O may be full-rank (r_99/d ≈ 1.0 like MLP) because the output projection W_O's gradient path more closely resembles W_up than W_Q. Mitigation: cheapest check is the 20-min spectrum measurement; if NO-GO, fall back to W_Q + W_K only (which AQFKV already established), still giving ~2× attention compression.

---

## R5-RAOK-FULL — RAOK 3-Tier Activation Codebook Full-Stack Cascade

**Track B — compression (activation quantization).**

RAOK was the run_001 Track B selection — a 3-tier activation scheme grounded in CF3 (K-dependent Jaccard: 0.1% static, 1% dynamic). This proposal escalates RAOK to a full-stack 70B deployment cascade. The 3-tier scheme: (T1) top ~2 channels (K=0.1%) pinned as FP16 static vectors, loaded once and reused; (T2) next ~18 channels (0.1%→1%) handled as per-token INT8 dynamic with a small per-layer codebook (CF10-compliant: ≤256 entries fit in 256 × 2048 × 1 byte = 512 KB per layer, far below calibration-limit); (T3) remaining 2028 channels (99%) INT4 static. AVX2 GEMV processes T3 in bulk INT4, adds T2 INT8 scatter, adds T1 FP16 residual. The cascade stages: Stage 1: validate T2 Jaccard stability at K=0.9% (18 channels) on Qwen3-1.7B → if inter-token Jaccard < 0.35 (confirming dynamicity), T2 codebook needs per-token dispatch, not static; Stage 2: implement 3-tier AVX2 GEMV kernel for W_up/W_down application; Stage 3: measure ΔNLL at 3-tier on 70B (via Qwen3-7B extrapolation) and tok/s on Ryzen.

**Residency arithmetic.**
- 70B at 3-tier: T3 (99% of weights) at INT4 = 0.5 bpw of total → T3 contributes 70B × 0.99 × 0.5 bits / 8 = 4.34 GB. T2 (1% of weights) at INT8 = 1 bpw contribution → 70B × 0.01 × 1 bit / 8 = 87.5 MB. T1 (0.1% of weights) at FP16 → 17.5 MB. Plus T2 codebook per layer: 28 layers × 6144 (W_up intermediate) × 18 channels × 1 byte ≈ 3 MB (negligible). Total ≈ 4.34 + 0.09 + 0.02 ≈ 4.45 GB → within 7.28 GiB. Effective bpw ≈ 4.45 × 8 / 70 × 10⁹ ≈ 0.51 bpw.
- DRAM bandwidth utilization: 4.45 GB × 11.5 GB/s⁻¹ → ~0.39 s/token → ~2.57 tok/s. Better than NanoQuant baseline (2.15 tok/s at 5.35 GiB) because RAOK compresses more aggressively and T1/T2 dispatch overhead is small (≤ 2% of total GEMV compute on AVX2).
- Quality cost: T3 at INT4 is the dominant loss. With T2 INT8 patching the 18 most dynamic channels, expected ΔNLL over clean INT4 is ≈ -0.10 to -0.20 nats improvement. Total vs bf16: ≈ INT4 ΔNLL minus correction.

**Novelty gloss.**
Closest kill list: PDAP (run_001 selection, confirmed K-dependent Jaccard = CF3) — RAOK-FULL is its escalation to 70B deployment arithmetic with an explicit AVX2 kernel design. Closest published: SmoothQuant handles channel-static; LLM.int8() handles static top-K. RAOK differs by (a) K=0.1% static + K=0.9% per-token dynamic distinction (the split CF3 measures, not present in prior work) and (b) explicit residency math showing sub-1-bpw territory.

**Smallest experiment.**
Claim: T2 per-token dispatch Jaccard at K=18 channels (0.9% of 2048) is < 0.35 on Qwen3-1.7B normal tokens, confirming the dynamic-handling is load-bearing vs static INT8.
Test: re-run PDAP Jaccard measurement at K=0.9% (18 channels) with separate bookkeeping for consecutive normal-token pairs across all 28 layers. The CF3 result gives K=1% Jaccard=0.308; 18 channels (0.879%) should be very close. Confirm that without T2 dynamic handling, quantization error in that band is measurable.
Runtime: ≈ 30–40 min on Ryzen (200-prompt run at Qwen3-1.7B, same as PDAP).
GO threshold: mean Jaccard < 0.35 at K=18 channels AND per-channel outlier frequency distribution has head >= 10 events/channel across 200 prompts (confirming a codebook is viable).
NO-GO finding: T2 channels are sparser than expected → T1-only static pinning suffices; RAOK simplifies to LLM.int8() with two pinned channels, which is already known.

**Primary risk.**
AVX2 scatter for T2 dynamic channels adds memory-access latency that erodes the effective tok/s gain. Mitigation: T2 is only 18 channels out of 2048 — at most 18 additional FP32 loads per GEMV row, overhead < 2% vs the INT4 bulk work. Quantify in Stage 2 profiling.

---

## R5-VOWN — W_V W_O Structured Nullspace Folding

**Track A — arch-transposition.**

In multi-head attention, W_V ∈ R^{d×d} and W_O ∈ R^{d×d} are always applied in sequence: O = (A V) W_O where V = X W_V. This means the effective operator is W_V W_O (applied to token X via the context sum). If the product W_V W_O has lower rank than either factor — because SGD drives W_V into a subspace that W_O then maps onto a smaller subspace — we can exploit the chain algebraically: replace W_V W_O with a single rank-r matrix R = U Σ V^T (truncated SVD of the product), saving one full matmul. The arch-transposition: instead of two matmuls (X W_V and A' W_O), compute the product W_VO = W_V W_O once at load time (cheap: 2048³ FMAs, ~17ms on Ryzen), then at inference apply only R = W_VO's rank-r truncation. This is the AERO analog for the value-output chain: an algebraic fold with no quality cost up to the truncation rank. Not CF8 MLP rank reduction — this targets the V-O product, a composed object not previously measured. AQFKV (CF11) measured W_Q and W_K individually; the product W_V W_O is a different object. If rank(W_V W_O) < d because of head-routing information compression (each head's "useful" directions are a fraction of d), a 4× compression is plausible.

**Residency arithmetic.**
- W_V W_O product per layer: 2048×2048 at bf16 = 8 MB. 28 layers = 224 MB. Replacing two 2048×2048 matrices (×2 = 16 MB/layer, 448 MB total) with one rank-r matrix: at r=512, storage is 2048×512×2 + 512×2048×2 = 4 MB/layer = 112 MB total → 336 MB saved across all layers. Small absolute gain at 1.7B scale; at 70B scale (d=8192): two matrices = 8192×8192×2×28×2B ≈ 7.73 GB → rank-512 product ≈ 8192×512×2 + 512×8192×2 = 16 MB/layer × 28 = 448 MB → saves ~7.3 GB → massive. Combined with NanoQuant MLP: 5.35 GiB - 7.3 GB is negative, implying this alone gets 70B DRAM-resident with MLP at higher quality than NanoQuant requires. Quality: hinges on rank of W_V W_O product — needs measurement.
- DRAM bandwidth for 70B: if attention weights at 0.44 GB (rank-512 W_VO) + MLP at ~5 GB (NanoQuant) + residuals ≈ 5.5 GB → 5.5 / 11.5 ≈ 0.48 s/token → 2.1 tok/s.

**Novelty gloss.**
No kill-list item covers the V-O product folding. AERO (arXiv:2410.13060) folds the SwiGLU activation + W_gate/W_up — the structural analogy motivates V-O folding but the target object is different. MLA (DeepSeek-V2) also motivates joint low-rank V-O but retrains; this proposal is no-retraining post-hoc. No published post-training V-O product folding found in literature survey. The structural argument is that W_V and W_O "cooperate" to route information through heads — the product may compress more than either factor.

**Smallest experiment.**
Claim: rank(W_V[L] × W_O[L]) ≤ 0.6 × d for at least one layer of Qwen3-1.7B, measured as r_99 of the product.
Test: compute W_VO = W_V[14] @ W_O[14] (matrix multiply, ~0.1s), compute SVD of W_VO (2048×2048, ~30s), plot singular value decay curve, measure r_99. Apply rank-256 and rank-512 truncation and measure ΔNLL on 512-token WikiText-2.
Runtime: ≈ 20 min total (one layer diagnostic + NLL sweep).
GO threshold: r_99(W_VO) / d < 0.70 AND ΔNLL at rank-512 < 0.50 nats.
NO-GO finding: W_VO is full-rank → the V and O matrices span complementary subspaces and do not compress jointly; the AERO-style fold analogy does not extend to the value-output chain; structural finding extends CF8 boundary to the V-O product.

**Primary risk.**
W_V and W_O may be spectrally "complementary" — if W_V maps to d directions and W_O uses all d of them, the product is full-rank. Given CF11 shows W_Q has r_99/d=0.63, there's hope W_V is similar, but the V-O product rank depends on their joint structure, not individual spectra. Mitigation: the smallest experiment checks product rank directly at negligible cost before any implementation investment.

---

## R5-LAYERGATE — Layer-1 Gate-Fold + Global Tiered Quantization Cascade

**Track B — compression.**

CF6 establishes that Layer 1 is anomalous: 36% of gate neurons have variance < threshold and are foldable, vs <2.5% in all other layers. This is not a global compression (R4-A SDZC killed the global scheme), but Layer 1 alone has 36% × 6144 ≈ 2211 neurons whose SwiGLU gate is effectively constant over the calibration distribution. For those neurons, silu(W_gate[L1]·x) ≈ c_i (a scalar), so the contribution is c_i × (W_up[L1, i] · x). This is exact on the calibration distribution and approximates well on natural text where Layer 1 activations are constrained by the embedding space. The fold: replace W_up[L1, foldable_rows, :] with c_i × W_up[L1, i, :] and eliminate the corresponding W_gate rows — a 36% reduction in Layer 1's W_gate and W_up storage for foldable neurons. Cascade on top of tiered quantization: (1) fold Layer 1 (free, no ΔNLL on calibration); (2) apply RAOK-style 3-tier quantization to all layers; (3) measure quality on held-out. The compounding argument: Layer-1 folding saves 36% × 2 × (d_model × d_intermediate × bpw) at Layer 1; the saved FP16 capacity can be reallocated to store Layer 1 W_up at higher precision, reducing the T3 quality loss specifically where firing is most vocabulary-conditioned.

**Residency arithmetic.**
- Layer 1 W_gate at bf16: 2048 × 6144 × 2B = 25.2 MB. 36% fold eliminates 9.1 MB of W_gate and W_up combined for those neurons. Tiny at 1.7B. At 70B: d_model=8192, d_int=28672 per layer. W_gate Layer 1 = 8192 × 28672 × 2B ≈ 470 MB. 36% fold ≈ 169 MB saved. Small absolutely but: the fold is zero ΔNLL on calibration distribution, so the "quality budget" it frees can be spent increasing the precision of the remaining 64% of Layer 1 neurons or of W_up.
- More useful framing: Layer-1 folding + RAOK tiering = a cascade where the first rung is free quality and the rest pays quantization cost. The compounded target: 70B at 0.50 bpw effective (vs NanoQuant's 0.55 bpw floor) by using the Layer-1 folding to absorb some of the T3 INT4 quality loss.
- DRAM at 0.50 bpw effective: 70B × 0.50 / 8 = 4.375 GB → within budget with 2.9 GB to spare.

**Novelty gloss.**
CF6 (Layer-1 anomaly) has not been exploited by any kill-list item. R4-A SDZC killed the GLOBAL gate folding scheme, but Layer-1-only folding was explicitly noted as "real but tiny" — this proposal rescues it by making it a free quality contribution to a cascade rather than a standalone compression scheme. No published method specifically exploits the Layer-1 gate-variance anomaly in SwiGLU models. The combination with tiered quantization is novel in that it uses the folded neurons' space for quality reallocation.

**Smallest experiment.**
Claim: on Qwen3-1.7B, applying the Layer-1 gate fold (replace 36% of W_gate/W_up contributions with calibrated constants) produces ΔNLL < 0.02 nats on held-out WikiText-2.
Test: load Qwen3-1.7B-Base, identify foldable gate neurons at Layer 1 (std < 0.05 threshold, using R4-A SDZC calibration data), compute c_i = mean(silu(W_gate[L1, i] · x)) over calibration corpus, replace W_gate[L1, foldable, :] with zero and add c_i × W_up[L1, i, :] into a residual bias vector, measure ΔNLL on 512-token WikiText-2 held-out.
Runtime: ≈ 15–20 min (calibration pass + NLL eval, Qwen3-1.7B).
GO threshold: ΔNLL < 0.02 nats on held-out (quality budget of ~0 is expected since these neurons are calibration-constant).
NO-GO finding: ΔNLL > 0.05 nats → gate output variance on held-out exceeds calibration, i.e., the "foldable" gate neurons are context-conditioned in ways the 566-token calibration corpus didn't capture; CF6 36% figure is calibration-specific and does not generalize.

**Primary risk.**
The 566-token calibration corpus (R4-A SDZC) may have undersampled the long-tail token distribution that activates dormant Layer-1 gate neurons; the fold produces garbage on out-of-distribution inputs. Mitigation: expand calibration to 5K diverse tokens before applying fold; measure ΔNLL stratified by token type (code, math, natural language).

---

## R5-ATTNBAND — [FREE SWING] Attention Score Bandwidth Quantization via Softmax Invariance

**Track B — compression.**

Softmax is shift-invariant: softmax(x + c·1) = softmax(x) for any scalar c. This means the attention score matrix S ∈ R^{L×L} (before softmax) can be stored with any additive per-row shift removed. Exploit: represent S not in absolute terms but as S - row_mean(S) · 1^T, then quantize the centered scores to INT4 or INT2. Since semantic content of S is encoded in the *relative* score differences within a row (who attends to whom), and the mean is a free gauge parameter, centering gives the quantization scheme maximum dynamic range on the content-carrying dimension. For KV cache: this isn't about weights — it's about the on-the-fly score tensor during attention. This matters primarily when the attention score matrix is large (long contexts, speculative decoding with large draft windows). The cascade: (1) measure the row variance distribution of S across layers in Qwen3-1.7B at varying sequence lengths (512, 2K, 8K tokens); (2) compare ΔNLL of INT4-centered-scores vs INT8-raw-scores; (3) at long context (32K), the score tensor L²=10⁹ FP16 values = 2 GB per batch → INT4 centered = 0.5 GB; 4× reduction in score-tensor bandwidth.

**Residency arithmetic.**
Attention score tensor at L=4096 tokens, 28 layers, 16 heads: 4096² × 28 × 16 × 2B = 14.9 GB per batch. At INT4 centered: 3.7 GB. Not stored persistently (computed per token in autoregressive decode — score is L×1 per new token, not L×L). The real payoff is speculative decoding tree attention (where multiple draft tokens are processed simultaneously, score tensors are L_draft×L_prefix and can be large) and long-context batch inference. At L=32K, score per new token = 32K × 16 heads × 2B = 1 MB → INT4 = 0.25 MB. Small for single token. The mechanism is more valuable for prefill bandwidth reduction. At 32K prefill, 28 layers: 32K² × 28 × 16 × 2B = 1.88 TB → INT4 centered: 470 GB. Not stored, but bandwidth during prefill determines time-to-first-token.

**Novelty gloss.**
No kill-list item covers attention-score quantization via softmax shift-invariance. Closest published: FlashAttention stores scores in tiles and uses softmax normalization implicitly — does not explicitly exploit the gauge freedom for storage quantization. INT8 attention scores have been explored in quantization papers (SmoothQuant, etc.) but not INT4 via centered quantization. The softmax-invariance argument is mathematically clean and does not depend on any SUMMARY.md finding — it follows from the softmax definition. [OUT-OF-ORIENTATION note: this idea doesn't fit the "70B DRAM-residency" Reach frame cleanly but has an architectural elegance argument.]

**Smallest experiment.**
Claim: centered-INT4 attention scores produce ΔNLL < 0.10 nats vs FP16 scores on Qwen3-1.7B at L=512.
Test: hook into Qwen3-1.7B-Base forward pass, intercept pre-softmax scores per layer, apply row-centering + INT4 quantization (symmetric, scale = max(abs(S - row_mean(S))) / 7), re-run softmax on quantized scores, measure ΔNLL on 512-token WikiText-2.
Runtime: ≈ 20–30 min (instrumented forward pass, PyTorch hooks).
GO threshold: ΔNLL < 0.10 nats at INT4, < 0.02 nats at INT8 centered (sanity).
NO-GO finding: centering doesn't increase dynamic range enough at INT4 → attention scores have heavy tails that INT4 can't represent even centered → quantization research must look elsewhere (INT6 or learned per-head scales).

**Primary risk.**
Attention softmax scores have heavy tails (a few dominant keys cause extreme score values) that centering doesn't resolve. The maximum absolute deviation post-centering may still be large, making INT4 lossy. Mitigation: measure the actual per-row score distribution on Qwen3-1.7B before committing; if max(|S_ij - row_mean|) is dominated by outliers, apply per-head clipping + centering.

---

## R5-WVWK-MLA — Post-Training MLA-Style W_V/W_K Joint Projection

**Track A — arch-transposition.**

CF11 shows 16 W_Q heads collectively span a ~128-dim subspace: per-layer W_Q stacks to rank ~128. This suggests MLA-style joint low-rank Q-K projection (noted as "empirically motivated" in CF11) is tractable post-training without retraining. This proposal extends to the V/K pair specifically: in GQA (Qwen3-1.7B uses grouped-query attention), W_K and W_V both operate on a shared set of key-value heads. If the per-layer stacked W_K (shape: n_kv_heads × d_head × d_model) has a joint subspace of rank r_K << n_kv_heads × d_head, then a MLA-style decomposition W_K = W_K_down W_K_up (d_model → r_K → n_kv_heads × d_head) reduces KV cache memory during inference: KV cache per token = n_kv_heads × d_head × 2 bytes → after MLA-fold: r_K × 2 bytes. At r_K=128 vs n_kv_heads × d_head=8×128=1024: 8× KV cache compression. For 32K context: 32K × 28 layers × 8 kv-heads × 128 dim × 2 × 2B = 3.67 GB → with MLA fold at r_K=128: 32K × 28 × 128 × 2B = 458 MB. This is the right shape for long-context deployment on 7.28 GiB. The arch-transposition: replace W_K and W_V inference with down-projected KV states cached at r_K dimensions, up-projected at attention-score and value-aggregation time. No retraining — calibrate the down/up split via least-squares on activations (CF10-compliant: n_params = r_K × d_model + n_kv_heads × d_head × r_K = 128×2048 + 1024×128 = 394K; n_samples = 5K tokens × 2048 dims = 10M independent samples → well-conditioned by CF10).

**Residency arithmetic.**
- W_K / W_V weight storage: 28 × 2 × (8 × 128 × 2048) × 2B = 471 MB bf16 → after MLA at r_K=128 → 28 × 2 × (2048 × 128 + 128 × 1024) × 2B = 296 MB. Modest weight saving (175 MB). The main payoff is KV cache: at 32K context → from 3.67 GB to 458 MB (8× reduction). At 4K context: 458 MB → 57 MB, marginal.
- Combined with weight compression: if 70B runs from NVMe at 0.55 bpw (NanoQuant) and context is 32K: KV cache would otherwise require 3.67 GB × 70B/1.7B weight scaling factor (n_kv_heads and d_head scale differently — 70B has n_kv_heads=8, d_head=128, d_model=8192, n_layers=80): KV cache per token = 8 × 128 × 80 × 2 × 2B = 327 KB/token; at 32K: 10.5 GB → doesn't fit in 7.28 GiB alongside weights. With MLA at r_K=256: 80 × 256 × 2B × 32K = 1.31 GB → KV + weights fit.
- This enables 70B at 32K context within 7.28 GiB. DRAM bandwidth for weight streaming is the bottleneck; KV cache access pattern is sequential and cache-friendly.

**Novelty gloss.**
Closest kill list: v2-CHEAP-TEST-001 (cross-layer W_Q shared basis — KILLED). Structural difference: this proposal is per-layer W_K/W_V joint decomposition within a single layer's GQA group (not cross-layer). The killed scheme stacked across layers; this exploits within-layer structure of the GQA kv-head group. Closest published: MLA (DeepSeek-V2) is the training-time version. Post-training MLA-style compression of existing GQA models via calibration-fitted projection is not published. CF11's explicit "MLA-style joint Q-K projection empirically motivated for Qwen3 post-training" anchors the novelty claim.

**Smallest experiment.**
Claim: stacked W_K across n_kv_heads in Qwen3-1.7B has r_99 / (n_kv_heads × d_head) < 0.70 at layer 14 (the same layer CF11 used for W_Q diagnosis).
Test: reshape W_K[14] from (8 heads × 128 × 2048) to (1024 × 2048), compute SVD, measure r_99. Apply rank-128 and rank-256 approximation, measure ΔNLL with W_Q/W_V/W_O/MLP at bf16.
Runtime: ≈ 15 min (same procedure as AQFKV W_K global).
GO threshold: r_99 / (n_kv_heads × d_head) < 0.70 AND ΔNLL at K=256 < 0.50 nats.
NO-GO finding: W_K stacked across GQA heads is full-rank → head diversity in the KV group is genuine and a low-rank KV projection would lose information; MLA post-training not viable for Qwen3 GQA without retraining.

**Primary risk.**
GQA shares W_K/W_V across 2 (Qwen3-1.7B, 16 heads / 8 kv-heads = 2 heads per kv-head) query heads per key-value head — the KV weight matrices are already smaller than the QO matrices. Compressing them further may be below the calibration-fitting noise floor. Mitigation: CF10 conditioning check is explicit (396K params vs 10M calibration values — well-conditioned), so calibration is not the failure mode; the structural question is whether the 8 kv-heads in a layer span a low-rank subspace.

---

## Convergence handles

- W_V and W_O spectra (untested per SUMMARY.md — cheaply resolvable; multiple ideas depend on whether the attention weight class is uniformly low-rank or split)
- W_V W_O product rank (fold-chain rank ≠ individual matrix ranks — new object not yet measured)
- GQA kv-head stacked-W_K rank (within-layer kv-head group subspace, distinct from cross-layer W_Q kill)
- K=0.1% vs K=0.9% activation Jaccard stratification (RAOK-FULL pivots on this split being sharp and stable)
- Layer-1 gate-fold held-out generalization (CF6 36% figure is calibration-specific; generalization is the gate for the Layer-1 folding cascade)
- MLA-style per-layer decomposition tractability without retraining (CF11 motivates; empirical cascade needed)
