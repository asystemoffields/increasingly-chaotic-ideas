# Stage 1 — Orientation R (Reach) — Run 004

Orientation: R — Reach. Independent of runs 001–003.
Kill observed: v2-CHEAP-TEST-001 / Cluster C1 (cross-layer W_Q shared basis). Per-layer K=128 is the W_Q ceiling.

---

## R4-1 — Attention-Weight Cascade: W_Q + W_K + W_V + W_O Joint Residency Reduction (AWCJR)
**R / Track B**

### Mechanism
Track B. CF11 confirmed W_Q global K=128 at ΔNLL=+0.98 nats, W_K global K=512 at ΔNLL=+0.29 nats. W_V and W_O spectra are UNTESTED — the AQFKV report flags them as cheap follow-ups. If W_V and W_O exhibit concentration comparable to W_K (r_99/d < 0.80), all four attention projection matrices admit simultaneous rank reduction. The Reach claim: a joint W_Q@K=128 + W_K@K=256 + W_V@K=256 + W_O@K=256 sweep of all 28 layers of Qwen3-1.7B compounds to meaningful attention-weight residency reduction at total ΔNLL < 1.5 nats, opening a real compression lever that does NOT touch MLP weights (respecting CF8). The mechanism stacks four per-matrix SVD truncations; compounding works only if ΔNLL effects are approximately additive (they may not be — orthogonality of the projections is the best case; interference is the risk). Not CF8 MLP rank reduction. Grounded in CF11 (W_Q + W_K measurements in hand); W_V / W_O remain the open gates.

### Residency arithmetic
Qwen3-1.7B-Base bf16: total attention weight per layer = 4 × d² = 4 × 2048² × 2 bytes = 32 MB / layer × 28 layers = 896 MB attention weights total out of ~3.4 GB model. At K=128 (W_Q) + K=256 (W_K, W_V, W_O): W_Q saves (2048−128)/2048 = 93.75%; W_K/V/O each save (2048−256)/2048 = 87.5%. Blended per-layer savings: (0.9375 + 3×0.875) / 4 = 0.891 of attention weight bytes. Attention weight reduction: 896 MB × 0.891 ≈ 798 MB. Remaining attention weights: ~98 MB. Model delta: −798 MB from 3.4 GB → ~2.6 GB. For 70B target: attention weights ≈ 4 × 8192² × 2 × 80 layers ≈ 43 GB; same compression ratio would recover ~38 GB — but 70B total is ~130 GB bf16, so attention is 33%. Practical ceiling: 33% × 0.891 = 29.4% total model compression from attention alone. Not sufficient for DRAM-resident 70B; but stacked with RAOK (W_up activation-tier quantization, currently ~2.2 bpw path) it contributes to the cascade. For Qwen3-1.7B: combined model from ~3.4 GB to ~2.6 GB while keeping MLP at bf16 — quality is the constraint, not residency. For 70B: any confirmed 30% reduction from attention + RAOK quantization on MLP together can push 70B bf16 (130 GB) toward 70 GB, still not DRAM-resident but meaningful for NVMe-tier throughput.

### Novelty gloss vs kill list and published landscape
Kill list closest: AQFKV (per-run-003 track A selection) tested W_Q and W_K; W_V/W_O explicitly deferred. This proposal's novelty is the **joint four-matrix simultaneous truncation** plus the residency-arithmetic compounding across all four projections — not tested by AQFKV. Closest published: MLA (DeepSeek-V2) jointly compresses Q and K in training; this is post-training SVD truncation without retraining, structurally distinct from MLA which is a training-time reparameterization. A3 (arXiv:2505.12942) uses attention-logit-error minimization — orthogonal loss function, attention-logit vs weight-Frobenius. The "W_V and W_O are also concentrated" premise is untested on Qwen3, so this is a falsifiable claim not yet in the literature.

### Smallest experiment
**Testable claim**: W_V r_99/d < 0.80 AND W_O r_99/d < 0.80 on Qwen3-1.7B-Base bf16 (same structural class as W_Q at 0.63 and W_K at 0.79, not MLP at ~1.0). Runtime: SVD spectrum of W_V[L14] and W_O[L14] (2048×2048 matrices): ~5 min each. Full cascade rung 1: spectrum plots for W_V and W_O across all 28 layers, ~30 min. Go: r_99/d < 0.80 for both. No-go finding: if W_V or W_O are full-rank (r_99/d > 0.90), the joint compression opportunity collapses and CF8 extends one step further into attention weights.

### Primary risk
W_V and W_O may be full-rank despite W_Q and W_K being concentrated — W_V holds value content, W_O projects into residual stream, and their spectra may behave more like MLP (routing) than W_Q (head-subspace spanning). Mitigation: run W_V/W_O SVD spectrum first before committing to joint truncation experiment.

---

## R4-2 — Layer-1 SDZC Free Win Stacked with Global Attention Compression (L1AW)
**R / Track B**

### Mechanism
Track B. CF6 confirmed 36% of gate neurons in layer 1 are foldable (std < 0.05) while all other layers are < 2.5%. This is a confirmed, free, zero-quality-cost saving on layer 1 that the pipeline has repeatedly acknowledged as real but isolated. The Reach framing: treat the layer-1 SDZC 36% fold as a confirmed rung-zero of a cascade that adds confirmed attention-weight compression (CF11 W_Q K=128 + W_K K=512 globally) and then asks: given that layer 1 has the most architectural softness (36% foldable gates, early-layer role), does the layer-1 W_Q / W_K spectrum show MORE concentration than later layers (motivating an even deeper per-layer compression budget for layer 1)? The compound claim is: layer 1 is architecturally dual-soft — both MLP-side (SDZC) and attention-side — and a layer-1-specific compression schedule materially differs from a flat global schedule. This requires one additional measurement (per-layer W_Q spectrum, not just L14 as AQFKV measured) but it's cheap and the payoff is a calibrated per-layer budget.

### Residency arithmetic
Layer-1 SDZC alone: layer 1 MLP weight rows foldable = 36% × 2×6144×2048 bytes ≈ 36% × 50 MB ≈ 18 MB. Tiny in isolation. The residency claim is NOT from this rung alone. Compounded with CF11 attention compression across all 28 layers (see R4-1 arithmetic): the per-layer schedule argument is that layer 1 can absorb K_Q=64 (CF11 per-head NO-GO at K=64, but per-head is different from global — need to verify global K=64 for layer 1 specifically) instead of K=128 globally. If layer 1 W_Q admits K=64 with ΔNLL <0.1 nats per layer, the per-layer approach recovers marginal extra bytes. The real payoff is the structural-finding cascade: empirically grounding a per-layer compression budget is a finding nobody has on Qwen3, and it extends the AQFKV result from one global schedule to a full per-layer curve.

### Novelty gloss vs kill list and published landscape
Kill list closest: AQFKV measured L14 specifically (a representative middle-to-deep layer); per-layer variation was not measured. This proposal's novelty is the **cross-finding coupling of CF6 (layer-1 MLP softness) with CF11 (attention spectrum concentration)** to motivate a per-layer compression schedule — the idea that the same architectural early-layer specialization shows up in BOTH the MLP gate and the attention projection simultaneously. Closest published: mixed-precision quantization (e.g., LLM.int8() sensitivity profiling, SmoothQuant layer-sensitivity) assigns per-layer quantization budgets, but these are sensitivity-based (second-order Hessian), not spectrum-structure-based. The spectrum-structure-based per-layer budget is a structurally distinct argument.

### Smallest experiment
**Testable claim**: W_Q spectrum at layer 1 shows r_99/d materially lower than the L14 measurement (0.63) — specifically, r_99/d(layer 1) < 0.50, motivating a deeper K budget for early layers. Runtime: SVD of W_Q[layer 1] on Qwen3-1.7B, 2048×2048 matrix: ~5 min. Go: r_99/d < 0.50. No-go: layer 1 W_Q is NOT more concentrated than L14 (no per-layer budget advantage), but the finding tightens the CF11 boundary from "global K=128" to "depth-invariant at K=128 even in the softtest layer."

### Primary risk
The CF6 layer-1 anomaly (MLP softness) may not correlate with attention-side softness — different mechanisms (early-layer embedding transformation vs. head-pooling geometry) could easily decouple. Mitigation: the W_Q[layer 1] SVD is a 5-min check; run it before investing in the per-layer schedule.

---

## R4-3 — RAOK + Attention-Tier Joint: Two-Matrix Outlier + Spectral Cascade (RATJ)
**R / Track B**

### Mechanism
Track B. RAOK (R2-Aware Outlier-Keyed Activation Codebook) is the strongest surviving Track B path per the kill list: 3-tier activation quantization of W_up GEMV output grounded in CF3 (K-dependent Jaccard: top-0.1% channels Jaccard=0.718 static, 1% Jaccard=0.31 dynamic). This proposal stacks RAOK (operating on activation-side, W_up output channels) with CF11 W_Q+W_K attention-weight compression (operating on weight-side, W_Q and W_K SVD truncation). The two mechanisms operate on orthogonal objects (activation tiers vs. weight spectra) and should compose with minimal destructive interaction. The joint cascade claim: RAOK on W_up outputs (targeting 1.5–2.5× effective bandwidth at DRAM tier) plus W_Q K=128 + W_K K=512 truncation across all 28 layers (see R4-1) produces a combined residency reduction that is multiplicative to first order. Destructive interaction check: RAOK changes the activation representation entering attention; W_Q/W_K truncation changes the weight representation in attention. The interface is the residual stream — if RAOK's outlier pinning and W_Q's low-rank projection jointly distort the Q computation, the interaction degrades quality faster than additive ΔNLL. This must be checked empirically; the proposal is a cascade plan, not a proof.

### Residency arithmetic
RAOK on W_up GEMV (activation tier, not weight residency per se): the quantization of W_up columns operates at DRAM tier. For 70B model on NVMe: W_up alone is ~35 GB of the 130 GB bf16 model (80 layers × 8192 × 28672 × 2 bytes ≈ 37 GB). At 2 bpw (RAOK 3-tier target): 37 GB → 4.6 GB. CF11 attention-weight compression: 43 GB → ~5 GB (from R4-1 arithmetic). Combined: W_up + attention ≈ 80 GB → ~10 GB; remaining W_gate (~37 GB) + W_down (~37 GB) + embeddings (~2 GB) = 76 GB at bf16. Total: ~86 GB — not DRAM-resident but substantial reduction from 130 GB. W_gate and W_down compression remain the blocking wall (both full-rank, CF8). At 2.2 bpw on ALL MLP weights (if RAOK's findings extend to W_down): MLP ~110 GB → ~13 GB + 10 GB (attention) + 2 GB (embed) = ~25 GB. Still not DRAM-resident for 70B on 7.28 GiB. For Qwen3-1.7B: 3.4 GB → with RAOK MLP at 2.2 bpw + CF11 attention → ~0.9 GB. Comfortably DRAM-resident.

### Novelty gloss vs kill list and published landscape
Kill list: RAOK is deferred (not killed), and CF11 attention compression is untested at joint level. The composition of activation-tier outlier handling (RAOK grounded in CF3) with spectral attention compression (CF11) is a novel cross-finding stack. Closest published: SmoothQuant + W8A8 stacks quantize both weights and activations, but do not compose activation-tier outlier pinning with attention-weight SVD truncation as joint mechanisms. The novelty is specifically the CF3 × CF11 grounding — two Qwen3-specific empirical findings composing into one deployment cascade.

### Smallest experiment
**Testable claim**: W_Q K=128 ΔNLL (+0.98 nats per AQFKV) is not significantly worsened by RAOK's 3-tier activation codebook on the same model (interaction ΔNLL < 1.5 nats, vs. expected sum ~1.5 nats if independent). Runtime: run RAOK-only (3-tier activation on W_up output, K=1% = 20 dynamic channels) and W_Q-K128-only perplexity measurements independently, then compose and measure joint ΔNLL. Each: ~20 min on Qwen3-1.7B. Go: joint ΔNLL ≤ RAOK-only + W_Q-only (i.e., composition is sub-additive, no destructive interaction). No-go: joint ΔNLL > sum (destructive interaction between activation distortion and low-rank Q projection). Structural finding either way: the interaction coefficient is the first empirical measurement of RAOK × CF11 composition on Qwen3.

### Primary risk
Destructive interaction: RAOK's activation approximation changes the distribution entering W_Q·x, potentially invalidating the CF11 low-rank approximation whose singular vectors were computed on the original distribution. Mitigation: run the 40-min interaction check before any deeper investment; if destructive, the two mechanisms must be applied independently in separate ablation sweeps.

---

## R4-4 — Untied lm_head Spectrum + Cascade to SVLD Revival (ULHC)
**R / Track B**

### Mechanism
Track B. CF12 confirmed tied lm_head is full-rank (r_99=1992/2048, catastrophic ΔNLL at r=1024: +19.96 nats). The kill list explicitly marks this as a kill for TIED configurations and notes "untied lm_head spectrum (Qwen3-8B has tie_word_embeddings=False) remains open." This proposal turns that open question into a Reach cascade. Qwen3-8B has an untied lm_head of shape 151936×4096. The Godey & Artzi gradient-bottleneck argument predicts the untied lm_head WILL have a concentrated spectrum (single gradient path, not dual), making SVD truncation viable. The Reach claim: untied lm_head at Qwen3-8B at r=256 has ΔNLL < 0.3 nats and r_99/d < 0.30, enabling a memory-mapped streaming SVD decomposition that keeps only the top-256 left singular vectors in DRAM while streaming right singular vectors from NVMe. The deployment consequence: lm_head is 151936×4096×2 bytes ≈ 1.2 GB in bf16 for 8B; at r=256 → (151936×256 + 4096×256) × 2 bytes ≈ 79 MB. 15× reduction of lm_head footprint. For 70B (151936×8192): 2.4 GB → 158 MB. At the Qwen3-72B tier: lm_head is ~2 GB; dropping it to ~150 MB is a meaningful NVMe-tier contribution.

### Residency arithmetic
Qwen3-8B bf16: total model ≈ 15.2 GB. lm_head = 151936×4096×2 = 1.24 GB. At r=256: lm_head drops to 79 MB (15.6× reduction). Model: 15.2 GB → 14.0 GB — marginal for 8B. For Qwen3-72B: model ≈ 130 GB, lm_head ≈ 2.4 GB, at r=256 → 150 MB, model → 128 GB. Not DRAM-resident but contributes to NVMe-tier throughput at 3 GB/s: lm_head token step drops from 2.4 GB / 3 GB/s = 0.8s/tok to 150 MB / 3 GB/s = 0.05s/tok (ignoring DRAM residency). Combined with RAOK + CF11 attention compression in R4-3, the cascade targets: MLP at 2.2 bpw (~13 GB) + attention at CF11 compression (~5 GB) + lm_head at r=256 (~150 MB) + embed (~2 GB) = ~20 GB for 70B. Still above 7.28 GiB DRAM target, but NVMe-resident 70B at ~20 GB with selective DRAM pinning of hot layers is tractable with the prefetch strategies.

### Novelty gloss vs kill list and published landscape
Kill list: LHQD / SVLD killed for tied configuration. This is the UNTIED sequel — explicitly the open case from the kill list. Closest published: Godey & Artzi (arXiv:2510.24966) analyzes lm_head spectral structure; their theorem applies to untied configurations. This proposal operationalizes their prediction as a Qwen3-8B measurement. Novelty: the streaming-SVD deployment (streaming right singular vectors from NVMe, keeping left singular vectors in DRAM) is a systems-level realization of the spectrum finding that Godey & Artzi do not address. No existing work combines the theoretical prediction with NVMe-streamed SVD inference.

### Smallest experiment
**Testable claim**: Qwen3-8B untied lm_head has r_99/d < 0.40 AND ΔNLL at r=256 < 0.30 nats. Runtime: blockwise SVD of 151936×4096 matrix in bf16 on Ryzen 5 7530U: ~45 min (LAPACK dgesdd on 4096×4096 blocks, 37 blocks of 4096×4096). NLL sweep at r ∈ {64, 128, 256, 512, 1024}: ~15 min on WikiText-2 subset. Total: ~60 min. Go: r_99/d < 0.40 AND ΔNLL@r=256 < 0.30 nats. No-go: untied lm_head is also full-rank, closing the CF12 boundary to ALL lm_head configurations and killing SVLD permanently.

### Primary risk
Qwen3-8B may exceed available RAM for blockwise SVD (model is 15.2 GB bf16 and the system has 7.28 GiB usable). Mitigation: load lm_head.weight alone (151936×4096 × 2 bytes = 1.24 GB; fits in RAM separately from the rest of the model) and run SVD on it in isolation before loading the full model.

---

## R4-5 — W_down Rank Sweep: Closing CF8 and Opening Quantization Ceiling (WDRS)
**R / Track B**

### Mechanism
Track B. CF8 kills no-retraining low-rank compression of MLP weights. W_gate (R3) and W_up (R4) are both confirmed full-rank. W_down (the output projection, 6144→2048 in Qwen3-1.7B) is UNTESTED. The kill list notes "applies to W_down too (untested but predictable)." This proposal makes that prediction testable as the cheapest possible structural-finding experiment. The Reach framing: IF W_down is also full-rank (as predicted), CF8 is now a triple-confirmed compound finding covering all three MLP projection matrices. This formally closes the "MLP low-rank" search branch and shifts the pipeline's compression mandate permanently to quantization (RAOK-style) and I/O-tier optimization. IF W_down is NOT full-rank (surprise), it opens the only remaining MLP compression lever. Either outcome is structurally load-bearing for every downstream ideator. Additionally, W_down's activation-input is the SwiGLU output — the post-nonlinearity distribution whose outlier structure CF3 characterized. A rank sweep of W_down under the ACTUAL SwiGLU-output activation distribution (rather than the uniform distribution implied by vanilla SVD) tests whether the activation-weighted effective rank of W_down differs from its weight-Frobenius rank — a more favorable case for compression, tested by ACTIVQ (weighted SVD using CF3's 3-tier structure as weight).

### Residency arithmetic
W_down in Qwen3-1.7B: 28 layers × 6144×2048 × 2 bytes ≈ 686 MB. At K=1024 (50% rank): IF ΔNLL comparable to W_gate's +0.77 nats (best case), 686 MB → 343 MB savings. If W_down is activation-weighted and the effective rank is lower (ACTIVQ hypothesis), K=512 might achieve ΔNLL < 0.5 nats, giving 514 MB savings. Neither is a large model reduction on its own for 70B, but confirming or closing W_down rank is a structural prerequisite for any future cascade that accounts for all three MLP matrices.

### Novelty gloss vs kill list and published landscape
Kill list: W_gate (R3 Idea 12), W_up (R4-B CLUBS), both killed. W_down explicitly flagged as untested. This proposal is the logical completion of the MLP rank-characterization program. Closest published: AQLM, GPTQ, SpQR all characterize MLP weights for quantization, not for pure rank structure. The activation-weighted SVD variant (ACTIVQ) using CF3's outlier tiers as the weighting kernel is novel — no published method conditions the SVD weight matrix on the measured per-tier activation structure from the same ladder.

### Smallest experiment
**Testable claim**: W_down at K=1024 (50% rank) produces ΔNLL > 1.0 nats (i.e., W_down is full-rank, consistent with CF8 prediction). Runtime: all-layer W_down rank sweep at K ∈ {2048, 1024, 512, 256, 128} on Qwen3-1.7B, identical protocol to R3 W_gate sweep: ~30 min. Go (surprise): ΔNLL@K=1024 < 0.5 nats → W_down IS compressible, opens new compression path. No-go (expected): ΔNLL@K=1024 > 1.0 nats → W_down confirms CF8, MLP rank program is closed. Either outcome maps the CF8 boundary completely. Secondary: activation-weighted SVD variant (ACTIVQ) adds ~15 min.

### Primary risk
W_down is predictably full-rank, making this a "confirming the null" experiment. The pipeline value is structural-finding closure, not a compression win. Mitigation: pair with the ACTIVQ activation-weighted variant so that even on "confirming null," a novel measurement (activation-conditioned effective rank vs. Frobenius rank) is produced.

---

## R4-6 — Head-Redundancy Routing: MLA-Style Q-K Shared Basis Post-Training (HRQK) [FREE SWING]
**R / Track A**

### Mechanism
Track A. CF11 confirmed that 16 Q-heads in Qwen3-1.7B collectively span only a ~128-dim subspace (global K=128 at ΔNLL=+0.98 nats). The structural finding 11 interpretation: "MLA-style joint Q-K low-rank projection in head space is empirically motivated for Qwen3 post-training." This proposal is the first Track A cascade attempt to operationalize that motivation. The mechanism: compute a joint basis B ∈ R^{2048×128} from the 28 per-layer W_Q stacks via PCA (NOT cross-layer stacking which killed v2-CHEAP-TEST-001 — this is WITHIN-layer head PCA, not cross-layer). Each layer's 16 query heads W_Q[L] ∈ R^{16×128×2048} are treated as 16 samples in R^{2048×128}; PCA over heads reveals the shared head subspace within that layer. Then W_Q[L]·x = B[L] · (B[L]^T · x) · coefficients[L], where B[L] ∈ R^{2048×K} is the per-layer head basis (K=64–128). This is a Track A change: the computation graph changes from 16 separate head-Q projections to one shared low-rank Q projection per layer. This is what MLA does at training time; this proposes doing it post-training from the measured head redundancy. The kill — v2-CHEAP-TEST-001 — killed CROSS-LAYER W_Q subspace sharing; this is WITHIN-layer head-pooling subspace, which is what CF11 actually measures.

### Residency arithmetic
W_Q for Qwen3-1.7B: 28 layers × 2048×2048 × 2 bytes = 235 MB. At K=128 per layer: W_Q residency drops from 2048×2048×2 to (2048×128 + 128×2048) × 2 = 1 MB/layer → 28 MB total. Savings: 207 MB from W_Q alone. Additionally, if the head-shared basis B[L] is the bottleneck and q_dim effectively compresses from 2048 to 128, the Q side of scaled dot-product attention shrinks, reducing the intermediate Q*K^T computation. Attention FLOP saving: K_heads × seq × d_head²  → K_basis × seq × K_basis (a secondary effect, not load-bearing). For 70B: W_Q per layer = 8192×8192×2 = 128 MB × 80 layers = 10.24 GB. At K=128: (8192×128 + 128×8192) × 2 × 80 = 0.32 GB. W_Q savings: ~10 GB. Meaningful fraction of the 130 GB 70B model.

### Novelty gloss vs kill list and published landscape
Kill list: v2-CHEAP-TEST-001 killed cross-layer W_Q shared basis — W_Q subspaces rotate independently across layers. THIS is within-layer head pooling, which is exactly what CF11 says holds (16 heads pooling into ~128 dims). MLA (DeepSeek-V2) does this at training time with a different loss function. A3 (arXiv:2505.12942) uses attention-logit-error minimization for post-training KV compression. The novelty here is post-training within-layer head-pooling SVD of W_Q, grounded in the AQFKV measurement that 16 heads span ~128 dims, without any retraining. No published work applies MLA-style factorization post-training using spectrum measurements as the grounding.

### Smallest experiment
**Testable claim**: PCA over 16 query heads of W_Q[L14] (treating each head's 128×2048 as a sample) yields a basis capturing ≥90% of head variance at K_basis=64. Runtime: reshape W_Q[L14] to 16×(128×2048), compute SVD, measure variance at K_basis ∈ {16, 32, 64, 128}: ~5 min. Go: K_basis=64 captures ≥90% head variance AND reconstruction error on W_Q[L14]·x gives ΔNLL < 1.5 nats on 200-token eval. No-go: per-head PCA concentrated (K_basis=128 needed for 90%) — finding 11 may require all 128 per-head dims to be preserved, and the MLA within-layer factorization is not further compressing beyond vanilla per-head K=128 (which was NO-GO per CF11).

### Primary risk
The 16 heads may have rotated but equally-important subspaces (diverse, not redundant), making the PCA low-rank only in a shared-coordinate sense that doesn't reduce per-head dimensionality after reconstruction. CF11's "global K=128" could mean "128 globally, but each head needs its own 8-dim allocation" — which is exactly the per-head NO-GO finding at K=64. Mitigation: the 5-min PCA at L14 settles this directly; the head-PCA variance curve is the falsifier.

---

## R4-7 — W_V / W_O Spectrum Joint with KV-Cache Compression (VOKV)
**R / Track B**

### Mechanism
Track B. W_V and W_O are the last uncharacterized attention weight matrices (per the AQFKV result's "what's still open"). W_V ∈ R^{2048×2048} maps residual stream to value space; W_O ∈ R^{2048×2048} maps concatenated head outputs back to residual stream. Their spectral structure determines whether low-rank projection of the value computation is viable WITHOUT retraining. The Reach claim: IF W_V and W_O have r_99/d < 0.75 (consistent with W_K at 0.79), THEN a joint W_V + W_O compression at K=256 each is viable. Furthermore: compressed W_V is directly coupled to KV-cache compression — if W_V·x ≈ B_V · (B_V^T · x) for a K=256 basis, then the value vectors stored in KV-cache can be stored in the compressed 256-dim space (not 2048-dim), reducing KV-cache residency by 8×. At 4K context: KV-cache for Qwen3-1.7B = 2 × 2048 × 4096 × 28 × 2 bytes ≈ 944 MB. At K=256: 944 MB × (256/2048) = 118 MB. KV-cache ceases to be a residency bottleneck. At 128K context: KV-cache normally 28 GB → 3.5 GB. This coupling (W_V spectrum → KV-cache compression) is the load-bearing novel mechanism: it transforms a weight-compression measurement into a memory-bound deployment win.

### Residency arithmetic
KV-cache for Qwen3-1.7B at 4K context: 944 MB → 118 MB (8× reduction). KV-cache for Qwen3-72B (d=8192, 64 heads, 80 layers) at 128K context: 2 × 8192 × 128K × 80 × 2 bytes = 2 × 8192 × 131072 × 80 × 2 ≈ 275 GB bf16 → at K=256: 275 × (256/8192) ≈ 8.6 GB. This is the DRAM-resident target for KV cache at long context. For the 7.28 GiB Ryzen with Qwen3-1.7B at 4K: KV cache goes from 944 MB (>12% of DRAM) to 118 MB (1.6%), freeing DRAM for weight residency. Quality cost: IF W_V rank truncation at K=256 produces ΔNLL < 0.5 nats (like W_K at K=512: +0.29 nats), this is an acceptable trade.

### Novelty gloss vs kill list and published landscape
Kill list: KHQL (KV head quantization with layer sensitivity), ASCQ (attention-sink channel quantization), and CLASE (cross-layer KV aliasing) all deferred for KV compression — but none ground the compression in W_V spectral structure as an algebraic identity. KVQuant, MiniKV, KITTY (existing literature) quantize the KV cache values directly; none exploit the W_V spectrum to compress into a lower-dimensional representation algebraically. The algebraic identity "compress W_V SVD → store KV in W_V's basis → same output via W_O reconstruction" is grounded in CF11-class measurements and is the structural novelty.

### Smallest experiment
**Testable claim**: W_V[L14] r_99/d < 0.75 AND W_O[L14] r_99/d < 0.75 on Qwen3-1.7B-Base bf16. Runtime: SVD of W_V[L14] and W_O[L14]: ~10 min total. Go: proceed to W_V K=256 NLL sweep + KV-cache compression protocol: ~30 min additional. No-go: W_V and/or W_O full-rank → KV-algebraic compression not viable; CF8 now covers all MLP AND (partially or fully) all attention weights. Structural finding value: closes the uncharacterized attention-weight territory from CF11 AQFKV.

### Primary risk
W_O is the output-side projection that writes into the residual stream, making it functionally more similar to W_down (the MLP output projection, untested but expected full-rank per CF8) than to W_Q or W_K. If W_O is full-rank, the KV-algebraic compression still works through W_V alone (storing in W_V's basis and reconstructing with the full W_O), but the residency savings on W_O weights disappear. Mitigation: the KV-cache compression mechanism survives on W_V spectrum alone; W_O compression is a bonus.

---

## Convergence handles

1. **W_V / W_O spectral concentration** — R4-1, R4-3, R4-7 all depend on; first 10-min check gates three cascades simultaneously.
2. **CF11 attention-weight compression composability** — whether per-layer attention SVD stacks sub-additively with RAOK activation quantization; R4-3 tests this directly.
3. **Untied lm_head Godey-Artzi prediction** — R4-4 depends on; closes CF12 boundary to all configurations if full-rank confirmed on Qwen3-8B.
4. **W_down full-rank prediction** — R4-5 closes CF8 MLP triple-confirmation; downstream kill or opens activation-weighted effective-rank distinction.
5. **Within-layer head-pooling PCA geometry** — R4-6 (FREE SWING) depends on; tests whether CF11's global K=128 finding reflects per-head redundancy (MLA-style compressible) vs. diverse-head rotation (only globally compressible via joint projection).
6. **KV-cache algebraic compression via W_V basis** — R4-7; couples weight-spectrum measurement to long-context memory binding, only works if W_V concentrated.
