# Stage 1 — Orientation R (Reach) — Run 002

Orientation: R — Reach
Run: 002 (cold start, does not see run_001 outputs)
Ideator: Sonnet claude-sonnet-4-6

---

## R1-A — QKJB: Joint Q/K Low-Rank Folding with Cascade Attention Bandwidth Cascade

**Track: A (arch-transposition)**

### Mechanism

Track A. CF11 establishes W_Q r_99/d ≈ 0.63 and W_K r_99/d ≈ 0.79 in Qwen3-1.7B; global K=128 yields ΔNLL=+0.98 nats for W_Q alone. W_V and W_O are untested; the structural hypothesis is that attention weight matrices as a class have more concentrated spectra than MLP matrices (CF8 boundary). This cascade starts with the cheapest untested rung: measure W_V and W_O spectra simultaneously with a rank-sweep (r ∈ {64,128,256,512,1024}). If W_V and W_O follow the W_Q pattern (r_99/d ≈ 0.6–0.8), then a joint attention-weight factorization applies: for each layer, the four matrices W_Q, W_K, W_V, W_O can be expressed as low-rank operators on their respective subspaces. The cascade payoff: at K=128 for all four attention matrices in Qwen3-72B (d_model=8192, n_heads=64), attention weight residency drops from 4 × 64 × 8192 × 128 × 2 bytes bf16 ≈ 536 MB per layer × 80 layers = 42.9 GB → at r=512 = 4 × 8192 × 512 × 2 + 4 × 512 × 128 × 2 × 64 = 33.6 MB per layer × 80 = 2.69 GB. Combined with INT4 quantization on MLP weights, this is the plausible path to sub-7.28 GiB 70B-class residency. Not CF8 MLP rank reduction; relies entirely on CF11's attention-specific finding.

### Residency arithmetic

Qwen3-72B (Llama-3-70B architecture proxy): d_model=8192, n_layers=80, n_heads=64, intermediate≈28672.
- Baseline total at Q4_K_M (~4.5 bpw): ≈ 40.5 GB. Does not fit in 7.28 GiB.
- Attention weights per layer (4 matrices): 4 × 8192 × 8192 × 2 bytes = 536 MB × 80 = 41.9 GB total attention weights at bf16 — but at Q4 = 4.5 bpw ≈ 23.5 GB; attention is ~58% of total parameter bytes at matched quantization.
- If W_Q, W_K, W_V, W_O all tolerate r=512 (based on CF11 W_Q at r=512: ΔNLL=+0.20 per layer scaled to 80 layers ≈ total ΔNLL +16 nats — too expensive at per-layer independence). **The arithmetic here is the constraint.** Per-layer ΔNLL of +0.20 × 80 layers = +16 nats cumulative if independent. This is the primary risk (layers are NOT independent under residual stream — the per-layer finding underpins a tighter interdependency). The actual question is: is ΔNLL cumulative-additive or does the residual skip-connection absorb cross-layer rank deficiency?
- Residency case for a 2-stage cascade: Stage 1 — W_V/W_O rank sweep determines whether attention weight compression is viable at all (≤1-day experiment, ≤1 nat total ΔNLL target at combined rank). Stage 2 — scale finding to 72B parameter count. If W_V/W_O behave like W_Q, attention weights compress 8× (K=512 at 8192-dim → 512/8192 = 6.25% of bf16 weight bytes). Combined with 3-bpw MLP quantization: (23.5 GB attention @ 8× compression = 2.94 GB) + (16.6 GB MLP @ 3 bpw = 6.0 GB) ≈ 8.94 GB. Still over 7.28 GiB; need one more rung.
- Stage 3 — KV cache: at 4K context with INT8 KV, Qwen3-72B KV adds ~2.56 GB. Net with Stage 1+2: ~11.5 GB. Still over. Stage 3 must be aggressive: cross-layer Q basis sharing (do all 80 layers share a common W_Q subspace?). If yes, store ONE shared basis B_Q ∈ R^{8192×512} plus per-layer coefficients C_L ∈ R^{512×8192}: per-layer cost = 512×8192×2 bytes = 8 MB vs 8192×8192×2 bytes = 134 MB; 16.8× per-layer saving. That alone cuts 10.5 GB → 0.64 GB for all 80 W_Q matrices. This cross-layer Q basis is the audacious rung; CF11 motivates it (16 heads collapse to ~1 subspace within a layer; the question is whether that subspace is also stable across layers).
- P(end-to-end success): P(W_V/W_O compress like W_Q) × P(per-layer ΔNLL compounds sublinearly) × P(cross-layer Q basis is stable) ≈ 0.6 × 0.5 × 0.35 ≈ 0.11. Low but each rung yields a structural finding.

### Novelty gloss

Closest kill-list item: CLUBS/RSTD (cross-layer Tucker on W_up) — KILLED because MLP matrices are full-rank. This proposal targets attention matrices, which CF11 shows are structurally different (r_99/d ≈ 0.63 vs 1.0 for MLP). The key novelty is the cross-layer W_Q shared basis claim — this is not in any published method. MLA (DeepSeek-V2) is training-time joint Q-K low-rank projection; A3 (arXiv:2505.12942) uses per-layer attention-logit-error minimization at training time. Neither tests cross-layer weight sharing without retraining, and neither relies on a measured basis-concentration finding on a trained model. Closest publish: model stitching — but that is cross-scale, not cross-layer on a single model.

### Smallest experiment

Claim: W_V and W_O at global rank r=512 (K/d=0.0625) in Qwen3-1.7B have ΔNLL ≤ +0.50 nats each (by analogy with W_Q at r=512: +0.20 nats; W_K at r=512: +0.29 nats).
Runtime: ~20 minutes (parallel to AQFKV follow-up; same script with W_V/W_O substituted).
Go threshold: ΔNLL ≤ +0.50 nats for BOTH W_V and W_O at r=512 simultaneously applied.
No-Go value: extends the CF11 attention-boundary map to the full set of attention weight matrices; informs whether attention-side compression is a 2-matrix or 4-matrix win.

### Primary risk

W_V and W_O may not follow the W_Q concentrated-spectrum pattern — W_O in particular is downstream of head concatenation and may have flat singular value decay, killing the 4-matrix joint case. Mitigation: the first rung (W_V/W_O spectrum) runs in 20 minutes and determines go/no-go before any cascade investment. If W_O is flat, fallback is W_Q + W_K only, which still gives 2× attention compression.

---

## R2-B — RAOK-SCALE: Activation-Outlier Tiered-Quantization Cascade to 70B Deployment

**Track: B (compression)**

### Mechanism

Track B. CF3 (K-dependent outlier dynamicity: Jaccard 0.718 at K=0.1%, 0.31 at K=1%) directly motivates a three-tier per-activation scheme: the top ~2 channels (K=0.1%) are channel-static (pin them in FP16 statically, no per-token cost), the next ~18 channels (K=1% minus K=0.1%) are token-dynamic (store as INT8 per-token index + value), and the remaining 2026 channels are dense INT4. This is the RAOK mechanism deferred in Track B R6 Stage 4. The Reach framing scales it: RAOK on a 70B-class model determines whether the CF3 K-dependent finding (measured on Qwen3-1.7B) generalizes to larger Qwen3/Llama-3 models, and if it does, what the combined tok/s gain is at 7.28 GiB. The cascade: Rung 1 = RAOK on Qwen3-1.7B (measure ΔNLL vs INT4 baseline; confirm static/dynamic partition works). Rung 2 = measure whether K-dependent Jaccard pattern holds at Qwen3-4B/7B (does the 0.718 top-tier static fraction scale or degrade?). Rung 3 = project residency arithmetic to 72B. The load-bearing novelty is the empirical K-dependent partition from CF3 — no published activation quantization method distinguishes K=0.1% (static) from K=1% (dynamic) in a cost-model-aware way. LLM.int8() uses a static channel partition but at a much larger fraction; SmoothQuant uses channel-static scaling without per-token dynamic fallback; PrefixQuant separates token-wise (BOS/delimiter) from channel-wise but does not use per-token Jaccard as the partition signal.

### Residency arithmetic

Qwen3-1.7B baseline: 1.7B params at INT4 ≈ 850 MB. RAOK adds per-token overhead: for 2048 channels, 2 FP16 static channels (no overhead, pinned once), 18 INT8 dynamic channels per token (18 × 2 bytes = 36 bytes/token/layer), 28 layers = 1008 bytes/token overhead. At 4K context: 4096 × 1008 = 4.1 MB — negligible vs 850 MB model. RAOK overhead is <0.5% of model residency. The payoff is quality: if ΔNLL vs INT4 is ≤+0.05 nats (vs INT4's typical +0.15 vs bf16), RAOK delivers better-than-INT4 quality at INT4-equivalent residency.

For 70B-class scaling: Qwen3-72B at INT4 ≈ 40.5 GB — does not fit. RAOK alone does not change weight residency; it changes activation quantization. The Reach angle is: combine RAOK (activation quality preservation) with the most aggressive weight quantization that stays within 7.28 GiB. The question RAOK answers is whether INT2 weight quantization becomes feasible when activations are handled properly. SOBIB (INT2 + sparse INT8 outlier correction) is deferred but adjacent. If RAOK demonstrates that per-token K=0.1% static + K=1% dynamic partitioning reduces effective activation outlier mass, it may unlock INT2-equivalent weight quantization where INT4 is currently required for quality. Combined: 72B at INT2 ≈ 18 GB — still over, but INT2 + RAOK + attention compression (R1-A) could reach 7.28 GiB.

DRAM bandwidth at 11.5 GB/s: 7.28 GiB / 11.5 GB/s ≈ 0.63 s/token = 1.6 tok/s ceiling for DRAM-resident 70B. Above 3 tok/s requires sub-3.5 GiB total residency — likely NVMe path or sub-0.5 bpw quantization that does not yet exist without severe quality loss.

P(end-to-end to DRAM-resident 70B): 0.7 (RAOK alone works on 1.7B) × 0.4 (scales to 72B architecture) × 0.3 (INT2 quality unlocked by RAOK) ≈ 0.08. But every rung has independent structural value.

### Novelty gloss

Closest kill-list: SOBIB (INT2 + sparse INT8 outliers) — deferred, not killed. Structural difference: RAOK uses the K-dependent Jaccard measurement (CF3) to set the tier boundaries; SOBIB uses calibration-only sensitivity. The partition point K=0.1% (static) vs K=1% (dynamic) comes from a measured structural property of the trained model, not a tuned hyperparameter. Closest published: LLM.int8() (static channel masking at a larger K), QuIP/AQLM (weight quantization, not activation), SmoothQuant (channel-static scaling only). The per-token dynamic tier keyed on the CF3 K-dependent Jaccard finding is the genuinely novel piece.

### Smallest experiment

Claim: RAOK three-tier (2 FP16 static / 18 INT8 dynamic / 2026 INT4) on Qwen3-1.7B achieves ΔNLL ≤ +0.10 nats vs INT4 baseline (INT4 activation quantization, not weight quantization — distinguishing activation quant from weight quant).
Runtime: ~2 hours (forward pass with instrumented activation quantization on the PDAP infrastructure already built in R2).
Go threshold: ΔNLL ≤ +0.10 nats vs INT4 activation quantization.
No-Go value: measures whether the CF3 K-dependent partition is noise or signal for quantization quality; informs whether the 0.718 / 0.31 Jaccard split represents recoverable vs irrecoverable quantization error.

### Primary risk

INT4 activation quantization may already perform well enough that the RAOK tiers add complexity without quality gain — i.e., the outlier handling in INT4 via group scaling already captures the channel-static phenomenon. Mitigation: measure INT4-activation vs FP16-activation ΔNLL first (1-hour baseline); if INT4-activation adds <0.05 nats vs FP16, RAOK is moot for quality (but may still accelerate decode via cheaper paths).

---

## R3-A — XLQB: Cross-Layer W_Q Basis Sharing — Dedicated Cascade Rung

**Track: A (arch-transposition)**

### Mechanism

Track A. CF11 establishes that W_Q in Qwen3-1.7B shows head-redundancy: 16 query heads collectively span a ~128-dim subspace within one layer. The Reach extension is the cross-layer question: do all 28 layers' W_Q matrices span the SAME ~128-dim subspace in R^{2048}? If yes, a single shared basis B ∈ R^{d_model × K} exists such that each layer's W_Q ≈ B C_L where C_L ∈ R^{K × d_qkv} is a small per-layer coefficient matrix. Storage: one B at 2048×128×2 bytes = 512 KB; 28 C_L at 128×2048×2 bytes = 14 MB total; vs 28 W_Q at 2048×2048×2 = 236 MB. Net saving: 236 MB → 14.5 MB = 16.3× on W_Q alone. Residency arithmetic on 70B: same ratio, 80 layers W_Q at 8192×8192×2 = 10.7 GB → 80 C_L at 512×8192×2 + 1 B at 8192×512×2 = 655 MB. That is a 16.3× saving on the query matrix alone, contributing ~10 GB to the overall 70B residency budget. The test: stack all 28 W_Q matrices into a tensor T ∈ R^{28 × 2048 × 2048}, compute PCA across the "layer" axis (i.e., reshape to 28 × 4M and find the top-K PCs), measure how much variance is captured at K=128. If ≥ 80% variance at K=128 across layers (the 80% threshold is conservative; even 60% might yield useful compression), the shared basis is real. This experiment runs in ~30 minutes on Qwen3-1.7B.

### Residency arithmetic

Qwen3-1.7B: 28 W_Q matrices at 2048×2048×bf16 = 236 MB total.
Shared basis at K=128: B ∈ R^{2048×128} = 512 KB; 28 C_L ∈ R^{128×2048} = 14.3 MB. Total: 14.8 MB.
Compression: 236 / 14.8 = 15.9× on W_Q.
DRAM bandwidth payoff: W_Q accounts for ~6.5% of Qwen3-1.7B parameter bytes (236 MB / ~3.6 GB). Saving 221 MB → ~0.02 s/token bandwidth saving at 11.5 GB/s. Small in absolute terms on 1.7B.

For 70B: 80 W_Q at 8192×8192×bf16 = 10.7 GB. At same 15.9× compression: 0.68 GB. Combined with W_K analogous compression (if W_K cross-layer basis also holds): W_K compression 10.7 GB → 0.68 GB. Total attention-Q/K saving: 21.4 GB → 1.35 GB. This is the big number. If this holds, the 70B model sheds ~20 GB just from Q and K cross-layer basis sharing — 40.5 GB → ~20.5 GB (still above 7.28 GiB; needs additional compression on MLP and V/O).

P(cross-layer basis is real at K=128): 0.35 — motivated by CF11 but untested. Each layer may span a different 128-dim subspace even if within each layer the heads are redundant.
P(full cascade to 7.28 GiB): XLQB + QKJB (R1-A) + 3-bpw MLP: (1.35 GB for Q/K) + (1.35 GB for V/O if same pattern) + 6.0 GB for MLP at 3 bpw = 8.7 GB — still slightly over. The math forces an additional rung.

### Novelty gloss

Closest kill-list: CLUBS/RSTD (cross-layer Tucker on W_up) — killed because W_up is full-rank. XLQB targets W_Q, which CF11 shows is structurally different. Cross-layer basis sharing for W_Q has no direct published analog. MLA (DeepSeek-V2) shares Q/K at training time via a single low-rank projection per model; it does not test cross-layer sharing. The "PCA across layers" formulation is a post-hoc structure measurement, not a training-time design choice. Closest published: parameter sharing in ALBERT (cross-layer weight sharing by construction) — but ALBERT ties ALL layers; XLQB proposes a measured soft-sharing via a shared basis with per-layer residuals, which is structurally different.

### Smallest experiment

Claim: PCA applied to the stacked tensor of 28 W_Q matrices in Qwen3-1.7B captures ≥ 50% of cross-layer variance at K=128 principal components (where K=128 was the within-layer effective rank from CF11).
Runtime: ~30 minutes (reshape, SVD, compute explained variance ratio).
Go threshold: ≥ 50% variance captured at K=128 across all 28 layers.
No-Go value: establishes whether the CF11 within-layer head-redundancy is also a cross-layer structural constraint or is layer-specific; either answer informs the architecture of any subsequent attention-side compression.

### Primary risk

The head-redundancy in W_Q (16 heads → ~128-dim space per layer) is a within-layer phenomenon; different layers may occupy orthogonal 128-dim subspaces in the full d=2048 space, so the cross-layer PCA captures nothing. Mitigation: cross-layer cosine similarity of top-128 singular vectors between layer pairs (i < j) — if any pair shares ≥ 0.5 cosine similarity, the experiment has signal even if full cross-layer PCA is weak.

---

## R4-B — TERNARY-NVMe: Ternary Weight + NVMe Paging Cascade for 70B Near-Fit

**Track: B (compression)**

### Mechanism

Track B. The KILL_LIST confirms PTQTP (arXiv:2509.16989) pre-empts bare dual-ternary decomposition. The Reach angle is different: exploit the structural finding that Qwen3 MLP weights are full-rank (CF8) as MOTIVATION for ternary quantization rather than as a blocker. Full-rank means the weights do not have a low-rank or structured-sparse representation; they must be represented densely. Given this constraint, the minimum-achievable-bpw for acceptable quality is the key unknown. PTQTP at 1.58 bpw for INT2-equivalent ternary is the benchmark. The cascade: Rung 1 — establish what bpw floor is achievable on Qwen3-1.7B MLP weights (W_gate, W_up, W_down) under ternary/INT2 quantization with calibration-based scaling, measuring ΔNLL; Rung 2 — profile NVMe streaming latency for a ternary-packed GGUF on this hardware (Windows 11, Ryzen 5 7530U, NVMe ~3 GB/s sequential); Rung 3 — if ternary achieves ≤+0.5 nats at 1.58 bpw on MLP weights AND attention compression (R1-A / R3-A) holds, what is the total model residency and projected tok/s at NVMe? Key number for 72B: 72B × 1.58 bpw = 14.2 GB — over 7.28 GiB for RAM but plausibly streamable from NVMe at 3 GB/s = 4.7 s/layer with 80 layers ≈ 376 s/token = infeasible. Correct number: streaming only MLP (50% of params = 7.1 GB) from NVMe while keeping attention (R1-A compressed to ~1.35 GB) in RAM. NVMe MLP streaming at 3 GB/s for 7.1 GB per forward pass = 2.37 s/forward = 0.42 tok/s. Below the 3 tok/s floor. So the cascade path to 3 tok/s on NVMe requires either: (a) sub-1-bpw MLP quantization (not known to exist without quality catastrophe), (b) extreme context-caching to amortize layer loads, or (c) aggressive token-batch decode to raise throughput. This arithmetic exercise is itself the structural finding of Rung 3 — it quantifies the NVMe bandwidth wall precisely. CF13 (NVMe prefetch speed) is not confirmed for v2; this cascade MUST re-derive the actual NVMe sequential read speed in Rung 2 before using it as a load-bearing number.

### Residency arithmetic

72B at 1.58 bpw (ternary MLP) + 1.35 GB compressed attention (from R1-A/R3-A): 72B × 1.58 bits / 8 bits = 14.2 GB MLP + attention 1.35 GB = 15.5 GB total. Does not fit in 7.28 GiB DRAM.

NVMe streaming path: keep attention + embeddings in DRAM (1.35 GB + 0.6 GB = ~2.0 GB), stream MLP layers from NVMe. NVMe read: 14.2 GB / 3.0 GB/s = 4.7 s/all-MLP-layers. Per token (one forward pass): 4.7 s. That is 0.21 tok/s — below the 3 tok/s floor by 14×.

The path to 3 tok/s on this hardware with a 70B+ class model: (3 tok/s × 7.28 GiB / 11.5 GB/s_DRAM) = model must fit in 7.28 GiB AND DRAM-bandwidth-bound. For NVMe: 3 tok/s × weight_bytes / 3 GB/s = model_size ≤ 1.0 GB to stream per forward. That is 8B parameters at 1 bit, which is not a 70B class model. **The honest arithmetic says: 70B at 3 tok/s on 7.28 GiB NVMe-only is algebraically ruled out by current NVMe bandwidth.** This cascade's value is precisely establishing this wall numerically and identifying what sub-problems remain open.

P(finding useful number): 0.9 (the arithmetic is almost certainly as above). P(structural finding that opens a door): 0.3 (e.g., per-layer caching with speculative early-exit, token-batch amortization, or a previously unmeasured AVX2 decode throughput for ternary).

### Novelty gloss

Closest kill-list: R2/S3 NMF Codebook (killed for different reasons — NMF runtime, not ternary). Track B R6 Stage 3 had THWR (tabulation hash) and CSKQ (Count-Sketch) killed for incorrect imported theory (CF9); TERNARY-NVMe imports no such theorem. Closest published: PTQTP (ternary quantization, Sep 2025). Structural difference: this cascade uses ternary as a component of a full 70B deployment system analysis, not as a standalone quantization scheme. The novelty is the deployment arithmetic cascade, not the quantization method. If the arithmetic kills the idea, the arithmetic is the finding.

### Smallest experiment

Claim: NVMe sequential read speed on this hardware for large files (simulating GGUF weight streaming) achieves ≥ 2.5 GB/s sustained (the v2 re-derivation of CF13*).
Runtime: ~15 minutes (fio benchmark or PowerShell file-read timing on a 4 GB test file).
Go threshold: ≥ 2.5 GB/s sustained sequential read.
No-Go value: establishes actual NVMe floor for all future NVMe-paging ideas; hard constraint on cascade arithmetic.

### Primary risk

NVMe bandwidth is fundamentally too low for 70B at any sub-2-bpw on this hardware — the arithmetic above strongly suggests this is the case. The cascade collapses at Rung 3. Mitigation: pivot to token-batch decode (B > 1 token per forward pass amortizes weight loading); for batch B=8, effective tok/s ≈ 8 × 0.21 = 1.7 tok/s. Still below 3 tok/s but closer; sets the real target at B=14 for 3 tok/s NVMe streaming.

---

## R5-A — WDLA-RESCUE: Calibration-Conditioned Cross-Scale Affine Surgery (Properly Scoped)

**Track: A (arch-transposition)**

### Mechanism

Track A. WDLA (Track A R2) achieved R²=1.0 on calibration and R²=-118000 on held-out due to ill-conditioning (CF10). The failure is NOT the frame's fault — it is a calibration data budget failure. The structural question survives: can a post-hoc affine map A_L: h_small^{(L)} → h_large^{(L)} be fit such that a Qwen3-0.6B forward pass, corrected layer-by-layer, approximates a Qwen3-1.7B forward pass without running 1.7B? If yes, this is the cheapest possible 70B emulation: run a 7B-class model and apply cross-scale affine corrections to approximate 70B quality. The Reach cascade: Rung 1 = establish the calibration scale required. CF10: n_params_to_fit / n_independent_samples << 1 required. A_L with forced rank r=64 has (1024 × 64 + 64 × 2048) = 196,608 parameters. At 1K tokens × 2048 output dims = 2.05M independent values, ratio = 196K / 2.05M = 0.096 — well-conditioned at rank 64. Rung 1 is a re-run of WDLA with rank-64 forced (not full-rank), 10K calibration tokens, ridge=1e-1. If R²_eval > 0.85 on held-out (Stage 6 amendment criterion), advance to Rung 2: measure PPL of Qwen3-0.6B + rank-64 affine corrections vs Qwen3-1.7B. Rung 3: project to Qwen3-7B (small draft) → Qwen3-72B (target) scale pair. If the quality gap between 7B and 72B is recoverable at r=256, run a 7B model + corrections = 7.28 GiB RAM case.

### Residency arithmetic

Qwen3-7B as draft: ~7B × 3 bpw = 2.63 GB DRAM resident. Cross-scale affine corrections: 28 layers × rank-256 affine = 28 × (7168 × 256 + 256 × 8192) × 2 bytes = 28 × (3.67 MB + 4.19 MB) = 220 MB. Total: 2.63 + 0.22 = 2.85 GB. Comfortably within 7.28 GiB. Quality target: within 0.3 nats of Qwen3-72B. This is the audacious claim — cross-scale affine surgery recovers 7B → 72B quality gap. No empirical basis confirms this yet; the cascade must establish it.

At the Qwen3-0.6B → Qwen3-1.7B pair (same 28 layers, same depth), the quality gap is ~4 nats (0.6B PPL vs 1.7B PPL). If rank-64 affine corrections close even 50% of that gap (2 nats), the mechanism is load-bearing. Scaling to 7B → 72B: the quality gap is smaller in relative terms (7B is already good), so the correction budget required may be smaller.

P(rank-64 WDLA on 0.6B→1.7B achieves R²_eval > 0.85): 0.6 (proper conditioning should fix the WDLA failure).
P(quality gap closes ≥ 50%): 0.35 (highly speculative — no prior work shows this).
P(scales to 7B → 72B): 0.25 (depth/architecture mismatch may not be linearly affine-correctable at longer scale).
P(end-to-end): 0.6 × 0.35 × 0.25 ≈ 0.05. Low; high structural value per rung.

### Novelty gloss

Closest kill-list: WDLA (Track A R2 NO-GO due to calibration ill-conditioning). This is a directly targeted rescue attempt with the CF10 fix applied: forced rank-64 by construction. Not re-proposing; re-running with the identified fix. Stage 6 WDLA amendment explicitly said "The frame is still potentially viable with larger calibration corpus AND lower-rank A_L by construction." Closest published: Model Stitching (NeurIPS 2025, arXiv:2506.06609) — training-time only, not inference-time. Frame novelty of inference-time cross-scale affine surgery remains unmatched in literature per Stage 6's search.

### Smallest experiment

Claim: Forced rank-64 affine map A_L with 10K calibration tokens and ridge=0.1 achieves R²_eval > 0.85 on Qwen3-0.6B → Qwen3-1.7B layer-pair (layers 10-28) on held-out tokens.
Runtime: ~3 hours (reconfigure WDLA script for rank-64, gather 10K calibration tokens, fit, evaluate).
Go threshold: R²_eval > 0.85 (Stage 6 gate from WDLA amendment).
No-Go value: establishes whether the WDLA frame is structurally broken (not just data-starved). If rank-64 + 10K tokens still gives R²_eval < 0, the affine surgery hypothesis is wrong, not just under-resourced.

### Primary risk

The 0.6B → 1.7B hidden state dimensionality mismatch (1024 → 2048) may be the deep blocker: no affine map can make a 1024-dim representation functionally equivalent to a 2048-dim one if the additional capacity is load-bearing. Mitigation: test on layers 0-5 (early layers have simpler representations; residual additivity means shallow layers carry less information per step). If early layers achieve high R²_eval, the frame lives for shallow correction; deep layers may require a different mechanism.

---

## R6-B — ATTN-SPECTRUM-SCALE: W_V / W_O Rank Cascade to Settle Full Attention-Weight Map

**Track: B (compression)**

### Mechanism

Track B. The full attention-weight compression case requires W_V and W_O spectra in addition to CF11's W_Q and W_K measurements. This is a two-measurement cascade that costs ~40 minutes total and closes the attention-weight-class characterization map definitively. Mechanism: apply the same rank-sweep script as AQFKV to W_V (d_model × d_v, d_v = d_model/n_heads × n_heads = d_model) and W_O (d_model × d_model). W_O is the output projection after head concatenation; its structure is different from W_Q because it maps from concatenated-head space back to residual stream. Hypothesis: W_V behaves similarly to W_K (r_99/d ≈ 0.7–0.8, K=512 gives ΔNLL ≈ +0.3–0.5 nats). W_O may be closer to full-rank because its output is the residual stream (same as MLP W_down, which CF8 predicts is full-rank — untested). If W_O is full-rank, the attention-side compression is limited to {W_Q, W_K, W_V} and W_O must remain at full precision. The Reach value: at 70B scale, W_O compression (or lack thereof) determines whether the R1-A cascade closes or needs a different path.

### Residency arithmetic

Qwen3-1.7B: W_V ∈ R^{2048×2048}, W_O ∈ R^{2048×2048}. At bf16: 16 MB each per layer, 28 layers = 448 MB for W_V and 448 MB for W_O. If both compress at K=512 (same as W_Q/W_K): 448 MB × (512/2048 × 2) ≈ 224 MB for U factor + 224 MB for V factor = same total at naive SVD. The compression is real only if the truncated SVD at K=512 captures ≥ 90% variance: storage = 2048 × 512 × 2 + 512 × 2048 × 2 bytes per layer = 16 MB per layer (same as before in storage, BUT the inference compute is cheaper: matmul(W, x) at K=512 is (2048×512)×x + (512×2048)×result = 2 matmuls at half d vs one full). Wait — this is not a storage saving, it's a compute saving. To get storage saving, the V^T (right singular) must be absorbed into the next layer (AERO-style fold). For W_V specifically: V_L^T x followed by attention computation, then U_L (output) — absorbing V_L^T into W_Q or W_K reshape requires per-head alignment. The mechanism needs formalization. Storage saving requires: (U ∈ R^{d×K} stored, V^T ∈ R^{K×d} stored) = 2 × d × K × 2 bytes vs d × d × 2 bytes = 2K/d ratio. At K=512/d=2048: 2×512/2048 = 0.5× saving. Per-layer: 16 MB → 8 MB. 28 layers: 448 MB → 224 MB per matrix.

For 70B: W_V, W_O each at d=8192, K=2048: 2 × 8192 × 2048 × 2 bytes = 67 MB per layer. 80 layers: 5.37 GB per matrix → 2.69 GB at K=2048 (4× compression). Both: 10.74 GB → 5.37 GB total saving. Combined with Q/K from R1-A/R3-A: total attention compression ~15 GB → ~4 GB at 70B scale.

### Novelty gloss

No kill-list entry directly covers W_V/W_O rank structure. AQFKV (R3-A selected) explicitly lists "W_V and W_O spectra: untested; cheap parallel measurement (5-10 min each)" as open. This is the natural next rung, not a novel frame — but it is genuinely open. The Reach value is the 70B arithmetic it enables. Closest published: LoRA / AQLM / GPTQ compression — none measure W_O rank structure post-training via SVD on a trained model without distillation. The key claim is whether W_O (residual-stream output) follows the same concentrated-spectrum pattern as W_Q (whose output is not directly into the residual stream).

### Smallest experiment

Claim: W_V in Qwen3-1.7B at global rank K=512 achieves ΔNLL ≤ +0.50 nats (analogous to W_Q at K=512: +0.20 nats and W_K at K=512: +0.29 nats).
Runtime: ~20 minutes (same script as AQFKV; substitute W_V tensor; W_O as secondary measurement).
Go threshold: W_V ΔNLL ≤ +0.50 nats at K=512 (GO); W_O ΔNLL ≤ +1.00 nats at K=512 (conditional — W_O into residual stream may be harder to compress).
No-Go value: complete CF11 boundary map: {W_Q: r_99/d=0.63, W_K: r_99/d=0.79, W_V: measured, W_O: measured} — closes the attention weight characterization and determines which matrices admit compression and which don't.

### Primary risk

W_O is structurally the most full-rank attention matrix because its output is directly added to the residual stream (analogous to W_down in MLP — untested but predicted to be full-rank by CF8 boundary analogy). If W_O is full-rank (r_99/d ≈ 1.0), the attention-side compression is 3-matrix (Q, K, V) and W_O must stay at full precision, limiting total attention compression to ~60% of what R1-A assumed. Mitigation: even 3-matrix compression at 70B yields ~8 GB saving — still significant.

---

## R7-B — HYBRID-TIER-70B: Hot/Warm/Cold Tiered-Quantization for Maximum DRAM-Resident Quality [FREE SWING]

**Track: B (compression)**

### Mechanism

Track B. The database-engine hot/warm/cold tier design principle (from Reach's cross-domain seed list) applied to weight quantization. The insight: not all weight matrices have equal quality sensitivity; the structural measurements (CF11, CF12, CF8) define a natural tiering. Tier assignment by empirical ΔNLL gradient:
- Cold (most compressible): W_Q (r_99/d=0.63) — INT2 or ternary; large absolute saving
- Warm: W_K (r_99/d=0.79), W_V (if similar to W_K), MLP weights at INT4 baseline
- Hot (least compressible): tied lm_head (CF12: catastrophic at r<2048), W_down (predicted full-rank), W_up (CF5: more rank-sensitive than W_gate)

The cascade: Rung 1 = build a per-matrix sensitivity table for Qwen3-1.7B covering all matrix types (W_Q, W_K, W_V, W_O, W_gate, W_up, W_down, W_E/lm_head) under rank truncation and bit-width reduction simultaneously. Measure ΔNLL per GB saved for each matrix type. Rung 2 = under a 7.28 GiB budget constraint, solve the knapsack: maximize model quality subject to total residency ≤ 7.28 GiB. Rung 3 = scale the optimal tier assignment to 70B-class model, project residency, verify quality.

The [FREE SWING] flag is because this is a systems-engineering optimization cascade rather than a novel algebraic mechanism. It exploits the SUMMARY.md structural measurements as cost-model inputs but does not produce a new structural finding — it produces a deployment configuration. The value is: given the empirical map CF8/CF11/CF12 now covers most matrix types, the optimal tiering is computable rather than guessed.

### Residency arithmetic

For Qwen3-1.7B parameter budget ~3.6 GB at bf16:
- W_Q at INT2 (1.58 bpw) vs bf16: 236 MB → 18.6 MB (12.7× saving, ~+1.5 nats at aggressive INT2)
- W_K at INT4 (4.5 bpw) vs bf16: 236 MB → 66.8 MB (3.5× saving, ~+0.5 nats)
- W_gate at INT4: 1.7 GB → 480 MB (3.5× saving, empirical confirms full-rank, tolerable ΔNLL ≈ CF5 reference)
- W_up at INT4: 1.7 GB → 480 MB (analogous; W_up more sensitive per CF5)
- lm_head at bf16 (tied): must remain full-rank; 792 MB
- Total at this tiering: 18.6 + 66.8 + 480 + 480 + 792 ≈ 1.84 GB. Within 7.28 GiB easily.

For 70B-class (Qwen3-72B proxy):
- W_Q (all layers, INT2): 10.7 GB → 0.84 GB
- W_K/W_V (INT4): 10.7 GB each → 3.03 GB each
- MLP W_gate/W_up/W_down (INT4): ~33 GB → 9.35 GB
- lm_head (untied in 8B+): lm_head spectrum untested; at untied Qwen3-8B may be compressible
Total: 0.84 + 3.03 + 3.03 + 9.35 + (lm_head/embed ~1.3 GB) ≈ 17.6 GB. Still over 7.28 GiB. Need further rung.

DRAM-resident 70B remains ~17 GB minimum even with aggressive tiering, confirming the bandwidth wall analysis from R4-B. The honest finding of this cascade is the floor of achievable residency given empirical CF findings.

### Novelty gloss

Closest kill-list: MDL-Selected Per-Layer bpw (R2/S2 killed — pre-empted by Compressibility Measures Complexity Oct 2025). Structural difference: HYBRID-TIER uses empirical ΔNLL-per-GB measurements across ALL matrix types (not just per-layer within one type) as the cost function, and solves a multi-class knapsack rather than selecting per-layer bpw. The CF8/CF11/CF12 boundary map is the novel input the published landscape did not have when MDL-per-layer-bpw was proposed. Closest published: mixed-precision quantization (GPTQ, SpQR, SqueezeLLM) — all use Hessian-based sensitivity; this uses empirical rank-structure measurements as the sensitivity proxy, which is structurally different and allows joint optimization across matrix types.

### Smallest experiment

Claim: per-matrix ΔNLL-per-GB sensitivity ratios vary by ≥ 10× across matrix types (W_Q vs W_up) in Qwen3-1.7B, making non-uniform tiering strictly better than uniform bit-width reduction.
Runtime: ~1 hour (collect ΔNLL at two quantization levels for each matrix type; 6 matrix types × 2 levels = 12 forward passes × ~5 min each = 60 min).
Go threshold: ratio of ΔNLL/GB between most-compressible and least-compressible matrix types ≥ 10× (go/no-go for the tiering argument).
No-Go value: if all matrix types have similar ΔNLL/GB ratios, uniform bit-width is already near-optimal — this is a useful negative finding that simplifies future quantization decisions.

### Primary risk

The sensitivity measurements are already partially known (CF5, CF11, CF12) for some matrices; the knapsack may be solvable without the full matrix — risk is that the result confirms what is already known without adding new deployment insight. Mitigation: the value is in the 70B-scale projection arithmetic (Rung 3), which is novel even if Rung 1 confirms known orderings.

---

## Convergence handles

1. **W_V and W_O rank structure** — multiple ideas (R1-A, R3-A, R6-B) depend on this measurement; it is the highest-leverage single experiment in this set.
2. **Cross-layer W_Q basis stability** — if 28 layers span the same ~128-dim W_Q subspace, the savings cascade dramatically; if not, per-layer compression is the only path.
3. **K-dependent Jaccard partition (CF3 K=0.1% static / K=1% dynamic)** — RAOK-SCALE (R2-B) and HYBRID-TIER-70B (R7-B) both depend on whether the CF3 boundary is an activation structure or a calibration artifact.
4. **Calibration scale / n_params conditioning (CF10)** — WDLA-RESCUE (R5-A) is gated on re-deriving the WDLA conditioning fix; any future calibration-fit idea must verify this ratio before proceeding.
5. **NVMe sequential throughput on this hardware** — TERNARY-NVMe (R4-B) must re-derive this as its first rung; affects the entire NVMe-paging family.
6. **Per-layer ΔNLL independence assumption** — all multi-layer compression arithmetic assumes approximate independence; the residual skip-connection may absorb or amplify errors non-additively. R1-A flags this explicitly; testing it on Qwen3-1.7B with multi-layer simultaneous truncation is the settlement.
