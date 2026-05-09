# Stage 1 — Reach Orientation (R) — Run 003

Orientation: R (Reach). Track A and Track B ideas. Kill-list hard constraint: no cross-layer W_Q shared-basis / stacked SVD (v2-CHEAP-TEST-001). CF8 applies to MLP weights only; CF11 opens attention W_Q at per-layer K=128 global; CF12 closes tied lm_head SVD. All residency arithmetic uses Ryzen 5 7530U: DRAM ~11.5 GB/s, NVMe ~700 MB/s cold / ~3 GB/s sequential extent-prefetch (CF13*, unverified in v2).

---

## R1-A / MHRS — Multi-Head Residency Staircase (Track A)

### Name
MHRS — Multi-Head Residency Staircase. R, Track A.

### Mechanism
Track A — arch-transposition. CF11 establishes that global K=128 (≈8× compression on W_Q) yields ΔNLL=+0.98 nats and that W_K at K=512 yields ΔNLL=+0.29 nats. The per-layer decomposition is already paid for (the 28 × K=128 projection matrices exist). This idea constructs a **staircase deployment**: layers 0–9 (early, lower representational load) use W_Q at K=64 (untested but structurally consistent with the spectral concentration — var@256=0.64 observed at L14 implies further compression may hold in early layers); layers 10–19 use K=128; layers 20–27 use K=256 (cf. +0.51 nats globally at K=256). W_K follows a parallel staircase: K=256 early, K=512 deep. W_V and W_O are held at BF16 pending their spectrum measurement (both untested per AQFKV open questions). The load-bearing novelty is the **layer-stratified** application: a global K optimized across all layers wastes bits on early layers where the head-redundancy signal is presumably stronger (early layers feed downstream — their outputs are repeatedly overwritten by later residual additions, so quality cost of early-layer W_Q errors compounds less than deep-layer errors). Not CF8 MLP rank reduction. Relies on CF11 spectral concentration and the layer-stratification hypothesis (testable cheaply). Closest prior art: A3 (arXiv:2505.12942) uses attention-logit-error minimization globally; this uses per-layer-tier weight Frobenius error with the staircase depth hypothesis as the compression lever. Not killed by v2-CHEAP-TEST-001 because that kill was cross-layer stacking of W_Q subspaces; this is per-layer rank reduction with different K per depth tier — independent subspaces, no stacking claim.

### Residency arithmetic
Qwen3-72B baseline at BF16: 72B × 2 bytes = 144 GB. IQ4_XS-equivalent (0.55 bpw equivalent) = ~5.35 GiB. We target the attention weight class only first.

Qwen3-72B: d_model=8192, 64 heads, 80 layers. W_Q per layer: 8192×8192 BF16 = 128 MB × 80 = 10.24 GB total W_Q. W_K: 8192×1024 (GQA 8 kv-heads) = 16 MB × 80 = 1.28 GB total W_K.

MHRS staircase (K values scaled from 1.7B findings proportionally; d_Q=8192):
- Layers 0–26 (early tier, K=256): W_Q compressed to 8192×256 BF16 = 4 MB × 27 = 108 MB
- Layers 27–53 (mid tier, K=512): 8192×512 = 8 MB × 27 = 216 MB
- Layers 54–79 (deep tier, K=1024): 8192×1024 = 16 MB × 26 = 416 MB

Naive staircase attention weight total: ~740 MB vs 10.24 GB = **13.8× W_Q compression**. W_K staircase similarly ~4× reduction = ~320 MB vs 1.28 GB.

Combined attention weight reduction: ~(10.24 + 1.28) GB → ~(0.74 + 0.32) GB = 10.46 GB → 1.06 GB = **9.9× attention weight compression**.

Fraction of total 70B weights: attention weights ≈ 25–30% of total (MLP dominates). If 70B at Q4_K = 38 GB, attention ≈ 9.5 GB; MHRS compresses to ~1 GB, saving 8.5 GB. Net model: ~30 GB, still too large for 7.28 GiB DRAM. Stage in a cascade: MHRS as rung 1, aggressive MLP quantization (RAOK-style or SOBIB) as rung 2. If MLP achieves 2.2 bpw (SOBIB target), MLP weights ≈ 20B × 2.2/16 × 2 = ~6 GB. Combined: 6 + 1 = 7 GB — fits 7.28 GiB DRAM.

Tok/s at DRAM bandwidth: 7 GB / 11.5 GB/s = 0.61 s/tok → ~1.6 tok/s. With quantized MLP matmul speedup (AVX2 INT2+outlier): ~2.5–4 tok/s estimated. Quality cost: ΔNLL ~+1.2 to +2.0 nats (staircase assumption; layer-stratified quality is better than uniform global K=128 across all depths). P(end-to-end): 0.30 — depends on three independent claims: (a) layer-stratified W_Q compression holds on 72B-scale (CF11 was 1.7B), (b) W_V/W_O spectrum is compressible (untested), (c) MLP quantization cascade achieves 2.2 bpw.

### Novelty gloss
Closest killed idea: AQFKV (R3-A, selected and run) — that was a global K sweep; MHRS is the layer-stratified follow-on that converts the CF11 finding into an deployable staircase. Not killed by v2-CHEAP-TEST-001 (no cross-layer basis sharing; independent per-layer decompositions). Closest published: A3 (attention low-rank post-training); LLM-in-a-Flash (Apple) which tiles and offloads but doesn't rank-reduce attention. The structural difference from A3: A3 minimizes attention-logit error via SGD-like fine-tuning; MHRS is zero-SGD post-hoc SVD exploiting the layer-depth hypothesis (early layers tolerate lower K). The layer-stratification is the novel rung — no published method stratifies W_Q rank by depth on Qwen3-family zero-SGD.

### Smallest experiment
**Claim**: W_Q at K=64 (50% further reduction below CF11's K=128 ceiling) achieves ΔNLL ≤ +1.5 nats on early layers (0–9) of Qwen3-1.7B-Base, while layers 10–27 remain at K=128; combined staircase ΔNLL < K=128 uniform.

Runtime: ~30 min on Qwen3-1.7B (28 layers; per-layer SVD at K=64 early, K=128 deep; NLL eval on 512-token WikiText-2).

Go threshold: ΔNLL_staircase < ΔNLL_uniform_K128 = +0.98 nats AND staircase ΔNLL < +1.5 nats.

No-go finding: if early-layer K=64 degrades worse than K=128 uniform, this establishes that early-layer W_Q concentration is NOT higher than deep-layer — the layer-stratification hypothesis is false, which is a structural finding about attention weight spectrum depth-dependence.

### Primary risk
Layer-stratification hypothesis may be reversed: deep layers in Qwen3 may have MORE concentrated W_Q (lower intrinsic dimensionality) than early layers, since early layers must process raw token embeddings without prior residual signal. Mitigation: measure per-layer W_Q r_99 in the smallest experiment before assuming the staircase direction; reverse the staircase if deep < early.

---

## R2-B / RAOK-72 — R2-Aware Outlier-Keyed Activation Codebook at 70B Scale (Track B)

### Name
RAOK-72 — R2-Aware Outlier-Keyed Activation Codebook at 70B Scale. R, Track B. Escalation of deferred RAOK (Track B R6) to full Reach cascade targeting 70B DRAM residency.

### Mechanism
Track B — compression. CF3 (confirmed R2) establishes a three-tier activation-channel structure on Qwen3 family: K=0.1% channels (≈2 channels at d=2048) have Jaccard 0.718 across consecutive tokens — effectively static; K=1% channels (≈20 channels) have Jaccard 0.308 — token-dynamic; K>1% channels are dense-dynamic. The RAOK mechanism exploits this as a **lossless-then-lossy partitioning of activation space**: (A) top 0.1% channels (2 at d=2048, scaling to ~8 at d=8192 for 72B) stored FP16 statically — zero runtime overhead for these channels; (B) channels 0.1%–1.0% stored as INT8 with per-token dynamic scaling — cheap to apply at runtime; (C) remaining 99% channels stored INT4 statically — standard Q4 cost. The three-tier codebook applies to W_down inputs (the post-SwiGLU activations that W_down multiplies), converting a weight-compression problem into an activation-aware storage problem. The W_down weight matrix itself is stored at standard INT4 or INT2+outlier; the activation-tier partitioning determines how to handle the matmul precision without changing W_down byte count. Distinct from CF10-violating calibration-fitted methods: the partitioning key is the empirically measured K-dependent Jaccard (CF3), not a least-squares fit. Not CF8 MLP weight rank reduction. Relies on CF3 (confirmed). The 72B scaling is the Reach rung: d=8192 implies ~32 static channels (0.1% × 8192 × 16 heads / head-dim scaling), ~320 channels for INT8 tier, rest INT4. The RAOK-72 cascade claim: activations partitioned this way tolerate lower-bit W_down storage without ΔNLL penalty equal to uniform quantization at the same bpw.

### Residency arithmetic
Qwen3-72B with RAOK-72 activation-aware W_down quantization:
- W_down per layer: 8192 × 29568 BF16 = 466 MB. At INT2+outlier (~2.2 bpw SOBIB target): 466 × (2.2/16) = 64 MB per layer × 80 = 5.12 GB total W_down.
- W_up, W_gate: same dimensions; SOBIB INT2+outlier = ~5.12 GB each × 2 = 10.24 GB MLP excl W_down.
- Attention (MHRS staircase, from R1-A): ~1 GB.
- Embeddings (tied or untied — untied at 72B): 152K × 8192 BF16 = 2.4 GB; CF12 kills SVD; carry at lower bpw IF untied (open). Conservative: 2.4 GB.

Total: 5.12 (W_down) + 10.24 (W_up+W_gate) + 1.0 (attn MHRS) + 2.4 (embed) + misc = **~19 GB** at 2.2 bpw MLP.

Still exceeds 7.28 GiB — but approaching NVMe-streaming feasibility. At NVMe sequential (~3 GB/s CF13*): 19 GB / 3 GB/s = 6.3 s/tok → ~0.16 tok/s. Insufficient. The DRAM-resident path requires sub-1-bpw for MLP. If RAOK-72's activation-aware partitioning allows W_down at 1.5 bpw without quality loss: W_down alone = 3.5 GB; total MLP ≈ 3.5 + 7.0 = 10.5 GB + 1 + 2.4 = 13.9 GB. At NVMe: 4.6 s/tok. Still slow. RAOK-72 is a cascade input, not a standalone floor-hitter. Its value: reduce quality cost at 2 bpw, making the overall cascade more aggressive without ΔNLL blowup. P(RAOK-72 tier-partitioning reduces ΔNLL at 2 bpw by ≥0.3 nats vs uniform): 0.55. P(full cascade reaches 7 GiB DRAM): 0.10 — needs sub-1 bpw MLP, which is outside the current empirical boundary.

### Novelty gloss
Closest killed: RAOK was deferred (Track B R6 Stage 4) and SOBIB was killed. RAOK-72 is not killed; it is the deferred RAOK escalated to Reach scale. Closest published: GPTVQ (Hessian-weighted codebook), KVQuant, PrefixQuant. Structural difference: PrefixQuant decomposes by token type (BOS vs normal); RAOK-72 decomposes by channel tier grounded in the empirically measured K-dependent Jaccard (CF3). PrefixQuant doesn't have the three-tier codebook grounded in per-token Jaccard measurements on Qwen3 family. The CF3 anchor is the moat: it's not on arXiv with this measurement's specificity.

### Smallest experiment
**Claim**: RAOK three-tier partition (2 FP16 static + 18 INT8 dynamic + 2028 INT4 static) applied to Qwen3-1.7B W_down inputs reduces ΔNLL at 2.2 bpw by ≥0.2 nats vs uniform INT2+outlier at the same bpw.

Runtime: ~45 min. Load Qwen3-1.7B BF16; identify top-0.1% and 0.1-1% channels by K-dependent Jaccard profile from PDAP (CF3 already measured on 200-prompt run); simulate three-tier matmul on calibration distribution; measure NLL on 512-token WikiText-2 hold-out.

Go threshold: ΔNLL_RAOK ≤ ΔNLL_uniform − 0.2 nats at matched bpw.

No-go finding: if tier-partitioning buys no NLL reduction, then the K-dependent Jaccard structure in activations does NOT translate to a quantization error reduction — structural finding that activation outlier dynamicity and quantization error are decoupled.

### Primary risk
The K-dependent Jaccard profile (CF3) was measured on activations (residual stream), not on W_down input activations (post-SwiGLU). The outlier structure may differ at the W_down input. Mitigation: measure the K-dependent Jaccard on post-SwiGLU vectors directly in the smallest experiment as a precondition gate.

---

## R3-A / WAVQ — W_V / W_O Attention Spectrum Cascade (Track A)

### Name
WAVQ — W_V / W_O Attention Spectrum Cascade. R, Track A.

### Mechanism
Track A — arch-transposition. CF11 (AQFKV) opened the W_Q and W_K spectrum question and confirmed concentration (W_Q r_99/d ≈ 0.63, W_K r_99/d ≈ 0.79). The AQFKV open questions explicitly list "W_V and W_O spectra: untested; cheap parallel measurement." WAVQ converts this cheap measurement into a Reach cascade: IF W_V (d×d_head × n_heads, or GQA variant) and W_O (n_heads×d_head × d) show r_99/d significantly below 1.0 (say ≤ 0.80), then ALL four attention weight matrices (Q, K, V, O) are jointly compressible. The cascade claim: W_V at K=512 + W_O at K=512 (by analogy to W_Q K=512 global +0.20 nats, W_K K=512 +0.29 nats) could reduce attention residency by an additional 2× on top of MHRS. The architectural novelty is in the **joint low-rank attention operator**: once W_Q, W_K, W_V, W_O are all rank-reduced, the product W_O × softmax(W_Q K^T / √d) × W_V is a low-rank operator end-to-end — the KV cache content is projected through a low-rank W_V, so KV cache entries are inherently lower-dimensional. This is a secondary Reach claim: KV cache compressed for free as a consequence of W_V rank reduction, without touching the KV cache quantization literature. Not CF8 MLP weight reduction. Not v2-CHEAP-TEST-001 cross-layer stacking — this is per-layer rank reduction of V and O independently. Relies on CF11 as motivating evidence; the W_V/W_O spectra are the first load-bearing empirical gate.

### Residency arithmetic
Qwen3-72B: W_V per layer (8 GQA kv-heads, d_head=128) = 8192×1024 = 16 MB BF16. W_O = 8192×8192 = 128 MB BF16. Total per layer = 144 MB × 80 layers = 11.5 GB.

At K=512 for W_V and W_O (analogous compression to W_K K=512 which gave +0.29 nats):
W_V compressed: 8192×512 × 8 heads = 8192×512 × 8 BF16 = 64 MB → 512/1024 = 50% reduction → 8 MB × 80 = 640 MB
W_O compressed: 8192×512 = 8 MB × 80 = 640 MB (vs 10.24 GB)

Total W_V + W_O: 640 + 640 = 1.28 GB vs 11.5 GB = **9× compression** on V+O.

Combined with W_Q (MHRS, ~0.74 GB) and W_K (staircase, ~0.32 GB):
Total attention: 0.74 + 0.32 + 0.64 + 0.64 = **2.34 GB** vs (10.24 + 1.28 + 11.5) = 23 GB original attention weights.

KV cache bonus: at K=512 W_V, KV cache for V side is effectively a 512-dim projection (8 GQA heads × 512 × 2 bytes × 32K context = 256 MB for 32K context) vs 1 GB at full W_V. KV cache compressed from ~1 GB to ~256 MB for 32K context at no additional code complexity.

Full model residency (MLP at 2.2 bpw SOBIB + attention WAVQ): 
~10.24 GB MLP + 2.34 GB attention + 2.4 GB embedding = **~15 GB**. NVMe-streaming territory.

P(W_V spectrum concentrated, r_99/d ≤ 0.80): 0.60 by structural analogy to W_K (0.79) and the attention-weight concentration pattern.
P(W_O spectrum concentrated, r_99/d ≤ 0.80): 0.50 (W_O maps multi-head outputs back to residual stream; possibly more diverse due to summing 16 heads).
P(end-to-end full attention compression cascade): 0.35.

### Novelty gloss
Closest killed: none directly — W_V/W_O spectrum has not been measured and not been proposed in the pipeline. Closest published: MLA (DeepSeek-V2) does joint low-rank KV at training time; this is post-training, zero-SGD, pure SVD, tested on Qwen3 family. FlashAttention and variants do not compress W_V. The KV cache reduction as a free consequence of W_V rank reduction is not in MLA (MLA still needs to materialize full KV at some point during precompute). The structural difference: WAVQ inherits the CF11 spectral-concentration finding and applies it to the untested V and O matrices — if the pattern holds, the compression ratio is load-bearing and produces a structural finding about all four attention weight classes.

### Smallest experiment
**Claim**: W_V r_99/d < 0.80 AND W_O r_99/d < 0.80 on Qwen3-1.7B-Base BF16, and W_V at K=512 achieves ΔNLL ≤ +0.50 nats.

Runtime: ~25 min. Compute SVD of W_V and W_O for all 28 layers (batched); plot singular value decay; measure NLL with W_V at K=512 (W_Q, W_K, W_O held at BF16). Go/no-go on r_99 threshold.

Go threshold: W_V r_99/d ≤ 0.80 AND ΔNLL_WV_K512 ≤ +0.50 nats.

No-go finding: if W_V is full-rank (r_99/d > 0.90), this closes the attention weight class characterization — all four matrices measured, only W_Q compressible meaningfully. This is a complete map of the attention weight spectrum family, worth producing regardless of sign.

### Primary risk
W_O (output projection) aggregates 16 heads' information and may require nearly full rank to preserve the diversity of head outputs — the mechanism predicts W_O is MORE sensitive to truncation than W_Q. If both W_V and W_O are full-rank, the cascade collapses; only W_Q and W_K matter for attention compression. Mitigation: run W_V and W_O spectrum measurements in parallel in the smallest experiment before assuming compression ratios.

---

## R4-B / SOBIB-V — INT2 + Sparse FP16 Outliers on W_down (Track B)

### Name
SOBIB-V — Selective Outlier Binary INT2 + Sparse FP16 outliers on W_down. R, Track B. Targeting W_down specifically, grounded in CF3 + CF5.

### Mechanism
Track B — compression. CF5 (confirmed) shows W_up is MORE rank-sensitive than W_gate (+2.34 vs +0.77 dNLL at 50% rank). This establishes W_up carries the dominant firing-rank signal. CF3 (confirmed) shows the post-SwiGLU activation channels are structured: top 0.1% static, 0.1%–1% token-dynamic outliers. W_down multiplies the post-SwiGLU activations — its input is the activation tensor with this three-tier structure. SOBIB-V exploits: **the static outlier channels (top 0.1%, Jaccard 0.718) correspond to rows of W_down that receive consistently high-magnitude inputs. Those rows are always load-bearing — they should be stored at higher precision.** Conversely, W_down columns corresponding to the 99% dynamic channels receive low-magnitude activations on average — those columns tolerate INT2. The SOBIB-V weight layout: (a) ~2% of W_down columns (mapped from the static outlier rows of the activation space) stored FP16; (b) 0.1%–1% columns (dynamic outlier region) stored INT8; (c) 99% columns stored INT2. No calibration fitting — the column selection key is the empirically measured K-dependent Jaccard threshold from CF3 (a binary partition, not a regression). Total bpw: 0.02 × 16 + 0.01 × 8 + 0.97 × 2 = 0.32 + 0.08 + 1.94 = **2.34 bpw** for W_down. This is the Reach rung: W_down at 2.34 bpw, W_up at standard INT4 (4 bpw), W_gate at standard INT4 = overall MLP ~3.4 bpw average, meaningfully below Q4. For full 70B: MLP weights at 3.4 bpw average = 72B × 3.4/16 = 15.3 GB. Still large; if W_up also gets SOBIB treatment (its input is x, the residual stream — CF3 measured on the residual stream) then W_up at 2.34 bpw also; MLP total = 72B × 2.34/16 = 10.5 GB. Not CF8 MLP rank reduction — this is quantization, not SVD truncation.

### Residency arithmetic
Qwen3-72B MLP weights at SOBIB-V on W_down + W_up (both using CF3-grounded column partitioning):
- W_down: 8192 × 29568 per layer × 80 layers BF16 = 37.3 GB. At 2.34 bpw: 37.3 × 2.34/16 = **5.45 GB**.
- W_up: 8192 × 29568 per layer × 80 layers BF16 = 37.3 GB. At 2.34 bpw: **5.45 GB**.
- W_gate: same dimensions. Less sensitive (CF5 reversed claim: W_gate at 4 bpw INT4): **5.45 GB** at 4 bpw = wait — 37.3 × 4/16 = 9.3 GB. Or if W_gate also SOBIB-V at 2.34 bpw: 5.45 GB.
- Conservative: W_gate at INT4 (9.3 GB) + W_down + W_up at SOBIB 2.34 bpw (5.45 × 2 = 10.9 GB) = **20.2 GB MLP**.
- Aggressive: all three at 2.34 bpw = **16.4 GB MLP**.
- Attention (MHRS staircase) = 2.34 GB. Embedding = 2.4 GB.
- Total aggressive: 16.4 + 2.34 + 2.4 = **21.1 GB** — NVMe streaming.

At NVMe sequential ~3 GB/s (CF13*): 21 GB / 3 = 7 s/tok → 0.14 tok/s. Insufficient for usable inference alone. However: if 7.28 GiB DRAM can be used as a hot-tier buffer for the most-accessed attention layers + embeddings, and MLP is streamed from NVMe, overlap improves. The Reach target is the cascade: SOBIB-V + MHRS + NVMe-prefetch = viable 70B.

For 7B DRAM-resident scenario: Qwen3-7B at d=3584, intermediate=18944, 28 layers. MLP at SOBIB 2.34 bpw = 7B × 2.34/16 × 2 bytes = ~2 GB. Attention at MHRS = ~0.3 GB. Embedding = 0.6 GB. Total = ~3 GB → fits 7.28 GiB DRAM. Tok/s at DRAM bandwidth: 3 GB / 11.5 = 0.26 s/tok → ~3.8 tok/s. Within target range. P(SOBIB column-partitioning works at 2.34 bpw with ≤0.5 nats): 0.45.

### Novelty gloss
Closest killed: SOBIB was killed in Track B R5 for "calibration-only sensitivity may underperform Hessian-based; AVX2 scatter overhead." SOBIB-V differs: the column selection key is the empirically measured K-dependent Jaccard from CF3 (not calibration sensitivity scoring), and the precision tiers are fixed by the measured threshold (0.1% static, 0.1%–1% dynamic) rather than fitted. The CF3 anchor grounds the partition without calibration regression. Closest published: SpQR (sparse INT4 + FP16 outliers by activation-weighted Hessian), SmoothQuant (channel-wise scaling). Structural difference: SOBIB-V uses a row/column partitioning based on the load-bearing activation-channel stability property (CF3), not a trained Hessian — zero additional optimization, zero overfitting risk (CF10 does not apply).

### Smallest experiment
**Claim**: W_down column partitioned by CF3 K-dependent Jaccard threshold (≤0.1% FP16, 0.1–1% INT8, rest INT2) on Qwen3-1.7B-Base achieves ΔNLL ≤ +0.8 nats vs BF16 baseline at implied ~2.34 bpw, outperforming uniform INT2 by ≥0.3 nats.

Runtime: ~40 min. Load Qwen3-1.7B; apply column partition to W_down using PDAP measurement results (already have per-channel Jaccard from R2 run); simulate matmul with tier-appropriate precision; measure NLL on 512-token WikiText-2.

Go threshold: ΔNLL_SOBIB-V ≤ +0.8 nats AND ΔNLL_SOBIB-V ≤ ΔNLL_uniform_INT2 − 0.3 nats.

No-go finding: if CF3-derived column partition buys no advantage over uniform INT2, establishes that activation-channel stability does NOT predict W_down column sensitivity — decouples the two. Structural finding regardless.

### Primary risk
W_down column indexing corresponds to input activation channels, but the outlier-to-column mapping requires knowing WHICH channels are the static-outlier channels at each layer — this is a per-layer, per-model calibration artifact from the PDAP measurement, not a universal constant. Runtime inference must carry the partition index for all 28 (or 80) layers. Mitigation: store partition index as a small side-table in GGUF metadata (~2 KB per layer, negligible).

---

## R5-B / NTQW — Nested-Tier Q-K-V-O Weight Cascade to DRAM-Resident 7B [FREE SWING]

### Name
NTQW — Nested-Tier Attention-First Weight Cascade. R, Track B. [FREE SWING]

### Mechanism
Track B — compression with systems cascade flavor. The ambitious claim: a 7B-class model (Qwen3-7B) can run fully DRAM-resident on 7.28 GiB at usable tok/s by combining ALL confirmed compressible weight classes without retraining. The cascade has four rungs, each with its own measurable gate:

Rung 1 — W_Q at global K=128 (CF11 confirmed: ΔNLL +0.98 nats, 8× compression on W_Q).
Rung 2 — W_K at global K=512 (CF11 confirmed: ΔNLL +0.29 nats, 2× compression on W_K).
Rung 3 — W_V and W_O at K=512 (WAVQ, to be measured; expected ΔNLL +0.3–0.5 nats by structural analogy; 8–10× compression on V+O combined).
Rung 4 — MLP (W_gate, W_up, W_down) at 3.4 bpw average via SOBIB-V column-partitioning (to be measured; expected ΔNLL +0.5–1.0 nats).

Residency arithmetic for Qwen3-7B: d=3584, intermediate=18944, 28 layers, 32 heads (GQA 8 kv-heads), d_head=128.
- W_Q (28 layers): 3584×3584 × 28 BF16 = 7.2 GB. At K=128: 3584×128×28 = 0.46 GB. **15.6× compression**.
- W_K (28): 3584×512 × 28 BF16 = 0.9 GB. At K=512 same: 0.9 GB (W_K already small due to GQA; K=512 = full K_dim; no compression). Rephrase: K_dim for 8 kv-heads = 8 × 128 = 1024. W_K = 3584×1024 × 28 BF16 = 1.82 GB. At K=512 (50%): 0.91 GB.
- W_V (28): 3584×1024 × 28 BF16 = 1.82 GB. At K=512: 0.91 GB.
- W_O (28): 3584×3584 × 28 BF16 = 7.2 GB. At K=512: 3584×512×28 = 0.91 GB. **7.9× compression**.
- Total attention: (0.46 + 0.91 + 0.91 + 0.91) = **3.19 GB** vs (7.2 + 1.82 + 1.82 + 7.2) = 18.04 GB original. **5.7× attention compression**.
- W_gate + W_up + W_down (28 layers): 3584×18944 × 3 × 28 × 2 bytes = 11.3 GB BF16. At 3.4 bpw: 11.3 × 3.4/16 = **2.4 GB**.
- Embedding (3584×152000 BF16): 1.08 GB (untied at 7B likely — CF12 applies to 1.7B tied; 7B untied may compress with SVD — TBD, hold at BF16).
- Total: 3.19 + 2.4 + 1.08 = **6.67 GB** — fits 7.28 GiB DRAM.

Tok/s at DRAM bandwidth: 6.67 GB / 11.5 GB/s = 0.58 s/tok → **~1.7 tok/s**. Better than NVMe cold (~0.1 tok/s). With quantized MLP arithmetic speedup (AVX2 INT2+INT4 mixed): ~2.5–3.5 tok/s estimated.

Quality budget: Rungs 1+2+3 (attention, independent): ~+0.98 + 0.29 + 0.40 = +1.67 nats (assuming W_V+W_O add ~0.40). Rung 4 (MLP): ~+0.7 nats. Total compositional ΔNLL: ~+2.4 nats worst case (assuming independence, which is optimistic). This is a quality floor at the bottom of usable for generation tasks; search and extraction tasks tolerate more.

P(end-to-end Rung 1+2): 0.85 (CF11 confirmed). P(Rung 3 W_V/W_O): 0.55. P(Rung 4 MLP): 0.45. P(all four rungs, cascade fits 7.28 GiB, and quality ΔNLL < 2.5 nats): **0.20**. Low but the structural findings on NO-GO at each rung are worthwhile.

### Novelty gloss
Closest killed: no prior idea has explicitly assembled the full four-matrix attention compression + MLP SOBIB cascade targeted at a 7B DRAM-resident deployment on 7.28 GiB. Individual rungs have precursors (AQFKV, RAOK, SOBIB), but the end-to-end cascade is novel as an assembled system. Closest published: Apple LLM-in-a-Flash does full-model NVMe offload with weight subset loading; does NOT do per-matrix rank reduction. GPTQ + Q4_K does MLP quantization without attention rank reduction. The structural novelty is the four-matrix attention rank reduction + MLP column-partitioned INT2 assembled as a single deployment package. The FREE SWING tag reflects the systems-engineering integration quality of this proposal over mechanism novelty.

### Smallest experiment
**Claim**: Rungs 1+2 (W_Q K=128 + W_K K=512) simultaneously applied to Qwen3-1.7B-Base give total ΔNLL ≤ +1.5 nats (not much worse than W_Q-only +0.98) — confirming rung independence.

Runtime: ~30 min. Apply W_Q K=128 and W_K K=512 simultaneously; measure NLL on 512-token WikiText-2 hold-out.

Go threshold: ΔNLL_simultaneous ≤ +1.5 nats (i.e., rungs add approximately, not multiplicatively).

No-go finding: if simultaneous application degrades far beyond additive (say > +2.5 nats), the rungs interact destructively through the attention score — a structural finding about attention rank reduction composability.

### Primary risk
Rung interaction: low-rank W_Q + low-rank W_K jointly produce attention logits W_Q·K^T which may lose critical precision even when each matrix compresses cleanly individually. Mitigation: measure the simultaneous application (Rung 1+2 joint) in the smallest experiment before assuming rung independence.

---

## R6-A / AERO-QW3 — AERO-Style SwiGLU Activation Removal for Qwen3 (Track A)

### Name
AERO-QW3 — AERO-Style SwiGLU Gate Elimination via Activation Removal, Qwen3 Zero-SGD. R, Track A.

### Mechanism
Track A — arch-transposition. AERO (arXiv:2410.13060) eliminates SiLU (ReLU-family) activations entirely and folds W_gate + W_up into a single combined matrix. The algebraic identity: if f(x) = SiLU(W_gate·x) ⊙ (W_up·x), and SiLU is replaced by the identity (SiLU(z) ≈ z on the regime of interest), then f(x) ≈ (W_gate·x) ⊙ (W_up·x) which is element-wise multiplication of two linear maps — not the same as a single linear map, so algebraic folding requires an additional approximation. The AERO paper shows that for trained models, there exists a regime where SiLU is approximately linear (positive activations, |z| >> 0 region). However, CF6 (confirmed) showed that SwiGLU gate output has median std 0.91 at layer 27 and only 1.5% of neurons have std < 0.05 globally — the gate output is overwhelmingly NOT near-constant, which was the SDZC kill premise. But AERO's claim is DIFFERENT from SDZC: AERO does not require gate outputs to be near-constant; it requires a **calibration-derived linear approximation** to SiLU that is accurate enough on the distribution. The linearization SiLU(z) ≈ α·z + β (Taylor expansion at z=z_mean calibrated on data) can be absorbed: f(x) ≈ (αW_gate·x + β) ⊙ (W_up·x) = α(W_gate·x ⊙ W_up·x) + β·W_up·x = α·W_gated(x) + β·W_up·x. This does NOT algebraically fold W_gate away (it still appears in W_gated). **AERO's mechanism for true folding requires removing the activation function entirely** — replacing SiLU(W_gate·x) with a single linear pass. On Qwen3, this would change the function (not a pure identity). The Reach question: what is the ΔNLL of AERO-style complete activation removal + W_gate/W_up matrix product fold on Qwen3-1.7B? The published AERO result (AERO paper) works for models fine-tuned with activation removed; the zero-SGD question (apply to a trained model without retraining) is genuinely open for SwiGLU families. The R3/S6 kill of CSMF was for a polynomial substitution that failed to eliminate W_gate — AERO's mechanism is different (removes activation entirely, then W_gate · W_up product collapses to element-wise, NOT to a single GEMV). Clarification: after activation removal, f(x) = (W_gate·x) ⊙ (W_up·x) = diag(W_gate·x) × W_up·x — this is still a two-matrix operation unless further structure is exploited. A rank-k expansion of diag(W_gate·x) could yield a low-rank product, but that requires W_gate·x to be approximately rank-k. CF4/CF8 says W_gate is full-rank; but at runtime, W_gate·x (a vector) is always rank-1 — the diagonal matrix diag(W_gate·x) always has a structured decomposition. The ACTUAL fold: f(x) = diag(W_gate·x) × W_up·x where diag(v) is the d_intermediate×d_intermediate diagonal. This computation has the same FLOPs as the original MLP. The only savings is the SiLU evaluation cost (negligible). **AERO-QW3's real target**: verify whether AERO's zero-SGD activation-removal version achieves meaningful ΔNLL on Qwen3, and if so, whether the eliminated SiLU non-linearity can be replaced by a structured low-rank interaction term between W_gate and W_up that enables storage reduction. The Reach claim: if ΔNLL_AERO_zero-SGD ≤ +0.5 nats, then W_gate and W_up together store a structured bilinear interaction (W_gate·x)⊙(W_up·x) that may admit a joint Khatri-Rao or Hadamard-product factorization cheaper than 2×full-rank.

### Residency arithmetic
If AERO-QW3 zero-SGD works and ΔNLL ≤ +0.5 nats: the MLP computation is now a pure bilinear form (W_gate·x) ⊙ (W_up·x). The two matrices W_gate and W_up together define a quadratic kernel in x. A joint Khatri-Rao factorization W_gate = UV^T, W_up = PQ^T at rank r = 1024 would yield f(x) = (UV^T·x) ⊙ (PQ^T·x) computable as a rank-r bilinear without materializing the full d_intermediate-dimensional intermediate. Storage: (d × r + r × d_int) × 2 matrices × 2 bytes = 2 × (3584×1024 + 1024×18944) × 2 = 2 × (7.3M + 19.4M) × 2 ≈ 214 MB vs W_gate + W_up BF16 = 2 × 3584 × 18944 × 2 ≈ 272 MB for 1.7B model. Marginal at this scale; at 72B: 8192×29568×2×2 = 3.7 GB for W_gate+W_up at BF16. At joint Khatri-Rao rank 4096: (8192×4096 + 4096×29568)×2×2 = (33.6M + 121.4M)×4 = 620 MB — **6× compression on W_gate+W_up combined**, if the approximation holds.

P(AERO zero-SGD ΔNLL ≤ +0.5 nats): 0.20 — optimistic; SwiGLU gate non-linearity is likely load-bearing in ways SiLU in ReLU-family models is not. P(joint Khatri-Rao factorization of the resulting bilinear achieves additional 6× compression): 0.15. P(end-to-end): 0.03. Very speculative; included for the structural finding value.

### Novelty gloss
AERO is published (arXiv:2410.13060) for trained-with-removed-activation models. The zero-SGD application to Qwen3 SwiGLU is genuinely untested. The CSMF kill (R3/S6) was for polynomial substitution, not activation removal — these are different mechanisms. The joint Khatri-Rao bilinear factorization post-removal is not in AERO or any adjacent work. The moat: CF4, CF6, CF8 make this a loaded empirical question — we have the measuring apparatus.

### Smallest experiment
**Claim**: AERO-style zero-SGD activation removal (replace SiLU with identity) on all 28 layers of Qwen3-1.7B-Base achieves ΔNLL ≤+2.0 nats (weak threshold — establishing feasibility territory).

Runtime: ~25 min. Monkeypatch SwiGLU forward: replace silu(gate_out) with gate_out directly; run forward pass; measure NLL on 512-token WikiText-2.

Go threshold (weak): ΔNLL ≤ +2.0 nats → nonlinearity removal is survivable, bilinear structure is real. Go threshold (strong): ΔNLL ≤ +0.5 nats → pursue Khatri-Rao factorization.

No-go finding: if ΔNLL >> 5 nats, SwiGLU gate nonlinearity is critical in zero-SGD regime — confirms SDZC-family impossibility at a deeper level, and motivates AERO as training-only.

### Primary risk
SwiGLU is intrinsically different from SiLU-only activations in published AERO — the element-wise product with W_up is the key nonlinear interaction. Removing SiLU but keeping the ⊙ operator does not simplify the computation as much as AERO claims for ReLU families. The algebraic fold may be impossible without retraining. Mitigation: the smallest experiment tests this directly; 25 min cost.

---

## R7-B / DQPM — Depth-Quantile Precision Map, Full 70B Deploy [FREE SWING]

### Name
DQPM — Depth-Quantile Precision Map, Full 70B Deployment Architecture. R, Track B. [OUT-OF-ORIENTATION for pure novelty; included as the Reach floor architecture.]

### Mechanism
Track B — compression + systems. This is the architectural integration proposal: given all the empirical findings from the ladder, what is the **tightest deployable specification** for a 70B model on 7.28 GiB DRAM? Rather than proposing a mechanism, DQPM proposes a specific deployment MAP that combines confirmed and estimated findings into a single precision schedule, layer by layer, weight by weight.

The DQPM map for Qwen3-72B (d=8192, intermediate=29568, 80 layers, 64 heads, 8 GQA kv-heads, d_head=128):

Per-layer precision assignment:
- W_Q: K=128 (CF11 confirmed GO, ΔNLL+0.98). BF16 factors stored: 8192×128 + 128×8192 per layer = 4 MB vs 128 MB = 32× compression.
- W_K: K=512 (CF11 confirmed GO, ΔNLL+0.29). 4 MB vs 16 MB = 4× compression.
- W_V: K=512 (WAVQ unconfirmed, estimated). 4 MB vs 16 MB = 4× compression.
- W_O: K=512 (WAVQ unconfirmed, estimated). 64 MB vs 128 MB = 2× compression.
- W_gate: INT4 standard (CF8 full-rank, 4 bpw).
- W_up: INT4 standard (CF5 full-rank, 4 bpw).
- W_down: SOBIB-V column-partitioned (CF3-grounded, 2.34 bpw).
- Layer 1 W_gate: eligible for 36% SDZC fold (CF6) → 0.64× W_gate size in layer 1 only.

Total per-layer residency (80 layers):
- Attention: (4 + 4 + 4 + 64) × 80 = 76 × 80 = 6.08 GB (vs 23 GB BF16)
- W_gate + W_up at INT4: 2 × (8192 × 29568 × 4/16) × 80 = 2 × 30.5 MB × 80 = 4.88 GB (each)
- W_down at SOBIB-V 2.34 bpw: (8192 × 29568 × 2.34/16) × 80 = 17.9 MB × 80 = 1.43 GB
- Total MLP: 4.88 + 4.88 + 1.43 = **11.19 GB**
- Embeddings: 152000 × 8192 × 4/16 = 0.62 GB (untied, INT4 with standard Q4_K)
- Layer norms, biases: ~0.1 GB
- **TOTAL: 6.08 + 11.19 + 0.62 + 0.1 = 17.99 GB**

Still exceeds 7.28 GiB. Conclusion: 70B DRAM-resident on 7.28 GiB is NOT achievable with the current CF set at acceptable quality. The Reach target (5.35 GiB at 0.55 bpw total) requires MLP to go below 2 bpw AND attention compression must work as estimated. At 2 bpw average (including outlier corrections): MLP = 72B × 2/16 = 9.0 GB + attention 6 GB + embed 0.6 = 15.6 GB — still NVMe territory.

The DQPM's value is **honest mapping**: the 70B-at-7.28-GiB claim requires either (a) sub-1 bpw MLP quantization (NanoQuant territory, requires custom training artifacts) OR (b) 7B-class models at aggressive compound compression (the R5-B NTQW 7B cascade gets to 6.67 GB). DQPM is the falsification of the 70B DRAM floor claim.

### Residency arithmetic
(Embedded above.) Key finding: **70B on 7.28 GiB DRAM requires sub-1 bpw average**, which is outside no-retraining zero-SGD territory given current evidence. The **7B achievable target** is more honest.

### Novelty gloss
DQPM is a synthesis / specification, not a mechanism. Its novelty is being the first complete deployment map that reads off the actual confirmed and estimated compression ratios from this ladder's specific measurements. No published method has this specific Qwen3 family characterization. The value is the deployment floor estimate, not an arXiv-novel technique.

### Smallest experiment
No new mechanism to test. The DQPM is fully determined by the outcome of R1-A (MHRS), R3-A (WAVQ), R4-B (SOBIB-V) smallest experiments. If those three all produce GO results, the DQPM total residency is computable from their confirmed ratios. The "experiment" is the arithmetic.

### Primary risk
The DQPM assumes rung independence (quality penalties add not multiply). If destructive rung interactions increase total ΔNLL significantly, the residency estimate is valid but quality is unacceptable. Mitigation: R5-B NTQW tests rung independence at 1.7B scale.

---

## Convergence handles

The following primitives underlie multiple ideas in this run's R output. Stage 4 should check whether other orientations (C, F, U, A) independently surface any of these:

1. **W_V and W_O spectral concentration** — load-bearing gate for R3-A (WAVQ) and R5-B (NTQW). If other orientations independently motivate measuring W_V/W_O, this is convergence signal.
2. **CF3 K-dependent Jaccard as a W_down column partition key** — load-bearing for R2-B (RAOK-72) and R4-B (SOBIB-V). The CF3 measurement is the moat; any idea that exploits the three-tier activation structure shares this handle.
3. **Layer-depth stratified W_Q rank** — load-bearing hypothesis for R1-A (MHRS). If early layers have lower intrinsic W_Q dimensionality than deep layers, all rank-reduction cascades should apply this staircase pattern.
4. **Rung independence / composability of attention rank reductions** — load-bearing for R5-B (NTQW cascade). If multiple orientations converge on the joint W_Q + W_K + W_V + W_O compression claim, the independence question is critical to test once, not per-rung.
5. **AERO-style zero-SGD activation removal feasibility on SwiGLU** — load-bearing for R6-A (AERO-QW3). If First-Principles orientation independently derives this as an algebraic identity target, convergence is strong signal to prioritize.
6. **7B at 7.28 GiB DRAM as the realistic target floor** (vs 70B) — the DQPM arithmetic (R7-B) makes explicit that the 70B-at-7.28-GiB floor requires sub-1-bpw territory. Any orientation reaching the same conclusion via different arithmetic is convergence on the deployment-tier question.
