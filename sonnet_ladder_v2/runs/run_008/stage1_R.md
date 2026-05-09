# Stage 1 — Orientation R (Reach) — Run 008

Orientation: R — Reach
Independent of runs 001–007.
Kills enforced: v2-CHEAP-TEST-001 (cross-layer W_Q stacked SVD class), v2-S3-R004-001 (arbitrary rotation GFQO), all SUMMARY.md class-level kills including arbitrary rotation, within-layer Q/K joint decomposition (per kill list), softmax × RoPE shift-invariance compound (R5-ATTNBAND territory, plus run-003 F3-SRSC confirmed RoPE kills softmax shift-invariance).
Run-specific kills per brief: cross-layer W_Q, arbitrary rotation, within-layer Q/K joint, softmax × RoPE.

---

## R8-WVWO-FOLD — Value-Output Product Rank Cascade

**Track A — arch-transposition.**

W_V and W_O are always applied in sequence during attention: for each head h, the contribution to the residual stream is (A_h X W_Vh) W_Oh where A_h is the attention weight. The composed map W_Vh W_Oh ∈ R^{d×d} is the only object that appears at inference. If rank(W_Vh W_Oh) << d — because the head's "useful" routing directions are a small subspace — the pair can be replaced by a single low-rank matrix W_VO_h = U_h Σ_h V_h^T computed once offline (28 layers × 16 heads × one matrix multiply of d×d, negligible one-time cost). At inference, two d×d matmuls (each 2048² FMAs) collapse to one rank-r application (2 × d × r FMAs). This is not CF8 MLP rank reduction — MLP matrices are full-rank (CF4, CF5, CF8). W_V and W_O are structurally different: the value path carries context-routed information, and SGD may drive the head's effective rank far below d because each head only needs to selectively extract and project a subset of features. AQFKV (CF11) measured W_Q and W_K spectra; W_V and W_O product rank is a distinct, unmeasured object. Cross-layer W_Q stacking is killed (v2-CHEAP-TEST-001); this is within-layer V-O product, not cross-layer.

**Residency arithmetic (7.28 GiB Ryzen target).**
- Qwen3-1.7B: 28 layers × 16 heads × 2 matrices × 2048² × 2B = 28 × 2 × 8.4 MB = 471 MB. At rank-r=512 per-head product: 28 × 16 × 2 × (2048×512 + 512×2048) × 2B = 28 × 16 × 4 MB = 1792 MB — worse, because 16 per-head matrices > 2 full matrices. Correct framing is global (across heads per layer): stacked W_VO[L] ∈ R^{d×d} = sum over heads of outer product — actually W_O ∈ R^{d×d} acts after attention-weighted V ∈ R^{L×d}. Correct composition: replace W_V (d×d) + W_O (d×d) with W_VO (d×d) at rank r. Storage: W_VO stored as U (d×r) + V^T (r×d) = 2×d×r×2B. At r=512: 2×2048×512×2B = 4.2 MB/layer vs 2×8.4 MB = 16.8 MB/layer → 4× compression per layer. 28-layer total: 117 MB vs 471 MB → saves 354 MB at 1.7B scale.
- 70B analogue (d=8192, 80 layers): W_V + W_O = 80 × 2 × 8192² × 2B = 21.5 GB. At rank-512: 80 × 2 × 8192 × 512 × 2B = 1.34 GB → saves 20.2 GB. Combined with NanoQuant MLP (5.35 GiB): 5.35 + 1.34 = 6.69 GiB → within 7.28 GiB budget. DRAM bandwidth: 6.69 / 11.5 → 0.58 s/token → 1.72 tok/s. At r=256: 0.67 GB attention → 6.02 GiB total → 1.91 tok/s. Quality gate: hinges on W_VO product rank at the chosen r.
- P(end-to-end success): W_V/W_O product rank is unmeasured. Prior: CF11 shows W_Q r_99/d ≈ 0.63; W_K r_99/d ≈ 0.79. If W_V is similar to W_K (value path analogous to key path), product rank may compound. Conservative P(GO at r=512 with ΔNLL < 0.5 nats) ≈ 0.45.

**Novelty gloss vs kill list and published landscape.**
Closest kill-list item: per-head W_Q rank truncation (CF11 per-head NO-GO at K=64). Structural difference: this proposal uses the product W_V W_O as a single object (fold at the composition boundary), not per-factor per-head truncation. Run-005's R5-VOWN proposed this but was not run. No run-005+ experiment has measured this. Closest published: MLA (DeepSeek-V2) trains a low-rank KV projection; this is post-training product folding, no retraining. AERO (arXiv:2410.13060) folds W_gate/W_up around activation removal; V-O fold is the analogous structure in attention. Not in any kill list.

**Smallest experiment.**
Claim: rank_99(W_V[L14] @ W_O[L14]) / d < 0.75 on Qwen3-1.7B-Base bf16.
Test: compute W_VO = W_V[14] @ W_O[14] (2048×2048, ~0.1s), SVD (~30s), measure r_99. Apply rank-{256, 512} truncation, measure ΔNLL on 512-token WikiText-2 held-out.
Runtime: ≈ 20 min.
GO threshold: r_99/d < 0.75 AND ΔNLL < 0.5 nats at rank-512.
NO-GO finding: W_VO is full-rank → V and O matrices span complementary subspaces; no fold is possible; extends CF8 boundary to V-O product; attention compression ceiling is W_Q/W_K only.

**Primary risk.**
W_V and W_O may be spectrally complementary: W_O has to invert whatever W_V selects, keeping the product full-rank. Mitigation: the smallest experiment resolves this in 20 min at negligible cost before any cascade investment.

---

## R8-RAOK-70B — RAOK 3-Tier Activation Codebook Full 70B Cascade

**Track B — compression.**

RAOK was run_001's Track B selection — 3-tier activation scheme grounded in CF3 (K=0.1% Jaccard=0.718 static; K=1% Jaccard=0.308 dynamic). v2-CF1 extends CF3 to all 28 layers (mean Jaccard at K=1% = 0.39, range [0.22, 0.531]), confirming per-token outlier handling is needed across the full stack. The 3-tier design: T1 = top ~2 channels (K=0.1%) pinned FP16 static; T2 = next ~18 channels (0.1%→1%) per-token INT8 dynamic; T3 = remaining 2028 channels INT4 static. AVX2 GEMV processes T3 bulk INT4, scatter-adds T2 INT8, adds T1 FP16 residual. This is not the prior-killed pure LLM.int8() or SmoothQuant: those handle channel-static only; RAOK's load-bearing claim is the K=0.1%/K=1% stratification the CF3 measurement exposes. Cascade rungs: (1) measure T2 per-token Jaccard at K=18 channels across all 28 layers using v2-CF1 data → go/no-go on whether T2 is genuinely dynamic (Jaccard < 0.35); (2) implement AVX2 3-tier GEMV kernel for W_up application; (3) measure ΔNLL + tok/s on Qwen3-1.7B extrapolated to 70B residency arithmetic.

**Residency arithmetic.**
- T3 (99% weights) INT4 = 0.5 bpw contribution: 70B × 0.99 × 0.5 / 8 = 4.34 GB.
- T2 (1% of weights) INT8: 70B × 0.01 × 1 / 8 = 87.5 MB.
- T1 (0.1%) FP16: 17.5 MB.
- T2 codebook: 28 layers × 28672 (W_up intermediate, 70B scale) × 18 channels × 1 byte = 14.5 MB.
- Total ≈ 4.45 GB → DRAM-resident within 7.28 GiB. Effective bpw ≈ 0.51 bpw.
- DRAM bandwidth: 4.45 / 11.5 → 0.387 s/token → **2.58 tok/s** — above 3 tok/s target with ~25% headroom gap; headroom closes if T2 overhead < 5% of total GEMV compute (18 INT8 scatter vs 2028 INT4 bulk — ratio is 0.9%).
- Quality: T3 INT4 is the dominant loss. T2 INT8 patching the 18 most dynamic channels reduces quantization error on the most volatile activations; expected ΔNLL improvement over pure INT4 ≈ 0.10–0.15 nats (estimate; measurement needed). Total vs bf16: approximately INT4 ΔNLL baseline minus T2 correction.
- P(end-to-end): CF3 is confirmed; v2-CF1 extends it. P(T2 Jaccard GO) ≈ 0.85 (CF3 K=1% Jaccard=0.308 already confirms dynamics). P(AVX2 kernel overhead < 5%) ≈ 0.80. P(combined quality target < 0.3 nats over INT4 baseline) ≈ 0.65. P(end-to-end) ≈ 0.44.

**Novelty gloss.**
Closest kill-list: PDAP (run_001 selection, confirmed CF3). Structural difference: RAOK-70B escalates to full 70B deployment arithmetic and an explicit AVX2 kernel design; PDAP only measured the Jaccard structure. Closest published: SmoothQuant (channel-static), LLM.int8() (static top-K), PrefixQuant (token-type stratification). RAOK differs by exploiting the K=0.1%/K=1% split that CF3 measured — specifically the threshold between static and dynamic regimes — which no published method uses as a design primitive.

**Smallest experiment.**
Claim: mean consecutive-pair Jaccard at K=18 channels (0.879% of 2048) across all 28 Qwen3-1.7B layers is < 0.40 (confirming T2 is genuinely dynamic and not static-pinnable).
Test: re-run PDAP Jaccard measurement (same code, 200-prompt corpus, Qwen3-1.7B-Base) at K=18 channels with v2-CF1 layer-sweep extension. Runtime: ≈ 30–40 min.
GO threshold: mean Jaccard < 0.40 across all 28 layers; per-channel outlier frequency ≥ 5 events/channel across 200 prompts (codebook viable).
NO-GO finding: Jaccard > 0.40 at K=18 → T2 channels are stable enough to pin statically; RAOK reduces to 2-tier (T1 static + T3 INT4); comparable to LLM.int8() without the dynamic overhead benefit.

**Primary risk.**
AVX2 scatter-add for T2 dynamic channels stalls the pipeline on out-of-order memory accesses. Mitigation: T2 is 18 channels; at 2048 INT4 bulk followed by 18 INT8 scatter, the overhead is a rounding error (18/2028 ≈ 0.9%). Profile with perf counters on Ryzen 5 7530U.

---

## R8-WVOS-SPEC — W_V/W_O Spectrum Sweep + Attention Weight Class Characterization

**Track B — compression.**

AQFKV (CF11) measured W_Q (r_99/d = 0.63) and W_K (r_99/d ≈ 0.79, computed from K=512 GO at ΔNLL=+0.29 nats). W_V and W_O are explicitly flagged in SUMMARY.md as "untested — cheap parallel measurement." This proposal is the Reach-framed cascade that closes the attention-weight characterization: measure W_V and W_O spectra at the same surgical cost as AQFKV (≈15–20 min each), then compose the four-matrix attention compression cascade. The Reach claim: if W_V and W_O have r_99/d ≤ 0.70 (similar to or better than W_Q), the four-matrix attention block can be compressed to 3–4× globally, saving 5–7 GB of attention weights on a 70B model. Combined with RAOK-70B's MLP activation scheme (not MLP weight compression — that's CF8-dead), the total 70B residency hits 5–6 GB. The structural finding on NO-GO (W_V/W_O full-rank) is equally valuable: it closes the attention weight class, sharpens CF8 boundary to the specific sub-class, and directs compression effort exclusively to W_Q/W_K.

**Residency arithmetic.**
- Scenario A (W_V r_99/d ≈ 0.63, W_O r_99/d ≈ 0.63): compression ratio at K=512 ≈ 2× each. 70B W_V + W_O ≈ 10.7 GB at bf16 (8192×8192×80×2×2B) → 5.35 GB at K=512. Combined with W_Q at K=512 (3.86 GB → 1.93 GB) and W_K at K=512 (0.96 GB → 0.48 GB): total attention ≈ 7.76 GB. With MLP at NanoQuant (5.35 GiB) the total doesn't fit — attention + MLP = 12.1 GB. But at K=128 for W_Q (CF11 GO): W_Q → 0.48 GB; W_K → 0.12 GB; and if W_V/W_O at K=256 (more aggressive): W_V + W_O → 2.67 GB. Total attention ≈ 3.3 GB. Total with MLP NanoQuant ≈ 8.65 GiB — still over budget by 1.4 GiB. Add RAOK 3-tier MLP activation quantization (bpw 0.51 for MLP activations reducing effective MLP bandwidth): 5.35 GiB effective → total ~8.65 GiB still over. This is the honest arithmetic; the cascade is tight and needs W_V/W_O to compress more aggressively than W_K.
- Scenario B (W_V/W_O full-rank, r_99/d ≈ 1.0): no V/O compression. Total attention = W_Q (K=128, 0.48 GB) + W_K (K=512, 0.48 GB) + W_V + W_O (bf16, 10.7 GB) + embeddings. Path to 70B DRAM-residency narrows to MLP activation quantization alone: not enough.
- P(Scenario A): guided by CF11 analogy, estimated P ≈ 0.40. P(Scenario B): 0.35. P(intermediate): 0.25.

**Novelty gloss.**
No kill-list item covers W_V/W_O spectra. CF11 explicitly notes them as untested. Run-005's R5-AWQKV proposed the four-matrix cascade; run-005's R5-VOWN proposed V-O product folding. Neither ran as the rung-1 cascade measurement. This is the direct measurement proposal that serves as the mandatory prerequisite for every attention-compression Reach cascade. Closest published: A3 (arXiv:2505.12942) addresses W_Q/W_K; ignores W_V/W_O. Post-training W_V/W_O spectrum characterization is not in the published landscape.

**Smallest experiment.**
Claim: r_99(W_V[L14]) / d < 0.85 or r_99(W_O[L14]) / d < 0.85 on Qwen3-1.7B-Base bf16.
Test: SVD of W_V[14] and W_O[14] (each 2048×2048 bf16 → float32 for numerical stability), compute r_99 for each. Apply rank-{128, 256, 512, 1024} truncation to each independently, measure ΔNLL on 512-token WikiText-2 held-out with all other weights at bf16.
Runtime: ≈ 25 min (two matrices, same cost as single AQFKV sweep).
GO threshold: r_99/d < 0.80 for either W_V or W_O AND ΔNLL < 1.0 nats at rank-512.
NO-GO finding: both W_V and W_O full-rank (r_99/d ≥ 0.95) → CF8 extends to the value-output chain; attention compression is capped at W_Q + W_K; the four-matrix cascade is infeasible; direct the Reach effort to activation quantization (RAOK) and NVMe offload scheduling.

**Primary risk.**
W_O is the "output projection" and may behave more like W_down (MLP output projection, full-rank per CF8 expectation) than like W_Q. Mitigation: W_O is linear (no nonlinearity coupling it to a partner matrix the way W_up couples to silu(W_gate·x)), so the structural argument for full-rankness is weaker than for MLP weights; the cheap test resolves it.

---

## R8-GUCB-SHARED — Cross-Tensor Codebook Sharing for Structurally Aligned Matrices [FREE SWING]

**Track B — compression.**

Consider the weight matrices that operate on the same linear subspace during inference: in each transformer layer, W_Q, W_K, W_V, and the residual input embedding (h_L) all use the same d_model=2048 input vector. Their rows are vectors in R^{d_model}. If the *row distributions* of W_Q, W_K, W_V are geometrically similar (occupy similar angular regions of R^{d_model}), then a single shared codebook C ∈ R^{K×d_model} can quantize all three matrices simultaneously with fewer bits per weight than independent per-matrix codebooks. This is the "within-layer cross-tensor codebook sharing" idea (deferred in Track B R5 as GUCB). The Reach framing: if row-space similarity between W_Q, W_K, W_V is ≥ 0.5 (measured as mean cosine similarity between nearest-codebook-centers of row-samples from each matrix), a single K=256-entry codebook shared across all three attention input matrices saves 1 codebook table per layer (256 × 2048 × 2B = 1 MB saved × 28 layers = 28 MB — tiny) but more importantly reduces codebook overhead as a fraction of total storage, enabling effective bits-per-weight = log2(K)/d_model × codebook_bits ≈ 0.008 bpw codebook overhead instead of 3× that for independent codebooks. At 4-bit weight codes: effective storage is dominated by the codes (0.5 bpw), not the codebook. The structural claim is that the shared codebook code-book entries match the row distribution of ALL three matrices more accurately than random, measured by mean quantization error (Frobenius norm of reconstruction) compared to three independent codebooks of the same total size.

The stronger form: if W_Q and W_K rows are drawn from distributions with substantial overlap (because both operate on the same residual stream and are optimized to produce compatible attention scores), a shared codebook trained on the union of rows might achieve the same reconstruction error as per-matrix codebooks with 1/3 the total codebook storage. The k-means on the union of W_Q + W_K + W_V rows vs k-means on each separately is a ≤5-minute measurement.

**Residency arithmetic.**
- At K=256 shared codebook: 3 matrices × 2048 × d_model × 0.5 bpw + 1 × 256 × 2048 × 2B (shared) vs 3 × 256 × 2048 × 2B (independent). Codebook saving: 2 × 256 × 2048 × 2B = 2 MB per layer × 28 layers = 56 MB. Trivial. The real gain is at larger codebooks (K=65536, 16-bit codes): 3 matrices saved 2 tables × 65536 × 2048 × 2B = 512 MB per layer × 28 = 14.3 GB — but 16-bit codes mean 2 bpw, not 0.5 bpw; above NanoQuant residency floor. The residency argument is weak in isolation; the value is in enabling shared training of a better-fitting codebook with fixed total budget. At 70B scale: if shared codebook achieves 0.50 bpw vs independent 0.55 bpw (because the shared fit on the union is tighter per entry), the saving is 70B × 0.05 / 8 = 437.5 MB — not decisive but meaningful as a quality-per-bit improvement.
- The real Reach value: GUCB as a cascade rung ON TOP OF RAOK-70B. If activation statistics (CF3 CF-grounded) align with attention weight row distributions (unmeasured), a shared codebook for activations AND weights enables an end-to-end quantization scheme with fewer codebook tables. P(alignment is real): unknown; this is the free-swing bet.

**Novelty gloss.**
No kill-list item covers cross-tensor codebook sharing for attention input matrices. GUCB was deferred in Track B R5 without a kill. Closest published: VPTQ (vector post-training quantization) uses per-layer codebooks independently. AQLM (additive quantization) uses residual codebooks per matrix independently. No published method trains a shared codebook across W_Q, W_K, W_V using their geometric alignment on the input subspace as a structural motivation. The structural claim (three matrices operating on the same input distribution may have aligned row distributions) is testable in minutes and has no equivalent in the kill list.

**Smallest experiment.**
Claim: the mean reconstruction error of a K=256 shared codebook trained on the union of W_Q[14], W_K[14], W_V[14] rows is within 1.05× the mean reconstruction error of three independent K=256 codebooks (each trained on one matrix separately).
Test: k-means clustering (scikit-learn KMeans, K=256, ~10 iterations) on (a) union of 3 × 2048 = 6144 row vectors of length 2048, (b) each matrix separately; compute per-row reconstruction error (squared L2 to nearest centroid) for each matrix under both schemes; compare ratios.
Runtime: ≈ 5–10 min on Ryzen (k-means on 6144 × 2048 float32 vectors).
GO threshold: shared codebook reconstruction error within 1.05× of independent codebooks for all three matrices.
NO-GO finding: shared codebook has >5% worse reconstruction → W_Q, W_K, W_V row distributions are sufficiently distinct that independent codebooks are necessary; structural independence between attention input matrices confirmed.

**Primary risk.**
W_Q and W_K rows may be geometrically distinct (W_Q projects queries, W_K projects keys — they may lie in different angular regions of R^{d_model} despite operating on the same input). Mitigation: the 10-minute k-means experiment directly answers this; no cascade investment precedes it.

---

## R8-DELTAROPE — Per-Layer W_K RoPE-Precomputed Absorption Cascade

**Track A — arch-transposition.**

RoPE (Rotary Position Embedding) applies a rotation R_θ(pos) to query and key vectors at inference time: Q_rot = Q R_θ(pos), K_rot = K R_θ(pos). The rotation is position-dependent, applied per-token. However, the weight matrix W_K produces the unrotated keys K = X W_K; the rotation is applied after. CF11 established W_K r_99/d ≈ 0.79 at global SVD (not per-head per the killed branch). Here is the arch-transposition: for the "static" positions that appear very frequently in the context (position 0 = BOS, positions 1–N for short contexts where position is essentially fixed), pre-absorb the expected rotation into a modified W_K^(pos) = W_K R_θ(pos). This is only exact for fixed positions; the gain is that position 0–31 (the first 32 positions, covering most short-context inference) can be served with pre-rotated W_K matrices, eliminating the runtime rotation step for those positions. This is NOT the killed "softmax × RoPE shift-invariance" — softmax shift-invariance is about the score matrix; this is about absorbing RoPE into the weight matrix for specific known positions. The kill was F3-SRSC (run_003) which applied the softmax shift-invariance argument in combination with RoPE — this proposal does not use softmax shift-invariance at all.

The mechanism: store position-specialized W_K matrices for positions 0–31 (32 × d × d_kv × 2B per layer = 32 × 2048 × 256 × 2B × 28 = 755 MB — too expensive). Alternative form: for a fixed-position serving mode (batch processing where the same prompt prefix is reused many times), precompute K_rot for the fixed prefix once and cache it persistently. This is standard KV caching; the arch-transposition is to go further: compress the stored KV by absorbing RoPE into the stored value. The per-position rotation is a fixed unitary matrix; if W_K has r_99 rank ≈ 0.79 × d, then K_rot = X W_K R_θ(pos) has rank ≤ rank(W_K) regardless of rotation (unitary does not change rank). The actual claim: post-rotation K_rot per layer can be stored at the same compressed rank as W_K without additional quality loss from the rotation, because R_θ is unitary and preserves singular values. This means KV cache compression at rank r can be applied to the rotated keys directly (matching CF11's W_K finding) without any rotation-induced rank inflation.

**Residency arithmetic.**
- KV cache at 4K context, 28 layers, Qwen3-1.7B: 4096 × 28 × 2 (K + V) × (8 kv-heads × 128 dim) × 2B = 4096 × 28 × 2 × 1024 × 2B = 469 MB. At CF11 W_K GO rank-512 compression (2× KV cache per token): 235 MB. Savings: 234 MB at 4K — marginal.
- 70B at 32K context: 32K × 80 × 2 × (8 × 128) × 2B = 32K × 80 × 2048 × 2B = 10.7 GB KV cache. At 2× compression: 5.35 GB → combined with weights (5.35 GiB NanoQuant MLP) = 10.7 GB — still over budget. At 4× compression (rank-256): 2.67 GB + 5.35 GiB = 8 GiB — within 7.28 GiB with tight margin. The cascade: rung 1 = confirm RoPE preserves rank (unitary argument, trivially true mathematically, but verify ΔNLL is not inflated by RoPE phase misalignment during truncation); rung 2 = validate compressed KV quality at 32K context.

**Novelty gloss.**
Kill list does not contain RoPE-invariant KV compression. Run-003 killed F3-SRSC (softmax + RoPE shift-invariance). This proposal does NOT use softmax shift-invariance; it uses the unitary nature of RoPE to argue that CF11 W_K rank compression applies equally post-rotation. Closest published: MLA (DeepSeek-V2) stores compressed KV; RoPE creates complications there (RoPE-decoupled MLA is a known design challenge). Post-training RoPE-aware KV compression exploiting rank-preservation of unitary transforms is not in the kill list and not in published prior. The structural argument (unitary ⇒ singular-value-preserving ⇒ rank-r truncation quality is rotation-invariant) is a mathematical identity, not a learned property.

**Smallest experiment.**
Claim: ΔNLL at W_K rank-512 compression applied to rotated keys (K_rot = X W_K R_θ(pos)) equals ΔNLL at W_K rank-512 compression applied to unrotated keys (K = X W_K), within ±0.05 nats.
Test: extend AQFKV experiment to apply rank-512 truncation in two modes: (a) truncate W_K before rotation (baseline, already measured at +0.29 nats), (b) truncate post-rotation key vectors. Measure ΔNLL difference on 512-token WikiText-2.
Runtime: ≈ 15–20 min (instrumented forward pass, Qwen3-1.7B-Base).
GO threshold: |ΔNLL(post-rotation truncation) - ΔNLL(pre-rotation truncation)| < 0.05 nats.
NO-GO finding: post-rotation truncation is significantly worse → RoPE phase structure prevents rank compression of rotated keys; KV cache compression must use different coordinates (e.g., store unrotated keys, apply RoPE at score computation — which is the standard approach anyway); the unitary invariance argument fails for the truncation operation due to quantization grid misalignment.

**Primary risk.**
The unitary argument is mathematically valid for continuous SVD truncation, but quantization on a post-rotation basis may interact badly with the periodic structure of RoPE phases (the basis is no longer axis-aligned). Mitigation: the 20-min smallest experiment tests this directly; if NO-GO, the fallback is standard KV compression on unrotated keys (which the AQFKV measurement already supports at ΔNLL=+0.29 for W_K at K=512).

---

## Convergence handles

- W_V and W_O spectra (multiple ideas depend on whether r_99/d < 0.80 — the cheapest unresolved measurement in the pipeline)
- W_V W_O product rank (distinct from individual spectra; if product compresses more than components, the fold is strictly better than two-matrix truncation)
- RoPE unitarity and SVD truncation interaction (whether continuous-domain rank preservation survives discrete truncation in the rotated basis)
- K=18-channel (0.9%) per-token Jaccard across all 28 layers (RAOK-70B tier-2 gate; v2-CF1 extended CF3 to all 28 layers but didn't specifically measure K=18)
- Cross-tensor row-distribution alignment (W_Q / W_K / W_V row geometric overlap as shared codebook feasibility signal)
