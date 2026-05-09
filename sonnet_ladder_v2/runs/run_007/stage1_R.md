# Stage 1 — Reach (R) — Run 7

Orientation: R — Reach. Independent ideator.

Kill constraints active: v2-CHEAP-TEST-001 (cross-layer W_Q subspace sharing), v2-S3-R004-001 (arbitrary rotation breaking RMSNorm). Hard kills from prompt: cross-layer W_Q, arbitrary rotation breaking RMSNorm, within-layer Q/K joint (PALU+TransMLA), softmax shift-inv × RoPE.

---

## R7-R-001 — Attention-Weight Tiered Residency Cascade (ATRC)

**Tags:** R, Track B

### Mechanism

Track B. CF11 gives W_Q global K=128 at ΔNLL=+0.98 (8× compression) and W_K global K=256 at ΔNLL<0.5. W_V and W_O spectra are untested but the CF11 pattern predicts they are also more concentrated than MLP matrices. The cascade: (1) sweep W_V and W_O spectra on Qwen3-1.7B; (2) if W_V/W_O show r_99/d < 0.80 (as W_K does at 0.79), apply global truncation to all four attention matrices; (3) quantize surviving attention directions to INT8 (not INT4 — CF11 shows the subspace is load-bearing at the boundary, so guard quality there); (4) MLP matrices stay at INT4 baseline (DRAM-resident case) or NVMe-paged at Q4_0. The multiplicative attention-weight saving is: W_Q saves 8×, W_K saves 4×, W_V/W_O save ~2–4× (predicted pending measurement). Across a 70B-class model where attention weights are ~33% of weight bytes, 4–8× attention compression × 4 bpw MLP = total residency of ~3.1–4.2 GiB. That is DRAM-resident 70B at operational quality. No retraining; no cross-layer rotation. Not CF8-violating (MLP untouched). Not v2-CHEAP-TEST-001 (no stacked W_Q across layers). Relies on CF11 as anchor for W_Q and W_K; W_V/W_O is the gap rung.

### Residency arithmetic

Qwen3-72B approximate weight distribution: 72B params × 2 bytes = 144 GB bf16. At INT4 (4 bpw) MLP, attention at 4 bpw baseline: ~36 GB. Target: 7.28 GiB = 58 GB/36 GB is already 1.6×, so we need more compression. But: at 2 bpw MLP (aggressive, boundary of quality cliff) + 1 bpw attention (8× over INT4) on attention weight fraction: attention is ~30% of 70B params ≈ 21B params. INT4 attention = 10.5 GB → 8× SVD truncation → ~1.3 GB. MLP 51B params at 2.25 bpw (NanoQuant territory) = ~14.3 GB. Total ≈ 15.6 GB. Still above 7.28 GiB. Revised: pair with NVMe offload of deep layers (CF2: residual stream near-constant cosine, so early-exit layers hit more). Let MLP layers 0–12 reside in DRAM at INT4 (5.1 GB), layers 13–27 on NVMe. Attention at global K=128 across all 28 layers ≈ 0.7 GB (K=128 × d_head × n_heads × 4 matrices × 28 layers × INT8). First cascade rung (W_V/W_O spectrum) settles whether 4× attention is achievable. DRAM-only scenario requires 70B class to have untied lm_head (e.g., Llama-3-70B) + NVMe partial offload. P(end-to-end) = P(W_V spectrum concentrated, r_99/d < 0.80) × P(W_O same) × P(joint attention truncation quality ≤0.5 nats) × P(quantization survives truncation) ≈ 0.6 × 0.6 × 0.7 × 0.8 = 0.20.

### Novelty gloss vs kill list and published landscape

Closest kill list item: AQFKV (R3-A selected, measured W_Q and W_K, deferred W_V/W_O as "cheap follow-up"). ATRC is the residency-cascade build-out of AQFKV's deferred measurement, not a repeat. Closest published: MLA (DeepSeek-V2) trains joint Q-K-V low-rank from scratch. ATRC applies post-training SVD truncation independently per matrix class, which is structurally distinct — no joint projection constraint, no retraining, each matrix truncated to its natural spectrum. A3 (arXiv:2505.12942) uses attention-logit-error minimization — same goal but different criterion; AQFKV showed Frobenius + spectrum is consistent with A3 conclusion for W_Q. ATRC is the first explicit W_V/W_O residency cascade grounded in v2 CF11 measurements.

### Smallest experiment

Claim: W_V global SVD truncation at K=256 gives ΔNLL ≤ 0.50 nats on Qwen3-1.7B-Base (analogous to W_K at K=256, ΔNLL=+0.82). Measure W_V spectrum (r_99, var@256) and compute ΔNLL at K ∈ {512, 256, 128}. Runtime: 30–40 min (same shape as AQFKV W_K measurement). Go: ΔNLL < 0.50 at K=256, confirming W_V is in the same spectrum class as W_K. No-go: W_V is MLP-like (r_99 ≈ d) — class-kills W_V truncation, sharpens CF8/CF11 boundary, produces structural finding regardless.

### Primary risk

W_V and W_O have r_99/d ≈ 1.0 (MLP-class, not attention-class), collapsing the cascade's attention-side saving from 4–8× to 1× and leaving residency target unmet. Mitigation: run the W_V/W_O spectrum sweep first (rung 1); if flat, recast as W_Q+W_K-only cascade (still 15–20% attention residency saving, less dramatic but not zero).

---

## R7-R-002 — Cross-Layer Outlier-Highway Tiered Store (CLOTS)

**Tags:** R, Track B

### Mechanism

Track B. v2-CF1 shows per-token outlier-channel-set dynamicity (K=1% Jaccard=0.39 mean, range 0.22–0.531 across all 28 layers). The K=0.1% tier is channel-static (Jaccard=0.718 from CF3). This gives a three-tier residency architecture whose topology is data-driven: (a) top ~2 channels per layer (0.1% of d=2048, K=0.1%) are **static outliers** — they can be stored separately at FP16 and are never swapped; (b) the next ~18 channels (K=1% minus K=0.1%) are **token-dynamic** and must be in DRAM at inference time; (c) the bulk 2028 channels are INT4 and can tolerate NVMe paging. The cascade key: CF1 cross-layer generalization means this partition applies to ALL 28 layers, not just sampled ones. For a 70B-class model with d_model=8192 (Llama-3-70B): static outliers = 8 channels/layer × 28 layers × weight bytes touched per channel ≈ tiny (<1 MB of storage routing, saving ~8 GB DRAM by not locking full layers). The real gain is DRAM-hot set management: keep only the token-dynamic tier (~1% per layer = 20 channels/layer × 28 layers × 8192 × INT8) in DRAM; remainder NVMe-paged with prefetch. This is a pure storage-tier design exploiting CF1 + CF3 as the cost model. No retraining; no structural approximation of weights themselves — the bytes are not changed, just routed.

### Residency arithmetic

Llama-3-70B: 70B params × 4 bpw baseline = 35 GB. NVMe paging at 3 GB/s with sequential reads: 35 GB / 3 GB/s = 11.7 s/token — too slow. But if DRAM holds the hot tier (K=1% channels per layer for all MLPs): MLP hot set = 1% × 70B MLP params ≈ 0.7B params × 2 bytes (FP16) = 1.4 GB. Attention matrices: CF11 global W_Q K=128 at 8× compression; pack into DRAM at INT8 ≈ 0.9 GB (rough). KV cache at 4K context = 0.4 GB. Total DRAM: ~3.0 GB hot tier + 0.4 GB KV. Leaves 3.8 GB DRAM budget for weight residency buffer. Deep MLP weights NVMe-paged at ~3 GB/s with stride-sequential access. Effective token/s depends on prefetch latency (CF13* — not confirmed; cascade rung 1 must measure). With double-buffering and static access order, theoretical: DRAM-bound at 11.5 GB/s on the hot tier → ~4 tok/s on hot subset + NVMe-limited on cold pages. Prefetch overlap quality is the bottleneck; cascade rung 2 measures it.

### Novelty gloss vs kill list and published landscape

Closest kill list: PDAP (R2 selected, advanced) tests activation-side outlier pinning directly on weight access patterns — that is the mechanism's motivating experiment. CLOTS extends PDAP's per-token outlier finding into a weight-residency tier design, which PDAP did not propose. PDAP was about measuring the outlier structure; CLOTS is about exploiting the measured structure for weight-storage tiering. Closest published: LLM.int8() uses a two-tier mixed-precision scheme but with channel-static partitioning only (Jaccard=0.718 is the LLM.int8() regime, not the K=1% operational regime). Apple LLM-in-a-flash (arXiv:2312.11514) pages full rows; CLOTS pages by channel tier, which is orthogonal to row-granularity. The channel-tier partition derived from empirical Jaccard measurements is not published.

### Smallest experiment

Claim: for Qwen3-1.7B-Base, the K=0.1% static outlier channels identified at calibration time are stable enough across an independent held-out set that a channel-static mask trained on 200 tokens achieves Jaccard ≥ 0.70 on a 500-token held-out set (replicating CF3 at K=0.1% on an independent split). Runtime: 20 min (activation collection + Jaccard computation on held-out split). Go: Jaccard ≥ 0.70 — static mask is load-bearing. No-go: Jaccard < 0.50 — channel-static tier is not reusable across contexts, CLOTS collapses to per-token handling everywhere (kills the storage-tier optimization).

### Primary risk

The static-mask reproducibility doesn't hold across domains (channel identity shifts between code, prose, math contexts), making the static tier useless. Mitigation: measure Jaccard stratified by domain; if it fails globally but holds within-domain, use domain-conditioned masks (higher per-domain cost but still cheaper than pure per-token handling).

---

## R7-R-003 — W_V / W_O Joint Folded Residency (VJOFR)

**Tags:** R, Track A

### Mechanism

Track A. CF11 shows W_Q is the most concentrated attention matrix (r_99/d ≈ 0.63); W_K is intermediate (r_99/d ≈ 0.79); W_V and W_O are unmeasured. Separately: the attention output computation is `W_O · (softmax(W_Q·x · (W_K·x)^T / √d) · W_V·x)`. W_V and W_O appear sequentially: `W_O · Attn_weights · W_V`. If W_O and W_V are left-composable without quality loss — specifically if W_O's row space and W_V's column space share significant structure (alignment angle < threshold) — then W_O · W_V is a single fused linear operator at inference. The fused W_{OV} = W_O W_V is d_model × d_head (per head), which is smaller than W_V alone when the attention output is contracted with the head before the output projection. This is not rank truncation of either matrix; it is a change to the computation graph: instead of `W_O · (Attn · W_V · x)`, compute `(W_O · W_V) · x` then apply the attention weights — which is the same algebraic result when the attention weights are row-independent of x's path through W_V. This rearrangement saves one matrix-multiply per layer at the cost of one pre-computed product. The real residency gain: store W_{OV} = W_O W_V at INT4 (d_head × d_model × 4 bpw) instead of W_O (d_model × d_model) + W_V (d_head × d_model). For d_model=2048, d_head=128, n_heads=16: W_V = 16 × 2048 × 128 = 4M params; W_O = 2048 × 2048 = 4M params. W_{OV} = 16 × 128 × 2048 = 4M params — same total, but the compute path changes. The gain is compute, not storage. UNLESS W_{OV} is lower-rank than either factor: if rank(W_O W_V) << min(rank(W_O), rank(W_V)), then the product is compressible below the factor storage. This is a falsifiable claim. The cascade: measure rank(W_{OV}) at each layer; if r_99(W_{OV}) << d, truncate the fused product. CF11 motivates this: head-shared W_Q structure suggests head-space coherence; the same mechanism might produce low-rank W_{OV}.

### Residency arithmetic

If r_99(W_{OV}) ~ 256 per layer (hypothesis): W_{OV} stored at rank 256 × INT8 = 2048 × 256 × 1 byte × 28 layers = 14.7 MB per head pair — negligible. But for all heads: 16 heads × 14.7 MB = 235 MB for all 28 layers, replacing 4M + 4M = 8M params × 2 bytes = 16 MB per layer × 28 = 448 MB. Saving: 448 → 235 MB, 1.9× for attention output+value. Modest in isolation. At 70B scale: W_V + W_O ≈ 2 × 8192 × 8192 × 28 × 2 bytes = 7.6 GB → at rank-512 truncation: 7.6 → 1.5 GB (5× saving on V+O weight fraction ≈ 20% of total). Combined with CF11 W_Q/W_K savings: 70B attention weights (≈30% of total) compressed 4–5× → total residency reduction on attention portion ≈ 7 GB saved, target DRAM residency gets closer. P(W_{OV} low-rank): moderate — product of two full-rank matrices is generally full-rank. This is the speculative rung; the spectrum measurement falsifies it quickly.

### Novelty gloss vs kill list and published landscape

No kill list item covers W_{OV} fused product spectrum measurement. Closest published: Transformer analysis papers (Elhage et al. circuits work) discuss W_{OV} as a conceptual circuit primitive, but never as a post-training compression target for its own rank structure. AERO (arXiv:2410.13060) folds W_gate/W_up after activation removal — that is the structural archetype; VJOFR applies the same "fuse two adjacent matrices if their product has lower rank than either factor" logic to W_O and W_V in attention. This is an unexplored extension of AERO's algebraic-fold idea to the attention path. If W_{OV} is full-rank, the idea dies, but the measurement is itself a structural finding extending CF11.

### Smallest experiment

Claim: for at least 5 of 28 layers in Qwen3-1.7B-Base, r_99(W_O · W_V) < 0.7 × min(r_99(W_O), r_99(W_V)) — i.e., the product is strictly lower-rank than either factor. Compute W_{OV} = W_O · W_V per layer (torch.linalg.svd on 2048 × 2048 matrix, per head; per-head d_head × d_model product). Runtime: 15 min. Go: at least 5 layers show r_99 < 0.70 × factor rank (product is compressible). No-go: all layers show r_99(W_{OV}) ≈ r_99(W_V) — product rank equals factor rank, no fold gain; extends CF8 class boundary further.

### Primary risk

W_{OV} is full-rank (algebraically, a product of two matrices in general position is full-rank up to the minimum dimension). The algebraic fold gives compute savings but zero storage savings if the product is not low-rank. Mitigation: the compute savings (one fewer matmul per layer) are real regardless; reframe the proposal as a compute-graph transposition even if the storage claim fails.

---

## R7-R-004 — Attention Subspace × Outlier Channel Joint Routing (ASOR)

**Tags:** R, Track A

### Mechanism

Track A. Two independent v2 findings pointing at the same inference-time quantity: (1) CF11 — W_Q head-shared subspace at K=128 spans d=128 directions across all 16 heads; (2) v2-CF1 — the K=0.1% static outlier channels (2 channels at d=2048) are stable across tokens. The hypothesis: the K=128 W_Q shared subspace directions and the K=0.1% static outlier channels are NOT independent. Specifically, claim: the static outlier channels (top ~2 by activation magnitude) are over-represented in the rowspan of the K=128 W_Q truncated basis. If true, this means: (a) query computation disproportionately steers attention toward tokens with high static-outlier activation — the outlier channels are routing channels for attention; (b) the W_Q truncation at K=128 implicitly preserves the outlier-channel routing signal; (c) a cheaper approximation is possible: store W_Q at full rank for the ~2 static outlier dimensions and at compressed rank-128 for the remaining 2046 dimensions. This is a mixed-precision W_Q scheme grounded in a structural interaction claim. The new residency math: W_Q per layer = (2 × d_head × n_heads × FP16) + ((d_model - 2) × K × INT8) = (2 × 128 × 16 × 2) + (2046 × 128 × 1) ≈ 8KB + 261KB per layer, vs 512KB per layer at K=128 INT8 baseline. Net gain over straight K=128: modest in absolute bytes (the outlier dimensions are tiny). The structural claim is the load-bearing piece: if the interaction holds, it motivates outlier-aware mixed-precision attention projection as a principled mechanism rather than a heuristic.

### Residency arithmetic

At 70B scale: W_Q per layer baseline (K=128, INT8): 8192 × 128 × 28 = 29.4 MB. With outlier separation: (8 FP16 dims × 128 heads) + (8184 × 128 × INT8) × 28 layers ≈ same order. Residency gain from this idea alone is negligible. The value is in the interaction-claim as a structural finding that enables composition downstream: if outlier channels ARE the attention routing axis, then the CLOTS outlier-tier management (R7-R-002) can be coupled to W_Q routing decisions, creating a joint outlier-handling scheme across activations AND weight projections. P(interaction claim holds): 0.4. P(composition produces further gain): 0.5. P(end-to-end): 0.2. This is primarily a structural-finding cascade rung, not a direct residency win.

### Novelty gloss vs kill list and published landscape

No kill list item covers attention subspace × activation outlier channel interaction. Closest published: SmoothQuant handles activation outliers via scale migration; it does not examine outlier channels' relationship to W_Q subspace structure. The claim that static outlier channels are over-represented in the W_Q rowspan is a new falsifiable hypothesis that either (a) supports a principled mixed-precision scheme or (b) refutes the hypothesis and shows outlier channels and W_Q subspace are independent, which is also informative for RAOK (Track B R6 deferred). Not adjacent to v2-CHEAP-TEST-001 (no cross-layer stacking).

### Smallest experiment

Claim: the top-2 static outlier channels (K=0.1% channels by abs activation magnitude, averaged over 200 calibration tokens) have projection magnitude onto the K=128 W_Q global basis > 2× the median channel projection. Compute: for each layer, (1) identify top-2 static outlier channels from calibration; (2) compute cos(e_outlier, U_Q) where U_Q is the K=128 right-singular vectors of W_Q global; (3) compare to median over all 2048 channels. Runtime: 15 min. Go: outlier channels at least 2× over-represented in W_Q subspace (p < 0.05 permutation test). No-go: outlier channels are uniformly distributed in W_Q subspace — interaction claim false, but confirms independence of the two findings (also a structural finding, kills the routing hypothesis).

### Primary risk

The static outlier channels (K=0.1%, Jaccard=0.718) may be structural artifacts of the tokenizer or embedding table rather than routing signals, in which case their projection onto W_Q subspace tells us about embedding geometry, not attention routing. Mitigation: stratify by token type (BOS, content, delimiter) and measure interaction per stratum; if BOS tokens drive the outlier-W_Q alignment, the routing claim is confounded.

---

## R7-R-005 — Quantization-Aware W_Q/W_K Cascade for 70B DRAM Residency (QAWKC) [FREE SWING]

**Tags:** R, Track B

### Mechanism

Track B. [FREE SWING] The CF11 findings establish W_Q at 8× compression (K=128, ΔNLL=+0.98) and W_K at 4× (K=256, ΔNLL<0.5) for 1.7B class. The question for 70B-class (Llama-3-70B, untied lm_head, d_model=8192): does the head-redundancy pattern (16 heads collapsing to ~128-dim subspace) scale to larger heads? At 70B, n_heads=64, d_head=128, d_model=8192. The K=128 finding in 1.7B captured all-head subspace in one head's worth of capacity (16 heads, K=128 = 1 head). In 70B, n_heads=64 — does the subspace scale linearly (K=512 still gives 8×?) or does redundancy per head stay roughly constant (K=512 for 64 heads still 8×)? If the K/n_heads ratio is the invariant (~8 dims/head), then 70B W_Q at K = 64×8 = 512 dims gives the same 8× compression as 1.7B K=128. This is a scaling hypothesis, not a confirmed finding. The cascade: measure W_Q spectrum and ΔNLL at K=512 on Llama-3-70B (or a surrogate larger model if 70B is RAM-infeasible on the Ryzen — Qwen3-4B has 32 heads, d_model=2560). If the K/n_heads ≈ 8 invariant holds, 70B W_Q is compressible 16× (64 heads × 128 dims / (64×8) dims = 128-dim subspace). Residency: 70B attention weights at 16× W_Q + 4× W_K + assumed 4× W_V/W_O = total attention compression ~6×. Attention is ~30% of 70B params. Baseline 70B INT4 = 35 GB. Attention baseline: 10.5 GB → 6×: 1.75 GB. MLP at 35 GB → INT4 → 24.5 GB. Total: 26.3 GB. Still over 7.28 GiB. Needs NVMe complement. With NVMe paging of bottom 40 layers (Llama-3-70B has 80 layers): DRAM holds 40 layers of compressed weights ≈ 12 GB + attention = ~13.8 GB. Needs 2× more reduction. This is the ceiling analysis; 70B DRAM-resident without NVMe requires sub-1-bpw MLP — territory of PTQTP/NanoQuant, not this idea. The idea's contribution is establishing the K/n_heads scaling invariant as a structural finding.

### Residency arithmetic

Qwen3-4B surrogate (32 heads, d_model=2560): W_Q per layer = 2560 × 2560 = 6.5M params. If K/n_heads=8 invariant holds: K_4B = 32×8 = 256. ΔNLL at K=256: in 1.7B the W_Q K=256 gave ΔNLL=+0.51. If 4B behaves similarly, K=256 = 10× compression of W_Q on 4B. Verify on the Ryzen in 40 min. If confirmed, extrapolate to 70B: K=512, 16× compression — not directly runnable on Ryzen, but surrogate chain builds the argument. P(K/n_heads invariant holds): 0.5. P(the invariant alone unlocks 70B deployment): 0.15 (needs NVMe too). Structural finding value independent of residency outcome.

### Novelty gloss vs kill list and published landscape

No kill list item covers attention compression scaling laws across model sizes. Closest published: AQFKV (v2 R3-A, selected) established the 1.7B measurement; ATRC (R7-R-001 above) proposes to complete the 1.7B W_V/W_O sweep. QAWKC tests the scaling hypothesis across model sizes, which neither AQFKV nor ATRC addresses. MLA (DeepSeek-V2) trains the joint Q-K-V at different scales but uses training-time optimization; this is a post-training structural question. No paper measures the K/n_heads ratio as a scale-invariant structural property of trained transformers.

### Smallest experiment

Claim: W_Q global SVD on Qwen3-4B-Base (32 heads, d_model=2560) gives ΔNLL < 1.0 at K = 32×8 = 256, consistent with K/n_heads ≈ 8 as the invariant (vs Qwen3-1.7B K=128/16 heads = 8 per head). Runtime: 40–60 min (SVD on 2560×2560 matrix per 36 layers, larger model). Go: ΔNLL(4B, K=256) < 1.0 → invariant holds at 2 data points (1.7B, 4B), strong prior for 70B. No-go: ΔNLL > 2.0 — invariant breaks, scaling is sublinear, 70B may need K >> n_heads×8 → attention compression gain smaller at scale (also informative).

### Primary risk

The Qwen3-4B model doesn't fit in the Ryzen's RAM for bf16 SVD sweep (4B × 2 bytes = 8 GB, exactly at the 7.28 GiB limit). Mitigation: use blockwise streaming SVD (compute singular values layer-by-layer, never hold full weight tensor in RAM simultaneously); or use Qwen3-1.7B vs a publicly available 3B model that fits.

---

## R7-R-006 — Layer-1 SDZC Precomputed Gate Fold + MLP-Side Outlier Pinning Composition (LGOP)

**Tags:** R, Track A (primary), Track B (component)

### Mechanism

Track A (primary). CF6: 36% of gate channels in layer 1 are foldable (std < 0.05, SDZC near-constant). CF3/v2-CF1: outlier channels are K-dependent and dynamic above K=0.1%. The composition: layer 1 is structurally anomalous in two simultaneously measurable ways — (a) 36% of its gate neurons are near-constant (foldable via AERO-style fold into W_up), AND (b) if its outlier channel set is also anomalous compared to layers 2–27, then the layer 1 activation pathway is doubly compressible. The cascade: (1) measure whether layer 1's K=1% outlier Jaccard is HIGHER than the v2-CF1 mean (0.39) — i.e., whether layer 1 is MORE static (outlier channels more stable per token). If yes: layer 1 is a compression-dense layer — fold its near-constant gate neurons AERO-style AND apply static-channel pinning for its outlier activations with reduced INT4 quantization error. (2) The AERO-style fold on layer 1's 36% near-constant gate neurons saves: 36% × d_ffn × d_model × 2 bytes = 36% × 6144 × 2048 × 2 ≈ 9.1 MB per model. Negligible at 70B scale (layer 1 is ~1 layer). BUT: the structural composition claim is worth measuring because it tests whether early-layer specialization (CF6 layer-1 anomaly) co-occurs with outlier-stability specialization. If yes, early-exit architectures that skip deep layers should preserve layer-1 treatment specially. The "Reach" value is: if the co-occurrence pattern extends to other identifiable anomalous layers (not just layer 1), a selective compression strategy targets those layers for high-compression treatment, extracting 2–3× savings on a fraction of layers without retraining.

### Residency arithmetic

Layer 1 compression alone: ~9 MB saving in gate matrix (negligible). If pattern extends to 5–10 identifiable anomalous layers across a 28-layer model: 5 × 9.1 MB = 45 MB saving from gate folding. Still small. The residency case for Reach is: layer-anomaly mapping produces a per-layer compression schedule, and if 20–30% of layers have measurable anomalous structure (gate near-constancy ≥ 20%), the gate-fold saving becomes meaningful: 8 layers × 9.1 MB = 73 MB on 1.7B. At 70B: 8 × (36864 × 8192 × 2 × 36%) = 1.7 GB saving on gate matrices alone. At 2.1 GiB target residency for Qwen3-72B attention tier (from ATRC), this is a meaningful complement. P(layer-anomaly extends beyond layer 1): 0.3. P(gate-fold + outlier-pinning compose without destructive interaction): 0.7. P(end-to-end): 0.21.

### Novelty gloss vs kill list and published landscape

Closest kill list: SDZC (R4-A, killed as a global scheme because only 1.5% global foldable). LGOP is not a global SDZC; it is a per-layer-anomaly identification cascade that uses SDZC as a scanner, not a blanket. That structural distinction is explicit: SDZC was killed because the global fraction is too small; LGOP asks whether the layer-1 anomaly is part of a larger pattern of layer-heterogeneous foldability. Adjacent published: PowerInfer (arXiv:2312.12456) does neuron-level cold/hot partitioning based on activation frequency — this is a similar "identify structural anomalies per layer and treat them differently" logic. LGOP differs: it uses near-constancy of gate output (not activation frequency), targets fold (not skip), and is post-hoc rather than learned. No paper measures layer-heterogeneous gate-near-constancy across transformer layers to build a compression schedule.

### Smallest experiment

Claim: at least 3 layers in Qwen3-1.7B-Base have > 20% near-constant gate neurons (std < 0.05, same threshold as CF6), confirmed by the same probe that found 36% in layer 1. Runtime: 20 min (gate output statistics collection across all 28 layers at calibration time, identical pipeline to R4-A SDZC but per-layer). Go: ≥ 3 layers with > 20% foldable gate neurons. No-go: layer 1 is the only anomalous layer — SDZC remains a single-layer micro-optimization, not a cascade foundation; finding still extends CF6.

### Primary risk

Gate near-constancy is context-dependent and the 36% CF6 finding was on a specific calibration corpus; a held-out corpus shows only 5–10% near-constant globally, same as other layers, and layer 1 anomaly disappears. Mitigation: measure with two disjoint corpora; if layer-1 anomaly is corpus-specific, the mechanism is fragile; report as context-dependent structural feature rather than kill.

---

## R7-R-007 — Attention-Weight NVMe Prefetch Oracle via Static Access Fingerprint (AANF)

**Tags:** R, Track B

### Mechanism

Track B. CF11 established that W_Q attention weight matrices have more concentrated spectra than MLP matrices. An inference-time consequence: if we store attention weights at reduced precision (INT8 at K=128 per CF11) while MLP weights remain at INT4 and are NVMe-paged, the DRAM-resident working set is: attention weights (INT8, K=128) + KV cache. The NVMe access pattern for MLP weights is now layer-sequential and DETERMINISTIC — same sequence every token. Windows 11 NTFS Cache Manager can detect sequential access patterns and engage aggressive read-ahead. The cascade: (1) re-layout the GGUF weight file so all MLP weight pages for layer L are contiguous in the file (extent-aligned); (2) access in strict L=0,1,...,27 order each token; (3) the OS prefetcher fires 2–3 extents ahead automatically (no custom kernel code required — this is standard NTFS Cache Manager behavior). This is Orientation U territory but is brought here as a Reach cascade component: the attention SVD compression (from CF11) reduces the DRAM hot-set enough that the memory pressure on the OS prefetcher is relieved, enabling more aggressive prefetch windows. The structural argument: CF11 enables DRAM-resident attention; DRAM-resident attention removes attention from the NVMe access queue; clean NVMe access queue allows the NTFS prefetcher to deliver deeper MLP prefetch without contention. The two mechanisms compose constructively — not just additively. CF13* (NTFS extent-prefetch effective read rate) must be re-derived in rung 1.

### Residency arithmetic

DRAM-resident attention (all 28 layers, K=128, INT8): 28 × 4 matrices × 2048 × 128 × 1 byte ≈ 29 MB (negligible DRAM cost). MLP weights (28 layers × 2 × 6144 × 2048 × 0.5 bytes INT4): 422 MB per model at 1.7B scale. At 70B: MLP INT4 ≈ 24.5 GB on NVMe. NTFS sequential prefetch throughput target: if CF13* re-derivation shows ≥ 2 GB/s sequential reads, 24.5 GB / 2 GB/s = 12 s/pass — still slow for real-time tok/s. The cascade needs a second compression lever; this idea contributes the storage-layout component. P(NTFS prefetch engages and delivers ≥ 2 GB/s sequential for this workload): conditional on CF13* re-derivation (unknown). P(CF11 attention compression reduces DRAM pressure enough to clear the prefetch window): 0.8. Combined P: dependent on CF13* measurement.

### Novelty gloss vs kill list and published landscape

Closest kill list: R2/S2 NVMe Prefetch Sequencer via RoPE Fingerprints (killed because wrong axis). AANF is different: the access fingerprint here is not RoPE-derived but rather layer-sequential GGUF extent layout, and the load-bearing mechanism is the interaction between attention weight compression and prefetch contention relief. Apple LLM-in-a-flash (arXiv:2312.11514) uses row-level prefetch based on activation sparsity prediction; AANF uses deterministic sequential access + OS-managed prefetch (no custom predictor). The interaction claim (CF11 attention compression → prefetch contention relief) is the novel structural composition.

### Smallest experiment

Claim: on the Ryzen 5 7530U with Qwen3-1.7B-Base GGUF file, sequential reads of extent-aligned MLP weight pages achieve ≥ 1.5 GB/s sustained throughput via NTFS Cache Manager read-ahead, vs < 0.7 GB/s for random-order access. Measure with a Python script that reads the GGUF file in layer-sequential order (sequential access) vs shuffled order (random access) and records MB/s. Runtime: 15 min. Go: ≥ 2× sequential/random speedup → re-derives CF13* for this stack. No-go: < 1.1× speedup → NTFS prefetch does not engage on this workload (adds to structural findings, kills AANF's prefetch premise).

### Primary risk

NVMe on the Ryzen 5 7530U is bottlenecked by the PCIe lane bandwidth or by Windows 11 Home I/O scheduler, not by prefetch depth — sequential reads don't outperform random reads because the hardware queue is already saturated. Mitigation: the smallest experiment measures this directly; if random and sequential are equal, the prefetch mechanism is irrelevant for this hardware configuration, and the idea is recast as a layout-cleanup (no tok/s gain, but reduced seek overhead for other access patterns).

---

## Convergence handles

- W_V and W_O spectrum concentration (r_99/d): whether these attention matrices are in the CF11 class (< 0.80) or the MLP class (≈ 1.0) — the single measurement that gates ATRC, VJOFR, and QAWKC
- K=0.1% static outlier channel mask reproducibility across contexts: gates CLOTS and ASOR
- K/n_heads ≈ 8 as a head-redundancy invariant across model scales: gates QAWKC scaling hypothesis
- Layer-heterogeneous gate near-constancy across all 28 layers: gates LGOP extension beyond CF6 layer-1 anomaly
- NTFS sequential vs random read throughput ratio on Ryzen (CF13* re-derivation): gates AANF and any other NVMe-prefetch cascade
- W_{OV} = W_O · W_V product rank vs factor ranks: structural finding that either extends AERO's fold logic to attention path or kills it
