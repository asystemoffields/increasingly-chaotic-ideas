# Stage 1 — Orientation C (Composition) — Run 001

Orientation: C — Composition
Round: 1 (v2), no prior v2 rounds, no saturated-frame list, no framing-diversity targets.

---

## C1-A — Joint QK Subspace Outlier Highway (JQSOH)

**Track A — arch-transposition.** CF11 shows W_Q has r_99/d ≈ 0.63 and 16 query heads collapse to a ~128-dim shared subspace. CF3 shows that per-token top-0.1% activation channels are channel-static (Jaccard = 0.718) while at K=1% they are fully token-dynamic (Jaccard = 0.31). The composition claim: **if the 128-dim shared W_Q subspace and the ~2-channel static outlier set measured at K=0.1% are aligned in the residual stream, then the static outlier channels constitute a persistent "highway" that the low-rank Q projection both reads from and projects through without loss.** Formally, let S ⊂ R^d be the set of ~2 channels with Jaccard ≥ 0.718, and let P_{Q} ∈ R^{d×128} be the head-shared W_Q projection basis from CF11. The coupling claim is:

  dim(span(e_i : i ∈ S)  ∩  colspan(P_Q)) ≥ 1

i.e., at least one of the two static outlier directions is preserved by the shared Q basis. If this holds, the outlier channels are not discarded by the K=128 head-shared W_Q compression — they're inside the surviving subspace. The practical consequence: a W_Q global K=128 compression (CF11 GO at +0.98 nats) need not apply any special treatment to static outlier channels because those channels already live in the surviving basis. The arch-transposition target: a single post-training low-rank replacement W_Q → P_Q Σ_Q V_Q^T (global, K=128) combined with static routing of the two outlier channels through an INT16 bypass buffer. The bypass adds 2 scalars per layer per token; the main path is INT8. Per-layer bandwidth: (128×d + 2×2) scalars vs 16×128×d scalars for the naive 16-head full layout — 16× head-parameter saving at K=128.

**Residency arithmetic.**
- Qwen3-1.7B attention weights: 28 layers × 4 matrices × d^2 weights (d=2048). W_Q and W_K at K=128: 28 × (2048×128 + 128×16×128) × 2 bytes ≈ 28 × (262144 + 262144) × 2 ≈ 28 MB. At full bf16: 28 × 4 × 2048² × 2 bytes ≈ 940 MB total attention block. Head-shared K=128 on W_Q alone saves ~(1 − 128/(16×128)) = 87.5% of W_Q bytes — ~117 MB on Qwen3-1.7B.
- For 70B scale (d=8192, 64 heads): W_Q at head-shared K=512 saves 64× nominal capacity → ~2 GB reduction in attention weights. Combined residency: 70B at ~0.55 bpw ≈ 5.35 GiB baseline; attention W_Q/W_K at K=512 remove ~2 GB → 3.35 GiB, comfortably DRAM-resident. This framing assumes CF11 generalizes to 70B; that's the coupling precondition.
- Quality cost: CF11 reports +0.98 nats at K=128 on 1.7B; that's above the 0.3-nat target but at K=512 dNLL=+0.20, which is within budget. 70B should be run at K=512.

**Novelty gloss.** Closest kill-list item: R3/S4-deferred AQFKV (Q-head SVD with KV preserved), which is a measurement proposal, not a deployment proposal. JQSOH is a composition: it takes the CF11 W_Q head-redundancy finding AND the CF3 K-dependent outlier-channel stationarity finding, and asks whether the static outlier directions survive the K=128 projection — a question AQFKV never asked. Closest published method: MLA (DeepSeek-V2) uses low-rank Q-K projection at training time; JQSOH proposes post-training application to an already-trained standard-attention model, exploiting measured head-redundancy that MLA cannot assume pre-training. The joint-alignment claim has no neighbor.

**Smallest experiment.**
Claim: at least 1 of the 2 K=0.1% static-outlier channel basis vectors has cosine similarity ≥ 0.3 with any column of the W_Q shared projection matrix P_Q (K=128 global SVD of stacked W_Q).
Runtime: compute W_Q SVD (K=128) for one layer of Qwen3-1.7B-Base: ~2 min. Compute K=0.1% static-outlier channels from CF3 data (already available from R2 PDAP). Compute pairwise cosines. Total: ~10 min on CPU.
Go threshold: max cosine ≥ 0.3 across any (outlier channel, P_Q column) pair in ≥ 14 of 28 layers.
No-go structural finding: if the outlier channels are orthogonal to P_Q, the INT16 bypass buffer IS required and the composition must be explicit; that's a different but still actionable architecture.

**Primary risk.** The static outlier channels (CF3 K=0.1%) may be fully outside the shared W_Q subspace, making the "highway preserved without special treatment" claim false. Mitigation: the bypass buffer (2 scalars per layer per token in INT16) makes the composition workable regardless; the experiment just resolves whether the bypass can be omitted.

---

## C2-B — Asymmetric bpw Allocation from Spectral Rank Boundary (ABAR)

**Track B — compression.** CF11 establishes a sharp spectral boundary: MLP weights have r_99/d ≈ 1.0 while W_Q has r_99/d ≈ 0.63 and W_K ≈ 0.79. CF12 establishes that the tied embed/lm_head matrix has r_99/d = 0.97 with catastrophic sensitivity (ΔNLL = +19.96 at r=1024, 50% rank). These three findings jointly define a **spectral rank hierarchy**: embed/lm_head > MLP > W_K > W_Q. The composition claim is a monotonicity bound:

  compress(X) cost ∝ r_99(X)/d × sensitivity(X)

The sensitivity of tied embed/lm_head is extreme (catastrophic at 50% rank); W_Q is 0.98 nats at 8× via rank reduction; MLP is intractable via rank reduction but tolerates quantization. This hierarchy forces a non-uniform bpw allocation that is the *opposite* of what a naive "compress the big matrices first" strategy would produce: the big tied embed/lm_head must be left at high bpw; W_Q gets the most aggressive compression (rank-based); MLP gets standard quantization (not rank reduction). The composition: allocate bpw by the measured sensitivity curve, not matrix size. Concretely, for Qwen3-1.7B-Base: embed/lm_head at bf16 (no reduction; tied config prohibits it per CF12); W_Q at K=512 global SVD (CF11 GO, +0.20 nats, 4× compression); W_K at K=512 (CF11 GO, +0.29 nats, 2× compression); MLP at 4-bit quantization (standard Q4 baseline); W_V/W_O at bf16 (pending spectrum measurement). The allocation is entirely driven by the empirical spectral sensitivity numbers, not a hand-tuned mixed-precision heuristic.

**Residency arithmetic.**
Qwen3-1.7B-Base bf16 baseline: ~3.4 GB.
- W_Q at K=512 (all 28 layers): 28 × 2048 × 512 × 2 bytes × 2 (Q_A and Q_B factors) ≈ 114 MB vs 28 × 2048² × 2 bytes ≈ 457 MB → saves 343 MB.
- W_K at K=512 similarly: saves 343 MB.
- MLP at Q4_0 (~4 bpw): 28 × 3 × 2048 × 6144 × 0.5 bytes ≈ 528 MB vs 1053 MB bf16 → saves 525 MB.
- Tied embed/lm_head: stays bf16, 151936 × 2048 × 2 bytes ≈ 590 MB unchanged.
Total new residency ≈ 590 (embed) + 114 (W_Q) + 114 (W_K) + 528 (MLP Q4) + 228 (W_V+W_O bf16 estimate) ≈ 1.57 GB.
That is a 2.2× total compression on Qwen3-1.7B. For 70B (where lm_head is untied per CF12 notes): embed/lm_head compression may be possible (untied case unconfirmed); attention W_Q/W_K at K=512 save ~3.5 GB; MLP at Q4 brings 70B from 35 GB → ~20 GB; lm_head IF untied at r=512 saves ~4 GB → ~16 GB. DRAM-resident 70B at 7.28 GiB still out of reach at this bpw profile alone, but the spectral-boundary-derived allocation is the correct input to any further compression cascade.

**Novelty gloss.** Closest kill-list items: MDL-Selected Per-Layer bpw (R2/S2 killed) and RSIDC (deferred). ABAR differs from both: MDL used per-layer compressibility as the selector (within one matrix type); ABAR uses cross-matrix-type spectral sensitivity (r_99/d × empirical ΔNLL curve) to allocate bpw types (rank-reduction vs quantization) across the tensor hierarchy. RSIDC estimated intrinsic dimension from residual stream activations; ABAR derives the allocation from weight matrix spectral measurements. Closest published method: mixed-precision quantization (GPTQ, AWQ) uses layer-sensitivity heuristics; ABAR's claim is that sensitivity is empirically determined by spectral rank, not estimated by Hessian-diagonal proxies on activations.

**Smallest experiment.**
Claim: W_Q K=512 + W_K K=512 (all-layer simultaneous, no W_V/W_O change) achieves ΔNLL < 0.5 nats on Qwen3-1.7B-Base on 512-token WikiText-2.
Runtime: SVD rank-k replacement of W_Q and W_K for all 28 layers: ~10 min. Eval: ~5 min. Total ≤ 20 min.
Go threshold: ΔNLL < 0.5 nats AND total attention-weight residency ≤ 228 MB (vs 915 MB bf16).
No-go structural finding: if W_K at K=512 degrades beyond 0.5 nats, the asymmetric allocation must move W_K to higher rank, tightening the residency case.

**Primary risk.** W_V and W_O spectra are unmeasured; if they are as concentrated as W_Q, leaving them at bf16 is a missed compression; if they are full-rank like MLP, the current plan is correct. Mitigation: W_V/W_O SVD spectrum is a ≤10 min measurement parallel to the smallest experiment.

---

## C3-A — Layer-1 Gate Anomaly × Head-Shared Q-Projection Fold (LGHF)

**Track A — arch-transposition.** CF6 shows that Layer 1 of Qwen3-1.7B has 36.1% of gate neurons with variance below the foldable threshold (std < 0.05). CF11 shows global W_Q compression to K=128 is feasible (ΔNLL = +0.98 nats). These two findings address different objects (MLP gate output vs attention W_Q), but both are layer-1 anomalies relative to the rest of the stack. The composition claim: **if Layer 1 has anomalously foldable gate structure AND anomalously compressible W_Q structure (if CF11's head-redundancy is even stronger in Layer 1 than the global mean), then Layer 1 admits a joint compression — folding 36% of gate neurons AND applying aggressive W_Q rank reduction — with a strictly smaller joint quality cost than applying either alone, because Layer 1 processes tokens before contextual information has accumulated (pre-context regime) and the two compressions are functionally independent (MLP gate-side vs attention Q-projection).**

The math: let Δ_gate(L1) be the ΔNLL cost of folding 36% of layer-1 gate neurons, and Δ_WQ(L1,K) be the ΔNLL cost of reducing W_Q to rank K in layer 1 only. Independence claim: Δ_{joint}(L1) ≈ Δ_gate(L1) + Δ_WQ(L1,K) ± ε, where ε is the interaction term. If the two compressions operate on disjoint subgraphs within Layer 1 (MLP sub-graph and attention sub-graph share only the residual stream input, which is unchanged by either compression), ε should be small. The joint compression of Layer 1 thus "pays once" for the residual-stream input shared by both paths.

**Residency arithmetic.**
Layer-1 MLP gate folding: 36% of gate neurons in Layer 1. Each foldable neuron eliminates one row of W_gate[L1] and one corresponding scalar. W_gate[L1]: 2048×6144 bf16 = 24 MB. 36% fold removes 36% of rows → saves ~8.6 MB on W_gate[L1]; corresponding W_up row savings ~8.6 MB. Total layer-1 MLP fold: ~17 MB.
Layer-1 W_Q at K=64 (more aggressive than global K=128 since layer 1 is pre-context): saves (2048 - 64)/2048 × (2048² × 2 bytes) ≈ 31 MB → saves ~31 MB.
Total Layer-1 joint: ~48 MB savings — tiny in absolute terms on 1.7B. The mechanism is the important discovery: if Layer 1 is genuinely more compressible across both paths simultaneously, the finding scales directly to all models in the Qwen3 family. On 70B (d=8192, intermediate=28672), Layer-1 savings scale as d²: 48 × (8192/2048)² = 48 × 16 = ~768 MB — non-trivial.

**Novelty gloss.** Closest kill-list items: R4-A SDZC (killed as global scheme, not Layer-1-specific), and Track A R2 LCAB (Layer-1 specific gate folding, deferred). LGHF is structurally different from LCAB: LGHF is a composition of Layer-1 gate folding WITH Layer-1 W_Q compression, hypothesizing independence and stronger compressibility in the pre-context regime — LCAB targeted only the gate. No kill-list item covers this cross-path Layer-1 composition. Closest published method: early-layer simplification in Matryoshka-style progressive networks (different axis entirely); no published work specifically targets the Layer-1 joint MLP+attention compression on the basis of empirical gate-variance and W_Q head-redundancy anomalies.

**Smallest experiment.**
Claim: Layer-1 W_Q at K=64 has ΔNLL < 0.5 nats (more aggressive than global K=128) due to pre-context regime.
Runtime: SVD rank-64 replacement of W_Q[Layer-1] only, eval on WikiText-2: ~5 min.
Go threshold: ΔNLL < 0.5 nats at K=64 AND ΔNLL(layer-1 gate fold + W_Q K=64 joint) < ΔNLL(gate fold alone) + ΔNLL(W_Q K=64 alone) + 0.1 nats.
No-go structural finding: if Layer-1 W_Q is NOT more compressible than global mean, the "pre-context regime" argument fails and Layer-1 anomalies are uncorrelated, weakening ABAR's simplest version.

**Primary risk.** The independence claim (Δ_joint ≈ Δ_gate + Δ_WQ) may fail: SwiGLU gate outputs feed into the residual stream which then feeds the next layer's attention, creating a non-additive interaction. Mitigation: the smallest experiment directly measures Δ_joint vs Δ_gate + Δ_WQ; interaction size is empirical.

---

## C4-B — Activation-Outlier Staticity as Quantization-Tier Selector (RAOK-Grounded)

**Track B — compression.** CF3 establishes a three-regime structure: K=0.1% channels are channel-static (Jaccard 0.718), K=1% are token-dynamic (Jaccard 0.31), K=10% are token-dynamic (Jaccard 0.24). CF6 establishes that deep layers (5-27) have 75-99% "live" neurons (std ≥ 0.1) in the gate, implying the activation distribution is non-trivial throughout the network. The composition claim: **the CF3 K-dependent Jaccard structure defines a natural quantization-tier boundary that is architecturally stable and calibration-free. Specifically, the ~2 channels with Jaccard ≥ 0.718 should be handled in FP16 (static, known in advance), the ~18 channels at K=1% with Jaccard ≈ 0.31 should be handled by per-token dynamic INT8 scaling (token-dynamic but only 18 channels), and the remaining 2028 channels at K≥10% should be handled by INT4 with a single per-tensor scale.** The tier boundaries are directly measurable from the Jaccard curve; they do not require per-layer calibration. The coupling between CF3 (outlier dynamics) and CF6 (deep-layer activation liveness) implies the tier-0 static channels cannot be pre-determined from gate variance alone (because gate output is live, not dead), confirming that the tier-0 channels are determined by activation magnitude distribution, not gate collapse.

Algebraically: let π_i(t) = activation magnitude rank of channel i at token t. Define:
- Tier 0: {i : E_t[π_i(t)] ≤ 2} — always top-2; static; FP16 pass-through.
- Tier 1: {i : 2 < E_t[π_i(t)] ≤ 20} — top-0.1-1%; dynamic; INT8.
- Tier 2: remaining 2028 channels; INT4 per-tensor.

The Jaccard numbers from CF3 directly calibrate the Tier 0 / Tier 1 boundary. The tier partition is derived from one measurement, not tuned per layer.

**Residency arithmetic.**
Baseline activations at BF16 per token per layer: 2048 × 2 bytes = 4096 bytes. Under the tier scheme: 2 channels × 2 bytes (FP16) + 18 channels × 1 byte (INT8) + 2028 channels × 0.5 bytes (INT4) = 4 + 18 + 1014 = 1036 bytes per activation vector. Compression ratio: 1036/4096 ≈ 0.25 — 4× activation bandwidth reduction. Per-token per-layer GEMV bandwidth: for W_down (6144→2048), input is post-SwiGLU (2028 INT4 + 18 INT8 + 2 FP16). On Ryzen 5 7530U with 11.5 GB/s DRAM bandwidth, W_down at 4 bpw: 2048×6144×0.5 bytes = 6 MB per layer. At ~25 layers/second (rough estimate), 26 layers = 6 MB × 26 / 11.5 GB/s ≈ 13.6 ms/token → ~73 tok/s on activations alone. The activation tier compression is secondary to weight bandwidth for long-context KV; at 4K context, it is noise. But at weight-level: if INT4-quantizing W_down's output-dimension weights to exploit INT4 activation inputs, combined residency of MLP block can approach 2 bpw.

This is a compression mechanism on activations, not on weights. The residency benefit is primarily in the KV cache (activations stored at K positions). At 4K context, 28 layers × 4K tokens × 2028 INT4 channels per key = 28 × 4096 × 1014 bytes ≈ 116 MB vs 28 × 4096 × 4096 bytes ≈ 471 MB at BF16 — 4× KV reduction with CF3-calibrated tier boundaries. KV compression is the primary payoff.

**Novelty gloss.** Closest kill-list item: R4/S4-deferred RAOK (R2-Aware Outlier-Keyed Activation Codebook) which deferred this direction. RAOK-Grounded is the composition step: it adds CF6 (deep-layer activation liveness) to CF3 (outlier dynamicity) to argue the tier boundaries are calibration-free (the Jaccard curve from CF3 is stable across diverse prompts, as confirmed in R2 with 200 prompts). RAOK as deferred did not have this composition; it required per-layer calibration. Closest published method: LLM.int8() uses a static outlier channel partition; RAOK-Grounded shows that partition is only accurate for K≤0.1% (CF3 Tier 0) and proposes a three-tier refinement whose boundaries come from the Jaccard curve.

**Smallest experiment.**
Claim: the Tier-0 channel set (top-2 channels by mean magnitude rank) is stable across prompts — same 2 channels appear in the top-0.1% across ≥90% of tokens on diverse calibration prompts, in ≥20 of 28 layers.
Runtime: extract top-2 channels per layer per token over 200 diverse prompts (available from R2 PDAP dataset); compute stability fraction: ~10 min.
Go threshold: ≥90% same-channel stability in ≥20/28 layers.
No-go structural finding: if Tier-0 stability is below 90%, the static FP16 bypass is not channel-static — it must be demoted to per-token detection, adding overhead.

**Primary risk.** The Tier-1 (18-channel) dynamic INT8 path requires per-token channel identification at runtime; if identification latency exceeds the GEMV latency saving, the tier scheme is net-negative. Mitigation: the R2 PDAP dataset provides a direct measurement of per-token Tier-1 stability (Jaccard at K=1% = 0.31); a predictor based on previous token's Tier-1 set (carry-forward) would be correct on 31% of channels and wrong on 69% — not good enough for zero-overhead; accept the overhead or demote to per-tensor INT8 with no tier split.

---

## C5-A — Cross-Layer W_Q Basis Sharing with Residual-Stream Alignment Check (CQBAL)

**Track A — arch-transposition.** CF11 shows all 28 W_Q matrices collapse to a common ~128-dim subspace per layer (16-head → 1 head's worth of shared basis). The question CF11 leaves open: do the 28 per-layer shared bases further share a common cross-layer basis? If so, a single cross-layer W_Q basis P_{cross} ∈ R^{d×r} (r << d) could replace all 28 per-layer W_Q with a shared basis plus 28 per-layer small correction matrices. CF2 establishes that the residual stream input to each layer has cos(h_L, h_{L+1}) ≈ 0.99 — residual stream states are nearly parallel across adjacent layers. The composition claim:

**If (a) within-layer W_Q is already ~128-rank (CF11) AND (b) adjacent residual stream inputs are nearly parallel (CF2, cos ≈ 0.99), THEN W_Q[L] and W_Q[L+1] operate on nearly identical input distributions, and their head-shared subspaces are likely similar (measured by Grassmannian distance). If the Grassmannian distance between W_Q[L] and W_Q[L+1] projected subspaces is small (≤ 0.1 in the canonical angles metric), a single cross-layer basis reduces all 28 W_Q to one shared P_{cross} + 28 small Δ_L matrices.**

Formally: let P_L = top-128 left singular vectors of W_Q[L] (stacked 16-head version). The coupling claim is:

  max_{L=1..27} d_Grassmannian(P_L, P_{L+1}) ≤ δ

for some δ measured empirically. If δ ≤ 0.15, then a single cross-layer basis P_{cross} (the SVD of the horizontally stacked P_L matrix) plus 28 small residual corrections Δ_L ∈ R^{d×r'} (r' << 128) approximates all W_Q. Total cross-layer W_Q storage: d×128 (shared basis) + 28 × d×r' (corrections). At r'=16: 2048×128×2 + 28×2048×16×2 bytes ≈ 0.5 MB + 1.8 MB = 2.3 MB vs 28×2048²×2 bytes = 457 MB for full bf16 W_Q — 200× compression on W_Q alone, at the cost of an extra per-layer small correction GEMV.

**Residency arithmetic.**
Full bf16 W_Q for 28 layers: 28 × 2048² × 2 bytes = 457 MB.
Cross-layer basis P_{cross} (d × 128): 2048 × 128 × 2 = 0.5 MB.
Per-layer correction Δ_L (d × r', r'=16): 28 × 2048 × 16 × 2 = 1.8 MB.
Total W_Q: 2.3 MB — 200× compression. Quality cost: CF11 shows K=128 global W_Q saves 8× at +0.98 nats; cross-layer sharing adds further approximation; estimate total +1.2-1.8 nats if cross-layer Grassmannian distance is small (empirically unknown — this is the coupling claim to test). At 70B scale: W_Q at d=8192, 64 heads, basis K=512: bf16 full = 28×8192²×2 ≈ 7.3 GB; cross-layer at K=512 basis + r'=64 correction = 8192×512×2 + 28×8192×64×2 ≈ 8 MB + 29 MB ≈ 37 MB. Eliminates 7.2 GB of W_Q at 70B. Combined with W_K K=512 savings (~3.6 GB), attention weight residency drops from ~14 GB to ~0.1 GB at 70B — transformative for DRAM-resident 70B deployment if quality holds.

**Novelty gloss.** Closest kill-list items: R1/S4-deferred Idea H (Krylov Subspace Iterative, killed as SVD variant); R3/S4-deferred Idea G RSTD (cross-layer Tucker on W_up, killed by W_up full-rank finding). CQBAL is structurally different: (a) it targets W_Q (not W_up or W_gate, which are full-rank); (b) the cross-layer sharing claim is ENABLED by the CF2 residual-stream parallelism that makes adjacent W_Q inputs similar; (c) no RSTD-style Tucker is needed — Grassmannian distance is the composing quantity. Closest published method: cross-layer weight sharing (ALBERT) uses identical weights; CQBAL uses a shared basis with per-layer corrections, never identical weights. The composition of CF2 (residual stream parallelism) with CF11 (W_Q head-redundancy) into a cross-layer Grassmannian-distance claim is novel.

**Smallest experiment.**
Claim: mean Grassmannian distance between adjacent layers' W_Q shared subspaces (K=128 per layer) is ≤ 0.20 (in canonical angles, 0=identical, π/2=orthogonal).
Runtime: compute per-layer W_Q top-128 SVD for all 28 layers; compute 27 adjacent Grassmannian distances; ~15 min total.
Go threshold: mean canonical angle ≤ 0.20 AND max canonical angle ≤ 0.40.
No-go structural finding: if cross-layer Grassmannian distance is large (> 0.40), the W_Q subspaces are layer-specific despite residual-stream similarity — CF2 does NOT imply W_Q subspace alignment. That finding refines CF2's interpretation and motivates per-layer K=128 (not cross-layer) as the best achievable.

**Primary risk.** The correction matrix Δ_L requires a per-layer small GEMV at inference (d×r' multiply), adding compute. If r' must be ≥ 64 to maintain quality, the saving is 10× not 200×, and the added GEMV latency may offset the bandwidth saving on DRAM-bound workloads. Mitigation: measure quality at r'=8, 16, 32 before committing to the deployment form.

---

## C6-B — W_Q Concentration × Tied-Embedding Rigidity as Joint Allocation Oracle (CTERA)

**Track B — compression.** [FREE SWING]

CF11 finds W_Q r_99/d ≈ 0.63 and W_K r_99/d ≈ 0.79. CF12 finds tied embed/lm_head r_99/d = 0.97 with catastrophic sensitivity (ΔNLL = +19.96 at 50% rank) and notes gradient flows through both input embedding AND output projection paths during training. The composition: **the tied embed/lm_head must remain at high bpw (CF12 forces this), but its high-bpw storage is adjacent in the computation graph to the residual stream that feeds the first-layer W_Q. The W_Q at K=128 reduction means we're projecting d=2048 inputs into a 128-dim space in layer 1 — yet the embed/lm_head has 2048 active dimensions that all contribute to the residual stream. These two facts are in tension: we are spending 590 MB to maintain all 2048 embed dimensions, then immediately projecting them to 128 dims at the first attention layer. The composition claim is that this tension resolves not by compressing the embed, but by co-designing the W_Q[L=1] basis to preferentially span the dimensions of the embed that carry the most per-token information, yielding a joint allocation where the embed is used more efficiently.** Formally: let E = embed matrix (151936 × 2048) and P_Q1 = W_Q[L=1] top-128 singular vectors. Define the "embed-query alignment":

  α = || E^T P_Q1 ||_F² / (||E||_F² × ||P_Q1||_F²)

If α > 0.5, the first-layer W_Q already preferentially attends to the embed's high-variance directions — the joint design is already implicit in training. If α < 0.2, there is a misalignment: the embed stores information in directions that W_Q[L=1] ignores. In the misalignment case, rotating the embed by P_Q1 (an O(d²)-cost one-time operation, with no weight modification) would improve the effective use of the embed's fixed high-bpw storage, yielding better-quality low-rank W_Q compression with the same residency budget.

**Residency arithmetic.**
No weight residency change in this mechanism — the payoff is quality improvement for a fixed W_Q K=128 budget. The relevance to the floor target: if CTERA's embed-query alignment rotation improves W_Q K=128 from +0.98 nats to +0.5 nats, that unlocks more aggressive W_Q compression (K=64 instead of K=128 at the same quality), saving an additional 50 MB on 1.7B and ~2 GB on 70B. The one-time rotation is a zero-residency-cost algebraic identity: replace E with E P_Q1^T (rotate embed once at model-load time); replace P_Q1 with P_Q1 P_Q1^T (identity on the range); no new parameters.

**Novelty gloss.** No kill-list item covers the embed/W_Q co-alignment claim. Closest published method: RoPE rotation absorption and similar embedding-space rotations (LoftQ, QuIP#) absorb quantization error by rotating weight matrices jointly; CTERA's mechanism is different — it asks whether the first-layer query projection is already aligned with the embed's high-variance directions, and uses alignment to diagnose whether a rotation would help. The CF12-CF11 coupling (tied embed rigidity forces high bpw, W_Q concentration creates a bottleneck) produces a joint allocation argument with no published neighbor.

**Smallest experiment.**
Claim: embed-query alignment α ≥ 0.3 (W_Q[L=1] already preferentially spans high-embed-variance directions).
Runtime: compute ||E^T P_Q1||_F² where E is the 151936×2048 embed and P_Q1 is the top-128 SVD basis of W_Q[L=1] from the existing R3-A AQFKV computation. E^T P_Q1 is 151936×128 — matrix-multiply in numpy, ~30 sec.
Go threshold: α ≥ 0.3 confirms implicit joint design; α < 0.2 flags misalignment and motivates the rotation step.
No-go structural finding: misalignment (α < 0.2) is a finding that the training optimization has NOT jointly optimized embed and query directions — a structural characterization of how the two findings (CF11, CF12) are DECOUPLED in practice, sharpening the CF9 precondition check for future compositions.

**Primary risk.** The rotation E → E P_Q1^T is only exactly weight-preserving if P_Q1 is square (d×d orthogonal); at K=128 it is not square, so the rotation is a projection that destroys 2048-128=1920 embed dimensions at first-layer query time — this is a loss, not a gain. Mitigation: the smallest experiment checks alignment first; if α is high (>0.5), the projection loss is already accepted by the trained model's W_Q and the rotation is redundant (no gain, no loss). The rotation is only useful when α is in the 0.2-0.5 range, where explicit optimization of the K=128 basis to span embed-variance directions would reduce the quality cost at K=128.

---

## C7-A — Attention Rank × Activation Outlier Tier for Mixed-Compute Layer Graph (MCLG)

**Track A — arch-transposition.** CF11 (W_Q r_99/d ≈ 0.63, head-redundancy: 16 heads → ~128 dims) and CF3 (K-dependent outlier stationarity: 2 static channels, 18 token-dynamic, rest homogeneous) together define two independent bottlenecks in the same token-processing pipeline: the query projection bottleneck (128 effective dimensions for Q) and the activation channel-tier bottleneck (2 + 18 + 2028 channel tiers). The composition claim: **these two bottlenecks, if simultaneously active in each layer, define a mixed-compute graph where the attention path (W_Q compressed to K=128, bandwidth 4×) and the MLP activation path (Tier-0 FP16 + Tier-1 INT8 + Tier-2 INT4, bandwidth 4×) have the SAME approximate compression ratio. This isomorphism means a single hardware dispatch scheme — streaming the compressed attention and compressed activation data at the same bus width — can service both paths simultaneously without a pipeline bubble.** The joint scheduling claim: at each layer, both paths compress to ~4× over BF16, so a dual-stream memory fetch (attention weights + activation data) at 4× compression fits in the same DRAM bandwidth envelope, enabling attention-MLP parallelism on the CPU without a stall.

Formally: if W_Q (K=128, ~4× compressed) and Tier-2 MLP activations (~4× compressed) are laid out contiguously in memory, a single AVX2 VMOVDQU64 stream at 64 bytes/cycle can prefetch both simultaneously. The pipeline: fetch 64 bytes of compressed W_Q; fetch 64 bytes of compressed activation input; issue both GEMVs; no stall. On Ryzen 5 7530U with 11.5 GB/s DRAM bandwidth, this doubles effective utilization vs serial fetch.

**Residency arithmetic.**
The isomorphism argument does not reduce residency further beyond C1+C4 already establish; it reduces memory-fetch latency by enabling dual-stream. Primary metric: tok/s. At 11.5 GB/s effective DRAM bandwidth and 70B at ~5.35 GiB (NanoQuant baseline), ~73 tok/s maximum without stalls. With a pipeline bubble from mismatched attention/activation compression ratios, effective bandwidth drops 30-40%. Eliminating the bubble by isomorphism compression → 10-15% tok/s improvement. On the Ryzen baseline of ~6.2 tok/s (from reference data), +15% ≈ 7.1 tok/s — not transformative alone, but composition with C2-B residency reduction brings the product to a meaningful improvement.

**Novelty gloss.** No kill-list item addresses dual-stream attention-MLP scheduling from a compression-isomorphism argument. Closest published method: Flash-Attention fuses attention compute; MCLG fuses attention-weight fetch and MLP-activation fetch based on matching compression ratios, not compute fusion. The compression-isomorphism claim (both paths compress to ~4×) is the novel coupling — it has no published neighbor in the ML literature (though CPU pipeline engineers would recognize the scheduling pattern from codec-plus-decompressor dual-stream design).

**Smallest experiment.**
Claim: W_Q K=128 compressed layout and Tier-2 INT4 MLP activation layout both achieve ≥3.5× byte reduction vs BF16 baseline on Qwen3-1.7B-Base in identical memory chunks.
Runtime: measure bytes-per-token for W_Q K=128 (direct calculation, 2 min) and bytes-per-token for INT4 Tier-2 activations (from R2 PDAP dataset, 5 min). Compute compression ratios.
Go threshold: both paths achieve ≥3.5× byte reduction.
No-go structural finding: if compression ratios diverge by >2×, the dual-stream scheduling requires padding — still possible but the isomorphism argument for bubble elimination is weakened.

**Primary risk.** The dual-stream memory fetch assumes both paths are memory-bound simultaneously; if the attention path is compute-bound (GEMV too cheap relative to DRAM latency), the fetch overlap provides no gain. Mitigation: profile attention GEMV compute vs fetch latency on Ryzen 5 7530U with K=128 W_Q at a single layer; if compute < fetch, the pipeline bubble exists and overlap helps.

---

## Convergence handles

1. **W_Q cross-layer Grassmannian distance** — load-bearing for C5-A (CQBAL); cheapest measurement not yet run; resolves whether 28 per-layer W_Q share a cross-layer basis.
2. **Tier-0 channel stability (top-2 by magnitude rank across prompts)** — load-bearing for C4-B (RAOK-Grounded); CF3 Jaccard=0.718 at K=0.1% is the anchor; per-layer stability fraction is the derived measurement.
3. **W_Q[L=1] × embed-matrix alignment angle α** — load-bearing for C6-B (CTERA); resolves whether tied-embed rigidity (CF12) and W_Q concentration (CF11) are co-optimized or decoupled in training.
4. **Layer-1 W_Q rank (K threshold for dNLL < 0.5 nats)** — load-bearing for C3-A (LGHF); if Layer-1 is more compressible, pre-context-regime compression is viable.
5. **Joint dNLL from CF6 gate fold + CF11 W_Q K=128 in Layer-1 simultaneously** — the interaction term ε in the independence claim; if small, Layer-1 admits a joint compression.
6. **Spectral rank boundary monotonicity: r_99/d for W_V and W_O** — load-bearing for C2-B (ABAR); the CF11-CF12 spectral hierarchy is incomplete without W_V/W_O measurements; these are cheap (<10 min each).
