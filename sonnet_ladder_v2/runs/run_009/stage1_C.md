# Stage 1 — Orientation C (Composition) — Run 009

Orientation: C — Composition
Run: 009 (independent cold start)
Kill compliance: v2-CHEAP-TEST-001 (cross-layer W_Q stacked SVD / shared basis — class kill). Per-head W_Q K=64 NO-GO (CF11). CF12 tied-embed SVD truncation dead. Arbitrary O(d) rotation (v2-S3-R004-001) dead. Within-layer Q/K joint (PALU/TransMLA) dead. Softmax × RoPE dead.

Collision check vs runs 001–008:
- CF3/v2-CF1 × CF11 W_Q outlier-alignment axes: exhausted (runs 001–007).
- v2-CF1 per-layer Jaccard × W_Q rank schedule: C7-JDWQ (run_007), C-LDOC/C-DEEP-SPREAD-QBUD (runs 008/006).
- CF12 × CF1 W_down/neuron coupling: C8/C10 (run_007).
- CF2 × CF12 lm_head factorization: C12-RUNE (run_007).
- CF6 × CF2 gate-fold: C-RKSC (run_008).
- W_K × CF3 sidecar: C-WKOL (run_008).
- CF1 × CF11 subspace overlap: C-UFDM (run_008).
- SRR cross-class sensitivity law: C-SPECQ (run_008).
This run targets four unexplored coupling axes: (1) E[token_id] row geometry × W_Q K=128 subspace, (2) CF6 Layer-1 c_1 constant injection × Layer-2 W_Q logit shift, (3) CF5 W_up sensitivity asymmetry × CF11 attention spectrum as joint budget oracle, (4) CF2 residual near-unity × CF12 tied-embed as lm_head skip gate.

Ideas produced: 5

---

## C13-EQSUB — Embedding Row Projection into W_Q K=128 Subspace (EQSUB)

**C / Track B**

### Mechanism

Track B — compression. CF11 established that global W_Q K=128 captures 99% of W_Q variance, with the 16 query heads collectively spanning a ~128-dim subspace. The CF11 experiment operated on W_Q weight matrices. The NEW coupling: the input to each W_Q layer is h_ℓ = E[token_id] + Σ residual updates, where E is the tied embedding matrix (CF12: full-rank, catastrophically sensitive). At Layer 1, h_1 ≈ E[token_id] because the residual update from Layer-0 attention is small (CF2: ‖Δh‖ ≈ 14% of ‖h‖ per layer). The coupling claim: **E[token_id] rows — the initial hidden states — lie preferentially INSIDE the W_Q K=128 global subspace.**

**Equation (EQSUB):** Let V_Q^{128} ∈ ℝ^{d×128} be the top-128 right singular vectors of the global W_Q stack at layer ℓ=1 (CF11 GO subspace). For each vocabulary row E[v] ∈ ℝ^{2048}, define the projection mass:

    μ_E = (1/|V|) Σ_{v=1}^{151936} ‖V_Q^{128T} E[v]‖₂² / ‖E[v]‖₂²

μ_E is the fraction of embedding-row energy inside the W_Q K=128 subspace, averaged over vocab. The coupling prediction: μ_E ≥ 0.60. Rationale: SGD jointly optimizes the embedding matrix E and the query matrices W_Q^ℓ to minimize the same language-modeling loss. The embeddings encode the token identities that queries need to route on; if 16 heads collapse to a 128-dim query subspace, the training pressure shapes E rows to concentrate their "query-relevant" information in the same 128 dims that W_Q already focuses on. This is NOT the same as CF12's full-rank finding (which measures E's internal rank) — it measures E's alignment to a different matrix's subspace.

If μ_E ≥ 0.60: the Layer-1 query W_Q^{L1} · h_1 is approximately W_Q^{L1} · (V_Q^{128} V_Q^{128T} h_1) — the projection step (V_Q^{128T} h_1) is computing with an input that already lives mostly in the subspace, meaning the K=128 truncation loses very little of E[token_id]'s content. This motivates Token-0-Only aggressive compression: at inference, if the first token's h_1 ≈ E[token_id], W_Q^{L1} at K=64 (tighter than global K=128) may still be adequate because the INPUT is already subspace-aligned.

If μ_E < 0.30: E rows are isotropic relative to W_Q's subspace; the K=128 global GO is NOT helped by embedding geometry; the compression budget for W_Q^{L1} is not loosened.

Does not rely on: cross-layer W_Q stacking (v2-CHEAP-TEST-001), MLP rank (CF8), CF13–CF15.

### Residency arithmetic

Direct residency saving: if μ_E ≥ 0.60 and W_Q^{L1} can be compressed to K=64 (ΔNLL ≤ +0.10 nats at L1), the Layer-1 W_Q saving is 2048×(128−64)×2B = 0.26 MB — negligible in absolute terms. The composition value is the SCALING ARGUMENT: if μ_E ≥ 0.60 holds across all layers (because E rows retain their subspace alignment through the residual stream), then the full 28-layer W_Q can target K=64 instead of K=128. Saving: 28 × 2 × 2048 × (128−64) × 2B = 14.7 MB vs 29.4 MB — halving W_Q storage at the same quality budget.

At 70B (W_Q total at K=512 = 1.34 GB): if the embedding-alignment argument scales (μ_E ≥ 0.60 at 70B scale), K=256 may suffice, halving W_Q to 0.67 GB. Joint with W_K K=512: attention weights at 70B become ~(0.67 + 0.34 GQA) ≈ 1.0 GB vs ~12 GB at bf16.

Quality: if μ_E ≥ 0.60, the projection W_Q^{ℓ, K=64} · h_ℓ loses only 40% × (1 − μ_E) ≤ 16% of the embedding energy on the ground that matters for query computation. This is substantially below the uniform-distribution 50%-rank-50%-energy expectation for K=1024 (full), consistent with the CF11 K=128 GO at +0.98 nats.

### Novelty gloss vs kill list and published landscape

Kill list closest: v2-CHEAP-TEST-001 killed cross-layer W_Q stacked SVD — EQSUB is per-layer projection mass of E rows, not stacking W_Q across layers. CF11 (head-shared W_Q subspace) is one of the two component findings; EQSUB adds the embedding geometry as the new coupling axis. No prior Composition run measured E row projection mass into W_Q principal directions — prior runs examined W_Q SVD × activation outliers (CF3), not E × W_Q alignment. Published closest: MLA (DeepSeek-V2) projects keys and values into a shared latent space jointly with embeddings at training time — EQSUB is post-training, zero-retraining, and asks whether the ALIGNMENT already exists in the trained weights rather than imposing it. No published work measures μ_E as a compression budget signal for post-training W_Q rank allocation.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base at layer L1, μ_E (mean projection mass of E rows into W_Q^{L1} K=128 subspace) ≥ 0.50.

Procedure: (a) load W_Q[L1] bf16 (2048×2048); SVD to get V_Q^{128} (top-128 right singular vectors); (b) load E (tied embed, 151936×2048); compute projection_mass(v) = ‖V_Q^{128T} E[v]‖₂² / ‖E[v]‖₂² for all 151936 rows; (c) compute μ_E = mean(projection_mass). Memory: E = 590 MB + V_Q^{128} = 1 MB — fits in 7.28 GiB. Runtime: ~20 min (SVD of W_Q[L1] in ~1 min; batch matrix-multiply E @ V_Q^{128} in ~8 min on Ryzen, no autograd needed; compute norms). Extend to all 28 layers: ~35 min.
GO threshold: μ_E ≥ 0.50 at L1 AND μ_E ≥ 0.45 mean across all 28 layers.
NO-GO structural finding: μ_E ≈ 0.128 (= 128/1024, the random-initialization expectation for 128-dim subspace of 2048-dim space) — E rows are isotropic relative to W_Q's principal subspace; the embedding geometry provides no additional compression budget; the K=128 truncation is already near-optimal for the actual residual stream distribution.

### Primary risk

The CF11 W_Q K=128 measurement used GLOBAL stacking of W_Q (all 28 layers simultaneously) to get V_Q^{128}. EQSUB measures projection mass into per-LAYER V_Q^{128} subspaces. The global and per-layer subspaces may differ substantially (and v2-CF1 suggests they do: layer Jaccard varies). Mitigation: compute μ_E using BOTH the global V_Q^{128} (reusing CF11 AQFKV factors) and the per-layer V_Q^{128} (L1, L14, L27); if μ_E is similar under both, the result is robust to the layerwise vs global distinction.

---

## C14-CFSHIFT — CF6 Layer-1 Constant Injection × W_Q[L2] Logit Pre-bias (CFSHIFT)

**C / Track A**

### Mechanism

Track A — arch-transposition. CF6 established that 36% of Layer-1 gate neurons have near-constant output (std < 0.05), implying these neurons contribute a fixed calibration-derived vector c_1 ∈ ℝ^{2048} to the Layer-1 residual update: h_2 = h_1 + c_1 + Δh_1^dynamic + attn_1. The c_1 term is token-independent and thus known at model-load time (computed once from calibration data). CF2 establishes cos(h_L, h_{L+1}) ≈ 0.99, but at the SINGLE-LAYER level ‖Δh_1‖ ≈ 14% × ‖h_1‖ — so c_1 is a meaningful fraction of the full Layer-1 update.

The composition claim: **the fixed vector c_1 injected into the residual stream before Layer 2 produces a predictable constant offset in the attention logits at Layer 2.**

**Equation (CFSHIFT):** The Layer-2 query vector for head h is q_2 = W_Q^{h,L2} · h_2. Since h_2 = h_1 + c_1 + Δh_1^{dynamic} + attn_1 (where the last two terms are token-dependent):

    q_2 = W_Q^{h,L2} · (h_1 + c_1) + W_Q^{h,L2} · (Δh_1^{dynamic} + attn_1)
        = [q_2^{base}(h_1) + δq_c1] + q_2^{dynamic}

where δq_c1 = W_Q^{h,L2} · c_1 is a CONSTANT head-specific query offset, computed once at model-load time. Similarly, key vectors at Layer 2 pick up a constant term from c_1 in the preceding residual. The pre-biased attention logit at Layer 2 is:

    A^{L2}_{qk} = (q_2^T k_2) / √d_head
                = (q_2^{base} + δq_c1)^T (k_2^{base} + δk_c1) / √d_head
                = A^{base} + (δq_c1^T k_2^{base} + q_2^{base T} δk_c1 + δq_c1^T δk_c1) / √d_head

The cross-terms (δq_c1^T k_2^{base}) and (q_2^{base T} δk_c1) are token-dynamic (because k_2^{base} and q_2^{base} are token-dependent), but the purely-constant term δq_c1^T δk_c1 / √d_head is a fixed scalar per (query-position, key-position) pair within the head — a CONSTANT LOGIT BIAS added to every attention logit at Layer 2. This biases attention patterns at Layer 2 in a fixed, calibration-derived direction.

**Arch-transposition:** Pre-absorb the constant logit bias into the Layer-2 attention mask (add δq_c1^T δk_c1 / √d_head to the static position bias for every query-key pair). This is a one-time computation at model-load time — no per-token cost. More importantly, the cross-terms δq_c1^T k_2^{base} / √d_head can be precomputed per KEY TOKEN as a scalar correction to each key's contribution to all query logits. This converts part of Layer-2's attention logit computation from a per-query-per-key inner product into a per-key scalar add — cheaper if the constant term is dominant.

If ‖δq_c1‖₂ / mean‖q_2^{base}‖₂ ≥ 0.20: the constant offset is ≥ 20% of the typical query magnitude. The Layer-2 logit is partially predictable from c_1 alone, before any per-token W_Q computation. This enables a 1-bit per-head pre-filter: if the full L2 logit is dominated by δq_c1^T k_2 (the constant query dotted against the key), the top-K attention pattern at Layer 2 is partially determined by c_1's projection onto W_Q^{L2}, enabling attention pruning before computing the dynamic component.

Does not rely on: CF13–CF15. MLP rank (CF8). Cross-layer W_Q (killed). Softmax × RoPE (killed — RoPE rotates keys, but c_1's CONTRIBUTION to the key vector is δk_c1 = W_K^{L2} · c_1, and RoPE rotation of δk_c1 at position p is a known rotation of a known vector — still precomputable at position p).

### Residency arithmetic

c_1: 2048 × 2B = 4 KB (calibration-derived, stored once). δq_c1 per head (16 heads): 16 × 128 × 2B = 4 KB. δk_c1 per head: same. These are negligible — zero residency overhead.

Compute saving at Layer 2: the δq_c1^T k_2^{base} inner product (per key token, per head) is a 128-dim dot product. For a sequence of n tokens, this saves 16 heads × n × 128 FMAs versus computing the full W_Q^{L2} · h_2 per query. At n=4096 and 16 heads: 16 × 4096 × 128 = 8.4M FMAs saved at Layer 2. Total Layer-2 attention FMAs: 2 × 16 × 4096 × 128 = 16.8M. Saving: 50% of Layer-2 cross-term computation if the constant term is dominant. This is not a residency win; it is a COMPUTE win at long context.

Quality: the pre-bias is exact (not an approximation) — absorbing the constant offset into the logit bias introduces zero quality cost. The arch-transposition changes the execution graph (splitting the logit into constant + dynamic contributions) without changing the output.

### Novelty gloss vs kill list and published landscape

Kill list closest: Track A R2 LCAB (Layer-1 gate folding, deferred but not killed) is about the GATE fold mechanism; CFSHIFT is about what the c_1 term DOES to the NEXT layer's attention logits. No prior Composition run asked whether CF6's constant injection propagates into the attention logit structure at Layer 2. The math is a direct consequence of linearity (W_Q applied to the sum h_1 + c_1 distributes), not a new mechanism — the novelty is recognizing the CONSTANT LOGIT BIAS as a structural identity exploitable at inference. Published closest: ALiBi (Press et al., 2022) adds a per-head position-dependent attention bias; this composition adds a calibration-derived CONTENT-dependent constant bias, grounded in CF6's gate-fold measurement. No published method derives a constant Layer-N+1 attention logit bias from the preceding layer's fixed-gate neuron statistics.

### Smallest experiment

**Claim:** At Qwen3-1.7B-Base, the calibration-derived constant query offset ‖δq_c1‖₂ / mean_calibration ‖W_Q^{L2,h} · h_2‖₂ ≥ 0.10 for at least 8 of 16 heads.

Procedure: (a) forward pass 200 PDAP calibration tokens; capture h_1 (Layer-1 input) and h_2 (Layer-2 input = h_1 + MLP_L1_output + Attn_L1_output); (b) identify Layer-1 gate foldable neurons (CF6, std < 0.05 per neuron over 200 tokens); (c) compute c_1 = mean_{tokens}(W_down^{L1}[:,foldable] · (μ_gate_foldable ⊙ mean(W_up^{L1}[foldable,:] · H_calib))); (d) compute δq_c1^h = W_Q^{h,L2} · c_1 for each of 16 heads; (e) compute mean query magnitudes ‖W_Q^{h,L2} · h_2(t)‖₂ per head over 200 tokens; (f) compare ratio. Runtime: ~30 min (reuse RKSC infrastructure from run_008 for steps a–c; add W_Q^{L2} projection in step d).
GO threshold: ≥ 8 heads with ratio ≥ 0.10.
NO-GO structural finding: δq_c1 is negligible (ratio < 0.02 for all heads) — the 36% foldable neurons at Layer 1 contribute a vector c_1 whose direction is ORTHOGONAL to all W_Q^{L2} principal directions; CF6 and CF11 are structurally decoupled at the cross-layer attention-logit level; the Layer-1 gate folding does not propagate as a structural bias into Layer-2 attention.

### Primary risk

c_1 as computed is a mean over the calibration distribution and is not truly constant — it is a mean-field approximation. Real inference tokens that deviate from the calibration distribution will see a different effective c_1, meaning the "constant logit bias" is an approximation rather than an exact identity. Mitigation: report the standard deviation of (W_down^{L1}[:,foldable] · (gate_foldable(t) ⊙ W_up^{L1}[foldable,:] · h_t)) across calibration tokens — if std < 10% of mean magnitude, the constant approximation holds.

---

## C15-SNRATE — CF5 W_up Sensitivity × CF11 Attention Spectrum as Joint Precision Budget Oracle (SNRATE)

**C / Track B**

### Mechanism

Track B — compression. CF5 established that W_up is MORE rank-sensitive than W_gate at matched rank fractions (W_up ΔNLL +2.34 vs W_gate +0.77 at K=1024, 50% rank). CF11 established that W_K r_99/d ≈ 0.79, W_Q r_99/d ≈ 0.63. The confirmed rank sensitivity ordering is:

    tied embed >> W_up > W_gate ≈ ? W_K > W_Q

The composition claim: **the sensitivity ordering and the spectrum concentration ordering define a two-dimensional precision budget surface — (spectrum_concentration, sensitivity) — and the PRODUCT of these two quantities predicts optimal bits-per-weight for mixed-precision quantization without Hessian calibration.**

**Equation (SNRATE):** Define for each weight class C:
- concentration(C) = 1 − r_99/d (higher = more concentrated = more compressible by rank truncation)
- sensitivity(C) = ΔNLL at 50% rank (higher = more sensitive = less aggressively quantizable)

Define the signal-to-noise ratio SNR(C) = concentration(C) / sensitivity(C). From confirmed data:

| Class | conc. (1−r_99/d) | sens. (ΔNLL@50%) | SNR |
|---|---|---|---|
| W_Q | 0.37 | +0.03 | 12.3 |
| W_K | 0.21 | +0.29 | 0.72 |
| W_gate | ≈0.0 | +0.77 | ~0.0 |
| W_up | ≈0.0 | +2.34 | ~0.0 |
| tied embed | 0.03 | +19.96 | 0.0015 |

W_Q has SNR >> all others: high concentration + low sensitivity = genuinely compressible. W_K is middle (moderate concentration, moderate sensitivity). MLP weights and tied embed have near-zero SNR (either flat spectrum or catastrophic sensitivity or both).

The coupling prediction: **optimal bpw for quantization scales INVERSELY with sensitivity and is modulated by concentration.** Specifically, for weight classes with SNR ≥ 1 (W_Q, possibly W_K), lower bpw is safe. For classes with SNR < 0.1 (MLP, embed), quantization should be conservative. The SNRATE law:

    recommended_bpw(C) = b_base × (1 + k × sensitivity(C))^{-1} × (1 + concentration(C))^{α}

where b_base, k, α are fit on the confirmed data points. With 5 data points (W_Q, W_K, W_gate, W_up, tied embed), this is a 3-parameter fit on 5 points — well-conditioned (CF10 check: n_samples=5, n_params=3 with rank constraint).

The structural consequence: given (r_99/d, ΔNLL@50%), the optimal bpw is predictable for any NEW weight class without running a fresh Hessian calibration. W_V and W_O (untested) can be budget-assigned from the SNRATE fit as soon as their r_99/d is measured (a 20-min experiment).

Does not rely on: MLP rank reduction (CF8 — the claim is about QUANTIZATION bpw, not rank truncation), CF13–CF15, cross-layer stacking (killed).

### Residency arithmetic

Payoff is QUANTIZATION BUDGET ALLOCATION across weight classes at 70B. Current uniform Q4_K_M on all weights. SNRATE predicts: W_Q → 2-3 bpw (SNR=12.3 supports aggressive compression), W_K → 3-4 bpw (SNR=0.72), W_gate → 4 bpw (SNR≈0), W_up → 4 bpw, tied embed (Qwen3-1.7B) → 8 bpw (SNR≈0, never compress).

For Qwen3-1.7B-Base (1.86 GB at bf16):
- W_Q (29.4 MB bf16) at 2 bpw: 3.7 MB (saving 25.7 MB)
- W_K (29.4 MB bf16) at 3 bpw: 5.5 MB (saving 23.9 MB)
- W_gate + W_up (2×1.18 GB) at 4 bpw: 591 MB (saving 591 MB from 1.18 GB each)
- Tied embed (590 MB at bf16) stays at 8 bpw: 295 MB

Total at SNRATE-derived mixed precision: 3.7 + 5.5 + 591 + 295 + (W_down, W_O, W_V at 4 bpw, est. 250 MB) ≈ 1.15 GB vs 1.86 GB bf16. At 70B (inference path without embed): W_Q + W_K at 2-3 bpw = ~1.0 GB total; MLP weights at 4 bpw = ~38 GB / 4 = 9.5 GB; attention V/O at 3-4 bpw = ~2 GB. Total ≈ 12.5 GB — under 14 GB NVMe-streamed with 1-GB active DRAM window.

### Novelty gloss vs kill list and published landscape

Kill list closest: C-SPECQ (run_008) proposed SRR(class) = ΔNLL(50%) / (1 − r_99/d) as a cross-class sensitivity law. SNRATE is the INVERSE — SNR = concentration / sensitivity — and derives a bpw schedule from it, which SPECQ explicitly declined to do (SPECQ: "this idea does not directly reduce residency"). The structural difference: SNRATE's novelty is the closed-form bpw prediction and the CF10-compliant fit (3 params, 5 data points), while SPECQ targeted the law itself as a structural finding. They complement rather than duplicate. Published closest: OmniQuant / AWQ calibrate per-layer sensitivity via Hessian; SNRATE replaces Hessian calibration with spectrum + rank-sensitivity (both measurable in minutes via SVD, no activation Hessian). SqueezeLLM uses sensitivity-aware quantization but requires Hessian. The spectrum-concentration axis as a SECOND input (alongside sensitivity) has no direct published analog.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base, the SNRATE-predicted bpw ordering (W_Q most aggressive < W_K < W_gate ≈ W_up < embed) is consistent with a direct bpw sweep: ΔNLL at 2-bpw W_Q quantization (all other weights at bf16) ≤ +0.20 nats, and ΔNLL at 2-bpw W_gate (all other weights at bf16) ≥ +1.00 nats.

Procedure: (a) quantize only W_Q (all layers) to INT2 (llama.cpp Q2_K or numpy round-to-nearest on 2-bit grid) while holding all other weights at bf16; eval ΔNLL on 455-token WikiText-2; (b) repeat for W_gate only at INT2. Runtime: ~40 min (2 quantization + eval passes). Note: 2-bpw W_Q may degrade more than +0.20 (the SNR prediction is a hypothesis); the experiment discriminates whether the sensitivity ordering is monotone even at 2 bpw, not whether it is exactly +0.20.
GO threshold: ΔNLL(W_Q at 2bpw) ≤ +0.50 nats AND ΔNLL(W_gate at 2bpw) ≥ 2× ΔNLL(W_Q at 2bpw).
NO-GO structural finding: W_Q at 2 bpw has ΔNLL ≥ +1.00 nats — quantization sensitivity does NOT track rank-truncation sensitivity (CF11 rank sensitivity ≠ quantization sensitivity at 2 bpw); the SNRATE law conflates two distinct sensitivity types; the bpw schedule requires separate calibration. This is still a structural finding: rank truncation and quantization sensitivity are orthogonal dimensions.

### Primary risk

CF5 and CF11 measured RANK TRUNCATION sensitivity (SVD + rank-K reconstruction), not quantization sensitivity. Low-rank truncation and quantization have structurally different error distributions (low-rank error is structured in singular-value space; quantization error is approximately uniform grid rounding). The SNRATE law hypothesizes these are correlated; this is an assumption, not a theorem. Mitigation: the smallest experiment directly tests the correlation; even NO-GO produces a clean finding separating the two sensitivity types.

---

## C16-RSEMBED — Residual Near-Unity × Tied Embed as Inference-Time Vocabulary Shortlist Gate (RSEMBED)

**C / Track A**

### Mechanism

Track A — arch-transposition. CF2 establishes cos(h_L, h_{L+1}) ≈ 0.99 (all hidden states track the initial embedding direction). CF12 establishes the tied embed E is full-rank (r_99=1992/2048) — E^T · h_L is the logit vector for the lm_head. The composition claim: **the near-unity cosine from CF2 implies that the FINAL hidden state h_L is approximately aligned with h_0 = E[token_id], and therefore the logit vector E^T · h_L is approximately the column of E^T E closest to the current token — measurably concentrated on tokens semantically near the current input token.**

**Equation (RSEMBED):** Let r = cos(h_L, h_0) ≈ 0.99 (CF2's cumulative observation across 28 layers — actually CF2 measures ADJACENT layer cosine at 0.99, which compounds to cos(h_L, h_0) ≈ 0.99^28 ≈ 0.75 across all 28 layers; CHECK: this is the critical factual claim to verify). If cos(h_L, h_0) ≥ 0.60 empirically:

    E^T · h_L ≈ E^T · (r · ê_0 + √(1−r²) · δ)

where ê_0 = h_0/‖h_0‖ and δ ⊥ ê_0 is the orthogonal residual component. The first term E^T · (r · ê_0) = r × (E^T E)[token_id,:] — a row of the 151936×151936 vocabulary similarity matrix, which is NOT precomputable cheaply (46 GB). However, we do not need E^T E; we need only WHICH TOKEN is argmax(E^T · h_L). The coupling claim:

    argmax_v E^T · h_L ≈ argmax_v E^T · h_0  with high probability for short-distance tokens

i.e., for tokens where the residual Δh = h_L − h_0 is small relative to the signal, the predicted next token is ALREADY approximately determined by the current input token's embedding. This is measurable as:

    skip_rate = P(argmax(E^T · h_L) ≠ argmax(E^T · (h_0 + r_ε)))

where r_ε is a small calibration-derived noise term. If skip_rate ≤ 10% of positions: 10% of lm_head computations can be short-circuited by checking whether E^T · h_0 has a dominant argmax (no full lm_head needed). The arch-transposition: before computing the full 311M-MAC lm_head, check if cos(E[argmax_prev], h_L) ≥ threshold_gate (a 2048-dot-product check, ~0.002% of lm_head cost); if above threshold, reuse the argmax from E^T · h_0 (the current input token or a cached prediction), skipping the full lm_head.

This is different from C12-RUNE (run_007), which decomposed the lm_head into a base + correction term. RSEMBED is a BINARY SKIP GATE — either the prediction is trivial (the model predicts something near the input token) or not; no approximation involved when skipped; full lm_head fallback when not skipped.

Does not rely on: CF13–CF15. MLP rank (CF8). Cross-layer W_Q (killed). The mechanism depends on cos(h_L, h_0) being measurably above random (> 0.50) — CF2 alone does not guarantee this; the 28-layer compound cosine must be measured.

### Residency arithmetic

Savings only apply when skip_rate > 0. The skip gate costs: 2048-dim dot product between h_L and E[argmax_prev] = 2048 FMAs = 0.0007% of lm_head cost. If skip_rate = 5%: 5% of lm_head calls replaced by gate check. lm_head bandwidth: 590 MB per token. Skip saves: 0.05 × 590 MB = 29.5 MB/token bandwidth. At 11.5 GB/s DRAM: 29.5/11500 = 2.6 ms/token saved. At 3 tok/s current baseline: saves 2.6 ms / 333 ms/token ≈ 0.8% speedup. At 70B NVMe scenario (335 ms/token): lm_head is 590 MB vs 35 GB total = 1.7%; skip saves 0.05 × 1.7% ≈ 0.08% of total. Modest.

The real payoff is ARCHITECTURE: if skip_rate ≥ 20% on natural language (e.g., function words, determiners, common prepositions that are highly predictable from context), the gate is a principled early-exit for a subset of decode steps, enabling speculative decode from the current token without a full draft model. Unlike speculative decoding, the skip gate requires no draft model and no acceptance criterion — it is a deterministic threshold on cos(h_L, E[gate_token]).

### Novelty gloss vs kill list and published landscape

Kill list closest: C12-RUNE (run_007) decomposes lm_head into h_0 + Δh components with a low-rank correction — architecturally similar motivation. RSEMBED is a BINARY SKIP gate (no approximation), not a decomposition, and the skip condition is a cosine threshold, not a rank projection. The skip event occurs only when the prediction is trivially concentrated near the input token; the full lm_head runs otherwise. Published closest: "Speculative Decoding" uses a small draft model to propose tokens and a large model to accept/reject. RSEMBED uses the IDENTITY (current token) as the implicit draft — if the model is likely to predict a token near the current token (measured by cos(h_L, E[input_token])), skip the full verification. Orca (speculative decoding with n-gram drafts) is closer: n-gram draft = predict the current n-gram repeats; RSEMBED generalizes this to a semantic proximity check on h_L. No published method uses the CF2 residual-proximity measurement to construct the skip gate criterion.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base at inference on 200 PDAP calibration prompts, cos(h_L(t), E[token_id(t)]) ≥ 0.60 for ≥ 15% of token positions (i.e., the final hidden state is substantially aligned with the current input token's embedding in a measurable fraction of positions).

Procedure: (a) forward pass 200 PDAP prompts through Qwen3-1.7B-Base; capture h_L(t) (final residual) and h_0(t) = E[token_id(t)]; (b) compute cos(h_L(t), h_0(t)) for each token; (c) histogram over positions; (d) report P(cos ≥ 0.60). Runtime: ~25 min (hook forward passes on 1.7B, compute pairwise cosine per token).
GO threshold: P(cos ≥ 0.60) ≥ 0.10 (≥ 10% of tokens have cos ≥ 0.60).
NO-GO structural finding: cos(h_L, h_0) is concentrated near 0 (mean ≈ 0.20) for all tokens — CF2's adjacent-layer 0.99 cosine does NOT compound into any meaningful h_L–h_0 alignment over 28 layers; the residual transformation across 28 layers decorrelates h from the initial embedding completely; the lm_head gate is never useful; CF2 and CF12 are decoupled along the forward-pass compounding axis. This is a clean structural measurement not currently in the empirical record.

### Primary risk

CF2 reports cos(h_L, h_{L+1}) ≈ 0.99 for ADJACENT layers; the compound cosine over 28 layers is cos(h_0, h_28) ≈ 0.99^28 ≈ 0.75 IF the residual additions are approximately aligned. But residual additions are not necessarily aligned with h_L — they are learned transformations that may rotate h away from its initial direction. The adjacent-layer cosine being 0.99 is consistent with LARGE accumulated rotation if the rotation direction changes each layer. Mitigation: the smallest experiment measures cos(h_L, h_0) directly; if it is near 0, this risk is confirmed and the idea fails cleanly.

---

## C17-LAYERTAX — v2-CF1 Jaccard × CF6 Layer-1 Anomaly: Joint Layer Characterization Tax (LAYERTAX) [FREE SWING]

**C / Track B**

### Mechanism

Track B — compression. [FREE SWING]. v2-CF1 measured K=1% Jaccard across all 28 layers, producing a depth profile. CF6 identified Layer 1 as structurally anomalous (36% gate neurons near-constant). These two findings were produced by INDEPENDENT experiments on different model properties (activation outlier dynamics vs gate variance). The composition claim: **v2-CF1's Jaccard profile is measuring the same underlying structural dimension as CF6's gate variance anomaly — both reflect how much a layer's computation is "token-agnostic" vs "token-specific," and a joint Layer Characterization Score (LCS) should predict the full feasible compression budget per layer more accurately than either finding alone.**

**Equation (LAYERTAX):** Define for each layer ℓ:
- J_ℓ = K=1% Jaccard (v2-CF1): measures outlier channel stability across tokens (higher = more token-agnostic)
- G_ℓ = fraction of gate neurons with std < 0.05 (CF6): measures gate output stability across tokens (higher = more token-agnostic)

The Layer Characterization Score:

    LCS(ℓ) = α × J_ℓ + (1−α) × G_ℓ

where α ∈ [0,1] is fit on the observed data. The coupling prediction: LCS(ℓ) is monotone in the layer's compressibility — layers with high LCS (both high J_ℓ and high G_ℓ) tolerate more aggressive bpw reduction and W_Q rank reduction than layers with low LCS.

Layer 1 is the structural outlier: G_1 = 0.361 (anomalous high); J_1 = ? (not reported in v2-CF1, which sampled specific layers — need to check if L1 is in the v2-CF1 sweep). If J_1 is also anomalously high or anomalously low, the joint LCS either reinforces or contradicts the CF6 anomaly.

The [FREE SWING] flag: this is a composition of two findings that have no direct algebraic coupling — LCS is a heuristic combination, not a derived identity. The structural novelty is testing whether two independently measured "token-agnosticity" proxies (one from activation outlier channels, one from MLP gate variance) co-vary in a way that predicts a unified per-layer compression budget. If LCS is a consistent predictor, it is a new empirical instrument for per-layer mixed-precision scheduling, grounded in two v2-confirmed findings. If it fails, the finding is that the two proxies measure DIFFERENT dimensions of layer heterogeneity, which sharpens the structural picture.

Does not rely on: CF13–CF15. Cross-layer W_Q (killed). MLP rank (CF8). Softmax × RoPE (killed).

### Residency arithmetic

LAYERTAX is a measurement-and-schedule tool, not a direct compression mechanism. Its payoff is a per-layer bpw schedule for the full 70B deployment. If LCS correctly predicts per-layer compression tolerance:
- Bottom-LCS layers (lowest 8 out of 28 in Qwen3-1.7B): stay at bf16 or 8 bpw.
- Mid-LCS layers (middle 12): 4-5 bpw.
- High-LCS layers (highest 8, including Layer 1): 2-3 bpw.

At 1.7B (3 weight matrices per FFN: W_gate, W_up, W_down, plus attention):
- Low-LCS 8 layers at bf16: 8 × (3×1.5 + 0.2) GB wait — compute: 8 × 3 × 6144 × 2048 × 2B + 8 × 4 × 2048 × 2048 × 2B = 8 × 75.5 MB + 8 × 33.6 MB = 87 MB.
- High-LCS 8 layers at 2 bpw: 87/8 = 11 MB. Saving ≈ 76 MB from these 8 layers.
- Mid-LCS 12 layers at 4 bpw: 12 × (75.5 + 33.6) / 4 = 12 × 27.3 = 328 MB vs 12 × 109.1 MB = 1.31 GB. Saving ≈ 983 MB.

Total estimated saving vs all-bf16 1.7B: ~1.1 GB — about 60% model compression without uniform quantization assumptions. The key is whether LCS validates the schedule.

### Novelty gloss vs kill list and published landscape

Kill list closest: C-LDOC (run_008) scheduled W_Q rank per layer from v2-CF1 Jaccard; LAYERTAX adds the CF6 gate-variance axis and extends the schedule from W_Q rank to all-weight bpw. C-DEEP-SPREAD-QBUD (run_006) used v2-CF1 spread for W_Q rank — similarly partial. LAYERTAX is more ambitious: using BOTH v2-CF1 AND CF6 jointly to predict total per-layer compression tolerance across ALL weight classes. Published closest: mixed-precision quantization schedules (HAQ, LLM-QP, GPTQ) use per-layer calibration sensitivity (Hessian) to assign bpw. LAYERTAX replaces the Hessian with a 2-factor score based on token-agnosticity proxies from two independent structural measurements. No published work combines outlier-channel stability and gate-variance as a joint per-layer characterization score.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base, G_ℓ (fraction of gate neurons with std < 0.05, measured at each of 5 probe layers L1, L7, L14, L21, L27) and J_ℓ (v2-CF1 Jaccard at the same 5 layers) have Spearman ρ(J_ℓ, G_ℓ) > 0.20 (positive co-variation: layers where outlier channels are more static also have more near-constant gate neurons).

Procedure: (a) G_ℓ at 5 probe layers: forward pass 200 PDAP prompts; per-neuron gate std at L1 (CF6 already measured), L7, L14, L21, L27; fraction below 0.05. Runtime: ~30 min new forward passes for L7, L14, L21, L27 (L1 is already measured from CF6). (b) J_ℓ: already available from v2-CF1 rraok_result.md for the v2-CF1 sampled layers; check if L7, L14, L21, L27 are included. (c) Spearman ρ on 5 data points. Runtime total: ~35 min.
GO threshold: ρ > 0.20 (positive trend even with 5 points gives direction; full 28-layer sweep would be needed for statistical significance).
NO-GO structural finding: G_ℓ and J_ℓ are ANTI-CORRELATED or uncorrelated — the gate-stability and outlier-channel-stability measure different structural dimensions; there is no single "token-agnosticity" axis; per-layer compression scheduling requires TWO independent budget constraints (one per finding), not a unified LCS.

### Primary risk

The v2-CF1 sweep sampled specific layers but may not have included L7, L14, L21, L27 at the level of detail needed for G_ℓ vs J_ℓ comparison. Mitigation: re-check rraok_result.md for the exact layer set covered; if L7 and L21 are missing from v2-CF1, the Spearman test shrinks to 3 data points (L1, L14, L27), which is statistically minimal but still directionally informative.

---

## Convergence handles

1. **cos(h_L, h_0) per-token distribution** — load-bearing for C16-RSEMBED; also relevant to C14-CFSHIFT's compounding argument; a single 20-min forward pass over PDAP tokens settles the CF2 compounding question for both.

2. **E[token_id] row projection mass into W_Q K=128 subspace (μ_E)** — load-bearing for C13-EQSUB; requires W_Q L1 SVD (1-2 min) + batch matrix multiply E @ V_Q (8-10 min); converges if Reach or First-Principles orientations independently ask "does the embedding lie in the attention principal subspace?"

3. **Layer-1 constant offset ‖δq_c1‖₂ / ‖W_Q^{L2} · h_2‖₂ per head** — load-bearing for C14-CFSHIFT; reuses CF6 gate-fold infrastructure (c_1 computation) and adds one W_Q^{L2} projection; converges with any orientation that independently surfaces the CF6 gate fold as an attention-logit modifier.

4. **Spearman ρ(J_ℓ, G_ℓ)** — load-bearing for C17-LAYERTAX; requires G_ℓ measurement at 4 new layers (L7, L14, L21, L27 gate-neuron variance) and v2-CF1 J_ℓ at the same layers; converges if any orientation independently surfaces "layer-characterization score from two independent token-agnosticity proxies."

5. **ΔNLL at 2-bpw W_Q vs 2-bpw W_gate (head-to-head)** — load-bearing for C15-SNRATE; a 40-min discriminative experiment between the SNR ordering of quantization sensitivity and rank sensitivity; converges if First-Principles or Reach orientations independently surface a compression budget law from (concentration, sensitivity).
