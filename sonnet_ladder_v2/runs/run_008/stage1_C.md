# Stage 1 — Orientation C (Composition) — Run 008

Orientation: C — Composition
Run: 008 (independent cold start; does not inherit kills from runs 001–007)
Kill note: v2-CHEAP-TEST-001 kills the entire cross-layer W_Q shared-basis class. Per-head W_Q truncation (K_per_head=64) is dead (CF11). CF12 tied-embed SVD truncation is dead. All ideas below honor these kills.
Ideas produced: 5

Prior-run collision check: runs 001–005 exhaustively explored CF3 K=0.1% static outlier × CF11 W_Q subspace alignment (JQSOH, OHAA, QAOC, AQOD, ZOIA). This run targets strictly different coupling axes — v2-CF1 depth structure, CF1×CF11 cross-domain coupling, CF8 boundary × CF12 sensitivity asymmetry.

---

## C-LDOC — L27 Deep Quasi-Static Outlier × W_Q Depth-Specific Compression Floor (LDOC)

**C / Track B**

### Mechanism

Track B — compression. v2-CF1 extended CF3 to all 28 layers and found a sharp anomaly: L27 Jaccard@K=1% = 0.531, versus mean 0.388 across all layers and 0.34-0.44 in layers L20-L26. L27 is the final transformer block; its output feeds the LM head directly. CF11 established that W_Q global K=128 is a GO (ΔNLL=+0.98 nats) treating all 28 layers uniformly. The coupling claim: **the L27 quasi-static outlier structure implies L27's W_Q subspace is less "contaminated" by token-dynamic directions, and L27 tolerates a more aggressive W_Q compression (K < 128) than the global mean.**

**Equation (LDOC):** Let J_ℓ = Jaccard@K=1% at layer ℓ (measured per v2-CF1). Define the outlier-load fraction δ_ℓ = J_ℓ / J_{global_mean} as a per-layer relative stability ratio. The composition predicts:

    ΔNLL(W_Q^ℓ at K=K_ℓ) is monotone in δ_ℓ at fixed K_ℓ

i.e., layers with higher Jaccard (more static outlier sets) tolerate deeper W_Q compression at fixed quality budget. For L27 (δ_27 = 0.531 / 0.388 ≈ 1.37), the prediction is that L27's W_Q can be compressed to K=64 or K=96 at ΔNLL cost proportional to the global K=128 cost scaled by (1 − δ_27 × correction_factor). The structural argument: W_Q queries at L27 are applied to a residual stream whose dominant directions are stable across tokens (high Jaccard = the top-K channels are mostly the same token-to-token). A nearly-static input distribution concentrates W_Q's "needed" dimensions into a smaller subspace, making the K=128 bound loose at L27.

Does not rely on: cross-layer W_Q (v2-CHEAP-TEST-001 killed), MLP rank (CF8 killed), CF13-CF15.

### Residency arithmetic

Per-layer W_Q at bf16 (Qwen3-1.7B): 2048 × 2048 × 2 bytes = 8 MB. At K=128 (CF11 global): 2 × 2048 × 128 × 2 bytes = 2 MB (4× per-layer). At K=64 for L27 only: 1 MB, saving 1 MB vs K=128.

Absolute saving from L27 alone: 1 MB on a 1.7B model — negligible. The value is the **coupling measurement**: if the Jaccard-stability-to-compression-depth relationship is confirmed at L27, it motivates applying the same per-layer K schedule to all 28 layers using their v2-CF1 Jaccard values as the compression budget signal. A full per-layer K schedule where K_ℓ ∝ 128 × (1 − f(J_ℓ)) could save 20-30% over uniform K=128 at the same total ΔNLL, by spending fewer parameters on mid-layer (high-Jaccard, more compressible) W_Q and more on L0-L5 (low-Jaccard, more dynamic, less compressible).

At 70B scale: W_Q at 80 layers × d=8192 at K=512 (CF11 GO at +0.20 nats): uniform K=512 = 80 × 2 × 8192 × 512 × 2 bytes = 1.34 GB. If the Jaccard-based schedule identifies 20 layers where K=256 suffices: saves 20 × 2 × 8192 × 256 × 2 bytes = 335 MB — meaningful at the 7.28 GiB floor.

Quality cost: the per-layer ΔNLL at non-uniform K is not directly estimable from CF11 (which reports global K=128, not per-layer). The smallest experiment provides the per-layer ΔNLL decomposition.

### Novelty gloss

Kill list closest: C1-ODSP (run_005, depth-stratified bpw schedule from CF3 depth gradient). LDOC differs: ODSP used the R2 7-layer CF3 estimate for deep-layer spread to schedule *activation quantization bpw*; LDOC uses v2-CF1's full 28-layer Jaccard profile to schedule *W_Q rank*, and the falsifiable coupling is monotonicity of ΔNLL in J_ℓ at fixed K_ℓ. The object (W_Q rank vs activation bpw) and the input signal (Jaccard-at-K=1% from v2-CF1 vs spread-fraction from PDAP) are different. C6-DSAF (run_005) also scheduled W_Q rank per layer but used CF3 outlier-spread as the input, not Jaccard directly, and predicted a floor (K_ℓ must exceed effective_dim) rather than a monotonicity. LDOC tests a slope, not a floor. Published closest: mixed-precision attention compression schedules (LLM-QP); none use per-token Jaccard stability as the rank budget signal.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base, ΔNLL for W_Q K=64 at L27 only is ≤ 0.15 nats, and ΔNLL for W_Q K=64 at L0 only is ≥ 0.25 nats (L0 has Jaccard=0.221, least static; L27 has Jaccard=0.531, most static; the monotonicity prediction requires the spread ≥ 0.10 nats).

Procedure: (1) replace W_Q[L27] with rank-64 SVD, evaluate WikiText-2 512 tokens, record ΔNLL; (2) replace W_Q[L0] with rank-64 SVD (restore L27), evaluate; (3) repeat for L14 (mid-layer, Jaccard=0.421 — mid-value). Three per-layer eval passes: ~25 min total.
GO threshold: ΔNLL(L0, K=64) − ΔNLL(L27, K=64) ≥ 0.10 nats.
NO-GO structural finding: ΔNLL is uniformly distributed across layers at fixed per-layer K — Jaccard stability does NOT predict W_Q compression sensitivity; the v2-CF1 layer-depth Jaccard structure and CF11's attention spectrum are decoupled along the compression-sensitivity axis.

### Primary risk

The per-layer W_Q ΔNLL decomposition assumes single-layer replacement is approximately additive (i.e., replacing L27 alone approximates the L27 contribution to the global 28-layer ΔNLL). This additivity is not guaranteed if layers interact through the residual stream. Mitigation: report the single-layer replacement ΔNLL directly, without claiming it equals the marginal contribution. If single-layer ΔNLL at K=64 is below 0.05 nats for all three layers, the idea is dead on scale even if the monotonicity holds.

---

## C-UFDM — W_up Firing-Rank Direction × W_Q Head-Shared Subspace as Joint Input Basis (UFDM)

**C / Track A**

### Mechanism

Track A — arch-transposition. CF1 establishes that post-SwiGLU neuron firing is dominated by the W_up·x factor, not W_gate·x. CF11 establishes that the 16 query heads collectively span a K=128 subspace of the 2048-dim row space (global GO at ΔNLL=+0.98). Both findings are shaped by the same object: the residual-stream vector h flowing into each layer. CF1 says the top-K directions of W_up relative to h determine which neurons fire strongly. CF11 says the top-128 right singular vectors of stacked W_Q capture the directions of h that matter for query computation. The composition claim: **the W_up firing-dominant directions (the right singular vectors of W_up that, when projected onto h, produce the largest mean post-SwiGLU output) and the W_Q K=128 query subspace are substantially overlapping in the residual-stream coordinate system.**

**Equation (UFDM):** Let V_up^{K} ∈ ℝ^{d×K} be the top-K right singular vectors of W_up^ℓ (the directions of h that produce the largest ‖W_up^ℓ·h‖ projection magnitude, i.e., highest firing magnitude). Let V_Q^{128} ∈ ℝ^{d×128} be the top-128 right singular vectors of W_Q^ℓ (the CF11 head-shared subspace). The coupling claim:

    dim(colspan(V_up^{128}) ∩ colspan(V_Q^{128})) ≥ α × 128,    α ≥ 0.4

If α ≥ 0.4 (40% subspace overlap), the two operations share a dominant input coordinate system. The arch-transposition: a single shared input projection P ∈ ℝ^{d×256} (unioning V_up^{128} and V_Q^{128}) replaces both W_up's K=128 firing-dominant projection AND W_Q's head-shared projection. Both downstream operations (SwiGLU gating and query computation) receive the same 256-dim compressed representation of h, computed once per layer instead of twice. This is AERO-adjacent in spirit (shared computation graph) but uses an input-side joint basis rather than an output-side activation fold.

The "why non-obvious" argument: CF1 and CF11 were measured by different experiments targeting different objects (firing prediction vs attention quality). Their respective dominant input subspaces have not been compared. If they overlap at α ≥ 0.4, it means the trained model is already using a shared low-dimensional representation of h for both MLP gating and attention querying — a structural identity nobody extracted because the two experiments were run in isolation.

Does not rely on: cross-layer W_Q sharing (killed), MLP weight rank reduction (CF8 killed), cross-layer basis (killed), CF13-CF15.

### Residency arithmetic

Shared input projection P at dimension 256: 2048 × 256 × 2 bytes = 1 MB per layer × 28 = 28 MB.
Savings vs separate W_Q K=128 (28 MB) + W_up top-128 firing-projection (28 MB) = 56 MB combined: saving 28 MB (50% of the joint projection cost). The arch-transposition also saves one GEMV per layer per token (the shared P replaces two independent matrix multiplications on h). FLOP saving: 2 × 2048 × 128 FMAs per layer per token (one W_up K=128 read + one W_Q K=128 read) → 2048 × 256 FMAs (one P read), saving ~0.5M FMAs/layer/token at 1.7B.

For 70B (d=8192): joint P at dim=1024: 8192 × 1024 × 2 = 16 MB per layer × 80 = 1.3 GB. Full W_Q K=512 (14 GB) + W_up K=512 firing projection (14 GB) = 28 GB separately. If α ≥ 0.4 at 70B, shared P at K=1024 covers 40% of both → saves ~56% over separate projections. This is speculative until the 1.7B coupling is measured.

Quality cost: if the joint basis P at K=256 captures 40% of both subspaces, the W_Q ΔNLL is reduced below CF11's +0.98 nats (because the 40% overlap means the joint basis is better for query computation than V_Q^{128} alone). The MLP quality effect is not directly estimable from existing CF. Best-case: +0.98 × (1 − α) nats from the attention side.

### Novelty gloss

Kill list closest: nothing in the kill list touches CF1 × CF11 coupling. The natural kill-list neighbors are CF8 (MLP rank reduction — LDOC is not that; UFDM uses the firing-rank direction subspace, not the W_up weight matrix rank) and CF11 (W_Q compression — UFDM uses it as one of two compared subspaces). UFDM's novelty is the *cross-domain comparison* between two findings that were never measured jointly: one MLP-side (CF1 firing dominance → high-magnitude projection directions of W_up) and one attention-side (CF11 head-shared W_Q subspace). Published closest: A3 (arXiv:2505.12942) optimizes W_Q compression via attention-logit error minimization — does not touch W_up. AERO (arXiv:2410.13060) folds W_gate/W_up via activation removal — changes the graph structure but does not share input projections between MLP and attention. There is no published method that derives a joint MLP-attention input basis from the overlap of their respective dominant input subspaces.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base at a representative layer (L14), the subspace overlap dim(colspan(V_up^{128}) ∩ colspan(V_Q^{128})) / 128 ≥ 0.3 (using principal angles between the two 128-dim subspaces, measuring the fraction of principal angles below 30°).

Procedure: (a) load W_up[L14], compute SVD, extract V_up^{128}; (b) load W_Q[L14] SVD (available from AQFKV prior experiment), extract V_Q^{128}; (c) compute principal angles between the two subspaces via `scipy.linalg.subspace_angles`; (d) fraction of angles < 30° = overlap fraction α. Runtime: < 5 min (SVDs already computed or computable on 1.7B matrices in 1-2 min each on Ryzen).
GO threshold: α ≥ 0.3 at L14 AND consistent measurement at L0 and L27 (α within ±0.10 of L14 value).
NO-GO structural finding: the W_up firing-dominant subspace and the W_Q head-shared subspace are orthogonal (α < 0.1); the trained model uses different input directions for MLP gating and attention querying; no joint basis exists; CF1 and CF11 are structurally independent at the input level.

### Primary risk

The "firing-dominant directions of W_up" are right singular vectors of W_up, which reflect the *matrix structure*, not necessarily the empirical distribution of h during inference (CF1's "firing rank dominance" was measured on actual activations, not on W_up's SVD). The correct V_up^K should be derived from W_up^T × X_calib (the projection of calibration activations onto W_up's row space), not from W_up's right SVD directly. Mitigation: compute the empirical firing subspace as SVD(W_up · H_calib) where H_calib is the 200-prompt activation matrix collected in R2 PDAP, projecting the actual residual-stream distribution through W_up. This takes ≤10 min.

---

## C-RKSC — Residual Cosine Flatness × L1 Gate-Fold Creates Predictable Layer-1 Output Scale (RKSC)

**C / Track B**

### Mechanism

Track B — compression. CF2 establishes cos(h_L, h_{L+1}) ≈ 0.99 across all layers of Qwen3 (residual additivity: ‖Δh_ℓ‖₂ ≈ 0.141 × ‖h_ℓ‖₂). CF6 establishes that Layer 1 has 36.1% of gate neurons with std < 0.05 (near-constant gate output). The composition claim: **after folding the 36% near-constant gate neurons at Layer 1 (replacing their contribution by a fixed scalar), the Layer-1 MLP output Δh_1 has a substantially lower variance than Δh at other layers.** CF2 constrains ‖Δh_ℓ‖₂ globally to ≈ 14% of ‖h_ℓ‖₂, but within Δh_ℓ, the MLP contribution Δh_ℓ^{MLP} = W_down · silu(W_gate · h) ⊙ (W_up · h) varies. At Layer 1, the 36% foldable gate neurons contribute a fixed vector c_1 = W_down · (μ_{gate} ⊙ (W_up · h)) where μ_{gate} is the per-neuron mean gate activation. The REMAINING (64%) non-foldable neurons contribute a random-looking Δh_1^{dynamic} with zero-mean correction.

**Equation (RKSC):** The Layer-1 MLP output decomposes as:
    Δh_1 = c_1 + Δh_1^{dynamic}

where c_1 is a calibration-data-derived constant vector (zero per-token variance, computable once), and Δh_1^{dynamic} is the 64% non-foldable contribution. CF2 says ‖Δh_1‖₂ ≈ 0.141 × ‖h_1‖₂. After folding, ‖Δh_1^{dynamic}‖₂ ≤ 0.141 × ‖h_1‖₂ (since the constant term is removed). The coupling predicts: the variance-per-channel of Δh_1^{dynamic} is 36% lower than Δh_1 (the foldable neurons contribute zero to inter-token variance), enabling INT4 quantization of Δh_1^{dynamic} at a 36% tighter range, reducing quantization MSE by a factor of (0.64)^2 ≈ 0.41 vs unfold.

The practical mechanism: during inference, Layer-1 MLP is run as (a) add fixed c_1 to the residual stream (one vector add, no GEMV), (b) compute only the 64% non-foldable neurons (two GEMVs at 0.64 of full size: W_gate[non-fold] and W_up[non-fold], both of shape 2048×3932 instead of 2048×6144). The computation reduces 36% in the Layer-1 MLP GEMV. The inputs to those GEMVs are quantized at tighter INT4 scale due to the 36% variance reduction from removing the fold. The CF2 bound ensures ‖Δh_1^{dynamic}‖₂ is in the same relative range as other layers.

Does not rely on: MLP weight rank (CF8 — this is GEMV truncation by removing rows, not rank decomposition), cross-layer W_Q (killed), CF13-CF15.

### Residency arithmetic

Layer-1 W_gate, W_up folded-neuron removal: 36% × 6144 = 2218 rows removed from W_gate[L1] and W_up[L1], each of shape 6144×2048 bf16 = 24 MB total. After fold removal: 3926×2048 × 2 bytes × 2 matrices = 30.7 MB remaining (instead of 48 MB). Saving: 17.3 MB. Plus W_down column removal: 2218 columns → 6144-col × 2048-row to 3926-col × 2048-row bf16 = 15.3 MB saved. Total Layer-1 MLP residency reduction: 32.6 MB from a total of 72 MB for Layer-1 MLP → 45% saving on Layer-1 MLP.

At full 28-layer model (Qwen3-1.7B): total MLP weight ≈ 28 × 3 × 6144 × 2048 × 2 bytes ≈ 2.1 GB. Layer-1 saving (32.6 MB) = 1.5% of total MLP — small. The CF2 coupling adds the quantization tightening argument: storing Δh_1^{dynamic} at INT4 (vs BF16 for the residual stream) for the W_down GEMV output saves additional bandwidth. At 70B: Layer-1 saving scales to ~500 MB of a 100 GB+ model — still small per-layer, but the independence measurement (whether removing 36% of gate neurons at L1 is additive with other compressions) is load-bearing for any multi-path compression plan.

### Novelty gloss

Kill list closest: R4-A SDZC (killed globally; Layer-1 isolated 36% is real); Track A R2 LCAB (Layer-1 gate folding, deferred). Run_005 C4-GCQI tested Layer-1 gate fold × W_Q K=64 independence interaction. RKSC is structurally different: it introduces the CF2 × CF6 coupling — the residual cosine flatness bound (CF2) constrains the POST-fold Layer-1 MLP output variance, and the INT4 quantization tightening argument is derived from that constraint. GCQI tested independence of MLP and attention compressions; RKSC tests whether CF2's ‖Δh‖ bound allows INT4 quantization of the post-fold MLP output with less error than a naive INT4 schedule. Published closest: AERO (activation removal + fold); AERO requires retraining. RKSC is a zero-retraining partial fold using calibration means, grounded in CF2's variance bound.

### Smallest experiment

**Claim:** The per-channel output variance of Δh_1^{dynamic} (Layer-1 MLP output with 36% near-constant neurons' contribution subtracted) is ≤ 0.64 × per-channel output variance of the full Δh_1, averaged across calibration tokens.

Procedure: (a) forward pass 200 calibration tokens on Qwen3-1.7B-Base; (b) capture Δh_1 = h after Layer-1 MLP block − h before Layer-1 MLP block; (c) identify foldable neurons (std(silu(W_gate[i,:]·h)) < 0.05 per CF6, for 200 tokens); (d) compute μ_{gate,foldable} per CF6 neuron over the 200 tokens; (e) compute c_1 = W_down[:,foldable] · (μ_{gate,foldable} ⊙ mean(W_up[foldable,:]·H_calib)); (f) compute per-channel variance of Δh_1 and Δh_1 − c_1; compare. Runtime: ~30 min (hook forward passes on 1.7B).
GO threshold: mean per-channel variance of (Δh_1 − c_1) / Var(Δh_1) ≤ 0.68 (at least 32% variance reduction, slightly below the theoretical 36% due to cross-neuron correlations).
NO-GO structural finding: the foldable neurons' contribution to the Layer-1 MLP output is correlated with the non-foldable contribution (non-zero covariance), so the fold reduces variance by less than (foldable fraction)^2; CF2 and CF6 compose additively in the Layer-1 MLP block, not multiplicatively.

### Primary risk

The calibration-derived c_1 may not generalize from the 200-prompt corpus to arbitrary test tokens (the mean gate activation μ_{gate} may shift between calibration and test distributions). Mitigation: measure c_1 on 100 prompts, evaluate on the other 100; if the variance reduction is comparable across the two halves, generalization is confirmed. This is part of the smallest experiment.

---

## C-WKOL — W_K K=512 GO × Static-Outlier Identity as Joint Keys Precision Map (WKOL)

**C / Track B**

### Mechanism

Track B — compression. CF11 established W_K K=512 GO at ΔNLL=+0.29 nats (r_99/d ≈ 0.79); W_K is more compressible than W_Q (r_99/d ≈ 0.63 for W_Q). CF3 established that K=0.1% channels (2 of 2048) have Jaccard=0.718 — near-static across tokens. The composition claim is different from runs 001-005's W_Q outlier alignment: **the 2 static outlier channels affect W_K·h (key computation) in a measurably concentrated way, and whether those 2 channels fall inside or outside W_K's K=512 subspace determines whether W_K compression induces a systematic bias in the attention logits for high-magnitude tokens.**

**Equation (WKOL):** Let V_K^{512} ∈ ℝ^{d×512} be the top-512 right singular vectors of W_K^ℓ. Let e_{O} be the standard basis vectors for the 2 static outlier channels. The projection residual for W_K:

    ρ_K^ℓ = ‖(I − V_K^{512} V_K^{512T}) e_O‖₂

If ρ_K^ℓ ≈ 0 (outlier channels inside W_K K=512 subspace): the static outlier channels' contribution to the key vector is preserved at K=512 compression. If ρ_K^ℓ ≈ 1 (orthogonal): the outlier channels' contribution to keys is ENTIRELY LOST at K=512, producing a systematic per-query-token attention bias error concentrated on the 2 static channels, which appear at every token and accumulate over sequence length.

The composition: CF3 says the static outlier channels appear consistently at every token (high Jaccard=0.718). CF11 says W_K K=512 costs +0.29 nats globally. The coupling predicts: IF ρ_K^ℓ is large, the +0.29 nats is concentrated on tokens with high magnitude in the static outlier channels — these are NOT random errors, they are systematic attention distortions at every token involving the 2 channels. This makes the +0.29 nats cost CORRELATED with sequence position (appearing wherever the static channels dominate), not iid noise. The practical remedy: add a 2-column FP16 sidecar for W_K (like RAOK Tier-0 for activations, but applied to the key projection matrix itself). The sidecar is 2 × d_K × 2 bytes = 2 × head_dim × n_heads × 2 bytes per layer = negligible (≈0.2 MB at 1.7B), and the combined scheme may reduce the +0.29 nats substantially if the outlier channels are the main contributor.

Does not rely on: cross-layer W_Q (killed), MLP weight rank (CF8 killed), CF13-CF15.

### Residency arithmetic

W_K at K=512 for all 28 layers of Qwen3-1.7B: 28 × (2048×512 + 512×2048) × 2 bytes ≈ 112 MB vs full bf16 28 × 2048² × 2 = 457 MB → 4× compression saving 345 MB on W_K. The 2-column sidecar per layer: 28 × 2 × 2048 × 2 bytes = 0.22 MB. Net: 112 + 0.22 ≈ 112 MB vs 457 MB. Quality: if ρ_K^ℓ is large and the sidecar recovers the outlier contribution, ΔNLL at W_K K=512 + sidecar < +0.29 nats. If ρ_K^ℓ ≈ 0 (outliers inside subspace), the sidecar provides no benefit but also costs nothing significant — the scheme degrades gracefully.

At 70B: W_K at d=8192, K=1024 (extrapolating CF11 ratio): 80 × 2 × 8192 × 1024 × 2 bytes ≈ 2.68 GB. Sidecar: 80 × 2 × 8192 × 2 = 2.6 MB. Negligible overhead with potentially 0.29+ nats recovery.

### Novelty gloss

Kill list closest: the CF3 × CF11 alignment class (runs 001-005) — all prior runs measured this for W_Q (C-ZOIA in run_005, AQOD in run_004, OHAA/QAOC/JQSOH in runs 001-003). WKOL applies the same alignment measurement to W_K. The structural novelty is that W_K and W_Q have different spectra (r_99/d = 0.79 vs 0.63) and different GO thresholds (K=512 vs K=128), so the outlier-projection residual ρ_K^ℓ is predicted to be different from ρ_Q^ℓ. If ρ_K^ℓ > ρ_Q^ℓ (outliers more OUTSIDE W_K subspace), the sidecar argument is stronger for W_K than W_Q, and the combined W_Q K=128 + W_K K=512 + W_K 2-col sidecar scheme has a different quality profile than W_Q alone. Published closest: A3 (arXiv:2505.12942) covers W_Q; no published method applies the outlier-subspace alignment test to W_K specifically with a sidecar remedy.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base at L14, the W_K outlier-projection residual ρ_K^{L14} = ‖(I − V_K^{512} V_K^{512T}) e_O‖₂ ≥ 0.3 for at least one of the 2 static outlier channels (i.e., the outlier channels are NOT fully inside W_K's K=512 subspace, motivating the sidecar).

Procedure: (a) W_K[L14] SVD to get V_K^{512} (available from AQFKV R3-A data; W_K K=512 was measured); (b) outlier channel identity O from CF3 PDAP R2; (c) compute ρ_K^{L14} = ‖(I − V_K^{512} V_K^{512T}) e_O‖₂. Runtime: < 5 min. Extend to all 28 layers: < 20 min.
GO threshold: mean ρ_K^ℓ ≥ 0.3 across ≥ 20 of 28 layers (outliers substantially outside W_K K=512 subspace).
NO-GO structural finding: static outlier channels lie INSIDE W_K K=512 subspace (ρ_K^ℓ < 0.15 in all layers) — the W_K compression does NOT systematically distort attention at the outlier channels; CF3 and CF11 are decoupled for key computation; no sidecar benefit. This is a clean structural finding: W_K's K=512 subspace is "aware of" the static outlier channels.

### Primary risk

The outlier channel identity O is defined on the MLP block input activations (W_gate/W_up feed), not on the attention block input. The static outlier channels in the MLP input may differ from those in the attention key input (they share the same residual stream, but the K=0.1% static channels may vary between the two measurement points). Mitigation: re-measure the static K=0.1% outlier identity specifically on the attention-block input activations using the existing PDAP framework (≤ 2-hour extension).

---

## C-SPECQ — Spectral Rank × Functional Sensitivity Asymmetry as Mixed-Precision Law (SPECQ) [FREE SWING]

**C / Track B**

### Mechanism

Track B — compression. This idea composes three confirmed findings — CF11 (W_Q r_99/d≈0.63, W_K r_99/d≈0.79), CF12 (tied embed r_99/d≈0.97, ΔNLL=+19.96 at K=1024), and CF8 (MLP W_gate/W_up r_99/d≈1.0) — into a **cross-class spectral sensitivity law** that decouples spectral flatness from functional sensitivity. The four object classes exhibit a striking violation of the naive expectation "flatter spectrum → less compressible":

| Class | r_99/d | ΔNLL at 50% rank |
|---|---|---|
| Tied embed | 0.97 | +19.96 (catastrophic) |
| MLP W_up | ≈1.0 | +2.34 |
| MLP W_gate | ≈1.0 | +0.77 |
| W_K | 0.79 | +0.29 |
| W_Q | 0.63 | +0.03 (K=512) |

The tied embed is nearly flat-spectrum (r_99/d=0.97 ≈ W_up) but 8.5× more sensitive to rank reduction than W_up at 50% rank. The composition claim: **the observed sensitivity ΔNLL is not monotone in r_99/d** — it is determined by both the spectral flatness AND the degree of functional redundancy across weight classes. Formally, define the **sensitivity-rank ratio** SRR(class) = ΔNLL(50% rank) / (1 − r_99/d). For W_Q: SRR = 0.03 / (1 − 0.63) = 0.08. For tied embed: SRR = 19.96 / (1 − 0.97) = 665. This 8000× range in SRR implies the compression tolerance cannot be predicted from spectral flatness alone.

**Equation (SPECQ):** The coupling prediction is:

    SRR(class) = ΔNLL(50%) / (1 − r_99/d) = f(functional_position, gradient_path_count)

where functional_position is the role of the class in the forward pass (input embedding = 2 gradient paths per CF12; output projection = 1 path; MLP weight = 1 path) and gradient_path_count is the number of distinct gradient flows through the weight matrix during training. The hypothesis: SRR is monotone in gradient_path_count at training time. Tied embed has 2 paths (input lookup + output projection), giving the highest SRR. All other classes have 1 path.

This is a testable law: any new weight class with known gradient_path_count should have SRR proportional to that count. For Qwen3-8B with untied lm_head (1 gradient path), the lm_head SRR should be comparable to MLP weights (SRR ~1-10), NOT comparable to tied embed (SRR ~665). This is a novel cross-class quantization constraint: knowing a weight matrix's gradient-path count from the training configuration predicts its functional sensitivity at inference, independent of its spectral rank.

Does not rely on: CF13-CF15. Does not propose any specific compression — it's a structural law that constrains all future compression proposals.

### Residency arithmetic

This idea does not directly reduce residency. Its payoff is a quantization-schedule law: for any model, given the training configuration (tied vs untied embeddings, shared vs unshared weight blocks), predict the minimum safe bpw for each weight class from (r_99/d, gradient_path_count) without running per-class calibration experiments. At 70B scale, if W_V and W_O are separately measured (untested), their SRR can be predicted and the mixed-precision budget allocation can be automated.

Indirect residency payoff: if W_gate's SRR (lower than W_up's despite similar r_99/d) reflects a 3:1 gradient-path asymmetry (some optimizer paths weight W_gate less), then W_gate can be quantized more aggressively than W_up, yielding 0.2-0.3 bpw differential. At 70B across 80 layers of W_gate (shape 28672×8192): 0.3 bpw differential × 80 × 28672 × 8192 bits = 5.6 GB / 8 = 700 MB saving. This is speculative — the gradient-path count hypothesis must be confirmed first.

### Novelty gloss

Kill list closest: no kill-list item addresses the cross-class sensitivity law. The closest is CF8 (the class boundary finding, which established that MLP r_99/d≈1.0 while attention is concentrated) — SPECQ extends CF8 by adding functional sensitivity as a second axis orthogonal to spectral flatness, and including CF12 as the extreme outlier that reveals the decoupling. Published closest: mixed-precision quantization literature (OmniQuant, LLM-QP) calibrates per-layer bpw from Hessian sensitivity — SPECQ proposes a training-derived structural predictor that replaces Hessian calibration entirely. SPECQ's "gradient_path_count as sensitivity predictor" has no direct published analog; the closest is theory from Molchanov et al. (ICLR 2017) on Fisher information in pruning, which operates within one matrix class. The cross-class version bridging embedding tables, MLP weights, and attention weights is novel.

### Smallest experiment

**Claim:** The SRR of Qwen3-1.7B-Base W_gate (ΔNLL at 50% rank from R3 experiment / (1 − r_99/d_gate)) is ≤ 0.5 × SRR of W_up (ΔNLL at 50% rank from R4 experiment / (1 − r_99/d_up)), reflecting the W_up firing-dominance giving W_up higher functional sensitivity per unit spectral flatness.

Procedure: (a) SRR_gate = CF8_gate ΔNLL(K=1024) / (1 − r_99/d_gate); from R3 data: ΔNLL=+0.77, r_99/d≈1.0 (but r_99 ~2048 means 1 − r_99/d ≈ 0.0 — need the actual r_99/d measurement from the CF8 rank sweep). Use the measured K=1024 ΔPPL instead and compute SRR = ΔNLL(K=1024) / (K_drop_fraction) where K_drop_fraction = 0.5 (50% rank drop). SRR_gate = 0.77/0.5 = 1.54. SRR_up = 2.34/0.5 = 4.68. Ratio = 4.68/1.54 = 3.04. (b) Prediction: SRR_up / SRR_gate ≥ 2.0 (W_up is >2× more sensitive per unit compression). (c) Extend to W_Q (SRR_Q = 0.03/0.5 = 0.06) and tied embed (SRR_embed = 19.96/0.5 = 39.9). The full SRR table tests the monotonicity claim. Runtime: all data available from prior experiments; computation is arithmetic on existing results (< 5 min).
GO threshold: SRR_embed >> SRR_MLP >> SRR_attention, with at least 3 order-of-magnitude spread from W_Q to tied embed.
NO-GO structural finding: SRR is not monotone in gradient-path count (e.g., SRR_W_gate ≈ SRR_W_up despite different roles); the sensitivity-rank decoupling has no gradient-path explanation; the law requires an alternative structural predictor. This is still a structural finding even on NO-GO.

### Primary risk

The SRR metric as defined uses 50% rank reduction, but the functional sensitivity is highly nonlinear (CF12's +19.96 at 50% rank vs +0.03 for W_Q suggest sensitivity THRESHOLDS, not linear slopes). Using a single point (50% rank) to define SRR may obscure the threshold structure. Mitigation: compute SRR at multiple rank fractions (10%, 25%, 50%) to characterize the sensitivity curve shape per class; report whether the W_Q / MLP / embed sensitivity ordering is consistent across fractions.

---

## Convergence handles

1. **Per-layer W_Q rank-64 ΔNLL (single-layer replacement)** — load-bearing for C-LDOC. Requires three per-layer eval passes (L0, L14, L27). Same script reused for C-UFDM's subspace measurement validation.

2. **Principal-angle overlap between V_up^{128} (empirical firing-direction SVD) and V_Q^{128} (CF11 W_Q right SVD) at representative layers** — load-bearing for C-UFDM. Requires W_up empirical activation subspace (W_up · H_calib SVD) and W_Q SVD (AQFKV data). Single scipy call after SVDs.

3. **Per-channel output variance of Δh_1 before and after foldable-neuron subtraction** — load-bearing for C-RKSC. Requires Layer-1 hook forward passes on 200 calibration tokens, identification of foldable neurons per CF6.

4. **W_K outlier-projection residual ρ_K^ℓ per layer** — load-bearing for C-WKOL. W_K K=512 SVD available from AQFKV R3-A; outlier channel identity O from PDAP R2. Cheap computation (<20 min).

5. **SRR table (ΔNLL / rank-drop-fraction for all measured weight classes)** — load-bearing for C-SPECQ. All data already exists from prior experiments (R3 W_gate, R4 W_up, CF11 W_Q/W_K, CF12 tied embed). Assembly is arithmetic on existing measurements.
