# Stage 1 — Orientation C (Composition) — Run 005

Orientation: C — Composition
Run: 005 (independent cold start; does not inherit kills from runs 001–004)
Kill note: v2-CHEAP-TEST-001 kills the entire cross-layer W_Q shared-basis class (CQBAL, CQST, CQSGC, XLQB variants). All ideas below avoid that dead family. Per-layer K=128 (CF11) is the W_Q ceiling; no cross-layer stacking.
Ideas produced: 6

---

## C1 — Outlier-Tier Deep-Layer Spread as Dynamic bpw Switch (ODSP)

**Tags:** C, Track B

### Mechanism

**Track B — compression.** CF3 establishes the K-dependent Jaccard structure: K=0.1% channels Jaccard=0.718 (channel-static), K=1% Jaccard=0.308 (token-dynamic), K=5-10% Jaccard≈0.25 (token-dynamic). The SUMMARY also records a within-CF3 measurement: deep layers (23-27) have 14-18% of channels covering 90% of outlier events, while early/mid layers (2-19) have only 5-8%. This depth gradient is confirmed empirical structure within CF3. CF11 establishes per-layer W_Q spectrum concentration with K=128 GO at ΔNLL=+0.98 nats. The composition claim:

**Equation (C1):** Let spread_ℓ = fraction of channels covering 90% of outlier events at layer ℓ (5-8% at layers 2-19, 14-18% at layers 23-27). Define the per-layer quantization safe-zone width as:

    bpw_safe(ℓ) = bpw_max × (1 - spread_ℓ / d)

The coupling: layers where outlier events are concentrated (small spread_ℓ) tolerate lower bpw on activation inputs without increasing quantization error on the dominant channels. Deep layers where spread_ℓ is 2-3× larger require either higher bpw OR per-token dynamic scaling on a wider channel set. Specifically: for W_down inputs (which are post-SwiGLU activations), the effective signal energy in the top K channels is higher when spread_ℓ is small — the quantization grid can be coarser. Formally: the per-token quantization MSE for a K-channel INT4 scheme scales as Var(activation) / 2^{4×2} ≈ 0.0039 × Var, while for the 14-18% deep-layer spread, the per-channel variance is spread across 3× more channels, implying each channel's individual variance is lower — but the INT4 per-tensor scale is set by the MAX channel, which is now in a larger set. The composition produces a per-layer bpw schedule that tightens (3 bpw or lower) for layers 2-19 and uses 4-5 bpw for layers 23-27, driven entirely by the empirical outlier-spread depth profile from CF3.

**Does not rely on:** MLP weight rank structure (CF8), cross-layer W_Q sharing (killed v2-CHEAP-TEST-001), CF13-CF15.

### Residency arithmetic

Target: Qwen3-1.7B-Base MLP input activations (W_down GEMV inputs); treating this as activation quantization schedule affecting GEMV compute bandwidth.

- Layers 2-19 (18 layers): spread_ℓ ≈ 5-8%, outlier spread is tight → 3 bpw activation quantization feasible. W_down input dimension: 6144 per token per layer.
- Layers 23-27 (5 layers): spread_ℓ ≈ 14-18%, wider → 4 bpw required.
- Mixed schedule for 25 active layers: (18 × 3 + 5 × 4 + 5 × 5) / 28 ≈ 3.4 bpw average on MLP activations vs 8 bpw (BF16 baseline) or 4 bpw (INT4 baseline).

Activation bandwidth for 1.7B W_down GEMV at 4K context: 28 layers × 6144 × 4K tokens × 4 bpw = 336 MB at INT4 baseline. At 3.4 bpw: 286 MB. Savings: 50 MB on activations at 4K context. Modest for 1.7B, but at 32K context: 2.7 GB → 2.3 GB — 400 MB KV-activation saving.

Quality cost: not estimable without experiment; the coupling claim is that depth-stratified bpw follows the outlier-spread depth profile from CF3, reducing quantization error at the same average bpw relative to a uniform schedule.

### Novelty gloss

Kill list closest: MDL-Selected Per-Layer bpw (R2/S2 killed as converging to mixed-precision). Structural difference: ODSP does not use MDL or calibration fitting — the bpw schedule derives from the empirical outlier-spread depth profile within CF3, which is a weight-free measurement (activation statistics only, already collected in R2). The schedule is calibration-free. Closest published method: SmoothQuant / OS+ per-layer sensitivity; those calibrate via Hessian or activation range. ODSP uses the Jaccard-derived spread statistic as the sensitivity proxy — a compositional claim that spread_ℓ predicts quantization sensitivity better than activation range alone.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base, the per-layer channel spread (fraction of channels covering 90% of outlier events) is monotonically higher in layers 23-27 than in layers 2-19, with a ratio ≥ 2.0.

Procedure: from R2 PDAP dataset (200 prompts, 7 sampled layers), extract per-layer outlier-spread metric; interpolate to all 28 layers via a ≤1-hour additional run adding layers 23-27 and 2-5 to the PDAP script. Compute spread_ℓ for all sampled layers. Spearman ρ between layer depth and spread_ℓ.
Runtime: ≤ 2 hours (reuse R2 PDAP data + 1-hour extension run on Ryzen 5 7530U).
**GO threshold:** Spearman ρ ≥ 0.6 AND spread(layers 23-27) / spread(layers 2-10) ≥ 2.0.
**NO-GO structural finding:** outlier spread is uniform across depth — the depth-gradient in SUMMARY's within-CF3 note is a sampling artifact from the 7-layer probe; per-layer bpw scheduling from CF3 is not motivated.

### Primary risk

The depth-gradient may be real but too small to justify bpw differentiation (3 bpw on layers 2-19 may still cause quality loss). Mitigation: the smallest experiment measures the spread ratio first; if ratio < 2.0, a per-layer bpw schedule is not motivated and the idea stops.

---

## C2 — Tied-Embed Static-Outlier Column Identity (TESOC)

**Tags:** C, Track B

### Mechanism

**Track B — compression.** CF12 establishes that Qwen3-1.7B-Base uses `tie_word_embeddings=True`: the embed/lm_head matrix E ∈ ℝ^{151936×2048} is full-rank (r_99=1992/2048), with catastrophic sensitivity to truncation (ΔNLL=+19.96 at r=1024) because gradient flows through both input lookup and output projection paths. CF3 establishes that in the residual stream, K=0.1% channels (≈2 channels of d=2048) have Jaccard=0.718 — these are channel-static outliers across tokens. The composition claim:

**Equation (C2):** Let O ⊂ [2048] be the 2 static outlier channel indices (K=0.1%, Jaccard=0.718). In the tied embed matrix E, the columns E[:, O] are accessed at every forward pass token (because the residual stream at every layer carries large magnitudes in those channels, and the lm_head projection W_out = E^T always reads from all columns). The hypothesis: **the columns E[:, O] account for a disproportionate share of the lm_head logit variance**:

    σ²_{logit, O} / σ²_{logit, total} >> |O| / d    (i.e., 2/2048 ≈ 0.001)

If this ratio is ≥ 0.05 (50× overrepresented), then the 2 static-outlier columns of E are load-bearing in the output distribution. This does NOT allow truncating them (CF12 prohibits truncation). Instead, the coupling produces a **quantization priority constraint**: the 2 outlier columns E[:, O] must be stored at higher precision (BF16) even when the rest of E is quantized to INT4, because their logit contribution is disproportionate AND their token-static nature means the quantization error accumulates coherently across tokens (no averaging out).

Practical consequence: quantizing E at INT4 except columns O at BF16 costs 2 × 151936 × 2 bytes = 608 KB overhead vs 2 × 151936 × 0.5 bytes. The net saving: INT4-quantize 2046 of 2048 columns → (2046/2048) × 151936 × 2048 × 0.5 bytes = ~300 MB vs full BF16 590 MB — saves ~290 MB at the cost of a 608 KB BF16 sidecar. Quality: the outlier columns are preserved at full precision; all other columns at INT4. Expected ΔNLL << CF12's catastrophic +19.96 (which was full-rank truncation, not quantization), because INT4 quantization of 2046 columns is additive noise, not rank truncation.

**Why non-obvious:** CF12 says E is full-rank and catastrophically sensitive to truncation. CF3 says 2 channels dominate per-token outlier structure. Neither finding alone implies the 2 outlier CHANNELS in the residual stream correspond to 2 outlier COLUMNS in E that dominate logit variance. The connection requires tracing the outlier channel identity from the residual stream through the lm_head projection.

**Does not rely on:** MLP rank structure (CF8), cross-layer W_Q (v2-CHEAP-TEST-001 killed), CF13-CF15.

### Residency arithmetic

- E at BF16 (baseline): 151936 × 2048 × 2 bytes = 590 MB.
- E at INT4 (2046 cols) + BF16 (2 cols): 151936 × 2046 × 0.5 + 151936 × 2 × 2 bytes ≈ 147.9 MB + 0.58 MB = 148.5 MB.
- Saving: 590 MB → 148.5 MB = **4.0× compression on the embed/lm_head**. This is the largest single-matrix saving available without truncation.
- CF12 prohibits rank truncation; TESOC uses quantization (not truncation) plus a 2-column sidecar. The 4× compression is achievable IF the quality cost of INT4 on 2046 columns is acceptable.
- At 70B with untied lm_head: the composition still applies to the input embedding (large matrix), and the untied lm_head may or may not have static-outlier columns — that is a separate experiment.

Quality cost: INT4 quantization of 2046 embed columns adds noise of magnitude ~0.01 × ‖E[:, j]‖ per column per token (rough INT4 error estimate). Per-token logit noise accumulates across 2046 columns; rough ΔNLL estimate: ≤ 0.5 nats (well below CF12's catastrophic +19.96, which was rank truncation). This is an estimate — the smallest experiment provides the number.

### Novelty gloss

Kill list closest: CF12 itself killed SVD truncation of tied lm_head (LHQD killed, SVLD killed, ENVC killed). TESOC uses quantization (not truncation) and adds the CF3 coupling to identify the 2 columns that must stay at BF16. No kill-list item addresses hybrid-precision quantization of the embed matrix informed by the outlier-channel identity. Closest published method: SpQR (arXiv:2306.03078) uses mixed-precision quantization with sensitive-weight isolation; TESOC's novelty is that the sensitive-column identity comes from CF3's activation-outlier measurement (not from weight-sensitivity calibration), and applies to the tied embed/lm_head specifically (where SVD truncation is catastrophic, so the motivation for careful column-level precision management is stronger than in standard weights).

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base, the 2 static-outlier residual-stream channels (K=0.1% from CF3) correspond to embed matrix columns whose contribution to output logit variance is ≥ 5× overrepresented: σ²_{logit, O} / σ²_{logit, all} ≥ 2 × (2/2048) = 0.002 (i.e., the 2 columns contribute ≥ 0.2% of logit variance despite being 0.1% of columns).

Procedure: (a) identify O (2 static outlier channels from CF3 PDAP dataset — already known); (b) load E (embed/lm_head); (c) compute per-column logit variance contribution: for 200 calibration tokens, accumulate E[:, j] @ h_final^T for each j; measure variance of E[:, O] contribution vs total; ~15 min.
**GO threshold:** σ²_{logit, O} / σ²_{logit, all} ≥ 10 × (2/2048) = 0.01 (10× overrepresentation; conservative).
**NO-GO structural finding:** outlier residual-stream channels do NOT correspond to disproportionate logit variance in E; the CF3 outlier channel identity is an activation-space property, not a weight-space property; CF3 and CF12 are decoupled; the 2-column BF16 sidecar provides no quality benefit over uniform INT4.

### Primary risk

The 2 static outlier channels in the residual stream may be outliers in intermediate activations but not in the final residual (post-layer-norm) that feeds lm_head. Mitigation: measure at the final LayerNorm output (pre-lm_head) specifically; the outlier identity may shift or persist at that stage.

---

## C3 — W_Q Per-Layer K=128 Subspace × Outlier-Channel Alignment for Zero-Overhead INT8 Attention (ZOIA)

**Tags:** C, Track A

### Mechanism

**Track A — arch-transposition.** CF11 establishes per-layer W_Q K=128 GO (ΔNLL=+0.98 nats) — the 16-head redundancy means all queries live in a 128-dim subspace. CF3 establishes K=0.1% static outlier channels (Jaccard=0.718). The composition claim (different from runs 001-002's version, which tested alignment angle; this version tests functional consequence):

**Equation (C3):** Let P_Q^ℓ ∈ ℝ^{d×128} be the top-128 right-singular vectors of W_Q^ℓ. Define the outlier-projection residual at layer ℓ:

    ρ_ℓ = ‖(I - P_Q^ℓ P_Q^{ℓT}) e_{O_ℓ}‖₂

where e_{O_ℓ} is the unit vector along the static outlier channel at layer ℓ. ρ_ℓ = 0 means the outlier channel IS inside the W_Q^ℓ subspace (full alignment). ρ_ℓ = 1 means it is orthogonal (no alignment). The composition: **if ρ_ℓ ≤ 0.3 in most layers, the W_Q K=128 projection already "captures" the outlier channel — meaning computing attention with quantized (INT8) residual-stream inputs loses almost nothing for the query computation, because the only high-magnitude input direction (the outlier) lives within the projected subspace**. This is the "zero-overhead" claim: we do not need to handle the outlier channel separately for the W_Q GEMV; it's already inside the K=128 subspace and the INT8 quantization error on the outlier direction is absorbed by the projection.

The arch-transposition: replace W_Q^ℓ with its rank-128 approximation and compute Q^ℓ = W_Q^{ℓ,128} × h_ℓ^{INT8} (where h_ℓ^{INT8} is the residual stream quantized uniformly to INT8 with a single per-vector scale). The static outlier channel does NOT need a BF16 sidecar for W_Q because the projection subspace absorbs its magnitude. This removes the INT16/BF16 bypass buffer that the run_001-run_002 proposals assumed was necessary.

Combined saving: W_Q at K=128 (8× attention-weight saving) + input activations at INT8 (2× vs BF16, with no outlier exception handling). The computation graph changes: standard BF16 W_Q GEMV → rank-128 projected GEMV with INT8 input. No new bookkeeping.

**Does not rely on:** cross-layer W_Q sharing (killed), MLP rank structure (CF8), CF13-CF15.

### Residency arithmetic

- W_Q at K=128 for all 28 layers of Qwen3-1.7B: 28 × (2048×128 + 128×2048) × 2 bytes ≈ 28 MB vs 28 × 2048² × 2 bytes ≈ 457 MB BF16 — 16× saving on W_Q alone.
- INT8 vs BF16 activation input to W_Q: per token, 2048 bytes (INT8) vs 4096 bytes (BF16) — 2× bandwidth reduction on activation reads.
- W_Q GEMV at K=128: input is 2048 INT8 → 128 FP32 output per head via two matrix multiplies. Total FLOP per layer drops from 16 × 128 × 2048 × 2 = 8.4M to 2048 × 128 + 128 × 2048 = 0.5M — 17× FLOP reduction on the query projection.
- 70B scale: W_Q at d=8192, 64 heads, K=512 (CF11 GO at +0.20 nats): saves ~14 GB of W_Q in BF16 → ~0.5 GB at K=512 rank representation.
- Quality: IF ρ_ℓ ≤ 0.3 confirms the outlier is inside the K=128 subspace, ΔNLL ≈ +0.98 nats (CF11 baseline at K=128) with no additional outlier-handling cost. If ρ_ℓ > 0.3 (outlier outside subspace), INT8 quantization of the outlier channel induces an additional ≤ 0.3 nat penalty (rough estimate).

### Novelty gloss

Kill list closest: run_001/run_002 Cluster C1 variants that tested cross-layer W_Q sharing — those are killed. ZOIA is per-layer (no cross-layer claim) and tests whether W_Q INT8 activation quantization eliminates outlier-exception overhead, not whether W_Q subspaces are shared. The distinction is sharp: ZOIA's claim is about the projection residual ρ_ℓ (how much outlier energy lies OUTSIDE the K=128 subspace), which has the same per-layer SVD as CF11 but asks a different question. Published closest: A3 (arXiv:2505.12942) minimizes attention-logit error for W_Q compression; ZOIA asks whether INT8 activation inputs compose cleanly with W_Q K=128 projection when the outlier channel aligns with the subspace. The "zero INT16 sidecar" claim has no published neighbor.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base, the outlier-projection residual ρ_ℓ = ‖(I − P_Q^ℓ P_Q^{ℓT}) e_{O_ℓ}‖₂ ≤ 0.3 for ≥ 50% of layers.

Procedure: (a) W_Q SVD per layer (top-128 right singular vectors; already-run in prior experiments); (b) outlier channel identity O_ℓ from CF3; (c) compute ρ_ℓ for each layer — inner product and norm subtraction, ≤ 5 min total.
**GO threshold:** mean ρ_ℓ ≤ 0.3 AND ρ_ℓ ≤ 0.3 in ≥ 14 of 28 layers.
**NO-GO structural finding:** outlier channels are substantially outside the W_Q subspace (ρ_ℓ > 0.3 in most layers) — the INT16 bypass sidecar IS needed; W_Q K=128 and outlier channel handling are independent structures that cannot be collapsed; the zero-overhead claim fails.

### Primary risk

The per-layer outlier channel O_ℓ may be layer-specific (different layers have different top-outlier channels), making the alignment measurement layer-specific. If O_ℓ changes across layers, the "single static outlier ID" assumed in CF3's description needs per-layer identity. Mitigation: use the per-layer outlier channel identity from CF3 data directly (R2 PDAP measured 7 layers; extend to all 28 layers as part of the ODSP experiment above).

---

## C4 — Gate-Near-Constancy × W_Q Compression Independence at Layer 1 (GCQI)

**Tags:** C, Track A

### Mechanism

**Track A — arch-transposition.** CF6 establishes Layer-1 has 36.1% of gate neurons foldable (std < 0.05), with all other layers < 2.5%. CF11 establishes global W_Q K=128 GO (ΔNLL=+0.98 nats) without layer-1-specific measurement. The composition: **Layer-1 MLP and Layer-1 attention W_Q can be simultaneously compressed, and their quality costs are additive rather than multiplicative, because the two paths share only the residual-stream input but otherwise compute through disjoint subgraphs.**

**Equation (C4):** Let Δ_gate(L1) be the NLL cost of folding F_1 gate neurons in Layer 1. Let Δ_WQ(L1, K=64) be the NLL cost of W_Q rank-64 at Layer 1 only (more aggressive than global K=128, testing the pre-context hypothesis — Layer 1 processes token embeddings before contextual attention has accumulated). Independence claim:

    |Δ_{joint}(L1) - Δ_gate(L1) - Δ_WQ(L1, K=64)| < ε

where ε is the interaction penalty. If ε < 0.2 nats, the two compressions are functionally independent and can be applied jointly. The structural argument for independence: the SwiGLU MLP sub-block and the multi-head attention sub-block in a transformer layer take the same residual stream input but compute through entirely separate weight matrices (W_gate, W_up, W_down vs W_Q, W_K, W_V, W_O). Their outputs add to the residual stream independently. The interaction term ε captures only the indirect coupling through the shared residual input h_{L1} (both paths see the same BOS+token-embedding residual, which is relatively simple at Layer 1 vs deep layers).

Additionally: if Layer-1 W_Q tolerates K=64 (more aggressive than global K=128) because Layer-1 queries are operating on token-embedding-level features (not yet context-shaped), the joint compression saves extra vs global treatment. The "pre-context hypothesis" is testable independently.

**Does not rely on:** cross-layer W_Q sharing (killed), MLP weight rank structure for non-Layer-1 layers, CF13-CF15.

### Residency arithmetic

Layer-1 gate folding (36% of 6144 neurons = 2218 neurons): W_gate^{L1} rows eliminated (replaced by scalar means) → saves 2218 × 2048 × 2 bytes ≈ 9.1 MB. Same for W_up^{L1}: +9.1 MB. Total Layer-1 MLP gate fold: ~18 MB.

Layer-1 W_Q at K=64 (vs global K=128): 2048 × 64 × 2 × 2 bytes = 0.5 MB vs 2048 × 128 × 2 × 2 bytes = 1 MB. Additional saving from K=64 over K=128: 0.5 MB on Layer-1 W_Q — negligible at 1.7B.

At 70B: Layer-1 savings scale as d² = (8192/2048)² = 16×. Layer-1 MLP gate fold: ~290 MB. Layer-1 W_Q K=64 vs K=128: ~8 MB. Total Layer-1 joint: ~300 MB. Not transformative at 70B scale alone; the **discovery value** is the independence measurement: if confirmed, it motivates iterating across all 28 layers for whatever L_{gate}(ℓ) values exist.

### Novelty gloss

Kill list closest: R4-A SDZC (killed as global scheme; Layer-1 isolated saving is real but small). Track A R2 LCAB (Layer-1 specific gate folding, deferred). GCQI is structurally different from LCAB: GCQI composes Layer-1 gate folding WITH Layer-1 W_Q compression and tests the independence interaction term ε. LCAB targeted only gate. The independence measurement at Layer 1 is the novel falsifiable claim — it provides the interaction coefficient that would be needed if any future multi-path compression proposal were to compose MLP and attention compressions at the same layer.

### Smallest experiment

**Claim (two-part):** (a) Layer-1 W_Q K=64 achieves ΔNLL < 0.7 nats; (b) the joint ΔNLL of Layer-1 gate fold + Layer-1 W_Q K=64 is within 0.2 nats of Δ_gate + Δ_WQ (additivity holds).

Procedure: (1) rank-64 SVD replacement of W_Q[L=1] only; eval on WikiText-2 512 tokens; (2) Layer-1 gate fold (std < 0.05 neurons → constant); eval; (3) both simultaneously; eval. Three evals on one model, ~15 min total.
**GO threshold:** (a) Δ_WQ(L1, K=64) < 0.7 nats; (b) |Δ_joint - Δ_gate - Δ_WQ| < 0.2 nats.
**NO-GO structural finding:** either (a) Layer-1 W_Q is NOT more compressible than global mean (pre-context hypothesis fails), OR (b) the interaction term ε is large (> 0.2 nats) — the two paths are NOT independent at Layer 1, meaning some coupling through the shared residual input matters; this produces the interaction coefficient as a structural finding.

### Primary risk

Layer-1 gate folding may require knowing the mean activation per folded neuron exactly; runtime estimation of μ_{F_1} from a calibration pass adds calibration-data dependency. Mitigation: use the mean over the R2 PDAP calibration tokens (200 prompts); if the calibration mean generalizes to WikiText-2 eval tokens, the folding is robust.

---

## C5 — Residual-Cosine Flatness Implies KV Recompute Delta Bound (RKDB)

**Tags:** C, Track B [FREE SWING]

### Mechanism

**Track B — compression.** CF2 establishes cos(h_L, h_{L+1}) ≈ 0.99 across all layers of Qwen3-0.6B (confirmed at 1.7B in KILL_LIST Round 1 structural finding 2). CF11 establishes W_K K=512 GO (ΔNLL=+0.29 nats), W_K r_99/d ≈ 0.79. The composition:

**Equation (C5):** cos(h_ℓ, h_{ℓ+1}) = 0.99 implies:

    ‖h_{ℓ+1} - h_ℓ‖₂ = ‖h_ℓ‖₂ × √(2(1 − 0.99)) ≈ 0.141 × ‖h_ℓ‖₂

Let K_ℓ(t) = W_K^ℓ · h_ℓ(t) be the key for token t at layer ℓ. Then:

    ‖K_{ℓ+1}(t) − W_K^{ℓ+1} · h_ℓ(t)‖₂ / ‖K_{ℓ+1}(t)‖₂ ≤ ‖W_K^{ℓ+1}‖_op × 0.141 × ‖h_ℓ‖₂ / ‖K_{ℓ+1}(t)‖₂

**The composition:** CF2 gives the δ norm; CF11 gives that W_K is not full-rank (r_99/d=0.79, K=512 GO), so W_K's operator norm is dominated by 79% of its singular values. The key recomputation approximation K_{ℓ+1}^{approx}(t) = W_K^{ℓ+1} · h_ℓ(t) (using the previous layer's residual instead of the true h_{ℓ+1}) has a relative error bounded above by a function of W_K^{ℓ+1}'s operator norm and CF2's δ. If W_K is compressed to K=512 (CF11 GO), its operator norm is reduced by 21% relative to full BF16 W_K, which ALSO reduces the recomputation error bound by 21%.

**Practical mechanism:** for long-context inference (>16K tokens), store K/V at even layers only and recompute odd-layer K/V on-the-fly from the adjacent stored layer's residual. The computation cost (one W_K GEMV per recomputed layer per decode step) is bounded by CF11's W_K K=512 GO — the compressed W_K is used for both storage and recomputation, sharing the same rank-512 factorization. The two findings compose: CF2 bounds the recomputation error, CF11 provides the compressed W_K that further tightens the bound AND is already stored efficiently.

**Why non-obvious:** CF2 and CF11 address different objects (residual stream vs W_K spectral structure). Composing them to bound KV recomputation error — where the bound tightens because the compressed W_K has a smaller operator norm — is not entailed by either finding alone.

**Does not rely on:** MLP rank structure (CF8), cross-layer W_Q sharing (killed), CF13-CF15.

### Residency arithmetic

Target: 70B model, context=32K tokens.
- KV cache at INT8: 80 layers × 2 × 32K × (8 KV-heads × 128-dim) × 1 byte ≈ 41 GB. This alone exceeds 7.28 GiB target.
- Storing every other layer: ~20 GB. Still too large.
- At K=512 W_K compression for recompute: the operator norm reduction also allows lower-precision storage of cached keys (INT4 instead of INT8), since the recompute error already accounts for 21% operator norm reduction. Combined: K/V at every other layer at INT4 ≈ 10 GB at 32K context. Closer but still above DRAM.
- At 4K context (short-context baseline): KV ≈ 5 GB at INT8 for 70B. Half-layer storage at INT4: ~1.25 GB. This fits in DRAM alongside quantized model.
- For 1.7B at 4K: KV at INT8 ~230 MB. Half-layer INT4: ~57 MB. Savings ~170 MB — modest but clean.

Quality cost: CF11 W_K K=512 ΔNLL = +0.29 nats (already measured). Additional cost from layer-skip recomputation: bounded by the relative error equation; requires empirical measurement from the smallest experiment.

### Novelty gloss

Kill list closest: CLASE (Track A R3 deferred, cross-layer KV aliasing with INT4 deltas; deferred for bottleneck mismatch at <32K context). RKDB differs: CLASE stores deltas; RKDB stores nothing for skipped layers and recomputes at inference time. The recompute-error bound is derived from CF2's δ norm AND CF11's operator norm — the dual-CF composition is the novel coupling. Closest published method: "Layer-Condensed KV Cache" (arXiv:2405.10637) reuses KV across layers at training time. RKDB uses no-retraining empirical bounds (CF2 + CF11) to justify the same approximation post-hoc. The operator-norm reduction from CF11 W_K K=512 providing a tighter recomputation error bound is the novel composition.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base with W_K compressed to K=512, the relative recomputation error ‖K_{ℓ+1}^{approx}(t) − K_{ℓ+1}^{true}(t)‖₂ / ‖K_{ℓ+1}^{true}(t)‖₂ ≤ 0.15 for ≥ 60% of (layer, token) pairs.

Procedure: forward pass 100 tokens on Qwen3-1.7B-Base; capture h_ℓ and h_{ℓ+1} at all layers; compute K_{ℓ+1}^{true} = W_K^{ℓ+1,512} · h_{ℓ+1} and K_{ℓ+1}^{approx} = W_K^{ℓ+1,512} · h_ℓ using K=512 truncated W_K; measure relative error per (layer, token).
Runtime: ≤ 25 min.
**GO threshold:** mean relative error ≤ 0.15 AND 60% of (layer, token) pairs below 0.15.
**NO-GO structural finding:** the δ_ℓ direction, though small in angle (CF2), is amplified by W_K^{ℓ+1,512} — the compressed W_K's operator norm is not reduced enough to bound the recomputation error; CF2 + CF11 do NOT compose into a useful KV-skip bound; the two findings are independent along this composition axis.

### Primary risk

The recomputation adds one W_K GEMV per skipped layer per decode step; if the decode latency is DRAM-bound (which it is at 70B), adding one GEMV may save DRAM bandwidth (no KV read) but adds compute latency. The net benefit depends on whether GEMV latency < NVMe read latency per token. Mitigation: measure one W_K^{512} GEMV latency on Ryzen 5 7530U vs one 32K-context NVMe KV read; if GEMV < read, the trade is favorable.

---

## C6 — Deep-Layer Outlier Spread × W_Q Compressed Subspace Dim as Attention-Compression Floor (DSAF)

**Tags:** C, Track B

### Mechanism

**Track B — compression.** CF3 establishes the K-dependent Jaccard structure AND the depth-gradient (layers 23-27 have 14-18% channel spread for 90% of outlier events vs 5-8% in early/mid layers). CF11 establishes W_Q at K=128 is GO globally (ΔNLL=+0.98) with r_99/d ≈ 0.63. The composition claim is a **floor on the effective dimensionality required to represent attention inputs at each layer**:

**Equation (C6):** The W_Q K=128 compression is a 16:1 head-redundancy finding. The outlier-channel spread at layer ℓ (fraction of channels covering 90% of outlier events) represents the effective dimension of the input distribution's "interesting" subspace at that layer. Define:

    effective_dim(ℓ) = spread_ℓ × d

where spread_ℓ is CF3's depth-gradient measurement. For layers 2-19: effective_dim ≈ 0.07 × 2048 ≈ 143. For layers 23-27: effective_dim ≈ 0.16 × 2048 ≈ 328. The coupling claim:

    If K_Q^ℓ (rank of W_Q at layer ℓ from CF11) < effective_dim(ℓ), then the W_Q compression is discarding input dimensions that carry outlier-event information.

For layers 2-19: K_Q=128 > effective_dim=143 is approximately equal — no discarding. For layers 23-27: K_Q=128 < effective_dim=328 — W_Q at K=128 may be insufficient for deep layers, which need K_Q ≥ 328 to avoid compressing away outlier structure. This gives a per-layer W_Q K schedule derived purely from CF3 and CF11:

- Layers 2-19: W_Q K=128 (as per CF11 global GO) — safe.
- Layers 23-27: W_Q K ≥ 256 or K=512 (to accommodate wider outlier spread) — higher bpw.

The composition does NOT use cross-layer sharing (killed). It uses per-layer CF3 depth measurements to refine the CF11 global K=128 GO into a per-layer K schedule.

**Does not rely on:** cross-layer W_Q (killed), MLP rank (CF8), CF13-CF15.

### Residency arithmetic

Qwen3-1.7B W_Q total at K=128 global: 28 × 2048 × 128 × 2 × 2 bytes ≈ 28 MB.

DSAF per-layer schedule: layers 2-19 (18 layers) at K=128, layers 23-27 (5 layers) at K=256, layers 20-22 (3 layers) at K=192 (interpolated):
- 18 × 2048 × 128 × 2 × 2 = 18.9 MB
- 5 × 2048 × 256 × 2 × 2 = 10.5 MB
- 3 × 2048 × 192 × 2 × 2 = 4.7 MB
Total: 34.1 MB vs 28 MB at uniform K=128. **DSAF actually costs more than uniform K=128** in residency — the value is quality: it avoids the quality degradation at deep layers that uniform K=128 may incur.

The payoff is ΔNLL reduction: if uniform K=128 costs +0.98 nats globally but costs more at layers 23-27 (where effective_dim > K=128), the DSAF schedule should improve to ≤ +0.6 nats by spending 6 MB more on deep-layer W_Q. This is a quality vs residency tradeoff — the metric is: at fixed total W_Q budget of 34 MB, is DSAF better than uniform K=128 for all layers?

At 70B scale (d=8192, 64 heads, deep layers likely have even wider outlier spread): DSAF gives a principled per-layer K schedule instead of blind uniform K=512. The saving vs uniform-high-K: if layers 1-40 of a 80-layer 70B model tolerate K=128 and layers 60-80 need K=512, uniform K=512 wastes 80 × 8192 × 512 × 2 × 2 bytes = 1.3 GB vs DSAF 40 × 8192 × 128 + 20 × 8192 × 512 = ~500 MB → saves 800 MB at 70B scale.

### Novelty gloss

Kill list closest: MDL-Selected Per-Layer bpw (R2/S2 killed for converging to mixed-precision). Structural difference: DSAF derives K per layer from the CF3 outlier-spread depth gradient (a within-CF3 measurement already in SUMMARY.md) and CF11's per-layer W_Q spectral structure — no MDL, no calibration fitting (CF10 safe), no per-tensor sensitivity Hessian. The coupling is: activation-side outlier spread (CF3) constrains the minimum viable attention-projection rank (CF11). That constraint arrow — from activation statistics to weight compression rank — has no published analog. Published closest: mixed-precision quantization schedules (LLM-QP, OmniQuant); DSAF is not a bpw schedule but a rank schedule, and the determining factor is the CF3 outlier-spread depth gradient rather than calibrated Hessian proxies.

### Smallest experiment

**Claim:** In Qwen3-1.7B-Base, W_Q K=128 at layers 23-27 alone (with all other layers at K=128) produces higher ΔNLL than the global K=128 baseline would predict from a uniform-per-layer contribution assumption, by ≥ 0.2 nats.

Procedure: (a) baseline: W_Q K=128 at all layers (CF11 global, ΔNLL=+0.98 nats); (b) partial: W_Q K=256 at layers 23-27 only, K=128 elsewhere; eval ΔNLL; compare to a per-layer contribution model (expected saving ≈ (5/28) × (ΔNLL_{K=256} − ΔNLL_{K=128}) from 5 layers). If deep layers contribute disproportionately more than their 5/28 fraction to the +0.98 nats total, the outlier-spread hypothesis is supported.
Runtime: two eval passes, ~20 min total.
**GO threshold:** ΔNLL with K=256 at layers 23-27 < ΔNLL_{K=128 global} − 0.2 nats (i.e., the 5 deep layers account for > (0.2/0.98) × 5/28 = their proportional share of the quality cost, confirming that deep-layer K is the binding constraint).
**NO-GO structural finding:** deep layers contribute proportionally to the total ΔNLL — W_Q quality cost is uniformly distributed across layers; CF3 outlier-spread depth gradient does NOT predict W_Q compression sensitivity; the two findings are independent along the "rank budget" axis.

### Primary risk

The CF3 depth-gradient measurement was from 7 sampled layers in R2; the deep-layer spread numbers (14-18%) are from those 7 layers but may not be precisely localized to layers 23-27 vs 20-22. Mitigation: run ODSP's smallest experiment (which extends the PDAP script to all 28 layers) before DSAF's experiment; use the full per-layer spread_ℓ as input to DSAF's K schedule.

---

## Convergence handles

1. **Per-layer outlier-channel spread_ℓ (fraction of channels covering 90% of outlier events, all 28 layers)** — load-bearing for C1 (ODSP bpw schedule), C6 (DSAF rank schedule). A single extension of the R2 PDAP script to all 28 layers produces this primitive.

2. **Outlier residual-stream channel identity O_ℓ per layer** — load-bearing for C2 (TESOC embed-column logit variance), C3 (ZOIA projection residual ρ_ℓ). Exists in R2 PDAP dataset for 7 layers; full 28-layer sweep needed.

3. **W_Q top-128 right-singular vectors per layer (P_Q^ℓ)** — load-bearing for C3 (ZOIA ρ_ℓ computation), C6 (DSAF K-schedule validation). Same 28-SVD pass resolves both; already partially available from prior rounds (run_001 R3-A AQFKV).

4. **Layer-1 gate-fold ΔNLL (Δ_gate(L1)) and W_Q K=64 ΔNLL at Layer 1** — load-bearing for C4 (GCQI interaction term ε). Three-eval script resolves the interaction measurement.

5. **W_K K=512 operator norm relative to full BF16 W_K** — load-bearing for C5 (RKDB recomputation error bound). Derivable from W_K SVD already measured in CF11; the operator norm is the largest singular value, accessible in ≤ 2 min from the R3-A AQFKV data.

6. **Tied embed column logit-variance contribution per column** — load-bearing for C2 (TESOC outlier-column identification). New measurement, ~15 min on Qwen3-1.7B-Base.
