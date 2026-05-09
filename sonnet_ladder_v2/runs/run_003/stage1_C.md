# Stage 1 — Composition Orientation — Run 003

Orientation C. 6 ideas. Kill-list compliance: v2-CHEAP-TEST-001 (cross-layer W_Q shared basis) is dead — none of the below propose cross-layer W_Q stacked SVD. Per-head W_Q truncation is dead — not proposed here.

---

## C1-QAOC — Q-Activation Outlier Channel Coupling

**C + Track B**

### Mechanism

Track B — compression. Couples CF11 (W_Q global K=128 head-redundancy: the 16 query heads collectively span a ~128-dim subspace of the 2048-dim row space) with CF3 (K=0.1% activation outlier channels are static at Jaccard 0.718; K=1% is already dynamic at 0.308). The coupling claim: the 2–4 static outlier channels at K=0.1% that CF3 identifies ALSO appear as high-variance directions in the W_Q global K=128 subspace. If so, a joint coordinate system simultaneously captures (a) the static outlier channels for activation pinning and (b) the load-bearing W_Q directions for attention, and the joint projection is cheaper than two independent ones.

Formally: let `D_static` ⊆ ℝ^2048 be the span of the top-0.1% static outlier channels (dim ~2–4 per CF3). Let `V_Q` ∈ ℝ^{2048×128} be the global W_Q K=128 right singular basis (per CF11). The coupling claim:

    dim(D_static ∩ colspan(V_Q)) ≥ 0.8 × dim(D_static)          (Eq. C1)

If Eq. C1 holds: the K=128 W_Q basis already encodes the static outlier directions "for free." This means static-outlier channels can be pinned using the same low-rank projection the compressed W_Q already computes — no separate outlier-masking step needed. The runtime consequence: fused Q-projection + outlier-detection at 128 GEMV flops instead of 2048 + outlier-scan.

This does NOT reduce W_Q residency further (CF11 already found K=128 is the ceiling for compression). The gain is that the K=128 W_Q subspace makes activation quantization cheaper: after applying V_Q, the static outliers live in a predictable 2-4 of the 128 new coordinates, not scattered across 2048 original channels.

Not CF8 MLP rank reduction. Not cross-layer W_Q sharing (killed, v2-CHEAP-TEST-001).

### Residency arithmetic

W_Q at K=128 is already a 2× attention-weight residency reduction per CF11 (+0.98 nats, tolerable). The present idea does not change that. Operational gain: the hybrid outlier-pinning scheme (PDAP-style: top-0.1% static + top-1% dynamic) becomes computable in the K=128 projected space instead of the full 2048-dim space. For GEMV on Qwen3-1.7B-Base (d=2048, 28 layers, 16 heads per layer, W_Q shape 2048×2048): if the 2–4 outlier directions are captured by the K=128 projection, the per-token outlier scan drops from O(2048 × n_tokens) to O(128 × n_tokens) — a 16× reduction in scan cost. This is a compute win, not a residency win.

For 70B-class target: the same coupling — if it holds — means a single K=512-projected attention kernel handles both W_Q compression and activation outlier routing, replacing two separate pipelines.

### Novelty gloss

Closest kill-list item: v2-CHEAP-TEST-001 (cross-layer W_Q sharing) — structurally different: this is within-layer coupling between W_Q column space and activation outlier channels, not cross-layer stacking. Closest published method: SmoothQuant / LLM.int8() handle activation outliers independently of attention weight structure; AQFKV (our CF11 experiment) handled W_Q compression independently of outlier dynamics (CF3). The composition — that the W_Q low-rank basis subsumes the static outlier basis — has no published antecedent we know of.

### Smallest experiment

**Claim:** For each of the 28 layers of Qwen3-1.7B-Base, at least 80% of the top-0.1% static outlier channel directions (by per-channel variance across the PDAP calibration corpus) lie within the top-128-singular-vector colspan of W_Q for that layer.

**Procedure:** (1) From the PDAP run_002 calibration data, compute per-channel variance of the residual stream h_L across 200 prompts; take top-4 channels by variance as `D_static`. (2) Load W_Q[L], compute SVD, take V_Q[:, :128]. (3) For each channel index in D_static, project the unit vector e_c onto V_Q V_Q^T; record ‖V_Q V_Q^T e_c‖². (4) Report mean across layers.

**Runtime:** ~20 minutes on Ryzen 5 7530U (SVD of 28 W_Q at shape 2048×2048 in bf16; PDAP variance is already computed).

**Go threshold:** mean projection score ≥ 0.80 across layers. If Eq. C1 holds at this threshold, the fused pipeline is motivated. **No-go finding:** "W_Q subspace and activation outlier channels are geometrically orthogonal at K=128" — this would mean activation quantization schemes and attention compression are independent problems on this architecture (itself a novel structural finding).

### Primary risk

W_Q row space concentrates on token-routing directions that are orthogonal to activation-outlier channels (which are shaped by W_up·x dominance per CF1, not by W_Q). **Mitigation:** Run the projection measurement before any implementation; the 20-minute experiment produces a go/no-go with clean structural interpretation on either outcome.

---

## C2-LAYERFOLD — Layer-1 Gate Fold × W_Q Head-Redundancy Cascade

**C + Track A**

### Mechanism

Track A — arch-transposition. Couples CF6 (layer-1 has 36% foldable gate neurons — uniquely exploitable; all other layers <2.5%) with CF11 (W_Q global K=128 head-redundancy across all 28 layers). The coupling claim: layer-1's structural anomalies should be exploitable at a different compression ratio than layers 2–27, AND the attention-side (W_Q at K=128) and MLP-side (gate folding at 36%) reductions at layer 1 are independently motivated and compose without destructive interaction if they target different weight tensors in the same layer.

The coupling equation: let `s_1` = fraction of foldable neurons at layer 1 (measured 0.36 per CF6). Let `r_Q` = compression ratio of W_Q at K=128 (8× per CF11). The joint saving at layer 1 is:

    bytes_saved(layer_1) = s_1 × d_FFN × bpw_gate + (1 - 1/r_Q) × 4 × d_model² × bpw_Q      (Eq. C2)

For Qwen3-1.7B-Base: d_FFN=6144, d_model=2048, bpw=2 bytes (bf16). Gate fold saving: 0.36 × 6144 × 2 ≈ 4.4 KB (tiny but nonzero). W_Q fold saving: (7/8) × 4 × 2048² × 2 ≈ 28 MB at 70B scale. The MLP contribution is microscopic on a 1.7B model but the question is whether layers 2–27 have any analogous gate-fold opening, and whether the CF6 anomaly at layer 1 predicts an attention-side anomaly too.

The deeper composition hypothesis: CF6 layer-1 anomaly (gate near-constancy) and CF11 (W_Q low effective dimensionality) both reflect the same underlying fact — layer 1 performs a lower-complexity function than later layers (early embedding normalization / input routing rather than deep contextual reasoning). If this is true, W_K and W_V at layer 1 might also have more concentrated spectra than at layer 27. This is a testable cross-finding prediction.

Coupling claim (Eq. C2b): The ratio r_99/d for W_Q at layer 1 is ≤ 0.40 (vs global average ~0.63 from CF11 per L14 measurement), because the same mechanism causing 36% gate foldability also reduces W_Q effective rank at layer 1.

### Residency arithmetic

If Eq. C2b holds and W_Q at layer 1 supports K=64 (vs global K=128 ceiling): layer-1 W_Q at 64 = 2048×64×2 bytes = 256 KB vs full 2048×2048×2 = 8 MB. Saving at layer-1 W_Q alone: 7.75 MB. Multiplied across hypothetical similar anomaly at layers 2–3 (untested): ~20 MB total on a 1.7B model. This is not a residency play on 1.7B; the arithmetic motivation is as a pattern — at 70B (d_model=8192, 80 layers, W_Q=8192²×2=134 MB per layer), even one layer being compressible to K=32 saves 117 MB.

The MLP gate-fold at layer 1 (36% foldable) saves: on 1.7B, 0.36 × 6144 × 2 = 4.4 KB. The W_gate matrix for one layer is 6144×2048×2=25 MB; foldable neurons can have their rows replaced by a constant vector, saving ~0.36×25=9 MB. Total layer-1 MLP+attn saving on 1.7B if both findings compose: ~17 MB (negligible at this scale). The proposal motivates itself by the cross-layer pattern-detection experiment rather than immediate residency payoff.

### Novelty gloss

Closest kill-list item: R4-A/SDZC (gate folding as global compression) — killed for being microscopic globally. This proposal is structurally different: it composes CF6 with CF11 to test whether layer-1's structural anomaly is a *general signal* that all layer-1 weight matrices (not just W_gate) are more compressible. The coupling is cross-tensor within layer 1, not MLP-only. Closest published method: "early-layer pruning" (e.g., Sheared LLaMA, depth pruning) removes entire early layers rather than testing within-layer cross-tensor spectrum heterogeneity.

### Smallest experiment

**Claim:** W_Q at layer 1 of Qwen3-1.7B-Base has r_99/d ≤ 0.50 (more concentrated than the L14 measurement of ~0.63 in CF11).

**Procedure:** Load W_Q[0] (layer 1 in 0-indexed), compute SVD, record r_99 and var@128. Compare to L14 values from CF11. Runtime ~3 minutes.

**Go threshold:** r_99 < 1050 (vs 1293 at L14) AND dNLL at K=64 < +0.50 nats (replicate AQFKV at layer 1 only). **No-go finding:** "layer-1 anomaly in gate space does not generalize to attention space" — implies CF6 and CF11 measure truly independent structural properties.

### Primary risk

Layer 1 W_Q may be as full-rank as layer 14 — CF6's gate-fold anomaly may be purely SwiGLU-specific (the gating function near-constancy in layer 1 is a trained artifact of early-token role, not a general compression signal). **Mitigation:** The smallest experiment runs in 3 minutes and yields a definitive answer before any implementation.

---

## C3-KDJB — K-Dependent Jaccard × W_K Spectrum Bridge

**C + Track B**

### Mechanism

Track B — compression. Couples CF3 (K-dependent outlier dynamicity: at K=1% Jaccard=0.308, meaning ~70% of the top-1% outlier channel set rotates per token) with CF11 extended to W_K (W_K global K=512 ΔNLL=+0.29 nats; W_K r_99/d ≈ 0.79). The coupling claim: the per-token rotation of outlier channels in the residual stream is CAUSALLY connected to the per-token variation in the dominant singular directions of the W_K-projected key vectors. If the key vectors' dominant directions vary with the same pattern as the outlier channels, then outlier-channel identity is a proxy for key-subspace occupancy — and a low-overhead key-subspace predictor falls out of the outlier-channel detector already needed for activation quantization (PDAP/RAOK).

Coupling equation: let `J(t)` = Jaccard of top-1% outlier channels at token t vs token t-1. Let `θ_K(t)` = principal angle between the top-64-singular-direction subspace of W_K · h_t and W_K · h_{t-1}. The coupling claim:

    ρ(J(t), cos(θ_K(t))) > 0.50     across tokens in the PDAP calibration corpus      (Eq. C3)

where ρ is Spearman rank correlation. If Eq. C3 holds: when the outlier channel set is stable (J≈0.72 at K=0.1%, which happens on "easy" continuation tokens), the key subspace is also stable, and the K=512 W_K truncation at those tokens loses essentially no quality. On "hard" tokens (J≈0.18 for massive-activation pairs per PDAP), the key subspace rotates too, and full W_K is needed.

This couples the activation-quantization router (already motivated by CF3) with a token-conditional W_K compression scheme: use W_K at K=512 on low-J tokens (cheap) and full W_K on high-J tokens (quality-preserving). The gate is the outlier-channel Jaccard, computed as a byproduct of the activation-quantization pipeline.

Not CF8 MLP rank reduction. Not cross-layer W_Q sharing. The W_K compression at K=512 is per-layer, not cross-layer — no v2-CHEAP-TEST-001 collision.

### Residency arithmetic

W_K at K=512 is a 2× reduction (per CF11: ΔNLL=+0.29 at K=512). If 60% of tokens are "low-J" (rough estimate from CF3: K=0.1% Jaccard=0.718 suggests high-stability fraction ~50–70%), then on average W_K bandwidth per token drops to 0.6×(0.5×full) + 0.4×full = 0.70× full. On Qwen3-1.7B-Base: W_K is 2048×2048×2 bytes × 28 layers = 235 MB. At 0.70× bandwidth: effective 165 MB per token sweep. DRAM bandwidth win at 11.5 GB/s: 165/11500 = 14.3 ms vs 20.4 ms for full W_K — 1.43× speedup on W_K specifically. In the full MLP+attn stack W_K is ~12% of total weight bandwidth, so total speedup ~1.05×. Small but additive with W_Q compression.

At 70B scale: W_K = 8192×1024×2×80 layers ≈ 1.34 GB. Token-conditional 2× on 60% of tokens → effective 960 MB. At NVMe 3 GB/s bottleneck this saves 0.13 s per full-stack sweep. Modest but no quality cost on 60% of tokens.

### Novelty gloss

Closest kill-list item: R2/S2 Bloom-Filter KV Retirement (kv-side, prediction-historical) — killed for wrong bottleneck. This proposal is structural-compression of W_K (a weight matrix), not KV cache retirement. The gate is the activation outlier Jaccard, not a frequency histogram. Closest published method: Scissorhands / StreamingLLM / H2O manage KV cache eviction; none gate W_K compression on per-token outlier Jaccard from the residual stream. The Jaccard-as-quality-proxy for key-subspace stability is the novel composition.

### Smallest experiment

**Claim:** Spearman ρ(J(t), cos(θ_K(t))) > 0.50 across the PDAP calibration corpus, where J(t) is top-1% outlier Jaccard between consecutive tokens and θ_K(t) is the principal angle between W_K · h_t and W_K · h_{t-1} key subspaces.

**Procedure:** On Qwen3-1.7B-Base, for 200 PDAP prompts: at each layer, compute (1) consecutive-pair top-1% Jaccard of |h_L| (already computed in CF3/PDAP run), (2) W_K · h_t for each token, top-64 SVD of stacked key vectors per window, principal angle between consecutive windows. Correlate. Runtime: ~40 minutes (reuses PDAP infrastructure).

**Go threshold:** Spearman ρ > 0.50 at ≥ 20 of 28 layers. **No-go finding:** "outlier-channel Jaccard is not a proxy for key-subspace stability" — implies CF3 and W_K spectrum behavior are decoupled, which also is a structural finding about the independence of activation and weight-matrix concentration.

### Primary risk

The principal-angle computation per token is O(d × K²) = O(2048 × 512²) ≈ 537M FLOPs per layer per token — may be prohibitive to compute at inference time, defeating the purpose even if the correlation holds. **Mitigation:** If ρ > 0.50 is confirmed, replace the full principal-angle gate with a cheap proxy (e.g., ℓ2 norm of the top-8 outlier channels suffices if norm is also correlated with θ_K) and verify the proxy correlation separately.

---

## C4-GEMBED — Tied-Embedding Gauge × W_Q Subspace Alignment

**C + Track A**

### Mechanism

Track A — arch-transposition. Couples CF12 (tied embed/lm_head is full-rank, r_99=1992/2048, catastrophic ΔNLL at r<2048) with CF11 (W_Q global K=128 captures ~6% of the row space but preserves quality to within +0.98 nats). The coupling claim: the tied embedding matrix W_E (shape 151936×2048) acts as both the token-to-residual input map AND the residual-to-logit output map. The W_Q global K=128 projection V_Q ∈ ℝ^{2048×128} defines the "live" subspace of query computation. If the columns of W_E (in embedding space) are approximately aligned with V_Q in the sense that:

    ‖(I - V_Q V_Q^T) W_E^T W_E V_Q‖_F / ‖W_E^T W_E V_Q‖_F < ε_align         (Eq. C4)

for ε_align < 0.20 (i.e., V_Q's directions are approximately eigenvectors of the embedding covariance W_E^T W_E), then the first attention layer's Q-projection can be absorbed INTO the embedding lookup itself — a token's "query component" is pre-computed at embedding time, not at attention time. This is an AERO-style algebraic fold: the embedding already applies a linear map to the token index, and V_Q is just another linear map composed on top. If the embedding covariance is approximately diagonal in V_Q's coordinates (Eq. C4), the two maps commute up to ε_align and the fold is lossless.

This does NOT change residency of W_E (CF12 guarantees full-rank, cannot be truncated). The gain is compute: the W_Q[layer=0] GEMV (2048×2048) is replaced by a table lookup into a pre-computed 128-dim query embedding (shape 151936×128×2 = 38.5 MB, computable at model-load time). At 70B scale W_Q[0] is 8192×8192×2 = 134 MB; the table is 151936×128×2 = 38.9 MB. Saves 95 MB of first-layer query memory bandwidth per token. More important: the query for layer 0 is then O(1) — a table lookup indexed by token id — instead of O(d²) GEMV.

### Residency arithmetic

On Qwen3-1.7B-Base: W_Q[layer=0] = 2048×2048×2 = 8 MB. Precomputed table: 151936×128×2 = 38.9 MB (larger! — bad trade). On 70B (d_model=8192, vocab=151936): W_Q[0] = 8192×8192×2 = 134 MB. Table: 151936×128×2 = 38.9 MB. Net saving at layer 0: 95 MB. For 80 layers: only layer 0 is foldable (input embedding only touches layer 0's residual stream). So saving is 95 MB flat at 70B, out of ~40 GB total weights — ~0.24%. Small at 70B.

The compute win is more interesting: first-layer Q-projection at 70B is 8192²=67M multiplications per token, replaced by a 128-int table lookup (1 memory access). At 3 tok/s (NVMe-bound), this saves ~22 ms per token from the first-layer Q-projection alone.

### Novelty gloss

Closest kill-list item: none directly — LLSC (lm_head SVD with RAM-pin smaller factor) was deferred in Track A R2 and targets lm_head compression, not embed-Q fusion. Closest published method: "Flash-Decoding" / "Flash-Attention" optimize attention kernel layout; they do not fold embedding lookup into query precomputation. "Prompt caching" precomputes KV for fixed prefixes; this precomputes Q for individual token embeddings. The algebraic fold via Eq. C4 alignment is the novelty.

### Smallest experiment

**Claim:** For Qwen3-1.7B-Base layer-0 W_Q, the ratio ‖(I - V_Q V_Q^T) W_E^T W_E V_Q‖_F / ‖W_E^T W_E V_Q‖_F < 0.20.

**Procedure:** Load W_Q[0] (2048×2048), compute top-128 right singular vectors V_Q. Load W_E (151936×2048), compute W_E^T W_E V_Q (via two GEMM passes; never form the 2048×2048 covariance explicitly — use 151936×128 intermediate). Compute residual fraction. Runtime: ~15 minutes (bottleneck is 151936×2048 matrix operations in bf16 on CPU).

**Go threshold:** residual fraction < 0.20. **No-go finding:** "embedding covariance and W_Q[0] subspace are geometrically misaligned" — implies W_Q is trained to attend to directions NOT well-represented by the embedding distribution, which is itself a structural insight about how layer-0 attention function diverges from embedding statistics.

### Primary risk

W_E^T W_E is high-rank (151936 rows) and approximately isotropic in the 2048-dim residual stream; embedding covariance has no reason to align with W_Q's top singular vectors. **Mitigation:** The 15-minute measurement settles this before any implementation.

---

## C5-WUPDOWN — W_up Full-Rank × W_down Spectrum Bridge

**C + Track B**

### Mechanism

Track B — compression. Couples CF5 (W_up MORE rank-sensitive than W_gate: dNLL +2.34 vs +0.77 at K=1024) with CF8's extended empirical structure: W_gate and W_up are both full-rank, and the asymmetry in rank-sensitivity is a trained property, not a random one. The coupling claim: if W_up dominates the firing-rank signal (CF1) AND is more rank-sensitive (CF5), then W_down (the projection back from intermediate to residual space) MUST compensate — specifically, the column space of W_down^T (i.e., the row space of W_down, shape d_FFN × d_model) should be more concentrated than W_up^T, because W_down applies the LAST linear transform before residual addition, and the residual stream's near-constancy (CF2: cos=0.99) implies W_down maps intermediate activations to a narrow subspace of the residual.

Coupling equation: let r_99(W) = the rank retaining 99% of spectral energy of W. The coupling prediction:

    r_99(W_down^T) / d_model  <  r_99(W_up^T) / d_FFN      (Eq. C5)

i.e., W_down^T is more spectrally concentrated (relative to its dimensions) than W_up^T. Rationale: W_up·x is a discriminative map (full-rank because it must distinguish firings across the d_FFN neurons); W_down takes the post-activation vector back to residual space (d_model=2048), and the residual stream's near-constancy constrains how much of d_FFN's dimensionality survives. The rank ceiling is d_model=2048, not d_FFN=6144.

If Eq. C5 holds AND r_99(W_down^T) / d_model < 0.80: W_down admits post-training rank truncation at K < d_model that W_up does not, because the output space is the bottleneck. This would be the first MLP-side matrix in Qwen3 that admits structural compression — exploiting the asymmetry CF5 hints at via the output-space argument.

### Residency arithmetic

W_down shape: d_FFN × d_model = 6144×2048, same byte count as W_gate and W_up at bf16 (25.2 MB per layer, 706 MB for 28 layers on 1.7B). If r_99(W_down^T)/d_model ≈ 0.70 and truncation at K=1433 gives ΔNLL < +0.30 nats: W_down at rank-1433 = 6144×1433×2 + 1433×2048×2 = 17.6+5.9 = 23.5 MB per layer, saving 1.7 MB per layer (6.8%). Not large. But: the asymmetry claim is load-bearing because it would be the FIRST demonstration that the CF8 full-rank finding has a direction — W_up > W_gate > W_down in rank-sensitivity — and W_down is the exploitable end. At 70B scale: W_down = 28672×8192×2 per layer ≈ 471 MB × 80 layers = 37.7 GB. Even 10% savings = 3.7 GB — meaningful at the 7.28 GiB target.

### Novelty gloss

Closest kill-list item: CF7/CF8 kill the class "no-SGD low-rank decomposition of MLP weights." This proposal directly challenges whether that kill extends to W_down by deriving a structural argument (output-space bottleneck from CF2's residual near-constancy) that predicts W_down is categorically different from W_up and W_gate. The coupling CF2 × CF5 → W_down exception hypothesis is the novel move. Closest published method: AQLM/GPTVQ/SqueezeLLM all apply quantization to all FFN matrices uniformly without exploiting inter-matrix rank asymmetry within one FFN block.

### Smallest experiment

**Claim:** r_99(W_down[L]^T) / d_model < r_99(W_up[L]^T) / d_FFN across ≥20 of 28 layers.

**Procedure:** For each of the 28 layers, compute SVD of W_down (6144×2048 — smaller than W_up's 2048×6144 in both SVD cost and r_99 ceiling since rank ≤ min(2048,6144)=2048). Record r_99. Compare to CF5's W_up r_99 values (already known from CLUBS/R4-B experiments: W_up is approximately full-rank up to its smaller dimension). Runtime: ~25 minutes for SVD of all 28 W_down matrices.

**Go threshold:** Eq. C5 holds for ≥20/28 layers AND r_99(W_down^T)/d_model ≤ 0.80. **No-go finding:** "W_down is also full-rank relative to its output dimension" — this would confirm CF8 extends to all three FFN matrices, which is the clean negative but closes a structural question the ladder has not yet answered.

### Primary risk

W_down's output dimension IS d_model=2048, but its input dimension is d_FFN=6144, so W_down^T has shape 2048×6144 — the rank is bounded by 2048. The argument that "residual near-constancy constrains W_down's effective rank" may be wrong because CF2 (cos=0.99) reflects the scale of the residual increment, not the rank of W_down. Even full-rank W_down produces small-magnitude outputs that appear as a tiny perturbation to the residual. **Mitigation:** The SVD experiment directly measures whether the spectrum is concentrated regardless of whether the theoretical argument is correct.

---

## C6-SPECDRIFT — Per-Layer Spectral Drift × K-Tier Codebook Locality [FREE SWING]

**C + Track B**

### Mechanism

Track B — compression. [FREE SWING] — this idea does not directly compose two confirmed CFs; it proposes an interaction between a MEASURED PATTERN (attention weight spectra vary per layer from CF11's L14 spot measurement) and a COMPRESSION PRIMITIVE (vector quantization codebook locality). The coupling claim: layers with more concentrated W_Q spectra (lower r_99) should support lower-bpw codebook quantization with smaller codebooks, because their weight rows inhabit a lower-dimensional effective manifold. Layers with less concentrated spectra need larger codebooks. A layer-stratified codebook where codebook size is proportional to r_99(W_Q[L]) achieves the same reconstruction error as a uniform codebook at lower mean bpw.

Let r[L] = r_99 of W_Q[L]. The per-layer codebook size K[L] = K_base × (r[L] / max_L r[L]). The coupling claim:

    ΔNLL(layer-stratified codebook) < ΔNLL(uniform codebook at same mean bpw)      (Eq. C6)

This is a composition of (a) the spectral heterogeneity of W_Q across layers (which CF11 established at a single layer L14 but has NOT been swept across all 28 layers) and (b) the standard VQ-codebook locality property (codebook quality scales with effective dimensionality of the data manifold). The composition predicts that per-layer spectrum heterogeneity, if it exists, directly implies per-layer codebook sizing is optimal — not a heuristic but a derived allocation.

The "FREE SWING" flag is because Eq. C6 depends on whether layer-stratified W_Q spectra actually vary significantly across the 28 layers (untested: CF11 measured only L14). If the spectrum is uniform across layers, Eq. C6 trivially holds at zero gain.

Not CF8 MLP compression. Not cross-layer W_Q shared basis (v2-CHEAP-TEST-001). The codebooks are per-layer — they are not shared across layers.

### Residency arithmetic

W_Q at K=128 SVD is already 8× compression per CF11. Codebook quantization on top of the K=128 reduced matrix: if W_Q is stored as V_Q (128-column basis, 2048×128×2 = 0.5 MB) plus coefficients (2048 rows × 128 dims × 1.5 bpw = 48 KB), total = 0.55 MB per layer vs 8 MB full. The present idea refines the coefficient bpw allocation by per-layer r_99 rather than fixing it at 1.5 bpw uniformly. Gain: 10–30% further reduction in coefficient storage if r[L] varies by 2× across layers. On 1.7B full model: W_Q total 28 × 8 MB = 224 MB → with K=128: 28 × 0.55 MB = 15.4 MB → with per-layer codebook: ~12–14 MB. Against 3 GB total model: 1–2 MB gain.

At 70B: W_Q = 80 × 134 MB = 10.7 GB. With K=512 SVD: 80 × (8192×512 + 512×8192)×2 = 80 × 16.8 MB = 1.3 GB. Per-layer codebook gain: 10–20% = 130–260 MB. Meaningful at 7.28 GiB target.

### Novelty gloss

Closest kill-list item: R3/S2 MDL-Selected Per-Layer bpw (killed for converging to mixed-precision territory with published prior). This proposal differs: it targets only W_Q, derives the per-layer allocation FROM the spectral concentration rather than from MDL/Hessian heuristics, and composes with CF11's already-measured K=128 basis structure. The allocation rule r[L]/max r is a closed-form prediction, not a calibration-fit parameter. Closest published method: GPTQ/AQLM use Hessian-weighted per-layer quantization; SpQR uses per-row sensitivity. None allocate codebook size based on the matrix's spectral r_99 relative to a per-layer sweep.

### Smallest experiment

**Claim:** W_Q r_99 varies by ≥ 1.5× across the 28 layers of Qwen3-1.7B-Base (prerequisite for any allocation gain).

**Procedure:** SVD of all 28 W_Q matrices (28 × 2048×2048 bf16; ~30 minutes). Record r_99[L] and var@128[L] per layer. Plot the distribution.

**Go threshold:** max(r_99[L]) / min(r_99[L]) ≥ 1.5. If yes: proceed to test Eq. C6 on W_Q VQ at matched mean bpw. **No-go finding:** "W_Q spectral concentration is uniform across all 28 layers" — means CF11's L14 measurement is representative and per-layer allocation buys nothing. This is also a structural finding (the head-redundancy property is layer-invariant).

### Primary risk

If r_99 is uniform (which is plausible given the residual stream architecture's layer-homogeneity), Eq. C6 gives zero gain and this idea dies on the prerequisite measurement. **Mitigation:** The SVD sweep is a byproduct of the AQFKV follow-up (CF11 opened "W_V and W_O spectra / per-layer W_Q sweep") and can be done as a free rider on that experiment — 30 minutes of shared compute.

---

## Convergence handles

- **W_Q top-K singular subspace** (V_Q ∈ ℝ^{2048×128}): load-bearing in C1, C4, C6; convergence signal if other orientations independently surface V_Q as a useful projection coordinate
- **Per-layer spectral r_99 sweep of all 28 W_Q** (not yet measured; CF11 is a single-layer spot): prerequisite for C6, informative for C2 and C3; cheap 30-minute shared experiment
- **Activation outlier channel identity as routing key** (CF3 Jaccard at K=0.1% vs 1%): coupling anchor in C1 and C3; if Unconventional or Reach orientations independently surface the same outlier-channel identity, strong convergence
- **W_down spectral structure** (untested; composition of CF2 + CF5 argues for relative concentration): C5's falsifiable prediction; first MLP-side matrix that may admit structural compression
- **Layer-1 anomaly as cross-tensor signal** (CF6 gate fold predicts CF11 attention fold at layer 1): C2's coupling experiment; if multiple orientations converge on layer-1 structural anomaly, the signal generalizes
- **Embedding covariance alignment with W_Q[0] subspace** (Eq. C4): C4's coupling; if First-Principles orientation derives the same alignment via a gauge-fixing argument, strong convergence signal
