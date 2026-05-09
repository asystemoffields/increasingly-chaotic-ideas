# Stage 1 — Orientation C (Composition) — Run 004

Orientation: **C — Composition**. Find non-obvious mechanisms that emerge from the interaction of two or more empirical findings. Each proposal is a coupling claim: *if CF_i AND CF_j hold simultaneously on the same model, then a third structure C is forced.*

Hard kills honored: v2-CHEAP-TEST-001 / Cluster C1 (cross-layer W_Q subspace sharing class). Per-layer K=128 (CF11) is the W_Q ceiling; cross-layer stacking is dead.

---

## C-AQOD — Attention-Outlier Disjoint Partition

**C / Track B**

### Name
Attention-Outlier Disjoint Partition (AQOD) — C / Track B

### Mechanism
Track B. CF11 establishes that W_Q has r_99/d ≈ 0.63 and admits global 8× compression at K=128 (ΔNLL=+0.98). CF3 establishes that at K=1% of d_model=2048 (~20 channels), the activation outlier set Jaccard ≈ 0.31 across consecutive tokens — substantially token-dynamic. The composition claim is: the ~20 persistent outlier channels (the static K=0.1% tier from CF3, Jaccard=0.718) and the per-layer W_Q compressed subspace (rowspan of W_Q at K=128) are **structurally disjoint in residual-stream coordinates**, because W_Q's low-rank subspace is shaped by query signal while the 0.1% outlier channels are dominated by magnitude extremes. If disjoint: the outlier channels can be pinned at FP16 (static tier, 2 channels × 28 layers × 2048 = 115 KB negligible), W_Q applied in its compressed K=128 basis, and the outlier channels' contribution recovered by a cheap additive correction. The correction is: `q = W_Q_K128 · (h − Π_static · h) + W_Q_FP16 · (Π_static · h)` where Π_static is the projection onto the 2 static outlier channels. This is a Track B restructuring of the attention weight application path, not a compute graph change.

**Coupling equation**: `dim(span(W_Q^{K128}) ∩ span(Π_static)) ≤ ε × K128` for ε < 0.05, asserting near-disjointness. If this holds, the compressed W_Q path loses no accuracy from ignoring the 2 static channels (already FP16-handled separately). If it fails (high overlap), the W_Q low-rank subspace actively encodes the static outlier directions — useful in its own right as a structural finding.

Grounding: CF11 (W_Q spectrum), CF3 (outlier static tier K=0.1%). Does not rely on CF13-15. Does not rely on cross-layer W_Q structure (killed). Does not require MLP rank reduction (CF8 killed).

### Residency arithmetic
Target: 1.7B DRAM-resident baseline. W_Q per layer: 2048×2048 bf16 = 8 MB × 28 = 224 MB total for all W_Q. At K=128 global (CF11): 2048×128 + 128×2048 = two factor matrices, 1 MB/layer → 28 MB total. Saving: 224 MB − 28 MB = 196 MB on W_Q. Quality cost already measured: ΔNLL = +0.98 nats at K=128 (CF11). The composition contribution: if the outlier-disjointness holds, the +0.98 is the ceiling; the partition recovers some of that because the static outlier directions are no longer asked to live inside K=128. Estimated ΔNLL with partition: +0.6 to +0.8 nats (rough; depends on how much of the +0.98 error was in the outlier directions). For a 70B model at equivalent ratio: W_Q ≈ 8× larger → 1.57 GB → 196 MB. Meaningful but not the dominant bottleneck (MLP weights dominate). The composition is a correctness-improving amendment to CF11's W_Q compression, not a primary residency lever.

### Novelty gloss
Closest kill-list item: C1 cluster (cross-layer sharing) — this is per-layer and relies on activation-stream disjointness, not cross-layer basis alignment. Closest published method: SmoothQuant / OS+ handle static outliers for activation quantization, not for W_Q projection correctness. The structural novelty is the disjointness *between* the static outlier subspace and the W_Q low-rank subspace — a composition of two orthogonal finding-classes (activation dynamics CF3, attention spectrum CF11) nobody has coupled.

### Smallest experiment
**Claim**: `cos²(e_i, V_{128})` < 0.05 for all i in the static outlier channel set, where V_{128} is the top-128 right singular vectors of W_Q at each layer of Qwen3-1.7B, and e_i are the standard basis vectors for the 2 static outlier channels identified by CF3.

Runtime: 20 min. Load W_Q for all 28 layers (bf16), compute top-128 SVD per layer (already done in AQFKV), identify the 2 top-K=0.1% channels from PDAP run, compute alignment angle.

Go threshold: mean cos² < 0.05 across all 28 layers → disjoint, partition is structurally clean.
No-go: if cos² ≥ 0.2, the W_Q low-rank subspace actively encodes the outlier directions — which itself is a structural finding (W_Q preferentially attends via outlier-channel directions), opening a different compression path.

### Primary risk
The 2 static outlier channels may be exactly the directions W_Q was trained to exploit (high-magnitude input → high-magnitude query). Mitigation: the smallest experiment settles this cheaply before any engineering investment; even NO-GO yields a structural finding.

---

## C-VKSD — W_V / W_K Spectral Delta Composition

**C / Track B**

### Name
V-K Spectral Delta Composition (VKSD) — C / Track B

### Mechanism
Track B. CF11 measured W_Q (r_99/d ≈ 0.63) and W_K (r_99/d ≈ 0.79) but W_V and W_O were flagged as untested. The composition claim requires measuring both, then asking: **do W_V and W_K share a dominant subspace?** Formally: `‖P_{V_K^{K256}} − P_{V_V^{K256}}‖_F < δ` for K=256, where P denotes the projection onto top-K right singular subspace. If W_K and W_V have aligned principal subspaces (as would happen if both project from the same residual-stream directions for different downstream purposes), the joint KV subspace is no larger than either alone. Concretely: instead of applying W_K and W_V separately to h, apply a single shared left-singular factor U_{shared} ∈ ℝ^{d×K} and two cheap "right-readout" matrices R_K ∈ ℝ^{K×d_k}, R_V ∈ ℝ^{K×d_v}: `k = R_K (U_{shared}^T h)`, `v = R_V (U_{shared}^T h)`. Residency savings: two full KV matrices (2 × 2048 × 2048 bf16 = 16 MB per layer) → one shared factor (2048×K) + two readout matrices (K×2048 each). At K=256: 3 × 256 × 2048 × 2B = 3.1 MB vs 16 MB — 5× compression of the KV write path. Does not touch MLP weights (CF8). Does not rely on CF7 / CF12. Does not rely on cross-layer sharing (v2-CHEAP-TEST-001 killed). This is purely per-layer joint diagonalization of W_K and W_V within the same layer.

**Coupling equation**: `subspace_gap(W_K^L, W_V^L) := ‖P_{K,256}^L − P_{V,256}^L‖_F`, and the joint compression is valid when this gap < threshold τ = 0.3 (i.e., the two 256-dim subspaces share ≥ 80% of their basis). The threshold derives from requiring that R_K and R_V be injective on the shared subspace (condition number < 10).

Grounding: CF11 (per-layer attention weight spectral structure), W_V/W_O flagged as untested open item in AQFKV results. No imported theorem — pure SVD alignment check. WDLA CF10 conditioning check: R_K and R_V are fixed by SVD, not fitted; no calibration conditioning risk.

### Residency arithmetic
Per layer: W_K + W_V = 2 × 2048 × 2048 × 2B = 16 MB. At K=256 shared: (2048×256 + 256×2048 + 256×2048) × 2B = 3.1 MB. Factor = 5.1×. Over 28 layers: 16 MB × 28 = 448 MB → 87 MB. Saving = 361 MB on Qwen3-1.7B. On 70B-scale (d_model≈8192, n_heads=64, n_kv_heads=8): W_K + W_V = 2 × 8192 × 1024 × 2B (GQA reduces W_K, W_V to n_kv_heads×d_head) ≈ 33 MB/layer × 80 layers = 2.6 GB. At 5×: 520 MB. Combined with W_Q compression (CF11 for W_Q at K≈512 at 70B scale): meaningful contribution toward DRAM residency. Quality: ΔNLL depends on W_V subspace alignment — need measurement.

### Novelty gloss
Closest kill-list item: R2/S2 Sparse PCA Cross-Head Basis (killed for CoSpaDi / Share-Your-Attention pre-emption). VKSD targets *within-layer K-V subspace alignment across two different matrices*, not cross-head or cross-layer — a different structural object. Closest published method: MLA (DeepSeek-V2) uses a shared KV projection but via training-time design. VKSD asks whether the *trained* W_K and W_V in a standard MHA model already have this alignment — a post-hoc measurement nobody has published for Qwen3 family. The move is pure observation: does training produce it anyway?

### Smallest experiment
**Claim**: `‖P_{K,256}^L − P_{V,256}^L‖_F < 0.3` for ≥ 20 of 28 layers in Qwen3-1.7B-Base.

Runtime: 30 min. Load W_K, W_V per layer (already loaded for AQFKV), compute top-256 SVD for each, measure F-norm of projection difference.

Go threshold: mean subspace gap < 0.3 → joint projection valid → proceed to W_K/W_V factored-readout reconstruction test.
No-go: gap ≥ 0.3 → W_K and W_V span independent subspaces → closes the "trained networks develop MLA-like structure spontaneously" question (structural finding useful to arch community).

### Primary risk
GQA: Qwen3-1.7B may use grouped query attention where W_K/W_V have fewer heads than W_Q, changing the shapes. Check model config before running SVD. Mitigation: if GQA is active, the composition applies to the smaller W_K/W_V matrices — weaker absolute savings but same structural question.

---

## C-SDLC — Layer-1 SDZC × W_Q Composition (Local Cascade)

**C / Track B**

### Name
SDZC-Q Layer-1 Composition Cascade (SDLC) — C / Track B

### Mechanism
Track B. CF6 (SDZC) shows 36% of gate neurons in layer 1 have variance below the foldable threshold — a hard local anomaly not present in layers 2-27 (< 2.5%). CF11 shows W_Q is compressible at K=128 globally. The composition claim is: **in layer 1 specifically, the foldable gate neurons define a sparse low-activation regime that allows W_Q^{layer1} to be compressed MORE aggressively than the global K=128 ceiling, because the residual stream entering layer 1's self-attention is less contaminated by MLP non-linearity.** Mechanism: (1) fold the 36% dead-gate neurons in layer 1's MLP (the gate-output is near-constant, absorbed into W_down bias — the SDZC move, valid even globally at 1.5% but powerful at 36% for layer 1). (2) After folding, the layer 1 MLP output is structurally simpler. (3) The modified h_{1}' = h_0 + attn_1(h_0) + MLP_1_folded(h_0) has lower effective dimensionality — specifically, the MLP_1_folded component lies in a ≤ 64% of the original neuron subspace. (4) This reduced output-dimensionality propagates to layer 2's W_Q input. Coupling equation: `r_99(W_Q^{layer2} · cov(h_1')^{1/2}) / r_99(W_Q^{layer2} · cov(h_1)^{1/2}) ≤ 0.8`, asserting that the effective rank of W_Q's interaction with the layer-1-folded residual stream drops by ≥ 20% relative to the original. If this holds, W_Q at layer 2 can be compressed to K=96 instead of K=128 with the same ΔNLL budget.

Does not rely on CF13-15. Does not rely on cross-layer W_Q basis sharing (killed). Composition of CF6 (layer-1 gate variance) × CF11 (W_Q spectrum per layer).

### Residency arithmetic
Layer-1 SDZC saving: 36% of 6144 neurons × (W_gate row + W_up row + W_down col) absorbed. Per neuron folded: removes 2 × 2048-dim vectors (W_gate row + W_up row gets baked into W_down adjustment) → saves 36% × 6144 × 2 × 2048 × 2B = 90 MB for layer 1 MLP. Tiny in isolation (3% of 3 GB model). Composition payoff: if layer-2 W_Q tolerates K=96 vs K=128, additional saving is (128-96) × 2048 × 2 × 2B × 27 layers ≈ 3.5 MB — also small. The structural value of this idea is not its residency saving but the **causal propagation measurement**: does a structural simplification in layer 1 propagate forward to reduce downstream attention dimensionality? This is a novel structural question about compositional information flow in transformers.

### Novelty gloss
Closest kill-list item: Track A R3 LCAB (Layer-1 specific gate folding, deferred). SDLC differs by composing the layer-1 folding with a downstream W_Q rank effect — a coupled measurement rather than isolated layer-1 compression. No published method addresses whether a layer's MLP simplification reduces the downstream layer's attention effective rank — this is a question about representational propagation in transformers.

### Smallest experiment
**Claim**: applying layer-1 SDZC folding (zero out the 36% near-constant neurons) produces a measurably lower effective rank in the resulting h_1' residual stream compared to h_1, quantified as PCA(h_1') capturing 99% variance at ≤ 80% of PCA(h_1)'s rank threshold, across the 200-prompt PDAP calibration set.

Runtime: 45 min. Run forward pass through Qwen3-1.7B-Base, ablate the 36% layer-1 neurons identified in SDZC (zero their contribution), capture h_1 (original) and h_1' (ablated), PCA both, compare r_99 counts.

Go threshold: r_99(h_1') / r_99(h_1) < 0.80 → composition cascade lives.
No-go: ratio ≥ 0.90 → MLP ablation leaves residual stream unaffected (as expected from residual additivity dominance, CF2). Even NO-GO is useful: it would confirm that residual additivity (CF2) neutralizes the layer-1 anomaly at the stream level.

### Primary risk
CF2 (residual additivity, cos ≈ 0.99) likely dominates: the MLP contribution in layer 1 is additive to a much larger residual, so ablating 36% of neurons may shift h_1 by only 2-3% in magnitude. Mitigation: measure the MLP-1 contribution magnitude relative to h_0 before committing — if |MLP_1_folded(h_0)| / |h_0| < 0.02, the experiment is pre-empted and need not run.

---

## C-RBHD — Residual-Band / Head-Diversity Decoupling

**C / Track B**

### Name
Residual-Band Head-Diversity Decoupling (RBHD) — C / Track B

### Mechanism
Track B. CF11 shows that globally, 16 heads collapse to ~1 head's worth of subspace at K=128 (head-redundancy finding). CF3 shows that at K=1% of channels, the activation outlier set is token-dynamic (Jaccard=0.31) while at K=0.1% it is near-static (Jaccard=0.718). The composition asks: **do the 16 heads of W_Q span *different* portions of the dynamic outlier regime?** Formally: partition the 20 K=1% dynamic outlier channels into per-head contributions by measuring `⟨W_Q^h · h, e_i⟩` for each head h and each channel i in the dynamic outlier set D. If each head preferentially activates a different subset of D, the heads are *functionally diverse at the outlier level* even though their weight subspaces overlap at K=128. This would reconcile CF11's head-redundancy (at the level of weight singular vectors) with a functional head-diversity at the activation level: the heads share a basis but use it to route attention toward different outlier channels per token. Coupling equation: `|D_h ∩ D_{h'}| / |D_h ∪ D_{h'}| < 0.3` for h ≠ h', where D_h is the top-5 outlier channels preferentially attended by head h. If this holds, the 16-head structure is NOT redundant at the functional level — it is a *routing* structure over the dynamic outlier set. If it fails (heads all attend the same outlier channels), then CF11's weight-level head-redundancy IS functional redundancy and K=128 compression discards genuinely redundant computation.

Track B because the insight either validates or challenges the W_Q K=128 compression ceiling — the mechanism is a compositional analysis, not an arch change.

### Residency arithmetic
If the composition finds that heads are functionally diverse over outlier channels: the W_Q K=128 compression is suboptimal — some heads need K>128 to preserve their outlier-routing function, others tolerate K<128. A per-head budget assignment (e.g., heads with high outlier diversity get K=128, heads with low get K=64) could recover quality at the same residency. At 7 heads × K=64 + 9 heads × K=128 = 7×64×2048 + 9×128×2048 × 2B per layer: 1.98 MB vs 2.1 MB (K=128 global) → marginal residency win but meaningful quality improvement. Quality: if head diversity is real and honored, ΔNLL recoverable from +0.98 to +0.6 nats at same K average. Payoff is quality-per-parameter, not raw residency.

### Novelty gloss
Closest kill-list item: v2-CHEAP-TEST-001 (cross-layer W_Q, killed). RBHD is strictly within-layer, measuring cross-head activation routing — structurally orthogonal. Closest published: "Share Your Attention" / CoSpaDi (cross-head weight sharing). RBHD instead measures activation-routing diversity within the shared subspace — the dynamic outlier targeting behavior of individual heads — which is not addressed by weight-sharing papers. The structural novelty is the coupling of per-head weight subspace (CF11) with per-token activation dynamics (CF3).

### Smallest experiment
**Claim**: for ≥ 10 of 28 layers in Qwen3-1.7B, the per-head outlier-channel Jaccard `|D_h ∩ D_{h'}| / |D_h ∪ D_{h'}|` between pairs of heads is < 0.3, where D_h is the set of top-5 residual-stream channels receiving the highest mean W_Q^h attention score on the PDAP 200-prompt corpus.

Runtime: 60 min. Reuse PDAP forward-pass runs; extract Q = W_Q · h for each head, measure mean absolute Q values per channel, compute pairwise head Jaccard on top-5 channels.

Go threshold: mean pairwise Jaccard < 0.3 → head diversity is real → per-head K budget assignment live.
No-go: Jaccard ≥ 0.5 → heads are functionally indistinct in outlier routing → CF11's weight-level redundancy IS functional redundancy → stronger justification for aggressive K=64 per-head (which is a NO-GO by CF11 weight-level measurement but now understood as quality-acceptable given functional redundancy).

### Primary risk
The measure D_h (top-5 channels per head) is a noisy statistic across a heterogeneous 200-prompt corpus. Per-head routing patterns may be context-dependent rather than stable across prompts. Mitigation: compute per-prompt Jaccard and report variance alongside mean; if variance is high, the effect does not generalize.

---

## C-WVKQ — W_V/W_K/W_Q Spectral Ladder Joint Compression [FREE SWING]

**C / Track A**

### Name
Spectral Ladder Joint KVQ Operator (WVKQ) — C / Track A [FREE SWING]

### Mechanism
Track A. This is a composition across three measurements: CF11's W_Q per-layer spectrum (r_99/d≈0.63), W_K spectrum (r_99/d≈0.79), and the VKSD open question on W_V spectrum. If W_V also has r_99/d < 1.0 (plausible given attention matrices' different training signal from MLP), all three can be jointly replaced by a single low-rank factored attention operator. The arch-transposition: replace the three separate projections `Q=W_Q h, K=W_K h, V=W_V h` with a single low-rank basis extraction `z = U^T h ∈ ℝ^K` (one shared projection, K=256 from a joint SVD of the stacked [W_Q; W_K; W_V] matrix), followed by three cheap readout projections `Q=R_Q z, K=R_K z, V=R_V z`. This is a Track A change because the computation graph changes: one shared GEMV replaces three separate GEMVs. The residual stream h is projected once to a K-dim space, and all three attention weight matrices read from that subspace. The coupling claim: `principal_angle(rowspan(W_Q^{K256}), rowspan(W_K^{K256}), rowspan(W_V^{K256})) < 45°` — the three subspaces share a substantial common component, making joint basis extraction lossless relative to separate extraction.

Grounding: CF11 (per-layer W_Q, W_K spectra confirmed), W_V flagged as untested open item. No cross-layer basis (killed). The [FREE SWING] tag reflects that W_V spectrum is unconfirmed and the joint-basis claim has no empirical anchor yet.

### Residency arithmetic
Per layer: W_Q + W_K + W_V + W_O = 4 × 2048 × 2048 × 2B = 32 MB. At K=256 joint basis + 3 readouts: (2048×256 + 3×256×2048 + 2048×2048) × 2B — W_O unchanged. Attention matrices: 2048×256 (shared U) + 3×256×2048 (R_Q, R_K, R_V) = 4×256×2048 = 4.19 MB vs 3×(2048×2048)×2B = 24 MB. Saving: 20 MB/layer. Over 28 layers: 560 MB. On 70B at GQA-64-8: W_Q dominates since there are 64 Q heads but only 8 KV heads; the joint basis must respect this asymmetry. Still: attention weight residency is a significant fraction of total. Quality: unknown — depends on subspace overlap. This is the wildcard: if the three subspaces are aligned, ΔNLL is dominated by the K=256 truncation loss (likely < 1 nat based on CF11 W_Q K=256 = +0.51 nats); if they are orthogonal, ΔNLL is catastrophic.

### Novelty gloss
Closest published: MLA (DeepSeek-V2) is training-time KV low-rank. LOLCATS / Performer replace softmax not the projection matrices. WVKQ is post-training joint basis extraction for Q+K+V simultaneously — the composition of three independently confirmed (or to-be-confirmed) per-matrix compressions into one joint operator. No known paper asks whether trained standard MHA weight matrices W_Q, W_K, W_V share a common input subspace post-training.

### Smallest experiment
**Claim**: `‖P_{Q,256} + P_{K,256} + P_{V,256} − P_{joint,256}‖_F / ‖P_{Q,256}‖_F < 0.2` on any single layer of Qwen3-1.7B, where P_{joint,256} is the projection onto the top-256 left singular vectors of the stacked matrix [W_Q; W_K; W_V] (3×2048 × 2048), and the three individual projections are as per CF11.

Runtime: 40 min. Load W_Q, W_K, W_V for layers {5, 14, 25}, compute joint SVD of stacked matrix (6144×2048 matrix, cheap), measure F-norm.

Go threshold: F-norm ratio < 0.2 → joint basis captures ≥ 80% of the union → Track A joint operator is well-motivated.
No-go: ratio > 0.5 → the three projections are largely orthogonal → closes the spontaneous-MLA-alignment question (structural finding).

### Primary risk
W_V spectrum is unverified — if W_V has r_99/d ≈ 1.0 (like MLP weights), the joint stacking has a full-rank component that pollutes the shared basis. Mitigation: run VKSD (C-VKSD above) as a precondition — check W_V spectrum before committing to the joint operator.

---

## C-ODWQ — Outlier-Drift Weighted W_Q Budget

**C / Track B**

### Name
Outlier-Drift Weighted W_Q Layer Budget (ODWQ) — C / Track B

### Mechanism
Track B. CF3 shows that deep layers (layers 23-27) spread outliers across 14-18% of channels for 90% of outlier events, while early-mid layers (2-19) concentrate them in 5-8% of channels. CF11 shows W_Q is globally compressible at K=128 across all 28 layers with ΔNLL=+0.98. The composition: **deep layers have higher activation-space dimensionality (per CF3's layer-stratified finding), which should require higher W_Q rank to maintain attention resolution.** Coupling equation: `Corr(outlier_spread_L, r_99(W_Q^L)) > 0` where outlier_spread_L is the number of channels covering 90% of outlier events at layer L (CF3's per-layer measurement from PDAP) and r_99(W_Q^L) is the per-layer effective rank from AQFKV. If this correlation is positive and strong, the W_Q compression budget should be non-uniform: give deep layers K=256, early layers K=64, instead of flat K=128. This preserves attention quality where the input is most structurally complex, and over-compresses where it is simple. Total budget at a non-uniform schedule (e.g., layers 1-19: K=64, layers 20-27: K=256): (19×64 + 8×256) × 2 × 2048 × 2 = (1216 + 2048) × 4096 = 13.4 MB vs flat K=128 at 28×128×2×2048×2 = 29.4 MB for the right-factor alone. Same ΔNLL or better by allocating budget where it matters.

Grounding: CF3 (layer-stratified outlier spread from PDAP full run), CF11 (per-layer W_Q r_99 from AQFKV). Both confirmed. No cross-layer sharing. Does not touch MLP.

### Residency arithmetic
W_Q right-factor (d × K) + left-factor (K × d) per layer. Flat K=128 all 28 layers: 28 × 2 × 2048 × 128 × 2B = 29.4 MB. Stratified schedule (layers 1-19: K=64, layers 20-27: K=256): (19×2×2048×64 + 8×2×2048×256) × 2B = (4.99 + 8.39) MB = 13.4 MB. Saving vs flat K=128: 16 MB additional, with quality expected to match or improve. On 70B scale: proportional savings larger. Quality: the ΔNLLs at K=64 for early layers (likely small, since CF11 showed K=256 costs only +0.51) and K=256 for deep layers (should be better than K=128 there) need per-layer measurement.

### Novelty gloss
Closest kill-list item: R2/S2 MDL-Selected Per-Layer bpw (killed for Compressibility Measures Complexity pre-emption). ODWQ differs by grounding the per-layer budget specifically in a measured cross-finding correlation (outlier spread CF3 × W_Q rank CF11), not in a general MDL principle. The coupling hypothesis (outlier spread correlates with W_Q effective rank) is a specific falsifiable claim about Qwen3 internal structure. No published method to date couples activation outlier dynamics to attention weight rank as a budget-allocation mechanism.

### Smallest experiment
**Claim**: Spearman correlation `ρ(outlier_spread_L, r_99(W_Q^L)) > 0.3` across 28 layers of Qwen3-1.7B, where outlier_spread_L is from the PDAP run (channels needed to cover 90% of outlier events at layer L) and r_99(W_Q^L) is the per-layer r_99 from the AQFKV run.

Runtime: 10 min. All data already exists in prior runs — PDAP produced per-layer outlier spreads, AQFKV produced per-layer r_99(W_Q). Simple Spearman correlation over 28 points.

Go threshold: ρ > 0.3 (and p < 0.05) → budget stratification is empirically motivated.
No-go: ρ < 0.1 → W_Q rank and activation-space complexity are independent by layer — which means the flat K=128 is already near-optimal (useful negative finding).

### Primary risk
The r_99 per-layer breakdown from AQFKV may have been measured only at the global-K sweep level, not per layer. If per-layer r_99 was not recorded, the correlation cannot be computed without re-running the AQFKV spectrum measurement per layer. Mitigation: check AQFKV result file — if only global K was measured, add a 15-min per-layer r_99 sweep as a pre-run step.

---

## Convergence handles

1. **Per-layer W_Q/W_K/W_V singular-value spectra** — VKSD, WVKQ, and ODWQ all need per-layer attention weight spectra; a single 28-layer SVD sweep over all four attention matrices closes all three experiments' preconditions simultaneously.
2. **Static outlier channel identity** — AQOD and RBHD both require the 2 static K=0.1% channels from the PDAP run; one lookup into the existing result file serves both.
3. **Per-layer outlier spread** — ODWQ needs the CF3 per-layer channel count; already in the PDAP result but may need to be extracted as a per-layer table rather than global mean.
4. **Layer-1 MLP contribution magnitude relative to residual** — SDLC's pre-run check (|MLP_1| / |h_0|) determines whether the experiment is worth running; this single ratio is free from any existing forward-pass run.
5. **W_V spectrum confirmation** — unverified; all ideas that touch V weights are conditioned on this. A 20-min SVD of W_V resolves it and activates or kills VKSD and WVKQ.
