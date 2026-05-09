# Stage 1 — Orientation A (Constraint-Alien) — Run 006

Orientation: **A — Constraint-Alien**
Run: 006
Ideator: Sonnet claude-sonnet-4-6

**KILL in effect**: gauge constraints producing arbitrary orthogonal rotation R ∈ O(d) are inadmissible (RMSNorm-incompatible, v2-S3-R004-001). The only residual-stream symmetry group that commutes with RMSNorm's per-channel diagonal γ is the generalized permutation group: sign-flips × permutations (Z₂^d × S_d). Ideas below do NOT invoke O(d) gauge freedom.

**Constraint saturation check across runs 001–005**: tropical attention (A-004/A-005), 10 MB seed synthesis (A-004/A-005), append-only KV log (A-004/A-005), register-only residual (A-004), FSM decode (A-004/A-005), content-addressable BLAKE3 dispatch (A-005), Z₂ sign-gauge codebook (A-005), peer-swap inference (run_006 prior draft), O(1)-token parallel decode (run_006 prior draft), debuggable-bytes INT8 (run_006 prior draft), reversible-residual (run_006 prior draft), vector-clock pipeline (run_006 prior draft), entropy-monotone early exit (run_006 prior draft). Ideas below draw from the unexplored corners of the constraint menu.

---

## A1-HPCI — Head-Permutation Canonical Inference [A / Track A]

**Constraint: The 16 attention heads within each layer must be computed in a canonical order determined by their action on the current token — no fixed head index is permitted to drive any scheduling or weight-fetch decision.** Head identity is mutable at inference time; only the action matters. This eliminates all head-indexed scheduling, fixed weight-prefetch orders, and any static assignment of head roles (~65% of attention-implementation mechanisms become inadmissible under this constraint).

### Mechanism

Track A. Softmax aggregation over multi-head outputs is permutation-invariant in the head dimension: O(x) = Σ_h W_O^h · head_h(x) is invariant to any permutation π of h (relabeling heads). This permutation symmetry is the constraint's mechanical anchor. The trained Qwen3 network has no canonical head ordering — the 16 W_Q matrices for 16 heads were assigned their indices by gradient descent without any head-ordering regularizer. The constraint forces a RUNTIME ordering of heads by their projection onto the current residual state: sort heads by ‖W_Q^h · h_L‖₂ descending at each token. The top-k heads (high query-energy) are computed first; the remaining 16-k heads may be skipped or lazily evaluated.

The constraint's forcing work: once heads are sorted by action, the top-k subset changes per token (dynamically). This is NOT a static head-pruning scheme (which would fix the dead heads at calibration time). It is a content-driven dynamic head ordering that the permutation-invariance constraint REQUIRES (because fixed ordering is arbitrary under permutation symmetry). The payoff: if 4 of 16 heads dominate at each token, skip 12 — 75% attention weight bandwidth reduction. CF11 measured W_Q global K=128 GO (8× head-redundancy at 16 heads); this is a different mechanism — per-token dynamic ordering rather than static shared-basis compression. Not killed by v2-CHEAP-TEST-001 (that kills cross-layer; this is within-layer per-token).

RMSNorm compatibility: no weight transformation required. The permutation acts on the head-compute ORDER, not on the residual stream channels. The generalized permutation group (S_16 on heads) is the constraint group; this does not interact with RMSNorm's per-channel diagonal.

### Residency arithmetic

Qwen3-1.7B attention weights per layer: W_Q (8 MB), W_K (4 MB GQA), W_V (4 MB GQA), W_O (8 MB) = 24 MB/layer. If 4 of 16 heads are computed per token: W_Q read = 8 MB × (4/16) = 2 MB; W_O read = 8 MB × (4/16) = 2 MB. W_K, W_V shared across GQA groups (8 KV heads for 16 Q heads): load proportional to which 4 Q heads need which KV group. Net: 24 MB → ~10 MB/layer at 4/16 skip rate. 28 layers × 14 MB saved = 392 MB per token. At DRAM 11.5 GB/s: 14 MB × 28 / 11.5 GB/s ≈ 34 ms saved per token vs baseline ~2 ms/layer attention = 56 ms total attention. Net: 56 → 22 ms attention bandwidth. Total token latency reduced proportionally; at 1.7B INT4 (1.05 GB) total bandwidth = 91 ms/token, attention is ~20% → saving 75% of that 20% = 15% total speedup ≈ 1.17×. Modest at 1.7B; at 70B where attention is a larger fraction (~35% of bandwidth): 70B × 0.35 × 0.75 = ~26% total speedup.

### Novelty gloss

Kill list: no head-permutation canonical ordering entry. A-CADH (run_004, within-layer cross-head W_Q similarity) tests static head similarity; this tests dynamic per-token ordering. Published: Deja Vu (ICML 2023) predicts ReLU head sparsity from prior attention; different (requires a predictor network). Sparse Transformer uses fixed strided patterns. Per-token dynamic head sorting by query-energy norm has no published instance. The constraint grounds the mechanism: permutation invariance is a verifiable structural property of the aggregation formula, not an approximation.

### Smallest experiment

**Claim**: On Qwen3-1.7B-Base, the top-4 heads by ‖W_Q^h · h_L‖₂ at each token account for ≥ 60% of the total weighted output magnitude ‖Σ_h w_h · head_h‖ (where w_h is the attention-head contribution norm) across 200 WikiText-2 tokens.

- Measure: forward pass 200 tokens through Qwen3-1.7B; at each layer record per-head query norms and per-head output contribution norms; compute what fraction of total output magnitude the top-4 heads by query-norm supply.
- Runtime: ~45 min (200 forward passes with per-head instrumentation, Qwen3-1.7B BF16, Ryzen).
- GO: top-4 query-norm heads supply ≥ 60% of output magnitude on average. NO-GO finding: per-head output distribution is flat → no dynamic head ordering gains exist → permutation invariance constraint forces degenerate scheduling; structural finding about head-contribution uniformity in Qwen3.

### Primary risk

Query norms ‖W_Q^h · h‖ may not predict output contribution magnitude — a head may have a large Q but attend to near-zero-V positions. Mitigation: the experiment measures output contribution directly (not proxy); if query-norm is a poor predictor, reframe the sorting key as post-softmax output norm, which is more expensive to compute but directly correct.

---

## A2-SINT — Softmax Shift-Canonical Attention: Integer Keys in Shift-Invariant Coordinates [A / Track A]

**Constraint: Attention computation must operate in shift-canonical coordinates — no representation of query-key dot products that is not shift-invariant under softmax is permitted.** Softmax(q·K^T) = Softmax(q·K^T + c·1) for any scalar c. The constraint requires the implementation to exploit this symmetry fully: all stored keys must be in a representation where the arbitrary additive constant has been eliminated by fixing a canonical shift. This eliminates ~70% of standard attention key-storage mechanisms (including all that store raw floating-point keys without shift normalization).

### Mechanism

Track A. The softmax shift-invariance is exact and unconditional: Softmax(z + c) = Softmax(z). This means any attention score vector q·K^T can be globally shifted by a constant without changing the attention weight matrix. The canonical shift is: center the score vector so its maximum is 0 (subtract max). After shift-canonicalization, all query-key dot products are in (-∞, 0]. This is always computed implicitly in numerically-stable softmax implementations (the log-sum-exp trick). The constraint forces it to be EXPLICIT in the KEY STORAGE format: keys K are stored in shift-canonical coordinates where the dot product with any stored representative query q₀ is ≤ 0, with the canonical reference q₀ chosen from calibration data.

The payoff: in shift-canonical coordinates, all dot products q·K^T fall in a bounded range [-C, 0] for some calibration-derived C. This bounded range admits integer quantization with no dynamic range loss. Standard INT8 key quantization must reserve range for both positive and negative values (splitting the 8-bit range); shift-canonical keys are ALWAYS non-positive after canonicalization relative to q₀, using the full INT8 range [0, 255] for the negative side only. Net effect: 1-bit resolution improvement over naive signed INT8. Or equivalently: the same quantization quality at 7 bits (INT7 packed as INT8 with the high bit always 0). This is a lossless coordinate-change with a quantization dividend.

Implementation: (a) compute reference query q₀ from calibration (mean of 1000 query vectors at layer L); (b) for each stored key k, compute the shift-adjustment s_k = q₀·k; (c) store k_shifted = k - s_k·q₀/‖q₀‖² (modified key such that q₀·k_shifted = 0); (d) at inference, compute attention as Softmax((q - q₀)·K_shifted^T) which equals Softmax(q·K^T - q₀·K^T). The score relative to q₀ is absorbed into the shift-canonical key storage. The q₀ subtraction adds one d_head-dim GEMV per query per layer (negligible cost).

### Residency arithmetic

KV cache at 4K context, Qwen3-1.7B: 28 × 2 × 2048 × 4096 × bf16 = 917 MB. K cache alone: 459 MB. Shift-canonical INT7 (packed in INT8): 459 MB × 7/8 = 402 MB → saves 57 MB. Modest at 4K. At 128K context: K cache = 14 GB → shift-canonical saves 1.75 GB. At 128K: weights (17 GB INT4) + KV (K at INT7 = 3.5 GB, V at bf16 = 7 GB) = 27.5 GB — still too large but closer. Pairs with V quantization and weight compression. The structural argument is the constraint-forced INT7 key quantization with no dynamic-range sacrifice — a lossless coordinate transformation that existing INT4/INT8 KV schemes don't exploit.

### Novelty gloss

Kill list: no shift-canonical key storage entry. ASCQ (Track B R6 deferred, attention-sink channel quantization) uses INT2/INT4 splits by token type — different; that's a dynamic scheme, not a coordinate-canonical scheme. Closest published: KVQuant (Hooper et al., 2024) uses non-uniform quantization for KV; doesn't exploit shift-invariance. FlashAttention's online softmax uses the max-shift internally but discards it after computing attention — doesn't change key storage. The constraint here makes the shift LOAD-BEARING in the storage format, not just a numerical trick, which has no published instance in KV compression.

### Smallest experiment

**Claim**: On Qwen3-1.7B-Base, shift-canonical INT8 key storage (keys shifted by the projection of a calibration-mean query, then quantized with unsigned INT8 to the non-positive range) achieves NLL within 0.05 nats of standard bf16 key storage on 200-token WikiText-2 sequences.

- Measure: compute q₀ (mean of 1000 query vectors at each layer); apply shift to all K vectors in a 200-token KV cache; quantize shifted K to unsigned INT8; run attention with dequantized shifted K; measure NLL delta.
- Runtime: ~2 hr (implementation in Python + measurement).
- GO: ΔNLL < 0.05 nats. NO-GO finding: shift-canonical INT8 loses significant quality → the shift is NOT sufficient to constrain the dynamic range to INT8 precision → the calibration reference q₀ is too coarse for per-token variation; structural finding about the distribution width of q·K^T in shift-canonical coordinates.

### Primary risk

The shift-adjustment vector q₀ is a calibration-average; per-query variation may mean that many queries still produce positive dot products with shifted keys, breaking the "all non-positive" assumption and restoring the need for signed range. Mitigation: measure the fraction of q·K_shifted^T values that are positive across calibration tokens; if > 5%, the shift-canonical assumption fails; fall back to a per-head q₀ (16 reference queries per layer) which may constrain the range more tightly.

---

## A3-IDPT — Idempotent Projection Layers: Force Residual Updates to be Fixed Points [A / Track A]

**Constraint: Each layer's residual update Δ_L = Layer_L(h_L) must be an idempotent projection — applying it twice produces the same result as applying it once. Only transformations satisfying Layer_L(Layer_L(x)) = Layer_L(x) are permitted.** This eliminates ~90% of standard transformer layer designs (MLP and attention are not idempotent). Forces: projection operators, binary-coded activations, softmax with hard-argmax limit.

### Mechanism

Track A. An idempotent linear map P satisfies P² = P — it is a projection onto a subspace. For a transformer residual update Δ_L = f(h_L), idempotency requires f(f(h_L) + h_L) ≈ f(h_L) for all h_L in the calibration distribution. This is NOT generally satisfied by trained Qwen3 layers. But the constraint FORCES the question: how many transformer layers are APPROXIMATELY idempotent? CF2 says cos(h_L, h_{L+1}) ~ 0.99 — the residual stream barely changes per layer. This means ‖Δ_L‖ << ‖h_L‖, so h_{L+1} ≈ h_L. Applying the layer AGAIN: f(h_{L+1}) = f(h_L + Δ_L) ≈ f(h_L) + J_L Δ_L where J_L is the Jacobian. Since Δ_L is small, the second application ≈ Δ_L + J_L Δ_L = (I + J_L)Δ_L. Idempotency requires (I + J_L)Δ_L ≈ Δ_L → J_L Δ_L ≈ 0. This holds when Δ_L lies in the nullspace of J_L, i.e., the layer update direction is in the near-nullspace of the layer's own Jacobian.

The structural question: is this approximately true for Qwen3? If yes: layers are approximately idempotent, which means LAYER REPETITION (running the same layer twice instead of the next layer) costs approximately zero additional quality while doubling that layer's feature processing. A network of 28 distinct layers could be replaced by K < 28 distinct idempotent projections, each run multiple times. The structural compression: if 20 of 28 layers are "close enough" to idempotent that repeating them costs < 0.05 nats, the effective number of DISTINCT layer weight sets needed is only 8, reducing residency by 20/28 ≈ 71%.

This is NOT weight-sharing / tying (which requires identical layer outputs at all positions). Idempotency allows different layers to be collapsed ONLY IF their updates satisfy the near-nullspace condition. The constraint identifies WHICH layers are collapse-eligible.

### Residency arithmetic

If 20 of 28 layers are idempotent-collapsible: store only 8 distinct layer weight sets. Qwen3-1.7B at INT4: each transformer layer (attn + MLP weights) ≈ 34 MB. 28 layers = 952 MB. 8 distinct layers = 272 MB. Saving: 680 MB. Total model: 1.05 GB INT4 → 370 MB (if token embeddings + lm_head ~100 MB added). DRAM-resident comfortably. Quality cost is the open question; the smallest experiment measures it.

### Novelty gloss

Kill list: no idempotent-projection constraint. Published: Universal Transformers (Dehghani et al. 2019) iterate the same layer; different — they tie weights by design during training. This proposes applying the idempotency CHECK post-hoc to a standard trained model without weight tying. Cross-layer weight sharing (Lan et al., ALBERT) requires retraining. The constraint angle is: identify empirically which of Qwen3's 28 layers are post-hoc idempotent-enough to collapse, using the CF2-motivated near-nullspace argument. No published work applies idempotency as an inference-time filter for layer-deduplication without retraining.

### Smallest experiment

**Claim**: At least 5 of 28 layers in Qwen3-1.7B-Base have Jacobian-nullspace alignment ‖J_L · Δ_L‖_F / ‖Δ_L‖_F ≤ 0.10 on the calibration distribution (i.e., the update direction is in the near-nullspace of that layer's own Jacobian, making the layer approximately idempotent).

- Measure: for each layer L, run 200 calibration tokens through the model; compute Δ_L = h_{L+1} - h_L; compute the first-order approximation J_L · Δ_L via a Jacobian-vector product (implemented as a forward-mode AD pass or finite-difference); measure ‖J_L Δ_L‖_F / ‖Δ_L‖_F.
- Runtime: ~4 hr (finite-difference JVP for each of 28 layers × 200 tokens; each JVP requires one extra forward pass per layer).
- GO: ≥ 5 layers with ratio ≤ 0.10. NO-GO finding: all layers have high Jacobian-alignment → transformer layers are NOT approximately idempotent → CF2's high cosine similarity does NOT imply near-nullspace Jacobian structure → structural finding about the difference between angle-of-change and Jacobian-sensitivity.

### Primary risk

Computing Jacobian-vector products for each layer may be impractical at 1.7B scale in pure Python. Mitigation: use finite-difference approximation (run layer twice with h_L + ε·Δ_L as input, measure output shift); ε = 1e-3 is sufficient for this order-of-magnitude measurement.

---

## A4-TMLP — Tropical MLP: Max-Plus Arithmetic Replaces the FFN Entirely [A / Track A]

**Constraint: The MLP sub-block (W_gate, W_up, W_down) must be computed entirely using the tropical semiring (max-plus arithmetic over ℤ) — no floating-point operations are permitted inside the MLP.** The constraint eliminates SwiGLU, all continuous gating functions, all floating-point matmuls inside the FFN (~85% of MLP implementations are inadmissible). This is distinct from prior tropical runs (A-004/005 applied tropical to attention dispatch or W_down routing); this targets the FULL MLP computation.

### Mechanism

Track A. In the tropical semiring (ℝ, ⊕, ⊗) with a ⊕ b = max(a,b) and a ⊗ b = a+b, the analog of a linear map W·x is the tropical matrix-vector product: (W ⊗_trop x)_i = max_j(W_ij + x_j). The analog of matrix composition is tropical matrix multiplication (max-plus convolution). The constraint forces the MLP to use this arithmetic throughout. In the tropical semiring, the SwiGLU gating f(Wx) ⊙ Vx vanishes — there is no element-wise multiply, only max and add. The forced architecture: the MLP becomes a 2-layer tropical network:

h_out = W_down ⊗_trop (W_up ⊗_trop x)

where ⊗_trop is tropical matmul. The tropical matmul (W_up ⊗_trop x)_i = max_j(W_up_ij + x_j) selects, for each hidden unit i, the most dominant input dimension j*_i. This is a HARD routing: each hidden unit is driven by exactly one input channel (the max), not a weighted sum. The entire MLP becomes a cascade of argmax selection steps — no multiplication, no addition of more than two terms at once (max and add), implementable as pure integer arithmetic.

Integer implementation: discretize W_up, W_down, and x to INT16 (after per-channel normalization). Tropical matmul is max over INT16 additions. On Zen-3, SIMD integer max (VPMAXSB / VPMAXSW) and addition are available at full 256-bit throughput — no FP unit pressure, no floating-point precision constraints. This is NOT INT4 quantization of the MLP weights (standard approach); it is a fundamentally different computation that eliminates MULTIPLY entirely.

CF connection: CF1 (W_up dominance) is the motivating signal — if W_up's output is dominated by a few strong input channels per hidden unit, the tropical argmax approximates the continuous W_up·x well. CF8 says W_up is full-rank (no low-rank structure), but tropical computation doesn't depend on rank — it depends on whether each hidden unit has a dominant input channel (concentrated column distribution), which is a different structural question.

### Residency arithmetic

Tropical W_up/W_down at INT16: 2 bytes/weight vs bf16's 2 bytes/weight — same size. The gain is compute, not residency. At Zen-3 AVX2, 256-bit lane: 16 INT16 values per lane vs 8 bf16 values per lane → 2× arithmetic throughput. MLP is bandwidth-bound at 1.7B; the compute gain is secondary. For 70B where the MLP accounts for ~60% of inference time: 2× arithmetic throughput on the MLP sub-block → 1.3× overall speedup estimate. Residency unchanged; quality cost is the open question.

Tropical approximation quality: if the tropical MLP captures the top activation patterns (the CF1 finding that W_up dominates firing), ΔNLL may be acceptable. The most pessimistic case: tropical matmul is a zero-temperature limit of continuous matmul; quality degrades linearly with the softness of the max.

### Novelty gloss

Kill list: A-TROP (run_004) applied tropical dispatch to W_down output rows with a stability predicate — different; that was a routing pre-filter, not a full replacement of MLP arithmetic. A2-TROP (run_005) applied tropical to ATTENTION softmax — different matrix family. Full MLP tropical replacement has no kill-list entry. Published: tropical neural networks exist as theory (Maragos, ICASSP 2021; Charisopoulos & Maragos, 2019) but have never been applied as an inference-time replacement for a trained transformer MLP without retraining. The constraint forces the question: can a standard bf16-trained MLP be evaluated in tropical arithmetic at inference time with acceptable quality loss?

### Smallest experiment

**Claim**: On Qwen3-1.7B-Base, replacing the MLP of layer 0 with a tropical matmul approximation (W_gate dropped; W_up and W_down evaluated as tropical matmuls over INT16 discretization) achieves ΔNLL ≤ 2.0 nats on a 100-token WikiText-2 sample.

- Measure: extract W_up and W_down for layer 0; discretize to INT16 (per-column normalization); implement tropical matmul (max over INT16 add); run hybrid forward pass (all other layers bf16, layer 0 MLP tropical); measure NLL delta.
- Runtime: ~2 hr (Python prototype + measurement).
- GO: ΔNLL ≤ 2.0 nats on layer 0 alone. NO-GO finding: tropical approximation degrades quality catastrophically even on layer 1 → the continuous MLP computation cannot be approximated by tropical arithmetic at inference time → structural finding about the sensitivity of SwiGLU to the transition from soft-max (softmax/SiLU) to hard-max (tropical).

### Primary risk

The tropical matmul is a zero-temperature limit; for a trained network that relies on distributed activation patterns (full-rank W_up per CF5), the hard argmax selection may destroy most of the computation's information. Mitigation: test on k-tropical (top-k, not top-1) for k ∈ {1, 4, 16}; the quality vs k curve is itself a structural finding about the effective sparsity of Qwen3's MLP activations.

---

## A5-GRPN — Group-Norm Permutation Canonical Form: S_d-Fixed Inference Coordinates [A / Track A]

**Constraint: The residual stream must be processed in permutation-canonical coordinates throughout inference — channel indices are never fixed and must be assigned by decreasing magnitude of the current residual state.** This eliminates all weight access patterns that rely on a fixed channel ordering (~80% of standard GEMV implementations, all structured quantization schemes that assign fixed-channel meaning).

### Mechanism

Track A. The residual stream x ∈ ℝ^d can be permuted by any π ∈ S_d without changing the model outputs IF all weight matrices are simultaneously permuted (W → WP^T on input, P W on output, for permutation matrix P). This is the generalized permutation group freedom — the S_d part of the Z₂^d × S_d group that commutes with RMSNorm's diagonal. The constraint requires choosing the canonical representative where channels are sorted by absolute magnitude at each layer boundary: x_π = P_x · x where P_x = argsort(|x|) descending. Then all weight matrix accesses use the permuted indices.

The payoff: in permutation-canonical coordinates, the residual stream always has its largest components in positions 0, 1, 2, …. Weight matrices must be evaluated against this dynamically-sorted input. The FORCED implication: the first k rows of any weight matrix W (in canonical coordinates) are always the k rows most correlated with the current input's dominant channels. This enables a hard threshold: ONLY the first k rows of W need to be loaded if the input is in canonical form and only the top-k contributions matter.

Concretely: sort the current residual x by |x_i| descending; permute W_up columns to match (W_up[:, π(x)]). Now W_up[:, 0] corresponds to the channel of x with highest magnitude; W_up[:, 1] to the second highest; etc. If the magnitude distribution is heavy-tailed (consistent with CF3: K=1% outlier channels have Jaccard 0.718 = mostly stable), the first ~20 columns of W_up[:, π(x)] dominate the product W_up · x. Load only those columns: fetch 20/2048 = ~1% of W_up column bytes = 0.24 MB per layer instead of 24 MB.

This is mechanically different from content-addressable dispatch (A5-run_005 used BLAKE3 hashing); here the dispatch key is the sort permutation π(x) derived from the current residual state, not a hash.

RMSNorm compatibility: the sort permutation is the S_d part of the generalized permutation group, which commutes with RMSNorm. Explicitly: RMSNorm(Px) = P · RMSNorm(x) because P^T γ P is just a permuted diagonal. The constraint is RMSNorm-compatible by construction.

### Residency arithmetic

If the top-20 columns of W_up (in permutation-canonical coordinates) capture ≥ 50% of the W_up · x magnitude per token: load 0.24 MB W_up instead of 24 MB per layer = 100× reduction on W_up bandwidth. 28 layers: saved bandwidth = 28 × (24 - 0.24) MB = 665 MB per token. At DRAM 11.5 GB/s: 58 ms saved per token (W_up alone). W_gate and W_down still full load. Total token cost: ~91 ms → 91 - 58 + (0.24 × 28 / 11.5 GB/s) = 91 - 58 + 0.6 = 33.6 ms ≈ 30 tok/s. This is the 3× speedup claim — entirely contingent on the top-20 column coverage exceeding 50%.

The overhead: the sort permutation π(x) requires sorting 2048 values per layer = O(d log d) ≈ 22K comparisons. At Ryzen AVX2 throughput (~1 ns/comparison): ~22 µs per sort. 28 layers: 0.6 ms overhead. Negligible vs the 58 ms bandwidth saved.

### Novelty gloss

Kill list: closest is A-CNTR (run_005, content-addressable W_up with BLAKE3 + CF3 routing key) — structurally different; CNTR used CF3's outlier-channel set as a pre-fixed dispatch key; GRPN uses the CURRENT residual state's sort permutation (dynamic per token) with no pre-fixed channel identity. The group-theoretic grounding (S_d canonical form) is distinct. Published: SmoothQuant migrates outlier magnitude across Q/K/V; ChannelMix permutes for better quantization at training time. No published method uses dynamic per-token S_d canonical form for weight-column selective loading at inference time. The mathematical anchor is the S_d part of the RMSNorm-compatible symmetry group — something the literature has not exploited.

### Smallest experiment

**Claim**: On Qwen3-1.7B-Base, the top-20 columns of W_up[:, π(x)] in permutation-canonical coordinates account for ≥ 40% of ‖W_up · x‖₁ on average across 200 calibration tokens (i.e., the first 20 of 2048 input channels, sorted by |x_i|, contribute ≥ 40% of the total MLP intermediate activation magnitude).

- Measure: for each of 200 tokens, sort the residual state channels by |x_i| descending; compute W_up · x; measure what fraction of ‖W_up · x‖₁ comes from the top-20 columns. Average across tokens and layers 0, 14, 27.
- Runtime: ~30 min (200 forward passes with per-channel contribution recording).
- GO: mean coverage ≥ 40%. NO-GO finding: permutation-canonical columns are NOT concentrated → the residual stream's magnitude distribution does not predict W_up column relevance → the S_d canonical form provides no sparsity benefit → structural finding about the relationship between input channel magnitude and output column contribution in Qwen3's MLP.

### Primary risk

CF3's K=1% outlier-channel stability (Jaccard 0.718) is for ACTIVATION outliers across tokens, not for residual stream magnitude correlation with W_up column importance. These are different quantities; the measurement may show low coverage even if CF3 holds. Mitigation: the smallest experiment tests the exact coverage claim independently; it's a 30-min experiment regardless of the CF3 connection.

---

## A6-BSCH — Binary-Scheduled Computation: Bit-Serial Layer Ordering with Halting [A / Track A] [FREE SWING]

**Constraint: The model must halt computation of each layer when the first bit of the output stabilizes — no layer runs to full floating-point precision if single-bit output stability is achieved earlier.** This eliminates all fixed-precision layer computations (~95% of standard implementations). Forces: bit-serial arithmetic, progressive computation, early stopping at bit-level precision thresholds.

### Mechanism

Track A. Bit-serial computation processes each arithmetic operation one bit at a time from MSB to LSB. For a matmul y = Wx, the MSB of each output element y_i is determined by the MSB of W and x alone (to leading-order). Lower bits refine the estimate. The halting constraint: once enough output bits have stabilized that argmax(y) is determined (the highest-magnitude output dimension is unambiguous), halt. For token selection (vocabulary argmax over logits), a single-bit determination of which logit is largest is sufficient — no full precision needed.

The forcing argument: token generation only needs to identify argmax(lm_head(h_final)). Argmax is a comparison, not a precise value. If the top logit is separated from the second-largest by Δ, full precision is not needed — only enough bits to distinguish Δ from noise. For "easy" tokens (large logit gap), the argmax is determined at 4-bit precision or less. For "hard" tokens (small gap), full precision is needed.

The mechanism extends layer-by-layer: at each layer L, compute W_Q · h, W_K · h, W_V · h in bit-serial order, halting attention computation for each head once the attention argmax (tropical-style top-1 head) is stable. Then compute MLP in bit-serial order, halting once the post-SwiGLU output's sign structure (which determines W_down's dominant input channels) stabilizes.

Implementation primitive: INT4 arithmetic on Zen-3 provides 4-bit granularity. Starting from INT4, if the argmax is stable at INT4 (which it is for "easy" tokens), skip INT8 refinement. This is equivalent to dynamic precision scheduling: INT4 for easy tokens, INT8 for hard, BF16 for very hard.

[FREE SWING] — this borrows from signal processing (bit-serial CORDIC, BCD arithmetic) and has no direct CF anchor. The stack primitive: INT4 arithmetic in llama.cpp/ggml is already implemented; the "early halt on argmax stability" is a new scheduling policy layered on top.

### Residency arithmetic

No change in weight residency (weights stored at INT4 baseline). The throughput gain is compute: if 50% of tokens achieve argmax stability at INT4 (4-bit precision), those tokens skip INT8 refinement. At baseline throughput of 12 tok/s on Qwen3-1.7B INT4, and if easy tokens (INT4-stable) run 1.5× faster (skip one refinement pass): 0.5 × 12 + 0.5 × (12 × 1.5) = 15 tok/s → 1.25× speedup. The fraction of INT4-stable tokens is the empirical unknown.

### Novelty gloss

Kill list: no bit-serial / halting-precision entry. Published: Early exit methods (MSDNet, CALM) halt at a LAYER depth; this halts at a PRECISION DEPTH within each layer's arithmetic. Per-token adaptive precision scheduling (not per-layer) has no named form in LLM inference literature. The closest is RSIDC (residual-stream intrinsic-dimension-conditioned precision switching, deferred Track B R5) — different; RSIDC selects bpw per layer per token based on intrinsic dimension; BSCH selects bit-precision based on argmax-stability during arithmetic. The halting criterion (argmax stability) is operationally different from a learned or intrinsic-dimension threshold.

### Smallest experiment

**Claim**: On Qwen3-1.7B-Base, for ≥ 40% of greedy-sampled tokens from WikiText-2, the top-1 logit from INT4-precision lm_head output matches the top-1 logit from bf16-precision lm_head output (i.e., argmax is stable at INT4 for those tokens).

- Measure: run 200 WikiText-2 tokens through Qwen3-1.7B BF16; compute lm_head output at both INT4 (quantize h_final to INT4 before lm_head) and bf16; compare top-1 argmax; measure agreement rate.
- Runtime: ~30 min.
- GO: ≥ 40% of tokens show top-1 argmax agreement between INT4 and bf16 lm_head evaluations. NO-GO finding: INT4 argmax agrees with bf16 argmax for < 20% of tokens → quantization noise at INT4 routinely flips the top logit → bit-serial halting at INT4 is unsafe → structural finding about minimum precision required for stable argmax in Qwen3.

### Primary risk

INT4 quantization of the residual state h_final introduces significant noise into the logits; the argmax may flip frequently even for "confident" tokens. Mitigation: test at INT8 as well; if INT8 argmax agrees with bf16 for ≥ 60% of tokens, the dynamic-precision argument holds at INT8 granularity even if INT4 fails.

---

## Convergence handles

- **Per-head query-energy norm as a routing key for dynamic head ordering** — A1-HPCI; overlaps with any future attention-side pruning or Reach (R) cascade that proposes per-token dynamic head selection
- **Shift-canonical KV key distribution width** — A2-SINT; the range of q·K^T in shift-canonical coordinates is a measurement relevant to any INT-precision KV scheme; likely convergence signal with Composition (C) ideas coupling CF11 attention spectrum with KV quantization
- **Jacobian-nullspace alignment of layer updates (idempotency ratio)** — A3-IDPT; if this measurement is cheap (4 hr) it should be run regardless; a high idempotency ratio in specific layers opens weight-sharing without retraining; convergence candidate with F (First-Principles) deriving conditions under which residual layers are provably near-projection
- **Permutation-canonical column coverage of W_up** — A5-GRPN; structurally adjacent to CF3 outlier-channel routing (run_005 A6-CNTR, run_006 A5-GRPN); measuring the S_d coverage fraction may re-derive the CF3 K=1% outlier finding from a different angle; cross-orientation convergence if C (Composition) also produces a column-coverage coupling
- **INT4 logit argmax stability fraction** — A6-BSCH; a cheap measurement (30 min) that gates the entire adaptive-precision scheduling line; also relevant to any early-exit idea that halts at logit-confidence thresholds
- **RMSNorm-compatible symmetry group (Z₂^d × S_d) as the only admissible gauge group** — run_006 structural constraint binding all A-orientation ideas; cross-orientation convergence handle if F (First-Principles) attempts a gauge-theoretic argument for weight compression
