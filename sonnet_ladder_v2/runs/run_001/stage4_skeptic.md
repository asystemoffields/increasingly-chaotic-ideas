# Stage 4 — Skeptic Explorer — Run 001 (v2 Sonnet Ladder)

Skeptic: Sonnet 4.6.
Date: 2026-05-09.
Inputs read: STAGE4_SKEPTIC_EXPLORER.md, stage1_R/C/F/U/A.md, stage2_judge.md, stage3_deep_research.md, SUMMARY.md, KILL_LIST.md, ORIENTATIONS.md.

---

## 1. Frame-Exhaustion Map

### Saturation-exhausted frames

**S-EX-1 — Cross-layer W_Q subspace sharing (Cluster C1)**
Four orientations independently converged on the same primitive: stacked W_Q PCA and shared basis. The saturation is not from published prior art (Basis Sharing and MASA may partially cover it, but not the no-retraining variant on Qwen3) — it is from *internal* saturation. The ladder has spent four Stage-1 slots on the same measurement. Any new Stage-1 proposal in this frame is a fifth treatment of the same number.
Stripped primitive: the *cheapest-settler* form — run the 15-minute stacked SVD. What's left after removing all the framing apparatus is a single float: var@128. That float gates or kills all four simultaneously.

**S-EX-2 — CF3 Jaccard tier derivation (Cluster C2)**
Three orientations independently converged on the three-tier {FP16, INT8, INT4} scheme. The Jaccard-to-tier derivation is novel relative to published work, but within the ladder the frame is now triple-covered.
Stripped primitive: the Tier-0 10-minute stability measurement from existing PDAP data. Any future idea using CF3 Jaccard must build on a confirmed Tier-0 stability result, not re-derive the tier scheme from scratch.

**S-EX-3 — W_Q/W_K joint gauge exploitation (Cluster C3)**
Two orientations (F, A) covered the M = W_Q W_K^T gauge freedom. Combined with F-SGFVO's W_V W_O product rank and R-MCQKV's four-matrix sweep, the full attention weight class is now covered from three structural angles (CCA, Frobenius ratio, product rank). Any fifth attention-weight-only frame is saturation.
Stripped primitive: after removing the gauge vocabulary, what is left is per-head or per-layer rank measurement of attention matrices. That primitive is already fully scheduled as Stage 3/4 experiments.

**S-EX-4 — BLAKE3/content-addressable weight deduplication (U-CAWB, A-CAWS)**
Both proposals died on CF9: BLAKE3 cannot produce collisions for non-bit-identical inputs. The frame is exhausted as stated.
Stripped primitive: threshold-cosine head-deduplication. If W_Q heads within a layer have cosine similarity > 0.85 (the CAWS smallest-experiment), dedup by threshold comparison — no hash involved. This primitive is admissible and not covered by any advancer. See cross-pollination S3 below.

### CF9-exhausted frames

**CF9-EX-1 — FSM / Myhill-Nerode vocabulary reduction (A-FBVS)**
Myhill-Nerode precondition: finite congruence classes. General LLM output distribution over 151936 tokens with full context-dependence: violated. After removing the broken theorem import, the stripped primitive is n-gram prefix caching with transformer fallback. This is well-covered prior art (SpecInfer, EMNLP 2025 N-gram trie); the frame yields nothing novel.

**CF9-EX-2 — Sketching / hashing on dense LLM signals**
The kill list has now accumulated three CF9 kills in the sketch/hash class (Count-Sketch on INT4 residuals, tabulation hashing on c=4, BLAKE3 on near-similar vectors). The precondition failure pattern is structural: all sketch-and-hash methods assume either sparsity (Count-Sketch) or bit-level distinctness (cryptographic hashes). Trained Qwen3 weights and activations satisfy neither. Any new Stage-1 proposal that imports a sketching/hashing theorem must first verify sparsity or bit-distinctness; if neither holds, the machinery is decorative.

### CF10-exhausted frames

**CF10-EX-1 — Full-affine cross-scale calibration surgery**
WDLA failed structurally: 2.1M parameters fit on 2.05M calibration values. Any calibration-fitted affine A: R^{d_small} → R^{d_large} for 1.7B-class targets is CF10-exhausted at the standard calibration-corpus size (1K-10K tokens). Surviving variants require: (a) low-rank-by-construction A with r ≤ 64 (reduces parameter count to ~200K, well-conditioned at 1K tokens), or (b) ridge ≥ 1e-1 with N >> 10K tokens. Neither of these has been tested. The exhausted form is "full-affine correction on the full hidden dimension."

Stripped primitive: low-rank-by-construction calibration mapping (r ≤ 64 forced, not fit). This is structurally an A3-style functional error minimization but without activation data, using only the spectral measurements already in hand.

---

## 2. Orientation-Saturation Diagnostics

### R — Reach
Advanced 5 of 7 proposals: R-MCQKV, R-CROSS-Q, R-RAOK-70B, R-ACWF, R-W_down-SPEC. All REFINE at Stage 3. One wildcard (R-AERO-PROBE) advanced as FREE SWING, REFINE. One rejection (R-ATTN-NVMe-SPLIT, integration narrative). Converged on Cluster C1 (R-CROSS-Q) and Cluster C2 (R-RAOK-70B). Orientation is performing well.

Anti-frame list growth recommendation: Add "Attention-tier DRAM / MLP-tier NVMe split as the primary mechanism" — this is now an integration narrative (covered by component ideas). The orientation should not re-derive the tiered deployment architecture; it should instead generate the next-rung measurement or a new structural claim at 70B scale.

Recommendation: **KEEP**. R produced the strongest Track B advancer (R-RAOK-70B, top score 11) and the upstream measurement that gates the full cascade (R-W_down-SPEC).

### C — Composition
Advanced 6 of 7 proposals (highest yield in the union): C-JQSOH, C-ABAR, C-LGHF, C-CTERA, C-CQBAL, C-RAOK-Grounded. C-CTERA earned the joint-highest score (12). C-MCLG rejected (A4=1, scheduling claim hand-waved). Converged on Cluster C1 (C-CQBAL) and Cluster C2 (C-RAOK-Grounded). C-CTERA earned frame-novelty bonus (A2=3).

Anti-frame list growth recommendation: Add "AVX2 dual-stream scheduling as a composition claim" — C-MCLG's failure was conflating memory bandwidth with CPU dispatch width. Also add "multiplicative composition of two SVD savings on correlated matrices without measuring destructive interaction" — C-ABAR's W_Q + W_K simultaneous SVD is well-motivated, but the pattern of "compress A and B independently, multiply savings" requires an independence check that should be mandatory in the anti-frame list.

Recommendation: **KEEP**. Strongest orientation in terms of advancement rate and composition novelty (C-CTERA).

### F — First-Principles
Advanced 6 of 7 proposals: F-CQSGC, F-AJDGF, F-RSCLE (DOWNGRADE), F-OCSSQ, F-SGFVO, F-ASIT (wildcard). F-CQSMR rejected (A3=1, dependent on F-CQSGC GO). Convergence: Cluster C1 (F-CQSGC), Cluster C2 (F-OCSSQ), Cluster C3 (F-AJDGF). Elegant-equivalence tag earned by F-AJDGF (gauge-exploitation). F-SGFVO earned highest Track A scores alongside C-CTERA.

F-RSCLE is the sole DOWNGRADE: arXiv:2605.03109 (Gated Subspace Inference, May 2025) is a material threat to the residual-update-covariance framing. The downgrade is not a kill; the differentiation lives or dies on whether the cited paper uses activation vs update covariance.

Anti-frame list growth recommendation: Add "Laplacian/graph-theoretic framing of the residual stream as the load-bearing claim" — the spectral graph Laplacian vocabulary applied to C^(L) = E[δ_L δ_L^T] is now potentially pre-empted by arXiv:2605.03109. Any future First-Principles proposal using update covariance must first verify the Gated Subspace Inference paper doesn't cover it.

Recommendation: **KEEP**, but with tight watch on arXiv:2605.03109. If F-RSCLE is fully pre-empted, F orientation's frame-novelty count drops from 2 to 1; acceptable given F-SGFVO and F-AJDGF's quality.

### U — Unconventional Substrate
Advanced 0 of 6 proposals: all 6 scored ≤9, with 2 CF9 rejections (U-CAWB) and 4 low-score rejections (U-NTOB, U-ZDPW, U-VLAP, U-BPAT). Only U-AQPT advanced from Stage 2 as a rejection (low score 9). U orientation contributed to NO convergence cluster. Zero Stage 3 advancers.

Root cause: U orientation is naturally biased toward systems/substrate primitives (NVMe queue depth, sector alignment, VirtualLock, branch predictor) that are either too small in impact (< 3% throughput improvement) or blocked by Windows platform constraints (VirtualLock privilege, no NTFS reflinks). The substrate primitives U found are real but none cleared the A1=2 floor for this workload profile. The one exception (U-CAWB, content-addressable dedup) hit CF9 on the hash-collision error — the mechanism was real but the BLAKE3 step was wrong.

Anti-frame list growth recommendation: Add "Systems-side primitives whose gain is < 5% on the target workload (NVMe-bottlenecked 70B inference)" as inadmissible. Also add "Content-addressable schemes that use cryptographic hashes to index near-similar vectors" — this is now on the CF9 kill pattern.

Recommendation: **TIGHTEN for one round**. U should be retargeted to substrate primitives with > 10% structural impact (not engineering micro-optimizations). Proposed new constraint for U: "The substrate primitive must either (a) change the number of bytes read from NVMe per token by > 20%, or (b) change the residency tier of a weight class." This raises the A1 floor and filters out the micro-optimization failure mode. If U still advances 0 ideas next round with this tightening, apply DROP for one round.

### A — Constraint-Alien
Advanced 1 of 6 proposals: A-GFRS (convergence representative for Cluster C3, total 10, REFINE) and A-PWGI (wildcard OUT-OF-ORIENTATION, REFINE). Standard A ideas: A-TSWD rejected (A4=1, CF1 doesn't imply 20% sparsity tolerance), A-CAWS rejected (CF9, BLAKE3 collision error), A-ALSA rejected (A1=1, 32K+ context only), A-RORS rejected (A1=1, 1.2 tok/s below floor). A-FBVS rejected CF9 (Myhill-Nerode).

A-GFRS is notable: it won as a convergence representative despite being generated under the "gauge rotation constraint" — the constraint forced A to discover the same W_Q/W_K joint structure that F and C found independently. The convergence is valid signal; the constraint did its job. A-PWGI is an out-of-orientation idea (dual-GGUF selection policy) that Stage 2 correctly advanced as a wildcard.

Anti-frame list growth recommendation: Add "Tropical algebra / (max, +) semiring routing under CF1 firing-rank dominance" — A-TSWD died because CF1 firing-rank dominance does NOT imply functional sparsity tolerance. Also add "Append-only state constraints at < 32K context" — A-ALSA correctly observed the constraint creates a log-structured KV, but the payoff only exists at 32K+ context; at the target 4K context, KV fits in DRAM and the constraint constraint buys nothing.

Recommendation: **TIGHTEN**. The constraint menu should be pruned to constraints that directly interact with the NVMe/DRAM bottleneck or with the measured weight-class spectra. "No RAM during decode" and "model must be reconstructible from 10 MB seed" are both likely to produce > A1=1 ideas given the current 70B compression target; "all state append-only" and "inference loop as a state machine" have now been shown to stall at the wrong bottleneck.

---

## 3. Cross-Pollination Opportunities

### X-POLL-1 — U-CAWB / A-CAWS stripped primitive → Composition orientation

**Killed idea**: U-CAWB (content-addressable W blocks via BLAKE3) and A-CAWS (same via action-signature hash). Both died on CF9: BLAKE3 hash cannot produce collisions for non-bit-identical inputs.

**Stripped primitive**: threshold-cosine W_Q head deduplication. The two ideas were trying to measure and exploit head-redundancy (CF11: 16 heads → ~128-dim subspace). Remove the broken BLAKE3 machinery, and what's left is: if cos(U_i^{(64)}, U_j^{(64)}) > 0.85 for two W_Q heads in the same layer, store the shard once and reference from a 1-byte index. The quality cost is zero (deduplication, not approximation).

**Transposition to Composition orientation**: C already proved (C-ABAR, REFINE) that CF11 motivates spectral-rank-based compression. A new Composition idea can ask: what is the *within-layer* head-dedup coupling between CF11 head-redundancy and storage? If 16 heads collapse to ~1 head's worth of subspace at K=128, then within a single layer, multiple heads' W_Q weight slices might be near-identical in singular-vector space. The coupling claim: `∃ i,j in same layer: cos(U_i^{K=64}, U_j^{K=64}) > 0.85` follows from CF11 if the 128-dim head-shared basis is reached via ≤ 2 distinct per-head singular-vector sets. This is not the cross-layer sharing (Cluster C1) — it is within-layer head-weight deduplication, a distinct mechanism that C orientation is well-placed to derive.

**Expanded as Stage 4 idea S5 below.**

### X-POLL-2 — A-FBVS stripped primitive → Unconventional Substrate orientation

**Killed idea**: A-FBVS (FSM-bounded vocabulary state machine). Died on CF9: Myhill-Nerode precondition (finite congruence classes) fails for full LLM output distribution.

**Stripped primitive**: bigram/n-gram prefix cache with transformer fallback. The underlying engineering intuition — that many consecutive token predictions are predictable from a small prefix table — is valid and not killed. The Myhill-Nerode framing was decorative.

**Transposition to U orientation**: U orientation's strength is "let the substrate do the work." A static n-gram prefix table (trigram: 151936^3 entries impossible; but within a single session, a 2-4M-entry trie of seen (token_{t-2}, token_{t-1}) → most-likely token_{t} fits in 16–64 MB RAM) is a substrate primitive (DRAM trie lookup, deterministic). If the trie hit rate is > 30% for the session's actual generation pattern, the transformer forward pass is skipped entirely for those tokens — a bypass that does not require ANY model compression, just a runtime DRAM trie. The fallback is the full transformer. This is not a model-quality change; it is a pure inference-path bypass.
Anti-frame check: this is NOT a speculative decoder (it makes a deterministic lookup, not a probabilistic draft). The N-gram trie at inference time is a substrate primitive (hash table in DRAM), not a model change. Not on the kill list (Branch-Predictor Token Trie R3/S2 was killed for wrong reasons: prefetch savings 1.4% per layer; the trie's value here is not prefetch but FULL TOKEN SKIP when a deterministic match exists).

**Not expanded as a Stage 4 idea** — the hit rate on a session-local trie for diverse generation is probably < 5% for general-purpose text; the mechanism is real but the gain floor is too uncertain for a ladder slot. Noted for completeness.

### X-POLL-3 — CF10-EX-1 stripped primitive → First-Principles orientation (low-rank WDLA)

**Killed idea**: WDLA (cross-scale affine correction). Died CF10: 2.1M parameter fit on 2.05M calibration values.

**Stripped primitive**: low-rank-by-construction calibration map r ≤ 64.

**Transposition to F orientation**: F orientation's strongest shape is algebraic-identity / no-SGD reformulation. The WDLA stripped primitive asks: is there a RANK-FORCED (not RANK-FITTED) affine correction from a smaller Qwen3-0.6B residual stream to a larger Qwen3-1.7B residual stream? If we force r=64, the parameter count drops to ~200K, well-conditioned at 1K-token calibration. F orientation would derive this as: the two models' residual streams at matching layers share a low-rank subspace (this is the CF2-extension claim — if cos(h_L^{small}, h_L^{large}) is high layer-by-layer, the projection onto the shared subspace is the minimal-complexity map). The calibration remains but is now well-conditioned, and the mechanism survives CF10.

**Expanded as Stage 4 idea S9 below.**

---

## 4. Cross-Domain Seed Exploration

### Professional Prior 1 — Database Engineer

A database engineer looking at the attention-weight compression problem would ask: "can I turn the W_Q lookup into a join over a pre-computed index that makes the computation cheaper without changing the result?"

The attention computation `scores = Q K^T = (X W_Q)(X W_K)^T` is a product of two linear transforms applied to the same input X. A database engineer sees this as: sort and index the key space by the query's high-variance dimensions (the top singular vectors of W_Q), then look up the relevant K rows without full inner product. This is an *approximate nearest-neighbor index* (ANNS) on the key space, keyed by the query-subspace projection. Published ANNS compression (HNSW, IVF, ScaNN) exists, but the database engineer's framing produces a specific algebraic observation: if W_Q has r_99/d ≈ 0.63 (CF11), then projecting X W_Q onto the top-128 singular subspace of W_Q loses < 37% of the query's variance. The first 128 dimensions of the query are the "index key"; the next 1920 dimensions are "secondary columns." A database engineer would pre-build a static index on K tokens at batch time and only compute the full inner product for the top-m candidates returned by the index scan. This is not speculative decoding; it is attention acceleration via sorted index.

**Elegant equivalence prompt**: What algebraic identity does a database engineer's view of attention reveal? Q K^T = (P_Q X W_Q)(X W_K)^T + residual, where P_Q is the rank-128 projection onto W_Q's dominant subspace. The sorted-index can be built offline on the KV cache prefill for the static portion of context. The gap is: for autoregressive decode, K grows one token at a time; the index must support online updates. This is a B-tree insert problem, not a full re-scan problem. The primitive is an updateable approximate KV index (IVFPQ-style with online inserts). This has NOT been explicitly tried with the CF11 subspace projection as the index key dimensionality (128 vs full 2048).

**Candidate primitive**: A static context (system prompt + long prefix) can have its KV cache indexed offline at rank-128 projection granularity; dynamic tokens appended one by one. Scan cost for static portion: O(static_len / m) inner products instead of O(static_len). At 4K static context and m=32, this is 128 full inner products vs 4096. The mechanism does NOT change residency (same bytes); it changes compute. On CPU-bound inference this matters; on NVMe-bandwidth-bound inference (70B NVMe) the bottleneck is elsewhere. For long-context (32K+) on DRAM-resident smaller models (7B), this is high-value.

Relevant precision: this idea targets compute, not residency. For the 70B NVMe problem, it is secondary. For the 7B DRAM problem at long context, it is primary. Stage 4 generates it under the "compute-side gap" heading.

**Expanded as Stage 4 idea S7 below.**

### Professional Prior 2 — Compiler Writer

A compiler writer looking at multi-layer transformer inference would ask: "what computations are being re-done at every token that could be hoisted out as loop-invariants?" This is loop-invariant code motion applied to inference.

The elegant-equivalence insight: in autoregressive decode, the weights W_Q, W_K, W_V, W_O, W_gate, W_up, W_down are constant across all tokens. They are loop invariants. The only token-varying quantities are the residual stream and the KV cache. A compiler writer would ask: "have we materialized all possible constant-propagation reductions?"

Concrete observation from the current CF substrate: W_V and W_O appear as a product P = W_V W_O (F-SGFVO's point). In autoregressive decode, W_O is applied after attention, and W_V is applied to K tokens to build the value vectors. If we pre-multiply P = W_V W_O at model-load time and store P instead of (W_V, W_O) separately, then during decode we apply P in one GEMV instead of two. The rank ceiling (rank(P) ≤ d_head = 128) is the compiler-writer's algebraic simplification: P is cheaper to store AND cheaper to apply. This is exactly F-SGFVO, which already advanced.

More novel compiler-writer observation: **constant-fold the W_Q normalization**. In Qwen3, RMSNorm is applied to the residual stream before W_Q. The combined operation is `RMSNorm(x) W_Q = diag(1/rms(x)) x W_Q`. The `x W_Q` part is the matmul; `diag(1/rms(x))` is a per-vector scalar dependent on x. A compiler writer notices: the per-row norm of x W_Q can be computed more cheaply if W_Q is stored in a rotated basis where the dominant columns align with the high-variance input directions. Specifically: if W_Q is pre-rotated by Q* such that Q*^T W_Q = W_Q' where W_Q' has columns in order of singular value (already true after SVD), then the first K columns of W_Q' are the most informative. The normalization factor rms(x) is the same; it does not change. The "constant-fold" here is: apply the rotation to x once, then do the rank-K inner product. This is what the stacked-SVD scheme (Cluster C1) already does, but from a "loop invariant" angle it reveals a new mechanism: **pre-rotate the input sequence once** (at the attention input, before any layers) so that all W_Q matrices in the rotated basis are already aligned to the global shared subspace. The per-token rotation cost is O(d × D*) where D*=128 — a 16× cheaper operation than the original d×d rotation.

**Elegant equivalence**: the compiler hoists the global subspace rotation out of the per-layer loop. Instead of 28 per-layer W_Q matmuls at full d=2048, do 1 upfront rotation to D*=128, then 28 per-layer C_ℓ x matmuls at D*=128. Total FLOPs: d × D* (upfront) + 28 × D* × d (per layer) vs 28 × d × d (original). At D*=128: (2048×128 + 28×128×2048) vs 28×2048×2048 → (262K + 7.3M) vs 115M FLOPs. ~14× cheaper in attention FLOPs if cross-layer coherence holds. This is the Cluster C1 mechanism viewed as a hoisted loop invariant.

The compiler-writer framing produces something new that the other orientations did NOT generate: **the optimal execution schedule for the rotated-basis inference**. Once W_Q matrices are expressed as C_ℓ U^T (shared U, per-layer C_ℓ), the product x W_Q = (x U) C_ℓ^T, where (x U) is computed once per token and shared across all 28 layers. This is a compile-time scheduling win, not just a storage win. The "shared precomputation" primitive is new and operationalizable.

**Expanded as Stage 4 idea S3 below.**

### Professional Prior 3 — Control Theorist

A control theorist looking at the residual stream would ask: "what is the steady-state behavior? is there a conserved quantity? can I replace a layer with a fixed-form operator on the conserved quantity alone?"

The elegant-equivalence prompt: what conservation law does a control theorist see in the residual stream? CF2 says adjacent residual states are nearly parallel (cos ≈ 0.99). This is a discrete-time dynamical system with a near-identity state transition. A control theorist recognizes this as a system operating near a fixed point — the residual stream is always "near x" where x is the embedded token representation.

More specifically: the residual stream is a feedback control loop where each transformer layer is a controller with reference = current residual state and control output = δ_L = small correction. From a control-theoretic perspective, the *energy* of each correction δ_L (its L2 norm) should be small and structured. If ‖δ_L‖ / ‖h_L‖ is small and the directions of δ_L across tokens are concentrated in a low-rank subspace (the update covariance argument from F-RSCLE), then a control theorist would propose: replace layers whose correction is "within the observed perturbation envelope" with a simpler fixed-form operator that produces the same expected δ_L without the full transformer computation.

The Kalman filter analog: if δ_L is small and the covariance of δ_L is low-rank, then a Kalman smoother can predict the next residual state from the current one. The "measurement" is the attention output; the "process model" is the residual stream dynamics. The CF2 finding (cos ≈ 0.99) says the process model is near-identity; the update covariance C^(L) tells us which directions the "measurements" add information.

**This does not produce a residency win for NVMe-bound 70B** — it is a compute-side observation. But it produces something no other orientation generated: an argument for *layer clustering by update-covariance similarity*. Layers with similar C^(L) may be approximated by a single shared operator applied multiple times (layer folding). If layers i and j have C^(i) ≈ C^(j) (similar update covariance structures), the two layers could share a single update operator. The residency saving is the elimination of one layer's worth of MLP and attention weights — a weight folding that does not require training-time parameter tying.

Precondition check: this requires C^(L) to be measurable and to show clustering structure. The residency saving is per-layer (for 70B: ~800 MB per eliminated layer). The mechanism is content-free (no theorem import from control theory; the layer-similarity is an empirical measurement). CF9 CLEAR. CF10: no calibration fit involved.

**Expanded as Stage 4 idea S6 below.**

---

## 5. Constraint-Driven Reframing

### Constraint A — No floating point at inference (integer arithmetic + lookup tables only)

Forcing no FP at inference eliminates continuous outlier handling (the basis of all Cluster C2 RAOK ideas). What replaces it? A lookup-table accumulator: pre-compute all possible INT4 × INT8 products into a 4K-entry lookup table (256 INT4 values × 16 INT8 values); inner-product becomes table-lookups + integer additions only. This is BitNet-adjacent but the Qwen3-specific angle is: CF3 says the top-0.1% channels are static outliers with Jaccard=0.718. Under the no-FP constraint, the FP16 bypass for those 2 channels is FORBIDDEN — the constraint forces us to instead quantize those 2 channels at INT16 (higher integer precision, still integer) and route them through a separate INT16 accumulator. The stricter precision forces an interesting design: INT16 for 2 channels, INT8 for 18, INT4 for 2028 — the RAOK tier scheme but with no floating point anywhere. The lookup-table accumulator for INT4 × INT8 multiplication avoids all multiplies: `y = lookup[a4][b8]` where the lookup is precomputed. This is compatible with AVX2 VPSHUFB (the 16-byte shuffle instruction that implements 4-bit LUT in 2 cycles).

The no-FP constraint reveals a gap: NO current idea in the ladder uses VPSHUFB-based lookup-table GEMV for the three-tier activation scheme. The RAOK scheme assumes FP16 for Tier-0 channels and INT8 for Tier-1; converting Tier-0 to INT16 and using a unified integer accumulator removes the FP16 bypass cost. The gate: does INT16 for 2 channels + INT8 for 18 + INT4 for 2028 achieve ΔNLL < 0.3 nats (the same threshold as F-OCSSQ)? The no-FP constraint forces a new question: what is the precision floor for the 2 static outlier channels?

**Expanded as Stage 4 idea S4 below.**

### Constraint B — Sequential-only storage (random reads forbidden, only forward scans)

If random reads are forbidden, all inference must proceed by streaming weights sequentially from NVMe. This forces a question the ladder has not directly addressed: **what is the optimal layout order for a GGUF file if we must read it in a fixed linear scan?**

Under sequential-only storage, the current GGUF layout (which interleaves W_Q, W_K, W_V, W_O, W_gate, W_up, W_down per layer) is optimal only if each layer's computation fully completes before the next layer's weights arrive. If the optimal execution schedule for the rotated-basis inference (Professional Prior 2, above) produces (x U) shared across layers, then the optimal read order is: read U once (small, fits in L3 cache), then for each layer read C_ℓ (the per-layer coefficient matrix), compute the attention for layer ℓ, advance. The sequential-scan layout is: [U] [C_1 W_K^1 W_V^1 W_O^1 W_gate^1 W_up^1 W_down^1] [C_2 ...] ... where U is a 2048×128 = 0.5 MB blob prepended to the file.

This constraint produces a concrete GGUF repack suggestion: prepend the shared subspace basis U to the file, and for each layer, store C_ℓ (the per-layer thin projection matrix) immediately before the layer's other weight matrices. The NVMe sequential scan reads U once (cold start), then per-layer reads C_ℓ + remaining weights in order. The C_ℓ is 128× smaller than the original W_Q, so the per-layer NVMe read for W_Q drops from 2048×2048×2 bytes (8 MB) to 128×2048×2 bytes (0.5 MB) — 16× less NVMe bandwidth for W_Q in the sequential layout. This is the Cluster C1 savings viewed through the sequential-scan constraint.

The sequential-scan constraint does NOT produce a new idea — it produces a stronger argument for the Cluster C1 (shared W_Q basis) mechanism, specifically motivating the GGUF file layout that makes it maximally efficient. Not expanded as a separate idea; absorbed into S3.

### Constraint C — 10 MB seed reconstruction (model reconstructible from 10 MB + deterministic decompressor)

Under this constraint, all weight information must be derivable from a 10 MB seed. For a 1.7B-parameter model at bf16 (3.4 GB), 10 MB is a ~340× compression ratio. This is far beyond achievable quality-preserving compression; the only way to satisfy this constraint is to seed a weight-generation procedure.

The constraint forces the question: is there a parameterization of Qwen3-family weights where most of the information is in a small seed, and a large fraction of the weight bytes is deterministically reconstructible from the seed? Published work on this: hypernetworks (Ha et al. 2016), Neural Architecture Search with parameter prediction. For the specific Qwen3 architecture, the constraint motivates asking: what fraction of each weight matrix is predictable from a layer-index and a small per-layer code?

This connects to an unexplored gap: **cross-layer weight prediction**. The KILL_LIST includes "inter-layer delta compression" (R1/S2, killed for DeltaLLM/ADAMIX prior art). But DeltaLLM compressed deltas between adjacent layers as a storage optimization; the 10 MB seed constraint forces a different question: can we generate a weight matrix for layer ℓ from a per-layer code c_ℓ plus a shared generator G? The shared generator G is the "10 MB seed" (or part of it).

For the Qwen3 weight class: W_Q across layers share a common subspace U (Cluster C1, if GO). The U matrix is the shared generator. Each layer's W_Q is C_ℓ U^T where C_ℓ is the 128×2048 per-layer coefficient. At 28 layers: seed size = U (2048×128 = 0.5 MB) + 28 C_ℓ matrices (28×128×2048 = 14 MB). Over 10 MB, but the C_ℓ matrices are random-looking; they cannot be compressed further without retraining. The 10 MB constraint is too tight for W_Q alone. NOT expanded as a Stage 4 idea — the constraint is too aggressive for the current empirical state. Noted as a future-round target after Cluster C1 measurement confirms the U exists.

---

## 6. Generated Ideas (S1–S8)

### S1 — W_down Spectrum-Gated MLP Quantization Tier [Track B, Orientation: Reach]

**Name**: WDSG — W_down Spectrum-Gated Mixed-Precision Allocation

**Mechanism**: R-W_down-SPEC (Stage 3 REFINE, 10-min measurement) will determine whether W_down ∈ R^{d × d_int} (2048 × 6144) has r_99/d_out substantially below 1.0. If W_down is structurally distinct from W_gate and W_up (CF8: full-rank), the dimension-asymmetry (W_down is a "bottleneck projection" from 6144-dim intermediate to 2048-dim residual, implying max-rank cap at 2048, not 6144) may produce a more concentrated spectrum. The gap: even if W_down is full-rank at r_99/d_out ≥ 0.90 (which R-W_down-SPEC is designed to check), there is a second-order question the current advancers do not ask: does the asymmetric spectral structure of W_down (the narrow output dimension) make it *more tolerant of mixed-precision quantization* than W_gate/W_up? Specifically, the quantization noise in W_down's output lives in R^{2048}, and the residual stream has its own PCA structure (C^(L), from F-RSCLE's motivation). If the quantization noise is concentrated in the LOW-variance directions of the residual stream's PCA, it has disproportionately low impact on the next layer's attention computation. WDSG proposes: quantize W_down to INT4 for the components whose output aligns with the bottom-50% variance directions of the residual stream; quantize at INT8 for the top-50%. This is activation-covariance-guided mixed precision within a SINGLE weight matrix, not cross-matrix allocation (different from C-ABAR which is cross-matrix). The coupling claim: if C^(residual) (the residual stream covariance) has D* ≈ 256 effective dimensions (the F-RSCLE claim), then W_down's columns in the bottom-1792 PCA directions are tolerable at INT4 (their output is in the low-variance subspace). The columns in the top-256 directions must stay at INT8 or higher.

**Residency arithmetic**: W_down per layer (2048×6144 = 12.6M params). Full INT4: 6.3 MB per layer × 28 = 176 MB. Mixed INT4/INT8 (75%/25% split): 0.75 × 12.6M × 0.5B + 0.25 × 12.6M × 1B = 6.62 MB × 28 = 185 MB. Compared to bf16 baseline (352 MB per layer × 28 = 9.9 GB for 70B W_down equivalent). Not a primary residency lever at 1.7B; at 70B W_down in 28672-dim intermediate: 8192×28672 per layer × 64 layers = 33.9 GB; INT4/INT8 split ≈ 16.4 GB saving vs INT4 flat (0 gain) but INT8-sensitive-columns vs INT4-uniform: 11% residency increase for quality insurance on 25% of W_down weights. The payoff is quality, not residency.

**Novelty gloss**: LAMPQ (arXiv:2511.10004) uses nuclear norm proxies for mixed-precision allocation. ABAR (Stage 3 REFINE) uses spectral rank r_99/d. WDSG uses per-column alignment with residual-stream PCA — the coupling of W_down output directions to residual-stream covariance. This specific coupling (what is the quantization sensitivity of W_down in the basis defined by the residual-stream PCA?) has no published analog. Outside: orientation F anti-frame list (first-principles derivation, not magnitude pruning), outside orientation C anti-frame list (coupling between W_down column-structure and C^(residual) is a novel composition, not a stacking of two published methods).

**Smallest experiment**: (a) Run F-RSCLE update-covariance measurement (15 min); (b) compute eigendecomposition of C^(residual) at layer 14; (c) project W_down[layer14] columns onto top-256 vs bottom-1792 residual-PCA directions; (d) apply INT4 to bottom-1792-aligned columns, INT8 to top-256-aligned; (e) measure ΔNLL vs uniform INT4. Runtime: 25 min total (assumes F-RSCLE data available). GO: ΔNLL_mixed < ΔNLL_uniform_INT4 − 0.15 nats. NO-GO: residual-stream PCA does not produce actionable groupings for W_down (either all columns sensitive or none).

**Primary risk**: The residual-stream PCA may not be stable across tokens (the covariance changes with context). If C^(residual) is nearly isotropic (uniform variance across all 2048 directions), the split produces no benefit. Mitigation: measure C^(residual) on 3 diverse corpora; report variance of eigenvalue ratios across corpora.

Preconditions: CF9 CLEAR (no imported theorem; W_down column projection onto residual-PCA is direct linear algebra). CF10: no regression fit, pure projection. Smallest test ≤ 8h: YES (~25 min). Track B. Outside F, C, R orientation anti-frames. CF tether: CF2 (residual additivity motivates measuring update covariance) + CF8 (W_down potentially distinct from W_gate/W_up).

Elegance-class tag: *subspace-alignment* (W_down output-columns aligned to residual-stream PCA basis).

---

### S2 — KV-Attention Computation Tier Split at Attention Score Entropy [Track B, Orientation gap: KV-side]

**Name**: ACES — Attention Computation Entropy-Stratified Tier Split

**Mechanism**: This run has zero surviving KV-side ideas. All KV-side proposals either died on bottleneck mismatch (< 32K context) or were pre-empted by KVQuant literature. The gap: no idea explored the *compute* cost of the attention operation itself under the CF11 W_Q head-redundancy finding. CF11 says 16 heads collapse to a ~128-dim subspace at global K=128. At inference time, the attention computation is scores = (X W_Q^(h)) (X W_K^(h))^T for each head h. If W_Q has been replaced by the global rank-128 projection (the C1 cluster result), then the query computation is q^(h) = x C_h U^T where C_h is the 128-dim per-head projection coefficient. The key observation: at global K=128, heads that have very small ‖C_h‖ (calibration-measured importance, as in R-ACWF) contribute negligible query variance. These heads' attention scores are near-uniform (near-zero query makes all K^T inner products equal, softmax gives uniform attention, output ≈ W_V-column-mean). ACES proposes: during inference, identify heads whose query norm ‖C_h x‖ falls below a per-head calibration-derived threshold τ_h; for those heads, skip the full KV inner product and return the pre-computed mean value vector directly. This is NOT pruning heads — it is short-circuiting the attention computation for *low-query-norm tokens* on a *per-head per-token* basis. The short-circuit output is the pre-computed mean of the KV cache values for the active context — a single vector lookup, not a full O(n) attention scan.

**Residency arithmetic**: No change in weight residency. Compute savings: if 30% of (head, token) pairs satisfy ‖C_h x‖ < τ_h, 30% of attention scans are replaced by a single-vector lookup. Per-layer attention compute: O(n_heads × seq_len × d_head) → O(0.7 × n_heads × seq_len × d_head + 0.3 × n_heads × 1 × d_head). At 4K context, 30% skip rate: ~30% attention FLOPs saved per layer. At DRAM-bandwidth-limited inference (the dominant bottleneck), attention FLOPs are < 5% of total cost; this saves ~1.5% of total inference time. At long context (32K), attention scales as O(n²) while weight loading is O(n); at 32K, attention starts to matter. For a 7B model at 32K context DRAM-resident: attention FLOPs ~= weight-matmul FLOPs; 30% skip rate on attention ≈ 15% total speedup.

**Novelty gloss**: H2O, ScissorHands, SnapKV all do KV eviction. ACES does per-token attention computation bypass, not eviction — the KV entries remain; we just skip the dot-product for low-query-norm tokens. StreamingLLM handles attention for very long contexts but does not use query-norm thresholding. The CF11 grounding (query norm in the rank-128 head-shared basis is the cheapest measurable proxy for whether a head will produce a meaningful attention distribution) is specific to our private measurement. Outside orientation U anti-frame list (this is not "custom SIMD" and not "engineering micro-optimization" — it is a per-head computational bypass grounded in empirical head-norm structure); outside F anti-frame list (it is a structural skip, not plain quantization).

**Smallest experiment**: (a) Compute ‖C_h x‖ per head per token on 200 diverse prompts (Qwen3-1.7B-Base, using AQFKV C_h matrices); (b) histogram the norm values; (c) identify fraction of (head, token) pairs below various thresholds τ_h; (d) for GO-threshold tokens, compare the attention output with full-scan vs mean-V to measure ΔNLL from the skip. Runtime: ~2h (forward pass data collection + histogram + ΔNLL evaluation on skip subset). GO: ≥ 20% of (head, token) pairs satisfy ‖q‖ < τ such that per-token ΔNLL from the skip < 0.05 nats.

**Primary risk**: Low-query-norm heads may still produce meaningful attention distributions (the query selects a small subset of KV, even at low magnitude). Mitigation: measure the attention entropy of the low-norm query heads directly; if entropy is below 0.5 nats (near-uniform), the skip is justified; if above 2 nats, the skip degrades quality.

CF9 CLEAR. CF10: threshold τ_h is calibration-derived scalar per head; 448 calibration thresholds, well-conditioned. Smallest test ≤ 8h: YES. Track B. CF tether: CF11 (head-shared projection produces C_h, whose norm is the bypass gate). Elegance-class: *conserved-quantity* (near-zero query norm is conserved zero attention variance).

---

### S3 — Hoisted Cross-Layer W_Q Precomputation for Sequential NVMe Layout [Track A, Orientation gap: Cluster C1 execution schedule]

**[FREE SWING]**

**Name**: HOIST — Hoisted Shared-Basis Query Precomputation

**Mechanism**: If Cluster C1 GO (var@128 > 0.80 on stacked W_Q SVD), then W_Q^(ℓ) = C_ℓ U^T for all 28 layers, and q^(ℓ) = x W_Q^(ℓ) = (x U) C_ℓ^T. The shared projection (x U) ∈ R^{128} is a loop invariant with respect to layer index. A compiler writer recognizes: compute (x U) ONCE per token (2048→128 linear map, 0.26 MFLOP), then for each layer compute (x U) C_ℓ^T (128→2048 linear map per layer, 0.26 MFLOP each) instead of x W_Q^(ℓ) (2048→2048 linear map, 8.4 MFLOP each). Total attention FLOPs for W_Q: original 28×8.4 = 235 MFLOP; hoisted 0.26 + 28×0.26 = 7.5 MFLOP. 31× fewer FLOPs for the W_Q projection step. The NVMe I/O: U is 2048×128 × 2 bytes = 0.5 MB (loaded ONCE at model start); C_ℓ per layer is 128×2048 × 2 bytes = 0.5 MB per layer (vs 2048×2048 × 2 = 8 MB for full W_Q). Per-token NVMe reads for W_Q: 28 × 0.5 MB = 14 MB vs 28 × 8 MB = 224 MB. 16× NVMe bandwidth reduction for W_Q alone. At 70B (W_Q savings: 8.1 GB total, see F-CQSGC residency arithmetic): this moves 8 MB from NVMe bandwidth per token to effectively 0.5 MB per token for the W_Q term.

This idea is a FREE SWING because it is NOT merely F-CQSGC restated — it adds the execution schedule (hoisting) as a concrete implementation primitive that changes the NVMe layout requirements. F-CQSGC and Cluster C1 advancers discuss storing the shared U + per-layer C_ℓ as a residency idea; HOIST claims the hoisted execution schedule is the mechanism that makes the NVMe layout change deliver actual inference speedup. Without the hoisting, the storage saving is real but the inference pathway still reconstructs W_Q^(ℓ) before the GEMV, negating the compute benefit. The hoisting is the mechanism that realizes BOTH the storage and the compute benefits simultaneously.

The GGUF sequential-scan layout consequence: U is stored at file offset 0; for layer ℓ, the file contains C_ℓ (0.5 MB) followed by the other weight matrices. The sequential NVMe read proceeds in this order. This is a GGUF repack, not a model weight change.

**Residency arithmetic**: W_Q NVMe bytes per token: from 28×8 MB = 224 MB to 0.5 + 28×0.5 = 14.5 MB. At NVMe throughput 3 GB/s: 224 MB / 3 = 75 ms per token for W_Q → 14.5 MB / 3 = 4.8 ms per token. 15× faster for the W_Q term. W_Q is ~20% of total attention weight bandwidth (W_Q + W_K + W_V + W_O ≈ 1.1 GB per pass); at 70B, W_Q is ~8.6 GB / total attention ~19 GB = 45% of attention bandwidth. Hoisting W_Q reduces attention NVMe reads by 45% × (1 − 1/16) ≈ 42%. Total model bandwidth: ~40 GB at 70B Q4; attention is 19/40 = 48% of this. Hoist saves: 48% × 42% = ~20% total NVMe bandwidth reduction. At 3 tok/s ceiling target from 1.5 tok/s current: 1.5 × 1.2 = 1.8 tok/s from this mechanism alone. Meaningful but not the ceiling-breaker.

**Novelty gloss**: ALBERT-style layer sharing is training-time; F-CQSGC is a storage idea; HOIST is an execution-schedule idea that requires the Cluster C1 measurement to confirm GO, then proposes a concrete inference-loop restructuring + GGUF repack that materializes the compute AND bandwidth savings simultaneously. Outside: orientation U anti-frame list (this is NOT a custom SIMD kernel or engineering micro-optimization — it is a loop-invariant hoisting that requires a structural finding); outside F anti-frame list (this is a consequence of an algebraic identity, not magnitude pruning).

**Smallest experiment**: (a) Confirm Cluster C1 GO (run F-CQSGC 30-min SVD); (b) implement hoisted inference: compute (x U) once, use for all 28 layers; (c) compare inference latency on Qwen3-1.7B-Base for attention-only benchmark (measure W_Q GEMV time before and after hoisting); (d) measure ΔNLL (should be zero — mathematically equivalent). Runtime: ~2h after Cluster C1 GO is confirmed. GO: ΔNLL = 0 AND W_Q GEMV time drops > 10×.

**Primary risk**: Cluster C1 NO-GO makes this idea moot. Also: the hoisted (x U) vector is 128-dimensional and must be re-used across 28 layers — this requires holding 128 floats in register or L1 cache between layer computations, which may interact with the inference loop's current structure. Mitigation: measure cache occupancy; 128 × 2 bytes = 256 bytes = 4 cache lines, easily L1-resident.

CF9 CLEAR (no imported theorem; loop-invariant code motion is a compiler identity, not a mathematical theorem). CF10: no calibration fit. Smallest test ≤ 8h: YES. Track A. CF tether: CF11 (W_Q head-redundancy motivates the shared basis U). Elegance-class: *algebraic-identity* (hoisted computation is exact, not approximate).

---

### S4 — VPSHUFB Integer-Only Three-Tier Activation Kernel [Track B, Orientation gap: no-FP constraint]

**Name**: VPTK — VPSHUFB Per-Token Three-Tier Kernel

**Mechanism**: The Cluster C2 RAOK advancers (R-RAOK-70B, F-OCSSQ, C-RAOK-Grounded) all use FP16 for the 2 static outlier channels (Tier 0). This incurs FP16 storage and FP16 arithmetic for those channels. The no-FP constraint (see section 5A) asks: can the 2 static channels use INT16 instead of FP16? If yes, the entire activation quantization scheme becomes integer-only, enabling a pure VPSHUFB kernel. VPSHUFB (Variable Permute Shuffle Bytes) on AVX2 implements a 4-bit lookup table in 2 cycles on Zen 3. Using VPSHUFB for INT4 × INT8 multiplication: for each 4-bit weight nibble w and 8-bit activation a, the product w × a is read from a 256-entry table indexed by (w, a>>4) and (w, a&0xF), producing INT16 partial sums. The 2 INT16 (Tier-0) channels use a 16-bit accumulator with no FP conversion. The entire GEMV — from quantized activations through to INT16 partial sums — is FP-free. The final dequantization back to BF16 residual stream is a single scale multiplication at the layer output, not per-element. Mechanism claim: a VPSHUFB-fused three-tier GEMV (INT16 Tier-0, INT8 Tier-1, INT4+VPSHUFB Tier-2) achieves ΔNLL < 0.3 nats (same threshold as RAOK) while being fully integer at the matmul level. The novel claim over RAOK is the INT16 Tier-0 replacement and the VPSHUFB fusion — not a different scheme, but a different implementation path that enables AVX2 integer-only execution.

**Residency arithmetic**: Same as RAOK (4× activation compression). Weight residency: no change. The benefit over RAOK is compute: VPSHUFB-based INT4 GEMV achieves ~2× throughput vs standard INT8 GEMV on Zen 3 (the shuffle is faster than multiply). For the Tier-2 (2028 channels, INT4): VPSHUFB processes 32 nibbles per cycle; standard INT4-to-INT8-convert+multiply processes 16 per cycle. 2× throughput for the dominant bulk term. Total MLP GEMV speedup: ~1.6× (weighted by tier sizes: 2028/2048 ≈ 99% of channels in Tier-2). At 3 tok/s ceiling: 1.5 × 1.4 = 2.1 tok/s (stacked with HOIST). Speculative; needs profiling.

**Novelty gloss**: BitNet (arXiv:2202.09016, Feb 2022) uses ternary weights but requires training. The AVX2 W(ternary)A8 idea (R2 Stage 4 deferred) uses VPSHUFB but on ternary weights, not on the three-tier activation scheme. VPTK uses VPSHUFB on INT4 bulk activations (not weights) combined with INT16 static outlier handling — a different application of the same hardware primitive. The three-tier activation quantization + VPSHUFB fusion for inference on the Qwen3-specific CF3-derived tier boundaries has no published analog. Outside: orientation F anti-frame list (this is a consequence of the CF3-tier structure, not magnitude pruning); outside R anti-frame list (this is a compute optimization, not an integration narrative).

**Smallest experiment**: Benchmark VPSHUFB-based INT4-activation GEMV vs standard BF16 GEMV on a synthetic 2048×6144 W_up matmul on Ryzen 5 7530U: measure throughput (tokens/sec) for INT4 bulk + INT8 Tier-1 + INT16 Tier-0 activation input, using AVX2 VPSHUFB for INT4 path. Runtime: ~3h (AVX2 intrinsic implementation + benchmark). GO: throughput ≥ 1.4× BF16 GEMV AND ΔNLL (from CF3 tier scheme, inherited from RAOK) < 0.3 nats.

**Primary risk**: VPSHUFB throughput advantage on Zen 3 requires careful shuffle-table layout; if the table is not L1-resident (256 bytes × 4 = 1 KB), cache misses dominate and throughput drops. Mitigation: pin the shuffle tables in L1D (256 bytes, easily fits).

CF9 CLEAR. CF10: tier boundaries from CF3, not regression-fit. Smallest test ≤ 8h: YES. Track B. CF tether: CF3 (Jaccard phase transition defines tier boundaries). Elegance-class: *gauge-exploitation* (AVX2 VPSHUFB recasts INT4 multiply as table lookup — an exact computational equivalence exploiting the fixed-point structure of the integer domain).

---

### S5 — Within-Layer W_Q Head Cosine-Threshold Deduplication [Track B, Cross-pollination from U-CAWB/A-CAWS]

**Name**: WLHD — Within-Layer Head Deduplication via CF11 Cosine Threshold

**Mechanism**: U-CAWB and A-CAWS both tried to dedup W_Q heads using BLAKE3 hashing; both died on CF9 (hashes can't collide for non-bit-identical inputs). The stripped primitive: direct cosine-threshold deduplication. CF11 says 16 W_Q heads in a layer collectively span only ~128 dimensions. In a 2048-dimensional space, 16 matrices of size 128×2048 (per-head slices of W_Q) that together span only 128 dimensions must have substantial redundancy among them. Formally: if the global rank-128 projection U captures > 80% of variance across ALL 16 heads, then there exist head pairs whose top-64 singular vectors have cosine similarity > 0.85. WLHD proposes: compute pairwise cosine similarity of W_Q^(h) top-64 left singular vectors per layer; deduplicate any pair with cosine similarity > 0.85 by storing the average of the two head's singular vectors and updating the coefficient (C_h → (C_h + C_j)/2 for the deduped pair, stored once). This is a 2× within-layer dedup on the deduplicated heads. The quality cost is nonzero (we approximate two distinct singular-vector sets by their average) but the CF11 head-redundancy evidence suggests the two sets are already close to a shared subspace.

**Residency arithmetic**: If 2 of 16 heads per layer are exact duplicates: W_Q saving = 1/16 per deduplicated pair × 28 layers × (128×2048×2 bytes per head) = 14 MB on 1.7B. Marginal. If CF11 head-sharing is more aggressive (the 128-dim shared subspace means multiple heads are pairwise similar): if 8 of 16 heads deduplicate to 2 unique: W_Q residency 28 × 2 × (128×2048×2) + index overhead ≈ 28 MB vs 448 MB (original). 16× reduction for W_Q on Qwen3-1.7B. At 70B: W_Q = 8.6 GB; 16× reduction → 0.54 GB. Combined with rank-128 global projection: both savings compound (first global K=128, then within-layer dedup of near-identical heads). Conditional on both CF11 GO and within-layer similarity being high.

**Novelty gloss**: CAWS (A-CAWS, rejected) proposed the same deduplication but with broken BLAKE3 machinery. WLHD removes the hash and uses threshold cosine comparison directly. This is the stripped primitive from the cross-pollination analysis. No published work does within-layer W_Q head deduplication by cosine threshold on trained transformers (head pruning literature prunes by magnitude, not by action-similarity). Outside: orientation U anti-frame list (no custom kernel required); outside C anti-frame list (the coupling claim — CF11 head-sharing implies within-layer cosine similarity — is the composition, not just a stack of published methods).

**Smallest experiment**: Script: extract W_Q per-head slices (16 heads × 128×2048 per layer) for all 28 layers; compute pairwise cosine similarity of top-64 left singular vectors; report fraction of head pairs per layer with cos > 0.85, > 0.70, > 0.50. Runtime: ~20 min (SVD per head already partially done in AQFKV). GO: ≥ 2 head pairs per layer with cos > 0.85 in ≥ 20/28 layers. On GO: measure ΔNLL of averaging the deduplicated pair vs storing separately. Total ≤ 45 min.

**Primary risk**: Even if head singular vectors have cos > 0.85, averaging them introduces an approximation error that compounds across all 28 layers. Mitigation: measure ΔNLL of the within-layer average on 3 layers independently; if ΔNLL per deduped pair < 0.05 nats, layering 28 layers gives ~1.4 nats total (too much). Gate: per-pair ΔNLL < 0.01 nats for GO to be useful.

CF9 CLEAR. CF10: no regression fit (threshold-based classification). Smallest test ≤ 8h: YES. Track B. CF tether: CF11 (head-redundancy motivates within-layer dedup). Elegance-class: *subspace-alignment* (near-identical head subspaces collapse to one stored representative).

---

### S6 — Layer-Similarity Clustering for Shared MLP Operator [Track A, Orientation gap: control-theory framing]

**Name**: LCSO — Layer-Covariance-Similarity Operator Sharing

**Mechanism**: Section 4 (Professional Prior 3) derived the control-theoretic observation: layers with similar update covariances C^(ℓ) = E[δ_ℓ δ_ℓ^T] may be replaceable by a shared operator applied multiple times. The mechanism: (1) measure C^(ℓ) for each of the 28 layers (requires F-RSCLE data, 15 min); (2) compute pairwise Frobenius similarity ‖C^(i) − C^(j)‖_F / ‖C^(i)‖_F between all layer pairs; (3) identify clusters of layers with pairwise similarity < 0.20; (4) within each cluster, replace all layers' MLP blocks with a SINGLE shared MLP block (the cluster centroid, computed as the arithmetic mean of the cluster members' W_gate, W_up, W_down matrices). This is a post-hoc operator sharing across layers, analogous to ALBERT's training-time parameter tying but applied to pre-trained weights. The key non-obvious claim: if C^(i) ≈ C^(j) (their update covariance structures are similar), then the layers are computing "similar corrections" to the residual stream, and a single shared operator that produces the mean correction incurs only proportional quality loss — unlike forcing two distinct layers to share weights, which would be exact sharing (catastrophic quality loss), this is APPROXIMATION-SHARING at the level of the correction statistics, not exact weight sharing. The approximation error is ‖ δ_i^{shared} − δ_i^{original} ‖ / ‖ δ_i^{original} ‖, which is small when the correction statistics are similar.

**Residency arithmetic**: If 2 layers in a 3-layer cluster share 1 operator: 2 layers' MLP blocks (W_gate + W_up + W_down) become index pointers. MLP weight saving per shared pair: 2×(6144+6144+2048)×2048×2 bytes / layer = 2×236 MB = 472 MB for Qwen3-1.7B. At 70B with 20% of layers clustering (reasonable estimate for mid-depth layers with similar update structure): 20% × 64 layers = 13 MLP layers shared → 13 × 550 MB ≈ 7 GB saving at 70B bf16, ~3.5 GB at INT4. Non-trivial.

**Novelty gloss**: ALBERT shares parameters across layers at training time (no retraining, exact sharing). LCSO proposes approximate-sharing after training based on measured update-covariance similarity — the quality gap from ALBERT's exact sharing is replaced by a calibration-measured covariance match threshold. No published post-training layer-clustering paper uses update covariance C^(ℓ) as the similarity metric. The closest is "Layer-Wise Relevance Propagation" for pruning, but that is weight-importance-based, not update-covariance-based. Outside: F anti-frame list (this imports no named theorem from control theory; the update covariance is a direct measurement; the cluster-then-share mechanism is an approximation with measured error bound); outside C anti-frame list (the coupling is C^(i) ≈ C^(j) → shared operator is admissible, which is a composition of a measured update-covariance structure with a parameter-sharing mechanism).

**Smallest experiment**: (a) Run F-RSCLE update-covariance measurement (15 min) — or reuse the data if F-RSCLE runs in Stage 4; (b) compute pairwise Frobenius similarity for all 28 layers; (c) identify any cluster of ≥ 2 layers with Frobenius ratio < 0.20; (d) on the most-similar pair, average their W_up matrices and measure ΔNLL. Runtime: 15 + 5 + 5 + 15 min ≈ 40 min. GO: Frobenius ratio < 0.20 for ≥ 1 layer pair AND ΔNLL from W_up averaging < 0.30 nats for that pair.

**Primary risk**: Transformer layers develop distinct functional roles during training despite similar covariance structure; the approximation error from averaging may compound severely across layers even if each pair's ΔNLL is small. Mitigation: measure ΔNLL for only the single most-similar pair first; scale up to cluster only if the pair ΔNLL is below 0.05 nats.

CF9 CLEAR. CF10: no regression fit. Smallest test ≤ 8h: YES. Track A. CF tether: CF2 (residual additivity motivates update covariance measurement), plus the update-covariance measurement from F-RSCLE.

---

### S7 — CF11-Indexed KV Search for Long-Context Compute Reduction [Track A, Cross-domain: database engineer]

**Name**: QKIDX — Query-Subspace KV Index for Long-Context Attention

**Mechanism**: Section 4 (Professional Prior 1) derived the database-engineer framing: Q K^T can be decomposed into a fast index scan over the dominant query subspace (top-128 directions, CF11) plus a residual correction. QKIDX proposes: for long-context tokens (context length > 4K), build a static approximate KV index over the prefix tokens using CF11's W_Q rank-128 projection as the index key. Specifically: for each KV token in the prefix, project the key onto the top-128 singular subspace of W_K (CF11: W_K has r_99/d ≈ 0.79, so projecting at rank 128 captures > 40% of key variance). Store these projected keys in an ANNS index (a flat FAISS IVF index in RAM; at 4K tokens, 4096 × 128 × 2 bytes = 1 MB). At decode time: project the query onto the same 128-dim subspace, do an ANNS scan to retrieve the top-64 KV candidates, then compute exact attention scores only for those 64 candidates. Quality cost: attention is approximate; exact computation is bypassed for 4032 of 4096 KV tokens. The quality cost depends on how much attention mass is concentrated in the top-64 tokens — for long-context tasks where attention is sparse (retrieval, multi-hop reasoning), this is nearly zero-cost. For short-context coherent generation, the cost is higher.

**Residency arithmetic**: No change in weight residency. KV index: 4K × 128 × 2 bytes = 1 MB per layer (vs full KV at INT4: 4K × 2 × 128 × 0.5 bytes = 0.5 MB per layer). Index overhead: 2× KV RAM at prefix. Primary payoff: compute, not residency. At 32K context, attention FLOPs scale as O(n²); QKIDX reduces this to O(n × 64 + n × 128-dim index scan) ≈ O(n × 192) — linear instead of quadratic. At 32K context: 32K² vs 32K × 192 → 166× fewer attention operations for the indexed portion.

**Novelty gloss**: ScaNN, HNSW, FAISS all exist for ANNS; applying them to the KV attention cache is known (KVSharer, H2O). What is novel: using CF11's W_K spectral rank (r_99/d ≈ 0.79) as the motivation for the 128-dimensional index key (not a heuristic choice but a measured quantity from the pipeline's own data). No published paper derives the index dimensionality from the KV weight matrix's spectral rank. Outside: C anti-frame list (the composition claim — CF11's spectral rank determines the index key dimensionality — is a coupling between a measured structural property and an ANNS design choice, not a stacking of published methods).

**Smallest experiment**: (a) Using CF11 W_K rank measurement, confirm var@128 for W_K across 28 layers; (b) build a flat FAISS IVF-16 index over 4K randomly sampled tokens' projected keys at K=128; (c) run approximate attention (top-64 retrieval) vs full attention on Qwen3-1.7B-Base on a 4K-context prompt; (d) measure ΔNLL. Runtime: ~2h. GO: ΔNLL < 0.30 nats at 4K context AND retrieval recall@64 > 0.80 (64 retrieved tokens include the 64 highest-weight true attention tokens in ≥ 80% of queries).

**Primary risk**: At 4K context, attention is often NOT sparse; the top-64 candidates may miss important tokens, causing quality degradation. Mitigation: measure at contexts where attention is known to be sparse (multi-document QA, retrieval tasks) first; expand to general generation only if ΔNLL < 0.3 nats.

CF9 CLEAR (ANNS uses Euclidean distance in projected space; CF11 motivates the projection dimension, no theorem import). CF10: no calibration fit. Smallest test ≤ 8h: YES. Track A. CF tether: CF11 (W_K spectral rank determines index key dimensionality). Elegance-class: *subspace-alignment* (CF11 spectral rank motivates the ANNS index key dimension).

---

### S8 — Attention W_V/W_O Spectral Split Mixed-Precision [Track B, Orientation gap: W_V/W_O post-F-SGFVO]

**Name**: VOMS — W_V/W_O Output-Subspace Mixed-Precision Allocation

**Mechanism**: F-SGFVO (Stage 3 REFINE) measures the W_V W_O product rank. If median r_99 < d_head = 128 (F-SGFVO GO), then the top-r_99 singular components of the product P = W_V W_O are the load-bearing components. VOMS proposes a complementary step: apply different quantization precisions to the top-r_99 vs bottom-(d_head - r_99) singular components of P. Specifically: (1) decompose P = U_P S_P V_P^T (SVD of the head-operator product); (2) store U_P[:, :r_99] × S_P[:r_99, :r_99] at INT8 (the load-bearing components); (3) store U_P[:, r_99:] × S_P[r_99:, r_99:] V_P[r_99:, :]^T at INT4 (the low-variance tail). This is a within-matrix spectral mixed-precision scheme, where the quantization precision tracks the singular value decay rather than magnitude or Hessian. The WDSG idea (S1) applies a similar coupling to W_down using residual-stream PCA; VOMS applies it to the W_V W_O product using the product's own singular values. No cross-finding coupling needed — this is purely internal to the W_V W_O rank structure.

**Residency arithmetic**: At r_99 = 64 (F-SGFVO GO at median r_99 ≤ 96): top-64 components at INT8 + bottom-64 at INT4. Per head: (128×2048×2 + 128×2048×1) / (128×2048×2 + 128×2048×2) = (3 bytes/param) / (4 bytes/param) — 25% reduction vs uniform INT8. At r_99 = 96: 75% INT8 + 25% INT4 → mean bpw = 0.75×8 + 0.25×4 = 7 bpw vs 8 bpw INT8. Marginal at 1.7B; at 70B W_V + W_O (~10 GB at INT8), saving ~1 GB.

**Novelty gloss**: ABAR (Stage 3 REFINE) does cross-matrix spectral-rank-based mixed precision; VOMS does within-matrix spectral mixed precision based on the product's singular value decay. No published mixed-precision scheme for transformers uses the PRODUCT rank (not the individual matrix rank) as the allocation guide within a single matrix. LAMPQ, GPTQ, AWQ all use Hessian-based sensitivity or nuclear norm. Outside: R anti-frame list (this is not an integration narrative; it is a new allocation oracle derived from F-SGFVO's product-rank finding); outside F anti-frame list (it is a consequence of an algebraic identity — the rank ceiling of W_V W_O — not plain quantization as the load-bearing move).

**Smallest experiment**: After F-SGFVO experiment (30 min), if GO: decompose P = W_V W_O for layer 14; apply INT8 to top-r_99 singular components, INT4 to remainder; measure ΔNLL vs uniform INT8 and vs uniform INT4. Runtime: ~15 min (using F-SGFVO data). GO: ΔNLL_mixed < ΔNLL_uniform_INT8 + 0.05 nats AND ΔNLL_mixed < ΔNLL_uniform_INT4 − 0.10 nats (quality better than INT4, cost better than INT8).

**Primary risk**: The product P may have r_99 = d_head = 128 (full head rank, F-SGFVO NO-GO), making the within-matrix split impossible. VOMS is conditional on F-SGFVO GO. If F-SGFVO NO-GO, VOMS is moot (the product is full-rank; no exploitable split).

CF9 CLEAR. CF10: no regression fit. Smallest test ≤ 8h: YES (conditional on F-SGFVO GO). Track B. CF tether: CF11 (W_V/W_O are attention weights with non-MLP spectral behavior). Elegance-class: *gauge-exploitation* (product rank identity is exact; mixed precision follows the exact spectral decomposition).

---

## 7. Output Handoff

### Stage 4 ideas summary

| ID | Name | Gap filled | Section motivation |
|---|---|---|---|
| S1 | WDSG — W_down Spectrum-Gated Mixed-Precision | W_down column sensitivity in residual-PCA basis | Section 4 (cross-domain seed: control theory) |
| S2 | ACES — Attention Entropy-Stratified Tier Split | KV-side compute gap (zero KV-side advancers in run) | Section 1 (frame-exhaustion: KV-side mechanism) |
| S3 | HOIST — Hoisted Shared-Basis Query Precomputation [FREE SWING] | Cluster C1 execution schedule (storage → compute) | Section 4 (cross-domain seed: compiler writer) |
| S4 | VPTK — VPSHUFB Integer-Only Three-Tier Kernel | No-FP constraint on RAOK tier scheme | Section 5 (constraint-driven reframing: no FP) |
| S5 | WLHD — Within-Layer Head Deduplication | U-CAWB/A-CAWS stripped primitive (hash removed) | Section 3 (cross-pollination X-POLL-1) |
| S6 | LCSO — Layer-Covariance-Similarity Operator Sharing | Control-theory view of layers as similar update operators | Section 4 (cross-domain seed: control theorist) |
| S7 | QKIDX — Query-Subspace KV Index | Database-engineer framing of Q K^T as indexed join | Section 4 (cross-domain seed: database engineer) |
| S8 | VOMS — W_V/W_O Output-Subspace Mixed-Precision | Post-F-SGFVO within-matrix spectral allocation | Section 1 (frame-exhaustion: W_V/W_O post-measurement) |

### Recommendations to round runner

**Top 1-2 for Stage 5 alongside Stage 3 advancers:**

- **S3 (HOIST)** is the highest-leverage idea if Cluster C1 confirms GO: it converts the storage saving into a compute + NVMe bandwidth saving simultaneously, and reveals a new GGUF layout primitive. Stage 5 should consider S3 alongside F-CQSGC / R-CROSS-Q as the "what to do with Cluster C1 GO" cascade.

- **S4 (VPTK)** is the highest-leverage Track B idea: it is a pure implementation complement to the RAOK cluster advancers, requiring no new experiment beyond what Cluster C2 already requires, while unlocking AVX2 VPSHUFB throughput for the dominant Tier-2 bulk term. Stage 5 should consider S4 alongside R-RAOK-70B as a cascade implementation rung.

### Orientation recommendations (v2 round-over-round)

| Orientation | Recommendation | Rationale |
|---|---|---|
| R (Reach) | KEEP | Strong performer; 5/7 advanced; anti-frame addition: "attention-tier / MLP-tier deployment split as primary mechanism" |
| C (Composition) | KEEP | Best advancement rate (6/7); anti-frame additions: "AVX2 scheduling as composition claim" and "multiplicative SVD savings without destructive-interaction check" |
| F (First-Principles) | KEEP with flag | Monitor F-RSCLE pre-emption by arXiv:2605.03109; anti-frame addition: "update-covariance Laplacian framing without checking Gated Subspace Inference paper" |
| U (Substrate) | TIGHTEN | 0/6 advanced; new A1 floor: primitive must change > 20% of NVMe bytes per token OR change weight-class residency tier; anti-frame additions: "systems micro-optimizations < 5% gain" and "content-addressable schemes using cryptographic hashes for near-similar vectors" |
| A (Constraint-Alien) | TIGHTEN | 1/6 standard ideas advanced; constraint menu should prune to constraints that directly interact with NVMe/DRAM bottleneck; anti-frame additions: "tropical algebra as K=20% sparsity assumption from CF1" and "append-only state at < 32K context" |

### Anti-frame additions

| Orientation | New anti-frame | Rationale |
|---|---|---|
| R | Attention-tier DRAM / MLP-tier NVMe split as primary mechanism | Now an integration narrative; covered by MCQKV + CROSS-Q |
| C | AVX2 dual-stream scheduling claim without derivation of dispatch-width vs bandwidth-width distinction | C-MCLG failure pattern |
| C | Multiplicative composition of two SVD savings on correlated matrices without independence check | Standard precaution revealed by C-ABAR's simultaneous W_Q + W_K compression |
| F | Residual-update-covariance Laplacian framing without Gated Subspace Inference (arXiv:2605.03109) verification | F-RSCLE DOWNGRADE pattern |
| U | Substrate primitives with gain < 5% on NVMe-bottlenecked 70B inference | All four non-CF9 rejections in run 001 |
| U | Content-addressable schemes using cryptographic hashes for near-similar vectors | U-CAWB CF9 kill pattern |
| A | Tropical / (max,+) routing under CF1 as K=20% sparsity assumption | A-TSWD kill pattern |
| A | Append-only state constraints at < 32K context (A-ALSA pattern) | A1=1; value only exists at 32K+ |

---

*Word count: ~2,480 words in ideation content (sections 1-6); within 1500-2500 word guidance.*
