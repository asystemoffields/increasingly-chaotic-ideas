# Stage 4 — Skeptic Explorer — Run 004
Date: 2026-05-09

Inputs: stage1_{R,C,F,U,A}.md (32 proposals), stage2_judge.md (24 advancers), stage3_deep_research.md (25 REFINE, 1 DOWNGRADE), SUMMARY.md (CF1-CF12 + v2-CF1 + v2-CF2), KILL_LIST.md (v2-S3-R004-001 / F-GFQO + v2-CHEAP-TEST-001).

---

## 1. Frame-Exhaustion Map

### Saturation-exhausted frames this round

**Frame SE-1: Attention weight rank reduction via standard SVD (W_Q / W_K alone)**
Stage 3 confirmed CF11 (W_Q global K=128 GO, per-head NO-GO) is now fully documented and the adjacent space is crowded — SVD-LLM, ASVD, LatentLLM, MHA2MLA, TransMLA, A3 all occupy the per-matrix SVD-truncation territory. Every advancer referencing attention SVD (AWCJR, VOKV, C-VKSD, F-WVOS) is PARTIAL in prior-art status. The frame is saturated for the single-matrix cuts; the open space is the product structure (W_V·W_O) and the GQA-specific shape. Stripped primitive: **W_V×W_O product spectrum** — this is a multiplication of two attention projections, not a sum, and can collapse to lower rank than either alone. This primitive passes SE-1's saturation boundary because no published paper measures the product rank.

**Frame SE-2: Per-layer / per-head outlier handling with static-channel pinning**
LLM.int8(), SmoothQuant, PrefixQuant, Quaff, NVFP4 thoroughly cover the static channel + token-dynamic decomposition space. CF3 / v2-CF1 add the all-28-layer K-dependent Jaccard picture, but any new proposal claiming "handle activation outliers with a 2-tier scheme" is adjacent to PrefixQuant at minimum. Stripped primitive: **outlier-channel ⊥ attention-subspace alignment** (C-AQOD claim) — this is not channel-handling per se but a structural claim about the two objects' disjointness, which is still NOVEL. The saturation applies to the handling scheme itself, not to the structural measurement of how the two objects relate.

**Frame SE-3: Cross-layer weight sharing (basis or subspace stacking)**
v2-CHEAP-TEST-001 killed the entire "cross-layer W_Q stacked SVD / shared basis" class. The experiment confirmed W_Q subspaces rotate independently across layers. This kill extends by implication to cross-layer W_K, W_V, W_O sharing (no evidence they behave differently; if W_Q doesn't share, the adjacent matrices are unlikely to either, though untested). Stripped primitive: **within-layer cross-matrix joint basis** — C-VKSD (within-layer W_K/W_V joint subspace) and F-SPECA (within-layer W_Q/W_K joint basis) both operate within a single layer, which the cheap test does NOT kill. Within-layer sharing remains open.

### CF9-exhausted frames

**Frame CF9-E1: General orthogonal rotation of residual stream for compression**
F-GFQO was downgraded precisely for this reason. The gauge-freedom argument produces a clean compression mechanism in principle, but RMSNorm's per-channel scale breaks O(d) symmetry to only {sign-flip × permutation} generators. Any proposal that imports "rotate the residual stream to expose low-rank structure" inherits this kill unless it explicitly restricts to RMSNorm-compatible transformations. Stripped primitive: **per-channel importance ordering for block-sparse W_Q assignment** — permutation is allowed by RMSNorm, so ordering channels by attention-subspace projection magnitude is live. This is the stripped-F-GFQO primitive that Stage 4 should turn into a concrete idea.

**Frame CF9-E2: Smooth / sparsity-assuming approximations on dense LLM residuals**
Count-Sketch (dense residuals), RFF (non-kernel objects), Winograd (no spatial reuse in GEMV), DCT/wavelets (no locality in weight matrices), belief propagation (no convergence on dense graphs) — these all failed the CF9 check in prior rounds. The pattern is firm: any algorithm whose error bound has a sparsity, smoothness, shift-invariance, or locality precondition fails on LLM weight matrices or dense activation residuals. The open space is **algorithms with preconditions that DO hold**: rank (concentrations measured in CF11), stability (top-0.1% outlier Jaccard = 0.718 from CF3), monotonicity (RoPE position-dependent score decay), or combinatorial structure (tropical max-dominance argmax, content-addressable hashing on binary signatures).

### CF10-exhausted frames

**Frame CF10-E1: High-parameter calibration fits without low-rank-by-construction forcing**
WDLA died for this reason. Any proposal that fits a d×d affine correction on fewer than d² independent calibration samples is in this kill class. The surviving variants are: force rank r << d (parameter count drops to d×r), use strong ridge (1e-1 or higher), or use very large calibration corpora (10K-100K tokens). A-SEED was flagged for a CF10 concern (MLP width 512 >> n_matrices) and given a specific mitigation. The frame is not dead but any Stage 4 calibration-fit proposal must pass the n_params / n_samples check internally.

---

## 2. Orientation-Saturation Diagnostics

**Orientation R (Reach):** Advanced 7/7 ideas to Stage 3 (AWCJR, L1AW, RATJ, ULHC, WDRS, HRQK, VOKV). All REFINE. Strong convergence participation (C1, C2, C3, C5). Heavy convergence on W_V/W_O measurement and AQFKV follow-up; the orientation did its job of aggressive cascade-building but most proposals depend on a shared cheap-measurement gate (W_V/W_O SVD). Anti-frame recommendation: add "cascades whose entire first rung is a spectrum measurement shared with another proposal — propose a structural claim that is novel EVEN IF the spectrum is flat." Recommendation: **KEEP / TIGHTEN** (add anti-frame above).

**Orientation C (Composition):** 5/6 advanced (C-AQOD, C-VKSD, C-RBHD, C-WVKQ, C-ODWQ). One rejected at Stage 2 (C-SDLC, CF2-pre-empted). Very strong coupling equation generation — C-ODWQ's Spearman ρ check is the cheapest decisive experiment in the run. Converged on C1 (W_V/W_O) and C2/C5 (outlier × attention interaction). Anti-frame recommendation: add "coupling claims between two attention-side findings (both CFs about attention weights) — the next richer territory is attention × MLP coupling, or training-dynamics × inference-structure coupling." Recommendation: **KEEP / TIGHTEN** (add anti-frame above, redirect toward cross-subsystem coupling).

**Orientation F (First-Principles):** 4/6 advanced (F-WVOS, F-GFQO downgraded, F-AERO-Q3, F-SPECA, F-WNORM). F-RMNQ and F-TIED rejected. F-GFQO downgraded (CF9 partial failure). F-AERO-Q3 is the highest-value proposal in the run (10-min decisive binary, algebraic-identity). Anti-frame recommendation: add "gauge-freedom arguments using unrestricted orthogonal rotation — must restrict to RMSNorm-compatible class (permutations + sign flips) or explicitly route RMSNorm around the rotation." Recommendation: **KEEP** (anti-frame addition keeps it from re-proposing F-GFQO variants).

**Orientation U (Unconventional Substrate):** 4/5 advanced (WSFI, NCZC, WWTE, GTBP, NNPA). DFSQ rejected. Very distinctive proposal set — all 4 active advancers are NOVEL with no published ML inference cousin. The orientation is functioning excellently; no saturation detected. Its proposals are the only ones in the run addressing the bottleneck measurement problem directly (NVMe stall, GC stall, working-set residency). Anti-frame recommendation: add "measuring which OS primitive engages (e.g., a pure timing experiment) without connecting to a residency or throughput win on the actual inference workload" — DFSQ's 48 KB saving exemplified this dead-end shape. Recommendation: **KEEP**.

**Orientation A (Constraint-Alien):** 4/7 advanced (A-TROP, A-SEED, A-ALOG, A-REGM, A-GFWV). A-FSMT rejected (CF9 analog on FSM transition quality). A-GFWV advanced as a convergence rep for C1. A-TROP produced the run's most novel Track B framing (tropical semiring, no prior). The hypernetwork A-SEED is the most speculative idea in the run. Anti-frame recommendation: add "constraint that is de facto the standard playbook in disguise — 'no continuous numbers' cannot mean 'I'll quantize'" (A-FSMT's failure mode). Recommendation: **KEEP**.

**Overall:** No orientation should be dropped. U and A are producing the most pre-empt-resistant ideas; F is producing the highest-value algebraic proposals. C is the most rigorous coupling generator. R is generating the most actionable multi-stage plans.

---

## 3. Cross-Pollination Opportunities

### CP-1: F-GFQO stripped primitive → Track B channel-importance W_Q block-precision assignment

F-GFQO was downgraded because general orthogonal R breaks RMSNorm. What survives: **per-channel W_Q projection magnitude as a mixed-precision assignment signal** (named in the KILL_LIST entry). This primitive is in-scope for a Track B compression idea. The claim: channels with low projection magnitude onto all 28 layers' W_Q/W_K/W_V row-spaces are "attention-invisible" — they carry no information into any attention computation. Storing those channels at INT4 while keeping attention-visible channels at INT8 is a CF9-clean operation (no rotation needed, pure permutation/selection). This is distinct from SmoothQuant (SQ smooths magnitudes across W and activation; this selects WHICH channels matter to attention, then applies mixed precision). The stripped primitive rescues F-GFQO's measurement (joint rowspan of [W_Q;W_K;W_V]) as a precision-assignment signal rather than a compression mechanism per se.

### CP-2: A-SEED CF10-mitigated → Track A training-dynamics probe (fitting residuals only, not full weights)

A-SEED needs a width-64 F to pass CF10. Once F is constrained to width-64 and the target is changed from full W_Q synthesis to **W_Q residual-from-mean** synthesis (subtract the mean W_Q across layers; synthesize the residual), the parameter count drops to 28 × (128×2048) in targets but the residual space is smaller (mean is subtracted, so variance is lower). This cross-pollinates with F-First-Principles' interest in per-layer variation structure: "what fraction of W_Q's weight-matrix variation across layers is predictable from a small latent?" is a well-posed question.

### CP-3: A-TROP stable argmax → Track B W_down quantization codebook assignment

A-TROP measures which W_down output rows have a stable dominant input channel. This stability structure is a natural grouping criterion for W_down output-row codebook assignment in RAOK's framework. CF3 measures per-token activation outlier dynamics; A-TROP measures per-row W_down INPUT-channel dominance. These are orthogonal objects. If ≥10% of W_down rows have stable dominant input (A-TROP GO), those rows can be stored with a specialized codebook keyed on the dominant channel — a Track B compression idea that composes RAOK (activation-aware quantization) with A-TROP (weight-row dominance structure).

---

## 4. Cross-Domain Seed Exploration

### Database engineer's view: Range-coded block index for KV-cache hot zones

A database engineer looking at the KV cache sees a time-series of (key, value) pairs partitioned by token position and layer. The standard database approach to time-series compression is **zone maps**: store per-block (min, max, mean) statistics and use them to skip cold blocks during queries. For KV cache, "cold blocks" are the distant tokens whose attention scores are below a threshold. The algebraic identity: if W_K[L] has rank 512 (CF11: r_99/d ≈ 0.79), then K[L,t] = W_K[L] h[t] lies in a 512-dim subspace; zone maps on the K-subspace projection are cheap to compute and allow block-skipping without materializing K[t] for distant tokens. This is **different from H2O/SnapKV** (which evict based on attention scores — backward-looking) — the zone-map approach is **forward-looking**: given a new query q, predict which past K-blocks can be skipped without computing QK^T for them, using q's projection onto W_K's subspace. The precondition: CF11 W_K concentration (r_99/d ≈ 0.79) holds, so the projection lives in a 512-dim subspace with 79% of variance captured. Zone maps in that subspace are computable in 512 mults per block rather than 2048. The algebraic equivalence: pre-compute the zone map (max projection magnitude per block) once during KV-cache fill; at each new token, dot the query against the zone-map vector to predict which blocks will have negligible attention contribution.

**What would a database engineer propose for the weight matrices?** A columnar layout where W_Q's d_model-dimensional rows are partitioned into "hot columns" (attention-visible per F-GFQO's stripped primitive) and "cold columns" (attention-invisible). This is exactly CP-1 above. The database engineer would call it a **column store with zone map** on the importance ordering — a materialized per-layer importance vector (512 entries) derived from the joint W_Q/W_K/W_V rowspan measurement.

### Compiler writer's view: Profile-guided dead-code elimination on W_down rows

A compiler writer looks at W_down (6144→2048) and sees a function with 6144 input "registers" (post-SwiGLU channels) and 2048 output "registers." Profile-guided optimization would ask: which input-output paths are "live" in the call graph? A-TROP's measurement is exactly this: it measures which (input, output) pairs are "hot" (stable argmax) vs cold. The compiler analogy: **dead-code elimination on W_down**: if the argmax(W_down[i, :] × h_mlp[:]) is stable at input j for row i, then the other inputs into row i are "dead code" — their contributions are dominated and could be pruned. This is NOT low-rank (W_down is full-rank per CF8 expectation), but it IS a sparse dispatch: for stable-argmax rows, only the dominant channel needs to be materialized. The compiler's insight: profile-guided dead-code elimination can operate on a per-row basis even in a full-rank matrix, because dominance is a statistical property of the computation trace, not a linear-algebra property of the weights.

**What algebraic identity does the compiler writer see?** For a W_down row i with stable dominant input j: W_down[i, :] × h = W_down[i,j] × h_j + Σ_{k≠j} W_down[i,k] × h_k ≈ W_down[i,j] × h_j + error. If the error is bounded by |W_down[i,j]| × δ for measured δ < 0.1, the row computes as a scalar multiply plus a bounded correction. The correction can be applied with a sparse INT4 residual. This is a **compiler-style peephole optimization** on the matrix-vector product: for stable-dominant rows, the hot path is a scalar, the cold path is a sparse correction.

---

## 5. Constraint-Driven Reframing

### Constraint: All state is append-only (monotonic KV growth, idempotent retrieval)

Under this constraint, the KV cache cannot be evicted or rewritten. Every attention computation reads the entire prefix. The mechanism this forces: **monotonic compression of the KV cache as a streaming data structure**. A-ALOG addresses this, but its load-bearing M1 (geometric decay) may fail. The constraint itself, divorced from M1, forces a different mechanism: if the cache is append-only, the compression must be **lossless for recent tokens** and **progressively lossy for distant tokens** — exactly a streaming wavelet or run-length structure. But CF9 kills wavelets on LLM state (no locality). What's left: **prefix summarization via a running mean/variance accumulator** — the append-only constraint allows maintaining a running statistics vector over older KV pairs (their mean + top-2 principal components), discarding the individual pairs after they age past a threshold. This is a **compression-without-eviction** mechanism: the old KV pairs are summarized in-place rather than evicted. The mechanism does NOT require RoPE geometric decay (kills A-ALOG's precondition) and is append-only (never mutates existing entries). The compressed form is a fixed-size statistics buffer that grows logarithmically with context length.

### Constraint: No floating point at inference (integer + lookup only)

Under this constraint, all GEMV operations are integer, and nonlinearities are lookup tables. The mechanism this forces: for SwiGLU, silu(z) = z × σ(z). Under the no-float constraint, silu becomes a lookup table indexed by the INT8/INT4 value of z. The lookup table for INT8 input has 256 entries — 512 bytes. This is essentially free. The interesting forcing comes on W_Q/W_K inner products: under INT8 arithmetic, the QK^T product is exact (INT8 × INT8 = INT16 accumulate) and the softmax becomes an approximate integer softmax. The connection to CF11: W_Q at K=128 gives ΔNLL=+0.98 nats with bf16 arithmetic; under INT8 quantization OF THE COMPRESSED W_Q, the additional quantization error must be budgeted. The Stage 4 idea: **integer-arithmetic feasibility measurement for W_Q-compressed attention** — compute the ΔNLL of W_Q@K=128 in INT8 arithmetic vs bf16 arithmetic, to verify whether the CF11 GO bound is maintained under integer-only compute. This fills the **runtime-fusion gap** that is underrepresented in run_004's advancers.

---

## 6. Generated Ideas (8 ideas)

---

### S1 — ATVIS: Attention-Invisible Channel Mixed-Precision W_Q Assignment

**Ambition (Track B).** Assign lower-precision storage to W_Q channels that have negligible projection magnitude onto the joint [W_Q;W_K;W_V] row-space across all layers — the "attention-invisible" channels. If 30-40% of the d_model=2048 channels contribute < 1% each to any attention GEMV, those channels can be stored at INT4 while attention-visible channels remain at INT8. Residency gain: up to 0.5 bpw reduction on W_Q storage if ~35% of channels qualify.

**Mechanism.** For each channel c (c = 0..2047), compute: importance[c] = max over all 28 layers, over W_Q/W_K/W_V, of |projection of e_c onto the top-K right singular vectors of [W_Q^L; W_K^L; W_V^L] stacked|. Channels with importance[c] below a threshold τ are attention-invisible. These can be stored at INT4 without harming attention quality, because their contribution to Q·K^T / V-weighted-sum is structurally negligible. The per-channel importance vector is computed once (offline) and stored as metadata (2048 floats = 8 KB). No RMSNorm compatibility issue: this is pure channel selection (permutation + tier assignment), not rotation. CF9 clean.

**Why our findings make this work.** F-GFQO's stripped primitive (per-channel W_Q projection magnitude) survived the Stage 3 CF9 correction. v2-CF2 confirmed W_Q subspaces rotate independently per layer — meaning the attention-invisible set must be identified jointly across all layers (a channel invisible to W_Q^{L1} may not be invisible to W_Q^{L14}). The cross-layer minimum is the conservative gate. CF11 provides the subspace size (K=128 captures 94% of W_Q variance) — channels outside the top-128 right singular space of every layer's W_Q are candidates.

**What would have to be true.** At least 25% of d_model channels have max-layer importance < 0.01 (essentially zero projection onto all attention weight rows across all layers). If the importance distribution is bimodal (few high-importance channels + many near-zero channels), the INT4 tier is large and the saving is meaningful. If importance is flat (all channels contribute equally), the saving is zero and the measurement still closes the question.

**Experiment cascade.** E1 (10 min): compute the 2048-dim importance vector using PDAP/AQFKV data already in hand (W_Q SVD right singular vectors per layer from AQFKV, residual-stream channel norms from PDAP). Histogram the importance distribution. GO: ≥25% of channels below τ=0.01. KILL: importance distribution flat (< 10% below τ). E2 (30 min, conditional on E1 GO): assign INT4 to low-importance channels in W_Q, measure ΔNLL vs INT8 W_Q. GO: ΔNLL < 0.1 nats above uniform-INT8 baseline.

**What you're NOT proposing.** Not a rotation-based method (killed by RMSNorm). Not cross-layer weight sharing (killed by v2-CHEAP-TEST-001). Not SmoothQuant (SQ scales activations; this selects channels for mixed-precision storage without touching activations). Tag: `subspace-alignment`. Sits outside C-orientation's anti-frame (no coupling equation between two CFs; this is a direct single-measurement structural claim).

---

### S2 — ZWMP: Zone-Map W_K Block-Skipping for Long-Context KV

**Ambition (Track A).** Maintain a per-block zone map on the KV cache's key vectors, projected into W_K's 512-dim subspace (CF11: W_K r_99/d ≈ 0.79). For each new query token, check query-projected inner product against zone-map maxima to identify blocks guaranteed to have negligible attention scores, skipping their QK^T computation. At 8K+ context, this enables sparse attention computation without training (no learned sparse patterns, no eviction). Target: 2× attention GEMV speedup at 8K context with ΔNLL < 0.2 nats.

**Mechanism.** During KV cache fill: for each block of B=64 past tokens, compute max_t(||proj_{W_K,512}(K[t])||) — the zone map vector (one scalar per block, computable in 512 mults). At query time: for new query q, compute q_proj = W_Q q (already computed); then for each block, bound the max attention score in that block by q_proj · zone_map_scalar / ||q_proj||. If the bound < threshold, skip the block. The algebraic identity: ||⟨q_proj, K_proj⟩||_max ≤ ||q_proj|| × zone_map_scalar (Cauchy-Schwarz in the projected subspace).

**Why our findings make this work.** CF11 shows W_K r_99/d ≈ 0.79, so 79% of K-vector variance lives in a 512-dim subspace. A zone-map computed in that subspace captures 79% of the attention score variance. The 21% residual limits false-negative skipping — if a block has attention signal only in the residual subspace, the zone map will miss it. The threshold must account for this residual. CF11's W_K K=512 ΔNLL=+0.29 nats suggests the top-512 subspace is nearly sufficient quality-wise.

**What would have to be true.** The residual 21% of K-vector variance must not carry most of the attention signal for typical prompts. This is verifiable: if the zone-map-based sparse attention gives ΔNLL > 0.5 nats at 8K context, the residual signal is too large. If ΔNLL < 0.2 nats, the subspace captures the functionally important part.

**Experiment cascade.** E1 (15 min): measure fraction of KV blocks that would be skipped at threshold=0.1, on 10 × 8K-context prompts. E2 (30 min, conditional on skip rate ≥ 30%): measure ΔNLL under block-skipping at threshold=0.05 / 0.1 / 0.2. GO: ΔNLL < 0.2 nats at skip rate ≥ 30%.

**What you're NOT proposing.** Not H2O/SnapKV (those evict by observed attention score; this skips by Cauchy-Schwarz bound on uncomputed score). Not cross-layer KV sharing (v2-CHEAP-TEST-001 kill). Not a training-time method. Tag: `subspace-alignment`. Sits outside U-orientation's anti-frame (not a substrate primitive, pure algebraic argument on CF11's subspace bound).

---

### S3 — RDCP: Run-Length + Delta Compression on the KV Prefix (Append-Only)

**Ambition (Track B).** Apply run-length + delta encoding to the KV cache's value vectors without eviction, maintaining full precision for recent tokens and progressively compressed older tokens. At 32K context, this reduces KV residency 2-4× with ΔNLL < 0.3 nats, by exploiting CF3's channel-static property at K=0.1% (top-2 channels have Jaccard 0.718 — they barely change across tokens) and the de facto monotonic growth of the prefix.

**Mechanism.** Segment the KV cache prefix into 3 tiers: (T1) recent 512 tokens: full bf16 storage; (T2) mid 512-4K tokens: delta-encode V[t] as V[t] - V_mean (mean over T1 window) in INT8; (T3) older tokens: store top-2 static channels in bf16 + the remaining 2046 channels as INT4 delta from a running mean. No eviction: every token is retained in some tier. The Jaccard 0.718 at K=0.1% means the top-2 channels of V[t] are nearly the same vector across tokens — run-length encoding compresses them to a single stored vector per run of ≤50 tokens where the top-2 channels don't change. T3 stores the top-2 at a single bf16 (amortized cost per token → ~0).

**Why our findings make this work.** CF3 / v2-CF1: K=0.1% channels are static (Jaccard 0.718), all-28-layers confirmed. These channels are the run-length-compressible part. K=1-10% channels are dynamic (Jaccard 0.31-0.24) — stored as INT8 deltas. The three-tier structure maps directly onto CF3's three regimes. The "append-only" constraint from Section 5 forces this mechanism: instead of evicting cold tokens, we keep them in progressively coarser representations.

**What would have to be true.** Top-2 static channels of V[t] must actually be the same 2 channels as top-2 static channels of K[t] (or at least overlapping) — if V's static channels are different from K's, the tier structure has to be learned per-tensor. v2-CF1 measured RESIDUAL STREAM outliers; V[t] is a different distribution (attention-output weighted combination). This is the key precondition to verify.

**Experiment cascade.** E1 (20 min): measure per-token Jaccard of V[t] channel-set at K=0.1% across 200 prompts on 4K context. GO: Jaccard ≥ 0.60 → run-length tier is viable. E2 (40 min, conditional): implement 3-tier KV storage, measure ΔNLL at 4K context. GO: ΔNLL < 0.3 nats AND KV size ≤ 60% of full bf16.

**What you're NOT proposing.** Not H2O/SnapKV/StreamingLLM (those evict — this is lossless-for-recent, lossy-for-old without eviction). Not cross-layer KV sharing. Tag: `subspace-alignment`. Fills the **KV-compute gap** (run_004 Stage 1 had no proposal targeting long-context KV compression grounded in CF3's actual numbers at all layers).

---

### S4 — TDIP: Training-Dynamics-Informed Precision (W_down Dominant-Row INT4 Path)

**Ambition (Track B).** Ground A-TROP's tropical-algebra measurement in a concrete codebook assignment: W_down rows with stable dominant input channel get a specialized 2-element INT4 codebook (dominant + correction); rows without stable dominance get a standard GPTVQ codebook. Goal: 2.2 bpw average on W_down with ΔNLL < 0.5 nats, exploiting the asymmetry that A-TROP's stable-dominant rows are much more compressible than standard K-means codebook because their output is dominated by one input channel.

**Mechanism.** A-TROP measures, for each W_down row i, the stability fraction σ_i = P(argmax_j W_down[i,j]×h_mlp[j] = j*). For rows with σ_i ≥ 0.85: the row's GEMV is dominated by one term → store W_down[i, j*] in fp16 and the residual W_down[i, k≠j*] in INT4. At inference, compute h_mlp[j*] × W_down[i,j*] first (fp16 multiply), add INT4 residual dot product. Effective bpw for stable rows: 16/(d_in) + 4×(d_in-1)/d_in ≈ 4.01 bpw (negligibly above INT4). For unstable rows: standard 3-bpw GPTVQ codebook. Weighted average: if 15% of rows are stable, bpw ≈ 0.15 × 4.01 + 0.85 × 3 ≈ 3.15 bpw. If 50% are stable, bpw ≈ 3.5 bpw. Compression win requires the stable path to be higher quality than 3-bpw GPTVQ — it likely is, because the dominant-channel signal is exact (fp16), not quantized.

**Why our findings make this work.** A-TROP (run_004 Stage 3 REFINE) measures stable dominant input channel fraction. CF3 confirms post-SwiGLU activations are token-dynamic at K=1-10% but individual row-dominance may still be stable (the dominant channel for a given output row may be stable even though the full activation distribution is dynamic). These are different statistics: A-TROP tracks which input channel dominates each output row; CF3 tracks which channels are large across the full activation vector.

**What would have to be true.** A-TROP GO (≥10% of W_down rows with σ ≥ 0.85). The dominant-channel fp16 coding must give lower reconstruction error than 3-bpw GPTVQ for those rows. The mixed-precision codebook must be implementable in AVX2 GEMV without excessive scatter overhead.

**Experiment cascade.** E1 (30 min): run A-TROP measurement (already in Stage 3 plan). E2 (conditional on A-TROP GO, 40 min): implement dominant-row codebook for W_down in a single layer, measure ΔNLL vs GPTVQ baseline. GO: ΔNLL < 0.3 nats above GPTVQ at same bpw AND ≥ 10% of rows qualify.

**What you're NOT proposing.** Not standard codebook methods (this uses activation-trajectory structure, not Hessian). Not low-rank decomposition of W_down (CF8 kills that). Fills the **training-dynamics gap** (no run_004 proposal uses the stability of argmax patterns as a codebook assignment criterion). Tag: `algebraic-identity` (dominant-row decomposition is an exact partition: W_down[i,:]·h = fp16_dominant + INT4_residual, no approximation in the partition itself).

---

### S5 — IQAT: Integer W_Q Attention Feasibility (runtime-fusion gap)

**Ambition (Track A/B hybrid).** Verify that W_Q at K=128 (CF11 GO at +0.98 nats bf16) maintains its ΔNLL advantage when executed under INT8 arithmetic (W_Q stored INT8, activations INT8). If the INT8 quantization error and the K=128 rank-truncation error are sub-additive, the combined bpw on W_Q drops to 1.0 bpw × (128/2048) rank + 1.0 bpw INT8 = ~1 bpw effective. This fills the runtime-fusion gap — no run_004 proposal addresses whether the W_Q compression can be fused with integer arithmetic for AVX2 throughput gain.

**Mechanism.** Three conditions to measure: (A) W_Q at K=128 bf16 = +0.98 nats (CF11, already measured). (B) W_Q at K=128 INT8 = ? (W_Q stored as INT8 rank-128 matrix). (C) Full-rank W_Q INT8 = ? (standard INT8 W_Q without rank truncation). If (B) - (A) < 0.2 nats, the INT8 quantization error is sub-additive with rank truncation, and the combined scheme (K=128 INT8 W_Q) gives ΔNLL ≈ +1.1 nats — acceptable and enables AVX2 INT8 GEMV for the compressed W_Q. The residency win: W_Q at K=128 INT8 = 128×2048×1B = 256 KB per layer (vs 2048×2048×2B = 8 MB bf16 full-rank). 32× reduction in W_Q storage.

**Why our findings make this work.** CF11 shows W_Q has concentrated spectrum (r_99/d ≈ 0.63) — the INT8 rounding error applied to a low-rank matrix is bounded by INT8 scale × remaining singular values, which are small for ranks > 128. The sub-additivity of rank truncation + quantization is a structural prediction from the concentrated spectrum. v2-CHEAP-TEST-001 killed cross-layer W_Q stacking but says nothing about INT8 compression of per-layer W_Q.

**What would have to be true.** The interaction between rank-128 truncation and INT8 quantization noise must be sub-additive (not superadditive). If the low-rank representation makes the INT8 grid coarser in the tail (since the tail is already truncated, the INT8 scale factor is set by the top-128 singular values), the quantization error could be lower than full-rank INT8.

**Experiment cascade.** E1 (20 min): quantize W_Q to INT8 per-layer (absmax scaling), apply rank-128 SVD, measure ΔNLL. Compare to CF11's bf16 K=128 reference (+0.98 nats). GO: ΔNLL ≤ 1.2 nats (sub-additive with CF11 reference). KILL: ΔNLL > 2.0 nats (superadditive — INT8 grid damages the low-rank structure).

**What you're NOT proposing.** Not a new compression method per se — a feasibility measurement for combining CF11 with INT8 quantization. Fills the **runtime-fusion gap** flagged in the task brief. Sits outside F-orientation's anti-frame (this is not plain quantization as the load-bearing move; the load-bearing claim is sub-additivity of rank truncation + quantization error, which is a falsifiable structural prediction from CF11's spectrum). Tag: `subspace-alignment` (INT8 quantization error bounded by sub-rank singular values, a spectrum-derived bound).

---

### S6 — PRKS: Prefix-Summary Running Statistics for Append-Only KV Compression [FREE SWING]

**Ambition (Track A).** Replace KV eviction with **in-place progressive summarization**: tokens older than a recency window (512 tokens) are compressed into a fixed-size running statistics buffer via a rank-2 sufficient statistic (mean + top-2 eigenvector, maintained incrementally via Oja's rule). The compressed summary is always exactly 2 × d_model floats per layer — independent of context length. The KV cache size at 32K context drops from O(32K × d_model) to 512 × d_model (recent window) + 2 × d_model (running summary) = ~514 × d_model.

**Mechanism.** During KV fill: maintain a running mean μ_V[L] ∈ R^{d_model} and a top-2 eigenvector online estimate U_V[L] ∈ R^{d_model×2} via Oja's rule (O(d_model) per step). When token t is evicted from the recency window, it is summarized into (μ_V, U_V) and discarded. During attention: queries in the recency window attend to the full KV pairs; queries outside attend to the (μ_V, U_V) summary via a rank-2 approximation of the old KV outer products. The quality depends on whether old tokens' V-vectors are approximately rank-2 (supported by CF3: top-2 channels Jaccard=0.718, suggesting V-vectors cluster around a 2-dim subspace for old tokens).

**Why our findings make this work.** CF3 at K=0.1% (top 2 channels) Jaccard=0.718 — for old tokens (away from BOS/delimiters), the top-2 V-channels are nearly static. A rank-2 running statistic captures these 2 static channels exactly and accumulates the dynamic channels in the mean. This is not a replication of StreamingLLM's attention-sink approach — PRKS compresses the full token into a summary rather than dropping it.

**What would have to be true.** V[t] for old tokens must be approximately rank-2 (justified by CF3's Jaccard). The Oja-rule online eigenvector estimate must converge in < 100 steps (standard Oja convergence guarantee for rank-1 streams; rank-2 requires modified power iteration). [FREE SWING]: this is speculative on the quality bound — E1 measures whether it holds.

**Experiment cascade.** E1 (30 min): measure rank of V[L,t] for old tokens (t < 512) on 50 × 4K prompts — if 90% of variance is in rank-2, the running statistic is sufficient. E2 (1 hr, conditional): implement running summary for a single layer, measure ΔNLL at 4K context with 512-token recency window.

**What you're NOT proposing.** Not H2O/SnapKV (eviction). Not StreamingLLM (attention sink only). Not a cross-layer approach. Fills the **output-side / KV-compute gap**. This is the PRKS [FREE SWING] slot.

---

### S7 — WDSC: W_down Spectrum Closure + Tropical Stable-Row Synergy

**Ambition (Track B).** Run the W_down SVD spectrum experiment (F-WNORM/WDRS, already Stage 3 REFINE) AND simultaneously measure A-TROP's stable-dominant-row fraction in the same script, producing a joint structural finding: (a) W_down CF8 closure, (b) stable-row fraction for TDIP (S4) feasibility. The synergy is that both measurements use the same forward-pass data (post-SwiGLU activations h_mlp over 200 calibration prompts). Running them jointly costs ~35 min instead of 65 min.

**Mechanism.** Single script: for each of 28 W_down layers, compute: (1) SVD of W_down[L] (2048×6144); record var@K for K ∈ {128,256,512,1024,2048}; (2) for each of 200 calibration prompts, extract h_mlp[L], compute argmax_j(W_down[L,i,j] × h_mlp[L,j]) for each output row i; compute stability fraction σ_i over 200 prompts. Output: (a) CF8 closure table for W_down, (b) histogram of σ_i across all 28 × 2048 output rows.

**Why our findings make this work.** CF8 closure (W_gate, W_up both full-rank) strongly predicts W_down full-rank. The d_out < d_in shape (2048 < 6144) means the effective rank is bounded by 2048, but 2048 full-rank is still expected. The measurement confirms this. Simultaneously, if A-TROP's stable-row fraction is ≥10%, TDIP (S4) is activated.

**Experiment cascade.** E1 (35 min): joint SVD + stability measurement. GO-CF8: W_down var@256 ≥ 0.99 in ≥ 20/28 layers → CF8 closes on W_down. GO-TROP: stable-row fraction ≥ 10% → S4/TDIP activated. Either-way structural finding.

**What you're NOT proposing.** Not a compression scheme — a joint structural measurement. Fills the stage-3-flagged W_down closure measurement with the added benefit of activating S4 at no extra cost.

---

### S8 — CQSB: Compressed W_Q with Sub-Additive INT8 Banding [FREE SWING]

**Ambition (Track A/B).** A [FREE SWING] proposal: instead of treating W_Q SVD truncation (K=128) and INT8 quantization as independent operations (S5), derive the OPTIMAL INT8 quantization grid FOR the K=128 truncated W_Q by exploiting the observation that the truncated matrix's singular value distribution is concentrated in the top 128 values. The key: after truncation to K=128, the remaining matrix has a specific condition number (ratio of max/min singular value among the 128 retained values) that determines the optimal INT8 scale factor. Using the actual condition number of the truncated matrix (rather than the absmax-based scale of the full matrix) gives a tighter INT8 grid and lower quantization error.

**Mechanism.** After SVD truncation to K=128: W_Q_r = U_r Σ_r V_r^T. The condition number κ_r = σ_max/σ_min of the truncated matrix (among K=128 singular values, not the original 2048) is much smaller than the full-matrix condition number (because the large-singular-value tail is removed). A per-matrix INT8 scale factor based on κ_r packs the 128 singular values more tightly into the INT8 grid, reducing round-off noise. The algebraic identity: quantize V_r^T in INT8 with scale = max(|V_r^T|)/127, quantize Σ_r separately in fp16, keep U_r in fp16. The inner product W_Q_r x = U_r (Σ_r (V_r^T x)) becomes a fp16 vector × fp16 diagonal × INT8 matrix-vector product — one cheap GEMV.

**What would have to be true.** The condition number of W_Q truncated to K=128 must be materially smaller than full W_Q's condition number. From CF11: W_Q has r_99/d=0.63, meaning the tail singular values are small — after truncation, σ_128/σ_1 is the reduced condition number, likely << full-matrix κ. This is measurable in 2 minutes from the AQFKV SVD result.

**Experiment cascade.** E1 (5 min): read σ_1 and σ_128 from AQFKV W_Q SVD result; compute κ_r. Compare to full-matrix condition number. GO: κ_r < κ_full / 2 → tighter INT8 grid meaningful. E2 (20 min, conditional): implement condition-number-aware INT8 W_Q and measure ΔNLL vs absmax INT8 at K=128. This is [FREE SWING]: the mechanism is novel and grounded but no prior work applies this condition-number argument to truncated-and-quantized attention weight matrices.

**What you're NOT proposing.** Not plain INT8 quantization. Not rank truncation alone. The structural novelty is the condition-number argument: truncation changes the optimal quantization grid, and the combined approach gives lower ΔNLL than treating them independently. Fills the **runtime-fusion gap** from a different angle than S5.

---

## 7. Output Handoff

### Stage 4 ideas summary

| ID | Track | One-line description | Motivating section |
|---|---|---|---|
| S1 — ATVIS | B | Attention-invisible channel selection → mixed-precision W_Q storage (INT4/INT8) | Sec 3 (F-GFQO stripped primitive) |
| S2 — ZWMP | A | Zone-map W_K block-skipping for long-context sparse attention (CF11 subspace bound) | Sec 4 (database engineer) |
| S3 — RDCP | B | Run-length + delta KV compression using CF3's K=0.1% static channels (append-only) | Sec 5 (append-only constraint) |
| S4 — TDIP | B | W_down dominant-row INT4 codebook from A-TROP stable-argmax structure | Sec 3 (A-TROP cross-pollination) |
| S5 — IQAT | B | INT8 × K=128 W_Q sub-additivity feasibility measurement (runtime-fusion gap) | Sec 5 (no-float constraint) |
| S6 — PRKS [FREE SWING] | A | Oja's-rule running rank-2 KV summary (append-only, no eviction) | Sec 5 (append-only constraint) |
| S7 — WDSC | B | Joint W_down SVD + tropical stable-row synergy (infrastructure for S4) | Sec 1 (CF8 closure) |
| S8 — CQSB [FREE SWING] | B | Condition-number-aware INT8 grid for truncated W_Q (κ_r < κ_full) | Sec 4 (compiler / runtime-fusion) |

Total: 8 ideas (6 standard + 2 [FREE SWING]).

Gaps filled:
- **KV-compute gap**: S2 (zone-map block skipping), S3 (run-length delta compression), S6 (running summary)
- **Output-side gap**: S6 partly (output of KV compression)
- **Training-dynamics gap**: S4 (tropical stable-row argmax stability = activation-trajectory structure)
- **Runtime-fusion gap**: S5, S8 (INT8 feasibility for compressed W_Q, condition-number-aware quantization)

### Orientation rotation recommendations

| Orientation | Recommendation | Rationale |
|---|---|---|
| R — Reach | KEEP / TIGHTEN | Add anti-frame: "cascade whose first rung is a shared spectrum measurement also proposed by another orientation — first rung must produce a structurally novel number" |
| C — Composition | KEEP / TIGHTEN | Add anti-frame: "coupling between two attention-side CFs only — next target is attention × MLP or training-dynamics × inference-structure coupling" |
| F — First-Principles | KEEP | Add anti-frame: "gauge-freedom arguments using unrestricted orthogonal rotation — must be restricted to RMSNorm-compatible class (permutations + sign flips)" |
| U — Unconventional Substrate | KEEP | Add anti-frame: "experiments that measure OS primitive engagement without connecting to measurable residency or throughput win on the inference workload" |
| A — Constraint-Alien | KEEP | Add anti-frame: "constraint whose practical binding collapses back to the standard playbook — 'no continuous numbers' → standard quantization is inadmissible" |

### Anti-frame additions to specific orientation anti-frame lists

- **R**: "First-rung shared spectrum measurement (W_V SVD at L14 is now in 5+ proposals; a Reach cascade must add a structural claim beyond what the shared measurement settles)"
- **C**: "Cross-attention-only coupling (W_Q × W_K × W_V interactions already cluster-saturated in run_004; next coupling must cross subsystem boundary: attention × MLP, attention × KV-dynamics, or weights × training-dynamics)"
- **F**: "Unrestricted-rotation gauge arguments (RMSNorm per-channel scale breaks O(d) symmetry; must restrict to permutation + sign-flip or demonstrate RMSNorm-bypass)"
- **U**: "GC stall / NVMe namespace proposals without first measuring whether GC stalls occur on this specific hardware config and workload (NNPA's E1 must run before NNPA's M2 is claimed)"
- **A**: "No-continuous-numbers constraint defaulting to standard quantization; no-RAM constraint defaulting to standard NVMe offload (the constraint must rule out, not just rename, the standard move)"

### Ideas recommended for Stage 5 consideration

**Primary (Stage 5 top 2):**
1. **S5 — IQAT**: fills the runtime-fusion gap with a 20-min decisive measurement. If ΔNLL ≤ 1.2 nats, it enables 32× W_Q storage reduction at sub-additive quality cost. The expected sub-additivity follows structurally from CF11's concentrated spectrum; either outcome is highly informative. Fastest Stage 4 idea.
2. **S3 — RDCP**: fills the KV-compute gap with a CF3-grounded compression scheme that is genuinely novel (append-only, no eviction, three-tier from CF3's exact Jaccard numbers). E1 (V-channel Jaccard measurement) is 20 min and uses the same PDAP infrastructure.

**Note for Stage 5 selector:** S7 (WDSC) should be run as infrastructure before S4 (TDIP); both can share a single experiment slot if bundled as "W_down joint characterization + stability probe."
