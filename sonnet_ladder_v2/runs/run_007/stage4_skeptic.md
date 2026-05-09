# Stage 4 — Skeptic Explorer — Run 007

Date: 2026-05-09
Input substrate: stage1_{R,C,F,U,A}, stage2_judge, stage3_deep_research (run_007); SUMMARY.md; KILL_LIST.md.

Run 007 signal: 9 REFINE (ATRC, WDCA, IDSM, FRCF, KSATT, PERM, QAWKC, RUNE, TBLK), 1 DOWNGRADE (WOVR — A3 pre-empts), 1 KILL (WDOS — CF12 isotropic), 1 REGENERATE (IOPB — CF9 partial).
Top 3 by Stage 3: ATRC, KSATT, FRCF.

---

## 1. Frame-Exhaustion Map

### 1.1 Saturation-exhausted frames (any new variant subsumed ≤ 2 quarters)

**MLP weight rank reduction (any form).**
CF7, CF8, R3–R4 direct measurement, R6-B Track B, R3-A Track A all killed this. Cross-layer Tucker, per-layer SVD truncation, NMF, JL-sketch (RFSK), Count-Sketch (CSKQ), tabulation-hash (THWR), Parity-Shard FFN — the kill list now has ~15 named variants. A3 (arXiv:2505.12942) covers W_V + W_O. SVD-LLM, ARA, GPTQ, AQLM, GPTVQ cover quantization-plus-compression. Any proposed "find structure in MLP weight matrices without retraining" will be killed by CF8 + prior art within one round.

Stripped primitive: the thing that's left after removing all the matrix-structure machinery is **per-neuron precision assignment from forward-pass statistics** — which is FRCF's remaining live frame.

**Tied lm_head compression.**
CF12 (catastrophic truncation at r<d), LHQD NO-GO, SVLD/ENVC all killed. The E^T-isotropic structure also killed WDOS this round. The stripped primitive is: **lm_head bandwidth reduction via factored-output arithmetic** (RUNE's live frame), not weight truncation.

**Cross-layer weight subspace sharing.**
v2-CHEAP-TEST-001 class kill. Cross-layer W_Q (8 variants), cross-layer W_V/W_K attempts — all dead. Stripped primitive: **per-layer spectral rank as a compression lever**, which is CF11's live frame (ATRC is the open extension).

**OS-paging / sequential weight layout tricks without compression.**
R1/S2 OS-Paging-Aware Weight Layout, R2/S2 NVMe Prefetch Sequencer, R3/S4 INPOLT, R7 AANF all hit the same wall: rearranging weight bytes without compressing them doesn't help at 70B. Stripped primitive: **byte-count reduction of what's on NVMe**, which maps to quantization / rank reduction, not layout.

### 1.2 CF9-exhausted frames

**Arbitrary rotation for residual-stream compression.**
v2-S3-R004-001 (GFQO) class-killed this. General O(d) rotation does not commute with RMSNorm per-channel γ. Variants that rely on "rotate the residual stream to expose structure" are dead unless the rotation is a permutation.

Stripped primitive: **permutation gauge** (S_{16} × S_{128} on attention blocks, confirmed to commute with RMSNorm) — PERM's live frame.

**Signal-processing spectral methods on weight matrices.**
RFSK (Bochner precondition: weight matrices are not shift-invariant kernels), CSKQ (Count-Sketch precondition: signal must be sparse), Haar wavelet Tucker (spatial structure precondition), DCT compression (R1/S2) — all CF9-killed on the same structural mismatch: LLM weight matrices are dense, non-stationary, non-sparse, and lack the geometric regularity that signal-processing machinery requires.

Stripped primitive: **algebraic properties of matrix products** — the W_{OV} composition, Kronecker reshuffling, and Δh PCA are the live frames that don't need regularity preconditions.

**IOCP as bandwidth governor.**
IOPB REGENERATED this round. The stripped primitive: **OS completion-port ordering as head-level computation serializer** — a narrower engineering primitive whose novelty dropped from A2=3 to approximately A2=2 after the CF9 strip.

### 1.3 CF10-exhausted frames

**High-parameter calibration-fit linear surgery.**
WDLA failure (R^2_eval = -118K), SPADC severity erosion (n_params >> n_samples). Any affine correction with >100K parameters fit on standard calibration data is CF10-dead without explicit ridge + calibration-data scaling. This exhausts: cross-scale model stitching, general-affine layer correction, MFAP mean-field propagation.

Stripped primitive: **low-rank-by-construction calibration fits** — at most 2 scalars (PERM's head-norm ratio) or per-neuron statistics (FRCF's F_j and c_E(j)), both safely conditioned.

---

## 2. Orientation-Saturation Diagnostics

**R (Reach):** Produced 5 advancers out of 7 (ATRC, CLOTS, VJOFR, ASOR, QAWKC). ATRC is the top Stage 5 candidate; CLOTS converges with v2-CF1 activation-outlier tier work. WOVR downgraded to A3 pre-emption; AANF killed for saturation overlap. Convergence clusters: R landed in C1 (W_V/W_O) and C2 (Jaccard × W_Q). Anti-frame addition: **"NVMe sequential layout tricks without weight byte reduction"** — AANF exemplified this; append to R's anti-frame list.

**C (Composition):** Produced 5 advancers out of 6 (JDWQ, JAQV, FRCF, JVSTAB, RUNE); WDOS killed by CF12. FRCF is the novel dual-signal coupling. JDWQ, JAQV, JVSTAB all require W_V and W_Q sweeps before they can progress — they are measurement-gated rather than structurally uncertain. Convergences: C2 (Jaccard × W_Q, three-orientation convergence) and C3 (W_down subspace). Anti-frame addition: **"E^T-invisible subspace compression under isotropic tied lm_head"** — WDOS exemplified this; append to C's anti-frame list.

**F (First-Principles):** Produced 4 advancers out of 6 (WOVR downgraded, LGFA rejected, WDCA/IDSM/KSATT/AORT advanced). KSATT is the most novel frame-novelty idea in this round. WDCA needs the skeptic-controls baseline per v2-CHEAP-TEST-001 discipline. LGFA killed (residency below threshold). Anti-frame addition: **"layer-1 gate-near-constancy as a global compression mechanism"** — CF6 SDZC already killed this; LGFA proposed it as a micro-optimization; append to F's anti-frame list.

**U (Unconventional Substrate):** Produced 1 advancer out of 5 (IOPB); all other U ideas killed or rejected at Stage 2 (RCWO, FSCW, TINA, WMCW insufficient A1 or speculation). IOPB was regenerated by Stage 3's CF9 strip. U orientation is **at-risk under the rotation rule**. However, IOPB's surviving narrow engineering experiment and CLOTS's cross-tier design both have U-flavored components, so U contributed to convergence even without direct advancement.

Recommendation: **DROP-FOR-ONE-ROUND (U)**. All five U ideas failed at Stage 2 or Stage 3 on the same failure mode — marginal A1 (≤ 2 ms/token or 21% NVMe reduction) against a 160-200 ms/token baseline that is weight-bandwidth-dominated. U's substrate-primitive framing is reaching for latency tricks while the pipeline needs byte-count reduction.

Proposed INTRODUCE-ROUND-SPECIFIC replacement: **"I/O Budget Accounting"** orientation.
- Creative directive: Treat the 70B inference token as a fixed I/O budget (bytes read per token = N). Every mechanism must be evaluated as "bytes read per token" not "speedup". Ideas that compress bytes (rank reduction, quantization) and ideas that reorder bytes (streaming, prefetch) are unified under the same budget lens. Cross-domain seed: file-system block allocator (Linux ext4 block groups, NTFS cluster runs) as the residency-budget model.
- Anti-frame list: NVMe layout reordering without compression; OS-paging tricks without byte reduction; DRAM-hot-set pinning of full-precision weights.
- Cross-domain seeds: SSD FTL (Flash Translation Layer) wear-leveling as a model for adaptive tier placement; TCP congestion control as a model for rate-adaptive weight streaming.
- Falsifiability requirement: each idea must report "bytes read per token, baseline vs proposed" as the primary metric; ideas claiming speedup without byte reduction are inadmissible.

**A (Constraint-Alien):** Produced 2 advancers out of 7 (A2-PERM via gauge-exploitation, A6-TBLK via wildcard). PERM is the strongest A-orientation output this round — it found the only admitted gauge symmetry in RMSNorm transformers (S_{16} × S_{128}) and is CF9-verified. TBLK is a free-swing wildcard. The non-advancers (PCAG, MONO, RDST, INTLOG, CADDR) all failed A4 or A1; their failure shapes are informative. Anti-frame addition: **"constraint-based projection that requires zeroing or sign-flipping trained weights"** — MONO and INTLOG both assumed that projecting trained weights onto a constrained grid would be lossless; CF4/CF5/CF8 refute this.

---

## 3. Cross-Pollination Opportunities

### 3.1 WDOS (KILL, CF12 isotropic) → stripped primitive

WDOS was killed because E^T has no dominant directions (r_99=1992/2048), so the "invisible subspace" for W_down is only ~56 dimensions. What survives: the **W_down output subspace as a structural object** — WDCA picks this up directly. But there is a cross-pollination the killer generated: if E^T is nearly isotropic and W_down's output must land somewhere in R^{2048}, then the question becomes NOT "which directions does E^T read?" but "which directions does W_down WRITE to?" — and whether those directions have any structural regularity that enables precision assignment independent of lm_head geometry.

Cross-pollination target: First-Principles orientation. WDCA is already asking this. The WDOS kill forces a cleaner formulation: **characterize W_down's output subspace not in relation to E^T but as a stand-alone structural object** — what fraction of R^{2048} does the union of all W_down outputs span? This is WDCA's exact question. The WDOS kill sharpens WDCA's framing.

**Cross-pollination CP1:** WDOS stripped primitive (W_down output subspace as structural object) → WDCA, resolved by the skeptic-controls-gated experiment already in Stage 3.

### 3.2 IOPB (REGENERATE, CF9 partial) → stripped primitive

IOPB's surviving primitive: OS completion-port ordering as a **head-level computation serializer** — not a bandwidth governor. The claim after strip: serializing attention head computations prevents peak DRAM pressure from concurrent head GEMVs. This is adjacent to the standard "tiled GEMV" kernel optimization.

Cross-pollination: the residency-cascade design in ATRC explicitly tracks attention weight bytes in DRAM (W_Q at CF11 K=128, W_V/W_O pending). Once ATRC's W_V/W_O sweep confirms their class, the IOPB stripped primitive re-enters as a **head-tiled streaming schedule** for DRAM-resident attention (not NVMe). The narrow engineering experiment is: for DRAM-resident attention with all four matrices (Q,K,V,O) compressed at CF11-derived ranks, does serializing head computation in groups of 4 reduce peak DRAM pressure by reducing simultaneous GEMV working sets?

**Cross-pollination CP2:** IOPB stripped primitive → Reach or U orientation in next round, as a head-tiled scheduling component of ATRC's DRAM-resident cascade.

### 3.3 WOVR (DOWNGRADE, A3 pre-emption) → stripped primitive

WOVR's compression application is dead to A3. What's left: the **U_O ≈ V_V alignment as a post-training SGD attractor** structural question. This is a free-swing structural measurement (8 min, a side-measurement of ATRC's W_V/W_O sweep). If the alignment is near-zero, it implies A3's activation-weighted fold is finding an algebraically trivial composition, not an activation-shaped one. If the alignment is large, it means A3's activation weighting is doing real work against a weight-space that resists the fold.

Cross-pollination: **the principal-angle measurement (U_O vs V_V) is a 3-minute add-on to ATRC's W_V/W_O SVD sweep** — zero additional experiment cost, produces a structural finding about why A3 works.

**Cross-pollination CP3:** WOVR structural science → ATRC experiment, 3-min add-on.

---

## 4. Cross-Domain Seed Exploration

### 4.1 Database engineer's view

A database engineer looking at the 70B inference problem sees: a relation R (weight matrices, 40 GB) and a hot-column set C (the per-token activated channels, ~1% per layer from CF3/v2-CF1). The standard database primitive for this pattern is **columnar storage with a per-column skip list** — store each column (=channel) of the weight matrix separately, with a B-tree index over which columns are hot (per-token dynamic outliers from v2-CF1).

**Elegant-equivalence prompt:** What algebraic identity does a database engineer's view of activation-channel outliers reveal — is the GEMV y = Wx expressible as a join over a pre-indexed column set C_hot (served from DRAM) and a remainder set C_cold (served from NVMe)?

The answer: **y = W[:,C_hot] · x[C_hot] + W[:,C_cold] · x[C_cold]**. This IS a column partition identity; it requires no approximation and no retraining. The savings come from storing W[:,C_hot] in DRAM (small: |C_hot| = 1% of d_model = 20 channels × d_model = 40K params per layer per matrix) and W[:,C_cold] on NVMe. If x[C_cold] ≈ 0 (it's not — CF3 says the set is token-dynamic, not sparse), savings are null. BUT: if x[C_cold] is quantized more aggressively (INT2 instead of INT4 for the cold columns), the error is bounded by the activation magnitude in C_cold — which is known from CF3 (K=0.1% = ~2 static channels carry most outlier energy; the dynamic 18 channels are large-but-bounded; the remaining 2028 are standard-magnitude).

This is RAOK's frame (Track B R6 deferred, still alive). The database join analogy makes it cleaner: **the activation column partition IS a database column projection**, and RAOK is the corresponding storage-tier assignment.

**Cross-domain seed 1:** Database columnar storage join → RAOK's per-channel tier assignment (DRAM hot / NVMe cold), grounded in the database join identity. This is a frame-novelty enrichment of an existing deferred idea, not a new idea.

### 4.2 Compiler writer's view

A compiler writer looking at transformer inference sees: **dead-code elimination** and **constant folding**. The constant-folding case is CF6: 36% of layer-1 gate outputs are near-constant across tokens → constant-fold those gate computations into a compile-time constant. LGFA tried this and was killed for negligible residency gain (0.25% of model). But the compiler writer's deeper question is: **profile-guided optimization (PGO)** — which code paths are actually exercised, and can infrequently-exercised paths be compiled to a cheaper representation?

**Elegant-equivalence prompt:** What computation-graph identity does a compiler writer's PGO view of SwiGLU neuron firings reveal — is there a subset of neurons whose firing-rank magnitude is effectively zero across all calibration tokens, enabling compile-time elimination of those neurons' W_up rows?

The profile data is FRCF's F_j statistic (mean firing magnitude per neuron). If some neurons have F_j ≈ 0 across ALL calibration tokens (not just token-dynamic — truly cold across all inputs), they can be eliminated without retraining. CF6's 1.5% globally near-constant gate neurons might correspond to near-zero F_j neurons, OR the two sets might be orthogonal. The compiler PGO view suggests: **profile-guided dead-neuron elimination** as a distinct class from gate-near-constancy. Dead neurons (F_j ≈ 0 globally) can be removed; low-F_j neurons are the candidate set for INT2 precision in FRCF.

**Cross-domain seed 2:** Compiler PGO dead-code elimination → dead-neuron identification via F_j globally-near-zero criterion. This motivates a new Stage 4 idea (S4 below).

### 4.3 Control theorist's view

A control theorist looking at the residual stream sees: a dynamical system h_{L+1} = h_L + δ_L(h_L) where the increment δ_L is a nonlinear function. The control theorist asks: **observability** — can the full state h_L be reconstructed from a compressed observation y_L = C·h_L where C ∈ R^{k×d} (k << d)? If the system is observable from a k-dimensional projection, the full-residual storage in the KV cache is unnecessary — only the k-dimensional observation need be stored.

**Elegant-equivalence prompt:** What conservation law does a control theorist see in the residual stream that would let a layer be replaced by a fixed-form operator on a conserved quantity?

CF2 (cos(h_L, h_{L+1}) ≈ 0.99) says the direction of h is approximately conserved; only the "perpendicular" component δ_L changes. A Kalman filter-style estimate: the state h_L lives near a one-dimensional manifold (the input embedding direction, per CF2 implication). The control-theoretic primitive is: **use a prior (the previous layer's state h_{L-1}) as a Kalman prediction of h_L; compute only the innovation δ_L = h_L - Kh_{L-1}**. If the innovation lives in a lower-dimensional subspace than h_L itself, storing the innovation instead of the full state compresses the KV cache.

This is RUNE's frame (Δh = h_L - h_0) but with a Kalman recursion instead of a one-shot projection. The control-theoretic version is more principled: the "base" is the running estimate, not just h_0.

**Cross-domain seed 3:** Control-theoretic Kalman innovation → innovation-compressed KV cache (store innovations, reconstruct states on read). This motivates a new Stage 4 idea (S5 below).

---

## 5. Constraint-Driven Reframing

**Constraint: No continuous numbers (forced ternary/binary).**
The frame: all weights are ternary {-1, 0, +1}. This is BitNet territory (arXiv:1702.00887, BitNet b1.58). Kill list has PTQTP (arXiv:2509.16989) which covers ternary compression at 1.58 bpw with SOTA results. This constraint's mechanism class is **saturated** — the kill list explicitly covers it. No new idea here.

**Constraint: Content-addressable (hash-keyed deduplication).**
CADDR (A3 this round) was rejected for hit-rate reasons. The stripped primitive: content-addressed deduplication of **weight blocks**, not residual states. LoRA-style weight deltas are often duplicated across layers of the same model — a Merkle hash of each weight block would reveal deduplication opportunities. But CF8 (MLP weights full-rank, no structure) means weight blocks are likely unique.
The mechanism class: **weight deduplication via content hash**. Most live in very saturated territory (GLDF was killed for being storage-only). No new idea here; constraint is CF9-adjacent for LLM weights.

**Constraint: No persistent state (weights reconstructed from residual stream alone).**
Per STAGE4_SKEPTIC template: "probably impossible — but the failure shape is a finding." The failure: W_gate and W_up are full-rank (CF8) and contribute information orthogonal to the residual stream — no residual-stream projection can reconstruct them. This is the CF8 mechanism. The structural finding IS already known. No new idea.

**Constraint: Storage sequential-only (forward scans only, no random reads).**
Forces: stream-oriented weight decode fused with matmul. This is the ANS-fused sparse residual idea (R2/S3, rescoped, still alive). The constraint motivates **stream-decode-fused GEMV** where weight decompression and multiplication happen in a single pass. This frame has a live idea (ANS-Fused Sparse Residual, R2/S3 advance). The Stage 4 question: is there a mechanism **not covered by ANS-fused** in this constraint class?

The NVMe sequential read pattern motivates: **layer-granular weight streaming with fused decompression** where the GGUF tensor layout is reorganized so GEMV-relevant weight rows are stored in access order. This is adjacent to killed AANF, BUT the sequential-only constraint eliminates the random-read problem that killed AANF. The relevant new primitive: **repack GGUF by GEMV row-access order for a specific batch size B**, so B consecutive inference tokens' weight accesses are sequential within each layer's GGUF extent. This is a batch-size-dependent layout optimization. This motivates idea S6 below.

**Constraint: 10 MB seed reconstruction (procedural model).**
Forces: generative / RNG-seeded weight construction. This is extremely speculative for full-precision LLM weights. No CF connects to this. Frame is too speculative for the ladder; skip.

**Constraint: No floating point (pure-integer + LUT).**
The alive mechanism: AVX2 W(ternary)A8 GEMV (R2/S4 deferred, R2/S3 H, still in pipeline). The constraint connects to ideas in the deferred list. The new angle: **lookup-table (LUT) based GEMV for the high-outlier channels** (the K=0.1% static channels from CF3 stored as INT8 with a 256-entry LUT per channel). This maps to RAOK's three-tier design but frames the static channel computation as LUT-based rather than FP16. This is a micro-optimization on top of RAOK, not a new idea.

---

## 6. Generated Ideas (S1–S8)

---

### S1 — Wdown Column-Space Entropy Budget for Layer-Wise Mixed-Precision MLP

**Elegance class:** `conserved-quantity`
**Track:** B
**Anti-frame white space:** Sits outside C (Composition) orientation's anti-frame "E^T-invisible subspace compression under isotropic tied lm_head" (which killed WDOS) — S1 does NOT depend on E^T structure at all. It also sits outside F (First-Principles) anti-frame "layer-1 gate-near-constancy global compression." It is in the C3 convergence white space that WDCA partially occupies.

**Ambition.** CF8 kills rank reduction of W_down; WDOS was killed because E^T is isotropic. But there is a third angle: even if W_down can't be rank-reduced and its output is not geometrically invisible to E^T, the **per-layer variance of W_down's output** may differ substantially across layers. Layers where W_down produces low-variance output (the MLP contributes little per token) can be quantized more aggressively than layers where the MLP output is high-variance and load-bearing.

**Mechanism.** Define σ²_down(L) = mean over calibration tokens of ‖W_down[L] · a_L‖²_2 where a_L is the post-SwiGLU activation. This is the per-layer MLP output energy, measured empirically over the calibration corpus. Hypothesis: σ²_down(L) varies substantially across layers (10–50× range), following a layer-depth profile similar to CF6's gate-variance finding (which found layer-1 is anomalously low). Mechanism: assign bpw inversely proportional to σ²_down(L). Layers with low MLP output energy (small σ²) tolerate INT2; layers with high output energy need INT4 or INT8.

**Why our findings make this work.** CF8 says W_down is likely full-rank (predicted by R3/R4 pattern). But "full-rank" and "high output energy" are different properties — a full-rank matrix can have a very small operator norm if all singular values are small. If some layers have small σ²_down, they are compressible to low precision because their quantization error magnitude (proportional to σ²_down × quantization_step) is small, not because they're rank-deficient. This is orthogonal to CF8.

**What would have to be true.** Spearman ρ(σ²_down(L), layer-ΔNLL-sensitivity(L)) > 0.30, AND σ²_down varies by ≥ 5× across layers. Both are measurable in 30 min.

**Experiment cascade.** (1) Compute σ²_down(L) for all 28 layers (20 min, calibration forward pass). (2) Compute per-layer ΔNLL at INT2 quantization of W_down[L] in isolation (one layer at a time, all others BF16 — 28 × 2 min = 56 min). (3) Spearman ρ. Total ≤ 90 min. Go: ρ > 0.30, σ²_down range ≥ 5×. No-go: flat σ²_down across layers — extends CF8 to say W_down is also uniformly load-bearing per layer, not just per rank.

**What you're NOT proposing.** Not W_down rank reduction (CF8-killed). Not E^T-invisible subspace (CF12-killed WDOS). Not calibration-fit linear correction (CF10 risk if the bpw-schedule is fitted; here bpw is assigned by a threshold, not a fit — CF10 safe).

---

### S2 — Attention Score Manifold PR as Adaptive KV Budget Scheduler [conserved-quantity]

**Elegance class:** `conserved-quantity`
**Track:** B
**Anti-frame white space:** Sits outside all orientations' anti-frame lists. IDSM was refined in Stage 3 but flagged for CF9 sample-size fix (T too large). S2 is the corrected version of IDSM targeted at a specific new application: **per-layer adaptive KV eviction budget** rather than a single global bound.

**Ambition.** IDSM (F7-2) proposed measuring the participation ratio PR(L) of the attention score distribution at each layer. Stage 3 identified a sample-size issue at T=128, n=200 (underdetermined). S2 proposes the corrected measurement AND a specific application of the per-layer PR(L) profile that IDSM did not: use PR(L) as a per-layer KV eviction budget. Layers with low PR (attention scores are low-dimensional) can afford aggressive KV eviction because the query is "looking at" a concentrated subset of keys anyway.

**Mechanism.** Measure PR(L) = (Σλ_i)²/(Σλ_i²) of the attention-score covariance at each layer (T=32 tokens, n=2000 sequences to avoid underdetermination — per Stage 3 CF9 fix). Predict: PR(L) is NOT uniform across layers — early layers are more concentrated (high PR, low dimensionality) and deep layers are more diffuse (low PR, high dimensionality). Use PR(L) to set the KV cache eviction budget: layers with PR/T² < 0.15 evict 50% of keys; layers with PR/T² > 0.50 evict only 10%. Total KV savings: weighted average over layers.

**Why our findings make this work.** CF11 showed attention weights are more structured than MLP weights (r_99/d ≈ 0.63 for W_Q vs 1.0 for MLP). If attention WEIGHTS are concentrated, the attention SCORES (products of queries and keys) should also exhibit concentration. PR(L) is the score-side analog of CF11's weight-side r_99/d. The coupling: layers with small r_99/d (W_Q) should have small PR/T² (concentrated scores) because query diversity is limited.

**What would have to be true.** Spearman ρ(r_99(W_Q^L)/d, PR(L)/T²) > 0.20 across the 28 layers. This couples CF11 (weight spectrum) with IDSM (score manifold) — a second-pass convergence.

**Experiment cascade.** (1) Run IDSM corrected (T=32, n=2000, per Stage 3 fix) — 25 min. (2) Check Spearman coupling with per-layer W_Q r_99 (reuse data from C7-JDWQ/C2 settler). (3) Simulate per-layer eviction budget (compute ΔNLL under PR-guided eviction vs uniform eviction at matched total-KV budget). Total ≤ 2 hours. Go: PR varies ≥ 3× across layers, ρ > 0.20, ΔNLL under PR-guided eviction < uniform at same budget. No-go: flat PR — attention score dimensionality is uniform, KV eviction budget must be uniform.

**What you're NOT proposing.** Not cross-layer W_Q stacking (v2-CHEAP-TEST-001 killed). Not H2O/ScissorHands engineering KV pruning (no structural grounding). Not CF15 (intrinsic dimension of full residual stream — unconfirmed; this measures attention SCORES, not residual).

---

### S3 — ATRC Companion: W_O and W_V Decomposition at GQA-Correct Shapes [second-pass convergence]

**Elegance class:** `conserved-quantity`
**Track:** B
**Anti-frame white space:** Converges with C1 cluster (ATRC, VJOFR, WOVR). Not a new idea in the Stage 4 sense — this is a convergence enforcer that resolves the highest-leverage open question identified in Stage 3. Tagged here because Stage 4 must include it to prevent omission.

**Note:** Stage 3 identified the W_V and W_O SVD sweep as the highest-priority pre-program experiment (35 min, resolves 5 advancers simultaneously). S3 formalizes this as a Stage 4 idea to ensure it enters Stage 5's selection pool with proper grounding.

**Ambition.** Complete CF11's attention-weight class characterization by measuring W_V (GQA shape: 8 heads × 128 × 2048 in Qwen3-1.7B) and W_O (standard: 16 heads × 128 × 2048) spectral concentration using the same CF11-style r_99/d metric. Either outcome (CF11-class or MLP-class) is a load-bearing structural finding that either opens or closes the attention-tiered cascade for 70B DRAM residency.

**Mechanism.** Standard per-matrix SVD truncation sweep: compute r_99/d per layer for W_V and W_O; measure ΔNLL at K ∈ {512, 256, 128} for W_V (GQA-corrected); measure ΔNLL at K ∈ {256, 128} for W_O. Sharpens CF11's boundary to include the full attention-weight family.

**What would have to be true.** W_V r_99/d < 0.80 (CF11-class, not MLP-class); ΔNLL(W_V, K=256) < 0.50 nats.

**Experiment cascade.** 35 min (per Stage 3 ATRC smallest-test specification). 8-hour gate: well within.

**What you're NOT proposing.** Not cross-layer W_V/W_O stacking (v2-CHEAP-TEST-001). Not activation-weighted fold (A3 covers that). Not joint Q-K-V training (MLA, retraining required).

---

### S4 — Profile-Guided Dead-Neuron Elimination: F_j Global Minimum as Compile-Time Constant

**Elegance class:** `algebraic-identity`
**Track:** A
**Anti-frame white space:** Sits outside all orientation anti-frame lists. LGFA (F7-6) was killed for negligible residency (0.25% from layer-1 gate constancy). CF6 SDZC was killed as a global scheme (1.5% globally foldable). S4 is structurally different: it targets neurons where **the FIRING MAGNITUDE F_j is near-zero across all calibration tokens**, not neurons where the gate OUTPUT is near-constant. These are different structural criteria that may or may not overlap.

**Ambition.** A compiler's PGO view: neurons that never fire strongly across calibration are the "dead code" of the MLP. If F_j = mean |silu(W_gate[j]·x) × W_up[j]·x| ≈ 0 globally (not just on BOS tokens), those neurons can be algebraically folded to zero-bias constant: their contribution to the MLP output is W_down[:,j] × 0 = 0. The fold is: remove W_gate row j, W_up row j, and W_down column j from the computation graph. This is algebraically exact on calibration; it is an approximation on out-of-distribution tokens.

**Mechanism.** Threshold: neuron j is "dead" if F_j < ε across all 200 calibration tokens (not mean < ε — the minimum is < ε, i.e., the neuron fires near-zero even on its most-activating calibration input). Expected fraction: likely 1–5% globally based on CF6's gate near-constancy results. If even 2% of d_FFN = 6144 neurons are globally dead: 0.02 × 6144 × 28 layers = 3441 neuron-slots. Savings: 3441 × (W_gate row + W_up row + W_down col) × 2B = 3441 × 3 × 2048 × 2B ≈ 40 MB. Negligible at 1.7B. At 70B with d_FFN = 28672: 0.02 × 28672 × 80 × 3 × 8192 × 0.5B (INT4) = 0.02 × 28672 × 80 × 3 × 4 KB ≈ 2.8 GB. Non-trivial if the 2% estimate holds at 70B.

**Why our findings make this work.** CF6 found 36% of layer-1 neurons have near-constant gate output. Near-constant gate output is related to near-zero F_j: if silu(W_gate[j]·x) ≈ c_j (a constant), then F_j = |c_j × mean(W_up[j]·x)|. If c_j is small (near-zero), F_j is near-zero. The CF6 finding motivates the hypothesis that a fraction of neurons have small F_j even globally. The FRCF experiment (S4 uses its F_j data as input) provides this measurement for free.

**CF9 check.** No imported theorem. F_j is a standard forward-pass statistic. Precondition: calibration corpus is representative enough that near-zero F_j on 200 tokens implies near-zero on production inputs. Mitigation: measure with a held-out 500-token validation split and verify F_j rank order is consistent.

**CF10 check.** No calibration fit. F_j is a statistic, not a parameter. Safe.

**What would have to be true.** ≥ 1% of neurons have F_j < 0.01 across all 200 calibration tokens AND the ΔNLL from removing those neurons < +0.05 nats (near-lossless).

**Experiment cascade.** (1) Compute F_j per neuron per layer (reuses FRCF's forward-pass data — 0 additional time if FRCF runs first). (2) Identify neurons with min(F_j across all calibration tokens) < 0.01. (3) Zero out those neurons; measure ΔNLL. Total: 20 min additional beyond FRCF. Go: ≥ 1% dead neurons, ΔNLL < +0.05. No-go: <0.1% dead neurons — CF8's "full-rank" structure extends to neuron-level firing; dead-neuron elimination is negligible.

**What you're NOT proposing.** Not gate near-constancy (SDZC, killed). Not MLP rank reduction (CF8). Not activation-outlier quantization (RAOK, PDAP).

---

### S5 — Kalman-Innovation KV Cache: Store Residual Increments, Not Full States

**Elegance class:** `conserved-quantity`
**Track:** A
**Anti-frame white space:** Sits outside all orientation anti-frame lists. Closest kill: RECS (EMA recurrent KV, killed by RetNet/RADLADS). S5 differs: it does NOT propose a recurrent state machine; it proposes **storing KV cache as innovations** (differences between adjacent-layer residuals) rather than full states, exploiting CF2's near-constant-direction property to compress KV cache on the storage side.

**Ambition.** CF2: cos(h_L, h_{L+1}) ≈ 0.99 means h_{L+1} ≈ h_L + δ where δ is a small innovation. The KV cache stores (K_L, V_L) = (W_K^L · h_L, W_V^L · h_L) for each (L, token) pair. Instead of storing (K_L, V_L) directly, store (ΔK_L, ΔV_L) = (K_L − K_{L-1}, V_L − V_{L-1}), the per-layer innovation. If δ is small, the innovations ΔK and ΔV are also small — potentially more compressible (lower entropy, smaller magnitude) than the full states.

**Mechanism.** At KV cache write time: compute K_L = W_K^L · h_L; store ΔK_L = K_L − K_{L-1} (or K_L − E[K_L] from calibration mean). At KV cache read time: reconstruct K_L = Σ_{l=1}^{L} ΔK_l + K_0. This reconstruction is a prefix sum — O(L) operations per attention query. The compression payoff: if ΔK_L has ≤ 4× smaller magnitude than K_L on average, then ANS/zstd entropy coding of ΔK_L achieves ≤ 4× better compression than coding K_L directly.

**CF9 check.** No imported theorem; prefix-sum reconstruction is standard. Precondition: the innovation ΔK_L is actually smaller-magnitude than K_L. This is the measurable claim. CF2 motivates the hypothesis but doesn't prove it: CF2 measures residual-stream cosine similarity, not KV-vector delta magnitudes.

**CF10 check.** No calibration fit. The innovation is a deterministic function of the KV states. Safe.

**What would have to be true.** mean(‖ΔK_L‖) / mean(‖K_L‖) < 0.30 for ≥ 20 of 28 layers (innovation < 30% of state magnitude on average). This would imply KV cache entropy after delta-coding is at least 3× lower than raw, enabling 3× KV cache compression without quality loss.

**Experiment cascade.** (1) Forward pass 200 calibration tokens; collect K_L for all 28 layers and all token positions. (2) Compute ΔK_L = K_L − K_{L-1}; measure ‖ΔK_L‖/‖K_L‖ per layer. (3) Entropy-estimate: zstd compress K_L bytes vs ΔK_L bytes on the calibration corpus. Total: 30 min. Go: delta ratio < 0.30 for ≥ 20 layers AND zstd compression ≥ 2× better on deltas. No-go: delta ratio ≥ 0.70 — CF2's residual near-unity does not propagate to K/V vector stability; the innovation is as large as the state; delta coding provides no benefit. Structural finding: KV vector stability is NOT predicted by residual-stream cosine similarity; the two measures are decoupled.

**What you're NOT proposing.** Not EMA recurrent state (RECS, killed). Not GQA / MQA / shared-K/V (training-time decisions). Not cross-layer K/V aliasing (CLASE, bottleneck mismatch at <32K context). Not KV eviction (H2O / SnapKV territory).

---

### S6 — Batch-Stratified GGUF Repack: Sequential-Scan Weight Layout for B-Token Parallelism

**Elegance class:** none (engineering mechanism)
**Track:** B
**Anti-frame white space:** Sits outside the saturation frame "NVMe sequential layout tricks without byte reduction" (which killed AANF) because S6 DOES reduce bytes-read-per-token by enabling batch processing. The critical difference from AANF: S6 changes the computation graph (batch B tokens simultaneously through each layer) to convert random reads into sequential reads — not just relayout for solo-token inference.

**Ambition.** Standard ik_llama.cpp inference processes one token at a time: for each layer L, read W_gate[L], W_up[L], W_down[L], W_Q[L], W_K[L], W_V[L], W_O[L] from NVMe. For B tokens processed in a batch: each matrix is read ONCE (not B times), amortized over B tokens. Effective bytes-per-token drops from (total model size) to (total model size / B). At B=4: 4× bytes-per-token reduction. At B=8: 8×. The GGUF layout for batched inference differs from solo-token inference: for batch processing, each layer's weight blocks should be contiguous so a single sequential read serves all B tokens' GEMV.

**Mechanism.** Re-layout GGUF to be batch-optimal: for each layer L, pack {W_gate[L], W_up[L], W_down[L], W_Q[L], W_K[L], W_V[L], W_O[L]} as a single contiguous extent. For B-token inference: issue one ReadFileEx per extent (not 7 reads × 28 layers × B tokens = 196B reads); process all B tokens' GEMVs from the same weight block before moving to L+1. This is the GEMM (matrix-matrix) path rather than the GEMV (matrix-vector) path: B tokens × GEMM replaces B × GEMV, trading serial reads for batch arithmetic.

**CF-grounding.** Connects to CF3/v2-CF1 (activation outlier dynamicity): for B-token batches, the static outlier channels (K=0.1%, CF3 Jaccard=0.718) are stable across the B tokens, enabling static-channel optimization within the batch. Connects to CF11 (W_Q compressed at K=128): with compressed attention weights in DRAM, the NVMe batch reads are only for MLP weights — simpler layout.

**CF9 check.** No imported theorem. Batch GEMM vs B×GEMV is standard; the advantage is that NVMe bandwidth is fully utilized (single large sequential read) rather than B small random reads. Precondition: the Ryzen 5 7530U's NVMe can sustain large-block reads at ≥ 2× the bandwidth of small-block reads. Measurable.

**CF10 check.** No calibration fit. Safe.

**What would have to be true.** GGUF NVMe sequential block reads of 100–500 MB achieve ≥ 1.5 GB/s throughput; batch GEMM at B=4 fits in 7.28 GiB DRAM without swapping. At 1.7B with 4 tokens: model 1 GB + KV 4× 56 MB = 224 MB + batch activations 4 × 0.5 MB ≈ 1.2 GB total. Fits. At 70B: 40 GB NVMe (model) + 4 × KV cache per token ≈ too large for DRAM; but sequential read amortization is the NVMe bandwidth gain.

**Experiment cascade.** (1) Measure GGUF sequential read throughput at block sizes {10 MB, 100 MB, 500 MB} using Python mmap (20 min — re-derives CF13*). (2) Implement B=4 batch forward pass in PyTorch on Qwen3-1.7B (GEMM path); measure tok/s vs B=1. (3) If (2) improves tok/s ≥ 1.5×, repack GGUF for layer-contiguous layout and re-test. Total: ≤ 4 hours. Go: ≥ 1.5× tok/s at B=4 batch inference. No-go: DRAM bandwidth bottleneck swamps the NVMe gain; batch processing hurts throughput due to DRAM pressure.

**What you're NOT proposing.** Not speculative decoding (draft model required). Not AANF (solo-token layout). Not attention-only optimization (this is primarily MLP-focused). Not the ANS-fused stream (different mechanism class).

---

### S7 — Kronecker Rank-r Decomposition vs SVD at Matched Parameters for W_Q

**Elegance class:** `algebraic-identity`
**Track:** B
**Anti-frame white space:** Sits directly in KSATT's frame (post-training Kronecker test) but is a NEW idea that goes beyond KSATT's smoke test. KSATT tests whether Kronecker structure EXISTS (≥ 50% Frobenius at rank-1). S7 assumes the KSATT result was positive and asks: at rank r matched to the same parameter budget as SVD K=128, does Kronecker compression provide lower ΔNLL than SVD?

**Ambition.** KSATT is a 5-min smoke test. If KSATT returns GO (≥ 50% Frobenius at rank-1), S7 is the follow-on: measure the ΔNLL of W_Q at rank-r Kronecker (r factors of A_i ⊗ B_i) vs rank-K SVD at matched total parameter count. If Kronecker achieves lower ΔNLL at equal parameters, it is a structurally superior basis for attention compression — the implication being that SGD implicitly induced Kronecker structure that SVD does not align with.

**Mechanism.** At p=32, q=64 (best partition from KSATT): rank-r Kronecker for W_Q per layer is A_r ∈ R^{32×32}, B_r ∈ R^{64×64} for each of r factors. Parameter count: r × (32²+64²) = r × 5120. Matched SVD: K = r×5120/(2×2048) = r×1.25 (approximately). At r=16 Kronecker = 81K params vs SVD K=40 = 40×2048×2 = 163K params — Kronecker is 2× more parameter-efficient. The comparison: ΔNLL(Kronecker r=16) vs ΔNLL(SVD K=40) at matched parameter count. If Kronecker ΔNLL < SVD ΔNLL: the Kronecker structure is "natural" to W_Q in a way that SVD misses.

**CF9 check.** Kronecker reshuffling is standard linear algebra. The precondition from KSATT: ≥ 50% Frobenius at rank-1 (the GO threshold). If KSATT was NO-GO, S7 is moot. Precondition is explicitly gated on KSATT result.

**CF10 check.** Kronecker factorization is a weight decomposition (no calibration data). Safe. The ΔNLL eval uses standard forward-pass evaluation, not calibration fitting.

**What would have to be true.** KSATT GO AND Kronecker ΔNLL < SVD ΔNLL at matched parameters AND the partition found by KSATT (p=32, q=64 or similar) provides ≥ 2× parameter efficiency gain over SVD at the same quality.

**Experiment cascade.** KSATT first (5 min). If GO: run W_Q Kronecker decomposition at r=4,16,64 for layer 14 head 0; compute ΔNLL; compare to CF11's SVD-K data at matched parameters. 30–45 min total. Go: Kronecker ΔNLL < SVD ΔNLL at matched params for ≥ 3 rank values. No-go: Kronecker ΔNLL ≥ SVD ΔNLL — the partial Frobenius structure from KSATT is "decorative" (doesn't help reconstruction quality).

**What you're NOT proposing.** Not Monarch matrices or Kaleidoscope (training-time). Not arbitrary rotation (O(d), killed by v2-S3-R004-001). Not cross-layer W_Q stacking (v2-CHEAP-TEST-001).

---

### S8 — W_Q Column-Norm Outlier Pinning: Static High-Energy Input Channels for Attention Projection [FREE SWING]

**Elegance class:** `subspace-alignment`
**Track:** B
**Free swing:** [FREE SWING] — connects to CF3 (static outlier channels at K=0.1%) and CF11 (W_Q compressed at K=128) but the specific coupling (W_Q column-norm outliers as static quantization targets) has no prior CF tether on the v2 ladder.

**Ambition.** CF3: K=0.1% activation channels are static (Jaccard=0.718) — the ~2 channels with highest activation magnitude are the same across tokens. Stage 3 KILL_LIST noted this finding motivated RAOK for activation quantization. S8 asks: do those same 2 static channels have disproportionately large W_Q column norms? If ‖W_Q[:,c_static]‖_2 >> ‖W_Q[:,c_random]‖_2 for the static channels c_static, then those channels punch above their weight in the Q projection — they are the "routing channels" for attention.

**Mechanism.** Per v2-S3-R009-001 (KILL_LIST entry for F2-CKDJ): "measure whether CF3 static channels = top W_Q column energy channels; if YES, motivates per-column precision allocation." S8 takes that pass from Stage 3's kill-list note and executes it: (1) identify top-2 static outlier channels c_static per layer from CF3/v2-CF1; (2) compute W_Q[:,c_static] column L2 norm vs median channel norm; (3) if c_static channels have column norm ≥ 2× median, they require FP16 precision in W_Q even after truncation to K=128 — they are the "escape cases" that CF11's global SVD compression might inadvertently discard.

**What would have to be true.** ‖W_Q[:,c_static]‖_2 / median(‖W_Q[:,:]‖_2) ≥ 2.0 at ≥ 20 of 28 layers. If YES: the static outlier channels from CF3 are also high-energy in W_Q — they are a consistent attention routing axis. The practical implication: any W_Q compression (CF11 K=128 truncation, PERM mixed-precision) must explicitly preserve the c_static channels at full precision to avoid losing the primary routing signal.

**CF9 check.** No imported theorem. Column L2 norm is standard. Precondition: the static channels identified by v2-CF1 (consecutive-pair Jaccard at K=0.1%) are the same channels as the top-energy W_Q columns. This is the falsifiable claim.

**CF10 check.** No calibration fit. Pure weight+activation statistics. Safe.

**Experiment cascade.** (1) Load W_Q for all 28 layers; compute per-column L2 norm (10 min). (2) Cross-reference with c_static channels from v2-CF1 RAOK result (already measured). (3) Report overlap. Total: 15 min. Go: norm ratio ≥ 2× for c_static channels (confirming activation routing and weight energy are colocated). No-go: uniform W_Q column norms — the static outlier channels are NOT attention routing axes; CF3 and CF11 are structurally independent.

**What you're NOT proposing.** Not per-column W_Q low-rank decomposition (CF11 is per-row/global-SVD, not per-column). Not cross-layer W_Q (killed). Not GFQO rotation (killed). The experiment is a 15-min measurement that either confirms or refutes the RAOK × ATRC structural coupling.

---

## 7. Output Handoff

### Stage 4 ideas, section motivation:

| # | Name | One-line | Section |
|---|---|---|---|
| S1 | W_down Layer-σ² Mixed-Precision | Per-layer MLP output energy as bpw scheduler — orthogonal to CF8's rank finding | §1 (stripped W_down primitive), §5 (sequential constraint) |
| S2 | PR-Guided KV Budget | Attention-score participation ratio as per-layer KV eviction scheduler | §3 CP1 + §4 control theorist |
| S3 | ATRC W_V/W_O Sweep (convergence enforcer) | Complete CF11 boundary to W_V and W_O — highest-leverage C1 experiment | §3 CP3 (convergence enforcer) |
| S4 | Profile-Guided Dead-Neuron Elimination | F_j global minimum as compile-time constant fold for near-zero neurons | §4 compiler PGO |
| S5 | Kalman-Innovation KV Cache | Delta-code KV cache using adjacent-layer increments; test entropy gain | §4 control theorist |
| S6 | Batch-Stratified GGUF Repack | B-token GEMM path converts B×GEMV random reads to single sequential scan | §5 sequential-only constraint |
| S7 | Kronecker vs SVD at Matched Parameters | Follow-on to KSATT: does implicit Kronecker structure beat SVD at equal param count? | §1 (KSATT stripped primitive), §4 database engineer (structured decomposition) |
| S8 | W_Q Column-Norm Outlier Pinning [FREE SWING] | CF3 static channels × W_Q column energy — routing axis identification | §3 CP2 (IOPB stripped primitive → routing characterization) |

**Count: 8 ideas (within 6–10 range). 1 FREE SWING (S8). No calibration fits (CF10 clean on all). All pass CF9 internal check.**

---

### Orientation rotation recommendations:

| Orientation | Recommendation | Rationale |
|---|---|---|
| R (Reach) | KEEP + TIGHTEN | 5/7 advancers; ATRC is top candidate; anti-frame addition: "NVMe layout without byte reduction" |
| C (Composition) | KEEP | 5/6 advancers; FRCF and RUNE are high-value; WDOS killed is first C-orientation kill but CF12 was the killer |
| F (First-Principles) | KEEP | KSATT is the round's most novel idea; WDCA/IDSM both REFINE; anti-frame tightening on CF6 SDZC |
| U (Unconventional Substrate) | DROP-FOR-ONE-ROUND | All 5 U ideas failed Stage 2 or Stage 3 on marginal A1; no convergence contributions except IOPB partial |
| A (Constraint-Alien) | TIGHTEN | 2/7 advancers (PERM, TBLK); 5 killed/rejected; anti-frame additions from MONO/INTLOG/RDST needed |

### Anti-frame additions:

- **R orientation:** Append "NVMe sequential-layout tricks without byte-count reduction (AANF exemplar)" — any idea that reorganizes weight bytes without compressing them is inadmissible for R's residency-cascade framing.
- **C orientation:** Append "E^T-invisible subspace compression under isotropic tied lm_head (WDOS exemplar)" — CF12 isotropy kills any coupling between lm_head structure and W_down output subspace geometry.
- **F orientation:** Append "layer-1 gate near-constancy as a global compression mechanism (LGFA exemplar)" — CF6 SDZC already killed this globally; layer-1 isolation saves < 0.3% of model bytes; not an F-frame idea.
- **A orientation:** Append "constraint-based weight projection via sign-zeroing or nearest-grid rounding (MONO/INTLOG exemplar)" — trained Qwen3 weights have no log-uniform or positive-only structure; any quantization grid that ignores the actual weight magnitude distribution is inadmissible.

### Stage 5 top recommendations (1–2):

**Primary — S3 (ATRC W_V/W_O spectrum sweep):** 35-min experiment that resolves the C1 cluster (5 advancers simultaneously), either extends CF11 to all attention matrices (opens the 70B attention-tiered cascade) or finds W_V/W_O are MLP-class (closes the cascade cleanly). This is the highest expected-information-gain single experiment available.

**Secondary — KSATT / S7 compound:** KSATT smoke test (5 min) gates S7 (30–45 min). If KSATT GO: S7 determines whether Kronecker compression beats SVD — potentially 2× better parameter efficiency than CF11's current best. If KSATT NO-GO: 5 min closes the entire Kronecker class and the slot goes to FRCF (45 min, dual-signal neuron precision, confirmed novel by Stage 3, CF10 safe).
