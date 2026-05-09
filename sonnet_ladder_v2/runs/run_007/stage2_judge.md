# Stage 2 — Judge — Run 007

Independent cold-start. Inputs: stage1_R, stage1_C, stage1_F, stage1_U, stage1_A. SUMMARY.md (CF1–CF12, v2-CF1, v2-CF2). KILL_LIST.md (v2-CHEAP-TEST-001, v2-S3-R004-001, full prior ladder). ORIENTATIONS.md. No prior run scores consulted.

Date: 2026-05-09

---

## 1. Union Score Table

| ID | Orientation | Track | Title | A1 | A2 | A3 | A4 | A5 | Total | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| R7-R-001 ATRC | R | B | Attention-Weight Tiered Residency Cascade | 2 | 2 | 3 | 2 | 3 | 12 | ADVANCE |
| R7-R-002 CLOTS | R | B | Cross-Layer Outlier-Highway Tiered Store | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE |
| R7-R-003 VJOFR | R | A | W_V / W_O Joint Folded Residency | 1 | 2 | 3 | 2 | 3 | 11 | ADVANCE-convergence |
| R7-R-004 ASOR | R | A | Attention Subspace × Outlier Channel Joint Routing | 1 | 2 | 3 | 2 | 3 | 11 | ADVANCE-convergence |
| R7-R-005 QAWKC | R | B | Quantization-Aware W_Q/W_K Cascade for 70B [FREE SWING] | 1 | 2 | 3 | 2 | 3 | 11 | ADVANCE-wildcard |
| R7-R-006 LGOP | R | A | Layer-1 SDZC + MLP-Side Outlier Pinning Composition | 1 | 1 | 3 | 2 | 2 | 9 | REJECT-low-score |
| R7-R-007 AANF | R | B | Attention-Weight NVMe Prefetch Oracle via Static Access Fingerprint | 1 | 1 | 2 | 1 | 2 | 7 | REJECT-low-score |
| C7-JDWQ | C | B | v2-CF1 Layer-Jaccard × W_Q Per-Layer r_99 Correlation Oracle | 1 | 2 | 3 | 2 | 3 | 11 | ADVANCE-convergence |
| C8-WDOS | C | B | W_down Output-Space × Tied-Embed Span Disjointness | 2 | 2 | 3 | 2 | 3 | 12 | ADVANCE |
| C9-JAQV | C | B | Jaccard-Layer Gradient × W_Q / W_V Rank Budget Joint Allocation | 1 | 2 | 3 | 2 | 3 | 11 | ADVANCE-convergence |
| C10-FRCF | C | B | Firing-Rank × Embed Span: W_up Row-Norm × Tied-Embed Column Dominance | 2 | 2 | 2 | 2 | 3 | 11 | ADVANCE |
| C11-JVSTAB | C | A | Jaccard Stability Tier × W_V Subspace Reuse Rate | 1 | 2 | 2 | 2 | 3 | 10 | ADVANCE-convergence |
| C12-RUNE | C | A | Residual Near-Unity × Embed Logit Factorization [FREE SWING] | 1 | 2 | 3 | 2 | 3 | 11 | ADVANCE-wildcard |
| F7-1 WOVR | F | A | W_O Left-Singular / W_V Right-Singular Spectral Coupling | 2 | 2 | 3 | 3 | 3 | 13 | ADVANCE |
| F7-2 IDSM | F | B | Intrinsic Dimension of Attention Score Manifold | 1 | 2 | 3 | 3 | 3 | 12 | ADVANCE |
| F7-3 AORT | F | A | Attention Output Subspace Dimension as W_Q Compression Bound | 1 | 2 | 3 | 2 | 3 | 11 | ADVANCE-convergence |
| F7-4 KSATT | F | B | Kronecker Structure in Attention Weight Matrices | 1 | 3 | 3 | 2 | 3 | 12 | ADVANCE-frame-novelty |
| F7-5 WDCA | F | B | W_down Column-Space Alignment with Residual Stream | 1 | 2 | 3 | 3 | 3 | 12 | ADVANCE |
| F7-6 LGFA | F | A | Layer-1 Gate Folding Verified Algebraic Identity [FREE SWING] | 0 | 1 | 3 | 3 | 2 | 9 | REJECT-low-score |
| U1-B RCWO | U | B | Windows RAM-Mapped Compressed Weight Overlay via CreateFileMapping | 1 | 2 | 2 | 2 | 2 | 9 | REJECT-low-score |
| U2-B FSCW | U | B | Windows File System Cache Write-Barrier as Inference-Epoch Marker | 1 | 2 | 2 | 2 | 2 | 9 | REJECT-low-score |
| U3-B TINA | U | B | GGUF Tensor-Interleave Reordering for NUMA-Like L3 Affinity | 0 | 2 | 2 | 1 | 3 | 8 | REJECT-low-score |
| U4-A IOPB | U | A | Windows I/O Completion Port Token Budget as Attention-Head Bandwidth Regulator | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE-frame-novelty |
| U5-B WMCW | U | B | Windows Memory Compression as Weight Cold-Tier Proxy [FREE SWING] | 1 | 2 | 2 | 2 | 2 | 9 | REJECT-low-score |
| A1-PCAG | A | A | Per-Channel Accumulator Gate — Integer-Only Residual Stream | 0 | 2 | 2 | 2 | 3 | 9 | REJECT-low-score |
| A2-PERM | A | B | Permutation-Only Gauge — Weight-Reordering for Codebook Alignment | 1 | 2 | 2 | 2 | 3 | 10 | ADVANCE-elegant-equivalence [gauge-exploitation] |
| A3-CADDR | A | B | Content-Addressed Residual Deduplication — Hash-Keyed KV Bypass | 0 | 3 | 3 | 1 | 3 | 10 | REJECT-low-score |
| A4-MONO | A | A | Monotone Computation Graph — Non-Decreasing Functions | 0 | 2 | 2 | 1 | 3 | 8 | REJECT-low-score |
| A5-RDST | A | A | Reversible Decode Stack — Bennett-Style No-Erase Inference | 0 | 3 | 2 | 1 | 3 | 9 | REJECT-low-score |
| A6-TBLK | A | A | Table-Driven Block Decode — Inference as Dictionary Lookup [FREE SWING] | 0 | 3 | 3 | 2 | 3 | 11 | ADVANCE-wildcard |
| A7-INTLOG | A | B | Integer-Logarithm Arithmetic — All Weights as Powers of Two | 0 | 2 | 2 | 1 | 2 | 7 | REJECT-low-score |

---

## 2. Per-Advancer Rationale

**R7-R-001 ATRC** (Track B, total 12). A1=2: W_Q + W_K at known CF11 compressions (8× and 4×) plus W_V/W_O extension targets 3–5× attention-weight reduction; modeled residency for a 70B model with NVMe complement is plausible. A3=3: W_V SVD spectrum sweep (K ∈ {512,256,128}; 30–40 min) is a clean binary settler — GO sharpens CF11 class boundary inward to V+O; NO-GO extends CF8 outward. A5=3: AQFKV measured W_Q and W_K but no W_V/W_O sweep has been published; the empirical gap is the moat. Load-bearing primitive: spectrum concentration of W_V/W_O (r_99/d < 0.80 vs ≈ 1.0 for MLP). Smallest test: 30-min W_V SVD sweep at K=256 on Qwen3-1.7B; go threshold ΔNLL < 0.50 nats.

**C8-WDOS** (Track B, total 12). A1=2: W_down is 706 MB across 28 layers; if output subspace is substantially E^T-invisible, the W_down quantization-error path is harmless at low precision — at 70B scale, W_down ≈ 37.7 GB and precision reduction to INT2 (if GO) is the largest single compression lever on the table. A3=3: principal angle between colspan(W_down) and colspan(E^T) per layer, ~30 min; GO/NO-GO is fully numerical (angle threshold 60°, fraction-of-layers threshold 0.70). A5=3: CF12 established E's rank structure; no paper has asked whether W_down's output subspace is geometrically invisible to the lm_head projection — this is a coupling of two v2-confirmed findings not on arXiv. Load-bearing primitive: principal angle between W_down output subspace and tied-embed top-k directions. Smallest test: 30-min per-layer W_down SVD (top-128) + CF12 E column projection.

**C10-FRCF** (Track B, total 11). A1=2: if Spearman ρ(F_j, c_E(j)) > 0.15, per-neuron BF16 sidecar for top-5% neurons replaces ~10% of W_up+W_down storage cost and scales to 70B where W_up+W_down ≈ 75 GB. A3=2: ρ measurement is 45 min (2 forward passes + column norms); the binary settler has a quantitative threshold (ρ > 0.15) and NO-GO independently confirms CF1 and CF12 are decoupled across the MLP-to-output path. A5=3: dual CF coupling (CF1 firing rank × CF12 lm_head logit sensitivity) with no published equivalent — both CFs are v2-pipeline outputs. Load-bearing primitive: neuron-level joint firing magnitude × lm_head sensitivity product. Smallest test: 45-min forward pass + ‖E · W_down[:,j]‖_2 per neuron at one representative layer.

**F7-1 WOVR** (Track A, total 13). A1=2: per-head W_{OV} fold at r_eff=32 saves 4× on V+O (336 MB at 1.7B; 6 GB at 70B); combined with CF11 W_Q this makes attention the primary residency target. A2=2: principal-angle coupling between U_O and V_V in a post-training composition has no published treatment; extends AERO's fold logic to the attention path. A4=3: the mechanism is constructive — U_O ≈ V_V is testable in 5 min on a single layer; the fold is an algebraic identity given the precondition. A3=3: 5-min binary settler at layer 14 head 0; angle ≤ 30° → GO; ≥ 60° → class-closed. A5=3: post-training W_{OV} composition rank has no arXiv treatment as of 2026-05-09. Elegant-equivalence class: if U_O ≈ V_V, the fold W_V_h W_O_h is a constructive algebraic identity that eliminates one matmul and reduces storage — `algebraic-identity` sub-class applies. Load-bearing primitive: principal angle alignment between W_O left-singular and W_V right-singular bases per head per layer.

**F7-2 IDSM** (Track B, total 12). A1=1: does not directly reduce weight residency but bounds KV compression analytically; at long context (>32K) KV dominates and the participation-ratio bound is load-bearing. A4=3: participation ratio is a standard covariance calculation; no imported theorem, no precondition to verify — just matrix PCA. A3=3: 20-min binary settler; PR/T² ≤ 0.20 → GO toward KV compression; PR ≥ 0.80 → class-kills attention-score-manifold KV methods. A5=3: the first-principles derivation of a KV compression bound from intrinsic geometry of attention scores rather than engineering heuristics has no named arXiv entry. Load-bearing primitive: participation ratio of attention-score covariance at layer 14.

**F7-4 KSATT** (Track B, total 12, frame-novelty). A2=3: Kronecker structure test on trained attention weights imports from structured matrix decomposition (Monarch/Kaleidoscope territory) but applies it post-training as a structural characterization, not as a training-time constraint — the frame has no named LLM compression cousin. The round-specific saturated-frame list (generic low-rank, generic sparsity, generic quantization) does not contain a "post-training Kronecker test" entry. A4=2: the reshuffled matrix SVD is computable in 3 min; the Frobenius fraction is a checkable number; precondition is just "Kronecker structure exists" which is what the experiment tests. A3=3: 3-min binary settler at layer 14 head 0; ≥ 50% → GO; < 20% → class-closed. A5=3: no paper tests for post-training implicit Kronecker structure in trained LLM attention weights as of 2026-05-09. Path 3 frame-novelty advancer (A2=3, A4=2, A3=3 — structural floor met).

**F7-5 WDCA** (Track B, total 12). A1=1: W_down output subspace finding enables mixed-precision assignment for MLP-inert residual directions; modest direct saving but structural finding extends CF8/CF12 boundary characterization. A4=3: stacked W_down PCA is standard; no imported theorem; 15-min computable. A3=3: var@1600 vs full-rank binary settler; GO/NO-GO each constrains the class of future mixed-precision ideas. A5=3: no paper measures the joint MLP output subspace across all 28 layers as a residual-direction allocation tool. Load-bearing primitive: r_col of stacked W_down column space (joint PCA of all 28 W_down column vectors in R^{2048}).

**R7-R-003 VJOFR** (Track A, convergence representative for C1 — W_{OV} product rank). A1=1: residency gain depends on W_{OV} being low-rank (speculative); if confirmed, 1.9× on V+O at 1.7B, 5× at 70B. A3=3: 15-min W_{OV} rank measurement per layer. Advancing as convergence representative: both R7-R-003 and F7-1 WOVR converge on the W_V + W_O pair as the unexplored attention-weight class; WOVR is the stronger representative (F7-1 total=13, constructive mechanism); VJOFR advances under Path 2 as the Reach framing that explicitly grounds the idea in residency arithmetic.

**R7-R-004 ASOR** (Track A, convergence representative for C2 — outlier channel × W_Q subspace interaction). A1=1: residency gain modest alone; structural interaction claim is the load-bearing piece. A3=3: 15-min projection magnitude test with permutation control. Converges with C7-JDWQ (which correlates Jaccard with W_Q r_99 per layer) and C9-JAQV (which uses Jaccard for per-layer rank budget). Multiple orientations independently arriving at the hypothesis that per-layer Jaccard predicts W_Q structural behavior.

**C7-JDWQ** (Track B, convergence representative for C2b — Jaccard × W_Q per-layer rank correlation). A3=3: Spearman ρ on 28 data points is a 30-min binary settler. A5=3: v2-CF1 (full 28-layer sweep, permutation-controlled) is the empirical anchor not on arXiv. Advancing as convergence representative; JDWQ, JAQV, and ASOR all hinge on whether per-layer Jaccard predicts per-layer W_Q structural behavior.

**C9-JAQV** (Track B, convergence representative for C2b). Extends C7-JDWQ to joint W_Q + W_V budget allocation with anticorrelation hypothesis. A3=2: W_V SVD sweep (40 min) has quantitative threshold. A5=3: v2-CF1 + untested W_V as joint anchor. Converges with C7-JDWQ and ASOR.

**C11-JVSTAB** (Track A, convergence representative for C3 — Jaccard-gated value-vector reuse). A1=1: modest bandwidth saving on stable-J layers; primarily a structural composition claim between v2-CF1 and W_V token-to-token dynamics. A3=2: 40-min ε_V measurement; GO threshold numerical (ε_V < 0.10 for ≥ 60% of consecutive pairs). Converges with F7-2 IDSM (both targeting KV/value vector compressibility from a first-principles bound perspective).

**R7-R-005 QAWKC [FREE SWING]** (Track B, wildcard). Structural floor met: mechanism connects to real stack (SVD on Qwen3-4B), smallest test is 40-60 min, A4=2 (no imported theorem — scaling extrapolation is a hypothesis, not a broken precondition). CF-tether requirement suspended. Advances because the K/n_heads scaling invariant is a genuinely new structural question with no published answer, and if confirmed, directly motivates 70B deployment arithmetic. Advanced as `[FREE SWING]` — structural floor met, CF-tether requirement suspended.

**C12-RUNE [FREE SWING]** (Track A, wildcard). The Δh intrinsic dimension measurement is a new v2 quantity (distinct from CF13–CF15 unconfirmed residual intrinsic dimension). CF-tether requirement suspended. A4=2: PCA of Δh = h_L − h_0 is standard; no broken theorem imports. A3=3: 20-min binary settler (r_99(Δh) ≤ 128). Advanced as `[FREE SWING]` — structural floor met, CF-tether requirement suspended.

**U4-A IOPB** (Track A, frame-novelty). A2=3: using IOCP `NumberOfConcurrentThreads` as a per-head bandwidth governor for attention weight streaming is a substrate primitive invocation that has no named LLM inference cousin — the frame is genuinely outside ML compression literature. A4=2: the mechanism is checkable (60-line C harness); no imported theorem. A3=2: 3-hour experiment; ≥ 5% speedup is a quantitative go/no-go with structural NO-GO finding (NVMe QD=1 documentation). Path 3 frame-novelty advancer (A2=3, A4=2, A3=2 — structural floor met; total 11 but advances under Path 3).

**A2-PERM** (Track B, elegant-equivalence). A2=2: permutation-group gauge exploitation on attention heads is in-vocabulary (head importance ranking is known) but the specific permutation-alignment → structured mixed-precision codebook assignment is non-obvious. A4=2: W_Q row norm computation is 10 min; precondition (head norms differ by > 2×) is checkable. A3=2: go threshold: ratio > 2.0 across 28 layers. Eligible Path 4 because the mechanism class is "gauge-exploitation" (using the S_{16} × S_{128} permutation symmetry — the only gauge freedom that commutes with RMSNorm — to force structured codebook alignment without SGD). Sub-class: `gauge-exploitation`. A4 bonus: +1 for constructive (auditable in one paragraph). Total effective A4=3 under Path 4 weight.

**A6-TBLK [FREE SWING]** (Track A, wildcard). A2=3: inverting neural inference to offline table construction + online B-tree NVMe retrieval is a frame with no LLM compression cousin. A4=2: mechanism connects to real stack primitives (BLAKE3, B-tree, NVMe binary search). A3=3: 30-min 4-gram coverage measurement. Structural floor met. Advanced as `[FREE SWING]` — CF-tether requirement suspended.

---

## 3. Convergence Map

**Convergence C1 — W_V and W_O spectral structure as the last untested attention-weight class.**
R7-R-001 ATRC (Reach, full cascade), R7-R-003 VJOFR (Reach, product-rank angle), F7-1 WOVR (First-Principles, per-head spectral alignment), C9-JAQV (Composition, W_V rank budget), C11-JVSTAB (Composition, W_V token-stability reuse) all converge on W_V and W_O as the single open question in CF11's attention-weight class characterization. Three distinct orientations (R, F, C) independently propose experiments that require W_V SVD spectrum as their first rung. Strongest representative: F7-1 WOVR (A4=3, constructive algebraic-identity mechanism, 5-min settler). Cheapest settler: W_V and W_O SVD sweep on Qwen3-1.7B, all 28 layers, K ∈ {512, 256, 128}; measure r_99 and ΔNLL (approx. 35–45 min). This is now the highest-leverage single pre-program experiment for Stage 5: one run resolves ATRC, VJOFR, WOVR, JAQV, and JVSTAB simultaneously.

**Convergence C2 — Per-layer Jaccard from v2-CF1 as a predictor of per-layer W_Q structural behavior.**
R7-R-004 ASOR (Reach, outlier-channel × W_Q rowspan alignment), C7-JDWQ (Composition, Jaccard × W_Q r_99 Spearman correlation), C9-JAQV (Composition, Jaccard × W_Q+W_V joint budget anticorrelation) all converge on: the per-layer Jaccard from v2-CF1 should correlate with something in the W_Q weight structure. The shared primitive is "layer-level outlier-channel dynamics as a proxy for layer-level query complexity." Strongest representative: C7-JDWQ (cleanest equation, 30-min Spearman ρ settler). Cheapest settler: per-layer W_Q SVD for all 28 layers (top-256 singular values; ~30 min) → Spearman ρ(J_ℓ, r_99(W_Q^ℓ)/d) with permutation control. This settler also partially resolves ASOR and JAQV.

**Convergence C3 — W_down output subspace as a compression-relevant structural object.**
C8-WDOS (Composition, W_down output-space disjointness from tied-embed span), F7-5 WDCA (First-Principles, joint MLP output subspace PCA), R7-R-006 LGOP (partially; gate folding implications for W_down-inert directions) converge on W_down column space as unmeasured territory. C8-WDOS and F7-5 WDCA ask complementary questions: WDOS asks whether W_down's output avoids the high-sensitivity lm_head directions (compression safety); WDCA asks whether W_down's outputs collectively span all of R^{d_model} (subspace dimensionality). Both experiments are ~30 min each and together fully characterize W_down's output geometry. Strongest representatives: C8-WDOS (A1=2, direct compression payoff) and F7-5 WDCA (A4=3, constructive). Stage 3 should pressure-test whether the WDOS disjointness claim survives when E^T is nearly isotropic (CF12 r_99=1992 ≈ full rank).

---

## 4. Frame-Novelty Bonus Advancers

**Slot 1: F7-4 KSATT** (A2=3). Frame: post-training structural characterization of trained transformer attention weights via Kronecker product decomposition. Nearest cousin: Monarch matrices (arXiv:2204.00595), Kaleidoscope (arXiv:2112.00029) — both are training-time structured-matrix approaches. KSATT differs from these by asking whether Kronecker structure *already exists* in trained weights as a consequence of SGD dynamics, not as a training-time constraint. The published saturation list (generic low-rank, generic sparsity, generic quantization, MLA-style joint Q-K) has no "post-training implicit Kronecker test" entry. Frame novelty is genuine because the question is structural-characterization (does SGD converge to Kronecker?) rather than compression-method-proposal — a different intellectual object.

**Slot 2: U4-A IOPB** (A2=3). Frame: IOCP `NumberOfConcurrentThreads` as an attention-head bandwidth governor — a Windows OS substrate primitive used as a computation-graph regulator for multi-head inference streaming. Nearest cousin from prior U runs: GTBP (run_004) used IOCP for sequential layer prefetch. IOPB uses the *throttle parameter* as a bandwidth governor for head-level parallelism — the throttle as a first-class inference primitive. No ML inference paper has used OS I/O concurrency controls to regulate model-internal attention-head parallelism. Frame is past the saturation boundary.

---

## 5. Rejected with Rationale

- **R7-R-006 LGOP** (total 9): A1=1 (layer-1 gate fold saves ~9 MB — negligible at scale; the composition with outlier-pinning adds uncertainty without residency leverage). A2=1: SDZC was already killed as a global scheme; LGOP is that same idea restricted to layer 1, which is acknowledged in the SDZC kill entry as not actionable. The gate-fold is already captured by F7-6 LGFA as a FREE SWING; LGOP's composition claim is redundant. Kill-list-style one-liner: *LGOP — gate near-constancy cascade — layer-1 SDZC extension; composition claim adds no residency vs single-layer LGFA; rejected as redundant with LGFA and insufficient A1.*

- **R7-R-007 AANF** (total 7): A2=1 (sequential GGUF extent layout + OS prefetch is the OS-Paging-Aware Weight Layout idea killed at R1/S2 — "doesn't reduce bpw, just engineers paging"). The "interaction claim" (CF11 attention compression relieves prefetch contention) is speculative and the contention relief hypothesis has no measurement behind it. A4=1: the prefetch engagement claim requires CF13* re-derivation, and without that, the mechanism is ungrounded. Kill-list-style: *AANF — NVMe sequential layout + OS prefetch — structurally adjacent to R1/S2 kill (OS-Paging-Aware Weight Layout); prefetch-contention interaction claim is unverified speculation; rejected.*

- **F7-6 LGFA [FREE SWING]** (total 9): A1=0 for a wildcard. The mechanism is structurally correct and layer-1 fold is algebraically valid, but residency payoff is 0.25% of model — below any threshold of practical relevance even as part of a cascade. The gate-fold idea has been considered and explicitly deferred (LCAB in Track A R2 Stage 4) with the same conclusion. As a FREE SWING the CF-tether requirement is suspended but the A1=0 AND total=9 does not clear the Path 1 threshold. Kill-list-style: *LGFA — layer-1 gate fold algebraic identity — 8.7 MB saving on 1.7B model; structural finding already captured in CF6 SDZC; residency payoff below actionable threshold.*

- **U1-B RCWO** (total 9): A3=2 (the smallest experiment requires ik_llama.cpp patching; ≥ 2-hour implementation before go/no-go; does not clearly meet ≤ 8-hour criterion for A3=3). A1=1: 21.5% NVMe read volume reduction; doesn't change weight count or precision. The scale-stratum compressibility is plausible but the DRAM headroom argument is tight and depends on CF13* re-derivation. Kill-list-style: *RCWO — CreateFileMapping scale-tier DRAM split — 21.5% NVMe read reduction; implementation cost 2–3 hours; scale stratum compressibility unverified; marginal A1, conditional on CF13*.*

- **U2-B FSCW** (total 9): A3=2 (2.5-hour implementation; go/no-go is 5% speedup on a stall that may not exist). A2=2 but mechanism is engineering rather than frame-novel. The proposal does not reduce residency; it reduces potential Cache Manager stalls. At 1.7B the stalls may not exist (DRAM is not fully loaded at 1.7B Q4). Kill-list-style: *FSCW — per-layer UnmapViewOfFile cache management — structural finding is whether reactive eviction is already sufficient; useful measurement but not a primary residency lever.*

- **U3-B TINA** (total 8): A1=0 at 1.7B (L3 conflict-eviction saving ≤ 2 ms/token against 160 ms/token baseline); A4=1 (virtual-address-based padding prediction requires physical address bit visibility that user-space doesn't have). The PMU measurement can still be done but the mechanism is speculative without physical address access. Kill-list-style: *TINA — Zen 3 L3 set-conflict padding — estimated <2 ms/token gain at 1.7B; physical address visibility unavailable in user space; speculative mechanism on correct hardware unit.*

- **U5-B WMCW [FREE SWING]** (total 9): A1=1 (Q4_K_M nibbles are near-incompressible under LZ77; VMC compression ratio ≈ 1.05 for nibbles; scale stratum gain ~350 MB). The mechanism depends on VMC engagement which is automatic and uncontrollable — the VirtualUnlock → VMC path is probabilistic. NOT confirmed to improve throughput vs pagefile eviction. Kill-list-style: *WMCW — Windows Memory Compression cold-tier — nibble payload near-incompressible; VMC engagement uncontrollable (may evict to pagefile not compressed store); structural floor marginal.*

- **A1-PCAG** (total 9): A1=0 (the mechanism saves 38.8 µs/token in residual-stream bandwidth against 160 ms/token total — sub-threshold). The correctness argument is clean but there is no residency payoff and no tok/s payoff at scale. Kill-list-style: *PCAG — int32 residual accumulator in L2 — 38.8 µs/token bandwidth saving (< 0.03% of total); negligible residency impact; L2 locality argument correct but not a residency lever.*

- **A3-CADDR** (total 10, rejected despite score): A4=1 (the content-addressed skip of layers assumes that int8-quantized residual states will collision-hash with meaningful frequency; the primary risk is near-zero hit rate in autoregressive decode — the ideator acknowledges "residual states encode full positional + causal context, likely unique per token." A3=3 but the go/no-go condition may structurally never fire. Without a confirmed > 2% hit rate in autoregressive generation, the mechanism is a null hypothesis. Does not meet advancement: A4=1 (mechanism specificity fails — the hash-collision premise is not checkable in advance without the domain-specificity caveat making the claim conditional). Kill-list-style: *CADDR — content-addressed residual hash layer-skip — hit rate near-zero for autoregressive decode (residual states are causally unique); domain-specific (repetitive template) use case too narrow; A4=1 due to unverified hash-collision premise.*

- **A4-MONO** (total 8): A1=0 (trained Qwen3 weights are ~50% negative by construction; zeroing negative weights is catastrophic). A4=1 (the monotone-circuit projection is not a lossless transformation; the experiment concedes ΔNLL collapse is expected). Kill-list-style: *MONO — monotone circuit projection — 50% weight zeroing catastrophic; useful sign-density structural probe but not a compression mechanism.*

- **A5-RDST** (total 9): A1=0 (no residency gain unless attention is invertible; the proposal concedes "transformers are emphatically not normalizing flows"). A4=1 (SwiGLU gates are non-injective; the reversibility claim requires Bennett-style padding that eliminates the storage savings). Kill-list-style: *RDST — Bennett reversible decode stack — transformers are not invertible (SwiGLU non-injective); reconstruction error expected to be high; no residency payoff without demonstrated invertibility.*

- **A7-INTLOG** (total 7): A1=0 (log-quantization at nearest power of two is cruder than INT4 for weights clustered near zero; ΔNLL collapse expected; storage is 8 bpw = INT8, worse than INT4 at 4 bpw). A4=1 (the quantization grid is provably suboptimal for non-log-uniform weight distributions). Kill-list-style: *INTLOG — integer-logarithm weight quantization — 8 bpw worse than INT4; nearest-power-of-two grid catastrophically coarse near zero for trained weights; useful structural probe but not a viable compression scheme.*

---

## 6. Hand-off to Stage 3

**Track A advancers (4 + convergence reps):**

1. **F7-1 WOVR** (First-Principles, A, 13) — Pressure-test: does the left-singular/right-singular alignment between W_O and W_V survive across all 28 layers and multiple heads, or is the alignment confined to a few heads in a few layers? Stage 3 should check whether any attention-circuit paper (Elhage et al. circuits work, A3 arXiv:2505.12942) has accidentally measured W_{OV} product rank.

2. **R7-R-001 ATRC** (Reach, B/A cascade, 12) — Pressure-test: is W_V's spectrum class closer to W_K (r_99/d ≈ 0.79) or MLP (r_99/d ≈ 1.0)? Stage 3 should check whether any paper has measured W_V/W_O spectral concentration post-training on Qwen3-family or similar models. The residency arithmetic depends on W_V being in the CF11 class.

3. **C8-WDOS** (Composition, B, 12) — Pressure-test: CF12 established E is nearly isotropic (r_99=1992/2048). If E^T has no dominant directions, the "E^T-invisible subspace" for W_down is essentially zero — the WDOS coupling may be vacuous. Stage 3 must verify whether the near-isotropic E^T materially undermines the disjointness payoff before this is selected for experiment.

4. **F7-5 WDCA** (First-Principles, B, 12) — Pressure-test: is cross-layer W_down column-space stacking distinguishable from the v2-CHEAP-TEST-001 pattern (where cross-layer W_Q stacked SVD gap was 0.0001 vs 0.072)? Stage 3 should apply the same skeptic-controls discipline: run a matched-spectrum random-orientation baseline on the stacked W_down column PCA to detect whether the apparent subspace structure is real learned structure or a trivial summation artifact.

5. **F7-4 KSATT** (frame-novelty slot 1, B, 12) — Pressure-test: are there any published tests of implicit Kronecker structure in trained transformer weights? Stage 3 should search for "Kronecker product test transformer weights" and "post-training structured matrix decomposition" in 2024–2026 arXiv. Also check whether the partition sensitivity (3 partition pairs tested) is sufficient to declare structure absent if all three fail.

**Track B additional advancers:**

6. **C10-FRCF** (Composition, B, 11) — Pressure-test: the c_E(j) = ‖E · W_down[:,j]‖_2 computation assumes E is loaded into RAM simultaneously with W_down per layer; on 7.28 GiB, this is tight. Stage 3 should verify memory feasibility and check whether SpQR (per-weight Hessian sensitivity) already subsumes the per-neuron dual-signal idea.

7. **U4-A IOPB** (frame-novelty slot 2, A, 11) — Pressure-test: does Windows IOCP `NumberOfConcurrentThreads` actually throttle I/O completions vs just limiting thread wakeups? The throttle semantics may differ from the bandwidth-governor claim. Stage 3 should check Windows Internals documentation and any published measurements of IOCP completion-throttling behavior under NVMe I/O workloads.

**Wildcard advancers (CF-tether suspended):**

8. **R7-R-005 QAWKC [FREE SWING]** (B, 11) — Question: does the K/n_heads ≈ 8 head-redundancy invariant hold across model scales? Cheapest test: W_Q sweep on Qwen3-4B.

9. **C12-RUNE [FREE SWING]** (A, 11) — Question: is r_99(Δh = h_L − h_0) ≤ 128? Cheapest test: 20-min PCA on Δh across 200 calibration tokens.

10. **A6-TBLK [FREE SWING]** (A, 11) — Question: is 4-gram table coverage > 20% on a narrow domain? Cheapest test: 30-min Python dict coverage check on WikiText-103 4-grams.

**Elegant-equivalence advancer:**

11. **A2-PERM** (gauge-exploitation, B, 10+) — Question: does the head norm ratio (top-4 / bottom-4) exceed 2.0 across layers? Cheapest test: 10-min W_Q per-head row-norm computation. Stage 3 should verify that calibration-based head importance ranking (not just weight norm) is not already addressed by SpAtten, A3, or any head-pruning paper that uses permutation-consistent precision assignment.

**Convergence pre-program experiments (highest leverage for Stage 5):**
- **Settle C1 first** (W_V / W_O SVD sweep, ~40 min): resolves ATRC, VJOFR, WOVR, JAQV, JVSTAB simultaneously.
- **Settle C2 second** (per-layer W_Q SVD all 28 layers, ~30 min): resolves JDWQ, JAQV correlation test, ASOR alignment test simultaneously.
- **Settle C3 third** (W_down SVD + principal angle to E^T top-128, ~30 min): resolves WDOS and WDCA complementarily.
