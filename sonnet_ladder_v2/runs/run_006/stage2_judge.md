# Stage 2 Judge — Run 006

Independent scoring. Kill-list enforced: v2-CHEAP-TEST-001 (cross-layer W_Q class), v2-S3-R004-001 (arbitrary O(d) rotation breaks RMSNorm γ). CF9 hard gate applied. No idea below re-treads either killed class.

---

## 1. Union Score Table

| ID | Orientation | Track | Title | A1 | A2 | A3 | A4 | A5 | Total | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| C-JOINT-DELNLL | C | A | Joint W_Q+W_K Truncation Interaction Coefficient | 2 | 2 | 3 | 3 | 3 | 13 | ADVANCE |
| R6-WVOP | R | A | W_V/W_O Product Rank Cascade | 2 | 2 | 3 | 2 | 3 | 12 | ADVANCE |
| F6-WALIGN | F | A | W_up/W_gate Row Alignment Fraction | 1 | 2 | 3 | 3 | 3 | 12 | ADVANCE-elegant-equivalence [algebraic-identity] |
| C-WVOK | C | B | W_V/W_O Rank Ordering Composes into Attention Budget | 2 | 2 | 3 | 2 | 3 | 12 | ADVANCE |
| F6-GQARED | F | A | GQA Head-Group Importance Asymmetry | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE |
| F6-SINKCONST | F | A | Attention Sink Key-Vector Constancy [FREE SWING] | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE-frame-novelty |
| A1-PEER | A | A | Peer-Swap Inference: Model Runs a Peer's Weights | 2 | 3 | 2 | 2 | 2 | 11 | ADVANCE-frame-novelty |
| A6-LLMR | A | A | Log-Likelihood Monotone Residual Early-Exit | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE-convergence |
| R6-ATTNQ-FULL | R | B | Joint 4-Matrix Attention Compression 70B Cascade | 3 | 1 | 3 | 2 | 2 | 11 | ADVANCE |
| U2-FNDP | U | B | FILE_FLAG_NO_BUFFERING Direct-I/O Weight Read Path | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE |
| U3-WMCE | U | B | Windows Memory Compression Store Exploitation | 2 | 3 | 2 | 1 | 3 | 11 | ADVANCE |
| R6-WDOWN-CONDP | R | A | W_down Conditional-Precision Activation Router [FREE SWING] | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE-elegant-equivalence [algebraic-identity] |
| C-DEEP-SPREAD-QBUD | C | B | Deep-Layer Outlier Spread Implies Per-Layer W_Q Rank Schedule | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE-convergence |
| F6-DHSUB | F | B | Residual-Stream Update Subspace Concentration | 1 | 2 | 2 | 2 | 3 | 10 | REJECT-low-score |
| C-KSTAT-QDYN | C | B | Static K=0.1% Outlier Channel Alignment vs W_Q Singulars | 1 | 2 | 3 | 2 | 2 | 10 | REJECT-low-score |
| U6-CSDL | U | B | CPU Idle-State Demotion Lock [FREE SWING] | 1 | 2 | 2 | 2 | 3 | 10 | REJECT-low-score |
| F6-WDCOL | F | B | W_down Activation-Covariance Weighted Column Subspace | 2 | 2 | 2 | 2 | 2 | 10 | REJECT-low-score |
| R6-MLA-POSTRAIN | R | A | Post-Training MLA Conversion via CF11 Head Redundancy | 2 | 2 | 3 | 2 | 1 | 10 | REJECT-low-score |
| A2-O1TK | A | A | O(1)-Token Parallel Decode [FREE SWING] | 2 | 3 | 2 | 1 | 2 | 10 | REJECT-low-score |
| R6-DSAF-70B | R | B | Depth-Spread Attention-Rank Schedule 70B | 1 | 2 | 2 | 2 | 2 | 9 | REJECT-low-score |
| U4-RFSW | U | B | ReadFileScatter Aligned Gather-Read | 1 | 2 | 2 | 2 | 2 | 9 | REJECT-low-score |
| U5-SCVP | U | B | CF3-Grounded Static-Channel VirtualLock Pin | 1 | 2 | 2 | 2 | 2 | 9 | REJECT-low-score |
| R6-RAOK-CF3EXT | R | B | RAOK CF3-Extended Depth-Stratified Activation Codebook | 1 | 1 | 2 | 2 | 2 | 8 | REJECT-low-score |
| U1-IOPR | U | B | I/O Priority Escalation via SetFilePriorityHint | 1 | 1 | 2 | 2 | 2 | 8 | REJECT-low-score |
| C-L1-ATTEN-GATE | C | A | Layer-1 Anomaly: Gate Fold + Tighter Attention [FREE SWING] | 1 | 1 | 2 | 2 | 2 | 8 | REJECT-low-score |
| A3-DBGB | A | B | Debuggable-Bytes INT8 Byte-Aligned Partial Row Loading | 1 | 2 | 2 | 1 | 2 | 8 | REJECT-low-score |
| A4-REVR | A | A | Reversible-Computation Residual Stream Zero-Copy Inversion | 1 | 2 | 2 | 1 | 2 | 8 | REJECT-low-score |
| A5-VECD | A | A | Vector-Clock Residual Consistency Attention-MLP Pipeline | 1 | 2 | 2 | 1 | 2 | 8 | REJECT-low-score |

---

## 2. Per-Advancer Rationale

**C-JOINT-DELNLL** (Track A, total 13): Carried by A3=3 and A4=3 — the coupling math is fully auditable in five lines (cross-term ΔW_Q ΔW_K^T is second-order; the prediction from subspace overlap precedes the joint ΔNLL measurement). Grounds CF11 bilaterally. Smallest test: subspace_overlap(V_Q^512, V_K^512) at L14 in ~2 min, then 10-min joint truncation ΔNLL. NO-GO produces a structural constraint on joint compression budgets — a class-level finding. This is the highest-value Track A measurement available right now.

**R6-WVOP** (Track A, total 12): AERO-style fold applied to the attention V-O chain — W_VO = W_V @ W_O computed once at load time. Load-bearing primitive: product rank rank(W_VO) ≤ min(rank(W_V), rank(W_O)). CF11's classification (attention matrices more compressible than MLP) grounds the plausibility; W_V and W_O are explicitly flagged as unmeasured in SUMMARY.md. Smallest test: SVD of W_V[14] @ W_O[14], r_99/d < 0.70 AND ΔNLL(K=512) < 0.80 nats in ~25 min. NO-GO resolves the W_O full-rank question and extends the CF8/CF11 boundary.

**F6-WALIGN** (Track A, total 12, **Path 4 elegant-equivalence, sub-class: algebraic-identity**): Exact algebraic identity — for neurons where cos(W_gate[i,:], W_up[i,:]) ≈ 1, the SwiGLU output collapses to a scalar function of one projection; W_up row is replaced by a scalar. No approximation for perfectly aligned neurons; the construction is one paragraph. Smallest test: row-wise cosine histogram across 172K neurons, ≥5% with |cos|≥0.90, in ~5 min pure tensor arithmetic. A4=3 because the construction is auditable in 10 lines of NumPy. The distribution shape is a structural finding regardless of outcome. Path 4 tag: the elegance is constructive and fresh relative to AERO (which merges matrices globally after activation removal); WALIGN folds individual neuron pairs without changing any global computation graph.

**C-WVOK** (Track B, total 12): Coupling equation: if W_Q's 16 heads collapse to a ~128-dim joint subspace (CF11), value routing through the same attention heads must carry redundant information — dim(effective_value_subspace) ≤ 128. Testable: r_99(W_V)/d ≤ 0.75 at L14, ΔNLL(K=512) < 0.40 nats in ~15 min. NO-GO sharpens CF11: head-redundancy is a query-side phenomenon, value projection decouples from it. A5=3 because the coupling equation is not on arXiv for Qwen3 and uses private CF11 numbers as its premise.

**F6-GQARED** (Track A, total 11): GQA's discrete outer-sum over 8 KV groups is an algebraic structure that makes group-dropping exact (not approximate). Measurement: per-KV-group mean attention weight on 200-token corpus; any group with mean < 0.05 is a candidate drop. Smallest test: ~15 min forward pass with attention weight logging. NO-GO (all groups equipotent) is a training-dynamics constraint finding. Distinct from Michel et al. 2019 (MHA head pruning) because GQA's group structure creates a true discrete factorization.

**F6-SINKCONST** (Track A, total 11, **Path 3 frame-novelty**, [FREE SWING]): Frame: first-principles algebraic explanation of attention sinks via W_K·e_BOS alignment with top W_Q singular vector. k_sink = W_K^g @ e_BOS is a fixed vector precomputed at model load (no forward pass); test against top-right singular vectors of W_Q per KV group. Runtime: ~10 min pure weight arithmetic. A2=3 because the sink explanation frame (spectral alignment between W_K @ embedding and W_Q) has no analog in StreamingLLM / H2O / SnapKV, which identify sinks empirically and offer no algebraic origin. CF-tether requirement suspended ([FREE SWING]).

**A1-PEER** (Track A, total 11, **Path 3 frame-novelty**): Frame: inference entirely in a peer (smaller) model's weight space — the trained model's own weights are inadmissible at inference time. Forces derivation of what a "peer-based inference" mechanism means, landing on a calibration-fit rank-64 projector P. Distinct from WDLA (which learned a residual correction on top of A's weights; died from CF10 conditioning); this is CF10-compliant by construction (rank-64 affine, 196K params, 10K-token calibration). Smallest test: R²_eval at mid-depth layer 14 in ~3h. A2=3 because "the model cannot hold its own weights" is a constraint-alien frame the published compression literature has no vocabulary for; closest are model stitching (training-time) and distillation (training-time).

**A6-LLMR** (Track A, total 11, **convergence representative C3**): Hard-monotone confidence constraint as early-exit policy — fires at the first layer where per-token entropy increases. No learned threshold; no precomputed compression valley (contrast CVEEKS). Smallest test: 30-min per-layer entropy profile on 200 tokens; ≥60% non-monotone is a class-level settlement. Advancing as convergence representative for the "per-layer lm_head entropy profile" cluster (the measurement itself is high-leverage for multiple future ideas across orientations). A3=3 because the NO-GO outcome (entropy is naturally monotone) is itself a novel structural finding about Qwen3 inference dynamics.

**R6-ATTNQ-FULL** (Track B, total 11): Full-cascade argument for DRAM-resident 70B (A1=3) via joint 4-matrix attention SVD compression. Contingent on W_V/W_O spectrum gates, which are gated by the WVOP/C-WVOK experiments already advancing. Smallest test: W_V[14] + W_O[14] SVD + ΔNLL in ~20 min. A4=2 because preconditions are clearly stated (requires W_V/W_O to have r_99/d < 0.85). The cascade inherits its structural grounding from CF11.

**U2-FNDP** (Track B, total 11): FILE_FLAG_NO_BUFFERING bypasses Cache Manager double-copy — NVMe DMA directly to aligned user-mode buffer. Arithmetic: at 11.5 GB/s DRAM bandwidth and 437 MB/layer, eliminating the kernel copy saves 38 ms/layer → ~20% layer-time reduction on NVMe-resident 70B. Smallest test: ~1h standalone C harness, ≥15% throughput improvement on 200 MB sequential read. NO-GO establishes a confirmed NVMe sequential throughput number (v2-CF13'' candidate). Grounded in documented Win32 behavior; no theorem imports.

**U3-WMCE** (Track B, total 11, advancing but flagged): Windows DRAM Compression Store as an intermediate tier between DRAM and NVMe — cold weight pages compressed in-DRAM rather than paged to NVMe, extending effective DRAM capacity ~1.5–2.5×. A2=3 (no LLM paper exploits this tier; Apple LLM-in-a-flash uses flash, not a DRAM compression tier). A4=1 because the key precondition — whether Windows Compression Store engages for file-backed mmap pages vs only anonymous pages — is explicitly unresolved and called out as a primary risk. Stage 3 must verify this OS behavior first; if mmap pages are ineligible, the mechanism requires switching to anonymous-memory loading (which also enables FNDP). Advancing because A4≥1 and A5=3 frame-novelty justifies Stage 3 investigation, but this is the weakest Path 1 advancer.

**R6-WDOWN-CONDP** (Track A, total 9, **Path 4 elegant-equivalence, sub-class: algebraic-identity**, [FREE SWING]): Exact algebraic split: y = W_down·a = W_down[:,H]·a_H + W_down[:,C]·a_C — no approximation in the partition. Precision asymmetry (INT8 on H, INT2 on C) at equal residency exploits CF1's firing-rank dominance not as a predictor but as a post-hoc partition key. Smallest test: 30-min single-layer ΔNLL comparison, INT8/INT2 equal-residency vs uniform INT4. The split identity is the elegant-equivalence structure; the quality question is whether INT2 tail error overwhelms the INT8 head gain. CF-tether requirement suspended ([FREE SWING]).

**C-DEEP-SPREAD-QBUD** (Track B, total 9, **convergence representative C1**): Coupling: per-layer channel_coverage_90% (v2-CF1) as predictor of W_Q rank sensitivity, enabling a depth-stratified per-layer rank schedule. Converges with R6-DSAF-70B (Orientation R), which independently derives the same schedule from the same v2-CF1 depth profile using identical coupling logic. Smallest test: 35-min Spearman ρ across 5 probe layers. Advancing as convergence representative because independent derivation of the same scheduling primitive across two orientations is the pipeline's strongest signal. Note: this is a quality-at-fixed-bytes improvement, not a standalone residency lever; its value is as a free improvement layered on top of the W_Q compression advances.

---

## 3. Convergence Map

**Convergence C1 — Per-layer outlier activation spread as W_Q rank-schedule predictor.**
Reach R6-DSAF-70B and Composition C-DEEP-SPREAD-QBUD independently couple v2-CF1's depth-varying outlier spread (channel_coverage_90% per layer, range 5–8% early layers vs 14–18% deep layers) to the appropriate W_Q rank budget per layer. Both arrive at an identical scheduling logic: lower K_Q for shallow layers, higher K_Q for deep layers, at zero residency cost vs uniform K=128. Strongest rep: C-DEEP-SPREAD-QBUD (more explicit coupling equation, Spearman ρ as cleaner go/no-go). Cheapest settler: extend v2-CF1 per-layer measurement to compute per-layer W_Q rank sensitivity at K=64/128/256 for 5 probe layers and compare Spearman ρ (~35 min). If ρ ≥ 0.70, the outlier-spread → rank-budget coupling is a reusable scheduling primitive for the full attention compression cascade.

**Convergence C2 — W_V/W_O spectrum as the gate to full attention residency target.**
Reach R6-WVOP (AERO fold on V-O chain), Reach R6-ATTNQ-FULL (joint 4-matrix SVD cascade), and Composition C-WVOK (CF11 head-redundancy → value-subspace coupling) all independently converge on the same measurement gate: r_99(W_V)/d and r_99(W_O)/d. All three proposals are contingent on this spectrum measurement; its result determines whether the 70B attention compression target is achievable. Three different orientations arriving at the same gate — W_V and W_O spectra — is the strongest residency-relevant convergence in this run. Cheapest settler: SVD of W_V[14] and W_O[14] in ~20 min total; this single measurement gates three proposals simultaneously and should be the next experiment queued.

**Convergence C3 — Per-layer lm_head entropy profile as routing oracle.**
A6-LLMR (Constraint-Alien) proposes measuring per-layer entropy of lm_head(h_L) as an early-exit monotone gate. No other orientation independently derived this measurement in run 006, but LLMR's measurement (200-token per-layer entropy profile, ~30 min) is a high-leverage output: it produces the data needed to validate any per-layer scheduling idea (depth-stratified rank schedule, layer-selective computation, early-exit confidence gates). Flagging as a convergence handle for Stage 4 rather than a full cross-orientation convergence cluster this round.

---

## 4. Frame-Novelty Bonus Advancers

**F6-SINKCONST [FREE SWING]** (A2=3, A4=2): Frame — algebraic derivation of attention sink origins from W_K @ embedding alignment with W_Q principal directions. The published sink literature (StreamingLLM, H2O, SnapKV, Attention Sink survey) identifies the phenomenon empirically and proposes eviction policies. None derives the sink from a static spectral relationship between W_K, W_Q, and the token embedding. The coupling (k_sink = W_K · e_BOS aligns with top W_Q singular vector) is a first-principles prediction that has no published cousin as of 2026-05-09. If confirmed, this is not just a structural finding — it provides an algebraic explanation for why sinks are stable across tokens (they are driven by fixed weight geometry, not by dynamic attention patterns). Frame-novelty is the point; compression payoff is secondary.

**A1-PEER** (A2=3, A4=2): Frame — the constraint "the model cannot hold its own weights" forces inference to run entirely in a peer model's weight coordinate system. This is outside the standard ML compression vocabulary (which always operates on the target model's own weights). The closest published frames (model stitching, distillation, knowledge transfer) are all training-time operations that produce a modified model. A1-PEER is inference-time peer-swap with no model surgery, grounded only in a calibration-fit projector. The R² measurement at mid-depth is the structural settle: if cross-scale hidden-state alignment is achievable post-training, the peer-swap frame opens a class of no-retraining cross-scale inference schemes that the literature has not explored. CF-tether not required (constraint-alien orientation); frame-novelty and structural floor are sufficient.

---

## 5. Rejected with Rationale

- **F6-DHSUB** (total 10, REJECT-low-score): Track B has 4 stronger advancers. ΔH_L effective rank is a genuine measurement gap, but A1=1 (residency effect is marginal — 5 µs/token activation bandwidth saving) and A5=3 don't overcome the lower ceiling. Surviving primitive: per-layer residual update subspace dimension; Stage 4 may want to couple this with RAOK's tier assignment if v2-CF1's outlier structure turns out to live in the same subspace as ΔH_L.

- **C-KSTAT-QDYN** (total 10, REJECT-low-score): A1=1 (operational simplification of RAOK tier-1, not a residency lever). The measurement itself (σ_static, ~5 min) is cheap and should be run as a Stage 5 pre-program experiment alongside RAOK deployment rather than as an independent pipeline investment. Add to RAOK instrumentation.

- **U6-CSDL** (total 10, REJECT-low-score): A1=1 acknowledged (FREE SWING); gain is sub-5% and only on DRAM-resident 1.7B. Surviving primitive: CPU C-state behavior during LLM inference decode on Ryzen 5 7530U is unmeasured; a powercfg C-state histogram would resolve whether CC6 is even reached during 100 µs inter-layer windows. Worth a one-line diagnostic; not pipeline investment.

- **F6-WDCOL** (total 10, REJECT-low-score): Strong mechanistic argument (activation-covariance-weighted right projection of W_down vs weight SVD) but Track B top-4 already occupied by higher-total ideas. The mechanism is CF10-adjacent — the go/no-go requires an OOD guard, and the distribution-specificity risk is real. Stage 4 may revisit if RAOK (the Track B frontier) fails, since WDCOL's K=128 savings on W_down are the most aggressive column-compression claim in this run.

- **R6-MLA-POSTRAIN** (total 10, REJECT-low-score): A5=1 because TransMLA (arXiv:2506.01189) and PALU (arXiv:2407.21118) are direct cousins of this proposal. The CF11-anchored rank selection (K=128 from the head-redundancy measurement) is the differentiator, but Stage 3 is likely to find it insufficient to establish non-replication. The experiment (30-min single-layer factorization ΔNLL vs SVD-only) is cheap and could be run as a Stage 5 supplement to AQFKV-follow-on work rather than as a main pipeline investment.

- **A2-O1TK** (total 10, REJECT-low-score): A4=1 — the quality degradation from removing the causal mask on a causal-trained model is acknowledged as "expected to be 3–10 nats" but is not derived from any mechanism; it is a gestured claim. The proposal's load-bearing quality prediction is hand-waved. After removing the ungrounded quality claim, what remains is a throughput argument for single-pass bidirectional inference on a causal model, which is an engineering observation (weight fetches amortized over L tokens) without a path to usable quality. Surviving primitive: amortized single-pass weight loading as a bandwidth multiplier; Stage 4 may want to test this with a model that was actually trained bidirectionally (masked LM baseline) to isolate the throughput claim from the quality claim.

- **R6-DSAF-70B** (total 9, REJECT-low-score): Subsumed by the convergence cluster C1; C-DEEP-SPREAD-QBUD is the stronger representative. Stage 3 carries the convergence note.

- **R6-RAOK-CF3EXT** (total 8, REJECT-low-score): Depends on RAOK base experiment which has not yet run. Premature as a pipeline investment; schedule as a post-RAOK refinement.

- **U4-RFSW** (total 9, REJECT-low-score): A1=1 (6% NVMe command overhead reduction); the gain may be subsumed by FNDP (U2) which composes with scatter-gather naturally. Stage 4 may package RFSW + FNDP as a combined NVMe path overhaul.

- **U5-SCVP** (total 9, REJECT-low-score): A1=1 (392 KB pinning, negligible absolute saving). The CF3-grounded pinning oracle is elegant and should be implemented as part of RAOK deployment (pin the static T1 channels in VirtualLock for zero inference cost), not as a standalone pipeline investment.

- **U1-IOPR** (total 8, REJECT-low-score): Latency variance reduction is real but not measurable as a residency improvement (A1=1). Surviving primitive: I/O priority queue behavior under Windows StorNVMe on this hardware; an ETW trace answers whether the hint is enforced. Low-cost diagnostic; not a pipeline Stage 3 target.

- **C-L1-ATTEN-GATE** (total 8, REJECT-low-score): A1=1 (10 MB absolute saving on L1 only). The per-layer W_Q rank measurement at L1 is a 5-min sub-experiment of the C-JOINT-DELNLL setup and should be collected there, not as a separate pipeline investment.

- **A3-DBGB** (total 8, REJECT-low-score): A4=1 due to NTFS 4 KB minimum read granularity potentially voiding the byte-alignment advantage — this is a near-CF9 precondition failure. If the 4 KB granularity floor applies, the partial-loading argument collapses to "load 4 KB blocks whether INT8 or INT4," eliminating the format-differential speedup. Surviving primitive: Windows NTFS minimum read unit vs INT8 row size; a 5-min OS query resolves it before any implementation. If the floor is 512 bytes (physical sector), byte-alignment survives; if it is 4 KB (NTFS cluster), the mechanism needs reformulation around 4 KB row groups.

- **A4-REVR** (total 8, REJECT-low-score): A4=1 because the BF16 inversion experiment is acknowledged as vacuously exact, and the interesting experiment (INT8-quantized Δ_L inversion error) is a reframe that is not fully specified. The mechanism's engineering payoff (speculative branching with zero activation memory overhead) is a promising capability but requires a more grounded mechanism derivation. Surviving primitive: reversible residual arithmetic in INT8; Stage 4 may want to attack this with an explicit zero-copy speculative branching scheme where the Δ_L quantization budget is the go/no-go criterion.

- **A5-VECD** (total 8, REJECT-low-score): A4=1 because the hypothesis that attention and MLP GEMVs can run on separate hyperthreads without DRAM bandwidth contention has no verified precondition; the author acknowledges this as the primary risk. On a memory-bandwidth-saturated system (which this hardware is for 1.7B BF16 inference), two threads sharing DRAM bandwidth is likely to show zero gain or negative gain. Surviving primitive: within-layer attention/MLP DRAM bandwidth utilization profile; AMD uProf or LIKWID on this hardware settles whether the two GEMVs can overlap in the memory subsystem.

---

## 6. Hand-off to Stage 3

**Track A advancers (5 + 2 path-3 + 1 path-4 = 8 total, shared measurement gate)**

1. **C-JOINT-DELNLL** (C, Track A, 13) — pressure-test: does subspace_overlap(V_Q^K, V_K^K) in WEIGHT space match the δ sign computed in ACTIVATION space? The weight-vs-activation alignment discrepancy is the largest source of Stage 3 risk.
2. **R6-WVOP** (R, Track A, 12) — pressure-test: is W_O full-rank? It aggregates 16 head outputs into the full residual stream; its spectrum may be flat like MLP W_up (CF8 boundary check). Prior literature on post-training V-O fusion (find or confirm absence, as of 2026-05-09).
3. **F6-WALIGN** (F, Track A, 12, elegant-equivalence: algebraic-identity) — pressure-test: does SGD systematically produce near-orthogonal (W_gate, W_up) row pairs as a feature (maximizing information/projection)? If mean |cos| ≈ 0 across all neurons, the idea produces only a structural finding, no compression. Also check whether AERO-family papers report per-neuron cosine statistics.
4. **F6-GQARED** (F, Track A, 11) — pressure-test: input-dependency risk — do the low-mean KV groups have high variance (critical on specific inputs)? Stage 3 should find post-training GQA group-pruning papers (Michel et al. 2019 covers MHA; GQA-specific post-training is probably open as of 2026-05-09).
5. **A6-LLMR** (A, Track A, 11, convergence C3) — pressure-test: SimLens (March 2026) and ConfLayers (April 2026) — do they enforce a monotone-confidence hard constraint, or only a learned threshold? The structural distinction matters for A5 re-scoring.
6. **F6-SINKCONST** (F, Track A, 11, frame-novelty, [FREE SWING]) — pressure-test: confirm that StreamingLLM and related work do not report W_K × W_Q spectral alignment as the origin of sinks. Also verify that the BOS embedding occupies a distinct region of embedding space from other high-frequency tokens (if it is just high-norm, the alignment may be trivial).
7. **A1-PEER** (A, Track A, 11, frame-novelty) — pressure-test: R² at mid-depth layer 14 is the entire open question. Stage 3 should bound the theoretical minimum reconstruction error under cross-scale projection (information bottleneck argument): d=1024 cannot represent all of d=2048's degrees of freedom without loss. What is the Frobenius lower bound on reconstruction error? This constrains the achievable R²_eval ceiling.
8. **R6-WDOWN-CONDP** (R, Track A, 9, elegant-equivalence: algebraic-identity, [FREE SWING]) — pressure-test: does INT2 tail quantization error (cold neurons, 83% of W_down) produce catastrophic per-layer ΔNLL? Measure INT2-tail-only ΔNLL as a sub-experiment first. If INT2 tail ΔNLL > 0.50 nats at single-layer level, the scheme is dead before full cascade.

**Track B advancers (4 total + 1 convergence rep = 5)**

9. **C-WVOK** (C, Track B, 12) — pressure-test: per-head W_V block SVD — if each head's 128×2048 W_V block is individually full-rank, the full-matrix SVD will be flat even if the coupling argument is correct. Stage 3 should verify the per-head vs full-matrix distinction before the experiment proceeds.
10. **R6-ATTNQ-FULL** (R, Track B, 11) — pressure-test: ΔNLL super-additivity across four simultaneously truncated matrices is the dominant risk. Stage 3 should find any paper reporting joint SVD truncation interaction coefficients for multiple attention matrices.
11. **U2-FNDP** (U, Track B, 11) — pressure-test: confirm that ik_llama.cpp uses mmap by default (not ReadFile); quantify the engineering work required to switch to FILE_FLAG_NO_BUFFERING weight loading as a non-mmap path. The standalone harness result is sufficient for Stage 3 go/no-go, but integration cost must be scoped.
12. **U3-WMCE** (U, Track B, 11) — pressure-test: **primary question for Stage 3** — does Windows Compression Store engage for file-backed mmap pages (GGUF tensors) or only for anonymous pages? This is a binary OS-behavior question answerable by RAMMap inspection in ~30 min. If mmap pages are excluded, the mechanism requires anonymous-memory loading, which composes with FNDP and should be flagged as a combined Track B NVMe path proposal.
13. **C-DEEP-SPREAD-QBUD** (C, Track B, 9, convergence C1) — pressure-test: v2-CF1's per-layer coverage was measured at layer OUTPUT (post-attention residual), not at W_Q input. Stage 3 should determine whether this distinction invalidates the coupling. If W_Q reads from the PREVIOUS layer's output (which is this layer's input), the measurement is actually measuring what W_Q encounters, not what it produces. Stage 3 resolves this direction ambiguity.
