# Stage 2 Judge — Run 008

Independent cold-start pass. Kill-list enforced: v2-CHEAP-TEST-001 (cross-layer W_Q stacked SVD class), v2-S3-R004-001 (arbitrary O(d) rotation). Saturated-frame list from prior rounds: cross-layer W_Q, per-head W_Q truncation, MLP weight rank reduction, tied-embed SVD, arbitrary rotation, softmax×RoPE shift-invariance, SmoothQuant-shaped activation-static-only schemes.

---

## 1. Union Score Table

| ID | Orientation | Track | Title | A1 | A2 | A3 | A4 | A5 | Total | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| R8-WVWO-FOLD | R | A | W_V/W_O Product Rank Cascade | 2 | 2 | 3 | 2 | 3 | 12 | ADVANCE |
| R8-RAOK-70B | R | B | RAOK 3-Tier Activation Codebook Full 70B | 2 | 2 | 3 | 3 | 2 | 12 | ADVANCE |
| R8-WVOS-SPEC | R | B | W_V/W_O Spectrum Sweep | 1 | 2 | 3 | 3 | 3 | 12 | ADVANCE-convergence |
| R8-GUCB-SHARED | R | B | Cross-Tensor Codebook Sharing [FREE SWING] | 1 | 2 | 3 | 1 | 2 | 9 | REJECT-low-score |
| R8-DELTAROPE | R | A | W_K RoPE-Precomputed Absorption | 1 | 2 | 3 | 2 | 2 | 10 | REJECT-low-score |
| C-LDOC | C | B | L27 Outlier Quasi-Static × W_Q Depth Compression | 1 | 2 | 3 | 2 | 2 | 10 | REJECT-low-score |
| C-UFDM | C | A | W_up Firing-Rank × W_Q Head-Shared Subspace | 2 | 3 | 3 | 2 | 3 | 13 | ADVANCE |
| C-RKSC | C | B | Residual Cosine × L1 Gate-Fold Variance Reduction | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE-convergence |
| C-WKOL | C | B | W_K K=512 × Static Outlier Identity Sidecar | 1 | 2 | 3 | 3 | 3 | 12 | ADVANCE |
| C-SPECQ | C | B | Spectral Rank × Functional Sensitivity Law [FREE SWING] | 0 | 3 | 3 | 2 | 3 | 11 | ADVANCE-frame-novelty |
| F1-WVWO-SPECTRUM | F | B | W_V/W_O Spectral Rank Characterization | 1 | 2 | 3 | 3 | 3 | 12 | ADVANCE-convergence |
| F2-UNTIED-LMHEAD | F | B | Untied lm_head Spectral Test Qwen3-8B | 2 | 2 | 3 | 3 | 2 | 12 | ADVANCE |
| F3-WO-HEADBLOCK-RANK | F | B | W_O Per-Head-Chunk Rank Decomposition | 2 | 2 | 3 | 3 | 2 | 12 | ADVANCE-convergence |
| F4-SVAL-CONSERVED | F | B | Sinkhorn Gauge as Rank Predictor | 1 | 3 | 2 | 2 | 2 | 10 | ADVANCE-frame-novelty |
| F5-WDOWN-SPECTRUM | F | B | W_down Spectral Rank [FREE SWING] | 1 | 2 | 2 | 1 | 2 | 8 | REJECT-low-score |
| U-WSTB | U | B | Windows Working-Set Trimmer Bias | 1 | 2 | 2 | 1 | 3 | 9 | REJECT-low-score |
| U-PGWT | U | B | NTFS Sparse File Page-Granularity Weight Tier | 1 | 3 | 3 | 2 | 3 | 12 | ADVANCE-frame-novelty |
| U-ZSTD-DICT | U | B | zstd Dictionary Weight Tile Compression | 1 | 2 | 3 | 1 | 2 | 9 | REJECT-low-score |
| U-NVME-QD | U | B | NVMe Queue-Depth Ladder Prefetch | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE |
| U-MMAPF | U | B | PrefetchVirtualMemory Fault Clustering | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE-convergence |
| U-FSCTL-PRIO | U | B | NTFS I/O Priority Steering [FREE SWING] | 1 | 2 | 3 | 2 | 2 | 10 | REJECT-low-score |
| A-PERM | A | B | Head-Permutation Gauge Collapse | 1 | 2 | 3 | 2 | 2 | 10 | REJECT-low-score |
| A-CLOG | A | A | Content-Addressable Log Inference | 1 | 3 | 3 | 2 | 3 | 12 | ADVANCE |
| A-PEER | A | A | Peer-Model Weight Delta Compression | 1 | 2 | 2 | 1 | 2 | 8 | REJECT-low-score |
| A-INTG | A | B | Integer-Only RMSNorm Folding | 1 | 2 | 3 | 3 | 2 | 11 | ADVANCE |
| A-SYMX | A | B | Channel-Permutation Canonical Ordering [FREE SWING] | 1 | 2 | 3 | 2 | 1 | 9 | REJECT-low-score |

---

## 2. Per-Advancer Rationale

**R8-WVWO-FOLD (Path 1, Track A, total 12).** Grounded in CF11's explicit open item (W_V and W_O untested). A3=3 because the 20-min smallest experiment settles the product-rank claim with a binary number (r_99/d threshold) and the NO-GO produces a clean extension of the CF8 boundary. A4=2 because the fold mechanism is algebraically sound (W_VO = W_V W_O is a standard matrix composition, no imported theorem), and the computation is auditable. A5=3 because the product-fold framing (two matrices collapsed at the composition boundary) has no published cousin for Qwen3 attention. Smallest test: SVD of W_V[14] @ W_O[14], ≤20 min. Load-bearing primitive: rank of the composed V-O projection across heads.

**R8-RAOK-70B (Path 1, Track B, total 12).** CF3 and v2-CF1 anchor the three-tier stratification rigorously. The K=0.1%/K=1% Jaccard boundary is the strongest private empirical finding in the pipeline; RAOK-70B is its natural 70B deployment arithmetic. A4=3: the tier boundaries are derived directly from measured Jaccard numbers, every step auditable in <10 lines NumPy. A5=2: SmoothQuant/LLM.int8() don't use the static/dynamic split at K=0.1% vs K=1%; the hybrid design is structurally moated. Smallest test: K=18-channel Jaccard across 28 layers, ~35 min, numerical go/no-go. Load-bearing primitive: the CF3 static-to-dynamic transition threshold.

**C-UFDM (Path 1, Track A, total 13).** Strongest single proposal in the union. The CF1×CF11 subspace overlap claim is a genuinely cross-domain coupling that neither finding alone could surface. A2=3: the joint MLP-attention input basis frame has no published analog — AERO shares computation on the output side; this shares on the input side, and the motivation comes from an orthogonal alignment claim across measurement regimes. A3=3: principal-angle computation in <5 min settles the coupling. A4=2: the subspace overlap metric is well-defined, but the correct V_up^K uses empirical activations (W_up · H_calib), not just SVD of W_up — a minor correction called out in the proposal and manageable. A5=3: no published method derives a joint input basis from CF1-firing-direction × CF11-W_Q-subspace alignment. Smallest test: principal angles between V_up^128 and V_Q^128 at L14, <5 min post-SVD. Load-bearing primitive: shared input coordinate system for MLP gating and attention querying.

**C-WKOL (Path 1, Track B, total 12).** Clean extension of the CF3/CF11 coupling class to W_K, not killed by any kill-list entry (prior runs only measured this coupling for W_Q). A4=3: the projection residual ρ_K^ℓ = ‖(I − V_K^{512} V_K^{512T}) e_O‖₂ is a one-line computation on existing SVD data. A5=3: W_K outlier-subspace alignment is untested in published literature; A3 (arXiv:2505.12942) covers W_Q only. Smallest test: ρ_K^{L14} using AQFKV W_K SVD + PDAP outlier identity, <20 min. Load-bearing primitive: whether the static outlier channels fall inside or outside the W_K K=512 compressed subspace.

**F2-UNTIED-LMHEAD (Path 1, Track B, total 12).** CF12 explicitly identifies the untied configuration as open. The Godey & Artzi mechanism makes a falsifiable prediction about untied lm_head spectrum that is structurally distinct from the tied case (single gradient path vs dual). A4=3: the Gram-matrix blockwise SVD is constructive and auditable; no imported theorem with unchecked preconditions. A5=2: CF12 result is fresh (no prior publication); the untied case is the natural open follow-up. Smallest test: blockwise Gram SVD of Qwen3-8B lm_head, ~25 min. Load-bearing primitive: whether single-gradient-path training produces concentrated lm_head spectrum (as predicted) or flat spectrum (as in the tied case).

**F3-WO-HEADBLOCK-RANK (Path 2, Track B, total 12).** Advances as convergence representative for Cluster C2 (see §3). The per-head-chunk W_O factorization algebraic identity (joint W_V/W_O compression via shared V_i^T) is constructive and auditable. A4=3: the algebraic identity W_O^(i) W_V^(i) → U_i Σ_i (V_i^T W_V^(i)) is a standard SVD composition, auditable in 5 lines. Smallest test: r_99 of W_O[14] head-chunks, 5 min. Load-bearing primitive: per-head W_O output-direction effective rank.

**A-CLOG (Path 1, Track A, total 12).** The content-addressable functional memoization frame has no published analog at layer-residual granularity (SGLang-style reuse is string-level KV, not residual-vector-level). A2=3: borrows from BLAKE3/IPFS content-addressing applied to transformer compute — a genuine frame import from distributed-systems content-addressable storage, distinct from ML inference literature. A4=2: the INT8 quantization scheme and hash-collision gateway are specified, but the correctness gate (secondary ‖cached - computed‖ check) is gestured rather than derived. A5=3: no published LLM inference method performs content-addressed memoization of per-layer residuals using approximate hash keys. CF2 (cos≈0.99 residual stream) supports the clustering premise. Smallest test: INT8 residual collision rate at L14 across 200 prompts, ~30 min. Load-bearing primitive: collision frequency of INT8-quantized residual vectors across diverse tokens.

**A-INTG (Path 1, Track B, total 11).** The γ-absorption algebraic identity is exact (provable, not approximated) and the integer-RMSNorm path forces the fold in a mechanistically clean way. A4=3: the fold W' = W × diag(γ) is a trivial column rescaling, and the integer isqrt error bound (1.4×10⁻³ per-layer accumulated) is derived analytically in the proposal. A3=3: go/no-go is a numerical precision test + 1.5× throughput gate, both ≤2 hours. The v2-S3-R004-001 kill does not apply (this is diagonal scaling, not O(d) rotation). Load-bearing primitive: algebraic fold of RMSNorm γ into adjacent weights under the integer-only constraint.

**C-RKSC (Path 2, Track B, total 10).** Advances as convergence representative for Cluster C3 (see §3). Raw total 10 is below the Path 1 floor; the convergence with A-INTG on the CF2×CF6 fold-and-variance structure earns the slot. A4=2: the variance decomposition Δh_1 = c_1 + Δh_1^{dynamic} is well-specified; the INT4 quantization tightening argument follows from the variance reduction bound. The calibration-generalization risk is real but mitigable. Smallest test: per-channel variance reduction after fold, 30 min. Load-bearing primitive: CF2's ‖Δh‖ bound constraining post-fold Layer-1 MLP output variance.

**U-NVME-QD (Path 1, Track B, total 11).** Concrete substrate primitive (Windows FILE_FLAG_OVERLAPPED + IOCP) with well-specified go/no-go. A1=2 for the 70B NVMe-offload scenario (8× throughput delta at QD8 vs QD1 is a real hardware gap). A4=2: the queue-depth arithmetic is grounded in NVMe spec behavior; the risk (driver serialization) is identified and mitigated. A5=2: no published LLM NVMe offload paper exploits Windows IOCP for deep-queue parallelism. Smallest test: QD1 vs QD8 overlapped read benchmark, 1 hour. Load-bearing primitive: NVMe hardware queue depth as an application-layer lever.

**U-MMAPF (Path 2, Track B, total 11).** Advances as convergence representative for Cluster C4. The PrefetchVirtualMemory primitive is a legitimate Windows analog of madvise(MADV_WILLNEED), applied to LLM inference without ML-level prediction. Smallest test: per-layer fault-time reduction under simulated NVMe pressure, 3–4 hours. Load-bearing primitive: OS-level asynchronous prefetch engine invoked by deterministic layer-access oracle.

**R8-WVOS-SPEC (Path 2, Track B, total 12).** Advances as convergence representative for Cluster C2. Three orientations (R, F, C) independently target W_V and W_O characterization. Smallest test: SVD of W_V[14] and W_O[14], ~25 min.

**C-SPECQ (Path 3, Track B, total 11, [FREE SWING]).** A2=3: the SRR(class) = ΔNLL(50%) / (1 − r_99/d) cross-class sensitivity law with gradient_path_count as the predictor is a genuine analytical frame import — it couples functional sensitivity to training graph structure, a frame absent from the ML compression literature. The "no residency reduction" A1=0 is noted; value is as a constraint on future proposals and a quantization-schedule law. A3=3: the computation is arithmetic on existing prior-run data, <5 min. A4=2: the SRR metric is well-defined; the gradient-path-count hypothesis is testable. Advanced as [FREE SWING] — CF-tether partially present (reuses CF8, CF11, CF12 data), structural floor met.

**F4-SVAL-CONSERVED (Path 3, Track B, total 10).** A2=3: Sinkhorn balancing as a canonical compression-gauge predictor is a genuine import from matrix-scaling / optimal-preconditioner theory. The connection between SmoothQuant's empirical γ-absorption and the Sinkhorn-optimal H_s minimization has no published LLM analog. A4=2: Sinkhorn convergence theorem applies (Gram W_Q W_Q^T > 0 is verified since W_Q is full-rank per CF11); the precondition is explicitly checked. A3=2 (not 3): the go/no-go is ≤8 hours but the experiment is not quite a class-level killer — NO-GO tells you W_Q is already near-balanced from training (an interesting finding about SGD and spectral concentration), but it doesn't kill the broader class the way a 3-grade settler would. Advanced under Path 3 for frame novelty, not Path 1.

**U-PGWT (Path 3, Track B, total 12).** A2=3: NTFS FSCTL_SET_ZERO_DATA as a weight-pruning storage primitive — the filesystem's own sparse-extent representation as a zero-compute zero-read substitute for weight-zero pages — is a genuine substrate-recognition move with no published LLM analog. Distinct from the killed Reflink idea (NTFS-Reflink killed for Home limitation; sparse files available on all NTFS volumes including Home). A4=2: the sparse-file zero-read semantics are documented Win32 behavior; the risk (mmap + sparse semantics interaction) is real and mitigated by the two-path test (ReadFile vs MapViewOfFile). A3=3: go/no-go is a 1–2 hour mmap trace with a binary outcome (zero NVMe reads vs NVMe reads still issued for punched offsets). Advanced under Path 3 as genuine substrate-primitive recognition.

---

## 3. Convergence Map

**Convergence C1 — W_V and W_O spectral rank as next attention-weight compression target.**
R8-WVOS-SPEC (R, proposed as explicit cascade prerequisite), F1-WVWO-SPECTRUM (F, first-principles functional-role prediction: no-RoPE → more concentrated than W_K), F3-WO-HEADBLOCK-RANK (F, per-head-chunk output rank tied to CF11 head-redundancy), and R8-WVWO-FOLD (R, product-rank fold approach) all converge on W_V and W_O characterization as the next mandatory measurement. Three of four share the load-bearing primitive: whether the V-O attention weight pair has r_99/d < 0.80. Strongest representative: F3-WO-HEADBLOCK-RANK (constructive algebraic identity + per-head structure independent of W_Q). Cheapest settler: SVD of W_V[14] and W_O[14], ~25 min — this single measurement unblocks four proposals simultaneously and closes the attention-weight class characterization.

**Convergence C2 — CF2/CF6 Layer-1 fold as structured constant-vector extraction.**
C-RKSC (C, composes CF2 variance bound with CF6 gate-fold to derive per-channel INT4 tightening) and A-INTG (A, γ-absorption fold forced by integer-only constraint, shares the CF6 gate-fold algebraic identity at Layer 1) converge on the same structural primitive: folding the near-constant Layer-1 gate contribution out of the forward path, then exploiting the reduced per-channel variance for quantization. Different vocabularies (Composition coupling vs Constraint-Alien algebraic identity), same load-bearing math. Strongest representative: C-RKSC (explicit quantization arithmetic from the fold). Cheapest settler: per-channel output variance of Δh_1 with and without c_1 subtraction, 30 min.

**Convergence C3 — NVMe layer-prefetch via OS asynchronous primitives.**
U-NQDL (IOCP overlapped queue-depth) and U-MMAPF (PrefetchVirtualMemory for demand-fault hiding) converge on the same load-bearing primitive: asynchronous OS-level prefetch of transformer layer weights ahead of compute, exploiting the deterministic layer-access pattern as a zero-ML-prediction oracle. Structural difference is the API path (IOCP vs PrefetchVirtualMemory); both require the same precondition measurement (NVMe effective throughput at QD1 vs QD8 for sequential large-block reads on Ryzen 5 7530U). Strongest representative: U-MMAPF (simpler API path, no mmap complications, also re-derives CF13* as a side effect). Cheapest settler: QD1 vs QD8 overlapped-read benchmark + PrefetchVirtualMemory mmap confirmation, 1–2 hours combined.

---

## 4. Frame-Novelty Bonus Advancers (Path 3)

**C-SPECQ — Sensitivity-Rank Ratio law with gradient-path-count predictor.** Frame: cross-class quantization sensitivity as a function of training-graph topology (gradient path count), not just spectral flatness. Named cousin: mixed-precision calibration (OmniQuant, LLM-QP) uses Hessian sensitivity, not structural training topology. Saturation boundary: no published LLM compression method derives compression tolerance from the training computation graph rather than from calibration activations. The cross-class SRR table (W_Q to tied embed, 8000× range in SRR) is a genuine empirical synthesis from existing CF data that produces a novel structural claim. Advances as [FREE SWING] structural measurement; Stage 3 should test the gradient-path-count predictor on the W_gate vs W_up SRR gap.

**U-PGWT — NTFS sparse-file zero-extent as inference-time weight pruning storage.** Frame: filesystem sparse-extent semantics (OS returns zeros with zero I/O for punched extents) as a weight-zeroing primitive that requires no model format change and no ML logic. Named cousin: magnitude pruning (ML side); NTFS-Reflink codebook dedup (kill list, different mechanism). Saturation boundary: no LLM inference paper uses FSCTL_SET_ZERO_DATA as a weight-sparsity storage mechanism. The move is pure substrate recognition — the filesystem has been providing zero-read semantics for sparse regions for 30 years and LLM inference has never pointed at it. Stage 3 should test whether Windows mmap honors sparse-file zero semantics under MapViewOfFile on NTFS.

**F4-SVAL-CONSERVED — Sinkhorn balance as optimal-H_s gauge for compression planning.** Frame: matrix-scaling theory (Sinkhorn-Knopp optimal balance) as a compression-gauge predictor — the "tightest-rank gauge" is derivable from first principles rather than from calibration. Named cousin: SmoothQuant (γ-absorption for quantization balance) — FMCP provides the theoretical foundation SmoothQuant never derived. Saturation boundary: no published LLM compression paper derives optimal per-matrix gauge from Sinkhorn balance theory. The precondition (W_Q W_Q^T > 0) is verified. Advances under Path 3.

Three candidates; the two strongest A4 are C-SPECQ (A4=2) and F4-SVAL-CONSERVED (A4=2), both tied; U-PGWT also at A4=2. Per STAGE2_JUDGE.md rule: pick two with strongest A4. All three tie at A4=2. Tiebreak by A3: C-SPECQ (A3=3) > F4-SVAL-CONSERVED (A3=2). C-SPECQ and U-PGWT advance as Path 3 slots (both A3=3, A4=2). F4-SVAL-CONSERVED is displaced; it advances separately under Path 4 (see below) due to the elegant-equivalence signal.

**Path 4 — Elegant equivalence: F4-SVAL-CONSERVED.** Sub-class: `gauge-exploitation`. The Sinkhorn-balanced form minimizes spectral entropy by a lossless coordinate change — the compression savings come from a gauge-fixing argument, not from approximation. A4=2 meets the Path 4 floor (≥2). The elegance is that the gauge-fixing is calibration-free (no activation data needed) and produces the tightest-rank representation by construction. Tag: `gauge-exploitation`. A2 +1 from elegant-equivalence (fresh relative to published compression gauge literature) → effective A2=3 for this proposal's Stage 3 brief. A4 +1 not triggered (the construction is auditable but not quite 1-paragraph level for the Sinkhorn convergence argument).

---

## 5. Rejected with Rationale

- **R8-GUCB-SHARED [FREE SWING]** — A4=1 (the structural claim that W_Q/W_K/W_V row distributions align is hand-waved; no theorem imported but the alignment premise is asserted not derived). A1=1 (residency saving marginal). Total 9, below Path 1 floor, no convergence or frame-novelty rescue. Surviving primitive: within-layer cross-tensor codebook row-distribution alignment test. Stage 4 may rescue as a cheap measurement (5-min k-means experiment), not a standalone proposal.

- **R8-DELTAROPE** — A1=1 (KV cache compression at 4K context yields 234 MB saving — marginal). A4=2 but mechanism rests on the unitary-SVD-truncation argument, which is mathematically valid yet the quantization-grid phase-interaction risk is real and the go/no-go is more nuanced than stated (the ±0.05 nats gate could pass trivially if truncation quality is dominated by rank, not phase). The idea is scientifically sound but the Stage 2 ceiling is 10/15 — below Path 1 and the NO-GO outcome does not kill a class (just re-derives what AQFKV already showed). Stage 4 could revisit if long-context (32K+) becomes the primary target.

- **C-LDOC** — A1=1 (saving from L27-only K=64 is 1 MB on 1.7B — negligible; the full per-layer schedule is the value but that is contingent on the coupling being real). A2=2 (Jaccard-to-compression-depth monotonicity is in-vocabulary for mixed-precision schedules, not fresh enough for A2=3). Total 10, below floor. The coupling measurement (per-layer single-replacement ΔNLL) is a legitimate cheap measurement that would flow into any per-layer schedule proposal. Not killed — deferred for inclusion in C-UFDM's Stage 3 as a supporting measurement if UFDM advances.

- **F5-WDOWN-SPECTRUM [FREE SWING]** — A3=2 (the go/no-go is activation-variance fraction, which does not directly decide NLL quality; the proposal acknowledges a second NLL experiment would be required before claiming savings — the smallest test is not a single decisive bit). A4=1 (the activation-distribution effective rank premise is gestured rather than derived: the claim "W_down is sensitive to rare activations" requires a theoretical grounding that is absent). Total 8. Surviving primitive after stripping: W_down SVD spectrum measurement is cheap (20 min) and is not killed — the kill list kills W_gate and W_up rank reduction, not W_down explicitly. Stage 4 may surface W_down as a measurement supporting RAOK-70B's activation-quantization cascade.

- **U-WSTB** — A3=2 (the smallest test requires background DRAM pressure setup; the go/no-go is p95 Wilcoxon at p<0.05, which is a statistical comparison requiring ≥20 trials → total runtime 2–3 hours with uncertain setup; not a clean single-bit outcome). A4=1 (the critical risk that mmap pages are managed by NTFS Cache Manager rather than WST is a structural precondition that might make the entire mechanism a no-op; this should have been the first experiment, but as stated it is the mitigation). Total 9. The VirtualLock + Cache Manager path disambiguation is a real open question; surviving primitive: whether Windows mmap-backed GGUF pages enter the WST or Cache Manager path (5-min pre-check). Stage 4 may pick this up as a prerequisite for U-MMAPF.

- **U-ZSTD-DICT** — A4=1 (bf16 neural weight tiles at 4–16 KB granularity have near-uniform byte entropy — the mechanism depends on per-group dictionary capturing recurring patterns, but the proposal acknowledges this is uncertain; no first-principles argument that same-layer-type bf16 tiles share enough byte-level structure for a zstd dictionary above ~1.05×). Total 9. The 30-min compressibility pre-check is cheap; if the ratio is ≥1.3×, the idea should be re-surfaced in Stage 4. Not a kill-list item yet — conditional on compression-ratio outcome.

- **U-FSCTL-PRIO [FREE SWING]** — A4=2, A1=1 (I/O priority scheduling is a scheduling intervention, not a residency or throughput gain in the strict sense; the 20–40% NVMe recapture from background contention is real but only meaningful in noisy-desktop scenarios, not in a clean bench). Total 10. Valid substrate primitive; below Path 1 floor with no convergence rescue. The I/O priority API is worth a 1-hour test in Stage 5 as a free add-on to NQDL/MMAPF.

- **A-PERM** — A1=1 (the permutation-sorted bit-width ladder saves 0 bytes net and yields a 0.25 bpw quality-equivalent saving — modest). A4=2 (the permutation symmetry argument is correct; the per-head r_99 measurement is the right smallest test). Total 10. The per-head W_Q r_99 distribution measurement (20 min) is load-bearing for C-UFDM and should be done as part of UFDM's Stage 3 regardless. The permutation-ladder idea itself is superseded by UFDM's joint-basis proposal which produces a structural saving rather than a quality-per-bit improvement.

- **A-PEER** — A3=2 (the smallest test is 45 min but the GO/NO-GO outcome — even if GO — doesn't produce a deployable mechanism immediately; it's a measurement of delta rank, not a deployable compression). A4=1 (the fundamental risk — independently trained models from the same recipe produce arbitrary weight differences — is a strong prior that the proposal acknowledges; without any theoretical argument for why same-family cross-scale W_Q deltas should be low-rank, the mechanism premise is hand-waved). Total 8. Surviving primitive: if W_Q deltas ARE low-rank (a genuine possibility given Qwen3's family-consistent training recipe), this becomes a novel compression angle. Stage 4 may revisit after WVWO-FOLD settles whether attention weights are structurally regular within a family.

- **A-SYMX [FREE SWING]** — A5=1 (the channel-permutation + FP16-outlier-sidecar framing is substantively adjacent to LLM.int8() and SmoothQuant; the "permutation makes outlier detection position-free" novelty is incremental over AWQ/SmoothQuant, which already derive per-channel importance from calibration activations). Total 9. The concentration test (top-2 vs median activation magnitude) is a cheap measurement that supports RAOK-70B's Tier 1 design; surviving primitive: canonical channel ordering as a stateless outlier-identification scheme. Pass to Stage 4 as a potential free simplification of RAOK-70B's T1 detection.

---

## 6. Hand-off to Stage 3

### Track A advancers

1. **C-UFDM** (C / Track A / 13) — Pressure-test: does the empirical firing subspace V_up^{128} (computed from W_up · H_calib, NOT from W_up SVD alone) actually align with V_Q^{128} at the ≥0.3 principal-angle overlap threshold? The fix-up (use activation-weighted V_up, not pure SVD) must be confirmed. Stage 3 should also check: does the AERO paper (arXiv:2410.13060) or A3 (arXiv:2505.12942) discuss input-side joint basis at all?

2. **R8-WVWO-FOLD** (R / Track A / 12) — Pressure-test: is the V-O product rank structurally lower than either individual matrix? W_O acts as an "output projection" analogous to W_down (MLP output), which is full-rank under CF8. If W_O shares W_down's rank structure, the product may be full-rank. Stage 3 should assess the functional-role analogy: W_O is linear (no nonlinearity coupling), unlike W_down which is coupled to post-SwiGLU activations — this distinction is the mechanism's moat.

3. **A-CLOG** (A / Track A / 12) — Pressure-test: what is the realistic INT8 collision rate? The CF2 cos≈0.99 argument supports clustering but the CF3 token-dynamic K=1% finding argues against it — diverse prompts will produce distinct residuals even at INT8. Stage 3 should check whether any memoization/caching literature (SGLang, vLLM prompt cache) measures residual-collision rates at the layer level; the claim needs a realistic hit-rate estimate before committing to the scheme.

### Track B advancers (top 4)

4. **R8-RAOK-70B** (R / Track B / 12) — Pressure-test: does the K=18-channel (0.9%) per-token Jaccard across all 28 layers hold below 0.40? v2-CF1 confirmed all-layer K=1% mean 0.388; K=18 is slightly below K=1% (18/2048=0.879%). The Jaccard could be slightly higher or lower at K=18 than at K=20 (K=1%). Stage 3 should also check the AVX2 scatter-add overhead estimate (18 INT8 scatter operations per 2048 INT4 bulk) — does ik_llama.cpp's GEMV loop tolerate scatter-add modification?

5. **F2-UNTIED-LMHEAD** (F / Track B / 12) — Pressure-test: can the Gram-matrix streaming SVD for lm_head (151936×4096) be executed on the Ryzen 5 7530U with 7.28 GiB RAM? The Gram matrix (4096×4096 float32 = 64 MB) is fine, but streaming row-by-row of 151936 rows × 4096 × 2 bytes = 1.19 GB requires a streaming iterator over the weight tensor. Confirm with the safetensors partial-load API that the 8B model's lm_head can be read without loading the full 8B model.

6. **C-WKOL** (C / Track B / 12) — Pressure-test: the outlier channel identity O is measured on MLP block inputs; Stage 3 must verify these are the same channels as the attention block inputs. The residual stream is shared, but the top-K=0.1% channels at MLP input vs attention input may differ per layer. This is the primary risk; a <2-hour PDAP re-run resolves it.

7. **F3-WO-HEADBLOCK-RANK** (F / Track B / 12, convergence rep for C1) — Pressure-test: the per-head r_99 measurement at L14 is the gate. If r_99(W_O^{(i)}) ≥ 100 for all heads, the mechanism dies. Stage 3 should assess whether head specialization in trained Qwen3 (CF11 head-redundancy, 16 heads → 128-dim subspace) predicts concentrated or full-rank per-head W_O. The CF11 head-redundancy finding says the heads are functionally redundant at the query level — does this extend to the value/output path?

### Convergence, frame-novelty, elegant-equivalence

8. **R8-WVOS-SPEC** (convergence rep C1 backup / Track B / 12) — carries forward as the shared prerequisite measurement for C1. Stage 3: confirm no published post-training W_V/W_O spectrum measurement for Qwen3-class models.

9. **C-SPECQ** (frame-novelty / Track B / 11, [FREE SWING]) — Stage 3: compute the full SRR table from existing experimental data and test whether gradient_path_count monotonicity holds. Check Molchanov et al. ICLR 2017 for prior art on Fisher sensitivity across weight classes.

10. **U-PGWT** (frame-novelty / Track B / 12) — Stage 3: confirm NTFS sparse-file zero-read semantics under MapViewOfFile. The critical question is whether the NTFS Cache Manager, not the File System Driver, handles the mmap path and whether Cache Manager honors FSCTL_SET_ZERO_DATA holes. The 1–2 hour experiment is the gate.

11. **F4-SVAL-CONSERVED** (elegant-equivalence `gauge-exploitation` / Track B / 10) — Stage 3: confirm Sinkhorn precondition (W_Q W_Q^T > 0 trivially satisfied per CF11 full-rank statement). Check whether SmoothQuant's empirical γ-absorption scale is Sinkhorn-equivalent (if it is, SmoothQuant IS Sinkhorn balance and F4 is a re-derivation, not novel). This is the pre-emption risk; Stage 3 must resolve it.
