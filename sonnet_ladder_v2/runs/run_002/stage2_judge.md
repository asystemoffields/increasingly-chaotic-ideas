# Stage 2 Judge — Run 002
Run: 002 | Date: 2026-05-09 | Judge: Sonnet claude-sonnet-4-6

---

## 1. Union Score Table

| ID | Orientation | Track | Title | A1 | A2 | A3 | A4 | A5 | Total | Verdict |
|----|-------------|-------|-------|----|----|----|----|-----|-------|---------|
| R1-A / QKJB | R | A | Joint Q/K Low-Rank Folding Cascade | 2 | 1 | 2 | 2 | 2 | 9 | ADVANCE |
| R2-B / RAOK-SCALE | R | B | Activation-Outlier Tiered-Quant Cascade | 2 | 2 | 2 | 2 | 2 | 10 | ADVANCE |
| R3-A / XLQB | R | A | Cross-Layer W_Q Basis Sharing (Reach) | 2 | 2 | 3 | 3 | 3 | 13 | ADVANCE |
| R4-B / TERNARY-NVMe | R | B | Ternary Weight + NVMe Paging Cascade | 1 | 1 | 2 | 2 | 1 | 7 | REJECT-low-score |
| R5-A / WDLA-RESCUE | R | A | Cross-Scale Affine Surgery (Rescoped) | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE |
| R6-B / ATTN-SPECTRUM | R | B | W_V/W_O Rank Cascade | 2 | 1 | 3 | 3 | 2 | 11 | ADVANCE |
| R7-B / HYBRID-TIER-70B | R | B | Hot/Warm/Cold Tiered-Quant [FREE SWING] | 1 | 1 | 2 | 2 | 1 | 7 | REJECT-low-score |
| C1 / OHAA | C | B | Outlier-Highway Attention Alignment | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE-convergence |
| C2 / CQST | C | B | Cross-Layer W_Q Subspace Telescoping | 2 | 2 | 3 | 3 | 3 | 13 | ADVANCE |
| C3 / LGOP | C | B | Layer-1 Gate-Fold × Outlier-Static Pinning | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE |
| C4 / QCDS | C | B | W_Q Cross-Layer Drift Quant Schedule | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE |
| C5 / ROPE | C | A | Residual Stream Outlier-Head Excision [FREE SWING] | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE-wildcard |
| C6 / RKGP | C | B | Residual-Cosine Flatness × W_Q Depth-Gradient | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE |
| F1 / JVOC | F | B | Joint W_V/W_O Spectral Collapse | 2 | 2 | 3 | 3 | 2 | 12 | ADVANCE |
| F2 / CQBS | F | B | Cross-Layer W_Q Basis Sharing (FP) | 2 | 2 | 3 | 3 | 3 | 13 | ADVANCE |
| F3 / RMGF | F | A | RMSNorm Gauge Freedom Exploitation | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE-frame-novelty |
| F4 / SSIF | F | A | Softmax Shift Invariance Folding | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE-frame-novelty |
| F5 / WDCA | F | B | W_down Spectral Audit / Output Subspace | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE |
| F6 / RSGO | F | A | Residual-Stream Gauge Orbit Folding [FREE SWING] | 1 | 3 | 2 | 2 | 2 | 10 | ADVANCE-wildcard |
| U1 / PrefetchVM | U | B | GGUF Extent Coalescing / PrefetchVirtualMemory | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE |
| U2 / VirtualLock | U | B | VirtualLock Hot-Tier KV Pinning | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE |
| U3 / zstd-dict | U | B | GGUF ANS-Dict Pre-Warmed zstd | 0 | 2 | 2 | 1 | 1 | 6 | REJECT-low-score |
| U4 / NTFS-sparse | U | B | NTFS Sparse File Sentinel | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE |
| U5 / MEM_LARGE_PAGES | U | B | HugePage Promotion for Weight Tensors | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE |
| U6 / IOCP-dispatch | U | A | Named Pipe Zero-Copy IOCP Dispatch [FREE SWING] | 0 | 2 | 1 | 1 | 2 | 6 | REJECT-no-smallest-test |
| U7 / ReadDirChanges | U | B | ReadDirectoryChangesW Weight Hot-Swap | 0 | 2 | 2 | 2 | 3 | 9 | REJECT-low-score |
| A1 / GHPI | A | A | Gauge-Fixed Head-Permutation Inference | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE |
| A2 / ASAKS | A | A | All-State Append-Only KV Store | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE |
| A3 / TAAR | A | A | Tropical-Algebra Attention Routing | 1 | 3 | 2 | 1 | 3 | 10 | ADVANCE-frame-novelty |
| A4 / RSLK | A | B | Register-Only Streaming Layer Kernel [FREE SWING] | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE-wildcard |
| A5 / CAWD | A | A | Content-Addressed Weight Dispatch | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE |
| A6 / RGPS | A | B | Residual-Stream Gauge-Fixed Precision Scheduling | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE-elegant-equivalence [gauge-exploitation] |
| A7 / FSMD | A | A | Finite-State-Machine Token Decoder | 1 | 3 | 2 | 1 | 3 | 10 | REJECT-CF9 |

---

## 2. Per-Advancer Rationale

**R3-A / XLQB** (Track A, total 13): Carried by A3=3 (30-min PCA stack, binary go/no-go on ≥50% cross-layer variance at K=128) and A4=3 (the construction is a reshape + SVD auditable in 5 lines; the within-layer CF11 head-redundancy finding is the structural motivation; cross-layer extension is a clean falsifiable hypothesis). A5=3 because the cross-layer shared basis for W_Q has no published analog post-training; closest is ALBERT (training-time tied weights), structurally distinct. Load-bearing primitive: whether the 28 W_Q matrices share a single ~128-dim subspace in R^{2048}. Smallest test: stack all 28 W_Q, PCA across layers, measure cumvar at K=128. **Convergence representative for Cluster C1 (see §3).**

**F2 / CQBS** (Track B, total 13): Mathematically identical object to XLQB approached from First-Principles framing (construct the 57344×2048 stacked matrix, run SVD, measure cumvar). A4=3 because the construction is fully auditable: reshape, SVD, read off singular values. A5=3 because this exact post-hoc measurement on a trained model has no published analog. No CF-tether gap: CF11 explicitly named "cross-layer Q basis: PCA-stack experiment" as open. Smallest test: cumvar(K=128) ≥ 0.85 in 20 min. **Shares the convergence cluster; strongest A4 of the cluster.**

**C2 / CQST** (Track B, total 13): Adds the CF2 coupling argument — if residual streams are near-parallel across layers (cos≈0.99), the forcing function for W_Q cross-layer alignment is articulated. The α_cross metric is more nuanced than PCA variance and provides a directional alignment measure. A3=3 because the go/no-go is a single mean statistic (Spearman ρ of alignment vs layer index) computable in the same 28-SVD pass as XLQB/CQBS. The CF2 + CF11 composition is the novelty above the bare measurement. **Convergence representative (alongside XLQB/CQBS in Cluster C1); strongest theoretical grounding of the cluster.**

**F1 / JVOC** (Track B, total 12): Targets the fused operator ∑_h W_O^h W_V^h — a distinct object from either per-matrix spectrum. A4=3: the construction is exact (sum over heads, run SVD on 2048×2048 result), auditable in <10 lines. A3=3: 25-min protocol, go/no-go at r_99/d ≤ 0.75. Closes the attention-weight class characterization opened by CF11. **Convergence representative for Cluster C2 (W_V/W_O characterization).**

**R6-B / ATTN-SPECTRUM** (Track B, total 11): Adds the 70B arithmetic needed by the R1-A cascade; explicitly measures W_V and W_O per-matrix spectra. A4=3: same AQFKV script with matrix substituted. A3=3: 20-min runtime, quantitative ΔNLL threshold. Load-bearing for whether 4-matrix attention compression is viable. **Convergence representative for Cluster C2 (complementary to JVOC).**

**C5 / ROPE** (Track A, total 11; FREE SWING): Proposes that static outlier channels (CF3) serve as the routing separator between heads — a structural claim absent from both CF3 and CF11 independently. A3=3: 7-min forward pass + β_k computation. Structural no-go reveals head-outlier independence, which is itself a load-bearing characterization. CF-tether requirement suspended; structural floor met. Smallest test: CV(β_k) ≤ 0.3 in >50% of layers.

**F3 / RMGF** (Track A, total 11; frame-novelty): Gauge-absorption of RMSNorm gain into downstream weight spectra to test spectrum concentration. A2=3 because the question "does folding g_L into W_Q change the effective spectral rank?" has no published measurement on any model family — SmoothQuant absorbs scale for quantization dynamic-range reasons; RMGF asks whether the fold is spectrum-shaping. No kill-list cousin. A4=2 (the fold itself is algebraically exact; the concentration claim is a measurement hypothesis). Smallest test: 20-min SVD comparison, median r_99/d ≤ 0.55. **Path 3 frame-novelty advancer #1.**

**F4 / SSIF** (Track A, total 11; frame-novelty): Uses softmax shift-invariance as a per-head logit-offset removal tool to sharpen Q/K dynamic range for quantization. A2=3 because the specific use case — per-head mean logit offset as a KV-precision sharpener — has no published form. The algebraic identity (softmax shift-invariance) is elementary and exact; the application to dynamic-range reduction is the novelty. RoPE complication is real but testable (the smallest experiment measures per-position offset variance). A4=2 because the mechanism requires the measured offset to have non-trivial norm. Smallest test: 30-min forward pass measurement. **Path 3 frame-novelty advancer #2.**

**A1 / GHPI** (Track A, total 11): Head-permutation gauge symmetry is a real symmetry of the computation graph (bit-for-bit invariant under head sort + W_O column repermutation). A2=3: the gauge-fixing framing for attention compression has no published form in the compression literature (equivariant networks know the gauge group; LLM compression papers do not use it). A4=2: the SVD + argsort construction is checkable; the cross-layer alignment claim in the sorted basis is measured, not derived. Smallest test: 20-min SVD across all heads and layers.

**A2 / ASAKS** (Track A, total 11): Log-structured append-only KV store transplanted from LSM-tree storage engines. A2=3: the append-only framing for KV is a genuine frame transplant — all published KV compression (KVQuant, MiniKV, KITTY) uses mutation-based operations; ASAKS is structurally distinct. A4=2: the dedup mechanism is specified (SimHash / cosine threshold), the smallest test is a direct measurement of K-vector near-duplicate rate. Smallest test: 15-min forward pass + cosine matrix.

**C1 / OHAA** (Track B, total 10; convergence): Projects the 2 static outlier channels (CF3) onto the W_Q dominant subspace (CF11) and tests dimensional overlap. Even if outlier channels are orthogonal to the W_Q subspace, the result closes a structural hypothesis that neither CF alone can answer. A3=3: 10-min computation (SVD is already done for CQST; dot-product check is free). Load-bearing primitive: the alignment equation ‖e_i^T V_Q^ℓ_{K=128}‖₂. **Convergence representative for Cluster C3 (CF3 × CF11 coupling).**

**C3 / LGOP** (Track B, total 10): Tests whether the Layer-1 anomaly (CF6: 36% foldable neurons) is mechanistically caused by the static outlier channels (CF3). A3=3: 10-min column-norm extraction. The composition hypothesis is specific and falsifiable. Closes whether SDZC's layer-1 anomaly is signal-driven or an artifact.

**C4 / QCDS** (Track B, total 10): Tests whether W_Q effective rank increases monotonically with layer depth (CF2 near-parallel residuals as forcing function). A3=3: Spearman ρ in the same 28-SVD pass as CQST. Free to run if CQST runs first.

**A3 / TAAR** (Track A, total 10; frame-novelty consideration): Tropical-algebra / max-plus semiring applied to attention. A2=3: genuinely novel frame — the constraint "no continuous numbers in attention" forced a discrete algebraic structure nobody has applied to transformer attention. However, A4=1: the approximation quality argument ("H(s)/log(n) bound") imports a heuristic relationship between softmax entropy and the argmax approximation that is not a proven theorem for arbitrary softmax inputs. The entropy bound is qualitative. Advancing under TAAR's own frame-novelty signal but A4=1 caps the path. **Narrowly eligible for Path 3 but displaced by RMGF/SSIF which have A4=2; held as honorable mention. See rejections.**

Actually, on re-score: TAAR has A4=1 (the approximation claim is hand-waved). Under Path 3 rules, the two highest-A4 A2=3 proposals advance. TAAR (A4=1) loses to RMGF (A4=2) and SSIF (A4=2). TAAR is **not** a Path 3 advancer. See rejections.

**A6 / RGPS** (Track B, total 9; elegant-equivalence `gauge-exploitation`): The residual-stream rotation symmetry is a true symmetry of transformer inference, and PCA-gauge fixing is a principled basis choice that makes per-direction precision assignment natural. A4=2 because the rotation W_gate·P_L is algebraically exact (bijective change of basis); the precision gain is measured, not derived, but the measurement is specified. A3=2: 40-min calibration + eigendecompose + matmul. The elegance is the gauge-exploitation sub-class — the gauge group is real and the PCA fix is canonical. Closest published: QuaRot uses Hadamard (random orthogonal); RGPS uses the empirically principled PCA fix. **Path 4 elegant-equivalence advancer; sub-class: `gauge-exploitation`.**

**R1-A / QKJB** (Track A, total 9): The first-rung W_V/W_O spectrum measurement is the key contribution; the 70B cascade arithmetic is aggressive but depends on many conditional rungs. A4=2: the SVD substitution is auditable; the per-layer ΔNLL compounding concern is explicitly flagged. A3=2: 20-min first rung. Advances as a structural-finding cascade even if the 70B end-to-end is unlikely.

**R2-B / RAOK-SCALE** (Track B, total 10): CF3's K-dependent Jaccard finding is the direct motivation for the 3-tier partition. A4=2: the tier boundaries are mechanistically derived from CF3 numbers (K=0.1% static, K=1% dynamic). A3=2: 2-hour ΔNLL measurement. The Reach-scale framing (does CF3 generalize to 7B/70B?) adds structural value above the existing RAOK entry in the deferred list.

**R5-A / WDLA-RESCUE** (Track A, total 9): The CF10 fix (forced rank-64, 10K tokens, ridge=0.1) is exactly the Stage 6 amendment prescription. A4=2: the conditioning check is explicit (196K params / 2.05M values = 0.096 — well-conditioned). A3=2: 3-hour experiment with R²_eval > 0.85 gate. Load-bearing: whether the WDLA frame is broken at all or just data-starved.

**R6-B / ATTN-SPECTRUM** already covered above.

**F5 / WDCA** (Track B, total 9): The activation-weighted output subspace of W_down is a distinct object from W_down's weight SVD (which CF8 would kill). The CF10 mitigation is correctly specified (pure SVD truncation, no least-squares fit). A3=2: 40-min run. A4=2: the SVD truncation construction is exact.

**C6 / RKGP** (Track B, total 9): KV layer-skip recomputation using CF2's near-parallel residuals. A4=2: the approximation bound derivation (δ ≈ 0.14×‖h‖) is auditable from CF2. A3=2: 20-min relative-error measurement. Not as strong as CQST but fills the KV cache side of the attention story.

**U1-U5, A4, A5**: All advance at borderline 9 with clean A3≥2, A4≥2 on substrate-specific mechanisms. U1 (PrefetchVirtualMemory), U2 (VirtualLock), U4 (NTFS-sparse), U5 (MEM_LARGE_PAGES) are all verifiable substrate primitives with quantitative smallest tests. A4/RSLK (FREE SWING) and A5/CAWD meet structural floor.

---

## 3. Convergence Map

**Convergence C1 — Cross-layer W_Q subspace alignment as a universal compression primitive.**

Four orientations independently arrived at the same structural question via different routes: R (Reach) via XLQB, proposing the PCA-across-layers experiment as the first rung of a 70B residency cascade; C (Composition) via CQST, composing CF11's per-layer head-redundancy with CF2's near-parallel residuals to predict cross-layer alignment; F (First-Principles) via CQBS, constructing the explicit 57344×2048 stacked matrix and measuring cumvar at K=128; and A (Constraint-Alien) via GHPI, using the head-permutation gauge symmetry argument to motivate looking for cross-layer lane structure in sorted coordinates. All four identify the same load-bearing primitive: the question of whether the 28 W_Q matrices in Qwen3-1.7B span a single low-dimensional subspace in R^{2048}. The four proposals use different vocabularies (PCA cascade, composition of CF2+CF11, constructive algebraic experiment, gauge symmetry collapse) but their experimental protocols converge on the same 20-30 min SVD-on-stacked-tensor computation. **Strongest representative: F2/CQBS** (A4=3, most explicit construction, lowest probability of hidden precondition). **Cheapest settler: cumvar(K=128) on the 57344×2048 matrix, ~20 min runtime.** If cumvar ≥ 0.85 → shared basis real, 16× W_Q compression; if cumvar < 0.70 → layers are independent, closes the cross-layer sharing route, and constrains all four proposals simultaneously.

**Convergence C2 — W_V/W_O characterization as the last unmeasured attention-weight class.**

Three orientations independently flagged W_V and W_O spectra as the next measurable object: R via ATTN-SPECTRUM (measuring per-matrix ΔNLL, completing the 4-matrix attention map for the 70B cascade), F via JVOC (measuring the fused operator ∑_h W_O^h W_V^h), and R via QKJB (requiring W_V/W_O as a precondition for the full 4-matrix compression cascade). All three depend on whether the attention-weight class compression confirmed for W_Q and W_K by CF11 extends to W_V and W_O. **Strongest representative: F1/JVOC** (A4=3, algebraically exact construction, fused operator measurement closes the class more definitively than per-matrix). **Cheapest settler: SVD of ∑_h W_O^h W_V^h per layer (~25 min); if r_99/d ≤ 0.75, the fused operator is compressible; if ≈1.0, extends CF8 to the V-O product class.**

**Convergence C3 — Static outlier channels (CF3 K=0.1%) as a multi-purpose structural primitive.**

Three orientations used CF3's K=0.1% static outlier channels (Jaccard=0.718) as a structural input: R via RAOK-SCALE (activation quantization tiering), C via OHAA (alignment with W_Q dominant subspace), and C via LGOP (mechanistic explanation of CF6's Layer-1 anomaly). This convergence is notable because CF3 was a Track B activation finding; C and F repurpose it as a coupling object to CF11 (weight spectra) and CF6 (gate variance). The load-bearing primitive is the identity of the 2 static channels. **Strongest representative: C1/OHAA** (the alignment test is the cheapest and most decisive coupling of CF3 to CF11). **Cheapest settler: 10-min dot-product projection of outlier channel indices onto W_Q right-singular vectors; if alignment ≥ 0.3 in >50% layers, the coupling is real.**

---

## 4. Frame-Novelty Bonus Advancers

**Path 3 Slot 1 — F3/RMGF** (A2=3, A4=2): The question "does folding the RMSNorm gain g_L into downstream weight matrices reshape the effective singular value spectrum of W_Q?" has no published measurement on any model family. SmoothQuant absorbs scale for activation dynamic-range reasons; QuaRot rotates for outlier suppression. RMGF's question is spectrum-concentration via gauge-absorption — distinct from both. The saturated-frame list contains no cousin. A4=2: the fold is exact (W_Q_gauged = W_Q × diag(g)) and the SVD comparison to CF11's baseline is auditable.

**Path 3 Slot 2 — F4/SSIF** (A2=3, A4=2): The use of softmax shift-invariance as a per-head logit-offset removal tool to sharpen Q/K quantization dynamic range is a fresh application of a well-known algebraic identity. The saturated-frame list contains no cousin. PrefixQuant handles token-wise outliers in the key-value domain; SSIF operates in the pre-softmax logit domain on a different object. A4=2: the shift-invariance identity is exact; the empirical question (does the per-head mean logit have non-trivial norm?) is specified and quantitative.

*Note: TAAR (A2=3, A4=1) was considered for slot 2 but lost to SSIF on A4. TAAR's "H(s)/log(n) bound" for the argmax approximation quality is a heuristic rather than a derived theorem for arbitrary softmax inputs. Under Path 3 rules the two highest-A4 A2=3 ideas advance; RMGF and SSIF both have A4=2.*

---

## 5. Rejected with Rationale

- **R4-B / TERNARY-NVMe** (total 7): A1=1 (arithmetic shows 70B at NVMe is algebraically ruled out at 3 tok/s: 70B at any sub-2-bpw still requires >1 GB/s effective, and NVMe at 3 GB/s gives 0.21 tok/s for a 14 GB model). A2=1 (ternary quantization + NVMe paging is a deployed-system analysis, not a novel frame). A5=1 (PTQTP arXiv:2509.16989 covers bare ternary; the cascade adds only arithmetic). The informational value of the NVMe-wall arithmetic is real but the proposal's load-bearing claim ("cascade path to 70B at 3 tok/s") is settled negative before the experiment runs. Stripped primitive: NVMe sequential bandwidth re-measurement as a first rung survives for other ideas (U1, etc.) but not as a standalone proposal. **Kill note: TERNARY-NVMe-R002 — 70B at 3 tok/s on NVMe algebraically ruled out at any achievable bpw on this hardware; arithmetic is the finding, not a go/no-go gate.**

- **R7-B / HYBRID-TIER-70B** (total 7, FREE SWING): A2=1 (the multi-class knapsack over empirical sensitivity measurements is a systems-engineering optimization, not a novel frame). A5=1 (MDL-Selected Per-Layer bpw was killed at R2/S2 for Compressibility Measures Complexity; HYBRID-TIER extends from per-layer to per-matrix-type but the per-matrix sensitivity ordering is already partially known from CF5/CF11/CF12 and the kill-list notes that mixed-precision quantization is a saturated frame). The FREE SWING status exempts it from CF-tether requirements but not from A2=0/A5=0 quality floor. **Kill note: HYBRID-TIER-70B-R002 — per-matrix knapsack optimization over known sensitivity ordering; mixed-precision quantization frame saturated; no structural finding produced.**

- **U3 / zstd-dict** (total 6): A1=0 (if compression ratio < 3×, the proposal is a net loss; the expected Q4 nibble entropy is near-maximal, making ratio ≈ 1.0–1.1× highly probable). The proposal's own residency arithmetic shows the compression is likely below break-even. Survives as a 30-minute negative-finding experiment but not as an advancer. **Kill note: zstd-dict-R002 — Q4 nibble entropy near-maximal (well-calibrated quantizer forces near-uniform distribution); LZ77 dict finds no locality; aligns with R1/S2 arithmetic-coded kill reasoning.**

- **U6 / IOCP-dispatch** (total 6, FREE SWING, A3=1): The smallest test is ≥1 day ("~200 lines wrapping IOCP dispatch around ggml_graph_compute"). This fails A3≥1 at the stated runtime — the go/no-go is not achievable in ≤8h. A4=1 (the per-layer dispatch overhead claim is plausible for large models but unverified at 1.7B layer times of <100 µs where IOCP transitions would dominate). **Kill note: IOCP-dispatch-R002 — smallest test is >1 day implementation; A3=0 under the stated protocol; at 1.7B layer times the IOCP overhead likely exceeds compute time.**

- **U7 / ReadDirectoryChangesW** (total 9 but A1=0): A1=0 — the mechanism produces no residency or tok/s improvement; it reduces experiment-iteration latency only. The pipeline target is 70B-class on 7.28 GiB at usable tok/s; a hot-swap utility does not move the needle on any of the scoring axes except A5=3 (pre-emption resistant by construction — but A1=0 disqualifies). Also, the mechanism is not a compression idea; it is development tooling. **Kill note: ReadDirectoryChangesW-R002 — A1=0 (no residency/tok/s effect); development-tooling scope, not inference optimization; advancing would distort Path 1 scoring.**

- **A7 / FSMD** (REJECT-CF9, total 10 but A4=1): The transition table of size K × V × L = 65536 × 151936 × 28 = 552 GB is infeasible. The "partial table" relaxation is correct in principle, but the precondition for the power-law coverage claim (that top-10K (state, token) pairs cover 80% of transitions) imports a Zipf assumption that has not been verified for discretized residual-stream states in a transformer. The Zipf distribution of natural language tokens (verified) does NOT imply a Zipf distribution of (discrete residual state, next token) joint frequencies — the discrete residual state is a VQ artifact, not a natural language unit, and its joint distribution with tokens is unmeasured. The mechanism reduces to a lookup-table speculative cache whose effectiveness depends entirely on the unverified coverage claim. A4=1 (the VQ codebook + hash mechanism is specified but the coverage assumption is imported without precondition verification). **Stripped primitive: the K-vector near-duplicate rate measurement (ASAKS A2) is the surviving structural primitive; Stage 4 may want to attack the "discrete residual state coverage" question from an orientation that does not require the full FSM table — e.g., a small-vocabulary discrete attention routing mechanism at a single layer.**

---

## 6. Hand-off to Stage 3

### Track A Advancers

| Rank | ID | Title | Orientation | Total | Stage 3 Pressure-Test |
|------|----|----|----|----|---|
| 1 | F2/CQBS | Cross-Layer W_Q Basis Sharing (FP) | F | 13 | Does cumvar(K=128) ≥ 0.85 hold for the 57344×2048 stacked W_Q matrix, or do per-layer SVDs show rotating bases? Run the experiment before investing in any W_Q cross-layer architecture. |
| 2 | R3-A/XLQB | Cross-Layer W_Q Basis Sharing (Reach) | R | 13 | Redundant with CQBS as a measurement; carry the 70B arithmetic cascade and the fallback plan (per-layer-pair cosine similarity) as Stage 3 input. |
| 3 | R5-A/WDLA-RESCUE | Cross-Scale Affine Surgery (Rescoped) | R | 9 | Is Model Stitching (arXiv:2506.06609) truly training-time only? Confirm the held-out search; if any inference-time analog exists, this kills. Also: does 0.6B→1.7B dimensionality mismatch (1024→2048) doom the frame regardless of data budget? |
| 4 | C5/ROPE | Outlier-Head Excision [FREE SWING] | C | 11 | Prior art check: does A3 (arXiv:2505.12942) or any paper decompose attention into outlier-channel vs residual-channel components? The β_k symmetry claim is specific and testable; confirm no published cousin before Stage 4. |
| +conv | A1/GHPI | Gauge-Fixed Head-Permutation Inference | A | 11 | Convergence C1 representative (Constraint-Alien angle). Pressure-test: does the head-permutation symmetry argument add anything over bare PCA of stacked W_Q? If CQBS settles the cross-layer basis question, does GHPI's gauge-framing produce a different compression primitive or just the same PCA in different language? |

### Track B Advancers

| Rank | ID | Title | Orientation | Total | Stage 3 Pressure-Test |
|------|----|----|----|----|---|
| 1 | F1/JVOC | Joint W_V/W_O Spectral Collapse | F | 12 | Does the fused operator ∑_h W_O^h W_V^h have different spectral properties than per-matrix SVD? This is the decisive experiment closing the attention-weight class. Also: does MLA (DeepSeek-V2) post-training work published since August 2025 cover the fused V-O object without retraining? |
| 2 | C2/CQST | Cross-Layer W_Q Subspace Telescoping | C | 13 | Convergence C1 strongest theoretical representative; the CF2+CF11 composition argument is the most novel. Pressure-test: does the near-parallel residual (CF2) actually FORCE cross-layer alignment, or is it a necessary but insufficient condition? The theory predicts alignment; empirical validation is the only gate. |
| 3 | R2-B/RAOK-SCALE | Activation-Outlier Tiered-Quant Cascade | R | 10 | Does RAOK on Qwen3-1.7B activation quantization deliver ΔNLL ≤ +0.10 nats vs INT4 baseline? Confirm the PDAP infrastructure can be repurposed for this. Also: PrefixQuant (arXiv:2410.05265) handles token-wise outliers; does RAOK's CF3-grounded partition (K=0.1% static vs K=1% dynamic) produce a non-redundant differentiation? |
| 4 | R6-B/ATTN-SPECTRUM | W_V/W_O Rank Cascade | R | 11 | Convergence C2 representative. Cheapest experiment in the set (~20 min). Pressure-test: W_O is the residual-stream output projection; analogous to W_down which CF8 predicts is full-rank. The central uncertainty is whether the head-concatenation structure of W_O's input makes it more concentrated than W_down or as full-rank. |
| +path4 | A6/RGPS | Gauge-Fixed Precision Scheduling | A | 9 | Path 4 elegant-equivalence `gauge-exploitation`. Pressure-test: QuaRot (arXiv:2404.00456) uses Hadamard rotation for outlier suppression; does the PCA gauge choice produce measurably different (higher) spectral concentration than a random Hadamard rotation? If not, RGPS is a principled variant of QuaRot rather than a distinct primitive. |

### Convergence Representatives (additional Stage 3 slots)

- **C1/OHAA** (Track B, total 10): Convergence C3 rep. CF3 × CF11 coupling. Cheapest test in the convergence cluster (10-min dot-product). Stage 3 should confirm whether this alignment measurement was ever published as part of any activation-quantization paper.
- **C3/LGOP** (Track B, total 10): CF6 × CF3 mechanistic composition. Stage 3 pressure-test: is the layer-1 gate anomaly a training-time artifact or a structural property? The column-norm test is cheap (10 min); confirm no published paper on early-layer specialization in SwiGLU models has measured this coupling.

### Frame-Novelty Bonus Slots

- **F3/RMGF** (Track A, total 11): No Stage 3 prior-art risk identified; the spectrum-concentration measurement via gauge-absorption is cleanly novel. Stage 3: confirm whether any rotation-based quantization paper (QuaRot, GPTQ) reports r_99/d before vs after rotation as a diagnostic. If so, RMGF is a replication rather than a new measurement.
- **F4/SSIF** (Track A, total 11): Stage 3 pressure-test: ASCQ (attention-sink channel quantization, Track A R3 deferred) addresses sink tokens in KV; confirm SSIF's per-head logit offset operates on a different domain (pre-softmax logit, not post-softmax KV values) and that no paper has used this identity for dynamic-range reduction specifically.

---

*Total advancing proposals: 24 (Track A: 9 + Track B: 15). Hard rejections: 6 (TERNARY-NVMe, HYBRID-TIER-70B, zstd-dict, IOCP-dispatch, ReadDirectoryChangesW, FSMD). Convergence clusters: 3. Frame-novelty bonus: 2. Elegant-equivalence: 1.*
