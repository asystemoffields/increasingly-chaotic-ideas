# Stage 2 — Judge Report — Run 003

Independent from runs 1 and 2. Kill-list: v2-CHEAP-TEST-001 (cross-layer W_Q subspace sharing, any variant) is hard-dead; any cross-layer W_Q treatment is killed on sight.

---

## 1. Union Score Table

| ID | Orient | Track | Title | A1 | A2 | A3 | A4 | A5 | Total | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| MHRS | R | A | Multi-Head Residency Staircase | 2 | 1 | 3 | 2 | 2 | 10 | ADVANCE |
| RAOK-72 | R | B | R2-Aware Outlier-Keyed Activation Codebook @70B | 2 | 2 | 2 | 3 | 2 | 11 | ADVANCE |
| WAVQ | R | A | W_V / W_O Attention Spectrum Cascade | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE |
| SOBIB-V | R | B | Selective Outlier INT2 + FP16 on W_down | 2 | 2 | 2 | 3 | 2 | 11 | ADVANCE |
| NTQW | R | B | Nested-Tier Q-K-V-O Cascade to 7B [FREE SWING] | 2 | 1 | 2 | 2 | 2 | 9 | ADVANCE (Path 1, wildcard) |
| AERO-QW3 | R | A | AERO-Style SwiGLU Activation Removal Zero-SGD | 2 | 3 | 3 | 2 | 2 | 12 | ADVANCE-frame-novelty |
| DQPM | R | B | Depth-Quantile Precision Map [OUT-OF-ORIENTATION] | 0 | 1 | 0 | 2 | 2 | 5 | REJECT-no-smallest-test |
| C1-QAOC | C | B | Q-Activation Outlier Channel Coupling | 1 | 2 | 3 | 3 | 3 | 12 | ADVANCE-convergence |
| C2-LAYERFOLD | C | A | Layer-1 Gate Fold × W_Q Head-Redundancy Cascade | 1 | 2 | 3 | 2 | 3 | 11 | ADVANCE |
| C3-KDJB | C | B | K-Dependent Jaccard × W_K Spectrum Bridge | 1 | 2 | 3 | 2 | 3 | 11 | ADVANCE |
| C4-GEMBED | C | A | Tied-Embedding Gauge × W_Q Subspace Alignment | 0 | 2 | 3 | 2 | 3 | 10 | REJECT-low-score (A1=0) |
| C5-WUPDOWN | C | B | W_up Full-Rank × W_down Spectrum Bridge | 1 | 2 | 3 | 3 | 3 | 12 | ADVANCE-convergence |
| C6-SPECDRIFT | C | B | Per-Layer Spectral Drift × Codebook Locality [FREE SWING] | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE (Path 1, wildcard) |
| F1-VOFR | F | B | W_V / W_O Fused Composition Rank | 2 | 3 | 3 | 3 | 3 | 14 | ADVANCE-convergence |
| F2-RGAS | F | A | RMSNorm Gain Absorption Spectrum Shift | 1 | 2 | 3 | 3 | 3 | 12 | ADVANCE |
| F3-SRSC | F | A | Softmax Row-Shift Calibration | 1 | 2 | 3 | 3 | 3 | 12 | ADVANCE |
| F4-WAOS | F | B | W_down Activation-Weighted Output Subspace | 2 | 3 | 3 | 3 | 3 | 14 | ADVANCE |
| F5-HPGO | F | A | Head Permutation Group Orbit Collapse | 1 | 3 | 3 | 3 | 2 | 12 | ADVANCE-frame-novelty |
| F6-ASFA | F | A | Attention Sink Fixed-Point Absorption [FREE SWING] | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE (Path 1, wildcard) |
| F7-TEGRA | F | B | Tied Embedding Gradient-Path Rank Asymmetry [OUT-OF-ORIENTATION] | 1 | 3 | 3 | 2 | 2 | 11 | ADVANCE |
| U1-GECP | U | B | GGUF Extent-Coalescence Prefetch | 2 | 2 | 3 | 2 | 3 | 12 | ADVANCE |
| U2-VLPE | U | B | VirtualLock Layer-Pin with Eviction Scheduling | 1 | 2 | 2 | 2 | 3 | 10 | ADVANCE |
| U3-NCCG | U | B | NTFS Compressed-Cluster GGUF Layout | 1 | 2 | 3 | 2 | 3 | 11 | ADVANCE |
| U4-WPLI | U | B | Windows Prefetcher Layout Imprinting | 1 | 2 | 2 | 2 | 3 | 10 | ADVANCE |
| U5-GTBP | U | B | GGUF Tensor-Table B-tree Page Layout | 0 | 1 | 2 | 2 | 3 | 8 | REJECT-low-score |
| U6-NQDS | U | B | NVMe Queue-Depth Saturation for Weight Streaming | 2 | 2 | 3 | 2 | 3 | 12 | ADVANCE-convergence |
| U7-NAWP | U | B | NUMA-Aware CCX-Local Weight Partitioning [FREE SWING] | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE (Path 1, wildcard) |
| A1-GFSM | A | B | Gauge-Fixed Softmax Machine | 0 | 3 | 3 | 2 | 3 | 11 | REJECT-low-score (A1=0; negligible residency) |
| A2-RTGF | A | B | Residual-Stream Sign-Flip Gauge Fold | 1 | 3 | 3 | 3 | 3 | 13 | ADVANCE-elegant-equivalence |
| A3-NTCA | A | A | No-Continuous-Numbers Token Codec [FREE SWING] | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE (Path 1, wildcard) |
| A4-APPA | A | A | Append-Only Persistent KV Accumulator | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE |
| A5-TCRD | A | A | Tropical-Coordinate Residual Dispatch | 1 | 3 | 2 | 2 | 2 | 10 | ADVANCE |
| A6-SSCF | A | B | Seed-Reconstructible Small-Codebook Factorization | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE-frame-novelty |

---

## 2. Per-Advancer Rationale

**MHRS (R/A, 10).** CF11 opened W_Q staircase stratification; MHRS converts that per-layer-unsampled finding into a deployable depth-tiered compression. A3=3 because the smallest experiment (30 min, Qwen3-1.7B) tests whether early-layer W_Q tolerates K=64 below the global K=128 ceiling — NO-GO is itself a structural finding about spectrum depth-uniformity. A4=2 because mechanism is checkable but the staircase direction hypothesis is not yet derived from data (per-layer r_99 unsampled). Primitive: CF11 spectral concentration. Smallest test: W_Q K=64 early + K=128 deep on 1.7B, ΔNLL vs uniform K=128.

**RAOK-72 (R/B, 11).** CF3-grounded three-tier activation channel partitioning applied at W_down. A4=3 because the partition key is the empirically measured K-dependent Jaccard (not a calibration fit), no CF9 theorem import, no CF10 conditioning issue — it's a binary partition rule. A2=2 because the CF3 moat is real and the 70B Reach framing adds frame novelty beyond PrefixQuant. Primitive: CF3 Jaccard profile as column-partition key. Smallest test: RAOK tier on W_down at Qwen3-1.7B, ΔNLL vs uniform INT2 at same bpw.

**WAVQ (R/A, 11).** First proposal to close the W_V/W_O spectrum question opened by AQFKV (CF11). A3=3: 25 min experiment, clean GO on r_99/d ≤ 0.80 threshold; NO-GO extends CF8 to all four attention matrices — a complete map. A4=2 because the cascade claim (joint low-rank operator + free KV cache compression) requires W_V/W_O measured, which is the first gate, but the second-rung reasoning is an analogy to W_K not an audited derivation. Primitive: CF11 spectral concentration, W_V/W_O as the untested attention class. Smallest test: SVD of W_V and W_O all 28 layers + ΔNLL at K=512 for W_V.

**SOBIB-V (R/B, 11).** CF3 K-dependent Jaccard directly grounds the column partition (static/dynamic/dense tiers). A4=3: mechanism is constructive — column assignment rule is a lookup against the existing PDAP measurement, no optimization loop. CF10 does not apply (no fitting). A2=2: the CF3-anchored column partition pre-empets generic INT2+outlier approaches; structural difference from SpQR is the Jaccard-threshold selection criterion. Primitive: CF3 Jaccard measurement as row sensitivity predictor. Smallest test: SOBIB-V column partition on W_down of Qwen3-1.7B at ~2.34 bpw vs uniform INT2.

**NTQW (R/B, 9) — [FREE SWING].** Integration of confirmed rungs (CF11 W_Q K=128 + W_K K=512) with unconfirmed rungs (W_V/W_O from WAVQ, MLP SOBIB-V). Advanced as wildcard because CF-tether requirement is suspended and structural floor is met: A4=2 (mechanism checkable via independence test), A3=2 (Rung 1+2 simultaneous test is 30 min). Residency math for 7B is genuine. Primitive: rung composability of CF11 compressions. Advanced as `[FREE SWING]` — structural floor met, CF-tether requirement suspended for unconfirmed rungs.

**AERO-QW3 (R/A, 12) — frame-novelty.** The zero-SGD SwiGLU activation removal + Khatri-Rao bilinear factorization path is genuinely distinct from the AERO paper (which trains-with-removal). A2=3: no published paper applies AERO-style activation removal to a trained Qwen3 SwiGLU model in a zero-SGD regime; the published cousin (arXiv:2410.13060) covers trained-with-removed-activation only. A3=3: 25 min monkeypatch forward pass test, quantitative ΔNLL threshold. Note: CSMF kill was polynomial substitution (doesn't eliminate W_gate); AERO-QW3 replaces the activation function itself — structurally distinct. Primitive: SwiGLU bilinear structure under activation removal.

**C1-QAOC (C/B, 12) — convergence.** Coupling equation Eq. C1: D_static ∩ colspan(V_Q) overlap ≥ 0.80. A3=3: 20 min experiment from PDAP data already in hand. A4=3: mechanism is a principal-angle measurement, no imported theorems. A5=3: no published paper couples W_Q subspace to activation outlier channel identity in the same coordinate test. Convergence with F4-WAOS and U6-NQDS on the outlier-channel-as-routing-key primitive.

**C2-LAYERFOLD (C/A, 11).** Coupling CF6 (layer-1 36% gate fold anomaly) to CF11 (W_Q spectrum) via the shared early-layer complexity hypothesis. A3=3: 3-min SVD of W_Q[0] alone, clean r_99 vs L14 comparison. Primary risk (the anomaly may be SwiGLU-specific) is addressed by noting it produces a structural finding either way. Primitive: layer-1 anomaly as cross-tensor signal.

**C3-KDJB (C/B, 11).** Couples CF3 K-dependent Jaccard to CF11 W_K spectrum: Spearman ρ(J(t), cos(θ_K(t))) > 0.50. A3=3: 40 min using PDAP infrastructure already run. A4=2: the correlation claim is well-posed but the causal mechanism (outlier channels cause key-subspace rotation) is not derived. Primitive: CF3 per-token Jaccard as proxy for W_K subspace stability.

**C5-WUPDOWN (C/B, 12) — convergence.** Coupling CF5 + CF2 → W_down output-space bottleneck prediction. Eq. C5 (r_99(W_down^T)/d_model < r_99(W_up^T)/d_FFN) is the falsifiable claim. A3=3: 25 min SVD of all 28 W_down matrices, clean inequality test. A4=3: the output-dimension argument is a one-paragraph derivation. A5=3: CF8's full-rank finding has never been tested on W_down; this is the first direct test of whether the class kill extends to the output-projection matrix. Strongest convergence representative for the W_down compressibility cluster.

**C6-SPECDRIFT (C/B, 10) — [FREE SWING].** Per-layer W_Q spectral heterogeneity as codebook allocation key. A3=3: 30 min SVD sweep of all 28 W_Q is the prerequisite and the settling measurement. Wildcard: free rider on the per-layer W_Q sweep that multiple other proposals depend on anyway. Primitive: per-layer r_99 as codebook-size allocation rule. Advanced as `[FREE SWING]` — structural floor met, CF-tether requirement suspended.

**F1-VOFR (F/B, 14) — convergence.** Highest-scored proposal in the union. The fused operator M_L = Σ_h W_O^h W_V^h is the right spectral measurement object (not per-matrix SVD). A4=3: constructive, 10-line auditable (compute M_L per layer, SVD, measure r_99/d). A5=3: no paper measures the spectrum of ∑_h W_O^h W_V^h on any GQA architecture. Convergence with WAVQ (R/A) on the W_V/W_O spectral question — VOFR proposes a theoretically cleaner object (fused composition) while WAVQ proposes per-matrix measurement; the two are structurally compatible experimental rungs. Primitive: residual-stream attention update operator M_L.

**F2-RGAS (F/A, 12).** RMSNorm gain absorption as a pre-compression coordinate change. A4=3: W_Q_gauged = W_Q @ diag(g) is auditable in one line; the spectrum measurement is a direct comparison. A3=3: 20 min with pre-check. A5=3: no paper measures SVD spectrum before vs after RMSNorm gain absorption. Primitive: RMSNorm gauge identity as spectral pre-conditioning step.

**F3-SRSC (F/A, 12).** Softmax shift-invariance exploited to subtract persistent W_Q row-mean offset from Q activations at inference. A4=3: algebraic identity is one-line (softmax(z+c) = softmax(z)), static measurement of row mean requires no forward pass. A3=3: 5 min measurement settles whether the mechanism is load-bearing. A5=3: no paper measures W_Q row-mean magnitude as a Q dynamic-range reduction mechanism. Primitive: softmax shift-invariance applied to W_Q row mean.

**F4-WAOS (F/B, 14).** Calibration-weighted output subspace of W_down (SVD of W_down @ A_L, not of W_down alone) is the key frame innovation that cleanly bypasses CF8 (weight rank is not the same as output subspace rank). A4=3: construction is auditable in 10 lines of NumPy. A3=3: 40 min, go/no-go on cumvar@K=512 ≥ 0.90. A5=3: no published paper applies output-subspace SVD of W_down under SwiGLU-gated activations where CF1 soft-sparsity is the concentration driver. This is the highest-confidence route to W_down compression. Primitive: post-SwiGLU activation soft-sparsity (CF1) concentrating W_down's effective output subspace.

**F5-HPGO (F/A, 12) — frame-novelty.** S_{16} head-permutation alignment before inter-layer delta compression. A2=3: the symmetry group argument (SGD leaves heads in random gauge of S_{n_heads}) has no published treatment in any LLM compression paper; it imports a structural argument from representation theory that DeltaLLM entirely missed. A4=3: bipartite matching on 16×16 cost matrix is a 10-line auditable implementation. A3=3: 15 min pure weight manipulation. A5=2: DeltaLLM is adjacent but misses the permutation alignment step. Frame: representation theory gauge freedom in head indexing.

**F6-ASFA (F/A, 9) — [FREE SWING].** Attention sink fixed-point bias absorption. A3=2 (30 min, but the primary risk about query-dependence of BOS attention weight weakens the go/no-go clarity). A4=2 (calibration mean is a well-posed quantity but CF10 adjacency noted: 28×2048 bias params fit on 200 calibration samples — actually well-conditioned). Advanced as wildcard at the floor. Primitive: attention sink V_BOS as approximately fixed vector. Advanced as `[FREE SWING]` — structural floor met.

**F7-TEGRA (F/B, 11) — [OUT-OF-ORIENTATION].** Frequency-stratified spectral split of tied embed/lm_head bypasses CF12's global full-rank kill with a non-global decomposition. A3=3: 25 min SVD of frequency-split sub-matrices settles whether the spectral split exists. A4=2: the Zipfian frequency → gradient magnitude proxy is a reasonable argument but the connection between training frequency and spectral concentration is not a derived equality. A5=2: CF12 territory but the non-global frequency-stratified approach has no direct prior. Primitive: BPE token frequency as a spectral concentration predictor for tied embed rows.

**U1-GECP (U/B, 12).** `PrefetchVirtualMemory` as explicit double-buffered NVMe weight streaming. A4=2: mechanism is a documented Win32 API; engagement on this hardware is the empirical question (no CF9 import at all). A3=3: 30 min cold-NVMe test with page cache flush; NO-GO produces the v2-confirmed CF13'' NVMe bandwidth number. A5=3: no published LLM serving framework uses `PrefetchVirtualMemory` for layer-ahead weight streaming. Primitive: Windows `PrefetchVirtualMemory` API on deterministic GGUF layer-access order.

**U2-VLPE (U/B, 10).** `VirtualLock` pinning to suppress working-set trimmer eviction. A3=2: 45 min experiment with simulated memory pressure. A4=2: mechanism is documented, privilege question is real but mitigable. Lower impact ceiling than GECP/NQDS. Primitive: Windows working-set trimmer and `VirtualLock`.

**U3-NCCG (U/B, 11).** NTFS cluster compression on GGUF for transparent LZ77 decode-on-read. A3=3: 20 min `compact /C` + size measurement + cold inference timing — clean binary. A4=2: LZ77 is documented NTFS behavior; compression ratio on INT4 weights is the empirical question. A5=3: no published method applies NTFS cluster compression to GGUF weight files. Primitive: `FILE_ATTRIBUTE_COMPRESSED` NTFS LZ77.

**U4-WPLI (U/B, 10).** SysMain Superfetch imprinting for cold-boot TTFT reduction. A3=2: 30 min including reboots; binary (SysMain pre-warms or it doesn't). Orthogonal to weight quality; zero ΔNLL. Primitive: Windows SysMain prefetch database.

**U6-NQDS (U/B, 12) — convergence.** Explicit overlapped IOCP at QD=4 for NVMe throughput saturation. A3=3: standalone 50-line C program, clean throughput ratio; NO-GO produces confirmed NVMe bandwidth for this device. A5=3: no published LLM inference framework uses IOCP for NVMe queue-depth saturation. Convergence cluster with U1-GECP on NVMe throughput maximization; NQDS is lower-level (application-controlled vs OS-hint), producing a joint bandwidth measurement. Primitive: NVMe hardware queue parallelism via Windows IOCP.

**U7-NAWP (U/B, 9) — [FREE SWING].** Q4_K_M scale-separation to enable hardware stride prefetcher on Zen3. A3=2 (1.5h including benchmark setup). A4=2: Zen3 hardware prefetcher stride behavior is documented; whether Q4_K_M interleaved layout defeats it is the empirical question. Primitive: AMD Zen3 hardware stride prefetcher. Advanced as `[FREE SWING]` — structural floor met, CF-tether requirement suspended.

**A2-RTGF (A/B, 13) — elegant-equivalence, `gauge-exploitation`.** Z₂ sign-flip gauge symmetry on the residual stream channels is a lossless pre-quantization redistribution. A4=3: the Z₂ transform is auditable in 2 lines (flip sign convention + update all reading/writing matrices). A2=3: no published quantization pipeline applies discrete gauge-fixing to weight sign conventions before the quantization grid. A3=3: 3 min measurement of per-channel sign statistics. The elegance: the Z₂ orbit is a fundamental symmetry of the computation graph that SGD leaves in random gauge; fixing it before quantization is a computational cost reduction by coordinate choice. Sub-class: `gauge-exploitation`. Path 4 advancer.

**A3-NTCA (A/A, 11) — [FREE SWING].** No-continuous-numbers constraint forces INT8 AVX2 path exploration. A2=3: the constraint-forced frame (tropical + integer-arithmetic inference) has no vocabulary in the ML compression literature. A4=2: mechanism at the single-layer benchmark level is checkable; full-model quality implications are gestured. Advanced as wildcard; the throughput claim on Ryzen Zen3 INT8 vs BF16 is a real and unstated measurement. Primitive: AVX2 INT8 SIMD throughput on Zen3. Advanced as `[FREE SWING]` — structural floor met.

**A4-APPA (A/A, 9).** Append-only KV cache under CF11 W_K rank-512 projection at write time. A4=2: the mechanism is checkable but the claim that K-cache vectors concentrate more than the weight matrix's spectrum is unverified (primary risk flagged). A3=2: 20 min K-projection cosine distance measurement. CF11 tether is real.

**A5-TCRD (A/A, 10).** Tropical dispatch + zero-overhead top-K cache. A2=3: tropical algebra framing of SwiGLU firing is a genuinely novel coordinate system for this problem. A4=2: the top-K identification still requires full W_up GEMV — the savings collapse analysis is partially correct but the salvage mechanism (predictive cache) needs CF3 Jaccard grounding more tightly. A3=2: 15 min measurement of within-sequence consecutive-token top-K cache hit rate.

**A6-SSCF (A/B, 11) — frame-novelty.** Kolmogorov-residual framing: trained weights minus structured random basis, measure whether residual is low-rank. A2=3: no paper has asked "what is the minimum description length of a trained LLM's weights relative to a structured random basis?" in this form; the constraint is doing genuine novel work. A3=2: 30 min for one layer, clean rank measurement. A4=2: the construction is auditable but the claim that W_up - R is low-rank when R has matching block structure is a hypothesis, not a derived identity. Structural floor met; CF8 is the motivating adversarial prior.

---

## 3. Convergence Map

**Convergence Cluster 1 — W_V / W_O spectrum measurement.**

WAVQ (R/A), F1-VOFR (F/B), C5-WUPDOWN (C/B, adjacently).

WAVQ proposes per-matrix SVD of W_V and W_O. F1-VOFR proposes the fused composition M_L = Σ_h W_O^h W_V^h as the correct measurement object. Both orientations (Reach and First-Principles) independently identify the W_V/W_O spectral question as the load-bearing open measurement. F1-VOFR is the strongest representative (A1=14 total; the fused object is the right residual-stream update operator, bypassing per-head/cross-head ambiguity). The two experiments are NOT redundant — F1-VOFR's M_L measurement is both more theoretically motivated and a harder test (if M_L is full-rank, individual matrix SVDs also fail; if individual matrices are concentrated, M_L may not be). Cheapest settler: SVD of M_L = Σ_h W_O^h W_V^h for all 28 layers (~25 min); if r_99/d ≤ 0.75, proceed to ΔNLL measurement.

**Convergence Cluster 2 — W_down compressibility via output-subspace / activation-weighting.**

F4-WAOS (F/B), C5-WUPDOWN (C/B).

Both orientations independently predict W_down is compressible despite CF8 — WAOS via the output-subspace (W_down @ A_L has lower rank than W_down alone), WUPDOWN via the rank-ceiling argument (r_99(W_down^T) ≤ d_model ≪ d_FFN). These are distinct measurements of the same structural question: does W_down's output space concentrate in practice? F4-WAOS is the stronger representative (A1+A4 are both 3; the mechanism is constructive and explicitly bypasses CF8 via the output-vs-weight-rank distinction). Cheapest settler: cumvar@K of Delta_L = W_down @ A_L on 200-token calibration forward pass (~40 min). A high r_eff from F4-WAOS also answers C5-WUPDOWN's prediction.

**Convergence Cluster 3 — Outlier-channel identity as routing / projection key.**

C1-QAOC (C/B), U6-NQDS (U/B adjacently), A5-TCRD (A/A).

C1-QAOC couples W_Q subspace to static outlier channels (CF3 × CF11). A5-TCRD uses top-K firing patterns as a cached dispatch key (CF1 × CF3). Both independently identify the per-token outlier structure as a routing primitive. The W_Q subspace coupling (C1) is the more tightly specified claim. Cheapest settler: C1-QAOC's 20 min principal-angle measurement between D_static and colspan(V_Q). If the coupling holds, it strengthens A5-TCRD's dispatch-key design.

**Convergence Cluster 4 — NVMe throughput maximization via OS substrate.**

U1-GECP (U/B), U6-NQDS (U/B).

Both propose to close the gap between NVMe rated throughput and demand-paging QD=1 baseline. GECP uses `PrefetchVirtualMemory` (OS hint-based); NQDS uses IOCP (application-controlled overlapped I/O). They are complementary rungs at different abstraction levels, and both depend on the same foundational measurement: actual QD=1 vs QD=4 NVMe throughput on this device. U6-NQDS is the stronger representative (A1=2 with QD-saturation argument; A3=3 with standalone program test). Cheapest settler: 50-line C program measuring sequential read throughput at QD=1 and QD=4 (~2h). This measurement settles whether substrate-level NVMe optimization is load-bearing for the 70B inference scenario.

---

## 4. Frame-Novelty Bonus Advancers

**AERO-QW3 (R/A, A2=3).** Frame: zero-SGD SwiGLU activation removal + bilinear Khatri-Rao factorization. Published cousin arXiv:2410.13060 (AERO) covers trained-with-activation-removed models; the zero-SGD application to a trained Qwen3 SwiGLU is genuinely open. The CSMF kill (polynomial substitution) is a different mechanism. A4=2: monkeypatch test is auditable. The saturation boundary: AERO's paper does not claim their method works post-training on SwiGLU without fine-tuning; that claim is unstated as of 2026-05-09.

**F5-HPGO (F/A, A2=3).** Frame: S_{n_heads} permutation group orbit canonicalization before inter-layer delta compression. The representation-theory argument (SGD leaves heads in random gauge of the head-permutation symmetry group) has no published treatment in any compression paper. DeltaLLM is adjacent but uses raw layer deltas without head alignment. A4=3: Hungarian algorithm on 16×16 cost matrix is fully auditable. The saturation boundary: no paper we can identify explicitly considers the head-permutation gauge as a compressible degree of freedom in a trained transformer.

Tie-break for third frame-novelty slot (not advanced): F1-VOFR (A2=3) and A6-SSCF (A2=3). F1-VOFR advances via convergence cluster 1 (Path 2). A6-SSCF advances on its own frame-novelty A2=3 but is the lower-A4 of the tied pair (A4=2 vs F5-HPGO A4=3). Only 2 Path 3 slots available; AERO-QW3 and F5-HPGO fill them. A6-SSCF is also independently in the Path 1 table at 11 total.

---

## 5. Rejected with Rationale

- **DQPM (R/B, [OUT-OF-ORIENTATION]).** A3=0: no smallest experiment — DQPM declares itself a synthesis/specification whose outcome is determined by other ideas' experiments. An idea that produces numbers only derivatively from other ideas' go/no-go has no independent falsifiability. The surviving structure (deployment floor arithmetic) is useful as a convergence note; Stage 4 should maintain it as a running calculation, not a standalone experiment. Kill-list note: "DQPM-style deployment synthesis arithmetic — advance as a live calculation in SELECTED.md, not as a standalone Stage 3 idea."

- **C4-GEMBED (C/A, 10).** A1=0: residency gain at 70B is 95 MB flat (0.24% of model weights), explicitly calculated in the proposal itself. The compute win at layer 0 is real but marginal; this is not a residency mechanism for the 7.28 GiB target. The coupling equation (Eq. C4, embedding covariance alignment with W_Q[0] subspace) is an interesting structural question but the payoff is negligible. Surviving primitive: embedding covariance alignment measurement is a cheap measurement that C1-QAOC's principal-angle machinery could extend; Stage 4 may ask whether the W_E covariance and the W_Q subspace coupling motivates a joint coordinate system for the embedding + first-layer projection.

- **U5-GTBP (U/B, 8).** A1=0 effectively (metadata overhead is 28 µs/token in the worst case — 0.1% of GEMV time at 1.7B; irrelevant at 70B where GEMV dominates). The idea is correct and technically valid; it is a free 1-hour engineering fix that belongs in implementation notes rather than as a research slot. Surviving primitive: GGUF metadata access overhead — addressed as a free engineering followup to GECP (U1), not as a standalone Stage 3 idea.

- **A1-GFSM (A/B, 11).** A1=0: residency gain is explicitly computed as 14 MB on a 3.3 GB model (0.4%) — below the 5% threshold for A1=1. The softmax shift-invariance gauge argument is mathematically correct and A4=2, but the mechanism produces no meaningful residency savings. The surviving primitive: softmax shift-invariance as a gauge degree of freedom in W_Q representation — this is the same symmetry exploited by F3-SRSC (which produces a meaningful runtime effect on Q activation dynamic range). Stage 4 may want to check whether F3-SRSC's row-mean subtraction and A1-GFSM's gauge-fixing are the same operation under different names (if so, they compose cleanly). Kill-list note: "A1-GFSM pure softmax gauge compression — residency gain is sub-0.5% at any model scale; mechanism folds into F3-SRSC's Q activation dynamic range work."

---

## 6. Hand-off to Stage 3

### Track A advancers (top 4 by score + convergence reps + frame-novelty):

1. **F4-WAOS** (F/B repurposed as A track mechanism, 14) — "Does post-SwiGLU activation concentration (CF1) collapse W_down's output subspace below rank-512? Stage 3: verify this is not prior-arted by any GPTVQ variant that measures output Hessian, and check whether 200-token calibration is conditioning-safe per CF10."

2. **F1-VOFR** (F/B, 14) — "Does M_L = Σ_h W_O^h W_V^h have r_99/d ≤ 0.75? Stage 3: check whether any MLA-adjacent paper measures the fused composition operator vs per-matrix decomposition on GQA models."

3. **AERO-QW3** (R/A, 12, frame-novelty) — "What is ΔNLL of zero-SGD SwiGLU activation removal on Qwen3-1.7B? Stage 3: find any 2025-2026 paper applying AERO-style removal to a pre-trained SwiGLU model without fine-tuning."

4. **F5-HPGO** (F/A, 12, frame-novelty) — "Do optimal inter-layer head permutations reduce W_Q delta Frobenius norm by ≥15%? Stage 3: check whether DeltaLLM or any successor paper applies head-alignment before differencing."

5. **A2-RTGF** (A/B, 13, elegant-equivalence `gauge-exploitation`) — "Is the Z₂ sign-flip distribution bimodal (many channels near 0 or 1) or balanced? Stage 3: check whether SmoothQuant/OS+ or any successor paper applies discrete sign-convention gauge fixing before quantization grid placement."

6. **WAVQ / convergence cluster 1 rep** (R/A, 11) — "W_V and W_O spectral concentration: does r_99/d ≤ 0.80 hold? Stage 3: find any post-training joint attention compression paper that measures all four of W_Q/W_K/W_V/W_O on the same model."

### Track B advancers (top 4 by score + convergence reps):

1. **F4-WAOS** (F/B, 14) — see above.

2. **F1-VOFR** (F/B, 14) — see above.

3. **RAOK-72** (R/B, 11) — "Does CF3-derived three-tier column partition on W_down reduce ΔNLL by ≥0.2 nats vs uniform INT2 at matched bpw? Stage 3: check whether any 2025-2026 paper applies per-channel Jaccard-stability threshold (not Hessian) as a quantization precision selector."

4. **SOBIB-V** (R/B, 11) — "Does W_down column partition keyed on K-dependent Jaccard achieve ≤+0.8 nats at 2.34 bpw? Stage 3: distinguish from SpQR (Hessian-weighted outlier detection) — the structural difference is the static vs dynamic Jaccard threshold criterion."

5. **C5-WUPDOWN / convergence cluster 2 rep** (C/B, 12) — "Does r_99(W_down^T)/d_model < r_99(W_up^T)/d_FFN hold for ≥20/28 layers? Stage 3: check CF8 literature for any weight that has been tested for spectrum concentration relative to its output dimension."

6. **U1-GECP** (U/B, 12) — "`PrefetchVirtualMemory` layer-ahead prefetch: does cold-NVMe latency drop ≥3× vs demand-paging? Stage 3: check ik_llama.cpp source for existing mmap prefetch flags before assuming the baseline is unoptimized."

### Notable non-advancers to watch:
- MHRS advances as Path 1 Track A but should be treated as a precondition measurement for NTQW; the staircase direction hypothesis is inverted from the proposal's assumption (Stage 3 should flag that early layers may need MORE capacity for raw embedding processing, not less).
- U6-NQDS (12) and U3-NCCG (11) are strong Track B substrate candidates; budget permitting they should run alongside GECP as the NVMe bandwidth measurement cluster.
