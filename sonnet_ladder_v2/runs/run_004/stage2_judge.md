# Stage 2 Judge — Run 004

Union of 5 orientations: R (7 proposals), C (6 proposals), F (6 proposals), U (6 proposals), A (7 proposals) = 32 proposals.
Kill list applied: v2-CHEAP-TEST-001 kills any cross-layer W_Q variant. CF9 hard gate enforced. CF10 conditioning check applied. CF8 boundary enforced.

---

## 1. Union Score Table

| ID | Orientation | Track | Title | A1 | A2 | A3 | A4 | A5 | Total | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| AWCJR | R | B | Attention-Weight Cascade: W_Q+W_K+W_V+W_O Joint | 2 | 1 | 3 | 2 | 2 | 10 | ADVANCE |
| L1AW | R | B | Layer-1 SDZC + Global Attention Compression | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE-convergence |
| RATJ | R | B | RAOK + Attention-Tier Joint Cascade | 2 | 2 | 2 | 2 | 2 | 10 | ADVANCE |
| ULHC | R | B | Untied lm_head Spectrum + SVLD Revival | 2 | 2 | 2 | 2 | 2 | 10 | ADVANCE |
| WDRS | R | B | W_down Rank Sweep: Closing CF8 | 1 | 1 | 3 | 3 | 3 | 11 | ADVANCE |
| HRQK | R | A | Head-Redundancy Routing: MLA-Style Q-K Post-Training [FREE SWING] | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE-wildcard |
| VOKV | R | B | W_V/W_O Spectrum Joint + KV-Cache Compression | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE-convergence |
| C-AQOD | C | B | Attention-Outlier Disjoint Partition | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE-convergence |
| C-VKSD | C | B | V-K Spectral Delta Composition | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE-convergence |
| C-SDLC | C | B | SDZC-Q Layer-1 Composition Cascade | 1 | 2 | 2 | 2 | 3 | 10 | REJECT-low-score |
| C-RBHD | C | B | Residual-Band Head-Diversity Decoupling | 1 | 2 | 2 | 2 | 3 | 10 | ADVANCE-convergence |
| C-WVKQ | C | A | Spectral Ladder Joint KVQ Operator [FREE SWING] | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE-wildcard |
| C-ODWQ | C | B | Outlier-Drift Weighted W_Q Layer Budget | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE-convergence |
| F-WVOS | F | A | W_V/W_O Spectral Structure + Algebraic Fold | 2 | 2 | 3 | 3 | 2 | 12 | ADVANCE |
| F-GFQO | F | A | Gauge-Fixing Residual Stream via Q-K Subspace | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE-frame-novelty |
| F-AERO-Q3 | F | A | AERO-Style W_gate Elimination Feasibility Probe | 2 | 2 | 3 | 3 | 2 | 12 | ADVANCE |
| F-SPECA | F | B | Spectral Equivalence W_K via W_Q Alignment | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE-convergence |
| F-WNORM | F | B | W_down Spectral Probe: Closing CF8 | 1 | 1 | 3 | 3 | 3 | 11 | ADVANCE-convergence |
| F-RMNQ | F | A | RMSNorm Scale Absorption: Gauge-Fix | 1 | 2 | 3 | 2 | 2 | 10 | REJECT-low-score |
| F-TIED | F | B | Tied Embedding Gauge Rotation [FREE SWING] | 1 | 2 | 2 | 1 | 2 | 8 | REJECT-low-score |
| WSFI | U | B | Windows SuperFetch Footprint Inversion | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE-frame-novelty |
| NCZC | U | B | NTFS Compressed Clusters as Zero-Copy Decode Cache | 1 | 2 | 3 | 2 | 3 | 11 | ADVANCE |
| WWTE | U | B | Windows Working Set Trimmer as Eviction Oracle | 1 | 2 | 3 | 2 | 3 | 11 | ADVANCE |
| GTBP | U | B | GGUF Tensor Offset B-Tree as Page-Fault Sequencer | 2 | 2 | 2 | 2 | 3 | 11 | ADVANCE |
| DFSQ | U | B | Denormal Float Sentinel as Quantization Tier Marker | 0 | 2 | 3 | 2 | 3 | 10 | REJECT-low-score |
| NNPA | U | B | NVMe Namespace Partitioning as Weight Tier [FREE SWING] | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE-wildcard |
| A-TROP | A | B | Tropical Weight Dispatch | 1 | 3 | 3 | 2 | 3 | 12 | ADVANCE |
| A-SEED | A | A | 10 MB Seed Weight Synthesis | 1 | 3 | 2 | 1 | 3 | 10 | ADVANCE-frame-novelty |
| A-ALOG | A | A | Append-Only Log Attention | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE |
| A-REGM | A | A | Register-Only Residual Decode | 1 | 2 | 2 | 2 | 3 | 10 | ADVANCE |
| A-FSMT | A | A | Finite-State Decode Machine [FREE SWING] | 1 | 3 | 2 | 1 | 3 | 10 | REJECT-low-score |
| A-GFWV | A | A | Gauge-Fixed W_V Folding | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE-convergence |
| A-CADH | A | A | Content-Addressable Attention Head Dispatch | 2 | 3 | 3 | 2 | 2 | 12 | ADVANCE |

---

## 2. Per-Advancer Rationale

**AWCJR (R/B, 10)** — Carried by A1=2 (joint four-matrix attention compression grounded in CF11) and A3=3 (W_V/W_O spectrum measurement is a 30-min binary settler gating the whole cascade). A4=2: the compounding-additivity assumption is explicit and flagged as the risk — checkable via the interaction measurement in RATJ. Grounded in CF11 (W_Q K=128, W_K K=512 confirmed); W_V/W_O are the open gates. Primitive: per-layer SVD truncation of all four attention projections. Smallest test: W_V[L14] r_99/d < 0.80 in ~5 min.

**L1AW (R/B, 10)** — Advances as convergence representative for Cluster C2 (Layer-1 structural anomaly coupling). CF6 × CF11 composition: layer-1 MLP softness (36% foldable) + W_Q global compression. Strongest convergence claim is per-layer compression budget differentiation. A3=3: W_Q[L1] SVD is a 5-min binary settler. Key primitive: per-layer W_Q effective rank; most informative if layer-1 shows r_99/d < 0.50. Smallest test: SVD of W_Q[layer 1] vs L14 reference.

**RATJ (R/B, 10)** — A1=2: CF3 × CF11 cross-finding stack (RAOK activation-tier + attention SVD), orthogonal objects with sub-additive composition as the falsifiable claim. A3=2 (interaction test is ~40 min, binary but requires RAOK as prerequisite). Pre-emption resistance A5=2: RAOK grounded in our private CF3 Jaccard numbers. Primitive: joint ΔNLL of RAOK + W_Q K=128 vs sum of independent effects. Smallest test: compose RAOK-only + W_Q-K128-only perplexities, then measure joint ΔNLL.

**ULHC (R/B, 10)** — A1=2: untied lm_head at Qwen3-8B per CF12's explicit open case. A3=2: 60-min blockwise SVD + NLL sweep (Godey & Artzi prediction is the falsifier). The CF12 kill is tied configurations; this is the sequel. Primitive: blockwise SVD of 151936×4096 untied lm_head; hypothesis: r_99/d < 0.40. Smallest test: SVD alone (45 min), gated on fitting lm_head.weight (1.24 GB) into RAM separately.

**WDRS (R/B, 11)** — A3=3: closes the MLP weight characterization with a clean binary result either way. A4=3: mechanism is exactly the R3 W_gate sweep protocol, already validated — constructive and auditable. A5=3: no paper reports W_down singular spectrum for Qwen3; private empirical finding either confirming CF8 or opening a new path. Advances on total score (11). Primitive: all-layer W_down rank sweep at K ∈ {2048, 1024, 512, 256, 128}. Smallest test: identical protocol to R3 Idea 12 (~30 min).

**HRQK [FREE SWING] (R/A, 11)** — CF-tether requirement suspended. Advances on total. A3=3: within-layer head PCA at L14 is a 5-min binary settler (K_basis=64 ≥ 90% head variance). Correctly distinguished from v2-CHEAP-TEST-001 (within-layer head pooling, not cross-layer). MLA-style post-training factorization grounded in CF11's 16-heads-to-128-dim head-redundancy finding. Primitive: PCA over 16 heads of W_Q[L14]. Advanced as [FREE SWING] — structural floor met, CF-tether requirement suspended.

**VOKV (R/B, 11)** — Advances as convergence representative for Cluster C1 (W_V/W_O spectrum characterization). A1=2: the W_V algebraic compression → KV-cache dimensionality reduction coupling is the load-bearing primitive — novel coupling. A3=3: W_V[L14] SVD is a 5-min binary settler gating the entire cascade. A5=2: private Qwen3 V+O spectral measurement. Primitive: W_V spectrum → compress KV cache storage in W_V's basis algebraically. Smallest test: SVD of W_V[L14] and W_O[L14] (~10 min).

**C-AQOD (C/B, 10)** — Advances as convergence representative for Cluster C3 (outlier × attention-spectrum disjointness). Coupling equation is explicit: cos²(e_i, V_{128}) < 0.05 for static outlier channels vs W_Q subspace. CF3 (K=0.1% static) × CF11 (W_Q K=128 subspace). A3=3: 20-min measurement of alignment angle between 2 static channels and W_Q top-128 right singular vectors. Primitive: disjointness of activation outlier channels and W_Q projection subspace. Smallest test: alignment angle computation (data already in hand from PDAP + AQFKV).

**C-VKSD (C/B, 10)** — Advances as convergence representative for Cluster C1. Coupling: W_K/W_V subspace gap < 0.3 as precondition for joint KV projection. A3=3: 30-min subspace-gap measurement. CF11 (W_K r_99/d≈0.79) × W_V (untested). Primitive: shared input subspace of W_K and W_V within a layer. Smallest test: F-norm of projection difference per layer.

**C-RBHD (C/B, 10)** — Advances as convergence representative for Cluster C4 (head-diversity vs weight-redundancy). A3=2 (60-min, requires PDAP forward pass reuse, go/no-go numerical but slower). CF11 (head-redundancy) × CF3 (per-token outlier dynamics). Falsifiable coupling: per-head outlier-routing Jaccard < 0.3 vs ≥ 0.5. Either way this disambiguates CF11's head-redundancy interpretation (which Stage 3 needs). Primitive: per-head attention-routing diversity across the token-dynamic outlier set.

**C-WVKQ [FREE SWING] (C/A, 11)** — CF-tether requirement suspended. A3=3: 40-min joint SVD of stacked [W_Q;W_K;W_V] at 3 representative layers, with a clean F-norm ratio threshold. The Track A arch change (one shared GEMV → three cheap readouts) is a genuine computation graph transposition. Advances as [FREE SWING] — structural floor met, CF-tether requirement suspended.

**C-ODWQ (C/B, 10)** — Advances as convergence representative for Cluster C5 (outlier-spread × W_Q rank correlation). A3=3: 10-min Spearman correlation over 28 layers using data already in hand from PDAP + AQFKV. CF3 (per-layer outlier spread) × CF11 (per-layer W_Q r_99). Primitive: budget stratification signal from outlier-spread/rank correlation. Smallest test: Spearman ρ on existing data.

**F-WVOS (F/A, 12)** — Top score in this run. A4=3: the W_V fold into W_O is a fully constructive algebraic identity (head_output = softmax(QK^T) · V · W_O ≡ softmax(QK^T) · (X W_V) · W_O; absorb low-rank W_V decomposition into W_O exactly). Auditable in 5 lines. A3=3: 30-min SVD + 10-min NLL eval, binary settler tied to CF11's boundary (r_99/d ≤ 0.80). CF11-class measurement on untested W_V/W_O. Primitive: W_V low-rank absorption into W_O — exact algebraic fold contingent on W_V spectrum. Smallest test: 28-layer W_V SVD + rank-256 fold NLL.

**F-GFQO (F/A, 11) [frame-novelty bonus]** — A2=3: gauge-freedom framing for residual-stream coordinate alignment is genuinely fresh relative to ML compression literature; no direct published cousin applies this frame to exposing block-sparse attention subspaces for inference cost reduction. A4=2: CF9 precondition correctly identified (RMSNorm breaks arbitrary rotation symmetry); mitigation via RMSNorm-compatible rotation class is specified. A3=2: 25-min SVD of stacked [W_Q;W_K;W_V] per layer, quantitative go/no-go threshold var@1600 ≥ 0.99. Advances under Path 3 (frame-novelty bonus, A2=3).

**F-AERO-Q3 (F/A, 12)** — A4=3: the silu→identity substitution is constructive and auditable — forward pass interception, no weight modification, 10-min runtime, clean ΔNLL measurement. Critically, the KILL_LIST (R3) explicitly opened this as untested on Qwen3. A3=3: binary settler in ≤10 min — the cleanest experiment in this entire run. A1=2: W_gate elimination is 20% of Qwen3-1.7B model if successful. Primitive: linear-region fraction of SwiGLU gate pre-activations on Qwen3 calibration distribution. Smallest test: replace silu with identity in forward pass, measure ΔNLL on 512-token eval.

**F-SPECA (F/B, 10)** — Advances as convergence representative for Cluster C1. Within-layer Q/K joint basis claim. A3=3: 45-min joint SVD + NLL eval. CF11 (W_Q, W_K independent spectra) → per-layer joint compression. Primitive: within-layer W_Q/W_K subspace overlap. Correctly NOT cross-layer.

**F-WNORM (F/B, 11)** — Advances as convergence representative for Cluster C6 (W_down rank characterization = CF8 completion). A4=3: W_down SVD is the same protocol as R3 Idea 12 and R4-B CLUBS rank sweeps; constructive and auditable. A3=3: 20-min binary settler, confirms or closes CF8 on all three MLP matrices. A5=3: no paper reports Qwen3 W_down singular spectrum. Primitive: W_down effective rank under output dimension (max rank 2048, not 6144). Note: F-WNORM and WDRS are the same experiment — advance F-WNORM as the canonical representative (marginally stronger A4 rationale on the d_out vs d_in framing).

**WSFI (U/B, 11) [frame-novelty bonus]** — A2=3: SuperFetch learning-loop exploitation for LLM inference is a genuinely fresh frame not present in any published LLM inference paper; the Windows SysMain prefetch-learning primitive has no ML compression cousin. A4=2: mechanism is correctly specified (slab layout → convergent PF trace → learned prefetch); the 3-run learning period is specific. A3=2: 2-hr experiment with per-layer NVMe latency comparison; go/no-go is quantitative (run-10 latency ≤ 70% of run-3) but requires 2 hrs. A5=3: private Windows-specific hardware measurement. Advances under Path 3 (frame-novelty bonus, A2=3). Note: R1/S2 "OS-Paging-Aware Weight Layout" was killed as "doesn't reduce bpw, just engineers paging." WSFI is structurally distinct (dynamic learning loop, not static layout); not a kill-list re-tread.

**NCZC (U/B, 11)** — A3=3: 30-min binary test (NTFS compression ratio on sorted vs unsorted GGUF). A5=3: no published LLM work uses NTFS cluster compression with layout optimization. A4=2: mechanism correctly invokes NTFS LZ77 compression unit granularity; the sequential-scan prerequisite is correctly identified. Primitive: GGUF row-sorted by magnitude tier → NTFS LZ77 compression ratio improvement. Smallest test: compact /c on two GGUFs, measure sizes.

**WWTE (U/B, 11)** — A3=3: 1-hr test of QueryWorkingSetEx residency across 10 inference passes. A5=3: no published method uses OS working-set age as hot-weight oracle (Apple LLM-in-a-Flash uses activation statistics). A4=2: substrate primitive precisely named (QueryWorkingSetEx + VirtualLock), engagement claim is falsifiable. Primitive: Windows OS LRU as zero-cost hot-weight oracle. Smallest test: 15% of GGUF pages persistently resident across 10 passes.

**GTBP (U/B, 11)** — A1=2: IOCP 2-layer lookahead directly targets the layer-fetch NVMe bottleneck at 70B scale. A3=2 (2-hr test including harness writing; go/no-go numerical but implementation-dependent). A5=3: confirmed absence from llama.cpp / ik_llama.cpp source. A4=2: IOCP OVERLAPPED async I/O is precisely specified. Primitive: B-tree-keyed deterministic async NVMe prefetch for GGUF layers. Smallest test: 50-line C harness measuring IOCP vs synchronous per-layer latency.

**NNPA [FREE SWING] (U/B, 11)** — A2=3: NVMe namespace partitioning for GC stall elimination in LLM inference is a genuinely fresh substrate frame; no published ML inference paper invokes NVMe namespace management. A3=2: 2-hr stochastic GC stall measurement (go/no-go is p95 latency with writes ≥ 1.3× without writes). Advances as [FREE SWING] — structural floor met, CF-tether requirement suspended.

**A-TROP (A/B, 12)** — A2=3: tropical semiring applied to W_down channel-dominance dispatch is fresh relative to all kill-list and published-landscape entries. A3=3: 30-min activation-stats pass — the dominant-channel stability fraction is binary and fast. A5=3: no published LLM method uses tropical algebra for activation-based routing. A4=2: mechanism correctly invokes tropical max-dominance (no CF9 failure — no imported theorem with violated precondition; pure combinatorial structure). Primitive: post-SwiGLU dominant-channel stability fraction in W_down rows. Smallest test: compute argmax_j(W_down[i,j] × h_mlp[j]) stability over 200 calibration tokens.

**A-SEED (A/A, 10) [frame-novelty bonus]** — A2=3: the 10 MB hypernetwork synthesis frame is genuinely outside the published compression literature for billion-parameter models; the inverse-direction hypernetwork experiment has no LLM cousin. A4=1: the synthesis function's quality bound is not derived analytically — it is a fitted approximation whose accuracy depends on the effective Kolmogorov complexity of trained weights (the question itself). A3=2 (2-hr offline fitting; go/no-go is quantitative cos-sim threshold but requires fitting F). Advances under Path 3 (frame-novelty + A4 ≥ 1 floor met; the unfamiliarity of the frame is exactly the Path 3 signal).

**A-ALOG (A/A, 9)** — A3=2: 3-hr experiment (100 long-context forward passes to fit the decay curve); go/no-go is numerical (λ R² ≥ 0.80) but large. A4=2: RoPE geometric-decay approximation is checkable; the cold-journal error bound is derivable from λ. Grounded: RoPE's position-based score decay is a measurable distribution property. Total 9 just clears the 9/15 threshold. Primitive: RoPE attention geometric-decay rate λ as a function of position distance. Smallest test: attention entropy curve on 100 × 4K forward passes.

**A-REGM (A/A, 10)** — A4=2: the L1-resident GEMV streaming schedule is constructive (residual 8 KB in L1, weight columns streamed from NVMe). A3=2: 4-hr implementation + measurement; go/no-go on peak RSS ≤ 512 MB. A5=3: no published analysis of register-only residual streaming; also re-derives CF13 (NVMe read bytes per token) as a v2-native measurement. Primitive: L1-resident residual stream + NVMe-streamed weight columns. Smallest test: single-layer L3-mlock GEMV streaming implementation with RSS monitoring.

**A-GFWV (A/A, 11)** — Advances as convergence representative for Cluster C1 (W_V/W_O characterization). A3=3: 30-min W_V·W_O product SVD, binary settler (mean r_99/d ≤ 0.70). A1=2: if W_V·W_O product is rank-64, 50% per-layer saving on {W_V,W_O} in 70B is meaningful. A4=2: product-fold algebraic identity is checkable (absorbed product operator). Distinct from v2-CHEAP-TEST-001: within-layer product, not cross-layer stacking. Primitive: W_V·W_O per-head product rank structure.

**A-CADH (A/A, 12)** — A2=3: content-addressable dispatch keyed on SVD-top-32 signature is a fresh frame with no ML inference cousin. A3=3: 10-min per-head cos-similarity measurement at L14 is one of the fastest binary settlers in this run. A1=2: within-layer cross-head W_Q sharing is grounded in CF11's 16-heads-to-128-dim finding — 16× bandwidth reduction on W_Q if confirmed. A4=2: BLAKE3-keyed dispatch is precisely specified, no broken theorem imports. Primitive: within-layer per-head W_Q matrix similarity as a content-addressable dispatch key.

---

## 3. Convergence Map

**Convergence C1 — W_V and W_O spectral characterization as the unlocking primitive.**
VOKV (R: KV-algebraic compression), C-VKSD (C: W_K/W_V joint projection), F-WVOS (F: algebraic W_V→W_O fold), F-SPECA (F: per-layer Q/K joint basis), C-WVKQ (C [FREE SWING]: joint KVQ operator), A-GFWV (A: W_V·W_O product fold) all converge on W_V and W_O spectral structure as the gating measurement. Strongest representative: F-WVOS (A4=3, constructive algebraic identity, auditable in 5 lines). Cheapest settler: SVD of W_V[L14] and W_O[L14] (~10 min, already flagged in AQFKV follow-up list). This measurement simultaneously activates or kills six cascades.

**Convergence C2 — Cross-finding CF3×CF11: outlier dynamics composing with attention spectrum.**
C-AQOD (disjoint partition), C-RBHD (head-diversity routing), C-ODWQ (outlier-spread × W_Q rank budget stratification) all compose CF3 (per-token outlier dynamics) with CF11 (W_Q spectrum). RATJ (R) also enters this cluster as the deployment-scale composition. Strongest representative: C-ODWQ (A3=3, 10-min Spearman on existing data — fastest settler in the cluster). The cheapest settlement: compute Spearman ρ(outlier_spread_L, r_99(W_Q^L)) over 28 layers using PDAP + AQFKV data already in hand. A positive ρ > 0.3 motivates all three budget-stratification ideas simultaneously.

**Convergence C3 — W_down rank characterization = CF8 MLP boundary completion.**
WDRS (R) and F-WNORM (F) are structurally identical proposals: all-layer W_down SVD sweep, same protocol as R3/R4. Strongest representative: F-WNORM (slightly richer A4 framing on d_ffn > d_model shape). Cheapest settler: same experiment (~20-30 min). Run once, credit both. This closes CF8 and gates A-TROP's secondary hypothesis (W_down activation-weighted effective rank).

**Convergence C4 — W_V·W_O PRODUCT rank as a richer object than individual spectra.**
A-GFWV (A: gauge-fix fold) and F-WVOS (F: algebraic fold) converge on the W_V·W_O product as the compression-enabling object. F-WVOS decomposes W_V independently then absorbs into W_O; A-GFWV decomposes the product directly. These are related but not identical: the product can be lower-rank than either individual matrix (multiplicative rank collapse). Cheapest settler: compute W_V·W_O per head for L14 and measure r_99 (~30 min), after confirming W_V spectrum in C1.

**Convergence C5 — Within-layer W_Q head-similarity structure (disambiguating CF11).**
HRQK (R [FREE SWING]: within-layer head PCA), A-CADH (A: cross-head cos-similarity dispatch), C-RBHD (C: per-head outlier-routing Jaccard) all need to know whether CF11's global K=128 finding reflects per-head similarities or diverse orthogonal heads. Cheapest settler: A-CADH's 10-min per-head pairwise cos-similarity at L14. If cos-sim ≥ 0.80 for ≥ 50% of pairs, heads are similar (CADH and HRQK both activated). If not, CF11 arises from orthogonal heads (RBHD's diversity hypothesis validated).

---

## 4. Frame-Novelty Bonus Advancers

**Slot 1: F-GFQO — Gauge-freedom/residual-stream block-sparsity frame.**
The gauge-fixing argument applied to residual-stream coordinates to expose attention-invisible dimensions has no published cousin in LLM compression literature. QuIP/QuIP# use random rotations for incoherence (opposite goal). The frame is from mathematical physics / differential geometry (gauge symmetry in field theories) imported to expose a coordinate-level structure in neural network inference. The saturated-frame list contains no named cousin. A4=2 (CF9 precondition correctly identified and mitigation specified). Frame genuinely past saturation boundary as of 2026-05.

**Slot 2: WSFI — Windows SuperFetch learning-loop for LLM inference.**
The SysMain CLR/PC-based prefetch learning is a Windows-specific substrate primitive that has never been invoked in any published LLM inference paper. Apple LLM-in-a-Flash uses mmap demand paging; no published system exploits OS prefetch-histogram learning as a convergent schedule adapter. The kill-list OS-Paging-Aware Layout (R1/S2) was a static layout trick; WSFI's dynamic learning loop is structurally distinct. A4=2 (mechanism correctly specified, 3-run convergence period is the engagement condition). Not in any saturated frame.

A-SEED was considered for Slot 2 but loses to WSFI on A4 (A-SEED A4=1 vs WSFI A4=2); Path 3 requires strongest A4 among the A2=3 candidates.

---

## 5. Rejected with Rationale

- **C-SDLC (C/B, 10)** — REJECT-low-score. A3=2 (45-min, requires ablation of layer-1 neurons and residual PCA, qualitative threshold r_99 ratio). Primary risk acknowledged in proposal: CF2 residual additivity likely dominates, making the experiment a near-certain NO-GO. Pipeline value is marginal vs the overlapping L1AW and ODWQ ideas which test the same CF6×CF11 coupling more directly. Does not meet the "carries information on NO-GO" bar at A3=2 when the NO-GO outcome duplicates CF2 (already confirmed). Stripped primitive: per-layer MLP contribution magnitude vs residual — Stage 4 may want this as a 5-min pre-check only.

- **F-RMNQ (F/A, 10)** — REJECT-low-score. Total 10 clears threshold but is crowded out by Track A advancers with stronger residency arguments. The indirect saving (spectrum improvement from scale absorption) is speculative and the direct saving (0.23 MB of γ parameters) is negligible. A1=1 at best. Survives as a pre-check for F-GFQO's gauge-rotation: if std(γ)/mean(γ) < 0.01, the symmetry is trivial anyway; that 1-min check is free. Do not advance separately.

- **F-TIED [FREE SWING] (F/B, 8)** — REJECT-low-score. A4=1: the k-means rotation target (semantic clusters vs dual-path gradient split) is explicitly identified as likely missing the functional structure it claims to exploit; the mechanism is hand-waved at the critical step. A3=2 (30-min k-means, but the go/no-go threshold — "clusters align with gradient paths" — is not precisely derivable from CF12). Total 8. Stripped primitive: tied-embedding row cluster structure — Stage 4 may want to test this as part of the quantization-alignment line if RAOK succeeds.

- **DFSQ (U/B, 10)** — REJECT-low-score. A1=0: 48 KB tier-map savings is negligible; the real payoff is an enabling mechanism for mixed-precision inference (RAOK/PDAP), but the proposal does not stand alone. Without a primary residency or throughput claim, A1=0 sets the proposal below advancement threshold regardless of other axes. Stripped primitive: IEEE 754 denormal as in-band sentinel — useful to any future mixed-precision tier encoding but does not warrant an independent Stage 3 slot.

- **A-FSMT [FREE SWING] (A/A, 10)** — REJECT-low-score. A4=1: the FSM transition function T(s, tok) is factored as Q(Embed[tok] ⊕ Centroid[s]) — the quality bound of this factorization is entirely ungrounded; the centroid addition step is a hand-wave (nearest-centroid in Euclidean space of residual-stream activations is not guaranteed to preserve model outputs). A3=2 (6-hr experiment; the NLL ≤ 2× threshold is unlikely to be met at N=4096, and the cascade relies on a scaling extrapolation that is speculative). Path 3 eligibility (A2=3) is blocked because A4=1, not A4≥1 with strong specificity; the mechanism is not checkable beyond "try it and see." Stage 4 may want the centroid cardinality vs NLL scaling curve as a data point for residual-stream intrinsic dimension estimation (RSIDC line).

---

## 6. Hand-off to Stage 3

Priority order within each track. Total advancers: 24 (Path 1: 12 in Track B + 5 in Track A; convergence reps: 7 additional; frame-novelty: 2; wildcard: 3).

For Stage 3 bandwidth, the following are the highest-leverage items to pressure-test:

**Track A (arch-transposition) — top 4:**

1. **F-AERO-Q3** (F/A, 12) — Pressure-test: does AERO's activation-removal fold work for SwiGLU without retraining? The R3 kill of CSMF established polynomial substitution is broken; is literal silu→identity also broken? Stage 3 should verify: (a) what fraction of Qwen3 gate pre-activations fall outside the linear silu region (|silu(z)−z|/|z| > 0.05) — if > 50%, the experiment is pre-predictably NO-GO; (b) whether AERO's published results on GELU models generalize to SwiGLU (check AERO paper's Table 1 for SwiGLU family coverage). Most aggressive check: does AERO cite Qwen3-family results? If not, the experiment is the first such test and Stage 3 should confirm no post-AERO paper has done this.

2. **F-WVOS** (F/A, 12) — Pressure-test: is the W_V fold algebraically exact or does it introduce a transpose/reshape error? Audit the construction: head_output = softmax(QK^T)·(X·W_V)·W_O; substituting W_V ≈ U_V·Σ_V·V_V^T means (X·W_V)·W_O ≈ (X·V_V^T)·Σ_V·U_V^T·W_O — two matrices remain, not one. Confirm the fold is a genuine parameter reduction (not a computational rearrangement). Stage 3 should verify the per-head vs full-attention head count consistency (GQA: Qwen3-1.7B may have 8 KV heads × 128 dim, not 16 × 128).

3. **A-CADH** (A/A, 12) — Pressure-test: does within-layer cross-head W_Q similarity survive the per-head vs global rank distinction from CF11? CF11 says global K=128 is a GO but per-head K=64 is a NO-GO (+1.53 nats). CADH claims within-layer heads are similar enough to share a single fetch. Critically: does "similar" mean they can be replaced by one representative, or merely that they span a shared 128-dim subspace? The cos-similarity threshold (0.80) needs to be tied to a quality bound. Stage 3 should check: if heads are K=128 each but in different orientations within the 128-dim subspace, pairwise cos-sim of the full 128×2048 matrices may be low even if they share a common 128-dim basis.

4. **HRQK [FREE SWING]** (R/A, 11) — Pressure-test: distinguish within-layer head-PCA from v2-CHEAP-TEST-001. The within-layer head PCA targets the 16 heads of a SINGLE layer's W_Q treating each head as a sample. Stage 3 should verify that the PCA is over head-dimension variation (not layer-dimension), and confirm that the K_basis=64 GO threshold is consistent with the CF11 finding that per-head K=64 is a NO-GO (+1.53 nats) — these must be reconciled: per-head truncation vs per-layer head-pooling basis are different operations.

**Track B (compression) — top 4:**

5. **WDRS / F-WNORM convergence** (R+F/B, 11) — Pressure-test: is the ACTIVQ activation-weighted SVD variant (WDRS secondary) a novel claim or a re-derivation of GPTVQ (which uses Hessian = X^TX = activation second moment)? Stage 3 should check whether activation-weighted SVD with CF3's 3-tier structure as the weighting kernel is different from GPTVQ's Hessian weighting. If identical, only the W_down rank confirmation (the primary experiment) survives; ACTIVQ is pre-empted by GPTVQ.

6. **A-TROP** (A/B, 12) — Pressure-test: is tropical max-dominance in W_down output rows a novel framing or a re-description of activation-outlier pinning (LLM.int8() / SmoothQuant)? The key distinction: TROP operates on input-channel dominance for W_down OUTPUT rows (which input j dominates W_down[i,·] contribution), not on activation outlier channels (which output channels of h_mlp are large). Stage 3 should verify this axis distinction holds for W_down's shape (6144→2048), and check whether any published codebook method for W_down uses dominant-input-channel as a codebook-assignment key.

7. **AWCJR** (R/B, 10) — Pressure-test: is the compounding ΔNLL additivity assumption valid? CF11 measured W_Q and W_K independently; joint truncation may have superadditive quality cost. Stage 3 should check whether A3 (arXiv:2505.12942) — which uses attention-logit-error minimization — covers joint Q/K/V/O truncation and if so what quality costs they report. If A3 covers this, AWCJR may be pre-empted for the joint truncation claim even if W_V/W_O spectra are novel.

8. **ULHC** (R/B, 10) — Pressure-test: can Qwen3-8B (15.2 GB bf16) be handled on the 7.28 GiB Ryzen target for the lm_head SVD? The proposal correctly notes lm_head.weight alone (1.24 GB) fits in RAM separately. Stage 3 should verify (a) that Qwen3-8B is accessible (model download feasibility), (b) that the blockwise SVD on 151936×4096 completes in the 45-min estimate (LAPACK dgesdd on 37 blocks of 4096×4096), and (c) that Godey & Artzi's theorem (arXiv:2510.24966) explicitly applies to the untied configuration only — confirm the paper's stated assumptions.

**Convergence reps for Stage 3 scrutiny:**

- **C1 (W_V/W_O spectrum)**: F-WVOS is the rep. Stage 3 confirms the SVD protocol matches AQFKV's W_Q/W_K methodology for apples-to-apples comparison.
- **C3 (W_down rank)**: F-WNORM is the rep. Stage 3 confirms no R3/R4-class pre-emption by distinguishing the d_ffn > d_model shape from the square-matrix cases already run.
- **C5 (head-similarity disambiguation)**: A-CADH and HRQK share the 10-min L14 head pairwise cos-similarity measurement as a common pre-check. Stage 3 should confirm this measurement is independent of and consistent with CF11's global K=128 finding.
