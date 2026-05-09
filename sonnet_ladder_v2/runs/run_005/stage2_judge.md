# Stage 2 Judge — Run 005 (v2 Sonnet Ladder)

Scored 2026-05-09. Kill enforced: v2-CHEAP-TEST-001 (cross-layer W_Q stacked SVD / shared basis — class dead). Union of 32 proposals across orientations R, C, F, U, A.

---

## 1. Union Score Table

| ID | Orientation | Track | Title | A1 | A2 | A3 | A4 | A5 | Total | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| R5-AWQKV | R | B | Attention Weight Joint Quartet Compression | 2 | 1 | 3 | 2 | 2 | 10 | ADVANCE |
| R5-RAOK-FULL | R | B | RAOK 3-Tier Activation Codebook Full-Stack | 2 | 2 | 3 | 3 | 2 | 12 | ADVANCE |
| R5-VOWN | R | A | W_V W_O Structured Nullspace Folding | 2 | 2 | 3 | 2 | 3 | 12 | ADVANCE |
| R5-LAYERGATE | R | B | Layer-1 Gate-Fold + Global Tiered Quant Cascade | 1 | 1 | 3 | 2 | 2 | 9 | ADVANCE |
| R5-ATTNBAND | R | B | Attn Score Bandwidth Quant via Softmax Invariance | 1 | 2 | 2 | 2 | 1 | 8 | ADVANCE-elegant-equivalence |
| R5-WVWK-MLA | R | A | Post-Training MLA-Style W_V/W_K Joint Projection | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE-convergence |
| C1-ODSP | C | B | Outlier-Tier Deep-Layer Spread as Dynamic bpw Switch | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE-convergence |
| C2-TESOC | C | B | Tied-Embed Static-Outlier Column Identity | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE |
| C3-ZOIA | C | A | W_Q Subspace × Outlier-Channel Alignment INT8 Attn | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE |
| C4-GCQI | C | A | Gate-Constancy × W_Q Independence at Layer 1 | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE |
| C5-RKDB | C | B | Residual-Cosine Flatness Implies KV Recompute Delta Bound | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE-convergence [FREE SWING] |
| C6-DSAF | C | B | Deep-Layer Outlier Spread × W_Q Compressed Subspace Floor | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE-convergence |
| F1-JQKLR | F | B | Joint W_Q / W_K Simultaneous Low-Rank | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE-convergence |
| F2-HPGF | F | A | Head-Permutation Gauge Fixing on W_Q / W_V | 1 | 3 | 3 | 2 | 3 | 12 | ADVANCE-frame-novelty |
| F3-AEROT | F | A | W_down/W_up AERO Precondition Test | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE |
| F4-ASCR | F | A | Attention Score Commutativity Under RoPE Gauge | 0 | 2 | 3 | 1 | 3 | 9 | REJECT-low-score |
| F5-SECP | F | B | Spectral Entropy as Calibration-Free Rank Proxy | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE |
| F6-GFNM | F | A | RMSNorm Gauge Freedom for W_Q Scale Absorption | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE [FREE SWING] |
| U1-NCCG | U | B | NTFS Compressed-Cluster GGUF Variant | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE |
| U2-SPHF | U | B | GGUF Tensor Hot-Tier Repack via Superfetch Hint | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE |
| U3-XDFR | U | A | NTFS Extent Defrag for Sequential NVMe Read | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE |
| U4-SBDS | U | B | CPU Store-Buffer Drain Scheduling for RMSNorm | 0 | 1 | 3 | 2 | 3 | 9 | REJECT-low-score |
| U5-HPST | U | B | GGUF Tensor Interleave Repack for HW Prefetcher | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE |
| U6-LPMT | U | A | Windows Large-Page Mapping for Weight Tensors | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE-wildcard [FREE SWING] |
| A1-GFRS-2 | A | B | Gauge-Fixed Residual Stream Sign-Flip Codebook | 1 | 3 | 3 | 2 | 3 | 12 | ADVANCE-frame-novelty |
| A2-TROP | A | A | Tropical Semiring Attention Approximation | 1 | 3 | 2 | 1 | 3 | 10 | ADVANCE-frame-novelty |
| A3-SEED | A | A | 10 MB Seed Weight Synthesis via Spectral Generator | 2 | 3 | 2 | 1 | 3 | 11 | REJECT-low-score |
| A4-APND | A | A | Append-Only Inference — Log-Structured KV | 1 | 2 | 3 | 2 | 2 | 10 | ADVANCE |
| A5-FSMI | A | A | Finite-State Machine Inference | 1 | 3 | 2 | 1 | 3 | 10 | REJECT-low-score [FREE SWING] |
| A6-CNTR | A | B | Content-Addressable Weight Dispatch | 1 | 3 | 3 | 2 | 3 | 12 | ADVANCE-frame-novelty |
| A7-NRAM | A | B | No-RAM Weight Streaming — NVMe Only [OUT-OF-ORIENTATION] | 1 | 1 | 3 | 2 | 1 | 8 | REJECT-low-score |
| R5-ATTNBAND | R | B | (scored above) | — | — | — | — | — | — | — |

**Score notes:**
- F4-ASCR: A1=0 because the residency/compute payoff is ~1–3% end-to-end — below the A1=1 threshold of 1.2× improvement. A4=1 because the RoPE equivariance precondition is unverified and gesturally motivated. Total 9 but A1=0 sets it to REJECT.
- U4-SBDS: A1=0 because the estimated gain is 0.3% effective bandwidth at best — far below 1.2× threshold. Smallest test is valid but the mechanism has sub-threshold ceiling.
- A3-SEED: A4=1 because the synthesis step has no constructive derivation connecting seed size to weight reconstruction fidelity — the "fits in L3" argument is not a mechanism, it is a storage note. A3=2 because the experiment (train a small MLP on one layer for 2 hours) barely qualifies as ≤8h but the go/no-go threshold (5% Frobenius) is well-defined.
- A5-FSMI [FREE SWING]: A4=1 because the state-space cardinality claim is entirely speculative and no calculation connects model rank to FSM state count. CF-tether suspended; structural floor met, but A4=1 caps advancement.
- A2-TROP: A4=1 — the tropical claim is mechanically checkable but the precision of the "how lossy is hard-argmax" question depends on a property of the model that hasn't been measured; the experiment is ≥4h (full forward pass + sweep). Advancing under frame-novelty despite modest A4.

---

## 2. Per-Advancer Rationale

**R5-RAOK-FULL** (A1=2, A2=2, A3=3, A4=3, A5=2; total 12). Carried by A3 and A4: the 3-tier Jaccard stratification at K=18 channels is a checkable binary claim grounded directly in CF3 (K=1% Jaccard=0.308 already measured; the 18-channel sub-band is a 30-min re-run). A4=3 because every step is auditable — T1/T2/T3 boundaries come from CF3's own numbers, and the AVX2 scatter cost estimate is a one-line calculation. Load-bearing primitive: K=0.1% static vs K=0.9% dynamic Jaccard stratification. Smallest test: re-run PDAP at K=18 channels; confirm mean Jaccard < 0.35.

**R5-VOWN** (A1=2, A2=2, A3=3, A4=2, A5=3; total 12). Carried by the AERO-analog framing (algebraic fold of the V-O chain) and residency arithmetic: at 70B, rank-512 W_VO saves ~7.3 GB. A4=2 because the mechanism is checkable but the rank claim on W_V×W_O product is unverified; it could be full-rank if W_V and W_O are spectrally complementary. A5=3 because post-training V-O product folding appears in no published work found as of 2026-05-09. Smallest test: SVD of W_V[14]@W_O[14]; r_99 < 0.70d is the gate.

**R5-WVWK-MLA** (A1=2, A2=2, A3=3, A4=2, A5=2; total 11). Grounded in CF11: SUMMARY.md explicitly calls MLA-style joint Q-K projection "empirically motivated" and "W_V and W_O spectra — untested." This fills the untested GQA kv-head stacking question. A4=2: the within-layer (not cross-layer) GQA stacking is structurally distinct from v2-CHEAP-TEST-001 but the kv-head rank claim needs measurement. Smallest test: reshape W_K[14] to (1024×2048), compute r_99, apply rank-128/256 approximation, measure ΔNLL. Advanced as convergence representative (see Convergence C1).

**R5-AWQKV** (total 10). Grounded in CF11 (W_Q, W_K GO measured; W_V, W_O explicitly noted as untested). A2=1 because the "extend CF11 to all 4 matrices" frame is in-vocabulary — it's the natural next step, not a novel frame. A4=2 because the per-layer global SVD path is constructive and the residency arithmetic is explicit. Smallest test: 15-min W_V/W_O spectrum measurement at layer 14.

**R5-LAYERGATE** (total 9). CF6 anchor is load-bearing (36% foldable neurons at Layer 1). The cascade framing with RAOK-style tiering rescues what was otherwise a tiny standalone win. A1=1 because the absolute residency save is ~169 MB at 70B — below 1.2× threshold in isolation but meaningful as the "free quality" rung in a quality-reallocation cascade. A2=1 because gate-variance folding is known; the "use the freed quality budget in the tier quantization cascade" framing is the only fresh move. Smallest test: ΔNLL after Layer-1 fold on held-out; expected <0.02 nats.

**R5-ATTNBAND** (total 8; advancing as elegant-equivalence under Path 4). Sub-class: `gauge-exploitation`. Softmax shift-invariance is an algebraic identity: softmax(x + c) = softmax(x). Representing attention scores in centered form before INT4 quantization is a lossless coordinate change followed by compression — the AERO-analog for the score domain. A4=2: the mechanism is constructive (one-line gauge argument, nothing imported). A3=2: the smallest test is ≤30 min with a numerical gate (ΔNLL < 0.10 at INT4). Advancing under Path 4 because the elegance is auditable, A4≥2, and the frame (softmax gauge freedom as quantization coordinate) appears novel. Note: primary payoff is at long-context / prefill; short-context residency benefit is marginal, hence A1=1.

**C1-ODSP** (total 10). Composition of CF3 depth-gradient (within-CF3 finding) with a per-layer bpw scheduling claim. A2=2: the depth-gradient-as-bpw-proxy framing imports from database cost-model tiering with no published LLM analog. A4=2: the coupling equation is a labeled inequality, checkable in ≤2 hours from R2 PDAP data. Advancing as convergence representative (Convergence C2). Smallest test: Spearman ρ ≥ 0.6 between depth and spread_ℓ.

**C2-TESOC** (total 10). Composition of CF12 (tied embed full-rank, catastrophic truncation) with CF3 (K=0.1% static outlier channels). The coupling — that K=0.1% residual-stream channels correspond to disproportionate logit variance in E — is a nontrivial claim. A4=2: the measurement is a one-pass projection variance computation. A5=2: hybrid-precision quantization of the embed matrix informed by outlier-channel identity from activation statistics has no published analog as of 2026-05-09. Smallest test: 15-min logit-variance contribution measurement for 2 outlier columns.

**C3-ZOIA** (total 11). Composition of CF11 (W_Q K=128 global GO) with CF3 (K=0.1% static outlier channel). The projection residual ρ_ℓ is a crisp geometric measurement derivable in 5 minutes from the already-run AQFKV SVDs. If ρ_ℓ ≤ 0.3, the "zero INT16 sidecar" claim follows algebraically. A4=2 because the precondition (outlier inside W_Q subspace) is directly measurable. Smallest test: ρ_ℓ measurement for all 28 layers; ≤5 min.

**C4-GCQI** (total 10). Composition of CF6 (Layer-1 gate-fold) with CF11 (W_Q K=128 GO). The independence interaction term ε is the novel falsifiable claim — it produces the joint-compression coefficient that every future multi-path compression proposal needs. A3=3: three-eval script in ~15 min, binary interaction measurement. Smallest test: simultaneous Layer-1 gate fold + W_Q K=64 eval; check additivity.

**C5-RKDB** (total 10; FREE SWING). CF2 × CF11 composition: the δ norm bound from CF2's 0.99 cosine similarity, combined with CF11's W_K K=512 operator-norm reduction, produces a computable KV-recompute error bound. A4=2: the inequality is written out and both CF dependencies are explicit. CF-tether requirement technically suspended (FREE SWING) but both CFs are load-bearing. Advancing as convergence representative (Convergence C3). Smallest test: 25-min relative error measurement on 100 tokens.

**C6-DSAF** (total 10). Composition of CF3 depth-gradient with CF11 W_Q per-layer spectrum. The per-layer K schedule derived from activation statistics is novel; no MDL, no calibration fitting. A4=2: the depth-stratified K schedule is derivable from the PDAP data with one formula. Advancing as convergence representative (Convergence C2). Smallest test: check whether deep-layer W_Q K=256 reduces ΔNLL by ≥0.2 nats vs uniform K=128.

**F1-JQKLR** (total 10). Within-layer joint Q+K compression via canonical correlation. Structurally distinct from v2-CHEAP-TEST-001 (per-layer, not cross-layer). A3=3: the 8-minute canonical-angle measurement is binary (≥30 angles < 10° or not). Advancing as convergence representative (Convergence C1). Smallest test: canonical correlation between W_Q and W_K top-256 left singular vectors at layer 14.

**F2-HPGF** (A1=1, A2=3, A3=3, A4=2, A5=3; total 12; frame-novelty). The head-permutation gauge freedom is a genuinely fresh frame: the loss invariance under simultaneous head-permutation is an exact algebraic identity, and the energy-sorted gauge-fixing is a coordinate choice that the published head-pruning literature does not make. A2=3 because no published LLM compression paper exploits the S_16 gauge to define an energy-sorted head ordering as a *precondition* for head elimination (as distinct from gradient-based importance). A4=2 because the gauge argument is constructive (one-line: permutation of both W_Q and W_O leaves function identical). Smallest test: 5-min head Frobenius norm distribution across 28 layers; check for cliff at M=4.

**F3-AEROT** (total 9). Post-hoc per-neuron linearizability test for a subset-AERO fold. A2=2: subset-AERO (post-training, no retraining, linearization-only for qualifying neurons) is not in the AERO paper. A4=2: the calibration-distribution linearization error is a computable one-pass statistic. A3=2: the experiment qualifies as ≤8h but the go/no-go threshold is "≥10% of neurons" — somewhat qualitative on the action side (ΔNLL from the fold is not measured in the smallest test). Smallest test: 30-min calibration pass, per-neuron max residual computation.

**F5-SECP** (total 10). Spectral entropy as calibration-free rank proxy. A2=2: weight-spectral-entropy as a compression guidance tool for LLMs has no named entry. A4=2: the prediction claim is specific (K* from H_s > 0.99 matches CF11 within 0.10 nats) and uses no free parameters. A5=2: no direct prior found as of 2026-05-09. Smallest test: 20-min SVD + entropy + inference run on 3 layers.

**F6-GFNM** (total 10; FREE SWING). RMSNorm gauge absorption as SVD precondition. CF-tether suspended but the mechanism connects to W_Q (the CF11 compression target). A4=2: W_Q_gauged = W_Q @ diag(gamma) is a single-line operation; spectral entropy comparison is checkable in 10 min. Advancing as FREE SWING — structural floor met. Smallest test: H_s comparison on all 28 layers; 10 min.

**U1-NCCG** (total 10). NTFS transparent compression on GGUF metadata stratum. A2=2: no published LLM inference work uses OS-level transparent file compression on weight files. A4=2: LZNT1 compression of fp16 scale values is measurable in <2 hours; the compressibility question has a clean yes/no. A5=2: no direct prior as of 2026-05-09 (Apple LLM-in-a-Flash uses DRAM staging, not OS transparent compression). Smallest test: FSCTL_SET_COMPRESSION on metadata file; measure on-disk size.

**U2-SPHF** (total 10). PrefetchVirtualMemory for cold-start first-token latency. A2=2: no published LLM inference work targets the first-token setup window as a prefetch opportunity on Windows. A3=3: TTFT measurement is binary and runs in <2 hours. A4=2: mechanism is a documented Win32 API call; precondition (setup window ≥ NVMe transfer time) is measurable. Smallest test: two-line patch + 20-run TTFT measurement.

**U3-XDFR** (total 10). NTFS extent defrag for NVMe firmware stride detection. A2=2: no published work addresses GGUF extent fragmentation and NVMe firmware prefetch interaction. A4=2: FSCTL_GET_RETRIEVAL_POINTERS is a standard API; the throughput comparison between 1-extent and 50-extent versions is a direct measurement. A3=3: binary go/no-go (≥10% throughput difference) in ≤3 hours. Smallest test: write GGUF twice (fresh vs fragmented); measure llama-bench per-token time.

**U5-HPST** (total 10). Q4_K_M split-field GGUF for CPU stride prefetcher. A2=2: cache-line stride alignment of quantized weight block layouts for CPU stride prefetcher has no published analog. A4=2: the non-power-of-two stride hypothesis is a falsifiable hardware claim. A3=3: ≤4-hour experiment with a ≥5% GEMV throughput gate. Smallest test: Python repacker + GEMV benchmark.

**U6-LPMT** (total 10; FREE SWING, ADVANCE-wildcard). Large-page TLB optimization. CF-tether suspended. A2=2: no published LLM inference paper applies MEM_LARGE_PAGES specifically to weight tensor access TLB pressure. A4=2: the mechanism is a documented Win32 flag; TLB miss reduction is a measurable PMU counter. Structural floor met. Advanced as FREE SWING — the gain is likely small at 1.7B but the structural finding (L2 TLB coverage of weight tensors) is load-bearing for any future 70B DRAM-resident approach.

**A1-GFRS-2** (A1=1, A2=3, A3=3, A4=2, A5=3; total 12; frame-novelty). The Z_2^d residual-stream gauge symmetry frame has no published LLM compression analog. A2=3: the constraint forces a coordinate-free codebook design that the literature has no vocabulary for; the gauge-fixing argument is algebraically exact (no theorem imports, no precondition violation). A4=2: per-row sign normalization is a computable one-line operation; entropy measurement is a standard calculation. A5=3: sign-flip gauge exploitation for quantization codebook design appears nowhere in the 2024-2026 literature as of 2026-05-09. Smallest test: 10-min H(sign) measurement on W_up layer 0 after gauge-fixing.

**A6-CNTR** (A1=1, A2=3, A3=3, A4=2, A5=3; total 12; frame-novelty). Content-addressable weight dispatch keyed by CF3 outlier-channel signatures. A2=3: no published LLM inference work uses BLAKE3-keyed content-addressable routing for weight dispatch; the frame imports from content-addressable storage (Git/IPFS) applied to weight rows. A4=2: the precision@20% measurement is a direct checkable calculation on CF3's already-measured outlier channel identities. A5=3: no published analog as of 2026-05-09. Smallest test: 20-min precision@20% check; does CF3 outlier-signature predict W_up firing row identities?

**A2-TROP** (A1=1, A2=3, A3=2, A4=1, A5=3; total 10; frame-novelty). Tropical semiring attention. A2=3: hard-argmax attention under a tropical constraint has no named entry in the LLM compression literature; the frame imports tropical algebra. A4=1: the mechanism is specified but the "how lossy is hard-argmax" question requires an unverified empirical property of Qwen3. A3=2: ≤30-min experiment but the go/no-go threshold (ΔNLL < 0.3 nats) is a qualitative-directional choice, not tied to a CF1-CF12 number. Advancing under Path 3 frame-novelty (A2=3, A4≥1, A3≥1). Carry-forward note: Stage 3 should pressure-test whether top-k tropical (k=4) is viable if top-1 fails.

**A4-APND** (total 10). Append-only inference / log-structured KV. CF2 anchor: cos(h_L, h_{L+1})~0.99 motivates delta compression. A2=2: log-structured merge-tree constraint applied to KV and residual stream has no LLM named entry. A4=2: the delta ratio ‖Δ_L‖/‖h_L‖ measurement is a direct 15-min computation. A3=3: binary delta-ratio gate in ≤15 min. A critical risk flag: CF2's high cosine similarity does NOT imply small L2 norm of Δ_L (two large near-parallel vectors can have a large difference). The smallest experiment tests this directly; advancement is contingent on the NO-GO structural finding being informative either way. Smallest test: 50-token delta-ratio measurement across all 28 layers.

---

## 3. Convergence Map

**Convergence C1 — Within-layer attention matrix joint subspace as compression primitive.**
R5-WVWK-MLA (Reach), F1-JQKLR (First-Principles), and C3-ZOIA (Composition) all converge on the within-layer joint compression of attention weight matrices (W_Q × W_K canonical angles; W_K/W_V GQA stacking; W_Q K=128 subspace absorbing outlier channels). Each arrives via different vocabulary — WVWK-MLA via CF11 MLA-style motivation; JQKLR via canonical correlation first-principles geometry; ZOIA via the outlier-projection residual measurement — but all share the primitive: **within-layer attention weight subspace structure as the post-training compression coordinate**. Strongest representative: R5-WVWK-MLA (highest A1+A4 sum, cleanest residency arithmetic). Cheapest settler: W_K[14] stacked-GQA SVD, 15 min — if r_99/(n_kv_heads × d_head) < 0.70, the joint within-layer attention subspace is real and all three proposals gain structural support simultaneously.

**Convergence C2 — Depth-stratified outlier spread as per-layer compression schedule primitive.**
C1-ODSP (Composition) and C6-DSAF (Composition) independently converge on the CF3 depth-gradient (layers 23-27 have 2-3× wider outlier channel spread than layers 2-19) as a per-layer parameter for compression scheduling — ODSP maps it to bpw, DSAF maps it to W_Q rank budget. Both require the same measurement: extend the PDAP script to all 28 layers for per-layer spread_ℓ. Strongest representative: C6-DSAF (A4=2 with explicit K-schedule derivation). Cheapest settler: full-28-layer PDAP extension run (≤2 hours) — produces the per-layer spread_ℓ vector, gating both proposals simultaneously and unlocking the per-layer K schedule for the AWQKV proposal as well.

**Convergence C3 — CF2 residual-near-parallelism as KV-skip / delta-compression primitive.**
C5-RKDB (Composition) and A4-APND (Constraint-Alien) both ground their mechanism in CF2's cos(h_L, h_{L+1}) ≈ 0.99. RKDB uses it to bound KV-recompute error; APND uses it to motivate delta compression of append-only residuals. They share the critical unverified question: does high cosine similarity imply small L2 delta (‖Δ_L‖ / ‖h_L‖ < 0.15)? If yes, both proposals gain traction; if no, CF2 cannot be used as an amplitude bound — only an angle bound. Cheapest settler: 15-min delta-ratio measurement from A4-APND's smallest experiment — single number that either validates or closes the CF2 amplitude-coupling assumption for both proposals.

---

## 4. Frame-Novelty Bonus Advancers (Path 3, up to 2 slots)

**Slot 1 — F2-HPGF** (A2=3, A4=2, A3=3). The head-permutation gauge fixing frame is genuinely new in the LLM compression context. The S_16 permutation invariance of multi-head attention has been mentioned in passing in theoretical analyses but never exploited as a practical coordinate choice for head elimination. The cousin in the published landscape is gradient-based head pruning (Michel et al. 2019), which is fundamentally different (heuristic importance vs exact symmetry). No published cousin in the 2024-2026 saturation-frame list found as of 2026-05-09. The 5-min experiment (head Frobenius energy distribution) is the cheapest structural measurement in the run and produces a load-bearing finding regardless of outcome.

**Slot 2 — A1-GFRS-2** (A2=3, A4=2, A3=3). The Z_2^d residual-stream gauge symmetry frame is the cleanest constraint-alien proposal in the run: the sign-flip invariance is algebraically exact (not approximate), the gauge-fixing is a one-time O(L×d) operation, and the sign-entropy measurement is a direct check with no imported theorem. The published literature on LLM quantization does not engage with gauge-invariance as a codebook design coordinate. No saturation-frame cousin found as of 2026-05-09. (A6-CNTR also has A2=3 and competes for this slot; A1-GFRS-2 is chosen for slot 2 because its A4=2 is better-grounded — the gauge argument requires no unverified empirical prediction about W_up firing, whereas A6-CNTR's precision@20% claim is empirically conditional. A6-CNTR advances under Path 1.)

---

## 5. Rejected with Rationale

- **F4-ASCR** (A1=0, A4=1): A1=0 — residency/compute payoff is ~1–3% end-to-end; below the 1.2× threshold for any score on A1. A4=1 — the RoPE equivariance hypothesis is gesturally motivated; the "most likely outcome is partial equivariance" statement in the proposal itself signals unverified precondition. Total 9 but A1=0 disqualifies advancement. Rejected as REJECT-low-score. Surviving structure: the RoPE-frequency block-diagonality measurement is itself a structural characterization of W_Q that Stage 4 may want to incorporate as a 5-min diagnostic alongside other W_Q spectrum measurements.

- **U4-SBDS** (A1=0): NT-store for RMSNorm scale writeback. A1=0 — estimated gain is 0.3% effective bandwidth; far below 1.2×. The primary risk section confirms the mechanism evaporates if the scale vector is reread before cache eviction, which it is. Rejected as REJECT-low-score. Surviving structure: the NT-store optimization for post-gate activation buffers (not RMSNorm) remains plausible; Stage 4 may want to revisit NT-store applied to the SwiGLU intermediate activations specifically, which are strictly write-once-then-read-once.

- **A3-SEED** (A4=1): The synthesis step has no constructive derivation; A3-SEED relies on training a small MLP on one layer (not inference-time synthesis), and the smallest experiment trains for ~2 hours — this is closer to "implement first then measure" than a structural diagnostic. A4=1 because the mechanism is speculative (no calculation bounds the synthesis fidelity from seed size). CF8's full-rank finding makes the Kolmogorov complexity argument against feasibility very strong. Rejected as REJECT-low-score. Surviving structure: the rate-distortion point of a 10 MB seed on one weight layer is worth measuring if Stage 4 wants to bound how compressible trained weights are from an information-theoretic perspective.

- **A5-FSMI [FREE SWING]** (A4=1): CF-tether suspended but A4=1 is the hard floor. The state-space cardinality claim ("65536 states suffice") has no constructive argument — the proposal itself acknowledges "almost certainly insufficient for general inference." The Myhill-Nerode argument is invoked but the precondition (the transformer's effective state space is finite and small) is both unverified and strongly contradicted by CF4/CF5/CF8. Rejected as REJECT-low-score (A4 floor). Surviving structure: the distinct-int8-quantized-states measurement is a cheap (~20 min) structural diagnostic for residual-stream compressibility; Stage 4 may want to run it as a 5-min add-on to another experiment.

- **A7-NRAM [OUT-OF-ORIENTATION]** (total 8): Closest published: Apple LLM-in-a-Flash (Wu et al. 2024) does substantively the same thing on similar hardware. A5=1 at best. A2=1 because the "zero RAM for weights" constraint doesn't force any new mechanism beyond the standard NVMe streaming offload framing already in LLM-in-a-Flash. Total 8 < 9 minimum. Rejected as REJECT-low-score. Note: the smallest experiment (NVMe unbuffered bandwidth measurement) is a useful infrastructure diagnostic; it overlaps with CF13* re-derivation and should be run as a background measurement regardless.

---

## 6. Hand-off to Stage 3

**Track A advancers (top + convergence + frame-novelty):**

1. **R5-VOWN** (R/A, total 12) — Stage 3 should pressure-test whether W_V × W_O product rank is genuinely lower than individual matrix ranks, or whether complementary subspaces make the product full-rank. The AERO analog is clean algebraically but the V-O product is a different mathematical object from the SwiGLU activation fold. Specific question: does training force W_V and W_O into a "cooperating low-rank channel" or do they each span independent directions?

2. **R5-WVWK-MLA / F1-JQKLR** (convergence C1 rep, R/A + F/B, totals 11 + 10) — Stage 3 should check whether post-training MLA-style KV projection has been published for GQA models (not DeepSeek training-time, but post-hoc calibration-fit). A3 (arXiv:2505.12942) does attention-logit error minimization for W_Q/W_K — does it cover the full (W_K, W_V) pair for GQA? That's the pre-emption risk.

3. **C3-ZOIA** (C/A, total 11) — Stage 3 should verify that the projection residual ρ_ℓ calculation is not equivalent to a previously published "outlier-aware attention quantization" scheme (SpQR, SmoothQuant). The critical distinction is the "zero INT16 sidecar" claim — find if any paper has argued that the W_Q compression subspace absorbs outlier channels specifically.

4. **F2-HPGF** (F/A, total 12, frame-novelty) — Stage 3 should check whether head-energy-sorted gauge fixing appears in any attention pruning paper from 2024-2026. The energy-sorted permutation is a natural idea; the question is whether anyone has framed it as an exact symmetry exploitation vs a heuristic.

**Track B advancers (top + convergence + frame-novelty):**

1. **R5-RAOK-FULL** (R/B, total 12) — Stage 3 should check whether RAOK's 3-tier Jaccard stratification at K=0.1%/0.9% has a published cousin in the 2024-2026 activation quantization literature (PrefixQuant, SmoothQuant V2, etc.). The specific split at K=0.1% static vs K=0.9% dynamic is grounded in CF3; the question is whether any paper independently derived the same split point.

2. **A1-GFRS-2** (A/B, total 12, frame-novelty) — Stage 3 should look for any paper applying Z_2 gauge symmetry of neural networks to quantization design specifically. The symmetry is well-known theoretically; the application to sign-plane entropy measurement and codebook design is the novel move.

3. **A6-CNTR** (A/B, total 12, frame-novelty) — Stage 3 should pressure-test the CF1 risk: that the content-addressable dispatch fails for the same reason the row-herding predictor failed (SwiGLU interaction makes W_up firing non-predictable from residual-stream signals alone). If the precision@20% experiment is negative, the surviving primitive (CF3 outlier channels as routing keys, with wider K) should be reframed.

4. **C2-TESOC** (C/B, total 10) — Stage 3 should check SpQR (arXiv:2306.03078) carefully: SpQR does mixed-precision quantization with outlier-weight isolation. The distinction is that TESOC's sensitive-column identity comes from activation statistics (CF3), not weight sensitivity calibration. If SpQR already handles embed matrix outlier columns, TESOC reduces to an incremental application.
