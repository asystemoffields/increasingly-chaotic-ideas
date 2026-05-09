# Stage 2 Judge — Run 009

Independent cold-start scoring. Kill list applied: v2-CHEAP-TEST-001 (cross-layer W_Q class), v2-S3-R004-001 (arbitrary rotation). CF9 gate enforced.

---

## 1. Union Score Table

| ID | Orientation | Track | Title | A1 | A2 | A3 | A4 | A5 | Total | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| R9-R1 | R | B | ATQKV — W_V / W_O spectrum audit + cascade | 2 | 2 | 3 | 2 | 3 | 12 | ADVANCE |
| R9-R2 | R | B | RAOK-REACH — tiered activation codebook as cascade anchor | 2 | 2 | 2 | 2 | 2 | 10 | ADVANCE |
| R9-R3 | R | A | HSMLA — post-training shared-basis MLA | 2 | 2 | 2 | 2 | 2 | 10 | ADVANCE |
| R9-R4 | R | B | NVME-ATTN-STREAM — interleaved attention/MLP NVMe layout | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE-convergence |
| R9-R5 | R | A | LAYERONE-FOLD — Layer-1 gate constant fold cascade [FREE SWING] | 1 | 2 | 2 | 2 | 3 | 10 | ADVANCE-wildcard |
| C13-EQSUB | C | B | Embedding row projection into W_Q K=128 subspace | 1 | 2 | 3 | 2 | 3 | 11 | ADVANCE-convergence |
| C14-CFSHIFT | C | A | CF6 constant injection × W_Q[L2] logit pre-bias | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE-frame-novelty |
| C15-SNRATE | C | B | W_up sensitivity × CF11 spectrum as precision budget oracle | 2 | 2 | 3 | 2 | 2 | 11 | ADVANCE |
| C16-RSEMBED | C | A | Residual near-unity × tied embed as lm_head skip gate | 1 | 2 | 2 | 1 | 2 | 8 | REJECT-low-score |
| C17-LAYERTAX | C | B | v2-CF1 Jaccard × CF6 gate-variance layer characterization [FREE SWING] | 1 | 2 | 2 | 2 | 2 | 9 | ADVANCE-convergence |
| F1-MQKP | F | B | Attention product M = W_Q W_K^T direct spectral compression | 2 | 3 | 3 | 3 | 2 | 13 | ADVANCE |
| F2-CKDJ | F | A | CF3 outlier channels × CF11 W_Q subspace alignment | 1 | 3 | 3 | 2 | 3 | 12 | ADVANCE-frame-novelty |
| F3-L1GNF | F | A | Layer-1 gate null-space fold | 1 | 2 | 3 | 3 | 2 | 11 | ADVANCE |
| F4-WQKRS | F | B | W_Q/W_K spectral asymmetry water-filling schedule | 2 | 2 | 3 | 3 | 2 | 12 | ADVANCE |
| F5-RSUC | F | B | Residual-stream update covariance depth profile | 1 | 2 | 3 | 2 | 3 | 11 | ADVANCE-convergence |
| F6-TROPICALDECODE | F | A | Tropical semiring attention argmax [FREE SWING] | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE-convergence |
| U1-B-ETWO | U | B | ETW file-I/O kernel trace as weight-access hotness oracle | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE |
| U2-B-ADST | U | B | NTFS ADS tier sidecar for mixed-precision GGUF | 1 | 2 | 2 | 2 | 3 | 10 | REJECT-low-score |
| U3-B-FIOP | U | B | SetFileInformationByHandle I/O priority elevation | 1 | 2 | 2 | 2 | 3 | 10 | REJECT-low-score |
| U4-B-OSRD | U | B | GGUF hot-layer partial residency via OS RAM disk | 1 | 2 | 2 | 2 | 2 | 9 | REJECT-low-score |
| U5-A-VGTP | U | A | VEH guard-page demand-tier promotion [FREE SWING] | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE-wildcard |
| A1-TROPFFN | A | A | Tropical semiring FFN dispatch | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE-convergence |
| A2-APPENDKV | A | A | Append-only KV accumulator with idempotent retrieval | 1 | 3 | 2 | 2 | 2 | 10 | ADVANCE |
| A3-FSMDEC | A | A | FSM decoder with N ≤ 256 states | 1 | 3 | 2 | 2 | 3 | 11 | ADVANCE |
| A4-SEEDNET | A | B | 10 MB generative seed for full-layer weight synthesis | 2 | 3 | 2 | 1 | 3 | 11 | REJECT-low-score |
| A5-REGONLY | A | A | Register-only decode [FREE SWING] | 1 | 1 | 2 | 2 | 1 | 7 | REJECT-low-score |

---

## 2. Per-Advancer Rationale

**R9-R1 / ATQKV** — A3=3 from a clean binary settler (30 min, GO threshold well-defined); A5=3 because the W_V/W_O spectrum is genuinely unmeasured and no arXiv paper audits all four attention matrices post-training on Qwen3. Grounded in CF11 (W_Q, W_K partial results); W_V and W_O are the natural extension. Smallest test: SVD of W_V and W_O + joint ΔNLL sweep, ~30 min. Load-bearing primitive: spectrum concentration of all four attention weight matrices.

**R9-R2 / RAOK-REACH** — CF3/v2-CF1 grounded (K-dependent Jaccard as tier boundary). A4=2 because the precision-matching benefit requires joint codebook tuning (one transition step gestured rather than derived from first principles). Smallest test: ΔNLL(3-tier activation) vs uniform INT4 at 4 bpw, ~2 hrs. Load-bearing primitive: per-token outlier-channel dynamicity as quantization tier boundary.

**R9-R3 / HSMLA** — CF11's head-redundancy (16:1 ratio at K=128 global) motivates the shared-basis inference factorization. A4=2 because the Frobenius reconstruction step from stacked-head SVD to shared basis is derivable but not fully audited in one paragraph. Smallest test: principal-angle check between per-head W_Q matrices (5 min early abort) + full stacked SVD + ΔNLL, ~45 min. Load-bearing primitive: head-shared subspace as a factored inference computation.

**R9-R4 / NVME-ATTN-STREAM** — Convergence advancer. A1=1 (1.44× throughput only at NVMe-tier, not standalone path to 70B residency). The interleaved layout producing deterministic prefetch is a real structural insight; convergence with F5-RSUC and other NVMe-aware ideas gives it a slot. First rung re-derives CF13*. Smallest test: 10 min NVMe sequential read benchmark under ik_llama.cpp workload. Load-bearing primitive: per-layer attention-then-MLP layout for overlap scheduling.

**R9-R5 / LAYERONE-FOLD [FREE SWING]** — CF6 is the direct anchor (36.1% Layer-1 foldable neurons). Wildcard tag is justified: the context-conditional cascade extension lacks grounding, but the base CF6 fold is structurally real. A4=2 because the AERO algebraic identity is well-established; the calibration approximation (c_i ≈ constant) is the only gap. Smallest test: ΔNLL on held-out after Layer-1 fold + extended variance probe, ~1.5 hrs. Load-bearing primitive: near-constant gate neurons as algebraic fold targets.

**C13-EQSUB** — Convergence advancer. Coupling claim (embedding row projection mass into W_Q K=128 subspace ≥ 0.50) is falsifiable in 35 min and intersects with HSMLA's shared-basis hypothesis from a different angle. A3=3 because the GO/NO-GO threshold is numerical and the NO-GO outcome (μ_E ≈ random-expectation 128/2048 = 0.0625) itself closes a structural question about embedding-attention coupling. Load-bearing primitive: alignment of E rows to the W_Q principal subspace.

**C14-CFSHIFT** — Frame-novelty advancer. A2=3: deriving a constant attention logit bias from the preceding layer's calibration-fitted gate output is genuinely novel framing — no published method propagates CF6's c_1 as a structural Layer-2 logit perturbation. The math is a direct linearity consequence (W_Q distributes over h_1 + c_1); the cross-term δq_c1^T k_2^{base} is the dynamic component. A4=2 because the c_1 term is a mean-field approximation; the RoPE treatment of δk_c1 is noted as checkable but not fully audited. Smallest test: ratio ‖δq_c1‖₂ / mean ‖W_Q^{L2} h_2‖₂ per head, ~30 min. Pre-emption resistance: ALiBi adds position bias; this adds a calibration-content bias from a prior layer's gate — no published method.

**C15-SNRATE** — A3=3 from a clean discriminative experiment (W_Q at 2 bpw vs W_gate at 2 bpw, 40 min, GO = 2× ΔNLL ratio). CF5 (W_up rank sensitivity) and CF11 (W_Q concentration) are the two founding findings; SNRATE builds a joint budget oracle on their product. The closed-form bpw schedule replaces Hessian calibration with two spectrum measurements — a clear engineering payoff. CF10 compliant (3 params, 5 data points). Load-bearing primitive: SNR(concentration/sensitivity) as per-class quantization budget signal.

**F1-MQKP** — Strongest single proposal in the union. A3=3: 15-min pilot (3 layers, truncated SVD of M = W_Q W_K^T) is a clean binary settler; NO-GO is itself a structural finding (W_Q and W_K are spectrum-aligned). A4=3: Eckart-Young theorem applies trivially to any real matrix — no precondition failure possible. A2=3: no prior F-orientation (runs 001-008) computed the product spectrum; every run compressed factors independently. The structural claim (M may have r_99/d < min(r_Q, r_K)) is a genuine algebraic question with an auditable answer. Load-bearing primitive: the attention score operator W_Q W_K^T as the compression target.

**F2-CKDJ** — Frame-novelty advancer. A2=3: coupling CF3's static outlier channels (K=0.1%, Jaccard=0.718) to CF11's W_Q column norms is a precision-allocation identity that no published method derives — the softmax shift-invariance strip for a calibration-identifiable constant input component is lossless and novel. A3=3: 35-min experiment (calibration forward pass + column norm comparison) with a binary outcome (static channels are / are not high-energy W_Q columns). Load-bearing primitive: static outlier channels as zero-cost W_Q constant-component strippable via softmax shift-invariance.

**F3-L1GNF** — CF6-anchored (Layer-1 36% foldable). A4=3: the null-space criterion (foldable gate rows lie in null(C_x^0)) is constructive and auditable in 10 lines of NumPy. A3=3: 25-min experiment with quantitative effect-size threshold. The null-space explanation is more general than the CF6 observation alone — if confirmed, it predicts fold eligibility from weight geometry without calibration. Load-bearing primitive: null-space alignment of gate rows to input covariance as fold-eligibility criterion.

**F4-WQKRS** — A4=3: water-filling is a classical result with trivially satisfied preconditions (continuous spectrum, separable matrices). A3=3: 30-min pilot (3 layers, both schedules, NLL comparison with quantitative 0.05-nat threshold). Grounded in CF11 asymmetry (r_99/d = 0.63 vs 0.79). The asymmetric allocation is the structurally correct answer to "given joint bit budget, how split between W_Q and W_K?" — not published in attention-compression literature. Load-bearing primitive: spectral asymmetry of W_Q vs W_K as input to a water-filling allocation.

**F5-RSUC** — Convergence advancer. Depth-dependent residual update covariance converges with R9-R4's NVMe-scheduling intuition (early vs late layer access patterns) and with C17-LAYERTAX's layer-characterization-score concept. A3=3: 25-min experiment, quantitative threshold (≥40% variation from L1 to L14). Load-bearing primitive: per-layer residual update covariance rank as a depth-stratified compression budget.

**F6-TROPICALDECODE [FREE SWING]** — Convergence advancer with A1-TROPFFN. A2=3 (tropical algebra applied to attention argmax is genuinely outside the published ML compression frame). A4=2 (max-plus GEMV is a standard operation; no fragile theorem import — CF9 not triggered). The convergence across F and A orientations on tropical semiring as an inference primitive is the signal. Smallest test: 20-min attention concentration measurement at Layer 1. Load-bearing primitive: one-hot-like attention distributions enabling exact tropical approximation.

**U1-B-ETWO** — A2=3: ETW file-I/O kernel trace as a tensor-granularity access histogram is a genuine substrate-novel idea — the OS event infrastructure has been doing this for 20 years and no published LLM paper uses it. A4=2: the ETW/page-fault fallback is correctly specified; primary risk (mmap emits page-fault not ReadFile) has an explicit mitigation. Smallest test: 3-hr ETW trace + histogram stability check. Load-bearing primitive: kernel-observable I/O event stream as zero-overhead access oracle.

**U5-A-VGTP [FREE SWING]** — A2=3: VEH guard-page interposition for per-block precision promotion has no kill-list analog and no published ML inference cousin. A4=2: VirtualProtect + AddVectoredExceptionHandler are fully documented; the page-fault-on-NVMe-backed-page risk is correctly identified and mitigated. Scale limitation (sub-7B only due to VEH round-trip overhead at 70B) docks A1 to 1. Structural floor met; advances as wildcard. Load-bearing primitive: hardware page-fault → VEH chain as per-block precision promotion trigger.

**A2-APPENDKV** — A2=3 (append-only KV with content-addressable retrieval is a constraint-forced architecture orthogonal to the published eviction-policy space). A4=2 (LSH over INT8 keys is a well-understood primitive; RoPE-stripping mitigation is correctly identified). Smallest test: cosine similarity of LSH-retrieved attention output vs exact attention, ~3 hrs. The constraint cleanly rules out 60% of published KV management approaches. Load-bearing primitive: append-only NVMe-resident KV log with DRAM-resident LSH index.

**A3-FSMDEC** — A2=3 (discrete FSM routing keyed on residual-stream cluster membership is novel; Deja Vu uses continuous predictors). A3=2 (1-hr experiment with quantitative firing-recall threshold). CF3's static outlier channels at K=0.1% as the state-assignment basis is the CF-connection. Load-bearing primitive: residual-stream centroid cluster as a zero-cost discrete routing key.

---

## 3. Convergence Map

**Convergence C1 — Attention-weight spectrum as joint compression primitive.**
R9-R1 (ATQKV: W_V/W_O spectrum audit), F1-MQKP (product M=W_Q W_K^T spectrum), F4-WQKRS (water-filling allocation), C13-EQSUB (embedding-row projection mass into W_Q subspace), R9-R3 (HSMLA shared-basis). All five independently arrive at the attention weight matrices as the most compressible class, each proposing a different angle on the same question: what is the compression-optimal treatment of the four attention matrices? Strongest representative: F1-MQKP (A4=3, 15-min settler, closes a structural question no prior run asked). Cheapest settler: M^L spectrum at 3 layers, ~15 min (resolves whether joint product compression beats independent factor compression — the key unasked question for all attention-matrix ideas).

**Convergence C2 — Layer heterogeneity as per-layer compression budget signal.**
F5-RSUC (residual update covariance depth profile), C17-LAYERTAX (v2-CF1 Jaccard × CF6 gate-variance joint LCS), R9-R4 (attention-MLP interleave exploiting layer depth). All three independently assert that layers are not compression-equivalent and that a depth-aware schedule dominates uniform compression. Strongest representative: F5-RSUC (A3=3, cleanest measurement, 25-min, most directly falsifiable). Cheapest settler: residual update covariance r_90% across all 28 layers — if profile is flat, C2 dies uniformly; if stratified, it motivates C17-LAYERTAX's LCS.

**Convergence C3 — Tropical semiring as inference alternative computation.**
F6-TROPICALDECODE (attention argmax via tropical max-plus) and A1-TROPFFN (FFN dispatch via tropical semiring) arrive independently at the same mathematical object. Different targets (attention vs FFN), same algebraic frame. Strongest representative: A1-TROPFFN (narrower claim, faster to test, AVX2 path validated). Cheapest settler: attention concentration measurement at Layer 1 (~20 min, settling whether tropical approximation is even worth pursuing for attention) — if layer-1 attention is diffuse, F6 dies but A1 proceeds independently.

**Convergence C4 — NVMe as a compute-aware weight bus.**
R9-R4 (interleaved layout for prefetch overlap), A5-REGONLY [rejected on score], A2-APPENDKV (NVMe-resident KV log). Multiple orientations independently force NVMe into the critical compute path. The shared primitive is the NVMe bandwidth budget as a first-class design constraint, not a fallback. R9-R4 is the strongest representative. Cheapest settler: CF13* re-derivation (10-min NVMe sequential read benchmark under actual inference load).

---

## 4. Frame-Novelty Bonus Advancers

**Slot 1: C14-CFSHIFT** — Frame: propagation of a calibration-derived mean-field vector through a linear layer as a structural attention logit bias. This frame has no published cousin in the ML compression/attention literature. ALiBi adds position-dependent biases; SmoothQuant migrates channel scales; neither derives a constant cross-layer logit perturbation from gate-output mean-field approximation. The frame is in-orientation (Composition of CF6 and CF11), and the specific algebraic consequence (δq_c1^T δk_c1 / √d as a constant scalar per head) is auditable in 1 paragraph. A4=2 (checkable but mean-field approximation is an unverified step). A3=2 (30-min experiment with quantitative ratio threshold).

**Slot 2: F2-CKDJ** — Frame: softmax shift-invariance as a lossless strip triggered by the empirical co-occurrence of (a) static outlier channels (CF3 K=0.1%) and (b) high-energy W_Q column directions (CF11). The joint claim is a structural composition whose algebraic identity (constant-input contribution strippable by adding −W_Q·(c_static) to the query bias) has no published treatment. CF9 check: no imported theorem with fragile preconditions — softmax shift-invariance (adding a constant to all logits = no softmax output change) is a trivially satisfied identity. The frame-novelty is specifically the coupling of two independent empirical findings into a zero-cost computation-graph identity.

---

## 5. Rejected with Rationale

- **C16-RSEMBED** — A4=1 only: the claim that cos(h_L, h_0) ≥ 0.60 for ≥15% of tokens is a hypothesis about compounding CF2 across 28 layers. CF2 measures *adjacent-layer* cosine at 0.99; the compound over 28 layers may plausibly be 0.75 (the proposal acknowledges this) but may equally be near-zero if residual rotations accumulate. The proposal correctly flags this as a risk and defers to measurement, but without A4 ≥ 2 the mechanism is not checkable. The skip-gate payoff (≤1% speedup at best) is also below the residency bar. Stage 4 may rescue: **stripped primitive: cos(h_L, h_0) distribution over inference is a real structural measurement; Stage 4 may use it as a grounding input for a logit-shortcut idea if the distribution turns out to be concentrated.**

- **U2-B-ADST** — A1=1 (12 KB tier map, negligible residency impact). A2=2. Mechanism is a correct and interesting observation about NTFS ADS as out-of-band metadata, but the residency/throughput payoff is near-zero on any practical model. Useful as an engineering pattern but not a structural finding. Stage 4 may use the ADS sidecar pattern as an implementation convenience for RAOK or PDAP tier maps without advancing it separately.

- **U3-B-FIOP** — A1=1 (6% mean throughput improvement, p95 variance reduction only). A3=2 (experiment well-structured but the mechanism is a latency-variance reducer, not a structural finding). Correctly scoped but below the structural-finding bar. Stage 4 may bundle this as an engineering recommendation alongside any NVMe-tier idea without separate advancement.

- **U4-B-OSRD** — A1=1 (30% speedup only for Qwen3-8B, not the 70B target). A2=2. The mechanism (OS RAM disk for partial layer residency) is structurally similar to VirtualLock but using a virtual block device — a real implementation difference but not a structural finding. Stage 4 may treat this as an implementation alternative to VirtualLock for the hot-zone pinning problem.

- **A4-SEEDNET** — A4=1: the hypernetwork G_θ fit is specified but the CF10 check for MLP weights is not closed (CF8 shows W_up/W_gate are full-rank — a 5M-param network fitting 1.7B weight values where cross-layer regularity is minimal for MLP is likely to produce high reconstruction error without any structural motivation). The W_Q-only experiment is valid (CF11 provides structural motivation for cross-layer W_Q regularity), but A1=2 is generous for a 20× reconstruction error tolerance. Stage 4 may rescue: **stripped primitive: cross-layer W_Q predictability from a compact hypernetwork is a real open question; if MQKP (F1) shows M's spectrum is concentrated, G_θ targeting M^L across layers becomes better motivated.**

- **A5-REGONLY** — A1=1 (re-derives Apple LLM-in-a-Flash architecture), A2=1 (the constraint-alien framing adds conceptual clarity but not a new mechanism), A5=1 (LLM-in-a-Flash does exactly this on Apple silicon; the Windows/NVMe instantiation is incremental). Correctly tagged [FREE SWING] by the ideator but does not meet the structural floor of contributing a new primitive. **Not a kill-list candidate — defers to the Apple paper's architecture as prior art.**

---

## 6. Hand-off to Stage 3

**Track A advancers (4 + convergence reps + bonuses):**

1. **F3-L1GNF** (F, A, total 11) — CF6-grounded Layer-1 gate null-space fold. Stage 3 pressure-test: does null(C_x^0) correctly predict the CF6-foldable neurons, and does the fold incur measurable quality loss on an OOD token distribution that activates those neurons? Verify C_x^0 conditioning at 200 vs 500 tokens.

2. **R9-R3 / HSMLA** (R, A, total 10) — Post-training shared-basis MLA from CF11 head-redundancy. Stage 3 pressure-test: can the 16 per-head W_Q matrices be factored into a single shared 128-dim basis via stacked SVD, or do the per-head principal directions diverge despite the global 128-dim GO? Compute Grassmannian principal angles between head pairs first.

3. **C14-CFSHIFT** (C, A, total 11) — Frame-novelty bonus. Stage 3 pressure-test: is the mean-field approximation for c_1 stable enough (std < 10% of mean) to make δq_c1 a real constant rather than a token-dependent noise term? Does the RoPE rotation of δk_c1 preserve the precomputation claim at position p?

4. **A2-APPENDKV** (A, A, total 10) — Append-only KV with LSH retrieval on NVMe. Stage 3 pressure-test: does RoPE positional encoding break the LSH index (keys at different positions are geometrically distinct even for semantically similar tokens)? What is the retrieval accuracy gap at 4K vs 32K context?

5. **F2-CKDJ** (F, A, total 12) — Frame-novelty bonus. Stage 3 pressure-test: are the CF3 K=0.1% static channels also top-5% W_Q column-energy channels? If YES, the softmax strip is lossless. If NO, the two findings are decoupled — itself a structural closure for this class of ideas.

**Track B advancers (4 + convergence reps + bonuses):**

1. **F1-MQKP** (F, B, total 13) — Attention product M = W_Q W_K^T spectrum. Stage 3 pressure-test: compute M's spectrum at 3 layers (15 min); if r_99/d < 0.55, it dominates all per-factor compressions and becomes the anchor for the Track B attention-compression cascade. If NOT, closes the joint-product compression class.

2. **F4-WQKRS** (F, B, total 12) — Water-filling asymmetric rank allocation for W_Q/W_K. Stage 3 pressure-test: quantify the quality gain of water-filling vs symmetric K at matched bit budget across all 28 layers. Is the 0.05-nat improvement threshold achievable, or is the attention score quality insensitive to rank asymmetry?

3. **R9-R1 / ATQKV** (R, B, total 12) — W_V/W_O spectrum audit. Stage 3 pressure-test: W_V is the highest-risk matrix (may be full-rank due to value-content preservation). If W_V r_99/d ≥ 0.90, the four-matrix joint compression cascade is blocked; the residency arithmetic shrinks to W_Q + W_K only.

4. **C15-SNRATE** (C, B, total 11) — SNR(concentration/sensitivity) precision budget oracle. Stage 3 pressure-test: does rank-truncation sensitivity (CF5, CF11) correlate with quantization sensitivity at 2 bpw? The key failure mode is that SVD truncation and uniform grid rounding have structurally different error distributions — the experiment directly tests whether the correlation holds.

5. **F5-RSUC** (F, B, total 11) — Convergence rep for C2 (layer heterogeneity). Stage 3 pressure-test: measure r_90%(L) across all 28 layers on 3 disjoint calibration batches. If profile is flat, C17-LAYERTAX and R9-R4's depth-stratified scheduling are both undermined.
