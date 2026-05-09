# Stage 4 — Skeptic Explorer — Run 006

**Agent**: Sonnet claude-sonnet-4-6 | **Date**: 2026-05-09
**Kill-list enforced**: v2-CHEAP-TEST-001; v2-S3-R004-001; CF8 (MLP weights full-rank); CF10 conditioning; CF9 precondition gate.
**Run 006 signal**: 12 REFINE, 0 KILL. Convergence clusters: W_V/W_O spectrum (C2, gates 3 proposals), CF6 Layer-1 anomaly (C-L1-ATTEN-GATE rejected), NVMe async (FNDP + WMCE composition). Top 3 per Stage 3: R6-WVOP, C-JOINT-DELNLL, F6-WALIGN.

---

## 1. Frame-Exhaustion Map

### Saturation-exhausted frames

**Frame SE-1: Post-training W_Q/W_K SVD rank reduction (per-matrix).** CF11 is confirmed; AQFKV is run. Any further "compress W_Q alone" or "compress W_K alone" variant is covered. The product-fold W_VO and joint Q+K interaction coefficient are the still-open extensions, both advancing in Stage 3. New ideas in this frame must be structurally orthogonal (e.g., the precision-allocation-per-column angle from v2-S3-R009-001's stripped primitive is alive, but per-matrix SVD truncation has no open runway). Stripped primitive: **per-column W_Q energy profile** (which channels W_Q loads most from).

**Frame SE-2: Cross-layer W_Q stacking / shared basis.** Class-killed by v2-CHEAP-TEST-001 with permutation control. Stripped primitive: **within-layer cross-head factorization** (MLA-POSTRAIN direction, which Stage 2 rejected on weak novelty vs TransMLA/PALU, but whose CF11-derived rank anchor remains unexploited).

**Frame SE-3: Direct NVMe throughput improvement via standard IOCP/async-ReadFile overlap.** Runs 003-005 exhausted the overlap-scheduling axis. Stage 2 advanced FNDP (Cache Manager bypass) and WMCE (Compression Store tier) — both are genuinely different. The "issue multiple requests in parallel / increase queue depth" axis is exhausted. Stripped primitive: **double-buffer DMA to user-mode aligned buffers**, which is exactly what FNDP captures.

**Frame SE-4: lm_head SVD truncation in tied-embedding configs.** CF12 killed this with catastrophic ΔNLL at r=1024. Stripped primitive: **untied lm_head spectrum** (Qwen3-8B+), which is explicitly flagged as open in SUMMARY.md and has no Stage 1 proposer in run 006.

### CF9-exhausted frames

**Frame CF9-1: Sparse-signal sketches on dense targets.** Count-Sketch (CSKQ run 006-B) already killed on this precondition; INT4 residuals are dense. Any sketch whose error bound requires sparsity (L1-based, median-of-medians) is dead for LLM weight matrices. The surviving stripped primitive here is **locality-sensitive hashing on dense embeddings for vocabulary pre-routing** — LSH has O(1) lookups without sparsity requirements. The LSEI logit shortlist (R2/S4-deferred) was in this territory; it remains unexecuted.

**Frame CF9-2: Shift-invariant kernel approximations on weight matrices.** RFF (RFSK run_006-B) was killed; Bochner's theorem requires shift-invariant kernels and weight matrices are not kernels. The stripped primitive — **JL random projection** as a dimensionality reduction — is dead because SVD strictly dominates JL for rank-k approximation of full matrices (and SVD was already proven insufficient for MLP weights by CF8). Closed for MLP weights; open for activations (random projection of the residual stream for cheap routing is untested).

### CF10-exhausted frames

**Frame CF10-1: Full-rank affine cross-scale maps with small calibration.** WDLA died here. A1-PEER (run 006 Stage 1) was advanced with a CF10-compliant rank-64 construction. Any new proposal that fits an affine A: R^d → R^d on standard calibration sets without explicit rank-by-construction enforcement is CF10-exhausted. The stripped primitive: **low-rank calibration-fit operators (rank ≤ 64) on cross-scale or cross-layer representations**, which is A1-PEER's territory.

---

## 2. Orientation-Saturation Diagnostics

**Orientation R (Reach):** Advanced 5 ideas to Stage 3 (WVOP, ATTNQ-FULL, DSAF-70B, RAOK-CF3EXT, WDOWN-CONDP); all received REFINE or advanced. R6-DSAF-70B was subsumed by convergence C1 (C-DEEP-SPREAD-QBUD is the stronger rep), and R6-RAOK-CF3EXT was rejected as premature (RAOK base not yet run). The orientation is producing high-quality convergence material. Its ideas concentrate on the W_V/W_O cascade and on the attention-compression-to-70B argument. KEEP. Anti-frame addition: **"attention matrix compression without the W_V/W_O gate measurement"** — any future Reach attention idea that does not acknowledge the W_V/W_O gate (C2 convergence) as a prerequisite is wasting a cascade rung.

**Orientation C (Composition):** Advanced all 5 Stage 1 ideas; C-JOINT-DELNLL is the highest-scored idea in the run (13/15). The coupling-equation approach is working well. Two proposals (KSTAT-QDYN, DEEP-SPREAD-QBUD) concentrated on v2-CF1 × CF11 compositions. Two ideas (C-L1-ATTEN-GATE, C-L1-focused things) were rejected for low residency payoff. KEEP. Anti-frame addition: **"per-layer single-matrix sensitivity probes without a coupling equation"** — measuring per-layer W_Q sensitivity alone is not in-orientation; the coupling must compose two findings into an unexpected third structure.

**Orientation F (First-Principles):** Advanced 3 of 5 ideas (WALIGN, GQARED, SINKCONST); DHSUB and WDCOL were rejected (low A1). F is generating algebraic-identity class ideas (WALIGN) and structural measurement ideas (GQARED, SINKCONST) but not unifying-object derivations in the style of the orientation's creative directive. The orientation produced no proposal that derives a joint diagonalization, gauge symmetry, or single-object unification across CF1–CF12. TIGHTEN. Anti-frame addition: **"measurement-only first-principles proposals that lack a unifying algebraic object"** — DHSUB (measure ΔH_L subspace) and WDCOL (measure W_down column projection) are measurements, not first-principles algebraic identities. The orientation should produce ideas of the shape "CFs i,j,k are projections of one operator X; run inference in X's eigenspace."

**Orientation U (Unconventional Substrate):** Advanced 2 of 6 ideas (FNDP, WMCE); both with caveats. The orientation is constrained by prior-round coverage (13 prior U axes exhausted). U6-CSDL (CPU C-state) and U1-IOPR (I/O priority) were rejected for low impact. U5-SCVP (CF3-grounded VirtualLock) and U4-RFSW (ReadFileScatter) were rejected as not standalone. The orientation is running low on unexplored substrate axes for the Ryzen/Windows target. TIGHTEN. Anti-frame addition: **"substrate proposals whose impact is solely latency variance reduction without a measurable throughput floor"** — IOPR and CSDL both fit this; the orientation should target primitives that reduce bytes-transferred or increase bandwidth utilization, not noise-floor improvements.

**Orientation A (Constraint-Alien):** Advanced 2 ideas (A1-PEER, A6-LLMR); A1-HPCI and A3-IDPT and others were rejected for low quality. The constraint-alien orientation is generating proposals that are mechanistically creative but have lower A1 scores (impact ceiling) than R/C/F. The permutation-canonical ordering (A5-GRPN) and tropical MLP (A4-TMLP) were not in the Stage 2 union table, suggesting the Stage 1 A ideator may have generated ideas not captured in the run_006 Stage 1 file. The constraint menu items used heavily in runs 001-005 include tropical, append-only, FSM, content-addressable. KEEP. Anti-frame addition: **"constraints that produce mechanisms equivalent to existing soft approximations"** — A2-SINT (shift-canonical keys) reduces to a coordinate change with INT7 benefit; A5-GRPN (permutation canonical form) might reduce to content-adaptive column loading. The constraint must force a mechanism the standard playbook has no words for at the inference-computation level, not just a coordinate choice.

---

## 3. Cross-Pollination Opportunities

**XP-1: WMCE precondition failure → FNDP+WMCE composition (orientation U → Track B hybrid).**
WMCE was advanced with REQUIRES-MITIGATION because its M1 precondition (Compression Store for mmap pages) likely fails. The surviving mechanism after stripping the failed precondition: anonymous-memory loading (FNDP's mechanism) + OS-automatic Compression Store on cold anonymous pages. This is a complete composition: FNDP provides the anonymous pages that make weights Compression-Store-eligible; WMCE provides the DRAM-tier expansion. The composite is stronger than either alone. Stage 4 should present FNDP+WMCE as a single Track B NVMe path proposal rather than two separate experiments.

**XP-2: F6-DHSUB stripped primitive → Composition orientation (Track B activation precision).**
DHSUB was rejected (A1=1, small bandwidth saving). Its stripped primitive — per-layer residual update subspace ΔH_L effective rank — is valuable as a RAOK tier assignment primitive. If ΔH_L has low effective rank (most of the per-layer update lives in r_eff dimensions out of 2048), the pass-through coordinates (outside the update subspace) carry no within-layer information. This is directly compositionally relevant to RAOK: the pass-through coordinates are precisely those that should be stored at the lowest precision (INT4 or less) because the layer does not modify them. The coupling: CF2 (small Δh magnitude) + ΔH_L subspace measurement → pass-through coordinate identification → RAOK tier-3 coordinate assignment. This is a C-orientation composition that produces a tighter RAOK tier boundary than the current activation-magnitude-only criterion.

**XP-3: A3-IDPT (idempotency probe) → F-orientation unifying object.**
A3-IDPT was not in the Stage 2 union table, but its stripped primitive — Jacobian-nullspace alignment of layer updates — is relevant to F-orientation. The idempotency condition ‖J_L Δ_L‖_F / ‖Δ_L‖_F ≤ ε defines a structural class of "nearly-projection" layers. If the F-orientation were to ask "what is the mathematical object that CF2 + idempotency + GQA head redundancy (CF11) all constrain jointly?", the answer might be a low-dimensional attractor manifold in the residual stream. This is a stronger unifying-object shape for F-orientation than what run 006 F produced. Not a complete Stage 4 idea by itself, but motivates idea S3 below.

---

## 4. Cross-Domain Seed Exploration

### Database engineer view

**What algebraic structure does a database engineer see in the attention mechanism?** A query-key matmul is a **join** between a query relation Q (n_q × d) and a key relation K (n_k × d). In database terms, the attention weight matrix A = softmax(QK^T/√d) is the output of a similarity join with a softmax normalization. Database engines accelerate joins via indices: for a low-dimensional join predicate, a B-tree or hash index pre-clusters the matching keys. CF11's head-redundancy finding (16 queries live in a 128-dim subspace) means the effective join dimension is 128, not 2048. A **blocking scheme** on the key space using the top-128 singular vectors of W_Q pre-clusters keys into ~16 buckets (one per effective query direction); attention computation then only contacts keys in the matching bucket. This is not approximate nearest-neighbor search (which is saturated in NLP) — it is a **partition-join** that exploits the algebraic structure identified by CF11. The elegant-equivalence question: does the low-dimensional join structure allow attention to be computed in two stages (bucket lookup + softmax within bucket) without approximate errors, because the softmax over all keys is already concentrated on the dominant bucket by the CF11 head-redundancy? Precondition check: CF11's K=128 GO at +0.98 nats means the K=128 subspace captures most of the query information, but there are still tails. The partition-join would incur approximation errors similar to CF11's +0.98 nats ΔNLL — not free. CF9: no external theorem imported; the partition-join is derived directly from CF11. This is **idea S2** below.

### Compiler writer view

**What dead-code elimination does a compiler writer see?** A compiler eliminates computations whose output is not used downstream. In the transformer forward pass, is any intermediate value computed but not used? CF6's Layer-1 gate anomaly (36% of gates near-constant) is exactly dead-code from a compiler's view: if silu(W_gate[i,:]·x) ≈ c_i for all x on the calibration distribution, the neuron's output is approximately c_i · (W_up[i,:]·x), which is equivalent to applying a rescaled row of W_up with a scalar multiplier. The "gate" computation is dead code — the gate's contribution to the output is a compile-time constant that can be folded into the W_up row at load time. CF8 says W_gate globally cannot be rank-reduced, but Layer-1 specifically (CF6: 36% foldable) is the exception. The compiler fold: for each Layer-1 neuron with silu variance < ε, replace the W_gate[i,:] row with a single scalar and fold it into W_up[i,:] at model-load time (AERO-style local fold, no retraining). This is LCAB (Track A R2 Stage 4 deferred) — already proposed and deferred. Not a new idea; absorbed into the composition below.

**Compiler loop-hoisting**: what invariant computations execute every token that could be lifted out of the loop? The BOS key vector k_sink = W_K · e_BOS is a loop-invariant (it is the same on every token since e_BOS is fixed). This is exactly F6-SINKCONST, already advanced. Compiler view corroborates the proposal.

**Dead-store elimination on the lm_head**: in tied configurations (CF12), every direction of the embedding matrix is load-bearing (catastrophic at r=1024). But in untied lm_heads (Qwen3-8B+), if some output directions are never the argmax-winner for any token, those rows of lm_head are dead stores. The vocabulary size is 151936; many tokens in the long tail may never win in typical text. Measuring the **active vocabulary fraction** (fraction of vocab tokens that are ever top-1 argmax in a large corpus) gives the dead-store count. Dead-store rows can be dropped from lm_head storage. This is partially covered by LRALP (deferred, Track A R3) and the LSEI logit shortlist (R2/S4-deferred), but the dead-store framing from the compiler view adds a distinct falsifiable claim: lm_head rows corresponding to the bottom 30-50% of vocabulary by corpus frequency are dead stores for typical inference distributions. This motivates **idea S4** below.

### Control theorist view

**What observer design does a control theorist see?** A Kalman filter maintains an estimate of hidden state and a covariance matrix. The residual stream h_L is the state estimate; the per-layer update Δh_L is the measurement. CF2 (small Δh) is the observer convergence condition: the state estimate is already near the "truth" after early layers. A control theorist would ask: **what is the minimum-norm control input (Δh_L) that steers the residual stream from its current state to the target (the final representation)?** Minimum-norm control is achieved by a Moore-Penrose pseudoinverse, which is exactly the low-rank structure of the per-layer update. The elegant-equivalence: if each layer's Δh_L is minimum-norm, then the per-layer update subspace (DHSUB's object) is as small as possible — and each layer contributes orthogonally to what previous layers have already achieved. This is not a compression mechanism directly, but it provides a structural prediction: ΔH_L should be lower-rank than H_L itself, and the rank of ΔH_L should decrease as L increases (diminishing returns per layer). This is DHSUB's prediction, which was rejected for low impact. Control theory confirms the prediction is mechanistically motivated but doesn't add a new compression angle not already in DHSUB.

### Network coder view

**What fountain code structure does a network coder see?** Fountain codes (LT codes, Raptor codes) encode k source symbols into a rateless stream of coded symbols; any k received symbols can decode the original. Applied to LLM weight storage: if each row of W_up can be encoded as a linear combination of a small common codebook (a generating matrix), then the model can be stored as codebook + per-row coefficients. This is the CLUBS/RSTD territory, killed by CF8 (W_up full-rank, no generating matrix with k << d). The fountain code angle adds nothing new for MLP weights.

**Applied to the KV cache instead**: if the KV cache rows at each layer can be written as linear combinations of a small set of "base keys" (the static channels from CF3), then the KV cache is compressible using a fountain-code-like scheme where the static base vectors (CF3's K=0.1% channels) are the systematic part and the dynamic channels are the coded redundancy. The KV cache is at 4K context ~670 MB — not the dominant bottleneck. This is low-priority.

**Highest-leverage professional prior for this round:** The database engineer's partition-join view (CF11 → bucket-based attention) and the compiler writer's dead-store view (lm_head active vocabulary fraction) are the two most actionable cross-domain seeds. Both produce falsifiable claims from existing CF anchors.

---

## 5. Constraint-Driven Reframing

**Constraint: Storage is sequential-only (no random reads during inference).** This forces every weight access pattern to be sequentially readable. Under this constraint, random-column access of W_down (for WDOWN-CONDP's H/C partition) is inadmissible — the hot H columns are scattered randomly across the weight tensor. The constraint forces: **repack W_down at model-load time so all H columns (top-17% by mean post-activation magnitude) are contiguous at the start of the W_down tensor, followed by all C columns.** At inference, read W_down sequentially: load the H block (17% of bytes) first, execute the H-partition GEMV, then continue loading the C block. This is a GGUF tensor repacking decision, not a new mechanism — it removes the random-column-access risk from WDOWN-CONDP without changing the math. It also enables prefetching: since the H block is always first, it can be prefetched while the previous layer runs. Stripped mechanism: **H/C column-sorted W_down layout as a GGUF tensor packing specification.** No new Stage 4 idea needed; this is a refinement note for WDOWN-CONDP.

**Constraint: No floating-point at inference (only integer arithmetic + lookup tables).** This forces the attention mechanism into discrete arithmetic. Under this constraint, softmax is inadmissible (requires exp). The only surviving attention variant is **max-product attention** (tropical-semiring softmax: replace exp-and-normalize with argmax). This was partly explored in run 006 Stage 1 A-SINT (shift-canonical keys) and A4-TMLP (tropical MLP). The attention version: replace softmax(QK^T/√d) with argmax(QK^T), producing a hard-routing attention that selects exactly one key per query. Under the no-float constraint, this is computable as INT16 dot-product comparisons (no divisions, no exp). CF9 check: does tropical attention have a functional precondition that may not hold? The issue: argmax attention is the zero-temperature limit of softmax; trained Qwen3 uses softmax. The quality degradation of replacing softmax with argmax is the open empirical question (similar to A4-TMLP for the MLP). This motivates **idea S6** (the tropical attention FREE SWING) below.

**Constraint: 10 MB seed reconstruction.** Forces all 70B weights to be derivable from a 10 MB description. Under this constraint, the only viable mechanism is a **structured weight generation procedure**: a small seed network generates the full weight matrices via deterministic expansion. CF11 provides a leverage point: W_Q lives in a 128-dim global subspace. If a seed network can generate the 128-dim W_Q basis for each layer from a 10 MB description (128 × 2048 × 28 × 2B = ~14 MB — already too large), this fails. The constraint is too tight for the current architecture. Skip; the mechanism class (hypernetwork synthesis) is in the kill list from prior runs and CF8's full-rank MLP weights preclude effective compression via seed generation.

**Highest-leverage constraint for this round:** The sequential-only constraint produces the H/C layout refinement note (not a new idea). The no-float constraint produces the tropical attention idea S6. These are the actionable outputs.

---

## 6. Generated Ideas (7 total)

### S1 — FNDP-WMCE Composition: Anonymous Weight Loading with OS-Tier DRAM Compression
**Track B | Orientation U composition | Motivated by XP-1 cross-pollination**

**Ambition.** Replace ik_llama.cpp's mmap-based weight loading with a FILE_FLAG_NO_BUFFERING + VirtualAlloc double-buffer path, which (a) eliminates the Cache Manager kernel-copy overhead (FNDP) and (b) makes the loaded weight pages anonymous-memory-eligible for Windows Compression Store automatic DRAM compression of cold layers (WMCE). Together: a free ~20% throughput improvement from eliminating the double-copy (FNDP component) + an effective DRAM-capacity expansion of ~1.5× for the cold weight pages (WMCE component), at zero quality cost.

**Mechanism.** At model load, open the GGUF file with FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED. Allocate a double-buffer of sector-aligned anonymous memory (two slots, each sized for one transformer layer). Implement a producer-consumer loop: while GEMV runs on buffer A, ReadFile fills buffer B via DMA directly from NVMe to user-mode aligned buffer (no Cache Manager). Weight pages are now anonymous — eligible for Windows Compression Store under memory pressure. The Windows Memory Manager automatically compresses cold anonymous pages at ~1.5–2.5× when free DRAM is scarce, keeping recently-accessed layers hot in uncompressed form and cold layers compressed in DRAM rather than evicted to NVMe.

**CF grounding.** No CF anchor required for the substrate mechanism. The precondition check: (a) FILE_FLAG_NO_BUFFERING is standard Win32; (b) Windows Compression Store for anonymous pages is documented behavior. The FNDP component re-derives a v2-CF13'' NVMe throughput measurement as a byproduct. CF9: no imported theorem.

**What must be true.** (1) Cache Manager double-copy is measurable overhead: NVMe bandwidth < DRAM bandwidth on this hardware (3 GB/s << 11.5 GB/s confirms copy overhead is ~3/11.5 ≈ 26% of layer read time). (2) Windows Compression Store engages for anonymous pages under memory pressure — documented. (3) INT4-quantized weight data achieves ≥1.3× compression in the Store (this is the unknown; INT4 data after quantization is denser in entropy than raw BF16, which may limit Store compression ratio).

**Experiment cascade.** E0 (45 min): standalone C harness — 200 MB sequential read in two modes: standard ReadFile vs FILE_FLAG_NO_BUFFERING + sector-aligned buffer. Measure throughput. GO if ≥15% improvement (FNDP component). E1 (1 hr): RAMMap inspection after anonymous loading + memory pressure — confirm Compression Store engages for anonymous pages AND measure compression ratio on INT4 GGUF tensor data. GO if ≥1.3× compression AND re-access latency ≤ 5 ms. E2 (4 hrs): integrate into ik_llama.cpp; implement double-buffer weight loading; benchmark tok/s on Qwen3-1.7B (DRAM-resident baseline) and Qwen3-72B (NVMe path). Total ≤ 8h for E0+E1; E2 is Stage 5 territory.

**What I am NOT proposing.** I am not proposing a custom NVMe driver, a RAID stripe layout, or a new OS-level page eviction policy. The mechanism uses exclusively standard Win32 APIs.

**Smallest test:** ≤8h. E0 alone (45 min) is the FNDP falsifier; E1 alone (1 hr) is the WMCE falsifier.

**Track B | Sits outside U orientation's "bandwidth amplification only" anti-frame addition because FNDP reduces bytes-through-DRAM and WMCE expands effective DRAM capacity — both are throughput levers, not variance-reduction.**

---

### S2 — QKPART: Partition-Join Attention via CF11 Query Subspace Bucketing
**Track A | Motivated by database engineer cross-domain seed | CF11 anchor**
**Elegance class: algebraic-identity (subspace-alignment)**

**Ambition.** CF11 establishes that 16 query heads collectively span a ~128-dim subspace. From a database-join perspective, the effective join dimension for query-key matching is 128, not 2048. This allows attention to be computed as a **two-stage partition join**: (1) bucket-assign each key vector to one of B buckets using its projection onto the 128-dim W_Q shared basis (computed once per token via a 128-dim GEMV), then (2) compute full softmax attention only within the matching bucket(s). Keys in non-matching buckets contribute negligibly to the softmax output because their projected distance from the query in the 128-dim subspace is large. The algebraic identity: if V_Q (128×2048) are the top W_Q singular vectors, and if keys are partitioned by argmax(k^T V_Q^T e_b) where e_b are B orthogonal bucket centroids in R^128, then the softmax over all n keys concentrates on the ~n/B keys in the matching bucket. Savings: B-fold reduction in K/V reads per token.

**Mechanism.** Pre-partition all KV cache entries into B=8 buckets at key-write time (when a new token is KV-cached): assign bucket = argmax over bucket centroids {c_1,...,c_8} ⊂ R^128 of (W_Q_V_Q · k)^T c_b. At query time: compute the query's bucket assignment (128-dim GEMV, ~0.05 ms), fetch only that bucket's K/V entries (1/8 of the KV cache), compute softmax over the sub-bucket. The bucket centroids c_b are derived from calibration data by k-means on the 128-dim projected keys.

**CF grounding.** CF11: W_Q global K=128 ΔNLL=+0.98 nats. The 128-dim basis captures most of the discriminative query-key separation. The partition-join approximation error is bounded by the fraction of attention mass assigned to non-matching buckets — empirically testable as the mean cross-bucket attention mass fraction. CF10 conditioning: k-means on 128-dim keys with B=8 centroids needs k-means with n_keys >> 8 samples; for 512-token context, n_keys=512 >> 8 — well-conditioned. No CF10 risk.

**What must be true.** CF11's 128-dim subspace captures ≥ 70% of query-key discriminative energy (i.e., the partition is well-separated in the 128-dim subspace). If keys and queries are nearly uniform in the 128-dim space after projection, the bucketing provides no concentration.

**Experiment cascade.** E0 (30 min): measure mean cross-bucket attention mass fraction on 200 calibration tokens at layer L14 — if top-1 bucket captures ≥ 50% of softmax mass, the approximation is usable. GO threshold: top-1 bucket mass ≥ 50% of total for ≥ 70% of (token, head) pairs. E1 (2 hr): implement and measure ΔNLL with B=8 bucket partition vs full attention on 512-token WikiText-2 evaluation.

**What I am NOT proposing.** I am not proposing approximate nearest-neighbor search (FAISS/ScaNN) or sparse attention patterns trained at training time. The bucketing is derived post-training from the calibration distribution using CF11's W_Q basis.

**Smallest test.** ≤8h. E0 at 30 min is the falsifier.

**Track A | Sits outside all orientation anti-frames (R's anti-frame: "cascades without a structural argument at one rung" — the partition-join identity is exactly the algebraic structural argument; F's anti-frame: "measurement without algebraic object" — the partition operator is the algebraic object; C's anti-frame: "stacking without coupling equation" — CF11 is the coupling).**

---

### S3 — JNULL: Layer Jacobian Near-Nullspace Alignment as a Depth-Stratified Compute Budget
**Track A | Motivated by XP-3 (A3-IDPT stripped primitive + F-orientation unifying object) | CF2 anchor**
**Elegance class: conserved-quantity**

**Ambition.** CF2 establishes that per-layer residual updates Δh_L are small (cos(h_L, h_{L+1}) ≈ 0.99). A control-theory interpretation: the transformer is near a fixed point of its own dynamics. Idempotency (from A3-IDPT's constraint) formalizes this: if J_L Δ_L ≈ 0, then applying the layer again costs nearly nothing. This motivates a **compute budget assignment** based on per-layer Jacobian-nullspace alignment: layers with ‖J_L Δ_L‖_F / ‖Δ_L‖_F < ε can be applied at reduced compute (e.g., INT4 instead of BF16 for W_Q and W_K, since the error in their GEMV outputs feeds into Δh which is in the near-nullspace of J_L). The unifying object: the Jacobian-nullspace defines a per-layer **compute slack** — directions in the output space where errors are self-corrected. The depth profile of this slack is a structural finding about how transformer layers modulate their own sensitivity.

**Mechanism.** For each layer L: compute the approximate Jacobian-vector product J_L Δ_L via finite difference (run layer once with h_L, once with h_L + ε·Δ_L; the difference is J_L Δ_L to first order). Measure the alignment ratio ‖J_L Δ_L‖_F / ‖Δ_L‖_F across 200 calibration tokens. Layers where this ratio < 0.10 have Δh in the near-nullspace — they are approximately idempotent. For those layers, assign a lower compute budget for attention (e.g., W_Q at K=64 rather than K=128) because the layer's own Jacobian will damp the additional error. The compression payoff: not all layers benefit equally from CF11's K=128 W_Q compression; near-nullspace layers can use K=64 (additional 2× compression over CF11) without the +1.53 nats ΔNLL seen globally.

**CF grounding.** CF2: small Δh (necessary condition for near-nullspace to be non-trivial). CF11: W_Q K=128 global GO, K=64 per-head NO-GO — but the K=64 was tested as a uniform policy; if near-nullspace layers apply K=64 at below-average quality cost, a mixed schedule (K=64 for nullspace layers, K=128 for others) may achieve lower total ΔNLL at the same bytes as uniform K=128. CF10: no calibration fitting; only a measurement. No CF10 risk.

**What must be true.** The Jacobian-nullspace alignment of Δh_L must vary across layers (some near-nullspace, others high alignment). If all layers have uniformly high alignment ratio, the mechanism provides no differentiation for compute scheduling.

**Experiment cascade.** E0 (4 hr): finite-difference Jacobian-vector products for all 28 layers × 200 calibration tokens. Measure per-layer alignment ratio distribution. E1 (1 hr if E0 GO): for layers with ratio < 0.10, apply K=64 (instead of K=128) for W_Q; measure ΔNLL of the mixed schedule vs uniform K=128 on 512-token WikiText-2.

**What I am NOT proposing.** I am not proposing to retrain layers or to cache Jacobian matrices at inference time. The measurement is offline; the compute budget assignment is a one-time calibration-derived schedule applied at inference.

**Track A | Sits outside F-orientation anti-frame "measurement-only without unifying object" because the Jacobian-nullspace IS the unifying algebraic object (the set of directions in the output space where the layer self-corrects errors). Sits outside A-orientation because it does not require a constraint binding.**

---

### S4 — VOCSKIP: Dead-Store Vocabulary Row Elimination from Untied lm_head
**Track B | Motivated by compiler dead-store cross-domain seed + CF12 stripped primitive | CF12 anchor**

**Ambition.** CF12 killed tied lm_head SVD truncation (Qwen3-1.7B). The surviving question: untied lm_head (Qwen3-8B+, Llama-3-70B) may be genuinely compressible. A different angle orthogonal to SVD truncation: **vocabulary dead-store elimination**. In any reasonable inference distribution, the bottom K% of the vocabulary by corpus frequency is never the argmax-winning logit. These lm_head rows are dead stores — they are computed but their output is never the argmax on any token. They can be dropped entirely (or quantized to INT2) with zero impact on greedy argmax. This is NOT lm_head SVD (killed for tied configs); it is vocabulary-selective row pruning based on empirical argmax-reachability.

**Mechanism.** Run Qwen3-1.7B on a large corpus (10K tokens from diverse domains). For each token position, record the argmax-winning vocabulary index. Track which vocabulary tokens are NEVER the argmax across the entire corpus. The dead-store fraction: vocabulary tokens that are never argmax on 10K tokens are candidates for elimination. At 151,936 vocabulary size, if 50% of tokens are never argmax (plausible for byte-piece tokens that only appear as sub-words): 76K dead-store rows × (2048 × 2B) = 312 MB of lm_head storage is droppable with zero impact on greedy argmax quality on the calibration distribution. The residency savings on 70B are proportional: Qwen3-72B's lm_head is 151936 × 8192 × 2B ≈ 2.48 GB → 50% dead-store elimination saves ~1.24 GB.

**CF grounding.** CF12: tied lm_head full-rank in Qwen3-1.7B. Specifically: per-row norms are full (bottom-decile 1.36), meaning no dead rows from a weight-magnitude perspective. VOCSKIP targets dead stores from a **functional** perspective (rows that are never argmax-winners on a realistic distribution), not a weight-magnitude perspective — distinct from CF12's finding. CF12 does NOT kill VOCSKIP because VOCSKIP does not attempt SVD truncation; it eliminates rows that are functionally inactive on the inference distribution. CF9: no imported theorem. CF10: the dead-store set is identified by empirical corpus sweep, not by a calibration fit — no parameter-fitting, no conditioning risk.

**What must be true.** A measurable fraction of vocabulary tokens are never argmax winners on realistic inference distributions. This is almost certainly true (many vocabulary tokens are sub-word pieces that only appear within larger tokens; they will never be the top logit in typical sentence completion). The quantitative question is what fraction.

**Experiment cascade.** E0 (30 min): run Qwen3-1.7B greedy inference on 10K diverse tokens (WikiText-2 + code + math). Record argmax-winning vocab index per step. Measure fraction of 151,936 vocab tokens that are never argmax. GO threshold: ≥ 20% of vocab never-argmax. E1 (20 min): mask the never-argmax rows (zero them out); measure ΔNLL on 512-token held-out. GO: ΔNLL < 0.01 nats (consistent with those rows being genuinely dead). E2: if the dead-store rate is high enough (≥ 30%), measure whether INT2 quantization of the dead-store rows further reduces storage while keeping ΔNLL < 0.01.

**What I am NOT proposing.** I am not proposing SVD truncation of lm_head or any approximation that changes the model's output on the active vocabulary. Dead-store rows are rows that are never the argmax winner; their removal is exact for the covered distribution.

**Track B | Sits outside F-orientation anti-frame (no unifying algebraic object) and C-orientation anti-frame (not a coupling equation). Clearly Track B / compression on a different object (lm_head rows) than any prior run's Track B proposals.**

---

### S5 — DHSUB-RAOK: Residual Update Subspace as RAOK Tier-3 Precision Boundary
**Track B | Motivated by XP-2 cross-pollination (DHSUB stripped primitive + RAOK composition) | CF2 + CF3 anchor**
**Elegance class: subspace-alignment**

**Ambition.** DHSUB was rejected (A1=1, small activation-bandwidth saving). Its stripped primitive — per-layer residual update subspace ΔH_L effective rank — is more valuable as a RAOK tier assignment oracle than as a standalone activation-precision switch. The RAOK scheme (deferred Track B R6, strongest surviving compression path) assigns T1/T2/T3 tiers based on K=0.1% vs K=1% outlier channels. DHSUB provides a complementary assignment: channels in the top-r_eff right singular directions of ΔH_L are modified by this layer's computation and require higher precision in W_up/W_down column addressing. Channels outside the top-r_eff directions are pass-through (the layer does not modify them). The coupling: CF3's top-0.1% outlier channels are static across tokens; if they also lie OUTSIDE the per-layer ΔH_L subspace, they are pass-through AND static — unconditionally INT4-safe with zero RAOK overhead. If they lie INSIDE ΔH_L, they are active in this layer's update and require RAOK's full treatment. The coupling equation:

    σ_pass = max_i {⟨e_{c_CF3}, v_i(ΔH_L)⟩ : i = 1,...,r_eff}

If σ_pass ≤ 0.10: CF3 static channels lie outside ΔH_L — RAOK's T1 detection is redundant with ΔH_L measurement, simplifying the tier boundary.
If σ_pass ≥ 0.70: CF3 static channels are inside ΔH_L — RAOK must maintain full T1 tracking.

**Mechanism.** Forward pass on 500 calibration tokens. At each layer: cache h_L and h_{L+1}. Compute ΔH_L = H_{L+1} − H_L (shape d × 500). Compute SVD of ΔH_L; record r_eff at var@threshold. Also compute σ_pass for the CF3 static outlier channels (identified from v2-CF1). This is a 35-min combined measurement using existing PDAP and AQFKV infrastructure.

**CF grounding.** CF2: small Δh (motivates r_eff < d). CF3/v2-CF1: static outlier channel identity known. The coupling is a composition of CF2 and CF3 producing a structural prediction about the tier boundary geometry.

**What must be true.** r_eff(ΔH_L) < d/4 in at least 15/28 layers (go threshold for DHSUB's mechanism). AND the CF3 static channels have a clean verdict on σ_pass (either clearly inside or clearly outside the ΔH_L subspace).

**Experiment cascade.** E0 (35 min): combined ΔH_L SVD + σ_pass computation across all 28 layers. GO threshold: var@256 ≥ 0.85 in ≥ 15/28 layers AND σ_pass either ≥ 0.70 or ≤ 0.10 (clean classification). NO-GO: ΔH_L is nearly rank-d despite CF2's small Δh magnitude — the update is small AND isotropic. Structural finding in either direction.

**What I am NOT proposing.** I am not proposing to change RAOK's tier structure or quantization scheme. This is a measurement that either tightens or corroborates the CF3-based RAOK tier boundary. The compression payoff is downstream of RAOK, not standalone.

**Track B | Composition orientation shape (two CFs, one coupling equation, one small-test). Sits outside C-orientation anti-frame (the coupling equation IS the mechanism, not just naming two findings). Outside F-orientation anti-frame (it IS an algebraic object: the subspace ΔH_L and its alignment with CF3's outlier direction is the object). Outside R-orientation anti-frame (this is not an ambitious cascade — it is a 35-min measurement that feeds the RAOK cascade).**

---

### S6 — TROPATT: Tropical Attention as a Zero-Parameter Argmax-Routing Primitive [FREE SWING]
**Track A | Motivated by no-float constraint reframing | No CF anchor required**

**Ambition.** The no-float constraint forces attention into tropical arithmetic: replace softmax(QK^T/√d) with hard-argmax routing. The resulting mechanism is a **parameter-free content-based routing** for the KV cache: each query attends to exactly one key (the argmax). This is the zero-temperature limit of attention, where attention mass is concentrated on a single position. The question: for what fraction of tokens and heads does Qwen3's softmax attention concentrate its mass on a single key position (effective temperature near zero)? If softmax mass distribution is peaked (entropy << log(n_keys)) on many (token, head) pairs, tropical attention is a low-error approximation — not for all tokens, but for a measurable fraction. The residency savings: on NVMe-resident inference at 128K context, if 60% of (token, head) pairs can use tropical attention (fetch 1 K/V entry instead of ~n_keys entries), KV bandwidth drops by 60% × (n_keys−1)/n_keys ≈ 60%.

**Mechanism.** Measure the per-(token, layer, head) entropy of the softmax attention weight distribution on a 200-token calibration corpus. Identify pairs with entropy < 1.0 nat (effectively one dominant key). For those pairs, implement tropical attention: argmax(QK^T/√d) selects a single key; V[argmax] is the attended value. The computation is: (a) compute full QK^T (standard); (b) select argmax (O(n) pass); (c) return V[argmax] × attn_score (constant, no sum). This reduces V reads from n_keys vectors to 1, saving substantial bandwidth at long context.

**CF9 check.** Tropical attention makes no imported theorem assumptions. The argmax is a standard operation; the quality degradation from discretizing softmax is the empirical question (measured by ΔNLL). No Bochner, no sketching, no sparsity assumption. CF9: PASS.

**CF10 check.** No calibration fitting. CF10: PASS.

**What must be true.** A measurable fraction of (token, layer, head) pairs have softmax entropy < 1.0 nat — i.e., the attention is already effectively hard-routing on those pairs.

**Experiment cascade.** E0 (30 min): forward pass on 200 tokens. Record per-(token, layer, head) attention entropy. Measure fraction with entropy < 1.0 nat. GO threshold: ≥ 25% of pairs with entropy < 1.0 nat. E1 (1 hr): implement tropical attention for low-entropy pairs; measure ΔNLL on 512-token WikiText-2 for the hybrid (tropical on low-entropy pairs, full softmax on high-entropy). GO: ΔNLL < 0.50 nats for the hybrid.

**What I am NOT proposing.** I am not proposing to train a model with tropical attention (requires retraining). I am not proposing sparse attention patterns or learned routing. TROPATT is an inference-time approximation for already-low-entropy attention distributions. It does not modify any weights.

**[FREE SWING] — no CF anchor. Stack primitive: attention score matrix is computed regardless; the argmax is free. The mechanism is the observation that low-entropy softmax distributions are already tropical.**

**Track A | Sits outside every orientation's anti-frame because the tropical constraint is the Constraint-Alien mechanism applied to attention (not MLP, which was A4-TMLP). The CFI-alien angle forces exploration of whether trained Qwen3 attention is "already tropical" on a fraction of heads.**

---

### S7 — WVOP-A3: W_VO Product Fold + A3-Paper-Aware Novelty Positioning
**Track A | Motivated by R6-WVOP advancing + v2-S3-R007-002 kill note | CF11 anchor**
**Elegance class: algebraic-identity**

**Note on v2-S3-R007-002.** The run_007 Stage 3 kill note (v2-S3-R007-002) states that A3 (arXiv:2505.12942) "explicitly computes the fused W_vo = W_v·W_o product and applies activation-aware SVD compression at 70B scale." This is significant: the W_VO fold itself is in A3. R6-WVOP advanced from Stage 3 of run 006 as REFINE, but Stage 3's researcher may not have had the run_007 kill note in context. The Stage 4 skeptic must flag this: **R6-WVOP's core mechanism (W_VO = W_V @ W_O fold + SVD) is pre-empted by A3 at the 70B application level.** What survives is the Qwen3-specific GQA shape analysis, the spectral measurement at Qwen3-1.7B, and — importantly — whether the fold achieves better ΔNLL than storing W_V and W_O separately (because rank(W_VO) ≤ min(rank(W_V), rank(W_O)), the product may be more compressible than either factor). A3's activation-aware SVD uses a Hessian-weighted objective; R6-WVOP uses pure Frobenius SVD on the product. The novelty claim must be repositioned: not "fold and compress W_VO" (A3 owns that) but "does the GQA-shaped W_VO product (1024×2048 in Qwen3-1.7B) achieve lower ΔNLL than the Frobenius-SVD on either factor individually, and does the product rank concentration (rank(W_VO) potentially less than rank(W_V) AND rank(W_O)) provide additional leverage over A3's activation-aware approach?"

The Stage 4 idea S7 is therefore: measure rank(W_VO) vs rank(W_V) and rank(W_O) independently, and compare Frobenius-SVD on the product vs Hessian-SVD (A3-style) on the individual matrices. If Frobenius-product-SVD achieves comparable ΔNLL to A3 at lower compute cost (no Hessian required), this is a practical simplification of A3 with a structural justification (the product rank is tighter). This is not a new mechanism but a repositioning of R6-WVOP against the A3 prior art.

**This idea is primarily a Stage 3/5 amendment note for R6-WVOP, not a new Stage 4 idea.** The Stage 4 output is: flag R6-WVOP for Stage 5 red-team on A3 prior art; add the product-rank-vs-individual-rank comparison as a required sub-measurement.

**Track A | Orientation R WVOP repositioning.**

---

## 7. Output Handoff

### Stage 4 ideas

| ID | Description | Motivating section |
|---|---|---|
| S1 | FNDP-WMCE: anonymous weight loading + OS-tier DRAM compression | XP-1 cross-pollination |
| S2 | QKPART: partition-join attention via CF11 128-dim query subspace bucketing | Sec 4 database engineer |
| S3 | JNULL: Jacobian near-nullspace depth-stratified compute budget | XP-3 + Sec 4 control theory |
| S4 | VOCSKIP: dead-store vocabulary row elimination from untied lm_head | Sec 4 compiler dead-store |
| S5 | DHSUB-RAOK: residual update subspace as RAOK tier-3 precision boundary | XP-2 cross-pollination |
| S6 | TROPATT: tropical attention argmax-routing for low-entropy softmax heads [FREE SWING] | Sec 5 no-float constraint |
| S7 | WVOP-A3 positioning note: flag R6-WVOP against A3 prior art (arXiv:2505.12942) | Frame exhaustion SE-1 |

### Orientation recommendations

| Orientation | Recommendation | Rationale |
|---|---|---|
| R (Reach) | KEEP | Strong convergence C2 representative; anti-frame addition: "attention matrix compression without the W_V/W_O gate measurement" |
| C (Composition) | KEEP | Highest-scored idea in run (C-JOINT-DELNLL, 13/15); anti-frame addition: "per-layer single-matrix sensitivity probes without coupling equation" |
| F (First-Principles) | TIGHTEN | No unifying-object proposals in run 006; anti-frame addition: "measurement-only first-principles proposals without a unifying algebraic object" |
| U (Unconventional) | TIGHTEN | Running low on unexplored substrate axes; anti-frame addition: "substrate proposals whose impact is solely latency-variance reduction without a throughput floor" |
| A (Constraint-Alien) | KEEP | Frame-novelty ideas (A1-PEER, A6-LLMR) are the highest-novelty outputs; anti-frame addition: "constraints that produce mechanisms equivalent to existing soft approximations" |

### Anti-frame additions (specific strings for ORIENTATIONS.md updates)

- **R**: Add to anti-frame list — "Attention matrix compression proposals that do not acknowledge the W_V/W_O gate (Convergence C2) as a prerequisite."
- **C**: Add to anti-frame list — "Per-layer single-matrix sensitivity probes presented as compositions. A coupling requires two findings and one equation; a single-matrix sensitivity profile is a First-Principles measurement."
- **F**: Add to anti-frame list — "Measurement-only proposals lacking a unifying algebraic object. The orientation's target is an object that CF_i, CF_j, CF_k are projections of — not a measurement of one CF's depth profile. If the proposal does not name a mathematical object, it belongs in Composition or Reach."
- **U**: Add to anti-frame list — "Substrate proposals whose sole measurable outcome is reduced latency variance (p95/median ratio) without a throughput floor improvement. Use the orientation's payoff metric as bytes-per-second or tok/s, not jitter."
- **A**: Add to anti-frame list — "Constraints that, when traced through the mechanism, produce outcomes equivalent to INT4 quantization, uniform quantization, or coordinate changes that the standard ML playbook already performs under a different name. The constraint must remain binding at the level of the inference computation graph, not just the framing."

### Priority for Stage 5

**Top picks for Stage 5 consideration:**

1. **S1 (FNDP-WMCE composition)** — Zero quality risk, 0 nats. E0 (45 min standalone C harness) is the cheapest next experiment in the run. It resolves both the FNDP claim and initiates the WMCE path. The composed mechanism has the highest expected value among Stage 4 ideas because it is entirely at the substrate level (no weight modification) and both components are falsifiable cheaply and independently.

2. **S2 (QKPART partition-join attention)** — Moderate risk but CF11-grounded. E0 (30 min attention entropy measurement) is the cheapest falsifier. If the top-1 bucket captures ≥ 50% of softmax mass on average, the mechanism is valid and ΔNLL will be near CF11's +0.98 nats — consistent with a useful deployment configuration. The database-join framing is genuinely alien to the published NLP attention literature.
