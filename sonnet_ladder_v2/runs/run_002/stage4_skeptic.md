# Stage 4 — Skeptic Explorer — Run 002
Run: 002 | Date: 2026-05-09 | Agent: Sonnet claude-sonnet-4-6 (Stage 4 Skeptic)

---

## 1. Frame-Exhaustion Map

### Saturation-Exhausted Frames

**SE-1: Cross-layer W_Q basis sharing (stacked-SVD measurement)**
Every orientation that touched attention weights independently converged on the same stacked-57344×2048-SVD experiment. CQBS (F), XLQB (R), CQST (C), GHPI (A) are four measurement proposals that resolve at the same computation. The frame is not dead — the experiment is genuinely unmeasured — but the measurement protocol is now fully worked out. Any fifth variant of "stack W_Q, run PCA, check cumvar@128" is strictly redundant. Stage 4 should not produce another variant of this experiment. Stripped primitive: the stacked-SVD protocol itself (legitimate infrastructure that pipelines into any W_? measurement; it is a tool, not an idea).

**SE-2: Softmax shift-invariance as a quantization lever**
SSIF (F4) is partially pre-empted by arXiv:2605.02907 (April 2026), which formalizes the energy field invariant. The frame of "softmax is shift-invariant, so we can remove a constant offset" is now published. The specific application to Q/K dynamic-range reduction for quantization is narrowly novel, but the frame itself is occupied. Stripped primitive: the per-head logit energy field's stable low-rank structure (arXiv:2605.02907 finds top-20 SVD components capture >90% variance) — this structure itself is an unexploited compression object.

**SE-3: RMSNorm scale absorption into downstream weights (for spectrum concentration)**
RMGF (F3) treads close to SmoothQuant/QuaRot/SpinQuant territory. Those papers absorb scales and rotations into weight matrices for activation-range reasons; RMGF asks whether the same fold concentrates the weight spectrum. The measurement is genuinely unmeasured, but the mechanism family (fold a diagonal scale into W) is deeply occupied. The spectrum-concentration hypothesis is the live piece; plain scale-absorption is exhausted. Stripped primitive: the hypothesis that g_L encodes spectral information that concentrates the W_Q singular value distribution — testable in 25 minutes, but the stripped primitive is the spectral question, not the fold itself.

**SE-4: Gauge-exploitation as an attention compression frame**
RGPS (A6), RSGO (F6), GHPI (A1), RMGF (F3), SSIF (F4) all use gauge arguments (residual-stream rotation, head-permutation, norm-gain absorption, softmax shift). The gauge-exploitation orientation-frame is now heavily colonized within run 002. Five ideas in this family represent saturation of this creative orientation, not saturation of the physics. There remain untouched gauge symmetries (sign-flip gauge, channel-scaling gauge at W_down input, per-layer scale-normalization gauge at lm_head input) — but any new idea in this family must pick a DISTINCT gauge.

**CF9-Exhausted Frames**

**CF9-1: FSMD residual-state FSM coverage claim**
FSMD was killed by Stage 2 (CF9) because the Zipf coverage precondition for (VQ-state, token) joint transitions is unverified. The stripped primitive — K-vector near-duplicate rate measurement — survives inside ASAKS. Any future FSM-decoder idea that imports a coverage theorem for discrete latent states must pre-verify the distribution's tail behavior on Qwen3. The family is not dead; it requires a coverage-measurement rung before any table-construction rung.

**CF9-2: Spectral methods that require smoothness/stationarity**
RFSK (Track B R6, prior round) was killed because Bochner's theorem requires shift-invariant kernels and weight matrices are not kernels. This family remains CF9-dead unless the target object is genuinely kernel-structured (attention score matrices, which ARE positive-definite per-head by construction, have a weak claim to this structure). Not revisited in run 002 but the precondition failure pattern is generalized.

**CF10-Exhausted Frames**

**CF10-1: Full-affine cross-scale hidden-state surgery at high rank**
WDLA's original failure (2.1M params, 1K calibration tokens, ridge=1e-3) established this frame as CF10-dead unless forced to low rank. WDLA-RESCUE (R5-A) correctly rescopes to rank-64. The class of "fit an affine A: ℝ^d → ℝ^{d'} without rank-by-construction constraint" is CF10-exhausted for 1.7B-class targets. Stripped primitive: rank-by-construction linear surgery (rank ≤ 64 by design) at inference time — this is what WDLA-RESCUE exploits and it is the live strip.

---

## 2. Orientation-Saturation Diagnostics

**Orientation R (Reach):** Advanced 5 ideas (QKJB, RAOK-SCALE, XLQB, WDLA-RESCUE, ATTN-SPECTRUM). All 5 at REFINE. Strong convergence participation (C1: XLQB, C2: ATTN-SPECTRUM + QKJB, C3: RAOK-SCALE). No kills. Structurally healthy. Anti-frame addition: tighten against "cascade whose intermediate rungs deduplicate with other orientations' experiments" — QKJB first rung and XLQB first rung both collapse to the same CQBS/ATTN-SPECTRUM experiments; Reach should own only the cascade arithmetic and the 70B deployment target, not the measurement itself. Recommendation: **KEEP**. Anti-frame addition: "Measurement experiments that are first-rung settlers without a downstream multi-rung cascade; assign those to First-Principles."

**Orientation C (Composition):** Advanced 5 ideas (OHAA, CQST, LGOP, QCDS, RKGP). All 5 at REFINE. Convergence participation in C1 (CQST), C2 adjacency (RKGP), C3 (OHAA, LGOP). Strong performance — the coupling-claims format (two findings, one equation) was executed cleanly in CQST and LGOP. Anti-frame addition: the CF2 × CF11 coupling (residual flatness forcing cross-layer Q alignment) has now been explored thoroughly via CQST and QCDS. Add "CF2 × CF11 compositions that reduce to yet another cross-layer W_Q subspace measurement." Recommendation: **KEEP**, tighten against CF2-reuse.

**Orientation F (First-Principles):** Advanced 5 ideas (JVOC, CQBS, RMGF, SSIF, WDCA, RSGO). All 6 at REFINE. F carried the C1 and C2 convergence cluster's strongest representatives (CQBS for C1, JVOC for C2). RMGF and SSIF advance on frame-novelty. Anti-frame addition: "Spectrum-concentration measurements via scale/rotation absorption that have a QuaRot/SmoothQuant cousin — verify spectral differentiation over published baselines before advancing." Also: "Algebraic-fold ideas targeting the same gauge (residual-stream rotation) as RGPS/RSGO/RMGF — orient toward untouched gauges (sign-flip, per-channel W_down output scale, lm_head row-norm gauge)." Recommendation: **KEEP**.

**Orientation U (Unconventional Substrate):** Advanced 5 ideas (PrefetchVM, VirtualLock, NTFS-sparse, MEM_LARGE_PAGES) plus rejected 3 (zstd-dict, IOCP-dispatch, ReadDirectoryChangesW). The 4 advancing substrate ideas are real primitives but modest — they are residency-management primitives, not compression mechanisms. U4/NTFS-sparse is likely a quick NO-GO (CF8 predicts no zero blocks at Q2). The substrate space that U has NOT explored: filesystem content-dedup (Windows Data Deduplication service, separate from NTFS sparse), CPU-specific CPUID feature queries for weight-format selection, NVMe namespace zoning (multi-namespace NVMe with predictable sequential-read patterns). Anti-frame addition: "Substrate primitives that affect latency variance without improving residency or tok/s floor — useful tooling but not pipeline candidates." Recommendation: **KEEP with tightened creative directive toward primitives with a structural payoff (residency, not just latency/jitter).**

**Orientation A (Constraint-Alien):** Advanced 5 ideas (GHPI, ASAKS, TAAR, RSLK, CAWD, RGPS). TAAR scored 10 but lost Path 3 slot to SSIF/RMGF on A4. GHPI and ASAKS are strong. The orientation used the gauge-symmetry and append-only constraints well. Under-used constraints: "10 MB seed reconstruction" (no idea in this run), "sequential-only storage" (no idea in this run), "no RAM — registers only" (RSLK approached this but as engineering rather than constraint-bound). Anti-frame addition: "Multiple constraints in one proposal — each proposal binds ONE constraint hard." Recommendation: **KEEP**, rotate creative directive toward constraints not yet explored (10 MB seed, peer-weight swap, O(1)-token constraint).

---

## 3. Cross-Pollination Opportunities

**CP-1: FSMD stripped primitive → single-layer discrete routing in Orientation F**

FSMD was killed because the full table (K × V × L = 552 GB) is infeasible and the Zipf coverage claim for (VQ-state, token) joint transitions is unverified. But ASAKS (A2) independently established that K-vector near-duplicate measurement is a live primitive. The cross-pollination: take FSMD's stripped primitive (discrete latent state at a SINGLE layer, not the full cross-layer FSM) and transpose it into First-Principles framing. In F's frame, the question becomes: "does W_Q^ℓ project the residual stream into a discrete-structured space with a small effective vocabulary?" — a much weaker precondition than FSMD's global Zipf claim. A single-layer VQ of W_Q^ℓ h_ℓ and measurement of its effective codebook size is a 30-minute experiment that doesn't require the Zipf assumption. This motivates **S1 below**.

**CP-2: HYBRID-TIER-70B stripped primitive → per-object precision schedule grounded in new spectra**

HYBRID-TIER-70B was rejected because the per-matrix sensitivity ordering is already partially known and the frame is adjacent to killed MDL-Per-Layer bpw. But the rejection was FOR A PER-LAYER schedule in a coverage-already-known sensitivity ordering. The run 002 measurements add new information: W_Q is the most compressible (r_99/d ≈ 0.63), W_K is next (0.79), W_V/W_O are unknown (JVOC will measure), W_gate/W_up are full-rank (r_99/d ≈ 1.0), tied lm_head is catastrophically sensitive. This is a NEW per-object sensitivity ordering, NOT published, and NOT the same per-layer bpw that MDL-Per-Layer bpw covered. A Stage 4 idea can build on the CF11 × JVOC × CF12 object-type sensitivity ordering — transposing HYBRID-TIER-70B's stripped primitive (knapsack assignment by object type) from the dead "per-layer" frame into the live "per-matrix-class" frame with new empirical constraints. This motivates **S2 below**.

**CP-3: IOCP-dispatch stripped primitive → asynchronous layer-prefetch with the W_Q basis amortized**

IOCP-dispatch was killed because its smallest test exceeds 8 hours (200+ lines of C code). The stripped primitive is asynchronous per-layer dispatch with overlap. Transpose into Orientation U's frame: if the CQBS shared W_Q basis is confirmed, the layer-fetch IO pattern changes — instead of fetching one full W_Q per layer, you fetch the thin C_L coefficients (128×2048, 512 KB) and the shared U basis is already resident. The IO pattern for a CQBS-compressed model is different enough that a PrefetchVirtualMemory plan for it (U1) is a new experiment, not a IOCP rebuild. This telescopes into **S3 below** which is a U-orientation idea motivated by the IOCP primitive.

---

## 4. Cross-Domain Seed Exploration

**Professional Prior 1: Compiler writer — profile-guided dead-code elimination**

What algebraic identity does a compiler writer see in the Qwen3 attention weight class?

A compiler writer profiles which code paths are executed (how often) and eliminates dead paths without changing output for any executed input. Applied to transformer inference: after profiling activation patterns across calibration tokens, which columns of W_gate are never accessed by any token in the calibration distribution? This is not about magnitude (CF3 shows outlier channels rotate per token at K=1%) but about STRUCTURAL dead zones — columns that are consistently below a statistical significance threshold not because activations are low-magnitude there, but because those columns are never in the TOP-K ACTIVATION SET across any of N=10K calibration tokens. This is a different claim from CF3's dynamicity finding: CF3 says the outlier SET rotates; the compiler view asks whether the TOTAL SUPPORT of the activated set covers all columns or leaves a structural dead zone. If W_gate has 10% of columns that are NEVER in the top-1% for any of 10K tokens, those columns are dead code — they can be pruned without any calibration fit (just measurement). This is an extremely cheap experiment: histogram the per-column frequency of appearing in top-K over 10K calibration tokens. If any column has frequency = 0, it is dead. If frequency < 1/10K = 0.01%, it may as well be dead. The profile-guided elimination is structural and requires no gradient. This motivates **S4 below**.

**Professional Prior 2: Database engineer — zone map / columnar skip scanning**

What algebraic identity does a database engineer see in the attention computation?

A database engineer building a columnar store uses zone maps: for each block of rows (e.g., 1000 rows of a column), store the (min, max) of that column. When a query predicate says "col > 5," entire blocks are skipped if max < 5. Applied to attention: for each K-cache block (e.g., 64 consecutive token positions), store (min, max) per head of the attention LOGIT score Q_h · K_j^T for a given query Q_h. If the entire block has max logit < (global top-K threshold - margin), skip it entirely. This is different from sparse attention (which requires training-time masking) and different from KV eviction (which discards tokens permanently). A zone map is a per-block pre-filter that skips block computation at inference time without modifying the KV cache and without approximating the softmax — it is EXACT for the tokens NOT in the skipped blocks. The zone map computation cost (max per 64-token block) is amortized over many queries. The precondition: are there 64-token K-cache blocks where ALL tokens score below the threshold for typical queries? This is empirically checkable in 20 minutes on a 4K context forward pass. This motivates **S5 below**.

**Professional Prior 3: Control theorist — Kalman filter / state observer**

What conservation law does a control theorist see in the residual stream?

A Kalman filter maintains a running estimate of a state variable as new observations arrive; the key property is that the state estimate update is linear in the innovation (new observation minus predicted observation). The transformer residual stream has a structure that resembles a Kalman update: h_{L+1} = h_L + f_L(h_L), where f_L is the attention + MLP contribution. CF2 says cos(h_L, h_{L+1}) ≈ 0.99 — the update is small relative to the state. A control theorist sees this as a near-constant-gain linear system operating near a fixed point. The observable state is h_L; the "measurement" is the attention/MLP correction f_L(h_L). The conservation law: in a linear time-invariant system with small innovations, the state has a conserved quantity — the direction of maximum information. If the residual stream's dominant direction (from PCA of h_L across calibration tokens, at any layer L) is conserved across layers (i.e., the same PCA-1 direction appears in h_1, h_5, h_15, h_28), then PCA-1 is a conserved quantity under the forward pass. This is a strictly DIFFERENT measurement from CF2's cosine similarity (which measures the full vector, not just the principal direction). If PCA-1 is conserved, the leading direction of the residual stream carries no layer-by-layer information and can be projected out before every attention/MLP computation — reducing the effective dimensionality by 1 for free. Checking whether the top-K PCA directions of h_L are conserved across layers requires a 30-minute calibration forward pass and cross-layer cosine similarity of the PCA basis. This motivates **S6 below**.

---

## 5. Constraint-Driven Reframing

**Constraint: 10 MB seed reconstruction**

The model must be reconstructible from a 10 MB compressed representation plus a deterministic decompressor. Applied to the Qwen3-1.7B-Base bf16 checkpoint (~3.4 GB): 10 MB is 0.3% of the model. This is far below any reasonable per-matrix compression, which means the mechanism cannot be compression — it must be GENERATION. The 10 MB seed must be a program that generates the weights. For random initialization, a neural network can be seeded with a small RNG state; the question is whether a TRAINED model's weights can be expressed as a small initial state plus a small program that generates approximately the correct weights. The relevant structural input: CF8 says MLP weights are full-rank (r_99/d ≈ 1.0) — they appear random. CF11 says W_Q has r_99/d ≈ 0.63 — more structured. A 10 MB seed for a 3.4 GB model requires finding the "generator program" that, starting from a small RNG state, produces weights whose low-rank structure matches the measured singular value distributions. This is Kolmogorov complexity: what is the minimum description length of the Qwen3-1.7B-Base weight distribution? The constraint does not yield a practical inference-time mechanism (3.4 GB → 10 MB is far beyond current no-retraining compression), but it DOES yield a measurement: what fraction of the weight energy is explained by a small number of deterministic random seeds (e.g., the first R=128 singular vectors of each weight matrix are approximately the top-R eigenvectors of a random Gaussian matrix of the same shape)? If yes, the "seed" is the Gaussian seed, and the deterministic decompressor is "run PRNG, take top-R singular vectors." This is the Marchenko-Pastur comparison: are the singular value distributions of W_gate / W_up consistent with a random Gaussian matrix (Marchenko-Pastur bulk) plus a few structured outliers? CF8's r_99/d ≈ 1.0 is consistent with but does not confirm random-matrix structure. Candidate idea from this constraint: **S7 [FREE SWING]**.

**Constraint: sequential-only storage (no random reads)**

Every weight chunk is accessible only by a forward sequential scan; no random-access reads allowed. This forces stream-oriented decode — the model must be laid out on NVMe in the order it is accessed during inference. For autoregressive generation, the access pattern IS sequential: layer 0 → layer 1 → ... → layer 27, repeated for every token. The standard GGUF file layout is roughly sequential by tensor, but within a layer the 4+ matrices (W_Q, W_K, W_V, W_O, W_gate, W_up, W_down) may not be interleaved in the optimal order for the compute graph. The constraint forces: (a) profile the actual NVMe access order during a forward pass; (b) re-layout the GGUF tensor table so that the access order exactly matches the sequential scan. This is purely layout optimization — but Stage 3 noted that U1/PrefetchVM depends on whether ik_llama.cpp already overlaps I/O. If it does NOT (the pre-check fails), re-layout for sequential access is the highest-leverage substrate move available. This motivates **S8 below** as a composition of the sequential-only constraint with U1's PrefetchVM primitive.

---

## 6. Generated Ideas

---

### S1 — VQWQ: VQ-Projected W_Q Effective Vocabulary

**Elegance class: algebraic-identity (discrete reformulation of continuous subspace)**
Track A | Outside F anti-frame (not plain SVD measurement; quantization of the projected representation is the novel step)

**Ambition**
Determine whether the W_Q attention projection h_ℓ → Q_h = W_Q^h h_ℓ effectively maps residual-stream tokens into a small discrete vocabulary in a single layer. If yes, at inference time the query computation for that layer can be replaced by a codebook lookup: for any token whose residual h_ℓ hashes to a known codebook entry, precompute Q_h and skip the GEMV. Combined with CF11's global-K=128 finding (all 16 heads collapse to 128 dims), the effective query vocabulary may be small enough that a 20,000-entry codebook covers >50% of tokens at inference time.

**Mechanism**
1. For each layer ℓ, project residual-stream activations onto the top-128 right singular vectors of W_Q^ℓ (the shared-basis computation from CQBS). This gives a 128-dim query summary vector per token.
2. VQ-cluster the projected 128-dim vectors using k-means with k ∈ {128, 512, 2048, 8192} over 10K calibration tokens.
3. Measure codebook coverage: what fraction of tokens fall within cosine-distance ≤ 0.05 of their nearest centroid? If k=2048 centroids cover ≥70% of calibration tokens at distance ≤ 0.05, the query space has a quasi-discrete effective vocabulary.
4. At inference: hash the projected 128-dim query summary; look up precomputed Q_h in the codebook; if cache hit, skip the GEMV for that head. If cache miss (30% of tokens), compute normally.

**Why our findings make this work**
CF11's head-redundancy finding (16 heads → 128-dim subspace) restricts the query space to 128 dimensions. In 128 dimensions, a k=2048 codebook has a much better coverage-per-centroid ratio than in 2048 dimensions. The projection is already motivated by CQBS; the VQ is free given the projection. The FSMD stripped primitive (discrete residual state → lookup table) is the ancestor; this is its CF9-safe version because no Zipf theorem is imported — coverage is measured directly.

**What would have to be true**
- The top-128 projected query space must be clusterable (low intrinsic dimension or multimodal distribution) so that k=2048 centroids give ≥70% coverage at cosine-distance ≤ 0.05.
- The coverage rate must be stable out-of-distribution (WikiText train vs test).

**Experiment cascade**
- E1 (2h): Run 10K calibration tokens, project each layer's residual stream onto W_Q^ℓ's top-128 singular vectors (free after CQBS). k-means cluster at k=2048. Measure coverage rate per layer. Go/no-go: ≥70% coverage in ≥10 layers.
- E2 (1h): If E1 GO, measure cache-hit throughput gain on 512-token WikiText-2 prefix inference. Expected gain: up to 35% GEMV reduction on covered tokens.
- E3 (4h): Evaluate NLL degradation at k=2048 coverage; confirm ΔNLL ≤ +0.15 nats.

**What you are NOT proposing**
Not a replacement for W_Q computation on all tokens. Not a clustering of the full residual-stream space (only the 128-dim projected space). Not retraining. Not any change to the weight matrices.

White space: outside F's anti-frame (not plain SVD/rank-truncation — the VQ clustering is a new mechanism). Outside A's anti-frame (CF9-safe — no Zipf theorem imported). Cross-orientation gap from FSMD kill + CF11 finding.

---

### S2 — PASS: Per-Attention-Subclass Sensitivity Schedule

**Elegance class: no-SGD-reformulation (bpw schedule derived directly from empirical spectrum class)**
Track B | Outside R anti-frame (this is per-object-class precision scheduling, NOT per-layer bpw; HYBRID-TIER-70B's dead variant was per-layer)

**Ambition**
Construct a 70B-deployment bpw schedule derived entirely from the CF11 + JVOC + CF12 per-matrix-class sensitivity ordering. This is not the dead per-layer MDL schedule (uniform handling of all matrices within a type); it is a per-matrix-TYPE schedule that assigns bpw based on r_99/d class membership. The ordering CF11 established (W_Q: 0.63, W_K: 0.79, W_V/W_O: TBD by JVOC, MLP: 1.0, tied lm_head: catastrophic) can be used DIRECTLY as the sensitivity prior for a 4-class mixed-precision quantization. No additional calibration fitting; the schedule is read off from the spectrum measurements.

**Mechanism**
1. Class 1 (W_Q): r_99/d ≈ 0.63 → INT4 global (confirmed safe at K=512 → 2× rank reduction, ΔNLL=+0.20). Under cross-layer sharing (CQBS), potentially INT2 for coefficient matrices.
2. Class 2 (W_K): r_99/d ≈ 0.79 → INT4 global (CF11 K=512, ΔNLL=+0.29 — borderline, needs verification).
3. Class 3 (W_V, W_O): r_99/d unknown (JVOC will measure). If ≤ 0.85: INT4. If ≥ 0.90: INT8 or leave at bf16.
4. Class 4 (W_gate, W_up, W_down): r_99/d ≈ 1.0 → INT4 floor (PTQTP/NanoQuant territory; CF8 applies).
5. Class 5 (tied lm_head): catastrophic at any rank reduction (CF12) → store at INT8 minimum (no rank reduction ever).

**Why our findings make this work**
The per-class sensitivity ordering is directly measurable from v2 findings (CF11, CF12, and JVOC when run). This is not an import from the published landscape — it is the v2 empirical substrate. The schedule requires NO gradient, NO calibration fitting; it requires only reading the singular value spectra already measured. The HYBRID-TIER-70B rejection applied to a knapsack over unknown sensitivities; here the sensitivities are empirically known from v2 measurements. The CF9/CF10 floors: no theorem imported, no calibration fit.

**What would have to be true**
- JVOC must run first to fill Class 3 (W_V/W_O spectra).
- The per-class sensitivity ordering must be stable across layers (QCDS tests this for W_Q; needed for all classes).
- The per-class INT4 vs INT8 thresholds must be derived from the same ΔNLL ≤ +0.05 nats per class, not assumed.

**Experiment cascade**
- E1 (30 min): Read the already-measured spectra (CF11, CF12) and compute implied residency arithmetic for Qwen3-1.7B and Qwen3-72B (extrapolated). No experiment needed — pure arithmetic.
- E2 (1h): After JVOC runs, fill the W_V/W_O class entry into the schedule. Compute total 72B residency under PASS schedule: (W_Q at INT4 global) + (W_K at INT4 global) + (W_V/W_O at CLASS3_BPW) + (MLP at INT4) + (lm_head at INT8). Target: ≤ 7.28 GB for 1.7B (confirm DRAM-residency for the small model as a proxy).
- E3 (2h): Implement PASS schedule on Qwen3-1.7B; measure ΔNLL vs uniform INT4 baseline.

**What you are NOT proposing**
Not per-layer bpw scheduling (the dead MDL frame). Not a calibration-fitted sensitivity measurement. Not a new quantization algorithm — just a principled bpw assignment rule derived directly from v2-confirmed spectral class membership.

White space: outside R's anti-frame (not aspirations of floor that other orientations consider ceiling — this is arithmetic that other orientations' measurement results directly motivate). Cross-pollination from HYBRID-TIER-70B's stripped primitive transposed into a spectrum-class-grounded frame.

---

### S3 — CQBS-IO: Sequential-IO Optimizer for CQBS-Compressed Model

**Elegance class: none (pure substrate)**
Track B | Orientation U white space | Outside U anti-frame (not custom kernel; not SIMD; exploits existing Windows I/O scheduler)

**Ambition**
If CQBS confirms that all 28 W_Q matrices share a 128-dim basis U (stored once), the model file layout for W_Q changes: U (2048×128, 512 KB) is fetched ONCE, then 28 thin coefficient matrices C_L (128×2048, 512 KB each, 14 MB total) replace 28 full W_Q matrices (235 MB total). The IO pattern for W_Q changes from 28 sequential 8 MB reads to 1 read of 512 KB + 28 reads of 512 KB. The PrefetchVirtualMemory API (U1) can schedule the thin C_L reads in a single batch, overlapping them with the prior layer's compute. The IO amplification for W_Q drops from 8 MB/layer to 512 KB/layer — a 16× bandwidth saving on W_Q I/O under the CQBS go condition.

**Mechanism**
1. Conditional on CQBS GO: re-layout the GGUF file so that U appears once at the start of the file, and each layer's C_L appears immediately before that layer's other weights.
2. Implement a custom GGUF tensor table (GGUF supports arbitrary tensor layout) that interleaves C_L in the layer-sequential order.
3. Invoke PrefetchVirtualMemory for each upcoming layer's C_L + other weights in a two-slot prefetch ring buffer (the same mechanism as U1, but with CQBS-aware addresses).
4. Measure: effective W_Q IO bandwidth at 4K context (expected: 235 MB → 14 MB + 0.5 MB amortized = ~14.5 MB; 94% reduction in W_Q IO).

**Why our findings make this work**
CQBS GO is the conditional gate. The GGUF file format already supports arbitrary tensor layouts; re-layout is a post-processing step on the checkpoint file. PrefetchVirtualMemory (U1) is already advancing through Stage 3; this idea piggybacks on U1 infrastructure with CQBS-specific addressing. The substrate is real (GGUF file format, Win32 PrefetchVirtualMemory API, NVMe sequential read patterns). The IO saving is algebraically exact from the CQBS storage arithmetic in Stage 3 §5.

**What would have to be true**
- CQBS GO (cumvar(K=128) ≥ 0.85).
- ik_llama.cpp must expose tensor-layout customization OR the re-layout must be done offline in the GGUF file.
- NVMe must be the actual bottleneck (not compute-bound at the relevant scale).

**Experiment cascade**
- E1 (30 min): Given CQBS GO, compute IO saving arithmetic per platform. Verify gguf-convert supports custom tensor ordering (check gguf spec).
- E2 (2h): Implement GGUF re-layout script for CQBS-compressed Qwen3-1.7B. Measure W_Q read time in ik_llama.cpp with and without layout.
- E3 (4h): Full inference benchmark under NVMe-forced conditions (pre-evict DRAM cache). Measure tok/s uplift vs baseline layout.

**What you are NOT proposing**
Not a new quantization scheme. Not a custom NVMe firmware change. Not a custom kernel. Purely a file layout optimization motivated by the compression structure CQBS produces, exploiting Win32 IO scheduler behavior with the existing PrefetchVirtualMemory API.

White space: outside U anti-frame (not custom kernel; real Win32 primitive; IO-layout motivated by compression finding). Motivated by IOCP-dispatch stripped primitive (async IO) transposed into the CQBS-conditional regime where it avoids the implementation cost.

---

### S4 — CDCA: Calibration Dead-Column Audit

**Elegance class: no-SGD-reformulation (profile-guided dead-code elimination without gradient)**
Track B | Outside F anti-frame (not magnitude pruning as load-bearing move — the load-bearing move is structural zero-coverage, not small magnitude) | Outside A anti-frame (no constraint needed; this is a pure measurement)

**Ambition**
Identify columns of W_gate that are structurally dead — never appearing in the top-K activation set across N=10K calibration tokens, for any token. These columns can be zeroed (or pruned) without any calibration fit, any gradient, any retraining assumption. The save is small (at most 10% of W_gate bandwidth if 10% of columns are dead), but the mechanism is qualitatively different from magnitude-based pruning: it is COVERAGE-BASED. A column that has small mean magnitude may still fire on rare inputs; a column that is never in the top-K for any calibration token is dead by the coverage criterion, which is a structural property of the calibration distribution.

**Mechanism**
1. Run 10K calibration tokens through Qwen3-1.7B-Base bf16.
2. For each layer ℓ, for each column j of W_gate (i.e., the j-th row of W_gate^T, which corresponds to the j-th hidden neuron), record whether neuron j appears in the top-K (K = 1% = 61 neurons of 6144) for each of the 10K tokens.
3. Compute column-coverage frequency f_j = (number of tokens where neuron j is in top-K) / 10K.
4. Dead-column threshold: f_j = 0 (never in top-K). Near-dead: f_j < 0.001 (< 10 tokens in 10K).
5. Report: total dead-column fraction per layer; distribution of f_j across all 6144 × 28 = 172K neurons.

**Why our findings make this work**
CF3 showed that at K=1%, the outlier SET rotates per token (Jaccard = 0.31). This means the top-K set changes with each token — but does every column EVER appear in that rotating set, or are some columns permanently excluded? CF3 measured Jaccard between CONSECUTIVE tokens, not global coverage. The global coverage question is new. The mechanism requires no theorem import (CF9: CLEAR). The calibration step is a measurement (not a fit), so CF10 does not apply.

**What would have to be true**
- Some non-trivial fraction of the 6144 × 28 = 172K columns must be never-in-top-K across 10K calibration tokens for the mechanism to have practical value.
- The never-in-top-K classification must generalize (columns dead on WikiText must also be dead on code, math, etc.).

**Experiment cascade**
- E1 (1.5h): Run 10K calibration tokens; record top-K neuron index sets; compute per-column coverage frequency f_j. This is the primary go/no-go: go if ≥1% of columns have f_j = 0 in ≥10 layers.
- E2 (30 min, conditional on E1 GO): Verify out-of-distribution stability: re-run on 1K tokens from a different domain (code). Compare dead-column sets.
- E3 (1h, conditional on E2): Zero the dead columns in W_gate; measure ΔNLL. Expected: ≈ 0 (dead columns contribute nothing to any token's output by construction).

**What you are NOT proposing**
Not magnitude-based pruning. Not calibration-fitted parameter elimination. Not any form of low-rank decomposition. Purely a coverage-based structural audit that exploits the compiler's insight: profile first, then eliminate provably dead code.

White space: outside all orientation anti-frames (CF3's dynamicity finding motivated the question; coverage is different from dynamicity; no imported theorem; no calibration fit). Inspired by compiler-writer professional prior.

---

### S5 — ZKAM: Zone-Map Accelerated KV-Cache Attention

**Elegance class: algebraic-identity (exact computation via block-skip identity)**
Track A | Outside kill list (KV Temporal Differencing was killed because RoPE-rotated deltas are not small; ZKAM uses block max/min, not deltas) | Outside C anti-frame (no stacked compressions; a single arithmetic identity enables block skip)

**Ambition**
At 4K+ context, use per-block (64-token) zone maps on the K-cache to skip entire blocks in attention computation when the block's maximum possible attention score is below the top-K threshold. This is an exact computation (no approximation for skipped blocks — they provably contribute ≤ ε to the softmax output), and it requires no modification to the model, no gradient, no new parameters. The speedup is proportional to the fraction of blocks that can be skipped, which depends on the distribution of Q · K^T scores across the KV cache.

**Mechanism**
1. For each KV-cache block B_b (64 consecutive token positions, per layer, per head), maintain max_b = max_{j ∈ B_b} Q_h · K_j^T (computed lazily as K-vectors are appended).
2. At attention time for query Q_h: first compute Q_h · max_K_b for each block (one dot product per block, where max_K_b is the per-block max-norm K-vector stored as metadata). If Q_h · max_K_b < θ (threshold = global top-K cutoff - margin), skip the entire block.
3. The threshold θ is set so that at most ε = 1e-4 of softmax probability mass is lost; verify empirically that ε-tight bound holds on calibration data.
4. Block skip saves 64 attention dot-products per skipped block.

**Why our findings make this work**
CF11 confirms W_Q global K=128 is GO — meaning Q_h lives in a 128-dim subspace. In 128 dimensions, K-vectors from semantically unrelated past tokens will typically have small dot-products with Q_h. At 4K context, if 60% of 64-token blocks have max attention score below threshold (a plausible density for content with varied topics), 60% of K-cache compute is skipped at ZERO quality loss. The DB zone-map prior is the mechanism; the precondition (blocks with uniformly low scores exist) is empirically testable on Qwen3-1.7B calibration data.

**What would have to be true**
- At 4K context, ≥30% of 64-token KV blocks must have max attention score below the global top-20 (0.5% of 4096 = 20 positions) threshold at ≥50% of query positions.
- The per-block max-K storage overhead (one 128-dim vector per 64-token block = 1/64 of K-cache size) must not exceed the compute savings.

**Experiment cascade**
- E1 (1h): Forward pass on Qwen3-1.7B with 4K WikiText-2 context; capture full attention logit matrices; measure per-64-token-block maximum attention score distribution; compute skip-able fraction at various thresholds.
- E2 (30 min, conditional on E1 GO): Implement zone-map forward pass (Python prototype); verify ΔNLL = 0 at the E1-calibrated threshold (exact computation claim check).
- E3 (3h): C-level integration prototype in ik_llama.cpp GEMV loop; measure throughput at 4K and 32K context.

**What you are NOT proposing**
Not sparse attention (which modifies the training objective). Not KV eviction (which permanently discards tokens). Not approximate attention (which introduces reconstruction error). The zone-map skip is an exact identity: skipped blocks contribute provably less than ε to the softmax sum, with ε set tightly enough to produce ΔNLL = 0 on calibration.

White space: KV Temporal Differencing was killed for RoPE-delta reasons (CF9 on smooth-delta assumption); ZKAM uses max-score blocks which have no smoothness precondition. Database zone-map professional prior transplant into KV attention. Anti-frame for C orientation (naming composition without math): here the math is exact — a block is skipped if and only if all its attention weights are provably below threshold, which is decidable from the zone-map metadata without materializing the K-vectors.

---

### S6 — RCPS: Residual-Stream Conserved-Principal-Subspace Projection

**Elegance class: conserved-quantity (control-theorist view of fixed-point residual stream)**
Track A | Outside F anti-frame (not plain magnitude pruning; residual projection is a structural equivalence class argument) | Motivated by Kalman-observer professional prior

**Ambition**
Determine whether the residual stream has a conserved subspace — one or more principal directions that are present with near-constant magnitude across all 28 layers and all calibration tokens. If the top-r conserved directions span a stable r-dimensional subspace across layers, the attention/MLP operations are spending flops to maintain this subspace (adding zero information to it, since it is conserved). Projecting out the conserved subspace before each attention/MLP block and re-adding it after is algebraically equivalent to the original computation (exact, no approximation) — and reduces the effective dimension of the computation from 2048 to 2048-r, enabling downstream rank-reduction experiments to be conducted in the (2048-r)-dim subspace.

**Mechanism**
1. Forward pass on 1K calibration tokens; record residual-stream states {h_L^(t)} at all 28 layers and all tokens t.
2. For each layer L, compute the PCA of {h_L^(t)}: top-r eigenvectors V_r^L (the "principal residual basis" at layer L).
3. Cross-layer alignment: compute Grassmann distance between V_r^L and V_r^{L'} for all layer pairs (L, L'). If the Grassmann distance < ε in ≥20 of 28 layers, the top-r subspace is conserved across layers.
4. If conserved: project out the top-r subspace from the input to each layer (h_L → h_L - V_r^L (V_r^L)^T h_L), apply attention+MLP in the (2048-r)-dim space, and add back the conserved component.

**Why our findings make this work**
CF2 says cos(h_L, h_{L+1}) ≈ 0.99 — layers make tiny updates to the residual stream. The control-theorist's interpretation: the residual stream is operating near a fixed point in some subspace. The conserved-subspace hypothesis generalizes CF2 from "the full vector is conserved (in direction)" to "the top-r principal directions are conserved (as a subspace)." If true, this is a genuine conservation law with structural consequences for all downstream measurements (CQBS, JVOC, RAOK-SCALE all operate in the full 2048-dim space; operating in the (2048-r)-dim subspace would improve all of them). The measurement is 30 minutes; the algebraic identity (project out, compute, add back) is exact.

**What would have to be true**
- The top-r PCA directions of h_L must be stable across layers (Grassmann distance < ε for some meaningful r ≥ 4) — verified in E1.
- The conserved directions must carry no per-token information that affects downstream computations (if they do carry token-specific information, projecting them out degrades quality — tested in E2).

**Experiment cascade**
- E1 (30 min): Forward pass 1K tokens; PCA at each layer; compute cross-layer Grassmann distance. Go: top-4 subspace Grassmann distance < 0.1 in ≥20 of 28 layers.
- E2 (1h, conditional on E1 GO): Measure ΔNLL with projected-out top-r subspace vs baseline (verify that projecting out the conserved directions doesn't lose quality).
- E3 (2h): If E2 ΔNLL ≤ +0.05 nats: repeat CF11's W_Q global-K experiment on the (2048-r)-dim projected residual stream. Expect lower K_eff to achieve the same cumvar — i.e., the W_Q compression is more effective in the projected subspace.

**What you are NOT proposing**
Not a bottleneck layer or an architectural change. Not a retraining step. The projection is a per-token inference-time operation (matrix multiply, 2048×r, same order as W_Q multiply). The goal is to reduce the effective dimensionality for all downstream compression experiments — a meta-level improvement that makes CQBS, JVOC, and RAOK-SCALE all work better if the conserved subspace is real.

White space: control-theorist professional prior. Not in any orientation's anti-frame list. Not covered by any kill-list entry (CF2 measured global cosine similarity, not per-subspace conservation law). The Grassmann distance between layer-level PCA bases is a new measurement object.

---

### S7 — MPRW: Marchenko-Pastur Random-Weight Bulk Audit [FREE SWING]

**Elegance class: no-SGD-reformulation (random matrix theory provides a principled null model)**
Track B | [FREE SWING] — no CF tether; structural argument stands on its own

**Ambition**
Determine which weight matrices in Qwen3-1.7B are consistent with a random Gaussian null model (Marchenko-Pastur bulk) and which have SIGNAL singular values above the bulk. The random-matrix bulk is structurally "compressible to zero" — it represents the part of the weight matrix that is no more structured than noise. A weight matrix whose entire singular value spectrum falls within the Marchenko-Pastur bulk has NO signal above noise at this rank range; a matrix with signal singular values has those as its compressible-to-low-rank component.

**Mechanism**
1. For each weight matrix W (shape m×n) in Qwen3-1.7B, compute the Marchenko-Pastur bulk upper bound σ_max^{MP} = (1 + √(n/m))² × σ_scale, where σ_scale is the calibrated noise variance from the bulk's fit.
2. Count the number of singular values exceeding σ_max^{MP} — call this the "signal rank" r_signal.
3. Measure r_signal for all matrices: W_Q, W_K, W_V, W_O, W_gate, W_up, W_down, tied lm_head.
4. Hypothesis: W_gate/W_up have r_signal ≈ 0 (consistent with random Gaussian, no structure above noise) and W_Q has r_signal ≈ 128 (signal equals the CF11 head-shared subspace).

**Why our findings make this work**
CF8 says W_gate/W_up have r_99/d ≈ 1.0 — flat spectra. The Marchenko-Pastur prediction for a random Gaussian m×n matrix is ALSO a flat bulk spectrum (all singular values in the [σ_min^{MP}, σ_max^{MP}] range). The hypothesis that W_gate/W_up are essentially random-matrix-like (which would explain their flat spectra) is consistent with CF8 but has never been tested with the MP null model. CF11 says W_Q has r_99/d ≈ 0.63 — more concentrated than random. The MP analysis would precisely quantify how many singular values are ABOVE the random baseline. This provides a data-driven "compressible rank" estimate that is independent of any quality threshold — the signal rank is the rank that the trained model learned, above the random-weight baseline.

**What would have to be true**
- CF9 precondition: the Marchenko-Pastur bulk applies to the singular values of a random m×n Gaussian matrix; the precondition is that the matrix has i.i.d. Gaussian entries at initialization. If Qwen3's training has structured gradients that push entries away from i.i.d. Gaussian, the MP bulk may not be the right null model. Precondition verification: test whether the EMPIRICAL singular value histogram of W_gate fits the MP density. If yes (good fit), the null model is valid. If no (heavy tails or bimodal), an alternative null model (e.g., GOE, Wishart) is needed but the spirit of the measurement survives.
- Smallest test: ≤ 8h. Yes: pure SVD computation on pre-loaded bf16 weights + MP density fit takes <1h.

**Experiment cascade**
- E1 (45 min): SVD all major weight matrices; compare singular value histograms to MP bulk fits. Report r_signal per matrix. Go: W_gate/W_up have r_signal < 50 (mostly random bulk) AND W_Q has r_signal ≈ 128.
- E2 (1h, conditional): If W_gate r_signal > 100 (unexpected signal), examine the top-r_signal singular vectors for interpretable structure (are they token-type dependent? layer-dependent?).
- E3 (1h): Report the r_signal vs r_99 discrepancy per matrix. This provides a principled "minimum representational rank" (r_signal) vs the "quality-preservation rank" (r_99) — two different measures of the same compressibility question.

**What you are NOT proposing**
Not a new compression mechanism. Not a new training method. Purely an auditing tool that provides a data-driven lower bound on the minimum compressible rank for each weight matrix class. The audit result feeds all downstream rank-reduction experiments with a principled null-model benchmark.

White space: [FREE SWING]. No CF tether required. The Marchenko-Pastur bulk audit has been applied to neural network weight matrices in the theoretical literature (Martin & Mahoney, 2021) but has never been run on Qwen3-family weights with the specific comparison to CF8/CF11 measurements. Smallest test: 45 min. CF9 internal check: MP bulk precondition is tested empirically in E1 (histogram fit), not assumed.

---

### S8 — SLAL: Sequential-Layout Adaptive Layer-Ordering

**Elegance class: none (substrate engineering)**
Track B | Orientation U white space | Motivated by sequential-only constraint

**Ambition**
Re-layout the Qwen3-1.7B GGUF file so that the access pattern during a single forward pass is exactly sequential — one contiguous read per forward pass, with no seeks. Currently, gguf layout interleaves tensors by type (all W_Q tensors first, then all W_K, etc., OR layer-by-layer but within-layer matrices may not be in compute-graph order). Profiling the actual access sequence and rewriting the GGUF tensor table to match it eliminates all non-sequential reads, maximizing the benefit of NVMe sequential-read bandwidth (3 GB/s, vs NVMe random-read bandwidth of 400K IOPS × 4 KB = 1.6 GB/s — a 1.9× gap for purely sequential vs purely random).

**Mechanism**
1. Instrument ik_llama.cpp to log tensor reads during a single-token forward pass: (tensor_name, file_offset, size).
2. Sort the tensor table by first-access offset to determine the current layout.
3. Build a re-ordered GGUF file where tensors appear in first-access order.
4. Benchmark both layouts: measure effective read throughput (GB/s) and tok/s on an NVMe-forced scenario (DRAM cache evicted before each forward pass).

**Why our findings make this work**
U1 (PrefetchVirtualMemory) and CQBS-IO (S3) both assume the model is accessed with some structure; this idea generates the profile that validates or refutes those assumptions. More directly: if the GGUF file is NOT in access order, then the Windows I/O scheduler cannot deliver sequential-read throughput even with PrefetchVirtualMemory, because the virtual address ranges being prefetched are non-contiguous on disk. The 1.9× sequential-vs-random NVMe gap is real and substrate-level; this idea captures it without any code change beyond a file-format pre-processing step (re-layout script + updated GGUF tensor offset table).

**What would have to be true**
- ik_llama.cpp must access tensors in a consistent, profiling-reproducible order across tokens (not random access).
- The current GGUF layout must NOT already be in access order (if it is, the idea produces a null result but confirms U1 and CQBS-IO can rely on sequential access).

**Experiment cascade**
- E1 (30 min): Instrument ik_llama.cpp; log tensor read sequence for one 512-token prompt. Compute fraction of reads that are in-order vs out-of-order relative to current GGUF layout. Go: ≥30% reads are out-of-order (sufficient disorder to benefit from re-layout).
- E2 (1h, conditional on E1 GO): Write GGUF re-layout script; produce re-ordered file.
- E3 (2h): Benchmark forward-pass throughput under NVMe-forced conditions with original and re-ordered GGUF. Expected gain: up to 1.9× at full sequential-access condition; realistic gain given mixed sequential/random: 1.2–1.5×.

**What you are NOT proposing**
Not a new file format. Not a custom NVMe queue scheduler. Not a kernel change. Purely a file-preprocessing step that makes the existing Windows I/O path more effective by matching the data layout to the access pattern.

White space: outside all orientation anti-frames. Motivated by the sequential-only constraint reframing. Combines naturally with U1 (PrefetchVirtualMemory can schedule the sequential layout efficiently) and S3 (CQBS-IO's re-layout uses the same GGUF tensor-table customization mechanism).

---

## 7. Output Handoff

### Stage 4 idea list (S1–S8)

| # | Name | One-line description | Motivating section |
|---|------|---------------------|-------------------|
| S1 | VQWQ | VQ-codebook coverage of W_Q projected-query space; FSMD stripped primitive, CF9-safe | §3 cross-pollination (CP-1) |
| S2 | PASS | Per-matrix-class bpw schedule derived from v2 spectral ordering; HYBRID-TIER-70B transposition | §3 cross-pollination (CP-2) |
| S3 | CQBS-IO | GGUF sequential-IO optimizer conditioned on CQBS shared-basis layout | §3 cross-pollination (CP-3) |
| S4 | CDCA | Compiler-guided dead-column audit of W_gate: coverage-based structural pruning | §4 professional prior (compiler) |
| S5 | ZKAM | Database zone-map block-skip for KV-cache attention; exact computation | §4 professional prior (database) |
| S6 | RCPS | Conserved-principal-subspace projection via control-theorist Kalman-observer framing | §4 professional prior (control) |
| S7 | MPRW | Marchenko-Pastur bulk audit of all weight matrices; random-matrix null model for compressibility [FREE SWING] | §5 constraint (10 MB seed) |
| S8 | SLAL | Sequential GGUF tensor-layout profiling and re-ordering for NVMe sequential-read capture | §5 constraint (sequential-only) |

### Recommendations to round runner

| Orientation | Verdict | Specific action |
|-------------|---------|----------------|
| R (Reach) | KEEP | Add anti-frame: "Measurement experiments whose first rung is identical to another orientation's settler — assign those settlers to First-Principles; Reach owns the cascade arithmetic." |
| C (Composition) | KEEP | Add anti-frame: "CF2 × CF11 compositions that reduce to another stacked-W_Q-SVD variant — the protocol is fully worked out." |
| F (First-Principles) | KEEP | Add anti-frame: "Algebraic-fold proposals targeting the residual-stream rotation gauge (already covered by RGPS, RSGO, RMGF) — orient toward untouched gauges: sign-flip gauge, per-channel W_down output scale, lm_head row-norm gauge in untied configurations." |
| U (Substrate) | KEEP, tighten | Add anti-frame: "Substrate primitives that affect latency variance only, without improving residency or tok/s floor — useful tooling but out of scope as pipeline candidates." Add positive signal: "Substrate primitives motivated by compression-induced layout changes (e.g., CQBS-IO, SLAL) are strongly in-orientation." |
| A (Constraint-Alien) | KEEP | Add anti-frame: "Constraints that have already been exhausted in run_002: head-permutation gauge-fixing, append-only KV, gauge-symmetry exploitation (covered by GHPI, ASAKS, RGPS, SSIF, RMGF). Rotate toward unexplored constraints: 10 MB seed, peer-weight swap, O(1)-token parallel decode." |

### Ideas recommended for Stage 5 weighting alongside Stage 3 advancers

**Weight S5 (ZKAM) heavily.** It fills the KV-side compute gap (not storage) that was explicitly identified as under-represented. The mechanism is an exact computation (no approximation for skipped blocks), the DB zone-map prior is a well-grounded professional transplant, the CF9 precondition (blocks with uniformly low attention scores exist) is empirically testable in 1h, and the payoff scales with context length (most valuable at 32K+). No current run_002 Stage 3 advancer targets the KV-side compute path.

**Weight S2 (PASS) heavily.** It directly converts the v2 empirical measurement substrate (CF11, CF12, JVOC-pending) into a bpw schedule that the published landscape has not computed from this specific per-matrix-class ordering. It is a free structural finding (no new experiment needed once JVOC runs) and it directly serves the 70B-residency target. The transposition from HYBRID-TIER-70B's stripped primitive into the CF11-grounded per-class frame is the key rescue.

*End of Stage 4 Skeptic Explorer — Run 002*
