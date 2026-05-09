# Stage 1 — Orientation F (First-Principles) — Run 009

**Hard kills in effect for this run:**
- Cross-layer W_Q subspace sharing: KILLED (v2-CHEAP-TEST-001; class-level)
- Arbitrary O(d) residual-stream rotation: KILLED (v2-S3-R004-001; RMSNorm incompatibility)
- Within-layer W_Q/W_K joint compression: KILLED per run instructions (run_005 F1 covered; also RoPE-rotation-on-K makes the shared-input argument degenerate)
- Softmax × RoPE interplay as a primary mechanism: KILLED per run instructions (run_003 F3-SRSC rejected on CF9; R003 stage-6 confirmed)

**Saturation signal:** W_V/W_O fusion spectrum, RMSNorm gain absorption, W_down output-subspace, head permutation gauge, AERO linearizability — all heavily covered in runs 001–008. Ideas below target genuinely fresh territory.

---

## F1-MQKP — Attention Product M = W_Q W_K^T Direct Spectral Compression (MQKP)
**F / Track B**

### Mechanism
Track B — compression. CF11 established W_Q r_99/d ≈ 0.63 and W_K r_99/d ≈ 0.79 *independently*. But the attention score computation uses the PRODUCT M^L = W_Q^L (W_K^L)^T ∈ R^{d×d} (shape 2048×2048 per layer), not W_Q and W_K separately. Compressing W_Q and W_K independently is suboptimal: two separate rank-512 approximations give W̃_Q W̃_K^T with squared approximation error ‖M − W̃_Q W̃_K^T‖_F that includes cross-terms the separate compressions cannot cancel. The first-principles identity: the MAP-optimal rank-r approximation of M = W_Q W_K^T is the rank-r SVD of M itself — NOT the product of the rank-r SVDs of W_Q and W_K separately. This is a standard matrix approximation theorem (Eckart–Young). The structural question: does M = W_Q W_K^T have a significantly more concentrated spectrum than either factor alone? Rank(M) ≤ min(rank(W_Q), rank(W_K)) ≤ r_99(W_Q) ≈ 1293. If M has r_99/d ≤ 0.50 — lower than either factor — then storing M's SVD factors directly is more efficient than storing W_Q and W_K separately at matched rank. Mechanism: replace (W_Q^L, W_K^L) with (U_M^L Σ_M^{L,1/2}, Σ_M^{L,1/2} V_M^{L,T}) at rank K, reading xW_Q and xW_K as x U_M Σ^{1/2} and x V_M Σ^{1/2} respectively. Does not require cross-layer sharing (killed). Does not require joint compression of Q+K input directions (killed in this run). Relies on CF11's established spectra but asks a structurally distinct question: the product's spectrum, not the factors'. CF9 precondition: Eckart–Young theorem applies to any real matrix; preconditions hold trivially.

### Residency arithmetic
Qwen3-1.7B: W_Q and W_K each 2048×2048 × 2 bytes × 28 layers = 448 MB each, total 896 MB. CF11 baseline (independent compression at K=512 each): each factor at K=512 gives 2×(2048×512) × 2 bytes × 28 layers = 224 MB each = 448 MB total for Q+K. If M = W_Q W_K^T has r_99/d ≤ 0.50 (hypothesis), joint K=400 captures 99%: U (2048×400) + V (400×2048) × 28 layers = 2×(2048×400)×2 bytes × 28 = ~179 MB. Saving vs CF11 independent compression: 448 MB → 179 MB, additional 60% reduction on the Q+K block. On 70B-scale: W_Q + W_K ≈ 14 GB; at K=400 (product-SVD) ≈ 3.5 GB vs ~7 GB (independent K=512). ΔNLL: the product approximation is Eckart–Young optimal for the attention score matrix; per-factor approximation is suboptimal. Expected ΔNLL ≤ the independent-compression bound at equal bit budget.

### Novelty gloss vs the kill list and the published landscape
Closest kill: within-layer W_Q/W_K joint compression (this run's kill). MQKP is categorically different: joint compression of the PRODUCT M = W_Q W_K^T, not of the input subspace. The kill blocks sharing the input basis of W_Q and W_K; MQKP shares the output space of M via its own SVD — the mathematical object being compressed is the attention score operator, not the projection matrices. Closest published: MLA (DeepSeek-V2) compresses the KV path at training time. A3 (arXiv:2505.12942) minimizes attention-logit error. Neither applies the Eckart–Young theorem directly to W_Q W_K^T as the compression target. The structural novelty: every run's F-orientation has measured W_Q and W_K independently; none has measured the product M's spectrum. If M has r_99/d substantially below either factor, the gap between product-SVD and independent-SVD compression is a NEW structural finding.

### Smallest experiment
**Claim**: M^L = W_Q^L (W_K^L)^T at layers L=7,14,21 has r_99/d ≤ 0.55 (i.e., at least 13% more concentrated than W_Q alone at 0.63).

Protocol: load W_Q^L and W_K^L for 3 representative layers (L=7, 14, 21), compute M^L = W_Q^L @ W_K^L.T (each 2048×2048, one matmul per layer), run truncated SVD at rank 600 (scipy.sparse.linalg.svds), compute var@{300,400,500,600}. Compare against CF11 per-factor baselines. Runtime: 3 matrix multiplies + 3 truncated SVDs, ~15 min on Ryzen.

**Go threshold**: var@400 ≥ 0.99 in all 3 layers (i.e., rank 400 captures 99% variance vs CF11's W_Q needing K=512 for 99%). **NO-GO**: product spectrum is no more concentrated than either factor; Eckart–Young offers no gain over independent compression. Structural finding: W_Q and W_K are "spectrum-aligned" — compressing them independently does not lose cross-term information.

### Primary risk
M = W_Q W_K^T is a 2048×2048 product; its spectrum at finite K may be ill-conditioned if W_Q and W_K have near-orthogonal column spans. Mitigation: run the 3-layer pilot; ill-conditioning shows up as a flat M spectrum even though W_Q and W_K are individually concentrated — itself a structural finding.

---

## F2-CKDJ — CF3 Outlier Channels × CF11 W_Q Subspace Alignment (CKDJ)
**F / Track A**

### Mechanism
Track A — arch-transposition (computation graph reorganization). This is a genuine COMPOSITION of two confirmed findings: CF3 (at K=1%, the top ~20 activation channels per layer are token-dynamic with Jaccard ≈ 0.31; at K=0.1%, the top ~2 channels are token-static with Jaccard ≈ 0.718) and CF11 (global W_Q spans a ~128-dimensional subspace of d=2048; 16 heads collectively use only that subspace). The coupling hypothesis: the 2 token-static outlier channels (CF3 K=0.1%, Jaccard=0.718) lie WITHIN the top-128-dimensional W_Q subspace, because those channels are the dominant input directions for the attention query computation. If the outlier channels are the same channels that W_Q cares about most (highest row-energy in the top-128 W_Q singular vectors), then the following algebraic fold is valid: (1) the static outlier channels index a fixed 2-dimensional slice of the W_Q input; (2) that slice contributes a CONSTANT additive term to every Q vector across tokens, exploiting softmax shift-invariance to strip it; (3) W_Q's residual (on the other 2046 channels) has lower dynamic range and tighter quantization. This is a first-principles derivation that runs from CF3's Jaccard measurement → CF11's subspace measurement → a specific algebraic strip that is lossless. The coupling claim as equation: let e_1, e_2 ∈ R^{2048} be the two static outlier channel indices (CF3 K=0.1% channels). If ‖W_Q e_1‖, ‖W_Q e_2‖ are among the top-2% of W_Q column norms, then the contribution W_Q diag(x)[e_1, e_2] to the query vector has a near-constant component across tokens (because x_{e_1}, x_{e_2} are the static outlier channels), strippable by softmax shift-invariance without any approximation.

### Residency arithmetic
This mechanism does not reduce weight residency; it reduces runtime query dynamic range. The payoff is in Q activation precision. If stripping the static-channel contribution reduces Q's L∞ norm by ≥ 20%, INT4 quantization of Q (which is bandwidth-bound in long contexts) achieves 0.10 nats lower error. For Qwen3-72B at 32K context: KV cache in INT4 = 2.5 GB vs INT8 = 5 GB → 2.5 GB KV saving when Q quantization becomes viable. The weight residency of the strip vector is 2 channels × 2048 (W_Q columns corresponding to the static channels) = negligible. The algebraic precision improvement is the payoff; the mechanism can be layered on top of any weight quantization scheme (PDAP, RAOK) without interfering.

### Novelty gloss vs the kill list and the published landscape
Closest kill: CF3 three-tier activation quantization (run 001 F4-OCSSQ). CKDJ is different: it uses the static outlier channels specifically to enable softmax shift-invariance stripping of W_Q's constant-input contribution — a COMPUTATION GRAPH change, not a quantization tier assignment. Closest published: PrefixQuant (arXiv:2410.05265) handles token-wise outliers in activation quantization. CKDJ couples the static-channel structure to W_Q's column norms, testing whether the same 2 channels are simultaneously (a) static across tokens (CF3) and (b) high-energy columns of W_Q (CF11). This coupling claim has no published treatment. The lossless strip (softmax shift-invariance + static channel identification) is the algebraic identity; whether the static channels are indeed the W_Q-heavy columns is the empirical question.

### Smallest experiment
**Claim**: The top-2 static outlier channels (CF3 K=0.1%, identified from calibration) are among the top-5% of W_Q column norms (by ‖W_Q[:, c]‖ / median column norm) in at least 20 of 28 layers.

Protocol: (1) Run 200-token calibration pass, record per-layer activation columns with highest Jaccard stability (static channels, K=0.1%); (2) Load W_Q per layer, compute column norms; (3) Check whether the 2 static channels rank in the top-5% (top ~100 of 2048) by column norm. Runtime: ~35 min (calibration forward pass + column norm computation).

**Go threshold**: top-2 static channels are in top-5% W_Q column norms in ≥ 20/28 layers. **NO-GO**: static activation channels are NOT high-energy W_Q columns — the two CF findings are structurally independent; softmax stripping cannot be applied losslessly. Structural finding: CF3 outlier dynamicity and CF11 W_Q concentration are decoupled phenomena.

### Primary risk
The CF3 static channels (K=0.1%) may be 2 attention-sink-adjacent channels (BOS-driven) rather than channels W_Q amplifies. If the static channels have low W_Q column energy, the strip contribution is negligible. Mitigation: the 5-step measurement settles this without cascade investment; any failure mode is itself a structural finding about the CF3/CF11 independence.

---

## F3-LAYER1FOLD — Layer 1 Gate Null-Space Fold: Structural Origin of CF6 Anomaly (L1GNF)
**F / Track A**

### Mechanism
Track A — arch-transposition. CF6 is specific: layer 1 of Qwen3-1.7B has 36.1% of SwiGLU gate neurons with variance < 0.05 (near-constant gate output). Layers 2–28 have < 2.5% foldable neurons. This is a 14× concentration of the foldable-neuron population in one layer. CF7/CF8 killed global low-rank MLP compression; CF6's layer-1 anomaly survived. The first-principles derivation of WHY layer 1 is different: layer 1 is the first layer to see the raw token embedding x^0 = W_E e_t (where e_t is the token's embedding row). The embedding space is high-dimensional but operates on discrete tokens — the input covariance C_x^0 = E[x^0 x^{0,T}] is determined entirely by the token frequency distribution. Hypothesis: for low-frequency tokens, the embedding vector x^0 may have small projections onto many W_gate^1 row directions, making silu(W_gate^1 x^0) ≈ silu(0) ≈ 0 ≈ constant for those neuron-token pairs. Specifically, the 36.1% foldable neurons at layer 1 have gate rows W_gate^1[j,:] that are nearly orthogonal to the high-frequency subspace of the token embedding space (the subspace spanned by the embeddings of the top-10K most frequent tokens, which dominate calibration data). The algebraic identity: for those neurons, silu(W_gate^1[j,:] x) ≈ α_j for all high-frequency tokens x, so the gate output is approximately constant and AERO's fold (absorb into W_up, drop W_gate row) applies. The mechanism: (1) measure C_x^0 from calibration; (2) find W_gate^1 rows with low projection onto the top-eigenvectors of C_x^0 (i.e., in the null(C_x^0) regime); (3) fold those rows into W_up^1 via the AERO identity W_up^1[:,j] × α_j → W_up^1[:,j]_new. The fold is exact on the calibration distribution. Only layer 1 has this property because subsequent layers' inputs are residual-stream vectors with dense covariance (all directions used), while layer 1 sees discrete token embeddings with structured null space. CF10 compliance: the fold uses calibration-estimated constants α_j (one scalar per neuron, ~6144 scalars); n_params = 6144 << n_samples × d = 200 × 2048 = 409K. Well-conditioned.

### Residency arithmetic
36.1% of 6144 layer-1 gate neurons = 2218 neurons foldable. Folding means dropping the corresponding W_gate^1 rows and absorbing into W_up^1: W_gate^1 storage saved = 2218 × 2048 × 2 bytes = 9.1 MB on a 3.4 GB model. Marginal on its own. BUT the structural value: if the null-space criterion is confirmed for layer 1, it predicts fold-eligibility for ALL models trained with tied-embedding or similar low-rank input structure. Extending to Qwen3-72B: layer 1 has d_intermediate ≈ 28672; at 36.1% = 10,350 foldable neurons → 10350 × 8192 × 2 bytes ≈ 170 MB saved on layer 1 alone. Additionally, the criterion may identify a small set of foldable neurons in layers 2-5 (if those layers also have partially structured input covariance) — running the null-space criterion across all 28 layers costs 30 min and could surface additional folds beyond CF6's explicit measurement.

### Novelty gloss vs the kill list and the published landscape
Closest kill: SDZC (SwiGLU Dead-Zone Gate Collapse) — killed as a global scheme. L1GNF is layer-1-specific AND proposes a structural derivation (null-space alignment), not global dead-zone detection. Closest published: AERO (arXiv:2410.13060) performs activation removal globally with retraining. L1GNF applies AERO's algebraic fold post-training, restricted to the CF6-identified population, with a structural prediction for why layer 1 is special. Run 005 F3 (AEROT) tested per-neuron linearizability; L1GNF tests the more specific hypothesis that the foldable neurons correspond to the null-space of the input covariance. These are distinct experiments: AEROT asks "how linear is silu?"; L1GNF asks "does the gate row lie in null(C_x)?". The null-space criterion, if confirmed, is a zero-calibration-sample fold criterion — applicable to new models without running silu-linearization tests.

### Smallest experiment
**Claim**: For layer 1's CF6-foldable neurons (std(silu(W_gate^1 x)) < 0.05 on calibration), the projection of W_gate^1[j,:] onto the top-256 eigenvectors of C_x^0 accounts for ≤ 20% of ‖W_gate^1[j,:]‖^2 — i.e., the foldable-neuron gate rows lie primarily in the null space of the token-embedding input distribution.

Protocol: (1) Compute C_x^0 from 200 calibration tokens (layer-1 input activations, shape 2048×2048). (2) Eigendecompose C_x^0; retain top-256 eigenvectors (covering ~90% of activation variance). (3) For each of 6144 gate neurons, compute null-space projection fraction = 1 - ‖W_gate^1[j,:] U_256‖^2 / ‖W_gate^1[j,:]‖^2. (4) Compare foldable vs non-foldable neurons. Runtime: 200-token forward pass + one 2048×2048 eigendecomposition + matrix projections → ~25 min.

**Go threshold**: mean null-space projection fraction for foldable neurons ≥ 0.70 AND significantly higher than non-foldable (effect size > 0.30). **NO-GO**: null-space projection is indistinguishable between foldable/non-foldable neurons — the CF6 anomaly is a calibration-distribution artifact, not a structural geometric property. Structural finding on NO-GO: layer-1 gate foldability is not explained by the input-covariance null-space; some other mechanism governs it (e.g., initialization artifacts, early training dynamics).

### Primary risk
The 200-token calibration set may produce an ill-conditioned C_x^0 (rank < 256). Mitigation: use the pre-tokenized WikiText-2 training split (>1M tokens are available); subsample 500 diverse tokens to ensure C_x^0 is well-conditioned before measuring null-space fractions.

---

## F4-WQKRESID — W_Q / W_K Spectral Asymmetry as a Rotation-Free Compression Schedule (WQKRS)
**F / Track B**

### Mechanism
Track B — compression. CF11 measured W_Q r_99/d ≈ 0.63 and W_K r_99/d ≈ 0.79 independently. This 16-percentage-point gap (W_Q is strictly more concentrated than W_K) is a structural asymmetry that no previous F-ideator has converted into a compression schedule. The first-principles derivation: the attention score Q K^T / √d has two factors with different intrinsic ranks. Compressing both to the same K wastes budget on W_Q (which could be compressed tighter) and over-compresses W_K (which needs more rank to maintain quality). The optimal rank assignment given a total bit budget B is derived from the per-matrix Pareto frontier: minimize ‖ΔNLL_Q(K_Q) + ΔNLL_K(K_K)‖ subject to K_Q × (2048 + 2048) × 2 bytes + K_K × (2048 + 2048) × 2 bytes ≤ B. Because W_Q's spectrum decays faster, the Lagrangian allocates fewer bits to W_Q and more to W_K than a symmetric allocation. The algebraic identity: for a fixed attention-score approximation error budget ε, the optimal (K_Q*, K_K*) satisfies σ_{K_Q*+1}(W_Q) / σ_{K_K*+1}(W_K) = λ for some Lagrange multiplier λ derived from the singular value decay rates. This is the "water-filling" spectral allocation (Shannon's classic result, applied to matrix approximation). The mechanism: measure the singular value decay curves of W_Q and W_K for each layer, compute the Pareto-optimal (K_Q, K_K) pair per layer under total bit budget = current (K_Q + K_K) × d × 2 bytes, and apply per-layer asymmetric rank allocation. No retraining; uses only CF11 spectrum data already partially measured. The schedule does NOT require cross-layer sharing or rotation — it is independent per-layer. The "rotation-free" qualifier distinguishes it from run 002's RSGO (which needed an orthogonal G).

### Residency arithmetic
Qwen3-1.7B, W_Q + W_K per layer: 2 × 2048×2048 × 2 bytes = 32 MB/layer × 28 = 896 MB. CF11 symmetric schedule at K=512 each: 2 × 2 × (2048×512) × 2 bytes × 28 = 448 MB. Water-filling at same budget but asymmetric: if W_Q gets K_Q=350 (because spectrum is more concentrated) and W_K gets K_K=670 (less concentrated, needs more rank), the bit budget is: 2048×(350+670) × 4 bytes × 28 = 2048×1020×4×28 = 233 MB — same as symmetric K=510 each, but ΔNLL should be lower because the allocation matches the per-matrix rate-distortion curve. The gain is not in total bytes but in quality at fixed bytes. If water-filling at K_total=1020 achieves ΔNLL = +0.15 nats vs symmetric K=510 each at +0.25 nats, that is a 0.10-nat quality improvement at zero additional storage cost. On 70B-scale, the gain compounds over 80 layers.

### Novelty gloss vs the kill list and the published landscape
Closest kill: per-head W_Q rank truncation (CF11 NO-GO). WQKRS is global-per-layer, not per-head. Closest published: GPTQ uses per-column Hessian weighting for asymmetric quantization; that is activation-weighted and requires calibration data. WQKRS uses only the pre-measured weight spectra (CF11 data) to derive the water-filling schedule — no calibration data needed for the allocation step. The asymmetric W_Q/W_K rank allocation under a joint bit budget has not appeared in the published attention-compression literature. The "water-filling" framing (Shannon 1948, applied to matrix approximation) is the algebraic identity making this first-principles.

### Smallest experiment
**Claim**: At layers 7, 14, 21, the water-filling (K_Q*, K_K*) schedule at total rank budget K_total = 1020 achieves ΔNLL ≤ +0.18 nats, outperforming the symmetric schedule (K_Q = K_K = 510) at ΔNLL ≤ +0.25 nats.

Protocol: (1) Full SVD of W_Q^L and W_K^L for 3 layers. (2) Compute water-filling K_Q*, K_K* from the singular value decay curves. (3) Apply both schedules, measure per-layer NLL contribution on 512-token held-out. Runtime: 3 SVDs + inference comparisons → ~30 min.

**Go threshold**: water-filling ΔNLL ≤ symmetric ΔNLL − 0.05 nats in all 3 layers. **NO-GO**: symmetric and water-filling achieve identical ΔNLL — the W_Q/W_K spectral asymmetry does not translate to a quality gap at matched bit budget. Structural finding: the attention score quality is insensitive to rank asymmetry between Q and K at the CF11-calibrated compression level.

### Primary risk
The water-filling gain may be below measurement noise for a 3-layer probe. Mitigation: run all 28 layers and aggregate; the per-layer gain is small but the aggregate over 28 layers may be measurable.

---

## F5-RESCOVAR — Residual-Stream Update Covariance: Active Subspace vs Layer Depth (RSUC)
**F / Track B**

### Mechanism
Track B — compression (compression-lr, model-structure). CF2 established cos(h_L, h_{L+1}) ≈ 0.99 for all layers — residual updates are small. Run 001 F3-RSCLE proposed measuring the update covariance C^L = E[δ_L δ_L^T] and using its top eigenvectors to define an active subspace for each layer. That idea was proposed but not run (run 001 was killed on the cross-layer W_Q question; RSCLE was not selected). This run revisits it with a sharper first-principles angle: not just measuring the active-subspace rank, but measuring how the active-subspace rank VARIES BY LAYER DEPTH. The hypothesis: early layers (L=1-5) have C^L concentrated in a small number of directions (only token-type-level updates needed early); mid-layers (L=10-20) have C^L spread across more directions (contextual integration); late layers (L=23-28) re-concentrate (output-probability-shaping). This depth-dependent covariance structure is a first-principles consequence of the residual stream's function: early layers specialize, mid-layers generalize, late layers collapse. If confirmed, it enables a LAYER-SPECIFIC compression schedule: compress early and late layers' weight matrices more aggressively (fewer active directions, higher rank concentration → lower K needed for same ΔNLL), compress mid-layers less. This is a depth-aware water-filling that strictly dominates uniform compression at fixed quality budget. CF10 compliance: the active-subspace measurement is an SVD of an empirical covariance — no fitted parameters; no CF10 risk.

### Residency arithmetic
Qwen3-1.7B: 28 layers × average ~24 MB per layer (all matrices combined) = 672 MB at bf16. A depth-heterogeneous schedule that compresses early layers at 2× and late layers at 2× but mid-layers at 1.5×: (28 × 24 MB × 1/2 × 10/28) + (28 × 24 MB × 1/1.5 × 10/28) + (28 × 24 MB × 1/2 × 8/28) = let's use rough averages: 240 MB early+late (K=512 equivalent) + 224 MB mid (K=768 equivalent) = 464 MB vs 672 MB baseline → 31% reduction from depth-aware schedule vs 33% from uniform K=512. The gain over uniform is modest, but the quality at that budget is higher (the mid-layers contribute more to quality and get more rank). On 70B-scale the compounding matters more.

### Novelty gloss vs the kill list and the published landscape
Closest kill: residual-stream active subspace (run 001 F3-RSCLE) — that proposal was for using the active subspace to reorganize inference; RSUC is explicitly a compression-schedule proposal, asking the depth-variation question as the primary output. Closest published: layer-wise sensitivity for mixed-precision quantization (e.g., GPTQ per-layer error) uses Hessian-based sensitivity; RSUC uses the geometric active-subspace rank, which is a weight-only measurement requiring no calibration data. No published work measures residual-stream update covariance across layers as a function of depth to derive a compression schedule. The depth-dependent concentration hypothesis (early/late more concentrated than mid) is derivable from first principles of the residual stream's function — it's testable with a 200-token forward pass.

### Smallest experiment
**Claim**: The top-r effective rank of C^L (update covariance) at r=90% variance threshold varies by ≥ 40% between the shallowest layer (L=1) and the deepest mid-layer (L=14) on Qwen3-1.7B.

Protocol: (1) Run 200-token forward pass, record h^L and h^{L+1} for all 28 layers. (2) Compute δ^L = h^{L+1} − h^L (shape n_tokens × d). (3) Compute C^L = δ^{L,T} δ^L / n_tokens (shape 2048×2048). (4) Eigendecompose; measure r_{90%} (rank needed for 90% variance). Plot r_{90%} vs layer depth. Runtime: 200-token forward pass ~10 min + 28 covariance eigendecompositions ~15 min = 25 min total.

**Go threshold**: r_{90%}(L=1) and r_{90%}(L=28) both ≤ 0.60 × r_{90%}(L=14) — early/late layers have ≥ 40% more concentrated update distributions than the mid-layer. **NO-GO**: r_{90%} is approximately constant across depth — residual updates have uniform directional diversity at all layers. Structural finding on NO-GO: the residual stream's information content is not depth-stratified, constraining any depth-aware compression schedule to be uniform.

### Primary risk
The 200-token calibration may show artificial depth-dependent concentration due to domain overfitting (WikiText-2 is relatively homogeneous). Mitigation: use 3 disjoint 67-token batches with different topics; report inter-batch consistency of the r_{90%}(L) profile. If the depth profile is stable across batches, it is structural.

---

## F6-TROPICALDECODE — Tropical Semiring Approximation of Attention Argmax [FREE SWING]
**F / Track A** [FREE SWING]

### Mechanism
Track A. In the tropical semiring (max-plus algebra), the operation ⊕ = max replaces addition and ⊗ = + replaces multiplication. A tropical matrix-vector product (A, x) → (max_j(A_{ij} + x_j))_i is a max-plus GEMV, computable with the same asymptotic operation count as a standard GEMV but using integer/fixed-point arithmetic if A is quantized. The first-principles claim: the argmax of the attention output (which head's attention mass is largest, and which key position receives it) is exactly the solution to a tropical GEMV — because argmax softmax(Qx)_j = argmax j of (Q x)_j, and max-plus captures this without the exponential. If we only need the top-1 or top-K attended positions (not the full softmax distribution), the tropical GEMV gives the answer with fewer operations than the full softmax. The inference mechanism: replace the full softmax attention computation for layers where the top-1 attended position captures ≥ 90% of attention mass (empirically: attention sinks at early layers for BOS-heavy sequences) with a tropical approximation — a max-plus GEMV using INT8 Q and K, identifying the dominant attended position, computing only that position's V contribution. No retraining. The algebraic identity: for a one-hot-like attention distribution (one dominant entry), softmax(QK^T)_j ≈ e^{max_j QK^T_j} / Z ≈ the one-hot vector on argmax j, and V × one-hot = V_{argmax j,:}. The tropical GEMV identifies argmax j without computing Z. Feasibility: INT8 max-plus GEMV is already a standard operation (max-pooling in SIMD); torch.max along the sequence dimension is implemented natively.

### Residency arithmetic
This does not change weight storage; it reduces GEMV compute and avoids the exponential-then-normalize softmax. For layers where attention is near-one-hot (concentrating on BOS / attention sink): the O(T × d_head) attention score computation becomes O(T) + O(d_head) (max-plus GEMV + one V lookup). At T=4096, d_head=128: saving ~99% of the attention computation for those layers IF the attention is sufficiently peaked. CF6's layer-1 anomaly is the most likely candidate: layer 1 may attend heavily to BOS. The tok/s uplift on Qwen3-1.7B if k=3 layers out of 28 are "tropical-eligible" (>90% attention mass on one position): 3/28 = 11% of attention computation reduced by 100% → ~3% end-to-end compute saving. Modest but structurally interesting as a novel operation type. On 70B where MLP is less dominant and attention is more, the fraction grows.

### Novelty gloss vs the kill list and the published landscape
No kill-list item touches tropical algebra for attention. Closest published: sparsifying attention by top-K selection (Sparseformer, Reformer, Longformer) reduces attention cost but still uses softmax on the selected subset. The tropical semiring proposal directly replaces softmax with max-plus, a different algebraic structure that exactly computes argmax without approximation error on the argmax itself (error only on the magnitude). Tropical algebra for neural networks has been studied theoretically (Cohen et al., 2016 on ReLU networks as tropical polynomials) but not applied to attention score computation. The CF9 precondition check: max-plus GEMV is a standard operation; no imported theorem with fragile preconditions needed.

### Smallest experiment
**Claim**: In Qwen3-1.7B at layer 1, the top-1 attended position (argmax of softmax attention) accounts for ≥ 80% of softmax attention mass on ≥ 60% of calibration tokens.

Protocol: run 200-token forward pass, record softmax attention weights at layer 1 for all heads; for each (token, head) pair, compute max(softmax_row) / sum(softmax_row) = concentration ratio. Report fraction of (token, head, layer) pairs above 0.80. Runtime: 20 min.

**Go threshold**: ≥ 60% of (token, head) pairs at layer 1 have concentration ≥ 0.80 → tropical approximation replaces softmax for majority of layer-1 computations. **NO-GO**: attention at layer 1 is diffuse (max fraction < 0.50 for most pairs) — tropical max-plus approximation would introduce substantial approximation error. Structural finding on NO-GO: even layer 1 has distributed attention (contradicting the "attention sink dominates layer 1" assumption); the tropical approximation requires a different targeting criterion.

### Primary risk
Even if attention concentration is high at layer 1, the V contribution of the non-dominant positions (the 20% non-argmax mass) may carry important information — the tropical approximation error in the output could be larger than expected. Mitigation: measure ΔNLL when tropical attention is applied only at layer 1 as the secondary experiment; the primary experiment is the concentration measurement alone.

---

## Convergence handles

1. **W_Q W_K^T product spectrum** (F1-MQKP: does M have r_99/d < min(r_Q, r_K)? This measurement is structurally independent of per-factor spectra and will be cited by any idea that compresses attention score computation directly)
2. **Static outlier channels (CF3 K=0.1%) × W_Q column energy alignment** (F2-CKDJ: whether the 2 static channels are high-energy W_Q columns; this coupling is a yes/no structural fact that any composition orientation will want for attention-quantization chain ideas)
3. **CF6 layer-1 gate null-space fraction** (F3-L1GNF: whether foldable neurons sit in null(C_x^0); a zero-calibration-cost fold criterion that generalizes across model families)
4. **Water-filling K_Q* / K_K* asymmetric rank schedule** (F4-WQKRS: the Pareto curve of ΔNLL vs rank for W_Q vs W_K; feeds any multi-matrix budget-constrained compression)
5. **Residual update covariance r_{90%}(L) depth profile** (F5-RSUC: whether r_{90%} varies by ≥40% from early to mid layers; this profile determines whether depth-aware compression schedules beat uniform schedules)
