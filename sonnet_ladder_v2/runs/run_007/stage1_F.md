# Stage 1 — Orientation F (First-Principles) — Run 007

**Kill acknowledgments for this run:**
- Cross-layer W_Q sharing: KILLED (v2-CHEAP-TEST-001)
- Arbitrary rotation breaking RMSNorm γ: KILLED (v2-S3-R004-001 GFQO)
- Within-layer Q/K joint decomposition (PALU+TransMLA flavor): KILLED per run-7 directive
- Softmax shift-invariance × RoPE: KILLED per run-7 directive
- MLP weight rank reduction (W_gate, W_up): KILLED (CF7, CF8)
- Tied lm_head SVD truncation: KILLED (CF12)

---

## F7-1 — W_O Left-Singular / W_V Right-Singular Spectral Coupling (WOVR)

**F / Track A**

### Mechanism

Track A. AQFKV measured W_Q (r_99/d ≈ 0.63) and W_K (r_99/d ≈ 0.79) but left W_V and W_O unmeasured. The algebraic structure of multi-head attention creates a specific coupling: for head h, the output is softmax(Q_h K_h^T/√d_k) · (X W_V_h) · W_O_h. The value projection W_V_h ∈ R^{d_model × d_head} and output projection W_O_h ∈ R^{d_head × d_model} compose as W_V_h W_O_h ∈ R^{d_model × d_model} — a rank-d_head matrix already. The first-principles question: does the LEFT singular basis of W_O_h align with the RIGHT singular basis of W_V_h? If U_{O,h} ≈ V_{V,h} (columns of U_O are approximately columns of V_V), then the composition W_V_h W_O_h is diagonal in those coordinates — its spectrum is the product σ_{V,i} · σ_{O,i} of the two matrices' singular values. Under this alignment, the effective rank of W_V_h W_O_h (the operator that matters for inference) is min(r_99(W_V_h), r_99(W_O_h)), and a single low-rank factorization W_V_h W_O_h ≈ U_r Σ_r V_r^T (r ≪ d_head) captures the dominant output behavior without storing either W_V_h or W_O_h separately. Residency: replace two d_model×d_head matrices with one d_model×r and one r×d_model. This is NOT cross-layer (killed); it is a per-layer algebraic fold of two SAME-LAYER matrices that already compose into a single operator in the forward pass. Precondition: U_O ≈ V_V measured by principal angles between U_O^L and V_V^L at each layer. Not CF8 (MLP); exploits CF11's attention-spectrum finding extended to the V+O pair.

### Residency arithmetic

Qwen3-1.7B: d_model=2048, d_head=128, num_heads=16. W_V: 16 heads × 128 × 2048 = 4M params. W_O: 16 heads × 2048 × 128 = 4M params. Both at bf16: 8 MB each, 16 MB/layer × 28 layers = 448 MB for W_V + W_O. If the per-layer W_V_h W_O_h composition has r_eff = r per head at r=32 (half of d_head=128): the factored form stores U_r (d_model × r_eff_total) + V_r^T (r_eff_total × d_model) with r_eff_total = num_heads × r = 16 × 32 = 512. Storage: 2048×512 + 512×2048 × 2 bytes = 4 MB/layer. Saving: 16 MB → 4 MB (4× on V+O) × 28 layers = 336 MB saved. Combined with W_Q at K=128 (CF11): total attention weight reduction ~390 MB on 3.4 GB = ~11.5%. At 70B scale: W_V + W_O ≈ 8 GB; 4× reduction = 6 GB saved. Quality cost: bounded by the principal-angle alignment; if U_O ≈ V_V to within 10°, the diagonal approximation error is O(sin(10°)) ≈ 0.17 fractional.

### Novelty gloss

Closest kill: F-WVOS (run_004) proposed W_V spectrum measurement and a fold but measured W_V independently. This idea tests the JOINT W_V · W_O composition spectrum and whether the two matrices are spectrally aligned (U_O ≈ V_V) — the alignment claim is distinct from independent spectrum measurement. Closest published: MLA (DeepSeek-V2) absorbs W_V into W_O at training time as a single KV-down projection; this tests whether that alignment ALREADY EXISTS post-training without any retraining intent. Principal-angle coupling between left/right singular bases of adjacent matrices in a composition has no published LLM treatment I know of.

### Smallest experiment

**Claim**: In Qwen3-1.7B at layer 14, the mean principal angle between U_O^{L14} (top-64 left singular vectors of W_O) and V_V^{L14} (top-64 right singular vectors of W_V) is ≤ 30°.

Measurement: load W_V[14] and W_O[14] (both 2048×128 per head; test head 0). Compute SVD of each, extract top-64 left/right singular vectors. Compute canonical angles via QR+SVD of (U_O^T V_V). Mean angle.

Runtime: ~5 min (two 2048×128 SVDs + small matrix multiply on Ryzen 5 7530U).

Go threshold: mean angle ≤ 30° → alignment is real; proceed to joint composition rank measurement and NLL test.
No-go: mean angle ≥ 60° → W_V and W_O are trained in orthogonal coordinate frames; no fold exists. Structural finding: attention V and O projections are NOT spectrally coupled post-training.

### Primary risk

W_V and W_O may be spectrally decoupled — SGD independently optimizes each, with W_O's left space shaped by the residual-stream output distribution and W_V's right space shaped by residual-stream input. Mitigation: 5-min experiment; failure is itself a structural finding (closes this class).

---

## F7-2 — Intrinsic Dimension of the Attention Score Manifold as a Compression Bound (IDSM)

**F / Track B**

### Mechanism

Track B. CF11 measured spectral rank of W_Q and W_K weight matrices. A distinct first-principles question: what is the intrinsic dimension of the ATTENTION SCORE DISTRIBUTION — the set of scalar products {q_i^T k_j : (i,j) pairs in context}? If the attention score matrix A ∈ R^{T×T} (T context length) lives on a low-dimensional manifold in distribution, then the KV cache entries needed to reconstruct A accurately are far fewer than T. Concretely: PCA on calibration-derived attention score matrices across layers. Compute A_L ∈ R^{T×T} for 500 calibration tokens; flatten to R^{T²}; compute intrinsic dimension via participation ratio = (Σ_i λ_i)² / (Σ_i λ_i²) of the attention-score covariance. If participation ratio PR ≪ T², the attention score matrix is intrinsically low-dimensional, which means: (a) KV cache can be compressed aggressively without fidelity loss; (b) the W_Q and W_K weight matrices generate outputs that occupy a much smaller information space than their matrix rank suggests. This is a unifying measurement: it explains both why CF11's W_Q has concentrated spectrum (the attention scores it must produce are low-dimensional) AND motivates KV cache compression from first principles (not from engineering, but from the intrinsic geometry of attention). Not CF8 (this targets the forward-pass product q^T k, not the weight rank). Not MLP. Not cross-layer. Falsifiable: PR is a number.

### Residency arithmetic

If PR/T² ≈ 0.05 (95% of variance in 5% of modes), the effective KV cache dimension per layer per head is 0.05 × T vectors instead of T vectors. At T=4096 (typical inference context), effective KV per head ≈ 205 vectors. Standard KV cache for Qwen3-1.7B at T=4096: 28 layers × 2 heads × 4096 × 128 × 2 bytes = 56 MB. Compressed at 5× (PR=0.05): 11 MB. The compression doesn't act on WEIGHTS (no residency gain there) but on KV CACHE (critical for long-context inference at 7.28 GiB). At 70B with T=32K context, KV cache would be ~20 GB — this PR measurement directly predicts whether a structured KV compression scheme (basis-projected cache) can hold 70B generation in 7.28 GiB.

### Novelty gloss

Closest kill: CLASE (cross-layer KV aliasing), ASCQ (attention-sink KV quantization) — those target KV quantization via engineering heuristics. This derives the compression bound analytically from the intrinsic dimension of the attention score manifold. Closest published: H2O and ScissorHands prune KV cache by attention magnitude heuristics. Linformer (arXiv:2006.04768) assumes low-rank attention but doesn't measure intrinsic dimension on trained models. This is a measurement that GROUNDS any KV compression claim in a first-principles bound.

### Smallest experiment

**Claim**: The participation ratio of the attention score covariance at layer 14 (T=128 tokens, Qwen3-1.7B) satisfies PR/T² ≤ 0.20.

Measurement: run 200 calibration sequences of T=128 tokens through Qwen3-1.7B, collect A_L[14] (128×128) for each sequence. Flatten each to 128²=16384-dim vector. Compute sample covariance (200 × 16384). Participation ratio = (Σλ)²/(Σλ²) normalized by T².

Runtime: ~20 min (200 forward passes on T=128 + covariance computation).

Go threshold: PR/T² ≤ 0.20 at layer 14 → attention score manifold is low-dimensional; proceed to measure PR across all 28 layers.
No-go: PR/T² ≥ 0.80 → attention scores are near-full-dimensional; KV compression via manifold structure is theoretically bounded poorly. Structural finding: attention diversity is empirically high; heuristic KV pruning success is not explained by distributional concentration.

### Primary risk

The participation ratio may be T-dependent (for small T=128, all 128² dimensions may be reachable). Mitigation: measure at T=128 and T=512; if PR/T² is approximately constant across T, the result is T-scale-invariant and the structural finding is robust.

---

## F7-3 — Residual Stream Rank-1 Update Decomposition: Are Attention Outputs Effectively Rank-1 Per Token? (AORT)

**F / Track A**

### Mechanism

Track A. The residual stream update from attention at layer L is: Δh_L = Attention_L(h_L) ∈ R^{d_model}. At single-token inference (decoding step), this is one d_model-dimensional vector update per layer. The question: what is the effective rank of the attention output over the calibration distribution? For a single forward pass, the attention output is a weighted sum of value vectors: out_L = Σ_h W_O_h · (Σ_j a_{ij}^h · v_j^h) where a_{ij}^h are attention weights and v_j^h are projected past values. This output, as a function of the query token only (conditioning on a fixed KV cache), is a SINGLE vector Δh_L ∈ R^{d_model}. But over the DISTRIBUTION of inputs, the set {Δh_L(x) : x ∈ calibration} forms a cloud in R^{d_model}. If this cloud is low-dimensional (intrinsic dimension ≪ d_model), then the attention output at decoding time lives in a small subspace, and W_Q's full d_model-dimensional projection is wasteful — the query only needs to identify a point in this low-dimensional output manifold. This motivates a different compression: instead of truncating W_Q directly (CF11), FIRST find the attention output subspace (call it S_out ∈ R^{d × r_out}), then truncate to the coordinates that project into S_out. This is the unifying object: r_out < K_Q (a tighter rank than CF11's K=128 GO, because S_out is the actual output space of attention, not just the input projection space). Algebraic path: project W_Q → P_{S_out}^T W_Q where P_{S_out} ∈ R^{d × r_out} is the calibration-PCA of Δh_L. Then the effective W_Q rank is bounded by r_out, which may be less than CF11's K=128. Not cross-layer; not arbitrary rotation (we're projecting W_Q, not rotating the residual stream arbitrarily).

### Residency arithmetic

If r_out = 64 (half of CF11's K=128 GO): global W_Q at rank 64 saves 50% vs CF11's K=128 baseline. CF11 K=128 = 8× compression. r_out=64 = 16× compression. At Qwen3-1.7B: full W_Q = 28 layers × 2048² × 2 bytes = 224 MB. CF11 K=128: 28 MB. r_out=64: 14 MB. Net additional saving: 14 MB — small on Qwen3 but the structural finding scales to 70B where W_Q ≈ 14 GB and the 16× bound vs 8× bound matters (7 GB vs 3.5 GB). Quality risk: r_out < K_Q is a new claim; ΔNLL at r_out=64 requires measuring.

### Novelty gloss

Closest kill: CF11 per-head W_Q NO-GO at K=64 (+1.53 nats) — that is PER-HEAD rank 64, not global rank 64 projected onto the attention output subspace. CF11's global K=128 is the current bound. This idea proposes a tighter bound by finding the attention OUTPUT subspace (not the input projection subspace) and projecting there. No published method derives W_Q compression bounds from the intrinsic dimension of attention's output distribution — existing methods use gradient (GPTQ) or activation covariance (SmoothQuant), not output-manifold PCA.

### Smallest experiment

**Claim**: The PCA intrinsic dimension of the attention output {Δh_L[14](x_t) : t=1..500} at layer 14 satisfies r_out ≤ 64 (var@64 ≥ 0.99).

Measurement: collect 500 attention output vectors Δh_L[14] from calibration tokens; run PCA; compute var@K for K ∈ {32, 64, 128, 256}.

Runtime: ~15 min (500 forward passes, PCA on 500 × 2048 matrix — trivial).

Go threshold: var@64 ≥ 0.99 → attention output is ≤64-dimensional; proceed to W_Q projection at r_out=64 and ΔNLL measurement.
No-go: var@128 < 0.90 → attention output is at least 128-dimensional; CF11's K=128 bound is already tight. Structural finding: the attention output subspace dimension matches the W_Q spectral rank — one does not imply the other, but they are consistent.

### Primary risk

The attention output cloud dimension may be exactly K=128 (matching CF11's W_Q spectral bound) — the output subspace is already the natural W_Q range, so this adds nothing. Mitigation: if r_out ≈ 128, the structural finding is that W_Q's spectral rank IS the output subspace dimension (a form of functional saturation), which is itself informative.

---

## F7-4 — Kronecker Structure in Attention Weight Matrices: SGD-Implicit Factorization Test (KSATT)

**F / Track B**

### Mechanism

Track B. The weight matrices W_Q, W_K, W_V ∈ R^{d_model × d_model} could, in principle, have Kronecker product structure: W ≈ A ⊗ B where A ∈ R^{p × p}, B ∈ R^{q × q}, pq = d_model = 2048. Storage: p² + q² instead of (pq)² — e.g., at p=q=√2048 ≈ 45 (not integer, but p=32, q=64 gives pq=2048): 32² + 64² = 1024 + 4096 = 5120 vs 2048² = 4M. Compression ratio: 4M/5K ≈ 800× (if exact). CF11 shows W_Q has concentrated singular values — could this be the consequence of an underlying Kronecker structure rather than arbitrary low-rank? The test: compute the Kronecker structured rank-1 approximation of W_Q (find A, B minimizing ||W_Q − A ⊗ B||_F via SVD of the "reshuffled" matrix M where M_{(i_1 i_2), (j_1 j_2)} = W_Q_{i_1 p + i_2, j_1 p + j_2}; the best rank-1 Kronecker factor pair is the outer product of the leading SVD components of M). The FIRST-PRINCIPLES question: does SGD implicitly favor Kronecker structure in attention matrices (which would make this NOT post-hoc decomposition but discovery of a learned structure)? Analogous to how CNNs learn convolutional structure, transformers might learn Kronecker structure in attention heads. Does NOT require cross-layer sharing. Does NOT require MLP rank reduction.

### Residency arithmetic

If W_Q ≈ Σ_r A_r ⊗ B_r at r=4 Kronecker factors (r × (p² + q²) = 4 × (32² + 64²) ≈ 20K params vs 4M): compression 200× for the W_Q weight if Frobenius error ≤ 5%. This is very optimistic; even r=64 Kronecker factors (64 × 5120 = 328K params) gives 12× compression if the approximation holds at ΔNLL ≤ 0.5 nats. At 70B scale: attention matrices ≈ 14 GB (W_Q + W_K + W_V + W_O); 12× Kronecker compression = ~1.2 GB for attention. This would be the most aggressive attention compression yet if preconditions hold.

### Novelty gloss

Closest kill: none on the kill list explicitly targets Kronecker structure of attention weights. Closest published: KRONECKER PRODUCT REGRESSION and Monarch matrices (arXiv:2204.00595) propose Kronecker-structured weight matrices at training time. Kaleidoscope (arXiv:2112.00029) learns butterfly/Kronecker factorizations with retraining. This is a POST-TRAINING test — does a Kronecker structure ALREADY EXIST in trained Qwen3 attention weights (not imposed, but measured). If yes, it's a latent structure SGD converged to; if no, the result closes this avenue. No published paper tests for implicit Kronecker structure in trained LLM attention weights.

### Smallest experiment

**Claim**: The best rank-1 Kronecker approximation of W_Q[14, head=0] (128×2048 matrix, reshaped as 32×32 × 64×64 = the Kronecker "reshuffling" at p=32, q=64) captures ≥ 50% of Frobenius norm.

Measurement: load W_Q for layer 14, head 0. Reshape to (32×64) × (32×64) = 2048×2048. Compute "matricization" M[i_A×j_A, i_B×j_B] = W_Q[i_A×p+i_B, j_A×q+j_B] for p=32, q=64. Compute leading SVD component of M (one singular value and vectors). Report ||rank-1 Kronecker approx||_F / ||W_Q||_F.

Runtime: ~3 min (one 2048×2048 SVD).

Go threshold: ≥ 50% Frobenius captured by rank-1 Kronecker factor → structure exists; sweep r and measure ΔNLL at r=4,16,64.
No-go: < 20% → W_Q has no Kronecker structure; close this class. Structural finding: attention weights are NOT factorizable in the Kronecker sense; CF11's concentrated spectrum is NOT a consequence of Kronecker structure.

### Primary risk

The choice of (p=32, q=64) partition is arbitrary; wrong partition could miss true Kronecker structure even if it exists. Mitigation: test three partition pairs (32×64, 64×32, 128×16) and take the max. Runtime still ≤ 10 min.

---

## F7-5 — W_down Column-Space Alignment with Residual Stream: Is the MLP Output Subspace Constant? (WDCA)

**F / Track B**

### Mechanism

Track B. The MLP output at layer L is W_down · silu(W_gate · h) ⊙ (W_up · h) ∈ R^{d_model}. The column space of W_down ∈ R^{d_model × d_ffn} spans the possible MLP output directions. CF8 established W_gate, W_up full-rank; W_down untested (F-WNORM run_004 proposed but not confirmed as run). Independent first-principles question about W_down: regardless of its singular value decay, does the COLUMN SPACE of W_down (its left singular vectors) align with a consistent subspace across layers? If the column spaces of W_down^L for L=1..28 are approximately equal (i.e., the MLP output always nudges the residual stream in the same d_model directions), then a joint column-space estimate P_col ∈ R^{d_model × r_col} (stacked SVD of all W_down matrices but taking LEFT singular vectors only) captures the full set of possible MLP outputs with r_col ≪ d_model. This is NOT cross-layer weight sharing (killed); it is measurement of the OUTPUT SUBSPACE of all MLPs jointly. If r_col < d_model, MLP output lives in a low-dimensional subspace of the residual stream — meaning a large fraction of residual-stream directions are NEVER modified by MLP layers. Those "MLP-inert" directions, if identifiable, could be allocated to lower-precision storage (since MLP never writes to them, quantization error there propagates only through attention). Algebraic basis: stack [W_down^1, W_down^2, ..., W_down^28]^T ∈ R^{28·d_ffn × d_model}, compute its right singular vectors (= the directions of R^{d_model} that the MLP system writes to). r_col = rank of the stacked system. NOT the same as cross-layer W_Q sharing (killed): W_Q writing subspace kills were about row spaces of W_Q; this is about column spaces of W_down.

### Residency arithmetic

Payoff is not direct residency but mixed-precision assignment: if MLP-inert dimensions of the residual stream (those not in col(W_down)) can be stored at INT2 instead of BF16, the residency benefit applies to W_Q, W_K projections from those dimensions. If r_col = 1200 of d_model=2048 (≈58% coverage), then 848 residual dimensions are MLP-inert. The W_Q rows corresponding to those 848 dimensions are projecting a signal that MLP never sets — calibration-stable, lower-entropy, lower-bit candidates. Mixed-precision assignment: 848/2048 ≈ 41% of W_Q rows at INT2 instead of BF16 would save 41% × 8 MB/layer × 28 layers ≈ 92 MB. Modest. The structural finding (whether r_col < d_model) is more valuable than the direct residency gain — it would be a new CF confirming that MLP outputs live in a strict subspace.

### Novelty gloss

Closest kill: v2-CHEAP-TEST-001 (cross-layer W_Q row-space sharing) — that was about W_Q rows; this is about W_down columns (left singular vectors of the MLP output matrix). Different object, different question. Cross-layer W_down column-space sharing hasn't been tested or killed. Closest published: "finding the intrinsic subspace of MLP outputs" doesn't appear in the LLM compression literature I know of. The standard approach treats W_down independently per layer; measuring the joint output subspace across all layers is a new structural characterization.

### Smallest experiment

**Claim**: The stacked W_down matrix [W_down^1 ... W_down^28]^T has var@1600 (of the 2048-dimensional right singular value space) ≥ 0.99 — i.e., MLP outputs live in ≤ 1600 of 2048 residual dimensions.

Measurement: load all 28 W_down matrices (2048 × 6144 each), stack transposed as 28×6144 × 2048 (or equivalently treat each as a batch of d_ffn=6144 column vectors in R^{2048}; compute covariance of all 28×6144 column vectors). PCA of the combined 28×6144 = 172K column vectors in R^{2048}. Record var@K for K ∈ {1024, 1400, 1600, 1800, 2048}.

Runtime: ~15 min (covariance of 172K × 2048 matrix, one PCA).

Go threshold: var@1600 ≥ 0.99 → MLP output subspace ≤ 1600-dimensional; proceed to mixed-precision assignment based on MLP-inert directions.
No-go: var@2048 required → W_down column space is full-rank; MLP writes to all residual directions. Structural finding: the residual stream has no MLP-inert subspace. Closes this class.

### Primary risk

If MLP output subspace fills R^{d_model}, this is a structural null result — but it definitively answers whether "MLP-free subspace" compression is possible. Mitigation: var@K computation at 15 min is cheap relative to the information value.

---

## F7-6 — Layer-1 Gate Folding as a Verified Algebraic Identity: Extracting the CF6 Anomaly (LGFA) [FREE SWING]

**F / Track A** [FREE SWING]

### Mechanism

Track A. CF6 (SDZC) found that globally only 1.5% of Qwen3-1.7B SwiGLU neurons are foldable (gate variance < threshold), BUT layer 1 alone has 36% foldable neurons. This is a localized anomaly, but it is measured and verified. The free-swing: exploit ONLY layer 1's 36% gate near-constancy as an algebraic fold, even though it doesn't scale globally. For layer 1: the 36% of neurons where silu(W_gate[1, i, :] · h) ≈ c_i (a per-neuron constant over calibration, measured variance < threshold): the SwiGLU output for those neurons is c_i · W_up[1, i, :] · h. The fold is exact in the limit: replace [W_gate[1] rows i that are near-constant] with a scalar c_i absorbed into W_up[1], eliminating those rows of W_gate entirely for layer 1. This folds W_gate row i and W_up row i into a single rescaled W_up row: W_up_folded[1, i, :] = c_i · W_up[1, i, :]. Reduction: 36% of layer 1's W_gate rows eliminated = 0.36 × 6144 = 2213 rows of W_gate[1] (each 2048 floats × 2 bytes = 4 KB/row) → 8.7 MB saved from W_gate[1] alone. Qwen3-1.7B total W_gate[1] = 6144 × 2048 × 2 bytes = 24 MB; save 36% = 8.7 MB. On the total model (3.4 GB), this is 0.25% — tiny in isolation. But this is a VERIFIED algebraic identity on a SPECIFIC object (layer 1) — it is provably lossless on the calibration distribution for those neurons, not an approximation. The FREE SWING value: it produces a structural finding about the nature of layer-1 specialization (early layers develop near-constant gate outputs for specific neuron subsets), which could motivate a layer-1-specific architecture variant that reduces W_gate capacity at layer 1 from the start.

### Residency arithmetic

Layer 1 savings: 8.7 MB (as above). Absolute residency: negligible (0.25% of Qwen3-1.7B). The arithmetic is honest — this is a micro-optimization. Its value is the structural finding and the verified algebraic identity, not the residency gain. Stacked across all layers at 1.5% global: additional 0.015 × 28 layers × 24 MB = 10 MB. Still small. NOT a path to 70B DRAM residence alone; frames as a structurally verified free saving that stacks with larger mechanisms.

### Novelty gloss

Closest kill: SDZC (run_004 KILL: "as a global compression scheme — KILLED"). The kill is explicit: "Layer-1 anomaly (36% foldable) is interesting but isolated; can't anchor 70B deployment." This proposal does NOT anchor 70B deployment — it proposes extracting ONLY the layer-1 saving as a verified algebraic identity, which the kill explicitly left open ("Layer-1 specific gate folding" LCAB was deferred in Track A R2). Closest published: AERO globally folds W_gate (requires retraining). This is a layer-1-specific post-training fold of only the verified-near-constant neurons — no retraining, exact on calibration, narrow scope.

### Smallest experiment

**Claim**: For Qwen3-1.7B layer 1, replacing the 36% near-constant W_gate neurons with their calibration-mean values (folding c_i into W_up_i) gives ΔNLL ≤ +0.05 nats.

Measurement: (1) identify near-constant neurons in layer 1 using CF6's threshold. (2) For each identified neuron i, compute c_i = mean(silu(W_gate[1, i, :] · h_t)) over 200 calibration tokens. (3) Set W_up_folded[1, i, :] = c_i × W_up[1, i, :]; zero out W_gate[1, i, :] (or remove from computation). (4) Measure NLL on 500-token WikiText-2 slice.

Runtime: ~10 min (200 calibration forward passes + fold + 500-token eval).

Go threshold: ΔNLL ≤ +0.05 nats → algebraic identity holds; the layer-1 anomaly is exploitable.
No-go: ΔNLL > 0.5 nats → the "near-constant" threshold used in CF6 was too loose; neurons classified as near-constant actually contribute variable outputs. Structural finding: CF6's threshold underestimates gate variability; update the SDZC threshold for layer-1 neurons.

### Primary risk

The ΔNLL threshold may not be met because even near-constant neurons produce token-dependent variance that compounds across the forward pass. However, SDZC measured gate variance directly; if that measurement is correct, the fold should be lossless by construction. Mitigation: if ΔNLL > 0.05, report the actual calibration std of the "foldable" neurons — this either confirms CF6 or refines the threshold.

---

## Convergence handles

1. **W_O left-singular / W_V right-singular principal angle** (F7-1; first measurement of V+O spectral coupling — feeds into any Reach cascade targeting attention-weight residency)
2. **Attention output subspace intrinsic dimension r_out** (F7-3; if r_out < CF11's K=128, opens a tighter W_Q compression bound than the current confirmed threshold)
3. **Participation ratio of attention score covariance** (F7-2; first-principles KV compression bound; convergence signal with any KV-side idea in C/U orientations)
4. **Kronecker structure fraction of W_Q** (F7-4; if ≥ 50%, opens a fundamentally different compression class; convergence signal with Constraint-Alien orientations using structured arithmetic)
5. **MLP output subspace dimension r_col** (F7-5; if < d_model, opens mixed-precision assignment for MLP-inert residual dimensions across all weight matrices)
6. **Layer-1 LGFA fold ΔNLL** (F7-6; closes the CF6 anomaly exploitation question; verifies whether SDZC's threshold was calibrated correctly)
