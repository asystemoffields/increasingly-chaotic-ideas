# Stage 1 — Orientation F (First-Principles) — Run 008

**Kill acknowledgments (run 008)**:
- Cross-layer W_Q stacked SVD / shared basis / joint diagonalization — class-killed (v2-CHEAP-TEST-001).
- Arbitrary orthogonal rotation of residual stream — killed (v2-S3-R004-001; RMSNorm γ breaks O(d)).
- Within-layer Q/K joint projection (PALU, TransMLA post-training variants) — KILL directive run 008.
- Softmax × RoPE shift-invariance composition — killed (CF9; precondition fails).

No idea below touches any of these flavors.

---

## F1-WVWO-SPECTRUM — W_V / W_O Spectral Rank Characterization (VOSS)
**F / Track B**

### Mechanism
Track B compression. CF11 measured W_Q r_99/d ≈ 0.63 and W_K r_99/d ≈ 0.79. W_V and W_O have never been measured on Qwen3; they are the only untested attention weight class on the stack. First-principles argument: W_V projects the residual stream into value space; W_O projects head-concatenated values back to residual space. W_V and W_O are co-trained with W_Q and W_K on the same residual stream but serve different functional roles — W_V is not involved in position matching (no RoPE), and W_O sums across heads rather than differentiating among them. These functional differences predict distinct spectral concentration patterns. Specifically: (a) W_O aggregates over 16 heads, so its column space is partitioned into 16 head-chunks; if heads are functionally redundant (CF11 head-redundancy finding), some head-chunks of W_O carry near-zero weight, making W_O effectively block-sparse in the head axis. (b) W_V operates without position bias; its spectrum should reflect residual-stream geometry alone, potentially more concentrated than W_K (which must preserve positional diversity). The first-principles experiment: measure r_99/d for W_V and W_O at all 28 layers, decomposed both globally and per-head-chunk. The unifying object: is there a rank boundary W_Q < W_V ≤ W_K < W_O < W_gate,up ≈ 1.0 that maps cleanly to functional role (query/value/key/output/MLP)? If yes, the CF11 finding is a shadow of a deeper structure: functional role determines spectral concentration, and W_V is the next low-hanging compression target.

Does not collide with CF8 (MLP weights). Does not rely on CF13/14/15. Does not require cross-layer structure (class-killed). Each layer's W_V and W_O are analyzed independently.

### Residency arithmetic
Qwen3-1.7B: d=2048, num_heads=16, head_dim=128. W_V: 2048×2048 = 4M params × 2 bytes = 8 MB/layer × 28 = 224 MB. W_O: 2048×2048 × 2 bytes = 224 MB. Total V+O: 448 MB.

Hypothesis: W_V r_99/d ≈ 0.55 (below W_K's 0.79, consistent with no-RoPE functional role). At r_99=512, safe truncation matching W_Q's K=512 ΔNLL=+0.20 threshold. W_V at K=512: factor storage 2048×512 + 512×2048 = 2M params × 2 bytes = 4 MB/layer. Savings: 4 MB/layer × 28 = 112 MB on W_V alone.

W_O hypothesis: block-sparse per head-chunk, with bottom-M head-chunks carrying <5% of Frobenius mass (consistent with CF11 head-redundancy). If 4 of 16 head-chunks are eliminable: 4/16 × 224 MB = 56 MB additional savings. Total V+O savings: ~168 MB on Qwen3-1.7B.

Scaling to 70B (32 layers, d=8192, 64 heads): W_V at 50% rank reduction: 64 heads × (8192² × 2 bytes) × 0.5 = ~4.3 GB → ~2.1 GB for W_V alone. At DRAM 11.5 GB/s, 2.1 GB loads in 182 ms/token — 91 ms saved, meaningful at ≥3 tok/s target.

Quality cost: estimating ΔNLL ≤ +0.3 nats based on W_K analogy; actual number unknown until measured.

### Novelty gloss
No W_V/W_O rank measurement on Qwen3 appears in the kill list or prior runs (runs 001–005 measured W_Q and W_K only). The CF11 finding explicitly notes "W_V and W_O spectra: untested; cheap parallel measurement (5-10 min each)" as open. Closest published: MLA (DeepSeek-V2) trains jointly; post-training attention-weight compression via SVD appears in A3 (arXiv:2505.12942) but targets Q+K only. The functional-role → spectral-concentration derivation (no-RoPE predicts higher W_V concentration than W_K) is a falsifiable first-principles prediction absent from published literature.

### Smallest experiment
**Claim**: W_V r_99/d < W_K r_99/d at the majority of layers in Qwen3-1.7B.

Measurement: load W_V and W_O for all 28 layers (bf16); compute top-512 SVD for each; compute r_99 (rank at which cumulative σ² ≥ 0.99 Σσ²); compare W_V vs W_K (already known from CF11: W_K r_99 ≈ 0.79d).

Runtime: ~15 min (28 × 2 SVDs at 2048×2048, parallelizable in PyTorch on CPU).

**Go threshold**: W_V r_99/d < 0.70 on ≥15 of 28 layers → W_V is more concentrated than W_K; proceed to NLL sweep at K=512 and K=256.
**No-go**: W_V r_99/d ≈ W_K r_99/d across layers → functional-role hypothesis fails; value projection has similar spectral structure to key projection despite no RoPE. Structural finding: RoPE presence/absence does not determine spectral concentration; gradient dynamics dominate.

### Primary risk
W_V and W_O may be trained to be full-rank if gradient flow through attention output is dense (analogous to tied lm_head being full-rank). Mitigation: the 15-min experiment directly falsifies this; if full-rank, it extends CF12's "tied gradient makes full-rank" pattern to the attention computation path.

---

## F2-UNTIED-LMHEAD — Untied lm_head Spectral Test on Qwen3-8B (ULHS)
**F / Track B**

### Mechanism
Track B. CF12 measured tied lm_head (Qwen3-1.7B) as full-rank (r_99=1992/2048, catastrophic ΔNLL=+19.96 at K=1024). The mechanism for full-rank was identified: gradient flows through TWO paths (embedding lookup AND output projection), keeping every direction load-bearing. For untied configurations (Qwen3-8B+, `tie_word_embeddings=False`), only the output-projection gradient path exists. The Godey & Artzi (lm_head gradient-bottleneck) prediction: untied lm_head develops concentrated spectrum because gradient only backpropagates through the output projection path — the embedding matrix and lm_head diverge during training, with lm_head developing a low-rank effective structure reflecting the model's output vocabulary distribution. The first-principles derivation: lm_head has shape 151936×4096 (Qwen3-8B); its row space is the vocabulary embedding space; if semantic clusters of tokens share output representations, the effective rank of lm_head should be substantially below 4096 (analogous to the "vocabulary spectrum concentrates on semantic axes" prediction from information-theoretic arguments). The algebraic identity: a rank-r lm_head allows the final projection to be decomposed as lm_head_U (151936×r) × lm_head_V (r×4096), with lm_head_V absorbing into the final layer norm; inference cost drops from 151936×4096 to (151936×r + r×4096), a ratio of 2r/4096 — at r=512, a 4× reduction in the lm_head GEMV.

Does not collide with CF12 (CF12 is specifically the tied-embedding case). Does not require cross-layer structure. Does not use CF13/14/15.

### Residency arithmetic
Qwen3-8B: lm_head weight = 151936 × 4096 × 2 bytes = 1.19 GB (tied config: 0 additional; untied: 1.19 GB incremental). At r=512: U (151936×512) + V (512×4096) = 77.9M + 2.1M = 80M params × 2 bytes = 156 MB. Savings: 1.19 GB → 0.156 GB = 1.03 GB saved. This is ~19% of 70B IQ4_XS = 5.35 GiB budget — large. At DRAM 11.5 GB/s, lm_head at full size takes 103 ms/token; at r=512: 13.5 ms/token. Tok/s impact: +0.8 tok/s at baseline 2 tok/s → 2.8 tok/s (if lm_head is binding; it is only binding at small sequence length or if matmul-dominated). Quality: if r=512 gives ΔNLL < 0.1 nats, this is a clean win.

### Novelty gloss
CF12 explicitly identifies "untied lm_head spectrum (Qwen3-8B has tie_word_embeddings=False) remains open; the Godey & Artzi prediction may hold there" as the open case. No prior run tested this. arXiv:2510.24966 (logit low-rank theorem) supports the prediction at the function level; this measures it at the weight level. Closest published: VPTQ (arXiv:2409.17066) and related codebook methods quantize lm_head but do not explicitly measure low-rank structure. The Godey & Artzi mechanism for lm_head spectrum concentration is explicitly not tested in the v2 ladder; the measurement is the contribution. Not pre-empted by LHQD (which killed tied config).

### Smallest experiment
**Claim**: in Qwen3-8B (untied config), lm_head singular value decay achieves 99% variance at r ≤ 1024 (r/d ≤ 0.25).

Measurement: blockwise streaming SVD of lm_head.weight (151936×4096) — materialize 4096×4096 Gram matrix (W W^T on the 4096-dim side), eigen-decompose it, derive singular values. No 151936×4096 full matrix needed in memory.

Runtime: ~25 min on Ryzen 5 7530U (Gram matrix computation: 151936 × 4096 × 4096 float32 ops ≈ 2.5 TFLOP, but streaming rowwise keeps peak RAM at 4096² × 4 bytes = 64 MB).

**Go threshold**: r_99/d < 0.25 AND reconstruction ΔNLL < 0.10 nats at r=1024 → lm_head is the single largest compressible weight in the untied 70B+8B family; proceed to deployment integration.
**No-go**: r_99/d > 0.60 → Godey & Artzi prediction fails for untied config; lm_head is also full-rank regardless of gradient path. Structural finding: lm_head spectrum is determined by vocabulary geometry (dense token clusters) not by gradient path count. Extends CF12's full-rank pattern to the untied case.

### Primary risk
Qwen3-8B is 12–14 GiB in bf16; loading only lm_head (1.19 GB) is feasible but requires careful slice loading from GGUF/safetensors. Streaming Gram computation keeps RAM below 2 GB. Mitigation: use safetensors partial-load API (already used in the v2 ladder for W_gate slices in Round 3).

---

## F3-WO-HEADBLOCK-RANK — W_O Per-Head-Chunk Rank Decomposition (OHCR)
**F / Track B**

### Mechanism
Track B compression. W_O ∈ R^{d×(h·head_dim)} = R^{2048×2048} aggregates outputs from h=16 heads. Each head-chunk of W_O is a 2048×128 sub-matrix (the "output projection" for head i). CF11 showed that 16 heads collectively span a ~128-dim subspace of the residual stream — 16:1 head-to-subspace ratio. If the head-redundancy is non-uniform (some heads have low-rank W_O head-chunks because those heads are functionally redundant), then W_O admits a structured rank reduction that is NOT per-layer global SVD (which would need to respect the head-chunk structure) but IS per-head-chunk SVD within each head's 2048×128 projection. First-principles argument: a head-chunk W_O^{(i)} ∈ R^{2048×128} has r_99 ≤ 128 trivially (shape bound). The non-trivial question is whether r_99 << 128 — i.e., whether the 128-dimensional value subspace that head i writes to the residual stream has a low-rank effective contribution. The algebraic identity: if r_99(W_O^{(i)}) = k < 128, then W_O^{(i)} = U_i S_i V_i^T where V_i is k×128 — the head's value vectors live in a k-dimensional subspace. The compression: replace each W_V^{(i)} (128 cols × 2048 rows) with W_V^{(i)} V_i (128 × k), and W_O^{(i)} with W_O^{(i)} V_i^T expanded-back — algebraic identity on the composition W_O^{(i)} W_V^{(i)}: both are compressed simultaneously via the shared V_i^T. This is a within-head W_V / W_O joint low-rank factorization that relies only on W_O^{(i)}'s spectrum (not W_Q or W_K), so it does not collide with the Q/K joint-projection kill.

Does not require retraining. Does not use cross-layer structure. Does not use W_Q/W_K. Does not rely on CF13/14/15.

### Residency arithmetic
At Qwen3-1.7B, h=16 heads, head_dim=128. W_O: 2048×2048 = 8 MB/layer bf16. W_V: 2048×2048 = 8 MB/layer. Per-head-chunk W_O^{(i)}: 2048×128 = 256K params. At r_99(W_O^{(i)}) = k_i:
- Storage per head: W_O^{(i)} U_i (2048×k_i) + W_V^{(i)} V_i (128×k_i) = (2048 + 128) × k_i × 2 bytes.
- At k_i = 64 (50% of head_dim=128): (2176 × 64 × 2) bytes = 278 KB per head vs (2048×128 + 2048×128) × 2 bytes = 1 MB for the pair. Ratio: 3.6× per head if k_i=64.
- If half the heads have k_i=64 and half have k_i=128 (full): average 2.8× on those pairs. Net W_V+W_O savings over 28 layers: ~(1/2) × 2.8× × 448 MB × 2 ≈ 160 MB.
- Quality: per-head compression is much softer than per-layer global compression; each head independently compressed. Estimated ΔNLL < 0.2 nats at k_i ≥ 64 (analogous to per-head K=96 in CF11 borderline at +0.94 nats — but that was W_Q; W_O may differ).

### Novelty gloss
Distinct from CF11 (W_Q global low-rank): CF11 is about query projection, head-sharing at the global level. This is per-head-chunk output-projection rank, independent of query structure. Kill list has no per-head W_O rank entry. Closest published: GQA (grouped-query attention) reduces W_K/W_V heads but not W_O and relies on training-time design. MLA compresses W_Q+W_K+W_V jointly at training time. Post-training per-head W_O chunk decomposition is not in the published literature for Qwen3-class models. The joint W_V / W_O compression via shared V_i^T (one SVD, two matrices compressed simultaneously) is the algebraic identity that makes this in-orientation for F.

### Smallest experiment
**Claim**: in Qwen3-1.7B, at least one head in at least one layer has r_99(W_O^{(i)}) ≤ 64 (< head_dim/2).

Measurement: load W_O for layer 14; reshape into 16 head-chunks of 2048×128; compute full SVD for each (128×128 Gram, trivial); report r_99 per head.

Runtime: 5 min (16 small SVDs per layer, one layer).

**Go threshold**: ≥4 heads in layer 14 with r_99 ≤ 64 → per-head W_O compression is real; proceed to joint W_V/W_O factorization and NLL sweep.
**No-go**: all heads have r_99 ≥ 100 → W_O head-chunks are full-rank at the 128-dim shape limit. Structural finding: W_O is functionally full-rank per head, extending CF12's full-rank pattern to the attention output path.

### Primary risk
Each head's W_O^{(i)} is trained to write a distinct direction into the residual stream; if heads specialize, W_O^{(i)} may be effectively rank-1 (one direction per head) or full-128 (all head dimensions used). Either extreme gives a clean result; partial compression is the nuanced middle case. Mitigation: the 5-min experiment covers all cases.

---

## F4-SVAL-CONSERVED — Frobenius-Mass Conservation as Gauge-Invariant Rank Predictor (FMCP)
**F / Track B**

### Mechanism
Track B compression. Consider any weight matrix W and a scale transformation: W → D_L W D_R^{-1} where D_L, D_R are positive-diagonal matrices (the SmoothQuant family of scale absorptions). The Frobenius norm changes under this transformation; singular values change; r_99 changes. But the **log-det of the Gram matrix** (Σ_i log σ_i²) is invariant under permutations and scaling of D_R (diagonal on the right) — it equals the sum of log squared singular values, which transforms as log det(D_L^2) + log det(W W^T). This is NOT invariant. However, the **rank** itself is strictly invariant: rank(D_L W D_R^{-1}) = rank(W) for invertible diagonals. The first-principles observation: there exists a canonical scale gauge (D_L, D_R) for each W that MINIMIZES spectral entropy H_s — i.e., maximizes the concentration of singular value mass. This is the "tightest-rank" gauge. The derivation: by matrix scaling theory (Sinkhorn-Knopp for positive matrices; the analogous problem for SVD concentration is related to the optimal preconditioner for iterative solvers). The question for Qwen3: when W_Q is expressed in the gauge where input and output channels are unit-variance (absorbing RMSNorm γ on input and W_O's column norms on output), does r_99/d drop below CF11's measured 0.63? This extends run_005's F6-GAUGE-NORM idea but provides the algebraic foundation: the optimal-concentration gauge is derivable via alternating normalization (Sinkhorn), not guessed as the RMSNorm-γ absorption alone.

Algebraic identity: for any W, min_{D_L, D_R invertible diagonal} H_s(D_L W D_R^{-1}) is achieved by the Sinkhorn-balanced form (equal row and column norms). If Sinkhorn-balanced W_Q has strictly lower H_s than CF11's measured W_Q, there is a gauge where W_Q is "more compressible" than CF11 measures, and safe truncation rank is lower.

Does not use cross-layer structure. Does not require retraining. Does not collide with v2-S3-R004-001 (arbitrary orthogonal rotation killed — this uses diagonal scaling, not rotation; diagonal scaling does NOT change rank, does NOT break RMSNorm because the scale can be pre-absorbed before the RMSNorm call). Uses only numpy/scipy Sinkhorn or manual alternating row/col normalization.

### Residency arithmetic
If Sinkhorn gauge reduces W_Q effective rank from r_99/d=0.63 to r_99/d=0.45, then safe K drops from 512 to 368. Per-layer W_Q storage: at K=368 vs K=512: (2048×368 + 368×2048) × 2 bytes = 3.0 MB vs (2048×512 + 512×2048) × 2 bytes = 4.2 MB. Saving: 1.2 MB/layer × 28 layers = 33.6 MB on Qwen3-1.7B. At 70B scale (8192²): proportional ~1.3 GB additional attention savings beyond CF11 baseline. Marginal on its own; significant in composition with VOSS (F1) if W_V also compresses further under Sinkhorn gauge.

Quality: Sinkhorn balance is a lossless coordinate change (exactly equivalent — same forward pass output after absorbing the scales into adjacent operations). The compression cost is the truncation at lower K, not the balancing itself.

### Novelty gloss
Run_005 F6-GAUGE-NORM proposed γ absorption specifically. This derives the optimal-concentration gauge from first principles (Sinkhorn balance minimizes H_s). Not in kill list. Closest published: SmoothQuant (specific γ absorption for quantization); Preconditioned SVD (numerical methods literature). The connection SmoothQuant ← Sinkhorn balance ← optimal H_s compression guide is not drawn in published LLM compression work. CF9 check: Sinkhorn converges on any positive matrix (the balancing theorem holds for W W^T > 0, which is satisfied when W is full-rank — which CF11 confirms for W_Q within each layer). Precondition verified.

### Smallest experiment
**Claim**: Sinkhorn-balanced W_Q at layer 14 has H_s(W_Q_balanced) / H_s(W_Q_original) < 0.90.

Measurement: load W_Q layer 14; run 50 iterations of alternating row-normalization / column-normalization (Sinkhorn on W_Q W_Q^T); compute H_s before and after; check ratio.

Runtime: 8 min (50 Sinkhorn iterations on 2048×2048, each O(d²); H_s via top-256 SVD).

**Go threshold**: ratio < 0.90 → Sinkhorn balancing concentrates W_Q spectrum; run K-sweep on balanced W_Q_balanced to find true safe K.
**No-go**: ratio ≥ 0.98 → W_Q is already near-Sinkhorn-balanced from training (plausible given weight decay regularization). Structural finding: SGD with weight decay spontaneously produces near-Sinkhorn-balanced weight matrices — a connection between optimization and spectral concentration.

### Primary risk
CF10 caution: Sinkhorn balancing is calibration-free (no calibration data, only weight statistics), so no ill-conditioning risk. The risk is that Sinkhorn converges to a form that reduces H_s but does NOT reduce the NLL-safe truncation rank — because NLL cost depends on input activation alignment with singular vectors, not H_s alone. Mitigation: if H_s decreases, always run NLL sweep at the predicted lower K before claiming savings.

---

## F5-WDOWN-SPECTRUM — W_down Spectral Rank Measurement (WDSR) [FREE SWING]
**F / Track B** [FREE SWING]

### Mechanism
Track B. CF8 (generalized): MLP W_gate and W_up are full-rank (r_99/d ≈ 1.0). W_down ∈ R^{d×d_int} = R^{2048×6144} maps from intermediate dimension back to residual space. W_down has a different functional role and a DIFFERENT shape (d × d_int, not d × d square): its left singular vectors are residual-stream directions; its right singular vectors are intermediate-dimension directions. The first-principles argument for W_down potentially being more compressible than W_gate/W_up: W_down sums the post-SwiGLU activations across d_int=6144 dimensions back to d=2048. By the low-effective-rank of the post-SwiGLU activation distribution (if the SwiGLU output concentrates on a low-dimensional manifold), W_down's effective rank in practice is bounded by the intrinsic dimension of its INPUT distribution. This is not the same as W_down's weight matrix rank — it is the effective rank in the sense that W_down restricted to the actual activation subspace has lower rank than W_down applied to the full d_int-dimensional space. The measurement needed: compute the SVD of W_down (2048×6144); separately, compute the covariance of the post-SwiGLU activations (6144×6144 empirical); ask whether the top K right singular vectors of W_down align with the top K principal components of the activation covariance. If they DO align, then W_down's "effective rank" on the calibration distribution is lower than its algebraic rank — and a calibration-derived W_down projection is justified.

This is NOT the CF7/CF8 kill (which killed no-retraining W_gate/W_up rank reduction). W_down is a different matrix with a different functional justification. The claim is about calibration-distribution effective rank (not algebraic rank), and any compression derived from it is calibration-conditioned.

CF10 precondition: calibration fitting here is a projection (choosing which K right singular vectors of W_down to keep), not a regression. No fitting of 2M parameters — just selecting K out of 6144 directions. CF10 does not apply.

### Residency arithmetic
Qwen3-1.7B W_down: 2048×6144 = 12.6M params × 2 bytes = 25.2 MB/layer × 28 layers = 705 MB. This is the largest single MLP matrix by bytes (W_gate and W_up are 2048×6144 = same, but W_down is 6144→2048 shaped equivalently). At K=3072 (50% right-side rank): factor storage = 2048×3072 + 3072×6144 = 6.3M + 18.9M = 25.2M params — same (factored form is not smaller at 50%). Savings only appear at K << d_int. At K=1024: (2048×1024 + 1024×6144) × 2 bytes = 2.1M + 6.3M = 8.4M params = 16.8 MB/layer vs 25.2 MB — 1.5× savings. On 70B: W_down savings at K=1024 could save ~2.8 GB. But the quality cost is likely catastrophic (analogous to W_gate rank at K=1024 → ΔPPL +13.7). The smallest experiment determines whether calibration-distribution effective rank enables lower-K truncation than pure algebraic rank allows.

### Novelty gloss
No prior run has measured W_down spectrum directly. The kill list kills W_gate and W_up rank reduction; W_down is absent from the kill list explicitly. The calibration-distribution effective rank argument (alignment of W_down right SVs with activation principal components) is a novel angle not in prior runs. Closest prior art: AERO (arXiv:2410.13060) folds W_gate into W_up but touches W_down only after the fold, keeping it full. The question here is prior to any fold: what is W_down's algebraic rank structure and does it differ from W_gate/W_up?

[FREE SWING note: this idea does not connect to CF1–CF12 as strongly as F1–F4. It is grounded in the existing stack (W_down is on disk, SVD is computable in ~20 min, calibration activations are obtainable in 30 min) but the effectiveness argument requires the activation-covariance alignment to hold, which is empirically unverified. Flagging as FREE SWING to protect against Stage 2 penalizing it for weaker CF anchoring.]

### Smallest experiment
**Claim**: in Qwen3-1.7B layer 14, the top-1024 right singular vectors of W_down capture ≥ 90% of the variance of the post-SwiGLU activation distribution (measured on 500 calibration tokens).

Measurement: (a) compute top-1024 SVD of W_down layer 14 (2048×6144 matrix, 20 min); (b) run 500-token forward pass, collect post-SwiGLU activations (shape: 500×seq_len×6144), compute covariance (6144×6144); (c) measure fraction of activation variance captured by top-1024 right singular vectors of W_down.

Runtime: ~40 min total.

**Go threshold**: ≥90% activation variance in W_down's top-1024 right SV subspace → calibration-distribution effective rank ≤ 1024; W_down compression at K=1024 preserves activations on calibration; proceed to NLL sweep.
**No-go**: <70% → W_down's weight directions and activation principal components are misaligned; no calibration-distribution shortcut. Structural finding: W_down is trained to be sensitive to all 6144 intermediate directions, not just the high-variance activation directions (possible if W_down is the "correction operator" that specifically attends to rare activations).

### Primary risk
Even if the right SV subspace captures 90% of activation variance, the NLL cost of W_down truncation may be dominated by the rare 10% (analogous to LLM.int8() outliers). Mitigation: if activation-variance condition holds (GO), immediately test NLL with and without the rare-activation correction (hold a sparse correction matrix for the top-1% activations).

---

## Convergence handles

1. **W_V r_99/d measurement** (F1; direct cascade from CF11; W_K's value 0.79 is the comparison baseline)
2. **W_O per-head-chunk r_99 distribution** (F3; whether head energy concentration from CF11 maps to W_O output-direction concentration)
3. **Untied lm_head spectrum via Gram blockwise SVD** (F2; explicit open item from CF12)
4. **Sinkhorn-balanced W spectral entropy vs raw W spectral entropy** (F4; calibration-free compression predictor; convergence signal if C orientation independently surfaces optimal preconditioned W structure)
5. **Activation-covariance / weight-SV alignment for W_down** (F5; convergence signal if U or A orientation surfaces activation-subspace-aware streaming)
