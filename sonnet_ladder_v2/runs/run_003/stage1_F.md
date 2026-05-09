# Stage 1 — Orientation F (First-Principles) — Run 003

Orientation: F — First-Principles
Run: 003 (independent cold-start; no inheritance from run_001 or run_002)

Hard kill (pre-read): v2-CHEAP-TEST-001 — cross-layer W_Q stacked SVD / shared basis / joint diagonalization. Class-dead. No variants proposed here.

---

## F1 — W_V / W_O Fused Composition Rank VOFR
**F / Track B**

### Mechanism
Track B — compression. CF11 measured W_Q (r_99/d ≈ 0.63) and W_K (r_99/d ≈ 0.79); MLP weights are full-rank (CF7/CF8); W_V and W_O have never been measured on Qwen3-1.7B. The first-principles move: the operationally relevant object is not W_V or W_O individually but their per-head composition M_h = W_O^h W_V^h (shape d_model × d_model = 2048×2048 per head) and the head-summed operator M = ∑_h M_h. The residual-stream update from attention is exactly x += M (weighted sum of values) — M is the matrix that actually touches the residual stream. SVD of M per layer is the right spectrum measurement, not per-matrix SVD of W_V or W_O. If M_L has r_99/d ≤ 0.70, store (U_L Σ_L^{1/2}) and (Σ_L^{1/2} V_L^T) separately — a two-factor storage at rank K with no retraining, no calibration fitting, purely algebraic. CF9 pre-check: M_h is a product of two real matrices; SVD is always valid; no theorem precondition to verify. Not per-head rank reduction (CF11 kills that); this is the global cross-head composition.

### Residency arithmetic
W_V per layer: 2048×2048 × 2 bytes × 28 layers = 448 MB. W_O same = 448 MB. Total W_V + W_O = 896 MB. If M_L has r_99/d ≈ 0.63 (same as W_Q by analogy with attention structure), global K=512 captures 99% variance. Factored storage: U_L (2048×512) + V_L^T (512×2048) at 2 bytes each = 2048×512×4 bytes × 28 layers = 112 MB per weight class × 2 = 224 MB. Savings: 896 MB → 224 MB = 672 MB. On Qwen3-72B (where attention is proportionally larger): ~4.5 GB → ~1.1 GB in attention V+O. At 11.5 GB/s DRAM bandwidth: 3.4 GB saved ÷ 11.5 GB/s = ~295 ms/token freed. At baseline 1.5 tok/s, that is ~+0.55 tok/s — meaningful. ΔNLL: if spectrum matches W_Q's profile, K=512 should be sub-0.30 nats (CF11 analog: W_Q at K=512 was +0.20 nats).

### Novelty gloss vs the kill list and the published landscape
Closest kill-list item: per-head W_Q rank truncation (CF11 NO-GO at K=64 per head). VOFR is global (cross-head sum), not per-head. Closest published method: MLA (DeepSeek-V2) decomposes K and V into a shared low-rank projection at training time; VOFR is post-training, no retraining, and targets the fused M = ∑_h W_O^h W_V^h — a different mathematical object (residual-stream update operator vs per-token KV projections). No published paper measures the spectrum of ∑_h W_O^h W_V^h on any Qwen3-family or comparable GQA architecture.

### Smallest experiment
**Claim**: The fused operator M_L = ∑_h W_O^h W_V^h (shape 2048×2048) has r_99/d ≤ 0.75 in at least 20 of 28 layers of Qwen3-1.7B-Base.
**Protocol**: load bf16 model, extract W_V^h and W_O^h per layer per head (16 heads each), compute M_L = ∑_{h=1}^{16} W_O^h W_V^h, run SVD (torch.linalg.svd on 2048×2048), measure r_99/d and cumvar@K=512 per layer. Runtime: ~25 min on Ryzen 5 7530U.
**Go threshold**: r_99/d ≤ 0.75 in ≥ 20 layers AND reconstruct at K=512, measure ΔNLL ≤ +0.35 nats on 512-token WikiText-2 held-out.
**NO-GO finding**: extends CF8 — W_V/W_O composition is also full-rank; the entire individual weight matrix class in Qwen3 is full-rank, only the cross-head summed Q/K operators are compressible. High-value structural boundary even on failure.

### Primary risk
Summing 16 heads may smooth out per-head concentration: individual head V×O products could have low-rank structure that cancels when summed. Mitigation: measure both ∑_h M_h AND a single representative M_h (h=0) as a sanity check; if per-head M_h is more concentrated than the sum, decompose per-head (which is not per-head per-projection, so CF11's per-head kill doesn't apply here).

---

## F2 — RMSNorm Gain Absorption Spectrum Shift RGAS
**F / Track A**

### Mechanism
Track A — arch-transposition. Every attention block in Qwen3 reads from the residual stream through an RMSNorm: h_normed = h / ‖h‖ × g, where g ∈ R^d_model is a learned per-channel gain vector. All downstream projections (W_Q, W_K, W_V, W_gate, W_up) left-multiply h_normed, so they effectively act on (h × diag(g) / ‖h‖). The gauge identity: W_proj h_normed = (W_proj diag(g)) (h / ‖h‖) = W_proj_gauged (h / ‖h‖). Absorbing g into W_proj is exact and requires no retraining — it is a coordinate change on the residual stream at one layer. The first-principles hypothesis: SGD may have distributed scale information between g and W — if g has high per-channel variance (some channels scaled by 5×, others by 0.2×), then W in raw coordinates contains compensating scale factors in its rows, inflating effective rank. After absorption, W_proj_gauged = W_proj diag(g) may have a more concentrated spectrum because the scale variance now lives fully in the matrix entries rather than being shared with g. The same logic applies to MLP: W_gate_gauged = W_gate diag(g_MLP) and W_up_gauged similarly. For MLP this is almost certainly neutral (CF8 shows MLP is full-rank regardless), but for W_Q (r_99/d ≈ 0.63) there may be measurable improvement. The mechanism: (1) fold g into W_Q and W_K per layer; (2) measure r_99/d before vs after; (3) if improved, the gauged coordinates are the better basis for further compression.

### Residency arithmetic
Direct: eliminating the RMSNorm multiply saves 28 × 4 (blocks per layer) × d_model × 2 bytes = 28 × 4 × 2048 × 2 ≈ 460 KB — negligible. The payoff is downstream: if gauge absorption reduces r_99/d of W_Q from 0.63 to 0.50, global K for ≤+0.20 nats drops from 512 to ~300, saving 40% of W_Q storage. On current baseline: W_Q 28 layers × 2048² × 2 bytes = 448 MB; at K=300 vs K=512 the saving is (1 - 300/2048)/(1 - 512/2048) ≈ modest. The real value: feeds VOFR (F1) and any subsequent attention compression — a pre-processing step that makes the chain work at lower K.

### Novelty gloss vs the kill list and the published landscape
Closest kill: CRANK (Track A R2, killed as SmoothQuant prior art). SmoothQuant migrates scale from activations to weights for INT8 quantization of activations; RGAS folds the RMSNorm gain into weight matrices for the purpose of testing whether the weight spectrum concentrates — an entirely different motivation and measurement target. SmoothQuant does not measure SVD spectra of the resulting weight matrices. The spectral-concentration hypothesis after gain absorption has no published treatment.

### Smallest experiment
**Claim**: Median r_99/d of W_Q_gauged = W_Q diag(g_RMSNorm) across 28 layers is ≤ 0.55 (vs CF11 baseline of 0.63 ungauged).
**Protocol**: load model, extract W_Q and g per layer (each g is shape 2048), compute W_Q_gauged = W_Q @ diag(g) (equivalent to scaling columns of W_Q by g), run SVD per layer, measure r_99/d. Pre-check: compute std(g_L) per layer — if max std(g_L) < 0.05 across all layers, abort (gain is near-constant, fold is a no-op). Runtime: ~20 min including pre-check.
**Go threshold**: median r_99/d ≤ 0.55 AND at least one layer with r_99/d ≤ 0.45.
**NO-GO finding**: gain absorption does not reshape the attention weight spectrum — the scale variance in g_L is orthogonal to the singular structure of W_Q. This constrains the family of "absorb normalization parameters" moves for spectrum shaping, and confirms CF11's 0.63 is the floor in any linear gauge.

### Primary risk
g_L may have high variance in a subspace that is already well-captured by top singular vectors of W_Q, making the fold redistribute rather than concentrate energy. Mitigation: measure cumvar@K=128 (the tightest compression point from CF11) before and after fold — if cumvar@128 improves by ≥ 3 percentage points, the fold is useful even if r_99/d improvement is modest.

---

## F3 — Softmax Row-Shift Calibration SRSC
**F / Track A**

### Mechanism
Track A — arch-transposition. Softmax is exactly invariant to additive shifts of its input rows: softmax(z + c·1) = softmax(z) for any scalar c per row. For attention, each row of QK^T / √d corresponds to one query position attending over key positions. If there exists a per-query-position scalar offset that is predictable from the query (not the keys), it can be precomputed and stripped — the softmax output is unchanged. The first-principles question: do trained Qwen3 query projections produce a persistent per-channel mean across diverse inputs? If W_Q has a non-zero row mean μ_row^h (per head), then Q·K^T = (W_Q x)(W_K y)^T contributes a persistent term μ_row^h E[W_K y]^T to all rows — a predictable per-head per-position offset in the logit matrix. Absorbing this into a stored per-head scalar bias eliminates it from Q's dynamic range. The resulting Q_residual = W_Q x - μ_row^h has lower dynamic range than W_Q x if the mean is substantial. Lower dynamic range → tighter INT8/INT4 quantization of Q activations at runtime → KV-path bandwidth reduction. The algebraic identity is exact: strip, compute, restore — no approximation.

### Residency arithmetic
The offset is tiny to store: 28 layers × 16 heads × 1 scalar × 4 bytes = 1.8 KB. The payoff is activation quantization quality. If Q's dynamic range decreases by 15% after mean subtraction, INT8 quantization of Q (which dominates KV cache write cost at long context) has proportionally lower quantization error. On Qwen3-72B at 32K context: KV cache is ~5 GB at FP16; INT8 reduces to ~2.5 GB; mean subtraction potentially closes the quality gap enough to make INT4 Q viable at ≤ +0.15 nats additional cost vs INT8 Q. INT4 KV vs INT8 KV saves 1.25 GB at 32K context — meaningful for the memory-wall scenario. For 1.7B at short context this is irrelevant; it is a long-context mechanism.

### Novelty gloss vs the kill list and the published landscape
Closest kill: KV Temporal Differencing (R1/S2, killed because RoPE makes key deltas non-small). SRSC does not touch K or V at all — it operates on the Q activation to reduce its dynamic range before softmax, using softmax invariance rather than differencing. RoPE affects K, not Q's mean subtraction. Closest published method: PrefixQuant (arXiv:2410.05265) handles token-wise activation outliers in Q/K by routing massive-activation tokens separately; SRSC handles the persistent per-head mean offset, a different object. The systematic row-mean structure of W_Q (as opposed to token-specific outliers) has not been measured.

### Smallest experiment
**Claim**: Per-head per-layer W_Q row mean has ‖μ_row^h‖ / median(‖W_Q^h row‖) ≥ 0.05 in at least 10 of 28 layers (for at least one head per layer).
**Protocol**: extract W_Q per layer per head, compute mean of each row (each W_Q^h is 128×2048, so row-mean is a 128-dim vector μ, compute ‖μ‖ vs median row norm). Static measurement, no forward pass needed. Runtime: ~5 min.
**Go threshold**: ‖μ_row‖ / median(‖W_Q row‖) ≥ 0.05 in ≥ 10 layers.
**NO-GO finding**: W_Q rows have near-zero mean — the network learned to center Q projections, making the shift extraction a no-op. This is a negative with informational content: it implies that Q quantization error is dominated by dynamic spread, not persistent offset, which narrows the space of activation-side precision improvements.

### Primary risk
RoPE adds per-position rotation to Q after the projection; the rotation changes the mean structure per position, potentially nullifying the static W_Q row-mean argument. Mitigation: measure the RoPE-rotated Q mean per position separately and check whether it decays with distance from the first token — if it does, the static mean is real; if rotated Q has zero persistent mean, RoPE has already centered it and the mechanism fails.

---

## F4 — W_down Activation-Weighted Output Subspace WAOS
**F / Track B**

### Mechanism
Track B — compression. W_gate and W_up are full-rank (CF7/CF8, confirmed by multiple experiments). W_down (shape d_model × d_intermediate = 2048 × 6144) has never been measured. But the measurement object for W_down should not be its weight SVD — it should be its effective output subspace under calibration activations. Define A_L = silu(W_gate^L · x) ⊙ (W_up^L · x) as the post-SwiGLU activation vector. The residual-stream update is δh_L = W_down^L A_L. The matrix of updates across a calibration corpus is Delta_L = W_down^L [A_L^{(1)} ... A_L^{(N)}] (shape d_model × N). SVD of Delta_L measures not W_down's spectrum but the spectrum of W_down's actual output under calibration distributions. CF1 established W_up dominates firing rank — the top-K neurons by |W_up · x| fire; the rest contribute near-zero to A_L. If only K_eff = 300 of 6144 neurons fire substantially per token (soft sparsity from CF1/CF3), then Delta_L has effective rank ≤ K_eff × (number of distinct firing patterns). If this effective rank r_eff ≪ d_model = 2048, then W_down^L restricted to its output subspace is storable as a much smaller matrix. Crucially: the truncation is on the OUTPUT subspace of W_down, not on its weight matrix directly, so CF8's weight-rank-kill does not apply — the effective-output-rank of a full-rank matrix under concentrated activations can be much smaller than the matrix's own rank.

### Residency arithmetic
W_down: 28 layers × 2048 × 6144 × 2 bytes = 1.34 GiB. If effective output rank r_eff ≤ 512 (conservatively), compressed storage: U_L ∈ R^{2048×512} + W_down_compressed^L ∈ R^{512×6144} per layer (U_L is the output basis, W_compressed = U_L^T W_down^L has 512 rows). Storage: 28 × (2048×512 + 512×6144) × 2 bytes = 28 × (2M + 3.1M) × 2 = ~280 MB. Savings: 1.34 GiB → 0.28 GiB = 1.06 GiB. Combined with W_V/W_O compression from F1 (~672 MB), total attention+MLP-output savings ≈ 1.73 GiB on the 1.7B model. This is a meaningful fraction of the 7.28 GiB target. CF10 compliance: compression is by SVD truncation of Delta_L (no fitted parameters); no calibration-fit least-squares step, no n_params/n_samples issue.

### Novelty gloss vs the kill list and the published landscape
Closest kill: low-rank W_down weight compression (CF8 spirit). WAOS explicitly targets the calibration-weighted OUTPUT subspace, not the weight matrix itself — SVD of W_down A_L (shape d_model × N) vs SVD of W_down alone (shape d_model × d_intermediate). Closest published method: GPTQ/GPTVQ use input Hessian X^T X to weight the input dimensions of W during quantization. WAOS targets the OUTPUT directions of W_down under calibration activations — the transpose-dual problem. No paper measures the output-subspace concentration of W_down under SwiGLU-gated activations where input concentration (CF1) is already measured.

### Smallest experiment
**Claim**: The calibration-weighted output matrix Delta_L = W_down^L A_L has r_eff ≤ 512 (90% variance in ≤ 512 of 2048 directions) in at least 15 of 28 layers, where A_L is post-SwiGLU activations on 200 diverse calibration tokens.
**Protocol**: run 200-token forward pass, cache post-SwiGLU activations A_L per layer (shape 6144 × 200), compute Delta_L = W_down^L @ A_L (shape 2048×200), SVD of Delta_L, measure cumvar@K. Runtime: ~40 min (forward pass dominates).
**Go threshold**: cumvar(K=512) ≥ 0.90 in ≥ 15 of 28 layers. Follow-on if GO: measure ΔNLL at K=512 reconstruction ≤ +0.25 nats.
**NO-GO finding**: SwiGLU soft-sparsity (CF1/CF3) does NOT concentrate W_down's output subspace — the firing-pattern diversity across tokens is sufficient to fill the residual-stream dimensionality. This closes the "activation-soft-sparsity → output-subspace compression" route, even for the W_down case where the weight matrix itself need not be low-rank.

### Primary risk
200 calibration tokens may undersample the activation distribution, making Delta_L artificially low-rank due to calibration concentration (a CF10 analog for subspace measurement rather than parameter fitting). Mitigation: measure cumvar@K separately on two 100-token disjoint subsets; if cumvar(K=512) differs by > 5 percentage points between subsets, the 200-token corpus is too narrow — extend to 1000 tokens before drawing conclusions.

---

## F5 — Head Permutation Group Orbit Collapse HPGO
**F / Track A**

### Mechanism
Track A — arch-transposition. The attention output is ∑_h softmax(Q_h K_h^T / √d) V_h, a sum over heads. This sum is invariant under permutations of the head index — swapping heads h and h' produces identical output if the corresponding W_Q, W_K, W_V triples are also swapped. Trained SGD does not break this symmetry (the loss is symmetric under head permutation), so the 16 heads of Qwen3-1.7B are in a random gauge of the permutation group S_{16}. The first-principles question: is there a permutation π^* ∈ S_{16} that, when applied consistently across all 28 layers, minimizes the total variation in the head weight matrices across layers? If heads "replay" similar roles across layers when permuted to a canonical order, then the inter-layer weight matrices have a compressible delta structure — and a single layer's worth of head weights plus small per-layer deltas suffices. The algebraic identity: define the inter-layer delta D_L^h = W_Q^{L,h} - W_Q^{L-1,π^*(h)}, where π^* is the optimal inter-layer permutation. If ‖D_L‖_F ≪ ‖W_Q^{L-1}‖_F on average, the delta can be compressed (quantized, low-ranked, etc.) without touching the base. This connects to DeltaLLM's layer-delta insight but applies to head-permutation-aligned deltas rather than naive layer deltas — a structurally motivated alignment step before differencing.

### Residency arithmetic
Baseline: W_Q alone is 448 MB across 28 layers. If optimal-permutation inter-layer deltas have ‖D_L‖_F / ‖W_Q^{L-1}‖_F ≤ 0.30 (a testable hypothesis), and deltas are quantized to INT4 while the base layer is FP16, storage becomes: 1 base layer (16 MB) + 27 delta layers at INT4 (27 × 16 MB / 4) = 16 MB + 108 MB = 124 MB. Savings from 448 MB: ~72% = 324 MB on W_Q alone. Analogous for W_K, W_V, W_O: if all four compress at 72%, attention weight savings = ~720 MB. On Qwen3-72B (7× parameter scale): ~5.0 GB attention weight saved. This is speculative until the delta-norm ratio is measured.

### Novelty gloss vs the kill list and the published landscape
Closest kill: DeltaLLM / ADAMIX (R1/S2, killed for per-layer delta compression without head alignment). HPGO differs structurally: it applies optimal head permutation alignment BEFORE computing deltas, which reduces delta norm; naive DeltaLLM uses raw layer deltas without permutation alignment. Closest published: DeltaLLM (arXiv:2312.11514) uses inter-layer weight deltas but without head-permutation canonicalization. The S_{16} symmetry argument (trained SGD leaves heads in random permutation gauge) is not applied in DeltaLLM or any paper we have encountered. The optimal-permutation alignment step is the structural novelty.

### Smallest experiment
**Claim**: Optimal head-permutation alignment reduces the Frobenius norm of inter-layer W_Q deltas by ≥ 15% relative to naive (unaligned) deltas, averaged over layers 2–28 of Qwen3-1.7B-Base.
**Protocol**: (1) For each layer pair (L-1, L), compute the optimal head assignment π^* via minimum-weight bipartite matching on pairwise ‖W_Q^{L,h} - W_Q^{L-1,h'}‖_F for all (h, h') pairs (Hungarian algorithm on 16×16 cost matrix, trivial cost). (2) Compute aligned delta norm vs unaligned delta norm. Runtime: ~15 min (no forward pass needed, pure weight manipulation).
**Go threshold**: mean (‖D_aligned‖_F / ‖D_naive‖_F) ≤ 0.85 across layers 2–28, i.e., ≥ 15% reduction.
**NO-GO finding**: heads are not inter-layer permutation-aligned — each layer's heads re-specialize independently, so the permutation group symmetry of the sum is never "almost broken" in a consistent direction. This kills the head-permutation-delta-compression route and constrains inter-layer weight structure analysis.

### Primary risk
The bipartite matching finds the best permutation for each layer independently; the resulting permutations may not compose consistently across all 28 layers (π^*_L ≠ π^*_{L+1} ∘ π^*_L in general), meaning the delta structure is inconsistent — delta from layer 1 to 2 uses π^*_2, but delta from 2 to 3 uses π^*_3 which may be incompatible. Mitigation: solve for a single global permutation π^*_global that minimizes ∑_L ‖W_Q^{L, π^*_global} - W_Q^{L-1, π^*_global}‖_F using the same Hungarian algorithm on the summed cost matrix; this is a single global alignment in S_{16} and bypasses the consistency issue.

---

## F6 — Attention Sink Fixed-Point Absorption ASFA [FREE SWING]
**F / Track A**

### Mechanism
Track A — arch-transposition. In transformer attention, "attention sinks" are positions (typically BOS, delimiters, specific high-frequency tokens) that receive near-uniform high attention weight regardless of query content. Empirically (R2 stage-6 notes; PrefixQuant arXiv:2410.05265), ~2 per 2048 tokens absorb 94.7% of quantization error and have Jaccard ≈ 0.182 vs normal tokens at K=1% — but the critical first-principles observation is: these sink positions receive high attention weight because their KEY vectors K_sink lie in the dominant singular direction of W_Q W_K^T. The first-principles move: if K_sink is approximately a FIXED POINT of the attention kernel (a vector v such that Q·K_sink ≈ c·‖Q‖ for all Q in the query distribution), then the sink token's contribution to the attention output is approximately a fixed vector V_sink × avg_attention_weight. This fixed contribution can be PRECOMPUTED and stored as a constant bias b_L per layer (shape d_model), added to the residual stream, and removed from the runtime attention sum. The resulting "de-sinked" attention computes over non-sink keys only, with a sparser and lower-variance softmax over (context_length - 2) positions. The algebraic identity: output = b_L + ∑_{non-sink positions} softmax_renormalized(Q K^T) V. No retraining. The bias b_L is calibrated (shape 28 × d_model = 28 × 2048 = 115 KB) — well within CF10 conditioning: 115K params vs millions of calibration samples.

### Residency arithmetic
Bias storage: 28 layers × 2048 × 2 bytes = 115 KB — negligible. The payoff is in runtime: (1) attention computation skips 2 sink positions per inference step, reducing KK^T compute by 2/seq_len; at seq_len=4096 this is 0.05% — negligible. (2) The non-sink attention score distribution has lower variance, enabling tighter INT4 softmax numerics and lower quantization error on the remaining ~99% of KV cache. If non-sink K/V can be quantized to INT4 at ≤+0.10 nats vs INT8 (because variance is reduced), on Qwen3-72B at 32K context: INT4 KV cache = 2.5 GB vs INT8 = 5 GB → 2.5 GB saved. (3) The mechanism enables a stricter attention-sink-aware KV eviction policy: sink positions' bias is absorbed, so they need not occupy KV cache slots at all at long context.

### Novelty gloss vs the kill list and the published landscape
Closest kill: no item directly. Closest published: StreamingLLM (Xiao et al., 2023) identifies attention sinks and keeps them as persistent KV entries; HFST (hierarchical sink attention) keeps sink tokens; SnapKV and similar methods handle sink-token KV allocation. ASFA differs: it REMOVES sink tokens from the attention computation entirely by precomputing their contribution as a fixed-point bias — not preserving them as KV entries but algebraically absorbing them. The "fixed-point absorption" framing (turning an attention sink into a residual-stream constant) is not in the StreamingLLM / SnapKV family, which all retain sink KV entries.

### Smallest experiment
**Claim**: The BOS token's contribution to the residual-stream update can be represented as a fixed bias b_L (per layer) with calibration error ≤ 10% of ‖δh_from_BOS‖ on held-out tokens, where b_L = mean over calibration of (attention_weight_BOS × V_BOS).
**Protocol**: run 200 calibration tokens, record per-layer attention weight assigned to position 0 (BOS) per head, record V_0 per head, compute b_L = ∑_h mean(attn_weight_BOS^h) × W_O^h V_0^h. Then on 50 held-out tokens, compare actual BOS contribution (attention × V) to b_L; measure relative error ‖actual - b_L‖_F / ‖actual‖_F. Runtime: ~30 min.
**Go threshold**: median relative error ≤ 0.10 across 28 layers (i.e., b_L captures ≥ 90% of the BOS contribution on held-out).
**NO-GO finding**: BOS contribution is too query-dependent to be absorbed as a fixed bias — the attention sink is a statistical regularity, not a fixed-point of the attention map. This constrains the applicability of any fixed-bias absorption strategy for sink tokens, even with calibration data.

### Primary risk
V_BOS (the value vector at position 0) varies with context because it is the embedding of the BOS token which is fixed, but W_V changes per layer and the BOS token embedding is fixed — so V_BOS^h = W_V^h e_{BOS} is a FIXED VECTOR (not context-dependent). The only variable is the attention weight assigned to BOS, which is query-dependent. The bias b_L therefore depends on the mean attention weight — which may vary substantially across diverse queries, making the calibration mean a poor approximation. Mitigation: stratify the mean by token position (early vs late in context), since attention sink weight decays with distance; compute a position-bucketed b_L(position_bucket) instead of a single scalar.

---

## F7 — Tied Embedding Gradient-Path Rank Asymmetry TEGRA [OUT-OF-ORIENTATION, adjacent]
**F / Track B**

### Mechanism
Track B — compression. CF12 confirmed that the tied embed/lm_head matrix in Qwen3-1.7B-Base has r_99=1992/2048 — catastrophically full-rank, because gradient flows through both input embedding lookup AND output projection paths. But the gradient magnitudes through the two paths are NOT equal: the output projection gradient is proportional to prediction loss (high per step), while the input embedding gradient is proportional to how often each token appears in training data (sparser). This asymmetry means: the lm_head direction (output logit rows corresponding to rare tokens) may be much more compressed than the embedding direction (input embedding rows for frequent tokens). The first-principles move: define a row-importance ranking of the embed/lm_head matrix by (embedding lookup frequency × gradient magnitude from embedding path) vs (logit prediction importance × gradient magnitude from output path). If the two paths have an asymmetric rank structure — e.g., output-path importance is concentrated in top-K frequency tokens while input-path importance is distributed — then a MIXED decomposition is possible: full-precision rows for top-N_vocab frequent tokens (both directions), compressed rows for the tail (output direction only, since tail tokens are rarely looked up as inputs). The algebraic argument is that training frequency creates an implicit two-tier structure in the joint matrix even though the full matrix appears full-rank. The measurability is direct: token frequency statistics are available, gradient magnitudes can be measured in one forward pass.

### Residency arithmetic
Qwen3-1.7B tied embed: 151936 × 2048 × 2 bytes = 590 MB. Full-rank, no global SVD truncation. Mixed decomposition: top-10K tokens (covering >95% of training probability mass in BPE vocabulary — standard Zipfian assumption) stored at full BF16: 10000 × 2048 × 2 = 40 MB. Remaining 141936 tokens: if output-side gradient makes their rows near-zero after frequent-token projection, store at INT4 (or SVD-compress the tail block at rank 256): 141936 × 256 × 2 + 256 × 2048 × 2 = 73 MB + 1 MB = 74 MB. Total: 40 + 74 = 114 MB vs 590 MB baseline = 81% compression. ΔNLL: tail tokens contribute to output logits for rare completions — quality on rare-token outputs may degrade substantially. Go threshold must be conservative: ≤ +0.15 nats on standard WikiText-2 (which uses common vocabulary).

### Novelty gloss vs the kill list and the published landscape
Closest kill: CF12 (tied lm_head SVD truncation, KILLED at r<d with catastrophic dNLL). TEGRA does NOT truncate globally — it uses a frequency-stratified decomposition where frequent tokens stay full-precision and tail tokens are compressed. CF12 killed global truncation; TEGRA is a non-global, frequency-conditioned split. No published paper exploits the two-path gradient asymmetry in tied embedding matrices for post-training compression — the insight (embedding lookup frequency × output logit frequency as a joint importance measure) is not in GPTQ, AQLM, VPTQ, or any paper in the kill list.

### Smallest experiment
**Claim**: The sub-matrix of the tied embed/lm_head formed by rows corresponding to the bottom-50% of tokens by training frequency (rank by BPE unigram frequency) has r_99/d ≤ 0.80, while the top-50% has r_99/d ≥ 0.95 — measurable split in spectral concentration between frequent and infrequent tokens.
**Protocol**: (1) obtain BPE unigram frequency for Qwen3 tokenizer (available from tokenizer.json). (2) Sort the 151936 × 2048 embed matrix by token frequency, split top/bottom 50%. (3) Run SVD on each 75968 × 2048 block, measure r_99/d. Runtime: ~25 min (SVD on two 75968×2048 matrices).
**Go threshold**: r_99/d of bottom-frequency block ≤ 0.80 AND r_99/d of top-frequency block ≥ 0.93.
**NO-GO finding**: tied embedding rows do not have a frequency-conditioned spectral structure — the full-rank property extends uniformly across the frequency spectrum. This refines CF12: the full-rank finding is not due to tail-token noise but is a genuine uniform spectral property of the joint matrix. Closes the "frequency-stratified compression of tied embeddings" route.

### Primary risk
Unigram token frequency from the tokenizer vocabulary (or an external corpus) may not match the actual gradient magnitude distribution in Qwen3's training — the model may have been trained on a corpus with different frequency statistics than the tokenizer's default BPE frequencies. Mitigation: use the Qwen3 tokenizer's built-in unigram probabilities (encoded in tokenizer.json's added_tokens and vocab entries) as a proxy; alternatively, run one forward pass on a 500-token diverse sample and record per-token embedding lookup frequency directly.

---

## Convergence handles

- W_V / W_O fused composition spectrum (M_L = ∑_h W_O^h W_V^h, shape 2048×2048): opens or closes the last unmeasured attention weight class (F1)
- RMSNorm gain absorption effect on W_Q spectral concentration (RGAS pre-check: std(g_L); SVD comparison before/after fold): feeds any downstream attention compression chain (F2)
- W_Q row mean magnitude ‖μ_row‖ / median(‖W_Q row‖): Q activation dynamic range precondition for INT4/INT8 KV path (F3)
- W_down activation-weighted output subspace rank under post-SwiGLU activations: distinct from weight rank; closes or opens MLP output-side compression (F4)
- Inter-layer head permutation delta norm reduction via S_{16} alignment: prerequisite for any delta-compression scheme on attention heads (F5)
- BOS / sink token fixed-point bias calibration quality ‖actual - b_L‖ / ‖actual‖: gating measurement for attention-sink algebraic absorption (F6)
- Frequency-stratified spectral split of tied embed/lm_head (r_99/d top-50% vs bottom-50% by token frequency): determines whether CF12's full-rank finding is uniform or frequency-conditioned (F7)
