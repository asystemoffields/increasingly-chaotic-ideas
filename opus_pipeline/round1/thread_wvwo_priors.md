# Side Thread: W_V / W_O Theoretical Priors (pre-measurement)

Investigative thread on theoretical predictions for W_V and W_O before measuring them. Opus agent landed 2026-05-08.

## Predictions Summary

### W_V (shape [1024, 2048], same as W_K)
- r_99/d_in: **0.70-0.85** (close to W_K's 0.79, slight bias toward more concentrated)
- Global SVD K=512 ΔNLL: **+0.20 to +0.45** (overlapping W_K's +0.29 envelope, possibly LOWER)
- Global SVD K=256 ΔNLL: **+0.55 to +0.95**
- Per-head K_ph=64 ΔNLL: **+0.6 to +1.4** (softer NO-GO than W_Q's per-head)
- Class boundary placement: same row as W_K, possibly slightly more compressible

**Headline**: W_V should be MARGINALLY MORE compressible than W_K. Reason: errors in V get averaged across many tokens via softmax(QK^T)·V (post-attention is convex combination over rows of V), while K errors are exponentiated through softmax. **W_V K=512 ΔNLL ∈ [+0.20, +0.30]** likely.

### W_O (shape [2048, 2048], same as W_Q)
- r_99/d: **0.65-0.80** (between W_Q's 0.63 and W_K's 0.79; closer to W_Q)
- Global SVD K=1024 ΔNLL: **+0.05 to +0.20**
- Global SVD K=512 ΔNLL: **+0.15 to +0.35**
- Global SVD K=128 ΔNLL: **+0.85 to +1.20** (similar to W_Q's +0.98)
- Per-head K_ph=64 ΔNLL: **+0.3 to +0.9** (softer NO-GO than W_Q)
- Per-head K_ph=96 ΔNLL: **+0.1 to +0.4** (POSSIBLY a per-head GO where W_Q wasn't)

**Headline**: W_O structurally similar to W_Q on global axis (square, 16-head, residual-touching) but **per-head structure is WEAKER** than W_Q's. The 16:1 head-redundancy is NOT symmetric across W_Q and W_O. Reason: W_Q reads the same residual stream 16× and projects each into a "query" — those projections collapse to shared subspace (AQFKV). W_O reads 16 DIFFERENT per-head attention outputs and combines them — each is functionally distinct.

### Reach prediction: Joint Q-K-V-O low-rank
- Predicted shared k_residual ≤ **1100 ± 150** subspace
- Joint summed-r_99 ≈ 4 × 0.55 × 2048 ≈ 4500 (vs naive 4×2048=8192)
- **~1.8× joint compression on attention block at <0.5 nats** if subspace shared rather than orthogonal
- If overlap > 70%, single shared down-projection per layer that all four attention matrices factor through

## Mechanism arguments

### W_V argument 1: Information role
V carries actual content moved by attention. Like K, every token must produce distinguishable V — softmax(QK^T) just selects which V to read; if V were low-info, attention would be useless. So V should be at least as concentrated as K, not less.

### W_V argument 2: Gradient/NTK
W_V reads from residual stream (input dim 2048). Gradient direction bounded by residual stream's effective rank k_residual. So r_99(W_V) along input axis bounded above by k_residual, same as W_K. By symmetry of read-side argument, r_99(W_V) ≈ r_99(W_K) to first order.

### W_V argument 3: Head structure / GQA
W_V is GQA-grouped (8 KV heads vs 16 Q heads), exactly like W_K. The 2:1 ratio means W_V CANNOT have the 16:1 collapse W_Q has. At most 8:1, but each KV head services 2 Q heads with potentially different attention patterns, so W_V should NOT exhibit dramatic head redundancy.

### W_V asymmetry argument: Why slightly MORE compressible than W_K
A·V is a CONVEX COMBINATION over rows of V. Errors in low-singular-value directions of V get **averaged across many tokens** before hitting residual stream. W_K errors get **exponentiated through softmax** — small W_K errors flip large attention weights. So W_V tolerates same rank cut with LESS logit damage.

### W_O argument: Distinct head outputs
W_O reads from concat(per-head attention outputs), each in column space of V_h. Per-head outputs are FUNCTIONALLY DISTINCT — each head has already selected its content via softmax(Q_h·K_h^T). W_O reads less head-redundant input than W_Q. Therefore **W_O's per-head 16:1 collapse is mechanistically weaker than W_Q's**.

### k_residual scaffold (cross-matrix prediction)
ALL four attention matrices in a single layer should converge to roughly the SAME r_99, because all four are bottlenecked by the same residual stream effective rank.

- W_Q.layer14: r_99 = 1293 (measured)
- W_K.layer14: r_99 = 814 (measured) — bounded by min(input_used, output_dim) = min(k_residual, 1024) ≈ 1024 ceiling, 814 empirical
- W_V.layer14: predicted r_99 ≈ 800-870
- W_O.layer14: predicted r_99 ≈ 1300-1500 (matching W_Q's k_residual bound)

## Decisive falsifier (most informative single test)

**W_O per-head K_ph=64 ΔNLL**:
- If << W_Q's +1.53 → 16-head structure SYMMETRIC in W_Q and W_O, can compress BOTH via per-head SVD of shared subspace (huge win)
- If ≈ W_Q's +1.53 → 16-head redundancy real and symmetric, same compression class
- If >> W_Q's +1.53 (e.g., +2.5) → W_O structurally less compressible per-head than W_Q; compression must go through global

## Smallest experiment

`scripts/aqfkv_vo_rank_sweep.py` (mirror of aqfkv_q_rank_sweep.py):
1. Reuse EVAL_PASSAGE from AQFKV (455 tokens, baseline NLL=2.4668)
2. Sanity gates: K=full-rank for both W_V and W_O
3. W_V global SVD sweep: K ∈ {1024, 512, 256, 128, 64}
4. W_O global SVD sweep: K ∈ {2048, 1024, 512, 256, 128, 64}
5. Per-head sweep on W_O only: K_ph ∈ {128, 96, 64, 32, 16}
6. Spectrum diagnostic at layers 0, 14, 27 for W_V and W_O
7. **Bonus joint-SVD**: stack [W_Q, W_K, W_V, W_O^T].layer14 vertically, joint SVD, measure rank-K reconstruction error at K ∈ {2048, 1500, 1000, 500}

~10-15 min on CPU. Tests prediction sets 1.1 + 1.2 + 1.3 in one run.

## Implications by outcome

**If W_V tracks W_K closely (predicted)**: Combine W_K + W_V global SVD at K=512 → ~2× attention-side weight compression at ΔNLL ≈+0.55. Combined W_QKV at K=512 ≈ 2× attention block compression at ΔNLL ≈+0.75.

**If W_V is MORE compressible than W_K**: Strong asymmetry. softmax bottleneck makes K errors expensive but V errors cheap. Suggests V can be aggressively quantized (4-bit or 3-bit) while K stays at higher precision. Empirically motivated knob for asymmetric KV-cache quantization.

**If W_O has weaker head redundancy (predicted)**: Per-head W_O compression is NOT pure recapitulation of per-head W_Q. Compression must go through GLOBAL SVD on W_O. Modest 2× at K=1024 with ΔNLL ≈+0.10. Mechanistic interpretability: the pre-softmax computation is shared; the post-softmax computation is differentiated.

**If joint Q-K-V-O union rank small (predicted ~1.5×)**: BIGGEST architectural finding from this line. Each transformer layer's attention block factors as U_layer · [small_Q, small_K, small_V, small_O] with U_layer ∈ R^{2048×1500}. **~3× attention weight compression at <0.5 nats** — bigger than QSVD.
