# Stage 1 First-Principles Agent — Class Boundary as Geometric Fingerprint

## 1. Ambition

Claim: the measured class boundary CF10 (MLP r_99/d ≈ 1.0, attention W_K ≈ 0.79, W_Q ≈ 0.63, tied embed/lm_head ≈ 0.97, K-dependent activation Jaccard collapse) is not a list of independent facts. It is a single geometric object: the residual stream of Qwen3 lives on a stratified manifold whose intrinsic dimension at each layer is bounded above by an "MLP-saturation dimension" d*. Predict d* ≈ 1400-1700 for Qwen3-1.7B (d_model=2048), and that representing the residual stream in this stratified basis admits a 1.4-1.6× lossless-equivalent compression of all weight matrices simultaneously, including the tied lm_head — which today appears irreducible.

If d* < d_model strictly (which findings imply it must), apparent full-rankness of W_gate, W_up, lm_head is *only* full-rank in the ambient basis. They are co-low-rank in the basis adapted to the residual stream's actual support. This converts the most painful structural finding (lm_head ΔNLL +19.96 at r=1024) into a tractable problem: the catastrophe is a *basis* artifact, not a *rank* fact.

Aspirational hardware payoff: uniform 1.5× weight compression composes with existing 4-bit quant to yield ~6.7× over fp16. For 70B at 0.55 bpw NanoQuant: 5.35 GB → 3.5 GB residency, ~1.2 s/token sequential streaming.

## 2. Mechanism

### 2.1 MLP-Saturation Inequality

SwiGLU MLP: y = W_down · (SiLU(W_gate · x) ⊙ (W_up · x)). Empirically W_gate, W_up have r_99/d ≈ 1.0: SVDs no compressible tail in ambient basis.

If x lives in intrinsic-d subspace S ⊂ R^d of dim k < d, any W applied to x only sees W|_S. Component of W orthogonal to S has no gradient and decays under weight decay.

Contrapositive (MLP-Saturation Inequality):
  r_99(W_gate) ≥ k_residual − ε(decay, training time)

W_gate r_99/d ≈ 1.0 ⟹ k_residual ≥ 0.99·d_model along rank-supporting subspace. But this is rank along *some* basis. Says nothing about *measure* — Gaussian on R^d has rank-d support but 99% energy on top-σ eigenspace.

### 2.2 Stratification

Define residual stream stratification at layer ℓ as filtration (V_1 ⊃ V_2 ⊃ ... ⊃ V_m):
- V_1 = R^d
- V_2 = top-99%-energy subspace under typical-token activations (Jaccard-stable channels at K=0.1%)
- V_3 = top-99%-energy subspace conditional on outlier-token strata
- V_m = residual stream conditional on particular semantic context

K-dependent Jaccard collapse (0.72 → 0.31 → 0.26 → 0.24 from K=0.1% → 10%) is direct evidence: 2 channels stable across tokens are V_2 \ V_3; 100+ token-dynamic channels are V_3.

This is a presheaf assigning local active subspace to each token-context, with restriction maps via softmax/SwiGLU compositions. Residual stream is a section of this presheaf, not an element of R^d.

### 2.3 Why W_Q ≈ 0.63 ≠ W_K ≈ 0.79

Q broadcasts (one query attends all keys), K is unique per token.
- I(Q_token; context) small — Q declares "I'm looking for X"
- I(K_token; context) ≈ I(token; context) — K must distinguish this token

K supports more bits of distinguishability per token. Empirically: 0.79 vs 0.63. The 16:1 head-redundancy in W_Q corresponds to all 16 heads addressing a shared low-dim "what to look up" code; W_K's heads address higher-dim "what I am" codes.

Implies: attention output lives in subspace dim ≤ rank(W_Q) ≈ 0.63·d. After W_O projects back, increment added to residual stream lives in W_O · range(softmax(QK^T)·V), bounded by min(rank(W_Q), rank(W_V), d_head·n_heads).

### 2.4 lm_head Catastrophe is Basis Mismatch

Tied W_E satisfies logit_v = ⟨W_E[v], h⟩. Logit ranking is invariant under O(d) Q applied jointly to W_E and h. So lm_head's "rank" in ambient basis confounds:
1. Functional rank: dim of subspace lm_head's output (logit vector) actually varies in
2. Representational rank: dim of subspace W_E rows span

arXiv:2510.24966 says (1) is small. Our ΔNLL +19.96 at r=1024 says (2) is large in standard basis. Compatible iff rotation Q* aligning views is non-trivial and entangled with residual stream's V_2/V_3 stratification.

**Prediction**: there exists orthogonal Q* such that W_E · Q* has r_99/d ≪ 0.97, and same Q* applied to W_gate, W_up makes them simultaneously low-rank.

## 3. Why findings make this work

- r_99(W_gate)/d ≈ 1.0 + W_Q at 0.63 (head-redundancy): residual stream uses every direction MLP can address but attention update is low-rank ⟹ anisotropic at high resolution but isotropic at low resolution = stratified geometry
- K-dependent Jaccard (0.72→0.24): Renyi-α entropy as α decreases. Slope 0.18/decade matches Zipf exponent ~1.2 — consistent with critical neural networks (Beggs-Plenz avalanche scaling)
- Layers 2-19 vs 23-27 (5-8% vs 14-18%): stratification deepens with depth. Sheaf has higher-dim fibers in deep layers. d* layer-dependent; earlier layers admit more aggressive compression
- W_Q per-head K=64 NO-GO (+1.53) but global K=128 OK (+0.98): 16 heads share basis. Z_16 group action on head index. Head decomposition is a SYMMETRY, not info partition

## 4. Preconditions

P1: Common low-energy basis exists across W_gate, W_up, W_E. Test: SVDs of all three, angular alignment of top-k singular vectors. ~2 min.

P2: Activations h_ℓ have effective dim k_ℓ < d_model in right basis (participation ratio under joint singular basis from P1). Test: collect ~100k residual stream vectors, compute participation ratio standard vs P1 basis. ~10 min.

P3: K-dependent Jaccard collapse follows Zipf law with exponent in [1.0, 1.5]. Test: re-fit existing Jaccard data to Zipf, R² > 0.95. 30 sec.

P4: Q* ∈ O(d) such that ‖W_E Q* − low-rank-r‖_F small for r ≪ d, AND ‖W_gate Q* − low-rank-r‖ small. Joint approximate diagonalization. Jacobi-style sweep. ~30 min for d=2048.

P5: Applying Q* preserves perplexity to within 0.1 NLL. Sanity. ~1 hour.

## 5. Cascade

**E1 (4 hrs): Joint Basis Discovery.** SVDs of W_gate, W_up, W_down^T, W_E for layers {0, 6, 12, 18, 24} of Qwen3-1.7B. Solve Q_ℓ minimizing joint nuclear-norm-tail (Jacobi sweep ~10^4 plane rotations). Report r_99/d in rotated basis.
- Go: aligned-basis r_99/d ≤ 0.85 for W_gate AND W_E simultaneously → E2
- No-go: r_99/d > 0.95 in joint basis → basis-mismatch hypothesis wrong; pivot to weight-tying-induced curvature analysis

**E2 (2 hrs): Stratified Activation Dimension.** 50k tokens through Qwen3-1.7B in rotated basis from E1. Compute layer-wise participation ratio.
- Go: PR(h_ℓ) ≤ 0.7·d for layers 2-19
- No-go: PR > 0.9·d uniformly → residual stream genuinely full-dim; stratification fails

**E3 (30 min): Zipf Fit.** Re-fit K=0.1%/1%/5%/10% Jaccard data to J(K) = 1 − c·K^α. Predict J at K=0.01%, K=20% and validate.
- Go: R² ≥ 0.95, exponent ∈ [1.0, 1.5]
- No-go: bad fit → activation distribution not scale-invariant; downgrade critical-network framing

**E4 (8 hrs): Rotated-Basis Truncation.** Apply Q* from E1, truncate to rank 0.6·d in rotated basis, measure wikitext perplexity. **The killer datapoint**: if ambient lm_head ΔNLL = +19.96 and rotated lm_head ΔNLL = +0.3, catastrophe is dissolved.
- Go: ΔNLL at 0.6·d truncation ≤ +0.3 for W_E in rotated basis
- No-go: ΔNLL > +2.0 in rotated basis → lm_head genuinely full-rank functionally; abandon CF10-as-basis-artifact framing

**E5 (1 day): Composite Compression on Qwen3-4B.** Apply E1-E4 pipeline. Measure memory residency, tok/s on Ryzen 5 7530U with q8_0 KV, wikitext perplexity.
- Go: 1.4× compression at ΔNLL ≤ +0.5, tok/s ≥ 5.0 (current baseline 5.5)
- No-go on tok/s: rotated weights cost runtime (extra matmul for Q^T) — back-compose Q into LayerNorms (orthogonal rotations commute through RMSNorm to next layer's weights) — but see CF9 risk below

## 6. What I'm NOT proposing

- Not retraining anything. No SGD; orthogonal post-hoc transformations only.
- Not per-head W_Q compression (per-head K=64 ΔNLL+1.53 NO-GO; head decomposition is symmetry, not compression axis)
- Not pruning channels by outlier static-ness (K=0.1% Jaccard 0.72 covers only ~2 channels)
- Not function-level low-rank lm_head via post-softmax distillation (RSIDC class). Different program; if E4 succeeds, RSIDC line becomes unnecessary because *weights* are low-rank in the right basis
- Not linear/kernel attention replacement
- Not MLA-style joint Q-K projection as primary line. If E1 succeeds, joint basis Q* already encodes whatever Q-K alignment exists

## Bottom Line

Every measured number is consistent with one hypothesis: residual stream is a section of stratified filtration with intrinsic dimension d* < d_model, and trained weights are full-rank only because SVD is computed in a basis confounding V_2/V_3/.../V_m strata. Find the right basis Q* and entire CF10 fingerprint becomes simultaneously compressible.

The critical experiment is E4: rotating tied lm_head and measuring whether +19.96 ΔNLL catastrophe is basis artifact or functional fact. **Decisive in <8 hours.**

## CRITICAL CF9 RISK (per Stage 2 review)

**RMSNorm breaks O(d) invariance.** RMSNorm(Q*h) ≠ Q*·RMSNorm(h) in general. Logit rotation invariance only holds in pre-norm basis. Must address before E4 by either:
1. Absorbing RMSNorm scale factors into the rotation (works for the per-element scale only, not for the L2 normalization step)
2. Restricting Q* to the subgroup that commutes with element-wise L2-normalization — which is permutations, not general O(d)
3. Acknowledging the prediction applies only in the pre-norm basis (and threading Q* through the full forward pass via paired rotations)

Option 3 is the most flexible — Q* applied immediately before W_E (after the final RMSNorm) and Q*^T immediately at the residual entry to the next layer would preserve function exactly. This is QuaRot-style fusion in spirit but for joint-low-rank objective rather than quantization smoothness.
