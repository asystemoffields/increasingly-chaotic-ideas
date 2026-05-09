# Stage 3 First-Principles v2 — Class Boundary as Geometric Fingerprint

## Path A — refined.

Core claim survives Stage 2. RMSNorm precondition resolved via paired-rotation fusion (option 3). Sheaf demoted to filtration. Zipf reframed as motivating observation. Z_16 reframed as shared-subspace tiling. MoDeGPT differentiation sharpened. E4 elevated as binary settling experiment.

## §1. The Geometric Object

CF10 class boundary (Qwen3-1.7B, d=2048, tied embedding): K-Jaccard slope drops with K; rotation by random Q catastrophically breaks lm_head (ΔNLL +19.96 at r=1024); per-matrix r_99/d hovers near 0.97; ΔNLL recovers asymmetrically across W_Q (≈0.63) vs W_K (≈0.79).

**Hypothesis (geometric, first-principles)**: these are not independent facts. They are six shadows of a single object — a stratified residual-stream **filtration**

V_1 ⊃ V_2 ⊃ ... ⊃ V_m, dim V_m = d* < d_model

such that trained weight matrices {W_E, W_gate, W_up, W_down, W_O, W_V} carry their information-theoretic mass inside the largest stratum V_1 (a d*-dim subspace of d_model-dim residual stream). Trained weights are full-rank only in **ambient computational basis**. There exists Q* ∈ O(d_model) such that, after right-multiplication by Q*, multiple weight matrices become **simultaneously** low-rank along the same coordinate split.

**Filtration not sheaf**: no restriction maps or pull-backs between stalks; residual stream composes additively. Filtration is the right object.

## §2. Predictions, Decisiveness Order

P1. **Joint low-rank.** d* ≈ 1400-1700 (≈0.7-0.83 × d_model). ∃ Q* ∈ O(d) such that for each M ∈ {W_E, W_gate, W_up}, r_99(M·Q*) ≤ d* well below 0.97·d ambient floor.

P2. **lm_head catastrophe dissolves under Q*.** +19.96 NLL penalty at r=1024 from random-Q rotation vanishes when rotation is Q* and threaded through full forward pass as paired rotations.

P3. **Asymmetry information-theoretic, not architectural.** r_99(W_Q·Q*) < r_99(W_K·Q*) by gap predicted from differential entropy of query vs key distributions.

P4. **Talking-Heads tiling.** Heads tile shared low-rank subspace — not literal Z_16 cyclic orbit. **Shared-subspace tiling**.

**NOT proposing:**
- NOT MoDeGPT (intra-module joint SVD excluding embedding/lm_head; we propose cross-module shared rotation INCLUDING tied lm_head).
- NOT QuaRot/SpinQuant (find rotations minimizing quantization clipping; Q* minimizes joint r_99).
- NOT QuIP (random orthogonal for incoherence; we use derived Q* for joint low-rankness).

The differentiation is now load-bearing: the surviving experimental claim is precisely the one no prior work has made.

## §3. MLP-Saturation Inequality (Derived)

**Claim**: For trained transformer FFN whose gate/up read from residual stream of effective rank k_residual:
  r_99(W_gate) ≥ k_residual − ε
with ε small relative to d_model.

**Sketch (gradient-decay / signal-propagation)**: ∇_{W_gate} L = (gradient_at_post_activation) ⊗ h. Under NTK-style signal propagation in pre-norm transformer, h has covariance Σ_h supported on k_residual-dim subspace V_residual ⊂ ℝ^d. Therefore ∇_{W_gate} restricted to columns whose right-singular vectors lie in V_residual^⊥ receives **zero gradient through h**.

Consequences:
1. Initialization fluctuations in W_gate columns aligned with V_residual^⊥ are not corrected by training.
2. Training drives relevant columns to span V_residual; remaining columns retain near-Gaussian initialization mass.

Empirically: r_99(W_gate)/d ≈ 0.97 in ambient basis — the ε. After right-rotation by Q* whose first d* columns span V_residual, columns d*+1...d_model carry only initialization noise → joint low-rank structure.

**Distinct from softmax bottleneck (Yang et al., ICLR 2018)**: bound on logit-domain rank (output side) vs our weight-domain bound (input side). Dual; not stated together in literature.

Non-circularity: k_residual measured independently from r_99 via residual stream covariance probing on held-out activations (E1).

## §4. RMSNorm Precondition Resolved (Option 3)

Stage 2 flagged: RMSNorm(Q*h) ≠ Q*·RMSNorm(h). Element-wise scaling doesn't commute with general orthogonal Q. Fatal for naïve "rotate lm_head only" test.

**Resolution**: thread Q* through **full forward pass** via paired rotations:
- Replace M_in (reading stream) with M_in · Q*
- Replace M_out (writing stream) with Q*^T · M_out

Stream now in rotated basis Q*^T h. RMSNorm computes h ↦ γ ⊙ h / ‖h‖_RMS. ‖·‖_RMS = ‖·‖_2 / √d is rotation-invariant. So in rotated coords ‖h'‖_RMS = ‖h‖_RMS — normalization commutes.

Non-commuting piece: element-wise scale γ. Resolution: **absorb γ into next linear layer**. Replace γ ⊙ x with diag(γ) x; absorb diag(γ) into subsequent W. After rewrite, RMSNorm = normalize-by-RMS only (rotation-invariant); per-channel scale is part of next weight matrix. Then Q* commutes cleanly.

QuaRot-style fusion engineering, different objective. Acknowledge QuaRot/SpinQuant as fusion precedent; differentiate by joint-low-rank objective vs quantization smoothness.

With this construction, rotation-invariance argument for logit ranking holds in rotated basis, and lm_head experiment (E4) is well-posed.

## §5. Joint Low-Rank Objective

  Q* = argmin_{Q ∈ O(d)} Σ_{M ∈ ℳ} r_99(M · Q)

where ℳ = {W_E, W_gate^{(ℓ)}, W_up^{(ℓ)}, W_O^{(ℓ)} acting through paired rotations for every ℓ}, with absorbed-RMSNorm-scale rewrite of §4.

Closest in form to **approximate joint diagonalization** (Kuleshov et al., arXiv:1501.06318) — but rectangular matrices sharing right-side rotation, and r_99 (rank functional) objective rather than off-diagonal Frobenius mass.

Solver: alternating projection. (a) fix Q, compute per-matrix singular bases. (b) update Q on Stiefel manifold by gradient descent on Σ_M r_99-surrogate (e.g., nuclear norm of truncated tail). Identical machinery to SpinQuant; only loss differs.

## §6. Decisive Experiments

**E1 (residual stratification, ~30 min).** Probe h^{(ℓ)} on held-out corpus across layers. Compute k_residual^{(ℓ)} = empirical 99th-percentile rank of activation covariance.
- Predict: k_residual^{(ℓ)} ≈ 1400-1700 for middle layers; lower at extremes (matching Shape of Learning bell shape).
- Falsified: k_residual ≈ d uniformly.

**E2 (MLP-saturation inequality, ~30 min).** Per layer compute r_99(W_gate^{(ℓ)}) and compare to k_residual^{(ℓ)} from E1.
- Predict: r_99(W_gate) ≥ k_residual − ε with ε ≤ 50.
- Falsified: W_gate systematically lower-rank than residual stream feeding it.

**E3 (joint Q* construction, ~4 hrs).** Solve joint optimization §5 over random subset of layers. Measure Σ_M r_99(M·Q*) vs Σ_M r_99(M·I) vs Σ_M r_99(M·Q_random).
- Predict: ≥ 1.4× compression on summed rank for Q* over identity and random baseline.
- Falsified: < 5% gain over random Hadamard.

**E4 (THE BINARY SETTLING EXPERIMENT, ~8 hrs).** Take Q* from E3. Apply full §4 paired-rotation fusion to network including tied lm_head. Compare:
- Random Q at r=1024: expected ΔNLL +19.96 (established catastrophe)
- **Q* at r=1024** (paired fusion, RMSNorm absorbed): **predicted ΔNLL ≪ +1.0**
- Identity at r=1024: ΔNLL ≈ 0 (sanity)

**Binary**. Either Q* dissolves catastrophe or it doesn't. Positive: Q* is viable post-training compression for Qwen3-class tied-embed models; geometric-fingerprint hypothesis supported in strongest form (joint shared-rotation low-rank INCLUDING embedding/lm_head — unpublished). Negative: hypothesis fails specifically at tied-embedding constraint; embedding's info mass occupies different subspace than FFN's — itself non-trivial finding.

**E5 (Q vs K asymmetry under Q*, ~2 hrs).** Predict r_99(W_Q·Q*) < r_99(W_K·Q*). Falsified if asymmetry purely architectural.

**E6 (head tiling, ~1 hr).** Pairwise singular-subspace overlap between heads after right-rotation by Q*. Predict shared-subspace tiling (high overlap on top-d*/16 directions) without literal cyclic structure.

## §7. Path / Changes from v1

1. **RMSNorm fix (§4)**: option 3 paired rotations + RMSNorm scale absorption. QuaRot/SpinQuant acknowledged as engineering precedent.
2. **Sheaf → filtration (§1)**: categorical baggage dropped.
3. **Zipf-Beggs-Plenz demoted**: motivating observation only, not load-bearing.
4. **Z_16 → shared-subspace tiling**: weaker but accurate. Consistent with Talking-Heads NeurIPS 2024.
5. **MoDeGPT differentiation explicit**: cross-module shared rotation including tied lm_head vs MoDeGPT intra-module excluding embedding.
6. **MLP-saturation inequality derived cleanly**: gradient-decay/NTK signal propagation. Distinguished from softmax-bottleneck. Non-circularity ensured by E1 independent measurement.
7. **E4 elevated to binary**: stated decisively. Either dissolves catastrophe or extends CF8 to tied-embed in a new way.

The math leads. Concrete remaining theorem: ∃ Q* ∈ O(d) such that with paired rotation/RMSNorm absorption, simultaneously r_99(W_gate·Q*), r_99(W_up·Q*), r_99(W_E·Q*) all fall to ≈ d* < 0.85·d AND ΔNLL on held-out text remains < +1.0. Stage 2 confirms no published work tests this. E4 settles it.
