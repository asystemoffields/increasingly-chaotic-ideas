# Side Thread: Calibration-Only AERO for SwiGLU

Investigative thread on activation removal + tensor decomposition for SwiGLU MLP. Opus agent landed 2026-05-08.

## Audacious target

4× compression of SwiGLU MLP weight footprint at ≤0.5 nat ΔNLL, calibration-only (no SGD), on Qwen3-class. Conservative fallback: 2× at ≤0.2 nat. 70B implication: MLP from 105 GB → 26 GB (Tucker), then → 7 GB (Q4 on Tucker factors). With Q4 attention: ~10 GB total hot set on 7.28 GiB laptop = **one streaming pass per token**, ~1.5-2 tok/s on 70B.

## Mechanism: Asymmetric Tucker-3 with calibration-fitted activation substitute

### Activation substitute (silu replacement)
NOT silu→identity (too lossy on gate-side outliers). NOT cubic Taylor (CSMF-killed quartic blowup). Use:

  silu(z) ≈ z · σ_fit(z), with σ_fit a per-layer 8-knot piecewise-linear fit to silu on empirical pre-activation distribution

Approximate σ_fit(W_gate·x) by a constant per-channel diagonal s ∈ R^{d_int} with s_m = E_calib[σ_fit((W_gate·x)_m)]. Then:

  silu(W_gate·x) ⊙ (W_up·x) ≈ (S·W_gate·x) ⊙ (W_up·x), S = diag(s)

**This absorbs into W_gate**: W̃_gate = S·W_gate. MLP becomes purely bilinear:

  y = W_down · ((W̃_gate·x) ⊙ (W_up·x))

### Bilinear fold + Tucker-3 decomposition

Natural CP form:
  T[k,i,j] = Σ_m W_down[k,m] · W̃_gate[m,i] · W_up[m,j]

Don't materialize T (8 GB). Factor as Tucker-3 with low-rank core:

  T ≈ Σ_a Σ_b V·C'·P_g[a,i]·P_u[b,j]  (V left factor, C' core, P_g/P_u factor matrices)

Storage: d·r_c + r_c·r_g·r_u + 2·d·r_g. At r_g=r_u=128, r_c=256: 5.2M params = **7.3× compression** vs original 38M.

### Fitting: calibration-only HOSVD

1. Collect (x_t, y_t) pairs on calibration (~1M token-pairs per layer)
2. **Mode-3 (output) factor V**: SVD of [y_1|y_2|...] truncated to r_c — **output-aligned, not weight-aligned** (key R2-aware step)
3. **Mode-1 (gate) factor P_g**: SVD of [W̃_gate·x_1|...] truncated to r_g — **post-activation subspace, not weight matrix** (key R1-aware step)
4. **Mode-2 (up) factor P_u**: SVD of [W_up·x_1|...] truncated to r_u
5. **Core C'**: closed-form least-squares min ||y_t − V·C'·(P_g·W̃_gate·x_t ⊗ P_u·W_up·x_t)||²

One HOSVD per layer + one closed-form lst-sq. ~30 sec per layer, ~15 min for 28 layers.

## Why R1/R2/CF8 enable this

**R1 (W_up dominance, gate anti-correlation in deep layers)**: post-gate activation has CONCENTRATED ENERGY. Σ_g should have rapid spectral decay, while Σ_u decays slowly. **Asymmetric Tucker (r_g ≪ r_u) is justified**, allowing more compression than symmetric.

**R2 (token-dynamic outliers at K=1-10%)**: y_t inherits outlier structure. Output-aligning V on calibration y_t (not on T's unfolding) adapts V to output-channel outliers.

**CF8 — the loophole**: CF8 says SVD of any single weight matrix is full-rank (no 2-mode structure). **But Tucker rank of the 3-mode tensor T can be lower than ANY unfolding's matrix rank.** CP rank ≤ d_int=6144. Tucker rank bounded by *subspace actually visited by calibration data*, NOT abstract weight rank. **CF8 does NOT close this loophole.**

Different claim: don't compress weights via SVD of individual matrices (CF8-killed). Compress combined-bilinear-on-data tensor in basis of activations.

## Preconditions

P1: Σ_g spectral decay — top-64 captures ≥95% energy in deep layers. ~5 min.
P2: silu→linearized cost — substitute, measure ΔNLL. ~2 hr. **Hard kill: ΔNLL>1 nat means program dies — Tucker can't recover.**
P3: Output-channel sparsity per R2 holds at y level. ~10 min.
P4: HOSVD knee in mode-1/2 spectra at rank well below 512. ~10 min/layer.

## Cascade

Exp 1 (8 hr): Single-layer surgery on Qwen3-1.7B layer 14. (r_g=128, r_u=128, r_c=256). Go: ΔNLL ≤0.05 nat one mid layer.
Exp 2 (12 hr): All-layers cascade. Go: ΔNLL ≤0.5 nat full-stack at 4.4× compression.
Exp 3 (6 hr): Ablate σ_fit substitution variants.
Exp 4 (10 hr, conditional): Per-layer rank tuning. Targets 6× at ≤0.5 nat.
Exp 5 (16 hr, audacious finale): Apply to Qwen3-30B. **Publishable if 4× MLP at ≤0.5 nat on 30B-class.**

Total: ~52 hr cascade.

## What this is NOT

- NOT CSMF (substituted silu polynomial keeping 3-matrix form; here we substitute then FOLD then compress)
- NOT SVD-of-weights (CF8 killed; we decompose 3-mode tensor T)
- NOT pure neuron pruning (CP rank truncation = neuron pruning; Tucker builds new basis vectors as linear combinations)
- NOT retraining (calibration-only HOSVD)

## The decisive empirical question

Is Tucker rank of T ≪ matrix rank of any unfolding on this specific Qwen3? Activation-subspace rank IS the load-bearing variable. Experiment 1 settles it.
