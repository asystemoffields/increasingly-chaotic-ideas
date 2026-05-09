# Stage 2 Prior-Art Filter — First-Principles / Class Boundary as Geometric Fingerprint

## Verdict: SURVIVES with critical RMSNorm precondition

## M1: Joint orthogonal rotation Q* simultaneously low-ranking W_gate, W_E, W_up
Status: PARTIAL with critical gap that is the actual novelty claim

Rotation literature thoroughly mapped:
- QuIP (arXiv:2307.13304): random orthogonal pre/post-multiply for quantization incoherence
- QuIP#: randomized Hadamard for O(n log n)
- QuaRot (arXiv:2404.00456): four structured Hadamard rotations R1-R4 fused into weights
- SpinQuant (arXiv:2405.16406, ICLR 2025): learnable rotations on Stiefel manifold
- OSTQuant (arXiv:2501.13987, ICLR 2025): learnable orthogonal + diagonal scaling per FC layer

**Critical gap**: All target QUANTIZATION SMOOTHNESS (minimizing clipping error), NOT joint low-rank. None claim/test that single Q* minimizes r_99 simultaneously across W_gate, W_up, W_E.

MoDeGPT (arXiv:2408.09632, ICLR 2025): joint SVD across matrix pairs within a module. Standard SVD in ambient basis — no shared rotation, EXCLUDES embedding/lm_head.

ASVD (arXiv:2312.05821): activation-aware scaling then SVD; handles lm_head SEPARATELY, excluded from compression ratios.

**Tied lm_head under shared rotation is an open hole.**

Sharpen: Frame as "Q* = argmin Σ_i r_99(M_i · Q*) over O(d) where {M_i} = {W_gate, W_up, W_E=W_lm_head}." Closer to approximate joint diagonalization (Kuleshov et al. arXiv:1501.06318) but rectangular weights with shared right-rotation, not square commuting matrices.

## M2: Stratified residual stream as filtration
Status: ADJACENT
- "Attention Layers Add Into Low-Dimensional Residual Subspaces" (arXiv:2508.16929): attention output effective rank ≈ 0.6×d_model universally; MLP outputs ≈ 0.9×d_model.
- "The Shape of Learning: Anisotropy and Intrinsic Dimensions" (arXiv:2311.05928, EACL 2024): bell-shaped anisotropy curve peaking middle layers.

What's missing: Sheaf framing not in those papers. They measure global effective rank, not context-conditional. K-dependent Jaccard collapse as evidence for stratification unreported.

CF9: "Sheaf" is loosely used. Formal sheaf requires restriction maps between stalks; residual stream is additive not pull-back. **State as filtration not sheaf.**

## M3: MLP-Saturation Inequality
Status: ADJACENT
Softmax bottleneck literature (Yang et al. ICLR 2018) parallel but in logit domain not weight domain. No paper states r_99(W_gate)/d as function of training time.

CF9: Inequality requires k_residual measured independently from r_99 (else circular). Gradient-decay argument needs signal-propagation/NTK derivation.

## M4: lm_head catastrophe as basis artifact
Status: PARTIAL — most dangerous partial pre-emption.

Logit rotation invariance is mathematically trivial (Yang et al. 2018 implicit; arXiv:2510.24966 explicit). Godey & Artzi (arXiv:2603.10145) covers gradient-bottleneck training-dynamics consequence.

**Specific gap**: No paper asks whether ΔNLL penalty at r=1024 dissolves under Q* found by minimizing r_99 across all weight matrices. Logit low-rank theorem says functional rank small, but no test of post-training rotation making representational rank small simultaneously with other weight matrices.

## CRITICAL CF9 BLOCKER

**RMSNorm breaks O(d) invariance.** RMSNorm(Q*h) ≠ Q*·RMSNorm(h). Logit ranking invariant under O(d) only holds in pre-norm basis. Qwen3 has final RMSNorm before lm_head.

Address via:
1. Absorb RMSNorm scale factors into rotation (works for per-element scale, not L2 normalization)
2. Restrict Q* to subgroup commuting with element-wise L2 norm — permutations only
3. **Acknowledge prediction applies in pre-norm basis; thread Q* through full forward via paired rotations** — Q* immediately before W_E, Q*^T at next layer's residual entry — preserves function exactly. **QuaRot-style fusion in spirit but for joint-low-rank objective**. Recommended.

## M5: Zipf-Beggs-Plenz framing
Status: NOVEL but tenuous chain. **Demote to motivating observation.**

## M6: Z_16 symmetry interpretation
Status: NOVEL framing, adjacent empirics. Talking Heads (NeurIPS 2024) finds shared singular subspaces. Z_16 symmetry implies all heads functionally equivalent (Talking Heads contradicts). **Better framing: "heads tile a shared subspace."**

## Overall Verdict

Most load-bearing surviving claim: ∃ Q* such that W_E·Q* has r_99/d ≪ 0.97 AND same Q* simultaneously low-ranks W_gate, W_up, with ΔNLL dissolving on tied lm_head. NOT attempted in literature.

MoDeGPT closest structural predecessor — extend to shared rotation across all matrices INCLUDING tied lm_head with group-theoretic and information-theoretic motivation. Differentiated enough to pursue.

Critical blocking precondition: RMSNorm O(d) invariance breakage. Resolve via option 3 (paired rotations through full forward pass) before E4.
