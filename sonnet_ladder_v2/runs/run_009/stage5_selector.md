# Stage 5 — Selector — Run 009 (v2 Sonnet Ladder)

Selector: Sonnet 4.6 (single-agent pass, both tracks).
Date: 2026-05-09.
Inputs read: STAGE5_SELECTOR.md, stage2_judge.md, stage3_deep_research.md, stage4_skeptic.md, SUMMARY.md, KILL_LIST.md.

Kill list applied: v2-S3-R009-001 (F2-CKDJ REGENERATE, CF9 fail); v2-S3-R008-001 (F4-SVAL-CONSERVED); v2-CHEAP-TEST-001 (cross-layer W_Q class); all prior-round kills.

---

## Scoring Summary

### Track B

Stage 3 top 3 advancers: F1-MQKP (S2=13), R9-R1/ATQKV (S2=12), C15-SNRATE (S2=11).
Stage 4 top 2 (per Stage 4 recommendation): F1-MQKP, R9-R1.

| ID | S2 Total | S3 Conf | Pre-Mult | Conv × | Elegance × | Frame | NO-GO | Final |
|---|---|---|---|---|---|---|---|---|
| F1-MQKP | 13 | +2 | 15 | 1.5 (C1, 5-orient) | 1.3 (algebraic-identity + 2+-orient convergence) | 0 | +2 (CF-candidate: product-SVD vs factor-SVD class boundary) | **31.25** |
| R9-R1/ATQKV | 12 | +2 | 14 | 1.5 (C1, 5-orient) | 1.1 (W_V/W_O structural measurement) | 0 | +2 (CF-candidate: extends CF8/CF11 boundary to W_V/W_O) | **25.3** |
| C15-SNRATE | 11 | +1 | 12 | 1.0 | 1.1 (calibration-fit, auditable) | 0 | +1 (falsifies rank-sensitivity ≈ quant-sensitivity proxy) | **14.2** |
| F4-WQKRS | 12 | +1 | 13 | 1.2 (C1, 2-orient) | 1.2 (water-filling classical, constructive) | 0 | +1 (spectral asymmetry as class finding) | **17.68** |

Convergence multiplier: C1 has 5 ideas (MQKP, ATQKV, WQKRS, EQSUB, HSMLA) → 1.5 for all C1 members. MQKP is the C1 strongest representative.

Elegance multiplier for MQKP: constructive algebraic-identity (Eckart-Young applied to M = W_Q W_K^T is exact and auditable) AND 3+-orientation convergence cluster (C1) → 1.3 cap (capped at 1.3 per STAGE5_SELECTOR §3b). Pre-multiplier score = 15 ≥ 9 → multiplier applies.

NO-GO bonus for MQKP: the NO-GO outcome (M's spectrum is no more concentrated than W_Q alone) itself closes the product-SVD vs factor-SVD class — a CF-candidate structural finding that refutes the algebraic argument, constraining any future "find a more compact form for W_Q W_K^T" proposals. +2.

Re-verification of novelty (as of 2026-05-09): no paper found computing SVD of M = W_Q W_K^T as a compression primitive. QSVD (arXiv:2510.16292) concatenates Q/K/V as separate matrices; CARE (arXiv:2603.17946) decomposes K and V via covariance weighting; neither takes the product M. Stage 3 search confirmed NOVEL.

**Track B pick: F1-MQKP**

Runner-up: R9-R1/ATQKV (score 25.3). If Stage 6 kills MQKP (hidden prior art or CF9 error in the product-SVD framing), advance ATQKV without re-running Stage 5.

---

### Track A

Stage 3 top 3 advancers: C14-CFSHIFT (S2=11, frame-novelty), F3-L1GNF (S2=11), R9-R3/HSMLA (S2=10).
Stage 4 has no Track A idea in its explicit "top 2" recommendation; Stage 4's S3-LRCS-WDLA and S1-MQKP-ZONEMAP are cross-pollination ideas, both gated on other experiments (MQKP and RSUC respectively).

| ID | S2 Total | S3 Conf | Pre-Mult | Conv × | Elegance × | Frame | NO-GO | Final |
|---|---|---|---|---|---|---|---|---|
| C14-CFSHIFT | 11 | +2 | 13 | 1.0 | 1.1 (calibration-fit, auditable; A4=2 so not constructive) | +2 (A2=3 + no published cousin) | +1 (cross-layer logit coupling class boundary) | **17.3** |
| F3-L1GNF | 11 | +2 | 13 | 1.0 | 1.2 (null-space criterion is constructive) | 0 | +2 (CF-candidate: null(C_x^0) alignment as fold predictor) | **17.6** |
| R9-R3/HSMLA | 10 | +1 | 11 | 1.5 (C1, 5-orient) | 1.2 (constructive, within-layer stacked SVD) | 0 | +1 (head-sharing geometry, closes W_Q per-head question) | **17.7** |

Re-verification for C14-CFSHIFT (as of 2026-05-09): "constant attention logit bias from prior-layer gate-fold calibration" — no paper found. ALiBi/xPos/YaRN add position biases, not calibration-content biases derived from gate statistics. NOVEL confirmed.

Re-verification for F3-L1GNF (as of 2026-05-09): null(C_x^0) as fold predictor is not in AERO (arXiv:2410.13060) or any post-training activation-removal paper. NOVEL confirmed.

HSMLA has a 1.5 convergence multiplier (C1), but TransMLA (arXiv:2502.07864, NeurIPS 2025 Spotlight) is a strong adjacent paper doing W_Q/KV post-training MLA conversion with fine-tuning; HSMLA's training-free W_Q-specific factorization is differentiated but the adjacency docks novelty. Pre-multiplier score = 11 is above the 9 floor → multiplier applies.

Three-way near-tie: F3-L1GNF (17.6), HSMLA (17.7), CFSHIFT (17.3). Tiebreak criteria: (1) lower runtime → L1GNF 25 min vs CFSHIFT 30 min vs HSMLA 45 min; (2) NO-GO finding quality → L1GNF NO-GO is a CF-candidate (null-space alignment as fold predictor, closes the class-level question about Layer-1 anomaly mechanistic explanation) vs HSMLA NO-GO (per-head W_Q geometric independence, a structural refinement but not a new CF) vs CFSHIFT NO-GO (logit coupling class boundary, +1 only because the class is narrow).

L1GNF wins the tiebreak: shortest runtime (25 min), NO-GO is a CF-candidate (+2) closing the CF6 mechanistic origin, and the elegance multiplier is the highest (1.2, constructive null-space criterion).

**Track A pick: F3-L1GNF**

Runner-up: C14-CFSHIFT. If Stage 6 kills L1GNF (CF10 re-examination of mean-field collapse at low calibration count, or AERO prior-art overlap), advance CFSHIFT.

---

## Track B Pick — Full Experiment Plan

### 1. Title and one-line description

**Round 9 / Track B — F1-MQKP: Attention Product M = W_Q W_K^T Spectral Compression**

Does SVD of the attention score operator M = W_Q W_K^T (rather than independent SVD of W_Q and W_K) yield strictly better compression by Eckart-Young optimality, and what is M's r_99/d on Qwen3-1.7B-Base?

### 2. Class tags

`compression-lr` / `attention-weight-compression`
Elegance class: `algebraic-identity` (Eckart-Young on a real matrix M, no precondition failures possible).

### 3. Hypothesis

**H1 (load-bearing)**: M^L = W_Q^L (W_K^L)^T ∈ ℝ^{2048×2048} has r_99/d < min(r_99(W_Q^L)/d, r_99(W_K^L)/d) ≈ 0.63. Product-SVD at rank r is Frobenius-optimal for attention score reconstruction; independent compression at matched rank is strictly suboptimal by the cross-term argument (M3 in Stage 3). This depends on CF11 (W_Q r_99/d ≈ 0.63, W_K r_99/d ≈ 0.79).

**H2 (structural claim)**: At K=350–400, M^L product-SVD gives ΔNLL ≤ +1.0 nats — matching or beating CF11's K=128 independent W_Q compression (+0.98 nats) while achieving a joint W_Q+W_K compression factor of approximately 8–10× vs bf16. The joint Q+K residency at K=400 drops from 896 MB (28 layers × 2 matrices × bf16) to ~179 MB, a 60% further reduction beyond CF11's independent K=512.

CF dependencies: CF11 (concentration of W_Q and W_K spectra as motivation); no other CF entries required. CF8 boundary (MLP full-rank) is confirmed not to apply to attention weights.

### 4. Smallest test

**Model**: Qwen3-1.7B-Base, bf16.
**Hardware**: Ryzen 5 7530U, 7.28 GiB RAM. Each M^L = W_Q @ W_K.T is 2048×2048 × 2B = 8 MB; SVD of a 2048×2048 matrix ≈ 1 min on Ryzen.

**Calibration corpus**: None required. Product M is computed directly from trained weights — no activation data.

**Eval corpus**: WikiText-2 test split, 455 tokens (same held-out used in all v2 ladder evaluations; `datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")[:512]` trimmed to first complete block).

**Layers**: Pilot on L7, L14, L21 (representative early/mid/late). Full 28-layer sweep runs only if pilot GO.

**Protocol**:
1. For each pilot layer L: load `model.layers[L].self_attn.q_proj.weight` (W_Q, shape 2048×2048) and `k_proj.weight` (W_K, same shape).
2. Compute M^L = W_Q @ W_K.T (shape 2048×2048, float32 accumulation).
3. Full SVD: `U, S, Vt = np.linalg.svd(M, full_matrices=False)`.
4. Compute var@K = cumsum(S²)[:K] / sum(S²) for K ∈ {256, 300, 350, 400, 450, 500}.
5. Compare against CF11 baselines: W_Q var@1289 = 0.99, W_K var@1618 = 0.99.
6. Factored inference: replace W_Q^L with U_M[:, :K] @ diag(S_M[:K]^{1/2}) and W_K^L with Vt_M[:K, :].T @ diag(S_M[:K]^{1/2}), for K = 350, 400.
7. Eval ΔNLL on 455-token held-out at K=350 and K=400 for pilot 3 layers.
8. If pilot GO (see threshold below): repeat for all 28 layers; run full-model ΔNLL sweep.

**Sweep parameters**:
- K ∈ {256, 300, 350, 400, 450, 500} (variance profile, pure measurement).
- K = 350, 400 (ΔNLL measurement).
- 3 pilot layers; 28 full layers on GO.

**Output path**: `experiments/stage0/ladder_v2/round9_mqkp/`
- `mqkp_variance_profile.json` — var@K per layer for K ∈ {256, 300, 350, 400, 450, 500}.
- `mqkp_nll_pilot.json` — ΔNLL at K=350,400 for pilot 3 layers.
- `mqkp_nll_full.json` — ΔNLL at best K across 28 layers (if pilot GO).

**Script**: `scripts/stage0_mqkp_pilot.py` (new script needed).

Description: Load Qwen3-1.7B-Base bf16 with `transformers.AutoModelForCausalLM`. For each target layer, extract W_Q and W_K weight tensors (detach, to CPU numpy float32). Compute M = W_Q @ W_K.T. Full SVD via `np.linalg.svd`. Record variance profile. For the ΔNLL eval: create a modified model where W_Q and W_K for the target layers are replaced with the factored form (two new nn.Linear layers each of shape d × K, initialized from the SVD factors). Run tokenized WikiText-2 through the modified model; compute per-token NLL; return mean ΔNLL vs unmodified model.

**Wall-clock**: 3 SVDs × ~1 min + 3 ΔNLL evals × ~3 min = ~15 min pilot. Full 28-layer: 28 SVDs × ~1 min + full eval ≈ ~45 min total. Well under 8-hour gate.

### 5. Go threshold

**Pilot GO**: var@350(M^L) ≥ 0.99 in all 3 pilot layers AND ΔNLL ≤ +1.0 nat at K=350 in pilot.

**Full GO**: var@350 ≥ 0.99 mean across all 28 layers AND ΔNLL ≤ +1.0 nat at K=350 on full 28-layer model.

**SOFT GO**: var@400 ≥ 0.99 (K=350 marginal) AND ΔNLL ≤ +1.0 nat at K=400. Records as SOFT-GO; motivates follow-up at K=380.

Threshold rationale: ΔNLL ≤ +1.0 nat mirrors CF11's K=128 global GO threshold (+0.98 nats), ensuring the product-SVD at K=350 achieves the same or better quality as independent W_Q compression at K=128, while jointly compressing both matrices.

### 6. No-go threshold

**NO-GO**: var@400(M^L) < 0.95 in ≥ 2 of 3 pilot layers OR ΔNLL > +2.0 nats at K=400 in pilot.

**Class-level kill**: NO-GO ⇒ M = W_Q W_K^T has spectrum no more concentrated than W_Q alone. The product operator inherits the individual matrices' spectra without concentration benefit — the algebraic composition of two concentrated spectra does NOT produce a more concentrated product spectrum when the singular spaces are misaligned. This is a class-level finding: any future "find a more compact W_Q W_K^T representation" proposal dies here. CF candidate for CF13: "Attention score operator M inherits the factor matrices' spectra without super-concentration."

### 7. Ambiguous-zone follow-up

**GRAY zone**: var@400 between 0.95 and 0.99, OR ΔNLL at K=400 between +1.0 and +2.0 nats.

Follow-up: run the full 28-layer sweep at K ∈ {400, 450, 500, 600, 800} to locate the K achieving ΔNLL ≤ +1.0 nat. If that K ≤ 800 (minimum 2.56× compression on the Q+K block), record as SOFT-GO with a revised compression factor. If no K ≤ 1024 achieves ΔNLL ≤ +1.0 nat, demote to NO-GO. Additional follow-up runtime: ~30 min. What it resolves: whether the product SVD provides ANY compression benefit over independent compression at matched quality — the key structural question.

### 8. Kill criteria (Stage 6 amendment slot)

Stage 6 should reject this pick if:
- Hidden prior art: a paper is found that explicitly takes SVD of M = W_Q W_K^T (not concatenation, not independent factor SVD) as a compression primitive.
- CF9 frame error: the product M = W_Q W_K^T is claimed to be a principled target because attention scores are xW_Q W_K^T x^T — but if batched multi-head attention applies per-head weight matrices (shape d_head × d_model rather than d_model × d_model), the M construction must be per-head, not full-dimension. Stage 6 must verify the W_Q shape used is the full-dimension (2048×2048) multi-head projection matrix, not the per-head slice.
- Calibration ill-conditioning: not applicable (no calibration data).
- Skeptic-controls insufficiency: see section 9 below.

### 9. Skeptic-controls declaration

The hypothesis claims:
- "Product-SVD compresses W_Q/W_K more effectively than independent SVD" (effectiveness claim, not a transfer/consistency claim).
- "ΔNLL at K=350 is ≤ +1.0 nats" (generalization from 3 pilot layers to 28-layer model).

**Permutation control**: Applicable. Run the same M = W_Q @ W_K.T product SVD but with W_K rows randomly permuted before multiplication (destroying the learned alignment between W_Q column space and W_K column space). GO threshold must be met by real M AND real ΔNLL must be substantially better than permuted M ΔNLL. Target gap: real ΔNLL at K=350 ≤ +1.0 nat AND permuted-M ΔNLL at K=350 ≥ +3.0 nats (3× worse). This verifies the concentration is a property of the trained weight alignment, not a structural artifact of any 2048×2048 real matrix product. Runtime: ~5 min additional (one extra SVD of permuted M at pilot layers; no additional eval needed — the permuted M's ΔNLL can be predicted from the permuted variance profile).

**Random-init control**: Applicable. Compute M = W_Q_rand @ W_K_rand.T for randomly initialized W_Q and W_K (Kaiming uniform, same shape). Measure var@350 of the random-init M. GO threshold: trained-model var@350 ≥ 0.99 AND random-init var@350 ≤ 0.80. This verifies that the spectrum concentration is a learned property, not an architectural constant of 2048×2048 matrix products. Random-init W_Q and W_K can be generated with `torch.nn.init.kaiming_uniform_` in < 1 min. The SVD of random-init M is approximately the Marchenko-Pastur distribution (flat); if var@350 ≈ 350/2048 ≈ 0.17, the random-init baseline is clearly distinguished.

**Multi-seed**: Applicable for the eval ΔNLL. The calibration/eval is deterministic (WikiText-2 held-out, fixed tokenization, no random sampling). However, the product M spectrum is invariant to seed (purely weight-derived). Multi-seed in the sense of "3 non-overlapping 455-token eval windows" is feasible: run ΔNLL eval on tokens [0:455], [455:910], [910:1365] of WikiText-2 test. Report mean ± std across 3 windows. GO threshold must be met by mean ΔNLL AND worst-of-3 must not exceed NO-GO (+2.0 nats). Runtime: +10 min for the additional two eval windows.

Total skeptic-controls overhead: ~15 min additional on top of the 15-min pilot = 30 min total pilot with controls. Still well under 8h gate.

### 10. Runtime estimate

| Phase | Duration |
|---|---|
| Setup: model load + script init | 3 min |
| Pilot: 3 layers × (M compute + SVD + variance) | 5 min |
| Pilot: permutation control (1 SVD per pilot layer) | 2 min |
| Pilot: random-init control (1 SVD per random-init layer) | 2 min |
| Pilot ΔNLL eval at K=350 and K=400 | 6 min |
| Pilot ΔNLL multi-seed (2 additional windows) | 8 min |
| Total pilot with controls | ~26 min |
| Full 28-layer sweep (if pilot GO): SVDs | 28 min |
| Full 28-layer ΔNLL eval | 15 min |
| Full 28-layer multi-seed (2 additional windows) | 16 min |
| Total full run with controls | ~85 min |

**End-to-end wall-clock (pilot + full on GO): ≤ 2 hours.** Passes 8-hour gate by large margin.

### 11. Script identification

New script needed: `scripts/stage0_mqkp_pilot.py`

Inputs: Qwen3-1.7B-Base path, list of pilot layer indices, K sweep values, output directory.
Outputs: variance profile JSON, ΔNLL JSON for each K, permutation-control variance profile, random-init variance profile, multi-seed ΔNLL stats.
Load-bearing computation: `M = W_Q.float() @ W_K.float().T` followed by `np.linalg.svd(M.numpy(), full_matrices=False)`. Factored inference: patched `forward()` replacing W_Q and W_K with the two low-rank factors.

### 12. Downstream implications

**GO unlocks**:
- S1-MQKP-ZONEMAP (Stage 4): zone-map-indexed KV scan over product-SVD projected cache — depends on MQKP GO as prerequisite.
- S4-ATTN-CSE (Stage 4): compiler CSE across W_Q/W_K/W_V — MQKP GO confirms the product compression validity; ATTN-CSE extends to the input-projection CSE angle.
- S7-INT8-LOOKUP-SOFTMAX (Stage 4): INT8 Q/K after product-SVD factorization — direct cascade from MQKP.
- F4-WQKRS (WQKRS water-filling): MQKP GO establishes that product compression dominates independent; WQKRS's water-filling is then a refinement within product-SVD space.
- v2-CF1 generalization to M spectrum: if M concentrates faster than W_Q, this would add a new boundary to the CF8/CF11 attention-weight characterization.

**NO-GO kills**:
- S1-MQKP-ZONEMAP — requires MQKP GO, dies trivially.
- S7-INT8-LOOKUP-SOFTMAX — the compactness argument (128-dim projected Q/K vectors for INT8) requires product-SVD concentration; dies on NO-GO.
- Any future "find a product form of Q/K that is more compressed than independent Q/K" class — class-level kill.
- Does NOT kill WQKRS (water-filling within independent factors survives), ATQKV (W_V/W_O are independent questions), or RAOK/SNRATE (activation and quantization paths are orthogonal).

### 13. Provenance

- **Originating orientation**: F (First-Principles). Run 009 Stage 1, F orientation ideator.
- **Convergence cluster**: C1 (5-orientation convergence: R9-R1, F1-MQKP, F4-WQKRS, C13-EQSUB, R9-R3). Strongest C1 representative (highest Stage 2 score, cheapest settler, gates the most downstream ideas).
- **Stage 4 gap-idea slot**: Not originating from Stage 4; Stage 4 confirmed MQKP as primary recommendation and built 4 dependent ideas (S1, S4, S7, S3 indirectly) on top of MQKP GO.
- **Frame-novelty bonus path**: Not applicable (MQKP advances on convergence strength and A4=3 algebraic-identity, not A2=3 frame-novelty).
- **Runner-up**: R9-R1/ATQKV (score 25.3). Stage 6 kill → advance ATQKV without re-running Stage 5.

---

## Track A Pick — Full Experiment Plan

### 1. Title and one-line description

**Round 9 / Track A — F3-L1GNF: Layer-1 Gate Null-Space Fold**

Is the CF6 Layer-1 gate-folding anomaly (36.1% foldable neurons) explained by those neurons' W_gate rows lying in the null space of the Layer-1 input covariance, making the fold both mechanistically predictable and quality-neutral?

### 2. Class tags

`arch-transposition` / `activation-removal`
Elegance class: `null-space-alignment` (constructive: fold eligibility is determined by an auditable eigendecomposition of the input covariance; the AERO fold identity is exact for truly null-space-aligned rows).

### 3. Hypothesis

**H1 (load-bearing)**: W_gate^{L1} rows that CF6 classifies as foldable (std(silu(W_gate^{L1}·x)) < 0.05) align significantly with the null space of the Layer-1 input covariance C_x^0 = E[x^{L1} x^{L1,T}] — specifically, their null-space projection fraction ≥ 0.70 while non-foldable rows project ≤ 0.30.

**H2 (downstream)**: Applying AERO-style fold (replace the foldable neuron's gate computation with its mean-field constant c_i = mean(silu(W_gate^{L1}·x)_i) ∈ ℝ, precomputed from calibration) produces ΔNLL ≤ 0.05 nats on held-out — confirming the fold is quality-neutral.

H1 depends on: CF6 (36.1% Layer-1 anomaly confirmed), the AERO fold identity (exact for constant gate output). H1 is falsifiable in 25 min from the eigendecomposition check alone. H2 is downstream of H1 GO.

### 4. Smallest test

**Model**: Qwen3-1.7B-Base, bf16.
**Hardware**: Ryzen 5 7530U, 7.28 GiB RAM.

**Calibration corpus**: WikiText-2 train split, 500 tokens (more than CF10's requirement; ensures stable mean estimates for all 2218 foldable neurons). Loader: `datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")[:500]`.

**Eval corpus**: WikiText-2 test split, 455 tokens (held-out, same as all v2 ladder evaluations).

**Layers/matrices touched**: W_gate at Layer 1 only. All other layers and matrices untouched.

**Protocol**:
1. Run 500-token calibration forward pass; collect Layer-1 input activations x^{L1} ∈ ℝ^{500 × 2048}.
2. Compute C_x^0 = x^{L1,T} @ x^{L1} / 500 (shape 2048×2048).
3. Eigendecompose C_x^0; identify null space as eigenvectors with eigenvalue < 1e-5 (let null_dim = count of such eigenvectors; expected nonzero due to token diversity = most eigenvectors live in low-eigenvalue region).
4. For each of the 6144 Layer-1 gate neurons: compute null-space projection fraction = |P_null W_gate_row|² / |W_gate_row|² where P_null projects onto the null eigenspace.
5. Compare projection fractions: foldable neurons (CF6: std < 0.05) vs non-foldable neurons (std ≥ 0.05). Report mean ± std of projection fraction for each group. Primary test: Mann-Whitney U test or simple threshold comparison.
6. Apply fold to foldable neurons: for each foldable neuron i, compute c_i = mean(silu(W_gate^{L1}·x)_i) over 500 calibration tokens. Replace the gate projection for neuron i with the constant c_i (set W_gate_row_i = 0, add bias c_i to the SwiGLU output for that neuron).
7. Eval ΔNLL on 455-token held-out.

**Sweep parameters**:
- Eigenvalue threshold for null space: {1e-5, 1e-4, 1e-3} (primary = 1e-5).
- Null-space rank K_null: measure projection at varying K_null ∈ {64, 128, 256, 512} to find where the foldable/non-foldable gap is sharpest.

**Output path**: `experiments/stage0/ladder_v2/round9_l1gnf/`
- `null_space_projection_profile.json` — per-neuron null-space projection fractions, foldable/non-foldable split.
- `fold_nll.json` — ΔNLL before and after fold, per-neuron constant estimates.

**Script**: `scripts/stage0_l1gnf_pilot.py` (new script needed).

Description: Load Qwen3-1.7B-Base. Hook Layer-1 input activations. Compute covariance, eigendecompose, extract null space. For each gate neuron: compute null-space projection fraction. Identify foldable neurons via CF6 criterion (std < 0.05) independently. Compare groups. Apply fold (modify gate bias; zero out gate weight rows). Eval on held-out.

**Wall-clock estimate**: 5 min setup + 5 min calibration pass + 5 min eigendecompose (2048×2048) + 5 min projection analysis + 5 min eval = ~25 min total. Passes 8-hour gate.

### 5. Go threshold

**Primary GO**: mean null-space projection fraction for foldable neurons ≥ 0.70 AND for non-foldable neurons ≤ 0.30 AND effect size (fold_mean − nonfold_mean) ≥ 0.30.

**Quality GO**: ΔNLL ≤ +0.05 nats on held-out after fold.

Both must hold for full GO. The null-space criterion (primary GO) is the structural claim; the ΔNLL (quality GO) confirms the fold is operationally viable.

### 6. No-go threshold

**NO-GO**: mean null-space projection fraction for foldable neurons ≤ 0.40 (indistinguishable from non-foldable neurons).

**Class-level kill**: NO-GO ⇒ CF6's Layer-1 anomaly is NOT explained by input null-space geometry; it is a calibration-distribution artifact (the specific WikiText-2 calibration tokens happen to avoid those neurons' activation directions, not a structural property of the embedding space). This closes the "geometric fold predictor" idea class: null-space alignment does not predict foldability in trained Qwen3 Layer-1. Implication for future rounds: any proposal that tries to derive fold eligibility from weight-geometry alone (without calibration data) is undermined by this NO-GO.

**Quality NO-GO**: ΔNLL > +0.30 nats after fold → the constant mean-field approximation is too coarse; foldable neurons are not truly constant even when their variance is below the CF6 threshold on the calibration set. This constrains the deployment viability of CF6-style folding even if H1 holds structurally.

### 7. Ambiguous-zone follow-up

**GRAY zone**: null-space projection fraction 0.40–0.70 for foldable neurons.

Follow-up: Test at higher null-space rank (K_null = 512 instead of default). If the 0.70 threshold is met at K_null = 512 (a larger null space definition), the null-space alignment is real but the effective null space is larger than expected — indicating that the input activations at Layer 1 are more constrained (lower effective rank) than a coarse threshold suggests. Additional runtime: ~10 min. Resolves: whether the null-space alignment holds at a broader definition of "null" (low-eigenvalue vs strictly zero).

### 8. Kill criteria (Stage 6 amendment slot)

Stage 6 should reject if:
- Prior art: a paper is found that explicitly uses null(C_x^0) alignment to predict SwiGLU gate-fold eligibility in transformer models.
- CF10 re-examination: if 500-token calibration is insufficient for stable c_i estimates for all 2218 foldable neurons (recall CF10 check passed for mean estimation, but Stage 6 should verify std(c_i estimate) < 0.01 for the gate constants to confirm they are operationally stable).
- AERO pre-emption check: AERO (arXiv:2410.13060) removes activations globally WITH retraining. If Stage 6 finds an AERO variant that operates post-training on specific neurons with a null-space criterion, this pick is pre-empted.

### 9. Skeptic-controls declaration

The experiment claims "the fold is quality-neutral" (H2) and "null-space alignment explains foldability" (H1).

**Permutation control**: Applicable for H1. Randomly permute the W_gate row assignments between foldable and non-foldable neurons (i.e., ask: does any random subset of 36.1% of neurons show the same null-space projection fraction?). Run the same null-space projection analysis on 100 randomly drawn subsets of 2218 neurons (same count as CF6-foldable). GO threshold for H1 must be met by the foldable-neuron group AND the foldable group's mean projection fraction must exceed the random-subset 95th percentile. Runtime: ~3 min (100 random subsets × projection computation is vectorizable). This rules out "any 36% of neurons trivially has high null-space projection due to general W_gate structure."

**Random-init control**: Applicable for H1. Compute null-space projection fractions for W_gate rows in a randomly initialized Qwen3-1.7B-Base (Kaiming uniform W_gate, same architecture). If random-init W_gate rows also show high null-space projection (> 0.50 for 36% of neurons), the effect is architectural, not learned. Expected: random-init gate rows should have null-space projection ≈ null_dim / 2048 (uniform distribution across directions). Runtime: ~5 min (no forward pass needed; only eigendecompose C_x^0 from trained model and project random-init rows). The trained model's C_x^0 is kept fixed; only the W_gate rows are replaced with random-init values. If trained rows show much higher null-space alignment than random-init rows (trained mean ≥ 0.70 vs random-init mean ≤ 0.30), the effect is learned.

**Multi-seed**: Applicable for H2 (ΔNLL eval). Run ΔNLL eval on three non-overlapping WikiText-2 test windows ([0:455], [455:910], [910:1365]). Report mean ± std. GO threshold (ΔNLL ≤ +0.05 nats) must be met by mean ΔNLL; worst-of-3 must not exceed +0.30 nats (the quality NO-GO threshold). Runtime: +10 min.

Total skeptic-controls overhead: ~18 min additional → total pilot runtime ~43 min. Still well under 8-hour gate.

### 10. Runtime estimate

| Phase | Duration |
|---|---|
| Setup: model load | 3 min |
| Calibration forward pass (500 tokens) | 5 min |
| Covariance eigendecomposition (2048×2048) | 5 min |
| Null-space projection analysis | 2 min |
| Permutation control (100 random subsets) | 3 min |
| Random-init control (W_gate projection) | 5 min |
| ΔNLL eval, 3 windows | 15 min |
| Post-processing / output | 2 min |
| **Total** | **~40 min** |

End-to-end wall-clock: **≤ 45 min**. Passes 8-hour gate.

### 11. Script identification

New script needed: `scripts/stage0_l1gnf_pilot.py`

Inputs: Qwen3-1.7B-Base path, calibration token count, eigenvalue threshold for null space, K_null sweep values, output directory.
Outputs: null-space projection fraction per neuron (with foldable/non-foldable label), permutation control distribution (100-sample mean projection fractions), random-init control projection fractions, fold constants c_i for all foldable neurons, ΔNLL JSON across 3 eval windows.
Load-bearing computation: `C = activations.T @ activations / n_tokens`; `eigvals, eigvecs = np.linalg.eigh(C.numpy())`; `null_basis = eigvecs[:, eigvals < threshold]`; per-row projection = `np.sum((W_gate_row @ null_basis)**2) / np.sum(W_gate_row**2)`.

### 12. Downstream implications

**GO unlocks**:
- R9-R5/LAYERONE-FOLD (Stage 2 advancer): R9-R5 proposed the Layer-1 fold cascade as a free swing. L1GNF GO gives the mechanistic grounding that makes R9-R5 structurally principled rather than empirical. The null-space criterion becomes the cascade's eligibility gate.
- C14-CFSHIFT (Stage 2 advancer, Track A runner-up): CFSHIFT requires CF6 foldable neurons to have non-negligible c_1 magnitude. L1GNF GO + the ΔNLL quality check (fold quality-neutral) confirms the mean-field approximation is stable — strengthening CFSHIFT's assumption that c_1 is meaningful.
- Future rounds: null-space alignment as a layer-specific compression eligibility criterion can be extended to other layers (Layer 2 has <2.5% foldable per CF6; null-space check would confirm the mechanism is Layer-1-specific or generalizes).
- v2 CF entry: if H1 + H2 both GO, propose CF16: "Layer-1 W_gate fold eligibility (36% of neurons) is explained by null(C_x^0) alignment; null-space projection fraction ≥ 0.70 for foldable neurons vs ≤ 0.30 for non-foldable."

**NO-GO kills**:
- R9-R5/LAYERONE-FOLD as a principled cascade mechanism (it could still be done empirically using CF6 calibration data, but loses the zero-calibration-sample predictor property).
- Any proposal premised on "find geometrically predictable foldable neurons in other layers via null-space alignment" — class-level kill on geometry-only fold prediction.
- Does NOT kill CF6 itself (the 36% fold rate remains empirically confirmed regardless of mechanistic explanation).
- Does NOT kill CFSHIFT (which depends only on CF6 + W_Q linearity, not on the null-space explanation).

### 13. Provenance

- **Originating orientation**: F (First-Principles). Run 009 Stage 1, F orientation ideator.
- **Convergence cluster**: None. L1GNF is a solo advancer (no multi-orientation convergence cluster). Convergence multiplier = 1.0.
- **Stage 4 gap-idea slot**: Not originating from Stage 4. Stage 4 noted L1GNF as a "compiler dead-code elimination" primitive and connected it to CSE analysis but did not propose a new idea based on it.
- **Frame-novelty bonus path**: Not selected on frame-novelty (CFSHIFT holds the A2=3 frame-novelty slot). L1GNF advances on elegance (constructive null-space criterion, A4=3) and NO-GO finding value (+2, CF-candidate).
- **Runner-up**: C14-CFSHIFT (score 17.3). Stage 6 kill → advance CFSHIFT without re-running Stage 5.
