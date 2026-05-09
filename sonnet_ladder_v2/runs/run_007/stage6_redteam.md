# Stage 6 — Red Team — Run 007

Date: 2026-05-09  
Model: claude-sonnet-4-6  
Inputs: stage5_selector (run_007), stage3_deep_research (run_007), SUMMARY.md (CF1–CF12, v2-CF1, v2-CF2), KILL_LIST.md. WebSearch live.

---

## Preamble

Two picks under review:

- **Track A — S3 ATRC**: W_V/W_O Spectrum Sweep (35 min, CF11-metric extension, convergence C1 cluster, NOVEL CF11-metric claim)
- **Track B — KSATT → S7**: Kronecker Smoke + Follow-on (5 + 30–45 min, algebraic-identity elegance-class, novel retroactive Kronecker characterization)

Runner-up assignments per Stage 5: Track A runner-up is S4 (gated on FRCF), Track B runner-up is FRCF (45 min, CF1×CF12 dual-signal neuron precision).

---

## Section 1 — Frame-Mismatch Re-check (CF9)

### Track A — ATRC W_V/W_O Spectrum Sweep

The experiment imports no named theorem. The operations are: per-matrix SVD, cumulative variance fraction, forward-pass ΔNLL measurement. Standard linear algebra throughout. No preconditions to verify.

**GQA shape claim.** Stage 5 section 4 states W_V shape as "8 KV heads × d_head × d_model = 8 × 128 × 2048." The script description correctly notes "reshape per GQA config." This is a structural correctness requirement, not a theorem precondition, but it is load-bearing: if the script treats W_V as a full 2048×2048 matrix rather than 8×128×2048 factored form, the r_99/d measurement is computed on a differently-shaped object than what is actually stored and used at inference. Stage 6 confirms: the GQA shape verification in section 4 ("Verify exact shape from config.json before running") must be the first script operation, not a postcondition. This is an existing kill criterion in section 8 of the plan — correctly scoped.

**WOVR add-on precondition.** The canonical-angle computation (U_O^T V_V via QR + SVD) is standard numerical linear algebra. No precondition at risk. The shapes are: U_O ∈ R^{2048×64}, V_V ∈ R^{128×64} — note the dimension mismatch. U_O is the top-64 LEFT singular vectors of W_O_h ∈ R^{2048×128}; V_V is the top-64 RIGHT singular vectors of W_V_h ∈ R^{128×2048}. The canonical angle computation requires both matrices to be in the same ambient space. Stage 5 plan §4 states "canonical angles between U_O (top-64 left singular vectors of W_O_h) and V_V (top-64 right singular vectors of W_V_h) via QR + SVD of (U_V^T V_O)."

**CF9 FLAG — WOVR dimension precondition.** W_O_h ∈ R^{d_model × d_head} = R^{2048×128}: its left singular vectors live in R^{2048}. W_V_h ∈ R^{d_head × d_model} = R^{128×2048}: its right singular vectors live in R^{2048}. The canonical angle is between U_O (first 64 columns of the 2048×128 U matrix) and V_V (last 64 rows of the 128×2048 V^T matrix, transposed). Both live in R^{2048}. The inner product matrix is (U_O)^T V_V ∈ R^{64×64}. This is dimensionally valid — the ambient space is R^{2048} for both. However, Stage 3 (§1.6) and Stage 5 (§4) write the inner product as "(U_V^T V_O)" — note the subscript mismatch (U_V vs U_O, V_O vs V_V). The label inconsistency between Stage 3 and Stage 5 should be resolved in the script before execution to ensure the correct singular vectors are used. **Amendment required: script must explicitly compute U_O = W_O_h[:,] left SVD, V_V = W_V_h right SVD (not cross-matrix), and form (U_O)^T V_V.**

CF9 verdict for Track A: **PASS** with one WOVR labeling amendment. The core ATRC experiment (W_V/W_O r_99/d sweep) has no precondition to fail. The WOVR add-on has a label inconsistency that could cause the wrong matrix pair to be used.

### Track B — KSATT → S7

**KSATT algebraic-identity claim (section 6c check).** The elegance-class is `algebraic-identity`: if W_Q_head reshaped as (p×q) × (p×q) has a dominant rank-1 structure, then W_Q_head ≈ A ⊗ B (a Kronecker product), which is an exact algebraic factorization. The precondition for this claim: W_Q_head is genuinely well-approximated as A ⊗ B when reshuffled. The smoke test directly measures the fraction of Frobenius norm captured — it is its own precondition check. No external theorem is imported. CF9 clean.

**S7 matched-parameter claim.** The parameter count matching formula in Stage 5 §4 states "K = r×5120/(2×2048) = r×1.25 (approximately)." This is dimensionally loose. Kronecker rank-r at partition (p=32, q=64): r × (32²+64²) = r × (1024+4096) = r × 5120 parameters. Matched SVD rank K: K × (d_model + d_model) / 1 = K × 4096 total (K left singular vectors of d_model dimensions + K right singular vectors of d_model dimensions). So K = r × 5120 / 4096 = r × 1.25. For r=16: Kronecker uses 81,920 params, SVD K=20 uses 20 × 4096 = 81,920. The arithmetic checks out. However, note: this parameter count applies to the GLOBAL W_Q (2048×2048); at per-head (128×2048), the Kronecker partition is different and the parameter savings are correspondingly different. The KSATT test targets PER-HEAD W_Q (128×2048 at p=32, q=64 requires reshaping a 128×2048 matrix as (32×4) × (64×32) — a 128×2048 head has dimensions d_head × d_model = 128 × 2048, so the (p×q) × (p×q) block-matrix reshuffling requires p×q = 128 (input rows) and p×q = 2048 (output columns). With p=32, q=64: p×q = 2048 for the column dimension, but the row dimension needs to be factored too. The per-head W_Q is 128×2048. A Kronecker factorization A ⊗ B with A ∈ R^{a×c}, B ∈ R^{b×d} where ab=128 and cd=2048. At (a=8, b=16, c=32, d=64): A ⊗ B is 128×2048. This is valid. The Stage 5 formulation (p=32, q=64) refers to A ∈ R^{32×32}, B ∈ R^{64×64}: valid only if the matrix being reshaped is square (2048×2048, the global W_Q) not per-head (128×2048).

**CF9 FLAG — reshape ambiguity for per-head KSATT.** Stage 5 §4 says "Matrices: W_Q concatenated global matrix (16 × 128 × 2048 → 2048 × 2048 for global; or per-head 128 × 2048)" and "Layer: 14, heads 0 and 7." But §9 of the same plan says "permutation control: run the same Frobenius fraction measurement on a randomly row-permuted W_Q_head." This implies the test is per-head (128×2048), not global (2048×2048). For per-head 128×2048, the valid Kronecker factorizations at (p=32, q=64) require A ∈ R^{p₁×p₂}, B ∈ R^{q₁×q₂} where p₁q₁=128, p₂q₂=2048. Stage 5's chosen (p=32, q=64) is literally the partition of 2048=32×64 (the column dimension) — the row dimension 128 must be handled separately. The standard VanLoan reshuffling for a rectangular m×n matrix to detect A ⊗ B (A ∈ R^{p×r}, B ∈ R^{q×s}, pq=m, rs=n) is well-defined: reshape W as a (p×q) × (r×s) block matrix and compute its rank. For 128×2048: p=8, q=16, r=32, s=64 (or other factorizations). The (p=32, q=64) partition as written applies to a square 2048×2048 global W_Q. **If the script applies the p=32, q=64 partition to the per-head 128×2048 W_Q_head, it will produce an incorrectly-sized block matrix.** This is a script-correctness requirement that must be resolved before execution. The kill criterion in Stage 5 §8 ("Stage 6 must verify the script handles both the per-head and global reshape correctly") was correctly identified — Stage 6 confirms this must be resolved as a pre-execution amendment.

CF9 verdict for Track B: **PASS** with reshape ambiguity amendment. The core Kronecker-structure claim is mathematically valid; the precondition (that VanLoan reshuffling is applied at the correct matrix shape) must be confirmed in the script before execution.

---

## Section 2 — Calibration Ill-Conditioning Re-check (CF10)

### Track A — ATRC

No calibration fit anywhere in the Track A plan. The experiment is: (a) compute SVD of weight matrices, (b) reconstruct at various ranks, (c) measure ΔNLL on 500 held-out WikiText-2 tokens. The ΔNLL measurement uses the model's own forward pass — no fitted parameters. `n_params_to_fit = 0`. CF10 not applicable. **PASS.**

### Track B — KSATT

Phase 1 (smoke test): pure weight matrix SVD and Frobenius fraction measurement. No calibration fit. CF10 not applicable.

Phase 2 (S7 Kronecker optimization): Fitting A_i, B_i Kronecker factors for each rank r. The fitting procedure for Kronecker factors (VanLoan alternating least squares or equivalent) is a rank-r decomposition, not a calibration-data fit. The factors are computed from the weight matrix alone, not from calibration data. `n_params_to_fit = 0` in the calibration-conditioning sense. CF10 not applicable. **PASS.**

The multi-seed requirement in Phase 2 (section 9 of Stage 5 plan) addresses initialization sensitivity of the Kronecker factor optimization — this is distinct from CF10 ill-conditioning (which concerns calibration data fit underdetermination). The multi-seed protocol is correctly scoped.

---

## Section 3 — Skeptic-Controls Check

### Track A — ATRC

The primary claim: "W_V is CF11-class (r_99/d < 0.80) — a structural characterization claim consistent across layers." This is a "X is consistent across Y" shape → controls required.

Stage 5 §9 declares three controls:

1. **Permutation control**: row-permute W_V, measure r_99/d, confirm permuted ≈ 1.0 and real substantially below. Expected baseline: r_99/d ≈ 1.0 for permuted. Gap must be stated explicitly. ✓ **Present and correctly specified.**

2. **Random-init control**: measure r_99/d on a randomly initialized Qwen3-1.7B W_V. GO requires trained r_99/d substantially below random-init r_99/d. Analogous to v2-CF2 where trained W_Q gap = +0.37 (0.63 trained vs 1.0 random). ✓ **Present and correctly specified.**

3. **Multi-seed**: permutation control with 3 seeds, mean ± std of permuted r_99/d. Confirms permuted baseline is stable. ✓ **Present and correctly specified.**

One concern: Section 10 (runtime estimate) allocates "5 min (single-layer spot checks; full 28-layer if time permits)" for permutation + random-init controls. **AMENDMENT REQUIRED**: The skeptic-controls must run on ALL 28 layers for the permutation control (the GO threshold requires "≥ 22 of 28 layers" meeting the r_99/d criterion — spot checks are insufficient to confirm the gap holds layer-uniformly). A single-layer spot check at layer 14 could pass while other layers fail, producing a misleading result. At ~30 seconds per SVD per matrix, 28-layer permutation control adds ~14 minutes. The revised total is ~52 min, still well within 8-hour ceiling.

Skeptic-controls verdict for Track A: **PASS with amendment** — controls are structurally present; the 5-min spot-check budget must be expanded to full 28-layer permutation control.

### Track B — KSATT

The KSATT claim: "Kronecker structure is consistent across heads" — partial "X is consistent" shape but limited to 2 heads at 1 layer in Phase 1. Phase 2 (if GO) adds the cross-r consistency claim: "Kronecker ΔNLL < SVD ΔNLL at matched parameters for ≥ 3 of 5 r values."

Stage 5 §9 declares three controls:

1. **Permutation control (Phase 1)**: row-permute W_Q_head, confirm permuted Frobenius fraction < 10%. ✓ **Present and correctly specified.** Expected baseline: random matrices have near-zero rank-1 Kronecker fraction (correct expectation). The "substantially exceed" language is appropriate.

2. **Random-init control**: same Kronecker test on randomly initialized W_Q. ✓ **Present and correctly specified.** Note Stage 3 §8.3 explicitly states: "for Kronecker, this is crucial: a 2048×2048 random matrix reshaped as 32×64 × 32×64 has a near-zero rank-1 Kronecker fraction by construction." This is the correct prior expectation.

3. **Multi-seed for S7 (Phase 2)**: 3 random seeds for Kronecker factor optimization, mean ± std ΔNLL. ✓ **Present and correctly specified.**

Skeptic-controls verdict for Track B: **PASS** — all three controls present, correctly scoped to each phase, with explicit gap thresholds.

---

## Section 4 — Hidden Prior-Art Search (Third Pass)

### Track A — ATRC: W_V/W_O Spectral Characterization

**CRITICAL PRIOR-ART DISCOVERY — Spectral Lifecycle (arXiv:2604.22778, April 2026)**

A third-pass search surfaced a paper missed by Stage 5's re-verification: **"The Spectral Lifecycle of Transformer Training: Transient Compression Waves, Persistent Spectral Gradients, and the Q/K–V Asymmetry"** (arXiv:2604.22778, submitted April 2026). This paper presents the **first systematic study of weight matrix singular value spectra during transformer pretraining**, tracking full SVD decompositions of **every weight matrix** — including W_V and W_O — at 25-step intervals across three model scales.

The paper's finding 3 is titled **"Q/K–V Functional Asymmetry"**: it explicitly states that **value/output projections compress uniformly** while query/key projections carry the full depth-dependent spectral dynamics. This is a direct characterization of W_V and W_O spectral behavior.

**Severity assessment:**

The Spectral Lifecycle paper operates on training checkpoints (GPT-2 124M–774M, Pythia 160M–1B), not post-training static weights. It tracks the spectral dynamics *during training*, not the static r_99/d metric on a single post-trained checkpoint. The ATRC experiment measures: on a fully post-trained Qwen3-1.7B-Base, what is r_99/d for W_V and W_O? This is a static structural characterization, not a dynamic training-time measurement.

However, the Spectral Lifecycle finding that W_V/W_O "compress uniformly" has a direct implication for ATRC's GO/NO-GO: if V/O projections compress uniformly across depth (low depth-gradient in their spectra), this is evidence FOR the ATRC GO case (W_V CF11-class, low r_99/d). It does NOT measure the CF11 metric (r_99/d = cumulative variance 99th percentile / d) for post-trained Qwen3-1.7B specifically. The paper validates GPT-2/Pythia family; Qwen3-1.7B's architecture (GQA, SiLU, RMSNorm) is different enough that the quantitative r_99/d must be measured independently.

**Partial pre-emption assessment**: The Spectral Lifecycle confirms the existence of W_V/W_O spectral concentration conceptually (the "uniform compression" finding) but does NOT report:
- The specific CF11-style r_99/d metric (99% cumulative variance rank / d)
- Post-training static values for Qwen3-family models
- ΔNLL consequences of SVD truncation at K ∈ {128, 256, 512}

The ATRC novelty claim survives at the precision of measurement (CF11-style r_99/d + ΔNLL on Qwen3-1.7B-Base post-training), but the qualitative directional claim ("W_V has more concentrated spectrum than MLP weights") is now partially anticipated by arXiv:2604.22778.

**Required action**: Stage 6 declares this a **partial pre-emption** of the qualitative GO claim for Track A. The experiment is NOT killed, but the novelty framing must be sharpened: the contribution is the **quantitative CF11-metric on Qwen3-1.7B-Base** (a different model family, different architecture, static post-training measurement) not the existence of V/O spectral compression in transformers generally.

The biased-framing section (Section 5) will address the consequence for the GO threshold.

**A3 (arXiv:2505.12942)** — confirmed still accurately described. Stage 5's re-verification correctly noted A3 does not report the CF11-style r_99/d per matrix. A3 has been updated to v3 (June 2025) — no evidence it adds r_99/d reporting. Not a new pre-emption.

**Generalized Fisher-Weighted SVD (arXiv:2505.17974, May 2025)** — uses Kronecker-factored Fisher approximation for SVD-based compression of all LLM weight matrices including W_V. This is an engineering compression paper (not a structural characterization paper) and does not report CF11-style r_99/d. Not a pre-emption of ATRC's measurement frame.

### Track B — KSATT: Kronecker Structure in Trained LLM Attention Weights

**Generalized Fisher-Weighted SVD (arXiv:2505.17974, May 2025)** uses Kronecker-factored Fisher approximation. This is NOT the same as testing whether trained W_Q has emergent Kronecker structure. GFWSVD uses Kronecker decomposition of the *Fisher information matrix* (a sensitivity matrix derived from activations), not of the weight matrices themselves. This does not pre-empt KSATT's structural characterization question ("did SGD converge to Kronecker-structured attention weights?").

**"Compressing LLMs with Structured Matrices"** (ACL 2025 Findings, arXiv:2506.02818) — addresses post-training compression with structured matrices including Kronecker products. Stage 3 search found this paper. Stage 5 confirmed no paper tests *retroactive* Kronecker characterization of standard trained LLM attention weights as a structural question (vs. as a compression method with imposed structure). The distinction holds.

**"Identifying Kronecker product factorizations"** (arXiv:2510.25292) — targets binary/sparse matrices. Not applicable.

**SoKA (arXiv:2506.15251)** — Kronecker for PEFT fine-tuning, not post-training characterization.

No new Kronecker-retroactive-characterization paper found in this third pass. KSATT novelty confirmed.

---

## Section 5 — Biased-Framing Audit

### Track A — ATRC

**GO threshold evaluation.** Stage 5 §5: "W_V r_99/d < 0.80 at ≥ 22 of 28 layers AND ΔNLL(W_V, K=256) < 0.50 nats (global simultaneous)." Both must hold.

The Spectral Lifecycle finding (arXiv:2604.22778) suggests W_V/W_O "compress uniformly" — this is a qualitative direction favoring GO. However, "uniform compression" could mean uniformly high concentration (GO) or uniformly low concentration (different kind of uniform). The paper's framing that Q/K carry the "depth-dependent dynamics" while V/O are "uniform" is ambiguous about whether uniform means concentrated or flat. The GO threshold of r_99/d < 0.80 is not gerrymandered: it matches the CF11 boundary calibration (W_K was at 0.79, the boundary itself). The ΔNLL threshold of < 0.50 nats is also calibrated to CF11's W_Q/W_K findings (W_Q at K=256 achieved +0.51 nats — the GO threshold is marginally above this; it is tight, not easy).

**NO-GO claim.** Stage 5 §6: "W_V r_99/d ≥ 0.95 at majority of layers OR ΔNLL(W_V, K=256) ≥ 1.50 nats → extends CF8 to W_V." The structural finding on NO-GO is precise and class-level. Not rhetorical.

**GRAY zone.** The GRAY zone (r_99/d 0.80–0.90) triggers a follow-up (2-hour per-layer K sweep). This is a legitimate ambiguous zone, not a post-hoc spin zone: the boundary between CF11-class and MLP-class passes through this region. No issue.

**Biased framing flag — Spectral Lifecycle implication.** The Stage 5 plan presents the experiment as "resolving an open question." Given arXiv:2604.22778's "uniform compression" finding for W_V/W_O, there is mild motivated framing in treating this as fully open when existing evidence weakly favors GO. The report should acknowledge that the Spectral Lifecycle provides prior evidence that V/O projections have more concentrated spectra than Q/K, making the GO case more likely. **Amendment: Stage 5 plan should note in section 13 (provenance) that arXiv:2604.22778 provides partial prior evidence favoring the GO direction, without fully resolving the CF11-metric quantification.**

### Track B — KSATT

**GO threshold.** "≥ 50% Frobenius at rank-1 Kronecker for any tested partition at layer 14, head 0 OR head 7." The OR condition (either head suffices) is mildly permissive — if one head shows structure and the other does not, this is a partial result. However, the design explicitly requires testing 4 partition shapes, and the GO is any partition × any head. This is not gerrymandered because random matrices are expected to produce < 10% (the permutation baseline establishes this). A 50% threshold with random baseline < 10% is a high bar.

**NO-GO threshold.** "< 20% at all tested partitions for both heads." The gap between GO (50%) and NO-GO (20%) leaves a GRAY zone of 20–50%, which is resolved by testing additional partition shapes. This is a legitimate ambiguous zone with explicit resolution protocol.

**S7 GO threshold.** "Kronecker ΔNLL < SVD ΔNLL at matched parameter count for ≥ 3 of 5 tested r values AND the gap ≥ 0.05 nats." The 0.05-nat gap requirement prevents noise-level differences from claiming GO. Appropriately tight.

No biased framing detected in Track B.

---

## Section 6 — Runtime / Scope Sanity

### Track A — ATRC

Stage 5 §10 estimates 38 min total. Breakdown: 3 (setup) + 28 (SVD 28 layers × 2 matrices) + 5 (eval passes) + 3 (WOVR) + 5 (controls, spot-check) + 2 (post-processing) = 46 min as written, but the spot-check notation implies only 5 min for controls.

With the Section 3 amendment (full 28-layer permutation control), the correct runtime is approximately:
- SVD (28 layers × 2 matrices): ~28 min
- Eval passes (4 K-values × 1 min): ~4 min
- WOVR add-on: ~3 min
- Permutation control (28 layers × W_V, 3 seeds): ~15 min
- Random-init control (28 layers × W_V): ~5 min
- Multi-seed baseline variance: ~2 min
- Setup + post-processing: ~5 min

**Revised total: ~62 min.** Well within the 8-hour ceiling. No scope concern.

The eval corpus (WikiText-2, 500 held-out tokens) is disjoint from the calibration corpus (N/A — no calibration used). CF10 corner case does not apply.

The smallest test (ATRC r_99/d measurement) directly produces the number that decides the GO threshold — it is not a noisy signal requiring follow-up. GO/NO-GO is determined by the first run. The SOFT-GO follow-up (2-hour per-layer K sweep) is correctly scoped as an ambiguous-zone response, not a primary outcome.

### Track B — KSATT → S7

Stage 5 §10: Phase 1 = 9 min (with controls); Phase 2 = 45 min (with 3-seed multi-seed). Total compound = 54 min. The Phase 1 estimate of 9 min includes 4 min for permutation + random-init controls. With the revised understanding that per-head reshape must be correctly handled (Section 1 amendment), add ~5 min for script verification before execution. Revised total: ~64 min. Still within 8-hour ceiling.

Phase 1 produces a binary result (Frobenius fraction) that directly decides the S7 GO threshold. No ambiguity requiring additional signal.

---

## Section 6b — Wildcard Non-Penalization

Track B KSATT is NOT tagged as [FREE SWING] or [OUT-OF-ORIENTATION] — it is a solo frame-novelty advancer (convergence multiplier 1.0, no cluster). Wildcard non-penalization applies only to convergence penalties; all gates apply normally. Track A ATRC has a convergence multiplier of 1.5. Neither pick is a wildcard in the formal sense; section 6b is not applicable.

---

## Section 6c — Elegant-Equivalence Rejection Ground

Track B KSATT carries `algebraic-identity` elegance-class. Section 6c requires verification that the algebraic-identity argument does not depend on a precondition that fails the CF9 check.

The identity claim: if W_Q_head ≈ A ⊗ B (Kronecker product), then storage = A params + B params << d_head × d_model. This is an exact identity for Kronecker-structured matrices — no approximation.

The precondition: W_Q_head actually IS approximately Kronecker-structured (≥ 50% Frobenius at rank-1 VanLoan reshuffling). This precondition is NOT assumed in advance — it is the smoke test's result. The elegance multiplier at Stage 5 was applied conditionally on KSATT GO. The Section 1 CF9 flag (reshape ambiguity) identified that the precondition could be falsely measured if the wrong matrix shape is used in the VanLoan reshuffling.

**Section 6c verdict**: The elegance argument's precondition is structurally sound IF the reshape is implemented correctly. The reshape ambiguity amendment (Section 1) is the gate. **If the script applies (p=32, q=64) to the global 2048×2048 W_Q (not per-head), the smoke test measures a different object than what S7 would then compress (per-head weights). The elegance identity would apply to the global matrix, but inference compression operates per-head. This is not a math error in the algebraic identity itself, but a scope mismatch: confirming Kronecker structure in global W_Q does not imply per-head Kronecker structure.** Amendment: KSATT script must consistently test EITHER global W_Q OR per-head W_Q, and S7's compression comparison must use the same granularity.

---

## Section 7 — CF13–CF15 Verification Check

Neither Track A nor Track B cites CF13, CF14, or CF15 in any load-bearing way. Track A §3 references "CF11 (confirmed W_Q/W_K) and CF8 (MLP full-rank boundary)" only. Track B §3 references "CF11 (W_Q is compressible)" only. No CF13–CF15 dependency. **PASS.**

---

## Section 8 — Verdict

### Track A — S3 ATRC

**ACCEPT-WITH-AMENDMENTS**

The experiment plan is structurally sound. No CF9 frame-mismatch, no CF10 ill-conditioning, skeptic-controls present, no fatal prior-art pre-emption. The Spectral Lifecycle (arXiv:2604.22778) is a partial prior-art discovery that weakly favors the GO direction but does not pre-empt the CF11-metric quantification on Qwen3-1.7B-Base.

Amendments:

1. **[WOVR labeling fix]** Script must compute U_O as left singular vectors of W_O_h (d_model × d_head) and V_V as right singular vectors of W_V_h (d_head × d_model); explicitly verify subscript consistency before execution. Inner product is (U_O)^T V_V where both live in R^{d_model}.

2. **[Permutation control scope]** Expand permutation control from "single-layer spot check" to full 28-layer W_V permutation sweep (3 seeds, mean ± std). Runtime addition: ~15 min. Required because GO threshold is layer-count gated (≥ 22 of 28 layers).

3. **[Provenance amendment]** Note in section 13: arXiv:2604.22778 (April 2026, "Spectral Lifecycle") provides prior qualitative evidence that V/O projections compress more uniformly than Q/K projections in trained transformers. ATRC contributes the quantitative CF11-metric (r_99/d + ΔNLL) on Qwen3-1.7B-Base, which is the distinct measurement gap.

### Track B — KSATT → S7

**ACCEPT-WITH-AMENDMENTS**

The Kronecker retroactive-characterization frame is confirmed novel. No pre-empting paper found on third pass. CF9 clean for the core algebraic identity. CF10 not applicable.

Amendments:

1. **[Reshape scope consistency]** KSATT script must decide upfront: test global W_Q (2048×2048) OR per-head W_Q (128×2048). If per-head: use valid non-square VanLoan partitions (e.g., a=8, b=16, c=32, d=64 for a 128×2048 matrix — A ∈ R^{8×32}, B ∈ R^{16×64}). If global: S7 compression comparison must operate at global granularity. The (p=32, q=64) partition as stated applies only to the global 2048×2048 W_Q. Stage 5 §8 notes this as a kill criterion; Stage 6 elevates it to a **mandatory pre-execution amendment** that must be resolved in the script before any run.

2. **[Section 9 controls runtime]** Phase 1 runtime declared as "9 min including controls." For a rigorous permutation baseline, run permutation at layer 14 on both heads 0 and 7. This is within the stated 9 min if SVDs are fast (~0.5s each at 2048×2048). Confirmed feasible.

3. **[S7 multi-seed scope]** If KSATT GO triggers at Phase 2: the 3-seed multi-seed requirement applies to the Kronecker factor optimization (fitting A_i, B_i per rank r). The GO threshold requires mean ΔNLL < SVD ΔNLL AND worst-of-3 not worse than SVD by more than 0.10 nats. Confirm this threshold is preserved from §9 into the script's reporting logic.

---

## Prior-Art Summary (for pipeline record)

| Paper | arXiv ID | Relevance |
|---|---|---|
| Spectral Lifecycle | 2604.22778 (Apr 2026) | **NEW** — W_V/W_O Q/K-V asymmetry; partial pre-emption of ATRC directional claim; CF11-metric on Qwen3 survives |
| A3 Low-Rank Attention | 2505.12942v3 (Jun 2025) | Confirmed prior — engineering compression pre-empted; measurement frame survives |
| GFWSVD Kronecker Fisher | 2505.17974 (May 2025) | New find — Kronecker of Fisher matrix, not weight matrix; does not pre-empt KSATT |
| Compressing LLMs Structured Matrices | 2506.02818 (ACL 2025) | Known — training-imposed structure; does not pre-empt retroactive characterization |

---

## Red-Team Summary

**Track A ATRC**: ACCEPT-WITH-AMENDMENTS. Three amendments (WOVR label fix, permutation control scope expansion, provenance note for arXiv:2604.22778). No fatal flaw. The Spectral Lifecycle paper is a significant find that should be cited in the experiment report but does not kill the CF11-metric measurement.

**Track B KSATT→S7**: ACCEPT-WITH-AMENDMENTS. One critical pre-execution amendment (reshape scope consistency for VanLoan partitioning) plus two housekeeping amendments. Frame-novelty confirmed intact on third pass. Elegance-class algebraic-identity survives section 6c check conditional on reshape correctness.

**Runner-up status**: S4 (Track A, gated on FRCF) and FRCF (Track B runner-up) are unaffected by this review. KSATT NO-GO at execution time would trigger FRCF as designed.

---

Sources:
- [The Spectral Lifecycle of Transformer Training: Q/K–V Asymmetry](https://arxiv.org/abs/2604.22778)
- [A3: Analytical Low-Rank Approximation Framework for Attention](https://arxiv.org/abs/2505.12942)
- [Generalized Fisher-Weighted SVD: Kronecker-Factored Compression](https://arxiv.org/abs/2505.17974)
- [Compressing LLMs with Structured Matrices (ACL 2025)](https://arxiv.org/pdf/2506.02818)
- [Predicting LLM Compression Degradation from Spectral Statistics](https://arxiv.org/pdf/2604.18085)
