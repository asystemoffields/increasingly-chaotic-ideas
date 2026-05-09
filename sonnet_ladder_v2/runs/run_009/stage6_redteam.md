# Stage 6 — Red Team — Run 009 (v2 Sonnet Ladder)

Completed: 2026-05-09. Adversarial review of Stage 5 Track A pick (F3-L1GNF) and Track B pick
(F1-MQKP). Fresh prior-art WebSearch run at red-team time.
Runner-up: Track A → C14-CFSHIFT; Track B → R9-R1/ATQKV.

---

## TRACK B — F1-MQKP (Attention Product M = W_Q W_K^T Spectral Compression)

`elegance-class: algebraic-identity` | 15-min pilot + controls | highest run score 31.25

---

### 1. Frame-Mismatch Re-check (CF9 — second pair of eyes)

**Load-bearing imported machinery:** Eckart-Young theorem applied to M = W_Q W_K^T.

**Precondition audit:**

Eckart-Young states: the best rank-r approximation of a real matrix M in Frobenius norm is its
rank-r truncated SVD. Precondition: M is a real matrix. M = W_Q W_K^T is real by construction.
The precondition holds trivially. **PASS.**

**M3 cross-term bound verification (Stage 3 claim: product-SVD strictly beats per-factor SVD):**

The claim is: if W_Q = U_Q Σ_Q V_Q^T and W_K = U_K Σ_K V_K^T independently, then
‖M − W̃_Q W̃_K^T‖_F ≥ ‖M − M̃‖_F at matched rank r. This holds because M̃ (rank-r SVD of M)
is the global minimizer over all rank-r matrices of ‖M − ·‖_F, including factored-form
candidates. The factored-form W̃_Q W̃_K^T is itself rank-r, so it cannot beat M̃. **PASS.**

**Inference-form correctness (Section 4, Step 6):**

The plan replaces W_Q with U_M[:, :K] @ diag(S_M[:K]^{1/2}) and W_K with Vt_M[:K, :].T @
diag(S_M[:K]^{1/2}). Then xW_Q (W_K^T x^T) → x(U Σ^{1/2})(Σ^{1/2} V^T x^T) = xU Σ V^T x^T =
x M̃ x^T. Correct reconstruction. **PASS.**

**RoPE interaction:** RoPE rotates q_h and k_h after the projection. The factored W_Q produces
the same pre-RoPE query vectors as the original W_Q, so RoPE applies identically. Stage 3
confirmed this. Independent re-verification: the plan's factored form computes the same d-dim
query and key vectors (equal linear maps) as the original, then RoPE is applied. No RoPE
incompatibility. **PASS.**

**CF9 verdict: PASS.** No frame-mismatch detected. The Eckart-Young import is trivially valid.

---

### 2. Calibration Ill-conditioning Re-check (CF10)

No calibration data required. M^L = W_Q @ W_K.T is computed entirely from trained weights.
n_params_to_fit = 0. CF10 does not apply. **SAFE.**

---

### 3. Skeptic-Controls Check

**Hypothesis shape:** H1 claims M has r_99/d < min(r_99(W_Q)/d, r_99(W_K)/d) ≈ 0.63 — a
structural MEASUREMENT claim. H2 claims ΔNLL ≤ +1.0 nat at K=350 — a GENERALIZATION claim
(from 3 pilot layers to 28-layer model). The skeptic-controls gate applies to H2.

**Permutation control:** Present and well-specified (Section 9). Permuted W_K rows before product;
target gap real ΔNLL ≤ +1.0 nat AND permuted ΔNLL ≥ +3.0 nats at K=350. The gap threshold of
≥ +2.0 nats (3× difference) is sufficiently large that post-hoc spin is not possible within the
GRAY zone. **ADEQUATE.** No amendment needed.

**Random-init control:** Present and well-specified (Section 9). Target: trained var@350 ≥ 0.99
AND random-init var@350 ≤ 0.80. Expected random-init var@350 ≈ 350/2048 ≈ 0.17 (Marchenko-Pastur
flat spectrum). The gap is large and the control is structurally valid. **ADEQUATE.**

**Multi-seed (three non-overlapping eval windows):** Present and well-specified (Section 9).
Three windows [0:455], [455:910], [910:1365] of WikiText-2 test. GO requires mean ΔNLL ≤ +1.0
nat AND worst-of-3 ≤ +2.0 nats. **ADEQUATE.**

**One sharpening needed:** The permutation control's ΔNLL gap target (≥ +2.0 nats) is specified
for K=350, but the plan only mentions measuring permuted-M variance profile (Section 9: "no
additional eval needed — the permuted M's ΔNLL can be predicted from the permuted variance
profile"). This is an inconsistency: the GO threshold specifies a gap in actual ΔNLL, but the
plan says ΔNLL is not measured for permuted M. Either (a) explicitly measure permuted-M ΔNLL
at K=350 on at least 1 pilot layer (adds ~3 min), OR (b) change the GO threshold to specify the
gap in terms of variance (permuted var@350 < 0.90, verifiable from the SVD measurement alone).
The current plan contradicts itself on this point.

**Stage 6 Amendment B1 (skeptic-controls gap consistency):** Either run the permuted-M ΔNLL
eval on 1 pilot layer (3 min overhead, preferred) and verify real ΔNLL − permuted ΔNLL ≥ 2.0
nats, OR retarget the permutation gap threshold to var@350(permuted) ≤ 0.90 (measurable without
an additional eval run). Document which option is chosen in Section 9.

**Skeptic-controls verdict:** ADEQUATE with Amendment B1.

---

### 4. Hidden Prior-Art Search — Third Pass (2026-05-09)

**Search 1:** "SVD product W_Q W_K attention score operator compression LLM 2025 2026"

**Search 2:** "A3 analytical low-rank approximation attention QK component Theorem 2 attention
score W_Q W_K product 2505.12942"

**CRITICAL PRIOR ART FINDING:**

**A3 (arXiv:2505.12942, v1 May 19 2025, v3 June 25 2025) — "A³: an Analytical Low-Rank
Approximation Framework for Attention"** by Wong et al., NeurIPS 2025.

A3's Theorem 2 (A3-QK) explicitly computes SVD of the fused matrix W_qk,i := W_q,i W_k,i^T
as the compression primitive for the QK component. Per the paper: "Theorem 2 decomposes the
fused weight matrix W̃_qk,i := W_q,i W_k,i^T, which represents the combined query-key
transformation for computing attention scores." The optimization target is functional error on
pre-softmax attention scores (not Frobenius weight error), using weighted SVD on the scaled
matrix R^{1/2}_{XX} W_qk,i R^{1/2}_{XX} (where R terms are input autocorrelation matrices).

**This is structurally the same primitive as MQKP's M = W_Q W_K^T.** A3 forms the same product,
applies SVD to it, and decomposes the result into factored W_q and W_k replacements at inference.

**Distinctions to assess:**

1. **Weighting:** A3 uses activation-covariance-weighted SVD (scale by R^{1/2}_{XX}); MQKP uses
   plain Frobenius SVD of M (no calibration weighting). MQKP's proposal is the no-calibration
   version; A3 is the calibration-weighted version. This is a difference in degree, not in the
   load-bearing primitive (both take the product M as the target).

2. **RoPE handling:** A3's Theorem 2 (QK) applies only to NoPE (no positional encoding). For
   RoPE, A3 switches to CUR approximation on the same product. Qwen3-1.7B uses RoPE. Therefore
   A3's closed-form SVD theorem does NOT directly apply to Qwen3; A3's own RoPE extension is
   CUR-based. MQKP proposes plain SVD on the product for RoPE — which A3 does not do.

3. **No-calibration claim:** MQKP's claim that no calibration is required (the product M is
   computed from weights alone) IS differentiated from A3's calibration-weighted approach.
   However, the NO-CALIBRATION variant is strictly weaker (Eckart-Young-optimal on M without
   input weighting) — it is a simplified version of A3, not a distinct mechanism.

**Pre-emption verdict:**

A3's Theorem 2 covers ≥ 2/3 of MQKP's load-bearing claims:
- M1 (M = W_Q W_K^T as compression target): **PRE-EMPTED** — A3 computes exactly this product.
- M2 (Eckart-Young applied to M): **PRE-EMPTED** — A3's Theorem 2 is a generalized version
  (activation-weighted SVD; plain Eckart-Young is the unweighted special case when R=I).
- M3 (product-SVD beats per-factor SVD): **PRE-EMPTED** — A3's theoretical motivation is
  identical: minimizing functional attention-score error by operating on the product M rather
  than the individual factors.

What MQKP adds that A3 does not: the empirical measurement of M's r_99/d on Qwen3-1.7B-Base
(whether M concentrates faster than W_Q alone). This is a STRUCTURAL CHARACTERIZATION that A3
does not perform on any Qwen3 model (A3 evaluates on LLaMA 3.1-70B primarily).

**A3 was NOT cited by Stage 3 or Stage 5.** Stage 3 cited QSVD (concatenated QKV stacking,
different primitive), CARE (separate W_K and W_V with activation weighting), and TransMLA.
Stage 5 re-verified novelty on 2026-05-09 and did not surface A3. This is a Stage 6 catch.

**A3 was available since May 2025 (10+ months before this red team). Stage 5's "re-verification
of novelty" on 2026-05-09 missed it. The mechanism is not engineering-integration of two known
ideas — A3 is a prior paper that explicitly targets the same product M = W_Q W_K^T.**

**Prior-art verdict: REJECT — A3 PRE-EMPTS the core primitive.**

---

### 5. Biased-Framing Audit

The GO threshold (var@350 ≥ 0.99 AND ΔNLL ≤ +1.0 nat) is reasonable and grounded in CF11.
The cross-term bound (M3) is mathematically correct. No gerrymandering detected.

The NO-GO class-level kill ("any future 'find a more compact W_Q W_K^T' proposal dies") is a
legitimate structural finding — the product-SVD class is now empirically characterized. However,
given the A3 pre-emption, the structural finding at NO-GO collapses to: "Qwen3-1.7B M's spectrum
does not concentrate faster than W_Q alone." This is a Qwen3-specific measurement, not a
class-level kill on the mechanism (A3 already shows the mechanism works on LLaMA 3.1-70B).

---

### 6. Runtime / Scope Sanity

The runtime estimate is consistent with the described work. Pilot (26 min with controls),
full run (~85 min): well within the 8-hour gate. The multi-seed adds +18 min as budgeted.
Cal/eval corpus: no calibration corpus; eval is WikiText-2 test, disjoint from training weights
by definition. **SCOPE CONSISTENT.** The scope issue is preemption, not over-scope.

---

### 7. CF13-CF15 Verification Check

No CF13/CF14/CF15 numbers cited in load-bearing positions. CF11 (W_Q r_99/d ≈ 0.63, W_K
r_99/d ≈ 0.79) is the only confirmed-findings dependency; CF11 is v1-confirmed. **SAFE.**

---

### 8. Verdict — Track B

**REJECT-PICK-RUNNER-UP**

A3 (arXiv:2505.12942, May 2025) explicitly computes SVD of the product W_Q W_K^T as the
compression primitive for the QK component of attention, with the same theoretical motivation
(functional attention-score error minimization > per-factor Frobenius error) as MQKP. The
mechanism — not just the adjacent neighborhood — is pre-empted.

MQKP's surviving contribution: the empirical measurement of M's r_99/d on Qwen3-1.7B-Base
(which A3 does not report for Qwen3 specifically), and the no-calibration-data variant (A3
requires input autocorrelations; MQKP proposes plain Eckart-Young). These are too thin to
sustain a 31.25-point pick at Stage 5. The no-calibration variant is a degenerate (R=I) special
case of A3 Theorem 2 — a simplification, not a novel primitive.

**Advance runner-up: R9-R1/ATQKV (score 25.3).**

ATQKV measures W_V and W_O spectra — weight matrices that A3 also addresses via its "OV
component," but ATQKV's focus on Qwen3-specific empirical measurements plus the four-matrix
joint ΔNLL accounting remain unstudied. Stage 6 recommends running ATQKV with A3 explicitly
added to the kill criteria (if W_V / W_O compression is also covered by A3's OV theorem,
ATQKV reduces to a pure Qwen3 characterization experiment — still scientifically valid but
with a narrowed novelty claim).

**KILL_LIST entry:**

`v2-S6-R009-001 / F1-MQKP — product-SVD of M = W_Q W_K^T as compression primitive` —
PRE-EMPTED by A3 (arXiv:2505.12942, May 2025). A3 Theorem 2 targets exactly W_Q W_K^T,
minimizes functional attention-score error via activation-weighted SVD, and has been evaluated
at 70B scale. MQKP's no-calibration variant is a degenerate special case. Surviving fragment:
empirical r_99/d measurement of M on Qwen3-1.7B-Base as a Qwen3-specific characterization
(scientifically useful, not a Stage 5 selection target).

---
---

## TRACK A — F3-L1GNF (Layer-1 Gate Null-Space Fold)

`elegance-class: null-space-alignment` | 25-min experiment (40 min with controls)

---

### 1. Frame-Mismatch Re-check (CF9 — second pair of eyes)

**Load-bearing imported machinery:**

(a) Eigendecomposition of C_x^0 = E[x^{L1} x^{L1,T}]: null space defined as eigenvectors with
eigenvalue < 1e-5.

(b) AERO fold identity: for a neuron whose gate output is near-constant c_i, replacing the gate
computation with c_i leaves the SwiGLU product approximately unchanged.

**Precondition audit (a) — null-space eigendecomposition:**

C_x^0 is an empirical covariance: X^T X / n where X ∈ ℝ^{500 × 2048}. By construction,
C_x^0 is positive semidefinite. Eigendecomposition exists and is exact. The "null space" (eigenvalue
< 1e-5) is the empirical null space — more precisely the near-null-space given finite samples.
With 500 diverse tokens, the empirical rank is likely close to 2048 (no exact zeros, many small
eigenvalues). This is not a precondition failure but a definitional clarification: the experiment
is testing WHETHER foldable neurons project onto the LOW-EIGENVALUE subspace of C_x^0, not a
geometrically exact null space.

**Precondition concern:** The GO threshold (null-space projection fraction ≥ 0.70 for foldable
neurons) depends on the choice of eigenvalue threshold for defining the "null space." Section 4
sweeps {1e-5, 1e-4, 1e-3}. If the projection fraction is sensitive to this choice — i.e., the
result changes from 0.70 at 1e-5 to 0.40 at 1e-4 — the claim becomes threshold-dependent rather
than structural. The plan correctly includes the sweep, but the GO threshold must specify which
eigenvalue threshold is primary.

Stage 3 verified the precondition correctly: C_x^0 is PSD by construction. Independent
re-verification: **PASS.**

**Precondition audit (b) — AERO fold identity:**

The fold claim: for neuron i, if std(silu(W_gate^{L1}·x)_i) < 0.05 across calibration tokens,
then replacing the gate output with c_i = mean(...) is quality-neutral. The AERO identity holds
EXACTLY when the gate output is literally constant. For neurons with std < 0.05 but not zero,
it is approximate. The quality bound depends on how close std < 0.05 is to truly zero. The CF6
criterion (std < 0.05) was defined empirically to yield ΔNLL ≤ negligible; H2 (ΔNLL ≤ +0.05
nats) is the operative check.

There is a subtle precondition the plan does not explicitly state: the mean-field constant c_i
must be computed on the SAME token distribution as the eval tokens, or the fold introduces a
distribution-shift error. The plan uses WikiText-2 train for calibration (c_i estimation) and
WikiText-2 test for eval. These are from the same domain but different tokens — acceptable.
However, the multi-seed ΔNLL control (Section 9) handles this by testing three non-overlapping
windows, which adequately verifies robustness. **PASS.**

**CF9 verdict: PASS.** No frame-mismatch detected. The eigendecomposition precondition holds
trivially; the AERO fold identity is approximate and the H2 ΔNLL check is the operative test.

---

### 2. Calibration Ill-conditioning Re-check (CF10)

**Calibration fit: c_i for foldable neurons.**

n_params_to_fit = ~2218 scalars (one mean per foldable neuron).
n_independent_samples = 500 tokens × 1 output dim per neuron = 500 per neuron.
Ratio: 500 >> 1 (vastly overconditioned — each c_i is estimated from 500 independent
observations of a scalar). There is no matrix regression here; this is straightforward
mean estimation. **CF10 SAFE.**

**Covariance eigendecomposition calibration:**

C_x^0 has 2048 × 2048 / 2 ≈ 2.1M unique entries, estimated from 500 tokens × 2048 dims.
n_samples × d = 500 × 2048 = 1.024M. With 2.1M unique covariance entries and only 1.024M
activation values, the covariance estimate is underdetermined in the matrix sense (rank ≤ 500,
many eigenvalues exactly zero by construction).

This is not a CF10 calibration-fitting failure — we are not FITTING the covariance (no
optimization, no parameter update); we are ESTIMATING it. The key question is whether the
top-500 eigenvectors (the non-null space) and the bottom-1548 eigenvectors (the null space)
are reliably identified from 500 tokens. With 500 tokens, the 500 non-null eigenvectors will
be well-estimated, but the relative ordering among the near-null eigenvectors (all eigenvalues
near zero) may be noisy. This affects which eigenvectors are labeled "null" at the 1e-5 threshold.

**Stage 6 amendment required (Amendment A1 — covariance stability check):** Run the covariance
estimation on two disjoint 250-token halves of the calibration corpus and report the Frobenius
distance between the two resulting null-space projection matrices (P_null estimated on half 1 vs
half 2). If ||P_null^{(1)} − P_null^{(2)}||_F > 0.10 × ||P_null||_F, the null-space estimate is
unstable and the eigenvalue threshold should be increased (fewer eigenvectors in the null space
definition). This check costs < 1 min. It catches the scenario where the GO threshold (≥ 0.70
projection fraction) is met only because the null space includes near-noise eigenvectors that
randomly project onto many neurons. **Add to Section 4 sweep: report null-space stability at
each eigenvalue threshold.**

---

### 3. Skeptic-Controls Check

**Hypothesis shape:** H1 ("null-space alignment explains foldability") is a STRUCTURAL PROPERTY
claim. H2 ("fold is quality-neutral, ΔNLL ≤ +0.05 nats") is a GENERALIZATION claim.
Skeptic-controls gate applies to both.

**Permutation control (H1):** Present and well-specified (Section 9). 100 random subsets of
2218 neurons (same size as CF6-foldable); foldable group's mean projection fraction must exceed
the 95th percentile of random subsets. This correctly controls for "any 36% subset trivially
shows high null-space projection due to general W_gate structure." **ADEQUATE.**

**Random-init control (H1):** Present and well-specified (Section 9). W_gate rows replaced with
Kaiming uniform; same C_x^0 (trained model); measure projection fractions. Expected random-init
projection ≈ null_dim / 2048 (uniform distribution over directions). **ADEQUATE.**

**Multi-seed (H2):** Three non-overlapping WikiText-2 test windows. GO threshold met by mean
ΔNLL; worst-of-3 must not exceed +0.30 nats. **ADEQUATE.**

**One concern not addressed:** The permutation control (100 random subsets) runs on the full set
of 6144 W_gate rows. However, the GO threshold for H1 says the foldable group must exceed the
95th percentile of the RANDOM-SUBSET distribution. If the W_gate rows have heterogeneous
null-space projections (some rows have high projection, some low), then a random 36% sample will
sometimes draw mostly high-projection rows by chance. The 95th percentile threshold correctly
handles this if 100 samples is enough to estimate the 95th percentile stably.

95th percentile of 100 samples has ±1 rank uncertainty at the extremes. This is adequate —
the 5th-highest of 100 samples is a stable estimate of the 95th percentile. **ADEQUATE.**

**Skeptic-controls verdict:** ADEQUATE with Amendment A1 (covariance stability check from
Section 2 above).

---

### 4. Hidden Prior-Art Search — Third Pass (2026-05-09)

**Search 1:** "null space input covariance gate neuron fold SwiGLU transformer post-training
2025 2026"

**Search 2:** "gated neurons transformer input-output functionality fold covariance null space
arxiv 2505.17936"

**Findings:**

**arXiv:2505.17936 (May 23, 2025) — "Understanding Gated Neurons in Transformers from Their
Input-Output Functionality"** by Gerstner & Schütze. This paper examines the cosine similarity
between input and output weights of gated neurons (enrichment vs depletion neurons). It does NOT
use null-space alignment to predict fold eligibility, does not use C_x^0 eigendecomposition,
and does not propose a fold operation. **Not a pre-emption.**

**AERO (arXiv:2410.13060):** Known adjacent; Stage 3 correctly distinguished L1GNF (post-training,
per-neuron, Layer-1-specific null-space criterion) from AERO (global, requires retraining).
Re-verified. No AERO variant uses null(C_x^0) as the fold predictor. **Not a pre-emption.**

**No paper found using null(C_x^0) alignment as a fold-eligibility predictor.** The null-space
criterion for selecting which neurons to fold, specifically derived from the Layer-1 input
covariance eigenbasis, is not in the published literature as of 2026-05-09.

**Prior-art verdict: NOVEL.** F3-L1GNF's core mechanism survives the third pass.

---

### 5. Biased-Framing Audit

**GO threshold (projection fraction ≥ 0.70 for foldable, ≤ 0.30 for non-foldable, effect size
≥ 0.30):** The thresholds are derived from what would constitute a meaningful structural
distinction. The 0.70 threshold is not trivially achievable — if the null space has 512 of 2048
dimensions, a uniform random row would project ~0.25, making 0.70 well above chance. **Not
gerrymandered.**

**NO-GO class-level kill:** Well-specified — if null-space fraction for foldable neurons ≤ 0.40,
the Layer-1 anomaly is a calibration-distribution artifact, not a geometric property. This is a
specific and useful structural finding. **Load-bearing, not rhetorical.**

**GRAY zone (0.40–0.70):** The follow-up (test at K_null = 512 instead of 256) resolves whether
the effect is present at a broader null-space definition. Pre-specified and specific. **Adequate.**

**Potential framing concern:** The H1 GO threshold (≥ 0.70 projection fraction) is ambiguous
about which eigenvalue threshold defines the "null space." If at threshold 1e-5 the projection
fraction is 0.67 (just below GO) but at 1e-4 it is 0.72 (just above GO), the GO/NO-GO call
becomes threshold-dependent. The plan sweeps three thresholds but does not specify which is
primary for the GO decision.

**Stage 6 amendment (Amendment A2 — primary eigenvalue threshold declaration):** Section 5
must declare one eigenvalue threshold as PRIMARY for the GO/NO-GO decision (recommend 1e-5 as
strictest, closest to true null). The alternative thresholds (1e-4, 1e-3) are exploratory and
reported but not used for the GO call. Add to Section 5: "Primary GO uses eigenvalue threshold
1e-5. Results at 1e-4 and 1e-3 are reported in the output JSON but do not change the GO call."
This prevents post-hoc threshold selection.

**Overall framing bias:** Minimal. The plan's elegance claim (constructive null-space criterion)
is correct — the criterion is directly computable and the experiment is designed to falsify it
cleanly. No defensive over-framing detected.

---

### 6. Runtime / Scope Sanity

**Runtime estimate (Section 10):** ~40 min total with controls.

Breakdown: 3 min setup + 5 min calibration pass + 5 min eigendecompose + 2 min projection
analysis + 3 min permutation control + 5 min random-init control + 15 min ΔNLL eval (3 windows)
+ 2 min output = 40 min. Adding Amendment A1 (covariance stability check) = ~41 min. Well within
8-hour gate. **CONSISTENT.**

**Smallest test decides GO:** The null-space projection check (Section 4, Steps 4-5) produces
the primary GO/NO-GO in ~15 min. The ΔNLL eval (H2) is a downstream quality check. The plan
correctly sequences these: H1 is the structural gate; H2 is the quality gate; both must pass for
full GO. **Well-scoped.**

**Cal/eval corpus disjoint:** Calibration (c_i estimation): WikiText-2 train split, 500 tokens.
Covariance calibration: same 500 tokens. Eval: WikiText-2 test split, three non-overlapping
windows. Train split and test split are disjoint by dataset definition. **SAFE.**

**Over-scope check:** The covariance eigendecomposition of a 2048×2048 matrix from 500 tokens
(~5 min) plus the 100-subset permutation control (vectorizable, ~3 min) and 15 min for 3 eval
windows total 41 min. The experiment is correctly described as ~40 min, and all skeptic controls
fit within this budget. **Not over-scoped.**

---

### 7. CF13-CF15 Verification Check

CF6 (Layer-1 anomaly, 36.1% foldable neurons) is the primary grounding. CF6 is v1-confirmed.
No CF13/CF14/CF15 numbers cited in load-bearing positions. **SAFE.**

---

### 8. Verdict — Track A

**ACCEPT-WITH-AMENDMENTS**

Two amendments required; neither is fatal. No prior art found; no CF9 or CF10 failure.

**Amendment A1 (covariance stability, structural — CF10 boundary):** In Section 4, add a
covariance stability check: compute C_x^0 on two disjoint 250-token halves of the calibration
corpus; report Frobenius distance between the two resulting null-space projection matrices
P_null at each eigenvalue threshold. If the distance exceeds 10% of ||P_null||_F, increase the
eigenvalue threshold until stable. Report the stable threshold used. Wall-clock addition: <1 min.

**Amendment A2 (GO threshold sharpening, anti-bias):** Declare eigenvalue threshold 1e-5 as
PRIMARY for the GO/NO-GO decision in Section 5. Alternative thresholds (1e-4, 1e-3) are
exploratory and reported but do not override the primary GO call. Add to Section 5: "Primary GO
uses eigenvalue threshold 1e-5. Alternative threshold results are exploratory." This closes the
post-hoc threshold selection risk identified in the biased-framing audit.

---

## Summary Table

| Track | Pick | CF9 | CF10 | Skeptic | Prior Art | Biased Frame | Runtime | Verdict |
|---|---|---|---|---|---|---|---|---|
| A | F3-L1GNF | PASS | SAFE | PASS (A1) | NOVEL | CLEAN (A2) | 41 min | **ACCEPT-WITH-AMENDMENTS (A1, A2)** |
| B | F1-MQKP | PASS | N/A | PASS (B1) | **PRE-EMPTED (A3)** | CLEAN | 26 min pilot | **REJECT-PICK-RUNNER-UP** |

---

## Prior-Art Notes for Run 010

**A3 (arXiv:2505.12942, May 2025 v1, June 2025 v3)** — Explicitly computes SVD of W_Q W_K^T as
the QK component compression primitive. Evaluates on LLaMA 3.1-70B. Uses activation-weighted
SVD (R^{1/2} W_qk R^{1/2}); RoPE is handled via CUR approximation on the same product. Any
run 010+ proposal targeting the product M = W_Q W_K^T compression must either (a) differentiate
by using no-calibration plain Eckart-Young with Qwen3-specific r_99/d measurements AND explicitly
compare against A3's calibration-weighted baseline, or (b) target a different joint product (e.g.,
M_VO = W_V W_O^T, which A3's OV theorem may also cover — needs verification).

**arXiv:2505.17936 (May 2025)** — Gated neuron input-output functionality enrichment/depletion
taxonomy. Relevant background for L1GNF IF GO: the "enrichment" neuron classification (high
input-output cosine similarity) may correlate with null-space projection fraction. If foldable
neurons are enrichment type (output in the same direction as input), null-space alignment would
suppress both input AND output of those neurons. This is a follow-up measurement worth adding if
L1GNF GO, but not a blocker.

---

## Runner-Up Status

**Track A runner-up: C14-CFSHIFT (score 17.3).** Stage 6 does not kill F3-L1GNF (ACCEPT-WITH-
AMENDMENTS); CFSHIFT remains on deck for Run 010 Track A if L1GNF NO-GO produces a structural
finding that CFSHIFT's premise depends on. CFSHIFT's Stage 5 plan is independent of L1GNF GO
(CFSHIFT depends only on CF6 + W_Q linearity).

**Track B runner-up: R9-R1/ATQKV (score 25.3).** Advance without re-running Stage 5. ATQKV
measures W_V and W_O spectra — A3 has an "OV component" (Theorem 3) that may pre-empt the
compression application, but the Qwen3-specific r_99/d characterization remains unstudied. Stage
5 for ATQKV should explicitly check whether A3 Theorem 3 (OV component) pre-empts the W_V/W_O
compression claim before scoring. If A3 Theorem 3 pre-empts, ATQKV reduces to a Qwen3
characterization experiment (valid science, narrowed novelty).
