# Stage 6 — Red Team — Run 005 (v2 Sonnet Ladder)

Completed: 2026-05-09. Adversarial review of Stage 5 Track A pick (A1-GFRS-2) and Track B pick
(S6-SEQFILT). Fresh prior-art WebSearch run at red-team time. Runner-up identified per track:
Track A → F2-HPGF; Track B → R5-RAOK-FULL.

---

## TRACK A — A1-GFRS-2 (Z_2^d Residual-Stream Sign-Entropy)

`elegance-class: gauge-exploitation` | 10-min experiment | frame-novelty advancer

---

### 1. Frame-Mismatch Re-check (CF9 — second pair of eyes)

**Load-bearing claim:** The transformer residual stream admits an exact Z_2^d sign-flip gauge
symmetry. Flipping the sign of every weight that reads or writes channel c leaves the network
function unchanged.

**Precondition audit (four sub-conditions independently verified):**

**(a) RMSNorm sign-agnosticism.** RMSNorm computes x / RMS(x). This is scale-invariant but NOT
sign-flip-invariant in isolation — however, the channel participates in RMSNorm as one element of
the d-dim vector. Per-channel sign flip of x[c] flips the numerator element x[c] AND leaves
RMS(x) unchanged (RMS depends on squares). Net effect: RMSNorm(x)[c] → −RMSNorm(x)[c] if x[c]
→ −x[c] JOINTLY with all weights reading channel c. **The gauge argument requires co-flipping all
weights simultaneously, not flipping the activation alone.** The plan states this correctly —
independent verification confirms the precondition holds for RMSNorm without learned bias.

**(b) Bias-free RMSNorm required.** Stage 5 correctly identifies: if RMSNorm has additive bias
b, then h → −h + bias ≠ −(h + bias). Qwen3-1.7B uses bias-free RMSNorm (confirmed by
architecture specification). Section 8 of the plan flags this as Kill Criterion 1. Independent
re-verification: this is the correct precondition and it is met for Qwen3. **PASS.**

**(c) Rotary embeddings (RoPE) on the residual stream.** RoPE is applied inside the attention
projection (to q and k after W_Q and W_K GEMVs), not to the residual stream h directly. The
residual stream carries h_L without rotation; rotations happen inside attention heads on
head-level projections. Therefore, the Z_2^d sign flip of channel c in h_L does NOT interact
with RoPE. **PASS.**

**(d) SwiGLU sign threading.** The plan claims: for W_gate and W_up both seeing the same
sign-flipped input, the element-wise product W_gate·(−x) ⊙ W_up·(−x) = (+W_gate·x) ⊙ (+W_up·x)
if BOTH W_gate and W_up rows for that channel are also sign-flipped. This requires the gauge
transformation to co-flip the corresponding rows of BOTH W_gate AND W_up simultaneously.

**This is a precondition the plan does not state explicitly as a measurement requirement.** The
experiment measures only W_up sign entropy AFTER gauge-fixing. The gauge-fixing procedure (flip
each W_up row if its max-abs element is negative) is applied to W_up in isolation. But the gauge
symmetry argument requires that when you flip W_up row i, you ALSO flip the corresponding row of
W_gate (because W_gate produces the gate activation silu(W_gate·x), and the sign flip of x[c]
must thread through BOTH projections consistently). If the gauge-fix is applied only to W_up
independently of W_gate, the canonical form is NOT the gauge-fixed form — the W_up rows are
normalized, but W_gate rows may point in the opposite direction, breaking the gauge equivalence.

**Specific risk:** The gauge-fixing procedure described in Section 4 normalizes W_up rows
independently (flip each row to make max-abs positive). This is NOT equivalent to the Z_2^d
gauge-fixing of the residual stream, which requires co-flipping all weight matrices that read
channel c (including W_gate, W_Q, W_K, W_V, and W_up simultaneously). Applying the gauge fix
only to W_up produces a DIFFERENT object — W_up in an arbitrary (non-canonical) coordinate frame
relative to W_gate.

**Implication:** The sign entropy measurement on the independently normalized W_up rows measures
the intrinsic sign distribution of W_up after per-row normalization, NOT the sign distribution in
the jointly gauge-fixed representation. The two measurements differ if (and only if) the per-row
sign of W_up's max-abs element is correlated with the per-row sign of W_gate's corresponding
row. If training jointly aligns W_gate and W_up per-neuron signs (plausible: the product
silu(W_gate·x) · (W_up·x) rewards alignment of sign for positive-firing neurons), then the
per-row normalization of W_up alone produces similar entropy to the joint gauge-fix. But this
alignment is an empirical question, not guaranteed.

**Stage 6 amendment required (Amendment A1):** Add a measurement of W_gate row signs
alongside W_up. Specifically: for each neuron i, record sign(argmax|W_up[i,:]|) and
sign(argmax|W_gate[i,:]|). Report the fraction of neurons where these two signs agree. If
agreement > 90%, the per-row gauge-fix of W_up is approximately equivalent to the joint gauge-fix
(W_gate rows already point consistently). If agreement < 70%, the per-row W_up normalization is
not the joint canonical gauge. This measurement costs < 1 minute and is structurally important
for the elegance claim.

**Overall CF9 verdict on A1-GFRS-2:** The core Z_2^d argument is mathematically correct for the
full joint gauge-fix. The experimental procedure (normalize W_up rows independently) may diverge
from the theoretical claim if W_gate / W_up sign alignment is weak. Amendment A1 closes this gap.
This is NOT a fatal flaw — the sign-entropy measurement is still informative regardless — but the
interpretation ("this is the gauge-fixed sign distribution") requires the alignment check.

### 2. Calibration Ill-conditioning Re-check (CF10)

**Measurement experiment (not a calibration fit).** The sign entropy H_L is a deterministic
function of the loaded weight matrix W_up. No calibration fitting is involved. n_params_to_fit = 0.

**Random-init control** involves measuring H_L on a randomly initialized model — still a pure
measurement. CF10 does not apply. **SAFE.**

### 3. Skeptic-Controls Check

**Hypothesis shape:** "After Z_2^d gauge-fixing, the sign plane of W_up in trained Qwen3-1.7B
has structured (low-entropy) patterns not present in random-initialization."

This is a STRUCTURAL PROPERTY claim, not a transfer/generalization claim. The claim is:
H_global(trained, gauge-fixed) < 0.8 bits/weight.

**Permutation control:** Section 9 correctly identifies that column permutation within a row
does not change per-row entropy H_r (since H_r = h(p_r) where p_r = fraction positive, and
p_r is invariant to column permutation). The plan correctly proposes std(p_r) as the
informative statistic instead of a permutation control on the SIGN PATTERN. **The substitution
of std(p_r) for a traditional permutation control is valid and documented.** The key question
is: does p_r vary meaningfully across rows (std > 0.15 → structured) or is it uniform
(std < 0.05 → unstructured)?

However, the permutation control AS STATED in the GO threshold (Section 5) does not explicitly
require that std(p_r) exceed a threshold. The GO threshold says: H_global < 0.8. It does not
say: "AND std(p_r) > 0.10" or "AND the random-sign-draw control shows H_L(real) < H_L(random
draw with same p_r)." If H_global < 0.8 is achieved because all rows have the same p_r ≈ 0.72
(globally skewed but structurally uniform), the sign plane is compressible but NOT by a
row-specific codebook — only by a global bias. The compression implication differs.

**Stage 6 amendment required (Amendment A2):** Add explicit GO condition: std(p_r) > 0.10 to
confirm row-heterogeneous sign structure (justifying a codebook) versus std(p_r) ≤ 0.05 which
motivates a global sign-bias correction but not a per-row codebook. This sharpens Section 7's
bimodal GRAY zone analysis.

**Random-init control:** Present and well-specified (Section 9). Trained model expected H ≈ 0.7–0.8
(if hypothesis holds); random-init expected H ≈ 1.0. Gap ≥ 0.10 required. **ADEQUATE.**

**Multi-seed:** Correctly documented as not applicable (deterministic weight-property computation).
Section 9 states this explicitly. **ACCEPTED.**

**Skeptic-controls verdict:** ACCEPT WITH AMENDMENTS (A1, A2). No structural deficiency that
would produce a false positive — the controls are adequate with the additions.

### 4. Hidden Prior-Art Search (Third Pass, 2026-05-09)

**Search 1:** "Z2 sign flip gauge symmetry transformer residual stream quantization 2025 2026"
→ No paper applying Z_2^d discrete sign-flip symmetry to LLM quantization codebook design found.

**Search 2:** "sign entropy weight quantization codebook LLM 'sign bit' structured 2026"
→ BTC-LLM (arXiv:2506.12040) uses binary codebooks; centroid updates use sign functions. This is
about LEARNED codebooks for sub-1-bit quantization, not the residual-stream gauge-fixing approach.
**Not a pre-emption.** BTC-LLM learns sign patterns through training; GFRS-2 exploits a symmetry
that is independent of training.

**Search 3:** "RMSNorm sign invariance transformer weight symmetry coordinate fixing 2025 2026"
→ **arXiv:2412.14543 (Dec 2024, van Nierop) — "Transformer models are gauge invariant: A
mathematical connection between AI and particle physics."** This paper establishes that
transformer architectures exhibit gauge invariance, including that the default representation
has partially but not fully removed gauge invariance, and outlines practical considerations for
reducing parameters without loss of representational power.

This is the most significant prior-art finding at Stage 6. The paper explicitly covers
transformer gauge invariance as a mathematical structure. However:

1. The paper's focus is the MATHEMATICAL STRUCTURE (the gauge group, its continuous components,
   implications for parameterization), NOT the application to quantization codebook design.
2. The paper does not mention sign-entropy measurement of the gauge-fixed weight matrix.
3. The paper does not propose exploiting the Z_2^d discrete subgroup specifically for sign-plane
   compression at quantization scale.
4. The compression application (sign entropy < 0.8 bits/weight → compressible sign plane) is
   not in the paper.

**→ arXiv:2412.14543 is ADJACENT, not pre-emptive.** It establishes the mathematical framework;
A1-GFRS-2 applies one specific discrete subgroup to a compression measurement. Stage 5 did not
cite this paper. Stage 6 requires it be CITED in the experiment plan as the prior-art reference
for the gauge invariance framework (Amendment A3). The novelty moat remains: the quantization
application of the Z_2^d subgroup is not in 2412.14543.

**→ arXiv:2602.18948 (Feb 2026, François & Ravera) — "Toward Manifest Relationality in
Transformers via Symmetry Reduction."** Covers reparameterization symmetries and symmetry
reduction for transformer blocks, focused on Q-K bilinear forms and continuous symmetries. Does
NOT address the Z_2^d discrete subgroup or quantization. **Not a pre-emption.**

**Prior-art verdict:** NOVEL (with citation of 2412.14543 required). No pre-emption found.

### 5. Biased-Framing Audit

**GO threshold (H_global < 0.8):** The 0.8 threshold is motivated by the Huffman break-even at
p = 0.8 positive (0.72 bits/symbol). This is a principled derivation, not gerrymandering. However,
the break-even applies to a GLOBAL codebook over all rows. For a per-row codebook, the relevant
statistic is p_r per row; the compression gain is ∫ h(p_r) dp_r vs the mean h(E[p_r]). If all
rows have p_r ≈ 0.8, the per-row codebook offers no gain over a single global code. The GO
threshold implicitly conflates global entropy with per-row structure. Amendment A2 above
addresses this.

**NO-GO bonus claim ("closes S2-BITSER and the sign-plane-as-separate-object family"):** This
claim is LOAD-BEARING and SPECIFIC. A NO-GO correctly kills the sign-plane codebook path.
However, the claim that NO-GO also closes "sign-coordinate choice improves quantization" is too
broad — sign-fixing as a pre-processing step for SVD or magnitude quantization survives even if
H(sign) ≈ 1.0. The plan acknowledges this in Section 6 ("does NOT kill gauge-fixing as a
pre-processing step for SVD"). **The scope of the NO-GO kill is correctly bounded.** No framing
bias detected here.

**GRAY zone (H_global ∈ [0.8, 0.95)):** The PARTIAL-GO resolution (layer-selective sign
compression if low-entropy cluster ≥ 8 contiguous early layers) is a GRAY zone with a specific
resolution criterion. This is not post-hoc spin — the criterion is pre-specified. **Adequate.**

**Overall framing bias:** None detected. The elegance multiplier applied at Stage 5 is grounded
in an algebraic identity that this Stage 6 review confirms as correct (with the Amendment A1
caveat about W_gate co-alignment).

### 6. Runtime / Scope Sanity

**Runtime table (Section 10):** 5 min model load + <1 min gauge-fix + <2 min entropy + 3 min
random-init + <1 min output = ~11 min. Reasonable for a weight-property computation on
Qwen3-1.7B-Base bf16. No concerns.

**Amendment A1 overhead:** W_gate row sign extraction adds < 1 min. Total revised estimate: ~12 min.

**Smallest test decides GO threshold:** Yes — H_global is computed directly from the 28 W_up
matrices. No follow-up needed for the GO/NO-GO determination. GRAY zone follow-up (layer
stratification) uses data already computed from the base run. **Well-scoped.**

**Cal/eval corpus:** Not applicable — this is a weight-property measurement. **SAFE.**

### 6b. Wildcard Non-Penalization

A1-GFRS-2 is a frame-novelty advancer (Path 3), not a tagged wildcard. Standard gates apply.
All gates met with amendments.

### 6c. Elegant-Equivalence Rejection Ground (Section 6c)

`elegance-class: gauge-exploitation`

**Check:** Does the gauge-exploitation claim depend on a symmetry the loss is invariant to?

The Z_2^d sign-flip gauge symmetry is an EXACT symmetry of the transformer computation for any
choice of signs — the function is invariant. This is stronger than RMSNorm γ (per-channel, not
a scalar — the LFAO failure mode). The sign-flip symmetry holds exactly at inference time because:
- All computations with channel c are linear in h[c] (the RMSNorm denominator uses squares, which
  are sign-invariant; all weight products thread the sign through linearly).
- No nonlinearity acts on the RAW residual-stream channel — nonlinearities (SwiGLU, softmax) act
  on PROJECTED values (after W_gate, W_Q, etc.), where the co-flipped weight rows absorb the sign.

**No section 6c failure detected.** The gauge-exploitation precondition (loss invariance to the
gauge transformation) is met exactly. The Amendment A1 issue (W_gate co-flipping) is not a 6c
failure — it is a measurement protocol issue, not a math error in the gauge argument itself.

### 7. CF13-CF15 Verification Check

No CF13/CF14/CF15 numbers cited in load-bearing positions. **SAFE.**

### 8. Verdict — Track A

**ACCEPT-WITH-AMENDMENTS**

Three amendments required:

**Amendment A1 (CF9 frame-completion, load-bearing):** In the measurement script, add extraction
of W_gate row signs alongside W_up. For each neuron i, compute and record
sign_agreement_fraction = fraction of neurons where sign(argmax|W_up[i,:]|) == sign(argmax|W_gate[i,:]|).
Report this in the JSON output. If sign_agreement_fraction < 0.70, note in results that the
per-row W_up gauge-fix is not equivalent to the joint residual-stream gauge-fix, and the sign
entropy measurement should be interpreted as W_up intrinsic sign structure, not gauge-fixed
structure. Wall-clock addition: < 1 min.

**Amendment A2 (skeptic-controls sharpening, structural):** Add explicit GO sub-condition:
mean std(p_r) across all 28 layers > 0.10 (row-heterogeneous sign structure confirmed, justifying
per-row codebook) versus std(p_r) ≤ 0.05 (global bias only). Add this to Section 5 GO threshold
alongside the H_global < 0.8 condition. Add "std(p_r) < 0.05 despite H_global < 0.8" as a
SOFT-GO-DOWNGRADE path: compression is possible but via global sign bias, not codebook.

**Amendment A3 (prior-art citation, non-blocking):** Add citation of arXiv:2412.14543 (van Nierop,
Dec 2024, "Transformer models are gauge invariant") as the prior mathematical framework reference.
The quantization application is not in that paper; the citation contextualizes the gauge argument.

---

## TRACK B — S6-SEQFILT (Sequential-Scan Inline RAOK Filter)

`elegance-class: algebraic-identity` | phase-gated 50min + 85min controls | depends on A6-CNTR Phase 0 GO

---

### 1. Frame-Mismatch Re-check (CF9 — second pair of eyes)

**Load-bearing claim:** The scatter-gather T2 row selection (RAOK's precomputed 18-row index)
is EQUIVALENT to an inline 20-element dot-product threshold test applied during sequential W_up
scan, provided threshold τ_L is calibrated so that the two methods agree on ≥ 90% of T2/T3
assignments.

**Imported algebraic identity audit:**

The claim is NOT that the two methods are algebraically identical — it is that they agree at
≥ 90% Jaccard on held-out. This is NOT an algebraic identity; it is a CALIBRATED APPROXIMATION.
The plan's elegance tag "algebraic-identity" is misleading. The actual claim is empirical: the
dot-product threshold approximates the scatter-gather index with ≥ 90% agreement.

**Precondition check for the approximation claim:**

The dot-product ⟨w_i[outlier_indices], x_outlier⟩ uses a FIXED set of "outlier_indices" for
the classification. But the T2 row identity in RAOK is defined by which rows w_i have the
HIGHEST absolute inner product with x_outlier (the top-18 by dot-product ranking). The
sequential SEQFILT classification uses a THRESHOLD τ_L (the 18th-largest dot-product value from
calibration tokens) to decide T2 vs T3.

**Key precondition that must hold for ≥ 90% agreement:** The calibration-derived threshold τ_L
must generalize to held-out tokens — i.e., the 18th-largest dot-product value must be stable
across tokens. This is what Section 8, Kill Criterion 2 flags (std of τ_L over calibration
tokens). Stage 6 re-verifies: if the dot-product distribution shifts significantly between
calibration and eval, the threshold τ_L will classify more or fewer rows as T2 on eval tokens,
and the 90% Jaccard agreement will not hold.

**Stage 3's precondition check on S6-SEQFILT:** Stage 3 was not called to check S6-SEQFILT
(it is a Stage 4 idea). The Stage 4 description in Section 5 of stage4_skeptic.md states the
precondition correctly: "A6-CNTR's precision@20% > 0.5 (outlier channels predict W_up rows)."
This is the RAOK routing-key precondition.

**Specific structural risk NOT flagged in the plan:** The SEQFILT agreement rate depends on
the threshold τ_L capturing exactly 18 T2 rows per token on average. But the number of rows
exceeding τ_L is a random variable per token — on some tokens it may be 12, on others 24.
If the threshold is designed to yield ~18 rows on average (from calibration), the actual T2
set size fluctuates, and the Jaccard agreement with RAOK's exactly-18-rows-per-token set will
be penalized by the size mismatch. The Jaccard formula: |A ∩ B| / |A ∪ B|. If SEQFILT flags 24
rows and RAOK flags 18 rows with 17 in common: Jaccard = 17/(24+18-17) = 17/25 = 0.68, below
the 0.80 NO-GO threshold.

**Stage 6 amendment required (Amendment B1):** Section 4 should report not only mean Jaccard
agreement but also (a) mean T2-set size produced by SEQFILT per token (should be ~18) and
(b) std of T2-set size per token. If T2-set size std is > 5 (i.e., SEQFILT produces 13–23 rows
routinely), the threshold-based approach is unstable and the inline classifier cannot guarantee
consistent T2 set cardinality. A fix: use "top-K threshold" instead of "fixed threshold" (always
take the top 18 rows by dot-product during the scan) — but this requires ranking during the scan,
which is O(n log n) instead of O(1) per row. The plan's elegance ("no additional memory access")
depends on O(1) per-row classification; ranking breaks this. Clarify in Section 4 whether SEQFILT
uses a FIXED threshold or an ADAPTIVE top-K approach, and if fixed threshold, report the
cardinality instability.

### 2. Calibration Ill-conditioning Re-check (CF10)

**Calibration fits in this experiment:**
- τ_L per layer (1 scalar per layer = 28 scalars total) derived as the 18th-largest dot-product
  value over calibration tokens.
- This is a single-scalar calibration per layer, not a parameter fit in the CF10 sense.
- n_params = 28; n_independent_samples = 200 tokens × 28 layers = 5600. Vastly overconditioned.

**CF10 SAFE.** No calibration ill-conditioning issue.

### 3. Skeptic-Controls Check

**Hypothesis shape:** "The inline 20-dot-product sequential T2 classifier produces the same
T2/T3 row assignments as RAOK's precomputed scatter-gather, at ≥ 90% mean Jaccard agreement on
held-out tokens — driven by trained activation-outlier structure (CF3), not architectural priors."

This IS a transfer/generalization claim: the classifier generalizes from calibration (threshold
calibration) to held-out tokens. The skeptic-controls gate applies.

**Permutation control:** Present and well-specified (Section 9). Permuted x_outlier should give
Jaccard ≈ 0.0088 (random 18-of-2048 vs 18-of-2048). GO requires real ≥ 0.90, permuted ≤ 0.10,
gap ≥ 0.80. **ADEQUATE.**

**Random-init control:** Present and well-specified (Section 9). Random-init activations should
produce SEQFILT T2 sets uncorrelated with trained-RAOK T2 sets. Expected agreement ≈ 0.0088.
GO requires random-init agreement ≤ 0.10. **ADEQUATE.**

**Multi-seed:** Present and well-specified (Section 9). 3 calibration splits; worst-of-3 ≥ 0.80.
Runtime addition: 60 min. Total estimated ~2.3 hours. **ADEQUATE.**

**One additional control not specified:** The permutation control permutes x_outlier's channel
indices. This destroys the CF3 identity of which channels are outliers. But there is a second
potential artifact: the threshold τ_L is calibrated on a specific set of calibration tokens.
If calibration tokens are chosen from the same WikiText-2 distribution as eval tokens (same
split, sequential), there may be autocorrelation between calibration and eval token activation
distributions, inflating the agreement rate. The multi-seed control partially addresses this by
using different calibration windows. **The multi-seed control as specified IS the right control
for threshold generalization.** No additional control needed.

**Skeptic-controls verdict:** ADEQUATE with Amendment B1 (cardinality stability check). No
structural deficiency.

### 4. Hidden Prior-Art Search (Third Pass, 2026-05-09)

**Search 1:** "sequential scan inline activation filter MLP row classification dynamic
quantization LLM 2025 2026" → No paper proposing inline sequential row classification during
GEMV as scatter-gather replacement found. Results cover mixed-precision quantization and
activation outlier handling but none with the sequential-scan-filter framing.

**Search 2:** "activation outlier routing MLP firing row prediction inline scatter-gather
replacement quantization 2026" → Results cover outlier suppression and scaling methods
(SmoothQuant family), ParoQuant (pairwise rotation, ICLR 2026), and activation regularization.
None propose the scatter-gather-to-sequential-filter equivalence.

**No pre-emption found.** The "inline 20-dot-product row classification during sequential scan
as scatter-gather equivalent" framing has no published analog as of 2026-05-09. The novelty
claim holds.

**Note on A6-CNTR dependency:** S6-SEQFILT depends on A6-CNTR Phase 0 (precision@20% > 0.50).
A6-CNTR itself cites CF1's finding that gate-side predictors fail in deep layers. Stage 6
confirms: CF3's outlier-channel identity is W_up-side (residual-stream input channels), not
gate-side. The CF1 risk is for gate-side predictors; SEQFILT uses the input activation
channel identity. The distinction is maintained correctly in the plan. If Phase 0 fails, the
Track B escalation to R5-RAOK-FULL is correct.

### 5. Biased-Framing Audit

**GO threshold (mean agreement ≥ 0.90, 10th-percentile ≥ 0.80):** The 0.90 threshold is
derived from "within 10% of scatter-gather precision" — a principled engineering tolerance.
The 10th-percentile ≥ 0.80 requirement ensures worst-case token performance is acceptable.
No gerrymandering detected.

**SOFT-GO path (mean ∈ [0.80, 0.90)):** "T2 misclassification rate ≤ 20%; acceptable with
24 channels instead of 18." This is a reasonable SOFT-GO path that avoids false-negative on
a marginally-below-threshold result. **Not post-hoc spin — the 24-channel follow-up is
well-specified.**

**NO-GO structural findings:** Two distinct NO-GO cases with different class-level kills are
well-articulated:
- Phase 0 NO-GO kills both SEQFILT and A6-CNTR.
- Phase 1 NO-GO kills SEQFILT specifically (20-dim dot product too coarse), not RAOK.

Both structural findings are specific and actionable. **No rhetorical "we'll learn something"
padding.**

**Potential framing concern:** The elegance tag "algebraic-identity" overstates the relationship.
The scatter-gather and the sequential filter are NOT algebraically equivalent — they are
empirically equivalent above a threshold. The elegance claim at Stage 5 used this tag to justify
the elegance multiplier. The underlying idea is sound (the sequential dot-product scan IS a
well-motivated approximation to the scatter-gather), but the algebraic-identity framing implies
exact equivalence that does not exist. The elegance multiplier at Stage 5 should be understood
as "elegant APPROXIMATION" not "exact identity." This does not invalidate the pick but the
section 6c check below addresses it.

### 6. Runtime / Scope Sanity

**Runtime table (Section 10):**
| Phase | Time |
|---|---|
| Phase 0 prerequisite | 20 min |
| Phase 1 agreement measurement | 30 min |
| Permutation control | 15 min |
| Random-init control | 10 min |
| Multi-seed (2 extra splits) | 60 min |
| Output | 5 min |
| **Total** | **~140 min** |

The total is ~2.3 hours with all controls. The prompt specifies "phase-gated 50min + 85min
controls" — this matches: Phase 0 (20 min) + Phase 1 (30 min) = 50 min; controls = 15+10+60+5
= 90 min ≈ 85 min stated. **Consistent.**

**Smallest test decides GO:** Phase 0 (20 min) produces a binary GO/NO-GO that gates Phase 1.
Phase 1 (30 min) produces the agreement-rate measurement that decides the primary GO threshold.
The controls (85 min) are required for the full skeptic-controls declaration but do not change
the Phase 0/1 decision. This sequencing is correct — the plan runs the cheapest decisive test
first. **Well-scoped.**

**Cal/eval corpus disjoint check:** Yes — calibration corpus is tokens 0:200; eval corpus is
tokens 201:400 from WikiText-2 validation. For multi-seed: seed 0 = tokens 0:200 (cal) + 201:400
(eval); seed 1 = tokens 201:400 (cal) + 401:600 (eval); seed 2 = tokens 400:600 (cal) + ...
Wait — seed 0 uses tokens 0:200 as calibration and 201:400 as eval. Seed 1 uses 201:400 as
calibration and 400:600 as eval. This means seed 1's calibration tokens are seed 0's eval tokens.
The multi-seed agreement rate for seed 1 may be inflated if there is autocorrelation between
consecutive WikiText-2 validation windows. This is a mild concern but not a fatal flaw since
WikiText-2 is a shuffled corpus. **Note in results whether seed-to-seed cal/eval overlap occurs.**

### 6b. Wildcard Non-Penalization

S6-SEQFILT is not tagged as a wildcard. It is a Stage 4 composition idea. Standard gates apply.

### 6c. Elegant-Equivalence Rejection Ground (Section 6c)

`elegance-class: algebraic-identity`

**Check:** Does the elegance argument depend on a precondition that fails?

The "algebraic identity" framing claims: scatter-gather result ≡ inline threshold test. As noted
in Sections 1 and 5, this is EMPIRICALLY APPROXIMATE, not algebraically exact. The precondition
for the algebraic-identity framing is that the dot-product threshold exactly recovers the same
row set as the dot-product ranking. This holds EXACTLY only if: (a) the threshold τ_L equals
exactly the 18th-largest dot product for every token (not just on average), AND (b) there are
no ties in the dot-product ranking at the threshold boundary.

For real activations, neither (a) nor (b) holds exactly. The claim is therefore NOT an algebraic
identity but a high-probability empirical equivalence.

**Section 6c verdict:** The elegance multiplier applied at Stage 5 is partially misfounded —
the "algebraic-identity" tag overstates the relationship. HOWEVER, this is NOT a fatal flaw.
The underlying idea is still valid: the sequential threshold test is a computationally cheap
APPROXIMATION that achieves ≥ 90% agreement. The elegance argument should be downgraded from
"exact algebraic identity" to "computationally efficient approximation" in the experiment plan,
but this does not change the experiment's validity or the GO/NO-GO structure.

**Stage 6 amendment required (Amendment B2):** Retag elegance-class from "algebraic-identity"
to "subspace-alignment" or "conserved-quantity" (the CF3 Jaccard structure is the conserved
quantity being exploited). The experiment plan should clarify in Section 3 that the scatter-gather
equivalence is a ≥ 90% empirical equivalence, not an exact algebraic identity. This is a framing
correction, not a mechanism change.

### 7. CF13-CF15 Verification Check

No CF13/CF14/CF15 numbers cited in load-bearing positions. The plan cites CF3 (Jaccard=0.308 at
K=1%), CF1 (W_up firing dominance), and v2-CF1 (CF3 generalizes to all 28 layers). All are
v2-confirmed. **SAFE.**

### 8. Verdict — Track B

**ACCEPT-WITH-AMENDMENTS**

Two amendments required:

**Amendment B1 (CF9 scope clarification, structural):** In Section 4 and Section 11 script
description, add measurement of T2-set cardinality per token (mean and std of |SEQFILT T2 set|
across all eval tokens and layers). Report whether SEQFILT produces a fixed-size or variable-size
T2 set per token. If std(|T2 set|) > 5, note this as cardinality instability and add to Section 7
ambiguous-zone follow-up: "try top-K approach instead of threshold approach if cardinality
instability is high." Add `mean_T2_set_size` and `std_T2_set_size` fields to the JSON output.

**Amendment B2 (framing correction, non-blocking):** In Section 2 (Hypothesis) and Section 2
(Class Tags), change "algebraic-identity" to "conserved-quantity" (the CF3 outlier-channel
identity is the conserved structural property being exploited). Add one sentence to Section 3:
"The scatter-gather ≡ sequential-threshold equivalence is an empirical ≥ 90% approximation, not
an exact algebraic identity; the experiment's primary GO criterion is agreement rate, not a
formal identity proof." Remove "algebraic-identity" from the class tags.

---

## Summary Table

| Track | Pick | CF9 | CF10 | Skeptic | Prior Art | Biased Frame | Runtime | 6c Elegant | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| A | A1-GFRS-2 | PASS (A1 amendment) | N/A | PASS (A2 amendment) | NOVEL (A3 citation) | CLEAN | 12 min | PASS | **ACCEPT-WITH-AMENDMENTS (A1, A2, A3)** |
| B | S6-SEQFILT | PASS (B1 amendment) | SAFE | PASS | NOVEL | CLEAN | 2.3 hrs | PARTIAL FAIL (B2 amendment) | **ACCEPT-WITH-AMENDMENTS (B1, B2)** |

---

## Prior-Art Notes for Run 006

- **arXiv:2412.14543 (van Nierop, Dec 2024)** — "Transformer models are gauge invariant" — establishes the gauge invariance mathematical framework for transformers. Any run_006 gauge-symmetry idea should cite this and differentiate from it. The paper covers continuous gauge groups; the Z_2^d discrete subgroup application to quantization is still open.
- **arXiv:2602.18948 (François & Ravera, Feb 2026)** — "Toward Manifest Relationality in Transformers via Symmetry Reduction" — covers reparameterization symmetries and symmetry reduction for transformer Q-K bilinear forms. Relevant anti-frame: do not re-propose "manifest symmetry reduction in Q-K space" without engaging this paper.
- **BTC-LLM (arXiv:2506.12040)** — Sub-1-bit LLM quantization via binary codebooks with sign-function centroid updates. Adjacent to S2-BITSER's downstream. If A1-GFRS-2 GO, S2-BITSER must differentiate from BTC-LLM (BTC-LLM learns binary codebooks through training; BITSER exploits a fixed gauge-theoretic structure post-training).

---

## Runner-Up Status

**Track A runner-up: F2-HPGF.** If A1-GFRS-2 is killed by Stage 6 (it is not — ACCEPT-WITH-AMENDMENTS verdict means escalation is not triggered), F2-HPGF advances directly. F2-HPGF has CF9-clean algebraic grounding (S_16 permutation invariance is exact), no calibration fitting, 5-min experiment. The arXiv:2412.14543 prior art covers continuous gauge groups; S_16 discrete permutation invariance as a compression coordinate choice is not in that paper. F2-HPGF's novelty is intact.

**Track B runner-up: R5-RAOK-FULL.** If S6-SEQFILT Phase 0 A6-CNTR fails, R5-RAOK-FULL advances directly without further Stage 6 review (it passed Stage 5 scoring at 18.4 and has no structural deficiencies identified in Stage 3 or Stage 4 that would require re-review). The runner-up plan is the standard RAOK-FULL plan from Stage 3.
