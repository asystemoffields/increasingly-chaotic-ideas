# Stage 6 — Red Team — Run 002
Run: 002 | Date: 2026-05-09 | Red-teamer: Sonnet claude-sonnet-4-6

---

## Preamble

Two picks under review: Track A (R3-A/XLQB) and Track B (F2/CQBS). Both resolve
to a single combined experiment (`scripts/cqbs_xlqb_combined.py`), reading the
same 57344×2048 stacked W_Q matrix and interpreting cumvar(K=128) against
different thresholds (≥0.50 for XLQB, ≥0.85 for CQBS). Stage 5 explicitly
called this out as "intentional." The central adversarial question is whether this
shared-experiment structure represents genuine signal convergence or selection
collapse that breaks inter-track diversity.

---

---

# ═══════════════════════════════════════════════════════
# TRACK A — R3-A/XLQB
# ═══════════════════════════════════════════════════════

---

## 1. Frame-Mismatch Re-Check (CF9)

**Load-bearing import**: the XLQB plan imports no external theorem. The
stacked-SVD protocol (concatenate 28 W_Q matrices along axis-0, run
`torch.linalg.svd`, compute cumvar) is a standard linear algebra procedure with
no non-trivial precondition. Shape compatibility is trivially satisfied: all 28
W_Q ∈ R^{2048×2048}, so the stacked matrix is 57344×2048 by construction.

**The CF9 subtlety to flag (not a reject, but an amendment)**:

XLQB's section 3 hypothesis conflates two distinguishable claims:

- **(A) "Stacked SVD captures a shared cross-layer subspace"** — true only if the
  within-layer 128-dim subspaces from different layers are mutually aligned.
- **(B) "Stacked SVD captures the concatenation of 28 independent concentrated
  subspaces"** — possible even if layers rotate their subspaces, because the
  stacked SVD sums energy across all blocks. If each block contributes K=128 dims
  independently, the stacked matrix cumvar at K=128 reflects the AVERAGE per-block
  concentration, not cross-layer alignment.

This distinction is the key mismatch: **cumvar(K=128) on the stacked matrix does
NOT by itself distinguish between (A) and (B)**. A stacked matrix of 28
*independent* 128-dim concentrated blocks has cumvar(K=128) that is NOT trivially
equal to K/d. It can be elevated simply because each block contributes its energy
in a different direction — the 28 blocks together span up to 28×128 = 3584 dims,
but the top-128 singular vectors of the stacked matrix capture whichever 128 of
those 3584 dimensions account for the most energy across the full stack.

Concretely: if each W_Q^ℓ has r_99/d ≈ 0.63 (CF11), and the 28 subspaces are
uncorrelated, the top-128 singular values of the stacked matrix will be boosted
relative to true random (expected K/d ≈ 0.063) — not by cross-layer alignment but
by within-layer concentration coinciding in the stacked object.

**This is exactly what the permutation control is designed to catch**, and Stage 5
correctly includes it (Section 9). The gap threshold (real cumvar ≥ permuted cumvar
+ 0.40) is the load-bearing diagnostic. **However, XLQB's Section 5 GO criterion
does NOT require the permutation gap**: it only requires cumvar(K=128) ≥ 0.50.
The permutation control is declared in Section 9 but is NOT wired into the GO
threshold in Section 5.

**CF9 verdict**: the frame is not broken, but the GO threshold in Section 5
is uncoupled from the permutation gap declared in Section 9. This is a structural
defect: if permuted cumvar = 0.48 and real cumvar = 0.52, the GO threshold fires
but the gap is only 0.04 — which Section 9's own text correctly identifies as NOT
a GO. The plan contradicts itself.

**Required amendment (FA-1)**: Section 5 GO criterion must be amended to:
`cumvar(K=128) ≥ 0.50 AND real cumvar ≥ permuted cumvar + 0.40 AND real cumvar ≥
random-init cumvar + 0.40`. The gap requirement is already in the plan's text
(Section 9) — it must appear in Section 5 as well. Without this, a result like
cumvar=0.51 with permuted=0.49 would appear as GO but is actually a null result.

---

## 2. Calibration Ill-Conditioning Re-Check (CF10)

XLQB has zero calibration fitting in its primary measurement. The stacked SVD is
a deterministic computation on the weight matrices. CF10: **N/A — confirmed.**

Secondary metrics (OHAA alignment scores) use 200 calibration tokens. The
multi-seed declaration (Section 9) correctly identifies this and specifies 3 seeds
for that component only. CF10 is correctly handled for the secondary measurement.

---

## 3. Skeptic-Controls Check

**Primary measurement claim**: "cross-layer W_Q basis sharing is a trained-model
structural property, not an architectural artifact."

This IS a claim of the form "X is a property of trained weights," triggering the
skeptic-controls gate.

**Permutation control**: Present (Section 9). Expected permuted cumvar range
(0.35–0.45) is stated. **Defect**: the gap threshold (≥0.40) is in Section 9 but
ABSENT from Section 5's GO criterion. See FA-1 above.

**Random-init control**: Present (Section 9). Expected random-init cumvar range
(≈0.063 to 0.35–0.45 depending on architecture effects) is stated. Gap threshold
(≥0.40) same defect as above.

**Multi-seed**: Correctly declared N/A for primary measurement (SVD is
deterministic). 3-seed declared for secondary calibration metrics. Justification
is valid per STAGE6_RED_TEAM.md criterion.

**Controls verdict**: Controls are intellectually present but the GO threshold in
Section 5 is structurally decoupled from them. **Amendment FA-1 resolves this.**

---

## 4. Hidden Prior-Art Search (Third Pass)

**Critical finding: arXiv:2410.03765 (Basis Sharing, Oct 2024)**

Basis Sharing (Wang et al.) applies SVD to decompose weight matrices in different
layers such that they share a set of basis vectors and have unique per-layer
coefficient matrices. It explicitly covers W_Q, W_K, W_V, W_O. It was tested
post-training on LLaMA, OPT-6.7B, Mistral-7B, GPT-2. This is **direct prior art
for the compression mechanism** (shared basis + per-layer coefficients for W_Q).

Stage 3 and Stage 5 cited only MASA (arXiv:2508.04581) as the closest cousin and
declared NOVEL on the grounds that MASA is "training-time" and that "post-hoc
stacked-SVD measurement on a deployed transformer" is novel. However:

- Basis Sharing (2410.03765) is explicitly **post-training** (it adopts a
  "pretrained LLM" and applies cross-layer SVD decomposition without fine-tuning).
- Basis Sharing covers W_Q specifically.
- Basis Sharing reports the shared-basis + per-layer-coefficient structure (the
  exact mechanism CQBS/XLQB names as its compression primitive).

**The novelty claim must be sharpened.** Basis Sharing does NOT, as far as the
search evidence shows, report a raw measurement of cumvar(K=128) on a stacked
57344×2048 W_Q matrix as a diagnostic. Its approach is algorithmic (find a good
shared basis by optimization), not a diagnostic (measure whether a shared basis
accidentally exists). But the COMPRESSION PRIMITIVE (shared U + per-layer C_L) is
occupied. XLQB's claim to novelty must pivot to: "we are measuring whether the
emergent post-training structure satisfies the shared-basis precondition to the
degree required for the Basis Sharing compression primitive to be lossless at
K=128 — without the optimization step Basis Sharing requires."

**ICLR 2026: LeSTD** (LLM Compression via Learning-based Sparse Tensor
Decomposition) learns a shared orthogonal basis across all heads AND Q/K/V/O
projections in a data-free post-training framework. It targets W_Q explicitly and
uses a shared-subspace Tucker decomposition. It was published at ICLR 2026. This
further colonizes the "shared subspace for W_Q across layers/heads" space.

**COMPOT (arXiv:2602.15200, Feb 2026)** uses sparse dictionary factorization with
orthogonal dictionaries across transformer layers. It explicitly notes that
"enforcing a single shared subspace can degrade accuracy even at moderate
compression" and proposes sparse per-layer coding as a refinement. This is directly
relevant: COMPOT's observation that single-shared-subspace is insufficient at
moderate compression is a prior experimental result that bears on whether
CQBS's binary (shared vs. not) framing is too coarse.

**Third-pass prior-art verdict for XLQB/CQBS**: the compression mechanism
(shared W_Q basis + per-layer coefficients) is **occupied by Basis Sharing (2410.03765)**,
which Stage 3 missed. The post-hoc diagnostic framing ("measure whether emergent
structure satisfies shared-basis precondition") survives as a novelty angle, but
the plan as written claims the GO implies a "15.6× compression primitive" — which
Basis Sharing already proposes as a mechanism. The ladder's value proposition for
this experiment must be repositioned: **not "discover and deploy a new compression
primitive" but "empirically characterize whether Qwen3 W_Q emergently satisfies
the Basis Sharing precondition at the Basis Sharing's own claimed compression
ratios, on a Qwen3 target Basis Sharing did not test."**

This is a **novelty repositioning**, not a kill. The measurement itself remains
worth doing and its outcome resolves a precondition Basis Sharing assumes rather
than measures. The GO-case narrative must be amended.

**Amendment FA-2**: Strike "the 15.6× W_Q storage reduction is a large
single-matrix residency lever" from the GO narrative in Sections 5 and 12, replace
with "cumvar(K=128) ≥ 0.50 confirms the shared-basis precondition that Basis
Sharing (2410.03765) assumes; the measurement quantifies whether Basis Sharing's
approach applies losslessly to Qwen3-1.7B W_Q at K=128." Cite Basis Sharing as
prior-art in the plan's Section 2 equivalent. This also flags LeSTD and COMPOT as
adjacent work to distinguish from.

---

## 5. Biased-Framing Audit

**XLQB GO threshold (0.50) vs CQBS GO threshold (0.85)**: a 70% spread between
the two tracks' GO thresholds on the same measurement creates a very wide
guaranteed-success envelope. If cumvar lands anywhere between 0.50 and 0.85, Track
A fires GO and Track B fires GRAY. This is not biased framing per se — XLQB is
genuinely asking a weaker question — but it does mean the combined experiment
cannot produce a true joint NO-GO on the shared measurement unless cumvar < 0.30
(Track A's NO-GO threshold). The GRAY zone for CQBS (0.70–0.85) overlaps with
XLQB's GO zone (0.50+), creating a structural narrative asymmetry.

**Assessment**: not a reject condition. XLQB's Section 7 (ambiguous-zone follow-up)
correctly handles this. The wide spread is a feature (different questions), not
bias. Flag for record.

**NO-GO structural-kill claim**: XLQB Section 6 states NO-GO kills the "entire C1
convergence cluster." This is load-bearing: if all four orientations (XLQB, CQBS,
CQST, GHPI) fall together, the class-level kill is high-value. The kill is
correctly scoped (within-layer CF11 survives).

**CF2 forcing argument for cross-layer claim**: XLQB Section 3 uses CF2
(cos(h_L, h_{L+1}) ≈ 0.99) to motivate the cross-layer shared subspace. CF2 says
adjacent residual states are nearly parallel — but this is a property of the
RESIDUAL STREAM INPUTS to W_Q, not of the W_Q matrices themselves. High input
cosine similarity doesn't imply the projection matrices have aligned column spaces.
The forcing argument is weak as stated, but it is motivational rather than
load-bearing (the experiment falsifies the question directly). This is not a
reject; it is a framing note.

---

## 6. Runtime / Scope Sanity

**Runtime discrepancy**: Section 4 (Smallest Test) estimates **~22 minutes**. Section
10 (Runtime Table) estimates **~38 minutes** (including controls). The 22-minute
figure in Section 4 omits the permutation control (10 min) and random-init control
(5 min). This is not a fatal discrepancy — Section 10 is the canonical estimate
and is the number the user will schedule from — but it is an internal inconsistency.

**Amendment FA-3**: Section 4 must reference Section 10 as the canonical runtime
estimate. The "~22 minutes" in Section 4 is incomplete (it omits mandatory skeptic
controls). The 8h gate passes on either estimate.

**Scope**: The combined script serves both tracks. One 38-minute run produces all
outputs. Efficient and well-designed.

**Eval corpus**: Not applicable (pure weight-space measurement). Cal/eval disjoint
check: N/A.

---

## 7. CF13–CF15 Verification Check

The plan cites no CF13/CF14/CF15 number in a load-bearing way. Section 12
explicitly flags Qwen3-72B extrapolation as "NOT load-bearing for this rung." The
first rung re-derives the cross-layer structure from scratch.

**CF13–CF15 check: PASSES.**

---

## 8. VERDICT — TRACK A

**ACCEPT-WITH-AMENDMENTS**

Three amendments required before execution:

- **FA-1** (load-bearing): Wire the permutation-gap and random-init-gap
  requirements into Section 5's formal GO criterion. The plan's Section 9 text
  already defines the gaps (≥0.40 each); Section 5 must mirror them. GO = cumvar
  (K=128) ≥ 0.50 AND gap(real, permuted) ≥ 0.40 AND gap(real, random-init) ≥ 0.40.

- **FA-2** (novelty repositioning): Cite Basis Sharing (arXiv:2410.03765) as
  prior-art in any novelty claim. Reframe the GO narrative: the experiment measures
  whether Qwen3 W_Q emergently satisfies the Basis Sharing precondition on a target
  Basis Sharing did not test, not "discovers a new compression primitive." Also note
  LeSTD (ICLR 2026) and COMPOT (arXiv:2602.15200) as adjacent work.

- **FA-3** (minor): Update Section 4 runtime to reference Section 10 (38 min with
  controls). The 22-minute figure is incomplete.

The underlying measurement is sound and high-value regardless of Basis Sharing
prior art: Basis Sharing didn't test Qwen3 and didn't report cumvar(K=128) as a
diagnostic. The experiment proceeds with the amended framing.

---

---

# ═══════════════════════════════════════════════════════
# TRACK B — F2/CQBS
# ═══════════════════════════════════════════════════════

---

## 1. Frame-Mismatch Re-Check (CF9)

**Load-bearing imports**: identical to XLQB — standard SVD, no non-trivial
preconditions. CF9: CLEAR for the mechanical protocol.

**Section 6c (elegant-equivalence check)**: CQBS carries `elegance-class:
subspace-alignment`. The elegant-equivalence claim is: "W_Q^ℓ x ≈ U (C_L x)
where the approximation error is precisely 1 − cumvar(K)." This is algebraically
correct: SVD gives the best rank-K approximation in Frobenius norm, so the claim
holds for the matrix reconstruction. The elegance argument does NOT depend on any
non-commuting operators or hidden gauge constraints (unlike LFAO's per-channel γ).

**However**, the elegance is conditional: the claim that "a single shared basis U
reconstructs ≥85% of the Frobenius mass of ALL 28 W_Q matrices" requires cumvar
≥ 0.85 — it doesn't claim elegance prior to the measurement. The elegance
multiplier (1.2×) was applied by Stage 5 because "the construction is
algebraically exact." That is true for the reconstruction *if* the cumvar threshold
is met. It is not a tautology — the experiment must be run to verify whether the
structure exists. The elegance-class tag is legitimate and the Section 6c check
passes: no precondition failure.

**The same CF9 subtlety as XLQB applies**: cumvar(K=128) on the stacked matrix
conflates (A) "28 subspaces are mutually aligned" and (B) "28 independent
concentrated subspaces happen to boost stacked cumvar." The permutation control
resolves this. **Unlike XLQB**, CQBS Section 5 explicitly states "cumvar(K=128)
≥ 0.85 AND real cumvar ≥ permuted cumvar + 0.35 AND real cumvar ≥ random-init
cumvar + 0.35." The gap thresholds ARE wired into the GO criterion in Section 5.

**CF9 verdict**: CLEAR. CQBS's GO criterion is well-formed — it already includes
the alignment-vs-concatenation disambiguation. CQBS does not have the FA-1
defect of XLQB.

---

## 2. Calibration Ill-Conditioning Re-Check (CF10)

Identical to XLQB: zero calibration fitting in the primary measurement. CF10:
**N/A — confirmed.**

---

## 3. Skeptic-Controls Check

**Permutation control**: Present and correctly wired into the GO criterion (gap
≥ 0.35 in Section 5). The reasoning for non-trivial permuted baseline (≈0.35–0.45)
is stated. **The gap threshold of 0.35 in CQBS vs 0.40 in XLQB is a minor
discrepancy** — both plans reference the same experiment, but XLQB Section 9
specifies ≥0.40 and CQBS Section 5 specifies ≥0.35. One combined script produces
one set of outputs. The combined `cumvar_table.csv` has a single permuted column.
The interpretation gate should be the MORE CONSERVATIVE threshold (≥0.40).

**Amendment FB-1**: Resolve the 0.35 vs 0.40 discrepancy in favor of the
stricter gap requirement (≥0.40). This aligns CQBS Section 5 with XLQB Section 9
and applies consistently across both tracks.

**Random-init control**: Present and wired into CQBS Section 5's GO criterion
(gap ≥ 0.35). Same discrepancy — should be ≥ 0.40.

**Multi-seed**: Correctly declared N/A for primary; 3-seed for OHAA secondary.
Valid.

**Controls verdict**: Skeptic controls are properly wired in CQBS (unlike XLQB).
One minor gap-threshold discrepancy (FB-1). Controls gate: PASSES with amendment.

---

## 4. Hidden Prior-Art Search (Third Pass)

All prior-art findings from Track A Section 4 apply equally:

**Basis Sharing (arXiv:2410.03765)**: the compression mechanism CQBS claims as its
GO output (shared U + per-layer C_L for W_Q) is explicitly described and applied
by Basis Sharing in a post-training setting. Stage 3 missed this paper — it cited
only MASA (training-time) and did not engage Basis Sharing (post-training).

The CQBS novelty claim must be repositioned (same amendment as FA-2): the
measurement is a diagnostic that characterizes whether Qwen3-1.7B W_Q emergently
satisfies the Basis Sharing precondition at Basis Sharing's claimed compression
ratios. The experiment's value is real — Basis Sharing did not test Qwen3, did not
report cumvar(K=128) as the threshold diagnostic, and did not measure whether the
shared basis arises naturally vs. requiring optimization. But the "15.6× compression
primitive" framing in CQBS Section 1, 5, and 12 must cite Basis Sharing and
acknowledge the mechanism is prior art.

**COMPOT (arXiv:2602.15200)**: COMPOT's observation that "enforcing a single shared
subspace can degrade accuracy even at moderate compression" is directly relevant to
CQBS's GO cascade. On GO, CQBS advances to "coefficient reconstruction quality
check (Rung 2)." COMPOT's finding suggests the single-shared-subspace format may
hit accuracy limits before reaching 15.6× compression. This is informative for the
cascade plan (Rung 2 should test reconstruction quality against COMPOT's accuracy
bound), not a kill for the measurement itself.

**Amendment FB-2**: Same as FA-2. Additionally, in Rung 2 (post-GO), cite COMPOT
as the reference point for expected accuracy ceiling of the single-shared-subspace
format.

**Third-pass prior-art verdict**: same as XLQB. Measurement proceeds; narrative
is repositioned.

---

## 5. Biased-Framing Audit

**GO threshold (0.85)**: is this gerrymandered? A GO at cumvar ≥ 0.85 means 85% of
the total Frobenius mass of all 28 W_Q matrices is captured by 128 dimensions out
of 2048 (6.25%). For comparison, CF11 shows that within-layer, 128 dims capture
enough variance to hold ΔNLL ≤ +0.98 — but that is a functional test, not a
Frobenius-cumvar measurement. The 0.85 threshold is plausible but not calibrated
against a published reference for cross-layer Frobenius cumvar. It is aggressive
(a high bar for GO, meaning a modest result won't inflate the outcome). This is
the correct direction of bias — conservative GO thresholds are appropriate.

**NO-GO claim (cumvar < 0.70)**: the Section 6 text says NO-GO kills "cross-layer
W_Q basis sharing as a post-training compression primitive." This is correctly
scoped. The structural finding is real — it would extend the CF8/CF11 spectral
characterization to the cross-layer dimension.

**Section 3 elegance framing**: "the approximation error is precisely 1 − cumvar(K)"
is factually correct for the Frobenius reconstruction. This is not over-claimed.

**No biased-framing issues identified beyond the Basis Sharing citation gap.**

---

## 6. Runtime / Scope Sanity

Section 10 estimates 44 minutes (CQBS's estimate is slightly longer than XLQB's
38 minutes because it adds 3 seeds for OHAA × 200 tokens). Both plans reference
the same script. The canonical runtime should be the longer one (44 min) if the
combined script runs the full CQBS protocol. **Well within 8h gate.**

**Amendment FB-3** (minor): Section 4 says "~38 minutes" (same as XLQB), but
Section 10 says "~44 minutes." The discrepancy is the 3×200-token OHAA seeds
(3 min). Section 4 should reference Section 10 for the canonical runtime.

---

## 7. CF13–CF15 Verification Check

Same as XLQB: no CF13/CF14/CF15 numbers are load-bearing. Qwen3-72B extrapolation
explicitly flagged as not load-bearing. First rung is self-contained.

**CF13–CF15 check: PASSES.**

---

## 8. Section 6b — Wildcard Non-Penalization

Neither pick is tagged `[FREE SWING]` or `[OUT-OF-ORIENTATION]`. Not applicable.

---

## 9. Selection Collapse Audit — Diversity Check

Both tracks selected the same measurement (stacked W_Q SVD). The question is
whether this represents (a) genuine convergence on a strong signal or (b) Stage 5
selection collapse.

**Assessment: genuine convergence, not collapse.** The four-orientation cluster
(XLQB/R, CQBS/F, CQST/C, GHPI/A) is structurally motivated by CF11, not by a
single narrative. Four independent Stage 1 ideators converged on the same
empirical question via different reasoning paths. The Stage 4 skeptic (SE-1)
confirmed the cluster is "saturation-exhausted for generation but not for
execution." Stage 5's decision to run them as one combined experiment is correct
efficiency: the experiment settles all four orientations simultaneously.

**However, inter-track diversity is broken.** Running the same experiment in both
tracks means that if the combined script succeeds or fails, Track A and Track B
produce correlated outcomes. The experiment budget is spent twice on one question.
The pipeline has two tracks precisely to explore diverse hypotheses in parallel.

**Stage 5's explicit justification** ("the convergence cluster is so strong that
both tracks are pointing at the same empirical question") is a valid argument for
the cluster being highly credible — but it is also an argument for resolving the
cluster in one track and spending the other track's budget on an independent
question.

**Amendment FB-4 (diversity — recommended, not required)**: Consider redirecting
Track B to F1/JVOC (runner-up, 27.2 points) to test the independent W_V/W_O
fused-operator hypothesis. This would: (a) settle the C1 cluster via Track A alone
(the cluster is strong enough), (b) open the C2 cluster (W_V/W_O characterization)
in parallel, (c) double the pipeline's information gain per round. If CQBS fires
GO, JVOC's outcome tells us whether the attention-weight concentration generalizes
to V/O; if CQBS fires NO-GO, JVOC provides an independent compression path in the
same round. The arguments for the C1 cluster being high-priority are real, but
not so overwhelming that both tracks must be consumed by the same question.

**This amendment is advisory.** If the user prefers the high-confidence C1
resolution with both tracks confirming, the combined run is acceptable. The
pipeline should note that Track B is not independently exploring; it is providing a
threshold-interpretation variant on the same measurement.

---

## 10. VERDICT — TRACK B

**ACCEPT-WITH-AMENDMENTS**

Three amendments required before execution; one advisory amendment for diversity:

- **FB-1** (load-bearing): Resolve permutation-gap and random-init-gap threshold
  discrepancy in favor of ≥0.40 (matching XLQB Section 9). Update CQBS Section 5
  GO criterion to: cumvar(K=128) ≥ 0.85 AND gap(real, permuted) ≥ 0.40 AND
  gap(real, random-init) ≥ 0.40.

- **FB-2** (novelty repositioning): Cite Basis Sharing (arXiv:2410.03765) as
  prior art for the shared-U + per-layer-C_L W_Q compression mechanism. Reframe:
  the measurement quantifies whether Qwen3 W_Q emergently satisfies the Basis
  Sharing precondition at K=128, on a target Basis Sharing did not test. Also note
  LeSTD (ICLR 2026) and COMPOT (arXiv:2602.15200). In Rung 2 (post-GO), benchmark
  reconstruction accuracy against COMPOT's finding that single-shared-subspace
  degrades at moderate compression.

- **FB-3** (minor): Section 4 runtime should reference Section 10 (~44 min).

- **FB-4** (advisory — diversity): Consider redirecting Track B to F1/JVOC to
  preserve inter-track diversity. If this redirect is accepted, Track A alone
  resolves the C1 cluster with the full CQBS-OR-XLQB threshold range; Track B
  opens the C2 (W_V/W_O) cluster in parallel.

---

---

# SUMMARY TABLE

| Check | Track A (XLQB) | Track B (CQBS) |
|---|---|---|
| CF9 frame-mismatch | CLEAR — amendment FA-1 for GO-criterion gap wiring | CLEAR — GO criterion already wired correctly |
| CF10 calibration | N/A (no calibration fitting) | N/A (no calibration fitting) |
| Skeptic controls | Present; gap not wired into GO (FA-1) | Present and wired (FB-1 gap discrepancy only) |
| Prior-art third pass | Basis Sharing (2410.03765) missed — novelty reframe (FA-2) | Same — novelty reframe (FB-2) |
| Biased framing | Wide GO/NO-GO band; not biased; CF2 forcing argument weak but motivational only | Conservative GO at 0.85; appropriate direction |
| Runtime/scope | FA-3 minor (22 vs 38 min inconsistency) | FB-3 minor (38 vs 44 min inconsistency) |
| CF13–CF15 | PASSES | PASSES |
| Selection collapse | Genuine convergence signal; but both tracks consume same experiment | Diversity amendment FB-4 recommended |
| **Verdict** | **ACCEPT-WITH-AMENDMENTS (FA-1, FA-2, FA-3)** | **ACCEPT-WITH-AMENDMENTS (FB-1, FB-2, FB-3; FB-4 advisory)** |

---

## Key Prior-Art Third-Pass Findings (for KILL_LIST / future rounds)

1. **arXiv:2410.03765 (Basis Sharing, Oct 2024)**: post-training cross-layer SVD
   with shared W_Q/W_K/W_V/W_O basis + per-layer coefficients. Directly occupies
   the XLQB/CQBS compression mechanism. Stage 3 missed it (cited only MASA as
   closest cousin). Not a kill for this measurement, but the GO-case narrative
   claiming "discovers a new primitive" is incorrect. Future rounds proposing any
   "cross-layer shared basis for attention weights" mechanism must engage this paper.

2. **LeSTD (ICLR 2026)**: data-free post-training shared orthogonal basis for all
   attention heads (Q/K/V/O). Tucker decomposition with shared factor matrices.
   Further colonizes the shared-subspace-for-W_Q space.

3. **COMPOT (arXiv:2602.15200, Feb 2026)**: explicitly finds that single-shared-
   subspace SVD degrades accuracy at moderate compression. Provides the expected
   accuracy ceiling for the CQBS GO-cascade Rung 2.

4. **Gated Subspace Inference (arXiv:2605.03109, May 2026)**: exploits low effective
   rank of token activation manifold per layer for inference acceleration. Uses
   cached low-rank weight image. Not directly about W_Q cross-layer sharing, but
   occupies adjacent "low-rank activation subspace" territory for acceleration.
   Not a kill but should be cited in cascade planning.

5. **On the Invariants of Softmax Attention (arXiv:2605.02907, May 2026)**: defines
   energy field and shows key incoherence across architectures. Relevant to XLQB's
   framing of "CF2 forces cross-layer W_Q alignment" — the key incoherence finding
   actually complicates that narrative.

---

*End of Stage 6 Red Team — Run 002*
