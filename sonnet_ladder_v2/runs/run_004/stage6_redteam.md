# Stage 6 — Red Team — Run 004 (v2 Sonnet Ladder)

Red-team agent: Sonnet 4.6.
Date: 2026-05-09.
Inputs: STAGE6_RED_TEAM.md, stage5_selector.md, stage3_deep_research.md, SUMMARY.md, KILL_LIST.md. WebSearch used.

Picks under review:
- **Track A — F-WVOS** (W_V/W_O Algebraic Fold, ~76 min, algebraic-identity, Cluster C1 6-orientation). Runner-up: A-CADH.
- **Track B — F-WNORM** (W_down Rank Sweep / CF8 Closure, ~42–57 min, algebraic-identity, Cluster C3 2-orientation). Runner-up: A-TROP.
- **Co-flight — F-AERO-Q3** (10-min pre-flight on Track A slot, not a primary pick, wildcard scope).

---

## 1. Frame-Mismatch Re-Check (CF9)

### Track A — F-WVOS

**Load-bearing algebraic claim:** The fold `O = softmax(QK^T) · x · W_V · W_O` can be replaced by `O = softmax(QK^T) · x · V_V^T · W_VO` where `W_VO = W_O · U_V[:, :r] · diag(S_V[:r])`. This fold is mathematically correct — it is a standard SVD low-rank approximation of a matrix product, with the only error being the SVD truncation. No imported theorem with external preconditions; the algebra is self-contained. **CF9 CLEAN for the algebraic identity claim itself.**

**GQA shape arithmetic:** Stage 3 raised and Stage 5 forwarded a mandatory pre-check: verify `model.layers[0].self_attn.v_proj.weight.shape`. In Qwen3-1.7B with GQA (8 KV heads × 128 dim), W_V is shape (1024, 2048), not (2048, 2048). Stage 5's residency arithmetic uses the (1024, 2048) shape throughout and is internally consistent: rank-256 fold produces `(256×2048) + (2048×256) = 1M params` vs original `1024×2048 + 2048×2048 = 6.3M params`. The 6× headline becomes ~6×. The shape note is already in Section 3 and Section 8 kill criterion 1. **No new CF9 concern here — the GQA shape issue was correctly flagged and forwarded.**

**Spectral Lifecycle import:** Stage 3 and Stage 5 cite arXiv:2604.22778 as supporting evidence that V/O projections "compress uniformly." Independent verification: the paper is confirmed (search returns it directly). Its title confirms the "Q/K–V Asymmetry" finding: value/output projections compress uniformly while Q/K carry depth-dependent dynamics. However, this paper tracks 30M–285M parameter models during training. Qwen3-1.7B is a different scale and architecture (SwiGLU decoder, GQA). The paper's claim is a training-dynamics observation, not a theorem about post-training Qwen3 spectra. **The import is used as motivation, not as a theorem with preconditions — this is appropriate. The H1 hypothesis is falsifiable by the 25-min SVD; the Spectral Lifecycle paper is background support, not a load-bearing precondition.**

**CF9 verdict for F-WVOS: CLEAN.** No load-bearing theorem imported with unverified preconditions.

---

### Track B — F-WNORM

**Load-bearing algebraic claim:** The experiment is a pure SVD measurement with no algebraic fold. The "algebraic-identity" elegance tag means it applies the same auditable R3 W_gate sweep protocol. There is no imported theorem. **CF9 CLEAN.**

**Permutation control logic:** The Stage 5 plan notes that row-permutation does NOT change singular values (by construction — row permutation is an orthogonal transformation), so the permutation control in F-WNORM serves as a SOFTWARE CORRECTNESS CHECK rather than a trained-vs-random discriminator. The plan explicitly documents this. This is a subtle but important distinction: if the pipeline's control framework expects the permutation gap to discriminate trained from architectural, F-WNORM must document the waiver clearly. **The Stage 5 plan does document this in Section 9.** No CF9 concern.

**Asymmetric shape framing:** W_down is (2048, 6144): the SVD gives min(2048,6144)=2048 singular values. The plan correctly frames r_99 relative to d_out=2048, not d_in=6144. This framing is appropriate given the matrix shape. **No CF9 concern.**

**CF9 verdict for F-WNORM: CLEAN.**

---

## 2. Calibration Ill-Conditioning Re-Check (CF10)

**F-WVOS:** No calibration fit anywhere in the experiment. The fold is computed from SVD of weight matrices (deterministic). NLL eval uses WikiText-2 held-out. No parameters are fit from data. **CF10 not triggered.**

**F-WNORM:** No calibration fit in the primary measurement (SVD only). The optional A-TROP joint measurement uses 200 calibration prompts to compute argmax stability fractions — but this is a threshold-counting statistic, not a parameter fit. n_params_to_fit = 0 for A-TROP stability fraction. **CF10 not triggered.**

**Both picks: CF10 CLEAN.**

---

## 3. Skeptic-Controls Check

### Track A — F-WVOS

**Hypothesis shape:** "W_V concentration is a trained property of Qwen3-1.7B-Base (not an architectural artifact); the fold produces meaningful compression." This is a transfer/consistency claim — controls are mandatory.

**Permutation control:** Present and specific. Section 9 specifies: row-permute each W_V before SVD; GO requires trained r_99/d ≤ 0.80 AND permuted r_99/d > 0.90; fold NLL with permuted W_V must be ≥ 5.0 nats. Gap threshold (≥ 0.10 in r_99/d) is explicitly stated. **PRESENT. Adequate.**

**Random-init control:** Present and specific. Section 9 specifies: `torch.manual_seed(42)` random-init of Qwen3-1.7B architecture; GO requires random-init r_99/d > 0.90; runtime ~15 min. **PRESENT. Adequate.**

**Multi-seed:** Present and scoped. Section 9 notes SVD is deterministic (no random component); multi-seed applied to NLL eval via 3 disjoint 512-token WikiText-2 segments (seeds 42, 123, 456). Mean ± std reported; worst-of-3 must not exceed 2.0 nats NO-GO. **PRESENT. Adequate.**

**One gap flagged:** The GO criterion in Section 5 states the permuted-W_V ΔNLL threshold as ≥ 5.0 nats, but Section 8 kill criterion 3 sets a softer kill floor: "Permuted-W_V ΔNLL < 3.0 nats → kill." These are inconsistent. If permuted ΔNLL = 4.0 nats, Section 5's GO threshold is not met (4.0 < 5.0) but Section 8's kill criterion is not triggered (4.0 > 3.0). The gray zone 3.0–5.0 is undocumented. **Amendment A1: tighten to a single gap threshold: GO requires permuted-W_V ΔNLL ≥ 4.0 nats AND real-model ΔNLL ≤ 0.5 nats; KILL if permuted ΔNLL < 4.0 nats. The current two-threshold inconsistency creates a post-hoc spin zone.**

**Skeptic-controls verdict for F-WVOS: PASS with amendment A1.**

---

### Track B — F-WNORM

**Hypothesis shape:** "W_down spectral structure is a trained property relevant to compression viability, not a purely architectural artifact." This is a structural measurement, not a transfer claim. The primary expected outcome (CF8-CONFIRM full-rank) is a structural finding, not "X works/transfers."

**Permutation control:** Present but repurposed. Section 9 explicitly documents that permutation CANNOT discriminate trained-from-architectural for this experiment (singular values are invariant to row permutation by construction) and repurposes the control as a correctness check. This waiver is justified — the control gate's permutation test does not apply when the claim is "W_down is full-rank" (not "trained concentration exists"). **The waiver is sound and documented.**

**Random-init control:** Present and specific. Section 9 specifies kaiming_uniform_ random init; Marchenko-Pastur prediction: random-init r_99/d ≈ 0.95–1.0; trained r_99/d ≈ random-init ≈ 1.0 in the CF8-CONFIRM case; trained r_99/d << random-init in the SURPRISE GO case. Both outcomes produce clear signal. Runtime ~15 min. **PRESENT. Adequate.**

**Multi-seed:** Section 9 states "multi-seed not applicable to SVD measurement; NLL sweep (if run) uses 3 dataset seeds." The justification is sound — SVD is deterministic; multi-seed applies to NLL only if the SURPRISE GO path is taken. **The "not applicable" waiver is correct and documented.**

**One concern:** The skeptic-controls GO criterion in Section 5 states: "the real trained r_99/d must differ materially from random-init r_99/d by ≥ 0.10." Then it notes that for the CF8-CONFIRM case, trained ≈ random-init ≈ 1.0, and says this is "EXPECTED AND CORRECT." This means the mandatory ≥ 0.10 gap threshold would FAIL in the expected outcome — the randomly initialized matrix is also full-rank. The plan acknowledges this but the GO criterion language is self-contradictory: it states the gap is mandatory, then immediately says the expected case doesn't meet it. **Amendment A2: rewrite the skeptic-controls gate for F-WNORM to have two branches: (a) CF8-CONFIRM branch: gap N/A because trained and random-init are both expected full-rank (r_99/d ≈ 1.0); structural finding is confirmed by trained ≈ 1.0 AND random ≈ 1.0 (null hypothesis holds); (b) SURPRISE GO branch: trained r_99/d ≤ 0.75 AND random-init r_99/d > 0.90 required (gap ≥ 0.15 confirms learned structure). The current single-gap-threshold language is logically inconsistent with the expected outcome.**

**Skeptic-controls verdict for F-WNORM: PASS with amendment A2.**

---

## 4. Hidden Prior-Art Search (Third Pass)

### Track A — F-WVOS: LatentLLM / A3 / MHA2MLA / Spectral Lifecycle

**LatentLLM (MERL TR2026-018):** Search confirms this paper "jointly decomposes value and output projections" and evaluates on "Qwen3 models at different sizes for 10–40% size reduction." The pipeline differentiator is "zero-shot pure-SVD without any fine-tuning." LatentLLM uses "activation-aware transform," which requires calibration data and activation statistics — this is NOT zero-shot pure weight SVD. **The differentiator (zero-shot vs activation-aware) survives.** However, LatentLLM does include Qwen3 results, and their Qwen3 OV decomposition measurements will report r_99-equivalent metrics indirectly via perplexity. The pipeline's private claim is the first direct r_99/d measurement on Qwen3-1.7B W_V — distinct from LatentLLM's perplexity numbers at compression ratios. **Differentiator intact but thin. Watch: if LatentLLM's Qwen3 perplexity at 20% compression reveals the spectrum is NOT concentrated, F-WVOS H1 would be refuted before running. Stage 6 cannot access the full LatentLLM PDF; the search confirms Qwen3 is included in their evaluation but does not confirm which specific matrices they report and at what compression ratio.**

**A3 (arXiv:2505.12942, v3 Jun 2025):** This paper performs analytical low-rank approximation of OV projections (W_V and W_O jointly) using "per-head attention output error minimization" involving the symmetric matrix square root. It achieves LLaMA 3.1-70B WikiText-2 perplexity of 4.69. This is POST-TRAINING, NO RETRAINING, and covers the OV (W_V × W_O) block. Search confirms this paper is v3 (updated Jun 2025) and targets Llama-family models; Qwen3 coverage is not confirmed in search results. **A3's method is closer to F-WVOS than LatentLLM because it is zero-shot. The key discriminators: (a) F-WVOS operates on Qwen3-specific GQA structure with the (1024×2048) W_V shape; (b) A3 uses attention-logit error minimization, not pure Frobenius SVD; (c) A3 appears to target Llama models with separate W_V and W_O matrices, not GQA with shared KV heads.** If A3 v3 (Jun 2025) added Qwen3 results, the measurement novelty for F-WVOS would be significantly reduced. **RISK ELEVATED: A3 is the closest prior art found in this pass. Stage 5 noted "adjacent" at Stage 3 but did not flag A3 as a near-miss for the zero-shot OV decomposition claim. This warrants an amendment rather than a kill because A3's loss function (attention-logit error) differs from pure weight-space SVD.**

**Amendment A3 (prior-art protection):** Before running F-WVOS, confirm A3 v3 does NOT include Qwen3-1.7B GQA results. If A3 covers Qwen3 at equivalent compression ratios with equal-or-better quality, pivot to the joint activation-weighted variant (F-WVOS+) to retain the Qwen3-specific empirical moat.

**MHA2MLA (arXiv:2502.14837):** Confirmed requires fine-tuning (0.3–0.6% of pretraining tokens). The zero-shot differentiator survives against MHA2MLA.

**Spectral Lifecycle (arXiv:2604.22778):** Confirmed. Covers training dynamics, not post-training Qwen3-1.7B r_99/d measurement. Not a pre-emption.

**Hidden prior-art finding for F-WVOS: A3 is an elevated risk.** Not a kill — A3's method differs and Qwen3 coverage is unconfirmed. But A3 is closer than Stage 3/5 acknowledged. Amendment A3 required.

---

### Track B — F-WNORM: ASVD / ARA / SVD-LLM

**ARA (arXiv:2510.19389, ICLR 2025):** Adaptive Rank Allocation applies SVD to all linear layers including W_down in Llama models with global rank optimization. Search confirms it covers "all three MLP matrices." However, it does not measure Qwen3-specific W_down r_99/d numerically — it uses ARA's per-layer rank assignment heuristic to compress. **No r_99/d characterization of Qwen3 W_down is reported. F-WNORM's private measurement is not pre-empted.**

**ASVD (arXiv:2312.05821):** Covers activation-weighted SVD on all weight matrices. The Stage 5 plan's activation-weighted follow-up path (Section 7, SURPRISE GO gray zone) is adjacent to ASVD. However, ASVD doesn't measure Qwen3-1.7B W_down r_99/d. **Not a pre-emption of the structural measurement claim.**

**Compound finding:** Multiple published papers have applied SVD to W_down in various LLMs, but none (found in this pass) have characterized Qwen3-1.7B W_down r_99/d relative to d_out=2048 as the primary measurement. **F-WNORM's structural measurement is intact.**

**Hidden prior-art verdict for F-WNORM: CLEAN.** No new pre-emption found.

---

## 5. Biased-Framing Audit

### Track A — F-WVOS

**GO threshold gerrymandering check:** Primary GO requires mean r_99/d ≤ 0.80 AND ΔNLL ≤ 0.5 nats at K=256. The CF-ground for 0.5 nats is CF11's W_Q result at K=128 (+0.98 nats); the plan argues W_V at K=256 (larger rank budget) should meet ≤ 0.5 nats if spectrum is at least as concentrated. This reasoning is CONDITIONAL on W_V being as concentrated as W_Q. If W_V r_99/d is 0.75 (less concentrated than W_Q's 0.63), the K=256 budget may not meet 0.5 nats. **The threshold is reasonable but the Soft GO at K=512 (≤ 0.5 nats) provides a fallback that covers the case where W_V is less concentrated than W_Q.** No obvious gerrymandering.

**GRAY zone post-hoc spin check:** GRAY zone is r_99/d in 0.80–0.90. The follow-up (activation-weighted SVD) is specific and adds 30 min. The GRAY outcome is not "we'll learn something" — it resolves via a specific additional measurement. **Acceptable.**

**Structural-finding-on-NO-GO specificity:** NO-GO claim is "CF8 extends to W_V; W_V fold provides no compression." This is specific: it kills VOKV, C-VKSD, A-GFWV, and closes the W_V compression question. **Load-bearing, not rhetorical.**

**One mild flag:** The stage 5 plan states the fold reduces residency "6× per layer" but the fold stores two matrices — V_V^T (r×2048) and W_VO (2048×r) — which together equal 2×r×2048 = 2×256×2048 = 1M params at r=256. The originals are W_V (1024×2048) + W_O (2048×2048) = 6.3M params. The ratio is 6.3×, not "up to 4×" as stated in the one-line description (Section 1). The 4× figure and 6× figure are both present in different places. **Amendment A4: standardize the compression ratio claim. At K=256, combined storage = 1M params; original = 6.3M params; ratio ≈ 6.3×. The 4× figure appears to use the wrong original size (possibly 2×2048×1024 = 4.2M vs 2048×2048 + 1024×2048 = 6.3M). Fix to a single canonical number.**

### Track B — F-WNORM

**GO threshold check:** The primary GO is CF8-CONFIRM (r_99/d_out ≥ 0.90 in ≥ 20/28 layers). This is essentially a threshold on a metric expected to be near 1.0 — extremely easy to meet if CF8 prediction is correct. This is not gerrymandering; it is an appropriately low bar for a confirmation experiment. **No concern.**

**NO-GO class-level kill specificity:** "W_down full-rank in Qwen3-1.7B; no-retraining MLP weight compression at all three matrices is dead." This is a specific, clean kill. **Acceptable.**

**Framing consistency:** The plan correctly frames the CF8-CONFIRM outcome as the pipeline-valuable result (not a "failure"). This framing is honest and appropriate. **No bias detected.**

---

## 6. Runtime / Scope Sanity

### Track A — F-WVOS (~76 min per table)

The runtime table adds to: 5 (setup) + 20 (SVD) + 1 (gate) + 5 (fold) + 15 (NLL 3-seed) + 10 (perm control) + 15 (random-init) + 5 (post) = **76 min.** The plan says ~76 min and the table confirms exactly 76 min. **Runtime claim is internally consistent.**

The GRAY zone follow-up adds 30 min → 106 min total. The "8-hour gate" remains easily met.

One scope observation: the random-init control in Section 4, Step 7 says "~15 min SVD only; no NLL eval." The table entry is "15 min." The permutation control runs SVD + fold + NLL on one permuted variant — 10 min. These scope decisions are defensible; no bloat detected.

**Eval/calibration corpus:** No calibration corpus (pure weight-based). Eval corpus: WikiText-2 held-out 512 tokens. The plan reuses the established v2 ladder eval corpus. **Disjoint by construction (no calibration corpus).**

**Scope verdict for F-WVOS: CLEAN.** 76 min is realistic for the described work.

### Track B — F-WNORM (~42–57 min per table)

CF8-CONFIRM path: 3+20+1+5+15+10+3 = **57 min.** Table says 42 min. **DISCREPANCY.** The table shows: Setup 3 + Step 1 20 + Gate 1 + NLL 15 (BUT notes "ONLY if SURPRISE GO") + Permutation 5 + Random-init 15 + A-TROP joint 10 (optional) + Post 3 = 72 min with all optionals, 42 min if NLL and A-TROP are both excluded, 57 min with random-init but no NLL and no A-TROP. The "42 min (CF8-CONFIRM path)" in the table appears to EXCLUDE A-TROP (10 min) and random-init control (15 min) from the total — but Section 9 states random-init control is MANDATORY. If mandatory controls are excluded from the "CF8-CONFIRM path" total, the stated 42 min is understated. **Amendment A5: re-calculate the CF8-CONFIRM path total as 3+20+1+0+5+15+0+3 = 47 min (NLL and A-TROP excluded, mandatory controls included). The table total "42 min" appears to exclude the random-init control. Fix to 47 min minimum.**

**Smallest-test produces a deciding number:** The gate check at Step 2 (mean r_99/d_out ≥ 0.90 in ≥ 20/28 layers) is a direct GO/NO-GO decision, not a noisy signal. The random-init control is a correctness verification, not a decision gate. The A-TROP bonus is additive information. **The experiment structure is sound.**

---

## 6b. Wildcard Non-Penalization

F-AERO-Q3 (co-scheduled as 10-min pre-flight) is not a Stage 5 primary pick and is not tagged as a wildcard — it is a co-scheduled measurement nominated by Stage 5 as a no-additional-cost bonus. It operates under the standard pipeline rules. Stage 6 does not need to adjudicate wildcard status for F-AERO-Q3 as a secondary scheduled item.

---

## 6c. Elegant-Equivalence Rejection Ground

### Track A — F-WVOS (elegance-class: algebraic-identity)

The elegance claim: the fold `O = softmax(QK^T) · x · W_V · W_O` → `O = softmax(QK^T) · x · V_V^T · W_VO` is exact (up to SVD truncation error). The precondition for this algebraic identity: any matrix W_V can be written as U_V Σ_V V_V^T via SVD. This is unconditionally true; SVD exists for any matrix. The elegance does NOT depend on W_V being low-rank — the fold is always valid algebraically. The compression benefit depends on W_V being low-rank (so that r << min(d_V, d_model)), but the algebraic identity itself has no additional preconditions. **The elegance-class claim is correctly categorized and the precondition always holds. CF9 section 1 PASSED in Section 1 above. The elegance multiplier at Stage 5 is not covering a math error.**

**One potential LFAO-pattern check:** Are there any per-head scale factors (analogous to RMSNorm γ) that would prevent the fold from being applied jointly to the GQA W_V? In GQA, W_V is stored as a single (1024×2048) matrix covering all 8 KV heads (not as 8 separate per-head matrices). The fold applies to this monolithic matrix. There is no per-head γ analog in W_V. The attention output for each head reads a slice of V=xW_V before applying W_O — the fold composes correctly because W_O operates on the concatenated heads. **No LFAO-pattern issue detected.**

### Track B — F-WNORM (elegance-class: algebraic-identity, same protocol as R3)

The elegance claim is that the R3 W_gate sweep protocol is reused exactly. The protocol IS correct (R3 already ran; the SVD measurement is valid). No new algebraic claim. **No CF9 violation via elegance class.**

---

## 7. CF13–CF15 Verification Check

**F-WVOS:** Stage 5 plan cites CF11 (W_Q global K=128 at ΔNLL=+0.98 nats) as the CF-ground for the 0.5-nat GO threshold. CF11 is confirmed (v2-confirmed, AQFKV experiment). The plan does NOT cite CF13–CF15. Spectral Lifecycle (arXiv:2604.22778) is cited as background motivation only, not as a load-bearing number. **No CF13–CF15 dependency. CLEAN.**

**F-WNORM:** Cites CF4 (W_gate full-rank), CF5 (W_up more rank-sensitive), CF7/CF8 (class-level MLP kill), and compound finding 8. All confirmed v1 findings. Does NOT cite CF13–CF15. **CLEAN.**

---

## 8. Verdict

### Track A — F-WVOS

**ACCEPT-WITH-AMENDMENTS**

Amendments required before execution:

- **A1** — Unify the permuted-W_V ΔNLL threshold. Section 5's GO requires permuted ≥ 5.0 nats; Section 8's kill criterion uses < 3.0 nats; the 3.0–5.0 nats zone is undefined. Standardize to: GO requires permuted ΔNLL ≥ 4.0 nats; KILL if permuted ΔNLL < 4.0 nats. This eliminates the gray zone.

- **A3** — Prior-art confirmation step: before committing to the full 76-min run, confirm A3 (arXiv:2505.12942 v3, Jun 2025) does not include Qwen3-1.7B GQA results at equivalent compression. If A3 covers Qwen3 with zero-shot OV decomposition at ΔNLL ≤ 0.5 nats at comparable rank, pivot to runner-up A-CADH. A3 uses a different loss (attention-logit error vs Frobenius) so the frame-discriminator is the loss function difference plus the GQA-specific shape; confirm this discriminator is sufficient.

- **A4** — Standardize the compression ratio headline. Section 1 says "up to 4×"; Section 2 arithmetic gives 6.3×. Use 6×–6.3× consistently in all result-facing sections.

None of these are math errors or missing controls. The controls are complete and well-specified; the calibration is absent (pure weight-based); the skeptic-controls gate is fully populated. The primary risk is A3 pre-emption, which is addressable by a pre-run literature check (5 min, not an experiment).

**Runner-up A-CADH remains on deck** if A3 is found to cover Qwen3-1.7B OV decomposition without retraining at equivalent quality. A-CADH is orthogonal (W_Q content-addressable dispatch vs W_V spectrum) and does not share the A3 risk.

---

### Track B — F-WNORM

**ACCEPT-WITH-AMENDMENTS**

Amendments required before execution:

- **A2** — Rewrite the skeptic-controls gate for the CF8-CONFIRM path. The current text simultaneously requires a ≥ 0.10 r_99/d gap between trained and random-init AND states that trained ≈ random-init ≈ 1.0 in the expected case. The two-branch structure (CF8-CONFIRM branch: gap N/A, both near 1.0 confirms null hypothesis; SURPRISE GO branch: gap ≥ 0.15 required to confirm learned concentration) must be written explicitly as two separate conditional clauses.

- **A5** — Fix the CF8-CONFIRM path runtime total. The table's "42 min" excludes the mandatory random-init control (15 min). Correct to "47 min (CF8-CONFIRM path, mandatory controls included)."

No math errors, no calibration ill-conditioning, no missing controls, no hidden prior art. The A-TROP joint measurement flag (`--run_trop`) is a structural bonus that costs only 10 additional minutes and activates the Stage 4 TDIP cascade — this is worth including unconditionally (change from "optional, recommended" to "mandatory, unless RAM OOM").

**Amendment A6 (bonus):** Promote the A-TROP joint measurement from "optional, recommended" to "mandatory unless OOM." The measurement adds 10 min, costs nothing structurally, and allows Stage 4's TDIP to be activated or killed in the same experiment. Omitting it would be wasteful given the marginal cost.

**Runner-up A-TROP (17.3) remains on deck** if any of the three current kill criteria in Section 8 are triggered. A-TROP is not pre-empted by any paper found in this pass.

---

## Summary Table

| Track | Pick | Verdict | Blocking issues | Amendments |
|---|---|---|---|---|
| A | F-WVOS | ACCEPT-WITH-AMENDMENTS | A3 (arXiv:2505.12942) OV pre-emption risk needs pre-run confirm | A1 (perm threshold), A3 (prior-art check), A4 (compression ratio) |
| B | F-WNORM | ACCEPT-WITH-AMENDMENTS | Runtime table discrepancy; controls gate self-contradiction | A2 (controls gate logic), A5 (runtime total), A6 (A-TROP mandatory) |

No CF9 frame-mismatch, no CF10 ill-conditioning, no missing skeptic controls, no fatal math errors found in either pick. Both picks should run after amendments are incorporated.
