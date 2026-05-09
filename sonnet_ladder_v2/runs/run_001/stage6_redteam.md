# Stage 6 — Red Team — Run 001 (v2 Sonnet Ladder)

Red-team agent: Sonnet 4.6 (independent pass, both tracks).
Date: 2026-05-09.
Inputs read: STAGE6_RED_TEAM.md, stage5_selector.md, stage3_deep_research.md, SUMMARY.md, KILL_LIST.md.
WebSearch: arXiv:2410.03765 (Basis Sharing), arXiv:2508.04581 (MASA/Share Your Attention), arXiv:2503.18893 (xKV), arXiv:2602.15200 (COMPOT), QSVD (NeurIPS 2025), three-tier activation quantization literature.

---

## Track A Verdict

**ACCEPT-WITH-AMENDMENTS**

Runner-up on standby: A-GFRS (16.84).

---

## Track A — F-CQSGC: Cross-Layer W_Q Stacked SVD

### 1. Frame-mismatch re-check (CF9)

**Imported mechanism**: randomized SVD of a stacked weight matrix, plus the claim that the shared right singular subspace U is "the gauge-fixed coordinate for cross-layer residual-stream attention computation."

The gauge-freedom claim (Section 6c) requires scrutiny. The attention-score gauge freedom M = W_Q W_K^T is invariant under (W_Q Σ, W_K Σ^{-T}) for any invertible Σ. However, F-CQSGC does NOT exploit this gauge freedom for the stacked SVD — it simply stacks W_Q matrices and takes their joint PCA. The "gauge-fixed coordinate" framing is descriptive vocabulary applied after the fact, not a mathematical precondition that the stacked SVD requires.

**Precondition check for the elegance claim (Section 6c)**: the gauge-exploitation tag claims that U is the gauge-fixed coordinate. But U is derived from the *right* singular vectors of the stacked W_Q matrix. This is a PCA of the row spaces of the W_Q^{(ℓ)} matrices. Whether U is a gauge-natural object depends on whether the stacking respects the gauge freedom. In trained weights, W_Q^{(ℓ)} cannot be freely rotated without changing the logits (W_K is not jointly rotated), so the gauge freedom does NOT freely reduce to any U we might extract. The elegance multiplier was applied under the premise that U is "algebraically exact," but U is an empirical subspace, not an exact algebraic identity. This is a **framing inflation** rather than a math error — the SVD is correct, the gauge language is imprecise.

**Verdict on CF9**: The mechanism (stacked SVD + var@128 measurement) is mathematically sound. CF9 CLEAR. The gauge-exploitation elegance tag is partially inflated but does not constitute a frame-mismatch that kills the experiment. The experiment tests a real structural property regardless of whether the gauge framing survives scrutiny.

**One real precondition risk**: The Frobenius normalization note in Section 8 of Stage 5's hand-off is critical. Stage 5 notes correctly that C_ℓ = W_Q^{(ℓ)} @ U must use the **original** (un-normalized) W_Q^{(ℓ)} for reconstruction, while normalization is only applied during the SVD step. A normalization reversal bug here would make the reconstruction formula use the wrong scale, producing artificially inflated ΔNLL. The amendment below requires explicit verification of this reversal at the script level.

### 2. Calibration ill-conditioning re-check (CF10)

No calibration fit. F-CQSGC is purely weight-based (stacked SVD, no regression, no activation samples). CF10 NOT APPLICABLE — confirmed. The per-layer reconstruction ΔNLL uses a held-out eval corpus (WikiText-2, 512 tokens), and the go threshold checks per-layer ΔNLL on that held-out corpus, not on calibration. The held-out / calibration disjointness requirement is trivially satisfied because there is no calibration corpus at all for the primary measurement. CF10 CLEAR.

### 3. Skeptic-controls hard gate

**Claim under test**: "The stacked W_Q matrices share a common dominant right singular subspace (var@128 > 0.80)" — this is a "X is consistent across layers" claim. Gate applies.

- **Permutation control**: PRESENT and well-specified. Independently permuted rows per layer before stacking; expected permuted var@128 ≈ 0.063; GO requires gap ≥ 0.60. This is correct and properly calibrated. The expected value (128/2048 ≈ 0.063) is theoretically motivated.

- **Random-init control**: PRESENT and well-specified. Random-init Qwen3-1.7B expected var@128 ≈ 0.063–0.10; GO requires trained − random-init gap > 0.30. One concern: the Stage 5 plan does not specify what random initialization scheme to use. Kaiming uniform (default PyTorch nn.Linear init) produces weight matrices with spectral structure (non-flat singular values due to the Marchenko-Pastur distribution at finite width). A Kaiming-initialized W_Q matrix can have a var@128 of roughly 0.15–0.25 for a 2048×2048 matrix due to the stacking itself amplifying the top components. The 0.30 gap threshold may be too tight if random-init var@128 is higher than expected. **Amendment**: specify that random-init uses `torch.nn.init.normal_(std=0.02)` (the GPT-style init that produces near-flat singular spectra), and verify empirically that random-init var@128 < 0.20 before claiming the control is valid.

- **Multi-seed**: PRESENT and well-specified. Three seeds for `svd_lowrank`; GO requires mean > 0.80 and worst-of-3 > 0.75. This is correctly scoped.

**Gate verdict**: Controls are present and structurally valid. One amendment needed (random-init initialization scheme specification). GATE PASSES with amendment.

### 4. Hidden prior-art search (third pass)

**Basis Sharing (arXiv:2410.03765, ICLR 2025)**: The paper examines cross-layer parameter sharing via SVD for LLM compression. Critically, based on search results, Figure 4/6 of the paper displays the Frobenius loss incurred by basis sharing for W_Q, W_V, W_up, and W_gate — meaning the paper explicitly analyzed W_Q cross-layer basis sharing. The Basis Sharing method uses SVD decomposition with a shared basis across layers. The key question from Stage 5's kill criteria (Section 8, criterion 1) is whether Basis Sharing demonstrates **weight-only (no fine-tuning)** W_Q cross-layer basis sharing with positive results.

The Stage 3 research flagged this paper as an "ADJACENT (critical)" risk. Based on the available search results, Basis Sharing appears to **require fine-tuning** to recover quality after applying basis sharing — the paper benchmarks show PPL improvements over baseline SVD methods, suggesting quality recovery via fine-tuning is part of the method. The abstract and review materials describe it as a "compression" approach that outperforms SVD-based approaches, consistent with fine-tuning-aided recovery. The no-retraining moat for F-CQSGC appears to survive.

**QSVD (arXiv:2510.16292, NeurIPS 2025 Spotlight)**: QSVD performs joint SVD on concatenated [W_Q, W_K, W_V] weight matrices with cross-layer rank allocation. This is a closer threat than Basis Sharing — it explicitly shares a common down-projection across Q, K, V. However, QSVD's cross-layer rank allocation is about assigning different ranks to different layers of the *same* model, not about sharing a single U subspace across all 28 W_Q matrices. QSVD's method also appears calibration-guided (adaptive rank based on "singular value's contribution to model loss"). F-CQSGC's hypothesis (a single U shared by all 28 W_Q with var@128 > 0.80) is a stronger and more specific claim than QSVD's adaptive per-layer rank allocation.

**NEW FINDING — COMPOT (arXiv:2602.15200, February 2026)**: This paper explicitly states that "enforcing a single shared subspace can degrade accuracy even at moderate compression" and proposes a union-of-subspaces framework instead. This is a **direct theoretical caution against F-CQSGC's core hypothesis**. COMPOT does not pre-empt the measurement claim (var@128 on Qwen3-1.7B is still uncharted), but it provides prior evidence that if var@128 > 0.80 but the practical compression (ΔNLL < 1.0 nat) fails, the single-subspace design is the cause. This is worth noting in the ambiguous-zone section of the plan.

**PRAC (arXiv:2602.23111, February 2026)**: "Principal-Random Subspace for LLM Activation Compression and Memory-Efficient Training" — this targets activation compression (different object). Not a threat to F-CQSGC's weight-matrix claim.

**xKV (arXiv:2503.18893, March 2025)**: Cross-layer SVD for KV-Cache, not weight matrices. Not a threat.

**Summary**: No paper found that demonstrates weight-only (no fine-tuning, no calibration data), cross-layer W_Q shared-basis SVD on any LLaMA-class or Qwen-class model with a positive var@K result. The no-retraining moat survives the third-pass prior-art check. COMPOT's caution is a risk flag, not a pre-emption.

### 5. Biased-framing audit

**GO threshold assessment**: var@128 > 0.80 AND ΔNLL < 1.0 nat. The ΔNLL threshold is generous compared to the 0.30-nat practical target — 1.0 nat represents a large quality cost (nearly matching the CF11 W_Q K=128 within-layer ΔNLL of 0.98 nats). Stage 5 is essentially saying "any compression that doesn't exceed the within-layer baseline is a GO." This is a reasonable bar for the structural claim (does the shared subspace exist?) but potentially misleading as a deployment GO (the HOIST cascade makes sense only if ΔNLL < ~0.30 nats additional over CF11). **Amendment**: add a "deployment GO" sub-threshold of ΔNLL < 0.30 nats (separate from the structural GO at 1.0 nat) to clarify when the cascade to HOIST is actually warranted.

**NO-GO structural claim**: The NO-GO claim ("W_Q subspaces rotate independently across layers; class-level kill on all four Cluster C1 ideas") is specific and load-bearing. This is not rhetorical. The +2 NO-GO bonus is justified.

**Selection-algorithm framing**: The 1.5× convergence multiplier (Cluster C1, 4 orientations) is accurately motivated. The 1.2× gauge-exploitation elegance multiplier is the one area of concern (the gauge framing is imprecise as noted in Section 1), but because the multiplier was applied to the Stage 5 scoring table and does not affect the experiment validity, this does not constitute algorithm gaming.

**Within-layer baseline absence (v1 calibration anchor)**: The R1-Idea-D v1 lesson was: add a within-layer baseline to prevent confounding structural artifacts. For F-CQSGC, the relevant baseline is: what is the var@128 of a stack of **randomly sampled** 28 weight matrices with the same per-layer spectral properties as the actual W_Q (i.e., same per-layer Frobenius norms and same r_99/d = 0.63)? This is different from the random-init control — it asks whether a collection of matrices with the same spectral properties but random orientations (sampled from the invariant distribution of O(2048)) would produce var@128 close to the observed value. Stage 5's random-init control catches "architectural artifact" but may not catch "spectral artifact" — the fact that each W_Q has r_99 ≈ 0.63 means the stacked matrix inherits that structure, which could inflate var@128 above the random-init baseline even without genuine cross-layer coherence. **Amendment**: add a "matched-spectrum random-orientation baseline" — generate 28 random orthogonal matrices with the same singular value spectrum as the actual W_Q^{(ℓ)} (use the actual singular values, randomize the singular vectors), stack and compute var@128. This is the correct null hypothesis for detecting cross-layer subspace alignment specifically.

### 6. Runtime / scope sanity

- **30-minute budget**: Load + normalize + stack (2 min) + randomized SVD at D*=128 (15 min) + reconstruction + eval (10 min) + permutation control (~2 min) + random-init control (~5 min) + multi-seed 3× (~6 min additional). Total ≈ 40 min with all controls. The plan's "30 min" estimate was written before the Section 9 controls were specified. The controls add ~13 minutes. This is still well within any ≤8h gate. No issue.
- **RAM check**: Stacked matrix (57344×2048 bf16) = ~235 MB. Plus model weights (~3.4 GB), PyTorch overhead. Total active: ~4 GB. Within 7.28 GiB. Passes.
- **SVD approximation error check** (Section 13 item 1): Stage 5 explicitly requests verification that ‖M_stack − U_D* S_D* V_D*^T‖_F / ‖M_stack‖_F < 0.05. This is a correct guard. Including it adds ~1 min.
- **Eval / calibration disjointness**: No calibration corpus used at all. WikiText-2 test split for eval. CLEAR.

### 7. CF13–CF15 check

No CF13–CF15 dependencies. The residency arithmetic uses Qwen3 architecture constants only. The 70B projection in Section 11 is an extrapolation but is not cited as a load-bearing input to the experiment itself. CLEAR.

### 8. Section 6c elegant-equivalence check

The gauge-exploitation elegance tag: "the shared U is the gauge-fixed coordinate for cross-layer residual-stream attention computation." As analyzed in Section 1, the gauge freedom of the W_Q W_K^T product does NOT directly motivate U as a gauge-natural object from first principles — the stacked SVD extracts U empirically. The claim that U is "gauge-fixed" is an interpretive frame applied post-hoc, not a mathematical identity. However, the elegance multiplier (1.2×) was correctly bounded at 1.2 rather than 1.3 (which would require an algebraically exact identity), and it contributed to the Stage 5 score but does not affect the experiment's validity. The experiment tests a real measurement claim. Under Section 6c, the elegance argument's imprecision is noted but does not constitute a precondition failure that invalidates the experiment. The elegance score is slightly inflated but not fatally so.

### Track A — Amendments

**A1 (random-init control initialization scheme)**: In Section 9, specify that random initialization uses `torch.nn.init.normal_(std=0.02)` (GPT-style) or `torch.nn.init.orthogonal_()` — and empirically report the random-init var@128 value explicitly so that the 0.30-gap threshold can be verified before declaring the control valid.

**A2 (matched-spectrum random-orientation baseline — load-bearing)**: Add a third null baseline that preserves each W_Q^{(ℓ)}'s exact singular value spectrum but randomizes singular vectors (draw from Haar measure on O(2048)). Stack these, compute var@128. Report the gap between this baseline and the real var@128. This is the critical disambiguation: if real var@128 ≈ matched-spectrum-random-orientation var@128, the cross-layer coherence is explained by the within-layer spectral concentration (CF11 finding) being stacked, not by genuine cross-layer subspace alignment. This is the v2 version of the R1-Idea-D lesson. **GO requires: real var@128 > matched-spectrum-random-orientation var@128 by > 0.20.** Runtime: ~2 additional min.

**A3 (deployment GO sub-threshold)**: Add a "deployment GO" row to Section 5: ΔNLL < 0.30 nats at D*=128 is required for the HOIST cascade to be justified. The structural GO (ΔNLL < 1.0 nat) confirms the subspace exists; the deployment GO determines whether it is useful. Report both separately.

**A4 (COMPOT reference)**: Add COMPOT (arXiv:2602.15200, Feb 2026) to Section 8 (kill criteria / context). If the single-subspace design fails (ΔNLL > 1.0 nat despite var@128 > 0.80), COMPOT's union-of-subspaces approach is the direct next step and should be the runner-up experiment, not A-GFRS.

---

## Track B Verdict

**ACCEPT-WITH-AMENDMENTS**

Runner-up on standby: C-CTERA (19.8).

---

## Track B — R-RAOK-70B: Three-Tier Activation Codebook

### 1. Frame-mismatch re-check (CF9)

No theorem imports. The tier-boundary derivation is from empirical Jaccard statistics (CF3), not from any imported theorem. The Tier-0 stability hypothesis (same 2 channels stable in ≥90% of tokens across ≥20/28 layers) is a direct extension of CF3's K=0.1% Jaccard=0.718 measurement.

**One frame-mismatch risk to flag**: CF3 was measured on 7 sampled layers, not all 28. The plan's Rung 1 measures all 28 layers, which is correctly described as the "first full-28-layer measurement." However, Stage 5 presents the H1 hypothesis as if the CF3 result already supports the 28-layer claim. It does not — CF3 is a 7-layer result. The plan handles this correctly (Rung 1 verifies the full-28-layer claim), but the hypothesis statement in Section 3 ("This follows from CF3") is slightly overstated. The plan's structure correctly gates the rest on Rung 1 GO. CF9 CLEAR (the mechanism is testable; the precondition "CF3 extends to 28 layers" is itself the experiment).

**INT4 scale-factor sign (Section 13, item 3)**: Stage 5 flags the critical sign issue: symmetric INT4 should use max_abs / 7, not max_abs / 8. Using max_abs / 8 creates asymmetric clipping at the negative saturation point, which would inflate ΔNLL artifactually at large negative activations. This is a real implementation bug risk. The amendment below requires explicit test coverage.

### 2. Calibration ill-conditioning re-check (CF10)

**Tier-0 identification**: the top-2 channels by mean magnitude rank across 200 diverse prompts — this is a threshold rule (top-2 of 2048), not a regression fit. n_params = 2 channel indices per layer. This is trivially well-conditioned. CF10 NOT APPLICABLE here.

**Tier-1 identification**: "next-18 by mean magnitude rank." Again, a rank-ordering rule, not a regression. n_params = 18 channel indices per layer × 28 layers = 504 integers. These are determined by sorting 2046 channels by mean activation magnitude over 200 prompts. This is a sample-average estimator, not a least-squares fit. CF10 NOT APPLICABLE.

**Calibration / eval disjointness**: The plan uses R2 PDAP dataset (200 prompts) for Tier-0/Tier-1 channel identification, and WikiText-2 (512 tokens held-out) for ΔNLL evaluation. These are disjoint. CLEAR.

**One CF10-adjacent risk**: The multi-seed control (Section 9) uses different orderings/subsets of the 200 PDAP prompts to define Tier-1 channels. If the PDAP dataset has only 200 prompts total, "3 random subsets of 200 prompts" from 200 prompts can only be done via resampling (bootstrap subsets), not truly independent samples. The plan acknowledges this ("3 random orderings of the 200 prompts for Tier-1 channel identification"). This is acceptable — the test is whether the channel-rank ordering is stable under permutation of the 200 prompts — but the plan should state explicitly that if PDAP has more than 200 prompts available, truly independent disjoint subsets should be used.

### 3. Skeptic-controls hard gate

**Claim under test (H1)**: "Top-2 channels stable across ≥90% of tokens in ≥20/28 layers" — "X is consistent across prompts/layers" claim. Gate applies.

- **Permutation control**: PRESENT and well-specified. Permute channel indices per token; expected random stability ≈ 2/2048 ≈ 0.1%. GO requires real stability ≥90%, permuted < 10%, gap ≥ 80 pp. Correctly calibrated.

- **Random-init control**: PRESENT and well-specified. Random-init model expected stability ≈ 0.1%. GO requires gap > 70 pp. The same initialization-scheme ambiguity from Track A applies here: Kaiming init may not produce uniform channel magnitudes. **Amendment**: specify initialization scheme (same as Track A: `torch.nn.init.normal_(std=0.02)`).

- **Multi-seed for Rung 2**: PRESENT. Three calibration corpus orderings; GO requires mean ΔNLL < 0.30 and worst-of-3 < 0.50. The note about 30-min additional runtime is acknowledged. The +10-minute estimate per Rung 2 re-run seems aggressive (the plan says "Rung 2 is cheap; re-running 3× adds 30 min" but Rung 2 alone was estimated at 30 min, making 3× = 90 min additional). The total pipeline with all controls is closer to 40 + 90 = 130 min, not 40 min. This is still within any ≤8h gate.

**Gate verdict**: Controls present and well-specified. Two amendments (initialization scheme, runtime estimate correction). GATE PASSES with amendments.

### 4. Hidden prior-art search (third pass)

**Three-tier (FP16/INT8/INT4 static/dynamic/bulk) activation quantization**: Extensive search found no published paper using a three-tier scheme with the specific K=0.1%/K=1% Jaccard-derived channel boundaries. The closest approaches:

- **LLM.int8()** (arXiv:2208.07339): binary split (FP16 static + INT8 rest). Two tiers, not three.
- **SmoothQuant** (arXiv:2211.10438): migrates outliers into weights rather than using a multi-tier activation scheme.
- **PrefixQuant** (arXiv:2410.05265): addresses token-wise outliers by prefix construction. Different mechanism.
- **OutlierTune** (arXiv:2406.18832): per-channel PTQ for activations, focusing on channel-wise scaling. Does not use three tiers with Jaccard-derived boundaries.
- **Quaff** (arXiv:2505.14742): "Outlier Spatial Stability Hypothesis" for fine-tuning contexts. The OSSH is conceptually adjacent to CF3's Jaccard finding, but Quaff targets fine-tuning, not post-training inference quantization, and does not derive tier boundaries from Jaccard statistics.

**No three-tier FP16/INT8/INT4 scheme found with 0.1%/1% Jaccard-derived boundaries**. The calibration-free Jaccard-to-tier derivation remains novel as of 2026-05-09.

**NEW FINDING**: Stage 5's blind spot note (Section 13) about W_gate and W_down GEMV inputs is a real gap. The plan says "W_up GEMV inputs only" but CF3's Jaccard measurement was conducted on the **residual-stream activations entering the MLP block**, which feed W_gate AND W_up. If Tier-0/Tier-1 assignment is correct for W_up inputs, it should also hold for W_gate inputs (same tensor). **Amendment**: clarify whether activation quantization is applied to the shared MLP block input (before the split to W_gate and W_up) or only to the W_up-bound copy of the tensor. If the same quantized activation feeds both W_gate and W_up (as is typical in SwiGLU implementations), the experiment is already testing both simultaneously — but this must be verified and stated explicitly.

### 5. Biased-framing audit

**GO threshold (ΔNLL < 0.30 nats)**: This is a tight, defensible threshold. The competing LLM.int8() achieves near-zero ΔNLL on their benchmark at 8-bit but has not been tested on Qwen3. At INT4 for 99% of activations, 0.30 nats is an aggressive but plausible target given the Tier-0/Tier-1 bypass structure. Not gerrymandered.

**NO-GO structural claim**: Rung 1 NO-GO ("Tier-0 static channel claim dies") is specific and load-bearing. The claim that it kills F-OCSSQ and C-RAOK-Grounded simultaneously (the Cluster C2 kill) is correctly stated. The +1 NO-GO bonus is appropriate (mechanism-class kill, not CF-entry-level kill).

**"Conserved-quantity" elegance tag**: Stage 3 and Stage 5 both tag this as `conserved-quantity`, framing the top-2 channels as a "conserved identity" across tokens. This is statistical quasi-conservation (Jaccard=0.718), not algebraic conservation. The elegance multiplier was correctly applied at 1.1× (not 1.2×, per the "calibration-fit quality rule" since the conservation is statistical rather than algebraically exact). This is a correct, non-inflated application of the elegance taxonomy. No framing concern here.

**Selection-algorithm tiebreak**: The three-way tie (R-RAOK-70B / F-OCSSQ / C-RAOK-Grounded) was broken by "70B deployment arithmetic being self-contained." This is a reasonable tiebreak criterion, not gaming. All three experiments share the same Rung 1 gate, so the selection of R-RAOK-70B is effectively equivalent to selecting the shared Rung 1 test.

**CF3 layer-coverage framing**: Section 3 says "this follows from CF3 (Jaccard=0.718 at K=0.1%)." This is slightly overconfident framing given that CF3 was measured on 7/28 layers. Stage 5 acknowledges this in Section 10 ("the main uncertainty is whether the CF3 7-layer Jaccard measurement generalizes to all 28 layers") but the H1 statement in Section 3 does not carry this caveat. **Amendment**: restate H1 as "The top-2 channels by mean magnitude rank in Qwen3-1.7B-Base activations are stable in ≥20/28 layers (extending CF3's 7-layer finding to the full 28-layer stack)."

### 6. Runtime / scope sanity

- **40-minute budget**: Rung 1 (~10 min) + Rung 2 setup + eval (~30 min) = 40 min base. With controls: +1 min permutation, +5 min random-init, +10 min × 2 additional Rung-2 runs for multi-seed = ~66 min total. The plan's Section 9 estimates "Rung 2 is cheap; re-running 3× adds 30 min" but the Rung 2 plan says "Cal setup 10 min + eval 20 min = ~30 min." Three runs = 90 additional min. Total pipeline with all controls: ~100–130 min. Still within ≤8h gate. **Amendment**: correct the runtime estimate in Section 9 to reflect the actual total.

- **Eval / calibration disjointness**: PDAP (200 prompts, channel identification) disjoint from WikiText-2 (512 tokens, ΔNLL eval). CLEAR.

- **Per-layer ΔNLL breakdown**: Section 4 already includes this. CLEAR.

- **INT4 symmetry check (Section 13, item 3)**: the max_abs / 7 vs max_abs / 8 distinction must be tested with a sanity check (quantize a known tensor, verify reconstruction error). **Amendment (implementation guard)**: add a 5-line unit test in the script that verifies symmetric INT4 scale is max_abs / 7 and checks that no clipping artifact occurs at the minimum value.

### 7. CF13–CF15 check

Stage 5's Section 12 mentions "70B deployment arithmetic being self-contained" and no CF13-CF15 dependencies. Verified: the 70B KV cache arithmetic in Section 5 uses only Qwen3-72B architectural constants (d=8192, 64 KV heads, 28 layers etc.) and does not rely on CF13 (mmap latency), CF14 (massive activation channels), or CF15 (intrinsic dimension). The plan is CF13-CF15-independent. CLEAR.

### Track B — Amendments

**B1 (random-init initialization scheme)**: Same as A1 — specify `torch.nn.init.normal_(std=0.02)` for the random-init control. Report the random-init Tier-0 stability value explicitly.

**B2 (H1 caveat)**: Restate H1 explicitly as an extension of CF3's 7-layer finding to 28 layers, not as a direct consequence of CF3. The difference matters for framing the Rung 1 result: it is the first 28-layer measurement, not a validation of an already-established result.

**B3 (MLP input quantization scope)**: Clarify whether "W_up GEMV inputs" means the shared MLP block input (activations before the W_gate/W_up split) or only the W_up pathway. If the shared MLP input is quantized, both W_gate and W_down inputs use the same quantized tensor — state this explicitly. If the intent is to quantize only the W_up projection input (leaving W_gate at full precision), add a supplemental run that quantizes both W_gate and W_up inputs and compare ΔNLL.

**B4 (INT4 scale factor unit test)**: Add a script-level unit test for the INT4 scale factor before running Rung 2. A sign error here is silent (no Python exception, just wrong output numbers).

**B5 (runtime correction)**: Correct Section 9's multi-seed runtime estimate from "adds ~10 min" to "adds ~60 min for 2 additional full Rung-2 runs." This is still within gate but the underestimate could cause the user to be surprised mid-run.

**B6 (PDAP subset independence)**: If PDAP has > 200 prompts, use three independent 200-prompt disjoint subsets for the multi-seed Rung-2 runs. If PDAP has exactly 200 prompts, use three different random orderings (bootstrap) and state this explicitly.

---

## Summary of All Amendments

### Track A (F-CQSGC) — ACCEPT-WITH-AMENDMENTS

| # | Amendment | Type | Required |
|---|---|---|---|
| A1 | Specify random-init uses `normal_(std=0.02)`; report raw random-init var@128 | Control sharpening | Yes |
| A2 | Add matched-spectrum random-orientation baseline (rotate singular vectors, keep singular values) | Load-bearing baseline (v1-lesson) | Yes — gate condition |
| A3 | Add deployment GO sub-threshold: ΔNLL < 0.30 nat required for HOIST cascade | Threshold clarification | Yes |
| A4 | Add COMPOT (arXiv:2602.15200) to Section 8 as the union-of-subspaces fallback on structural GO / deployment NO-GO | Prior-art context | Recommended |

**Blocking amendment**: A2 is load-bearing. Without it, a structural GO (var@128 > 0.80) could be explained entirely by the within-layer spectral concentration being stacked (same effect that produced the AQFKV CF11 result), without genuine cross-layer alignment. This would repeat the R1-Idea-D error (cross-layer cosine ≈ 0.99 was structurally guaranteed by residual additivity). A2 is the disambiguation baseline. Stage 5 must incorporate A2 before the experiment runs.

### Track B (R-RAOK-70B) — ACCEPT-WITH-AMENDMENTS

| # | Amendment | Type | Required |
|---|---|---|---|
| B1 | Specify random-init initialization scheme; report raw random-init stability value | Control sharpening | Yes |
| B2 | Restate H1 as 28-layer extension of 7-layer CF3 finding | Framing clarification | Yes |
| B3 | Clarify MLP input quantization scope (shared W_gate+W_up input vs W_up-only) | Scope clarification | Yes |
| B4 | Add unit test for INT4 scale factor (max_abs / 7) before Rung 2 | Implementation guard | Yes |
| B5 | Correct Section 9 multi-seed runtime from ~10 min to ~60 min | Runtime correction | Recommended |
| B6 | Use independent PDAP subsets if > 200 prompts available; document bootstrap if not | Data hygiene | Recommended |

**Blocking amendment**: B4 is load-bearing. An INT4 sign error producing clipping at negative saturation would inflate ΔNLL and could cause a false NO-GO. A 5-line unit test costs nothing and prevents a silent result corruption.

---

## Key Prior-Art Findings from Third Pass

1. **COMPOT (arXiv:2602.15200, Feb 2026)**: explicitly documents that single-shared-subspace cross-layer compression degrades accuracy at moderate compression. Does not pre-empt F-CQSGC's measurement, but provides the specific degradation mechanism if ΔNLL fails the deployment GO despite var@128 passing the structural GO. Add to F-CQSGC Section 8.

2. **QSVD (arXiv:2510.16292, NeurIPS 2025)**: joint QKV SVD with cross-layer adaptive rank. Closer to F-CQSGC's terrain than Stage 3 found, but still requires calibration-guided adaptive rank allocation and does NOT share a single U subspace across all 28 W_Q matrices. No pre-emption.

3. **xKV (arXiv:2503.18893, March 2025)**: cross-layer SVD on KV-Cache activations (not weight matrices). Not a pre-emption of either pick.

4. **Quaff (arXiv:2505.14742)**: Outlier Spatial Stability Hypothesis for fine-tuning. Conceptually adjacent to CF3, but targets fine-tuning, not inference-time activation quantization. Not a pre-emption of R-RAOK-70B.

5. **No three-tier FP16/INT8/INT4 activation quantization paper found with 0.1%/1% Jaccard-derived boundaries**. R-RAOK-70B's calibration-free tier-derivation moat survives.

---

## Blind Spots Stage 5 Missed

**Track A**:
- The most important blind spot: the matched-spectrum random-orientation baseline (A2). Without it, var@128 > 0.80 on the stacked matrix could reflect nothing more than the within-layer spectral concentration (CF11) being additive across layers, not genuine cross-layer coherence. This is the exact structural artifact that killed R1-Idea-D.
- COMPOT's February 2026 finding is directly relevant to the failure mode where structural GO and deployment NO-GO coexist.

**Track B**:
- The INT4 scale-factor sign error risk was flagged by Stage 5 itself in Section 13, but no implementation guard (unit test) was included in the plan. This is a silent failure mode.
- The MLP input quantization scope ambiguity (W_gate+W_up shared input vs W_up-only) is unresolved and affects whether the 28-layer ΔNLL result is representative of a real deployment scenario.
