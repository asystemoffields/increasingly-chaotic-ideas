# M-Strat v2 Controls — Verdict

**Date**: 2026-05-09
**Claim under test**: CF15 / RSIDC v2 (Structural Finding 15)
**Model**: Qwen/Qwen3-1.7B-Base, d=2048, 28 layers
**Controls run**: baseline re-derive, permutation, multi-seed (N=5), strat-prompt-dist, random-init
**Source JSON**: `opus_pipeline/round1/m_strat_v2_controls.json` (timestamp: 2026-05-09T05:26:35Z)

**Quick verdict**: Sub-claim B (channel consistency) is STRONGLY CONFIRMED as a genuine training-induced structural feature. Sub-claim A (intrinsic dim ~223) is SUBSTANTIALLY WEAKER than stated — the number is real and stable, but it is largely an architectural ceiling with only ~10% trained-induced contribution, and it collapses 4× under code-like input.

---

## Sub-claim B Verdict — Channel Consistency [1999, 1793]: STRONGLY CONFIRMED

**Original claim**: top-2 magnitude channels are consistent across layers 2-27 at indices [1999, 1793], with layer 0 differing at [784, 1371].

**Baseline replication**: Channel [1999, 1793] held at layers 4–27 in every probe (with [1999, 2033] at layer 2, a close cousin). Layer 0 was [784, 1371] as claimed. The claim replicates exactly.

**Random-init decisive contrast**:

| Condition | Layer 2 | Layer 4 | Layer 6 | Layer 8 | Layer 10 | Layer 12 | Layer 14 | Layer 16 | ... |
|---|---|---|---|---|---|---|---|---|---|
| Trained | [1999, 2033] | [1999, 1793] | [1999, 1793] | [1999, 1793] | [1999, 1793] | [1999, 1793] | [1999, 1793] | [1999, 1793] | consistent |
| Random-init | [126, 478] | [1334, 387] | [387, 372] | [382, 630] | [1121, 630] | [1121, 99] | [1610, 821] | [1507, 1258] | random/layer |

The random-init model shows **zero cross-layer consistency** — every layer has a different top-2 pair drawn from essentially uniform positions across d=2048. The trained model's persistent [1999, 1793] pair is categorically absent from the random-init model. This establishes CF15-B as a **training-induced structural feature**, not an architectural prior or estimator artifact.

**Strat-prompt-dist confirmation**: both natural_language and code_like strata show [1999, 1793] at layers 4–27. Channel identity is input-distribution-invariant; only the magnitudes differ. CF15-B is corpus-conditional only in the sense of being tied to this trained model — the specific channels are robust across prompt types.

**Verdict**: CF15-B CONFIRMED. The [1999, 1793] highway is a real learned structural feature of Qwen3-1.7B-Base, not an artifact of measurement or architecture initialization.

---

## Sub-claim A Verdict — Intrinsic Dim ~223/2048: SUBSTANTIALLY WEAKER

**Original claim**: typical-token residual stream lives in ~223/2048 dim (k_99/d ≈ 0.11) in middle layers, after stripping top-2 magnitude channels + first 4 BOS tokens.

The number ~223 is **real and reproducible** but the claim is substantially weaker for three reasons:

### Reason 1: Architecture provides ~90% of the compression; training adds ~10%

| Condition | k_99_stripped mid-layers (layers 4–27) |
|---|---|
| Trained model | 218–229 (mean 219.58 ± 0.60) |
| Random-init model | 242–245 |

Gap: **~23 dims** out of 2048. Training tightens the subspace by approximately 23/245 ≈ **9.4%** relative to random initialization. The architecture itself (Qwen3-1.7B residual stream with its specific depth/width/skip structure) forces k_99 to ~245 even with random weights. The claim that "the residual stream lives in ~223 dims" is therefore mostly a statement about the Qwen3-1.7B **architecture**, not specifically about the learned representations.

### Reason 2: Prompt-distribution dependence — 4× range across strata

| Stratum | k_99_stripped mid-layers | k_99/d |
|---|---|---|
| Mixed baseline | 218–229 | 0.107–0.112 |
| natural_language | 165–172 | 0.081–0.084 |
| code_like | 56–60 | 0.027–0.029 |

The code_like stratum gives k_99 ≈ **58**, a 4× compression relative to the baseline's ~223. Natural language alone gives ~168. The original "~223" figure is a mixed-corpus artifact — the true range is 58–229 depending on input type. Code-like inputs are dramatically more concentrated (k_99/d ≈ 0.028), and natural language is notably tighter (k_99/d ≈ 0.082) than the blended baseline.

### Reason 3: Multi-seed stability is within a stratum, not across strata

The multi-seed result (219.58 ± 0.60, max σ = 0.80 across 5 bootstrap resamples) shows excellent measurement stability — for the mixed-corpus baseline. This does not imply the 223 figure is universal; it means the measurement is precise for whatever distribution is being sampled. The ±0.60 is within-stratum reproducibility, not across-distribution robustness.

**Verdict**: CF15-A is REPRODUCED as a number but SUBSTANTIALLY WEAKER as a claim. The ~223 figure is stable (measurement artifact ruled out), architecturally constrained (not solely training-induced), and corpus-conditional (4× range across input types).

---

## Per-Control Table

| Control | What it tested | Key finding | Verdict for CF15 |
|---|---|---|---|
| **Baseline re-derive** | Direct replication of CF14→CF15 transition; k_99_stripped and top-2 channels | k_99_stripped 218–229 mid-layers; [1999, 1793] at layers 4–27 exactly as claimed | PASS — replicates |
| **Permutation** | Whether ~223 is an estimator artifact (would shuffle preserve structure?) | k_99_stripped 1735–1821 permuted (vs 218–229 real); ratio ≈ 8× | PASS — ~223 is structural, not artifact |
| **Multi-seed (N=5)** | Sampling variance across bootstrap resamples of token pool | Mean k_99_stripped 219.58 ± 0.60; max σ = 0.80 across 5 seeds | PASS — measurement stable |
| **Strat-prompt-dist** | Whether ~223 is universal across input types | NL: 165–172; code: 56–60; 4× range; channel identity invariant | WEAKENS CF15-A; confirms CF15-B |
| **Random-init** | Whether ~223 and [1999, 1793] are training-induced or architectural | k_99 ≈ 242–245 (arch ceiling); channels random per layer (no [1999, 1793] pattern) | WEAKENS CF15-A quantitatively; STRONGLY CONFIRMS CF15-B as trained feature |

---

## Refined CF15 — Proposed Restatement

**CF15-B (channel consistency) — strengthened**:

> In trained Qwen3-1.7B-Base, the top-2 variance-dominant ("massive activation") channels in the residual stream are index [1999, 1793] across layers 4–27 (layer 2: [1999, 2033]; layer 0: [784, 1371]). This cross-layer identity is a training-induced structural feature: a random-init model with identical architecture shows no channel-to-channel consistency (distinct pairs at every layer). The channel identity is robust across prompt-distribution strata (natural language, code-like).

**CF15-A (intrinsic dimension) — revised and conditional**:

> After projecting out the top-2 variance-dominant channels and the first 4 BOS tokens, the typical-token residual stream of trained Qwen3-1.7B-Base occupies approximately:
> - **58 ± 2 dims** (k_99/d ≈ 0.028) for code-like inputs (layers 4–27)
> - **168 ± 4 dims** (k_99/d ≈ 0.082) for natural-language inputs (layers 4–27)
> - **221 ± 1 dims** (k_99/d ≈ 0.108) for a mixed-corpus baseline (layers 4–27; σ = 0.60 across bootstrap seeds)
>
> Training contributes approximately 9% of the compression (from the architectural floor of ~244 dims for random-init with same architecture). The original "~223 dim" claim is valid for the specific mixed-corpus input distribution used in the original measurement, but should not be treated as an architecture-universal or task-universal intrinsic dimension.

---

## Implications for Downstream Proposals

### First-Principles GeoFingerprint — Q* basis dissolving the +19.96 lm_head catastrophe

CF15 was the load-bearing empirical support for the following chain:
1. Residual stream typical-token subspace is ~223-dim (CF15-A) → Q* basis can be computed in that subspace
2. Joint Q* construction preserves the full functional information of the residual stream
3. Projecting lm_head to r=256 in Q* basis avoids the +19.96 catastrophe that destroyed r=1024 truncation in standard SVD basis (CF12/LHQD)

**What survives the refined CF15**:

- The core GeoFingerprint mechanism — that there is a low-dimensional typical-token subspace — **survives**. The code-like stratum result (k_99 ≈ 58) actually strengthens the case: for code completion tasks, the subspace is dramatically smaller than ~223, suggesting Q*-basis projection is even more favorable for those tasks.
- CF15-B surviving means the two-channel "highway" is a real learned feature. Q* basis construction that respects the highway/subspace decomposition has a concrete structural target.

**What changes**:

- The original "~223 dim" was used to motivate r=256 as a safe truncation radius. The revised claim shows this is corpus-conditional. For natural-language tasks r=256 is marginal (168-dim typical subspace with little headroom). For code tasks r=256 is very conservative (58-dim subspace). **A single fixed r=256 is not universally safe for NL tasks in Q* basis** unless the Q* construction specifically captures the 168-dim NL subspace rather than the 58-dim code subspace.
- The ~9% trained-induced compression means Q*-basis gain over random SVD basis is smaller than the original framing suggested. The architecture itself contributes the bulk of the low-dimensionality; training sharpens it only modestly. This reduces (but does not eliminate) the claim that the Q* basis is uniquely privileged.

**Net assessment**: First-Principles GeoFingerprint **survives in conditional form**. The FP E4 binary settling experiment (does r=256 lm_head in Q* basis dissolve +19.96?) remains scientifically motivated. However, the experiment must be designed with input-stratification: test separately on NL and code inputs, and verify the Q* basis was constructed from the appropriate stratum. The "dissolves the catastrophe" claim is most likely to hold for code-like inputs (58-dim subspace ⊂ r=256 sphere comfortably); it is more uncertain for natural-language inputs.

---

## Open Questions / Follow-up Controls

1. **Scale generalization (CF15-B)**: Does [1999, 1793] channel consistency generalize to Qwen3-0.6B and Qwen3-8B? The 0.6B has d=1024 so indices are incomparable, but the cross-scale control (currently flagged `cross_scale=False`) could test whether the *fraction* of width occupied by the massive-activation channels is conserved. If the massive-activation channel position is architecture-version-specific rather than scale-invariant, the highway framing is family-level, not class-level.

2. **Stratum interaction with layer depth**: The strat-prompt-dist control was run at all layers but the code_like collapse (k_99 ≈ 58) — does it persist deep into layers 24–27, or do late layers re-expand the code subspace? The per-layer code data shows 56–60 flat across all probed layers, suggesting no late-layer re-expansion, but a finer probe (every layer, not every 2) would confirm.

3. **Q* basis orthogonality to the highway**: Are the [1999, 1793] highway channels inside or outside the typical-token operating subspace (as measured by the ~223-dim k_99 ball)? If outside, Q* construction that strips the highway before PCA is well-founded. If inside, stripping changes Q*. This is the most critical unresolved question for the FP GeoFingerprint pathway.

4. **What drives code_like compression?** k_99 ≈ 58 for code is a 4× drop. Is this syntactic regularity (code tokens have less positional diversity in the residual stream), tokenizer effects (code tokens are over-represented in training so their representations are more collapsed), or something structural? A control using code-like prompts on a non-code-specialized base model would disambiguate.

5. **Random-init layer 0 vs layers 2–27 discontinuity**: k_99_stripped jumps from 178 (layer 0) to 242–245 (layers 2–27) in the random-init model. This mirrors the trained model's layer-0 anomaly (173 vs 218–229). The layer-0 discontinuity appears to be architecturally prior (embedding layer has different variance structure), not a training phenomenon. Worth characterizing explicitly for the highway/bus/bulk decomposition.

6. **Untied lm_head (Qwen3-8B+)**: CF12 catastrophe was measured in the tied-embedding configuration of Qwen3-1.7B-Base. The GeoFingerprint hypothesis depends on whether the +19.96 catastrophe also appears in untied configurations and whether the ~58–223 dim subspace is preserved there.
