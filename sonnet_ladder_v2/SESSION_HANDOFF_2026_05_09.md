# Session Handoff — 2026-05-09

This document is the canonical handoff for the v2 Sonnet ladder + RAOK adversarial pursuit done in this session. Read this first if resuming after context loss.

## One-line summary

The v2 ladder ran 5 rounds end-to-end + 4 partial; the standout selection (RAOK three-tier activation quantizer) is currently being pressure-tested for *deployability*, not novelty — the user wants a binary deployable / not-deployable verdict on whether the assembly (RAOK + W4 weights + KV compression) enables an order-of-magnitude jump in runnable-model-size on a 7.28 GiB CPU-only laptop. Three tests still in flight. **No more open-ended "we need more tests" framing — commit to a verdict on whatever lands.**

---

## Where we are RIGHT NOW (in-flight)

### Three tests running simultaneously

1. **Sharp ablations v2** — agent ID `aa6c88714bb1e5bd1`. Calibration-derived per-channel INT4 (fixes the bug in v1 sharp ablations that gave bit-exact ΔNLL = 0.000). Expected output: `sonnet_ladder_v2/cheap_tests/run_001/rraok_sharp_ablations_v2_result.md` and `experiments/stage0/ladder_v2/round1_raok70b/sharp_ablations_v2_results.json`.

   **Resolves**: does RAOK reduce to standard per-channel INT4 (then it's published-art assembly) or does the tier structure provide non-trivial work beyond it (then it's novel engineering)?

2. **Cross-scale Qwen3-4B** — agent ID `a6896f3a5a6ccb22e`. Tries to apply the three-tier scheme on Qwen3-4B-Base. May OOM on 7.28 GiB; agent has fallback to smaller model. Expected output: `sonnet_ladder_v2/cheap_tests/run_001/rraok_cross_scale_4b_result.md` + `cross_scale_4b_results.json`.

   **Resolves**: does RAOK generalize beyond Qwen3-1.7B's specifics, or is it overfit to one model's idiosyncrasies?

3. **W4A4 composition on Qwen3-1.7B** — agent ID `accffd6f05dd0bbf3`. Per-channel-INT4 weights × RAOK three-tier activations. Three forward passes per seed (baseline / W4-only / W4A4). Expected output: `sonnet_ladder_v2/cheap_tests/run_001/rraok_w4a4_composition_result.md` + `w4a4_composition_results.json`.

   **Resolves**: do RAOK activations and INT4 weights compose without destructive interference? (The bridge between activation work and the deployment story.)

### Verdict criteria — what to commit to when these land

**DEPLOYABLE** — if all three pass: the assembly enables a category jump (Qwen3-14B+, possibly 32B-class) at usable quality. Path: assemble published components plus RAOK-class activation discipline.

**DEPLOYABLE-AS-PUBLISHED-ASSEMBLY** — if v2 sharp ablations show RAOK collapses to per-channel INT4: components are off-the-shelf today (per-channel INT4 + W4 weight quant + TurboQuant for KV + Q4_K_M-class weight kernels via llama.cpp/ik_llama.cpp). The win is that the assembly works, not that any single piece is novel. Order-of-magnitude jump still likely possible.

**NOT-DEPLOYABLE-IN-CLAIMED-FORM** — if cross-scale fails OR W4A4 destructively interferes: the deployment claim breaks. State which test failed and what's salvageable.

**No "we'd need more tests" exit.** The user explicitly demands a binary call.

---

## Critical context for the deployability question

### TurboQuant pre-empts the KV-cache branch (NEW finding this session)

arXiv:2504.19874 (Google DeepMind, ICLR 2026): **3-bit keys / 2-bit values, ~6× KV cache compression, statistically lossless via random rotation + Lloyd-Max scalar quantizer + 1-bit QJL bias correction.** Better than RAOK on KV. So in the deployability assembly:

- **Activations (MLP block input)**: RAOK or per-channel INT4 (TBD by sharp ablations v2)
- **KV cache**: **TurboQuant** (published, deployable today)
- **Weights**: Q4_K_M / Q4_0 via llama.cpp (published, deployable today)
- **Storage layout**: GGUF + mmap (deployed today; CF13 NEXTENT measurements not load-bearing)

The deployment story for "order-of-magnitude jump" doesn't require RAOK to be novel — it requires the assembly to work. RAOK's contribution is the activation-side discipline.

### User framing (load-bearing)

The user's explicit goals (per memory `feedback_order_of_magnitude_not_specific_target.md`):

> What I want is an order of magnitude change in what can be run locally — not necessarily a specific number.
> If I can suddenly run a 45B on this machine, that's an unbelievable, world-changing feat.

So "70B-on-7.28GiB" was illustrative. The bar is "category jump in runnable model class." Currently the laptop runs Qwen3-4B IQ4_XS at ~6 tok/s. Targets that count: Qwen3-14B usable, Qwen3-32B usable, anything in 30-50B range.

### User has no Anthropic API key

All Sonnet/Opus calls go through Claude Code's Agent tool. There's no SDK driver. If resuming, fire agents via the Agent tool, not direct API.

---

## What got KILLED this session

These are now in `sonnet_ladder_v2/KILL_LIST.md`. Future runs will see them.

| Kill | Reason | Source |
|---|---|---|
| **Cluster C1 (cross-layer W_Q subspace)** — ALL variants (R-CROSS-Q, C-CQBAL, F-CQSGC, A-GFRS, F2-CQBS, R3-XLQB, C2-CQST, A1-GHPI) | Matched-spectrum random-orientation baseline showed real var@128 = 0.2560 vs permuted-rows var@128 = 0.2559 (gap 0.0001). The "cross-layer subspace coherence" was within-layer CF11 concentration trivially summed. W_Q subspaces rotate independently across all 28 layers. | `sonnet_ladder_v2/cheap_tests/run_001/fcqsgc_result.md`. KILL entry `v2-CHEAP-TEST-001`. |
| **F-GFQO (gauge-freedom general residual rotation)** | CF9 partial: arbitrary orthogonal rotation incompatible with RMSNorm per-channel γ. Only generalized permutations (sign-flip × permutation) commute with diagonals. | Run 004 Stage 3. KILL entry `v2-S3-R004-001`. |
| **F4-SVAL-CONSERVED (Sinkhorn-balance W_Q)** | Pre-empted by **SINQ (arXiv:2509.22944, Sep 2025, Huawei)** — same Sinkhorn-Knopp gauge-balancing for LLM compression, evaluated on Qwen3 family AND in HuggingFace Transformers. | Run 008 Stage 3. KILL entry `v2-S3-R008-001`. |
| **F2-CKDJ (softmax shift-invariance × static channel offset)** | CF9 RoPE × shift-invariance: the static offset varies per key position post-RoPE rotation. Same failure pattern as F3-SRSC (run 003). | Run 009 Stage 3. KILL entry `v2-S3-R009-001`. |
| **F3-SRSC (softmax row-shift calibration)** | CF9 RoPE breaks shift-invariance. Run 003 Stage 6 caught after Stage 3+5 missed. Track A picked F5-HPGO runner-up. | Run 003 Stage 6. |
| **F1-MQKP (Eckart-Young on W_Q W_K^T product)** | A3 (arXiv:2505.12942) Theorem 2 explicitly does this. Stages 3+5 missed; Stage 6 caught. Run 009 Track B picked R9-R1/ATQKV runner-up. | Run 009 Stage 6. |
| **F1-VOFR / R6-WVOP / F-WVOS (W_V·W_O fold)** — DOWNGRADE not kill | A3 v3 (Jun 2025) jointly compresses OV at 70B SOTA. Survives as "Qwen3-1.7B specific characterization measurement" but novelty narrowed. | Run 003/006/004 Stage 6. |
| **R6-MLA-POSTRAIN (post-training MLA conversion via CF11)** | Pre-empted by TransMLA (NeurIPS 2025 Spotlight). | Run 006 Stage 3. |

### Cross-cutting structural finding: A3 is the dominant 2025 attention-compression paper

A3 (arXiv:2505.12942) hit at Stage 6 in runs 004, 007, 009, and indirectly run 006. **Stage 6's third-pass prior-art search is the actual ceiling on novelty claims** in this domain. Without it the cascade would have committed to re-deriving A3 multiple times.

---

## What SURVIVED this session (still alive as ideas)

These came up in runs 001-009 and are *not* on the kill list. Per the user's "treat anything from prior to this session as off-limits" directive, these are the only legal targets for follow-on testing.

| Idea | What it is | Cost |
|---|---|---|
| **F6-WALIGN** | Per-neuron W_gate/W_up cosine alignment; algebraic-identity foldability test. ICLR 2025 Yadav notes alignment but no compression framing. | 5-10 min |
| **A1-GFRS-2** | Z_2^d residual-stream sign-entropy gauge on W_up. Gauge-symmetry novelty. | 10 min |
| **F3-L1GNF** | CF6 Layer-1 null-space fold predictor (does null(C_x^0) alignment explain the 36% foldable anomaly?). NO-GO is itself a CF candidate. | 25 min |
| **F-WNORM (W_down rank sweep)** | Closes CF8 across all 3 MLP matrices. Worry: ARA + SVD-LLM V2 report W_V resists SVD; W_down is different. | 47 min |
| **C-WVOK (CF11→W_V rank coupling)** | Within-layer W_V SVD; cluster convergence rep. Stage 6 flagged elevated NO-GO prior (ARA results). | 20 min |
| **C14-CFSHIFT** | Calibration-derived CF6 Layer-1 constant produces precomputable Layer-2 query offset. Zero per-token runtime cost. | 30 min |
| **R9-R1/ATQKV (W_V/W_O spectrum audit)** | Replaces F1-MQKP after A3 kill. Closes the 4-matrix attention picture. | ~1 hr |
| **S3 ATRC** (run 007) | W_V/W_O spectrum sweep with CF11-metric framing. Spectral Lifecycle (2604.22778) partially anticipates direction; Qwen3-1.7B-specific characterization survives. | 35 min |
| **KSATT → S7 Kronecker compound** (run 007) | Retroactive Kronecker characterization of trained W_Q. Novel, but Stage 6 caught critical scope ambiguity (global vs per-head VanLoan partition). | 5 + 30 min |

---

## What got VALIDATED (real empirical results)

Recorded in `sonnet_ladder_v2/SUMMARY.md` as `v2-CF1` and `v2-CF2`:

- **v2-CF1 — CF3 generalizes to all 28 layers (Qwen3-1.7B)**: K=1% Jaccard < 0.50 in 27/28 layers (mean 0.39, range [0.22, 0.531]). L27 marginal at 0.531. Permutation control 0.0053 validates measurement. Random-init gap +0.18 (learned structure with some architectural contribution). Source: `cheap_tests/run_001/rraok_result.md`.

- **v2-CF2 — Cross-layer W_Q subspace sharing class-killed**: details above under "What got KILLED."

### M-Strat v2 verdict (separate workstream from RAOK)

CF15's two sub-claims tested via skeptic controls:

- **CF15-B (channel consistency at [1999, 1793] across layers 2-27)**: STRONGLY CONFIRMED as training-induced. Random-init shows random different channels per layer with no consistency.
- **CF15-A (intrinsic dim ~223)**: SUBSTANTIALLY WEAKER. Architecture caps at ~245 (random-init); training contributes ~9% concentration. Strongly corpus-dependent (code_like 58, NL 168, baseline 220).

Source: `opus_pipeline/round1/m_strat_v2_controls.md`.

---

## RAOK status detail (the active question)

### What's confirmed about RAOK-on-Qwen3-1.7B

- **Rung 1 GO**: CF3 K=1% Jaccard < 0.50 in 27/28 layers.
- **Rung 2 GO**: three-tier scheme gives ΔNLL = 0.083 ± 0.016 nats on 512-token WikiText-2 eval, 3 seeds.
- **Audit Check 5 PASS**: 2K eval gives ΔNLL = 0.0712 (sublinear drift). Result holds at longer eval.
- **Audit Check 6 PASS**: code corpus mean ΔNLL = 0.051 (even better than NL).
- **Audit Check 3 PASS**: Tier-2-zero gives 12.28 nats (proves bulk INT4 carries real signal, not artifact).
- **Adv-1 SURVIVES**: flat single-scale INT4 destroys (5.75 nats). Tier *structure* required.
- **Adv-3 SURVIVES**: static Tier-1 mean magnitude is 5.7× worse than dynamic. Per-token rotation IS load-bearing.

### What COLLAPSED about RAOK's novelty narrative

- **Adv-2 KILL**: random Tier-0 = 0.057 nats vs CF3-derived = 0.083 nats. Random is *better*. CF3-specific channel selection contributes nothing. The "calibration-free CF3-Jaccard-derived" novelty claim is dead.
- **Audit Check 7 SOFT-FAIL** (resolved by Check 5): 0.083 below 512-token marginal noise floor 0.145. Resolved by 2K eval at 0.0712 (above marginal floor at that size).

### What's still genuinely uncertain (these tests resolve it)

- **Sharp ablations v2** (running): does **calibration-derived per-channel INT4** alone match RAOK's 0.083? If yes → RAOK collapses to published-art per-channel INT4 (SmoothQuant family). If no → tier structure is genuinely load-bearing.
  - v1 sharp ablations had a bug (per-token-per-channel scale = trivial bit-exact reconstruction → ΔNLL = 0.000). v2 fixes by using calibration max-abs across many tokens per channel.

- **Cross-scale on Qwen3-4B** (running): does the scheme transfer to a different model size? Tier-0 channel identity may be Qwen3-1.7B-specific.

- **W4A4 composition** (running): RAOK activations × INT4 weights — destructive interference would kill the deployment story regardless of RAOK's standalone goodness.

---

## Files of record

```
sonnet_ladder_v2/
├── SUMMARY.md                      # Empirical substrate (CF1-CF12 confirmed; v2-CF1, v2-CF2 added)
├── KILL_LIST.md                    # Cumulative kills + new v2 entries
├── META_RUN_STATUS.md              # 50-run table (5 complete, 4 partial)
├── SELECTED.md                     # (note: v2 didn't write into this; per-run stage5_selector.md has the picks)
├── prompts/                        # Agent A & B's redesigned prompts
│   ├── STAGE1_IDEATOR.md
│   ├── ORIENTATIONS.md             # 5 orientations (R/C/F/U/A) with FREE SWING + OUT-OF-ORIENTATION wildcards
│   ├── STAGE2_JUDGE.md
│   ├── STAGE3_DEEP_RESEARCHER.md
│   ├── STAGE4_SKEPTIC_EXPLORER.md
│   ├── STAGE5_SELECTOR.md
│   ├── STAGE6_RED_TEAM.md
│   ├── PIPELINE.md                 # Full v2 runbook
│   └── ROUND_OVER_ROUND.md
├── runs/run_001 .. run_009/        # Per-run stage outputs
└── cheap_tests/run_001/            # RAOK ladder of validation tests
    ├── rraok_result.md             # Rung 1
    ├── rraok_rung2_result.md       # Rung 2 (the headline 0.083 nat result)
    ├── rraok_adversarial_result.md # Adv-1/2/3
    ├── rraok_audit_result.md       # Checks 1-7 of easy-explanation audit
    ├── rraok_sharp_ablations_result.md         # v1 (BUGGY — per-channel INT4 was trivial bit-exact)
    ├── rraok_sharp_ablations_v2_result.md      # v2 (in flight — fixed)
    ├── rraok_cross_scale_4b_result.md          # In flight
    ├── rraok_w4a4_composition_result.md        # In flight
    └── fcqsgc_result.md            # The Cluster C1 kill verdict

scripts/
├── rraok_rung1.py                  # Validated
├── rraok_rung2.py                  # Validated
├── rraok_int4_unit_test.py         # B4 scale=max_abs/7 verified
├── rraok_adversarial.py            # Validated
├── rraok_audit_tier2_zero.py       # Validated (Checks 3/5/6/7)
├── rraok_sharp_ablations.py        # BUGGY — DO NOT USE
├── rraok_sharp_ablations_v2.py     # Fixed (in flight as of 9:30 AM)
├── rraok_cross_scale_4b.py         # In flight
├── rraok_w4a4_composition.py       # In flight
├── fcqsgc_cross_layer_wq_svd.py    # The Cluster C1 killer
├── m_strat_rsidc_v2_controls.py    # CF15 verification (DONE)
└── (many more pre-existing scripts under sonnet_ladder/)

experiments/stage0/ladder_v2/round1_raok70b/
├── tier0_stability_per_layer.json  # Rung 1
├── rung2_results.json              # Rung 2 numbers
├── adversarial_results.json        # Adv-1/2/3
├── audit_results.json              # Audit checks
├── sharp_ablations_results.json    # v1 (buggy)
└── (v2, cross_scale, w4a4 files arriving as agents finish)
```

---

## How to resume after context loss

1. **Read this file first.** It's the canonical state.
2. Check whether the three in-flight tests have completed: list `cheap_tests/run_001/` for `rraok_sharp_ablations_v2_result.md`, `rraok_cross_scale_4b_result.md`, `rraok_w4a4_composition_result.md`.
3. If any are missing: check Python processes (`Get-Process python`) — if alive, wait. If dead, check the agent IDs (`aa6c88714bb1e5bd1`, `a6896f3a5a6ccb22e`, `accffd6f05dd0bbf3`) and SendMessage for status, OR re-fire if the script ran but output didn't write.
4. When all three land, **write a binary verdict** at `sonnet_ladder_v2/RAOK_DEPLOYABILITY_VERDICT.md`:
   - One of: DEPLOYABLE / DEPLOYABLE-AS-PUBLISHED-ASSEMBLY / NOT-DEPLOYABLE-IN-CLAIMED-FORM
   - The order-of-magnitude framing: does this assembly enable a category jump in runnable-model class on the user's hardware?
   - Specific ΔNLL numbers from the three tests
   - Path forward: what the user should do to actually deploy (which tooling, which compositions)
5. Don't run more tests in pursuit of more confidence — commit to the verdict on what's available, per user's strict directive.

---

## User preferences captured this session (memory entries)

- **`feedback_signal_not_gate`** — encode strong preferences as signals + escape-hatch slots, not softened wording.
- **`feedback_complete_specified_counts`** — when user says "run N times," count is the bar; don't downscope on convergence signals. (50-run goal was dropped explicitly mid-session, so this didn't fire.)
- **`feedback_capture_and_fire_cheap_tests`** — auto-fire obvious cheap validation tests during multi-run ideation; aggregate standouts.
- **`feedback_kills_are_progress_research_for_fun`** — call kills plainly when evidence falsifies; user does research for curiosity, hedging reduces information.
- **`feedback_order_of_magnitude_not_specific_target`** (NEW) — capability targets are illustrative; the bar is a category jump in runnable-model-class.

These aren't optional. They shape every verdict and report.

---

## Things I'd still want answered if I could pick (after the in-flight tests)

(For continuity, listed but NOT to be auto-fired without user direction.)

- Does the same RAOK-class scheme apply to KV cache, or is **TurboQuant** (arXiv:2504.19874) the right primitive there? (Almost certainly TurboQuant — published, more compression, statistically lossless.)
- Does ik_llama.cpp's existing Q4_K_M weight kernel + a custom RAOK-like activation pre-processing actually produce a runnable assembly? End-to-end inference test.
- The four runs (006-009) were stage-completed end-to-end. None of their selections were cheap-tested; some (e.g., F6-WALIGN at 5 min) are ripe targets if the user wants a follow-on experiment when RAOK verdict lands.

---

End of handoff.
