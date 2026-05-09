# Session Findings — 2026-05-09

End-of-session synthesis. Companion to `RAOK_DEPLOYABILITY_VERDICT.md` and `SESSION_HANDOFF_2026_05_09.md`.

---

## Headline

**RAOK is closed: collapsed to per-channel INT4 activation quantization (SmoothQuant-class). No novel research contribution.**

The ladder cascade plus the cheap-test discipline did exactly what they were designed to do — surface a confident-looking finding, then kill it under skeptic-controls discipline. The system worked; the specific bet didn't.

The session's three subsequent kills (F6-WALIGN, A1-GFRS-2, F-WNORM) extended this pattern: **the trained Qwen3-1.7B MLP weight tensor has no novel local-statistical structure beyond what published per-element-quantization methods already extract.** This is itself a v2 structural finding worth recording.

---

## Empirical findings recorded as v2-CFs

| ID | Finding | Status |
|---|---|---|
| v2-CF1 | CF3 generalizes to all 28 layers (Jaccard@K=1% < 0.50 in 27/28) | Confirmed |
| v2-CF2 | Cluster C1 cross-layer W_Q subspace sharing — CLASS-KILLED via matched-spectrum baseline | Class kill |
| v2-CF3 | CF8 extends to all 3 MLP matrices: W_down also full-rank (r_99 = 92% of d_model) | Class kill |
| v2-CF4 | Per-neuron W_gate/W_up alignment is distributed (~5× random median) but not localized (0/172k at \|cos\|≥0.99) | Soft structural |
| v2-CF5 | W_up canonical-gauge sign entropy = 0.9996 bits/weight (uniform random, identical to random-init) | Hard kill on sign-codebook |

**Combined implication**: trained Qwen3-1.7B MLP weights are essentially "well-mixed" at all surface-statistical structures we tested. Whatever non-uniform structure exists is captured by per-channel magnitude scaling (which is what published per-channel INT4 / Q4_K_M / SmoothQuant variants exploit). Other surface-statistical exploits — sign distribution, per-neuron foldability, low-rank decomposition — yielded no novel compression headroom.

---

## What happened to RAOK in 12 layers of testing

1. **Rung 1 GO**: CF3 K-dependent dynamicity generalizes to 28 layers (real)
2. **Rung 2 GO**: three-tier scheme gives ΔNLL = 0.083 nats (real)
3. **Audit Check 5 GO**: 2K-token eval ΔNLL = 0.071, sublinear drift (real)
4. **Audit Check 6 GO**: code corpus ΔNLL = 0.051 (real)
5. **Adv-1 SURVIVES**: tier structure required (flat-INT4 destroys at 5.75 nats)
6. **Adv-3 SURVIVES**: dynamic Tier-1 required (static gives 0.47 nats vs 0.07)
7. **Audit Check 3 PASS**: bulk INT4 carries signal, hook works (12.3 nat gap on Tier-2-zero)
8. **Adv-2 KILL**: random Tier-0 = 0.057 nats, *better* than CF3-derived = 0.083. **CF3-specific channel selection is not load-bearing.**
9. **Sharp ablations v2 KILL**: calibration-derived per-channel INT4 alone = 0.107 nats (within noise of RAOK's 0.083). **Tier-0 + Tier-1 not load-bearing beyond plain per-channel INT4.**
10. **W4A4 composition CLEAN**: composition delta = 0.008 nats (RAOK + W4 weights stack additively, deployment math holds)
11. **TurboQuant pre-empts KV branch**: 6× KV cache compression at 3-bit keys / 2-bit values, statistically lossless (Google DeepMind, ICLR 2026)
12. **Cross-scale infeasible** without bitsandbytes; only Qwen3-0.6B fallback measured (downward, less informative)

**Verdict**: RAOK = published per-channel INT4 activation quantization with extra bookkeeping. The deployment recipe (per-channel INT4 activations + W4 weights + TurboQuant KV + mmap'd storage) is achievable today via ik_llama.cpp and similar tools.

---

## What the cascade infrastructure proved (separate from RAOK)

The v2 ladder is structurally sound:

- **Skeptic-controls gate worked**: Adv-2 (random Tier-0) and the sharp ablations bug-fix v2 caught the novelty collapse that Stages 1-5 missed.
- **Stage 6 prior-art third-pass worked**: A3 (arXiv:2505.12942), TurboQuant, SINQ, PALU, TransMLA all caught at Stage 6 across various runs after Stage 3 cleared them.
- **Cheap-test discipline worked**: F-CQSGC (run 001) and the RAOK adversarial sequence both produced kills that would have propagated as "validated structural findings" without immediate empirical pressure-testing.
- **Cross-run convergence detection worked**: cross-layer W_Q showed up as a 4-orientation convergence in run 001 AND run 002 — the cheap test caught the shared error before it propagated.

The cascade infrastructure is reusable for future research. The framework discriminates real findings from artifacts. RAOK is a clean negative; that's exactly the kind of result the system is supposed to produce.

---

## Path forward for the user's actual goal

The deployment goal — **order-of-magnitude jump in runnable model size on 7.28 GiB CPU-only Ryzen** — is achievable but **does not require novel research from this session**. The path:

1. **Try Qwen3-14B at IQ2_XS or IQ3_XS via ik_llama.cpp.** This is the empirical test that would settle whether step-change deployment works with off-the-shelf tools today.
2. If that works at acceptable quality (ΔNLL < 0.5) and speed (≥ 1 tok/s), the deployment goal is met.
3. If not, the productive research direction is **better aggressive weight quantization** (GPTQ / AWQ / QuIP-class on Qwen3 family) — a totally separate research program from RAOK.

The cascade's most useful output is *not* a novel mechanism but a *clear no-result map* — knowing which directions are dead saves future work.

---

## Files of record

```
sonnet_ladder_v2/
├── RAOK_DEPLOYABILITY_VERDICT.md   # Binary call on RAOK
├── SESSION_HANDOFF_2026_05_09.md   # Full state for resume
├── SESSION_FINDINGS_2026_05_09.md  # This file
├── SUMMARY.md                       # Updated with v2-CF1..v2-CF5
├── KILL_LIST.md                     # 4 v2-cycle kills + v1 inheritance
├── META_RUN_STATUS.md               # 5 runs end-to-end + 4 partial
└── cheap_tests/run_001/             # 8 tests including 3 NO-GO closure tests
    ├── rraok_result.md              # Rung 1 GO
    ├── rraok_rung2_result.md        # Rung 2 GO (0.083 ΔNLL)
    ├── rraok_adversarial_result.md  # Adv-1/2/3
    ├── rraok_audit_result.md        # Checks 1-7
    ├── rraok_sharp_ablations_result.md         # v1 (BUGGY)
    ├── rraok_w4a4_composition_result.md        # W4 + RAOK clean
    ├── fcqsgc_result.md             # Cluster C1 kill
    ├── f6_walign_result.md          # NO-GO
    ├── a1_gfrs_2_result.md          # NO-GO
    └── f_wnorm_result.md            # NO-GO

experiments/stage0/ladder_v2/
├── round1_raok70b/                  # All RAOK results JSON
├── round1_walign/                   # F6-WALIGN
├── round1_gfrs2/                    # A1-GFRS-2
└── round1_wnorm/                    # F-WNORM
```
