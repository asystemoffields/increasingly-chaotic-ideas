# RAOK Deployability Verdict

Date: 2026-05-09
Bar: order-of-magnitude jump in runnable model size on 7.28 GiB Ryzen 5 7530U (CPU-only)

---

## Verdict: **DEPLOYABLE-AS-PUBLISHED-ASSEMBLY (NO NOVEL CONTRIBUTION FROM RAOK)**

The mechanism RAOK isolates is real and works (per-channel INT4 activation quantization with calibration-derived scales gives ΔNLL ≈ 0.08-0.11 on Qwen3-1.7B at 4× activation compression). It composes cleanly with W4 weight quantization. **But the mechanism is published prior art (SmoothQuant-class)**; RAOK's novelty claim collapses under controlled adversarial testing. The deployment recipe RAOK points toward — aggressive activation/weight quantization on a CPU-only laptop — is already available via ik_llama.cpp's existing kernels and quantization formats. No new capability is enabled by RAOK that wasn't already achievable.

---

## What was tested

### Confirmed real

- **Rung 1 (CF3 dynamicity to all 28 layers)**: K=1% Jaccard < 0.50 in 27/28 layers. Real structural property of trained Qwen3-1.7B activations. ✓
- **Rung 2 (three-tier ΔNLL)**: 0.083 ± 0.016 nats on Qwen3-1.7B WikiText-2, 3 seeds. Holds at 2K eval (Check 5 → 0.0712). Generalizes to code corpus (Check 6 → 0.051 mean). ✓
- **Adv-1 (flat-INT4 destroys)**: 5.75 nats. Tier *structure* (some kind of magnitude isolation) matters. ✓
- **Adv-3 (static Tier-1 worse)**: 0.473 nats — 5.7× worse than dynamic. Per-token rotation is load-bearing. ✓
- **W4A4 composition** (Qwen3-1.7B): RAOK + per-channel-INT4 weights compose additively. Composition delta = 0.008 nats. ✓
- **Audit Check 3** (Tier-2-zero): 12.7 nats. Bulk INT4 channels carry real signal. Hook is wired correctly. ✓

### Killed

- **CF3-specific Tier-0 channel selection**: KILL (Adv-2). Random Tier-0 (any 2 channels at FP16) gives ΔNLL = 0.057, *better* than CF3-derived Tier-0's 0.083. The "calibration-free CF3-Jaccard-derived static channel oracle" claim is dead.
- **Three-tier scheme novelty over per-channel INT4**: KILL (Sharp Ablations v2). Calibration-derived per-channel INT4 alone gives ΔNLL = 0.107 (gap from RAOK = 0.024 — within noise band). Removing Tier-1 INT8 shell entirely (P2): ΔNLL = 0.103 (gap = 0.020). **The Tier-0 + Tier-1 structure are not load-bearing beyond plain per-channel INT4.** RAOK reduces to SmoothQuant-class per-channel INT4 activation quantization.
- **KV-cache application of RAOK**: pre-empted by **TurboQuant** (arXiv:2504.19874, Google DeepMind, ICLR 2026 — 6× KV compression at 3-bit keys / 2-bit values, statistically lossless). RAOK at best matches a less efficient version of an existing primitive on the KV side.

### Not tested (and why this is acceptable)

- **Cross-scale to Qwen3-4B BF16**: infeasible on 7.28 GiB without bitsandbytes or streaming-quantize-load. Even running it on 4B doesn't prove step-change because **4B already runs on this machine** at ~6 tok/s via ik_llama.cpp Q4_K_M. Testing the assembly on 4B is feasibility check, not deployability check.
- **W2 + RAOK on 1.7B**: deferred. The bottleneck for aggressive weight quantization (W2/W3) is the weight-quant *method* (GPTQ / AWQ / QuIP / OmniQuant family), not activation handling. RAOK-class activation discipline doesn't fundamentally rescue weight-quant noise; it provides bounded activations for fast INT×INT kernels but doesn't fix the per-weight precision loss that limits W2.

---

## Numbers

| Test | Mean ΔNLL | Stddev | Verdict |
|---|---|---|---|
| RAOK three-tier (Rung 2) | 0.083 | 0.016 | baseline reference |
| Per-channel INT4 calibration (P1-fixed) | 0.107 | 0.033 | **within noise of RAOK** |
| Tier-0 + per-channel INT4 (P2-fixed minimal RAOK) | 0.103 | 0.018 | **within noise of RAOK** |
| Random Tier-0 instead of CF3 | 0.057 | 0.017 | better than CF3 |
| Static (non-dynamic) Tier-1 | 0.473 | 0.063 | 5.7× worse — dynamicity matters |
| Flat single-scale INT4 | 5.750 | 0.128 | catastrophic — published baseline |
| W4-only on Qwen3-1.7B (per-channel INT4 weights) | 0.590 | 0.010 | naive W4 baseline |
| W4 + RAOK | 0.681 | 0.021 | composition delta = 0.008 |

Audit:
- 2K-token eval ΔNLL = 0.071 (sublinear vs 512-token's 0.069)
- Code corpus ΔNLL = 0.051 mean
- Tier-2 zero ΔNLL = 12.28 (bulk channels real)
- Permutation gap = 5.61 nats (within-tier shuffle destroys; tier assignment is keyed)

---

## What this means for the deployment goal

The user's bar (per `feedback_order_of_magnitude_not_specific_target`): "an order of magnitude change in what can be run locally."

Current ceiling on this hardware (7.28 GiB, CPU-only Ryzen 5 7530U):
- Qwen3-4B at IQ4_XS: ~6 tok/s (existing, via ik_llama.cpp)
- Qwen3-7B at Q4_K_M: probably runs (memory: weights ~4.5 GB, fits)
- Qwen3-14B at Q4_K_M: ~7.5 GB weights — borderline, likely doesn't fit comfortably
- Qwen3-14B at IQ2_XS: ~4-4.5 GB weights — **fits, untested empirically by user**
- Qwen3-32B+: doesn't fit at any current quant on this hardware without NVMe streaming

The capability that would constitute step-change: running Qwen3-14B (or larger) at usable quality and speed.

The mechanism that enables this: **aggressive weight quantization (Q2_K / IQ2_XS / IQ3_XS)**, which is already in ik_llama.cpp, NOT activation-side discipline.

RAOK's contribution to this picture: **none beyond what published primitives provide**. Per-channel INT4 activation quantization is in SmoothQuant; KV compression is in TurboQuant; aggressive weight quant is in IQ2_XS/Q2_K. The composition is what enables deployment, and the composition is achievable today.

---

## Specific recommendation for the user

**The actually unfinished work for the deployment goal is:**

1. **Try ik_llama.cpp + Qwen3-14B at IQ2_XS or IQ3_XS on this machine.** This is the test that would empirically settle whether step-change deployment is currently feasible with off-the-shelf tools. If it works at acceptable quality (ΔNLL < 0.5 vs FP16) and speed (≥1 tok/s), the deployment goal is *already met without any novel research contribution*.

2. If 14B at IQ2/IQ3 is too lossy, the productive research direction is **better aggressive weight quantization**, not better activation quantization. GPTQ / AWQ / QuIP-class methods on Qwen3 family is the relevant frontier.

3. KV cache compression (TurboQuant) for long-context use cases is its own composable layer.

RAOK as a research lead is closed. The cascade did its job — produced a confident-looking finding, then surfaced (via skeptic controls) that the finding is published prior art. The cascade infrastructure itself worked exactly as designed; this just isn't the breakthrough.

---

## What the v2 cascade demonstrated about itself (separate from RAOK)

- **Skeptic-controls gate works**: Adv-2 (random Tier-0) and Sharp Ablations v2 caught the novelty collapse that Stages 1-5 missed.
- **Stage 6 prior-art third-pass works**: A3 (arXiv:2505.12942), TurboQuant, SINQ, PALU, TransMLA were all caught at Stage 6 across various runs when Stage 3 had cleared them.
- **Cheap-test discipline works**: F-CQSGC (run 001) and the RAOK adversarial sequence both produced kills that would have propagated as "validated structural findings" without immediate empirical pressure-testing.

The cascade is structurally sound. It's a research methodology that produces honest negative findings. RAOK just isn't the positive finding.

---

## Files of record

```
sonnet_ladder_v2/cheap_tests/run_001/
├── rraok_result.md                    # Rung 1 GO
├── rraok_rung2_result.md              # Rung 2 GO (0.083 ΔNLL)
├── rraok_adversarial_result.md        # Adv-1/2/3
├── rraok_audit_result.md              # Checks 1-7
├── rraok_sharp_ablations_result.md    # v1 BUGGY
└── rraok_w4a4_composition_result.md   # W4A4 composition

experiments/stage0/ladder_v2/round1_raok70b/
└── sharp_ablations_v2_results.json    # The decisive collapse test
```
