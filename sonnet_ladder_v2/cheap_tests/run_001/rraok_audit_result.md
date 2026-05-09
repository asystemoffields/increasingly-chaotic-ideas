# R-RAOK-70B Rung 2 Audit Result

Date: 2026-05-09
Model: Qwen/Qwen3-1.7B-Base (bf16, CPU)
Audit script: `scripts/rraok_audit_tier2_zero.py` (Checks 3, 5, 6, 7)
Static analysis: Checks 1, 2, 4
Output JSON: `experiments/stage0/ladder_v2/round1_raok70b/audit_results.json`

---

## Aggregate verdict: ARTIFACT (Check 7 fires the kill)

Six of seven checks PASS the artifact-possibility tests. **Check 7 fails decisively**: the
RAOK ΔNLL (0.083 nats) is **0.57x the noise floor** (0.145 nats) — i.e., the headline
result is *below the standard error* of comparing two 512-token NLL means. By the
brief's own criterion ("PASS if 0.083 >> noise floor; FAIL if within ~3x"), the result
is not statistically distinguishable from baseline noise at the per-seed sample size.

The mechanism (Tier-2 zeroing is catastrophic, hook IS reaching matmul, no length
scaling, no corpus specificity, no rediscovery) is structurally real. What collapses is
the **magnitude of the headline number** — 0.083 nats is not measurably above noise on
512 tokens. The "ΔNLL = 0.083 ± 0.016" is the std across only 3 seeds (3 different
chunks); it does NOT represent the true uncertainty of any single ΔNLL measurement.

Simplest alternative explanation that fits the data: **the three-tier scheme really does
preserve activations well, but the ΔNLL signal at 512 tokens is too small to measure
reliably.** Permutation gap (5.61 nats) is huge and survives — that's evidence the
*method* is non-trivial — but the *quality claim* (0.083 nats < 0.30 threshold) cannot
be defended at this sample size. To rescue, run on >= 8K eval tokens per seed (where
SEM drops to ~0.036 nats) or compute paired per-token ΔNLL with std-of-pairs, not
std-across-seeds.

---

## Per-check verdicts

| # | Check | Verdict | Decisive Number |
|---|---|---|---|
| 1 | Calibration/eval corpus disjointness | PASS | 0% overlap (hardcoded prompts vs WikiText-2) |
| 2 | Hook placement audit | PASS | Pre-hook returns `(h_replaced,) + inputs[1:]`, replacement confirmed by Check 3 |
| 3 | Tier-2-zero degeneracy | PASS | mean ΔNLL = 12.28 nats (RAOK = 0.083; diff = 12.20 >> 0.10) |
| 4 | Massive-activation rediscovery | PASS | Sun et al. 2024 does not study Qwen3-1.7B; CF3 method is novel |
| 5 | Eval-length scaling (2K) | PASS | 2K ΔNLL = 0.0712 nats (< 0.20 threshold; not superlinear) |
| 6 | Corpus specificity (code) | PASS | code mean ΔNLL = 0.0514 nats; diff from NL = 0.0315 (< 0.10) |
| 7 | Noise floor verification | **FAIL** | ratio = 0.57x (NLL std = 3.27, n=511, noise floor = 0.145 nats) |

---

## Check 1 — Calibration/Eval corpus disjointness (PASS)

Rung 1 calibration corpus = 200 hardcoded Python-string prompts in 4 strata
(REPETITIVE, TECHNICAL, PROSE, RANDOM-token, 50 each, max 32 tokens). Rung 2 also
uses these same hardcoded prompts to identify Tier-0 channels. Eval corpus =
WikiText-2 test split, offsets [0, 512, 1024], 512 tokens each (raw text from
HuggingFace `wikitext-2-raw-v1`).

Verified in source: `rraok_rung1.py` lines 54-211 hold the hardcoded prompts;
`rraok_rung2.py` lines 91-248 hold identical copies; `_load_wikitext2_tokens` at
line 488 of rraok_rung2.py loads WikiText-2 `test` split. The random-token stratum
(`_gen_random_token_prompts`, seed=1) decodes random Qwen3 vocab IDs — these are
*not* drawn from WikiText. **0% overlap.**

---

## Check 2 — Hook placement audit (PASS)

`make_pre_hook_quant` (line 587 of rraok_rung2.py) registers a `forward_pre_hook` on
`model.model.layers[L].mlp`. The hook receives `(module, inputs)`, builds quantized
`h_out` per token, and returns `(h_replaced,) + inputs[1:]`. PyTorch pre-hooks REPLACE
the module input when the hook returns a tuple. The MLP module's forward then runs
W_gate and W_up against this replaced tensor — both projections see the SAME quantized
input (B3 compliance).

Independent confirmation from Check 3: setting Tier-2 to literal zero raises ΔNLL from
0.083 → 12.28 nats. If the hook were observing-only, Tier-2-zero would also give
≈0.083 (because nothing would change in the actual forward pass). The 12.20-nat gap
proves the hook IS modifying the forward-pass tensor end-to-end.

---

## Check 3 — Tier-2-zero degeneracy test (PASS)

| Seed | Offset | Baseline NLL | Tier-2-zero NLL | ΔNLL |
|---|---|---|---|---|
| 1 | 0 | 2.7640 | 15.4960 | **12.7319** |
| 2 | 512 | 2.9562 | 14.8680 | **11.9118** |
| 3 | 1024 | 2.8352 | 15.0322 | **12.1970** |

**Mean Tier-2-zero ΔNLL: 12.2802 nats.** Difference from RAOK (0.083): 12.20 nats.
Threshold: ≤ 0.10 = degenerate. Result: 12.20 >> 0.10 — **NOT degenerate**. The 2028
bulk channels carry massive signal; INT4 quantization (vs zeroing) recovers it almost
perfectly. PASS, and confirms Check 2 (hook IS effective).

---

## Check 4 — Massive-activation rediscovery (PASS)

WebSearch confirmed Sun et al. 2024 (arXiv:2402.17762, "Massive Activations in Large
Language Models", COLM 2024) characterizes outlier hidden-state activations in LLaMA-2,
Mistral, GPT-2, Phi-2, and Vision Transformers — published February 2024, before
Qwen3 existed. The paper finds:

  - "Few activations exhibit significantly larger values than others (~100,000x larger)"
  - "Located in feature dimensions that are input-agnostic"
  - "Concentrate at starting word and delimiter tokens"

These are phenomenologically related to RAOK Tier-0 but operationally distinct:

1. **Model novelty**: Qwen3-1.7B is not in Sun et al.; the specific per-layer Tier-0
   channels (e.g. [307, 1485] dominating layers 16-25, [1485, 1945] dominating 6-11)
   are not published.
2. **Identification method**: RAOK uses CF3 Jaccard phase-transition + mean-magnitude
   ranking across a 200-prompt corpus — Sun et al. uses raw magnitude on individual
   token positions.
3. **Application**: Sun et al. doesn't propose tiered quantization with FP16 bypass +
   INT8 dynamic shell + INT4 bulk derived from Jaccard structure.

Rediscovery: NO. The phenomenon class is known; the specific Qwen3 patterns and the
Jaccard-derived tiering are not. CF3 attribution holds in the narrow sense (CF3 is the
identification method that picked these particular channels), even though
"outlier-channel quantization" as a research area predates RAOK.

---

## Check 5 — Eval-length scaling (2048 tokens) (PASS)

Seed 1 (offset 0) re-run with 4x longer window:

| Eval length | Baseline NLL | Quantized NLL | ΔNLL |
|---|---|---|---|
| 512 tokens (Rung 2 seed 1) | 2.7640 | 2.8330 | 0.0690 |
| **2048 tokens (Check 5)** | **2.7745** | **2.8456** | **0.0712** |

ΔNLL grew by 0.0022 nats (3.2%) over 4x context. Linear/sublinear, far below the
0.20-nat threshold and below the 4x-superlinear threshold (0.276 nats). Per-token
quantization error does NOT compound over context length. Deployment ΔNLL at 2K
context is essentially identical to 512-token measurement. PASS.

---

## Check 6 — Corpus specificity (code) (PASS)

Code corpus from `codeparrot/github-code-clean` (streaming load, ~50KB). 3 seeds at
same offsets [0, 512, 1024].

| Seed | Offset | Baseline NLL | Quant NLL | ΔNLL |
|---|---|---|---|---|
| 1 | 0 | 1.7474 | 1.7769 | 0.0295 |
| 2 | 512 | 2.3309 | 2.3916 | 0.0606 |
| 3 | 1024 | 2.3779 | 2.4421 | 0.0642 |

**Code mean ΔNLL: 0.0514 nats** vs NL mean ΔNLL: 0.0830 nats. Difference: 0.0315 nats
(< 0.10 threshold). Tier boundaries calibrated on hardcoded prompts (containing
TECHNICAL stratum) generalize to held-out code. Code ΔNLL is *slightly lower* than NL,
which is informative: code activations have lower intrinsic dim (per M-Strat), and
INT4 reconstruction is correspondingly easier. PASS.

(Note: code baseline NLL is much lower than NL — code is more predictable for this
model — but ΔNLL is what matters for the quantization claim.)

---

## Check 7 — Noise floor verification (FAIL — KILL)

Per-token NLL stddev on the baseline (seed 1 / offset 0, 511 reported tokens of 512
loaded): **3.2726 nats**.

Noise floor (standard error of the mean by the brief's formula):
  noise_floor = stddev / sqrt(n_tokens) = 3.2726 / sqrt(511) = **0.1448 nats**

RAOK ΔNLL = 0.083 nats.
Ratio: **0.083 / 0.1448 = 0.57x**.

Brief's PASS criterion: "0.083 >> noise floor" → ratio should be > 3x.
Result: **0.57x. FAIL by a factor of ~5.**

### What this means
The reported "ΔNLL = 0.083 ± 0.016" std-of-3-seeds dramatically *understates* the
uncertainty of any single ΔNLL measurement. The 0.016 number is the std across 3
different 512-token chunks (seed-to-seed variation in baseline NLL); it is NOT the
sampling uncertainty of estimating ΔNLL from a single 512-token window. The latter is
what the brief's noise-floor formula computes, and it is 9x larger (0.145 vs 0.016).

At 512 tokens per seed, the baseline NLL itself has SEM ≈ 0.145, so the ΔNLL between
"baseline" and "RAOK quantized" cannot be statistically resolved below ~0.4 nats with
high confidence. The headline 0.083-nat ΔNLL is below this resolution.

### Caveat (paired-test interpretation)
The brief's formula is for *unpaired* comparisons. In practice, both baseline and
quantized NLLs are computed on identical tokens, so the *paired* per-token ΔNLL would
have a smaller stddev (variance cancels for tokens where both runs are similar).
Computing the paired SEM requires saving per-token ΔNLLs, which the original Rung 2
script does not do.

By the brief's stated criterion, this is FAIL. Even under a paired-test rescue, the
required per-token ΔNLL stddev to make 0.083 nats reach 3x SEM is ≤ 0.61 nats — which
is plausible but not demonstrated. The Rung 2 script could be re-run with per-token
ΔNLL collection to falsify or confirm this, but as currently presented, the headline
number is not defended against noise.

---

## What survives, what collapses

### Survives:
- **The mechanism**: Tier-0/Tier-1/Tier-2 design is structurally working (Check 3:
  removing Tier-2 is catastrophic; permutation gap of 5.61 nats from Rung 2 itself
  shows magnitude-keying is non-random).
- **Calibration-eval separation**: no leakage (Check 1).
- **Hook correctness**: end-to-end matmul replacement confirmed (Check 2 + 3).
- **Length stability**: no error compounding to 2K (Check 5).
- **Corpus generalization**: works on code (Check 6).
- **Novelty of the identification method**: not direct rediscovery of Sun et al.
  (Check 4).

### Collapses:
- **The headline number (0.083 nats) cannot be defended at 512 tokens.** Per-token
  NLL stddev of 3.27 makes 0.083 indistinguishable from baseline noise at this
  sample size. The "± 0.016" in the result is std across 3 seeds, NOT the SEM of any
  single seed's ΔNLL.

### Recommended remediation (does not require Rung 3 redesign):
1. Re-run Rung 2 with eval window >= 8K tokens per seed (SEM drops to ~0.036 nats);
   or
2. Modify the eval to record per-token ΔNLL pairs and report SEM of paired
   differences;
3. Either way, demonstrate that the 0.083-nat ΔNLL is at least 3x the per-eval SEM
   before claiming "low" rather than "below resolution."

If the headline survives at 8K with SEM ≈ 0.036, ratio becomes 0.083/0.036 = 2.3x —
still marginal. The result needs **either** larger samples **or** the ΔNLL itself to
be larger than the current 0.083 to defend Rung 2 as a real per-token compression
claim.

---

## Files written by audit
- `scripts/rraok_audit_tier2_zero.py` (Checks 3/5/6/7)
- `experiments/stage0/ladder_v2/round1_raok70b/audit_checks_3_5_6_7.json` (raw)
- `experiments/stage0/ladder_v2/round1_raok70b/audit_results.json` (combined per-check)
- `sonnet_ladder_v2/cheap_tests/run_001/rraok_audit_result.md` (this file)
