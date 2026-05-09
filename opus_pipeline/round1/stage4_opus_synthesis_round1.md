# Opus Pipeline Round 1 — Synthesis

Date: 2026-05-08
Inputs: Stage 3 refined proposals (Reach TC-MR v2, Composition Outlier Highway v2, First-Principles GeoFingerprint v2, Unconventional PFOR+NEXTENT v2). Stage 2 prior-art filters. Pipeline-accumulated structural findings (R2, CF8, CF10).

---

## Convergence map

Four creative orientations were run in parallel against the same empirical substrate. Where they converge, the model's structure is independently asserting itself.

### C1. Outlier-channel identity is the routing primitive (Composition v2 ∩ Unconventional v2 ∩ Reach v2)

- **Composition v2 M2/M5/M6**: outlier span aligns with W_Q^shared; massive activations form an L1-resident persistent residual highway; ≥0.85 information concentration at ≤1.1% cardinality.
- **Unconventional v2 ZIPFLOCK**: the top-30% Zipf weights (~6 GiB) are the right thing to VirtualLock; PFOR spectral ordering is a globally-optimal version of "outlier-co-access stays on the same page".
- **Reach v2 π predictor**: token-conditional bond dimension χ_t is selected from per-token "what matters" — and the cheapest signal for "what matters" is the outlier channel pattern of the activation entering the layer.

Convergence: *the outlier channel pattern at layer entry is the routing key*. Composition says it's already aligned with attention's most important subspace; Unconventional says it's also the right key for storage layout; Reach says it's the right key for compute precision. Three orientations, one mechanism: **outlier-conditional layer dispatch**.

### C2. Storage-layout-IS-the-predictor (Unconventional v2 ∩ Reach v2)

- **Unconventional NEXTENT**: PFOR-permuted weights aligned to 1 MB extent boundaries → the NTFS Cache Manager's read-ahead implements the predictor for free.
- **Reach E0a/E0b NVMe streaming**: sub-matrix 4 KB block-aligned NVMe streaming is the substrate; the question is whether OS demand-paging at 100-300 μs/fault is competitive with explicit io_uring-style prefetch.

These are the same load-bearing measurement: **what is the realistic random 4 KB read latency on this NVMe under Windows mmap?** Both proposals branch their cascade on this number. Reach's E0a optimistic / E0b realistic split and Unconventional's E0a / E0b branch are isomorphic. Run it once.

### C3. Joint structure across "different" matrices (First-Principles v2 ∩ Composition v2)

- **First-Principles M1**: a single orthogonal Q* simultaneously low-ranks W_gate, W_up, AND tied W_E. Class boundary CF10 is shadows of one object.
- **Composition v2 M2**: outlier channels span a subspace shared between activations and W_Q^shared rowspan — a different "joint structure" claim, different matrices, but same epistemic shape: *low-dimensional shared geometry that the four-block decomposition view of a transformer obscures*.

Convergence: stop treating each weight matrix as independent. The CF10 results (W_Q head-redundant 0.63, W_K 0.79, MLP 1.0 full-rank, embed 0.97) already say weight-side is heterogeneous; the Composition+First-Principles convergence says **activation-side and weight-side share low-dim subspaces, and so do "unrelated" weight matrices that route through the residual stream**. The residual stream is the conserved coordinate system.

### C4. Per-token routing as the unifying inference-time primitive (all four)

- Reach: per-token χ_t.
- Composition: per-token outlier-set indexes into pre-permuted storage.
- First-Principles: per-token rotated coordinate system (Q* applied to residual stream).
- Unconventional: per-token extent prefetch, per-token VirtualLock retention.

All four converge on the same *control surface*: a small per-token decision that selects from a structured precomputed substrate. The substrate differs (precision/storage/coordinate/residency) but the control point is identical.

---

## Dependency graph

Experiments ordered by what they unlock.

```
                         ┌─────────────────────────────────────┐
                         │ M-NVMe: Windows mmap page-fault     │
                         │   latency benchmark (1 hr)          │
                         │   = Reach E0a/E0b = Unconv E0       │
                         └────────────┬────────────────────────┘
                                      │ branches Reach + Unconv
                                      ▼
                         ┌─────────────────────────────────────┐
                         │ M-Align: outlier × W_Q^shared       │
                         │   principal angles (4 hr)           │
                         │   = Composition E1                  │
                         └────────────┬────────────────────────┘
                                      │ kills Composition if intersection<0.3
                                      ▼
                         ┌─────────────────────────────────────┐
                         │ M-Strat: residual-stream            │
                         │   stratified intrinsic dim (30 min) │
                         │   = First-Principles E1 = RSIDC     │
                         └────┬───────────────┬────────────────┘
                              │               │
                              ▼               ▼
              ┌──────────────────────┐  ┌────────────────────────┐
              │ FP E2: MLP-saturation│  │ Comp E2: concentration │
              │   inequality (30 min)│  │   ≥0.85 at ≤1.1% (2 hr)│
              └──────────┬───────────┘  └───────────┬────────────┘
                         │                          │
                         ▼                          ▼
              ┌──────────────────────┐  ┌────────────────────────┐
              │ FP E3: joint Q*      │  │ Comp E3: persistence   │
              │   construction (4 hr)│  │   across layers        │
              └──────────┬───────────┘  └───────────┬────────────┘
                         ▼                          ▼
              ┌──────────────────────┐  ┌────────────────────────┐
              │ FP E4: rotated tied  │  │ Reach E1: per-token    │
              │   lm_head SETTLING   │  │   weight-contribution  │
              │   (8 hr) — BINARY    │  │   support              │
              └──────────────────────┘  └────────────────────────┘
```

Key observations:
- **M-NVMe gates two of four cascades** (Reach, Unconventional). Run first.
- **M-Align is decisive for Composition** (NO-GO if intersection-fraction <0.3) and is 4 hr.
- **M-Strat (RSIDC) is shared between First-Principles E1 and Composition M6** — one run produces both.
- FP E4 (rotated tied lm_head ΔNLL) is the **single highest-information experiment in the bundle**. It's a binary settler: First-Principles claims ΔNLL << +1.0 vs random Q's +19.96. 8 hours, stake-in-the-ground.

---

## Top 3 directions, ranked

### Rank 1 — First-Principles GeoFingerprint v2 (joint Q* with rotated tied lm_head)

- **Source**: Stage 1 First-Principles, refined Stage 3 with paired-rotation fusion + γ absorption resolving RMSNorm precondition.
- **Ambition ceiling**: 1.4–1.6× lossless-equivalent compression of W_gate/W_up/tied-W_E simultaneously; composes with 4-bit quant for ~6.7× over fp16; **70B → ~3.5 GiB residency, ~1.2 s/token**. Note: residency claim is the most aggressive of the four directions and is the load-bearing one — if residency hits, tok/s follows trivially because the model fits in RAM.
- **Pre-emption resistance per Stage 2**: HIGH. QuIP/QuaRot/SpinQuant/OSTQuant all use rotation for *quantization smoothness*, not joint low-rank. MoDeGPT does intra-module joint SVD but explicitly excludes embedding. The cross-module shared rotation including tied lm_head is a genuine gap.
- **First decisive experiment**: M-Strat (RSIDC) — 30 min — measures whether the residual-stream stratified intrinsic dimension d* < d_model. NO-GO threshold: if d*/d_model > 0.95 across all class-boundary strata, the geometric-fingerprint hypothesis is dead and we save the 8 hours of E4.
- **Why rank 1**: Stage 2 explicitly identifies this as the gap with cleanest novelty. The cascade has a 30-minute screening experiment (RSIDC) that produces a structural finding regardless of go/no-go — the residual-stream intrinsic dimension across CF10 strata is independently valuable for the next round of the pipeline.

#### Honest reach assessment for Rank 1

If the cascade succeeds end-to-end:
- **Qwen3-4B baseline (currently 5.5 tok/s, IQ4_XS, q8_0 KV)**: rotated joint Q* gives 1.4–1.6× weight compression on top of IQ4_XS. Doesn't directly raise tok/s on a model that already fits — gain is residency headroom that lets us turn on bf16 KV (improves quality, not speed) or larger context. Realistic: **5.5 → 5.5–6.2 tok/s** (modest), but **ΔNLL improvement at fixed bit budget**.
- **70B target**: at ~6.7× over fp16, 70B fp16 is 140 GB → 70B compressed is ~21 GB. **That doesn't fit in 7.28 GiB.** The honest residency claim is 70B at ~3.5 GiB only if you also assume 4-bit-equivalent compression on top, which means the 1.4–1.6× from joint Q* is the *first* factor in a stacked compression chain. Realistic 70B residency under fully successful cascade: **3.5–6 GiB, 1.2–2.5 s/token range**, contingent on the joint Q* compressing weights *additively* with quantization (no destructive interaction with quantization grid).
- **Quality cost**: First-Principles predicts ΔNLL << +1.0 for rotated tied lm_head (vs random Q's +19.96). If true, joint Q* on W_gate/W_up adds negligibly more. Realistic ΔNLL budget: **+0.1 to +0.3 nats** at the target compression.
- **P(end-to-end success)**: M-Strat passes (~70%, residual stream is known to have low intrinsic dim in literature). E4 settling within ΔNLL ≤ +1.0 (~45%, RMSNorm-paired rotation is novel and the γ-absorption math has not been bench-tested). Composes with 4-bit (~60%, conditional on E4). **P(all three) ≈ 0.70 × 0.45 × 0.60 ≈ 0.19**. End-to-end probability ~19%.
- But: even on partial success (M-Strat + E4 pass, quant composition fails), the structural finding "tied lm_head admits non-trivial rotation under paired-fusion" is independently a publishable result and motivates round 2.

### Rank 2 — Composition Outlier Highway v2 (alignment-first)

- **Source**: Stage 1 Composition, refined Stage 3 with M2 alignment claim front-loaded as decisive.
- **Ambition ceiling**: 70B 2–3 tok/s reach, 32B 5–8 tok/s probable. Realistic page-fault-adjusted (100–300 μs).
- **Pre-emption resistance per Stage 2**: HIGH on M2 and M5 (no published work measures principal-angle entanglement between outlier span and W_Q^shared; massive activations as exploitable communication medium is novel framing). M1/M3/M4 demoted to substrate.
- **First decisive experiment**: M-Align (Composition E1) — 4 hr — measures dim(span(D^1%) ∩ rowspan(W_Q^shared)) and median principal angle. NO-GO threshold: intersection-fraction <0.3 OR median principal angle >55°.
- **Why rank 2**: 4-hour decisive experiment with crisp numerical go/no-go, and the *measurement itself* — outlier-span × W_Q-rowspan principal angles across layers — is the structural finding regardless of GO/NO-GO. NO-GO kills the highway-as-W_Q-channel framing but produces a cross-layer alignment map that informs Reach's π predictor. GO promotes outlier identity to a free routing key.

#### Honest reach assessment for Rank 2

End-to-end success delivers:
- **Qwen3-4B baseline**: outlier-pinned + concentrated path means smaller working-set residency. 5.5 → likely 6.0–7.0 tok/s on Qwen3-4B by reducing memory traffic via L1-resident outlier highway (Vega 7 / shared cache). Modest direct speedup but large headroom.
- **70B**: under E0b realistic page-fault assumption, the outlier-routing-as-storage-layout combo lets 70B IQ3 stream at 2–3 tok/s. This is the *most credible* 70B-on-7.28GiB number across all four directions because it doesn't rely on stacked novel compression — it relies on storage layout matching access pattern.
- **P(end-to-end)**: M-Align passes (~50% — the claim is bold and pre-art is silent), persistence holds (~70%), gather primitive bench within target (~65%). **P ≈ 0.50 × 0.70 × 0.65 ≈ 0.23**.

### Rank 3 — Unified M-NVMe + ZIPFLOCK (Unconventional v2 substrate, with Reach E0)

- **Source**: Unconventional Stage 1, refined v2 with NEXTENT + ZIPFLOCK + iGPU overlap. Reach E0a/E0b is the *same* gating measurement.
- **Ambition ceiling**: 70B IQ3_XXS at ≥1.5 tok/s under E0b, ≥2.5 tok/s under E0a.
- **Pre-emption resistance per Stage 2**: MEDIUM-HIGH on NEXTENT (extent-aligned read-ahead at 1 MB granularity is not in the published llama.cpp/FlexGen/PowerInfer literature). PFOR alone is partially preempted by Apple two-tensor bundling, but global graph-permutation generalization is novel.
- **First decisive experiment**: M-NVMe — **1 hr** — Windows 11 NVMe mmap demand-page latency benchmark (random 4 KB reads, cold cache, populated working set, varied io_uring depth). Gates not just this direction but Reach's entire cascade.
- **Why rank 3**: lowest experimental cost, gates two cascades, and the measurement itself is a high-value structural finding (current Stage 0 ladder lacks a Windows-specific page-fault number). Direction is rank 3 not rank 1 because the *mechanism* is more engineering than science — even fully successful, it's a systems result rather than a model-structure result.

#### Honest reach assessment for Rank 3

- **Qwen3-4B**: minimal direct effect; the model already fits in RAM.
- **70B**: success delivers the 70B-on-laptop substrate. Under E0a optimistic, 2.5 tok/s; under E0b realistic, 1.5 tok/s. **This is the only direction that produces a directly-shippable 70B-on-7.28GiB result without other cascades succeeding.**
- **P(end-to-end)**: M-NVMe is just a measurement (P=1 of producing a number). PFOR spectral ordering improves over baseline mmap (~70%). NEXTENT extent-alignment provides additional gain (~50%). ZIPFLOCK VirtualLock helps (~80%). End-to-end ≥1.5 tok/s on 70B IQ3_XXS: **~40%**.

---

## Pre-program measurements (run first regardless)

Three measurements benefit two-or-more directions and are individually cheap:

### PRE-1 — M-NVMe (Windows mmap page-fault latency)

- **Beneficiaries**: Reach (E0a/E0b branch), Unconventional (E0 branch). Decides whether the OS demand-pager is competitive or io_uring is required.
- **Cost**: 1 hour.
- **Output**: number (μs / 4 KB random read, cold cache) at queue depths 1, 4, 16, 64. Plus extent-locality bonus (1 MB sequential within extent vs random across extents).
- **Script**: `scripts/m_nvme_pagefault.py`.
- **Output path**: `experiments/stage0/ladder/m_nvme_pagefault/`.
- **Structural finding regardless**: Stage 0 ladder gains a hardware-latency baseline currently absent.

### PRE-2 — M-Strat / RSIDC (residual stream intrinsic dim, stratified by CF10 class)

- **Beneficiaries**: First-Principles E1 (direct), Composition M6 (concentration is dual to intrinsic dim), all four directions (gives a per-layer "how much room is in the residual stream" map).
- **Cost**: 30 minutes.
- **Output**: d*(layer, class) for layers 0..27 of Qwen3-4B, classes from CF10 stratification.
- **Script**: `scripts/m_strat_rsidc.py`.
- **Output path**: `experiments/stage0/ladder/m_strat_rsidc/`.
- **Structural finding regardless**: per-layer residual-stream intrinsic dimension is independently a publishable Stage 0 result. Even if d* ≈ d_model everywhere (kills First-Principles), the *map* of where the residual stream is wide vs narrow is the next round's prior.

### PRE-3 — M-Align (outlier span × W_Q^shared rowspan principal angles)

- **Beneficiaries**: Composition E1 (decisive), Reach π predictor (alignment quantifies how good outlier-identity is as a routing key), First-Principles (cross-module shared subspace evidence).
- **Cost**: 4 hours.
- **Output**: dim(intersection), median principal angle, per-layer plot, per-head plot.
- **Script**: `scripts/m_align_outlier_wq.py`.
- **Output path**: `experiments/stage0/ladder/m_align_outlier_wq/`.
- **Structural finding regardless**: cross-orientation alignment map between activation outlier subspace and attention's most-shared rowspan. NO-GO (low intersection) is informative — it would mean outlier-routing and attention-routing are independent, which forces the Reach predictor to be more expensive but liberates the Unconventional storage layout to be optimized for one without compromising the other.

### Deferred (round 2 fodder, named here so they're not lost):

- **Untied lm_head spectrum (Qwen3-8B+)**: tests whether CF10's 0.97 tied-embed result generalizes when the constraint is removed.
- **W_V / W_O spectra**: cheap follow-up to AQFKV; needed to extend joint Q* claim beyond W_Q.
- **Cross-layer Q basis sharing**: do layers 0..27 share principal directions in residual coordinates? Underlies First-Principles + Composition persistence.
- **Per-token weight-contribution support (R2 → weight side)**: extends activation-side outlier dynamicity (Jaccard 0.72 → 0.31) to which weight rows actually contributed. Reach E1 dependency.

---

## What's fired off this round

In execution order. PRE-1 and PRE-2 can run in parallel (different resources: NVMe vs CPU/RAM). PRE-3 depends on PRE-2 finishing (loads the same model state).

### Step 1 (parallel): PRE-1 (M-NVMe) + PRE-2 (M-Strat / RSIDC)

```
scripts/m_nvme_pagefault.py
  --device C:
  --working-set-gb 8
  --queue-depths 1,4,16,64
  --pattern random,extent-sequential
  --duration-s 600
  --out experiments/stage0/ladder/m_nvme_pagefault/round1.json

scripts/m_strat_rsidc.py
  --model Qwen3-4B-IQ4_XS
  --layers 0..27
  --strata cf10-classes
  --estimator twonn,mle,levina-bickel
  --tokens 50000
  --out experiments/stage0/ladder/m_strat_rsidc/round1.json
```

Decision points:
- **M-NVMe**: if median 4 KB random read at QD=1 is <50 μs → E0a optimistic branch lives; >300 μs → E0b realistic branch is forced for both Reach and Unconventional.
- **M-Strat**: if median d*/d_model > 0.95 across all CF10 strata → First-Principles geometric-fingerprint claim is dead, Rank 1 demoted, save the 12.5 hours of E2/E3/E4. If median d*/d_model < 0.7 in any CF10-coherent stratum → green-light E2/E3/E4.

### Step 2 (sequential, gated on PRE-2 data): PRE-3 (M-Align)

```
scripts/m_align_outlier_wq.py
  --model Qwen3-4B-IQ4_XS
  --outlier-pctile 1.0,0.5,0.1
  --layers 0..27
  --metric principal-angles,intersection-fraction,grassmann
  --out experiments/stage0/ladder/m_align_outlier_wq/round1.json
```

Decision points:
- intersection-fraction ≥ 0.5 AND median angle ≤ 35° → Composition v2 fully alive, M2 confirmed.
- intersection-fraction 0.3–0.5 → Composition v2 partial; outlier-routing remains useful but not "for free".
- intersection-fraction < 0.3 → Composition M2 dead, Composition demoted to engineering substrate (M3/M4 still useful), Rank 2 reassessed.

### Step 3 (gated on PRE-2 GO): First-Principles E2 (MLP-saturation inequality)

30 minutes. Tests whether the gradient-decay / NTK signal-propagation derivation predicts the observed CF8 ΔNLL pattern (W_gate K=50% rank +0.77; W_up +2.34). If the inequality fits within 0.2 nats → green-light E3 (joint Q* construction, 4 hrs).

### Step 4 (gated on E2 GO): First-Principles E3 → E4

E3 = joint Q* construction (4 hrs) → E4 = rotated tied lm_head settling experiment (8 hrs). E4 is the **binary settling experiment of the entire round**: predict ΔNLL << +1.0 at the agreed bit budget vs random Q's +19.96. Number falls out cleanly; no ambiguity.

### Round-1 budget

Total wall-clock if everything runs sequentially through E4: 1 + 0.5 + 4 + 0.5 + 4 + 8 = **18 hours**. Parallelizing PRE-1 with PRE-2 saves 1 hour. Conservative wall-clock budget: **~17 hours of compute over ~3 calendar days** with the M-NVMe and M-Strat going out in step 1.

### What's not fired this round

- Reach v2 full cascade — gated on M-NVMe result; revisit after PRE-1.
- Unconventional v2 PFOR/NEXTENT/ZIPFLOCK implementation — gated on M-NVMe result.
- Composition E2/E3/E4/E5 — gated on M-Align result.
- Untied lm_head, W_V/W_O, cross-layer Q sharing, per-token weight support — round 2.

---

## Single-paragraph summary for the orchestrator

The four refined proposals converge on three things: (1) outlier-channel identity as the universal per-token routing key, (2) NVMe page-fault latency as the load-bearing systems number that two cascades branch on, and (3) joint low-dimensional structure across "different" weight matrices through the residual stream. We fire three pre-program measurements first — M-NVMe (1 hr, gates Reach + Unconventional), M-Strat / RSIDC (30 min, gates First-Principles and produces the round-2 prior regardless of GO/NO-GO), and M-Align (4 hr, decisive for Composition with structural-map fallback on NO-GO). Then conditionally First-Principles E2 → E3 → E4 because E4 is a clean binary settler (rotated tied lm_head ΔNLL vs random Q's +19.96) that produces the highest-information structural finding of the round whether GO or NO-GO. Total budget ~17 hours compute. Top-ranked direction is First-Principles GeoFingerprint v2 by leverage; its honest reach is ~19% probability of full end-to-end success delivering 70B at 3.5–6 GiB residency / 1.2–2.5 s/token, but its NO-GO still produces an independently publishable cross-layer residual-stream intrinsic-dimension map.
