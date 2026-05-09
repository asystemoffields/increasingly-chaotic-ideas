# increasingly chaotic ideas

A two-stage generative ideation pipeline for hard ML research questions, with a real-world run on the Nocturnal One project (run 70B-class LLMs on a 7.28 GiB Ryzen laptop without retraining).

The pipeline produces audacious-but-grounded research proposals at increasing levels of theoretical reach, filtered for prior-art collisions and refined into concrete experiments.

## Design

Two pipelines, used in sequence.

### `sonnet_ladder/` — Sonnet pipeline (the narrower)

A 6-stage cascade run as a search algorithm over the solution space:

```
Stage 1 (Ideator) → Stage 2 (Judge) → Stage 3 (Deep Researcher)
   → Stage 4 (Skeptic Explorer) → Stage 5 (Selector) → Stage 6 (Red Team)
```

Each round picks ONE experiment to run, executes it, records the empirical structural finding, and feeds the result into the next round's ideators. The kill list grows monotonically; ideators read it before generating to avoid re-proposing exhausted ideas.

Two parallel tracks:
- **Track A**: arch-transposition (operator-structure changes via no-SGD procedures)
- **Track B**: compression (bytes-per-weight reduction)

Output: a cumulative `KILL_LIST.md` of dead lines, a `SUMMARY.md` of empirical structural findings, and a queue of `SELECTED.md` experiments with go/no-go thresholds.

The ladder's strength is **systematic narrowing** under prior-art pressure. Its weakness is that ideators stay close to the published ML compression vocabulary — Sonnet generates from a training distribution where the conventional moves dominate.

### `opus_pipeline/` — Opus pipeline (the wider)

Inhabits the open solution space the Sonnet ladder produced and pushes for genuinely audacious mechanisms. 4 stages:

```
Stage 1 (Diverse parallel generation) → Stage 2 (Sonnet prior-art filter)
   → Stage 3 (Opus refinement) → Stage 4 (Synthesis)
```

**Stage 1 spawns four Opus agents in parallel**, each with the same shared context (the empirical findings from the Sonnet ladder, hardware reference numbers, what's open) plus a different creative orientation:

- **Reach** — articulate the most ambitious useful outcome and a credible cascade
- **Composition** — find non-obvious mechanisms that emerge from interactions between findings
- **First-Principles** — derive what should work from the structure already measured
- **Unconventional Angles** — close-to-metal, OS sideways, format outward, physics inward, anywhere except the standard playbook

A universal "think big" directive across all four agents invites reach into adjacent intellectual territory: information theory, statistical mechanics, quantum information primitives, abstract math, hardware/OS reformulations, operations from outside ML. Hard constraint: grounded falsifiability — every mechanism must connect to a measured number from the findings or a primitive that exists on the actual stack.

**Stage 2** runs four Sonnet prior-art filters in parallel, one per Stage 1 proposal. Identifies pre-empted mechanisms, partial pre-emption, adjacent-but-distinct, and genuinely novel components. Flags CF9 risks (imported theory whose preconditions may not hold for the target object).

**Stage 3** is each Opus agent receiving its own prior-art report and refining or regenerating in the same orientation. Either reframes around the surviving novel territory or produces a fresh proposal in the same orientation that dodges pre-emption.

**Stage 4** is one synthesis Opus reading all four refined proposals: identifies convergences across orientations, constructs the dependency graph across cascades, scores top directions by leverage, writes committed experiment plans for the next experiments to run, flags fresh measurements that benefit multiple directions.

### Why this composition works

The Sonnet ladder narrows. The Opus pipeline widens. They compose because the empirical findings that the ladder produces — the ones that *aren't on arXiv yet* — become the load-bearing inputs for Opus pipeline's wider proposals. Those measurements are pre-emption-resistant by construction (they're not in any Sonnet's training distribution). Audacious proposals grounded in private empirical findings get the leverage of both depth (real numbers anchor them) and width (orientations push past the standard ML playbook).

## What's in this repo

```
opus_pipeline/
  STAGE2_PRIOR_ART_TEMPLATE.md      # template for Sonnet prior-art filter agents
  STAGE3_REFINEMENT_TEMPLATE.md     # template for Opus refinement agents
  STAGE4_SYNTHESIS_TEMPLATE.md      # template for synthesis Opus
  round1/
    stage1_<orientation>_*.md       # Stage 1 outputs from four parallel Opus agents
    stage2_<orientation>_priorart.md # Stage 2 prior-art filter reports
    stage3_<orientation>_*V2.md     # Stage 3 refined proposals
    stage4_opus_synthesis_round1.md # Stage 4 synthesis with experiment plan
    thread_*.md                     # Side investigative threads spawned during the run
    e0_nvme_pagefault_results.json  # M-NVMe empirical results
    m_*_run.log                     # raw run logs

sonnet_ladder/
  PIPELINE.md                       # the ladder's runbook + accumulated process insights
  SUMMARY.md                        # empirical structural findings (CF1..CF15)
  KILL_LIST.md                      # cumulative kill ledger
  SELECTED.md                       # queued + completed experiments with thresholds
  ROUND_*_TRANSCRIPT.md             # per-round narrative
  round*_*/                         # per-round outputs (results JSON, plots, summary.md)
  track_A_arch/                     # Track A (arch-transposition) round dirs
  track_B_compress/                 # Track B (compression) round dirs
  m_strat_rsidc/                    # M-Strat / RSIDC measurement
  m_strat_rsidc_v2/                 # stratified version (BOS-stripped + top-2 magnitude channel projection)

scripts/
  cross_layer_sim_probe.py    # Round 1 cross-layer dot-product predictor
  pdap_outlier_jaccard.py     # Round 2 K-dependent outlier dynamicity (R2/CF3)
  wgate_rank_sweep.py         # Round 3 W_gate SVD rank sweep
  wup_rank_sweep.py           # Round 4 W_up SVD rank sweep
  sdzc_gate_variance.py       # SwiGLU dead-zone gate-variance probe
  aqfkv_q_rank_sweep.py       # Round 3-A AQFKV — Q-Head SVD with KV preserved
  lhqd_lmhead_spectrum.py     # Round 6-B LHQD — lm_head SVD spectrum
  m_strat_rsidc.py            # M-Strat / RSIDC residual-stream intrinsic dimension
  m_strat_rsidc_v2_stratified.py  # M-Strat v2 (BOS-stripped + top-K magnitude channel projection)
  outlier_wq_alignment.py     # Composition E1 — outlier × W_Q rowspan principal angles
  nvme_pagefault_benchmark.py # M-NVMe / E0 — Windows 11 NTFS mmap page-fault latency
  fp_e2_mlp_saturation.py     # First-Principles E2 — MLP-saturation inequality
  fp_e3_joint_qstar.py        # First-Principles E3 — joint Q* construction (cross-module shared rotation)
  fp_e4_rotated_lmhead.py     # First-Principles E4 — rotated tied lm_head binary settling experiment
  aqfkv_vo_rank_sweep.py      # Round 4 W_V/W_O sweep + joint Q-K-V-O SVD diagnostic
  untied_lmhead_streaming_svd.py  # untied lm_head streaming SVD on Qwen3-8B+ class models
```

## Key empirical findings from the run on Nocturnal One

(See `sonnet_ladder/SUMMARY.md` for the full numbered ledger of structural findings.)

### Confirmed empirical breakthroughs

- **CF13 / M-NVMe — NTFS Cache Manager extent prefetch confirmed (2026-05-08)**: single-thread random 4 KB read on Windows 11 NTFS mmap = **137μs median** (24× the optimistic 5.6μs assumption). BUT same-extent followon at **median 1.9μs** = **72× speedup**. The Cache Manager's read-ahead heuristic engages on mmap; after the first fault on a 1 MB extent, the rest of the extent is in standby cache for ~free. This validates the Unconventional v2 NEXTENT mechanism (NTFS extent prefetch as routing primitive) on Windows 11 — a real OS substrate that cooperates with information-theoretically optimal weight layouts.

- **CF15 / RSIDC v2 — stratified residual-stream intrinsic dimension (2026-05-08)**: when massive activations are stripped (top-2 magnitude channels per layer + first 4 BOS/attention-sink positions excluded), the typical-token residual stream lives in **~223 dimensions out of 2048 (k_99/d ≈ 0.11)** in middle layers. This is *much stronger* stratification than First-Principles' original 1400-1700 prediction. Independently: top-2 magnitude channels are **CONSISTENT across layers 2-27 at indices [1999, 1793]** (layer 0 differs at [784, 1371]) — the L1-resident "highway" of static persistent channels really is two specific channels that persist across nearly all layers. This empirically validates Composition v2's M5 highway claim AND supports First-Principles' geometric-fingerprint hypothesis.

### Theoretically grounded conjectures with clean falsification paths (not yet measured)

- **HoBRA (Holographic Boundary Reconstruction Attention)**: AQFKV's 16:1 head-redundancy IS the bond dimension χ ≈ 128 of a tensor network with low cross-head bond and full per-head local Hilbert. MERA-style: bulk Gram tensor G[t,t'] computed once per layer; per-head softmax on reconstructed attention. Predicted ~2× KV reduction + 1.8× attention wall-clock at ΔNLL ≤+0.10. (`opus_pipeline/round1/thread_hobra_attention.md`)

- **AERO-Tucker SwiGLU folding**: CF8 (weight-matrix SVD full-rank) does NOT kill 3-mode tensor rank, which is bounded by activation-subspace dimension not weight rank. Calibration-only Tucker-3 decomposition of the bilinear tensor T = W_down·((W̃_gate·x)⊙(W_up·x)). 4× MLP compression target at <0.5 nat. (`opus_pipeline/round1/thread_aero_swiglu.md`)

- **Joint Q\* dissolving lm_head catastrophe**: cross-module shared orthogonal rotation Q* simultaneously low-ranks W_gate, W_up, AND tied W_E. Predicted ΔNLL ≪ +1.0 at r=1024 vs random Q's +19.96 (the established catastrophe baseline). RMSNorm precondition resolved via paired-rotation fusion + γ absorption. (`opus_pipeline/round1/stage3_firstprinciples_GeoFingerprintV2.md`)

- **PFOR via λ−1 hypergraph partitioning**: optimal weight layout at 1 MB extent granularity is the VLSI placement problem with the connectivity-1 objective (NOT spectral cut, NOT Cuthill-McKee bandwidth). KaHyPar/hMETIS are 30-year-mature solvers. Predicted 250→50 unique extents/token = 27 ms/token saved on 110 ms compute = 5.7→8.2 tok/s on 70B. (`opus_pipeline/round1/thread_pfor_algorithm_space.md`)

## Re-running the pipeline on a different project

The pipeline is project-shaped but not project-specific. To run it on a different research problem:

1. Replace the empirical findings in `sonnet_ladder/SUMMARY.md` with your project's measured structural findings.
2. Replace the goal / hardware-reference / constraint preamble in the Stage 1 brief with your project's.
3. Run the Sonnet ladder for 3-6 rounds first to populate `KILL_LIST.md` with idea-class deaths and accumulate empirical structural findings.
4. Run the Opus pipeline once those findings are dense enough to anchor proposals (typically after CF8-class compound findings emerge).
5. Use the Stage 2/3/4 templates as drop-in components.

## Status

Round 1 of the Opus pipeline complete. Top-3 directions queued for execution. Pre-program measurements (M-NVMe, M-Strat v2) completed. Composition E1 (outlier × W_Q alignment) and First-Principles E3-E4 (joint Q* + rotated lm_head settling) are the next high-information experiments.
