# Stage 4 — Opus Synthesis (single agent reads all four refined proposals)

## Brief

You are the synthesis agent for the Opus generative pipeline. Four refined research programs from parallel Opus agents (Reach, Composition, First-Principles, Unconventional Angles) follow. Each was filtered through Sonnet prior-art and refined or regenerated. Your job is integration.

[FOUR STAGE 3 PROPOSALS INSERTED]

## Tasks

1. **Find shared substrate.** Where do two or more proposals converge on the same underlying mechanism, even when they describe it differently? Convergences are signal — they're the mechanism the model's empirical structure is independently pointing four creative orientations at.

2. **Find composition opportunities.** Where does proposal A depend on a measurement that proposal B's experiment cascade would produce? Those are dependencies that let us run B's cheap experiments first and unlock A. Identify the dependency graph across all four cascades.

3. **Score by leverage.** For each distinct direction, estimate:
   - Ambition ceiling (what does success deliver in tok/s and residency on the 70B target?)
   - Probability of producing a structural finding (independent of mechanism success)
   - Inverse experiment cost (1 / wall-clock hours for the first decisive experiment)
   - Pre-emption resistance (per Stage 2 report)
   
   Pick top 2-3 directions.

4. **Write committed experiment plans.** For the top 2-3 directions, produce the next concrete experiment with:
   - Go/no-go thresholds (numerical)
   - Runtime estimate on the Ryzen 5 7530U
   - The structural finding it would produce regardless of direction (so a NO-GO is still load-bearing)
   - Script template name (e.g., `scripts/<name>.py`)
   - Output path under `experiments/stage0/ladder/round*` or `experiments/stage0/opus_pipeline/round1/`

5. **Flag fresh measurements needed.** Which proposals depend on measurements we haven't taken yet? List as pre-program experiments — these are cheap and high-information regardless of which program survives:
   - RSIDC (residual stream intrinsic dimension)
   - Untied lm_head spectrum on Qwen3-8B+
   - W_V / W_O spectra
   - Cross-layer Q basis
   - Per-token weight-contribution support (extension of R2 to weight side)
   - [Add any additional measurements proposals require]

6. **Honest reach assessment.** For the top-scored direction, what does the ambition statement actually deliver if the experiment cascade succeeds? Be specific:
   - tokens/sec on Qwen3-4B (current 5.5)
   - tokens/sec on Qwen3-72B / Llama-3-70B target
   - residency in GiB
   - quality cost in nats vs full-precision baseline
   - probability assigned to the cascade succeeding end-to-end

## Output format

Markdown synthesis report (1500-3000 words):

```
# Opus Pipeline Round 1 — Synthesis

## Convergence map
[which mechanisms appear across multiple proposals]

## Dependency graph
[which experiments unlock which programs]

## Top 2-3 directions, ranked
For each:
- Name + 1-line description
- Source proposals (which Stage 1 agents proposed it)
- Ambition ceiling
- Pre-emption resistance per Stage 2
- First decisive experiment plan with go/no-go
- Honest reach assessment

## Pre-program measurements (run first regardless)
[list of cheap structural measurements all top directions benefit from]

## What's fired off this round
[the specific experiment(s) to run next, in order]
```

## Style

This is the integration step. Don't generate new ideas — synthesize the four. If you find yourself adding mechanisms not in any Stage 3 proposal, stop and re-read the proposals. Your job is to identify the joint signal across the four and pick what the pipeline should run next.

The pipeline values map-of-the-territory over single-mechanism wins. A direction that produces a strong structural finding even on NO-GO is preferred over a direction with higher ambition but no fallback signal.
