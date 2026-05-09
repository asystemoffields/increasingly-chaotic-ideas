# v2 Ladder — 50-run Meta Status

Each row is one independent end-to-end ladder run (Stage 1 → Stage 6). Runs share the *initial* state (`KILL_LIST.md`, `SUMMARY.md`) but do not share kills or selections with each other — each run is an independent sample of the v2 cascade.

Started: 2026-05-09
Driver: Claude Code Agent-tool orchestration (no API key)
Per-run cost: 5 parallel Stage 1 agents + Stages 2-6 sequential ≈ 10 agent calls
Output location: `runs/run_NNN/`

## Status table

| Run | Stage 1 (R/C/F/U/A) | S2 | S3 | S4 | S5 (A/B) | S6 (A/B) | Track A pick | Track B pick | Notes |
|-----|---|---|---|---|---|---|---|---|---|
| 001 | 7/7/7/6/7=34 | 16 adv, 5 conv, 3 rej | 16 REFINE, 1 DOWN, 0 KILL | 8 new | F-CQSGC / R-RAOK-70B | A: AMEND / B: AMEND | F-CQSGC (cross-layer W_Q SVD) | R-RAOK-70B (3-tier act codebook) | Cheap test KILLED Cluster C1 (gap=0.0001 vs permuted) — class-level kill recorded as v2-CHEAP-TEST-001 |
| 002 | 7/6/6/7/7=33 | 18 adv, 3 conv, 6 rej | 26 REFINE, 0 KILL | 8 new | R3-XLQB / F2-CQBS | A: AMEND / B: AMEND | R3-XLQB (cross-layer W_Q stacked SVD) | F2-CQBS (cross-layer W_Q basis sharing) | Both picks Cluster C1 (independently re-converged) — empirically dead per run 001 cheap test; Stage 6 caught Basis Sharing prior-art Stage 3 missed |
| 003 | 7/6/7/7/6=33 | many adv, 4 conv | 26 REFINE, 0 KILL | 8 new | F3-SRSC→F5-HPGO / F1-VOFR | A: REJECT-runner-up / B: AMEND | F5-HPGO (head-perm gauge) | F1-VOFR (W_V/W_O fused) | Track A F3-SRSC rejected on CF9 (RoPE breaks softmax shift-inv); F5-HPGO advanced |
| 004 | 7/6/6/6/7=32 | many adv, 5 conv | 25 REFINE, 1 DOWN (F-GFQO killed) | 8 new | F-WVOS / F-WNORM | A: AMEND / B: AMEND | F-WVOS (W_V/W_O algebraic fold) | F-WNORM (W_down rank sweep) | Stage 6 elevated A3 (2505.12942 v3) — pre-run mandatory check |
| 005 | 6/6/6/6/7=31 | 18 adv | 18 REFINE, 4 DOWN (PALU/TransMLA pre-empt WVWK-MLA) | 8 new | A1-GFRS-2 / S6-SEQFILT | A: AMEND / B: AMEND | A1-GFRS-2 (Z_2^d sign-entropy) | S6-SEQFILT (sequential RAOK filter) | Stage 6 retagged Track B elegance from algebraic-identity to conserved-quantity |
| 006 | — | — | — | — | — | — | — | — | starting |
| 007 | — | — | — | — | — | — | — | — | starting |
| 008 | — | — | — | — | — | — | — | — | starting |
| 009 | — | — | — | — | — | — | — | — | starting |

(Runs 003–050 elided until they start.)

## Aggregator (after all 50 done)

`META_AGGREGATE.md` will surface:
- Cross-run convergences (mechanisms appearing in ≥3 runs)
- Standout single-run finds (e.g., a `[FREE SWING]` that survives Stage 6)
- Elegant-equivalence-class selections
- NO-GO selections that produce structural findings
- Auto-fired cheap tests and their results
