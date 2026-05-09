# v2 Round Runbook

(This file is also referenced internally as `ROUND_OVER_ROUND.md` in some prompt files; both names point here.)

How to run a single round of the v2 ladder. The 6-stage cascade structure is unchanged from v1 (Ideator → Judge → Deep Researcher → Skeptic Explorer → Selector → Red Team); the structural change is at Stage 1. Stages 2-6 are Agent B's territory and are gestured at here, not specified.

---

## Stage 1 — Spawn K parallel orientation-bound ideators

**Default K = 5**, one per orientation in `ORIENTATIONS.md` (R, C, F, U, A). Some rounds drop or replace an orientation per the rotation rule below; in those rounds K may be 4 or K may be 5 with one slot replaced by a custom round-specific orientation.

Each parallel ideator receives a self-contained brief:

1. The full text of `prompts/STAGE1_IDEATOR.md`.
2. The orientation block from `prompts/ORIENTATIONS.md` for its assigned orientation only (NOT the full registry — this prevents leakage between orientations during generation).
3. The full text of `SUMMARY.md` (CF1-CF12 confirmed + CF13-CF15 in-flight reference; the ideator template specifies how each tier may be used).
4. The full text of `KILL_LIST.md` (cumulative across v1 + v2 rounds; v1 list is read-only reference, v2 list grows monotonically).
5. The round-specific framing-diversity targets paragraph (assembled by the runner from the previous round's transcript — names which idea-class taxonomies are over-represented and which are under-explored).
6. The saturated-frame list for the round (named flavors that hit the saturation boundary in recent rounds).
7. Hardware reference numbers (carried in SUMMARY.md header).

Each ideator runs in its own Sonnet conversation, with no cross-talk. The K ideators run concurrently.

Each ideator produces 4-8 candidate ideas in a single file at `round{N}/stage1_{R|C|F|U|A}.md`. Each idea inside the file uses the per-idea structure from `STAGE1_IDEATOR.md` (Name / Mechanism / Residency arithmetic / Novelty gloss / Smallest experiment / Primary risk) and is delimited by its Name as a level-2 header. Each idea self-classifies Track A or Track B at the start of its Mechanism field.

**Spawn command shape (illustrative):**

```
for orientation in active_orientations(round=N):    # typically [R, C, F, U, A]
    spawn_sonnet_agent(
        system_prompt = STAGE1_IDEATOR.md
                      + ORIENTATIONS.md[orientation]
                      + SUMMARY.md
                      + KILL_LIST.md
                      + framing_diversity_targets(round=N)
                      + saturated_frames(round=N),
        user_prompt = "Generate 4-8 candidate ideas in your assigned orientation.",
        write_to    = f"round{N}/stage1_{orientation}.md"
    )
parallel_wait_all()
```

Total Stage 1 cost: ~5 Sonnet conversations × ~3500 tokens output each = ~17k tokens output. Compares with v1 single-stream Stage 1 ~5k tokens. The ~3-4× cost is the price of the structural change; the user's observation that the parallel-orientation pattern produces measurably more interesting proposals is what justifies it.

---

## Stage 1 → Stage 2 handoff

The round runner bundles the K orientation files (4-8 ideas per file × ~5 orientations = 20-40 ideas) into a single Stage 2 input. Bundle format:

```
round{N}/stage1_bundle.md
  ## Convergence-handle index
  - "outlier-channel identity as routing key": ideas C2, R3, U1
  - "Windows mmap page-fault latency under prefetch": U2, R4
  - ...
  ## Ideas, grouped by orientation then track
  ### Orientation R (Reach)
  - Idea R1 [Track A] — Name, Mechanism, Residency, Novelty, Smallest exp, Primary risk
  - Idea R2 [Track B] — ...
  ### Orientation C (Composition)
  - ...
```

The convergence-handle index is computed by the runner via text intersection over each file's "Convergence handles" tail list. This is cheap (string match) and is a free input for Stage 2's per-idea scoring rubric (Stage 2 may explicitly favor ideas appearing in multi-orientation handles).

The Stage 2 Judge reads this bundle and advances 6-10 ideas across the entire union (NOT top-K-per-orientation). Stage 2-6 details are Agent B's design.

**What Agent B should know about the Stage 1 → Stage 2 contract:**

1. **Volume**: Stage 2 receives 20-40 ideas total, not 8-12. Judge throughput must scale.
2. **Depth-per-idea**: each idea is the six-field structure (Name / Mechanism / Residency arithmetic / Novelty gloss / Smallest experiment / Primary risk), 150-400 words per idea. This is more compact than the Opus pipeline's 600-1000-word Stage 1 proposals — the depth is in the residency arithmetic and the smallest-experiment specificity, not in cascade-length narration. Judge can score on substance.
3. **Convergence-handle index**: pre-computed by the runner. Agent B's Stage 2 may use it as a free input; Agent B's Stage 4-equivalent will use it heavily for synthesis (per opus_pipeline R1 Stage 4 convergence map pattern).
4. **Track self-classification**: every idea is pre-tagged Track A or Track B at the top of Mechanism. Stage 2 can route directly to the appropriate track-shared kill list and prior-art lens.
5. **CF13/14/15 gating**: any idea citing CF13/14/15 has, per the Stage 1 ideator contract, included a re-derivation step in its Smallest Experiment. Agent B's Stage 5 / Stage 6 should treat re-derivation passing as a hard precondition for cascade GO; if the re-derivation fails, the idea downgrades whether or not the rest of its mechanism is sound.
6. **Audacity gradient**: orientation R ideas stack more independent novel claims than orientation U ideas. Stage 2 judges should expect lower P(end-to-end success) on R ideas; the value of an R idea is the cascade itself, not the destination. Stage 6 red-team should be especially careful not to kill R ideas on grounds of "the stack is too tall" — that is the orientation working as intended.

---

## Round 2+ — orientation rotation rule

After Round N closes (Stage 6 complete, KILL_LIST and SELECTED updated), the runner decides which orientations carry into Round N+1. The slot count is fixed at 5; orientations rotate within the slots.

**Rotation rule:**

1. **An orientation whose proposal advanced to Stage 5 selection in Round N**: KEEP for Round N+1. Track-record-weighted carry.
2. **An orientation whose ideas all died at Stage 2 (Judge scored ≤ 11/15) in Round N**: this is the saturation signal at orientation granularity. The orientation's anti-frame list is now too lax (the ideator is generating saturated mechanisms within it). Two options:
   - DROP the orientation for one round; replace with a Constraint-Alien run on a fresh constraint from the menu.
   - Or: KEEP the orientation but replace its anti-frame list with a stricter version derived from the Stage 2 kill reasons (the runner appends to the orientation's anti-frame: "do not propose anything resembling [killed flavor]").
3. **Two orientations whose Round N ideas converged on the same convergence handle**: KEEP both. The convergence is the round's signal per opus_pipeline R1 Stage 4. Orientation diversity is preserved because they reached the same handle from different starting points.
4. **An orientation that produced ONLY ideas on the kill list (Stage 1 self-recognized the kills, generated nothing fresh)**: DROP for one round. Replace with a custom-defined orientation derived from the Round N Stage 4 (Skeptic Explorer) transcript — specifically, the named under-explored frames the skeptic surfaced. Custom orientations are documented in the round transcript and may be promoted to the registry if they produce a Stage 5 selection.
5. **Orientation A (Constraint-Alien) constraint rotation**: every round, A picks a different constraint from the menu in `ORIENTATIONS.md` (or a constraint named in the previous Stage 4 skeptic transcript). The constraint cycles even if A keeps producing GO results — this is by design, because each constraint is a different lens and the value of A is in the lens-rotation rather than in any one constraint.

**Weighted spawn (optional, for round 4+):** if specific orientations have a clear track record (e.g., F has produced 2/3 GO-trending selections), the runner may spawn 2 ideators on F (independent random seeds) and 0 on a dropped orientation. Cap: at most 2 ideators per orientation per round, to preserve cross-orientation diversity.

---

## Stage 2-6 — gesture at the handoff

These are Agent B's territory. The contract from Stage 1 is described above. A few process gates the v1 PIPELINE.md flagged that v2 Stage 2-6 should preserve and adapt:

- **Theoretical frame mismatch check (CF9)**: Stage 2/3 must verify imported-theorem preconditions. v2 Stage 1 proposals are deeper, so frame mismatches will be more elaborate. Agent B should expect to spend more Stage 3 time on this.
- **Calibration ill-conditioning check (CF10)**: Stage 5/6 must check n_params vs n_samples. v2 proposals from Reach and Composition orientations may stack multiple calibration fits; Agent B's Stage 5 should sum the parameter counts across stages, not just check each stage in isolation.
- **Track A publication-velocity policy**: re-check novelty at every stage; date-stamp claims; prefer structural/algebraic novelty over specific-architecture imitation; favor pre-empt-resistant ideas. v2 orientation U is most exposed to publication-velocity risk; orientation A is most exposed to "novelty without traction" risk; Agent B's Stage 2 prior-art filter should weight accordingly.
- **Saturation as signal**: when Stage 2 mass-pre-empts a round, that's the signal. Round N+1 rotates orientations and tightens anti-frames per the rule above.
- **Convergence-as-signal**: when two orientations produce the same convergence handle, that is independently confirming the structure. Agent B's Stage 4-equivalent should compute this explicitly.

---

## v2-specific stop conditions

Stop the v2 ladder when one of these holds:

1. **Stage 1 saturation across orientations**: in Round N, ≥3 of the K active orientations produce Stage 1 files where >50% of ideas are on the kill list or in killed flavors (each orientation is independently saturating). This is a stronger version of v1's "≥3 ideas already on kill list" stop condition because the v2 spread across orientations means saturation in one orientation no longer triggers the stop alone.

2. **Stage 2 score ceiling**: in Round N, the highest-scoring idea from any orientation scores ≤ 12/15. (Slightly stricter than v1's ≤ 11/15 because v2 ideas have residency arithmetic and a smallest-experiment field — a v2 idea scoring 11 has had fair chance.)

3. **Stage 5 mechanistic-overlap recurrence**: Stage 5 of Round N selects an experiment with ≥80% mechanistic overlap to a prior round's selection. Carried forward unchanged from v1.

4. **Convergence collapse**: Round N's convergence handles are >70% the same as Round N-1's (the orientations are converging on a single handle round over round, which means cross-orientation diversity has broken down). Rotate orientations aggressively or stop.

5. **Empirical exhaustion**: SUMMARY.md has not gained a CF_i in 3 rounds AND the kill list has not gained a class-level kill in 3 rounds. The map is no longer being built; stop and reformulate.

6. **User manual stop**: the SELECTED experiment looks compelling enough to commit to.

---

## Per-round directory layout

```
sonnet_ladder_v2/
  SUMMARY.md
  KILL_LIST.md
  SELECTED.md
  prompts/
    STAGE1_IDEATOR.md
    ORIENTATIONS.md
    ROUND_RUNBOOK.md
  round{N}/
    stage1_R.md            (one file per ideator, 4-8 ideas inside)
    stage1_C.md
    stage1_F.md
    stage1_U.md
    stage1_A.md
    stage1_bundle.md       (assembled by runner; includes convergence-handle index)
    stage2_*.md            (Agent B)
    stage3_*.md
    stage4_*.md
    stage5_*.md
    stage6_*.md
    ROUND_{N}_TRANSCRIPT.md
    framing_diversity_targets.md  (assembled by runner from round N-1)
```

---

## When the runner returns to round N+1

1. Read `SUMMARY.md` for the current empirical state.
2. Read `KILL_LIST.md` (bottom for newest entries).
3. Read `round{N}/ROUND_{N}_TRANSCRIPT.md` for the previous round's outcome.
4. Apply the orientation rotation rule above.
5. Assemble Round N+1's framing-diversity targets paragraph.
6. Spawn the 5 (or weighted-spawn) parallel Stage 1 ideators.
7. Stage 2-6 proceeds per Agent B's design.
