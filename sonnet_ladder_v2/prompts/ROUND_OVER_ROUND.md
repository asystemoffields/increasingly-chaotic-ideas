# Round-over-Round — Orientation Evolution

How the orientation set evolves across rounds in the Sonnet Ladder v2.

The orientation set in `ORIENTATIONS.md` is the default starting set: R, C, F, U, A. The round runner is permitted to keep / drop / tighten / introduce orientations between rounds based on the previous round's outcomes. The mechanism preserves diversity (orientation count must remain ≥3) while letting the pipeline learn which orientations are productive on this empirical substrate.

## Round 1 — all orientations fire

The first round of the v2 ladder spawns one Sonnet ideator per orientation in `ORIENTATIONS.md`. All five orientations are active. Anti-frame lists begin at the static lists in `ORIENTATIONS.md`. There is no round-specific orientation override at Round 1; the menu is the menu.

Round 1's job is also to populate the per-orientation track records. Stage 5 selections from Round 1 are tagged by the orientation that originated the surviving idea, so that Round 2's runner has the data to apply the rotation rule.

## Round N+1 rotation rule

After Round N closes (Stage 6 has produced the round transcript and the GO/NO-GO of Round N's selected experiment is known), the round runner applies the following rule to determine Round N+1's active orientation set:

**KEEP** an orientation if either:
- A Stage 5 selection that originated in this orientation produced a **GO** result in Round N, OR
- The orientation's Stage 1 output appeared in a **convergence cluster** with another orientation's output at Stage 4 (per the convergence-handles mechanism — `STAGE1_IDEATOR.md`'s "Convergence handles" section). Convergence is itself signal; both orientations stay.

**TIGHTEN** an orientation if it produced a Stage 5 selection that ran and was **NO-GO**, or if its Stage 1 output advanced to Stage 3 but was killed there. Tightening means: the round runner appends to the orientation's anti-frame list the specific named flavor that was killed. Anti-frame lists grow round-over-round; they do not shrink. (This mirrors the kill list — both are monotonic-knowledge structures.) The orientation stays active in Round N+1 with the tightened anti-frame.

**DROP for one round** an orientation if **all** of its Round N proposals died at Stage 2 (judged as kill-list re-treads or saturated frames). This is the saturation signal from `sonnet_ladder/PIPELINE.md` — the orientation's natural moves have been exhausted on the current substrate. Dropping for one round opens the slot for a Constraint-Alien run on a constraint not yet tried, OR for a custom round-specific orientation defined for a frame the previous round's transcript flagged as under-explored. After one round of dropping, the orientation can be re-armed for Round N+2 with whatever new anti-frames the kill list has accumulated.

**INTRODUCE** a new orientation when the previous round's Stage 4 transcript explicitly identifies a frame as under-explored (the v1 ladder's PIPELINE.md "framing-diversity targets" mechanism). The new orientation is defined for that round only and gets a custom orientation block (creative directive + anti-frame list + cross-domain seed list + falsifiability requirement, same fields as `ORIENTATIONS.md`). If the round-specific orientation produces GO results, it can be promoted to a permanent slot in `ORIENTATIONS.md` (replacing a dropped orientation) — this is how the orientation set itself evolves.

## Diversity floor

**The active orientation set must remain ≥3 in every round.** This is the diversity floor — below 3, the cross-orientation parallelism that v2 imports from the Opus pipeline collapses back into the v1 single-stream behavior. If the rotation rule would reduce the set below 3, the round runner overrides by re-arming the most-recently-dropped orientation early.

A practical consequence: any single round can drop at most 2 of the 5 default orientations. If 3 orientations are simultaneously eligible for "drop for one round," the runner picks the 2 with the weakest Stage 1 output by Stage 2's union-rank and keeps the third active with a tightened anti-frame.

## Anti-frame list growth

Anti-frame lists are the v2 ladder's primary saturation-resistance mechanism within an orientation. They grow round-over-round on three triggers:

1. **NO-GO at Stage 5**: the named flavor of the killed mechanism is appended to the originating orientation's anti-frame list.
2. **Stage 3 kill**: same — the flavor is appended.
3. **Mass pre-emption at Stage 2**: if multiple ideas in the same orientation are killed for the same prior-art collision, the *common frame* of those ideas is appended (per PIPELINE.md "anti-frame lists, not just kill lists").

The cumulative kill list (`KILL_LIST.md`) is shared across all orientations; the anti-frame list is per-orientation. They are complementary: the kill list blocks specific instances, the anti-frame list blocks flavors.

## Orientation set evolution example (illustrative)

A possible trajectory:
- **Round 1**: R, C, F, U, A all active. Stage 5 selects from Composition (GO at GO threshold) and from First-Principles (NO-GO).
- **Round 2**: R, C, F, U, A all active. F's anti-frame list tightens (the killed flavor is appended). A new convergence between U and Reach at Stage 4 keeps both. Stage 5 selects from Unconventional (NO-GO; structural finding). Constraint-Alien produced 4 proposals all killed at Stage 2 as saturated.
- **Round 3**: R, C, F, U active (4 orientations). A is dropped for this round. The slot is filled with a custom round-3-only orientation, "Information-Theoretic Bounds" (because Round 2 Stage 4 transcript flagged that as under-explored). U's anti-frame list tightens. Five active orientations again, with one custom slot.
- **Round 4**: A is re-armed with whatever new anti-frame items its dropped round added. The Round 3 custom "Information-Theoretic Bounds" orientation is evaluated: if it produced GO results, it is promoted to a permanent slot, replacing whichever default orientation has the longest current dry streak. If not, it retires.

The orientation set is a small evolutionary system: the rotation rule is selection pressure; anti-frame growth is mutation pressure; cross-orientation convergence is the fitness signal. The diversity floor is the stop-gap that prevents collapse.

## Stop conditions inherited from v1

The four PIPELINE.md stop conditions still apply at the round level:
1. Stage 1 of Round N proposes ≥3 ideas already on the kill list (search saturated within Sonnet's prior).
2. Stage 2 scores all of Round N's ideas ≤ 11/15 (no idea strong enough).
3. Stage 5 of Round N selects an experiment ≥80% mechanistically overlapping a prior-round selection.
4. User manually stops.

In the v2 ladder, condition 1 is more aggressive (5 orientations × 4–8 ideas each = 20–40 candidate ideas; if ≥3 of those overlap the kill list, the saturation signal is louder than in v1). The runner should treat this as the moment to introduce a new orientation rather than declaring saturation; if the new orientation also fails to escape, the run actually saturates and the loop should stop or escalate to a different model family per PIPELINE.md "Diversity / bias-handling."
