# Appendix: the robustness variants (DEL-44)

Every table below answers one question: **would a reviewer who disagrees
with a single methodological choice change the paper's story?** The main
claims are the category ordering and the size of the planned/JJC gap; each
variant moves one lever and reports rank agreement with the main run, never
a level difference (Eq. 2's min-max runs per run, so levels are not
comparable across runs).

| variant | what moves | against | result | table |
|---|---|---|---|---|
| distance bands | adjacency radius 1 / 5 / 10 km | touch (main) | ordering stable; ρ 0.795 / 0.738 / 0.738; top four hold at 5–10 km | `docs/data/distance_bands.md` |
| media reclassification | settlement type from the students' sheet (105 changes, 81 RV→UV) | ratified | ordering identical; only UV's own level moves on density | `docs/data/media_reclass.md` |
| decay form | off / inverse-power 0.5, 2 / exponential 2, 5 km / boundary | inverse-linear (main) | ρ 0.989–0.999; only the fragile SDA/RUAC pair swaps | `docs/data/decay_sweep.md` |
| adjacency rule | bbox (the published rule) / 0 km intersection | touch (main) | bbox ρ 0.865, 60 % top-decile overlap, top two swap; 0 km ρ 0.991, same order | `docs/data/adjacency_variants.md` |
| lane-weighted roads | road amount = length × lanes | length (main) | ordering identical; every mean moves ≤ 0.03 | `docs/data/roads_lane_weighted.md` |
| service subsets | without ration / without banking / core services | full set | shipped earlier as profiles (`services-*`); tables in `docs/data/rule_effects.md` and the DEL-40/41/42 records | `docs/data/rule_effects.md` |

**The one variant that matters is the published adjacency.** Everything
else leaves the ordering alone. bbox — the rule the July 2025 numbers used —
does not: it builds 50 % more links, isolates far fewer settlements, and
swaps the top two types. That is exactly the change the ratified profile
makes, and exactly why the recalculation was worth doing.

**Second-order:** the two appendices that shift levels without moving order
(media reclassification’s UV, lane weighting’s small across-the-board dip)
should carry the one-line note that the level is not the claim — the
ordering and the gap are.

Nothing here required a production-code change beyond the knobs already
shipped; the variants are profiles, exactly as the methodology-config
procedure intends.
