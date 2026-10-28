# Status to Raj — the recalculation and the appendix (28 Oct 2026)

**Short version.** The ratified rules are encoded, the full recalculation has
run on the real data, the paper figures now come from that run, and every
robustness variant is in the appendix. The main story holds; the one place
the numbers differ materially from the published version is the adjacency
rule, and that difference is the ratified change itself.

## What ran

- **The ratified profile** (`ratified-2026`): border-sharing adjacency, a
  partially blocked boundary discounted rather than severed (5 m buffer,
  kept as a setting), overlap lending outside the receiver, own-roads-only,
  planned-relative scale (average planned colony = 1), and the excluded
  types RV / Industrial / Other.
- **Inputs unchanged**: 4,357 settlements; 15 dropped for missing
  population; **4,062 scored**. The only row-count change from the old run
  is the 69 Industrial/Other settlements now excluded (RV was already out).
- Outputs: `~/delhi_data/phase4_ratified/`, with dated copies and a manifest
  in `delivery_2026-10-27/`. Figures in `docs/figures/ratified-2026/`.

## Headline, planned-relative (average planned colony = 1)

| | popdensity | population |
|---|---|---|
| JJC | 0.06 | 1.78 |
| SDA | 0.52 | 1.23 |
| UAC | 0.53 | 1.06 |
| Planned | 1.00 | 1.00 |
| RUAC | 0.92 | 0.72 |
| UV | 1.27 | 0.85 |
| JJR | 1.18 | 0.40 |

JJC sits high on population and near zero on density. That is a property of
the two denominators, not of the new scale — it moved the same way before.

## The appendix (one line each)

| variant | verdict |
|---|---|
| Distance bands (1/5/10 km) | ordering stable (ρ 0.795 / 0.738 / 0.738); the 10 km point is a city-wide average and belongs in the appendix |
| Decay forms (off, power, exponential, boundary) | not load-bearing (ρ 0.989–0.999) — keeping 1/(1+d) is supported |
| Adjacency: bbox (the published rule) | **the one real difference**: ρ 0.865, 60 % top-decile overlap, top two types swap |
| Lane-weighted roads | ordering identical; every mean moves ≤ 0.03 |
| Media reclassification | ordering identical; only UV's own level moves |
| Service subsets (ration, banking, core) | shipped earlier; reported in the effects tables |

**The honest sentence for the paper:** the only robustness variant that
changes the ordering is the adjacency rule we deliberately changed. Nothing
else moves the story, and the appendix says so rather than hiding it.

## Five things I need from you

1. **Licence** — brief written (`docs/licence-options.md`); recommendation:
   MIT for code, CC-BY-4.0 for data/figures. This blocks the public release
   and the SSRN posting.
2. **Category collapse** — keep the ten-type identity mapping (current), or
   collapse to five? Parked since 28 Aug.
3. **Per-service distance expectations** — do you have numbers or a source
   for "a school may be farther than water"? Without one I will not invent
   the scales (`docs/superpowers/specs/2026-10-28-del-38-per-service-decay-note.md`).
4. **Release scope** — may the derived per-settlement outputs be published,
   or only the aggregate tables and figures?
5. **5 m barrier buffer** — confirm it, so it stops being a provisional
   value in the profile.

Nothing else is waiting on you.
