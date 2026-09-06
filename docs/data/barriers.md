# The barrier layers: an inventory

Raj asked on 28 Aug 2026 where the barrier layers came from — city data, or
drawn by hand (decision log
`docs/decisions/2026-08-28-raj-methodology-decisions.md` § 4, DEL-51). What
this repository can establish mechanically is an INVENTORY; it is below,
produced by `scripts/inventory_barriers.py`, which reads the layers named by
the `code-2025` profile, flags settlements through the pipeline's own
`geometry.barrier_flags` + `neighbors.combine_barrier_flags`, and writes
nothing under the data directory. `tests/test_inventory_barriers.py` re-runs
it and compares the blocks (it skips when the data is not present).

Numbers quoted in prose below in `backticks` are block values verbatim;
percentages and other derived quantities are written with a `%` sign or
without backticks.

## What this tells us about provenance

**No claim about an agency is made here.** The evidence, and what it
suggests:

- **The attribute schemas are an official GIS office's, not a hand
  digitiser's.** The canal layer carries `CAN_NM` (canal name), `CAN_CLSF`
  (classification), `EL_GND` (ground elevation) and `DIST_NM`; the railway
  layer carries `RL_ZONE`, whose values name a railway zone; the drain layer
  carries `Drain_type`, `Drain_Name`, `MAINTAINED`, `AC_NAME` (assembly
  constituency) and `DISTRICT`. Ground elevations, maintenance
  responsibility and assembly constituencies are fields a utility or survey
  department keeps; nobody tracing lines over a basemap invents them. The
  `attributes` block below carries the schemas and the distinct values, so
  this paragraph can be checked rather than believed.
- **The clipped files were written by ArcGIS on 2 Aug 2020.** `Canal.shp.xml`
  and `Railway_Line.shp.xml` carry ESRI `<CreaDate>`/`<CreaTime>` elements
  and a `lineage` of `RepairGeometry` runs from ArcGIS Pro; the `layers`
  block reports the dates. A `.qpj` sidecar beside each layer says QGIS also
  opened them. `Major_Drain.shp` has no `.shp.xml`.
- **The paper's "manually marked" sentence describes the CLIPPING, not the
  digitising.** The April 2026 draft (pp. 15–16) says the team "manually
  marked areas that had river or railroad tracks". The directory name is
  `Barrier_Clip/`, and the extra copies at the data root
  (`canal.data/`, `railway.data/`, `drain.data/`) are the unclipped
  originals — so what was done by hand is plausibly the selection and
  clipping of an existing layer to the study area.

**The question that remains, for Bijoy (batched reply, item 7):** which
agency's layer are these — and what exactly did "manually marked" cover:
selecting features, clipping to the study area, or drawing lines?

- **Run date:** _pending — the run step fills this in_
- **Inputs:** _pending_
- **Commit:** _pending_
- **Command:** `uv run python scripts/inventory_barriers.py --config code-2025 --all-candidates --work-dir <work-dir>`

*(The values above and the two fenced blocks are inserted by the run step,
which is when those facts exist.)*
