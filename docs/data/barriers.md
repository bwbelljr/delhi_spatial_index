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

## Measured on 2026-09-05

- **Run date:** 2026-09-05
- **Inputs:** `~/delhi_data/Barrier_Clip/{Canal/Canal.shp, Railway/Railway_Line.shp, Drain/Major_Drain.shp}` (the profile's layers) plus, under `--all-candidates`, `Barrier_Clip/Canal/new/checked_Canal.shp` and the root-level `canal.data/`, `railway.data/`, `drain.data/` copies; settlement layer `uso_update_sep2021`
- **Commit:** `205eb7c`
- **Command:** `uv run python scripts/inventory_barriers.py --config code-2025 --all-candidates --work-dir ~/measure_work/cache`

```text
block: layers
settlements: 4357
canal_features: 43
canal_geom_types: LineString;MultiLineString
canal_crs: EPSG:4326
canal_length_km: 33.9485
canal_within_settlement_bbox: yes
canal_crea_date: 20200802
canal_crea_time: 17595100
canal_mod_date: none
canal_has_qpj: yes
canal_flagged_settlements: 28
railway_features: 5356
railway_geom_types: LineString;MultiLineString
railway_crs: EPSG:4326
railway_length_km: 751.827
railway_within_settlement_bbox: yes
railway_crea_date: 20200802
railway_crea_time: 18034000
railway_mod_date: none
railway_has_qpj: yes
railway_flagged_settlements: 240
drain_features: 616
drain_geom_types: LineString;MultiLineString
drain_crs: EPSG:4326
drain_length_km: 403.317
drain_within_settlement_bbox: no
drain_crea_date: none
drain_crea_time: none
drain_mod_date: none
drain_has_qpj: yes
drain_flagged_settlements: 390
canal_checked_features: 45
canal_checked_geom_types: LineString;MultiLineString
canal_checked_crs: EPSG:4326
canal_checked_length_km: 33.9485
canal_checked_within_settlement_bbox: yes
canal_checked_crea_date: none
canal_checked_crea_time: none
canal_checked_mod_date: none
canal_checked_has_qpj: yes
canal_checked_flagged_settlements: 28
canal_root_features: 43
canal_root_geom_types: LineString;MultiLineString
canal_root_crs: EPSG:4326
canal_root_length_km: 33.9485
canal_root_within_settlement_bbox: yes
canal_root_crea_date: none
canal_root_crea_time: none
canal_root_mod_date: none
canal_root_has_qpj: no
canal_root_flagged_settlements: 28
railway_root_features: 5356
railway_root_geom_types: LineString;MultiLineString
railway_root_crs: EPSG:4326
railway_root_length_km: 751.827
railway_root_within_settlement_bbox: yes
railway_root_crea_date: none
railway_root_crea_time: none
railway_root_mod_date: none
railway_root_has_qpj: no
railway_root_flagged_settlements: 240
drain_root_features: 616
drain_root_geom_types: LineString;MultiLineString
drain_root_crs: EPSG:4326
drain_root_length_km: 403.317
drain_root_within_settlement_bbox: no
drain_root_crea_date: none
drain_root_crea_time: none
drain_root_mod_date: none
drain_root_has_qpj: no
drain_root_flagged_settlements: 390
flagged_any: 595
```
```text
block: attributes
canal_columns: FID_1;CAN_NM;CAN_CLSF;EL_GND;DIST_NM
canal_values_CAN_NM: AGRA CANAL;GURGAON CANAL;WESTERN YAMUNA CANAL;WESTREN YAMUNA CANAL
railway_columns: FID_1;RL_ZONE
railway_values_RL_ZONE: NORTHERN RAILWAY
drain_columns: FID;Drain_type;Drain_Name;MAINTAINED;AC_NAME;DISTRICT
drain_values_Drain_Name: ALI DRAIN;ALIPUR LINK DRAIN;Asola Drain;BAKKARWALA OUTFALL DRAIN;BAWANA JHEEL LINK DRAIN;BURARI CREEK NEW COURSE;Bankner Link Drain;Bawana Drain;Bawana Escape Drain;Bazidpur Drain;Bhupania Chudania Drain;Biharipur Drain;Bijwasan Drain;Bund Drain;Burari Creek Drain;Burari Drain;Daryapur Pond Drain;Dichaon Kalan Link Drain;Drain No. 6;Escape Drain No. I;... (+44 more)
drain_values_DISTRICT: CENTRAL;EAST;NORTH;NORTH EAST;NORTH WEST;SHAHDARA;SOUTH;SOUTH EAST;SOUTH WEST;WEST;YAMUNA RIVER
drain_values_AC_NAME: ADARSH NAGAR;BABARPUR;BADARPUR;BADLI;BAWANA;BIJWASAN;BURARI;CHHATARPUR;DWARKA;GANDHI NAGAR;GOKALPUR;JANAKPURI;KARAWAL NAGAR;KIRARI;KONDLI;LAXMI NAGAR;MADIPUR;MANGOL PURI;MATIALA;MODEL TOWN;... (+28 more)
canal_checked_columns: FID;CAN_NM;CAN_CLSF;EL_GND;DIST_NM
canal_checked_values_CAN_NM: AGRA CANAL;GURGAON CANAL;WESTERN YAMUNA CANAL;WESTREN YAMUNA CANAL
canal_root_columns: index;FID_1;CAN_NM;CAN_CLSF;EL_GND;DIST_NM;geom_type
canal_root_values_CAN_NM: AGRA CANAL;GURGAON CANAL;WESTERN YAMUNA CANAL;WESTREN YAMUNA CANAL
railway_root_columns: index;FID_1;RL_ZONE;geom_type
railway_root_values_RL_ZONE: NORTHERN RAILWAY
drain_root_columns: index;FID;Drain_type;Drain_Name;MAINTAINED;AC_NAME;DISTRICT;geom_type
drain_root_values_Drain_Name: ALI DRAIN;ALIPUR LINK DRAIN;Asola Drain;BAKKARWALA OUTFALL DRAIN;BAWANA JHEEL LINK DRAIN;BURARI CREEK NEW COURSE;Bankner Link Drain;Bawana Drain;Bawana Escape Drain;Bazidpur Drain;Bhupania Chudania Drain;Biharipur Drain;Bijwasan Drain;Bund Drain;Burari Creek Drain;Burari Drain;Daryapur Pond Drain;Dichaon Kalan Link Drain;Drain No. 6;Escape Drain No. I;... (+44 more)
drain_root_values_DISTRICT: CENTRAL;EAST;NORTH;NORTH EAST;NORTH WEST;SHAHDARA;SOUTH;SOUTH EAST;SOUTH WEST;WEST;YAMUNA RIVER
drain_root_values_AC_NAME: ADARSH NAGAR;BABARPUR;BADARPUR;BADLI;BAWANA;BIJWASAN;BURARI;CHHATARPUR;DWARKA;GANDHI NAGAR;GOKALPUR;JANAKPURI;KARAWAL NAGAR;KIRARI;KONDLI;LAXMI NAGAR;MADIPUR;MANGOL PURI;MATIALA;MODEL TOWN;... (+28 more)
```

### Finding

The three layers the profile uses hold `43` canal features (`33.9485` km),
`5356` railway features (`751.827` km) and `616` major-drain features
(`403.317` km), all in EPSG:4326 and all line geometries. Under today's
rule the canal flags `28` settlements, the railway `240` and the drains
`390`; `595` settlements are flagged by at least one layer, which is the
figure the barrier discussion has used since the oracle memo. The drain
layer extends beyond the settlement layer's bounding box; the canal and
railway layers do not.

The extra copies are not different data. The root-level `canal.data/`,
`railway.data/` and `drain.data/` shapefiles have the same feature counts,
the same lengths to the metre and flag the same number of settlements as the clipped
layers; they carry two extra columns (`index`, `geom_type`) that mark them
as re-exports of the same frames, not unclipped originals.
`checked_Canal.shp` has `45` features against the canal layer's `43`, with
the same total length and the same count of `28` flagged settlements, so the
difference is two split features, not new geometry. The ESRI sidecars agree
on the date: both the canal and the railway layer were created on
`20200802`; the drain layer and the copies have no sidecar.

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
- **The paper's "manually marked" sentence describes a selection step, not
  the digitising.** The April 2026 draft (pp. 15–16) says the team "manually
  marked areas that had river or railroad tracks". The directory name is
  `Barrier_Clip/`, and the layers' attribute schemas are those of a source
  that predates this project — so what was done by hand is plausibly the
  selection of an existing layer's features for the study area. The extra
  copies at the data root are re-exports of the same clipped data (finding
  above), so the unclipped source is not in the data folder.

**The question that remains, for Bijoy (batched reply, item 7):** which
agency's layer are these — and what exactly did "manually marked" cover:
selecting features, clipping to the study area, or drawing lines?

