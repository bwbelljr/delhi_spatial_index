"""JJC road access, and the one-factor effect of `roads: eq4_own_only`
(DEL-49, spec § 2.1).

Two questions, one script, two blocks:

  access      For every settlement in the deduplicated, reprojected universe:
              is there a major road INSIDE its own polygon, only in a `touch`
              neighbour's, or neither — counted by USO_FINAL type. "Inside"
              is positive intersection length, the same membership
              `delhi_psi.index.road_lengths` uses, so `road_inside` is
              exactly `road_length > 0` in today's outputs. Neighbours use
              `touch` (Raj's ratified rule, DEL-19), not today's bbox: the
              question is about the world after the decision.

  one_factor  `code-2025` with ONE value changed — `methodology.roads:
              eq4_own_only` — run against the SAME neighbours artifact, and
              diffed by type against the proven `code-2025` outputs, which
              are READ from --verify-dir and never recomputed here.

READ-ONLY over --data-dir and --verify-dir. Everything this script writes
goes under --work-dir, which is never inside the data directory.

    uv run python scripts/measure_roads_access.py --config code-2025 \
        --verify-dir ~/delhi_data/phase3_verify --work-dir ~/measure_work/cache
"""

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

from delhi_psi import geometry, io, neighbors, pipeline
from delhi_psi.config import PROFILES_DIR, load_config
from delhi_psi.pipeline import ID_COL, NBRS_COL, TYPE_COL
from scripts._measure_common import load_settlements, render, resolve_work_dir

# The 28 Aug 2026 decisions: seven reported types, three dropped.
REPORTED_TYPES = ("Planned", "UAC", "RUAC", "JJC", "JJR", "UV", "SDA")
DROPPED_TYPES = ("RV", "Industrial", "Other")
ALL_TYPES = REPORTED_TYPES + DROPPED_TYPES
DENOMINATORS = ("pop", "popdensity")
OWN_ONLY_PROFILE = "roads-own-only"
ROAD_SERVICE = "road"
ROAD_IDX_COL = "road_idx"
ROAD_LENGTH_COL = "road_length"
PSI_COL = "unnorm_psi"
ACCESS_KINDS = ("road_inside", "road_via_neighbor", "no_road")
MISMATCH_SAMPLE = 10


def road_inside_ids(gdf, roads, *, id_col=ID_COL):
    """Ids whose polygon contains a POSITIVE LENGTH of road.

    `index.road_lengths` clips the road layer to the polygon and sums the
    pieces, so a road that merely touches the boundary at a point contributes
    nothing and is not "inside". The sjoin narrows the candidates; the length
    test decides.
    """
    frame = gdf[[id_col, "geometry"]].reset_index(drop=True)
    lines = roads[["geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(frame, lines, how="inner", predicate="intersects")
    ids, geoms, line_geoms = frame[id_col], frame.geometry, lines.geometry
    inside = set()
    for left, right in zip(joined.index, joined["index_right"]):
        settlement = ids.iloc[left]
        if settlement in inside:
            continue
        if geoms.iloc[left].intersection(line_geoms.iloc[right]).length > 0:
            inside.add(settlement)
    return inside


def assert_road_inside_matches_output(inside_ids, verify_csv_path, id_col):
    """`road_inside` must BE `road_length > 0` in today's outputs — spec
    § 2.1 (a) requires this script to assert it, not to claim it.

    `road_inside_ids` reads and reprojects the road layer itself and tests
    intersection length; production's `index.road_lengths` clips the same
    layer inside `compute`. They are two implementations of one membership,
    and if they ever disagree — a different duplicate-dropping rule, a
    different sjoin predicate, a MultiLineString with mixed zero/positive
    parts — the `access` block would quietly stop describing the outputs it
    claims to describe. So this RAISES; it never warns.

    The comparison is over the ids the CSV reports, not over the whole layer:
    the code-2025 run excludes some settlements (RV and friends — 4,131 rows
    out of 4,357), and their absence from the output is not a mismatch. Every
    id the CSV DOES carry must fall on the same side of the line in both.

    Returns None; raises ValueError naming up to MISMATCH_SAMPLE offenders in
    each direction, because which direction it fell tells you which side to
    look at.
    """
    reported = pd.read_csv(verify_csv_path, usecols=[id_col, ROAD_LENGTH_COL])
    reported_ids = set(reported[id_col])
    positive = set(reported.loc[reported[ROAD_LENGTH_COL] > 0, id_col])
    classified = set(inside_ids) & reported_ids

    inside_but_zero = sorted(classified - positive)
    positive_but_outside = sorted(positive - classified)
    if not inside_but_zero and not positive_but_outside:
        return None
    raise ValueError(
        f"road_inside does not match {ROAD_LENGTH_COL} > 0 in "
        f"{verify_csv_path}: {len(inside_but_zero)} classified road_inside "
        f"with {ROAD_LENGTH_COL} == 0 "
        f"{inside_but_zero[:MISMATCH_SAMPLE]}, and "
        f"{len(positive_but_outside)} with {ROAD_LENGTH_COL} > 0 not "
        f"classified road_inside {positive_but_outside[:MISMATCH_SAMPLE]} "
        f"(showing at most {MISMATCH_SAMPLE} of each). The access block "
        "would misdescribe today's outputs; spec § 2.1 (a).")


def measure_access(gdf, roads, *, id_col=ID_COL, type_col=TYPE_COL,
                   types=ALL_TYPES, inside=None):
    """Block `access`: road inside / only via a `touch` neighbour / neither,
    per settlement type and in total.

    A `touch` neighbour may be a DROPPED type; that is correct — dropped
    settlements still lend (28 Aug decision, semantics (a)), and a road in the
    rural village next door is exactly what Raj asked about.

    `inside` lets a caller pass a `road_inside_ids` result it has ALREADY
    computed — `measure` does, because it must run the
    `assert_road_inside_matches_output` check on that same set before using
    it — so the sjoin happens once. Default None recomputes it.
    """
    unknown = sorted(set(gdf[type_col].dropna()) - set(types))
    if unknown:
        raise ValueError(
            f"the layer carries settlement types this report has no key for: "
            f"{unknown}; known types: {list(types)}")
    if inside is None:
        inside = road_inside_ids(gdf, roads, id_col=id_col)
    frame = neighbors.adjacency(gdf, id_col=id_col, neighbor_col=NBRS_COL,
                                rule="touch")
    kinds = {}
    for _, row in frame.iterrows():
        if row[id_col] in inside:
            kind = "road_inside"
        elif any(neighbor in inside for neighbor in row[NBRS_COL]):
            kind = "road_via_neighbor"
        else:
            kind = "no_road"
        kinds[row[id_col]] = (kind, row[type_col])

    report = {}
    for kind in ACCESS_KINDS:
        for name in (*types, "total"):
            report[f"{kind}_{name}"] = sum(
                1 for found, found_type in kinds.values()
                if found == kind and (name == "total" or found_type == name))
    return report


def base_profile_path(base):
    """A shipped profile NAME or a path to a YAML file — `load_config`'s own
    rule, so `--config code-2025` and `--config some/derived.yaml` both work
    (the Oraculum proof passes a derived path)."""
    candidate = Path(base)
    if candidate.suffix in (".yaml", ".yml"):
        return candidate
    return PROFILES_DIR / f"{base}.yaml"


def derived_profile(base, work_dir, *, profile_name=OWN_ONLY_PROFILE):
    """`base` with ONE methodology value changed: `roads: eq4_own_only`.

    `paths.neighbors_artifact` and `paths.out_dir` are dropped, so the
    per-profile default name applies (colonies_neighbors_<profile>.joblib) and
    --out-dir decides where the run writes. Every other key is byte-equal
    after the YAML round trip. It is written to DISK, not held in memory, so
    the run is reproducible by hand with

        delhi-psi compute
            --config <work-dir>/roads-own-only/roads-own-only.yaml
            --data-dir <data-dir>
            --out-dir <work-dir>/roads-own-only

    — the --out-dir matters: with `paths.out_dir` dropped, a hand run without
    it would not write beside the staged artifact and would not find it.
    """
    raw = yaml.safe_load(base_profile_path(base).read_text())
    raw["profile"] = profile_name
    raw["methodology"]["roads"] = "eq4_own_only"
    paths = dict(raw.get("paths", {}))
    paths.pop("neighbors_artifact", None)
    paths.pop("out_dir", None)
    raw["paths"] = paths
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"{profile_name}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def stage_artifacts(verify_dir, run_dir, *, source_name, artifact_name):
    """Copy the PROVEN neighbours artifact into the run directory under the
    name the derived profile looks for.

    THE ARTIFACT ALONE (spec § 2.1 (b) item 2). `compute` reads the neighbours
    joblib, the population CSV and the service layers — never a
    `*.dedup.gpkg`. Those caches belong to `preprocess`, which this script
    never runs: the roads switch is applied downstream in `index_frames`, so
    the stored neighbour lists stay valid (spec § 2.1 (c)). Do not add
    dedup-cache copying here, and do not stage dedup caches into the run
    directory from outside either — the only dedup cache this script benefits
    from is the one behind `load_settlements` for the `access` block, which
    lives in --work-dir itself, not in <work-dir>/roads-own-only.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / artifact_name
    shutil.copy2(Path(verify_dir) / source_name, target)
    return target


def _mean(values):
    """Means are pre-formatted, like the pathology block's areas, so the
    drift comparison is exact."""
    return "nan" if values.empty else f"{values.mean():.6g}"


def measure_effect(decayed, own, *, denom, id_col=ID_COL, type_col=TYPE_COL,
                   types=REPORTED_TYPES):
    """Block `one_factor`, one denominator.

    `decayed` is the proven `code-2025` output read from --verify-dir; `own`
    is this script's run with only `roads` changed. Both must report exactly
    the same settlements, or the comparison is not one-factor.
    """
    left = decayed.set_index(id_col)
    right = own.set_index(id_col)
    if set(left.index) != set(right.index):
        raise ValueError(
            f"{denom}: the two runs report different settlements "
            f"(decayed-only {len(set(left.index) - set(right.index))}, "
            f"own-only {len(set(right.index) - set(left.index))})")
    right = right.reindex(left.index)

    selectors = {name: (left[type_col] == name) for name in types}
    selectors["total"] = pd.Series(True, index=left.index)

    report = {}
    for name, rows in selectors.items():
        report[f"n_{denom}_{name}"] = int(rows.sum())
    for name, rows in selectors.items():
        report[f"road_idx_decayed_{denom}_{name}"] = _mean(
            left.loc[rows, ROAD_IDX_COL])
    for name, rows in selectors.items():
        report[f"road_idx_own_{denom}_{name}"] = _mean(
            right.loc[rows, ROAD_IDX_COL])
    for name, rows in selectors.items():
        report[f"psi_decayed_{denom}_{name}"] = _mean(left.loc[rows, PSI_COL])
    for name, rows in selectors.items():
        report[f"psi_own_{denom}_{name}"] = _mean(right.loc[rows, PSI_COL])
    for name, rows in selectors.items():
        # FELL to zero: it had something to lose and lost all of it.
        report[f"road_idx_zeroed_{denom}_{name}"] = int(
            ((right.loc[rows, ROAD_IDX_COL] == 0)
             & (left.loc[rows, ROAD_IDX_COL] > 0)).sum())
    return report


def measure(cfg, work_dir, *, base, verify_dir):
    """Both blocks: {"access": {...}, "one_factor": {...}}."""
    id_col = cfg.layers.settlements.id_col
    type_col = cfg.layers.settlements.type_col
    settlements = load_settlements(cfg, work_dir)
    roads = io.read_layer(cfg.paths.data_dir / cfg.services.line[ROAD_SERVICE])
    # `compute` drops exact-duplicate service rows before counting; do the
    # same here so a duplicated road row cannot change the membership.
    roads = roads.drop_duplicates().reset_index(drop=True)
    roads = geometry.reproject(roads, cfg.crs.epsg)

    inside = road_inside_ids(settlements, roads, id_col=id_col)
    # --verify-dir is required by main(), so on the real run this ALWAYS
    # fires: the block's `road_inside` claim is checked against the column it
    # claims to equal before a single count is written (spec § 2.1 (a)).
    # `road_length` is the same in both denominators' outputs; the `pop` one
    # is read.
    assert_road_inside_matches_output(
        inside,
        Path(verify_dir) / f"{pipeline.output_basename(cfg, 'pop')}.csv",
        id_col)
    access = measure_access(settlements, roads, id_col=id_col,
                            type_col=type_col, inside=inside)

    stamp = pipeline.methodology_stamp(cfg.methodology)
    # BOTH shapes. `methodology_stamp` returns one entry per methodology
    # concern, so the likely way `roads` arrives is a new TOP-LEVEL block —
    # which a `stamp.values()` scan alone would walk straight past, leaving
    # `compute`'s `check_methodology_stamp` to report the re-preprocess as an
    # unrelated-looking stale-artifact error.
    if "roads" in stamp or any("roads" in block for block in stamp.values()):
        raise SystemExit(
            "pipeline.methodology_stamp now carries `roads`: the neighbours "
            "artifact would have to be rebuilt and this script's one-factor "
            "run cannot reuse --verify-dir's. Re-read spec § 2.1 (b) before "
            "changing anything.")

    run_dir = Path(work_dir) / OWN_ONLY_PROFILE
    profile_path = derived_profile(base, run_dir)
    own_cfg = load_config(profile_path, data_dir=str(cfg.paths.data_dir),
                          out_dir=str(run_dir))
    stage_artifacts(verify_dir, run_dir,
                    source_name=cfg.paths.neighbors_artifact,
                    artifact_name=own_cfg.paths.neighbors_artifact)
    pipeline.compute(own_cfg)

    one_factor = {}
    for denom in DENOMINATORS:
        decayed = pd.read_csv(
            Path(verify_dir) / f"{pipeline.output_basename(cfg, denom)}.csv")
        own = pd.read_csv(
            run_dir / f"{pipeline.output_basename(own_cfg, denom)}.csv")
        one_factor.update(measure_effect(decayed, own, denom=denom,
                                         id_col=id_col, type_col=type_col))
    return {"access": access, "one_factor": one_factor}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code-2025",
                        help="profile that names the layers (default code-2025)")
    parser.add_argument("--data-dir", default=None,
                        help="data root, opened READ-ONLY")
    parser.add_argument("--work-dir", default=None,
                        help="scratch (dedup cache, the derived profile, its "
                             "outputs); default a fresh temporary directory. "
                             "Never under --data-dir.")
    parser.add_argument("--verify-dir", required=True,
                        help="an existing, complete code-2025 run "
                             "(colonies_neighbors.joblib + both output CSVs), "
                             "opened READ-ONLY")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, data_dir=args.data_dir)
    work_dir = resolve_work_dir(args.work_dir, data_dir=cfg.paths.data_dir,
                                prefix="delhi_psi_roads_")
    verify_dir = Path(args.verify_dir).expanduser()

    print(f"layer: {cfg.paths.data_dir / cfg.layers.settlements.path}")
    print(f"roads: {cfg.paths.data_dir / cfg.services.line[ROAD_SERVICE]}")
    print(f"verify-dir: {verify_dir}")
    print(f"work-dir: {work_dir}")
    blocks = measure(cfg, work_dir, base=args.config, verify_dir=verify_dir)
    for name, report in blocks.items():
        print(render(report, name=name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
