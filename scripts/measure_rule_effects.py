"""What the cycle-3E rule changes do to today's numbers (spec § 7).

Two blocks:

  partial_barriers  `code-2025` with ONE thing changed —
                    `methodology.barrier`: {rule: partial_weighted,
                    combine: any, buffer_m: 5} — everything else including
                    `adjacency: bbox` left alone, so the diff is
                    attributable to the barrier rule alone. Reports the
                    directed links by weight class, how many links each rule
                    severs, how many settlements' lists changed, the
                    fractional count and median weight, the preprocess
                    wall-clock, and the per-type PSI shift against the proven
                    `code-2025` outputs read from --verify-dir.

  overlap_lending   `code-2025` with ONE thing changed —
                    `methodology.overlap`: {lending: outside_receiver} — so
                    the diff is attributable to the lending rule alone.
                    Unlike the barrier block this one REUSES the proven
                    neighbours artifact: the overlap rule is downstream of
                    the neighbour structure and is not in the methodology
                    stamp, which the script asserts before staging it.
                    Reports how many ordered (pair, service) entries the
                    sparse shared structure holds per service, how many
                    settlements have an overlapping neighbour at all, how
                    many settlements' PCEN moved and in which direction, and
                    the per-type PSI shift against the proven `code-2025`
                    outputs read from --verify-dir.

Unlike the roads measurement the `partial_barriers` block CANNOT reuse
--verify-dir's neighbours artifact: `methodology.barrier` is in the
methodology stamp, so the artifact has to be rebuilt. It is rebuilt into
--work-dir, never beside the proven one. `overlap_lending` reuses it, like
the roads measurement.

READ-ONLY over --data-dir and --verify-dir. Everything this script writes
goes under --work-dir, which is never inside the data directory.

    uv run python scripts/measure_rule_effects.py --config code-2025 \
        --verify-dir ~/delhi_data/phase3_verify --work-dir ~/measure_work/cache
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from delhi_psi import geometry, index, io, pipeline
from delhi_psi.config import PROFILES_DIR, load_config
from delhi_psi.pipeline import ID_COL, NBRS_COL, NBRS_WEIGHT_COL, TYPE_COL
from scripts._measure_common import emit, emit_check, render, resolve_work_dir
from scripts.measure_roads_access import stage_artifacts

REPORTED_TYPES = ("Planned", "UAC", "RUAC", "JJC", "JJR", "UV", "SDA")
DENOMINATORS = ("pop", "popdensity")
PARTIAL_PROFILE = "partial-barriers-5m"
OVERLAP_PROFILE = "overlap-outside-receiver"
BLOCKS = ("partial_barriers", "overlap_lending")
BUFFER_M = 5
PSI_COL = "unnorm_psi"
PCEN_SUFFIX = "_pcen"
WEIGHT_CLASSES = ("links_w_one", "links_fractional", "links_severed")


def base_profile_path(base):
    """A shipped profile NAME or a path to a YAML file — `load_config`'s own
    rule, so `--config code-2025` and a derived path both work."""
    candidate = Path(base)
    if candidate.suffix in (".yaml", ".yml"):
        return candidate
    return PROFILES_DIR / f"{base}.yaml"


def derived_profile(base, work_dir, *, profile_name, methodology):
    """`base` with the named methodology BLOCKS replaced wholesale.

    Blocks are replaced, never deep-merged: `methodology.<block>` is a
    complete statement, exactly as tests/variants.py and
    oraculum_fixtures.oracle_profile_path treat it. `paths.neighbors_artifact`
    and `paths.out_dir` are dropped so the per-profile default name applies
    and --out-dir decides where the run writes. Written to DISK, so the run
    is reproducible by hand.
    """
    raw = yaml.safe_load(base_profile_path(base).read_text())
    raw["profile"] = profile_name
    for block, values in methodology.items():
        raw["methodology"][block] = dict(values)
    paths = dict(raw.get("paths", {}))
    paths.pop("neighbors_artifact", None)
    paths.pop("out_dir", None)
    raw["paths"] = paths
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"{profile_name}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def weight_classes(frame, *, weight_col=NBRS_WEIGHT_COL):
    """Directed links by weight class, plus the median fractional weight.

    `links_severed` is 0 BY CONSTRUCTION on a stored artifact: apply_barrier
    prunes a w == 0 link out of both columns, so a severed link leaves no
    trace here. It is reported anyway, as the assertion that nothing
    survived at weight 0 — how many links each rule severs is
    `compare_link_sets`' job, from the two rules' kept counts. Medians are
    pre-formatted, like the roads block's means, so the drift comparison is
    exact.
    """
    weights = [w for row in frame[weight_col] for _, w in row]
    fractional = [w for w in weights if 0.0 < w < 1.0]
    return {
        "links_w_one": sum(1 for w in weights if w == 1.0),
        "links_fractional": len(fractional),
        "links_severed": sum(1 for w in weights if w == 0.0),
        "median_fractional_w": ("nan" if not fractional
                                else f"{statistics.median(fractional):.6g}"),
    }


def link_sets(frame, *, id_col=ID_COL, neighbor_col=NBRS_COL):
    """{i: frozenset(js)} — the directed neighbour lists, for diffing."""
    return {row[id_col]: frozenset(row[neighbor_col])
            for _, row in frame.iterrows()}


def compare_link_sets(before, after):
    """How the two rules' stored lists differ.

    KEPT links, not severed ones: a severed link is absent from both
    artifacts, so the only honest counts are what each rule left behind. The
    stated bound — the partial rule severs FEWER links, because the global
    rule drops every link INTO a flagged settlement while the partial rule
    drops only fully covered boundaries — reads as
    `links_kept_partial > links_kept_code_2025`.
    """
    changed = [i for i in before if before[i] != after.get(i, frozenset())]
    return {
        "links_kept_code_2025": sum(len(js) for js in before.values()),
        "links_kept_partial": sum(len(js) for js in after.values()),
        "settlements_list_changed": len(changed),
    }


def _mean(values):
    return "nan" if values.empty else f"{values.mean():.6g}"


def measure_effect(before, after, *, denom, label="partial", id_col=ID_COL,
                   type_col=TYPE_COL, types=REPORTED_TYPES):
    """The one-factor per-type PSI shift, the DEL-49 `one_factor` shape.

    `before` is the proven `code-2025` output read from --verify-dir; `after`
    is this script's run with only ONE methodology block changed. Both must
    report exactly the same settlements, or the comparison is not
    one-factor. `label` names the "after" run in the report's key names —
    it defaults to `"partial"`, so the already-committed barrier block's key
    names stay byte-identical.
    """
    left = before.set_index(id_col)
    right = after.set_index(id_col)
    if set(left.index) != set(right.index):
        raise ValueError(
            f"{denom}: the two runs report different settlements "
            f"(before-only {len(set(left.index) - set(right.index))}, "
            f"after-only {len(set(right.index) - set(left.index))})")
    right = right.reindex(left.index)

    selectors = {name: (left[type_col] == name) for name in types}
    selectors["total"] = pd.Series(True, index=left.index)

    report = {}
    for name, rows in selectors.items():
        report[f"n_{denom}_{name}"] = int(rows.sum())
    for name, rows in selectors.items():
        report[f"psi_code_{denom}_{name}"] = _mean(left.loc[rows, PSI_COL])
    for name, rows in selectors.items():
        report[f"psi_{label}_{denom}_{name}"] = _mean(right.loc[rows, PSI_COL])
    if "norm_psi" in left.columns and "norm_psi" in right.columns:
        for name, rows in selectors.items():
            report[f"norm_code_{denom}_{name}"] = _mean(
                left.loc[rows, "norm_psi"])
        for name, rows in selectors.items():
            report[f"norm_{label}_{denom}_{name}"] = _mean(
                right.loc[rows, "norm_psi"])
    return report


def measure_partial_barriers(cfg, work_dir, *, base, verify_dir):
    """Block `partial_barriers`: rebuild the neighbours under the partial
    rule, then diff against the proven code-2025 outputs."""
    run_dir = Path(work_dir) / PARTIAL_PROFILE
    profile_path = derived_profile(
        base, run_dir, profile_name=PARTIAL_PROFILE,
        methodology={"barrier": {"rule": "partial_weighted",
                                 "combine": "any", "buffer_m": BUFFER_M}})
    run_cfg = load_config(profile_path, data_dir=str(cfg.paths.data_dir),
                          out_dir=str(run_dir))

    started = time.monotonic()
    pipeline.preprocess(run_cfg)
    elapsed = time.monotonic() - started
    pipeline.compute(run_cfg)

    after_frame = io.read_neighbors(
        run_dir / run_cfg.paths.neighbors_artifact)
    before_frame = io.read_neighbors(
        Path(verify_dir) / cfg.paths.neighbors_artifact)

    report = weight_classes(after_frame)
    report.update(compare_link_sets(link_sets(before_frame),
                                    link_sets(after_frame)))
    report["preprocess_seconds"] = f"{elapsed:.6g}"
    for denom in DENOMINATORS:
        before = pd.read_csv(
            Path(verify_dir) / f"{pipeline.output_basename(cfg, denom)}.csv")
        after = pd.read_csv(
            run_dir / f"{pipeline.output_basename(run_cfg, denom)}.csv")
        report.update(measure_effect(before, after, denom=denom,
                                     id_col=cfg.layers.settlements.id_col,
                                     type_col=cfg.layers.settlements.type_col))
    return report


def service_layers(cfg):
    """The service layers `compute` loads, deduplicated and reprojected the
    way `index_frames` reprojects them — so the counts below describe the
    frames the run actually scored, not the files on disk."""
    data_dir = cfg.paths.data_dir
    out = {}
    for name, path in {**cfg.services.point, **cfg.services.line}.items():
        gdf = io.read_layer(data_dir / path).drop_duplicates().reset_index(
            drop=True)
        out[name] = geometry.reproject(gdf, cfg.crs.epsg)
    return out


def shared_pair_counts(frame, layers, *, point_names, id_col=ID_COL,
                       neighbor_col=NBRS_COL):
    """{'shared_pairs_<svc>': n} — the number of ORDERED (i, j) entries
    `index.shared_amounts` builds per service, plus the total.

    Built with the SAME function `index_frames` calls, on the same frames,
    so these are the real structure's sizes and not an estimate. It costs
    one extra pass of the amount helpers, because the line branch skips
    pairs where either side owns none of the service and those amounts are
    not in the stored artifact.
    """
    amounts = frame
    kinds = {name: ("point" if name in point_names else "line")
             for name in layers}
    for name, gdf in layers.items():
        column = index.service_amount_column(name, kinds[name])
        if kinds[name] == "point":
            amounts = index.point_counts(amounts, gdf, count_col=column,
                                         id_col=id_col)
        else:
            amounts = index.road_lengths(amounts, gdf, length_col=column,
                                         id_col=id_col)
    report = {}
    for name, gdf in layers.items():
        table = index.shared_amounts(
            amounts, gdf, kind=kinds[name],
            amount_col=index.service_amount_column(name, kinds[name]),
            neighbor_col=neighbor_col, id_col=id_col)
        report[f"shared_pairs_{name}"] = len(table)
    report["shared_pairs_total"] = sum(report.values())
    return report


def overlapping_neighbours(frame, *, id_col=ID_COL, neighbor_col=NBRS_COL):
    """Ids with at least one STORED neighbour whose polygon overlaps theirs
    (positive-area intersection). The lending rule cannot move any other
    settlement's PCEN, so the changed set must be a subset of this one —
    which is the containment bound stated before the run."""
    geoms = frame.set_index(id_col).geometry
    out = set()
    for _, row in frame.iterrows():
        i = row[id_col]
        for j in row[neighbor_col]:
            if j in geoms.index and geoms[i].intersection(geoms[j]).area > 0:
                out.add(i)
                break
    return out


def pcen_changes(before, after, *, id_col=ID_COL):
    """(report, changed ids) — how the PCEN columns moved, and which way.

    Lending is only ever REDUCED (|S_j \\ S_i| <= |S_j|), so a RISEN PCEN is
    a bug and not a finding: `settlements_pcen_rose` must be 0.
    """
    left = before.set_index(id_col)
    right = after.set_index(id_col).reindex(left.index)
    columns = [c for c in left.columns if c.endswith(PCEN_SUFFIX)]
    changed, rose = set(), set()
    for column in columns:
        diff = right[column] - left[column]
        changed |= set(left.index[diff != 0])
        rose |= set(left.index[diff > 0])
    return ({"settlements_pcen_changed": len(changed),
             "settlements_pcen_rose": len(rose)}, changed)


def measure_overlap_lending(cfg, work_dir, *, base, verify_dir):
    """Block `overlap_lending`: the SAME neighbours, one methodology value
    changed."""
    stamp = pipeline.methodology_stamp(cfg.methodology)
    # BOTH shapes, like measure_roads_access's guard: a new concern is most
    # likely to arrive as a new TOP-LEVEL block, which a values() scan alone
    # would walk straight past.
    if "overlap" in stamp or any("overlap" in block for block in stamp.values()):
        raise SystemExit(
            "pipeline.methodology_stamp now carries `overlap`: the neighbours "
            "artifact would have to be rebuilt and this block's one-factor "
            "run cannot reuse --verify-dir's. Re-read spec § 3.2 before "
            "changing anything.")

    run_dir = Path(work_dir) / OVERLAP_PROFILE
    profile_path = derived_profile(
        base, run_dir, profile_name=OVERLAP_PROFILE,
        methodology={"overlap": {"lending": "outside_receiver"}})
    run_cfg = load_config(profile_path, data_dir=str(cfg.paths.data_dir),
                          out_dir=str(run_dir))
    stage_artifacts(verify_dir, run_dir,
                    source_name=cfg.paths.neighbors_artifact,
                    artifact_name=run_cfg.paths.neighbors_artifact)
    pipeline.compute(run_cfg)

    frame = io.read_neighbors(run_dir / run_cfg.paths.neighbors_artifact)
    report = shared_pair_counts(frame, service_layers(cfg),
                                point_names=tuple(cfg.services.point),
                                id_col=cfg.layers.settlements.id_col)
    overlapping = overlapping_neighbours(
        frame, id_col=cfg.layers.settlements.id_col)
    report["settlements_with_an_overlapping_neighbour"] = len(overlapping)

    for denom in DENOMINATORS:
        before = pd.read_csv(
            Path(verify_dir) / f"{pipeline.output_basename(cfg, denom)}.csv")
        after = pd.read_csv(
            run_dir / f"{pipeline.output_basename(run_cfg, denom)}.csv")
        moved, changed = pcen_changes(
            before, after, id_col=cfg.layers.settlements.id_col)
        report.update({f"{key}_{denom}": value
                       for key, value in moved.items()})
        report[f"pcen_changed_outside_the_overlap_set_{denom}"] = len(
            changed - overlapping)
        report.update(measure_effect(before, after, denom=denom,
                                     label="outside",
                                     id_col=cfg.layers.settlements.id_col,
                                     type_col=cfg.layers.settlements.type_col))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code-2025",
                        help="profile that names the layers (default code-2025)")
    parser.add_argument("--data-dir", default=None,
                        help="data root, opened READ-ONLY")
    parser.add_argument("--work-dir", default=None,
                        help="scratch (the derived profile, its artifact and "
                             "outputs); default a fresh temporary directory. "
                             "Never under --data-dir.")
    parser.add_argument("--verify-dir", required=True,
                        help="an existing, complete code-2025 run "
                             "(colonies_neighbors.joblib + both output CSVs), "
                             "opened READ-ONLY")
    parser.add_argument("--only", choices=BLOCKS, default=None,
                        help="run ONE block instead of both (the barrier "
                             "block re-runs preprocess and costs minutes; "
                             "the overlap block stages the proven artifact "
                             "and runs compute alone)")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--out", default=None,
                        help="write the blocks here instead of stdout — "
                             "BLOCKS ONLY; REFUSES (exit 1) to overwrite a "
                             "target that already holds hand-written prose, "
                             "since that would delete every caption and "
                             "Finding — use --splice to refresh such a "
                             "document in place instead")
    target.add_argument("--splice", default=None,
                        help="refresh the blocks INSIDE this committed "
                             "document in place, preserving every caption "
                             "and Finding (DEL-59); with --only, refreshes "
                             "only the named block and leaves the other run "
                             "untouched")
    args = parser.parse_args(argv)

    emit_check(out=args.out, splice=args.splice)

    cfg = load_config(args.config, data_dir=args.data_dir)
    work_dir = resolve_work_dir(args.work_dir, data_dir=cfg.paths.data_dir,
                                prefix="delhi_psi_rules_")
    verify_dir = Path(args.verify_dir).expanduser()

    print(f"layer: {cfg.paths.data_dir / cfg.layers.settlements.path}",
          file=sys.stderr)
    print(f"verify-dir: {verify_dir}", file=sys.stderr)
    print(f"work-dir: {work_dir}", file=sys.stderr)
    wanted = BLOCKS if args.only is None else (args.only,)
    measures = {"partial_barriers": measure_partial_barriers,
                "overlap_lending": measure_overlap_lending}
    text = "\n".join(render(measures[name](cfg, work_dir, base=args.config,
                                           verify_dir=verify_dir), name=name)
                     for name in wanted)
    emit(text, out=args.out, splice=args.splice)
    return 0


if __name__ == "__main__":
    sys.exit(main())
