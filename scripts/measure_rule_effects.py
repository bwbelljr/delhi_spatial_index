"""What the partial-barrier rule does to today's numbers (DEL-48, spec § 7).

One block for now:

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

Unlike the roads measurement this block CANNOT reuse --verify-dir's
neighbours artifact: `methodology.barrier` is in the methodology stamp, so
the artifact has to be rebuilt. It is rebuilt into --work-dir, never beside
the proven one.

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

from delhi_psi import io, pipeline
from delhi_psi.config import PROFILES_DIR, load_config
from delhi_psi.pipeline import ID_COL, NBRS_COL, NBRS_WEIGHT_COL, TYPE_COL
from scripts._measure_common import render, resolve_work_dir

REPORTED_TYPES = ("Planned", "UAC", "RUAC", "JJC", "JJR", "UV", "SDA")
DENOMINATORS = ("pop", "popdensity")
PARTIAL_PROFILE = "partial-barriers-5m"
BUFFER_M = 5
PSI_COL = "unnorm_psi"
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


def measure_effect(before, after, *, denom, id_col=ID_COL, type_col=TYPE_COL,
                   types=REPORTED_TYPES):
    """The one-factor per-type PSI shift, the DEL-49 `one_factor` shape.

    `before` is the proven `code-2025` output read from --verify-dir; `after`
    is this script's run with only the barrier block changed. Both must
    report exactly the same settlements, or the comparison is not
    one-factor.
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
        report[f"psi_partial_{denom}_{name}"] = _mean(right.loc[rows, PSI_COL])
    if "norm_psi" in left.columns and "norm_psi" in right.columns:
        for name, rows in selectors.items():
            report[f"norm_code_{denom}_{name}"] = _mean(
                left.loc[rows, "norm_psi"])
        for name, rows in selectors.items():
            report[f"norm_partial_{denom}_{name}"] = _mean(
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
    args = parser.parse_args(argv)

    cfg = load_config(args.config, data_dir=args.data_dir)
    work_dir = resolve_work_dir(args.work_dir, data_dir=cfg.paths.data_dir,
                                prefix="delhi_psi_rules_")
    verify_dir = Path(args.verify_dir).expanduser()

    print(f"layer: {cfg.paths.data_dir / cfg.layers.settlements.path}")
    print(f"verify-dir: {verify_dir}")
    print(f"work-dir: {work_dir}")
    print(render(measure_partial_barriers(cfg, work_dir, base=args.config,
                                          verify_dir=verify_dir),
                 name="partial_barriers"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
