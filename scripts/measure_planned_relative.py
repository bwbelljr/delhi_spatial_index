"""The reported relative scale: rel_psi = norm_psi / mean(norm_psi | Planned)
(DEL-63, spec § 5).

ONE block. `code-2025` with exactly one methodology value changed —
`second_normalization: planned_relative` — run against the PROVEN neighbours
artifact from --verify-dir: the normalisation is applied downstream in
`compute` and is not in the methodology stamp, which this script asserts
before staging the artifact. Reports, per denominator: the Planned mean
that divides `norm_psi`, the per-category mean of both `rel_psi` and
`norm_psi`, and the largest
|mean(rel | type) − mean(norm | type) / planned_mean| — the identity the
relative scale claims, measured independently rather than assumed.

READ-ONLY over --data-dir and --verify-dir. Everything this script writes
goes under --work-dir, which is never inside the data directory.

    uv run python scripts/measure_planned_relative.py --config code-2025 \
        --verify-dir ~/delhi_data/phase3_verify \
        --work-dir ~/measure_work/planned-relative
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from delhi_psi import pipeline
from delhi_psi.config import PROFILES_DIR, load_config
from delhi_psi.pipeline import ID_COL, TYPE_COL
from scripts._measure_common import emit, emit_check, render, resolve_work_dir
from scripts.measure_roads_access import stage_artifacts

PROFILE = "planned-relative"
DENOMINATORS = ("pop", "popdensity")
# The nine categories the code-2025 outputs carry: the ten Delhi source
# types minus RV, the one `code-2025` excludes. A missing type is a
# KeyError, not an empty row — the block names what the run reports.
REPORTED_TYPES = ("Planned", "UAC", "RUAC", "JJC", "JJR", "UV", "SDA",
                  "Industrial", "Other")
REL_COL = "rel_psi"
BASE_COL = "norm_psi"


def base_profile_path(base):
    """A shipped profile NAME or a path to a YAML file — `load_config`'s own
    rule, so `--config code-2025` and a derived path both work."""
    candidate = Path(base)
    if candidate.suffix in (".yaml", ".yml"):
        return candidate
    return PROFILES_DIR / f"{base}.yaml"


def derived_profile(base, work_dir, *, profile_name):
    """`base` with `second_normalization: planned_relative`.

    A SCALAR assignment, not a block replacement: `second_normalization`
    is one methodology key, so the `dict(values)` used by the block-shaped
    helpers would explode on the string. `paths.neighbors_artifact` and
    `paths.out_dir` are dropped so the per-profile default name applies and
    --out-dir decides where the run writes; the profile is written to DISK
    so the run is reproducible by hand.
    """
    raw = yaml.safe_load(base_profile_path(base).read_text())
    raw["profile"] = profile_name
    raw["methodology"]["second_normalization"] = "planned_relative"
    paths = dict(raw.get("paths", {}))
    paths.pop("neighbors_artifact", None)
    paths.pop("out_dir", None)
    raw["paths"] = paths
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"{profile_name}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def measure(cfg, work_dir, *, base, verify_dir):
    """The block: one factor changed, the proven artifact reused."""
    stamp = pipeline.methodology_stamp(cfg.methodology)
    # BOTH shapes, like measure_rule_effects' guard: a new concern is most
    # likely to arrive as a new TOP-LEVEL block, which a values() scan alone
    # would walk straight past.
    if ("second_normalization" in stamp
            or any("second_normalization" in block
                   for block in stamp.values())):
        raise SystemExit(
            "pipeline.methodology_stamp now carries `second_normalization`: "
            "the neighbours artifact is no longer reusable across this value "
            "and this run would silently report over a different method. "
            "Re-read spec § 2.1 before changing anything.")

    run_dir = Path(work_dir) / PROFILE
    profile_path = derived_profile(base, run_dir, profile_name=PROFILE)
    run_cfg = load_config(profile_path, data_dir=str(cfg.paths.data_dir),
                          out_dir=str(run_dir))
    stage_artifacts(verify_dir, run_dir,
                    source_name=cfg.paths.neighbors_artifact,
                    artifact_name=run_cfg.paths.neighbors_artifact)
    pipeline.compute(run_cfg)

    report = {}
    for denom in DENOMINATORS:
        frame = pd.read_csv(
            run_dir / f"{pipeline.output_basename(run_cfg, denom)}.csv")
        report.update(measure_denominator(
            frame, denom=denom,
            id_col=cfg.layers.settlements.id_col,
            type_col=cfg.layers.settlements.type_col))
    return report


def measure_denominator(frame, *, denom, id_col=ID_COL, type_col=TYPE_COL,
                        types=REPORTED_TYPES):
    """One denominator's keys: the divisor, both columns' per-type means,
    and the largest departure from the ratio identity.

    `id_col` is signature parity with the sibling blocks only — the frame
    is already one row per settlement and the grouping is by type.
    """
    planned = frame.loc[frame[type_col] == "Planned", BASE_COL]
    if planned.empty:
        raise ValueError(
            f"planned_relative needs at least one row whose {type_col!r} "
            f"is 'Planned': {denom} has {len(frame)} rows and none of them "
            "is, so there is no denominator to divide by")
    planned_mean = float(planned.mean())
    if planned_mean == 0.0:
        raise ValueError(
            "planned_relative divides by the mean norm_psi of the "
            f"{len(planned)} Planned rows, and that mean is 0.0 — every "
            "Planned settlement has the minimum unnormalized PSI, so the "
            "relative scale is undefined")
    grouped = frame.groupby(type_col)
    means_rel = grouped[REL_COL].mean()
    means_norm = grouped[BASE_COL].mean()
    report = {f"planned_norm_mean_{denom}": f"{planned_mean:.6g}"}
    for name in types:
        report[f"mean_{REL_COL}_{denom}_{name}"] = (
            f"{means_rel[name]:.6g}")
        report[f"mean_{BASE_COL}_{denom}_{name}"] = (
            f"{means_norm[name]:.6g}")
    worst = max(abs(means_rel[name] - means_norm[name] / planned_mean)
                for name in types)
    report[f"rel_vs_ratio_maxdiff_{denom}"] = f"{worst:.6g}"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code-2025",
                        help="profile that names the layers (default code-2025)")
    parser.add_argument("--data-dir", default=None,
                        help="data root, opened READ-ONLY")
    parser.add_argument("--work-dir", default=None,
                        help="scratch (the derived profile, its staged "
                             "artifact and outputs); default a fresh "
                             "temporary directory. Never under --data-dir.")
    parser.add_argument("--verify-dir", required=True,
                        help="an existing, complete code-2025 run "
                             "(colonies_neighbors.joblib + both output CSVs), "
                             "opened READ-ONLY")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--out", default=None,
                        help="write the block here instead of stdout — "
                             "BLOCKS ONLY; REFUSES (exit 1) to overwrite a "
                             "target that already holds hand-written prose, "
                             "since that would delete every caption and "
                             "Finding — use --splice to refresh such a "
                             "document in place instead")
    target.add_argument("--splice", default=None,
                        help="refresh the block INSIDE this committed "
                             "document in place, preserving every caption "
                             "and Finding (DEL-59)")
    args = parser.parse_args(argv)

    emit_check(out=args.out, splice=args.splice)

    cfg = load_config(args.config, data_dir=args.data_dir)
    work_dir = resolve_work_dir(args.work_dir, data_dir=cfg.paths.data_dir,
                                prefix="delhi_psi_planned_relative_")
    verify_dir = Path(args.verify_dir).expanduser()

    print(f"layer: {cfg.paths.data_dir / cfg.layers.settlements.path}",
          file=sys.stderr)
    print(f"verify-dir: {verify_dir}", file=sys.stderr)
    print(f"work-dir: {work_dir}", file=sys.stderr)
    report = measure(cfg, work_dir, base=args.config, verify_dir=verify_dir)
    emit(render(report, name="planned_relative"),
         out=args.out, splice=args.splice)
    return 0


if __name__ == "__main__":
    sys.exit(main())
