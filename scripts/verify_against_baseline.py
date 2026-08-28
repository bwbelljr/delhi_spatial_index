"""Compare a fresh pipeline run against the read-only July 2025 baseline.

Baseline files are opened read-only; this script never writes to the data
directory. Exit code 0 = equivalent within tolerance; 1 = deviations found.

--config locates the FRESH files (they follow the profile's
paths.neighbors_artifact and outputs.name_template). The baseline paths stay
this script's own arguments, because they exist only for code-2025 — which is
why there is no `verify` CLI stage and no baseline key in the config schema.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from delhi_psi.config import load_config
from delhi_psi.io import read_neighbors, resolve_data_dir
from delhi_psi.pipeline import output_basename
from delhi_psi.verify import (
    ATOL, RTOL, compare_neighbor_frames, compare_numeric_frames,
)

BASELINE_NEIGHBORS = "colonies_bbox_nbrs2025.joblib"
BASELINE_PSI = {
    "pop": "psi_2020_results/delhi_psi_bbox_popsize2020_norv_12Sep2021.csv",
    "popdensity":
        "psi_2020_results/delhi_psi_bbox_popdensity2020_norv_12Sep2021.csv",
}


def fresh_paths(config, *, verify_dir, data_dir=None):
    """Where the fresh run's files live, per the profile.

    No out_dir: every path below hangs off `verify_dir`, so the config's own
    out_dir never reaches this function's result.
    """
    cfg = load_config(config, data_dir=data_dir)
    verify_dir = Path(verify_dir)
    return {
        "neighbors": verify_dir / cfg.paths.neighbors_artifact,
        "psi": {str(denominator):
                verify_dir / f"{output_basename(cfg, denominator)}.csv"
                for denominator in cfg.outputs.denominators},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code-2025",
                        help="profile that produced the fresh run")
    parser.add_argument("--data-dir", default=None,
                        help="data root holding the baseline")
    parser.add_argument(
        "--verify-dir", default=None,
        help="directory holding the fresh run (default: <data-dir>/phase3_verify)")
    parser.add_argument("--baseline-neighbors", default=None,
                        help=f"default: <data-dir>/{BASELINE_NEIGHBORS}")
    parser.add_argument("--baseline-psi-pop", default=None,
                        help=f"default: <data-dir>/{BASELINE_PSI['pop']}")
    parser.add_argument("--baseline-psi-popdensity", default=None,
                        help=f"default: <data-dir>/{BASELINE_PSI['popdensity']}")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    verify_dir = (Path(args.verify_dir).expanduser() if args.verify_dir
                  else data_dir / "phase3_verify")
    fresh = fresh_paths(args.config, verify_dir=verify_dir,
                        data_dir=args.data_dir)

    baseline_psi = {
        "pop": Path(args.baseline_psi_pop).expanduser()
        if args.baseline_psi_pop else data_dir / BASELINE_PSI["pop"],
        "popdensity": Path(args.baseline_psi_popdensity).expanduser()
        if args.baseline_psi_popdensity
        else data_dir / BASELINE_PSI["popdensity"],
    }
    baseline_neighbors = (Path(args.baseline_neighbors).expanduser()
                          if args.baseline_neighbors
                          else data_dir / BASELINE_NEIGHBORS)

    all_issues = []

    print("== Neighbors artifact ==")
    base_nbrs = read_neighbors(baseline_neighbors)
    new_nbrs = read_neighbors(fresh["neighbors"])
    issues = compare_neighbor_frames(new_nbrs, base_nbrs)
    print(f"  {len(base_nbrs)} baseline colonies; {len(issues)} issue(s)")
    all_issues.extend(issues)

    for denominator, new_path in fresh["psi"].items():
        if denominator not in baseline_psi:
            print(f"== psi {denominator} == (no baseline; skipped)")
            continue
        print(f"== psi {denominator} ==")
        base_df = pd.read_csv(baseline_psi[denominator])
        new_df = pd.read_csv(new_path)
        issues, report = compare_numeric_frames(new_df, base_df, "USO_AREA_U",
                                                RTOL, ATOL)
        print("\n".join(report))
        all_issues.extend(f"psi {denominator}: {i}" for i in issues)

    if all_issues:
        print(f"\nFAIL — {len(all_issues)} deviation(s) from baseline:")
        for issue in all_issues[:50]:
            print(f"  - {issue}")
        sys.exit(1)
    print("\nPASS — new run equivalent to July 2025 baseline within tolerance")


if __name__ == "__main__":
    main()
