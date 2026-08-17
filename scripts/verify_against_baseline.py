"""Compare a fresh pipeline run against the read-only July 2025 baseline.

Baseline files are opened read-only; this script never writes to the data
directory. Exit code 0 = equivalent within tolerance; 1 = deviations found.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.common import resolve_data_dir  # noqa: E402

RTOL = 1e-9
ATOL = 1e-12


def compare_neighbor_frames(new_df, base_df):
    """Compare colony IDs, neighbor sets, and neighbor distances.

    Returns a list of human-readable issue strings (empty = equivalent).
    """
    issues = []
    new_ids = set(new_df["USO_AREA_U"])
    base_ids = set(base_df["USO_AREA_U"])
    if new_ids != base_ids:
        only_new = sorted(new_ids - base_ids)[:5]
        only_base = sorted(base_ids - new_ids)[:5]
        issues.append(
            f"colony ID sets differ: {len(only_new)} extra in new (e.g. {only_new}), "
            f"{len(only_base)} missing from new (e.g. {only_base})"
        )
        return issues

    new_idx = new_df.set_index("USO_AREA_U")
    base_idx = base_df.set_index("USO_AREA_U")
    for uso_id in base_idx.index:
        new_nbrs = list(new_idx.at[uso_id, "nbrs_bbox"])
        base_nbrs = list(base_idx.at[uso_id, "nbrs_bbox"])
        if set(new_nbrs) != set(base_nbrs):
            issues.append(
                f"{uso_id}: neighbor sets differ "
                f"(new={sorted(set(new_nbrs))}, baseline={sorted(set(base_nbrs))})"
            )
            continue
        # nbrs_dist_bbox is a list of (neighbor_id, distance) tuples
        # (see calc_nbr_dist in spatial_index_utils.py) — build the lookup
        # directly from the tuples, never by zipping against nbrs_bbox.
        new_dist = dict(new_idx.at[uso_id, "nbrs_dist_bbox"])
        base_dist = dict(base_idx.at[uso_id, "nbrs_dist_bbox"])
        for nbr in base_dist:
            if not np.isclose(new_dist[nbr], base_dist[nbr], rtol=RTOL, atol=ATOL):
                issues.append(
                    f"{uso_id}->{nbr}: neighbor distance differs "
                    f"(new={new_dist[nbr]!r}, baseline={base_dist[nbr]!r})"
                )
    return issues


def compare_numeric_frames(new_df, base_df, id_col, rtol, atol):
    """Compare all shared numeric columns after aligning on id_col.

    Returns (issues, report_lines). report_lines always contains one line per
    compared column with its max absolute deviation.
    """
    issues = []
    report = []
    if len(new_df) != len(base_df):
        issues.append(f"row counts differ: new={len(new_df)}, baseline={len(base_df)}")
    merged = base_df.merge(
        new_df, on=id_col, suffixes=("_base", "_new"), how="inner"
    )
    if len(merged) != len(base_df):
        issues.append(
            f"only {len(merged)} of {len(base_df)} baseline rows matched on {id_col}"
        )
    # Exclude incidental/positional columns: the pandas row index written by
    # to_csv (read back as "Unnamed: 0") and the notebook-era "index" column
    # reflect row order, not computed quantities.
    incidental = {"index", "level_0", ""}
    expected_cols = [
        c
        for c in base_df.columns
        if c != id_col
        and c not in incidental
        and not str(c).startswith("Unnamed")
        and pd.api.types.is_numeric_dtype(base_df[c])
    ]
    # A baseline column absent from the new run is itself a deviation —
    # never silently skip it (a dependency-driven rename/drop would
    # otherwise produce a bogus PASS).
    missing = [c for c in expected_cols if c not in new_df.columns]
    if missing:
        issues.append(f"columns missing from new run: {missing}")
    numeric_cols = [
        c
        for c in expected_cols
        if c in new_df.columns and pd.api.types.is_numeric_dtype(new_df[c])
    ]
    for col in numeric_cols:
        base_vals = merged[f"{col}_base"].to_numpy(dtype=float)
        new_vals = merged[f"{col}_new"].to_numpy(dtype=float)
        both_nan = np.isnan(base_vals) & np.isnan(new_vals)
        close = np.isclose(new_vals, base_vals, rtol=rtol, atol=atol) | both_nan
        with np.errstate(invalid="ignore"):
            max_abs = np.nanmax(np.abs(new_vals - base_vals)) if len(base_vals) else 0.0
        report.append(f"  {col}: max abs deviation {max_abs:.3e}")
        if not close.all():
            n_bad = int((~close).sum())
            issues.append(
                f"{col}: {n_bad} value(s) beyond tolerance (max abs dev {max_abs:.3e})"
            )
    return issues, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="data root holding the baseline")
    parser.add_argument(
        "--verify-dir",
        default=None,
        help="directory holding the fresh run (default: <data-dir>/phase1_verify)",
    )
    args = parser.parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    verify_dir = (
        Path(args.verify_dir).expanduser()
        if args.verify_dir
        else data_dir / "phase1_verify"
    )

    all_issues = []

    print("== Neighbors joblib ==")
    with open(data_dir / "colonies_bbox_nbrs2025.joblib", "rb") as f:
        base_nbrs = joblib.load(f)
    with open(verify_dir / "colonies_bbox_nbrs_aug2026.joblib", "rb") as f:
        new_nbrs = joblib.load(f)
    issues = compare_neighbor_frames(new_nbrs, base_nbrs)
    print(f"  {len(base_nbrs)} baseline colonies; {len(issues)} issue(s)")
    all_issues.extend(issues)

    pairs = [
        (
            "psi popsize",
            data_dir / "psi_2020_results" / "delhi_psi_bbox_popsize2020_norv_12Sep2021.csv",
            verify_dir / "psi_2020_results" / "delhi_psi_bbox_popsize2020_norv_aug2026.csv",
        ),
        (
            "psi popdensity",
            data_dir / "psi_2020_results" / "delhi_psi_bbox_popdensity2020_norv_12Sep2021.csv",
            verify_dir / "psi_2020_results" / "delhi_psi_bbox_popdensity2020_norv_aug2026.csv",
        ),
    ]
    for label, base_path, new_path in pairs:
        print(f"== {label} ==")
        base_df = pd.read_csv(base_path)
        new_df = pd.read_csv(new_path)
        issues, report = compare_numeric_frames(new_df, base_df, "USO_AREA_U", RTOL, ATOL)
        print("\n".join(report))
        all_issues.extend(f"{label}: {i}" for i in issues)

    if all_issues:
        print(f"\nFAIL — {len(all_issues)} deviation(s) from baseline:")
        for issue in all_issues[:50]:
            print(f"  - {issue}")
        sys.exit(1)
    print("\nPASS — new run equivalent to July 2025 baseline within tolerance")


if __name__ == "__main__":
    main()
