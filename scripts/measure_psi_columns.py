"""Which PSI column and which denominator do the paper's figures report?
(DEL-52, spec § 2.4.)

The April 2026 draft's Figure 4 is a bar chart of the mean PSI per settlement
type, y-axis "Mean Public Services Index (per person per square kilometer)",
eight bars, footnote 12 saying the average "rarely exceeds 0.05". This script
turns the read-off bar values into a measured match: for each of the four
candidates {unnorm_psi, norm_psi} x {popsize, popdensity} it computes the
mean per USO_FINAL type in the July 2025 baseline outputs and scores it
against the figure.

The baseline files are the ones that PRODUCED the figures, so they are the
comparison; --verify-dir is a cross-check that the refactored `code-2025` run
would give the same answer. Both are opened READ-ONLY.

    uv run python scripts/measure_psi_columns.py \
        --baseline-dir ~/delhi_data/psi_2020_results \
        --verify-dir ~/delhi_data/phase3_verify

Prints provenance lines, then one fenced block.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from delhi_psi.config import load_config
from delhi_psi.pipeline import TYPE_COL
from scripts._measure_common import emit, emit_check, render, resolve_work_dir

# Read off Figure 4 ("Mean public service index by settlement", p. 40 of the
# April 2026 draft PDF) on 5 Sep 2026. Eight bars: no RV, no Other. These are
# chart read-offs, not data — hence FIGURE_TOLERANCE (spec § 8 item 6).
FIGURE_4_BARS = {
    "JJR": 0.037,
    "JJC": 0.0015,
    "SDA": 0.028,
    "Planned": 0.044,
    "RUAC": 0.021,
    "UAC": 0.017,
    "UV": 0.038,
    "Industrial": 0.038,
}
FIGURE_TOLERANCE = 0.002
# Fewer than this many matched bars for every candidate means the figure was
# not produced from these columns as-is — an escalation, not a guess.
MATCH_FLOOR = 6

CANDIDATE_COLUMNS = ("unnorm_psi", "norm_psi")
BASELINE_FILES = {
    "popsize": "delhi_psi_bbox_popsize2020_norv_12Sep2021.csv",
    "popdensity": "delhi_psi_bbox_popdensity2020_norv_12Sep2021.csv",
}
# The same two denominators from the refactored code-2025 run (--verify-dir);
# `popsize` is what that profile calls `pop`.
VERIFY_FILES = {
    "popsize": "delhi_psi_code-2025_pop_2020.csv",
    "popdensity": "delhi_psi_code-2025_popdensity_2020.csv",
}
BASELINE_SUBDIR = "psi_2020_results"


def type_means(frame, *, column, type_col=TYPE_COL):
    """Mean of `column` per settlement type, ordered by type name."""
    if column not in frame.columns:
        raise KeyError(f"{column!r} is not a column of this file; it has "
                       f"{sorted(frame.columns)[:20]}")
    grouped = frame.groupby(type_col)[column].mean().sort_index()
    return {str(name): float(value) for name, value in grouped.items()}


def score_candidate(means, *, bars=FIGURE_4_BARS, tolerance=FIGURE_TOLERANCE):
    """(bars matched within `tolerance`, max absolute gap over all eight).

    A figure type absent from the file is unmatched with an INFINITE gap: a
    column that does not even carry the type cannot be what the figure was
    drawn from.
    """
    matched, maxgap = 0, 0.0
    for name, bar in bars.items():
        if name not in means:
            maxgap = float("inf")
            continue
        gap = abs(means[name] - bar)
        maxgap = max(maxgap, gap)
        if gap <= tolerance:
            matched += 1
    return matched, maxgap


def score_candidates(frames, *, columns=CANDIDATE_COLUMNS, type_col=TYPE_COL,
                     bars=FIGURE_4_BARS, tolerance=FIGURE_TOLERANCE):
    """The block: every candidate's per-type means, its score, and the best
    candidate — most bars matched, ties broken by the smaller max gap and
    then by name, so the answer is deterministic."""
    report, scores = {}, {}
    for column in columns:
        for denom, frame in frames.items():
            means = type_means(frame, column=column, type_col=type_col)
            for name, value in means.items():
                report[f"mean_{column}_{denom}_{name}"] = f"{value:.6g}"
            matched, maxgap = score_candidate(means, bars=bars,
                                              tolerance=tolerance)
            report[f"matched_{column}_{denom}"] = matched
            report[f"maxgap_{column}_{denom}"] = f"{maxgap:.4f}"
            scores[f"{column}_{denom}"] = (matched, maxgap)
    report["best_candidate"] = min(
        scores, key=lambda name: (-scores[name][0], scores[name][1], name))
    return report


def cross_check(baseline, verify, *, columns=CANDIDATE_COLUMNS,
                type_col=TYPE_COL, atol=1e-9):
    """The refactored `code-2025` run must give the same per-type means as
    the July 2025 baseline — the cheap proof that this comparison would come
    out the same on the pipeline as it stands today."""
    out = {}
    for column in columns:
        for denom, frame in baseline.items():
            left = type_means(frame, column=column, type_col=type_col)
            right = type_means(verify[denom], column=column, type_col=type_col)
            if set(left) != set(right):
                raise ValueError(
                    f"{column}/{denom}: baseline types {sorted(left)} != "
                    f"verify types {sorted(right)}")
            worst = max(abs(left[name] - right[name]) for name in left)
            if worst > atol:
                raise ValueError(
                    f"{column}/{denom}: the code-2025 run differs from the "
                    f"July 2025 baseline by {worst:.3e} (> {atol:.0e})")
            out[f"verify_maxdiff_{column}_{denom}"] = f"{worst:.1e}"
    return out


def measure(baseline_dir, *, verify_dir=None):
    """The whole report, as an ordered {key: value} mapping (one block)."""
    baseline = {denom: pd.read_csv(Path(baseline_dir) / name)
                for denom, name in BASELINE_FILES.items()}
    report = score_candidates(baseline)
    if verify_dir:
        verify = {denom: pd.read_csv(Path(verify_dir) / name)
                  for denom, name in VERIFY_FILES.items()}
        report.update(cross_check(baseline, verify))
        # keep the answer last, whatever else was appended
        report["best_candidate"] = report.pop("best_candidate")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code-2025",
                        help="profile that locates the data root "
                             "(default code-2025)")
    parser.add_argument("--data-dir", default=None,
                        help="data root, opened READ-ONLY")
    parser.add_argument("--work-dir", default=None,
                        help="scratch; this script writes nothing, but the "
                             "flag is refused inside --data-dir like every "
                             "other measurement script's")
    parser.add_argument("--baseline-dir", default=None,
                        help=f"the July 2025 outputs; default "
                             f"<data-dir>/{BASELINE_SUBDIR}")
    parser.add_argument("--verify-dir", default=None,
                        help="a complete code-2025 run, for the cross-check")
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
                             "and Finding (DEL-59)")
    args = parser.parse_args(argv)

    emit_check(out=args.out, splice=args.splice)

    cfg = load_config(args.config, data_dir=args.data_dir)
    work_dir = resolve_work_dir(args.work_dir, data_dir=cfg.paths.data_dir,
                                prefix="delhi_psi_psi_columns_")
    baseline_dir = (Path(args.baseline_dir).expanduser() if args.baseline_dir
                    else cfg.paths.data_dir / BASELINE_SUBDIR)
    verify_dir = Path(args.verify_dir).expanduser() if args.verify_dir else None

    print(f"baseline-dir: {baseline_dir}", file=sys.stderr)
    print(f"verify-dir: {verify_dir}", file=sys.stderr)
    print(f"work-dir: {work_dir}", file=sys.stderr)
    report = measure(baseline_dir, verify_dir=verify_dir)
    emit(render(report), out=args.out, splice=args.splice)
    best = report["best_candidate"]
    if report[f"matched_{best}"] < MATCH_FLOOR:
        print(f"WARNING: the best candidate ({best}) matches only "
              f"{report[f'matched_{best}']} of {len(FIGURE_4_BARS)} bars — "
              "the figure was not produced from these columns as-is "
              "(spec § 2.4: escalate, do not guess)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
