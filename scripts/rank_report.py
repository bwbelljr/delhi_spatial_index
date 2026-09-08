"""DEL-35: a rank-based report of ONE run (spec
docs/superpowers/specs/2026-09-07-del-35-rank-report-design.md).

This is reading (b) of the ticket's ambiguity (spec § 1): Eq. 1 is
unchanged, and this script only SUMMARISES an already-computed run by rank
rather than by level — mean percentile rank per settlement category, and
each category's share of the top and bottom deciles (as the WORKPLAN's own
intergenerational-mobility analogy describes: rank outcomes, report a
transition-like composition, do not build the outcome out of ranks).

EXTRACTION, NOT NEW STATISTICS (spec § 2). DEL-55 already built and tested
every statistic this report needs, inside `scripts/summarize_sweep.py`:
`percentile_rank`, `category_order`, `decile_set`/`decile_is_gated`,
`decile_share`, `bootstrap_rank_intervals`. All imported here, none
reimplemented. `decile_k` and the small-`n` addition to `decile_set`'s
gating rule (both in `summarize_sweep.py`) were added FOR this ticket, but
live there rather than here, because DEL-55's existing callers share the
same tie-inclusive decile rule and two copies of it is exactly the drift
this repo has spent the week removing (spec § 3).

`_fmt` is imported too, despite the underscore: it is the one place a
`None`/NaN value renders the em dash, and every value this report gates (a
decile share on a run too small to have one) needs exactly that renderer —
reimplementing "format to n dp, else em dash" a second time in this file
would be the identical kind of duplication the reuse-not-reimplement
instruction is about, just for a formatter instead of a statistic.

THE HONEST PART (spec § 4): Oraculum's 7 settlements make a decile
0.7 settlements — less than one real observation. `decile_set`'s
generalised gate (added alongside this ticket) catches that case
independently of whether the extreme values happen to tie, so a report run
on a population this small renders `—` for both decile-share columns
rather than a number that describes one arbitrary settlement as if it were
a decile.
"""

import argparse

from scripts._measure_common import emit, render
from scripts.summarize_sweep import (
    PSI_COL, _fmt, bootstrap_rank_intervals, category_order, decile_is_gated,
    decile_k, decile_share, load_output_frame, percentile_rank,
)

CATEGORY_COL = "category"
BLOCKS = ("categories", "summary")


def category_rows(frame, *, psi_col=PSI_COL, category_col=CATEGORY_COL,
                  seed=0, n=1000):
    """One row per category (spec § 3): `n`, the mean and median percentile
    rank (1 dp), the top/bottom decile share (3 dp, `—` when gated), and the
    bootstrap rank interval on the category's mean rank (seed 0, 1,000
    draws by default, matching `bootstrap_rank_intervals`'s own defaults).

    Returns `(rows, top_gated, bottom_gated)` — the two gate flags are
    returned alongside the rows (rather than making a caller re-derive them
    from a `—` string) because `render_summary_block` needs the identical
    flags for its own `either_decile_gated` field, and computing them twice
    from two different code paths is exactly the kind of drift this ticket
    exists to avoid.
    """
    order = category_order(frame, psi_col, category_col=category_col)
    pct = percentile_rank(frame[psi_col])
    bootstrap = bootstrap_rank_intervals(frame, psi_col=psi_col,
                                         category_col=category_col,
                                         seed=seed, n=n)
    top_gated = decile_is_gated(frame[psi_col], top=True)
    bottom_gated = decile_is_gated(frame[psi_col], top=False)

    rows = []
    for cat in order:
        mask = frame[category_col] == cat
        cat_pct = pct[mask]
        lo, hi = bootstrap["ci"][cat]
        top_share = decile_share(frame[psi_col], frame[category_col], cat,
                                 top=True)
        bottom_share = decile_share(frame[psi_col], frame[category_col], cat,
                                    top=False)
        rows.append({
            "category": cat,
            "n": int(mask.sum()),
            "mean_pct_rank": _fmt(cat_pct.mean(), 1),
            "median_pct_rank": _fmt(cat_pct.median(), 1),
            "top_decile_share": _fmt(top_share, 3),
            "bottom_decile_share": _fmt(bottom_share, 3),
            "rank_ci_lo": lo,
            "rank_ci_hi": hi,
        })
    return rows, top_gated, bottom_gated


def summary_fields(frame, *, psi_col=PSI_COL):
    """The one summary line (spec § 3): `n_reported`, `decile_k`,
    `n_psi_tied_at_zero`, and whether either decile gated."""
    top_gated = decile_is_gated(frame[psi_col], top=True)
    bottom_gated = decile_is_gated(frame[psi_col], top=False)
    return {
        "n_reported": len(frame),
        "decile_k": decile_k(len(frame)),
        "n_psi_tied_at_zero": int((frame[psi_col] == 0).sum()),
        "either_decile_gated": bool(top_gated or bottom_gated),
    }


def render_categories_block(frame, *, psi_col=PSI_COL,
                            category_col=CATEGORY_COL, seed=0, n=1000):
    rows, *_ = category_rows(frame, psi_col=psi_col, category_col=category_col,
                             seed=seed, n=n)
    return "\n".join(render(row, name="categories") for row in rows)


def render_summary_block(frame, *, psi_col=PSI_COL):
    return render(summary_fields(frame, psi_col=psi_col), name="summary")


# --- CLI ---------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        prog="rank_report",
        description="DEL-35: a rank-based report of one PSI run — mean "
                    "percentile rank per settlement category and each "
                    "category's share of the top/bottom deciles. Eq. 1 is "
                    "unchanged; this only reports an already-computed run "
                    "by rank rather than by level.")
    parser.add_argument("csv", help="one pipeline output CSV, in the same "
                                    "OUTPUT_USECOLS shape "
                                    "summarize_sweep.load_output_frame reads")
    parser.add_argument("--seed", type=int, default=0,
                        help="bootstrap seed (default: 0)")
    parser.add_argument("--bootstrap-n", type=int, default=1000,
                        help="bootstrap draw count (default: 1000)")
    parser.add_argument("--out", default=None,
                        help="write the blocks here instead of stdout — "
                             "BLOCKS ONLY, OVERWRITES any prose in the "
                             "target; use --splice for a committed document")
    parser.add_argument("--splice", default=None,
                        help="refresh the blocks INSIDE this committed "
                             "document in place, preserving every caption "
                             "and Finding (DEL-58)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    frame = load_output_frame(args.csv)
    text = "\n".join([
        render_categories_block(frame, seed=args.seed, n=args.bootstrap_n),
        render_summary_block(frame),
    ])
    emit(text, out=args.out, splice=args.splice)


if __name__ == "__main__":
    main()
