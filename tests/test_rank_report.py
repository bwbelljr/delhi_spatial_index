"""DEL-35: the rank-based report of one PSI run.

Spec: docs/superpowers/specs/2026-09-07-del-35-rank-report-design.md. This is
extraction, not new statistics (spec § 2): `percentile_rank`, `category_order`,
`decile_set`/`decile_is_gated`, `decile_share` and `bootstrap_rank_intervals`
are DEL-55's, imported from `scripts.summarize_sweep`, not reimplemented here.
The one genuinely new behaviour is the honesty gate spec § 4 demands: on a
tiny population (Oraculum's 7 settlements) a "decile" of k=1 rounds UP from
less than one real observation, and DEL-55's tie-block gate alone does not
catch that (a single untied extreme value never trips an oversized-tie-block
test). `scripts.summarize_sweep.decile_set` was generalised, not duplicated,
to also gate on `fraction * n < 1` — see
`tests/test_summarize_sweep.py::test_decile_set_gates_when_the_unrounded_k_is_below_one`
for the pin on that primitive; the tests below exercise it through the
report.
"""
import json

import pandas as pd
import pytest

from scripts import rank_report as R
from scripts import summarize_sweep as S
from scripts._measure_common import FENCE, parse_block
from tests.cities import ORACULUM

REPO_FIXTURES = ORACULUM.fixtures


def frame(**cols):
    return pd.DataFrame(cols)


# --- the rank transform, hand-computed, including ties ---------------------
def test_percentile_rank_hand_computed_with_ties():
    # values [10, 20, 20, 40] -> Series.rank(method="average") = [1, 2.5,
    # 2.5, 4] (the two 20s split ranks 2 and 3) -> pct = (rank-1)/3*100 =
    # [0, 50, 50, 100].
    got = S.percentile_rank(pd.Series([10.0, 20.0, 20.0, 40.0]))
    assert list(got) == [0.0, pytest.approx(50.0), pytest.approx(50.0), 100.0]


# --- a category's mean (and median) percentile rank, hand-computed ---------
def test_category_mean_and_median_percentile_rank_hand_computed():
    """Ten rows, values 1..10 (no ties) so rank == value's position and
    pct(v) = (v-1)/9*100: 0, 11.111, 22.222, 33.333, 44.444, 55.556,
    66.667, 77.778, 88.889, 100.

    Category A owns the SKEWED rank set {1,2,3,4,10} (pct
    [0, 11.111, 22.222, 33.333, 100]):
      mean  = (0+11.111+22.222+33.333+100)/5 = 166.666/5 = 33.3332 -> "33.3"
      median (middle of the 5 sorted values) = 22.222 -> "22.2"
    Distinguishing mean from median is the point: a contiguous rank block
    would make them equal by symmetry, so A's top rank (10) is placed away
    from its otherwise-contiguous 1..4 block on purpose.

    Category B owns the CONTIGUOUS rank set {5,6,7,8,9} (pct
    [44.444, 55.556, 66.667, 77.778, 88.889]):
      mean = 66.6668 -> "66.7"; median = 66.667 -> "66.7" (equal, as a
      contiguous block always is).

    n=10, fraction=0.10 -> decile_k=1, fraction*n=1.0 (not < 1, so the
    honesty gate from test_summarize_sweep.py does not fire here) and every
    value is distinct (no oversized tie block either), so both decile shares
    are real numbers, not gated: the single lowest value (rank 1, pct 0) and
    the single highest (rank 10, pct 100) both belong to category A, so A's
    top and bottom decile shares are each 1.0 and B's are each 0.0.
    """
    values = list(range(1, 11))
    categories = ["A", "A", "A", "A", "B", "B", "B", "B", "B", "A"]
    f = frame(category=categories, norm_psi=[float(v) for v in values])

    rows, top_gated, bottom_gated = R.category_rows(f)
    assert not top_gated and not bottom_gated
    by_cat = {row["category"]: row for row in rows}

    assert by_cat["A"]["n"] == 5
    assert by_cat["A"]["mean_pct_rank"] == "33.3"
    assert by_cat["A"]["median_pct_rank"] == "22.2"
    assert by_cat["B"]["n"] == 5
    assert by_cat["B"]["mean_pct_rank"] == "66.7"
    assert by_cat["B"]["median_pct_rank"] == "66.7"

    assert by_cat["A"]["top_decile_share"] == "1.000"
    assert by_cat["A"]["bottom_decile_share"] == "1.000"
    assert by_cat["B"]["top_decile_share"] == "0.000"
    assert by_cat["B"]["bottom_decile_share"] == "0.000"

    # rank_ci_lo/hi come straight from DEL-55's own tested
    # bootstrap_rank_intervals — not re-verified numerically here, just
    # checked for shape (present, integral, ordered).
    for row in rows:
        assert isinstance(row["rank_ci_lo"], int)
        assert isinstance(row["rank_ci_hi"], int)
        assert row["rank_ci_lo"] <= row["rank_ci_hi"]


# --- decile gating fires -----------------------------------------------------
def test_decile_gating_fires_on_a_small_population():
    """Seven distinct, untied values: DEL-55's tie-block rule alone would
    never gate this (every tie block has size 1, nowhere near 1.5x a k of
    1), yet 0.10 * 7 = 0.7 < 1 — less than one real observation. A rank
    report that quietly printed "the top decile is 100% G" from a single
    settlement here would be exactly the authoritative-looking nonsense
    spec § 4 warns against, so both deciles must gate."""
    f = frame(category=list("ABCDEFG"),
             norm_psi=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    rows, top_gated, bottom_gated = R.category_rows(f)
    assert top_gated and bottom_gated
    for row in rows:
        assert row["top_decile_share"] == S.DASH
        assert row["bottom_decile_share"] == S.DASH

    summary = R.summary_fields(f)
    assert summary["n_reported"] == 7
    assert summary["decile_k"] == 1
    assert summary["either_decile_gated"] is True


def test_rendered_categories_block_carries_the_dash_not_a_number():
    """The rendered fenced block itself must show the em dash for a gated
    decile share, not a formatted float — the honesty requirement has to
    survive all the way to the text a reader sees."""
    f = frame(category=list("ABCDEFG"),
             norm_psi=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    text = R.render_categories_block(f)
    assert FENCE in text
    body = parse_block(text, name="categories")
    assert body["top_decile_share"] == "—"
    assert body["bottom_decile_share"] == "—"


# --- end-to-end against the committed production fixture -------------------
def _wide_output_csv(tmp_path, *, scenario="baseline", denom="popdensity"):
    """Build a frame in the exact shape `summarize_sweep.load_output_frame`
    reads (spec § 3: "same OUTPUT_USECOLS shape") out of the COMMITTED
    `tests/fixtures/oraculum/production/code-2025.csv` (long format:
    profile,scenario,denom,settlement,metric,value — see
    `scripts/generate_production_fixtures.py`) plus the settlement
    categories from `tests/fixtures/oraculum/settlements.geojson`. No
    external layer and no real-data run: everything here is committed.

    `clinic_*` is renamed to `health_*` — the fixture city's own service
    name for the layer the real `code-2025` profile calls `health`
    (`summarize_sweep.py`'s own top-of-file comment makes the same
    substitution for the same reason).
    """
    long_path = REPO_FIXTURES / "production" / "code-2025.csv"
    long_df = pd.read_csv(long_path)
    pivoted = long_df.pivot(index=["scenario", "denom", "settlement"],
                            columns="metric", values="value")
    wide = pivoted.loc[(scenario, denom)].copy()
    wide = wide.rename(columns={
        f"clinic_{suffix}": f"health_{suffix}"
        for suffix in ("count", "pcen", "idx")})

    settlements = json.loads((REPO_FIXTURES / "settlements.geojson")
                             .read_text())
    category_by_id = {f["properties"]["USO_AREA_U"]: f["properties"]["USO_FINAL"]
                      for f in settlements["features"]}
    wide["category"] = [category_by_id[sid] for sid in wide.index]
    # load_output_frame's OUTPUT_USECOLS also names USO_FINAL (the
    # pre-mapping category column); rank_report never reads it, but
    # pd.read_csv's usecols requires every named column to be present.
    wide["USO_FINAL"] = wide["category"]
    wide.index.name = "USO_AREA_U"

    out = tmp_path / "code-2025_output.csv"
    wide.reset_index().to_csv(out, index=False)
    return out


def test_end_to_end_against_the_oraculum_production_fixture(tmp_path):
    """Oraculum: 7 settlements (A, B, C, D, E, RV, IND), 6 categories
    (Planned owns both A and D; every other category owns exactly one
    settlement) — small enough to check by hand, and exactly the shape
    spec § 4 says must gate rather than pretend a 7-row decile means
    something."""
    csv_path = _wide_output_csv(tmp_path)
    f = S.load_output_frame(csv_path)
    assert len(f) == 7
    assert set(f["category"]) == {"Planned", "UC", "JJC", "RV", "RUAC", "IND"}

    rows, top_gated, bottom_gated = R.category_rows(f)
    assert top_gated and bottom_gated, (
        "0.10 * 7 = 0.7 settlements: the decile must gate")
    by_cat = {row["category"]: row for row in rows}
    assert by_cat["Planned"]["n"] == 2
    for cat in ("UC", "JJC", "RV", "RUAC", "IND"):
        assert by_cat[cat]["n"] == 1
    for row in rows:
        assert row["top_decile_share"] == S.DASH
        assert row["bottom_decile_share"] == S.DASH

    summary = R.summary_fields(f)
    assert summary["n_reported"] == 7
    assert summary["decile_k"] == 1
    # only settlement C is exactly at norm_psi == 0 in this scenario (see
    # the fixture: sorted norm_psi is [0.0, 0.257, 0.284, 0.308, 0.345,
    # 0.870, 1.0] — a single value, not a tie).
    assert summary["n_psi_tied_at_zero"] == 1
    assert summary["either_decile_gated"] is True

    text = "\n".join([R.render_categories_block(f), R.render_summary_block(f)])
    # one fenced block per category (6, Oraculum's category count) plus one
    # summary block.
    assert text.count(FENCE) == 6 + 1
    assert parse_block(text, name="summary")["either_decile_gated"] == "True"
