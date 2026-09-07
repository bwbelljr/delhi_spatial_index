"""The sweep statistics, each against an answer computed by hand.

Nothing above this line's helpers reads a real output file. The point of
those tests is that the arithmetic is right; the point of the rendering and
real-data tests below (Task 6) is that the document's numbers came from
this arithmetic, and that the machinery survives contact with the actual
partial sweep in `~/psi_sweep` / `~/delhi_data/phase3_verify`.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import summarize_sweep as S
from scripts._measure_common import _blocks

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "data" / "phase6_sweep.md"

# Reused rather than re-declared: `tests/test_measure_common.py` already
# resolves `DATA_DIR` from `DELHI_DATA_DIR` (default `~/delhi_data`) and
# skips real-data tests when it is absent. This module adds the ONE thing
# that module has no reason to know about: where the (still in-flight)
# sweep run's manifests and output CSVs live.
from tests.test_measure_common import DATA_DIR  # noqa: E402

SWEEP_DIR = Path(os.environ.get("DELHI_PSI_SWEEP_DIR", "~/psi_sweep")).expanduser()
BASELINE_DIR = DATA_DIR / "phase3_verify"

needs_sweep_data = pytest.mark.skipif(
    not (SWEEP_DIR.exists() and BASELINE_DIR.exists()),
    reason=f"real sweep output not present at {SWEEP_DIR} and/or "
           f"{BASELINE_DIR}")


def frame(**cols):
    return pd.DataFrame(cols)


def test_the_popdensity_denominator_is_population_over_area():
    f = frame(population=[100.0, 50.0], area_km2=[2.0, 0.5])
    assert list(S.denominator_values(f, "popdensity")) == [50.0, 100.0]
    assert list(S.denominator_values(f, "pop")) == [100.0, 50.0]


def test_own_share_is_own_over_own_plus_neighbour():
    # one service; own = 2 over a denominator of 50 -> own_pcen 0.04.
    # pcen 0.10 means the neighbour term contributed 0.06.
    f = frame(population=[100.0], area_km2=[2.0],
              bank_count=[2.0], bank_pcen=[0.10])
    got = S.own_share(f, "popdensity", columns=[("bank_count", "bank_pcen")])
    assert got.iloc[0] == pytest.approx(0.4)


def test_own_share_of_one_when_there_is_no_neighbour_term():
    f = frame(population=[100.0], area_km2=[2.0],
              bank_count=[2.0], bank_pcen=[0.04])
    got = S.own_share(f, "popdensity", columns=[("bank_count", "bank_pcen")])
    assert got.iloc[0] == pytest.approx(1.0)


def test_an_own_share_above_one_is_refused_not_reported():
    """The load-bearing self-check of spec § 6.2: the neighbour term cannot be
    negative, so a share above 1 means the denominator reconstruction is wrong
    and every statistic built on it is wrong too."""
    f = frame(population=[100.0], area_km2=[2.0],
              bank_count=[2.0], bank_pcen=[0.01])
    with pytest.raises(ValueError, match="own_share"):
        S.own_share(f, "popdensity", columns=[("bank_count", "bank_pcen")])


def test_own_share_is_nan_not_zero_when_pcen_is_zero():
    """A settlement owning nothing and receiving nothing has no share to
    report: 0/0 is NaN, never 0 (0 would drag `own_share_p50` toward the
    `smoothed` flag on every point — spec § 6.2)."""
    f = frame(population=[100.0], area_km2=[2.0],
              bank_count=[0.0], bank_pcen=[0.0])
    got = S.own_share(f, "popdensity", columns=[("bank_count", "bank_pcen")])
    assert pd.isna(got.iloc[0])


def test_own_only_psi_is_minmax_of_own_counts_summed():
    # two settlements, two services, denominator 1 -> own pcens are the counts.
    # bank: [0, 4] -> idx [0, 1]; school: [3, 3] -> min == max.
    f = frame(population=[1.0, 1.0], area_km2=[1.0, 1.0],
              bank_count=[0.0, 4.0])
    got = S.own_only_psi(f, "popdensity")
    # one service present -> Eq. 1's mean over `*_idx` is just that column's
    # minmax, [0, 1]; the second minmax leaves an already-[0,1] series alone.
    assert list(got) == [0.0, 1.0]

    # assert the constant-column case raises the same ValueError Eq. 2 does
    f2 = frame(population=[1.0, 1.0], area_km2=[1.0, 1.0],
               bank_count=[0.0, 4.0], school_count=[3.0, 3.0])
    with pytest.raises(ValueError, match="min-max"):
        S.own_only_psi(f2, "popdensity")


def test_percentile_rank_is_zero_to_hundred_and_average_ties():
    got = S.percentile_rank(pd.Series([1.0, 2.0, 2.0, 4.0]))
    assert list(got) == [0.0, pytest.approx(50.0), pytest.approx(50.0), 100.0]


def test_category_order_sorts_by_mean_percentile_rank_descending():
    # psi = [10, 20, 1, 2], no ties -> ranks 3,4,1,2 (1-indexed) ->
    # percentiles (r-1)/3*100 = [66.667, 100.0, 0.0, 33.333].
    # category A = rows 0,1 -> mean (66.667+100)/2 = 83.333
    # category B = rows 2,3 -> mean (0+33.333)/2 = 16.667 -> A above B.
    f = frame(category=["A", "A", "B", "B"], psi=[10.0, 20.0, 1.0, 2.0])
    assert S.category_order(f, "psi") == ["A", "B"]


def test_cliffs_delta_is_one_when_every_x_beats_every_y():
    assert S.cliffs_delta([3, 4, 5], [1, 2]) == 1.0
    assert S.cliffs_delta([1, 2], [3, 4, 5]) == -1.0
    assert S.cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_cliffs_delta_reads_as_a_probability():
    # (delta + 1) / 2 == P(a random x outranks a random y), ties at half
    x, y = [1.0, 3.0], [2.0, 4.0]
    assert (S.cliffs_delta(x, y) + 1) / 2 == pytest.approx(0.25)


def test_cohens_d_on_a_known_pair():
    # x = [2, 4, 6] (mean 4, var ddof=1 = ((2-4)^2+(4-4)^2+(6-4)^2)/2 = 4)
    # y = [1, 2, 3] (mean 2, var ddof=1 = ((1-2)^2+(2-2)^2+(3-2)^2)/2 = 1)
    # pooled var = ((3-1)*4 + (3-1)*1) / (3+3-2) = (8+2)/4 = 2.5
    # d = (4 - 2) / sqrt(2.5) = 2 / 1.5811388300841898 = 1.2649110640673518
    got = S.cohens_d([2, 4, 6], [1, 2, 3])
    assert got == pytest.approx(2 / (2.5 ** 0.5))
    assert got == pytest.approx(1.2649110640673518)


def test_kendall_tau_of_a_reversed_ordering_is_minus_one():
    order = ["Planned", "UAC", "JJC"]
    assert S.kendall_tau_order(order, order) == 1.0
    assert S.kendall_tau_order(order, list(reversed(order))) == -1.0


def test_kendall_tau_order_restricts_to_the_common_categories():
    """A category absent from one run's ordering is dropped, not an error."""
    a = ["Planned", "UAC", "JJC", "Other"]
    b = ["JJC", "Planned", "UAC"]  # "Other" absent here
    # common, in a's order: [Planned, UAC, JJC] -> positions in a: [0,1,2];
    # positions in b: Planned=1, UAC=2, JJC=0 -> ys=[1,2,0].
    # pairs (0,1): xa 0<1, ya 1<2 concordant. (0,2): xa 0<2, ya 1>0 discordant.
    # (1,2): xa 1<2, ya 2>0 discordant. C=1, D=2, n0=3, no ties ->
    # tau = (1-2)/3 = -1/3.
    got = S.kendall_tau_order(a, b)
    assert got == pytest.approx(-1 / 3)


def test_the_bootstrap_is_seeded_and_reproducible():
    # A and B's raw values overlap (0.55-0.9 vs 0.45-0.85), so a resample
    # can plausibly flip which of the two has the higher mean rank; C's
    # values (0.1-0.2) never overlap either, so C's rank is a point mass.
    f = frame(category=["A", "A", "A", "B", "B", "B", "C", "C", "C"],
              norm_psi=[0.9, 0.6, 0.55, 0.85, 0.5, 0.45, 0.2, 0.15, 0.1])
    got1 = S.bootstrap_rank_intervals(f, seed=0, n=200)
    got2 = S.bootstrap_rank_intervals(f, seed=0, n=200)
    assert got1["order"] == got2["order"] == ["A", "B", "C"]
    assert got1["ci"] == got2["ci"]
    assert np.array_equal(got1["draws"], got2["draws"])

    # A different seed draws a different random stream, so the 200-draw
    # rank matrix differs (verified empirically for this frame/seed pair;
    # not a mathematical certainty for arbitrary data, but this data's A/B
    # overlap makes an all-200-draws coincidence astronomically unlikely).
    got3 = S.bootstrap_rank_intervals(f, seed=1, n=200)
    assert not np.array_equal(got1["draws"], got3["draws"])

    # C never overlaps A or B's raw values, so its rank is a point mass at 3
    # in every possible resample.
    assert got1["ci"]["C"] == (3, 3)
    assert got1["point_rank"] == {"A": 1, "B": 2, "C": 3}


def test_the_bootstrap_breaks_ties_randomly_not_by_category_order():
    """The tie-break inside a resample must be a per-draw coin flip, not a
    fixed function of which category's block was concatenated first.

    Two categories, A and B, share EVERY value (all zero) — so within any
    draw the only way to say which one "wins" is arbitrary, and a fair
    tie-break should split roughly 50/50 across many draws. Run against the
    UNFIXED code (stable argsort over columns concatenated in a fixed,
    point-estimate order) this printed:
        category_order: ['A', 'B']
        A got rank 2 in 20000 / 20000 draws
        B got rank 2 in 0 / 20000 draws
    i.e. the category listed earlier (A, whose block sorts first) lost the
    tie in literally every draw — not sampling noise, a deterministic
    artifact of column order. A genuine random tie-break must land far from
    that 100/0 split.
    """
    f = frame(category=["A"] * 50 + ["B"] * 50, norm_psi=[0.0] * 100)
    got = S.bootstrap_rank_intervals(f, seed=0, n=20000)
    first = got["categories"][0]
    i_first = got["categories"].index(first)
    frac_first_gets_worse_rank = (got["draws"][:, i_first] == 2).mean()
    assert 0.3 < frac_first_gets_worse_rank < 0.7


def test_bootstrap_p_greater_is_one_when_a_strictly_dominates_b():
    """Hand-computable extreme: A's values are all above B's, so every
    resampled mean of A (with replacement, from {0.9, 0.8}) is at least 0.8,
    and every resampled mean of B (from {0.2, 0.1}) is at most 0.2 — A's
    resampled mean exceeds B's on every one of the n draws, for any seed."""
    f = frame(category=["A", "A", "B", "B"], norm_psi=[0.9, 0.8, 0.2, 0.1])
    got = S.bootstrap_p_greater(f, "A", "B", seed=0, n=200)
    assert got == 1.0


def test_decile_jaccard_of_a_frame_with_itself_is_one():
    s = pd.Series(np.linspace(0.0, 1.0, 20))
    assert S.decile_jaccard(s, s, top=True) == pytest.approx(1.0)
    assert S.decile_jaccard(s, s, top=False) == pytest.approx(1.0)


def test_a_tie_block_at_the_cut_is_taken_whole():
    """Membership must be a property of the numbers. With values
    [0,0,0,0,0,0,0,0,1,2] and k=1, the bottom decile is all EIGHT zeros, not
    whichever one pandas happened to sort first."""
    got, gated = S.decile_set(pd.Series([0]*8 + [1, 2]), top=False)
    assert len(got) == 8


def test_an_oversized_tie_block_gates_the_cell():
    """Eight rows is 8x the decile of 1 — past 1.5x, so every statistic built
    on this set renders an em dash rather than a number measuring sort order."""
    _, gated = S.decile_set(pd.Series([0]*8 + [1, 2]), top=False)
    assert gated
    assert S.decile_jaccard(pd.Series([0]*8 + [1, 2]),
                            pd.Series([0]*8 + [1, 2]), top=False) is None


def test_the_gate_multiplier_is_1_5x_specifically():
    """Pins the 1.5x constant itself, not just the two real-data shapes
    (ratio 1.09, not gated; ratio 4.44, gated) that pass under 1.5x OR a
    looser constant like 2.0x alike. 1,000 rows, 170 tied at the cut against
    a decile of k = int(0.10 * 1000) = 100 -> ratio 1.7: past 1.5x (150) but
    under 2.0x (200), so this case gates under the documented rule and would
    NOT gate under a 2.0x rule — the one case that tells the two apart."""
    s = pd.Series([0.0] * 170 + list(np.linspace(0.1, 1.0, 830)))
    got, gated = S.decile_set(s, top=False)
    assert len(got) == 170
    assert gated


def test_the_gate_multiplier_is_pinned_from_BELOW_as_well():
    """The 1.7 case above only catches a LOOSER constant. A stricter one —
    1.2x, say — passes it too, so on its own the pin is one-sided and the
    suite would accept a rule that gates cells the spec says to report.
    (Found by the fix-round re-review: mutating 1.5 -> 1.2 broke nothing.)

    130 tied at the cut against a decile of k = 100 -> ratio 1.3: past 1.2x
    (120) but under 1.5x (150), so it must NOT gate under the documented rule
    and WOULD gate under a stricter one. With the case above, the constant is
    now bracketed on both sides."""
    s = pd.Series([0.0] * 130 + list(np.linspace(0.1, 1.0, 870)))
    got, gated = S.decile_set(s, top=False)
    assert len(got) == 130
    assert not gated


def test_the_real_baseline_shape_ties_but_does_not_gate():
    """The two real shapes, and they land on opposite sides of the gate.

    Baseline: 452 rows at norm_psi == 0 against a decile of 413. Tie-inclusive
    gives a well-defined set of 452 — 9 % oversized, under the 1.5x gate, so it
    is REPORTED. The rule that matters here is tie-inclusion, not gating: it is
    what stops nsmallest(413) from picking 413 of the 452 by sort order.
    """
    baseline = pd.Series([0.0]*452 + list(np.linspace(0.1, 1.0, 4131 - 452)))
    got, gated = S.decile_set(baseline, top=False)
    assert len(got) == 452 and not gated
    assert not S.decile_set(baseline, top=True)[1]


def test_the_own_only_anchor_shape_gates():
    """Own-only: 1,834 of 4,131 rows own nothing, so 44 % of the universe sits
    at exactly 0. A "decile" of 1,834 against k=413 is 4.4x — past the gate,
    and the cell renders an em dash."""
    anchor = pd.Series([0.0]*1834 + list(np.linspace(0.1, 1.0, 4131 - 1834)))
    got, gated = S.decile_set(anchor, top=False)
    assert len(got) == 1834 and gated


def test_kendall_tau_b_differs_from_tau_a_when_there_are_ties():
    """An implementation that silently computes tau-a passes every tie-free
    case. Hand-computed: x = [1,1,2,3], y = [1,2,2,3].
    Pairs: 6 total; concordant 4, discordant 0, 1 tied in x only,
    1 tied in y only -> tau_b = 4 / sqrt(5 * 5) = 0.8, while tau_a = 4/6.
    """
    got = S.kendall_tau_b(pd.Series([1, 1, 2, 3]), pd.Series([1, 2, 2, 3]))
    assert got == pytest.approx(0.8)
    assert got != pytest.approx(4 / 6)


def test_spearman_rho_averages_tied_ranks():
    # x = [1, 1, 3, 4] has a tie at positions 0,1 (value 1) -> average rank
    # 1.5 each: rx = [1.5, 1.5, 3, 4]. y = [1, 2, 3, 4] is tie-free: ry =
    # [1, 2, 3, 4]. Both have mean 2.5.
    # cov (sum of centred products) = (-1)(-1.5)+(-1)(-0.5)+(0.5)(0.5)+(1.5)(1.5)
    #   = 1.5 + 0.5 + 0.25 + 2.25 = 4.5
    # var(rx) (sum of squared deviations) = 1+1+0.25+2.25 = 4.5
    # var(ry) = 2.25+0.25+0.25+2.25 = 5.0
    # rho = 4.5 / sqrt(4.5 * 5.0) = 4.5 / sqrt(22.5) = 0.9486832980505138
    got = S.spearman_rho(pd.Series([1, 1, 3, 4]), pd.Series([1, 2, 3, 4]))
    assert got == pytest.approx(4.5 / (22.5 ** 0.5))
    assert got == pytest.approx(0.9486832980505138)


def test_flags_fire_on_the_documented_conditions():
    assert S.flags({"own_share_p50": 0.05}) == ("smoothed",)
    assert S.flags({"n_isolates": 3}, baseline_isolates=2) == ("isolates",)
    assert S.flags({"n_at_psi1": 1, "p99_psi": 0.3}) == ("pinned",)
    assert S.flags({"rho_vs_own": 0.4, "n_fragile_pairs": 3}) == ("reshuffled",)
    # a clean row raises no flags
    assert S.flags({}) == ()
    # multiple flags fire together, in documented order
    assert (S.flags({"n_isolates": 3, "own_share_p50": 0.05},
                    baseline_isolates=2)
            == ("isolates", "smoothed"))
    # "pinned" requires BOTH conditions: one settlement pinned at 1 is not
    # enough on its own if the 99th percentile is not also low.
    assert S.flags({"n_at_psi1": 1, "p99_psi": 0.9}) == ()
    # "reshuffled" requires BOTH a low rho and enough fragile pairs
    assert S.flags({"rho_vs_own": 0.4, "n_fragile_pairs": 2}) == ()


def test_isolates_flag_is_relative_to_the_baseline_not_absolute():
    """Corrected 6 Sep 2026 (spec commit 67e3f24): the measured bbox
    baseline itself has 360 isolated settlements out of 4,131 reported, an
    artifact of code-2025's `global_asymmetric` barrier rule (it severs
    every link INTO a flagged settlement) with nothing to do with any swept
    factor. `n_isolates > 0` would therefore fire on every point including
    both anchors — a constant, not a flag — so the flag is instead
    STRICTLY more isolates than the baseline."""
    # more isolates than the baseline -> flags (e.g. band-0km's real 697
    # against the real baseline's 360)
    assert S.flags({"n_isolates": 697}, baseline_isolates=360) == ("isolates",)
    # fewer than the baseline -> does not flag
    assert S.flags({"n_isolates": 100}, baseline_isolates=360) == ()
    # exactly equal to the baseline -> does not flag: the comparison is
    # strict, so the baseline itself (and any point matching it exactly)
    # never flags on its own count.
    assert S.flags({"n_isolates": 360}, baseline_isolates=360) == ()


def test_isolates_flag_is_not_evaluated_without_a_baseline():
    """`baseline_isolates=None` means the comparison has no basis — the
    own-only anchor has no neighbourhood at all, and a point whose own
    artifact was unreadable has `n_isolates: None` (`run_sweep.py`'s
    `degree_from: "artifact unreadable: ..."` case). Either way this is "no
    flag", not "False" pretending to be an answer, and it must not raise
    even when `n_isolates` is itself huge or None."""
    assert S.flags({"n_isolates": 10_000}) == ()  # baseline_isolates default
    assert S.flags({"n_isolates": 10_000}, baseline_isolates=None) == ()
    assert S.flags({"n_isolates": None}, baseline_isolates=None) == ()
    assert S.flags({}) == ()  # n_isolates absent entirely


def test_category_area_and_swing_is_hand_computed():
    """Fix round item 1: the DEL-52 mechanism ("popdensity's denominator
    rewards large-area settlements") made reproducible. Three categories,
    six rows: `pop_frame`'s `norm_psi` is already sorted [1..6], so its
    percentile ranks are 0,20,40,60,80,100 and A/B/C's MEAN pop percentiles
    are 10, 50, 90. `density_frame` reorders the SAME six values as
    [6,5,1,3,4,2], whose percentiles are 100,80,0,40,60,20, giving mean
    density percentiles of A=90, B=20, C=40. Swing (density-pop) is then
    A: 90-10=80, B: 20-50=-30, C: 40-90=-50 — hand-computable without the
    helper under test.
    """
    pop = frame(category=["A", "A", "B", "B", "C", "C"],
               norm_psi=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
               area_km2=[1.0, 3.0, 10.0, 20.0, 100.0, 200.0])
    density = frame(category=["A", "A", "B", "B", "C", "C"],
                    norm_psi=[6.0, 5.0, 1.0, 3.0, 4.0, 2.0],
                    area_km2=[1.0, 3.0, 10.0, 20.0, 100.0, 200.0])
    got = S.category_area_and_swing(pop, density)
    assert got.loc["A", "swing"] == pytest.approx(80.0)
    assert got.loc["B", "swing"] == pytest.approx(-30.0)
    assert got.loc["C", "swing"] == pytest.approx(-50.0)
    # median area is just the median of each category's two rows.
    assert got.loc["A", "median_area_km2"] == pytest.approx(2.0)
    assert got.loc["B", "median_area_km2"] == pytest.approx(15.0)
    assert got.loc["C", "median_area_km2"] == pytest.approx(150.0)


def test_category_area_and_swing_restricts_to_categories_in_both_frames():
    """A category present in only one frame has no swing to report (its mean
    percentile in the other frame is undefined), so it is dropped rather
    than raising or silently treated as a swing of NaN."""
    pop = frame(category=["A", "A", "B", "B"], norm_psi=[1.0, 2.0, 3.0, 4.0],
               area_km2=[1.0, 1.0, 1.0, 1.0])
    density = frame(category=["A", "A", "C", "C"],
                    norm_psi=[1.0, 2.0, 3.0, 4.0], area_km2=[1.0, 1.0, 1.0, 1.0])
    got = S.category_area_and_swing(pop, density)
    assert list(got.index) == ["A"]


def test_denominator_area_swing_fields_is_hand_computed():
    """Same six-row frame as above: the SIGN of the Spearman correlation
    between median area [2, 15, 150] (A<B<C) and swing [80, -30, -50]
    (A>B>C, decreasing) is unambiguous by inspection — a perfectly ordered
    reversal, without a single tie, so rho is exactly -1 regardless of the
    exact area/swing values, which is what this test pins alongside the
    per-category min/max fields spec § 4.3's follow-up asks for."""
    pop = frame(category=["A", "A", "B", "B", "C", "C"],
               norm_psi=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
               area_km2=[1.0, 3.0, 10.0, 20.0, 100.0, 200.0])
    density = frame(category=["A", "A", "B", "B", "C", "C"],
                    norm_psi=[6.0, 5.0, 1.0, 3.0, 4.0, 2.0],
                    area_km2=[1.0, 3.0, 10.0, 20.0, 100.0, 200.0])
    got = S.denominator_area_swing_fields(pop, density)
    assert got == {
        "spearman_area_vs_swing": "-1.000",
        "median_area_km2_min_swing": "150.000",
        "median_area_km2_max_swing": "2.000",
        "swing_min": "-50.0",
        "swing_min_category": "C",
        "swing_max": "80.0",
        "swing_max_category": "A",
    }


# =========================================================================
# Task 6: rendering, the document, and the real-partial-sweep smoke tests.
# =========================================================================
def test_health_is_the_real_service_name_not_clinic():
    """Regression: `SERVICES`/`AMOUNT_COLUMNS` originally copied
    `scripts/generate_production_fixtures.py`'s FIXTURE-city service name
    ("clinic"), but the real `code-2025` config names that layer "health"
    (`delhi_psi/profiles/code-2025.yaml`'s `layers.point`), and every real
    sweep output CSV carries `health_count`/`health_pcen`/`health_idx`
    columns, never `clinic_*`. Caught only by Task 6's end-to-end smoke run
    against `~/psi_sweep` — no hand-built test frame ever spelled the
    service name out, so this pins it going forward."""
    assert "health" in S.SERVICES
    assert "clinic" not in S.SERVICES
    assert S.AMOUNT_COLUMNS["health"] == "health_count"
    assert "health_pcen" in S.OUTPUT_USECOLS
    assert "clinic_pcen" not in S.OUTPUT_USECOLS


def test_every_block_round_trips_through_the_parser():
    report = {"point": "band-1km", "n_reported": 4131, "own_share_p50": 0.412}
    text = S.render(report, name="points")
    assert S.parse_block(text, name="points") == {k: str(v)
                                                   for k, v in report.items()}


def test_the_document_carries_every_block():
    doc = DOC.read_text()
    for name in S.BLOCKS:
        S.parse_block(doc, name=name)


def test_every_caption_says_the_run_is_provisional():
    """Spec § 6.9: no number from this run may reach the manuscript, and the
    only defence against that is that the document says so at every table.

    Fix round 1 item 3: `doc.count("DRY RUN") >= len(BLOCKS)` (the brief's
    literal snippet) is a threshold, not a per-section check — with 5
    occurrences against a threshold of 4 in the committed document, deleting
    the label from any ONE section still passed. This walks every `## `
    heading, takes the text up to the next heading (or EOF for the last
    one), and requires the label to appear LITERALLY inside that span — so
    removing it from any single section fails, regardless of how many
    other sections still carry it."""
    doc = DOC.read_text()
    headings = list(re.finditer(r"^## .*$", doc, re.M))
    assert len(headings) >= len(S.BLOCKS)
    # The preamble too. The fix-round re-review found that a heading-only walk
    # leaves the introduction — the first thing a reader sees, and the only
    # part of the document a skimmer may read — with no coverage at all:
    # deleting its label passed. A span nobody checks is where the label goes
    # missing.
    assert "DRY RUN" in doc[:headings[0].start()], (
        "the introduction, before the first '## ' heading, is missing the "
        "DRY RUN label")
    for i, heading in enumerate(headings):
        start = heading.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(doc)
        section = doc[start:end]
        assert "DRY RUN" in section, (
            f"{heading.group(0)!r} is missing the DRY RUN label in its own "
            "section")


@needs_sweep_data
def test_a_fresh_run_reproduces_the_committed_blocks():
    """The real-data drift check every other `docs/data/*.md` module in this
    repo has — see `tests/test_measure_roads_access.py::
    test_a_fresh_run_reproduces_the_committed_blocks` — and this one lacked:
    `test_the_document_carries_every_block` (above) asserts nothing beyond
    "this label parses at all", and `test_prose_numbers_come_from_the_blocks`
    (below) checks the document's own prose against the document's own
    blocks, never against a fresh run. This runs the summariser for real and
    asserts every block it prints, label and body together and in order,
    equals what `docs/data/phase6_sweep.md` currently commits.

    SEQUENCING NOTE (fix round, final whole-branch review): this compares a
    fresh run against whatever the document holds RIGHT NOW. If fix-round
    item 1 (`spearman_area_vs_swing` and its five siblings in the
    `denominator_check` block) changed that block's field set, this test
    WILL fail until the controller regenerates and re-splices the document
    after this commit — that is expected, reported in this task's report
    rather than hidden by hand-editing `phase6_sweep.md` here, and is
    exactly what this test existing is for: it should fail the moment the
    document and the script disagree, including right now.
    """
    doc = DOC.read_text()
    committed = [(label, body) for label, body in _blocks(doc)
                if label in S.BLOCKS]

    stdout = _run_summarizer()
    fresh = [(label, body) for label, body in _blocks(stdout)
            if label in S.BLOCKS]

    assert fresh == committed


def _committed_blocks():
    """EVERY block in the document, not just the first per name.

    `docs/data/rule_effects.md`'s own `committed_blocks` takes the first
    match per name because each of its blocks is a single row. Here every
    block type is multi-row (one `points`/`ordering`/`gap` block per sweep
    point, all sharing one label — this task's resolution #4), so limiting
    the guard to the first row per name would make it reject a true prose
    citation of, say, `adj-touch`'s isolate count. Gathering every row is
    what makes the guard meaningful here rather than accidentally narrow."""
    doc = DOC.read_text()
    return [body for label, body in _blocks(doc) if label in S.BLOCKS]


def test_prose_numbers_come_from_the_blocks():
    from tests.test_measure_common import assert_prose_numbers_come_from_the_blocks
    doc = DOC.read_text()
    assert_prose_numbers_come_from_the_blocks(doc, _committed_blocks())


def test_the_document_records_its_provenance():
    doc = DOC.read_text()
    for label in ("**Run date:**", "**Inputs:**", "**Commit:**",
                 "**Command:**"):
        assert label in doc, label


def test_the_out_flags_help_warns_it_overwrites_prose():
    """Fix round item 5: `--out` writes BLOCKS ONLY, and pointing it at
    `docs/data/phase6_sweep.md` deletes every hand-written caption and
    Finding — it already happened once. The help text must say so
    explicitly rather than reading like a harmless stdout redirect."""
    help_text = S.build_parser().format_help()
    assert "OVERWRITES" in help_text
    assert "prose" in help_text


# --- end-to-end smoke test against the real (partial) sweep --------------
def _run_summarizer(*extra_args):
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.summarize_sweep",
         "--work-dir", str(SWEEP_DIR), "--baseline-dir", str(BASELINE_DIR),
         *extra_args],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc.stdout


@needs_sweep_data
def test_points_block_does_not_crash_against_the_real_partial_sweep():
    """The end-to-end smoke check this task's brief asks for. Pins the
    numbers this task's brief states were measured today (2026-09-06)
    against the real data, so a regression in the renderer — not just a
    crash — is caught: the bbox baseline's 21,211 links / 360 isolates /
    4.87 mean degree, and the own-only anchor's 1,834 zero-owning
    settlements (spec § 6.2) and its `jaccard_bottom10` gate (spec § 6.3's
    worked real example)."""
    stdout = _run_summarizer("--block", "points")
    points = {body["point"]: body
             for label, body in _blocks(stdout) if label == "points"}

    baseline = points["baseline"]
    assert baseline["n_reported"] == "4131"
    assert baseline["n_isolates"] == "360"
    assert baseline["n_links"] == "21211"
    assert baseline["deg_mean"] == "4.9"  # 4.868... at 1 dp
    assert baseline["preprocess_s"] == "—"
    assert baseline["compute_s"] == "—"

    own_only = points["own-only"]
    assert own_only["own_share_p50"] == "1.000"
    assert own_only["n_own_share_undef"] == "1834"
    for key in ("n_isolates", "n_links", "deg_mean", "preprocess_s"):
        assert own_only[key] == "—"
    # spec § 6.3's own worked example: the bottom-decile tie block at the
    # own-only anchor (1,834 rows) is 4.4x the decile of 413, past the 1.5x
    # gate, so the cell renders the em dash rather than a number that would
    # measure sort order.
    assert own_only["jaccard_bottom10"] == "—"

    adj_touch = points.get("adj-touch")
    if adj_touch is not None:
        assert adj_touch["n_links"] == "14641"
        assert adj_touch["n_isolates"] == "715"
        # isolates flag is baseline-RELATIVE (715 > 360)
        assert "isolates" in adj_touch["flag"]

    for name, body in points.items():
        assert body.get("status") != "FAILED", (name, body)


@needs_sweep_data
def test_ordering_block_does_not_crash_against_the_real_partial_sweep():
    stdout = _run_summarizer("--block", "ordering")
    rows = [body for label, body in _blocks(stdout) if label == "ordering"]
    assert len(rows) >= 2  # at least the two anchors
    baseline = next(r for r in rows if r["point"] == "baseline")
    assert baseline["seed"] == "0"
    assert baseline["n"] == "1000"
    assert re.match(r"^\d+ \[\d+-\d+\]$", baseline["JJC"])


@needs_sweep_data
def test_gap_block_does_not_crash_against_the_real_partial_sweep():
    """`p_a_gt_b` is measured to be `1.000` on every row of the real partial
    sweep (spec § 6.5's predicted degenerate case at n=4,131) — the
    renderer therefore DROPS it (fix round 1 item 4: `_finalize_gap_rows`)
    and emits one generated note instead, so this pins that the note fires
    for real, not just on a hand-built frame."""
    stdout = _run_summarizer("--block", "gap")
    blocks = [body for label, body in _blocks(stdout) if label == "gap"]
    rows = [b for b in blocks if "note" not in b]
    notes = [b for b in blocks if "note" in b]
    assert len(rows) >= 4  # >= 2 anchors x 2 groups each
    assert all("p_a_gt_b" not in r for r in rows)
    assert len(notes) == 1
    assert "1.000" in notes[0]["note"]
    baseline_planned_jjc = next(r for r in rows if r["point"] == "baseline"
                               and r["group"] == "Planned_vs_JJC")
    # Fix round item 6: `> 0.5` passed against the committed `0.90` and
    # would pass equally well against a regression that dropped it to, say,
    # 0.51 — pin the exact measured value and its bootstrap CI bounds.
    assert baseline_planned_jjc["cliffs_delta"] == "0.90"
    assert baseline_planned_jjc["cliffs_delta_ci_lo"] == "0.88"
    assert baseline_planned_jjc["cliffs_delta_ci_hi"] == "0.92"


def test_finalize_gap_rows_drops_a_constant_p_a_gt_b_and_notes_it():
    """Fix round 1 item 4: the document must stay MECHANICALLY equal to
    this script's stdout, so the column drop spec § 6.5 authorizes has to
    be generated by the renderer, not typed by hand into the doc."""
    rows = [{"point": "a", "group": "Planned_vs_JJC", "p_a_gt_b": "1.000"},
           {"point": "a", "group": "formal_vs_informal", "p_a_gt_b": "1.000"},
           {"point": "b", "group": "Planned_vs_JJC", "p_a_gt_b": "1.000"}]
    got = S._finalize_gap_rows(rows)
    real_rows = got[:-1]
    note_row = got[-1]
    assert real_rows == [{"point": "a", "group": "Planned_vs_JJC"},
                         {"point": "a", "group": "formal_vs_informal"},
                         {"point": "b", "group": "Planned_vs_JJC"}]
    assert set(note_row) == {"note"}
    assert "1.000" in note_row["note"]
    assert "constant" in note_row["note"] or "dropped" in note_row["note"]


def test_finalize_gap_rows_keeps_a_non_constant_p_a_gt_b():
    """If `p_a_gt_b` is ever NOT constant, that is a real signal (spec
    § 6.5) and the column must survive untouched — no note, no drop."""
    rows = [{"point": "a", "p_a_gt_b": "1.000"},
           {"point": "b", "p_a_gt_b": "0.750"}]
    got = S._finalize_gap_rows(rows)
    assert got == rows


def test_finalize_gap_rows_leaves_a_failed_rows_status_alone():
    """A `FAILED` row carries no `p_a_gt_b` at all; it must pass through
    unchanged regardless of what the real rows decide."""
    rows = [{"point": "band-10km", "status": "FAILED"},
           {"point": "a", "p_a_gt_b": "1.000"},
           {"point": "b", "p_a_gt_b": "1.000"}]
    got = S._finalize_gap_rows(rows)
    assert got[0] == {"point": "band-10km", "status": "FAILED"}


@needs_sweep_data
def test_denominator_check_matches_the_measured_disagreement():
    """Measured directly (this task): the baseline's category ordering
    under `pop` and `popdensity` DISAGREE — a real finding for DEL-52, not
    a crash and not a rounding artifact.

    Fix round item 2: the old version of this test asserted only
    `block["agreement"] in ("AGREE", "DISAGREE")` and `block["point"] ==
    "baseline"` — both hardcoded in the producer (`render_denominator_check_
    block` can only ever emit one of those two strings, and always writes
    `"point": "baseline"`), so every field that actually carries this task's
    finding — `tau_pop_vs_popdensity`, both `cliffs_delta_planned_jjc_*`,
    both `cat_order_*` — was parsed and never checked. Pinned here as exact
    strings, measured on 6 Sep 2026 against the real baseline CSVs.
    """
    stdout = _run_summarizer("--block", "denominator_check")
    block = S.parse_block(stdout, name="denominator_check")
    assert block == {
        "point": "baseline",
        "cat_order_pop": "SDA>Other>Planned>UV>UAC>JJC>Industrial>RUAC>JJR",
        "cat_order_popdensity":
            "Other>Industrial>JJR>SDA>UV>Planned>RUAC>UAC>JJC",
        "tau_pop_vs_popdensity": "0.17",
        "cliffs_delta_planned_jjc_pop": "0.22",
        "cliffs_delta_planned_jjc_popdensity": "0.90",
        "agreement": "DISAGREE",
        "spearman_area_vs_swing": "0.933",
        "median_area_km2_min_swing": "0.003",
        "median_area_km2_max_swing": "0.315",
        "swing_min": "-26.0",
        "swing_min_category": "JJC",
        "swing_max": "33.3",
        "swing_max_category": "JJR",
    }


@needs_sweep_data
def test_swapping_the_two_denominator_frames_flips_the_agreement_fields():
    """Fix round item 2: proves the test above is not itself a tautology by
    doing the swap the review asked for and confirming the fields actually
    move. Report (per the fix-round brief) of what the swap printed BEFORE
    this test pinned it, measured directly against the real baseline CSVs:

        cat_order_pop:         (unswapped) Other>Industrial>JJR>SDA>UV>Planned>RUAC>UAC>JJC
        cat_order_popdensity:  (unswapped) SDA>Other>Planned>UV>UAC>JJC>Industrial>RUAC>JJR
        tau_pop_vs_popdensity: 0.17 (Kendall tau is symmetric — unchanged)
        cliffs_delta_planned_jjc_pop: 0.90 (was popdensity's committed value)
        cliffs_delta_planned_jjc_popdensity: 0.22 (was pop's committed value)
        agreement: DISAGREE (unchanged — swapping which side is called
            "pop" and which "popdensity" cannot make two different orderings
            equal)
        spearman_area_vs_swing: -0.933 (swing is antisymmetric in which
            frame is "density" vs "pop", so its sign flips; median_area_*
            and swing_min/max are computed off the SAME six-column swing
            table and flip their category labels along with it)

    Directly monkeypatches `S.load_output_frame` to hand back the pop CSV's
    frame when the popdensity path is requested and vice versa — the same
    substitution the brief describes as swapping the two `load_output_frame`
    calls at the call site, done here without touching production code.
    """
    baseline_dir = Path(BASELINE_DIR).expanduser()
    pop_path = baseline_dir / "delhi_psi_code-2025_pop_2020.csv"
    density_path = baseline_dir / "delhi_psi_code-2025_popdensity_2020.csv"
    real_load = S.load_output_frame

    def swapped_load(path):
        path = Path(path)
        if path == pop_path:
            return real_load(density_path)
        if path == density_path:
            return real_load(pop_path)
        return real_load(path)

    import unittest.mock as mock
    with mock.patch.object(S, "load_output_frame", side_effect=swapped_load):
        text = S.render_denominator_check_block(baseline_dir)
    block = S.parse_block(text, name="denominator_check")

    assert block == {
        "point": "baseline",
        "cat_order_pop": "Other>Industrial>JJR>SDA>UV>Planned>RUAC>UAC>JJC",
        "cat_order_popdensity":
            "SDA>Other>Planned>UV>UAC>JJC>Industrial>RUAC>JJR",
        "tau_pop_vs_popdensity": "0.17",
        "cliffs_delta_planned_jjc_pop": "0.90",
        "cliffs_delta_planned_jjc_popdensity": "0.22",
        "agreement": "DISAGREE",
        "spearman_area_vs_swing": "-0.933",
        "median_area_km2_min_swing": "0.315",
        "median_area_km2_max_swing": "0.003",
        "swing_min": "-33.3",
        "swing_min_category": "JJR",
        "swing_max": "26.0",
        "swing_max_category": "JJC",
    }
