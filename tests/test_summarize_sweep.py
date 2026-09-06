"""The sweep statistics, each against an answer computed by hand.

Nothing here reads a real output file. The point of these tests is that the
arithmetic is right; the point of the drift test in Task 6 is that the
document's numbers came from this arithmetic.
"""
import numpy as np
import pandas as pd
import pytest

from scripts import summarize_sweep as S


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
