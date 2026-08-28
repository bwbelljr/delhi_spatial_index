"""The manuscript profile against Bob's hand-ratified anchors.

Every number below is quoted from docs/oracle/derivation-worksheet.md
(RATIFIED 2026-08-24). No generator can rewrite these — they are the reason
the reference implementation has authority at all.

The worksheet prints irrational PCENs to 8 decimals, so each of those is
asserted twice: against the printed value at 5e-9 (half a unit in the last
printed place) and against the worksheet's own closed-form arithmetic at
1e-12. Terminating decimals are asserted at 1e-12 directly.

The manuscript profile maps to reference rule-set `ideal`; scenario
`baseline` is `exclusion.types: []` with the profile's own
`absent_neighbor: contributes` (spec §§ 4, 7).
"""
import math

import pytest

from tests.oraculum_fixtures import compute_oracle_frame

PROFILE = "manuscript"

# Worksheet, "Map: Decays": 1 km -> 1/2; 1.5 km -> 0.4; sqrt(2) km -> sqrt(2)-1
DECAY_1KM = 0.5
DECAY_1_5KM = 0.4
DECAY_SQRT2 = 1 / (1 + math.sqrt(2))


@pytest.fixture(scope="module")
def baseline_pop():
    return compute_oracle_frame(PROFILE, types=(), stage="post_neighbors",
                                denom="pop")


@pytest.fixture(scope="module")
def baseline_popdensity():
    return compute_oracle_frame(PROFILE, types=(), stage="post_neighbors",
                                denom="popdensity")


def test_the_decay_constants_are_the_worksheets():
    assert DECAY_SQRT2 == pytest.approx(math.sqrt(2) - 1, abs=1e-15)
    assert DECAY_SQRT2 == pytest.approx(0.414214, abs=5e-7)
    assert 1 / (1 + 1.5) == DECAY_1_5KM


# --- "## Clinics (counts: A 2, B 1, E 1, RV 2) — Eq. 3, popsize" -------
CLINIC_EXACT = {"B": 3.5 / 200, "RV": 2.5 / 100, "D": 0.4 / 100,
                "IND": 0.4 / 10}
CLINIC_PRINTED = {"A": 0.02914214, "C": 0.00228553, "E": 0.00776142}
CLINIC_CLOSED_FORM = {
    "A": (2 + 1 * DECAY_1KM + 1 * DECAY_SQRT2) / 100,
    "C": (0 + 1 * DECAY_1KM + 1 * DECAY_SQRT2) / 400,
    "E": (1 + 2 * DECAY_SQRT2 + 1 * DECAY_1KM) / 300,
}


@pytest.mark.parametrize("sid,value", sorted(CLINIC_EXACT.items()))
def test_clinic_pcen_exact_anchors(baseline_pop, sid, value):
    assert baseline_pop.loc[sid, "clinic_pcen"] == pytest.approx(value,
                                                                 abs=1e-12)


@pytest.mark.parametrize("sid", sorted(CLINIC_PRINTED))
def test_clinic_pcen_irrational_anchors(baseline_pop, sid):
    closed = CLINIC_CLOSED_FORM[sid]
    assert closed == pytest.approx(CLINIC_PRINTED[sid], abs=5e-9), \
        "the worksheet's printed value disagrees with its own arithmetic"
    assert baseline_pop.loc[sid, "clinic_pcen"] == pytest.approx(closed,
                                                                 abs=1e-12)


def test_clinic_minmax_anchors_are_c_and_ind_and_unique(baseline_pop):
    """Worksheet: 'Eq. 2 anchors: min = C (0.00228553), max = IND (0.04) —
    both unique.'"""
    pcen = baseline_pop["clinic_pcen"]
    assert pcen.idxmin() == "C" and pcen.idxmax() == "IND"
    assert (pcen == pcen.min()).sum() == 1
    assert (pcen == pcen.max()).sum() == 1


def test_clinic_index_for_a(baseline_pop):
    """Worksheet: 'A_idx = 0.02685661/0.03771447 = 0.71210346
    (CSV: 0.7121034578830464 ... check to ~6 decimals)'."""
    got = baseline_pop.loc["A", "clinic_idx"]
    assert got == pytest.approx(0.71210346, abs=5e-7)
    assert got == pytest.approx(0.7121034578830464, abs=1e-12)


# --- "## Schools (A 1, D 1, E 1) — Eq. 3, popsize" ---------------------
SCHOOL_EXACT = {"B": 1.0 / 200, "RV": 0.0, "D": 1.4 / 100, "IND": 0.4 / 10}
SCHOOL_PRINTED = {"A": 0.01414214, "C": 0.00103553, "E": 0.00604738}
SCHOOL_CLOSED_FORM = {
    "A": (1 + 1 * DECAY_SQRT2) / 100,
    "C": (0 + 1 * DECAY_SQRT2) / 400,
    "E": (1 + 1 * DECAY_SQRT2 + 1 * DECAY_1_5KM) / 300,
}


@pytest.mark.parametrize("sid,value", sorted(SCHOOL_EXACT.items()))
def test_school_pcen_exact_anchors(baseline_pop, sid, value):
    assert baseline_pop.loc[sid, "school_pcen"] == pytest.approx(value,
                                                                 abs=1e-12)


@pytest.mark.parametrize("sid", sorted(SCHOOL_PRINTED))
def test_school_pcen_irrational_anchors(baseline_pop, sid):
    closed = SCHOOL_CLOSED_FORM[sid]
    assert closed == pytest.approx(SCHOOL_PRINTED[sid], abs=5e-9)
    assert baseline_pop.loc[sid, "school_pcen"] == pytest.approx(closed,
                                                                 abs=1e-12)


def test_school_near_tie_between_a_and_d_survives(baseline_pop):
    """Worksheet: 'the deliberate near-tie A vs D (0.014142 vs 0.014)'."""
    assert baseline_pop.loc["A", "school_pcen"] > \
        baseline_pop.loc["D", "school_pcen"]
    assert baseline_pop.loc["A", "school_pcen"] - \
        baseline_pop.loc["D", "school_pcen"] == pytest.approx(0.000142136,
                                                              abs=5e-9)


# --- "## Roads — Eq. 4 literally (NO neighbor term)" -------------------
def test_road_lengths_are_075_km_for_a_and_e(baseline_pop):
    assert baseline_pop.loc["A", "road_length"] == pytest.approx(0.75,
                                                                 abs=1e-12)
    assert baseline_pop.loc["E", "road_length"] == pytest.approx(0.75,
                                                                 abs=1e-12)


def test_road_pcen_pop_is_eq4_with_a_tied_zero_minimum(baseline_pop):
    """'pop: A = 0.75/100 = 0.0075; E = 0.75/300 = 0.0025;
    B = C = RV = D = IND = 0 exactly (tied minimum)'."""
    assert baseline_pop.loc["A", "road_pcen"] == pytest.approx(0.0075,
                                                               abs=1e-12)
    assert baseline_pop.loc["E", "road_pcen"] == pytest.approx(0.0025,
                                                               abs=1e-12)
    for sid in ("B", "C", "RV", "D", "IND"):
        assert baseline_pop.loc[sid, "road_pcen"] == 0.0, sid


def test_road_pcen_popdensity(baseline_popdensity):
    """'popdensity: A = 0.0075; E = 0.75/150 = 0.005'."""
    assert baseline_popdensity.loc["A", "road_pcen"] == pytest.approx(
        0.0075, abs=1e-12)
    assert baseline_popdensity.loc["E", "road_pcen"] == pytest.approx(
        0.005, abs=1e-12)


# --- "## Singleton services (bank@A, police@B, ration@D, transport@E)" -
def test_police_singleton_table(baseline_pop):
    """'B = 1/200 = 0.005; A = 1*1/2/100 = 0.005 (tied argmax);
    C = 0.00125; RV = 0.005 (three-way tie A/B/RV); E = 0.00166667;
    D = 0; IND = 0.'"""
    police = baseline_pop["police_pcen"]
    assert police["B"] == pytest.approx(0.005, abs=1e-12)
    assert police["A"] == pytest.approx(0.005, abs=1e-12)
    assert police["RV"] == pytest.approx(0.005, abs=1e-12)
    assert police["C"] == pytest.approx(0.00125, abs=1e-12)
    assert police["E"] == pytest.approx(0.00166667, abs=5e-9)
    assert police["E"] == pytest.approx(1 * DECAY_1KM / 300, abs=1e-12)
    assert police["D"] == 0.0 and police["IND"] == 0.0


# --- "## Worked extras (complete the anchor subset)" -------------------
def test_extra_1_exclusion_delta_for_b():
    """'B, ideal, excl_removed, pop: (1 + 2*1/2 + 0 + 1*1/2)/200
    = 2.5/200 = 0.0125 (vs 0.0175 baseline - the RV contribution effect,
    -0.005)'."""
    removed = compute_oracle_frame(PROFILE, types=("RV", "IND"),
                                   stage="pre_neighbors", denom="pop")
    assert removed.loc["B", "clinic_pcen"] == pytest.approx(0.0125, abs=1e-12)
    baseline = compute_oracle_frame(PROFILE, types=(),
                                    stage="post_neighbors", denom="pop")
    assert baseline.loc["B", "clinic_pcen"] - removed.loc["B", "clinic_pcen"] \
        == pytest.approx(0.005, abs=1e-12)


def test_extra_2_renormalization_delta_for_a():
    """'A clinic_idx, ideal, excl_ind_removed, pop = 1.0 exactly
    (was 0.71210346) - anchor movement with zero numerator change. This
    delta is denominator-INVARIANT because A, C, IND all have area
    1.0 km^2.'"""
    for denom in ("pop", "popdensity"):
        frame = compute_oracle_frame(PROFILE, types=("IND",),
                                     stage="pre_neighbors", denom=denom)
        assert frame.loc["A", "clinic_idx"] == pytest.approx(1.0, abs=1e-12), \
            denom


def test_extra_3_popdensity_coverage_for_e(baseline_pop, baseline_popdensity):
    """'E clinic, ideal, baseline: popsize 2.328427/300 = 0.00776142;
    popdensity divides by pop/area = 300/2 = 150 -> 2.328427/150
    = 0.01552285'."""
    assert baseline_pop.loc["E", "clinic_pcen"] == pytest.approx(0.00776142,
                                                                 abs=5e-9)
    assert baseline_popdensity.loc["E", "clinic_pcen"] == pytest.approx(
        0.01552285, abs=5e-9)
    assert baseline_popdensity.loc["E", "clinic_pcen"] == pytest.approx(
        (1 + 2 * DECAY_SQRT2 + 1 * DECAY_1KM) / (300 / 2), abs=1e-12)


def test_extra_4_road_eq4_value_for_a(baseline_pop):
    """'Road Eq. 4 value (A, pop): 0.75/100 = 0.0075'."""
    assert baseline_pop.loc["A", "road_pcen"] == pytest.approx(0.0075,
                                                               abs=1e-12)


def test_manuscript_profile_has_no_second_normalization(baseline_pop):
    """second_normalization: false, so the column is absent (spec § 4)."""
    assert "unnorm_psi" in baseline_pop.columns
    assert "norm_psi" not in baseline_pop.columns
