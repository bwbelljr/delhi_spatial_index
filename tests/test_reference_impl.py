"""Hand-derived anchors (spec 'Canonical numbers') pinning reference_impl.

Every number here is derived on paper from Eq. 1-4 in the manuscript and
double-derived by the spec's ultracode review; the derivation worksheet
(docs/oracle/derivation-worksheet.md) shows the arithmetic.
"""

import itertools
import math
from pathlib import Path

import pandas as pd
import pytest

from tests.oraculum_fixtures import (
    load_settlements, load_barriers, load_services,
)
from tests.reference_impl import (
    RULESETS, adjacency, apply_barrier, compute_city, emit_expected_values,
)

CSV = Path(__file__).resolve().parent / "fixtures" / "oraculum" / "expected_values.csv"

SQ2 = math.sqrt(2)
W_SQRT2 = 1 / (1 + SQ2)          # decay at 1000*sqrt(2) m
W_HALF = 0.5                      # decay at 1000 m
W_25 = 1 / 2.5                    # decay at 1500 m


@pytest.fixture(scope="module")
def city():
    return load_settlements()


@pytest.fixture(scope="module")
def barriers():
    return load_barriers()


@pytest.fixture(scope="module")
def services():
    return load_services()


IDEAL = {"A": {"B", "E"}, "B": {"A", "C", "RV", "E"}, "C": {"B", "E", "IND"},
         "RV": {"B"}, "D": {"E"}, "E": {"A", "B", "C", "D", "IND"},
         "IND": {"C", "E"}}
CODE = {"A": {"B", "E"}, "B": {"C", "RV", "E"}, "C": {"B", "E", "IND"},
        "RV": {"B"}, "D": {"E"}, "E": {"B", "C", "IND"}, "IND": {"C", "E"}}


def test_border_adjacency_severed_pairwise(city, barriers):
    nbrs = apply_barrier(adjacency(city, "border"), city, barriers, "pair")
    assert nbrs == IDEAL


def test_bbox_adjacency_with_global_barrier(city, barriers):
    nbrs = apply_barrier(adjacency(city, "bbox"), city, barriers, "global")
    assert nbrs == CODE


def test_bbox_equals_border_pre_barrier_for_rectangles(city):
    assert adjacency(city, "bbox") == adjacency(city, "border")


def _city_df(city, services, barriers, rule, **overrides):
    kwargs = dict(RULESETS[rule], scenario="baseline", denom="pop")
    kwargs.update(overrides)
    return compute_city(city, services, barriers, **kwargs)


def test_clinic_pcen_ideal_baseline_pop(city, services, barriers):
    df = _city_df(city, services, barriers, "ideal")
    exp = {
        "A": (2 + W_HALF + W_SQRT2) / 100,
        "B": 0.0175,
        "C": (W_HALF + W_SQRT2) / 400,
        "RV": 0.025,
        "D": 0.004,
        "E": (1 + 2 * W_SQRT2 + W_HALF) / 300,
        "IND": 0.04,
    }
    for sid, v in exp.items():
        assert df.loc[sid, "clinic_pcen"] == pytest.approx(v, abs=1e-12), sid


def test_clinic_pcen_code_rule_differences(city, services, barriers):
    df = _city_df(city, services, barriers, "code")
    assert df.loc["B", "clinic_pcen"] == pytest.approx(0.0125, abs=1e-12)
    assert df.loc["E", "clinic_pcen"] == pytest.approx(0.005, abs=1e-12)
    assert df.loc["A", "clinic_pcen"] == pytest.approx(
        (2 + W_HALF + W_SQRT2) / 100, abs=1e-12)


def test_school_pcen_ideal_and_unique_anchors(city, services, barriers):
    df = _city_df(city, services, barriers, "ideal")
    exp = {"A": SQ2 / 100, "B": 0.005, "C": (SQ2 - 1) / 400, "RV": 0.0,
           "D": 0.014, "E": (1 + (SQ2 - 1) + 0.4) / 300, "IND": 0.04}
    for sid, v in exp.items():
        assert df.loc[sid, "school_pcen"] == pytest.approx(v, abs=1e-12), sid
    pcen = df["school_pcen"]
    assert pcen.idxmax() == "IND" and (pcen == pcen.max()).sum() == 1
    assert pcen.idxmin() == "RV" and (pcen == pcen.min()).sum() == 1


def test_popdensity_denominator(city, services, barriers):
    df = _city_df(city, services, barriers, "ideal", denom="popdensity")
    # E: pop 300 / area 2.0 -> denominator 150
    assert df.loc["E", "clinic_pcen"] == pytest.approx(
        (1 + 2 * W_SQRT2 + W_HALF) / 150, abs=1e-12)
    # A: area 1.0 -> identical to popsize
    assert df.loc["A", "clinic_pcen"] == pytest.approx(
        (2 + W_HALF + W_SQRT2) / 100, abs=1e-12)


def test_expected_values_csv_complete():
    df = pd.read_csv(CSV)
    assert set(df.columns) == {"rule", "scenario", "denom", "settlement",
                               "metric", "value"}
    for rule, scenario, denom in itertools.product(
            ("ideal", "code"),
            ("baseline", "excl_contributing", "excl_removed",
             "excl_ind_removed", "excl_rv_only"),
            ("pop", "popdensity")):
        sub = df[(df["rule"] == rule) & (df["scenario"] == scenario)
                 & (df["denom"] == denom)]
        assert len(sub) > 0, (rule, scenario, denom)
        assert ("norm_psi" in set(sub["metric"])) == (rule == "code")


def _lookup(df, rule, scenario, denom, settlement, metric):
    m = df[(df["rule"] == rule) & (df["scenario"] == scenario)
           & (df["denom"] == denom) & (df["settlement"] == settlement)
           & (df["metric"] == metric)]
    assert len(m) == 1, (rule, scenario, denom, settlement, metric)
    return float(m["value"].iloc[0])


def test_csv_matches_hand_anchors():
    df = pd.read_csv(CSV)
    assert _lookup(df, "ideal", "baseline", "pop", "B", "clinic_pcen") == \
        pytest.approx(0.0175, abs=1e-12)
    assert _lookup(df, "ideal", "excl_removed", "pop", "B", "clinic_pcen") == \
        pytest.approx(0.0125, abs=1e-12)
    assert _lookup(df, "ideal", "excl_contributing", "pop", "B", "clinic_pcen") == \
        pytest.approx(0.0175, abs=1e-12)
    assert _lookup(df, "ideal", "baseline", "pop", "A", "road_pcen") == \
        pytest.approx(0.0075, abs=1e-12)
    assert _lookup(df, "ideal", "baseline", "popdensity", "E", "road_pcen") == \
        pytest.approx(0.005, abs=1e-12)
    assert _lookup(df, "code", "baseline", "pop", "A", "road_pcen") == \
        pytest.approx(0.010606601717798213, abs=1e-12)
    assert _lookup(df, "code", "baseline", "pop", "IND", "road_pcen") == \
        pytest.approx(0.03, abs=1e-12)


def test_code_excl_contributing_collapses_to_removed():
    """Schema self-consistency: the reference impl's `swallowed` knob makes
    the two scenarios' CSV blocks identical BY CONSTRUCTION. The
    production-facing pin of rule-set gap #5 (the real except:pass swallow)
    lives in tests/test_oracle.py::test_production_collapse_gap5."""
    df = pd.read_csv(CSV)
    a = df[(df["rule"] == "code") & (df["scenario"] == "excl_contributing")]
    b = df[(df["rule"] == "code") & (df["scenario"] == "excl_removed")]
    key = ["denom", "settlement", "metric"]
    merged = a.merge(b, on=key, suffixes=("_a", "_b"))
    assert len(merged) == len(a) == len(b)
    pd.testing.assert_series_equal(
        merged["value_a"], merged["value_b"], check_names=False,
        rtol=0, atol=1e-15)


def test_ideal_excl_contributing_differs_from_removed():
    df = pd.read_csv(CSV)
    va = _lookup(df, "ideal", "excl_contributing", "pop", "B", "clinic_pcen")
    vb = _lookup(df, "ideal", "excl_removed", "pop", "B", "clinic_pcen")
    assert va != pytest.approx(vb, abs=1e-9)


def test_ind_removal_is_pure_renormalization():
    """IND is serviceless: only _idx/psi move, never counts or pcen."""
    df = pd.read_csv(CSV)
    base = df[(df["rule"] == "ideal") & (df["scenario"] == "baseline")]
    ind = df[(df["rule"] == "ideal") & (df["scenario"] == "excl_ind_removed")]
    key = ["denom", "settlement", "metric"]
    merged = base.merge(ind, on=key, suffixes=("_base", "_ind"))
    pcen_rows = merged[merged["metric"].str.endswith(("_pcen", "_count", "_length_km"))]
    pd.testing.assert_series_equal(
        pcen_rows["value_base"], pcen_rows["value_ind"], check_names=False,
        rtol=0, atol=1e-15)
    clinic_idx = merged[merged["metric"] == "clinic_idx"]
    assert (clinic_idx["value_base"] != clinic_idx["value_ind"]).any()


def test_recorded_ties_are_ground_truth():
    df = pd.read_csv(CSV)
    # police tied argmax A/B (ideal, baseline, pop)
    pa = _lookup(df, "ideal", "baseline", "pop", "A", "police_pcen")
    pb = _lookup(df, "ideal", "baseline", "pop", "B", "police_pcen")
    assert pa == pytest.approx(pb, abs=1e-15) == pytest.approx(0.005, abs=1e-12)
    # road Eq.4 tied zero minimum
    for sid in ("B", "C", "RV", "D", "IND"):
        assert _lookup(df, "ideal", "baseline", "pop", sid, "road_pcen") == 0.0


def test_invariants_guard_csv_wide():
    from scripts.check_oraculum_invariants import check
    assert check() == []


def test_expected_values_csv_is_regenerable(tmp_path):
    """The committed CSV must be exactly what reference_impl produces.

    Without this, a red build could be 'fixed' by hand-editing the CSV,
    silently turning the oracle into a record of production behavior
    instead of the equations (code review round 1, Critical).
    """
    regen = tmp_path / "regen.csv"
    emit_expected_values(regen)
    assert regen.read_text() == CSV.read_text()
