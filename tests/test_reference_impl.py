"""Hand-derived anchors (spec 'Canonical numbers') pinning reference_impl.

Every number here is derived on paper from Eq. 1-4 in the manuscript and
double-derived by the spec's ultracode review; the derivation worksheet
(docs/oracle/derivation-worksheet.md) shows the arithmetic.
"""

import math

import pytest

from tests.oraculum_fixtures import (
    load_settlements, load_barriers, load_services,
)
from tests.reference_impl import RULESETS, adjacency, apply_barrier, compute_city

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
