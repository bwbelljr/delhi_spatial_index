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

from tests.cities import CITIES
from tests.oraculum_fixtures import (
    load_settlements, load_barriers, load_services,
)
from tests.reference_impl import (
    RULESETS, adjacency, apply_barrier, compute_city, emit_expected_values,
    emit_variant_expected_values,
)
from tests.variants import BAND_RADII_KM, VARIANTS

CSV = Path(__file__).resolve().parent / "fixtures" / "oraculum" / "expected_values.csv"

SQ2 = math.sqrt(2)
W_SQRT2 = 1 / (1 + SQ2)          # decay at 1000*sqrt(2) m
W_HALF = 0.5                      # decay at 1000 m
W_25 = 1 / 2.5                    # decay at 1500 m


@pytest.fixture(scope="module")
def settlements():
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


def test_border_adjacency_severed_pairwise(settlements, barriers):
    nbrs = apply_barrier(adjacency(settlements, "border"), settlements,
                         barriers, "pair")
    assert nbrs == IDEAL


def test_bbox_adjacency_with_global_barrier(settlements, barriers):
    nbrs = apply_barrier(adjacency(settlements, "bbox"), settlements,
                         barriers, "global")
    assert nbrs == CODE


def test_bbox_equals_border_pre_barrier_for_rectangles(settlements):
    assert adjacency(settlements, "bbox") == adjacency(settlements, "border")


def _city_df(settlements, services, barriers, rule, **overrides):
    kwargs = dict(RULESETS[rule], scenario="baseline", denom="pop")
    kwargs.update(overrides)
    return compute_city(settlements, services, barriers, **kwargs)


def test_clinic_pcen_ideal_baseline_pop(settlements, services, barriers):
    df = _city_df(settlements, services, barriers, "ideal")
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


def test_clinic_pcen_code_rule_differences(settlements, services, barriers):
    df = _city_df(settlements, services, barriers, "code")
    assert df.loc["B", "clinic_pcen"] == pytest.approx(0.0125, abs=1e-12)
    assert df.loc["E", "clinic_pcen"] == pytest.approx(0.005, abs=1e-12)
    assert df.loc["A", "clinic_pcen"] == pytest.approx(
        (2 + W_HALF + W_SQRT2) / 100, abs=1e-12)


def test_school_pcen_ideal_and_unique_anchors(settlements, services, barriers):
    df = _city_df(settlements, services, barriers, "ideal")
    exp = {"A": SQ2 / 100, "B": 0.005, "C": (SQ2 - 1) / 400, "RV": 0.0,
           "D": 0.014, "E": (1 + (SQ2 - 1) + 0.4) / 300, "IND": 0.04}
    for sid, v in exp.items():
        assert df.loc[sid, "school_pcen"] == pytest.approx(v, abs=1e-12), sid
    pcen = df["school_pcen"]
    assert pcen.idxmax() == "IND" and (pcen == pcen.max()).sum() == 1
    assert pcen.idxmin() == "RV" and (pcen == pcen.min()).sum() == 1


def test_popdensity_denominator(settlements, services, barriers):
    df = _city_df(settlements, services, barriers, "ideal", denom="popdensity")
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


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_invariants_guard_csv_wide(city):
    from scripts.check_oraculum_invariants import check
    assert check(city=city) == []


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_expected_values_csv_is_regenerable(city, tmp_path):
    """The committed CSV must be exactly what reference_impl produces.

    Without this, a red build could be 'fixed' by hand-editing the CSV,
    silently turning the oracle into a record of production behavior
    instead of the equations (code review round 1, Critical).
    """
    regen = tmp_path / "regen.csv"
    emit_expected_values(regen, city)
    # bytes, not text: read_text() would normalise nothing here but would
    # hide a line-ending or encoding change in the committed CSV.
    assert regen.read_bytes() == (
        city.fixtures / "expected_values.csv").read_bytes()


# --- 3C: the reference generalisations (spec § 3) ----------------------
def test_compute_city_accepts_an_explicit_scenario_table(settlements, services,
                                                         barriers):
    """The drop mechanics are untouched; only where the table comes from
    moves. A caller-supplied table must NOT leak into the module global —
    scripts/render_oracle_maps.py used to mutate it, which would have
    widened the round-trip-tested fixture CSV."""
    from tests.reference_impl import SCENARIOS

    before = dict(SCENARIOS)
    table = {"nothing_dropped": (frozenset(), False)}
    got = compute_city(settlements, services, barriers,
                       scenario="nothing_dropped", denom="pop",
                       scenarios=table, **RULESETS["ideal"])
    expected = _city_df(settlements, services, barriers, "ideal")
    assert list(got.index) == list(expected.index)
    for sid in expected.index:
        assert got.loc[sid, "clinic_pcen"] == pytest.approx(
            expected.loc[sid, "clinic_pcen"], abs=1e-15), sid
    assert dict(SCENARIOS) == before, "the module table was mutated"


def test_service_amounts_sums_every_road_row(settlements, services):
    """`_service_amounts` used the FIRST road row only. The messy city has
    two, so the sum is load-bearing; pinned here on Oraculum with a second
    row bolted on, so the pin does not depend on the messy fixtures."""
    import geopandas as gpd
    from shapely.geometry import LineString

    from tests.reference_impl import _service_amounts

    base = 1_000_000
    # 500 m of road strictly inside D (x in [-500, 500], y in [0, 1000]),
    # touching no other settlement.
    extra = LineString([(base - 250, base + 500), (base + 250, base + 500)])
    two_rows = gpd.GeoDataFrame(
        {"service": ["road", "road"]},
        geometry=[services["road"].geometry.iloc[0], extra],
        crs=settlements.crs)

    amounts = _service_amounts(
        settlements, {**services, "road": two_rows})["road"]
    assert amounts["A"] == pytest.approx(0.75, abs=1e-12)
    assert amounts["E"] == pytest.approx(0.75, abs=1e-12)
    assert amounts["D"] == pytest.approx(0.5, abs=1e-12), \
        "the second road row was ignored"
    for sid in ("B", "C", "RV", "IND"):
        assert amounts[sid] == 0.0, sid


def test_emit_expected_values_takes_a_city_and_defaults_to_oraculum(tmp_path):
    from tests.cities import ORACULUM

    implicit = tmp_path / "implicit.csv"
    explicit = tmp_path / "explicit.csv"
    emit_expected_values(implicit)
    emit_expected_values(explicit, ORACULUM)
    assert implicit.read_bytes() == explicit.read_bytes() == CSV.read_bytes()


# --- 3D: the variants fixture (spec § 3, § 4.4) ------------------------
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_variants_expected_values_csv_is_regenerable(city, tmp_path):
    """Same contract as expected_values.csv: the committed file must be
    exactly what the reference produces, or a red build could be 'fixed' by
    editing the fixture."""
    regen = tmp_path / "regen.csv"
    emit_variant_expected_values(regen, city)
    assert regen.read_bytes() == (
        city.fixtures / "variants_expected_values.csv").read_bytes()


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_variants_csv_passes_the_csv_wide_invariants_guard(city):
    """`check` groups by (rule, scenario, denom, metric), so it is CSV-shape
    agnostic: the variants file gets the same degenerate-min-max and
    tied-anchor guarantees as expected_values.csv."""
    from scripts.check_oraculum_invariants import check

    frame = pd.read_csv(city.fixtures / "variants_expected_values.csv")
    assert check(frame, city=city) == []


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_variants_csv_has_one_scenario_and_every_variant(city):
    path = city.fixtures / "variants_expected_values.csv"
    frame = pd.read_csv(path)
    assert set(frame["rule"]) == set(VARIANTS)
    assert set(frame["scenario"]) == {city.scenarios[0].name}
    assert set(frame["denom"]) == {"pop", "popdensity"}
    assert list(frame.columns) == ["rule", "scenario", "denom", "settlement",
                                   "metric", "value"]
    assert b"\r" not in path.read_bytes(), "fixtures are LF-only"


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_band_guard_passes_for_both_cities(city):
    from scripts.check_oraculum_invariants import check_bands

    assert check_bands(city) == []


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_band_guard_reports_a_wrong_count(city):
    """The guard must be able to FAIL: move a vertex so a band gains or
    loses a pair and the generator has to refuse to write."""
    from scripts.check_oraculum_invariants import check_bands

    violations = check_bands(city, expected={km: 0 for km in BAND_RADII_KM})
    assert len(violations) == len(BAND_RADII_KM)
    assert all("pair count" in violation for violation in violations)
