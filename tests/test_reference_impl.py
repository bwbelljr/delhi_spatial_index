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


def test_reference_minmax_raises_on_a_degenerate_group():
    """The reference is the equations, and Eq. 2 is undefined when hi == lo.
    It used to emit 0.0 — an invention production does not share (DEL-54).
    Unreachable through the committed fixtures: check_oraculum_invariants
    refuses to write a city with a degenerate min-max group."""
    import geopandas as gpd
    from shapely.geometry import box

    # Two settlements, no services at all (not even a road row inside
    # either one) and no barriers: every *_pcen column is 0 for both rows,
    # so the very first min-max (clinic_pcen) is degenerate.
    settlements = gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B"], "population": [100.0, 200.0],
         "area_km2": [1.0, 1.0]},
        geometry=[box(0, 0, 1000, 1000), box(10000, 0, 11000, 1000)],
        crs="EPSG:7760")
    barriers = gpd.GeoDataFrame(geometry=[], crs="EPSG:7760")
    services = {"road": gpd.GeoDataFrame(geometry=[], crs="EPSG:7760")}

    with pytest.raises(ValueError) as excinfo:
        compute_city(settlements, services, barriers,
                     scenario="nothing_dropped", denom="pop",
                     scenarios={"nothing_dropped": (frozenset(), False)},
                     **RULESETS["ideal"])
    assert "clinic_pcen" in str(excinfo.value)


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


# --- DEL-48: partial_weighted, reference side (spec § 2.1, § 6.4) ------
def _pair_city(geom_a, geom_b):
    """Two settlements A and B with the given geometries, no services."""
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B"], "population": [100.0, 200.0],
         "area_km2": [1.0, 1.0]},
        geometry=[geom_a, geom_b], crs="EPSG:7760")


def _barrier_frame(*geoms):
    import geopandas as gpd

    return gpd.GeoDataFrame(geometry=list(geoms), crs="EPSG:7760")


def _w(geom_a, geom_b, *barrier_geoms, buffer_m=5.0):
    """w(A, B) under partial_weighted, asserted symmetric."""
    from tests.reference_impl import partial_weights

    weights = partial_weights({"A": {"B"}, "B": {"A"}},
                              _pair_city(geom_a, geom_b),
                              _barrier_frame(*barrier_geoms), buffer_m)
    assert weights[("A", "B")] == weights[("B", "A")], "w must be symmetric"
    return weights[("A", "B")]


def test_reference_partial_weight_on_a_fully_covered_edge_is_zero():
    """A barrier lying along the whole shared edge blocks all 1000 m, so
    partial_weighted agrees with pairwise: the pair is severed."""
    from shapely.geometry import LineString, box

    assert _w(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
              LineString([(1000, 0), (1000, 1000)])) == 0.0


def test_reference_partial_weight_on_a_half_covered_edge_is_not_one_half():
    """The 5 m round caps extend the blocked span 5 m past each end, so the
    middle 500 m of a 1000 m edge blocks 510 m, not 500. Pinned so nobody
    'fixes' 0.49 into 0.5."""
    from shapely.geometry import LineString, box

    assert _w(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
              LineString([(1000, 250), (1000, 750)])) == pytest.approx(
                  0.49, abs=1e-12)
    # from the corner: one cap falls off the end of the edge, so 505 m
    assert _w(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
              LineString([(1000, 0), (1000, 500)])) == pytest.approx(
                  0.495, abs=1e-12)


def test_reference_a_perpendicular_crossing_blocks_only_the_buffer():
    """The owner's 'a point crossing severs nothing': a barrier crossing the
    shared edge at right angles blocks 2 x 5 m, so w = 0.99 and the link is
    KEPT — where pairwise severs it outright."""
    from shapely.geometry import LineString, box

    assert _w(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
              LineString([(900, 500), (1100, 500)])) == pytest.approx(
                  0.99, abs=1e-12)


def test_reference_a_barrier_just_off_the_edge_still_blocks_it():
    """The buffer's purpose: a canal drawn 4 m off a sliver gap is within
    5 m of every boundary point, so w = 0. A barrier 200 m away is not."""
    from shapely.geometry import LineString, box

    assert _w(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
              LineString([(996, 0), (996, 1000)])) == 0.0
    assert _w(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
              LineString([(1200, 0), (1200, 1000)])) == 1.0


def test_reference_an_overlapping_pair_uses_the_intersection_boundary():
    """The owner's overlap rule made numeric: O1 and O2 overlap in a
    200 x 1000 m strip whose BOUNDARY is its 2400 m perimeter. A barrier
    crossing the strip end to end cuts that perimeter twice (20 m); a barrier
    lying along the strip's own long edge blocks 1000 m + 2 caps."""
    from shapely.geometry import LineString, box

    o1, o2 = box(10000, 0, 11000, 1000), box(10800, 0, 11800, 1000)
    assert _w(o1, o2, LineString([(10900, 0), (10900, 1000)])) == \
        pytest.approx(1 - 20 / 2400, abs=1e-12)
    assert _w(o1, o2, LineString([(11000, 0), (11000, 1000)])) == \
        pytest.approx(1 - 1010 / 2400, abs=1e-12)


def test_reference_a_multipolygon_neighbour_sums_both_shared_edges():
    """A two-part neighbour shares 400 m along each part, so L_shared is
    800 m; a barrier over one part's edge blocks 400 of them."""
    from shapely.geometry import LineString, MultiPolygon, box

    multi = MultiPolygon([box(1000, 0, 2000, 400), box(1000, 600, 2000, 1000)])
    assert _w(box(0, 0, 1000, 1000), multi,
              LineString([(1000, 0), (1000, 400)])) == pytest.approx(
                  0.5, abs=1e-12)


def test_reference_a_mixed_intersection_is_decomposed_part_by_part():
    """The one place a naive `shared.boundary` is WRONG: a neighbour that
    overlaps on one side and shares an edge on another intersects in a
    GeometryCollection, whose `.boundary` is None in shapely 2.1. SB is the
    overlap polygon's 1000 m perimeter plus the 400 m shared line."""
    from shapely.geometry import MultiPolygon, box

    mixed = MultiPolygon([box(900, 0, 1900, 400), box(1000, 600, 1900, 1000)])
    square = box(0, 0, 1000, 1000)
    assert square.intersection(mixed).geom_type == "GeometryCollection"
    assert square.intersection(mixed).boundary is None
    assert _w(square, mixed) == 1.0          # no barrier: SB length 1400, w 1


def test_reference_a_corner_only_contact_is_never_severed():
    """L_shared == 0, so there is no boundary to block and w = 1 even with a
    barrier straight through the corner (spec § 2.1 step 3). pairwise severs
    this pair; this is the documented difference between the two rules."""
    from shapely.geometry import LineString, box

    assert _w(box(0, 0, 1000, 1000), box(1000, 1000, 2000, 2000),
              LineString([(900, 1100), (1100, 900)])) == 1.0


def test_reference_partial_weighted_requires_a_positive_buffer():
    from shapely.geometry import box

    from tests.reference_impl import apply_barrier

    city = _pair_city(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000))
    nbrs = {"A": {"B"}, "B": {"A"}}
    with pytest.raises(ValueError, match="barrier_buffer_m"):
        apply_barrier(nbrs, city, _barrier_frame(), "partial_weighted")
    with pytest.raises(ValueError, match="barrier_buffer_m"):
        apply_barrier(nbrs, city, _barrier_frame(), "partial_weighted",
                      buffer_m=0)


@pytest.mark.parametrize("rule", ["global", "pair"])
def test_reference_the_other_rules_reject_a_buffer(rule):
    """An unimplemented combination must RAISE — the mapped-knob test relies
    on it."""
    from shapely.geometry import box

    from tests.reference_impl import apply_barrier

    city = _pair_city(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000))
    with pytest.raises(ValueError, match="barrier_buffer_m"):
        apply_barrier({"A": {"B"}, "B": {"A"}}, city, _barrier_frame(), rule,
                      buffer_m=5.0)


# --- 3E: production == reference on synthetic geometry (spec § 6.4) ----
def synthetic_partial_city():
    """Three settlements built for the case no fixture city can carry.

    P and Q OVERLAP (so their shared boundary is the intersection polygon's
    perimeter); R is a two-part MultiPolygon TOUCHING P along both parts
    (a MultiLineString shared boundary); a canal partially covers the P-R
    boundary, so at least one weight is strictly between 0 and 1.

    EVERY point service the reference scores gets a layer, because
    `compute_city` min-maxes all six of `POINT_SERVICES` plus road and
    DEL-54's guard raises on a constant column — a city with only a clinic
    layer would make school/bank/police/ration/transport all-zero and stop
    the comparison before it started. One point per settlement, reused for
    every service: the three denominators (100, 200, 400) are distinct, so
    no PCEN column can be constant. Each point is strictly interior to
    exactly one settlement and none lies in the P-Q overlap, so
    production's boundary-inclusive `intersects` and the reference's strict
    `within` agree on every one (rule-set gap #6 is not in scope here).
    """
    import geopandas as gpd
    from shapely.geometry import LineString, MultiPolygon, Point, box

    settlements = gpd.GeoDataFrame(
        {"USO_AREA_U": ["P", "Q", "R"], "USO_FINAL": ["Planned"] * 3,
         "population": [100.0, 200.0, 400.0],
         "area_km2": [1.2, 1.0, 0.8]},
        geometry=[box(0, 0, 1200, 1000),
                  box(1000, 0, 2000, 1000),
                  MultiPolygon([box(-400, 0, 0, 400),
                                box(-400, 600, 0, 1000)])],
        crs="EPSG:7760")
    # The canal covers y in [0, 400] of the x = 0 boundary P shares with R's
    # lower part: 400 of the 800 m shared boundary, so w_PR is strictly
    # fractional (0.5). The round cap adds nothing — the canal's endpoints
    # coincide with that segment's own endpoints and the shared boundary
    # does not continue past them; the test asserts 0 < w_PR < 1, so this
    # comment is description, not a pin.
    barriers = gpd.GeoDataFrame(
        {"name": ["canal"]},
        geometry=[LineString([(0, 0), (0, 400)])], crs="EPSG:7760")
    # P only, Q only, R's upper part only — none in the P-Q overlap.
    hosts = [Point(600, 500), Point(1600, 500), Point(-200, 800)]
    services = {
        name: gpd.GeoDataFrame({"service": [name] * 3},
                               geometry=list(hosts), crs="EPSG:7760")
        for name in ("clinic", "school", "bank", "police", "ration",
                     "transport")
    }
    # 300 m inside R's lower part, 1100 m inside P, 100 m inside Q (the
    # overlap stretch counts for both owners, on both sides) — three
    # distinct lengths, so road_pcen is not constant either.
    services["road"] = gpd.GeoDataFrame(
        {"service": ["road"]},
        geometry=[LineString([(-300, 200), (1100, 200)])], crs="EPSG:7760")
    return settlements, barriers, services


def test_production_matches_the_reference_on_synthetic_partial_geometry():
    """The fractional-weight x overlap x MultiPolygon case, scored by BOTH
    implementations at 1e-12. It cannot live in a fixture city without
    moving an existing expected value (spec § 12 item 3), so it lives here
    and costs no fixture file.
    """
    from delhi_psi.config import (
        AbsentNeighbor, AdjacencyConfig, AdjacencyRule, BarrierConfig,
        BarrierRule, DecayConfig, DecayDistance, DecayForm, ExclusionConfig,
        ExclusionStage, MethodologyConfig, RoadsFormula,
    )
    from delhi_psi.pipeline import compute_frames
    from tests.test_profiles_match_reference import METRIC_MAP

    settlements, barriers, services = synthetic_partial_city()
    methodology = MethodologyConfig(
        adjacency=AdjacencyConfig(rule=AdjacencyRule.BBOX),
        barrier=BarrierConfig(rule=BarrierRule.PARTIAL_WEIGHTED,
                              combine="any", buffer_m=5.0),
        decay=DecayConfig(form=DecayForm.INVERSE_LINEAR, distance_unit="km",
                          distance=DecayDistance.CENTROID),
        roads=RoadsFormula.DECAYED,
        second_normalization=True,
        exclusion=ExclusionConfig(types=(), stage=ExclusionStage.POST_NEIGHBORS,
                                  absent_neighbor=AbsentNeighbor.SWALLOWED))

    for denom in ("pop", "popdensity"):
        got = compute_frames(settlements, {"canal": barriers}, services, None,
                             methodology, denom,
                             mapping={"Planned": "Planned"},
                             scheme="synthetic").set_index("USO_AREA_U")
        exp = compute_city(
            settlements, services, barriers, adjacency_rule="bbox",
            barrier_rule="partial_weighted", barrier_buffer_m=5.0,
            roads_formula="decayed", scenario="none", denom=denom,
            second_norm=True, absent_neighbor_contribution="swallowed",
            scenarios={"none": (frozenset(), False)})
        assert set(got.index) == set(exp.index)
        # Every METRIC_MAP column exists on both sides: the city carries all
        # six point services plus road, and second_norm is on, so nothing is
        # skipped and the comparison cannot pass by omission.
        for prod_col, metric in METRIC_MAP.items():
            for sid in exp.index:
                assert got.loc[sid, prod_col] == pytest.approx(
                    exp.loc[sid, metric], abs=1e-12), (denom, sid, prod_col)


def test_the_synthetic_city_really_carries_a_fractional_weight():
    """The test above would still pass if every weight were 1 or 0 — that is
    exactly the failure mode it exists to rule out."""
    from tests.reference_impl import adjacency, partial_weights

    settlements, barriers, _ = synthetic_partial_city()
    weights = partial_weights(adjacency(settlements, "bbox"), settlements,
                              barriers, 5.0)
    fractional = [w for w in weights.values() if 0.0 < w < 1.0]
    assert fractional, weights
    assert weights[("P", "R")] == weights[("R", "P")]
