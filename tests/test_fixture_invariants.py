"""Geometry invariants + the empirical pin (spec build-order step 1).

If test_empirical_pin_* fails: STOP — the spec's directed neighbor table is
wrong and must be corrected by the owner before anything downstream is
built (spec 'Risks': hard red line).
"""

import itertools
import math

import pytest
from shapely.geometry import box

from tests.oraculum_fixtures import (
    load_settlements, load_barriers, load_services, run_production_chain,
)

BASE = 1_000_000

GEOMETRIC_PAIRS_KM = {
    frozenset(p): d for p, d in {
        ("A", "B"): 1.0, ("A", "D"): math.sqrt(5) / 2, ("A", "E"): math.sqrt(2),
        ("B", "C"): 1.0, ("B", "RV"): 1.0, ("B", "E"): 1.0,
        ("C", "E"): math.sqrt(2), ("C", "IND"): math.sqrt(5) / 2,
        ("D", "E"): 1.5, ("E", "IND"): 1.5,
    }.items()
}

IDEAL_DIRECTED = {"A": {"B", "E"}, "B": {"A", "C", "RV", "E"},
                  "C": {"B", "E", "IND"}, "RV": {"B"}, "D": {"E"},
                  "E": {"A", "B", "C", "D", "IND"}, "IND": {"C", "E"}}
CODE_DIRECTED = {"A": {"B", "E"}, "B": {"C", "RV", "E"},
                 "C": {"B", "E", "IND"}, "RV": {"B"}, "D": {"E"},
                 "E": {"B", "C", "IND"}, "IND": {"C", "E"}}


@pytest.fixture(scope="module")
def city():
    return load_settlements().set_index("USO_AREA_U")


def test_all_rectangles_bbox_equals_geometry(city):
    for sid, row in city.iterrows():
        assert row.geometry.equals(box(*row.geometry.bounds)), sid


def test_touching_pairs_share_edges_never_points(city):
    seen = set()
    for i, j in itertools.combinations(city.index, 2):
        gi, gj = city.loc[i].geometry, city.loc[j].geometry
        inter = gi.intersection(gj)
        if not inter.is_empty:
            assert inter.length > 0, f"{i}-{j} touch only at a point"
            seen.add(frozenset((i, j)))
    assert seen == set(GEOMETRIC_PAIRS_KM), "pair set differs from spec table"


def test_pair_distances_match_spec(city):
    for pair, d_km in GEOMETRIC_PAIRS_KM.items():
        i, j = tuple(pair)
        d = city.loc[i].geometry.centroid.distance(city.loc[j].geometry.centroid)
        assert d / 1000 == pytest.approx(d_km, abs=1e-9), pair


def test_canal_inside_ad_edge_touches_exactly_a_and_d(city):
    canal = load_barriers().geometry.iloc[0]
    shared = city.loc["A"].geometry.intersection(city.loc["D"].geometry)
    assert canal.within(shared.buffer(1e-9))
    touching = {sid for sid, row in city.iterrows()
                if row.geometry.intersects(canal)}
    assert touching == {"A", "D"}
    for sid in touching:
        assert city.loc[sid].geometry.intersection(canal).length > 0, \
            f"canal touches {sid} only at a point"


def test_road_lengths_and_canal_clearance(city):
    road = load_services()["road"].geometry.iloc[0]
    canal = load_barriers().geometry.iloc[0]
    assert not road.intersects(canal)
    for sid, expected_km in [("A", 0.75), ("E", 0.75)]:
        got = road.intersection(city.loc[sid].geometry).length / 1000
        assert got == pytest.approx(expected_km, abs=1e-12), sid
    for sid in ("B", "C", "RV", "D", "IND"):
        assert road.intersection(city.loc[sid].geometry).length == 0, sid


def test_service_points_inside_their_hosts(city):
    services = load_services()
    for name, gdf in services.items():
        if name == "road":
            continue
        for _, row in gdf.iterrows():
            assert row.geometry.within(city.loc[row["host"]].geometry), \
                f"{name} point not inside {row['host']}"


def test_empirical_pin_code_rule_neighbors():
    """THE GATE: production code must reproduce the spec's directed table."""
    result = run_production_chain(
        load_settlements(), load_barriers(), load_services(), "pop")
    got = {row["USO_AREA_U"]: set(row["nbrs_bbox"])
           for _, row in result.iterrows()}
    assert got == CODE_DIRECTED


def test_empirical_pin_distances_are_km_tuples():
    result = run_production_chain(
        load_settlements(), load_barriers(), load_services(), "pop")
    row = result[result["USO_AREA_U"] == "B"].iloc[0]
    dist = dict(row["nbrs_dist_bbox"])
    assert dist["E"] == pytest.approx(1.0, abs=1e-9)
    assert dist["RV"] == pytest.approx(1.0, abs=1e-9)
