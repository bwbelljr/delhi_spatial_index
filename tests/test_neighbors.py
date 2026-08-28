"""delhi_psi.neighbors — adjacency rules, barrier rules, centroid distances.

The `bbox` + `global_asymmetric` combination must reproduce production's
directed lists exactly (the empirical pin from Phase 2); `touch` + `pairwise`
must reproduce the manuscript's symmetric lists from the worksheet.
"""
import pytest

from delhi_psi import geometry, neighbors
from tests.oraculum_fixtures import load_barriers, load_settlements

# docs/oracle/derivation-worksheet.md, "Ideal neighbor lists"
IDEAL_DIRECTED = {"A": {"B", "E"}, "B": {"A", "C", "RV", "E"},
                  "C": {"B", "E", "IND"}, "RV": {"B"}, "D": {"E"},
                  "E": {"A", "B", "C", "D", "IND"}, "IND": {"C", "E"}}
# plan 2026-08-17 "Canonical numbers": flagged {A, D} stripped from every list
CODE_DIRECTED = {"A": {"B", "E"}, "B": {"C", "RV", "E"},
                 "C": {"B", "E", "IND"}, "RV": {"B"}, "D": {"E"},
                 "E": {"B", "C", "IND"}, "IND": {"C", "E"}}


def prepared():
    city = geometry.barrier_flags(load_settlements(), {"canal": load_barriers()})
    city = neighbors.combine_barrier_flags(city, layers=("canal",),
                                           combine="any")
    city["centroid"] = city.centroid
    return city


def lists_of(frame, col="nbrs_bbox"):
    return {row["USO_AREA_U"]: set(row[col]) for _, row in frame.iterrows()}


def test_combine_any_ors_every_layer():
    city = load_settlements().copy()
    city["canal"] = [True, False, False, False, False, False, False]
    city["railway"] = [False, True, False, False, False, False, False]
    out = neighbors.combine_barrier_flags(city, layers=("canal", "railway"),
                                          combine="any")
    assert list(out["barrier"]) == [True, True, False, False, False, False,
                                    False]


def test_combine_selects_named_layers_only():
    city = load_settlements().copy()
    city["canal"] = [True, False, False, False, False, False, False]
    city["railway"] = [False, True, False, False, False, False, False]
    out = neighbors.combine_barrier_flags(city, layers=("canal", "railway"),
                                          combine=("railway",))
    assert list(out["barrier"]) == [False, True, False, False, False, False,
                                    False]


def test_bbox_adjacency_then_global_barrier_matches_production():
    city = prepared()
    nbrs = neighbors.adjacency(city, rule="bbox")
    nbrs = neighbors.apply_barrier(nbrs, list(load_barriers().geometry),
                                   rule="global_asymmetric")
    assert lists_of(nbrs) == CODE_DIRECTED


def test_touch_adjacency_then_pairwise_barrier_matches_the_manuscript():
    city = prepared()
    nbrs = neighbors.adjacency(city, rule="touch")
    nbrs = neighbors.apply_barrier(nbrs, list(load_barriers().geometry),
                                   rule="pairwise")
    assert lists_of(nbrs) == IDEAL_DIRECTED


def test_touch_adjacency_excludes_bbox_only_neighbours():
    """C and A share no boundary, but A's bbox reaches C under `bbox`."""
    city = prepared()
    touch = lists_of(neighbors.adjacency(city, rule="touch"))
    assert "A" not in touch["C"] and "C" not in touch["A"]


def test_unknown_adjacency_rule_raises_value_error():
    with pytest.raises(ValueError, match="diagonal"):
        neighbors.adjacency(prepared(), rule="diagonal")


def test_unknown_barrier_rule_raises_value_error():
    city = neighbors.adjacency(prepared(), rule="bbox")
    with pytest.raises(ValueError, match="sideways"):
        neighbors.apply_barrier(city, list(load_barriers().geometry),
                                rule="sideways")


def test_centroid_distances_are_km_tuples():
    city = prepared()
    nbrs = neighbors.adjacency(city, rule="bbox")
    nbrs = neighbors.apply_barrier(nbrs, list(load_barriers().geometry),
                                   rule="global_asymmetric")
    nbrs = neighbors.centroid_distances(nbrs)
    row = nbrs[nbrs["USO_AREA_U"] == "B"].iloc[0]
    dist = dict(row["nbrs_dist_bbox"])
    assert dist["E"] == pytest.approx(1.0, abs=1e-9)
    assert dist["RV"] == pytest.approx(1.0, abs=1e-9)


# --- 3D: the distance band and boundary distances (spec § 2.1) ---------
# Verified against the fixture geometry. Polygon-to-polygon, so `A` reaches
# `RV` (0.100 km) at 0.25 km, and `B` reaches `D` and `IND` (0.500 km) at
# 0.75 km.
BAND_DIRECTED = {
    0.0: {"A": {"B", "D", "E"}, "B": {"A", "C", "E", "RV"},
          "C": {"B", "E", "IND"}, "RV": {"B"}, "D": {"A", "E"},
          "E": {"A", "B", "C", "D", "IND"}, "IND": {"C", "E"}},
    0.25: {"A": {"B", "D", "E", "RV"}, "B": {"A", "C", "E", "RV"},
           "C": {"B", "E", "IND", "RV"}, "RV": {"A", "B", "C"},
           "D": {"A", "E"}, "E": {"A", "B", "C", "D", "IND"},
           "IND": {"C", "E"}},
    0.75: {"A": {"B", "D", "E", "RV"},
           "B": {"A", "C", "D", "E", "IND", "RV"},
           "C": {"B", "E", "IND", "RV"}, "RV": {"A", "B", "C"},
           "D": {"A", "B", "E"}, "E": {"A", "B", "C", "D", "IND"},
           "IND": {"B", "C", "E"}},
}


@pytest.mark.parametrize("km", [0.0, 0.25, 0.75])
def test_within_distance_lists_match_the_hand_table(km):
    got = lists_of(neighbors.adjacency(prepared(), rule="within_distance",
                                       max_distance_km=km))
    assert got == BAND_DIRECTED[km]


@pytest.mark.parametrize("km", [0.0, 0.25, 0.75, 1.0])
def test_the_dwithin_join_selects_what_brute_force_selects(km):
    """The sjoin is an optimisation, not a definition: it must agree with
    `geom_i.distance(geom_j) <= X` pair for pair (spec § 7)."""
    from tests.reference_impl import adjacency as reference_adjacency

    got = lists_of(neighbors.adjacency(prepared(), rule="within_distance",
                                       max_distance_km=km))
    assert got == reference_adjacency(load_settlements(), "within_distance",
                                      km)


def test_no_neighbour_list_picks_up_a_missing_join_partner():
    """A left join with no match yields NaN. Every polygon is within 0 m of
    itself, so that cannot happen here — pinned, because a NaN id would be
    silently swallowed by pcen's lookup miss instead of failing."""
    frame = neighbors.adjacency(prepared(), rule="within_distance",
                                max_distance_km=0.0)
    for ids in frame["nbrs_bbox"]:
        assert all(isinstance(i, str) for i in ids), ids


def test_within_distance_requires_a_radius():
    with pytest.raises(ValueError, match="max_distance_km"):
        neighbors.adjacency(prepared(), rule="within_distance")


@pytest.mark.parametrize("rule", ["bbox", "touch"])
def test_a_radius_with_another_rule_is_a_value_error(rule):
    """Mirrors the config rule: `build_neighbors` forwards the configured
    value unconditionally, and it is None for every non-band rule."""
    with pytest.raises(ValueError, match="max_distance_km"):
        neighbors.adjacency(prepared(), rule=rule, max_distance_km=1.0)


def test_boundary_distances_have_the_centroid_shape_and_the_gap_values():
    """Same [(id, km), ...] shape as centroid_distances, different numbers:
    A's band-0.25 neighbours B, D and E all touch it (0 km), while RV is
    0.100 km away — where the CENTROID distance is 1.414214 km."""
    frame = neighbors.adjacency(prepared(), rule="within_distance",
                                max_distance_km=0.25)
    boundary = neighbors.boundary_distances(frame)
    centroid = neighbors.centroid_distances(frame)
    row = boundary[boundary["USO_AREA_U"] == "A"].iloc[0]
    assert dict(row["nbrs_dist_boundary"]) == pytest.approx(
        {"B": 0.0, "D": 0.0, "E": 0.0, "RV": 0.1}, abs=1e-12)
    assert [i for i, _ in row["nbrs_dist_boundary"]] == list(row["nbrs_bbox"])
    centroid_row = centroid[centroid["USO_AREA_U"] == "A"].iloc[0]
    assert dict(centroid_row["nbrs_dist_bbox"])["RV"] == pytest.approx(
        1.4142135623730951, abs=1e-12)
