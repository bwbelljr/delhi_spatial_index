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


def test_shim_still_matches_the_new_path():
    import spatial_index_utils

    city = prepared()
    old = spatial_index_utils.add_polygon_neighbors_column_fast(
        polygon_gdf=city.copy(),
        right_gdf=geometry.bbox_frame(city.copy()),
        id_colname="USO_AREA_U", neighbor_colname="nbrs_bbox",
        barrier_colname="barrier")
    new = neighbors.apply_barrier(
        neighbors.adjacency(city, rule="bbox"),
        list(load_barriers().geometry), rule="global_asymmetric")
    assert lists_of(old) == lists_of(new)
