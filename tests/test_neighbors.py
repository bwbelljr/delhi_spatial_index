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


# --- 3E: partial_weighted (spec § 2.1, § 2.6, § 6.4) -------------------
def two_squares(geom_a=None, geom_b=None):
    """A and B, 1 km squares sharing the 1000 m edge at x = 1000 unless the
    caller supplies its own geometries. The `test_index.city_with_neighbours`
    shape: hand-built, EPSG:7760, metre coordinates."""
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B"], "nbrs_bbox": [["B"], ["A"]],
         "barrier": [False, False]},
        geometry=[geom_a if geom_a is not None else box(0, 0, 1000, 1000),
                  geom_b if geom_b is not None else box(1000, 0, 2000, 1000)],
        crs="EPSG:7760")


def weights_of(frame, col="nbrs_barrier_weight"):
    return {row["USO_AREA_U"]: dict(row[col]) for _, row in frame.iterrows()}


def partial(frame, *barrier_geoms, buffer_m=5.0):
    return neighbors.apply_barrier(frame, list(barrier_geoms),
                                   rule="partial_weighted",
                                   buffer_m=buffer_m)


def test_partial_weighted_on_a_fully_covered_edge_prunes_like_pairwise():
    """w = 0 exactly, so the pair leaves the list and the frame equals what
    `pairwise` produces. The 'full-coverage variant' the memo imagined is
    this unit test, not a fixture."""
    from shapely.geometry import LineString

    canal = LineString([(1000, 0), (1000, 1000)])
    got = partial(two_squares(), canal)
    assert lists_of(got) == {"A": set(), "B": set()}
    assert weights_of(got) == {"A": {}, "B": {}}
    severed = neighbors.apply_barrier(two_squares(), [canal], rule="pairwise")
    assert lists_of(got) == lists_of(severed)


def test_partial_weighted_on_a_half_covered_edge_is_not_one_half():
    """5 m round caps extend the blocked span past each end: the middle
    500 m blocks 510, and 500 m from the corner blocks 505. Pinned so nobody
    'fixes' 0.49 into 0.5."""
    from shapely.geometry import LineString

    middle = partial(two_squares(), LineString([(1000, 250), (1000, 750)]))
    assert weights_of(middle)["A"]["B"] == pytest.approx(0.49, abs=1e-12)
    corner = partial(two_squares(), LineString([(1000, 0), (1000, 500)]))
    assert weights_of(corner)["A"]["B"] == pytest.approx(0.495, abs=1e-12)


def test_a_perpendicular_crossing_is_kept_where_pairwise_severs():
    """The owner's 'a point crossing severs nothing'. This is the one
    documented case where partial_weighted and pairwise disagree."""
    from shapely.geometry import LineString

    crossing = LineString([(900, 500), (1100, 500)])
    got = partial(two_squares(), crossing)
    assert weights_of(got)["A"]["B"] == pytest.approx(0.99, abs=1e-12)
    assert lists_of(got) == {"A": {"B"}, "B": {"A"}}
    severed = neighbors.apply_barrier(two_squares(), [crossing],
                                      rule="pairwise")
    assert lists_of(severed) == {"A": set(), "B": set()}


def test_a_barrier_just_off_the_edge_still_blocks_it():
    """The buffer's purpose: a barrier drawn 4 m off a sliver gap is within
    5 m of every boundary point, so w = 0. One 200 m away is not."""
    from shapely.geometry import LineString

    close = partial(two_squares(), LineString([(996, 0), (996, 1000)]))
    assert lists_of(close) == {"A": set(), "B": set()}
    far = partial(two_squares(), LineString([(1200, 0), (1200, 1000)]))
    assert weights_of(far)["A"]["B"] == 1.0


def test_an_overlapping_pair_weighs_the_intersection_boundary():
    """The owner's overlap rule, made numeric on the messy O1/O2 shape: the
    shared boundary is the 200 x 1000 m strip's 2400 m PERIMETER."""
    from shapely.geometry import LineString, box

    frame = two_squares(box(10000, 0, 11000, 1000),
                        box(10800, 0, 11800, 1000))
    crossing = partial(frame, LineString([(10900, 0), (10900, 1000)]))
    assert weights_of(crossing)["A"]["B"] == pytest.approx(
        1 - 20 / 2400, abs=1e-12)
    along = partial(frame, LineString([(11000, 0), (11000, 1000)]))
    assert weights_of(along)["A"]["B"] == pytest.approx(
        1 - 1010 / 2400, abs=1e-12)


def test_a_multipolygon_neighbour_sums_both_shared_edges():
    from shapely.geometry import LineString, MultiPolygon, box

    multi = MultiPolygon([box(1000, 0, 2000, 400), box(1000, 600, 2000, 1000)])
    got = partial(two_squares(geom_b=multi),
                  LineString([(1000, 0), (1000, 400)]))
    assert weights_of(got)["A"]["B"] == pytest.approx(0.5, abs=1e-12)


def test_a_mixed_intersection_is_decomposed_part_by_part():
    """Overlap on one side, shared edge on another: shapely returns a
    GeometryCollection, whose `.boundary` is None. SB is the overlap
    polygon's 1000 m perimeter plus the 400 m line, so a barrier over the
    line alone blocks exactly 400 of 1400.

    The round cap adds NOTHING here, unlike the Oraculum canal: this
    barrier's endpoints coincide with the line piece's own endpoints, and
    the shared boundary does not continue past them — the polygon piece is
    200 m away, far beyond the 5 m buffer. Verified in shapely 2.1.2: the
    blocked length is 400.0 at buffer 5 AND at buffer 1. A cap only wins
    extra length where the boundary continues past the barrier's end, which
    is the Oraculum case (canal [25, 475] strictly inside a 500 m edge) and
    the closed-perimeter overlap case above."""
    from shapely.geometry import LineString, MultiPolygon, box

    mixed = MultiPolygon([box(900, 0, 1900, 400), box(1000, 600, 1900, 1000)])
    square = box(0, 0, 1000, 1000)
    assert square.intersection(mixed).boundary is None
    shared = neighbors.shared_boundary(square, mixed)
    assert shared.length == pytest.approx(1400.0, abs=1e-9)
    got = partial(two_squares(square, mixed),
                  LineString([(1000, 600), (1000, 1000)]))
    assert weights_of(got)["A"]["B"] == pytest.approx(
        1 - 400 / 1400, abs=1e-12)


def test_a_corner_only_contact_is_never_severed():
    """L_shared == 0: there is no boundary to block, so w = 1 even with a
    barrier through the corner (spec § 2.1 step 3)."""
    from shapely.geometry import LineString, box

    frame = two_squares(geom_b=box(1000, 1000, 2000, 2000))
    got = partial(frame, LineString([(900, 1100), (1100, 900)]))
    assert weights_of(got)["A"]["B"] == 1.0
    assert lists_of(got) == {"A": {"B"}, "B": {"A"}}


@pytest.mark.parametrize("barrier_geom", [
    "collinear", "middle", "perpendicular", "overlap"])
def test_the_weight_is_symmetric(barrier_geom):
    """Same GEOS calls on the same operands from either side, so w(i, j) and
    w(j, i) must agree BIT for bit, not to a tolerance."""
    from shapely.geometry import LineString, box

    cases = {
        "collinear": (two_squares(), LineString([(1000, 0), (1000, 1000)])),
        "middle": (two_squares(), LineString([(1000, 250), (1000, 750)])),
        "perpendicular": (two_squares(), LineString([(900, 500), (1100, 500)])),
        "overlap": (two_squares(box(10000, 0, 11000, 1000),
                                box(10800, 0, 11800, 1000)),
                    LineString([(10900, 0), (10900, 1000)])),
    }
    frame, barrier = cases[barrier_geom]
    got = neighbors.apply_barrier(frame, [barrier], rule="partial_weighted",
                                  buffer_m=5.0)
    forward = dict(got.iloc[0]["nbrs_barrier_weight"])
    backward = dict(got.iloc[1]["nbrs_barrier_weight"])
    assert forward.get("B") == backward.get("A")


def test_partial_weighted_with_no_barriers_writes_weights_of_one():
    """The messy city has an EMPTY barrier layer and is scored under this
    rule, so the column must still exist and hold 1.0 — otherwise `pcen`
    raises KeyError on a city with nothing to block."""
    got = partial(two_squares())
    assert weights_of(got) == {"A": {"B": 1.0}, "B": {"A": 1.0}}


def test_the_weight_column_exists_only_under_partial_weighted():
    """code-2025's artifact must be byte-identical, and an artifact built
    before 3E must still load."""
    for rule in ("global_asymmetric", "pairwise"):
        got = neighbors.apply_barrier(two_squares(), [], rule=rule)
        assert "nbrs_barrier_weight" not in got.columns


def test_buffer_m_is_required_by_partial_weighted_and_rejected_otherwise():
    with pytest.raises(ValueError, match="buffer_m"):
        neighbors.apply_barrier(two_squares(), [], rule="partial_weighted")
    with pytest.raises(ValueError, match="buffer_m"):
        neighbors.apply_barrier(two_squares(), [], rule="partial_weighted",
                                buffer_m=0)
    for rule in ("global_asymmetric", "pairwise"):
        with pytest.raises(ValueError, match="buffer_m"):
            neighbors.apply_barrier(two_squares(), [], rule=rule,
                                    buffer_m=5.0)


def test_combine_selects_the_layers_the_geometry_rules_see():
    """spec § 2.6: `combine` chooses the LAYERS whose geometries pairwise and
    partial_weighted read — which is what the stamp's '`combine` decides who
    is severed' already claims. With combine=('railway',) a canal over the
    shared edge severs nothing; with 'any' it severs."""
    import geopandas as gpd
    from shapely.geometry import LineString

    canal = LineString([(1000, 0), (1000, 1000)])
    barriers = {
        "canal": gpd.GeoDataFrame(geometry=[canal], crs="EPSG:7760"),
        "railway": gpd.GeoDataFrame(
            geometry=[LineString([(5000, 0), (5000, 1000)])],
            crs="EPSG:7760"),
    }
    railway_only = neighbors.selected_barrier_geoms(barriers,
                                                    combine=("railway",))
    every = neighbors.selected_barrier_geoms(barriers, combine="any")
    assert len(railway_only) == 1 and len(every) == 2

    kept = neighbors.apply_barrier(two_squares(), railway_only,
                                   rule="pairwise")
    assert lists_of(kept) == {"A": {"B"}, "B": {"A"}}
    cut = neighbors.apply_barrier(two_squares(), every, rule="pairwise")
    assert lists_of(cut) == {"A": set(), "B": set()}
    weighted = neighbors.apply_barrier(two_squares(), railway_only,
                                       rule="partial_weighted", buffer_m=5.0)
    assert weights_of(weighted)["A"]["B"] == 1.0


def test_selected_barrier_geoms_rejects_an_unconfigured_layer():
    """The same message `combine_barrier_flags` gives, from one helper."""
    import geopandas as gpd

    barriers = {"canal": gpd.GeoDataFrame(geometry=[], crs="EPSG:7760")}
    with pytest.raises(ValueError, match="drain"):
        neighbors.selected_barrier_geoms(barriers, combine=("drain",))
