"""The divergence exhibit: where bbox/intersects adjacency and the
manuscript's border-sharing rule disagree, with pinned PCEN deltas.

These tests PASS by asserting the documented divergence itself; a failure
means adjacency behavior changed without updating the record (spec §
Divergence exhibit)."""

import math

import pytest

from tests.oraculum_fixtures import load_exhibit
from tests.reference_impl import adjacency


@pytest.fixture(scope="module")
def exhibit():
    gdf = load_exhibit().rename(columns={"id": "USO_AREA_U"})
    return gdf


def _clinic_pcen(gdf, nbrs):
    cent = gdf.set_index("USO_AREA_U").geometry.centroid
    counts = gdf.set_index("USO_AREA_U")["clinics"]
    pops = gdf.set_index("USO_AREA_U")["population"]
    out = {}
    for i in counts.index:
        total = float(counts[i])
        for j in nbrs[i]:
            d_km = cent[i].distance(cent[j]) / 1000
            total += counts[j] / (1 + d_km)
        out[i] = total / pops[i]
    return out


def test_border_rule_no_neighbors(exhibit):
    nbrs = adjacency(exhibit, "border")
    assert nbrs == {"P": set(), "Q": set(), "R": set(), "S": set()}


def test_bbox_rule_invents_both_divergence_flavors(exhibit):
    """bbox catches the containment phantom (Q->P, directed) AND the corner
    touch (R<->S, since rectangles' bboxes equal their geometry) —
    production-verified in plan review round 1."""
    nbrs = adjacency(exhibit, "bbox")
    assert nbrs["Q"] == {"P"}          # Q's geometry lies inside P's bbox
    assert nbrs["P"] == set()          # P's geometry misses Q's bbox
    assert nbrs["R"] == {"S"} and nbrs["S"] == {"R"}


def test_intersects_rule_only_corner_touch(exhibit):
    nbrs = adjacency(exhibit, "intersects")
    assert nbrs["R"] == {"S"} and nbrs["S"] == {"R"}
    assert nbrs["P"] == set() and nbrs["Q"] == set()


def test_pinned_pcen_deltas(exhibit):
    border = _clinic_pcen(exhibit, adjacency(exhibit, "border"))
    bbox = _clinic_pcen(exhibit, adjacency(exhibit, "bbox"))
    inter = _clinic_pcen(exhibit, adjacency(exhibit, "intersects"))

    assert border["Q"] == pytest.approx(0.01, abs=1e-12)
    assert bbox["Q"] - border["Q"] == pytest.approx(0.005147186, abs=1e-9)
    assert bbox["P"] - border["P"] == pytest.approx(0.0, abs=1e-15)
    assert bbox["S"] - border["S"] == pytest.approx(0.016568542494923802,
                                                   abs=1e-12)
    assert bbox["R"] - border["R"] == pytest.approx(0.0, abs=1e-15)

    assert inter["S"] - border["S"] == pytest.approx(0.016568542, abs=1e-9)
    assert inter["R"] - border["R"] == pytest.approx(0.0, abs=1e-15)
    assert inter["Q"] - border["Q"] == pytest.approx(0.0, abs=1e-15)

    # spot-check the geometry behind Q's delta: P centroid at (833.33, 833.33)
    d_km = math.hypot(1500 - 2500 / 3, 1500 - 2500 / 3) / 1000
    assert 1 / (1 + d_km) / 100 == pytest.approx(bbox["Q"] - border["Q"],
                                                 abs=1e-9)


def test_production_bbox_adjacency_on_exhibit(exhibit):
    """Pin rule-set gap #1 against PRODUCTION, not just the reference impl.

    The main city is all-rectangles (bbox == geometry), so it cannot tell
    bbox adjacency apart from polygon-intersects; the exhibit can. Code
    review round 1 proved a bbox->geometry mutation in create_bbox_gdf
    survived the entire suite without this test.
    """
    import geopandas as gpd

    import spatial_index_utils

    gdf = exhibit.copy()
    gdf["barrier"] = False
    bbox_gdf = spatial_index_utils.create_bbox_gdf(gdf)
    bbox_gdf = gpd.GeoDataFrame(bbox_gdf, crs=gdf.crs)
    result = spatial_index_utils.add_polygon_neighbors_column_fast(
        polygon_gdf=gdf, right_gdf=bbox_gdf, id_colname="USO_AREA_U",
        neighbor_colname="nbrs_bbox", barrier_colname="barrier")
    got = {row["USO_AREA_U"]: set(row["nbrs_bbox"])
           for _, row in result.iterrows()}
    assert got == {"P": set(), "Q": {"P"}, "R": {"S"}, "S": {"R"}}


def test_production_bbox_geometry_is_exact_envelope(exhibit):
    """Pin the bbox GEOMETRY, not just its adjacency consequences.

    Code review round 2: bboxes could be dilated by ~99 m with a green
    suite, because both fixtures sit on a coarse grid. On real data,
    polygon pairs sit metres apart, so an accidental buffer would rewrite
    neighbor lists city-wide while the oracle stayed green.
    """
    import spatial_index_utils

    gdf = exhibit.copy()
    bbox_gdf = spatial_index_utils.create_bbox_gdf(gdf)
    for original, produced in zip(gdf.geometry, bbox_gdf["geometry"]):
        assert produced.equals(original.envelope)
