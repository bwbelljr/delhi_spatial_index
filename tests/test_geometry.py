"""delhi_psi.geometry — the moved geometry primitives, pinned on fixtures."""
import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, box

from delhi_psi import geometry
from tests.oraculum_fixtures import load_barriers, load_settlements


def test_row_index_finds_the_row():
    city = load_settlements()
    idx = geometry.row_index(city, "USO_AREA_U", "C")
    assert city.loc[idx, "USO_AREA_U"] == "C"


def test_reproject_changes_crs_and_moves_geometry():
    city = load_settlements()
    out = geometry.reproject(city, 4326)
    assert out.crs.to_epsg() == 4326
    assert out.geometry.iloc[0].bounds != city.geometry.iloc[0].bounds


def test_remove_duplicate_geom_keeps_first_occurrence():
    geom = box(0, 0, 1, 1)
    gdf = gpd.GeoDataFrame({"name": ["a", "b", "c"]},
                           geometry=[geom, box(2, 2, 3, 3), box(0, 0, 1, 1)],
                           crs="EPSG:7760")
    out = geometry.remove_duplicate_geom(gdf)
    assert list(out["name"]) == ["a", "b"]


def test_bbox_frame_is_the_exact_envelope():
    city = load_settlements()
    boxes = geometry.bbox_frame(city)
    assert list(boxes["USO_AREA_U"]) == list(city["USO_AREA_U"])
    for original, produced in zip(city.geometry, boxes.geometry):
        assert produced.equals(original.envelope)
    assert boxes.crs == city.crs


def test_barrier_flags_one_column_per_layer():
    city = load_settlements()
    out = geometry.barrier_flags(city, {"canal": load_barriers()})
    flagged = set(out.loc[out["canal"], "USO_AREA_U"])
    # the fixture canal is a strict interior sub-segment of the A|D edge
    assert flagged == {"A", "D"}


def test_barrier_flags_missing_layer_is_all_false():
    city = load_settlements()
    empty = gpd.GeoDataFrame(
        {"name": ["far"]},
        geometry=[LineString([(9_000_000, 9_000_000), (9_000_001, 9_000_001)])],
        crs=city.crs)
    out = geometry.barrier_flags(city, {"railway": empty})
    assert not out["railway"].any()


def test_distance_to_point_km_is_metres_over_1000():
    city = load_settlements()
    city = city.copy()
    city["centroid"] = city.centroid
    centre = Point(1_001_500, 1_001_500)
    out = geometry.distance_to_point_km(city, centre)
    row = out[out["USO_AREA_U"] == "A"].iloc[0]
    assert out["ndmc_dist_km"].dtype.kind == "f"
    assert row["ndmc_dist_km"] == pytest.approx(
        centre.distance(row["centroid"]) / 1000, abs=1e-12)
