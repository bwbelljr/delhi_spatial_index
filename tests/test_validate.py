"""delhi_psi.validate — every check on synthetic frames, pass and fail.

The notebooks' eyeball checks become assertions that RAISE (DEL-25).
"""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from delhi_psi import validate

BOUNDS = gpd.GeoDataFrame({"name": ["bounds"]},
                          geometry=[box(0, 0, 10_000, 10_000)],
                          crs="EPSG:7760")


def polygons(geoms, crs="EPSG:7760"):
    return gpd.GeoDataFrame({"id": list(range(len(geoms)))},
                            geometry=list(geoms), crs=crs)


def test_has_duplicate_rows_both_ways():
    gdf = polygons([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    assert validate.has_duplicate_rows(gdf) is False
    doubled = pd.concat([gdf, gdf.iloc[[0]]], ignore_index=True)
    doubled["id"] = [0, 1, 0]
    assert validate.has_duplicate_rows(doubled) is True


def test_invalid_geometries_lists_offending_rows():
    bowtie = LineString([(0, 0), (1, 1)]).buffer(0).union(
        box(0, 0, 1, 1))          # valid
    good = polygons([box(0, 0, 1, 1), bowtie])
    assert validate.invalid_geometries(good) == ()
    from shapely.geometry import Polygon
    bad = polygons([Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])])
    assert validate.invalid_geometries(bad) == (0,)


def test_geometries_are_accepts_multipolygon_for_polygon():
    from shapely.geometry import MultiPolygon
    gdf = polygons([box(0, 0, 1, 1),
                    MultiPolygon([box(2, 2, 3, 3), box(4, 4, 5, 5)])])
    assert validate.geometries_are(gdf, "Polygon") is True
    assert validate.geometries_are(gdf, "Point") is False


def test_geometries_are_accepts_linestring_for_line():
    gdf = gpd.GeoDataFrame({"id": [0]},
                           geometry=[LineString([(0, 0), (1, 1)])],
                           crs="EPSG:7760")
    assert validate.geometries_are(gdf, "Line") is True


def test_geometries_are_rejects_a_bad_geom_type_argument():
    with pytest.raises(ValueError, match="Curve"):
        validate.geometries_are(polygons([box(0, 0, 1, 1)]), "Curve")


def test_geometries_are_does_not_mutate_the_frame():
    gdf = polygons([box(0, 0, 1, 1)])
    before = list(gdf.columns)
    validate.geometries_are(gdf, "Polygon")
    assert list(gdf.columns) == before


def test_within_bounds_both_ways():
    inside = polygons([box(10, 10, 20, 20)])
    outside = polygons([box(10, 10, 20_000, 20_000)])
    assert validate.within_bounds(inside, BOUNDS) is True
    assert validate.within_bounds(outside, BOUNDS) is False


def test_check_layer_reports_ok():
    report = validate.check_layer(polygons([box(10, 10, 20, 20)]),
                                  name="settlements", geom_type="Polygon",
                                  bounds_gdf=BOUNDS)
    assert report.ok is True
    assert report.name == "settlements" and report.n_rows == 1


def test_check_layer_reports_not_ok_without_raising():
    report = validate.check_layer(polygons([box(10, 10, 20_000, 20_000)]),
                                  name="settlements", geom_type="Polygon",
                                  bounds_gdf=BOUNDS)
    assert report.ok is False
    assert report.within_bounds is False


def test_require_layer_raises_on_a_bad_layer():
    with pytest.raises(validate.ValidationError) as exc:
        validate.require_layer(polygons([box(10, 10, 20_000, 20_000)]),
                               name="settlements", geom_type="Polygon",
                               bounds_gdf=BOUNDS)
    assert "settlements" in str(exc.value)
    assert "within_bounds" in str(exc.value)


def test_require_layer_returns_the_report_when_ok():
    report = validate.require_layer(polygons([box(10, 10, 20, 20)]),
                                    name="settlements", geom_type="Polygon",
                                    bounds_gdf=BOUNDS)
    assert report.ok is True


def test_check_missing_population_passes_at_the_limit_and_raises_above():
    assert validate.check_missing_population(15, maximum=15) is None
    with pytest.raises(validate.ValidationError) as exc:
        validate.check_missing_population(16, maximum=15)
    assert "16" in str(exc.value) and "15" in str(exc.value)


def test_check_no_negative_passes_and_raises():
    good = pd.DataFrame({"bank_count": [0, 1], "bank_pcen": [0.0, 0.5],
                         "bank_idx": [0.0, 1.0], "ignored": [-9, -9]})
    assert validate.check_no_negative(good) is None
    bad = good.copy()
    bad["bank_pcen"] = [-1.0, 0.5]
    with pytest.raises(validate.ValidationError) as exc:
        validate.check_no_negative(bad)
    assert "bank_pcen" in str(exc.value)


def test_check_crs_match_passes_and_raises():
    a = polygons([box(0, 0, 1, 1)])
    b = polygons([box(2, 2, 3, 3)])
    assert validate.check_crs_match({"a": a, "b": b}) is None
    c = b.to_crs(epsg=4326)
    with pytest.raises(validate.ValidationError) as exc:
        validate.check_crs_match({"a": a, "c": c})
    assert "c" in str(exc.value)


def test_check_crs_defined_raises_when_a_frame_has_no_crs():
    a = polygons([box(0, 0, 1, 1)])
    naked = polygons([box(0, 0, 1, 1)]).set_crs(None, allow_override=True)
    assert validate.check_crs_defined({"a": a}) is None
    with pytest.raises(validate.ValidationError) as exc:
        validate.check_crs_defined({"a": a, "naked": naked})
    assert "naked" in str(exc.value)


def test_read_layer_missing_file_raises_file_not_found(tmp_path):
    # pyogrio raises DataSourceError; io must translate it so the CLI's
    # exit-code mapping (FileNotFoundError/OSError -> 1) holds.
    from delhi_psi import io
    with pytest.raises(FileNotFoundError):
        io.read_layer(tmp_path / "nope" / "missing.shp")
