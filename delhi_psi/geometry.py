"""Geometry primitives: reprojection, deduplication, bounding boxes,
barrier flags, point distances.

Pure functions with explicit keyword arguments — never imports
delhi_psi.config. The math is copied verbatim from spatial_index_utils.py;
the O(n^2) remove_duplicate_geom algorithm is deliberately unchanged
(spec § 6).
"""

import logging
from itertools import islice

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import box
from tqdm import tqdm

log = logging.getLogger(__name__)


def row_index(gdf, id_col, id_num):
    """Row index of a GeoDataFrame given a unique id."""
    return gdf[gdf[id_col] == id_num].index.values[0]


def reproject(gdf, epsg_code):
    """Reproject to the CRS with this EPSG code (WKT, as production does)."""
    target_projection = CRS.from_epsg(epsg_code).to_wkt()
    reprojected_gdf = gdf.to_crs(target_projection)
    log.debug("GeoDataFrame now has CRS %s", reprojected_gdf.crs)
    return reprojected_gdf


def remove_duplicate_geom(gdf, geom_col="geometry"):
    """Remove rows with duplicate geometries (Shapely `equals`), O(n^2).

    Keeps the first occurrence. Returns a frame with a NEW index.
    """
    old_size = len(gdf)
    gdf["not_duplicate"] = True

    for idx, row in tqdm(gdf.iterrows()):
        row_geom = row[geom_col]
        for idx2, row2 in islice(gdf.iterrows(), idx + 1, None):
            other_geom = row2[geom_col]
            if row_geom.equals(other_geom):
                gdf.loc[idx2, "not_duplicate"] = False

    gdf = gdf[gdf["not_duplicate"]]
    gdf = gdf.drop(columns=["not_duplicate"])
    gdf = gdf.reset_index()
    log.info("deduplicated %d rows to %d", old_size, len(gdf))
    return gdf


def bbox_frame(polygon_gdf):
    """GeoDataFrame whose geometry is each row's bounding box."""
    gdf_bbox = gpd.GeoDataFrame(
        pd.concat([polygon_gdf, polygon_gdf.bounds], axis=1))
    gdf_bbox["bbox"] = None
    for idx, row in gdf_bbox.iterrows():
        row_bbox = box(row["minx"], row["miny"], row["maxx"], row["maxy"])
        gdf_bbox.loc[idx, "bbox"] = row_bbox
    gdf_bbox = gdf_bbox.drop(
        columns=["geometry", "minx", "miny", "maxx", "maxy"])
    gdf_bbox = gdf_bbox.rename(columns={"bbox": "geometry"})
    gdf_bbox = gdf_bbox.set_geometry("geometry")
    gdf_bbox.crs = polygon_gdf.crs
    return gdf_bbox


def _flag_one(polygon_gdf, barrier_gdf, flag_col, id_col):
    polygon_gdf[flag_col] = False
    joined = gpd.sjoin(polygon_gdf, barrier_gdf, how="inner")
    ids_with_intersection = list(joined[id_col].unique())
    for polygon_id in ids_with_intersection:
        idx = polygon_gdf[polygon_gdf[id_col] == polygon_id].index.values[0]
        polygon_gdf.loc[idx, flag_col] = True
    return polygon_gdf


def barrier_flags(polygon_gdf, barriers, *, id_col="USO_AREA_U"):
    """One boolean column per barrier layer, named after the layer.

    Every configured layer's flag column is always computed (spec § 3);
    which of them OR into `barrier` is neighbors.combine_barrier_flags'
    job.
    """
    out = polygon_gdf.copy()
    for name, barrier_gdf in barriers.items():
        out = _flag_one(out, barrier_gdf, name, id_col)
    return out


def distance_to_point_km(polygon_gdf, point, *, centroid_col="centroid",
                         out_col="ndmc_dist_km"):
    """Distance in km from `point` to each row's centroid (the NDMC column)."""
    out = polygon_gdf.copy()
    out[out_col] = 0.0
    for idx, row in out.iterrows():
        out.loc[idx, out_col] = point.distance(row[centroid_col]) / 1000
    return out
