"""Loaders for the Oraculum fixtures + the production-chain helper.

run_production_chain mirrors, call for call, how scripts/preprocess.py and
scripts/compute_psi.py drive spatial_index_utils — so the library-first
oracle tests exercise exactly the production wiring on fixture data.
"""

from pathlib import Path

import geopandas as gpd

import spatial_index_utils

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "oraculum"
EPSG = 7760


def _read(path):
    gdf = gpd.read_file(path)
    return gdf.set_crs(epsg=EPSG, allow_override=True)


def load_settlements():
    return _read(FIXTURES / "settlements.geojson")


def load_barriers():
    return _read(FIXTURES / "barriers.geojson")


def load_services():
    gdf = _read(FIXTURES / "services.geojson")
    return {name: grp.reset_index(drop=True) for name, grp in gdf.groupby("service")}


def load_exhibit():
    return _read(FIXTURES / "divergence" / "exhibit.geojson")


def run_production_chain(settlements, barriers, services, pcen_denom,
                         drop_ids_post=frozenset()):
    """Preprocess-style neighbor computation + compute_psi-style indexing.

    drop_ids_post: ids removed AFTER neighbor computation (the scripts'
    post-drop semantics — e.g. {'RV'} replicates compute_psi's RV filter).
    """
    colonies = settlements.copy()
    colonies = spatial_index_utils.barrier_intersection(colonies, barriers, "canal")
    colonies["barrier"] = colonies["canal"]
    colonies["centroid"] = colonies.centroid

    colonies_bbox = spatial_index_utils.create_bbox_gdf(colonies)
    colonies_bbox = gpd.GeoDataFrame(colonies_bbox, crs=colonies.crs)

    nbrs = spatial_index_utils.add_polygon_neighbors_column_fast(
        polygon_gdf=colonies, right_gdf=colonies_bbox,
        id_colname="USO_AREA_U", neighbor_colname="nbrs_bbox",
        barrier_colname="barrier")
    nbrs = spatial_index_utils.calc_nbr_dist(
        polygon_gdf=nbrs, nbr_dist_colname="nbrs_dist_bbox",
        centroid_colname="centroid", neighbor_colname="nbrs_bbox",
        neighbor_id_col="USO_AREA_U")
    nbrs["index"] = nbrs.index

    if drop_ids_post:
        nbrs = nbrs[~nbrs["USO_AREA_U"].isin(drop_ids_post)]

    point_services = {k: v for k, v in services.items() if k != "road"}
    line_services = {"road": services["road"]}
    return spatial_index_utils.calc_all_services(
        polygon_gdf=nbrs, point_services=point_services,
        line_services=line_services, epsg_code=EPSG,
        pcen_denom=pcen_denom, nbr_dist_colname="nbrs_dist_bbox")
