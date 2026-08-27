"""Pre-processing pipeline: colonies + barriers -> bbox neighbors joblib.

Mechanical port of 'Colonies Dataset Pre-Processing (2025).ipynb' (deleted in
Phase 1; see git history). No logic changes.
"""

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spatial_index_utils  # noqa: E402
from scripts.common import resolve_data_dir, resolve_out_dir  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="input data root")
    parser.add_argument(
        "--out-dir", default=None, help="output directory (default: data dir)"
    )
    args = parser.parse_args()
    data_dir = str(resolve_data_dir(args.data_dir))
    out_dir = str(resolve_out_dir(args.out_dir, data_dir))

    # WGS 84 / Delhi
    epsg_code = 7760

    colony_filepath = os.path.join(
        data_dir, "uso_update_sep2021", "uso_update_sep2021.shp"
    )
    barrier_directory = os.path.join(data_dir, "Barrier_Clip")
    canal_filepath = os.path.join(barrier_directory, "Canal", "Canal.shp")
    drain_filepath = os.path.join(barrier_directory, "Drain", "Major_Drain.shp")
    railway_filepath = os.path.join(barrier_directory, "Railway", "Railway_Line.shp")
    ndmc_center_filepath = os.path.join(
        data_dir, "ndmc_center7760", "ndmc_center7760.shp"
    )
    delhi_bounds_filepath = os.path.join(
        data_dir, "delhi_bounds_buffer", "delhi_bounds_buffer.shp"
    )

    filepath_list = [
        colony_filepath,
        canal_filepath,
        drain_filepath,
        railway_filepath,
        delhi_bounds_filepath,
        ndmc_center_filepath,
    ]
    for filepath in filepath_list:
        if not os.path.exists(filepath):
            print("{} does not exist".format(filepath))

    canal_deduplicated_dir = os.path.join(data_dir, "canal.data")
    drain_deduplicated_dir = os.path.join(data_dir, "drain.data")
    railway_deduplicated_dir = os.path.join(data_dir, "railway.data")
    colonies_deduplicated_dir = os.path.join(data_dir, "colonies.data")
    colonies_deduplicated_csv = os.path.join(
        data_dir, "colonies_no_duplicates_July2025.csv"
    )

    colonies_bbox_joblib_path = os.path.join(
        out_dir, "colonies_bbox_nbrs_aug2026.joblib"
    )

    colonies = gpd.read_file(colony_filepath)
    canal = gpd.read_file(canal_filepath)
    drain = gpd.read_file(drain_filepath)
    railway = gpd.read_file(railway_filepath)
    ndmc_center = gpd.read_file(ndmc_center_filepath)

    spatial_index_utils.check_shapefile(
        gdf=colonies,
        gdf_name="colonies",
        geom_type="Polygon",
        delhi_bounds_filepath=delhi_bounds_filepath,
    )
    spatial_index_utils.check_shapefile(
        gdf=canal,
        gdf_name="canal",
        geom_type="Line",
        delhi_bounds_filepath=delhi_bounds_filepath,
    )
    spatial_index_utils.check_shapefile(
        gdf=drain,
        gdf_name="drain",
        geom_type="Line",
        delhi_bounds_filepath=delhi_bounds_filepath,
    )
    spatial_index_utils.check_shapefile(
        gdf=railway,
        gdf_name="railway",
        geom_type="Line",
        delhi_bounds_filepath=delhi_bounds_filepath,
    )

    if not os.path.exists(canal_deduplicated_dir):
        canal = spatial_index_utils.remove_duplicate_geom(canal)
        canal.to_file(canal_deduplicated_dir, index=False)

    if not os.path.exists(drain_deduplicated_dir):
        drain = spatial_index_utils.remove_duplicate_geom(drain)
        drain.to_file(drain_deduplicated_dir, index=False)

    if not os.path.exists(railway_deduplicated_dir):
        railway = spatial_index_utils.remove_duplicate_geom(railway)
        railway.to_file(railway_deduplicated_dir, index=False)

    if not os.path.exists(colonies_deduplicated_dir):
        colonies = spatial_index_utils.remove_duplicate_geom(colonies)
        colonies.to_file(colonies_deduplicated_dir, index=False)
        colonies.to_csv(colonies_deduplicated_csv, index=False)

    print("colonies:", len(colonies))
    print("colonies crs:", colonies.crs)

    colonies = spatial_index_utils.reproject_gdf(colonies, epsg_code)
    canal = spatial_index_utils.reproject_gdf(canal, epsg_code)
    drain = spatial_index_utils.reproject_gdf(drain, epsg_code)
    railway = spatial_index_utils.reproject_gdf(railway, epsg_code)

    print(
        "all CRS equal:",
        colonies.crs == drain.crs == canal.crs == railway.crs,
    )

    colonies["area_km2"] = colonies.area / 1000000
    print("area_km2 max:", colonies["area_km2"].max())
    print("area_km2 min:", colonies["area_km2"].min())

    def remove_select_cols(gdf):
        candidate_cols = {"index", "level_0"}
        cols_to_remove = candidate_cols.intersection(gdf.columns)
        return gdf.drop(columns=cols_to_remove)

    colonies = remove_select_cols(colonies)
    drain = remove_select_cols(drain)
    canal = remove_select_cols(canal)
    railway = remove_select_cols(railway)

    # Create new columns showing intersection with canal, railway and drain
    colonies = spatial_index_utils.barrier_intersection(colonies, canal, "canal")
    colonies = spatial_index_utils.barrier_intersection(colonies, railway, "railway")
    colonies = spatial_index_utils.barrier_intersection(colonies, drain, "drain")

    # Create barrier column as being intersection with canal, railway or drain
    colonies["barrier"] = colonies["canal"] | colonies["railway"] | colonies["drain"]

    colonies["centroid"] = colonies.centroid

    # Extract NDMC Center as Shapely Point
    ndmc_center_point = ndmc_center["geometry"].values[0]

    # Compute distance from NDMC to centroid of each polygon (kilometers)
    colonies["ndmc_dist_km"] = 0.0
    for idx, row in colonies.iterrows():
        colonies.loc[idx, "ndmc_dist_km"] = (
            ndmc_center_point.distance(row["centroid"]) / 1000
        )

    colonies_bbox = spatial_index_utils.create_bbox_gdf(colonies)
    colonies_bbox_updated = gpd.GeoDataFrame(colonies_bbox, crs=colonies.crs)

    colonies_bbox_nbrs = spatial_index_utils.add_polygon_neighbors_column_fast(
        polygon_gdf=colonies,
        right_gdf=colonies_bbox_updated,
        id_colname="USO_AREA_U",
        neighbor_colname="nbrs_bbox",
        barrier_colname="barrier",
    )

    colonies_bbox_nbrs = spatial_index_utils.calc_nbr_dist(
        polygon_gdf=colonies_bbox_nbrs,
        nbr_dist_colname="nbrs_dist_bbox",
        centroid_colname="centroid",
        neighbor_colname="nbrs_bbox",
        neighbor_id_col="USO_AREA_U",
    )

    colonies_bbox_nbrs["index"] = colonies_bbox_nbrs.index
    colonies_bbox_nbrs = colonies_bbox_nbrs.drop(columns=["geom_type"])

    with open(colonies_bbox_joblib_path, "wb") as f:
        joblib.dump(colonies_bbox_nbrs, f)
    print("wrote", colonies_bbox_joblib_path)


if __name__ == "__main__":
    main()
