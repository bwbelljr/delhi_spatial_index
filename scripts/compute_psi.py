"""PSI pipeline: neighbors joblib + population + services -> PSI outputs.

Mechanical port of 'Colonies Public Services Index Calculations Updated
(no RV) 2025.ipynb' (deleted in Phase 1; see git history). No logic changes.
"""

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spatial_index_utils  # noqa: E402
from spatial_index_utils import calc_all_services  # noqa: E402
from scripts.common import resolve_data_dir, resolve_out_dir  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="input data root")
    parser.add_argument(
        "--out-dir", default=None, help="output directory (default: data dir)"
    )
    parser.add_argument(
        "--neighbors-file",
        default=None,
        help="neighbors joblib from preprocess.py "
        "(default: <data-dir>/colonies_bbox_nbrs2025.joblib)",
    )
    args = parser.parse_args()
    data_dir = str(resolve_data_dir(args.data_dir))
    out_dir = str(resolve_out_dir(args.out_dir, data_dir))

    # WGS 84 / Delhi
    epsg_code = 7760

    colonies_bbox_file = (
        str(Path(args.neighbors_file).expanduser())
        if args.neighbors_file
        else os.path.join(data_dir, "colonies_bbox_nbrs2025.joblib")
    )

    popfile_2020 = os.path.join(data_dir, "pop_colony_wp_2020_jjc_adjusted.csv")
    for popfile in [popfile_2020]:
        print(f"{popfile} exists: {os.path.exists(popfile)}")

    services_dir = os.path.join(data_dir, "Public Services")
    bank_fp = os.path.join(services_dir, "Banking", "Banking.shp")
    health_fp = os.path.join(services_dir, "Health", "Health.shp")
    road_fp = os.path.join(services_dir, "Major Road", "Road.shp")
    police_fp = os.path.join(services_dir, "Police", "Police Station.shp")
    ration_fp = os.path.join(services_dir, "Ration", "Ration.shp")
    school_fp = os.path.join(services_dir, "School", "schools7760.shp")
    transport_fp = os.path.join(services_dir, "Transport", "Transport.shp")

    # Dated like every other output so a default --out-dir run can never
    # collide with the baseline's missing_colonies.csv
    missing_colonies_csv_path = os.path.join(out_dir, "missing_colonies_aug2026.csv")

    delhi_bounds_filepath = os.path.join(
        data_dir, "delhi_bounds_buffer", "delhi_bounds_buffer.shp"
    )

    filepath_list = [
        bank_fp,
        health_fp,
        road_fp,
        police_fp,
        ration_fp,
        school_fp,
        transport_fp,
        delhi_bounds_filepath,
    ]
    for filepath in filepath_list:
        if not os.path.exists(filepath):
            print("{} does not exist".format(filepath))

    psi_results_dir = os.path.join(out_dir, "psi_2020_results")
    os.makedirs(psi_results_dir, exist_ok=True)
    colonies_bbox_psi_csv_file = os.path.join(
        psi_results_dir, "delhi_psi_bbox_popsize2020_norv_aug2026.csv"
    )
    colonies_bbox_psi_joblib_file = os.path.join(
        psi_results_dir, "colonies_bbox_psi_popsize2020_norv_aug2026.joblib"
    )
    colonies_bbox_psi_popdensity_csv_file = os.path.join(
        psi_results_dir, "delhi_psi_bbox_popdensity2020_norv_aug2026.csv"
    )
    colonies_bbox_psi_popdensity_joblib_file = os.path.join(
        psi_results_dir, "colonies_bbox_psi_popdensity2020_norv_aug2026.joblib"
    )
    colonies_bbox_psi_popsize_file = os.path.join(
        psi_results_dir, "delhi_psi_bbox_popsize2020_norv_aug2026.shp"
    )
    colonies_bbox_psi_popdensity_file = os.path.join(
        psi_results_dir, "delhi_psi_bbox_popdensity2020_norv_aug2026.shp"
    )

    with open(colonies_bbox_file, "rb") as f:
        colonies_bbox_nbrs = joblib.load(f)

    updated_pop = pd.read_csv(popfile_2020)

    # Import services
    bank = gpd.read_file(bank_fp)
    health = gpd.read_file(health_fp)
    road = gpd.read_file(road_fp)
    police = gpd.read_file(police_fp)
    ration = gpd.read_file(ration_fp)
    school = gpd.read_file(school_fp)
    transport = gpd.read_file(transport_fp)

    print("colonies:", len(colonies_bbox_nbrs))

    # rename population column to make distinct in upcoming merge
    updated_pop = updated_pop.rename(columns={"population": "population_new"})
    updated_pop = updated_pop[["population_new", "uso_area_u"]]

    # Left merge updated population data with colonies data
    colonies_bbox_nbrs = colonies_bbox_nbrs.merge(
        updated_pop, how="left", left_on="USO_AREA_U", right_on="uso_area_u"
    )

    colonies_with_missing_population = colonies_bbox_nbrs[
        colonies_bbox_nbrs["population_new"].isna()
    ]
    colonies_with_missing_population.to_csv(missing_colonies_csv_path, index=False)
    colonies_bbox_nbrs.drop(index=colonies_with_missing_population.index, inplace=True)

    colonies_bbox_nbrs = colonies_bbox_nbrs.drop(columns=["uso_area_u"])
    colonies_bbox_nbrs = colonies_bbox_nbrs.rename(
        columns={"population_new": "population"}
    )

    print("colonies with population:", len(colonies_bbox_nbrs))
    print(
        "missing population estimates:",
        sum(colonies_bbox_nbrs["population"].isna()),
    )

    colonies_bbox_nbrs = colonies_bbox_nbrs[colonies_bbox_nbrs["USO_FINAL"] != "RV"]
    print("colonies after RV exclusion:", len(colonies_bbox_nbrs))

    spatial_index_utils.check_shapefile(
        gdf=bank,
        gdf_name="bank",
        geom_type="Point",
        delhi_bounds_filepath=delhi_bounds_filepath,
    )
    print("bank shape before dedup:", bank.shape)
    # Remove duplicate rows found in bank DataFrame
    bank.drop_duplicates(inplace=True)
    print("bank shape after dedup:", bank.shape)

    spatial_index_utils.check_shapefile(
        gdf=health,
        gdf_name="health",
        geom_type="Point",
        delhi_bounds_filepath=delhi_bounds_filepath,
    )
    spatial_index_utils.check_shapefile(
        gdf=road,
        gdf_name="road",
        geom_type="Line",
        delhi_bounds_filepath=delhi_bounds_filepath,
    )
    spatial_index_utils.check_shapefile(
        gdf=police,
        gdf_name="police",
        geom_type="Point",
        delhi_bounds_filepath=delhi_bounds_filepath,
    )
    spatial_index_utils.check_shapefile(
        gdf=ration,
        gdf_name="ration",
        geom_type="Point",
        delhi_bounds_filepath=delhi_bounds_filepath,
    )
    spatial_index_utils.check_shapefile(
        gdf=school,
        gdf_name="school",
        geom_type="Point",
        delhi_bounds_filepath=delhi_bounds_filepath,
    )
    spatial_index_utils.check_shapefile(
        gdf=transport,
        gdf_name="transport",
        geom_type="Point",
        delhi_bounds_filepath=delhi_bounds_filepath,
    )

    print(
        "service CRS equal:",
        bank.crs
        == health.crs
        == road.crs
        == police.crs
        == ration.crs
        == school.crs
        == transport.crs,
    )
    print("colonies CRS == bank CRS:", colonies_bbox_nbrs.crs == bank.crs)

    # Define all point services as dictionary
    point_services = {
        "bank": bank,
        "health": health,
        "police": police,
        "ration": ration,
        "school": school,
        "transport": transport,
    }
    line_services = {"road": road}

    colonies_bbox_psi_popsize = calc_all_services(
        polygon_gdf=colonies_bbox_nbrs,
        point_services=point_services,
        line_services=line_services,
        epsg_code=epsg_code,
        pcen_denom="pop",
        nbr_dist_colname="nbrs_dist_bbox",
    )
    colonies_bbox_psi_popsize = colonies_bbox_psi_popsize.rename(
        columns={"road_count": "road_length"}
    )

    colonies_bbox_psi_popdensity = calc_all_services(
        polygon_gdf=colonies_bbox_nbrs,
        point_services=point_services,
        line_services=line_services,
        epsg_code=epsg_code,
        pcen_denom="popdensity",
        nbr_dist_colname="nbrs_dist_bbox",
    )
    colonies_bbox_psi_popdensity = colonies_bbox_psi_popdensity.rename(
        columns={"road_count": "road_length"}
    )

    # Sanity checks (ported from notebook): negative values in derived columns
    for suffix in ["_count", "_pcen", "_idx"]:
        cols = [
            colname
            for colname in colonies_bbox_psi_popdensity.columns
            if colname.endswith(suffix)
        ]
        for col in cols:
            n_negative = len(
                colonies_bbox_psi_popdensity[colonies_bbox_psi_popdensity[col] < 0]
            )
            print(
                "There are",
                n_negative,
                "negative values in",
                col,
                "column",
            )

    print(colonies_bbox_psi_popdensity["bank_idx"].describe())
    print(colonies_bbox_psi_popdensity["unnorm_psi"].describe())
    print(colonies_bbox_psi_popdensity["norm_psi"].describe())

    bbox_drop_columns = ["nbrs_bbox", "nbrs_dist_bbox", "centroid"]
    colonies_bbox_psi_popsize.drop(columns=bbox_drop_columns).to_file(
        colonies_bbox_psi_popsize_file
    )
    colonies_bbox_psi_popdensity.drop(columns=bbox_drop_columns).to_file(
        colonies_bbox_psi_popdensity_file
    )

    colonies_bbox_psi_popsize.to_csv(colonies_bbox_psi_csv_file)
    colonies_bbox_psi_popdensity.to_csv(colonies_bbox_psi_popdensity_csv_file)

    with open(colonies_bbox_psi_joblib_file, "wb") as f:
        joblib.dump(colonies_bbox_psi_popsize, f)
    with open(colonies_bbox_psi_popdensity_joblib_file, "wb") as f:
        joblib.dump(colonies_bbox_psi_popdensity, f)
    print("wrote outputs to", psi_results_dir)


if __name__ == "__main__":
    main()
