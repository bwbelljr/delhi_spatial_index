"""Loaders for the Oraculum fixtures + the production-chain helper.

run_production_chain mirrors, call for call, how scripts/preprocess.py and
scripts/compute_psi.py used to drive spatial_index_utils — so the
library-first oracle tests exercise exactly the production wiring on fixture
data, now through delhi_psi.
"""

from pathlib import Path

import geopandas as gpd

from delhi_psi import geometry, index, neighbors

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

    The step-0 production snapshot's backend. It mirrors, call for call, the
    wiring the snapshot was generated with — now through delhi_psi rather than
    the deleted spatial_index_utils. Retired once the generator swaps to
    pipeline.compute_frames and the diff is proven a no-op.

    drop_ids_post: ids removed AFTER neighbor computation (the scripts'
    post-drop semantics — e.g. {'RV'} replicates compute_psi's RV filter).
    """
    colonies = geometry.barrier_flags(settlements.copy(),
                                      {"canal": barriers})
    colonies["barrier"] = colonies["canal"]
    colonies["centroid"] = colonies.centroid

    nbrs = neighbors.adjacency(colonies, id_col="USO_AREA_U",
                               neighbor_col="nbrs_bbox", rule="bbox")
    nbrs = neighbors.apply_barrier(nbrs, list(barriers.geometry),
                                   id_col="USO_AREA_U",
                                   neighbor_col="nbrs_bbox",
                                   rule="global_asymmetric",
                                   flag_col="barrier")
    nbrs = neighbors.centroid_distances(
        nbrs, neighbor_col="nbrs_bbox", nbr_dist_col="nbrs_dist_bbox",
        centroid_col="centroid", id_col="USO_AREA_U")
    nbrs["index"] = nbrs.index

    if drop_ids_post:
        nbrs = nbrs[~nbrs["USO_AREA_U"].isin(drop_ids_post)]

    layout = [(name, "line" if name == "road" else "point")
              for name in services]
    layout.sort(key=lambda item: item[1] == "line")

    out = nbrs
    for service, kind in layout:
        amount_col = index.service_amount_column(service, kind)
        projected = geometry.reproject(services[service], EPSG)
        if kind == "point":
            out = index.point_counts(out, projected, count_col=amount_col)
        else:
            out = index.road_lengths(out, projected, length_col=amount_col)
        out = index.service_index(out, amount_col, service=service,
                                  denominator=pcen_denom,
                                  nbr_dist_col="nbrs_dist_bbox",
                                  absent_neighbor="swallowed")
    return index.overall_psi(out, second_normalization=True)


# --- profile-driven helpers (Phase 3A) --------------------------------
# The § 7 scenario table: a profile plus `types`/`stage` overrides ONLY.
# `absent_neighbor` always comes from the profile, because in the reference
# it is a rule-set property, not a scenario property.
ORACLE_SCENARIOS = [
    # (reference scenario, exclusion.types, exclusion.stage)
    ("baseline", (), "post_neighbors"),
    ("excl_rv_only", ("RV",), "post_neighbors"),
    ("excl_contributing", ("RV", "IND"), "post_neighbors"),
    ("excl_removed", ("RV", "IND"), "pre_neighbors"),
    ("excl_ind_removed", ("IND",), "pre_neighbors"),
]


def methodology_with(profile, *, types=None, stage=None):
    """The shipped profile's methodology with the two allowed overrides."""
    from dataclasses import replace

    from delhi_psi.config import ExclusionStage, load_config

    methodology = load_config(profile).methodology
    exclusion = methodology.exclusion
    if types is not None:
        exclusion = replace(exclusion, types=tuple(types))
    if stage is not None:
        exclusion = replace(exclusion, stage=ExclusionStage(stage))
    return replace(methodology, exclusion=exclusion)


def compute_oracle_frame(profile, *, types, stage, denom):
    """compute_frames on the Oraculum city, indexed by settlement id.

    The fixture city carries its own `population` column, so population=None.
    """
    from delhi_psi.pipeline import compute_frames

    return compute_frames(
        load_settlements(), {"canal": load_barriers()}, load_services(),
        None, methodology_with(profile, types=types, stage=stage), denom,
    ).set_index("USO_AREA_U")
