"""Neighbour construction: adjacency rule, barrier rule, centroid distances.

Pure functions with explicit keyword arguments — never imports
delhi_psi.config. The `bbox` adjacency path and the `global_asymmetric`
barrier path are copied verbatim from spatial_index_utils'
add_polygon_neighbors_column_fast (split into two passes, which selects the
same ids); `touch` and `pairwise` implement the manuscript's border-sharing
and pair-severing rules.
"""

import logging

import geopandas as gpd
import numpy as np
from tqdm import tqdm

from delhi_psi.geometry import bbox_frame, row_index

log = logging.getLogger(__name__)


def combine_barrier_flags(polygon_gdf, *, layers, combine, out_col="barrier"):
    """OR the selected per-layer flag columns into `out_col`.

    combine == "any" uses every configured layer; otherwise it is a sequence
    of layer names. Every configured layer's own flag column is left intact.
    """
    out = polygon_gdf.copy()
    selected = tuple(layers) if combine == "any" else tuple(combine)
    unknown = [name for name in selected if name not in layers]
    if unknown:
        raise ValueError(
            f"barrier.combine names layers that are not configured: {unknown}; "
            f"configured layers: {sorted(layers)}")
    flag = None
    for name in selected:
        column = out[name].fillna(False).astype(bool)
        flag = column if flag is None else (flag | column)
    out[out_col] = False if flag is None else flag
    return out


def _adjacency_bbox(polygon_gdf, id_col, neighbor_col):
    """Production's spatial join of polygons against bounding boxes."""
    right_gdf = gpd.GeoDataFrame(bbox_frame(polygon_gdf),
                                 crs=polygon_gdf.crs)
    joined_gdf = gpd.sjoin(polygon_gdf, right_gdf, how="left")

    id_col_left = id_col + "_left"
    id_col_right = id_col + "_right"
    joined_grouped = joined_gdf.groupby(id_col_left)

    out = polygon_gdf.copy()
    out[neighbor_col] = np.empty((len(out), 0)).tolist()

    for group in tqdm(joined_grouped.groups):
        group_list = list(joined_grouped.get_group(group)[id_col_right])
        # a polygon intersects itself
        group_list.remove(group)
        group_idx = row_index(out, id_col, group)
        out.loc[group_idx, neighbor_col].extend(group_list)
    return out


def _adjacency_touch(polygon_gdf, id_col, neighbor_col):
    """Border sharing: the intersection must be a line of positive length."""
    out = polygon_gdf.copy()
    out[neighbor_col] = np.empty((len(out), 0)).tolist()
    geoms = {row[id_col]: row["geometry"] for _, row in out.iterrows()}
    for idx, row in tqdm(out.iterrows(), total=len(out)):
        i = row[id_col]
        for j, other in geoms.items():
            if i == j:
                continue
            shared = geoms[i].intersection(other)
            if not shared.is_empty and shared.length > 0:
                out.loc[idx, neighbor_col].append(j)
    return out


def adjacency(polygon_gdf, *, id_col="USO_AREA_U", neighbor_col="nbrs_bbox",
              rule="bbox"):
    """Directed neighbour lists under `rule` ("bbox" or "touch").

    The column keeps its historical name `nbrs_bbox` under both rules — it is
    part of the July 2025 baseline's column contract (spec § 5).
    """
    if rule == "bbox":
        return _adjacency_bbox(polygon_gdf, id_col, neighbor_col)
    if rule == "touch":
        return _adjacency_touch(polygon_gdf, id_col, neighbor_col)
    raise ValueError(
        f"unknown adjacency rule {rule!r}; allowed values: ['bbox', 'touch']")


def apply_barrier(polygon_gdf, barrier_geoms, *, id_col="USO_AREA_U",
                  neighbor_col="nbrs_bbox", rule="global_asymmetric",
                  flag_col="barrier"):
    """Sever neighbour links across barriers.

    global_asymmetric: drop every neighbour whose `flag_col` is True — the
        production rule (a per-polygon flag, so severing is one-directional).
    pairwise: drop j from i's list when a barrier geometry intersects the
        boundary i and j share — the manuscript rule.
    """
    if rule not in ("global_asymmetric", "pairwise"):
        raise ValueError(
            f"unknown barrier rule {rule!r}; allowed values: "
            "['global_asymmetric', 'pairwise']")
    out = polygon_gdf.copy()
    if not barrier_geoms:
        return out
    geoms = {row[id_col]: row["geometry"] for _, row in out.iterrows()}
    flags = {row[id_col]: bool(row[flag_col]) for _, row in out.iterrows()} \
        if rule == "global_asymmetric" else {}

    for idx, row in out.iterrows():
        i = row[id_col]
        kept = []
        for j in row[neighbor_col]:
            if rule == "global_asymmetric":
                if not flags[j]:
                    kept.append(j)
            else:
                shared = geoms[i].intersection(geoms[j])
                if not any(b.intersects(shared) for b in barrier_geoms):
                    kept.append(j)
        out.at[idx, neighbor_col] = kept
    return out


def centroid_distances(polygon_gdf, *, neighbor_col="nbrs_bbox",
                       nbr_dist_col="nbrs_dist_bbox",
                       centroid_col="centroid", id_col="USO_AREA_U"):
    """Add [(neighbor_id, distance_km), ...] per row (verbatim calc_nbr_dist)."""
    gdf_copy = polygon_gdf.copy()
    gdf_copy[nbr_dist_col] = np.empty((len(gdf_copy), 0)).tolist()

    with tqdm(total=len(gdf_copy)) as pbar:
        for idx, row in gdf_copy.iterrows():
            row_centroid = row[centroid_col]
            neighbor_ids = row[neighbor_col]

            for neighbor_id in neighbor_ids:
                neighbor_row = gdf_copy[gdf_copy[id_col] == neighbor_id]
                neighbor_centroid = neighbor_row[centroid_col].array[0]
                neighbor_distance = row_centroid.distance(neighbor_centroid)
                neighbor_distance = neighbor_distance / 1000
                gdf_copy.loc[idx, nbr_dist_col].append(
                    (neighbor_id, neighbor_distance))

            pbar.update(1)

    return gdf_copy
