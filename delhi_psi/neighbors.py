"""Neighbour construction: adjacency rules (`bbox`, `touch`,
`within_distance`), barrier rule, and distances in two definitions
(centroid and boundary).

Pure functions with explicit keyword arguments — never imports
delhi_psi.config. The `bbox` adjacency path and the `global_asymmetric`
barrier path are copied verbatim from spatial_index_utils'
add_polygon_neighbors_column_fast (split into two passes, which selects the
same ids); `touch` and `pairwise` implement the manuscript's border-sharing
and pair-severing rules. `partial_weighted` discounts a neighbour's
contribution by the covered share of the boundary it shares with i instead
of severing the link (DEL-48, spec § 2.1) — see `partial_weight`.
"""

import logging

import geopandas as gpd
import numpy as np
from shapely import STRtree
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union
from tqdm import tqdm

from delhi_psi.geometry import bbox_frame, row_index

log = logging.getLogger(__name__)

_POLYGONAL = frozenset({"Polygon", "MultiPolygon"})
_LINEAL = frozenset({"LineString", "LinearRing", "MultiLineString"})


def selected_layers(layers, combine):
    """The layer names `combine` selects: every configured layer for "any"."""
    selected = tuple(layers) if combine == "any" else tuple(combine)
    unknown = [name for name in selected if name not in layers]
    if unknown:
        raise ValueError(
            f"barrier.combine names layers that are not configured: {unknown}; "
            f"configured layers: {sorted(layers)}")
    return selected


def selected_barrier_geoms(barriers, *, combine):
    """The geometries of the layers `combine` selects, in layer order.

    `pairwise` and `partial_weighted` read GEOMETRIES, so `combine` has to
    choose which layers they see — which is what the methodology stamp's
    "`combine` decides who is severed" already claims for every rule (spec
    § 2.6). `global_asymmetric` reads the flag column `combine_barrier_flags`
    builds and never comes here.
    """
    return [geom for name in selected_layers(barriers, combine)
            for geom in barriers[name].geometry]


def shared_boundary(geom_i, geom_j):
    """The boundary i and j share: SB_ij (spec § 2.1 steps 1-2).

    The boundary of every POLYGONAL component of the intersection — the
    owner's overlap rule, "the shared boundary of an overlapping pair is the
    boundary of the intersection polygon" — plus every LINEAL component.
    Point components contribute nothing.

    A GeometryCollection is decomposed into its parts FIRST. This is the one
    place a naive `shared.boundary` would be wrong (shapely returns None for
    a collection), and it is a real-layer case: a polygon that overlaps its
    neighbour on one side and shares an edge on another.

    Returns a possibly EMPTY geometry, so callers test `.length`, never
    `is None`.
    """
    shared = geom_i.intersection(geom_j)
    if shared.is_empty:
        return shared
    parts = list(shared.geoms) if hasattr(shared, "geoms") else [shared]
    pieces = [part.boundary if part.geom_type in _POLYGONAL else part
              for part in parts
              if part.geom_type in _POLYGONAL or part.geom_type in _LINEAL]
    return unary_union(pieces) if pieces else GeometryCollection()


def partial_weight(shared, buffered_barriers):
    """w = 1 - L_blocked / L_shared (spec § 2.1 steps 3-5).

    `buffered_barriers` are ALREADY buffered: the caller builds them once.
    The blocked PIECES are unioned before their length is taken, so two
    overlapping barrier buffers never count the same metre twice.

    L_shared == 0 (an empty intersection, or a corner-only contact) means
    there is nothing to block, so w = 1 — the owner's "a point crossing
    severs nothing". The `>=` guards the float case where the intersection
    returns the whole boundary plus a rounding hair; nothing is rounded.
    """
    length = shared.length
    if length == 0:
        return 1.0
    pieces = [piece for piece in
              (shared.intersection(b) for b in buffered_barriers)
              if not piece.is_empty]
    if not pieces:
        return 1.0
    covered = unary_union(pieces).length
    return 0.0 if covered >= length else 1 - covered / length


def combine_barrier_flags(polygon_gdf, *, layers, combine, out_col="barrier"):
    """OR the selected per-layer flag columns into `out_col`.

    combine == "any" uses every configured layer; otherwise it is a sequence
    of layer names. Every configured layer's own flag column is left intact.
    """
    out = polygon_gdf.copy()
    flag = None
    for name in selected_layers(layers, combine):
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


def _adjacency_within_distance(polygon_gdf, id_col, neighbor_col,
                               max_distance_km):
    """Polygon-to-polygon band: j is a neighbour of i iff their shortest
    distance is <= max_distance_km * 1000 metres (EPSG:7760 is metric).

    `dwithin` is symmetric and matches every polygon with ITSELF at every
    radius (distance 0), so the left join never yields a missing partner and
    the self pair is the only one that has to be removed. Lists are written
    in the frame's row order, like the other two rules.
    """
    if max_distance_km is None:
        raise ValueError(
            "adjacency rule 'within_distance' requires max_distance_km")
    joined_gdf = gpd.sjoin(polygon_gdf, polygon_gdf, how="left",
                           predicate="dwithin",
                           distance=max_distance_km * 1000)
    id_col_left = id_col + "_left"
    id_col_right = id_col + "_right"
    joined_grouped = joined_gdf.groupby(id_col_left)

    out = polygon_gdf.copy()
    out[neighbor_col] = np.empty((len(out), 0)).tolist()

    for group in tqdm(joined_grouped.groups):
        group_list = list(joined_grouped.get_group(group)[id_col_right])
        # a polygon is within any distance of itself
        group_list.remove(group)
        group_idx = row_index(out, id_col, group)
        out.loc[group_idx, neighbor_col].extend(group_list)
    return out


def adjacency(polygon_gdf, *, id_col="USO_AREA_U", neighbor_col="nbrs_bbox",
              rule="bbox", max_distance_km=None):
    """Directed neighbour lists under `rule` ("bbox", "touch" or
    "within_distance").

    The column keeps its historical name `nbrs_bbox` under EVERY rule — it is
    part of the July 2025 baseline's column contract (spec § 5).

    max_distance_km is used by `within_distance` alone; passing it with any
    other rule is an error, mirroring the config rule (`build_neighbors`
    forwards the configured value unconditionally, and it is None there).
    """
    if rule != "within_distance" and max_distance_km is not None:
        raise ValueError(
            "max_distance_km is only used by adjacency rule "
            f"'within_distance', not {rule!r}")
    if rule == "bbox":
        return _adjacency_bbox(polygon_gdf, id_col, neighbor_col)
    if rule == "touch":
        return _adjacency_touch(polygon_gdf, id_col, neighbor_col)
    if rule == "within_distance":
        return _adjacency_within_distance(polygon_gdf, id_col, neighbor_col,
                                          max_distance_km)
    raise ValueError(
        f"unknown adjacency rule {rule!r}; allowed values: "
        "['bbox', 'touch', 'within_distance']")


def apply_barrier(polygon_gdf, barrier_geoms, *, id_col="USO_AREA_U",
                  neighbor_col="nbrs_bbox", rule="global_asymmetric",
                  flag_col="barrier", buffer_m=None,
                  weight_col="nbrs_barrier_weight"):
    """Sever, or discount, neighbour links across barriers.

    global_asymmetric: drop every neighbour whose `flag_col` is True — the
        production rule (a per-polygon flag, so severing is one-directional).
    pairwise: drop j from i's list when a barrier geometry intersects the
        boundary i and j share — the manuscript rule.
    partial_weighted: keep j with weight w_ij = 1 - L_blocked/L_shared, and
        drop it only when w_ij == 0 (DEL-48, spec § 2.1). `neighbor_col`
        keeps its existing contract — the pruned list of ids — and the
        weights travel in `weight_col` as [(neighbor_id, w), ...] in the SAME
        order, the `nbrs_dist_bbox` 2-tuple shape. The column is written ONLY
        under this rule, so an artifact built under either other rule (and
        every artifact built before cycle 3E) is unchanged.

    buffer_m is used by `partial_weighted` alone: required by it, rejected by
    the other two, which have no buffer at all. It mirrors the config rule
    (`build_neighbors` forwards the configured value unconditionally, and it
    is None there).
    """
    if rule not in ("global_asymmetric", "pairwise", "partial_weighted"):
        raise ValueError(
            f"unknown barrier rule {rule!r}; allowed values: "
            "['global_asymmetric', 'pairwise', 'partial_weighted']")
    if rule == "partial_weighted":
        if buffer_m is None:
            raise ValueError(
                "barrier rule 'partial_weighted' requires buffer_m — the "
                "distance in metres within which a barrier blocks a boundary")
        if not buffer_m > 0:
            raise ValueError(
                f"buffer_m must be > 0, got {buffer_m!r}: a zero buffer is "
                "EMPTY in shapely and would make every weight 1 silently")
    elif buffer_m is not None:
        raise ValueError(
            "buffer_m is only used by barrier rule 'partial_weighted', not "
            f"{rule!r}")

    out = polygon_gdf.copy()
    if rule == "partial_weighted":
        # Buffer ONCE and index the buffers: per link only the candidates the
        # tree returns are intersected. The naive alternative — one
        # unary_union of every buffer, overlaid per link — scales each
        # overlay with the union's vertex count (spec § 2.2).
        buffered = [geom.buffer(buffer_m) for geom in barrier_geoms]
        tree = STRtree(buffered) if buffered else None
        geoms = {row[id_col]: row["geometry"] for _, row in out.iterrows()}
        out[weight_col] = np.empty((len(out), 0)).tolist()
        for idx, row in tqdm(out.iterrows(), total=len(out)):
            kept, weights = [], []
            for j in row[neighbor_col]:
                shared = shared_boundary(geoms[row[id_col]], geoms[j])
                if tree is None or shared.length == 0:
                    weight = 1.0
                else:
                    weight = partial_weight(
                        shared, [buffered[k] for k in tree.query(shared)])
                if weight > 0.0:
                    kept.append(j)
                    weights.append((j, weight))
            out.at[idx, neighbor_col] = kept
            out.at[idx, weight_col] = weights
        return out

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


def boundary_distances(polygon_gdf, *, neighbor_col="nbrs_bbox",
                       nbr_dist_col="nbrs_dist_boundary",
                       id_col="USO_AREA_U"):
    """Add [(neighbor_id, distance_km), ...] per row, measured POLYGON TO
    POLYGON — 0 for every touching or overlapping neighbour.

    Same OUTPUT shape as `centroid_distances`, but built over an
    id -> geometry dict made ONCE (the `_adjacency_touch` pattern), never the
    per-neighbour boolean-mask lookup `centroid_distances` inherited from the
    2025 script. On a MultiPolygon shapely's `distance` is the minimum over
    the parts, which is the intended meaning.
    """
    out = polygon_gdf.copy()
    out[nbr_dist_col] = np.empty((len(out), 0)).tolist()
    geoms = {row[id_col]: row["geometry"] for _, row in out.iterrows()}
    for idx, row in tqdm(out.iterrows(), total=len(out)):
        geom = geoms[row[id_col]]
        out.at[idx, nbr_dist_col] = [
            (neighbor_id, geom.distance(geoms[neighbor_id]) / 1000)
            for neighbor_id in row[neighbor_col]]
    return out
