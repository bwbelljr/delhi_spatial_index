"""Pipeline stages. `compute_frames` is the in-memory seam the oracle uses;
`preprocess` / `compute` (added next) are the path-based stages.

This is the ONLY module that sees a Config. The math modules take explicit
keyword arguments.
"""

import logging

from delhi_psi import geometry, index, neighbors, validate

log = logging.getLogger(__name__)

ID_COL = "USO_AREA_U"
TYPE_COL = "USO_FINAL"
NBRS_COL = "nbrs_bbox"
NBRS_DIST_COL = "nbrs_dist_bbox"
CENTROID_COL = "centroid"

POINT_GEOMS = frozenset({"Point", "MultiPoint"})
LINE_GEOMS = frozenset({"LineString", "MultiLineString", "LinearRing"})

# layers.population.missing — the same two values config.py validates
MISSING_POPULATION = ("drop", "error")


def service_kind(name, gdf):
    """Classify a service layer as "point" or "line" from its geometries."""
    kinds = set(gdf.geom_type.dropna().unique())
    if kinds and kinds <= POINT_GEOMS:
        return "point"
    if kinds and kinds <= LINE_GEOMS:
        return "line"
    raise ValueError(
        f"service layer {name!r}: cannot classify geometry types "
        f"{sorted(kinds)}; expected every geometry to be a point or a line")


def service_layout(services):
    """[(service, kind, amount_col)], point services before line services —
    the order production used, so output columns keep their familiar order."""
    layout = [(name, service_kind(name, gdf)) for name, gdf in services.items()]
    layout.sort(key=lambda item: item[1] == "line")
    return [(name, kind, index.service_amount_column(name, kind))
            for name, kind in layout]


def attach_population(settlements, population, *, id_col=ID_COL,
                      population_id_col="uso_area_u",
                      population_value_col="population"):
    """Attach a `population` column; return (frame, ids with no population).

    population=None means the settlements frame already carries the column
    (the oracle city does); the missing rule then applies to that column.
    Otherwise this is compute_psi.py's merge, verbatim: rename to
    population_new, keep two columns, left-merge, drop the join key, rename
    back. Nothing is dropped here — the exclusion `stage` decides when.
    """
    if population is None:
        out = settlements.copy()
    else:
        updated = population.rename(
            columns={population_value_col: "population_new"})
        updated = updated[["population_new", population_id_col]]
        out = settlements.merge(updated, how="left", left_on=id_col,
                                right_on=population_id_col)
        out = out.drop(columns=[population_id_col])
        out = out.rename(columns={"population_new": "population"})
    missing = frozenset(out.loc[out["population"].isna(), id_col])
    return out, missing


def excluded_ids(frame, *, types, id_col=ID_COL, type_col=TYPE_COL):
    """Ids whose settlement type is in `types` (raw USO_FINAL strings)."""
    if not types:
        return frozenset()
    return frozenset(frame.loc[frame[type_col].isin(list(types)), id_col])


def build_neighbors(settlements, barriers, methodology, *, epsg_code=7760,
                    id_col=ID_COL):
    """Barrier flags, combined flag, centroids, neighbour lists, distances.

    Always built on the FULL settlement universe — preprocess never excludes
    (spec § 3).
    """
    frame = geometry.barrier_flags(settlements, barriers, id_col=id_col)
    frame = neighbors.combine_barrier_flags(
        frame, layers=tuple(barriers), combine=methodology.barrier.combine)
    frame[CENTROID_COL] = frame.centroid

    frame = neighbors.adjacency(frame, id_col=id_col, neighbor_col=NBRS_COL,
                                rule=methodology.adjacency.rule)
    barrier_geoms = [geom for gdf in barriers.values() for geom in gdf.geometry]
    frame = neighbors.apply_barrier(frame, barrier_geoms, id_col=id_col,
                                    neighbor_col=NBRS_COL,
                                    rule=methodology.barrier.rule)
    frame = neighbors.centroid_distances(
        frame, neighbor_col=NBRS_COL, nbr_dist_col=NBRS_DIST_COL,
        centroid_col=CENTROID_COL, id_col=id_col)
    frame["index"] = frame.index
    return frame


def apply_exclusion(neighbor_frame, *, dropped, stage, id_col=ID_COL):
    """Return (universe, reported).

    universe — the frame neighbour AMOUNTS are read from.
    reported — the rows that get PCEN and index values.

    post_neighbors: excluded rows leave the reported frame; their ids stay in
        other settlements' neighbour lists (today's production).
    pre_neighbors: excluded ids are ALSO stripped from every neighbour list.
        Stripping is exactly what re-running adjacency on the reduced universe
        would give — adjacency and the barrier rules are pairwise, so removing
        a row removes precisely that id from every other list — and it is what
        the universe-wide stored artifact allows (spec § 3).
    """
    if stage not in ("post_neighbors", "pre_neighbors"):
        raise ValueError(
            f"unknown exclusion stage {stage!r}; allowed values: "
            "['post_neighbors', 'pre_neighbors']")
    universe = neighbor_frame.copy()
    if stage == "pre_neighbors" and dropped:
        for idx, row in universe.iterrows():
            universe.at[idx, NBRS_COL] = [
                j for j in row[NBRS_COL] if j not in dropped]
            universe.at[idx, NBRS_DIST_COL] = [
                (j, d) for j, d in row[NBRS_DIST_COL] if j not in dropped]
        universe = universe[~universe[id_col].isin(dropped)]
    reported = universe[~universe[id_col].isin(dropped)] if dropped else universe
    return universe, reported


def index_frames(neighbor_frame, services, methodology, denominator, *,
                 dropped=frozenset(), epsg_code=7760, id_col=ID_COL):
    """Amounts, PCEN, min-max and the overall PSI for one denominator."""
    exclusion = methodology.exclusion
    universe, _ = apply_exclusion(neighbor_frame, dropped=dropped,
                                  stage=exclusion.stage, id_col=id_col)

    # Own amounts are computed over the WHOLE universe, so excluded
    # settlements still have something to lend under absent_neighbor
    # "contributes". They are per-row independent, so computing them for rows
    # that are dropped a moment later cannot change a kept row's value.
    amounts = universe
    layout = service_layout(services)
    for service, kind, amount_col in layout:
        projected = geometry.reproject(services[service], epsg_code)
        if kind == "point":
            amounts = index.point_counts(amounts, projected,
                                         count_col=amount_col, id_col=id_col)
        else:
            amounts = index.road_lengths(amounts, projected,
                                         length_col=amount_col, id_col=id_col)

    out = amounts[~amounts[id_col].isin(dropped)] if dropped else amounts

    for service, kind, amount_col in layout:
        include_neighbors = not (kind == "line"
                                 and methodology.roads == "eq4_own_only")
        out = index.service_index(
            out, amount_col, service=service, denominator=denominator,
            nbr_dist_col=NBRS_DIST_COL, lookup_frame=amounts,
            absent_neighbor=exclusion.absent_neighbor,
            include_neighbors=include_neighbors,
            decay_form=methodology.decay.form,
            distance_unit=methodology.decay.distance_unit,
            id_col=id_col)

    return index.overall_psi(
        out, second_normalization=methodology.second_normalization)


def compute_frames(settlements, barriers, services, population, methodology,
                   denominator, *, epsg_code=7760, id_col=ID_COL,
                   type_col=TYPE_COL, population_id_col="uso_area_u",
                   population_value_col="population",
                   missing_population="drop", max_missing_population=None):
    """The documented in-memory entry point (spec § 2).

    settlements: settlement polygons with `area_km2` (and `population` when
        `population` is None).
    barriers: {layer name: GeoDataFrame} — every layer gets its own flag
        column; `methodology.barrier.combine` decides which OR into `barrier`.
    services: {service name: GeoDataFrame}; point/line is read off the
        geometries. Output columns are named after these keys — the oracle
        fixture's `clinic` maps to config `health` in the path-based stages'
        test wiring, exactly as tests/test_oracle_e2e.SERVICE_LAYOUT does.
    population: the population table, or None when settlements already carry
        the column.
    methodology: a MethodologyConfig. Exclusion overrides are applied by
        constructing a modified MethodologyConfig, never by mutating frames.
    denominator: "pop" | "popdensity" | "one".
    missing_population: "drop" | "error" — what to do about settlements with
        no population row (layers.population.missing).
    """
    if missing_population not in MISSING_POPULATION:
        raise ValueError(
            f"unknown missing_population {missing_population!r}; allowed "
            f"values: {list(MISSING_POPULATION)}")
    frame, missing = attach_population(
        settlements, population, id_col=id_col,
        population_id_col=population_id_col,
        population_value_col=population_value_col)
    if missing and missing_population == "error":
        raise validate.ValidationError(
            f"{len(missing)} settlements have no population row and "
            "layers.population.missing is 'error': "
            f"{sorted(missing)[:10]}")
    if max_missing_population is not None:
        validate.check_missing_population(
            len(missing), maximum=max_missing_population)

    neighbor_frame = build_neighbors(frame, barriers, methodology,
                                     epsg_code=epsg_code, id_col=id_col)
    dropped = excluded_ids(neighbor_frame, types=methodology.exclusion.types,
                           id_col=id_col, type_col=type_col) | set(missing)
    return index_frames(neighbor_frame, services, methodology, denominator,
                        dropped=dropped, epsg_code=epsg_code, id_col=id_col)
