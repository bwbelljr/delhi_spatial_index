"""Pipeline stages. `compute_frames` is the in-memory seam the oracle uses;
`preprocess` / `compute` (added next) are the path-based stages.

This is the ONLY module that sees a Config. The math modules take explicit
keyword arguments.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from delhi_psi import categories, geometry, index, io, neighbors, validate

log = logging.getLogger(__name__)

ID_COL = "USO_AREA_U"
TYPE_COL = "USO_FINAL"
NBRS_COL = "nbrs_bbox"
NBRS_DIST_COL = "nbrs_dist_bbox"
CENTROID_COL = "centroid"
CATEGORY_COL = categories.CATEGORY_COLUMN

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


def excluded_ids(frame, *, types, id_col=ID_COL, category_col=CATEGORY_COL):
    """Ids whose CATEGORY is in `types`.

    From cycle 3B `methodology.exclusion.types` holds CATEGORY names — the
    values of `categories.mapping` — and matches the mapped column the
    prelude has just added, not the raw source-type column.
    """
    if not types:
        return frozenset()
    return frozenset(frame.loc[frame[category_col].isin(list(types)), id_col])


def build_neighbors(settlements, barriers, methodology, *, id_col=ID_COL):
    """Barrier flags, combined flag, centroids, neighbour lists, distances.

    Always built on the FULL settlement universe — preprocess never excludes
    (spec § 3). Reprojects nothing: the caller hands it frames already in the
    target CRS, which is why there is no epsg_code parameter.
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
    """Return the universe: the frame neighbour AMOUNTS are read from.

    Which rows are REPORTED is not decided here — `index_frames` drops
    `dropped` after amounts, so the exclusion rule has exactly one home.

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
    return universe


def index_frames(neighbor_frame, services, methodology, denominator, *,
                 dropped=frozenset(), epsg_code=7760, id_col=ID_COL):
    """Amounts, PCEN, min-max and the overall PSI for one denominator."""
    exclusion = methodology.exclusion
    universe = apply_exclusion(neighbor_frame, dropped=dropped,
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


def _population_and_exclusion(frame, population, *, id_col, type_col,
                              population_id_col, population_value_col,
                              missing_population, max_missing_population,
                              exclusion_types, mapping,
                              category_col=CATEGORY_COL):
    """The prelude both entry points share: attach population, apply the
    missing rule, map source types to categories, and work out which ids are
    dropped.

    Returns (frame with population and `category`, dropped ids, ids with no
    population). `dropped` is the CATEGORY exclusion UNION the unpriced
    rows — the same set in `compute_frames` and `compute`, so the rule (and
    its message) lives once. The mapping is applied here, immediately before
    `excluded_ids`, which is why both entry points get the `category` column
    and identical exclusion semantics from one place.
    """
    out, missing = attach_population(
        frame, population, id_col=id_col,
        population_id_col=population_id_col,
        population_value_col=population_value_col)
    if missing and missing_population == "error":
        raise validate.ValidationError(
            f"{len(missing)} settlements have no population row and "
            f"layers.population.missing is 'error': {sorted(missing)[:10]}")
    if max_missing_population is not None:
        validate.check_missing_population(
            len(missing), maximum=max_missing_population)
    out = categories.apply_mapping(out, type_col=type_col, mapping=mapping,
                                   out_col=category_col)
    # The same subset rule load_config enforces, repeated at run time:
    # in-memory callers build a MethodologyConfig directly and never pass
    # through the loader, and a category the mapping does not produce would
    # exclude nothing at all, silently.
    allowed = categories.categories_of(mapping)
    unknown = sorted(item for item in exclusion_types if item not in allowed)
    if unknown:
        raise validate.ValidationError(
            f"methodology.exclusion.types {unknown} are not categories "
            "produced by categories.mapping; it produces: "
            f"{sorted(allowed)}")
    dropped = excluded_ids(out, types=exclusion_types, id_col=id_col,
                           category_col=category_col) | set(missing)
    return out, dropped, missing


def compute_frames(settlements, barriers, services, population, methodology,
                   denominator, *, epsg_code=7760, id_col=ID_COL,
                   type_col=TYPE_COL, population_id_col="uso_area_u",
                   population_value_col="population",
                   missing_population="drop", max_missing_population=None,
                   mapping=None, scheme="identity"):
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
    mapping: {source type: category}, or None to build the identity over the
        types this city carries. `scheme` names the mapping in the result's
        `attrs` (and, through `compute`, in the joblib output).
    """
    if missing_population not in MISSING_POPULATION:
        raise ValueError(
            f"unknown missing_population {missing_population!r}; allowed "
            f"values: {list(MISSING_POPULATION)}")
    if mapping is None:
        # The identity over the types this city actually carries: existing
        # in-memory callers keep their call shape and their numbers.
        mapping = {t: t for t in settlements[type_col].unique()}
    frame, dropped, _ = _population_and_exclusion(
        settlements, population, id_col=id_col, type_col=type_col,
        population_id_col=population_id_col,
        population_value_col=population_value_col,
        missing_population=missing_population,
        max_missing_population=max_missing_population,
        exclusion_types=methodology.exclusion.types,
        mapping=mapping)

    # `dropped` is read off `frame`, not the neighbours frame: build_neighbors
    # adds columns and never adds or removes a row, so the two give the same
    # ids.
    neighbor_frame = build_neighbors(frame, barriers, methodology,
                                     id_col=id_col)
    result = index_frames(neighbor_frame, services, methodology, denominator,
                          dropped=dropped, epsg_code=epsg_code, id_col=id_col)
    # After the last index_frames call, never before: pandas drops `attrs`
    # across the merges inside it (a caller that merges this result further
    # loses the stamp too, as pandas documents).
    result.attrs["categories"] = {"scheme": scheme, "mapping": dict(mapping)}
    return result


@dataclass(frozen=True)
class PreprocessResult:
    neighbors_path: Path
    n_settlements: int
    n_barrier_flagged: int
    reports: tuple


@dataclass(frozen=True)
class ComputeResult:
    outputs: tuple
    missing_population_path: Path
    n_missing_population: int
    n_reported: int


def output_basename(cfg, denominator):
    return cfg.outputs.name_template.format(profile=cfg.profile,
                                            denominator=str(denominator))


def methodology_stamp(methodology):
    """The methodology keys that SHAPE the stored neighbour lists.

    Only these: adjacency decides who is a neighbour, the barrier rule and
    `combine` decide who is severed. Everything else (decay, roads,
    exclusion, normalization) is applied downstream in `compute`, so an
    artifact stays valid across those.
    """
    combine = methodology.barrier.combine
    return {
        "adjacency": {"rule": str(methodology.adjacency.rule)},
        "barrier": {
            "rule": str(methodology.barrier.rule),
            "combine": combine if isinstance(combine, str)
            else [str(layer) for layer in combine],
        },
    }


def check_methodology_stamp(frame, cfg):
    """Refuse a neighbours artifact built by a different methodology.

    Without this, `preprocess --config A` followed by `compute --config B`
    silently reports B's numbers over A's neighbour lists.
    """
    stored = frame.attrs.get("methodology")
    if not stored:
        raise validate.ValidationError(
            "neighbours artifact has no methodology stamp — re-run preprocess")
    for block, keys in methodology_stamp(cfg.methodology).items():
        for key, configured in keys.items():
            found = stored.get(block, {}).get(key)
            if found != configured:
                raise validate.ValidationError(
                    f"neighbours artifact was built with "
                    f"methodology.{block}.{key}={found!r}, but this config "
                    f"says {configured!r} — re-run preprocess "
                    f"(artifact profile: {frame.attrs.get('profile')!r}, "
                    f"config profile: {cfg.profile!r})")


def _dedup_cached(gdf, cache_dir, name, source_path):
    """Deduplicate once, caching under out_dir keyed on source mtime+size.

    The O(n^2) algorithm is unchanged (spec § 6); only the cache location and
    the staleness rule move here from scripts/preprocess.py, which keyed the
    cache on existence alone.
    """
    stat = Path(source_path).stat()
    stamp = cache_dir / f"{name}.dedup.stamp"
    # Explicit GeoPackage: a suffix-less path makes GDAL fall back to a
    # *directory* shapefile (plan review R2). GPKG is a single file and
    # lossless for geometry and attributes.
    cached = cache_dir / f"{name}.dedup.gpkg"
    signature = f"{stat.st_mtime_ns}:{stat.st_size}\n"
    if cached.exists() and stamp.exists() and stamp.read_text() == signature:
        log.info("reusing dedup cache %s", cached)
        return io.read_layer(cached)
    deduped = geometry.remove_duplicate_geom(gdf)
    cache_dir.mkdir(parents=True, exist_ok=True)
    deduped.to_file(cached, driver="GPKG", index=False)
    stamp.write_text(signature)
    return deduped


def preprocess(cfg):
    """Settlements + barriers -> the universe-wide neighbours artifact."""
    data_dir = cfg.paths.data_dir
    out_dir = io.resolve_out_dir(cfg.paths.out_dir, data_dir)

    bounds = io.read_layer(data_dir / cfg.layers.bounds)
    settlements = io.read_layer(data_dir / cfg.layers.settlements.path)
    barriers = {name: io.read_layer(data_dir / path)
                for name, path in cfg.layers.barriers.items()}

    reports = [validate.require_layer(settlements, name="settlements",
                                      geom_type="Polygon", bounds_gdf=bounds)]
    for name, gdf in barriers.items():
        reports.append(validate.require_layer(gdf, name=name,
                                              geom_type="Line",
                                              bounds_gdf=bounds))

    settlements = _dedup_cached(settlements, out_dir, "settlements",
                                data_dir / cfg.layers.settlements.path)
    barriers = {
        name: _dedup_cached(gdf, out_dir, name,
                            data_dir / cfg.layers.barriers[name])
        for name, gdf in barriers.items()}

    epsg = cfg.crs.epsg
    settlements = geometry.reproject(settlements, epsg)
    barriers = {name: geometry.reproject(gdf, epsg)
                for name, gdf in barriers.items()}
    validate.check_crs_match({"settlements": settlements, **barriers})

    settlements["area_km2"] = settlements.area / 1000000
    drop = {"index", "level_0"}.intersection(settlements.columns)
    settlements = settlements.drop(columns=drop)

    frame = build_neighbors(settlements, barriers, cfg.methodology,
                            id_col=cfg.layers.settlements.id_col)

    if cfg.layers.ndmc_center:
        centre = io.read_layer(data_dir / cfg.layers.ndmc_center)
        reports.append(validate.require_layer(centre, name="ndmc_center",
                                              geom_type="Point",
                                              bounds_gdf=bounds))
        centre = geometry.reproject(centre, epsg)
        frame = geometry.distance_to_point_km(
            frame, centre["geometry"].values[0], centroid_col=CENTROID_COL,
            out_col="ndmc_dist_km")

    # Bind the artifact to the methodology that built it; pandas `attrs` are
    # pickled with the frame, so the stamp survives the joblib round trip.
    frame.attrs["profile"] = cfg.profile
    frame.attrs["methodology"] = methodology_stamp(cfg.methodology)

    path = io.write_neighbors(frame, out_dir / cfg.paths.neighbors_artifact)
    return PreprocessResult(
        neighbors_path=path,
        n_settlements=len(frame),
        n_barrier_flagged=int(frame["barrier"].sum()),
        reports=tuple(reports))


def compute(cfg):
    """Neighbours artifact + population + services -> one PSI set per
    outputs.denominators entry."""
    data_dir = cfg.paths.data_dir
    out_dir = io.resolve_out_dir(cfg.paths.out_dir, data_dir)
    id_col = cfg.layers.settlements.id_col

    neighbor_frame = io.read_neighbors(out_dir / cfg.paths.neighbors_artifact)
    # Before any math: the stored neighbour lists must come from THIS
    # methodology, or every number below describes a method nobody ran.
    check_methodology_stamp(neighbor_frame, cfg)
    bounds = io.read_layer(data_dir / cfg.layers.bounds)
    population = io.read_population(data_dir / cfg.layers.population.path)

    services = {}
    for name, path in {**cfg.services.point, **cfg.services.line}.items():
        gdf = io.read_layer(data_dir / path)
        # compute_psi.py (pre-refactor) dropped exact-duplicate rows from the
        # bank layer before its checks; every service layer gets the same
        # treatment here, before the battery below, so a layer with
        # duplicate rows (e.g. the real bank layer) does not fail
        # has_duplicate_rows.
        n_before = len(gdf)
        gdf = gdf.drop_duplicates().reset_index(drop=True)
        n_dropped = n_before - len(gdf)
        if n_dropped:
            log.info("service %s: dropped %d duplicate rows", name, n_dropped)
        # Spec § 6: the compute stage's CRS check, run per-service BEFORE the
        # layer battery. require_layer's within_bounds reprojects to the
        # bounds CRS, which raises a raw (uncaught) ValueError on a CRS-less
        # frame instead of the intended ValidationError — checked here first
        # so a service with no CRS fails cleanly instead of crashing.
        validate.check_crs_defined({name: gdf})
        geom_type = "Point" if name in cfg.services.point else "Line"
        validate.require_layer(gdf, name=name, geom_type=geom_type,
                               bounds_gdf=bounds)
        services[name] = gdf
    # Services are reprojected per-service inside index_frames, so this
    # additionally asserts the neighbours artifact is in the target CRS.
    validate.check_crs_defined({"neighbors": neighbor_frame})
    if neighbor_frame.crs.to_epsg() != cfg.crs.epsg:
        raise validate.ValidationError(
            f"neighbors artifact is in {neighbor_frame.crs}, "
            f"config crs.epsg is {cfg.crs.epsg}")

    frame, dropped, missing = _population_and_exclusion(
        neighbor_frame, population, id_col=id_col,
        type_col=cfg.layers.settlements.type_col,
        population_id_col=cfg.layers.population.id_col,
        population_value_col=cfg.layers.population.value_col,
        missing_population=cfg.layers.population.missing,
        max_missing_population=cfg.validate.max_missing_population,
        exclusion_types=cfg.methodology.exclusion.types,
        mapping=cfg.categories.mapping)

    # One INFO line per excluded category, even one that matched zero rows
    # — the silent case this closes. `missing` is logged separately so a
    # row dropped for both reasons is not double-counted into a category.
    for category in cfg.methodology.exclusion.types:
        rows = frame.loc[frame[CATEGORY_COL] == category, id_col]
        log.info("excluded: category=%s rows=%d",
                 category, len(set(rows) - set(missing)))
    if missing:
        log.info("excluded: missing_population rows=%d", len(missing))

    missing_path = out_dir / "missing_population.csv"
    frame[frame[id_col].isin(missing)].drop(
        columns=[c for c in io.SHAPEFILE_DROP_COLUMNS if c in frame.columns]
    ).to_csv(missing_path, index=False)

    # CSV and shapefile cannot carry `attrs`, so for those formats the
    # record of which scheme produced these rows is this line plus the
    # `category` column itself. The scheme is never a column.
    stamp = {"scheme": cfg.categories.scheme,
             "mapping": dict(cfg.categories.mapping)}
    log.info("categories: scheme=%s n_categories=%d", cfg.categories.scheme,
             len(categories.categories_of(cfg.categories.mapping)))

    outputs = []
    n_reported = 0
    for denominator in cfg.outputs.denominators:
        result = index_frames(frame, services, cfg.methodology,
                              str(denominator), dropped=dropped,
                              epsg_code=cfg.crs.epsg, id_col=id_col)
        validate.check_no_negative(result)
        n_reported = len(result)
        # Immediately before the write, mirroring the neighbours stamp:
        # index_frames' merges drop `attrs`, so stamping earlier vanishes.
        result.attrs["categories"] = stamp
        outputs.extend(io.write_outputs(
            result, out_dir, basename=output_basename(cfg, denominator),
            formats=cfg.outputs.formats))

    return ComputeResult(outputs=tuple(outputs),
                         missing_population_path=missing_path,
                         n_missing_population=len(missing),
                         n_reported=n_reported)
