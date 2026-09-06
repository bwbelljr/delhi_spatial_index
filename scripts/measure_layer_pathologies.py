"""Measure the real settlement layer's pathologies (spec 3C § 5).

The messy-city tier's premises came from an ad-hoc session; this script is
their reproducible source — the same numbers, from the same layers, through
the same pipeline functions, on demand.

READ-ONLY over --data-dir. The deduplication cache goes under --cache-dir
(default: a fresh temporary directory) and NEVER under the data directory:
~/delhi_data is bisynced to the shared drive, so a stray file there
propagates to everyone.

    uv run python scripts/measure_layer_pathologies.py --config code-2025

Prints a provenance line, then the fenced block that
docs/data/layer_pathologies.md carries verbatim. The first run is slow — the
pipeline's O(n^2) dedup takes about three minutes on 4,357 rows. A WARM
--cache-dir is only safe for scripts that never inspect `geom_type`: reusing
one here upcasts every settlement to MultiPolygon on the re-read, which
corrupts the `multipolygons` count. This script must always run COLD — do not
pass a persistent --cache-dir to "reuse" it.
"""

import argparse
import sys

import geopandas as gpd

from delhi_psi import geometry, io, neighbors, pipeline
from delhi_psi.config import load_config
from scripts._measure_common import (load_settlements, parse_block, render,
                                     resolve_work_dir)


def count_rectangles(gdf, *, rtol=1e-9):
    """A polygon is a rectangle iff it fills its own bounding box."""
    return int(sum(1 for geom in gdf.geometry
                   if abs(geom.area - geom.envelope.area)
                   <= rtol * geom.envelope.area))


def count_multipolygons(gdf):
    return int((gdf.geom_type == "MultiPolygon").sum())


def count_isolated(gdf, *, id_col, rule):
    """Settlements with an EMPTY neighbour list under `rule` ("bbox" or
    "touch"). The column keeps its historical name `nbrs_bbox` under both
    rules (delhi_psi.neighbors.adjacency's own contract)."""
    frame = neighbors.adjacency(gdf, id_col=id_col, neighbor_col="nbrs_bbox",
                                rule=rule)
    return int(sum(1 for nbrs in frame["nbrs_bbox"] if len(nbrs) == 0))


def count_isolated_bbox(gdf, *, id_col):
    """Settlements with an EMPTY neighbour list under the production rule."""
    return count_isolated(gdf, id_col=id_col, rule="bbox")


def count_isolated_touch(gdf, *, id_col):
    """Settlements with an EMPTY neighbour list under the `touch` rule.

    bbox-neighbours are a superset of touch-neighbours (a touching pair's
    polygons intersect, and a polygon is contained in its own bounding box,
    so a touch-neighbour is always a bbox-neighbour too): isolated_bbox <=
    isolated_touch always holds.
    """
    return count_isolated(gdf, id_col=id_col, rule="touch")


def count_no_population(gdf, cfg):
    """The pipeline's own join, so the key and the rule are not re-invented."""
    population = io.read_population(
        cfg.paths.data_dir / cfg.layers.population.path)
    _, missing = pipeline.attach_population(
        gdf, population, id_col=cfg.layers.settlements.id_col,
        population_id_col=cfg.layers.population.id_col,
        population_value_col=cfg.layers.population.value_col)
    return len(missing)


def count_overlapping_pairs(gdf, *, id_col):
    """Pairs whose intersection has POSITIVE AREA — a shared border is not an
    overlap. The sjoin narrows the candidates so the area test never runs on
    all n^2 pairs; a self-join yields each unordered pair twice, and
    `left < right` keeps exactly one.
    """
    frame = gdf[[id_col, "geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(frame, frame, how="inner", predicate="intersects")
    geoms = frame.geometry
    return int(sum(
        1 for left, right in zip(joined.index, joined["index_right"])
        if left < right
        and geoms.iloc[left].intersection(geoms.iloc[right]).area > 0))


def corner_only_pairs(gdf, *, id_col):
    """Pairs whose intersection is NON-EMPTY but has zero length and zero
    area — they meet at one or more isolated points. `touch` (positive shared
    length) does NOT make them neighbours; a 0 km distance band does.

    Same sjoin-then-test shape as `count_overlapping_pairs`: the join narrows
    the candidates so the measure test never runs on all n^2 pairs, and
    `left < right` keeps one of each unordered pair. Shapely returns a Point
    or MultiPoint for such an intersection, but the test is on MEASURES, not
    on `geom_type`, so a GeometryCollection of points also counts and one
    containing a line does not.
    """
    frame = gdf[[id_col, "geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(frame, frame, how="inner", predicate="intersects")
    geoms = frame.geometry
    ids = frame[id_col]
    out = []
    for left, right in zip(joined.index, joined["index_right"]):
        if left >= right:
            continue
        shared = geoms.iloc[left].intersection(geoms.iloc[right])
        if not shared.is_empty and shared.length == 0 and shared.area == 0:
            out.append((ids.iloc[left], ids.iloc[right]))
    return out


def count_corner_only_pairs(gdf, *, id_col):
    """How many pairs meet at a point and nowhere else."""
    return len(corner_only_pairs(gdf, id_col=id_col))


def count_multi_settlement_points(gdf, points, *, id_col):
    """Service points that fall inside MORE THAN ONE settlement (production
    counts such a point for every one of them)."""
    frame = gdf[[id_col, "geometry"]]
    pts = points[["geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(pts, frame, how="inner", predicate="intersects")
    per_point = joined.groupby(joined.index).size()
    return int((per_point > 1).sum())


def measure(cfg, cache_dir):
    """The whole report, as an ordered {key: value} mapping."""
    id_col = cfg.layers.settlements.id_col
    gdf = load_settlements(cfg, cache_dir)
    areas = gdf["area_km2"]
    report = {
        "settlements": len(gdf),
        "rectangles": count_rectangles(gdf),
        "multipolygons": count_multipolygons(gdf),
        "isolated_bbox": count_isolated_bbox(gdf, id_col=id_col),
        "isolated_touch": count_isolated_touch(gdf, id_col=id_col),
        "no_population": count_no_population(gdf, cfg),
        "area_km2_min": f"{areas.min():.6g}",
        "area_km2_median": f"{areas.median():.6g}",
        "area_km2_max": f"{areas.max():.6g}",
        "overlapping_pairs": count_overlapping_pairs(gdf, id_col=id_col),
    }
    pairs = corner_only_pairs(gdf, id_col=id_col)
    report["corner_only_pairs"] = len(pairs)
    report["corner_only_settlements"] = len(
        {settlement for pair in pairs for settlement in pair})
    for service, path in sorted(cfg.services.point.items()):
        points = io.read_layer(cfg.paths.data_dir / path)
        # `compute` drops exact-duplicate service rows before counting; do the
        # same here so a duplicated point is not reported as a pathology.
        points = points.drop_duplicates().reset_index(drop=True)
        points = geometry.reproject(points, cfg.crs.epsg)
        report[f"multi_settlement_points_{service}"] = \
            count_multi_settlement_points(gdf, points, id_col=id_col)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code-2025",
                        help="profile that names the layers (default code-2025)")
    parser.add_argument("--data-dir", default=None,
                        help="data root, opened READ-ONLY")
    parser.add_argument("--cache-dir", default=None,
                        help="where the dedup cache goes; default a fresh "
                             "temporary directory. Never under --data-dir.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, data_dir=args.data_dir)
    cache_dir = resolve_work_dir(args.cache_dir, data_dir=cfg.paths.data_dir,
                                 prefix="delhi_psi_pathologies_")

    print(f"layer: {cfg.paths.data_dir / cfg.layers.settlements.path}")
    print(f"cache: {cache_dir}")
    print(render(measure(cfg, cache_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
