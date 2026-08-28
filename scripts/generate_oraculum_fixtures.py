"""Generate the Oraculum fixture city (spec: 2026-08-17-phase2-oracle-design.md).

Deterministic: running twice produces byte-identical files. Coordinates are
EPSG:7760 meters, offsets from BASE. GeoJSON is written with json.dump (not
a GDAL driver) so the files stay human-readable and diff-stable; loaders
re-apply the CRS on read.
"""

import json
from pathlib import Path

from scripts.check_oraculum_invariants import (
    emit_checked_expected_values, emit_checked_variant_expected_values,
)
from tests.cities import ORACULUM

BASE_X, BASE_Y = 1_000_000, 1_000_000
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "oraculum"

SETTLEMENTS = {
    #  id: (x0, y0, x1, y1, uso_final, population)
    "A":   (0, 1000, 1000, 2000, "Planned", 100),
    "B":   (1000, 1000, 2000, 2000, "UC", 200),
    "C":   (2000, 1000, 3000, 2000, "JJC", 400),
    "RV":  (1100, 2000, 1900, 3000, "RV", 100),
    "D":   (-500, 0, 500, 1000, "Planned", 100),
    "E":   (500, 0, 2500, 1000, "RUAC", 300),
    "IND": (3000 - 500, 0, 3500, 1000, "IND", 10),
}
# NOTE: IND x-range is [2500, 3500]; written as (3000-500) to make the
# 2500 boundary shared with E visually explicit.

POINT_SERVICES = {
    "clinic": [("A", 300, 1300), ("A", 700, 1700), ("B", 1500, 1600),
               ("E", 2000, 700), ("RV", 1500, 2600), ("RV", 1400, 2200)],
    "school": [("A", 400, 1200), ("D", 100, 400), ("E", 1600, 300)],
    "bank": [("A", 800, 1900)],
    "police": [("B", 1200, 1100)],
    "ration": [("D", -300, 700)],
    "transport": [("E", 900, 200)],
}
ROAD = [(750, 250), (750, 1750)]           # 0.75 km inside E, 0.75 km inside A
CANAL = [(25, 1000), (475, 1000)]          # strict interior sub-segment of A-D edge

EXHIBIT = {
    # id: (polygon coordinate ring(s), population, n_clinics)
    "P": ([[(0, 0), (2000, 0), (2000, 1000), (1000, 1000), (1000, 2000), (0, 2000), (0, 0)]], 100, 1),
    "Q": ([[(1200, 1200), (1800, 1200), (1800, 1800), (1200, 1800), (1200, 1200)]], 100, 1),
    "R": ([[(4000, 0), (5000, 0), (5000, 1000), (4000, 1000), (4000, 0)]], 100, 2),
    "S": ([[(5000, 1000), (6000, 1000), (6000, 2000), (5000, 2000), (5000, 1000)]], 50, 0),
}


def _pt(x, y):
    return [BASE_X + x, BASE_Y + y]


def _rect_ring(x0, y0, x1, y1):
    return [[_pt(x0, y0), _pt(x1, y0), _pt(x1, y1), _pt(x0, y1), _pt(x0, y0)]]


def _feature(geom, props):
    return {"type": "Feature", "properties": props, "geometry": geom}


def _dump(path, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection",
          "crs_note": "coordinates are EPSG:7760 meters; loaders apply set_crs(7760)",
          "features": features}
    path.write_text(json.dumps(fc, indent=1, sort_keys=True) + "\n")


def main():
    settlement_feats = []
    for sid, (x0, y0, x1, y1, uso, pop) in SETTLEMENTS.items():
        area_km2 = abs(x1 - x0) * abs(y1 - y0) / 1_000_000
        settlement_feats.append(_feature(
            {"type": "Polygon", "coordinates": _rect_ring(x0, y0, x1, y1)},
            {"USO_AREA_U": sid, "USO_FINAL": uso, "population": pop,
             "area_km2": area_km2}))
    _dump(OUT / "settlements.geojson", settlement_feats)

    service_feats = []
    for service, pts in POINT_SERVICES.items():
        for host, x, y in pts:
            service_feats.append(_feature(
                {"type": "Point", "coordinates": _pt(x, y)},
                {"service": service, "host": host}))
    service_feats.append(_feature(
        {"type": "LineString", "coordinates": [_pt(*p) for p in ROAD]},
        {"service": "road", "host": "A+E"}))
    _dump(OUT / "services.geojson", service_feats)

    _dump(OUT / "barriers.geojson", [_feature(
        {"type": "LineString", "coordinates": [_pt(*p) for p in CANAL]},
        {"name": "canal"})])

    exhibit_feats = []
    for eid, (rings, pop, clinics) in EXHIBIT.items():
        exhibit_feats.append(_feature(
            {"type": "Polygon",
             "coordinates": [[_pt(x, y) for (x, y) in ring] for ring in rings]},
            {"id": eid, "population": pop, "clinics": clinics}))
    _dump(OUT / "divergence" / "exhibit.geojson", exhibit_feats)
    print(f"wrote fixtures to {OUT}")

    # The CSV is part of the fixture, so the generator owns it too: the CI
    # drift step globs generate_*_fixtures.py, which is what makes a
    # reference change without a regenerated CSV fail the build. The output
    # is byte-identical to what tests/reference_impl.py's __main__ wrote.
    print(f"wrote {emit_checked_expected_values(ORACULUM, OUT / 'expected_values.csv')}")

    # The variants fixture (spec 3D § 3): same city, ONE scenario, eight
    # derived rule-sets. Guarded by the band check AND the CSV-wide
    # invariants guard before anything is written.
    print(f"wrote {emit_checked_variant_expected_values(ORACULUM, OUT / 'variants_expected_values.csv')}")


if __name__ == "__main__":
    main()
