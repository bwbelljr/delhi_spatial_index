"""Generate the messy-city fixture (spec: 2026-08-28-messy-city-tier-design.md).

Eleven settlements, each carrying one real-layer pathology Oraculum omits by
construction. Deterministic: running twice produces byte-identical files.
Coordinates are EPSG:7760 metre offsets from BASE, written with json.dump
(not a GDAL driver) so the files stay human-readable and diff-stable;
loaders re-apply the CRS on read. Same conventions as
generate_oraculum_fixtures.py.

`_assert_relations` re-derives EVERY relation the tier exists to pin, from
the geometries themselves, before a byte is written: move a vertex so that H
starts touching L, or so that some third envelope reaches G, and this script
fails loudly instead of quietly emitting a city that pins nothing. Then the
reference implementation scores the city and the invariants guard runs on
the result, so a fixture with a tied clinic/school anchor is never written.

    uv run python scripts/generate_messy_fixtures.py
"""

import json
from pathlib import Path

from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from scripts.check_oraculum_invariants import (
    emit_checked_expected_values, emit_checked_variant_expected_values,
)
from tests.cities import MESSY

BASE_X, BASE_Y = 1_000_000, 1_000_000
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "messy"


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# id: (USO_FINAL, population, [ring, ...])  -- more than one ring is a
# MultiPolygon. Rings are open (the writer closes them). Populations are all
# distinct, so no two settlements can tie by construction; `U` alone has
# none, which is the no-population pathology.
SETTLEMENTS = {
    # irregular hexagon: H cap L is EMPTY, yet each geometry reaches into the
    # other's envelope, so they are bbox neighbours BOTH ways and touch
    # neighbours neither way.
    "H":  ("Planned", 110, [[(400, 2400), (1600, 2300), (2200, 1600),
                             (1900, 1000), (1100, 1000), (900, 2200)]]),
    # concave L: its envelope [0,2000]x[0,2000] swallows H's lower half, and
    # its vertical arm (x<=800) reaches back under H's envelope at (400,1500).
    "L":  ("Planned", 200, [[(0, 0), (2000, 0), (2000, 800), (800, 800),
                             (800, 2000), (0, 2000)]]),
    # triangle meeting L at the single point (2000, 800): bbox neighbour,
    # never a touch neighbour.
    "T":  ("Planned", 300, [[(2000, 800), (3000, 200), (3000, 700)]]),
    # two equal parts with a 1 km gap -> centroid (6500, 500) lies OUTSIDE M.
    "M":  ("Planned", 400, [_rect(5000, 0, 6000, 1000),
                            _rect(7000, 0, 8000, 1000)]),
    # 100 m square centred EXACTLY on M's centroid, inside M's envelope and
    # disjoint from both parts: M is in G's bbox list, G is not in M's, and
    # the centroid distance is exactly 0 (weight exactly 1).
    "G":  ("Planned", 50,  [_rect(6450, 450, 6550, 550)]),
    # overlapping pair: a 200 m x 1000 m strip in common.
    "O1": ("Planned", 600, [_rect(10000, 0, 11000, 1000)]),
    "O2": ("Planned", 700, [_rect(10800, 0, 11800, 1000)]),
    # 17 km away: no neighbours under any rule.
    "I":  ("Planned", 800, [_rect(20000, 0, 21000, 1000)]),
    # the one RV settlement: what `code-2025` excludes by CATEGORY.
    "N":  ("RV",      900, [_rect(11800, 0, 12800, 1000)]),
    # no population row -> production drops it unconditionally.
    "U":  ("Planned", None, [_rect(9000, 0, 10000, 1000)]),
    # 2 m x 1 m sliver on H's southern edge: area 2e-6 km2, popdensity
    # denominator 5e7.
    "S":  ("Planned", 100, [_rect(1400, 999, 1402, 1000)]),
}

# All seven services are placed: the reference scores every service in
# POINT_SERVICES regardless, and production's PSI averages over the services
# present, so both sides must carry the same seven. `T` and `G` get a school
# so that `I` is the UNIQUE school minimum under the touch rule.
POINT_SERVICES = {
    "clinic": [("H", 1500, 1500), ("L", 400, 400), ("T", 2800, 550),
               ("M", 5500, 500), ("G", 6500, 500),
               ("O1+O2", 10900, 500),          # strictly inside the overlap
               ("I", 20500, 500)],
    "school": [("L", 600, 200), ("M", 7500, 500), ("O2", 11400, 500),
               ("N", 12300, 500), ("S", 1401.5, 999.5), ("T", 2900, 450),
               ("G", 6480, 480)],
    "bank": [("H", 1600, 1400), ("I", 20600, 600)],
    "police": [("L", 200, 600), ("O1", 10400, 500)],
    "ration": [("M", 5600, 600), ("S", 1400.5, 999.5)],
    "transport": [("H", 1700, 1300), ("N", 12500, 400)],
}

# TWO LineString rows, so "sum every road row" is load-bearing: the first row
# alone gives M nothing.
ROADS = [
    ("H+L", [(1500, 200), (1500, 2200)]),   # 0.6 km in L, 1.2 km in H
    ("M",   [(4800, 800), (8200, 800)]),    # 1.0 km in each part of M
]

ROAD_KM = {"H": 1.2, "L": 0.6, "M": 2.0}
SLIVER_AREA_KM2 = 2e-06


def _pt(x, y):
    return [BASE_X + x, BASE_Y + y]


def _ring(coords):
    """A closed GeoJSON linear ring from an open coordinate list."""
    return [[_pt(x, y) for x, y in list(coords) + [coords[0]]]]


def _shapely(rings):
    parts = [Polygon([(BASE_X + x, BASE_Y + y) for x, y in ring])
             for ring in rings]
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def _ring_area_m2(ring):
    """Shoelace area in m^2 — analytic, as Oraculum's generator does it."""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        total += x0 * y1 - x1 * y0
    return abs(total) / 2


def _area_km2(rings):
    return sum(_ring_area_m2(ring) for ring in rings) / 1_000_000


def _feature(geom, props):
    return {"type": "Feature", "properties": props, "geometry": geom}


def _dump(path, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection",
          "crs_note": "coordinates are EPSG:7760 meters; loaders apply set_crs(7760)",
          "features": features}
    path.write_text(json.dumps(fc, indent=1, sort_keys=True) + "\n")


def _bbox_nbrs(geoms):
    """Directed bbox adjacency, exactly as BOTH implementations define it:
    j is in i's list iff geom_i intersects envelope_j."""
    return {i: {j for j in geoms
                if j != i and geoms[i].intersects(geoms[j].envelope)}
            for i in geoms}


def _touch_nbrs(geoms):
    """Border sharing: the intersection must have positive length."""
    out = {}
    for i in geoms:
        out[i] = set()
        for j in geoms:
            if i == j:
                continue
            shared = geoms[i].intersection(geoms[j])
            if not shared.is_empty and shared.length > 0:
                out[i].add(j)
    return out


def _assert_relations(geoms, points, roads):
    """Every spec § 2 / § 4.3 relation, re-derived from the geometries."""
    bbox, touch = _bbox_nbrs(geoms), _touch_nbrs(geoms)

    for sid, geom in geoms.items():
        assert geom.is_valid, f"{sid} is not a valid geometry"

    # H / L: disjoint, but each reaches into the other's envelope.
    assert geoms["H"].intersection(geoms["L"]).is_empty, "H cap L must be empty"
    assert not geoms["H"].intersection(geoms["L"].envelope).is_empty, \
        "geom_H must reach into envelope_L"
    assert not geoms["L"].intersection(geoms["H"].envelope).is_empty, \
        "geom_L must reach into envelope_H (the L's arm under H's envelope)"
    assert "L" in bbox["H"] and "H" in bbox["L"], bbox
    assert "L" not in touch["H"] and "H" not in touch["L"], touch

    # T / L: a single point of contact.
    shared = geoms["T"].intersection(geoms["L"])
    assert shared.geom_type == "Point" and shared.length == 0, shared.geom_type
    assert "T" in bbox["L"] and "L" in bbox["T"], bbox
    assert "T" not in touch["L"] and "L" not in touch["T"], touch

    # M: two parts, centroid in the gap.
    assert len(geoms["M"].geoms) == 2, "M must be a two-part MultiPolygon"
    assert not geoms["M"].centroid.within(geoms["M"]), \
        "M's centroid must lie OUTSIDE M, in the gap"

    # G: centred exactly on M's centroid, the directed-bbox exhibit.
    assert geoms["G"].disjoint(geoms["M"]), "G must be disjoint from M"
    assert geoms["G"].centroid.equals(geoms["M"].centroid), \
        "G must be centred exactly on M's centroid"
    assert geoms["G"].centroid.distance(geoms["M"].centroid) == 0.0, \
        "the decay weight must be exactly 1"
    assert bbox["G"] == {"M"}, f"nbrs_bbox(G) must be exactly {{M}}: {bbox['G']}"
    assert "G" not in bbox["M"], "G must NOT be in M's bbox list"
    assert touch["G"] == set() and touch["M"] == set(), touch

    # O1 / O2: positive-area overlap, neighbours under BOTH rules.
    overlap = geoms["O1"].intersection(geoms["O2"])
    assert overlap.area > 0, "O1 cap O2 must have positive area"
    assert "O2" in bbox["O1"] and "O1" in bbox["O2"], bbox
    assert "O2" in touch["O1"] and "O1" in touch["O2"], \
        "overlapping polygons are touch neighbours (the DEL-19 finding)"
    shared_clinic = Point(*_pt(10900, 500))
    assert shared_clinic.within(geoms["O1"]) and shared_clinic.within(geoms["O2"]), \
        "the shared clinic must be STRICTLY inside the overlap"

    # I: isolated under both rules, disjoint from everything.
    assert bbox["I"] == set() and touch["I"] == set(), (bbox["I"], touch["I"])
    for sid, geom in geoms.items():
        if sid != "I":
            assert geoms["I"].disjoint(geom), f"I must be disjoint from {sid}"

    # U / N: the no-population settlement is NOT the excluded one.
    assert SETTLEMENTS["U"][1] is None, "U must have no population"
    assert SETTLEMENTS["N"][1] is not None, "N must HAVE a population"
    rv = [sid for sid, (uso, _, _) in SETTLEMENTS.items() if uso == "RV"]
    assert rv == ["N"], f"exactly one RV settlement, got {rv}"
    assert "U" in touch["O1"] and "N" in touch["O2"], touch

    # S: the area-extreme sliver, against H.
    assert _area_km2(SETTLEMENTS["S"][2]) == SLIVER_AREA_KM2, "S must be 2 m^2"
    assert geoms["S"].area > 0
    assert "H" in touch["S"] and "S" in touch["H"], touch

    # populations: all present ones distinct.
    pops = [pop for _, pop, _ in SETTLEMENTS.values() if pop is not None]
    assert len(pops) == len(set(pops)), f"tied populations: {pops}"

    # vocabulary matches the City declaration.
    assert {uso for uso, _, _ in SETTLEMENTS.values()} == set(MESSY.vocabulary)

    # every service point strictly inside EXACTLY its declared hosts, and
    # `within` (reference) agrees with `intersects` (production) on all of
    # them — the only multi-host point is the deliberate overlap clinic.
    for service, hosts, geom in points:
        expected = set(hosts)
        assert {sid for sid in geoms if geom.within(geoms[sid])} == expected, \
            (service, hosts)
        assert {sid for sid in geoms if geom.intersects(geoms[sid])} == expected, \
            (service, hosts)
    assert set(POINT_SERVICES) == {"clinic", "school", "bank", "police",
                                   "ration", "transport"}
    placed = {service: {host for host, _, _ in pts}
              for service, pts in POINT_SERVICES.items()}
    assert placed["clinic"] == {"H", "L", "T", "M", "G", "O1+O2", "I"}
    assert placed["school"] == {"L", "M", "O2", "N", "S", "T", "G"}
    assert placed["bank"] == {"H", "I"}
    assert placed["police"] == {"L", "O1"}
    assert placed["ration"] == {"M", "S"}
    assert placed["transport"] == {"H", "N"}

    # roads: two rows, and the SUM is load-bearing.
    assert len(roads) == 2, "the road layer must have two rows"
    summed = {sid: sum(road.intersection(geom).length for road in roads) / 1000
              for sid, geom in geoms.items()}
    for sid, km in ROAD_KM.items():
        assert abs(summed[sid] - km) < 1e-9, (sid, summed[sid], km)
    for sid in geoms:
        if sid not in ROAD_KM:
            assert summed[sid] == 0.0, sid
    first_only = roads[0].intersection(geoms["M"]).length / 1000
    assert first_only == 0.0, \
        "M must get its road length from the SECOND row only"


def main():
    geoms = {sid: _shapely(rings)
             for sid, (_, _, rings) in SETTLEMENTS.items()}
    points = [(service, host.split("+"), Point(*_pt(x, y)))
              for service, pts in POINT_SERVICES.items()
              for host, x, y in pts]
    roads = [LineString([_pt(*p) for p in coords]) for _, coords in ROADS]
    _assert_relations(geoms, points, roads)

    settlement_feats = []
    for sid, (uso, pop, rings) in SETTLEMENTS.items():
        area_km2 = _area_km2(rings)
        assert abs(geoms[sid].area / 1_000_000 - area_km2) <= 1e-12 * area_km2, sid
        geom = ({"type": "Polygon", "coordinates": _ring(rings[0])}
                if len(rings) == 1 else
                {"type": "MultiPolygon",
                 "coordinates": [_ring(ring) for ring in rings]})
        settlement_feats.append(_feature(
            geom, {"USO_AREA_U": sid, "USO_FINAL": uso, "population": pop,
                   "area_km2": area_km2}))
    _dump(OUT / "settlements.geojson", settlement_feats)

    service_feats = []
    for service, pts in POINT_SERVICES.items():
        for host, x, y in pts:
            service_feats.append(_feature(
                {"type": "Point", "coordinates": _pt(x, y)},
                {"service": service, "host": host}))
    for host, coords in ROADS:
        service_feats.append(_feature(
            {"type": "LineString", "coordinates": [_pt(*p) for p in coords]},
            {"service": "road", "host": host}))
    _dump(OUT / "services.geojson", service_feats)

    # No barrier coverage in this tier (spec § 2): an EMPTY collection, which
    # both implementations short-circuit on.
    _dump(OUT / "barriers.geojson", [])

    path = emit_checked_expected_values(MESSY, OUT / "expected_values.csv")
    print(f"wrote fixtures to {OUT}")
    print(f"wrote {path}")

    variants = emit_checked_variant_expected_values(
        MESSY, OUT / "variants_expected_values.csv")
    print(f"wrote {variants}")


if __name__ == "__main__":
    main()
