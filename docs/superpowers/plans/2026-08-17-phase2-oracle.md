# Phase 2: Mythical-City Oracle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Oraculum fixture city, an independent reference implementation of the manuscript's equations, and test suites establishing three-way agreement (production code == reference == hand arithmetic), plus maps, derivation worksheet, and the exclusion-semantics memo for Raj.

**Architecture:** A generator script writes deterministic GeoJSON fixtures; `tests/reference_impl.py` implements Eq. 1–4 from the manuscript with seven parameterization knobs (never touching `spatial_index_utils`); `expected_values.csv` is emitted by the reference impl and anchored by hand-derived test assertions; production code is tested against the `rule=code` rows via a chain that mirrors `scripts/preprocess.py`/`compute_psi.py` wiring; an e2e test runs the real CLIs on a temp data dir. Build order is normative (spec): fixtures+empirical pin FIRST — if the pin contradicts the spec's directed neighbor tables, STOP (owner red line).

**Tech Stack:** Python 3.13/uv, geopandas 1.1, shapely 2.1, numpy, pandas 2.3, pytest, matplotlib.

**Spec:** `docs/superpowers/specs/2026-08-17-phase2-oracle-design.md` (rev 4 — read it in full first; its tables are the source of truth for every number here)

## Global Constraints

- **No changes to `spatial_index_utils.py` or `scripts/preprocess.py`/`compute_psi.py`/`verify_against_baseline.py`.** Ideal-vs-code gaps are findings, never fixes.
- **`~/delhi_data` is read-only.** All test IO under pytest `tmp_path` or the repo.
- **Reference impl independence**: `tests/reference_impl.py` must not import, call, or textually mirror `spatial_index_utils.py`.
- Coordinates: EPSG:7760, offsets from BASE = (1_000_000, 1_000_000) meters. Distances in km (divide meters by 1000) for the 1/(1+D) decay.
- Settlement ids are strings: "A","B","C","RV","D","E","IND". Column names for production compatibility: `USO_AREA_U` (id), `USO_FINAL` (type code: A/D → "Planned", B → "UC", C → "JJC", RV → "RV", E → "RUAC", IND → "IND"), `population`, `area_km2`.
- Services (7, matching the real pipeline): point `clinic`(→Health), `school`(→School), `bank`(→Banking), `police`(→Police), `ration`(→Ration), `transport`(→Transport); line `road`(→Major Road).
- Numeric tolerance in all assertions: `pytest.approx(..., abs=1e-12)` unless stated.
- Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

## File Structure

```
scripts/generate_oraculum_fixtures.py   # Task 1 — writes all GeoJSON fixtures
tests/fixtures/oraculum/                # Task 1 outputs (committed)
  settlements.geojson  services.geojson  barriers.geojson
  divergence/exhibit.geojson
  expected_values.csv                   # Task 3 output (committed)
tests/oraculum_fixtures.py              # Task 1 — loaders + production-chain helper
tests/test_fixture_invariants.py        # Task 1 — geometry invariants + empirical pin (THE GATE)
tests/reference_impl.py                 # Tasks 2–3 — independent Eq. 1–4 implementation
tests/test_reference_impl.py            # Tasks 2–3 — hand-anchor assertions
tests/test_oracle.py                    # Task 4 — production vs expected_values.csv
tests/test_oracle_e2e.py                # Task 5 — real CLIs on temp data dir
tests/test_divergence_exhibit.py        # Task 6
scripts/check_oraculum_invariants.py    # Task 3 — spec §7 CSV-wide guard
scripts/render_oracle_maps.py           # Task 7
docs/oracle/oraculum_city.png  oraculum_exclusion_variants.png  oraculum_divergence.png
docs/oracle/derivation-worksheet.md     # Task 8
docs/oracle/exclusion-semantics-memo.md # Task 8
CHANGELOG.md                            # Task 8 (modify)
```

## Canonical numbers (derived from the spec; used as anchors throughout)

Decay weights: d=1.0 km → 1/2; d=1.5 → 1/2.5 = 0.4; d=√2 → 1/(1+√2) = √2−1 ≈ 0.41421356237309515; d=√5/2 → 1/(1+√5/2) ≈ 0.47213595499957939.

Ideal-rule directed neighbor lists (== symmetric): A:[B,E] B:[A,C,RV,E] C:[B,E,IND] RV:[B] D:[E] E:[A,B,C,D,IND] IND:[C,E].
Code-rule (flagged {A,D} removed from every list): A:[B,E] B:[C,RV,E] C:[B,E,IND] RV:[B] D:[E] E:[B,C,IND] IND:[C,E].

Clinic PCEN, ideal/baseline/pop (counts: A2 B1 E1 RV2):
A=(2+1·½+1·(√2−1))/100  B=(1+2·½+2·½+1·½)/200=3.5/200=0.0175  C=(1·½+1·(√2−1))/400  RV=(2+1·½)/100=0.025  D=(1·0.4)/100=0.004  E=(1+2·(√2−1)+1·½)/300  IND=(1·0.4)/10=0.04.
Code-rule differences only: B=2.5/200=0.0125, E=1.5/300=0.005.

School PCEN, ideal/baseline/pop (schools: A,D,E): A=(1+(√2−1))/100=√2/100≈0.014142136 B=(1·½+1·½)/200=0.005 C=(√2−1)/400≈0.001035534 RV=0 D=(1+0.4)/100=0.014 E=(1+(√2−1)+0.4)/300≈0.006047379 IND=0.4/10=0.04.
Code-rule: B=0.5/200=0.0025, E=1/300≈0.003333333; others unchanged.

Road, ideal (Eq. 4 literal, no neighbor term), lengths A 0.75 km E 0.75 km: pop: A=0.0075 E=0.0025 others 0; popdensity: A=0.0075 E=0.005 others 0.
Road, code (decayed): pop: A=(0.75+0.75(√2−1))/100≈0.010606602 B=0.375/200=0.001875 C=0.75(√2−1)/400≈0.000776650 RV=0 D=0.3/100=0.003 E=0.75/300=0.0025 IND=0.3/10=0.03.

Singleton services (1 point each): bank@A police@B ration@D transport@E. Pattern: PCEN_i = (own + [X∈nbrs(i)]·decay(i,X))/denom_i. Police ideal/pop: B own 1/200=0.005; A gets ½/100=0.005; RV gets ½/100=0.005 — a THREE-way tied argmax A/B/RV (recorded ground truth; ties outside clinics/schools are expected, per the spec's positive-list invariant. Note: the spec's §7 example text says "A and B" — this plan's three-way count is the computed truth and the worksheet records it).

Exclusion anchors (clinic, pop): ideal excl_contributing B=0.0175; ideal excl_removed B=(1+2·½+1·½)/200=0.0125; code excl_contributing == code excl_removed cell-for-cell (B=1.5/200=0.0075).

Exhibit: P(L-shape,pop100,1 clinic) Q(pop100,1 clinic) R(pop100,2 clinics) S(pop50,0). Border rule: all four have zero neighbors → PCENs P=Q=0.01 R=0.02 S=0. bbox (directed): Q→[P] AND R↔S (R and S are rectangles, so bbox≡geometry and the corner touch registers — production-verified): Q=(1+1·1/(1+0.942809))/100≈0.015147186 (delta +0.005147186; P delta 0); S=(2·(√2−1))/50≈0.016568542494923802 (delta; R unchanged, S serviceless). intersects: ONLY R↔S (Q's polygon never touches P's): S delta identical; Q delta 0. Three rules, three distinct neighbor sets.

---

### Task 1: Fixtures, loaders, invariants, and the empirical pin (THE GATE)

**Files:**
- Create: `scripts/generate_oraculum_fixtures.py`, `tests/oraculum_fixtures.py`, `tests/test_fixture_invariants.py`, `tests/fixtures/oraculum/*.geojson`

**Interfaces:**
- Produces: fixture GeoJSONs; loader functions `load_settlements() -> GeoDataFrame` (columns USO_AREA_U, USO_FINAL, population, area_km2, geometry; CRS 7760), `load_services() -> dict[str, GeoDataFrame]` (keys clinic/school/bank/police/ration/transport/road), `load_barriers() -> GeoDataFrame`, `load_exhibit() -> GeoDataFrame` (columns id, population, clinics, geometry); `run_production_chain(settlements, barriers, services, pcen_denom, drop_ids_post=frozenset()) -> GeoDataFrame` mirroring preprocess+compute_psi wiring and returning the calc_all_services output.
- This task is the spec's build-order step 1: if `test_empirical_pin` fails, STOP THE RUN (owner red line) — do not adjust expectations to make it pass.

- [ ] **Step 1: Write the generator**

Create `scripts/generate_oraculum_fixtures.py`:

```python
"""Generate the Oraculum fixture city (spec: 2026-08-17-phase2-oracle-design.md).

Deterministic: running twice produces byte-identical files. Coordinates are
EPSG:7760 meters, offsets from BASE. GeoJSON is written with json.dump (not
a GDAL driver) so the files stay human-readable and diff-stable; loaders
re-apply the CRS on read.
"""

import json
from pathlib import Path

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


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

Run: `uv run python scripts/generate_oraculum_fixtures.py && ls tests/fixtures/oraculum tests/fixtures/oraculum/divergence`
Expected: `settlements.geojson services.geojson barriers.geojson` and `exhibit.geojson` listed.

- [ ] **Step 3: Write the loaders + production-chain helper**

Create `tests/oraculum_fixtures.py`:

```python
"""Loaders for the Oraculum fixtures + the production-chain helper.

run_production_chain mirrors, call for call, how scripts/preprocess.py and
scripts/compute_psi.py drive spatial_index_utils — so the library-first
oracle tests exercise exactly the production wiring on fixture data.
"""

from pathlib import Path

import geopandas as gpd

import spatial_index_utils

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

    drop_ids_post: ids removed AFTER neighbor computation (the scripts'
    post-drop semantics — e.g. {'RV'} replicates compute_psi's RV filter).
    """
    colonies = settlements.copy()
    colonies = spatial_index_utils.barrier_intersection(colonies, barriers, "canal")
    colonies["barrier"] = colonies["canal"]
    colonies["centroid"] = colonies.centroid

    colonies_bbox = spatial_index_utils.create_bbox_gdf(colonies)
    colonies_bbox = gpd.GeoDataFrame(colonies_bbox, crs=colonies.crs)

    nbrs = spatial_index_utils.add_polygon_neighbors_column_fast(
        polygon_gdf=colonies, right_gdf=colonies_bbox,
        id_colname="USO_AREA_U", neighbor_colname="nbrs_bbox",
        barrier_colname="barrier")
    nbrs = spatial_index_utils.calc_nbr_dist(
        polygon_gdf=nbrs, nbr_dist_colname="nbrs_dist_bbox",
        centroid_colname="centroid", neighbor_colname="nbrs_bbox",
        neighbor_id_col="USO_AREA_U")
    nbrs["index"] = nbrs.index

    if drop_ids_post:
        nbrs = nbrs[~nbrs["USO_AREA_U"].isin(drop_ids_post)]

    point_services = {k: v for k, v in services.items() if k != "road"}
    line_services = {"road": services["road"]}
    return spatial_index_utils.calc_all_services(
        polygon_gdf=nbrs, point_services=point_services,
        line_services=line_services, epsg_code=EPSG,
        pcen_denom=pcen_denom, nbr_dist_colname="nbrs_dist_bbox")
```

- [ ] **Step 4: Write the invariants + empirical-pin tests (failing first is fine — they run against real fixtures)**

Create `tests/test_fixture_invariants.py`:

```python
"""Geometry invariants + the empirical pin (spec build-order step 1).

If test_empirical_pin_* fails: STOP — the spec's directed neighbor table is
wrong and must be corrected by the owner before anything downstream is
built (spec 'Risks': hard red line).
"""

import itertools
import math

import pytest
from shapely.geometry import box

from tests.oraculum_fixtures import (
    load_settlements, load_barriers, load_services, run_production_chain,
)

BASE = 1_000_000

GEOMETRIC_PAIRS_KM = {
    frozenset(p): d for p, d in {
        ("A", "B"): 1.0, ("A", "D"): math.sqrt(5) / 2, ("A", "E"): math.sqrt(2),
        ("B", "C"): 1.0, ("B", "RV"): 1.0, ("B", "E"): 1.0,
        ("C", "E"): math.sqrt(2), ("C", "IND"): math.sqrt(5) / 2,
        ("D", "E"): 1.5, ("E", "IND"): 1.5,
    }.items()
}

IDEAL_DIRECTED = {"A": {"B", "E"}, "B": {"A", "C", "RV", "E"},
                  "C": {"B", "E", "IND"}, "RV": {"B"}, "D": {"E"},
                  "E": {"A", "B", "C", "D", "IND"}, "IND": {"C", "E"}}
CODE_DIRECTED = {"A": {"B", "E"}, "B": {"C", "RV", "E"},
                 "C": {"B", "E", "IND"}, "RV": {"B"}, "D": {"E"},
                 "E": {"B", "C", "IND"}, "IND": {"C", "E"}}


@pytest.fixture(scope="module")
def city():
    return load_settlements().set_index("USO_AREA_U")


def test_all_rectangles_bbox_equals_geometry(city):
    for sid, row in city.iterrows():
        assert row.geometry.equals(box(*row.geometry.bounds)), sid


def test_touching_pairs_share_edges_never_points(city):
    seen = set()
    for i, j in itertools.combinations(city.index, 2):
        gi, gj = city.loc[i].geometry, city.loc[j].geometry
        inter = gi.intersection(gj)
        if not inter.is_empty:
            assert inter.length > 0, f"{i}-{j} touch only at a point"
            seen.add(frozenset((i, j)))
    assert seen == set(GEOMETRIC_PAIRS_KM), "pair set differs from spec table"


def test_pair_distances_match_spec(city):
    for pair, d_km in GEOMETRIC_PAIRS_KM.items():
        i, j = tuple(pair)
        d = city.loc[i].geometry.centroid.distance(city.loc[j].geometry.centroid)
        assert d / 1000 == pytest.approx(d_km, abs=1e-9), pair


def test_canal_inside_ad_edge_touches_exactly_a_and_d(city):
    canal = load_barriers().geometry.iloc[0]
    shared = city.loc["A"].geometry.intersection(city.loc["D"].geometry)
    assert canal.within(shared.buffer(1e-9))
    touching = {sid for sid, row in city.iterrows()
                if row.geometry.intersects(canal)}
    assert touching == {"A", "D"}
    for sid in touching:
        assert city.loc[sid].geometry.intersection(canal).length > 0, \
            f"canal touches {sid} only at a point"


def test_road_lengths_and_canal_clearance(city):
    road = load_services()["road"].geometry.iloc[0]
    canal = load_barriers().geometry.iloc[0]
    assert not road.intersects(canal)
    for sid, expected_km in [("A", 0.75), ("E", 0.75)]:
        got = road.intersection(city.loc[sid].geometry).length / 1000
        assert got == pytest.approx(expected_km, abs=1e-12), sid
    for sid in ("B", "C", "RV", "D", "IND"):
        assert road.intersection(city.loc[sid].geometry).length == 0, sid


def test_service_points_inside_their_hosts(city):
    services = load_services()
    for name, gdf in services.items():
        if name == "road":
            continue
        for _, row in gdf.iterrows():
            assert row.geometry.within(city.loc[row["host"]].geometry), \
                f"{name} point not inside {row['host']}"


def test_empirical_pin_code_rule_neighbors():
    """THE GATE: production code must reproduce the spec's directed table."""
    result = run_production_chain(
        load_settlements(), load_barriers(), load_services(), "pop")
    got = {row["USO_AREA_U"]: set(row["nbrs_bbox"])
           for _, row in result.iterrows()}
    assert got == CODE_DIRECTED


def test_empirical_pin_distances_are_km_tuples():
    result = run_production_chain(
        load_settlements(), load_barriers(), load_services(), "pop")
    row = result[result["USO_AREA_U"] == "B"].iloc[0]
    dist = dict(row["nbrs_dist_bbox"])
    assert dist["E"] == pytest.approx(1.0, abs=1e-9)
    assert dist["RV"] == pytest.approx(1.0, abs=1e-9)
```

- [ ] **Step 5: Run the suite**

Run: `uv run pytest tests/test_fixture_invariants.py -v`
Expected: ALL PASS. If `test_empirical_pin_code_rule_neighbors` fails, STOP the pipeline and report to the owner (red line — do not "fix" the expectation).

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_oraculum_fixtures.py tests/oraculum_fixtures.py tests/test_fixture_invariants.py tests/fixtures/oraculum
git commit -m "feat(oracle): Oraculum fixtures, loaders, invariants + empirical pin"
```

---

### Task 2: Reference implementation core (adjacency, barriers, point-service PCEN)

**Files:**
- Create: `tests/reference_impl.py`
- Test: `tests/test_reference_impl.py`

**Interfaces:**
- Consumes: loaders from `tests/oraculum_fixtures.py` (fixture data only — NEVER `spatial_index_utils`).
- Produces (used by Tasks 3–6):
  - `adjacency(settlements, rule) -> dict[str, set[str]]` — rule ∈ {"border","bbox","intersects"}; "border" symmetric (shared boundary with length > 0), "bbox"/"intersects" DIRECTED: nbrs(i) = {j : geom_i intersects bbox_j} / {j : geom_i intersects geom_j}.
  - `apply_barrier(nbrs, settlements, barriers, rule) -> dict[str, set[str]]` — rule ∈ {"pair","global"}; "pair": remove i↔j iff a barrier intersects their shared boundary; "global": remove every barrier-flagged id from every list.
  - `compute_city(settlements, services, barriers, *, adjacency_rule, barrier_rule, roads_formula, scenario, denom, second_norm, absent_neighbor_contribution) -> DataFrame` — index=settlement id; columns `<svc>_count`, `<svc>_pcen`, `<svc>_idx` for each service (road count column is `road_length_km`), plus `psi_eq1` and (if second_norm) `norm_psi`. scenario ∈ {"baseline","excl_contributing","excl_removed","excl_ind_removed","excl_rv_only"}; roads_formula ∈ {"eq4","decayed"}; absent_neighbor_contribution ∈ {"contributes","swallowed"}.
  - `RULESETS = {"ideal": dict(adjacency_rule="border", barrier_rule="pair", roads_formula="eq4", second_norm=False, absent_neighbor_contribution="contributes"), "code": dict(adjacency_rule="bbox", barrier_rule="global", roads_formula="decayed", second_norm=True, absent_neighbor_contribution="swallowed")}`

- [ ] **Step 1: Write the failing anchor tests**

Create `tests/test_reference_impl.py`:

```python
"""Hand-derived anchors (spec 'Canonical numbers') pinning reference_impl.

Every number here is derived on paper from Eq. 1-4 in the manuscript and
double-derived by the spec's ultracode review; the derivation worksheet
(docs/oracle/derivation-worksheet.md) shows the arithmetic.
"""

import math

import pytest

from tests.oraculum_fixtures import (
    load_settlements, load_barriers, load_services,
)
from tests.reference_impl import RULESETS, adjacency, apply_barrier, compute_city

SQ2 = math.sqrt(2)
W_SQRT2 = 1 / (1 + SQ2)          # decay at 1000*sqrt(2) m
W_HALF = 0.5                      # decay at 1000 m
W_25 = 1 / 2.5                    # decay at 1500 m


@pytest.fixture(scope="module")
def city():
    return load_settlements()


@pytest.fixture(scope="module")
def barriers():
    return load_barriers()


@pytest.fixture(scope="module")
def services():
    return load_services()


IDEAL = {"A": {"B", "E"}, "B": {"A", "C", "RV", "E"}, "C": {"B", "E", "IND"},
         "RV": {"B"}, "D": {"E"}, "E": {"A", "B", "C", "D", "IND"},
         "IND": {"C", "E"}}
CODE = {"A": {"B", "E"}, "B": {"C", "RV", "E"}, "C": {"B", "E", "IND"},
        "RV": {"B"}, "D": {"E"}, "E": {"B", "C", "IND"}, "IND": {"C", "E"}}


def test_border_adjacency_severed_pairwise(city, barriers):
    nbrs = apply_barrier(adjacency(city, "border"), city, barriers, "pair")
    assert nbrs == IDEAL


def test_bbox_adjacency_with_global_barrier(city, barriers):
    nbrs = apply_barrier(adjacency(city, "bbox"), city, barriers, "global")
    assert nbrs == CODE


def test_bbox_equals_border_pre_barrier_for_rectangles(city):
    assert adjacency(city, "bbox") == adjacency(city, "border")


def _city_df(city, services, barriers, rule, **overrides):
    kwargs = dict(RULESETS[rule], scenario="baseline", denom="pop")
    kwargs.update(overrides)
    return compute_city(city, services, barriers, **kwargs)


def test_clinic_pcen_ideal_baseline_pop(city, services, barriers):
    df = _city_df(city, services, barriers, "ideal")
    exp = {
        "A": (2 + W_HALF + W_SQRT2) / 100,
        "B": 0.0175,
        "C": (W_HALF + W_SQRT2) / 400,
        "RV": 0.025,
        "D": 0.004,
        "E": (1 + 2 * W_SQRT2 + W_HALF) / 300,
        "IND": 0.04,
    }
    for sid, v in exp.items():
        assert df.loc[sid, "clinic_pcen"] == pytest.approx(v, abs=1e-12), sid


def test_clinic_pcen_code_rule_differences(city, services, barriers):
    df = _city_df(city, services, barriers, "code")
    assert df.loc["B", "clinic_pcen"] == pytest.approx(0.0125, abs=1e-12)
    assert df.loc["E", "clinic_pcen"] == pytest.approx(0.005, abs=1e-12)
    assert df.loc["A", "clinic_pcen"] == pytest.approx(
        (2 + W_HALF + W_SQRT2) / 100, abs=1e-12)


def test_school_pcen_ideal_and_unique_anchors(city, services, barriers):
    df = _city_df(city, services, barriers, "ideal")
    exp = {"A": SQ2 / 100, "B": 0.005, "C": (SQ2 - 1) / 400, "RV": 0.0,
           "D": 0.014, "E": (1 + (SQ2 - 1) + 0.4) / 300, "IND": 0.04}
    for sid, v in exp.items():
        assert df.loc[sid, "school_pcen"] == pytest.approx(v, abs=1e-12), sid
    pcen = df["school_pcen"]
    assert pcen.idxmax() == "IND" and (pcen == pcen.max()).sum() == 1
    assert pcen.idxmin() == "RV" and (pcen == pcen.min()).sum() == 1


def test_popdensity_denominator(city, services, barriers):
    df = _city_df(city, services, barriers, "ideal", denom="popdensity")
    # E: pop 300 / area 2.0 -> denominator 150
    assert df.loc["E", "clinic_pcen"] == pytest.approx(
        (1 + 2 * W_SQRT2 + W_HALF) / 150, abs=1e-12)
    # A: area 1.0 -> identical to popsize
    assert df.loc["A", "clinic_pcen"] == pytest.approx(
        (2 + W_HALF + W_SQRT2) / 100, abs=1e-12)
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/test_reference_impl.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.reference_impl'`.

- [ ] **Step 3: Write the implementation**

Create `tests/reference_impl.py`:

```python
"""Independent reference implementation of the PSI (manuscript Eq. 1-4).

Written from the equations in "Making the City Unequal" (pp. 14-16):
  Eq. 1: PSI = (ServiceIndex_1 + ... + ServiceIndex_n) / n
  Eq. 2: ServiceIndex_i = (PCEN_i - PCEN_min) / (PCEN_max - PCEN_min)
  Eq. 3: PCEN_mobile,i = (x_i + sum_j x_j * 1/(1 + d_ij)) / Population_i,
         j over neighbors of i, d in km
  Eq. 4: RoadsIndex_i from LengthPavedRoads_i / Population_i, min-maxed,
         with NO neighbor term.

INDEPENDENCE RULE: this module must never import, call, or mirror the
production spatial-index library module. It exists so production code can
be checked against the equations, not against itself.

Knobs (spec 'two rule-sets'): adjacency_rule, barrier_rule, roads_formula,
scenario, denom, second_norm, absent_neighbor_contribution. RULESETS binds
the ideal (manuscript) and code (empirical) combinations.
"""

import pandas as pd
from shapely.geometry import box

RULESETS = {
    "ideal": dict(adjacency_rule="border", barrier_rule="pair",
                  roads_formula="eq4", second_norm=False,
                  absent_neighbor_contribution="contributes"),
    "code": dict(adjacency_rule="bbox", barrier_rule="global",
                 roads_formula="decayed", second_norm=True,
                 absent_neighbor_contribution="swallowed"),
}

SCENARIOS = {
    # name: (dropped ids, dropped_before_neighbors)
    "baseline": (frozenset(), False),
    "excl_contributing": (frozenset({"RV", "IND"}), False),
    "excl_removed": (frozenset({"RV", "IND"}), True),
    "excl_ind_removed": (frozenset({"IND"}), True),
    "excl_rv_only": (frozenset({"RV"}), False),
}

POINT_SERVICES = ("clinic", "school", "bank", "police", "ration", "transport")


def adjacency(settlements, rule):
    idx = settlements.set_index("USO_AREA_U").geometry
    out = {}
    for i in idx.index:
        nbrs = set()
        for j in idx.index:
            if i == j:
                continue
            if rule == "border":
                inter = idx[i].intersection(idx[j])
                if not inter.is_empty and inter.length > 0:
                    nbrs.add(j)
            elif rule == "bbox":
                if idx[i].intersects(box(*idx[j].bounds)):
                    nbrs.add(j)
            elif rule == "intersects":
                if idx[i].intersects(idx[j]):
                    nbrs.add(j)
            else:
                raise ValueError(rule)
        out[i] = nbrs
    return out


def apply_barrier(nbrs, settlements, barriers, rule):
    if barriers is None or len(barriers) == 0:
        return nbrs
    idx = settlements.set_index("USO_AREA_U").geometry
    barrier_geoms = list(barriers.geometry)
    flagged = {i for i in idx.index
               if any(idx[i].intersects(b) for b in barrier_geoms)}
    out = {}
    for i, js in nbrs.items():
        if rule == "global":
            out[i] = js - flagged
        elif rule == "pair":
            kept = set()
            for j in js:
                shared = idx[i].intersection(idx[j])
                crossed = any(b.intersects(shared) for b in barrier_geoms)
                if not crossed:
                    kept.add(j)
            out[i] = kept
        else:
            raise ValueError(rule)
    return out


def _centroid_km(settlements):
    cent = settlements.set_index("USO_AREA_U").geometry.centroid
    return {i: cent[i] for i in cent.index}


def _service_amounts(settlements, services):
    """Per-settlement own amounts: counts for point services, km for road."""
    idx = settlements.set_index("USO_AREA_U").geometry
    amounts = {}
    for svc in POINT_SERVICES:
        gdf = services.get(svc)
        amounts[svc] = {
            i: 0 if gdf is None else
            int(sum(1 for g in gdf.geometry if g.within(idx[i])))
            for i in idx.index}
    road = services["road"].geometry.iloc[0]
    amounts["road"] = {i: road.intersection(idx[i]).length / 1000
                       for i in idx.index}
    return amounts


def compute_city(settlements, services, barriers, *, adjacency_rule,
                 barrier_rule, roads_formula, scenario, denom, second_norm,
                 absent_neighbor_contribution):
    dropped, drop_before = SCENARIOS[scenario]
    universe = settlements[~settlements["USO_AREA_U"].isin(dropped)] \
        if drop_before else settlements

    nbrs = apply_barrier(adjacency(universe, adjacency_rule),
                         universe, barriers, barrier_rule)
    cent = _centroid_km(universe)
    amounts = _service_amounts(universe, services)

    indexed = [i for i in universe["USO_AREA_U"]
               if drop_before or i not in dropped]
    meta = universe.set_index("USO_AREA_U")

    def denominator(i):
        pop = meta.loc[i, "population"]
        return pop / meta.loc[i, "area_km2"] if denom == "popdensity" else pop

    def contribution_weight(i, j):
        d_km = cent[i].distance(cent[j]) / 1000
        return 1 / (1 + d_km)

    rows = {}
    for i in indexed:
        row = {}
        for svc in POINT_SERVICES + ("road",):
            own = amounts[svc][i]
            decayed_sum = 0.0
            for j in nbrs[i]:
                if (not drop_before and j in dropped
                        and absent_neighbor_contribution == "swallowed"):
                    continue
                decayed_sum += amounts[svc][j] * contribution_weight(i, j)
            if svc == "road":
                row["road_length_km"] = own
                pcen = (own if roads_formula == "eq4"
                        else own + decayed_sum) / denominator(i)
                row["road_pcen"] = pcen
            else:
                row[f"{svc}_count"] = own
                row[f"{svc}_pcen"] = (own + decayed_sum) / denominator(i)
        rows[i] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    idx_cols = []
    for svc in POINT_SERVICES + ("road",):
        pcen = df[f"{svc}_pcen"]
        lo, hi = pcen.min(), pcen.max()
        df[f"{svc}_idx"] = 0.0 if hi == lo else (pcen - lo) / (hi - lo)
        idx_cols.append(f"{svc}_idx")
    df["psi_eq1"] = df[idx_cols].mean(axis=1)
    if second_norm:
        p = df["psi_eq1"]
        lo, hi = p.min(), p.max()
        df["norm_psi"] = 0.0 if hi == lo else (p - lo) / (hi - lo)
    return df
```

- [ ] **Step 4: Run to verify GREEN**

Run: `uv run pytest tests/test_reference_impl.py tests/test_fixture_invariants.py -v`
Expected: all pass.

- [ ] **Step 5: Independence check**

Run: `grep -nE '^[[:space:]]*(import|from)[[:space:]]+spatial_index_utils|spatial_index_utils\.' tests/reference_impl.py || echo INDEPENDENT`
Expected: `INDEPENDENT` (the docstring may MENTION the module name; imports and attribute calls may not exist).

- [ ] **Step 6: Commit**

```bash
git add tests/reference_impl.py tests/test_reference_impl.py
git commit -m "feat(oracle): independent Eq.1-4 reference implementation with rule-set knobs"
```

---

### Task 3: expected_values.csv emission + scenario/divergence anchors

**Files:**
- Modify: `tests/reference_impl.py` (append `emit_expected_values` + `__main__`)
- Modify: `tests/test_reference_impl.py` (append tests)
- Create: `tests/fixtures/oraculum/expected_values.csv` (generated, committed)

**Interfaces:**
- Produces: `emit_expected_values(out_path) -> DataFrame` writing long-format CSV with columns `rule,scenario,denom,settlement,metric,value` for rule ∈ {ideal, code} × all 5 scenarios × both denoms; metrics: `<svc>_count`/`road_length_km`, `<svc>_pcen`, `<svc>_idx` for all 7 services, `psi_eq1`, and `norm_psi` (rule=code only). Tasks 4–6 read this file.

- [ ] **Step 1: Write the failing tests (append to `tests/test_reference_impl.py`)**

```python
import itertools
from pathlib import Path

import pandas as pd

from tests.reference_impl import emit_expected_values

CSV = Path(__file__).resolve().parent / "fixtures" / "oraculum" / "expected_values.csv"


def test_expected_values_csv_complete():
    df = pd.read_csv(CSV)
    assert set(df.columns) == {"rule", "scenario", "denom", "settlement",
                               "metric", "value"}
    for rule, scenario, denom in itertools.product(
            ("ideal", "code"),
            ("baseline", "excl_contributing", "excl_removed",
             "excl_ind_removed", "excl_rv_only"),
            ("pop", "popdensity")):
        sub = df[(df["rule"] == rule) & (df["scenario"] == scenario)
                 & (df["denom"] == denom)]
        assert len(sub) > 0, (rule, scenario, denom)
        assert ("norm_psi" in set(sub["metric"])) == (rule == "code")


def _lookup(df, rule, scenario, denom, settlement, metric):
    m = df[(df["rule"] == rule) & (df["scenario"] == scenario)
           & (df["denom"] == denom) & (df["settlement"] == settlement)
           & (df["metric"] == metric)]
    assert len(m) == 1, (rule, scenario, denom, settlement, metric)
    return float(m["value"].iloc[0])


def test_csv_matches_hand_anchors():
    df = pd.read_csv(CSV)
    assert _lookup(df, "ideal", "baseline", "pop", "B", "clinic_pcen") == \
        pytest.approx(0.0175, abs=1e-12)
    assert _lookup(df, "ideal", "excl_removed", "pop", "B", "clinic_pcen") == \
        pytest.approx(0.0125, abs=1e-12)
    assert _lookup(df, "ideal", "excl_contributing", "pop", "B", "clinic_pcen") == \
        pytest.approx(0.0175, abs=1e-12)
    assert _lookup(df, "ideal", "baseline", "pop", "A", "road_pcen") == \
        pytest.approx(0.0075, abs=1e-12)
    assert _lookup(df, "ideal", "baseline", "popdensity", "E", "road_pcen") == \
        pytest.approx(0.005, abs=1e-12)
    assert _lookup(df, "code", "baseline", "pop", "A", "road_pcen") == \
        pytest.approx(0.010606601717798213, abs=1e-12)
    assert _lookup(df, "code", "baseline", "pop", "IND", "road_pcen") == \
        pytest.approx(0.03, abs=1e-12)


def test_code_excl_contributing_collapses_to_removed():
    """Schema self-consistency: the reference impl's `swallowed` knob makes
    the two scenarios' CSV blocks identical BY CONSTRUCTION. The
    production-facing pin of rule-set gap #5 (the real except:pass swallow)
    lives in tests/test_oracle.py::test_production_collapse_gap5."""
    df = pd.read_csv(CSV)
    a = df[(df["rule"] == "code") & (df["scenario"] == "excl_contributing")]
    b = df[(df["rule"] == "code") & (df["scenario"] == "excl_removed")]
    key = ["denom", "settlement", "metric"]
    merged = a.merge(b, on=key, suffixes=("_a", "_b"))
    assert len(merged) == len(a) == len(b)
    pd.testing.assert_series_equal(
        merged["value_a"], merged["value_b"], check_names=False,
        rtol=0, atol=1e-15)


def test_ideal_excl_contributing_differs_from_removed():
    df = pd.read_csv(CSV)
    va = _lookup(df, "ideal", "excl_contributing", "pop", "B", "clinic_pcen")
    vb = _lookup(df, "ideal", "excl_removed", "pop", "B", "clinic_pcen")
    assert va != pytest.approx(vb, abs=1e-9)


def test_ind_removal_is_pure_renormalization():
    """IND is serviceless: only _idx/psi move, never counts or pcen."""
    df = pd.read_csv(CSV)
    base = df[(df["rule"] == "ideal") & (df["scenario"] == "baseline")]
    ind = df[(df["rule"] == "ideal") & (df["scenario"] == "excl_ind_removed")]
    key = ["denom", "settlement", "metric"]
    merged = base.merge(ind, on=key, suffixes=("_base", "_ind"))
    pcen_rows = merged[merged["metric"].str.endswith(("_pcen", "_count", "_length_km"))]
    pd.testing.assert_series_equal(
        pcen_rows["value_base"], pcen_rows["value_ind"], check_names=False,
        rtol=0, atol=1e-15)
    clinic_idx = merged[merged["metric"] == "clinic_idx"]
    assert (clinic_idx["value_base"] != clinic_idx["value_ind"]).any()


def test_recorded_ties_are_ground_truth():
    df = pd.read_csv(CSV)
    # police tied argmax A/B (ideal, baseline, pop)
    pa = _lookup(df, "ideal", "baseline", "pop", "A", "police_pcen")
    pb = _lookup(df, "ideal", "baseline", "pop", "B", "police_pcen")
    assert pa == pytest.approx(pb, abs=1e-15) == pytest.approx(0.005, abs=1e-12)
    # road Eq.4 tied zero minimum
    for sid in ("B", "C", "RV", "D", "IND"):
        assert _lookup(df, "ideal", "baseline", "pop", sid, "road_pcen") == 0.0
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/test_reference_impl.py -v -k "csv or collapses or renorm or ties or differs"`
Expected: FAIL (`emit_expected_values` not defined / CSV missing).

- [ ] **Step 3: Append to `tests/reference_impl.py`**

```python
def emit_expected_values(out_path):
    from tests.oraculum_fixtures import (
        load_settlements, load_barriers, load_services,
    )
    city, barriers, services = (
        load_settlements(), load_barriers(), load_services())
    records = []
    for rule, kwargs in RULESETS.items():
        for scenario in SCENARIOS:
            for denom in ("pop", "popdensity"):
                df = compute_city(city, services, barriers, scenario=scenario,
                                  denom=denom, **kwargs)
                for sid, row in df.iterrows():
                    for metric, value in row.items():
                        records.append((rule, scenario, denom, sid,
                                        metric, value))
    out = pd.DataFrame(records, columns=["rule", "scenario", "denom",
                                         "settlement", "metric", "value"])
    out.to_csv(out_path, index=False, float_format="%.17g")
    return out


if __name__ == "__main__":
    from pathlib import Path
    target = (Path(__file__).resolve().parent / "fixtures" / "oraculum"
              / "expected_values.csv")
    emit_expected_values(target)
    print(f"wrote {target}")
```

- [ ] **Step 4: Generate the CSV and verify GREEN**

Run: `uv run python -m tests.reference_impl && uv run pytest tests/test_reference_impl.py -v`
Expected: CSV written; all tests pass.

- [ ] **Step 5: Create the consistency guard (spec §7's CSV-wide scope)**

Create `scripts/check_oraculum_invariants.py`:

```python
"""Spec §7 consistency guard over expected_values.csv (CSV-wide scope;
geometry-scope checks live in tests/test_fixture_invariants.py).

Run standalone (exit 1 on violation) or via its pytest wrapper:
    uv run python scripts/check_oraculum_invariants.py
"""

import sys
from pathlib import Path

import pandas as pd

CSV = (Path(__file__).resolve().parent.parent / "tests" / "fixtures"
       / "oraculum" / "expected_values.csv")
SERVICES = ("clinic", "school", "bank", "police", "ration", "transport",
            "road")
UNIQUE_ANCHOR_SERVICES = ("clinic", "school")


def check(df=None):
    df = pd.read_csv(CSV) if df is None else df
    violations = []
    groups = df[df["metric"].str.endswith("_pcen")].groupby(
        ["rule", "scenario", "denom", "metric"])
    for (rule, scenario, denom, metric), grp in groups:
        vals = grp["value"]
        if not vals.max() > vals.min():
            violations.append(
                f"degenerate min-max: {rule}/{scenario}/{denom}/{metric}")
        svc = metric[: -len("_pcen")]
        if svc in UNIQUE_ANCHOR_SERVICES:
            if (vals == vals.max()).sum() != 1:
                violations.append(
                    f"tied argmax: {rule}/{scenario}/{denom}/{metric}")
            if (vals == vals.min()).sum() != 1:
                violations.append(
                    f"tied argmin: {rule}/{scenario}/{denom}/{metric}")
    return violations


if __name__ == "__main__":
    problems = check()
    for p in problems:
        print("VIOLATION:", p)
    print("OK" if not problems else f"{len(problems)} violation(s)")
    sys.exit(1 if problems else 0)
```

Append its pytest wrapper to `tests/test_reference_impl.py`:

```python
def test_invariants_guard_csv_wide():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from check_oraculum_invariants import check
    assert check() == []
```

(add `import sys` to the file's imports if not present.)

Run: `uv run python scripts/check_oraculum_invariants.py && uv run pytest tests/test_reference_impl.py -v`
Expected: `OK`; all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/reference_impl.py tests/test_reference_impl.py tests/fixtures/oraculum/expected_values.csv scripts/check_oraculum_invariants.py
git commit -m "feat(oracle): emit expected_values.csv; anchors + CSV-wide invariants guard"
```

---

### Task 4: Library-first oracle suite (production vs expected values)

**Files:**
- Create: `tests/test_oracle.py`

**Interfaces:**
- Consumes: `run_production_chain` (Task 1), `expected_values.csv` (Task 3).

- [ ] **Step 1: Write the tests**

Create `tests/test_oracle.py`:

```python
"""Production code vs the oracle's expected values (rule=code rows).

run_production_chain mirrors the real scripts' wiring; every comparison here
is production-vs-hand-anchored-reference. A failure means production
behavior changed (or the oracle is wrong) — investigate, never blindly
update the CSV (oracle contract: docs/superpowers/specs/
2026-08-17-phase2-oracle-design.md).
"""

from pathlib import Path

import pandas as pd
import pytest

from tests.oraculum_fixtures import (
    load_settlements, load_barriers, load_services, run_production_chain,
)

CSV = Path(__file__).resolve().parent / "fixtures" / "oraculum" / "expected_values.csv"

# production column name -> expected_values metric name
METRICS = {
    "clinic_pcen": "clinic_pcen", "clinic_idx": "clinic_idx",
    "school_pcen": "school_pcen", "school_idx": "school_idx",
    "bank_pcen": "bank_pcen", "police_pcen": "police_pcen",
    "ration_pcen": "ration_pcen", "transport_pcen": "transport_pcen",
    "road_pcen": "road_pcen", "road_idx": "road_idx",
    "unnorm_psi": "psi_eq1", "norm_psi": "norm_psi",
}


@pytest.fixture(scope="module")
def expected():
    return pd.read_csv(CSV)


def _expected_frame(expected, scenario, denom):
    sub = expected[(expected["rule"] == "code")
                   & (expected["scenario"] == scenario)
                   & (expected["denom"] == denom)]
    return sub.pivot(index="settlement", columns="metric", values="value")


def _production_frame(denom, drop_ids_post=frozenset(), drop_ids_pre=frozenset()):
    city = load_settlements()
    if drop_ids_pre:
        city = city[~city["USO_AREA_U"].isin(drop_ids_pre)]
    result = run_production_chain(
        city, load_barriers(), load_services(), denom,
        drop_ids_post=drop_ids_post)
    return result.set_index("USO_AREA_U")


SCENARIO_WIRING = [
    # (scenario, drop_pre, drop_post)
    ("baseline", frozenset(), frozenset()),
    ("excl_rv_only", frozenset(), frozenset({"RV"})),
    ("excl_contributing", frozenset(), frozenset({"RV", "IND"})),
    ("excl_removed", frozenset({"RV", "IND"}), frozenset()),
    ("excl_ind_removed", frozenset({"IND"}), frozenset()),
]


@pytest.mark.parametrize("denom", ["pop", "popdensity"])
@pytest.mark.parametrize("scenario,drop_pre,drop_post", SCENARIO_WIRING)
def test_production_matches_code_rows(expected, scenario, drop_pre,
                                      drop_post, denom):
    exp = _expected_frame(expected, scenario, denom)
    got = _production_frame(denom, drop_ids_post=drop_post,
                            drop_ids_pre=drop_pre)
    assert set(got.index) == set(exp.index)
    for prod_col, metric in METRICS.items():
        for sid in exp.index:
            assert got.loc[sid, prod_col] == pytest.approx(
                exp.loc[sid, metric], abs=1e-12), (scenario, denom, sid, prod_col)


def test_zero_service_settlement(expected):
    got = _production_frame("pop")
    assert got.loc["C", "clinic_count"] == 0
    assert got.loc["C", "clinic_pcen"] > 0  # entirely from decayed neighbors


def test_second_order_neighbor_excluded(expected):
    got = _production_frame("pop")
    assert "A" not in set(got.loc["C", "nbrs_bbox"])


def test_barrier_rule_is_global_and_directed(expected):
    got = _production_frame("pop")
    assert "A" not in set(got.loc["B", "nbrs_bbox"])   # A stripped from B
    assert set(got.loc["A", "nbrs_bbox"]) == {"B", "E"}  # A keeps its own


def test_popdensity_differs_from_popsize(expected):
    pop = _production_frame("pop")
    dens = _production_frame("popdensity")
    assert pop.loc["E", "clinic_pcen"] != pytest.approx(
        dens.loc["E", "clinic_pcen"], abs=1e-15)


def test_road_decay_divergence(expected):
    """Code roads are decayed; Eq. 4 has no neighbor term (rule-set gap #3)."""
    got = _production_frame("pop")
    ideal = expected[(expected["rule"] == "ideal")
                     & (expected["scenario"] == "baseline")
                     & (expected["denom"] == "pop")
                     & (expected["metric"] == "road_pcen")] \
        .set_index("settlement")["value"]
    assert got.loc["D", "road_pcen"] == pytest.approx(0.003, abs=1e-12)
    assert ideal["D"] == 0.0
    assert got.loc["A", "road_pcen"] == pytest.approx(0.010606601717798213,
                                                      abs=1e-12)
    assert ideal["A"] == pytest.approx(0.0075, abs=1e-12)


def test_second_normalization_divergence(expected):
    got = _production_frame("pop")
    assert got["norm_psi"].min() == pytest.approx(0.0, abs=1e-12)
    assert got["norm_psi"].max() == pytest.approx(1.0, abs=1e-12)
    assert not got["unnorm_psi"].equals(got["norm_psi"])


def test_minmax_anchors_unique(expected):
    got = _production_frame("pop")
    for svc in ("clinic", "school"):
        pcen = got[f"{svc}_pcen"]
        assert (pcen == pcen.max()).sum() == 1, svc
        assert (pcen == pcen.min()).sum() == 1, svc


@pytest.mark.parametrize("denom", ["pop", "popdensity"])
def test_production_collapse_gap5(expected, denom):
    """Rule-set gap #5, pinned against PRODUCTION: dropping rows after
    neighbor computation (except:pass swallows the missing contributions)
    equals dropping them before — semantics (a) degenerates to (b) in the
    real code, not just in the reference impl's model of it."""
    post = _production_frame(denom, drop_ids_post=frozenset({"RV", "IND"}))
    pre = _production_frame(denom, drop_ids_pre=frozenset({"RV", "IND"}))
    assert set(post.index) == set(pre.index)
    for col in [c for c in post.columns
                if c.endswith(("_pcen", "_idx")) or c in ("unnorm_psi",
                                                          "norm_psi")]:
        for sid in post.index:
            assert post.loc[sid, col] == pytest.approx(
                pre.loc[sid, col], abs=1e-12), (denom, sid, col)
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_oracle.py -v`
Expected: ALL PASS. Any failure here is a production-vs-oracle mismatch: investigate against the reference impl and worksheet; if production genuinely disagrees with the pinned code-rule rows, STOP (methodology red line) — that means the empirical pin drifted mid-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_oracle.py
git commit -m "feat(oracle): library-first suite — production vs expected values"
```

---

### Task 5: End-to-end CLI test

**Files:**
- Create: `tests/test_oracle_e2e.py`

**Interfaces:**
- Consumes: fixture loaders (Task 1); `expected_values.csv` `rule=code, scenario=excl_rv_only` rows (Task 3); the real `scripts/preprocess.py` + `scripts/compute_psi.py`.

- [ ] **Step 1: Write the test**

Create `tests/test_oracle_e2e.py`:

```python
"""Real-CLI end-to-end: temp data dir -> preprocess -> compute_psi -> compare.

Manifest per spec §3: colonies shapefile, three Barrier_Clip layers (drain/
railway empty-but-valid), ndmc_center, delhi_bounds_buffer, seven Public
Services layers (four singletons so no service is degenerate), population
CSV. The CLI path hardcodes RV-only exclusion -> compare against
rule=code / scenario=excl_rv_only.
"""

import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from tests.oraculum_fixtures import (
    EPSG, load_settlements, load_barriers, load_services,
)

CSV = Path(__file__).resolve().parent / "fixtures" / "oraculum" / "expected_values.csv"
REPO = Path(__file__).resolve().parent.parent

SERVICE_LAYOUT = {
    "clinic": ("Health", "Health.shp"),
    "school": ("School", "schools7760.shp"),
    "bank": ("Banking", "Banking.shp"),
    "police": ("Police", "Police Station.shp"),
    "ration": ("Ration", "Ration.shp"),
    "transport": ("Transport", "Transport.shp"),
    "road": ("Major Road", "Road.shp"),
}


def _empty_line_layer(path):
    gdf = gpd.GeoDataFrame(
        {"name": ["placeholder"]},
        geometry=[LineString([(0, 0), (1, 1)])], crs=f"EPSG:{EPSG}")
    # a real but far-away line so the layer is valid yet touches nothing
    gdf.geometry = gdf.translate(xoff=2_000_000, yoff=2_000_000)
    gdf.to_file(path)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("oraculum_data")
    city = load_settlements()

    (root / "uso_update_sep2021").mkdir()
    # Drop `population`: compute_psi merges the population CSV and renames
    # it to `population` — a second column of the same name crashes it
    # (verified in plan review; the real dataset's shapefile has no
    # population field either).
    city.drop(columns=["population"]).to_file(
        root / "uso_update_sep2021" / "uso_update_sep2021.shp")

    barrier_dir = root / "Barrier_Clip"
    (barrier_dir / "Canal").mkdir(parents=True)
    load_barriers().to_file(barrier_dir / "Canal" / "Canal.shp")
    for sub, fname in [("Drain", "Major_Drain.shp"),
                       ("Railway", "Railway_Line.shp")]:
        (barrier_dir / sub).mkdir()
        _empty_line_layer(barrier_dir / sub / fname)

    (root / "ndmc_center7760").mkdir()
    gpd.GeoDataFrame({"name": ["ndmc"]},
                     geometry=[Point(1_001_500, 1_001_500)],
                     crs=f"EPSG:{EPSG}").to_file(
        root / "ndmc_center7760" / "ndmc_center7760.shp")

    (root / "delhi_bounds_buffer").mkdir()
    gpd.GeoDataFrame({"name": ["bounds"]},
                     geometry=[box(1_000_000 - 10_000, 1_000_000 - 10_000,
                                   1_000_000 + 10_000, 1_000_000 + 10_000)],
                     crs=f"EPSG:{EPSG}").to_file(
        root / "delhi_bounds_buffer" / "delhi_bounds_buffer.shp")

    services = load_services()
    for svc, (folder, fname) in SERVICE_LAYOUT.items():
        d = root / "Public Services" / folder
        d.mkdir(parents=True)
        services[svc].to_file(d / fname)

    pop = load_settlements()[["USO_AREA_U", "population"]].rename(
        columns={"USO_AREA_U": "uso_area_u"})
    pop.to_csv(root / "pop_colony_wp_2020_jjc_adjusted.csv", index=False)
    return root


def _run(script, *args):
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *args],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout}\n{proc.stderr}"
    return proc


def test_full_cli_chain_matches_excl_rv_only(data_dir, tmp_path):
    out_dir = tmp_path / "out"
    _run("preprocess.py", "--data-dir", str(data_dir),
         "--out-dir", str(out_dir))
    nbrs_file = out_dir / "colonies_bbox_nbrs_aug2026.joblib"
    assert nbrs_file.exists()
    _run("compute_psi.py", "--data-dir", str(data_dir),
         "--neighbors-file", str(nbrs_file), "--out-dir", str(out_dir))

    got = pd.read_csv(out_dir / "psi_2020_results"
                      / "delhi_psi_bbox_popsize2020_norv_aug2026.csv")
    got = got.set_index("USO_AREA_U")

    exp = pd.read_csv(CSV)
    exp = exp[(exp["rule"] == "code") & (exp["scenario"] == "excl_rv_only")
              & (exp["denom"] == "pop")] \
        .pivot(index="settlement", columns="metric", values="value")

    assert set(got.index) == set(exp.index) == {"A", "B", "C", "D", "E", "IND"}
    # real pipeline service naming: clinic->health, road count renamed length
    mapping = {
        "health_pcen": "clinic_pcen", "health_idx": "clinic_idx",
        "school_pcen": "school_pcen", "school_idx": "school_idx",
        "bank_pcen": "bank_pcen", "police_pcen": "police_pcen",
        "ration_pcen": "ration_pcen", "transport_pcen": "transport_pcen",
        "road_pcen": "road_pcen", "road_idx": "road_idx",
        "unnorm_psi": "psi_eq1", "norm_psi": "norm_psi",
    }
    for got_col, metric in mapping.items():
        for sid in exp.index:
            assert got.loc[sid, got_col] == pytest.approx(
                exp.loc[sid, metric], abs=1e-9), (got_col, sid)
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_oracle_e2e.py -v`
Expected: PASS (slow-ish: two subprocesses; well under a minute on fixture data). Column-name mismatches with the real pipeline output are fix-forward territory — adjust the test's `mapping`, never the scripts.

- [ ] **Step 3: Commit**

```bash
git add tests/test_oracle_e2e.py
git commit -m "feat(oracle): end-to-end CLI test against excl_rv_only expected rows"
```

---

### Task 6: Divergence exhibit suite

**Files:**
- Create: `tests/test_divergence_exhibit.py`

**Interfaces:**
- Consumes: `load_exhibit` (Task 1), `adjacency` (Task 2).

- [ ] **Step 1: Write the tests**

Create `tests/test_divergence_exhibit.py`:

```python
"""The divergence exhibit: where bbox/intersects adjacency and the
manuscript's border-sharing rule disagree, with pinned PCEN deltas.

These tests PASS by asserting the documented divergence itself; a failure
means adjacency behavior changed without updating the record (spec §
Divergence exhibit)."""

import math

import pytest

from tests.oraculum_fixtures import load_exhibit
from tests.reference_impl import adjacency


@pytest.fixture(scope="module")
def exhibit():
    gdf = load_exhibit().rename(columns={"id": "USO_AREA_U"})
    return gdf


def _clinic_pcen(gdf, nbrs):
    cent = gdf.set_index("USO_AREA_U").geometry.centroid
    counts = gdf.set_index("USO_AREA_U")["clinics"]
    pops = gdf.set_index("USO_AREA_U")["population"]
    out = {}
    for i in counts.index:
        total = float(counts[i])
        for j in nbrs[i]:
            d_km = cent[i].distance(cent[j]) / 1000
            total += counts[j] / (1 + d_km)
        out[i] = total / pops[i]
    return out


def test_border_rule_no_neighbors(exhibit):
    nbrs = adjacency(exhibit, "border")
    assert nbrs == {"P": set(), "Q": set(), "R": set(), "S": set()}


def test_bbox_rule_invents_both_divergence_flavors(exhibit):
    """bbox catches the containment phantom (Q->P, directed) AND the corner
    touch (R<->S, since rectangles' bboxes equal their geometry) —
    production-verified in plan review round 1."""
    nbrs = adjacency(exhibit, "bbox")
    assert nbrs["Q"] == {"P"}          # Q's geometry lies inside P's bbox
    assert nbrs["P"] == set()          # P's geometry misses Q's bbox
    assert nbrs["R"] == {"S"} and nbrs["S"] == {"R"}


def test_intersects_rule_only_corner_touch(exhibit):
    nbrs = adjacency(exhibit, "intersects")
    assert nbrs["R"] == {"S"} and nbrs["S"] == {"R"}
    assert nbrs["P"] == set() and nbrs["Q"] == set()


def test_pinned_pcen_deltas(exhibit):
    border = _clinic_pcen(exhibit, adjacency(exhibit, "border"))
    bbox = _clinic_pcen(exhibit, adjacency(exhibit, "bbox"))
    inter = _clinic_pcen(exhibit, adjacency(exhibit, "intersects"))

    assert border["Q"] == pytest.approx(0.01, abs=1e-12)
    assert bbox["Q"] - border["Q"] == pytest.approx(0.005147186, abs=1e-9)
    assert bbox["P"] - border["P"] == pytest.approx(0.0, abs=1e-15)
    assert bbox["S"] - border["S"] == pytest.approx(0.016568542494923802,
                                                   abs=1e-12)
    assert bbox["R"] - border["R"] == pytest.approx(0.0, abs=1e-15)

    assert inter["S"] - border["S"] == pytest.approx(0.016568542, abs=1e-9)
    assert inter["R"] - border["R"] == pytest.approx(0.0, abs=1e-15)
    assert inter["Q"] - border["Q"] == pytest.approx(0.0, abs=1e-15)

    # spot-check the geometry behind Q's delta: P centroid at (833.33, 833.33)
    d_km = math.hypot(1500 - 2500 / 3, 1500 - 2500 / 3) / 1000
    assert 1 / (1 + d_km) / 100 == pytest.approx(bbox["Q"] - border["Q"],
                                                 abs=1e-9)
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_divergence_exhibit.py -v`
Expected: ALL PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_divergence_exhibit.py
git commit -m "feat(oracle): divergence exhibit — directed bbox phantom + corner touch, pinned deltas"
```

---

### Task 7: Maps

**Files:**
- Create: `scripts/render_oracle_maps.py`, `docs/oracle/oraculum_city.png`, `docs/oracle/oraculum_exclusion_variants.png`, `docs/oracle/oraculum_divergence.png`

**Interfaces:**
- Consumes: fixture loaders, `reference_impl.compute_city`, `expected_values.csv`.
- **Before writing any chart code, load the `dataviz` skill** (repo rule) — the code below fixes CONTENT (what each figure must show); the skill governs styling and may restyle freely as long as content assertions hold.

- [ ] **Step 1: Write the renderer**

Create `scripts/render_oracle_maps.py`:

```python
"""Render the three Oraculum figures (spec §4) deterministically from
fixtures. Content contract per figure is in the spec; regenerate with:
    uv run python scripts/render_oracle_maps.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests.oraculum_fixtures import (   # noqa: E402
    load_settlements, load_barriers, load_services, load_exhibit,
)
from tests.reference_impl import RULESETS, adjacency, apply_barrier  # noqa: E402

OUT = REPO / "docs" / "oracle"
CSV = REPO / "tests" / "fixtures" / "oraculum" / "expected_values.csv"

TYPE_COLORS = {
    "Planned": "#4c78a8", "UC": "#f58518", "JJC": "#e45756",
    "RV": "#72b7b2", "RUAC": "#54a24b", "IND": "#b279a2",
}
SERVICE_MARKERS = {"clinic": ("o", "#d62728"), "school": ("s", "#1f77b4"),
                   "bank": ("^", "#2ca02c"), "police": ("v", "#9467bd"),
                   "ration": ("D", "#8c564b"), "transport": ("P", "#e377c2")}


def _draw_city(ax, city, barriers, services, nbrs=None, ghost=frozenset(),
               annotate=None):
    for _, row in city.iterrows():
        sid = row["USO_AREA_U"]
        ghosted = sid in ghost
        ax.add_patch(plt.Polygon(
            list(row.geometry.exterior.coords),
            facecolor=TYPE_COLORS[row["USO_FINAL"]],
            alpha=0.25 if ghosted else 0.6,
            edgecolor="grey" if ghosted else "black",
            linestyle="--" if ghosted else "-", linewidth=1.2))
        c = row.geometry.centroid
        label = f"{sid}\npop {row['population']}\n{row['area_km2']:g} km²"
        if annotate and sid in annotate:
            label += f"\n{annotate[sid]}"
        ax.annotate(label, (c.x, c.y), ha="center", va="center", fontsize=7)
    for _, b in barriers.iterrows():
        x, y = b.geometry.xy
        ax.plot(x, y, color="#00bfff", linewidth=4, solid_capstyle="butt",
                zorder=5, label="canal (barrier)")
    if services:
        for svc, gdf in services.items():
            if svc == "road":
                x, y = gdf.geometry.iloc[0].xy
                ax.plot(x, y, color="#555555", linewidth=2.5,
                        linestyle=(0, (4, 2)), zorder=4)
                continue
            marker, color = SERVICE_MARKERS[svc]
            xs = [g.x for g in gdf.geometry]
            ys = [g.y for g in gdf.geometry]
            ax.scatter(xs, ys, marker=marker, color=color, s=28, zorder=6,
                       label=svc)
    if nbrs is not None:
        cent = city.set_index("USO_AREA_U").geometry.centroid
        for i, js in nbrs.items():
            for j in js:
                dashed = i in ghost or j in ghost
                ax.annotate(
                    "", xy=(cent[j].x, cent[j].y),
                    xytext=(cent[i].x, cent[i].y),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                    linestyle="--" if dashed else "-",
                                    color="#333333", alpha=0.55,
                                    shrinkA=18, shrinkB=18))
    ax.set_aspect("equal")
    ax.set_axis_off()


def _code_nbrs(city, barriers):
    return apply_barrier(adjacency(city, "bbox"), city, barriers, "global")


def render_city():
    city, barriers, services = (load_settlements(), load_barriers(),
                                load_services())
    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    _draw_city(ax, city, barriers, services, nbrs=_code_nbrs(city, barriers))
    ax.set_title("Oraculum — settlements, services, canal, and the code "
                 "rule's DIRECTED neighbor graph\n(arrow i→j: i counts j's "
                 "services; note A→E has no reverse arrow)")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=7)
    fig.savefig(OUT / "oraculum_city.png", bbox_inches="tight")
    plt.close(fig)


def render_exclusion_variants():
    city, barriers, services = (load_settlements(), load_barriers(),
                                load_services())
    exp = pd.read_csv(CSV)

    def psi_note(scenario):
        sub = exp[(exp["rule"] == "code") & (exp["scenario"] == scenario)
                  & (exp["denom"] == "pop") & (exp["metric"] == "clinic_pcen")]
        return {r["settlement"]: f"clinic {r['value']:.4f}"
                for _, r in sub.iterrows()
                if r["settlement"] in ("B", "C", "E")}

    panels = [
        ("baseline — all seven", frozenset(), frozenset(), "baseline"),
        ("excl_contributing — RV/IND ghosted\n(code rule: contributions "
         "SWALLOWED — collapses to removal)", frozenset({"RV", "IND"}),
         frozenset(), "excl_contributing"),
        ("excl_removed — RV/IND gone", frozenset({"RV", "IND"}),
         frozenset({"RV", "IND"}), "excl_removed"),
        ("excl_ind_removed — IND only\n(pure renormalization)",
         frozenset({"IND"}), frozenset({"IND"}), "excl_ind_removed"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), dpi=150)
    for ax, (title, ghost, hide, scenario) in zip(axes, panels):
        sub_city = city[~city["USO_AREA_U"].isin(hide)]
        _draw_city(ax, sub_city, barriers, None,
                   nbrs=_code_nbrs(sub_city, barriers),
                   ghost=ghost - hide, annotate=psi_note(scenario))
        ax.set_title(title, fontsize=9)
    fig.suptitle("Oraculum exclusion scenarios (code rule, popsize) — "
                 "annotated clinic PCENs show contribution vs renormalization "
                 "effects", fontsize=11)
    fig.savefig(OUT / "oraculum_exclusion_variants.png", bbox_inches="tight")
    plt.close(fig)


def render_divergence():
    ex = load_exhibit().rename(columns={"id": "USO_AREA_U"})
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for _, row in ex.iterrows():
        ax.add_patch(plt.Polygon(list(row.geometry.exterior.coords),
                                 facecolor="#4c78a8", alpha=0.45,
                                 edgecolor="black"))
        b = row.geometry.bounds
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                   fill=False, linestyle="--",
                                   edgecolor="#e45756", linewidth=1.2))
        c = row.geometry.centroid
        ax.annotate(f"{row['USO_AREA_U']}\npop {row['population']}, "
                    f"{row['clinics']} clinic(s)", (c.x, c.y),
                    ha="center", fontsize=8)
    cent = ex.set_index("USO_AREA_U").geometry.centroid
    ax.annotate("", xy=(cent["P"].x, cent["P"].y),
                xytext=(cent["Q"].x, cent["Q"].y),
                arrowprops=dict(arrowstyle="-|>", color="#e45756", lw=1.6))
    ax.annotate("phantom bbox link Q→P\nQ clinic PCEN +0.005147",
                xy=(cent["Q"].x, cent["Q"].y + 500), ha="center", fontsize=8,
                color="#e45756")
    ax.annotate("point touch R–S\n(bbox AND `intersects` count it:\n"
                "S +0.016569)",
                xy=(1_000_000 + 5000, 1_000_000 + 950), ha="center",
                fontsize=8, color="#b279a2")
    ax.plot([cent["R"].x, cent["S"].x], [cent["R"].y, cent["S"].y],
            color="#b279a2", lw=1.4, linestyle=":")
    ax.set_title("Divergence exhibit — polygon (solid) vs bounding box "
                 "(dashed): where bbox/intersects adjacency invents "
                 "neighbors that border-sharing denies")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.autoscale_view()
    fig.savefig(OUT / "oraculum_divergence.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    render_city()
    render_exclusion_variants()
    render_divergence()
    print(f"wrote 3 figures to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Load the `dataviz` skill, then render and inspect**

Run: `uv run python scripts/render_oracle_maps.py`
Expected: three PNGs in `docs/oracle/`. Open each (Read tool) and verify the content contract: directed arrows visible with the A→E asymmetry; ghosted panels distinguishable; bbox overlay + phantom arrow labeled. Restyle per dataviz guidance if needed (content assertions above must survive).

- [ ] **Step 3: Commit**

```bash
git add scripts/render_oracle_maps.py docs/oracle/*.png
git commit -m "feat(oracle): deterministic maps — city, exclusion variants, divergence exhibit"
```

---

### Task 8: Worksheet, memo, changelog

**Files:**
- Create: `docs/oracle/derivation-worksheet.md`, `docs/oracle/exclusion-semantics-memo.md`
- Modify: `CHANGELOG.md` (`[Unreleased]`)

**Interfaces:**
- Consumes: `expected_values.csv` for the machine-checked tables; the anchor numbers below are HAND-DERIVED and must be typed from the derivations, not copied from the CSV (that is the whole point).

- [ ] **Step 1: Write the worksheet**

Create `docs/oracle/derivation-worksheet.md` with exactly this content (the anchor-subset arithmetic is complete; extend only the boilerplate lists mechanically):

```markdown
# Oraculum Derivation Worksheet

**STATUS: RATIFICATION PENDING** — expected values derived by Claude from
Eq. 1–4 of "Making the City Unequal" (pp. 14–16); awaiting hand verification
by Bob (and Raj). To ratify: check the anchor subset below with a
calculator, then change this line to `STATUS: RATIFIED by <name> on <date>`
and commit.

**Scope of hand ratification (the ~15-minute pass):** the
`ideal`/`baseline`/`pop` configuration for all seven settlements
(clinic/school/road below; the four singleton services follow the same
two-line pattern), plus the four worked extras at the end. Everything else
in `tests/fixtures/oraculum/expected_values.csv` is machine-cross-checked by
the independent reference implementation (`tests/reference_impl.py`), whose
correctness these anchors ratify. The e2e CLI leg is three-way-verified only
for the `excl_rv_only` configuration (the real pipeline's hardcoded filter).

Map: `oraculum_city.png`. Decays: 1 km → 1/2; 1.5 km → 0.4;
√2 km → √2−1 ≈ 0.414214; (√5)/2 km → ≈ 0.472136.

Ideal neighbor lists (A–D severed by canal, both directions):
A:[B,E] B:[A,C,RV,E] C:[B,E,IND] RV:[B] D:[E] E:[A,B,C,D,IND] IND:[C,E]

## Clinics (counts: A 2, B 1, E 1, RV 2) — Eq. 3, popsize

| i | arithmetic | PCEN |
|---|-----------|------|
| A | (2 + 1·½ [B] + 1·(√2−1) [E]) / 100 = 2.914214/100 | 0.02914214 |
| B | (1 + 2·½ [A] + 0 [C] + 2·½ [RV] + 1·½ [E]) / 200 = 3.5/200 | 0.01750000 |
| C | (0 + 1·½ [B] + 1·(√2−1) [E] + 0 [IND]) / 400 = 0.914214/400 | 0.00228553 |
| RV | (2 + 1·½ [B]) / 100 = 2.5/100 | 0.02500000 |
| D | (0 + 1·0.4 [E]) / 100 | 0.00400000 |
| E | (1 + 2·(√2−1) [A] + 1·½ [B] + 0 + 0 + 0) / 300 = 2.328427/300 | 0.00776142 |
| IND | (0 + 0 [C] + 1·0.4 [E]) / 10 | 0.04000000 |

Eq. 2 anchors: min = C (0.00228553), max = IND (0.04) — both unique.
Example index: A_idx = (0.02914214 − 0.00228553)/(0.04 − 0.00228553)
= 0.02685661/0.03771447 = **0.71210747**.

## Schools (A 1, D 1, E 1) — Eq. 3, popsize

| i | arithmetic | PCEN |
|---|-----------|------|
| A | (1 + 1·(√2−1) [E]) / 100 = √2/100 | 0.01414214 |
| B | (0 + 1·½ [A] + 1·½ [E]) / 200 | 0.00500000 |
| C | (0 + 1·(√2−1) [E]) / 400 | 0.00103553 |
| RV | 0 / 100 (B has no school) | 0 |
| D | (1 + 1·0.4 [E]) / 100 | 0.01400000 |
| E | (1 + 1·(√2−1) [A] + 1·0.4 [D]) / 300 = 1.814214/300 | 0.00604738 |
| IND | (0 + 1·0.4 [E]) / 10 | 0.04000000 |

min = RV (0), max = IND (0.04) — unique. Note the deliberate near-tie A vs
D (0.014142 vs 0.014): E's school at different decay is what separates them.

## Roads — Eq. 4 literally (NO neighbor term), lengths A 0.75 km, E 0.75 km

pop: A = 0.75/100 = **0.0075**; E = 0.75/300 = **0.0025**;
B = C = RV = D = IND = **0 exactly** (tied minimum — recorded ground truth;
Eq. 4 gives every road-less settlement zero).
popdensity: A = 0.0075; E = 0.75/150 = 0.005.
(The production code decays roads like Eq. 3 — a documented divergence, not
part of this ideal derivation; see the memo.)

## Singleton services (bank@A, police@B, ration@D, transport@E)

Pattern: PCEN_i = (own + [X ∈ nbrs(i)] · decay) / pop_i. E.g. police (X=B):
B = 1/200 = 0.005; A = 1·½/100 = 0.005 (**tied argmax — recorded**);
C = 1·½/400 = 0.00125; RV = 1·½/100 = 0.005 — wait: RV's list is [B], so
RV = 1·½/100 = 0.005 as well (three-way tie A/RV at 0.005 with B — all
recorded in the CSV; ties outside clinics/schools are expected ground
truth). E = 1·½/300 = 0.00166667; D = 0; IND = 0.

## Worked extras (complete the anchor subset)

1. **Exclusion delta (B, ideal, excl_removed, pop):** RV and IND removed
   before neighbor computation → B's list [A,C,E]:
   (1 + 2·½ + 0 + 1·½)/200 = 2.5/200 = **0.0125** (vs 0.0175 baseline —
   the RV contribution effect, −0.005).
2. **Renormalization delta (A clinic_idx, ideal, excl_ind_removed, pop):**
   PCENs unchanged (IND serviceless); max moves from IND (0.04) to A
   (0.02914214); min still C. A_idx = (0.02914214−0.00228553)/
   (0.02914214−0.00228553) = **1.0** exactly (was 0.71210747) — anchor
   movement with zero numerator change. This delta is denominator-INVARIANT
   because A, C, IND all have area 1.0 km².
3. **Popdensity coverage (E clinic, ideal, baseline):**
   popsize 2.328427/300 = **0.00776142**; popdensity divides by
   pop/area = 300/2 = 150 → 2.328427/150 = **0.01552285**.
4. **Road Eq. 4 value (A, pop):** 0.75/100 = **0.0075** (worked above).

## Machine-checked remainder

All other configurations (code rule-set incl. directed barrier asymmetry
and decayed roads; excl_contributing/excl_rv_only; norm_psi; popdensity
tables; the four singleton services' full tables; the divergence exhibit
deltas) are asserted equal, to 1e-12, between the reference implementation
and the production code by `uv run pytest` — their authority derives from
these hand anchors plus the reviewed independence of the reference
implementation.
```

NOTE (for the implementer): the police RV line above shows honest hand math
discovering a THREE-way tie (A, RV at 0.005 alongside B... verify: B own =
0.005; A = 0.005; RV = 0.005). Cross-check against the CSV; if the CSV
shows the same values, keep the text as-is (it is a recorded tie, allowed
outside clinics/schools). If the CSV disagrees, STOP — that is an
oracle-contract mismatch.

- [ ] **Step 2: Write the memo**

Create `docs/oracle/exclusion-semantics-memo.md`:

````markdown
# Memo to Raj: Exclusion semantics, and what the code actually does

*(Machine-verified against the Oraculum oracle; no recommendation is made —
this grounds WORKPLAN "Open decisions A". Figures:
`oraculum_exclusion_variants.png`, `oraculum_city.png`.)*

## The question

When we drop settlement types (RV now; RV + industrial after the
recategorization), do the dropped settlements still LEND their services to
adjacent settlements' accessibility (semantics **a**), or vanish entirely
(semantics **b**)?

## What the mythical city shows (ideal rule, popsize, clinic PCEN for B)

| configuration | B's clinic PCEN | why |
|---------------|-----------------|-----|
| baseline (all 7) | 0.0175 | B counts RV's 2 clinics at decay ½ |
| semantics (a) — RV/IND excluded but contributing | 0.0175 | index rows dropped; services still lend |
| semantics (b) — RV/IND fully removed | 0.0125 | RV's clinics vanish from B's numerator |

## Full per-settlement delta tables (both denominators)

*(Implementer: generate the two tables below from
`tests/fixtures/oraculum/expected_values.csv` with this snippet and paste
the markdown output here — one table for `denom=pop`, one for
`denom=popdensity`; rows = all indexed settlements; columns = clinic_pcen
and clinic_idx under baseline / excl_contributing / excl_removed /
excl_ind_removed; rule=code. The snippet keeps the memo mechanically in
sync with the oracle.)*

```python
# uv run python - <<'PY'
import pandas as pd
df = pd.read_csv("tests/fixtures/oraculum/expected_values.csv")
for denom in ("pop", "popdensity"):
    sub = df[(df["rule"] == "code") & (df["denom"] == denom)
             & (df["metric"].isin(["clinic_pcen", "clinic_idx"]))]
    wide = sub.pivot_table(index="settlement",
                           columns=["metric", "scenario"], values="value")
    print(f"\n### denom = {denom}\n")
    print(wide.round(6).to_markdown())
# PY
```

Separately, removing serviceless IND alone changes NOBODY's numerator but
moves the clinic max-anchor from IND (0.04) to A (0.0291): every
settlement's clinic index rescales (A: 0.712 → 1.000 exactly). Dropping a
settlement type can change results through *renormalization alone*.

## What the current code actually does (empirically pinned)

1. **Semantics (a) is not implementable in the current code.** A bare
   `except: pass` in `calc_pcen_mobile` silently swallows contributions
   from any neighbor missing from the frame — so excluded-but-contributing
   degenerates, cell-for-cell, to fully-removed. The current no-RV pipeline
   therefore implements semantics (b) de facto
   (`test_excl_contributing_collapses_to_removed`). The silent exception
   swallowing is flagged for the Phase 3 bug audit.
2. **The barrier rule is global and asymmetric**, not pair-severing: a
   barrier-crossed settlement is deleted from everyone else's neighbor
   lists but keeps its own (in Oraculum: A counts E's services; E does not
   count A's). The manuscript describes severing the crossing only.
3. Also documented for completeness (full details in the spec): roads are
   neighbor-decayed in code though Eq. 4 has no neighbor term; `norm_psi`
   is a second normalization absent from Eq. 1; the popdensity denominator
   has no manuscript equation.

## Decisions this memo requests (none urgent; Phase 3/4 gates)

- Semantics (a) vs (b) for dropped settlement types — the code currently
  gives (b); choosing (a) requires a code fix.
- Whether the min–max universe should renormalize after drops (it does
  today) — the IND exhibit isolates exactly this effect.
- Whether the barrier asymmetry and the roads/norm_psi deviations should be
  fixed to match the manuscript, or ratified and written into the methods.
````

- [ ] **Step 3: Update CHANGELOG `[Unreleased]`**

Add under `## [Unreleased]`:

```markdown
### Added
- Mythical-city oracle ("Oraculum"): hand-verifiable fixtures, an
  independent Eq. 1–4 reference implementation, and pytest suites
  establishing production == reference == hand-arithmetic agreement
  (`tests/fixtures/oraculum/`, `tests/reference_impl.py`,
  `tests/test_oracle*.py`, `tests/test_fixture_invariants.py`,
  `tests/test_divergence_exhibit.py`)
- Oracle maps, derivation worksheet (ratification pending), and the
  exclusion-semantics memo for Raj (`docs/oracle/`)
- Empirically pinned manuscript-vs-code divergences: directed bbox
  adjacency, global asymmetric barrier rule, neighbor-decayed roads,
  code-only `norm_psi` and popdensity denominator, and exclusion
  semantics (a) degenerating to (b) via silent exception swallowing
  (flagged for Phase 3 bug audit)
```

- [ ] **Step 4: Full suite + grep checks**

Run: `uv run pytest -v 2>&1 | tail -15 && grep -c "RATIFICATION PENDING" docs/oracle/derivation-worksheet.md`
Expected: all tests pass (existing 14 + new suites); grep prints 1.

- [ ] **Step 5: Commit**

```bash
git add docs/oracle/derivation-worksheet.md docs/oracle/exclusion-semantics-memo.md CHANGELOG.md
git commit -m "docs(oracle): derivation worksheet (ratification pending), exclusion memo, changelog"
```

---

## Pipeline phases 5–7 (controller, not subagent tasks)

- Smoke/run: `uv run pytest` (full suite green).
- PR from `phase2-oracle` → merge per decision log; then on `main`: check
  off WORKPLAN Phase 2 items (worksheet ratification stays pending —
  that is by design, post-merge).
```
