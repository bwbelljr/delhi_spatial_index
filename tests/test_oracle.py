"""Production code vs the oracle's expected values (rule=code rows).

Every comparison here is production-vs-hand-anchored-reference. A failure
means production behavior changed (or the oracle is wrong) — investigate,
never blindly update the CSV (oracle contract: docs/superpowers/specs/
2026-08-17-phase2-oracle-design.md).

Scenarios are expressed as the `code-2025` profile plus `exclusion.types` and
`exclusion.stage` overrides ONLY; `absent_neighbor` comes from the profile
(`swallowed` for code-2025), because in the reference it is a rule-set
property, not a scenario property (spec § 7).
"""

from pathlib import Path

import pandas as pd
import pytest

from delhi_psi.config import load_config
from tests.oraculum_fixtures import (
    ORACLE_SCENARIOS, compute_oracle_frame, load_barriers, load_services,
    load_settlements, run_production_chain,
)

CSV = Path(__file__).resolve().parent / "fixtures" / "oraculum" / "expected_values.csv"
PROFILE = "code-2025"

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


def _frame(denom, *, types=(), stage="post_neighbors"):
    return compute_oracle_frame(PROFILE, types=types, stage=stage, denom=denom)


def _compared_metrics():
    """The metric set is derived from the profile: norm_psi only when the
    profile turns the second normalization on."""
    cfg = load_config(PROFILE)
    if cfg.methodology.second_normalization:
        return METRICS
    return {k: v for k, v in METRICS.items() if k != "norm_psi"}


@pytest.mark.parametrize("denom", ["pop", "popdensity"])
@pytest.mark.parametrize("scenario,types,stage", ORACLE_SCENARIOS)
def test_production_matches_code_rows(expected, scenario, types, stage, denom):
    exp = _expected_frame(expected, scenario, denom)
    got = _frame(denom, types=types, stage=stage)
    assert set(got.index) == set(exp.index)
    for prod_col, metric in _compared_metrics().items():
        for sid in exp.index:
            assert got.loc[sid, prod_col] == pytest.approx(
                exp.loc[sid, metric], abs=1e-12), (scenario, denom, sid, prod_col)


def test_zero_service_settlement(expected):
    got = _frame("pop")
    assert got.loc["C", "clinic_count"] == 0
    assert got.loc["C", "clinic_pcen"] > 0  # entirely from decayed neighbors


def test_second_order_neighbor_excluded(expected):
    got = _frame("pop")
    assert "A" not in set(got.loc["C", "nbrs_bbox"])


def test_barrier_rule_is_global_and_directed(expected):
    got = _frame("pop")
    assert "A" not in set(got.loc["B", "nbrs_bbox"])   # A stripped from B
    assert set(got.loc["A", "nbrs_bbox"]) == {"B", "E"}  # A keeps its own


def test_popdensity_differs_from_popsize(expected):
    pop = _frame("pop")
    dens = _frame("popdensity")
    assert pop.loc["E", "clinic_pcen"] != pytest.approx(
        dens.loc["E", "clinic_pcen"], abs=1e-15)


def test_road_decay_divergence(expected):
    """Code roads are decayed; Eq. 4 has no neighbor term (rule-set gap #3)."""
    got = _frame("pop")
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
    got = _frame("pop")
    assert got["norm_psi"].min() == pytest.approx(0.0, abs=1e-12)
    assert got["norm_psi"].max() == pytest.approx(1.0, abs=1e-12)
    assert not got["unnorm_psi"].equals(got["norm_psi"])


def test_minmax_anchors_unique(expected):
    got = _frame("pop")
    for svc in ("clinic", "school"):
        pcen = got[f"{svc}_pcen"]
        assert (pcen == pcen.max()).sum() == 1, svc
        assert (pcen == pcen.min()).sum() == 1, svc


@pytest.mark.parametrize("denom", ["pop", "popdensity"])
def test_production_collapse_gap5(expected, denom):
    """Rule-set gap #5, pinned against PRODUCTION: dropping rows after
    neighbor computation (absent_neighbor=swallowed drops the missing
    contributions) equals dropping them before — semantics (a) degenerates to
    (b) in the real code, not just in the reference impl's model of it."""
    post = _frame(denom, types=("RV", "IND"), stage="post_neighbors")
    pre = _frame(denom, types=("RV", "IND"), stage="pre_neighbors")
    assert set(post.index) == set(pre.index)
    for col in [c for c in post.columns
                if c.endswith(("_pcen", "_idx")) or c in ("unnorm_psi",
                                                          "norm_psi")]:
        for sid in post.index:
            assert post.loc[sid, col] == pytest.approx(
                pre.loc[sid, col], abs=1e-12), (denom, sid, col)


def test_gap6_border_point_is_double_counted_by_production():
    """Rule-set gap #6 (found by code-review round 2 mutation testing).

    Production's point counting uses gpd.sjoin's default `intersects`
    predicate, so a service point lying exactly on a shared settlement border
    is counted for BOTH neighbors. The manuscript's per-settlement counts say
    only "within an administrative unit" and are silent on the boundary case;
    the reference impl resolves that as strict containment, counting it for
    neither.

    Measured against the real Delhi layers (Aug 2026), this gap is LATENT:
    zero service points lie exactly on a colony boundary in any of the six
    point layers (closest approach 1.3 mm). The real double-counting today
    comes from a different mechanism — 4,050 overlapping colony polygon
    pairs put ~450 service points inside two or more colonies, which
    `within` would not fix. Both are routed to the Phase 3 bug audit.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    from delhi_psi import index
    from tests.reference_impl import _service_amounts

    city = load_settlements()
    # (1_001_000, 1_001_500) lies exactly on the A|B shared edge
    border_point = gpd.GeoDataFrame(
        {"service": ["clinic"]},
        geometry=[Point(1_001_000, 1_001_500)], crs=city.crs)

    counted = index.point_counts(city.copy(), border_point,
                                 count_col="probe_count")
    counts = counted.set_index("USO_AREA_U")["probe_count"]
    assert counts["A"] == 1 and counts["B"] == 1, "production double-counts"

    ref = _service_amounts(city, {"clinic": border_point,
                                  "road": load_services()["road"]})
    assert ref["clinic"]["A"] == 0 and ref["clinic"]["B"] == 0, \
        "reference impl (manuscript-literal `within`) counts it for neither"


def test_reprojection_is_load_bearing():
    """Feed a service layer in a different CRS and require the same answers.

    Code review round 2: every fixture is already EPSG:7760, so reprojection
    could be replaced by the identity function with a green suite — yet every
    real service layer depends on it.
    """
    from delhi_psi.pipeline import compute_frames

    services = load_services()
    services_wgs84 = dict(services)
    services_wgs84["clinic"] = services["clinic"].to_crs(epsg=4326)
    assert services_wgs84["clinic"].crs.to_epsg() == 4326

    from tests.oraculum_fixtures import methodology_with
    methodology = methodology_with(PROFILE, types=(), stage="post_neighbors")
    baseline = compute_frames(load_settlements(), {"canal": load_barriers()},
                              services, None, methodology, "pop")
    reprojected = compute_frames(load_settlements(),
                                 {"canal": load_barriers()},
                                 services_wgs84, None, methodology, "pop")
    for sid in baseline["USO_AREA_U"]:
        got = reprojected[reprojected["USO_AREA_U"] == sid]["clinic_pcen"].iloc[0]
        exp = baseline[baseline["USO_AREA_U"] == sid]["clinic_pcen"].iloc[0]
        assert got == pytest.approx(exp, abs=1e-12), sid


# --- step-0 snapshot backend (retired in migration step 5) -------------
# `scripts/generate_production_fixtures.py` still generates
# tests/fixtures/oraculum/production/code-2025.csv through these two, so the
# snapshot keeps being produced by the SAME wiring it was created with. They
# are deleted in the task that swaps the generator to compute_frames and
# proves a no-op diff. Do not delete them earlier.
SCENARIO_WIRING = [
    # (scenario, drop_pre, drop_post)
    ("baseline", frozenset(), frozenset()),
    ("excl_rv_only", frozenset(), frozenset({"RV"})),
    ("excl_contributing", frozenset(), frozenset({"RV", "IND"})),
    ("excl_removed", frozenset({"RV", "IND"}), frozenset()),
    ("excl_ind_removed", frozenset({"IND"}), frozenset()),
]


def _production_frame(denom, drop_ids_post=frozenset(),
                      drop_ids_pre=frozenset()):
    city = load_settlements()
    if drop_ids_pre:
        city = city[~city["USO_AREA_U"].isin(drop_ids_pre)]
    result = run_production_chain(
        city, load_barriers(), load_services(), denom,
        drop_ids_post=drop_ids_post)
    return result.set_index("USO_AREA_U")
