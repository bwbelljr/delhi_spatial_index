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
