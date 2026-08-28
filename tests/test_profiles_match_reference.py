"""Every shipped profile reproduces the reference at its mapped knobs.

Called through compute_frames with an explicit `denominator=`, for BOTH
reference denominators, independent of the profile's outputs.denominators
(spec § 7). The mapping test reads the SAME table the config enums are
generated from, so a value with no reference knob cannot be added silently.
"""
import pandas as pd
import pytest

from delhi_psi.config import ENUMS, ENUM_KEYS, REFERENCE_KNOBS, load_config
from tests.cities import CITIES
from tests.oraculum_fixtures import (
    ORACLE_SCENARIOS, compute_oracle_frame, load_barriers, load_services,
    load_settlements,
)
from tests.reference_impl import RULESETS, compute_city

REFERENCE_DENOMS = ("pop", "popdensity")

# profile -> the reference rule-set whose rows it must reproduce (spec § 4)
PROFILE_RULES = {"code-2025": "code", "manuscript": "ideal"}

# production column -> reference metric
METRIC_MAP = {
    "clinic_pcen": "clinic_pcen", "clinic_idx": "clinic_idx",
    "school_pcen": "school_pcen", "school_idx": "school_idx",
    "bank_pcen": "bank_pcen", "bank_idx": "bank_idx",
    "police_pcen": "police_pcen", "police_idx": "police_idx",
    "ration_pcen": "ration_pcen", "ration_idx": "ration_idx",
    "transport_pcen": "transport_pcen", "transport_idx": "transport_idx",
    "road_pcen": "road_pcen", "road_idx": "road_idx",
    "road_length": "road_length_km",
    "unnorm_psi": "psi_eq1", "norm_psi": "norm_psi",
}

# (city, scenario) — each city brings its OWN scenario table (spec § 6).
CASES = [(city, scenario) for city in CITIES for scenario in city.scenarios]


def case_id(case):
    city, scenario = case
    return f"{city.name}-{scenario.name}"


@pytest.fixture(scope="module")
def expected():
    return {city.name: pd.read_csv(city.fixtures / "expected_values.csv")
            for city in CITIES}


def reference_block(expected, rule, scenario, denom):
    sub = expected[(expected["rule"] == rule)
                   & (expected["scenario"] == scenario)
                   & (expected["denom"] == denom)]
    return sub.pivot(index="settlement", columns="metric", values="value")


def metrics_for(profile):
    cfg = load_config(profile)
    skip = set() if cfg.methodology.second_normalization else {"norm_psi"}
    return {k: v for k, v in METRIC_MAP.items() if k not in skip}


@pytest.mark.parametrize("denom", REFERENCE_DENOMS)
@pytest.mark.parametrize("case", CASES, ids=case_id)
@pytest.mark.parametrize("profile", sorted(PROFILE_RULES))
def test_profile_matches_reference(expected, profile, case, denom):
    city, scenario = case
    exp = reference_block(expected[city.name], PROFILE_RULES[profile],
                          scenario.name, denom)
    got = compute_oracle_frame(profile, types=scenario.exclusion_types,
                               stage=scenario.stage, denom=denom, city=city)
    assert set(got.index) == set(exp.index)
    for prod_col, metric in metrics_for(profile).items():
        for sid in exp.index:
            assert got.loc[sid, prod_col] == pytest.approx(
                exp.loc[sid, metric], abs=1e-12), (city.name, profile,
                                                   scenario.name, denom, sid,
                                                   prod_col)


def test_enums_cover_exactly_the_reference_table():
    assert set(ENUMS) == set(ENUM_KEYS)
    for key in ENUM_KEYS:
        assert {m.value for m in ENUMS[key]} == set(REFERENCE_KNOBS[key]), key


def test_every_mapped_knob_is_one_the_reference_actually_implements():
    """Drive compute_city once per mapped knob value; an unimplemented knob
    raises ValueError inside the reference, so this fails loudly."""
    city, barriers, services = (load_settlements(), load_barriers(),
                                load_services())
    base = dict(RULESETS["code"], scenario="baseline", denom="pop")
    knob_for_key = {
        "methodology.adjacency.rule": "adjacency_rule",
        "methodology.barrier.rule": "barrier_rule",
        "methodology.roads": "roads_formula",
        "methodology.second_normalization": "second_norm",
        "methodology.exclusion.absent_neighbor": "absent_neighbor_contribution",
        "outputs.denominators[]": "denom",
    }
    for key, knob in knob_for_key.items():
        for config_value, reference_value in REFERENCE_KNOBS[key].items():
            kwargs = dict(base)
            kwargs[knob] = reference_value
            frame = compute_city(city, services, barriers, **kwargs)
            assert len(frame) == 7, (key, config_value)


def test_exclusion_stage_maps_onto_dropped_before_neighbors():
    """`stage` has no compute_city keyword — it selects the SCENARIO, whose
    second element is `dropped_before_neighbors` (spec § 3 table)."""
    from tests.reference_impl import SCENARIOS

    stage_of = REFERENCE_KNOBS["methodology.exclusion.stage"]
    for scenario, types, stage in ORACLE_SCENARIOS:
        _, drop_before = SCENARIOS[scenario]
        assert drop_before is stage_of[stage], scenario
