"""The two fixture cities and the compat views over them (spec 3C § 3).

`tests/cities.py` is fixture plumbing: it says where a city's files are, what
its vocabulary is, and which scenarios it is scored under. It must never
reach into `delhi_psi` (the reference implementation imports it, and the
INDEPENDENCE RULE forbids the reference from seeing production code) nor into
`tests.reference_impl` (which imports it).

The scenario pin below is the one that keeps the two sides honest: the
reference drops settlements by ID, production excludes them by CATEGORY and
then unions the rows with no population. This asserts the two agree, per city
and per scenario, instead of assuming it.
"""
from pathlib import Path

import pytest

from delhi_psi.categories import apply_mapping
from delhi_psi.pipeline import attach_population, excluded_ids
from tests.cities import CITIES, FIXTURES_ROOT, MESSY, ORACULUM

# Task 3 (scripts/generate_messy_fixtures.py) widens this to CITIES. Until the
# messy GeoJSON files exist, only Oraculum can be loaded from disk.
FIXTURED = (ORACULUM,)
SCENARIO_CASES = [(city, scenario) for city in FIXTURED
                  for scenario in city.scenarios]


def case_id(case):
    city, scenario = case
    return f"{city.name}-{scenario.name}"


DECLARED = {
    "oraculum": ("oracle-6", ("Planned", "UC", "JJC", "RV", "RUAC", "IND")),
    "messy": ("messy-2", ("Planned", "RV")),
}


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_scheme_vocabulary_and_epsg(city):
    scheme, vocabulary = DECLARED[city.name]
    assert city.scheme == scheme
    assert city.vocabulary == vocabulary
    assert city.epsg == 7760
    assert len(city.vocabulary) == len(set(city.vocabulary)), "duplicate type"


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_mapping_is_the_identity_over_the_vocabulary(city):
    assert city.mapping() == {t: t for t in city.vocabulary}


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_fixtures_path_is_tests_fixtures_slash_name(city):
    """`fixtures` is a real field (spec § 3), so pin it against `name` — the
    two must not drift."""
    assert city.fixtures == FIXTURES_ROOT / city.name
    assert FIXTURES_ROOT == Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_scenario_names_are_unique(city):
    names = [scenario.name for scenario in city.scenarios]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_stage_agrees_with_dropped_before_neighbors(city):
    """One set, one flag: production applies ONE `stage` to `excluded ∪
    missing`, so the reference's `dropped_before_neighbors` is exactly
    `stage == pre_neighbors` (spec § 3, rev 3)."""
    for scenario in city.scenarios:
        assert scenario.stage in ("post_neighbors", "pre_neighbors"), scenario
        assert scenario.dropped_before_neighbors is (
            scenario.stage == "pre_neighbors"), scenario.name


def test_oraculum_scenarios_are_todays_reference_table_in_order():
    """Order wins: it fixes expected_values.csv's row order, which is
    round-trip tested byte for byte."""
    assert [(s.name, set(s.dropped), s.dropped_before_neighbors,
             s.exclusion_types, s.stage) for s in ORACULUM.scenarios] == [
        ("baseline", set(), False, (), "post_neighbors"),
        ("excl_contributing", {"RV", "IND"}, False, ("RV", "IND"),
         "post_neighbors"),
        ("excl_removed", {"RV", "IND"}, True, ("RV", "IND"), "pre_neighbors"),
        ("excl_ind_removed", {"IND"}, True, ("IND",), "pre_neighbors"),
        ("excl_rv_only", {"RV"}, False, ("RV",), "post_neighbors"),
    ]


def test_messy_scenarios_are_the_spec_table():
    """Every messy scenario drops `U` with the scenario's OWN flag, because
    production drops a no-population id unconditionally and applies its single
    `stage` to the whole drop set (spec § 3, rev 3)."""
    assert [(s.name, set(s.dropped), s.dropped_before_neighbors,
             s.exclusion_types, s.stage) for s in MESSY.scenarios] == [
        ("nopop_only", {"U"}, False, (), "post_neighbors"),
        ("excl_rv_post", {"U", "N"}, False, ("RV",), "post_neighbors"),
        ("excl_rv_pre", {"U", "N"}, True, ("RV",), "pre_neighbors"),
    ]
    assert MESSY.scenarios[0].dropped != MESSY.scenarios[1].dropped, \
        "nopop_only and excl_rv_post must differ, or category exclusion is " \
        "never exercised on this city"


def test_cities_module_imports_no_production_code():
    """INDEPENDENCE RULE: tests/reference_impl.py imports this module, so this
    module must not reach production code — nor back into the reference."""
    source = (Path(__file__).resolve().parent / "cities.py").read_text()
    assert "delhi_psi" not in source
    assert "reference_impl" not in source


@pytest.mark.parametrize("city", FIXTURED, ids=lambda c: c.name)
def test_every_layer_loads(city):
    settlements = city.load_settlements()
    assert len(settlements) > 0
    assert set(settlements.columns) >= {"USO_AREA_U", "USO_FINAL",
                                        "population", "area_km2", "geometry"}
    assert settlements.crs.to_epsg() == city.epsg
    assert city.load_barriers().crs.to_epsg() == city.epsg
    services = city.load_services()
    assert set(services) >= {"clinic", "school", "bank", "police", "ration",
                             "transport", "road"}


@pytest.mark.parametrize("city", FIXTURED, ids=lambda c: c.name)
def test_vocabulary_is_exactly_the_types_the_layer_carries(city):
    """No more (it would hide an unmapped type), no fewer (the run errors)."""
    assert set(city.load_settlements()["USO_FINAL"]) == set(city.vocabulary)


@pytest.mark.parametrize("city", FIXTURED, ids=lambda c: c.name)
def test_there_is_at_least_one_road_row(city):
    assert len(city.load_services()["road"]) >= 1


@pytest.mark.parametrize("case", SCENARIO_CASES, ids=case_id)
def test_dropped_is_excluded_ids_union_missing(case):
    """THE agreement pin: the reference's id-based `dropped` is exactly
    production's `excluded_ids(types) ∪ missing` for this city."""
    city, scenario = case
    frame, missing = attach_population(city.load_settlements(), None)
    frame = apply_mapping(frame, type_col="USO_FINAL", mapping=city.mapping())
    excluded = excluded_ids(frame, types=scenario.exclusion_types)
    assert excluded | missing == scenario.dropped, (
        city.name, scenario.name, sorted(excluded), sorted(missing))
    # ... and every id the scenario names is really a settlement of this city.
    assert scenario.dropped <= set(frame["USO_AREA_U"]), scenario.name


# --- 3C: the backward-compatible module views (spec § 3) ---------------
def test_reference_scenarios_view_is_oraculums_table():
    """`reference_impl.SCENARIOS` keeps the 2-tuple shape consumed today,
    with Oraculum's order — which fixes expected_values.csv's row order."""
    from tests.reference_impl import SCENARIOS

    assert SCENARIOS == {s.name: (s.dropped, s.dropped_before_neighbors)
                         for s in ORACULUM.scenarios}
    assert list(SCENARIOS) == [s.name for s in ORACULUM.scenarios]


def test_oracle_scenarios_view_is_the_three_tuple_of_oraculums_table():
    from tests.oraculum_fixtures import ORACLE_SCENARIOS

    assert ORACLE_SCENARIOS == [(s.name, s.exclusion_types, s.stage)
                                for s in ORACULUM.scenarios]


def test_oracle_scheme_and_vocabulary_are_oraculum_aliases():
    from tests.oraculum_fixtures import (
        ORACLE_SCHEME, ORACLE_VOCABULARY, oracle_mapping,
    )

    assert ORACLE_SCHEME == ORACULUM.scheme == "oracle-6"
    assert ORACLE_VOCABULARY == ORACULUM.vocabulary
    assert oracle_mapping() == ORACULUM.mapping()


def test_render_oracle_maps_does_not_mutate_the_reference_scenario_table():
    """Importing the map script must not widen reference_impl.SCENARIOS: the
    fixture CSV is round-trip tested at its current row count, and a
    setdefault at import time would add a sixth scenario to every later
    emit in the same process."""
    from tests.reference_impl import SCENARIOS

    before = dict(SCENARIOS)
    from scripts.render_oracle_maps import MAP_SCENARIOS

    assert dict(SCENARIOS) == before
    assert "rv_removed" not in SCENARIOS
    assert MAP_SCENARIOS["rv_removed"] == (frozenset({"RV"}), True)
    assert all(MAP_SCENARIOS[name] == value
               for name, value in SCENARIOS.items())
