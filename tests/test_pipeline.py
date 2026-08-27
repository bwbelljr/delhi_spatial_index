"""`compute_frames`' population path — the branch the oracle never takes.

Every oracle test passes `population=None` (the Oraculum city carries its own
column), so `attach_population`'s merge branch, the `missing_population` rule
and `max_missing_population` are pinned here instead. No number is
re-derived: the merge is proven by feeding the SAME population values through
the file path and requiring the oracle city's answers back.
"""

import geopandas as gpd
import pandas as pd
import pytest

from delhi_psi import pipeline, validate
from delhi_psi.config import load_config
from delhi_psi.pipeline import attach_population, compute_frames
from tests.oraculum_fixtures import (
    load_barriers, load_services, load_settlements, methodology_with,
)

PROFILE = "code-2025"

# The settlement left out of the population table. Must be one whose removal
# leaves every service's pcen column non-constant: bank/police/ration/
# transport are singletons in this city, so dropping A, B, D or E flattens one
# of them and `index.minmax` (deliberately without a hi == lo guard, verbatim
# `calc_service_index`) divides 0/0. C, RV and IND are the safe choices; that
# pre-existing divergence is not this test file's subject.
MISSING_ID = "C"


def _methodology():
    """The profile's methodology with no type exclusion, so the only rows
    that can leave the frame are the ones the population rule drops."""
    return methodology_with(PROFILE, types=(), stage="post_neighbors")


def _city_and_population(*, omit=()):
    """The oracle city with its `population` column moved into a separate
    table, as the real pipeline sees it (the shapefile has no population
    field). `omit` leaves those ids out of the table entirely."""
    city = load_settlements()
    table = pd.DataFrame({
        "uso_area_u": list(city["USO_AREA_U"]),
        "population": list(city["population"]),
    })
    table = table[~table["uso_area_u"].isin(list(omit))].reset_index(drop=True)
    return city.drop(columns=["population"]), table


def _compute(city, population, **kwargs):
    return compute_frames(city, {"canal": load_barriers()}, load_services(),
                          population, _methodology(), "pop", **kwargs)


# --- attach_population, directly ---------------------------------------
def test_merge_lands_population_on_the_right_ids():
    city, table = _city_and_population()
    merged, missing = attach_population(city, table)

    assert missing == frozenset()
    expected = load_settlements().set_index("USO_AREA_U")["population"]
    got = merged.set_index("USO_AREA_U")["population"]
    assert set(got.index) == set(expected.index)
    for sid in expected.index:
        assert got[sid] == expected[sid], sid


def test_merge_reports_ids_with_no_population_row():
    city, table = _city_and_population(omit=(MISSING_ID,))
    merged, missing = attach_population(city, table)

    assert missing == frozenset({MISSING_ID})
    row = merged.set_index("USO_AREA_U").loc[MISSING_ID]
    assert pd.isna(row["population"])
    # the join key is consumed, not left behind as a column
    assert "uso_area_u" not in merged.columns
    assert isinstance(merged, gpd.GeoDataFrame)


def test_merge_honours_custom_column_names():
    city, table = _city_and_population()
    renamed = table.rename(columns={"uso_area_u": "colony_id",
                                    "population": "pop_2020"})
    merged, missing = attach_population(
        city, renamed, population_id_col="colony_id",
        population_value_col="pop_2020")

    assert missing == frozenset()
    assert list(merged["population"]) == list(load_settlements()["population"])


def test_population_frame_is_not_mutated_by_the_merge():
    city, table = _city_and_population()
    before = table.copy()
    attach_population(city, table)
    pd.testing.assert_frame_equal(table, before)


# --- the merge produces the oracle's numbers ---------------------------
def test_population_file_path_matches_the_carried_column():
    """Same population values, delivered two ways -> identical numbers."""
    city, table = _city_and_population()
    from_file = _compute(city, table).set_index("USO_AREA_U")
    carried = _compute(load_settlements(), None).set_index("USO_AREA_U")

    assert set(from_file.index) == set(carried.index)
    for col in ("clinic_pcen", "school_pcen", "road_pcen", "unnorm_psi",
                "norm_psi"):
        for sid in carried.index:
            assert from_file.loc[sid, col] == pytest.approx(
                carried.loc[sid, col], abs=1e-12), (col, sid)


# --- missing_population and max_missing_population ---------------------
def test_missing_population_drop_removes_only_the_unpriced_rows():
    city, table = _city_and_population(omit=(MISSING_ID,))
    got = _compute(city, table).set_index("USO_AREA_U")

    assert MISSING_ID not in got.index
    assert set(got.index) == set(load_settlements()["USO_AREA_U"]) - {MISSING_ID}


def test_missing_population_error_raises_naming_the_id():
    city, table = _city_and_population(omit=(MISSING_ID,))
    with pytest.raises(validate.ValidationError) as excinfo:
        _compute(city, table, missing_population="error")
    message = str(excinfo.value)
    assert MISSING_ID in message
    assert "layers.population.missing" in message


def test_missing_population_error_is_silent_when_nothing_is_missing():
    city, table = _city_and_population()
    got = _compute(city, table, missing_population="error")
    assert len(got) == len(load_settlements())


def test_max_missing_population_below_the_count_raises():
    city, table = _city_and_population(omit=(MISSING_ID,))
    with pytest.raises(validate.ValidationError) as excinfo:
        _compute(city, table, max_missing_population=0)
    assert "max_missing_population" in str(excinfo.value)


def test_max_missing_population_at_the_count_passes():
    city, table = _city_and_population(omit=(MISSING_ID,))
    got = _compute(city, table, max_missing_population=1)
    assert MISSING_ID not in set(got["USO_AREA_U"])
    assert len(got) == len(load_settlements()) - 1


# --- the knob is validated like every other one ------------------------
def test_unknown_missing_population_value_is_rejected():
    city, table = _city_and_population(omit=(MISSING_ID,))
    with pytest.raises(ValueError) as excinfo:
        _compute(city, table, missing_population="dropp")
    message = str(excinfo.value)
    assert "'dropp'" in message
    assert "['drop', 'error']" in message


def test_unknown_missing_population_value_is_rejected_before_any_work():
    """The check is up front, so a typo fails even on an unusable city — it
    can never silently behave like 'drop'."""
    with pytest.raises(ValueError, match="unknown missing_population"):
        compute_frames(None, None, None, None, None, "pop",
                       missing_population="")


def test_every_config_value_for_the_knob_is_one_compute_frames_accepts():
    """The shipped profile's layers.population.missing must be a value
    compute_frames handles — the config loader and this knob share one
    allowed set."""
    assert load_config(PROFILE).layers.population.missing \
        in pipeline.MISSING_POPULATION
