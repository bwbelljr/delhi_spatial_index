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
# of them and `index.minmax` (which now has a hi == lo guard, DEL-54) raises
# instead of computing anything. C, RV and IND are the safe choices; that is
# not this test file's subject.
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


# --- the pipeline seam's own surface ----------------------------------
def _two_settlement_frame():
    return pd.DataFrame({
        "USO_AREA_U": ["A", "B"],
        "nbrs_bbox": [["B"], ["A"]],
        "nbrs_dist_bbox": [[("B", 1.0)], [("A", 1.0)]],
    })


def test_apply_exclusion_returns_only_the_universe_frame():
    """One exclusion filter, one return value.

    The second ('reported') frame it used to return had no caller —
    `index_frames` re-derives the same filter from `dropped` after amounts —
    so returning it invited two copies of the exclusion rule.
    """
    universe = pipeline.apply_exclusion(
        _two_settlement_frame(), dropped=frozenset({"B"}),
        stage="post_neighbors")

    # post_neighbors: the excluded row stays in the amounts universe and in
    # everyone's neighbour lists
    assert list(universe["USO_AREA_U"]) == ["A", "B"]
    assert list(universe.loc[universe["USO_AREA_U"] == "A", "nbrs_bbox"]) \
        == [["B"]]


def test_apply_exclusion_pre_neighbors_strips_the_excluded_ids():
    universe = pipeline.apply_exclusion(
        _two_settlement_frame(), dropped=frozenset({"B"}),
        stage="pre_neighbors")

    assert list(universe["USO_AREA_U"]) == ["A"]
    assert list(universe["nbrs_bbox"]) == [[]]
    assert list(universe["nbrs_dist_bbox"]) == [[]]


def test_build_neighbors_has_no_epsg_code_parameter():
    """It reprojects nothing — the caller hands it an already-projected
    frame — so the parameter was dead and misleading."""
    import inspect

    assert "epsg_code" not in inspect.signature(
        pipeline.build_neighbors).parameters


# --- 3B: the category mapping on the in-memory path (spec 3B §§ 3, 5) ---
def test_mapping_none_equals_an_explicit_identity_mapping():
    """`mapping=None` builds the identity over the types the city actually
    carries, so every existing in-memory caller keeps its call shape AND
    its numbers."""
    city = load_settlements()
    identity = {t: t for t in city["USO_FINAL"].unique()}
    implicit = _compute(city, None).set_index("USO_AREA_U")
    explicit = compute_frames(
        city, {"canal": load_barriers()}, load_services(), None,
        _methodology(), "pop", mapping=identity,
        scheme="explicit-identity").set_index("USO_AREA_U")

    assert set(implicit.index) == set(explicit.index)
    metrics = [c for c in implicit.columns
               if str(c).endswith(("_count", "_pcen", "_idx", "_length"))
               or c in ("unnorm_psi", "norm_psi", "population", "area_km2")]
    assert "unnorm_psi" in metrics and "clinic_pcen" in metrics
    for column in metrics:
        for sid in implicit.index:
            assert implicit.loc[sid, column] == pytest.approx(
                explicit.loc[sid, column], abs=1e-12), (column, sid)


def test_compute_frames_stamps_the_scheme_and_mapping_on_its_result():
    """pandas drops `attrs` across the merges inside index_frames, so the
    stamp has to go on the frame compute_frames actually returns."""
    mapping = {"Planned": "Planned", "UC": "UC", "JJC": "JJC", "RV": "RV",
               "RUAC": "RUAC", "IND": "IND"}
    result = compute_frames(
        load_settlements(), {"canal": load_barriers()}, load_services(),
        None, _methodology(), "pop", mapping=mapping, scheme="oracle-6")
    assert result.attrs["categories"] == {"scheme": "oracle-6",
                                          "mapping": mapping}


def test_compute_frames_defaults_the_scheme_to_identity():
    result = _compute(load_settlements(), None)
    assert result.attrs["categories"]["scheme"] == "identity"
    assert result.attrs["categories"]["mapping"] == {
        "Planned": "Planned", "UC": "UC", "JJC": "JJC", "RV": "RV",
        "RUAC": "RUAC", "IND": "IND"}


def test_exclusion_type_outside_the_mapping_raises_on_the_in_memory_path():
    """The LOAD-time check cannot see this call: compute_frames and the
    oracle helpers build a MethodologyConfig directly and never pass through
    load_config. Without the run-time repeat, a category the mapping no
    longer produces would exclude nothing — silently."""
    city = load_settlements()
    methodology = methodology_with(PROFILE, types=("non-urban",),
                                   stage="post_neighbors")
    with pytest.raises(validate.ValidationError) as excinfo:
        compute_frames(city, {"canal": load_barriers()}, load_services(),
                       None, methodology, "pop",
                       mapping={t: t for t in city["USO_FINAL"].unique()})
    message = str(excinfo.value)
    assert "non-urban" in message
    assert "RV" in message, "the message lists the categories the mapping produces"


# --- 3E: the compute-local weight column (spec § 2.3) ------------------
def test_the_stamp_records_the_buffer():
    """The buffer SHAPES the stored lists — a barrier 3 m off an edge blocks
    at 5 m and not at 2 m — so an artifact built at another buffer must be
    refused. It is None for the two rules that have no buffer."""
    from dataclasses import replace

    from delhi_psi.config import BarrierConfig, BarrierRule
    from tests.oraculum_fixtures import oracle_config

    cfg = oracle_config("code-2025")
    assert pipeline.methodology_stamp(cfg.methodology)["barrier"] == {
        "rule": "global_asymmetric", "combine": "any", "buffer_m": None}
    partial = replace(cfg.methodology, barrier=BarrierConfig(
        rule=BarrierRule.PARTIAL_WEIGHTED, combine="any", buffer_m=5.0))
    assert pipeline.methodology_stamp(partial)["barrier"] == {
        "rule": "partial_weighted", "combine": "any", "buffer_m": 5.0}


def test_the_weight_column_is_dropped_before_index_frames_returns():
    """Like NBRS_DIST_BOUNDARY_COL: compute-local, so the CSV/shapefile
    column set is identical under every barrier rule."""
    from dataclasses import replace

    from delhi_psi.config import BarrierConfig, BarrierRule
    from delhi_psi.pipeline import compute_frames
    from tests.cities import ORACULUM
    from tests.oraculum_fixtures import methodology_with

    methodology = methodology_with("code-2025", types=(), stage=None)
    methodology = replace(methodology, barrier=BarrierConfig(
        rule=BarrierRule.PARTIAL_WEIGHTED, combine="any", buffer_m=5.0))
    got = compute_frames(ORACULUM.load_settlements(),
                         {"canal": ORACULUM.load_barriers()},
                         ORACULUM.load_services(), None, methodology, "pop",
                         mapping=ORACULUM.mapping(), scheme=ORACULUM.scheme)
    assert pipeline.NBRS_WEIGHT_COL not in got.columns

    baseline = compute_frames(ORACULUM.load_settlements(),
                              {"canal": ORACULUM.load_barriers()},
                              ORACULUM.load_services(), None,
                              methodology_with("code-2025", types=(),
                                               stage=None),
                              "pop", mapping=ORACULUM.mapping(),
                              scheme=ORACULUM.scheme)
    assert list(got.columns) == list(baseline.columns)


def test_the_weight_column_is_in_the_shapefile_drop_list():
    """`compute` cuts missing_population.csv from the NEIGHBOURS frame,
    where the column IS present, using this list."""
    from delhi_psi import io

    assert pipeline.NBRS_WEIGHT_COL in io.SHAPEFILE_DROP_COLUMNS


def test_pre_neighbours_exclusion_strips_the_weight_column_too():
    """An id removed from nbrs_bbox must leave the weight list as well, or
    pcen's KeyError guard fires on a legitimate run."""
    import geopandas as gpd
    from shapely.geometry import box

    frame = gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B", "C"],
         "nbrs_bbox": [["B", "C"], ["A"], ["A"]],
         "nbrs_dist_bbox": [[("B", 1.0), ("C", 2.0)], [("A", 1.0)],
                            [("A", 2.0)]],
         pipeline.NBRS_WEIGHT_COL: [[("B", 0.5), ("C", 1.0)], [("A", 0.5)],
                                    [("A", 1.0)]]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:7760")
    got = pipeline.apply_exclusion(frame, dropped={"C"},
                                   stage="pre_neighbors")
    row = got[got["USO_AREA_U"] == "A"].iloc[0]
    assert row["nbrs_bbox"] == ["B"]
    assert row[pipeline.NBRS_WEIGHT_COL] == [("B", 0.5)]


# --- 3E: overlap lending is wired per service (spec § 3.2) -------------
def _messy_overlap_frames(lending):
    """The messy city scored through compute_frames with ONE methodology
    value changed, both denominators' worth of plumbing in one call."""
    from dataclasses import replace

    from delhi_psi.config import OverlapConfig, OverlapLending
    from delhi_psi.pipeline import compute_frames
    from tests.cities import MESSY
    from tests.oraculum_fixtures import methodology_with

    methodology = methodology_with("code-2025", types=(), stage=None,
                                   city=MESSY)
    methodology = replace(methodology, overlap=OverlapConfig(
        lending=OverlapLending(lending)))
    return compute_frames(MESSY.load_settlements(),
                          {"canal": MESSY.load_barriers()},
                          MESSY.load_services(), None, methodology, "pop",
                          mapping=MESSY.mapping(),
                          scheme=MESSY.scheme).set_index("USO_AREA_U")


def test_index_frames_builds_one_shared_structure_per_service():
    """The clinic moves and the school does not, in ONE run: a single dict
    reused across services would move both, and a dict built for the wrong
    service would move the wrong one."""
    whole = _messy_overlap_frames("whole")
    outside = _messy_overlap_frames("outside_receiver")
    assert outside.loc["O1", "clinic_pcen"] == pytest.approx(
        1 / 600, abs=1e-12)
    assert whole.loc["O1", "clinic_pcen"] == pytest.approx(
        (1 + 1 / 1.8) / 600, abs=1e-12)
    assert outside.loc["O1", "school_pcen"] == whole.loc["O1", "school_pcen"]
    assert outside.loc["O1", "police_pcen"] == whole.loc["O1", "police_pcen"]


def test_the_overlap_rule_adds_no_column_and_no_stamp_entry():
    """It is applied downstream in `compute`, like decay and roads, so one
    stored artifact serves both values and the output column set is
    lending-independent."""
    from dataclasses import replace

    from delhi_psi.config import OverlapConfig, OverlapLending
    from tests.cities import MESSY
    from tests.oraculum_fixtures import methodology_with

    assert list(_messy_overlap_frames("outside_receiver").columns) == \
        list(_messy_overlap_frames("whole").columns)
    methodology = methodology_with("code-2025", types=(), stage=None,
                                   city=MESSY)
    outside = replace(methodology, overlap=OverlapConfig(
        lending=OverlapLending.OUTSIDE_RECEIVER))
    assert pipeline.methodology_stamp(outside) == \
        pipeline.methodology_stamp(methodology)


def test_no_shared_structure_is_built_under_whole(monkeypatch):
    """`whole` must not pay for a rule it does not use, and must not go near
    the arithmetic: the builder is never called AT ALL, which is the only
    way to state "bit-identical" as a test rather than as a comparison
    against a number that would be equal by construction."""
    from delhi_psi import index

    def refuse(*args, **kwargs):
        raise AssertionError("shared_amounts must not be called under whole")

    monkeypatch.setattr(index, "shared_amounts", refuse)
    got = _messy_overlap_frames("whole")
    assert got.loc["O1", "clinic_pcen"] == pytest.approx(
        (1 + 1 / 1.8) / 600, abs=1e-12)
