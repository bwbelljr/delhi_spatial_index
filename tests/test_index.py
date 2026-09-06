"""delhi_psi.index — counts, lengths, PCEN (Eq. 3), min-max (Eq. 2), PSI (Eq. 1).

The exclusion axes are tested here directly, because this is where DEL-21's
`except: pass` becomes an explicit lookup.
"""
import math
import sys

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from delhi_psi import index
from tests.oraculum_fixtures import load_services, load_settlements


def city_with_neighbours():
    """Two settlements, one clinic each in X, distance 1 km -> decay 1/2."""
    gdf = gpd.GeoDataFrame(
        {"USO_AREA_U": ["X", "Y"], "population": [100.0, 200.0],
         "area_km2": [1.0, 2.0],
         "nbrs_dist_bbox": [[("Y", 1.0)], [("X", 1.0)]],
         "clinic_count": [2.0, 0.0]},
        geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)],
        crs="EPSG:7760")
    return gdf


def test_point_counts_uses_intersects_and_fills_zero():
    city = load_settlements()
    counted = index.point_counts(city, load_services()["clinic"],
                                 count_col="clinic_count")
    counts = counted.set_index("USO_AREA_U")["clinic_count"]
    assert counts["A"] == 2 and counts["B"] == 1 and counts["C"] == 0
    assert counts.dtype.kind == "i"


def test_road_lengths_are_kilometres():
    city = load_settlements()
    lengths = index.road_lengths(city.copy(), load_services()["road"],
                                 length_col="road_length")
    values = lengths.set_index("USO_AREA_U")["road_length"]
    assert values["A"] == pytest.approx(0.75, abs=1e-12)
    assert values["E"] == pytest.approx(0.75, abs=1e-12)
    assert values["C"] == 0.0


def test_service_amount_column_names():
    assert index.service_amount_column("clinic", "point") == "clinic_count"
    assert index.service_amount_column("road", "line") == "road_length"
    with pytest.raises(ValueError, match="polygon"):
        index.service_amount_column("road", "polygon")


def test_pcen_pop_denominator_matches_eq3_by_hand():
    got = index.pcen(city_with_neighbours(), amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop")
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["X"] == pytest.approx(2 / 100, abs=1e-12)
    assert values["Y"] == pytest.approx((0 + 2 * 0.5) / 200, abs=1e-12)


def test_pcen_popdensity_divides_by_population_over_area():
    got = index.pcen(city_with_neighbours(), amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="popdensity")
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["Y"] == pytest.approx((0 + 2 * 0.5) / (200 / 2), abs=1e-12)


def test_pcen_include_neighbors_false_is_eq4():
    got = index.pcen(city_with_neighbours(), amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     include_neighbors=False)
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["Y"] == 0.0


def test_swallowed_skips_a_neighbour_with_no_row():
    """Today's behaviour: an absent neighbour contributes nothing."""
    frame = city_with_neighbours()
    reported = frame[frame["USO_AREA_U"] == "Y"]
    got = index.pcen(reported, amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     absent_neighbor="swallowed")
    assert got.set_index("USO_AREA_U").loc["Y", "clinic_pcen"] == 0.0


def test_contributes_uses_the_pre_exclusion_frame():
    """DEL-21: excluded settlements still lend their services (Eq. 3)."""
    frame = city_with_neighbours()
    reported = frame[frame["USO_AREA_U"] == "Y"]
    got = index.pcen(reported, amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     absent_neighbor="contributes", lookup_frame=frame)
    assert got.set_index("USO_AREA_U").loc["Y", "clinic_pcen"] == \
        pytest.approx((0 + 2 * 0.5) / 200, abs=1e-12)


def test_contributes_without_a_lookup_frame_is_a_value_error():
    with pytest.raises(ValueError, match="lookup_frame"):
        index.pcen(city_with_neighbours(), amount_col="clinic_count",
                   pcen_col="clinic_pcen", denominator="pop",
                   absent_neighbor="contributes")


def test_contributes_with_an_id_absent_from_the_lookup_frame_raises():
    frame = city_with_neighbours()
    reported = frame[frame["USO_AREA_U"] == "Y"]
    with pytest.raises(KeyError, match="X"):
        index.pcen(reported, amount_col="clinic_count",
                   pcen_col="clinic_pcen", denominator="pop",
                   absent_neighbor="contributes", lookup_frame=reported)


@pytest.mark.parametrize("kwargs,match", [
    (dict(denominator="households"), "households"),
    (dict(absent_neighbor="maybe"), "maybe"),
    (dict(decay_form="sideways"), "sideways"),
    (dict(decay_form="exponential"), "scale_km"),
    (dict(distance_unit="m"), "'m'"),
])
def test_pcen_rejects_unknown_values(kwargs, match):
    call = dict(amount_col="clinic_count", pcen_col="clinic_pcen",
                denominator="pop")
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        index.pcen(city_with_neighbours(), **call)


def test_minmax_is_eq2():
    frame = pd.DataFrame({"pcen": [1.0, 2.0, 5.0]})
    got = index.minmax(frame, source_col="pcen", target_col="idx")
    assert list(got["idx"]) == pytest.approx([0.0, 0.25, 1.0], abs=1e-12)


def test_minmax_raises_on_a_constant_column():
    """Eq. 2 is undefined when every settlement scores the same: (v-lo)/(hi-lo)
    is 0/0. A constant column means something upstream is wrong — an empty
    service layer, or an exclusion set that removed every settlement that had
    the service — so the guard names the column instead of dividing."""
    frame = gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B"], "bank_pcen": [0.25, 0.25]},
        geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:7760")
    with pytest.raises(ValueError) as excinfo:
        index.minmax(frame, source_col="bank_pcen", target_col="bank_idx")
    message = str(excinfo.value)
    assert "'bank_pcen'" in message
    assert "0.25" in message
    assert "max == min" in message


def test_minmax_raises_on_a_single_row_frame():
    """One reported settlement is the degenerate case that reaches this in
    practice (an exclusion set that leaves one row)."""
    frame = gpd.GeoDataFrame(
        {"USO_AREA_U": ["A"], "bank_pcen": [0.4]},
        geometry=[Point(0, 0)], crs="EPSG:7760")
    with pytest.raises(ValueError, match="'bank_pcen'"):
        index.minmax(frame, source_col="bank_pcen", target_col="bank_idx")


def test_service_index_adds_pcen_and_idx():
    got = index.service_index(city_with_neighbours(), "clinic_count",
                              service="clinic", denominator="pop")
    assert list(got["clinic_idx"]) == pytest.approx([1.0, 0.0], abs=1e-12)


def test_overall_psi_averages_idx_columns():
    frame = pd.DataFrame({"a_idx": [0.0, 1.0], "b_idx": [1.0, 1.0],
                          "other": [9.0, 9.0]})
    got = index.overall_psi(frame, second_normalization=True)
    assert list(got["unnorm_psi"]) == pytest.approx([0.5, 1.0], abs=1e-12)
    assert list(got["norm_psi"]) == pytest.approx([0.0, 1.0], abs=1e-12)


def test_overall_psi_omits_norm_psi_when_second_normalization_is_false():
    frame = pd.DataFrame({"a_idx": [0.0, 1.0]})
    got = index.overall_psi(frame, second_normalization=False)
    assert "unnorm_psi" in got.columns
    assert "norm_psi" not in got.columns


def test_service_index_propagates_the_guard():
    """service_index = pcen then minmax; a constant PCEN column must surface
    as the same ValueError, not as a NaN idx column."""
    frame = gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B"], "bank_count": [1, 1],
         "population": [100.0, 100.0], "area_km2": [1.0, 1.0],
         "nbrs_dist_bbox": [[], []]},
        geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:7760")
    with pytest.raises(ValueError, match="'bank_pcen'"):
        index.service_index(frame, "bank_count", service="bank",
                            denominator="pop")


def test_overall_psi_second_normalization_propagates_the_guard():
    """The second min-max is the other caller: a frame whose per-service
    indices average to the same value everywhere now names unnorm_psi. Also
    proves the guard does NOT fire when second_normalization=False."""
    frame = gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B"], "bank_idx": [0.5, 0.5]},
        geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:7760")
    with pytest.raises(ValueError, match="'unnorm_psi'"):
        index.overall_psi(frame, second_normalization=True)

    got = index.overall_psi(frame, second_normalization=False)
    assert list(got["unnorm_psi"]) == pytest.approx([0.5, 0.5], abs=1e-12)


# --- 3D: the four decay forms (spec § 2.2) -----------------------------
@pytest.mark.parametrize("form,kwargs,expected", [
    ("inverse_linear", {}, 1 / 1.5),
    ("none", {}, 1.0),
    ("inverse_power", {"exponent": 1}, 1 / 1.5),
    ("inverse_power", {"exponent": 2}, 1 / 1.5 ** 2),
    ("exponential", {"scale_km": 1.0}, math.exp(-0.5)),
    ("exponential", {"scale_km": 2.0}, math.exp(-0.25)),
])
def test_decay_forms_at_half_a_kilometre(form, kwargs, expected):
    assert index._decay(0.5, form, "km", **kwargs) == pytest.approx(
        expected, abs=1e-15)


@pytest.mark.parametrize("form,kwargs", [
    ("inverse_linear", {}), ("none", {}),
    ("inverse_power", {"exponent": 2}), ("exponential", {"scale_km": 1.0}),
])
def test_every_form_gives_weight_one_at_zero_distance(form, kwargs):
    """Why `decay.distance: boundary` leaves every touching or overlapping
    neighbour undecayed, under all four forms."""
    assert index._decay(0.0, form, "km", **kwargs) == 1.0


@pytest.mark.parametrize("args,kwargs,match", [
    (("sideways", "km"), {}, "sideways"),
    (("inverse_power", "km"), {}, "exponent"),
    (("exponential", "km"), {}, "scale_km"),
    (("inverse_linear", "km"), {"exponent": 2}, "exponent"),
    (("none", "km"), {"scale_km": 1.0}, "scale_km"),
    (("inverse_linear", "m"), {}, "'m'"),
])
def test_decay_rejects_unknown_forms_and_misplaced_parameters(args, kwargs,
                                                              match):
    with pytest.raises(ValueError, match=match):
        index._decay(0.5, *args, **kwargs)


def test_pcen_uses_the_form_and_its_parameter():
    """Same two-settlement city as test_pcen_pop_denominator_matches_eq3_by_hand
    (X owns 2 clinics, Y owns 0, each is the other's only neighbour at
    1.0 km, Y's population is 200) with the weight changed: under
    `inverse_power` 2 the neighbour lends 2 * 1/(1+1)**2 instead of 2 * 1/2.
    """
    got = index.pcen(city_with_neighbours(), amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     decay_form="inverse_power", exponent=2)
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["Y"] == pytest.approx((0 + 2 * 1 / (1 + 1.0) ** 2) / 200,
                                        abs=1e-12)   # 0.0025


def test_service_index_forwards_the_decay_parameters():
    """`service_index` is what `index_frames` actually calls, so the
    parameters have to survive that hop too: `exponential` with scale_km 1
    gives Y (0 + 2 * e^-1) / 200."""
    import math

    got = index.service_index(city_with_neighbours(), "clinic_count",
                              service="clinic", denominator="pop",
                              decay_form="exponential", scale_km=1.0)
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["Y"] == pytest.approx((0 + 2 * math.exp(-1.0)) / 200,
                                        abs=1e-12)


# --- 3E: the partial-barrier weight (spec § 2.4) -----------------------
def city_with_weights(weight):
    """city_with_neighbours plus the weight column apply_barrier writes."""
    gdf = city_with_neighbours()
    gdf["nbrs_barrier_weight"] = [[("Y", weight)], [("X", weight)]]
    return gdf


def test_pcen_multiplies_the_neighbour_term_by_its_barrier_weight():
    """Y owns nothing and borrows X's 2 clinics at 1 km (decay 1/2); a
    half-blocked shared boundary halves what it borrows."""
    got = index.pcen(city_with_weights(0.5), amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     nbr_weight_col="nbrs_barrier_weight")
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["Y"] == pytest.approx((0 + 0.5 * 2 * 0.5) / 200, abs=1e-12)
    assert values["X"] == pytest.approx(2 / 100, abs=1e-12)


def test_a_weight_of_one_is_bit_identical_to_no_weight_column():
    """1.0 * x is exact in IEEE and multiplication is left-associative, so
    the weighted loop cannot move a number when every weight is 1. This is
    what keeps code-2025 byte-identical."""
    weighted = index.pcen(city_with_weights(1.0), amount_col="clinic_count",
                          pcen_col="clinic_pcen", denominator="pop",
                          nbr_weight_col="nbrs_barrier_weight")
    plain = index.pcen(city_with_neighbours(), amount_col="clinic_count",
                       pcen_col="clinic_pcen", denominator="pop")
    assert list(weighted["clinic_pcen"]) == list(plain["clinic_pcen"])


def test_a_neighbour_with_no_weight_is_a_loud_key_error():
    """Never a silent 1.0: a distance list and a weight list that disagree
    mean the artifact and the frame came from different runs."""
    frame = city_with_weights(0.5)
    frame.at[1, "nbrs_barrier_weight"] = []
    with pytest.raises(KeyError, match="X"):
        index.pcen(frame, amount_col="clinic_count", pcen_col="clinic_pcen",
                   denominator="pop", nbr_weight_col="nbrs_barrier_weight")


def test_service_index_forwards_the_weight_column():
    got = index.service_index(city_with_weights(0.5), "clinic_count",
                              service="clinic", denominator="pop",
                              nbr_weight_col="nbrs_barrier_weight")
    values = got.set_index("USO_AREA_U")
    assert values.loc["Y", "clinic_pcen"] == pytest.approx(
        (0 + 0.5 * 2 * 0.5) / 200, abs=1e-12)
    # min-max still runs: X is the max, Y the min
    assert values.loc["X", "clinic_idx"] == 1.0
    assert values.loc["Y", "clinic_idx"] == 0.0


# --- 3E: overlap lending (spec § 3.1-3.3, § 6.4) -----------------------
def overlap_city():
    """P and Q OVERLAP in x in [1000, 1200]; Z is 4 km away and disjoint.

    One clinic sits in the overlap (so it is P's own AND Q's own), one in P
    alone; one road runs from x = 500 to x = 1500 at y = 200, so 200 m of it
    lie inside BOTH P and Q. The amount columns are the ones
    `index_frames` would have computed: clinic P 2, Q 1, Z 0; road P 0.7 km,
    Q 0.5 km, Z 0.
    """
    return gpd.GeoDataFrame(
        {"USO_AREA_U": ["P", "Q", "Z"],
         "nbrs_bbox": [["Q"], ["P"], []],
         "clinic_count": [2, 1, 0],
         "road_length": [0.7, 0.5, 0.0]},
        geometry=[box(0, 0, 1200, 1000), box(1000, 0, 2000, 1000),
                  box(5000, 0, 6000, 1000)],
        crs="EPSG:7760")


def overlap_clinics():
    return gpd.GeoDataFrame(
        {"service": ["clinic", "clinic"]},
        geometry=[Point(1100, 500), Point(600, 500)], crs="EPSG:7760")


def overlap_roads():
    from shapely.geometry import LineString

    return gpd.GeoDataFrame(
        {"service": ["road"]},
        geometry=[LineString([(500, 200), (1500, 200)])], crs="EPSG:7760")


def test_shared_amounts_counts_a_point_inside_two_settlements():
    """One ENTRY per ordered pair that shares something, and nothing at all
    for the point inside P alone or for the disjoint third settlement — the
    sparse representation the cost argument rests on."""
    got = index.shared_amounts(overlap_city(), overlap_clinics(),
                               kind="point", amount_col="clinic_count")
    assert got == {("P", "Q"): 1, ("Q", "P"): 1}


def test_shared_amounts_measures_the_road_inside_the_overlap():
    """200 m of the road lie in P n Q, so 0.2 km is lent by neither side."""
    got = index.shared_amounts(overlap_city(), overlap_roads(),
                               kind="line", amount_col="road_length")
    assert got == {("P", "Q"): pytest.approx(0.2, abs=1e-12),
                   ("Q", "P"): pytest.approx(0.2, abs=1e-12)}


def test_shared_amounts_is_empty_when_nothing_is_shared():
    """A clean layer costs nothing: every point inside one settlement, every
    neighbour pair a plain border. The dict is EMPTY, not full of zeroes."""
    city = overlap_city()
    only_p = gpd.GeoDataFrame({"service": ["clinic"]},
                              geometry=[Point(600, 500)], crs="EPSG:7760")
    assert index.shared_amounts(city, only_p, kind="point",
                                amount_col="clinic_count") == {}


def test_shared_amounts_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="polygon"):
        index.shared_amounts(overlap_city(), overlap_clinics(),
                             kind="polygon", amount_col="clinic_count")


def city_with_a_shared_clinic():
    """X and Y, 1 km apart (decay 1/2), each owning the SAME one clinic —
    X owns a second of its own. This is the O1/O2 shape at the pcen level."""
    gdf = city_with_neighbours()
    gdf["clinic_count"] = [2.0, 1.0]
    return gdf


def test_pcen_subtracts_what_the_receiver_already_holds():
    """Y already holds the shared clinic, so X lends it (2 - 1) = 1; X holds
    Y's only clinic, so Y lends it nothing at all."""
    shared = {("X", "Y"): 1, ("Y", "X"): 1}
    got = index.pcen(city_with_a_shared_clinic(), amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     shared_amounts=shared)
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["Y"] == pytest.approx((1 + (2 - 1) * 0.5) / 200, abs=1e-12)
    assert values["X"] == pytest.approx((2 + (1 - 1) * 0.5) / 100, abs=1e-12)


def test_an_empty_shared_structure_is_bit_identical_to_no_structure():
    """A pair with no entry is `|S_j \\ S_i| == |S_j|` EXACTLY: the sparse 0
    is the representation of 'nothing shared', not a swallowed miss."""
    sparse = index.pcen(city_with_a_shared_clinic(),
                        amount_col="clinic_count", pcen_col="clinic_pcen",
                        denominator="pop", shared_amounts={})
    plain = index.pcen(city_with_a_shared_clinic(),
                       amount_col="clinic_count", pcen_col="clinic_pcen",
                       denominator="pop")
    assert list(sparse["clinic_pcen"]) == list(plain["clinic_pcen"])


def test_the_barrier_weight_and_the_overlap_rule_compose():
    """The one place both multipliers act on one pair: a half-blocked shared
    boundary halves what is left after the overlap subtraction."""
    frame = city_with_a_shared_clinic()
    frame["nbrs_barrier_weight"] = [[("Y", 0.5)], [("X", 0.5)]]
    got = index.pcen(frame, amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     nbr_weight_col="nbrs_barrier_weight",
                     shared_amounts={("X", "Y"): 1, ("Y", "X"): 1})
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["Y"] == pytest.approx(
        (1 + 0.5 * (2 - 1) * 0.5) / 200, abs=1e-12)


def test_a_shared_amount_larger_than_the_neighbours_own_raises():
    """S_j n S_i is part of S_j, so shared_ij <= amount_j on both sides by
    construction. A negative lent means the two frames came from different
    runs; validate.check_no_negative would report it much later as a data
    problem, so it is caught here instead."""
    with pytest.raises(ValueError, match="would lend"):
        index.pcen(city_with_a_shared_clinic(), amount_col="clinic_count",
                   pcen_col="clinic_pcen", denominator="pop",
                   shared_amounts={("Y", "X"): 5})


def _lend_with_shared(shared_xy):
    """Y's clinic_pcen when X's shared amount toward Y is `shared_xy`."""
    got = index.pcen(city_with_a_shared_clinic(), amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     shared_amounts={("Y", "X"): shared_xy})
    return got.set_index("USO_AREA_U")["clinic_pcen"]["Y"]


def test_a_true_ulp_of_over_subtraction_is_clamped_not_raised():
    """A LINE service's own amount and its shared part are two independent
    GEOS clips of the same road, so the shared part can exceed the whole by
    an ulp. The real layer produced exactly this: one pair of 4,069
    overlapping ones lent -1.1102230246251565e-16 km of road, a tenth of a
    picometre, and the unclamped guard aborted a 4,357-settlement run.

    `math.nextafter`, NOT `own + 1.1e-16`: adding a literal smaller than
    half an ulp rounds straight back to `own`, so the earlier version of
    this test compared equal values, never reached the clamp, and passed
    identically with the fix reverted. The final review of 6 Sep 2026
    proved that by reverting it.

    What this test DOES pin, verified by mutation: reverting to the pre-fix
    strict `if lent < 0: raise` fails it. What no test can pin, and it is
    honest to say so: deleting the `lent = 0.0` clamp while KEEPING the
    tolerance changes no observable number, because a one-ulp negative
    propagated into the sum lands ~1e-18 from the answer — below any
    tolerance worth asserting. The clamp is hygiene against a negative
    reaching `validate.check_no_negative`, not an arithmetic correction.
    """
    own = city_with_a_shared_clinic().set_index("USO_AREA_U").loc[
        "X", "clinic_count"]
    over = math.nextafter(own, math.inf)
    assert over > own, "the perturbation must survive rounding"
    # X's whole amount minus a hair more than itself: a real negative.
    assert own - over < 0.0
    # Y keeps only its own clinic — the clamp took the deficit to exactly 0.
    assert _lend_with_shared(over) == pytest.approx(1 / 200, abs=1e-15)


def test_the_tolerance_boundary_is_pinned_on_both_sides():
    """The constant itself, not just the mechanism. A deficit just inside
    `_SHARED_TOLERANCE` clamps; one just outside raises. Without this, the
    tolerance could be widened by orders of magnitude — silently absorbing
    a real mismatch — and every other test would still pass.
    """
    own = city_with_a_shared_clinic().set_index("USO_AREA_U").loc[
        "X", "clinic_count"]
    band = index._SHARED_TOLERANCE * max(1.0, abs(own))

    inside = own + band * 0.5
    assert own - inside < 0.0, "must be a real over-subtraction"
    assert _lend_with_shared(inside) == pytest.approx(1 / 200, abs=1e-15)

    outside = own + band * 2.0
    with pytest.raises(ValueError, match="would lend"):
        _lend_with_shared(outside)


def test_the_tolerance_is_a_derived_multiple_of_machine_epsilon():
    """Not an arbitrary round number: it is sized to a few dozen rounding
    steps of a double, which is what two GEOS clip orders can differ by.
    An arbitrary 1e-9 — the first value shipped — would have absorbed a
    nanometre-scale REAL error without a word."""
    assert index._SHARED_TOLERANCE == 64 * sys.float_info.epsilon
    assert index._SHARED_TOLERANCE < 1e-13
    # comfortably above the 1.1e-16 the real layer actually produced
    assert index._SHARED_TOLERANCE > 1.1e-16 * 10


def test_service_index_forwards_the_shared_structure():
    got = index.service_index(city_with_a_shared_clinic(), "clinic_count",
                              service="clinic", denominator="pop",
                              shared_amounts={("X", "Y"): 1, ("Y", "X"): 1})
    values = got.set_index("USO_AREA_U")
    assert values.loc["Y", "clinic_pcen"] == pytest.approx(
        (1 + (2 - 1) * 0.5) / 200, abs=1e-12)
    # min-max still runs: X is the max, Y the min
    assert values.loc["X", "clinic_idx"] == 1.0
    assert values.loc["Y", "clinic_idx"] == 0.0
