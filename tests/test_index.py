"""delhi_psi.index — counts, lengths, PCEN (Eq. 3), min-max (Eq. 2), PSI (Eq. 1).

The exclusion axes are tested here directly, because this is where DEL-21's
`except: pass` becomes an explicit lookup.
"""
import math

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
