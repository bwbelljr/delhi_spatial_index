"""The index math: counts, lengths, PCEN (Eq. 3), min-max (Eq. 2), PSI (Eq. 1).

Pure functions with explicit keyword arguments — never imports
delhi_psi.config. Every expression is copied verbatim from
spatial_index_utils.py. Two deliberate non-changes:
  * `minmax` has NO hi == lo guard, exactly as `calc_service_index` had none.
  * the -1.0 sentinel initialisations stay.
The one behavioural change is DEL-21: `calc_pcen_mobile`'s bare
`except: pass` becomes an explicit lookup miss, which is what makes
`absent_neighbor: contributes` implementable at all.
"""

import logging
import math

import geopandas as gpd

log = logging.getLogger(__name__)

DENOMINATORS = ("pop", "popdensity", "one")
ABSENT_NEIGHBOR = ("swallowed", "contributes")


def point_counts(polygon_gdf, point_gdf, *, count_col, id_col="USO_AREA_U"):
    """Count points inside each polygon (gpd.sjoin's default `intersects`).

    NOTE: boundary-inclusive, so a point exactly on a shared border counts
    for both neighbours (rule-set gap #6, latent on the real layers — pinned
    by tests/test_oracle.py::test_gap6_border_point_is_double_counted...).
    Unchanged here on purpose.
    """
    point_cnt = gpd.sjoin(polygon_gdf, point_gdf).groupby(id_col).size().\
        reset_index()
    point_cnt = point_cnt.rename(columns={0: count_col})
    out = polygon_gdf.merge(point_cnt, how="left", on=id_col)
    out[count_col] = out[count_col].fillna(0)
    out[count_col] = out[count_col].astype(int)
    return out


def _length_in_polygon(small_gdf, poly_geom_col, line_geom_col):
    """Total length (km) of the line pieces inside one polygon."""
    total_length = 0
    for i, row in small_gdf.iterrows():
        polygon = row[poly_geom_col]
        line = row[line_geom_col]
        intersection = polygon.intersection(line)
        length = intersection.length / 1000
        total_length += length
    return total_length


def road_lengths(polygon_gdf, line_gdf, *, length_col, id_col="USO_AREA_U"):
    """Total (poly)line length in km inside each polygon."""
    polygon_gdf[length_col] = 0.0

    line_geom_col = "line_geometry"
    line_gdf[line_geom_col] = line_gdf["geometry"]

    joined = gpd.sjoin(polygon_gdf, line_gdf)
    joined_grouped = joined.groupby(id_col)

    for name, group in joined_grouped:
        name_index = polygon_gdf[polygon_gdf[id_col] == name].index.values[0]
        total_road_length = _length_in_polygon(
            small_gdf=group, poly_geom_col="geometry",
            line_geom_col=line_geom_col)
        polygon_gdf.loc[name_index, length_col] = total_road_length

    return polygon_gdf


def service_amount_column(service, kind):
    """The column a service's own amount lands in.

    Point services count (`clinic_count`); line services measure length
    (`road_length`) — which is what production ended up with after its
    `road_count -> road_length` rename, so the special case disappears.
    """
    if kind == "point":
        return f"{service}_count"
    if kind == "line":
        return f"{service}_length"
    raise ValueError(
        f"unknown service kind {kind!r}; allowed values: ['point', 'line']")


DECAY_FORMS = ("inverse_linear", "none", "inverse_power", "exponential")


def _decay(distance_km, decay_form, distance_unit, *, exponent=None,
           scale_km=None):
    """The distance-decay weight w(D).

    inverse_linear: 1/(1+D) — the July 2025 rule.
    none:           1 — every neighbour counts in full.
    inverse_power:  1/(1+D)^exponent; exponent 1 reproduces inverse_linear.
    exponential:    exp(-D/scale_km).

    A parameter the form does not use is an error, never an ignored value.
    """
    if distance_unit != "km":
        raise ValueError(
            f"unknown decay distance unit {distance_unit!r}; allowed values: "
            "['km']")
    if decay_form not in DECAY_FORMS:
        raise ValueError(
            f"unknown decay form {decay_form!r}; allowed values: "
            f"{list(DECAY_FORMS)}")
    if decay_form == "inverse_power":
        if exponent is None:
            raise ValueError("decay form 'inverse_power' requires exponent")
    elif exponent is not None:
        raise ValueError(
            f"exponent is not used by decay form {decay_form!r}; it is used "
            "by 'inverse_power'")
    if decay_form == "exponential":
        if scale_km is None:
            raise ValueError("decay form 'exponential' requires scale_km")
    elif scale_km is not None:
        raise ValueError(
            f"scale_km is not used by decay form {decay_form!r}; it is used "
            "by 'exponential'")

    if decay_form == "inverse_linear":
        return 1 / (1 + distance_km)
    if decay_form == "none":
        return 1.0
    if decay_form == "inverse_power":
        return 1 / (1 + distance_km) ** exponent
    return math.exp(-distance_km / scale_km)


def pcen(polygon_gdf, *, amount_col, pcen_col, denominator,
         nbr_dist_col="nbrs_dist_bbox", lookup_frame=None,
         absent_neighbor="swallowed", include_neighbors=True,
         decay_form="inverse_linear", distance_unit="km", exponent=None,
         scale_km=None,
         pop_col="population", area_col="area_km2", id_col="USO_AREA_U"):
    """Eq. 3: effective service count per denominator, with distance decay.

    absent_neighbor="swallowed": a neighbour id with no row in the compute
        frame contributes nothing (today's behaviour, as an explicit lookup
        miss — never a bare `except`).
    absent_neighbor="contributes": amounts are looked up in `lookup_frame`,
        the PRE-EXCLUSION frame, so excluded settlements still contribute
        their services; an id absent from that frame too is an error.
    include_neighbors=False: Eq. 4 as written — own amount only, no
        neighbour term (`roads: eq4_own_only`).
    """
    if denominator not in DENOMINATORS:
        raise ValueError(
            f"unknown denominator {denominator!r}; allowed values: "
            f"{list(DENOMINATORS)}")
    if absent_neighbor not in ABSENT_NEIGHBOR:
        raise ValueError(
            f"unknown absent_neighbor {absent_neighbor!r}; allowed values: "
            f"{list(ABSENT_NEIGHBOR)}")

    gdf_copy = polygon_gdf.copy()

    if absent_neighbor == "contributes":
        if lookup_frame is None:
            raise ValueError(
                "absent_neighbor='contributes' requires lookup_frame — the "
                "pre-exclusion frame the neighbour amounts are read from")
        lookup = lookup_frame
    else:
        lookup = gdf_copy

    # probe the decay knobs once, so a bad value fails even on a city with
    # no neighbour links at all
    _decay(0.0, decay_form, distance_unit, exponent=exponent,
           scale_km=scale_km)

    gdf_copy[pcen_col] = -1.0

    for idx, row in gdf_copy.iterrows():
        if denominator == "popdensity":
            denom = row[pop_col] / row[area_col]
        elif denominator == "pop":
            denom = row[pop_col]
        else:
            denom = 1

        poly_count = row[amount_col]

        if include_neighbors:
            for nbr_id, nbr_dist in row[nbr_dist_col]:
                match = lookup[lookup[id_col] == nbr_id]
                if len(match) == 0:
                    if absent_neighbor == "contributes":
                        raise KeyError(
                            f"neighbour {nbr_id!r} of {row[id_col]!r} has no "
                            "row in the pre-exclusion lookup frame")
                    continue
                nbr_count = match[amount_col].array[0]
                poly_count += nbr_count * _decay(nbr_dist, decay_form,
                                                 distance_unit,
                                                 exponent=exponent,
                                                 scale_km=scale_km)

        gdf_copy.loc[idx, pcen_col] = poly_count / denom

    return gdf_copy


def minmax(polygon_gdf, *, source_col, target_col):
    """Eq. 2: rescale a column to [0, 1] across the frame.

    Verbatim `calc_service_index` — deliberately WITHOUT a hi == lo guard.
    """
    gdf_copy = polygon_gdf.copy()

    pcen_min = gdf_copy[source_col].min()
    pcen_max = gdf_copy[source_col].max()

    gdf_copy[target_col] = -1.0

    for idx, row in gdf_copy.iterrows():
        result = (row[source_col] - pcen_min) / (pcen_max - pcen_min)
        gdf_copy.loc[idx, target_col] = result

    return gdf_copy


def service_index(polygon_gdf, amount_col, *, service, denominator,
                  nbr_dist_col="nbrs_dist_bbox", lookup_frame=None,
                  absent_neighbor="swallowed", include_neighbors=True,
                  decay_form="inverse_linear", distance_unit="km",
                  exponent=None, scale_km=None,
                  pop_col="population", area_col="area_km2",
                  id_col="USO_AREA_U"):
    """pcen then minmax for one service — replaces BOTH create_service_index
    variants (DEL-16). Fed by point_counts() or road_lengths()."""
    pcen_col = f"{service}_pcen"
    idx_col = f"{service}_idx"
    out = pcen(polygon_gdf, amount_col=amount_col, pcen_col=pcen_col,
               denominator=denominator, nbr_dist_col=nbr_dist_col,
               lookup_frame=lookup_frame, absent_neighbor=absent_neighbor,
               include_neighbors=include_neighbors, decay_form=decay_form,
               distance_unit=distance_unit, exponent=exponent,
               scale_km=scale_km, pop_col=pop_col,
               area_col=area_col, id_col=id_col)
    return minmax(out, source_col=pcen_col, target_col=idx_col)


def overall_psi(polygon_gdf, *, second_normalization):
    """Eq. 1: the mean of every `*_idx` column, plus the optional second
    normalization (`norm_psi`); the column is absent when it is off."""
    out = polygon_gdf.copy()
    idx_columns = [column for column in out.columns
                   if column.endswith("_idx")]
    out["unnorm_psi"] = out[idx_columns].mean(axis=1)
    if second_normalization:
        out = minmax(out, source_col="unnorm_psi", target_col="norm_psi")
    return out
