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
scenario, denom, second_norm, absent_neighbor_contribution,
max_distance_km, decay_form, exponent, scale_km, decay_distance. RULESETS
binds the ideal (manuscript) and code (empirical) combinations.
"""

import math

import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

from tests.cities import ORACULUM
from tests.variants import VARIANTS

RULESETS = {
    "ideal": dict(adjacency_rule="border", barrier_rule="pair",
                  roads_formula="eq4", second_norm=False,
                  absent_neighbor_contribution="contributes"),
    "code": dict(adjacency_rule="bbox", barrier_rule="global",
                 roads_formula="decayed", second_norm=True,
                 absent_neighbor_contribution="swallowed"),
}

# tests/variants.py speaks CONFIG vocabulary; the only difference is the
# KEY names, so this map is a rename and never a translation of values.
# `decay.distance_unit` has no reference knob (the reference is km-only,
# as the manuscript is), so it is deliberately absent.
VARIANT_KNOBS = {
    ("adjacency", "rule"): "adjacency_rule",
    ("adjacency", "max_distance_km"): "max_distance_km",
    ("barrier", "rule"): "barrier_rule",
    ("barrier", "buffer_m"): "barrier_buffer_m",
    ("decay", "form"): "decay_form",
    ("decay", "distance"): "decay_distance",
    ("decay", "exponent"): "exponent",
    ("decay", "scale_km"): "scale_km",
    ("overlap", "lending"): "overlap_lending",
    ("transform", "form"): "transform_form",
    ("transform", "stage"): "transform_stage",
    ("aggregation", "rule"): "aggregation_rule",
}
# `barrier.combine` has no reference knob: the reference uses EVERY barrier
# row, which is what `any` means on a one-layer city, and both fixture cities
# have one layer or none. `decay.distance_unit` has none either (the
# reference is km-only, as the manuscript is).
IGNORED_VARIANT_KEYS = frozenset({("decay", "distance_unit"),
                                  ("barrier", "combine")})


def _variant_overrides(spec):
    out = {}
    for block, mapping in spec.items():
        for key, value in mapping.items():
            if (block, key) in IGNORED_VARIANT_KEYS:
                continue
            if (block, key) not in VARIANT_KNOBS:
                raise ValueError(
                    f"tests/variants.py: {block}.{key} has no reference "
                    f"knob; add one to VARIANT_KNOBS or to "
                    f"IGNORED_VARIANT_KEYS")
            out[VARIANT_KNOBS[(block, key)]] = value
    return out


# `code` base + the table's overrides: a variant is today's empirical
# rule-set with one or two values changed, so a difference in the output is
# attributable to those values alone.
VARIANT_RULESETS = {name: dict(RULESETS["code"], **_variant_overrides(spec))
                    for name, spec in VARIANTS.items()}

# Backward-compatible view of Oraculum's table in the 2-tuple shape this
# module has always consumed: {name: (dropped ids, dropped_before_neighbors)}.
# ORACULUM.scenarios' ORDER is today's order, which fixes expected_values.csv.
# (tests/cities.py imports geopandas and nothing from this repo, so the
# INDEPENDENCE RULE is intact: it is fixture plumbing, not index math.)
SCENARIOS = {s.name: (s.dropped, s.dropped_before_neighbors)
             for s in ORACULUM.scenarios}

POINT_SERVICES = ("clinic", "school", "bank", "police", "ration", "transport")


def adjacency(settlements, rule, max_distance_km=None):
    """Directed neighbour lists under `rule`.

    within_distance: j is a neighbour of i iff the POLYGON-TO-POLYGON
        shortest distance is <= max_distance_km * 1000 metres. At 0 km that
        is `intersects` — corner-only touches and overlaps included — which
        is what the § 4.1 pins compare it against.
    """
    if rule == "within_distance":
        if max_distance_km is None:
            raise ValueError(
                "adjacency rule 'within_distance' requires max_distance_km")
    elif max_distance_km is not None:
        raise ValueError(
            f"max_distance_km is only used by rule 'within_distance', not "
            f"{rule!r}")
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
            elif rule == "within_distance":
                if idx[i].distance(idx[j]) <= max_distance_km * 1000:
                    nbrs.add(j)
            else:
                raise ValueError(rule)
        out[i] = nbrs
    return out


def _shared_boundary(geom_i, geom_j):
    """SB_ij: the boundary of every POLYGONAL component of the intersection
    (the owner's overlap rule) plus every LINEAL component. Points contribute
    nothing, and a GeometryCollection is decomposed part by part — shapely
    does not define `.boundary` for a collection, and a polygon that overlaps
    its neighbour on one side and shares an edge on another produces exactly
    that.
    """
    shared = geom_i.intersection(geom_j)
    if shared.is_empty:
        return None
    parts = list(shared.geoms) if hasattr(shared, "geoms") else [shared]
    pieces = []
    for part in parts:
        if part.geom_type in ("Polygon", "MultiPolygon"):
            pieces.append(part.boundary)
        elif part.geom_type in ("LineString", "LinearRing", "MultiLineString"):
            pieces.append(part)
    return unary_union(pieces) if pieces else None


def partial_weights(nbrs, settlements, barriers, buffer_m):
    """{(i, j): w_ij} for every DIRECTED link in `nbrs` (spec § 2.1).

    w_ij = 1 - L_blocked / L_shared, where L_blocked is the length of the
    shared boundary within `buffer_m` metres of any barrier feature. A
    zero-length shared boundary (empty intersection, or a corner-only
    contact) has nothing to block, so w = 1.

    Independent of production by construction: its own decomposition, and it
    unions ALL the barrier buffers ONCE rather than intersecting piece by
    piece against STRtree candidates. Both fixture cities carry one barrier
    row or none, so the union is trivial here (spec § 9).
    """
    if buffer_m is None or not buffer_m > 0:
        raise ValueError(
            "barrier rule 'partial_weighted' requires barrier_buffer_m > 0, "
            f"got {buffer_m!r}")
    idx = settlements.set_index("USO_AREA_U").geometry
    geoms = ([] if barriers is None or len(barriers) == 0
             else list(barriers.geometry))
    blocked_area = (unary_union([g.buffer(buffer_m) for g in geoms])
                    if geoms else None)
    out = {}
    for i, js in nbrs.items():
        for j in js:
            shared = _shared_boundary(idx[i], idx[j])
            length = 0.0 if shared is None else shared.length
            if length == 0 or blocked_area is None:
                out[(i, j)] = 1.0
                continue
            blocked = shared.intersection(blocked_area).length
            out[(i, j)] = (0.0 if blocked >= length
                           else 1 - blocked / length)
    return out


def apply_barrier(nbrs, settlements, barriers, rule, buffer_m=None):
    """Sever (or, under partial_weighted, prune at w == 0) neighbour links.

    Returns the same {i: set} shape under every rule, so no caller changes.
    The rule and buffer_m are validated BEFORE the empty-barriers
    short-circuit — a city with no barriers must still refuse a bad
    combination, which is what production does and what the messy city (no
    barriers at all) exercises.
    """
    if rule not in ("global", "pair", "partial_weighted"):
        raise ValueError(rule)
    if rule == "partial_weighted":
        weights = partial_weights(nbrs, settlements, barriers, buffer_m)
        return {i: {j for j in js if weights[(i, j)] > 0.0}
                for i, js in nbrs.items()}
    if buffer_m is not None:
        raise ValueError(
            "barrier_buffer_m is only used by barrier rule "
            f"'partial_weighted', not {rule!r}")
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
        else:
            kept = set()
            for j in js:
                shared = idx[i].intersection(idx[j])
                crossed = any(b.intersects(shared) for b in barrier_geoms)
                if not crossed:
                    kept.add(j)
            out[i] = kept
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
    # EVERY road row, not just the first: a city may carry the road network
    # as several LineStrings (the messy city does), and production's
    # `road_lengths` already sums all of them per settlement.
    road_geoms = list(services["road"].geometry)
    amounts["road"] = {
        i: sum(road.intersection(idx[i]).length for road in road_geoms) / 1000
        for i in idx.index}
    return amounts


OVERLAP_LENDINGS = ("whole", "outside_receiver")


def shared_amounts(nbrs, settlements, services):
    """{svc: {(i, j): amount}} — how much of `svc` lies inside BOTH i and j.

    Points: the number of that service's points `within` both, the same
    strict predicate `_service_amounts` uses — so `shared_ij <= amount_j` by
    construction and the subtraction in `compute_city` can never go
    negative. Roads: the clipped length in km of every road row inside
    `geom_i n geom_j`, summed, which is `_service_amounts`' own road
    arithmetic restricted to the intersection.

    SPARSE and SYMMETRIC. Only pairs that actually share something get an
    entry, in both orders; `compute_city` reads it with `.get((i, j), 0)`
    and that 0 IS the representation of "nothing shared", not a swallowed
    miss. On a city with no overlapping polygons every table comes back
    empty (Oraculum), which is why `overlap_outside` is degenerate there.

    The point tables are built from the SERVICE POINTS' containment, never
    from the pair list: a point inside one settlement — every point on a
    clean layer — is looked at once and contributes nothing.
    """
    idx = settlements.set_index("USO_AREA_U").geometry
    out = {}
    for svc in POINT_SERVICES:
        gdf = services.get(svc)
        table = {}
        if gdf is not None:
            for point in gdf.geometry:
                inside = [i for i in idx.index if point.within(idx[i])]
                if len(inside) < 2:
                    continue
                for i in inside:
                    for j in inside:
                        if i != j:
                            table[(i, j)] = table.get((i, j), 0) + 1
        out[svc] = table
    road_geoms = list(services["road"].geometry)
    table = {}
    # Sorted UNDIRECTED pairs: each is measured once and written both ways,
    # so the table is symmetric by construction and the order a set would
    # have iterated in cannot reach the arithmetic.
    for i, j in sorted({tuple(sorted((i, j)))
                        for i, js in nbrs.items() for j in js}):
        overlap = idx[i].intersection(idx[j])
        if overlap.is_empty:
            continue
        length = sum(road.intersection(overlap).length
                     for road in road_geoms) / 1000
        if length > 0:
            table[(i, j)] = length
            table[(j, i)] = length
    out["road"] = table
    return out


DECAY_FORMS = ("inverse_linear", "none", "inverse_power", "exponential")
DECAY_DISTANCES = ("centroid", "boundary")

# DEL-34: alternatives to `none` for the compressed 0-1 effect sizes (spec
# § 3). Independently derived from the equations — this module must never
# import delhi_psi.index.
TRANSFORM_FORMS = ("none", "log1p", "cbrt")
TRANSFORM_STAGES = ("pcen", "psi")


def _apply_transform(value, transform_form):
    if transform_form == "none":
        return value
    if transform_form == "log1p":
        return math.log1p(value)
    return math.cbrt(value)


def compute_city(settlements, services, barriers, *, adjacency_rule,
                 barrier_rule, roads_formula, scenario, denom, second_norm,
                 absent_neighbor_contribution, scenarios=None,
                 max_distance_km=None, barrier_buffer_m=None,
                 decay_form="inverse_linear", exponent=None, scale_km=None,
                 decay_distance="centroid", overlap_lending="whole",
                 transform_form="none", transform_stage=None,
                 aggregation_rule="mean_minmax"):
    # Every parameter a form does not use is REJECTED, not ignored — the
    # mapped-knob test relies on an unimplemented combination raising.
    if decay_form not in DECAY_FORMS:
        raise ValueError(f"unknown decay form {decay_form!r}; allowed "
                         f"values: {list(DECAY_FORMS)}")
    if decay_distance not in DECAY_DISTANCES:
        raise ValueError(f"unknown decay distance {decay_distance!r}; "
                         f"allowed values: {list(DECAY_DISTANCES)}")
    if overlap_lending not in OVERLAP_LENDINGS:
        raise ValueError(f"unknown overlap lending {overlap_lending!r}; "
                         f"allowed values: {list(OVERLAP_LENDINGS)}")
    if decay_form == "inverse_power":
        if exponent is None:
            raise ValueError("decay form 'inverse_power' requires exponent")
    elif exponent is not None:
        raise ValueError(f"exponent is not used by decay form "
                         f"{decay_form!r}; it is used by 'inverse_power'")
    if decay_form == "exponential":
        if scale_km is None:
            raise ValueError("decay form 'exponential' requires scale_km")
    elif scale_km is not None:
        raise ValueError(f"scale_km is not used by decay form "
                         f"{decay_form!r}; it is used by 'exponential'")
    if transform_form not in TRANSFORM_FORMS:
        raise ValueError(f"unknown transform form {transform_form!r}; "
                         f"allowed values: {list(TRANSFORM_FORMS)}")
    if transform_form == "none":
        if transform_stage is not None:
            raise ValueError(
                f"transform_stage {transform_stage!r} is not used by "
                "transform form 'none'; it is used when form is 'log1p' or "
                "'cbrt'")
    elif transform_stage not in TRANSFORM_STAGES:
        raise ValueError(
            f"transform form {transform_form!r} requires transform_stage in "
            f"{list(TRANSFORM_STAGES)}, got {transform_stage!r}")
    if aggregation_rule not in ("mean_minmax", "mean_rank"):
        raise ValueError(
            f"unknown aggregation rule {aggregation_rule!r}; allowed "
            "values: ['mean_minmax', 'mean_rank']")

    # `scenarios` defaults to the module table, so every existing call keeps
    # working; a caller may pass its own WITHOUT mutating the global (which
    # is what scripts/render_oracle_maps.py used to do).
    table = SCENARIOS if scenarios is None else scenarios
    dropped, drop_before = table[scenario]
    universe = settlements[~settlements["USO_AREA_U"].isin(dropped)] \
        if drop_before else settlements

    adjacent = adjacency(universe, adjacency_rule, max_distance_km)
    nbrs = apply_barrier(adjacent, universe, barriers, barrier_rule,
                         barrier_buffer_m)
    # The weights are recomputed here rather than threaded out of
    # apply_barrier, which keeps its {i: set} contract. Both fixture cities
    # are seven and eleven settlements, so the second pass is free.
    barrier_w = (partial_weights(adjacent, universe, barriers,
                                 barrier_buffer_m)
                 if barrier_rule == "partial_weighted" else None)
    cent = _centroid_km(universe)
    geom = universe.set_index("USO_AREA_U").geometry
    amounts = _service_amounts(universe, services)
    # Built on the post-barrier links, which are exactly the pairs the
    # neighbour sum below looks up. Nothing is built at all under `whole`.
    shared = (shared_amounts(nbrs, universe, services)
              if overlap_lending == "outside_receiver" else None)

    indexed = [i for i in universe["USO_AREA_U"]
               if drop_before or i not in dropped]
    meta = universe.set_index("USO_AREA_U")

    def denominator(i):
        pop = meta.loc[i, "population"]
        return pop / meta.loc[i, "area_km2"] if denom == "popdensity" else pop

    def contribution_weight(i, j):
        # boundary: polygon-to-polygon, so every touching or overlapping
        # neighbour is at distance 0 and lends its amount undecayed.
        if decay_distance == "boundary":
            d_km = geom[i].distance(geom[j]) / 1000
        else:
            d_km = cent[i].distance(cent[j]) / 1000
        if decay_form == "inverse_linear":
            return 1 / (1 + d_km)
        if decay_form == "none":
            return 1.0
        if decay_form == "inverse_power":
            return 1 / (1 + d_km) ** exponent
        return math.exp(-d_km / scale_km)

    rows = {}
    for i in indexed:
        row = {}
        for svc in POINT_SERVICES + ("road",):
            own = amounts[svc][i]
            decayed_sum = 0.0
            # Deterministic order: `nbrs[i]` is a set, its iteration order
            # depends on the hash seed, and float addition is not
            # associative — an unsorted sum differs by 1 ULP between
            # processes and the %.17g CSV then drifts.
            for j in sorted(nbrs[i]):
                if (not drop_before and j in dropped
                        and absent_neighbor_contribution == "swallowed"):
                    continue
                w = 1.0 if barrier_w is None else barrier_w[(i, j)]
                lent = amounts[svc][j]
                if shared is not None:
                    lent -= shared[svc].get((i, j), 0)
                decayed_sum += w * lent * contribution_weight(i, j)
            if svc == "road":
                row["road_length_km"] = own
                pcen_value = (own if roads_formula == "eq4"
                             else own + decayed_sum) / denominator(i)
            else:
                row[f"{svc}_count"] = own
                pcen_value = (own + decayed_sum) / denominator(i)
            # DEL-34, stage='pcen': the reported `*_pcen` value BECOMES the
            # transformed value, before Eq. 2's min-max below.
            if transform_stage == "pcen":
                pcen_value = _apply_transform(pcen_value, transform_form)
            row[f"{svc}_pcen"] = pcen_value
        rows[i] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    idx_cols = []
    n = len(df)
    for svc in POINT_SERVICES + ("road",):
        col = f"{svc}_pcen"
        pcen = df[col]
        if aggregation_rule == "mean_rank":
            # DEL-57: rank ascending with tie blocks averaged, rescaled so
            # min -> 0 and max -> 1. Written out longhand rather than via
            # Series.rank, so this stays an INDEPENDENT statement of the
            # rule rather than a second call to the same library routine
            # the production side uses.
            if n < 2:
                raise ValueError(
                    f"percentile rank of {col!r} is undefined across {n} "
                    "row(s): the rescaling divides by (n - 1)")
            order = sorted(range(n), key=lambda k: pcen.iloc[k])
            ranks = [0.0] * n
            position = 0
            while position < n:
                stop = position
                while (stop + 1 < n
                       and pcen.iloc[order[stop + 1]] == pcen.iloc[order[position]]):
                    stop += 1
                average = (position + stop) / 2 + 1
                for k in range(position, stop + 1):
                    ranks[order[k]] = average
                position = stop + 1
            df[f"{svc}_idx"] = [(r - 1.0) / (n - 1.0) for r in ranks]
        else:
            lo, hi = pcen.min(), pcen.max()
            if hi == lo:
                raise ValueError(
                    f"min-max of {col!r} is undefined: all {len(pcen)} "
                    f"values equal {lo!r} (hi == lo), so Eq. 2 divides 0/0 "
                    "— every settlement scores the same on this service")
            df[f"{svc}_idx"] = (pcen - lo) / (hi - lo)
        idx_cols.append(f"{svc}_idx")
    df["psi_eq1"] = df[idx_cols].mean(axis=1)
    # DEL-34, stage='psi': the composite is transformed BEFORE the second
    # normalization — the 2021 notebook's log(unnorm_psi + 1).
    if transform_stage == "psi":
        df["psi_eq1"] = df["psi_eq1"].map(
            lambda value: _apply_transform(value, transform_form))
    if second_norm:
        p = df["psi_eq1"]
        lo, hi = p.min(), p.max()
        if hi == lo:
            raise ValueError(
                f"min-max of 'psi_eq1' is undefined: all {len(p)} values "
                f"equal {lo!r} (hi == lo), so Eq. 2 divides 0/0 — every "
                f"settlement's unnormalized PSI is the same")
        df["norm_psi"] = (p - lo) / (hi - lo)
    return df


def emit_expected_values(out_path, city=ORACULUM):
    """Score `city` under every rule-set x scenario x denominator and write
    the long-format CSV. `out_path` stays FIRST so existing callers (the
    round-trip test, the generators) are unchanged.
    """
    settlements, barriers, services = (city.load_settlements(),
                                       city.load_barriers(),
                                       city.load_services())
    scenarios = {s.name: (s.dropped, s.dropped_before_neighbors)
                 for s in city.scenarios}
    records = []
    for rule, kwargs in RULESETS.items():
        for scenario in scenarios:
            for denom in ("pop", "popdensity"):
                df = compute_city(settlements, services, barriers,
                                  scenario=scenario, denom=denom,
                                  scenarios=scenarios, **kwargs)
                for sid, row in df.iterrows():
                    for metric, value in row.items():
                        records.append((rule, scenario, denom, sid,
                                        metric, value))
    out = pd.DataFrame(records, columns=["rule", "scenario", "denom",
                                         "settlement", "metric", "value"])
    out.to_csv(out_path, index=False, float_format="%.17g")
    return out


def emit_variant_expected_values(out_path, city):
    """Score `city` under every VARIANT_RULESETS entry and write the
    long-format CSV.

    ONE scenario — `city.scenarios[0]` (Oraculum `baseline`, messy
    `nopop_only`; the messy city has no scenario literally named `baseline`,
    because `U` is dropped by every one of them). The exclusion machinery is
    proven elsewhere and is orthogonal to these two knobs. Both denominators,
    `%.17g`, same columns as emit_expected_values.
    """
    settlements, barriers, services = (city.load_settlements(),
                                       city.load_barriers(),
                                       city.load_services())
    scenario = city.scenarios[0]
    scenarios = {s.name: (s.dropped, s.dropped_before_neighbors)
                 for s in city.scenarios}
    records = []
    for rule, kwargs in VARIANT_RULESETS.items():
        for denom in ("pop", "popdensity"):
            df = compute_city(settlements, services, barriers,
                              scenario=scenario.name, denom=denom,
                              scenarios=scenarios, **kwargs)
            for sid, row in df.iterrows():
                for metric, value in row.items():
                    records.append((rule, scenario.name, denom, sid, metric,
                                    value))
    out = pd.DataFrame(records, columns=["rule", "scenario", "denom",
                                         "settlement", "metric", "value"])
    out.to_csv(out_path, index=False, float_format="%.17g")
    return out


if __name__ == "__main__":
    from tests.cities import CITIES

    for target_city in CITIES:
        target = target_city.fixtures / "expected_values.csv"
        emit_expected_values(target, target_city)
        print(f"wrote {target}")
