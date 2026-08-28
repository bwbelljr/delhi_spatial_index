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
    ("decay", "form"): "decay_form",
    ("decay", "distance"): "decay_distance",
    ("decay", "exponent"): "exponent",
    ("decay", "scale_km"): "scale_km",
}
IGNORED_VARIANT_KEYS = frozenset({("decay", "distance_unit")})


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


def apply_barrier(nbrs, settlements, barriers, rule):
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
        elif rule == "pair":
            kept = set()
            for j in js:
                shared = idx[i].intersection(idx[j])
                crossed = any(b.intersects(shared) for b in barrier_geoms)
                if not crossed:
                    kept.add(j)
            out[i] = kept
        else:
            raise ValueError(rule)
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


DECAY_FORMS = ("inverse_linear", "none", "inverse_power", "exponential")
DECAY_DISTANCES = ("centroid", "boundary")


def compute_city(settlements, services, barriers, *, adjacency_rule,
                 barrier_rule, roads_formula, scenario, denom, second_norm,
                 absent_neighbor_contribution, scenarios=None,
                 max_distance_km=None, decay_form="inverse_linear",
                 exponent=None, scale_km=None, decay_distance="centroid"):
    # Every parameter a form does not use is REJECTED, not ignored — the
    # mapped-knob test relies on an unimplemented combination raising.
    if decay_form not in DECAY_FORMS:
        raise ValueError(f"unknown decay form {decay_form!r}; allowed "
                         f"values: {list(DECAY_FORMS)}")
    if decay_distance not in DECAY_DISTANCES:
        raise ValueError(f"unknown decay distance {decay_distance!r}; "
                         f"allowed values: {list(DECAY_DISTANCES)}")
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

    # `scenarios` defaults to the module table, so every existing call keeps
    # working; a caller may pass its own WITHOUT mutating the global (which
    # is what scripts/render_oracle_maps.py used to do).
    table = SCENARIOS if scenarios is None else scenarios
    dropped, drop_before = table[scenario]
    universe = settlements[~settlements["USO_AREA_U"].isin(dropped)] \
        if drop_before else settlements

    nbrs = apply_barrier(adjacency(universe, adjacency_rule, max_distance_km),
                         universe, barriers, barrier_rule)
    cent = _centroid_km(universe)
    geom = universe.set_index("USO_AREA_U").geometry
    amounts = _service_amounts(universe, services)

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
                decayed_sum += amounts[svc][j] * contribution_weight(i, j)
            if svc == "road":
                row["road_length_km"] = own
                pcen = (own if roads_formula == "eq4"
                        else own + decayed_sum) / denominator(i)
                row["road_pcen"] = pcen
            else:
                row[f"{svc}_count"] = own
                row[f"{svc}_pcen"] = (own + decayed_sum) / denominator(i)
        rows[i] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    idx_cols = []
    for svc in POINT_SERVICES + ("road",):
        pcen = df[f"{svc}_pcen"]
        lo, hi = pcen.min(), pcen.max()
        df[f"{svc}_idx"] = 0.0 if hi == lo else (pcen - lo) / (hi - lo)
        idx_cols.append(f"{svc}_idx")
    df["psi_eq1"] = df[idx_cols].mean(axis=1)
    if second_norm:
        p = df["psi_eq1"]
        lo, hi = p.min(), p.max()
        df["norm_psi"] = 0.0 if hi == lo else (p - lo) / (hi - lo)
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
