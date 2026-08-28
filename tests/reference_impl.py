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
scenario, denom, second_norm, absent_neighbor_contribution. RULESETS binds
the ideal (manuscript) and code (empirical) combinations.
"""

import pandas as pd
from shapely.geometry import box

from tests.cities import ORACULUM

RULESETS = {
    "ideal": dict(adjacency_rule="border", barrier_rule="pair",
                  roads_formula="eq4", second_norm=False,
                  absent_neighbor_contribution="contributes"),
    "code": dict(adjacency_rule="bbox", barrier_rule="global",
                 roads_formula="decayed", second_norm=True,
                 absent_neighbor_contribution="swallowed"),
}

# Backward-compatible view of Oraculum's table in the 2-tuple shape this
# module has always consumed: {name: (dropped ids, dropped_before_neighbors)}.
# ORACULUM.scenarios' ORDER is today's order, which fixes expected_values.csv.
# (tests/cities.py imports geopandas and nothing from this repo, so the
# INDEPENDENCE RULE is intact: it is fixture plumbing, not index math.)
SCENARIOS = {s.name: (s.dropped, s.dropped_before_neighbors)
             for s in ORACULUM.scenarios}

POINT_SERVICES = ("clinic", "school", "bank", "police", "ration", "transport")


def adjacency(settlements, rule):
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


def compute_city(settlements, services, barriers, *, adjacency_rule,
                 barrier_rule, roads_formula, scenario, denom, second_norm,
                 absent_neighbor_contribution, scenarios=None):
    # `scenarios` defaults to the module table, so every existing call keeps
    # working; a caller may pass its own WITHOUT mutating the global (which
    # is what scripts/render_oracle_maps.py used to do).
    table = SCENARIOS if scenarios is None else scenarios
    dropped, drop_before = table[scenario]
    universe = settlements[~settlements["USO_AREA_U"].isin(dropped)] \
        if drop_before else settlements

    nbrs = apply_barrier(adjacency(universe, adjacency_rule),
                         universe, barriers, barrier_rule)
    cent = _centroid_km(universe)
    amounts = _service_amounts(universe, services)

    indexed = [i for i in universe["USO_AREA_U"]
               if drop_before or i not in dropped]
    meta = universe.set_index("USO_AREA_U")

    def denominator(i):
        pop = meta.loc[i, "population"]
        return pop / meta.loc[i, "area_km2"] if denom == "popdensity" else pop

    def contribution_weight(i, j):
        d_km = cent[i].distance(cent[j]) / 1000
        return 1 / (1 + d_km)

    rows = {}
    for i in indexed:
        row = {}
        for svc in POINT_SERVICES + ("road",):
            own = amounts[svc][i]
            decayed_sum = 0.0
            for j in nbrs[i]:
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


if __name__ == "__main__":
    from tests.cities import CITIES

    for target_city in CITIES:
        target = target_city.fixtures / "expected_values.csv"
        emit_expected_values(target, target_city)
        print(f"wrote {target}")
