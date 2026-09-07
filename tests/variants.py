"""The variant table: one row per Phase-6-shaped configuration.

Imports NOTHING from this repo (like tests/cities.py). BOTH sides read it:
`tests/reference_impl.py` builds VARIANT_RULESETS from it, and the
production-side tests build a MethodologyConfig from it. Every value it
names — `within_distance`, the four decay forms, `centroid`/`boundary` —
has the SAME spelling in the config vocabulary and in the reference's
keyword vocabulary, so there is no translation layer to get wrong; a
variant that one day needs `touch`/`bbox` will add one then.

A variant states each block it overrides IN FULL, because the CLI round trip
replaces `methodology.<block>` wholesale: no key is ever inherited. A
variant that does not override `adjacency` at all keeps the `code` base's
`bbox`.

The band constants live here for the same reason the table does: the fixture
generators' guard (scripts/check_oraculum_invariants.check_bands) and the
reference pins (tests/test_variant_rules.py) must not hold two copies of
them. All distances are kilometres; pairs are sorted 2-tuples of settlement
ids and are UNDIRECTED and PRE-BARRIER — they are counted on the adjacency
function's own output, never on anything downstream of a barrier rule.
"""

VARIANTS = {
    # X = 0 is intersection-inclusive: every polygon that intersects i,
    # corner-only touches and overlaps included. With `none` the weights are
    # all 1, so the pcen is a plain sum of counts.
    "band0_none": {
        "adjacency": {"rule": "within_distance", "max_distance_km": 0.0},
        "decay": {"form": "none", "distance": "centroid",
                  "distance_unit": "km"},
    },
    # B1 and B2 sit in a gap of BOTH cities' polygon-to-polygon distance
    # lists (>= 26 m clearance), so no pair lands on the `<=` boundary.
    "band_small": {
        "adjacency": {"rule": "within_distance", "max_distance_km": 0.25},
        "decay": {"form": "inverse_linear", "distance": "centroid",
                  "distance_unit": "km"},
    },
    "band_large": {
        "adjacency": {"rule": "within_distance", "max_distance_km": 0.75},
        "decay": {"form": "inverse_linear", "distance": "centroid",
                  "distance_unit": "km"},
    },
    # The only variant where Oraculum can tell boundary from centroid:
    # A-RV and C-RV are 0.100 km apart at the boundary and 1.414214 km apart
    # at the centroids.
    "band_small_boundary": {
        "adjacency": {"rule": "within_distance", "max_distance_km": 0.25},
        "decay": {"form": "inverse_linear", "distance": "boundary",
                  "distance_unit": "km"},
    },
    "pow1": {
        "decay": {"form": "inverse_power", "exponent": 1,
                  "distance": "centroid", "distance_unit": "km"},
    },
    "pow2": {
        "decay": {"form": "inverse_power", "exponent": 2,
                  "distance": "centroid", "distance_unit": "km"},
    },
    "exp1": {
        "decay": {"form": "exponential", "scale_km": 1.0,
                  "distance": "centroid", "distance_unit": "km"},
    },
    # Degenerate on Oraculum (every `code` neighbour there is in contact, so
    # every weight is 1); messy's G-M pair carries the proof.
    "boundary": {
        "decay": {"form": "inverse_linear", "distance": "boundary",
                  "distance_unit": "km"},
    },
    # DEL-48: the only genuinely fractional weight either fixture city can
    # produce. Oraculum's canal is [25, 475] on the 500 m A-D edge; its 5 m
    # round-capped buffer covers [20, 480], so w_AD = 1 - 460/500 = 0.08 and
    # every other link is untouched (spec § 6.1). Degenerate on the messy
    # city — it has no barriers, so every weight is 1 and its rows equal the
    # `code` base, exactly as `boundary` is degenerate on Oraculum.
    "partial_5m": {
        "barrier": {"rule": "partial_weighted", "combine": "any",
                    "buffer_m": 5.0},
    },
    # DEL-20: a neighbour lends only |S_j \ S_i|. Degenerate on Oraculum —
    # no overlapping polygons and no point inside two settlements, so the
    # shared structure is empty and the rows equal the `code` base, exactly
    # as `boundary` is degenerate there. The messy city carries the pin: its
    # O1/O2 clinic is inside both, so O1's clinic PCEN falls from
    # (1 + 1/1.8)/600 to 1/600 while its school PCEN does not move at all
    # (spec § 6.2).
    "overlap_outside": {
        "overlap": {"lending": "outside_receiver"},
    },
    # Both 3E switches at once: proves the two kwargs are accepted together
    # and both paths run in one compute_frames/compute_city call. It equals
    # `partial_5m` on Oraculum (no overlaps) and `overlap_outside` on the
    # messy city (no barriers) — no fixture city has a barrier across an
    # overlap, so the two multipliers are only simultaneously non-trivial in
    # the synthetic in-test geometry (spec § 5, § 6.4).
    "partial_5m_outside": {
        "barrier": {"rule": "partial_weighted", "combine": "any",
                    "buffer_m": 5.0},
        "overlap": {"lending": "outside_receiver"},
    },
    # DEL-55: the decay CONSTANTS the Phase 6 sweep uses that 3D never pinned.
    # Same code paths as pow1/pow2/exp1 at different numbers, so the risk is
    # low — but the rule since 3D is that a methodology value ships as a row
    # here, scored by both implementations and compared at 1e-12.
    "decay_none": {
        "decay": {"form": "none", "distance": "centroid",
                  "distance_unit": "km"},
    },
    "pow05": {
        "decay": {"form": "inverse_power", "exponent": 0.5,
                  "distance": "centroid", "distance_unit": "km"},
    },
    "exp2": {
        "decay": {"form": "exponential", "scale_km": 2.0,
                  "distance": "centroid", "distance_unit": "km"},
    },
    "exp5": {
        "decay": {"form": "exponential", "scale_km": 5.0,
                  "distance": "centroid", "distance_unit": "km"},
    },
    # DEL-55: the sweep's three real radii. Unlike 0.25 and 0.75 km, these were
    # NOT chosen to avoid the `<=` boundary — they are the methodological
    # points DEL-36 asks for, and the Delhi run uses them as they are. Both
    # cities distinguish all three (Oraculum spans 4.0 x 3.0 km, messy
    # 21.0 x 2.4 km; undirected pairs 16/21/21 and 13/27/48), and 1 km lands
    # EXACTLY on the boundary: two pairs at 1000.000 m in Oraculum, one in
    # messy, one more at 10 km in messy. That is the point of pinning them —
    # if the two implementations ever disagree about a distance sitting
    # exactly on the radius, it surfaces here rather than on 4,357 settlements.
    "band_1km": {
        "adjacency": {"rule": "within_distance", "max_distance_km": 1.0},
        "decay": {"form": "inverse_linear", "distance": "centroid",
                  "distance_unit": "km"},
    },
    "band_5km": {
        "adjacency": {"rule": "within_distance", "max_distance_km": 5.0},
        "decay": {"form": "inverse_linear", "distance": "centroid",
                  "distance_unit": "km"},
    },
    "band_10km": {
        "adjacency": {"rule": "within_distance", "max_distance_km": 10.0},
        "decay": {"form": "inverse_linear", "distance": "centroid",
                  "distance_unit": "km"},
    },
    # DEL-34: the index transforms — alternatives to today's `none` for the
    # compression Eq. 2's per-service min-max produces on a heavily
    # right-skewed PCEN distribution (spec § 1). Neither shipped profile
    # adopts one; this is measurement machinery, scored by both
    # implementations like every other methodology value.
    "transform_log1p_pcen": {
        "transform": {"form": "log1p", "stage": "pcen"},
    },
    "transform_log1p_psi": {
        "transform": {"form": "log1p", "stage": "psi"},
    },
    "transform_cbrt_pcen": {
        "transform": {"form": "cbrt", "stage": "pcen"},
    },
    "transform_cbrt_psi": {
        "transform": {"form": "cbrt", "stage": "psi"},
    },
}

BAND_RADII_KM = (0.0, 0.25, 0.75, 1.0, 5.0, 10.0)

# Undirected pair counts on adjacency()'s own output, BEFORE any barrier.
EXPECTED_BAND_PAIRS = {
    "oraculum": {0.0: 10, 0.25: 12, 0.75: 14, 1.0: 16, 5.0: 21, 10.0: 21},
    "messy": {0.0: 5, 0.25: 8, 0.75: 10, 1.0: 13, 5.0: 27, 10.0: 48},
}

# Exactly which pairs each radius adds to the one below it.
ADDED_BAND_PAIRS = {
    "oraculum": {
        0.25: {("A", "RV"), ("C", "RV")},          # 0.100 km each
        0.75: {("B", "D"), ("B", "IND")},          # 0.500 km each
        # BOTH pairs are at EXACTLY 1000.000 m — the `<=` boundary. See
        # test_a_pair_exactly_at_the_radius_is_a_neighbour.
        1.0: {("A", "C"), ("E", "RV")},            # 1.000 km each, ON the edge
        5.0: {("A", "IND"), ("C", "D"),            # 1.500 km each
              ("D", "IND"),                        # 2.000 km
              ("D", "RV"), ("IND", "RV")},         # 1.166190379 km each
        # Nothing: 21 pairs IS the complete graph on 7 settlements, so the
        # city cannot distinguish 5 km from 10 km. The empty set is the pin.
        10.0: set(),
    },
    "messy": {
        0.25: {("H", "L"), ("H", "T"), ("L", "S")},   # 0.131519/0.223607/0.199
        0.75: {("G", "M"), ("S", "T")},               # 0.450 / 0.630242
        # M-U is at EXACTLY 1000.000 m — the `<=` boundary.
        1.0: {("M", "U"),                             # 1.000 km, ON the edge
              ("N", "O1"), ("O2", "U")},              # 0.800 km each
        5.0: {("G", "H"), ("G", "L"), ("G", "O1"), ("G", "O2"), ("G", "T"),
              ("G", "U"), ("H", "M"), ("L", "M"), ("M", "N"), ("M", "O1"),
              ("M", "O2"), ("M", "S"), ("M", "T"), ("N", "U")},
        # I-U is at EXACTLY 10000.000 m — the `<=` boundary again.
        10.0: {("G", "N"), ("G", "S"), ("H", "N"), ("H", "O1"), ("H", "O2"),
               ("H", "U"), ("I", "N"), ("I", "O1"), ("I", "O2"),
               ("I", "U"),                            # 10.000 km, ON the edge
               ("L", "N"), ("L", "O1"), ("L", "O2"), ("L", "U"), ("N", "T"),
               ("O1", "S"), ("O1", "T"), ("O2", "S"), ("O2", "T"), ("S", "U"),
               ("T", "U")},
    },
}
