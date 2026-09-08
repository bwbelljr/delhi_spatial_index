"""Loaders for the Oraculum fixtures.

Every public name here is now a thin wrapper over `tests/cities.py`'s
city-taking functions with `city=ORACULUM` bound, so no existing test
changes its call shape or its expected value (spec 3C § 3).
"""

from pathlib import Path

import geopandas as gpd

from tests.cities import ORACULUM

FIXTURES = ORACULUM.fixtures
EPSG = ORACULUM.epsg


def _read(path):
    gdf = gpd.read_file(path)
    return gdf.set_crs(epsg=EPSG, allow_override=True)


def load_settlements(city=ORACULUM):
    return city.load_settlements()


def load_barriers(city=ORACULUM):
    return city.load_barriers()


def load_services(city=ORACULUM):
    return city.load_services()


def load_exhibit():
    """The divergence exhibit is Oraculum-only (spec § 4.5: unchanged)."""
    return _read(ORACULUM.fixtures / "divergence" / "exhibit.geojson")


# --- profile-driven helpers (Phase 3A) --------------------------------
# The § 7 scenario table: a profile plus `types`/`stage` overrides ONLY.
# `absent_neighbor` always comes from the profile, because in the reference
# it is a rule-set property, not a scenario property.
#
# 3B: `types` entries are CATEGORY names. They select the right rows here
# only because the derived oracle profile's mapping is the IDENTITY over the
# fixture vocabulary — and the reference implementation drops settlements by
# ID, which agrees only because the fixture gives RV and IND ids equal to
# their types. Both facts are pinned in tests/test_fixture_invariants.py.
# 3C: the 3-tuple view of ORACULUM.scenarios. Its LIST ORDER now follows
# reference_impl.SCENARIOS rather than today's hand-written order; only
# pytest collection order is affected, because
# generate_production_fixtures.write_fixture sorts by
# (scenario, denom, settlement, metric) before writing.
ORACLE_SCENARIOS = [(s.name, s.exclusion_types, s.stage)
                    for s in ORACULUM.scenarios]


# --- the derived test-only profile (spec 3B § 2) ----------------------
# The oracle city is NOT Delhi: its vocabulary is not covered by the shipped
# profiles' `uso-10` mapping, and padding those with `UC`/`IND` would blunt
# the unmapped-type guard on real data. Every test that runs the CLI or the
# oracle helpers on this city therefore uses a DERIVED profile whose mapping
# is the identity over the fixture vocabulary. The single exception is the
# test that proves the guard fires (tests/test_cli.py).
ORACLE_SCHEME = ORACULUM.scheme
ORACLE_VOCABULARY = ORACULUM.vocabulary


def oracle_mapping(city=ORACULUM):
    """The identity over the fixture city's source types."""
    return city.mapping()


def oracle_config(base, city=ORACULUM):
    """`base`'s shipped Config with the oracle-6 identity categories block.

    Purely in memory — no file, no pytest fixture — because
    scripts/generate_production_fixtures.py reaches it through
    `compute_oracle_frame` and runs as a plain script outside pytest, where
    there is no `tmp_path`.

    Precondition: `base`'s `exclusion.types` must be category names present
    in the oracle vocabulary (`ORACLE_VOCABULARY`, since the swapped-in
    mapping is its identity) — the shipped profiles satisfy this. A profile
    whose `exclusion.types` names a category the oracle-6 identity does not
    produce (e.g. a collapsing profile's `non-urban`) fails to load through
    this helper with the "not categories produced by categories.mapping"
    error.
    """
    from dataclasses import replace

    from delhi_psi.config import CategoriesConfig, load_config

    return replace(load_config(base),
                   categories=CategoriesConfig(scheme=city.scheme,
                                               mapping=city.mapping()))


def oracle_profile_path(base, directory, city=ORACULUM, *,
                        methodology_overrides=None, name=None):
    """Write the derived profile as YAML into `directory`; return the path,
    for the tests that drive the real CLI with `--config <path>`.

    `methodology_overrides` is a mapping of TOP-LEVEL methodology sub-blocks
    (`adjacency`, `decay`, `exclusion`) that REPLACE `raw["methodology"]
    [<block>]` wholesale — never a deep merge. A variant therefore always
    states its full block and no key is ever inherited from the base
    profile, which is what `tests/variants.py` is written for.

    `name` distinguishes two derived profiles written into ONE directory;
    without it the filename is the historic `<base>.oracle.yaml`.

    Precondition: `base`'s `exclusion.types` (or the override's) must be
    category names present in the oracle vocabulary (`ORACLE_VOCABULARY`,
    since the swapped-in mapping is its identity) — the shipped profiles
    satisfy this. A profile whose `exclusion.types` names a category the
    oracle-6 identity does not produce (e.g. a collapsing profile's
    `non-urban`) fails to load through this helper with the "not categories
    produced by categories.mapping" error.
    """
    import yaml

    from delhi_psi.config import PROFILES_DIR

    raw = yaml.safe_load((PROFILES_DIR / f"{base}.yaml").read_text())
    raw["categories"] = {"scheme": city.scheme, "mapping": city.mapping()}
    for block, values in (methodology_overrides or {}).items():
        raw["methodology"][block] = dict(values)
    stem = base if name is None else f"{base}.{name}"
    path = Path(directory) / f"{stem}.oracle.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def methodology_with(profile, *, types=None, stage=None, city=ORACULUM):
    """The DERIVED profile's methodology with the two allowed overrides.

    It reads only `.methodology`, so the derived categories block leaves the
    result identical to the shipped profile's.
    """
    from dataclasses import replace

    from delhi_psi.config import ExclusionStage

    methodology = oracle_config(profile, city).methodology
    exclusion = methodology.exclusion
    if types is not None:
        exclusion = replace(exclusion, types=tuple(types))
    if stage is not None:
        exclusion = replace(exclusion, stage=ExclusionStage(stage))
    return replace(methodology, exclusion=exclusion)


def variant_methodology(base, variant, *, city=ORACULUM, types=None,
                        stage=None):
    """`base`'s methodology with `tests/variants.py`'s `variant` applied.

    Layered on `methodology_with`, so the SCENARIO travels with it: pass the
    scenario's `types`/`stage`. Without that, `code-2025`'s own
    `exclusion.types: [RV]` would drop RV (and messy's N) from the production
    frame while the variants CSV keeps them — and RV is the settlement the
    § 4.1 pins name.

    A block the variant does not mention keeps `base`'s: today the band
    variants override `adjacency`, `partial_5m` overrides `barrier`,
    `overlap_outside` overrides `overlap` and `partial_5m_outside` overrides
    both, and every variant states each block it does override IN FULL.
    """
    from dataclasses import replace

    from delhi_psi.config import (
        AdjacencyConfig, AdjacencyRule, AggregationConfig, AggregationRule,
        BarrierConfig, BarrierRule, DecayConfig, DecayDistance, DecayForm,
        OverlapConfig, OverlapLending, TransformConfig, TransformForm,
        TransformStage,
    )
    from tests.variants import VARIANTS

    methodology = methodology_with(base, types=types, stage=stage, city=city)
    spec = VARIANTS[variant]
    if "adjacency" in spec:
        block = spec["adjacency"]
        methodology = replace(methodology, adjacency=AdjacencyConfig(
            rule=AdjacencyRule(block["rule"]),
            max_distance_km=block.get("max_distance_km")))
    if "barrier" in spec:
        block = spec["barrier"]
        methodology = replace(methodology, barrier=BarrierConfig(
            rule=BarrierRule(block["rule"]),
            combine=block["combine"],
            buffer_m=block.get("buffer_m")))
    if "overlap" in spec:
        block = spec["overlap"]
        methodology = replace(methodology, overlap=OverlapConfig(
            lending=OverlapLending(block["lending"])))
    if "decay" in spec:
        block = spec["decay"]
        methodology = replace(methodology, decay=DecayConfig(
            form=DecayForm(block["form"]),
            distance_unit=block["distance_unit"],
            distance=DecayDistance(block["distance"]),
            exponent=block.get("exponent"),
            scale_km=block.get("scale_km")))
    if "transform" in spec:
        block = spec["transform"]
        methodology = replace(methodology, transform=TransformConfig(
            form=TransformForm(block["form"]),
            stage=(TransformStage(block["stage"])
                  if "stage" in block else None)))
    if "aggregation" in spec:
        block = spec["aggregation"]
        methodology = replace(methodology, aggregation=AggregationConfig(
            rule=AggregationRule(block["rule"])))
    return methodology


# --- service-subset filtering (DEL-40 spec § 1-2) -----------------------
# The reference config's service key differs from the fixture city's group
# name in exactly one place: config `health` names the file the fixture
# writes as its `clinic` group (tests/test_cli.py::SERVICE_LAYOUT writes the
# `clinic` layer to Public Services/Health/Health.shp, under the config's
# `health` key). Every other service name is the identity between the two.
CONFIG_TO_FIXTURE_SERVICE = {"health": "clinic"}


def configured_fixture_services(cfg):
    """The fixture-service names (`city.load_services()` keys) that `cfg`
    configures: its `services.point`/`services.line` keys, translated
    through CONFIG_TO_FIXTURE_SERVICE. A profile that drops a service from
    `services.point` (e.g. `services-no-ration`) is what makes this return
    fewer than all seven names — that subset is what compute_oracle_frame
    filters the fixture city's services down to.
    """
    keys = {**cfg.services.point, **cfg.services.line}
    return {CONFIG_TO_FIXTURE_SERVICE.get(name, name) for name in keys}


def compute_oracle_frame(profile, *, types, stage, denom, city=ORACULUM):
    """compute_frames on the Oraculum city under the DERIVED profile's own
    category mapping, indexed by settlement id.

    The fixture city carries its own `population` column, so population=None.
    Passing the profile's mapping (not letting compute_frames default to the
    identity) is what makes a future COLLAPSING profile's fixture record the
    numbers the CLI actually produces; under today's identity profiles it is
    a no-op.

    Services are filtered to `cfg.services` (DEL-40 spec § 1-2): passing
    `city.load_services()` unconditionally would emit full-service numbers
    for a profile that subsets services, a silent pin.
    """
    from delhi_psi.pipeline import compute_frames

    cfg = oracle_config(profile, city)
    configured = configured_fixture_services(cfg)
    services = {name: gdf for name, gdf in city.load_services().items()
               if name in configured}
    return compute_frames(
        city.load_settlements(), {"canal": city.load_barriers()},
        services, None,
        methodology_with(profile, types=types, stage=stage, city=city),
        denom, mapping=cfg.categories.mapping, scheme=cfg.categories.scheme,
    ).set_index("USO_AREA_U")
