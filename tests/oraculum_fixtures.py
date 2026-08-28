"""Loaders for the Oraculum fixtures."""

from pathlib import Path

import geopandas as gpd

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "oraculum"
EPSG = 7760


def _read(path):
    gdf = gpd.read_file(path)
    return gdf.set_crs(epsg=EPSG, allow_override=True)


def load_settlements():
    return _read(FIXTURES / "settlements.geojson")


def load_barriers():
    return _read(FIXTURES / "barriers.geojson")


def load_services():
    gdf = _read(FIXTURES / "services.geojson")
    return {name: grp.reset_index(drop=True) for name, grp in gdf.groupby("service")}


def load_exhibit():
    return _read(FIXTURES / "divergence" / "exhibit.geojson")


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
ORACLE_SCENARIOS = [
    # (reference scenario, exclusion.types, exclusion.stage)
    ("baseline", (), "post_neighbors"),
    ("excl_rv_only", ("RV",), "post_neighbors"),
    ("excl_contributing", ("RV", "IND"), "post_neighbors"),
    ("excl_removed", ("RV", "IND"), "pre_neighbors"),
    ("excl_ind_removed", ("IND",), "pre_neighbors"),
]


# --- the derived test-only profile (spec 3B § 2) ----------------------
# The oracle city is NOT Delhi: its vocabulary is not covered by the shipped
# profiles' `uso-10` mapping, and padding those with `UC`/`IND` would blunt
# the unmapped-type guard on real data. Every test that runs the CLI or the
# oracle helpers on this city therefore uses a DERIVED profile whose mapping
# is the identity over the fixture vocabulary. The single exception is the
# test that proves the guard fires (tests/test_cli.py).
ORACLE_SCHEME = "oracle-6"
ORACLE_VOCABULARY = ("Planned", "UC", "JJC", "RV", "RUAC", "IND")


def oracle_mapping():
    """The identity over the fixture city's six source types."""
    return {source: source for source in ORACLE_VOCABULARY}


def oracle_config(base):
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
                   categories=CategoriesConfig(scheme=ORACLE_SCHEME,
                                               mapping=oracle_mapping()))


def oracle_profile_path(base, directory):
    """Write the same derived profile as YAML into `directory`; return the
    path, for the tests that drive the real CLI with `--config <path>`.

    Precondition: `base`'s `exclusion.types` must be category names present
    in the oracle vocabulary (`ORACLE_VOCABULARY`, since the swapped-in
    mapping is its identity) — the shipped profiles satisfy this. A profile
    whose `exclusion.types` names a category the oracle-6 identity does not
    produce (e.g. a collapsing profile's `non-urban`) fails to load through
    this helper with the "not categories produced by categories.mapping"
    error.
    """
    import yaml

    from delhi_psi.config import PROFILES_DIR

    raw = yaml.safe_load((PROFILES_DIR / f"{base}.yaml").read_text())
    raw["categories"] = {"scheme": ORACLE_SCHEME, "mapping": oracle_mapping()}
    path = Path(directory) / f"{base}.oracle.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def methodology_with(profile, *, types=None, stage=None):
    """The DERIVED profile's methodology with the two allowed overrides.

    It reads only `.methodology`, so the derived categories block leaves the
    result identical to the shipped profile's.
    """
    from dataclasses import replace

    from delhi_psi.config import ExclusionStage

    methodology = oracle_config(profile).methodology
    exclusion = methodology.exclusion
    if types is not None:
        exclusion = replace(exclusion, types=tuple(types))
    if stage is not None:
        exclusion = replace(exclusion, stage=ExclusionStage(stage))
    return replace(methodology, exclusion=exclusion)


def compute_oracle_frame(profile, *, types, stage, denom):
    """compute_frames on the Oraculum city under the DERIVED profile's own
    category mapping, indexed by settlement id.

    The fixture city carries its own `population` column, so population=None.
    Passing the profile's mapping (not letting compute_frames default to the
    identity) is what makes a future COLLAPSING profile's fixture record the
    numbers the CLI actually produces; under today's identity profiles it is
    a no-op.
    """
    from delhi_psi.pipeline import compute_frames

    cfg = oracle_config(profile)
    return compute_frames(
        load_settlements(), {"canal": load_barriers()}, load_services(),
        None, methodology_with(profile, types=types, stage=stage), denom,
        mapping=cfg.categories.mapping, scheme=cfg.categories.scheme,
    ).set_index("USO_AREA_U")
