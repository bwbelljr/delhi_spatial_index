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
ORACLE_SCENARIOS = [
    # (reference scenario, exclusion.types, exclusion.stage)
    ("baseline", (), "post_neighbors"),
    ("excl_rv_only", ("RV",), "post_neighbors"),
    ("excl_contributing", ("RV", "IND"), "post_neighbors"),
    ("excl_removed", ("RV", "IND"), "pre_neighbors"),
    ("excl_ind_removed", ("IND",), "pre_neighbors"),
]


def methodology_with(profile, *, types=None, stage=None):
    """The shipped profile's methodology with the two allowed overrides."""
    from dataclasses import replace

    from delhi_psi.config import ExclusionStage, load_config

    methodology = load_config(profile).methodology
    exclusion = methodology.exclusion
    if types is not None:
        exclusion = replace(exclusion, types=tuple(types))
    if stage is not None:
        exclusion = replace(exclusion, stage=ExclusionStage(stage))
    return replace(methodology, exclusion=exclusion)


def compute_oracle_frame(profile, *, types, stage, denom):
    """compute_frames on the Oraculum city, indexed by settlement id.

    The fixture city carries its own `population` column, so population=None.
    """
    from delhi_psi.pipeline import compute_frames

    return compute_frames(
        load_settlements(), {"canal": load_barriers()}, load_services(),
        None, methodology_with(profile, types=types, stage=stage), denom,
    ).set_index("USO_AREA_U")
