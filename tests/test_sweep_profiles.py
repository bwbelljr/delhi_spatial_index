"""Every sweep profile is `code-2025` with exactly the keys it claims moved.

A Phase 6 sweep point is only interpretable if its diff against the baseline
is ONE factor. A profile that quietly also changed, say, `roads` would produce
a number nobody could attribute, and no other test in this repo would notice:
the production fixtures pin what the profile DOES, not what it differs from.
"""
import dataclasses

import pytest

from delhi_psi.config import load_config

BASE = "code-2025"

# profile -> the dotted methodology keys it moves, and their values.
# Task 2 appends the six decay profiles here.
SWEEP_PROFILES = {
    "adj-touch": {"adjacency.rule": "touch"},
    "band-0km": {"adjacency.rule": "within_distance",
                 "adjacency.max_distance_km": 0.0},
    "band-1km": {"adjacency.rule": "within_distance",
                 "adjacency.max_distance_km": 1.0},
    "band-5km": {"adjacency.rule": "within_distance",
                 "adjacency.max_distance_km": 5.0},
    "band-10km": {"adjacency.rule": "within_distance",
                  "adjacency.max_distance_km": 10.0},
}

# The profiles whose neighbourhood differs from `code-2025`, so each needs its
# own artifact and must NOT pin a name (the per-profile default keeps two
# points from overwriting each other).
OWN_ARTIFACT = set(SWEEP_PROFILES)


def flatten(obj, prefix=""):
    """{dotted key: comparable value} over a nested dataclass."""
    if dataclasses.is_dataclass(obj):
        out = {}
        for field in dataclasses.fields(obj):
            out.update(flatten(getattr(obj, field.name), f"{prefix}{field.name}."))
        return out
    key = prefix.rstrip(".")
    if isinstance(obj, (list, tuple)):
        return {key: tuple(str(v) for v in obj)}
    # Enum members are str-valued; compare on the string so a test table can
    # be written in plain YAML vocabulary.
    return {key: str(obj) if not isinstance(obj, bool) else obj}


@pytest.fixture(scope="module")
def base_methodology():
    return flatten(load_config(BASE).methodology)


@pytest.mark.parametrize("profile", sorted(SWEEP_PROFILES))
def test_the_profile_moves_exactly_the_keys_it_claims(profile, base_methodology):
    got = flatten(load_config(profile).methodology)
    moved = {k: v for k, v in got.items() if base_methodology.get(k) != v}
    expected = {k: (str(v) if not isinstance(v, bool) else v)
                for k, v in SWEEP_PROFILES[profile].items()}
    assert moved == expected


@pytest.mark.parametrize("profile", sorted(SWEEP_PROFILES))
def test_the_profile_copies_the_baseline_categories_verbatim(profile):
    """Each sweep profile hand-copies the ten-category identity mapping, and
    the one-factor guard above only inspects `methodology`. A typo in a
    category name would change which settlements are excluded — a second
    factor, invisible to every other test in the repo."""
    base = load_config(BASE).categories
    got = load_config(profile).categories
    assert got.mapping == base.mapping
    assert got.scheme == base.scheme


@pytest.mark.parametrize("profile", sorted(SWEEP_PROFILES))
def test_the_profile_reports_one_denominator_as_csv(profile):
    cfg = load_config(profile)
    assert [str(d) for d in cfg.outputs.denominators] == ["popdensity"]
    assert [str(f) for f in cfg.outputs.formats] == ["csv"]


@pytest.mark.parametrize("profile", sorted(SWEEP_PROFILES))
def test_a_profile_with_its_own_neighbourhood_does_not_pin_an_artifact(profile):
    if profile not in OWN_ARTIFACT:
        pytest.skip("shares the bbox artifact; Task 2 pins that case")
    cfg = load_config(profile)
    assert str(cfg.paths.neighbors_artifact) == f"colonies_neighbors_{profile}.joblib"
