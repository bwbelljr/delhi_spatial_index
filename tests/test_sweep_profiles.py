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
    "decay-none": {"decay.form": "none"},
    "decay-power05": {"decay.form": "inverse_power", "decay.exponent": 0.5},
    "decay-power2": {"decay.form": "inverse_power", "decay.exponent": 2.0},
    "decay-exp2km": {"decay.form": "exponential", "decay.scale_km": 2.0},
    "decay-exp5km": {"decay.form": "exponential", "decay.scale_km": 5.0},
    "decay-boundary": {"decay.distance": "boundary"},
}

# profile -> the dotted `services.point`/`services.line` keys it moves (a
# missing key vs `code-2025` is spelled `None`, since that is what the
# profile file DOES to it — omits it). DEL-40's services-no-ration is the
# first profile whose one factor is `services`, not `methodology`: it must
# be guarded here or it ships unguarded, the same silent-pin failure this
# whole file exists to catch, in a different block (spec § 3).
SERVICE_PROFILES = {
    "services-no-ration": {"point.ration": None},
}

# The union of every one-factor profile this guard knows about, whichever
# block its one factor lives in.
ALL_ONE_FACTOR_PROFILES = sorted(set(SWEEP_PROFILES) | set(SERVICE_PROFILES))

# The profiles whose neighbourhood differs from `code-2025`, so each needs its
# own artifact and must NOT pin a name (the per-profile default keeps two
# points from overwriting each other). Task 1's five ADJACENCY points only —
# decay does not change who the neighbours are, so the six decay profiles
# added below share one artifact instead (SHARED_ARTIFACT, below).
OWN_ARTIFACT = {"adj-touch", "band-0km", "band-1km", "band-5km", "band-10km"}


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


@pytest.mark.parametrize("profile", ALL_ONE_FACTOR_PROFILES)
def test_the_profile_moves_exactly_the_keys_it_claims(profile, base_methodology):
    got = flatten(load_config(profile).methodology)
    moved = {k: v for k, v in got.items() if base_methodology.get(k) != v}
    expected = {k: (str(v) if not isinstance(v, bool) else v)
                for k, v in SWEEP_PROFILES.get(profile, {}).items()}
    assert moved == expected


def flatten_services(cfg):
    """{dotted services key: path}, `point.<name>` / `line.<name>` — the
    services analogue of `flatten()`, kept separate because ServicesConfig's
    fields are plain dicts (source name -> path), not nested dataclasses,
    and because a REMOVED key (services-no-ration's dropped `ration`) needs
    to show up in the diff, which a same-keys dict comparison would miss."""
    out = {}
    for name, path in cfg.services.point.items():
        out[f"point.{name}"] = path
    for name, path in cfg.services.line.items():
        out[f"line.{name}"] = path
    return out


@pytest.fixture(scope="module")
def base_services():
    return flatten_services(load_config(BASE))


@pytest.mark.parametrize("profile", ALL_ONE_FACTOR_PROFILES)
def test_the_profile_moves_exactly_the_services_it_claims(profile, base_services):
    """The third comparison the one-factor guard needed (spec § 3): a
    service-subset profile moves neither `methodology` nor `categories` —
    it moves `services`. Without this, services-no-ration (or any future
    service-subset profile) ships unguarded: a second, unnoticed change to
    its services block would be silently invisible to every other test in
    this file."""
    got = flatten_services(load_config(profile))
    keys = set(base_services) | set(got)
    moved = {k: got.get(k) for k in keys if base_services.get(k) != got.get(k)}
    assert moved == SERVICE_PROFILES.get(profile, {})


@pytest.mark.parametrize("profile", ALL_ONE_FACTOR_PROFILES)
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


SHARED_ARTIFACT = "colonies_neighbors_sweep-bbox.joblib"
DECAY_PROFILES = sorted(set(SWEEP_PROFILES) - OWN_ARTIFACT)


@pytest.mark.parametrize("profile", DECAY_PROFILES)
def test_every_decay_point_pins_the_one_shared_artifact(profile):
    """`methodology_stamp` covers the adjacency and barrier blocks only, so a
    decay change leaves a neighbours artifact valid. Six identical
    preprocesses of an identical neighbourhood would be six identical answers
    at six times the cost, so all six read one file — and
    `check_methodology_stamp` is what makes that safe rather than merely
    conventional: it refuses an artifact whose adjacency or barrier differs.
    """
    assert str(load_config(profile).paths.neighbors_artifact) == SHARED_ARTIFACT


def test_the_shared_artifact_is_not_the_baseline_artifact():
    """A typo here would point the decay sweep at the PROVEN code-2025
    artifact and let a sweep run overwrite the July 2025 correctness proof."""
    assert SHARED_ARTIFACT != str(load_config(BASE).paths.neighbors_artifact)
