"""Config schema: defaults, enum validation, reserved values, precedence.

Spec § 3. The reference-pinned enums are generated from ONE table
(config.REFERENCE_KNOBS); tests/test_profiles_match_reference.py reads the
same table, so a value without a reference knob cannot be added silently.
"""
from pathlib import Path

import pytest

from delhi_psi.categories import categories_of
from delhi_psi.config import (
    ENUMS, ENUM_KEYS, REFERENCE_KNOBS, RESERVED_KEYS, RESERVED_VALUES,
    Config, ConfigError, load_config, shipped_profiles,
)

CATEGORIES_BLOCK = """categories:
  scheme: uso-10
  mapping:
    Planned: Planned
    UAC: UAC
    JJC: JJC
    RUAC: RUAC
    RV: RV
    UV: UV
    SDA: SDA
    JJR: JJR
    Industrial: Industrial
    Other: Other
"""

MINIMAL = """profile: minimal
""" + CATEGORIES_BLOCK + """methodology:
  adjacency: {rule: bbox}
  barrier: {rule: global_asymmetric, combine: any}
  decay: {form: inverse_linear, distance_unit: km}
  roads: decayed
  second_normalization: true
  exclusion: {types: [RV], stage: post_neighbors, absent_neighbor: swallowed}
"""

# The 10 Delhi source types, measured 27 Aug 2026 (spec 3B § 1).
DELHI_TYPES = {"Planned", "UAC", "JJC", "RUAC", "RV", "UV", "SDA", "JJR",
               "Industrial", "Other"}


def write(tmp_path, text, name="p.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_both_profiles_ship():
    assert sorted(shipped_profiles()) == ["code-2025", "manuscript"]


def test_profile_loads_by_name_and_by_path(tmp_path):
    by_name = load_config("code-2025", data_dir=str(tmp_path))
    from delhi_psi.config import PROFILES_DIR
    by_path = load_config(PROFILES_DIR / "code-2025.yaml",
                          data_dir=str(tmp_path))
    assert isinstance(by_name, Config)
    assert by_name == by_path


def test_defaults_equal_code_2025(tmp_path):
    """Every non-methodology key defaults to the code-2025 value (§ 3)."""
    minimal = load_config(write(tmp_path, MINIMAL), data_dir=str(tmp_path))
    full = load_config("code-2025", data_dir=str(tmp_path))
    assert minimal.profile == "minimal" and full.profile == "code-2025"
    assert minimal.crs == full.crs
    assert minimal.layers == full.layers
    assert minimal.services == full.services
    assert minimal.validate == full.validate
    assert minimal.outputs == full.outputs
    # paths.neighbors_artifact is the one defaulted key that is NOT the
    # code-2025 value: the default is profile-specific so two profiles cannot
    # overwrite each other's artifact, while code-2025 pins the historic
    # filename explicitly (the real-data proof depends on it).
    assert minimal.paths.neighbors_artifact == "colonies_neighbors_minimal.joblib"
    assert full.paths.neighbors_artifact == "colonies_neighbors.joblib"
    # methodology was written out in full, so it matches too
    assert minimal.methodology == full.methodology


def test_methodology_is_required_in_full(tmp_path):
    partial = MINIMAL.replace("  roads: decayed\n", "")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, partial))
    assert "methodology.roads" in str(exc.value)


def test_profile_key_is_required(tmp_path):
    without = MINIMAL.replace("profile: minimal\n", "")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, without))
    assert "profile" in str(exc.value)


def test_unknown_key_is_rejected(tmp_path):
    text = MINIMAL + "\nnonsense: 1\n"
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    assert "nonsense" in str(exc.value)


def test_unknown_nested_key_is_rejected(tmp_path):
    text = MINIMAL.replace("  roads: decayed",
                           "  roads: decayed\n  bogus_knob: 1")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    assert "methodology.bogus_knob" in str(exc.value)


@pytest.mark.parametrize("key,bad", [
    ("methodology.adjacency.rule", "  adjacency: {rule: diagonal}"),
    ("methodology.barrier.rule", "  barrier: {rule: sideways, combine: any}"),
    ("methodology.roads", "  roads: sideways"),
    ("methodology.exclusion.stage",
     "  exclusion: {types: [RV], stage: midway, absent_neighbor: swallowed}"),
    ("methodology.exclusion.absent_neighbor",
     "  exclusion: {types: [RV], stage: post_neighbors, absent_neighbor: maybe}"),
])
def test_out_of_enum_names_key_and_allowed_values(tmp_path, key, bad):
    line_start = bad.split(":")[0]
    text = "\n".join(bad if line.startswith(line_start) else line
                     for line in MINIMAL.splitlines())
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert key in message
    for allowed in REFERENCE_KNOBS[key]:
        assert str(allowed) in message


def test_bad_denominator_names_key_and_allowed_values(tmp_path):
    text = MINIMAL + "\noutputs: {denominators: [households]}\n"
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    assert "outputs.denominators" in str(exc.value)
    assert "pop" in str(exc.value) and "popdensity" in str(exc.value)


def test_reserved_partial_weighted(tmp_path):
    text = MINIMAL.replace("  barrier: {rule: global_asymmetric, combine: any}",
                           "  barrier: {rule: partial_weighted, combine: any}")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    assert str(exc.value).endswith(
        RESERVED_VALUES["methodology.barrier.rule"]["partial_weighted"])


def test_reserved_denominator_one(tmp_path):
    text = MINIMAL + "\noutputs: {denominators: [one]}\n"
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    assert str(exc.value).endswith(
        RESERVED_VALUES["outputs.denominators[]"]["one"])


@pytest.mark.parametrize("value", ["reported", "all", "true"])
def test_reserved_key_minmax_universe_rejects_every_value(tmp_path, value):
    """A KNOWN optional key: any value takes the reserved path, never the
    unknown-key path (spec § 3)."""
    text = MINIMAL.replace(
        "  exclusion: {types: [RV], stage: post_neighbors, "
        "absent_neighbor: swallowed}",
        "  exclusion: {types: [RV], stage: post_neighbors, "
        f"absent_neighbor: swallowed, minmax_universe: {value}}}")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert "unknown key" not in message
    assert message.endswith(RESERVED_KEYS["methodology.exclusion.minmax_universe"])
    assert "A.2" in message


def test_enums_are_generated_from_the_reference_table():
    assert set(ENUMS) == set(ENUM_KEYS)
    for key in ENUM_KEYS:
        assert {member.value for member in ENUMS[key]} == set(REFERENCE_KNOBS[key])


def test_enum_members_compare_equal_to_their_strings():
    from delhi_psi.config import AdjacencyRule, Denominator
    assert AdjacencyRule.BBOX == "bbox"
    # str()/format() must give the BARE value: outputs.name_template is
    # rendered with .format(denominator=...), and a plain `Enum` with a str
    # mixin would render "Denominator.POP" there.
    assert str(Denominator.POP) == "pop"
    assert f"{Denominator.POPDENSITY}" == "popdensity"
    assert "delhi_psi_{profile}_{denominator}_2020".format(
        profile="code-2025", denominator=Denominator.POP) == \
        "delhi_psi_code-2025_pop_2020"


def test_data_dir_precedence_flag_beats_env_beats_yaml(tmp_path, monkeypatch):
    yaml_dir = tmp_path / "from_yaml"
    env_dir = tmp_path / "from_env"
    flag_dir = tmp_path / "from_flag"
    text = MINIMAL + f"\npaths: {{data_dir: {yaml_dir}}}\n"
    path = write(tmp_path, text)

    monkeypatch.delenv("DELHI_DATA_DIR", raising=False)
    assert load_config(path).paths.data_dir == yaml_dir

    monkeypatch.setenv("DELHI_DATA_DIR", str(env_dir))
    assert load_config(path).paths.data_dir == env_dir

    assert load_config(path, data_dir=str(flag_dir)).paths.data_dir == flag_dir


def test_data_dir_falls_back_to_home_delhi_data(tmp_path, monkeypatch):
    monkeypatch.delenv("DELHI_DATA_DIR", raising=False)
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.paths.data_dir == Path("~/delhi_data").expanduser()


def test_out_dir_defaults_to_data_dir_and_flag_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("DELHI_DATA_DIR", raising=False)
    data = tmp_path / "data"
    out = tmp_path / "out"
    cfg = load_config(write(tmp_path, MINIMAL), data_dir=str(data))
    assert cfg.paths.out_dir == data
    cfg = load_config(write(tmp_path, MINIMAL), data_dir=str(data),
                      out_dir=str(out))
    assert cfg.paths.out_dir == out
    # load_config resolves paths only; it never creates directories
    assert not out.exists()


def test_unknown_profile_name_lists_the_shipped_ones():
    with pytest.raises(ConfigError) as exc:
        load_config("no-such-profile")
    assert "code-2025" in str(exc.value) and "manuscript" in str(exc.value)


# --- I2: no shipped profile pins out_dir to a literal path ------------
@pytest.mark.parametrize("profile", ["code-2025", "manuscript"])
def test_shipped_profiles_let_the_data_dir_drive_out_dir(tmp_path, monkeypatch,
                                                         profile):
    """A literal `out_dir:` in a shipped profile would silently ignore
    --data-dir (and --out-dir would be the only way to redirect output)."""
    monkeypatch.delenv("DELHI_DATA_DIR", raising=False)
    assert load_config(profile, data_dir=str(tmp_path)).paths.out_dir == tmp_path


# --- C1: the default artifact name is profile-specific ----------------
def test_default_neighbors_artifact_name_is_profile_specific(tmp_path):
    """Two profiles writing to one out_dir must not overwrite each other's
    neighbour lists. code-2025 keeps the historic filename explicitly."""
    assert load_config("manuscript", data_dir=str(tmp_path)) \
        .paths.neighbors_artifact == "colonies_neighbors_manuscript.joblib"
    assert load_config("code-2025", data_dir=str(tmp_path)) \
        .paths.neighbors_artifact == "colonies_neighbors.joblib"


# --- minor (b): crs and validate reject unknown keys like every block --
@pytest.mark.parametrize("block,bogus,dotted", [
    ("crs", "crs: {epsg: 7760, bogus: 1}", "crs.bogus"),
    ("validate", "validate: {max_missing_population: 15, bogus: 1}",
     "validate.bogus"),
])
def test_unknown_key_in_crs_or_validate_raises_config_error(tmp_path, block,
                                                            bogus, dotted):
    """Not a TypeError from the dataclass constructor — a ConfigError naming
    the key, like every other block."""
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, MINIMAL + f"\n{bogus}\n"))
    assert dotted in str(exc.value)


# --- 3B: the categories block (spec 3B §§ 2, 5) ------------------------
def test_categories_block_is_required(tmp_path):
    """A profile states its method completely — `categories` is required
    from cycle 3B on, exactly like `methodology`."""
    without = MINIMAL.replace(CATEGORIES_BLOCK, "")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, without))
    assert "categories" in str(exc.value)


def test_exclusion_type_that_is_not_a_category_is_rejected(tmp_path):
    """`methodology.exclusion.types` holds CATEGORY names — the values of
    the mapping — and the error lists the categories the mapping produces."""
    text = MINIMAL.replace("types: [RV]", "types: [non-urban]")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert "methodology.exclusion.types" in message
    assert "non-urban" in message
    for category in sorted(DELHI_TYPES):
        assert category in message, category


def test_duplicate_mapping_key_is_rejected_naming_the_key(tmp_path):
    """PyYAML's default loader keeps the LAST occurrence silently — half
    the layer would be mapped by a rule nobody can see."""
    text = MINIMAL.replace("    JJC: JJC\n", "    JJC: JJC\n    JJC: Other\n")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert "duplicate key" in message
    assert "JJC" in message


def test_duplicate_key_anywhere_in_the_profile_is_rejected(tmp_path):
    """The checking loader is not scoped to `categories` — a repeated
    `profile:` is the same silent overwrite."""
    text = MINIMAL + "\nprofile: second\n"
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert "duplicate key" in message
    assert "profile" in message


def test_categories_default_is_reserved(tmp_path):
    """A catch-all category is deliberately not offered: it is a KNOWN
    optional key, so it takes the reserved path, never the unknown-key
    path."""
    text = MINIMAL.replace("  scheme: uso-10\n",
                           "  scheme: uso-10\n  default: Other\n")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert "unknown key" not in message
    assert message.endswith(RESERVED_KEYS["categories.default"])


@pytest.mark.parametrize("bad,fragment", [
    ("    JJC:\n", "JJC"),          # empty value -> None
    ("    JJC: 7\n", "7"),          # non-string value
])
def test_non_string_mapping_values_are_rejected(tmp_path, bad, fragment):
    text = MINIMAL.replace("    JJC: JJC\n", bad)
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert "categories.mapping" in message
    assert fragment in message


@pytest.mark.parametrize("bad", ["  scheme: ''\n", "  scheme:\n"])
def test_scheme_must_be_a_non_empty_string(tmp_path, bad):
    text = MINIMAL.replace("  scheme: uso-10\n", bad)
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    assert "categories.scheme" in str(exc.value)


@pytest.mark.parametrize("profile", ["code-2025", "manuscript"])
def test_shipped_profiles_map_the_ten_delhi_types_to_themselves(profile,
                                                                tmp_path):
    """Both shipped profiles use the identity scheme — which is what makes
    'this cycle changes no numbers' true."""
    cfg = load_config(profile, data_dir=str(tmp_path))
    assert cfg.categories.scheme == "uso-10"
    assert set(cfg.categories.mapping) == DELHI_TYPES
    assert categories_of(cfg.categories.mapping) == frozenset(DELHI_TYPES)
    assert all(source == category
               for source, category in cfg.categories.mapping.items())


def test_shipped_exclusion_types_are_categories(tmp_path):
    """`code-2025` keeps [RV], `manuscript` keeps [] — both valid category
    names under the identity mapping."""
    code = load_config("code-2025", data_dir=str(tmp_path))
    assert code.methodology.exclusion.types == ("RV",)
    assert set(code.methodology.exclusion.types) <= categories_of(
        code.categories.mapping)
    manuscript = load_config("manuscript", data_dir=str(tmp_path))
    assert manuscript.methodology.exclusion.types == ()
