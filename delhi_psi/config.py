"""Config schema (spec § 3): one YAML per profile, frozen dataclasses.

Required keys: `profile`, the whole `methodology` block and (from cycle 3B)
the whole `categories` block — a profile is a complete statement of method,
never inherited. Everything else defaults to the `code-2025` values.
Unknown keys, missing required keys and out-of-enum values raise ConfigError
naming the key and the allowed values.

The reference-pinned enums are generated from ONE table, REFERENCE_KNOBS,
which maps each config value to its `tests.reference_impl.compute_city` knob.
tests/test_profiles_match_reference.py reads the same table, so a value with
no reference knob cannot be added to a reference-pinned key.
"""

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from delhi_psi.categories import categories_of
from delhi_psi.io import resolve_data_dir, out_dir_path

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"


class ConfigError(ValueError):
    """Unknown key, missing required key, out-of-enum or reserved value."""


# --- the single enum table (spec § 3) ----------------------------------
# dotted key -> {config value: reference knob value}
REFERENCE_KNOBS = {
    "methodology.adjacency.rule": {"bbox": "bbox", "touch": "border"},
    "methodology.barrier.rule": {"global_asymmetric": "global",
                                 "pairwise": "pair"},
    "methodology.roads": {"decayed": "decayed", "eq4_own_only": "eq4"},
    "methodology.second_normalization": {True: True, False: False},
    "methodology.exclusion.stage": {"post_neighbors": False,
                                    "pre_neighbors": True},
    "methodology.exclusion.absent_neighbor": {"swallowed": "swallowed",
                                              "contributes": "contributes"},
    "outputs.denominators[]": {"pop": "pop", "popdensity": "popdensity"},
}

# `second_normalization` is a bool, not an enum; every other reference-pinned
# key gets a str-valued Enum generated from the table above.
ENUM_KEYS = (
    "methodology.adjacency.rule",
    "methodology.barrier.rule",
    "methodology.roads",
    "methodology.exclusion.stage",
    "methodology.exclusion.absent_neighbor",
    "outputs.denominators[]",
)


def _make_enum(name, key):
    # StrEnum (not Enum + str mixin): its members compare equal to their
    # string values AND str()/format() return the bare value, which
    # outputs.name_template.format(denominator=...) depends on.
    return StrEnum(name, {value.upper(): value
                          for value in REFERENCE_KNOBS[key]})


AdjacencyRule = _make_enum("AdjacencyRule", "methodology.adjacency.rule")
BarrierRule = _make_enum("BarrierRule", "methodology.barrier.rule")
RoadsFormula = _make_enum("RoadsFormula", "methodology.roads")
ExclusionStage = _make_enum("ExclusionStage", "methodology.exclusion.stage")
AbsentNeighbor = _make_enum("AbsentNeighbor",
                            "methodology.exclusion.absent_neighbor")
Denominator = _make_enum("Denominator", "outputs.denominators[]")

ENUMS = {
    "methodology.adjacency.rule": AdjacencyRule,
    "methodology.barrier.rule": BarrierRule,
    "methodology.roads": RoadsFormula,
    "methodology.exclusion.stage": ExclusionStage,
    "methodology.exclusion.absent_neighbor": AbsentNeighbor,
    "outputs.denominators[]": Denominator,
}

# --- reserved values and keys (spec §§ 3, 4, 9) ------------------------
RESERVED_VALUES = {
    "methodology.barrier.rule": {
        "partial_weighted":
            "reserved: w_ij = 1 - L_blocked/L_shared "
            "(docs/oracle/suggested-fixes-memo.md § 2) is config-ready but "
            "reference-pending. Unblock it by adding the reference rule to "
            "tests/reference_impl.py, a hand anchor in "
            "docs/oracle/derivation-worksheet.md, and regenerating "
            "tests/fixtures/oraculum/expected_values.csv (cycle 3C).",
    },
    "outputs.denominators[]": {
        "one":
            "reserved: production supports denom='one' but the reference does "
            "not. Unblock it by adding `denom == \"one\"` to "
            "tests.reference_impl.compute_city and regenerating "
            "tests/fixtures/oraculum/expected_values.csv first.",
    },
}

RESERVED_KEYS = {
    "methodology.exclusion.minmax_universe":
        "reserved: Open Decision A.2 — whether Eq. 2's min/max spans reported "
        "settlements only or all settlements. There is no knob for it in the "
        "reference implementation or in production. Unblock it by putting the "
        "question to Raj (DEL-13) and adding the reference rule.",
    "categories.default":
        "reserved: a catch-all category is deliberately NOT offered — an "
        "unmapped source type must fail the run, because silence is the "
        "failure mode this layer exists to prevent (spec 3B § 2). Map every "
        "source type explicitly instead.",
}


# --- dataclasses -------------------------------------------------------
@dataclass(frozen=True)
class CrsConfig:
    epsg: int = 7760


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    out_dir: Path
    # Already rendered: `_paths` formats DEFAULT_PATHS' `{profile}` in.
    neighbors_artifact: str


@dataclass(frozen=True)
class LayerSpec:
    path: str
    id_col: str | None = None
    type_col: str | None = None


@dataclass(frozen=True)
class PopulationSpec:
    path: str
    id_col: str
    value_col: str
    missing: str = "drop"          # drop | error


@dataclass(frozen=True)
class CategoriesConfig:
    scheme: str
    mapping: dict                  # source type -> category name


@dataclass(frozen=True)
class LayersConfig:
    settlements: LayerSpec
    population: PopulationSpec
    bounds: str
    ndmc_center: str | None
    barriers: dict


@dataclass(frozen=True)
class ServicesConfig:
    point: dict
    line: dict


@dataclass(frozen=True)
class AdjacencyConfig:
    rule: AdjacencyRule


@dataclass(frozen=True)
class BarrierConfig:
    rule: BarrierRule
    combine: object                # "any" or a tuple of layer names


@dataclass(frozen=True)
class DecayConfig:
    form: str
    distance_unit: str


@dataclass(frozen=True)
class ExclusionConfig:
    types: tuple
    stage: ExclusionStage
    absent_neighbor: AbsentNeighbor


@dataclass(frozen=True)
class MethodologyConfig:
    adjacency: AdjacencyConfig
    barrier: BarrierConfig
    decay: DecayConfig
    roads: RoadsFormula
    second_normalization: bool
    exclusion: ExclusionConfig


@dataclass(frozen=True)
class ValidateConfig:
    max_missing_population: int = 15


@dataclass(frozen=True)
class OutputsConfig:
    denominators: tuple = (Denominator.POP, Denominator.POPDENSITY)
    formats: tuple = ("csv", "shp", "joblib")
    name_template: str = "delhi_psi_{profile}_{denominator}_2020"


@dataclass(frozen=True)
class Config:
    profile: str
    methodology: MethodologyConfig
    categories: CategoriesConfig
    crs: CrsConfig
    paths: PathsConfig
    layers: LayersConfig
    services: ServicesConfig
    validate: ValidateConfig
    outputs: OutputsConfig


# --- defaults for every non-methodology block (spec § 3) ---------------
DEFAULT_LAYERS = {
    "settlements": {"path": "uso_update_sep2021/uso_update_sep2021.shp",
                    "id_col": "USO_AREA_U", "type_col": "USO_FINAL"},
    "population": {"path": "pop_colony_wp_2020_jjc_adjusted.csv",
                   "id_col": "uso_area_u", "value_col": "population",
                   "missing": "drop"},
    "bounds": "delhi_bounds_buffer/delhi_bounds_buffer.shp",
    "ndmc_center": "ndmc_center7760/ndmc_center7760.shp",
    "barriers": {"canal": "Barrier_Clip/Canal/Canal.shp",
                 "railway": "Barrier_Clip/Railway/Railway_Line.shp",
                 "drain": "Barrier_Clip/Drain/Major_Drain.shp"},
}
DEFAULT_SERVICES = {
    "point": {"bank": "Public Services/Banking/Banking.shp",
              "health": "Public Services/Health/Health.shp",
              "police": "Public Services/Police/Police Station.shp",
              "ration": "Public Services/Ration/Ration.shp",
              "school": "Public Services/School/schools7760.shp",
              "transport": "Public Services/Transport/Transport.shp"},
    "line": {"road": "Public Services/Major Road/Road.shp"},
}
DEFAULT_CRS = {"epsg": 7760}
# `neighbors_artifact` is rendered with the profile name at load time, so two
# profiles pointed at one out_dir cannot overwrite each other's neighbour
# lists. code-2025 pins the historic filename explicitly (the real-data proof
# compares against it), which is why the default is not simply that name.
DEFAULT_PATHS = {"data_dir": "~/delhi_data", "out_dir": None,
                 "neighbors_artifact": "colonies_neighbors_{profile}.joblib"}
DEFAULT_VALIDATE = {"max_missing_population": 15}
DEFAULT_OUTPUTS = {"denominators": ["pop", "popdensity"],
                   "formats": ["csv", "shp", "joblib"],
                   "name_template": "delhi_psi_{profile}_{denominator}_2020"}

TOP_LEVEL_KEYS = ("profile", "categories", "crs", "paths", "layers",
                  "services", "methodology", "validate", "outputs")


# --- validation helpers ------------------------------------------------
def _reject_unknown(mapping, allowed, prefix):
    for key in mapping:
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if dotted in RESERVED_KEYS:
            raise ConfigError(f"{dotted}: {RESERVED_KEYS[dotted]}")
        if key not in allowed:
            raise ConfigError(
                f"unknown key {dotted!r}; allowed keys here: "
                f"{sorted(allowed)}")


def _require(mapping, key, prefix):
    dotted = f"{prefix}.{key}" if prefix else str(key)
    if key not in mapping:
        raise ConfigError(f"missing required key {dotted!r}")
    return mapping[key]


def _coerce_enum(key, value):
    reserved = RESERVED_VALUES.get(key, {})
    if value in reserved:
        raise ConfigError(f"{key}: {reserved[value]}")
    try:
        return ENUMS[key](value)
    except ValueError:
        allowed = sorted(str(v) for v in REFERENCE_KNOBS[key])
        extra = sorted(reserved)
        note = f" (reserved: {extra})" if extra else ""
        raise ConfigError(
            f"{key}: {value!r} is not allowed; allowed values: "
            f"{allowed}{note}") from None


def _bool(key, value):
    if not isinstance(value, bool):
        raise ConfigError(f"{key}: {value!r} is not allowed; "
                          "allowed values: [True, False]")
    return value


# --- loader ------------------------------------------------------------
class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a repeated mapping key.

    PyYAML's default keeps the LAST occurrence and says nothing, so a
    profile with two `Planned:` lines under `categories.mapping` would map
    half its layer by a rule nobody can see. That is the exact silence this
    whole layer exists to prevent, so it is an error here — everywhere in
    the profile, not only inside `categories`.
    """

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                mark = key_node.start_mark
                raise ConfigError(
                    f"duplicate key {key!r} at {mark.name}:{mark.line + 1}; "
                    "PyYAML would silently keep the last occurrence")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def shipped_profiles():
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def _profile_path(profile_or_path):
    candidate = Path(profile_or_path)
    if candidate.suffix in (".yaml", ".yml"):
        if not candidate.exists():
            raise ConfigError(f"config file not found: {candidate}")
        return candidate
    shipped = PROFILES_DIR / f"{profile_or_path}.yaml"
    if not shipped.exists():
        raise ConfigError(
            f"unknown profile {str(profile_or_path)!r}; shipped profiles: "
            f"{shipped_profiles()} (or pass a path to a .yaml file)")
    return shipped


def _methodology(raw, *, allowed_categories):
    _reject_unknown(raw, {"adjacency", "barrier", "decay", "roads",
                          "second_normalization", "exclusion"}, "methodology")

    adjacency_raw = _require(raw, "adjacency", "methodology")
    _reject_unknown(adjacency_raw, {"rule"}, "methodology.adjacency")
    adjacency = AdjacencyConfig(rule=_coerce_enum(
        "methodology.adjacency.rule",
        _require(adjacency_raw, "rule", "methodology.adjacency")))

    barrier_raw = _require(raw, "barrier", "methodology")
    _reject_unknown(barrier_raw, {"rule", "combine"}, "methodology.barrier")
    combine = _require(barrier_raw, "combine", "methodology.barrier")
    if combine != "any":
        if not isinstance(combine, list) or not all(
                isinstance(item, str) for item in combine):
            raise ConfigError(
                "methodology.barrier.combine: expected 'any' or a list of "
                f"layer names, got {combine!r}")
        combine = tuple(combine)
    barrier = BarrierConfig(
        rule=_coerce_enum("methodology.barrier.rule",
                          _require(barrier_raw, "rule", "methodology.barrier")),
        combine=combine)

    decay_raw = _require(raw, "decay", "methodology")
    _reject_unknown(decay_raw, {"form", "distance_unit"}, "methodology.decay")
    form = _require(decay_raw, "form", "methodology.decay")
    unit = _require(decay_raw, "distance_unit", "methodology.decay")
    if form != "inverse_linear":
        raise ConfigError(f"methodology.decay.form: {form!r} is not allowed; "
                          "allowed values: ['inverse_linear']")
    if unit != "km":
        raise ConfigError(
            f"methodology.decay.distance_unit: {unit!r} is not allowed; "
            "allowed values: ['km']")
    decay = DecayConfig(form=form, distance_unit=unit)

    exclusion_raw = _require(raw, "exclusion", "methodology")
    _reject_unknown(exclusion_raw, {"types", "stage", "absent_neighbor"},
                    "methodology.exclusion")
    types = _require(exclusion_raw, "types", "methodology.exclusion")
    if not isinstance(types, list) or not all(
            isinstance(item, str) for item in types):
        raise ConfigError("methodology.exclusion.types: expected a list of "
                          f"settlement-type strings, got {types!r}")
    unknown = [item for item in types if item not in allowed_categories]
    if unknown:
        raise ConfigError(
            f"methodology.exclusion.types: {unknown} are not categories "
            "produced by categories.mapping — exclusion is written in "
            "CATEGORY names, not source types; allowed values: "
            f"{sorted(allowed_categories)}")
    exclusion = ExclusionConfig(
        types=tuple(types),
        stage=_coerce_enum("methodology.exclusion.stage",
                           _require(exclusion_raw, "stage",
                                    "methodology.exclusion")),
        absent_neighbor=_coerce_enum(
            "methodology.exclusion.absent_neighbor",
            _require(exclusion_raw, "absent_neighbor",
                     "methodology.exclusion")))

    return MethodologyConfig(
        adjacency=adjacency,
        barrier=barrier,
        decay=decay,
        roads=_coerce_enum("methodology.roads",
                           _require(raw, "roads", "methodology")),
        second_normalization=_bool(
            "methodology.second_normalization",
            _require(raw, "second_normalization", "methodology")),
        exclusion=exclusion)


def _categories(raw):
    """The `categories` block: a scheme name and a source type -> category
    map. Several sources mapping to one category (X:1) is the point; a
    category name equal to a source name is fine (identity)."""
    _reject_unknown(raw, {"scheme", "mapping"}, "categories")

    scheme = _require(raw, "scheme", "categories")
    if not isinstance(scheme, str) or not scheme.strip():
        raise ConfigError(
            f"categories.scheme: {scheme!r} is not allowed; expected a "
            "non-empty string naming the scheme (e.g. 'uso-10')")

    mapping = _require(raw, "mapping", "categories")
    if not isinstance(mapping, dict) or not mapping:
        raise ConfigError(
            "categories.mapping: expected a non-empty map of source type -> "
            f"category, got {mapping!r}")
    for source, category in mapping.items():
        if not isinstance(source, str) or not source:
            raise ConfigError(
                f"categories.mapping: source type {source!r} is not "
                "allowed; expected a non-empty string")
        if not isinstance(category, str) or not category:
            raise ConfigError(
                f"categories.mapping[{source!r}]: {category!r} is not "
                "allowed; expected a non-empty category name")
    return CategoriesConfig(scheme=scheme, mapping=dict(mapping))


def _layers(raw):
    merged = {**DEFAULT_LAYERS, **raw}
    _reject_unknown(merged, set(DEFAULT_LAYERS), "layers")
    settlements = {**DEFAULT_LAYERS["settlements"], **merged["settlements"]}
    _reject_unknown(settlements, {"path", "id_col", "type_col"},
                    "layers.settlements")
    population = {**DEFAULT_LAYERS["population"], **merged["population"]}
    _reject_unknown(population, {"path", "id_col", "value_col", "missing"},
                    "layers.population")
    if population["missing"] not in ("drop", "error"):
        raise ConfigError(
            f"layers.population.missing: {population['missing']!r} is not "
            "allowed; allowed values: ['drop', 'error']")
    return LayersConfig(
        settlements=LayerSpec(**settlements),
        population=PopulationSpec(**population),
        bounds=merged["bounds"],
        ndmc_center=merged["ndmc_center"],
        barriers=dict(merged["barriers"]))


def _services(raw):
    merged = {**DEFAULT_SERVICES, **raw}
    _reject_unknown(merged, set(DEFAULT_SERVICES), "services")
    return ServicesConfig(point=dict(merged["point"]),
                          line=dict(merged["line"]))


def _outputs(raw):
    merged = {**DEFAULT_OUTPUTS, **raw}
    _reject_unknown(merged, set(DEFAULT_OUTPUTS), "outputs")
    denominators = merged["denominators"]
    if not isinstance(denominators, list) or not denominators:
        raise ConfigError("outputs.denominators: expected a non-empty list, "
                          f"got {denominators!r}")
    formats = merged["formats"]
    for fmt in formats:
        if fmt not in ("csv", "shp", "joblib"):
            raise ConfigError(f"outputs.formats: {fmt!r} is not allowed; "
                              "allowed values: ['csv', 'shp', 'joblib']")
    return OutputsConfig(
        denominators=tuple(_coerce_enum("outputs.denominators[]", d)
                           for d in denominators),
        formats=tuple(formats),
        name_template=merged["name_template"])


def _crs(raw):
    merged = {**DEFAULT_CRS, **raw}
    _reject_unknown(merged, set(DEFAULT_CRS), "crs")
    return CrsConfig(**merged)


def _validate(raw):
    merged = {**DEFAULT_VALIDATE, **raw}
    _reject_unknown(merged, set(DEFAULT_VALIDATE), "validate")
    return ValidateConfig(**merged)


def _paths(raw, cli_data_dir, cli_out_dir, profile):
    merged = {**DEFAULT_PATHS, **raw}
    _reject_unknown(merged, set(DEFAULT_PATHS), "paths")
    yaml_data_dir = merged["data_dir"]
    if cli_data_dir:
        data_dir = Path(cli_data_dir).expanduser()
    elif os.environ.get("DELHI_DATA_DIR"):
        data_dir = Path(os.environ["DELHI_DATA_DIR"]).expanduser()
    else:
        data_dir = resolve_data_dir(yaml_data_dir)
    out_dir = out_dir_path(cli_out_dir or merged["out_dir"], data_dir)
    return PathsConfig(
        data_dir=data_dir, out_dir=out_dir,
        neighbors_artifact=merged["neighbors_artifact"].format(profile=profile))


def load_config(profile_or_path, *, data_dir=None, out_dir=None):
    """Load a shipped profile by name, or a YAML file by path.

    Path precedence: --data-dir/--out-dir argument > DELHI_DATA_DIR env var
    > the YAML value > ~/delhi_data. Resolution only — no directory is
    created here (the pipeline stages do that).
    """
    path = _profile_path(profile_or_path)
    # An open file handle, not read_text(): PyYAML then puts the file's path
    # into every mark, so _UniqueKeyLoader's message names the file and line.
    with path.open() as handle:
        raw = yaml.load(handle, Loader=_UniqueKeyLoader) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    _reject_unknown(raw, set(TOP_LEVEL_KEYS), "")

    profile = _require(raw, "profile", "")
    # categories first: _methodology's exclusion check needs the category set.
    categories = _categories(_require(raw, "categories", ""))
    return Config(
        profile=profile,
        methodology=_methodology(
            _require(raw, "methodology", ""),
            allowed_categories=categories_of(categories.mapping)),
        categories=categories,
        crs=_crs(raw.get("crs", {})),
        paths=_paths(raw.get("paths", {}), data_dir, out_dir, profile),
        layers=_layers(raw.get("layers", {})),
        services=_services(raw.get("services", {})),
        validate=_validate(raw.get("validate", {})),
        outputs=_outputs(raw.get("outputs", {})))
