"""Emit the per-profile production fixtures (spec § 4).

Long format, one row per (profile, scenario, denom, settlement, metric):
columns `profile,scenario,denom,settlement,metric,value`, sorted by
(scenario, denom, settlement, metric), `value` at %.17g, LF line endings.
Geometry, centroid and neighbor-list columns are never serialized — their
reprs are not stable.

The numbers come from delhi_psi.pipeline.compute_frames, driven by the
profile's own methodology plus the § 7 scenario overrides. Migration step 0
generated the code-2025 fixture from the pre-refactor wiring; that committed
file is the refactor's correctness proof, so this generator must reproduce it
byte for byte.

Regenerate with:
    uv run python scripts/generate_production_fixtures.py
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

from delhi_psi.config import load_config
from tests.cities import CITIES, ORACULUM
from tests.oraculum_fixtures import compute_oracle_frame, configured_fixture_services


def production_dir(city=ORACULUM):
    """Where `city`'s per-profile production fixtures live."""
    return city.fixtures / "production"

# Every profile with a committed production fixture. Adding a profile is one
# YAML plus one entry here, then a regeneration commit (spec § 4).
PROFILES = ("code-2025", "manuscript",
            "adj-touch", "band-0km", "band-1km", "band-5km", "band-10km",
            "decay-none", "decay-power05", "decay-power2", "decay-exp2km",
            "decay-exp5km", "decay-boundary",
            "services-no-ration", "services-no-bank")

POINT_SERVICES = ("clinic", "school", "bank", "police", "ration", "transport")
LINE_SERVICES = ("road",)
SERVICES = POINT_SERVICES + LINE_SERVICES
DENOMS = ("pop", "popdensity")

HEADER = ["profile", "scenario", "denom", "settlement", "metric", "value"]


def metric_columns(*, second_normalization, point_services=POINT_SERVICES,
                   line_services=LINE_SERVICES):
    """The spec § 4 metric set, in a fixed order (the CSV is sorted anyway).

    `point_services`/`line_services` default to the full seven-service set,
    so every existing caller keeps its shape and its numbers. A profile
    that subsets services (DEL-40) passes the narrower tuples emit_profile
    derives from `cfg.services`, and the columns for a dropped service are
    then absent from the result entirely — never present and zeroed.
    """
    columns = [f"{svc}_count" for svc in point_services]
    for svc in line_services:
        columns.append(f"{svc}_length")
    for svc in (*point_services, *line_services):
        columns.append(f"{svc}_pcen")
        columns.append(f"{svc}_idx")
    columns.append("unnorm_psi")
    if second_normalization:
        columns.append("norm_psi")
    columns.append("population")
    columns.append("area_km2")
    return columns


def profile_service_columns(cfg):
    """(point_services, line_services) fixture-name tuples `cfg` configures,
    in POINT_SERVICES/LINE_SERVICES canonical order — what `metric_columns`
    needs to emit metrics for only the services a subset profile carries
    (DEL-40 spec § 1, Task 1/3)."""
    configured = configured_fixture_services(cfg)
    point = tuple(svc for svc in POINT_SERVICES if svc in configured)
    line = tuple(svc for svc in LINE_SERVICES if svc in configured)
    return point, line


def frame_records(profile, frame, scenario, denom, columns):
    """One record per (settlement, metric); `frame` is indexed by settlement."""
    return [(profile, scenario, denom, sid, metric, row[metric])
            for sid, row in frame.iterrows()
            for metric in columns]


def write_fixture(path, records):
    ordered = sorted(records, key=lambda r: (r[1], r[2], r[3], r[4]))
    df = pd.DataFrame(ordered, columns=HEADER)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def emit_profile(profile, out_path, city=ORACULUM):
    """Write `profile`'s production fixture for `city` to out_path; return it."""
    cfg = load_config(profile)
    methodology = cfg.methodology
    point_services, line_services = profile_service_columns(cfg)
    columns = metric_columns(
        second_normalization=methodology.second_normalization,
        point_services=point_services, line_services=line_services)
    records = []
    for scenario in city.scenarios:
        for denom in DENOMS:
            frame = compute_oracle_frame(profile,
                                         types=scenario.exclusion_types,
                                         stage=scenario.stage, denom=denom,
                                         city=city)
            records.extend(frame_records(profile, frame, scenario.name, denom,
                                         columns))
    write_fixture(out_path, records)
    return out_path


def main():
    for city in CITIES:
        for profile in PROFILES:
            out_path = emit_profile(profile,
                                    production_dir(city) / f"{profile}.csv",
                                    city)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
