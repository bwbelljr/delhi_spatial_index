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
from tests.oraculum_fixtures import compute_oracle_frame


def production_dir(city=ORACULUM):
    """Where `city`'s per-profile production fixtures live."""
    return city.fixtures / "production"

# Every profile with a committed production fixture. Adding a profile is one
# YAML plus one entry here, then a regeneration commit (spec § 4).
PROFILES = ("code-2025", "manuscript")

POINT_SERVICES = ("clinic", "school", "bank", "police", "ration", "transport")
SERVICES = POINT_SERVICES + ("road",)
DENOMS = ("pop", "popdensity")

HEADER = ["profile", "scenario", "denom", "settlement", "metric", "value"]


def metric_columns(*, second_normalization):
    """The spec § 4 metric set, in a fixed order (the CSV is sorted anyway)."""
    columns = [f"{svc}_count" for svc in POINT_SERVICES]
    columns.append("road_length")
    for svc in SERVICES:
        columns.append(f"{svc}_pcen")
        columns.append(f"{svc}_idx")
    columns.append("unnorm_psi")
    if second_normalization:
        columns.append("norm_psi")
    columns.append("population")
    columns.append("area_km2")
    return columns


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
    methodology = load_config(profile).methodology
    columns = metric_columns(
        second_normalization=methodology.second_normalization)
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
