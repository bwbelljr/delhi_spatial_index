"""Emit the per-profile production fixtures (spec § 4).

Long format, one row per (profile, scenario, denom, settlement, metric):
columns `profile,scenario,denom,settlement,metric,value`, sorted by
(scenario, denom, settlement, metric), `value` at %.17g, LF line endings.
Geometry, centroid and neighbor-list columns are never serialized — their
reprs are not stable.

STEP-0 BACKEND (spec § 5 step 0): the numbers come from
`tests.test_oracle._production_frame`, i.e. today's pre-refactor wiring
through `spatial_index_utils`. The committed output is the target the
refactored pipeline must reproduce string-for-string. Migration step 5 swaps
the backend to `delhi_psi.pipeline.compute_frames` and proves a no-op diff;
nothing else about this file changes.

Regenerate with:
    uv run python scripts/generate_production_fixtures.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
# Bootstrap: the package is not installed yet at migration step 0, so the
# repo root is not on sys.path when this script is run by path (the CI drift
# guard does exactly that). Removed in migration step 1, once `uv sync`
# installs the project editable and puts the root on sys.path for good.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.test_oracle import SCENARIO_WIRING, _production_frame  # noqa: E402

PRODUCTION_DIR = REPO / "tests" / "fixtures" / "oraculum" / "production"

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


def emit_profile(profile, out_path):
    """Write `profile`'s production fixture to out_path; return out_path."""
    if profile != "code-2025":
        raise ValueError(
            f"unknown profile {profile!r}: the step-0 backend only knows "
            "'code-2025' (migration step 5 generalises this)")
    columns = metric_columns(second_normalization=True)
    records = []
    for scenario, drop_pre, drop_post in SCENARIO_WIRING:
        for denom in DENOMS:
            frame = _production_frame(denom, drop_ids_post=drop_post,
                                      drop_ids_pre=drop_pre)
            records.extend(frame_records(profile, frame, scenario, denom,
                                         columns))
    write_fixture(out_path, records)
    return out_path


def main():
    out_path = emit_profile("code-2025", PRODUCTION_DIR / "code-2025.csv")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
