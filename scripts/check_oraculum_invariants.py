"""Spec §7 consistency guard over a city's expected_values.csv (CSV-wide
scope; geometry-scope checks live in tests/test_fixture_invariants.py).

From cycle 3C this module also owns the "only write a VALID fixture" step:
`emit_checked_expected_values` is what both geometry generators call, so a
city whose numbers would violate the guard is never committed.

Run standalone over every city (exit 1 on violation) or via its pytest
wrapper:
    uv run python scripts/check_oraculum_invariants.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

from tests.cities import CITIES, ORACULUM

SERVICES = ("clinic", "school", "bank", "police", "ration", "transport",
            "road")
UNIQUE_ANCHOR_SERVICES = ("clinic", "school")


def expected_values_path(city=ORACULUM):
    return city.fixtures / "expected_values.csv"


CSV = expected_values_path(ORACULUM)


def check(df=None, *, city=ORACULUM):
    df = pd.read_csv(expected_values_path(city)) if df is None else df
    violations = []
    groups = df[df["metric"].str.endswith("_pcen")].groupby(
        ["rule", "scenario", "denom", "metric"])
    for (rule, scenario, denom, metric), grp in groups:
        vals = grp["value"]
        if not vals.max() > vals.min():
            violations.append(
                f"degenerate min-max: {rule}/{scenario}/{denom}/{metric}")
        svc = metric[: -len("_pcen")]
        if svc in UNIQUE_ANCHOR_SERVICES:
            if (vals == vals.max()).sum() != 1:
                violations.append(
                    f"tied argmax: {rule}/{scenario}/{denom}/{metric}")
            if (vals == vals.min()).sum() != 1:
                violations.append(
                    f"tied argmin: {rule}/{scenario}/{denom}/{metric}")
    return violations


def emit_checked_expected_values(city, out_path):
    """Emit `city`'s expected_values.csv, but ONLY if it passes `check`.

    The reference scores the city into a temporary file, the guard runs on
    exactly the bytes that would be committed, and the file is moved into
    place only when there are no violations. On any violation NOTHING is
    written and the process exits 1 — a fixture that ties a clinic/school
    anchor or flattens a min-max group is not a fixture, it is a silently
    degenerate oracle.

    The reference import is local so `check` stays importable without
    dragging in geopandas.
    """
    from tests.reference_impl import emit_expected_values

    out_path = Path(out_path)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "expected_values.csv"
        emit_expected_values(staged, city)
        violations = check(pd.read_csv(staged))
        if violations:
            for violation in violations:
                print(f"VIOLATION [{city.name}]:", violation)
            raise SystemExit(
                f"{len(violations)} invariant violation(s) for city "
                f"{city.name!r}; refusing to write {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # The staging directory is OUTSIDE the repo, so a failed run can
        # never leave an untracked file under tests/fixtures/ for the CI
        # drift guard to trip over.
        shutil.move(str(staged), str(out_path))
    return out_path


if __name__ == "__main__":
    problems = []
    for target_city in CITIES:
        problems.extend(f"{target_city.name}: {problem}"
                        for problem in check(city=target_city))
    for p in problems:
        print("VIOLATION:", p)
    print("OK" if not problems else f"{len(problems)} violation(s)")
    sys.exit(1 if problems else 0)
