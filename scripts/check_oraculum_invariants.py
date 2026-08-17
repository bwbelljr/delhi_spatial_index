"""Spec §7 consistency guard over expected_values.csv (CSV-wide scope;
geometry-scope checks live in tests/test_fixture_invariants.py).

Run standalone (exit 1 on violation) or via its pytest wrapper:
    uv run python scripts/check_oraculum_invariants.py
"""

import sys
from pathlib import Path

import pandas as pd

CSV = (Path(__file__).resolve().parent.parent / "tests" / "fixtures"
       / "oraculum" / "expected_values.csv")
SERVICES = ("clinic", "school", "bank", "police", "ration", "transport",
            "road")
UNIQUE_ANCHOR_SERVICES = ("clinic", "school")


def check(df=None):
    df = pd.read_csv(CSV) if df is None else df
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


if __name__ == "__main__":
    problems = check()
    for p in problems:
        print("VIOLATION:", p)
    print("OK" if not problems else f"{len(problems)} violation(s)")
    sys.exit(1 if problems else 0)
