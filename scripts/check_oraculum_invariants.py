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
from tests.variants import ADDED_BAND_PAIRS, BAND_RADII_KM, EXPECTED_BAND_PAIRS

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

    The reference import is local to avoid an import cycle:
    `reference_impl` imports `cities`, and the fixture generators import
    this module (`check`) before they import `reference_impl` themselves —
    so `reference_impl` has to stay out of this module's top level.
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


def variant_expected_values_path(city=ORACULUM):
    return city.fixtures / "variants_expected_values.csv"


def check_bands(city, *, expected=None, added=None):
    """Re-derive the band neighbourhoods from the geometry (spec § 3).

    Against `adjacency(...)` DIRECTLY — never anything downstream of a
    barrier rule, which would fold the canal's severing into the band's
    numbers. Move a vertex so a radius gains or loses a pair, or so two
    bands coincide without that being the pinned saturation case, and this
    returns violations instead of quietly emitting a fixture that pins
    nothing.

    The bands no longer have to be pairwise distinct: DEL-55 adds a 10 km
    radius, and Oraculum's 5 km band is already the complete graph on its
    seven settlements (21 pairs), so 5 km and 10 km are EQUAL there. The
    invariant is therefore: a wider radius never drops a pair (1); growth is
    STRICT wherever `ADDED_BAND_PAIRS` says the wider radius adds something,
    and where it adds nothing the two bands must be equal AND the narrower
    one must already be the complete graph — saturation pinned as a fact
    about the city, not tolerated as a gap (2); and the added set matches
    exactly (3).
    """
    from tests.reference_impl import adjacency

    expected = EXPECTED_BAND_PAIRS[city.name] if expected is None else expected
    added = ADDED_BAND_PAIRS[city.name] if added is None else added
    settlements = city.load_settlements()
    n = len(settlements)
    pairs = {}
    for km in BAND_RADII_KM:
        nbrs = adjacency(settlements, "within_distance", km)
        pairs[km] = {tuple(sorted((i, j)))
                     for i, js in nbrs.items() for j in js}

    violations = []
    for km in BAND_RADII_KM:
        if len(pairs[km]) != expected[km]:
            violations.append(
                f"band {km} km: pair count {len(pairs[km])}, expected "
                f"{expected[km]}")
    for lower, upper in zip(BAND_RADII_KM, BAND_RADII_KM[1:]):
        if not pairs[lower] <= pairs[upper]:
            violations.append(
                f"band {lower} km is not a subset of band {upper} km "
                "— a wider radius must never drop a pair")
        added_upper = added[upper]
        if added_upper:
            if not pairs[lower] < pairs[upper]:
                violations.append(
                    f"band {lower} km is not a STRICT subset of band "
                    f"{upper} km, though {upper} km is expected to add "
                    f"{sorted(added_upper)}")
        else:
            if pairs[lower] != pairs[upper]:
                violations.append(
                    f"band {upper} km differs from band {lower} km even "
                    "though its added set is empty — the two bands should "
                    "be equal (saturation)")
            if len(pairs[lower]) != n * (n - 1) // 2:
                violations.append(
                    f"band {lower} km has {len(pairs[lower])} pairs but an "
                    f"empty added-set at {upper} km requires it to already "
                    f"be the complete graph on {n} settlements "
                    f"({n * (n - 1) // 2} pairs)")
        got = pairs[upper] - pairs[lower]
        if got != added_upper:
            violations.append(
                f"band {upper} km adds {sorted(got)}, expected "
                f"{sorted(added_upper)}")
    return violations


def emit_checked_variant_expected_values(city, out_path):
    """Emit `city`'s variants_expected_values.csv, but ONLY if the band
    neighbourhoods are the ones the spec fixed AND the emitted numbers pass
    `check`.

    Same staging discipline as emit_checked_expected_values: the temporary
    file lives OUTSIDE the repo, so a failed run can never leave an untracked
    file under tests/fixtures/ for the CI drift guard to trip over.
    """
    from tests.reference_impl import emit_variant_expected_values

    out_path = Path(out_path)
    band_violations = check_bands(city)
    if band_violations:
        for violation in band_violations:
            print(f"VIOLATION [{city.name}]:", violation)
        raise SystemExit(
            f"{len(band_violations)} band violation(s) for city "
            f"{city.name!r}; refusing to write {out_path}")
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "variants_expected_values.csv"
        emit_variant_expected_values(staged, city)
        violations = check(pd.read_csv(staged))
        if violations:
            for violation in violations:
                print(f"VIOLATION [{city.name}]:", violation)
            raise SystemExit(
                f"{len(violations)} invariant violation(s) for city "
                f"{city.name!r}; refusing to write {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(out_path))
    return out_path


if __name__ == "__main__":
    problems = []
    for target_city in CITIES:
        problems.extend(f"{target_city.name}: {problem}"
                        for problem in check(city=target_city))
        problems.extend(f"{target_city.name}: {problem}"
                        for problem in check_bands(target_city))
        problems.extend(
            f"{target_city.name} (variants): {problem}"
            for problem in check(
                pd.read_csv(variant_expected_values_path(target_city)),
                city=target_city))
    for p in problems:
        print("VIOLATION:", p)
    print("OK" if not problems else f"{len(problems)} violation(s)")
    sys.exit(1 if problems else 0)
