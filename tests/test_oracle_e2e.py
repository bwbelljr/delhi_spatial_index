"""Real-CLI end-to-end: temp data dir -> delhi-psi preprocess -> compute.

The code-2025 profile excludes RV, so the output is compared against
rule=code / scenario=excl_rv_only. This leg stays a SUBPROCESS run: it proves
the installed console script works, which the in-process tests in
tests/test_cli.py cannot.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.oraculum_fixtures import oracle_profile_path
from tests.test_cli import data_dir  # noqa: F401  (module-scoped fixture)

CSV = Path(__file__).resolve().parent / "fixtures" / "oraculum" / "expected_values.csv"


def _run(*args):
    proc = subprocess.run([sys.executable, "-m", "delhi_psi.cli", *args],
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"failed:\n{proc.stdout}\n{proc.stderr}"
    return proc


def test_full_cli_chain_matches_excl_rv_only(data_dir, tmp_path):  # noqa: F811
    out_dir = tmp_path / "out"
    # The derived profile (spec 3B § 2): code-2025's method, with the
    # identity mapping over THIS city's vocabulary. Every methodology value
    # and every assertion below is code-2025's, unchanged.
    profile = str(oracle_profile_path("code-2025", tmp_path))
    _run("preprocess", "--config", profile, "--data-dir", str(data_dir),
         "--out-dir", str(out_dir))
    assert (out_dir / "colonies_neighbors.joblib").exists()
    _run("compute", "--config", profile, "--data-dir", str(data_dir),
         "--out-dir", str(out_dir))

    got = pd.read_csv(out_dir / "delhi_psi_code-2025_pop_2020.csv")
    got = got.set_index("USO_AREA_U")

    exp = pd.read_csv(CSV)
    exp = exp[(exp["rule"] == "code") & (exp["scenario"] == "excl_rv_only")
              & (exp["denom"] == "pop")] \
        .pivot(index="settlement", columns="metric", values="value")

    assert set(got.index) == set(exp.index) == {"A", "B", "C", "D", "E", "IND"}
    # real pipeline service naming: the fixture's clinic layer is written to
    # Public Services/Health/Health.shp, so the config service is `health`
    mapping = {
        "health_pcen": "clinic_pcen", "health_idx": "clinic_idx",
        "school_pcen": "school_pcen", "school_idx": "school_idx",
        "bank_pcen": "bank_pcen", "police_pcen": "police_pcen",
        "ration_pcen": "ration_pcen", "transport_pcen": "transport_pcen",
        "road_pcen": "road_pcen", "road_idx": "road_idx",
        "unnorm_psi": "psi_eq1", "norm_psi": "norm_psi",
    }
    for got_col, metric in mapping.items():
        for sid in exp.index:
            assert got.loc[sid, got_col] == pytest.approx(
                exp.loc[sid, metric], abs=1e-9), (got_col, sid)
