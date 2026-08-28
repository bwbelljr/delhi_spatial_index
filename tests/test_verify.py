"""Tests for the baseline comparison functions (synthetic data, no geo deps)."""
import pandas as pd

from delhi_psi.verify import (
    compare_neighbor_frames,
    compare_numeric_frames,
)


def _nbr_frame(nbrs_b=("A", "C")):
    # nbrs_dist_bbox mirrors the real schema from calc_nbr_dist:
    # a list of (neighbor_id, distance) tuples, NOT a parallel float list.
    dists = {"A": 1.5, "C": 2.5}
    return pd.DataFrame(
        {
            "USO_AREA_U": ["A", "B"],
            "nbrs_bbox": [["B"], list(nbrs_b)],
            "nbrs_dist_bbox": [
                [("B", 1.5)],
                [(n, dists[n]) for n in nbrs_b],
            ],
        }
    )


def test_identical_neighbor_frames_pass():
    assert compare_neighbor_frames(_nbr_frame(), _nbr_frame()) == []


def test_neighbor_set_mismatch_reported():
    issues = compare_neighbor_frames(_nbr_frame(nbrs_b=("A",)), _nbr_frame())
    assert any("neighbor" in i.lower() for i in issues)


def test_missing_colony_reported():
    new = _nbr_frame().iloc[:1]
    issues = compare_neighbor_frames(new, _nbr_frame())
    assert any("id" in i.lower() or "colon" in i.lower() for i in issues)


def _psi_frame(psi=0.5):
    return pd.DataFrame(
        {"USO_AREA_U": ["A", "B"], "norm_psi": [psi, 0.7], "bank_idx": [0.1, 0.2]}
    )


def test_identical_numeric_frames_pass():
    issues, report = compare_numeric_frames(
        _psi_frame(), _psi_frame(), "USO_AREA_U", 1e-9, 1e-12
    )
    assert issues == []
    assert report  # per-column deviation lines exist


def test_numeric_deviation_reported():
    issues, _ = compare_numeric_frames(
        _psi_frame(psi=0.5001), _psi_frame(), "USO_AREA_U", 1e-9, 1e-12
    )
    assert any("norm_psi" in i for i in issues)


def test_tiny_float_noise_tolerated():
    issues, _ = compare_numeric_frames(
        _psi_frame(psi=0.5 + 1e-14), _psi_frame(), "USO_AREA_U", 1e-9, 1e-12
    )
    assert issues == []


def test_incidental_columns_ignored():
    # The to_csv row index ("Unnamed: 0") and notebook-era "index" column
    # reflect row order, not results — differences there must not FAIL.
    base = _psi_frame().assign(**{"Unnamed: 0": [0, 1], "index": [0, 1]})
    new = _psi_frame().assign(**{"Unnamed: 0": [5, 9], "index": [7, 3]})
    issues, _ = compare_numeric_frames(new, base, "USO_AREA_U", 1e-9, 1e-12)
    assert issues == []


def test_missing_column_reported():
    # A baseline column absent from the new run must FAIL loudly, never be
    # silently skipped (guards against dependency-driven column drops).
    new = _psi_frame().drop(columns=["norm_psi"])
    issues, _ = compare_numeric_frames(new, _psi_frame(), "USO_AREA_U", 1e-9, 1e-12)
    assert any("norm_psi" in i and "missing" in i for i in issues)


def test_fresh_paths_come_from_the_config(tmp_path):
    """--config locates the fresh files; the baseline paths stay the
    script's own arguments (they exist only for code-2025)."""
    from scripts.verify_against_baseline import fresh_paths

    paths = fresh_paths("code-2025", verify_dir=tmp_path)
    assert paths["neighbors"] == tmp_path / "colonies_neighbors.joblib"
    assert paths["psi"]["pop"] == tmp_path / "delhi_psi_code-2025_pop_2020.csv"
    assert paths["psi"]["popdensity"] == \
        tmp_path / "delhi_psi_code-2025_popdensity_2020.csv"
