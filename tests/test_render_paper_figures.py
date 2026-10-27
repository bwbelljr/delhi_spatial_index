"""The paper figures' data helpers and the end-to-end render (DEL-33).

The helpers are pure over plain frames; the end-to-end test needs the DEL-32
run and the real settlement layer, so it skips without them (CI).
"""
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

REPO = Path(__file__).resolve().parent.parent

from scripts.render_paper_figures import (  # noqa: E402
    REPORTED_TYPES, FigureError, load_run, map_frame, ramp_bounds, type_means,
)
from tests.test_measure_common import DATA_DIR, needs_data  # noqa: E402

RUN_DIR = DATA_DIR / "phase4_ratified"

EXPECTED_FIGURES = (
    "fig4_psi_by_type.png",
    "fig5_psi_map.png",
    "fig6_jjc_vs_planned.png",
    "figA1_psi_by_type_minmax.png",
    "figA2_psi_map_minmax.png",
)


def frame(rows):
    return pd.DataFrame(
        rows, columns=["USO_AREA_U", "category", "rel_psi", "norm_psi"])


def test_load_run_finds_both_denominators_and_names_the_glob_when_missing(
        tmp_path):
    for denom in ("pop", "popdensity"):
        frame([("A", "Planned", 1.0, 0.5)]).to_csv(
            tmp_path / f"delhi_psi_ratified-2026_{denom}_2020.csv",
            index=False)
    got = load_run(tmp_path)
    assert set(got) == {"pop", "popdensity"}
    assert len(got["pop"]) == 1

    with pytest.raises(FigureError) as exc:
        load_run(tmp_path / "nope")
    message = str(exc.value)
    assert "delhi_psi_*" in message and "popdensity" in message


def test_load_run_rejects_an_ambiguous_match(tmp_path):
    for denom in ("pop", "popdensity"):
        for profile in ("a", "b"):
            frame([("A", "Planned", 1.0, 0.5)]).to_csv(
                tmp_path / f"delhi_psi_{profile}_{denom}_2020.csv",
                index=False)
    with pytest.raises(FigureError, match="expected exactly one"):
        load_run(tmp_path)


def test_type_means_keeps_only_reported_types_and_sorts_by_value():
    means = type_means(frame([
        ("A", "Planned", 1.0, 0.5),
        ("B", "Planned", 1.0, 0.5),
        ("C", "JJC", 0.5, 0.2),
        ("D", "JJC", 0.5, 0.2),
        ("E", "RV", 9.0, 9.0),          # not one of the seven: dropped
    ]))
    assert list(means.index) == ["Planned", "JJC"]
    assert means.loc["Planned", "n"] == 2
    assert means.loc["JJC", "mean_rel"] == pytest.approx(0.5)
    assert "RV" not in means.index
    assert set(means.index) <= set(REPORTED_TYPES)


def test_map_frame_keeps_every_settlement_and_marks_unscored():
    settlements = gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B", "C", "D"],
         "USO_FINAL": ["Planned", "JJC", "RV", "Other"]},
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)],
        crs="EPSG:7760")
    got = map_frame(settlements, frame([
        ("A", "Planned", 1.0, 0.5),
        ("B", "JJC", 0.5, 0.2),
    ]))
    assert len(got) == 4
    assert int(got["scored"].sum()) == 2
    unscored = got[~got["scored"]]
    assert set(unscored["USO_AREA_U"]) == {"C", "D"}
    assert unscored["rel_psi"].isna().all()
    assert got.set_index("USO_AREA_U").loc["C", "USO_FINAL"] == "RV"


def test_ramp_bounds_are_the_first_and_ninety_ninth_percentile():
    lo, hi = ramp_bounds(pd.Series(range(101), dtype=float))
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(99.0)


def _run_cli(run_dir, out_dir):
    return subprocess.run(
        [sys.executable, "scripts/render_paper_figures.py",
         "--run-dir", str(run_dir), "--data-dir", str(DATA_DIR),
         "--out-dir", str(out_dir)],
        cwd=REPO, capture_output=True, text=True)


def test_a_missing_run_dir_is_a_clean_error(tmp_path):
    proc = _run_cli(tmp_path / "missing", tmp_path / "out")
    assert proc.returncode == 1
    assert "delhi_psi_*" in proc.stderr
    assert "Traceback" not in proc.stderr


@needs_data
@pytest.mark.skipif(not RUN_DIR.exists(), reason="DEL-32 run not present")
def test_the_five_figures_render_and_are_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = _run_cli(RUN_DIR, a)
    second = _run_cli(RUN_DIR, b)
    assert first.returncode == 0, first.stderr[-4000:]
    assert second.returncode == 0, second.stderr[-4000:]
    assert sorted(p.name for p in a.glob("*.png")) == sorted(EXPECTED_FIGURES)
    for name in EXPECTED_FIGURES:
        assert (a / name).stat().st_size > 20_000, name
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    manifest = (a / "manifest.txt").read_text()
    assert "reported: 4062" in manifest
    assert "unscored: 295" in manifest
