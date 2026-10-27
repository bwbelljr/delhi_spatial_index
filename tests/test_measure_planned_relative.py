"""The relative-scale block must be exactly what the generator emits.

`docs/data/planned_relative.md` states Raj's reported scale (DEL-63): each
type's mean `rel_psi`, where 1 is the average Planned colony. The pure
tests pin the arithmetic and the guards; the `needs_data` test proves a
fresh run reproduces the committed block byte for byte.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from delhi_psi.config import load_config
from scripts._measure_common import FENCE, parse_block
from scripts.measure_planned_relative import (
    DENOMINATORS, REPORTED_TYPES, main, measure_denominator,
)
from tests.test_measure_common import (
    DATA_DIR, assert_prose_numbers_come_from_the_blocks, needs_data,
)

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "data" / "planned_relative.md"
VERIFY_DIR = DATA_DIR / "phase3_verify"


def committed_block():
    if not DOC.exists() or FENCE not in DOC.read_text():
        pytest.skip(f"{DOC} carries no measured block yet — the run step "
                    "pastes it")
    return parse_block(DOC.read_text(), name="planned_relative")


def frame(**columns):
    return pd.DataFrame(columns)


def test_the_block_covers_every_reported_type():
    assert REPORTED_TYPES == ("Planned", "UAC", "RUAC", "JJC", "JJR", "UV",
                              "SDA", "Industrial", "Other")
    assert DENOMINATORS == ("pop", "popdensity")


def test_measure_denominator_reports_both_columns_and_the_identity():
    norm = pd.Series([0.2, 0.4, 0.6])
    planned_mean = float(norm[:2].mean())
    got = measure_denominator(
        frame(USO_FINAL=["Planned", "Planned", "UAC"],
              norm_psi=norm, rel_psi=norm / planned_mean),
        denom="pop", types=("Planned", "UAC"))
    assert got["planned_norm_mean_pop"] == f"{planned_mean:.6g}"
    assert got["mean_norm_psi_pop_Planned"] == f"{norm[:2].mean():.6g}"
    assert got["mean_rel_psi_pop_Planned"] == \
        f"{(norm[:2] / planned_mean).mean():.6g}"
    assert got["mean_rel_psi_pop_UAC"] == f"{(norm[2] / planned_mean):.6g}"
    assert got["rel_vs_ratio_maxdiff_pop"] == "0"


def test_measure_denominator_refuses_a_city_with_no_planned_rows():
    with pytest.raises(ValueError, match="'Planned'"):
        measure_denominator(
            frame(USO_FINAL=["UAC"], norm_psi=[1.0], rel_psi=[1.0]),
            denom="pop", types=("Planned", "UAC"))


def test_measure_denominator_refuses_a_zero_planned_mean():
    with pytest.raises(ValueError, match="mean is 0.0"):
        measure_denominator(
            frame(USO_FINAL=["Planned", "UAC"], norm_psi=[0.0, 1.0],
                  rel_psi=[0.0, 0.0]),
            denom="pop", types=("Planned", "UAC"))


def test_the_script_refuses_a_stamp_that_shapes_the_neighbours(
        tmp_path, monkeypatch):
    """If `methodology_stamp` ever gains this key, --verify-dir's artifact
    is no longer valid and the script must stop BEFORE staging it."""
    from delhi_psi import pipeline
    from scripts import measure_planned_relative as script

    monkeypatch.setattr(
        pipeline, "methodology_stamp",
        lambda methodology: {"second_normalization": {"rule": "minmax"}})
    with pytest.raises(SystemExit, match="second_normalization"):
        script.measure(load_config("code-2025"), tmp_path,
                       base="code-2025", verify_dir=tmp_path)


def test_the_doc_block_has_every_required_key():
    block = committed_block()
    for denom in DENOMINATORS:
        assert float(block[f"planned_norm_mean_{denom}"]) > 0
        for name in REPORTED_TYPES:
            for column in ("rel_psi", "norm_psi"):
                key = f"mean_{column}_{denom}_{name}"
                assert key in block, key
                float(block[key])
        assert float(block[f"rel_vs_ratio_maxdiff_{denom}"]) < 1e-12


def test_the_planned_category_reads_one_on_the_relative_scale():
    block = committed_block()
    for denom in DENOMINATORS:
        assert float(block[f"mean_rel_psi_{denom}_Planned"]) == (
            pytest.approx(1.0, abs=1e-9))


def test_popdensity_orders_the_types_as_the_spot_values_did():
    block = committed_block()
    means = {name: float(block[f"mean_rel_psi_popdensity_{name}"])
             for name in REPORTED_TYPES}
    assert means["UV"] > means["RUAC"] > means["UAC"] > means["JJC"]
    assert means["UV"] == pytest.approx(0.86, abs=0.05)


def test_the_doc_records_its_provenance_and_quotes_only_block_numbers():
    text = DOC.read_text()
    for label in ("**Run date:**", "**Inputs:**", "**Commit:**",
                  "**Command:**"):
        assert label in text, label
    assert_prose_numbers_come_from_the_blocks(text, [committed_block()])


@needs_data
def test_a_fresh_run_reproduces_the_committed_block(tmp_path):
    """PLAIN `needs_data`, not the measure-cache gate: the script reads two
    CSVs and one staged joblib and runs `compute` alone — no settlement
    dedup and no O(n^2) adjacency pass."""
    committed = committed_block()
    proc = subprocess.run(
        [sys.executable, "scripts/measure_planned_relative.py",
         "--config", "code-2025", "--data-dir", str(DATA_DIR),
         "--verify-dir", str(VERIFY_DIR),
         "--work-dir", str(tmp_path / "work")],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert parse_block(proc.stdout, name="planned_relative") == committed


def test_main_prints_its_usage_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--verify-dir" in capsys.readouterr().out


def test_out_refuses_to_overwrite_a_target_that_holds_prose(tmp_path):
    target = tmp_path / "prose.md"
    before = "# Planned relative\n\nA hand-written caption.\n"
    target.write_text(before)
    with pytest.raises(SystemExit):
        # --verify-dir is required by argparse; the prose guard fires in
        # emit_check() before the path is ever read.
        main(["--out", str(target), "--verify-dir", str(tmp_path)])
    assert target.read_text() == before
