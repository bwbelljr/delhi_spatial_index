"""Which PSI column and which denominator the paper's figures report
(DEL-52, spec § 2.4).

The scoring functions are proven on hand-built frames whose means are known
by construction; the real comparison against the July 2025 baseline files is
the run step's.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts._measure_common import FENCE, holds_prose, parse_block
from scripts.measure_psi_columns import (BASELINE_FILES, FIGURE_4_BARS,
                                         FIGURE_TOLERANCE, cross_check, main,
                                         score_candidate, score_candidates,
                                         type_means)
from tests.test_measure_common import (DATA_DIR,
                                       assert_prose_numbers_come_from_the_blocks,
                                       needs_data)

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "data" / "psi_columns.md"
BASELINE_DIR = DATA_DIR / "psi_2020_results"
VERIFY_DIR = DATA_DIR / "phase3_verify"
CANDIDATES = ("unnorm_psi_popsize", "unnorm_psi_popdensity",
              "norm_psi_popsize", "norm_psi_popdensity")


def frame(**columns):
    """A tiny PSI-output-shaped frame: one row per (type, value) pair."""
    return pd.DataFrame(columns)


def committed_block():
    if not DOC.exists() or FENCE not in DOC.read_text():
        pytest.skip(f"{DOC} carries no measured block yet — the run step "
                    "pastes it")
    return parse_block(DOC.read_text())


def test_figure_4_bars_has_the_eight_types_from_the_april_2026_draft():
    """Figure 4 has eight bars — no RV, no Other (spec § 2.4)."""
    assert set(FIGURE_4_BARS) == {"JJR", "JJC", "SDA", "Planned", "RUAC",
                                  "UAC", "UV", "Industrial"}
    assert all(0 < value < 0.05 for value in FIGURE_4_BARS.values())
    assert FIGURE_TOLERANCE == 0.002


def test_type_means_averages_per_settlement_type():
    got = type_means(frame(USO_FINAL=["JJC", "JJC", "Planned"],
                           unnorm_psi=[0.0, 0.004, 0.044]),
                     column="unnorm_psi")
    assert got == {"JJC": 0.002, "Planned": 0.044}


def test_score_candidate_counts_matches_and_the_max_gap():
    """Two bars hit exactly, one misses by 0.01 — one match short of the
    figure's eight, and the gap is the miss."""
    means = dict(FIGURE_4_BARS)
    means["JJC"] = FIGURE_4_BARS["JJC"] + 0.01
    matched, maxgap = score_candidate(means)
    assert matched == 7
    assert maxgap == pytest.approx(0.01)


def test_a_missing_figure_type_can_never_match():
    means = {name: value for name, value in FIGURE_4_BARS.items()
             if name != "UV"}
    matched, maxgap = score_candidate(means)
    assert matched == 7
    assert maxgap == float("inf")


def test_score_candidates_picks_the_column_that_reproduces_the_figure():
    """`unnorm_psi` under popdensity carries the figure values exactly;
    every other candidate is stretched to [0, 1] or shifted."""
    types = list(FIGURE_4_BARS)
    exact = [FIGURE_4_BARS[name] for name in types]
    stretched = [value * 20 for value in exact]
    frames = {
        "popsize": frame(USO_FINAL=types, unnorm_psi=stretched,
                         norm_psi=stretched),
        "popdensity": frame(USO_FINAL=types, unnorm_psi=exact,
                            norm_psi=stretched),
    }
    got = score_candidates(frames)
    assert got["best_candidate"] == "unnorm_psi_popdensity"
    assert got["matched_unnorm_psi_popdensity"] == 8
    assert got["maxgap_unnorm_psi_popdensity"] == "0.0000"
    assert got["mean_unnorm_psi_popdensity_JJC"] == "0.0015"
    assert got["matched_unnorm_psi_popsize"] == 0


def test_the_best_candidate_tie_break_prefers_the_smaller_max_gap():
    """Both candidates match the same number of bars; the one that is closer
    on the bar it misses wins."""
    types = list(FIGURE_4_BARS)
    near = [FIGURE_4_BARS[name] for name in types]
    near[0] += 0.003            # just outside the tolerance
    far = [FIGURE_4_BARS[name] for name in types]
    far[0] += 0.009
    frames = {"popsize": frame(USO_FINAL=types, unnorm_psi=far, norm_psi=far),
              "popdensity": frame(USO_FINAL=types, unnorm_psi=near,
                                  norm_psi=far)}
    got = score_candidates(frames)
    assert got["matched_unnorm_psi_popdensity"] == 7
    assert got["matched_unnorm_psi_popsize"] == 7
    assert got["best_candidate"] == "unnorm_psi_popdensity"


def test_cross_check_raises_when_the_refactored_run_disagrees():
    types = list(FIGURE_4_BARS)
    values = [FIGURE_4_BARS[name] for name in types]
    baseline = {"popsize": frame(USO_FINAL=types, unnorm_psi=values,
                                 norm_psi=values)}
    moved = [value + 1e-6 for value in values]
    verify = {"popsize": frame(USO_FINAL=types, unnorm_psi=moved,
                               norm_psi=values)}
    with pytest.raises(ValueError, match="differs from the July 2025 baseline"):
        cross_check(baseline, verify)


def test_cross_check_reports_the_max_difference_when_the_runs_agree():
    types = list(FIGURE_4_BARS)
    values = [FIGURE_4_BARS[name] for name in types]
    frames = {"popsize": frame(USO_FINAL=types, unnorm_psi=values,
                               norm_psi=values)}
    got = cross_check(frames, frames)
    assert got["verify_maxdiff_unnorm_psi_popsize"] == "0.0e+00"


def test_the_doc_block_has_every_required_key():
    committed = committed_block()
    for candidate in CANDIDATES:            # <column>_<denom>
        assert committed[f"matched_{candidate}"].isdigit()
        float(committed[f"maxgap_{candidate}"])
        for name in FIGURE_4_BARS:
            key = f"mean_{candidate}_{name}"
            assert key in committed, key
            float(committed[key])
    assert committed["best_candidate"] in CANDIDATES


def test_the_doc_records_its_provenance_and_quotes_only_block_numbers():
    text = DOC.read_text()
    for label in ("**Run date:**", "**Inputs:**", "**Commit:**",
                  "**Command:**"):
        assert label in text, label
    assert_prose_numbers_come_from_the_blocks(text, [committed_block()])


@needs_data
def test_a_fresh_run_reproduces_the_committed_block(tmp_path):
    """This one keeps the PLAIN `needs_data` gate — no DELHI_PSI_MEASURE_CACHE
    marker. The script reads two CSVs and never touches the settlement layer,
    so there is no dedup and no O(n^2) pass: it costs seconds, and it is
    honest for it to run in every implementer's suite run on this machine."""
    committed = committed_block()
    proc = subprocess.run(
        [sys.executable, "scripts/measure_psi_columns.py",
         "--baseline-dir", str(BASELINE_DIR), "--verify-dir", str(VERIFY_DIR),
         "--data-dir", str(DATA_DIR), "--work-dir", str(tmp_path / "work")],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert parse_block(proc.stdout) == committed


def test_main_prints_its_usage_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--baseline-dir" in capsys.readouterr().out


# --- DEL-59: --out / --splice ------------------------------------------
def test_out_and_splice_appear_in_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--out" in help_text
    assert "--splice" in help_text


def test_out_and_splice_together_are_refused_by_argparse(tmp_path):
    splice_target = tmp_path / "doc.md"
    splice_target.write_text("existing doc\n")
    with pytest.raises(SystemExit) as exc_info:
        main(["--out", str(tmp_path / "dump.md"),
              "--splice", str(splice_target)])
    assert exc_info.value.code == 2


def test_out_refuses_to_overwrite_a_target_that_holds_prose(tmp_path):
    """--out writes blocks only; a target already holding hand-written prose
    must be refused (exit 1) rather than clobbered, and left untouched."""
    target = tmp_path / "prose.md"
    before = "# PSI columns\n\nA hand-written caption.\n"
    target.write_text(before)
    with pytest.raises(SystemExit):
        main(["--out", str(target)])
    assert target.read_text() == before


def _write_below_floor_baseline(baseline_dir):
    """A baseline whose only settlement type ("JJR") is nowhere near any
    Figure 4 bar for either candidate column — matched is 0 for every
    candidate, well under MATCH_FLOOR, for both --out and plain stdout."""
    baseline_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "USO_FINAL": ["JJR"],
        "unnorm_psi": [5.0],
        "norm_psi": [5.0],
    })
    for name in BASELINE_FILES.values():
        frame.to_csv(baseline_dir / name, index=False)


def test_a_match_below_match_floor_is_refused_when_out_is_given(tmp_path):
    """DEL-59 fix round item 3: --out must not write a document that carries
    a match this weak — refuse (non-zero exit, the warning as the message)
    rather than write suspect numbers and bury the warning under stderr's
    tqdm noise with exit 0."""
    baseline_dir = tmp_path / "baseline"
    _write_below_floor_baseline(baseline_dir)
    target = tmp_path / "out.md"
    with pytest.raises(SystemExit) as exc_info:
        main(["--baseline-dir", str(baseline_dir),
              "--data-dir", str(tmp_path / "data"),
              "--work-dir", str(tmp_path / "work"),
              "--out", str(target)])
    assert "WARNING" in str(exc_info.value)
    assert "escalate, do not guess" in str(exc_info.value)
    assert not target.exists()


def test_a_match_below_match_floor_still_warns_and_exits_zero_on_stdout(
        capsys, tmp_path):
    """Plain stdout runs (no --out/--splice) keep today's behaviour: warn on
    stderr and exit 0."""
    baseline_dir = tmp_path / "baseline"
    _write_below_floor_baseline(baseline_dir)
    result = main(["--baseline-dir", str(baseline_dir),
                  "--data-dir", str(tmp_path / "data"),
                  "--work-dir", str(tmp_path / "work")])
    assert result == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert FENCE in captured.out


@needs_data
def test_stdout_carries_blocks_and_nothing_else(tmp_path):
    """DEL-59: the provenance lines move to stderr so --out and --splice
    have a clean stream to work with. `parse_block` always ignored those
    lines, so this is the first test that would notice one coming back.

    `holds_prose` is the whole-stream check, and it is the assertion that
    matters: it is true of ANY non-blank line outside a fenced block, so it
    catches every missed diagnostic rather than one named one.

    The `WARNING:` line is not asserted unconditionally here: against this
    machine's real baseline/verify data the best candidate matches all 8
    bars (well above MATCH_FLOOR), so the warning never prints at all —
    `not holds_prose(proc.stdout)` still catches it if it ever leaked to
    stdout.
    """
    proc = subprocess.run(
        [sys.executable, "scripts/measure_psi_columns.py",
         "--baseline-dir", str(BASELINE_DIR), "--verify-dir", str(VERIFY_DIR),
         "--data-dir", str(DATA_DIR), "--work-dir", str(tmp_path / "work")],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert not holds_prose(proc.stdout)
    assert "baseline-dir:" in proc.stderr
    assert "verify-dir:" in proc.stderr
    assert "work-dir:" in proc.stderr
