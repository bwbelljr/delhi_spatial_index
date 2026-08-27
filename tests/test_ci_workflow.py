"""Structural contract for .github/workflows/ci.yml (spec: docs/superpowers/
specs/2026-08-24-ci-workflow-design.md). The workflow is itself under test
so a later edit cannot silently drop --locked or the fixture-drift guard.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def wf():
    assert WORKFLOW.exists(), f"missing {WORKFLOW}"
    return yaml.safe_load(WORKFLOW.read_text())


@pytest.fixture(scope="module")
def run_lines(wf):
    steps = wf["jobs"]["test"]["steps"]
    return [s["run"] for s in steps if "run" in s]


def test_triggers_push_main_and_pull_request(wf):
    # PyYAML parses the bare key `on` as boolean True.
    on = wf.get("on", wf.get(True))
    assert on["push"]["branches"] == ["main"]
    # Bare `pull_request:` — no branches/types filter — so PRs against ANY
    # base branch run (review R1: a base-branch restriction slipped through).
    assert "pull_request" in on
    assert on["pull_request"] is None


def test_concurrency_cancels_in_progress(wf):
    assert wf["concurrency"]["cancel-in-progress"] is True
    assert "github.ref" in wf["concurrency"]["group"]


def test_permissions_read_only(wf):
    assert wf["permissions"] == {"contents": "read"}


def test_single_job_on_ubuntu(wf):
    assert list(wf["jobs"]) == ["test"]
    assert wf["jobs"]["test"]["runs-on"] == "ubuntu-latest"


def test_uses_checkout_and_setup_uv_with_cache(wf):
    steps = wf["jobs"]["test"]["steps"]
    uses = [s.get("uses", "") for s in steps]
    assert any(u.startswith("actions/checkout@v4") for u in uses)
    setup = next(s for s in steps if s.get("uses", "").startswith("astral-sh/setup-uv@v5"))
    assert setup["with"]["enable-cache"] is True


def test_python_pinned_in_workflow(run_lines):
    assert any("uv python install 3.13" in r for r in run_lines)
    assert not (REPO / ".python-version").exists()


def test_sync_is_locked(run_lines):
    assert any("uv sync --locked" in r for r in run_lines)


def test_pytest_treats_warnings_as_errors(run_lines):
    # DEL-23: the suite is warning-free (DEL-26 removed the pandas
    # FutureWarnings), so CI now fails on any new warning.
    pytest_lines = [r for r in run_lines if "uv run pytest" in r]
    assert pytest_lines, "no pytest step"
    assert all("-W error" in r for r in pytest_lines)


def _drift_step_script(wf):
    steps = wf["jobs"]["test"]["steps"]
    runs = [s["run"] for s in steps
            if "scripts/generate_*_fixtures.py" in s.get("run", "")]
    assert len(runs) == 1, "exactly one fixture-drift step expected"
    # The throwaway repo below is not a uv project, so run the generators
    # with this interpreter instead of `uv run python`. Nothing else changes.
    return runs[0].replace("uv run python", shlex.quote(sys.executable))


def _run_drift_guard(tmp_path, wf, *, content, extra=False):
    """Execute the workflow's drift step verbatim inside a fresh git repo
    whose one generator writes tests/fixtures/demo.txt (committed as "v1").
    Returns the step's exit code."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "tests" / "fixtures").mkdir(parents=True)
    (repo / "scripts" / "generate_demo_fixtures.py").write_text(
        "import os, pathlib\n"
        "out = pathlib.Path('tests/fixtures')\n"
        "(out / 'demo.txt').write_text(os.environ['DEMO_CONTENT'])\n"
        "if os.environ.get('DEMO_EXTRA'):\n"
        "    (out / 'extra.txt').write_text('new file')\n")
    (repo / "tests" / "fixtures" / "demo.txt").write_text("v1")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    subprocess.run([*git, "add", "-A"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-qm", "v1"], cwd=repo, check=True)
    env = {**os.environ, "DEMO_CONTENT": content}
    if extra:
        env["DEMO_EXTRA"] = "1"
    proc = subprocess.run(["bash", "-e", "-c", _drift_step_script(wf)],
                          cwd=repo, env=env, capture_output=True, text=True)
    return proc.returncode


def test_drift_guard_passes_when_generators_match_committed(tmp_path, wf):
    assert _run_drift_guard(tmp_path, wf, content="v1") == 0


def test_drift_guard_fails_when_generator_output_changed(tmp_path, wf):
    # Proves the generators are actually re-run, the polarity is right,
    # and the step exits non-zero (review R1: three substring gaps).
    assert _run_drift_guard(tmp_path, wf, content="v2") != 0


def test_drift_guard_fails_on_untracked_generator_output(tmp_path, wf):
    # `git diff` alone would miss this (plan review R1 Critical).
    assert _run_drift_guard(tmp_path, wf, content="v1", extra=True) != 0


def test_no_data_dependency(wf):
    text = WORKFLOW.read_text()
    for forbidden in ("delhi_data", "DELHI_DATA_DIR", "verify_against_baseline"):
        assert forbidden not in text
