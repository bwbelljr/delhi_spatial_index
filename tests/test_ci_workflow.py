"""Structural contract for .github/workflows/ci.yml (spec: docs/superpowers/
specs/2026-08-24-ci-workflow-design.md). The workflow is itself under test
so a later edit cannot silently drop --locked or the fixture-drift guard.
"""
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
    assert "pull_request" in on


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


def test_pytest_plain_no_warnings_as_errors(run_lines):
    pytest_lines = [r for r in run_lines if "uv run pytest" in r]
    assert pytest_lines, "no pytest step"
    assert all("-W" not in r for r in pytest_lines)


def test_fixture_drift_guard_globs_generators_and_sees_untracked(run_lines):
    drift = [r for r in run_lines if "scripts/generate_*_fixtures.py" in r]
    assert len(drift) == 1
    # Must use porcelain status, not `git diff`: diff ignores untracked files,
    # so a generator gaining a new output file would pass vacuously.
    assert "git status --porcelain -- tests/fixtures/" in drift[0]
    assert "git diff" not in drift[0]


def test_no_data_dependency(wf):
    text = WORKFLOW.read_text()
    for forbidden in ("delhi_data", "DELHI_DATA_DIR", "verify_against_baseline"):
        assert forbidden not in text
