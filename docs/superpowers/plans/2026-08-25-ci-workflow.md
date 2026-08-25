# GitHub Actions CI (DEL-27) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A GitHub Actions workflow that runs the oracle test suite and a fixture-drift guard on every push to `main` and every pull request, so a green check certifies the three things the spec lists.

**Architecture:** One workflow file, `.github/workflows/ci.yml`, one job on `ubuntu-latest`. A structural pytest (`tests/test_ci_workflow.py`) parses the YAML and asserts the spec's contract, so the workflow is itself under test and a later edit that drops `--locked` or the drift step fails locally. PyYAML joins the dev dependency group for that test only. The proof that the workflow works on GitHub is the PR itself (green run, then a deliberate red run, reverted).

**Tech Stack:** GitHub Actions (`actions/checkout@v4`, `astral-sh/setup-uv@v5`), uv, pytest, PyYAML (dev only).

**Spec:** `docs/superpowers/specs/2026-08-24-ci-workflow-design.md`

## Global Constraints

- Python pinned in the workflow: `uv python install 3.13` (no `.python-version` file added).
- Dependencies installed with `uv sync --locked` — must fail on a stale `uv.lock`.
- `uv run pytest` plain — **no** `-W error` (364 warnings today; DEL-23).
- Fixture generators are named `scripts/generate_*_fixtures.py` and write only under `tests/fixtures/`; the drift step globs that pattern and diffs `tests/fixtures/`.
- Workflow `permissions: contents: read` only. No secrets, no data download, no matrix.
- Triggers: `push` to `main`, `pull_request` (any target). `concurrency` keyed on the ref, `cancel-in-progress: true`.
- Nothing in the workflow may reference `~/delhi_data` or the baseline; `verify_against_baseline.py` stays local.
- Branch: `del-27-ci`. Every /ship run updates `CHANGELOG.md` `[Unreleased]`.

---

### Task 1: Structural test + workflow file

**Files:**
- Modify: `pyproject.toml` (dev dependency group — add `pyyaml`)
- Create: `tests/test_ci_workflow.py`
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: nothing.
- Produces: `.github/workflows/ci.yml` with a single job named `test`; `tests/test_ci_workflow.py` reading it via `WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"`.

- [ ] **Step 1: Add PyYAML to the dev group and sync**

In `pyproject.toml`, change the dev group to:

```toml
[dependency-groups]
dev = [
    "pytest>=8.4",
    "pyyaml>=6.0",
]
```

Run: `uv sync` (updates `uv.lock`). Expected: lockfile gains `pyyaml`; `uv run python -c "import yaml"` succeeds.

- [ ] **Step 2: Write the failing structural test**

Create `tests/test_ci_workflow.py`:

```python
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_ci_workflow.py -q`
Expected: every test errors at the `wf` fixture with `AssertionError: missing .../ci.yml`.

- [ ] **Step 4: Write the workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Install Python
        run: uv python install 3.13

      - name: Install dependencies (lockfile must be current)
        run: uv sync --locked

      - name: Oracle test suite
        run: uv run pytest

      - name: Fixture drift guard (generators must match committed fixtures)
        run: |
          for g in scripts/generate_*_fixtures.py; do
            uv run python "$g"
          done
          # Any modified, deleted OR untracked file under tests/fixtures/ fails.
          # (git diff alone ignores untracked files — plan review R1, Critical.)
          if [ -n "$(git status --porcelain -- tests/fixtures/)" ]; then
            git status --short -- tests/fixtures/
            echo "::error::committed fixtures do not match their generators"
            exit 1
          fi
```

- [ ] **Step 5: Run the test to verify it passes, and the whole suite**

Run: `uv run pytest tests/test_ci_workflow.py -q` — Expected: 10 passed.
Run: `uv run pytest -q` — Expected: 75 passed (65 + 10).

- [ ] **Step 6: Rehearse the drift step locally**

Run exactly the drift step's shell:

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done \
  && test -z "$(git status --porcelain -- tests/fixtures/)" && echo DRIFT-OK
```

Expected: `wrote fixtures to …` then `DRIFT-OK`, exit 0. (Proves the generator is deterministic against the committed fixtures before GitHub runs it.)

Then prove the guard sees an untracked file (the review-R1 Critical):

```bash
touch tests/fixtures/oraculum/zz_untracked.geojson
test -z "$(git status --porcelain -- tests/fixtures/)" && echo DRIFT-OK || echo DRIFT-CAUGHT
rm tests/fixtures/oraculum/zz_untracked.geojson
```

Expected: `DRIFT-CAUGHT`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock tests/test_ci_workflow.py .github/workflows/ci.yml
git commit -m "ci: GitHub Actions workflow — oracle suite + fixture drift guard (DEL-27)"
```

---

### Task 2: Changelog

**Files:**
- Modify: `CHANGELOG.md` (`[Unreleased]` section)

**Interfaces:** none.

- [ ] **Step 1: Add the entry**

Under `## [Unreleased]`, add:

```markdown
- CI: `.github/workflows/ci.yml` runs `uv sync --locked`, the oracle suite
  and a fixture-drift guard (regenerate `scripts/generate_*_fixtures.py`,
  `git diff --exit-code tests/fixtures/`) on every push to `main` and every
  PR. Drift guard uses `git status --porcelain` so untracked generator output also fails. Structural contract pinned by `tests/test_ci_workflow.py`; PyYAML added
  to the dev group. Spec: `docs/superpowers/specs/2026-08-24-ci-workflow-design.md`.
```

- [ ] **Step 2: Commit and push the branch**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog entry for CI workflow (DEL-27)"
git push -u origin del-27-ci
```

---

### Task 3: Prove it on GitHub, then ship

This task is the spec's verification plan; it needs the remote.

**Files:** none new. Touches `tests/fixtures/oraculum/expected_values.csv` in a commit that is reverted before merge.

- [ ] **Step 1: Open the PR**

```bash
gh pr create --base main --head del-27-ci \
  --title "ci: GitHub Actions — oracle suite + fixture drift guard (DEL-27)" \
  --body-file - <<'EOF'
Implements docs/superpowers/specs/2026-08-24-ci-workflow-design.md.

A green check certifies: uv.lock current (`uv sync --locked`); 75 tests pass on Python 3.13; committed fixtures match `scripts/generate_*_fixtures.py` (porcelain check — new/untracked output fails too).

Verification per spec: first run green; a deliberate sabotage commit (one CSV value changed) turned the check red; reverted before merge — see commit history.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
EOF
```

- [ ] **Step 2: Wait for the first run; it must be green**

Run: `gh pr checks --watch` (or `gh run watch`). Expected: job `test` succeeds. If it fails for a workflow reason (action version, cache, shell), fix forward and push. If it fails inside pytest for a reason that implies a real suite bug, **stop and ask** (spec's stopping rule).

- [ ] **Step 3: Sabotage commit — prove the check has teeth**

```bash
python3 - <<'EOF'
p='tests/fixtures/oraculum/expected_values.csv'
s=open(p).read().splitlines(keepends=True)
# line 3 is 'ideal,baseline,pop,A,clinic_pcen,0.0291421356...'; corrupt it
assert s[2].startswith('ideal,baseline,pop,A,clinic_pcen,')
s[2]='ideal,baseline,pop,A,clinic_pcen,0.03\n'
open(p,'w').write(''.join(s))
EOF
git commit -am "test(ci): SABOTAGE — corrupt one expected value; must turn CI red (will be reverted)"
git push
gh pr checks --watch
```

Expected: job `test` **fails** at the "Oracle test suite" step (the reference-vs-CSV comparison and the round-trip test both catch it). Record the failing run URL.

- [ ] **Step 4: Revert the sabotage**

```bash
git revert --no-edit HEAD
git push
gh pr checks --watch
```

Expected: green again. Confirm `git diff main -- tests/fixtures/` is empty.

- [ ] **Step 5: Merge**

```bash
gh pr merge --squash --delete-branch
git checkout main && git pull
```

Expected: `main` contains `.github/workflows/ci.yml`; the push to `main` triggers a run; `gh run list --branch main --limit 1` shows success.

- [ ] **Step 6: Sync trackers**

Tick WORKPLAN Phase 3 "Add GitHub Actions CI [DEL-27]"; transition DEL-27 → Done with the PR link in a comment. Owner follow-up (not this run): make `test` a required check in branch protection.
