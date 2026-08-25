# GitHub Actions CI — Design Spec (DEL-27)

Date: 2026-08-24
Status: **draft — awaiting owner review**
Branch: `del-27-ci` (off `origin/main`)
Parent plan: `WORKPLAN.md` Phase 3, item "Add GitHub Actions CI"; Jira DEL-27

## Purpose

Run the oracle test suite automatically on every push and pull request so
that no change to `spatial_index_utils.py` or the pipeline scripts can reach
`main` without the 65-test oracle passing. Today the suite exists but runs
only when someone remembers to run it. Every later Phase 3 ticket (pandas
uncap, dead-code removal, the refactor, the bug fixes) lands as a PR; this
workflow is what makes their green check mean something.

## What a green check means

A green check on a commit certifies, on a clean Ubuntu runner with no
access to the Delhi data:

1. `uv.lock` is consistent with `pyproject.toml` (`uv sync --locked`).
2. All tests pass under Python 3.13 (`uv run pytest`). This already
   includes the expected-values CSV round-trip
   (`test_expected_values_csv_is_regenerable`) and the CSV-wide invariant
   guard via its pytest wrapper.
3. The committed GeoJSON fixtures are byte-identical to what
   `scripts/generate_oraculum_fixtures.py` produces — i.e. nobody
   hand-edited a fixture or changed the generator without regenerating.

It does **not** certify: agreement with the July 2025 baseline
(`scripts/verify_against_baseline.py` needs `~/delhi_data`, stays a local
step), figure freshness (matplotlib PNGs are not byte-stable across
runners), or absence of warnings (364 today; DEL-23's job — `-W error`
is a follow-up once that lands).

## Design

One workflow file, `.github/workflows/ci.yml`, one job.

**Triggers.** `push` to `main`; `pull_request` targeting any branch. A
`concurrency` group keyed on the ref with `cancel-in-progress: true`, so
rapid successive pushes to a PR run only the latest.

**Job `test`** on `ubuntu-latest`:

| step | action / command | why |
|---|---|---|
| checkout | `actions/checkout@v4` | |
| uv | `astral-sh/setup-uv@v5`, `enable-cache: true` | cache keyed on `uv.lock`; warm runs ≈ 30 s |
| python | `uv python install 3.13` | repo has no `.python-version`; pin in the workflow rather than add a file |
| deps | `uv sync --locked` | fails on a stale lockfile — the check DEL-26 will rely on |
| tests | `uv run pytest` | plain; no `-W error` yet |
| fixtures | `uv run python scripts/generate_oraculum_fixtures.py && git diff --exit-code -- tests/fixtures/` | generator/fixture drift guard |

Permissions: `contents: read` only. No secrets, no data download, no
matrix (the project pins `>=3.13`; one interpreter is enough for a
research pipeline).

## Verification plan

1. Open a PR from `del-27-ci`; the first run must be green.
2. Push a deliberate sabotage commit on the same branch — change one
   value in `tests/fixtures/oraculum/expected_values.csv` — and confirm
   the run goes red at the pytest step. Revert it. This is the only proof
   the check has teeth; do it once, in the PR, before merge.
3. Merge via the normal PR route.

## Follow-ups (not in this ticket)

- Mark the `test` job as a required status check in branch protection for
  `main` — a GitHub settings change, done by the owner after the first
  green run on `main`.
- README badge.
- `-W error` once DEL-23 clears the warnings.
- Dependabot for GitHub Actions versions, if the pinned action majors ever
  matter.

## Autonomy and stopping rules

Same terms as the Phase 1/2 specs: implement, push, open the PR, run the
verification plan, fix forward. Stop and ask if anything requires a
repository *settings* change (branch protection, secrets) or if the
first run fails for a reason that implies a real bug in the suite rather
than in the workflow.
