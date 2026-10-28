# Security policy

This is a research pipeline, not a service. It reads local files and writes
local files; it makes no network calls at runtime.

## Reporting

Please use GitHub's private vulnerability reporting (Security → Report a
vulnerability) rather than a public issue. We will acknowledge within a
week. There is no bug bounty.

## What is in scope

- The pipeline code in `delhi_psi/`, `scripts/` and the test harness.
- The packaged CLI (`delhi-psi`).

## What is not

- The data layers: they are not distributed in this repository, and their
  provenance and terms are a separate question (see
  [`docs/licence-options.md`](docs/licence-options.md)).
- Dependency vulnerabilities: report those upstream; Dependabot keeps the
  lockfile current here.

## Supported versions

Only `main` is supported. Releases are tagged from `main`; older tags are
archived snapshots for reproducibility and receive no fixes.
