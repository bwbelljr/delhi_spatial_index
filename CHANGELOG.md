# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/) conventions. Each WORKPLAN
phase lands as one entry set when its branch merges; the `[Unreleased]`
section accumulates changes on in-flight branches.

## [Unreleased]

### Added
- Repo-scoped `/ship` build-and-ship pipeline command
  (`.claude/commands/ship.md`)
- Phase 1 design spec (`docs/superpowers/specs/2026-08-16-phase1-runnable-pipeline-design.md`)

## [2026-08-16] Repository restructure (pre-phase)

### Changed
- `main` became the default branch (content from `bb_update`); 2020–2021
  code archived under `archive/master-2021/`; `master` and `bb_update`
  branches removed with history preserved in `main`
- README updated for the new layout

### Added
- `WORKPLAN.md` — sequenced plan toward HAS submission, with meta-planning
  decisions and open Raj/group questions
