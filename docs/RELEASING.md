# Releasing (DEL-8 / DEL-47)

The release is a **reproducibility artefact plus a paper submission**: the
tagged code, the outputs, and the tables/figures the paper cites. Nothing
here is publishable until the licence question is answered (DEL-56) — the
checklist enforces that.

## Before tagging

- [ ] `uv run pytest -q -W error` green on `main`, including the real-data
      checks (`DELHI_DATA_DIR` set, `DELHI_PSI_MEASURE_CACHE` set where the
      suite asks for it).
- [ ] `CHANGELOG.md`'s `[Unreleased]` section describes everything since the
      last tag; move it under a dated version heading.
- [ ] `pyproject.toml` version bumped; `CITATION.cff` `version` and
      `date-released` match.
- [ ] **`LICENSE` and `DATA-LICENSE` present** (DEL-56), `license` fields in
      `pyproject.toml` and `CITATION.cff` filled.
- [ ] `docs/data/*.md` measurement blocks regenerate cleanly
      (`scripts/measure_*.py --out /tmp/check` diffed against the committed
      files).
- [ ] `WORKPLAN.md` synced to the Jira board.
- [ ] No transcripts, call attachments, credentials or large data files in
      the tree (`git ls-files | xargs -n1 du -h | sort -h | tail`).

## The release

```bash
uv build                              # sdist + wheel in dist/
git tag -a v0.4.0 -m "v0.4.0 — ratified methodology"
git push origin v0.4.0
gh release create v0.4.0 dist/* --title "v0.4.0" --notes-file notes.md
```

Attach, or copy to the shared drive for HAS:

- the tagged outputs under `~/delhi_data/phase4_ratified/` (dated delivery
  folder), with its `README.txt` manifest;
- the figures (`docs/figures/ratified-2026/`) and the measurement tables
  (`docs/data/`);
- the paper's submission bundle (external).

## After

- Post the paper to SSRN once the journal's embargo terms allow (DEL-47).
- Record in the Jira release epic: tag, archive location, submission date.
- Keep `main` the only supported branch; the tag is the snapshot.

## Rollback

A release is immutable. If a number is wrong after tagging, do not move the
tag: cut a patch release, and say in its notes which table it corrects.
