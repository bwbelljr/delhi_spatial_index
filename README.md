# Delhi Public Services Index

Code and a verification oracle for the spatial index of public-service access
described in the forthcoming paper *"Towards an Urban Public Services Index"*
from Georgetown's [Urban Spatial Observatory](https://www.urbanspatialobservatory.org/).

The index scores each of Delhi's settlements on how much public service — banks,
health facilities, police, ration shops, schools, transport, roads — is
reachable from it, corrected for population, and compares those scores across
settlement types (planned colonies, unauthorised colonies, JJ clusters, and so
on).

**What this repository is, in one line:** the implementation of that method,
plus a hand-built oracle that lets anyone check the implementation is correct
without needing Delhi's data.

## What is here, and what is not

| | |
|---|---|
| **The code** | Yes — the `delhi_psi` package and its pipeline. |
| **The oracle** | Yes — two small hand-built cities with hand-derived expected values, and an independent second implementation to check the first against. |
| **Delhi's data** | **No.** The settlement, population, service and barrier layers are third-party and are not redistributed here. |

So you can verify that this code computes what the paper says it computes. You
cannot recompute Delhi's published numbers from this repository alone — that
needs the layers, which are not ours to hand out. Researchers who need them
should contact the authors.

## Quickstart — no data required

Everything in this section runs on a fresh clone with nothing else installed
and no external data.

```bash
# 1. Install uv:  https://docs.astral.sh/uv/
git clone https://github.com/bwbelljr/delhi_spatial_index.git
cd delhi_spatial_index
uv sync

# 2. Run the whole test suite, including the oracle
uv run pytest -q

# 3. Or just the oracle, which is the fast part
uv run pytest -q tests/test_oracle.py tests/test_oracle_e2e.py tests/test_reference_impl.py
```

Measured on a fresh clone with no data present: the full suite is
**914 passed, 19 skipped in about 80 seconds**, and the oracle alone is
**68 passed in about 9 seconds**. The 19 skips are the tests that need
Delhi's layers; they skip rather than fail, which is why the suite is green
without them.

If that passes, the index math on this machine agrees with the hand-derived
answers in [`docs/oracle/derivation-worksheet.md`](docs/oracle/derivation-worksheet.md)
to twelve decimal places.

## What the oracle proves

Most research code is checked by running it and looking at the output. That
finds crashes, not wrong answers. This repository is checked a second way.

- **Two independent implementations.** `delhi_psi/` is the production
  pipeline. [`tests/reference_impl.py`](tests/reference_impl.py) is a separate,
  deliberately naive implementation of the same equations, written to import
  nothing from the package it checks. Every rule is scored by both and compared
  at `1e-12`.
- **Two hand-built cities.** *Oraculum* is seven settlements laid out on graph
  paper so that every expected value could be worked out by hand — the
  arithmetic is in
  [`docs/oracle/derivation-worksheet.md`](docs/oracle/derivation-worksheet.md),
  and the maps are in [`docs/oracle/`](docs/oracle/). The *messy* city adds the
  pathologies a real layer has: overlapping polygons, a settlement inside
  another, corner-only contact, duplicate geometries.
- **When they disagree, the hand arithmetic wins.** That is what makes it an
  oracle rather than a fixture. It has caught real defects — a min-max that
  divided 0/0 on a degenerate group, an over-subtraction in the overlap rule,
  and the exact behaviour of a distance band when a pair sits precisely on the
  radius.

It does **not** prove the method is the right method, or that Delhi's input
layers are accurate. It proves the code implements the documented method
faithfully.

## Porting the method to another city

The code is written around Delhi's data model, not as a general-purpose tool:
the default projection is EPSG:7760, the settlement vocabulary is Delhi's
`uso-10` scheme, the service taxonomy is the paper's, and one optional layer is
literally the distance to Delhi's city centre. **Pointing it at another city's
files is not expected to work without real adaptation.**

What this repository offers a port instead:

1. **A reference implementation** to read, precise where prose would be
   ambiguous.
2. **The method written down** — [`docs/methodology-config.md`](docs/methodology-config.md)
   lists every methodology switch, its allowed values, and what each one
   changes.
3. **The oracle to check your port against.** The fixture cities and their
   hand-derived expected values are the useful export here: if your
   reimplementation reproduces them, it computes the same index.

## Running on Delhi data

For those who have the layers. Every methodology choice is a config value, not
code:

```bash
uv run delhi-psi preprocess --config code-2025 --data-dir ~/delhi_data --out-dir ~/delhi_data/run
uv run delhi-psi compute    --config code-2025 --data-dir ~/delhi_data --out-dir ~/delhi_data/run
uv run python scripts/verify_against_baseline.py --config code-2025 \
    --data-dir ~/delhi_data --verify-dir ~/delhi_data/run
```

`--config` takes a shipped profile name or a path to a YAML file. Two profiles
ship: `code-2025` reproduces the July 2025 production behaviour exactly, and
`manuscript` implements the paper's equations as written. To change a
methodology choice, or turn a decision into a new profile, follow
[`docs/methodology-config.md`](docs/methodology-config.md).

## Repository layout

* `delhi_psi/` — the installable package: `config` (profiles and validation),
  `geometry`, `neighbors`, `index` (the math, as pure functions with keyword
  knobs), `io`, `validate`, `pipeline` (the stages), `cli`, `verify`
* `delhi_psi/profiles/` — one YAML per methodology profile
* `tests/` — the oracle, the reference implementation, the fixture cities and
  their committed expected values
* `docs/oracle/` — the derivation worksheet and the fixture-city maps
* `docs/methodology-config.md` — how to change a methodology choice
* `docs/data/` — measurements taken from the real layers, each regenerated by
  a script and drift-tested against it
* `docs/decisions/` — the record of methodology decisions and who made them
* `scripts/` — the pipeline's supporting tools: baseline verification, fixture
  generation, measurement, and the Phase 6 sweep
* `archive/master-2021/` — the original 2020–21 notebooks that produced the
  first published numbers, kept as a frozen snapshot. See
  `archive/master-2021/ARCHIVE_README.md`.
* `WORKPLAN.md` — the project's own plan of record

## Status

This is a repository attached to a paper still in progress, and it is honest
about which parts are settled.

- **Settled and proven:** the pipeline, the config system, the oracle, and
  that the `code-2025` profile reproduces the July 2025 numbers exactly.
- **Open:** several methodology decisions are still being made — see
  [`docs/decisions/`](docs/decisions/) and `WORKPLAN.md`. Numbers in
  `docs/data/` that are labelled as dry runs describe a superseded rule set and
  are not the paper's results.
- **Licence:** not yet chosen. Until one is added, no licence is granted;
  please ask before reusing.

## Citation

The paper is forthcoming. Until it appears, please cite this repository and
contact the authors.
