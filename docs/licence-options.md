# Licence options — a decision brief (DEL-56)

**Status:** the choice is Raj's and Bob's; this page exists so the choice is
quick. The repository currently has **no `LICENSE` file**, and the
`pyproject.toml` has no `license` field — deliberately, until the decision
is made.

## What is being licensed

Two different things, and they need not share a licence:

1. **The code** — the pipeline, tests, scripts, fixtures. This is what a
   software licence covers.
2. **The data and the figures** — the PSI outputs, the maps, the tables.
   These are research outputs; a data licence is the usual choice, and it
   can be CC-BY-4.0 while the code is MIT/Apache.

**Third-party position (measured from the installed packages, 2026-10-28):**

| package | licence |
|---|---|
| geopandas, pandas, shapely, joblib, numpy | BSD-3-Clause (numpy also MIT/0BSD/Zlib components) |
| pyproj, pyogrio, pyyaml | MIT |
| matplotlib | PSF-based |
| tqdm | MPL-2.0 AND MIT |
| certifi | MPL-2.0 |
| pillow | MIT-CMU |

All permissive; the only file-level copyleft is MPL-2.0 (tqdm, certifi),
which imposes nothing on our code. **MIT, BSD-3-Clause and Apache-2.0 are
all compatible with this dependency set.**

## The three realistic options for the code

| option | what it gives you | what it costs |
|---|---|---|
| **MIT** | The simplest, most familiar permissive licence. Maximum reuse, least friction for reviewers and collaborators. | No explicit patent grant, no trademark clause. |
| **BSD-3-Clause** | MIT plus a no-endorsement clause (useful if the university's name must not be implied as endorsing derivatives). | Marginally more text; same practical reuse. |
| **Apache-2.0** | Explicit patent grant, trademark clause, NOTICE file machinery. The safest for a public-sector or commercial reuser. | Longer; some downstream projects find it heavier to comply with. |

**Recommendation on the evidence available:** **MIT for the code** and
**CC-BY-4.0 for the data and figures**. MIT matches the dependency set, asks
nothing of a reviewer who wants to run the pipeline, and the attribution
requirement that matters to an academic author is better served by CC-BY on
the outputs plus the paper's citation. Choose Apache-2.0 instead if the
intent is for a government or commercial body to adopt the code and the
patent grant matters more than simplicity.

## What is not a licence question

- The **data-source terms**: the settlement and service layers come from
  Delhi government sources; their terms have to be checked before the
  *derived* outputs are published. That is a separate question from the
  code licence and belongs with DEL-8's release scope.
- The **paper text** — published under the journal's terms, not this repo's.

## To finish DEL-56

1. Pick the code licence (MIT recommended) and the data licence (CC-BY-4.0
   recommended).
2. Add `LICENSE` (code) and `DATA-LICENSE` (data/figures) at the repo root.
3. Add the `license` field to `pyproject.toml` and the `license:` line to
   `CITATION.cff`.
4. This unblocks DEL-47 (ship to HAS / SSRN).
