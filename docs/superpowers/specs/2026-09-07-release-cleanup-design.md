# Release cleanup — DEL-45 (and DEL-46's front door)

**Ticket:** DEL-45, with the README half of DEL-46
**Branch:** `del-45-release-cleanup` off `main` at `27e2f62`
**Date:** 7 Sep 2026
**Authorisation:** Bob's 5 Sep three-cycle run, cycle 3 of 3. Explicit
constraint: **no licence file** — Bob settles that with Raj.

---

## 1. The problem, stated concretely

The ticket says people "will run the repo (and point Claude at it) first thing,
so find issues before they do." So the test is: **a stranger clones this repo
and types the first command in the README. What happens?**

Today it fails. Every command under *Running the pipeline* needs
`--data-dir ~/delhi_data`, and the README says plainly that the data is not in
the repository. The quickstart is unrunnable by exactly the audience it is
written for, and the reader's first experience is a missing-file error.

That is the defect this cycle exists to fix, and it has a clean fix that costs
nothing: **the repo already contains a complete, self-contained, runnable
proof of the method** — the Oraculum and messy fixture cities, the independent
reference implementation, and 923 tests that need no external data. Leading
with those turns a broken quickstart into a working one and simultaneously
delivers DEL-46's "oracle as a documented release feature".

## 2. Scope

**In:**

1. A README restructured so the first runnable command needs no data, with a
   Delhi section for readers who have the layers.
2. The oracle presented as a feature: what it is, why a reader should care,
   how to run it, what it proves.
3. A data-access section written to the **code + fixtures floor** — what ships
   today, what is pending, who decides.
4. An honest statement of what is and is not settled: this is a repo attached
   to a paper in progress, and several methodology decisions are open.
5. A decision about what ships: `archive/` (1.7 MB of 2021 notebooks),
   `WORKPLAN.md` (44 KB), `docs/superpowers/` (1.6 MB of specs, plans and
   ledgers).
6. CI verified against a **fresh clone**, not this worktree.

**Out, and deliberately:**

- **No `LICENSE` file, and no licence text anywhere.** Bob's instruction of
  5 Sep, to be settled with Raj. The README says the licence is pending rather
  than staying silent, so a reader knows it is unresolved rather than assuming
  permission.
- **No full data-access instructions.** Decision B (DEL-14) is Raj's and
  unanswered. § 4 below is the floor that holds regardless of how he decides.
- No repackaging for PyPI, no versioned release, no DOI. DEL-47 ships.

## 3. What ships, and what does not

Three directories are development artefacts. The reflex is to delete them; for
a paper's companion repo that reflex is wrong.

| | verdict | why |
|---|---|---|
| `archive/master-2021/` | **ships** | The original 2020-21 notebooks that produced the published numbers. For a methods paper making claims about what the code did, the historical code IS evidence. It already has its own `ARCHIVE_README.md` saying it is a frozen snapshot. Deleting it would remove the only artefact showing where the method came from. |
| `docs/superpowers/` | **ships** | Every spec, plan and design decision behind the rebuild, including the oracle's derivation and the reasoning for each methodology switch. A reviewer asking "why does the code do X" is answered here. It is 1.6 MB of text in a repo that already carries 1.7 MB of notebooks. |
| `WORKPLAN.md` | **ships, with a header** | It is the project's own record of what is done and what is open — genuinely useful to a reader trying to judge maturity. But it reads as an internal to-do list, so it gets a one-paragraph header saying what it is and that Jira is the live source. |

**Nothing is deleted in this cycle.** If Bob or Raj wants any of it out before
release, removing a directory is a one-line change at that point; un-deleting
it after a public push is not. The default for a repo that has not shipped yet
is to keep evidence and label it.

What DOES get checked: that no file under any of them contains a credential, a
private URL, or anything personally identifying. That check is mechanical and
is part of task 4.

## 4. The data-access posture

**Bob's steer, 7 Sep 2026: the Delhi layers are not intended for general
release. What ships for others to run and check is the code plus the oracle.**
That is now the working posture of this document, and it makes the § 5
restructure the *point* of the release rather than a convenience: the oracle
is not a friendly on-ramp to the real thing, it IS the reproducible artefact.

So the README says, affirmatively:

- **Released, and runnable by anyone:** the code, and the two fixture cities
  with their hand-derived expected values. Enough to exercise and check every
  methodology rule end to end, on a fresh clone, with nothing else.
- **Not released:** the Delhi settlement, population, service and barrier
  layers. Third-party, and not ours to hand out.
- **Researchers who need Delhi specifically:** contact the authors.

**The consequence, which the README must state rather than let a reviewer
discover.** If the layers do not ship, then nobody outside the project can
independently recompute the paper's Delhi numbers from this repo. What they
CAN check is that the implementation is correct — that is exactly what the
oracle is for, and it is a real and unusual guarantee, but it is not the same
thing. Saying so plainly is better than implying more; a methods reviewer will
ask, and the answer is stronger when we volunteered it.

One line stays soft: **DEL-14 is formally Raj's/the group's**, and Bob's steer
is an intent rather than a ratified decision. The README states what is
released today, which needs no retraction if the decision later loosens (add
a section) or tightens (nothing changes). No text asserts that the question is
settled.

## 5. The README's shape

Ordered by what a stranger can actually do, soonest first:

1. **What this is** — one paragraph, the paper, the index, the status.
2. **Quickstart (no data needed)** — `uv sync`, then run the test suite, then
   run the oracle on the fixture city. Works on a fresh clone with nothing
   else. This is the section that fixes the defect in § 1.
3. **What the oracle proves** — DEL-46's front door. Two independent
   implementations, two hand-built cities, agreement at 1e-12, and what that
   does and does not guarantee.
4. **Running on Delhi data** — today's content, moved down, with the data
   situation stated before the commands rather than after.
5. **Changing the methodology** — pointer to `docs/methodology-config.md`,
   which already exists and is good.
6. **Repository layout** — kept, extended with `tests/`, `docs/data/`.
7. **Status and open decisions** — what is settled, what is not, licence
   pending. Honest rather than promotional.
8. **Citation** — the paper, marked forthcoming.

## 6. Definition of done

- A fresh `git clone` + `uv sync` + the quickstart's first command **succeeds
  with no external data**, verified in a clean directory outside this worktree.
- `uv run pytest -q -W error` green on that fresh clone.
- No `LICENSE` file; the README says the licence is pending.
- No secret, private URL or personal identifier anywhere in the shipped tree.
- Every README command copy-pasteable and actually executed once.
- DEL-45 closed; DEL-46's README half done, its remaining half (the
  reproducibility appendix) tracked separately.
