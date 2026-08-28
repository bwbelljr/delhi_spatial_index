---
description: Autonomous build-and-ship pipeline — plan → review → implement → review → test → run → merge, with multi-model subagents and ultracode review loops.
argument-hint: <spec path> [smoke: <cmd>] [run: <cmd>]
---

# Build-and-ship pipeline

Standing instructions for taking an **approved spec/design** all the way to a
**merged** change. Run the phases in order. Do not start until a spec exists
and the owner has approved it.

**Where this sits in this repo's process.** Every WORKPLAN phase follows the
same arc:

1. `superpowers:brainstorming` — questions, approaches, design; spec written
   to `docs/superpowers/specs/` and **approved by the owner** (interactive;
   never part of /ship)
2. `/ship <spec>` — everything below: plan → plan review → TDD
   implementation → code review → smoke → run → PR → merge
3. `superpowers:finishing-a-development-branch` semantics apply at the end:
   the branch merges only when its phase's definition-of-done from
   WORKPLAN.md is demonstrably met

/ship is the execution half of the discipline; the thinking half (spec +
approval) always precedes it.

**Inputs**

- **Spec/design doc** (required) — the approved design to implement
  (specs live in `docs/superpowers/specs/`).
- **Smoke test** (optional) — a fast command that exercises the change
  cheaply. Only run if specified.
- **Actual run** (optional) — the real end-to-end command. Only run if
  specified.

**Repo-specific guidance for this project (delhi_spatial_index)**

- Typical `smoke:` — `uv run pytest -q -W error` (the mythical-city oracle
  suite is the canonical fast gate: it proves the index math).
- Typical `run:` — the full pipeline + baseline verification via the
  `delhi-psi` CLI: `delhi-psi preprocess --config code-2025 --data-dir
  ~/delhi_data --out-dir <dir>` → `delhi-psi compute --config code-2025
  --data-dir ~/delhi_data --out-dir <dir>` → `uv run python
  scripts/verify_against_baseline.py --config code-2025 --data-dir
  ~/delhi_data --verify-dir <dir>`. **Full runs are long** (the neighbors
  computation iterates over 4,352 colonies) — budget for it, run it in the
  background, and never schedule more full runs than the phase actually
  needs.
- The July 2025 outputs in the data directory are a **read-only baseline**:
  no pipeline invocation may write into the baseline files; verification
  runs write to a separate output directory.
- Conventions: uv + `pyproject.toml`, pytest. (A fuller conventions doc /
  CLAUDE.md is planned with the Phase 3 refactor — until then, match the
  style of the file you are editing.)
- **Changelog**: every /ship run updates `CHANGELOG.md` (`[Unreleased]`
  section, Keep a Changelog style) as part of the PR; autonomous-run
  deviations from plan are recorded there and in the PR description.

**Model roles** (assign per phase; parallelize independent work)
| Role | Models |
| --- | --- |
| Planning | Opus (lead), Sonnet |
| Review loops (find + verify) | Sonnet, Haiku |
| Implementation (coding subagents) | Opus, Sonnet, Haiku |

**What "ultracode review loop" means here:** a **Workflow-orchestrated
multi-agent review** — fan out finders across dimensions, adversarially
verify each finding (independent skeptics), then apply fixes — repeated until
a pass yields no surviving blocking findings. This is the local Workflow tool,
**not** the cloud `/code-review ultra` (which is user-triggered/billed and
cannot be launched programmatically).

## Phases

1. **Plan.** Write the implementation plan from the approved spec
   (superpowers `writing-plans`), using **Opus/Sonnet**. Output a concrete,
   step-ordered plan with tasks, files, and tests.

2. **Review the plan (ultracode loop, Sonnet/Haiku).** Run the review loop on
   the _plan_. Fix every confirmed Critical/Important finding. **Stop per the
   stopping rule below — NOT "until clean."** Then proceed.

3. **Implement.** TDD is **mandatory and non-negotiable**, and is owned by
   `superpowers:test-driven-development` (its Iron Law and "Verify RED —
   watch it fail" step) plus `writing-plans`' step structure — **invoke
   them; this file does not restate them.** The only enforcement clause
   that belongs here: code produced before its test is not "done with a
   test gap", it is **unverified — discard and redo it**, never retrofit a
   test onto it.

   Write the code with **subagents on Opus, Sonnet, and
   Haiku**, parallelizing independent tasks (isolate with worktrees if agents
   mutate files concurrently). Follow the repo conventions above.

4. **Review the code (ultracode loop, Sonnet/Haiku).** Run the review loop on
   the _code_. **Fix anything that blocks running the script**, plus confirmed
   Critical/Important findings. **Stop per the stopping rule below.**

5. **Smoke test (if specified).** Run the smoke command. Fixing code during
   the smoke run is expected — do it.

6. **Actual run (if specified).** Run the real command. Fixing code during the
   run is expected — do it.

7. **Ship.** When everything above is green, create the pull request and
   merge it (respect branch protection / required checks).

## Review-loop stopping rule

**"Repeat until clean" is wrong and is not the rule.** Reviewing to zero
findings does not converge: every round surfaces fresh prose nits, and each
fix can introduce new defects. Use severity, not exhaustion.

**Severity taxonomy** — the review prompt must require one per confirmed finding:

| Severity      | Meaning                                                                                                  | Gates the loop? |
| ------------- | -------------------------------------------------------------------------------------------------------- | --------------- |
| **Critical**  | Would fail, void, or **silently corrupt** the run or its result                                          | **Yes**         |
| **Important** | Real wrong behavior, or a gap that makes the result unmeasurable / uninterpretable                       | **Yes**         |
| **Minor**     | Style, lint, formatting, naming, docstring wording, test naming, expected-count drift, anything cosmetic | **No**          |

**STOP as soon as a round returns zero Critical and zero Important.** Minor
findings never gate: batch them into ONE fix pass, apply without re-review,
and move on. A round that is all-Minor is the signal to stop, not to re-run.

**Hard caps.** Plan review: **max 3 rounds**. Code review: **max 3 rounds**. On
hitting a cap, proceed and report what is outstanding — do not keep going.

**Never spend a review round on what a tool answers.** Lint, formatting,
import order, type annotations, naming rules, and line length are decided by
the linter / `pytest` / the type checker in _seconds_ on the real artifact.
Do not fan out agents to read prose for them. If a round's findings are
dominated by tool-checkable issues, that is proof the loop has hit
diminishing returns — stop and build.

**Churn detector — the important one.** If a round's Critical/Important was
_introduced by the previous round's fix_, the loop is churning, not
converging. Stop reviewing and move to code: real tests are a stronger gate
than more prose review.

**Review budget.** Review must not cost more than building. If review
wall-clock exceeds the estimated implementation time, stop and implement. A
plan review that outlasts writing the code has inverted its own purpose.

**Prose vs. code asymmetry.** Plan review can only reason about text; code
review gets to _run things_. Bias toward shipping the plan into code sooner —
defects the plan review would eventually find are usually found faster, and
more reliably, by executing the test suite.

_Evidence (imported from a prior project's run that motivated this rule):_
plan review ran **5 rounds / 2h18m**; implementation of all 6 tasks took
**29m** (4.7× inversion). Rounds 1–2 earned their keep — 3 Criticals,
including a scoring step that silently contradicted the training objective.
But **R3 and R5 found zero Criticals**, and **R4's lone Critical was
introduced by R2's own fix**. R3–R5's remaining findings were almost all
lint-tool territory. Stopping at R3, the first zero-Critical round, would
have saved ~1.5h of a time-boxed overnight window.

## Guardrails

- **Evidence before "done":** never mark a phase complete without the command
  output that proves it (tests pass, smoke/run succeeded). Show the output.
- **Verify findings before fixing:** the review loop's verify step exists to
  kill plausible-but-wrong findings — don't implement unverified suggestions.
- **Fix-forward during test/run is allowed** (phases 5–6) — that's the point
  of running.
- **Commit + push** at meaningful checkpoints; keep the branch current.
- **Stop and ask** only on a genuine fork the owner must decide (ambiguous
  scope, destructive/irreversible action, external side effects). In this
  repo that explicitly includes: anything that would alter the index
  methodology (oracle expected values, Eq. 1–4 semantics, exclusion rules)
  and anything writing to the baseline data. Otherwise proceed autonomously
  through the pipeline.
- **Merge is the terminal step** — PR created and merged, branch integrated.

### Unattended runs (overnight, cron, any window with no owner present)

The superpowers skills this pipeline calls are written for interactive use,
and several route decisions to a human mid-flight —
`subagent-driven-development` sends any review finding that conflicts with
the plan to the owner ("present the finding and the plan text, ask which
governs"), and `writing-plans` ends in an interactive execution-handoff.
Those steps **silently stall or get improvised** when nobody is awake. The
skills come from the installed superpowers plugin — do not edit them;
override here.

**Before an unattended window starts, get these decided and write them into
the spec's decision log** — an undocumented rule invented at 2am is the
thing this section exists to prevent:

1. **Plan-vs-reviewer conflicts.** Default: a **CONFIRMED Critical** finding
   governs over the plan — deviate to fix it and record the deviation in the
   ledger and the morning report. Every non-Critical conflict follows the
   plan and is surfaced for adjudication rather than acted on. (Confirm with
   the owner; this default is only a default.)
2. **Autonomy scope.** Explicitly: may the run fix forward on failure, commit,
   and push? Each is a separate yes/no.
3. **Failure policy.** Which stages are load-bearing (a failure halts
   dependents) versus best-effort (log, skip, continue).

**Also required for unattended runs:** a durable progress ledger written per
completed task (`.superpowers/sdd/progress.md`), since conversation memory
does not survive compaction and a controller that loses its place can
re-dispatch completed work. And a morning report that leads with what
FAILED and what deviated, not with what succeeded.

## Notes

- Scale the review-loop breadth to the change size (a few finders for a small
  change; larger fan-out + multi-vote verify for a big one).
- If the spec names a `smoke:` and/or `run:` command in the invocation, use
  those; otherwise infer from the spec's acceptance criteria and confirm the
  command before executing anything expensive.
