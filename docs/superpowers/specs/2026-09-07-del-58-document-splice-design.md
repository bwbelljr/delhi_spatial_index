# Regenerable `docs/data/` documents — DEL-58

**Ticket:** DEL-58. **Branch:** `del-58-splice` off `main` at `1b85dc8`.
**Date:** 7 Sep 2026. Cycle 5, ticket 1 of 3.

---

## 1. The failure, and why the existing mitigation is not a fix

`scripts/summarize_sweep.py --out docs/data/phase6_sweep.md` writes **blocks
only**. Pointed at the committed document it deletes every hand-written
caption and every `### Finding` section — 154 insertions against 292
deletions when it happened during DEL-55.

Two things make this worse than an ordinary footgun:

1. **The drift test cannot see it.** `test_a_fresh_run_reproduces_the_committed_blocks`
   compares *parsed blocks*. A document stripped of all its prose has
   **identical blocks**, so that test passes on the wreckage.

   **Corrected after the whole-branch review, which ran the accident instead
   of reasoning about it.** This spec first said "the suite stays green while
   the analysis is gone". That is false, and the reviewer disproved it by
   overwriting a scratch copy of `docs/data/phase6_sweep.md` with a real
   blocks-only dump and running the suite: **two pre-existing tests fail**
   (`test_every_caption_says_the_run_is_provisional` and
   `test_the_document_records_its_provenance`). Destroying all seven
   documents fails a per-document provenance test for **six of the seven**.

   The narrow claim holds — that one drift test *is* blind, verified — but
   the generalisation to "the suite" was wrong, and it was wrong in the
   direction that overstates this ticket's necessity. What the new guard is
   actually worth, stated honestly:

   - it is the **only** test in the repo that references
     `docs/data/uso_final_vocabulary.md` — the one document with no coverage
     at all before this branch;
   - it is glob-parametrised, so a *future* document inherits it without
     anyone remembering to write a per-document provenance test;
   - its second clause catches the loss of a single section's caption, which
     nothing else covers.

   That is a smaller claim than the one this spec opened with, and it is the
   true one.
2. **The command that destroys it is the command the harness exists for.**
   Whoever re-runs Phase 6 against the ratified profile (DEL-31) is the next
   person to type it, and by then the prose describes the paper's actual
   results rather than a dry run's.

DEL-46 added a warning to the `--out` help text and a line in the document's
header. Both require reading before typing. That is mitigation, not a fix.

## 2. Three parts, because prevention and detection are different jobs

The ticket offered three options and observed that (1) plus (3) is the
honest combination. All three ship, because each covers a case the others do
not:

| part | what it does | the case it covers |
|---|---|---|
| **`--splice`** | replaces the block runs in place, leaves everything else byte-identical | the legitimate need: refresh the numbers in a committed document |
| **`--out` refusal** | errors when the target holds prose, naming `--splice` | the accident: muscle memory types `--out <committed doc>` |
| **prose-aware drift test** | asserts every committed document still has its structure, by glob | the document nobody wrote a per-document guard for — today `uso_final_vocabulary.md`, tomorrow whichever document is added next |

`--splice` alone would leave `--out` as a loaded gun. The refusal alone
turns silent loss into a message but gives no way to do the thing the user
wanted. The test alone catches the loss after the fact — but it is cheap and
it is the only part that scales to documents nobody has written yet.

## 3. Where it lives: one implementation, two callers

`scripts/_measure_common.py` already owns the whole fenced-block contract —
`FENCE`, `render`, `_blocks`, `parse_block`. The splice is the missing
inverse of `render` at document scale, so it belongs there, and both
`summarize_sweep.py` and `rank_report.py` import it.

**This is the load-bearing decision.** Two copies of a splice rule is the
same drift DEL-35 refused for the decile rule, and the two scripts render
identically today (`render(row, name=...)` joined by newlines) precisely
because they share that module.

## 4. The splice rule, and the evidence it fits

**Blocks are replaced in runs, matched by label.**

A *run* is a maximal sequence of contiguous blocks sharing a label. The
fresh output's runs replace the document's same-labelled runs; every byte
outside those spans — headings, captions, `### Finding` sections, blank
lines — is preserved exactly.

Runs, not individual blocks, because **a run is not one block**:
`rank_report.render_categories_block` emits one block per category, all
labelled `categories`, and `summarize_sweep`'s `points` run is 13 blocks in
the committed document. A one-block-to-one-block rule cannot express "these
13 blocks became 11".

**Verified against every committed document** rather than assumed:

| document | blocks | label runs |
|---|---|---|
| `phase6_sweep.md` | 54 | `points`, `ordering`, `gap`, `denominator_check` |
| `barriers.md` | 2 | `layers`, `attributes` |
| `roads_access.md` | 2 | `access`, `one_factor` |
| `rule_effects.md` | 2 | `partial_barriers`, `overlap_lending` |
| `layer_pathologies.md` | 1 | *(unlabelled)* |
| `psi_columns.md` | 1 | *(unlabelled)* |
| `uso_final_vocabulary.md` | 0 | — hand-written, no blocks |

**No label appears as two separate runs in any document.** The rule is not a
guess about how these documents are shaped; it is a description of how all
seven actually are.

**Refuse rather than guess.** Every one of these is an error that writes
nothing:

- a label in the fresh output that the document does not have — naming it,
  because the likely cause is splicing into the wrong file;
- a label appearing as two non-contiguous runs in the document — ambiguous,
  and no committed document does this today;
- a document with no blocks at all when the fresh output has some.

**Labels the fresh output does not mention are left alone**, which is what
makes `--block points --splice <doc>` work: refresh one run, keep the rest.
An unlabelled block is its own run, keyed on the absence of a label, so the
single-block documents splice by the same rule as the rest.

## 5. The `--out` refusal

`--out` keeps working for its real use — dumping blocks to a scratch file —
and refuses when the target **exists and contains prose**: any non-blank
line outside a fenced block.

That test rather than the ticket's proposed "contains a `### Finding`"
because only two of the seven documents have Findings at all (`phase6_sweep.md`
and `barriers.md`). A Finding-based guard would leave the other five exactly
as exposed as they are today, and the caption above a block is worth no less
than a Finding below it.

The message names `--splice` and the file, so the recovery is in the error
rather than in this spec.

Overwriting a blocks-only file — a previous `--out` dump — stays allowed:
there is no prose to lose. No `--force` flag; `--splice` covers the
legitimate case and a force flag is a second way to make the original
mistake.

## 6. The prose-aware drift test

Over **every** `docs/data/*.md`, not a hard-coded list — a new document must
inherit the guard without anyone remembering to add it:

1. the document has at least one `## ` heading;
2. every `## ` section that contains a fenced block also contains at least
   one non-blank line outside a block.

(1) is what actually fails on the DEL-55 accident: a blocks-only dump has no
headings whatsoever. (2) catches the narrower loss of one section's caption.

**Verified true for all seven documents today** — 0 sections carrying blocks
without prose across all of them — so this pins the current state rather
than describing an aspiration.

`uso_final_vocabulary.md` has no blocks and is hand-written throughout; it
satisfies both clauses, and the test must not require a document to be
script-generated.

## 7. Out of scope

- **No real-data run.** Nothing here reads Delhi layers; the splice operates
  on committed text and script output.
- **No change to any statistic, block, or rendered value.** Every
  `docs/data/*.md` must be byte-identical after this ticket. If one moves,
  the splice is rewriting something it should be preserving.
- **No new dependency.**
- **No licence file** (DEL-56, pending Raj).

## 8. Definition of done

- `splice_blocks(document_text, fresh_text) -> str` in
  `scripts/_measure_common.py`, with the run rule and all three refusals.
- `--splice <path>` on both `summarize_sweep.py` and `rank_report.py`,
  sharing that one implementation.
- `--out` refuses a target holding prose, naming `--splice`.
- The prose-aware drift test over all of `docs/data/`.
- A round-trip test: splicing a document's **own** blocks back into it is a
  byte-identical no-op — the strongest available statement that the splice
  preserves what it is not replacing.
- Every committed `docs/data/*.md` byte-identical; full suite green.
