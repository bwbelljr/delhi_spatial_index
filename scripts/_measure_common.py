"""Shared plumbing for the `scripts/measure_*.py` measurement scripts.

Nine public names now (spec § 1 started this at five; DEL-58 added four):
the work-dir guard (`resolve_work_dir`), the settlement loader
(`load_settlements`), the fenced-block renderer (`render`), its parser
(`parse_block`), and the fence itself (`FENCE`) — plus `splice_blocks` (the
document splicer), `holds_prose` (the `--out` guard's prose detector),
`emit` (the `--out`/`--splice` CLI behaviour), and `emit_check` (`emit`'s
cheap pre-flight, callable before any computed input exists). NOT a
`delhi_psi` module — these are measurement utilities, not pipeline API, and
they are imported by path-sibling scripts the way `tests/cities.py` is
imported by tests.
"""

import tempfile
from pathlib import Path

from delhi_psi import geometry, io, pipeline

FENCE = "```text"


def resolve_work_dir(cli_value=None, *, data_dir=None,
                     prefix="delhi_psi_measure_", create=True):
    """Where a script's scratch output goes — NEVER the data directory.

    ~/delhi_data is bisynced to the shared drive, so a stray file there
    propagates to everyone. With `data_dir` given the guard fires; with
    `data_dir` None this only RESOLVES a path (no guard, no mkdir), which is
    the shape the default test uses.

    `create` (default True) gates ONLY the mkdir at the end — the guard
    itself always fires when `data_dir` is given, `create` or not. Pass
    `create=False` to get the identical refusal without creating the
    directory (a `--dry-run` caller's shape): every existing caller passes
    `data_dir` and wants the directory made, so the default keeps their
    behaviour unchanged.
    """
    work_dir = (Path(cli_value).expanduser() if cli_value
                else Path(tempfile.mkdtemp(prefix=prefix)))
    if data_dir is None:
        return work_dir
    data_dir = Path(data_dir).expanduser().resolve()
    resolved = work_dir.resolve()
    if resolved == data_dir or data_dir in resolved.parents:
        raise SystemExit(
            f"work directory {work_dir} is inside the data directory "
            f"{data_dir}, which these scripts never write to (it is bisynced "
            "to the shared drive)")
    if create:
        work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def load_settlements(cfg, cache_dir):
    """Read, deduplicate and reproject exactly as `pipeline.preprocess` does,
    so every count below describes the universe the pipeline actually scores.

    WARM CACHE UPCASTS Polygon -> MultiPolygon. `pipeline._dedup_cached`
    returns the in-memory frame on a COLD cache but re-reads its own
    GeoPackage on a WARM one, and a GeoPackage layer carries a single
    geometry type: the raw layer's 3,801 Polygon + 556 MultiPolygon all come
    back as MultiPolygon after that round trip. So a caller that inspects
    `geom_type` — today only `count_multipolygons` in
    measure_layer_pathologies.py — MUST pass a cold `cache_dir` (a fresh
    directory), or its answer is 4,357 instead of 556. Every other predicate
    these scripts use (`intersects`, intersection length, `touch` adjacency,
    barrier flags) is type-agnostic and may share a warm cache.
    """
    source = cfg.paths.data_dir / cfg.layers.settlements.path
    gdf = io.read_layer(source)
    gdf = pipeline._dedup_cached(gdf, cache_dir, "settlements", source)
    # `remove_duplicate_geom` reset_index()es, which leaves an `index`
    # column; preprocess drops exactly these two, and bbox_frame's
    # pd.concat needs the same shape.
    gdf = gdf.drop(columns={"index", "level_0"}.intersection(gdf.columns))
    gdf = geometry.reproject(gdf, cfg.crs.epsg)
    gdf["area_km2"] = gdf.area / 1_000_000
    return gdf


def render(report, *, name=None):
    """The fenced block a `docs/data/*.md` carries verbatim.

    `name` labels the block for a multi-block script; without it the output
    is byte-identical to what the pathology script has always printed.
    """
    lines = [FENCE]
    if name is not None:
        lines.append(f"block: {name}")
    lines.extend(f"{key}: {value}" for key, value in report.items())
    lines.append("```")
    return "\n".join(lines)


def _block_spans(text):
    """[(label, start_line, end_line_exclusive)] for every fenced block.

    THE one scanner: `_blocks` reads bodies out of these spans, and the
    splice replaces the spans themselves. Two scanners that disagree about
    where a block ends is the drift this module exists to prevent.
    """
    lines = text.splitlines()
    out, index = [], 0
    while index < len(lines):
        if lines[index].strip() != FENCE:
            index += 1
            continue
        start = index
        index += 1
        label, body, closed = None, {}, False
        while index < len(lines):
            line = lines[index]
            index += 1
            if line.strip() == "```":
                closed = True
                break
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key == "block" and label is None and not body:
                label = value
            else:
                body[key] = value
        if not closed:
            raise ValueError(f"unterminated {FENCE} block")
        out.append((label, body, start, index))
    return out


def _blocks(text):
    """[(label or None, {key: value})] for every fenced block, in order."""
    return [(label, body) for label, body, _, _ in _block_spans(text)]


def _label_runs(text):
    """[(label, start, end, separator)] — maximal sequences of contiguous
    same-label blocks, plus the lines the document puts between them.

    A run BREAKS on intervening prose, so a label separated by a paragraph
    is two runs, not one. That is what makes the ambiguity refusal in
    `splice_blocks` meaningful: a label with two runs has no single place
    for the fresh blocks to go, and no committed document has one.

    `separator` is the lines between the run's first two blocks — one blank
    line in `docs/data/phase6_sweep.md`, nothing in the other documents.
    Preserving it is why a splice does not silently reformat a document.
    """
    lines = text.splitlines()
    runs = []
    for label, _, start, end in _block_spans(text):
        if runs:
            prev_label, prev_start, prev_end, separator = runs[-1]
            gap = lines[prev_end:start]
            if prev_label == label and not any(line.strip() for line in gap):
                runs[-1] = (label, prev_start, end,
                            gap if separator is None else separator)
                continue
        runs.append((label, start, end, None))
    return [(label, start, end, separator or [])
            for label, start, end, separator in runs]


def splice_blocks(document, fresh):
    """`document` with each of `fresh`'s block runs substituted in place.

    Everything outside those runs — headings, captions, `### Finding`
    sections, blank lines — is preserved byte for byte. Labels `fresh` does
    not mention are left alone, which is what makes `--block points` refresh
    one run and keep the rest.

    Refuses rather than guesses (spec § 4). Every refusal writes nothing.
    """
    doc_lines = document.splitlines(keepends=True)
    fresh_lines = fresh.splitlines()
    doc_runs = _label_runs(document)
    fresh_runs = _label_runs(fresh)

    if not doc_runs:
        raise ValueError(
            f"the document has no {FENCE} block to splice into; "
            "--out writes a fresh blocks-only file")

    def _one_run_per_label(runs, what):
        seen = {}
        for label, *_ in runs:
            if label in seen:
                raise ValueError(
                    f"{what}: block label {label!r} appears as two separate "
                    "runs, so there is no single place to splice it")
            seen[label] = True
        return seen

    doc_labels = _one_run_per_label(doc_runs, "document")
    _one_run_per_label(fresh_runs, "fresh output")

    for label, *_ in fresh_runs:
        if label not in doc_labels:
            raise ValueError(
                f"block label {label!r} is not in the document (it has "
                f"{sorted(str(name) for name in doc_labels)}); splicing into "
                "the wrong file?")

    replacement = {}
    for label, start, end, _ in fresh_runs:
        replacement[label] = fresh_lines[start:end]

    out, cursor = [], 0
    for label, start, end, separator in doc_runs:
        out.extend(doc_lines[cursor:start])
        cursor = end
        if label not in replacement:
            out.extend(doc_lines[start:end])
            continue
        text = "\n".join(_joined(replacement[label], separator))
        # Only re-add the newline the replaced span actually ended with. An
        # unconditional "\n" fabricates a trailing newline for a document
        # that ends at its last block without one (plan review, finding 2).
        out.append(text + ("\n" if doc_lines[end - 1].endswith("\n") else ""))
    out.extend(doc_lines[cursor:])
    return "".join(out)


def _joined(fresh_run_lines, separator):
    """The fresh run's lines re-joined with the document's own separator.

    The blank-line skip is load-bearing, and the plan review caught its
    absence by executing the code: when `fresh` IS a document (the
    round-trip case, `splice_blocks(doc, doc)`), the fresh run's lines
    already carry that document's own blank separators. Without the skip
    those blanks are absorbed into the NEXT block's accumulator and the
    document's separator is inserted on top of them — two blank lines where
    there was one, and the round-trip test this ticket rests on fails.
    """
    if not separator:
        return list(fresh_run_lines)
    out, block = [], []
    for line in fresh_run_lines:
        if not block and not line.strip():
            continue
        block.append(line)
        if line.strip() == "```":
            if out:
                out.extend(separator)
            out.extend(block)
            block = []
    out.extend(block)
    return out


def holds_prose(text):
    """True when `text` has any non-blank line outside a fenced block.

    The `--out` guard (spec § 5). Not "contains a `### Finding`": only two
    of the seven documents have Findings, and a caption above a block is
    worth no less than a Finding below it.
    """
    lines = text.splitlines()
    covered = set()
    for _, _, start, end in _block_spans(text):
        covered |= set(range(start, end))
    return any(line.strip() for index, line in enumerate(lines)
               if index not in covered)


def emit_check(*, out=None, splice=None):
    """The cheap half of `emit`'s `--out`/`--splice` guards (fix round item
    6): the part that needs no computed `text` at all, so a `main()` can
    call this FIRST — before any loading or rendering — and refuse a bad
    flag combination without paying for the work first. `emit` calls this
    itself too, so its own checks (needed by callers, mainly tests, that
    invoke `emit` directly without going through a `main()`) stay in force.

    Refuses (`SystemExit`) in exactly the two cases `emit` refuses in:
    `--out` pointed at a target that already holds hand-written prose, and
    `--splice` pointed at a document that does not exist.
    """
    if splice is not None:
        target = Path(splice)
        if not target.exists():
            raise SystemExit(
                f"{target} does not exist; --splice refreshes an existing "
                "document's blocks in place — use --out to write a new file "
                "instead")
    if out is not None:
        target = Path(out)
        if target.exists() and holds_prose(target.read_text()):
            raise SystemExit(
                f"{target} holds hand-written prose, and --out writes blocks "
                "only — it would delete every caption and Finding. Use "
                f"--splice {target} to refresh its blocks in place.")


def emit(text, *, out=None, splice=None):
    """The `--out` / `--splice` behaviour shared by two of the CLIs in
    `scripts/` — `summarize_sweep.py` and `rank_report.py` — not "every
    measurement CLI": the four `measure_*.py` scripts print to stdout only
    and offer neither flag.

    ONE implementation, because the failure this guards against — blocks
    written over a document's prose — happened once already, and a second
    copy of the guard is a second chance to get it wrong.

    `emit_check` runs first (raising `SystemExit` for either refusal before
    any of this function's own work happens); `splice_blocks`'s `ValueError`
    is then caught and re-raised as `SystemExit(str(exc))` so a bad
    `--splice` target (wrong document, ambiguous label, ...) reports a clear
    one-line message rather than a raw traceback (fix round item 5).
    """
    emit_check(out=out, splice=splice)
    if splice:
        target = Path(splice)
        try:
            spliced = splice_blocks(target.read_text(), text)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        target.write_text(spliced)
        return
    if out:
        target = Path(out)
        target.write_text(text + "\n")
        return
    print(text)


def parse_block(text, *, name=None):
    """The inverse of `render`. The SAME parser reads the committed document
    and a script's stdout, so the drift test compares like with like.

    `name` selects a labelled block; without it the FIRST block is returned,
    which is the single-block scripts' shape.
    """
    blocks = _blocks(text)
    if not blocks:
        raise ValueError(f"no {FENCE} block found")
    if name is None:
        return blocks[0][1]
    for label, body in blocks:
        if label == name:
            return body
    raise ValueError(f"no {FENCE} block labelled {name!r}; found "
                     f"{[label for label, _ in blocks]}")
