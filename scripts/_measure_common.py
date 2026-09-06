"""Shared plumbing for the `scripts/measure_*.py` measurement scripts.

Exactly five public names (spec § 1): the work-dir guard, the settlement
loader, the fenced-block renderer, its parser, and the fence itself. NOT a
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


def _blocks(text):
    """[(label or None, {key: value})] for every fenced block, in order."""
    out = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != FENCE:
            index += 1
            continue
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
        out.append((label, body))
    return out


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
