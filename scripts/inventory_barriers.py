"""Inventory the barrier layers the pipeline uses (DEL-51, spec § 2.3).

Raj asked where the barrier layers came from. What the repo can establish
MECHANICALLY is an inventory: how many features, what geometry, which CRS,
how long, what attribute schema, which ESRI/QGIS sidecars sit beside the
files, and how many settlements each layer flags under today's rule. The
provenance sentence itself is prose in docs/data/barriers.md, written from
this evidence — this script makes no claim about an agency.

READ-ONLY over --data-dir. Scratch (the settlement dedup cache) goes under
--work-dir, which is never inside the data directory.

    uv run python scripts/inventory_barriers.py --config code-2025 \
        --all-candidates --work-dir ~/measure_work/cache

Prints provenance lines, then two fenced blocks: `layers` (the drift-tested
counts, lengths, dates and flagged settlements) and `attributes` (each
layer's schema and the distinct values of its name-like columns).
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from delhi_psi import geometry, io, neighbors
from delhi_psi.config import load_config
from delhi_psi.pipeline import ID_COL
from scripts._measure_common import (emit, emit_check, load_settlements,
                                     render, resolve_work_dir)

DEFAULT_EPSG = 7760
# The columns that say WHOSE layer this is, per the 5 Sep 2026 survey.
NAME_COLUMNS = ("CAN_NM", "RL_ZONE", "Drain_Name", "DISTRICT", "AC_NAME")
VALUE_CAP = 20
METADATA_TAGS = {"CreaDate": "crea_date", "CreaTime": "crea_time",
                 "ModDate": "mod_date"}
# The other copies found under the data root on 5 Sep 2026 (--all-candidates).
CANDIDATES = {
    "canal_checked": "Barrier_Clip/Canal/new/checked_Canal.shp",
    "canal_root": "canal.data/canal.shp",
    "railway_root": "railway.data/railway.shp",
    "drain_root": "drain.data/drain.shp",
}


def metadata_dates(text):
    """CreaDate / CreaTime / ModDate from an ESRI `.shp.xml` sidecar."""
    root = ET.fromstring(text)
    out = {}
    for tag, key in METADATA_TAGS.items():
        node = next((found for found in root.iter(tag)
                     if (found.text or "").strip()), None)
        out[key] = "none" if node is None else node.text.strip()
    return out


def sidecar_facts(path):
    """The ESRI dates beside a shapefile, and whether QGIS wrote a `.qpj`.

    `path` is the `.shp` itself; ESRI names its sidecar `<file>.shp.xml`, so
    this appends rather than replacing the suffix. No path (an in-memory
    fixture layer) means every fact is unknown, which prints as "none".
    """
    unknown = {key: "none" for key in METADATA_TAGS.values()}
    if path is None:
        return {**unknown, "has_qpj": "none"}
    path = Path(path)
    xml = path.with_name(path.name + ".xml")
    dates = metadata_dates(xml.read_text()) if xml.exists() else unknown
    return {**dates,
            "has_qpj": "yes" if path.with_suffix(".qpj").exists() else "no"}


def _within(inner, outer):
    """Bounds containment, done on numbers: a barrier layer's bounding box
    can be degenerate (a straight canal has zero height), and shapely's
    `within` is false for a zero-area polygon."""
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def layer_facts(gdf, projected, *, settlements):
    """`gdf` as read (its own CRS); `projected` the same layer in the target
    CRS, where `.length` is metres and the bounding box is comparable with
    the settlement layer's."""
    return {
        "features": len(gdf),
        "geom_types": ";".join(sorted(set(gdf.geom_type.dropna()))),
        "crs": gdf.crs.to_string() if gdf.crs is not None else "none",
        "length_km": f"{projected.length.sum() / 1000:.6g}",
        "within_settlement_bbox": (
            "yes" if _within(projected.total_bounds, settlements.total_bounds)
            else "no"),
    }


def attributes(layers, *, columns=NAME_COLUMNS, cap=VALUE_CAP):
    """The `attributes` block: each layer's schema, and the distinct values
    of its name-like columns capped at `cap` — the point is to see whose
    layer this is, not to dump 616 drain names."""
    report = {}
    for name, gdf in layers.items():
        report[f"{name}_columns"] = ";".join(
            str(column) for column in gdf.columns if column != "geometry")
        for column in columns:
            if column not in gdf.columns:
                continue
            values = sorted({str(value) for value in gdf[column].dropna()})
            tail = f";... (+{len(values) - cap} more)" if len(values) > cap \
                else ""
            report[f"{name}_values_{column}"] = ";".join(values[:cap]) + tail
    return report


def barrier_flagged(settlements, layers, *, combine="any", configured=None,
                    id_col=ID_COL):
    """Production's own flag columns: one per layer, plus the combined
    `barrier` over the CONFIGURED layers only, so a --all-candidates copy
    cannot inflate the number production would produce. Every frame must
    already be in the same CRS."""
    configured = tuple(configured) if configured is not None else tuple(layers)
    frame = geometry.barrier_flags(settlements, layers, id_col=id_col)
    return neighbors.combine_barrier_flags(frame, layers=configured,
                                           combine=combine)


def inventory(layers, settlements, *, paths=None, epsg=DEFAULT_EPSG,
              id_col=ID_COL, combine="any", configured=None):
    """Both blocks: {"layers": {...}, "attributes": {...}}."""
    paths = paths or {}
    configured = tuple(configured) if configured is not None else tuple(layers)
    projected = {name: geometry.reproject(gdf, epsg)
                 for name, gdf in layers.items()}
    flagged = barrier_flagged(settlements, projected, combine=combine,
                              configured=configured, id_col=id_col)

    report = {"settlements": len(settlements)}
    for name, gdf in layers.items():
        facts = {**layer_facts(gdf, projected[name], settlements=settlements),
                 **sidecar_facts(paths.get(name))}
        report.update({f"{name}_{key}": value
                       for key, value in facts.items()})
        report[f"{name}_flagged_settlements"] = int(flagged[name].sum())
    report["flagged_any"] = int(flagged["barrier"].sum())
    return {"layers": report, "attributes": attributes(layers)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code-2025",
                        help="profile that names the layers (default code-2025)")
    parser.add_argument("--data-dir", default=None,
                        help="data root, opened READ-ONLY")
    parser.add_argument("--work-dir", default=None,
                        help="scratch (the settlement dedup cache); default a "
                             "fresh temporary directory. Never under --data-dir.")
    parser.add_argument("--all-candidates", action="store_true",
                        help="also inventory the other copies of these layers "
                             "found under the data root on 5 Sep 2026")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--out", default=None,
                        help="write the blocks here instead of stdout — "
                             "BLOCKS ONLY; REFUSES (exit 1) to overwrite a "
                             "target that already holds hand-written prose, "
                             "since that would delete every caption and "
                             "Finding — use --splice to refresh such a "
                             "document in place instead")
    target.add_argument("--splice", default=None,
                        help="refresh the blocks INSIDE this committed "
                             "document in place, preserving every caption "
                             "and Finding (DEL-59)")
    args = parser.parse_args(argv)

    emit_check(out=args.out, splice=args.splice)

    cfg = load_config(args.config, data_dir=args.data_dir)
    work_dir = resolve_work_dir(args.work_dir, data_dir=cfg.paths.data_dir,
                                prefix="delhi_psi_barriers_")

    paths = {name: cfg.paths.data_dir / path
             for name, path in cfg.layers.barriers.items()}
    if args.all_candidates:
        for name, relative in CANDIDATES.items():
            candidate = cfg.paths.data_dir / relative
            if candidate.exists():
                paths[name] = candidate
            else:
                print(f"candidate missing: {candidate}", file=sys.stderr)
    layers = {name: io.read_layer(path) for name, path in paths.items()}
    settlements = load_settlements(cfg, work_dir)

    for name, path in paths.items():
        print(f"layer {name}: {path}", file=sys.stderr)
    print(f"work-dir: {work_dir}", file=sys.stderr)
    blocks = inventory(layers, settlements, paths=paths, epsg=cfg.crs.epsg,
                       id_col=cfg.layers.settlements.id_col,
                       combine=cfg.methodology.barrier.combine,
                       configured=tuple(cfg.layers.barriers))
    text = "\n".join(render(report, name=name)
                     for name, report in blocks.items())
    emit(text, out=args.out, splice=args.splice)
    return 0


if __name__ == "__main__":
    sys.exit(main())
