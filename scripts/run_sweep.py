"""Run the Phase 6 methodology sweep and record what each point cost.

A "sweep point" is one profile from `delhi_psi/profiles/` — `code-2025` with
exactly one methodology factor moved (DEL-37 / DEL-55) — carried through the
CLI's `preprocess` and `compute` stages and summarised into one manifest JSON
under `<work-dir>/manifest/<profile>.json`. Eleven points exist; `GROUPS`
orders them cheapest-first and lets a run cover one group at a time.

The work directory may NEVER be the data directory (or a child of it):
`~/delhi_data` is bisynced hourly to a shared drive, so a stray sweep file
there propagates to everyone and `~/delhi_data/phase3_verify` holds this
repo's real-data correctness proof, which a careless overwrite would destroy.
`scripts._measure_common.resolve_work_dir` enforces this for both a real run
and `--dry-run` — the latter passes its `create=False` keyword, so the
identical refusal fires without creating anything.

THIS IS A DRY RUN. Every profile here is frozen `code-2025` plus one factor;
Raj's 28 Aug decisions supersede several of these methodology choices, so the
numbers this script produces describe the July 2025 rule set, not a
recommendation. They exist to be COSTED (wall-clock seconds, link counts,
degree distributions), not adopted.

Every expensive step is a subprocess (`run_stage` shells out to
`python -m delhi_psi.cli`), never an in-process pipeline call: a segfault or
an OOM in one point's preprocess must not take the other ten down with it,
and the CLI is the interface the docs already tell a human to use.
"""

import argparse
import datetime as dt
import json
import logging
import re
import subprocess
import sys
import time
from collections import namedtuple
from pathlib import Path

from delhi_psi import io, pipeline, validate
from delhi_psi.config import load_config
from scripts._measure_common import resolve_work_dir  # re-exported below

log = logging.getLogger(__name__)

Point = namedtuple("Point", "profile artifact stages reason")

# Cheapest-first order (spec § 7). `decay-none` leads `all` on purpose: its
# `bbox` preprocess warms the settlement dedup cache (about a quarter of
# every later preprocess) and establishes the per-link compute rate at the
# baseline's link count.
DECAY = ("decay-none", "decay-power05", "decay-power2", "decay-exp2km",
        "decay-exp5km", "decay-boundary")
ADJACENCY = ("adj-touch", "band-0km")
BANDS = ("band-1km", "band-5km", "band-10km")
GROUPS = {"decay": DECAY, "adjacency": ADJACENCY, "bands": BANDS,
         "all": DECAY + ADJACENCY + BANDS}

# The manifest's documented key set (spec item 8), in the order it is
# written. A schema test pins this literal set for both an OK and a FAILED
# manifest, so a key that quietly stops being written fails a test instead
# of showing up as a blank column months later.
MANIFEST_KEYS_OK = (
    "profile", "status", "stages_run", "skip_reason", "stamp",
    "preprocess_s", "dedup_cache_warm", "compute_s", "n_links", "deg_mean",
    "deg_p50", "deg_max", "n_isolates", "degree_from", "n_settlements",
    "n_barrier_flagged", "n_reported", "n_missing_population",
    "outputs", "commit", "run_date")
MANIFEST_KEYS_FAILED = MANIFEST_KEYS_OK + (
    "failed_stage", "returncode", "stderr_tail")
_DEGREE_KEYS = ("n_links", "deg_mean", "deg_p50", "deg_max", "n_isolates")

_FORMAT_EXT = {"csv": ".csv", "shp": ".shp", "joblib": ".joblib"}


class StageFailed(Exception):
    """`run_stage` raises this on a non-zero exit from the CLI subprocess."""

    def __init__(self, stage, returncode, stderr_tail):
        super().__init__(f"{stage} failed (exit {returncode}): {stderr_tail}")
        self.stage = stage
        self.returncode = returncode
        self.stderr_tail = stderr_tail


# --- planning ------------------------------------------------------------
_STAMP_KEY_RE = re.compile(r"methodology\.(\w+\.\w+)=")


def artifact_matches(path_or_frame, cfg):
    """True iff a neighbours artifact was built under `cfg`'s adjacency and
    barrier settings.

    Accepts either a loaded frame (as a caller that already has one in hand
    can pass directly) or a path (loaded here with `io.read_neighbors`) —
    the shape `plan_point` uses, so a mismatch is never discovered only
    after paying to load the file twice. A path that does not exist, OR
    cannot be loaded at all, is simply "does not match": `io.write_neighbors`
    is not atomic (a plain `joblib.dump` straight to the final path), so a
    preprocess killed mid-write (an OOM or a segfault — the case
    `run_stage`'s subprocess isolation exists for) leaves a truncated file
    that raises an unpickling/EOF error, not a `ValidationError`, on read.
    ANY load failure means "rebuild it", never "abort the group" — the one
    invariant this whole module exists for. `plan_point` is what tells
    "artifact missing", "artifact unreadable" and "stamp mismatch" apart for
    its human-readable `reason`.

    Reuses `pipeline.check_methodology_stamp`'s own comparison rather than
    re-implementing it — the two could otherwise drift apart silently.
    """
    if isinstance(path_or_frame, (str, Path)):
        path = Path(path_or_frame)
        if not path.exists():
            return False
        try:
            frame = io.read_neighbors(path)
        except Exception:
            return False
    else:
        frame = path_or_frame
    try:
        pipeline.check_methodology_stamp(frame, cfg)
    except validate.ValidationError:
        return False
    return True


def _mismatch_reason(frame, cfg):
    """A short human string naming the mismatched key, e.g. `"stamp
    mismatch: adjacency.max_distance_km"`, from an ALREADY LOADED frame —
    `plan_point` passes the one frame it read itself, so this never issues
    a second `io.read_neighbors` call just to describe a mismatch it has
    already detected (on `band-10km`, a ~4.37M-link frame, that second load
    would cost minutes spent building a string). Extracted from
    `check_methodology_stamp`'s own message (regex, not a second
    comparison) so this text can never disagree with `artifact_matches`."""
    try:
        pipeline.check_methodology_stamp(frame, cfg)
    except validate.ValidationError as exc:
        match = _STAMP_KEY_RE.search(str(exc))
        return f"stamp mismatch: {match.group(1) if match else exc}"
    return "artifact matches — preprocess skipped"  # pragma: no cover


def plan_point(profile, work_dir):
    """What `profile` needs against the artifact under `work_dir`: both
    stages when the artifact is missing, unreadable, or its stamp
    mismatches `profile`'s config; `compute` alone when it already matches.

    Reads the artifact AT MOST ONCE: when it exists and loads cleanly, that
    same in-memory frame is handed to `artifact_matches` (which accepts a
    frame directly) and, on a mismatch, to `_mismatch_reason` — never a
    second `io.read_neighbors` of a file that can be millions of links.
    A file that fails to load (missing or corrupt) is never read twice
    either: `artifact_matches` gets the untouched path and makes its own
    (cheap, exists-check-first) determination.
    """
    cfg = load_config(profile)
    artifact = Path(work_dir) / cfg.paths.neighbors_artifact
    frame = None
    if artifact.exists():
        try:
            frame = io.read_neighbors(artifact)
        except Exception:
            frame = None  # corrupt/truncated — treated exactly like missing

    if artifact_matches(frame if frame is not None else artifact, cfg):
        return Point(profile, artifact, ("compute",),
                    "artifact matches — preprocess skipped")

    if frame is not None:
        reason = _mismatch_reason(frame, cfg)
    elif artifact.exists():
        reason = "artifact unreadable — rebuilding"
    else:
        reason = "artifact missing"
    return Point(profile, artifact, ("preprocess", "compute"), reason)


def _plan_sequence(profiles, work_dir):
    """Points for `profiles`, in order, as `run_group` would actually plan
    them if it ran them one after another: once an earlier point's plan
    includes `preprocess` for a given artifact, a later point naming the
    SAME artifact is planned as `compute`-only — exactly what would be true
    once that preprocess had really run.

    --dry-run ONLY. `run_group` never needs this: by the time it plans a
    later point, an earlier point's preprocess (if any) has actually run
    and the real file on disk already reflects it. A dry run never writes
    that file, so without this a dry-run listing would show every point in
    a shared-artifact group as "missing", which is not what actually
    running the group would do.
    """
    built = set()
    points = []
    for profile in profiles:
        point = plan_point(profile, work_dir)
        if point.artifact.name in built:
            point = point._replace(
                stages=("compute",),
                reason="artifact matches — preprocess skipped")
        elif "preprocess" in point.stages:
            built.add(point.artifact.name)
        points.append(point)
    return points


# --- degree summary --------------------------------------------------------
def _holds_id_lists(series):
    """True iff this column's non-empty rows hold plain id lists (like
    `nbrs_bbox`) rather than [(id, value), ...] pairs (like
    `nbrs_dist_bbox` or `nbrs_barrier_weight`)."""
    for value in series:
        if isinstance(value, list) and value:
            return not isinstance(value[0], tuple)
    return False


def degree_report(frame, id_col, *, nbr_col=None):
    """`n_links` (DIRECTED — sum of list lengths), `deg_mean`, `deg_p50`,
    `deg_max`, `n_isolates` over `frame`'s neighbour-list column.

    `nbr_col` defaults to the single `nbrs_*` column holding plain id lists:
    `nbrs_bbox` is `code-2025`'s (and every adjacency rule's — the column
    keeps its historical name regardless of `rule`, per
    `delhi_psi.neighbors.adjacency`), but this does not hard-code that name
    so a future artifact shape is still handled without edits here.
    """
    if id_col not in frame.columns:
        raise KeyError(f"degree_report: {id_col!r} is not a column of this "
                       "frame")
    if nbr_col is None:
        candidates = [c for c in frame.columns if c.startswith("nbrs_")]
        holding = [c for c in candidates if _holds_id_lists(frame[c])]
        if len(holding) != 1:
            raise ValueError(
                "degree_report: cannot pick a neighbour-list column among "
                f"{candidates}; expected exactly one holding plain id lists")
        nbr_col = holding[0]
    degrees = frame[nbr_col].apply(len)
    n = len(degrees)
    return {
        "n_links": int(degrees.sum()),
        "deg_mean": float(degrees.mean()) if n else 0.0,
        "deg_p50": float(degrees.median()) if n else 0.0,
        "deg_max": int(degrees.max()) if n else 0,
        "n_isolates": int((degrees == 0).sum()),
    }


def _degree_from_disk(work_dir, artifact_name):
    """A manifest, from an earlier run of this script, whose point actually
    built `artifact_name` via `preprocess` — used only when this run's own
    in-memory cache has nothing (a `--only` selection, or a resumed run,
    that never (re-)planned the point that built the shared artifact)."""
    manifest_dir = Path(work_dir) / "manifest"
    if not manifest_dir.is_dir():
        return None
    for path in sorted(manifest_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("degree_from") != data.get("profile"):
            continue  # this manifest itself borrowed its degree data
        try:
            built = load_config(data["profile"]).paths.neighbors_artifact
        except Exception:
            continue
        if built == artifact_name:
            return {key: data[key] for key in _DEGREE_KEYS}, data["profile"]
    return None


# --- running one stage -----------------------------------------------------
def run_stage(profile, stage, *, data_dir, work_dir):
    """One `delhi-psi <stage>` subprocess, timed. Raises `StageFailed` on a
    non-zero exit; otherwise returns `{"seconds", "stdout", "stderr"}`.

    stdout and stderr are captured SEPARATELY, deliberately: the stage's
    dataclass result (`PreprocessResult`/`ComputeResult`) is `print()`ed to
    stdout by `delhi_psi.cli.main` on success, which is where the manifest's
    counts are parsed from (see `_extract_int` below) — interleaving it with
    stderr's log lines would make that parse fragile for no benefit, and on
    failure only stderr is needed for `stderr_tail`.
    """
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "delhi_psi.cli", stage, "--config", profile,
         "--data-dir", str(data_dir), "--out-dir", str(work_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.monotonic() - start
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-40:])
        raise StageFailed(stage, proc.returncode, tail)
    return {"seconds": elapsed, "stdout": proc.stdout, "stderr": proc.stderr}


def _extract_int(text, field):
    """Recover one integer field from a stage's printed dataclass repr,
    e.g. `n_settlements=1234` out of `PreprocessResult(..., n_settlements=
    1234, ...)`. `None` (never a guessed 0) when the field is not present —
    a stage that failed before printing, or stdout the caller didn't
    capture (a monkeypatched `run_stage` in a test)."""
    match = re.search(rf"\b{re.escape(field)}=(\d+)", text)
    return int(match.group(1)) if match else None


def _dedup_cache_warm(work_dir):
    """True iff `delhi_psi.pipeline._dedup_cached`'s settlement/barrier
    `*.dedup.stamp` files already exist under `work_dir` — i.e. some earlier
    preprocess IN THIS WORK DIR already paid the one-off O(n^2) dedup cost
    that `decay-none` is ordered first (spec § 7) specifically to absorb.
    Checked and recorded BEFORE a point's own preprocess runs, so a
    `preprocess_s` reading is self-describing: a reader can tell a cold-
    cache number from a warm one without reconstructing what happened —
    the gap this closes is `decay-none` failing and the NEXT point silently
    paying (and reporting) the cold-cache cost instead.
    """
    return any(Path(work_dir).glob("*.dedup.stamp"))


def _expected_outputs(cfg, work_dir):
    """The paths `compute` should have written, recovered from the config's
    own `outputs.denominators` / `outputs.formats` / `name_template` plus
    the `--out-dir work_dir` contract `run_stage` invokes the CLI with — NOT
    parsed from stdout. `ComputeResult.outputs` holds `Path` objects; its
    printed repr is a tuple of `PosixPath(...)` reprs, which would need a
    second, more fragile parser than the scalar counts `_extract_int`
    handles. Only paths that actually exist are reported.
    """
    paths = []
    for denominator in cfg.outputs.denominators:
        base = pipeline.output_basename(cfg, denominator)
        for fmt in cfg.outputs.formats:
            candidate = Path(work_dir) / f"{base}{_FORMAT_EXT[fmt]}"
            if candidate.exists():
                paths.append(str(candidate))
    return paths


# --- manifests ---------------------------------------------------------
def manifest_path(work_dir, profile):
    return Path(work_dir) / "manifest" / f"{profile}.json"


def _write_manifest(work_dir, profile, manifest):
    path = manifest_path(work_dir, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path


def run_group(group, *, work_dir, data_dir, run_date, commit, only=None):
    """Run every point of `group` (or only the ones named in `only`),
    writing one manifest per point. A point that raises `StageFailed`
    writes a FAILED manifest and the loop continues — one expensive point
    falling over must not cost the others.
    """
    work_dir = Path(work_dir)
    profiles = GROUPS[group]
    if only:
        profiles = tuple(p for p in profiles if p in only)

    # artifact filename -> (degree dict, profile that built it), THIS run
    # only. `_degree_from_disk` is the fallback for everything this cache
    # does not cover (a `--only` run, or a resumed one).
    built_degree = {}

    for profile in profiles:
        point = plan_point(profile, work_dir)
        cfg = load_config(profile)

        manifest = dict.fromkeys(MANIFEST_KEYS_OK)
        manifest["profile"] = profile
        manifest["skip_reason"] = point.reason
        manifest["stamp"] = pipeline.methodology_stamp(cfg.methodology)
        manifest["outputs"] = []
        manifest["commit"] = commit
        manifest["run_date"] = run_date

        completed = []
        failure = None
        for stage in point.stages:
            if stage == "preprocess":
                # Recorded BEFORE the subprocess runs — the dedup stamp
                # files this checks are exactly what THIS preprocess is
                # about to write, so checking after would always read
                # "warm" and defeat the point.
                manifest["dedup_cache_warm"] = _dedup_cache_warm(work_dir)
            try:
                timing = run_stage(profile, stage, data_dir=data_dir,
                                   work_dir=work_dir)
            except StageFailed as exc:
                failure = exc
                break
            completed.append(stage)
            stdout = timing.get("stdout", "")
            if stage == "preprocess":
                manifest["preprocess_s"] = timing["seconds"]
                manifest["n_settlements"] = _extract_int(stdout,
                                                          "n_settlements")
                manifest["n_barrier_flagged"] = _extract_int(
                    stdout, "n_barrier_flagged")
            else:
                manifest["compute_s"] = timing["seconds"]
                manifest["n_reported"] = _extract_int(stdout, "n_reported")
                manifest["n_missing_population"] = _extract_int(
                    stdout, "n_missing_population")
                manifest["outputs"] = _expected_outputs(cfg, work_dir)
        manifest["stages_run"] = completed

        # Degree summary: loaded fresh exactly once per artifact, only when
        # THIS point actually ran preprocess (the artifact is not otherwise
        # in the runner's memory, since the stage is a subprocess) — the
        # peak-memory moment of the cycle for band-10km (4.37 M links). A
        # point that skipped preprocess copies the summary of whichever
        # point built its artifact; if that cannot be found, the degree
        # keys are null (never a guessed zero, which would misread as "no
        # links" and trip the isolates flag).
        if "preprocess" in completed:
            frame = io.read_neighbors(point.artifact)
            degree = degree_report(frame, cfg.layers.settlements.id_col)
            manifest.update(degree)
            manifest["degree_from"] = profile
            built_degree[point.artifact.name] = (degree, profile)
        else:
            cached = (built_degree.get(point.artifact.name)
                     or _degree_from_disk(work_dir, point.artifact.name))
            if cached is not None:
                degree, source = cached
                manifest.update(degree)
                manifest["degree_from"] = source
            else:
                for key in _DEGREE_KEYS:
                    manifest[key] = None
                manifest["degree_from"] = "artifact predates this run"

        if failure is not None:
            manifest["status"] = "FAILED"
            manifest["failed_stage"] = failure.stage
            manifest["returncode"] = failure.returncode
            manifest["stderr_tail"] = failure.stderr_tail
        else:
            manifest["status"] = "OK"

        _write_manifest(work_dir, profile, manifest)


# --- CLI -------------------------------------------------------------------
def _current_commit():
    try:
        proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, check=True)
        return proc.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="run_sweep",
        description="Phase 6 dry-run sweep: run and cost the eleven "
                    "methodology profiles against real Delhi data (DEL-55).")
    parser.add_argument("--group", required=True, choices=sorted(GROUPS),
                        help="which sweep points to run")
    parser.add_argument("--data-dir", default="~/delhi_data",
                        help="input data root (default: ~/delhi_data)")
    parser.add_argument("--work-dir", default="~/psi_sweep",
                        help="scratch/output directory — never the data "
                             "directory (default: ~/psi_sweep)")
    parser.add_argument("--only", action="append", default=None,
                        help="restrict to this profile (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan for every point; run and "
                             "create nothing")
    parser.add_argument("--run-date", default=None,
                        help="stamped into every manifest (default: today), "
                             "so a re-run's manifests are reproducible when "
                             "set explicitly")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), "INFO"),
                        format="%(levelname)s %(name)s: %(message)s")
    data_dir = Path(args.data_dir).expanduser()

    if args.dry_run:
        # Same containment check a real run gets (resolve_work_dir's guard
        # fires identically either way), via its create=False keyword,
        # which only skips the mkdir — so the two paths cannot diverge.
        work_dir = resolve_work_dir(args.work_dir, data_dir=data_dir,
                                    create=False)
        profiles = GROUPS[args.group]
        if args.only:
            profiles = tuple(p for p in profiles if p in args.only)
        for point in _plan_sequence(profiles, work_dir):
            print(f"{point.profile}: {' + '.join(point.stages)} "
                 f"[{point.reason}] -> {point.artifact.name}")
        return

    work_dir = resolve_work_dir(args.work_dir, data_dir=data_dir)
    run_date = args.run_date or dt.date.today().isoformat()
    commit = _current_commit()
    run_group(args.group, work_dir=work_dir, data_dir=data_dir,
             run_date=run_date, commit=commit, only=args.only)


if __name__ == "__main__":
    main()
