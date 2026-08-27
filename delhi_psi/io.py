"""Filesystem seam: path resolution, layer/CSV reads, output writes.

Absorbs scripts/common.py. Never imports delhi_psi.config — the pipeline
passes explicit values.
"""

import logging
import os
import warnings
from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd
from pyogrio.errors import DataSourceError

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "~/delhi_data"

# Shapefiles cannot hold list or geometry-valued columns; production drops
# exactly these three before to_file (spec § 5).
SHAPEFILE_DROP_COLUMNS = ("nbrs_bbox", "nbrs_dist_bbox", "centroid")


def resolve_data_dir(cli_value=None):
    """Resolve the data directory from flag, env var, or default."""
    if cli_value:
        return Path(cli_value).expanduser()
    env_value = os.environ.get("DELHI_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser()
    return Path(DEFAULT_DATA_DIR).expanduser()


def out_dir_path(cli_value, data_dir):
    """Resolve the output directory (default: the data directory). No mkdir."""
    return Path(cli_value).expanduser() if cli_value else Path(data_dir)


def resolve_out_dir(cli_value, data_dir):
    """Resolve the output directory and create it."""
    out_dir = out_dir_path(cli_value, data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def read_layer(path, *, epsg=None):
    """Read a vector layer; optionally force a CRS (fixtures do this).

    A missing or unreadable source raises FileNotFoundError. pyogrio raises
    its own DataSourceError for that case, which the CLI's exit-code mapping
    (FileNotFoundError/OSError -> exit 1) would not catch (plan review R1,
    Critical).
    """
    try:
        gdf = gpd.read_file(path)
    except DataSourceError as exc:
        raise FileNotFoundError(f"cannot read vector layer {path}: {exc}") from exc
    if epsg is not None:
        gdf = gdf.set_crs(epsg=epsg, allow_override=True)
    return gdf


def read_population(path):
    return pd.read_csv(path)


def read_neighbors(path):
    with open(path, "rb") as handle:
        return joblib.load(handle)


def write_neighbors(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        joblib.dump(frame, handle)
    log.info("wrote %s", path)
    return path


def _write_shapefile(frame, path):
    """Write a shapefile, muting the two warnings this repo accepts.

    geopandas truncates column names over 10 characters and pyogrio emits one
    'Normalized/laundered field name' RuntimeWarning per truncated column from
    its C error handler (under -W error it can surface as
    PytestUnraisableExceptionWarning). Both are accepted behaviour — they were
    invisible only because the old e2e test ran the scripts in a subprocess.
    The filter is scoped to THIS write and nothing else.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Column names longer than 10 characters",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Normalized/laundered field name",
            category=RuntimeWarning,
        )
        frame.to_file(path)
    return path


def write_outputs(frame, out_dir, *, basename, formats):
    """Write one PSI result set. Returns the paths written, in order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        if fmt == "csv":
            path = out_dir / f"{basename}.csv"
            frame.to_csv(path, index=False)
        elif fmt == "shp":
            path = out_dir / f"{basename}.shp"
            droppable = [c for c in SHAPEFILE_DROP_COLUMNS
                         if c in frame.columns]
            _write_shapefile(frame.drop(columns=droppable), path)
        elif fmt == "joblib":
            path = out_dir / f"{basename}.joblib"
            with open(path, "wb") as handle:
                joblib.dump(frame, handle)
        else:
            raise ValueError(
                f"unknown output format {fmt!r}; allowed values: "
                "['csv', 'shp', 'joblib']")
        written.append(path)
        log.info("wrote %s", path)
    return written
