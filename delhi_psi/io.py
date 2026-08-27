"""Filesystem seam: path resolution, layer/CSV reads, output writes.

Absorbs scripts/common.py. Never imports delhi_psi.config — the pipeline
passes explicit values.
"""

import os
from pathlib import Path

DEFAULT_DATA_DIR = "~/delhi_data"


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
