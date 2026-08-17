"""Shared helpers for the pipeline scripts (Phase 1).

Path resolution order (spec): --data-dir flag > DELHI_DATA_DIR env var
> ~/delhi_data.
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


def resolve_out_dir(cli_value, data_dir):
    """Resolve the output directory (default: the data directory) and create it."""
    out_dir = Path(cli_value).expanduser() if cli_value else Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
