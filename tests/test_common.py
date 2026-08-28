"""Tests for delhi_psi.io path resolution (was scripts/common.py)."""
from pathlib import Path

from delhi_psi.io import resolve_data_dir, resolve_out_dir


def test_flag_beats_env_and_default(monkeypatch, tmp_path):
    monkeypatch.setenv("DELHI_DATA_DIR", "/env/ignored")
    assert resolve_data_dir(str(tmp_path)) == tmp_path


def test_env_beats_default(monkeypatch, tmp_path):
    monkeypatch.setenv("DELHI_DATA_DIR", str(tmp_path))
    assert resolve_data_dir(None) == tmp_path


def test_default_is_home_delhi_data(monkeypatch):
    monkeypatch.delenv("DELHI_DATA_DIR", raising=False)
    assert resolve_data_dir(None) == Path("~/delhi_data").expanduser()


def test_flag_expands_user(monkeypatch):
    monkeypatch.delenv("DELHI_DATA_DIR", raising=False)
    assert resolve_data_dir("~/somewhere") == Path("~/somewhere").expanduser()


def test_out_dir_defaults_to_data_dir(tmp_path):
    out = resolve_out_dir(None, tmp_path)
    assert out == tmp_path


def test_out_dir_flag_wins_and_is_created(tmp_path):
    target = tmp_path / "sub" / "verify"
    out = resolve_out_dir(str(target), tmp_path)
    assert out == target
    assert target.is_dir()
