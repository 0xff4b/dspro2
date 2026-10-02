"""Tests for rentml.config."""

import os
from pathlib import Path

import pytest

from rentml.config import CANTON_ABBR, QUANTILE_LEVELS, ProjectPaths, load_env


def test_discover_finds_project_root_from_subdir(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "notebooks").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "rentml"\n', encoding="utf-8")
    paths = ProjectPaths.discover(root / "notebooks")
    assert paths.project_root == paths.repo_root == root.resolve()
    assert paths.figures == (root / "docs" / "fig").resolve()
    assert paths.dspro1_csv_dir == (root / "data" / "dspro1_snapshot").resolve()
    assert paths.baseline_model.parent == (root / "models" / "baseline").resolve()


def test_discover_ignores_nested_project_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "dspro2"
    nested.mkdir()
    (nested / "pyproject.toml").write_text('[project]\nname = "rentml"\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        ProjectPaths.discover(tmp_path)


def test_discover_raises_without_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "other"\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        ProjectPaths.discover(tmp_path)


def test_ensure_creates_directories(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path / "repo")
    paths.ensure()
    assert paths.cache.is_dir() and paths.results.is_dir()


def test_load_env_does_not_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = ProjectPaths.from_root(tmp_path / "repo")
    paths.project_root.mkdir(parents=True)
    (paths.project_root / ".env").write_text("RENTML_TEST_VAR=from_file\n", encoding="utf-8")
    monkeypatch.setenv("RENTML_TEST_VAR", "preset")
    loaded = load_env(paths)
    assert loaded == [paths.project_root / ".env"]
    assert os.environ["RENTML_TEST_VAR"] == "preset"


def test_constants_are_consistent() -> None:
    assert len(CANTON_ABBR) == 26
    assert list(QUANTILE_LEVELS) == sorted(QUANTILE_LEVELS)
    assert 0.5 in QUANTILE_LEVELS
