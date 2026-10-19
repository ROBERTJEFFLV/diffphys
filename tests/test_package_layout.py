"""Package paths must also work for source binding and audit snapshots."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_source_lives_in_package():
    assert not list(ROOT.glob("*.py"))
    assert (ROOT / "diffphys/__init__.py").is_file()


def test_source_manifest_includes_package_and_resolves_from_repository_root():
    from diffphys.response_training import ROOT as source_root, SOURCE_FILES, source_hash

    assert source_root == ROOT
    assert "diffphys/__init__.py" in SOURCE_FILES
    assert "diffphys/env_raptor.py" in SOURCE_FILES
    assert "tools/train_response_control.py" in SOURCE_FILES
    assert all((ROOT / name).is_file() for name in SOURCE_FILES)
    assert {p.relative_to(ROOT).as_posix() for p in (ROOT / "diffphys").glob("*.py")} - {
        "diffphys/response_long_eval.py"
    } <= set(SOURCE_FILES)
    assert len(source_hash()) == 64
