from __future__ import annotations

import subprocess
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PACKAGE_ROOT / "src" / "semafold"
EXPECTED_SOURCE_FILES = {
    "src/semafold/__init__.py",
    "src/semafold/_version.py",
    "src/semafold/errors.py",
    "src/semafold/py.typed",
    "src/semafold/core/__init__.py",
    "src/semafold/core/accounting.py",
    "src/semafold/core/evidence.py",
    "src/semafold/core/models.py",
    "src/semafold/vector/__init__.py",
    "src/semafold/vector/codecs/__init__.py",
    "src/semafold/vector/codecs/passthrough.py",
    "src/semafold/vector/codecs/scalar_reference.py",
    "src/semafold/vector/models.py",
    "src/semafold/vector/protocols.py",
    "src/semafold/turboquant/__init__.py",
    "src/semafold/turboquant/codebook.py",
    "src/semafold/turboquant/codec_mse.py",
    "src/semafold/turboquant/codec_prod.py",
    "src/semafold/turboquant/backends/__init__.py",
    "src/semafold/turboquant/backends/_mlx.py",
    "src/semafold/turboquant/backends/_numpy.py",
    "src/semafold/turboquant/backends/_protocol.py",
    "src/semafold/turboquant/backends/_registry.py",
    "src/semafold/turboquant/backends/_torch.py",
    "src/semafold/turboquant/kv/__init__.py",
    "src/semafold/turboquant/kv/layout.py",
    "src/semafold/turboquant/kv/preview.py",
    "src/semafold/turboquant/packing.py",
    "src/semafold/turboquant/qjl.py",
    "src/semafold/turboquant/quantizer.py",
    "src/semafold/turboquant/rotation.py",
}


def test_source_tree_matches_current_inventory() -> None:
    actual = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in sorted(SOURCE_ROOT.rglob("*"))
        if path.is_file()
    }
    assert actual == EXPECTED_SOURCE_FILES


def test_generated_artifacts_are_not_tracked_in_git() -> None:
    completed = subprocess.run(
        ["git", "ls-files", "."],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    }
    forbidden = {
        path
        for path in tracked
        if "__pycache__/" in path or path.startswith(".pytest_cache/") or path.endswith((".pyc", ".pyo"))
    }
    assert forbidden == set()


def test_package_tree_has_no_generated_bytecode_artifacts() -> None:
    pycache_dirs = sorted(path.relative_to(PACKAGE_ROOT).as_posix() for path in PACKAGE_ROOT.rglob("__pycache__"))
    compiled = sorted(path.relative_to(PACKAGE_ROOT).as_posix() for path in PACKAGE_ROOT.rglob("*.py[co]"))
    assert pycache_dirs == []
    assert compiled == []


def test_source_tree_has_no_type_ignore_markers() -> None:
    offenders: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "type: ignore" in text or "pyright: ignore" in text:
            offenders.append(path.relative_to(SOURCE_ROOT).as_posix())
    assert offenders == []
