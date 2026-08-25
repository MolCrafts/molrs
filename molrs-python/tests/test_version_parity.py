"""Every crate / Python manifest shares one version string."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = "0.14.0"
_PRERELEASE = re.compile(r"\.dev|rc\d*|(?<![a-z])a\d|(?<![a-z])b\d")


def _manifests() -> list[Path]:
    out: list[Path] = []
    for p in ROOT.rglob("Cargo.toml"):
        if any(part in {"target", "target-aarch64", ".git"} for part in p.parts):
            continue
        out.append(p)
    pyproject = ROOT / "molrs-python" / "pyproject.toml"
    if pyproject.is_file():
        out.append(pyproject)
    assert out, "no manifests discovered"
    return out


def _version_strings(path: Path) -> list[str]:
    data = tomllib.loads(path.read_text())
    found: list[str] = []
    pkg = data.get("package") or data.get("project") or {}
    if isinstance(pkg.get("version"), str) and pkg["version"] != "workspace":
        found.append(pkg["version"])
    ws = data.get("workspace", {})
    if isinstance(ws.get("package", {}).get("version"), str):
        found.append(ws["package"]["version"])
    for table in (data.get("dependencies") or {}, (ws.get("dependencies") or {})):
        for name, spec in table.items():
            if name not in {"molcrafts-molrs", "molrs", "molrs-ffi", "molcrafts-molrs-ffi"}:
                continue
            if isinstance(spec, dict) and isinstance(spec.get("version"), str):
                found.append(spec["version"])
    return found


class TestVersionParity:
    def test_all_manifests_agree(self) -> None:
        versions: list[tuple[Path, str]] = []
        for path in _manifests():
            for v in _version_strings(path):
                versions.append((path, v))
        assert versions, "no version strings found"
        bad = [(p, v) for p, v in versions if v != EXPECTED]
        assert not bad, f"expected {EXPECTED}, got {bad}"

    def test_no_prerelease_suffix(self) -> None:
        hits: list[tuple[Path, str]] = []
        for path in _manifests():
            for v in _version_strings(path):
                if _PRERELEASE.search(v):
                    hits.append((path, v))
        assert not hits, f"prerelease versions: {hits}"
