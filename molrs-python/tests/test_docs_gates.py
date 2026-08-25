"""Static gates for 0.14 docs: molpy spelling, fs time, no deleted names."""

from __future__ import annotations

import re
from pathlib import Path

_SITE = Path(__file__).resolve().parents[1] / "site-src"
_COMPUTE = Path(__file__).resolve().parents[1] / "python" / "molrs" / "compute"
_ZENSICAL = Path(__file__).resolve().parents[1] / "zensical.toml"
_MIGRATION = _SITE / "getting-started" / "migration-0-14.md"
_MD_GUIDE = _SITE / "guides" / "md.md"

_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.S)
_DELETED = (
    "energy_to_md",
    "preset_energy_to_md",
    "kb_md",
    "set_energy_scale",
    "prec=",
    "resolve_prec",
)


def _python_blocks(path: Path) -> list[str]:
    return _PYTHON_FENCE.findall(path.read_text(encoding="utf-8"))


class TestSitePythonSpellsMolpy:
    def test_python_fences_do_not_import_molrs(self) -> None:
        hits: list[str] = []
        for path in _SITE.rglob("*.md"):
            for i, block in enumerate(_python_blocks(path), 1):
                if "import molrs" in block or "from molrs" in block or "molrs." in block:
                    hits.append(f"{path.relative_to(_SITE)} block {i}")
        assert not hits, "site-src python fences still spell molrs:\n" + "\n".join(hits)


class TestAnalysisTimeIsFs:
    def test_no_ps_in_dt_lag_analysis_context(self) -> None:
        ctx = re.compile(
            r"(dt|lag|analysis).{0,40}\bps\b|\bps\b.{0,40}(dt|lag|analysis)", re.I
        )
        hits: list[str] = []
        for root in (_SITE, _COMPUTE):
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix not in {".md", ".py"}:
                    continue
                for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if ctx.search(line):
                        hits.append(f"{path}:{i}:{line.strip()}")
        assert not hits, "ps still used as analysis time:\n" + "\n".join(hits)


class TestNav:
    def test_new_pages_are_in_zensical_nav(self) -> None:
        nav = _ZENSICAL.read_text(encoding="utf-8")
        assert "getting-started/migration-0-14.md" in nav
        assert "guides/md.md" in nav


class TestDeletedNamesStayGone:
    def test_migration_and_md_guides_omit_deleted_symbols(self) -> None:
        text = (
            _MIGRATION.read_text(encoding="utf-8")
            + "\n"
            + _MD_GUIDE.read_text(encoding="utf-8")
        )
        found = [name for name in _DELETED if name in text]
        assert not found, f"deleted symbols still in 0.14 guides: {found}"
