"""Glob-run every example script. No file-name list."""

from __future__ import annotations

import glob
import subprocess
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_every_example_exits_zero() -> None:
    files = sorted(glob.glob(str(_EXAMPLES / "*.py")))
    assert files, f"no examples in {_EXAMPLES}"
    for path in files:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            cwd=str(_EXAMPLES.parent),
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"{path} exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
