"""Block CSV in and out, over the native parser — ``molrs.io.read_block_csv``
/ ``write_block_csv``.

Private: :mod:`molrs.io` is these names' public path.
"""

from __future__ import annotations

from io import StringIO
from os import PathLike
from pathlib import Path
from typing import Any

from .._lib import read_block_csv as _read_block_csv
from .._lib import write_block_csv as _write_block_csv

PathInput = str | PathLike[str]


def read_block_csv(
    source: PathInput | StringIO,
    *,
    delimiter: str = ",",
    encoding: str = "utf-8",
    header: list[str] | None = None,
    skip_empty_fields: bool = False,
) -> Any:
    """Read CSV into a :class:`Block`.

    Accepts the same sources every other reader in this module does — a path,
    or in-memory text via :class:`io.StringIO`. A bare ``str`` is a path when it
    names an existing file and CSV text otherwise.
    """
    if isinstance(source, StringIO):
        text = source.getvalue()
    else:
        path = Path(source)
        text = (
            path.read_text(encoding=encoding)
            if (isinstance(source, PathLike) or path.exists())
            else str(source)
        )
    d = delimiter if len(delimiter) == 1 else ","
    if skip_empty_fields:
        text = "\n".join(
            d.join(part for part in line.split(d) if part != "")
            for line in text.splitlines()
        )
    return _read_block_csv(text, d, header)


def write_block_csv(
    block: Any,
    filepath: PathInput | None = None,
    *,
    delimiter: str = ",",
    header: bool = True,
    encoding: str = "utf-8",
) -> str | None:
    """Write a :class:`Block` as CSV — the inverse of :func:`read_block_csv`.

    Returns the text when *filepath* is ``None``, else writes it and returns
    ``None``.
    """
    d = delimiter if len(delimiter) == 1 else ","
    text = _write_block_csv(block, d, header)
    if filepath is None:
        return text
    Path(filepath).write_text(text, encoding=encoding)
    return None
