"""CSV tables as a :class:`~molrs.core.Block` — ``molrs.io.read_csv_block``,
``read_csv_block_str``, ``write_csv_block`` and ``write_csv_block_str``, over
the native parser (``molrs::io``'s doors of the same names).

Private: :mod:`molrs.io` is these names' public path.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

from .._native import read_csv_block_str as _native_read_csv_block_str
from .._native import write_csv_block_str as _native_write_csv_block_str

PathInput = str | PathLike[str]


def _delimiter(delimiter: str) -> str:
    """``delimiter``, which must be one character; ``ValueError`` otherwise."""
    if len(delimiter) != 1:
        raise ValueError(f"a CSV delimiter is one character, got {delimiter!r}")
    return delimiter


def read_csv_block_str(
    text: str,
    *,
    delimiter: str = ",",
    header: list[str] | None = None,
    skip_empty_fields: bool = False,
) -> Any:
    """Read CSV text into a :class:`~molrs.core.Block`.

    Each column's dtype is inferred int → float → str. When ``header`` is
    given the text is treated as headerless and those names are used;
    otherwise the first non-empty line names the columns. With
    ``skip_empty_fields`` an empty field (a doubled delimiter) is dropped
    rather than read as a value. A ``delimiter`` that is not one character
    raises ``ValueError``.
    """
    d = _delimiter(delimiter)
    if skip_empty_fields:
        text = "\n".join(
            d.join(part for part in line.split(d) if part != "")
            for line in text.splitlines()
        )
    return _native_read_csv_block_str(text, d, header)


def read_csv_block(
    path: PathInput,
    *,
    delimiter: str = ",",
    encoding: str = "utf-8",
    header: list[str] | None = None,
    skip_empty_fields: bool = False,
) -> Any:
    """Read the CSV file at ``path`` into a :class:`~molrs.core.Block` —
    :func:`read_csv_block_str` over the file's text."""
    text = Path(path).read_text(encoding=encoding)
    return read_csv_block_str(
        text, delimiter=delimiter, header=header, skip_empty_fields=skip_empty_fields
    )


def write_csv_block_str(block: Any, *, delimiter: str = ",", header: bool = True) -> str:
    """Write a :class:`~molrs.core.Block` as CSV text — the inverse of
    :func:`read_csv_block_str`."""
    return _native_write_csv_block_str(block, _delimiter(delimiter), header)


def write_csv_block(
    path: PathInput,
    block: Any,
    *,
    delimiter: str = ",",
    header: bool = True,
    encoding: str = "utf-8",
) -> None:
    """Write a :class:`~molrs.core.Block` as a CSV file at ``path`` — the
    inverse of :func:`read_csv_block`."""
    Path(path).write_text(
        write_csv_block_str(block, delimiter=delimiter, header=header), encoding=encoding
    )
