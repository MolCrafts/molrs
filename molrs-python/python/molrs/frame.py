"""Rich Frame/Block Python layer over the molrs PyO3 core.

A pandas-style convenience API — dict-like mapping, selector/mask indexing,
sorting, row iteration, and CSV (``Block`` only) — layered in pure Python on top
of the Rust typed-column core (``molrs.Block`` / ``molrs.Frame`` from the PyO3
extension). Numeric / bool / string columns live in the Rust Store and are
exposed as zero-copy numpy views; this layer adds no per-access data copies.

Only numpy-representable dtypes (float / int / bool / str) are supported — there
is no Python-side object-column overflow. ``Block`` is the tidy columnar table
and owns ``to_csv`` / ``from_csv``; ``Frame`` is a container of named blocks and
deliberately has no CSV methods.
"""

from collections.abc import Iterator, Mapping, MutableMapping
from io import StringIO
from pathlib import Path
from typing import Any, Self, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .molrs import Block as _RsBlock
from .molrs import BlockDtypeError
from .molrs import Frame as _RsFrame
from .molrs import MetaValue

type BlockLike = Mapping[str, ArrayLike]


def _is_array_like(value: Any) -> bool:
    """True when ``value`` is a real 1-D-representable array (not a scalar).

    Used to decide whether a failed column assignment is a genuine
    not-array-like error (scalar / object) or a real array whose own error
    (length / shape / dtype mismatch) must be surfaced unmasked.
    """
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.ndarray) or hasattr(value, "__array__"):
        return True
    return isinstance(value, (list, tuple))


class Block(_RsBlock, MutableMapping[str, np.ndarray]):
    """Tidy columnar table mapping name -> 1D/2D numpy column.

    Inherits the PyO3 ``molrs.Block`` so a rich ``Block`` IS-A core block and is
    accepted by every ``molrs.*`` API with no conversion. All columns live in the
    Rust Store (numpy-representable dtypes only); reads are zero-copy views.

    Behaves like a dict and supports advanced indexing: by key (column), by
    int/slice (row / sub-block), by boolean mask, by list of keys (2D array), and
    by any callable selector (``key(self)``).
    """

    # No __slots__ — PyO3 base classes forbid subclass slot layouts; the single
    # Python-only attribute (_source) lives on __dict__.

    def __new__(cls, vars_: BlockLike | None = None) -> "Block":
        return super().__new__(cls)

    def __init__(self, vars_: BlockLike | None = None) -> None:
        super().__init__()
        # When set, numeric ops route through this external molrs.Block (a live
        # alias into a parent Frame's store) so frame[key][col] = arr writes
        # through. Populated by Block.from_dict(molrs.Block).
        self._source: "_RsBlock | None" = None
        if vars_ is not None:
            if not isinstance(vars_, dict):
                raise ValueError(
                    "Block value must be a dict of column-name -> array, "
                    f"got {type(vars_)}"
                )
            for k, v in vars_.items():
                try:
                    self[k] = v
                except BlockDtypeError:
                    # Surface the precise numpy-only contract error unchanged
                    # (object / None / ragged column) rather than masking it.
                    raise
                except Exception as e:
                    # Only claim "not array-like" when the value genuinely is
                    # not — a length/shape mismatch on a real array carries its
                    # own actionable message ("axis-0 length 4 but block
                    # expects 6"), which must NOT be masked.
                    if _is_array_like(v):
                        raise
                    raise ValueError(
                        f"Value must be array-like for key {k!r}, got {type(v)}"
                    ) from e

    # --- write-through routing ---------------------------------------------

    def _backing(self) -> "_RsBlock":
        """The molrs.Block numeric ops target (alias source or self)."""
        return self._source if self._source is not None else self

    def _as_storage(self) -> "_RsBlock":
        """The molrs.Block to hand to Frame.__setitem__ (the live storage)."""
        return self._backing()

    def view(self, key: str):  # type: ignore[override]
        return _RsBlock.view(self._backing(), key)

    def insert(self, key: str, array) -> None:  # type: ignore[override]
        _RsBlock.insert(self._backing(), key, array)

    def remove(self, key: str) -> None:  # type: ignore[override]
        _RsBlock.remove(self._backing(), key)

    def dtype(self, key: str) -> str:  # type: ignore[override]
        return _RsBlock.dtype(self._backing(), key)

    # --- core mapping API ---------------------------------------------------

    @overload
    def __getitem__(self, key: str) -> np.ndarray: ...
    @overload
    def __getitem__(self, key: int | slice) -> "Block": ...  # type: ignore[override]
    @overload
    def __getitem__(self, key: list[str]) -> np.ndarray: ...  # type: ignore[override]
    @overload
    def __getitem__(self, key: np.ndarray) -> "Block": ...  # type: ignore[override]

    def __getitem__(self, key):  # type: ignore[override]
        if isinstance(key, str):
            val = _RsBlock.view(self._backing(), key)
            return np.asarray(val) if isinstance(val, list) else val
        elif isinstance(key, (int, np.integer)) and not isinstance(key, bool):

            def _row(v):
                item = v[key]
                if np.ndim(item) > 0:
                    return item
                return item.item() if hasattr(item, "item") else item

            return {k: _row(v) for k, v in self._as_dict().items()}
        elif isinstance(key, slice):
            # Row slice -> Rust-native gather (no per-column NumPy slicing).
            indices = list(range(*key.indices(self.nrows)))
            return Block.from_dict(_RsBlock.select_rows(self._backing(), indices))
        elif isinstance(key, list):
            if not key:
                raise KeyError("Empty list not allowed for indexing")
            for k in key:
                if k not in self:
                    raise KeyError(f"Key '{k}' not found in Block")
            arrays = [self._view_array(k) for k in key]
            first = arrays[0]
            for i, arr in enumerate(arrays[1:], 1):
                if arr.shape != first.shape:
                    raise ValueError(
                        f"Arrays must have the same shape. Array {key[0]} has shape "
                        f"{first.shape}, but array {key[i]} has shape {arr.shape}"
                    )
                if arr.dtype != first.dtype:
                    raise ValueError(
                        f"Arrays must have the same dtype. Array {key[0]} has dtype "
                        f"{first.dtype}, but array {key[i]} has dtype {arr.dtype}"
                    )
            return np.column_stack(arrays)
        elif isinstance(key, tuple):
            return np.array([self[k] for k in key])
        elif isinstance(key, np.ndarray):
            # Boolean mask or integer fancy-index -> Rust-native row gather.
            n = self.nrows
            if key.dtype == bool:
                if key.shape[0] != n:
                    raise IndexError(
                        f"boolean index did not match block: block has {n} "
                        f"rows but mask has length {key.shape[0]}"
                    )
                idx = np.nonzero(key)[0]
            else:
                idx = key
            indices = [int(i) % n if n and int(i) < 0 else int(i) for i in idx]
            return Block.from_dict(_RsBlock.select_rows(self._backing(), indices))
        elif callable(key):
            # Duck-typed selector: any callable that filters/derives from a Block.
            return key(self)
        else:
            raise KeyError(
                f"Invalid key type: {type(key)}. "
                "Expected str, int, slice, list[str], ndarray, or callable."
            )

    def __setitem__(self, key: str, value: Any) -> None:  # type: ignore[override]
        arr = np.asarray(value)
        if arr.ndim == 0:
            raise ValueError(
                f"Block column '{key}' must be at least 1-D; got a scalar "
                f"({value!r}). Wrap it in a sequence (e.g. [{value!r}]) or "
                "broadcast it to the column length — scalar columns are not "
                "stored silently."
            )
        if key in _RsBlock.keys(self._backing()):
            _RsBlock.remove(self._backing(), key)
        _RsBlock.insert(self._backing(), key, arr)

    def __delitem__(self, key: str) -> None:  # type: ignore[override]
        _RsBlock.remove(self._backing(), key)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        yield from _RsBlock.keys(self._backing())

    def __len__(self) -> int:  # type: ignore[override]
        return len(_RsBlock.keys(self._backing()))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return _RsBlock.__contains__(self._backing(), key)

    def keys(self) -> list[str]:  # type: ignore[override]
        """All column names."""
        return list(_RsBlock.keys(self._backing()))

    # --- helpers ------------------------------------------------------------

    def _view_array(self, key: str) -> np.ndarray:
        val = _RsBlock.view(self._backing(), key)
        return np.asarray(val) if isinstance(val, list) else val

    def _as_dict(self) -> dict[str, np.ndarray]:
        return {k: self._view_array(k) for k in _RsBlock.keys(self._backing())}

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return each column as a numpy array (views into Rust memory)."""
        return {k: np.asarray(self._view_array(k)) for k in self.keys()}

    @classmethod
    def from_dict(cls, data: "dict[str, ArrayLike] | _RsBlock") -> "Block":
        """Build a Block from a dict, or alias a bare ``molrs.Block``.

        An already-rich Block is returned as-is. A bare ``molrs.Block`` is
        aliased (the returned block routes reads/writes through it — a live view
        of its storage, used for the ``frame[key][col] = arr`` write-through).
        """
        if isinstance(data, cls):
            return data
        if isinstance(data, _RsBlock):
            block = cls()
            block._source = data
            return block
        return cls({k: np.asarray(v) for k, v in data.items()})

    @classmethod
    def from_csv(
        cls,
        source: "str | Path | StringIO",
        *,
        delimiter: str = ",",
        encoding: str = "utf-8",
        header: list[str] | None = None,
        skipinitialspace: bool = False,
    ) -> "Block":
        """Create a Block from CSV.

        The CSV grammar + per-column dtype inference (int → float → str) is
        implemented in the molrs Rust core; this wrapper only resolves *source*
        (text, a file path, or a ``StringIO``) to text and adopts the parsed
        core block as a rich :class:`Block`.

        Args:
            skipinitialspace: When True, runs of the delimiter are collapsed so
                whitespace-aligned columns (e.g. LAMMPS data sections) parse
                cleanly. The core also trims each field; combined, leading and
                repeated delimiters never produce spurious empty columns.
        """
        if isinstance(source, StringIO):
            text = source.getvalue()
        elif isinstance(source, Path) or (
            isinstance(source, str) and Path(source).exists()
        ):
            text = Path(source).read_text(encoding=encoding)
        else:
            text = str(source)  # already CSV text
        d = delimiter if len(delimiter) == 1 else ","
        if skipinitialspace:
            # Collapse consecutive delimiters per line so aligned/whitespace-
            # padded columns don't yield empty fields.
            text = "\n".join(
                d.join(part for part in line.split(d) if part != "")
                for line in text.splitlines()
            )
        return cls.from_dict(_RsBlock.from_csv(text, d, header))

    def to_csv(
        self,
        filepath: "str | Path | None" = None,
        *,
        delimiter: str = ",",
        header: bool = True,
        encoding: str = "utf-8",
    ) -> "str | None":
        """Serialize the block to CSV (inverse of :meth:`from_csv`).

        CSV serialization lives in the molrs Rust core; this wrapper only writes
        the produced text to *filepath* (returning ``None``) or returns it as a
        string when *filepath* is ``None``.
        """
        d = delimiter if len(delimiter) == 1 else ","
        text = _RsBlock.to_csv(self._backing(), d, header)
        if filepath is None:
            return text
        Path(filepath).write_text(text, encoding=encoding)
        return None

    def copy(self) -> "Block":
        """Deep copy (data copied into a new Rust Store)."""
        new = Block()
        for k in _RsBlock.keys(self._backing()):
            new[k] = np.asarray(self._view_array(k))
        return new

    def rename(self, old_key: str, new_key: str) -> None:
        """Rename a column in place. Raises KeyError if *old_key* is absent."""
        if old_key not in _RsBlock.keys(self._backing()):
            raise KeyError(f"Column '{old_key}' not found in Block")
        arr = self._view_array(old_key)
        _RsBlock.remove(self._backing(), old_key)
        _RsBlock.insert(self._backing(), new_key, arr)

    def sort(self, key: str, *, reverse: bool = False) -> "Block":
        """Return a new Block sorted by *key* (original unchanged).

        The argsort + per-column gather runs in the Rust core
        (``molrs.Block.sort``); this is a thin call, not a NumPy reimplementation.
        """
        if self.nrows == 0:
            return self.copy()
        if key not in self:
            raise KeyError(f"Variable '{key}' not found in block")
        return Block.from_dict(_RsBlock.sort(self._backing(), key, reverse))

    def sort_(self, key: str, *, reverse: bool = False) -> "Self":
        """Sort the block in place by *key*; returns self."""
        if self.nrows == 0:
            return self
        if key not in self:
            raise KeyError(f"Variable '{key}' not found in block")
        ordered = _RsBlock.sort(self._backing(), key, reverse)
        cols = {
            k: np.asarray(_RsBlock.view(ordered, k)).copy()
            for k in _RsBlock.keys(ordered)
        }
        backing = self._backing()
        for k in list(_RsBlock.keys(backing)):
            _RsBlock.remove(backing, k)
        for k, v in cols.items():
            _RsBlock.insert(backing, k, v)
        return self

    def __repr__(self) -> str:
        contents = ", ".join(
            f"{k}: shape={self._view_array(k).shape}" for k in self.keys()
        )
        return f"Block({contents})"

    @property
    def nrows(self) -> int:  # type: ignore[override]
        """Number of rows (0 if empty)."""
        backing = self._backing()
        n = _RsBlock.nrows.__get__(backing, type(backing))  # type: ignore[attr-defined]
        return n if (n is not None and n > 0) else 0

    @property
    def shape(self) -> tuple[int, ...]:
        """(nrows, ncols), or () when empty."""
        if self.nrows == 0:
            return ()
        return self.nrows, len(self)

    def iterrows(self, n: int | None = None) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield (index, row_dict) for each row."""
        nrows = self.nrows if n is None else n
        if nrows == 0:
            return
        names = list(self.keys())
        for i in range(nrows):
            row: dict[str, Any] = {}
            for name in names:
                data = self._view_array(name)
                if i < len(data):
                    row[name] = data.item() if data.ndim == 0 else data[i]
                else:
                    row[name] = None
            yield i, row

    def itertuples(self, index: bool = True, name: str = "Row") -> Iterator[Any]:
        """Yield a named tuple per row."""
        from collections import namedtuple

        nrows = self.nrows
        if nrows == 0:
            return
        names = list(self.keys())
        fields = ["Index", *names] if index else names
        RowTuple = namedtuple(name, fields)
        for i in range(nrows):
            values: list[Any] = [i] if index else []
            for col in names:
                data = self._view_array(col)
                if i < len(data):
                    values.append(data.item() if data.ndim == 0 else data[i])
                else:
                    values.append(None)
            yield RowTuple(*values)


class Frame(_RsFrame):
    """Container of named :class:`Block` tables plus a box and metadata.

    Inherits the PyO3 ``molrs.Frame``: a rich ``Frame`` IS-A core frame, accepted
    by every ``molrs.*`` API with no conversion. ``__getitem__`` upgrades the
    stored block to a rich :class:`Block`. The ``box`` is the native
    ``molrs.Box`` (inherited). Frame has no CSV methods — CSV belongs to Block.
    """

    def __new__(
        cls,
        blocks: "dict[str, Block | BlockLike] | None" = None,
        meta: "dict[str, MetaValue] | None" = None,
    ) -> "Frame":
        return super().__new__(cls)

    def __init__(
        self,
        blocks: "dict[str, Block | BlockLike] | None" = None,
        meta: "dict[str, MetaValue] | None" = None,
    ) -> None:
        super().__init__()
        if meta is not None:
            self.meta = meta
        if blocks is not None:
            if not isinstance(blocks, dict):
                raise ValueError(f"blocks must be a dict, got {type(blocks)}")
            for key, value in blocks.items():
                if not isinstance(key, str):
                    raise ValueError(f"Block keys must be strings, got {type(key)}")
                self[key] = value if isinstance(value, Block) else Block(value)

    def __getitem__(self, key: str) -> Block:  # type: ignore[override]
        """Return the named block as a rich :class:`Block` (live view)."""
        return Block.from_dict(_RsFrame.__getitem__(self, key))

    def __setitem__(self, key: str, value: "BlockLike | Block") -> None:  # type: ignore[override]
        if isinstance(value, Block):
            mblock = value
        elif isinstance(value, _RsBlock):
            mblock = Block.from_dict(value)
        else:
            mblock = Block(value)
        _RsFrame.__setitem__(self, key, mblock._as_storage())

    def __delitem__(self, key: str) -> None:  # type: ignore[override]
        _RsFrame.__delitem__(self, key)

    def __contains__(self, key: str) -> bool:  # type: ignore[override]
        return _RsFrame.__contains__(self, key)

    def __len__(self) -> int:  # type: ignore[override]
        return _RsFrame.__len__(self)

    def keys(self):  # type: ignore[override]
        return _RsFrame.keys(self)

    @property
    def _blocks(self) -> dict[str, Block]:
        return {name: self[name] for name in self.keys()}

    @property
    def blocks(self) -> Iterator["Block"]:
        """Iterate over the stored blocks (as rich Blocks)."""
        return iter(self._blocks.values())

    def to_dict(self) -> dict[str, Any]:
        """Frame as ``{"blocks": {name: block.to_dict()}, "meta": {...}}``."""
        return {
            "blocks": {name: self[name].to_dict() for name in self.keys()},
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: "dict[str, Any] | _RsFrame") -> "Frame":
        """Build a Frame from a dict, or upgrade a bare ``molrs.Frame``."""
        if isinstance(data, cls):
            return data
        if isinstance(data, _RsFrame):
            frame = cls()
            for name in _RsFrame.keys(data):
                frame[name] = _RsFrame.__getitem__(data, name)
            raw_box = _RsFrame.box.__get__(data, type(data))
            if raw_box is not None:
                frame.box = raw_box
            if data.meta:
                frame.meta = dict(data.meta)
            return frame
        if set(data) != {"blocks", "meta"}:
            raise ValueError("frame dict must contain exactly 'blocks' and 'meta'")
        blocks = {name: Block.from_dict(blk) for name, blk in data["blocks"].items()}
        return cls(blocks=blocks, meta=data["meta"])

    @classmethod
    def _from_ffi_frameref_capsule(cls, capsule: Any) -> "Frame":
        """Build a rich ``Frame`` from a ``"molrs.FrameRef"`` capsule.

        The **return path** for a downstream Rust consumer (e.g. molpack hands a
        packed frame back as an FFI capsule): the Rust base resolves the capsule
        to a bare ``_RsFrame`` sharing the producer's store, then ``from_dict``
        upgrades it to this rich subclass so callers get an ``isinstance``-correct
        ``molrs.Frame``. Shadows the base ``_RsFrame`` staticmethod of the same
        name, the way ``from_dict`` adds the rich-layer wrapping.
        """
        base = _RsFrame._from_ffi_frameref_capsule(capsule)
        return cls.from_dict(base)

    def copy(self) -> "Frame":
        """Deep copy (blocks copied into new storage; box + metadata copied)."""
        new = Frame()
        for name in self.keys():
            new[name] = self[name].copy()
        new.box = self.box
        if self.meta:
            new.meta = dict(self.meta)
        return new

    def __repr__(self) -> str:
        txt = ["Frame("]
        for name in self.keys():
            blk = self[name]
            for k in blk.keys():
                txt.append(f"  [{name}] {k}: shape={blk[k].shape}")
        return "\n".join(txt) + "\n)"
