r"""Ragged 3-frame Zarr trajectory: bit-exact round trip, bounded file count,
named refusal of the pre-0.14 layout (0.14.15, ac-032).

One `molrs.Trajectory` carries frames of 3 / 5 / 4 atoms with an f64 + i64 +
bool + u32 column set and one per-step meta scalar. It is written to a store,
read back through the public door, and every coordinate, column, step, time and
meta value is compared against the literals below **bit-for-bit** — no
tolerance: floats are compared as their IEEE-754 bit patterns, so -0.0, a NaN,
`DBL_MAX` and the smallest subnormal are all load-bearing, and the integer
goldens carry values (2**53+1, 2**32-1, 2**31) that no f64 or i32 detour can
reproduce. The file count is then held under a bound derived from the layout,
and a hand-grafted `trajectory/frames/` node must be refused by name rather
than read as an empty trajectory.

Provenance of the goldens: hand-written literals, no external oracle and no
third-party scientific package at run time (`molrs` + `numpy` only, numpy being
how molrs hands out columns). The file bound is arithmetic over the node layout,
spelled out at `FILE_BOUND`. Runner:

    uv --directory ../molrec run python \
        regressions/release-0-14-15-molrec-zarr-trajectory.py

(any environment carrying a molrs >= 0.14.15 wheel will do; 2026-08-29).
"""
from __future__ import annotations

import json
import os
import tempfile
import warnings

import numpy as np
from numpy.testing import assert_array_equal

warnings.filterwarnings("ignore", category=FutureWarning)

import molrs

# --- goldens ---------------------------------------------------------------
# Ragged: three frames, three different atom counts. Reading frame i back must
# yield exactly N_ATOMS[i] rows -- the sequence stores one flat column per name
# and cuts it with an offset array, so a wrong cut shows up here first.
N_ATOMS = (3, 5, 4)

# Explicit step numbers (dimensionless) and times (fs, per science.md). The
# times are the doubles nearest 0.1+0.2 and 0.2+0.4: decimal-unfriendly on
# purpose, so a lossy encoding cannot round-trip them.
STEPS = (0, 10, 20)
TIMES = (0.0, 0.30000000000000004, 0.6000000000000001)

# f64 coordinates. Frame 0 carries the negative zero, the largest finite
# double and the smallest positive subnormal; frame 2 carries a NaN.
ATOM_X = (
    (0.0, 0.1, -0.1),
    (0.30000000000000004, 1.1, 2.2, 3.3000000000000003, 4.4),
    (9.9, 8.8, 7.7, 6.6),
)
ATOM_Y = (
    (-0.0, 1.7976931348623157e308, 5e-324),
    (-1.5, -2.5, -3.5, -4.5, -5.5),
    (float("nan"), 0.0, 2.220446049250313e-16, -1.0),
)
ATOM_Z = (
    (3.141592653589793, -2.718281828459045, 1.4142135623730951),
    (1e-300, 1e300, 0.0, 123456789.12345679, 6.02214076e23),
    (0.5, 0.25, 0.125, 0.0625),
)

# i64, on a name the canonical Frame vocabulary does not declare (a signed user
# tag). 2**53+1 and its negation are not representable as f64, so a column that
# ever passed through a double comes back off by one here. The magnitudes are
# chosen to pin the width, not to look like plausible labels.
ATOM_TAG = (
    (1, 2, 3),
    (-1, -2, -3, -4, -5),
    (9007199254740993, 9007199254740994, -9007199254740993, 4),
)
# bool: mixed patterns, including an all-False frame.
ATOM_FROZEN = (
    (True, False, True),
    (False, False, False, False, False),
    (True, True, False, True),
)
# u32, again on a name the canonical vocabulary leaves unconstrained: the
# declared `uint` keys (`id`, `type_id`, ...) are stored at the store's own
# `uint` width, and what is under test here is that a column keeps the width it
# arrived with. 2**32-1 and 2**31 do not fit an i32, so a narrowing detour is
# visible.
ATOM_KIND = (
    (1, 2, 3),
    (1, 2, 3, 4, 5),
    (4294967295, 0, 2147483648, 7),
)

# One per-step meta scalar (f64), decimal-unfriendly on purpose.
META_KEY = "temperature"
META_TEMPERATURE = (300.1, 0.30000000000000004, -273.15)

# Nodes in the written store, counted once from the layout:
#   groups (metadata only):  /, /meta, /trajectory, /trajectory/meta,
#                            /trajectory/atoms                       -> 5
#   arrays (metadata + <= 1 chunk file, since 12 rows fit one chunk):
#     trajectory/step, trajectory/time, trajectory/meta/temperature,
#     trajectory/atoms/{x,y,z,tag,frozen,kind},
#     trajectory/atoms/offset, trajectory/atoms/step_index           -> 11
#   5 + 2 * 11 = 27
# The bound is a function of the *schema*, not of the frame count: appending
# more frames grows the chunks, not the node set. (The pre-0.14 layout spent
# about this many files on every single frame.)
FILE_BOUND = 27


def _f64(values: tuple[float, ...]) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def _bits(values: np.ndarray) -> np.ndarray:
    """IEEE-754 bit patterns, so -0.0 != 0.0 and NaN payloads must match."""
    return np.ascontiguousarray(values, dtype=np.float64).view(np.int64)


def _frame(i: int) -> molrs.Frame:
    block = molrs.Block()
    block["x"] = _f64(ATOM_X[i])
    block["y"] = _f64(ATOM_Y[i])
    block["z"] = _f64(ATOM_Z[i])
    block["tag"] = np.array(ATOM_TAG[i], dtype=np.int64)
    block["frozen"] = np.array(ATOM_FROZEN[i], dtype=np.bool_)
    block["kind"] = np.array(ATOM_KIND[i], dtype=np.uint32)
    return molrs.Frame({"atoms": block}, {META_KEY: META_TEMPERATURE[i]})


def _count_files(root: str) -> int:
    return sum(len(files) for _dir, _subdirs, files in os.walk(root))


def _write_group(path: str) -> None:
    """A minimal Zarr V3 group node, hand-built."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), "w", encoding="utf-8") as handle:
        json.dump({"zarr_format": 3, "node_type": "group", "attributes": {}}, handle)


with tempfile.TemporaryDirectory() as tmp:
    # --- 1. write ----------------------------------------------------------
    trajectory = molrs.Trajectory(
        [_frame(i) for i in range(3)],
        step=np.array(STEPS, dtype=np.int64),
        time=_f64(TIMES),
    )
    assert len(trajectory) == 3, f"built {len(trajectory)} frames, expected 3"
    store = os.path.join(tmp, "traj.zarr")
    trajectory.write(store)

    # --- 2. read back, bit for bit ----------------------------------------
    loaded = molrs.Trajectory.read(store)
    assert len(loaded) == 3, f"read back {len(loaded)} frames, expected 3"
    assert loaded.count_frames() == 3, f"count_frames() = {loaded.count_frames()}"
    assert_array_equal(np.asarray(loaded.step), np.array(STEPS, dtype=np.int64))
    assert_array_equal(_bits(np.asarray(loaded.time)), _bits(_f64(TIMES)))

    for i in range(3):
        frame = loaded[i]
        assert frame.keys() == ["atoms"], f"frame {i} blocks: {frame.keys()}"
        atoms = frame["atoms"]
        assert atoms.nrows == N_ATOMS[i], (
            f"frame {i} came back with {atoms.nrows} atoms, expected {N_ATOMS[i]}"
        )

        for name, golden in (("x", ATOM_X[i]), ("y", ATOM_Y[i]), ("z", ATOM_Z[i])):
            got = np.asarray(atoms[name])
            want = _f64(golden)
            assert got.dtype == np.float64, f"frame {i} {name}: dtype {got.dtype}"
            assert_array_equal(got, want, err_msg=f"frame {i} column {name}")
            assert_array_equal(
                _bits(got), _bits(want), err_msg=f"frame {i} column {name} (bits)"
            )

        got_tag = np.asarray(atoms["tag"])
        assert got_tag.dtype == np.int64, f"frame {i} tag: dtype {got_tag.dtype}"
        assert_array_equal(got_tag, np.array(ATOM_TAG[i], dtype=np.int64))

        got_frozen = np.asarray(atoms["frozen"])
        assert got_frozen.dtype == np.bool_, (
            f"frame {i} frozen: dtype {got_frozen.dtype}"
        )
        assert_array_equal(got_frozen, np.array(ATOM_FROZEN[i], dtype=np.bool_))

        got_kind = np.asarray(atoms["kind"])
        assert got_kind.dtype == np.uint32, f"frame {i} kind: dtype {got_kind.dtype}"
        assert_array_equal(got_kind, np.array(ATOM_KIND[i], dtype=np.uint32))

        meta = dict(frame.meta)
        assert set(meta) == {META_KEY}, f"frame {i} meta keys: {sorted(meta)}"
        entry = meta[META_KEY]
        assert entry.dtype == "f64", f"frame {i} meta dtype: {entry.dtype}"
        value = float(entry.value)
        assert value == META_TEMPERATURE[i], (
            f"frame {i} meta {META_KEY}: {value!r} != {META_TEMPERATURE[i]!r}"
        )
        assert np.float64(value).view(np.int64) == np.float64(
            META_TEMPERATURE[i]
        ).view(np.int64), f"frame {i} meta {META_KEY} differs in the low bits"

    # --- 3. file count is bounded by the schema, not by the frames --------
    n_files = _count_files(store)
    assert n_files <= FILE_BOUND, (
        f"{n_files} files under the store, bound is {FILE_BOUND}: "
        "the sequence layout is spending files per frame again"
    )

    # --- 4. the pre-0.14 layout is refused by name ------------------------
    legacy = os.path.join(tmp, "legacy.zarr")
    trajectory.write(legacy)
    _write_group(os.path.join(legacy, "trajectory", "frames"))
    _write_group(os.path.join(legacy, "trajectory", "frames", "0"))
    try:
        molrs.Trajectory.read(legacy)
    except Exception as exc:  # noqa: BLE001 -- the message is the assertion
        message = str(exc)
    else:
        raise AssertionError(
            "a store carrying trajectory/frames/ was read instead of refused"
        )
    assert "legacy layout (written by molrs" in message, (
        f"must name the layout: {message}"
    )
    assert "0.13); re-write with 0.13" in message, (
        f"must say which writer produced it and how to migrate: {message}"
    )

print(
    f"molrec-zarr-trajectory ok: frames=3 atoms={'/'.join(map(str, N_ATOMS))} "
    f"files={n_files}<={FILE_BOUND} legacy-layout refused"
)
