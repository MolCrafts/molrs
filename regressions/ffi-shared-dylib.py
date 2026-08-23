"""Runtime half of the shared-dylib gate: one interpreter, both extensions.

`scripts/verify-shared-dylib.sh` proves the *link-time* half (both extensions
name the same libmolrs_ffi). This script proves the *process-time* half: with
molrs and molpack loaded into a single interpreter, a molrs Frame crosses into
molpack through the versioned `molrs_ffi` capsule, zero-copy.

Public API only. Zero third-party imports beyond numpy, which is a molrs
runtime dependency (the arrays are the documented way to fill a `Block`).

`import molpack` is itself an assertion: molpack's extension init runs
`interop::check_abi`, which calls `molrs._ffi_abi_token()` on the *installed*
molrs wheel and raises ImportError on a minor-line mismatch. An ImportError
here is a failure of this regression, not an environment problem.

Run standalone:  python regressions/ffi-shared-dylib.py
"""

from __future__ import annotations

import numpy as np

import molrs

# Handshake happens during this import; keep it after `import molrs` so a
# failure is unambiguously the handshake and not a missing producer.
import molpack

# ── Hard-coded goldens ───────────────────────────────────────────────────────
# minor line == ABI version. These change only when molrs moves to a new minor
# line, and changing them is a deliberate, reviewed act — never a refresh to
# make a red run green. Same shape as molrs-python/tests/test_ffi_abi.py.
ABI_LINE: str = "0.14"
FRAME_CAPSULE_NAME: str = "molrs.FrameRef/0.14"
ELEMENTS: list[str] = ["C", "C", "O"]

# ── 1. ABI token ─────────────────────────────────────────────────────────────
token: tuple[str, str, str, str] = molrs._ffi_abi_token()
assert len(token) == 4, f"expected a 4-tuple ABI token, got {token!r}"
assert token[0] == ABI_LINE, f"ABI line {token[0]!r} != golden {ABI_LINE!r}"
assert token[2] == FRAME_CAPSULE_NAME, (
    f"FrameRef capsule name {token[2]!r} != golden {FRAME_CAPSULE_NAME!r}"
)

# ── 2. A three-atom frame through the public Block/Frame API ─────────────────
block = molrs.Block()
block.insert("element", ELEMENTS)
block.insert("x", np.array([0.0, 1.5, 3.0], dtype=np.float64))
block.insert("y", np.array([0.0, 0.0, 0.0], dtype=np.float64))
block.insert("z", np.array([0.0, 0.0, 0.0], dtype=np.float64))
block.insert("id", np.array([1, 2, 3], dtype=np.uint32))

frame = molrs.Frame()
frame["atoms"] = block

# ── 3. The capsule carries the versioned name ────────────────────────────────
# The capsule repr embeds its name: <capsule object "..." at 0x...>.
capsule_repr: str = repr(frame._ffi_frameref_capsule())
assert f'"{FRAME_CAPSULE_NAME}"' in capsule_repr, (
    f"capsule repr does not carry {FRAME_CAPSULE_NAME!r}: {capsule_repr}"
)

# ── 4. Cross the boundary zero-copy ──────────────────────────────────────────
# Accessors verified against molpack/python/src/target.rs (#[getter]s at
# lines 263-291) and molpack.pyi's `class Target`: name, natoms, count,
# elements, radii, is_fixed. Asserting on natoms / count / elements only —
# radii are element-derived vdW values, not this gate's concern.
target = molpack.Target(frame, 1)
assert target.natoms == 3, f"natoms {target.natoms} != 3"
assert target.count == 1, f"count {target.count} != 1"
assert target.elements == ELEMENTS, f"elements {target.elements} != {ELEMENTS}"

print("ffi-shared-dylib regression: OK")
