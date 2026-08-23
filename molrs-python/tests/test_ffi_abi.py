"""FFI ABI handshake: versioned capsule names + the `_ffi_abi_token` contract.

The project rule is minor-line = ABI version: capsule names carry
``major.minor`` so a cross-minor handle exchange fails the capsule name check
cleanly instead of dereferencing a possibly drifted layout. See
``molrs-ffi/src/abi.rs`` (single source of the names) and its layout snapshot
gate (the supply-side freeze).
"""

from __future__ import annotations

import ctypes
import importlib.metadata

import pytest

import molrs


def _abi_line() -> str:
    major, minor = importlib.metadata.version("molcrafts-molrs").split(".")[:2]
    return f"{major}.{minor}"


class TestAbiToken:
    def test_token_shape_and_line(self) -> None:
        line, version, frame_name, ff_name = molrs._ffi_abi_token()
        assert line == _abi_line()
        assert version == importlib.metadata.version("molcrafts-molrs")
        assert frame_name == f"molrs.FrameRef/{line}"
        assert ff_name == f"molrs.ForceFieldRef/{line}"


class TestVersionedCapsule:
    def test_capsule_carries_the_abi_line(self) -> None:
        cap = molrs.Frame()._ffi_frameref_capsule()
        # The capsule repr embeds its name: <capsule object "..." at 0x...>.
        assert f'"molrs.FrameRef/{_abi_line()}"' in repr(cap)

    def test_round_trip_shares_the_store(self) -> None:
        frame = molrs.Frame()
        back = molrs.Frame._from_ffi_frameref_capsule(frame._ffi_frameref_capsule())
        assert isinstance(back, molrs.Frame)

    def test_legacy_unversioned_capsule_is_rejected(self) -> None:
        # A pre-0.14 producer exports the unversioned name "molrs.FrameRef".
        # The payload pointer is bogus on purpose: the name check must reject
        # the capsule before any dereference happens.
        new_capsule = ctypes.pythonapi.PyCapsule_New
        new_capsule.restype = ctypes.py_object
        new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        legacy = new_capsule(ctypes.c_void_p(0xDEAD), b"molrs.FrameRef", None)
        with pytest.raises(ValueError, match="ABI line"):
            molrs.Frame._from_ffi_frameref_capsule(legacy)

    def test_forcefield_capsule_carries_the_abi_line(self) -> None:
        cap = molrs.ff.ForceField()._ffi_forcefield_capsule()
        assert f'"molrs.ForceFieldRef/{_abi_line()}"' in repr(cap)
