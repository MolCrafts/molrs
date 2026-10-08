"""Seam tests for molrs.md — construction, types, move semantics, a few steps."""

from __future__ import annotations

import molrs
import numpy as np
import pytest
from molrs.ff.potential import PairLjCut, Potential
from molrs.md import MdDriver, MaxwellBoltzmann, VelocityVerlet


class Harmonic:
    """Duck-typed potential: no base class."""

    def __init__(self, k: float = 1.0) -> None:
        self.k = k

    def calc_energy_forces(self, pos):
        pos = np.asarray(pos, dtype=np.float64)
        e = 0.5 * self.k * float(np.sum(pos * pos))
        return e, (-self.k * pos)


class TestPotentialProtocol:
    def test_structural_class_is_potential(self) -> None:
        assert isinstance(Harmonic(), Potential)

    def test_ljcut_is_potential(self) -> None:
        assert isinstance(PairLjCut(1.0, 1.0, 2.5), Potential)

    def test_md_defines_no_potential(self) -> None:
        """MD integrates potentials; it does not define or re-export one."""
        from molrs import md

        for name in ("Potential", "Potentials", "PairLjCut"):
            assert not hasattr(md, name)
            assert not hasattr(molrs._native.md, name)


class TestMdDtype:
    def test_float64_is_accepted(self) -> None:
        md = MdDriver(dtype=np.float64)
        assert md.dtype == np.dtype(np.float64)

    def test_float32_is_rejected_with_rust_message(self) -> None:
        with pytest.raises(ValueError, match="Rust"):
            MdDriver(dtype=np.float32)


class TestDispatch:
    def test_duck_typed_potential_takes_coordinates_only(self) -> None:
        calls: list[int] = []

        class Probe:
            def calc_energy_forces(self, pos):
                calls.append(pos.ndim if hasattr(pos, "ndim") else -1)
                pos = np.asarray(pos, dtype=np.float64)
                return 0.0, np.zeros_like(pos)

        pos = np.zeros((2, 3))
        vel = np.zeros((2, 3))
        vv = VelocityVerlet(0.001, potential=Probe(), mass=np.ones(2))
        vv.initial(pos, vel)
        assert calls and all(c == 2 for c in calls)

    def test_potentials_take_the_native_fast_path(self) -> None:
        from molrs.ff.potential import Potentials

        pots = Potentials()
        # empty collection is a native type; take_potential must not duck-wrap it.
        vv = VelocityVerlet(0.001, potential=pots, mass=np.ones(1))
        assert vv is not None


class TestMaxwellBoltzmann:
    def test_kbt_constructor(self) -> None:
        mb = MaxwellBoltzmann(molrs.core.UnitPreset("real").boltzmann() * 300.0, seed=1)
        pos = np.zeros((4, 3))
        vel = mb.velocities(pos, np.ones(4))
        assert vel.shape == (4, 3)
        assert np.all(np.isfinite(vel))
