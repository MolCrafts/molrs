"""The kernels of ``molrs.ff.potential``, built by hand and moved into ``Potentials``.

Each class takes explicit instances in the force field's convention (LAMMPS's,
angles in degrees). The tests prove the Python surface: every kernel prices its
LAMMPS energy, ``Potentials.push`` moves it (a second push refuses), and a
collection assembled by hand prices a molecule exactly as the compiled force
field does.
"""

from __future__ import annotations

import math

import molrs
import numpy as np
import pytest
from molrs.ff import Potentials
from molrs.ff.potential import (
    AngleHarmonic,
    BondHarmonic,
    DihedralPeriodic,
    ImproperCvff,
    ImproperPeriodic,
    LJCut,
    PairCoulCut,
)

# A non-planar four-atom chain: every angle and the dihedral are generic.
XYZ = np.array(
    [[1.2, -0.4, 0.3], [0.0, 0.0, 0.0], [-0.2, 1.5, 0.1], [0.9, 2.1, -0.8]],
    dtype=np.float64,
)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    return math.acos(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    b1, b2, b3 = b - a, c - b, d - c
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return math.atan2(m1 @ n2, n1 @ n2)


def _fd_forces(kernel_factory, pos: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference ``-dE/dx`` with a fresh kernel per evaluation."""
    out = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for d in range(3):
            plus, minus = pos.copy(), pos.copy()
            plus[i, d] += h
            minus[i, d] -= h
            e_plus = kernel_factory().calc_energy_forces(plus)[0]
            e_minus = kernel_factory().calc_energy_forces(minus)[0]
            out[i, d] = -(e_plus - e_minus) / (2 * h)
    return out


class TestKernels:
    def test_bond_harmonic_prices_k_r_minus_r0_squared(self) -> None:
        e, f = BondHarmonic([0], [1], [300.0], [1.4]).calc_energy_forces(XYZ)
        r = np.linalg.norm(XYZ[0] - XYZ[1])
        assert math.isclose(e, 300.0 * (r - 1.4) ** 2, rel_tol=1e-12)
        np.testing.assert_allclose(
            f, _fd_forces(lambda: BondHarmonic([0], [1], [300.0], [1.4]), XYZ), atol=1e-5
        )

    def test_angle_harmonic_takes_theta0_in_degrees(self) -> None:
        e, _ = AngleHarmonic([0], [1], [2], [50.0], [109.5]).calc_energy_forces(XYZ)
        theta = _angle(XYZ[0], XYZ[1], XYZ[2])
        assert math.isclose(e, 50.0 * (theta - math.radians(109.5)) ** 2, rel_tol=1e-12)

    def test_dihedral_periodic_sums_its_series_with_phases_in_degrees(self) -> None:
        k = np.array([[1.3, 0.4, 0.0]])
        n = np.array([[1.0, 2.0, 3.0]])
        phase = np.array([[0.0, 180.0, 0.0]])

        def make() -> DihedralPeriodic:
            return DihedralPeriodic([0], [1], [2], [3], k, n, phase)

        e, f = make().calc_energy_forces(XYZ)
        phi = _dihedral(*XYZ)
        hand = sum(
            k[0, t] * (1 + math.cos(n[0, t] * phi - math.radians(phase[0, t])))
            for t in range(3)
        )
        assert math.isclose(e, hand, rel_tol=1e-12)
        np.testing.assert_allclose(f, _fd_forces(make, XYZ), atol=1e-5)

    def test_improper_cvff_prices_the_dihedral_of_the_stored_order(self) -> None:
        e, _ = ImproperCvff([1], [0], [2], [3], [2.0], [-1.0], [2.0]).calc_energy_forces(XYZ)
        phi = _dihedral(XYZ[1], XYZ[0], XYZ[2], XYZ[3])
        assert math.isclose(e, 2.0 * (1 - math.cos(2 * phi)), rel_tol=1e-12)

    def test_improper_periodic_takes_its_phase_in_degrees(self) -> None:
        e, _ = ImproperPeriodic([0], [2], [1], [3], [1.1], [2.0], [180.0]).calc_energy_forces(
            XYZ
        )
        phi = _dihedral(XYZ[0], XYZ[2], XYZ[1], XYZ[3])
        assert math.isclose(e, 1.1 * (1 + math.cos(2 * phi - math.pi)), rel_tol=1e-12)

    def test_lj_cut_compiled_prices_one_row_per_pair(self) -> None:
        e, _ = LJCut.compiled([0], [3], [0.2], [3.1]).calc_energy_forces(XYZ)
        r = np.linalg.norm(XYZ[0] - XYZ[3])
        assert math.isclose(e, 4 * 0.2 * ((3.1 / r) ** 12 - (3.1 / r) ** 6), rel_tol=1e-12)

    def test_pair_coul_cut_prices_the_buffered_coulomb_law(self) -> None:
        e, _ = PairCoulCut([0], [3], [-0.12], coulomb=332.06371).calc_energy_forces(XYZ)
        r = np.linalg.norm(XYZ[0] - XYZ[3])
        assert math.isclose(e, 332.06371 * -0.12 / r, rel_tol=1e-12)

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="one entry per instance"):
            BondHarmonic([0, 1], [1], [300.0], [1.4])


class TestPotentialsAssembly:
    def test_push_moves_the_kernel(self) -> None:
        bond = BondHarmonic([0], [1], [300.0], [1.4])
        pots = Potentials()
        pots.push(bond)
        assert len(pots) == 1
        with pytest.raises(ValueError, match="moved"):
            pots.push(bond)

    def test_a_hand_assembled_collection_prices_what_the_compiled_field_prices(self) -> None:
        ff = molrs.ff.ForceField("chain", units="real")
        atoms = ff.def_style("atom", "full")
        c = atoms.def_type("C", mass=12.011)
        ff.def_style("bond", "harmonic").def_type("CC", c, c, k=300.0, r0=1.4)
        ff.def_style("angle", "harmonic").def_type("CCC", c, c, c, k=50.0, theta0=109.5)
        ff.def_style("dihedral", "periodic").def_type(
            "CCCC",
            c,
            c,
            c,
            c,
            k1=1.3,
            periodicity1=1.0,
            phase1=0.0,
            k2=0.4,
            periodicity2=2.0,
            phase2=180.0,
        )
        frame = molrs.Frame()
        block = molrs.Block()
        for d, key in enumerate("xyz"):
            block.insert(key, XYZ[:, d].copy())
        block.insert("type", ["C"] * 4)
        frame["atoms"] = block
        for name, rows, label in (
            ("bonds", [(0, 1), (1, 2), (2, 3)], "CC"),
            ("angles", [(0, 1, 2), (1, 2, 3)], "CCC"),
            ("dihedrals", [(0, 1, 2, 3)], "CCCC"),
        ):
            topo = molrs.Block()
            for column, atom in zip(("atomi", "atomj", "atomk", "atoml"), zip(*rows)):
                topo.insert(column, np.array(atom, dtype=np.uint32))
            topo.insert("type", [label] * len(rows))
            frame[name] = topo
        compiled = molrs.ff.PotentialCompiler(ff).compile(frame).calc_energy_forces(XYZ.reshape(-1))

        pots = Potentials()
        pots.push(BondHarmonic([0, 1, 2], [1, 2, 3], [300.0] * 3, [1.4] * 3))
        pots.push(AngleHarmonic([0, 1], [1, 2], [2, 3], [50.0] * 2, [109.5] * 2))
        pots.push(
            DihedralPeriodic(
                [0],
                [1],
                [2],
                [3],
                np.array([[1.3, 0.4]]),
                np.array([[1.0, 2.0]]),
                np.array([[0.0, 180.0]]),
            )
        )
        by_hand = pots.calc_energy_forces(XYZ.reshape(-1))
        assert math.isclose(by_hand[0], compiled[0], rel_tol=1e-12)
        np.testing.assert_allclose(by_hand[1], compiled[1], rtol=1e-12, atol=1e-12)
