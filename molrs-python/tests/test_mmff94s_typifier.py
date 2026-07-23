"""The two MMFF front doors on the Python surface (spec ``mmff-typifier-split``, ac-007).

RED until ``molrs.MMFF94Typifier`` / ``molrs.MMFF94STypifier`` are registered and the
Rust ``frame_builder`` stops hardcoding ``MmffVariant::Mmff94``. Today ``molrs`` exposes
a single ``MMFFTypifier``, and MMFF94s is unreachable from Python entirely.

Why the energies are the discriminator: MMFF94 and MMFF94s share all 95 atom types and
every bond / angle / stretch-bend / vdW / charge row. They differ in 11 out-of-plane rows
and 42 torsion rows, all centred on delocalised trivalent nitrogen (MMFF types 10 ``NC=O``
and 40 ``NC=C``). So an atom-type assertion would pass vacuously; ``koop`` and the total
energy would not.

Molecules come from the RDKit-generated fixtures the Rust side already validates against
(``tests/fixtures/mmff``) — never hand-built coordinates.
"""

from pathlib import Path

import numpy as np
import pytest

import molrs

# --------------------------------------------------------------------------------------
# Fixtures — the RDKit SDF conformers, shared with the Rust suite.
# --------------------------------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mmff"

#: Amide nitrogen (MMFF type 10) — MMFF94 pyramidalises it, MMFF94s flattens it.
NMA = "s_nmethylacetamide"
#: No trivalent nitrogen at all: nothing may differ between the two variants.
ETHANE = "e_ethane"

#: kcal/mol. "These are not the same force field."
ENERGY_GAP = 1e-3


def _load_sdf(name: str) -> "molrs.Atomistic":
    """Minimal V2000 reader preserving atom order and bond orders."""
    path = FIXTURES / f"{name}.sdf"
    assert path.is_file(), f"missing RDKit fixture {path}"
    lines = path.read_text().splitlines()
    n_atoms = int(lines[3][0:3])
    n_bonds = int(lines[3][3:6])

    mol = molrs.Atomistic()
    ids = []
    for k in range(n_atoms):
        line = lines[4 + k]
        x, y, z = float(line[0:10]), float(line[10:20]), float(line[20:30])
        ids.append(mol.add_atom(line[31:34].strip(), x, y, z))
    for k in range(n_bonds):
        line = lines[4 + n_atoms + k]
        i, j = int(line[0:3]) - 1, int(line[3:6]) - 1
        order = float(line[6:9])
        handle = mol.add_bond(ids[i], ids[j])
        mol.set_bond_order(handle, order)
    return mol


def _energy(typifier, mol) -> float:
    """Total energy through the generic path: typify -> Frame -> compile -> evaluate.

    The docstring was already true; the code was not. It called ``typifier.build(mol)``,
    the one-step shortcut that hid the Frame — and with it, until mmff-orthogonal-01,
    an entire missing electrostatic term. mmff-orthogonal-02 deleted the shortcut, so
    the three steps it folded together are now written out. The assertions below are
    unchanged.
    """
    frame = typifier.typify(mol).to_frame()
    frame["pairs"] = molrs.intramolecular_pairs(frame)
    potentials = typifier.forcefield().to_potentials(frame)
    return float(potentials.calc_energy(frame))


def _koop(typifier, mol) -> np.ndarray:
    """The out-of-plane force constants baked onto the typed frame (md*A*rad^-2)."""
    frame = typifier.typify(mol).to_frame()
    return np.asarray(frame["impropers"]["koop"])


def _atom_types(typifier, mol) -> list[str]:
    frame = typifier.typify(mol).to_frame()
    return list(frame["atoms"]["type"])


# --------------------------------------------------------------------------------------
# Surface
# --------------------------------------------------------------------------------------


def test_both_front_doors_are_importable():
    from molrs import MMFF94STypifier, MMFF94Typifier  # noqa: F401


def test_each_front_door_names_its_own_force_field():
    assert molrs.MMFF94Typifier().forcefield().name == "MMFF94"
    assert molrs.MMFF94STypifier().forcefield().name == "MMFF94s"


def test_the_old_name_is_gone():
    """A clean break: renamed, not aliased (owner's ruling; the repo is pre-1.0)."""
    with pytest.raises(AttributeError):
        getattr(molrs, "MMFFTypifier")


# --------------------------------------------------------------------------------------
# Discrimination — the two doors are two force fields
# --------------------------------------------------------------------------------------


def test_mmff94s_bakes_a_different_koop_on_the_amide_nitrogen():
    """Path 1: the ``koop`` column the out-of-plane kernel actually reads.

    ``E_oop = 0.5 * 143.9325 * koop * chi**2`` with chi in radians, so the sign of
    ``koop`` decides whether the planar nitrogen is a minimum (MMFF94s) or a maximum
    (MMFF94). N-methylacetamide's amide N is MMFF type 10: MMFF94 gives it -0.020,
    MMFF94s +0.015.
    """
    mol = _load_sdf(NMA)
    k94 = _koop(molrs.MMFF94Typifier(), mol)
    k94s = _koop(molrs.MMFF94STypifier(), mol)

    assert k94.shape == k94s.shape
    assert np.any(np.abs(k94 - k94s) > 1e-12), (
        "every improper carries the same koop under both variants — "
        "MMFF94STypifier is silently emitting MMFF94 parameters"
    )
    # The amide-N rows are the ones that move, and they move to +0.015.
    moved = k94s[np.abs(k94 - k94s) > 1e-12]
    assert np.allclose(moved, 0.015, atol=1e-12)
    assert np.all(moved > 0.0), "MMFF94s must make planar amide N an energy minimum"


def test_total_energies_differ_on_n_methylacetamide():
    mol = _load_sdf(NMA)
    e94 = _energy(molrs.MMFF94Typifier(), mol)
    e94s = _energy(molrs.MMFF94STypifier(), mol)
    assert np.isfinite(e94) and np.isfinite(e94s)
    assert abs(e94 - e94s) > ENERGY_GAP, (
        f"MMFF94 ({e94}) and MMFF94s ({e94s}) agree to {ENERGY_GAP} kcal/mol on a molecule "
        "whose amide nitrogen is re-parameterised — MMFF94s is not reaching the potentials"
    )


def test_atom_types_are_identical_so_a_type_only_test_would_be_vacuous():
    """Recorded as an assertion, not just prose: the 95 atom types are shared."""
    mol = _load_sdf(NMA)
    assert _atom_types(molrs.MMFF94Typifier(), mol) == _atom_types(molrs.MMFF94STypifier(), mol)


# --------------------------------------------------------------------------------------
# Identity guard — without it, "they differ" could pass on a difference that isn't real
# --------------------------------------------------------------------------------------


def test_total_energies_are_identical_on_ethane():
    mol = _load_sdf(ETHANE)
    e94 = _energy(molrs.MMFF94Typifier(), mol)
    e94s = _energy(molrs.MMFF94STypifier(), mol)
    assert e94 == e94s, (
        f"ethane has no delocalised nitrogen: MMFF94 ({e94}) and MMFF94s ({e94s}) must be "
        "exactly equal"
    )


def test_typed_frames_are_identical_on_ethane():
    mol = _load_sdf(ETHANE)
    f94 = molrs.MMFF94Typifier().typify(mol).to_frame()
    f94s = molrs.MMFF94STypifier().typify(mol).to_frame()

    assert sorted(f94.keys()) == sorted(f94s.keys())
    for block in f94.keys():
        b94, b94s = f94[block], f94s[block]
        assert sorted(b94.keys()) == sorted(b94s.keys()), f"{block}: column sets differ"
        for column in b94.keys():
            left = np.asarray(b94[column])
            right = np.asarray(b94s[column])
            assert np.array_equal(left, right), f"{block}/{column} differs between variants"
