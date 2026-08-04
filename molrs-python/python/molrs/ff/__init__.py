"""Force fields — ``molrs::ff``.

One subpackage per Rust submodule, so the Python path and the Rust path are the
same word:

* :mod:`~molrs.ff.forcefield` — the chainable :class:`ForceField` plus its
  ``Style`` / ``Type`` handle views
* :mod:`~molrs.ff.typifier` — graph-in / graph-out atom typers (OPLS-AA, MMFF94,
  MMFF94s, ATD)
* :mod:`~molrs.ff.charge` — partial-charge models (AM1-BCC / ABCG2, Mulliken,
  Gasteiger)
* :mod:`~molrs.ff.potential` — the parameter interface of the compiled kernels

The names below are re-exported here because they are the force-field surface
callers reach for; the submodule path stays available when you need to say
*which* concern a name belongs to.
"""

from __future__ import annotations

from . import charge, forcefield, potential, typifier
from .._lib import (
    FragmentScaling as FragmentScaling,
    Potentials as Potentials,
    compute_k_ij as compute_k_ij,
    extract_coords as extract_coords,
    fragment_scaling_data as fragment_scaling_data,
    intramolecular_pairs as intramolecular_pairs,
    scale_lj as scale_lj,
)
from .charge import BccModel, GasteigerModel, MullikenModel
from .forcefield import (
    AngleHarmonicStyle,
    AngleStyle,
    AngleType,
    AtomStyle,
    AtomType,
    BondHarmonicStyle,
    BondStyle,
    BondType,
    DihedralOPLSStyle,
    DihedralStyle,
    DihedralType,
    ForceField,
    ImproperStyle,
    ImproperType,
    PairCoulLongStyle,
    PairStyle,
    PairType,
    Parameters,
    Style,
    Type,
    read_forcefield_xml,
    read_forcefield_xml_str,
    read_lammps_forcefield,
    read_lammps_forcefield_str,
    read_opls_xml,
    read_opls_xml_str,
    write_lammps_forcefield,
    write_lammps_forcefield_str,
)
from .typifier import (
    AtdTypifier,
    MMFF94STypifier,
    MMFF94Typifier,
    OPLSAATypifier,
    Typifier,
)

__all__ = [
    # subpackages
    "charge",
    "forcefield",
    "potential",
    "typifier",
    # force field + its handle views
    "ForceField",
    "Style",
    "AtomStyle",
    "BondStyle",
    "AngleStyle",
    "DihedralStyle",
    "ImproperStyle",
    "PairStyle",
    "BondHarmonicStyle",
    "AngleHarmonicStyle",
    "DihedralOPLSStyle",
    "PairCoulLongStyle",
    "Type",
    "AtomType",
    "BondType",
    "AngleType",
    "DihedralType",
    "ImproperType",
    "PairType",
    "Parameters",
    "Potentials",
    # force-field file formats
    "read_forcefield_xml",
    "read_forcefield_xml_str",
    "read_opls_xml",
    "read_opls_xml_str",
    "read_lammps_forcefield",
    "read_lammps_forcefield_str",
    "write_lammps_forcefield",
    "write_lammps_forcefield_str",
    # typifiers
    "Typifier",
    "OPLSAATypifier",
    "MMFF94Typifier",
    "MMFF94STypifier",
    "AtdTypifier",
    # charge models
    "BccModel",
    "MullikenModel",
    "GasteigerModel",
    # pair helpers + polarizable fragment scaling
    "intramolecular_pairs",
    "extract_coords",
    "scale_lj",
    "FragmentScaling",
    "compute_k_ij",
    "fragment_scaling_data",
]
