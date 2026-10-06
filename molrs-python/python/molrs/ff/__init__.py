"""Force fields — ``molrs::ff``.

One subpackage per Rust submodule, so the Python path and the Rust path are the
same word:

* :class:`ForceField` with its ``Style`` / ``Type`` handles and the
  force-field file readers and writers — native, re-exported here
* :mod:`~molrs.ff.typifier` — the subclassable ``Typifier`` base and its
  ``Match``, plus the graph-in / graph-out atom typers (OPLS-AA, MMFF94,
  MMFF94s, ATD)
* :mod:`~molrs.ff.charge` — partial-charge models (AM1-BCC / ABCG2, Mulliken,
  Gasteiger)
* :mod:`~molrs.ff.potential` — the parameter interface of the compiled kernels

The names below are re-exported here because they are the force-field surface
callers reach for; the submodule path stays available when you need to say
*which* concern a name belongs to.
"""

from __future__ import annotations

from .._lib import (
    AMBER_COULOMB as AMBER_COULOMB,
)
from .._lib import (
    AMBER_SCEE as AMBER_SCEE,
)
from .._lib import (
    AMBER_SCNB as AMBER_SCNB,
)
from .._lib import (
    AngleStyle,
    AngleType,
    AtomStyle,
    AtomType,
    BondStyle,
    BondType,
    CmapStyle,
    CmapType,
    DihedralStyle,
    DihedralType,
    ForceField,
    ImproperStyle,
    ImproperType,
    PairStyle,
    PairType,
    Style,
    Type,
    read_amber_prmtop_ff,
    read_forcefield_xml,
    read_gromacs_top_ff,
    read_lammps_data_coeffs,
    read_lammps_forcefield,
    read_opls_xml,
    write_amber_frcmod,
    write_forcefield_xml,
    write_gromacs_top_ff,
    write_lammps_data_coeffs,
    write_lammps_forcefield,
    write_lammps_forcefield_str,
)
from .._lib import (
    FragmentScaling as FragmentScaling,
)
from .._lib import (
    PotentialCompiler as PotentialCompiler,
)
from .._lib import (
    Potentials as Potentials,
)
from .._lib import (
    compute_k_ij as compute_k_ij,
)
from .._lib import (
    fragment_scaling_data as fragment_scaling_data,
)
from .._lib import (
    intramolecular_pairs as intramolecular_pairs,
)
from .._lib import (
    scale_lj as scale_lj,
)
from . import charge, potential, typifier
from .charge import BccModel, GasteigerModel, MullikenModel
from .potential import Potential
from .typifier import (
    AtdTypifier,
    Match,
    MMFF94STypifier,
    MMFF94Typifier,
    OPLSAATypifier,
    Typifier,
)

__all__ = [
    "AMBER_COULOMB",
    "AMBER_SCEE",
    "AMBER_SCNB",
    "AngleStyle",
    "AngleType",
    "AtdTypifier",
    "AtomStyle",
    "AtomType",
    # charge models
    "BccModel",
    "BondStyle",
    "BondType",
    "CmapStyle",
    "CmapType",
    "DihedralStyle",
    "DihedralType",
    # force field + its handle views
    "ForceField",
    "FragmentScaling",
    "GasteigerModel",
    "ImproperStyle",
    "ImproperType",
    "MMFF94STypifier",
    "MMFF94Typifier",
    "Match",
    "MullikenModel",
    "OPLSAATypifier",
    "PairStyle",
    "PairType",
    "Potential",
    "PotentialCompiler",
    "Potentials",
    "Style",
    "Type",
    # typifiers
    "Typifier",
    # subpackages
    "charge",
    "compute_k_ij",
    "fragment_scaling_data",
    # pair helpers + polarizable fragment scaling
    "intramolecular_pairs",
    "potential",
    "read_amber_prmtop_ff",
    # force-field file formats
    "read_forcefield_xml",
    "read_gromacs_top_ff",
    "read_lammps_data_coeffs",
    "read_lammps_forcefield",
    "read_opls_xml",
    "scale_lj",
    "typifier",
    "write_amber_frcmod",
    "write_forcefield_xml",
    "write_gromacs_top_ff",
    "write_lammps_data_coeffs",
    "write_lammps_forcefield",
    "write_lammps_forcefield_str",
]
