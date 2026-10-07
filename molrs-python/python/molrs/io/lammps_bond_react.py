"""LAMMPS ``fix bond/react`` templates — ``molrs::io::data::lammps_bond_react``.

:class:`BondReactTemplate` is one reaction's pre/post template pair;
:func:`molrs.io.write_lammps_bond_react_system` and
:func:`molrs.io.write_bond_react_map` write the file set LAMMPS reads.
"""

from .._lib import BondReactTemplate

__all__ = ["BondReactTemplate"]
