//! Force-field file formats: a file mapped to and from a
//! [`ForceField`](crate::ff::forcefield::ForceField), the data model `ff`
//! owns.
//!
//! Every file reader and writer is `io`'s — structure, trajectory and
//! force-field files alike; [`crate::ff`] holds the data model and never
//! reads a file. A reader owns the translation from a foreign format —
//! element and attribute names, **and unit and factor normalization** — into
//! the force-field IR (adopts the LAMMPS standard); the matching writer owns
//! the inverse, so unit conversion stays at one boundary pair.
//!
//! | module | its one job |
//! |---|---|
//! | [`readers`] | external force-field files in: LAMMPS `*.ff` / data coefficients / CMAP grids, GROMACS `.top`/`.itp`, AMBER prmtop, OPLS / OpenMM XML, CL&Pol `alpha.ff` |
//! | [`writers`] | a force field out: LAMMPS, GROMACS `.top`, AMBER frcmod, molrs XML |
//! | [`xml`] | molrs's own XML schema in, and the typing-metadata halves of an OPLS-AA / MMFF XML a typifier is built from |
//! | [`lammps_units`] | a LAMMPS `units` token mapped onto a unit preset, for the LAMMPS reader and writer |

pub mod lammps_units;
pub mod readers;
pub mod writers;
pub mod xml;
