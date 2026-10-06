pub mod charge;
pub(crate) mod constants;
#[cfg(test)]
mod convention_invariance;
pub mod forcefield;
pub mod mmff;
pub mod params;
pub mod potential;
pub mod scale_lj;
pub mod typifier;

// Common API re-exports so callers don't have to spell the deep module path.
pub use charge::{BccModel, BccParameterSet, ChargeError, ChargeModel, MullikenModel};
pub use forcefield::readers::{
    ForceFieldReader,
    gromacs::GromacsTopFfReader,
    lammps::LammpsFfReader,
    opls::OplsXmlReader,
    prmtop::{AmberPrmtopFfReader, read_amber_prmtop_ff},
};
pub use forcefield::writers::{
    ForceFieldWriter,
    frcmod::{AmberFrcmodFfWriter, write_amber_frcmod, write_amber_frcmod_str},
    gromacs::GromacsTopFfWriter,
    lammps::{LammpsFfWriter, LammpsWriteOptions},
    xml::{XmlForceFieldWriter, write_forcefield_xml, write_forcefield_xml_str},
};
pub use forcefield::xml::{read_forcefield_xml, read_forcefield_xml_str};
pub use forcefield::{ForceField, SpecialBonds};
pub use scale_lj::{FragmentAtoms, FragmentScaling, ScaleLjError, compute_k_ij, scale_lj};
pub use typifier::gaff::GaffParameterSet;
