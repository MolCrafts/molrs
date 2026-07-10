//! Embedded data files shipped with molrs-core.
//!
//! Canonical force-field parameter sets are compiled into the binary so callers
//! never need to locate a file on disk.

/// MMFF94 parameter XML (embedded at compile time).
pub const MMFF94_XML: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/mmff94.xml"));

/// MMFF94s parameter XML (embedded at compile time).
pub const MMFF94S_XML: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/mmff94s.xml"));

/// Canonical OPLS-AA parameter XML (embedded at compile time).
///
/// This is the durable, committed-in-molrs copy that backs
/// `OPLSAATypifier::oplsaa` — the
/// OPLS typifier ships standalone and does not depend on any external file.
///
/// Aromatic ring carbons / hydrogens use lowercase SMARTS `c` (RDKit-faithful
/// aromatic matching), so the molrs SMARTS engine perceives benzene-type rings
/// as aromatic and types them exactly as molpy's ground truth. Aliphatic
/// (`[C;X4]` / `[C;X3]`) chain carbons, nitrile (`[C;X2]`) carbons, and ring
/// nitrogens keep uppercase atomic-number SMARTS.
pub const OPLSAA_XML: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/oplsaa.xml"));

/// Embedded Antechamber AM1-BCC bond-charge correction table.
///
/// Consumed by the AM1-BCC typifier as native table data.
pub const ANTECHAMBER_BCCPARM_DAT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/antechamber/BCCPARM.DAT"
));

/// Embedded Antechamber ABCG2 bond-charge correction table.
pub const ANTECHAMBER_BCCPARM_ABCG2_DAT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/antechamber/BCCPARM_ABCG2.DAT"
));

/// Embedded Antechamber AM1-BCC atom type definition rules.
pub const ANTECHAMBER_ATOMTYPE_BCC_DEF: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/antechamber/ATOMTYPE_BCC.DEF"
));

/// Embedded Antechamber ABCG2 atom type definition rules.
pub const ANTECHAMBER_ATOMTYPE_ABCG2_DEF: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/antechamber/ATOMTYPE_ABCG2.DEF"
));
