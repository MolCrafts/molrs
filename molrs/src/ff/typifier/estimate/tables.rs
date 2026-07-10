//! Compile-time constant tables for the [`ParameterEstimator`](super::ParameterEstimator).
//!
//! Two authoritative GAFF / parmchk2 data assets are embedded at build time
//! (mirroring [`molrs::data::MMFF94_XML`](crate::core::data::MMFF94_XML)):
//!
//! - [`GAFF_EMPIRICAL_JSON`] — `molrs/data/gaff_empirical.json`: the Badger
//!   bond-`k` per-element-pair `ln Kij` table, the GAFF angle `Z`/`C` per-element
//!   factors, and the `143.9` / `m = 4.5` constants. Transcribed verbatim from
//!   embedded reference `dat/antechamber/PARM_BLBA_GAFF.DAT` and Wang et al.
//!   *J. Comput. Chem.* 2004, 25:1157–1174 (Eqs. 3, 5, 6; Tables 3, 4).
//! - [`GAFF_EQUIV_JSON`] — `molrs/data/gaff_equiv.json`: the parmchk2 equivalent
//!   (`EQUA`) / corresponding (`CORR`) atom-type substitution table with per-row
//!   penalties + the global penalty weights / defaults. Transcribed verbatim from
//!   embedded reference `dat/antechamber/PARMCHK.DAT`.
//!
//! # Units
//!
//! All embedded values are in molrs internal units: bond length Å, bond force
//! constant kcal/mol/Å², angle force constant kcal/mol/rad², angle θ₀ stored in
//! the table in **degrees** as θ_eq but the empirical formula consumes radians
//! (see [`empirical_angle_k`](super::ParameterEstimator::estimate_angle)).

use std::collections::HashMap;

use serde::Deserialize;

/// GAFF empirical bond / angle constant table (embedded at compile time).
pub const GAFF_EMPIRICAL_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/gaff_empirical.json"
));

/// parmchk2 equivalent / corresponding substitution table (embedded at compile
/// time).
pub const GAFF_EQUIV_JSON: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/gaff_equiv.json"));

// ---------------------------------------------------------------------------
// Empirical bond / angle tables (gaff_empirical.json)
// ---------------------------------------------------------------------------

/// One row of the Badger bond-`k` table: element pair + `ln(Kij)` coefficient.
#[derive(Debug, Clone, Deserialize)]
pub struct BondLnK {
    pub e1: String,
    pub e2: String,
    /// Reference (equilibrium) bond length in Å (Wang2004 Table 3 `rref`).
    pub rref: f64,
    /// `ln(Kij)` coefficient (Wang2004 Table 3 `ln Kij`); `Kij = exp(ln_kij)`.
    pub ln_kij: f64,
}

/// One row of the GAFF angle `Z`/`C` factor table (per element).
#[derive(Debug, Clone, Deserialize)]
pub struct AngleZC {
    pub e: String,
    /// `C` factor (used when the element is the angle *centre* atom).
    pub c: f64,
    /// `Z` factor (used when the element is an angle *end* atom).
    pub z: f64,
}

/// Parsed `gaff_empirical.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct EmpiricalTable {
    /// Power-law exponent `m` in `K_r = exp(ln_kij)/r^m` (Wang2004 Eq.3, m=4.5).
    pub bond_power_m: f64,
    pub bond_lnk: Vec<BondLnK>,
    pub angle_zc: Vec<AngleZC>,
}

impl EmpiricalTable {
    /// Parse the embedded empirical-constant table.
    pub fn load() -> Self {
        serde_json::from_str(GAFF_EMPIRICAL_JSON)
            .expect("embedded gaff_empirical.json is well-formed")
    }

    /// `ln(Kij)` for an (unordered) element pair, if tabulated.
    pub fn bond_lnk(&self, e1: &str, e2: &str) -> Option<f64> {
        self.bond_lnk
            .iter()
            .find(|r| (r.e1 == e1 && r.e2 == e2) || (r.e1 == e2 && r.e2 == e1))
            .map(|r| r.ln_kij)
    }

    /// The angle `C` factor for an element (centre atom), if tabulated.
    pub fn angle_c(&self, e: &str) -> Option<f64> {
        self.angle_zc.iter().find(|r| r.e == e).map(|r| r.c)
    }

    /// The angle `Z` factor for an element (end atom), if tabulated.
    pub fn angle_z(&self, e: &str) -> Option<f64> {
        self.angle_zc.iter().find(|r| r.e == e).map(|r| r.z)
    }
}

// ---------------------------------------------------------------------------
// Substitution / equivalence table (gaff_equiv.json)
// ---------------------------------------------------------------------------

/// One corresponding-type substitution row: target type + per-arity penalty.
#[derive(Debug, Clone, Deserialize)]
pub struct CorrRow {
    /// The corresponding atom type this row maps *to*.
    pub to: String,
    /// Bond-length substitution penalty (`-1` ⇒ use the bond default).
    pub bond: f64,
    /// Angle (end-atom) substitution penalty (`-1` ⇒ use the angle default).
    pub angle: f64,
    /// Angle-centre substitution penalty (`-1` ⇒ use the angle-centre default).
    pub angle_ctr: f64,
    /// Torsion substitution penalty (`-1` ⇒ use the torsion default).
    pub torsion: f64,
}

/// One PARM block: equivalent types (penalty 0) + corresponding types.
#[derive(Debug, Clone, Deserialize)]
pub struct TypeEntry {
    /// Equivalent atom types — resonance / geometric twins, penalty 0.
    #[serde(default)]
    pub equa: Vec<String>,
    /// Corresponding atom types with per-arity substitution penalties.
    #[serde(default)]
    pub corr: Vec<CorrRow>,
}

/// Global penalty weights (PARMCHK.DAT `WEIGHT_*`).
#[derive(Debug, Clone, Deserialize)]
pub struct Weights {
    pub bond: f64,
    pub angle: f64,
    pub torsion: f64,
    /// Extra penalty for substituting through an *equivalent* type.
    pub equtype: f64,
    /// Extra penalty for crossing an atom-type *group* boundary.
    pub group: f64,
    /// Inner-atom (angle centre) multiplier — CGenFF inner-atom ×10.
    pub angle_center_mult: f64,
    /// Inner-atom (dihedral inner two) multiplier — CGenFF inner-atom ×10.
    pub torsion_center_mult: f64,
}

/// Per-arity default penalties (PARMCHK.DAT `DEFAULT_*`), applied when a row's
/// penalty is `-1`.
#[derive(Debug, Clone, Deserialize)]
pub struct Defaults {
    pub bond: f64,
    pub angle: f64,
    pub angle_ctr: f64,
    pub torsion: f64,
    pub torsion_ctr: f64,
}

/// Parsed `gaff_equiv.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct EquivTable {
    pub weights: Weights,
    pub defaults: Defaults,
    pub types: HashMap<String, TypeEntry>,
}

impl EquivTable {
    /// Parse the embedded substitution table.
    pub fn load() -> Self {
        serde_json::from_str(GAFF_EQUIV_JSON).expect("embedded gaff_equiv.json is well-formed")
    }

    /// The substitution entry for `atom_type`, if any.
    pub fn entry(&self, atom_type: &str) -> Option<&TypeEntry> {
        self.types.get(atom_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_table_loads_and_has_known_rows() {
        let t = EmpiricalTable::load();
        assert!(
            (t.bond_power_m - 4.5).abs() < 1e-12,
            "Badger exponent m = 4.5"
        );
        // C-C row (Wang2004 Table 3 / PARM_BLBA_GAFF.DAT).
        assert!((t.bond_lnk("C", "C").unwrap() - 7.643).abs() < 1e-9);
        // unordered lookup.
        assert_eq!(t.bond_lnk("H", "C"), t.bond_lnk("C", "H"));
        assert!((t.bond_lnk("C", "H").unwrap() - 6.217).abs() < 1e-9);
        // angle Z/C factors.
        assert!((t.angle_z("C").unwrap() - 1.183).abs() < 1e-9);
        assert!((t.angle_c("C").unwrap() - 1.339).abs() < 1e-9);
        assert!((t.angle_z("H").unwrap() - 0.784).abs() < 1e-9);
    }

    #[test]
    fn equiv_table_loads_with_weights_defaults_and_corr() {
        let t = EquivTable::load();
        assert!((t.weights.angle_center_mult - 10.0).abs() < 1e-12);
        assert!((t.weights.torsion_center_mult - 10.0).abs() < 1e-12);
        assert!((t.defaults.bond - 20.0).abs() < 1e-12);
        // os ↔ oh is a low-penalty correspondence (ether ↔ hydroxyl O).
        let os = t.entry("os").expect("os present");
        let oh_corr = os.corr.iter().find(|r| r.to == "oh").expect("os→oh");
        assert!(
            oh_corr.bond > 0.0 && oh_corr.bond < 5.0,
            "os→oh low bond penalty"
        );
    }
}
