//! The estimator's two constant tables — the lookups, not a second copy.
//!
//! Both tables are [`ff::params`](crate::ff::params) data: typed Rust `const`s
//! emitted by `scripts/gen_param_tables.py` and guarded by `MANIFEST.sha256`, the
//! same path `gaff.dat` and the seven `ATOMTYPE_*.DEF` take. Nothing here is
//! parsed at runtime; a malformed table is a **compile** error.
//!
//! - [`ParmchkTable`] ([`PARMCHK`]) — `PARMCHK.DAT`: parmchk2's atom-type
//!   substitution table (`EQUA` / `CORR` rows with their nine penalty columns),
//!   its `WEIGHT_*` / `DEFAULT_*` scalars, and the `improper_flag` column that
//!   says which types may be an improper centre.
//! - [`EmpiricalTable`] ([`EMPIRICAL_GAFF`] / [`EMPIRICAL_GAFF2`]) —
//!   `PARM_BLBA_GAFF*.DAT`: the Badger bond `ln(Kij)` per element pair and the
//!   angle `C` / `Z` factors per element (Wang et al. *J. Comput. Chem.* 2004,
//!   25:1157–1174, Eqs. 3 and 5).
//!
//! # One table, one type
//!
//! This module adds **inherent methods** to those two types rather than wrapping
//! them in views of its own. The wrapper is the thing to avoid: two structs
//! describing one table drift, and a reader has to learn which of them is the
//! table. `ff::params` owns each table's *shape* (its rows and columns, verbatim
//! from upstream); this module owns the *questions the estimator asks it*, which
//! are the estimator's business and not the table's.
//!
//! # Units
//!
//! Bond length Å, bond force constant kcal/mol/Å², angle force constant
//! kcal/mol/rad². The empirical angle formula consumes θ₀ in **radians**.

use molrs::Element;

use crate::ff::params::{
    EMPIRICAL_GAFF, EMPIRICAL_GAFF2, EmpiricalBondRow, EmpiricalTable, PARMCHK, ParmchkCorr,
    ParmchkPenalty, ParmchkTable, ParmchkType,
};

/// Which force field's empirical constants to use.
///
/// The two files differ (`PARM_BLBA_GAFF2.DAT` has 239 rows to GAFF's 146), so
/// the choice is a parameter rather than a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmpiricalSet {
    /// `PARM_BLBA_GAFF.DAT`.
    Gaff,
    /// `PARM_BLBA_GAFF2.DAT`.
    Gaff2,
}

impl EmpiricalSet {
    /// The compiled table this set names.
    pub fn table(self) -> EmpiricalTable {
        match self {
            Self::Gaff => EMPIRICAL_GAFF,
            Self::Gaff2 => EMPIRICAL_GAFF2,
        }
    }
}

/// The compiled `PARMCHK.DAT` — the one substitution table.
pub const fn substitution_table() -> ParmchkTable {
    PARMCHK
}

/// The questions the cascade asks the substitution table.
impl ParmchkTable {
    /// The block declaring `atom_type`, if the table knows it.
    pub fn entry(&self, atom_type: &str) -> Option<&'static ParmchkType> {
        self.get(atom_type)
    }

    /// Whether an atom of `atom_type` may be the CENTRE of an improper.
    ///
    /// The `improper_flag` column, and the reason benzene's `ca` carries a ring
    /// planarity term while methylamine's `n3` carries none.
    pub fn is_improper_centre(&self, atom_type: &str) -> bool {
        self.entry(atom_type).is_some_and(|t| t.improper)
    }

    /// The atomic number `PARMCHK.DAT` records for `atom_type`.
    pub fn atomic_number(&self, atom_type: &str) -> Option<u8> {
        self.entry(atom_type).map(|t| t.atomic_number)
    }

    /// The element symbol `PARMCHK.DAT` records for `atom_type`.
    pub fn element(&self, atom_type: &str) -> Option<&'static str> {
        let z = self.atomic_number(atom_type)?;
        Element::by_number(z).map(Element::symbol)
    }

    /// Are `a` and `b` **equivalent** types (an `EQUA` row, either way round)?
    ///
    /// Substituting one for the other costs nothing: `gaff2.dat`'s `ns` is `n`
    /// with a different name, so `X-c-n-X` covers an `ns` torsion exactly.
    pub fn equivalent(&self, a: &str, b: &str) -> bool {
        let one = |x: &str, y: &str| self.entry(x).is_some_and(|t| t.equivalent.contains(&y));
        one(a, b) || one(b, a)
    }

    /// The `CORR` row taking `from` to `to`, in that direction only.
    ///
    /// Direction is load-bearing: `PARMCHK.DAT` lists `c` → `c2` but **not**
    /// `c2` → `c`, and parmchk2 charges the *default* torsion penalty for the
    /// latter rather than reading the former backwards.
    pub fn correspondence(&self, from: &str, to: &str) -> Option<&'static ParmchkCorr> {
        self.entry(from)?
            .corresponding
            .iter()
            .find(|row| row.to == to)
    }

    /// Does `atom_type` take part in the substitution table at all?
    ///
    /// `c3` does not — it has no `CORR` row — and parmchk2 will not substitute it
    /// for anything, not even at the default penalty. That is what stops
    /// `c2-c2-ss-c3` from standing in for thiophene's `cc-cd-ss-cd`.
    pub fn substitutable(&self, atom_type: &str) -> bool {
        self.entry(atom_type)
            .is_some_and(|t| !t.corresponding.is_empty())
    }

    /// The overall similarity of substituting `from` with `to` (the 11th `CORR`
    /// column), in either direction. `0` for identical types.
    pub fn similarity(&self, from: &str, to: &str) -> Option<f64> {
        if from == to {
            return Some(0.0);
        }
        let column = ParmchkPenalty::Similarity;
        self.correspondence(from, to)
            .and_then(|row| row.get(column))
            .or_else(|| {
                self.correspondence(to, from)
                    .and_then(|row| row.get(column))
            })
    }
}

/// The questions the empirical fallback asks its constants.
///
/// The table is keyed by atomic number (that is what upstream writes); these
/// resolve an element **symbol** to one, so the estimator can stay in the
/// force field's own vocabulary.
impl EmpiricalTable {
    /// `ln(Kij)` for an unordered element pair, if tabulated.
    pub fn bond_ln_k(&self, e1: &str, e2: &str) -> Option<f64> {
        self.bond_pair(e1, e2).map(|row| row.ln_k)
    }

    /// The reference bond length of an element pair, in Å, if tabulated.
    pub fn bond_length(&self, e1: &str, e2: &str) -> Option<f64> {
        self.bond_pair(e1, e2).map(|row| row.r_ref)
    }

    /// The angle `C` factor of an element (the angle's CENTRE), if tabulated.
    pub fn angle_c(&self, element: &str) -> Option<f64> {
        self.angle(atomic_number(element)?).map(|row| row.c)
    }

    /// The angle `Z` factor of an element (an angle END atom), if tabulated.
    pub fn angle_z(&self, element: &str) -> Option<f64> {
        self.angle(atomic_number(element)?).map(|row| row.z_factor)
    }

    fn bond_pair(&self, e1: &str, e2: &str) -> Option<&'static EmpiricalBondRow> {
        self.bond(atomic_number(e1)?, atomic_number(e2)?)
    }
}

/// The atomic number of an element symbol, or `None` if it is not an element.
///
/// [`Element`] is `#[repr(u8)]` with the atomic number as its discriminant, so
/// the cast is the number rather than an ordinal that happens to agree with it.
fn atomic_number(symbol: &str) -> Option<u8> {
    Element::by_symbol(symbol).map(|e| e as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_table_carries_the_upstream_rows() {
        let t = EmpiricalSet::Gaff.table();
        assert!(
            (t.bond_power - 4.5).abs() < 1e-12,
            "Badger exponent m = 4.5"
        );
        // C-C (PARM_BLBA_GAFF.DAT `BL C 6 C 6 1.5260 7.6430`).
        assert!((t.bond_ln_k("C", "C").unwrap() - 7.643).abs() < 1e-9);
        assert!((t.bond_length("C", "C").unwrap() - 1.526).abs() < 1e-9);
        // The pair lookup is unordered.
        assert_eq!(t.bond_ln_k("H", "C"), t.bond_ln_k("C", "H"));
        assert!((t.bond_ln_k("C", "H").unwrap() - 6.217).abs() < 1e-9);
        // Angle Z / C factors.
        assert!((t.angle_z("C").unwrap() - 1.183).abs() < 1e-9);
        assert!((t.angle_c("C").unwrap() - 1.339).abs() < 1e-9);
        assert!((t.angle_z("H").unwrap() - 0.784).abs() < 1e-9);
        // An element upstream does not tabulate.
        assert_eq!(t.angle_z("He"), None);
    }

    #[test]
    fn the_two_empirical_sets_are_different_tables() {
        assert_ne!(
            EmpiricalSet::Gaff.table().bonds.len(),
            EmpiricalSet::Gaff2.table().bonds.len(),
            "GAFF2 tabulates more element pairs than GAFF"
        );
    }

    #[test]
    fn substitution_table_carries_weights_defaults_and_corr() {
        let t = substitution_table();
        let w = t.weights;
        assert!((w.weight_angle_centre - 10.0).abs() < 1e-12);
        assert!((w.weight_torsion_centre - 10.0).abs() < 1e-12);
        assert!((w.weight_improper - 10.0).abs() < 1e-12);
        assert!((w.weight_wildcard - 10.0).abs() < 1e-12);
        assert!((w.default_bond_length - 20.0).abs() < 1e-12);
        assert!((w.default_torsion - 87.0).abs() < 1e-12);

        // os -> oh is a low-penalty correspondence (ether <-> hydroxyl oxygen).
        let row = t.correspondence("os", "oh").expect("os -> oh is tabulated");
        let bond = row
            .get(ParmchkPenalty::BondLength)
            .expect("a bond-length penalty");
        assert!((0.0..5.0).contains(&bond), "os -> oh low bond penalty");
    }

    #[test]
    fn the_improper_flag_separates_planar_types_from_sp3_ones() {
        let t = substitution_table();
        for planar in ["ca", "c", "c2", "n", "na", "nh", "no", "cc", "cd"] {
            assert!(t.is_improper_centre(planar), "{planar} is planar");
        }
        for tetrahedral in ["c3", "n3", "n4", "os", "oh", "hc", "ha"] {
            assert!(
                !t.is_improper_centre(tetrahedral),
                "{tetrahedral} carries no improper"
            );
        }
    }

    #[test]
    fn correspondence_is_directional_but_similarity_is_not() {
        let t = substitution_table();
        // PARMCHK.DAT lists c -> c2, and NOT c2 -> c.
        assert!(t.correspondence("c", "c2").is_some());
        assert!(t.correspondence("c2", "c").is_none());
        // Similarity reads either way round, so both resolve.
        assert_eq!(t.similarity("c", "c2"), t.similarity("c2", "c"));
        assert_eq!(t.similarity("c3", "c3"), Some(0.0));
        // c3 has no CORR row at all: it is never substituted.
        assert!(!t.substitutable("c3"));
        assert!(t.substitutable("c2"));
    }

    #[test]
    fn equivalent_types_are_free_to_swap() {
        let t = substitution_table();
        // gaff2's `ns` is `n` under another name; `c5` is `c3`.
        assert!(t.equivalent("ns", "n"));
        assert!(t.equivalent("n", "ns"));
        assert!(t.equivalent("c5", "c3"));
        assert!(!t.equivalent("c3", "c2"));
    }

    #[test]
    fn the_table_knows_each_types_element() {
        let t = substitution_table();
        assert_eq!(t.element("c3"), Some("C"));
        assert_eq!(t.element("os"), Some("O"));
        assert_eq!(t.element("br"), Some("Br"));
        assert_eq!(t.element("opls_135"), None, "not a GAFF type");
    }
}
