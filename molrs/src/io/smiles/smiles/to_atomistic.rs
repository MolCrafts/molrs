//! Convert [`SmilesIR`] into [`Atomistic`] molecular graphs.
//!
//! This is the second stage of the pipeline:
//!
//! ```text
//! SMILES string → parse_smiles() → SmilesIR → to_atomistic() → Atomistic
//! ```
//!
//! The conversion walks the IR tree, creates atoms with element symbols, creates
//! bonds from the chain structure and ring closures, and sets properties
//! (charge, isotope, chirality, hydrogen count).

use std::collections::{HashMap, HashSet};

use crate::io::smiles::chem::ast::*;
use crate::io::smiles::error::{SmilesError, SmilesErrorKind};
use crate::io::smiles::smiles::canonical_element_symbol;
use molrs::system::atomistic::{AtomId, Atomistic};
use molrs::system::bond::{BondNumber, BondType};
use molrs::system::molgraph::PropValue;

/// Convert a parsed SMILES IR into an [`Atomistic`] molecular graph.
///
/// This resolves ring closures into bonds, sets atom properties (charge,
/// isotope, chirality), and records bond orders. Implicit hydrogens are
/// **not** added — call [`add_hydrogens`](crate::perceive::hydrogens::add_hydrogens)
/// separately if needed.
///
/// # Aromaticity
///
/// Lowercase symbols are *notation*, not element symbols: the `element`
/// component always holds the canonical symbol (`c` → `"C"`), and the
/// declared aromaticity is carried by the project-wide markers instead —
/// `is_aromatic = 1` on the atom, and `bond_type = Aromatic` on every bond the
/// notation declares aromatic. Its localized `bond_number` is left `Unknown`:
/// the notation declares delocalization, not a Kekulé phase, and kekulization
/// is what picks one. A bond written *without* a
/// symbol between two aromatic atoms is aromatic (the Daylight rule); an
/// explicit `-` between two aromatic atoms (biphenyl) is not.
///
/// # Hydrogen counts
///
/// A bracket atom states its hydrogen count exactly, so every bracket atom
/// gets an `h_count` component — `0` when the notation omits it. Organic-subset
/// atoms get none and are left to valence-based
/// [`add_hydrogens`](crate::perceive::hydrogens::add_hydrogens).
///
/// # Errors
///
/// Returns an error if ring closures are unmatched or if the IR contains
/// SMARTS query atoms (which have no single atomistic interpretation).
///
/// # Examples
///
/// ```
/// use molrs::io::smiles::{parse_smiles, to_atomistic};
///
/// let ir = parse_smiles("C(=O)O").unwrap();
/// let mol = to_atomistic(&ir).unwrap();
/// assert_eq!(mol.n_atoms(), 3);
/// assert_eq!(mol.n_bonds(), 2);
/// ```
pub fn to_atomistic(ir: &SmilesIR) -> Result<Atomistic, SmilesError> {
    let mut builder = Builder::new(ir);

    for component in &ir.components {
        builder.build_chain(component, None)?;
    }

    builder.close_rings()?;

    Ok(builder.mol)
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Pending ring closure: the atom that opened it and the optional bond kind.
struct PendingRing {
    atom: AtomId,
    bond: Option<BondKind>,
    span: Span,
}

/// Reduce a SMARTS-style [`BondQuery`] back to a single [`BondKind`]. The
/// SMILES → atomistic pipeline cannot represent SMARTS logical bond
/// operators (there's no single concrete order for `!=` or `-,=`), so a
/// query that isn't a plain `Kind(_)` is rejected with a clean error.
fn bond_query_to_kind(q: Option<&BondQuery>) -> Result<Option<BondKind>, SmilesError> {
    match q {
        None => Ok(None),
        Some(BondQuery::Kind(k)) => Ok(Some(*k)),
        Some(_) => Err(SmilesError::new(
            SmilesErrorKind::InvalidQueryPrimitive(
                "SMARTS bond query cannot be atomized".to_owned(),
            ),
            crate::io::smiles::chem::ast::Span::new(0, 0),
            "",
        )),
    }
}

struct Builder<'a> {
    mol: Atomistic,
    /// Maps ring numbers to pending (unmatched) ring-closure openers.
    open_rings: HashMap<u16, PendingRing>,
    /// Atoms the notation declared aromatic (lowercase symbol). Needed while
    /// building because a symbol-less bond between two of them is aromatic.
    aromatic_atoms: HashSet<AtomId>,
    /// Reference to the original IR for error messages.
    ir: &'a SmilesIR,
}

impl<'a> Builder<'a> {
    fn new(ir: &'a SmilesIR) -> Self {
        Self {
            mol: Atomistic::new(),
            open_rings: HashMap::new(),
            aromatic_atoms: HashSet::new(),
            ir,
        }
    }

    /// Build a chain, returning the [`AtomId`] of the head atom.
    ///
    /// `prev` is the atom to bond the head to (if any — `None` for top-level).
    fn build_chain(
        &mut self,
        chain: &Chain,
        prev: Option<(AtomId, Option<BondKind>)>,
    ) -> Result<AtomId, SmilesError> {
        let head_id = self.add_atom_node(&chain.head)?;

        // Bond head to the previous atom (if coming from a branch or sequence).
        if let Some((prev_id, bond)) = prev {
            self.add_bond(prev_id, head_id, bond)?;
        }

        let mut current = head_id;

        for elem in &chain.tail {
            match elem {
                ChainElement::BondedAtom { bond, atom } => {
                    let atom_id = self.add_atom_node(atom)?;
                    self.add_bond(current, atom_id, bond_query_to_kind(bond.as_ref())?)?;
                    current = atom_id;
                }
                ChainElement::Branch { bond, chain, .. } => {
                    // Branch: build sub-chain rooted at `current`.
                    self.build_chain(chain, Some((current, bond_query_to_kind(bond.as_ref())?)))?;
                    // `current` does NOT change — branches don't advance the main chain.
                }
                ChainElement::RingClosure { bond, rnum, span } => {
                    self.handle_ring_closure(
                        current,
                        *rnum,
                        bond_query_to_kind(bond.as_ref())?,
                        *span,
                    )?;
                }
            }
        }

        Ok(head_id)
    }

    /// Create an atom from an [`AtomNode`] and return its id.
    fn add_atom_node(&mut self, node: &AtomNode) -> Result<AtomId, SmilesError> {
        match &node.spec {
            AtomSpec::Organic { symbol, aromatic } => {
                let id = self.mol.add_atom_bare(&canonical_element_symbol(symbol));
                if *aromatic {
                    self.mark_aromatic(id);
                }
                Ok(id)
            }
            AtomSpec::Bracket {
                isotope,
                symbol,
                chirality,
                hcount,
                charge,
                atom_class,
            } => {
                let sym = match symbol {
                    BracketSymbol::Element { symbol, .. } => symbol.clone(),
                    BracketSymbol::Any => "*".to_owned(),
                    BracketSymbol::Aliphatic | BracketSymbol::Aromatic => "*".to_owned(),
                };

                let aromatic = matches!(symbol, BracketSymbol::Element { aromatic: true, .. });

                let id = self.mol.add_atom_bare(&canonical_element_symbol(&sym));

                if aromatic {
                    self.mark_aromatic(id);
                }
                if let Some(iso) = isotope {
                    self.set_prop(id, "isotope", *iso as f64);
                }
                if let Some(ch) = chirality {
                    let s = match ch {
                        Chirality::CounterClockwise => "CCW",
                        Chirality::Clockwise => "CW",
                    };
                    self.set_prop_str(id, "stereo", s);
                }
                // A bracket atom states its hydrogen count exactly; an omitted
                // count means zero, not "fill the valence".
                self.set_prop(id, "h_count", hcount.unwrap_or(0) as f64);
                if let Some(c) = charge {
                    self.set_prop(id, "formal_charge", *c as f64);
                }
                if let Some(cls) = atom_class {
                    self.set_prop(id, "atom_class", *cls as f64);
                }

                Ok(id)
            }
            AtomSpec::Wildcard => {
                let id = self.mol.add_atom_bare("*");
                Ok(id)
            }
            AtomSpec::Query(_) => Err(SmilesError::new(
                SmilesErrorKind::InvalidQueryPrimitive(
                    "SMARTS query atoms cannot be converted to Atomistic".into(),
                ),
                node.span,
                "", // input not available here; span is enough
            )),
        }
    }

    fn add_bond(
        &mut self,
        a: AtomId,
        b: AtomId,
        bond: Option<BondKind>,
    ) -> Result<(), SmilesError> {
        let bid = self.mol.add_bond(a, b).map_err(|e| {
            SmilesError::new(
                SmilesErrorKind::InvalidElement(e.to_string()),
                self.ir.span,
                "",
            )
        })?;

        // A bond with no symbol between two aromatic atoms is aromatic — that
        // is what makes `c1ccccc1` a ring of 1.5-order bonds rather than six
        // single bonds. An explicit symbol always wins (biphenyl's `-`).
        let kind = bond.or_else(|| {
            (self.aromatic_atoms.contains(&a) && self.aromatic_atoms.contains(&b))
                .then_some(BondKind::Aromatic)
        });

        if let Some(kind) = kind {
            let _ =
                self.mol
                    .set_bond_class(bid, bond_kind_to_type(kind), bond_kind_to_number(kind));
            match kind {
                BondKind::Up => {
                    let _ = self
                        .mol
                        .set_bond_prop(bid, "stereo", PropValue::Str("up".to_owned()));
                }
                BondKind::Down => {
                    let _ =
                        self.mol
                            .set_bond_prop(bid, "stereo", PropValue::Str("down".to_owned()));
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Record the notation's aromatic declaration on `id`, using the same
    /// `is_aromatic` marker that [`crate::perceive::aromaticity`] writes.
    fn mark_aromatic(&mut self, id: AtomId) {
        self.aromatic_atoms.insert(id);
        let _ = self.mol.set_atom(id, "is_aromatic", PropValue::Int(1));
    }

    fn handle_ring_closure(
        &mut self,
        current: AtomId,
        rnum: u16,
        bond: Option<BondKind>,
        span: Span,
    ) -> Result<(), SmilesError> {
        if let Some(pending) = self.open_rings.remove(&rnum) {
            // Close the ring: bond `pending.atom` ↔ `current`.
            // Use the bond type from whichever side specified one
            // (the opener or the closer). If both specified, they must agree.
            let effective_bond = match (pending.bond, bond) {
                (Some(a), Some(b)) if a != b => {
                    return Err(SmilesError::new(
                        SmilesErrorKind::RingBondConflict { rnum },
                        span,
                        "",
                    ));
                }
                (Some(a), _) => Some(a),
                (_, Some(b)) => Some(b),
                (None, None) => None,
            };
            self.add_bond(pending.atom, current, effective_bond)?;
        } else {
            // Open a new ring closure.
            self.open_rings.insert(
                rnum,
                PendingRing {
                    atom: current,
                    bond,
                    span,
                },
            );
        }
        Ok(())
    }

    fn close_rings(&self) -> Result<(), SmilesError> {
        if let Some((&rnum, pending)) = self.open_rings.iter().next() {
            return Err(SmilesError::new(
                SmilesErrorKind::UnmatchedRingClosure(rnum),
                pending.span,
                "",
            ));
        }
        Ok(())
    }

    fn set_prop(&mut self, id: AtomId, key: &str, val: f64) {
        let _ = self.mol.set_atom(id, key, val);
    }

    fn set_prop_str(&mut self, id: AtomId, key: &str, val: &str) {
        let _ = self.mol.set_atom(id, key, val);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a [`BondKind`] to the bond's chemical class.
///
/// `Aromatic` is its own class, not a number: the notation declares the ring
/// delocalized and says nothing about which Kekulé structure to pick. The
/// localized [`BondNumber`] is left `Unknown` for kekulization to decide.
fn bond_kind_to_type(kind: BondKind) -> BondType {
    match kind {
        BondKind::Single | BondKind::Up | BondKind::Down => BondType::Single,
        BondKind::Double => BondType::Double,
        BondKind::Triple => BondType::Triple,
        // A quadruple bond has no aromatic character; it is a plain class whose
        // number the notation states outright.
        BondKind::Quadruple => BondType::Double,
        BondKind::Aromatic => BondType::Aromatic,
        BondKind::Any | BondKind::Ring => BondType::Single,
    }
}

/// The localized number a [`BondKind`] states, when it states one.
fn bond_kind_to_number(kind: BondKind) -> BondNumber {
    match kind {
        BondKind::Single | BondKind::Up | BondKind::Down => BondNumber::Single,
        BondKind::Double => BondNumber::Double,
        BondKind::Triple => BondNumber::Triple,
        BondKind::Quadruple => BondNumber::Quadruple,
        // The notation declares delocalization, not a Kekulé phase.
        BondKind::Aromatic => BondNumber::Unknown,
        BondKind::Any | BondKind::Ring => BondNumber::Single,
    }
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn smiles_to_mol(input: &str) -> Atomistic {
        let ir = parse_smiles(input).unwrap();
        to_atomistic(&ir).unwrap_or_else(|e| panic!("to_atomistic({input:?}) failed: {e}"))
    }

    // -- basic molecules ----------------------------------------------------

    #[test]
    fn test_single_atom() {
        let mol = smiles_to_mol("C");
        assert_eq!(mol.n_atoms(), 1);
        assert_eq!(mol.n_bonds(), 0);
        let (_, atom) = mol.atoms().next().unwrap();
        assert_eq!(atom.get_str("element"), Some("C"));
    }

    #[test]
    fn test_ethane() {
        let mol = smiles_to_mol("CC");
        assert_eq!(mol.n_atoms(), 2);
        assert_eq!(mol.n_bonds(), 1);
    }

    #[test]
    fn test_ethanol() {
        let mol = smiles_to_mol("CCO");
        assert_eq!(mol.n_atoms(), 3);
        assert_eq!(mol.n_bonds(), 2);
    }

    // -- bond orders --------------------------------------------------------

    #[test]
    fn test_double_bond() {
        let mol = smiles_to_mol("C=O");
        assert_eq!(mol.n_atoms(), 2);
        assert_eq!(mol.n_bonds(), 1);
        let (_, bond) = mol.bonds().next().unwrap();
        assert_eq!(bond.props.get("bond_type"), Some(&PropValue::Int(2)));
        assert_eq!(bond.props.get("bond_number"), Some(&PropValue::Int(2)));
    }

    #[test]
    fn test_triple_bond() {
        let mol = smiles_to_mol("C#N");
        let (_, bond) = mol.bonds().next().unwrap();
        assert_eq!(bond.props.get("bond_type"), Some(&PropValue::Int(3)));
        assert_eq!(bond.props.get("bond_number"), Some(&PropValue::Int(3)));
    }

    // -- branches -----------------------------------------------------------

    #[test]
    fn test_branch_isobutane() {
        // isobutane: CC(C)C
        let mol = smiles_to_mol("CC(C)C");
        assert_eq!(mol.n_atoms(), 4);
        assert_eq!(mol.n_bonds(), 3);
    }

    #[test]
    fn test_acetic_acid() {
        // CC(=O)O
        let mol = smiles_to_mol("CC(=O)O");
        assert_eq!(mol.n_atoms(), 4);
        assert_eq!(mol.n_bonds(), 3);
    }

    // -- ring closures ------------------------------------------------------

    #[test]
    fn test_cyclohexane() {
        let mol = smiles_to_mol("C1CCCCC1");
        assert_eq!(mol.n_atoms(), 6);
        assert_eq!(mol.n_bonds(), 6); // 5 chain + 1 ring closure
    }

    #[test]
    fn test_benzene() {
        let mol = smiles_to_mol("c1ccccc1");
        assert_eq!(mol.n_atoms(), 6);
        assert_eq!(mol.n_bonds(), 6);
        // Check aromatic flag
        let (_, atom) = mol.atoms().next().unwrap();
        assert_eq!(atom.get("is_aromatic"), Some(&PropValue::Int(1)));
    }

    #[test]
    fn test_aromatic_element_symbol_is_canonical() {
        // The lowercase `c` of the organic aromatic subset is *notation*, not an
        // element symbol: every element-keyed lookup downstream expects "C".
        let mol = smiles_to_mol("c1ccccc1");
        for (_, atom) in mol.atoms() {
            assert_eq!(atom.get_str("element"), Some("C"));
        }
    }

    #[test]
    fn test_aromatic_bracket_element_symbol_is_canonical() {
        // Pyrrole — the bracket `[nH]` must land as element "N", flagged aromatic.
        let mol = smiles_to_mol("c1cc[nH]c1");
        let elements: Vec<String> = mol
            .atoms()
            .filter_map(|(_, a)| a.get_str("element").map(str::to_owned))
            .collect();
        assert_eq!(elements, ["C", "C", "C", "N", "C"]);
        for (_, atom) in mol.atoms() {
            assert_eq!(atom.get("is_aromatic"), Some(&PropValue::Int(1)));
        }
    }

    #[test]
    fn test_implicit_aromatic_bond_gets_the_aromatic_class() {
        // A bond written without a symbol between two aromatic atoms *is* an
        // aromatic bond (Daylight rule) — order 1.5, not the single-bond default.
        let mol = smiles_to_mol("c1ccccc1");
        for (_, bond) in mol.bonds() {
            // The notation declares the class; it declares no Kekulé phase, so
            // the localized number stays unknown until standardization.
            assert_eq!(bond.props.get("bond_type"), Some(&PropValue::Int(4)));
            assert_eq!(bond.props.get("bond_number"), Some(&PropValue::Int(0)));
        }
    }

    #[test]
    fn test_aromatic_smarts_discriminates_ring_from_chain() {
        // The markers exist so SMARTS can tell an aromatic ring from a
        // saturated one: `c` must match benzene and `C` must not, and the
        // reverse for cyclohexane.
        use crate::perceive::smarts::{MatchOptions, SmartsPattern};

        let n_matches = |pattern: &str, smiles: &str| {
            SmartsPattern::parse(pattern)
                .expect("parse SMARTS")
                .find(&smiles_to_mol(smiles), MatchOptions::default())
                .len()
        };

        assert_eq!(n_matches("[c]", "c1ccccc1"), 6, "benzene is aromatic");
        assert_eq!(n_matches("[C]", "c1ccccc1"), 0, "benzene is not aliphatic");
        assert_eq!(n_matches("[C]", "C1CCCCC1"), 6, "cyclohexane is aliphatic");
        assert_eq!(
            n_matches("[c]", "C1CCCCC1"),
            0,
            "cyclohexane is not aromatic"
        );
    }

    #[test]
    fn test_bond_from_aromatic_to_aliphatic_is_not_aromatic() {
        // Toluene: the ring→methyl bond has one aliphatic end, so it stays single.
        let mol = smiles_to_mol("Cc1ccccc1");
        let n_aromatic = mol
            .bonds()
            .filter(|(_, b)| b.props.get("bond_type") == Some(&PropValue::Int(4)))
            .count();
        assert_eq!(n_aromatic, 6, "only the 6 ring bonds are aromatic");
    }

    #[test]
    fn test_aliphatic_bond_does_not_get_the_aromatic_class() {
        let mol = smiles_to_mol("CC");
        let (_, bond) = mol.bonds().next().unwrap();
        assert_ne!(bond.props.get("bond_type"), Some(&PropValue::Int(4)));
    }

    #[test]
    fn test_bracket_atom_records_exact_hydrogen_count() {
        // In a bracket atom the H count is *exact* — an omitted count means zero,
        // it does not mean "fill the valence".
        let mol = smiles_to_mol("[C]");
        let (_, atom) = mol.atoms().next().unwrap();
        assert_eq!(atom.get("h_count"), Some(&PropValue::F64(0.0)));
    }

    #[test]
    fn test_two_digit_ring() {
        let mol = smiles_to_mol("C%12CCCCC%12");
        assert_eq!(mol.n_atoms(), 6);
        assert_eq!(mol.n_bonds(), 6);
    }

    // -- bracket atoms with properties --------------------------------------

    #[test]
    fn test_isotope() {
        let mol = smiles_to_mol("[13CH4]");
        assert_eq!(mol.n_atoms(), 1);
        let (_, atom) = mol.atoms().next().unwrap();
        assert_eq!(atom.get_f64("isotope"), Some(13.0));
        assert_eq!(atom.get_f64("h_count"), Some(4.0));
    }

    #[test]
    fn test_charge() {
        let mol = smiles_to_mol("[Fe+2]");
        let (_, atom) = mol.atoms().next().unwrap();
        assert_eq!(atom.get_str("element"), Some("Fe"));
        assert_eq!(atom.get_f64("formal_charge"), Some(2.0));
    }

    #[test]
    fn test_negative_charge() {
        let mol = smiles_to_mol("[O-]");
        let (_, atom) = mol.atoms().next().unwrap();
        assert_eq!(atom.get_f64("formal_charge"), Some(-1.0));
    }

    #[test]
    fn test_chirality() {
        let mol = smiles_to_mol("[C@@H](F)(Cl)Br");
        assert_eq!(mol.n_atoms(), 4); // C, F, Cl, Br (H is in h_count)
        let atoms: Vec<_> = mol.atoms().collect();
        let c_atom = &atoms
            .iter()
            .find(|(_, a)| a.get_str("element") == Some("C"))
            .unwrap()
            .1;
        assert_eq!(c_atom.get_str("stereo"), Some("CW"));
    }

    #[test]
    fn test_atom_class() {
        let mol = smiles_to_mol("[CH3:1]");
        let (_, atom) = mol.atoms().next().unwrap();
        assert_eq!(atom.get_f64("atom_class"), Some(1.0));
    }

    // -- disconnected components --------------------------------------------

    #[test]
    fn test_salt() {
        let mol = smiles_to_mol("[Na+].[Cl-]");
        assert_eq!(mol.n_atoms(), 2);
        assert_eq!(mol.n_bonds(), 0); // disconnected
    }

    // -- directional bonds --------------------------------------------------

    #[test]
    fn test_cis_trans() {
        let mol = smiles_to_mol("F/C=C/F");
        assert_eq!(mol.n_atoms(), 4);
        assert_eq!(mol.n_bonds(), 3);
    }

    // -- real molecules -----------------------------------------------------

    #[test]
    fn test_caffeine() {
        let mol = smiles_to_mol("Cn1cnc2c1c(=O)n(c(=O)n2C)C");
        assert!(mol.n_atoms() >= 14);
    }

    #[test]
    fn test_aspirin() {
        let mol = smiles_to_mol("CC(=O)Oc1ccccc1C(=O)O");
        assert!(mol.n_atoms() >= 13);
    }

    // -- error cases --------------------------------------------------------

    #[test]
    fn test_unmatched_ring() {
        let ir = parse_smiles("CC1CC").unwrap();
        let err = to_atomistic(&ir).unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::UnmatchedRingClosure(1)));
    }

    #[test]
    fn test_smarts_query_rejected() {
        let ir = crate::io::smiles::parse_smarts("[!C]").unwrap();
        let err = to_atomistic(&ir);
        assert!(err.is_err());
    }
}
