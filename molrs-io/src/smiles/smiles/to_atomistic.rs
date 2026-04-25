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

use std::collections::HashMap;

use crate::smiles::chem::ast::*;
use crate::smiles::error::{SmilesError, SmilesErrorKind};
use molrs::atomistic::Atomistic;
use molrs::molgraph::{AtomId, PropValue};

/// Convert a parsed SMILES IR into an [`Atomistic`] molecular graph.
///
/// This resolves ring closures into bonds, sets atom properties (charge,
/// isotope, chirality), and records bond orders. Implicit hydrogens are
/// **not** added — call [`add_hydrogens`](crate::smiles::add_hydrogens) separately
/// if needed.
///
/// # Errors
///
/// Returns an error if ring closures are unmatched or if the IR contains
/// SMARTS query atoms (which have no single atomistic interpretation).
///
/// # Examples
///
/// ```
/// use molrs_io::smiles::{parse_smiles, to_atomistic};
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
            crate::smiles::chem::ast::Span::new(0, 0),
            "",
        )),
    }
}

struct Builder<'a> {
    mol: Atomistic,
    /// Maps ring numbers to pending (unmatched) ring-closure openers.
    open_rings: HashMap<u16, PendingRing>,
    /// Reference to the original IR for error messages.
    ir: &'a SmilesIR,
}

impl<'a> Builder<'a> {
    fn new(ir: &'a SmilesIR) -> Self {
        Self {
            mol: Atomistic::new(),
            open_rings: HashMap::new(),
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
                let id = self.mol.add_atom_bare(symbol);
                if *aromatic {
                    self.set_prop(id, "aromatic", 1.0);
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

                // Capitalise aromatic single-char symbols for element lookup.
                let canon_sym =
                    if sym.len() == 1 && sym.chars().next().unwrap().is_ascii_lowercase() {
                        sym.to_ascii_uppercase()
                    } else {
                        sym.clone()
                    };

                let id = self.mol.add_atom_bare(&canon_sym);

                if aromatic {
                    self.set_prop(id, "aromatic", 1.0);
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
                if let Some(h) = hcount {
                    self.set_prop(id, "h_count", *h as f64);
                }
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

        if let Some(kind) = bond {
            let order = bond_kind_to_order(kind);
            if let Ok(b) = self.mol.get_bond_mut(bid) {
                b.props.insert("order".to_owned(), PropValue::F64(order));
                match kind {
                    BondKind::Up => {
                        b.props
                            .insert("stereo".to_owned(), PropValue::Str("up".to_owned()));
                    }
                    BondKind::Down => {
                        b.props
                            .insert("stereo".to_owned(), PropValue::Str("down".to_owned()));
                    }
                    _ => {}
                }
            }
        }

        Ok(())
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
        if let Ok(atom) = self.mol.get_atom_mut(id) {
            atom.set(key, val);
        }
    }

    fn set_prop_str(&mut self, id: AtomId, key: &str, val: &str) {
        if let Ok(atom) = self.mol.get_atom_mut(id) {
            atom.set(key, val);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a [`BondKind`] to a numeric bond order.
fn bond_kind_to_order(kind: BondKind) -> f64 {
    match kind {
        BondKind::Single | BondKind::Up | BondKind::Down => 1.0,
        BondKind::Double => 2.0,
        BondKind::Triple => 3.0,
        BondKind::Quadruple => 4.0,
        BondKind::Aromatic => 1.5,
        BondKind::Any | BondKind::Ring => 1.0,
    }
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smiles::parse_smiles;

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
        assert_eq!(bond.props.get("order"), Some(&PropValue::F64(2.0)));
    }

    #[test]
    fn test_triple_bond() {
        let mol = smiles_to_mol("C#N");
        let (_, bond) = mol.bonds().next().unwrap();
        assert_eq!(bond.props.get("order"), Some(&PropValue::F64(3.0)));
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
        assert_eq!(atom.get_f64("aromatic"), Some(1.0));
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
        let c_atom = atoms
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
        let ir = crate::smiles::parse_smarts("[!C]").unwrap();
        let err = to_atomistic(&ir);
        assert!(err.is_err());
    }
}
