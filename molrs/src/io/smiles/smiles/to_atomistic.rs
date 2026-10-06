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
//! (mass, charge, isotope, chirality, hydrogen count).
//!
//! [`fragment_to_atomistic`] is the fragment-dialect entry point and shares
//! that one walk: it additionally hands back the bonding descriptors the
//! notation anchored on each atom, which [`to_atomistic`] refuses rather than
//! drops.

use std::collections::{HashMap, HashSet};

use crate::io::smiles::chem::ast::*;
use crate::io::smiles::error::{Notation, SmilesError, SmilesErrorKind};
use crate::io::smiles::smiles::canonical_element_symbol;
use molrs::Element;
use molrs::error::MolRsError;
use molrs::store::keys;
use molrs::system::atomistic::{AtomId, Atomistic};
use molrs::system::molgraph::PropValue;

/// Convert a parsed SMILES IR into an [`Atomistic`] molecular graph.
///
/// This resolves ring closures into bonds, sets atom properties (charge,
/// isotope, chirality), and records bond orders. Implicit hydrogens are
/// **not** added — call [`Perceive::find_hydrogens`](crate::perceive::Perceive::find_hydrogens)
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
/// [`Perceive::find_hydrogens`](crate::perceive::Perceive::find_hydrogens).
///
/// # Errors
///
/// Returns [`SmilesErrorKind::UnmatchedRingClosure`] for a ring digit that is
/// never closed, [`SmilesErrorKind::InvalidQueryPrimitive`] for SMARTS query
/// atoms and bond queries (which have no single atomistic interpretation), and
/// [`SmilesErrorKind::DescriptorsUnconvertible`] for a node carrying a bonding
/// descriptor — this is the plain conversion, and dropping the descriptor
/// silently is the one outcome it will not produce; use
/// [`fragment_to_atomistic`] instead.
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
    let mut builder = Builder::new(ir, /*collect_descriptors*/ false);

    for component in &ir.components {
        builder.build_chain(component, None)?;
    }

    builder.close_rings()?;

    Ok(builder.mol)
}

/// Convert a parsed SMILES **fragment** IR into an [`Atomistic`] graph plus the
/// descriptor map.
///
/// The input is a SMILES fragment body with `CGsmiles` / `BigSMILES` bonding
/// descriptors, as produced by
/// [`parse_fragment_smiles`](crate::io::smiles::parse_fragment_smiles); this is
/// the fragment-dialect sibling of [`to_atomistic`], which refuses
/// descriptor-bearing input rather than drop it. The graph is built by the same
/// walk, so descriptor-free input gives the same atoms, bonds and properties
/// through either entry point.
///
/// The second return value pairs each descriptor with the [`AtomId`] of the
/// atom it binds to. Its order is a contract, not an accident: the later
/// stages that turn descriptors into *ports* — the named joining sites a
/// fragment offers — index this vector, so the order is fixed as atom-visit
/// order first, then each atom's own descriptor-list order. That is not text
/// order: in `C(N[<])[>]` the `>` the parser folded onto the head carbon
/// precedes the `<` on the branch nitrogen, because the walk reaches the head
/// carbon before it descends into the branch. Inputs with several
/// `.`-separated components are walked component by component, in written
/// order.
///
/// Descriptors are data, not atoms: no hydrogens are added, and an unpaired
/// descriptor creates neither an atom nor a bond here. Filling open sites — by
/// pairing them with another fragment's, or by capping them with hydrogen — is
/// a later step that reads this map.
///
/// Descriptors are passed through, not re-checked: the parser validated every
/// descriptor at its single construction site, so a hand-built IR carrying a
/// label or order no parser would have produced is returned as given.
///
/// # Errors
///
/// Returns [`SmilesErrorKind::UnmatchedRingClosure`] for a ring digit that is
/// never closed and [`SmilesErrorKind::InvalidQueryPrimitive`] for SMARTS
/// query atoms and bond queries (which have no single atomistic
/// interpretation). A bonding descriptor is *not* an error here — carrying it
/// out to the caller is what this entry point is for.
pub fn fragment_to_atomistic(
    ir: &SmilesIR,
) -> Result<(Atomistic, Vec<(AtomId, BondingDescriptor)>), SmilesError> {
    let mut builder = Builder::new(ir, /*collect_descriptors*/ true);

    for component in &ir.components {
        builder.build_chain(component, None)?;
    }

    builder.close_rings()?;

    let Builder {
        mol, descriptors, ..
    } = builder;
    Ok((mol, descriptors))
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
            Notation::Smarts,
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
    /// Which entry point this walk serves: `true` on the fragment path, which
    /// records every descriptor against its atom; `false` on the plain path,
    /// where a node carrying one is an error rather than a silent loss.
    collect_descriptors: bool,
    /// Each descriptor against the atom it binds to, in visit order. Empty on
    /// the plain path, which never gets past the guard in
    /// [`Builder::add_atom_node`]. Owned, not borrowed: the map is returned by
    /// value.
    descriptors: Vec<(AtomId, BondingDescriptor)>,
}

impl<'a> Builder<'a> {
    fn new(ir: &'a SmilesIR, collect_descriptors: bool) -> Self {
        Self {
            mol: Atomistic::new(),
            open_rings: HashMap::new(),
            aromatic_atoms: HashSet::new(),
            ir,
            collect_descriptors,
            descriptors: Vec::new(),
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
    ///
    /// Every atom of every chain, branch and component flows through here, so
    /// this is also where bonding descriptors are either refused (plain path)
    /// or recorded against their atom (fragment path).
    ///
    /// # Mass
    ///
    /// Every atom that names an element carries `mass` (amu, `f64`): the
    /// standard atomic weight [`Element::atomic_mass`] states, the same table
    /// hydrogen repletion reads. An atom written with an explicit isotope
    /// (`[13C]`) carries its **mass number** A instead — an approximation,
    /// since molrs has no isotope-mass table; the error is below 0.02 amu for
    /// the light elements. The wildcard (`*`, and the bracket `[*]`, `[A]`,
    /// `[a]`) names no element and carries no mass, isotope or not: an isotope
    /// on a wildcard is an attachment label, not a nucleus.
    fn add_atom_node(&mut self, node: &AtomNode) -> Result<AtomId, SmilesError> {
        if !self.collect_descriptors && !node.descriptors.is_empty() {
            return Err(SmilesError::new(
                SmilesErrorKind::DescriptorsUnconvertible,
                node.span,
                "", // input not available here; span is enough
                Notation::Smiles,
            ));
        }

        let id = match &node.spec {
            AtomSpec::Organic { symbol, .. } => {
                let element = canonical_element_symbol(symbol);
                let id = self.mol.add_atom_bare(&element);
                self.set_mass(id, &element, None, node.span)?;
                if node.spec.written_aromatic() {
                    self.mark_aromatic(id, node.span)?;
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

                let element = canonical_element_symbol(&sym);
                let id = self.mol.add_atom_bare(&element);
                self.set_mass(id, &element, *isotope, node.span)?;

                if node.spec.written_aromatic() {
                    self.mark_aromatic(id, node.span)?;
                }
                if let Some(iso) = isotope {
                    self.set_prop(id, "isotope", *iso as f64, node.span)?;
                }
                if let Some(ch) = chirality {
                    let s = match ch {
                        Chirality::CounterClockwise => "CCW",
                        Chirality::Clockwise => "CW",
                    };
                    self.set_prop_str(id, "stereo", s, node.span)?;
                }
                // A bracket atom states its hydrogen count exactly; an omitted
                // count means zero, not "fill the valence".
                self.set_prop(id, "h_count", hcount.unwrap_or(0) as f64, node.span)?;
                if let Some(c) = charge {
                    self.set_prop(id, "formal_charge", *c as f64, node.span)?;
                }
                if let Some(cls) = atom_class {
                    self.set_prop(id, "atom_class", *cls as f64, node.span)?;
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
                Notation::Smarts,
            )),
        }?;

        if self.collect_descriptors {
            self.descriptors
                .extend(node.descriptors.iter().map(|desc| (id, desc.clone())));
        }

        Ok(id)
    }

    fn add_bond(
        &mut self,
        a: AtomId,
        b: AtomId,
        bond: Option<BondKind>,
    ) -> Result<(), SmilesError> {
        let bid = self
            .mol
            .add_bond(a, b)
            .map_err(|e| self.build_error(&e, self.ir.span))?;

        // A bond with no symbol between two aromatic atoms is aromatic — that
        // is what makes `c1ccccc1` a ring of 1.5-order bonds rather than six
        // single bonds. An explicit symbol always wins (biphenyl's `-`).
        let kind = bond.or_else(|| {
            (self.aromatic_atoms.contains(&a) && self.aromatic_atoms.contains(&b))
                .then_some(BondKind::Aromatic)
        });

        if let Some(kind) = kind {
            self.mol
                .set_bond_class(bid, kind.bond_type(), kind.bond_number())
                .map_err(|e| self.build_error(&e, self.ir.span))?;
            let stereo = match kind {
                BondKind::Up => Some("up"),
                BondKind::Down => Some("down"),
                _ => None,
            };
            if let Some(stereo) = stereo {
                self.mol
                    .set_bond_prop(bid, "stereo", PropValue::Str(stereo.to_owned()))
                    .map_err(|e| self.build_error(&e, self.ir.span))?;
            }
        }

        Ok(())
    }

    /// Wrap a graph-write failure as [`SmilesErrorKind::Build`] at `span`.
    ///
    /// Every write this builder makes is to a node or relation it has just
    /// created, with a key and a value the grammar fixes, so a failure is an
    /// internal invariant of the builder rather than anything the input said.
    /// Reporting it by value is what keeps the walk free of discarded
    /// `Result`s: a silently dropped write would leave an `Atomistic` missing
    /// a component the conversion promised to set, and no caller would learn.
    /// The input text is not available at this stage, so the error carries the
    /// span alone.
    fn build_error(&self, e: &MolRsError, span: Span) -> SmilesError {
        SmilesError::new(
            SmilesErrorKind::Build(e.to_string()),
            span,
            "",
            Notation::Smiles,
        )
    }

    /// Record the notation's aromatic declaration on `id`, using the same
    /// `is_aromatic` marker that [`crate::perceive::aromaticity`] writes.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::Build`] at `span` when the marker cannot be written.
    fn mark_aromatic(&mut self, id: AtomId, span: Span) -> Result<(), SmilesError> {
        self.aromatic_atoms.insert(id);
        self.mol
            .set_atom(id, "is_aromatic", PropValue::Int(1))
            .map_err(|e| self.build_error(&e, span))
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
                        Notation::Smiles,
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
                Notation::Smiles,
            ));
        }
        Ok(())
    }

    /// Write `mass` on `id` for the element symbol `element`, per the rule
    /// [`Builder::add_atom_node`] states: the mass number when `isotope` is
    /// written, the table's atomic weight otherwise, nothing when `element`
    /// names no element.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::Build`] at `span` — see [`Builder::build_error`].
    fn set_mass(
        &mut self,
        id: AtomId,
        element: &str,
        isotope: Option<u16>,
        span: Span,
    ) -> Result<(), SmilesError> {
        let Some(table) = Element::by_symbol(element) else {
            return Ok(());
        };
        let mass = match isotope {
            Some(a) => f64::from(a),
            None => f64::from(table.atomic_mass()),
        };
        self.set_prop(id, keys::MASS, mass, span)
    }

    /// Write one numeric component on `id`.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::Build`] at `span` — see [`Builder::build_error`].
    fn set_prop(&mut self, id: AtomId, key: &str, val: f64, span: Span) -> Result<(), SmilesError> {
        self.mol
            .set_atom(id, key, val)
            .map_err(|e| self.build_error(&e, span))
    }

    /// Write one string component on `id`.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::Build`] at `span` — see [`Builder::build_error`].
    fn set_prop_str(
        &mut self,
        id: AtomId,
        key: &str,
        val: &str,
        span: Span,
    ) -> Result<(), SmilesError> {
        self.mol
            .set_atom(id, key, val)
            .map_err(|e| self.build_error(&e, span))
    }
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::{parse_fragment_smiles, parse_smiles};

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

    // -- mass ---------------------------------------------------------------
    //
    // The expected mass of an element atom is read from molrs's own `Element`
    // table (the source of truth, `.claude/specs/assembly-03-template.md`
    // § Domain basis); an explicit isotope writes its mass number A. No
    // external program produced any value here.

    /// The standard atomic weight of `symbol` from the `Element` table, as
    /// the `f64` a mass prop stores.
    fn table_mass(symbol: &str) -> f64 {
        f64::from(
            molrs::Element::by_symbol(symbol)
                .unwrap_or_else(|| panic!("{symbol:?} is not in the Element table"))
                .atomic_mass(),
        )
    }

    #[test]
    fn test_element_atoms_carry_their_table_mass() {
        let mol = smiles_to_mol("CCO");
        let masses: Vec<(String, Option<f64>)> = mol
            .atoms()
            .map(|(_, a)| {
                (
                    a.get_str("element").unwrap_or("").to_owned(),
                    a.get_f64(molrs::store::keys::MASS),
                )
            })
            .collect();
        let expected = ["C", "C", "O"];
        assert_eq!(masses.len(), expected.len());
        for ((element, mass), want) in masses.iter().zip(expected) {
            assert_eq!(element, want);
            let mass = mass.unwrap_or_else(|| panic!("{element} atom carries no mass"));
            assert!(
                (mass - table_mass(want)).abs() < 1e-12,
                "{element}: mass {mass} != table {}",
                table_mass(want)
            );
        }
    }

    #[test]
    fn test_isotope_atom_carries_its_mass_number() {
        let mol = smiles_to_mol("[13CH4]");
        let (_, atom) = mol.atoms().next().unwrap();
        let mass = atom
            .get_f64(molrs::store::keys::MASS)
            .expect("an isotope atom carries a mass");
        assert!((mass - 13.0).abs() < 1e-12, "mass {mass} != 13.0");
    }

    #[test]
    fn test_wildcard_atom_carries_no_mass() {
        let mol = smiles_to_mol("*C");
        let atoms: Vec<_> = mol.atoms().map(|(_, a)| a).collect();
        assert_eq!(atoms.len(), 2);
        assert_eq!(atoms[0].get_str("element"), Some("*"));
        assert!(
            !atoms[0].contains_key(molrs::store::keys::MASS),
            "the wildcard has no element, hence no mass"
        );
        assert_eq!(atoms[1].get_str("element"), Some("C"));
        let mass = atoms[1]
            .get_f64(molrs::store::keys::MASS)
            .expect("the carbon carries a mass");
        assert!((mass - table_mass("C")).abs() < 1e-12);
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

    /// Hand-built IR for `CCCC1` — a chain of four carbons whose last element
    /// is a ring digit that never closes.
    ///
    /// The parser refuses an unmatched ring closure itself, so this unit can
    /// only be reached with an IR built directly.
    fn unmatched_ring_ir() -> SmilesIR {
        fn carbon(start: usize) -> AtomNode {
            AtomNode {
                spec: AtomSpec::Organic {
                    symbol: "C".to_owned(),
                    aromatic: false,
                },
                span: Span::new(start, start + 1),
                descriptors: Vec::new(),
            }
        }

        SmilesIR {
            components: vec![Chain {
                head: carbon(0),
                tail: vec![
                    ChainElement::BondedAtom {
                        bond: None,
                        atom: carbon(1),
                    },
                    ChainElement::BondedAtom {
                        bond: None,
                        atom: carbon(2),
                    },
                    ChainElement::BondedAtom {
                        bond: None,
                        atom: carbon(3),
                    },
                    ChainElement::RingClosure {
                        bond: None,
                        rnum: 1,
                        span: Span::new(4, 5),
                    },
                ],
            }],
            span: Span::new(0, 5),
        }
    }

    #[test]
    fn test_unmatched_ring() {
        let ir = unmatched_ring_ir();
        let err = to_atomistic(&ir).unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::UnmatchedRingClosure(1)));
    }

    #[test]
    fn test_smarts_query_rejected() {
        let ir = crate::io::smiles::parse_smarts("[!C]").unwrap();
        let err = to_atomistic(&ir);
        assert!(err.is_err());
    }

    // -- fragment dialect: bonding descriptors ------------------------------

    fn fragment_to_mol(input: &str) -> (Atomistic, Vec<(AtomId, BondingDescriptor)>) {
        let ir = parse_fragment_smiles(input).unwrap();
        fragment_to_atomistic(&ir)
            .unwrap_or_else(|e| panic!("fragment_to_atomistic({input:?}) failed: {e}"))
    }

    /// Atom ids in the order the walker created them.
    fn atom_ids(mol: &Atomistic) -> Vec<AtomId> {
        mol.atoms().map(|(id, _)| id).collect()
    }

    /// Element symbols in creation order.
    fn elements(mol: &Atomistic) -> Vec<String> {
        mol.atoms()
            .filter_map(|(_, a)| a.get_str("element").map(str::to_owned))
            .collect()
    }

    /// Declared bond class (`bond_type`, `bond_number`) in creation order.
    fn bond_classes(mol: &Atomistic) -> Vec<(Option<PropValue>, Option<PropValue>)> {
        mol.bonds()
            .map(|(_, b)| {
                (
                    b.props.get("bond_type").cloned(),
                    b.props.get("bond_number").cloned(),
                )
            })
            .collect()
    }

    #[test]
    fn test_to_atomistic_rejects_a_descriptor_bearing_ir() {
        // The plain entry point routes the caller instead of silently dropping
        // the descriptors the fragment dialect parsed.
        let ir = parse_fragment_smiles("[$]COC[$]").unwrap();
        let err = to_atomistic(&ir).unwrap_err();
        assert!(matches!(
            err.kind,
            SmilesErrorKind::DescriptorsUnconvertible
        ));
    }

    #[test]
    fn test_fragment_to_atomistic_builds_the_heavy_atom_graph() {
        // `[$]COC[$]` is three heavy atoms and two bonds; descriptors are not
        // atoms and add neither.
        let (mol, _) = fragment_to_mol("[$]COC[$]");
        assert_eq!(mol.n_atoms(), 3);
        assert_eq!(mol.n_bonds(), 2);
    }

    #[test]
    fn test_fragment_to_atomistic_maps_each_descriptor_to_its_atom() {
        // R4.2: the leading `$` binds to the first carbon, the trailing one to
        // the last, and the map lists them in that (visit) order.
        let (mol, ports) = fragment_to_mol("[$]COC[$]");
        let ids = atom_ids(&mol);
        assert_eq!(ports.len(), 2);
        assert_eq!(
            (ports[0].0, ports[0].1.kind),
            (ids[0], DescriptorKind::Symmetric)
        );
        assert_eq!(
            (ports[1].0, ports[1].1.kind),
            (ids[2], DescriptorKind::Symmetric)
        );
    }

    #[test]
    fn test_fragment_to_atomistic_keeps_per_atom_descriptor_order() {
        // R4.3: atom 0 carries `[>]` then `[$1]` in written order, so the map
        // emits both before moving on to the next atom's descriptors.
        let (mol, ports) = fragment_to_mol("[>][$1]COC[<]");
        let ids = atom_ids(&mol);
        assert_eq!(ports.len(), 3);
        assert_eq!(
            (ports[0].0, ports[0].1.kind, ports[0].1.label.as_str()),
            (ids[0], DescriptorKind::Right, "")
        );
        assert_eq!(
            (ports[1].0, ports[1].1.kind, ports[1].1.label.as_str()),
            (ids[0], DescriptorKind::Symmetric, "1")
        );
        assert_eq!(
            (ports[2].0, ports[2].1.kind, ports[2].1.label.as_str()),
            (ids[2], DescriptorKind::Left, "")
        );
    }

    #[test]
    fn test_fragment_to_atomistic_map_follows_the_atom_descriptor_list() {
        // `C([$]O)[>]`: `(` is not a node, so the branch's leading `$` and the
        // trailing `>` both anchor on the first carbon; the map keeps that
        // atom's list order, `$` before `>`.
        let (mol, ports) = fragment_to_mol("C([$]O)[>]");
        let ids = atom_ids(&mol);
        assert_eq!(ports.len(), 2);
        assert_eq!(
            (ports[0].0, ports[0].1.kind),
            (ids[0], DescriptorKind::Symmetric)
        );
        assert_eq!(
            (ports[1].0, ports[1].1.kind),
            (ids[0], DescriptorKind::Right)
        );
    }

    #[test]
    fn test_fragment_to_atomistic_map_is_visit_order_not_text_position() {
        // `C(N[<])[>]`: `<` anchors on the branch nitrogen (atom 1), `>` on the
        // head carbon (atom 0), which the parser folds on after the branch. The
        // walker visits the head before the branch, so the `>` entry comes
        // first — the reverse of the text, where `[<]` precedes `[>]`.
        let (mol, ports) = fragment_to_mol("C(N[<])[>]");
        let ids = atom_ids(&mol);
        assert_eq!(ports.len(), 2);
        assert_eq!(
            (ports[0].0, ports[0].1.kind),
            (ids[0], DescriptorKind::Right)
        );
        assert_eq!(
            (ports[1].0, ports[1].1.kind),
            (ids[1], DescriptorKind::Left)
        );
    }

    #[test]
    fn test_fragment_to_atomistic_matches_to_atomistic_without_descriptors() {
        // One walker behind two entry points: descriptor-free input must give
        // the same graph through either, and an empty map.
        let plain = to_atomistic(&parse_smiles("CC(=O)O").unwrap()).unwrap();
        let (fragment, ports) = fragment_to_mol("CC(=O)O");
        assert!(ports.is_empty());
        assert_eq!(fragment.n_atoms(), plain.n_atoms());
        assert_eq!(fragment.n_bonds(), plain.n_bonds());
        assert_eq!(elements(&fragment), elements(&plain));
        assert_eq!(bond_classes(&fragment), bond_classes(&plain));
    }
}
