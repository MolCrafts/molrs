//! Expansion of the lowest `CGsmiles` level into one [`Atomistic`] graph.
//!
//! Each node of that level is one instance of its fragment definition: the
//! body is converted once per definition, cloned once per node, stamped with
//! the node's index as `frag_id`, and merged into one accumulating graph.
//! Every [`ResolvedPair`](crate::io::smiles::ResolvedPair) of the level then
//! becomes one real bond, given the class the pairing resolved rather than the
//! single bond [`Atomistic::add_bond`] defaults to.
//!
//! What comes out is **topology only** — the heavy atoms the bodies wrote, the
//! bonds they and the pairing imply, and the properties the notation declared.
//! A line notation states no geometry, so no atom carries a position; no
//! hydrogen is added and no perception is run. Those are steps a caller
//! composes.
//!
//! Only the lowest level becomes atoms. The pairs of an intermediate level
//! stay on the IR for a reader that wants the coarse graph itself.

use crate::io::smiles::cgsmiles::ast::{CGSmilesIR, FragmentBody, PairEnd, ResolvedPair};
use crate::io::smiles::cgsmiles::resolve::FragmentCache;
use crate::io::smiles::error::{Notation, SmilesError, SmilesErrorKind};
use molrs::system::atomistic::{AtomId, Atomistic};
use molrs::system::molgraph::PropValue;
use molrs::types::I;

impl CGSmilesIR {
    /// Expand the lowest coarse-grained level into one [`Atomistic`] graph.
    ///
    /// Every node of that level becomes a disjoint copy of its fragment body,
    /// and every pair the reader resolved becomes one bond between two of
    /// those copies, carrying the class the pairing decided: the bond symbol
    /// the notation wrote next to either descriptor if it wrote one, otherwise
    /// aromatic if both port atoms are written aromatic, otherwise single.
    /// Ports left unpaired create nothing at all.
    ///
    /// # Per-atom instance membership
    ///
    /// Every atom carries the key **`frag_id`**, a [`PropValue::Int`] holding
    /// the index of the lowest-level node it came from, so a caller can
    /// partition the result by instance without re-deriving the grouping. It
    /// is not `mol_id` (a molecule id — coarse instances are sub-molecular)
    /// and not `res_id` (a biopolymer residue); the name mirrors the
    /// `CGsmiles` reference implementation's `fragid`.
    ///
    /// # Topology only
    ///
    /// No coordinates, no hydrogen repletion, no aromaticity perception and no
    /// kekulization: `to_atomistic` builds the graph the notation states and
    /// stops. The atom count is the heavy-atom count the bodies wrote.
    ///
    /// # Indices, and what is guaranteed about them
    ///
    /// The pairing walks edges in **parse order**, while the reference
    /// implementation walks its graph library's adjacency order — for
    /// `{[#A]1[#B][#C]1}` that is (0,1), (1,2), (0,2) here against (0,1),
    /// (0,2), (1,2) there — so a ring-bearing string can pair different ports
    /// and lay its atoms out in a different order. What is contracted is the
    /// **connectivity**: the expansion is isomorphic to the reference's, and to
    /// what the molpy builder builds from the same string, as an unlabelled
    /// graph. Atom indices are not part of that contract; `frag_id` and the
    /// element are.
    ///
    /// Bond classes are not part of it either. The reference sets order 1.5
    /// whenever both endpoint atoms are aromatic, over a written symbol as
    /// well: biphenyl's `-[$]` bond is 1.5 there and
    /// [`BondKind::Single`](crate::io::smiles::BondKind) here,
    /// and a `=`-on-`=` pair is 1.5 there and `Double` here. molrs follows the
    /// Daylight Theory Manual, *SMILES* § 3.2.2 — a written symbol states the
    /// bond's order — and reads aromaticity only where neither side wrote one.
    ///
    /// A method, not a free function: the data it reads is this value's, and
    /// the free name
    /// [`to_atomistic`](crate::io::smiles::to_atomistic()) already belongs to
    /// the SMILES pipeline, where it converts a [`SmilesIR`](crate::io::smiles::SmilesIR).
    /// A second free function of the same name in the same public module would
    /// be told apart only by its argument type.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::CgNotExpandable`] when there is no atomistic body to
    /// expand: a base-only string (payload `base-only string (no fragment
    /// table)`) or a lowest level whose bodies are coarse graphs (payload the
    /// fragment's name).
    ///
    /// [`SmilesErrorKind::CgBuild`] when a structural write to the graph
    /// fails — adding a bond, classing it, or stamping `frag_id` — or when the
    /// IR names a fragment its own table does not define, or a port the
    /// converted body does not hold. No fallible call on this path is
    /// discarded.
    ///
    /// Whatever converting a fragment body raises — an unmatched ring closure
    /// inside it, say — propagates with the converter's **own** kind, stamped
    /// [`Notation::CGsmiles`](crate::io::smiles::Notation::CGsmiles) and with
    /// no input text: a value returned by
    /// [`parse_cgsmiles`](crate::io::smiles::parse_cgsmiles) has bodies that
    /// already converted once, so only a hand-built IR reaches this.
    ///
    /// # Examples
    ///
    /// An OH-capped PEO trimer: five beads over two definitions, eleven heavy
    /// atoms and ten bonds.
    ///
    /// ```
    /// use molrs::io::smiles::parse_cgsmiles;
    ///
    /// let ir = parse_cgsmiles("{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}").unwrap();
    /// let mol = ir.to_atomistic().unwrap();
    ///
    /// assert_eq!(mol.n_atoms(), 11);
    /// assert_eq!(mol.n_bonds(), 10);
    /// ```
    pub fn to_atomistic(&self) -> Result<Atomistic, SmilesError> {
        // One pair list per level is what the reader builds. A hand-built IR
        // that breaks it would otherwise expand to a bondless molecule, so it
        // is refused here rather than silently honoured.
        if self.pairs.len() != self.levels.len() {
            let reason = format!(
                "pairs/levels misaligned: {} pair lists for {} levels",
                self.pairs.len(),
                self.levels.len()
            );
            return Err(self.build_error(reason));
        }
        // No fragment table (or no level) is no atomistic body: a base-only
        // string writes beads and never says what they are made of. The check
        // above makes the lowest level's pair list present exactly when the
        // lowest level is.
        let (Some(defs), Some(level), Some(pairs)) =
            (self.fragments.last(), self.levels.last(), self.pairs.last())
        else {
            let payload = "base-only string (no fragment table)".to_owned();
            return Err(self.not_expandable(payload));
        };
        let mut cache = FragmentCache::default();
        let mut mol = Atomistic::new();
        let mut instances = Vec::with_capacity(level.nodes.len());
        for (instance, node) in level.nodes.iter().enumerate() {
            let Some(def) = defs.get(&node.name) else {
                let reason = format!("no definition for fragment '{}'", node.name);
                return Err(self.build_error(reason));
            };
            let FragmentBody::Smiles(body) = &def.body else {
                return Err(self.not_expandable(node.name.clone()));
            };
            let (converted, map) = cache
                .get_or_build(&node.name, body)
                .map_err(|e| SmilesError::new(e.kind, e.span, "", Notation::CGsmiles))?;
            // One conversion per definition, one copy per instance. A clone
            // preserves handles, so the cached port ids address the copy too,
            // until the merge remaps them.
            let copy = converted.clone();
            let ports: Vec<AtomId> = map.iter().map(|(atom, _)| *atom).collect();
            instances.push(self.merge_instance(&mut mol, copy, &ports, instance)?);
        }
        self.bond_pairs(&mut mol, &instances, pairs)?;
        Ok(mol)
    }

    /// Stamp one converted body with its instance index, merge it in, and
    /// report where its ports ended up in the accumulated graph.
    ///
    /// `ports` holds the port atoms as the body numbers them;
    /// [`Atomistic::merge`] remaps every handle, so the returned vector is the
    /// same ports read through that remapping — the only form later bonding
    /// may use.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::CgBuild`] if the stamp cannot be written, if the
    /// instance index does not fit the [`PropValue::Int`] the key is stored
    /// as, or if the merge did not carry a port atom across.
    fn merge_instance(
        &self,
        mol: &mut Atomistic,
        mut body: Atomistic,
        ports: &[AtomId],
        instance: usize,
    ) -> Result<Vec<AtomId>, SmilesError> {
        let id = I::try_from(instance)
            .map_err(|e| self.build_error(format!("instance {instance} is not an Int: {e}")))?;
        let atoms: Vec<AtomId> = body.atoms().map(|(atom, _)| atom).collect();
        for atom in atoms {
            body.set_atom(atom, "frag_id", PropValue::Int(id))
                .map_err(|e| self.build_error(format!("frag_id on instance {instance}: {e}")))?;
        }
        let handles = mol.merge(body);
        ports
            .iter()
            .enumerate()
            .map(|(port, atom)| {
                handles.get(atom).copied().ok_or_else(|| {
                    self.build_error(format!("port {port} of instance {instance} is lost"))
                })
            })
            .collect()
    }

    /// Turn every resolved pair of the lowest level into one classed bond.
    ///
    /// `instances` is the port map of each node of that level, as
    /// [`merge_instance`](CGSmilesIR::merge_instance) returned it, and `pairs`
    /// that level's resolved pairs — passed in rather than read off `self`, so
    /// this step cannot mistake another level's list for its own.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::CgBuild`] when a pair does not name two last-level
    /// ports, when it names an instance or a port the level does not offer, or
    /// when the graph refuses the bond or its class.
    fn bond_pairs(
        &self,
        mol: &mut Atomistic,
        instances: &[Vec<AtomId>],
        pairs: &[ResolvedPair],
    ) -> Result<(), SmilesError> {
        for pair in pairs {
            let src = self.port_atom(&pair.src, instances)?;
            let dst = self.port_atom(&pair.dst, instances)?;
            let bond = mol
                .add_bond(src, dst)
                .map_err(|e| self.build_error(format!("bond between two fragments: {e}")))?;
            // `add_bond` takes no order and defaults to a single bond, so the
            // class the pairing resolved has to be written explicitly.
            mol.set_bond_class(bond, pair.kind.bond_type(), pair.kind.bond_number())
                .map_err(|e| self.build_error(format!("class of an inter-fragment bond: {e}")))?;
        }
        Ok(())
    }

    /// The accumulated-graph atom one pair end names.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::CgBuild`] for an end that is not a last-level port
    /// ([`PairEnd::Sub`] belongs to an intermediate level, which grows no
    /// atoms), or for an instance or port index the level does not offer.
    fn port_atom(&self, end: &PairEnd, instances: &[Vec<AtomId>]) -> Result<AtomId, SmilesError> {
        let PairEnd::Body { instance, port } = end else {
            return Err(self.build_error(format!("{end:?} is not a last-level port")));
        };
        instances
            .get(*instance)
            .and_then(|ports| ports.get(*port))
            .copied()
            .ok_or_else(|| self.build_error(format!("no port {port} on instance {instance}")))
    }

    /// A refusal that there is nothing atomistic to expand, spanned at the
    /// whole string.
    fn not_expandable(&self, payload: String) -> SmilesError {
        SmilesError::new(
            SmilesErrorKind::CgNotExpandable(payload),
            self.span,
            "",
            Notation::CGsmiles,
        )
    }

    /// A refusal that an invariant of the reader was violated on the expansion
    /// path, spanned at the whole string.
    ///
    /// The input text is not carried: an IR is a value, and by the time it is
    /// expanded the string it was read from is the caller's, not this type's.
    fn build_error(&self, reason: String) -> SmilesError {
        SmilesError::new(
            SmilesErrorKind::CgBuild(reason),
            self.span,
            "",
            Notation::CGsmiles,
        )
    }
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::io::smiles::cgsmiles::test_support::{body, pair};
    use crate::io::smiles::{
        BondKind, CGBondOrder, CGEdge, CGFragmentDef, CGGraph, CGNode, CGSmilesIR, EdgeOrigin,
        FragmentBody, SmilesErrorKind, Span, parse_cgsmiles, parse_fragment_smiles,
    };
    use molrs::store::keys;
    use molrs::system::atomistic::{AtomId, Atomistic};
    use molrs::system::bond::{BondNumber, BondType};
    use molrs::system::molgraph::PropValue;

    // Every count below is hand-derived from the fixtures of § Domain basis of
    // `.claude/specs/cgsmiles-01d-resolve.md`: heavy atoms are counted off the
    // written bodies (no hydrogen is added here), intra-fragment bonds off the
    // same bodies, and one inter-fragment bond is added per resolved pair.
    // R4.14 fixes the class of an inter-fragment bond, R4.17 drops unpaired
    // ports, and R5.1 stamps each atom with its instance as `frag_id`. No
    // external program produced any value here.

    // -- helpers ------------------------------------------------------------

    /// The atomistic expansion of a `CGsmiles` string that must parse.
    fn expanded(text: &str) -> Atomistic {
        parse_cgsmiles(text)
            .unwrap_or_else(|e| panic!("parse_cgsmiles({text:?}) failed: {e}"))
            .to_atomistic()
            .unwrap_or_else(|e| panic!("to_atomistic({text:?}) failed: {e}"))
    }

    /// The instance an expanded atom belongs to, as the `frag_id` stamp states
    /// it. Panics unless the stamp is an [`PropValue::Int`] — the variant the
    /// contract names, and the one `mol_id` already uses.
    fn frag_id(mol: &Atomistic, id: AtomId) -> i32 {
        let atom = mol
            .get_atom(id)
            .unwrap_or_else(|e| panic!("atom {id:?} is missing: {e}"));
        match atom.get("frag_id") {
            Some(PropValue::Int(value)) => *value,
            other => panic!("atom {id:?} carries frag_id {other:?}"),
        }
    }

    /// `frag_id` of every atom, sorted: how many atoms each instance
    /// contributed, and that every atom carries the stamp.
    fn frag_ids(mol: &Atomistic) -> Vec<i32> {
        let mut ids: Vec<i32> = mol.atoms().map(|(id, _)| frag_id(mol, id)).collect();
        ids.sort();
        ids
    }

    /// The class of every bond whose two atoms belong to **different**
    /// instances — the bonds resolution created, told from the bodies' own by
    /// the stamp rather than by index arithmetic.
    fn inter_fragment_bonds(mol: &Atomistic) -> Vec<(BondType, BondNumber)> {
        let mut classes = Vec::new();
        for (bid, bond) in mol.bonds() {
            let (i, j) = (bond.nodes[0], bond.nodes[1]);
            if frag_id(mol, i) != frag_id(mol, j) {
                classes.push((mol.bond_type(bid), mol.bond_number(bid)));
            }
        }
        classes
    }

    /// The number of bonds at each atom of instance `instance`, sorted.
    fn degrees_of(mol: &Atomistic, instance: i32) -> Vec<usize> {
        let mut degrees: Vec<usize> = mol
            .atoms()
            .filter(|(id, _)| frag_id(mol, *id) == instance)
            .map(|(id, _)| mol.neighbor_bonds(id).count())
            .collect();
        degrees.sort();
        degrees
    }

    /// A coarse node named `name`, with nothing else on it: expansion reads a
    /// node's name and its index, never its descriptors.
    fn node(name: &str) -> CGNode {
        CGNode {
            name: name.to_owned(),
            charge: None,
            annotations: Vec::new(),
            descriptors: Vec::new(),
            parent: None,
            span: Span::new(0, 0),
        }
    }

    /// A written single edge between `i` and `j`.
    fn edge(i: usize, j: usize) -> CGEdge {
        CGEdge {
            i,
            j,
            order: CGBondOrder::Single,
            span: Span::new(0, 0),
            origin: EdgeOrigin::Written,
        }
    }

    /// One fragment table of atomistic bodies, `name` → parsed body.
    fn table(entries: &[(&str, &str)]) -> BTreeMap<String, CGFragmentDef> {
        entries
            .iter()
            .map(|(name, text)| {
                let body = parse_fragment_smiles(text)
                    .unwrap_or_else(|e| panic!("body {text:?} must parse: {e}"));
                (
                    (*name).to_owned(),
                    CGFragmentDef {
                        name: (*name).to_owned(),
                        body: FragmentBody::Smiles(body),
                        span: Span::new(0, 0),
                    },
                )
            })
            .collect()
    }

    // -- F5 and F7 from a hand-built IR -------------------------------------

    /// F5, polystyrene, as a resolved IR: BB0(PH1), BB2(PH3) with edges
    /// (0,1), (2,3), (0,2) and the three pairs resolution derives from them —
    /// BB's `$` to each phenyl's `$`, and BB0's `>` to BB2's `<`. Built by
    /// hand so expansion is exercised as its own unit.
    fn f5_ir() -> CGSmilesIR {
        CGSmilesIR {
            levels: vec![CGGraph {
                nodes: vec![node("BB"), node("PH"), node("BB"), node("PH")],
                edges: vec![edge(0, 1), edge(2, 3), edge(0, 2)],
            }],
            fragments: vec![table(&[("BB", "[>]CC[<][$]"), ("PH", "[$]c1ccccc1")])],
            pairs: vec![vec![
                pair(0, 0, body(0, 2), body(1, 0), BondKind::Single),
                pair(1, 0, body(2, 2), body(3, 0), BondKind::Single),
                pair(2, 0, body(0, 0), body(2, 1), BondKind::Single),
            ]],
            span: Span::new(0, 0),
        }
    }

    /// Two backbone carbons and six ring carbons twice over: 16 atoms. Bonds:
    /// one per backbone pair (2), six per ring (12), and three inter-fragment
    /// bonds — 17. The two unpaired ports add nothing (R4.17).
    #[test]
    fn test_f5_hand_built_ir_expands_to_sixteen_atoms_and_seventeen_bonds() {
        let mol = f5_ir().to_atomistic().expect("F5 must expand");
        assert_eq!(mol.n_atoms(), 16);
        assert_eq!(mol.n_bonds(), 17);
    }

    /// BB2's second carbon carries two resolved pairs — the `<` that bound it
    /// to BB0 and the `$` that bound it to its phenyl — on top of its own
    /// backbone bond, so it has degree 3, while BB2's first carbon keeps its
    /// unpaired `>` and stays at degree 1.
    #[test]
    fn test_f5_backbone_carbon_of_two_pairs_has_degree_three() {
        let mol = f5_ir().to_atomistic().expect("F5 must expand");
        assert_eq!(degrees_of(&mol, 2), vec![1, 3]);
    }

    /// F7, dichlorotoluene over three beads, as a resolved IR: the labels
    /// `$a`, `$b` and the unlabelled `$` leave the three edges no freedom, and
    /// every port atom is written aromatic, so every pair is aromatic.
    fn f7_ir() -> CGSmilesIR {
        CGSmilesIR {
            levels: vec![CGGraph {
                nodes: vec![node("SC4"), node("SX3"), node("SX3A")],
                edges: vec![edge(0, 1), edge(1, 2), edge(0, 2)],
            }],
            fragments: vec![table(&[
                ("SC4", "Cc[$a]c[$]"),
                ("SX3", "Clc[$a]c[$b]"),
                ("SX3A", "Clc[$b]c[$]"),
            ])],
            pairs: vec![vec![
                pair(0, 0, body(0, 0), body(1, 0), BondKind::Aromatic),
                pair(1, 0, body(1, 1), body(2, 0), BondKind::Aromatic),
                pair(2, 0, body(0, 1), body(2, 1), BondKind::Aromatic),
            ]],
            span: Span::new(0, 0),
        }
    }

    /// Three heavy atoms per body (`Cl` is one atom): 9. Two bonds per body
    /// plus three inter-bead bonds: 9, one ring.
    #[test]
    fn test_f7_hand_built_ir_expands_to_nine_atoms_and_nine_bonds() {
        let mol = f7_ir().to_atomistic().expect("F7 must expand");
        assert_eq!(mol.n_atoms(), 9);
        assert_eq!(mol.n_bonds(), 9);
    }

    // -- F2: counts, stamps and inter-fragment classes ----------------------

    /// F2 — an OH-capped PEO trimer.
    const F2: &str = "{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}";

    /// One oxygen per cap and three heavy atoms per `[$]COC[$]`: 11 atoms.
    /// Two bonds per PEO body (6) and four inter-fragment bonds: 10 — a tree.
    #[test]
    fn test_f2_expands_to_eleven_atoms_and_ten_bonds() {
        let mol = expanded(F2);
        assert_eq!(mol.n_atoms(), 11);
        assert_eq!(mol.n_bonds(), 10);
    }

    /// Every atom carries the index of the lowest-level node it came from, as
    /// an `Int`: one atom for each cap, three for each of the three PEO
    /// instances.
    #[test]
    fn test_f2_stamps_every_atom_with_its_instance_index() {
        let mol = expanded(F2);
        assert_eq!(frag_ids(&mol), vec![0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]);
    }

    /// The four bonds resolution created are plain single bonds — no symbol is
    /// written on any F2 descriptor and no port atom is aromatic — and they
    /// are given that class explicitly rather than left to `add_bond`'s
    /// default.
    #[test]
    fn test_f2_inter_fragment_bonds_are_single() {
        let mol = expanded(F2);
        assert_eq!(
            inter_fragment_bonds(&mol),
            vec![(BondType::Single, BondNumber::Single); 4]
        );
    }

    // -- F3: an aromatic ring of beads --------------------------------------

    /// Martini benzene: two carbons per bead (6) and one bond per bead plus
    /// three inter-bead bonds (6), closing one ring.
    #[test]
    fn test_f3_expands_to_six_atoms_and_six_bonds() {
        let mol = expanded("{[#TC5]1[#TC5][#TC5]1}.{#TC5=[$]cc[$]}");
        assert_eq!(mol.n_atoms(), 6);
        assert_eq!(mol.n_bonds(), 6);
    }

    /// The notation declares delocalization, not a Kekulé phase: the three
    /// inter-bead bonds are of the aromatic class with no localized number.
    #[test]
    fn test_f3_inter_bead_bonds_are_aromatic_with_an_unknown_number() {
        let mol = expanded("{[#TC5]1[#TC5][#TC5]1}.{#TC5=[$]cc[$]}");
        assert_eq!(
            inter_fragment_bonds(&mol),
            vec![(BondType::Aromatic, BondNumber::Unknown); 3]
        );
    }

    // -- F4: a double coarse edge is two single bonds -----------------------

    /// Martini cyclohexane: three carbons per bead (6), two bonds per bead
    /// plus the two bonds the one `=` edge stands for (6), closing one ring.
    #[test]
    fn test_f4_expands_to_six_atoms_and_six_bonds() {
        let mol = expanded("{[#SC3]=[#SC3]}.{#SC3=[$]CCC[$]}");
        assert_eq!(mol.n_atoms(), 6);
        assert_eq!(mol.n_bonds(), 6);
    }

    /// Both of them are single bonds. A single double bond here would be the
    /// coarse multiplicity read as chemistry.
    #[test]
    fn test_f4_inter_bead_bonds_are_both_single() {
        let mol = expanded("{[#SC3]=[#SC3]}.{#SC3=[$]CCC[$]}");
        assert_eq!(
            inter_fragment_bonds(&mol),
            vec![(BondType::Single, BondNumber::Single); 2]
        );
    }

    // -- F6: a tripeptide ---------------------------------------------------

    /// `[>]NCC(=O)[<]` is four heavy atoms and three bonds, `[>]NC(C)C(=O)[<]`
    /// five and four: 13 atoms, 10 intra-fragment bonds and 2 inter-fragment
    /// bonds.
    #[test]
    fn test_f6_expands_to_thirteen_atoms_and_twelve_bonds() {
        let mol = expanded("{[#GLY][#ALA][#GLY]}.{#GLY=[>]NCC(=O)[<],#ALA=[>]NC(C)C(=O)[<]}");
        assert_eq!(mol.n_atoms(), 13);
        assert_eq!(mol.n_bonds(), 12);
    }

    // -- what expansion does not do -----------------------------------------

    /// Topology only: a notation string states no geometry, so no atom is
    /// given a position for a later step to mistake for one.
    #[test]
    fn test_expansion_writes_no_atom_positions() {
        let mol = expanded(F2);
        for (id, atom) in mol.atoms() {
            for key in [keys::X, keys::Y, keys::Z] {
                assert!(
                    !atom.contains_key(key),
                    "atom {id:?} carries {key:?}: {atom:?}"
                );
            }
        }
    }

    /// Hydrogen repletion is a step the caller composes: the expansion holds
    /// exactly the heavy atoms the bodies wrote — one per cap, three per PEO.
    #[test]
    fn test_expansion_adds_no_hydrogen_atoms() {
        let mol = expanded(F2);
        assert_eq!(mol.n_atoms(), 11);
        for (id, atom) in mol.atoms() {
            assert_ne!(
                atom.get_str(keys::ELEMENT),
                Some("H"),
                "atom {id:?} is a hydrogen"
            );
        }
    }

    // -- what expansion refuses ---------------------------------------------

    /// A lowest level whose bodies are coarse graphs has no atoms to build,
    /// and the refusal names the fragment it stopped at.
    #[test]
    fn test_graph_body_at_the_lowest_level_is_not_expandable() {
        let ir = CGSmilesIR {
            levels: vec![CGGraph {
                nodes: vec![node("A")],
                edges: Vec::new(),
            }],
            fragments: vec![BTreeMap::from([(
                "A".to_owned(),
                CGFragmentDef {
                    name: "A".to_owned(),
                    body: FragmentBody::Graph(CGGraph {
                        nodes: vec![node("X")],
                        edges: Vec::new(),
                    }),
                    span: Span::new(0, 0),
                },
            )])],
            pairs: vec![Vec::new()],
            span: Span::new(0, 0),
        };
        let err = ir
            .to_atomistic()
            .expect_err("a coarse lowest level cannot be expanded");
        assert!(
            matches!(err.kind, SmilesErrorKind::CgNotExpandable(ref name) if name == "A"),
            "kind was {:?}",
            err.kind
        );
    }

    /// One pair list per level is what the reader builds. A hand-built IR that
    /// drops the list — F5 with its `pairs` emptied — would otherwise expand
    /// to a bondless molecule, so the misalignment is named rather than
    /// honoured.
    #[test]
    fn test_ir_without_a_pair_list_per_level_is_refused() {
        let mut ir = f5_ir();
        ir.pairs = vec![];
        let err = ir
            .to_atomistic()
            .expect_err("an IR with no pair list cannot be expanded");
        assert!(
            matches!(
                err.kind,
                SmilesErrorKind::CgBuild(ref msg) if msg.contains("pairs/levels misaligned")
            ),
            "kind was {:?}",
            err.kind
        );
    }

    /// A base-only string names no fragment at all, so the refusal has no
    /// fragment to name and says so in the phrase `to_fragment` reuses.
    #[test]
    fn test_base_only_ir_is_not_expandable() {
        let ir = parse_cgsmiles("{[#PEO][#PEO][#PEO]}").expect("base-only string must parse");
        let err = ir
            .to_atomistic()
            .expect_err("a base-only string has no body to expand");
        assert!(
            matches!(
                err.kind,
                SmilesErrorKind::CgNotExpandable(ref payload)
                    if payload == "base-only string (no fragment table)"
            ),
            "kind was {:?}",
            err.kind
        );
    }
}
