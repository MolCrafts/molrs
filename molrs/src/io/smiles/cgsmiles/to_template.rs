//! Whole-molecule templates: one [`Fragment`] per `CGsmiles` string.
//!
//! Where [`CGSmilesIR::to_fragment`] builds one template per *definition*,
//! [`CGSmilesIR::to_template`] builds one for the *whole* lowest level: a
//! multi-bead monomer written as a `CGsmiles` string becomes a single library
//! entry, its beads recorded per atom and its open valences made ports.

use std::collections::HashSet;

use crate::io::smiles::cgsmiles::ast::{CGSmilesIR, PairEnd};
use crate::io::smiles::cgsmiles::to_atomistic::Expansion;
use crate::io::smiles::cgsmiles::to_fragment::{OpenSite, cap_open_sites, cg_build};
use crate::io::smiles::error::SmilesError;
use molrs::error::MolRsError;
use molrs::store::keys;
use molrs::system::atomistic::AtomId;
use molrs::system::fragment::Fragment;
use molrs::system::molgraph::FRAG_ID;

impl CGSmilesIR {
    /// Build one instance-free [`Fragment`] template of the whole lowest
    /// level.
    ///
    /// The lowest level expands exactly as
    /// [`to_atomistic`](Self::to_atomistic) expands it — the same bodies, the
    /// same [`ResolvedPair`](crate::io::smiles::ResolvedPair) bonds, the same
    /// bond-class rule — and the result is then made a template:
    ///
    /// - **Bead stamps.** Every atom carries `bead` (a
    ///   [`PropValue::Int`](molrs::system::molgraph::PropValue::Int)) = the
    ///   index of the lowest-level node it came from, and `bead_type` = that
    ///   node's name. `bead` is template-local; no atom carries `frag_id`,
    ///   because a template has no instances.
    /// - **Open valences.** Every descriptor that no pair of the lowest level
    ///   consumed becomes a capping hydrogen bonded to its anchor, carrying
    ///   the hydrogen mass (g/mol) and its anchor's bead stamps, plus a port with the
    ///   descriptor's kind, label and order — exactly as
    ///   [`to_fragment`](Self::to_fragment) caps a definition. Handles are
    ///   appended after every heavy atom in (node index, descriptor index)
    ///   order, so a port's handle row orders the ports.
    /// - **No geometry.** A line notation states none, so no atom carries a
    ///   coordinate. A `builder::TracePlacer` needs `x`/`y`/`z` (Å) on every
    ///   atom, so the caller supplies geometry first (for example by conformer
    ///   generation). Masses (g/mol) come from the element table, as the
    ///   SMILES builder and the capping step write them.
    ///
    /// # Errors
    ///
    /// Those of [`to_atomistic`](Self::to_atomistic):
    /// [`SmilesErrorKind::CgNotExpandable`](crate::io::smiles::SmilesErrorKind::CgNotExpandable)
    /// when there is no atomistic body (a base-only string, or a coarse
    /// lowest level), and
    /// [`SmilesErrorKind::CgBuild`](crate::io::smiles::SmilesErrorKind::CgBuild)
    /// when a structural write fails — including capping a handle or
    /// recording its port. A descriptor whose written bond symbol states no
    /// multiplicity is
    /// [`SmilesErrorKind::InvalidDescriptorOrder`](crate::io::smiles::SmilesErrorKind::InvalidDescriptorOrder),
    /// as in `to_fragment`; the parser refuses one, so only a hand-built IR
    /// reaches it.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::io::smiles::parse_cgsmiles;
    ///
    /// let ir = parse_cgsmiles("{[#A][#B]}.{#A=[<]CC[$],#B=[$]CO[>]}")?;
    /// let monomer = ir.to_template()?;
    ///
    /// // Four heavy atoms plus one capping hydrogen per unpaired `<` / `>`.
    /// assert_eq!(monomer.n_atoms(), 6);
    /// assert_eq!(monomer.n_bonds(), 5);
    /// assert_eq!(monomer.n_ports(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_template(&self) -> Result<Fragment, SmilesError> {
        const CONTEXT: &str = "whole-molecule template";
        let Expansion {
            mut mol,
            ports,
            level,
            pairs,
            bodies,
        } = self.expand_lowest_level()?;
        let build = |e: MolRsError| cg_build(self.span, format!("{CONTEXT}: {e}"));

        // The expansion stamps each atom with its lowest-level node index as
        // `frag_id`; in a template that same index is the bead, under the
        // template-local key, and no instance id remains.
        mol.as_molgraph_mut()
            .rename_node_column(FRAG_ID, keys::BEAD)
            .map_err(build)?;
        let atoms: Vec<AtomId> = mol.as_molgraph().node_ids().collect();
        for atom in atoms {
            let bead = mol
                .as_molgraph()
                .node_table()
                .get_i32(atom, keys::BEAD)
                .map_err(build)?;
            let node = usize::try_from(bead)
                .ok()
                .and_then(|index| level.nodes.get(index))
                .ok_or_else(|| {
                    cg_build(self.span, format!("atom {atom:?} names no node {bead}"))
                })?;
            mol.set_atom(atom, keys::BEAD_TYPE, node.name.as_str())
                .map_err(build)?;
        }

        let consumed: HashSet<(usize, usize)> = pairs
            .iter()
            .flat_map(|pair| [&pair.src, &pair.dst])
            .filter_map(|end| match end {
                PairEnd::Body { instance, port } => Some((*instance, *port)),
                PairEnd::Sub { .. } => None,
            })
            .collect();
        let mut sites = Vec::new();
        for (index, (node, anchors)) in level.nodes.iter().zip(&ports).enumerate() {
            let bead = molrs::types::I::try_from(index)
                .map_err(|e| cg_build(self.span, format!("bead {index} is not an Int: {e}")))?;
            // The expansion converted every definition its level names, and
            // `anchors` was read off that body's port map, entry for entry.
            let (_, map) = bodies
                .get(&node.name)
                .expect("the expansion converted every definition its level names");
            for (port, (anchor, (_, descriptor))) in anchors.iter().zip(map).enumerate() {
                if !consumed.contains(&(index, port)) {
                    sites.push(OpenSite {
                        anchor: *anchor,
                        descriptor,
                        bead,
                        bead_type: &node.name,
                    });
                }
            }
        }
        cap_open_sites(mol, &sites, CONTEXT, self.span)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::io::smiles::{SmilesErrorKind, parse_cgsmiles};
    use molrs::store::keys;
    use molrs::system::atomistic::AtomId;
    use molrs::system::fragment::{Fragment, Port, PortKind};
    use molrs::system::molgraph::PropValue;

    // Every value below is hand-derived from the fixture, following
    // `.claude/specs/assembly-03-template.md` § Design 4. `#A=[<x]CC[$m]` and
    // `#B=[$m]CO[>y]` are four heavy atoms (C0 C1 | C2 O3) and two body bonds;
    // the `$m` pair joins C1–C2 (one inter-fragment bond). The two unpaired
    // descriptors `<x` (on C0, node 0) and `>y` (on O3, node 1) each become a
    // capping hydrogen bonded to its anchor plus a port, appended in (node,
    // descriptor) order: 4 + 2 = 6 atoms, 2 + 1 + 2 = 5 bonds, 2 ports. No
    // external program produced any value here.

    /// Two beads, one internal `$m` bond, one `<x` and one `>y` left open.
    const AB: &str = "{[#A][#B]}.{#A=[<x]CC[$m],#B=[$m]CO[>y]}";

    // -- helpers ------------------------------------------------------------

    /// The whole-molecule template of a `CGsmiles` string that must convert.
    fn template(text: &str) -> Fragment {
        parse_cgsmiles(text)
            .unwrap_or_else(|e| panic!("parse_cgsmiles({text:?}) failed: {e}"))
            .to_template()
            .unwrap_or_else(|e| panic!("to_template({text:?}) failed: {e}"))
    }

    /// Every port of a template keyed by its handle atom.
    fn ports_by_handle(frag: &Fragment) -> HashMap<AtomId, Port> {
        frag.ports()
            .map(|id| {
                let port = frag
                    .port(id)
                    .unwrap_or_else(|e| panic!("port {id:?} does not read back: {e}"));
                (port.handle, port)
            })
            .collect()
    }

    /// Atom ids in row order, split into (heavy atoms, handles).
    fn heavy_and_handles(frag: &Fragment) -> (Vec<AtomId>, Vec<AtomId>) {
        let handles = ports_by_handle(frag);
        frag.node_ids()
            .partition(|atom| !handles.contains_key(atom))
    }

    /// The `bead` stamp of one atom; panics unless it is an `Int`.
    fn bead(frag: &Fragment, atom: AtomId) -> i32 {
        let props = frag
            .get_node(atom)
            .unwrap_or_else(|e| panic!("atom {atom:?} is missing: {e}"));
        match props.get(keys::BEAD) {
            Some(PropValue::Int(v)) => *v,
            other => panic!("atom {atom:?} carries bead {other:?}"),
        }
    }

    /// The `bead_type` stamp of one atom.
    fn bead_type(frag: &Fragment, atom: AtomId) -> String {
        frag.get_node(atom)
            .unwrap_or_else(|e| panic!("atom {atom:?} is missing: {e}"))
            .get_str(keys::BEAD_TYPE)
            .unwrap_or_else(|| panic!("atom {atom:?} carries no bead_type"))
            .to_owned()
    }

    // -- shape ----------------------------------------------------------------

    #[test]
    fn builds_six_atoms_five_bonds_two_ports() {
        let frag = template(AB);
        assert_eq!(frag.n_atoms(), 6);
        assert_eq!(frag.n_bonds(), 5);
        assert_eq!(frag.n_ports(), 2);
    }

    // -- bead stamps ----------------------------------------------------------

    #[test]
    fn stamps_heavy_atoms_with_their_node_bead() {
        let frag = template(AB);
        let (heavy, _) = heavy_and_handles(&frag);
        let beads: Vec<i32> = heavy.iter().map(|&a| bead(&frag, a)).collect();
        let types: Vec<String> = heavy.iter().map(|&a| bead_type(&frag, a)).collect();
        assert_eq!(beads, vec![0, 0, 1, 1]);
        assert_eq!(types, vec!["A", "A", "B", "B"]);
    }

    #[test]
    fn stamps_each_handle_with_its_anchor_bead() {
        let frag = template(AB);
        let ports = ports_by_handle(&frag);
        assert_eq!(ports.len(), 2);
        for port in ports.values() {
            assert_eq!(
                bead(&frag, port.handle),
                bead(&frag, port.anchor),
                "handle {:?} of port {port:?}",
                port.handle
            );
            assert_eq!(
                bead_type(&frag, port.handle),
                bead_type(&frag, port.anchor),
                "handle {:?} of port {port:?}",
                port.handle
            );
        }
    }

    // -- ports ----------------------------------------------------------------

    #[test]
    fn appends_unpaired_ports_in_node_descriptor_order() {
        let frag = template(AB);
        let ports = ports_by_handle(&frag);
        let (_, handles) = heavy_and_handles(&frag);
        let in_row_order: Vec<(PortKind, String)> = handles
            .iter()
            .map(|h| {
                let port = &ports[h];
                (port.kind, port.label.clone())
            })
            .collect();
        assert_eq!(
            in_row_order,
            vec![
                (PortKind::Left, "x".to_owned()),
                (PortKind::Right, "y".to_owned()),
            ]
        );
    }

    // -- instance- and geometry-free ------------------------------------------

    #[test]
    fn writes_no_frag_id_and_no_coordinates() {
        let frag = template(AB);
        for atom in frag.node_ids() {
            let props = frag
                .get_node(atom)
                .unwrap_or_else(|e| panic!("atom {atom:?} is missing: {e}"));
            for key in [keys::X, keys::Y, keys::Z, "frag_id"] {
                assert!(!props.contains_key(key), "atom {atom:?} carries '{key}'");
            }
            assert_eq!(frag.frag_id(atom), None);
        }
    }

    // -- refusals -------------------------------------------------------------

    #[test]
    fn rejects_a_base_only_ir() {
        let ir = parse_cgsmiles("{[#A][#B]}").expect("a base-only string must parse");
        let err = ir
            .to_template()
            .expect_err("a base-only string has no body to expand");
        assert!(
            matches!(err.kind, SmilesErrorKind::CgNotExpandable(_)),
            "kind was {:?}",
            err.kind
        );
    }
}
