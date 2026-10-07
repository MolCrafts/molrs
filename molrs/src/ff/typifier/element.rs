//! Element typing: type labels derived from element symbols alone.
//!
//! [`ElementTypifier`] gives every atom and link a `type` label built from the
//! elements it touches, for writers that need type labels (LAMMPS data Type
//! Labels) on a molecule no force field has typed.

use std::collections::HashMap;

use indexmap::IndexMap;
use molrs::core::Atomistic;
use molrs::core::TypeName;
use molrs::core::keys;
use molrs::core::schema::block_names::{ANGLES, BONDS, DIHEDRALS};
use molrs::core::{MolGraph, NodeId, PropValue};

use crate::ff::forcefield::ForceField;
use crate::ff::typifier::{Annotation, TypeAssignment, Typifier};

/// A typifier whose labels are the elements themselves; it defines no force
/// field.
///
/// # Labels
///
/// Every label is written under `type`:
///
/// - an atom: its `element`, e.g. `"C"`;
/// - a bond, angle or dihedral: the elements of its endpoints, oriented to
///   the smaller (slot by slot) of the tuple read forward and reversed, then
///   [`TypeName::join`]ed. Bond O–H is `"H-O"`, angle H–C–C is `"C-C-H"`,
///   dihedral O–C–C–H is `"H-C-C-O"`.
///
/// Angles and dihedrals are labelled only when the graph has them; nothing is
/// generated.
///
/// # Stamp-only
///
/// The [`TypeAssignment`] carries only [`Annotation::Value`] entries and no styles or
/// pair rows, so [`Typing::forcefield`](crate::ff::typifier::Typing::forcefield)
/// stays empty. Masses and charges are left on the atoms as they are.
///
/// # Errors
///
/// - an atom without a string `element`: `Err` naming its row;
/// - a graph with improper rows: `Err` naming their count. Improper labels
///   are not derived.
///
/// # Example
///
/// ```
/// use molrs::core::Atomistic;
/// use molrs::ff::typifier::{ElementTypifier, Typing};
///
/// let mut mol = Atomistic::new();
/// let o = mol.add_atom_bare("O");
/// let h = mol.add_atom_bare("H");
/// mol.add_bond(o, h).unwrap();
///
/// let mut typing = Typing::new(ElementTypifier::new());
/// let typed = typing.typify(&mol).unwrap();
///
/// let (_, bond) = typed.bonds().next().unwrap();
/// assert_eq!(bond.props["type"], "H-O".into());
/// assert!(typing.forcefield().styles().is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct ElementTypifier {
    library: ForceField,
}

impl ElementTypifier {
    /// A typifier with an empty library named `"element"`.
    pub fn new() -> Self {
        Self {
            library: ForceField::new("element"),
        }
    }
}

impl Default for ElementTypifier {
    fn default() -> Self {
        Self::new()
    }
}

/// One `type = label` annotation list.
fn type_value(label: &str) -> Vec<(String, Annotation)> {
    vec![(
        "type".to_owned(),
        Annotation::Value(PropValue::Str(label.to_owned())),
    )]
}

/// The label of every row of relation kind `kind`, in row order: the
/// [`TypeName`] of its endpoint elements in [`TypeName::orient`]'s spelling,
/// cached per element tuple.
/// Empty when the graph has no such kind or no rows of it.
fn link_labels(
    graph: &MolGraph,
    kind: &str,
    elements: &HashMap<NodeId, &str>,
    cache: &mut HashMap<Vec<String>, String>,
) -> Result<Vec<Vec<(String, Annotation)>>, String> {
    let Some(kind_id) = graph.kind_id(kind) else {
        return Ok(Vec::new());
    };
    let mut rows = Vec::new();
    for (position, id) in graph.relation_ids(kind_id).enumerate() {
        let nodes = graph
            .relation_nodes(kind_id, id)
            .map_err(|e| format!("{kind} {position}: {e}"))?;
        let tuple: Vec<&str> = nodes
            .iter()
            .map(|n| {
                elements
                    .get(n)
                    .copied()
                    .ok_or_else(|| format!("{kind} {position}: endpoint {n:?} is not an atom"))
            })
            .collect::<Result<_, _>>()?;
        let key: Vec<String> = tuple.iter().map(|e| (*e).to_owned()).collect();
        let label = match cache.get(&key) {
            Some(label) => label.clone(),
            None => {
                let label = TypeName::join(&TypeName::orient(&tuple))
                    .map_err(|e| format!("{kind} {position}: {e}"))?
                    .as_str()
                    .to_owned();
                cache.insert(key, label.clone());
                label
            }
        };
        rows.push(type_value(&label));
    }
    Ok(rows)
}

impl Typifier for ElementTypifier {
    fn assign(&self, graph: &mut Atomistic) -> Result<TypeAssignment, String> {
        if graph.n_impropers() > 0 {
            return Err(format!(
                "ElementTypifier derives no improper labels; the graph has {} impropers",
                graph.n_impropers()
            ));
        }
        let graph = graph.as_molgraph();
        let table = graph.node_table();
        let mut elements: HashMap<NodeId, &str> = HashMap::new();
        let mut nodes = Vec::new();
        for (row, id) in graph.node_ids().enumerate() {
            let element = table.get_str(id, keys::ELEMENT).map_err(|e| {
                format!(
                    "ElementTypifier: atom {row} has no string '{}': {e}",
                    keys::ELEMENT
                )
            })?;
            elements.insert(id, element);
            nodes.push(type_value(element));
        }

        let mut cache: HashMap<Vec<String>, String> = HashMap::new();
        let mut links = IndexMap::new();
        for kind in [BONDS, ANGLES, DIHEDRALS] {
            links.insert(
                kind.to_owned(),
                link_labels(graph, kind, &elements, &mut cache)?,
            );
        }
        Ok(TypeAssignment {
            nodes,
            links,
            ..TypeAssignment::default()
        })
    }

    fn source_forcefield(&self) -> &ForceField {
        &self.library
    }
}

#[cfg(test)]
mod tests {
    //! `ElementTypifier` through `Typing::typify` on hand-built graphs. Every
    //! expected label is written by hand.

    use molrs::core::{Atom, PropValue};
    use molrs::core::{Atomistic, NodeId};

    use super::*;
    use crate::ff::typifier::Typing;

    fn str_value(s: &str) -> Option<PropValue> {
        Some(PropValue::Str(s.to_owned()))
    }

    fn atom_types(g: &Atomistic) -> Vec<Option<PropValue>> {
        g.atoms().map(|(_, a)| a.get("type").cloned()).collect()
    }

    fn bond_types(g: &Atomistic) -> Vec<Option<PropValue>> {
        g.bonds()
            .map(|(_, b)| b.props.get("type").cloned())
            .collect()
    }

    fn angle_types(g: &Atomistic) -> Vec<Option<PropValue>> {
        g.angles()
            .map(|(_, a)| a.props.get("type").cloned())
            .collect()
    }

    fn dihedral_types(g: &Atomistic) -> Vec<Option<PropValue>> {
        g.dihedrals()
            .map(|(_, d)| d.props.get("type").cloned())
            .collect()
    }

    /// O bonded to two H: bonds written O–H and H–O.
    fn water() -> Atomistic {
        let mut g = Atomistic::new();
        let o = g.add_atom_bare("O");
        let h1 = g.add_atom_bare("H");
        let h2 = g.add_atom_bare("H");
        g.add_bond(o, h1).unwrap();
        g.add_bond(h2, o).unwrap();
        g
    }

    /// C1–C2–H with bonds C1–C2, C2–H and one angle written H–C2–C1.
    fn cch() -> (Atomistic, [NodeId; 3]) {
        let mut g = Atomistic::new();
        let c1 = g.add_atom_bare("C");
        let c2 = g.add_atom_bare("C");
        let h = g.add_atom_bare("H");
        g.add_bond(c1, c2).unwrap();
        g.add_bond(c2, h).unwrap();
        (g, [c1, c2, h])
    }

    #[test]
    fn water_atoms_get_elements_and_bonds_the_oriented_pair() {
        let mol = water();
        let atoms_before: Vec<Atom> = mol.atoms().map(|(_, a)| a).collect();
        let bonds_before: Vec<_> = mol.bonds().map(|(_, b)| (b.nodes, b.props)).collect();
        let mut typing = Typing::new(ElementTypifier::new());

        let typed = typing.typify(&mol).unwrap();

        assert_eq!(
            atom_types(&typed),
            vec![str_value("O"), str_value("H"), str_value("H")]
        );
        assert_eq!(bond_types(&typed), vec![str_value("H-O"), str_value("H-O")]);
        assert!(
            typing.forcefield().styles().is_empty(),
            "{:?}",
            typing.forcefield().styles()
        );
        let atoms_after: Vec<Atom> = mol.atoms().map(|(_, a)| a).collect();
        let bonds_after: Vec<_> = mol.bonds().map(|(_, b)| (b.nodes, b.props)).collect();
        assert_eq!(atoms_after, atoms_before);
        assert_eq!(bonds_after, bonds_before);
    }

    #[test]
    fn angle_label_is_the_oriented_element_sequence() {
        let (mut mol, [c1, c2, h]) = cch();
        mol.add_angle(h, c2, c1).unwrap();
        let mut typing = Typing::new(ElementTypifier::default());

        let typed = typing.typify(&mol).unwrap();

        assert_eq!(bond_types(&typed), vec![str_value("C-C"), str_value("C-H")]);
        assert_eq!(angle_types(&typed), vec![str_value("C-C-H")]);
    }

    #[test]
    fn dihedral_label_is_the_oriented_element_sequence() {
        let mut mol = Atomistic::new();
        let o = mol.add_atom_bare("O");
        let c1 = mol.add_atom_bare("C");
        let c2 = mol.add_atom_bare("C");
        let h = mol.add_atom_bare("H");
        mol.add_bond(o, c1).unwrap();
        mol.add_bond(c1, c2).unwrap();
        mol.add_bond(c2, h).unwrap();
        mol.add_dihedral(o, c1, c2, h).unwrap();
        let mut typing = Typing::new(ElementTypifier::new());

        let typed = typing.typify(&mol).unwrap();

        assert_eq!(dihedral_types(&typed), vec![str_value("H-C-C-O")]);
    }

    #[test]
    fn a_graph_without_angles_or_dihedrals_gets_no_such_labels() {
        let (mol, _) = cch();
        let mut typing = Typing::new(ElementTypifier::new());

        let typed = typing.typify(&mol).unwrap();

        assert_eq!(typed.n_angles(), 0);
        assert_eq!(typed.n_dihedrals(), 0);
        assert_eq!(bond_types(&typed), vec![str_value("C-C"), str_value("C-H")]);
    }

    #[test]
    fn a_graph_with_an_improper_is_refused_naming_impropers() {
        let mut mol = Atomistic::new();
        let c = mol.add_atom_bare("C");
        let ids: Vec<NodeId> = (0..3).map(|_| mol.add_atom_bare("H")).collect();
        for &h in &ids {
            mol.add_bond(c, h).unwrap();
        }
        mol.add_improper(c, ids[0], ids[1], ids[2]).unwrap();
        let mut typing = Typing::new(ElementTypifier::new());

        let err = typing.typify(&mol).unwrap_err();

        assert!(err.contains("improper"), "{err}");
        assert!(err.contains('1'), "{err}");
    }

    #[test]
    fn an_atom_without_element_is_refused_naming_its_row() {
        let mut mol = Atomistic::new();
        mol.add_atom_bare("C");
        mol.add_atom(Atom::new());
        let mut typing = Typing::new(ElementTypifier::new());

        let err = typing.typify(&mol).unwrap_err();

        assert!(err.contains("element"), "{err}");
        assert!(err.contains("atom 1"), "{err}");
    }
}
