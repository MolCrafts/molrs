//! Mark the atoms a reaction may bind.
//!
//! A **site** is an ordinary atom carrying the [`keys::SITE`] field. The field
//! is a plain unordered name, not a port: a reaction SMARTS finds it with a
//! `%label` predicate, and a port is a fragment's declared valence. Both exist
//! because they answer different questions — which atoms may react, and which
//! valences a fragment offers.
//!
//! Labelling and preparing a leaving group are one step because they are one
//! decision: an atom's site label is only meaningful together with the group
//! that leaves when it reacts.

use crate::store::keys;
use crate::system::molgraph::{MolGraph, NodeId, PropValue, node_to_u64};

/// The field marking an atom as a reaction site.
pub const SITE_KEY: &str = keys::SITE;

/// Stored on a leaving hydrogen: that hydrogen's own charge before
/// [`SiteMap::prepare_leaving`] folded it onto its site.
///
/// Its presence marks the fold as done. An unreacted site thaws by
/// `q(site) -= q0(H)` and `q(H) = q0(H)`, which restores both charges within
/// floating-point rounding.
pub const PRE_REACTION_CHARGE_KEY: &str = keys::Q0;

/// A [`SiteMap`] could not mark what it was asked to mark.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SiteError {
    /// No site name was supplied.
    NoNames,
    /// Fewer candidate atoms than names to give them.
    TooFewAtoms { needed: usize, found: usize },
    /// A stride below one would mark nothing.
    StepTooSmall { step: usize },
    /// A site-labelled atom has no hydrogen to leave.
    NoLeavingHydrogen { node: NodeId },
    /// A charge fold was requested but only one of the site and its leaving
    /// hydrogen carries a charge, so no fold can conserve the net charge.
    OneSidedCharge { site: NodeId, hydrogen: NodeId },
    /// The graph refused an edit (a stale handle, a type conflict).
    Graph(String),
}

impl std::fmt::Display for SiteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoNames => write!(f, "at least one site name is required"),
            Self::TooFewAtoms { needed, found } => {
                write!(f, "need {needed} atoms, found {found}")
            }
            Self::StepTooSmall { step } => write!(f, "step must be >= 1, got {step}"),
            Self::NoLeavingHydrogen { node } => write!(
                f,
                "node {} carries a site label but no hydrogen neighbour to leave",
                node_to_u64(*node)
            ),
            Self::OneSidedCharge { site, hydrogen } => write!(
                f,
                "cannot fold charge: only one of site {} and its leaving hydrogen {} \
                 carries a charge",
                node_to_u64(*site),
                node_to_u64(*hydrogen)
            ),
            Self::Graph(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for SiteError {}

impl From<crate::error::MolRsError> for SiteError {
    fn from(error: crate::error::MolRsError) -> Self {
        Self::Graph(error.to_string())
    }
}

/// Name the atoms of one graph that a reaction may bind.
///
/// Borrows the graph for the marking; nothing is copied.
pub struct SiteMap<'a> {
    graph: &'a mut MolGraph,
}

impl<'a> SiteMap<'a> {
    /// Mark atoms on `graph`.
    pub fn new(graph: &'a mut MolGraph) -> Self {
        Self { graph }
    }

    /// Label one atom with a site name.
    ///
    /// # Errors
    ///
    /// Fails when the node is stale or the field's type conflicts.
    pub fn label(&mut self, node: NodeId, name: &str) -> Result<(), SiteError> {
        self.graph
            .set_node(node, SITE_KEY, PropValue::Str(name.to_string()))?;
        Ok(())
    }

    /// Label `nodes` with `names`, in order.
    ///
    /// # Errors
    ///
    /// Fails on an empty name list, on fewer nodes than names, or on a refused
    /// write.
    pub fn label_atoms(
        &mut self,
        nodes: &[NodeId],
        names: &[&str],
    ) -> Result<Vec<NodeId>, SiteError> {
        if names.is_empty() {
            return Err(SiteError::NoNames);
        }
        if nodes.len() < names.len() {
            return Err(SiteError::TooFewAtoms {
                needed: names.len(),
                found: nodes.len(),
            });
        }
        let mut marked = Vec::with_capacity(names.len());
        for (&node, &name) in nodes.iter().zip(names) {
            self.label(node, name)?;
            marked.push(node);
        }
        Ok(marked)
    }

    /// Label the first `names.len()` atoms of `element`, in node order.
    ///
    /// # Errors
    ///
    /// Fails when fewer atoms carry that element than there are names — reusing
    /// one atom for two names would corrupt the reaction map.
    pub fn label_elements(
        &mut self,
        element: &str,
        names: &[&str],
    ) -> Result<Vec<NodeId>, SiteError> {
        let matches: Vec<NodeId> = self
            .graph
            .node_ids()
            .filter(|&node| {
                self.graph
                    .get_node(node)
                    .ok()
                    .and_then(|atom| atom.get_str(keys::ELEMENT).map(str::to_string))
                    .is_some_and(|symbol| symbol == element)
            })
            .collect();
        if matches.len() < names.len() {
            return Err(SiteError::TooFewAtoms {
                needed: names.len(),
                found: matches.len(),
            });
        }
        self.label_atoms(&matches, names)
    }

    /// Label `nodes[0::step]`, optionally preparing each one's leaving hydrogen.
    ///
    /// # Errors
    ///
    /// Fails on a stride below one, or on a marked atom with no hydrogen.
    pub fn every_nth(
        &mut self,
        nodes: &[NodeId],
        step: usize,
        site: &str,
        leaving: Option<&str>,
        fold_charge: bool,
    ) -> Result<Vec<NodeId>, SiteError> {
        if step < 1 {
            return Err(SiteError::StepTooSmall { step });
        }
        let mut marked = Vec::new();
        for &node in nodes.iter().step_by(step) {
            self.label(node, site)?;
            marked.push(node);
            if let Some(leaving) = leaving {
                self.prepare_leaving(node, leaving, fold_charge)?;
            }
        }
        Ok(marked)
    }

    /// Prepare a leaving hydrogen on every atom already labelled `site`.
    ///
    /// # Errors
    ///
    /// Fails on a marked atom with no hydrogen neighbour.
    pub fn prepare_leaving_hydrogens(
        &mut self,
        site: &str,
        leaving: &str,
        fold_charge: bool,
    ) -> Result<usize, SiteError> {
        let labelled: Vec<NodeId> = self
            .graph
            .node_ids()
            .filter(|&node| {
                self.graph
                    .get_node(node)
                    .ok()
                    .and_then(|atom| atom.get_str(SITE_KEY).map(str::to_string))
                    .is_some_and(|label| label == site)
            })
            .collect();
        for node in &labelled {
            self.prepare_leaving(*node, leaving, fold_charge)?;
        }
        Ok(labelled.len())
    }

    /// Mark the lowest-handle hydrogen next to `node` as the leaving group.
    ///
    /// With `fold_charge`, and when both atoms carry a charge, the hydrogen's
    /// charge moves onto `node` and is stashed on the hydrogen under
    /// [`PRE_REACTION_CHARGE_KEY`], so a reaction that deletes the hydrogen
    /// conserves net charge and an unreacted site can be thawed. A hydrogen
    /// that already carries that key has been folded; folding it again is a
    /// no-op, so repeating the call changes no charge.
    ///
    /// # Errors
    ///
    /// Fails when `node` has no hydrogen neighbour, or with
    /// [`SiteError::OneSidedCharge`] when `fold_charge` is set and exactly one
    /// of `node` and the hydrogen carries a charge.
    pub fn prepare_leaving(
        &mut self,
        node: NodeId,
        leaving: &str,
        fold_charge: bool,
    ) -> Result<NodeId, SiteError> {
        let hydrogen = self
            .graph
            .neighbors(node)
            .filter(|&neighbour| {
                self.graph
                    .get_node(neighbour)
                    .ok()
                    .and_then(|atom| atom.get_str(keys::ELEMENT).map(str::to_string))
                    .is_some_and(|symbol| symbol == "H")
            })
            .min()
            .ok_or(SiteError::NoLeavingHydrogen { node })?;

        let folded = self
            .graph
            .get_node(hydrogen)
            .ok()
            .is_some_and(|atom| atom.get(PRE_REACTION_CHARGE_KEY).is_some());
        if fold_charge && !folded {
            let charge_of = |atom: NodeId| {
                self.graph
                    .get_node(atom)
                    .ok()
                    .and_then(|atom| atom.get(keys::CHARGE).and_then(PropValue::as_f64))
            };
            let (hydrogen_charge, site_charge) = (charge_of(hydrogen), charge_of(node));
            if hydrogen_charge.is_some() != site_charge.is_some() {
                return Err(SiteError::OneSidedCharge {
                    site: node,
                    hydrogen,
                });
            }
            if let (Some(hydrogen_charge), Some(site_charge)) = (hydrogen_charge, site_charge) {
                self.graph.set_node(
                    hydrogen,
                    PRE_REACTION_CHARGE_KEY,
                    PropValue::F64(hydrogen_charge),
                )?;
                self.graph.set_node(
                    node,
                    keys::CHARGE,
                    PropValue::F64(site_charge + hydrogen_charge),
                )?;
                self.graph
                    .set_node(hydrogen, keys::CHARGE, PropValue::F64(0.0))?;
            }
        }
        self.label(hydrogen, leaving)?;
        Ok(hydrogen)
    }

    /// Clear site labels: on `nodes`, or on the whole graph when `None`.
    ///
    /// # Errors
    ///
    /// Fails when the graph refuses a write.
    pub fn clear(&mut self, nodes: Option<&[NodeId]>) -> Result<(), SiteError> {
        let targets: Vec<NodeId> = match nodes {
            Some(nodes) => nodes.to_vec(),
            None => self.graph.node_ids().collect(),
        };
        for node in targets {
            if self
                .graph
                .get_node(node)
                .ok()
                .and_then(|atom| atom.get_str(SITE_KEY).map(str::to_string))
                .is_some()
            {
                self.label(node, "")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PropValue;
    use crate::system::atomistic::Atomistic;

    fn water() -> Atomistic {
        let mut mol = Atomistic::new();
        let o = mol.add_atom_xyz("O", 0.0, 0.0, 0.0);
        let h1 = mol.add_atom_xyz("H", 0.96, 0.0, 0.0);
        let h2 = mol.add_atom_xyz("H", -0.24, 0.93, 0.0);
        mol.add_bond(o, h1).unwrap();
        mol.add_bond(o, h2).unwrap();
        mol
    }

    #[test]
    fn labels_the_first_atoms_of_an_element_in_order() {
        let mut mol = water();
        let marked = SiteMap::new(mol.as_molgraph_mut())
            .label_elements("O", &["a"])
            .unwrap();
        assert_eq!(marked.len(), 1);
        let oxygen = marked[0];
        assert_eq!(
            mol.as_molgraph()
                .get_node(oxygen)
                .unwrap()
                .get_str(SITE_KEY),
            Some("a")
        );
    }

    #[test]
    fn too_few_atoms_is_refused_rather_than_reused() {
        let mut mol = water();
        let error = SiteMap::new(mol.as_molgraph_mut())
            .label_elements("O", &["a", "b"])
            .expect_err("one oxygen cannot carry two names");
        assert_eq!(
            error,
            SiteError::TooFewAtoms {
                needed: 2,
                found: 1
            }
        );
    }

    #[test]
    fn preparing_a_leaving_hydrogen_folds_its_charge() {
        let mut mol = water();
        let oxygen = mol.as_molgraph().node_ids().next().unwrap();
        mol.set_atom(oxygen, keys::CHARGE, PropValue::F64(-0.8))
            .unwrap();
        for hydrogen in mol.as_molgraph().neighbors(oxygen).collect::<Vec<_>>() {
            mol.set_atom(hydrogen, keys::CHARGE, PropValue::F64(0.4))
                .unwrap();
        }
        SiteMap::new(mol.as_molgraph_mut())
            .label(oxygen, "a")
            .unwrap();
        let leaving = SiteMap::new(mol.as_molgraph_mut())
            .prepare_leaving(oxygen, "h", true)
            .unwrap();

        let atom = mol.as_molgraph().get_node(oxygen).unwrap();
        assert!((atom.get_f64(keys::CHARGE).unwrap() - (-0.4)).abs() < 1e-12);
        let hydrogen = mol.as_molgraph().get_node(leaving).unwrap();
        assert!(hydrogen.get_f64(keys::CHARGE).unwrap().abs() < 1e-12);
        assert!((hydrogen.get_f64(PRE_REACTION_CHARGE_KEY).unwrap() - 0.4).abs() < 1e-12);
        assert_eq!(hydrogen.get_str(SITE_KEY), Some("h"));
    }

    /// Every atom's `charge` and pre-fold charge, in node order.
    fn charges(mol: &Atomistic) -> Vec<(Option<PropValue>, Option<PropValue>)> {
        mol.as_molgraph()
            .node_ids()
            .map(|node| {
                let atom = mol.as_molgraph().get_node(node).unwrap();
                (
                    atom.get(keys::CHARGE).cloned(),
                    atom.get(PRE_REACTION_CHARGE_KEY).cloned(),
                )
            })
            .collect()
    }

    fn total_charge(mol: &Atomistic) -> f64 {
        mol.as_molgraph()
            .node_ids()
            .filter_map(|node| {
                mol.as_molgraph()
                    .get_node(node)
                    .unwrap()
                    .get(keys::CHARGE)
                    .and_then(PropValue::as_f64)
            })
            .sum()
    }

    /// Water with `charge` written on the oxygen and/or on both hydrogens.
    fn charged_water(
        oxygen: Option<PropValue>,
        hydrogen: Option<PropValue>,
    ) -> (Atomistic, NodeId) {
        let mut mol = water();
        let o = mol.as_molgraph().node_ids().next().unwrap();
        if let Some(q) = oxygen {
            mol.set_atom(o, keys::CHARGE, q).unwrap();
        }
        if let Some(q) = hydrogen {
            for h in mol.as_molgraph().neighbors(o).collect::<Vec<_>>() {
                mol.set_atom(h, keys::CHARGE, q.clone()).unwrap();
            }
        }
        (mol, o)
    }

    #[test]
    fn preparing_the_same_leaving_hydrogen_twice_changes_nothing() {
        let (mut mol, oxygen) =
            charged_water(Some(PropValue::F64(-0.8)), Some(PropValue::F64(0.4)));
        let first = SiteMap::new(mol.as_molgraph_mut())
            .prepare_leaving(oxygen, "h", true)
            .unwrap();
        let after_first = charges(&mol);
        let total = total_charge(&mol);

        let second = SiteMap::new(mol.as_molgraph_mut())
            .prepare_leaving(oxygen, "h", true)
            .unwrap();

        assert_eq!(first, second);
        // Every charge and every stashed pre-fold charge, wherever it lives,
        // survives a repeat call untouched.
        assert_eq!(charges(&mol), after_first);
        assert!((total_charge(&mol) - total).abs() < 1e-12);
    }

    #[test]
    fn an_integer_charge_is_folded_like_a_float() {
        let (mut mol, oxygen) = charged_water(Some(PropValue::Int(-2)), Some(PropValue::Int(1)));
        let before = total_charge(&mol);
        let leaving = SiteMap::new(mol.as_molgraph_mut())
            .prepare_leaving(oxygen, "h", true)
            .unwrap();

        let site = mol.as_molgraph().get_node(oxygen).unwrap();
        let hydrogen = mol.as_molgraph().get_node(leaving).unwrap();
        let site_charge = site.get(keys::CHARGE).and_then(PropValue::as_f64).unwrap();
        let hydrogen_charge = hydrogen
            .get(keys::CHARGE)
            .and_then(PropValue::as_f64)
            .unwrap();
        assert!(
            (site_charge - (-1.0)).abs() < 1e-12,
            "site charge {site_charge}"
        );
        assert!(
            hydrogen_charge.abs() < 1e-12,
            "leaving charge {hydrogen_charge}"
        );
        assert!((total_charge(&mol) - before).abs() < 1e-12);
    }

    #[test]
    fn folding_a_charge_only_one_side_carries_is_refused() {
        for (oxygen, hydrogen) in [
            (Some(PropValue::F64(-0.8)), None),
            (None, Some(PropValue::F64(0.4))),
        ] {
            let label = format!("site {oxygen:?}, hydrogen {hydrogen:?}");
            let (mut mol, site) = charged_water(oxygen, hydrogen);
            let error = SiteMap::new(mol.as_molgraph_mut())
                .prepare_leaving(site, "h", true)
                .expect_err(&format!("a one-sided fold cannot conserve charge: {label}"));
            assert_ne!(
                error,
                SiteError::NoLeavingHydrogen { node: site },
                "{label}"
            );
        }
    }

    #[test]
    fn a_site_atom_without_a_hydrogen_is_refused() {
        let mut mol = Atomistic::new();
        let carbon = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let error = SiteMap::new(mol.as_molgraph_mut())
            .prepare_leaving(carbon, "h", true)
            .expect_err("a bare carbon has no hydrogen to leave");
        assert_eq!(error, SiteError::NoLeavingHydrogen { node: carbon });
    }

    #[test]
    fn every_nth_marks_at_the_stride() {
        let mut mol = Atomistic::new();
        let nodes: Vec<NodeId> = (0..5)
            .map(|i| mol.add_atom_xyz("C", i as f64, 0.0, 0.0))
            .collect();
        let marked = SiteMap::new(mol.as_molgraph_mut())
            .every_nth(&nodes, 2, "x", None, true)
            .unwrap();
        assert_eq!(marked.len(), 3);
        assert_eq!(marked, vec![nodes[0], nodes[2], nodes[4]]);
        SiteMap::new(mol.as_molgraph_mut()).clear(None).unwrap();
        let cleared = mol
            .as_molgraph()
            .get_node(nodes[0])
            .unwrap()
            .get_str(SITE_KEY)
            .map(str::to_string);
        assert_eq!(cleared.as_deref(), Some(""));
    }
}
