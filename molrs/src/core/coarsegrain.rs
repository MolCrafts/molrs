//! Coarse-grained molecular graph: [`CoarseGrain`].

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use ndarray::Array1;
use slotmap::Key;

use crate::core::Block;
use crate::core::Frame;
use crate::core::MolRsError;
use crate::core::keys;

use crate::core::EntityTable;
use crate::core::{Atom, KindId, MolGraph, NodeId, Relation, RelationId};
use crate::op::Idx;

/// Result of [`CoarseGrain::extract_subgraph`].
#[derive(Debug, Clone)]
pub struct ExtractedCoarseGrain {
    /// The extracted bead graph (membership copied for selected beads).
    pub graph: CoarseGrain,
    /// Selected parent beads with a CG-bond neighbour outside the ball.
    pub boundary: Vec<NodeId>,
    /// New bead id → parent bead id.
    pub parent_of: HashMap<NodeId, NodeId>,
    /// Parent bead id → hops (CG bonds) from the nearest center.
    pub hops: HashMap<NodeId, i64>,
    /// Parent bead id → new bead id.
    pub node_map: HashMap<NodeId, NodeId>,
}

/// Coarse-grained molecular graph.
///
/// A *coarse-grained* (CG) model describes a molecule with fewer particles
/// than it has atoms: each particle, a **bead**, stands for a group of atoms
/// (a monomer, a few CH₂ units, a water cluster) and sits at a representative
/// position of that group. A **CG bond** says two beads are connected; it has
/// no chemical bond order.
///
/// `CoarseGrain` wraps the domain-agnostic [`MolGraph`] where every node is a
/// bead. It registers its own `bonds` kind and exposes the
/// bead / CG-bond vocabulary; `MolGraph` itself stays chemistry-agnostic.
/// Generic graph methods (`nodes`, `neighbors`, …) remain available via
/// `Deref`/`DerefMut`.
///
/// Invariant: every node has a `"bead_type"` property.
///
/// A bead additionally owns a **membership**: the set of underlying atom handles
/// it groups. Membership is variable-size directed ownership across worlds (the
/// atoms live in a separate all-atom world), so it is **not** a fixed-arity peer
/// relation and is stored here as opaque atom handles keyed by bead, not as a
/// scalar component. Resolving a handle back to an atom view is the caller's job
/// (it owns the source world); this layer owns only the handle topology.
///
/// # Examples
///
/// ```
/// use molrs::core::CoarseGrain;
///
/// let mut cg = CoarseGrain::new();
/// let b1 = cg.add_bead("W", 0.0, 0.0, 0.0);
/// let b2 = cg.add_bead("W", 3.0, 0.0, 0.0);
/// cg.add_bond(b1, b2).unwrap();
///
/// assert_eq!(cg.n_beads(), 2);
/// assert_eq!(cg.n_bonds(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct CoarseGrain {
    graph: MolGraph,
    bond: KindId,
    members: HashMap<NodeId, Vec<u64>>,
}

impl Deref for CoarseGrain {
    type Target = MolGraph;
    fn deref(&self) -> &MolGraph {
        &self.graph
    }
}

impl DerefMut for CoarseGrain {
    fn deref_mut(&mut self) -> &mut MolGraph {
        &mut self.graph
    }
}

impl Default for CoarseGrain {
    fn default() -> Self {
        Self::new()
    }
}

impl CoarseGrain {
    /// Create an empty coarse-grained molecular graph with the CG `bonds` kind
    /// registered.
    pub fn new() -> Self {
        let mut graph = MolGraph::new();
        let bond = graph.register_kind("bonds", 2);
        Self {
            graph,
            bond,
            members: HashMap::new(),
        }
    }

    /// Add a bead with type name and 3D coordinates.
    ///
    /// # Panics
    ///
    /// Panics when a value of the bag this builds contradicts the element
    /// type an existing bead column holds for that key (a string `x` where
    /// `x` is an `f64` column). The bag is built here out of typed arguments,
    /// so that is a defect in the graph's own vocabulary and not a data
    /// condition; the generic
    /// [`MolGraph::add_node_with`](crate::core::MolGraph::add_node_with)
    /// reached through [`as_molgraph_mut`](Self::as_molgraph_mut) returns the
    /// conflict for callers holding a foreign bag.
    pub fn add_bead(&mut self, bead_type: &str, x: f64, y: f64, z: f64) -> NodeId {
        let mut a = Atom::new();
        a.set(keys::BEAD_TYPE, bead_type);
        a.set(keys::X, x);
        a.set(keys::Y, y);
        a.set(keys::Z, z);
        self.graph
            .add_node_with(a)
            .expect("caller-built bead bag contradicts an existing bead column")
    }

    /// Add a bead with type name only (no coordinates).
    ///
    /// # Panics
    ///
    /// Panics when the `bead_type` column already holds a different element
    /// type — see [`add_bead`](Self::add_bead).
    pub fn add_bead_bare(&mut self, bead_type: &str) -> NodeId {
        let mut a = Atom::new();
        a.set(keys::BEAD_TYPE, bead_type);
        self.graph
            .add_node_with(a)
            .expect("caller-built bead bag contradicts an existing bead column")
    }

    /// Remove a bead and all incident CG bonds (and its membership).
    pub fn remove_bead(&mut self, id: NodeId) -> Result<Atom, MolRsError> {
        self.members.remove(&id);
        self.graph.remove_node(id)
    }

    // ---- bead → atom membership (opaque foreign atom handles) ----

    /// Set the atom handles a bead groups (replaces any existing membership).
    /// An empty slice clears the membership.
    pub fn set_bead_members(&mut self, bead: NodeId, atoms: Vec<u64>) {
        if atoms.is_empty() {
            self.members.remove(&bead);
        } else {
            self.members.insert(bead, atoms);
        }
    }

    /// The atom handles a bead groups (empty if none recorded).
    pub fn bead_members(&self, bead: NodeId) -> &[u64] {
        self.members.get(&bead).map_or(&[], Vec::as_slice)
    }

    /// Beads whose membership includes `atom`, in bead-handle order.
    pub fn beads_of_atom(&self, atom: u64) -> Vec<NodeId> {
        let mut out: Vec<NodeId> = self
            .members
            .iter()
            .filter(|(_, atoms)| atoms.contains(&atom))
            .map(|(&bead, _)| bead)
            .collect();
        out.sort_by_key(|b| b.data().as_ffi());
        out
    }

    /// Materialize a bead's property bag (owned copy of its set components).
    pub fn get_bead(&self, id: NodeId) -> Result<Atom, MolRsError> {
        self.graph.get_node(id)
    }

    /// Iterate over all `(NodeId, Atom)` pairs (each property bag materialized).
    pub fn beads(&self) -> impl Iterator<Item = (NodeId, Atom)> + '_ {
        self.graph.nodes()
    }

    /// Number of beads.
    pub fn n_beads(&self) -> usize {
        self.graph.n_nodes()
    }

    /// The positions `[x, y, z]` of `beads`, in Å as stored, in the order of
    /// `beads` (a bead listed twice appears twice). O(k) for k listed beads;
    /// each coordinate column is looked up once.
    ///
    /// # Errors
    ///
    /// Reporting the first offender in slice order:
    ///
    /// - [`MolRsError::NotFound`] when a handle is stale or belongs to
    ///   another graph;
    /// - [`MolRsError::Validation`] when a bead's `x`, `y` or `z` is missing or
    ///   not finite, naming the bead by its integer handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::CoarseGrain;
    ///
    /// let mut cg = CoarseGrain::new();
    /// let a = cg.add_bead("W", 0.0, 1.0, 2.0);
    /// let b = cg.add_bead("P1", 3.0, 4.0, 5.0);
    /// assert_eq!(cg.positions(&[b, a])?, vec![[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]);
    /// # Ok::<(), molrs::core::MolRsError>(())
    /// ```
    pub fn positions(&self, beads: &[NodeId]) -> Result<Vec<[f64; 3]>, MolRsError> {
        self.vectors(beads, keys::COORDS, "coordinate")
    }

    /// The site axes `[axis_x, axis_y, axis_z]` of `beads`, in Å as stored,
    /// in the order of `beads`. A site made by
    /// `builder::Coarsener::coarsen` carries
    /// the vector from the first member of its group to the site, which
    /// fixes the site's direction; a one-member site's axis is zero. O(k) for
    /// k listed beads.
    ///
    /// # Errors
    ///
    /// As [`positions`](Self::positions), for the axis columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::keys;
    /// use molrs::core::CoarseGrain;
    ///
    /// let mut cg = CoarseGrain::new();
    /// let a = cg.add_bead("S", 0.0, 0.0, 0.0);
    /// for (key, v) in keys::AXIS.into_iter().zip([1.0, 2.0, 3.0]) {
    ///     cg.set_node(a, key, v)?;
    /// }
    /// assert_eq!(cg.axes(&[a])?, vec![[1.0, 2.0, 3.0]]);
    /// # Ok::<(), molrs::core::MolRsError>(())
    /// ```
    pub fn axes(&self, beads: &[NodeId]) -> Result<Vec<[f64; 3]>, MolRsError> {
        self.vectors(beads, keys::AXIS, "axis")
    }

    /// The three f64 columns `columns` of `beads`, each finite.
    fn vectors(
        &self,
        beads: &[NodeId],
        columns: [&str; 3],
        what: &str,
    ) -> Result<Vec<[f64; 3]>, MolRsError> {
        let table = self.graph.node_table();
        // An absent (or non-f64) column reads as "missing" for every bead.
        let columns = columns.map(|key| (key, table.column_f64(key).ok()));
        beads
            .iter()
            .map(|&bead| {
                let row = Self::bead_row(table, bead)?;
                let mut point = [0.0; 3];
                for (value, (key, column)) in point.iter_mut().zip(&columns) {
                    *value = match column {
                        Some((data, valid)) if valid.get(row) && data[row].is_finite() => data[row],
                        _ => {
                            return Err(MolRsError::validation(format!(
                                "bead {} has no finite '{key}' {what}",
                                bead.data().as_ffi()
                            )));
                        }
                    };
                }
                Ok(point)
            })
            .collect()
    }

    /// The `bead_type` of each of `beads`, in the order of `beads` (a bead
    /// listed twice appears twice). O(k) for k listed beads; the column is
    /// looked up once.
    ///
    /// # Errors
    ///
    /// Reporting the first offender in slice order:
    ///
    /// - [`MolRsError::NotFound`] when a handle is stale or belongs to
    ///   another graph;
    /// - [`MolRsError::Validation`] when a bead's `bead_type` was cleared
    ///   through the inner graph, naming the bead by its integer handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::CoarseGrain;
    ///
    /// let mut cg = CoarseGrain::new();
    /// let a = cg.add_bead("W", 0.0, 0.0, 0.0);
    /// let b = cg.add_bead("P1", 1.0, 0.0, 0.0);
    /// assert_eq!(cg.bead_types(&[b, a, b])?, ["P1", "W", "P1"]);
    /// # Ok::<(), molrs::core::MolRsError>(())
    /// ```
    pub fn bead_types(&self, beads: &[NodeId]) -> Result<Vec<String>, MolRsError> {
        let table = self.graph.node_table();
        let column = table.column_str(keys::BEAD_TYPE).ok();
        beads
            .iter()
            .map(|&bead| {
                let row = Self::bead_row(table, bead)?;
                match column {
                    Some((data, valid)) if valid.get(row) => Ok(data[row].clone()),
                    _ => Err(MolRsError::validation(format!(
                        "bead {} carries no '{}'",
                        bead.data().as_ffi(),
                        keys::BEAD_TYPE
                    ))),
                }
            })
            .collect()
    }

    /// The row of `bead` in the node table, or `NotFound` naming its integer
    /// handle.
    fn bead_row(table: &EntityTable<NodeId>, bead: NodeId) -> Result<usize, MolRsError> {
        table
            .row(bead)
            .ok_or_else(|| MolRsError::not_found("bead", format!("bead {}", bead.data().as_ffi())))
    }

    /// Add a CG bond between two existing beads.
    pub fn add_bond(&mut self, a: NodeId, b: NodeId) -> Result<RelationId, MolRsError> {
        self.graph.add_relation(self.bond, &[a, b])
    }

    /// Materialize a CG bond (endpoints + properties).
    pub fn get_bond(&self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.get_relation(self.bond, id)
    }

    /// Iterate over all `(RelationId, Relation)` pairs (each materialized).
    pub fn bonds(&self) -> impl Iterator<Item = (RelationId, Relation)> + '_ {
        self.graph.relations(self.bond)
    }

    /// Number of CG bonds.
    pub fn n_bonds(&self) -> usize {
        self.graph.n_relations(self.bond)
    }

    /// Export to a tabular [`Frame`] in the canonical vocabulary every graph
    /// type shares: one `atoms` row per bead, one `bonds` row per CG bond
    /// (`atomi` / `atomj` = the endpoint beads' `atoms` rows), plus a
    /// `members` block for bead membership. The frame validates canonically.
    ///
    /// `members` has one row per (bead, atom) pair: `ibead` (UInt, the bead's
    /// row in `atoms`) and `atom` (UInt, the opaque atom handle), grouped by
    /// bead in `atoms` row order and keeping each bead's member order. The
    /// block is absent when no bead has members.
    ///
    /// # Known limit
    ///
    /// Every property a bead carries is written, `mol_id` included, while
    /// [`Self::from_frame`] reads exactly one molecule. So
    /// `from_frame(to_frame(cg))` refuses a CoarseGrain whose beads carry two
    /// or more distinct `mol_id` values; select one molecule with
    /// [`Frame::subset`] first.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a bead or bond property contradicts the
    /// dtype the Frame schema declares for its key; the message names the
    /// refused column. The inner [`MolGraph`] accepts any value under a key it
    /// has no column for, so a string written under `"x"` is legal in the
    /// graph and only refused here.
    pub fn to_frame(&self) -> Result<Frame, MolRsError> {
        let mut frame = self.graph.to_frame()?;

        // `node_ids` is the row order the graph just wrote `atoms` in.
        let mut ibead: Vec<Idx> = Vec::new();
        let mut atom: Vec<Idx> = Vec::new();
        for (row, id) in self.graph.node_ids().enumerate() {
            for &handle in self.bead_members(id) {
                ibead.push(row as Idx);
                atom.push(handle);
            }
        }
        if !ibead.is_empty() {
            let mut members = Block::new();
            members
                .insert("ibead", Array1::from_vec(ibead).into_dyn())
                .and_then(|()| members.insert("atom", Array1::from_vec(atom).into_dyn()))
                // `ibead` indexes this frame's bead rows; stated, not only
                // conventional. `atom` is a handle into the all-atom side,
                // which only the caller can name as a target.
                .and_then(|()| members.set_target("ibead", "atoms"))
                .map_err(|e| MolRsError::validation(format!("Frame 'members' block: {e}")))?;
            frame.insert("members", members);
        }
        Ok(frame)
    }

    /// Build from a [`Frame`] that holds **one molecule** in the canonical
    /// `atoms` / `bonds` vocabulary — the frame [`Self::to_frame`] writes,
    /// or one molecule of a LAMMPS data frame.
    ///
    /// Every `atoms` row becomes one bead and every `bonds` row one CG bond
    /// (`atomi` / `atomj` are 0-based `atoms` rows). Every `atoms` column is
    /// copied onto its bead and every `bonds` column onto its bond.
    ///
    /// `bead_type` comes from the first of these `atoms` columns present:
    ///
    /// 1. `bead_type` (Str), used as is;
    /// 2. `type` (Str), copied;
    /// 3. `type_id` (UInt), rendered in decimal.
    ///
    /// The source column also stays on each bead as a plain property. A null
    /// row in the chosen column is an error; it never falls through to the
    /// next source.
    ///
    /// A frame of many molecules is refused: when `atoms` carries `mol_id`,
    /// every row must hold the same value. Select one molecule with
    /// [`Frame::subset`] first. A frame without `mol_id` is one molecule.
    ///
    /// Only `atoms`, `bonds` and `members` are read. `angles`, `dihedrals`
    /// and every other block are ignored, so a CG relation kind other than
    /// `bonds` does not survive a round trip. A `members` block (see
    /// [`Self::to_frame`]) is read after the graph: row `ibead` of `atoms`
    /// gains atom handle `atom`, in block row order. The caller's frame is
    /// never mutated.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when the frame has no `atoms` rows; when
    /// `mol_id` has a null row, is not UInt, or holds two distinct values
    /// (naming both and `Frame::subset`); when none of `bead_type` / `type` /
    /// `type_id` is present, or the chosen one has a null row or the wrong
    /// dtype; when the `members` block lacks a UInt `ibead` or `atom` column,
    /// an `ibead` names no bead row, or an `(ibead, atom)` row repeats; when a
    /// column of `atoms` or `bonds` is not 1-D (an `(N, 3)` coordinate column,
    /// say), naming the block and the column — plus every other error of the
    /// graph read itself (e.g. a bond endpoint past the atom rows).
    pub fn from_frame(frame: &Frame) -> Result<Self, MolRsError> {
        let atoms = frame
            .get("atoms")
            .filter(|a| a.n_rows().unwrap_or(0) > 0)
            .ok_or_else(|| {
                MolRsError::validation("Frame has no 'atoms' rows to read beads from")
            })?;

        if atoms.contains_key(keys::MOL_ID) {
            if atoms.validity(keys::MOL_ID).is_some() {
                return Err(MolRsError::validation(
                    "Frame 'atoms' column 'mol_id' has a null row, so a bead belongs to no \
                     molecule",
                ));
            }
            let mol_id = atoms
                .get(keys::MOL_ID)
                .and_then(|c| c.as_uint())
                .ok_or_else(|| {
                    MolRsError::validation("Frame 'atoms' column 'mol_id' is not UInt")
                })?;
            let mut ids = mol_id.iter();
            if let Some(&first) = ids.next()
                && let Some(&other) = ids.find(|&&id| id != first)
            {
                return Err(MolRsError::validation(format!(
                    "Frame 'atoms' holds more than one molecule (mol_id {first} and \
                     {other}); select one molecule with `Frame::subset` first"
                )));
            }
        }

        let sources = [keys::BEAD_TYPE, keys::TYPE, keys::TYPE_ID];
        let Some(source) = sources.into_iter().find(|k| atoms.contains_key(k)) else {
            return Err(MolRsError::validation(format!(
                "Frame 'atoms' block has none of the columns {} to take bead_type from",
                sources.map(|k| format!("'{k}'")).join(", ")
            )));
        };
        if atoms.validity(source).is_some() {
            return Err(MolRsError::validation(format!(
                "Frame 'atoms' column '{source}' has a null row, so a bead would have no \
                 bead_type"
            )));
        }
        let wrong_dtype = |expected: &str| {
            MolRsError::validation(format!(
                "Frame 'atoms' column '{source}' is {}, not {expected}",
                atoms.dtype(source).map_or("unknown", |d| d.name())
            ))
        };
        // `None`: the source already is a Str `bead_type`.
        let bead_type: Option<Vec<String>> = if source == keys::BEAD_TYPE {
            atoms
                .get(source)
                .and_then(|c| c.as_string())
                .ok_or_else(|| wrong_dtype("Str"))?;
            None
        } else if source == keys::TYPE {
            let col = atoms
                .get(source)
                .and_then(|c| c.as_string())
                .ok_or_else(|| wrong_dtype("Str"))?;
            Some(col.iter().cloned().collect())
        } else {
            let col = atoms
                .get(source)
                .and_then(|c| c.as_uint())
                .ok_or_else(|| wrong_dtype("UInt"))?;
            Some(col.iter().map(ToString::to_string).collect())
        };

        let mut beads = atoms.clone();
        if let Some(bead_type) = bead_type {
            beads
                .insert(keys::BEAD_TYPE, Array1::from_vec(bead_type).into_dyn())
                .map_err(|e| {
                    MolRsError::validation(format!("Frame 'atoms' column '{source}': {e}"))
                })?;
        }
        let mut canonical = Frame::new();
        canonical.insert("atoms", beads);
        if let Some(bonds) = frame.get("bonds") {
            canonical.insert("bonds", bonds.clone());
        }
        let mut cg = Self::from_canonical_frame(&canonical)?;
        if let Some(members) = frame.get("members") {
            cg.attach_members(members)?;
        }
        Ok(cg)
    }

    /// Read a frame already in [`MolGraph`]'s canonical vocabulary (`atoms` /
    /// `bonds`) into a validated `CoarseGrain`: every bead must carry
    /// `bead_type`, as [`Self::try_from_molgraph`] requires.
    fn from_canonical_frame(canonical: &Frame) -> Result<Self, MolRsError> {
        let mut graph = MolGraph::new();
        graph.register_kind("bonds", 2);
        graph.extend_from_frame(canonical)?;
        Self::try_from_molgraph(graph)
    }

    /// Attach the membership a `members` block states: row `r` gives bead row
    /// `ibead[r]` the atom handle `atom[r]`. Bead rows are the graph's row
    /// order, which a freshly read graph shares with its `atoms` block. A
    /// repeated `(ibead, atom)` row is refused, since it would list the atom
    /// twice in its bead.
    fn attach_members(&mut self, members: &Block) -> Result<(), MolRsError> {
        let column = |key: &str| {
            members.get(key).and_then(|c| c.as_uint()).ok_or_else(|| {
                MolRsError::validation(format!("Frame 'members' block has no UInt '{key}' column"))
            })
        };
        let ibead = column("ibead")?;
        let atom = column("atom")?;
        let bead_ids: Vec<NodeId> = self.graph.node_ids().collect();
        for (row, (&bead_row, &handle)) in ibead.iter().zip(atom.iter()).enumerate() {
            let Some(&bead) = bead_ids.get(bead_row as usize) else {
                return Err(MolRsError::validation(format!(
                    "Frame 'members' row {row} names bead {bead_row}, past the {} beads",
                    bead_ids.len()
                )));
            };
            let listed = self.members.entry(bead).or_default();
            if listed.contains(&handle) {
                return Err(MolRsError::validation(format!(
                    "Frame 'members' row {row} repeats atom {handle} of bead {bead_row}"
                )));
            }
            listed.push(handle);
        }
        Ok(())
    }

    /// Promote from a [`MolGraph`], validating all nodes have `"bead_type"`.
    ///
    /// The CG `bonds` kind is re-registered **by name**: an existing `bonds`
    /// kind of arity 2 keeps its id, and a missing one is registered fresh.
    /// Resolving by dense id instead would report a foreign kind's relations as
    /// this system's CG bonds whenever the graph registered something else
    /// first, and the next `add_bond` would write into it.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] when a node carries no `"bead_type"`,
    /// and when the graph already spells `bonds` at an arity other than 2
    /// (naming the kind and both arities).
    pub fn try_from_molgraph(mut mol: MolGraph) -> Result<Self, MolRsError> {
        for (id, atom) in mol.nodes() {
            if atom.get_str(keys::BEAD_TYPE).is_none() {
                return Err(MolRsError::validation(format!(
                    "node {:?} missing 'bead_type' property",
                    id
                )));
            }
        }
        let bond = mol.try_register_kind("bonds", 2)?;
        Ok(Self {
            graph: mol,
            bond,
            members: HashMap::new(),
        })
    }

    /// Unwrap to the inner [`MolGraph`] (zero cost).
    pub fn into_inner(self) -> MolGraph {
        self.graph
    }

    /// Borrow the inner [`MolGraph`].
    pub fn as_molgraph(&self) -> &MolGraph {
        &self.graph
    }

    /// Mutably borrow the inner [`MolGraph`].
    pub fn as_molgraph_mut(&mut self) -> &mut MolGraph {
        &mut self.graph
    }

    /// Place `transforms.len()` rigid copies of `template`, copy `c` moved by
    /// `transforms[c]` (rotation, then a translation in Å) and stamped
    /// `frag_id = frag_ids[c]`; returns the new beads copy-major. Column-wise
    /// and atomic — see [`MolGraph::replicate`].
    ///
    /// **No bead membership is copied.** Membership names foreign atoms of a
    /// separate all-atom world, and the copies do not own them; every
    /// replicated bead starts with an empty membership.
    ///
    /// # Errors
    ///
    /// The errors of [`MolGraph::replicate`]; `self` is unchanged then.
    pub fn replicate(
        &mut self,
        template: &CoarseGrain,
        transforms: &[crate::op::Rigid],
        frag_ids: &[crate::op::I],
    ) -> Result<Vec<NodeId>, MolRsError> {
        self.graph.replicate(&template.graph, transforms, frag_ids)
    }

    // ---- subgraph extraction / composition ----

    /// Induced subgraph on an explicit bead set. Stale handles fail-fast.
    pub fn induced_subgraph(
        &self,
        beads: &[NodeId],
    ) -> Result<(CoarseGrain, HashMap<NodeId, NodeId>), MolRsError> {
        let induced = self.graph.induced_subgraph(beads)?;
        let mut cg = CoarseGrain::try_from_molgraph(induced.graph)?;
        for (&old, &new) in &induced.node_map {
            let mem = self.bead_members(old);
            if !mem.is_empty() {
                cg.set_bead_members(new, mem.to_vec());
            }
        }
        Ok((cg, induced.node_map))
    }

    /// Radius ball around `centers` over CG bonds. Membership (opaque foreign
    /// atom handles) is copied for selected beads unchanged.
    pub fn extract_subgraph(
        &self,
        centers: &[NodeId],
        radius: i64,
    ) -> Result<ExtractedCoarseGrain, MolRsError> {
        let ball = self.graph.extract_ball(
            centers,
            radius,
            self.bond,
            /* copy_higher_order */ true,
            /* whole_groups */ &[],
        )?;
        let mut cg = CoarseGrain::try_from_molgraph(ball.graph)?;
        for (&old, &new) in &ball.node_map {
            let mem = self.bead_members(old);
            if !mem.is_empty() {
                cg.set_bead_members(new, mem.to_vec());
            }
        }
        Ok(ExtractedCoarseGrain {
            graph: cg,
            boundary: ball.boundary,
            parent_of: ball.parent_of,
            hops: ball.hops,
            node_map: ball.node_map,
        })
    }

    /// Structural merge. Returns `handle in other → handle in self`. Remaps bead
    /// membership keys; foreign atom handles inside membership stay as-is.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a bead or relation property of `other`
    /// contradicts the element type `self` holds for that key — see
    /// [`MolGraph::merge`](crate::core::MolGraph::merge), whose
    /// partial-write contract this inherits.
    pub fn merge(&mut self, other: CoarseGrain) -> Result<HashMap<NodeId, NodeId>, MolRsError> {
        let node_map = self.graph.merge(other.graph)?;
        for (old_bead, members) in other.members {
            if let Some(&new_bead) = node_map.get(&old_bead)
                && !members.is_empty()
            {
                self.members.insert(new_bead, members);
            }
        }
        Ok(node_map)
    }

    /// Independent deep copy. **Handles are preserved**.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    // ---- structural graph hash (see [`crate::core::graph_hash`]) ----

    /// Isomorphism-invariant Weisfeiler–Lehman structural hash of the bead graph
    /// (bead-type node labels, bond-order edge labels). Shares the same
    /// [`MolGraph`] primitive that serves the all-atom case.
    pub fn structural_hash(&self) -> u64 {
        crate::core::structural_hash(&self.graph)
    }

    /// Deterministic canonical bead ordering from the WL refinement (see
    /// [`crate::core::canonical_order`]).
    pub fn canonical_order(&self) -> Vec<NodeId> {
        crate::core::canonical_order(&self.graph)
    }

    /// Whether `self` and `other` are isomorphic as labeled bead graphs.
    pub fn is_isomorphic(&self, other: &CoarseGrain) -> bool {
        crate::core::is_isomorphic(&self.graph, &other.graph)
    }
}

impl crate::core::FromMolGraph for CoarseGrain {
    fn from_molgraph(graph: MolGraph) -> Result<Self, MolRsError> {
        CoarseGrain::try_from_molgraph(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_add_bead() {
        let mut cg = CoarseGrain::new();
        let b1 = cg.add_bead("W", 0.0, 0.0, 0.0);
        let b2 = cg.add_bead("P1", 3.0, 0.0, 0.0);
        cg.add_bond(b1, b2).unwrap();
        assert_eq!(cg.n_beads(), 2);
        assert_eq!(cg.n_bonds(), 1);
    }

    #[test]
    fn test_bead_has_type() {
        let mut cg = CoarseGrain::new();
        let b = cg.add_bead("W", 1.0, 2.0, 3.0);
        let bead = cg.get_bead(b).unwrap();
        assert_eq!(bead.get_str("bead_type"), Some("W"));
        assert_eq!(bead.get_f64("x"), Some(1.0));
    }

    #[test]
    fn to_frame_writes_atoms_and_bonds_that_validate() {
        let mut cg = CoarseGrain::new();
        let a = cg.add_bead("W", 0.0, 0.0, 0.0);
        let b = cg.add_bead("P1", 3.0, 0.0, 0.0);
        cg.add_bond(a, b).unwrap();

        let frame = cg.to_frame().expect("a schema-conforming graph converts");
        let mut names: Vec<&str> = frame.keys().collect();
        names.sort_unstable();
        assert_eq!(names, vec!["atoms", "bonds"]);
        let bead_type: Vec<String> = frame["atoms"]
            .get("bead_type")
            .and_then(|c| c.as_string())
            .expect("atoms carries a Str bead_type")
            .iter()
            .cloned()
            .collect();
        assert_eq!(bead_type, vec!["W", "P1"]);
        let endpoint = |col: &str| -> Vec<Idx> {
            frame["bonds"]
                .get(col)
                .and_then(|c| c.as_uint())
                .expect("bonds carries UInt endpoints")
                .iter()
                .copied()
                .collect()
        };
        assert_eq!(endpoint("atomi"), vec![0]);
        assert_eq!(endpoint("atomj"), vec![1]);
        frame.validate().expect("the CG frame is a canonical frame");

        let restored = CoarseGrain::from_frame(&frame).expect("CG frame round-trip");
        assert_eq!(restored.n_beads(), 2);
        assert_eq!(restored.n_bonds(), 1);
    }

    /// A `CoarseGrain` owns no bead setter of its own, so the door a caller
    /// reaches is the inner graph's `set_node` — and it carries the same
    /// schema opinion there: a string under the float key `charge` never
    /// becomes a bead property, so no bead column can contradict the Frame
    /// schema in the first place.
    #[test]
    fn set_node_through_the_inner_graph_refuses_a_str_under_a_schema_float_key() {
        use crate::core::DType;
        let mut cg = CoarseGrain::new();
        let b = cg.add_bead("W", 0.0, 0.0, 0.0);

        let err = cg
            .as_molgraph_mut()
            .set_node(b, "charge", "negative")
            .expect_err("a str cannot be stored at a schema-float key");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("'charge'"),
            "the error names the key, got {msg}"
        );
        assert!(
            msg.contains(DType::Float.name()) && msg.contains(DType::String.name()),
            "the error names both dtypes, got {msg}"
        );
    }

    #[test]
    fn test_try_from_molgraph_missing_bead_type() {
        let mut g = MolGraph::new();
        g.register_kind("bonds", 2);
        g.add_node_with(Atom::new()).expect("fixture node");
        assert!(CoarseGrain::try_from_molgraph(g).is_err());
    }

    /// A graph whose *first* registered kind is not `bonds` — what a fragment
    /// graph looks like — must not have that foreign kind's relations reported
    /// as its CG bonds, nor be written into by the next `add_bond`.
    #[test]
    fn try_from_molgraph_resolves_bonds_by_name_not_kind_zero() {
        let mut graph = MolGraph::new();
        let ports = graph.register_kind("ports", 2);
        let mut bead = Atom::new();
        bead.set("bead_type", "W");
        let b1 = graph.add_node_with(bead.clone()).expect("fixture node");
        let b2 = graph.add_node_with(bead).expect("fixture node");
        graph.add_relation(ports, &[b1, b2]).unwrap();

        let mut cg = CoarseGrain::try_from_molgraph(graph).expect("bead types are present");
        let ports = cg.kind_id("ports").expect("the foreign kind survives");
        assert_eq!(cg.n_bonds(), 0, "a port relation is not a CG bond");

        cg.add_bond(b1, b2).expect("a bond can still be added");
        assert_eq!(cg.n_bonds(), 1);
        assert_eq!(
            cg.n_relations(ports),
            1,
            "add_bond must not write into the foreign kind"
        );
    }

    /// A caller-supplied graph that spells `bonds` at another arity is a data
    /// condition, so the promotion returns an error instead of aborting the
    /// process inside `register_kind`.
    #[test]
    fn try_from_molgraph_rejects_conflicting_arity() {
        let mut graph = MolGraph::new();
        graph.register_kind("bonds", 3);
        let err = CoarseGrain::try_from_molgraph(graph)
            .expect_err("a 3-ary 'bonds' kind conflicts with the CG bond kind");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn test_into_inner() {
        let mut cg = CoarseGrain::new();
        cg.add_bead("W", 0.0, 0.0, 0.0);
        let g: MolGraph = cg.into_inner();
        assert_eq!(g.n_nodes(), 1);
    }

    #[test]
    fn scale_multiplies_offsets_from_the_centre_per_axis() {
        // p' = (p - c) * f + c. Point (1, 2, 3), factor (2, 3, 0.5):
        //   about c = (1, 1, 1) -> (1, 4, 2);  about the origin -> (2, 6, 1.5).
        let factor = [2.0, 3.0, 0.5];
        let cases: [(Option<[f64; 3]>, [f64; 3]); 2] = [
            (Some([1.0, 1.0, 1.0]), [1.0, 4.0, 2.0]),
            (None, [2.0, 6.0, 1.5]),
        ];
        for (about, expected) in cases {
            let mut sys = CoarseGrain::new();
            let id = sys.add_bead("W", 1.0, 2.0, 3.0);
            let fixed = sys.add_bead("W", 1.0, 1.0, 1.0);
            sys.as_molgraph_mut().scale(factor, about);
            let moved = sys.get_bead(id).expect("live handle");
            for (key, want) in ["x", "y", "z"].into_iter().zip(expected) {
                let got = moved.get_f64(key).expect("coordinate kept");
                assert!(
                    (got - want).abs() < 1e-12,
                    "{about:?} {key}: {got} != {want}"
                );
            }
            if about.is_some() {
                // The centre itself is a fixed point of the map.
                let centre = sys.get_bead(fixed).expect("live handle");
                for key in ["x", "y", "z"] {
                    let got = centre.get_f64(key).expect("coordinate kept");
                    assert!((got - 1.0).abs() < 1e-12, "centre {key} moved to {got}");
                }
            }
        }
    }

    // ---- from_frame: atoms/bonds vocabulary, validation, membership ----

    use crate::core::Block;
    use ndarray::Array1;

    fn float_col(values: &[f64]) -> ndarray::ArrayD<f64> {
        Array1::from_vec(values.to_vec()).into_dyn()
    }

    fn uint_col(values: &[u64]) -> ndarray::ArrayD<u64> {
        Array1::from_vec(values.to_vec()).into_dyn()
    }

    fn str_col(values: &[&str]) -> ndarray::ArrayD<String> {
        Array1::from_vec(values.iter().map(|s| (*s).to_owned()).collect()).into_dyn()
    }

    /// An `atoms` block of `n` rows with x/y/z only; the caller adds the
    /// bead-type source under test.
    fn xyz_atoms(x: &[f64], y: &[f64], z: &[f64]) -> Block {
        let mut atoms = Block::new();
        atoms.insert("x", float_col(x)).unwrap();
        atoms.insert("y", float_col(y)).unwrap();
        atoms.insert("z", float_col(z)).unwrap();
        atoms
    }

    fn frame_of(atoms: Block) -> Frame {
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame
    }

    fn bead_types(cg: &CoarseGrain) -> Vec<String> {
        cg.beads()
            .map(|(_, bead)| {
                bead.get_str("bead_type")
                    .expect("every bead carries bead_type")
                    .to_owned()
            })
            .collect()
    }

    #[test]
    fn from_frame_takes_bead_type_before_type() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.0; 2], &[0.0; 2]);
        atoms.insert("bead_type", str_col(&["W", "W"])).unwrap();
        atoms.insert("type", str_col(&["CT", "OH"])).unwrap();

        let cg = CoarseGrain::from_frame(&frame_of(atoms)).expect("a bead_type column");
        assert_eq!(bead_types(&cg), vec!["W", "W"]);
    }

    #[test]
    fn from_frame_takes_type_before_type_id() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.0; 2], &[0.0; 2]);
        atoms.insert("type", str_col(&["CT", "OH"])).unwrap();
        atoms.insert("type_id", uint_col(&[1, 2])).unwrap();

        let cg = CoarseGrain::from_frame(&frame_of(atoms)).expect("a type column");
        assert_eq!(bead_types(&cg), vec!["CT", "OH"]);
    }

    #[test]
    fn from_frame_renders_a_uint_type_id_in_decimal() {
        let mut atoms = xyz_atoms(&[0.0], &[0.0], &[0.0]);
        atoms.insert("type_id", uint_col(&[2])).unwrap();

        let cg = CoarseGrain::from_frame(&frame_of(atoms)).expect("a UInt type_id column");
        assert_eq!(bead_types(&cg), vec!["2"]);
    }

    #[test]
    fn from_frame_copies_a_type_column_and_every_other_atoms_column() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.5, 2.0], &[-1.0, 3.0]);
        atoms.insert("type", str_col(&["CT", "OH"])).unwrap();
        atoms.insert("charge", float_col(&[0.25, -0.25])).unwrap();

        let cg = CoarseGrain::from_frame(&frame_of(atoms)).expect("a Str type column");
        assert_eq!(cg.n_beads(), 2);
        let beads: Vec<Atom> = cg.beads().map(|(_, bead)| bead).collect();
        let expected = [("CT", 0.0, 0.5, -1.0, 0.25), ("OH", 1.0, 2.0, 3.0, -0.25)];
        for (bead, (ty, x, y, z, q)) in beads.iter().zip(expected) {
            assert_eq!(bead.get_str("bead_type"), Some(ty));
            assert_eq!(bead.get_str("type"), Some(ty), "the source stays a prop");
            assert_eq!(bead.get_f64("x"), Some(x));
            assert_eq!(bead.get_f64("y"), Some(y));
            assert_eq!(bead.get_f64("z"), Some(z));
            assert_eq!(bead.get_f64("charge"), Some(q));
        }
    }

    #[test]
    fn from_frame_turns_bond_rows_into_cg_bonds() {
        let mut atoms = xyz_atoms(&[0.0, 1.0, 2.0], &[0.0; 3], &[0.0; 3]);
        atoms.insert("type", str_col(&["A", "B", "C"])).unwrap();
        // Bond endpoints address 0-based atoms rows.
        let mut bonds = Block::new();
        bonds.insert("atomi", uint_col(&[0, 1])).unwrap();
        bonds.insert("atomj", uint_col(&[2, 2])).unwrap();
        bonds.insert("type_id", uint_col(&[1, 1])).unwrap();
        let mut frame = frame_of(atoms);
        frame.insert("bonds", bonds);

        let cg = CoarseGrain::from_frame(&frame).expect("atoms + bonds");
        assert_eq!(cg.n_bonds(), 2);
        let ids: Vec<NodeId> = cg.node_ids().collect();
        let mut endpoints: Vec<[NodeId; 2]> = cg
            .bonds()
            .map(|(_, bond)| [bond.nodes[0], bond.nodes[1]])
            .collect();
        endpoints.sort_by_key(|[a, _]| ids.iter().position(|id| id == a));
        assert_eq!(endpoints, vec![[ids[0], ids[2]], [ids[1], ids[2]]]);
    }

    #[test]
    fn from_frame_ignores_an_angles_block() {
        let mut atoms = xyz_atoms(&[0.0, 1.0, 2.0], &[0.0; 3], &[0.0; 3]);
        atoms.insert("type", str_col(&["A", "B", "C"])).unwrap();
        let mut bonds = Block::new();
        bonds.insert("atomi", uint_col(&[0, 1])).unwrap();
        bonds.insert("atomj", uint_col(&[1, 2])).unwrap();
        let mut angles = Block::new();
        angles.insert("atomi", uint_col(&[0])).unwrap();
        angles.insert("atomj", uint_col(&[1])).unwrap();
        angles.insert("atomk", uint_col(&[2])).unwrap();
        let mut frame = frame_of(atoms);
        frame.insert("bonds", bonds);
        frame.insert("angles", angles);

        let cg = CoarseGrain::from_frame(&frame).expect("an angles block is ignored");
        assert_eq!(cg.n_beads(), 3);
        assert_eq!(cg.n_bonds(), 2);
    }

    #[test]
    fn from_frame_accepts_a_single_mol_id() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.0; 2], &[0.0; 2]);
        atoms.insert("type", str_col(&["A", "B"])).unwrap();
        atoms.insert("mol_id", uint_col(&[7, 7])).unwrap();

        let cg = CoarseGrain::from_frame(&frame_of(atoms)).expect("one molecule, id 7");
        assert_eq!(cg.n_beads(), 2);
    }

    #[test]
    fn from_frame_refuses_a_frame_without_atoms() {
        let err =
            CoarseGrain::from_frame(&Frame::new()).expect_err("no atoms block to read beads from");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_frame_refuses_an_atoms_block_with_no_rows() {
        let mut atoms = Block::new();
        atoms.insert("bead_type", str_col(&[])).unwrap();

        let err = CoarseGrain::from_frame(&frame_of(atoms))
            .expect_err("a CoarseGrain needs at least one bead row");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_frame_refuses_more_than_one_mol_id() {
        let mut atoms = xyz_atoms(&[0.0, 1.0, 2.0], &[0.0; 3], &[0.0; 3]);
        atoms.insert("type", str_col(&["A", "B", "C"])).unwrap();
        atoms.insert("mol_id", uint_col(&[0, 0, 1])).unwrap();

        let err = CoarseGrain::from_frame(&frame_of(atoms))
            .expect_err("molecules 0 and 1 are two molecules");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("mol_id"), "the error names mol_id, got {msg}");
        assert!(
            msg.contains("Frame::subset"),
            "the error points at Frame::subset, got {msg}"
        );
    }

    #[test]
    fn from_frame_refuses_a_null_mol_id() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.0; 2], &[0.0; 2]);
        atoms.insert("type", str_col(&["A", "B"])).unwrap();
        atoms
            .insert_nullable("mol_id", uint_col(&[7, 0]), vec![true, false])
            .unwrap();

        let err =
            CoarseGrain::from_frame(&frame_of(atoms)).expect_err("bead 1 belongs to no molecule");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_frame_refuses_atoms_without_a_bead_type_source() {
        let atoms = xyz_atoms(&[0.0, 1.0], &[0.0, 0.0], &[0.0, 0.0]);

        let err = CoarseGrain::from_frame(&frame_of(atoms))
            .expect_err("none of bead_type, type, type_id is present");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        for source in ["bead_type", "type", "type_id"] {
            assert!(msg.contains(source), "the error names {source}, got {msg}");
        }
    }

    #[test]
    fn from_frame_refuses_a_null_type_row() {
        // A present `type_id` is not a fallback: the first present source
        // decides, and its null row is an error.
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.0; 2], &[0.0; 2]);
        atoms
            .insert_nullable("type", str_col(&["CT", ""]), vec![true, false])
            .unwrap();
        atoms.insert("type_id", uint_col(&[1, 2])).unwrap();

        let err = CoarseGrain::from_frame(&frame_of(atoms))
            .expect_err("bead 1 has a null type and no fall-through");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_frame_refuses_a_2d_atoms_column() {
        let mut atoms = Block::new();
        atoms.insert("type", str_col(&["A", "B"])).unwrap();
        atoms
            .insert(
                "xyz",
                ndarray::Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                    .unwrap()
                    .into_dyn(),
            )
            .unwrap();

        let err = CoarseGrain::from_frame(&frame_of(atoms))
            .expect_err("a (2, 3) column is not a per-bead property");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert!(err.to_string().contains("'xyz'"), "{err}");
    }

    #[test]
    fn bead_membership_survives_to_frame_then_from_frame() {
        let mut cg = CoarseGrain::new();
        let a = cg.add_bead("W", 0.0, 0.0, 0.0);
        let b = cg.add_bead("P1", 1.0, 0.0, 0.0);
        cg.add_bead("P2", 2.0, 0.0, 0.0);
        cg.set_bead_members(a, vec![10, 11]);
        cg.set_bead_members(b, vec![12]);

        let frame = cg.to_frame().expect("a schema-conforming graph converts");
        let restored = CoarseGrain::from_frame(&frame).expect("CG frame round-trip");
        let ids: Vec<NodeId> = restored.node_ids().collect();
        assert_eq!(ids.len(), 3);
        assert_eq!(restored.bead_members(ids[0]), &[10, 11]);
        assert_eq!(restored.bead_members(ids[1]), &[12]);
        assert!(restored.bead_members(ids[2]).is_empty());
    }

    #[test]
    fn from_frame_refuses_a_members_row_past_the_bead_count() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.0, 0.0], &[0.0, 0.0]);
        atoms.insert("bead_type", str_col(&["W", "W"])).unwrap();
        let mut members = Block::new();
        members.insert("ibead", uint_col(&[0, 2])).unwrap();
        members.insert("atom", uint_col(&[10, 11])).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("members", members);

        let err =
            CoarseGrain::from_frame(&frame).expect_err("ibead 2 names no bead of a two-bead frame");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_frame_refuses_a_repeated_members_row() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.0, 0.0], &[0.0, 0.0]);
        atoms.insert("bead_type", str_col(&["W", "W"])).unwrap();
        let mut members = Block::new();
        members.insert("ibead", uint_col(&[0, 0])).unwrap();
        members.insert("atom", uint_col(&[10, 10])).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("members", members);

        let err =
            CoarseGrain::from_frame(&frame).expect_err("the (ibead 0, atom 10) row appears twice");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    // ---- replicate ----

    /// Membership names foreign atoms the copies do not own, so a replicated
    /// bead carries none even when its template bead does.
    #[test]
    fn replicate_copies_no_bead_membership() {
        use crate::op::Rigid;

        let mut template = CoarseGrain::new();
        let w = template.add_bead("W", 0.0, 0.0, 0.0);
        let p = template.add_bead("P1", 1.0, 0.0, 0.0);
        template.add_bond(w, p).unwrap();
        template.set_bead_members(w, vec![10, 11]);
        template.set_bead_members(p, vec![12]);

        let shifted = Rigid {
            rotation: Rigid::IDENTITY.rotation,
            translation: [5.0, 0.0, 0.0],
        };
        let mut out = CoarseGrain::new();
        let beads = out
            .replicate(&template, &[Rigid::IDENTITY, shifted], &[0, 1])
            .expect("two copies of a two-bead template replicate");

        assert_eq!(beads.len(), 4);
        assert_eq!(out.n_beads(), 4);
        assert_eq!(out.n_bonds(), 2);
        for &bead in &beads {
            assert!(
                out.bead_members(bead).is_empty(),
                "bead {bead:?} was given membership it does not own"
            );
        }
        assert_eq!(template.bead_members(w), &[10, 11], "template untouched");
    }

    // ---- center ----

    /// The group restricts the node set: an unlisted mass-100 bead at
    /// (50,0,0) leaves masses 1 and 3 at x = 0 and 4 centred at (3,0,0).
    #[test]
    fn center_of_a_group_ignores_beads_outside_it() {
        let mut cg = CoarseGrain::new();
        let light = cg.add_bead("W", 0.0, 0.0, 0.0);
        let outside = cg.add_bead("W", 50.0, 0.0, 0.0);
        let heavy = cg.add_bead("P1", 4.0, 0.0, 0.0);
        for (bead, mass) in [(light, 1.0), (outside, 100.0), (heavy, 3.0)] {
            cg.as_molgraph_mut()
                .set_node(bead, crate::core::keys::MASS, mass)
                .unwrap();
        }
        assert_eq!(
            cg.as_molgraph().center(&[light, heavy]),
            Ok([3.0, 0.0, 0.0])
        );
    }

    // ---- positions / bead_types ----

    #[test]
    fn positions_come_back_in_the_order_asked() {
        let mut cg = CoarseGrain::new();
        let b0 = cg.add_bead("W", 0.5, -1.25, 2.0);
        cg.add_bead("P1", 3.0, 4.0, 5.0);
        let b2 = cg.add_bead("P2", -0.75, 8.5, 0.125);

        let positions = cg.positions(&[b2, b0]).expect("both beads are placed");

        assert_eq!(positions, vec![[-0.75, 8.5, 0.125], [0.5, -1.25, 2.0]]);
    }

    #[test]
    fn bead_types_repeat_a_repeated_bead() {
        let mut cg = CoarseGrain::new();
        cg.add_bead("W", 0.0, 0.0, 0.0);
        let b1 = cg.add_bead("P1", 1.0, 0.0, 0.0);

        let types = cg.bead_types(&[b1, b1]).expect("b1 is live");

        assert_eq!(types, vec!["P1".to_owned(), "P1".to_owned()]);
    }

    #[test]
    fn a_stale_bead_is_not_found_by_both_accessors() {
        let mut cg = CoarseGrain::new();
        let kept = cg.add_bead("W", 0.0, 0.0, 0.0);
        let gone = cg.add_bead("P1", 1.0, 0.0, 0.0);
        cg.remove_bead(gone).expect("fixture removal");

        let err = cg.positions(&[kept, gone]).expect_err("gone is stale");
        assert!(matches!(err, MolRsError::NotFound { .. }), "{err:?}");
        let err = cg.bead_types(&[kept, gone]).expect_err("gone is stale");
        assert!(matches!(err, MolRsError::NotFound { .. }), "{err:?}");
    }

    #[test]
    fn a_bead_without_z_is_refused_by_its_integer_handle() {
        let mut cg = CoarseGrain::new();
        cg.add_bead("W", 0.0, 0.0, 0.0);
        let flat = cg.add_bead_bare("P1");
        for key in [keys::X, keys::Y] {
            cg.as_molgraph_mut().set_node(flat, key, 1.0).unwrap();
        }

        let err = cg.positions(&[flat]).expect_err("flat has no z");

        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        let handle = flat.data().as_ffi().to_string();
        assert!(msg.contains(&handle), "names handle {handle}: {msg}");
        assert!(!msg.contains("NodeId("), "no debug id: {msg}");
    }

    #[test]
    fn a_non_finite_coordinate_is_refused() {
        let mut cg = CoarseGrain::new();
        let bad = cg.add_bead("W", 0.0, f64::NAN, 0.0);

        let err = cg.positions(&[bad]).expect_err("y is NaN");

        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }
}
