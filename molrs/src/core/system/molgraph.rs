//! Domain-agnostic dynamic graph for editing-oriented CRUD operations.
//!
//! [`MolGraph`] is a **pure graph** with no chemistry vocabulary: it holds nodes
//! plus a set of **kind-tagged, fixed-arity relations**. A *kind* is registered
//! once by an arbitrary name + arity (`register_kind("bond", 2)`) and addressed
//! thereafter by a dense [`KindId`] (an array index — never a per-access string
//! hash). Relations of the same arity but different meaning (e.g. a 4-ary
//! "dihedral" vs a 4-ary "improper") are distinguished by their [`KindId`], not
//! by arity. The graph itself does not know what a "bond" or an "atom" is —
//! those domain concepts live in the leaf types
//! ([`Atomistic`](crate::system::atomistic::Atomistic) /
//! [`CoarseGrain`](crate::system::coarsegrain::CoarseGrain)) that register their kinds
//! and expose the named convenience API.
//!
//! Storage uses generational arenas ([`slotmap::SlotMap`]) for O(1) insert /
//! remove / lookup with stable handles, and a [`SmallVec`] for each relation's
//! endpoints so the common arities (≤4) stay inline / heap-allocation-free.
//!
//! Every node is a property bag ([`Atom`]): coordinates live as `"x"`, `"y"`,
//! `"z"` keys, matching the Python `Entity(UserDict)` convention.
//!
//! ## Relations vs. containment
//!
//! The `kinds` / relation store is for **fixed-arity peer topology only**.
//! Hierarchical *containment* (a residue owning atoms, a chain owning residues,
//! a coarse-grained bead owning its atoms) is variable-size, nested, directed
//! ownership — **not** a fixed-arity peer relation — and is therefore **not**
//! modeled as a relation kind (doing so would put group handles into the node
//! arena and contaminate every consumer that iterates [`MolGraph::nodes`]). The
//! [`GroupId`] / [`Group`] / `MolGraph::groups` field is **reserved** for a
//! future, independent containment axis; it carries no behavior in this module.
//!
//! # Examples
//!
//! ```
//! use molrs::system::molgraph::{Atom, MolGraph};
//!
//! let mut g = MolGraph::new();
//! let bond = g.register_kind("bond", 2);
//!
//! let o = g.add_node_with(Atom::xyz("O", 0.0, 0.0, 0.0));
//! let h1 = g.add_node_with(Atom::xyz("H", 0.96, 0.0, 0.0));
//! g.add_relation(bond, &[o, h1]).expect("add bond");
//!
//! assert_eq!(g.n_nodes(), 2);
//! assert_eq!(g.n_relations(bond), 1);
//!
//! molrs::spatial::geometry::translate(&mut g, [1.0, 0.0, 0.0]);
//! assert!((g.get_node(o).expect("get node").get_f64("x").unwrap() - 1.0).abs() < 1e-12);
//! ```

use std::collections::HashMap;
use std::ops::{Index, IndexMut};

use slotmap::{Key, KeyData, SecondaryMap, SlotMap, new_key_type};
use smallvec::SmallVec;

use crate::error::MolRsError;
use crate::store::block::Block;
use crate::store::frame::Frame;
use crate::store::keys;
use crate::system::entity_table::{Cell, EntityTable};
use crate::types::{F, I, U};

// ---------------------------------------------------------------------------
// PropValue
// ---------------------------------------------------------------------------

/// Heterogeneous property value stored in an [`Atom`].
#[derive(Debug, Clone, PartialEq)]
pub enum PropValue {
    F64(f64),
    Str(String),
    Int(I),
    Bool(bool),
}

impl PropValue {
    /// Numeric value as `f64`, accepting both `F64` and `Int` variants.
    ///
    /// Use this for quantities that are conceptually numeric but may be stored
    /// as either type depending on the producer — e.g. a bond `"order"` written
    /// as `2` (Int) vs `2.0` (F64). Returns `None` for non-numeric (`Str`,
    /// `Bool`).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            PropValue::F64(v) => Some(*v),
            PropValue::Int(v) => Some(*v as f64),
            PropValue::Str(_) | PropValue::Bool(_) => None,
        }
    }
}

impl From<f64> for PropValue {
    fn from(v: f64) -> Self {
        PropValue::F64(v)
    }
}
impl From<I> for PropValue {
    fn from(v: I) -> Self {
        PropValue::Int(v)
    }
}
impl From<&str> for PropValue {
    fn from(v: &str) -> Self {
        PropValue::Str(v.to_owned())
    }
}
impl From<bool> for PropValue {
    fn from(v: bool) -> Self {
        PropValue::Bool(v)
    }
}
impl From<String> for PropValue {
    fn from(v: String) -> Self {
        PropValue::Str(v)
    }
}

/// Coerce a value to the canonical dtype registered for `key` (if any) so a
/// field's column type stays stable across writers. Currently the only canonical
/// dtype is `Float`, so an `Int` written to a float field (e.g. `x` / `charge` /
/// bond `order` as `1`) is widened to `F64`; everything else passes through.
fn coerce_canonical(key: &str, pv: PropValue) -> PropValue {
    match (keys::canonical_dtype(key), pv) {
        (Some(crate::store::block::DType::Float), PropValue::Int(v)) => PropValue::F64(v as f64),
        (_, pv) => pv,
    }
}

// ---------------------------------------------------------------------------
// Atom  (dynamic node prop bag — also used for beads via `type Bead = Atom`)
// ---------------------------------------------------------------------------

/// A dynamic property bag representing a graph node (an atom or a bead).
///
/// All data — including coordinates (`"x"`, `"y"`, `"z"`), element symbol,
/// mass, charge, etc. — is stored as key-value pairs. The name is historical;
/// `MolGraph` treats it purely as an opaque node payload.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Atom {
    props: HashMap<String, PropValue>,
}

impl Atom {
    /// Create an empty atom.
    pub fn new() -> Self {
        Self::default()
    }

    /// Convenience: create an atom with symbol + xyz (via the [`crate::store::keys`]
    /// field convention — no literal field names).
    pub fn xyz(symbol: &str, x: f64, y: f64, z: f64) -> Self {
        let mut a = Self::new();
        a.set(keys::ELEMENT, symbol);
        a.set(keys::X, x);
        a.set(keys::Y, y);
        a.set(keys::Z, z);
        a
    }

    // ---- dict-like API ----

    /// Insert or update a property.
    pub fn set(&mut self, key: &str, val: impl Into<PropValue>) {
        self.props.insert(key.to_owned(), val.into());
    }

    /// Get a reference to a property value.
    pub fn get(&self, key: &str) -> Option<&PropValue> {
        self.props.get(key)
    }

    /// Get a mutable reference to a property value.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut PropValue> {
        self.props.get_mut(key)
    }

    /// Try to read a property as `f64`.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        match self.props.get(key)? {
            PropValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to read a property as `&str`.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.props.get(key)? {
            PropValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to read a property as `I`.
    pub fn get_int(&self, key: &str) -> Option<I> {
        match self.props.get(key)? {
            PropValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Check whether a key exists.
    pub fn contains_key(&self, key: &str) -> bool {
        self.props.contains_key(key)
    }

    /// Remove a property, returning its value if present.
    pub fn remove(&mut self, key: &str) -> Option<PropValue> {
        self.props.remove(key)
    }

    /// Iterate over all property keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.props.keys().map(|k| k.as_str())
    }

    /// Iterate over all `(key, value)` property pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &PropValue)> {
        self.props.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of properties.
    pub fn len(&self) -> usize {
        self.props.len()
    }

    /// Whether there are no properties.
    pub fn is_empty(&self) -> bool {
        self.props.is_empty()
    }
}

impl Index<&str> for Atom {
    type Output = PropValue;
    fn index(&self, key: &str) -> &Self::Output {
        self.props
            .get(key)
            .unwrap_or_else(|| panic!("Atom does not contain key '{}'", key))
    }
}

impl IndexMut<&str> for Atom {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        self.props
            .get_mut(key)
            .unwrap_or_else(|| panic!("Atom does not contain key '{}'", key))
    }
}

/// Alias for coarse-grained usage — same node payload, different prop keys by
/// convention.
pub type Bead = Atom;

// ---------------------------------------------------------------------------
// Key types
// ---------------------------------------------------------------------------

new_key_type! {
    /// Stable handle to a node in a [`MolGraph`].
    pub struct NodeId;
    /// Stable handle to a relation (any kind) in a [`MolGraph`].
    pub struct RelationId;
    /// Stable handle to a reserved containment group (see module docs).
    pub struct GroupId;
}

/// Convert a [`NodeId`] to/from a stable opaque `u64` handle (the generational
/// slotmap key's FFI form). For exposing stable entity handles across an FFI /
/// language boundary without leaking the `slotmap` types.
pub fn node_to_u64(id: NodeId) -> u64 {
    id.data().as_ffi()
}
/// See [`node_to_u64`].
pub fn node_from_u64(h: u64) -> NodeId {
    NodeId::from(KeyData::from_ffi(h))
}
/// Convert a [`RelationId`] to a stable opaque `u64` handle. See [`node_to_u64`].
pub fn relation_to_u64(id: RelationId) -> u64 {
    id.data().as_ffi()
}
/// See [`relation_to_u64`].
pub fn relation_from_u64(h: u64) -> RelationId {
    RelationId::from(KeyData::from_ffi(h))
}

/// Dense index identifying a registered relation kind. Resolved once at
/// registration; all hot-path relation access goes through this array index,
/// never a per-access string hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KindId(pub u16);

// ---------------------------------------------------------------------------
// Relation
// ---------------------------------------------------------------------------

/// A kind-tagged, fixed-arity relation over [`MolGraph`] nodes.
///
/// `nodes.len()` equals the registered arity of the relation's kind. Endpoints
/// are stored inline for arity ≤ 4 (the common case).
#[derive(Debug, Clone)]
pub struct Relation {
    /// The participating node handles, in order (length == kind arity).
    pub nodes: SmallVec<[NodeId; 4]>,
    /// Per-relation property bag (domain meaning, e.g. a bond `"order"`).
    pub props: HashMap<String, PropValue>,
}

/// Storage for one relation kind: properties live in the aligned column table
/// (the same [`EntityTable`] machinery as nodes — relations are entities with
/// components), while the fixed-arity endpoints are kept structurally in a
/// [`SecondaryMap`] keyed by the same [`RelationId`] the property table mints.
#[derive(Debug, Clone)]
struct RelationKind {
    /// Property columns, keyed by `RelationId` (the kind's relation arena).
    props: EntityTable<RelationId>,
    /// Endpoint node handles per relation (length == kind arity).
    endpoints: SecondaryMap<RelationId, SmallVec<[NodeId; 4]>>,
}

impl RelationKind {
    fn new() -> Self {
        Self {
            props: EntityTable::new(),
            endpoints: SecondaryMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Group (reserved containment axis — no behavior in this module)
// ---------------------------------------------------------------------------

/// Reserved container for the future containment axis (residue ⊃ atoms,
/// chain ⊃ residues, bead ⊃ atoms). Carries no behavior yet; see module docs.
#[derive(Debug, Clone, Default)]
pub struct Group {
    /// Member node handles owned by this group.
    pub members: Vec<NodeId>,
    /// Optional parent group (for nesting).
    pub parent: Option<GroupId>,
    /// Per-group property bag (e.g. `resname`, `resid`).
    pub props: HashMap<String, PropValue>,
}

// ---------------------------------------------------------------------------
// MolGraph
// ---------------------------------------------------------------------------

/// A dynamic, domain-agnostic graph: nodes plus kind-tagged, fixed-arity
/// relations. Knows nothing of "atoms" or "bonds" — those live in leaf types.
#[derive(Debug, Clone)]
pub struct MolGraph {
    /// Node entities + their components, stored as an aligned column table.
    nodes: EntityTable<NodeId>,
    /// Relation kinds (props columns + endpoints), indexed by `KindId.0`.
    kinds: Vec<RelationKind>,
    /// Arity of each kind, indexed by `KindId.0`.
    kind_arity: Vec<usize>,
    /// Registered name of each kind, indexed by `KindId.0` (used as the
    /// [`Frame`] block name in `to_frame` / `read_frame`).
    kind_name: Vec<String>,
    /// Reverse lookup: name → KindId (resolved once, not on the hot path).
    name_to_kind: HashMap<String, KindId>,
    /// Adjacency over arity-2 relations: node → list of `(kind, relation)`.
    adjacency: HashMap<NodeId, Vec<(KindId, RelationId)>>,
    /// Reserved containment axis — see module docs. Unused by all behavior here;
    /// carried so the future containment spec is additive, not a re-key.
    #[allow(dead_code)]
    groups: SlotMap<GroupId, Group>,
}

impl Default for MolGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MolGraph {
    /// Create an empty graph with **no** kinds registered.
    pub fn new() -> Self {
        Self {
            nodes: EntityTable::new(),
            kinds: Vec::new(),
            kind_arity: Vec::new(),
            kind_name: Vec::new(),
            name_to_kind: HashMap::new(),
            adjacency: HashMap::new(),
            groups: SlotMap::with_key(),
        }
    }

    // =====================================================================
    // Kind registry (generic, domain-neutral)
    // =====================================================================

    /// Register a relation kind by name + fixed arity, returning its dense
    /// [`KindId`]. Idempotent: re-registering the same name with the same arity
    /// returns the existing id. Re-registering with a conflicting arity panics
    /// (a programming error — leaf constructors register fixed kinds).
    pub fn register_kind(&mut self, name: &str, arity: usize) -> KindId {
        if let Some(&kid) = self.name_to_kind.get(name) {
            assert_eq!(
                self.kind_arity[kid.0 as usize], arity,
                "kind '{name}' already registered with a different arity"
            );
            return kid;
        }
        let kid = KindId(self.kinds.len() as u16);
        self.kinds.push(RelationKind::new());
        self.kind_arity.push(arity);
        self.kind_name.push(name.to_owned());
        self.name_to_kind.insert(name.to_owned(), kid);
        kid
    }

    /// Resolve a kind name to its registered [`KindId`], if any.
    pub fn kind_id(&self, name: &str) -> Option<KindId> {
        self.name_to_kind.get(name).copied()
    }

    /// Registered name of a kind.
    pub fn kind_name(&self, kind: KindId) -> &str {
        &self.kind_name[kind.0 as usize]
    }

    /// Arity of a registered kind.
    pub fn arity(&self, kind: KindId) -> usize {
        self.kind_arity[kind.0 as usize]
    }

    /// Iterate over all registered [`KindId`]s in registration order.
    pub fn kind_ids(&self) -> impl Iterator<Item = KindId> + '_ {
        (0..self.kinds.len() as u16).map(KindId)
    }

    // =====================================================================
    // Node CRUD (generic)
    // =====================================================================

    /// Insert a field-less node, returning its stable handle. The node has no
    /// components set (no `element` / `bead_type` is forced).
    pub fn add_node(&mut self) -> NodeId {
        let id = self.nodes.spawn();
        self.adjacency.insert(id, Vec::new());
        id
    }

    /// Insert a node carrying a property bag, returning its stable handle.
    pub fn add_node_with(&mut self, payload: Atom) -> NodeId {
        let id = self.add_node();
        self.write_atom(id, &payload);
        id
    }

    /// Remove a node and every relation that references it, across **all**
    /// registered kinds (registry-driven cascade). Returns the node's property
    /// bag (materialized).
    pub fn remove_node(&mut self, id: NodeId) -> Result<Atom, MolRsError> {
        if !self.nodes.contains(id) {
            return Err(MolRsError::not_found("node", format!("NodeId {:?}", id)));
        }
        let payload = self.read_atom(id);

        for kid in 0..self.kinds.len() {
            let doomed: Vec<RelationId> = self.kinds[kid]
                .endpoints
                .iter()
                .filter(|(_, eps)| eps.contains(&id))
                .map(|(rid, _)| rid)
                .collect();
            for rid in doomed {
                self.detach_relation_from_adjacency(KindId(kid as u16), rid, Some(id));
                let k = &mut self.kinds[kid];
                k.props.despawn(rid);
                k.endpoints.remove(rid);
            }
        }
        self.adjacency.remove(&id);
        self.nodes.despawn(id);
        Ok(payload)
    }

    /// Materialize a node's property bag (owned copy of its set components).
    pub fn get_node(&self, id: NodeId) -> Result<Atom, MolRsError> {
        if !self.nodes.contains(id) {
            return Err(MolRsError::not_found("node", format!("NodeId {:?}", id)));
        }
        Ok(self.read_atom(id))
    }

    /// Set a single component on a node.
    pub fn set_node(
        &mut self,
        id: NodeId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        match coerce_canonical(key, val.into()) {
            PropValue::F64(v) => self.nodes.set_f64(id, key, v),
            PropValue::Int(v) => self.nodes.set_i32(id, key, v),
            PropValue::Str(s) => self.nodes.set_str(id, key, &s),
            PropValue::Bool(v) => self.nodes.set_bool(id, key, v),
        }
    }

    /// Clear a single component on a node (no-op if absent).
    pub fn clear_node(&mut self, id: NodeId, key: &str) -> Result<(), MolRsError> {
        self.nodes.clear(id, key)
    }

    /// Iterate over all `(NodeId, Atom)` pairs (each property bag materialized).
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, Atom)> + '_ {
        self.nodes.handles().map(move |id| (id, self.read_atom(id)))
    }

    /// Live node handles in row order.
    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.handles()
    }

    /// Borrow the underlying node column table (for zero-copy column access).
    pub fn node_table(&self) -> &EntityTable<NodeId> {
        &self.nodes
    }

    /// Mutable access to the underlying node column table.
    pub fn node_table_mut(&mut self) -> &mut EntityTable<NodeId> {
        &mut self.nodes
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Write an [`Atom`]'s properties into node `id`'s columns.
    fn write_atom(&mut self, id: NodeId, atom: &Atom) {
        for (key, val) in atom.iter() {
            let r = match coerce_canonical(key, val.clone()) {
                PropValue::F64(v) => self.nodes.set_f64(id, key, v),
                PropValue::Int(v) => self.nodes.set_i32(id, key, v),
                PropValue::Str(s) => self.nodes.set_str(id, key, &s),
                PropValue::Bool(v) => self.nodes.set_bool(id, key, v),
            };
            debug_assert!(r.is_ok(), "component type conflict writing '{key}'");
            let _ = r;
        }
    }

    /// Materialize node `id`'s set components into an [`Atom`].
    pub(crate) fn read_atom(&self, id: NodeId) -> Atom {
        let mut atom = Atom::new();
        for (key, cell) in self.nodes.row_cells(id) {
            match cell {
                Cell::F64(v) => atom.set(key, v),
                Cell::I32(v) => atom.set(key, v),
                Cell::Str(s) => atom.set(key, s),
                Cell::Bool(v) => atom.set(key, v),
            }
        }
        atom
    }

    /// Iterate over neighbor node IDs of a given node (via arity-2 relations).
    pub fn neighbors(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.neighbor_relations(id).map(|(_, _, other)| other)
    }

    /// Iterate over `(kind, relation, other_node)` for each arity-2 relation
    /// incident to a node. Domain leaves build typed neighbor queries on this.
    pub fn neighbor_relations(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = (KindId, RelationId, NodeId)> + '_ {
        self.adjacency
            .get(&id)
            .into_iter()
            .flatten()
            .filter_map(move |&(kind, rid)| {
                let eps = self.kinds[kind.0 as usize].endpoints.get(rid)?;
                let other = if eps[0] == id { eps[1] } else { eps[0] };
                Some((kind, rid, other))
            })
    }

    // =====================================================================
    // Relation CRUD (generic, kind-tagged)
    // =====================================================================

    /// Add a relation of the given kind over the given nodes. Validates the
    /// kind is registered, the node count matches its arity, and every node
    /// exists. Maintains the adjacency index for arity-2 relations.
    pub fn add_relation(
        &mut self,
        kind: KindId,
        nodes: &[NodeId],
    ) -> Result<RelationId, MolRsError> {
        let kidx = kind.0 as usize;
        if kidx >= self.kinds.len() {
            return Err(MolRsError::not_found("kind", format!("KindId {:?}", kind)));
        }
        let arity = self.kind_arity[kidx];
        if nodes.len() != arity {
            return Err(MolRsError::validation(format!(
                "kind '{}' expects arity {}, got {} nodes",
                self.kind_name[kidx],
                arity,
                nodes.len()
            )));
        }
        for &n in nodes {
            if !self.nodes.contains(n) {
                return Err(MolRsError::not_found("node", format!("NodeId {:?}", n)));
            }
        }
        let k = &mut self.kinds[kidx];
        let rid = k.props.spawn();
        k.endpoints.insert(rid, SmallVec::from_slice(nodes));
        if arity == 2 {
            self.adjacency
                .entry(nodes[0])
                .or_default()
                .push((kind, rid));
            self.adjacency
                .entry(nodes[1])
                .or_default()
                .push((kind, rid));
        }
        Ok(rid)
    }

    /// Materialize a relation (endpoints + properties) by kind + handle.
    pub fn get_relation(&self, kind: KindId, id: RelationId) -> Result<Relation, MolRsError> {
        let k = self
            .kinds
            .get(kind.0 as usize)
            .ok_or_else(|| MolRsError::not_found("kind", format!("KindId {:?}", kind)))?;
        if !k.props.contains(id) {
            return Err(MolRsError::not_found(
                "relation",
                format!("RelationId {:?}", id),
            ));
        }
        Ok(self.read_relation(kind, id))
    }

    /// Endpoint node handles of a relation.
    pub fn relation_nodes(
        &self,
        kind: KindId,
        id: RelationId,
    ) -> Result<SmallVec<[NodeId; 4]>, MolRsError> {
        self.kinds
            .get(kind.0 as usize)
            .and_then(|k| k.endpoints.get(id).cloned())
            .ok_or_else(|| MolRsError::not_found("relation", format!("RelationId {:?}", id)))
    }

    /// Set a single property on a relation.
    pub fn set_relation_prop(
        &mut self,
        kind: KindId,
        id: RelationId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        let k = self
            .kinds
            .get_mut(kind.0 as usize)
            .ok_or_else(|| MolRsError::not_found("kind", format!("KindId {:?}", kind)))?;
        if !k.props.contains(id) {
            return Err(MolRsError::not_found(
                "relation",
                format!("RelationId {:?}", id),
            ));
        }
        match coerce_canonical(key, val.into()) {
            PropValue::F64(v) => k.props.set_f64(id, key, v),
            PropValue::Int(v) => k.props.set_i32(id, key, v),
            PropValue::Str(s) => k.props.set_str(id, key, &s),
            PropValue::Bool(v) => k.props.set_bool(id, key, v),
        }
    }

    /// Clear a single property on a relation (no-op if absent).
    pub fn clear_relation_prop(
        &mut self,
        kind: KindId,
        id: RelationId,
        key: &str,
    ) -> Result<(), MolRsError> {
        let k = self
            .kinds
            .get_mut(kind.0 as usize)
            .ok_or_else(|| MolRsError::not_found("kind", format!("KindId {:?}", kind)))?;
        k.props.clear(id, key)
    }

    /// Remove a relation by kind + handle, updating adjacency. Returns the
    /// materialized relation.
    pub fn remove_relation(
        &mut self,
        kind: KindId,
        id: RelationId,
    ) -> Result<Relation, MolRsError> {
        let exists = self
            .kinds
            .get(kind.0 as usize)
            .is_some_and(|k| k.props.contains(id));
        if !exists {
            return Err(MolRsError::not_found(
                "relation",
                format!("RelationId {:?}", id),
            ));
        }
        let rel = self.read_relation(kind, id);
        self.detach_relation_from_adjacency(kind, id, None);
        let k = &mut self.kinds[kind.0 as usize];
        k.props.despawn(id);
        k.endpoints.remove(id);
        Ok(rel)
    }

    /// Iterate over `(RelationId, Relation)` for one kind (each materialized).
    pub fn relations(&self, kind: KindId) -> impl Iterator<Item = (RelationId, Relation)> + '_ {
        let ids: Vec<RelationId> = self.kinds[kind.0 as usize].props.handles().collect();
        ids.into_iter()
            .map(move |rid| (rid, self.read_relation(kind, rid)))
    }

    /// Relation handles of a kind, in row order.
    pub fn relation_ids(&self, kind: KindId) -> impl Iterator<Item = RelationId> + '_ {
        self.kinds[kind.0 as usize].props.handles()
    }

    /// Number of relations of a kind.
    pub fn n_relations(&self, kind: KindId) -> usize {
        self.kinds[kind.0 as usize].props.len()
    }

    /// Materialize a relation's endpoints + properties.
    pub(crate) fn read_relation(&self, kind: KindId, id: RelationId) -> Relation {
        let k = &self.kinds[kind.0 as usize];
        let nodes = k.endpoints.get(id).cloned().unwrap_or_default();
        let mut props = HashMap::new();
        for (key, cell) in k.props.row_cells(id) {
            let pv = match cell {
                Cell::F64(v) => PropValue::F64(v),
                Cell::I32(v) => PropValue::Int(v),
                Cell::Str(s) => PropValue::Str(s.to_owned()),
                Cell::Bool(v) => PropValue::Bool(v),
            };
            props.insert(key.to_owned(), pv);
        }
        Relation { nodes, props }
    }

    /// Write a property bag into a relation's columns.
    pub(crate) fn write_relation_props(
        &mut self,
        kind: KindId,
        id: RelationId,
        props: &HashMap<String, PropValue>,
    ) {
        let k = &mut self.kinds[kind.0 as usize];
        for (key, val) in props {
            let r = match coerce_canonical(key, val.clone()) {
                PropValue::F64(v) => k.props.set_f64(id, key, v),
                PropValue::Int(v) => k.props.set_i32(id, key, v),
                PropValue::Str(s) => k.props.set_str(id, key, &s),
                PropValue::Bool(v) => k.props.set_bool(id, key, v),
            };
            let _ = r;
        }
    }

    /// Remove an arity-2 relation from the adjacency lists of its endpoints.
    /// `skip` lets `remove_node` avoid touching the node currently being dropped.
    fn detach_relation_from_adjacency(
        &mut self,
        kind: KindId,
        id: RelationId,
        skip: Option<NodeId>,
    ) {
        if self.kind_arity[kind.0 as usize] != 2 {
            return;
        }
        let endpoints: Option<[NodeId; 2]> = self.kinds[kind.0 as usize]
            .endpoints
            .get(id)
            .map(|eps| [eps[0], eps[1]]);
        if let Some(eps) = endpoints {
            for ep in eps {
                if Some(ep) == skip {
                    continue;
                }
                if let Some(adj) = self.adjacency.get_mut(&ep) {
                    adj.retain(|(_, rid)| *rid != id);
                }
            }
        }
    }

    // Spatial transforms are free-function *systems* — see [`crate::spatial::geometry`]
    // (`translate`, `rotate`). The data structure carries no geometry methods.

    // =====================================================================
    // Composition
    // =====================================================================

    /// Merge another `MolGraph` into `self`, consuming `other`.
    ///
    /// Registry-driven: every relation of every kind in `other` is transferred
    /// (kinds matched by name, registered on `self` if missing) — so all kinds
    /// are carried across.
    ///
    /// **Handle contract:** every node of `other` is remapped to a fresh handle in
    /// `self`. Returns the map `NodeId in other → NodeId in self`. (By contrast,
    /// [`Clone`] **preserves** handles in the independent copy.)
    pub fn merge(&mut self, other: MolGraph) -> HashMap<NodeId, NodeId> {
        let mut node_map: HashMap<NodeId, NodeId> = HashMap::new();
        for old_id in other.nodes.handles() {
            let payload = other.read_atom(old_id);
            let new_id = self.add_node_with(payload);
            node_map.insert(old_id, new_id);
        }

        for okid in other.kind_ids() {
            let oidx = okid.0 as usize;
            let name = &other.kind_name[oidx];
            let arity = other.kind_arity[oidx];
            let self_kind = self.register_kind(name, arity);
            let orids: Vec<RelationId> = other.relation_ids(okid).collect();
            for orid in orids {
                let rel = other.read_relation(okid, orid);
                let mapped: SmallVec<[NodeId; 4]> = rel.nodes.iter().map(|n| node_map[n]).collect();
                if let Ok(rid) = self.add_relation(self_kind, &mapped) {
                    self.write_relation_props(self_kind, rid, &rel.props);
                }
            }
        }
        node_map
    }

    // =====================================================================
    // Frame conversion (shared mechanism)
    //
    // The PUBLIC `to_frame` / `from_frame` API is provided by the **leaf**
    // types ([`Atomistic`](crate::system::atomistic::Atomistic) /
    // [`CoarseGrain`](crate::system::coarsegrain::CoarseGrain)), since converting to/from
    // the central [`Frame`] is a domain operation with leaf-specific block/kind
    // requirements. These `pub(crate)` methods are only the shared,
    // registry-driven implementation the leaves call — not data-struct API.
    // =====================================================================

    /// Shared implementation of leaf `to_frame`. Each node-component becomes a
    /// column in the `"atoms"` block; every non-empty relation kind becomes a
    /// block (named by the kind) with `atomi`/`atomj`/… columns referencing node
    /// row order plus one column per relation property — registry-driven.
    pub(crate) fn to_frame(&self) -> Frame {
        use ndarray::Array1;

        let mut frame = Frame::new();

        let node_ids: Vec<NodeId> = self.nodes.handles().collect();
        let n = node_ids.len();
        let id_to_row: HashMap<NodeId, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        // ---- atoms (node) block: one column per component (zero-copy reads;
        // columns are already dense and aligned to node row order) ----
        let mut all_keys: Vec<String> = self.nodes.columns().map(|s| s.to_owned()).collect();
        all_keys.sort();

        let mut atoms_block = Block::new();
        for key in &all_keys {
            if let Ok((data, _)) = self.nodes.column_f64(key) {
                let _ =
                    atoms_block.insert(key.as_str(), Array1::from_vec(data.to_vec()).into_dyn());
            } else if let Ok((data, _)) = self.nodes.column_i32(key) {
                let _ =
                    atoms_block.insert(key.as_str(), Array1::from_vec(data.to_vec()).into_dyn());
            } else if let Ok((data, _)) = self.nodes.column_str(key) {
                let _ =
                    atoms_block.insert(key.as_str(), Array1::from_vec(data.to_vec()).into_dyn());
            }
        }
        if n > 0 {
            frame.insert("atoms", atoms_block);
        }

        // ---- one block per non-empty relation kind ----
        for kid in self.kind_ids() {
            let kidx = kid.0 as usize;
            let k = &self.kinds[kidx];
            if k.props.is_empty() {
                continue;
            }
            let arity = self.kind_arity[kidx];
            // Relation row order: shared by the endpoint columns and the (already
            // aligned, dense) property columns.
            let rids: Vec<RelationId> = k.props.handles().collect();
            let mut block = Block::new();

            for pos in 0..arity {
                let col: Vec<U> = rids
                    .iter()
                    .map(|rid| id_to_row[&k.endpoints[*rid][pos]] as U)
                    .collect();
                let _ = block.insert(rel_col_name(pos), Array1::from_vec(col).into_dyn());
            }

            // Property columns read straight from the column table.
            let mut prop_keys: Vec<String> = k.props.columns().map(|s| s.to_owned()).collect();
            prop_keys.sort();
            for key in &prop_keys {
                if let Ok((data, _)) = k.props.column_f64(key) {
                    let _ = block.insert(key.as_str(), Array1::from_vec(data.to_vec()).into_dyn());
                } else if let Ok((data, _)) = k.props.column_i32(key) {
                    let _ = block.insert(key.as_str(), Array1::from_vec(data.to_vec()).into_dyn());
                } else if let Ok((data, _)) = k.props.column_str(key) {
                    let _ = block.insert(key.as_str(), Array1::from_vec(data.to_vec()).into_dyn());
                }
            }

            frame.insert(&self.kind_name[kidx], block);
        }

        frame
    }

    /// Read a [`Frame`] into `self`: the `"atoms"` block becomes nodes; each
    /// **already-registered** kind's block (matched by name) becomes relations
    /// via its `atomi`/`atomj`/… columns, with any extra columns read back as
    /// props. Kinds not registered on `self` are skipped (a bare graph keeps
    /// only nodes; register kinds first to read their relations).
    pub(crate) fn read_frame(&mut self, frame: &Frame) -> Result<(), MolRsError> {
        let atoms_block = frame
            .get("atoms")
            .ok_or_else(|| MolRsError::parse("Frame missing 'atoms' block"))?;

        let nrows = atoms_block.nrows().unwrap_or(0);
        let col_keys: Vec<String> = atoms_block.keys().map(|k| k.to_owned()).collect();

        let mut float_cols: Vec<(&str, &ndarray::ArrayD<F>)> = Vec::new();
        let mut i64_cols: Vec<(&str, &ndarray::ArrayD<I>)> = Vec::new();
        let mut str_cols: Vec<(&str, &ndarray::ArrayD<String>)> = Vec::new();
        for key in &col_keys {
            if let Some(arr) = atoms_block.get_float(key) {
                float_cols.push((key.as_str(), arr));
            } else if let Some(arr) = atoms_block.get_int(key) {
                i64_cols.push((key.as_str(), arr));
            } else if let Some(arr) = atoms_block.get_string(key) {
                str_cols.push((key.as_str(), arr));
            }
        }

        let mut node_ids: Vec<NodeId> = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let mut node = Atom::new();
            for &(key, arr) in &float_cols {
                #[allow(clippy::unnecessary_cast)]
                node.set(key, arr[[row]] as f64);
            }
            for &(key, arr) in &i64_cols {
                node.set(key, PropValue::Int(arr[[row]]));
            }
            for &(key, arr) in &str_cols {
                node.set(key, PropValue::Str(arr[[row]].clone()));
            }
            node_ids.push(self.add_node_with(node));
        }

        let kind_specs: Vec<(KindId, String, usize)> = self
            .kind_ids()
            .map(|kid| {
                let i = kid.0 as usize;
                (kid, self.kind_name[i].clone(), self.kind_arity[i])
            })
            .collect();

        for (kid, block_name, arity) in kind_specs {
            let Some(block) = frame.get(&block_name) else {
                continue;
            };
            let mut endpoint_cols: Vec<&ndarray::ArrayD<U>> = Vec::with_capacity(arity);
            let mut ok = true;
            for pos in 0..arity {
                match block.get_uint(&rel_col_name(pos)) {
                    Some(c) => endpoint_cols.push(c),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            let nrel = block.nrows().unwrap_or(0);
            let endpoint_names: Vec<String> = (0..arity).map(rel_col_name).collect();
            // Collect non-endpoint relation property columns by element type so
            // a to_frame -> read_frame round-trip restores int and string
            // relation props (e.g. a string bond label), not just floats.
            let mut prop_f: Vec<(String, &ndarray::ArrayD<F>)> = Vec::new();
            let mut prop_i: Vec<(String, &ndarray::ArrayD<I>)> = Vec::new();
            let mut prop_s: Vec<(String, &ndarray::ArrayD<String>)> = Vec::new();
            for k in block.keys() {
                if endpoint_names.iter().any(|e| e == k) {
                    continue;
                }
                if let Some(a) = block.get_float(k) {
                    prop_f.push((k.to_owned(), a));
                } else if let Some(a) = block.get_int(k) {
                    prop_i.push((k.to_owned(), a));
                } else if let Some(a) = block.get_string(k) {
                    prop_s.push((k.to_owned(), a));
                }
            }

            for row in 0..nrel {
                let mut nodes: SmallVec<[NodeId; 4]> = SmallVec::new();
                let mut valid = true;
                for col in &endpoint_cols {
                    let idx = col[[row]] as usize;
                    if idx >= node_ids.len() {
                        valid = false;
                        break;
                    }
                    nodes.push(node_ids[idx]);
                }
                if !valid {
                    continue;
                }
                if let Ok(rid) = self.add_relation(kid, &nodes) {
                    for (k, arr) in &prop_f {
                        #[allow(clippy::unnecessary_cast)]
                        let _ = self.set_relation_prop(kid, rid, k, arr[[row]] as f64);
                    }
                    for (k, arr) in &prop_i {
                        let _ = self.set_relation_prop(kid, rid, k, PropValue::Int(arr[[row]]));
                    }
                    for (k, arr) in &prop_s {
                        let _ =
                            self.set_relation_prop(kid, rid, k, PropValue::Str(arr[[row]].clone()));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Endpoint column name for the `pos`-th node of a relation block.
pub(crate) fn rel_col_name(pos: usize) -> String {
    match crate::store::keys::ENDPOINTS.get(pos) {
        Some(name) => (*name).to_owned(),
        None => format!("atom{pos}"),
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----- PropValue & Atom dict-like API -----

    #[test]
    fn test_propvalue_from() {
        let v: PropValue = std::f64::consts::PI.into();
        assert_eq!(v, PropValue::F64(std::f64::consts::PI));
        let v: PropValue = (42 as I).into();
        assert_eq!(v, PropValue::Int(42 as I));
        let v: PropValue = "H".into();
        assert_eq!(v, PropValue::Str("H".to_owned()));
    }

    #[test]
    fn test_atom_dict_api() {
        let mut a = Atom::new();
        a.set("x", 1.5);
        a.set("element", "C");
        a.set("type_id", PropValue::Int(3 as I));
        assert_eq!(a.get_f64("x"), Some(1.5));
        assert_eq!(a.get_str("element"), Some("C"));
        assert_eq!(a.get_int("type_id"), Some(3 as I));
        assert_eq!(a.get_f64("missing"), None);
        assert!(a.contains_key("x"));
        assert_eq!(a.len(), 3);
        a.remove("type_id");
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_atom_index() {
        let mut a = Atom::xyz("O", 1.0, 2.0, 3.0);
        assert_eq!(a["x"], PropValue::F64(1.0));
        a["x"] = PropValue::F64(99.0);
        assert_eq!(a.get_f64("x"), Some(99.0));
    }

    // ----- Kind registry -----

    #[test]
    fn test_register_kind_dense_idempotent() {
        let mut g = MolGraph::new();
        let bond = g.register_kind("bond", 2);
        let angle = g.register_kind("angle", 3);
        assert_eq!(bond, KindId(0));
        assert_eq!(angle, KindId(1));
        // idempotent
        assert_eq!(g.register_kind("bond", 2), bond);
        assert_eq!(g.kind_id("angle"), Some(angle));
        assert_eq!(g.arity(bond), 2);
        assert_eq!(g.kind_name(angle), "angle");
    }

    #[test]
    #[should_panic]
    fn test_register_kind_conflicting_arity_panics() {
        let mut g = MolGraph::new();
        g.register_kind("bond", 2);
        g.register_kind("bond", 3);
    }

    // ----- Node CRUD -----

    #[test]
    fn test_add_node_field_less() {
        let mut g = MolGraph::new();
        let n = g.add_node();
        assert!(g.get_node(n).unwrap().is_empty());
        crate::spatial::geometry::translate(&mut g, [1.0, 2.0, 3.0]);
        assert!(g.get_node(n).unwrap().get_f64("x").is_none());
        g.set_node(n, "element", "C").unwrap();
        assert_eq!(g.get_node(n).unwrap().get_str("element"), Some("C"));
    }

    #[test]
    fn test_add_remove_node() {
        let mut g = MolGraph::new();
        assert_eq!(g.n_nodes(), 0);
        let id = g.add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0));
        assert_eq!(g.n_nodes(), 1);
        assert_eq!(g.get_node(id).unwrap().get_str("element"), Some("C"));
        g.remove_node(id).unwrap();
        assert_eq!(g.n_nodes(), 0);
        assert!(g.get_node(id).is_err());
    }

    // ----- Relation CRUD -----

    #[test]
    fn test_generic_relation_crud() {
        let mut g = MolGraph::new();
        let angle = g.register_kind("angle", 3);
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        let rid = g.add_relation(angle, &[a, b, c]).unwrap();
        assert_eq!(g.n_relations(angle), 1);
        assert_eq!(
            g.get_relation(angle, rid).unwrap().nodes.as_slice(),
            &[a, b, c][..]
        );
        assert!(g.add_relation(angle, &[a, b]).is_err()); // wrong arity
        g.remove_node(c).unwrap();
        assert!(g.add_relation(angle, &[a, b, c]).is_err()); // missing node
        assert_eq!(g.n_relations(angle), 0); // cascaded
    }

    #[test]
    fn test_same_arity_distinct_kind() {
        let mut g = MolGraph::new();
        let dih = g.register_kind("dihedral", 4);
        let imp = g.register_kind("improper", 4);
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        let d = g.add_node();
        g.add_relation(dih, &[a, b, c, d]).unwrap();
        g.add_relation(imp, &[a, b, c, d]).unwrap();
        assert_eq!(g.n_relations(dih), 1);
        assert_eq!(g.n_relations(imp), 1);
    }

    #[test]
    fn test_cascade_across_all_kinds() {
        let mut g = MolGraph::new();
        let bond = g.register_kind("bond", 2);
        let angle = g.register_kind("angle", 3);
        let dih = g.register_kind("dihedral", 4);
        let imp = g.register_kind("improper", 4);
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        let d = g.add_node();
        g.add_relation(bond, &[a, b]).unwrap();
        g.add_relation(angle, &[b, a, c]).unwrap();
        g.add_relation(dih, &[b, a, c, d]).unwrap();
        g.add_relation(imp, &[b, a, c, d]).unwrap();
        g.remove_node(a).unwrap();
        assert_eq!(g.n_relations(bond), 0);
        assert_eq!(g.n_relations(angle), 0);
        assert_eq!(g.n_relations(dih), 0);
        assert_eq!(g.n_relations(imp), 0);
    }

    // ----- Neighbors -----

    #[test]
    fn test_neighbors() {
        let mut g = MolGraph::new();
        let bond = g.register_kind("bond", 2);
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        g.add_relation(bond, &[a, b]).unwrap();
        g.add_relation(bond, &[a, c]).unwrap();
        let mut n: Vec<NodeId> = g.neighbors(a).collect();
        n.sort_by_key(|id| id.0);
        assert_eq!(n.len(), 2);
        assert!(n.contains(&b) && n.contains(&c));
        assert_eq!(g.neighbors(b).collect::<Vec<_>>(), vec![a]);
        // removing the relation clears adjacency
        let bid = g.relations(bond).next().unwrap().0;
        g.remove_relation(bond, bid).unwrap();
        assert_eq!(g.neighbors(a).count(), 1);
    }

    // ----- Spatial -----

    #[test]
    fn test_translate_and_rotate() {
        let mut g = MolGraph::new();
        let id = g.add_node_with(Atom::xyz("C", 1.0, 0.0, 0.0));
        crate::spatial::geometry::translate(&mut g, [10.0, 20.0, 30.0]);
        let a = g.get_node(id).unwrap();
        assert!((a.get_f64("x").unwrap() - 11.0).abs() < 1e-12);
        let id2 = g.add_node_with(Atom::xyz("C", 1.0, 0.0, 0.0));
        crate::spatial::geometry::rotate(
            &mut g,
            [0.0, 0.0, 1.0],
            std::f64::consts::FRAC_PI_2,
            None,
        );
        let b = g.get_node(id2).unwrap();
        assert!((b.get_f64("x").unwrap()).abs() < 1e-12);
        assert!((b.get_f64("y").unwrap() - 1.0).abs() < 1e-12);
    }

    // ----- Frame round-trip (generic) -----

    #[test]
    fn test_to_read_frame_roundtrip() {
        let mut g = MolGraph::new();
        let bond = g.register_kind("bonds", 2);
        let o = g.add_node_with(Atom::xyz("O", 0.0, 0.0, 0.0));
        let h1 = g.add_node_with(Atom::xyz("H", 0.96, 0.0, 0.0));
        let h2 = g.add_node_with(Atom::xyz("H", -0.24, 0.93, 0.0));
        g.add_relation(bond, &[o, h1]).unwrap();
        g.add_relation(bond, &[o, h2]).unwrap();
        let frame = g.to_frame();
        assert!(frame.contains_key("atoms"));
        assert!(frame.contains_key("bonds"));
        assert_eq!(frame["atoms"].nrows(), Some(3));
        assert_eq!(frame["bonds"].nrows(), Some(2));

        // read back into a graph with the same kind registered
        let mut g2 = MolGraph::new();
        let bond2 = g2.register_kind("bonds", 2);
        g2.read_frame(&frame).unwrap();
        assert_eq!(g2.n_nodes(), 3);
        assert_eq!(g2.n_relations(bond2), 2);
    }

    #[test]
    fn test_read_frame_restores_int_and_str_relation_props() {
        let mut g = MolGraph::new();
        let bond = g.register_kind("bonds", 2);
        let a = g.add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0));
        let b = g.add_node_with(Atom::xyz("C", 1.5, 0.0, 0.0));
        let rid = g.add_relation(bond, &[a, b]).unwrap();
        g.set_relation_prop(bond, rid, "order", 2.0_f64).unwrap();
        g.set_relation_prop(bond, rid, "ring_size", PropValue::Int(6))
            .unwrap();
        g.set_relation_prop(bond, rid, "label", PropValue::Str("aromatic".to_owned()))
            .unwrap();

        let frame = g.to_frame();

        let mut g2 = MolGraph::new();
        let bond2 = g2.register_kind("bonds", 2);
        g2.read_frame(&frame).unwrap();

        let (_, r) = g2.relations(bond2).next().expect("relation round-trips");
        assert_eq!(
            r.props.get("ring_size"),
            Some(&PropValue::Int(6)),
            "int relation prop lost on round-trip"
        );
        assert_eq!(
            r.props.get("label"),
            Some(&PropValue::Str("aromatic".to_owned())),
            "string relation prop lost on round-trip"
        );
        match r.props.get("order") {
            Some(PropValue::F64(v)) => assert!((v - 2.0).abs() < 1e-12),
            other => panic!("float relation prop lost: {other:?}"),
        }
    }

    // ----- Merge (registry-driven; covers all kinds) -----

    #[test]
    fn test_merge_transfers_all_kinds() {
        let mut src = MolGraph::new();
        let bond = src.register_kind("bond", 2);
        let imp = src.register_kind("improper", 4);
        let a = src.add_node();
        let b = src.add_node();
        let c = src.add_node();
        let d = src.add_node();
        src.add_relation(bond, &[a, b]).unwrap();
        src.add_relation(imp, &[a, b, c, d]).unwrap();

        let mut dst = MolGraph::new();
        let dbond = dst.register_kind("bond", 2);
        let dimp = dst.register_kind("improper", 4);
        let e = dst.add_node();
        let f = dst.add_node();
        dst.add_relation(dbond, &[e, f]).unwrap();

        let map = dst.merge(src);
        assert_eq!(dst.n_nodes(), 6);
        assert_eq!(dst.n_relations(dbond), 2);
        assert_eq!(dst.n_relations(dimp), 1, "merge must carry impropers");
        assert_eq!(map.len(), 4, "merge returns one entry per node of other");
    }

    #[test]
    fn test_clone_preserves_handles() {
        let mut g = MolGraph::new();
        let bond = g.register_kind("bonds", 2);
        let a = g.add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0));
        let b = g.add_node_with(Atom::xyz("H", 1.0, 0.0, 0.0));
        let rid = g.add_relation(bond, &[a, b]).unwrap();
        let g2 = g.clone();
        assert!(g2.get_node(a).is_ok());
        assert!(g2.get_node(b).is_ok());
        assert!(g2.get_relation(bond, rid).is_ok());
        assert_eq!(g2.get_node(a).unwrap().get_str("element"), Some("C"));
    }

    // ----- Clone independence -----

    #[test]
    fn test_clone_independence() {
        let mut g = MolGraph::new();
        let id = g.add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0));
        g.add_node_with(Atom::xyz("H", 1.0, 0.0, 0.0));
        let g2 = g.clone();
        g.set_node(id, "x", 99.0).unwrap();
        assert_eq!(g2.get_node(id).unwrap().get_f64("x"), Some(0.0));
        assert_eq!(g2.n_nodes(), 2);
    }

    // ----- Reserved containment axis -----

    #[test]
    fn test_groups_reserved_empty() {
        let g = MolGraph::new();
        assert_eq!(g.groups.len(), 0);
    }
}
