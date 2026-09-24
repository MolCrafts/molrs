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
//! let o = g.add_node_with(Atom::xyz("O", 0.0, 0.0, 0.0)).expect("add O");
//! let h1 = g.add_node_with(Atom::xyz("H", 0.96, 0.0, 0.0)).expect("add H");
//! g.add_relation(bond, &[o, h1]).expect("add bond");
//!
//! assert_eq!(g.n_nodes(), 2);
//! assert_eq!(g.n_relations(bond), 1);
//!
//! molrs::spatial::geometry::translate(&mut g, [1.0, 0.0, 0.0]);
//! assert!((g.get_node(o).expect("get node").get_f64("x").unwrap() - 1.0).abs() < 1e-12);
//! ```

use std::collections::{HashMap, HashSet};
use std::ops::{Index, IndexMut};

use ndarray::ArrayD;
use slotmap::{Key, KeyData, SecondaryMap, SlotMap, new_key_type};
use smallvec::SmallVec;

use crate::error::MolRsError;
use crate::store::block::Block;
use crate::store::frame::Frame;
use crate::store::keys;
use crate::system::entity_table::{Cell, EntityTable, Validity};
use crate::types::{F, I, Idx};

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

/// Admit a value into a node or relation column only if it can be stored at
/// the dtype the Frame schema declares for `key`, widening where width is not
/// semantics and refusing otherwise.
///
/// A key the vocabulary does not declare (`tag`, `frag_id`, a perceived fact,
/// a per-instance force-field parameter) carries whatever the caller stores:
/// no frame column type contradicts it.
///
/// For a declared key the refusal belongs *here*, at the write, because the
/// declared dtype is what every later stage already assumes: [`MolGraph::to_frame`]
/// must hand [`Block::insert`] a column of exactly that dtype, the infallible
/// leaf constructors ([`Atomistic::add_atom_xyz`](crate::system::atomistic::Atomistic::add_atom_xyz)) write their components into
/// a column whose element type a stray write has already fixed, and the Python
/// `Block` boundary re-states the same rule. Every one of those stages is past
/// the point where the caller still holds the value — a `set_node` that
/// accepted a string `"x"` would hand back a graph whose only remaining moves
/// are to lose the property or to fail on the way out. [`set_node`](MolGraph::set_node)
/// has a `Result`, so it is the last door where the caller can still act.
///
/// Width is not semantics: an `Int` under a float key (`x`, `charge` written
/// as `1`) is widened to `F64`. A *sign* is semantics, so a negative under an
/// unsigned key (`id`, `mol_id`, `type_id`, …) keeps its own refusal naming
/// the sign rather than a dtype mismatch. A declared dtype the graph store has
/// no element type for (a narrow width, a complex pair) can hold no
/// [`PropValue`] at all, so every value is refused there rather than stored at
/// a width `to_frame` could not emit.
///
/// The match over the declared dtype is exhaustive by construction: a dtype
/// added to the vocabulary is a compile error here, not a value that slips
/// through a catch-all and detonates downstream.
///
/// # Errors
///
/// [`MolRsError::Validation`] when the value's element type cannot be stored
/// at the declared dtype, naming the key and both dtypes
/// (`'x' is declared float by the Frame schema; got string`), or when a
/// negative is offered under an unsigned key.
fn coerce_canonical(key: &str, pv: PropValue) -> Result<PropValue, MolRsError> {
    use crate::store::block::DType;

    let Some(declared) = keys::canonical_dtype(key) else {
        return Ok(pv);
    };
    let offered = match &pv {
        PropValue::F64(_) => DType::Float,
        PropValue::Int(_) => DType::Int,
        PropValue::Str(_) => DType::String,
        PropValue::Bool(_) => DType::Bool,
    };
    let refuse = || {
        Err(MolRsError::validation(format!(
            "'{key}' is declared {} by the Frame schema; got {}",
            declared.name(),
            offered.name()
        )))
    };

    match declared {
        DType::Float => match pv {
            PropValue::Int(v) => Ok(PropValue::F64(v as f64)),
            PropValue::F64(_) => Ok(pv),
            PropValue::Str(_) | PropValue::Bool(_) => refuse(),
        },
        DType::Int => match pv {
            PropValue::Int(_) => Ok(pv),
            PropValue::F64(_) | PropValue::Str(_) | PropValue::Bool(_) => refuse(),
        },
        DType::UInt => match pv {
            PropValue::Int(v) if v < 0 => Err(MolRsError::validation(format!(
                "'{key}' is declared unsigned by the Frame schema; got {v}"
            ))),
            PropValue::Int(_) => Ok(pv),
            PropValue::F64(_) | PropValue::Str(_) | PropValue::Bool(_) => refuse(),
        },
        DType::String => match pv {
            PropValue::Str(_) => Ok(pv),
            PropValue::F64(_) | PropValue::Int(_) | PropValue::Bool(_) => refuse(),
        },
        DType::Bool => match pv {
            PropValue::Bool(_) => Ok(pv),
            PropValue::F64(_) | PropValue::Int(_) | PropValue::Str(_) => refuse(),
        },
        DType::Float16
        | DType::Float32
        | DType::Int8
        | DType::Int16
        | DType::Int64
        | DType::U8
        | DType::UInt16
        | DType::UInt32
        | DType::Complex64
        | DType::Complex128 => refuse(),
    }
}

/// Materialize `table`'s column `key` into `block` at the dtype the Frame
/// schema declares for that key, carrying the column's validity mask.
///
/// [`EntityTable`] has no unsigned column, so a canonical unsigned field
/// (`id`, `mol_id`, `type_id`, …) lives in the store as [`I`] and has to be
/// re-typed on the way out: [`Block::insert`] validates the key against the
/// schema vocabulary, so handing it the signed column is *rejected* — and the
/// column would vanish from the frame instead of reaching the caller.
///
/// A component is set per entity, so a column is in general partial: the rows
/// no entity ever wrote hold the element type's default. Emitting them as
/// values would state a `frag_id` of instance zero, a charge of zero, an empty
/// name — so the mask travels with the column through
/// [`Block::insert_nullable`], which normalises the fully-populated case back
/// to a plain column.
///
/// Returns `false` when `key` names no column of `table`.
fn emit_column<K: Key>(
    block: &mut Block,
    table: &EntityTable<K>,
    key: &str,
) -> Result<bool, MolRsError> {
    use crate::store::block::DType;
    use ndarray::Array1;

    let inserted = if let Ok((data, valid)) = table.column_f64(key) {
        block.insert_nullable(key, Array1::from_vec(data.to_vec()).into_dyn(), mask(valid))
    } else if let Ok((data, valid)) = table.column_i32(key) {
        if keys::canonical_dtype(key) == Some(DType::UInt) {
            let unsigned: Vec<Idx> = data
                .iter()
                .map(|&v| {
                    Idx::try_from(v).map_err(|_| {
                        MolRsError::validation(format!(
                            "'{key}' is declared unsigned by the Frame schema; got {v}"
                        ))
                    })
                })
                .collect::<Result<_, _>>()?;
            block.insert_nullable(key, Array1::from_vec(unsigned).into_dyn(), mask(valid))
        } else {
            block.insert_nullable(key, Array1::from_vec(data.to_vec()).into_dyn(), mask(valid))
        }
    } else if let Ok((data, valid)) = table.column_str(key) {
        block.insert_nullable(key, Array1::from_vec(data.to_vec()).into_dyn(), mask(valid))
    } else if let Ok((data, valid)) = table.column_bool(key) {
        block.insert_nullable(key, Array1::from_vec(data.to_vec()).into_dyn(), mask(valid))
    } else {
        return Ok(false);
    };
    inserted.map_err(|e| MolRsError::validation(e.to_string()))?;
    Ok(true)
}

/// The block-side form of an [`EntityTable`] validity mask.
fn mask(valid: &Validity) -> Vec<bool> {
    valid.as_slice().to_vec()
}

/// A [`Block`]'s columns split by element type, each paired with its validity
/// mask — the reading counterpart of [`emit_column`].
///
/// Both halves of [`MolGraph::read_frame`] need the same thing: walk a block's
/// columns once, then ask each row for the properties it actually carries. A
/// masked-off cell holds the element type's default, which is a value like any
/// other to the block, so the mask is what keeps an unset `frag_id` from
/// arriving as instance zero.
struct MaskedColumns<'a> {
    float: Vec<MaskedColumn<'a, F>>,
    int: Vec<MaskedColumn<'a, I>>,
    uint: Vec<MaskedColumn<'a, Idx>>,
    string: Vec<MaskedColumn<'a, String>>,
    boolean: Vec<MaskedColumn<'a, bool>>,
}

/// One column of a [`Block`] as [`MaskedColumns`] reads it: its key, its dense
/// values, and its validity mask — `None` when every row holds a value.
type MaskedColumn<'a, T> = (&'a str, &'a ArrayD<T>, Option<&'a [bool]>);

impl<'a> MaskedColumns<'a> {
    /// Split `block`'s columns, skipping the keys in `skip` (a relation
    /// block's endpoint columns, which are structure rather than properties).
    ///
    /// Unsigned columns are kept apart from signed ones because the canonical
    /// `id` / `mol_id` / `type_id` fields are UInt in the Frame schema and the
    /// graph stores them signed: they need narrowing, not a cast.
    fn of(block: &'a Block, skip: &[String]) -> Self {
        let mut cols = MaskedColumns {
            float: Vec::new(),
            int: Vec::new(),
            uint: Vec::new(),
            string: Vec::new(),
            boolean: Vec::new(),
        };
        for key in block.keys() {
            if skip.iter().any(|s| s == key) {
                continue;
            }
            let mask = block.validity(key);
            if let Some(arr) = block.get_float(key) {
                cols.float.push((key, arr, mask));
            } else if let Some(arr) = block.get_int(key) {
                cols.int.push((key, arr, mask));
            } else if let Some(arr) = block.get_uint(key) {
                cols.uint.push((key, arr, mask));
            } else if let Some(arr) = block.get_string(key) {
                cols.string.push((key, arr, mask));
            } else if let Some(arr) = block.get_bool(key) {
                cols.boolean.push((key, arr, mask));
            }
        }
        cols
    }

    /// The properties row `row` carries: one entry per column whose mask marks
    /// the row as holding a value, in no particular order.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when an unsigned value exceeds the signed
    /// range the graph stores it in.
    fn cells(&self, row: usize) -> Result<Vec<(&'a str, PropValue)>, MolRsError> {
        let mut out: Vec<(&'a str, PropValue)> = Vec::new();
        for &(key, arr, mask) in &self.float {
            if is_set(mask, row) {
                #[allow(clippy::unnecessary_cast)]
                out.push((key, PropValue::F64(arr[[row]] as f64)));
            }
        }
        for &(key, arr, mask) in &self.int {
            if is_set(mask, row) {
                out.push((key, PropValue::Int(arr[[row]])));
            }
        }
        for &(key, arr, mask) in &self.uint {
            if is_set(mask, row) {
                out.push((key, PropValue::Int(narrow_uint(key, arr[[row]])?)));
            }
        }
        for &(key, arr, mask) in &self.string {
            if is_set(mask, row) {
                out.push((key, PropValue::Str(arr[[row]].clone())));
            }
        }
        for &(key, arr, mask) in &self.boolean {
            if is_set(mask, row) {
                out.push((key, PropValue::Bool(arr[[row]])));
            }
        }
        Ok(out)
    }
}

/// Whether row `row` of a column with mask `mask` holds a value. A column with
/// no mask is fully valid — see [`Block::validity`].
fn is_set(mask: Option<&[bool]>, row: usize) -> bool {
    mask.is_none_or(|mask| mask[row])
}

/// The node handles a relation row addresses, or `None` when one of its
/// endpoint columns points past the rows the `"atoms"` block had.
fn endpoints_at(
    cols: &[&ArrayD<Idx>],
    row: usize,
    node_ids: &[NodeId],
) -> Option<SmallVec<[NodeId; 4]>> {
    let mut nodes: SmallVec<[NodeId; 4]> = SmallVec::new();
    for col in cols {
        nodes.push(*node_ids.get(col[[row]] as usize)?);
    }
    Some(nodes)
}

/// Narrow one value of a frame's unsigned column to the signed [`I`] the entity
/// table stores, refusing the values that would wrap into a negative rather
/// than storing an identifier the next [`MolGraph::to_frame`] cannot re-widen.
fn narrow_uint(key: &str, v: Idx) -> Result<I, MolRsError> {
    I::try_from(v).map_err(|_| {
        MolRsError::validation(format!(
            "'{key}' value {v} exceeds the signed range the graph stores"
        ))
    })
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

    /// Register a relation kind the way a `Result`-returning constructor must:
    /// idempotent on a matching name + arity, an error — never a panic — on a
    /// conflicting one.
    ///
    /// [`register_kind`](Self::register_kind) asserts on an arity conflict
    /// because a leaf constructor registers fixed kinds, which is a programming
    /// error. A promotion such as
    /// [`Atomistic::try_from_molgraph`](crate::system::atomistic::Atomistic::try_from_molgraph)
    /// registers its standard kinds onto a **caller-supplied** graph, where a
    /// conflicting arity is a data condition and must come back as an `Err`.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] when `name` is already registered at
    /// an arity other than `arity`, naming the kind and both arities.
    pub(crate) fn try_register_kind(
        &mut self,
        name: &str,
        arity: usize,
    ) -> Result<KindId, MolRsError> {
        if let Some(kid) = self.kind_id(name) {
            let found = self.arity(kid);
            if found != arity {
                return Err(MolRsError::validation(format!(
                    "kind '{name}' is registered with arity {found}, but {arity} is required"
                )));
            }
            return Ok(kid);
        }
        Ok(self.register_kind(name, arity))
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
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a value of `payload` contradicts the
    /// element type an existing node column holds for that key (a string
    /// `tag` into an `i32` `tag` column); the message names the conflicting
    /// key. Likewise when a value cannot be stored at the dtype the Frame
    /// schema declares for a canonical key (a string under `"x"`), which is
    /// refused here rather than on the way out to a `Frame`. The bag is
    /// not always the caller's own — [`merge`](Self::merge)
    /// and the leaves' `from_frame` hand over property bags read out of
    /// another graph or a file, where a disagreeing key is data and not a
    /// defect — so the conflict is returned rather than asserted.
    ///
    /// The node is **not** rolled back: it keeps the prefix of `payload` that
    /// was written before the conflict, as `self` does for any other partial
    /// write (see [`merge`](Self::merge)).
    pub fn add_node_with(&mut self, payload: Atom) -> Result<NodeId, MolRsError> {
        let id = self.add_node();
        self.write_atom(id, &payload)?;
        Ok(id)
    }

    /// Remove a node and every relation that references it, across **all**
    /// registered kinds (registry-driven cascade). Returns the node's property
    /// bag (materialized).
    pub fn remove_node(&mut self, id: NodeId) -> Result<Atom, MolRsError> {
        Ok(self
            .remove_nodes(&[id])?
            .pop()
            .expect("one validated node produces one payload"))
    }

    /// Remove several nodes in one registry scan.
    ///
    /// This is the batch primitive reaction compilation needs: deleting one
    /// leaving group at a time scans every relation table once per edit, while
    /// deleting the compiled set scans each table exactly once.
    pub fn remove_nodes(&mut self, ids: &[NodeId]) -> Result<Vec<Atom>, MolRsError> {
        let doomed_nodes: HashSet<NodeId> = ids.iter().copied().collect();
        if doomed_nodes.len() != ids.len() {
            return Err(MolRsError::validation(
                "remove_nodes received a duplicate node handle",
            ));
        }
        for &id in ids {
            if !self.nodes.contains(id) {
                return Err(MolRsError::not_found("node", format!("NodeId {:?}", id)));
            }
        }
        let payloads = ids.iter().map(|&id| self.read_atom(id)).collect();

        for kid in 0..self.kinds.len() {
            let doomed: Vec<RelationId> = self.kinds[kid]
                .endpoints
                .iter()
                .filter(|(_, eps)| eps.iter().any(|id| doomed_nodes.contains(id)))
                .map(|(rid, _)| rid)
                .collect();
            for rid in doomed {
                self.detach_relation_from_adjacency(KindId(kid as u16), rid, None);
                let k = &mut self.kinds[kid];
                k.props.despawn(rid);
                k.endpoints.remove(rid);
            }
        }
        for &id in ids {
            self.adjacency.remove(&id);
            self.nodes.despawn(id);
        }
        Ok(payloads)
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
        match coerce_canonical(key, val.into())? {
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
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a value's element type contradicts the
    /// component the table already holds for that key, or when the value does
    /// not fit the dtype the Frame schema declares for a canonical key. The
    /// property is *not* written in that case, which is precisely why the
    /// error is returned rather than swallowed: a dropped property is a
    /// molecule that quietly lost a label.
    fn write_atom(&mut self, id: NodeId, atom: &Atom) -> Result<(), MolRsError> {
        for (key, val) in atom.iter() {
            coerce_canonical(key, val.clone()).and_then(|pv| match pv {
                PropValue::F64(v) => self.nodes.set_f64(id, key, v),
                PropValue::Int(v) => self.nodes.set_i32(id, key, v),
                PropValue::Str(s) => self.nodes.set_str(id, key, &s),
                PropValue::Bool(v) => self.nodes.set_bool(id, key, v),
            })?;
        }
        Ok(())
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
        match coerce_canonical(key, val.into())? {
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
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a value's element type contradicts the
    /// component the kind already holds for that key, or when the value does
    /// not fit the dtype the Frame schema declares for a canonical key. As on
    /// the node side, the property is not written, so the error is the only
    /// thing standing between the caller and a silently dropped bond label.
    pub(crate) fn write_relation_props(
        &mut self,
        kind: KindId,
        id: RelationId,
        props: &HashMap<String, PropValue>,
    ) -> Result<(), MolRsError> {
        let k = &mut self.kinds[kind.0 as usize];
        for (key, val) in props {
            coerce_canonical(key, val.clone()).and_then(|pv| match pv {
                PropValue::F64(v) => k.props.set_f64(id, key, v),
                PropValue::Int(v) => k.props.set_i32(id, key, v),
                PropValue::Str(s) => k.props.set_str(id, key, &s),
                PropValue::Bool(v) => k.props.set_bool(id, key, v),
            })?;
        }
        Ok(())
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
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a property of `other` cannot be written
    /// into `self` — an incoming string `tag` where `self` holds an `i32`
    /// `tag`, say. `other`'s properties are foreign data to `self`, so the two
    /// vocabularies can genuinely disagree; the merged graph would otherwise
    /// come back missing exactly the properties the caller handed over. The
    /// merge is *not* rolled back: `self` keeps what was written before the
    /// conflict, as it does for any other partial write.
    pub fn merge(&mut self, other: MolGraph) -> Result<HashMap<NodeId, NodeId>, MolRsError> {
        let mut node_map: HashMap<NodeId, NodeId> = HashMap::new();
        for old_id in other.nodes.handles() {
            let new_id = self.add_node_with(other.read_atom(old_id))?;
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
                    self.write_relation_props(self_kind, rid, &rel.props)?;
                }
            }
        }
        Ok(node_map)
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
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a column is refused by the [`Block`] it
    /// is written into; the message names the column. A value whose element
    /// type the Frame schema does not declare for its key is refused at the
    /// write ([`set_node`](Self::set_node) no longer accepts a string under
    /// `"x"`), so the property API cannot build such a column; the raw column
    /// table ([`node_table_mut`](Self::node_table_mut)) still can, because it
    /// writes an element type without consulting the key's declared dtype.
    /// That is a caller reaching past the door, not a broken invariant, so it
    /// is returned rather than asserted.
    pub(crate) fn to_frame(&self) -> Result<Frame, MolRsError> {
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
            emit_column(&mut atoms_block, &self.nodes, key)?;
        }
        if n > 0 {
            frame.insert("atoms", atoms_block);
        }

        // ---- one block per non-empty relation kind ----
        for kid in self.kind_ids() {
            let kidx = kid.0 as usize;
            if self.kinds[kidx].props.is_empty() {
                continue;
            }
            let block = self.relation_block(kid, &id_to_row)?;
            frame.insert(&self.kind_name[kidx], block);
        }

        Ok(frame)
    }

    /// The [`Frame`] block of one relation kind: the `atomi`/`atomj`/… endpoint
    /// columns in position order, addressing nodes by the row order
    /// `id_to_row` fixes, plus one column per relation property.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] if a column is refused by the block: a
    /// column of the wrong row count, or a canonical key written at a dtype
    /// the Frame schema forbids. [`to_frame`](Self::to_frame) propagates it.
    fn relation_block(
        &self,
        kind: KindId,
        id_to_row: &HashMap<NodeId, usize>,
    ) -> Result<Block, MolRsError> {
        use ndarray::Array1;

        let kidx = kind.0 as usize;
        let k = &self.kinds[kidx];
        // Relation row order: shared by the endpoint columns and the (already
        // aligned, dense) property columns.
        let rids: Vec<RelationId> = k.props.handles().collect();
        let mut block = Block::new();

        for pos in 0..self.kind_arity[kidx] {
            let col: Vec<Idx> = rids
                .iter()
                .map(|rid| id_to_row[&k.endpoints[*rid][pos]] as Idx)
                .collect();
            block
                .insert(rel_col_name(pos), Array1::from_vec(col).into_dyn())
                .map_err(|e| MolRsError::validation(e.to_string()))?;
        }

        // Property columns read straight from the column table.
        let mut prop_keys: Vec<String> = k.props.columns().map(|s| s.to_owned()).collect();
        prop_keys.sort();
        for key in &prop_keys {
            emit_column(&mut block, &k.props, key)?;
        }
        Ok(block)
    }

    /// Read a [`Frame`] into `self`: the `"atoms"` block becomes nodes; each
    /// **already-registered** kind's block (matched by name) becomes relations
    /// via its `atomi`/`atomj`/… columns, with any extra columns read back as
    /// props.
    ///
    /// # An unreadable relation block is refused, not skipped
    ///
    /// A registered kind with no block in the frame is fine — the frame simply
    /// carries no relations of that kind. The reverse is not: a *relation*
    /// block (one carrying the endpoint columns
    /// [`to_frame`](Self::to_frame) emits) whose name matches no registered
    /// kind has nowhere to go, and reading on would hand back a graph missing
    /// rows the frame plainly stated. `Atomistic::from_frame` on a
    /// `Fragment`'s frame used to drop every `ports` row exactly this way and
    /// return a molecule indistinguishable from one that never had any. Such a
    /// block is an error naming itself, so the caller can register the kind or
    /// pick the leaf type that owns it. A block that *does* name a registered
    /// kind but carries fewer endpoint columns than that kind's arity — an
    /// `angles` block with `atomi` and `atomj` and no `atomk` — is unreadable
    /// in the same way and refused in the same way.
    ///
    /// Blocks that are not relation blocks — metadata, a box, a grid — carry
    /// no endpoint column and are left alone: a frame may legitimately hold
    /// more than a graph reads.
    ///
    /// # Null cells
    ///
    /// A column's [validity mask](Block::validity) is honoured: a row the mask
    /// marks as null sets no property, so the unlabelled atoms of a partially
    /// labelled frame come back unlabelled rather than carrying the default
    /// the column had to store for them.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Parse`] when the frame has no `"atoms"` block, and
    /// [`MolRsError::Validation`] when a column value does not fit its
    /// canonical type, when a column's dtype contradicts a component `self`
    /// already holds under that key, or when the frame carries an unreadable
    /// relation block.
    pub(crate) fn read_frame(&mut self, frame: &Frame) -> Result<(), MolRsError> {
        let atoms_block = frame
            .get("atoms")
            .ok_or_else(|| MolRsError::parse("Frame missing 'atoms' block"))?;
        let node_ids = self.read_node_rows(atoms_block)?;

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
            self.read_relation_block(kid, &block_name, arity, block, &node_ids)?;
        }

        self.reject_unreadable_relation_blocks(frame)
    }

    /// Add one node per row of the `"atoms"` block, in row order, and return
    /// the handles at that order — the addressing every relation block's
    /// endpoint columns use.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a column value does not fit its
    /// canonical type, or when a property contradicts a component `self`
    /// already holds under that key.
    fn read_node_rows(&mut self, atoms: &Block) -> Result<Vec<NodeId>, MolRsError> {
        let nrows = atoms.nrows().unwrap_or(0);
        let columns = MaskedColumns::of(atoms, &[]);
        let mut node_ids: Vec<NodeId> = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let mut node = Atom::new();
            for (key, value) in columns.cells(row)? {
                node.set(key, value);
            }
            node_ids.push(self.add_node_with(node)?);
        }
        Ok(node_ids)
    }

    /// Read one relation block into the kind it names: each row becomes a
    /// relation over the nodes its endpoint columns address, carrying every
    /// other column of the block as a property.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when the block carries fewer endpoint columns
    /// than the kind's arity — the rows cannot be read at all, and reading on
    /// would hand back a graph missing relations the frame plainly stated —
    /// and when a property column's dtype contradicts the component the kind
    /// already holds under that key.
    fn read_relation_block(
        &mut self,
        kind: KindId,
        block_name: &str,
        arity: usize,
        block: &Block,
        node_ids: &[NodeId],
    ) -> Result<(), MolRsError> {
        let endpoint_names: Vec<String> = (0..arity).map(rel_col_name).collect();
        let mut endpoint_cols: Vec<&ndarray::ArrayD<Idx>> = Vec::with_capacity(arity);
        for name in &endpoint_names {
            let col = block.get_uint(name).ok_or_else(|| {
                MolRsError::validation(format!(
                    "Frame block '{block_name}' names the relation kind '{block_name}' of arity \
                     {arity} but carries no '{name}' endpoint column, so its rows cannot be read"
                ))
            })?;
            endpoint_cols.push(col);
        }

        // Non-endpoint columns are properties, read back at every dtype the
        // frame can carry (a string bond label, a UInt `type_id`, an `is_14`
        // flag), each honouring its validity mask.
        let props = MaskedColumns::of(block, &endpoint_names);

        for row in 0..block.nrows().unwrap_or(0) {
            let Some(nodes) = endpoints_at(&endpoint_cols, row, node_ids) else {
                let stated: Vec<Idx> = endpoint_cols.iter().map(|col| col[[row]]).collect();
                return Err(MolRsError::validation(format!(
                    "block '{block_name}' row {row} names an atom index outside the atoms \
                     block: endpoints {stated:?}, {} atoms",
                    node_ids.len()
                )));
            };
            // Every rejection `add_relation` knows is unreachable here (the kind is
            // registered, the arity matches by construction, the handles were just
            // minted), so a failure is surfaced rather than dropped with the row.
            let rid = self.add_relation(kind, &nodes)?;
            for (key, value) in props.cells(row)? {
                self.set_relation_prop(kind, rid, key, value)?;
            }
        }
        Ok(())
    }

    /// Refuse every relation block of `frame` that names no registered kind.
    ///
    /// [`to_frame`](Self::to_frame) writes a relation block's endpoints into
    /// the canonical columns [`atomi`](crate::store::keys::ATOMI),
    /// `atomj`, … in position order, so a block carrying `atomi` is a relation
    /// block whoever wrote it — including `ports`, which is outside the
    /// [`Frame`] vocabulary. That column is therefore the test, rather than a
    /// lookup in the block schema, which would miss every kind a caller
    /// registers under a name of its own.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] naming every offending block. The names are
    /// sorted so the message does not depend on the frame's hash order.
    fn reject_unreadable_relation_blocks(&self, frame: &Frame) -> Result<(), MolRsError> {
        let mut unreadable: Vec<&str> = frame
            .iter()
            .filter(|(name, block)| {
                !self.name_to_kind.contains_key(*name)
                    && block.get_uint(crate::store::keys::ATOMI).is_some()
            })
            .map(|(name, _)| name)
            .collect();
        if unreadable.is_empty() {
            return Ok(());
        }
        unreadable.sort_unstable();
        Err(MolRsError::validation(format!(
            "Frame block(s) [{}] carry relation endpoints but name no relation \
             kind registered on this graph, so their rows cannot be read; \
             register the kind first, or read the frame into the leaf type \
             that owns it",
            unreadable.join(", ")
        )))
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
        let id = g
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        assert_eq!(g.n_nodes(), 1);
        assert_eq!(g.get_node(id).unwrap().get_str("element"), Some("C"));
        g.remove_node(id).unwrap();
        assert_eq!(g.n_nodes(), 0);
        assert!(g.get_node(id).is_err());
    }

    /// `add_node_with` builds a node out of a property bag the caller hands
    /// it, and a bag whose value contradicts an existing node column is a
    /// data condition the caller can act on — so it is returned, not
    /// asserted by a panic.
    ///
    /// The fixture key is deliberately one the Frame schema declares nothing
    /// about: under a canonical key the setter refuses the seed's element
    /// type before a column ever exists, which is a different rule. What is
    /// tested here is the *column* the graph already holds.
    #[test]
    fn add_node_with_returns_the_conflict_when_the_payload_contradicts_a_node_column() {
        let mut g = MolGraph::new();
        let seeded = g.add_node();
        g.set_node(seeded, "tag", -0.5_f64)
            .expect("a fresh 'tag' column takes an f64");

        let mut payload = Atom::new();
        payload.set("tag", "negative");

        let err = g
            .add_node_with(payload)
            .expect_err("a str 'tag' cannot enter an f64 'tag' column");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert!(
            err.to_string().contains("'tag'"),
            "the error names the conflicting key, got {err:?}"
        );
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
        let id = g
            .add_node_with(Atom::xyz("C", 1.0, 0.0, 0.0))
            .expect("fixture node");
        crate::spatial::geometry::translate(&mut g, [10.0, 20.0, 30.0]);
        let a = g.get_node(id).unwrap();
        assert!((a.get_f64("x").unwrap() - 11.0).abs() < 1e-12);
        let id2 = g
            .add_node_with(Atom::xyz("C", 1.0, 0.0, 0.0))
            .expect("fixture node");
        crate::spatial::geometry::rotate(
            &mut g,
            [0.0, 0.0, 1.0],
            std::f64::consts::FRAC_PI_2,
            None,
        )
        .expect("the z axis is a direction");
        let b = g.get_node(id2).unwrap();
        assert!((b.get_f64("x").unwrap()).abs() < 1e-12);
        assert!((b.get_f64("y").unwrap() - 1.0).abs() < 1e-12);
    }

    // ----- Frame round-trip (generic) -----

    #[test]
    fn test_to_read_frame_roundtrip() {
        let mut g = MolGraph::new();
        let bond = g.register_kind("bonds", 2);
        let o = g
            .add_node_with(Atom::xyz("O", 0.0, 0.0, 0.0))
            .expect("fixture node");
        let h1 = g
            .add_node_with(Atom::xyz("H", 0.96, 0.0, 0.0))
            .expect("fixture node");
        let h2 = g
            .add_node_with(Atom::xyz("H", -0.24, 0.93, 0.0))
            .expect("fixture node");
        g.add_relation(bond, &[o, h1]).unwrap();
        g.add_relation(bond, &[o, h2]).unwrap();
        let frame = g.to_frame().expect("a schema-conforming graph converts");
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
        let a = g
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        let b = g
            .add_node_with(Atom::xyz("C", 1.5, 0.0, 0.0))
            .expect("fixture node");
        let rid = g.add_relation(bond, &[a, b]).unwrap();
        // A genuinely float-valued relation prop. `bond_number` is not one —
        // it is a schema uint — and a *computed* fractional bond order is
        // exactly the kind of quantity that gets its own key.
        g.set_relation_prop(bond, rid, "wiberg_index", 1.47_f64)
            .unwrap();
        g.set_relation_prop(bond, rid, "ring_size", PropValue::Int(6))
            .unwrap();
        g.set_relation_prop(bond, rid, "label", PropValue::Str("aromatic".to_owned()))
            .unwrap();

        let frame = g.to_frame().expect("a schema-conforming graph converts");

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
        match r.props.get("wiberg_index") {
            Some(PropValue::F64(v)) => assert!((v - 1.47).abs() < 1e-12),
            other => panic!("float relation prop lost: {other:?}"),
        }
    }

    #[test]
    fn test_to_frame_keeps_schema_unsigned_node_columns() {
        // `id` / `mol_id` are UInt in the Frame schema but live in the node
        // table as I. Emitting them signed made `Block::insert` reject the
        // column against the schema — and, with the error swallowed, the
        // identifiers vanished from the frame entirely.
        let mut g = MolGraph::new();
        let a = g
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        let b = g
            .add_node_with(Atom::xyz("H", 1.1, 0.0, 0.0))
            .expect("fixture node");
        g.set_node(a, "id", PropValue::Int(1)).unwrap();
        g.set_node(b, "id", PropValue::Int(2)).unwrap();
        g.set_node(a, "mol_id", PropValue::Int(7)).unwrap();
        g.set_node(b, "mol_id", PropValue::Int(7)).unwrap();

        let atoms = &g.to_frame().expect("a schema-conforming graph converts")["atoms"];
        assert_eq!(
            atoms
                .get_uint("id")
                .expect("'id' reaches the frame")
                .iter()
                .copied()
                .collect::<Vec<Idx>>(),
            vec![1, 2]
        );
        assert_eq!(
            atoms
                .get_uint("mol_id")
                .expect("'mol_id' reaches the frame")
                .iter()
                .copied()
                .collect::<Vec<Idx>>(),
            vec![7, 7]
        );
    }

    #[test]
    fn test_to_frame_keeps_bool_node_columns() {
        let mut g = MolGraph::new();
        let a = g
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        let b = g
            .add_node_with(Atom::xyz("H", 1.1, 0.0, 0.0))
            .expect("fixture node");
        g.set_node(a, "frozen", true).unwrap();
        g.set_node(b, "frozen", false).unwrap();

        assert_eq!(
            g.to_frame().expect("a schema-conforming graph converts")["atoms"]
                .get_bool("frozen")
                .expect("bool column reaches the frame")
                .iter()
                .copied()
                .collect::<Vec<bool>>(),
            vec![true, false]
        );
    }

    #[test]
    fn test_read_frame_restores_unsigned_and_bool_node_props() {
        let mut g = MolGraph::new();
        let a = g
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        g.set_node(a, "id", PropValue::Int(4)).unwrap();
        g.set_node(a, "mol_id", PropValue::Int(9)).unwrap();
        g.set_node(a, "frozen", true).unwrap();

        let mut g2 = MolGraph::new();
        g2.read_frame(&g.to_frame().expect("a schema-conforming graph converts"))
            .unwrap();

        let (_, atom) = g2.nodes().next().expect("node round-trips");
        assert_eq!(atom.get("id"), Some(&PropValue::Int(4)));
        assert_eq!(atom.get("mol_id"), Some(&PropValue::Int(9)));
        assert_eq!(atom.get("frozen"), Some(&PropValue::Bool(true)));
    }

    /// The Frame schema declares `x` a float, so a string is not a narrower
    /// or wider `x` — it is a different kind of thing. The door that owns the
    /// key's dtype is the setter, which has a `Result`; letting the value in
    /// and refusing it at `to_frame` leaves the graph holding a column no
    /// frame can ever carry.
    #[test]
    fn set_node_refuses_a_str_under_the_schema_float_key_x() {
        use crate::store::block::DType;
        let mut g = MolGraph::new();
        let n = g.add_node();

        let err = g
            .set_node(n, "x", "left")
            .expect_err("a str cannot be stored at a schema-float key");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("'x'"), "the error names the key, got {msg}");
        assert!(
            msg.contains(DType::Float.name()),
            "the error names the declared dtype, got {msg}"
        );
        assert!(
            msg.contains(DType::String.name()),
            "the error names the offered dtype, got {msg}"
        );
    }

    /// Same refusal under a different float key: the rule is the schema's, not
    /// a special case carved out for coordinates.
    #[test]
    fn set_node_refuses_a_str_under_the_schema_float_key_charge() {
        use crate::store::block::DType;
        let mut g = MolGraph::new();
        let n = g.add_node();

        let err = g
            .set_node(n, "charge", "negative")
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

    /// A bool is numeric-adjacent and still not a float: `true` is not `1.0`
    /// under a key the schema declares float.
    #[test]
    fn set_node_refuses_a_bool_under_a_schema_float_key() {
        use crate::store::block::DType;
        let mut g = MolGraph::new();
        let n = g.add_node();

        let err = g
            .set_node(n, "y", true)
            .expect_err("a bool cannot be stored at a schema-float key");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("'y'"), "the error names the key, got {msg}");
        assert!(
            msg.contains(DType::Float.name()) && msg.contains(DType::Bool.name()),
            "the error names both dtypes, got {msg}"
        );
    }

    /// The refusal runs in both directions: `element` is declared a string, so
    /// an atomic number written there is refused rather than creating an int
    /// column under a string key.
    #[test]
    fn set_node_refuses_an_int_under_a_schema_string_key() {
        use crate::store::block::DType;
        let mut g = MolGraph::new();
        let n = g.add_node();

        let err = g
            .set_node(n, "element", PropValue::Int(6 as I))
            .expect_err("an int cannot be stored at a schema-string key");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("'element'"),
            "the error names the key, got {msg}"
        );
        assert!(
            msg.contains(DType::String.name()) && msg.contains(DType::Int.name()),
            "the error names both dtypes, got {msg}"
        );
    }

    /// Width is not semantics: an int under a float key is still a number, so
    /// it is widened rather than refused, and reads back as the float the
    /// schema declares.
    #[test]
    fn set_node_widens_an_int_under_a_schema_float_key() {
        let mut g = MolGraph::new();
        let n = g.add_node();

        g.set_node(n, "x", PropValue::Int(1 as I))
            .expect("an int is a number and widens into a float key");
        assert_eq!(
            g.get_node(n).expect("node exists").get("x"),
            Some(&PropValue::F64(1.0))
        );
    }

    /// An unsigned key is not a numeric key: `1.0` is a float, and storing it
    /// under `id` leaves the graph holding a float column the Frame schema
    /// declares unsigned — a column no frame can carry. The setter owns the
    /// key's dtype, so it refuses there.
    #[test]
    fn set_node_refuses_an_f64_under_a_schema_uint_key() {
        use crate::store::block::DType;
        let mut g = MolGraph::new();
        let n = g.add_node();

        let err = g
            .set_node(n, "id", 1.0_f64)
            .expect_err("an f64 cannot be stored at a schema-uint key");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("'id'"), "the error names the key, got {msg}");
        assert!(
            msg.contains(DType::UInt.name()) && msg.contains(DType::Float.name()),
            "the error names both dtypes, got {msg}"
        );
    }

    /// The refusal above is about the element type, not about `id` being
    /// closed: a non-negative int is an identifier and still goes in, and
    /// reads back as the int the store holds it as.
    #[test]
    fn set_node_accepts_an_int_under_a_schema_uint_key() {
        let mut g = MolGraph::new();
        let n = g.add_node();

        g.set_node(n, "id", 1 as I)
            .expect("a non-negative int is an identifier");
        assert_eq!(
            g.get_node(n).expect("node exists").get("id"),
            Some(&PropValue::Int(1))
        );
    }

    /// Tightening the element-type door must not restate the sign rule: a
    /// negative under an unsigned key keeps refusing with the message that
    /// names the sign, not a dtype mismatch.
    #[test]
    fn set_node_keeps_the_unsigned_message_for_a_negative_under_a_uint_key() {
        let mut g = MolGraph::new();
        let n = g.add_node();

        let err = g
            .set_node(n, "id", PropValue::Int(-1 as I))
            .expect_err("a negative is not an identifier");
        assert!(
            err.to_string()
                .contains("declared unsigned by the Frame schema"),
            "the sign refusal keeps its own message, got {err}"
        );
    }

    /// The vocabulary is closed but the key space is open: a key the schema
    /// declares nothing about carries whatever the caller stores, because no
    /// frame column type contradicts it.
    #[test]
    fn set_node_accepts_a_str_under_a_key_the_schema_does_not_declare() {
        let mut g = MolGraph::new();
        let n = g.add_node();

        g.set_node(n, "tag", "left")
            .expect("'tag' is not in the canonical vocabulary");
        assert_eq!(
            g.get_node(n).expect("node exists").get_str("tag"),
            Some("left")
        );
    }

    #[test]
    fn test_set_node_rejects_negative_under_unsigned_key() {
        // A sign is semantics, not width: the Frame schema declares `id`
        // unsigned, so this must fail where a Result exists rather than blow up
        // later in `to_frame`, which has none.
        let mut g = MolGraph::new();
        let a = g.add_node();
        assert!(g.set_node(a, "id", PropValue::Int(-1)).is_err());
        assert!(g.set_node(a, "id", PropValue::Int(3)).is_ok());
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

        let map = dst.merge(src).expect("merge succeeds on compatible graphs");
        assert_eq!(dst.n_nodes(), 6);
        assert_eq!(dst.n_relations(dbond), 2);
        assert_eq!(dst.n_relations(dimp), 1, "merge must carry impropers");
        assert_eq!(map.len(), 4, "merge returns one entry per node of other");
    }

    #[test]
    fn test_clone_preserves_handles() {
        let mut g = MolGraph::new();
        let bond = g.register_kind("bonds", 2);
        let a = g
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        let b = g
            .add_node_with(Atom::xyz("H", 1.0, 0.0, 0.0))
            .expect("fixture node");
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
        let id = g
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        g.add_node_with(Atom::xyz("H", 1.0, 0.0, 0.0))
            .expect("fixture node");
        let g2 = g.clone();
        g.set_node(id, "x", 99.0).unwrap();
        assert_eq!(g2.get_node(id).unwrap().get_f64("x"), Some(0.0));
        assert_eq!(g2.n_nodes(), 2);
    }

    // ----- Contract B: read_frame and the node/relation writers keep data -----

    /// A relation property column whose dtype contradicts the component the
    /// kind already holds cannot be stored, and dropping the value hands back
    /// a graph whose bonds silently lost the label the frame carried.
    #[test]
    fn read_frame_rejects_a_relation_prop_whose_dtype_conflicts_with_the_kind() {
        use ndarray::Array1;

        let mut graph = MolGraph::new();
        let bonds = graph.register_kind("bonds", 2);
        let a = graph.add_node_with(Atom::new()).expect("fixture node");
        let b = graph.add_node_with(Atom::new()).expect("fixture node");
        let rid = graph.add_relation(bonds, &[a, b]).unwrap();
        graph
            .set_relation_prop(bonds, rid, "tag", PropValue::Int(1 as I))
            .unwrap();

        let mut atoms = Block::new();
        atoms
            .insert(
                "element",
                Array1::from_vec(vec!["C".to_owned(), "C".to_owned()]).into_dyn(),
            )
            .unwrap();
        let mut bonds_block = Block::new();
        bonds_block
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn())
            .unwrap();
        bonds_block
            .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn())
            .unwrap();
        bonds_block
            .insert(
                "tag",
                Array1::from_vec(vec!["single".to_owned()]).into_dyn(),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds_block);

        let err = graph
            .read_frame(&frame)
            .expect_err("a str 'tag' cannot enter an int 'tag' component");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    /// `merge` moves node property bags into `self`'s columns. A bag whose
    /// dtype contradicts an existing column cannot be written, and dropping it
    /// returns a merged graph missing a property the caller handed over.
    #[test]
    fn merge_rejects_a_node_prop_whose_dtype_conflicts_with_an_existing_column() {
        let mut dst = MolGraph::new();
        let kept = dst.add_node_with(Atom::new()).expect("fixture node");
        dst.set_node(kept, "tag", PropValue::Int(1 as I)).unwrap();

        let mut src = MolGraph::new();
        let mut incoming = Atom::new();
        incoming.set("tag", "single");
        src.add_node_with(incoming).expect("fixture node");

        let err = dst
            .merge(src)
            .expect_err("a str 'tag' cannot enter an int 'tag' component");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    /// The same rule on the relation side: `merge` writes the incoming
    /// relation's property bag into the kind it matched by name.
    #[test]
    fn merge_rejects_a_relation_prop_whose_dtype_conflicts_with_an_existing_component() {
        let mut dst = MolGraph::new();
        let bonds = dst.register_kind("bonds", 2);
        let a = dst.add_node_with(Atom::new()).expect("fixture node");
        let b = dst.add_node_with(Atom::new()).expect("fixture node");
        let rid = dst.add_relation(bonds, &[a, b]).unwrap();
        dst.set_relation_prop(bonds, rid, "tag", PropValue::Int(1 as I))
            .unwrap();

        let mut src = MolGraph::new();
        let src_bonds = src.register_kind("bonds", 2);
        let c = src.add_node_with(Atom::new()).expect("fixture node");
        let d = src.add_node_with(Atom::new()).expect("fixture node");
        let src_rid = src.add_relation(src_bonds, &[c, d]).unwrap();
        src.set_relation_prop(src_bonds, src_rid, "tag", "single")
            .unwrap();

        let err = dst
            .merge(src)
            .expect_err("a str 'tag' cannot enter an int 'tag' component");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    // ----- Reserved containment axis -----

    #[test]
    fn test_groups_reserved_empty() {
        let g = MolGraph::new();
        assert_eq!(g.groups.len(), 0);
    }
}
