//! Canonical column vocabulary: what a column key means and how it is stored.

use crate::store::block::DType;

/// Structural shape of a column beyond axis 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColShape {
    /// `(nrows,)` — one value per row.
    Scalar,
    /// `(nrows, n)` — a fixed-width vector per row.
    Vec(usize),
}

impl ColShape {
    /// Whether an ndarray shape is admissible for this spec.
    pub fn admits(&self, shape: &[usize]) -> bool {
        match self {
            ColShape::Scalar => shape.len() == 1,
            ColShape::Vec(n) => shape.len() == 2 && shape[1] == *n,
        }
    }
}

impl std::fmt::Display for ColShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColShape::Scalar => write!(f, "scalar"),
            ColShape::Vec(n) => write!(f, "vec({n})"),
        }
    }
}

/// One canonical column of the Frame vocabulary.
///
/// A spec binds a **column key**, wherever that key appears — in `atoms`, in
/// `bonds`, in a relation block `MolGraph::to_frame` minted on the fly. That is
/// what lets [`Block::insert`](crate::store::block::Block::insert) enforce
/// dtype without knowing which block it is about to live in.
///
/// The vocabulary is closed; the *block* set is open. A key with no spec is
/// unconstrained — that is the extension point for perceived facts,
/// per-instance force-field parameters, and format-local columns.
#[derive(Debug, Clone, Copy)]
pub struct ColumnSpec {
    /// Canonical key as it appears in a `Block` (`"x"`, `"atomi"`).
    pub key: &'static str,
    /// Rust/Python constant name (`"X"`, `"ATOMI"`), so `keys::X` and
    /// `molrs.keys.X` are generated from this table rather than hand-mirrored.
    pub const_name: &'static str,
    /// The one admissible storage dtype. Not a set — see the module doc on
    /// [`super`] for why a key that needs two dtypes is two keys.
    pub dtype: DType,
    /// Shape beyond axis 0.
    pub shape: ColShape,
    /// Physical unit as a bare symbol (`"angstrom"`, `"e"`, `"amu"`, `""` for
    /// dimensionless). Documentation and IO-boundary conversion only; never
    /// enforced — molrs stores raw numbers.
    pub unit: &'static str,
    /// One-line meaning. Never empty (asserted by the vocabulary gate).
    pub doc: &'static str,
}
