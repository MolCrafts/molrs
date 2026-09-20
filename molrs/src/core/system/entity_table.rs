//! Aligned columnar component store for the ECS world.
//!
//! [`EntityTable`] stores entities — identified by stable generational
//! [`slotmap`] handles — as **rows**, and their components as **shared-row
//! aligned dense columns + per-column validity masks** (the Arrow / dataframe
//! model):
//!
//! - One `handle → row` map ([`keys`](EntityTable)) is shared by every column,
//!   so column `i` and column `i` always describe the *same* entity. That makes
//!   a column a contiguous `&[T]` of length `n_rows` — directly mappable to a
//!   zero-copy numpy view *and* already aligned for tabular projection
//!   (`to_frame`), with no gather.
//! - Sparsity is expressed by a per-column [`Validity`] mask: an entity may
//!   have `charge` but not `port`; the row's validity bit is simply unset.
//! - Columns are created lazily on first write and their element type is fixed
//!   at that point; a later write of a different type is a hard error (no
//!   silent coercion).
//! - Deletion is swap-remove: the moved row's *handle* is unchanged (handles
//!   are the stable identity), only its internal row index compacts.
//!
//! This is the storage substrate for [`crate::system::molgraph::MolGraph`] under the
//! ECS refactor; it is generic over the slotmap key type so the same machinery
//! backs both the node table and each relation-kind table.

use std::collections::HashMap;

use slotmap::{Key, SlotMap};

use crate::error::MolRsError;
use crate::types::{F, I};

/// Per-column validity mask — one flag per row (`true` ⇒ the row holds a value).
///
/// Backed by `Vec<bool>` (one byte per row) for a directly mappable mask; a
/// packed-bit representation is a future optimization that does not change the
/// API.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Validity(Vec<bool>);

impl Validity {
    fn with_len(n: usize) -> Self {
        Validity(vec![false; n])
    }
    /// Whether `row` holds a value.
    pub fn get(&self, row: usize) -> bool {
        self.0.get(row).copied().unwrap_or(false)
    }
    fn set(&mut self, row: usize, v: bool) {
        self.0[row] = v;
    }
    fn push(&mut self, v: bool) {
        self.0.push(v);
    }
    fn swap_remove(&mut self, row: usize) {
        self.0.swap_remove(row);
    }
    /// The mask as a contiguous slice (zero-copy mappable).
    pub fn as_slice(&self) -> &[bool] {
        &self.0
    }
    /// Number of rows.
    pub fn len(&self) -> usize {
        self.0.len()
    }
    /// Whether the mask is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A typed component column: a dense `Vec<T>` aligned to the table's rows plus a
/// [`Validity`] mask. `data[i]` is meaningful iff `validity().get(i)`.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    /// 64-bit float column.
    F64(Vec<F>, Validity),
    /// 32-bit integer column.
    I32(Vec<I>, Validity),
    /// UTF-8 string column.
    Str(Vec<String>, Validity),
    /// Boolean column.
    Bool(Vec<bool>, Validity),
}

impl Column {
    /// Element type name (for error messages).
    pub fn type_name(&self) -> &'static str {
        match self {
            Column::F64(..) => "f64",
            Column::I32(..) => "i32",
            Column::Str(..) => "str",
            Column::Bool(..) => "bool",
        }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            Column::F64(d, _) => d.len(),
            Column::I32(d, _) => d.len(),
            Column::Str(d, _) => d.len(),
            Column::Bool(d, _) => d.len(),
        }
    }

    /// Whether the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The column's validity mask.
    pub fn validity(&self) -> &Validity {
        match self {
            Column::F64(_, v) => v,
            Column::I32(_, v) => v,
            Column::Str(_, v) => v,
            Column::Bool(_, v) => v,
        }
    }

    /// Append one null slot (default value, validity unset) — keeps the column
    /// aligned when a new row is spawned.
    fn push_null(&mut self) {
        match self {
            Column::F64(d, v) => {
                d.push(0.0);
                v.push(false);
            }
            Column::I32(d, v) => {
                d.push(0);
                v.push(false);
            }
            Column::Str(d, v) => {
                d.push(String::new());
                v.push(false);
            }
            Column::Bool(d, v) => {
                d.push(false);
                v.push(false);
            }
        }
    }

    /// Swap-remove `row` (move the last row into `row`, drop the last).
    fn swap_remove(&mut self, row: usize) {
        match self {
            Column::F64(d, v) => {
                d.swap_remove(row);
                v.swap_remove(row);
            }
            Column::I32(d, v) => {
                d.swap_remove(row);
                v.swap_remove(row);
            }
            Column::Str(d, v) => {
                d.swap_remove(row);
                v.swap_remove(row);
            }
            Column::Bool(d, v) => {
                d.swap_remove(row);
                v.swap_remove(row);
            }
        }
    }

    fn set_valid(&mut self, row: usize, v: bool) {
        match self {
            Column::F64(_, mask) => mask.set(row, v),
            Column::I32(_, mask) => mask.set(row, v),
            Column::Str(_, mask) => mask.set(row, v),
            Column::Bool(_, mask) => mask.set(row, v),
        }
    }
}

/// A borrowed view of one component value (the dynamic counterpart to the typed
/// `get_*` accessors). Lets a caller read a cell without statically knowing its
/// element type — used to materialize a node's full property set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cell<'a> {
    /// 64-bit float value.
    F64(F),
    /// 32-bit integer value.
    I32(I),
    /// String value.
    Str(&'a str),
    /// Boolean value.
    Bool(bool),
}

fn missing(key: &str) -> MolRsError {
    MolRsError::NotFound {
        entity: "component",
        message: format!("component '{key}' is absent for this entity"),
    }
}

fn type_conflict(key: &str, want: &str, got: &str) -> MolRsError {
    MolRsError::Validation {
        message: format!("component '{key}' is typed {got}, not {want}"),
    }
}

/// Borrow the value at `row` of `col` as a [`Cell`]. Caller guarantees the row
/// is valid (present).
fn cell_at(col: &Column, row: usize) -> Cell<'_> {
    match col {
        Column::F64(d, _) => Cell::F64(d[row]),
        Column::I32(d, _) => Cell::I32(d[row]),
        Column::Str(d, _) => Cell::Str(&d[row]),
        Column::Bool(d, _) => Cell::Bool(d[row]),
    }
}

/// A stable-handle, aligned-column entity table — the ECS storage substrate.
///
/// Generic over the slotmap key type `K`, so the same machinery backs both the
/// node table and each relation-kind table.
#[derive(Debug, Clone)]
pub struct EntityTable<K: Key> {
    /// `handle → row index`.
    keys: SlotMap<K, u32>,
    /// `row index → handle` (iteration / alignment order).
    rows: Vec<K>,
    /// Component columns, keyed by name; every column has length `rows.len()`.
    cols: HashMap<String, Column>,
}

impl<K: Key> Default for EntityTable<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key> EntityTable<K> {
    /// An empty table.
    pub fn new() -> Self {
        Self {
            keys: SlotMap::with_key(),
            rows: Vec::new(),
            cols: HashMap::new(),
        }
    }

    /// Number of live entities (rows).
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the table has no entities.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Whether `k` is a live handle in this table.
    pub fn contains(&self, k: K) -> bool {
        self.keys.contains_key(k)
    }

    /// Live handles in row (alignment) order.
    pub fn handles(&self) -> impl Iterator<Item = K> + '_ {
        self.rows.iter().copied()
    }

    /// The internal row index of `k`, if live. O(1).
    pub fn row(&self, k: K) -> Option<usize> {
        self.keys.get(k).map(|&r| r as usize)
    }

    /// Registered component names.
    pub fn columns(&self) -> impl Iterator<Item = &str> {
        self.cols.keys().map(String::as_str)
    }

    /// The validity mask of column `key`, regardless of element type
    /// (`None` if the column is absent). Aligned to row order.
    pub fn col_validity(&self, key: &str) -> Option<&Validity> {
        self.cols.get(key).map(Column::validity)
    }

    /// The typed column `key` together with its validity, or `None` when no
    /// entity has ever carried the component.
    pub fn column(&self, key: &str) -> Option<&Column> {
        self.cols.get(key)
    }

    /// Spawn a new entity: appends a null row across all existing columns and
    /// returns its stable handle. O(n_columns).
    pub fn spawn(&mut self) -> K {
        let row = self.rows.len() as u32;
        let k = self.keys.insert(row);
        self.rows.push(k);
        for col in self.cols.values_mut() {
            col.push_null();
        }
        k
    }

    /// Despawn `k` via swap-remove. Returns `false` if `k` is not live.
    ///
    /// Every *other* handle stays valid — only the row previously at the end is
    /// moved into `k`'s slot, and that entity's handle is unchanged (just its
    /// internal row index). Handles are the stable identity; rows are internal.
    pub fn despawn(&mut self, k: K) -> bool {
        let row = match self.keys.get(k) {
            Some(&r) => r as usize,
            None => return false,
        };
        self.keys.remove(k);
        self.rows.swap_remove(row);
        if row < self.rows.len() {
            let moved = self.rows[row];
            self.keys[moved] = row as u32;
        }
        for col in self.cols.values_mut() {
            col.swap_remove(row);
        }
        true
    }

    fn require_row(&self, k: K) -> Result<usize, MolRsError> {
        self.row(k).ok_or(MolRsError::NotFound {
            entity: "entity",
            message: "stale or unknown entity handle".to_owned(),
        })
    }

    /// Whether entity `k` currently holds component `key`.
    pub fn has(&self, k: K, key: &str) -> bool {
        match self.row(k) {
            Some(row) => self.cols.get(key).is_some_and(|c| c.validity().get(row)),
            None => false,
        }
    }

    /// Read one component value of entity `k` without statically knowing its
    /// element type. `None` if the handle is stale, the column is absent, or the
    /// value is null for this entity.
    pub fn value(&self, k: K, key: &str) -> Option<Cell<'_>> {
        let row = self.row(k)?;
        let col = self.cols.get(key)?;
        if !col.validity().get(row) {
            return None;
        }
        Some(cell_at(col, row))
    }

    /// Iterate over every present `(component_name, value)` of entity `k` (skips
    /// null/absent components). Empty for a stale handle.
    pub fn row_cells(&self, k: K) -> impl Iterator<Item = (&str, Cell<'_>)> {
        let row = self.row(k);
        self.cols.iter().filter_map(move |(name, col)| {
            let row = row?;
            if !col.validity().get(row) {
                return None;
            }
            Some((name.as_str(), cell_at(col, row)))
        })
    }

    /// Clear component `key` for entity `k` (set null). No-op if the column or
    /// value is absent. Errors only on a stale handle.
    pub fn clear(&mut self, k: K, key: &str) -> Result<(), MolRsError> {
        let row = self.require_row(k)?;
        if let Some(col) = self.cols.get_mut(key) {
            col.set_valid(row, false);
        }
        Ok(())
    }
}

/// Generates the typed `set_*` / `get_*` / `column_*` accessor trio for one
/// element type, keeping the lazy-create + type-conflict + null logic in one
/// place.
macro_rules! typed_accessors {
    ($set:ident, $get:ident, $col:ident, $variant:ident, $ty:ty, $name:literal) => {
        impl<K: Key> EntityTable<K> {
            #[doc = concat!("Set the `", $name, "` component `key` on entity `k`, creating the column on first use.")]
            ///
            /// Errors on a stale handle, or if `key` already exists with a
            /// different element type (no silent coercion).
            pub fn $set(&mut self, k: K, key: &str, val: $ty) -> Result<(), MolRsError> {
                let row = self.require_row(k)?;
                let n = self.rows.len();
                match self.cols.get_mut(key) {
                    None => {
                        let mut data = vec![<$ty>::default(); n];
                        let mut valid = Validity::with_len(n);
                        data[row] = val;
                        valid.set(row, true);
                        self.cols
                            .insert(key.to_owned(), Column::$variant(data, valid));
                    }
                    Some(Column::$variant(data, valid)) => {
                        data[row] = val;
                        valid.set(row, true);
                    }
                    Some(other) => {
                        return Err(type_conflict(key, $name, other.type_name()));
                    }
                }
                Ok(())
            }

            #[doc = concat!("Get the `", $name, "` component `key` of entity `k`.")]
            ///
            /// Errors if the handle is stale, the value is absent (column
            /// missing or null for this entity), or the column has a different
            /// element type. Strict by design — no fallback default.
            pub fn $get(&self, k: K, key: &str) -> Result<$ty, MolRsError> {
                let row = self.require_row(k)?;
                match self.cols.get(key) {
                    Some(Column::$variant(data, valid)) if valid.get(row) => {
                        Ok(data[row].clone())
                    }
                    Some(Column::$variant(..)) => Err(missing(key)),
                    Some(other) => Err(type_conflict(key, $name, other.type_name())),
                    None => Err(missing(key)),
                }
            }

            #[doc = concat!("Borrow the whole `", $name, "` column `key` (zero-copy slice) plus its validity mask.")]
            ///
            /// The slice has length `len()` and is aligned to row order. Errors
            /// if the column is absent or has a different element type.
            pub fn $col(&self, key: &str) -> Result<(&[$ty], &Validity), MolRsError> {
                match self.cols.get(key) {
                    Some(Column::$variant(data, valid)) => Ok((data, valid)),
                    Some(other) => Err(type_conflict(key, $name, other.type_name())),
                    None => Err(missing(key)),
                }
            }
        }
    };
}

typed_accessors!(set_f64, get_f64, column_f64, F64, F, "f64");
typed_accessors!(set_i32, get_i32, column_i32, I32, I, "i32");
typed_accessors!(set_bool, get_bool, column_bool, Bool, bool, "bool");

impl<K: Key> EntityTable<K> {
    /// Mutably borrow the whole `f64` column `key` (zero-copy slice) plus its
    /// validity mask, for in-place vectorized updates over the dense, row-aligned
    /// column (rows are compacted, so the slice spans exactly the live entities).
    /// Errors if the column is absent or has a different element type.
    pub fn column_f64_mut(&mut self, key: &str) -> Result<(&mut [F], &Validity), MolRsError> {
        match self.cols.get_mut(key) {
            Some(Column::F64(data, valid)) => Ok((data.as_mut_slice(), &*valid)),
            Some(other) => Err(type_conflict(key, "f64", other.type_name())),
            None => Err(missing(key)),
        }
    }
}

// `Str` accessors are written by hand: `get` borrows rather than clones, and
// `set` takes `&str`.
impl<K: Key> EntityTable<K> {
    /// Set the string component `key` on entity `k` (creates the column on
    /// first use). Errors on a stale handle or an element-type conflict.
    pub fn set_str(&mut self, k: K, key: &str, val: &str) -> Result<(), MolRsError> {
        let row = self.require_row(k)?;
        let n = self.rows.len();
        match self.cols.get_mut(key) {
            None => {
                let mut data = vec![String::new(); n];
                let mut valid = Validity::with_len(n);
                data[row] = val.to_owned();
                valid.set(row, true);
                self.cols.insert(key.to_owned(), Column::Str(data, valid));
            }
            Some(Column::Str(data, valid)) => {
                data[row] = val.to_owned();
                valid.set(row, true);
            }
            Some(other) => return Err(type_conflict(key, "str", other.type_name())),
        }
        Ok(())
    }

    /// Borrow the string component `key` of entity `k`. Errors if stale, absent,
    /// or a different element type.
    pub fn get_str(&self, k: K, key: &str) -> Result<&str, MolRsError> {
        let row = self.require_row(k)?;
        match self.cols.get(key) {
            Some(Column::Str(data, valid)) if valid.get(row) => Ok(&data[row]),
            Some(Column::Str(..)) => Err(missing(key)),
            Some(other) => Err(type_conflict(key, "str", other.type_name())),
            None => Err(missing(key)),
        }
    }

    /// Borrow the whole string column `key` (slice) plus its validity mask.
    pub fn column_str(&self, key: &str) -> Result<(&[String], &Validity), MolRsError> {
        match self.cols.get(key) {
            Some(Column::Str(data, valid)) => Ok((data, valid)),
            Some(other) => Err(type_conflict(key, "str", other.type_name())),
            None => Err(missing(key)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::new_key_type;

    new_key_type! {
        struct TestId;
    }

    type T = EntityTable<TestId>;

    #[test]
    fn spawn_assigns_rows_in_order() {
        let mut t = T::new();
        let a = t.spawn();
        let b = t.spawn();
        let c = t.spawn();
        assert_eq!(t.len(), 3);
        assert_eq!(t.row(a), Some(0));
        assert_eq!(t.row(b), Some(1));
        assert_eq!(t.row(c), Some(2));
        assert_eq!(t.handles().collect::<Vec<_>>(), vec![a, b, c]);
        assert!(t.contains(a) && t.contains(b) && t.contains(c));
    }

    #[test]
    fn lazy_column_creation_and_typed_get() {
        let mut t = T::new();
        let a = t.spawn();
        let b = t.spawn();
        // No column yet → strict get errors.
        assert!(t.get_f64(a, "charge").is_err());
        t.set_f64(a, "charge", 0.5).unwrap();
        assert_eq!(t.get_f64(a, "charge").unwrap(), 0.5);
        // `b` shares the column but has no value → null → error, not a default.
        assert!(t.get_f64(b, "charge").is_err());
        assert!(!t.has(b, "charge"));
        assert!(t.has(a, "charge"));
    }

    #[test]
    fn sparse_via_null_mask() {
        let mut t = T::new();
        let a = t.spawn();
        let b = t.spawn();
        t.set_f64(a, "charge", 1.0).unwrap();
        t.set_str(b, "port", "head").unwrap();
        // a has charge not port; b has port not charge.
        assert!(t.has(a, "charge") && !t.has(a, "port"));
        assert!(t.has(b, "port") && !t.has(b, "charge"));
        // Columns are aligned to length n_rows regardless of sparsity.
        let (data, valid) = t.column_f64("charge").unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(valid.as_slice(), &[true, false]);
    }

    #[test]
    fn columns_share_row_order_alignment() {
        let mut t = T::new();
        let a = t.spawn();
        let b = t.spawn();
        t.set_f64(a, "x", 1.0).unwrap();
        t.set_f64(b, "x", 2.0).unwrap();
        t.set_str(a, "el", "C").unwrap();
        t.set_str(b, "el", "O").unwrap();
        let (xs, _) = t.column_f64("x").unwrap();
        let (els, _) = t.column_str("el").unwrap();
        // Row i of every column is the same entity.
        assert_eq!(xs[t.row(a).unwrap()], 1.0);
        assert_eq!(els[t.row(a).unwrap()], "C");
        assert_eq!(xs[t.row(b).unwrap()], 2.0);
        assert_eq!(els[t.row(b).unwrap()], "O");
    }

    #[test]
    fn type_conflict_is_an_error() {
        let mut t = T::new();
        let a = t.spawn();
        t.set_f64(a, "k", 1.0).unwrap();
        // Re-typing the same column is rejected, not silently coerced.
        assert!(t.set_str(a, "k", "x").is_err());
        assert!(t.get_str(a, "k").is_err());
        assert!(t.set_i32(a, "k", 3).is_err());
        // Original value intact.
        assert_eq!(t.get_f64(a, "k").unwrap(), 1.0);
    }

    #[test]
    fn despawn_swap_remove_keeps_other_handles_stable() {
        let mut t = T::new();
        let a = t.spawn();
        let b = t.spawn();
        let c = t.spawn();
        t.set_f64(a, "x", 10.0).unwrap();
        t.set_f64(b, "x", 20.0).unwrap();
        t.set_f64(c, "x", 30.0).unwrap();
        // Remove the middle entity.
        assert!(t.despawn(b));
        assert_eq!(t.len(), 2);
        assert!(!t.contains(b));
        // a and c handles still resolve to their own data (c moved rows
        // internally, but its handle and value are unchanged).
        assert_eq!(t.get_f64(a, "x").unwrap(), 10.0);
        assert_eq!(t.get_f64(c, "x").unwrap(), 30.0);
        // b's row was filled by c; the column stays compact and aligned.
        let (xs, _) = t.column_f64("x").unwrap();
        assert_eq!(xs.len(), 2);
        let mut got: Vec<f64> = t.handles().map(|h| t.get_f64(h, "x").unwrap()).collect();
        got.sort_by(|p, q| p.partial_cmp(q).unwrap());
        assert_eq!(got, vec![10.0, 30.0]);
    }

    #[test]
    fn despawn_last_and_stale_handle() {
        let mut t = T::new();
        let a = t.spawn();
        let b = t.spawn();
        assert!(t.despawn(b)); // remove the last row
        assert_eq!(t.len(), 1);
        assert!(!t.despawn(b)); // already gone → false, no panic
        assert!(t.get_f64(b, "x").is_err()); // stale handle errors
        assert!(t.contains(a));
    }

    #[test]
    fn value_and_row_cells_read_dynamically() {
        let mut t = T::new();
        let a = t.spawn();
        t.set_f64(a, "x", 1.5).unwrap();
        t.set_i32(a, "n", 7).unwrap();
        t.set_str(a, "el", "C").unwrap();
        assert_eq!(t.value(a, "x"), Some(Cell::F64(1.5)));
        assert_eq!(t.value(a, "n"), Some(Cell::I32(7)));
        assert_eq!(t.value(a, "el"), Some(Cell::Str("C")));
        assert_eq!(t.value(a, "absent"), None);
        // row_cells yields exactly the present components.
        let mut cells: Vec<(String, Cell<'_>)> =
            t.row_cells(a).map(|(k, v)| (k.to_owned(), v)).collect();
        cells.sort_by(|p, q| p.0.cmp(&q.0));
        assert_eq!(
            cells,
            vec![
                ("el".to_owned(), Cell::Str("C")),
                ("n".to_owned(), Cell::I32(7)),
                ("x".to_owned(), Cell::F64(1.5)),
            ]
        );
        // A second, sparser entity only surfaces its own present cells.
        let b = t.spawn();
        t.set_f64(b, "x", 9.0).unwrap();
        assert_eq!(t.value(b, "el"), None);
        assert_eq!(t.row_cells(b).count(), 1);
    }

    #[test]
    fn clear_unsets_without_removing_column() {
        let mut t = T::new();
        let a = t.spawn();
        t.set_f64(a, "charge", 2.0).unwrap();
        assert!(t.has(a, "charge"));
        t.clear(a, "charge").unwrap();
        assert!(!t.has(a, "charge"));
        assert!(t.get_f64(a, "charge").is_err());
        // Column still exists (other entities may use it).
        assert!(t.column_f64("charge").is_ok());
    }
}
