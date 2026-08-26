//! Small helpers shared by LAMMPS data and dump readers/writers.
//!
//! Column-name aliases follow the LAMMPS `dump custom` / `compute property/atom`
//! attribute list: <https://docs.lammps.org/dump.html>,
//! <https://docs.lammps.org/compute_property_atom.html>.

use molrs::store::block::Block;
use molrs::store::keys;
use molrs::types::{F, I, Idx};
use ndarray::{Array1, ArrayD, IxDyn};
use std::collections::HashMap;

// ============================================================================
// Errors / tokens / parse
// ============================================================================

pub(crate) fn err_mapper<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}

/// Split a line on whitespace. Token count is small (≤ ~20); float parsing
/// dominates cost on large files, not the token `Vec`.
///
/// Inline comments (``# …``) are stripped first — LAMMPS data files from
/// VMD/TopoTools routinely write ``# element RES`` after atom rows and
/// ``# type-name`` after Masses; those must not pollute the column layout.
pub(crate) fn tokenize(line: &str) -> Vec<&str> {
    let code = line.split('#').next().unwrap_or(line);
    code.split_whitespace().collect()
}

pub(crate) fn parse_i(token: &str) -> std::io::Result<I> {
    token.parse::<I>().map_err(err_mapper)
}

pub(crate) fn parse_f(token: &str) -> std::io::Result<F> {
    token.parse::<F>().map_err(err_mapper)
}

// ============================================================================
// Array / Block helpers
// ============================================================================

pub(crate) fn arr1_f(v: Vec<F>, n: usize) -> std::io::Result<ArrayD<F>> {
    Array1::from_vec(v)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(err_mapper)
        .map(|a| a.into_dyn())
}

pub(crate) fn arr1_i(v: Vec<I>, n: usize) -> std::io::Result<ArrayD<I>> {
    Array1::from_vec(v)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(err_mapper)
        .map(|a| a.into_dyn())
}

pub(crate) fn arr1_u(v: Vec<Idx>, n: usize) -> std::io::Result<ArrayD<Idx>> {
    Array1::from_vec(v)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(err_mapper)
        .map(|a| a.into_dyn())
}

pub(crate) fn insert_f(block: &mut Block, key: &str, v: Vec<F>, n: usize) -> std::io::Result<()> {
    block.insert(key, arr1_f(v, n)?).map_err(err_mapper)
}

pub(crate) fn insert_i(block: &mut Block, key: &str, v: Vec<I>, n: usize) -> std::io::Result<()> {
    block.insert(key, arr1_i(v, n)?).map_err(err_mapper)
}

pub(crate) fn insert_u(block: &mut Block, key: &str, v: Vec<Idx>, n: usize) -> std::io::Result<()> {
    block.insert(key, arr1_u(v, n)?).map_err(err_mapper)
}

pub(crate) fn insert_str(
    block: &mut Block,
    key: &str,
    v: Vec<String>,
    n: usize,
) -> std::io::Result<()> {
    let arr = ArrayD::from_shape_vec(IxDyn(&[n]), v).map_err(err_mapper)?;
    block.insert(key, arr).map_err(err_mapper)
}

// ============================================================================
// Type references (numeric or label)
// ============================================================================

/// Atom / bond / angle / … type as written in a data file.
#[derive(Debug, Clone)]
pub(crate) enum TypeRef {
    Id(I),
    Label(String),
}

impl TypeRef {
    pub(crate) fn parse(token: &str) -> Self {
        match token.parse::<I>() {
            Ok(id) => TypeRef::Id(id),
            Err(_) => TypeRef::Label(token.to_string()),
        }
    }

    pub(crate) fn resolve(&self, label_to_id: &HashMap<String, I>) -> I {
        match self {
            TypeRef::Id(id) => *id,
            TypeRef::Label(label) => label_to_id.get(label).copied().unwrap_or(1),
        }
    }
}

pub(crate) fn invert_type_labels(id_to_label: &HashMap<String, String>) -> HashMap<String, I> {
    let mut out = HashMap::with_capacity(id_to_label.len());
    for (id_str, label) in id_to_label {
        if let Ok(id) = id_str.parse::<I>() {
            out.insert(label.clone(), id);
        }
    }
    out
}

pub(crate) fn labels_to_meta(id_to_label: &HashMap<String, String>) -> Option<String> {
    if id_to_label.is_empty() {
        return None;
    }
    Some(
        id_to_label
            .iter()
            .map(|(id, label)| format!("{id}:{label}"))
            .collect::<Vec<_>>()
            .join(","),
    )
}

// ============================================================================
// Optional column builder
// ============================================================================

#[derive(Debug, Clone)]
pub(crate) struct OptCol<T> {
    pub data: Vec<T>,
    pub present: bool,
}

impl<T: Copy + Default> OptCol<T> {
    pub(crate) fn with_capacity(n: usize) -> Self {
        Self {
            data: Vec::with_capacity(n),
            present: false,
        }
    }

    pub(crate) fn push(&mut self, v: T) {
        self.data.push(v);
        self.present = true;
    }
}

// ============================================================================
// Dump / property/atom column aliases
// ============================================================================

/// LAMMPS-native attribute → canonical Frame column.
///
/// Only entries that **rename**. Attributes already identical to our keys
/// (`x`, `vx`, `mux`, `diameter`, `quatw`, `mass`, …) are left unchanged.
///
/// Sources:
/// - dump custom attributes: https://docs.lammps.org/dump.html
/// - compute property/atom: https://docs.lammps.org/compute_property_atom.html
const DUMP_COLUMN_ALIASES: &[(&str, &str)] = &[
    // Core renames used across the stack (molpy / keys)
    ("q", keys::CHARGE),
    ("mol", keys::MOL_ID),
    // A dump's `type` column holds LAMMPS' numeric type ordinal, not the
    // force-field label — the vocabulary keeps those apart as `type_id` and
    // `type`, so the rename happens here at the format boundary.
    ("type", keys::TYPE_ID),
    // Legacy / alternate spellings we normalise on read
    ("molecule", keys::MOL_ID),
    ("molecule_id", keys::MOL_ID),
    // Forces (already short names; kept as-is) — listed only if renamed
    // EFF package: spin was renamed to espin in LAMMPS (15Sep2022)
    ("spin", "espin"),
    // SPH package energy attribute `e` is ambiguous; leave as "e"
];

/// Integer-typed dump attributes (writer format + promote-on-demand hints).
///
/// From dump custom + property/atom integer attributes.
pub(crate) fn is_integer_dump_column(name: &str) -> bool {
    matches!(
        name,
        // ids / types / images / ownership
        "id" | "type" | "typelabel" | "mol" | keys::MOL_ID | "proc" | "procp1"
            | "ix" | "iy" | "iz"
            // flags / discrete
            | "bodyflag" | "espin" | "spin" | "status" | "shape_flag"
            | "template_index" | "template_atom" | "nbonds"
            // dump local index
            | "index"
    )
}

/// String-typed dump attributes.
pub(crate) fn is_string_dump_column(name: &str) -> bool {
    name == "element"
}

/// Reader exit: rename a LAMMPS-native dump column to its canonical field name.
pub(crate) fn canonical_dump_column(name: &str) -> String {
    DUMP_COLUMN_ALIASES
        .iter()
        .find(|(native, _)| *native == name)
        .map_or_else(|| name.to_string(), |(_, c)| (*c).to_string())
}

/// Writer entry: inverse of [`canonical_dump_column`].
pub(crate) fn native_dump_column(name: &str) -> &str {
    DUMP_COLUMN_ALIASES
        .iter()
        .find(|(_, canonical)| *canonical == name)
        .map_or(name, |(native, _)| *native)
}
