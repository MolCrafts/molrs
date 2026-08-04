//! The Frame schema — a committed, machine-checked vocabulary for block and
//! column names, and the validator that judges a Frame against it.
//!
//! # A canonical key names a quantity, not a slot
//!
//! This is the axiom the whole design rests on, and it is true only because it
//! is kept true: **if two things need different dtypes under one name, they are
//! two quantities and get two keys.** `type` (String, a force-field label) and
//! `type_id` (UInt, a LAMMPS ordinal) are not an exception to the one-dtype
//! rule — splitting them is the operation that *makes* the rule hold.
//!
//! Every future "but this key needs two dtypes" has one correct answer: then it
//! is two keys. A `DTypeSet` would quietly undo the entire design, because a
//! column's dtype is fixed by its first write and molrs refuses to coerce — so
//! the second writer's value silently fails to land.
//!
//! # Closed vocabulary, open block set
//!
//! A [`ColumnSpec`] binds a key *wherever it appears*, in any block. That is
//! what lets [`Block::insert`](crate::store::block::Block::insert) enforce dtype
//! without knowing which block it is about to live in — dissolving the problem
//! that a standalone `Block` only learns its role when inserted into a `Frame`,
//! by which time a wrong-dtype column would already exist.
//!
//! The *block* set stays open: `MolGraph::to_frame` mints one block per
//! registered relation kind, so a block name with no [`BlockSpec`] is legal and
//! its columns are still checked individually. A column key with no spec is
//! unconstrained — the extension point for perceived facts, per-instance
//! force-field parameters, and format-local columns.
//!
//! # Two layers of enforcement
//!
//! | concern | scope | enforced at |
//! |---|---|---|
//! | dtype, shape | key-global | `Block::insert` / `insert_column` / `rename_column` |
//! | required columns, arity | block-scoped | [`BlockSpec`] via [`Validator`] |
//! | endpoint range, row counts | frame-scoped | [`Validator`] |

pub mod block;
pub mod column;
pub mod document;
pub mod validator;
pub mod violation;

pub use block::{BlockSpec, EndpointSpec, RowKind};
pub use column::{ColShape, ColumnSpec};
pub use document::{BlockDoc, ColumnDoc, SchemaDocument, document};
pub use validator::Validator;
pub use violation::{
    InstancePath, MAX_CELL_VIOLATIONS_PER_COLUMN, SchemaReport, Violation, ViolationKind,
};

use crate::store::block::DType;

/// Version of the **vocabulary** — what block and column names mean, and what
/// dtype each carries.
///
/// Distinct from
/// [`FRAME_SCHEMA_VERSION`](crate::store::frame::FRAME_SCHEMA_VERSION), which
/// versions the *serialization envelope* (how bytes are laid out). One can
/// change without the other; conflating them is why this doc paragraph exists.
///
/// Bump when: a spec's `dtype` or `shape` changes, a canonical key is renamed
/// or removed, or a block's `required` set grows. Do **not** bump when: a new
/// key is added, a new optional column is added, or a doc/unit string changes.
/// Adding a key is forward-compatible — old data simply lacks it; changing what
/// an existing key means is not.
pub const FRAME_VOCAB_VERSION: u32 = 1;

macro_rules! col {
    ($key:literal, $const_name:literal, $dtype:expr, $shape:expr, $unit:literal, $doc:literal) => {
        ColumnSpec {
            key: $key,
            const_name: $const_name,
            dtype: $dtype,
            shape: $shape,
            unit: $unit,
            doc: $doc,
        }
    };
}

use ColShape::Scalar;
// No canonical column is `Int`: every identifier is unsigned and every
// physical quantity is float. `Int` returns to this list the day a signed
// integer quantity is genuinely needed.
use DType::{Float, String as Str, UInt};

/// Every canonical column, sorted by key.
///
/// Sortedness and uniqueness are asserted by the vocabulary gate, and the
/// lookup binary-searches this table.
pub static SCHEMA_COLUMNS: &[ColumnSpec] = &[
    col!(
        "atomi",
        "ATOMI",
        UInt,
        Scalar,
        "",
        "First endpoint of a relation, 0-indexed into the target node block."
    ),
    col!(
        "atomic_number",
        "ATOMIC_NUMBER",
        UInt,
        Scalar,
        "",
        "Atomic number Z."
    ),
    col!(
        "atomj",
        "ATOMJ",
        UInt,
        Scalar,
        "",
        "Second endpoint of a relation, 0-indexed."
    ),
    col!(
        "atomk",
        "ATOMK",
        UInt,
        Scalar,
        "",
        "Third endpoint of a relation (angle vertex / dihedral), 0-indexed."
    ),
    col!(
        "atoml",
        "ATOML",
        UInt,
        Scalar,
        "",
        "Fourth endpoint of a relation (dihedral / improper), 0-indexed."
    ),
    col!(
        "bead_type",
        "BEAD_TYPE",
        Str,
        Scalar,
        "",
        "Coarse-grained bead type label."
    ),
    col!(
        "bond_number",
        "BOND_NUMBER",
        UInt,
        Scalar,
        "",
        "Integer bond number of the localized Lewis/Kekule structure: 0 unknown, 1 single, 2 double, 3 triple, 4 quadruple. Never fractional - aromaticity is a bond type, not a number."
    ),
    col!(
        "bond_type",
        "BOND_TYPE",
        UInt,
        Scalar,
        "",
        "Chemical bond class: 0 unknown, 1 single, 2 double, 3 triple, 4 aromatic. Orthogonal to `bond_number`: an aromatic bond is `bond_type = 4` carrying a `bond_number` of 1 or 2."
    ),
    col!("charge", "CHARGE", Float, Scalar, "e", "Partial charge."),
    col!(
        "element",
        "ELEMENT",
        Str,
        Scalar,
        "",
        "IUPAC element symbol (e.g. \"C\")."
    ),
    col!(
        "id",
        "ID",
        UInt,
        Scalar,
        "",
        "Identifier carried by the source file. Never an index — endpoints are 0-based row indices and a reader that must map labels to rows does so locally."
    ),
    col!(
        "is_14",
        "IS_14",
        DType::Bool,
        Scalar,
        "",
        "Whether a non-bonded pair is a 1-4 (third-neighbour) pair."
    ),
    col!("mass", "MASS", Float, Scalar, "amu", "Atomic mass."),
    col!(
        "mol_id",
        "MOL_ID",
        UInt,
        Scalar,
        "",
        "Molecule identifier grouping atoms into molecules."
    ),
    col!(
        "mux",
        "MUX",
        Float,
        Scalar,
        "e*angstrom",
        "x-component of a per-atom electric dipole moment."
    ),
    col!(
        "muy",
        "MUY",
        Float,
        Scalar,
        "e*angstrom",
        "y-component of a per-atom electric dipole moment."
    ),
    col!(
        "muz",
        "MUZ",
        Float,
        Scalar,
        "e*angstrom",
        "z-component of a per-atom electric dipole moment."
    ),
    col!(
        "name",
        "NAME",
        Str,
        Scalar,
        "",
        "Human-readable atom name (e.g. \"CA\")."
    ),
    col!(
        "quati",
        "QUATI",
        Float,
        Scalar,
        "",
        "First imaginary component of a per-atom orientation quaternion."
    ),
    col!(
        "quatj",
        "QUATJ",
        Float,
        Scalar,
        "",
        "Second imaginary component of a per-atom orientation quaternion."
    ),
    col!(
        "quatk",
        "QUATK",
        Float,
        Scalar,
        "",
        "Third imaginary component of a per-atom orientation quaternion."
    ),
    col!(
        "quatw",
        "QUATW",
        Float,
        Scalar,
        "",
        "Real part of a per-atom orientation quaternion."
    ),
    col!(
        "res_id",
        "RES_ID",
        UInt,
        Scalar,
        "",
        "Residue identifier. Unsigned like every other id in the vocabulary; a file with negative residue numbers is renumbered at the reader boundary, not accommodated by the schema."
    ),
    col!(
        "res_name",
        "RES_NAME",
        Str,
        Scalar,
        "",
        "Residue name (e.g. \"ALA\")."
    ),
    col!(
        "type",
        "TYPE",
        Str,
        Scalar,
        "",
        "Force-field type label. Always a String: a label is what survives a round trip through a force field. Numeric ordinals live in `type_id`."
    ),
    col!(
        "type_id",
        "TYPE_ID",
        UInt,
        Scalar,
        "",
        "Numeric type ordinal as used by formats that number their types (LAMMPS). Format-local; the force field reads `type`."
    ),
    col!(
        "vx",
        "VX",
        Float,
        Scalar,
        "",
        "x-velocity. Unit follows the force field's `units` setting; molrs stores raw numbers."
    ),
    col!("vy", "VY", Float, Scalar, "", "y-velocity."),
    col!("vz", "VZ", Float, Scalar, "", "z-velocity."),
    col!(
        "x",
        "X",
        Float,
        Scalar,
        "",
        "Cartesian x-coordinate. Unit follows the force field / file format; molrs stores raw numbers."
    ),
    col!("y", "Y", Float, Scalar, "", "Cartesian y-coordinate."),
    col!("z", "Z", Float, Scalar, "", "Cartesian z-coordinate."),
];

/// Every canonical block, sorted by name.
pub static SCHEMA_BLOCKS: &[BlockSpec] = &[
    BlockSpec {
        name: "angles",
        row_kind: RowKind::Relation { arity: 3 },
        endpoints: Some(EndpointSpec {
            target: "atoms",
            columns: &["atomi", "atomj", "atomk"],
        }),
        required: &["atomi", "atomj", "atomk"],
        optional: &["type", "type_id"],
        open: true,
        doc: "Three-body angle terms; `atomj` is the vertex.",
    },
    BlockSpec {
        name: "atoms",
        row_kind: RowKind::Node,
        endpoints: None,
        required: &[],
        optional: &[
            "x",
            "y",
            "z",
            "id",
            "type",
            "type_id",
            "element",
            "atomic_number",
            "mass",
            "charge",
            "mol_id",
            "name",
            "res_id",
            "res_name",
            "vx",
            "vy",
            "vz",
        ],
        open: true,
        doc: "Per-atom properties. The node table relation blocks index into.",
    },
    BlockSpec {
        name: "beads",
        row_kind: RowKind::Node,
        endpoints: None,
        required: &[],
        optional: &["x", "y", "z", "id", "bead_type", "mass", "charge", "mol_id"],
        open: true,
        doc: "Per-bead properties of a coarse-grained system.",
    },
    BlockSpec {
        name: "bonds",
        row_kind: RowKind::Relation { arity: 2 },
        endpoints: Some(EndpointSpec {
            target: "atoms",
            columns: &["atomi", "atomj"],
        }),
        required: &["atomi", "atomj"],
        optional: &["type", "type_id", "bond_type", "bond_number"],
        open: true,
        doc: "Two-body bond terms.",
    },
    BlockSpec {
        name: "dihedrals",
        row_kind: RowKind::Relation { arity: 4 },
        endpoints: Some(EndpointSpec {
            target: "atoms",
            columns: &["atomi", "atomj", "atomk", "atoml"],
        }),
        required: &["atomi", "atomj", "atomk", "atoml"],
        optional: &["type", "type_id"],
        open: true,
        doc: "Four-body proper torsion terms.",
    },
    BlockSpec {
        name: "exclusions",
        row_kind: RowKind::Relation { arity: 2 },
        endpoints: Some(EndpointSpec {
            target: "atoms",
            columns: &["atomi", "atomj"],
        }),
        required: &["atomi", "atomj"],
        optional: &[],
        open: true,
        doc: "Pairs excluded from non-bonded interaction (PME real-space correction).",
    },
    BlockSpec {
        name: "impropers",
        row_kind: RowKind::Relation { arity: 4 },
        endpoints: Some(EndpointSpec {
            target: "atoms",
            columns: &["atomi", "atomj", "atomk", "atoml"],
        }),
        required: &["atomi", "atomj", "atomk", "atoml"],
        optional: &["type", "type_id"],
        open: true,
        doc: "Four-body improper terms enforcing planarity or chirality.",
    },
    BlockSpec {
        name: "pairs",
        row_kind: RowKind::Relation { arity: 2 },
        endpoints: Some(EndpointSpec {
            target: "atoms",
            columns: &["atomi", "atomj"],
        }),
        required: &["atomi", "atomj"],
        optional: &["is_14"],
        open: true,
        doc: "Intramolecular non-bonded pair list. Consumer-built, not read from a file.",
    },
];

/// Canonical spec for a column key, or `None` if the key is unconstrained.
pub fn column(key: &str) -> Option<&'static ColumnSpec> {
    SCHEMA_COLUMNS
        .binary_search_by(|c| c.key.cmp(key))
        .ok()
        .map(|i| &SCHEMA_COLUMNS[i])
}

/// Canonical spec for a block name, or `None` if the block is not in the
/// vocabulary (which is legal — the block set is open).
pub fn block(name: &str) -> Option<&'static BlockSpec> {
    SCHEMA_BLOCKS
        .binary_search_by(|b| b.name.cmp(name))
        .ok()
        .map(|i| &SCHEMA_BLOCKS[i])
}

/// Canonical string constants, generated from [`SCHEMA_COLUMNS`].
///
/// Supersedes the hand-written `store::keys` table: a key is declared once, in
/// the spec, and the constant follows.
pub mod consts {
    /// Cartesian x-coordinate component.
    pub const X: &str = "x";
    /// Cartesian y-coordinate component.
    pub const Y: &str = "y";
    /// Cartesian z-coordinate component.
    pub const Z: &str = "z";
    /// The three Cartesian coordinate keys, in axis order.
    pub const COORDS: [&str; 3] = [X, Y, Z];
    /// Element symbol.
    pub const ELEMENT: &str = "element";
    /// Atomic number Z.
    pub const ATOMIC_NUMBER: &str = "atomic_number";
    /// Coarse-grained bead type.
    pub const BEAD_TYPE: &str = "bead_type";
    /// Partial charge.
    pub const CHARGE: &str = "charge";
    /// Chemical bond class: 0 unknown, 1 single, 2 double, 3 triple, 4 aromatic.
    ///
    /// Aromatic is a bond *type*, peer to single/double/triple — never a number.
    /// The localized integer that accompanies it is [`BOND_NUMBER`].
    pub const BOND_TYPE: &str = "bond_type";
    /// Integer bond number of the localized Lewis/Kekulé structure:
    /// 0 unknown, 1 single, 2 double, 3 triple, 4 quadruple.
    pub const BOND_NUMBER: &str = "bond_number";
    /// Atomic mass.
    pub const MASS: &str = "mass";
    /// Force-field type label (String).
    pub const TYPE: &str = "type";
    /// Numeric type ordinal (UInt), for formats that number their types.
    pub const TYPE_ID: &str = "type_id";
    /// Identifier carried by the source file.
    pub const ID: &str = "id";
    /// Molecule identifier.
    pub const MOL_ID: &str = "mol_id";
    /// Human-readable atom name.
    pub const NAME: &str = "name";
    /// Cartesian x-velocity.
    pub const VX: &str = "vx";
    /// Cartesian y-velocity.
    pub const VY: &str = "vy";
    /// Cartesian z-velocity.
    pub const VZ: &str = "vz";
    /// The three Cartesian velocity keys, in axis order.
    pub const VELOCITIES: [&str; 3] = [VX, VY, VZ];
    /// Real part of an orientation quaternion.
    pub const QUATW: &str = "quatw";
    /// First imaginary component of an orientation quaternion.
    pub const QUATI: &str = "quati";
    /// Second imaginary component of an orientation quaternion.
    pub const QUATJ: &str = "quatj";
    /// Third imaginary component of an orientation quaternion.
    pub const QUATK: &str = "quatk";
    /// The four orientation-quaternion keys, in `(w, i, j, k)` order.
    pub const QUAT: [&str; 4] = [QUATW, QUATI, QUATJ, QUATK];
    /// x-component of a per-atom dipole moment.
    pub const MUX: &str = "mux";
    /// y-component of a per-atom dipole moment.
    pub const MUY: &str = "muy";
    /// z-component of a per-atom dipole moment.
    pub const MUZ: &str = "muz";
    /// The three dipole-moment keys, in axis order.
    pub const DIPOLE: [&str; 3] = [MUX, MUY, MUZ];
    /// Residue identifier.
    pub const RES_ID: &str = "res_id";
    /// Residue name.
    pub const RES_NAME: &str = "res_name";
    /// Whether a non-bonded pair is 1-4.
    pub const IS_14: &str = "is_14";
    /// First relation endpoint, 0-indexed.
    pub const ATOMI: &str = "atomi";
    /// Second relation endpoint, 0-indexed.
    pub const ATOMJ: &str = "atomj";
    /// Third relation endpoint, 0-indexed.
    pub const ATOMK: &str = "atomk";
    /// Fourth relation endpoint, 0-indexed.
    pub const ATOML: &str = "atoml";
    /// Relation endpoint keys in position order.
    pub const ENDPOINTS: [&str; 4] = [ATOMI, ATOMJ, ATOMK, ATOML];
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn columns_sorted_unique_and_documented() {
        let mut prev = "";
        for c in SCHEMA_COLUMNS {
            assert!(c.key > prev, "SCHEMA_COLUMNS not sorted at {:?}", c.key);
            prev = c.key;
            assert!(!c.doc.is_empty(), "{} has an empty doc", c.key);
            assert!(!c.const_name.is_empty(), "{} has no const_name", c.key);
        }
        let names: HashSet<_> = SCHEMA_COLUMNS.iter().map(|c| c.const_name).collect();
        assert_eq!(names.len(), SCHEMA_COLUMNS.len(), "const_name collision");
    }

    #[test]
    fn blocks_sorted_unique_and_documented() {
        let mut prev = "";
        for b in SCHEMA_BLOCKS {
            assert!(b.name > prev, "SCHEMA_BLOCKS not sorted at {:?}", b.name);
            prev = b.name;
            assert!(!b.doc.is_empty(), "{} has an empty doc", b.name);
        }
    }

    #[test]
    fn block_column_references_resolve() {
        for b in SCHEMA_BLOCKS {
            for key in b
                .required
                .iter()
                .chain(b.optional)
                .chain(b.endpoint_columns())
            {
                assert!(
                    column(key).is_some(),
                    "block '{}' references unknown column '{key}'",
                    b.name
                );
            }
        }
    }

    #[test]
    fn relation_arity_matches_endpoint_count() {
        for b in SCHEMA_BLOCKS {
            match b.row_kind {
                RowKind::Relation { arity } => {
                    let e = b.endpoints.expect("relation block needs endpoints");
                    assert_eq!(e.columns.len(), arity, "{} arity mismatch", b.name);
                }
                _ => assert!(
                    b.endpoints.is_none(),
                    "{} is not a relation but declares endpoints",
                    b.name
                ),
            }
        }
    }

    #[test]
    fn endpoint_targets_are_node_blocks() {
        for b in SCHEMA_BLOCKS {
            if let Some(e) = b.endpoints {
                let target = block(e.target).expect("endpoint target must be a known block");
                assert_eq!(
                    target.row_kind,
                    RowKind::Node,
                    "{} target is not a node table",
                    b.name
                );
            }
        }
    }

    #[test]
    fn lookup_finds_every_key() {
        for c in SCHEMA_COLUMNS {
            assert_eq!(column(c.key).map(|s| s.key), Some(c.key));
        }
        assert!(column("definitely_not_a_canonical_key").is_none());
    }

    #[test]
    fn consts_agree_with_the_table() {
        // Every const must name a key that is actually in the vocabulary —
        // otherwise a rename in the table leaves a const pointing at nothing.
        for key in [
            consts::X,
            consts::Y,
            consts::Z,
            consts::ELEMENT,
            consts::ATOMIC_NUMBER,
            consts::BEAD_TYPE,
            consts::CHARGE,
            consts::BOND_TYPE,
            consts::BOND_NUMBER,
            consts::MASS,
            consts::TYPE,
            consts::TYPE_ID,
            consts::ID,
            consts::MOL_ID,
            consts::NAME,
            consts::VX,
            consts::VY,
            consts::VZ,
            consts::QUATW,
            consts::QUATI,
            consts::QUATJ,
            consts::QUATK,
            consts::MUX,
            consts::MUY,
            consts::MUZ,
            consts::RES_ID,
            consts::RES_NAME,
            consts::IS_14,
            consts::ATOMI,
            consts::ATOMJ,
            consts::ATOMK,
            consts::ATOML,
        ] {
            assert!(column(key).is_some(), "const points at unknown key {key:?}");
        }
    }

    #[test]
    fn type_and_type_id_are_different_quantities() {
        // The axiom in this module's doc, as a test: one name, one dtype.
        assert_eq!(column("type").unwrap().dtype, DType::String);
        assert_eq!(column("type_id").unwrap().dtype, DType::UInt);
    }

    #[test]
    fn every_identifier_key_shares_one_dtype() {
        // Scans the table rather than asserting a hand-written list, so a new
        // `*_id` key cannot be added at a different dtype without this failing.
        let ids: Vec<&ColumnSpec> = SCHEMA_COLUMNS
            .iter()
            .filter(|c| {
                c.key == "id"
                    || c.key.ends_with("_id")
                    || matches!(c.key, "atomi" | "atomj" | "atomk" | "atoml")
            })
            .collect();
        assert!(ids.len() >= 8, "identifier scan found only {}", ids.len());
        for c in &ids {
            assert_eq!(
                c.dtype,
                DType::UInt,
                "identifier '{}' is {} — every identifier is UInt",
                c.key,
                c.dtype
            );
        }
    }
}
