//! Canonical molecular field-name convention — the single source of truth for
//! component (property) keys.
//!
//! ECS components are addressed by name. To keep field names from being
//! hard-coded as scattered string literals, **all** code references the
//! constants defined here rather than writing `"x"` / `"element"` inline. A
//! field is renamed in exactly one place. These names are kept in sync with
//! molpy's `core/fields.py` canonical field set so the two layers agree at the
//! boundary.
//!
//! # Examples
//!
//! ```
//! use molrs::store::keys;
//!
//! assert_eq!(keys::X, "x");
//! assert_eq!(keys::COORDS, [keys::X, keys::Y, keys::Z]);
//! ```

/// Cartesian x-coordinate component.
pub const X: &str = "x";
/// Cartesian y-coordinate component.
pub const Y: &str = "y";
/// Cartesian z-coordinate component.
pub const Z: &str = "z";

/// The three Cartesian coordinate component keys, in axis order.
///
/// Geometry systems (`translate`, `rotate`, …) read coordinates through this
/// constant rather than literal `"x"`/`"y"`/`"z"`.
pub const COORDS: [&str; 3] = [X, Y, Z];

/// Element symbol of an atom (e.g. `"C"`). Written by the all-atom builder.
pub const ELEMENT: &str = "element";
/// Coarse-grained bead type (e.g. `"W"`). Written by the CG builder.
pub const BEAD_TYPE: &str = "bead_type";

/// Partial charge.
pub const CHARGE: &str = "charge";
/// Bond order (e.g. `1.0`, `2.0`). Default seeded by the bond builder.
pub const ORDER: &str = "order";
/// Atomic mass.
pub const MASS: &str = "mass";
/// Force-field / atom type label.
pub const TYPE: &str = "type";
/// Stable per-entity identifier (when assigned by a reader/writer).
pub const ID: &str = "id";
/// Molecule identifier (groups atoms into molecules).
pub const MOL_ID: &str = "mol_id";
/// Element/site symbol when keyed separately from `element` (e.g. crystal sites).
pub const SYMBOL: &str = "symbol";
/// Human-readable / atom name (e.g. `"CA"`). Capped at PDB's 4-char column on write.
pub const NAME: &str = "name";

/// Cartesian x-velocity component.
pub const VX: &str = "vx";
/// Cartesian y-velocity component.
pub const VY: &str = "vy";
/// Cartesian z-velocity component.
pub const VZ: &str = "vz";
/// The three Cartesian velocity component keys, in axis order.
pub const VELOCITIES: [&str; 3] = [VX, VY, VZ];

/// Real part of a per-atom orientation quaternion.
pub const QUATW: &str = "quatw";
/// First imaginary component of a per-atom orientation quaternion.
pub const QUATI: &str = "quati";
/// Second imaginary component of a per-atom orientation quaternion.
pub const QUATJ: &str = "quatj";
/// Third imaginary component of a per-atom orientation quaternion.
pub const QUATK: &str = "quatk";
/// The four orientation-quaternion component keys, in `(w, i, j, k)` order.
///
/// Names follow LAMMPS `compute property/atom quatw quati quatj quatk`, the same
/// provenance as [`VELOCITIES`]. Consumers that need a 2-D orientation angle
/// derive it as `θ = 2·atan2(k, w)` rather than storing a separate column.
pub const QUAT: [&str; 4] = [QUATW, QUATI, QUATJ, QUATK];

/// Cartesian x-component of a per-atom electric dipole moment.
pub const MUX: &str = "mux";
/// Cartesian y-component of a per-atom electric dipole moment.
pub const MUY: &str = "muy";
/// Cartesian z-component of a per-atom electric dipole moment.
pub const MUZ: &str = "muz";
/// The three dipole-moment component keys, in axis order.
///
/// Names follow LAMMPS `atom_style dipole`.
pub const DIPOLE: [&str; 3] = [MUX, MUY, MUZ];

/// Position 3-vector key, when coordinates are stored as one `xyz` column
/// instead of decomposed `x`/`y`/`z` scalars.
pub const XYZ: &str = "xyz";

/// Residue identifier (groups atoms into residues).
pub const RES_ID: &str = "res_id";
/// Residue name (e.g. `"ALA"`).
pub const RES_NAME: &str = "res_name";

/// First endpoint of a relation block (bond/angle/dihedral), 0-indexed into atoms.
pub const ATOMI: &str = "atomi";
/// Second endpoint of a relation block, 0-indexed into atoms.
pub const ATOMJ: &str = "atomj";
/// Third endpoint of a relation block (angle vertex / dihedral), 0-indexed.
pub const ATOMK: &str = "atomk";
/// Fourth endpoint of a relation block (dihedral/improper), 0-indexed.
pub const ATOML: &str = "atoml";

/// Relation endpoint column keys in position order. `rel_col_name(pos)` reads
/// from this array so the bond/angle/dihedral index convention is defined once.
pub const ENDPOINTS: [&str; 4] = [ATOMI, ATOMJ, ATOMK, ATOML];

use crate::store::block::DType;

/// Canonical storage dtype for a known field, if any.
///
/// Component/relation-property writers coerce a value to this dtype so a field's
/// column type is stable regardless of the literal a caller passed — e.g. a bond
/// `order` written as the int `1` is stored as `1.0`, so a later `1.5` write is
/// accepted instead of being rejected by an `i32` column. This registry is the
/// single source of truth for canonical field dtypes; fields not listed here
/// take the dtype of their first write. Keep it in sync with molpy's field set.
pub fn canonical_dtype(key: &str) -> Option<DType> {
    match key {
        // Continuous physical quantities are float-canonical: a value written as
        // an int (e.g. a velocity `0`) is stored as `0.0` so a later fractional
        // write is accepted rather than rejected by an int column.
        X | Y | Z | VX | VY | VZ | CHARGE | ORDER | MASS => Some(DType::Float),
        QUATW | QUATI | QUATJ | QUATK | MUX | MUY | MUZ => Some(DType::Float),
        // Integer ids and string labels (id/mol_id/res_id/type/element/name/…)
        // and the UInt relation endpoints (atomi/atomj/…) intentionally take the
        // dtype of their first write — they are written with a consistent type
        // at their source, so coercion here would be wrong (endpoints are UInt,
        // not Int).
        _ => None,
    }
}
