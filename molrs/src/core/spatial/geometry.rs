//! Geometry *systems* — free functions over a [`MolGraph`]'s node
//! coordinates: the in-place transforms [`translate`], [`scale`] and
//! [`rotate`], and the read-only query [`center`] (the mass-weighted centre of
//! a chosen set of nodes).
//!
//! The vocabulary is that of the entity–component–system (ECS) design molrs
//! uses for its graphs: a node is an *entity*, its properties (`x`, `mass`, …)
//! are *components*, and a *system* is a function that runs over the whole
//! data set (the *world*). The graph is pure data, so spatial transforms and
//! reductions live here as free functions rather than as methods on the data
//! structure. Coordinates are read and written through the canonical
//! [`crate::store::keys`] coordinate convention — no field-name literals — and
//! are in Å by molrs convention.
//!
//! No function here applies a periodic image convention: [`MolGraph`] holds no
//! box. Callers unwrap and wrap with
//! [`SimBox`](crate::spatial::SimBox) themselves.

use std::fmt;

use crate::error::MolRsError;
use crate::op::rigid::{self, apply, axis_angle};
use crate::op::superpose::centroid;
use crate::store::keys;
use crate::system::{MolGraph, NodeId};

/// Translate every node that has coordinates by `delta` (Å; nodes without a
/// full coordinate set are left untouched).
///
/// Operates directly on each dense coordinate column rather than per-node handle
/// lookups, so cost is linear in the number of atoms with one pass per axis.
pub fn translate(mol: &mut MolGraph, delta: [f64; 3]) {
    let table = mol.node_table_mut();
    for (i, key) in keys::COORDS.iter().enumerate() {
        if let Ok((data, valid)) = table.column_f64_mut(key) {
            for (row, val) in data.iter_mut().enumerate() {
                if valid.get(row) {
                    *val += delta[i];
                }
            }
        }
    }
}

/// Scale every node that has coordinates by a per-axis `factor`
/// (dimensionless) about an optional center `about` (Å; defaults to the
/// origin). Pass `[s, s, s]` for a uniform scale. Nodes missing any coordinate
/// are left untouched.
///
/// Operates directly on each dense coordinate column (one pass per axis), so
/// cost is linear in the number of atoms.
pub fn scale(mol: &mut MolGraph, factor: [f64; 3], about: Option<[f64; 3]>) {
    let origin = about.unwrap_or([0.0, 0.0, 0.0]);
    let table = mol.node_table_mut();
    for (i, key) in keys::COORDS.iter().enumerate() {
        if let Ok((data, valid)) = table.column_f64_mut(key) {
            for (row, val) in data.iter_mut().enumerate() {
                if valid.get(row) {
                    *val = (*val - origin[i]) * factor[i] + origin[i];
                }
            }
        }
    }
}

/// Rotate every node that has coordinates around `axis` by `angle` radians
/// (right-handed), optionally about a center point `about` (Å; defaults to the
/// origin). Nodes missing any coordinate are left untouched.
///
/// Only the direction of `axis` matters; its length does not.
///
/// # Errors
///
/// [`MolRsError::Validation`] when `axis` has no direction (every component
/// zero, a component non-finite, or a squared length that overflows) or
/// `angle` is not finite. Nothing is written then.
pub fn rotate(
    mol: &mut MolGraph,
    axis: [f64; 3],
    angle: f64,
    about: Option<[f64; 3]>,
) -> Result<(), MolRsError> {
    // `axis_angle` is the one rule for "has a direction": any finite nonzero
    // axis, whatever its length.
    let rotation = axis_angle(axis, angle).ok_or_else(|| MolRsError::Validation {
        message: format!("rotation axis {axis:?} has no direction"),
    })?;
    if !angle.is_finite() {
        return Err(MolRsError::Validation {
            message: format!("rotation angle {angle} is not finite"),
        });
    }
    let motion = rigid::about(rotation, about.unwrap_or([0.0, 0.0, 0.0]));

    let table = mol.node_table_mut();

    // Read the three dense coordinate columns once, rotate only rows whose
    // x/y/z are all present, and write each column back in a single pass.
    // Rows missing any coordinate keep their original value (identity write).
    let (nx, ny, nz) = {
        // A graph without a coordinate column has nothing to rotate.
        let (x, vx) = match table.column_f64(keys::X) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };
        let (y, vy) = match table.column_f64(keys::Y) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };
        let (z, vz) = match table.column_f64(keys::Z) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };
        let mut nx = x.to_vec();
        let mut ny = y.to_vec();
        let mut nz = z.to_vec();
        for row in 0..x.len() {
            if !(vx.get(row) && vy.get(row) && vz.get(row)) {
                continue;
            }
            [nx[row], ny[row], nz[row]] = apply(&motion, [x[row], y[row], z[row]]);
        }
        (nx, ny, nz)
    };

    // Safe: columns exist (checked above) and lengths are unchanged.
    table
        .column_f64_mut(keys::X)
        .unwrap()
        .0
        .copy_from_slice(&nx);
    table
        .column_f64_mut(keys::Y)
        .unwrap()
        .0
        .copy_from_slice(&ny);
    table
        .column_f64_mut(keys::Z)
        .unwrap()
        .0
        .copy_from_slice(&nz);
    Ok(())
}

/// Why [`center`] has no answer for a node set. Node-carrying variants name
/// the first offending node in slice order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CenterError {
    /// The node slice is empty.
    Empty,
    /// `node` does not resolve in this graph: it is *stale* (the node was
    /// removed). A handle taken from a different graph is detected only when
    /// no live node of this graph occupies the same storage slot.
    NotFound {
        /// The offending node.
        node: NodeId,
    },
    /// `node` lacks one of [`keys::X`], [`keys::Y`], [`keys::Z`] as an `f64`,
    /// or one of them is non-finite.
    BadPosition {
        /// The offending node.
        node: NodeId,
    },
    /// `node` lacks [`keys::MASS`] as an `f64`, or its mass is negative or
    /// non-finite. A mass of exactly 0 is legal.
    BadMass {
        /// The offending node.
        node: NodeId,
    },
    /// The total mass Σm is not positive and finite (e.g. every listed node
    /// has mass 0).
    ZeroMass,
}

impl fmt::Display for CenterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "center of an empty node set"),
            Self::NotFound { node } => write!(f, "node {node:?} is not in this graph"),
            Self::BadPosition { node } => write!(
                f,
                "node {node:?} has a missing or non-finite '{}'/'{}'/'{}'",
                keys::X,
                keys::Y,
                keys::Z
            ),
            Self::BadMass { node } => write!(
                f,
                "node {node:?} has a missing, negative or non-finite '{}'",
                keys::MASS
            ),
            Self::ZeroMass => write!(f, "total mass is not positive and finite"),
        }
    }
}

impl std::error::Error for CenterError {}

/// Mass-weighted centre of the nodes `nodes`:
///
/// **R** = Σᵢ mᵢ **r**ᵢ / Σᵢ mᵢ
///
/// with **r**ᵢ read from [`keys::X`] / [`keys::Y`] / [`keys::Z`] and mᵢ from
/// [`keys::MASS`]. **R** is in the coordinates' length unit (Å by molrs
/// convention); only mass ratios enter it, so the mass unit does not matter.
/// For atoms this is the centre of mass; for coarse-grained beads it is the
/// bead-mass-weighted centre, since a bead carries `x/y/z/mass` exactly as an
/// atom does.
///
/// - **Weighting is per occurrence:** a node listed twice counts twice.
/// - **Mass 0 is legal** and contributes nothing (e.g. a virtual site — a
///   massless interaction point such as the extra charge site of a four-site
///   water model); it still needs a finite position.
/// - **Unlisted nodes are never read.** Cost is O(k) for k listed nodes.
/// - **The result can be non-finite** if Σ m·**r** overflows `f64`.
///
/// # Periodic boundaries
///
/// In a periodic simulation box an atom that leaves through one face re-enters
/// through the opposite one, so a stored ("wrapped") molecule can appear split
/// across the box, and the average of its split coordinates lands in the empty
/// space between the pieces. No periodic image convention is applied here —
/// [`MolGraph`] holds no box. If the
/// listed nodes may be split across the boundary, unwrap them first with
/// [`SimBox::unwrap`](crate::spatial::SimBox::unwrap); after placing
/// everything, wrap the final world once with
/// [`SimBox::wrap`](crate::spatial::SimBox::wrap). When two centres
/// are subtracted (e.g. a coarse-grained group and the all-atom molecule that
/// replaces it), both must be in one length unit.
///
/// # Errors
///
/// Checked in this order, reporting the first offender:
///
/// 1. [`CenterError::Empty`] — `nodes` is empty.
/// 2. Per node, in slice order, position before mass:
///    - [`CenterError::NotFound`] — the handle does not resolve in this graph
///      (see the variant for stale and foreign handles);
///    - [`CenterError::BadPosition`] — a coordinate is missing or non-finite;
///    - [`CenterError::BadMass`] — the mass is missing, negative or non-finite.
/// 3. [`CenterError::ZeroMass`] — Σm is not positive and finite.
///
/// # Examples
///
/// ```
/// use molrs::system::MolGraph;
/// use molrs::spatial::center;
/// use molrs::store::keys;
///
/// let mut mol = MolGraph::new();
/// let mut ids = Vec::new();
/// for (x, mass) in [(0.0, 1.0), (4.0, 3.0)] {
///     let id = mol.add_node();
///     let props = [(keys::X, x), (keys::Y, 0.0), (keys::Z, 0.0), (keys::MASS, mass)];
///     for (key, value) in props {
///         mol.set_node(id, key, value).unwrap();
///     }
///     ids.push(id);
/// }
/// // (1·0 + 3·4) / (1 + 3) = 3
/// assert_eq!(center(&mol, &ids), Ok([3.0, 0.0, 0.0]));
/// ```
pub fn center(mol: &MolGraph, nodes: &[NodeId]) -> Result<[f64; 3], CenterError> {
    if nodes.is_empty() {
        return Err(CenterError::Empty);
    }
    let table = mol.node_table();
    // Each dense column is looked up once; an absent (or non-f64) column
    // reads as "missing" for every node.
    let coords = keys::COORDS.map(|key| table.column_f64(key).ok());
    let mass = table.column_f64(keys::MASS).ok();

    let mut points = Vec::with_capacity(nodes.len());
    let mut masses = Vec::with_capacity(nodes.len());
    for &node in nodes {
        let row = table.row(node).ok_or(CenterError::NotFound { node })?;
        let mut point = [0.0; 3];
        for (value, column) in point.iter_mut().zip(&coords) {
            *value = match column {
                Some((data, valid)) if valid.get(row) && data[row].is_finite() => data[row],
                _ => return Err(CenterError::BadPosition { node }),
            };
        }
        let m = match mass {
            Some((data, valid)) if valid.get(row) && data[row].is_finite() && data[row] >= 0.0 => {
                data[row]
            }
            _ => return Err(CenterError::BadMass { node }),
        };
        points.push(point);
        masses.push(m);
    }
    centroid(&points, &masses).ok_or(CenterError::ZeroMass)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::Atomistic;

    #[test]
    fn rotating_about_an_axis_with_no_direction_is_an_error() {
        for axis in [
            [0.0, 0.0, 0.0],
            [f64::NAN, 0.0, 0.0],
            [f64::INFINITY, 0.0, 0.0],
        ] {
            let mut mol = Atomistic::new();
            let atom = mol.add_atom_xyz("C", 1.0, 2.0, 3.0);
            let result = rotate(mol.as_molgraph_mut(), axis, 1.0, None);
            assert!(result.is_err(), "axis {axis:?} was accepted");
            let node = mol.as_molgraph().get_node(atom).unwrap();
            let position = [keys::X, keys::Y, keys::Z].map(|key| node.get_f64(key).unwrap());
            assert_eq!(position, [1.0, 2.0, 3.0], "axis {axis:?} moved the atom");
        }
    }

    // ---- center ----

    /// A node carrying exactly the `f64` properties in `props`.
    fn node_with(mol: &mut MolGraph, props: &[(&str, f64)]) -> NodeId {
        let id = mol.add_node();
        for &(key, value) in props {
            mol.set_node(id, key, value).unwrap();
        }
        id
    }

    /// A node at `xyz` carrying `mass`.
    fn massive(mol: &mut MolGraph, xyz: [f64; 3], mass: f64) -> NodeId {
        node_with(
            mol,
            &[
                (keys::X, xyz[0]),
                (keys::Y, xyz[1]),
                (keys::Z, xyz[2]),
                (keys::MASS, mass),
            ],
        )
    }

    /// Golden 1: masses 1 and 3 at x = 0 and 4 give (1*0 + 3*4) / 4 = 3.
    #[test]
    fn center_weights_positions_by_mass() {
        let mut mol = MolGraph::new();
        let light = massive(&mut mol, [0.0, 0.0, 0.0], 1.0);
        let heavy = massive(&mut mol, [4.0, 0.0, 0.0], 3.0);
        assert_eq!(center(&mol, &[light, heavy]), Ok([3.0, 0.0, 0.0]));
    }

    /// Golden 2: equal masses at (0,0,0) and (2,4,6) give the midpoint (1,2,3).
    #[test]
    fn center_of_equal_masses_is_the_midpoint() {
        let mut mol = MolGraph::new();
        let a = massive(&mut mol, [0.0, 0.0, 0.0], 2.0);
        let b = massive(&mut mol, [2.0, 4.0, 6.0], 2.0);
        assert_eq!(center(&mol, &[a, b]), Ok([1.0, 2.0, 3.0]));
    }

    /// Golden 1 plus an unlisted mass-100 node at (50,0,0) and a listed
    /// mass-0 node at (-9,0,0): neither moves the centre off (3,0,0).
    #[test]
    fn center_ignores_unlisted_nodes_and_zero_mass_nodes() {
        let mut mol = MolGraph::new();
        let light = massive(&mut mol, [0.0, 0.0, 0.0], 1.0);
        let _unlisted = massive(&mut mol, [50.0, 0.0, 0.0], 100.0);
        let heavy = massive(&mut mol, [4.0, 0.0, 0.0], 3.0);
        let virtual_site = massive(&mut mol, [-9.0, 0.0, 0.0], 0.0);
        assert_eq!(
            center(&mol, &[light, heavy, virtual_site]),
            Ok([3.0, 0.0, 0.0])
        );
    }

    #[test]
    fn center_of_an_empty_slice_is_empty() {
        let mut mol = MolGraph::new();
        let _ = massive(&mut mol, [1.0, 2.0, 3.0], 1.0);
        assert_eq!(center(&mol, &[]), Err(CenterError::Empty));
    }

    #[test]
    fn center_reports_a_removed_handle_as_not_found() {
        let mut mol = MolGraph::new();
        let kept = massive(&mut mol, [0.0, 0.0, 0.0], 1.0);
        let removed = massive(&mut mol, [4.0, 0.0, 0.0], 3.0);
        mol.remove_nodes(&[removed]).unwrap();
        assert_eq!(
            center(&mol, &[kept, removed]),
            Err(CenterError::NotFound { node: removed })
        );
    }

    #[test]
    fn center_reports_a_missing_or_non_finite_coordinate_as_bad_position() {
        let mut mol = MolGraph::new();
        let good = massive(&mut mol, [0.0, 0.0, 0.0], 1.0);

        // The z column exists (the good node has one); this node lacks it.
        let no_z = node_with(
            &mut mol,
            &[(keys::X, 1.0), (keys::Y, 2.0), (keys::MASS, 1.0)],
        );
        assert_eq!(
            center(&mol, &[good, no_z]),
            Err(CenterError::BadPosition { node: no_z })
        );

        let nan_x = massive(&mut mol, [f64::NAN, 0.0, 0.0], 1.0);
        assert_eq!(
            center(&mol, &[good, nan_x]),
            Err(CenterError::BadPosition { node: nan_x })
        );

        // Lacking both position and mass: position is checked first.
        let bare = mol.add_node();
        assert_eq!(
            center(&mol, &[good, bare]),
            Err(CenterError::BadPosition { node: bare })
        );
    }

    #[test]
    fn center_reports_a_missing_negative_or_non_finite_mass_as_bad_mass() {
        let mut mol = MolGraph::new();
        let good = massive(&mut mol, [0.0, 0.0, 0.0], 1.0);

        let massless = node_with(&mut mol, &[(keys::X, 1.0), (keys::Y, 0.0), (keys::Z, 0.0)]);
        assert_eq!(
            center(&mol, &[good, massless]),
            Err(CenterError::BadMass { node: massless })
        );

        let negative = massive(&mut mol, [1.0, 0.0, 0.0], -1.0);
        assert_eq!(
            center(&mol, &[good, negative]),
            Err(CenterError::BadMass { node: negative })
        );

        let infinite = massive(&mut mol, [1.0, 0.0, 0.0], f64::INFINITY);
        assert_eq!(
            center(&mol, &[good, infinite]),
            Err(CenterError::BadMass { node: infinite })
        );

        // [good, bad1, bad2]: the first offender in slice order is reported.
        assert_eq!(
            center(&mol, &[good, negative, massless]),
            Err(CenterError::BadMass { node: negative })
        );
    }

    #[test]
    fn center_of_only_zero_mass_nodes_is_zero_mass() {
        let mut mol = MolGraph::new();
        let a = massive(&mut mol, [0.0, 0.0, 0.0], 0.0);
        let b = massive(&mut mol, [4.0, 0.0, 0.0], 0.0);
        assert_eq!(center(&mol, &[a, b]), Err(CenterError::ZeroMass));
    }
}
