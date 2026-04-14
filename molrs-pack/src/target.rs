//! Target builder for molecular packing.

use std::sync::Arc;

use crate::frame::frame_to_coords_and_elements;
use crate::relaxer::Relaxer;
use crate::restraint::Restraint;
use molrs::types::F;

/// Centering behavior for structure coordinates.
///
/// Packmol semantics:
/// - `Auto`: free molecules are centered; fixed molecules are not centered.
/// - `Center`: force centering.
/// - `None`: keep input coordinates unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CenteringMode {
    #[default]
    Auto,
    Center,
    None,
}

/// Fixed molecule placement (Euler angles in "human" convention + translation).
#[derive(Debug, Clone)]
pub struct FixedPlacement {
    /// Translation vector [x, y, z].
    pub position: [F; 3],
    /// Euler angles [beta, gama, teta] in the `eulerfixed` convention (x/y/z rotations).
    pub euler: [F; 3],
}

/// Describes one type of molecule to be packed.
#[derive(Debug, Clone)]
pub struct Target {
    /// Input coordinates as provided by the source structure.
    pub input_coords: Vec<[F; 3]>,
    /// Flat list of atom positions — the centered reference coordinates.
    /// Shape: natoms × 3, stored as Vec<[F; 3]>.
    pub ref_coords: Vec<[F; 3]>,
    /// Van der Waals radii per atom.
    pub radii: Vec<F>,
    /// Element symbols per atom (e.g. `"C"`, `"O"`). Defaults to `"X"` if unknown.
    pub elements: Vec<String>,
    /// Number of copies to pack.
    pub count: usize,
    /// Optional name for logging.
    pub name: Option<String>,
    /// Restraints applied to every atom of every molecule copy.
    pub molecule_restraints: Vec<Arc<dyn Restraint>>,
    /// Per-atom-subset restraints: `(atom_indices_0_based, restraint)`.
    /// Each entry holds the 0-based atom indices (converted from Packmol's
    /// 1-based convention at registration time) and the restraint applied to them.
    pub atom_restraints: Vec<(Vec<usize>, Arc<dyn Restraint>)>,
    /// Optional structure-level limit for movebad (`maxmove` in Packmol).
    pub maxmove: Option<usize>,
    /// Centering policy.
    pub centering: CenteringMode,
    /// Rotation constraints in Euler variable order:
    /// [beta(y), gama(z), teta(x)] => (center_rad, half_width_rad).
    pub constrain_rotation: [Option<(F, F)>; 3],
    /// If Some, this molecule is fixed (one copy, placed at the given location).
    pub fixed_at: Option<FixedPlacement>,
    /// Per-target in-loop relaxers (e.g. torsion MC). Called in order each iteration.
    pub relaxers: Vec<Box<dyn Relaxer>>,
}

impl Target {
    /// Create a new target from a `molrs::Frame` (read from PDB/XYZ) and a copy count.
    ///
    /// Positions are extracted from the `"atoms"` block (`"x"`, `"y"`, `"z"` columns)
    /// and automatically centered at the geometric center.
    /// VdW radii and element symbols are looked up from the `"element"` column.
    pub fn new(frame: molrs::Frame, count: usize) -> Self {
        let (positions, radii, elements) = frame_to_coords_and_elements(&frame);
        Self::from_parts(&positions, &radii, elements, count)
    }

    /// Create a new target directly from coordinate arrays.
    ///
    /// Useful for testing or when coordinates are already available.
    /// Stores both raw input coordinates and a geometrically centered reference copy.
    /// Effective usage follows [`CenteringMode::Auto`] unless overridden.
    pub fn from_coords(frame_positions: &[[F; 3]], radii: &[F], count: usize) -> Self {
        let n = frame_positions.len();
        Self::from_parts(frame_positions, radii, vec!["X".to_string(); n], count)
    }

    fn from_parts(
        frame_positions: &[[F; 3]],
        radii: &[F],
        elements: Vec<String>,
        count: usize,
    ) -> Self {
        assert_eq!(
            frame_positions.len(),
            radii.len(),
            "positions and radii must have the same length"
        );
        let input_coords = frame_positions.to_vec();
        let ref_coords = centered_coords(frame_positions);
        Self {
            input_coords,
            ref_coords,
            radii: radii.to_vec(),
            elements,
            count,
            name: None,
            molecule_restraints: Vec::new(),
            atom_restraints: Vec::new(),
            maxmove: None,
            centering: CenteringMode::Auto,
            constrain_rotation: [None, None, None],
            fixed_at: None,
            relaxers: Vec::new(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Attach a restraint applied to every atom of every molecule copy.
    pub fn with_restraint(mut self, r: impl Restraint + 'static) -> Self {
        self.molecule_restraints.push(Arc::new(r));
        self
    }

    /// Attach a restraint for selected atoms of every molecule copy.
    ///
    /// # Atom indexing
    ///
    /// Indices follow **Packmol's 1-based convention**: atom `1` is the first
    /// atom in the PDB/XYZ file. They are converted to 0-based internally.
    /// For example, `&[1, 2, 3]` selects the first three atoms.
    pub fn with_restraint_for_atoms(
        mut self,
        indices: &[usize],
        r: impl Restraint + 'static,
    ) -> Self {
        let zero_indexed: Vec<usize> = indices.iter().map(|&i| i.saturating_sub(1)).collect();
        self.atom_restraints.push((zero_indexed, Arc::new(r)));
        self
    }

    /// Attach an in-loop relaxer for this target.
    ///
    /// Multiple relaxers can be attached (called in order).
    /// Relaxers require `count == 1` because all copies share reference coords.
    ///
    /// Mirrors [`with_restraint`](Self::with_restraint) — a per-target builder method.
    pub fn with_relaxer(mut self, relaxer: impl Relaxer + 'static) -> Self {
        assert!(
            self.count <= 1,
            "relaxers require count == 1 (all copies share ref coords)"
        );
        self.relaxers.push(Box::new(relaxer));
        self
    }

    /// Set structure-level `maxmove` for movebad heuristic.
    pub fn with_maxmove(mut self, maxmove: usize) -> Self {
        self.maxmove = Some(maxmove);
        self
    }

    /// Equivalent to Packmol `center` keyword for this structure.
    pub fn with_center(mut self) -> Self {
        self.centering = CenteringMode::Center;
        self
    }

    /// Keep input coordinates unchanged (disable automatic centering).
    pub fn without_centering(mut self) -> Self {
        self.centering = CenteringMode::None;
        self
    }

    /// Equivalent to Packmol `constrain_rotation x center delta` (degrees).
    pub fn constrain_rotation_x(mut self, center_deg: F, half_width_deg: F) -> Self {
        self.constrain_rotation[2] = Some((deg_to_rad(center_deg), deg_to_rad(half_width_deg)));
        self
    }

    /// Equivalent to Packmol `constrain_rotation y center delta` (degrees).
    pub fn constrain_rotation_y(mut self, center_deg: F, half_width_deg: F) -> Self {
        self.constrain_rotation[0] = Some((deg_to_rad(center_deg), deg_to_rad(half_width_deg)));
        self
    }

    /// Equivalent to Packmol `constrain_rotation z center delta` (degrees).
    pub fn constrain_rotation_z(mut self, center_deg: F, half_width_deg: F) -> Self {
        self.constrain_rotation[1] = Some((deg_to_rad(center_deg), deg_to_rad(half_width_deg)));
        self
    }

    /// Fix this molecule at a specific position with zero rotation.
    pub fn fixed_at(mut self, position: [F; 3]) -> Self {
        self.fixed_at = Some(FixedPlacement {
            position,
            euler: [0.0, 0.0, 0.0],
        });
        self.count = 1;
        self
    }

    /// Fix this molecule at a specific position and Euler orientation.
    pub fn fixed_at_with_euler(mut self, position: [F; 3], euler: [F; 3]) -> Self {
        self.fixed_at = Some(FixedPlacement { position, euler });
        self.count = 1;
        self
    }

    pub fn natoms(&self) -> usize {
        self.ref_coords.len()
    }
}

fn centered_coords(coords: &[[F; 3]]) -> Vec<[F; 3]> {
    let (cx, cy, cz) = geometric_center(coords);
    coords
        .iter()
        .map(|p| [p[0] - cx, p[1] - cy, p[2] - cz])
        .collect()
}

fn geometric_center(coords: &[[F; 3]]) -> (F, F, F) {
    if coords.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let n = coords.len() as F;
    let cx = coords.iter().map(|p| p[0]).sum::<F>() / n;
    let cy = coords.iter().map(|p| p[1]).sum::<F>() / n;
    let cz = coords.iter().map(|p| p[2]).sum::<F>() / n;
    (cx, cy, cz)
}

fn deg_to_rad(v: F) -> F {
    v * (std::f64::consts::PI as F) / 180.0
}
