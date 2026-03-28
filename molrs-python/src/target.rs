//! Python wrapper for packing `Target`.
//!
//! A [`PyTarget`] describes one type of molecule to pack: its template
//! coordinates, the number of copies, and optional geometric constraints.
//! Targets are passed to [`PyPacker::pack`](crate::packer::PyPacker::pack) to
//! build a packed system.

use crate::constraint::extract_molecule_constraint;
use crate::frame::PyFrame;
use crate::helpers::NpF;
use molrs_pack::target::Target;
use molrs_pack::F;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Molecule specification for molecular packing.
///
/// Exposed to Python as `molrs.Target`.
///
/// A target holds the template geometry (atom positions + radii) for one
/// molecular species and the number of copies to pack. Constraints and
/// other options are added via builder-style methods that return **new**
/// `Target` instances (the original is unchanged).
///
/// Examples
/// --------
/// >>> target = Target(frame, count=100)
/// >>> target = target.with_name("water")
/// >>> target = target.with_constraint(InsideBox([0,0,0], [30,30,30]))
/// >>> result = packer.pack([target])
#[pyclass(name = "Target", from_py_object)]
#[derive(Clone)]
pub struct PyTarget {
    pub(crate) inner: Target,
}

#[pymethods]
impl PyTarget {
    /// Create a target from a Frame and a copy count.
    ///
    /// The frame must contain an ``"atoms"`` block with ``x``, ``y``, ``z``
    /// float columns.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     Template molecular geometry.
    /// count : int
    ///     Number of copies to pack.
    ///
    /// Returns
    /// -------
    /// Target
    ///
    /// Examples
    /// --------
    /// >>> target = Target(water_frame, count=500)
    #[new]
    fn new(frame: &PyFrame, count: usize) -> PyResult<Self> {
        Ok(PyTarget {
            inner: Target::new(frame.clone_core_frame()?, count),
        })
    }

    /// Create a target directly from coordinate and radii arrays.
    ///
    /// Useful when the molecule is not stored in a Frame.
    ///
    /// Parameters
    /// ----------
    /// positions : numpy.ndarray, shape (N, 3), dtype float
    ///     Template atom positions.
    /// radii : numpy.ndarray, shape (N,), dtype float
    ///     Atomic radii (used for overlap detection).
    /// count : int
    ///     Number of copies to pack.
    ///
    /// Returns
    /// -------
    /// Target
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``positions`` does not have 3 columns, or lengths mismatch.
    ///
    /// Examples
    /// --------
    /// >>> t = Target.from_coords(positions, radii, count=100)
    #[staticmethod]
    fn from_coords(
        positions: PyReadonlyArray2<'_, NpF>,
        radii: PyReadonlyArray1<'_, NpF>,
        count: usize,
    ) -> PyResult<Self> {
        let pos = positions.as_array();
        let rad = radii.as_array();
        if pos.ncols() != 3 {
            return Err(PyValueError::new_err("positions must have shape (N, 3)"));
        }
        if pos.nrows() != rad.len() {
            return Err(PyValueError::new_err(
                "positions and radii must have the same length",
            ));
        }
        let coords: Vec<[F; 3]> = pos.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
        let radii_vec: Vec<F> = rad.to_vec();
        Ok(PyTarget {
            inner: Target::from_coords(&coords, &radii_vec, count),
        })
    }

    /// Attach a descriptive name (for logging). Returns a new `Target`.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Human-readable name (e.g. ``"water"``, ``"ethanol"``).
    ///
    /// Returns
    /// -------
    /// Target
    fn with_name(&self, name: &str) -> Self {
        PyTarget {
            inner: self.inner.clone().with_name(name),
        }
    }

    /// Add a geometric constraint. Returns a new `Target`.
    ///
    /// Parameters
    /// ----------
    /// constraint : InsideBox | InsideSphere | OutsideSphere | AbovePlane | BelowPlane | MoleculeConstraint
    ///     Constraint to apply to all copies of this molecule.
    ///
    /// Returns
    /// -------
    /// Target
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``constraint`` is not a recognized constraint type.
    ///
    /// Examples
    /// --------
    /// >>> target = target.with_constraint(InsideBox([0,0,0], [30,30,30]))
    fn with_constraint(&self, constraint: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Self> {
        let mc = extract_molecule_constraint(constraint)?;
        Ok(PyTarget {
            inner: self.inner.clone().with_constraint(mc),
        })
    }

    /// Add a geometric constraint for a subset of atoms using Packmol-style
    /// 1-based atom indices. Returns a new `Target`.
    ///
    /// Parameters
    /// ----------
    /// indices : list[int]
    ///     Atom indices using Packmol's 1-based convention.
    /// constraint : InsideBox | InsideSphere | OutsideSphere | AbovePlane | BelowPlane | MoleculeConstraint
    ///     Constraint applied only to the selected atoms.
    ///
    /// Returns
    /// -------
    /// Target
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If any index is outside ``1..natoms``.
    /// TypeError
    ///     If ``constraint`` is not a recognized constraint type.
    fn with_constraint_for_atoms(
        &self,
        indices: Vec<usize>,
        constraint: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<Self> {
        validate_atom_indices(&indices, self.inner.natoms())?;
        let mc = extract_molecule_constraint(constraint)?;
        Ok(PyTarget {
            inner: self.inner.clone().with_constraint_for_atoms(&indices, mc),
        })
    }

    /// Set the ``maxmove`` parameter for the movebad heuristic.
    /// Returns a new `Target`.
    ///
    /// Parameters
    /// ----------
    /// maxmove : int
    ///     Maximum number of molecules to relocate per movebad iteration.
    ///
    /// Returns
    /// -------
    /// Target
    fn with_maxmove(&self, maxmove: usize) -> Self {
        PyTarget {
            inner: self.inner.clone().with_maxmove(maxmove),
        }
    }

    /// Enable centering of the molecule at the constraint region center.
    /// Returns a new `Target`.
    ///
    /// Returns
    /// -------
    /// Target
    fn with_center(&self) -> Self {
        PyTarget {
            inner: self.inner.clone().with_center(),
        }
    }

    /// Disable centering. Returns a new `Target`.
    ///
    /// Returns
    /// -------
    /// Target
    fn without_centering(&self) -> Self {
        PyTarget {
            inner: self.inner.clone().without_centering(),
        }
    }

    /// Constrain rotation around the x axis in degrees. Returns a new
    /// `Target`.
    fn constrain_rotation_x(&self, center_deg: NpF, half_width_deg: NpF) -> Self {
        PyTarget {
            inner: self
                .inner
                .clone()
                .constrain_rotation_x(center_deg, half_width_deg),
        }
    }

    /// Constrain rotation around the y axis in degrees. Returns a new
    /// `Target`.
    fn constrain_rotation_y(&self, center_deg: NpF, half_width_deg: NpF) -> Self {
        PyTarget {
            inner: self
                .inner
                .clone()
                .constrain_rotation_y(center_deg, half_width_deg),
        }
    }

    /// Constrain rotation around the z axis in degrees. Returns a new
    /// `Target`.
    fn constrain_rotation_z(&self, center_deg: NpF, half_width_deg: NpF) -> Self {
        PyTarget {
            inner: self
                .inner
                .clone()
                .constrain_rotation_z(center_deg, half_width_deg),
        }
    }

    /// Fix this molecule at a specific Cartesian position. Returns a new
    /// `Target`.
    ///
    /// A fixed target is excluded from optimization -- its gradient is zeroed
    /// and its position is not updated.
    ///
    /// Parameters
    /// ----------
    /// position : list[float] | tuple[float, float, float]
    ///     ``[x, y, z]`` position to fix at.
    ///
    /// Returns
    /// -------
    /// Target
    fn fixed_at(&self, position: [NpF; 3]) -> Self {
        PyTarget {
            inner: self.inner.clone().fixed_at(position),
        }
    }

    /// Fix this molecule at a specific Cartesian position and Euler
    /// orientation. Returns a new `Target`.
    fn fixed_at_with_euler(&self, position: [NpF; 3], euler: [NpF; 3]) -> Self {
        PyTarget {
            inner: self.inner.clone().fixed_at_with_euler(position, euler),
        }
    }

    /// Number of atoms in the template molecule.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    fn natoms(&self) -> usize {
        self.inner.natoms()
    }

    /// Number of copies to pack.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    fn count(&self) -> usize {
        self.inner.count
    }

    /// Element symbols for each atom in the template.
    ///
    /// Returns
    /// -------
    /// list[str]
    ///     e.g. ``["O", "H", "H"]`` for water.
    #[getter]
    fn elements(&self) -> Vec<String> {
        self.inner.elements.clone()
    }

    /// Whether this target is fixed at a specific position.
    ///
    /// Returns
    /// -------
    /// bool
    #[getter]
    fn is_fixed(&self) -> bool {
        self.inner.fixed_at.is_some()
    }

    fn __repr__(&self) -> String {
        format!(
            "Target(natoms={}, count={}, name={:?})",
            self.inner.natoms(),
            self.inner.count,
            self.inner.name
        )
    }
}

fn validate_atom_indices(indices: &[usize], natoms: usize) -> PyResult<()> {
    for &index in indices {
        if index == 0 || index > natoms {
            return Err(PyValueError::new_err(format!(
                "atom indices must use Packmol 1-based indexing in 1..={natoms}, got {index}",
            )));
        }
    }
    Ok(())
}
