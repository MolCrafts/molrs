//! Python wrapper for `Atomistic`, the all-atom molecular graph.
//!
//! An [`PyAtomistic`] represents the molecular connectivity as a graph where
//! every node is an atom identified by its element symbol and every edge is a
//! chemical bond. It is the starting point for 3D coordinate generation
//! ([`crate::embed`]) and MMFF94 force-field evaluation
//! ([`crate::forcefield`]).

use pyo3::prelude::*;

use molrs::atomistic::Atomistic;
use molrs::molgraph::{AtomId, PropValue};

use crate::frame::PyFrame;
use crate::helpers::molrs_error_to_pyerr;

/// All-atom molecular graph, exposed to Python as `molrs.Atomistic`.
///
/// Atoms are identified by element symbol and optional Cartesian coordinates.
/// Bonds connect pairs of atoms and carry an optional bond order property.
///
/// # Python Examples
///
/// ```python
/// from molrs import Atomistic
///
/// mol = Atomistic()
/// c = mol.add_atom("C")
/// o = mol.add_atom("O")
/// mol.add_bond(c, o)
/// mol.set_bond_order(c, o, 2.0)
/// print(mol)  # Atomistic(atoms=2, bonds=1)
/// ```
#[pyclass(name = "Atomistic", unsendable)]
pub struct PyAtomistic {
    pub(crate) inner: Atomistic,
}

#[pymethods]
impl PyAtomistic {
    /// Create an empty molecular graph with no atoms or bonds.
    ///
    /// Returns
    /// -------
    /// Atomistic
    #[new]
    fn new() -> Self {
        Self {
            inner: Atomistic::new(),
        }
    }

    /// Add an atom with the given element symbol and optional 3D coordinates.
    ///
    /// Parameters
    /// ----------
    /// symbol : str
    ///     Chemical element symbol (e.g. ``"C"``, ``"O"``, ``"H"``).
    /// x : float, optional
    ///     X coordinate in angstroms.
    /// y : float, optional
    ///     Y coordinate in angstroms.
    /// z : float, optional
    ///     Z coordinate in angstroms.
    ///
    /// Returns
    /// -------
    /// int
    ///     Zero-based index of the newly added atom.
    ///
    /// Notes
    /// -----
    /// If any one of ``x``, ``y``, ``z`` is ``None``, no coordinates are
    /// stored for this atom. All three must be provided together.
    ///
    /// Examples
    /// --------
    /// >>> mol = Atomistic()
    /// >>> idx = mol.add_atom("C", x=0.0, y=0.0, z=0.0)
    /// >>> idx
    /// 0
    #[pyo3(signature = (symbol, x=None, y=None, z=None))]
    fn add_atom(&mut self, symbol: &str, x: Option<f64>, y: Option<f64>, z: Option<f64>) -> usize {
        let id = match (x, y, z) {
            (Some(xv), Some(yv), Some(zv)) => self.inner.add_atom_xyz(symbol, xv, yv, zv),
            _ => self.inner.add_atom_bare(symbol),
        };
        self.atom_id_to_index(id)
    }

    /// Add a bond between two atoms (referenced by index).
    ///
    /// Parameters
    /// ----------
    /// i : int
    ///     Zero-based index of the first atom.
    /// j : int
    ///     Zero-based index of the second atom.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If ``i`` or ``j`` is out of range.
    /// ValueError
    ///     If the bond already exists or ``i == j``.
    ///
    /// Examples
    /// --------
    /// >>> mol.add_bond(0, 1)
    fn add_bond(&mut self, i: usize, j: usize) -> PyResult<()> {
        let ai = self.index_to_atom_id(i)?;
        let aj = self.index_to_atom_id(j)?;
        self.inner.add_bond(ai, aj).map_err(molrs_error_to_pyerr)?;
        Ok(())
    }

    /// Set the bond order between two bonded atoms.
    ///
    /// Parameters
    /// ----------
    /// i : int
    ///     Zero-based index of the first atom.
    /// j : int
    ///     Zero-based index of the second atom.
    /// order : float
    ///     Bond order: ``1.0`` (single), ``1.5`` (aromatic), ``2.0`` (double),
    ///     ``3.0`` (triple).
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If ``i`` or ``j`` is out of range.
    /// ValueError
    ///     If no bond exists between atoms ``i`` and ``j``.
    ///
    /// Examples
    /// --------
    /// >>> mol.set_bond_order(0, 1, 2.0)  # double bond
    fn set_bond_order(&mut self, i: usize, j: usize, order: f64) -> PyResult<()> {
        let ai = self.index_to_atom_id(i)?;
        let aj = self.index_to_atom_id(j)?;
        // Find the bond ID first to avoid overlapping borrows.
        let target_bid = self.inner.bonds().find_map(|(bid, bond)| {
            if (bond.atoms[0] == ai && bond.atoms[1] == aj)
                || (bond.atoms[0] == aj && bond.atoms[1] == ai)
            {
                Some(bid)
            } else {
                None
            }
        });
        match target_bid {
            Some(bid) => {
                self.inner
                    .get_bond_mut(bid)
                    .map_err(molrs_error_to_pyerr)?
                    .props
                    .insert("order".into(), PropValue::F64(order));
                Ok(())
            }
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "no bond between atoms {} and {}",
                i, j
            ))),
        }
    }

    /// Number of atoms in this molecular graph.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    /// Number of bonds in this molecular graph.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    fn n_bonds(&self) -> usize {
        self.inner.n_bonds()
    }

    /// Export the molecular graph as a :class:`Frame`.
    ///
    /// The returned frame contains:
    ///
    /// - ``"atoms"`` block with ``symbol`` (str), ``x``/``y``/``z`` (float)
    /// - ``"bonds"`` block with ``i``/``j`` (uint), ``order`` (float)
    ///
    /// Returns
    /// -------
    /// Frame
    ///
    /// Examples
    /// --------
    /// >>> frame = mol.to_frame()
    /// >>> frame["atoms"].nrows
    /// 3
    fn to_frame(&self) -> PyResult<PyFrame> {
        let frame = self.inner.to_frame();
        PyFrame::from_core_frame(frame)
    }

    fn __repr__(&self) -> String {
        format!(
            "Atomistic(atoms={}, bonds={})",
            self.inner.n_atoms(),
            self.inner.n_bonds()
        )
    }
}

impl PyAtomistic {
    /// Convert an internal `AtomId` to its sequential index.
    fn atom_id_to_index(&self, target: AtomId) -> usize {
        self.inner
            .atoms()
            .enumerate()
            .find(|(_, (id, _))| *id == target)
            .map(|(idx, _)| idx)
            .expect("atom id not found")
    }

    /// Convert a sequential index to the internal `AtomId`.
    fn index_to_atom_id(&self, index: usize) -> PyResult<AtomId> {
        self.inner
            .atoms()
            .nth(index)
            .map(|(id, _)| id)
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "atom index {} out of range",
                    index
                ))
            })
    }
}
