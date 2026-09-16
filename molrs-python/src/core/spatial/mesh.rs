//! Python wrapper for [`molrs::spatial::TriMesh`].
//!
//! A triangle surface with a shared vertex table — what
//! [`read_stl`](crate::io::read_stl) reads and what a
//! [`Polyhedron`](crate::core::spatial::region::PyPolyhedron) is bounded by.

use crate::helpers::NpF;
use molrs::spatial::TriMesh;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Triangle surface mesh: a vertex table plus faces indexing into it.
///
/// Exposed to Python as `molrs.TriMesh`. Lengths are whatever unit the
/// vertices are in; :meth:`scaled` converts.
///
/// Examples
/// --------
/// >>> mesh = molrs.io.read_stl("cavity.stl").scaled(4.18)
/// >>> mesh.is_watertight()
/// True
/// >>> region = molrs.Polyhedron(mesh)
#[pyclass(module = "molrs", name = "TriMesh", from_py_object)]
#[derive(Clone)]
pub struct PyTriMesh {
    pub(crate) inner: TriMesh,
}

#[pymethods]
impl PyTriMesh {
    /// Build a mesh from a vertex table and a face list.
    ///
    /// Parameters
    /// ----------
    /// vertices : numpy.ndarray, shape (V, 3), dtype float
    ///     Corner coordinates.
    /// faces : numpy.ndarray, shape (F, 3), dtype uint32
    ///     Corner indices per triangle, in winding order.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a face points past the vertex table, or the shapes are wrong.
    #[new]
    fn new(
        vertices: PyReadonlyArray2<'_, NpF>,
        faces: PyReadonlyArray2<'_, u32>,
    ) -> PyResult<Self> {
        let v = vertices.as_array();
        let f = faces.as_array();
        if v.ncols() != 3 {
            return Err(PyValueError::new_err("vertices must have shape (V, 3)"));
        }
        if f.ncols() != 3 {
            return Err(PyValueError::new_err("faces must have shape (F, 3)"));
        }
        let verts: Vec<[NpF; 3]> = v.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
        let face_list: Vec<[u32; 3]> = f.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
        let inner = TriMesh::from_indexed(verts, face_list).map_err(|index| {
            PyValueError::new_err(format!("face {index} points past the vertex table"))
        })?;
        Ok(Self { inner })
    }

    /// The vertex table, shape ``(V, 3)``.
    fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        let verts = self.inner.vertices();
        let mut a = Array2::zeros((verts.len(), 3));
        for (i, v) in verts.iter().enumerate() {
            for k in 0..3 {
                a[[i, k]] = v[k];
            }
        }
        a.into_pyarray(py)
    }

    /// The face list, shape ``(F, 3)``, dtype uint32.
    fn faces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        let faces = self.inner.faces();
        let mut a = Array2::zeros((faces.len(), 3));
        for (i, f) in faces.iter().enumerate() {
            for k in 0..3 {
                a[[i, k]] = f[k];
            }
        }
        a.into_pyarray(py)
    }

    /// Number of vertices.
    #[getter]
    fn n_vertices(&self) -> usize {
        self.inner.n_vertices()
    }

    /// Number of faces.
    #[getter]
    fn n_faces(&self) -> usize {
        self.inner.n_faces()
    }

    /// Whether every directed edge has exactly one opposite — a closed
    /// surface, the gate a :class:`Polyhedron` needs.
    fn is_watertight(&self) -> bool {
        self.inner.is_watertight()
    }

    /// The same mesh with every coordinate multiplied by ``factor``.
    fn scaled(&self, factor: NpF) -> PyResult<Self> {
        if !(factor.is_finite() && factor > 0.0) {
            return Err(PyValueError::new_err(format!(
                "scale factor must be finite and > 0, got {factor}"
            )));
        }
        Ok(Self {
            inner: self.inner.scaled(factor),
        })
    }

    /// Axis-aligned bounds, shape ``(3, 2)``: ``[[xmin, xmax], ...]``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the mesh has no vertices.
    fn bounds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        let (lo, hi) = self
            .inner
            .aabb()
            .ok_or_else(|| PyValueError::new_err("mesh has no vertices"))?;
        let mut b = Array2::zeros((3, 2));
        for d in 0..3 {
            b[[d, 0]] = lo[d];
            b[[d, 1]] = hi[d];
        }
        Ok(b.into_pyarray(py))
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, pyo3::types::PyTuple>)> {
        let py = slf.py();
        let this = slf.borrow();
        crate::helpers::reduce_via_type(slf.as_any(), (this.vertices(py), this.faces(py)))
    }

    fn __repr__(&self) -> String {
        format!(
            "TriMesh(n_vertices={}, n_faces={}, watertight={})",
            self.inner.n_vertices(),
            self.inner.n_faces(),
            self.inner.is_watertight()
        )
    }
}
