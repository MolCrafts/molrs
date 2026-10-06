//! Python bindings for `molrs::ff::potential` — kernels built by hand.
//!
//! * `kernel(category, style, atoms, *, charges=None, **params)` — the one
//!   way to build the kernel of **any** registered style (built-in, custom,
//!   expression- or Python-priced, a custom category's) over explicit
//!   instances: atom indices and one parameter row per term, as stored (the
//!   force-field IR's units, angle values in degrees). It is
//!   [`molrs::ff::potential::Instances`], so the kernel is priced by the code
//!   a compiled force field is. It returns a `Potentials`, which
//!   `Potentials.push` moves into a larger collection.
//! * `LJCut` — the one-type `lj/cut` kernel a neighbour loop feeds (MD's
//!   nonbond kernel: `eval`, `eval_table`, `eval_pairs`, per-pair calls).
//!
//! ```text
//! pots = Potentials()
//! pots.push(kernel("bond", "harmonic", [[0, 1]], k=300.0, r0=1.4))
//! pots.push(kernel("angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=109.5))
//! energy, forces = pots.calc_energy_forces(pos)
//! ```

use molrs::ff::forcefield::Params;
use molrs::ff::ir::{self as rir, ParamKind, StyleSpec};
use molrs::ff::potential::pair::{LJCut, PairPotential};
use molrs::ff::potential::{Instances, Potential};
use molrs::types::F;
use ndarray::{Array2, ArrayD, Axis};
use numpy::{
    IntoPyArray, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use super::ir::{self, column, declared};
use super::{PotBacking, PyPotentials};
use crate::core::spatial::neighborlist::{PyNeighbors, PyVerletSkin};
use crate::helpers::NpF;
use crate::md::check_nx3;

/// The terms' atoms: an ``(n, arity)`` array of indices.
fn term_atoms(atoms: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<usize>>> {
    let rows: Vec<Vec<usize>> = atoms.extract().map_err(|_| {
        PyTypeError::new_err("atoms: an (n, arity) array of atom indices, one row per term")
    })?;
    Ok(rows)
}

/// The value of per-term parameter `key` for each of `n` terms.
enum Column {
    Num(Vec<F>),
    Array(Vec<ArrayD<F>>),
    Text(Vec<String>),
}

fn per_term(
    py: Python<'_>,
    what: &str,
    kind: Option<&ParamKind>,
    value: &Bound<'_, PyAny>,
    n: usize,
) -> PyResult<Column> {
    match kind {
        Some(ParamKind::Text { .. }) => {
            let col = match value.cast::<PyString>() {
                Ok(s) => vec![s.to_str()?.to_owned(); n],
                Err(_) => value.extract::<Vec<String>>()?,
            };
            if col.len() != n {
                return Err(PyValueError::new_err(format!(
                    "{what}: {} strings for {n} terms",
                    col.len()
                )));
            }
            Ok(Column::Text(col))
        }
        Some(ParamKind::Array { rank }) => {
            let a = py
                .import("numpy")?
                .call_method1("asarray", (value, "float64"))?
                .cast_into::<PyArrayDyn<F>>()?
                .readonly()
                .as_array()
                .to_owned();
            let rank = *rank as usize;
            match a.ndim() {
                d if d == rank => Ok(Column::Array(vec![a; n])),
                d if d == rank + 1 && a.shape()[0] == n => Ok(Column::Array(
                    a.axis_iter(Axis(0)).map(|row| row.to_owned()).collect(),
                )),
                _ => Err(PyValueError::new_err(format!(
                    "{what}: an array of rank {rank}, or {n} of them stacked; got shape {:?}",
                    a.shape()
                ))),
            }
        }
        Some(ParamKind::Scalar) | None => Ok(Column::Num(column(what, value, n)?)),
    }
}

/// Build the kernel of one style over explicit instances.
///
/// Works for every style the force-field IR prices: a built-in, a style
/// registered from Python (by expression or kernel, :mod:`molrs.ff.ir`), a
/// style of a custom category, or an unregistered style given its
/// ``expression=``. The kernel is built exactly as
/// :meth:`PotentialCompiler.compile` builds it, one type per term.
///
/// Parameters
/// ----------
/// category, style : str
///     The style (``"bond", "harmonic"``; ``"pair", "lj/cut"``; …).
/// atoms : array_like of int, shape (n, arity)
///     Each term's atoms. A pair term is an atom pair, priced with its own
///     row (the pair's cross row).
/// charges : array_like of float, optional
///     Per-atom charges (``atoms.charge``), read by ``coul/cut`` and by
///     pair expressions through ``q1``, ``q2``.
/// **params
///     Style parameters (``cutoff``, ``mixing``, ``coulomb``, …, an
///     unregistered style's ``expression``) as a number or a string; every
///     other one per term **as stored** (angle values in degrees): a number
///     (broadcast) or one value per term, an array parameter one array (or
///     ``n`` stacked), a text parameter a string or one per term. Indexed
///     families are spelled ``k1``, ``k2``, …. A style whose numbers are per
///     instance (``coul/cut``) takes none.
///
/// Returns
/// -------
/// Potentials
///     One member; ``Potentials.push`` moves it into another collection.
///
/// Raises
/// ------
/// IrError
///     The IR's refusal, by its subclass: ``UnknownCategory``, ``Arity``
///     (a row of the wrong length), ``NoKernel``, ``MissingParam``, ….
/// TypeError
///     A parameter the registered style does not declare.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from molrs.ff.potential import kernel
/// >>> pots = kernel("bond", "harmonic", [[0, 1]], k=300.0, r0=1.5)
/// >>> e, f = pots.calc_energy_forces(np.array([0.0, 0, 0, 1.6, 0, 0]))
/// >>> round(e, 12)
/// 3.0
#[pyfunction]
#[pyo3(signature = (category, style, atoms, *, charges=None, **params))]
fn kernel(
    py: Python<'_>,
    category: &str,
    style: &str,
    atoms: &Bound<'_, PyAny>,
    charges: Option<Vec<F>>,
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyPotentials> {
    let who = format!("{category} `{style}`");
    let atoms = term_atoms(atoms)?;
    let n = atoms.len();
    let (pair, spec): (bool, Option<StyleSpec>) = rir::with_global(|r| {
        (
            r.category(category).is_some_and(|c| c.is_pair_driven()),
            r.style(category, style).map(|(s, _)| s.clone()),
        )
    });
    let mut style_params = Params::new();
    let mut columns: Vec<(String, Column)> = Vec::new();
    for (k, v) in params.into_iter().flat_map(|d| d.iter()) {
        let key: String = k.extract()?;
        let what = format!("{who}: `{key}`");
        let decl =
            match &spec {
                Some(spec) => Some(declared(spec, pair, &key).ok_or_else(|| {
                    PyTypeError::new_err(format!("{who} has no parameter `{key}`"))
                })?),
                None => None,
            };
        if pair && matches!(decl, Some((None, _))) {
            return Err(PyTypeError::new_err(format!(
                "{who}: `{key}` is per atom; pass charges= (a term's row is its cross row)"
            )));
        }
        let style_level = match decl {
            Some((_, style_level)) => style_level,
            // An unregistered style declares nothing: its expression is the
            // one style parameter, every other value is per term.
            None => key == "expression",
        };
        if style_level {
            match v.cast::<PyString>() {
                Ok(s) => style_params.set_str(&key, s.to_str()?),
                Err(_) => style_params.set(&key, v.extract::<F>()?),
            }
            continue;
        }
        let kind = decl.and_then(|(p, _)| p).map(|p| &p.kind);
        columns.push((key, per_term(py, &what, kind, &v, n)?));
    }
    let mut terms = Instances::new(category, style).style_params(style_params);
    if columns.is_empty() {
        terms = terms.atoms(atoms);
    } else {
        for (t, atoms) in atoms.iter().enumerate() {
            let mut row = Params::new();
            for (key, col) in &columns {
                match col {
                    Column::Num(c) => row.set(key, c[t]),
                    Column::Array(c) => row.set_array(key, c[t].clone()),
                    Column::Text(c) => row.set_str(key, &c[t]),
                }
            }
            terms = terms.term(atoms, row);
        }
    }
    if let Some(q) = charges {
        terms = terms.charges(q);
    }
    ir::clear_kernel_err();
    let pots = terms.compile().map_err(ir::compile_err)?;
    ir::take_kernel_err()?;
    Ok(PyPotentials {
        inner: PotBacking::Compiled(pots),
        err_slots: vec![ir::kernel_err_slot()],
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLJCut>()?;
    m.add_function(wrap_pyfunction!(kernel, m)?)?;
    Ok(())
}

/// LAMMPS ``pair_style lj/cut``: the one-type cut Lennard-Jones / Mie kernel
/// a neighbour loop feeds pairs to (MD's nonbond kernel). A pair list with a
/// row per pair is ``kernel("pair", "lj/cut", pairs, epsilon=…, sigma=…)``.
#[pyclass(name = "LJCut", module = "molrs.ff.potential", subclass)]
pub struct PyLJCut {
    pub(crate) inner: LJCut,
}

#[pymethods]
impl PyLJCut {
    #[new]
    #[pyo3(signature = (epsilon, sigma, cutoff, *, n=12, m=6, shifted=true, smeared=false))]
    fn new(
        epsilon: F,
        sigma: F,
        cutoff: F,
        n: i32,
        m: i32,
        shifted: bool,
        smeared: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LJCut::new(epsilon, sigma, cutoff, n, m, shifted, smeared)
                .map_err(PyValueError::new_err)?,
        })
    }

    #[getter]
    fn epsilon(&self) -> F {
        self.inner.epsilon()
    }
    #[getter]
    fn sigma(&self) -> F {
        self.inner.sigma()
    }
    #[getter]
    fn cutoff(&self) -> F {
        self.inner.cutoff()
    }
    #[getter]
    fn n(&self) -> i32 {
        self.inner.n()
    }
    #[getter]
    fn m(&self) -> i32 {
        self.inner.m()
    }
    #[getter]
    fn shifted(&self) -> bool {
        self.inner.shifted()
    }
    #[getter]
    fn smeared(&self) -> bool {
        self.inner.smeared()
    }

    fn pair_energy(&self, r2: F, disp: [F; 3]) -> Option<F> {
        self.inner.pair_energy(r2, disp)
    }
    fn pair_force(&self, r2: F, disp: [F; 3]) -> Option<[F; 3]> {
        self.inner.pair_force(r2, disp)
    }
    fn pair_eval(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        self.inner.pair_eval(r2, disp)
    }

    fn calc_energy_forces<'py>(
        &self,
        py: Python<'py>,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        check_nx3(&pos, "pos")?;
        let view = pos.as_array();
        let n = view.nrows();
        // A standard-layout `(N, 3)` array *is* the flat `[x0, y0, z0, …]` a
        // kernel wants; only a strided view is copied.
        let owned: Vec<F>;
        let flat: &[F] = match view.as_slice() {
            Some(slice) => slice,
            None => {
                owned = view.iter().copied().collect();
                &owned
            }
        };
        let (energy, forces) = Potential::calc_energy_forces(&self.inner, flat);
        let arr = Array2::from_shape_vec((n, 3), forces)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((energy, arr.into_pyarray(py)))
    }

    fn eval<'py>(
        &self,
        py: Python<'py>,
        neighbors: &mut PyVerletSkin,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        check_nx3(&pos, "pos")?;
        let nl = neighbors.get_mut()?;
        let (e, f) = self
            .inner
            .eval(nl, pos.as_array())
            .map_err(PyValueError::new_err)?;
        Ok((e, f.into_pyarray(py)))
    }

    fn eval_table<'py>(
        &self,
        py: Python<'py>,
        n_atoms: usize,
        neighbors: &PyNeighbors,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        let (e, f) = self
            .inner
            .eval_table(n_atoms, &neighbors.inner)
            .map_err(PyValueError::new_err)?;
        Ok((e, f.into_pyarray(py)))
    }

    #[pyo3(signature = (n_atoms, i, j, disp, dist_sq=None))]
    fn eval_pairs<'py>(
        &self,
        py: Python<'py>,
        n_atoms: usize,
        i: PyReadonlyArray1<'_, u32>,
        j: PyReadonlyArray1<'_, u32>,
        disp: PyReadonlyArray2<'_, NpF>,
        dist_sq: Option<PyReadonlyArray1<'_, NpF>>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        check_nx3(&disp, "disp")?;
        let d2 = match dist_sq.as_ref() {
            Some(a) => Some(a.as_slice()?),
            None => None,
        };
        let (e, f) = self
            .inner
            .eval_pairs(n_atoms, i.as_slice()?, j.as_slice()?, disp.as_array(), d2)
            .map_err(PyValueError::new_err)?;
        Ok((e, f.into_pyarray(py)))
    }
}
