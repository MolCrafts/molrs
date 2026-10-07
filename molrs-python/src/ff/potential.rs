//! Python bindings for `molrs::ff::potential` (`molrs.ff.potential`): the
//! evaluable force terms.
//!
//! * [`PyPotentialCompiler`] — a `ForceField` compiled into kernels:
//!   [`PyPotentials`] over a fixed topology, or [`PyTypedPotentials`] (each
//!   kernel with its special-bonds weights) for a neighbour-driven integrator.
//! * [`PyPotentials`] — a collection of kernels evaluated together
//!   (`calc_energy_forces`); `push` moves one more member in.
//! * `kernel(category, style, atoms, *, charges=None, **params)` — the kernel
//!   of **any** registered style over explicit instances (atom indices and
//!   one parameter row per term, as stored: the force-field IR's units, angle
//!   values in degrees). It is [`molrs::ff::potential::Instances`], so the
//!   kernel is priced by the code a compiled force field is.
//! * [`PyLJCut`] — the one-type `lj/cut` kernel a neighbour loop feeds (the
//!   MD integrators' nonbond kernel: `eval`, `eval_table`, `eval_pairs`).
//! * `intramolecular_pairs` — the special-bonds pair list of a typed frame.
//!
//! ```text
//! pots = Potentials()
//! pots.push(kernel("bond", "harmonic", [[0, 1]], k=300.0, r0=1.4))
//! pots.push(kernel("angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=109.5))
//! energy, forces = pots.calc_energy_forces(pos)
//! ```
//!
//! # The Python seam
//!
//! A Python object with ``calc_energy_forces`` (a
//! `molrs.ff.potential.Potential`) is a potential like any other:
//! [`take_potential`] turns any exposed potential — `LJCut`, `Potentials`, or
//! a duck-typed object wrapped as [`SubclassPotential`] — into one
//! [`Member`]; [`take_members`] does the same for a whole
//! [`PyTypedPotentials`]. The `Potential` trait has no error channel, so a
//! Python exception raised mid-evaluation is parked in an [`ErrSlot`] and
//! re-raised by the caller that drove the evaluation ([`take_err`]). The MD
//! integrators and `molrs.optimize.Lbfgs` consume potentials through here.

use super::ir;
use super::ir::{column, declared};
use crate::core::block::PyBlock;
use crate::core::frame::PyFrame;
use crate::core::neighborlist::{PyNeighbors, PyVerletSkin};
use crate::ff::forcefield::PyForceField;
use molrs::ff::forcefield::{ForceField, Params};
use molrs::ff::ir::{self as rir, ParamKind, StyleSpec};
use molrs::ff::potential::pair::{LJCut, PairPotential};
use molrs::ff::potential::{Instances, Member, Potential, PotentialCompiler, Potentials};
use molrs::op::F;
use ndarray::{Array2, ArrayD, Axis};
use numpy::{
    IntoPyArray, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    ToPyArray,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use std::sync::{Arc, Mutex};

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

/// Register `molrs.ff.potential`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPotentialCompiler>()?;
    m.add_class::<PyPotentials>()?;
    m.add_class::<PyTypedPotentials>()?;
    m.add_class::<PyLJCut>()?;
    crate::add_function(m, "molrs.ff.potential", wrap_pyfunction!(kernel, m)?)?;
    crate::add_function(
        m,
        "molrs.ff.potential",
        wrap_pyfunction!(intramolecular_pairs_py, m)?,
    )?;
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
        pos: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<(F, Bound<'py, PyArray2<f64>>)> {
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
        pos: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<(F, Bound<'py, PyArray2<f64>>)> {
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
    ) -> PyResult<(F, Bound<'py, PyArray2<f64>>)> {
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
        disp: PyReadonlyArray2<'_, f64>,
        dist_sq: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<(F, Bound<'py, PyArray2<f64>>)> {
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

/// The kernels of a neighbour-driven force evaluation, each with the
/// special-bonds weights that scale it.
///
/// Opaque on purpose: what a caller does with this is hand it to an
/// integrator. Taking it apart in Python would mean re-deciding which member
/// is which and how its close neighbours are scaled — the two things
/// :meth:`PotentialCompiler.compile_typed` exists to decide once.
#[pyclass(name = "TypedPotentials", module = "molrs.ff.potential", subclass)]
pub struct PyTypedPotentials {
    /// Taken by the integrator that consumes it; `None` afterwards.
    pub(crate) members: Option<Vec<(Member, molrs::ff::potential::SpecialWeights)>>,
}

#[pymethods]
impl PyTypedPotentials {
    /// How many kernels this carries.
    fn __len__(&self) -> usize {
        self.members.as_ref().map_or(0, |m| m.len())
    }
}

/// Compiled force-field potentials for energy and force evaluation.
///
/// Exposed to Python as `molrs.ff.potential.Potentials`.
///
/// Operates on flat coordinate arrays in the layout
/// ``[x0, y0, z0, x1, y1, z1, ...]`` (length 3N).
///
/// Examples
/// --------
/// >>> typifier = MMFF94Typifier()
/// >>> frame = typifier.typify(mol).to_frame()
/// >>> frame["pairs"] = molrs.ff.potential.intramolecular_pairs(frame)
/// >>> potentials = molrs.ff.potential.PotentialCompiler(typifier.forcefield()).compile(frame)
/// >>> energy, forces = potentials.eval(coords)
#[pyclass(module = "molrs.ff.potential", name = "Potentials", subclass)]
pub struct PyPotentials {
    pub(crate) inner: PotBacking,
    /// Error slots of every Python-callable member (see `ErrSlot`);
    /// checked after each evaluation so a callable's exception re-raises.
    pub(crate) err_slots: Vec<ErrSlot>,
}

/// A [`PyPotentials`] is either already compiled against a molecule's topology
/// (the MMFF / pre-bound path), *deferred* (it holds the force field and binds
/// the topology lazily from the `Frame` passed to
/// ``calc_energy``/``calc_forces`` — what ``PotentialCompiler.defer()``
/// returns, matching the molpy evaluation model), or *moved*: the
/// Rust `Potentials` has been moved into an MD integrator or another
/// collection.
pub(crate) enum PotBacking {
    Compiled(Potentials),
    Deferred(ForceField),
    Moved,
}

pub(crate) fn potentials_moved_err() -> PyErr {
    PyValueError::new_err(
        "this Potentials has been moved into an integrator or another \
         Potentials; rebuild with PotentialCompiler.compile(frame)",
    )
}

impl PotBacking {
    /// The compiled potentials, or an error if this set is still deferred and
    /// no `Frame` has been supplied to bind its topology (or already moved).
    pub(crate) fn compiled(&self) -> PyResult<&Potentials> {
        match self {
            PotBacking::Compiled(p) => Ok(p),
            PotBacking::Deferred(_) => Err(PyValueError::new_err(
                "this Potentials is not bound to a molecule; \
                 call calc_energy(frame)/calc_forces(frame) with a Frame, \
                 or build it from a typifier",
            )),
            PotBacking::Moved => Err(potentials_moved_err()),
        }
    }

    /// Mutable access with the same gating as [`compiled`](Self::compiled).
    fn compiled_mut(&mut self) -> PyResult<&mut Potentials> {
        match self {
            PotBacking::Compiled(p) => Ok(p),
            PotBacking::Deferred(_) => Err(PyValueError::new_err(
                "this Potentials is not bound to a molecule; \
                 call calc_energy(frame)/calc_forces(frame) with a Frame, \
                 or build it from a typifier",
            )),
            PotBacking::Moved => Err(potentials_moved_err()),
        }
    }
}

impl PyPotentials {
    /// Evaluate energy + forces against either a [`PyFrame`] (binds topology and
    /// reads coordinates from the frame's ``atoms`` block — the molpy model) or a
    /// flat coordinate array (requires already-compiled potentials).
    ///
    /// A Python-callable member's exception is re-raised afterwards (see
    /// `ErrSlot`).
    fn eval_any(&self, arg: &Bound<'_, PyAny>) -> PyResult<(f64, Vec<f64>)> {
        let ef = if let Ok(frame) = arg.extract::<PyRef<'_, PyFrame>>() {
            let core = frame.clone_core_frame()?;
            let coords: Vec<f64> = core
                .coords()
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_iter()
                .collect();
            match &self.inner {
                PotBacking::Compiled(p) => p.calc_energy_forces(&coords),
                PotBacking::Deferred(ff) => PotentialCompiler::new(ff)
                    .compile(&core)
                    .map_err(ir::compile_err)?
                    .calc_energy_forces(&coords),
                PotBacking::Moved => return Err(potentials_moved_err()),
            }
        } else {
            let arr = arg.extract::<numpy::PyReadonlyArray1<'_, f64>>()?;
            let slice = arr.as_slice()?;
            self.inner.compiled()?.calc_energy_forces(slice)
        };
        take_err(&self.err_slots)?;
        Ok(ef)
    }

    /// Move the compiled Rust `Potentials` (and the error slots of its
    /// Python-callable members) out, leaving this object in the moved state —
    /// the MD integrators and `Potentials.push` consume through here.
    pub(crate) fn take_compiled(&mut self) -> PyResult<(Potentials, Vec<ErrSlot>)> {
        match std::mem::replace(&mut self.inner, PotBacking::Moved) {
            PotBacking::Compiled(p) => Ok((p, std::mem::take(&mut self.err_slots))),
            deferred @ PotBacking::Deferred(_) => {
                self.inner = deferred;
                Err(PyValueError::new_err(
                    "this Potentials is not bound to a molecule; \
                     compile with PotentialCompiler.compile(frame) before moving it into \
                     an integrator",
                ))
            }
            PotBacking::Moved => Err(potentials_moved_err()),
        }
    }
}

#[pymethods]
impl PyPotentials {
    /// An empty collection; compose members with :meth:`push`.
    #[new]
    fn new() -> Self {
        Self {
            inner: PotBacking::Compiled(Potentials::new()),
            err_slots: Vec::new(),
        }
    }

    /// Number of compiled potential kernels, or ``0`` while still deferred
    /// (not yet bound to a molecule) or moved into an integrator.
    fn __len__(&self) -> usize {
        match &self.inner {
            PotBacking::Compiled(p) => p.len(),
            PotBacking::Deferred(_) | PotBacking::Moved => 0,
        }
    }

    /// Move one more member into the collection: an ``LJCut`` nonbond term, a
    /// callable ``Potential``, or another ``Potentials``.
    fn push(&mut self, potential: &Bound<'_, PyAny>) -> PyResult<()> {
        // Gate first so a failed push does not consume the pushed potential.
        self.inner.compiled_mut()?;
        let (member, mut slots) = take_potential(potential)?;
        self.inner.compiled_mut()?.push(member);
        self.err_slots.append(&mut slots);
        Ok(())
    }

    /// Returns ``(energy, forces)``, forces shape ``(N, 3)``.
    fn calc_energy_forces<'py>(
        &self,
        py: Python<'py>,
        arg: &Bound<'_, PyAny>,
    ) -> PyResult<(f64, Bound<'py, PyArray2<f64>>)> {
        let (energy, forces) = self.eval_any(arg)?;
        let n = forces.len() / 3;
        Ok((
            energy,
            Array2::from_shape_vec((n, 3), forces)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                .to_pyarray(py),
        ))
    }

    /// Evaluate total energy (kcal/mol) against a :class:`Frame` or coordinates.
    fn calc_energy(&self, arg: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(self.eval_any(arg)?.0)
    }

    /// Compute forces (= -gradient) in kcal/(mol·Å), shape ``(N, 3)``.
    fn calc_forces<'py>(
        &self,
        py: Python<'py>,
        arg: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let forces = self.eval_any(arg)?.1;
        let n = forces.len() / 3;
        Ok(Array2::from_shape_vec((n, 3), forces)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .to_pyarray(py))
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            PotBacking::Compiled(p) => format!("Potentials(n_kernels={})", p.len()),
            PotBacking::Deferred(_) => "Potentials(deferred)".to_string(),
            PotBacking::Moved => "Potentials(<moved into integrator>)".to_string(),
        }
    }
}

/// Compiles a :class:`ForceField` into evaluable kernels.
///
/// Exposed to Python as ``molrs.ff.potential.PotentialCompiler``. It owns a **copy** of
/// the force field, taken at construction: later edits to that
/// :class:`ForceField` do not reach a compiler that already exists — make a
/// new one.
///
/// Three doors, each doing one thing:
///
/// * :meth:`compile` — bind a typed :class:`Frame` now;
/// * :meth:`defer` — a :class:`Potentials` that binds the topology of the
///   :class:`Frame` it is evaluated on;
/// * :meth:`compile_typed` — the kernels of a neighbour-driven evaluation,
///   for MD.
///
/// Examples
/// --------
/// >>> compiler = molrs.ff.potential.PotentialCompiler(typifier.forcefield())
/// >>> potentials = compiler.compile(frame)
/// >>> energy = potentials.calc_energy(frame)
#[pyclass(module = "molrs.ff.potential", name = "PotentialCompiler", subclass)]
pub struct PyPotentialCompiler {
    ff: ForceField,
}

#[pymethods]
impl PyPotentialCompiler {
    /// Copy ``forcefield`` into a new compiler.
    ///
    /// Parameters
    /// ----------
    /// forcefield : ForceField
    ///     The force field to compile. Copied; later edits do not reach this
    ///     compiler.
    #[new]
    fn new(forcefield: PyRef<'_, PyForceField>) -> Self {
        Self {
            ff: forcefield.inner.clone(),
        }
    }

    /// Build evaluable :class:`Potentials` against a typed ``frame``.
    ///
    /// The frame must carry the topology + ``type`` columns each style
    /// resolves (``atoms``/``bonds``/``angles``/``dihedrals``/``impropers``/
    /// ``pairs``), as produced by a typifier or an external emitter. Every
    /// pair style is resolved against the frame's ``pairs`` block — a fixed
    /// list, right for a molecule in free space — and prices a row only
    /// inside its ``cutoff`` (``r < cutoff``, with its switch where it has
    /// one), as :meth:`compile_typed` and LAMMPS do; a style stating no
    /// ``cutoff`` prices every row. ``pair coul/long/pme`` reads the
    /// frame's periodic ``box``.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     Typed molecular data. Required; for potentials that bind at
    ///     evaluation time use :meth:`defer`.
    ///
    /// Returns
    /// -------
    /// Potentials
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``frame`` is not a :class:`Frame` (``None`` included).
    /// ValueError
    ///     If a style has no registered kernel, a type label is unknown, the
    ///     force field's 1-2 / 1-3 weights are not 0 or 1, or a style that
    ///     reads the box (``coul/long/pme``) meets a frame without a periodic
    ///     one.
    fn compile(&self, frame: &PyFrame) -> PyResult<PyPotentials> {
        ir::clear_kernel_err();
        let potentials = frame
            .with_frame(|core| PotentialCompiler::new(&self.ff).compile(core))?
            .map_err(ir::compile_err)?;
        ir::take_kernel_err()?;
        Ok(PyPotentials {
            inner: PotBacking::Compiled(potentials),
            err_slots: vec![ir::kernel_err_slot()],
        })
    }

    /// A :class:`Potentials` that compiles when it is evaluated.
    ///
    /// It holds this compiler's force field and binds the topology and
    /// coordinates of the :class:`Frame` passed to ``calc_energy(frame)`` /
    /// ``calc_forces(frame)`` (the molpy evaluation model). It has no members
    /// until then, so ``len`` is ``0``, and it cannot be evaluated on a bare
    /// coordinate array or moved into an integrator.
    ///
    /// Returns
    /// -------
    /// Potentials
    fn defer(&self) -> PyPotentials {
        PyPotentials {
            inner: PotBacking::Deferred(self.ff.clone()),
            err_slots: vec![ir::kernel_err_slot()],
        }
    }

    /// Build the kernels for a **neighbour-driven** evaluation, with the
    /// special-bonds weights each one takes.
    ///
    /// The counterpart of :meth:`compile`, and what periodic MD needs. That
    /// one resolves every pair style against the frame's ``pairs`` block — a
    /// fixed list, right for a free-boundary molecule and wrong for a
    /// periodic system; over the same pairs the two price the same energy. This one resolves them against the
    /// **atoms**, reads no ``pairs`` block, and requires the style's declared
    /// cutoff.
    ///
    /// The weights come from the force field's ``special_bonds`` walked over
    /// the frame's bond graph. Without them a neighbour table would count a
    /// bonded pair twice: once by the bond term and once at full non-bonded
    /// strength, at bond length.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     Typed molecular data.
    ///
    /// Returns
    /// -------
    /// TypedPotentials
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a style cannot be built, or a pair style has no
    ///     neighbour-driven form.
    fn compile_typed(&self, frame: &PyFrame) -> PyResult<PyTypedPotentials> {
        let (topo, members) = frame.with_frame(|core| -> PyResult<_> {
            let topo = molrs::core::Topology::from_frame(core)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            ir::clear_kernel_err();
            let members = PotentialCompiler::new(&self.ff)
                .compile_typed(core)
                .map_err(ir::compile_err)?;
            ir::take_kernel_err()?;
            Ok((topo, members))
        })??;
        let bound = members
            .into_iter()
            .map(|(pot, weights)| {
                let special = weights
                    .map(|w| w.special_weights(&topo))
                    .unwrap_or_default();
                (pot, special)
            })
            .collect();
        Ok(PyTypedPotentials {
            members: Some(bound),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PotentialCompiler(forcefield='{}', styles={})",
            self.ff.name,
            self.ff.styles().len()
        )
    }
}

/// Build the intramolecular non-bonded neighbour list for a typed frame.
///
/// Returns a :class:`Block` with ``atomi`` / ``atomj`` / ``is_14`` columns — the
/// exact list :meth:`PotentialCompiler.compile` consumes for the pair (van der
/// Waals + Coulomb) kernels. 1-4 pairs (from ``dihedrals``) are flagged so the
/// kernels apply the force field's 1-4 scaling.
///
/// Which 1-2 / 1-3 neighbours belong in the list is the **force field's**
/// decision, so pass it: LAMMPS ``special_bonds fene`` (``[0, 1, 1]``) keeps
/// 1-3 pairs at full strength, and a bead-spring chain without them has
/// nothing holding it open. Omitting ``forcefield`` excludes both classes —
/// what every Amber-family force field wants, and what this function always
/// did before it could be told otherwise.
///
/// Insert the result as the frame's ``"pairs"`` block before
/// :meth:`PotentialCompiler.compile` when you need the non-bonded terms — e.g. a
/// single-molecule geometry optimization where intramolecular van der Waals
/// drives chain collapse. (``compile`` silently drops the pair styles when
/// no ``"pairs"`` block is present, so bonded-only optimizations need nothing.)
///
/// Parameters
/// ----------
/// frame : Frame
///     A typed frame with ``atoms`` and the topology blocks
///     (``bonds`` / ``angles`` / ``dihedrals``) used for exclusions.
/// forcefield : ForceField, optional
///     The force field whose ``special_bonds`` decide the 1-2 / 1-3 rows.
///     Defaults to excluding both.
///
/// Returns
/// -------
/// Block
///
/// Raises
/// ------
/// ValueError
///     The force field scales 1-2 or 1-3 neighbours by a fraction, or scales
///     them differently for van der Waals and Coulomb. A list of rows cannot
///     say either; use :meth:`PotentialCompiler.compile_typed`, which carries a
///     per-pair weight.
#[pyfunction]
#[pyo3(name = "intramolecular_pairs", signature = (frame, forcefield = None))]
pub fn intramolecular_pairs_py(
    frame: &PyFrame,
    forcefield: Option<&PyForceField>,
) -> PyResult<PyBlock> {
    let owned;
    let special = match forcefield {
        Some(ff) => ff.inner.special_bonds(),
        None => {
            owned = molrs::ff::forcefield::SpecialBonds::default();
            &owned
        }
    };
    let block = frame
        .with_frame(|core| molrs::ff::potential::intramolecular_pairs(core, special))?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    PyBlock::from_core_block(block)
}

pub(crate) fn check_nx3(arr: &PyReadonlyArray2<'_, f64>, label: &str) -> PyResult<()> {
    if arr.as_array().ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape (N, 3)"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The one Potential seam — Python subclasses and post-evaluation error relay.
// ---------------------------------------------------------------------------

/// Shared slot where a Python-subclass potential parks an exception raised
/// mid-evaluation — the `Potential` trait has no error channel, so the
/// evaluation returns NaNs and the Python-facing caller that drove it checks
/// the slot and re-raises the original exception.
pub(crate) type ErrSlot = Arc<Mutex<Option<PyErr>>>;

/// Re-raise the first parked exception, clearing its slot.
pub(crate) fn take_err(slots: &[ErrSlot]) -> PyResult<()> {
    for slot in slots {
        if let Some(err) = slot.lock().expect("error slot poisoned").take() {
            return Err(err);
        }
    }
    Ok(())
}

/// A Python object with `calc_energy_forces` (a `molrs.ff.potential.Potential`)
/// as the one `Potential` concept — the seam for NN / external forces. Holds a reference to the instance and
/// dispatches to its overridden ``calc_energy_forces`` under the GIL.
pub struct SubclassPotential {
    obj: Py<PyAny>,
    error: ErrSlot,
}

impl SubclassPotential {
    fn call(&self, py: Python<'_>, coords: &[F]) -> PyResult<(F, Vec<F>)> {
        let n = coords.len() / 3;
        let pos = Array2::from_shape_vec((n, 3), coords.to_vec())
            .expect("flat coords have 3N elements")
            .into_pyarray(py);
        let result = self
            .obj
            .bind(py)
            .call_method1("calc_energy_forces", (pos,))?;
        let (energy, forces): (F, PyReadonlyArray2<'_, f64>) = result.extract().map_err(|_| {
            PyValueError::new_err(
                "Potential.calc_energy_forces must return \
                 (energy: float, forces: float64 (N, 3) ndarray)",
            )
        })?;
        let forces = forces.as_array();
        if forces.shape() != [n, 3] {
            return Err(PyValueError::new_err(format!(
                "Potential.calc_energy_forces returned forces shape {:?} for {n} atoms",
                forces.shape()
            )));
        }
        Ok((energy, forces.iter().copied().collect()))
    }
}

impl Potential for SubclassPotential {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        Python::attach(|py| match self.call(py, coords) {
            Ok(out) => out,
            Err(err) => {
                *self.error.lock().expect("error slot poisoned") = Some(err);
                (F::NAN, vec![F::NAN; coords.len()])
            }
        })
    }
}

/// The members a provider will evaluate, with the weights each one takes.
///
/// A [`TypedPotentials`](PyTypedPotentials) already knows both —
/// which kernel is which and how its close neighbours are scaled — because
/// `PotentialCompiler::compile_typed` decided it. Anything else is one member
/// that scales nothing.
pub(crate) type Members = Vec<(Member, molrs::ff::potential::SpecialWeights)>;

/// Move the Rust potential out of any exposed potential class.
///
/// Arm order is a hard invariant: concrete Rust types first, duck-typed
/// fallback last. Putting the fallback first would wrap every `Potentials`
/// as a Python dispatch object.
pub(crate) fn take_members(obj: &Bound<'_, PyAny>) -> PyResult<(Members, Vec<ErrSlot>)> {
    if let Ok(typed) = obj.cast::<PyTypedPotentials>() {
        let members = typed.borrow_mut().members.take().ok_or_else(|| {
            PyValueError::new_err(
                "these TypedPotentials were already given to an integrator; \
                 build them again from the force field",
            )
        })?;
        // A Python kernel among them parks its exception here.
        return Ok((members, vec![ir::kernel_err_slot()]));
    }
    let (pot, slots) = take_potential(obj)?;
    Ok((
        vec![(pot, molrs::ff::potential::SpecialWeights::default())],
        slots,
    ))
}

pub(crate) fn take_potential(obj: &Bound<'_, PyAny>) -> PyResult<(Member, Vec<ErrSlot>)> {
    // Each arm also settles which part the member plays. A pair kernel and an
    // aggregate of them read a neighbour table; a duck-typed Python object has
    // only `calc_energy_forces`, so it reads coordinates and nothing else —
    // and, being unable to tally a virial over pairs, makes the step's virial
    // `None` rather than a number that moves with the box origin.
    if let Ok(lj) = obj.cast::<PyLJCut>() {
        return Ok((Member::pair(lj.borrow().inner.clone()), Vec::new()));
    }
    if let Ok(pots) = obj.cast::<PyPotentials>() {
        let (inner, slots) = pots.borrow_mut().take_compiled()?;
        return Ok((Member::pair(inner), slots));
    }
    if obj.hasattr("calc_energy_forces")? && obj.getattr("calc_energy_forces")?.is_callable() {
        let error: ErrSlot = Arc::default();
        return Ok((
            Member::plain(SubclassPotential {
                obj: obj.clone().unbind(),
                error: Arc::clone(&error),
            }),
            vec![error],
        ));
    }
    Err(PyTypeError::new_err(
        "expected a potential with callable calc_energy_forces (LJCut, Potentials, or duck-typed)",
    ))
}
