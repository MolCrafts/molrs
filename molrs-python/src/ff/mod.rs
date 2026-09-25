//! Python wrappers for MMFF force-field typification and compiled potentials.
//!
//! The workflow is:
//!
//! 1. Create a typifier — [`PyMMFF94Typifier`] (MMFF94) or [`PyMMFF94STypifier`]
//!    (MMFF94s, the "static" variant). Both load their embedded parameter set at
//!    construction; the variant is the class, never a flag.
//! 2. Call `typify` to assign atom types + bonded parameters, producing a typed
//!    [`PyAtomistic`] (materialize it with `to_frame()` for a [`PyFrame`]).
//! 3. Build the neighbour list (`molrs.intramolecular_pairs`) and compile with
//!    `PotentialCompiler(typifier.forcefield()).compile(frame)` — the same route every other
//!    force field in molrs uses.
//! 4. Use [`PyPotentials::eval`] to evaluate energy and forces on flat
//!    coordinate arrays.
//!
//! There is deliberately **no** one-step `build(mol)` and no free
//! `build_mmff_potentials(mol)`. Both existed, sat adjacent in the same namespace
//! with nothing to tell them apart, and one of them silently omitted the entire
//! electrostatic term (150 kcal/mol on caffeine) because no `ForceField` ever
//! defined `pair/mmff_ele`. A typifier's contract is `typify`; compiling
//! potentials is `PotentialCompiler.compile`.
//!
//! The antechamber-derived bindings live in their own modules rather than here:
//! [`atd`] (the ATD atom typifier, one engine over seven `ATOMTYPE_*.DEF` tables)
//! and [`charge`] (the three charge models). This file is already large, and they
//! are self-contained.
//!
//! # References
//!
//! - Halgren, T.A. (1996). J. Comput. Chem. 17, 490-519. (MMFF94 force field)
//! - Halgren, T.A. (1999). J. Comput. Chem. 20, 720-729. (MMFF94s option)

pub mod atd;
pub mod charge;

use std::collections::HashMap;
use std::fs;

use pyo3::exceptions::{
    PyKeyError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList, PyMapping, PyString, PySuper, PyTuple, PyType};

use molrs::ff::ForceField;
use molrs::ff::potential::{Member, PotentialCompiler, Potentials, extract_coords, write_coords};
use molrs::ff::typifier::mmff::{MMFF94STypifier, MMFF94Typifier};
use molrs::ff::typifier::opls::OPLSAATypifier;
use molrs::ff::typifier::{Annotation, Match, Typifier, Typing};
use molrs::optimize::{LBFGS, OptReport};
use molrs_ffi::ForceFieldRef;

use crate::core::store::block::PyBlock;
use crate::core::store::frame::PyFrame;
use crate::core::system::molgraph::{PyAtomistic, py_to_prop};
use crate::helpers::{NpF, py_value_err};

use ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArrayDyn, ToPyArray};

/// Where a [`PyTypifier`]'s typing state lives.
enum TypifierState {
    /// A native typifier class: the Rust base owns the matcher and the output.
    Native(Typing<Box<dyn Typifier + Send + Sync>>),
    /// A Python subclass: the matcher is its ``match`` method and this base
    /// holds the output, unset until first seeded (see [`PyTypifier::seed`]).
    Python(Option<ForceField>),
}

fn unseeded_output() -> PyErr {
    PyRuntimeError::new_err("typifier output accessed before it was seeded")
}

/// The base of every graph typifier: one ``match`` hook plus the output force
/// field its typing accumulates.
///
/// Exposed to Python as ``molrs.ff.typifier.Typifier`` and subclassable. A
/// subclass implements :meth:`match` (and optionally :meth:`library`) and
/// nothing else; :meth:`typify` is the one execution path and the only writer
/// of :meth:`forcefield`. Defining ``typify`` on a subclass raises
/// ``TypeError`` at class creation.
///
/// The native typifier classes (``MMFF94Typifier``, ``MMFF94STypifier``,
/// ``OPLSAATypifier``, ``AtdTypifier``) extend this base and only construct:
/// their ``match`` runs the Rust matcher and their ``typify`` the Rust
/// ``Typing::typify``.
#[pyclass(module = "molrs.ff.typifier", name = "Typifier", subclass)]
pub struct PyTypifier {
    state: TypifierState,
}

impl PyTypifier {
    /// The base of a native typifier class: `typifier` wrapped in [`Typing`],
    /// whose output starts as `typifier.library().empty_like()`.
    pub(crate) fn native(typifier: impl Typifier + Send + Sync + 'static) -> Self {
        Self {
            state: TypifierState::Native(Typing::new(Box::new(typifier))),
        }
    }

    /// The library's name, for the native classes' `__repr__`; empty for a
    /// Python subclass, whose library is whatever its `library()` returns.
    pub(crate) fn library_name(&self) -> &str {
        match &self.state {
            TypifierState::Native(typing) => &typing.library().name,
            TypifierState::Python(_) => "",
        }
    }

    /// Seed a Python subclass's output on first access, exactly as
    /// [`Typing::new`] does: `self.library().empty_like()` — the library's
    /// name and declared units and special_bonds. A subclass without a
    /// `library()` (it raises `NotImplementedError`) gets an empty force field
    /// named after its class. A no-op once seeded, and for a native typifier.
    fn seed(slf: &Bound<'_, Self>) -> PyResult<()> {
        if !matches!(slf.borrow().state, TypifierState::Python(None)) {
            return Ok(());
        }
        let py = slf.py();
        // `library()` is dispatched through Python (a subclass overrides it),
        // so no borrow of `slf` is held across the call.
        let seed = match slf.call_method0(intern!(py, "library")) {
            Ok(library) => {
                let library = library.cast_into::<PyForceField>().map_err(|err| {
                    PyTypeError::new_err(format!("library() must return a ForceField: {err}"))
                })?;
                library.borrow().inner.empty_like()
            }
            Err(err) if err.is_instance_of::<PyNotImplementedError>(py) => {
                ForceField::new(&slf.get_type().name()?.to_string())
            }
            Err(err) => return Err(err),
        };
        if let TypifierState::Python(output) = &mut slf.borrow_mut().state {
            // `library()` may itself have reached `forcefield()` and seeded.
            output.get_or_insert(seed);
        }
        Ok(())
    }
}

#[pymethods]
impl PyTypifier {
    /// A Python-subclass base with an unset output.
    ///
    /// Accepts and ignores any arguments, as ``object.__new__`` does, so a
    /// subclass's own ``__init__(...)`` signature is left to the subclass.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self {
            state: TypifierState::Python(None),
        }
    }

    /// Reject a subclass that defines ``typify`` in its own body.
    ///
    /// ``typify`` is the only writer of :meth:`forcefield`; an override would
    /// silently bypass the output. ``typing.final`` is only a static check.
    #[classmethod]
    #[pyo3(signature = (**kwargs))]
    fn __init_subclass__(
        cls: &Bound<'_, PyType>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let py = cls.py();
        if cls
            .getattr(intern!(py, "__dict__"))?
            .contains(intern!(py, "typify"))?
        {
            return Err(PyTypeError::new_err(format!(
                "{} defines typify; a Typifier subclass implements match (and optionally \
                 library) only — typify is the base's and the only writer of forcefield()",
                cls.name()?
            )));
        }
        PySuper::new(&py.get_type::<Self>(), cls.as_any())?.call_method(
            intern!(py, "__init_subclass__"),
            (),
            kwargs,
        )?;
        Ok(())
    }

    /// Match ``graph`` and return what it assigns, as a :class:`Match`.
    ///
    /// The one hook a subclass implements. ``match`` may write intermediate
    /// results (generated topology, perceived bond types) onto the graph it is
    /// given; :meth:`typify` always gives it a private copy. On a native
    /// typifier this runs the Rust matcher on ``graph``.
    ///
    /// Raises
    /// ------
    /// NotImplementedError
    ///     On the base, when a subclass does not implement ``match``.
    /// ValueError
    ///     If a native matcher cannot match the graph.
    #[pyo3(name = "match")]
    fn r#match(&self, graph: &Bound<'_, PyAny>) -> PyResult<PyMatch> {
        match &self.state {
            TypifierState::Native(typing) => {
                let graph = graph.cast::<PyAtomistic>()?;
                let inner = typing
                    .typifier()
                    .r#match(graph.borrow_mut().core_mut())
                    .map_err(PyValueError::new_err)?;
                Ok(PyMatch { inner })
            }
            TypifierState::Python(_) => Err(PyNotImplementedError::new_err(
                "Typifier.match must be implemented by a concrete typifier",
            )),
        }
    }

    /// Type ``mol``: do not override; the only writer of :meth:`forcefield`.
    ///
    /// Copies ``mol`` (``mol.copy()``), calls :meth:`match` on the copy and
    /// writes the returned :class:`Match` onto the copy and the output force
    /// field — stamping every annotation and defining every type. ``mol`` is
    /// never touched.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     The typed copy.
    ///
    /// Raises
    /// ------
    /// NotImplementedError
    ///     If the typifier has no ``match``.
    /// TypeError
    ///     If ``match`` returns something other than a :class:`Match`.
    /// ValueError
    ///     If the match does not fit the graph or contradicts a definition the
    ///     output already holds. The output is then unchanged.
    fn typify(slf: &Bound<'_, Self>, mol: &Bound<'_, PyAtomistic>) -> PyResult<Py<PyAtomistic>> {
        let py = slf.py();
        let native = match &mut slf.borrow_mut().state {
            TypifierState::Native(typing) => Some(
                typing
                    .typify(mol.borrow().core())
                    .map_err(PyValueError::new_err)?,
            ),
            TypifierState::Python(_) => None,
        };
        if let Some(typed) = native {
            return PyAtomistic::from_core(py, typed);
        }

        Self::seed(slf)?;
        let typed = mol
            .call_method0(intern!(py, "copy"))?
            .cast_into::<PyAtomistic>()?;
        let returned = slf.call_method1(intern!(py, "match"), (&typed,))?;
        let matched = returned
            .cast::<PyMatch>()
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "match must return a Match, got {}",
                    returned.get_type()
                ))
            })?
            .get()
            .inner
            .clone();
        match &mut slf.borrow_mut().state {
            TypifierState::Python(Some(output)) => matched
                .write_onto(typed.borrow_mut().core_mut(), output)
                .map_err(PyValueError::new_err)?,
            TypifierState::Native(_) | TypifierState::Python(None) => {
                return Err(unseeded_output());
            }
        }
        Ok(typed.unbind())
    }

    /// The accumulated output — exactly the definitions :meth:`typify` has
    /// assigned — as an independent copy.
    ///
    /// Edits to the returned force field do not reach the typifier;
    /// :meth:`typify` is the only writer. Before the first ``typify`` this is
    /// the seeded empty output (see :meth:`library`).
    fn forcefield(slf: &Bound<'_, Self>) -> PyResult<Py<PyForceField>> {
        Self::seed(slf)?;
        let output = match &slf.borrow().state {
            TypifierState::Native(typing) => typing.forcefield().clone(),
            TypifierState::Python(Some(output)) => output.clone(),
            TypifierState::Python(None) => return Err(unseeded_output()),
        };
        PyForceField::from_core(slf.py(), output)
    }

    /// The force field this typifier matches against, as an independent copy.
    ///
    /// The output starts as its empty likeness: the library's name and
    /// declared units and special_bonds, no styles or types. A Python subclass
    /// may override it; one that does not has no library (this raises
    /// ``NotImplementedError``), and its output starts as an empty force field
    /// named after the class, with nothing declared.
    fn library(&self, py: Python<'_>) -> PyResult<Py<PyForceField>> {
        match &self.state {
            TypifierState::Native(typing) => PyForceField::from_core(py, typing.library().clone()),
            TypifierState::Python(_) => Err(PyNotImplementedError::new_err(
                "Typifier.library is not implemented by this typifier",
            )),
        }
    }
}

/// What a typifier's ``match`` assigns to one graph, exposed to Python as
/// ``molrs.ff.typifier.Match`` (the Rust `Match`).
///
/// Parameters
/// ----------
/// nodes : sequence of mapping
///     One mapping of ``key -> annotation`` per node, positional against
///     ``graph.nodes``. An empty mapping gives that node nothing.
/// links : mapping, optional
///     Relation class (``Bond``, ``Angle``, ``Dihedral``, ``Improper``) to a
///     sequence of mappings, positional against
///     ``graph.links.exact_bucket(cls)`` — the kind's own rows, so an improper
///     never shifts a dihedral position. The class's ``_kind`` selects the
///     kind; an unknown kind raises ``TypeError``.
/// styles : sequence of (category, style, params), optional
///     Styles to declare, in order.
/// pairs : sequence of (style, name, endpoints, params), optional
///     Pair rows to define.
///
/// An annotation is a ``str``, ``bool``, ``int`` or ``float`` (stamped, defines
/// nothing), or a type: ``(style, name, params)`` or
/// ``(style, name, endpoints, params)``, which stamps ``name`` and every param
/// and defines the type under the style. Param values are numbers or strings.
#[pyclass(module = "molrs.ff.typifier", name = "Match", frozen)]
pub struct PyMatch {
    inner: Match,
}

impl PyMatch {
    /// One positional vector: a sequence of `key -> annotation` mappings.
    fn rows(rows: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<(String, Annotation)>>> {
        rows.try_iter()?
            .map(|row| {
                let row = row?;
                let row = row.cast::<PyMapping>()?;
                row.items()?
                    .iter()
                    .map(|item| {
                        let (key, value): (String, Bound<'_, PyAny>) = item.extract()?;
                        let annotation = Self::annotation(&value).map_err(|err| {
                            prefix_err(value.py(), err, &format!("annotation '{key}'"))
                        })?;
                        Ok((key, annotation))
                    })
                    .collect()
            })
            .collect()
    }

    /// A tuple is a type; anything else a stamped value (`bool` before `int`).
    fn annotation(value: &Bound<'_, PyAny>) -> PyResult<Annotation> {
        let Ok(tuple) = value.cast::<PyTuple>() else {
            return py_to_prop(value).map(Annotation::Value);
        };
        let (style, name, endpoints, params) = match tuple.len() {
            3 => {
                let (style, name, params): (String, String, Bound<'_, PyDict>) = tuple.extract()?;
                (style, name, None, params)
            }
            4 => {
                let (style, name, endpoints, params): (
                    String,
                    String,
                    Vec<String>,
                    Bound<'_, PyDict>,
                ) = tuple.extract()?;
                (style, name, Some(endpoints), params)
            }
            n => {
                return Err(PyTypeError::new_err(format!(
                    "a type annotation is (style, name, params) or (style, name, endpoints, \
                     params); got a {n}-tuple"
                )));
            }
        };
        Ok(Annotation::Type {
            style,
            name,
            endpoints,
            params: params_from_dict(Some(&params))?,
        })
    }
}

/// `err` with `context` prepended to its message, same exception type.
fn prefix_err(py: Python<'_>, err: PyErr, context: &str) -> PyErr {
    PyErr::from_type(err.get_type(py), format!("{context}: {}", err.value(py)))
}

#[pymethods]
impl PyMatch {
    #[new]
    #[pyo3(
        signature = (nodes, links = None, *, styles = Vec::new(), pairs = Vec::new()),
        text_signature = "(nodes, links=None, *, styles=(), pairs=())"
    )]
    fn new(
        nodes: &Bound<'_, PyAny>,
        links: Option<&Bound<'_, PyAny>>,
        styles: Vec<(String, String, Bound<'_, PyDict>)>,
        pairs: Vec<(String, String, Vec<String>, Bound<'_, PyDict>)>,
    ) -> PyResult<Self> {
        let mut inner = Match {
            nodes: Self::rows(nodes)?,
            ..Match::default()
        };
        if let Some(links) = links {
            let mut seen: Vec<String> = Vec::new();
            for item in links.cast::<PyMapping>()?.items()?.iter() {
                let (cls, rows): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item.extract()?;
                let kind: String = cls
                    .getattr(intern!(cls.py(), "_kind"))
                    .and_then(|kind| kind.extract())
                    .map_err(|_| {
                        PyTypeError::new_err(format!(
                            "links keys must be relation classes declaring _kind, got {cls}"
                        ))
                    })?;
                let slot = match kind.as_str() {
                    "bonds" => &mut inner.bonds,
                    "angles" => &mut inner.angles,
                    "dihedrals" => &mut inner.dihedrals,
                    "impropers" => &mut inner.impropers,
                    other => {
                        return Err(PyTypeError::new_err(format!(
                            "{cls}: a Match carries bonds, angles, dihedrals and impropers, \
                             not relation kind '{other}'"
                        )));
                    }
                };
                if seen.contains(&kind) {
                    return Err(PyValueError::new_err(format!(
                        "links name relation kind '{kind}' more than once"
                    )));
                }
                *slot = Self::rows(&rows).map_err(|err| prefix_err(rows.py(), err, &kind))?;
                seen.push(kind);
            }
        }
        for (category, name, params) in styles {
            let params = params_from_dict(Some(&params))?;
            inner.styles.push((category, name, params));
        }
        for (style, name, endpoints, params) in pairs {
            let params = params_from_dict(Some(&params))?;
            inner.pairs.push((style, name, endpoints, params));
        }
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        let m = &self.inner;
        format!(
            "Match(nodes={}, bonds={}, angles={}, dihedrals={}, impropers={}, styles={}, \
             pairs={})",
            m.nodes.len(),
            m.bonds.len(),
            m.angles.len(),
            m.dihedrals.len(),
            m.impropers.len(),
            m.styles.len(),
            m.pairs.len()
        )
    }
}

/// Outcome of a geometry optimization, exposed to Python as `molrs.OptReport`.
#[pyclass(module = "molrs.optimize", name = "OptReport")]
pub struct PyOptReport {
    inner: OptReport,
}

#[pymethods]
impl PyOptReport {
    /// Whether ``fmax`` convergence was reached within ``max_steps``.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    /// Number of outer L-BFGS iterations performed.
    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps
    }

    /// Potential energy at the returned geometry (kcal/mol).
    #[getter]
    fn final_energy(&self) -> f64 {
        self.inner.final_energy
    }

    /// Maximum per-atom force magnitude at the returned geometry
    /// (kcal/mol/angstrom).
    #[getter]
    fn final_fmax(&self) -> f64 {
        self.inner.final_fmax
    }

    fn __repr__(&self) -> String {
        format!(
            "OptReport(converged={}, n_steps={}, final_energy={:.6}, final_fmax={:.6})",
            if self.inner.converged {
                "True"
            } else {
                "False"
            },
            self.inner.n_steps,
            self.inner.final_energy,
            self.inner.final_fmax
        )
    }
}

impl From<OptReport> for PyOptReport {
    fn from(inner: OptReport) -> Self {
        Self { inner }
    }
}

/// The kernels of a neighbour-driven force evaluation, each with the
/// special-bonds weights that scale it.
///
/// Opaque on purpose: what a caller does with this is hand it to an
/// integrator. Taking it apart in Python would mean re-deciding which member
/// is which and how its close neighbours are scaled — the two things
/// :meth:`PotentialCompiler.compile_typed` exists to decide once.
#[pyclass(name = "TypedPotentials", module = "molrs.ff")]
pub struct PyTypedPotentials {
    /// Taken by the integrator that consumes it; `None` afterwards.
    pub(crate) members: Option<Vec<(Member, molrs::md::SpecialWeights)>>,
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
/// Exposed to Python as `molrs.ff.Potentials`.
///
/// Operates on flat coordinate arrays in the layout
/// ``[x0, y0, z0, x1, y1, z1, ...]`` (length 3N).
///
/// Examples
/// --------
/// >>> typifier = MMFF94Typifier()
/// >>> frame = typifier.typify(mol).to_frame()
/// >>> frame["pairs"] = molrs.intramolecular_pairs(frame)
/// >>> potentials = molrs.ff.PotentialCompiler(typifier.forcefield()).compile(frame)
/// >>> energy, forces = potentials.eval(coords)
#[pyclass(module = "molrs.ff", name = "Potentials")]
pub struct PyPotentials {
    inner: PotBacking,
    /// Error slots of every Python-callable member (see `crate::md::ErrSlot`);
    /// checked after each evaluation so a callable's exception re-raises.
    err_slots: Vec<crate::md::ErrSlot>,
}

/// A [`PyPotentials`] is either already compiled against a molecule's topology
/// (the MMFF / pre-bound path), *deferred* (it holds the force field and binds
/// the topology lazily from the `Frame` passed to
/// ``calc_energy``/``calc_forces`` — what ``PotentialCompiler.defer()``
/// returns, matching the molpy evaluation model), or *moved*: the
/// Rust `Potentials` has been moved into an MD integrator or another
/// collection.
enum PotBacking {
    Compiled(Potentials),
    Deferred(ForceField),
    Moved,
}

fn potentials_moved_err() -> PyErr {
    PyValueError::new_err(
        "this Potentials has been moved into an integrator or another \
         Potentials; rebuild with PotentialCompiler.compile(frame)",
    )
}

impl PotBacking {
    /// The compiled potentials, or an error if this set is still deferred and
    /// no `Frame` has been supplied to bind its topology (or already moved).
    fn compiled(&self) -> PyResult<&Potentials> {
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

/// Force-field definition metadata exposed to Python as `molrs.ForceField`.
#[pyclass(module = "molrs._lib", name = "ForceField", subclass)]
pub struct PyForceField {
    pub(crate) inner: ForceField,
}

/// CL&Pol fragment scaling data backed by the native force-field layer.
#[pyclass(
    module = "molrs",
    name = "FragmentScaling",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFragmentScaling {
    name: String,
    q: f64,
    mu: f64,
    alpha: f64,
    polarizable: bool,
}

impl From<PyFragmentScaling> for molrs::ff::FragmentScaling {
    fn from(value: PyFragmentScaling) -> Self {
        Self {
            name: value.name,
            q: value.q,
            mu: value.mu,
            alpha: value.alpha,
            polarizable: value.polarizable,
        }
    }
}

impl From<molrs::ff::FragmentScaling> for PyFragmentScaling {
    fn from(value: molrs::ff::FragmentScaling) -> Self {
        Self {
            name: value.name,
            q: value.q,
            mu: value.mu,
            alpha: value.alpha,
            polarizable: value.polarizable,
        }
    }
}

#[pymethods]
impl PyFragmentScaling {
    #[new]
    #[pyo3(signature = (name, q, mu, alpha, polarizable=false))]
    fn new(name: String, q: f64, mu: f64, alpha: f64, polarizable: bool) -> Self {
        Self {
            name,
            q,
            mu,
            alpha,
            polarizable,
        }
    }

    fn __repr__(&self) -> String {
        format!("FragmentScaling(name='{}')", self.name)
    }
}

/// Native SAPT epsilon-scaling factor.
#[pyfunction(name = "compute_k_ij")]
pub fn compute_k_ij_py(
    fr_i: PyRef<'_, PyFragmentScaling>,
    fr_j: PyRef<'_, PyFragmentScaling>,
    r: f64,
) -> PyResult<f64> {
    molrs::ff::compute_k_ij(&fr_i.clone().into(), &fr_j.clone().into(), r).map_err(py_value_err)
}

/// Return the compiled-in CL&Pol fragment table.
#[pyfunction(name = "fragment_scaling_data")]
pub fn fragment_scaling_data_py(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let result = PyDict::new(py);
    for (name, scaling) in molrs::ff::scale_lj::builtin_fragment_scaling() {
        result.set_item(name, Py::new(py, PyFragmentScaling::from(scaling))?)?;
    }
    Ok(result)
}

/// Clone and scale LJ parameters using native COM and force-field transforms.
#[pyfunction(name = "scale_lj")]
#[pyo3(signature = (ff, fragments, frag_data=None, scale_sigma=false))]
pub fn scale_lj_py(
    py: Python<'_>,
    ff: &Bound<'_, PyForceField>,
    fragments: &Bound<'_, PyDict>,
    frag_data: Option<&Bound<'_, PyDict>>,
    scale_sigma: bool,
) -> PyResult<Py<PyForceField>> {
    let mut native_fragments = Vec::with_capacity(fragments.len());
    for (label, value) in fragments.iter() {
        let name = label.extract::<String>()?;
        let (atom_types, coords, masses) =
            value.extract::<(Vec<String>, Vec<[f64; 3]>, Vec<f64>)>()?;
        native_fragments.push(molrs::ff::FragmentAtoms {
            name,
            atom_types,
            coords,
            masses,
        });
    }

    let mut scaling = HashMap::new();
    if let Some(data) = frag_data {
        for (label, value) in data.iter() {
            let item = value.extract::<PyRef<'_, PyFragmentScaling>>()?;
            scaling.insert(label.extract::<String>()?, item.clone().into());
        }
    } else {
        scaling = molrs::ff::scale_lj::builtin_fragment_scaling();
    }

    let inner = molrs::ff::scale_lj(&ff.borrow().inner, &native_fragments, &scaling, scale_sigma)
        .map_err(|error| match error {
        molrs::ff::ScaleLjError::MissingFragment(name) => {
            PyKeyError::new_err(format!("no scaling data for fragment '{name}'"))
        }
        other => py_value_err(other),
    })?;
    PyForceField::from_core(py, inner)
}

/// Convert an optional Python ``dict[str, float | str]`` of parameters into
/// [`Params`](molrs::ff::forcefield::Params). A ``str`` value goes to the string
/// side, a number to the numeric side; anything else raises ``TypeError``. A
/// missing dict yields no params.
fn params_from_dict(params: Option<&Bound<'_, PyDict>>) -> PyResult<molrs::ff::forcefield::Params> {
    let mut out = molrs::ff::forcefield::Params::new();
    let Some(d) = params else {
        return Ok(out);
    };
    for (k, v) in d.iter() {
        let key = k.extract::<String>()?;
        if let Ok(text) = v.cast::<PyString>() {
            out.set_str(&key, text.to_str()?);
        } else if let Ok(number) = v.extract::<f64>() {
            out.set(&key, number);
        } else {
            return Err(PyTypeError::new_err(format!(
                "param '{key}' must be a number or a str, got {}",
                v.get_type().name()?
            )));
        }
    }
    Ok(out)
}

impl PyPotentials {
    /// Evaluate energy + forces against either a [`PyFrame`] (binds topology and
    /// reads coordinates from the frame's ``atoms`` block — the molpy model) or a
    /// flat coordinate array (requires already-compiled potentials).
    ///
    /// A Python-callable member's exception is re-raised afterwards (see
    /// `crate::md::ErrSlot`).
    fn eval_any(&self, arg: &Bound<'_, PyAny>) -> PyResult<(f64, Vec<NpF>)> {
        let ef = if let Ok(frame) = arg.extract::<PyRef<'_, PyFrame>>() {
            let core = frame.clone_core_frame()?;
            let coords = extract_coords(&core).map_err(PyValueError::new_err)?;
            match &self.inner {
                PotBacking::Compiled(p) => p.calc_energy_forces(&coords),
                PotBacking::Deferred(ff) => PotentialCompiler::new(ff)
                    .compile(&core)
                    .map_err(PyValueError::new_err)?
                    .calc_energy_forces(&coords),
                PotBacking::Moved => return Err(potentials_moved_err()),
            }
        } else {
            let arr = arg.extract::<numpy::PyReadonlyArray1<'_, NpF>>()?;
            let slice = arr.as_slice()?;
            self.inner.compiled()?.calc_energy_forces(slice)
        };
        crate::md::take_err(&self.err_slots)?;
        Ok(ef)
    }

    /// Move the compiled Rust `Potentials` (and the error slots of its
    /// Python-callable members) out, leaving this object in the moved state —
    /// the MD integrators and `Potentials.push` consume through here.
    pub(crate) fn take_compiled(&mut self) -> PyResult<(Potentials, Vec<crate::md::ErrSlot>)> {
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
        let (member, mut slots) = crate::md::take_potential(potential)?;
        self.inner.compiled_mut()?.push(member);
        self.err_slots.append(&mut slots);
        Ok(())
    }

    /// Returns ``(energy, forces)``, forces shape ``(N, 3)``.
    fn calc_energy_forces<'py>(
        &self,
        py: Python<'py>,
        arg: &Bound<'_, PyAny>,
    ) -> PyResult<(f64, Bound<'py, PyArray2<NpF>>)> {
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
    ) -> PyResult<Bound<'py, PyArray2<NpF>>> {
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

/// L-BFGS geometry optimizer, exposed as `molrs.LBFGS`.
///
/// Construct with potentials + knobs on ``new``, then ``run`` a :class:`Frame`
/// (primary) or a coordinate array (single / batch by rank).
///
/// Examples
/// --------
/// >>> pots = molrs.ff.PotentialCompiler(molrs.MMFF94Typifier().forcefield()).compile(frame)
/// >>> opt = molrs.LBFGS(pots, fmax=0.05, max_steps=500)
/// >>> frame, report = opt.run(frame)
/// >>> coords, report = opt.run(coords)         # (N, 3)
#[pyclass(module = "molrs.optimize", name = "LBFGS")]
pub struct PyLBFGS {
    potentials: Py<PyPotentials>,
    fmax: f64,
    max_steps: usize,
    max_step: f64,
    memory: usize,
}

#[pymethods]
impl PyLBFGS {
    #[new]
    #[pyo3(signature = (potentials, *, fmax = 0.05, max_steps = 500, max_step = 0.2, memory = 8))]
    fn new(
        potentials: Py<PyPotentials>,
        fmax: f64,
        max_steps: usize,
        max_step: f64,
        memory: usize,
    ) -> Self {
        Self {
            potentials,
            fmax,
            max_steps,
            max_step,
            memory,
        }
    }

    /// Relax a :class:`Frame` or coordinates by L-BFGS.
    ///
    /// * ``Frame`` → ``(Frame, OptReport)`` (frame coordinates updated; a new
    ///   Python frame object is returned with the minimized coords).
    /// * ``(N, 3)`` / ``(3N,)`` → ``((N, 3) array, OptReport)``
    /// * ``(B, N, 3)`` → ``((B, N, 3) array, list[OptReport])``
    fn run<'py>(&self, py: Python<'py>, arg: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        // Frame path (primary).
        if let Ok(frame) = arg.extract::<PyRef<'_, PyFrame>>() {
            let mut core = frame.clone_core_frame()?;
            let pots = self.potentials.borrow(py);
            // Compile against this frame if deferred, then minimize with free mask.
            let compiled;
            let pot: &dyn molrs::ff::potential::Potential = match &pots.inner {
                PotBacking::Compiled(p) => p,
                PotBacking::Deferred(ff) => {
                    compiled = PotentialCompiler::new(ff)
                        .compile(&core)
                        .map_err(pyo3::exceptions::PyValueError::new_err)?;
                    &compiled
                }
                PotBacking::Moved => return Err(potentials_moved_err()),
            };
            // Borrowed one-shot on flat coords extracted from frame, then write back.
            let mut flat =
                extract_coords(&core).map_err(pyo3::exceptions::PyValueError::new_err)?;
            let report = LBFGS::minimize(
                pot,
                &mut flat,
                self.fmax,
                self.max_steps,
                self.max_step,
                self.memory,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            write_coords(&mut core, &flat).map_err(pyo3::exceptions::PyValueError::new_err)?;
            let out_frame = PyFrame::from_core_frame(core)?;
            return Ok((out_frame, PyOptReport::from(report))
                .into_pyobject(py)?
                .into_any());
        }

        let pots = self.potentials.borrow(py);
        let pot = pots.inner.compiled()?;
        let readonly = arg.extract::<PyReadonlyArrayDyn<'_, NpF>>()?;
        let arr = readonly.as_array();
        let shape = arr.shape();
        match shape.len() {
            1 | 2 => {
                let mut flat: Vec<NpF> = arr.iter().copied().collect();
                let n_elem = flat.len();
                if !n_elem.is_multiple_of(3) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "coords has {n_elem} elements, not a multiple of 3 (expected (N, 3) or (3N,))"
                    )));
                }
                let report = LBFGS::minimize(
                    pot,
                    &mut flat,
                    self.fmax,
                    self.max_steps,
                    self.max_step,
                    self.memory,
                )
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
                let out: Bound<'py, PyArray2<NpF>> = Array2::from_shape_vec((n_elem / 3, 3), flat)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    .to_pyarray(py);
                Ok((out, PyOptReport::from(report))
                    .into_pyobject(py)?
                    .into_any())
            }
            3 => {
                if shape[2] != 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "batch coords must be (B, N, 3); trailing axis is {} not 3",
                        shape[2]
                    )));
                }
                let (b, n) = (shape[0], shape[1]);
                let expected = pot.n_atoms();
                if expected != 0 && n != expected {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "structure atom count N={n} does not match this Potentials' atom count {expected}"
                    )));
                }
                let mut flat: Vec<NpF> = arr.iter().copied().collect();
                let reports = LBFGS::minimize_batch(
                    pot,
                    &mut flat,
                    n,
                    b,
                    self.fmax,
                    self.max_steps,
                    self.max_step,
                    self.memory,
                )
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
                let out: Bound<'py, PyArray3<NpF>> = Array3::from_shape_vec((b, n, 3), flat)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    .to_pyarray(py);
                let reports: Vec<PyOptReport> =
                    reports.into_iter().map(PyOptReport::from).collect();
                Ok((out, reports).into_pyobject(py)?.into_any())
            }
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "arg must be Frame, 1-D (3N,), 2-D (N, 3), or 3-D (B, N, 3); got {other}-D array"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LBFGS(fmax={}, max_steps={}, max_step={}, memory={})",
            self.fmax, self.max_steps, self.max_step, self.memory
        )
    }
}

/// Bind one MMFF front door to Python.
///
/// MMFF94 and MMFF94s are the same engine over two parameter sets, and molrs
/// exposes them as two **named types** rather than one type with a variant flag —
/// so the binder mirrors that shape exactly: two `#[pyclass]`es, each wrapping its
/// own core typifier, generated from one forwarding body so they cannot drift.
macro_rules! py_mmff_front_door {
    (
        $(#[$doc:meta])*
        $py_ty:ident, $core:ty, $name:literal
    ) => {
        $(#[$doc])*
        #[pyclass(module = "molrs", name = $name, extends = PyTypifier)]
        pub struct $py_ty;

        #[pymethods]
        impl $py_ty {
            /// Create the typifier with its embedded parameter tables.
            ///
            /// Never fails: the parameter set is compiled into the extension module.
            #[new]
            fn new() -> (Self, PyTypifier) {
                (Self, PyTypifier::native(<$core>::new()))
            }

            fn __repr__(slf: PyRef<'_, Self>) -> String {
                format!("{}(forcefield='{}')", $name, slf.as_super().library_name())
            }
        }
    };
}

py_mmff_front_door! {
    /// MMFF94 atom-type assigner.
    ///
    /// Exposed to Python as `molrs.ff.MMFF94Typifier`.
    ///
    /// Loads the embedded MMFF94 parameter tables at construction time. Use
    /// :meth:`typify` to label a molecular graph (atom types, partial charges, and
    /// the per-instance force constants the kernels read: ``koop`` (md*A*rad^-2)
    /// on the improper rows, ``v1``/``v2``/``v3`` on the dihedral rows), then
    /// compile it through :meth:`forcefield` — the definitions typing assigned —
    /// the standard route, shared with every other force field in molrs.
    /// :meth:`typify` raises ``ValueError`` when atom types cannot be determined
    /// (e.g. unsupported elements).
    ///
    /// See :class:`MMFF94STypifier` for the "static" variant used in energy
    /// minimization.
    ///
    /// # References
    ///
    /// - Halgren, T.A. (1996). J. Comput. Chem. 17, 490-519.
    ///
    /// Examples
    /// --------
    /// >>> typifier = MMFF94Typifier()
    /// >>> frame = typifier.typify(mol).to_frame()          # labels + charges
    /// >>> frame["pairs"] = molrs.intramolecular_pairs(frame)
    /// >>> pots = molrs.ff.PotentialCompiler(typifier.forcefield()).compile(frame)
    PyMMFF94Typifier, MMFF94Typifier, "MMFF94Typifier"
}

py_mmff_front_door! {
    /// MMFF94s ("static") atom-type assigner and potential builder.
    ///
    /// Exposed to Python as `molrs.ff.MMFF94STypifier`.
    ///
    /// Identical to :class:`MMFF94Typifier` except on delocalised trivalent
    /// nitrogen (MMFF numeric types 10 ``NC=O`` and 40 ``NC=C``), where MMFF94s
    /// re-parameterises 11 out-of-plane rows and 42 torsion rows so the nitrogen
    /// minimizes to a **planar** geometry — the one seen in crystal structures.
    ///
    /// The mechanism is the out-of-plane force constant ``koop`` (md*A*rad^-2) that
    /// :meth:`typify` bakes onto the improper rows. The kernel evaluates
    /// ``E_oop = 0.5 * 143.9325 * koop * chi**2`` with ``chi`` the Wilson
    /// out-of-plane angle in radians, so ``koop > 0`` makes the planar centre an
    /// energy minimum. MMFF94s sets it to ``+0.015`` (type 10) / ``+0.030``
    /// (type 40); MMFF94's values on those rows run from ``-0.033`` to ``+0.004``.
    ///
    /// All 95 atom types, and every bond / angle / stretch-bend / vdW / charge
    /// parameter, are shared with MMFF94 — so a molecule with no such nitrogen gets
    /// bit-for-bit the same answer from both classes.
    ///
    /// # References
    ///
    /// - Halgren, T.A. (1999). J. Comput. Chem. 20, 720-729. (MMFF94s)
    ///
    /// Examples
    /// --------
    /// >>> typifier = MMFF94STypifier()
    /// >>> typifier.forcefield().name
    /// 'MMFF94s'
    PyMMFF94STypifier, MMFF94STypifier, "MMFF94STypifier"
}

fn oplsaa_source_xml(source: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    let Some(source) = source else {
        return Ok(None);
    };
    let raw = match source.extract::<String>() {
        Ok(value) => value,
        Err(_) => source.call_method0("__fspath__")?.extract::<String>()?,
    };
    if raw.trim_start().starts_with('<') {
        return Ok(Some(raw));
    }
    fs::read_to_string(&raw).map(Some).map_err(|err| {
        PyValueError::new_err(format!("failed to read OPLS-AA XML source {raw:?}: {err}"))
    })
}

/// OPLS-AA atom-type assigner and potential builder.
///
/// Exposed to Python as `molrs.ff.OPLSAATypifier`. It loads the embedded canonical
/// OPLS-AA parameter set by default, or reads one XML source at construction.
/// :meth:`typify` returns a typed :class:`Atomistic` (``ValueError`` when atom
/// typing fails); :meth:`forcefield` holds the definitions it assigned.
///
/// Parameters
/// ----------
/// source : str or path-like, optional
///     OPLS-AA XML text or a path to an XML file. ``None`` uses the embedded
///     canonical OPLS-AA table.
/// strict : bool, default True
///     When True, a bonded term with no force-field match is an error. When
///     False, such terms are skipped (left unparametrized).
///
/// Examples
/// --------
/// >>> typifier = OPLSAATypifier()
/// >>> typed = typifier.typify(mol)        # typed Atomistic
/// >>> # compose: typify → to_frame → intramolecular_pairs → PotentialCompiler(forcefield()).compile
#[pyclass(module = "molrs.ff.typifier", name = "OPLSAATypifier", extends = PyTypifier)]
pub struct PyOPLSAATypifier;

#[pymethods]
impl PyOPLSAATypifier {
    /// Create an OPLS-AA typifier from embedded data, XML text, or an XML path.
    #[new]
    #[pyo3(signature = (source = None, *, strict = true))]
    fn new(source: Option<&Bound<'_, PyAny>>, strict: bool) -> PyResult<(Self, PyTypifier)> {
        let typifier = match oplsaa_source_xml(source)? {
            Some(xml) => OPLSAATypifier::from_xml_str(&xml).map_err(PyValueError::new_err)?,
            None => OPLSAATypifier::oplsaa(),
        }
        .with_strict(strict);
        Ok((Self, PyTypifier::native(typifier)))
    }

    fn __repr__(slf: PyRef<'_, Self>) -> String {
        format!(
            "OPLSAATypifier(forcefield='{}')",
            slf.as_super().library_name()
        )
    }
}

/// Extract a flat coordinate array from a Frame's ``"atoms"`` block.
///
/// Reads the ``x``, ``y``, ``z`` columns from the ``"atoms"`` block and
/// interleaves them into a flat 1D array: ``[x0, y0, z0, x1, y1, z1, ...]``.
///
/// Parameters
/// ----------
/// frame : Frame
///     Frame with an ``"atoms"`` block containing ``x``, ``y``, ``z``
///     float columns.
///
/// Returns
/// -------
/// numpy.ndarray, shape (3*N,), dtype float
///     Flat coordinate array suitable for :meth:`Potentials.eval`.
///
/// Raises
/// ------
/// ValueError
///     If the ``"atoms"`` block or required columns are missing.
///
/// Examples
/// --------
/// >>> coords = extract_coords(frame)
/// >>> energy, forces = potentials.eval(coords)
#[pyfunction]
#[pyo3(name = "extract_coords")]
pub fn extract_coords_py<'py>(
    py: Python<'py>,
    frame: &PyFrame,
) -> PyResult<Bound<'py, PyArray1<NpF>>> {
    let core_frame = frame.clone_core_frame()?;
    let coords = extract_coords(&core_frame)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(coords.to_pyarray(py))
}

/// Read a force-field definition from an XML file.
#[pyfunction]
#[pyo3(name = "read_forcefield_xml")]
pub fn read_forcefield_xml_py(path: &str) -> PyResult<PyForceField> {
    let forcefield = molrs::ff::read_forcefield_xml(path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyForceField { inner: forcefield })
}

/// `Send` wrapper around a `*mut ForceFieldRef` so it can ride inside a
/// `PyCapsule` (whose payload must be `Send`).
///
/// `ForceFieldRef` is `!Send` (it holds an `Rc`) and raw pointers are `!Send`,
/// but the capsule is only ever created, read, and destroyed while the Python
/// GIL is held, so no cross-thread `Rc` access occurs. `#[repr(transparent)]`
/// makes the capsule's `void*` reinterpretable as `*mut *mut ForceFieldRef`,
/// matching the frame convention a consumer resolves (mirrors
/// [`crate::core::store::frame`]'s `FrameRefPtr`).
#[repr(transparent)]
struct ForceFieldRefPtr(*mut ForceFieldRef);

// SAFETY: GIL-guarded, single-threaded use only — see the type-level doc.
unsafe impl Send for ForceFieldRefPtr {}

/// A `[1-2, 1-3, 1-4]` weight triple as Python sees it.
type Weights14 = (f64, f64, f64);

impl PyForceField {
    /// Wrap a core [`ForceField`] as the public ``molrs.ff.ForceField``, so a
    /// force field handed out by the native layer carries the Python builder
    /// methods like every other one.
    pub(crate) fn from_core(py: Python<'_>, inner: ForceField) -> PyResult<Py<PyForceField>> {
        let public = py.import("molrs.ff")?.getattr("ForceField")?;
        if public.is(py.get_type::<PyForceField>()) {
            return Py::new(py, PyForceField { inner });
        }
        let object: Py<PyForceField> = public.call1((inner.name.clone(),))?.extract()?;
        object.borrow_mut(py).inner = inner;
        Ok(object)
    }

    /// The existing `(category, style)`; a missing one raises ``ValueError``
    /// (the type primitives never create a style implicitly).
    fn existing_style_mut(
        &mut self,
        category: &str,
        style: &str,
    ) -> PyResult<&mut molrs::ff::forcefield::Style> {
        self.inner.get_style_mut(category, style).ok_or_else(|| {
            py_value_err(molrs::ff::forcefield::DefError::UnknownStyle {
                category: category.to_owned(),
                name: style.to_owned(),
            })
        })
    }
}

#[pymethods]
impl PyForceField {
    /// Construct an empty force field. Populate it with :meth:`def_style` and
    /// the type primitives, or load one with :func:`read_forcefield_xml`.
    /// ``units`` declares the unit system when given; left out, the force
    /// field declares none (:meth:`declared_units` is ``None``).
    #[new]
    #[pyo3(signature = (name = "forcefield", units = None))]
    fn new(name: &str, units: Option<&str>) -> Self {
        let mut inner = ForceField::new(name);
        if let Some(units) = units {
            inner.set_units(units);
        }
        Self { inner }
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// The unit system the parameters are expressed in (a LAMMPS ``units``
    /// name); ``"real"`` when none is declared. Assigning declares it.
    #[getter]
    fn units(&self) -> String {
        self.inner.units().to_owned()
    }

    #[setter]
    fn set_units(&mut self, units: &str) {
        self.inner.set_units(units);
    }

    /// The declared unit system, or ``None`` when the force field declares none.
    fn declared_units(&self) -> Option<String> {
        self.inner.declared_units().map(str::to_owned)
    }

    /// The declared special-bond weights as ``((lj12, lj13, lj14), (coul12,
    /// coul13, coul14))``, or ``None`` when the force field declares none.
    fn declared_special_bonds(&self) -> Option<(Weights14, Weights14)> {
        self.inner.declared_special_bonds().map(|sb| {
            (
                (sb.lj[0], sb.lj[1], sb.lj[2]),
                (sb.coul[0], sb.coul[1], sb.coul[2]),
            )
        })
    }

    /// Merge ``other`` into this force field, in place, and return ``self``.
    ///
    /// The union of both definitions: ``other``'s styles and types are defined
    /// through the same primitives, so an identical re-definition is a no-op
    /// and a different one raises ``ValueError``. Declared ``units`` and
    /// ``special_bonds`` are adopted when this force field declares none; two
    /// declared values that differ raise ``ValueError``. On error nothing
    /// changes.
    fn merge<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyForceField>,
    ) -> PyResult<Bound<'py, Self>> {
        // Merging a force field into itself is the identical overlap: a no-op
        // (and borrowing it both ways at once would fail).
        if !slf.is(other) {
            let other = other.borrow();
            slf.borrow_mut()
                .inner
                .merge(&other.inner)
                .map_err(py_value_err)?;
        }
        Ok(slf.clone())
    }

    /// Lennard-Jones 1-2 / 1-3 / 1-4 scale weights (copy of length 3).
    ///
    /// Entries ``[0]`` and ``[1]`` are stored and round-tripped for format
    /// fidelity but are never applied by molrs kernels (1-2/1-3 exclusion is
    /// by omitting pairs from the neighbour list). Index ``[2]`` is the 1-4
    /// weight kernels consume.
    #[getter]
    fn special_bonds_lj<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        Array1::from(self.inner.special_bonds().lj.to_vec()).into_pyarray(py)
    }

    /// Coulomb 1-2 / 1-3 / 1-4 scale weights (copy of length 3).
    ///
    /// Entries ``[0]`` and ``[1]`` are stored and round-tripped for format
    /// fidelity but are never applied by molrs kernels. Index ``[2]`` is the
    /// 1-4 weight kernels consume.
    #[getter]
    fn special_bonds_coul<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        Array1::from(self.inner.special_bonds().coul.to_vec()).into_pyarray(py)
    }

    /// Declare both LJ and Coulomb special-bond triples.
    ///
    /// Whole-struct write: to change only Coulomb, read ``special_bonds_lj``
    /// and pass it back. Length-3 sequences required; a wrong length raises
    /// ``ValueError``. Entries ``[0]``/``[1]`` are stored but not applied.
    fn set_special_bonds(&mut self, lj: [f64; 3], coul: [f64; 3]) {
        self.inner
            .set_special_bonds(molrs::ff::forcefield::SpecialBonds { lj, coul });
    }

    fn style_names(&self) -> Vec<String> {
        self.inner
            .styles()
            .iter()
            .map(|style| format!("{}:{}", style.category(), style.name()))
            .collect()
    }

    /// Export this force field's FFI handle as a ``PyCapsule``.
    ///
    /// The force-field analogue of :meth:`Frame._ffi_frameref_capsule`. The
    /// capsule wraps a :class:`molrs_ffi.ForceFieldRef` that **shares** this
    /// force field's parameters (one ``Rc`` clone — no deep copy), so a
    /// downstream Rust consumer (e.g. the molpack relaxer) can resolve it and
    /// compile potentials with **no marshalling**. The capsule's ``void*`` is
    /// ``*mut *mut`` :class:`molrs_ffi.ForceFieldRef`, matching the frame
    /// convention; its name is ``molrs_ffi::abi::forcefield_capsule_name()``
    /// — ``"molrs.ForceFieldRef/<major.minor>"``, carrying the ABI line so a
    /// cross-minor consumer fails the name check cleanly. The capsule's
    /// destructor reclaims the boxed handle, dropping its ``Rc``.
    ///
    /// Returns
    /// -------
    /// capsule
    ///     A ``PyCapsule`` named ``"molrs.ForceFieldRef/<major.minor>"``.
    fn _ffi_forcefield_capsule<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        // Box a shared handle (Rc clone of this force field) and hand the raw
        // pointer to the capsule. See `ForceFieldRefPtr` for the Send / layout
        // contract.
        let raw = ForceFieldRefPtr(Box::into_raw(Box::new(ForceFieldRef::new(
            self.inner.clone(),
        ))));
        let name = molrs_ffi::abi::forcefield_capsule_name().to_owned();
        PyCapsule::new_with_destructor(py, raw, Some(name), |ptr: ForceFieldRefPtr, _ctx| {
            // SAFETY: `ptr.0` came from `Box::into_raw` above and is reclaimed
            // exactly once when the capsule dies.
            drop(unsafe { Box::from_raw(ptr.0) });
        })
    }

    // -- builder: the three definition primitives ------------------------------

    /// Define the ``category`` style ``name`` with style-level ``params``
    /// (numbers and strings, e.g. ``{"cutoff": 10.0, "mixing": "geometric"}``).
    /// Re-defining it with equal ``params`` keeps the existing style; with
    /// different ``params`` it raises ``ValueError``, as does an unknown
    /// category.
    #[pyo3(signature = (category, name, params = None))]
    fn def_style(
        &mut self,
        category: &str,
        name: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let params = params_from_dict(params)?;
        self.inner
            .def_style(category, name, params)
            .map(|_| ())
            .map_err(py_value_err)
    }

    /// Define a type on the existing ``(category, style)``, its endpoints parsed
    /// from ``name``. A missing style, a malformed name or a conflicting
    /// re-definition raises ``ValueError``.
    /// Private: the public door is ``Style.def_type`` in ``molrs.ff``.
    #[pyo3(signature = (category, style, name, params = None))]
    fn _def_type(
        &mut self,
        category: &str,
        style: &str,
        name: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let params = params_from_dict(params)?;
        self.existing_style_mut(category, style)?
            .def_type(name, params)
            .map(|_| ())
            .map_err(py_value_err)
    }

    /// Define a type named ``name`` with the given ``endpoints`` on the existing
    /// ``(category, style)``. A missing style, an endpoint count that does not
    /// match the category or a conflicting re-definition raises ``ValueError``. Private: the public door is
    /// ``Style.def_type_at`` in ``molrs.ff``.
    #[pyo3(signature = (category, style, name, endpoints, params = None))]
    fn _def_type_at(
        &mut self,
        category: &str,
        style: &str,
        name: &str,
        endpoints: Vec<String>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let params = params_from_dict(params)?;
        let endpoints: Vec<&str> = endpoints.iter().map(String::as_str).collect();
        self.existing_style_mut(category, style)?
            .def_type_at(name, &endpoints, params)
            .map(|_| ())
            .map_err(py_value_err)
    }

    // -- read accessors (round-trip + P1-A migration) ------------------------

    /// Style-level params for ``category``/``style``, numeric and string (e.g. a
    /// pair style's ``cutoff`` and ``mixing``).
    fn style_params<'py>(
        &self,
        py: Python<'py>,
        category: &str,
        style: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let s = self
            .inner
            .get_style(category, style)
            .ok_or_else(|| PyValueError::new_err(format!("no {category} style named '{style}'")))?;
        let d = PyDict::new(py);
        for (k, v) in s.params().iter() {
            d.set_item(k, v)?;
        }
        for (k, v) in s.params().iter_strings() {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    /// List ``(type_name, params)`` tuples for ``category``/``style``.
    fn types<'py>(
        &self,
        py: Python<'py>,
        category: &str,
        style: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        let s = self
            .inner
            .get_style(category, style)
            .ok_or_else(|| PyValueError::new_err(format!("no {category} style named '{style}'")))?;
        let out = PyList::empty(py);
        for (name, params) in s.defs().collect_type_params() {
            let d = PyDict::new(py);
            for (k, v) in params.iter() {
                d.set_item(k, v)?;
            }
            for (k, v) in params.iter_strings() {
                d.set_item(k, v)?;
            }
            out.append((name, d))?;
        }
        Ok(out)
    }

    // -- handle-view support (Style/Type live in the Python layer over these) --

    /// Endpoint atom-type names of one type, e.g. ``["CT","CT"]`` for a bond.
    /// ``None`` if no such type; ``[]`` for atom styles.
    fn type_endpoints(
        &self,
        category: &str,
        style: &str,
        name: &str,
    ) -> PyResult<Option<Vec<String>>> {
        let s = self
            .inner
            .get_style(category, style)
            .ok_or_else(|| PyValueError::new_err(format!("no {category} style named '{style}'")))?;
        Ok(s.type_endpoints(name))
    }

    /// Set (or add) a single param on one type. Raises if the type is absent.
    fn set_type_param(
        &mut self,
        category: &str,
        style: &str,
        name: &str,
        key: &str,
        value: f64,
    ) -> PyResult<()> {
        let s = self
            .inner
            .get_style_mut(category, style)
            .ok_or_else(|| PyValueError::new_err(format!("no {category} style named '{style}'")))?;
        if s.set_type_param(name, key, value) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "no {category} type named '{name}' in style '{style}'"
            )))
        }
    }

    /// Set (or add) a single **string** param on one type (e.g. ``element``).
    /// Raises if the type is absent.
    fn set_type_str_param(
        &mut self,
        category: &str,
        style: &str,
        name: &str,
        key: &str,
        value: &str,
    ) -> PyResult<()> {
        let s = self
            .inner
            .get_style_mut(category, style)
            .ok_or_else(|| PyValueError::new_err(format!("no {category} style named '{style}'")))?;
        if s.set_type_str_param(name, key, value) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "no {category} type named '{name}' in style '{style}'"
            )))
        }
    }

    /// Rename type ``old`` -> ``new`` in ``(category, style)``; returns the
    /// count renamed (0 or 1). Renaming onto a name already defined with other
    /// endpoints or params raises ``ValueError`` and changes nothing.
    fn rename_type(
        &mut self,
        category: &str,
        style: &str,
        old: &str,
        new: &str,
    ) -> PyResult<usize> {
        let s = self
            .inner
            .get_style_mut(category, style)
            .ok_or_else(|| PyValueError::new_err(format!("no {category} style named '{style}'")))?;
        let renamed = s.rename_type(old, new).map_err(py_value_err)?;
        Ok(usize::from(renamed))
    }

    /// Remove every type ``name`` in ``(category, style)``; returns count.
    fn remove_type(&mut self, category: &str, style: &str, name: &str) -> PyResult<usize> {
        let s = self
            .inner
            .get_style_mut(category, style)
            .ok_or_else(|| PyValueError::new_err(format!("no {category} style named '{style}'")))?;
        Ok(s.remove_type(name))
    }

    /// Remove a whole style ``(category, name)``; returns whether one was removed.
    fn remove_style(&mut self, category: &str, name: &str) -> bool {
        self.inner.remove_style(category, name)
    }

    fn __repr__(&self) -> String {
        format!(
            "ForceField(name='{}', styles={})",
            self.inner.name,
            self.inner.styles().len()
        )
    }
}

/// Compiles a :class:`ForceField` into evaluable kernels.
///
/// Exposed to Python as ``molrs.ff.PotentialCompiler``. It owns a **copy** of
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
/// >>> compiler = molrs.ff.PotentialCompiler(typifier.forcefield())
/// >>> potentials = compiler.compile(frame)
/// >>> energy = potentials.calc_energy(frame)
#[pyclass(module = "molrs.ff", name = "PotentialCompiler")]
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
    /// list with no spatial cutoff, right for a molecule in free space.
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
    ///     If a style has no registered kernel, a type label is unknown, or
    ///     the force field's 1-2 / 1-3 weights are not 0 or 1.
    fn compile(&self, frame: &PyFrame) -> PyResult<PyPotentials> {
        let core = frame.clone_core_frame()?;
        let potentials = PotentialCompiler::new(&self.ff)
            .compile(&core)
            .map_err(PyValueError::new_err)?;
        Ok(PyPotentials {
            inner: PotBacking::Compiled(potentials),
            err_slots: Vec::new(),
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
            err_slots: Vec::new(),
        }
    }

    /// Build the kernels for a **neighbour-driven** evaluation, with the
    /// special-bonds weights each one takes.
    ///
    /// The counterpart of :meth:`compile`, and what periodic MD needs. That
    /// one resolves every pair style against the frame's ``pairs`` block — a
    /// fixed list with no spatial cutoff, right for a free-boundary molecule
    /// and wrong for a periodic system. This one resolves them against the
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
        let core = frame.clone_core_frame()?;
        let topo =
            molrs::Topology::from_frame(&core).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let members = PotentialCompiler::new(&self.ff)
            .compile_typed(&core)
            .map_err(PyValueError::new_err)?;
        let bound = members
            .into_iter()
            .map(|(pot, weights)| {
                let special = weights
                    .map(|w| molrs::md::SpecialWeights::new(&topo.special_weights(&w)))
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

/// Parse a force-field definition from an XML string (same schema as
/// :func:`read_forcefield_xml`).
#[pyfunction]
#[pyo3(name = "read_forcefield_xml_str")]
pub fn read_forcefield_xml_str_py(xml: &str) -> PyResult<PyForceField> {
    let forcefield = molrs::ff::read_forcefield_xml_str(xml)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyForceField { inner: forcefield })
}

/// Read an OPLS-AA / GROMACS force-field XML file into a :class:`ForceField`.
///
/// Parses the OpenMM-style OPLS-AA XML (GROMACS units — nm, kJ/mol,
/// Ryckaert-Bellemans torsions) and normalizes it to molrs units (Å, kcal/mol,
/// radians, e): bond/angle/pair conversions plus the RB → OPLS 4-cosine
/// (``f1..f4``) inversion happen in the reader, so the returned force field is
/// pure molrs units. Distinct from :func:`read_forcefield_xml`, which reads
/// molrs's own native schema.
///
/// Parameters
/// ----------
/// path : str
///     Path to an ``oplsaa.xml`` (OpenMM/GROMACS layout).
///
/// Returns
/// -------
/// ForceField
///
/// Raises
/// ------
/// ValueError
///     On a malformed document, an unknown section, or a missing/non-numeric
///     required attribute (reading is total — never a silent skip).
#[pyfunction]
#[pyo3(name = "read_opls_xml")]
pub fn read_opls_xml_py(path: &str) -> PyResult<PyForceField> {
    use molrs::ff::ForceFieldReader;
    let forcefield = molrs::ff::OplsXmlReader::new()
        .read(path)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Parse an OPLS-AA / GROMACS force field from an XML string (same schema and
/// unit normalization as :func:`read_opls_xml`).
#[pyfunction]
#[pyo3(name = "read_opls_xml_str")]
pub fn read_opls_xml_str_py(xml: &str) -> PyResult<PyForceField> {
    use molrs::ff::ForceFieldReader;
    let forcefield = molrs::ff::OplsXmlReader::new()
        .read_str(xml)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read a LAMMPS force-field include (``*.ff``) into a :class:`ForceField`.
///
/// Parses the ``pair_style``/``pair_coeff`` + ``bond_style``/``angle_style``/
/// ``dihedral_style`` (``fourier``) [+ optional ``improper_style``] include that
/// :func:`write_lammps_forcefield` emits (AMBER/GAFF flavour), normalizing it
/// to molrs units (Å, kcal/mol, radians, e): LAMMPS harmonic ``K`` → molrs
/// ``k = 2K``, angle/phase values deg → rad at this boundary, and the
/// ``fourier`` dihedral maps to the molrs ``periodic`` kernel. AMBER 1-4 scaling
/// (LJ ×0.5, Coulomb ×1/1.2) is recorded on the force field's special bonds.
/// Distinct from :func:`read_forcefield_xml` (molrs's own schema) and
/// :func:`read_opls_xml` (OPLS-AA / GROMACS XML).
///
/// Per-atom charge and mass live in the LAMMPS *data* file, not this include, so
/// they are not read here: Coulomb charges are drawn from the frame at evaluation
/// time and masses are irrelevant to geometry relaxation.
///
/// Parameters
/// ----------
/// path : str
///     Path to a LAMMPS force-field include (``*.ff``).
///
/// Returns
/// -------
/// ForceField
///
/// Raises
/// ------
/// ValueError
///     On an unsupported style, a coefficient before its style declaration, a
///     wrong-arity type label, or a non-numeric parameter (reading is total —
///     never a silent skip).
#[pyfunction]
#[pyo3(name = "read_lammps_forcefield")]
pub fn read_lammps_forcefield_py(path: &str) -> PyResult<PyForceField> {
    use molrs::ff::ForceFieldReader;
    let forcefield = molrs::ff::LammpsFfReader::new()
        .read(path)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Parse a LAMMPS force-field include from a string (same format and unit
/// normalization as :func:`read_lammps_forcefield`).
#[pyfunction]
#[pyo3(name = "read_lammps_forcefield_str")]
pub fn read_lammps_forcefield_str_py(text: &str) -> PyResult<PyForceField> {
    use molrs::ff::ForceFieldReader;
    let forcefield = molrs::ff::LammpsFfReader::new()
        .read_str(text)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read AMBER prmtop force-field parameter tables into a :class:`ForceField`.
///
/// Structure/connectivity is :func:`molrs.io.read_amber_prmtop`; this parses
/// harmonic bond/angle tables (``k = 2·K`` form map), Fourier dihedrals, and
/// LJ A/B → σ/ε. Store units are molrs (Å, kcal/mol, radians, e).
#[pyfunction]
#[pyo3(name = "read_amber_prmtop_ff")]
pub fn read_amber_prmtop_ff_py(path: &str) -> PyResult<PyForceField> {
    let forcefield =
        molrs::ff::read_amber_prmtop_ff(path).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Parse AMBER prmtop force-field tables from a string.
#[pyfunction]
#[pyo3(name = "read_amber_prmtop_ff_str")]
pub fn read_amber_prmtop_ff_str_py(text: &str) -> PyResult<PyForceField> {
    use molrs::ff::ForceFieldReader;
    let forcefield = molrs::ff::AmberPrmtopFfReader::new()
        .read_str(text)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read a GROMACS ``.top`` / ``.itp`` into a :class:`ForceField`.
///
/// Parses ``[ atoms ]`` / ``[ bonds ]`` / ``[ angles ]`` / ``[ dihedrals ]`` /
/// ``[ pairs ]`` tables. Bonded parameters (when present) are converted from
/// GROMACS units (nm, kJ/mol, degrees) to molrs store units. ``include``
/// controls ``#include`` expansion (default false).
#[pyfunction]
#[pyo3(name = "read_gromacs_top_ff", signature = (path, include = false))]
pub fn read_gromacs_top_ff_py(path: &str, include: bool) -> PyResult<PyForceField> {
    use molrs::ff::ForceFieldReader;
    let forcefield = molrs::ff::GromacsTopFfReader::new()
        .with_include(include)
        .read(path)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Parse GROMACS topology force-field tables from a string.
#[pyfunction]
#[pyo3(name = "read_gromacs_top_ff_str", signature = (text, include = false))]
pub fn read_gromacs_top_ff_str_py(text: &str, include: bool) -> PyResult<PyForceField> {
    use molrs::ff::ForceFieldReader;
    let forcefield = molrs::ff::GromacsTopFfReader::new()
        .with_include(include)
        .read_str(text)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Write a ForceField to GROMACS ``.top`` / ``.itp`` force-field tables.
#[pyfunction]
#[pyo3(name = "write_gromacs_top_ff", signature = (path, forcefield, precision = 6))]
pub fn write_gromacs_top_ff_py(
    path: &str,
    forcefield: &PyForceField,
    precision: usize,
) -> PyResult<()> {
    molrs::ff::write_gromacs_top_ff(path, &forcefield.inner, precision)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Serialize a ForceField to a GROMACS topology force-field string.
#[pyfunction]
#[pyo3(name = "write_gromacs_top_ff_str", signature = (forcefield, precision = 6))]
pub fn write_gromacs_top_ff_str_py(
    forcefield: &PyForceField,
    precision: usize,
) -> PyResult<String> {
    molrs::ff::write_gromacs_top_ff_str(&forcefield.inner, precision)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Write a ForceField to OpenMM-style XML.
#[pyfunction]
#[pyo3(name = "write_forcefield_xml", signature = (path, forcefield, precision = 6))]
pub fn write_forcefield_xml_py(
    path: &str,
    forcefield: &PyForceField,
    precision: usize,
) -> PyResult<()> {
    molrs::ff::write_forcefield_xml(path, &forcefield.inner, precision)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Serialize a ForceField to OpenMM-style XML string.
#[pyfunction]
#[pyo3(name = "write_forcefield_xml_str", signature = (forcefield, precision = 6))]
pub fn write_forcefield_xml_str_py(
    forcefield: &PyForceField,
    precision: usize,
) -> PyResult<String> {
    molrs::ff::write_forcefield_xml_str(&forcefield.inner, precision)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Parse LAMMPS data-file ``* Coeffs`` sections into a :class:`ForceField`.
///
/// ``coeffs_text`` may contain ``Pair Coeffs`` / ``Bond Coeffs`` / … blocks
/// (and an optional ``units`` line). Default styles are harmonic / ``lj/cut``
/// when the data file has no style directives. Optional ``*_labels`` maps are
/// 1-based type id → label string (from Type Labels sections).
#[pyfunction]
#[pyo3(
    name = "read_lammps_data_coeffs",
    signature = (
        coeffs_text,
        units = "real",
        atom_labels = None,
        bond_labels = None,
        angle_labels = None,
        dihedral_labels = None,
        improper_labels = None,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn read_lammps_data_coeffs_py(
    coeffs_text: &str,
    units: &str,
    atom_labels: Option<std::collections::HashMap<u32, String>>,
    bond_labels: Option<std::collections::HashMap<u32, String>>,
    angle_labels: Option<std::collections::HashMap<u32, String>>,
    dihedral_labels: Option<std::collections::HashMap<u32, String>>,
    improper_labels: Option<std::collections::HashMap<u32, String>>,
) -> PyResult<PyForceField> {
    use molrs::ff::LammpsFfReader;
    use molrs::ff::forcefield::lammps_units::parse_style;
    use molrs::ff::forcefield::readers::lammps::LammpsTypeLabelMaps;
    use std::collections::BTreeMap;

    let units = parse_style(units).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let to_btree = |m: Option<std::collections::HashMap<u32, String>>| -> BTreeMap<u32, String> {
        m.unwrap_or_default().into_iter().collect()
    };
    let labels = LammpsTypeLabelMaps {
        atom: to_btree(atom_labels),
        bond: to_btree(bond_labels),
        angle: to_btree(angle_labels),
        dihedral: to_btree(dihedral_labels),
        improper: to_btree(improper_labels),
    };
    let forcefield = LammpsFfReader::new()
        .read_data_coeffs(coeffs_text, &labels, units)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Write a :class:`ForceField` to a LAMMPS force-field include (``*.ff``).
///
/// Coefficient writing, keyed by the system's type labels: ``frame``'s
/// ``atoms`` / ``bonds`` / ``angles`` / ``dihedrals`` / ``impropers`` type
/// labels are walked in id order and each is looked up in ``forcefield``
/// (bond, angle and dihedral labels in either orientation, impropers exactly).
/// Force-field types no label uses are not written.
///
/// Inverse of :func:`read_lammps_forcefield`: molrs store (Å, kcal/mol, radians,
/// ``½k`` harmonic form for physical styles) → LAMMPS file units
/// (``K = k/2``, angles in degrees). Energy/length conversion for
/// ``units="metal"`` / ``"lj"`` goes through the lj reduced hub — never
/// hard-coded eV/kcal factors. A split ``lj/cut`` + ``coul/cut`` pair is
/// recombined as ``lj/cut/coul/cut`` so geometric mixing is not defeated by a
/// hybrid wildcard. Tier A: ``bond``/``angle``/``improper`` harmonic;
/// ``dihedral`` fourier / opls / harmonic.
///
/// Parameters
/// ----------
/// path : str
///     Destination path for the include.
/// forcefield : ForceField
///     Force field in molrs store units.
/// frame : Frame
///     The system whose type labels select the coefficients.
/// precision : int, optional
///     Decimal places for floating coefficients (default 6).
/// skip_pair_style : bool, optional
///     When true, omit ``pair_style`` **and** ``special_bonds`` (caller sets
///     both in the input). A coeff-only include that still writes Amber
///     ``special_bonds`` (coul 1-4 = 1/1.2) silently applies those weights.
/// skip_units : bool, optional
///     When true, omit the ``units`` line so the include can follow ``units``
///     already set in the input script.
/// units : str, optional
///     LAMMPS ``units`` style for the written file: ``"real"`` (default),
///     ``"metal"``, or ``"lj"``.
///
/// Raises
/// ------
/// ValueError
///     On a frame type label the force field does not define (the message
///     names the block and the label), an unsupported style holding a used
///     type, a bad units keyword, or missing required parameters.
#[pyfunction]
#[pyo3(
    name = "write_lammps_forcefield",
    signature = (
        path,
        forcefield,
        frame,
        *,
        precision = 6,
        skip_pair_style = false,
        skip_units = false,
        units = "real",
    )
)]
pub fn write_lammps_forcefield_py(
    path: &str,
    forcefield: &PyForceField,
    frame: &PyFrame,
    precision: usize,
    skip_pair_style: bool,
    skip_units: bool,
    units: &str,
) -> PyResult<()> {
    use molrs::ff::forcefield::lammps_units::parse_style;
    use molrs::ff::{ForceFieldWriter, LammpsFfWriter, LammpsWriteOptions};
    use molrs::store::type_labels::TypeLabels;
    let units = parse_style(units).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let labels = frame
        .with_frame(TypeLabels::from_frame)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let writer = LammpsFfWriter::with_options(
        &labels,
        LammpsWriteOptions {
            precision,
            skip_pair_style,
            skip_units,
            units,
        },
    );
    writer
        .write(&forcefield.inner, path)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Serialize a :class:`ForceField` to a LAMMPS force-field include string
/// (same labels, format and unit conversion as :func:`write_lammps_forcefield`).
#[pyfunction]
#[pyo3(
    name = "write_lammps_forcefield_str",
    signature = (
        forcefield,
        frame,
        *,
        precision = 6,
        skip_pair_style = false,
        skip_units = false,
        units = "real",
    )
)]
pub fn write_lammps_forcefield_str_py(
    forcefield: &PyForceField,
    frame: &PyFrame,
    precision: usize,
    skip_pair_style: bool,
    skip_units: bool,
    units: &str,
) -> PyResult<String> {
    use molrs::ff::forcefield::lammps_units::parse_style;
    use molrs::ff::{ForceFieldWriter, LammpsFfWriter, LammpsWriteOptions};
    use molrs::store::type_labels::TypeLabels;
    let units = parse_style(units).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let labels = frame
        .with_frame(TypeLabels::from_frame)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let writer = LammpsFfWriter::with_options(
        &labels,
        LammpsWriteOptions {
            precision,
            skip_pair_style,
            skip_units,
            units,
        },
    );
    writer
        .write_str(&forcefield.inner)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Serialize a :class:`ForceField` to LAMMPS data-file ``* Coeffs`` sections.
///
/// Same labels, form map and ``units`` conversion as
/// :func:`write_lammps_forcefield`, but emits ``Pair Coeffs`` / ``Bond Coeffs``
/// / … blocks whose integer ids are ``frame``'s type-label ids, not
/// input-script ``*_coeff`` lines. ``Pair Coeffs`` holds self pairs only; a
/// used explicit cross pair raises ``ValueError``.
#[pyfunction]
#[pyo3(
    name = "write_lammps_data_coeffs",
    signature = (
        forcefield,
        frame,
        *,
        precision = 6,
        units = "real",
    )
)]
pub fn write_lammps_data_coeffs_py(
    forcefield: &PyForceField,
    frame: &PyFrame,
    precision: usize,
    units: &str,
) -> PyResult<String> {
    use molrs::ff::forcefield::lammps_units::parse_style;
    use molrs::ff::{LammpsFfWriter, LammpsWriteOptions};
    use molrs::store::type_labels::TypeLabels;
    let units = parse_style(units).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let labels = frame
        .with_frame(TypeLabels::from_frame)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let writer = LammpsFfWriter::with_options(
        &labels,
        LammpsWriteOptions {
            precision,
            units,
            ..LammpsWriteOptions::default()
        },
    );
    writer
        .write_data_coeffs_str(&forcefield.inner)
        .map_err(pyo3::exceptions::PyValueError::new_err)
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
    let core = frame.clone_core_frame()?;
    let owned;
    let special = match forcefield {
        Some(ff) => ff.inner.special_bonds(),
        None => {
            owned = molrs::ff::forcefield::SpecialBonds::default();
            &owned
        }
    };
    let block = molrs::ff::potential::intramolecular_pairs(&core, special)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    PyBlock::from_core_block(block)
}
