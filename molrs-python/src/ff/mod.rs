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
//! [`atd`] (the ATD atom typifier, one engine over seven `ATOMTYPE_*.DEF` tables),
//! [`gaff`] (the GAFF / GAFF2 bonded-term typifier) and [`charge`] (the three
//! charge models). This file is already large, and they
//! are self-contained.
//!
//! # References
//!
//! - Halgren, T.A. (1996). J. Comput. Chem. 17, 490-519. (MMFF94 force field)
//! - Halgren, T.A. (1999). J. Comput. Chem. 20, 720-729. (MMFF94s option)

pub mod atd;
pub mod charge;
pub mod clpol;
pub mod engine;
pub mod forms;
pub mod gaff;
pub mod handles;
pub mod ir;
pub mod param_columns;
pub mod potential;
pub mod section;

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use pyo3::exceptions::{
    PyKeyError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList, PyMapping, PyString, PySuper, PyTuple, PyType};

use molrs::ff::ForceField;
use molrs::ff::potential::{Member, PotentialCompiler, Potentials};
use molrs::ff::typifier::ElementTypifier;
use molrs::ff::typifier::mmff::{MMFF94STypifier, MMFF94Typifier};
use molrs::ff::typifier::opls::OPLSAATypifier;
use molrs::ff::typifier::{Annotation, Match, Typifier, Typing};
use molrs::optimize::{LBFGS, OptReport};
use molrs_ffi::ForceFieldRef;

use crate::core::store::block::PyBlock;
use crate::core::store::frame::PyFrame;
use crate::core::system::molgraph::{PyAtomistic, py_to_prop};
use crate::core::system::views::RelationClass;
use crate::helpers::{NpF, path_str, py_value_err};

use ndarray::{Array2, Array3};
use numpy::{PyArray2, PyArray3, PyReadonlyArrayDyn, ToPyArray};

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

    /// Reject a subclass that defines ``typify`` in its own body, and a
    /// subclass of a native typifier that defines ``match`` or ``library``.
    ///
    /// ``typify`` is the only writer of :meth:`forcefield`; an override would
    /// silently bypass the output. A native typifier (``OPLSAATypifier``,
    /// ``MMFF94Typifier``, …) types in Rust and never calls a Python ``match``
    /// or ``library``, so overriding either on its subclass would be silently
    /// ignored. ``typing.final`` is only a static check.
    #[classmethod]
    #[pyo3(signature = (**kwargs))]
    fn __init_subclass__(
        cls: &Bound<'_, PyType>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let py = cls.py();
        let own = cls.getattr(intern!(py, "__dict__"))?;
        if own.contains(intern!(py, "typify"))? {
            return Err(PyTypeError::new_err(format!(
                "{} defines typify; a Typifier subclass implements match (and optionally \
                 library) only — typify is the base's and the only writer of forcefield()",
                cls.name()?
            )));
        }
        let native = cls.is_subclass_of::<PyOPLSAATypifier>()?
            || cls.is_subclass_of::<PyMMFF94Typifier>()?
            || cls.is_subclass_of::<PyMMFF94STypifier>()?
            || cls.is_subclass_of::<PyElementTypifier>()?
            || cls.is_subclass_of::<atd::PyAtdTypifier>()?
            || cls.is_subclass_of::<gaff::PyGaffTypifier>()?;
        if native {
            for hook in ["match", "library"] {
                if own.contains(hook)? {
                    return Err(PyTypeError::new_err(format!(
                        "{} defines {hook} on a native typifier, whose typify runs in Rust \
                         and never calls it; subclass molrs.ff.typifier.Typifier to supply \
                         your own {hook}",
                        cls.name()?
                    )));
                }
            }
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
            return mol.borrow().derive(py, typed);
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
///     ``graph.atoms``. An empty mapping gives that node nothing.
/// links : mapping, optional
///     Relation kind to a sequence of mappings, positional against that
///     kind's own rows (an improper never shifts a dihedral position). A key
///     is a relation class (``Bond``, ``Angle``, ``Dihedral``, ``Improper``,
///     ``Port``), rows as ``graph.links.exact_bucket(cls)``, or a kind name
///     (``"bonds"``, or ``"urey_bradleys"`` registered with
///     ``graph.register_kind``), rows as ``graph.relation_ids(kind)``. A type
///     annotation under a kind defines a type of the category whose block
///     the kind is (``urey_bradleys`` → ``urey_bradley``). Any other key
///     raises ``TypeError``; a kind named twice raises ``ValueError``.
/// styles : sequence of (category, style, params), optional
///     Styles to declare, in order.
/// pairs : sequence of (style, name, endpoints, params), optional
///     Pair rows to define.
///
/// Notes
/// -----
/// An annotation is a ``str``, ``bool``, ``int`` or ``float`` (stamped, defines
/// nothing), or a type: ``(style, name, endpoints, params)``, which stamps
/// ``name`` and every param and defines the type ``name`` on ``endpoints``
/// (atom-type names; empty for an atom type) under the style. The name is
/// never read for endpoints. Param values are numbers or strings.
#[pyclass(module = "molrs.ff.typifier", name = "Match", frozen, subclass)]
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
        if tuple.len() != 4 {
            return Err(PyTypeError::new_err(format!(
                "a type annotation is (style, name, endpoints, params); got a {}-tuple",
                tuple.len()
            )));
        }
        let (style, name, endpoints, params): (String, String, Vec<String>, Bound<'_, PyDict>) =
            tuple.extract()?;
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
            for item in links.cast::<PyMapping>()?.items()?.iter() {
                let (key, rows): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item.extract()?;
                let kind = match RelationClass::atomistic_kind(&key) {
                    Some(kind) => kind.to_owned(),
                    None => key.extract::<String>().map_err(|_| {
                        PyTypeError::new_err(format!(
                            "links keys must be Atomistic relation view classes or relation \
                             kind names, got {key}"
                        ))
                    })?,
                };
                if inner.links.contains_key(&kind) {
                    return Err(PyValueError::new_err(format!(
                        "links name relation kind '{kind}' more than once"
                    )));
                }
                let rows = Self::rows(&rows).map_err(|err| prefix_err(rows.py(), err, &kind))?;
                inner.links.insert(kind, rows);
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
        let links: Vec<String> = m
            .links
            .iter()
            .map(|(kind, rows)| format!("{kind}={}", rows.len()))
            .collect();
        format!(
            "Match(nodes={}, links={{{}}}, styles={}, pairs={})",
            m.nodes.len(),
            links.join(", "),
            m.styles.len(),
            m.pairs.len()
        )
    }
}

/// Outcome of a geometry optimization, exposed to Python as `molrs.OptReport`.
#[pyclass(module = "molrs.optimize", name = "OptReport", subclass)]
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
#[pyclass(name = "TypedPotentials", module = "molrs.ff", subclass)]
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
#[pyclass(module = "molrs.ff", name = "Potentials", subclass)]
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

/// Force-field definition metadata exposed to Python as `molrs.ff.ForceField`.
///
/// Subclassable, like every core data class (molnex's `ForceField` extends it).
/// Styles and types
/// are read and written through their handles (:mod:`handles`).
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
    skip_from_py_object,
    subclass
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

/// `value` as an array param: a numpy array of at least one dimension, or a
/// list or tuple of numbers (any nesting), converted to float64 by
/// ``numpy.asarray``; `None` for any other value (a 0-d array is a number). A
/// sequence numpy cannot make a numeric array of (strings, ragged nesting)
/// raises ``TypeError``.
pub(crate) fn array_param(value: &Bound<'_, PyAny>) -> PyResult<Option<ndarray::ArrayD<f64>>> {
    let py = value.py();
    let np = py.import("numpy")?;
    let is_array = if value.is_instance(&np.getattr("ndarray")?)? {
        value.getattr("ndim")?.extract::<usize>()? > 0
    } else {
        value.cast::<PyList>().is_ok() || value.cast::<PyTuple>().is_ok()
    };
    if !is_array {
        return Ok(None);
    }
    let converted = np
        .call_method1("asarray", (value, np.getattr("float64")?))
        .map_err(|e| {
            PyTypeError::new_err(format!(
                "an array param must be numbers of one rectangular shape: {e}"
            ))
        })?;
    let array: PyReadonlyArrayDyn<'_, f64> = converted.extract()?;
    Ok(Some(array.as_array().to_owned()))
}

/// Convert an optional Python ``dict[str, float | str | array]`` of parameters
/// into [`Params`](molrs::ff::forcefield::Params). A ``str`` value goes to the
/// string side, a number to the numeric side, an array (a numpy array or a
/// nested list / tuple of numbers, stored as float64) to the array side;
/// anything else raises ``TypeError``. A missing dict yields no params.
fn params_from_dict(params: Option<&Bound<'_, PyDict>>) -> PyResult<molrs::ff::forcefield::Params> {
    let mut out = molrs::ff::forcefield::Params::new();
    let Some(d) = params else {
        return Ok(out);
    };
    for (k, v) in d.iter() {
        let key = k.extract::<String>()?;
        if let Ok(text) = v.cast::<PyString>() {
            out.set_str(&key, text.to_str()?);
        } else if let Some(array) = array_param(&v)? {
            out.set_array(&key, array);
        } else if let Ok(number) = v.extract::<f64>() {
            out.set(&key, number);
        } else {
            return Err(PyTypeError::new_err(format!(
                "param '{key}' must be a number, a str or an array of numbers, got {}",
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
            let coords: Vec<NpF> = core
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
#[pyclass(module = "molrs.optimize", name = "LBFGS", subclass)]
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
                        .map_err(ir::compile_err)?;
                    &compiled
                }
                PotBacking::Moved => return Err(potentials_moved_err()),
            };
            // Borrowed one-shot on flat coords extracted from frame, then write back.
            let mut xyz = core
                .coords()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            // `Frame::coords` is a fresh row-major N×3 array: its buffer is the
            // flat `[x0, y0, z0, …]` the minimizer takes.
            let flat = xyz
                .as_slice_mut()
                .expect("Frame::coords returns a standard-layout array");
            let report = LBFGS::minimize(
                pot,
                flat,
                self.fmax,
                self.max_steps,
                self.max_step,
                self.memory,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            crate::md::take_err(&pots.err_slots)?;
            core.set_coords(xyz.view())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
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
                crate::md::take_err(&pots.err_slots)?;
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
                crate::md::take_err(&pots.err_slots)?;
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
        #[pyclass(module = "molrs.ff.typifier", name = $name, extends = PyTypifier, subclass)]
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
#[pyclass(module = "molrs.ff.typifier", name = "OPLSAATypifier", extends = PyTypifier, subclass)]
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

/// Element typing: ``type`` labels from element symbols alone, with no force
/// field — ``molrs.ff.typifier.ElementTypifier``.
///
/// :meth:`typify` returns a typed :class:`Atomistic` whose atoms carry
/// ``type = element`` (e.g. ``"C"``) and whose bonds, angles and dihedrals
/// carry their endpoint elements joined with ``-`` in the byte-wise smaller
/// orientation (bond O–H is ``"H-O"``). It is for writers that need type
/// labels (LAMMPS data) on a molecule no force field has typed;
/// :meth:`forcefield` stays empty. Takes no arguments.
///
/// Raises
/// ------
/// ValueError
///     From :meth:`typify`, when an atom has no string ``element`` or the
///     molecule has impropers.
///
/// Examples
/// --------
/// >>> typed = molrs.ff.typifier.ElementTypifier().typify(water)
/// >>> list(typed.to_frame()["bonds"]["type"])
/// ['H-O', 'H-O']
#[pyclass(module = "molrs.ff.typifier", name = "ElementTypifier", extends = PyTypifier, subclass)]
pub struct PyElementTypifier;

#[pymethods]
impl PyElementTypifier {
    #[new]
    fn new() -> (Self, PyTypifier) {
        (Self, PyTypifier::native(ElementTypifier::new()))
    }

    fn __repr__(&self) -> String {
        "ElementTypifier()".to_owned()
    }
}

/// Read a force-field definition from an XML file.
#[pyfunction]
#[pyo3(name = "read_forcefield_xml")]
pub fn read_forcefield_xml_py(path: PathBuf) -> PyResult<PyForceField> {
    let forcefield = molrs::ff::read_forcefield_xml(path_str(&path)?)
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

impl PyForceField {
    /// A new Python ``ForceField`` holding `inner`.
    pub(crate) fn from_core(py: Python<'_>, inner: ForceField) -> PyResult<Py<PyForceField>> {
        Py::new(py, PyForceField { inner })
    }

    /// Every style handle whose category `selection` names, in definition
    /// order.
    fn style_handles(
        slf: &Bound<'_, Self>,
        selection: &handles::Selection,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let py = slf.py();
        let styles: Vec<(handles::Category, String)> = slf
            .try_borrow()?
            .inner
            .styles()
            .iter()
            .filter_map(|style| {
                let category = handles::Category::of_style(style);
                selection
                    .contains(&category)
                    .then(|| (category, style.name().to_owned()))
            })
            .collect();
        let ff = slf.clone().unbind();
        styles
            .iter()
            .map(|(category, name)| category.style_handle(py, &ff, name))
            .collect()
    }

    /// The pickled definition: `(name, declared units, declared special
    /// bonds, [(category, arity, style, params, [(type, endpoints,
    /// params)])])`.
    fn definition<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let styles = PyList::empty(py);
        for style in self.inner.styles() {
            let types = PyList::empty(py);
            for (name, endpoints, params) in style.type_rows() {
                types.append((name, endpoints, params_to_dict(py, params)?))?;
            }
            styles.append((
                style.category(),
                style.arity(),
                style.name(),
                params_to_dict(py, style.params())?,
                types,
            ))?;
        }
        let special_bonds = self
            .inner
            .declared_special_bonds()
            .map(|sb| (sb.lj.to_vec(), sb.coul.to_vec()));
        (
            self.inner.name.clone(),
            self.inner.declared_units().map(str::to_owned),
            special_bonds,
            styles,
        )
            .into_pyobject(py)
    }

    /// The force field a [`definition`](Self::definition) describes.
    #[allow(
        clippy::type_complexity,
        reason = "the pickled definition, see `definition`"
    )]
    fn from_definition(definition: &Bound<'_, PyAny>) -> PyResult<ForceField> {
        let (name, units, special_bonds, styles): (
            String,
            Option<String>,
            Option<([f64; 3], [f64; 3])>,
            Vec<(
                String,
                usize,
                String,
                Bound<'_, PyDict>,
                Vec<(String, Vec<String>, Bound<'_, PyDict>)>,
            )>,
        ) = definition.extract()?;
        let mut inner = ForceField::new(&name);
        if let Some(units) = units {
            inner.set_units(&units);
        }
        if let Some((lj, coul)) = special_bonds {
            inner.set_special_bonds(molrs::ff::forcefield::SpecialBonds { lj, coul });
        }
        for (category, arity, style_name, params, types) in styles {
            // The arity travels with the style: a category no registry of
            // the unpickling process declares is still a relation of it.
            let style = inner
                .def_style_with_arity(
                    &category,
                    arity,
                    &style_name,
                    params_from_dict(Some(&params))?,
                )
                .map_err(ir::def_err)?;
            for (type_name, endpoints, params) in types {
                let endpoints: Vec<&str> = endpoints.iter().map(String::as_str).collect();
                style
                    .def_type(&type_name, &endpoints, params_from_dict(Some(&params))?)
                    .map_err(ir::def_err)?;
            }
        }
        Ok(inner)
    }
}

/// `params` as a dict: numbers, strings, and arrays as new float64 numpy
/// arrays.
fn params_to_dict<'py>(
    py: Python<'py>,
    params: &molrs::ff::forcefield::Params,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (key, value) in params.iter() {
        out.set_item(key, value)?;
    }
    for (key, value) in params.iter_strings() {
        out.set_item(key, value)?;
    }
    for (key, value) in params.iter_arrays() {
        out.set_item(key, value.to_pyarray(py))?;
    }
    Ok(out)
}

#[pymethods]
impl PyForceField {
    /// Construct an empty force field. Populate it with :meth:`def_style` and
    /// the style handles' ``def_type``, or load one with a reader
    /// (:func:`read_forcefield_xml`, …). ``units`` declares the unit system
    /// when given; left out, the force field declares none and :attr:`units`
    /// reads ``"real"``.
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
    /// name); ``"real"`` when none is declared.
    #[getter]
    fn units(&self) -> String {
        self.inner.units().to_owned()
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

    /// The special-bond triples ``(lj, coul)``, each ``[1-2, 1-3, 1-4]`` — what
    /// :meth:`set_special_bonds` declared, or the default ``[0, 0, 1]``.
    #[getter]
    fn special_bonds(&self) -> ([f64; 3], [f64; 3]) {
        let sb = self.inner.special_bonds();
        (sb.lj, sb.coul)
    }

    /// Declare both LJ and Coulomb special-bond triples (1-2, 1-3, 1-4).
    ///
    /// Length-3 sequences required; a wrong length raises ``ValueError``.
    /// Entries ``[0]``/``[1]`` are stored but not applied (1-2/1-3 exclusion
    /// is by omitting pairs from the neighbour list).
    fn set_special_bonds(&mut self, lj: [f64; 3], coul: [f64; 3]) {
        self.inner
            .set_special_bonds(molrs::ff::forcefield::SpecialBonds { lj, coul });
    }

    /// Write this force field's 1-4 pricing of ``frame``'s 1-4 pairs as
    /// per-pair override cells (``epsilon``, ``sigma``, ``lj_scale``,
    /// ``coul_scale``) on its ``pairs`` block, and return how many rows were
    /// filled.
    ///
    /// Every ``pairs`` row flagged ``is_14`` (each 1-4 pair once; the block is
    /// built when the frame has none) that no ``dihedral charmm`` ``w > 0``
    /// covers gets its null cells: ``epsilon``/``sigma`` are the pair's 1-4
    /// Lennard-Jones parameters — under ``lj/charmm`` with ``one_four =
    /// "epsilon14"`` the cross (NBFIX) row or the two types'
    /// ``epsilon14``/``sigma14`` mixed, else the regular pair parameters — and
    /// the scales are the ``special_bonds`` 1-4 weights. Cells already set are
    /// kept. A field whose ``lj/charmm`` declares ``one_four = "epsilon14"``
    /// (an OpenMM ``<LennardJonesForce>`` with ``sigma14``/``epsilon14``, a
    /// GROMACS ``[ pairtypes ]`` table) needs this on its frames before it
    /// compiles; LAMMPS's writers refuse the override columns.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     A typed frame (``atoms.type``), modified in place.
    ///
    /// Returns
    /// -------
    /// int
    ///     The number of ``pairs`` rows given override cells.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     An untyped frame, a type without a pair row, two Lennard-Jones
    ///     styles or an invalid ``one_four``.
    fn materialize_one_four(&self, frame: &PyFrame) -> PyResult<usize> {
        frame
            .with_frame_mut(|core| self.inner.materialize_one_four(core))?
            .map_err(py_value_err)
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

    // -- styles: defined here, read and written through their handles ----------

    /// Define the ``category`` style ``name`` with style-level ``params``
    /// (numbers and strings, e.g. ``{"cutoff": 10.0, "mixing": "geometric"}``)
    /// and return its handle (``AtomStyle`` … ``PairStyle``, ``CmapStyle``),
    /// whose typed ``def_type`` defines types. Any other category the
    /// force-field IR registry declares (``drude``, a registered custom
    /// category, …), or one this force field holds already, returns a
    /// ``RelationStyle``, whose ``def_type(name, *endpoints, **params)``
    /// takes as many endpoints as the category's arity. Re-defining it with
    /// equal ``params`` keeps the existing style.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     On different ``params`` for an existing style, or an unknown
    ///     category.
    #[pyo3(signature = (category, name, params = None))]
    fn def_style(
        slf: &Bound<'_, Self>,
        category: &str,
        name: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let params = params_from_dict(params)?;
        let category = {
            let mut ff = slf.try_borrow_mut()?;
            let style = ff
                .inner
                .def_style(category, name, params)
                .map_err(ir::def_err)?;
            handles::Category::of_style(style)
        };
        category.style_handle(slf.py(), &slf.clone().unbind(), name)
    }

    /// Every style, in definition order.
    #[getter(styles)]
    fn every_style(slf: &Bound<'_, Self>) -> PyResult<Vec<Py<PyAny>>> {
        Self::style_handles(slf, &handles::Selection::Every)
    }

    /// The ``category`` style ``name``, or ``None``.
    fn get_style(slf: &Bound<'_, Self>, category: &str, name: &str) -> PyResult<Option<Py<PyAny>>> {
        let Some(category) = slf
            .try_borrow()?
            .inner
            .get_style(category, name)
            .map(handles::Category::of_style)
        else {
            return Ok(None);
        };
        category
            .style_handle(slf.py(), &slf.clone().unbind(), name)
            .map(Some)
    }

    /// The styles of a category — a name (``"bond"``) or a style class
    /// (``BondStyle``; ``RelationStyle`` selects every category beyond the
    /// seven, ``Style`` every category).
    fn get_styles(slf: &Bound<'_, Self>, category: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        let selection = handles::Category::selected(category, &slf.try_borrow()?.inner)?;
        Self::style_handles(slf, &selection)
    }

    /// The types of a category — a name (``"bond"``) or a type class
    /// (``BondType``; ``RelationType`` selects every category beyond the
    /// seven, ``Type`` every category) — style by style.
    fn get_types(slf: &Bound<'_, Self>, category: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        let py = slf.py();
        let mut types = Vec::new();
        let selection = handles::Category::selected(category, &slf.try_borrow()?.inner)?;
        for style in Self::style_handles(slf, &selection)? {
            types.extend(
                style
                    .bind(py)
                    .getattr(intern!(py, "types"))?
                    .extract::<Vec<Py<PyAny>>>()?,
            );
        }
        Ok(types)
    }

    // -- pickling ----------------------------------------------------------------

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let definition = slf.try_borrow()?.definition(py)?;
        crate::helpers::reduce_with_state(
            slf.as_any(),
            PyTuple::empty(py),
            PyTuple::new(py, [definition])?.into_any(),
        )
    }

    fn __setstate__(&mut self, state: (Bound<'_, PyAny>,)) -> PyResult<()> {
        self.inner = Self::from_definition(&state.0)?;
        Ok(())
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
#[pyclass(module = "molrs.ff", name = "PotentialCompiler", subclass)]
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
            let topo = molrs::Topology::from_frame(core)
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
                    .map(|w| molrs::md::SpecialWeights::new(&w.special_weights(&topo)))
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

/// Read an OpenMM force-field XML file into a :class:`ForceField`.
///
/// Reads OpenMM's own ``<ForceField>`` schema — CHARMM36, AMBER and OPLS-AA
/// ports alike (nm, kJ/mol, radians, ``½k`` harmonic terms) — into the
/// force-field IR, whose definitions follow LAMMPS (Å, kcal/mol, degrees,
/// un-halved ``K``): harmonic bonds and angles, Urey-Bradley
/// (``angle charmm``), periodic and Ryckaert-Bellemans torsions
/// (``dihedral periodic`` / ``multi/harmonic``), periodic impropers (stored in
/// the order OpenMM prices them) and CHARMM's harmonic ``CustomTorsionForce``
/// impropers, CMAP (``cmap charmm``), ``NonbondedForce`` (``lj/cut`` +
/// ``coul/cut``) and ``LennardJonesForce`` with NBFIX and 1-4 parameters
/// (``lj/charmm`` + ``coul/charmm``). Coulomb styles state OpenMM's own
/// constant. A field whose ``lj/charmm`` declares ``one_four="epsilon14"``
/// needs :meth:`ForceField.materialize_one_four` on its frames before it
/// compiles. Distinct from :func:`read_forcefield_xml`, which also reads
/// molrs's own schema.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     Path to an OpenMM force-field XML (``charmm36.xml``, ``oplsaa.xml``, …).
///
/// Returns
/// -------
/// ForceField
///
/// Raises
/// ------
/// ValueError
///     On a malformed document, a section or row with no IR form (named:
///     ``Custom*Force`` other than the harmonic improper, ``<Script>``,
///     ``ordering="smirnoff"``, …), or a missing/non-numeric required
///     attribute (reading is total — never a silent skip).
#[pyfunction]
#[pyo3(name = "read_opls_xml")]
pub fn read_opls_xml_py(path: PathBuf) -> PyResult<PyForceField> {
    use molrs::ff::ForceFieldReader;
    let forcefield = molrs::ff::OplsXmlReader::new()
        .read(path_str(&path)?)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read a LAMMPS force-field include (``*.ff``) into a :class:`ForceField`.
///
/// Parses the ``pair_style``/``pair_coeff`` + ``bond_style``/``angle_style``/
/// ``dihedral_style``/``improper_style`` include that
/// :func:`write_lammps_forcefield` emits. the force-field IR follows the LAMMPS standard, so
/// every coefficient is stored as written (``K``, degrees) and the force field
/// declares the file's ``units``; the ``fourier`` dihedral is molrs's
/// ``periodic``. The ``special_bonds`` line is recorded on the force field.
/// Distinct from :func:`read_forcefield_xml` (molrs's own schema) and
/// :func:`read_opls_xml` (OPLS-AA / GROMACS XML).
///
/// Per-atom charge and mass live in the LAMMPS *data* file, not this include, so
/// they are not read here: Coulomb charges are drawn from the frame at evaluation
/// time and masses are irrelevant to geometry relaxation.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
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
pub fn read_lammps_forcefield_py(path: PathBuf) -> PyResult<PyForceField> {
    use molrs::ff::ForceFieldReader;
    let forcefield = molrs::ff::LammpsFfReader::new()
        .read(path_str(&path)?)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read AMBER prmtop force-field parameter tables into a :class:`ForceField`.
///
/// Structure/connectivity is :func:`molrs.io.read_amber_prmtop`; this parses
/// harmonic bond/angle tables (``k = K``, AMBER's and LAMMPS's form; θ₀ to
/// degrees), periodic dihedrals (phases to degrees), impropers in AMBER's atom
/// order, and LJ A/B → σ/ε, in LAMMPS ``real`` units.
#[pyfunction]
#[pyo3(name = "read_amber_prmtop_ff")]
pub fn read_amber_prmtop_ff_py(path: PathBuf) -> PyResult<PyForceField> {
    let forcefield =
        molrs::ff::read_amber_prmtop_ff(path).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read the force-field directives of a GROMACS topology into a
/// :class:`ForceField`.
///
/// Reads ``[ defaults ]`` (nbfunc 1; comb-rule 2 or 3 → ``mixing``
/// ``arithmetic`` / ``geometric``; gen-pairs, fudgeLJ, fudgeQQ →
/// ``special_bonds``), ``[ atomtypes ]``, ``[ nonbond_params ]`` (explicit
/// cross rows), ``[ pairtypes ]`` (``lj/charmm`` ``epsilon14`` / ``sigma14``,
/// declared ``one_four = "epsilon14"``), ``[ bondtypes ]`` (funct 1, 3),
/// ``[ angletypes ]`` (funct 1, 5 → ``angle charmm``), ``[ dihedraltypes ]``
/// (funct 1, 9 → ``dihedral periodic``; 3 → ``multi/harmonic`` /
/// ``nharmonic``; 5 → ``opls``; 2, 4 → impropers) and ``[ cmaptypes ]``
/// (``cmap charmm``), converting GROMACS units (nm, kJ/mol, ``½k`` harmonic
/// terms) to the force-field IR (LAMMPS standard, ``real``: Å, kcal/mol,
/// ``K = k/2``; degrees stay degrees). See the Force-field IR guide,
/// "GROMACS topologies".
///
/// What the IR cannot hold raises ``ValueError`` naming it: an unsupported
/// function code or comb-rule, ``[ constrainttypes ]``,
/// ``[ implicit_genborn_params ]``, any unknown section, and every molecule
/// section — read a whole topology with :func:`read_gromacs_system`, or skip
/// them here.
///
/// ``include`` follows ``#include`` relative to the including file and then
/// each of ``include_dirs`` (default false: ignored). Each name in
/// ``skip_directives`` (bracket-less, case-insensitive, e.g.
/// ``"constrainttypes"``) is read past, rows and all, instead of refused.
#[pyfunction]
#[pyo3(
    name = "read_gromacs_top_ff",
    signature = (path, include = false, *, include_dirs = Vec::new(), skip_directives = Vec::new())
)]
pub fn read_gromacs_top_ff_py(
    path: PathBuf,
    include: bool,
    include_dirs: Vec<PathBuf>,
    skip_directives: Vec<String>,
) -> PyResult<PyForceField> {
    use molrs::ff::ForceFieldReader;
    let forcefield = gromacs_top_ff_reader(include, &include_dirs, &skip_directives)
        .read(path_str(&path)?)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read a whole GROMACS topology into a :class:`ForceField` and a typed
/// :class:`Frame`.
///
/// The directives as :func:`read_gromacs_top_ff` reads them, and the molecule
/// sections: ``atoms`` (``type``, ``charge``, ``mass``, ``name``, ``res_id``,
/// ``res_name``, ``mol_id``), ``bonds``, ``angles``, ``dihedrals``,
/// ``impropers``, ``cmaps`` (each row's ``type`` the force-field type
/// GROMACS's own lookup picks; a row with parameters of its own gets a type
/// of its own), ``constraints``, ``exclusions`` and ``pairs`` (every
/// intramolecular pair GROMACS prices; the ``[ pairs ]`` rows flagged
/// ``is_14``, with per-pair override columns where they carry parameters).
/// Atom indices are 0-based. ``#include`` is followed, relative to the
/// including file and then each of ``include_dirs`` (GROMACS's
/// ``share/gromacs/top`` for ``#include "charmm27.ff/forcefield.itp"``).
/// Coordinates come from the ``.gro`` (:func:`molrs.io.read_gro`).
///
/// Returns ``(forcefield, frame)``. Anything the IR cannot hold (virtual
/// sites, restraints, free-energy B states, …) raises ``ValueError`` naming
/// it and where.
#[pyfunction]
#[pyo3(
    name = "read_gromacs_system",
    signature = (path, *, include_dirs = Vec::new(), skip_directives = Vec::new())
)]
pub fn read_gromacs_system_py(
    path: PathBuf,
    include_dirs: Vec<PathBuf>,
    skip_directives: Vec<String>,
) -> PyResult<(PyForceField, PyFrame)> {
    let (forcefield, frame) = gromacs_top_ff_reader(true, &include_dirs, &skip_directives)
        .read_system(path_str(&path)?)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((
        PyForceField { inner: forcefield },
        PyFrame::from_core_frame(frame)?,
    ))
}

/// The GROMACS reader the Python entry points configure.
fn gromacs_top_ff_reader(
    include: bool,
    include_dirs: &[PathBuf],
    skip_directives: &[String],
) -> molrs::ff::GromacsTopFfReader {
    let reader = include_dirs.iter().fold(
        molrs::ff::GromacsTopFfReader::new().with_include(include),
        |reader, dir| reader.with_include_dir(dir),
    );
    skip_directives
        .iter()
        .fold(reader, |reader, name| reader.with_skipped_directive(name))
}

/// Write a ForceField as GROMACS force-field directives.
///
/// Writes ``[ defaults ]``, ``[ atomtypes ]``, ``[ nonbond_params ]``,
/// ``[ pairtypes ]``, ``[ bondtypes ]``, ``[ angletypes ]``,
/// ``[ dihedraltypes ]`` and ``[ cmaptypes ]`` in GROMACS units (nm, kJ/mol,
/// degrees) — the inverse of :func:`read_gromacs_top_ff`. No molecule section
/// is written: a force field holds no molecule. A style or parameter GROMACS
/// directives cannot express raises ``ValueError`` naming it. ``precision`` is
/// the number of decimal places for floating coefficients.
#[pyfunction]
#[pyo3(name = "write_gromacs_top_ff", signature = (path, forcefield, precision = 6))]
pub fn write_gromacs_top_ff_py(
    path: PathBuf,
    forcefield: &PyForceField,
    precision: usize,
) -> PyResult<()> {
    use molrs::ff::ForceFieldWriter;
    molrs::ff::GromacsTopFfWriter::new()
        .with_precision(precision)
        .write(&forcefield.inner, path_str(&path)?)
        .map_err(crate::ff::ir::write_err)
}

/// Write a ForceField as an AMBER frcmod file.
///
/// Writes ``MASS``, ``BOND``, ``ANGLE``, ``DIHE``, ``IMPROPER`` and ``NONBON``
/// in AMBER's conventions (``RK = k``, ``TK = k``, degrees — molrs's own —
/// and ``R*/2`` from sigma), so tleap can ``loadamberparams`` it. A style or parameter a frcmod
/// cannot express raises ``ValueError`` naming it.
#[pyfunction]
#[pyo3(name = "write_amber_frcmod", signature = (path, forcefield))]
pub fn write_amber_frcmod_py(path: PathBuf, forcefield: &PyForceField) -> PyResult<()> {
    molrs::ff::write_amber_frcmod(path_str(&path)?, &forcefield.inner)
        .map_err(crate::ff::ir::write_err)
}

/// Write a ForceField to OpenMM force-field XML.
///
/// Every style is written in OpenMM's own schema and units (nm, kJ/mol,
/// radians, ½k harmonic terms), or refused naming the style when OpenMM's
/// tags cannot hold it (see the force-field IR guide, "OpenMM").
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     Output file.
/// forcefield : ForceField
/// precision : int, optional
///     Decimals per number; ``None`` (default) writes each number in the
///     shortest form that reads back to the same float.
///
/// Raises
/// ------
/// ValueError
///     A style, parameter or 1-4 setting with no OpenMM form, named.
#[pyfunction]
#[pyo3(name = "write_forcefield_xml", signature = (path, forcefield, precision = None))]
pub fn write_forcefield_xml_py(
    path: PathBuf,
    forcefield: &PyForceField,
    precision: Option<usize>,
) -> PyResult<()> {
    molrs::ff::write_forcefield_xml(path_str(&path)?, &forcefield.inner, precision)
        .map_err(crate::ff::ir::write_err)
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
/// (every label matched to a type name exactly).
/// Force-field types no label uses are not written.
///
/// Inverse of :func:`read_lammps_forcefield`, and the identity on coefficients:
/// the force-field IR follows the LAMMPS standard. A force field declared in another LAMMPS
/// unit style than ``units`` has its energies and lengths converted through
/// the lj reduced hub — never hard-coded eV/kcal factors. A split ``lj/cut`` +
/// ``coul/cut`` pair is recombined as ``lj/cut/coul/cut`` so geometric mixing
/// is not defeated by a hybrid wildcard. Every style is written through its
/// LAMMPS form in the IR registry (``molrs.ff.ir``, ``StyleInfo.lammps``):
/// the built-ins LAMMPS has (``dihedral periodic`` as ``fourier``,
/// ``improper periodic`` as ``cvff``, the ``class2`` styles with their
/// cross-term lines at zero, …) and a style registered with
/// ``lammps="positional"``, each parameter converted by its dimension.
///
/// Parameters
/// ----------
/// path : str
///     Destination path for the include.
/// forcefield : ForceField
///     Force field, in molrs's (LAMMPS's) convention.
/// frame : Frame
///     The system whose type labels select the coefficients.
/// precision : int, optional
///     Decimal places for floating coefficients (default 6).
/// skip_pair_style : bool, optional
///     When true, omit the ``pair_style`` line: the input script sets its own
///     before the include. Only that line is skipped: ``special_bonds`` and
///     ``pair_modify mix`` / ``shift`` are the force field's and stay
///     (LAMMPS's defaults, ``0 0 0`` and ``geometric`` for ``lj/cut``, are not
///     molrs's); ``pair_modify`` needs a pair style, so read such an include
///     after the input's ``pair_style``.
/// skip_special_bonds : bool, optional
///     When true, omit ``special_bonds``: the input script states its own 1-4
///     weights, which an include read after them would override.
/// skip_units : bool, optional
///     When true, omit the ``units`` line so the include can follow ``units``
///     already set in the input script.
/// units : str, optional
///     LAMMPS ``units`` style for the written file: ``"real"`` (default),
///     ``"metal"``, or ``"lj"``.
/// cmap_file : str, optional
///     The ``fix cmap`` file the include names on its ``fix cmap all cmap
///     <file>`` line (where :func:`write_lammps_cmap` saved it). Required
///     exactly when ``frame`` has a ``cmaps`` block; that fix must reach
///     LAMMPS before ``read_data <data> fix cmap crossterm CMAP``.
///
/// Raises
/// ------
/// ValueError
///     On a frame type label the force field does not define (the message
///     names the block and the label), a style without a LAMMPS form holding
///     a used type ("LAMMPS has no form for …"), a bad units keyword, or
///     missing required parameters.
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
        skip_special_bonds = false,
        skip_units = false,
        units = "real",
        cmap_file = None,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn write_lammps_forcefield_py(
    path: PathBuf,
    forcefield: &PyForceField,
    frame: &PyFrame,
    precision: usize,
    skip_pair_style: bool,
    skip_special_bonds: bool,
    skip_units: bool,
    units: &str,
    cmap_file: Option<String>,
) -> PyResult<()> {
    use molrs::ff::forcefield::lammps_units::parse_style;
    use molrs::ff::{ForceFieldWriter, LammpsFfWriter, LammpsWriteOptions};
    use molrs::store::type_labels::TypeLabels;
    let units = parse_style(units).map_err(pyo3::exceptions::PyValueError::new_err)?;
    frame
        .with_frame(molrs::ff::forcefield::writers::lammps::refuse_pair_overrides)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let labels = frame
        .with_frame(TypeLabels::from_frame)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let writer = LammpsFfWriter::with_options(
        &labels,
        LammpsWriteOptions {
            precision,
            skip_pair_style,
            skip_special_bonds,
            skip_units,
            units,
            cmap_file,
        },
    );
    writer
        .write(&forcefield.inner, path_str(&path)?)
        .map_err(crate::ff::ir::write_err)
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
        skip_special_bonds = false,
        skip_units = false,
        units = "real",
        cmap_file = None,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn write_lammps_forcefield_str_py(
    forcefield: &PyForceField,
    frame: &PyFrame,
    precision: usize,
    skip_pair_style: bool,
    skip_special_bonds: bool,
    skip_units: bool,
    units: &str,
    cmap_file: Option<String>,
) -> PyResult<String> {
    use molrs::ff::forcefield::lammps_units::parse_style;
    use molrs::ff::{ForceFieldWriter, LammpsFfWriter, LammpsWriteOptions};
    use molrs::store::type_labels::TypeLabels;
    let units = parse_style(units).map_err(pyo3::exceptions::PyValueError::new_err)?;
    frame
        .with_frame(molrs::ff::forcefield::writers::lammps::refuse_pair_overrides)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let labels = frame
        .with_frame(TypeLabels::from_frame)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let writer = LammpsFfWriter::with_options(
        &labels,
        LammpsWriteOptions {
            precision,
            skip_pair_style,
            skip_special_bonds,
            skip_units,
            units,
            cmap_file,
        },
    );
    writer
        .write_str(&forcefield.inner)
        .map_err(crate::ff::ir::write_err)
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
    frame
        .with_frame(molrs::ff::forcefield::writers::lammps::refuse_pair_overrides)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
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
        .map_err(crate::ff::ir::write_err)
}

/// Build ``frame``'s ``cmaps`` block from its dihedrals and return the number
/// of CMAP crossterms.
///
/// A crossterm is five atoms ``(a, b, c, d, e)`` whose dihedrals
/// ``(a, b, c, d)`` and ``(b, c, d, e)`` are both rows of ``frame["dihedrals"]``
/// (in either stored direction) and whose ``atoms`` ``type`` labels equal a
/// ``cmap`` row's ``itom`` … ``mtom`` in that order — never reversed. The block
/// (``atomi`` … ``atomm``, ``type`` = the row's name) replaces any ``cmaps``
/// block ``frame`` had, and is removed when nothing matches.
///
/// Raises
/// ------
/// ValueError
///     Cmap rows in two styles, two rows on the same five types, a path that
///     matches one row forward and another backward, or an untyped frame.
#[pyfunction]
#[pyo3(name = "assign_cmaps")]
pub fn assign_cmaps_py(frame: &PyFrame, forcefield: &PyForceField) -> PyResult<usize> {
    frame
        .with_frame_mut(|core| molrs::ff::assign_cmaps(core, &forcefield.inner))?
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Read a LAMMPS ``fix cmap`` file (CHARMM format) into a :class:`ForceField`
/// of one ``cmap charmm`` style.
///
/// Map ``t`` of the file (1-based: the crossterm type a data file's ``CMAP``
/// section gives) is the row named ``"t"``, with the synthetic endpoints
/// ``t-t-t-t-t``, its ``grid`` the 24×24 map, φ-major, as written. The force
/// field is in the file's ``UNITS:`` tag, else ``real``. An incomplete map, a
/// seventh map, or a line running past a map's end raises ``ValueError``
/// (LAMMPS would drop the values).
#[pyfunction]
#[pyo3(name = "read_lammps_cmap")]
pub fn read_lammps_cmap_py(path: PathBuf) -> PyResult<PyForceField> {
    let text = std::fs::read_to_string(&path)
        .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{}: {e}", path.display())))?;
    let forcefield = molrs::ff::LammpsFfReader::new()
        .read_cmap_str(&text)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Write the LAMMPS ``fix cmap`` file of ``frame``'s CMAP crossterms.
///
/// The ``cmaps`` block's type labels are walked in id order — the crossterm
/// types :func:`molrs.io.write_lammps_data` gives the ``CMAP`` section — and
/// each label's ``cmap charmm`` grid is written in CHARMM's layout (a
/// ``# UNITS:`` line, ``# <φ>`` rows of five ``precision``-decimal values),
/// energies converted to ``units``. A CHARMM file read with
/// :func:`read_lammps_cmap` is written back line for line.
///
/// Raises
/// ------
/// ValueError
///     No ``cmaps`` label, a label without a row, a grid that is not 24×24,
///     more than six maps, or a style other than ``charmm``.
#[pyfunction]
#[pyo3(
    name = "write_lammps_cmap",
    signature = (path, forcefield, frame, *, precision = 6, units = "real")
)]
pub fn write_lammps_cmap_py(
    path: PathBuf,
    forcefield: &PyForceField,
    frame: &PyFrame,
    precision: usize,
    units: &str,
) -> PyResult<()> {
    use molrs::ff::forcefield::lammps_units::parse_style;
    use molrs::ff::{LammpsFfWriter, LammpsWriteOptions};
    use molrs::store::type_labels::TypeLabels;
    let units = parse_style(units).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let labels = frame
        .with_frame(TypeLabels::from_frame)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let options = LammpsWriteOptions {
        precision,
        units,
        ..LammpsWriteOptions::default()
    };
    let text = LammpsFfWriter::with_options(&labels, options)
        .write_cmap_str(&forcefield.inner)
        .map_err(crate::ff::ir::write_err)?;
    std::fs::write(&path, text)
        .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{}: {e}", path.display())))
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
