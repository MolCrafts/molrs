//! SMILES (`molrs::io::smiles`): [`PySmilesIr`], the parsed text, and the
//! `molrs.io.read_smiles_str` / `write_smiles_str` doors. SMARTS — parsing,
//! matching, and a pattern written from a molecule — is `molrs.perceive`'s.

use crate::core::molgraph::PyAtomistic;
use crate::error::smiles_error_to_pyerr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

/// Intermediate representation of a parsed SMILES string (or SMILES fragment
/// body).
///
/// This is the raw syntax tree produced by the parser. Convert it to a
/// molecular graph via :meth:`to_atomistic`.
///
/// Attributes
/// ----------
/// n_components : int
///     Number of disconnected components (fragments separated by ``'.'``
///     in the SMILES string).
///
/// Examples
/// --------
/// >>> ir = molrs.io.smiles.SmilesIr("CCO")
/// >>> ir.n_components
/// 1
/// >>> mol = ir.to_atomistic()
/// >>> mol.n_atoms
/// 3
#[pyclass(module = "molrs.io.smiles", name = "SmilesIr")]
pub struct PySmilesIr {
    inner: molrs::io::smiles::SmilesIr,
    input: String,
}

impl PySmilesIr {
    /// Wrap an existing core [`SmilesIr`] as a Python `SmilesIr` object.
    ///
    /// Exists because one binding hands out an IR it did not parse from a
    /// bare SMILES string: `CgFragmentDef.body` (`io::cgsmiles`) returns the
    /// atomistic body a `CGsmiles` fragment table already holds, and the only
    /// alternative — writing that body back to text and re-parsing it — would
    /// make a second parse the price of reading a field.
    ///
    /// `input` is the source text the IR came from; it feeds `__repr__` only
    /// and is never re-parsed. Not a `#[pymethods]` entry, so this adds
    /// nothing to the Python surface — the same shape as `PyLammpsLog::new`
    /// in `io::lammps_log`.
    ///
    /// [`SmilesIr`]: molrs::io::smiles::SmilesIr
    pub(crate) fn from_core(inner: molrs::io::smiles::SmilesIr, input: String) -> Self {
        Self { inner, input }
    }
}

#[pymethods]
impl PySmilesIr {
    /// Parse `smiles` into its intermediate representation.
    ///
    /// Parameters
    /// ----------
    /// smiles : str
    ///     SMILES string (e.g. ``"CCO"`` for ethanol, ``"c1ccccc1"``).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the SMILES string is syntactically invalid.
    ///
    /// Examples
    /// --------
    /// >>> molrs.io.smiles.SmilesIr("CCO").to_atomistic().n_atoms
    /// 3
    #[new]
    fn new(smiles: &str) -> PyResult<Self> {
        let inner = molrs::io::smiles::SmilesIr::parse(smiles).map_err(smiles_error_to_pyerr)?;
        Ok(Self {
            inner,
            input: smiles.to_owned(),
        })
    }

    /// Parse a fragment body — SMILES plus bonding descriptors — into its IR.
    ///
    /// The dialect a ``CGsmiles`` fragment table writes its bodies in:
    /// ``[<]OCC[>]``, ``[$]COC[$]``. Where the plain constructor refuses a
    /// descriptor, this one keeps it, so the IR can become a ported unit
    /// through :meth:`to_template`.
    ///
    /// Parameters
    /// ----------
    /// body : str
    ///     Fragment body, e.g. ``"[<]OCC[>]"``.
    ///
    /// Raises
    /// ------
    /// SmilesError
    ///     (a ``ValueError``) if the body is not valid fragment notation.
    ///
    /// Examples
    /// --------
    /// >>> molrs.io.smiles.SmilesIr.from_fragment("[<]OCC[>]").to_template().n_ports
    /// 2
    #[classmethod]
    fn from_fragment(_cls: &Bound<'_, PyType>, body: &str) -> PyResult<Self> {
        let inner =
            molrs::io::smiles::SmilesIr::from_fragment(body).map_err(smiles_error_to_pyerr)?;
        Ok(Self {
            inner,
            input: body.to_owned(),
        })
    }

    /// Build the ported :class:`~molrs.core.Atomistic` template of this body.
    ///
    /// The one-unit form of :meth:`CgSmilesIr.templates`: the heavy atoms of
    /// the body, plus one capping hydrogen *handle* and one port per bonding
    /// descriptor (``<``, ``>``, ``$``, with its label and bond order). No
    /// coordinates, no ``frag_id``; an IR without descriptors gives a
    /// template without ports.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///
    /// Raises
    /// ------
    /// SmilesError
    ///     (a ``ValueError``) if the body does not convert.
    ///
    /// Examples
    /// --------
    /// >>> eo = molrs.io.smiles.SmilesIr.from_fragment("[<]OCC[>]").to_template()
    /// >>> eo.n_atoms, eo.n_ports
    /// (5, 2)
    fn to_template(&self, py: Python<'_>) -> PyResult<Py<PyAtomistic>> {
        let mol = self.inner.to_template().map_err(smiles_error_to_pyerr)?;
        PyAtomistic::from_core(py, mol)
    }

    /// Number of disconnected molecular components.
    ///
    /// Fragments separated by ``'.'`` in the SMILES string are counted as
    /// separate components.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    fn n_components(&self) -> usize {
        self.inner.components.len()
    }

    /// Convert the SMILES intermediate representation to an all-atom
    /// molecular graph.
    ///
    /// Hydrogen atoms that are implicit in the SMILES string are **not**
    /// added here; use :class:`Conformer` with ``add_hydrogens=True`` for
    /// that.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     Molecular graph with atoms and bonds.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a ring-closure digit is never closed, if the IR holds SMARTS
    ///     query atoms or query bonds (which have no single atomistic
    ///     reading), or if any node carries a bonding descriptor — the mark a
    ///     `CGsmiles` fragment body writes to say where it may be joined.
    ///     This is the plain conversion and it will not drop a descriptor
    ///     silently; build such a body's ported unit with
    ///     :meth:`to_template`, or expand a whole string through
    ///     :meth:`CgSmilesIr.to_atomistic`.
    ///
    /// Examples
    /// --------
    /// >>> mol = molrs.io.smiles.SmilesIr("c1ccccc1").to_atomistic()
    /// >>> mol.n_atoms
    /// 6
    fn to_atomistic(&self, py: Python<'_>) -> PyResult<Py<PyAtomistic>> {
        let mol = self.inner.to_atomistic().map_err(smiles_error_to_pyerr)?;
        PyAtomistic::from_core(py, mol)
    }

    /// One graph per disconnected component, in input order.
    ///
    /// ``to_atomistic`` returns a *single* graph holding every component;
    /// this returns them separately. Splitting happens on the parsed
    /// components, not by cutting the string on ``'.'`` — a separator is only
    /// a separator once the parser says so.
    ///
    /// Returns
    /// -------
    /// list of Atomistic
    ///
    /// Examples
    /// --------
    /// >>> len(molrs.io.smiles.SmilesIr("CCO.O").components())
    /// 2
    fn components(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAtomistic>>> {
        self.inner
            .components
            .iter()
            .map(|chain| {
                let one = molrs::io::smiles::SmilesIr {
                    components: vec![chain.clone()],
                    span: self.inner.span,
                };
                let mol = one.to_atomistic().map_err(smiles_error_to_pyerr)?;
                PyAtomistic::from_core(py, mol)
            })
            .collect()
    }

    /// Build a concrete IR from an :class:`~molrs.core.Atomistic` graph.
    ///
    /// All science/representation choices are **keyword-only flags** forwarded
    /// to ``molrs::io::smiles::SmilesEmitOptions``. This is an *io* alternate
    /// constructor — it does **not** live on ``Atomistic`` (no dependency
    /// inversion into core).
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     Molecular graph to serialise.
    /// canonical : bool, default True
    /// root : int or None
    ///     Optional atom handle to use as the SMILES root.
    /// aromatic : {"as_marked", "kekule_only"}, default "as_marked"
    /// hydrogens : {"organic_subset", "explicit_all", "as_stored"}, default "organic_subset"
    /// include_stereo : bool, default False
    /// multi_component : {"error_if_multiple", "join_dot", "first_only"}, default "error_if_multiple"
    /// organic_subset : bool, default True
    #[classmethod]
    #[pyo3(signature = (
        mol,
        *,
        canonical = true,
        root = None,
        aromatic = "as_marked",
        hydrogens = "organic_subset",
        include_stereo = false,
        multi_component = "error_if_multiple",
        organic_subset = true,
    ))]
    #[allow(clippy::too_many_arguments, reason = "Public Python keyword arguments")]
    fn from_atomistic(
        _cls: &Bound<'_, PyType>,
        mol: &PyAtomistic,
        canonical: bool,
        root: Option<u64>,
        aromatic: &str,
        hydrogens: &str,
        include_stereo: bool,
        multi_component: &str,
        organic_subset: bool,
    ) -> PyResult<Self> {
        let opts = build_smiles_emit_options(
            canonical,
            root,
            aromatic,
            hydrogens,
            include_stereo,
            multi_component,
            organic_subset,
        )?;
        let ir = molrs::io::smiles::SmilesIr::from_atomistic(mol.core(), &opts)
            .map_err(smiles_error_to_pyerr)?;
        let input = molrs::io::write_smiles_str(mol.core(), &opts)
            .unwrap_or_else(|_| "<from_atomistic>".to_owned());
        Ok(Self { inner: ir, input })
    }

    fn __repr__(&self) -> String {
        format!(
            "SmilesIr('{}', components={})",
            self.input,
            self.inner.components.len()
        )
    }
}

fn build_smiles_emit_options(
    canonical: bool,
    root: Option<u64>,
    aromatic: &str,
    hydrogens: &str,
    include_stereo: bool,
    multi_component: &str,
    organic_subset: bool,
) -> PyResult<molrs::io::smiles::SmilesEmitOptions> {
    use molrs::core::node_from_u64;
    use molrs::io::smiles::{AromaticEmit, HydrogenEmit, MultiComponentEmit, SmilesEmitOptions};

    let aromatic = match aromatic {
        "as_marked" => AromaticEmit::AsMarked,
        "kekule_only" => AromaticEmit::KekuleOnly,
        other => {
            return Err(PyValueError::new_err(format!(
                "aromatic must be 'as_marked' or 'kekule_only', got {other:?}"
            )));
        }
    };
    let hydrogens = match hydrogens {
        "organic_subset" => HydrogenEmit::OrganicSubset,
        "explicit_all" => HydrogenEmit::ExplicitAll,
        "as_stored" => HydrogenEmit::AsStored,
        other => {
            return Err(PyValueError::new_err(format!(
                "hydrogens must be 'organic_subset', 'explicit_all', or 'as_stored', got {other:?}"
            )));
        }
    };
    let multi_component = match multi_component {
        "error_if_multiple" => MultiComponentEmit::ErrorIfMultiple,
        "join_dot" => MultiComponentEmit::JoinDot,
        "first_only" => MultiComponentEmit::FirstOnly,
        other => {
            return Err(PyValueError::new_err(format!(
                "multi_component must be 'error_if_multiple', 'join_dot', or 'first_only', got {other:?}"
            )));
        }
    };
    Ok(SmilesEmitOptions {
        canonical,
        root: root.map(node_from_u64),
        aromatic,
        hydrogens,
        include_stereo,
        multi_component,
        organic_subset,
    })
}

/// Read one molecule from a SMILES string.
///
/// Connectivity only: hydrogens implicit in the SMILES are **not** added, and
/// no coordinates are generated. Filling open valences is a perception step
/// (:meth:`molrs.perceive.Perceive.find_hydrogens`), and 3D embedding a
/// conformer step (:mod:`molrs.conformer`).
///
/// Parameters
/// ----------
/// smiles : str
///     A SMILES string naming exactly one connected molecule.
///
/// Returns
/// -------
/// Atomistic
///     The parsed graph.
///
/// Raises
/// ------
/// SmilesError
///     A :class:`ValueError`: if ``smiles`` is syntactically invalid, or names
///     more than one component. A ``'.'``-separated string is a *set* of
///     molecules, not a molecule; take them apart with
///     ``molrs.io.smiles.SmilesIr(s).components()``.
///
/// Examples
/// --------
/// >>> molrs.io.read_smiles_str("CCO").n_atoms
/// 3
#[pyfunction]
pub fn read_smiles_str(py: Python<'_>, smiles: &str) -> PyResult<Py<PyAtomistic>> {
    let mol = molrs::io::read_smiles_str(smiles).map_err(smiles_error_to_pyerr)?;
    PyAtomistic::from_core(py, mol)
}

/// Write a molecule as SMILES text — the inverse of :func:`read_smiles_str`.
///
/// The representation choices are keyword-only flags, the same as
/// :meth:`molrs.io.smiles.SmilesIr.from_atomistic`'s.
///
/// Parameters
/// ----------
/// mol : Atomistic
///     The molecule.
/// canonical : bool, default True
/// root : int or None
///     Optional atom handle to start the string at.
/// aromatic : {"as_marked", "kekule_only"}, default "as_marked"
/// hydrogens : {"organic_subset", "explicit_all", "as_stored"}, default "organic_subset"
/// include_stereo : bool, default False
/// multi_component : {"error_if_multiple", "join_dot", "first_only"}, default "error_if_multiple"
/// organic_subset : bool, default True
///
/// Returns
/// -------
/// str
///
/// Raises
/// ------
/// SmilesError
///     A :class:`ValueError`: an empty molecule, several components under
///     ``"error_if_multiple"``, or an atom the notation cannot write.
///
/// Examples
/// --------
/// >>> molrs.io.write_smiles_str(molrs.io.read_smiles_str("OCC"))
/// 'CCO'
#[pyfunction]
#[pyo3(signature = (
    mol,
    *,
    canonical = true,
    root = None,
    aromatic = "as_marked",
    hydrogens = "organic_subset",
    include_stereo = false,
    multi_component = "error_if_multiple",
    organic_subset = true,
))]
#[allow(clippy::too_many_arguments, reason = "Public Python keyword arguments")]
pub fn write_smiles_str(
    mol: &PyAtomistic,
    canonical: bool,
    root: Option<u64>,
    aromatic: &str,
    hydrogens: &str,
    include_stereo: bool,
    multi_component: &str,
    organic_subset: bool,
) -> PyResult<String> {
    let opts = build_smiles_emit_options(
        canonical,
        root,
        aromatic,
        hydrogens,
        include_stereo,
        multi_component,
        organic_subset,
    )?;
    molrs::io::write_smiles_str(mol.core(), &opts).map_err(smiles_error_to_pyerr)
}

/// Register this module's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "SmilesError",
        m.py().get_type::<crate::error::SmilesError>(),
    )?;
    m.add_class::<PySmilesIr>()?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_smiles_str, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_smiles_str, m)?)?;
    Ok(())
}
