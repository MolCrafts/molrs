//! Python binding for the ATD atom typifier (`molrs::ff::typifier::atd`).
//!
//! One rule engine, seven `ATOMTYPE_*.DEF` tables. The table is chosen by name at
//! construction and there is **no default**: seven exist, they disagree, and picking
//! one silently would be picking for the caller.
//!
//! # The names are antechamber's `-at` flags, not the tables' file names
//!
//! `antechamber -at gaff` walks `ATOMTYPE_GFF.DEF`, and the Rust enum is named after
//! the *table* ([`AtdParameterSet::Gff`]) while the acceptance contract words the
//! Python argument as the *flag* (`AtdTypifier(parameter_set="gaff")`). The two
//! spellings differ for exactly the two GAFF columns, which is precisely where a
//! wrong-set binding would hide — so the mapping is written down once, in
//! [`parameter_set_from_name`], and read back out by the `parameter_set` getter.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use molrs::ff::typifier::Typifier;
use molrs::ff::typifier::atd::{AtdParameterSet, AtdTypifier};

use crate::core::system::molgraph::PyAtomistic;
use crate::ff::PyTypifier;

/// The `-at` flag of every table, paired with the set that walks it.
///
/// The one place the flag ↔ table mapping lives. Iterated (rather than matched twice)
/// so that the parser and the getter cannot drift apart, and so the error message
/// lists exactly the names the parser accepts.
const PARAMETER_SETS: &[(&str, AtdParameterSet)] = &[
    ("bcc", AtdParameterSet::Bcc),
    ("abcg2", AtdParameterSet::Abcg2),
    ("gas", AtdParameterSet::Gas),
    ("gaff", AtdParameterSet::Gff),
    ("gaff2", AtdParameterSet::Gff2),
    ("amber", AtdParameterSet::Amber),
    ("sybyl", AtdParameterSet::Sybyl),
];

/// The parameter set named by an antechamber `-at` flag.
///
/// # Errors
///
/// `ValueError` — an unknown name. Never a fallback to a default table: an atom type
/// from the wrong table is a plausible-looking answer, which is the failure mode this
/// refusal exists to prevent.
pub(crate) fn parameter_set_from_name(name: &str) -> PyResult<AtdParameterSet> {
    PARAMETER_SETS
        .iter()
        .find(|(flag, _)| *flag == name)
        .map(|(_, set)| *set)
        .ok_or_else(|| {
            let known: Vec<&str> = PARAMETER_SETS.iter().map(|(flag, _)| *flag).collect();
            PyValueError::new_err(format!(
                "unknown atom-type parameter set {name:?}; expected one of {}",
                known.join(", ")
            ))
        })
}

/// The `-at` flag of a parameter set — the inverse of [`parameter_set_from_name`].
fn parameter_set_name(set: AtdParameterSet) -> &'static str {
    PARAMETER_SETS
        .iter()
        .find(|(_, known)| *known == set)
        .map(|(flag, _)| *flag)
        .unwrap_or("")
}

/// Antechamber atom typifier — `molrs.AtdTypifier`.
///
/// One rule engine bound to one `ATOMTYPE_*.DEF` table. :meth:`typify` perceives
/// antechamber bond types, derives the facts each rule can ask about, and labels
/// every atom with the first rule of the table that matches it.
///
/// Parameters
/// ----------
/// parameter_set : str
///     The antechamber ``-at`` flag naming the table: ``"bcc"``, ``"abcg2"``,
///     ``"gas"``, ``"gaff"``, ``"gaff2"``, ``"amber"`` or ``"sybyl"``. Required —
///     there is no default table.
///
/// Raises
/// ------
/// ValueError
///     If ``parameter_set`` names no table.
///
/// Examples
/// --------
/// >>> typed = molrs.AtdTypifier(parameter_set="gaff").typify(benzene)
/// >>> typed.get(carbon, molrs.keys.TYPE)
/// 'ca'
#[pyclass(module = "molrs.ff.typifier", name = "AtdTypifier", extends = PyTypifier)]
#[derive(Debug)]
pub struct PyAtdTypifier {
    inner: AtdTypifier,
}

#[pymethods]
impl PyAtdTypifier {
    /// Bind the engine to the table `parameter_set` names.
    #[new]
    #[pyo3(signature = (*, parameter_set))]
    fn new(parameter_set: &str) -> PyResult<(Self, PyTypifier)> {
        Ok((
            Self {
                inner: AtdTypifier::new(parameter_set_from_name(parameter_set)?),
            },
            PyTypifier,
        ))
    }

    /// The antechamber ``-at`` flag of the table this typifier walks.
    #[getter]
    fn parameter_set(&self) -> &'static str {
        parameter_set_name(self.inner.parameter_set())
    }

    /// Assign atom types to a molecular graph.
    ///
    /// Graph in / graph out: the caller's handles still address their atoms on the
    /// returned clone, and the input's ``type`` column is left alone — the standard
    /// AM1-BCC workflow needs the caller's force-field types *and* these at once.
    ///
    /// An atom no rule matches comes back labelled ``"DU"``, the table's own
    /// catch-all row; that is antechamber's answer, not a fallback the engine
    /// invents. Refusing ``DU`` is the *charge* model's job.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecule to type; left untouched.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     A clone of ``mol`` with the table's atom type in ``keys.TYPE`` on every
    ///     atom, and the perceived ``bcc_bond_type`` on every bond.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the molecule's facts cannot be derived.
    fn typify(&self, py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
        let typed = self
            .inner
            .typify(mol.core())
            .map_err(PyValueError::new_err)?;
        PyAtomistic::from_core(py, typed)
    }

    fn __repr__(&self) -> String {
        format!("AtdTypifier(parameter_set='{}')", self.parameter_set())
    }
}
