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
//! [`AtdParameterSet::name`] / [`AtdParameterSet::from_name`], and read back
//! out by the `parameter_set` getter.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use molrs::ff::typifier::{AtdBondOrders, AtdParameterSet, AtdTypifier};

use super::PyTypifier;

/// Antechamber atom typifier — `molrs.ff.typifier.AtdTypifier`.
///
/// One rule engine bound to one `ATOMTYPE_*.DEF` table. :meth:`typify` perceives
/// antechamber bond types, derives the facts each rule can ask about, and labels
/// every atom with the first rule of the table that matches it.
///
/// Graph in / graph out: :meth:`typify` returns a clone of ``mol`` with the
/// table's atom type in ``keys.TYPE`` on every atom and the perceived
/// ``bcc_bond_type`` on every bond; the caller's handles still address their
/// atoms on it, and ``mol`` is left untouched — the standard AM1-BCC workflow
/// needs the caller's force-field types *and* these at once. It raises
/// ``ValueError`` if the molecule's facts cannot be derived.
///
/// An atom no rule matches comes back labelled ``"DU"``, the table's own
/// catch-all row; that is antechamber's answer, not a fallback the engine
/// invents. Refusing ``DU`` is the *charge* model's job.
///
/// Parameters
/// ----------
/// parameter_set : str
///     The antechamber ``-at`` flag naming the table: ``"bcc"``, ``"abcg2"``,
///     ``"gas"``, ``"gaff"``, ``"gaff2"``, ``"amber"`` or ``"sybyl"``. Required —
///     there is no default table.
/// bond_orders : str, default ``"perceive"``
///     Which bond orders the types follow. ``"perceive"`` judges them from the
///     connectivity alone, as antechamber does by default (``bondtype -j
///     full``): the molecule's own orders are ignored, and on a molecule with
///     two Kekulé structures (azulene, cyclooctatetraene) the colouring
///     (``cc`` / ``cd`` …) is the one antechamber's search settles on for the
///     same atom and bond order. Every hydrogen must be drawn. ``"input"``
///     keeps the molecule's own orders (aromatic bonds without one are
///     kekulized).
///
/// Raises
/// ------
/// ValueError
///     If ``parameter_set`` names no table, or ``bond_orders`` no source.
///
/// Examples
/// --------
/// >>> typed = molrs.ff.typifier.AtdTypifier(parameter_set="gaff").typify(benzene)
/// >>> typed.get(carbon, molrs.core.keys.TYPE)
/// 'ca'
#[pyclass(module = "molrs.ff.typifier", name = "AtdTypifier", extends = PyTypifier, subclass)]
#[derive(Debug)]
pub struct PyAtdTypifier {
    parameter_set: AtdParameterSet,
    bond_orders: AtdBondOrders,
}

#[pymethods]
impl PyAtdTypifier {
    /// Bind the engine to the table `parameter_set` names.
    #[new]
    #[pyo3(signature = (*, parameter_set, bond_orders = "perceive"))]
    fn new(parameter_set: &str, bond_orders: &str) -> PyResult<(Self, PyTypifier)> {
        let parameter_set =
            AtdParameterSet::from_name(parameter_set).map_err(PyValueError::new_err)?;
        let bond_orders = AtdBondOrders::from_name(bond_orders).map_err(PyValueError::new_err)?;
        Ok((
            Self {
                parameter_set,
                bond_orders,
            },
            PyTypifier::native(AtdTypifier::new(parameter_set).with_bond_orders(bond_orders)),
        ))
    }

    /// The antechamber ``-at`` flag of the table this typifier walks.
    #[getter]
    fn parameter_set(&self) -> &'static str {
        self.parameter_set.name()
    }

    /// Which bond orders the types follow: ``"perceive"`` or ``"input"``.
    #[getter]
    fn bond_orders(&self) -> &'static str {
        AtdBondOrders::name(self.bond_orders)
    }

    fn __repr__(&self) -> String {
        format!(
            "AtdTypifier(parameter_set='{}', bond_orders='{}')",
            self.parameter_set(),
            self.bond_orders()
        )
    }
}
