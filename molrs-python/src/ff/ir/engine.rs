//! `molrs.ff.ir`'s engine forms (`ff-ir-02-protocol` §8): a style's
//! LAMMPS form, from Python.
//!
//! * `register_style(..., lammps="positional")` (and `ir.StyleSpec`'s
//!   `lammps = "positional"`) gives a style the positional LAMMPS form: its
//!   `params` in order on the `*_coeff` line, each converted by its
//!   dimension; `"positional:<name>"` writes it under the LAMMPS style
//!   `<name>`; `None` (the default) gives it none, so LAMMPS refuses it by
//!   name ([`NoEngineFormError`](crate::ff::ir::errors::NoEngineFormError));
//! * `register_engine_form(engine, category, name, form)` gives a style
//!   registered without one its form afterwards;
//! * `StyleSpec.lammps` names a style's form: `"positional"`,
//!   `"positional:<name>"`, `"custom:<name>"` (a codec of its own, the
//!   built-ins whose LAMMPS line is not positional) or `None`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use molrs::ff::ir::{self as rir, Engine, LammpsForm, StyleSpec};

use super::refuse;

/// The LAMMPS form a Python `lammps=` names: `None`, `"positional"` or
/// `"positional:<name>"`.
pub(crate) fn lammps_form(value: Option<&str>) -> PyResult<LammpsForm> {
    match value {
        None => Ok(LammpsForm::None),
        Some("positional") => Ok(LammpsForm::positional()),
        Some(v) => match v.strip_prefix("positional:") {
            Some(name) if !name.is_empty() => Ok(LammpsForm::positional_named(name.to_owned())),
            _ => Err(PyValueError::new_err(format!(
                "lammps={v:?}: None, \"positional\" or \"positional:<LAMMPS style name>\""
            ))),
        },
    }
}

/// How `spec`'s LAMMPS form is spelled in Python.
pub(crate) fn lammps_form_name(spec: &StyleSpec) -> Option<String> {
    match &spec.lammps {
        LammpsForm::None => None,
        LammpsForm::Positional { name: None } => Some("positional".into()),
        LammpsForm::Positional { name: Some(n) } => Some(format!("positional:{n}")),
        LammpsForm::Custom(_) => spec.lammps.lammps_name(spec).map(|n| format!("custom:{n}")),
    }
}

/// Give a registered style an engine form: today a LAMMPS form for a style
/// registered without one.
///
/// Parameters
/// ----------
/// engine : str
///     ``"lammps"`` (``"openmm"``, ``"gromacs"``, ``"prmtop"``,
///     ``"frcmod"`` take no registered form and raise ``NoEngineForm``: the
///     OpenMM XML writer writes an expression style's ``Custom*Force`` from
///     its expression, GROMACS and AMBER hold built-in styles only).
/// category, name : str
///     The registered style.
/// form : str
///     ``"positional"`` or ``"positional:<LAMMPS style name>"``.
///
/// Raises
/// ------
/// NoKernelError
///     The style is not registered.
/// NoEngineFormError
///     Another engine, or a positional form the style's spec cannot have (a
///     Text, Array or indexed parameter, a style parameter other than
///     ``cutoff`` / ``mixing``, a category without a ``*_style`` command).
/// Sealed, Conflict
///     The style is built in, or has another form already.
///
/// Examples
/// --------
/// >>> ir.register_style("bond", "fene/doc", params={"k": "E/L^2", "r0": "L",
/// ...     "epsilon": "E", "sigma": "L"},
/// ...     expression="-0.5*k*r0^2*log(1-(r/r0)^2)")
/// >>> ir.register_engine_form("lammps", "bond", "fene/doc", "positional:fene")
/// >>> [s.lammps for s in ir.styles("bond") if s.name == "fene/doc"]
/// ['positional:fene']
/// >>> ir.unregister_style("bond", "fene/doc")
#[pyfunction]
pub(crate) fn register_engine_form(
    engine: &str,
    category: &str,
    name: &str,
    form: &str,
) -> PyResult<()> {
    let engine = Engine::parse(engine).map_err(PyValueError::new_err)?;
    let form = lammps_form(Some(form))?;
    rir::register_engine_form(engine, category, name, form).map_err(refuse)
}

/// Register the engine-form functions on `molrs.ff.ir`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(register_engine_form, m)?)?;
    Ok(())
}
