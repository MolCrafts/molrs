//! Style and type handles over a [`PyForceField`].
//!
//! A handle is the owning force field plus the identifiers of one style or
//! one type, and nothing else: every read and write goes through the one
//! native [`molrs::ff::ForceField`]. `ForceField.def_style` returns the
//! category's style handle ([`PyAtomStyle`] … [`PyPairStyle`]); its typed
//! ``def_type`` door defines a type and returns the type's handle
//! ([`PyAtomType`] … [`PyPairType`]).

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple, PyType};

use molrs::ff::forcefield::Style;

use super::{PyForceField, params_from_dict, params_to_dict};
use crate::helpers::py_value_err;

/// The six categories and their handle classes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Category {
    Atom,
    Bond,
    Angle,
    Dihedral,
    Improper,
    Pair,
}

impl Category {
    pub(crate) const ALL: [Self; 6] = [
        Self::Atom,
        Self::Bond,
        Self::Angle,
        Self::Dihedral,
        Self::Improper,
        Self::Pair,
    ];

    pub(crate) fn of(name: &str) -> PyResult<Self> {
        Self::ALL
            .into_iter()
            .find(|category| category.name() == name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown force-field category '{name}'")))
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Atom => "atom",
            Self::Bond => "bond",
            Self::Angle => "angle",
            Self::Dihedral => "dihedral",
            Self::Improper => "improper",
            Self::Pair => "pair",
        }
    }

    fn style_class<'py>(self, py: Python<'py>) -> Bound<'py, PyType> {
        match self {
            Self::Atom => py.get_type::<PyAtomStyle>(),
            Self::Bond => py.get_type::<PyBondStyle>(),
            Self::Angle => py.get_type::<PyAngleStyle>(),
            Self::Dihedral => py.get_type::<PyDihedralStyle>(),
            Self::Improper => py.get_type::<PyImproperStyle>(),
            Self::Pair => py.get_type::<PyPairStyle>(),
        }
    }

    fn type_class<'py>(self, py: Python<'py>) -> Bound<'py, PyType> {
        match self {
            Self::Atom => py.get_type::<PyAtomType>(),
            Self::Bond => py.get_type::<PyBondType>(),
            Self::Angle => py.get_type::<PyAngleType>(),
            Self::Dihedral => py.get_type::<PyDihedralType>(),
            Self::Improper => py.get_type::<PyImproperType>(),
            Self::Pair => py.get_type::<PyPairType>(),
        }
    }

    /// The categories `selector` names: a category name its own; a style or
    /// type class its category; the base ``Style`` / ``Type`` every one.
    ///
    /// # Errors
    ///
    /// `ValueError` for an unknown category name, `TypeError` when
    /// `selector` is neither a name nor a style or type class.
    pub(crate) fn selected(selector: &Bound<'_, PyAny>) -> PyResult<Vec<Self>> {
        let py = selector.py();
        if let Ok(name) = selector.extract::<&str>() {
            return Ok(vec![Self::of(name)?]);
        }
        if selector.is(py.get_type::<PyStyle>()) || selector.is(py.get_type::<PyFfType>()) {
            return Ok(Self::ALL.to_vec());
        }
        Self::ALL
            .into_iter()
            .find(|category| {
                selector.is(category.style_class(py)) || selector.is(category.type_class(py))
            })
            .map(|category| vec![category])
            .ok_or_else(|| {
                PyTypeError::new_err(format!(
                    "expected a category name or a Style / Type class, got {}",
                    selector.repr().map(|r| r.to_string()).unwrap_or_default()
                ))
            })
    }

    pub(crate) fn style_handle(
        self,
        py: Python<'_>,
        ff: &Py<PyForceField>,
        name: &str,
    ) -> PyResult<Py<PyAny>> {
        let base = PyStyle {
            ff: ff.clone_ref(py),
            category: self,
            name: name.to_owned(),
        };
        let init = PyClassInitializer::from(base);
        Ok(match self {
            Self::Atom => Py::new(py, init.add_subclass(PyAtomStyle {}))?.into_any(),
            Self::Bond => Py::new(py, init.add_subclass(PyBondStyle {}))?.into_any(),
            Self::Angle => Py::new(py, init.add_subclass(PyAngleStyle {}))?.into_any(),
            Self::Dihedral => Py::new(py, init.add_subclass(PyDihedralStyle {}))?.into_any(),
            Self::Improper => Py::new(py, init.add_subclass(PyImproperStyle {}))?.into_any(),
            Self::Pair => Py::new(py, init.add_subclass(PyPairStyle {}))?.into_any(),
        })
    }

    fn type_handle(
        self,
        py: Python<'_>,
        ff: &Py<PyForceField>,
        style: Option<&str>,
        name: &str,
    ) -> PyResult<Py<PyAny>> {
        let base = PyFfType {
            ff: ff.clone_ref(py),
            category: self,
            style: style.map(str::to_owned),
            name: name.to_owned(),
        };
        let init = PyClassInitializer::from(base);
        Ok(match self {
            Self::Atom => Py::new(py, init.add_subclass(PyAtomType {}))?.into_any(),
            Self::Bond => Py::new(py, init.add_subclass(PyBondType {}))?.into_any(),
            Self::Angle => Py::new(py, init.add_subclass(PyAngleType {}))?.into_any(),
            Self::Dihedral => Py::new(py, init.add_subclass(PyDihedralType {}))?.into_any(),
            Self::Improper => Py::new(py, init.add_subclass(PyImproperType {}))?.into_any(),
            Self::Pair => Py::new(py, init.add_subclass(PyPairType {}))?.into_any(),
        })
    }
}

fn missing_style(category: Category, name: &str) -> PyErr {
    PyValueError::new_err(format!("no {} style named '{name}'", category.name()))
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

/// Handle of one style of a :class:`ForceField`.
///
/// Made by ``ForceField.def_style`` / ``get_style`` / ``styles``; its
/// category's subclass (``BondStyle``, …) carries the typed ``def_type``.
/// Two handles are equal when they name the same category and style of the
/// same force field.
#[pyclass(module = "molrs.ff", name = "Style", frozen, subclass)]
pub struct PyStyle {
    ff: Py<PyForceField>,
    category: Category,
    name: String,
}

impl PyStyle {
    fn with_style<R>(&self, py: Python<'_>, f: impl FnOnce(&Style) -> R) -> PyResult<R> {
        let ff = self.ff.bind(py).try_borrow()?;
        let style = ff
            .inner
            .get_style(self.category.name(), &self.name)
            .ok_or_else(|| missing_style(self.category, &self.name))?;
        Ok(f(style))
    }

    fn type_handles(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let names: Vec<String> = self.with_style(py, |style| {
            style
                .type_rows()
                .into_iter()
                .map(|(name, _, _)| name.to_owned())
                .collect()
        })?;
        names
            .iter()
            .map(|name| {
                self.category
                    .type_handle(py, &self.ff, Some(&self.name), name)
            })
            .collect()
    }

    /// The one door every typed ``def_type`` goes through: define `name` on
    /// `endpoints` (``AtomType`` handles) with `params`.
    fn define(
        &self,
        py: Python<'_>,
        name: &str,
        endpoints: &[&Bound<'_, PyAny>],
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let mut ends = Vec::with_capacity(endpoints.len());
        for end in endpoints {
            let atom = end.cast::<PyAtomType>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "{} type '{name}': an endpoint is an AtomType handle, got {} {}",
                    self.category.name(),
                    end.get_type()
                        .name()
                        .map(|n| n.to_string())
                        .unwrap_or_default(),
                    end.repr().map(|r| r.to_string()).unwrap_or_default()
                ))
            })?;
            ends.push(atom.as_super().get().name.clone());
        }
        let params = params_from_dict(params)?;
        let stored = {
            let mut ff = self.ff.bind(py).try_borrow_mut()?;
            let style = ff
                .inner
                .get_style_mut(self.category.name(), &self.name)
                .ok_or_else(|| missing_style(self.category, &self.name))?;
            let ends: Vec<&str> = ends.iter().map(String::as_str).collect();
            style.def_type(name, &ends, params).map_err(py_value_err)?;
            // A pair restating a stored pair under another name is that row.
            match style.type_params(name) {
                Some(_) => name.to_owned(),
                None => style
                    .get_pairtype(ends[0], ends.get(1).copied())
                    .map(|row| row.name.clone())
                    .expect("def_type stored the row or found the pair it restates"),
            }
        };
        self.category
            .type_handle(py, &self.ff, Some(&self.name), &stored)
    }
}

#[pymethods]
impl PyStyle {
    /// The style name (``"harmonic"``, ``"lj/cut"``, …).
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// The category (``"atom"``, ``"bond"``, …, ``"pair"``).
    #[getter]
    fn category(&self) -> &'static str {
        self.category.name()
    }

    /// Every type of this style, in definition order.
    #[getter(types)]
    fn every_type(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.type_handles(py)
    }

    /// The types of this style that are instances of `type_cls` (every type
    /// when it is ``None``).
    #[pyo3(signature = (type_cls = None))]
    fn get_types(
        &self,
        py: Python<'_>,
        type_cls: Option<&Bound<'_, PyType>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let types = self.type_handles(py)?;
        let Some(cls) = type_cls else {
            return Ok(types);
        };
        let mut kept = Vec::with_capacity(types.len());
        for handle in types {
            if handle.bind(py).is_instance(cls)? {
                kept.push(handle);
            }
        }
        Ok(kept)
    }

    /// The type named `name`, or ``None``.
    fn get_type_by_name(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
        let defined = self.with_style(py, |style| style.type_endpoints(name).is_some())?;
        if !defined {
            return Ok(None);
        }
        self.category
            .type_handle(py, &self.ff, Some(&self.name), name)
            .map(Some)
    }

    /// The style-level params (e.g. a pair style's ``cutoff``, ``mixing``),
    /// as a new dict.
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.with_style(py, |style| params_to_dict(py, style.params()))?
    }

    /// One style-level param, or ``None``.
    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.params(py)?.get_item(key)
    }

    /// Set one style-level param: a number (how a caller declares a cutoff)
    /// or a string (``s["mixing"] = "geometric"``).
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``value`` is neither a number nor a str.
    fn __setitem__(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut ff = self.ff.bind(py).try_borrow_mut()?;
        let style = ff
            .inner
            .get_style_mut(self.category.name(), &self.name)
            .ok_or_else(|| missing_style(self.category, &self.name))?;
        if let Ok(text) = value.cast::<PyString>() {
            style.set_str_param(key, text.to_str()?);
        } else if let Ok(number) = value.extract::<f64>() {
            style.set_param(key, number);
        } else {
            return Err(PyTypeError::new_err(format!(
                "param '{key}' must be a number or a str, got {}",
                value.get_type().name()?
            )));
        }
        Ok(())
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (self.ff.as_ptr() as usize, self.category.name(), &self.name).hash(&mut hasher);
        hasher.finish()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<PyStyle>().is_ok_and(|other| {
            let other = other.get();
            other.ff.as_ptr() == self.ff.as_ptr()
                && other.category == self.category
                && other.name == self.name
        })
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(format!("<{}: {}>", slf.get_type().name()?, slf.get().name))
    }
}

/// The atom style: ``def_type(name, **params)``.
#[pyclass(module = "molrs.ff", name = "AtomStyle", extends = PyStyle, frozen, subclass)]
pub struct PyAtomStyle {}

#[pymethods]
impl PyAtomStyle {
    /// Define the atom type ``name`` with ``params`` (numbers and strings,
    /// e.g. ``mass=12.011, element="C"``) and return its handle — the
    /// endpoint the bonded and pair styles take.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     On a conflicting re-definition.
    #[pyo3(signature = (name, **params))]
    fn def_type(
        slf: &Bound<'_, Self>,
        name: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        slf.as_super().get().define(slf.py(), name, &[], params)
    }
}

/// The bond style: ``def_type(name, itom, jtom, **params)``.
#[pyclass(module = "molrs.ff", name = "BondStyle", extends = PyStyle, frozen, subclass)]
pub struct PyBondStyle {}

#[pymethods]
impl PyBondStyle {
    /// Define the bond type ``name`` between the atom types ``itom`` and
    /// ``jtom`` with ``params`` and return its handle. The name is stored as
    /// given and never read for endpoints.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If an endpoint is not an ``AtomType``.
    /// ValueError
    ///     On a conflicting re-definition.
    #[pyo3(signature = (name, itom, jtom, **params))]
    fn def_type(
        slf: &Bound<'_, Self>,
        name: &str,
        itom: &Bound<'_, PyAny>,
        jtom: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        slf.as_super()
            .get()
            .define(slf.py(), name, &[itom, jtom], params)
    }
}

/// The angle style: ``def_type(name, itom, jtom, ktom, **params)``.
#[pyclass(module = "molrs.ff", name = "AngleStyle", extends = PyStyle, frozen, subclass)]
pub struct PyAngleStyle {}

#[pymethods]
impl PyAngleStyle {
    /// Define the angle type ``name`` on ``itom``–``jtom``–``ktom``
    /// (``jtom`` the vertex) with ``params`` and return its handle.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If an endpoint is not an ``AtomType``.
    /// ValueError
    ///     On a conflicting re-definition.
    #[pyo3(signature = (name, itom, jtom, ktom, **params))]
    fn def_type(
        slf: &Bound<'_, Self>,
        name: &str,
        itom: &Bound<'_, PyAny>,
        jtom: &Bound<'_, PyAny>,
        ktom: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        slf.as_super()
            .get()
            .define(slf.py(), name, &[itom, jtom, ktom], params)
    }
}

/// The dihedral style: ``def_type(name, itom, jtom, ktom, ltom, **params)``.
#[pyclass(module = "molrs.ff", name = "DihedralStyle", extends = PyStyle, frozen, subclass)]
pub struct PyDihedralStyle {}

#[pymethods]
impl PyDihedralStyle {
    /// Define the dihedral type ``name`` on ``itom``–``jtom``–``ktom``–``ltom``
    /// with ``params`` and return its handle.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If an endpoint is not an ``AtomType``.
    /// ValueError
    ///     On a conflicting re-definition.
    #[pyo3(signature = (name, itom, jtom, ktom, ltom, **params))]
    fn def_type(
        slf: &Bound<'_, Self>,
        name: &str,
        itom: &Bound<'_, PyAny>,
        jtom: &Bound<'_, PyAny>,
        ktom: &Bound<'_, PyAny>,
        ltom: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        slf.as_super()
            .get()
            .define(slf.py(), name, &[itom, jtom, ktom, ltom], params)
    }
}

/// The improper style: ``def_type(name, itom, jtom, ktom, ltom, **params)``.
#[pyclass(module = "molrs.ff", name = "ImproperStyle", extends = PyStyle, frozen, subclass)]
pub struct PyImproperStyle {}

#[pymethods]
impl PyImproperStyle {
    /// Define the improper type ``name`` on ``itom``, ``jtom``, ``ktom``,
    /// ``ltom`` in the style's slot order with ``params`` and return its
    /// handle.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If an endpoint is not an ``AtomType``.
    /// ValueError
    ///     On a conflicting re-definition.
    #[pyo3(signature = (name, itom, jtom, ktom, ltom, **params))]
    fn def_type(
        slf: &Bound<'_, Self>,
        name: &str,
        itom: &Bound<'_, PyAny>,
        jtom: &Bound<'_, PyAny>,
        ktom: &Bound<'_, PyAny>,
        ltom: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        slf.as_super()
            .get()
            .define(slf.py(), name, &[itom, jtom, ktom, ltom], params)
    }
}

/// The pair style: ``def_type(name, itom, jtom=None, **params)``.
#[pyclass(module = "molrs.ff", name = "PairStyle", extends = PyStyle, frozen, subclass)]
pub struct PyPairStyle {}

#[pymethods]
impl PyPairStyle {
    /// Define the pair type ``name`` between ``itom`` and ``jtom`` with
    /// ``params`` and return its handle; ``jtom=None`` is the self pair of
    /// ``itom``.
    ///
    /// A pair is its two atom types in either order, so a style holds one
    /// row per pair: restating a stored pair under any name with equal
    /// parameters (annotations such as ``desc`` aside) stores nothing and
    /// returns the stored row's handle.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If an endpoint is not an ``AtomType``.
    /// ValueError
    ///     On a conflicting re-definition, or a restatement of a stored pair
    ///     with different parameters.
    #[pyo3(signature = (name, itom, jtom = None, **params))]
    fn def_type(
        slf: &Bound<'_, Self>,
        name: &str,
        itom: &Bound<'_, PyAny>,
        jtom: Option<&Bound<'_, PyAny>>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let style = slf.as_super().get();
        match jtom {
            None => style.define(slf.py(), name, &[itom], params),
            Some(jtom) => style.define(slf.py(), name, &[itom, jtom], params),
        }
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Handle of one type of a :class:`ForceField`.
///
/// Its params read as a mapping (``t["k"]``, ``"k" in t``, ``t.keys()``) and
/// write one at a time (``t["k"] = 300.0``, ``t["element"] = "C"``).
/// ``endpoints`` are the ``AtomType`` handles it is defined on. Two handles
/// are equal when they name the same category, style and type of the same
/// force field.
#[pyclass(module = "molrs.ff", name = "Type", frozen, subclass)]
pub struct PyFfType {
    ff: Py<PyForceField>,
    category: Category,
    /// `None` for an endpoint atom type no atom style defines.
    style: Option<String>,
    name: String,
}

impl PyFfType {
    fn params_of<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let Some(style) = &self.style else {
            return Ok(PyDict::new(py));
        };
        let ff = self.ff.bind(py).try_borrow()?;
        let style = ff
            .inner
            .get_style(self.category.name(), style)
            .ok_or_else(|| missing_style(self.category, style))?;
        match style.type_params(&self.name) {
            Some(params) => params_to_dict(py, params),
            None => Ok(PyDict::new(py)),
        }
    }

    fn endpoint(slf: &Bound<'_, Self>, index: usize) -> PyResult<Py<PyAny>> {
        Ok(Self::endpoints(slf)?
            .bind(slf.py())
            .get_item(index)?
            .unbind())
    }
}

#[pymethods]
impl PyFfType {
    /// The type name — the label a Frame's ``type`` column carries.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// The category (``"atom"``, ``"bond"``, …, ``"pair"``).
    #[getter]
    fn category(&self) -> &'static str {
        self.category.name()
    }

    /// The type's params (numbers and strings), as a new dict.
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.params_of(py)
    }

    /// One param, or ``None``.
    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.params_of(py)?.get_item(key)
    }

    /// One param, or `default`.
    #[pyo3(signature = (key, default = None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        key: &str,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(self.params_of(py)?.get_item(key)?.or(default))
    }

    fn __contains__(&self, py: Python<'_>, key: &str) -> PyResult<bool> {
        self.params_of(py)?.contains(key)
    }

    /// Set one param: a number, or a string (``t["element"] = "C"``).
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``value`` is neither a number nor a str.
    /// ValueError
    ///     If the type is an endpoint no atom style defines.
    fn __setitem__(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let Some(style) = &self.style else {
            return Err(PyValueError::new_err(format!(
                "atom type '{}' is defined by no atom style; it has no params to set",
                self.name
            )));
        };
        let mut ff = self.ff.bind(py).try_borrow_mut()?;
        let style = ff
            .inner
            .get_style_mut(self.category.name(), style)
            .ok_or_else(|| missing_style(self.category, style))?;
        let set = if let Ok(text) = value.cast::<PyString>() {
            style.set_type_str_param(&self.name, key, text.to_str()?)
        } else if let Ok(number) = value.extract::<f64>() {
            style.set_type_param(&self.name, key, number)
        } else {
            return Err(PyTypeError::new_err(format!(
                "param '{key}' must be a number or a str, got {}",
                value.get_type().name()?
            )));
        };
        if !set {
            return Err(PyValueError::new_err(format!(
                "no {} type named '{}'",
                self.category.name(),
                self.name
            )));
        }
        Ok(())
    }

    /// The param names.
    fn keys(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.params_of(py)?.keys().extract()
    }

    /// ``(name, value)`` param pairs.
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
        self.params_of(py)?.items().extract()
    }

    /// The ``AtomType`` handles this type is defined on (none for an atom
    /// type; two for a pair type, a self pair naming one atom type twice).
    #[getter]
    fn endpoints(slf: &Bound<'_, Self>) -> PyResult<Py<PyTuple>> {
        let py = slf.py();
        let this = slf.get();
        let Some(style) = &this.style else {
            return Ok(PyTuple::empty(py).unbind());
        };
        let ff = this.ff.bind(py).try_borrow()?;
        let names = ff
            .inner
            .get_style(this.category.name(), style)
            .and_then(|style| style.type_endpoints(&this.name))
            .unwrap_or_default();
        // Each endpoint resolves to the atom style that defines it.
        let atom_styles: Vec<&Style> = ff.inner.get_styles(Category::Atom.name());
        let resolved: Vec<(Option<String>, String)> = names
            .into_iter()
            .map(|name| {
                let style = atom_styles
                    .iter()
                    .find(|style| style.type_endpoints(&name).is_some())
                    .map(|style| style.name().to_owned());
                (style, name)
            })
            .collect();
        drop(ff);
        let handles = resolved
            .iter()
            .map(|(style, name)| Category::Atom.type_handle(py, &this.ff, style.as_deref(), name))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, handles)?.unbind())
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (
            self.ff.as_ptr() as usize,
            self.category.name(),
            &self.style,
            &self.name,
        )
            .hash(&mut hasher);
        hasher.finish()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<PyFfType>().is_ok_and(|other| {
            let other = other.get();
            other.ff.as_ptr() == self.ff.as_ptr()
                && other.category == self.category
                && other.style == self.style
                && other.name == self.name
        })
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(format!("<{}: {}>", slf.get_type().name()?, slf.get().name))
    }
}

/// An atom type; the endpoint every other type is defined on.
#[pyclass(module = "molrs.ff", name = "AtomType", extends = PyFfType, frozen, subclass)]
pub struct PyAtomType {}

/// The `itom` / `jtom` endpoint accessors of a type class.
macro_rules! first_two_endpoints {
    ($ty:ty) => {
        #[pymethods]
        impl $ty {
            /// The first endpoint atom type.
            #[getter]
            fn itom(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                PyFfType::endpoint(slf.as_super(), 0)
            }

            /// The second endpoint atom type.
            #[getter]
            fn jtom(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                PyFfType::endpoint(slf.as_super(), 1)
            }
        }
    };
}

/// The `ktom` endpoint accessor of a type class with three or more endpoints.
macro_rules! third_endpoint {
    ($ty:ty) => {
        #[pymethods]
        impl $ty {
            /// The third endpoint atom type.
            #[getter]
            fn ktom(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                PyFfType::endpoint(slf.as_super(), 2)
            }
        }
    };
}

/// The `ltom` endpoint accessor of a type class with four endpoints.
macro_rules! fourth_endpoint {
    ($ty:ty) => {
        #[pymethods]
        impl $ty {
            /// The fourth endpoint atom type.
            #[getter]
            fn ltom(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                PyFfType::endpoint(slf.as_super(), 3)
            }
        }
    };
}

/// A bond type.
#[pyclass(module = "molrs.ff", name = "BondType", extends = PyFfType, frozen, subclass)]
pub struct PyBondType {}
first_two_endpoints!(PyBondType);

/// An angle type (``jtom`` the vertex).
#[pyclass(module = "molrs.ff", name = "AngleType", extends = PyFfType, frozen, subclass)]
pub struct PyAngleType {}
first_two_endpoints!(PyAngleType);
third_endpoint!(PyAngleType);

/// A dihedral type.
#[pyclass(module = "molrs.ff", name = "DihedralType", extends = PyFfType, frozen, subclass)]
pub struct PyDihedralType {}
first_two_endpoints!(PyDihedralType);
third_endpoint!(PyDihedralType);
fourth_endpoint!(PyDihedralType);

/// An improper type.
#[pyclass(module = "molrs.ff", name = "ImproperType", extends = PyFfType, frozen, subclass)]
pub struct PyImproperType {}
first_two_endpoints!(PyImproperType);
third_endpoint!(PyImproperType);
fourth_endpoint!(PyImproperType);

/// A pair type.
#[pyclass(module = "molrs.ff", name = "PairType", extends = PyFfType, frozen, subclass)]
pub struct PyPairType {}
first_two_endpoints!(PyPairType);
