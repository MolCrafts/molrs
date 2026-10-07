//! `molrs.ff.ir`: the force-field IR's registry, from Python.
//!
//! The IR is a protocol (`molrs::ff::ir`): a category, a style with ordered
//! parameters and their dimensions, and an energy given as an expression or
//! a kernel. This module lets Python register into the process-wide
//! registry every compile reads, with nothing rebuilt:
//!
//! * `Param` — one parameter (`ParamSpec`), its `Dim` parsed at once;
//! * `register_category` / `register_style` / `unregister` — the registry
//!   calls, refusing what does not conform with an `IrError` subclass named
//!   after the Rust variant (D17);
//! * Python callables as Tier-2 kernels ([`PyScalarKernel`],
//!   [`PyCompoundKernel`]): one call per style per evaluation, an exception
//!   or a wrongly shaped result parked in the kernel error slot and
//!   re-raised by whatever drove the evaluation (`Potentials.calc_*`, a
//!   compile, an integrator);
//! * `styles` / `categories` / `evaluate` — introspection, and a style's
//!   form evaluated on a batch of coordinates.
//!
//! The Python face (`StyleSpec` with `__init_subclass__`, the docs) is
//! `python/molrs/ff/ir.py`; a style's engine forms are [`engine`]'s.

mod engine;

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use ndarray::{Array3, ArrayD, Axis};
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use molrs::ff::forcefield::DefError;
use molrs::ff::ir::ExpressionForm;
use molrs::ff::ir::{
    self as rir, Arity, CategorySpec, Coordinate, Dim, EndpointOrder, IrError, Kernel, Mix,
    ParamKind, ParamSource, ParamSpec, Registry, Sample, SpecialClass, StyleSpec, Value,
};
use molrs::ff::potential::CompileError;
use molrs::ff::potential::generic::{CompoundForm, ParamCols, ScalarForm};
use molrs::op::F;

use crate::ff::potential::ErrSlot;

// ---------------------------------------------------------------------------
// Errors: `IrError(ValueError)` and one subclass per Rust variant (D17)
// ---------------------------------------------------------------------------

/// The exception classes of `molrs.ff.ir`.
pub mod errors {
    use pyo3::create_exception;
    use pyo3::prelude::*;

    create_exception!(
        molrs.ff.ir,
        IrError,
        pyo3::exceptions::PyValueError,
        "The force-field IR refused a category, a style, a kernel or a term. \
         Each refusal is raised as the subclass named after the Rust \
         `molrs::ff::ir::IrError` variant (`Sealed`, `NoKernel`, …), its \
         message naming the item; the variant's fields are attributes \
         (`err.category`, `err.style`, `err.param`, …). Subclasses \
         `ValueError`."
    );

    macro_rules! variants {
        ($($name:ident: $doc:literal),* $(,)?) => {
            $(create_exception!(molrs.ff.ir, $name, IrError, $doc);)*

            /// Add `IrError` and every variant class to `m`.
            pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
                let py = m.py();
                m.add("IrError", py.get_type::<IrError>())?;
                $(m.add(stringify!($name), py.get_type::<$name>())?;)*
                Ok(())
            }
        };
    }

    variants! {
        UnknownCategory: "A style in a category neither built in nor registered.",
        BadName: "A category or parameter name outside its pattern.",
        Arity: "A custom category's arity outside 2..=5, or a type with the wrong number of endpoints.",
        BlockName: "A custom category whose block is not `<name>s`.",
        ReservedParam: "A parameter named like a structural column, a variable or a pair input.",
        DuplicateParam: "A parameter declared twice.",
        Dim: "An unparsable or forbidden dimension.",
        Parse: "An expression that does not parse, or whose definitions are out of order.",
        UnboundVariable: "A name an expression reads that is neither a variable of the category nor a declared numeric parameter.",
        UnknownFunction: "A function an expression calls that the expression language does not have.",
        FunctionArity: "A function called with the wrong number of arguments.",
        Point: "A point `pk` beyond the category's arity, or any point in a pair style.",
        CoordinateMismatch: "A kernel tier the category cannot take.",
        Derivative: "A kernel's derivative against a central difference of its energy, beyond relative 1e-6.",
        Disagree: "An expression against the kernel beside it, beyond relative 1e-10.",
        Asymmetric: "A pair energy that changes when its two atoms are exchanged.",
        Sealed: "A built-in, which cannot be overridden or removed.",
        Conflict: "A different registration under a taken name.",
        NoKernel: "Nothing can price the style: no kernel and no expression.",
        NoMixing: "An unlike pair with no cross row stating a parameter that does not mix.",
        MissingParam: "A row (or the style) without a value its kernel needs, and no default.",
        BadValue: "A row's (or the style's) value of a declared parameter that is not of its declared kind or domain: text for a number, an array of another rank, text outside its choices.",
        KernelShape: "A kernel's output of the wrong shape or dtype, or a Python kernel that raised (the exception is the `__cause__`).",
        NoEngineForm: "An engine with no form for the style.",
        FormConflict: "A form family without exactly one canonical style.",
        NoForm: "A style that belongs to no form family.",
        OutOfImage: "An exact form conversion refused: a row outside the image of the target style.",
        Malformed: "A spec whose declarations contradict each other.",
    }
}

/// `e` as its `molrs.ff.ir` exception, carrying `message` and the
/// variant's fields as attributes.
pub(crate) fn ir_err(e: &IrError, message: String) -> PyErr {
    use errors as x;
    let s = |v: &str| Field::Str(v.to_owned());
    let n = |v: usize| Field::Int(v);
    let f = |v: F| Field::Float(v);
    let (err, fields): (PyErr, Vec<(&str, Field)>) = match e {
        IrError::UnknownCategory { category } => (
            x::UnknownCategory::new_err(message),
            vec![("category", s(category))],
        ),
        IrError::BadName { what, name } => (
            x::BadName::new_err(message),
            vec![("what", s(what)), ("name", s(name))],
        ),
        IrError::Arity { category, arity } => (
            x::Arity::new_err(message),
            vec![("category", s(category)), ("arity", n(*arity))],
        ),
        IrError::BlockName { category, block } => (
            x::BlockName::new_err(message),
            vec![("category", s(category)), ("block", s(block))],
        ),
        IrError::ReservedParam { style, param } => (
            x::ReservedParam::new_err(message),
            vec![("style", s(style)), ("param", s(param))],
        ),
        IrError::DuplicateParam { style, param } => (
            x::DuplicateParam::new_err(message),
            vec![("style", s(style)), ("param", s(param))],
        ),
        IrError::Dim { param, dim, reason } => (
            x::Dim::new_err(message),
            vec![("param", s(param)), ("dim", s(dim)), ("reason", s(reason))],
        ),
        IrError::Parse {
            expression,
            at,
            reason,
        } => (
            x::Parse::new_err(message),
            vec![
                ("expression", s(expression)),
                ("at", n(*at)),
                ("reason", s(reason)),
            ],
        ),
        IrError::UnboundVariable { style, name } => (
            x::UnboundVariable::new_err(message),
            vec![("style", s(style)), ("name", s(name))],
        ),
        IrError::UnknownFunction { name } => (
            x::UnknownFunction::new_err(message),
            vec![("name", s(name))],
        ),
        IrError::FunctionArity {
            name,
            given,
            expected,
        } => (
            x::FunctionArity::new_err(message),
            vec![
                ("name", s(name)),
                ("given", n(*given)),
                ("expected", n(*expected)),
            ],
        ),
        IrError::Point {
            style,
            point,
            arity,
        } => (
            x::Point::new_err(message),
            vec![
                ("style", s(style)),
                ("point", s(point)),
                ("arity", n(*arity)),
            ],
        ),
        IrError::CoordinateMismatch { category, kernel } => (
            x::CoordinateMismatch::new_err(message),
            vec![("category", s(category)), ("kernel", s(kernel))],
        ),
        IrError::Derivative { style, at, rel } => (
            x::Derivative::new_err(message),
            vec![("style", s(style)), ("at", s(at)), ("rel", f(*rel))],
        ),
        IrError::Disagree { style, at, rel } => (
            x::Disagree::new_err(message),
            vec![("style", s(style)), ("at", s(at)), ("rel", f(*rel))],
        ),
        IrError::Asymmetric { style } => {
            (x::Asymmetric::new_err(message), vec![("style", s(style))])
        }
        IrError::Sealed { category, style } => (
            x::Sealed::new_err(message),
            vec![("category", s(category)), ("style", s(style))],
        ),
        IrError::Conflict { category, style } => (
            x::Conflict::new_err(message),
            vec![("category", s(category)), ("style", s(style))],
        ),
        IrError::NoKernel { category, style } => (
            x::NoKernel::new_err(message),
            vec![("category", s(category)), ("style", s(style))],
        ),
        IrError::NoMixing { style, param, pair } => (
            x::NoMixing::new_err(message),
            vec![("style", s(style)), ("param", s(param)), ("pair", s(pair))],
        ),
        IrError::MissingParam {
            style,
            type_,
            param,
        } => (
            x::MissingParam::new_err(message),
            vec![("style", s(style)), ("type", s(type_)), ("param", s(param))],
        ),
        IrError::BadValue {
            style,
            type_,
            param,
            reason,
        } => (
            x::BadValue::new_err(message),
            vec![
                ("style", s(style)),
                ("type", s(type_)),
                ("param", s(param)),
                ("reason", s(reason)),
            ],
        ),
        IrError::KernelShape { style, reason } => (
            x::KernelShape::new_err(message),
            vec![("style", s(style)), ("reason", s(reason))],
        ),
        IrError::NoEngineForm {
            engine,
            category,
            style,
            reason,
        } => (
            x::NoEngineForm::new_err(message),
            vec![
                ("engine", s(engine)),
                ("category", s(category)),
                ("style", s(style)),
                ("reason", s(reason)),
            ],
        ),
        IrError::FormConflict { family, reason } => (
            x::FormConflict::new_err(message),
            vec![("family", s(family)), ("reason", s(reason))],
        ),
        IrError::NoForm { category, style } => (
            x::NoForm::new_err(message),
            vec![("category", s(category)), ("style", s(style))],
        ),
        IrError::OutOfImage {
            from,
            to,
            type_,
            reason,
        } => (
            x::OutOfImage::new_err(message),
            vec![
                ("from_", s(from)),
                ("to", s(to)),
                ("type", s(type_)),
                ("reason", s(reason)),
            ],
        ),
        IrError::Malformed { style, reason } => (
            x::Malformed::new_err(message),
            vec![("style", s(style)), ("reason", s(reason))],
        ),
    };
    Python::attach(|py| {
        let value = err.value(py);
        for (name, field) in fields {
            // An attribute that cannot be set leaves the message, which
            // names the item too.
            let _ = match field {
                Field::Str(v) => value.setattr(name, v),
                Field::Int(v) => value.setattr(name, v),
                Field::Float(v) => value.setattr(name, v),
            };
        }
    });
    err
}

/// A field of an [`IrError`] variant, as a Python attribute.
enum Field {
    Str(String),
    Int(usize),
    Float(F),
}

/// `e` as its `molrs.ff.ir` exception, with its own message.
pub(crate) fn refuse(e: IrError) -> PyErr {
    let message = e.to_string();
    ir_err(&e, message)
}

/// A compile's error as a Python exception: a Python kernel's parked
/// exception first (it is the cause of whatever followed), else an IR
/// refusal as its `IrError` subclass, else a plain `ValueError`.
pub(crate) fn compile_err(e: CompileError) -> PyErr {
    if let Err(parked) = take_kernel_err() {
        return parked;
    }
    match e {
        CompileError::Ir(e) => refuse(e),
        CompileError::Invalid(message) => PyValueError::new_err(message),
        other @ CompileError::NoBox { .. } => PyValueError::new_err(other.to_string()),
    }
}

/// A writer's error: its typed refusal (`NoEngineForm`) as the `IrError`
/// subclass, anything else a `ValueError`.
pub(crate) fn write_err(e: molrs::io::forcefield::writers::WriteError) -> PyErr {
    match e.ir() {
        Some(refusal) => ir_err(refusal, e.to_string()),
        None => PyValueError::new_err(e.to_string()),
    }
}

/// A force-field definition error as a Python exception: the IR refusal
/// it is (`Arity`, `UnknownCategory`) as its subclass, with the
/// definition's own message; else a plain `ValueError`.
pub(crate) fn def_err(e: DefError) -> PyErr {
    match e.ir() {
        Some(refusal) => ir_err(&refusal, e.to_string()),
        None => PyValueError::new_err(e.to_string()),
    }
}

// ---------------------------------------------------------------------------
// The kernel error slot
// ---------------------------------------------------------------------------

/// Where every Python kernel parks the first exception (or shape error) of
/// an evaluation: the `Potential` trait has no error channel, so the kernel
/// returns NaNs and the Python-facing caller re-raises from here.
fn kernel_slot() -> &'static ErrSlot {
    static SLOT: OnceLock<ErrSlot> = OnceLock::new();
    SLOT.get_or_init(ErrSlot::default)
}

/// The kernel error slot, for a caller's list of slots to check after an
/// evaluation (`Potentials`, the integrators).
pub(crate) fn kernel_err_slot() -> ErrSlot {
    Arc::clone(kernel_slot())
}

/// Re-raise the parked kernel exception, clearing the slot.
pub(crate) fn take_kernel_err() -> PyResult<()> {
    crate::ff::potential::take_err(std::slice::from_ref(kernel_slot()))
}

/// Forget a parked exception nobody re-raised (before a fresh compile or
/// registration, so a stale one is not blamed on it).
pub(crate) fn clear_kernel_err() {
    kernel_slot().lock().expect("error slot poisoned").take();
}

fn park(err: PyErr) {
    let mut slot = kernel_slot().lock().expect("error slot poisoned");
    if slot.is_none() {
        *slot = Some(err);
    }
}

// ---------------------------------------------------------------------------
// Python callables as Tier-2 kernels
// ---------------------------------------------------------------------------

/// The non-numeric inputs a Python kernel receives by name: array and text
/// per-type parameters, text style parameters. (Numeric ones are every
/// column the generic kernel supplies.)
#[derive(Clone, Debug, Default)]
struct Extras {
    arrays: Vec<String>,
    texts: Vec<String>,
    style_texts: Vec<String>,
}

impl Extras {
    fn of(spec: &StyleSpec) -> Self {
        let mut out = Self::default();
        for p in &spec.params {
            match p.kind {
                ParamKind::Array { .. } => out.arrays.push(p.name.to_string()),
                ParamKind::Text { .. } => out.texts.push(p.name.to_string()),
                ParamKind::Scalar => {}
            }
        }
        for p in &spec.style_params {
            if let ParamKind::Text { .. } = p.kind {
                out.style_texts.push(p.name.to_string());
            }
        }
        out
    }

    /// The keyword arguments of one call: numeric columns as `ndarray[n]`,
    /// arrays `[n, S…]`, text per-type `list[str]`, text style `str`.
    fn kwargs<'py>(&self, py: Python<'py>, p: &ParamCols<'_>) -> PyResult<Bound<'py, PyDict>> {
        let kw = PyDict::new(py);
        for name in p.names() {
            if let Some(col) = p.get(name) {
                kw.set_item(name, PyArray1::from_slice(py, col))?;
            }
        }
        for name in &self.arrays {
            if let Some(a) = p.array(name) {
                kw.set_item(name, a.to_pyarray(py))?;
            }
        }
        for name in &self.texts {
            if let Some(t) = p.text(name) {
                kw.set_item(name, PyList::new(py, t.iter().copied())?)?;
            }
        }
        for name in &self.style_texts {
            if let Some(t) = p.style_text(name) {
                kw.set_item(name, t)?;
            }
        }
        Ok(kw)
    }
}

/// `KernelShape` for `style`.
fn shape_err(style: &str, reason: String) -> PyErr {
    refuse(IrError::KernelShape {
        style: style.to_owned(),
        reason,
    })
}

/// A Python kernel's exception as `KernelShape`, the original its cause.
fn raised(py: Python<'_>, style: &str, err: PyErr) -> PyErr {
    let kind = err
        .get_type(py)
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "Exception".into());
    let wrapped = shape_err(
        style,
        format!("the Python kernel raised {kind}: {}", err.value(py)),
    );
    wrapped.set_cause(py, Some(err));
    wrapped
}

/// `shape` as Python prints it: `(2,)`, `(2, 3, 3)`.
fn py_shape(shape: &[usize]) -> String {
    match shape {
        [n] => format!("({n},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// What `obj` is, for a shape error.
fn describe(obj: &Bound<'_, PyAny>) -> String {
    match obj.cast::<numpy::PyUntypedArray>() {
        Ok(a) => format!(
            "an ndarray of dtype {} and shape {}",
            a.dtype(),
            py_shape(a.shape())
        ),
        Err(_) => obj
            .get_type()
            .name()
            .map(|n| format!("a {n}"))
            .unwrap_or_else(|_| "an object".into()),
    }
}

/// The two outputs of a kernel call, `(first, second)`.
fn two<'py>(
    style: &str,
    out: &Bound<'py, PyAny>,
    spelling: &str,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    out.cast::<PyTuple>()
        .ok()
        .filter(|t| t.len() == 2)
        .map(|t| (t.get_item(0), t.get_item(1)))
        .and_then(|(a, b)| Some((a.ok()?, b.ok()?)))
        .ok_or_else(|| {
            shape_err(
                style,
                format!(
                    "the kernel must return a tuple {spelling}, got {}",
                    describe(out)
                ),
            )
        })
}

/// A kernel output `what`: a float64 ndarray of exactly `shape`.
fn output(style: &str, what: &str, obj: &Bound<'_, PyAny>, shape: &[usize]) -> PyResult<Vec<F>> {
    let wrong = || {
        shape_err(
            style,
            format!(
                "`{what}` must be a float64 ndarray of shape {}, got {}",
                py_shape(shape),
                describe(obj)
            ),
        )
    };
    let arr = obj.cast::<PyArrayDyn<F>>().map_err(|_| wrong())?;
    if arr.shape() != shape {
        return Err(wrong());
    }
    Ok(arr.readonly().as_array().iter().copied().collect())
}

/// A Python callable `f(q, **params) -> (e, de_dq)` as a [`ScalarForm`].
struct PyScalarKernel {
    func: Py<PyAny>,
    style: String,
    extras: Extras,
}

impl PyScalarKernel {
    fn call(&self, py: Python<'_>, q: &[F], p: &ParamCols<'_>) -> PyResult<(Vec<F>, Vec<F>)> {
        let n = q.len();
        let kw = self.extras.kwargs(py, p)?;
        let out = self
            .func
            .bind(py)
            .call((PyArray1::from_slice(py, q),), Some(&kw))
            .map_err(|e| raised(py, &self.style, e))?;
        let (e, de) = two(&self.style, &out, "(e, de_dq)")?;
        Ok((
            output(&self.style, "e", &e, &[n])?,
            output(&self.style, "de_dq", &de, &[n])?,
        ))
    }
}

impl ScalarForm for PyScalarKernel {
    fn eval(&self, q: &[F], p: &ParamCols<'_>, e: &mut [F], de_dq: &mut [F]) {
        Python::attach(|py| match self.call(py, q, p) {
            Ok((ev, dv)) => {
                e.copy_from_slice(&ev);
                de_dq.copy_from_slice(&dv);
            }
            Err(err) => {
                park(err);
                e.fill(F::NAN);
                de_dq.fill(F::NAN);
            }
        })
    }
}

/// A Python callable `f(x, **params) -> (e, grad)` as a [`CompoundForm`],
/// `x` and `grad` of shape `(n, arity, 3)`.
struct PyCompoundKernel {
    func: Py<PyAny>,
    style: String,
    extras: Extras,
}

impl PyCompoundKernel {
    fn call(
        &self,
        py: Python<'_>,
        x: &[[F; 3]],
        arity: usize,
        p: &ParamCols<'_>,
    ) -> PyResult<(Vec<F>, Vec<F>)> {
        let n = x.len() / arity.max(1);
        let flat: Vec<F> = x.iter().flatten().copied().collect();
        let pos = Array3::from_shape_vec((n, arity, 3), flat)
            .expect("n_terms · arity points")
            .into_pyarray(py);
        let kw = self.extras.kwargs(py, p)?;
        let out = self
            .func
            .bind(py)
            .call((pos,), Some(&kw))
            .map_err(|e| raised(py, &self.style, e))?;
        let (e, grad) = two(&self.style, &out, "(e, grad)")?;
        Ok((
            output(&self.style, "e", &e, &[n])?,
            output(&self.style, "grad", &grad, &[n, arity, 3])?,
        ))
    }
}

impl CompoundForm for PyCompoundKernel {
    fn eval(
        &self,
        x: &[[F; 3]],
        arity: usize,
        p: &ParamCols<'_>,
        e: &mut [F],
        grad: &mut [[F; 3]],
    ) {
        Python::attach(|py| match self.call(py, x, arity, p) {
            Ok((ev, gv)) => {
                e.copy_from_slice(&ev);
                let (points, _) = gv.as_chunks::<3>();
                grad.copy_from_slice(points);
            }
            Err(err) => {
                park(err);
                e.fill(F::NAN);
                grad.fill([F::NAN; 3]);
            }
        })
    }
}

/// The Python callables registered as kernels, by `(category, style)`: an
/// identical re-registration (the same callable, the same spec) reuses the
/// kernel, so the registry sees the same kernel and makes it a no-op.
type PyKernels = Mutex<HashMap<(String, String), (Py<PyAny>, Kernel)>>;

fn py_kernels() -> &'static PyKernels {
    static KERNELS: OnceLock<PyKernels> = OnceLock::new();
    KERNELS.get_or_init(Default::default)
}

// ---------------------------------------------------------------------------
// Param
// ---------------------------------------------------------------------------

/// `text` as the [`Dim`] of parameter `param`, refused as `IrError.Dim`.
fn parse_dim(param: &str, text: &str) -> PyResult<Dim> {
    let dim_err = |reason: String| {
        refuse(IrError::Dim {
            param: param.to_owned(),
            dim: text.to_owned(),
            reason,
        })
    };
    let dim: Dim = text.parse().map_err(dim_err)?;
    dim.check().map_err(dim_err)?;
    Ok(dim)
}

fn parse_mix(mix: Option<&Bound<'_, PyAny>>) -> PyResult<Mix> {
    let Some(mix) = mix.filter(|m| !m.is_none()) else {
        return Ok(Mix::None);
    };
    if let Ok(rule) = mix.extract::<String>() {
        return match rule.as_str() {
            "none" => Ok(Mix::None),
            "arithmetic" => Ok(Mix::Arithmetic),
            "geometric" => Ok(Mix::Geometric),
            _ => Err(PyValueError::new_err(format!(
                "mix={rule:?}: one of None, 'arithmetic', 'geometric', \
                 ('lj_epsilon', <sigma>), ('lj_sigma', <epsilon>)"
            ))),
        };
    }
    let (rule, partner): (String, String) = mix.extract().map_err(|_| {
        PyTypeError::new_err(
            "mix: None, 'arithmetic', 'geometric', ('lj_epsilon', <sigma>) or \
             ('lj_sigma', <epsilon>)",
        )
    })?;
    match rule.as_str() {
        "lj_epsilon" => Ok(Mix::LjEpsilon {
            sigma: partner.into(),
        }),
        "lj_sigma" => Ok(Mix::LjSigma {
            epsilon: partner.into(),
        }),
        _ => Err(PyValueError::new_err(format!(
            "mix=({rule:?}, {partner:?}): the joint rule is 'lj_epsilon' or 'lj_sigma'"
        ))),
    }
}

fn mix_to_py<'py>(py: Python<'py>, mix: &Mix) -> PyResult<Bound<'py, PyAny>> {
    Ok(match mix {
        Mix::None => py.None().into_bound(py),
        Mix::Arithmetic => PyString::new(py, "arithmetic").into_any(),
        Mix::Geometric => PyString::new(py, "geometric").into_any(),
        Mix::LjEpsilon { sigma } => ("lj_epsilon", sigma.as_ref()).into_pyobject(py)?.into_any(),
        Mix::LjSigma { epsilon } => ("lj_sigma", epsilon.as_ref()).into_pyobject(py)?.into_any(),
    })
}

fn value_to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Num(x) => x.into_pyobject(py)?.into_any(),
        Value::Text(t) => PyString::new(py, t).into_any(),
    })
}

/// A number or a string, as a [`Value`].
fn value_of(what: &str, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::Text(s.into()));
    }
    obj.extract::<F>()
        .map(Value::Num)
        .map_err(|_| PyTypeError::new_err(format!("{what}: a number or a string")))
}

/// One parameter of a force-field IR style: its name, its dimension, its
/// kind, its default, its mixing rule (pair styles) and whether it is an
/// indexed family.
///
/// Parameters
/// ----------
/// name : str
///     ``^[A-Za-z_][A-Za-z0-9_]*$``, not a reserved name (checked when the
///     style registers).
/// dim : str, default "1"
///     The dimension, as exponents of ``E`` (energy), ``L`` (length), ``A``
///     (angle), ``Q`` (charge) and ``M`` (mass): ``"E/L^2"``, ``"E*L^6"``,
///     ``"A"`` (an angle value, stored in degrees), ``"E/A^2"`` (per
///     radian), ``"1"``. Refused at once as :class:`Dim`.
/// kind : {"scalar", "array", "text"}, default "scalar"
///     One number per row; an ``f64`` array of ``rank`` per row (not an
///     expression variable); a string, one of ``choices`` when given.
/// rank : int, optional
///     The array rank (``kind="array"`` only).
/// choices : list of str, optional
///     The allowed strings (``kind="text"`` only).
/// default : float or str, optional
///     The value a row (or the style) without the parameter takes; none
///     makes it required.
/// mix : None, "arithmetic", "geometric", ("lj_epsilon", sigma) or ("lj_sigma", epsilon)
///     How a pair style's parameter combines from the two self rows when no
///     cross row states it.
/// indexed : bool, default False
///     A numbered family ``<name>1 … <name>M``.
///
/// Examples
/// --------
/// >>> from molrs.ff import ir
/// >>> ir.Param("k", "E/L^2")
/// Param('k', 'E/L^2')
/// >>> ir.Param("epsilon", "E", mix=("lj_epsilon", "sigma")).mix
/// ('lj_epsilon', 'sigma')
/// >>> ir.Param("k", "E/L^^2")
/// Traceback (most recent call last):
///     ...
/// molrs.ff.ir.Dim: parameter `k`: dimension "E/L^^2": ...
#[pyclass(
    module = "molrs.ff.ir",
    name = "Param",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Clone, PartialEq)]
pub struct PyParam {
    inner: ParamSpec,
}

#[pymethods]
impl PyParam {
    #[new]
    #[pyo3(signature = (name, dim="1", *, kind="scalar", rank=None, choices=None, default=None, mix=None, indexed=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        dim: &str,
        kind: &str,
        rank: Option<u8>,
        choices: Option<Vec<String>>,
        default: Option<Bound<'_, PyAny>>,
        mix: Option<Bound<'_, PyAny>>,
        indexed: bool,
    ) -> PyResult<Self> {
        let dim = parse_dim(&name, dim)?;
        let bad = |why: String| PyValueError::new_err(format!("Param {name:?}: {why}"));
        let kind = match (kind, rank, &choices) {
            ("scalar", None, None) => ParamKind::Scalar,
            ("array", Some(rank @ 1..), None) => ParamKind::Array { rank },
            ("array", _, None) => return Err(bad("an array parameter needs rank >= 1".into())),
            ("text", None, _) => ParamKind::Text {
                choices: choices
                    .clone()
                    .map(|c| c.into_iter().map(Cow::Owned).collect()),
            },
            ("scalar" | "array" | "text", ..) => {
                return Err(bad(format!(
                    "rank= is for kind='array' and choices= for kind='text', not kind={kind:?}"
                )));
            }
            _ => {
                return Err(bad(format!(
                    "kind={kind:?}: one of 'scalar', 'array', 'text'"
                )));
            }
        };
        let default = match default.filter(|d| !d.is_none()) {
            None => None,
            Some(d) => Some(value_of("default", &d)?),
        };
        match (&kind, &default) {
            (_, None) | (ParamKind::Scalar, Some(Value::Num(_))) => {}
            (ParamKind::Text { choices }, Some(Value::Text(t))) => {
                if choices.as_ref().is_some_and(|c| !c.contains(t)) {
                    return Err(bad(format!("default {t:?} is not one of its choices")));
                }
            }
            _ => return Err(bad("the default does not fit the parameter's kind".into())),
        }
        let mut inner = ParamSpec::new(name, dim)
            .kind(kind)
            .mix(parse_mix(mix.as_ref())?);
        inner.default = default;
        inner.indexed = indexed;
        Ok(Self { inner })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// The dimension, printed in its canonical spelling (``"E/L^2"``).
    #[getter]
    fn dim(&self) -> String {
        self.inner.dim.to_string()
    }

    /// ``"scalar"``, ``"array"`` or ``"text"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            ParamKind::Scalar => "scalar",
            ParamKind::Array { .. } => "array",
            ParamKind::Text { .. } => "text",
        }
    }

    #[getter]
    fn rank(&self) -> Option<u8> {
        match self.inner.kind {
            ParamKind::Array { rank } => Some(rank),
            _ => None,
        }
    }

    #[getter]
    fn choices(&self) -> Option<Vec<String>> {
        match &self.inner.kind {
            ParamKind::Text { choices } => choices
                .as_ref()
                .map(|c| c.iter().map(|s| s.to_string()).collect()),
            _ => None,
        }
    }

    #[getter]
    fn default<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.inner
            .default
            .as_ref()
            .map(|v| value_to_py(py, v))
            .transpose()
    }

    #[getter]
    fn mix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        mix_to_py(py, &self.inner.mix)
    }

    #[getter]
    fn indexed(&self) -> bool {
        self.inner.indexed
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mut out = format!("Param({:?}, {:?}", self.inner.name, self.dim());
        match &self.inner.kind {
            ParamKind::Scalar => {}
            ParamKind::Array { rank } => out.push_str(&format!(", kind='array', rank={rank}")),
            ParamKind::Text { choices } => {
                out.push_str(", kind='text'");
                if let Some(c) = choices {
                    out.push_str(&format!(", choices={c:?}"));
                }
            }
        }
        if let Some(d) = &self.inner.default {
            out.push_str(&format!(", default={}", value_to_py(py, d)?.repr()?));
        }
        if self.inner.mix != Mix::None {
            out.push_str(&format!(
                ", mix={}",
                mix_to_py(py, &self.inner.mix)?.repr()?
            ));
        }
        if self.inner.indexed {
            out.push_str(", indexed=True");
        }
        out.push(')');
        Ok(out.replace('"', "'"))
    }
}

/// `params=`: a list of :class:`Param`, or a ``{name: dim}`` dict of
/// required scalars.
fn parse_params(arg: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<ParamSpec>> {
    let Some(arg) = arg.filter(|a| !a.is_none()) else {
        return Ok(Vec::new());
    };
    if let Ok(dict) = arg.cast::<PyDict>() {
        return dict
            .iter()
            .map(|(k, v)| {
                let name: String = k.extract()?;
                let dim: String = v.extract().map_err(|_| {
                    PyTypeError::new_err(format!(
                        "params: {{name: dim}} takes a dimension string for {name:?}"
                    ))
                })?;
                Ok(ParamSpec::new(name.clone(), parse_dim(&name, &dim)?))
            })
            .collect();
    }
    arg.try_iter()?
        .map(|item| {
            let item = item?;
            item.cast::<PyParam>()
                .map(|p| p.get().inner.clone())
                .map_err(|_| {
                    PyTypeError::new_err(format!(
                        "params: a list of molrs.ff.ir.Param or a {{name: dim}} dict, got {}",
                        describe(&item)
                    ))
                })
        })
        .collect()
}

/// `samples=`: dicts ``{"q": (lo, hi), <param>: value, …}``.
fn parse_samples(arg: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Sample>> {
    let Some(arg) = arg.filter(|a| !a.is_none()) else {
        return Ok(Vec::new());
    };
    arg.try_iter()?
        .map(|item| {
            let item = item?;
            let dict = item.cast::<PyDict>().map_err(|_| {
                PyTypeError::new_err("samples: a list of dicts {'q': (lo, hi), <param>: value}")
            })?;
            let mut q = None;
            let mut params = Vec::new();
            for (k, v) in dict.iter() {
                let key: String = k.extract()?;
                if key == "q" {
                    q = Some(v.extract::<(F, F)>().map_err(|_| {
                        PyTypeError::new_err("samples: 'q' is a (lo, hi) pair of numbers")
                    })?);
                } else {
                    let value = value_of(&format!("samples[{key:?}]"), &v)?;
                    params.push((Cow::Owned(key), value));
                }
            }
            let q = q.ok_or_else(|| PyValueError::new_err("samples: every sample needs 'q'"))?;
            Ok(Sample { params, q })
        })
        .collect()
}

fn parse_coordinate(text: &str) -> PyResult<Coordinate> {
    Ok(match text {
        "compound" => Coordinate::Compound,
        "distance" => Coordinate::Distance,
        "angle" => Coordinate::Angle,
        "dihedral" => Coordinate::Dihedral,
        "improper" => Coordinate::Improper,
        "none" => Coordinate::None,
        _ => {
            return Err(PyValueError::new_err(format!(
                "coordinate={text:?}: one of 'compound', 'distance', 'angle', 'dihedral', \
                 'improper'"
            )));
        }
    })
}

fn coordinate_name(c: Coordinate) -> &'static str {
    match c {
        Coordinate::None => "none",
        Coordinate::Distance => "distance",
        Coordinate::Angle => "angle",
        Coordinate::Dihedral => "dihedral",
        Coordinate::Improper => "improper",
        Coordinate::Compound => "compound",
    }
}

fn parse_order(text: &str) -> PyResult<EndpointOrder> {
    Ok(match text {
        "reversible" => EndpointOrder::Reversible,
        "ordered" => EndpointOrder::Ordered,
        "unordered" => EndpointOrder::Unordered,
        _ => {
            return Err(PyValueError::new_err(format!(
                "order={text:?}: one of 'reversible', 'ordered', 'unordered'"
            )));
        }
    })
}

fn order_name(o: EndpointOrder) -> &'static str {
    match o {
        EndpointOrder::Reversible => "reversible",
        EndpointOrder::Ordered => "ordered",
        EndpointOrder::Unordered => "unordered",
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register a new category of the force-field IR.
///
/// Its terms name ``arity`` atoms and live in the Frame block
/// ``f"{name}s"`` (molrec's rule, so a record alone locates them).
///
/// Parameters
/// ----------
/// name : str
///     ``^[a-z][a-z0-9_]*$``.
/// arity : int
///     Atoms per term, 2 to 5.
/// coordinate : {"compound", "distance", "angle", "dihedral", "improper"}
///     What its energy is a function of: the atoms' positions (an expression
///     over ``p1 … pA``, or a compound kernel), or the geometric variable
///     its arity has (``distance`` 2, ``angle`` 3, ``dihedral`` /
///     ``improper`` 4).
/// order : {"reversible", "ordered", "unordered"}
///     Which orders of a type row's endpoints name the same row.
///
/// Raises
/// ------
/// BadName, Arity, CoordinateMismatch, Sealed, Conflict
///     A malformed name, an arity outside 2..=5, a coordinate the arity
///     cannot carry, a built-in's name, another category's name.
///
/// Examples
/// --------
/// >>> ir.register_category("urey_bradley", 3)
/// >>> [c.block for c in ir.categories() if c.name == "urey_bradley"]
/// ['urey_bradleys']
#[pyfunction]
#[pyo3(signature = (name, arity, *, coordinate="compound", order="reversible"))]
fn register_category(name: String, arity: usize, coordinate: &str, order: &str) -> PyResult<()> {
    let coordinate = parse_coordinate(coordinate)?;
    let order = parse_order(order)?;
    let arity = u8::try_from(arity).map_err(|_| {
        refuse(IrError::Arity {
            category: name.clone(),
            arity,
        })
    })?;
    rir::register_category(CategorySpec::custom(name, arity, coordinate, order)).map_err(refuse)
}

/// The kernel a Python callable is registered as: reused when the same
/// callable is registered again for the style.
fn python_kernel(
    py: Python<'_>,
    spec: &StyleSpec,
    func: &Bound<'_, PyAny>,
    compound: bool,
    reuse: bool,
) -> PyResult<Kernel> {
    if !func.is_callable() {
        return Err(PyTypeError::new_err(format!(
            "kernel for {} `{}`: a callable, got {}",
            spec.category,
            spec.name,
            describe(func)
        )));
    }
    let key = (spec.category.to_string(), spec.name.to_string());
    if reuse {
        let kernels = py_kernels().lock().expect("kernel table poisoned");
        if let Some((known, kernel)) = kernels.get(&key)
            && matches!(kernel, Kernel::Compound(_)) == compound
            && known.bind(py).eq(func)?
        {
            return Ok(kernel.clone());
        }
    }
    let (func, style, extras) = (
        func.clone().unbind(),
        spec.name.to_string(),
        Extras::of(spec),
    );
    Ok(if compound {
        Kernel::Compound(Arc::new(PyCompoundKernel {
            func,
            style,
            extras,
        }))
    } else {
        Kernel::Scalar(Arc::new(PyScalarKernel {
            func,
            style,
            extras,
        }))
    })
}

/// Register a new style of the force-field IR — priced by an expression, a
/// Python kernel, or both (then they must agree).
///
/// Every compile reads the registry, so the style prices at both compile
/// doors (``compile``, ``compile_typed``) and in MD with nothing rebuilt.
///
/// Parameters
/// ----------
/// category : str
///     A built-in category (``bond``, ``angle``, ``dihedral``, ``improper``,
///     ``pair``, ``cmap``, …) or one from :func:`register_category`.
/// name : str
///     The style name.
/// params : list of Param or dict of {name: dim}
///     The per-type parameters, **ordered** (the LAMMPS ``*_coeff`` order
///     when a LAMMPS style of the name exists).
/// style_params : list of Param or dict of {name: dim}
///     Style-level parameters; ``cutoff`` (``L``), ``mixing`` and
///     ``special`` (text) keep their reserved meanings.
/// expression : str, optional
///     The unweighted per-term energy, a Lepton expression over the
///     category's variable (``r``, ``theta``, ``phi``, ``chi``; points
///     ``p1 … pA`` in a compound category; ``q1``, ``q2``, ``x1``/``x2`` in
///     a pair) and the numeric parameters **as stored** (angle values in
///     degrees).
/// kernel : callable, optional
///     A vectorised kernel, called once per style per evaluation. Scalar:
///     ``kernel(q, **params) -> (e, de_dq)``, ``q`` and every numeric
///     parameter an ``ndarray[n]``. Compound (``compound=True`` or a
///     ``"compound"`` category): ``kernel(x, **params) -> (e, grad)``, ``x``
///     and ``grad`` ``ndarray[n, arity, 3]``. Outputs are float64 arrays of
///     exactly those shapes (else :class:`KernelShape`); an exception it
///     raises is re-raised as :class:`KernelShape` with the exception as
///     ``__cause__``. A kernel must not call ``molrs.ff.ir`` itself.
/// compound : bool, default False
///     ``kernel`` takes positions, not the category's coordinate.
/// special : {"lj", "coul"}, optional
///     Pair styles: which special-bonds weights scale it (default ``lj``).
/// samples : list of dict, optional
///     ``{"q": (lo, hi), <param>: value, …}``: the derivative (and
///     expression agreement) checks run on 16 seeded points of each at
///     registration. Without samples they run once per process at the
///     style's first compile.
/// replace : bool, default False
///     Replace a style registered at run time under this name (a built-in
///     is :class:`Sealed` regardless). Without it, a different registration
///     under a taken name is :class:`Conflict`, an identical one a no-op.
/// lammps : {"positional", "positional:<name>"}, optional
///     The style's LAMMPS form: ``"positional"`` writes and reads it as
///     ``<category>_style <name>`` with ``params`` in order on the
///     ``*_coeff`` line, each converted by its dimension (``pair_style
///     <name> <cutoff>``, ``pair_modify mix <mixing>``); ``"positional:fene"``
///     under the LAMMPS style ``fene``. ``None`` (the default): LAMMPS
///     refuses it by name (:class:`NoEngineForm`; the installed LAMMPS has
///     no LEPTON package for an expression). A positional form the spec
///     cannot have (a Text, Array or indexed parameter, a style parameter
///     other than ``cutoff`` / ``mixing``) is :class:`NoEngineForm`.
///
/// Raises
/// ------
/// IrError
///     The subclass naming what does not conform: :class:`UnknownCategory`,
///     :class:`BadName`, :class:`ReservedParam`, :class:`DuplicateParam`,
///     :class:`Dim`, :class:`Parse`, :class:`UnboundVariable`,
///     :class:`UnknownFunction`, :class:`FunctionArity`, :class:`Point`,
///     :class:`CoordinateMismatch`, :class:`Derivative`, :class:`Disagree`,
///     :class:`Asymmetric`, :class:`KernelShape`, :class:`Sealed`,
///     :class:`Conflict`, :class:`NoKernel`, :class:`Malformed`.
///
/// Examples
/// --------
/// LAMMPS ``bond_style fene``, by its expression:
///
/// >>> ir.register_style(
/// ...     "bond", "fene",
/// ...     params=[ir.Param("k", "E/L^2"), ir.Param("r0", "L"),
/// ...             ir.Param("epsilon", "E"), ir.Param("sigma", "L")],
/// ...     expression="-0.5*k*r0^2*log(1-(r/r0)^2)"
/// ...                "+step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)",
/// ... )
#[pyfunction]
#[pyo3(signature = (category, name, *, params=None, style_params=None, expression=None, kernel=None, compound=false, special=None, samples=None, replace=false, lammps=None))]
#[allow(clippy::too_many_arguments)]
fn register_style(
    py: Python<'_>,
    category: String,
    name: String,
    params: Option<Bound<'_, PyAny>>,
    style_params: Option<Bound<'_, PyAny>>,
    expression: Option<String>,
    kernel: Option<Bound<'_, PyAny>>,
    compound: bool,
    special: Option<&str>,
    samples: Option<Bound<'_, PyAny>>,
    replace: bool,
    lammps: Option<&str>,
) -> PyResult<()> {
    let mut spec = StyleSpec::new(category.clone(), name.clone())
        .params(parse_params(params.as_ref())?)
        .style_params(parse_params(style_params.as_ref())?);
    spec.expression = expression;
    spec.samples = parse_samples(samples.as_ref())?;
    spec.lammps = engine::lammps_form(lammps)?;
    spec.special = match special {
        None => None,
        Some("lj") => Some(SpecialClass::Vdw),
        Some("coul") => Some(SpecialClass::Coulomb),
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "special={other:?}: 'lj' or 'coul'"
            )));
        }
    };
    let compound = compound
        || rir::with_global(|r| {
            r.category(&category)
                .is_some_and(|c| c.coordinate == Coordinate::Compound)
        });
    let kernel = kernel
        .filter(|k| !k.is_none())
        .map(|k| python_kernel(py, &spec, &k, compound, !replace).map(|kernel| (k, kernel)))
        .transpose()?;

    let key = (category.clone(), name.clone());
    // `replace=True` takes a run-time style out first, and puts it back if
    // the new one is refused.
    let previous = rir::with_global(|r| {
        r.style(&category, &name)
            .filter(|_| !r.is_sealed(&category, &name))
            .map(|(s, k)| (s.clone(), k.cloned()))
    });
    let removed = match previous {
        Some(old) if replace => {
            rir::unregister_style(&category, &name).map_err(refuse)?;
            Some(old)
        }
        _ => None,
    };
    clear_kernel_err();
    let registered = rir::register_style(spec, kernel.as_ref().map(|(_, k)| k.clone()));
    let outcome = match registered {
        Ok(()) => take_kernel_err().inspect_err(|_| {
            // The samples ran the kernel and it raised: take it out again.
            let _ = rir::unregister_style(&category, &name);
        }),
        Err(e) => Err(refuse(e)),
    };
    if let Err(err) = outcome {
        if let Some((spec, kernel)) = removed {
            // It was registered before; it registers again.
            let _ = rir::register_style(spec, kernel);
            clear_kernel_err();
        }
        return Err(err);
    }
    let mut kernels = py_kernels().lock().expect("kernel table poisoned");
    match kernel {
        Some((func, kernel)) => {
            kernels.insert(key, (func.unbind(), kernel));
        }
        None => {
            kernels.remove(&key);
        }
    }
    Ok(())
}

/// Remove a style registered at run time.
///
/// Raises
/// ------
/// Sealed
///     A built-in style.
/// NoKernel
///     No style of that name is registered.
#[pyfunction]
fn unregister(category: &str, name: &str) -> PyResult<()> {
    rir::unregister_style(category, name).map_err(refuse)?;
    py_kernels()
        .lock()
        .expect("kernel table poisoned")
        .remove(&(category.to_owned(), name.to_owned()));
    Ok(())
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

/// One category of the force-field IR, as :func:`categories` lists it.
#[pyclass(module = "molrs.ff.ir", name = "CategoryInfo", frozen)]
pub struct PyCategoryInfo {
    spec: CategorySpec,
    builtin: bool,
}

#[pymethods]
impl PyCategoryInfo {
    #[getter]
    fn name(&self) -> &str {
        &self.spec.name
    }

    /// Endpoints per type row (2 for a pair).
    #[getter]
    fn arity(&self) -> usize {
        self.spec.arity.endpoints()
    }

    /// Whether its terms are the pairs a neighbour search finds.
    #[getter]
    fn pair(&self) -> bool {
        self.spec.arity == Arity::SelfOrPair
    }

    /// The Frame block its terms live in.
    #[getter]
    fn block(&self) -> &str {
        &self.spec.block
    }

    #[getter]
    fn coordinate(&self) -> &'static str {
        coordinate_name(self.spec.coordinate)
    }

    #[getter]
    fn order(&self) -> &'static str {
        order_name(self.spec.order)
    }

    /// Whether it is one of molrs's own (sealed).
    #[getter]
    fn builtin(&self) -> bool {
        self.builtin
    }

    fn __repr__(&self) -> String {
        format!(
            "CategoryInfo('{}', arity={}, block='{}', coordinate='{}')",
            self.spec.name,
            self.arity(),
            self.spec.block,
            self.coordinate()
        )
    }
}

/// One style of the force-field IR, as :func:`styles` lists it.
#[pyclass(module = "molrs.ff.ir", name = "StyleInfo", frozen)]
pub struct PyStyleInfo {
    spec: StyleSpec,
    kernel: Option<&'static str>,
    builtin: bool,
}

impl PyStyleInfo {
    fn of(r: &Registry, spec: &StyleSpec, kernel: Option<&Kernel>) -> Self {
        let tier = match kernel {
            Some(Kernel::Expression(_)) => Some("expression"),
            Some(Kernel::Scalar(_)) => Some("scalar"),
            Some(Kernel::Compound(_)) => Some("compound"),
            Some(Kernel::Ctor { .. }) => Some("constructor"),
            None => spec.expression.as_ref().map(|_| "expression"),
        };
        Self {
            spec: spec.clone(),
            kernel: tier,
            builtin: r.is_sealed(&spec.category, &spec.name),
        }
    }
}

#[pymethods]
impl PyStyleInfo {
    #[getter]
    fn category(&self) -> &str {
        &self.spec.category
    }

    #[getter]
    fn name(&self) -> &str {
        &self.spec.name
    }

    /// The per-type parameters, in order.
    #[getter]
    fn params(&self) -> Vec<PyParam> {
        self.spec
            .params
            .iter()
            .map(|p| PyParam { inner: p.clone() })
            .collect()
    }

    #[getter]
    fn style_params(&self) -> Vec<PyParam> {
        self.spec
            .style_params
            .iter()
            .map(|p| PyParam { inner: p.clone() })
            .collect()
    }

    /// The energy as an expression, byte for byte as registered.
    #[getter]
    fn expression(&self) -> Option<&str> {
        self.spec.expression.as_deref()
    }

    /// What prices it: ``"expression"``, ``"scalar"`` / ``"compound"`` (a
    /// form kernel: Rust or a Python callable), ``"constructor"`` (a native
    /// kernel), or ``None`` (a category that prices nothing).
    #[getter]
    fn kernel(&self) -> Option<&'static str> {
        self.kernel
    }

    /// Whether it is one of molrs's own (sealed).
    #[getter]
    fn builtin(&self) -> bool {
        self.builtin
    }

    /// ``"type_rows"`` or ``"per_instance"``: where its numbers come from.
    #[getter]
    fn source(&self) -> &'static str {
        match self.spec.source {
            ParamSource::TypeRows => "type_rows",
            ParamSource::PerInstance => "per_instance",
        }
    }

    /// The style's LAMMPS form: ``"positional"``, ``"positional:<name>"``,
    /// ``"custom:<name>"`` (a codec of its own) or ``None``.
    #[getter]
    fn lammps(&self) -> Option<String> {
        engine::lammps_form_name(&self.spec)
    }

    /// A pair style's special-bonds class: ``"lj"`` or ``"coul"``.
    #[getter]
    fn special(&self) -> Option<&'static str> {
        self.spec.special.map(|s| match s {
            SpecialClass::Vdw => "lj",
            SpecialClass::Coulomb => "coul",
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "StyleInfo('{}', '{}', kernel={})",
            self.spec.category,
            self.spec.name,
            self.kernel.map_or("None".into(), |k| format!("'{k}'"))
        )
    }
}

/// Every registered style, or those of ``category``, sorted by
/// ``(category, name)``.
///
/// Examples
/// --------
/// >>> {s.kernel for s in ir.styles("bond") if s.name == "harmonic"}
/// {'constructor'}
#[pyfunction]
#[pyo3(signature = (category=None))]
fn styles(category: Option<&str>) -> Vec<PyStyleInfo> {
    rir::with_global(|r| {
        r.styles(category)
            .map(|(s, k)| PyStyleInfo::of(r, s, k))
            .collect()
    })
}

/// Every registered category, sorted by name.
#[pyfunction]
fn categories() -> Vec<PyCategoryInfo> {
    let builtin: Vec<String> = rir::builtin_categories()
        .into_iter()
        .map(|c| c.name.into_owned())
        .collect();
    rir::with_global(|r| {
        r.categories()
            .map(|c| PyCategoryInfo {
                spec: c.clone(),
                builtin: builtin.iter().any(|b| *b == c.name),
            })
            .collect()
    })
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------

/// The inputs of one [`evaluate`] call, owned.
#[derive(Default)]
struct Inputs {
    nums: Vec<(String, Vec<F>)>,
    arrays: Vec<(String, ArrayD<F>)>,
    texts: Vec<(String, Vec<String>)>,
    style_texts: Vec<(String, String)>,
}

impl Inputs {
    fn has(&self, name: &str) -> bool {
        self.nums.iter().any(|(n, _)| n == name)
            || self.arrays.iter().any(|(n, _)| n == name)
            || self.texts.iter().any(|(n, _)| n == name)
            || self.style_texts.iter().any(|(n, _)| n == name)
    }

    fn num(&self, name: &str) -> Option<&Vec<F>> {
        self.nums.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }

    /// `f` over the inputs as one batch's [`ParamCols`].
    fn with_cols<R>(&self, f: impl FnOnce(&ParamCols<'_>) -> R) -> R {
        let texts: Vec<Vec<&str>> = self
            .texts
            .iter()
            .map(|(_, t)| t.iter().map(String::as_str).collect())
            .collect();
        let mut cols = ParamCols::new();
        for (name, col) in &self.nums {
            cols.push(name, col);
        }
        for (name, a) in &self.arrays {
            cols.push_array(name, a.view());
        }
        for ((name, _), t) in self.texts.iter().zip(&texts) {
            cols.push_text(name, t);
        }
        for (name, t) in &self.style_texts {
            cols.push_style_text(name, t);
        }
        f(&cols)
    }
}

/// The declaration a keyword of [`evaluate`] names: a declared parameter,
/// a member `<name><m>` of an indexed family, or on a pair `q1`, `q2`,
/// `<name>1`, `<name>2` (numeric).
pub(crate) fn declared<'s>(
    spec: &'s StyleSpec,
    pair: bool,
    key: &str,
) -> Option<(Option<&'s ParamSpec>, bool)> {
    if let Some(p) = spec.param(key) {
        return Some((Some(p), false));
    }
    if let Some(p) = spec.style_param(key) {
        return Some((Some(p), true));
    }
    let base = key.trim_end_matches(|c: char| c.is_ascii_digit());
    if base.len() < key.len()
        && let Some(p) = spec.param(base).filter(|p| p.indexed)
    {
        return Some((Some(p), false));
    }
    if pair
        && (matches!(key, "q1" | "q2")
            || key
                .strip_suffix(['1', '2'])
                .is_some_and(|b| spec.param(b).is_some()))
    {
        return Some((None, false));
    }
    None
}

/// `obj` as `n` numbers: a scalar broadcast, or a length-`n` sequence.
pub(crate) fn column(what: &str, obj: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<F>> {
    if let Ok(v) = obj.extract::<F>() {
        return Ok(vec![v; n]);
    }
    let col: Vec<F> = obj
        .extract()
        .map_err(|_| PyTypeError::new_err(format!("{what}: a number or a 1-D array of numbers")))?;
    if col.len() != n {
        return Err(PyValueError::new_err(format!(
            "{what}: {} values for {n} terms",
            col.len()
        )));
    }
    Ok(col)
}

/// The keyword inputs of [`evaluate`], defaults filled in.
fn inputs(
    py: Python<'_>,
    spec: &StyleSpec,
    pair: bool,
    kwargs: Option<&Bound<'_, PyDict>>,
    n: usize,
) -> PyResult<Inputs> {
    let who = format!("{} `{}`", spec.category, spec.name);
    let mut out = Inputs::default();
    let np = py.import("numpy")?;
    for (k, v) in kwargs.into_iter().flat_map(|d| d.iter()) {
        let key: String = k.extract()?;
        let what = format!("{who}: `{key}`");
        let Some((decl, style_level)) = declared(spec, pair, &key) else {
            return Err(PyTypeError::new_err(format!(
                "{who} has no parameter `{key}`"
            )));
        };
        match decl.map(|d| &d.kind) {
            Some(ParamKind::Text { .. }) if style_level => {
                out.style_texts.push((key, v.extract()?));
            }
            Some(ParamKind::Text { .. }) => {
                let col = match v.extract::<String>() {
                    Ok(s) => vec![s; n],
                    Err(_) => v.extract::<Vec<String>>()?,
                };
                if col.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "{what}: {} strings for {n} terms",
                        col.len()
                    )));
                }
                out.texts.push((key, col));
            }
            Some(ParamKind::Array { rank }) => {
                let arr = np
                    .call_method1("asarray", (v, "float64"))?
                    .cast_into::<PyArrayDyn<F>>()?;
                let a = arr.readonly().as_array().to_owned();
                let a = match a.ndim() {
                    d if d == *rank as usize => {
                        let views = vec![a.view(); n];
                        ndarray::stack(Axis(0), &views)
                            .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?
                    }
                    d if d == *rank as usize + 1 && a.shape()[0] == n => a,
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "{what}: an array of rank {rank}, or {n} of them stacked; got \
                             shape {:?}",
                            a.shape()
                        )));
                    }
                };
                out.arrays.push((key, a));
            }
            Some(ParamKind::Scalar) | None => {
                out.nums.push((key, column(&what, &v, n)?));
            }
        }
    }
    for p in spec.params.iter().chain(&spec.style_params) {
        let style_level = spec.style_param(&p.name).is_some();
        let (Some(default), false) = (&p.default, out.has(&p.name) || p.indexed) else {
            continue;
        };
        match default {
            Value::Num(v) => out.nums.push((p.name.to_string(), vec![*v; n])),
            Value::Text(t) if style_level => {
                out.style_texts.push((p.name.to_string(), t.to_string()))
            }
            Value::Text(t) => out.texts.push((p.name.to_string(), vec![t.to_string(); n])),
        }
    }
    Ok(out)
}

/// Evaluate a style's energy on a batch of terms, exactly as a compile
/// would price them (unweighted, untruncated).
///
/// Parameters
/// ----------
/// category, name : str
///     The style.
/// q : array_like, shape (n,)
///     The coordinate per term, for a style priced by a scalar form: ``r``
///     (length), ``theta`` or ``phi`` (radians).
/// x : array_like, shape (n, arity, 3), optional
///     The atoms' positions per term, for a style priced over positions (a
///     compound category, or an expression using ``distance``/``angle``/
///     ``dihedral``).
/// **params
///     Each parameter **as stored** (angle values in degrees): a number
///     (broadcast) or one value per term; a text parameter a string; on a
///     pair ``q1``, ``q2`` and the self rows ``<name>1``, ``<name>2``
///     (default: the pair value). Missing ones take their default.
///
/// Returns
/// -------
/// e : ndarray, shape (n,)
///     The energy per term.
/// de_dq or grad : ndarray, shape (n,) or (n, arity, 3)
///     ``dE/dq``, or ``∂E/∂x`` (not the force).
///
/// Raises
/// ------
/// MissingParam
///     A parameter the form reads, not given and with no default.
/// KernelShape
///     A Python kernel that raised or returned the wrong shape.
/// TypeError
///     An undeclared parameter, or ``q``/``x`` not the form's input.
///
/// Examples
/// --------
/// >>> e, de = ir.evaluate("bond", "harmonic", [1.0, 1.5], k=300.0, r0=1.2)
/// >>> e.tolist()
/// [12.0, 27.0]
#[pyfunction]
#[pyo3(signature = (category, name, q=None, *, x=None, **params))]
fn evaluate<'py>(
    py: Python<'py>,
    category: &str,
    name: &str,
    q: Option<Bound<'py, PyAny>>,
    x: Option<Bound<'py, PyAny>>,
    params: Option<&Bound<'py, PyDict>>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let who = format!("{category} `{name}`");
    let (cat, spec, kernel, compiler) = rir::with_global(|r| {
        let cat = r.category(category).cloned().ok_or_else(|| {
            refuse(IrError::UnknownCategory {
                category: category.to_owned(),
            })
        })?;
        let (spec, kernel) = r.style(category, name).ok_or_else(|| {
            refuse(IrError::NoKernel {
                category: category.to_owned(),
                style: name.to_owned(),
            })
        })?;
        PyResult::Ok((cat, spec.clone(), kernel.cloned(), r.expression_compiler()))
    })?;
    let form = match kernel {
        Some(Kernel::Expression(k)) => k.form(),
        Some(Kernel::Scalar(f)) => ExpressionForm::Scalar(f),
        Some(Kernel::Compound(f)) => ExpressionForm::Compound(f),
        // A native constructor builds a whole kernel from a frame; its
        // expression is what evaluates on a batch.
        Some(Kernel::Ctor { .. }) | None => match (compiler, &spec.expression) {
            (Some(compile), Some(_)) => compile(&cat, &spec).map_err(refuse)?.form(),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "{who} has no expression and no form kernel to evaluate on a batch \
                     (a native constructor prices whole frames: compile it)"
                )));
            }
        },
    };
    let pair = cat.is_pair_driven();
    let np = py.import("numpy")?;
    let ascontiguous = |obj: Bound<'py, PyAny>| -> PyResult<Bound<'py, PyArrayDyn<F>>> {
        Ok(np
            .call_method1("ascontiguousarray", (obj, "float64"))?
            .cast_into::<PyArrayDyn<F>>()?)
    };
    match form {
        ExpressionForm::Scalar(f) => {
            if x.is_some() {
                return Err(PyTypeError::new_err(format!(
                    "{who} is a function of its coordinate: pass q, not x"
                )));
            }
            let q = q.ok_or_else(|| {
                PyTypeError::new_err(format!(
                    "{who}: evaluate() needs q, the coordinate per term"
                ))
            })?;
            let q: Vec<F> = ascontiguous(q)?
                .readonly()
                .as_array()
                .iter()
                .copied()
                .collect();
            let n = q.len();
            let mut inp = inputs(py, &spec, pair, params, n)?;
            require(&spec, f.inputs(), &mut inp, pair)?;
            let (mut e, mut de) = (vec![0.0; n], vec![0.0; n]);
            clear_kernel_err();
            inp.with_cols(|p| f.eval(&q, p, &mut e, &mut de));
            take_kernel_err()?;
            Ok((
                e.into_pyarray(py).into_any(),
                de.into_pyarray(py).into_any(),
            ))
        }
        ExpressionForm::Compound(f) => {
            if q.is_some() {
                return Err(PyTypeError::new_err(format!(
                    "{who} is priced over its atoms' positions: pass x, not q"
                )));
            }
            let arity = cat.arity.endpoints();
            let x = x.ok_or_else(|| {
                PyTypeError::new_err(format!(
                    "{who}: evaluate() needs x, the positions per term, shape (n, {arity}, 3)"
                ))
            })?;
            let x = ascontiguous(x)?;
            let shape = x.shape().to_vec();
            if shape.len() != 3 || shape[1] != arity || shape[2] != 3 {
                return Err(PyValueError::new_err(format!(
                    "{who}: x must have shape (n, {arity}, 3), got {shape:?}"
                )));
            }
            let n = shape[0];
            let flat: Vec<F> = x.readonly().as_array().iter().copied().collect();
            let points: Vec<[F; 3]> = flat.as_chunks::<3>().0.to_vec();
            let mut inp = inputs(py, &spec, pair, params, n)?;
            require(&spec, f.inputs(), &mut inp, pair)?;
            let (mut e, mut grad) = (vec![0.0; n], vec![[0.0; 3]; n * arity]);
            clear_kernel_err();
            inp.with_cols(|p| f.eval(&points, arity, p, &mut e, &mut grad));
            take_kernel_err()?;
            let grad = Array3::from_shape_vec((n, arity, 3), grad.into_iter().flatten().collect())
                .expect("n · arity points");
            Ok((
                e.into_pyarray(py).into_any(),
                grad.into_pyarray(py).into_any(),
            ))
        }
    }
}

/// Refuse a missing input: those the form states it reads (an expression),
/// else every declared parameter without a default (a Python kernel). A
/// pair's self rows `<x>1`, `<x>2` default to the pair value `<x>`.
fn require(spec: &StyleSpec, reads: Vec<String>, inp: &mut Inputs, pair: bool) -> PyResult<()> {
    let missing = |param: &str| {
        refuse(IrError::MissingParam {
            style: spec.name.to_string(),
            type_: String::new(),
            param: param.to_owned(),
        })
    };
    if reads.is_empty() {
        let needed = spec
            .params
            .iter()
            .chain(&spec.style_params)
            .filter(|p| !p.indexed);
        for p in needed {
            if !inp.has(&p.name) {
                return Err(missing(&p.name));
            }
        }
        return Ok(());
    }
    for name in reads {
        if inp.num(&name).is_some() {
            continue;
        }
        let pair_value = pair
            .then(|| name.strip_suffix(['1', '2']))
            .flatten()
            .and_then(|base| inp.num(base).cloned());
        match pair_value {
            Some(col) => inp.nums.push((name, col)),
            None => return Err(missing(&name)),
        }
    }
    Ok(())
}

/// Register the classes and functions of `molrs.ff.ir` on `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    errors::register(m)?;
    m.add_class::<PyParam>()?;
    m.add_class::<PyStyleInfo>()?;
    m.add_class::<PyCategoryInfo>()?;
    m.add_function(wrap_pyfunction!(register_category, m)?)?;
    m.add_function(wrap_pyfunction!(register_style, m)?)?;
    m.add_function(wrap_pyfunction!(unregister, m)?)?;
    m.add_function(wrap_pyfunction!(styles, m)?)?;
    m.add_function(wrap_pyfunction!(categories, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    engine::register(m)?;
    Ok(())
}
