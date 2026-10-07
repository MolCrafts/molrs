//! The force-field IR as a protocol: anything that conforms to its form can
//! extend it, with nothing in molrs rebuilt.
//!
//! The IR adopts the LAMMPS standard for every style LAMMPS has
//! (`molrs-python/docs/guides/forcefield-ir.md`); this module states that
//! standard as data, so a new style or a new category is a registration,
//! not an edit:
//!
//! * a **category** is a [`CategorySpec`] — its arity, the Frame block it
//!   prices, the [`Coordinate`] its energy is a function of;
//! * a **style** is a [`StyleSpec`] — its ordered per-type parameters, each
//!   with a [`Dim`], its style parameters, where its numbers come from — and
//!   a [`Kernel`] in one of three tiers: an expression
//!   ([`ExpressionKernel`]), a batch form of one coordinate or of the atoms'
//!   positions ([`ScalarForm`](crate::ff::potential::generic::ScalarForm),
//!   [`CompoundForm`](crate::ff::potential::generic::CompoundForm), built into
//!   the generic kernels of [`crate::ff::potential::generic`]), or a constructor that
//!   builds a whole kernel (every built-in kernel; `dihedral rb` is a
//!   built-in priced by its expression alone);
//! * the [`Registry`] refuses anything that does not conform
//!   ([`conformance`], [`IrError`]) and seals the built-ins;
//! * [`expr`] compiles a style's Lepton `expression` into its kernel, with
//!   exact derivatives — installed in every registry
//!   [`Registry::builtin`] makes;
//! * a style's **engine forms** ([`Engine`], [`EngineCodec`]): its [`LammpsForm`] (positional,
//!   derived from the spec with conversion per [`Dim`], or a [`LammpsCodec`]
//!   of its own) drives the LAMMPS reader and writer; an expression style is
//!   written to OpenMM XML as its category's `Custom*Force`; every engine
//!   that cannot hold a style refuses it with [`IrError::NoEngineForm`];
//! * a style of a **form family** registers a [`FormCodec`] — its exact maps
//!   to and from the family's canonical style — and the form machinery converts a force
//!   field between the family's styles: exactly
//!   (`ForceField::canonical`, `ForceField::to_form`) or by least squares
//!   with a residual (`ForceField::fit_form`).
//!
//! ```
//! use std::sync::Arc;
//! use molrs::ff::ir::{Dim, Kernel, ParamSpec, Registry, StyleSpec};
//! use molrs::ff::potential::generic::{ParamCols, ScalarForm};
//!
//! /// LAMMPS `bond_style harmonic`, as a third party would write it.
//! struct Harmonic;
//! impl ScalarForm for Harmonic {
//!     fn eval(&self, r: &[f64], p: &ParamCols<'_>, e: &mut [f64], de_dr: &mut [f64]) {
//!         let (k, r0) = (p.get("k").unwrap(), p.get("r0").unwrap());
//!         for t in 0..r.len() {
//!             e[t] = k[t] * (r[t] - r0[t]).powi(2);
//!             de_dr[t] = 2.0 * k[t] * (r[t] - r0[t]);
//!         }
//!     }
//! }
//!
//! let mut registry = Registry::builtin();
//! let spec = StyleSpec::new("bond", "my_harmonic").params(vec![
//!     ParamSpec::new("k", "E/L^2".parse().unwrap()),
//!     ParamSpec::new("r0", Dim::LENGTH),
//! ]);
//! registry
//!     .register_style(spec, Some(Kernel::Scalar(Arc::new(Harmonic))))
//!     .unwrap();
//! ```

mod category;
pub mod conformance;
mod dim;
mod engine;
mod error;
pub mod expr;
pub(crate) mod expression;
mod form;
pub(crate) mod registry;
mod spec;
mod style_table;

pub use category::{
    Arity, CategorySpec, Coordinate, EndpointOrder, builtin_categories, category_arity,
};
pub use dim::Dim;
pub use engine::{
    Engine, EngineCodec, LammpsCodec, LammpsCoeffs, LammpsForm, Token, UnitScale, positional,
};
pub use error::IrError;
pub use expression::{CompiledExpression, compile_expression};
pub use form::{FormCodec, FormFn, Metric, Refusal, Residual, TypeParams, TypeResidual};
pub use registry::{
    ExpressionCompiler, ExpressionForm, ExpressionKernel, Kernel, KernelConstructor, ParamSource,
    Registry, RegistryRef, RowSource, SpecialClass, register_category, register_engine_form,
    register_form, register_style, set_expression_compiler, unregister_style, with_global,
};
pub use spec::{Mix, ParamKind, ParamSpec, Sample, StyleSpec, Value, builtin_styles};
pub use style_table::{ANNOTATION_COLUMNS, CMAP_GRID, ENDPOINT_COLUMNS, is_parameter_column};

#[cfg(test)]
mod builtin_conformance;
#[cfg(test)]
mod tests;
