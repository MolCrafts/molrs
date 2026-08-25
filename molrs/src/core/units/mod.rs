//! Pint-inspired unit system: dimensions, units, registry, parser, quantities.
//!
//! A [`UnitRegistry`] holds [`UnitDef`] entries over the 7 SI base dimensions,
//! an internal recursive-descent parser turns compound expressions
//! (`"kcal/mol/angstrom"`) into self-contained [`Unit`] values, and
//! [`Quantity`] provides dimension-checked arithmetic and conversion.
//!
//! Every unit is reduced at parse time to `(factor, offset, Dimension)` with
//! conversion to SI base defined as `si = value * factor + offset`; the
//! `offset` is non-zero only for affine units such as `degC`. Conversion
//! factors are SI-2019 exact or CODATA 2018 recommended values (see
//! [`constants`] and the registry preload tables in `registry.rs`).
//!
//! Reference: design follows pint (Python),
//! <https://pint.readthedocs.io/en/stable/>.
//!
//! # Examples
//!
//! Energy conversion and dimension checking:
//!
//! ```
//! use molrs::units::{UnitRegistry, UnitsError};
//!
//! let reg = UnitRegistry::new();
//!
//! // 1 kcal/mol = 4.184 kJ/mol (thermochemical calorie, exact).
//! let e = reg.quantity(1.0, "kcal/mol")?;
//! let kj = e.to(&reg.parse("kJ/mol")?)?;
//! assert!((kj.value() - 4.184).abs() < 1e-12);
//!
//! // Converting an energy to a length is a dimension error.
//! let err = e.to(&reg.parse("angstrom")?).unwrap_err();
//! assert!(matches!(err, UnitsError::DimensionMismatch { .. }));
//! # Ok::<(), UnitsError>(())
//! ```

pub mod constants;
pub mod dimension;
pub mod error;
mod parse;
pub mod preset;
pub mod quantity;
pub mod registry;
pub mod unit;

pub use dimension::Dimension;
pub use error::UnitsError;
pub use preset::{
    PRESET_DIMENSIONS, UnitPreset, UnitPresetRegistry, lookup_preset, register_preset,
};
pub use quantity::Quantity;
pub use registry::{UnitDef, UnitRegistry};
pub use unit::Unit;
