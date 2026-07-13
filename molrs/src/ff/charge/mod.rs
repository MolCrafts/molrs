//! Charge models: molecule in, charges out.
//!
//! One trait, [`ChargeModel`], and the models behind it:
//!
//! | model | needs QM input? | topology correction? | [`needs_equivalencing`] |
//! |---|---|---|---|
//! | [`MullikenModel`] | yes | no (pass-through) | `false` |
//! | [`BccModel`] (AM1-BCC) | yes | yes (bond increments) | `true` |
//! | [`BccModel`] (ABCG2) | yes | yes (bond increments) | `true` |
//!
//! # The seam is a push, not a pull
//!
//! QM charges are an **argument** — `assign(&mol, Some(&am1))`,
//! `BccModel::correct(&mol, &am1)` — because molrs does not compute them. It used to
//! ask for them through a backend trait, and every implementor that trait ever had
//! ignored the molecule it was handed and returned a vector computed elsewhere; the
//! production path had to dress a `Vec<f64>` up as a solver to get it in. The
//! charges now go in the direction they actually travel.
//!
//! # Nothing is written into the molecule
//!
//! Every model takes `&Atomistic` and returns `Vec<f64>`. The charges are the return
//! value, and the models' internal atom types (BCC codes like `11` / `91`) stay
//! internal — a caller keeps their GAFF or OPLS types in [`keys::TYPE`](molrs::store::keys::TYPE)
//! and gets BCC charges back, which is what the standard AM1-BCC workflow needs.
//!
//! # Equivalencing is the model's declaration, and the model honours it
//!
//! [`ChargeModel::needs_equivalencing`] is antechamber's `-eq`, whose default is per
//! charge *method*: `1` for `bcc` / `abcg2` / `resp`, `0` for everything else. So
//! [`assign`](ChargeModel::assign) applies the class-mean itself when the model
//! declares it needs it. Handed the raw `sqm` Mulliken charges an AM1 solver really
//! produces, [`BccModel`] averages them over the topological-equivalence classes and
//! then corrects; [`MullikenModel`] hands the same bits back.
//!
//! ```
//! use molrs::Atomistic;
//! use molrs::ff::charge::{BccModel, BccParameterSet, ChargeModel, MullikenModel};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut mol = Atomistic::new();
//! let o = mol.add_atom_xyz("O", 0.0, 0.0, 0.0);
//! let h1 = mol.add_atom_xyz("H", 0.96, 0.0, 0.0);
//! let h2 = mol.add_atom_xyz("H", -0.24, 0.93, 0.0);
//! mol.add_bond(o, h1)?;
//! mol.add_bond(o, h2)?;
//!
//! // The model is chosen at runtime — the trait is object-safe, and no caller has
//! // to know which one it is holding.
//! let models: Vec<Box<dyn ChargeModel>> = vec![
//!     Box::new(BccModel::new(BccParameterSet::Bcc)?),
//!     Box::new(MullikenModel),
//! ];
//! let am1 = [-0.826, 0.417, 0.409];
//! for model in &models {
//!     let q: Vec<f64> = model.assign(&mol, Some(&am1))?;
//!     assert_eq!(q.len(), mol.n_atoms());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! [`needs_equivalencing`]: ChargeModel::needs_equivalencing

mod bcc;
mod error;
mod model;
mod mulliken;

pub use bcc::BccModel;
pub use error::ChargeError;
pub use model::ChargeModel;
pub use mulliken::MullikenModel;

/// The correction family a [`BccModel`] applies: `BCCPARM.DAT` or `BCCPARM_ABCG2.DAT`.
///
/// Re-exported from the module that owns the tables
/// ([`ff::typifier::am1bcc`](crate::ff::typifier::am1bcc)), so that a caller building
/// a charge model never has to reach into the typifier tree for the one argument the
/// model needs.
pub use crate::ff::typifier::am1bcc::BccParameterSet;
