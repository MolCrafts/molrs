//! Antechamber and AMBER `parm` parameter tables, compiled to typed Rust `const`s.
//!
//! DO NOT HAND-EDIT — regenerate with `scripts/gen_param_tables.py`.
//!
//! The row types live in [`crate::ff::params`]; this module holds only data.

pub mod atomtype_abcg2;
pub mod atomtype_amber;
pub mod atomtype_bcc;
pub mod atomtype_gas;
pub mod atomtype_gff;
pub mod atomtype_gff2;
pub mod atomtype_sybyl;
pub mod bccparm;
pub mod bccparm_abcg2;
pub mod gaff;
pub mod gaff2;
pub mod gaff_empirical;
pub mod gaff_equiv;
pub mod gasparm;
