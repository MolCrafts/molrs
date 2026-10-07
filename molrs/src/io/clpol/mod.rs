//! CL&Pol's `alpha.ff`, the Drude polarisation table of the CL&Pol
//! polarisable force field (paduagroup/clandpol).
//!
//! The doors are functions of [`crate::io`]:
//! [`read_clpol_alpha`](crate::io::read_clpol_alpha) and
//! [`read_clpol_alpha_str`](crate::io::read_clpol_alpha_str). This module holds
//! the format's row, [`ClpolAlphaRow`].

pub(crate) mod codec;

pub use codec::ClpolAlphaRow;
