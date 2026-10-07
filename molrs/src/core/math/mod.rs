//! Special functions and physics numerics: complex arithmetic, spherical
//! harmonics, Wigner symbols and the virial.
//!
//! The pure 3×3 linear algebra and the symmetric eigensolvers live in
//! [`crate::op`] (`det3`, `inv3`, `eigh_sym_3x3`); the vector kernels in [`crate::op::vec3`].

mod complex;
mod spherical_harmonics;
mod virial;
mod wigner3j;
mod wigner_d;

pub use complex::Complex;
pub use spherical_harmonics::{legendre_plm, ylm_all, ylm_complex, ylm_normalization, ylm_real};
pub use virial::Virial;
pub use wigner_d::{wigner_d_element, wigner_d_matrix, wigner_small_d};
pub use wigner3j::wigner_3j;
