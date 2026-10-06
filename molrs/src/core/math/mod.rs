//! Special functions and physics numerics: complex arithmetic, spherical
//! harmonics, Wigner symbols and the virial.
//!
//! The pure 3×3 linear algebra and the symmetric eigensolvers live in
//! [`crate::op::linalg`]; the vector kernels in [`crate::op::vec3`].

pub mod complex;
pub mod spherical_harmonics;
mod virial;
pub mod wigner3j;
pub mod wigner_d;

pub use virial::Virial;
