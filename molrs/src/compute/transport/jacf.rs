//! Green–Kubo ionic current route — see [`GreenKuboConductivity`](super::GreenKuboConductivity).
//!
//! Assemble the collective current series \(J(t)=\sum_a q_a v_a(t)\) yourself
//! (or via an upstream Frame pipeline), then compose:
//!
//! 1. [`GreenKuboConductivity`](super::GreenKuboConductivity) — raw ACF
//! 2. [`CumulativeTrapezoid`](crate::compute::fitting::CumulativeTrapezoid) — ∫C
//! 3. scale by \(1/(3 V k_B T)\) (SI prefactor in the fit / caller units table)
//!
//! There is no separate `Jacf` type. molpy must not invent one either.
