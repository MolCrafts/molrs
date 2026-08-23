//! Integrator components: advance an [`MDState`].
//!
//! Required pieces go in the constructor — no `bind_*` afterthoughts:
//!
//! ```ignore
//! VelocityVerlet::new(dt, potential, neighbors, mass)?;
//! Langevin::new(dt, gamma, kbt, potential, neighbors, mass, seed)?;
//! ```
//!
//! Two schemes, two types — no `gamma=0` switch:
//!
//! * [`VelocityVerlet`] — NVE (B-A-A-B; the two half-drifts stay as separate
//!   adds).
//! * [`Langevin`] — BAOAB Langevin (γ > 0). Ordering (Leimkuhler & Matthews):
//!   B (half kick) → A (half drift) → O (Ornstein-Uhlenbeck) → A → B. The O
//!   step `v ← c1·v + c2·σ·ξ` with `c1 = e^{-γΔt}`, `c2 = √(1-c1²)`,
//!   `σ = √(k_BT/m)`.
//!
//! Units — one self-consistent system (amu, Å, fs) with energy in
//! amu·Å²/fs². See [`super::units`].
//!
//! Reference:
//!     Leimkuhler & Matthews, "Rational Construction of Stochastic Numerical
//!     Methods for Molecular Sampling", Appl. Math. Res. Express 2013.
//!     https://doi.org/10.1093/amrx/abs010

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

use molrs::spatial::neighbors::VerletSkin;
use molrs::types::{F, FNx3};

use super::error::MdError;
use super::lj::LJ;
use super::types::{ForceOutput, MDState};

fn as_mass_col(mass: ArrayView1<'_, F>) -> Result<Array2<F>, MdError> {
    if mass.iter().any(|&m| !m.is_finite() || m <= 0.0) {
        return Err(MdError::Invalid("mass must be strictly positive".into()));
    }
    Ok(mass.to_owned().insert_axis(ndarray::Axis(1)))
}

fn check_state_shape(
    pos: ArrayView2<'_, F>,
    vel: ArrayView2<'_, F>,
    n_mass: usize,
) -> Result<(), MdError> {
    if pos.shape() != vel.shape() {
        return Err(MdError::Invalid(format!(
            "pos shape {:?} disagrees with vel shape {:?}",
            pos.shape(),
            vel.shape()
        )));
    }
    if pos.ncols() != 3 {
        return Err(MdError::Invalid(format!(
            "pos must have shape (N, 3), got {:?}",
            pos.shape()
        )));
    }
    if pos.nrows() != n_mass {
        return Err(MdError::Invalid(format!(
            "n_atoms {} disagrees with mass length {n_mass}",
            pos.nrows()
        )));
    }
    Ok(())
}

/// NVE velocity-Verlet (B-A-A-B).
///
/// Construct with timestep, pair [`LJ`], [`VerletSkin`] neighbour list, and mass.
pub struct VelocityVerlet {
    /// Timestep Δt in fs.
    pub dt: F,
    potential: LJ,
    neighbors: VerletSkin,
    mass_col: Array2<F>,
    /// Per-atom 1/m (`(N,)`).
    inv_mass: Array1<F>,
}

impl VelocityVerlet {
    /// `dt` (fs), pair potential, neighbour list, per-atom mass `(N,)`.
    pub fn new(
        dt: F,
        potential: LJ,
        neighbors: VerletSkin,
        mass: ArrayView1<'_, F>,
    ) -> Result<Self, MdError> {
        let mass_col = as_mass_col(mass)?;
        let inv_mass = mass_col.column(0).mapv(|m| 1.0 / m);
        Ok(Self {
            dt,
            potential,
            neighbors,
            mass_col,
            inv_mass,
        })
    }

    /// Per-atom mass column `(N, 1)`.
    pub fn mass(&self) -> &Array2<F> {
        &self.mass_col
    }

    /// Inverse mass `(N,)`.
    pub fn inv_mass(&self) -> &Array1<F> {
        &self.inv_mass
    }

    /// Degrees of freedom the temperature estimator must not count (`3N − 3`).
    pub fn removed_dof(&self) -> usize {
        3
    }

    /// Energy and forces at `pos`.
    pub fn eval_force(&mut self, pos: ArrayView2<'_, F>) -> Result<ForceOutput, MdError> {
        let mut forces = Array2::<F>::zeros((pos.nrows(), 3));
        let energy = self
            .potential
            .eval_into(&mut self.neighbors, pos, &mut forces)?;
        Ok(ForceOutput { energy, forces })
    }

    /// Seed an [`MDState`], evaluating the entry force.
    pub fn initial(&mut self, pos: FNx3, vel: FNx3) -> Result<MDState, MdError> {
        check_state_shape(pos.view(), vel.view(), self.mass_col.nrows())?;
        let out = self.eval_force(pos.view())?;
        Ok(MDState {
            pos,
            vel,
            forces: out.forces,
            energy: out.energy,
        })
    }

    /// One NVE step from the cached entry force (in-place arithmetic).
    pub fn step(&mut self, mut state: MDState) -> Result<MDState, MdError> {
        let half_dt = 0.5 * self.dt;
        // B
        Zip::from(state.vel.rows_mut())
            .and(state.forces.rows())
            .and(&self.inv_mass)
            .for_each(|mut v, f, &im| {
                v[0] += half_dt * f[0] * im;
                v[1] += half_dt * f[1] * im;
                v[2] += half_dt * f[2] * im;
            });
        // A, A — two separate half-drifts (not fused to `dt * vel`)
        for _ in 0..2 {
            Zip::from(state.pos.rows_mut())
                .and(state.vel.rows())
                .for_each(|mut p, v| {
                    p[0] += half_dt * v[0];
                    p[1] += half_dt * v[1];
                    p[2] += half_dt * v[2];
                });
        }
        let energy =
            self.potential
                .eval_into(&mut self.neighbors, state.pos.view(), &mut state.forces)?;
        // B
        Zip::from(state.vel.rows_mut())
            .and(state.forces.rows())
            .and(&self.inv_mass)
            .for_each(|mut v, f, &im| {
                v[0] += half_dt * f[0] * im;
                v[1] += half_dt * f[1] * im;
                v[2] += half_dt * f[2] * im;
            });
        state.energy = energy;
        Ok(state)
    }

    /// One eager step.
    pub fn advance(&mut self, state: MDState) -> Result<MDState, MdError> {
        self.step(state)
    }

    /// Advance `n_steps` eagerly.
    pub fn advance_n(&mut self, mut state: MDState, n_steps: usize) -> Result<MDState, MdError> {
        for _ in 0..n_steps {
            state = self.advance(state)?;
        }
        Ok(state)
    }
}

/// Langevin velocity-Verlet (BAOAB). γ must be strictly positive.
///
/// NVE is [`VelocityVerlet`] — not this type with `gamma=0`.
pub struct Langevin {
    /// Timestep Δt in fs.
    pub dt: F,
    /// Langevin friction γ in fs⁻¹.
    pub gamma: F,
    c1: F,
    c2: F,
    kbt: F,
    potential: LJ,
    neighbors: VerletSkin,
    mass_col: Array2<F>,
    inv_mass: Array1<F>,
    sigma: Array1<F>,
    rng: rand::rngs::StdRng,
}

impl Langevin {
    /// BAOAB integrator with all required pieces at construction.
    pub fn new(
        dt: F,
        gamma: F,
        kbt: F,
        potential: LJ,
        neighbors: VerletSkin,
        mass: ArrayView1<'_, F>,
        seed: u64,
    ) -> Result<Self, MdError> {
        if gamma <= 0.0 {
            return Err(MdError::Invalid(
                "Langevin requires gamma > 0; use VelocityVerlet for NVE".into(),
            ));
        }
        if kbt <= 0.0 {
            return Err(MdError::Invalid("Langevin requires kbt > 0".into()));
        }
        let mass_col = as_mass_col(mass)?;
        let inv_mass = mass_col.column(0).mapv(|m| 1.0 / m);
        let sigma = mass_col.column(0).mapv(|m| (kbt / m).sqrt());
        let c1 = (-gamma * dt).exp();
        let c2 = (1.0 - c1 * c1).max(0.0).sqrt();
        Ok(Self {
            dt,
            gamma,
            c1,
            c2,
            kbt,
            potential,
            neighbors,
            mass_col,
            inv_mass,
            sigma,
            rng: rand::SeedableRng::seed_from_u64(seed),
        })
    }

    pub fn c1(&self) -> F {
        self.c1
    }

    pub fn c2(&self) -> F {
        self.c2
    }

    /// Thermostat temperature k_B T in MD energy units.
    pub fn kbt(&self) -> F {
        self.kbt
    }

    pub fn mass(&self) -> &Array2<F> {
        &self.mass_col
    }

    pub fn sigma(&self) -> &Array1<F> {
        &self.sigma
    }

    pub fn inv_mass(&self) -> &Array1<F> {
        &self.inv_mass
    }

    /// `0` — the O step agitates all 3N DoF, COM included.
    pub fn removed_dof(&self) -> usize {
        0
    }

    pub fn eval_force(&mut self, pos: ArrayView2<'_, F>) -> Result<ForceOutput, MdError> {
        let mut forces = Array2::<F>::zeros((pos.nrows(), 3));
        let energy = self
            .potential
            .eval_into(&mut self.neighbors, pos, &mut forces)?;
        Ok(ForceOutput { energy, forces })
    }

    pub fn initial(&mut self, pos: FNx3, vel: FNx3) -> Result<MDState, MdError> {
        check_state_shape(pos.view(), vel.view(), self.mass_col.nrows())?;
        let out = self.eval_force(pos.view())?;
        Ok(MDState {
            pos,
            vel,
            forces: out.forces,
            energy: out.energy,
        })
    }

    pub fn step(
        &mut self,
        mut state: MDState,
        noise: ArrayView2<'_, F>,
    ) -> Result<MDState, MdError> {
        if noise.shape() != state.vel.shape() {
            return Err(MdError::Invalid(format!(
                "noise shape {:?} disagrees with vel shape {:?}",
                noise.shape(),
                state.vel.shape()
            )));
        }
        let half_dt = 0.5 * self.dt;
        let c1 = self.c1;
        let c2 = self.c2;
        // B
        Zip::from(state.vel.rows_mut())
            .and(state.forces.rows())
            .and(&self.inv_mass)
            .for_each(|mut v, f, &im| {
                v[0] += half_dt * f[0] * im;
                v[1] += half_dt * f[1] * im;
                v[2] += half_dt * f[2] * im;
            });
        // A
        Zip::from(state.pos.rows_mut())
            .and(state.vel.rows())
            .for_each(|mut p, v| {
                p[0] += half_dt * v[0];
                p[1] += half_dt * v[1];
                p[2] += half_dt * v[2];
            });
        // O
        Zip::from(state.vel.rows_mut())
            .and(&self.sigma)
            .and(noise.rows())
            .for_each(|mut v, &sig, xi| {
                v[0] = c1 * v[0] + c2 * sig * xi[0];
                v[1] = c1 * v[1] + c2 * sig * xi[1];
                v[2] = c1 * v[2] + c2 * sig * xi[2];
            });
        // A
        Zip::from(state.pos.rows_mut())
            .and(state.vel.rows())
            .for_each(|mut p, v| {
                p[0] += half_dt * v[0];
                p[1] += half_dt * v[1];
                p[2] += half_dt * v[2];
            });
        let energy =
            self.potential
                .eval_into(&mut self.neighbors, state.pos.view(), &mut state.forces)?;
        // B
        Zip::from(state.vel.rows_mut())
            .and(state.forces.rows())
            .and(&self.inv_mass)
            .for_each(|mut v, f, &im| {
                v[0] += half_dt * f[0] * im;
                v[1] += half_dt * f[1] * im;
                v[2] += half_dt * f[2] * im;
            });
        state.energy = energy;
        Ok(state)
    }

    pub fn draw_noise(&mut self, n_atoms: usize) -> Array2<F> {
        let mut noise = Array2::<F>::zeros((n_atoms, 3));
        for x in noise.iter_mut() {
            *x = standard_normal(&mut self.rng);
        }
        noise
    }

    pub fn advance(&mut self, state: MDState) -> Result<MDState, MdError> {
        let n = state.vel.nrows();
        let noise = self.draw_noise(n);
        self.step(state, noise.view())
    }

    pub fn advance_n(&mut self, mut state: MDState, n_steps: usize) -> Result<MDState, MdError> {
        for _ in 0..n_steps {
            state = self.advance(state)?;
        }
        Ok(state)
    }
}

fn standard_normal(rng: &mut rand::rngs::StdRng) -> F {
    use rand::RngExt;
    let u1 = rng.random::<F>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<F>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Broadcast a scalar mass to `(n,)` for a homogeneous system.
pub fn scalar_mass(mass: F, n: usize) -> Result<Array1<F>, MdError> {
    if !mass.is_finite() || mass <= 0.0 {
        return Err(MdError::Invalid("mass must be strictly positive".into()));
    }
    Ok(Array1::from_elem(n, mass))
}

/// Kinetic energy `½ Σ m_i |v_i|²` in the integrator energy unit.
pub fn kinetic_energy(mass: ArrayView1<'_, F>, vel: ArrayView2<'_, F>) -> Result<F, MdError> {
    if mass.len() != vel.nrows() {
        return Err(MdError::Invalid(format!(
            "mass length {} disagrees with n_atoms={}",
            mass.len(),
            vel.nrows()
        )));
    }
    let mut ke = 0.0;
    Zip::from(mass).and(vel.rows()).for_each(|&m, v| {
        ke += m * v.dot(&v);
    });
    Ok(0.5 * ke)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, ArrayView2, array};

    use molrs::spatial::neighbors::{NeighborList, NeighborPolicy, VerletSkin};
    use molrs::spatial::simbox::SimBox;

    use super::super::lj::LJ;
    use super::*;

    fn cube(a: F) -> SimBox {
        SimBox::cube(a, array![0.0, 0.0, 0.0], [true, true, true]).unwrap()
    }

    fn soft_lj_skin(n: usize, box_a: F) -> (LJ, VerletSkin, Array2<F>) {
        let cutoff = 2.5;
        let skin = 0.5;
        let mut pos = Array2::<F>::zeros((n, 3));
        for i in 0..n {
            pos[[i, 0]] = (i as F) * 1.1;
        }
        let nl = VerletSkin::new(
            NeighborList::new(cutoff + skin),
            cutoff,
            NeighborPolicy {
                skin,
                ..NeighborPolicy::default()
            },
            pos.view(),
            cube(box_a),
        )
        .unwrap();
        let lj = LJ::lj126(1.0, 1.0, cutoff).unwrap();
        (lj, nl, pos)
    }

    fn arrays_close(a: ArrayView2<'_, F>, b: ArrayView2<'_, F>, tol: F) -> bool {
        a.shape() == b.shape() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn step_constants_match_the_closed_form() {
        let dt = 0.05;
        let gamma = 2.0;
        let kbt = 1.5;
        let mass = 2.0;
        let (lj, nl, pos) = soft_lj_skin(1, 20.0);
        let ig = Langevin::new(
            dt,
            gamma,
            kbt,
            lj,
            nl,
            scalar_mass(mass, 1).unwrap().view(),
            0,
        )
        .unwrap();
        assert!((ig.c1() - (-gamma * dt).exp()).abs() < 1e-15);
        assert!((ig.c2() - (1.0 - (-2.0 * gamma * dt).exp()).sqrt()).abs() < 1e-15);
        assert!((ig.sigma()[0] - (kbt / mass).sqrt()).abs() < 1e-15);
        assert!((ig.inv_mass()[0] - 1.0 / mass).abs() < 1e-15);
        let _ = pos;
    }

    #[test]
    fn nve_step_preserves_finite_energy() {
        let (lj, nl, pos) = soft_lj_skin(4, 40.0);
        let vel = Array2::zeros(pos.raw_dim());
        let mut ig =
            VelocityVerlet::new(0.01, lj, nl, scalar_mass(1.0, 4).unwrap().view()).unwrap();
        let state = ig.initial(pos, vel).unwrap();
        let next = ig.advance(state).unwrap();
        assert!(next.energy.is_finite());
        assert!(next.forces.iter().all(|f| f.is_finite()));
    }

    #[test]
    fn removed_dof_follows_the_scheme() {
        let (lj, nl, _) = soft_lj_skin(2, 40.0);
        let nve = VelocityVerlet::new(0.01, lj, nl, scalar_mass(1.0, 2).unwrap().view()).unwrap();
        assert_eq!(nve.removed_dof(), 3);
        let (lj, nl, _) = soft_lj_skin(2, 40.0);
        let lgv = Langevin::new(
            0.01,
            1.0,
            1.0,
            lj,
            nl,
            scalar_mass(1.0, 2).unwrap().view(),
            0,
        )
        .unwrap();
        assert_eq!(lgv.removed_dof(), 0);
    }

    #[test]
    fn mass_must_be_positive() {
        let (lj, nl, _) = soft_lj_skin(2, 40.0);
        assert!(VelocityVerlet::new(0.01, lj, nl, array![-1.0, 1.0].view()).is_err());
    }

    #[test]
    fn langevin_rejects_gamma_zero() {
        let (lj, nl, _) = soft_lj_skin(1, 20.0);
        let err = Langevin::new(0.01, 0.0, 1.0, lj, nl, array![1.0].view(), 0);
        match err {
            Err(e) => assert!(e.to_string().contains("VelocityVerlet")),
            Ok(_) => panic!("expected Langevin gamma=0 to fail"),
        }
    }

    #[test]
    fn langevin_rejects_nonpositive_kbt() {
        let (lj, nl, _) = soft_lj_skin(1, 20.0);
        assert!(Langevin::new(0.01, 1.0, 0.0, lj, nl, array![1.0].view(), 0).is_err());
    }

    #[test]
    fn force_caching_one_eval_per_step() {
        // Skin with check: force changes when atoms move inside half-skin.
        let (lj, nl, mut pos) = soft_lj_skin(4, 40.0);
        let mut ig =
            VelocityVerlet::new(0.01, lj, nl, scalar_mass(1.0, 4).unwrap().view()).unwrap();
        let f0 = ig.eval_force(pos.view()).unwrap().forces;
        pos[[1, 0]] += 0.05;
        let f1 = ig.eval_force(pos.view()).unwrap().forces;
        assert!(!arrays_close(f0.view(), f1.view(), 1e-15));
    }

    #[test]
    fn advance_n_matches_manual_advance_loop() {
        let (lj, nl, pos) = soft_lj_skin(4, 40.0);
        let vel = Array2::from_elem(pos.raw_dim(), 0.01);
        let mut a = VelocityVerlet::new(0.01, lj, nl, scalar_mass(1.0, 4).unwrap().view()).unwrap();
        let (lj, nl, _) = soft_lj_skin(4, 40.0);
        let mut b = VelocityVerlet::new(0.01, lj, nl, scalar_mass(1.0, 4).unwrap().view()).unwrap();
        let s0 = a.initial(pos.clone(), vel.clone()).unwrap();
        let end_a = a.advance_n(s0, 5).unwrap();
        let mut state = b.initial(pos, vel).unwrap();
        for _ in 0..5 {
            state = b.advance(state).unwrap();
        }
        assert!(arrays_close(end_a.pos.view(), state.pos.view(), 0.0));
        assert!(arrays_close(end_a.vel.view(), state.vel.view(), 0.0));
    }
}
