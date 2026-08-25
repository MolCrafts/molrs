//! Integrator components: advance an [`MDState`].
//!
//! Required pieces go in the constructor — no `bind_*` afterthoughts:
//!
//! ```ignore
//! VelocityVerlet::new(dt, lj, Some(neighbors), mass)?;
//! Langevin::new(dt, gamma, kbt, potentials, None, mass, seed)?;
//! ```
//!
//! `potential` is anything implementing [`Potential`] (boxed internally):
//! [`super::LJCut`] as the nonbond term, a [`Potentials`] collection to merge
//! several terms (bonded + nonbond + external), or any external
//! implementation. The integrator owns the optional
//! [`VerletSkin`]: every force evaluation runs the skin's update policy and,
//! after a rebuild, feeds the current pairs to the potential
//! ([`Potential::set_pairs`]) — neighbour bookkeeping is the loop's concern,
//! never the potential's. Two schemes, two types — no `gamma=0` switch:
//!
//! [`Potentials`]: molrs::ff::potential::Potentials
//!
//! * [`VelocityVerlet`] — NVE (B-A-A-B; the two half-drifts stay as separate
//!   adds).
//! * [`Langevin`] — BAOAB Langevin (γ > 0). Ordering (Leimkuhler & Matthews):
//!   B (half kick) → A (half drift) → O (Ornstein-Uhlenbeck) → A → B. The O
//!   step `v ← c1·v + c2·σ·ξ` with `c1 = e^{-γΔt}`, `c2 = √(1-c1²)`,
//!   `σ = √(k_BT/m)`.
//!
//! Units are the caller's. MD has no unit knowledge.
//!
//! Reference:
//!     Leimkuhler & Matthews, "Rational Construction of Stochastic Numerical
//!     Methods for Molecular Sampling", Appl. Math. Res. Express 2013.
//!     https://doi.org/10.1093/amrx/abs010

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

use molrs::ff::potential::Potential;
use molrs::spatial::neighbors::VerletSkin;
use molrs::types::{F, FNx3};

use super::error::MdError;
use super::types::{ForceOutput, MDState};

fn as_mass_col(mass: ArrayView1<'_, F>) -> Result<Array2<F>, MdError> {
    if mass.iter().any(|&m| !m.is_finite() || m <= 0.0) {
        return Err(MdError::Invalid("mass must be strictly positive".into()));
    }
    Ok(mass.to_owned().insert_axis(ndarray::Axis(1)))
}

/// One force evaluation: materialize the current pair table once and share it.
fn eval_potential(
    potential: &dyn Potential,
    neighbors: &mut Option<VerletSkin>,
    pos: ArrayView2<'_, F>,
) -> Result<ForceOutput, MdError> {
    let n_atoms = pos.nrows();
    let (energy, forces) = if let Some(skin) = neighbors.as_mut() {
        let pairs = skin.pairs_at(pos)?;
        match pos.as_slice() {
            Some(flat) => potential.calc_energy_forces_with_pairs(flat, pairs),
            None => {
                let flat: Vec<F> = pos.iter().copied().collect();
                potential.calc_energy_forces_with_pairs(&flat, pairs)
            }
        }
    } else {
        match pos.as_slice() {
            Some(flat) => potential.calc_energy_forces(flat),
            None => {
                let flat: Vec<F> = pos.iter().copied().collect();
                potential.calc_energy_forces(&flat)
            }
        }
    };
    let n_components = forces.len();
    let forces = Array2::from_shape_vec((n_atoms, 3), forces).map_err(|_| {
        MdError::Invalid(format!(
            "potential returned {n_components} force components for {n_atoms} atoms"
        ))
    })?;
    Ok(ForceOutput { energy, forces })
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
/// Construct with timestep, a [`Potential`], the optional [`VerletSkin`] the
/// loop runs for the nonbond term, and mass.
pub struct VelocityVerlet {
    dt: F,
    potential: Box<dyn Potential>,
    neighbors: Option<VerletSkin>,
    mass_col: Array2<F>,
    /// Per-atom 1/m (`(N,)`).
    inv_mass: Array1<F>,
}

impl VelocityVerlet {
    /// `dt` (fs), potential, optional neighbour state, per-atom mass `(N,)`.
    pub fn new(
        dt: F,
        potential: impl Potential + 'static,
        neighbors: Option<VerletSkin>,
        mass: ArrayView1<'_, F>,
    ) -> Result<Self, MdError> {
        let mass_col = as_mass_col(mass)?;
        let inv_mass = mass_col.column(0).mapv(|m| 1.0 / m);
        Ok(Self {
            dt,
            potential: Box::new(potential),
            neighbors,
            mass_col,
            inv_mass,
        })
    }

    /// Read-only view of the integrator-owned Verlet skin (rebuild counters,
    /// edge count); `None` when constructed without neighbour state.
    pub fn neighbors(&self) -> Option<&VerletSkin> {
        self.neighbors.as_ref()
    }

    /// Timestep Δt in fs.
    pub fn dt(&self) -> F {
        self.dt
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

    /// Energy and forces at `pos` (runs the neighbour update policy first).
    pub fn eval_force(&mut self, pos: ArrayView2<'_, F>) -> Result<ForceOutput, MdError> {
        eval_potential(&*self.potential, &mut self.neighbors, pos)
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
        let out = self.eval_force(state.pos.view())?;
        state.forces = out.forces;
        // B
        Zip::from(state.vel.rows_mut())
            .and(state.forces.rows())
            .and(&self.inv_mass)
            .for_each(|mut v, f, &im| {
                v[0] += half_dt * f[0] * im;
                v[1] += half_dt * f[1] * im;
                v[2] += half_dt * f[2] * im;
            });
        state.energy = out.energy;
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
    dt: F,
    gamma: F,
    c1: F,
    c2: F,
    kbt: F,
    potential: Box<dyn Potential>,
    neighbors: Option<VerletSkin>,
    mass_col: Array2<F>,
    inv_mass: Array1<F>,
    sigma: Array1<F>,
    rng: rand::rngs::StdRng,
}

impl Langevin {
    /// BAOAB integrator with all required pieces at construction.
    ///
    /// `seed` fixes the internal noise stream, so [`advance`](Self::advance)
    /// is deterministic given the seed.
    pub fn new(
        dt: F,
        gamma: F,
        kbt: F,
        potential: impl Potential + 'static,
        neighbors: Option<VerletSkin>,
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
            potential: Box::new(potential),
            neighbors,
            mass_col,
            inv_mass,
            sigma,
            rng: rand::SeedableRng::seed_from_u64(seed),
        })
    }

    /// Read-only view of the integrator-owned Verlet skin (rebuild counters,
    /// edge count); `None` when constructed without neighbour state.
    pub fn neighbors(&self) -> Option<&VerletSkin> {
        self.neighbors.as_ref()
    }

    /// Timestep Δt in fs.
    pub fn dt(&self) -> F {
        self.dt
    }

    /// Langevin friction γ in fs⁻¹.
    pub fn gamma(&self) -> F {
        self.gamma
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

    /// Energy and forces at `pos` (runs the neighbour update policy first).
    pub fn eval_force(&mut self, pos: ArrayView2<'_, F>) -> Result<ForceOutput, MdError> {
        eval_potential(&*self.potential, &mut self.neighbors, pos)
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

    /// One BAOAB step with caller-supplied standard-normal `noise` `(N, 3)`.
    ///
    /// [`advance`](Self::advance) draws from the seeded internal RNG instead.
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
        let out = self.eval_force(state.pos.view())?;
        state.forces = out.forces;
        // B
        Zip::from(state.vel.rows_mut())
            .and(state.forces.rows())
            .and(&self.inv_mass)
            .for_each(|mut v, f, &im| {
                v[0] += half_dt * f[0] * im;
                v[1] += half_dt * f[1] * im;
                v[2] += half_dt * f[2] * im;
            });
        state.energy = out.energy;
        Ok(state)
    }

    /// Draw `(n_atoms, 3)` standard normals from the seeded internal RNG.
    pub fn draw_noise(&mut self, n_atoms: usize) -> Array2<F> {
        let mut noise = Array2::<F>::zeros((n_atoms, 3));
        for x in noise.iter_mut() {
            *x = standard_normal(&mut self.rng);
        }
        noise
    }

    /// One BAOAB step, noise drawn from the seeded internal RNG.
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

    use molrs::ff::potential::Potentials;
    use molrs::spatial::neighbors::{NeighborList, NeighborPolicy, VerletSkin};
    use molrs::spatial::simbox::SimBox;

    use molrs::units::UnitRegistry;
    use molrs::units::constants::BOLTZMANN;

    use super::super::LJCut;
    use super::super::maxwell::MaxwellBoltzmann;
    use super::*;

    fn cube(a: F) -> SimBox {
        SimBox::cube(a, array![0.0, 0.0, 0.0], [true, true, true]).unwrap()
    }

    fn soft_lj(n: usize, box_a: F) -> (LJCut, VerletSkin, Array2<F>) {
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
        let lj = LJCut::lj126(1.0, 1.0, cutoff).unwrap();
        (lj, nl, pos)
    }

    /// Fixed energy + uniform x-force — a bonded-category stand-in for
    /// merge tests (default no-op `set_pairs`).
    struct Uniform {
        energy: F,
        fx: F,
    }

    impl Potential for Uniform {
        fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
            let mut forces = vec![0.0; coords.len()];
            for row in forces.chunks_mut(3) {
                row[0] = self.fx;
            }
            (self.energy, forces)
        }
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
        let (lj, nl, pos) = soft_lj(1, 20.0);
        let ig = Langevin::new(
            dt,
            gamma,
            kbt,
            lj,
            Some(nl),
            scalar_mass(mass, 1).unwrap().view(),
            0,
        )
        .unwrap();
        assert!((ig.c1() - (-gamma * dt).exp()).abs() < 1e-15);
        assert!((ig.c2() - (1.0 - (-2.0 * gamma * dt).exp()).sqrt()).abs() < 1e-15);
        assert!((ig.sigma()[0] - (kbt / mass).sqrt()).abs() < 1e-15);
        assert!((ig.inv_mass()[0] - 1.0 / mass).abs() < 1e-15);
        assert!((ig.dt() - dt).abs() < 1e-15);
        assert!((ig.gamma() - gamma).abs() < 1e-15);
        let _ = pos;
    }

    /// NVE energy conservation is *the* correctness gate for the whole stack:
    /// LJCut forces (analytic gradient), the VerletSkin rebuild policy (skin
    /// completeness across rebuilds) run through the loop→`set_pairs` seam,
    /// and the B-A-A-B splitting.
    ///
    /// 64 Ar-like atoms (ε = 0.238 kcal/mol, σ = 3.405 Å, m = 39.948 amu) on
    /// a cubic lattice at the pair-minimum spacing, MaxwellBoltzmann
    /// velocities at 100 K (seeded), dt = 1 fs, 1000 steps. Measured relative
    /// drift max|E(t) − E(0)| / |E̅| = 1.71e-5 on this setup (shifted,
    /// unsmeared 12-6: the cutoff force step is the dominant source), in line
    /// with the ~1e-5 the same physics measured at the Python level; asserted
    /// at 5e-5 (≈3x headroom).
    #[test]
    fn nve_conserves_total_energy_over_a_thousand_steps() {
        let epsilon = UnitRegistry::global()
            .quantity(0.238, "kilocalorie_per_mole")
            .unwrap()
            .to_parsed("amu * angstrom ** 2 / femtosecond ** 2")
            .unwrap()
            .value();
        let sigma = 3.405;
        let mass_ar = 39.948;
        let cutoff = 6.0;
        let skin = 0.8;
        let spacing = (2.0_f64).powf(1.0 / 6.0) * sigma;
        let n_side = 4usize;
        let n = n_side * n_side * n_side;
        let mut pos = Array2::<F>::zeros((n, 3));
        for i in 0..n {
            pos[[i, 0]] = (i % n_side) as F * spacing;
            pos[[i, 1]] = ((i / n_side) % n_side) as F * spacing;
            pos[[i, 2]] = (i / (n_side * n_side)) as F * spacing;
        }
        let nl = VerletSkin::new(
            NeighborList::new(cutoff + skin),
            cutoff,
            NeighborPolicy {
                skin,
                ..NeighborPolicy::default()
            },
            pos.view(),
            cube(n_side as F * spacing),
        )
        .unwrap();
        let lj = LJCut::lj126(epsilon, sigma, cutoff).unwrap();
        let mass = scalar_mass(mass_ar, n).unwrap();
        let kb = UnitRegistry::global()
            .quantity(BOLTZMANN, "J / K")
            .unwrap()
            .to_parsed("amu * angstrom ** 2 / femtosecond ** 2 / kelvin")
            .unwrap()
            .value();
        let vel = MaxwellBoltzmann::new(kb * 100.0, 42)
            .unwrap()
            .velocities(pos.view(), mass.view())
            .unwrap();
        let mut ig = VelocityVerlet::new(1.0, lj, Some(nl), mass.view()).unwrap();
        let mut state = ig.initial(pos, vel).unwrap();

        let total = |s: &MDState| s.energy + kinetic_energy(mass.view(), s.vel.view()).unwrap();
        let e0 = total(&state);
        assert!(e0.is_finite());
        let mut e_sum = e0;
        let mut max_dev: F = 0.0;
        let chunks = 100;
        let steps_per_chunk = 10;
        for _ in 0..chunks {
            state = ig.advance_n(state, steps_per_chunk).unwrap();
            let e = total(&state);
            max_dev = max_dev.max((e - e0).abs());
            e_sum += e;
        }
        let e_mean = e_sum / (chunks + 1) as F;
        assert!(
            e_mean.abs() > 1e-4,
            "test setup degenerate: |E_mean| = {:.3e} too close to zero for a relative bound",
            e_mean
        );
        let drift = max_dev / e_mean.abs();
        assert!(
            drift < 5e-5,
            "NVE relative energy drift {drift:.3e} exceeds 5e-5 \
             (E0 = {e0:.6e}, E_mean = {e_mean:.6e}, max|dE| = {max_dev:.3e})"
        );
    }

    #[test]
    fn removed_dof_follows_the_scheme() {
        let (lj, nl, _) = soft_lj(2, 40.0);
        let nve =
            VelocityVerlet::new(0.01, lj, Some(nl), scalar_mass(1.0, 2).unwrap().view()).unwrap();
        assert_eq!(nve.removed_dof(), 3);
        let (lj, nl, _) = soft_lj(2, 40.0);
        let lgv = Langevin::new(
            0.01,
            1.0,
            1.0,
            lj,
            Some(nl),
            scalar_mass(1.0, 2).unwrap().view(),
            0,
        )
        .unwrap();
        assert_eq!(lgv.removed_dof(), 0);
    }

    #[test]
    fn mass_must_be_positive() {
        let (lj, nl, _) = soft_lj(2, 40.0);
        assert!(VelocityVerlet::new(0.01, lj, Some(nl), array![-1.0, 1.0].view()).is_err());
    }

    #[test]
    fn langevin_rejects_gamma_zero() {
        let (lj, nl, _) = soft_lj(1, 20.0);
        let err = Langevin::new(0.01, 0.0, 1.0, lj, Some(nl), array![1.0].view(), 0);
        match err {
            Err(e) => assert!(e.to_string().contains("VelocityVerlet")),
            Ok(_) => panic!("expected Langevin gamma=0 to fail"),
        }
    }

    #[test]
    fn langevin_rejects_nonpositive_kbt() {
        let (lj, nl, _) = soft_lj(1, 20.0);
        assert!(Langevin::new(0.01, 1.0, 0.0, lj, Some(nl), array![1.0].view(), 0).is_err());
    }

    #[test]
    fn langevin_advance_is_deterministic_given_the_seed() {
        let (lj, nl, pos) = soft_lj(4, 40.0);
        let vel = Array2::from_elem(pos.raw_dim(), 0.01);
        let mut a = Langevin::new(
            0.01,
            1.0,
            1.0,
            lj,
            Some(nl),
            scalar_mass(1.0, 4).unwrap().view(),
            9,
        )
        .unwrap();
        let (lj, nl, _) = soft_lj(4, 40.0);
        let mut b = Langevin::new(
            0.01,
            1.0,
            1.0,
            lj,
            Some(nl),
            scalar_mass(1.0, 4).unwrap().view(),
            9,
        )
        .unwrap();
        let sa = a.initial(pos.clone(), vel.clone()).unwrap();
        let sb = b.initial(pos, vel).unwrap();
        let ea = a.advance_n(sa, 3).unwrap();
        let eb = b.advance_n(sb, 3).unwrap();
        assert!(arrays_close(ea.pos.view(), eb.pos.view(), 0.0));
        assert!(arrays_close(ea.vel.view(), eb.vel.view(), 0.0));
    }

    #[test]
    fn force_caching_one_eval_per_step() {
        // Skin with check: force changes when atoms move inside half-skin.
        let (lj, nl, mut pos) = soft_lj(4, 40.0);
        let mut ig =
            VelocityVerlet::new(0.01, lj, Some(nl), scalar_mass(1.0, 4).unwrap().view()).unwrap();
        let f0 = ig.eval_force(pos.view()).unwrap().forces;
        pos[[1, 0]] += 0.05;
        let f1 = ig.eval_force(pos.view()).unwrap().forces;
        assert!(!arrays_close(f0.view(), f1.view(), 1e-15));
    }

    #[test]
    fn advance_n_matches_manual_advance_loop() {
        let (lj, nl, pos) = soft_lj(4, 40.0);
        let vel = Array2::from_elem(pos.raw_dim(), 0.01);
        let mut a =
            VelocityVerlet::new(0.01, lj, Some(nl), scalar_mass(1.0, 4).unwrap().view()).unwrap();
        let (lj, nl, _) = soft_lj(4, 40.0);
        let mut b =
            VelocityVerlet::new(0.01, lj, Some(nl), scalar_mass(1.0, 4).unwrap().view()).unwrap();
        let s0 = a.initial(pos.clone(), vel.clone()).unwrap();
        let end_a = a.advance_n(s0, 5).unwrap();
        let mut state = b.initial(pos, vel).unwrap();
        for _ in 0..5 {
            state = b.advance(state).unwrap();
        }
        assert!(arrays_close(end_a.pos.view(), state.pos.view(), 0.0));
        assert!(arrays_close(end_a.vel.view(), state.vel.view(), 0.0));
    }

    #[test]
    fn potentials_merge_nonbond_and_bonded_terms() {
        // A Potentials collection [LJCut, Uniform] through the integrator
        // must equal the lone LJ evaluation plus the uniform offsets.
        let (lj, nl, pos) = soft_lj(2, 40.0);
        let mut lone =
            VelocityVerlet::new(0.01, lj, Some(nl), scalar_mass(1.0, 2).unwrap().view()).unwrap();
        let base = lone.eval_force(pos.view()).unwrap();

        let (lj, nl, _) = soft_lj(2, 40.0);
        let mut pots = Potentials::new();
        pots.push(Box::new(lj));
        pots.push(Box::new(Uniform {
            energy: 0.25,
            fx: -1.5,
        }));
        let mut ig =
            VelocityVerlet::new(0.01, pots, Some(nl), scalar_mass(1.0, 2).unwrap().view()).unwrap();
        let out = ig.eval_force(pos.view()).unwrap();
        assert!((out.energy - (base.energy + 0.25)).abs() < 1e-12);
        for i in 0..2 {
            assert!((out.forces[[i, 0]] - (base.forces[[i, 0]] - 1.5)).abs() < 1e-12);
            assert!((out.forces[[i, 1]] - base.forces[[i, 1]]).abs() < 1e-12);
        }
    }

    #[test]
    fn empty_potentials_is_the_zero_potential() {
        let pos = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut ig = VelocityVerlet::new(
            0.01,
            Potentials::new(),
            None,
            scalar_mass(1.0, 2).unwrap().view(),
        )
        .unwrap();
        let out = ig.eval_force(pos.view()).unwrap();
        assert_eq!(out.energy, 0.0);
        assert!(out.forces.iter().all(|&f| f == 0.0));
    }

    #[test]
    fn neighbor_rebuilds_flow_from_the_loop_to_the_nonbond_potential() {
        // March one atom far enough to force repeated skin rebuilds; the
        // integrator must feed each fresh pair list to the potential, so the
        // final force must match a freshly wired integrator at the final
        // geometry.
        let cutoff = 2.0;
        let skin = 1.0;
        let pos0 = array![[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]];
        let make = |pos: ArrayView2<'_, F>| {
            let nl = VerletSkin::new(
                NeighborList::new(cutoff + skin),
                cutoff,
                NeighborPolicy {
                    skin,
                    ..NeighborPolicy::default()
                },
                pos,
                cube(20.0),
            )
            .unwrap();
            let lj = LJCut::lj126(1.0, 1.0, cutoff).unwrap();
            VelocityVerlet::new(0.01, lj, Some(nl), scalar_mass(1.0, 2).unwrap().view()).unwrap()
        };
        let mut ig = make(pos0.view());
        let mut x = 1.1;
        let mut last = None;
        for _ in 0..8 {
            x += 0.4;
            let pos = array![[0.0, 0.0, 0.0], [x, 0.0, 0.0]];
            last = Some(ig.eval_force(pos.view()).unwrap());
        }
        let rebuilds = ig.neighbors().unwrap().rebuild_count();
        assert!(
            rebuilds >= 2,
            "expected repeated rebuilds on a moving system, got {rebuilds}"
        );
        let pos_end = array![[0.0, 0.0, 0.0], [x, 0.0, 0.0]];
        let mut fresh = make(pos_end.view());
        let reference = fresh.eval_force(pos_end.view()).unwrap();
        let marched = last.unwrap();
        assert!((marched.energy - reference.energy).abs() < 1e-12);
        assert!(arrays_close(
            marched.forces.view(),
            reference.forces.view(),
            1e-12
        ));
    }
}
