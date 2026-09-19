//! Integrator components: advance an [`MDState`].
//!
//! Required pieces go in the constructor — no `bind_*` afterthoughts:
//!
//! ```ignore
//! VelocityVerlet::new(dt, MicPairs::new(lj, skin).unwrap(), mass, Some(bx))?;
//! Langevin::new(dt, gamma, kbt, Direct::new(potentials), mass, seed, None)?;
//! ```
//!
//! The second argument is a [`ForceProvider`],
//! and it is the only thing an integrator knows about force fields. The
//! potential, the neighbour bookkeeping and the periodic régime all live behind
//! it: [`Direct`](super::forces::Direct) hands the potential raw coordinates,
//! [`MicPairs`](super::forces::MicPairs) gives it minimum-image pairs, and
//! [`GhostPairs`](super::forces::GhostPairs) gives it periodic copies and folds
//! the forces back. An integrator holds no skin, no halo and no potential, so
//! adding a fourth way to make a force changes nothing here.
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
//! Units are the caller's. MD has no unit knowledge.
//!
//! Reference:
//!     Leimkuhler & Matthews, "Rational Construction of Stochastic Numerical
//!     Methods for Molecular Sampling", Appl. Math. Res. Express 2013.
//!     <https://doi.org/10.1093/amrx/abs010>

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

use molrs::spatial::simbox::SimBox;

use super::forces::ForceProvider;
use molrs::types::{F, FNx3};

use super::error::MdError;
use super::types::{ForceOutput, MDState};

fn as_mass_col(mass: ArrayView1<'_, F>) -> Result<Array2<F>, MdError> {
    if mass.iter().any(|&m| !m.is_finite() || m <= 0.0) {
        return Err(MdError::Invalid("mass must be strictly positive".into()));
    }
    Ok(mass.to_owned().insert_axis(ndarray::Axis(1)))
}

/// Fold the drifted positions back into the cell and bank the crossings.
///
/// This runs after **every** position update, not only at a neighbour rebuild.
/// LAMMPS remaps at reneighbouring because its `exchange`/`borders` transaction
/// is bound to that point; here the cost is one pass that touches only the
/// atoms that actually crossed — `SimBox::wrap` returns an in-cell point
/// untouched to the bit — and in exchange the coordinates never grow past the
/// cell, so nothing downstream has to reason about how far they might have
/// drifted.
///
/// `m` is the shift the wrap actually applied, so the flags cannot disagree
/// with the positions they belong to.
fn wrap_and_bank(simbox: Option<&SimBox>, state: &mut MDState) -> Array2<i64> {
    let Some(bx) = simbox else {
        return Array2::zeros((state.pos.nrows(), 3));
    };
    let (wrapped, m) = bx.wrap_shifts(state.pos.view());
    state.pos = wrapped;
    state.images += &m;
    m
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
/// Construct with timestep, the [`ForceProvider`] the loop evaluates, and
/// mass.
/// The mechanics both schemes share.
///
/// [`VelocityVerlet`] and [`Langevin`] differ in which moves they make and in
/// what order — B-A-A-B against B-A-O-A-B — and in nothing else. Everything
/// else was two copies: the force seam, the cell, the masses, the entry
/// evaluation, the half kick, the half drift, and the fold that has to sit
/// between the last drift and the force. Two copies of a thing that must agree
/// is two places to change it and one place to forget.
struct Stepper {
    dt: F,
    /// Everything that makes a force, behind one seam.
    forces: Box<dyn ForceProvider>,
    /// Cell the positions are folded into each step; `None` is free boundary.
    simbox: Option<SimBox>,
    mass_col: Array2<F>,
    /// Per-atom 1/m (`(N,)`).
    inv_mass: Array1<F>,
}

impl Stepper {
    fn new(
        dt: F,
        forces: impl ForceProvider + 'static,
        mass: ArrayView1<'_, F>,
        simbox: Option<SimBox>,
    ) -> Result<Self, MdError> {
        let mass_col = as_mass_col(mass)?;
        let inv_mass = mass_col.column(0).mapv(|m| 1.0 / m);
        Ok(Self {
            dt,
            forces: Box::new(forces),
            simbox,
            mass_col,
            inv_mass,
        })
    }

    /// Energy and forces at `pos`, for a caller that has not folded anything.
    fn eval_force(&mut self, pos: ArrayView2<'_, F>) -> Result<ForceOutput, MdError> {
        let no_fold = Array2::zeros((pos.nrows(), 3));
        self.forces.compute(pos, no_fold.view())
    }

    /// Seed an [`MDState`], evaluating the entry force.
    fn initial(&mut self, pos: FNx3, vel: FNx3) -> Result<MDState, MdError> {
        check_state_shape(pos.view(), vel.view(), self.mass_col.nrows())?;
        // Fold the entry configuration too, so step 0 already satisfies the
        // invariant every later step maintains. Flags start at zero: they count
        // crossings *during this run*, and an atom's history before it is not
        // this integrator's to claim.
        let n_atoms = pos.nrows();
        let mut state = MDState {
            pos,
            images: Array2::zeros((n_atoms, 3)),
            vel,
            forces: FNx3::zeros((n_atoms, 3)),
            energy: 0.0,
            virial: None,
        };
        let folded = wrap_and_bank(self.simbox.as_ref(), &mut state);
        state.images.fill(0);
        let seeded = self.forces.compute(state.pos.view(), folded.view())?;
        state.forces = seeded.forces;
        state.energy = seeded.energy;
        state.virial = seeded.virial;
        Ok(state)
    }

    /// **B** — half kick, `v += (Δt/2)·f/m`.
    fn kick(&self, state: &mut MDState, half_dt: F) {
        Zip::from(state.vel.rows_mut())
            .and(state.forces.rows())
            .and(&self.inv_mass)
            .for_each(|mut v, f, &im| {
                v[0] += half_dt * f[0] * im;
                v[1] += half_dt * f[1] * im;
                v[2] += half_dt * f[2] * im;
            });
    }

    /// **A** — half drift, `x += (Δt/2)·v`. The two halves of a full drift stay
    /// separate adds rather than one `dt * v`.
    fn drift(&self, state: &mut MDState, half_dt: F) {
        Zip::from(state.pos.rows_mut())
            .and(state.vel.rows())
            .for_each(|mut p, v| {
                p[0] += half_dt * v[0];
                p[1] += half_dt * v[1];
                p[2] += half_dt * v[2];
            });
    }

    /// Fold the drifted positions back into the cell and evaluate the force
    /// there, caching both on `state`.
    ///
    /// The fold has to happen after the last drift and before the force, and
    /// this is the only place either scheme does it. A halo has to reconcile
    /// that fold in the same breath — which is why the shift is handed to
    /// [`ForceProvider::compute`] rather than re-derived from the positions,
    /// where it cannot be seen: a fold relabels an atom without moving it.
    fn refold_and_eval(&mut self, state: &mut MDState) -> Result<(), MdError> {
        let folded = wrap_and_bank(self.simbox.as_ref(), state);
        let out = self.forces.compute(state.pos.view(), folded.view())?;
        state.forces = out.forces;
        state.energy = out.energy;
        state.virial = out.virial;
        Ok(())
    }
}

pub struct VelocityVerlet {
    inner: Stepper,
}

impl VelocityVerlet {
    /// `dt` (fs), the force provider, per-atom mass `(N,)`, and the cell
    /// positions are folded into each step (`None` = free boundary, no
    /// wrapping and image flags stay zero).
    pub fn new(
        dt: F,
        forces: impl ForceProvider + 'static,
        mass: ArrayView1<'_, F>,
        simbox: Option<SimBox>,
    ) -> Result<Self, MdError> {
        Ok(Self {
            inner: Stepper::new(dt, forces, mass, simbox)?,
        })
    }

    /// The force provider, for the counters it chooses to expose.
    pub fn forces(&self) -> &dyn ForceProvider {
        &*self.inner.forces
    }

    /// Timestep Δt in fs.
    pub fn dt(&self) -> F {
        self.inner.dt
    }

    /// Per-atom mass column `(N, 1)`.
    pub fn mass(&self) -> &Array2<F> {
        &self.inner.mass_col
    }

    /// Inverse mass `(N,)`.
    pub fn inv_mass(&self) -> &Array1<F> {
        &self.inner.inv_mass
    }

    /// Degrees of freedom the temperature estimator must not count (`3N − 3`).
    pub fn removed_dof(&self) -> usize {
        3
    }

    /// Energy and forces at `pos` (runs the neighbour update policy first).
    pub fn eval_force(&mut self, pos: ArrayView2<'_, F>) -> Result<ForceOutput, MdError> {
        self.inner.eval_force(pos)
    }

    /// Seed an [`MDState`], evaluating the entry force.
    pub fn initial(&mut self, pos: FNx3, vel: FNx3) -> Result<MDState, MdError> {
        self.inner.initial(pos, vel)
    }

    /// One NVE step from the cached entry force: **B-A-A-B**.
    pub fn step(&mut self, mut state: MDState) -> Result<MDState, MdError> {
        let half_dt = 0.5 * self.inner.dt;
        self.inner.kick(&mut state, half_dt);
        self.inner.drift(&mut state, half_dt);
        self.inner.drift(&mut state, half_dt);
        self.inner.refold_and_eval(&mut state)?;
        self.inner.kick(&mut state, half_dt);
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
    inner: Stepper,
    gamma: F,
    c1: F,
    c2: F,
    kbt: F,
    sigma: Array1<F>,
    rng: rand::rngs::StdRng,
}

impl Langevin {
    /// BAOAB integrator with all required pieces at construction.
    ///
    /// Seven arguments is the cost of this module's rule that required pieces
    /// go in the constructor rather than into `bind_*` setters afterwards: a
    /// half-built integrator is a state nobody should be able to hold.
    ///
    /// `seed` fixes the internal noise stream, so [`advance`](Self::advance)
    /// is deterministic given the seed.
    pub fn new(
        dt: F,
        gamma: F,
        kbt: F,
        forces: impl ForceProvider + 'static,
        mass: ArrayView1<'_, F>,
        seed: u64,
        simbox: Option<SimBox>,
    ) -> Result<Self, MdError> {
        if gamma <= 0.0 {
            return Err(MdError::Invalid(
                "Langevin requires gamma > 0; use VelocityVerlet for NVE".into(),
            ));
        }
        if kbt <= 0.0 {
            return Err(MdError::Invalid("Langevin requires kbt > 0".into()));
        }
        let inner = Stepper::new(dt, forces, mass, simbox)?;
        let sigma = inner.mass_col.column(0).mapv(|m| (kbt / m).sqrt());
        let c1 = (-gamma * dt).exp();
        // `1 − e^{−2γΔt}` written directly loses a digit for every decade that
        // `γΔt` is below one — at `γΔt = 1e-8` the noise amplitude keeps barely
        // half its bits, and the sampled temperature carries the error.
        // `exp_m1` computes it to the last bit at any `γΔt`.
        let c2 = (-(-2.0 * gamma * dt).exp_m1()).max(0.0).sqrt();
        Ok(Self {
            inner,
            gamma,
            c1,
            c2,
            kbt,
            sigma,
            rng: rand::SeedableRng::seed_from_u64(seed),
        })
    }

    /// The force provider, for the counters it chooses to expose.
    pub fn forces(&self) -> &dyn ForceProvider {
        &*self.inner.forces
    }

    /// Timestep Δt in fs.
    pub fn dt(&self) -> F {
        self.inner.dt
    }

    /// Friction γ (1/fs).
    pub fn gamma(&self) -> F {
        self.gamma
    }

    /// `c1 = e^{−γΔt}` — the velocity damping of the O step.
    pub fn c1(&self) -> F {
        self.c1
    }

    /// `c2 = √(1 − c1²)` — the noise amplitude of the O step.
    pub fn c2(&self) -> F {
        self.c2
    }

    /// Target `k_B T` in the caller's energy units.
    pub fn kbt(&self) -> F {
        self.kbt
    }

    /// Per-atom mass column `(N, 1)`.
    pub fn mass(&self) -> &Array2<F> {
        &self.inner.mass_col
    }

    /// Per-atom `σ = √(k_BT/m)` `(N,)`.
    pub fn sigma(&self) -> &Array1<F> {
        &self.sigma
    }

    /// Inverse mass `(N,)`.
    pub fn inv_mass(&self) -> &Array1<F> {
        &self.inner.inv_mass
    }

    /// Degrees of freedom the temperature estimator must not count.
    ///
    /// Zero: the thermostat does not conserve momentum, so nothing is removed.
    pub fn removed_dof(&self) -> usize {
        0
    }

    /// Energy and forces at `pos` (runs the neighbour update policy first).
    pub fn eval_force(&mut self, pos: ArrayView2<'_, F>) -> Result<ForceOutput, MdError> {
        self.inner.eval_force(pos)
    }

    /// Seed an [`MDState`], evaluating the entry force.
    pub fn initial(&mut self, pos: FNx3, vel: FNx3) -> Result<MDState, MdError> {
        self.inner.initial(pos, vel)
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
        let half_dt = 0.5 * self.inner.dt;
        self.inner.kick(&mut state, half_dt);
        self.inner.drift(&mut state, half_dt);
        self.ornstein_uhlenbeck(&mut state, noise);
        self.inner.drift(&mut state, half_dt);
        self.inner.refold_and_eval(&mut state)?;
        self.inner.kick(&mut state, half_dt);
        Ok(state)
    }

    /// **O** — `v ← c1·v + c2·σ·ξ`. The only move `VelocityVerlet` does not
    /// make, and the only reason these are two types.
    fn ornstein_uhlenbeck(&self, state: &mut MDState, noise: ArrayView2<'_, F>) {
        let (c1, c2) = (self.c1, self.c2);
        Zip::from(state.vel.rows_mut())
            .and(&self.sigma)
            .and(noise.rows())
            .for_each(|mut v, &sig, xi| {
                v[0] = c1 * v[0] + c2 * sig * xi[0];
                v[1] = c1 * v[1] + c2 * sig * xi[1];
                v[2] = c1 * v[2] + c2 * sig * xi[2];
            });
    }

    /// Draw `(n_atoms, 3)` standard normals from the seeded internal RNG.
    pub fn draw_noise(&mut self, n_atoms: usize) -> Array2<F> {
        let mut noise = Array2::<F>::zeros((n_atoms, 3));
        for x in noise.iter_mut() {
            *x = standard_normal(&mut self.rng);
        }
        noise
    }

    /// One step with noise drawn from the seeded internal RNG.
    pub fn advance(&mut self, state: MDState) -> Result<MDState, MdError> {
        let noise = self.draw_noise(state.vel.nrows());
        self.step(state, noise.view())
    }

    /// Advance `n_steps` eagerly.
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

    use molrs::ff::potential::{Potential, Potentials};
    use molrs::spatial::neighbors::{NeighborList, NeighborPolicy, VerletSkin};
    use molrs::spatial::simbox::SimBox;

    use molrs::units::UnitRegistry;
    use molrs::units::constants::BOLTZMANN;

    use super::super::forces::{Direct, MicPairs};
    use super::super::maxwell::MaxwellBoltzmann;
    use super::*;
    use molrs::ff::potential::pair::LJCut;

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
            MicPairs::new(lj, nl).unwrap(),
            scalar_mass(mass, 1).unwrap().view(),
            0,
            None,
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
        let mut ig =
            VelocityVerlet::new(1.0, MicPairs::new(lj, nl).unwrap(), mass.view(), None).unwrap();
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
        let nve = VelocityVerlet::new(
            0.01,
            MicPairs::new(lj, nl).unwrap(),
            scalar_mass(1.0, 2).unwrap().view(),
            None,
        )
        .unwrap();
        assert_eq!(nve.removed_dof(), 3);
        let (lj, nl, _) = soft_lj(2, 40.0);
        let lgv = Langevin::new(
            0.01,
            1.0,
            1.0,
            MicPairs::new(lj, nl).unwrap(),
            scalar_mass(1.0, 2).unwrap().view(),
            0,
            None,
        )
        .unwrap();
        assert_eq!(lgv.removed_dof(), 0);
    }

    #[test]
    fn mass_must_be_positive() {
        let (lj, nl, _) = soft_lj(2, 40.0);
        assert!(
            VelocityVerlet::new(
                0.01,
                MicPairs::new(lj, nl).unwrap(),
                array![-1.0, 1.0].view(),
                None
            )
            .is_err()
        );
    }

    #[test]
    fn langevin_rejects_gamma_zero() {
        let (lj, nl, _) = soft_lj(1, 20.0);
        let err = Langevin::new(
            0.01,
            0.0,
            1.0,
            MicPairs::new(lj, nl).unwrap(),
            array![1.0].view(),
            0,
            None,
        );
        match err {
            Err(e) => assert!(e.to_string().contains("VelocityVerlet")),
            Ok(_) => panic!("expected Langevin gamma=0 to fail"),
        }
    }

    #[test]
    fn langevin_rejects_nonpositive_kbt() {
        let (lj, nl, _) = soft_lj(1, 20.0);
        assert!(
            Langevin::new(
                0.01,
                1.0,
                0.0,
                MicPairs::new(lj, nl).unwrap(),
                array![1.0].view(),
                0,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn langevin_advance_is_deterministic_given_the_seed() {
        let (lj, nl, pos) = soft_lj(4, 40.0);
        let vel = Array2::from_elem(pos.raw_dim(), 0.01);
        let mut a = Langevin::new(
            0.01,
            1.0,
            1.0,
            MicPairs::new(lj, nl).unwrap(),
            scalar_mass(1.0, 4).unwrap().view(),
            9,
            None,
        )
        .unwrap();
        let (lj, nl, _) = soft_lj(4, 40.0);
        let mut b = Langevin::new(
            0.01,
            1.0,
            1.0,
            MicPairs::new(lj, nl).unwrap(),
            scalar_mass(1.0, 4).unwrap().view(),
            9,
            None,
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
        let mut ig = VelocityVerlet::new(
            0.01,
            MicPairs::new(lj, nl).unwrap(),
            scalar_mass(1.0, 4).unwrap().view(),
            None,
        )
        .unwrap();
        let f0 = ig.eval_force(pos.view()).unwrap().forces;
        pos[[1, 0]] += 0.05;
        let f1 = ig.eval_force(pos.view()).unwrap().forces;
        assert!(!arrays_close(f0.view(), f1.view(), 1e-15));
    }

    #[test]
    fn advance_n_matches_manual_advance_loop() {
        let (lj, nl, pos) = soft_lj(4, 40.0);
        let vel = Array2::from_elem(pos.raw_dim(), 0.01);
        let mut a = VelocityVerlet::new(
            0.01,
            MicPairs::new(lj, nl).unwrap(),
            scalar_mass(1.0, 4).unwrap().view(),
            None,
        )
        .unwrap();
        let (lj, nl, _) = soft_lj(4, 40.0);
        let mut b = VelocityVerlet::new(
            0.01,
            MicPairs::new(lj, nl).unwrap(),
            scalar_mass(1.0, 4).unwrap().view(),
            None,
        )
        .unwrap();
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
        let mut lone = VelocityVerlet::new(
            0.01,
            MicPairs::new(lj, nl).unwrap(),
            scalar_mass(1.0, 2).unwrap().view(),
            None,
        )
        .unwrap();
        let base = lone.eval_force(pos.view()).unwrap();

        let (lj, nl, _) = soft_lj(2, 40.0);
        let mut pots = Potentials::new();
        pots.push(Box::new(lj));
        pots.push(Box::new(Uniform {
            energy: 0.25,
            fx: -1.5,
        }));
        let mut ig = VelocityVerlet::new(
            0.01,
            MicPairs::new(pots, nl).unwrap(),
            scalar_mass(1.0, 2).unwrap().view(),
            None,
        )
        .unwrap();
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
            Direct::new(Potentials::new()),
            scalar_mass(1.0, 2).unwrap().view(),
            None,
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
            VelocityVerlet::new(
                0.01,
                MicPairs::new(lj, nl).unwrap(),
                scalar_mass(1.0, 2).unwrap().view(),
                None,
            )
            .unwrap()
        };
        let mut ig = make(pos0.view());
        let mut x = 1.1;
        let mut last = None;
        for _ in 0..8 {
            x += 0.4;
            let pos = array![[0.0, 0.0, 0.0], [x, 0.0, 0.0]];
            last = Some(ig.eval_force(pos.view()).unwrap());
        }
        let rebuilds = ig
            .forces()
            .neighbor_stats()
            .rebuilds
            .expect("this integrator was built with a minimum-image skin");
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

#[cfg(test)]
mod ghost_path_tests {
    use super::super::forces::{GhostPairs, MicPairs};
    use super::*;
    use molrs::ff::potential::pair::LJCut;
    use molrs::spatial::neighbors::{NeighborList, NeighborPolicy, VerletSkin};
    use molrs::spatial::simbox::SimBox;
    use ndarray::array;

    use super::super::pairs::Comm;

    /// The two periodic régimes are two ways of computing one thing, and this
    /// is the assertion that says so: the same system, the same steps, once
    /// through the minimum image and once through ghost atoms, has to produce
    /// the same trajectory.
    ///
    /// If it does not, one of them is wrong — and the ghost route is the one MD
    /// will use for potentials that read positions, so a disagreement here is a
    /// disagreement about physics, not about bookkeeping.
    #[test]
    fn the_ghost_route_and_the_mic_route_are_the_same_trajectory() {
        let l = 12.0_f64;
        let cutoff = 5.0;
        // Zero skin on both sides, deliberately. A Verlet skin is a *caching*
        // policy, not a periodic one: it hands back the edges frozen at the
        // last rebuild, so a pair that closes from `cutoff + skin` to `cutoff`
        // between rebuilds is invisible until the next one. The ghost route as
        // written searches afresh every step and has no such staleness, so with
        // a skin the two disagree by that approximation — a real difference,
        // and not the one under test here. Setting it to zero makes both exact
        // every step and leaves only the question this test asks: do the two
        // periodic régimes compute the same physics?
        let skin = 0.0;
        let bx = SimBox::cube(l, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();

        // Eight atoms on a 4 Å cube whose every edge runs *through* a face:
        // 11.0 to 3.0 is 4 Å the short way round in a 12 Å cell. Nearest
        // neighbours sit at 4 Å, just past the LJ minimum (2^(1/6)·σ = 3.82 Å),
        // so the system is bound rather than exploding, and face diagonals at
        // 5.66 Å fall outside the 5 Å cutoff. Every interaction is periodic.
        let pos0 = array![
            [11.0_f64, 11.0, 11.0],
            [3.0, 11.0, 11.0],
            [11.0, 3.0, 11.0],
            [11.0, 11.0, 3.0],
            [3.0, 3.0, 11.0],
            [3.0, 11.0, 3.0],
            [11.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
        ];
        let vel0 = array![
            [0.020_f64, -0.008, 0.012],
            [-0.016, 0.010, 0.006],
            [0.004, 0.018, -0.014],
            [-0.012, -0.006, 0.016],
            [0.008, 0.014, -0.004],
            [-0.018, 0.002, 0.010],
            [0.006, -0.016, -0.008],
            [0.014, 0.004, 0.018],
        ];
        let n = pos0.nrows();
        let lj = || LJCut::new(0.3, 3.4, cutoff, 12, 6, false, false).unwrap();
        let mass = scalar_mass(12.0, n).unwrap();

        // --- minimum image ---
        let skin_nl = VerletSkin::new(
            NeighborList::new(cutoff + skin),
            cutoff,
            NeighborPolicy {
                skin,
                ..NeighborPolicy::default()
            },
            pos0.view(),
            bx.clone(),
        )
        .unwrap();
        let mut mic = VelocityVerlet::new(
            1.0,
            MicPairs::new(lj(), skin_nl).unwrap(),
            mass.view(),
            Some(bx.clone()),
        )
        .unwrap();

        // --- ghosts ---
        let comm = Comm::new(bx.clone(), pos0.view(), cutoff, skin).unwrap();
        let mut gho = VelocityVerlet::new(
            1.0,
            GhostPairs::new(lj(), comm).unwrap(),
            mass.view(),
            Some(bx.clone()),
        )
        .unwrap();

        let mut a = mic.initial(pos0.clone(), vel0.clone()).unwrap();
        let mut b = gho.initial(pos0.clone(), vel0.clone()).unwrap();

        let scale = a.energy.abs().max(1.0);
        assert!(
            (a.energy - b.energy).abs() / scale < 1e-11,
            "entry energy: mic {} vs ghosts {}",
            a.energy,
            b.energy
        );

        // Molecular dynamics is chaotic, so two routes that are mathematically
        // identical but sum the same pairs in different orders separate
        // exponentially: ~1e-16 of relative disagreement per step, amplified by
        // the Lyapunov instability of the trajectory itself. The bar is
        // therefore tight early, where a real defect would already show, and
        // loose later, where only the amplification remains. A missing or
        // double-counted pair moves the energy by O(0.1) kcal/mol and fails
        // either bar immediately.
        for step in 1..=150 {
            a = mic.advance(a).unwrap();
            b = gho.advance(b).unwrap();

            let (tol_e, tol_x) = if step <= 20 {
                (1e-12, 1e-11)
            } else {
                (1e-7, 1e-7)
            };

            let scale = a.energy.abs().max(1.0);
            assert!(
                (a.energy - b.energy).abs() / scale < tol_e,
                "step {step} energy: mic {} vs ghosts {}",
                a.energy,
                b.energy
            );
            for i in 0..n {
                for k in 0..3 {
                    assert!(
                        (a.pos[[i, k]] - b.pos[[i, k]]).abs() < tol_x,
                        "step {step} atom {i} axis {k}: mic {} vs ghosts {}",
                        a.pos[[i, k]],
                        b.pos[[i, k]]
                    );
                }
                assert_eq!(
                    a.images.row(i).to_vec(),
                    b.images.row(i).to_vec(),
                    "step {step} atom {i}: image flags must agree"
                );
            }
        }

        // The run actually exercised the periodic machinery rather than
        // trivially agreeing: atoms crossed, and the halo was rebuilt.
        let crossed: i64 = a.images.iter().map(|&m| m.abs()).sum();
        assert!(
            crossed > 0,
            "no atom crossed a face; the test proves nothing"
        );
        let rebuilds = gho
            .forces()
            .neighbor_stats()
            .rebuilds
            .expect("a ghost provider counts its rebuilds");
        assert!(rebuilds > 0, "the halo was never rebuilt");
    }

    /// A halo that outlives a fold has to be reconciled with it, and the
    /// observable consequence is that nothing happens: the energy does not jump
    /// when an atom crosses a face.
    ///
    /// This is deliberately *not* a comparison against the minimum-image route.
    /// A skin makes the two differ by list staleness, and with the skin set to
    /// zero — as the comparison test must — the halo is rebuilt every step and
    /// the reconciliation is never reached. So the property is asserted on its
    /// own terms: a crossing is not a physical event, so no physical quantity
    /// may notice one.
    ///
    /// Without `s_g += m` a copy jumps a whole cell when its owner folds, pairs
    /// appear and vanish, and the energy steps by kcal/mol between one
    /// femtosecond and the next.
    #[test]
    fn crossing_a_face_does_not_disturb_the_energy() {
        let l = 12.0_f64;
        let cutoff = 5.0;
        let bx = SimBox::cube(l, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();

        // Sitting a tenth of an Ångström inside the +x face, so the crossing
        // happens within the first few steps and the configuration has no time
        // to evolve far from where it started. Neighbours are still 4 Å apart
        // the short way round (12 − 11.9 + 3.9).
        let pos0 = array![
            [11.9_f64, 11.9, 11.9],
            [3.9, 11.9, 11.9],
            [11.9, 3.9, 11.9],
            [11.9, 11.9, 3.9],
            [3.9, 3.9, 11.9],
            [3.9, 11.9, 3.9],
            [11.9, 3.9, 3.9],
            [3.9, 3.9, 3.9],
        ];
        // A uniform drift along +x, so every atom crosses. The
        // lattice is not in equilibrium — on each axis an atom has a neighbour
        // at 4 Å one way and 8 Å the other — so the potential energy genuinely
        // evolves, and asserting it is *constant* would be wrong. What must
        // hold is that the total energy is conserved: NVE has no term that
        // could absorb a pair appearing or vanishing.
        let drift = 0.02_f64;
        let vel0 =
            FNx3::from_shape_fn((pos0.nrows(), 3), |(_, k)| if k == 0 { drift } else { 0.0 });

        // A skin large enough that the halo survives many steps, so folds
        // happen *between* rebuilds — the case the reconciliation exists for.
        let comm = Comm::new(bx.clone(), pos0.view(), cutoff, 0.8).unwrap();
        // 0.2 fs: at 1 fs the velocity-Verlet truncation error on this LJ
        // lattice is itself 1e-4 of the total energy, which would swamp the
        // signal this test is looking for.
        let mut ig = VelocityVerlet::new(
            0.2,
            GhostPairs::new(
                LJCut::new(0.3, 3.4, cutoff, 12, 6, false, false).unwrap(),
                comm,
            )
            .unwrap(),
            scalar_mass(12.0, pos0.nrows()).unwrap().view(),
            Some(bx.clone()),
        )
        .unwrap();

        let mass = scalar_mass(12.0, pos0.nrows()).unwrap();
        let mut state = ig.initial(pos0.clone(), vel0).unwrap();
        let total = |st: &MDState| st.energy + kinetic_energy(mass.view(), st.vel.view()).unwrap();
        let e0 = total(&state);
        let scale = e0.abs().max(1.0);

        for step in 1..=200 {
            state = ig.advance(state).unwrap();
            let drift = (total(&state) - e0).abs() / scale;
            // The bar is set by what the defect does, not by what looks tidy.
            // A copy that jumps a cell removes or adds a pair worth ~0.1
            // kcal/mol from a total near 4, i.e. ~2.5e-2 of relative change in
            // a single step — twenty-five times this bound. What remains below
            // it is the integrator's own truncation error, which is a property
            // of velocity-Verlet and not of the ghost layer.
            assert!(
                drift < 1e-3,
                "step {step}: total energy drifted by {drift} (relative). \
                 A copy that jumped a cell makes pairs appear and vanish, and \
                 NVE has nothing to absorb that"
            );
        }

        // The run has to have exercised the thing it claims to test.
        // The drift carries the run 0.8 Å along +x, so the four atoms that
        // start a tenth of an Ångström inside the face cross and the four at
        // 3.9 do not. Four folds are what this exercises.
        let crossed: i64 = state.images.iter().map(|&m| m.abs()).sum();
        assert!(crossed >= 4, "atoms should have crossed; got {crossed}");
        let rebuilds = ig
            .forces()
            .neighbor_stats()
            .rebuilds
            .expect("a ghost provider counts its rebuilds");
        assert!(
            rebuilds < 200,
            "the halo must survive some folds, or the reconciliation is never reached"
        );
    }

    /// The virial reaches the integrator, and it is a property of the
    /// configuration rather than of where the cell's origin happens to fall.
    ///
    /// A rigid translation of a periodic system is a symmetry: the same atoms
    /// at the same separations, so the same physics. It is emphatically *not*
    /// the same bookkeeping — moving everything a third of a cell puts a
    /// different set of atoms near the faces, so a different set of copies is
    /// materialised and the pairs are found through different images. A virial
    /// that moved under it would be reporting the halo instead of the physics.
    ///
    /// The translation is deliberately not a lattice vector. Shifting by a
    /// whole cell and wrapping gives back bit-identical coordinates, which
    /// asserts nothing about the halo at all.
    #[test]
    fn the_integrator_reports_a_virial_that_a_rigid_translation_cannot_move() {
        let l = 12.0_f64;
        let cutoff = 5.0;
        let bx = SimBox::cube(l, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let base = array![
            [11.0_f64, 11.0, 11.0],
            [3.0, 11.0, 11.0],
            [11.0, 3.0, 11.0],
            [11.0, 11.0, 3.0],
            [3.0, 3.0, 11.0],
            [3.0, 11.0, 3.0],
            [11.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
        ];
        let n = base.nrows();
        let mass = scalar_mass(12.0, n).unwrap();

        let virial_after_one_step = |shift: F| {
            let mut pts = base.clone();
            pts.iter_mut().for_each(|x| *x += shift);
            let (wrapped, _m) = bx.wrap_shifts(pts.view());
            let comm = Comm::new(bx.clone(), wrapped.view(), cutoff, 0.0).unwrap();
            let mut ig = VelocityVerlet::new(
                1.0,
                GhostPairs::new(
                    LJCut::new(0.3, 3.4, cutoff, 12, 6, false, false).unwrap(),
                    comm,
                )
                .unwrap(),
                mass.view(),
                Some(bx.clone()),
            )
            .unwrap();
            let state = ig.initial(wrapped, FNx3::zeros((n, 3))).unwrap();
            state
                .virial
                .expect("the ghost provider tallies a virial, and the state keeps it")
        };

        let a = virial_after_one_step(0.0);
        let b = virial_after_one_step(l / 3.0);
        let scale = a.components.iter().fold(1.0_f64, |m, c| m.max(c.abs()));
        assert!(
            scale > 1.0,
            "the virial must be non-trivial for this test to mean anything"
        );
        for c in 0..6 {
            assert!(
                (a.components[c] - b.components[c]).abs() / scale < 1e-10,
                "component {c}: {} vs {} after a rigid translation",
                a.components[c],
                b.components[c]
            );
        }
    }

    /// The two régimes agree on the virial, having derived it two different
    /// ways.
    ///
    /// The ghost route sums `Σ_a f_a ⊗ x_a` over owned atoms *and copies*,
    /// each at its own position, before the copies' forces are folded back.
    /// The minimum-image route never has a copy: each pair kernel tallies
    /// `Σ f_ij ⊗ r_ij` inside the loop that made the forces, from the folded
    /// displacement. Those are the same tensor by an identity, not by
    /// construction — so agreeing is evidence, where a shared code path would
    /// have been none.
    #[test]
    fn the_two_regimes_derive_the_same_virial() {
        use molrs::ff::potential::pair::lj_cut::Mixing;

        let l = 12.0_f64;
        let cutoff = 5.0;
        let bx = SimBox::cube(l, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // Every edge of this cube runs through a face, so every pair is found
        // through a periodic image and the two derivations have to work for it.
        let pos = array![
            [11.0_f64, 11.0, 11.0],
            [3.0, 11.0, 11.0],
            [11.0, 3.0, 11.0],
            [11.0, 11.0, 3.0],
            [3.0, 3.0, 11.0],
            [3.0, 11.0, 3.0],
            [11.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
        ];
        let n = pos.nrows();
        let per_type = [(0.3_f64, 3.4_f64), (0.9, 2.6)];
        let type_id: Vec<u32> = (0..n).map(|i| (i % 2) as u32).collect();
        let lj = || {
            LJCut::typed(
                type_id.clone(),
                &per_type,
                Mixing::Arithmetic,
                cutoff,
                12,
                6,
                false,
                false,
            )
            .unwrap()
        };
        let no_fold = Array2::<i64>::zeros((n, 3));

        let skin = VerletSkin::new(
            NeighborList::new(cutoff),
            cutoff,
            NeighborPolicy {
                skin: 0.0,
                ..NeighborPolicy::default()
            },
            pos.view(),
            bx.clone(),
        )
        .unwrap();
        let mic = MicPairs::new(lj(), skin)
            .unwrap()
            .compute(pos.view(), no_fold.view())
            .unwrap()
            .virial
            .expect("a typed pair kernel tallies its virial");

        let comm = Comm::new(bx, pos.view(), cutoff, 0.0).unwrap();
        let ghost = GhostPairs::new(lj(), comm)
            .unwrap()
            .compute(pos.view(), no_fold.view())
            .unwrap()
            .virial
            .expect("the halo tallies one too");

        let scale = mic.components.iter().fold(1.0_f64, |m, c| m.max(c.abs()));
        assert!(
            scale > 1.0,
            "the virial must be non-trivial for this to assert anything"
        );
        for c in 0..6 {
            assert!(
                (mic.components[c] - ghost.components[c]).abs() / scale < 1e-12,
                "component {c}: {} through the image vs {} through copies",
                mic.components[c],
                ghost.components[c]
            );
        }
    }
}

#[cfg(test)]
mod wrapped_state_tests {
    use super::super::forces::Direct;
    use super::*;
    use molrs::ff::potential::Potentials;
    use molrs::spatial::simbox::SimBox;
    use ndarray::array;

    /// Validation A of the wrapped+image design: one atom driven straight
    /// through the boundary many times over.
    ///
    /// Three things have to hold together, and only together do they mean the
    /// state is canonical: the stored coordinate stays in the cell however long
    /// the run goes, the flag counts the crossings, and the reconstruction
    /// `pos + H·images` is the continuous line the atom actually travelled.
    /// Any two of the three can be satisfied by an implementation that is
    /// quietly wrong about the third.
    #[test]
    fn an_atom_driven_through_the_boundary_stays_wrapped_and_stays_continuous() {
        let l = 10.0_f64;
        let bx = SimBox::cube(l, array![0.0, 0.0, 0.0], [true, true, true]).unwrap();
        // No forces: the trajectory is exactly known, so any drift is the
        // bookkeeping's fault and not the dynamics'.
        let dt = 1.0;
        let vx = 0.25; // Å/fs -> 0.25 Å per step
        let mut ig = VelocityVerlet::new(
            dt,
            Direct::new(Potentials::new()),
            scalar_mass(1.0, 1).unwrap().view(),
            Some(bx.clone()),
        )
        .unwrap();

        let mut state = ig
            .initial(array![[0.5, 5.0, 5.0]], array![[vx, 0.0, 0.0]])
            .unwrap();
        assert_eq!(
            state.images.row(0).to_vec(),
            vec![0_i64; 3],
            "flags start clean"
        );

        let steps = 400; // 100 Å of travel = ten crossings
        for n in 1..=steps {
            state = ig.advance(state).unwrap();

            // (1) the stored coordinate never leaves the cell
            let f = bx.to_frac(state.pos.view());
            for d in 0..3 {
                assert!(
                    (0.0..1.0).contains(&f[[0, d]]),
                    "step {n}: fractional {} escaped [0, 1)",
                    f[[0, d]]
                );
            }

            // (2) the reconstruction is the line the atom actually travelled
            let travelled = 0.5 + vx * dt * n as F;
            let u = bx.unwrap(state.pos.view(), state.images.view());
            assert!(
                (u[[0, 0]] - travelled).abs() < 1e-9,
                "step {n}: unwrapped {} vs travelled {travelled}",
                u[[0, 0]]
            );
        }

        // (3) the flag counted the crossings rather than drifting on its own
        let expected = ((0.5 + vx * dt * steps as F) / l).floor() as i64;
        assert_eq!(
            state.images[[0, 0]],
            expected,
            "after {steps} steps the atom is {expected} cells downstream"
        );
        assert_eq!(state.images[[0, 1]], 0);
        assert_eq!(state.images[[0, 2]], 0);
    }

    /// A free-boundary run has no cell to fold into, so nothing is wrapped and
    /// the flags stay zero — the same integrator, with the periodic layer
    /// switched off rather than special-cased downstream.
    #[test]
    fn free_boundary_leaves_positions_and_flags_alone() {
        let mut ig = VelocityVerlet::new(
            1.0,
            Direct::new(Potentials::new()),
            scalar_mass(1.0, 1).unwrap().view(),
            None,
        )
        .unwrap();
        let mut state = ig
            .initial(array![[0.0, 0.0, 0.0]], array![[3.0, 0.0, 0.0]])
            .unwrap();
        for _ in 0..50 {
            state = ig.advance(state).unwrap();
        }
        assert!(state.pos[[0, 0]] > 100.0, "nothing folded it back");
        assert_eq!(state.images.row(0).to_vec(), vec![0_i64; 3]);
    }

    /// Wrapping must not change the physics: the same run, started from a
    /// configuration shifted by a whole cell, produces the same wrapped
    /// coordinates and the same energies. The flags differ by exactly that
    /// shift, which is what tells the two runs apart.
    #[test]
    fn a_lattice_shifted_start_is_the_same_trajectory() {
        let l = 10.0_f64;
        let bx = SimBox::cube(l, array![0.0, 0.0, 0.0], [true, true, true]).unwrap();
        let run = |start: FNx3| {
            let mut ig = VelocityVerlet::new(
                1.0,
                Direct::new(Potentials::new()),
                scalar_mass(1.0, 2).unwrap().view(),
                Some(bx.clone()),
            )
            .unwrap();
            let mut st = ig
                .initial(start, array![[0.3, 0.0, 0.0], [-0.2, 0.1, 0.0]])
                .unwrap();
            for _ in 0..120 {
                st = ig.advance(st).unwrap();
            }
            st
        };

        let a = run(array![[1.0, 5.0, 5.0], [6.0, 5.0, 5.0]]);
        // The same configuration, three cells over in x and one in y.
        let b = run(array![[31.0, 15.0, 5.0], [36.0, 15.0, 5.0]]);

        for i in 0..2 {
            for d in 0..3 {
                assert!(
                    (a.pos[[i, d]] - b.pos[[i, d]]).abs() < 1e-9,
                    "atom {i} axis {d}: wrapped positions must agree"
                );
            }
        }
        assert!((a.energy - b.energy).abs() < 1e-12);
    }
}
