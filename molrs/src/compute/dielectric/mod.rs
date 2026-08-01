//! Dielectric raw observables + static dielectric constant.
//!
//! Computes the **raw/defined** dielectric quantities of a polar fluid from
//! molecular-dynamics dipole trajectories: the instantaneous dipole moment,
//! current density, current partition, and the static dielectric constant
//! ε(0).
//!
//! # Routes
//!
//! - [`static_dielectric_constant`] / [`static_dielectric_constant_components`]
//!   — Neumann fluctuation formula (Neumann, *Mol. Phys.* **50**, 841 (1983);
//!   conducting/tin-foil Ewald boundary conditions assumed).
//!
//! The frequency-dependent permittivity ε*(ω) = ε′(ω) − i·ε″(ω) is **no longer
//! computed here**: the raw fluctuation dipole / current ACFs come from the
//! [`DebyeRelaxation`](crate::compute::DebyeRelaxation) /
//! [`GreenKuboConductivity`](crate::compute::GreenKuboConductivity) raw
//! computes, and the window + FFT + prefactor transform is the
//! [`EinsteinHelfandSpectrum`](crate::compute::spectroscopy::EinsteinHelfandSpectrum) /
//! [`GreenKuboSpectrum`](crate::compute::spectroscopy::GreenKuboSpectrum)
//! [`Fit`](crate::compute::traits::Fit) in `compute::spectroscopy` (windowing +
//! transforming a raw ACF into ε(ω) is a *fit*).
//!
//! # Units
//!
//! All inputs and outputs use LAMMPS *real* units throughout:
//!
//! | quantity        | unit                |
//! |-----------------|---------------------|
//! | length          | Å                   |
//! | charge          | e                   |
//! | energy          | kcal / mol          |
//! | time            | ps                  |
//! | temperature     | K                   |
//! | volume          | Å³                  |
//! | dipole moment   | e · Å               |
//! | current density | e · Å⁻² · ps⁻¹      |
//! | ε permittivity  | dimensionless       |

use ndarray::{Array1, Array2, Array3};

use crate::compute::error::ComputeError;

// ── Physical constants (MD real units: kcal, mol, Angstrom, e, K) ─────────
//
// MD-real and SI values are defined once in `molrs-core::units::constants`;
// the names below are the local spellings the kernels use.

use molrs::units::constants::COULOMB_REAL as KAPPA;

/// Boltzmann constant in kcal/(mol·K) — MD "real" units. Shared with the
/// spectral validation checks ([`crate::compute::spectroscopy`]) so there is one value.
pub use molrs::units::constants::BOLTZMANN_REAL as K_B;

const FOUR_PI_OVER_3: f64 = 4.1887902047863905; // 4π/3

// ── Basic observables ─────────────────────────────────────────────────────

/// Total instantaneous dipole moment **M** = Σᵢ qᵢ · **rᵢ**.
///
/// # Arguments
/// * `charges` — partial charges (`n_atoms`), units **e**.
/// * `positions` — atomic coordinates `(n_atoms, 3)`, units **Å**.
///   Coordinates must already be unwrapped across periodic images;
///   otherwise the dipole jumps by `q · L` whenever an atom crosses a
///   box face. (The caller does the unwrapping.)
///
/// # Returns
/// Length-3 vector **M** = (Mₓ, M_y, M_z) in **e · Å**.
///
/// # Errors
/// * `DimensionMismatch` if `positions.shape() != (n_atoms, 3)`.
/// * `NonFinite` if any charge is NaN/inf.
pub fn compute_dipole_moment(
    charges: &Array1<f64>,
    positions: &Array2<f64>,
) -> Result<Array1<f64>, ComputeError> {
    let n = charges.len();
    if positions.shape() != [n, 3] {
        return Err(ComputeError::DimensionMismatch {
            expected: n * 3,
            got: positions.len(),
            what: "positions (expected (n_atoms, 3))",
        });
    }
    let mut m = Array1::zeros(3);
    for i in 0..n {
        let q = charges[i];
        if !q.is_finite() {
            return Err(ComputeError::NonFinite {
                where_: "charges",
                index: i,
            });
        }
        m[0] += q * positions[[i, 0]];
        m[1] += q * positions[[i, 1]];
        m[2] += q * positions[[i, 2]];
    }
    Ok(m)
}

/// System current density **J**(t) = (**M**(t) − **M**(t − Δt)) / (V · Δt).
///
/// # Arguments
/// * `dipole_moments` — total dipole trajectory `(n_frames, 3)`, **e · Å**.
/// * `dt` — timestep between consecutive frames, **ps**.
/// * `volume` — system volume, **Å³** (constant; assumes NVT/NVE).
///
/// # Returns
/// `(n_frames, 3)` array in **e · Å⁻² · ps⁻¹**. **Row 0 is filled with
/// `f64::NAN`** because no previous frame exists to form the finite
/// difference. All downstream consumers must skip row 0 before forming the
/// current ACF (the Green–Kubo ε(ω) caller slices it off; bare `np.mean`
/// would poison its output).
///
/// # Errors
/// * `DimensionMismatch` if shape is not `(_, 3)`.
/// * `OutOfRange` if `dt ≤ 0` or `volume ≤ 0`.
pub fn compute_current_density(
    dipole_moments: &Array2<f64>,
    dt: f64,
    volume: f64,
) -> Result<Array2<f64>, ComputeError> {
    let shape = dipole_moments.shape();
    if shape[1] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: shape[1],
            what: "dipole_moments (expected (n_frames, 3))",
        });
    }
    if dt <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "dt",
            value: dt.to_string(),
        });
    }
    if volume <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "volume",
            value: volume.to_string(),
        });
    }
    let n_frames = shape[0];
    let mut j = Array2::from_elem((n_frames, 3), f64::NAN);
    if n_frames < 2 {
        return Ok(j);
    }
    let scale = 1.0 / (volume * dt);
    for t in 1..n_frames {
        for d in 0..3 {
            j[[t, d]] = (dipole_moments[[t, d]] - dipole_moments[[t - 1, d]]) * scale;
        }
    }
    Ok(j)
}

// ── Static dielectric constant ─────────────────────────────────────────────

/// Static dielectric constant ε(0) from the Neumann fluctuation formula.
///
/// `ε(0) = ε_∞ + (4π/3) · (1/(V·k_B·T)) · ⟨|**M** − ⟨**M**⟩|²⟩`
///
/// in Gaussian/MD units (in LAMMPS *real* units the prefactor includes
/// the Coulomb constant `KAPPA = 332.0637 kcal·Å·mol⁻¹·e⁻²` so that
/// the formula evaluates to a dimensionless number).
///
/// **Assumes conducting / tin-foil Ewald boundary conditions** (the
/// only case for which this prefactor is exact). For reaction-field
/// boundary conditions an additional `2(ε_RF − 1)/(2ε_RF + 1)` factor
/// is required (Neumann 1983 §IV).
///
/// # Arguments
/// * `dipole_moments` — `(n_frames, 3)`, **e · Å**, `n_frames ≥ 2`.
/// * `volume` — **Å³**.
/// * `temperature` — **K**.
/// * `epsilon_inf` — high-frequency / electronic permittivity ε_∞,
///   dimensionless (typically 1.0 for non-polarizable force fields,
///   1.5–2.5 for water with polarizable models).
///
/// # Reference
/// Neumann, M. *Mol. Phys.* **50** (4), 841 (1983),
/// "Dipole moment fluctuation formulas in computer simulations of
/// polar systems."
pub fn static_dielectric_constant(
    dipole_moments: &Array2<f64>,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
) -> Result<f64, ComputeError> {
    let shape = dipole_moments.shape();
    if shape[1] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: shape[1],
            what: "dipole_moments (expected (n_frames, 3))",
        });
    }
    if shape[0] < 2 {
        return Err(ComputeError::EmptyInput);
    }
    if volume <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "volume",
            value: volume.to_string(),
        });
    }
    if temperature <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "temperature",
            value: temperature.to_string(),
        });
    }

    // Two-pass centered variance: stable when |M| ≫ |δM| (avoids the
    // catastrophic cancellation of ⟨M²⟩ − ⟨M⟩² that bites at ~10⁶ frames).
    let n = shape[0] as f64;
    let mut mean_m = Array1::<f64>::zeros(3);
    for t in 0..shape[0] {
        for d in 0..3 {
            mean_m[d] += dipole_moments[[t, d]];
        }
    }
    for d in 0..3 {
        mean_m[d] /= n;
    }

    let mut variance = 0.0;
    for t in 0..shape[0] {
        for d in 0..3 {
            let dev = dipole_moments[[t, d]] - mean_m[d];
            variance += dev * dev;
        }
    }
    variance /= n;
    let prefactor = FOUR_PI_OVER_3 * KAPPA / (volume * K_B * temperature);

    Ok(epsilon_inf + prefactor * variance)
}

/// Per-axis static dielectric constant (MDAnalysis-compatible output).
///
/// Returns directional ε_x, ε_y, ε_z and their isotropic mean, plus the
/// per-axis dipole statistics. Uses the same Neumann (1983) fluctuation
/// formula as [`static_dielectric_constant`] but preserves the Cartesian
/// decomposition.
///
/// # Arguments
/// Same as [`static_dielectric_constant`].
///
/// # Reference
/// MDAnalysis `DielectricConstant`:
/// <https://docs.mdanalysis.org/stable/documentation_pages/analysis/dielectric.html>
pub fn static_dielectric_constant_components(
    dipole_moments: &Array2<f64>,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
) -> Result<StaticDielectricResult, ComputeError> {
    let shape = dipole_moments.shape();
    if shape[1] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: shape[1],
            what: "dipole_moments (expected (n_frames, 3))",
        });
    }
    if shape[0] < 2 {
        return Err(ComputeError::EmptyInput);
    }
    if volume <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "volume",
            value: volume.to_string(),
        });
    }
    if temperature <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "temperature",
            value: temperature.to_string(),
        });
    }

    let n = shape[0] as f64;
    let n_frames = shape[0];

    let mut mean_m = Array1::<f64>::zeros(3);
    let mut mean_sq = Array1::<f64>::zeros(3);

    for t in 0..n_frames {
        for d in 0..3 {
            let m = dipole_moments[[t, d]];
            mean_m[d] += m;
            mean_sq[d] += m * m;
        }
    }
    for d in 0..3 {
        mean_m[d] /= n;
        mean_sq[d] /= n;
    }

    let mut fluctuation = Array1::<f64>::zeros(3);
    let mut eps = Array1::<f64>::zeros(3);
    // Per-axis prefactor: 4π·KAPPA / (V·k_B·T).  This is 3× the
    // isotropic prefactor because the diagonal dielectric-tensor
    // component integrates the full dipole in one direction, while the
    // isotropic ε averages over 3 directions.
    let per_axis_prefactor = 3.0 * FOUR_PI_OVER_3 * KAPPA / (volume * K_B * temperature);

    for d in 0..3 {
        fluctuation[d] = mean_sq[d] - mean_m[d] * mean_m[d];
        eps[d] = epsilon_inf + per_axis_prefactor * fluctuation[d];
    }

    let eps_mean = (eps[0] + eps[1] + eps[2]) / 3.0;

    Ok(StaticDielectricResult {
        dipole_mean: mean_m,
        dipole_sq_mean: mean_sq,
        fluctuation,
        eps,
        eps_mean,
        epsilon_inf,
        n_frames,
    })
}

// ── System decomposition ───────────────────────────────────────────────────

/// Partition per-particle current into two disjoint groups by mask.
///
/// The "water/ion" naming is suggestive only — the function is a pure
/// boolean partition: particles with `mask[p] == true` are summed
/// into the first output, the rest into the second. The total
/// `J_a + J_b` equals the system current to floating-point precision.
///
/// # Arguments
/// * `per_particle_current` — `(n_particles, n_frames, 3)`,
///   **e · Å⁻² · ps⁻¹**.
/// * `water_mask` — boolean array of length `n_particles`; `true`
///   → first bucket.
///
/// # Returns
/// `(J_a, J_b)`, each `(n_frames, 3)`, same units as the input.
pub fn decompose_current(
    per_particle_current: &Array3<f64>,
    water_mask: &Array1<bool>,
) -> Result<(Array2<f64>, Array2<f64>), ComputeError> {
    let shape = per_particle_current.shape();
    let n_particles = shape[0];
    let n_frames = shape[1];
    if shape[2] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: shape[2],
            what: "per_particle_current (expected (n_particles, n_frames, 3))",
        });
    }
    if water_mask.len() != n_particles {
        return Err(ComputeError::DimensionMismatch {
            expected: n_particles,
            got: water_mask.len(),
            what: "water_mask",
        });
    }

    let mut j_water = Array2::zeros((n_frames, 3));
    let mut j_ion = Array2::zeros((n_frames, 3));

    for p in 0..n_particles {
        let target = if water_mask[p] {
            &mut j_water
        } else {
            &mut j_ion
        };
        for t in 0..n_frames {
            for d in 0..3 {
                target[[t, d]] += per_particle_current[[p, t, d]];
            }
        }
    }

    Ok((j_water, j_ion))
}

// The Einstein–Helfand ionic conductivity is now the explicit composition of
// the raw [`EinsteinConductivity`](crate::compute::EinsteinConductivity) collective-dipole
// MSD compute with the [`LinearFit`](crate::compute::fitting::LinearFit) slope and a
// caller-applied `slope / (6·V·k_B·T)` MD→SI prefactor. The legacy bundled
// `ConductivityResult` + `einstein_helfand_conductivity` free function (which
// baked the OLS slope and σ into the raw result) were removed in
// compute-fit-03-cleanup.

/// Per-axis static dielectric constant result (MDAnalysis-compatible).
///
/// Follows the MDAnalysis `DielectricConstant.results` layout:
/// per-axis dipole mean `M`, squared mean `M2`, fluctuation, and
/// directional ε.
#[derive(Debug, Clone)]
pub struct StaticDielectricResult {
    /// Per-axis mean dipole moment ⟨**M**⟩, **e · Å**, length 3.
    pub dipole_mean: Array1<f64>,
    /// Per-axis mean squared dipole ⟨**M**²⟩, **(e · Å)²**, length 3.
    pub dipole_sq_mean: Array1<f64>,
    /// Per-axis fluctuation ⟨M_d²⟩ − ⟨M_d⟩², **(e · Å)²**, length 3.
    pub fluctuation: Array1<f64>,
    /// Per-axis static dielectric constant ε_d (d = x, y, z), dimensionless.
    pub eps: Array1<f64>,
    /// Isotropic average (ε_x + ε_y + ε_z) / 3, dimensionless.
    pub eps_mean: f64,
    /// High-frequency / electronic permittivity used (ε_∞).
    pub epsilon_inf: f64,
    /// Number of frames analysed.
    pub n_frames: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Axis, arr1};

    // The conductivity MSD-exactness and Nernst–Einstein scientific-regression
    // tests moved to `compute::transport` alongside the
    // `EinsteinConductivity` + `LinearFit` composition that replaced the removed
    // `einstein_helfand_conductivity` free function.

    #[test]
    fn test_dipole_moment_two_charges() {
        let charges = arr1(&[1.0, -1.0]);
        let positions = ndarray::arr2(&[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let m = compute_dipole_moment(&charges, &positions).unwrap();
        assert!((m[0] - 2.0).abs() < 1e-10);
        assert!((m[1] - 0.0).abs() < 1e-10);
        assert!((m[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_dipole_moment_zero_charge() {
        let charges = arr1(&[0.0, 0.0, 0.0]);
        let positions = ndarray::Array2::zeros((3, 3));
        let m = compute_dipole_moment(&charges, &positions).unwrap();
        assert!((m[0].abs() + m[1].abs() + m[2].abs()) < 1e-10);
    }

    #[test]
    fn test_dipole_moment_wrong_shape() {
        let charges = arr1(&[1.0, 2.0]);
        let positions = ndarray::Array2::zeros((3, 3));
        assert!(compute_dipole_moment(&charges, &positions).is_err());
    }

    #[test]
    fn test_current_density_constant_dipole() {
        let dm = ndarray::Array2::from_elem((3, 3), 1.0);
        let j = compute_current_density(&dm, 1.0, 1.0).unwrap();
        assert_eq!(j.shape(), &[3, 3]);
        assert!(j[[0, 0]].is_nan());
        assert!((j[[1, 0]]).abs() < 1e-10);
        assert!((j[[2, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_current_density_linear() {
        let dm = ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let j = compute_current_density(&dm, 1.0, 1.0).unwrap();
        assert!(j[[0, 0]].is_nan());
        assert!((j[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((j[[2, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_current_density_dt_scaling() {
        let dm = ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let j1 = compute_current_density(&dm, 1.0, 1.0).unwrap();
        let j2 = compute_current_density(&dm, 2.0, 1.0).unwrap();
        assert!((j2[[1, 0]] * 2.0 - j1[[1, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_static_dielectric_zero_fluctuation() {
        let dm = ndarray::Array2::from_elem((10, 3), 0.0);
        let eps = static_dielectric_constant(&dm, 1000.0, 300.0, 1.0).unwrap();
        assert!((eps - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_static_dielectric_known_fluctuation() {
        let dm = ndarray::arr2(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);
        let eps = static_dielectric_constant(&dm, 1000.0, 300.0, 1.0).unwrap();
        // ⟨M⟩ = 0, ⟨M²⟩ = (1²+(-1)²)/2 = 1.0
        // ε(0) = 1.0 + (4π/3)*332.0637*1.0/(1000*1.9872e-3*300)
        let expected = 1.0 + FOUR_PI_OVER_3 * KAPPA * 1.0 / (1000.0 * K_B * 300.0);
        assert!((eps - expected).abs() < 1e-10);
    }

    #[test]
    fn test_static_dielectric_single_frame_rejected() {
        let dm = ndarray::Array2::zeros((1, 3));
        assert!(static_dielectric_constant(&dm, 1000.0, 300.0, 1.0).is_err());
    }

    #[test]
    fn test_decompose_current_conservation() {
        let n_particles = 4;
        let n_frames = 5;
        let mut current = Array3::zeros((n_particles, n_frames, 3));
        for p in 0..n_particles {
            for t in 0..n_frames {
                current[[p, t, 0]] = p as f64 + t as f64;
                current[[p, t, 1]] = (p as f64) * 2.0;
                current[[p, t, 2]] = t as f64 * 0.5;
            }
        }
        let mask = arr1(&[true, true, false, false]);
        let (j_w, j_i) = decompose_current(&current, &mask).unwrap();
        assert_eq!(j_w.shape(), &[n_frames, 3]);
        assert_eq!(j_i.shape(), &[n_frames, 3]);
        // Total should be sum of all particles
        let total: Array2<f64> = current.sum_axis(Axis(0));
        for t in 0..n_frames {
            for d in 0..3 {
                assert!(((j_w[[t, d]] + j_i[[t, d]]) - total[[t, d]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_decompose_current_mask_mismatch() {
        let current = Array3::zeros((2, 3, 3));
        let mask = arr1(&[true, false, true]);
        assert!(decompose_current(&current, &mask).is_err());
    }

    #[test]
    fn test_immutability_dipole_moment() {
        let charges = arr1(&[1.0, -1.0]);
        let positions = ndarray::arr2(&[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let pos_copy = positions.clone();
        compute_dipole_moment(&charges, &positions).unwrap();
        assert_eq!(positions, pos_copy);
    }

    #[test]
    fn test_immutability_current_density() {
        let dm = ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let dm_copy = dm.clone();
        compute_current_density(&dm, 1.0, 1.0).unwrap();
        assert_eq!(dm, dm_copy);
    }

    #[test]
    fn test_static_dielectric_components() {
        let dm = ndarray::arr2(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);
        let result = static_dielectric_constant_components(&dm, 1000.0, 300.0, 1.0).unwrap();
        assert_eq!(result.dipole_mean.len(), 3);
        assert_eq!(result.eps.len(), 3);
        assert_eq!(result.n_frames, 2);
        // M fluctuates in x only → ε_x > ε_y, ε_z
        assert!(result.eps[0] > result.eps[1]);
        assert!((result.eps[1] - 1.0).abs() < 1e-10);
        assert!((result.eps[2] - 1.0).abs() < 1e-10);
        // Isotropic mean matches scalar version.
        let scalar = static_dielectric_constant(&dm, 1000.0, 300.0, 1.0).unwrap();
        assert!((result.eps_mean - scalar).abs() < 1e-10);
    }

    #[test]
    fn test_static_dielectric_components_isotropic() {
        // Isotropic dipole (equal in all directions) → ε_x = ε_y = ε_z = ε_mean.
        let dm = ndarray::arr2(&[[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]);
        let result = static_dielectric_constant_components(&dm, 1000.0, 300.0, 1.0).unwrap();
        let eps_x = result.eps[0];
        assert!((result.eps[1] - eps_x).abs() < 1e-12);
        assert!((result.eps[2] - eps_x).abs() < 1e-12);
        assert!((result.eps_mean - eps_x).abs() < 1e-12);
    }
}
