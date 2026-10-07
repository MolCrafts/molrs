use ndarray::Array1;
use std::f64::consts::PI;

/// Equally-spaced one-sided angular-frequency grid.
///
/// `ω_k = k · 2π / (n_fft · dt)` for `k = 0, 1, …, n_fft/2` —
/// length `n_fft/2 + 1`, spacing `Δω = 2π / (n_fft · dt)`, Nyquist
/// `π / dt`. Bin 0 is DC.
///
/// # Units
/// `dt` is the time-domain sample spacing; the output is in
/// `rad / [time]` where `[time]` matches the unit of `dt`. For LAMMPS
/// real units (`dt` in ps) the grid is in `rad · ps⁻¹`.
pub fn frequency_grid(n_fft: usize, dt: f64) -> Array1<f64> {
    let n_points = n_fft / 2 + 1;
    let t_total = n_fft as f64 * dt;
    let d_omega = 2.0 * PI / t_total;
    Array1::from_iter((0..n_points).map(|i| i as f64 * d_omega))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_frequency_grid_basic() {
        let grid = frequency_grid(8, 0.5);
        assert_eq!(grid.len(), 5); // 8/2 + 1
        assert!((grid[0] - 0.0).abs() < 1e-10);
        let expected_nyq = PI / 0.5; // 2π
        assert!((grid[4] - expected_nyq).abs() < 1e-10);
        let expected_spacing = 2.0 * PI / (8.0 * 0.5); // 2π/4 = π/2
        assert!((grid[1] - grid[0] - expected_spacing).abs() < 1e-10);
    }

    #[test]
    fn test_frequency_grid_minimal() {
        let grid = frequency_grid(2, 1.0);
        assert_eq!(grid.len(), 2); // 2/2 + 1
        assert!((grid[0] - 0.0).abs() < 1e-10);
        assert!((grid[1] - PI).abs() < 1e-10);
    }

    #[test]
    fn test_frequency_grid_uniform_spacing() {
        let grid = frequency_grid(1024, 0.001);
        assert_eq!(grid.len(), 513);
        let spacing = 2.0 * PI / (1024.0 * 0.001);
        for i in 1..grid.len() {
            let diff = grid[i] - grid[i - 1];
            assert!(
                (diff - spacing).abs() < 1e-10,
                "non-uniform spacing at index {i}"
            );
        }
    }

    #[test]
    fn test_frequency_grid_large_n() {
        let grid = frequency_grid(65536, 0.002);
        assert_eq!(grid.len(), 32769);
        assert!((grid[0] - 0.0).abs() < 1e-10);
        assert!((grid[32768] - PI / 0.002).abs() < 1e-10);
    }

    #[test]
    fn test_frequency_grid_dt_variation() {
        let grid = frequency_grid(4, 2.0);
        assert_eq!(grid.len(), 3);
        let expected_spacing = 2.0 * PI / (4.0 * 2.0); // π/4
        assert!((grid[1] - grid[0] - expected_spacing).abs() < 1e-10);
        let expected_nyq = PI / 2.0;
        assert!((grid[2] - expected_nyq).abs() < 1e-10);
    }
}
