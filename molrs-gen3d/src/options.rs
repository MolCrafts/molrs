//! Public options for 3D generation.

/// Stage-1 embedding algorithm selector.
///
/// Names are algorithm-based (not toolkit-based), so backends can evolve
/// without coupling public API names to a specific implementation source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedAlgorithm {
    /// Rule- and fragment-based coordinate construction.
    ///
    /// This is the algorithm family currently implemented in this crate.
    FragmentRules,
    /// Distance-geometry based embedding.
    DistanceGeometry,
}

/// Force-field backend selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForceFieldKind {
    /// Merck Molecular Force Field 94.
    MMFF94,
    /// Universal Force Field.
    Uff,
    /// Prefer MMFF94, fall back to UFF.
    Auto,
}

/// Preset quality/speed profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gen3DSpeed {
    /// Short minimization and fewer rotor trials.
    Fast,
    /// Balanced defaults.
    Medium,
    /// More rotor trials and longer minimization.
    Better,
}

/// Options for [`super::generate_3d`].
#[derive(Debug, Clone)]
pub struct Gen3DOptions {
    /// Stage-1 embedding algorithm.
    pub embed_algorithm: EmbedAlgorithm,
    /// Target force-field family (or auto selection).
    pub forcefield: ForceFieldKind,
    /// Throughput/quality preset.
    pub speed: Gen3DSpeed,
    /// Add explicit hydrogens before generation.
    pub add_hydrogens: bool,
    /// Total optimization budget. `0` means "use speed preset default".
    pub max_steps: usize,
    /// Optional deterministic RNG seed.
    pub rng_seed: Option<u64>,
}

impl Default for Gen3DOptions {
    fn default() -> Self {
        Self {
            embed_algorithm: EmbedAlgorithm::FragmentRules,
            forcefield: ForceFieldKind::Auto,
            speed: Gen3DSpeed::Medium,
            add_hydrogens: true,
            max_steps: 0,
            rng_seed: None,
        }
    }
}

impl Gen3DOptions {
    /// Effective optimization step budget.
    pub(crate) fn effective_max_steps(&self) -> usize {
        if self.max_steps > 0 {
            return self.max_steps;
        }
        match self.speed {
            Gen3DSpeed::Fast => 120,
            Gen3DSpeed::Medium => 260,
            Gen3DSpeed::Better => 520,
        }
    }

    /// Coarse minimization steps.
    pub(crate) fn coarse_steps(&self) -> usize {
        (self.effective_max_steps() * 35 / 100).max(20)
    }

    /// Final minimization steps.
    pub(crate) fn final_steps(&self) -> usize {
        (self.effective_max_steps() * 50 / 100).max(20)
    }

    /// Rotor search attempts.
    pub(crate) fn rotor_attempts(&self, n_rot_bonds: usize) -> usize {
        if n_rot_bonds == 0 {
            return 0;
        }
        match self.speed {
            Gen3DSpeed::Fast => (n_rot_bonds * 4).max(8),
            Gen3DSpeed::Medium => (n_rot_bonds * 8).max(20),
            Gen3DSpeed::Better => (n_rot_bonds * 16).max(40),
        }
    }

    /// Maximum per-step rotor perturbation (radians).
    pub(crate) fn rotor_max_delta(&self) -> f64 {
        match self.speed {
            Gen3DSpeed::Fast => std::f64::consts::PI / 5.0,
            Gen3DSpeed::Medium => std::f64::consts::PI / 3.0,
            Gen3DSpeed::Better => std::f64::consts::PI / 2.0,
        }
    }
}
