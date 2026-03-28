//! Potential energy evaluation traits and kernel registry.
//!
//! A [`Potential`] stores pre-resolved topology indices and parameters.
//! Callers pass only flat coordinates — no [`Frame`] in the hot loop.
//! Construction from a [`Frame`] happens once via [`ForceField::compile`].

pub mod geometry;

pub mod angle;
pub mod bond;
pub mod dihedral;
pub mod improper;
pub mod kspace;
pub mod pair;

/// Backward-compatible re-exports for existing consumers.
pub mod kernels {
    pub use super::angle::harmonic::{AngleHarmonic, angle_harmonic_ctor};
    pub use super::bond::harmonic::{BondHarmonic, bond_harmonic_ctor};
    pub use super::pair::lj_cut::{PairLJCut as PairLJ126, pair_lj_cut_ctor as pair_lj126_ctor};
}

use std::collections::HashMap;

use crate::forcefield::{ForceField, Params};
use crate::frame::Frame;
use crate::types::F;

// ---------------------------------------------------------------------------
// Potential trait
// ---------------------------------------------------------------------------

/// Interface for computing potential energy and forces.
///
/// Implementations store pre-resolved atom indices and parameters.
/// The `eval()` method computes both energy and forces in a single pass,
/// avoiding redundant geometry computation.
pub trait Potential: Send + Sync {
    /// Compute energy and forces (= -gradient) in one pass.
    /// Returns `(energy, forces)` where forces has length `coords.len()`.
    fn eval(&self, coords: &[F]) -> (F, Vec<F>);

    /// Compute total potential energy.
    fn energy(&self, coords: &[F]) -> F {
        self.eval(coords).0
    }

    /// Compute forces (= -gradient) and return a length-3N vector.
    fn forces(&self, coords: &[F]) -> Vec<F> {
        self.eval(coords).1
    }
}

// ---------------------------------------------------------------------------
// Potentials collection
// ---------------------------------------------------------------------------

/// Aggregates multiple potentials; energy/forces are summed.
pub struct Potentials {
    inner: Vec<Box<dyn Potential>>,
}

impl std::fmt::Debug for Potentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Potentials")
            .field("len", &self.inner.len())
            .finish()
    }
}

impl Potentials {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn push(&mut self, pot: Box<dyn Potential>) {
        self.inner.push(pot);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Compute total energy and forces in one pass over all potentials.
    pub fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
        let n = coords.len();
        let mut total_e: F = 0.0;
        let mut total_f = vec![0.0; n];

        for p in &self.inner {
            let (e, f) = p.eval(coords);
            total_e += e;
            for (t, fi) in total_f.iter_mut().zip(f.iter()) {
                *t += fi;
            }
        }

        (total_e, total_f)
    }

    pub fn energy(&self, coords: &[F]) -> F {
        self.eval(coords).0
    }

    pub fn forces(&self, coords: &[F]) -> Vec<F> {
        self.eval(coords).1
    }
}

impl Default for Potentials {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Kernel registry
// ---------------------------------------------------------------------------

/// Constructor signature: takes style-level params, per-type params, and a
/// Frame (for topology resolution). Returns a boxed Potential with pre-resolved
/// indices — the Frame is NOT retained.
pub type KernelConstructor = fn(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String>;

/// Registry mapping `(category, style_name)` -> [`KernelConstructor`].
pub struct KernelRegistry {
    map: HashMap<(String, String), KernelConstructor>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        let mut reg = Self {
            map: HashMap::new(),
        };
        reg.register_builtins();
        reg
    }

    pub fn register(&mut self, category: &str, name: &str, ctor: KernelConstructor) {
        self.map
            .insert((category.to_owned(), name.to_owned()), ctor);
    }

    /// Look up a constructor by category and name.
    pub fn get(&self, category: &str, name: &str) -> Option<&KernelConstructor> {
        self.map.get(&(category.to_owned(), name.to_owned()))
    }

    fn register_builtins(&mut self) {
        // Generic kernels
        self.register("bond", "harmonic", bond::harmonic::bond_harmonic_ctor);
        self.register("angle", "harmonic", angle::harmonic::angle_harmonic_ctor);
        self.register("pair", "lj/cut", pair::lj_cut::pair_lj_cut_ctor);

        // MMFF94 kernels
        self.register("bond", "mmff_bond", bond::mmff::mmff_bond_ctor);
        self.register("angle", "mmff_angle", angle::mmff::mmff_angle_ctor);
        self.register("angle", "mmff_stbn", angle::mmff::mmff_stbn_ctor);
        self.register(
            "dihedral",
            "mmff_torsion",
            dihedral::mmff::mmff_torsion_ctor,
        );
        self.register("improper", "mmff_oop", improper::mmff::mmff_oop_ctor);
        self.register("pair", "mmff_vdw", pair::mmff::mmff_vdw_ctor);
        self.register("pair", "mmff_ele", pair::mmff::mmff_ele_ctor);

        // K-space kernels
        self.register("kspace", "pme", kspace::pme::pme_ctor);
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Frame helpers
// ---------------------------------------------------------------------------

/// Extract flat coordinate vector from Frame's `"atoms"` block.
///
/// Reads `"x"`, `"y"`, `"z"` float columns.
/// Returns `[x0,y0,z0, x1,y1,z1, ...]` as `Vec<F>`.
pub fn extract_coords(frame: &Frame) -> Result<Vec<F>, String> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| "Frame has no \"atoms\" block".to_string())?;

    let (x, y, z) = (
        atoms.get_float("x"),
        atoms.get_float("y"),
        atoms.get_float("z"),
    );
    let (Some(x), Some(y), Some(z)) = (x, y, z) else {
        return Err("atoms block missing x/y/z float columns".into());
    };

    let xs: Vec<F> = x.iter().copied().collect();
    let ys: Vec<F> = y.iter().copied().collect();
    let zs: Vec<F> = z.iter().copied().collect();

    let n = xs.len();
    if ys.len() != n || zs.len() != n {
        return Err("atoms x/y/z columns have mismatched lengths".into());
    }

    let mut coords = Vec::with_capacity(n * 3);
    for i in 0..n {
        coords.push(xs[i]);
        coords.push(ys[i]);
        coords.push(zs[i]);
    }
    Ok(coords)
}

impl ForceField {
    /// Compile all styles into [`Potential`] objects using the given registry
    /// and a [`Frame`] for topology resolution.
    ///
    /// The Frame is read once to resolve type labels into flat index arrays.
    /// The resulting Potentials do NOT retain the Frame.
    pub fn compile_with(
        &self,
        registry: &KernelRegistry,
        frame: &Frame,
    ) -> Result<Potentials, String> {
        let mut pots = Potentials::new();

        for style in self.styles() {
            let category = style.category();
            let ctor = registry.get(category, &style.name).ok_or_else(|| {
                format!(
                    "No kernel registered for style category '{}' with name '{}'",
                    category, style.name
                )
            })?;

            let type_params = style.defs.collect_type_params();
            if type_params.is_empty() {
                return Err(format!(
                    "Style '{}' ({}) has no type definitions",
                    style.name, category
                ));
            }

            let type_refs: Vec<(&str, &Params)> = type_params
                .iter()
                .map(|(name, params)| (name.as_str(), params))
                .collect();
            let pot = ctor(&style.params, &type_refs, frame)?;
            pots.push(pot);
        }

        Ok(pots)
    }

    /// Compile using the built-in kernel registry.
    pub fn compile(&self, frame: &Frame) -> Result<Potentials, String> {
        self.compile_with(&KernelRegistry::default(), frame)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use crate::types::U;
    use ndarray::Array1;

    struct DummyPotential {
        value: F,
    }

    impl Potential for DummyPotential {
        fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
            (self.value, vec![self.value; coords.len()])
        }
    }

    fn make_atoms_only_frame() -> Frame {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![0.0 as F, 2.0 as F]).into_dyn())
            .unwrap();
        atoms
            .insert("y", Array1::from_vec(vec![0.0 as F, 0.0 as F]).into_dyn())
            .unwrap();
        atoms
            .insert("z", Array1::from_vec(vec![0.0 as F, 0.0 as F]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);
        frame
    }

    fn make_lj_frame() -> Frame {
        let mut frame = make_atoms_only_frame();

        let mut pairs = Block::new();
        pairs
            .insert("atomi", Array1::from_vec(vec![0 as U]).into_dyn())
            .unwrap();
        pairs
            .insert("atomj", Array1::from_vec(vec![1 as U]).into_dyn())
            .unwrap();
        pairs
            .insert("type", Array1::from_vec(vec!["A".to_string()]).into_dyn())
            .unwrap();
        frame.insert("pairs", pairs);

        frame
    }

    #[test]
    fn test_potentials_collection() {
        let mut pots = Potentials::new();
        pots.push(Box::new(DummyPotential { value: 1.0 }));
        pots.push(Box::new(DummyPotential { value: 2.0 }));

        assert_eq!(pots.len(), 2);

        let coords: Vec<F> = vec![0.0; 6];
        assert!((pots.energy(&coords) - 3.0).abs() < 1e-5);

        let forces = pots.forces(&coords);
        for f in &forces {
            assert!((*f - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_registry_builtins() {
        let reg = KernelRegistry::default();
        assert!(reg.get("bond", "harmonic").is_some());
        assert!(reg.get("angle", "harmonic").is_some());
        assert!(reg.get("pair", "lj/cut").is_some());
        assert!(reg.get("kspace", "pme").is_some());
        assert!(reg.get("pair", "nonexistent").is_none());
    }

    #[test]
    fn test_compile_is_strict_for_unsupported_style() {
        let ff = ForceField::new("test").with_atomstyle("full");
        let frame = make_atoms_only_frame();
        let err = ff.compile(&frame).expect_err("expected compile to fail");
        assert!(err.contains("No kernel registered"));
    }

    #[test]
    fn test_compile_requires_types() {
        let mut ff = ForceField::new("test");
        ff.def_bondstyle("harmonic");
        let frame = make_atoms_only_frame();
        let err = ff.compile(&frame).expect_err("expected compile to fail");
        assert!(err.contains("has no type definitions"));
    }

    #[test]
    fn test_compile_energy() {
        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

        let frame = make_lj_frame();
        let pots = ff.compile(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();

        let (energy, _) = pots.eval(&coords);
        let expected: F = 4.0 * (1.0 / 4096.0 - 1.0 / 64.0);
        assert!((energy - expected).abs() < 1e-5);
    }

    #[test]
    fn test_compile_forces() {
        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

        let frame = make_lj_frame();
        let pots = ff.compile(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();

        let (_, forces) = pots.eval(&coords);

        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-5);
        }
    }

    #[test]
    fn test_compile_empty_ff() {
        let ff = ForceField::new("test");
        let frame = make_lj_frame();
        let pots = ff.compile(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();

        let (energy, forces) = pots.eval(&coords);
        assert!(energy.abs() < 1e-5);
        assert_eq!(forces.len(), 6);
        assert!(forces.iter().all(|x| x.abs() < 1e-5));
    }

    #[test]
    fn test_compile_missing_topology_is_error() {
        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

        let frame = make_atoms_only_frame();
        let err = ff.compile(&frame).unwrap_err();
        assert!(err.contains("pairs"));
    }
}
