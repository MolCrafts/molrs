//! Kernel registry: maps `(category, style_name)` → [`KernelConstructor`] plus
//! the [`ParamSource`] that says **where that kernel's parameters come from**.
//!
//! `ForceField::to_potentials` resolves each style's kernel through this
//! registry instead of a hard-coded match, so a new potential is added by
//! *registering* its constructor rather than editing core dispatch. The
//! built-ins are seeded on first use; [`register_kernel`] adds or overrides
//! entries at runtime (the advertised extension point).
//!
//! # Why a registration carries a `ParamSource`
//!
//! Most kernels resolve their numbers from the style's per-type rows (`tp`):
//! `bond/harmonic` looks up `k` / `r0` by the bond's type label, `pair/lj/cut`
//! looks up `sigma` / `epsilon` by the atom's. Some cannot. MMFF's bond, angle,
//! stretch-bend, torsion and out-of-plane parameters depend on aromaticity, ring
//! size, four-level equivalence degradation, and — on a table miss — empirical
//! rules invented from covalent radii; the typifier resolves them **per instance**
//! and bakes them into Frame columns, and the kernels read those columns and
//! ignore `tp` entirely. The same is true of `pair/coul/cut` and `kspace/pme`,
//! whose charges are per-atom Frame data by construction.
//!
//! That is correct — but until [`ParamSource`] existed there was no way to *say*
//! it, so those styles registered as table-driven anyway and
//! [`Style::to_potential`](crate::ff::forcefield::Style::to_potential)'s
//! "has type definitions" guard had to be bribed with 4,065 rows of MMFF XML that
//! no code reads. Naming the distinction is what lets the guard ask the right
//! question, and `tests/ff/potential/param_source_gate.rs` holds the two halves
//! together: **a ctor ignores `tp` if and only if it is registered
//! [`ParamSource::PerInstance`]**.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use molrs::store::frame::Frame;

use super::{angle, bond, dihedral, improper, kspace, pair};

/// Builds a molecule-bound [`Potential`] from a style's params, its per-type
/// params (`(type_label, params)`), and a typed [`Frame`]. Every kernel
/// constructor in the crate matches this signature.
pub type KernelConstructor =
    fn(&Params, &[(&str, &Params)], &Frame) -> Result<Box<dyn Potential>, String>;

/// Where a kernel's parameters come from — the question the empty-type-params
/// guard must ask before it rejects a style with no type rows.
///
/// A kernel constructor that binds its type-params as `_tp` (i.e. resolves
/// nothing from them) **is not a table-driven style**, and must say so by being
/// registered [`PerInstance`](ParamSource::PerInstance).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamSource {
    /// Parameters come from the style's type-definition rows (the `tp` slice).
    /// A style with no rows resolves nothing, and is an error.
    TypeRows,
    /// Parameters are resolved per interaction by the typifier and baked into
    /// [`Frame`] columns; `tp` is ignored and may legitimately be empty.
    PerInstance,
}

/// Maps `(category, style_name)` to the constructor that builds its potential
/// and the [`ParamSource`] it resolves parameters from.
#[derive(Default)]
pub struct KernelRegistry {
    ctors: HashMap<(String, String), (KernelConstructor, ParamSource)>,
}

impl KernelRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or override) a table-driven ([`ParamSource::TypeRows`]) kernel
    /// for `(category, name)`.
    ///
    /// The thin wrapper over [`register_with`](Self::register_with): table-driven
    /// is what every kernel outside MMFF / `coul/cut` / `pme` is, so it stays the
    /// short form.
    pub fn register(&mut self, category: &str, name: &str, ctor: KernelConstructor) {
        self.register_with(category, name, ctor, ParamSource::TypeRows);
    }

    /// Register (or override) the kernel for `(category, name)`, declaring where
    /// it resolves its parameters from.
    pub fn register_with(
        &mut self,
        category: &str,
        name: &str,
        ctor: KernelConstructor,
        source: ParamSource,
    ) {
        self.ctors
            .insert((category.to_owned(), name.to_owned()), (ctor, source));
    }

    /// The constructor registered for `(category, name)`, if any.
    pub fn get(&self, category: &str, name: &str) -> Option<KernelConstructor> {
        self.ctors
            .get(&(category.to_owned(), name.to_owned()))
            .map(|(ctor, _)| *ctor)
    }

    /// The [`ParamSource`] declared for `(category, name)`, if it is registered.
    pub fn param_source(&self, category: &str, name: &str) -> Option<ParamSource> {
        self.ctors
            .get(&(category.to_owned(), name.to_owned()))
            .map(|(_, source)| *source)
    }

    /// Number of registered kernels.
    pub fn len(&self) -> usize {
        self.ctors.len()
    }

    /// Whether the registry has no kernels.
    pub fn is_empty(&self) -> bool {
        self.ctors.is_empty()
    }

    /// A registry seeded with every built-in kernel.
    pub fn builtin() -> Self {
        let mut r = Self::new();
        // bonded
        r.register("bond", "harmonic", bond::harmonic::bond_harmonic_ctor);
        r.register("bond", "class2", bond::class2::bond_class2_ctor);
        r.register("bond", "morse", bond::morse::bond_morse_ctor);
        r.register("angle", "harmonic", angle::harmonic::angle_harmonic_ctor);
        r.register("angle", "class2", angle::class2::angle_class2_ctor);
        r.register("dihedral", "opls", dihedral::opls::dihedral_opls_ctor);
        r.register("dihedral", "charmm", dihedral::charmm::dihedral_charmm_ctor);
        r.register(
            "dihedral",
            "multi/harmonic",
            dihedral::multi_harmonic::dihedral_multi_harmonic_ctor,
        );
        r.register(
            "dihedral",
            "periodic",
            dihedral::periodic::dihedral_periodic_ctor,
        );
        r.register(
            "dihedral",
            "fourier",
            dihedral::periodic::dihedral_periodic_ctor,
        );
        r.register("dihedral", "class2", dihedral::class2::dihedral_class2_ctor);
        // pair / nonbonded
        r.register("pair", "lj/cut", pair::lj_cut::pair_lj_cut_ctor);
        r.register("pair", "lj/class2", pair::lj_class2::pair_lj_class2_ctor);
        r.register("pair", "buck", pair::buck::pair_buck_ctor);
        r.register("pair", "morse", pair::morse::pair_morse_ctor);
        r.register("pair", "thole", pair::thole::pair_thole_ctor);
        r.register(
            "pair",
            "coul/tt",
            pair::tang_toennies::pair_tang_toennies_ctor,
        );
        // `coul/cut` reads per-atom `charge` off the Frame — there is no charge
        // type-row to read, and its ctor binds `_type_params`.
        r.register_with(
            "pair",
            "coul/cut",
            pair::coul_cut::pair_coul_cut_ctor,
            ParamSource::PerInstance,
        );
        // MMFF94 — six per-instance styles. Their kernels read the columns the
        // typifier bakes (`kb`/`r0`, `ka`/`theta0`, `kba_*`, `v1`/`v2`/`v3`,
        // `koop`, `charge`), never a type row: MMFF's context rules (aromaticity,
        // ring size, equivalence degradation, empirical fallbacks) are not a
        // `(type_i, type_j, …) → params` table and cannot be made into one.
        r.register_with(
            "bond",
            "mmff_bond",
            bond::mmff::mmff_bond_ctor,
            ParamSource::PerInstance,
        );
        r.register_with(
            "angle",
            "mmff_angle",
            angle::mmff::mmff_angle_ctor,
            ParamSource::PerInstance,
        );
        r.register_with(
            "angle",
            "mmff_stbn",
            angle::mmff::mmff_stbn_ctor,
            ParamSource::PerInstance,
        );
        r.register_with(
            "dihedral",
            "mmff_torsion",
            dihedral::mmff::mmff_torsion_ctor,
            ParamSource::PerInstance,
        );
        r.register_with(
            "improper",
            "mmff_oop",
            improper::mmff::mmff_oop_ctor,
            ParamSource::PerInstance,
        );
        r.register_with(
            "pair",
            "mmff_ele",
            pair::mmff::mmff_ele_ctor,
            ParamSource::PerInstance,
        );
        r.register(
            "improper",
            "harmonic",
            improper::harmonic::improper_harmonic_ctor,
        );
        r.register("improper", "cvff", improper::cvff::improper_cvff_ctor);
        r.register(
            "improper",
            "periodic",
            improper::periodic::improper_periodic_ctor,
        );
        // vdW is the one MMFF style that genuinely IS a per-atom-type table:
        // 95 types, 95 rows, and `mmff_vdw_ctor` opens by indexing `tp`.
        r.register("pair", "mmff_vdw", pair::mmff::mmff_vdw_ctor);
        // k-space: PME reads per-atom `charge` off the Frame, like `coul/cut`.
        r.register_with(
            "kspace",
            "pme",
            kspace::pme::pme_ctor,
            ParamSource::PerInstance,
        );
        r
    }
}

/// The process-wide kernel registry, initialized with the built-ins on first use.
fn global() -> &'static RwLock<KernelRegistry> {
    static REGISTRY: OnceLock<RwLock<KernelRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(KernelRegistry::builtin()))
}

/// Register (or override) a table-driven ([`ParamSource::TypeRows`]) kernel in
/// the global registry. The extension point for new potentials — no core
/// dispatch edit required.
pub fn register_kernel(category: &str, name: &str, ctor: KernelConstructor) {
    global().write().unwrap().register(category, name, ctor);
}

/// Register (or override) a kernel in the global registry, declaring its
/// [`ParamSource`].
///
/// Use this — with [`ParamSource::PerInstance`] — for a kernel whose parameters
/// are baked into [`Frame`] columns rather than resolved from type rows; it is
/// what exempts the style from the "has type definitions" check.
pub fn register_kernel_with(
    category: &str,
    name: &str,
    ctor: KernelConstructor,
    source: ParamSource,
) {
    global()
        .write()
        .unwrap()
        .register_with(category, name, ctor, source);
}

/// Look up a kernel constructor in the global registry.
pub fn lookup_kernel(category: &str, name: &str) -> Option<KernelConstructor> {
    global().read().unwrap().get(category, name)
}

/// Look up the [`ParamSource`] a style's kernel declared.
pub fn lookup_param_source(category: &str, name: &str) -> Option<ParamSource> {
    global().read().unwrap().param_source(category, name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_has_core_kernels() {
        let r = KernelRegistry::builtin();
        assert!(r.get("bond", "harmonic").is_some());
        assert!(r.get("pair", "lj/cut").is_some());
        assert!(r.get("pair", "buck").is_some());
        assert!(r.get("kspace", "pme").is_some());
        assert!(r.get("bond", "does-not-exist").is_none());
    }

    #[test]
    fn register_overrides_and_adds() {
        let mut r = KernelRegistry::new();
        assert!(r.is_empty());
        r.register("pair", "lj/cut", pair::lj_cut::pair_lj_cut_ctor);
        assert_eq!(r.len(), 1);
        assert!(r.get("pair", "lj/cut").is_some());
        // re-registering the same key overrides, not duplicates
        r.register("pair", "lj/cut", pair::buck::pair_buck_ctor);
        assert_eq!(r.len(), 1);
    }
}
