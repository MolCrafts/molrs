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
//! ignore `tp` entirely. The same is true of `pair/coul/cut` and `pair/coul/long/pme`,
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
use crate::ff::potential::Member;
use molrs::store::frame::Frame;

use super::{angle, bond, dihedral, improper, kspace, pair};

/// Builds a molecule-bound [`Member`] from a style's params, its per-type
/// params (`(type_label, params)`), and a typed [`Frame`]. Every kernel
/// constructor in the crate matches this signature.
pub type KernelConstructor = fn(&Params, &[(&str, &Params)], &Frame) -> Result<Member, String>;

/// Where a kernel's parameters come from — the question the empty-type-params
/// guard must ask before it rejects a style with no type rows.
///
/// A kernel constructor that binds its type-params as `_tp` (i.e. resolves
/// nothing from them) **is not a table-driven style**, and must say so by being
/// registered [`PerInstance`](ParamSource::PerInstance).
/// Which `Frame` block decides whether a style has any rows to act on.
///
/// `Style::to_potential` skips a style whose topology is absent — a bond style
/// with no bonds contributes nothing, and letting the kernel fault on the
/// missing block instead would be a worse way to say so. Which block that is,
/// is a property of the **kernel**, not of its category.
///
/// PME is the case that proves it: registered under `pair` because that is
/// where an electrostatic style belongs, it reads per-atom charges and
/// `exclusions` and never looks at `pairs`. Gated on `pairs`, it was skipped
/// outright for any system whose caller had not built a pair list — deleting
/// the entire long-range electrostatics, silently, to exactly zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RowSource {
    /// The category's own topology block: `bonds`, `angles`, `dihedrals`,
    /// `impropers` or `pairs`. Absent or empty means the style contributes
    /// nothing.
    #[default]
    CategoryBlock,
    /// The atoms, or rows the kernel finds for itself. Nothing gates it.
    Atoms,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamSource {
    /// Parameters come from the style's type-definition rows (the `tp` slice).
    /// A style with no rows resolves nothing, and is an error.
    TypeRows,
    /// Parameters are resolved per interaction by the typifier and baked into
    /// [`Frame`] columns; `tp` is ignored and may legitimately be empty.
    PerInstance,
}

/// Which of a force field's special-bonds weight sets scales a pair style.
///
/// A force field may scale close van-der-Waals and electrostatic neighbours
/// differently — Amber uses `1/2` and `1/1.2` — and in molrs those are
/// separate kernels, so each has to say which set is its own. Declared at
/// registration rather than guessed from the style's name: a name is a label,
/// and this is a fact about the physics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialClass {
    /// Scaled by the force field's van-der-Waals weights.
    Vdw,
    /// Scaled by its electrostatic weights.
    Coulomb,
}

/// Maps `(category, style_name)` to the constructor that builds its potential
/// and the [`ParamSource`] it resolves parameters from.
#[derive(Default)]
pub struct KernelRegistry {
    ctors: HashMap<(String, String), Registration>,
}

/// What is known about one `(category, style_name)`.
struct Registration {
    ctor: KernelConstructor,
    source: ParamSource,
    /// Which block's emptiness means this style contributes nothing. Reset to
    /// the default by an override, for the reason `typed` is — a re-registered
    /// style is a different force law and inherits none of the old one's
    /// declarations.
    rows: RowSource,
    /// The neighbour-driven form of a pair style, when it has one, and which
    /// special-bonds weights scale it.
    ///
    /// The registered constructor resolves its parameters against a **pair
    /// list** and can only answer for that list. A neighbour table is a
    /// different list every rebuild, so an evaluation driven by one needs a
    /// kernel that finds its parameters from the atoms instead. Same
    /// signature, different question — so it is a second registration rather
    /// than a flag on the first.
    typed: Option<(KernelConstructor, SpecialClass)>,
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
        // A registration is one unit. Keeping the previous `typed` across an
        // override would leave `to_potentials` and `to_typed_potentials`
        // evaluating *different force fields* for the same style name, with
        // nothing to say so — so an override clears it and the caller
        // re-registers both.
        self.ctors.insert(
            (category.to_owned(), name.to_owned()),
            Registration {
                ctor,
                source,
                rows: RowSource::CategoryBlock,
                typed: None,
            },
        );
    }

    /// Register the neighbour-driven form of an already-registered pair style.
    ///
    /// A style without one cannot be evaluated over a neighbour table at all,
    /// and [`ForceField::to_typed_potentials`](crate::ff::forcefield::ForceField::to_typed_potentials)
    /// says so rather than quietly falling back to the compiled form, whose
    /// parameters would belong to a pair list nobody is evaluating.
    /// # Panics
    ///
    /// If `(category, name)` has no compiled registration. A neighbour-driven
    /// form is an *alternative* way to build a style that already exists, so a
    /// silent no-op here would leave the caller believing their style works
    /// under MD when `to_typed_potential` will refuse it.
    /// Declare where a registered style's rows come from.
    ///
    /// Only needed to say [`RowSource::Atoms`]; the default is the category's
    /// topology block. Like [`register_typed`](Self::register_typed) this is a
    /// second statement about a style that must already exist, and an override
    /// resets it — a re-registered style is a different force law.
    pub fn declare_rows(&mut self, category: &str, name: &str, rows: RowSource) {
        let r = self
            .ctors
            .get_mut(&(category.to_owned(), name.to_owned()))
            .unwrap_or_else(|| {
                panic!(
                    "declare_rows('{category}', '{name}') before the style is registered: \
                     where a kernel's rows come from is a statement about that kernel"
                )
            });
        r.rows = rows;
    }

    /// Where this style's rows come from, or `None` if it is not registered.
    pub fn row_source(&self, category: &str, name: &str) -> Option<RowSource> {
        self.ctors
            .get(&(category.to_owned(), name.to_owned()))
            .map(|r| r.rows)
    }

    pub fn register_typed(
        &mut self,
        category: &str,
        name: &str,
        ctor: KernelConstructor,
        special: SpecialClass,
    ) {
        let r = self
            .ctors
            .get_mut(&(category.to_owned(), name.to_owned()))
            .unwrap_or_else(|| {
                panic!(
                    "register_typed('{category}', '{name}') before the style is registered: \
                     a neighbour-driven form is an alternative to a compiled one, not a \
                     registration of its own"
                )
            });
        r.typed = Some((ctor, special));
    }

    /// The constructor registered for `(category, name)`, if any.
    pub fn get(&self, category: &str, name: &str) -> Option<KernelConstructor> {
        self.ctors
            .get(&(category.to_owned(), name.to_owned()))
            .map(|r| r.ctor)
    }

    /// The neighbour-driven constructor for `(category, name)` and the weight
    /// set that scales it, if it has one.
    pub fn get_typed(
        &self,
        category: &str,
        name: &str,
    ) -> Option<(KernelConstructor, SpecialClass)> {
        self.ctors
            .get(&(category.to_owned(), name.to_owned()))
            .and_then(|r| r.typed)
    }

    /// The [`ParamSource`] declared for `(category, name)`, if it is registered.
    pub fn param_source(&self, category: &str, name: &str) -> Option<ParamSource> {
        self.ctors
            .get(&(category.to_owned(), name.to_owned()))
            .map(|r| r.source)
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
        //
        // It is the BUFFERED Coulomb, `E = k·qᵢqⱼ/(D·(r + δ))`, with k / D / δ all
        // read from the style. `δ = 0` is the textbook Coulomb (OPLS, LAMMPS);
        // `k = 332.0716, D = 1.0, δ = 0.05` is MMFF's electrostatics. MMFF owns no
        // electrostatic kernel of its own — it configures this one.
        r.register_with(
            "pair",
            "coul/cut",
            pair::coul_cut::pair_coul_cut_ctor,
            ParamSource::PerInstance,
        );
        // MMFF94 — five per-instance BONDED styles. Their kernels read the columns
        // the typifier bakes (`kb`/`r0`, `ka`/`theta0`, `kba_*`, `v1`/`v2`/`v3`,
        // `koop`), never a type row: MMFF's context rules (aromaticity, ring size,
        // equivalence degradation, empirical fallbacks) are not a
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
        // UFF — per-instance bonded + LJ (typifier bakes kb/r0, ka/order/c*, V/order)
        r.register_with(
            "bond",
            "uff_bond",
            bond::uff::uff_bond_ctor,
            ParamSource::PerInstance,
        );
        r.register_with(
            "angle",
            "uff_angle",
            angle::uff::uff_angle_ctor,
            ParamSource::PerInstance,
        );
        r.register_with(
            "dihedral",
            "uff_torsion",
            dihedral::uff::uff_torsion_ctor,
            ParamSource::PerInstance,
        );
        r.register_with(
            "pair",
            "uff_lj",
            pair::uff::uff_lj_ctor,
            ParamSource::PerInstance,
        );
        r.register_with(
            "improper",
            "uff_inversion",
            improper::uff::uff_inversion_ctor,
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
        r.register_with(
            "pair",
            "coul/long/pme",
            kspace::pme::pme_ctor,
            ParamSource::PerInstance,
        );
        // PME sums in reciprocal space over the atoms' charges and subtracts
        // `exclusions`; it reads no `pairs` block, so the pair category's gate
        // must not delete it when there is none.
        r.declare_rows("pair", "coul/long/pme", RowSource::Atoms);
        // The neighbour-driven counterparts. Same styles, parameters keyed on
        // the atoms rather than on a `pairs` block, which is what an evaluation
        // over a rebuilt neighbour table needs.
        r.register_typed(
            "pair",
            "lj/cut",
            pair::lj_cut::pair_lj_cut_typed_ctor,
            SpecialClass::Vdw,
        );
        r.register_typed(
            "pair",
            "lj/class2",
            pair::lj_class2::pair_lj_class2_typed_ctor,
            SpecialClass::Vdw,
        );
        r.register_typed(
            "pair",
            "buck",
            pair::buck::pair_buck_typed_ctor,
            SpecialClass::Vdw,
        );
        r.register_typed(
            "pair",
            "morse",
            pair::morse::pair_morse_typed_ctor,
            SpecialClass::Vdw,
        );
        r.register_typed(
            "pair",
            "uff_lj",
            pair::uff::uff_lj_typed_ctor,
            SpecialClass::Vdw,
        );
        r.register_typed(
            "pair",
            "mmff_vdw",
            pair::mmff::mmff_vdw_typed_ctor,
            SpecialClass::Vdw,
        );
        r.register_typed(
            "pair",
            "coul/cut",
            pair::coul_cut::pair_coul_cut_typed_ctor,
            SpecialClass::Coulomb,
        );
        r.register_typed(
            "pair",
            "coul/tt",
            pair::tang_toennies::pair_tang_toennies_typed_ctor,
            SpecialClass::Coulomb,
        );
        r.register_typed(
            "pair",
            "thole",
            pair::thole::pair_thole_typed_ctor,
            SpecialClass::Coulomb,
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
///
/// # No in-tree caller, by design
///
/// Nothing in molrs, molpack or any binder calls this; only its own unit test
/// does. That is what an extension point looks like, and it is load-bearing
/// rather than speculative: [`Style::to_potential`] resolves its kernel through
/// [`lookup_kernel`] on the **global** registry, and no API accepts a
/// [`KernelRegistry`] of the caller's own, so an out-of-tree kernel has no
/// other door. `architecture-rules.md` names this registry as the project's
/// open-dispatch mechanism.
///
/// [`Style::to_potential`]: crate::ff::forcefield::Style::to_potential
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

/// Look up the neighbour-driven kernel a pair style declared, and the weight
/// set that scales it.
pub fn lookup_typed_kernel(
    category: &str,
    name: &str,
) -> Option<(KernelConstructor, SpecialClass)> {
    global().read().unwrap().get_typed(category, name)
}

/// Look up the [`ParamSource`] a style's kernel declared.
pub fn lookup_param_source(category: &str, name: &str) -> Option<ParamSource> {
    global().read().unwrap().param_source(category, name)
}

/// Where a style's rows come from, from the global registry.
pub fn lookup_row_source(category: &str, name: &str) -> Option<RowSource> {
    global().read().unwrap().row_source(category, name)
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
        assert!(r.get("pair", "coul/long/pme").is_some());
        assert!(r.get("kspace", "pme").is_none());
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

    /// Every neighbour-driven form registered by [`KernelRegistry::builtin`] is
    /// still there when `builtin` returns.
    ///
    /// Re-registering a style clears its typed entry on purpose — an override
    /// replaces the force law, and a stale neighbour-driven form would make
    /// `to_potentials` and `to_typed_potentials` evaluate different physics for
    /// the same name. That makes registration **order-sensitive**: a
    /// `register_typed` followed later by a `register` for the same key drops
    /// the typed form silently, and the only symptom is `to_typed_potentials`
    /// reporting a style it was told about as unknown. This pins the order.
    #[test]
    fn every_typed_registration_survives_builtin() {
        let r = KernelRegistry::builtin();
        for name in [
            "lj/cut",
            "lj/class2",
            "buck",
            "morse",
            "uff_lj",
            "mmff_vdw",
            "coul/cut",
            "coul/tt",
            "thole",
        ] {
            assert!(
                r.get_typed("pair", name).is_some(),
                "pair '{name}' lost its neighbour-driven form: a later \
                 register/register_with for the same key must come *before* \
                 its register_typed"
            );
        }
        // The same ordering trap, for the other per-style declaration.
        assert_eq!(
            r.row_source("pair", "coul/long/pme"),
            Some(RowSource::Atoms),
            "PME lost its row-source declaration: a later register/register_with \
             for the same key must come *before* its declare_rows"
        );
    }

    /// PME survives a frame with no `pairs` block.
    ///
    /// It is registered under `pair` because that is where an electrostatic
    /// style belongs, and `Style::to_potential` skips a pair style whose
    /// `pairs` block is absent or empty — a rule that is right for every other
    /// pair kernel and deleted PME outright. The symptom was a system with
    /// zero long-range electrostatics and no error: the style was declared,
    /// accepted, and dropped.
    #[test]
    fn pme_is_not_gated_on_a_pairs_block() {
        let r = KernelRegistry::builtin();
        assert_eq!(
            r.row_source("pair", "coul/long/pme"),
            Some(RowSource::Atoms),
            "PME reads charges and exclusions, never `pairs`"
        );
        for gated in ["lj/cut", "coul/cut", "buck", "thole"] {
            assert_eq!(
                r.row_source("pair", gated),
                Some(RowSource::CategoryBlock),
                "'{gated}' does read `pairs`, so an empty one still means no work"
            );
        }
    }

    /// An override drops the neighbour-driven form rather than keeping a form
    /// built for the force law that was just replaced.
    #[test]
    fn re_registering_a_style_clears_its_typed_form() {
        let mut r = KernelRegistry::new();
        r.register("pair", "lj/cut", pair::lj_cut::pair_lj_cut_ctor);
        r.register_typed(
            "pair",
            "lj/cut",
            pair::lj_cut::pair_lj_cut_typed_ctor,
            SpecialClass::Vdw,
        );
        assert!(r.get_typed("pair", "lj/cut").is_some());
        r.register("pair", "lj/cut", pair::buck::pair_buck_ctor);
        assert!(
            r.get_typed("pair", "lj/cut").is_none(),
            "the typed form was built for lj/cut's parameters, not buck's"
        );
    }
}
