//! mmff-ele-compose ac-002 / ac-003 — `pair/coul/cut` **is** the buffered Coulomb,
//! and it refuses to invent the numbers the force field did not give it.
//!
//! ```text
//! E(r) = k · qᵢqⱼ / (D · (r + δ))          r < r_cut
//! ```
//!
//! with `k` (the Coulomb constant), `D` (dielectric) and `δ` (buffering distance)
//! all taken from the **style params**. MMFF's electrostatics is not a kernel: it
//! is this kernel at `k = 332.0716`, `D = 1.0`, `δ = 0.05 Å`. `coul/cut`'s own
//! module doc already said its kernel was "per-atom, mirroring `mmff_ele`" — two
//! implementations of one physics, and the author knew.
//!
//! # The two halves, and why they are one file
//!
//! * **ac-002 — the generalization.** `δ = 0` must reproduce today's kernel
//!   BIT-FOR-BIT (`delta_zero_reproduces_the_unbuffered_kernel_bit_for_bit`); the
//!   MMFF parameterization must reproduce MMFF's buffered form
//!   (`mmff_electrostatics_is_the_generic_kernel_parameterised`). The generalization
//!   is only allowed to *add* a capability, never to move an existing number.
//!
//! * **ac-003 — the backdoor.** A style that does not carry `coulomb` /
//!   `dielectric` / `coulomb14scale` must be an `Err`. `mmff-orthogonal-02`'s
//!   bite-proof found that with `def_pairstyle("mmff_ele", &[])` — style present,
//!   params EMPTY — **18 of 19 energy tests still passed**, because the kernel read
//!   `sp.get("dielectric").unwrap_or(1.0)` / `sp.get("delta").unwrap_or(ELE_DELTA)`.
//!   The force field silently stopped being the source of MMFF's constants and the
//!   kernel used its own hardcoded copies: **the right numbers for the wrong
//!   reason.** `coul/cut` has the same disease
//!   (`style_params.get("coulomb14scale").unwrap_or(1.0)`).
//!
//! # The distinction this file encodes, honestly
//!
//! Not every default is a lie. Two of them are real, meaningful choices and they
//! are asserted to SURVIVE:
//!
//! | param | missing means | verdict |
//! |---|---|---|
//! | `cutoff` | "do not truncate" (∞) | genuine semantic default — KEEP |
//! | `delta` | "no buffer" (0.0) — textbook Coulomb | genuine semantic default — KEEP |
//! | `coulomb` | the kernel picks a Coulomb constant for you | **the kernel pretending the force field spoke — Err** |
//! | `dielectric` | the kernel picks D = 1 for you | **same — Err** |
//! | `coulomb14scale` | the kernel picks 1.0 for you | **same — Err** |
//!
//! # Why `k` must be a style param and not a shared constant
//!
//! MMFF uses Halgren's 332.0716; `coul/cut` used CODATA's 332.06371. The 2.4e-5
//! relative difference is worth 0.0036 kcal/mol on caffeine's `E_ele = −150.48` —
//! **above the 1e-3 RDKit parity tolerance** (proved from the frozen breakdown in
//! `mmff/energy.rs::the_two_coulomb_constants_are_not_interchangeable`). Both values
//! are correct; the force field decides. "Merge the two kernels by picking one
//! constant" would silently break parity.

use crate::helpers::{flat_coords, pairs_block, typed_atoms_block};
use molrs::ff::forcefield::{ForceField, Params, SpecialBonds};
use molrs::ff::potential::{Potential, lookup_kernel};
use molrs::store::frame::Frame;
use molrs::types::F;
use molrs::units::constants::COULOMB_REAL;

// ---------------------------------------------------------------------------
// MMFF's electrostatic parameters — the ORACLE, not a molrs constant.
//
// These four numbers are RDKit's (`Nonbonded.cpp`, `calcEleEnergy`): Halgren's
// Coulomb constant, the 0.05 Å buffering distance, the unit dielectric, and the
// 0.75 1-4 weight. They live here as the test's reference values; in the tree they
// must live in `ff/params/mmff.rs::MMFF_ELE_STYLE` — they are MMFF's PARAMETERS,
// not molrs's constants (ac-004, gated in `electrostatics_compose_gate.rs`).
// ---------------------------------------------------------------------------

/// Halgren's Coulomb constant, kcal·Å·mol⁻¹·e⁻² (MMFF.I eq. 5 / RDKit).
const COULOMB_MMFF: F = 332.0716;
/// MMFF's electrostatic buffering distance δ, Å.
const MMFF_DELTA: F = 0.05;
/// MMFF's dielectric D.
const MMFF_DIELECTRIC: F = 1.0;
/// MMFF's 1-4 Coulomb weight.
const MMFF_COUL_14: F = 0.75;

/// Energy agreement with a closed form — the "exact" column (kcal/mol).
const E_EXACT: F = 1.0e-10;
/// Force agreement with a closed form — the "exact" column (kcal/mol/Å).
const F_EXACT: F = 1.0e-8;
/// Analytic force vs central finite difference — the "numerical" column.
const F_NUMERICAL: F = 1.0e-4;

// ---------------------------------------------------------------------------
// Fixtures: a two-atom frame in the documented per-atom convention
// ---------------------------------------------------------------------------

/// Atom 0 at the origin, atom 1 at `sep`, with per-atom `charge` and a one-row
/// neighbour list (`atomi`/`atomj`/`is_14`) — exactly what a consumer hands
/// `ForceField::to_potentials` after typification.
fn two_atom_frame(sep: [F; 3], charges: [F; 2], is_14: bool) -> (Frame, Vec<F>) {
    let coords = [[0.0, 0.0, 0.0], sep];
    let mut frame = Frame::new();
    frame.insert(
        "atoms",
        typed_atoms_block(&coords, &["X", "X"], Some(&charges)),
    );
    frame.insert("pairs", pairs_block(&[0], &[1], &[is_14]));
    (frame, flat_coords(&coords))
}

/// Compile a `pair/coul/cut` style with exactly `params`, and a force field whose
/// 1-4 Coulomb weight is `coul14`.
///
/// The documented route: `ForceField` → `Style::to_potential` (which is what
/// projects `special_bonds` into `coulomb14scale`) → kernel.
fn compile(
    params: &[(&str, F)],
    coul14: F,
    frame: &Frame,
) -> Result<Option<Box<dyn Potential>>, String> {
    let mut ff = ForceField::new("probe");
    ff.def_pairstyle("coul/cut", params);
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, 1.0],
        coul: [0.0, 0.0, coul14],
    });
    let style = ff
        .get_style("pair", "coul/cut")
        .expect("the style just defined");
    style.to_potential(frame, ff.special_bonds())
}

/// The same, unwrapped to a live potential (panics with the ctor's own message).
fn potential(params: &[(&str, F)], coul14: F, frame: &Frame) -> Box<dyn Potential> {
    compile(params, coul14, frame)
        .unwrap_or_else(|e| panic!("pair/coul/cut failed to compile: {e}"))
        .expect("a one-row neighbour list is topology enough to build a pair potential")
}

/// MMFF's parameterization of the generic kernel — the whole thesis of this spec,
/// as a style-param list.
fn mmff_params() -> Vec<(&'static str, F)> {
    vec![
        ("coulomb", COULOMB_MMFF),
        ("dielectric", MMFF_DIELECTRIC),
        ("delta", MMFF_DELTA),
    ]
}

// ---------------------------------------------------------------------------
// ac-002 — the generic kernel carries k / D / δ
// ---------------------------------------------------------------------------

/// MMFF's buffered Coulomb, computed by `pair/coul/cut`.
///
/// RED today: the kernel ignores both `coulomb` and `delta` and evaluates
/// `COULOMB_REAL · qᵢqⱼ / r` — it is the ONLY assertion that can tell a kernel that
/// reads its constants from the force field apart from one that has its own copies.
#[test]
fn mmff_electrostatics_is_the_generic_kernel_parameterised() {
    let (frame, coords) = two_atom_frame([2.0, 0.0, 0.0], [0.5, -0.5], false);
    let pot = potential(&mmff_params(), MMFF_COUL_14, &frame);
    let (got, _) = pot.calc_energy_forces(&coords);

    // MMFF.I eq. 5 at r = 2 Å: E = k·q_i q_j / (D·(r + δ)).
    let want = COULOMB_MMFF * (0.5 * -0.5) / (MMFF_DIELECTRIC * (2.0 + MMFF_DELTA));
    assert!(
        (got - want).abs() < E_EXACT,
        "pair/coul/cut with MMFF's style params gave {got:.9}, want {want:.9}. The style says \
         coulomb={COULOMB_MMFF}, dielectric={MMFF_DIELECTRIC}, delta={MMFF_DELTA}; a kernel that \
         evaluates {} instead is using its own hardcoded constant ({COULOMB_REAL}) and no buffer \
         — the right shape for the wrong reason.",
        COULOMB_REAL * (0.5 * -0.5) / 2.0
    );
}

/// The 1-4 weight scales the *buffered* term, not some other term.
///
/// MMFF scales 1-4 Coulomb by 0.75 (and 1-4 vdW by nothing). RED today for the same
/// reason as above: the constant and the buffer are wrong, so the scaled value is too.
#[test]
fn the_1_4_weight_scales_the_buffered_coulomb() {
    let (frame, coords) = two_atom_frame([2.0, 0.0, 0.0], [0.5, -0.5], true);
    let pot = potential(&mmff_params(), MMFF_COUL_14, &frame);
    let (got, _) = pot.calc_energy_forces(&coords);

    let unscaled = COULOMB_MMFF * (0.5 * -0.5) / (MMFF_DIELECTRIC * (2.0 + MMFF_DELTA));
    let want = unscaled * MMFF_COUL_14;
    assert!(
        (got - want).abs() < E_EXACT,
        "a 1-4 pair under MMFF's params gave {got:.9}, want {want:.9} (= {unscaled:.9} × \
         {MMFF_COUL_14}). The 1-4 weight comes from `special_bonds` via `Style::to_potential`; it \
         must scale the SAME buffered term, not a second, unbuffered one."
    );
}

/// **The regression guard.** `δ = 0` reproduces today's kernel bit-for-bit.
///
/// GREEN today, and it must stay green through the generalization — that is the
/// entire claim of "pure refactor" at the kernel level. The geometries are chosen so
/// that `r` is exactly representable (r = 2, and the 3-4-5 triangle → r = 5), which
/// makes `r·r == dx²+dy²+dz²` exactly and leaves the comparison independent of how
/// the implementation associates its arithmetic. So this is a true `assert_eq!` on
/// `f64`, not a tolerance: the buffered kernel at `δ = 0`, `D = 1`, `k = COULOMB_REAL`
/// IS the kernel that exists today, down to the bit.
#[test]
fn delta_zero_reproduces_the_unbuffered_kernel_bit_for_bit() {
    let params = [
        ("coulomb", COULOMB_REAL),
        ("dielectric", 1.0),
        ("delta", 0.0),
    ];
    let charges = [0.8, -0.6];

    for sep in [[2.0, 0.0, 0.0], [3.0, 4.0, 0.0]] {
        let (frame, coords) = two_atom_frame(sep, charges, false);
        let pot = potential(&params, 1.0, &frame);
        let (energy, forces) = pot.calc_energy_forces(&coords);

        // Today's kernel, transcribed: `qq = k·qiqj`, `E += qq/r`, `factor = qq/(r²·r)`.
        let qq = COULOMB_REAL * (charges[0] * charges[1]);
        let r2 = sep[0] * sep[0] + sep[1] * sep[1] + sep[2] * sep[2];
        let r = r2.sqrt();
        let want_e = qq / r;
        let factor = qq / (r2 * r);

        assert_eq!(
            energy, want_e,
            "at δ = 0 the buffered kernel must BE the unbuffered one, bit for bit. r = {r}: got \
             {energy:?}, want {want_e:?}. The generalization is allowed to add a capability; it is \
             not allowed to move a number that exists today."
        );
        for dim in 0..3 {
            let want_f = factor * sep[dim];
            assert_eq!(
                forces[3 + dim],
                want_f,
                "δ = 0, r = {r}, force component {dim} on atom j: got {:?}, want {:?}",
                forces[3 + dim],
                want_f
            );
            assert_eq!(
                forces[dim], -want_f,
                "δ = 0, r = {r}, force component {dim} on atom i must be the negation of j's"
            );
        }
    }
}

/// The cutoff still truncates. (GREEN today; the generalization must not lose it.)
#[test]
fn the_cutoff_still_truncates() {
    let (frame, coords) = two_atom_frame([2.0, 0.0, 0.0], [0.5, -0.5], false);
    let params = [
        ("coulomb", COULOMB_MMFF),
        ("dielectric", MMFF_DIELECTRIC),
        ("delta", MMFF_DELTA),
        ("cutoff", 1.5),
    ];
    let pot = potential(&params, MMFF_COUL_14, &frame);
    let (e, forces) = pot.calc_energy_forces(&coords);
    assert_eq!(
        e, 0.0,
        "a pair at 2 Å with cutoff 1.5 Å must contribute exactly zero, got {e:e}"
    );
    assert!(
        forces.iter().all(|f| *f == 0.0),
        "a truncated pair must contribute no force either: {forces:?}"
    );
}

// ---------------------------------------------------------------------------
// ac-003 — THE BACKDOOR: a style missing force-field data is an Err
// ---------------------------------------------------------------------------

/// A `pair/coul/cut` style with **empty params** must not compile.
///
/// This is the exact shape of the backdoor: `def_pairstyle("mmff_ele", &[])` left 18
/// of 19 energy tests green because the kernel fell back to its own hardcoded copies
/// of MMFF's constants. RED today: `coul/cut` builds happily from nothing and
/// computes `COULOMB_REAL·qᵢqⱼ/r` out of constants nobody gave it.
#[test]
fn an_empty_coul_cut_style_does_not_compile() {
    let (frame, coords) = two_atom_frame([2.0, 0.0, 0.0], [0.5, -0.5], false);

    let result = compile(&[], 0.75, &frame);
    let Err(err) = result else {
        let e = result
            .expect("checked Ok above")
            .expect("pairs present")
            .calc_energy_forces(&coords)
            .0;
        panic!(
            "a `pair/coul/cut` style with EMPTY params compiled, and computed E = {e:.6} kcal/mol \
             from constants no force field ever gave it. That is the backdoor this spec exists to \
             close: with `def_pairstyle(\"mmff_ele\", &[])` — style present, params empty — 18 of \
             19 MMFF energy tests still passed, because the kernel had \
             `sp.get(\"dielectric\").unwrap_or(1.0)`. A kernel that supplies the force field's own \
             constants is not reading the force field."
        );
    };
    assert!(
        err.contains("coulomb") || err.contains("dielectric"),
        "the error must name the parameter the force field failed to supply, got: {err:?}"
    );
}

/// Missing `coulomb` — the Coulomb constant is the force field's to choose.
///
/// 332.0716 (Halgren) and 332.06371 (CODATA) are BOTH correct, and they are not
/// interchangeable at the 1e-3 parity tolerance. A kernel that defaults this is
/// choosing a force field for its caller.
#[test]
fn a_coul_cut_style_without_the_coulomb_constant_does_not_compile() {
    let (frame, _) = two_atom_frame([2.0, 0.0, 0.0], [0.5, -0.5], false);
    let params = [("dielectric", MMFF_DIELECTRIC), ("delta", MMFF_DELTA)];
    let Err(err) = compile(&params, MMFF_COUL_14, &frame) else {
        panic!(
            "`pair/coul/cut` compiled with no `coulomb` style param. It would evaluate with \
             whatever constant the kernel happens to hold — silently CODATA's 332.06371 for a \
             force field (MMFF) whose constant is Halgren's 332.0716. Both are correct; the FORCE \
             FIELD decides, and a kernel with a default has taken the decision away from it."
        );
    };
    assert!(
        err.contains("coulomb"),
        "the error must name `coulomb`, got: {err:?}"
    );
}

/// Missing `dielectric` — `D = 1.0` is the kernel pretending the force field spoke.
#[test]
fn a_coul_cut_style_without_a_dielectric_does_not_compile() {
    let (frame, _) = two_atom_frame([2.0, 0.0, 0.0], [0.5, -0.5], false);
    let params = [("coulomb", COULOMB_MMFF), ("delta", MMFF_DELTA)];
    let Err(err) = compile(&params, MMFF_COUL_14, &frame) else {
        panic!(
            "`pair/coul/cut` compiled with no `dielectric` style param. `unwrap_or(1.0)` is not a \
             semantic default — a dielectric is a physical property of the medium the force field \
             was parameterised in, and the kernel does not know it."
        );
    };
    assert!(
        err.contains("dielectric"),
        "the error must name `dielectric`, got: {err:?}"
    );
}

/// Missing `coulomb14scale` — the kernel must not invent the 1-4 weight either.
///
/// This one can only be asked of the KERNEL: `Style::to_potential` always projects
/// `special_bonds.coul_14()` into the pair style's params (house rule), so via the
/// ForceField route the key is never absent. That projection is exactly why the
/// kernel's `unwrap_or(1.0)` looks harmless — and exactly why it is not: a kernel
/// reached by any other route (a hand-built `Params`, a future scheduler, a binder)
/// would silently unscale every 1-4 pair. MMFF's is 0.75; unscaling it is a real
/// energy error on every molecule with a 1-4 pair.
///
/// RED today: `pair_coul_cut_ctor` reads `style_params.get("coulomb14scale").unwrap_or(1.0)`.
#[test]
fn the_kernel_refuses_to_invent_the_1_4_coulomb_weight() {
    let (frame, coords) = two_atom_frame([2.0, 0.0, 0.0], [0.5, -0.5], true);
    let ctor = lookup_kernel("pair", "coul/cut")
        .expect("`pair/coul/cut` must be registered — this spec composes MMFF out of it");

    // Everything the force field owes the kernel EXCEPT the 1-4 weight.
    let params = Params::from_pairs(&[
        ("coulomb", COULOMB_MMFF),
        ("dielectric", MMFF_DIELECTRIC),
        ("delta", MMFF_DELTA),
    ]);

    let built = ctor(&params, &[], &frame);
    let Err(err) = built else {
        let got = built
            .expect("checked Ok above")
            .calc_energy_forces(&coords)
            .0;
        let want =
            COULOMB_MMFF * (0.5 * -0.5) * MMFF_COUL_14 / (MMFF_DIELECTRIC * (2.0 + MMFF_DELTA));
        panic!(
            "`pair_coul_cut_ctor` built a potential from params carrying no `coulomb14scale`, and \
             it computed E = {got:.6} kcal/mol. The frame's one pair is FLAGGED 1-4, so the kernel \
             weighted it by a factor it chose itself (1.0) instead of the force field's \
             ({MMFF_COUL_14} for MMFF), which says {want:.6}. `coulomb14scale` is projected out of \
             `SpecialBonds` by `Style::to_potential`: it is force-field data, and a kernel that \
             defaults it has stopped being told."
        );
    };
    assert!(
        err.contains("coulomb14scale"),
        "the error must name `coulomb14scale`, got: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// ac-003, the other direction — the SEMANTIC defaults survive
//
// "Delete every unwrap_or" is the wrong lesson and would be a worse bug than the
// one being fixed. `cutoff = ∞` ("do not truncate") and `delta = 0` ("no buffer")
// are real, meaningful choices that a force field is entitled to leave unsaid.
// ---------------------------------------------------------------------------

/// No `cutoff` → do not truncate. (GREEN today; must stay green.)
#[test]
fn a_missing_cutoff_is_a_semantic_default_not_an_error() {
    // 50 Å apart: anything but an infinite cutoff would drop this pair.
    let (frame, coords) = two_atom_frame([50.0, 0.0, 0.0], [0.5, -0.5], false);
    let params = [("coulomb", COULOMB_REAL), ("dielectric", 1.0)];

    let pot = compile(&params, 1.0, &frame)
        .unwrap_or_else(|e| {
            panic!(
                "a `coul/cut` style with no `cutoff` must COMPILE: `cutoff = ∞` (\"do not \
                 truncate\") is a genuine semantic default — a real, meaningful choice, not the \
                 kernel pretending the force field spoke. Got: {e}"
            )
        })
        .expect("pairs present");

    let (e, _) = pot.calc_energy_forces(&coords);
    let want = COULOMB_REAL * (0.5 * -0.5) / 50.0;
    assert!(
        (e - want).abs() < E_EXACT,
        "with no cutoff the pair at 50 Å must still interact: got {e:e}, want {want:e}"
    );
}

/// No `delta` → no buffer (textbook Coulomb). (GREEN today; must stay green.)
///
/// The counterpart of the `dielectric` test above, and the pair that makes the
/// distinction honest: `delta = 0` degenerates the kernel into the one every other
/// force field wants, while `dielectric = 1.0` would be the kernel *answering a
/// question only the force field can answer*.
#[test]
fn a_missing_delta_is_a_semantic_default_not_an_error() {
    let (frame, coords) = two_atom_frame([2.0, 0.0, 0.0], [0.5, -0.5], false);
    let params = [("coulomb", COULOMB_REAL), ("dielectric", 1.0)];

    let pot = compile(&params, 1.0, &frame)
        .unwrap_or_else(|e| {
            panic!(
                "a `coul/cut` style with no `delta` must COMPILE: δ = 0 (\"no buffer\") is a \
                 genuine semantic default — it is what every non-MMFF force field means by \
                 Coulomb. Got: {e}"
            )
        })
        .expect("pairs present");

    let (e, _) = pot.calc_energy_forces(&coords);
    let want = COULOMB_REAL * (0.5 * -0.5) / 2.0;
    assert_eq!(
        e, want,
        "with no `delta` the kernel must be the textbook Coulomb, bit for bit: got {e:?}, want \
         {want:?}"
    );
}

// ---------------------------------------------------------------------------
// The buffer's reason to exist, and the kernel invariants
// ---------------------------------------------------------------------------

/// A **buffered** pair at zero distance has finite energy — that is what δ is FOR.
///
/// MMFF's charges sit on nuclei and its buffered form keeps `E` finite at `R = 0`
/// (`E = k·qᵢqⱼ/(D·δ)`); RDKit's `calcEleEnergy` and the deleted `MMFFElectrostatic`
/// both add exactly that term, and skip only the (undefined) force direction. RED
/// today: `coul/cut` skips any pair with `r² < 1e-24` outright, so it would silently
/// drop MMFF's contribution — a behaviour change dressed as a refactor.
#[test]
fn a_buffered_pair_at_zero_distance_has_finite_energy() {
    let (frame, coords) = two_atom_frame([0.0, 0.0, 0.0], [0.5, -0.5], false);
    let pot = potential(&mmff_params(), MMFF_COUL_14, &frame);
    let (e, forces) = pot.calc_energy_forces(&coords);

    let want = COULOMB_MMFF * (0.5 * -0.5) / (MMFF_DIELECTRIC * MMFF_DELTA);
    assert!(
        (e - want).abs() < E_EXACT,
        "two coincident charges under MMFF's buffered Coulomb must contribute \
         k·qᵢqⱼ/(D·δ) = {want:.6} kcal/mol, got {e:.6}. Keeping E finite at R = 0 is the \
         BUFFER'S ENTIRE PURPOSE — a generalized kernel that inherited `coul/cut`'s \
         `r² < 1e-24 → skip` would drop the term and call it a refactor."
    );
    assert!(
        forces.iter().all(|f| f.is_finite()),
        "the force at R = 0 has no direction and must be zero/finite, got {forces:?}"
    );
}

/// An **unbuffered** pair at zero distance is skipped (no NaN, no ∞).
/// (GREEN today; the δ = 0 path must keep its guard.)
#[test]
fn an_unbuffered_pair_at_zero_distance_is_skipped() {
    let (frame, coords) = two_atom_frame([0.0, 0.0, 0.0], [0.5, -0.5], false);
    let params = [
        ("coulomb", COULOMB_REAL),
        ("dielectric", 1.0),
        ("delta", 0.0),
    ];
    let pot = potential(&params, 1.0, &frame);
    let (e, forces) = pot.calc_energy_forces(&coords);
    assert_eq!(
        e, 0.0,
        "with δ = 0 a coincident pair would be k·qq/0 = −∞; the kernel must skip it, got {e:e}"
    );
    assert!(
        forces.iter().all(|f| f.is_finite()),
        "forces at R = 0 with δ = 0 must be finite, got {forces:?}"
    );
}

/// Newton's third law on the buffered kernel.
#[test]
fn newtons_third_law_holds_for_the_buffered_kernel() {
    let (frame, coords) = two_atom_frame([1.7, 0.3, -0.4], [0.8, -0.6], false);
    let pot = potential(&mmff_params(), MMFF_COUL_14, &frame);
    let (_, f) = pot.calc_energy_forces(&coords);
    for dim in 0..3 {
        assert!(
            (f[dim] + f[3 + dim]).abs() < F_EXACT,
            "dim {dim}: f_i = {}, f_j = {} — a pair kernel's forces are equal and opposite",
            f[dim],
            f[3 + dim]
        );
    }
}

/// Analytic forces vs central finite differences on the buffered kernel.
///
/// The δ-dependent gradient is `−k·qᵢqⱼ/(D·(r+δ)²) · r̂`: the `(r+δ)²` in the
/// denominator, not `r²`. An implementation that buffers the ENERGY and forgets to
/// buffer the FORCE passes every energy assertion in this file and fails here.
#[test]
fn buffered_forces_match_finite_difference() {
    let (frame, coords) = two_atom_frame([1.7, 0.3, -0.4], [0.8, -0.6], false);
    let pot = potential(&mmff_params(), MMFF_COUL_14, &frame);
    let (_, analytic) = pot.calc_energy_forces(&coords);
    let numerical =
        crate::helpers::numerical_forces(|c| pot.calc_energy_forces(c).0, &coords, 1e-7);
    for i in 0..coords.len() {
        assert!(
            (analytic[i] - numerical[i]).abs() < F_NUMERICAL,
            "component {i}: analytic {} vs finite-difference {} — the buffered gradient is \
             −k·qᵢqⱼ/(D·(r+δ)²)·r̂, with (r+δ)² in the denominator, not r²",
            analytic[i],
            numerical[i]
        );
    }
}
