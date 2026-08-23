//! The AM1-BCC bridge, called from C++ — chem-perceive-12 ac-001 and ac-002.
//!
//! Both criteria are claims about the FFI boundary, so both are tested across it:
//! every call below leaves Rust, enters the C++ translation unit
//! `tests/cxx/bridge_probe.cc`, and re-enters molrs through the cxx-generated
//! `molrs::am1_bcc_assign_frame_from_base`. A Rust-only test that called
//! `am1_bcc_assign_frame_from_base_result` and asserted `Err` would prove nothing:
//! that function ALREADY returns `Err` today, and would have passed before this
//! spec was written. What is broken is the shim wrapped around it —
//! `.expect(...)`, behind a declaration cxx marks `noexcept` — and only C++ can
//! see it.
//!
//! ## ac-001 — a user-chemistry error must be catchable, not fatal
//!
//! Boron is the spec's example and it is a real one: `BF4⁻`, the
//! tetrafluoroborate anion molcrafts actually uses in its ionic-liquid
//! electrolytes, has no row in `BCCPARM.DAT`. molrs says so —
//! `missing BCC correction for bond B|71|1` — and today the bridge turns that
//! sentence into `panic! -> abort`, killing the engine that asked. The tests
//! demand it arrive as a `rust::Error` a C++ `catch` can hold.
//!
//! "The process does NOT abort" is not observable from inside the process that
//! aborts, so the boron calls run in a **child process** (a re-exec of this test
//! binary, `--exact <name> --ignored`) and the parent asserts on its exit status.
//! Today that child dies of SIGABRT; that is the RED. A test that made the call
//! in-process would, today, take the whole test binary — ac-002 included — down
//! with it, and report the bug as an unexplained signal 6.
//!
//! ## ac-002 — ABCG2 must be reachable from C++
//!
//! The bridge hardcodes `BccParameterSet::Bcc` (lib.rs), so ABCG2 — a whole
//! parameter table molrs already ships and already tests — is unreachable from
//! Atomiverse. The expected numbers are anchored on the offline AM1-BCC reference table
//! (`tests/am1bcc_reference.rs`),
//! not re-derived here, and the molecule is chosen so the two parameter sets
//! DISAGREE — otherwise "ABCG2 charges" and "BCC charges" would be the same
//! assertion, and a bridge that ignored the selector would pass.

/// Offline reference table (no AmberTools at test time).
///
/// Included by path rather than copied: the ABCG2 column is the oracle's, and a
/// second transcription of it in this crate would be a second thing to keep in
/// step with `scripts/gen_am1bcc_oracle.py`. The file is plain data with no
/// imports, which is what makes this safe.
#[path = "am1bcc_reference.rs"]
mod am1bcc_reference;

use am1bcc_reference::{CASES, ReferenceCase};

/// Reference charges are to 4 decimals and the BCC increments carry 4, so
/// anything above 1e-4 is a real disagreement rather than formatting. Same
/// constant, same reason, as the deleted molrs-side typifier tests used.
const CHARGE_TOL: f64 = 1.0e-4;

/// The oracle case with the given name.
fn case(name: &str) -> &'static ReferenceCase {
    CASES
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no reference case named '{name}'"))
}

/// The premise of ac-002: on this molecule the two parameter sets really do
/// disagree, so "ABCG2 charges" is a distinguishable claim and a bridge that
/// ignored the selector could not pass by accident.
///
/// Not gated on the C++ probe: it is a property of the oracle, and if it ever
/// stops holding, every ABCG2 assertion below silently becomes a BCC assertion.
#[test]
fn the_two_parameter_sets_disagree_on_water() {
    let water = case("water");
    let gap = water
        .bcc_charges
        .iter()
        .zip(water.abcg2_charges)
        .map(|(bcc, abcg2)| (bcc - abcg2).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        gap > 10.0 * CHARGE_TOL,
        "water's BCC and ABCG2 columns differ by only {gap:.6} e — pick another oracle case, \
         this one cannot tell the two parameter sets apart"
    );
}

/// The C++ half of these tests could not be built, so ac-001 and ac-002 are
/// untested. Fails loudly — the one thing it must never do is pass.
///
/// `tests/cxx/bridge_probe.cc` accepts the pre-spec-12 signature AND both shapes
/// the spec allows (`total_charge` kept or dropped), so a failure here means the
/// bridge grew a shape none of them cover — most likely a parameter-set selector
/// that is not the `&str` these tests pin. Reconcile `bridge_probe.cc` with
/// `build.rs`'s `CXX_BRIDGE_SCHEMA`; `cargo build --manifest-path molrs-cxxapi/Cargo.toml -vv`
/// prints the compiler's diagnostic, and build.rs emits it as a `cargo::warning`.
#[cfg(not(cxx_probe))]
#[test]
fn the_cpp_probe_compiles_against_the_generated_bridge() {
    panic!(
        "tests/cxx/bridge_probe.cc did not compile against the generated bridge header — \
         ac-001 (a catchable rust::Error) and ac-002 (ABCG2 from C++) are untested"
    );
}

/// Everything that crosses into C++.
///
/// One inline module, gated once, so that a build without the probe has no dead
/// code rather than a dozen `#[cfg]`s and a pile of unused-import warnings.
#[cfg(cxx_probe)]
mod cxx {
    use std::ffi::{CStr, CString, c_char, c_int, c_void};
    use std::process::{Command, ExitStatus};

    use molrs::store::keys;
    use molrs::system::bond::{BondNumber, BondType};
    use molrs::{AtomId, Atomistic};
    use molrs_cxxapi::FrameRef;

    use super::{CHARGE_TOL, ReferenceCase, case};

    // ── The C++ side ─────────────────────────────────────────────────────────

    /// Status codes of [`molrs_cxxapi_test_am1_bcc_assign`]; see `bridge_probe.cc`.
    const OK: c_int = 0;
    const CAUGHT_RUST_ERROR: c_int = 1;
    const CAUGHT_STD_EXCEPTION: c_int = 2;
    const CAUGHT_UNKNOWN: c_int = 3;
    const NO_SELECTOR: c_int = 4;

    unsafe extern "C" {
        /// Calls `molrs::am1_bcc_assign_frame_from_base` inside
        /// `try { … } catch (const rust::Error&)`, in C++, against the generated
        /// bridge header.
        fn molrs_cxxapi_test_am1_bcc_assign(
            frame_ref: *mut c_void,
            am1: *const f64,
            n_am1: usize,
            parameter_set: *const c_char,
            out: *mut f64,
            out_cap: usize,
            out_len: *mut usize,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
    }

    /// What C++ saw when it called the bridge.
    #[derive(Debug, Clone, PartialEq)]
    enum Outcome {
        /// The bridge returned; these are the corrected charges.
        Charges(Vec<f64>),
        /// `catch (const rust::Error&)` — the outcome ac-001 is about.
        CaughtRustError(String),
        /// `catch (const std::exception&)` — e.g. cxx's own, on a bad `rust::Str`.
        CaughtStdException(String),
        /// `catch (...)`.
        CaughtUnknown(String),
        /// The bridge declares no parameter-set argument: ABCG2 is unreachable.
        NoSelector,
    }

    /// Call the bridge from C++ and report what came back.
    ///
    /// The frame crosses as an opaque `*mut FrameRef`, which is what cxx hands
    /// Atomiverse behind `rust::Box<FrameRef>`: `FrameRef` is
    /// `#[repr(transparent)]` over `molrs_ffi::FrameRef`, and the C++ side only
    /// ever holds it by reference.
    fn assign_from_cpp(fref: &mut FrameRef, am1: &[f64], parameter_set: &str) -> Outcome {
        let set = CString::new(parameter_set).expect("the parameter set has no interior NUL");
        let mut charges = vec![0.0_f64; am1.len().max(1)];
        let mut n_charges: usize = 0;
        let mut err: Vec<c_char> = vec![0; 512];

        // SAFETY: every pointer is to a live local that outlives the call, and
        // every capacity handed over is the true length of the buffer it
        // describes. `frame_ref` is a `*mut FrameRef`, which the C++ side
        // dereferences as the opaque `molrs::FrameRef` cxx generated it for.
        let status = unsafe {
            molrs_cxxapi_test_am1_bcc_assign(
                std::ptr::from_mut(fref).cast::<c_void>(),
                am1.as_ptr(),
                am1.len(),
                set.as_ptr(),
                charges.as_mut_ptr(),
                charges.len(),
                &raw mut n_charges,
                err.as_mut_ptr(),
                err.len(),
            )
        };

        // SAFETY: the probe writes `err[0] = '\0'` before anything else, and
        // NUL-terminates within `err.len()` on every catch path.
        let message = || {
            unsafe { CStr::from_ptr(err.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        };

        match status {
            OK => {
                charges.truncate(n_charges);
                Outcome::Charges(charges)
            }
            CAUGHT_RUST_ERROR => Outcome::CaughtRustError(message()),
            CAUGHT_STD_EXCEPTION => Outcome::CaughtStdException(message()),
            CAUGHT_UNKNOWN => Outcome::CaughtUnknown(message()),
            NO_SELECTOR => Outcome::NoSelector,
            other => panic!("bridge_probe.cc returned an unknown status {other}"),
        }
    }

    // ── Molecules ────────────────────────────────────────────────────────────

    /// Build a frame the way a molrs user's molecule actually reaches the bridge:
    /// element, coordinates, formal charge, bond order, aromatic flag — never
    /// reference perceived types (offline golden).
    fn frame_of(
        elements: &[&str],
        xyz: &[[f64; 3]],
        formal_charges: &[i32],
        bonds: &[(usize, usize, f64, bool)],
    ) -> FrameRef {
        let mut mol = Atomistic::new();
        let ids: Vec<AtomId> = elements
            .iter()
            .zip(xyz)
            .map(|(el, [x, y, z])| mol.add_atom_xyz(el, *x, *y, *z))
            .collect();
        for (aid, fc) in ids.iter().zip(formal_charges) {
            mol.set_atom(*aid, "formal_charge", *fc)
                .expect("set formal_charge");
        }
        for (i, j, order, aromatic) in bonds {
            let bid = mol.add_bond(ids[*i], ids[*j]).expect("add bond");
            // The fixture states the two facts separately, as the model does:
            // an aromatic bond is `Aromatic` carrying the localized number the
            // oracle's own Kekulé structure gives it.
            let number = BondNumber::from_code(order.round() as u32);
            let class = if *aromatic {
                BondType::Aromatic
            } else {
                BondType::from_code(order.round() as u32)
            };
            mol.set_bond_class(bid, class, number)
                .expect("set bond class");
        }

        let inner = molrs_ffi::FrameRef::new_standalone();
        inner
            .with_mut(|frame| *frame = mol.to_frame())
            .expect("populate the frame");
        FrameRef(inner)
    }

    /// An oracle molecule, as a frame.
    fn frame_of_case(case: &ReferenceCase) -> FrameRef {
        frame_of(case.elements, case.xyz, case.formal_charges, case.bonds)
    }

    /// Tetrafluoroborate, `BF4⁻` — chemistry the BCC table does not cover.
    ///
    /// A real molecule from a real molcrafts system (the ionic-liquid
    /// electrolytes), not a synthetic pathology: boron simply has no correction
    /// row, which is the ordinary way a user's molecule falls outside AM1-BCC.
    /// The geometry is an idealized tetrahedron at the experimental B–F distance
    /// (1.39 Å); the BCC stage reads topology, so the coordinates only have to be
    /// chemically sane.
    fn tetrafluoroborate() -> FrameRef {
        const D: f64 = 0.8025; // 1.39 / sqrt(3)
        frame_of(
            &["B", "F", "F", "F", "F"],
            &[
                [0.0, 0.0, 0.0],
                [D, D, D],
                [-D, -D, D],
                [-D, D, -D],
                [D, -D, -D],
            ],
            &[-1, 0, 0, 0, 0],
            &[
                (0, 1, 1.0, false),
                (0, 2, 1.0, false),
                (0, 3, 1.0, false),
                (0, 4, 1.0, false),
            ],
        )
    }

    /// Compare against a reference column, atom by atom.
    fn assert_charges(what: &str, got: &[f64], want: &[f64]) {
        assert_eq!(
            got.len(),
            want.len(),
            "{what}: expected one charge per atom, got {}",
            got.len()
        );
        for (i, (got, want)) in got.iter().zip(want).enumerate() {
            assert!(
                (got - want).abs() <= CHARGE_TOL,
                "{what}: atom {i} is {got:.6} e, reference says {want:.6} e \
                 (tolerance {CHARGE_TOL:.0e})"
            );
        }
    }

    // ── ac-001: the error is catchable in C++, and the process lives ─────────

    /// Marker lines the child prints. Their absence is a failure, so a filter that
    /// matched no test cannot be mistaken for a pass.
    const CAUGHT_MARKER: &str = "MOLRS_CXXAPI_CAUGHT:";
    const SURVIVED_MARKER: &str = "MOLRS_CXXAPI_SURVIVED:";

    /// Set only in the child, so that a stray `cargo test -- --ignored` cannot run
    /// a call which (today) aborts the process.
    const CHILD_ENV: &str = "MOLRS_CXXAPI_BRIDGE_CHILD";

    /// Re-exec this test binary to run one `#[ignore]`d child test; give back how
    /// it died and what it printed.
    ///
    /// The child's name is spelled out because `--exact` needs its full path
    /// within the binary. If it is ever renamed, the child runs nothing, prints no
    /// marker, and the parent fails — which is the safe direction.
    fn run_child(test_name: &str) -> (ExitStatus, String) {
        let exe = std::env::current_exe().expect("the running test binary has a path");
        let out = Command::new(exe)
            .args([
                "--exact",
                test_name,
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD_ENV, "1")
            .output()
            .expect("re-exec the test binary");

        let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&out.stderr));
        (out.status, text)
    }

    /// Explain an exit status the way the criterion words it.
    fn report(status: &ExitStatus, output: &str) -> String {
        format!(
            "the child process did not survive the call\n  exit: {status}\n  \
             (`signal: 6 (SIGABRT)` is exactly the bug this criterion is about: the Rust panic \
             reached the cxx shim, which is `extern \"C\"`, and aborted the process instead of \
             becoming a rust::Error that C++ could catch)\n  child output:\n{}",
            indent(output)
        )
    }

    fn indent(text: &str) -> String {
        text.lines().map(|line| format!("    | {line}\n")).collect()
    }

    /// ac-001 — the core criterion. A molecule with no BCC parameter reaches C++
    /// as a `rust::Error`, and the process that asked is alive to catch it.
    #[test]
    fn a_missing_bcc_parameter_is_caught_in_cpp_as_a_rust_error() {
        let (status, output) = run_child("cxx::child_calls_the_bridge_with_boron");

        assert!(status.success(), "{}", report(&status, &output));
        assert!(
            output.contains(CAUGHT_MARKER),
            "the child never reported a caught rust::Error (marker '{CAUGHT_MARKER}' \
             absent)\n  exit: {status}\n  child output:\n{}",
            indent(&output)
        );
        assert!(
            output.contains("missing BCC correction"),
            "C++ caught an exception, but not the chemistry error molrs raises for boron \
             ('missing BCC correction for bond B|71|1')\n  child output:\n{}",
            indent(&output)
        );
    }

    /// ac-001 — and the catch leaves molrs usable: a chemistry error is an ordinary
    /// refusal, not a poisoned bridge.
    #[test]
    fn the_bridge_still_works_after_a_caught_chemistry_error() {
        let (status, output) = run_child("cxx::child_keeps_using_the_bridge_after_catching");

        assert!(status.success(), "{}", report(&status, &output));
        assert!(
            output.contains(SURVIVED_MARKER),
            "the child did not get charges back from a valid molecule after catching the boron \
             error (marker '{SURVIVED_MARKER}' absent)\n  exit: {status}\n  child output:\n{}",
            indent(&output)
        );
    }

    /// Child of [`a_missing_bcc_parameter_is_caught_in_cpp_as_a_rust_error`].
    #[test]
    #[ignore = "spawned as a child process by the ac-001 tests"]
    fn child_calls_the_bridge_with_boron() {
        if std::env::var_os(CHILD_ENV).is_none() {
            return; // not spawned by the parent: this call can abort the process
        }
        let mut bf4 = tetrafluoroborate();

        match assign_from_cpp(&mut bf4, &[0.0; 5], "bcc") {
            Outcome::CaughtRustError(msg) => println!("{CAUGHT_MARKER} {msg}"),
            other => panic!(
                "the bridge did not raise a catchable rust::Error for BF4-, it returned {other:?}"
            ),
        }
    }

    /// Child of [`the_bridge_still_works_after_a_caught_chemistry_error`].
    #[test]
    #[ignore = "spawned as a child process by the ac-001 tests"]
    fn child_keeps_using_the_bridge_after_catching() {
        if std::env::var_os(CHILD_ENV).is_none() {
            return;
        }
        let mut bf4 = tetrafluoroborate();
        let caught = assign_from_cpp(&mut bf4, &[0.0; 5], "bcc");
        assert!(
            matches!(caught, Outcome::CaughtRustError(_)),
            "expected a caught rust::Error for BF4-, got {caught:?}"
        );

        // Same process, same bridge, a molecule the table does cover.
        let water = case("water");
        let mut frame = frame_of_case(water);
        match assign_from_cpp(&mut frame, water.am1_charges, "bcc") {
            Outcome::Charges(q) => println!("{SURVIVED_MARKER} {q:?}"),
            other => panic!("the bridge was left unusable by the caught error: {other:?}"),
        }
    }

    // ── ac-002: ABCG2 is reachable from C++ ─────────────────────────────────

    /// ac-002 — selecting ABCG2 across the bridge yields ABCG2 charges.
    #[test]
    fn the_abcg2_selector_yields_abcg2_charges() {
        let water = case("water");
        let mut frame = frame_of_case(water);

        let got = match assign_from_cpp(&mut frame, water.am1_charges, "abcg2") {
            Outcome::Charges(q) => q,
            Outcome::NoSelector => panic!(
                "the bridge takes no parameter-set argument, so ABCG2 is unreachable from C++ \
                 (lib.rs hardcodes BccParameterSet::Bcc)"
            ),
            other => panic!("the bridge refused the 'abcg2' parameter set: {other:?}"),
        };

        assert_charges("abcg2", &got, water.abcg2_charges);
        assert_ne!(
            got, water.bcc_charges,
            "the bridge returned the BCC column for an 'abcg2' request — the selector is \
             being ignored"
        );
    }

    /// ac-002 — and the other family still works: BCC did not become ABCG2.
    #[test]
    fn the_bcc_selector_still_yields_bcc_charges() {
        let water = case("water");
        let mut frame = frame_of_case(water);

        match assign_from_cpp(&mut frame, water.am1_charges, "bcc") {
            Outcome::Charges(q) => assert_charges("bcc", &q, water.bcc_charges),
            Outcome::NoSelector => panic!(
                "the bridge takes no parameter-set argument (lib.rs hardcodes \
                 BccParameterSet::Bcc)"
            ),
            other => panic!("the bridge refused the 'bcc' parameter set: {other:?}"),
        }
    }

    /// ac-002 — the selector is a name molrs must recognize, and an unknown one is
    /// a user error like any other: a `rust::Error`, not an abort, and above all
    /// not a silent fallback to BCC.
    #[test]
    fn an_unknown_parameter_set_is_a_catchable_error() {
        let water = case("water");
        let mut frame = frame_of_case(water);

        match assign_from_cpp(&mut frame, water.am1_charges, "resp") {
            Outcome::CaughtRustError(_) => {}
            Outcome::Charges(q) => panic!(
                "the bridge silently accepted the unknown parameter set 'resp' and returned \
                 {q:?} — an unrecognized name must be refused, not defaulted"
            ),
            Outcome::NoSelector => panic!("the bridge takes no parameter-set argument"),
            other => panic!("expected a rust::Error for an unknown parameter set, got {other:?}"),
        }
    }

    /// ac-002 — the charges the selector produced are the ones written back into
    /// the frame, which is the copy Atomiverse actually reads.
    #[test]
    fn the_abcg2_charges_are_written_into_the_frame() {
        let water = case("water");
        let mut frame = frame_of_case(water);

        match assign_from_cpp(&mut frame, water.am1_charges, "abcg2") {
            Outcome::Charges(_) => {}
            other => panic!("the 'abcg2' request did not return charges: {other:?}"),
        }

        let written = frame
            .0
            .block("atoms")
            .expect("the frame keeps its atoms block")
            .copy_f(keys::CHARGE)
            .expect("read atoms.charge")
            .expect("the bridge writes atoms.charge")
            .0;

        assert_charges(
            "abcg2 (written into atoms.charge)",
            &written,
            water.abcg2_charges,
        );
    }
}
