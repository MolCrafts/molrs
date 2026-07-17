//! MMFF94 vs MMFF94s through the **typifier** — the two named front doors.
//!
//! Spec `mmff-typifier-split` (ac-001 … ac-005, ac-009). RED until
//! `MMFF94Typifier` / `MMFF94STypifier` exist and `frame_builder::annotate_mmff`
//! stops hardcoding `MmffVariant::Mmff94`.
//!
//! # Why an atom-type test would be VACUOUS
//!
//! `diff molrs/data/mmff94.xml molrs/data/mmff94s.xml` is 54 changed rows:
//! 1 force-field name + 11 `<Oop>` + 42 `<Torsion>`. The 95 atom types and every
//! `<Def>` / `<Prop>` / `<BCI>` / `<PBCI>` / bond / angle / stretch-bend / vdW row
//! is **byte-identical**. So a test that only compares atom types passes whether or
//! not the variant is ever threaded through. The money is in `koop` and `(v1,v2,v3)`.
//!
//! # The physics (kernel source, not hearsay)
//!
//! `molrs/src/ff/potential/improper/mmff.rs:1,46` — `E_oop = 0.5·143.9325·koop·χ²`,
//! χ = Wilson out-of-plane angle in **radians**, `koop` in md·Å·rad⁻². Hence
//! `koop > 0` ⇔ planar (χ=0) is an energy **minimum**; `koop < 0` ⇔ planar is a
//! **maximum**. MMFF94s ("s" = static) flattens delocalised trivalent nitrogen by
//! making `koop` positive.
//!
//! # Measured `koop`, straight out of the shipped tables
//!
//! Every differing `<Oop>` row is centred on MMFF numeric type **10** (`NC=O`,
//! amide N — 6 rows) or **40** (`NC=C`, enamine-type N — 5 rows):
//!
//! | Oop key (i_centre_k_l) | MMFF94 `koop` | MMFF94s `koop` |
//! |---|---|---|
//! | `0_10_0_0`   | −0.020 | +0.015 |
//! | `1_10_1_3`   | −0.020 | +0.015 |
//! | `1_10_3_6`   | −0.033 | +0.015 |
//! | `1_10_3_28`  | −0.020 | +0.015 |
//! | `3_10_3_28`  | −0.030 | +0.015 |
//! | `3_10_28_28` | −0.019 | +0.015 |
//! | `0_40_0_0`   | −0.005 | +0.030 |
//! | `1_40_28_37` | −0.006 | +0.030 |
//! | `2_40_28_28` | −0.007 | +0.030 |
//! | `3_40_28_28` | −0.007 | +0.030 |
//! | `28_40_28_37`| **+0.004** | +0.030 |
//!
//! **Honest boundary**: the last row is *already positive* under MMFF94 — and it is
//! exactly the row aniline's NH₂ resolves to (N type 40, peripherals 28/28/37). So
//! "MMFF94 koop is negative" is **false in general** and this file never asserts it.
//! What it asserts is the contract the acceptance file words: under MMFF94s the
//! centre carries `+0.015` (type 10) / `+0.030` (type 40), and the MMFF94 value on
//! the same improper differs. The per-fixture MMFF94 numbers below are *recorded
//! measurements* of today's (unchanged) MMFF94 path, so they double as the ac-006
//! "no MMFF94 value moved" guard.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use molrs::Atomistic;
use molrs::ff::forcefield::{ForceField, Params, StyleDefs};
use molrs::ff::typifier::Typifier;
use molrs::ff::typifier::mmff::{MMFF94STypifier, MMFF94Typifier};
use molrs::store::frame::Frame;
use molrs::system::molgraph::{Atom, PropValue};

// ---------------------------------------------------------------------------
// Fixtures (RDKit-generated; the same files the energy path validates against —
// `tests/ff/mmff/energy.rs`). Every energy fixture now carries both reference
// energies; the type-10/40 partition is computed below.
// ---------------------------------------------------------------------------

/// MMFF numeric types whose out-of-plane row is re-parameterised by MMFF94s.
const N_CENTRE_TYPES: [u32; 2] = [10, 40];

/// MMFF94s `koop` for a type-10 (`NC=O`) centre, md·Å·rad⁻².
const KOOP_S_TYPE_10: f64 = 0.015;
/// MMFF94s `koop` for a type-40 (`NC=C`) centre, md·Å·rad⁻².
const KOOP_S_TYPE_40: f64 = 0.030;

/// The nitrogen centre of each fixture: `(fixture, oop label, centre type,
/// MMFF94 koop AS MEASURED, number of N-centred Wilson rows)`.
///
/// The MMFF94 column is a **measurement of today's shipped path**, taken by running
/// the existing (pre-rename) typifier over these fixtures — not a value copied out of
/// a paper. It is pinned here so this file doubles as the ac-006 guard: this spec
/// re-plumbs the pipeline and must not move a single MMFF94 number.
///
/// Note `s_aniline`: **+0.004**, positive under MMFF94. MMFF94's "delocalised N is
/// pyramidal" is not a universal negative `koop`; for the `28_40_28_37` row it is a
/// small positive one that MMFF94s multiplies by 7.5. So the discriminating invariant
/// is *the value changes*, never *the sign flips*.
///
/// Each trigonal centre contributes three Wilson permutations sharing one `koop`
/// (`annotate_mmff` step 6), hence 3 rows per N — and 6 for urea's two nitrogens.
const N_CENTRE_ORACLE: [(&str, &str, u32, f64, usize); 4] = [
    ("s_acetamide", "3_10_28_28", 10, -0.019, 3),
    ("s_nmethylacetamide", "1_10_3_28", 10, -0.020, 3),
    ("s_aniline", "28_40_28_37", 40, 0.004, 3),
    ("s_urea", "3_10_28_28", 10, -0.019, 6),
];

/// Exact-match tolerance for a force constant read straight back out of a table:
/// no arithmetic happens between the XML row and the Frame column.
const EXACT: f64 = 1e-12;
/// Energy tolerance for "these two force fields are not the same force field".
const ENERGY_GAP: f64 = 1e-3;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ff/mmff/fixtures")
}

/// Every energy fixture on disk.  Partitions below are derived from the typed
/// molecule; no fixture name is allowed to decide which assertions it receives.
fn energy_fixtures() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(fixtures_dir())
        .expect("read fixtures dir")
        .map(|e| e.expect("dir entry").file_name())
        .filter_map(|f| {
            f.to_string_lossy()
                .strip_suffix(".energy.json")
                .map(str::to_owned)
        })
        .collect();
    names.sort();
    assert!(
        !names.is_empty(),
        "no MMFF energy fixtures found; a computed partition would be vacuous"
    );
    names
}

/// Compute the MMFF94/MMFF94s partition from numeric atom types 10/40.
fn fixture_partition() -> (Vec<String>, Vec<String>) {
    let (mut with_n, mut without_n) = (Vec::new(), Vec::new());
    for name in energy_fixtures() {
        let mol = load_sdf(&name);
        let typed = MMFF94Typifier::new()
            .typify(&mol)
            .unwrap_or_else(|e| panic!("{name}: MMFF94 typify: {e}"));
        let baked = dissect(&typed);
        if baked.types.iter().any(|t| N_CENTRE_TYPES.contains(t)) {
            with_n.push(name);
        } else {
            without_n.push(name);
        }
    }
    assert!(
        !with_n.is_empty() && !without_n.is_empty(),
        "computed type-10/40 partition is vacuous: {} with, {} without",
        with_n.len(),
        without_n.len()
    );
    (with_n, without_n)
}

/// Minimal V2000 SDF loader preserving atom order and bond orders (same shape as
/// `tests/ff/mmff/energy.rs::load_sdf`, which is a sibling *private* test module).
fn load_sdf(name: &str) -> Atomistic {
    let path = fixtures_dir().join(format!("{name}.sdf"));
    let text = std::fs::read_to_string(&path).expect("read sdf");
    let lines: Vec<&str> = text.lines().collect();
    let counts = lines[3];
    let n_atoms: usize = counts[0..3].trim().parse().expect("n_atoms");
    let n_bonds: usize = counts[3..6].trim().parse().expect("n_bonds");

    let mut g = Atomistic::new();
    let mut ids = Vec::with_capacity(n_atoms);
    for k in 0..n_atoms {
        let line = lines[4 + k];
        let x: f64 = line[0..10].trim().parse().expect("x");
        let y: f64 = line[10..20].trim().parse().expect("y");
        let z: f64 = line[20..30].trim().parse().expect("z");
        let element = line[31..34].trim().to_string();
        let mut atom = Atom::xyz(&element, x, y, z);
        atom.set("formal_charge", PropValue::Int(0));
        ids.push(g.add_atom(atom));
    }
    for k in 0..n_bonds {
        let line = lines[4 + n_atoms + k];
        let i: usize = line[0..3].trim().parse::<usize>().expect("bond i") - 1;
        let j: usize = line[3..6].trim().parse::<usize>().expect("bond j") - 1;
        let order: f64 = line[6..9].trim().parse().expect("bond order");
        let bid = g.add_bond(ids[i], ids[j]).expect("add bond");
        g.set_bond_prop(bid, "order", PropValue::F64(order))
            .expect("bond");
    }
    for line in &lines[4 + n_atoms + n_bonds..] {
        let t = line.trim_end();
        if t == "M  END" || t == "$$$$" {
            break;
        }
        if let Some(rest) = t.strip_prefix("M  CHG") {
            let toks: Vec<&str> = rest.split_whitespace().collect();
            if toks.is_empty() {
                continue;
            }
            let pairs: usize = toks[0].parse().unwrap_or(0);
            for q in 0..pairs {
                let ai: usize = toks[1 + 2 * q].parse::<usize>().expect("chg atom") - 1;
                let chg: i32 = toks[2 + 2 * q].parse().expect("chg val");
                g.set_atom(ids[ai], "formal_charge", PropValue::Int(chg))
                    .expect("atom");
            }
        }
    }
    g
}

// ---------------------------------------------------------------------------
// What the typifier BAKED (path 1) — read back off the labeled graph, keyed by
// atom-index tuple so the comparison never depends on iteration order.
// ---------------------------------------------------------------------------

/// The per-instance numbers the MMFF kernels actually read: `koop` on impropers
/// (`mmff_oop_ctor` reads the `"koop"` Frame column) and `(v1,v2,v3)` on dihedrals
/// (`mmff_torsion_ctor` reads the `"v1"/"v2"/"v3"` columns).
struct Baked {
    /// atom index → MMFF numeric type.
    types: Vec<u32>,
    /// improper (i, j=centre, k, l) → (`koop`, oop type label).
    koop: HashMap<[usize; 4], (f64, String)>,
    /// dihedral (i, j, k, l) → (v1, v2, v3).
    torsion: HashMap<[usize; 4], (f64, f64, f64)>,
    /// The materialised typed Frame (what a consumer compiles potentials from).
    frame: Frame,
}

impl Baked {
    /// Impropers whose CENTRE atom (`atomj`, per `annotate_mmff`) has one of
    /// `N_CENTRE_TYPES`.
    fn n_centred_impropers(&self) -> Vec<([usize; 4], f64, String, u32)> {
        let mut out: Vec<_> = self
            .koop
            .iter()
            .filter_map(|(idx, (koop, label))| {
                let centre_type = self.types[idx[1]];
                N_CENTRE_TYPES
                    .contains(&centre_type)
                    .then(|| (*idx, *koop, label.clone(), centre_type))
            })
            .collect();
        out.sort_by_key(|(idx, ..)| *idx);
        out
    }
}

/// Dissect a labeled `Atomistic` into the numbers the kernels consume.
fn dissect(labeled: &Atomistic) -> Baked {
    let index_of: HashMap<_, usize> = labeled
        .atoms()
        .enumerate()
        .map(|(i, (id, _))| (id, i))
        .collect();

    let mut types = vec![0u32; index_of.len()];
    for (id, atom) in labeled.atoms() {
        let PropValue::Str(t) = atom.get("type").expect("typifier baked an atom `type`") else {
            panic!("atom `type` is not a string");
        };
        types[index_of[&id]] = t.parse().expect("MMFF numeric type");
    }

    let f64_prop = |props: &HashMap<String, PropValue>, key: &str| -> f64 {
        props
            .get(key)
            .and_then(PropValue::as_f64)
            .unwrap_or_else(|| panic!("typifier did not bake `{key}`"))
    };

    let mut koop = HashMap::new();
    for (_, imp) in labeled.impropers() {
        let idx = [
            index_of[&imp.nodes[0]],
            index_of[&imp.nodes[1]],
            index_of[&imp.nodes[2]],
            index_of[&imp.nodes[3]],
        ];
        let PropValue::Str(label) = imp.props.get("type").expect("improper `type`") else {
            panic!("improper `type` is not a string");
        };
        koop.insert(idx, (f64_prop(&imp.props, "koop"), label.clone()));
    }

    let mut torsion = HashMap::new();
    for (_, dih) in labeled.dihedrals() {
        let idx = [
            index_of[&dih.nodes[0]],
            index_of[&dih.nodes[1]],
            index_of[&dih.nodes[2]],
            index_of[&dih.nodes[3]],
        ];
        torsion.insert(
            idx,
            (
                f64_prop(&dih.props, "v1"),
                f64_prop(&dih.props, "v2"),
                f64_prop(&dih.props, "v3"),
            ),
        );
    }

    Baked {
        types,
        koop,
        torsion,
        frame: labeled.to_frame(),
    }
}

/// `(MMFF94 baked, MMFF94s baked)` for one fixture — the two front doors, same input.
fn baked_pair(name: &str) -> (Baked, Baked) {
    let mol = load_sdf(name);
    let m94 = MMFF94Typifier::new()
        .typify(&mol)
        .unwrap_or_else(|e| panic!("{name}: MMFF94 typify: {e}"));
    let m94s = MMFF94STypifier::new()
        .typify(&mol)
        .unwrap_or_else(|e| panic!("{name}: MMFF94s typify: {e}"));
    (dissect(&m94), dissect(&m94s))
}

/// Total energy of a fixture through a front door: typify → Frame → to_potentials.
///
/// One macro, two doors: the two front doors are deliberately distinct concrete
/// types (the variant is picked by picking a type, never by a flag), so a helper
/// that erased the difference would be testing a surface molrs does not offer.
macro_rules! total_energy {
    ($fn_name:ident, $door:ty, $label:literal) => {
        fn $fn_name(name: &str) -> f64 {
            let mol = load_sdf(name);
            let t = <$door>::new();
            let mut frame = t.typify(&mol).expect("typify").to_frame();
            frame.insert("pairs", molrs::ff::potential::intramolecular_pairs(&frame));
            let coords = molrs::ff::potential::extract_coords(&frame).expect("coords");
            let pots = t
                .ff()
                .to_potentials(&frame)
                .unwrap_or_else(|e| panic!("{name}: {} to_potentials: {e}", $label));
            pots.calc_energy_forces(&coords).0
        }
    };
}

total_energy!(total_energy_94, MMFF94Typifier, "MMFF94");
total_energy!(total_energy_94s, MMFF94STypifier, "MMFF94s");

// ---------------------------------------------------------------------------
// ac-001 — the public surface is two named front doors
// ---------------------------------------------------------------------------

/// Both front doors construct from the embedded parameter set and each names its
/// own force field. `new()` is infallible: the XML is a compile-time constant.
#[test]
fn front_doors_name_their_own_force_field() {
    assert_eq!(MMFF94Typifier::new().ff().name, "MMFF94");
    assert_eq!(MMFF94STypifier::new().ff().name, "MMFF94s");
}

/// Both front doors also parse a **caller-supplied** XML — with the variant pinned by
/// the door, never by an argument.
///
/// The input is a caller's XML precisely because molrs ships none: chem-perceive-14
/// ac-001 compiled the three parameter sets into `ff::params` and deleted
/// `molrs::data::*_XML`, which this test used to feed back into `from_xml_str`. That
/// was never a real test of the entry point anyway — it re-parsed the crate's own
/// embedded text and could not tell the caller's bytes from the shipped ones. Here the
/// XML is unmistakably the caller's (`name="caller-supplied"`), so a door that quietly
/// ignored the argument and returned its embedded set would fail on the first line.
///
/// A `<ForceField>` structure literal, not sample data being round-tripped: the same
/// convention `forcefield::parse_generic_xml_then_compile_and_eval` documents.
///
/// The bond style is *declared*, not tabulated. mmff-orthogonal-02 deleted the
/// `<BondStretchParams>` reader along with the 4,065 type-def rows no kernel ever
/// read: `bond/mmff_bond` is a `ParamSource::PerInstance` style, so it carries no
/// rows and the typifier bakes `kb` / `r0` onto each bond. The assertion below is
/// unchanged — the caller's bytes must still win, and a door that ignored the
/// argument and returned its embedded set still fails on the first line.
#[test]
fn front_doors_accept_caller_supplied_xml() {
    let xml = r#"
        <ForceField name="caller-supplied">
          <AtomProperties>
            <Prop type="1" atno="6" crd="4" val="4" pilp="0" mltb="0" arom="0" linh="0" sbmb="0" />
          </AtomProperties>
          <BondStyle name="mmff_bond" />
        </ForceField>
    "#;

    let t94 = MMFF94Typifier::from_xml_str(xml).expect("MMFF94 door parses caller XML");
    let t94s = MMFF94STypifier::from_xml_str(xml).expect("MMFF94s door parses caller XML");

    // The caller's bytes won, through both doors — neither fell back to a shipped set.
    assert_eq!(t94.ff().name, "caller-supplied");
    assert_eq!(t94s.ff().name, "caller-supplied");
    assert!(
        !t94.ff().get_styles("bond").is_empty(),
        "caller's bond style"
    );
    assert!(
        !t94s.ff().get_styles("bond").is_empty(),
        "caller's bond style"
    );

    // One params reader behind both doors: same XML in, same typing metadata out.
    assert_eq!(t94.params().get_prop(1).expect("type 1").atno, 6);
    assert_eq!(t94s.params().get_prop(1).expect("type 1").atno, 6);
}

/// Both doors are `Typifier`s, so a consumer can hold either behind the trait.
#[test]
fn both_front_doors_implement_the_typifier_trait() {
    let mol = load_sdf("s_acetamide");
    let (t94, t94s) = (MMFF94Typifier::new(), MMFF94STypifier::new());
    let doors: [&dyn Typifier<Mol = Atomistic>; 2] = [&t94, &t94s];
    for door in doors {
        let labeled = door.typify(&mol).expect("typify via trait");
        assert!(
            labeled.n_impropers() > 0,
            "acetamide has trigonal centres; the trait object must reach the same engine"
        );
    }
}

// `both_front_doors_forward_the_classification_helpers` lived here. It asserted
// that both doors forward `typify_bond` / `typify_angle` / `typify_dihedral` and
// agree — `bond_arom == 1`, `angle == 2`, `dihedral == 1`.
//
// RDKit contradicts all three, and the test could not see it: it only checked that
// the two doors agreed WITH EACH OTHER, which is guaranteed by construction (one
// engine, one `classify.rs`) and which locked the wrong values in. An aromatic bond
// is bond type **0** (`getMMFFBondType` requires SINGLE; after aromaticity
// perception a ring bond is AROMATIC), so the angle built from two of them is 0 and
// the torsion across one is 0.
//
// mmff-orthogonal-02 deletes `classify.rs`, the three methods, and this test. The
// values are now asserted against RDKit's rules on real molecules, off the typed
// Frame: `tests/ff/typifier/mmff_labels.rs`.

// ---------------------------------------------------------------------------
// ac-003 — path 1: the `koop` baked into the Frame (what `mmff_oop_ctor` reads)
// ---------------------------------------------------------------------------

/// MMFF94s puts a POSITIVE `koop` on every delocalised trivalent nitrogen:
/// +0.015 (type 10, `NC=O`) / +0.030 (type 40, `NC=C`) — planar N is a minimum.
#[test]
fn mmff94s_bakes_positive_koop_on_every_n_centre() {
    for name in fixture_partition().0 {
        let (_, s) = baked_pair(&name);
        let centres = s.n_centred_impropers();
        assert!(
            !centres.is_empty(),
            "{name}: no type-10/40 nitrogen centre — the fixture cannot discriminate"
        );
        for (idx, koop, label, centre_type) in centres {
            let expected = match centre_type {
                10 => KOOP_S_TYPE_10,
                40 => KOOP_S_TYPE_40,
                other => unreachable!("centre type {other}"),
            };
            assert!(
                koop > 0.0,
                "{name}: MMFF94s koop must be positive (planar N = minimum), \
                 got {koop} on {label} at {idx:?}"
            );
            assert!(
                (koop - expected).abs() < EXACT,
                "{name}: MMFF94s koop on type-{centre_type} centre {label} at {idx:?}: \
                 expected {expected}, got {koop}"
            );
        }
    }
}

/// The same improper carries a DIFFERENT `koop` under MMFF94.
///
/// Deliberately NOT "MMFF94 koop is negative": aniline's N (type 40, peripherals
/// 28/28/37) resolves to Oop row `28_40_28_37`, whose MMFF94 `koop` is **+0.004** —
/// positive already. The invariant is *difference*, not sign.
#[test]
fn mmff94_bakes_a_different_koop_on_every_n_centre() {
    for name in fixture_partition().0 {
        let (base, s) = baked_pair(&name);
        for (idx, koop_s, label, centre_type) in s.n_centred_impropers() {
            let (koop_94, label_94) = base
                .koop
                .get(&idx)
                .unwrap_or_else(|| panic!("{name}: MMFF94 has no improper at {idx:?}"));
            assert_eq!(
                *label_94, label,
                "{name}: the two variants must agree on the oop type LABEL at {idx:?} \
                 (types are identical between MMFF94 and MMFF94s)"
            );
            assert!(
                (koop_94 - koop_s).abs() > EXACT,
                "{name}: type-{centre_type} centre {label} at {idx:?} carries the SAME koop \
                 ({koop_94}) under both variants — MMFF94s is not reaching the Frame"
            );
        }
    }
}

/// The measured table, pinned: on each fixture the nitrogen centre resolves to a
/// known Oop row, MMFF94 keeps the `koop` it has today, and MMFF94s carries the
/// re-parameterised one.
///
/// This is the strongest form of ac-003: it fails if MMFF94s does not reach the
/// Frame (the bug), AND it fails if MMFF94 quietly moved (the ac-006 regression).
#[test]
fn n_centre_koop_matches_the_measured_table() {
    for (name, label, centre_type, koop_94, n_rows) in N_CENTRE_ORACLE {
        let (base, s) = baked_pair(name);
        let centres = base.n_centred_impropers();
        assert_eq!(
            centres.len(),
            n_rows,
            "{name}: expected {n_rows} Wilson rows centred on a type-10/40 N, got {}",
            centres.len()
        );
        let koop_s = if centre_type == 10 {
            KOOP_S_TYPE_10
        } else {
            KOOP_S_TYPE_40
        };
        for (idx, koop, got_label, got_type) in centres {
            assert_eq!(got_type, centre_type, "{name}: centre type at {idx:?}");
            assert_eq!(got_label, label, "{name}: oop type label at {idx:?}");
            assert!(
                (koop - koop_94).abs() < EXACT,
                "{name}: MMFF94 koop on {label} at {idx:?} moved: expected the measured \
                 {koop_94}, got {koop} — this spec re-plumbs the pipeline, it does not \
                 re-parameterise MMFF94"
            );
            let (koop_s_got, _) = s.koop[&idx];
            assert!(
                (koop_s_got - koop_s).abs() < EXACT,
                "{name}: MMFF94s koop on {label} at {idx:?}: expected {koop_s}, got \
                 {koop_s_got}"
            );
        }
    }
}

/// Non-vacuity of ac-003: every improper whose centre is NOT a type-10/40 nitrogen
/// (e.g. the carbonyl carbon of acetamide, the aromatic carbons of aniline) keeps a
/// `koop` identical between the two variants. MMFF94s re-parameterises 11 Oop rows,
/// not the whole table.
#[test]
fn non_nitrogen_oop_centres_are_untouched_by_mmff94s() {
    for name in fixture_partition().0 {
        let (base, s) = baked_pair(&name);
        let mut compared = 0usize;
        for (idx, (koop_94, label)) in &base.koop {
            if N_CENTRE_TYPES.contains(&base.types[idx[1]]) {
                continue;
            }
            let (koop_s, _) = s
                .koop
                .get(idx)
                .unwrap_or_else(|| panic!("{name}: MMFF94s has no improper at {idx:?}"));
            assert_eq!(
                koop_94.to_bits(),
                koop_s.to_bits(),
                "{name}: non-nitrogen centre {label} at {idx:?} changed koop \
                 ({koop_94} -> {koop_s}); MMFF94s must only touch type-10/40 N"
            );
            compared += 1;
        }
        assert!(
            compared > 0,
            "{name}: no non-nitrogen oop centre compared — this guard would be vacuous"
        );
    }
}

/// The two variants agree on every atom type — stated as an assertion so that the
/// vacuity of an atom-type-only test is recorded in the suite, not just in prose.
#[test]
fn atom_types_are_identical_between_variants() {
    for name in energy_fixtures() {
        let (base, s) = baked_pair(&name);
        assert_eq!(
            base.types, s.types,
            "{name}: atom types must be identical — the MMFF94s difference is Oop/Torsion only"
        );
    }
}

// ---------------------------------------------------------------------------
// ac-003 — path 1: the torsion coefficients baked into the Frame
// ---------------------------------------------------------------------------

/// 42 `<Torsion>` rows differ between the two parameter sets, all around the same
/// nitrogens. At least one dihedral of N-methylacetamide must therefore carry
/// different `(v1, v2, v3)`.
#[test]
fn mmff94s_bakes_different_torsion_coefficients() {
    let name = "s_nmethylacetamide";
    let (base, s) = baked_pair(name);
    let differing: Vec<_> = base
        .torsion
        .iter()
        .filter_map(|(idx, v94)| {
            let v94s = s.torsion.get(idx).expect("same dihedral set");
            let diff = (v94.0 - v94s.0).abs() + (v94.1 - v94s.1).abs() + (v94.2 - v94s.2).abs();
            (diff > EXACT).then_some((*idx, *v94, *v94s))
        })
        .collect();
    assert!(
        !differing.is_empty(),
        "{name}: every dihedral carries identical (v1,v2,v3) under both variants — \
         the MMFF94s torsion table is not reaching the Frame"
    );
}

// ---------------------------------------------------------------------------
// ac-004 — path 2: the ForceField tree (`to_potentials` compiles from this)
// ---------------------------------------------------------------------------

/// `(category, style, type row name)` → parameters, flattened out of a ForceField.
fn type_rows(ff: &ForceField) -> HashMap<(String, String, String), Vec<(String, f64)>> {
    let mut rows = HashMap::new();
    for style in ff.styles() {
        if matches!(style.defs, StyleDefs::KSpace) {
            continue;
        }
        for (name, params) in style.defs.collect_type_params() {
            rows.insert(
                (style.category().to_owned(), style.name.clone(), name),
                sorted_params(&params),
            );
        }
    }
    rows
}

fn sorted_params(params: &Params) -> Vec<(String, f64)> {
    let mut p: Vec<(String, f64)> = params.iter().map(|(k, v)| (k.to_owned(), v)).collect();
    p.sort_by(|a, b| a.0.cmp(&b.0));
    p
}

/// The MMFF94 → MMFF94s delta in the compiled tree is now **the name alone**, and
/// every remaining type row is identical.
///
/// This test used to assert "the name, 11 differing improper rows, 42 differing
/// dihedral rows". Those rows are gone: mmff-orthogonal-02 deleted the 4,065
/// type-defs that no kernel read, and `improper/mmff_oop` / `dihedral/mmff_torsion`
/// are `ParamSource::PerInstance` styles carrying zero rows (asserted in
/// `tests/ff/potential/param_source.rs`). Only `pair/mmff_vdw`'s 95 rows survive,
/// and MMFF94s does not touch vdW.
///
/// **The 11 + 42 claim did not disappear — it moved to where it was always true.**
/// The variant difference never entered an energy through these rows; it enters
/// through the `koop` and `(v1, v2, v3)` the typifier resolves and bakes into Frame
/// columns, which the kernels actually read. That is what
/// `mmff94s_bakes_positive_koop_on_every_n_centre`,
/// `n_centre_koop_matches_the_measured_table` and
/// `mmff94s_bakes_different_torsion_coefficients` assert — on molecules — and what
/// `mmff94s_total_energy_matches_rdkit` (`tests/ff/mmff/energy.rs`) checks against
/// RDKit's own MMFF94s oracle.
#[test]
fn forcefield_tree_differs_only_in_name() {
    let ff94 = MMFF94Typifier::new();
    let ff94s = MMFF94STypifier::new();
    assert_ne!(ff94.ff().name, ff94s.ff().name);

    let a = type_rows(ff94.ff());
    let b = type_rows(ff94s.ff());

    let keys_a: HashSet<_> = a.keys().cloned().collect();
    let keys_b: HashSet<_> = b.keys().cloned().collect();
    assert_eq!(
        keys_a, keys_b,
        "the two parameter sets must define the SAME type rows — only values move"
    );

    // Non-vacuity: the surviving table is vdW's, and it is real.
    assert!(
        !a.is_empty(),
        "no type rows at all in the MMFF tree — `pair/mmff_vdw` must still carry its \
         95 per-atom-type rows, or this comparison proves nothing"
    );

    let differing: HashMap<String, Vec<String>> = a
        .iter()
        .filter(|(key, params_a)| **params_a != b[*key])
        .fold(HashMap::new(), |mut acc, (key, _)| {
            acc.entry(key.0.clone()).or_default().push(key.2.clone());
            acc
        });

    assert!(
        differing.is_empty(),
        "every surviving type row must be identical between MMFF94 and MMFF94s — the \
         only rows left are vdW's, and MMFF94s re-parameterises out-of-plane and \
         torsion, not van der Waals. Differing: {differing:?}"
    );
}

// ---------------------------------------------------------------------------
// ac-005 — end to end, with the identity guard that makes it non-vacuous
// ---------------------------------------------------------------------------

/// Both front doors compile potentials, and the totals are different force fields.
#[test]
fn total_energies_differ_on_every_nitrogen_fixture() {
    for name in fixture_partition().0 {
        let e94 = total_energy_94(&name);
        let e94s = total_energy_94s(&name);
        assert!(e94.is_finite(), "{name}: MMFF94 energy not finite: {e94}");
        assert!(
            e94s.is_finite(),
            "{name}: MMFF94s energy not finite: {e94s}"
        );
        assert!(
            (e94 - e94s).abs() > ENERGY_GAP,
            "{name}: MMFF94 ({e94}) and MMFF94s ({e94s}) total energies agree to \
             {ENERGY_GAP} kcal/mol — MMFF94STypifier is silently emitting MMFF94"
        );
    }
}

/// Regression for the hole the former hand-written partition created: caffeine
/// (and `e_big`) both carry a type-10/40 centre and must therefore enter the
/// computed difference half.  Caffeine's energy is asserted directly because it
/// was the omitted molecule that made the old green result misleading.
#[test]
fn computed_nitrogen_partition_catches_caffeine() {
    let (with_n, _) = fixture_partition();
    assert!(
        with_n.iter().any(|name| name == "e_caffeine"),
        "typed caffeine did not enter the type-10/40 partition"
    );
    assert!(
        with_n.iter().any(|name| name == "e_big"),
        "typed e_big did not enter the type-10/40 partition"
    );
    let e94 = total_energy_94("e_caffeine");
    let e94s = total_energy_94s("e_caffeine");
    assert!(
        (e94 - e94s).abs() > ENERGY_GAP,
        "caffeine has a delocalised nitrogen but MMFF94 ({e94}) and MMFF94s \
         ({e94s}) do not differ"
    );
}

/// THE IDENTITY GUARD. Without it, "the two differ" could pass on a difference
/// that isn't real. `e_ethane` has no trivalent nitrogen at all; `e_benzene` HAS
/// out-of-plane centres (six type-37 carbons) but no type-10/40 nitrogen — so on
/// both molecules the two front doors must be bit-for-bit indistinguishable.
#[test]
fn typed_frames_are_bit_identical_without_a_delocalised_nitrogen() {
    for name in fixture_partition().1 {
        let (base, s) = baked_pair(&name);
        assert!(
            base.types.iter().all(|t| !N_CENTRE_TYPES.contains(t)),
            "{name}: fixture contains a type-10/40 nitrogen — it cannot serve as the guard"
        );
        assert_frames_bit_identical(&base.frame, &s.frame, &name);
    }
}

/// The guard is only worth something if benzene actually exercises the Oop kernel.
#[test]
fn benzene_guard_is_not_vacuous_it_has_out_of_plane_centres() {
    let (base, s) = baked_pair("e_benzene");
    assert!(
        !base.koop.is_empty(),
        "e_benzene has no impropers — the identity guard would prove nothing about koop"
    );
    assert_eq!(base.koop.len(), s.koop.len());
}

/// Same molecules, same total energy, to the bit.
#[test]
fn total_energies_are_bit_identical_without_a_delocalised_nitrogen() {
    for name in fixture_partition().1 {
        let e94 = total_energy_94(&name);
        let e94s = total_energy_94s(&name);
        assert_eq!(
            e94.to_bits(),
            e94s.to_bits(),
            "{name}: MMFF94 ({e94}) and MMFF94s ({e94s}) must be bit-for-bit equal — \
             this molecule has no type-10/40 nitrogen, so nothing may differ"
        );
    }
}

// ---------------------------------------------------------------------------
// ac-001 / ac-002 / ac-009 — structural gates.
//
// These are properties of what the source may CONTAIN, and they are invisible to
// a behavioural test: a `frame_builder` that hardcodes `MmffVariant::Mmff94` still
// returns a perfectly well-formed typed Frame — just the wrong one. The hardcode
// is the whole bug this spec closes, so it gets a gate that cannot come back.
//
// Scanner discipline is the file's own (see `source_gate`): line-comments stripped,
// biased towards false negatives — stripping text can remove a match, never invent
// one.
// ---------------------------------------------------------------------------

/// ac-002 — `frame_builder.rs` names no variant of its own.
///
/// Today it names `MmffVariant::Mmff94` three times (`MmffMolProperties::compute`,
/// `eparams::torsion_params`, `eparams::oop_koop`), which is exactly why
/// `MMFF94STypifier` cannot exist yet. Matching is exact: `MmffVariant::Mmff94s`
/// is a different literal and is not counted (it is a prefix trap, not a hit).
#[test]
fn frame_builder_hardcodes_no_variant() {
    let (path, src) = super::source_gate::ff_source("typifier/mmff/frame_builder.rs");
    let hits: Vec<&str> = src
        .lines()
        .filter(|line| names_variant_mmff94(line))
        .collect();
    assert!(
        hits.is_empty(),
        "{}: `MmffVariant::Mmff94` is hardcoded {} time(s) — MMFF94s can never reach \
         the Frame while it is:\n{}",
        path.display(),
        hits.len(),
        hits.join("\n")
    );
}

/// `MmffVariant::Mmff94` as an exact path — `MmffVariant::Mmff94s` does not count.
fn names_variant_mmff94(line: &str) -> bool {
    const NEEDLE: &str = "MmffVariant::Mmff94";
    let mut from = 0usize;
    while let Some(at) = line[from..].find(NEEDLE) {
        let end = from + at + NEEDLE.len();
        let next = line[end..].chars().next();
        if !matches!(next, Some(c) if c.is_alphanumeric() || c == '_') {
            return true;
        }
        from = end;
    }
    false
}

/// ac-002 — the variant reaches `annotate_mmff` as a parameter instead.
#[test]
fn annotate_mmff_takes_the_variant_as_a_parameter() {
    let (path, src) = super::source_gate::ff_source("typifier/mmff/frame_builder.rs");
    let at = src
        .find("fn annotate_mmff")
        .unwrap_or_else(|| panic!("{}: no `annotate_mmff`", path.display()));
    let signature = signature_at(&src[at..]);
    assert!(
        signature.contains("MmffVariant"),
        "{}: `annotate_mmff` must receive the variant (it forwards it to \
         MmffMolProperties::compute / torsion_params / oop_koop), got:\n{signature}",
        path.display()
    );
}

/// ac-001 — no `pub fn` under `src/ff/typifier/mmff/` leaks `MmffVariant`.
///
/// The variant is the engine's private field. A `pub fn` that takes one IS the
/// flag-style constructor the owner ruled out, whatever it is called.
#[test]
fn no_public_signature_under_typifier_mmff_names_the_variant() {
    let mut leaks = Vec::new();
    for (path, src) in super::source_gate::typifier_sources() {
        if !path.to_string_lossy().contains("typifier/mmff") {
            continue;
        }
        let mut from = 0usize;
        while let Some(at) = src[from..].find("pub fn ") {
            let start = from + at;
            let signature = signature_at(&src[start..]);
            if signature.contains("MmffVariant") {
                leaks.push(format!("{}: {}", path.display(), signature.trim()));
            }
            from = start + "pub fn ".len();
        }
    }
    assert!(
        leaks.is_empty(),
        "the public MMFF typifier surface must carry no variant flag — users get two \
         named front doors, never `MMFFTypifier::new(variant)`:\n{}",
        leaks.join("\n")
    );
}

/// A function signature: everything from `fn` up to the opening brace (or `;`).
fn signature_at(src: &str) -> String {
    let end = src.find('{').or_else(|| src.find(';')).unwrap_or(src.len());
    src[..end].to_owned()
}

/// ac-008 (structural half) — the `MmffVariant` doc no longer claims the `_S`
/// tables are missing.
///
/// `ff/mmff/mod.rs` says the MMFF94s Oop/Tor tables are "not yet present in
/// [`tables`]". They ARE: `MMFF_OOP_S` (117 rows) and `MMFF_TOR_S` (926 rows), and the
/// energy path has been reading them all along. That comment predates the tables and
/// has already misled one reader (this spec was drafted believing MMFF94s was the
/// gap), so it is a defect with a test, not a typo.
#[test]
fn the_stale_mmff94s_table_comment_is_gone() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ff/mmff/mod.rs");
    let src = std::fs::read_to_string(&path).expect("read ff/mmff/mod.rs");
    assert!(
        !src.contains("not yet present in"),
        "{}: the MmffVariant doc still claims the MMFF94s Oop/Tor tables are absent — \
         MMFF_OOP_S and MMFF_TOR_S are shipped and in use",
        path.display()
    );
}

/// ac-001 — the old name is gone from the whole Rust surface. A clean break: no
/// alias, no `#[deprecated]` shim (the owner's ruling; the repo is pre-1.0).
#[test]
fn the_name_mmfftypifier_is_gone_from_src() {
    let mut hits = Vec::new();
    for dir in src_trees() {
        for (path, src) in rust_sources(&dir) {
            for (n, line) in src.lines().enumerate() {
                // `MMFF94Typifier` / `MMFF94STypifier` do not contain `MMFFTypifier`,
                // so a plain substring search cannot false-positive on the new names.
                if line.contains("MMFFTypifier") {
                    hits.push(format!("{}:{}: {}", path.display(), n + 1, line.trim()));
                }
            }
        }
    }
    assert!(
        hits.is_empty(),
        "`MMFFTypifier` must be renamed to `MMFF94Typifier` everywhere (no compat alias):\n{}",
        hits.join("\n")
    );
}

// ac-009 — "the MMFF94s data has a reader in `src/`" lived here as a grep for
// `MMFF94S_XML`. That const is deleted by chem-perceive-14 ac-001 (the parameter sets
// are compiled tables now, and molrs ships no XML), so the grep would pass by asserting
// the absence of the very thing it was written to find.
//
// The criterion outlived its wording, and it is now enforced generically and
// permanently, one layer up: `ff::tables_gate::no_parameter_table_is_unreachable`
// requires EVERY table under `ff/params/` — mmff94s among them — to be declared and
// named from production code outside `ff/params/`. That gate is what held this spec
// behind `mmff-typifier-split` in the first place: it refuses a reader-less table.

fn molrs_src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// The Rust surfaces the rename must cover: the crate, and the PyO3 binder when it
/// is present (it is a sibling workspace, absent from a packaged `.crate`; the
/// Python side is additionally covered by `test_mmff94s_typifier.py`).
fn src_trees() -> Vec<PathBuf> {
    let mut trees = vec![molrs_src_dir()];
    let binder = Path::new(env!("CARGO_MANIFEST_DIR")).join("../molrs-python/src");
    if binder.is_dir() {
        trees.push(binder);
    }
    trees
}

/// Every `.rs` under `dir`, raw (comments included — a stale doc naming the old
/// type is a hit, because the rename is a documented breaking change).
fn rust_sources(dir: &Path) -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    collect_rs(dir, &mut out);
    assert!(
        !out.is_empty(),
        "no .rs under {} — the gate would pass vacuously",
        dir.display()
    );
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn collect_rs(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
    for entry in std::fs::read_dir(dir).expect("read a source tree") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let src = std::fs::read_to_string(&path).expect("read a source file");
            out.push((path, src));
        }
    }
}

/// Every column of every block, compared by bits (floats included).
fn assert_frames_bit_identical(a: &Frame, b: &Frame, what: &str) {
    let mut blocks_a: Vec<&str> = a.keys().collect();
    let mut blocks_b: Vec<&str> = b.keys().collect();
    blocks_a.sort_unstable();
    blocks_b.sort_unstable();
    assert_eq!(blocks_a, blocks_b, "{what}: block sets differ");

    for block in blocks_a {
        let (ba, bb) = (a.get(block).expect("block"), b.get(block).expect("block"));
        let mut cols_a: Vec<&str> = ba.keys().collect();
        let mut cols_b: Vec<&str> = bb.keys().collect();
        cols_a.sort_unstable();
        cols_b.sort_unstable();
        assert_eq!(cols_a, cols_b, "{what}/{block}: column sets differ");
        assert_eq!(ba.nrows(), bb.nrows(), "{what}/{block}: row counts differ");

        for col in cols_a {
            let at = format!("{what}/{block}/{col}");
            if let (Some(x), Some(y)) = (ba.get_float(col), bb.get_float(col)) {
                let bits_x: Vec<u64> = x.iter().map(|v| v.to_bits()).collect();
                let bits_y: Vec<u64> = y.iter().map(|v| v.to_bits()).collect();
                assert_eq!(bits_x, bits_y, "{at}: float column differs bit-for-bit");
            } else if let (Some(x), Some(y)) = (ba.get_int(col), bb.get_int(col)) {
                assert_eq!(x, y, "{at}: int column differs");
            } else if let (Some(x), Some(y)) = (ba.get_uint(col), bb.get_uint(col)) {
                assert_eq!(x, y, "{at}: uint column differs");
            } else if let (Some(x), Some(y)) = (ba.get_bool(col), bb.get_bool(col)) {
                assert_eq!(x, y, "{at}: bool column differs");
            } else if let (Some(x), Some(y)) = (ba.get_u8(col), bb.get_u8(col)) {
                assert_eq!(x, y, "{at}: u8 column differs");
            } else if let (Some(x), Some(y)) = (ba.get_string(col), bb.get_string(col)) {
                assert_eq!(x, y, "{at}: string column differs");
            } else {
                panic!("{at}: column type mismatch between the two typed frames");
            }
        }
    }
}
