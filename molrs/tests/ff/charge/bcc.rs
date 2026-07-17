//! `BccModel` — the push API, and the type pollution it ends.
//!
//! `correct(&mol, &am1) -> Result<Vec<f64>, ChargeError>` is a pure function: the
//! molecule goes in behind a shared reference, the AM1 base charges go in as a
//! slice, the corrected charges come back. There is no backend to fake, no trait
//! object in the way, and — the point of ac-004 — nothing is written back into the
//! caller's molecule.
//!
//! What that last clause is worth: the old path typed the molecule with
//! `ATOMTYPE_BCC.DEF` and then did `*mol = typed`, so a caller who asked for AM1-BCC
//! charges got their GAFF atom types REPLACED by BCC codes (`"c3"` -> `"11"`). The
//! standard AM1-BCC workflow needs both at once — GAFF for LJ and bonded terms, BCC
//! for the charges — so that silently destroyed the force field it was supposed to
//! be completing. The C++ path never got bitten only because it ran on a throwaway
//! `Atomistic` and copied the charge column back out, which is a caller working
//! around a broken contract, not a contract that works.
//!
//! The tests below say it three ways, because `&Atomistic` alone only proves molrs
//! cannot write to the molecule — not that it does not *read* what is there:
//!
//! 1. the atom and bond `type` columns are unchanged afterwards (the mutation);
//! 2. the charges are bit-identical whether the molecule carries GAFF types, hostile
//!    LAMMPS bond-type ids, or nothing at all (the read);
//! 3. no BCC code ever appears in `keys::TYPE` (the symptom a user would see).

use molrs::ff::charge::{BccModel, BccParameterSet, ChargeError};
use molrs::ff::typifier::Typifier;
use molrs::ff::typifier::atd::{AtdParameterSet, AtdTypifier};
use molrs::store::keys;
use molrs::system::molgraph::PropValue;
use molrs::{AtomId, Atomistic};

use crate::typifier::antechamber_oracle::{AntechamberCase, CASES};
use crate::typifier::oracle_mol::build_case;

fn bcc() -> BccModel {
    BccModel::new(BccParameterSet::Bcc).expect("build the BCC charge model")
}

fn case(name: &str) -> &'static AntechamberCase {
    CASES
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("{name} in the oracle"))
}

fn methane() -> (Atomistic, AtomId, Vec<AtomId>) {
    let mut mol = Atomistic::new();
    let c = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
    let mut hs = Vec::new();
    for [x, y, z] in [
        [0.63, 0.63, 0.63],
        [-0.63, -0.63, 0.63],
        [-0.63, 0.63, -0.63],
        [0.63, -0.63, -0.63],
    ] {
        let h = mol.add_atom_xyz("H", x, y, z);
        mol.add_bond(c, h).expect("add C-H");
        hs.push(h);
    }
    (mol, c, hs)
}

fn methanol() -> Atomistic {
    let mut mol = Atomistic::new();
    let c = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
    let o = mol.add_atom_xyz("O", 1.43, 0.0, 0.0);
    let h1 = mol.add_atom_xyz("H", -0.63, 0.63, 0.63);
    let h2 = mol.add_atom_xyz("H", -0.63, -0.63, 0.63);
    let h3 = mol.add_atom_xyz("H", -0.63, 0.0, -0.89);
    let ho = mol.add_atom_xyz("H", 1.76, 0.89, 0.0);
    for (a, b) in [(c, o), (c, h1), (c, h2), (c, h3), (o, ho)] {
        mol.add_bond(a, b).expect("add bond");
    }
    mol
}

fn chloromethane() -> Atomistic {
    let mut mol = Atomistic::new();
    let c = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
    let cl = mol.add_atom_xyz("Cl", 1.78, 0.0, 0.0);
    let h1 = mol.add_atom_xyz("H", -0.63, 0.63, 0.63);
    let h2 = mol.add_atom_xyz("H", -0.63, -0.63, 0.63);
    let h3 = mol.add_atom_xyz("H", -0.63, 0.0, -0.89);
    for (a, b) in [(c, cl), (c, h1), (c, h2), (c, h3)] {
        mol.add_bond(a, b).expect("add bond");
    }
    mol
}

/// Trimethylborane, B(CH3)3 — a molecule the BCC table cannot correct.
///
/// Every C–H bond it has is an ordinary `11|91|1` row. The B–C bonds are not:
/// `ATOMTYPE_BCC.DEF` types boron as `"B"` (the element fallback rule), and
/// `BCCPARM.DAT` has no `"B"` row at all — AM1-BCC simply never parameterized
/// boron. So this is the molecule that reaches the correction stage cleanly typed
/// and then finds no row, which is the ONLY way to see `MissingCorrection` come out
/// of a model built from a real, fully-populated parameter set.
fn trimethylborane() -> (Atomistic, Vec<AtomId>) {
    let mut mol = Atomistic::new();
    let b = mol.add_atom_xyz("B", 0.0, 0.0, 0.0);
    let mut ids = vec![b];
    for (dx, dy) in [(1.56, 0.0), (-0.78, 1.35), (-0.78, -1.35)] {
        let c = mol.add_atom_xyz("C", dx, dy, 0.0);
        let bid = mol.add_bond(b, c).expect("add B-C");
        mol.set_bond_prop(bid, keys::ORDER, 1.0).expect("set order");
        ids.push(c);
        for [hx, hy, hz] in [
            [0.4 * dx, 0.4 * dy, 1.03],
            [0.4 * dx, 0.4 * dy, -1.03],
            [1.6 * dx, 1.6 * dy, 0.0],
        ] {
            let h = mol.add_atom_xyz("H", hx, hy, hz);
            let bid = mol.add_bond(c, h).expect("add C-H");
            mol.set_bond_prop(bid, keys::ORDER, 1.0).expect("set order");
            ids.push(h);
        }
    }
    (mol, ids)
}

/// The `type` label of every atom, in atom order.
fn atom_types(mol: &Atomistic, ids: &[AtomId]) -> Vec<Option<String>> {
    ids.iter()
        .map(|aid| {
            mol.get_atom(*aid)
                .expect("atom")
                .get_str(keys::TYPE)
                .map(str::to_owned)
        })
        .collect()
}

/// The `type` prop of every bond, in bond order.
fn bond_types(mol: &Atomistic) -> Vec<Option<PropValue>> {
    mol.bonds()
        .map(|(_, b)| b.props.get(keys::TYPE).cloned())
        .collect()
}

// ── the correction table, applied through the model ──────────────────────────

/// The embedded BCCPARM row for a C(11)–H(91) single bond is +0.0393 e.
///
/// Fed a base of zero, methane's carbon is the `left` of four such bonds and each
/// hydrogen the `right` of one, so the corrections alone must come out
/// +4 × 0.0393 = +0.1572 on C and −0.0393 on each H.
///
/// The molecule handed in is UNTYPED — the model perceives the BCC types itself,
/// which is precisely what makes it a model rather than a table.
#[test]
fn correct_applies_the_embedded_methane_row() {
    let (mol, c, hs) = methane();
    let q = bcc()
        .correct(&mol, &[0.0; 5])
        .expect("correct an untyped methane");

    let ids: Vec<AtomId> = mol.atoms().map(|(aid, _)| aid).collect();
    let index = |a: AtomId| ids.iter().position(|x| *x == a).expect("atom");
    assert!((q[index(c)] - 0.1572).abs() < 1e-12, "{:?}", q[index(c)]);
    for h in hs {
        assert!((q[index(h)] + 0.0393).abs() < 1e-12, "{:?}", q[index(h)]);
    }
}

/// The three hand-checked AM1-BCC vectors, through the push API.
///
/// Same molecules, same base charges and same expected results as when this was
/// spelled with a `FakeAM1Backend`; only the seam changed. They are kept next to
/// the 37-molecule oracle because they are readable by eye — if the oracle ever
/// regenerates into nonsense, these three still say what methanol must do.
#[test]
fn correct_reproduces_reference_am1bcc_vectors() {
    let cases = [
        (
            methane().0,
            vec![-0.266000, 0.066000, 0.066000, 0.066000, 0.066000],
            vec![-0.108800, 0.026700, 0.026700, 0.026700, 0.026700],
        ),
        (
            methanol(),
            vec![-0.073000, -0.326000, 0.068000, 0.068000, 0.068000, 0.195000],
            vec![0.116700, -0.598800, 0.028700, 0.028700, 0.028700, 0.396000],
        ),
        (
            chloromethane(),
            vec![-0.177000, -0.117000, 0.098000, 0.098000, 0.098000],
            vec![0.014300, -0.190400, 0.058700, 0.058700, 0.058700],
        ),
    ];

    for (mol, am1, expected) in cases {
        let got = bcc().correct(&mol, &am1).expect("apply BCC corrections");
        assert_eq!(got.len(), expected.len());
        for (q, q_ref) in got.iter().zip(expected.iter()) {
            assert!((q - q_ref).abs() < 5.0e-7, "{q} != {q_ref}");
        }
    }
}

// ── ac-004: charge assignment never clobbers force-field atom types ──────────

/// A GAFF-typed molecule comes back with its GAFF types untouched.
///
/// The molecule is typed with the real `ATOMTYPE_GFF.DEF` engine, so it carries the
/// exact labels the oracle says antechamber gives methanol — `c3`, `oh`, `h1`, `ho`
/// — and those labels are compared byte-for-byte before and after. The old pipeline
/// overwrote them with `11`, `31`, `91`.
#[test]
fn correct_leaves_gaff_atom_types_byte_identical() {
    let case = case("methanol");
    let (mol, ids) = build_case(case);
    let gaff = AtdTypifier::new(AtdParameterSet::Gff)
        .typify(&mol)
        .expect("GAFF-type methanol");

    // The premise: this molecule really is carrying force-field atom types.
    let before = atom_types(&gaff, &ids);
    let want: Vec<Option<String>> = case
        .gff_atom_types
        .iter()
        .map(|t| Some((*t).to_owned()))
        .collect();
    assert_eq!(before, want, "the GAFF typifier did not produce GAFF types");

    let before_bonds = bond_types(&gaff);
    let q = bcc()
        .correct(&gaff, case.am1_charges)
        .expect("correct a GAFF-typed molecule");

    assert_eq!(
        atom_types(&gaff, &ids),
        before,
        "charge assignment overwrote the force-field atom types — the AM1-BCC \
         workflow needs GAFF types and BCC charges AT THE SAME TIME"
    );
    assert_eq!(
        bond_types(&gaff),
        before_bonds,
        "charge assignment overwrote the bond types"
    );
    // And it still produced antechamber's charges while doing so.
    for (k, (got, w)) in q.iter().zip(case.bcc_charges).enumerate() {
        assert!((got - w).abs() < 1.0e-4, "atom {k}: {got:+.4} vs {w:+.4}");
    }
}

/// No BCC code ever appears in `keys::TYPE`.
///
/// The symptom, stated as a user would see it. BCC codes are numeric strings
/// (`"11"`, `"31"`, `"91"`); GAFF types never are. If a single atom's `type` came
/// back all-digits, a force field somewhere just lost its atom types.
#[test]
fn bcc_codes_never_appear_in_the_type_column() {
    for case in CASES {
        let (mol, ids) = build_case(case);
        let gaff = AtdTypifier::new(AtdParameterSet::Gff)
            .typify(&mol)
            .unwrap_or_else(|e| panic!("{}: GAFF typing: {e}", case.name));

        let _ = bcc()
            .correct(&gaff, case.am1_charges)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));

        for (k, ty) in atom_types(&gaff, &ids).iter().enumerate() {
            let ty = ty.as_deref().expect("every atom is typed");
            assert!(
                !ty.chars().all(|c| c.is_ascii_digit()),
                "{}: atom {k} carries `{ty}`, a BCC code, in keys::TYPE — the \
                 charge model wrote its internal atom types into the caller's \
                 force field (antechamber's GAFF type here is `{}`)",
                case.name,
                case.gff_atom_types[k]
            );
        }
    }
}

/// The model does not READ `keys::TYPE` either — it perceives its own BCC types.
///
/// The sharp half of ac-004, and the one a `&Atomistic` signature cannot give you
/// for free. Two molecules, identical but for the atom types they carry: one bare,
/// one labelled `c3` / `oh` / `h1` / `ho`. If the model looked its corrections up by
/// the `type` column it was handed, the GAFF-typed one would either error (there is
/// no `c3|oh` row in BCCPARM.DAT) or silently correct through different rows. It
/// must instead produce the SAME BITS, because the types it uses are its own.
#[test]
fn correct_ignores_the_atom_types_the_molecule_arrives_with() {
    for case in CASES {
        let (mol, _) = build_case(case);
        let gaff = AtdTypifier::new(AtdParameterSet::Gff)
            .typify(&mol)
            .unwrap_or_else(|e| panic!("{}: GAFF typing: {e}", case.name));

        let bare = bcc()
            .correct(&mol, case.am1_charges)
            .unwrap_or_else(|e| panic!("{} (untyped): {e}", case.name));
        let typed = bcc()
            .correct(&gaff, case.am1_charges)
            .unwrap_or_else(|e| panic!("{} (GAFF-typed): {e}", case.name));

        for (k, (a, b)) in bare.iter().zip(&typed).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{}: atom {k} corrected to {a} without atom types and {b} with GAFF \
                 ones — the charge model is reading keys::TYPE instead of perceiving \
                 its own BCC types",
                case.name
            );
        }
    }
}

/// Nor does it read the bond `type` column.
///
/// Acetate is the molecule that makes this bite. Its two C–O bonds must be perceived
/// as antechamber bond type **9** (delocalized) — that is what makes the carboxylate
/// oxygens equivalent. Here the molecule arrives carrying LAMMPS bond-type ids
/// (1, 2, 3, ...) in the very same `keys::TYPE` prop the BCC bond type lives in. A
/// model that trusted them would look up single-bond rows, split the two oxygens by
/// ~0.2 e, and never say a word.
#[test]
fn correct_ignores_the_bond_types_the_molecule_arrives_with() {
    let case = case("acetate");
    let (mol, _) = build_case(case);

    let mut lammps = mol.clone();
    let bond_ids: Vec<_> = lammps.bonds().map(|(bid, _)| bid).collect();
    for (n, bid) in bond_ids.iter().enumerate() {
        lammps
            .set_bond_prop(*bid, keys::TYPE, (n % 3 + 1) as i32)
            .expect("set a LAMMPS bond type id");
    }
    let before = bond_types(&lammps);

    let clean = bcc()
        .correct(&mol, case.am1_charges)
        .expect("correct acetate");
    let hostile = bcc()
        .correct(&lammps, case.am1_charges)
        .expect("correct acetate carrying LAMMPS bond type ids");

    for (k, (a, b)) in clean.iter().zip(&hostile).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "atom {k}: {a} vs {b} — the model re-interpreted the caller's LAMMPS \
             bond-type ids as BCC bond types instead of perceiving its own"
        );
    }
    assert_eq!(
        bond_types(&lammps),
        before,
        "the caller's LAMMPS bond-type ids were overwritten"
    );
    // The delocalized perception is what all of that protects: both oxygens equal.
    assert_eq!(
        hostile[2].to_bits(),
        hostile[3].to_bits(),
        "acetate's two carboxylate oxygens came out different — bond type 9 was \
         not perceived"
    );
}

// ── ac-005 (bit-exact half): no charge is ever shifted to reach the net ──────

/// A lone ion keeps its AM1 charge — to the bit.
///
/// Sodium types cleanly (`ATOMTYPE_BCC.DEF` gives it `"Na"`) and has no bonds, so
/// AM1-BCC has exactly nothing to do: the corrected charge IS the AM1 charge. That
/// makes it the sharpest possible probe for the renormalizer that used to live in
/// `normalize_total_charge` — handed +0.997 on an ion of formal charge +1, a
/// renormalizing model returns +1.0 (a 0.003 e lie, and a plausible-looking one),
/// while a correct one returns the same 0.997 bits it was given.
#[test]
fn a_lone_ion_is_never_renormalized_to_its_formal_charge() {
    let mut mol = Atomistic::new();
    let na = mol.add_atom_xyz("Na", 0.0, 0.0, 0.0);
    mol.set_atom(na, "formal_charge", 1).expect("formal charge");

    let am1 = [0.997_f64];
    let q = bcc().correct(&mol, &am1).expect("correct a lone Na+");

    assert_eq!(q.len(), 1);
    assert_eq!(
        q[0].to_bits(),
        am1[0].to_bits(),
        "a bondless atom has no BCC correction, so its charge must come back \
         untouched — got {} from {} (a renormalizer would return exactly 1.0)",
        q[0],
        am1[0]
    );
}

// ── typed errors: C++ and Python must be able to tell these apart ────────────

/// A bond with no row in the parameter set is `MissingCorrection`, and it names
/// the bond.
///
/// Not a `String`, and not a silent zero. The twin of
/// `typifier::am1bcc::missing_bcc_correction_row_is_an_error_not_zero`, which pins
/// the same failure at the corrector level with a synthetic table; this one pins it
/// on the real, fully-populated `BCCPARM.DAT` — the table a user actually gets —
/// and pins that the error is *typed*, which is what the C++ and Python bridges have
/// to discriminate on.
#[test]
fn a_bond_with_no_correction_row_is_a_typed_missing_correction() {
    let (mol, ids) = trimethylborane();
    let am1 = vec![0.0; ids.len()];

    let err = bcc()
        .correct(&mol, &am1)
        .expect_err("BCCPARM.DAT has no boron row; correcting B(CH3)3 must fail");

    match err {
        ChargeError::MissingCorrection {
            left,
            right,
            bond_type,
        } => {
            let pair = [left.as_str(), right.as_str()];
            assert!(
                pair.contains(&"B"),
                "the error must name the boron bond it could not correct, got \
                 {pair:?}|{bond_type}"
            );
            assert!(
                pair.contains(&"11"),
                "the other end of the uncorrectable bond is the sp3 carbon `11`, \
                 got {pair:?}|{bond_type}"
            );
            assert_eq!(bond_type, 1, "B–C is a single bond");
        }
        other => panic!("an uncorrectable bond must be a typed MissingCorrection, got {other:?}"),
    }
}

/// An atom the table cannot type is `MissingAtomType` — never a defaulted type.
///
/// A bare sulfur atom is the one element `ATOMTYPE_BCC.DEF` leaves unmatched at zero
/// connections (its rules are S with 1, 2, 3 or 4 bonds), which is why the
/// element-coverage test in the typifier tree skips z=16. Charge assignment must
/// surface that as a typed error and not, say, shrug and pass the AM1 charge back:
/// an atom the BCC table cannot even name is an atom whose BCC charge is unknown.
#[test]
fn an_untypeable_atom_is_a_typed_missing_atom_type() {
    let mut mol = Atomistic::new();
    mol.add_atom_xyz("S", 0.0, 0.0, 0.0);

    let err = bcc()
        .correct(&mol, &[0.0])
        .expect_err("ATOMTYPE_BCC.DEF cannot type a bare sulfur atom");

    assert!(
        matches!(err, ChargeError::MissingAtomType { .. }),
        "an untypeable atom must be a typed MissingAtomType, got {err:?}"
    );
}

/// The wrong number of AM1 charges is `ChargeCountMismatch`, with both counts.
///
/// The push API's one new failure mode: the charges arrive as a slice, so they can
/// disagree with the molecule. It must be caught at the boundary and reported with
/// the numbers, because the caller (a C++ bridge marshalling an AM1 result) is the
/// only one who can fix it.
#[test]
fn a_charge_count_mismatch_is_typed_and_carries_both_counts() {
    let (mol, _, _) = methane();

    let err = bcc()
        .correct(&mol, &[0.0; 3])
        .expect_err("three charges for a five-atom molecule");

    match err {
        ChargeError::ChargeCountMismatch { expected, got } => {
            assert_eq!(expected, 5, "methane has five atoms");
            assert_eq!(got, 3, "three charges were supplied");
        }
        other => panic!("expected a typed ChargeCountMismatch, got {other:?}"),
    }
}
