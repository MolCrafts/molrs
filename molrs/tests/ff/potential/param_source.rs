//! ac-003 + ac-004 — what `ParamSource` changes, and what it must NOT change.
//!
//! These two criteria are a matched pair, and they are only meaningful together:
//!
//! * **ac-003** — the six MMFF per-instance styles carry **zero** type-def rows
//!   and still compile to a `Potential`. That is the whole point: today they are
//!   handed 4,065 rows of XML (`<Bond>` 493, `<Angle>` 2342, `<StretchBend>` 282,
//!   `<Torsion>` 926, `<Oop>` 117) whose *only* job is to satisfy
//!   `Style::to_potential`'s `type_params.is_empty()` guard. No kernel reads a
//!   single one of them.
//!
//! * **ac-004** — a `TypeRows` style with no type definitions **still errors**.
//!   The relaxation must apply only to per-instance styles. The lazy fix — delete
//!   the guard, or widen the `category != "pair"` escape hatch — would make every
//!   empty parameter table in molrs compile to a silently-zero potential, which is
//!   a strictly worse bug than the one being fixed.
//!
//! And the boundary between them: **`pair/mmff_vdw` keeps its 95 rows.** MMFF's
//! van der Waals parameters genuinely ARE a per-atom-type table (`mmff_vdw_ctor`
//! opens by building a `HashMap<&str, &Params>` from `tp`). A deletion that swept
//! vdW in with the other five would take out a table that is actually read.

use molrs::ff::forcefield::{ForceField, SpecialBonds};
use molrs::ff::potential::intramolecular_pairs;
use molrs::ff::typifier::mmff::{MMFF94STypifier, MMFF94Typifier};
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::system::molgraph::PropValue;
use molrs::types::{F, U};
use molrs::{AtomId, Atomistic};
use ndarray::Array1;

/// The six styles whose parameters the typifier bakes into Frame columns.
///
/// Every one of them is a `_tp`-ignoring kernel (see `param_source_gate.rs`), so
/// every one of them must carry zero type-def rows once `ParamSource` exists.
const PER_INSTANCE_STYLES: [(&str, &str); 6] = [
    ("bond", "mmff_bond"),
    ("angle", "mmff_angle"),
    ("angle", "mmff_stbn"),
    ("dihedral", "mmff_torsion"),
    ("improper", "mmff_oop"),
    ("pair", "mmff_ele"),
];

/// The one MMFF style that is a genuine type-row lookup, and its exact row count.
const VDW_STYLE: (&str, &str) = ("pair", "mmff_vdw");
const VDW_TYPE_ROWS: usize = 95;

fn bond(mol: &mut Atomistic, a: AtomId, b: AtomId, order: F) {
    let bid = mol.add_bond(a, b).expect("add bond");
    mol.set_bond_prop(bid, "order", PropValue::F64(order))
        .expect("set bond order");
}

/// Ethylene (C2H4), planar, at a plausible geometry.
///
/// Chosen because it is the smallest molecule that exercises **all six**
/// per-instance styles at once: two trigonal centres (so `improper/mmff_oop` has
/// rows), a C=C (so `dihedral/mmff_torsion` has rows), and four H···H 1-4 pairs
/// (so the pair styles have a non-empty neighbour list). Ethane would silently
/// skip the impropers — `Style::to_potential` returns `Ok(None)` for a style with
/// no topology of its kind, which looks exactly like success.
fn ethylene() -> Atomistic {
    let mut mol = Atomistic::new();
    let atoms = [
        ("C", [0.000, 0.000, 0.0]),
        ("C", [1.330, 0.000, 0.0]),
        ("H", [-0.570, 0.925, 0.0]),
        ("H", [-0.570, -0.925, 0.0]),
        ("H", [1.900, 0.925, 0.0]),
        ("H", [1.900, -0.925, 0.0]),
    ];
    let ids: Vec<AtomId> = atoms
        .iter()
        .map(|(s, [x, y, z])| mol.add_atom_xyz(s, *x, *y, *z))
        .collect();
    bond(&mut mol, ids[0], ids[1], 2.0);
    bond(&mut mol, ids[0], ids[2], 1.0);
    bond(&mut mol, ids[0], ids[3], 1.0);
    bond(&mut mol, ids[1], ids[4], 1.0);
    bond(&mut mol, ids[1], ids[5], 1.0);
    mol
}

/// A typed ethylene Frame, complete with the consumer-built neighbour list —
/// i.e. exactly what `ForceField::to_potentials` is handed on the generic route.
fn typed_ethylene_frame() -> Frame {
    let typed = MMFF94Typifier::new()
        .typify(&ethylene())
        .expect("typify ethylene");
    let mut frame = typed.to_frame();
    let pairs = intramolecular_pairs(&frame);
    frame.insert("pairs", pairs);
    frame
}

/// Flat `[x0,y0,z0, ...]` coordinates from a Frame's atoms block.
fn coords_of(frame: &Frame) -> Vec<F> {
    let atoms = frame.get("atoms").expect("atoms block");
    let (x, y, z) = (
        atoms.get_float("x").expect("x"),
        atoms.get_float("y").expect("y"),
        atoms.get_float("z").expect("z"),
    );
    (0..x.len())
        .flat_map(|i| [x[i], y[i], z[i]])
        .collect::<Vec<F>>()
}

// ---------------------------------------------------------------------------
// ac-003 — zero type-def rows on the six per-instance styles
// ---------------------------------------------------------------------------

/// The six per-instance styles carry **no** type definitions.
///
/// RED today by 4,065 rows: MMFF94 ships 493 `<Bond>`, 2,342 `<Angle>`, 282
/// `<StretchBend>`, 926 `<Torsion>` and 117 `<Oop>` type-def rows — 438 KB of XML,
/// parsed on every `MMFF94Typifier::new()`, read by nothing.
#[test]
fn the_six_per_instance_mmff_styles_carry_zero_type_def_rows() {
    let mut fails = Vec::new();
    for (label, ff) in [
        ("MMFF94", MMFF94Typifier::new().ff().clone()),
        ("MMFF94s", MMFF94STypifier::new().ff().clone()),
    ] {
        for (category, name) in PER_INSTANCE_STYLES {
            let style = ff
                .styles()
                .iter()
                .find(|s| s.category() == category && s.name == name)
                .unwrap_or_else(|| {
                    panic!("{label}: no `{category}/{name}` style — mmff-orthogonal-01 defines all seven")
                });
            let rows = style.defs.collect_type_params().len();
            if rows != 0 {
                fails.push(format!(
                    "{label}: {category}/{name} carries {rows} type-def rows, want 0 — its \
                     kernel ignores `tp` entirely (parameters are baked into Frame columns by \
                     the typifier), so every one of those rows is dead data kept alive only to \
                     satisfy the empty-type-params guard"
                ));
            }
        }
    }
    assert!(
        fails.is_empty(),
        "per-instance styles still carry type-def rows:\n  {}",
        fails.join("\n  ")
    );
}

/// ...and they still compile to a `Potential` with zero rows.
///
/// The reverse assertion, and the one that proves the guard was actually taught
/// to consult `ParamSource` rather than simply deleted. A style with no rows must
/// build — not error, and not quietly vanish into `Ok(None)`.
#[test]
fn the_six_per_instance_mmff_styles_still_compile_to_a_potential() {
    let typifier = MMFF94Typifier::new();
    let ff = typifier.ff();
    let frame = typed_ethylene_frame();
    let coords = coords_of(&frame);

    let mut fails = Vec::new();
    for (category, name) in PER_INSTANCE_STYLES {
        let style = ff
            .styles()
            .iter()
            .find(|s| s.category() == category && s.name == name)
            .unwrap_or_else(|| panic!("no `{category}/{name}` style"));

        match style.to_potential(&frame, ff.special_bonds()) {
            Err(e) => fails.push(format!("{category}/{name}: to_potential failed: {e}")),
            Ok(None) => fails.push(format!(
                "{category}/{name}: compiled to NOTHING. Ethylene has bonds, angles, dihedrals, \
                 two trigonal centres and four 1-4 pairs, so every one of the six styles has \
                 topology to act on — `Ok(None)` here means the style silently contributes zero \
                 energy, which is how the electrostatic hole stayed invisible"
            )),
            Ok(Some(pot)) => {
                let (e, forces) = pot.calc_energy_forces(&coords);
                if !e.is_finite() || forces.iter().any(|f| !f.is_finite()) {
                    fails.push(format!(
                        "{category}/{name}: non-finite energy/forces (E={e})"
                    ));
                }
            }
        }
    }
    assert!(
        fails.is_empty(),
        "per-instance styles must compile from zero type rows:\n  {}",
        fails.join("\n  ")
    );
}

/// `pair/mmff_vdw` keeps **exactly 95** type-def rows.
///
/// vdW genuinely IS a per-atom-type lookup — one row per MMFF atom type, and
/// `mmff_vdw_ctor` reads them (`let type_map: HashMap<&str, &Params> = tp.iter()
/// .copied().collect();`). This is the assertion that stops the deletion from
/// over-reaching: 95 is not 0, and it is not 94 either.
#[test]
fn the_vdw_style_keeps_its_95_type_rows() {
    let (category, name) = VDW_STYLE;
    for (label, ff) in [
        ("MMFF94", MMFF94Typifier::new().ff().clone()),
        ("MMFF94s", MMFF94STypifier::new().ff().clone()),
    ] {
        let style = ff
            .styles()
            .iter()
            .find(|s| s.category() == category && s.name == name)
            .unwrap_or_else(|| panic!("{label}: no `{category}/{name}` style"));
        let rows = style.defs.collect_type_params().len();
        assert_eq!(
            rows, VDW_TYPE_ROWS,
            "{label}: {category}/{name} has {rows} type rows, want {VDW_TYPE_ROWS}. MMFF's van \
             der Waals parameters are a real per-atom-type table (95 types, 95 rows) and the \
             kernel really does resolve from `tp` — this style is NOT per-instance and must not \
             be swept into the deletion"
        );
    }
}

// ---------------------------------------------------------------------------
// ac-004 — the guard survives for everyone else
// ---------------------------------------------------------------------------

/// A minimal Frame with one bond — enough topology that `to_potential` reaches
/// the type-params guard instead of short-circuiting on an empty `bonds` block.
fn one_bond_frame() -> Frame {
    let mut atoms = Block::new();
    for (key, vals) in [
        ("x", [0.0, 1.0]),
        ("y", [0.0, 0.0]),
        ("z", [0.0, 0.0]),
        ("charge", [0.0, 0.0]),
    ] {
        atoms
            .insert(key, Array1::from_vec(vals.to_vec()).into_dyn())
            .expect("insert atom column");
    }
    atoms
        .insert(
            "type",
            Array1::from_vec(vec!["a".to_string(), "a".to_string()]).into_dyn(),
        )
        .expect("insert atom type");

    let mut bonds = Block::new();
    bonds
        .insert("atomi", Array1::from_vec(vec![0 as U]).into_dyn())
        .expect("atomi");
    bonds
        .insert("atomj", Array1::from_vec(vec![1 as U]).into_dyn())
        .expect("atomj");
    bonds
        .insert("type", Array1::from_vec(vec!["a-a".to_string()]).into_dyn())
        .expect("bond type");

    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame.insert("bonds", bonds);
    frame
}

/// A `TypeRows` bonded style with an empty type-def list still returns `Err`.
///
/// GREEN today, and it must **stay** green. This is the criterion that stops the
/// `ParamSource` relaxation from being implemented as "delete the guard": a
/// `bond/harmonic` style with no `<BondType>` rows resolves `k` and `r0` for
/// nothing, and must say so rather than quietly evaluating to zero.
#[test]
fn a_type_rows_bonded_style_with_no_type_defs_still_errors() {
    let mut ff = ForceField::new("empty");
    ff.def_bondstyle("harmonic");
    let style = ff
        .styles()
        .iter()
        .find(|s| s.category() == "bond")
        .expect("the bond style just defined");

    let Err(err) = style.to_potential(&one_bond_frame(), &SpecialBonds::default()) else {
        panic!(
            "a `bond/harmonic` style with zero type definitions must NOT compile. `harmonic` is \
             a TypeRows style — `bond_harmonic_ctor` resolves `k` / `r0` from its type rows — so \
             with no rows it can resolve nothing. If ParamSource's relaxation reached this \
             style, every empty parameter table in molrs now compiles to a silently-zero \
             potential"
        );
    };

    assert!(
        err.contains("no type definitions"),
        "expected the empty-type-definitions error, got: {err:?}"
    );
}

/// The same guard, on every bonded category — angle, dihedral, improper.
///
/// One category proving the point is one category; the guard is a property of
/// `Style::to_potential`, and the relaxation is a single `if`. If it is widened
/// wrongly, it is widened for all of them at once.
#[test]
fn every_type_rows_bonded_category_still_errors_without_type_defs() {
    let mut fails = Vec::new();
    for (category, style_name) in [
        ("bond", "harmonic"),
        ("angle", "harmonic"),
        ("dihedral", "opls"),
        ("improper", "harmonic"),
    ] {
        let mut ff = ForceField::new("empty");
        match category {
            "bond" => {
                ff.def_bondstyle(style_name);
            }
            "angle" => {
                ff.def_anglestyle(style_name);
            }
            "dihedral" => {
                ff.def_dihedralstyle(style_name);
            }
            _ => {
                ff.def_improperstyle(style_name);
            }
        }
        let style = &ff.styles()[0];
        // `to_potential` short-circuits to `Ok(None)` when the frame carries no
        // topology of the style's kind, so each category needs its own one-row
        // block — otherwise the guard is never reached and the test proves nothing.
        let mut frame = one_bond_frame();
        if category != "bond" {
            let mut block = Block::new();
            let cols: &[&str] = match category {
                "angle" => &["atomi", "atomj", "atomk"],
                "dihedral" | "improper" => &["atomi", "atomj", "atomk", "atoml"],
                _ => unreachable!(),
            };
            for (n, col) in cols.iter().enumerate() {
                block
                    .insert(*col, Array1::from_vec(vec![n as U]).into_dyn())
                    .expect("index column");
            }
            block
                .insert("type", Array1::from_vec(vec!["t".to_string()]).into_dyn())
                .expect("type column");
            let key = match category {
                "angle" => "angles",
                "dihedral" => "dihedrals",
                _ => "impropers",
            };
            frame.insert(key, block);
        }

        match style.to_potential(&frame, &SpecialBonds::default()) {
            Err(e) if e.contains("no type definitions") => {}
            Err(e) => fails.push(format!("{category}/{style_name}: wrong error: {e}")),
            Ok(_) => fails.push(format!(
                "{category}/{style_name}: compiled with ZERO type definitions. It is a TypeRows \
                 style; its kernel resolves every parameter from rows that do not exist"
            )),
        }
    }
    assert!(
        fails.is_empty(),
        "the empty-type-definitions guard was relaxed for TypeRows styles:\n  {}",
        fails.join("\n  ")
    );
}
