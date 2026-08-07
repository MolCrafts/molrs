//! OPLS-AA / GROMACS force-field XML reader.
//!
//! Parses the OpenMM-style OPLS-AA XML (as bundled with molpy, GROMACS units —
//! nm, kJ/mol, Ryckaert–Bellemans torsions) into a molrs [`ForceField`] in molrs
//! units (Å, kcal/mol, radians, e). The schema:
//!
//! ```xml
//! <ForceField name="OPLS-AA" combining_rule="geometric">
//!   <AtomTypes>
//!     <Type name="opls_001" class="opls_001" element="C" mass="12.011"/>
//!   </AtomTypes>
//!   <HarmonicBondForce>
//!     <Bond class1="OW" class2="HW" length="0.09572" k="502080.0"/>   <!-- nm, kJ/mol/nm² -->
//!   </HarmonicBondForce>
//!   <HarmonicAngleForce>
//!     <Angle class1="HW" class2="OW" class3="HW" angle="1.911" k="627.6"/>  <!-- rad, kJ/mol/rad² -->
//!   </HarmonicAngleForce>
//!   <RBTorsionForce>
//!     <Proper class1="Br" class2="C" class3="CT" class4="HC"
//!             c0="0.75" c1="2.26" c2="0.0" c3="-3.01" c4="0.0" c5="0.0"/>  <!-- kJ/mol, RB -->
//!   </RBTorsionForce>
//!   <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
//!     <Atom type="opls_001" charge="0.5" sigma="0.375" epsilon="0.43932"/> <!-- e, nm, kJ/mol -->
//!   </NonbondedForce>
//! </ForceField>
//! ```
//!
//! # Naming vocabularies
//!
//! Bonded forces key on the **class** attribute (chemical classes like `CT`,
//! `HC`), while nonbonded/atom definitions key on the **type** attribute
//! (`opls_NNN`). These are distinct vocabularies in the source file; the reader
//! transcribes both faithfully into separate styles. Reconciling class ↔ type
//! per atom is the typifier's job (a later sink), not the reader's.
//!
//! # Units
//!
//! - length nm → Å (× 10): bond `length` → `r0`, pair `sigma`.
//! - energy kJ/mol → kcal/mol (÷ 4.184): pair `epsilon`, dihedral coefficients.
//! - bond `k` kJ/mol/nm² → kcal/mol/Å² (÷ 4.184 ÷ 100). molrs and GROMACS both
//!   use the `½k(r−r₀)²` form, so no extra ½ factor (unlike a LAMMPS target).
//! - angle `k` kJ/mol/rad² → kcal/mol/rad² (÷ 4.184); `angle` already in radians.
//! - RB `c0..c5` → OPLS 4-cosine `f1..f4` via the private `rb_to_opls` helper
//!   (GROMACS Eqs. 200–201), in kcal/mol — matching the `dihedral:opls` kernel.
//! - charge `e`, mass `amu`: unchanged.

use roxmltree::Node;

use super::ForceFieldReader;
use crate::ff::constants::VACUUM_DIELECTRIC;
use crate::ff::forcefield::{ForceField, SpecialBonds};
use molrs::units::constants::COULOMB_REAL;

/// kJ/mol → kcal/mol.
const KJ_PER_KCAL: f64 = 4.184;
/// nm → Å.
const NM_TO_ANGSTROM: f64 = 10.0;

/// Reader for OPLS-AA / GROMACS XML (nm, kJ/mol, RB torsions).
#[derive(Debug, Default, Clone, Copy)]
pub struct OplsXmlReader;

impl OplsXmlReader {
    pub fn new() -> Self {
        Self
    }
}

impl ForceFieldReader for OplsXmlReader {
    fn read_str(&self, text: &str) -> Result<ForceField, String> {
        let doc =
            roxmltree::Document::parse(text).map_err(|e| format!("OPLS XML parse error: {}", e))?;
        let root = doc.root_element();
        if root.tag_name().name() != "ForceField" {
            return Err(format!(
                "root element must be <ForceField>, got <{}>",
                root.tag_name().name()
            ));
        }

        // molpy / foyer convention: missing ``name`` → ``"Unknown"`` (not a
        // hard-coded pack brand). Named packs (oplsaa) still set it on the root.
        let mut ff = ForceField::new(root.attribute("name").unwrap_or("Unknown"));

        // Two-pass for atom data: AtomTypes carries mass + class/element/def,
        // NonbondedForce carries charge/sigma/epsilon, both keyed by type name.
        let mut atom_rows: Vec<AtomTypeRow> = Vec::new();
        let mut nonbonded: Vec<NonbondedRow> = Vec::new();
        let mut coulomb14 = 0.5_f64;
        let mut lj14 = 0.5_f64;

        for sec in root.children().filter(Node::is_element) {
            match sec.tag_name().name() {
                "AtomTypes" => {
                    for t in sec.children().filter(Node::is_element) {
                        require_tag(&t, "Type")?;
                        let name = require_str(&t, "name")?.to_owned();
                        let mass = opt_f64(&t, "mass")?.unwrap_or(0.0);
                        atom_rows.push(AtomTypeRow {
                            name,
                            mass,
                            class: t.attribute("class").map(str::to_owned),
                            element: t.attribute("element").map(str::to_owned),
                            def: t.attribute("def").map(str::to_owned),
                            desc: t.attribute("desc").map(str::to_owned),
                            overrides: t.attribute("overrides").map(str::to_owned),
                        });
                    }
                }
                "HarmonicBondForce" => parse_bonds(&mut ff, &sec)?,
                "HarmonicAngleForce" => parse_angles(&mut ff, &sec)?,
                "RBTorsionForce" => parse_dihedrals(&mut ff, &sec)?,
                // CL&P / foyer: Fourier coeffs c0..c3 in kJ/mol under this tag.
                "PeriodicTorsionForce" => parse_periodic_torsions(&mut ff, &sec)?,
                "NonbondedForce" => {
                    coulomb14 = opt_f64(&sec, "coulomb14scale")?.unwrap_or(0.5);
                    lj14 = opt_f64(&sec, "lj14scale")?.unwrap_or(0.5);
                    for a in sec.children().filter(Node::is_element) {
                        // Skip NonbondedForce children that are not Atom rows
                        // (e.g. UseAttributeFromResidue).
                        if a.tag_name().name() != "Atom" {
                            continue;
                        }
                        nonbonded.push(NonbondedRow {
                            ty: require_str(&a, "type")?.to_owned(),
                            // TIP3P et al. use ``UseAttributeFromResidue`` and
                            // omit charge on NonbondedForce/Atom — do not invent
                            // ``0.0`` (that would overwrite charges on the graph
                            // at assign time).
                            charge: opt_f64(&a, "charge")?,
                            sigma: require_f64(&a, "sigma")? * NM_TO_ANGSTROM,
                            epsilon: require_f64(&a, "epsilon")? / KJ_PER_KCAL,
                        });
                    }
                }
                // OpenMM residue templates / alternate torsion spellings are not
                // potential tables for this reader — skip rather than fail so
                // tip3p.xml / clp.xml (and other OpenMM-style packs) load.
                "Residues"
                | "ImproperTorsionForce"
                | "PeriodicImproperForce"
                | "CustomTorsionForce"
                | "CustomBondForce"
                | "CustomAngleForce"
                | "CustomNonbondedForce" => {}
                other => {
                    return Err(format!("unknown OPLS section <{}>", other));
                }
            }
        }

        build_nonbonded(&mut ff, &atom_rows, &nonbonded);
        ensure_class_wildcards(&mut ff, &atom_rows);
        // OPLS excludes 1-2/1-3 and scales 1-4 by the <NonbondedForce> values
        // (commonly 0.5 / 0.5). Owned by the ForceField, consumed by the pair
        // kernels.
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, lj14],
            coul: [0.0, 0.0, coulomb14],
        });
        Ok(ff)
    }
}

/// One `<Type>` row of `<AtomTypes>` (mass + string metadata).
struct AtomTypeRow {
    name: String,
    mass: f64,
    class: Option<String>,
    element: Option<String>,
    def: Option<String>,
    desc: Option<String>,
    overrides: Option<String>,
}

/// One `<Atom>` row of `<NonbondedForce>`, already in molrs units.
struct NonbondedRow {
    ty: String,
    charge: Option<f64>,
    sigma: f64,
    epsilon: f64,
}

/// Build the atom style (`full`: mass + charge per type) and the two
/// nonbonded pair styles (`lj/cut`: per-atom ε/σ; `coul/cut`: charges come from
/// atoms at evaluation time). Combining rules and 1-4 scaling are NOT baked here
/// — combining is the kernel's job, and the 1-4 weights live on the
/// ForceField's `special_bonds` (set by the caller).
///
/// String metadata on each atom type matches molpy's reader contract:
/// ``type_`` (type name), ``class_`` (chemical class), ``element``, ``def_``.
/// :class:`~molpy.typifier._matching.TypeClassIndex` keys bonded matching off
/// these.
///
/// `coul/cut` is the **buffered** Coulomb `E = k·qᵢqⱼ/(D·(r + δ))`; OPLS is the
/// unbuffered case (δ = 0, the semantic default) in vacuum, with CODATA's `k`.
fn build_nonbonded(ff: &mut ForceField, atom_rows: &[AtomTypeRow], nonbonded: &[NonbondedRow]) {
    if !atom_rows.is_empty() {
        let atom = ff.def_atomstyle("full");
        for row in atom_rows {
            let charge = nonbonded
                .iter()
                .find(|r| r.ty == row.name)
                .and_then(|r| r.charge);
            let mut numeric: Vec<(&str, f64)> = vec![("mass", row.mass)];
            if let Some(q) = charge {
                numeric.push(("charge", q));
            }
            atom.def_atomtype(&row.name, &numeric);
            // type_ / class_ / element / def_ are string params used by typifiers.
            atom.set_type_str_param(&row.name, "type_", &row.name);
            if let Some(ref class) = row.class {
                atom.set_type_str_param(&row.name, "class_", class);
            }
            if let Some(ref element) = row.element {
                atom.set_type_str_param(&row.name, "element", element);
            }
            if let Some(ref def) = row.def {
                atom.set_type_str_param(&row.name, "def_", def);
            }
            if let Some(ref desc) = row.desc {
                atom.set_type_str_param(&row.name, "desc", desc);
            }
            if let Some(ref overrides) = row.overrides {
                atom.set_type_str_param(&row.name, "overrides", overrides);
            }
        }
    }

    if !nonbonded.is_empty() {
        let lj = ff.def_pairstyle("lj/cut", &[]);
        for r in nonbonded {
            lj.def_pairtype(&r.ty, None, &[("epsilon", r.epsilon), ("sigma", r.sigma)]);
        }
        ff.def_pairstyle(
            "coul/cut",
            &[("coulomb", COULOMB_REAL), ("dielectric", VACUUM_DIELECTRIC)],
        );
    }
}

/// Class-only bond/angle endpoints need a placeholder AtomType with
/// ``type_="*"`` and ``class_=<class>`` so TypeClassIndex / class-keyed
/// matching can resolve them (molpy XML reader parity).
fn ensure_class_wildcards(ff: &mut ForceField, atom_rows: &[AtomTypeRow]) {
    use std::collections::HashSet;

    let real_names: HashSet<String> = atom_rows.iter().map(|r| r.name.clone()).collect();
    let mut endpoint_classes: HashSet<String> = HashSet::new();

    // Classes declared on AtomTypes that are not themselves type names.
    for row in atom_rows {
        if let Some(ref c) = row.class
            && !real_names.contains(c)
        {
            endpoint_classes.insert(c.clone());
        }
    }
    // Bond / angle endpoint labels that aren't real atom-type names
    // (class-keyed HarmonicBondForce rows).
    for bt in ff.get_bondtypes() {
        for part in [&bt.itom, &bt.jtom] {
            if !real_names.contains(part) {
                endpoint_classes.insert(part.clone());
            }
        }
    }
    for at in ff.get_angletypes() {
        for part in [&at.itom, &at.jtom, &at.ktom] {
            if !real_names.contains(part) {
                endpoint_classes.insert(part.clone());
            }
        }
    }
    for dt in ff.get_dihedraltypes() {
        for part in [&dt.itom, &dt.jtom, &dt.ktom, &dt.ltom] {
            if !real_names.contains(part) {
                endpoint_classes.insert(part.clone());
            }
        }
    }

    if endpoint_classes.is_empty() {
        return;
    }

    // Prefer the existing "full" atom style; create one only if needed.
    if ff.get_style("atom", "full").is_none() && ff.get_styles("atom").is_empty() {
        ff.def_atomstyle("full");
    }
    let style_name = if ff.get_style("atom", "full").is_some() {
        "full".to_owned()
    } else {
        ff.get_styles("atom")
            .first()
            .map(|s| s.name.clone())
            .unwrap_or_else(|| "full".to_owned())
    };
    if ff.get_style("atom", &style_name).is_none() {
        ff.def_atomstyle(&style_name);
    }
    let atom = ff
        .get_style_mut("atom", &style_name)
        .expect("atom style just ensured");

    for class_name in endpoint_classes {
        if real_names.contains(&class_name) {
            continue;
        }
        if atom.get_atomtype(&class_name).is_some() {
            continue;
        }
        atom.def_atomtype(&class_name, &[]);
        atom.set_type_str_param(&class_name, "type_", "*");
        atom.set_type_str_param(&class_name, "class_", &class_name);
    }
}

/// OpenMM packs use either `classN` (chemical class) or `typeN` (atom type name).
/// Missing both falls back to the wildcard ``*`` so incomplete writers
/// (e.g. moltemplate XML without endpoint labels) still round-trip.
fn class_or_type<'a>(node: &'a Node, n: usize) -> Result<&'a str, String> {
    let class_key = format!("class{n}");
    let type_key = format!("type{n}");
    Ok(node
        .attribute(class_key.as_str())
        .or_else(|| node.attribute(type_key.as_str()))
        .unwrap_or("*"))
}

fn parse_bonds(ff: &mut ForceField, sec: &Node) -> Result<(), String> {
    let style = ff.def_bondstyle("harmonic");
    for b in sec.children().filter(Node::is_element) {
        require_tag(&b, "Bond")?;
        let c1 = class_or_type(&b, 1)?;
        let c2 = class_or_type(&b, 2)?;
        let r0 = require_f64(&b, "length")? * NM_TO_ANGSTROM;
        // kJ/mol/nm² → kcal/mol/Å² : ÷4.184 (energy) ÷100 (nm²→Å²). Same ½ form.
        let k = require_f64(&b, "k")? / (KJ_PER_KCAL * 100.0);
        // Emit both `k` (molpy / LAMMPS surface) and `k0` (kernel alias).
        style.def_bondtype(c1, c2, &[("k", k), ("k0", k), ("r0", r0)]);
    }
    Ok(())
}

fn parse_angles(ff: &mut ForceField, sec: &Node) -> Result<(), String> {
    let style = ff.def_anglestyle("harmonic");
    for a in sec.children().filter(Node::is_element) {
        require_tag(&a, "Angle")?;
        let c1 = class_or_type(&a, 1)?;
        let c2 = class_or_type(&a, 2)?;
        let c3 = class_or_type(&a, 3)?;
        let theta0 = require_f64(&a, "angle")?; // already radians
        let k = require_f64(&a, "k")? / KJ_PER_KCAL; // kJ/mol/rad² → kcal/mol/rad²
        // Emit both `k` (molpy surface) and `k0` (kernel alias).
        style.def_angletype(c1, c2, c3, &[("k", k), ("k0", k), ("theta0", theta0)]);
    }
    Ok(())
}

fn parse_dihedrals(ff: &mut ForceField, sec: &Node) -> Result<(), String> {
    let style = ff.def_dihedralstyle("opls");
    for d in sec.children().filter(Node::is_element) {
        require_tag(&d, "Proper")?;
        let c1 = require_str(&d, "class1")?;
        let c2 = require_str(&d, "class2")?;
        let c3 = require_str(&d, "class3")?;
        let c4 = require_str(&d, "class4")?;
        let rb = [
            opt_f64(&d, "c0")?.unwrap_or(0.0),
            opt_f64(&d, "c1")?.unwrap_or(0.0),
            opt_f64(&d, "c2")?.unwrap_or(0.0),
            opt_f64(&d, "c3")?.unwrap_or(0.0),
            opt_f64(&d, "c4")?.unwrap_or(0.0),
            opt_f64(&d, "c5")?.unwrap_or(0.0),
        ];
        let [f1, f2, f3, f4] = rb_to_opls(rb);
        style.def_dihedraltype(
            c1,
            c2,
            c3,
            c4,
            &[("f1", f1), ("f2", f2), ("f3", f3), ("f4", f4)],
        );
    }
    Ok(())
}

/// CL&P / foyer `PeriodicTorsionForce` rows carry OPLS Fourier coeffs
/// ``c0..c3`` in kJ/mol (not the OpenMM k/periodicity/phase form). Convert
/// to kcal/mol ``f1..f4`` on the `opls` dihedral style.
fn parse_periodic_torsions(ff: &mut ForceField, sec: &Node) -> Result<(), String> {
    let style = ff.def_dihedralstyle("opls");
    for d in sec.children().filter(Node::is_element) {
        if d.tag_name().name() != "Proper" {
            // Improper children under PeriodicTorsionForce are rare; skip.
            continue;
        }
        let c1 = class_or_type(&d, 1)?;
        let c2 = class_or_type(&d, 2)?;
        let c3 = class_or_type(&d, 3)?;
        let c4 = class_or_type(&d, 4)?;
        // Prefer foyer/CL&P Fourier spelling (c0..c3); fall back to zero terms.
        let f1 = opt_f64(&d, "c0")?.unwrap_or(0.0) / KJ_PER_KCAL;
        let f2 = opt_f64(&d, "c1")?.unwrap_or(0.0) / KJ_PER_KCAL;
        let f3 = opt_f64(&d, "c2")?.unwrap_or(0.0) / KJ_PER_KCAL;
        let f4 = opt_f64(&d, "c3")?.unwrap_or(0.0) / KJ_PER_KCAL;
        style.def_dihedraltype(
            c1,
            c2,
            c3,
            c4,
            &[("f1", f1), ("f2", f2), ("f3", f3), ("f4", f4)],
        );
    }
    Ok(())
}

/// Convert Ryckaert–Bellemans coefficients `[c0..c5]` (kJ/mol) to OPLS 4-cosine
/// Fourier coefficients `[f1, f2, f3, f4]` (kcal/mol).
///
/// The OPLS torsion is
/// `V = ½[F1(1+cosφ) + F2(1−cos2φ) + F3(1+cos3φ) + F4(1−cos4φ)]`, the RB form is
/// `V = Σ Cₙ(cosψ)ⁿ`, ψ = φ − π. GROMACS manual Eqs. 200–201 give the exact
/// analytic inversion (independent of `c0` and `c5`):
///
/// ```text
/// F1 = −2·C1 − 1.5·C3
/// F2 =   −C2 −     C4
/// F3 =        −0.5·C3
/// F4 =       −0.25·C4
/// ```
///
/// The kJ/mol → kcal/mol factor (÷ 4.184) is applied here, matching molpy's
/// `rb_to_opls(..., units="kJ")`.
fn rb_to_opls([_c0, c1, c2, c3, c4, _c5]: [f64; 6]) -> [f64; 4] {
    let f1 = -2.0 * c1 - 1.5 * c3;
    let f2 = -c2 - c4;
    let f3 = -0.5 * c3;
    let f4 = -0.25 * c4;
    [
        f1 / KJ_PER_KCAL,
        f2 / KJ_PER_KCAL,
        f3 / KJ_PER_KCAL,
        f4 / KJ_PER_KCAL,
    ]
}

// --- attribute helpers (total: missing/malformed → Err) -------------------

fn require_tag(node: &Node, expect: &str) -> Result<(), String> {
    let got = node.tag_name().name();
    if got == expect {
        Ok(())
    } else {
        Err(format!("expected <{}>, got <{}>", expect, got))
    }
}

fn require_str<'a>(node: &'a Node, attr: &str) -> Result<&'a str, String> {
    node.attribute(attr).ok_or_else(|| {
        format!(
            "<{}> missing required attribute `{}`",
            node.tag_name().name(),
            attr
        )
    })
}

fn require_f64(node: &Node, attr: &str) -> Result<f64, String> {
    let raw = require_str(node, attr)?;
    raw.parse::<f64>().map_err(|_| {
        format!(
            "<{}> attribute `{}` is not a number: {:?}",
            node.tag_name().name(),
            attr,
            raw
        )
    })
}

fn opt_f64(node: &Node, attr: &str) -> Result<Option<f64>, String> {
    match node.attribute(attr) {
        None => Ok(None),
        Some(raw) => raw.parse::<f64>().map(Some).map_err(|_| {
            format!(
                "<{}> attribute `{}` is not a number: {:?}",
                node.tag_name().name(),
                attr,
                raw
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny but genuine OPLS-AA/GROMACS XML excerpt (rows copied from molpy's
    /// bundled `oplsaa.xml`), exercising every section. Used for conversion and
    /// edge-case unit tests; full-file parity lives in the bm-molrs-molpy harness.
    const MINI: &str = r#"<ForceField name="OPLS-AA" combining_rule="geometric">
  <AtomTypes>
    <Type name="opls_001" class="opls_001" element="C" mass="12.011"/>
    <Type name="opls_002" class="opls_002" element="O" mass="15.9994"/>
  </AtomTypes>
  <HarmonicBondForce>
    <Bond class1="OW" class2="HW" length="0.09572" k="502080.0"/>
  </HarmonicBondForce>
  <HarmonicAngleForce>
    <Angle class1="HW" class2="OW" class3="HW" angle="1.91113553093" k="627.6"/>
  </HarmonicAngleForce>
  <RBTorsionForce>
    <Proper class1="Br" class2="C" class3="CT" class4="HC" c0="0.75312" c1="2.25936" c2="0.0" c3="-3.01248" c4="0.0" c5="0.0"/>
  </RBTorsionForce>
  <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
    <Atom type="opls_001" charge="0.5" sigma="0.375" epsilon="0.43932"/>
    <Atom type="opls_002" charge="-0.5" sigma="0.296" epsilon="0.87864"/>
  </NonbondedForce>
</ForceField>"#;

    #[test]
    fn rb_to_opls_matches_gromacs_inversion() {
        // c1=2.25936, c3=-3.01248 (kJ); others 0.
        let [f1, f2, f3, f4] = rb_to_opls([0.75312, 2.25936, 0.0, -3.01248, 0.0, 0.0]);
        // F1 = -2*c1 - 1.5*c3 = -4.51872 + 4.51872 = 0  → /4.184 = 0
        assert!((f1 - 0.0).abs() < 1e-12, "f1 {f1}");
        // F2 = -c2 - c4 = 0
        assert!((f2 - 0.0).abs() < 1e-12, "f2 {f2}");
        // F3 = -0.5*c3 = 1.50624 kJ → /4.184 = 0.360 kcal
        assert!((f3 - (1.50624 / 4.184)).abs() < 1e-12, "f3 {f3}");
        // F4 = -0.25*c4 = 0
        assert!((f4 - 0.0).abs() < 1e-12, "f4 {f4}");
    }

    #[test]
    fn reads_all_sections_with_molrs_units() {
        let ff = OplsXmlReader::new().read_str(MINI).unwrap();

        // bond: length 0.09572 nm → 0.9572 Å; k 502080 kJ/mol/nm² → /418.4 kcal/mol/Å².
        let bond = ff.get_style("bond", "harmonic").unwrap();
        let bt = bond.get_bondtype("OW", "HW").unwrap();
        assert!((bt.params.get("r0").unwrap() - 0.9572).abs() < 1e-9);
        assert!((bt.params.get("k0").unwrap() - 502080.0 / 418.4).abs() < 1e-6);

        // angle: theta0 unchanged (rad); k 627.6 → /4.184 = 150.0 kcal/mol/rad².
        let angle = ff.get_style("angle", "harmonic").unwrap();
        let at = &angle_types(angle)[0];
        assert!((at.params.get("theta0").unwrap() - 1.91113553093).abs() < 1e-9);
        assert!((at.params.get("k0").unwrap() - 627.6 / 4.184).abs() < 1e-9);

        // dihedral opls f1..f4 present.
        let dih = ff.get_style("dihedral", "opls").unwrap();
        assert!(dihedral_types(dih)[0].params.get("f3").is_some());

        // pair lj/cut: sigma 0.375 nm → 3.75 Å; epsilon 0.43932 kJ → /4.184 kcal.
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let pt = lj.get_pairtype("opls_001", None).unwrap();
        assert!((pt.params.get("sigma").unwrap() - 3.75).abs() < 1e-9);
        assert!((pt.params.get("epsilon").unwrap() - 0.43932 / 4.184).abs() < 1e-9);

        // coul/cut style present (charges resolved per-atom from the frame).
        assert!(ff.get_style("pair", "coul/cut").is_some());

        // The 1-4 scales live on the ForceField's special_bonds (1-2/1-3
        // excluded) — the single source the pair kernels consume.
        let sb = ff.special_bonds();
        assert_eq!(sb.lj, [0.0, 0.0, 0.5]);
        assert_eq!(sb.coul, [0.0, 0.0, 0.5]);

        // atom style carries mass + charge per opls type.
        let atom = ff.get_style("atom", "full").unwrap();
        let a1 = atom.get_atomtype("opls_001").unwrap();
        assert!((a1.params.get("mass").unwrap() - 12.011).abs() < 1e-9);
        assert!((a1.params.get("charge").unwrap() - 0.5).abs() < 1e-12);
        let a2 = atom.get_atomtype("opls_002").unwrap();
        assert!((a2.params.get("charge").unwrap() + 0.5).abs() < 1e-12);
    }

    #[test]
    fn missing_required_attr_errors() {
        let xml = r#"<ForceField name="x"><HarmonicBondForce>
            <Bond class1="OW" class2="HW" length="0.1"/>
        </HarmonicBondForce></ForceField>"#;
        let err = OplsXmlReader::new().read_str(xml).unwrap_err();
        assert!(err.contains('k'), "err: {err}");
    }

    #[test]
    fn non_numeric_attr_errors() {
        let xml = r#"<ForceField name="x"><HarmonicBondForce>
            <Bond class1="OW" class2="HW" length="oops" k="1.0"/>
        </HarmonicBondForce></ForceField>"#;
        let err = OplsXmlReader::new().read_str(xml).unwrap_err();
        assert!(err.contains("not a number"), "err: {err}");
    }

    #[test]
    fn wrong_root_errors() {
        let err = OplsXmlReader::new()
            .read_str(r#"<System name="x"/>"#)
            .unwrap_err();
        assert!(err.contains("ForceField"), "err: {err}");
    }

    #[test]
    fn unknown_section_errors() {
        let xml = r#"<ForceField name="x"><MysteryForce/></ForceField>"#;
        let err = OplsXmlReader::new().read_str(xml).unwrap_err();
        assert!(err.contains("unknown OPLS section"), "err: {err}");
    }

    // -- small helpers to reach into StyleDefs for assertions --
    use crate::ff::forcefield::{AngleType, DihedralType, Style, StyleDefs};
    fn angle_types(s: &Style) -> &[AngleType] {
        match &s.defs {
            StyleDefs::Angle(v) => v,
            _ => unreachable!(),
        }
    }
    fn dihedral_types(s: &Style) -> &[DihedralType] {
        match &s.defs {
            StyleDefs::Dihedral(v) => v,
            _ => unreachable!(),
        }
    }
}
