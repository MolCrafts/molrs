//! OPLS-AA SMARTS atom typifier.
//!
//! Mirrors [`mmff`](crate::ff::typifier::mmff): typing metadata
//! ([`OplsTypingMeta`]) is read *separately* from the potential [`ForceField`],
//! both from the same OPLS-AA XML. [`OPLSAATypifier`] owns both and implements
//! [`Typifier`], assigning `opls_NNN` atom types by SMARTS matching with
//! overrides / priority / layer conflict resolution.
//!
//! After atom typing, its `r#match` ([`Typifier`]) matches every bond /
//! angle / dihedral against the force field's bonded tables by OPLS
//! specificity + overlay layer (chain 2).
//! [`Typing`](crate::ff::typifier::Typing) runs the match: it stamps
//! the typed copy and defines exactly the types used in its output force field.
//! Callers compose `Typing::typify` → `to_frame` → pairs →
//! `PotentialCompiler::new(typing.forcefield()).compile`.
//!
//! # B-line reversal
//!
//! This reverses the "typifier does not sink (B-line)" decision of
//! `opls-ef-01-kernels-seam`: OPLS bonded-parameter assignment now happens in
//! Rust (here), not in a post-typify Python pass over a molpy `ForceField`.
//!
//! # Scope
//!
//! Only types carrying a SMARTS `def` participate; legacy `oplsaa.xml` rows
//! (`opls_001`–`opls_134`, no `def`) are out of scope for auto-typing. Improper
//! matching is out of scope. Uncovered bonded terms follow the [`NoMatch`]
//! policy; a consumer that wants to fill them can attach its own [`Estimator`]
//! via [`OPLSAATypifier::with_estimator`], or the restored
//! [`Parmchk2Estimator`] via [`OPLSAATypifier::with_default_estimator`].

use std::collections::HashSet;

use molrs::{AtomId, Atomistic};

use crate::ff::forcefield::ForceField;
use crate::ff::forcefield::readers::{ForceFieldReader, opls::OplsXmlReader};

use crate::ff::typifier::estimate::Parmchk2Estimator;
use crate::ff::typifier::{Match, Typifier};

pub mod assign;
pub mod deps;
mod embedded;
pub mod layered;
pub mod meta;
pub(crate) mod typing;

pub use assign::{BondedTerm, CandidateTables, Estimator, NoMatch};
pub use meta::{LAYER_PRIORITY_STRIDE, OplsTypeRow, OplsTypingMeta};

use assign::typify_bonded_with;
use typing::typify_atoms;

/// OPLS-AA typifier — owns typing metadata and force-field parameters.
///
/// Primary constructor [`from_xml_str`](Self::from_xml_str) parses both the
/// typing metadata ([`OplsTypingMeta`]) and the potential parameters
/// ([`ForceField`]) from a single OPLS-AA XML string, then precomputes the
/// bonded candidate tables ([`CandidateTables`]) once.
pub struct OPLSAATypifier {
    meta: OplsTypingMeta,
    ff: ForceField,
    tables: CandidateTables,
    /// No-match policy for bonded terms with no force-field candidate.
    no_match: NoMatch,
    /// Optional missing-parameter interpolator for bonded terms.
    estimator: Option<Box<dyn Estimator + Send + Sync>>,
}

impl OPLSAATypifier {
    /// Build a typifier from an OPLS-AA / GROMACS XML string.
    ///
    /// Reads typing metadata and potential parameters in one call. The two are
    /// read by independent parsers from the same XML and never share state.
    /// The bonded candidate tables are built once from the parsed force field.
    /// Defaults to strict bonded matching ([`NoMatch::Error`]).
    ///
    /// # Errors
    ///
    /// Returns `Err` if either parse fails.
    pub fn from_xml_str(xml: &str) -> Result<Self, String> {
        let meta = crate::ff::forcefield::xml::read_opls_typing_xml_str(xml)?;
        let ff = OplsXmlReader::new().read_str(xml)?;
        Ok(Self::new(meta, ff))
    }

    /// Build a typifier over the shipped canonical OPLS-AA parameter set
    /// ([`crate::ff::params::oplsaa`]).
    ///
    /// The parameters are compiled-in typed Rust, so this is the standalone
    /// path: the OPLS typifier needs no external file on disk and parses nothing
    /// at runtime. The shipped set uses lowercase SMARTS `c` for aromatic ring
    /// carbons / hydrogens (RDKit-faithful aromatic matching), so benzene-type
    /// rings type exactly as molpy's ground truth. Mirrors
    /// [`MMFF94Typifier::new`](crate::ff::typifier::mmff::MMFF94Typifier::new).
    ///
    /// Infallible: the parameters are compile-time constants, the same policy
    /// as `MMFF94Typifier::new` and `UFFTypifier::new`.
    pub fn oplsaa() -> Self {
        Self::new(embedded::typing_meta(), embedded::force_field())
    }

    /// Construct directly from already-parsed metadata and force field
    /// (strict bonded matching).
    pub fn new(meta: OplsTypingMeta, ff: ForceField) -> Self {
        let tables = CandidateTables::build(&ff, &meta);
        Self {
            meta,
            ff,
            tables,
            no_match: NoMatch::Error,
            estimator: None,
        }
    }

    /// Set the bonded no-match policy (chaining). `strict=true` →
    /// [`NoMatch::Error`]; `strict=false` → [`NoMatch::Skip`].
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.no_match = if strict {
            NoMatch::Error
        } else {
            NoMatch::Skip
        };
        self
    }

    /// Attach a bonded-term parameter estimator (chaining, opt-in).
    ///
    /// Exact force-field table matches still win first. To keep strict mode's
    /// contract stable, attached estimators are consulted only when this
    /// typifier is configured with [`NoMatch::Skip`] via [`with_strict(false)`](Self::with_strict).
    pub fn with_estimator<E>(mut self, estimator: E) -> Self
    where
        E: Estimator + Send + Sync + 'static,
    {
        self.estimator = Some(Box::new(estimator));
        self
    }

    /// Attach the default GAFF/parmchk2-style similarity estimator built from
    /// this typifier's force field and typing metadata.
    pub fn with_default_estimator(self) -> Self {
        let estimator = Parmchk2Estimator::new(&self.ff, &self.meta);
        self.with_estimator(estimator)
    }

    /// Access the typing metadata.
    pub fn meta(&self) -> &OplsTypingMeta {
        &self.meta
    }
}

impl Typifier for OPLSAATypifier {
    /// Type the atoms, then match every bonded term.
    ///
    /// Node annotations: `type` (a `Type` under `atom/full` with the row's
    /// `mass` and `charge`, or a plain value when the library has no row) and
    /// `class`. Bonded annotations: `type`, the matched class-keyed name with
    /// its library params, or an estimate named by
    /// [`BondedTerm::type_name`]. Styles: every library style, in library
    /// order. Pairs: the library's pair rows among the atom types used.
    ///
    /// # Errors
    ///
    /// Propagates atom-typing and bonded-matching errors. In strict mode
    /// ([`NoMatch::Error`]) also returns `Err` naming every atom no def typed,
    /// before any bonded term is matched.
    fn r#match(&self, graph: &mut Atomistic) -> Result<Match, String> {
        let atoms = typify_atoms(graph, &self.meta, &self.ff)?;
        if self.no_match == NoMatch::Error {
            let untyped: Vec<AtomId> = graph
                .atoms()
                .map(|(id, _)| id)
                .filter(|id| !atoms.types.contains_key(id))
                .collect();
            if !untyped.is_empty() {
                return Err(format!(
                    "OPLS: no atom type for {}",
                    assign::name_atoms(graph, &untyped)
                ));
            }
        }
        let estimator = match self.no_match {
            NoMatch::Error => None,
            NoMatch::Skip => self.estimator.as_deref().map(|e| e as &dyn Estimator),
        };
        let mut m =
            typify_bonded_with(graph, &atoms.types, &self.tables, self.no_match, estimator)?;
        let used: HashSet<&str> = atoms.types.values().map(String::as_str).collect();
        m.nodes = atoms.nodes;
        m.declare_styles_of(&self.ff);
        m.add_pairs_among(&self.ff, &used);
        Ok(m)
    }

    /// The OPLS-AA force field this typifier matches against.
    fn library(&self) -> &ForceField {
        &self.ff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::forcefield::Params;
    use crate::ff::typifier::Typing;
    use molrs::Atom;
    use molrs::system::BondType;
    use molrs::system::molgraph::PropValue;
    use std::collections::{BTreeMap, BTreeSet, HashMap};

    /// Explicit-H 1,3-butadiene `H2C=CH-CH=CH2`, hand-built.
    ///
    /// Atom order (0-based graph index): carbons `C0=C1-C2=C3` are 0..=3; the
    /// hydrogens follow — 4, 5 on C0; 6 on C1; 7 on C2; 8, 9 on C3.
    fn butadiene() -> Atomistic {
        let mut g = Atomistic::new();
        let c: Vec<_> = (0..4)
            .map(|k| g.add_atom(Atom::xyz("C", 1.4 * k as f64, 0.0, 0.0)))
            .collect();
        let double = g.add_bond(c[0], c[1]).unwrap();
        g.set_bond_type(double, BondType::Double).unwrap();
        g.add_bond(c[1], c[2]).unwrap();
        let double = g.add_bond(c[2], c[3]).unwrap();
        g.set_bond_type(double, BondType::Double).unwrap();
        for (carbon, n_h) in [(c[0], 2), (c[1], 1), (c[2], 1), (c[3], 2)] {
            for k in 0..n_h {
                let h = g.add_atom(Atom::xyz("H", 0.3 * k as f64, 1.0, 0.0));
                g.add_bond(carbon, h).unwrap();
            }
        }
        g
    }

    /// The name an untyped-atom error gives one atom: its 0-based index in graph
    /// atom order and its element, the ATD typifier's `atom {i} ({element})`.
    fn atom_label(index: usize, element: &str) -> String {
        format!("atom {index} ({element})")
    }

    /// Strict typing of a molecule with atoms no def matches is an `Err` that
    /// names every untyped atom — and only those.
    ///
    /// Hand-derived from the shipped defs: an unmarked SMARTS bond is
    /// single-or-aromatic, so `[C;X3](C)(H)H` (opls_143) cannot see a terminal
    /// carbon's `=` neighbour and `[C;X3](C)(C)H` (opls_142) sees only one of an
    /// inner carbon's two carbon neighbours. All four carbons stay untyped; every
    /// hydrogen matches `[H][C;X3]` (opls_144).
    #[test]
    fn strict_typify_names_every_untyped_atom() {
        let err = Typing::new(OPLSAATypifier::oplsaa().with_strict(true))
            .typify(&butadiene())
            .expect_err("strict typing must refuse a partly typed molecule");

        for i in 0..4 {
            let label = atom_label(i, "C");
            assert!(err.contains(&label), "err names {label}: {err}");
        }
        for i in 4..10 {
            let label = atom_label(i, "H");
            assert!(
                !err.contains(&label),
                "err must not name typed {label}: {err}"
            );
        }
    }

    /// Non-strict typing of the same molecule keeps its current contract: `Ok`,
    /// untyped atoms left without a `type`.
    #[test]
    fn non_strict_typify_accepts_a_partly_typed_molecule() {
        let typed = Typing::new(OPLSAATypifier::oplsaa().with_strict(false)).typify(&butadiene());
        assert!(typed.is_ok(), "non-strict typing stays Ok: {typed:?}");
    }

    /// A typifier whose defs cover C and H only, over a force field whose
    /// all-wildcard bonded rows match every fully typed term — so a strict
    /// failure can come from atom coverage alone, never from a bonded miss.
    fn c_h_only_typifier() -> OPLSAATypifier {
        let mut meta = OplsTypingMeta::new();
        let row = |class: &str, def: &str| OplsTypeRow {
            class: class.to_string(),
            def: Some(def.to_string()),
            overrides: Vec::new(),
            priority: None,
            layer: 0,
        };
        meta.insert("opls_c", row("CT", "[C;X4]"));
        meta.insert("opls_h", row("HC", "[H]"));

        let mut ff = ForceField::new("OPLS-AA");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type_at(
                "X-X",
                &["X", "X"],
                Params::from_pairs(&[("k", 1.0), ("r0", 1.0)]),
            )
            .unwrap();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type_at(
                "X-X-X",
                &["X", "X", "X"],
                Params::from_pairs(&[("k", 1.0), ("theta0", 1.9)]),
            )
            .unwrap();
        ff.def_style("dihedral", "opls", Params::new())
            .unwrap()
            .def_type_at(
                "X-X-X-X",
                &["X", "X", "X", "X"],
                Params::from_pairs(&[("k1", 0.0), ("k2", 0.0), ("k3", 0.0), ("k4", 0.0)]),
            )
            .unwrap();
        OPLSAATypifier::new(meta, ff).with_strict(true)
    }

    /// Methanol `CH3-OH`, hand-built: C is atom 0, O is atom 1, the three methyl
    /// hydrogens are 2..=4, the hydroxyl hydrogen is 5.
    fn methanol() -> Atomistic {
        let mut g = Atomistic::new();
        let c = g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
        let o = g.add_atom(Atom::xyz("O", 1.4, 0.0, 0.0));
        g.add_bond(c, o).unwrap();
        for k in 0..3 {
            let h = g.add_atom(Atom::xyz("H", -0.3 * k as f64, 1.0, 0.0));
            g.add_bond(c, h).unwrap();
        }
        let h = g.add_atom(Atom::xyz("H", 1.8, 0.9, 0.0));
        g.add_bond(o, h).unwrap();
        g
    }

    // -- Typing<OPLSAATypifier> output (system-forcefield-07) -----------------

    /// Ethane `CH3-CH3`, hand-built: carbons are atoms 0 and 1, the three
    /// hydrogens on C0 are 2..=4, the three on C1 are 5..=7.
    fn ethane() -> Atomistic {
        let mut g = Atomistic::new();
        let c0 = g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
        let c1 = g.add_atom(Atom::xyz("C", 1.5, 0.0, 0.0));
        g.add_bond(c0, c1).unwrap();
        for c in [c0, c1] {
            for k in 0..3 {
                let h = g.add_atom(Atom::xyz("H", 0.3 * k as f64, 1.0, 0.0));
                g.add_bond(c, h).unwrap();
            }
        }
        g
    }

    /// The params the stub estimator fills every bond with: `k`, `r0` and the
    /// four provenance keys of `estimate/provenance.rs`.
    fn estimated_bond_params() -> Params {
        let mut p = Params::from_pairs(&[
            ("k", 536.0),
            ("r0", 1.529),
            ("estimated", 1.0),
            ("estimate_penalty", 0.0),
        ]);
        p.set_str("estimate_method", "analogy");
        p.set_str("estimate_analog", "CT-CT");
        p
    }

    /// Fills every bond with [`estimated_bond_params`]; declines every other
    /// term.
    struct StubBondEstimator;

    impl crate::ff::typifier::estimate::ParameterInterpolator for StubBondEstimator {
        type Term = BondedTerm;

        fn interpolate(&self, term: &BondedTerm) -> Result<Option<Params>, String> {
            Ok(match term {
                BondedTerm::Bond(_) => Some(estimated_bond_params()),
                _ => None,
            })
        }
    }

    /// A two-type library (`opls_135` CT carbon, `opls_140` HC hydrogen) plus
    /// one extra type ethane never uses (`opls_154` OH oxygen) with its own
    /// atom, pair and bond rows. The library has no `CT-CT` bond row, so the C-C
    /// bond is filled by [`StubBondEstimator`] (lenient mode consults it).
    fn ethane_library_typifier() -> OPLSAATypifier {
        let mut meta = OplsTypingMeta::new();
        let row = |class: &str, def: &str| OplsTypeRow {
            class: class.to_string(),
            def: Some(def.to_string()),
            overrides: Vec::new(),
            priority: None,
            layer: 0,
        };
        meta.insert("opls_135", row("CT", "[C;X4]"));
        meta.insert("opls_140", row("HC", "[H]"));
        meta.insert("opls_154", row("OH", "[O;X2]"));

        let mut ff = ForceField::new("OPLS-AA");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type_at(
                "opls_135",
                &[],
                Params::from_pairs(&[("mass", 12.011), ("charge", -0.18)]),
            )
            .unwrap()
            .def_type_at(
                "opls_140",
                &[],
                Params::from_pairs(&[("mass", 1.008), ("charge", 0.06)]),
            )
            .unwrap()
            .def_type_at(
                "opls_154",
                &[],
                Params::from_pairs(&[("mass", 15.999), ("charge", -0.683)]),
            )
            .unwrap();
        ff.def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap()
            .def_type_at(
                "opls_135",
                &["opls_135"],
                Params::from_pairs(&[("epsilon", 0.066), ("sigma", 3.5)]),
            )
            .unwrap()
            .def_type_at(
                "opls_140",
                &["opls_140"],
                Params::from_pairs(&[("epsilon", 0.03), ("sigma", 2.5)]),
            )
            .unwrap()
            .def_type_at(
                "opls_154",
                &["opls_154"],
                Params::from_pairs(&[("epsilon", 0.17), ("sigma", 3.12)]),
            )
            .unwrap();
        ff.def_style("pair", "coul/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap();
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type_at(
                "CT-HC",
                &["CT", "HC"],
                Params::from_pairs(&[("k", 680.0), ("r0", 1.09)]),
            )
            .unwrap()
            .def_type_at(
                "CT-OH",
                &["CT", "OH"],
                Params::from_pairs(&[("k", 640.0), ("r0", 1.41)]),
            )
            .unwrap();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type_at(
                "HC-CT-HC",
                &["HC", "CT", "HC"],
                Params::from_pairs(&[("k", 66.0), ("theta0", 1.881)]),
            )
            .unwrap()
            .def_type_at(
                "CT-CT-HC",
                &["CT", "CT", "HC"],
                Params::from_pairs(&[("k", 75.0), ("theta0", 1.932)]),
            )
            .unwrap();
        ff.def_style("dihedral", "opls", Params::new())
            .unwrap()
            .def_type_at(
                "HC-CT-CT-HC",
                &["HC", "CT", "CT", "HC"],
                Params::from_pairs(&[("k1", 0.0), ("k2", 0.0), ("k3", 0.6276), ("k4", 0.0)]),
            )
            .unwrap();

        OPLSAATypifier::new(meta, ff)
            .with_strict(false)
            .with_estimator(StubBondEstimator)
    }

    /// Ethane typed through the base: the typed copy and the base.
    fn typed_ethane() -> (Atomistic, crate::ff::typifier::Typing<OPLSAATypifier>) {
        let mut typing = crate::ff::typifier::Typing::new(ethane_library_typifier());
        let typed = typing.typify(&ethane()).expect("ethane types");
        (typed, typing)
    }

    fn str_prop(props: &HashMap<String, PropValue>, key: &str) -> Option<String> {
        match props.get(key) {
            Some(PropValue::Str(s)) => Some(s.clone()),
            _ => None,
        }
    }

    /// Per `(category, style)` of `ff`, the set of type names; styles holding
    /// no type are left out.
    fn output_names(ff: &ForceField) -> BTreeMap<(String, String), BTreeSet<String>> {
        ff.styles()
            .iter()
            .filter_map(|s| {
                let names: BTreeSet<String> = s
                    .defs()
                    .collect_type_params()
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect();
                (!names.is_empty()).then(|| ((s.category().to_owned(), s.name().to_owned()), names))
            })
            .collect()
    }

    /// Every output `(category, style)` holds exactly the labels stamped on the
    /// typed graph: the atom `type`s under `atom/full` and as the self rows of
    /// `pair/lj/cut`, the bond / angle / dihedral `type`s under their one style.
    #[test]
    fn typing_output_names_equal_the_stamped_labels() {
        let (typed, typing) = typed_ethane();

        let atoms: BTreeSet<String> = typed
            .atoms()
            .map(|(_, a)| a.get_str("type").expect("every atom typed").to_owned())
            .collect();
        let bonds: BTreeSet<String> = typed
            .bonds()
            .map(|(_, b)| str_prop(&b.props, "type").expect("every bond typed"))
            .collect();
        let angles: BTreeSet<String> = typed
            .angles()
            .map(|(_, a)| str_prop(&a.props, "type").expect("every angle typed"))
            .collect();
        let dihedrals: BTreeSet<String> = typed
            .dihedrals()
            .map(|(_, d)| str_prop(&d.props, "type").expect("every dihedral typed"))
            .collect();

        // Hand-derived from the defs: [C;X4] types both carbons, [H] every H.
        let expected_atoms: BTreeSet<String> = ["opls_135", "opls_140"].map(str::to_owned).into();
        assert_eq!(atoms, expected_atoms);

        let key = |c: &str, s: &str| (c.to_owned(), s.to_owned());
        let expected = BTreeMap::from([
            (key("atom", "full"), atoms.clone()),
            (key("pair", "lj/cut"), atoms),
            (key("bond", "harmonic"), bonds),
            (key("angle", "harmonic"), angles),
            (key("dihedral", "opls"), dihedrals),
        ]);
        assert_eq!(output_names(typing.forcefield()), expected);
    }

    /// The unused library type `opls_154` and its rows (`atom/full`, the
    /// `pair/lj/cut` self row, the `CT-OH` bond) do not reach the output.
    #[test]
    fn typing_output_omits_the_unused_library_rows() {
        let (_, typing) = typed_ethane();
        let out = typing.forcefield();

        let atom_full = out.get_style("atom", "full").expect("atom/full declared");
        assert!(atom_full.get_atomtype("opls_154").is_none());
        let lj = out
            .get_style("pair", "lj/cut")
            .expect("pair/lj/cut declared");
        assert!(lj.get_pairtype("opls_154", None).is_none());
        let bonds = out
            .get_style("bond", "harmonic")
            .expect("bond/harmonic declared");
        assert!(bonds.type_endpoints("CT-OH").is_none());
    }

    /// The estimator-filled C-C bond is stamped `type` =
    /// `BondedTerm::type_name()` of its endpoint atom types (the `TypeName`
    /// join, `opls_135-opls_135`), and that name is defined in the output with
    /// the estimator's params, provenance strings included.
    #[test]
    fn typing_names_and_defines_an_estimator_filled_bond() {
        let (typed, typing) = typed_ethane();
        let c0 = typed.atoms().next().expect("atom 0").0;
        let c1 = typed.atoms().nth(1).expect("atom 1").0;
        let (_, cc) = typed
            .bonds()
            .find(|(_, b)| b.nodes.contains(&c0) && b.nodes.contains(&c1))
            .expect("the C-C bond");

        let term = BondedTerm::Bond(["opls_135".to_owned(), "opls_135".to_owned()]);
        let name = term.type_name().expect("a bond term has a type name");
        assert_eq!(name.as_str(), "opls_135-opls_135");
        assert_eq!(str_prop(&cc.props, "type").as_deref(), Some(name.as_str()));
        assert_eq!(
            str_prop(&cc.props, "estimate_method").as_deref(),
            Some("analogy")
        );

        let defined = typing
            .forcefield()
            .get_style("bond", "harmonic")
            .expect("bond/harmonic declared")
            .defs()
            .collect_type_params()
            .into_iter()
            .find(|(n, _)| n == name.as_str())
            .map(|(_, p)| p)
            .expect("the estimated name is defined");
        assert_eq!(defined, estimated_bond_params());
    }

    /// A four-carbon library whose one dihedral row `CZ-CT-CT-CW` covers no
    /// torsion of [`but_1_ene_skeleton`], with the default estimator attached
    /// (lenient mode, so it is consulted).
    ///
    /// Types, by def: `opls_za` (class `CZ`) the terminal `=C`, `opls_tb`
    /// (`CT`) the inner carbon on the double bond, `opls_tc` (`CT`) the inner
    /// carbon between two single bonds, `opls_yd` (`CY`) the terminal `-C`.
    /// Bonds and angles all have exact class rows, so the torsion is the only
    /// estimated term.
    fn one_torsion_estimating_typifier() -> OPLSAATypifier {
        let mut meta = OplsTypingMeta::new();
        let row = |class: &str, def: &str| OplsTypeRow {
            class: class.to_string(),
            def: Some(def.to_string()),
            overrides: Vec::new(),
            priority: None,
            layer: 0,
        };
        meta.insert("opls_za", row("CZ", "[C;X1]=C"));
        meta.insert("opls_tb", row("CT", "[C;X2]=C"));
        meta.insert("opls_tc", row("CT", "[C;X2](-C)-C"));
        meta.insert("opls_yd", row("CY", "[C;X1]-C"));

        let mut ff = ForceField::new("OPLS-AA");
        let atoms = ff.def_style("atom", "full", Params::new()).unwrap();
        for name in ["opls_za", "opls_tb", "opls_tc", "opls_yd"] {
            atoms
                .def_type_at(
                    name,
                    &[],
                    Params::from_pairs(&[("mass", 12.011), ("charge", 0.0)]),
                )
                .unwrap();
        }
        let bonds = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        for (name, ends) in [
            ("CZ-CT", ["CZ", "CT"]),
            ("CT-CT", ["CT", "CT"]),
            ("CT-CY", ["CT", "CY"]),
        ] {
            bonds
                .def_type_at(
                    name,
                    &ends,
                    Params::from_pairs(&[("k", 600.0), ("r0", 1.5)]),
                )
                .unwrap();
        }
        let angles = ff.def_style("angle", "harmonic", Params::new()).unwrap();
        for (name, ends) in [
            ("CZ-CT-CT", ["CZ", "CT", "CT"]),
            ("CT-CT-CY", ["CT", "CT", "CY"]),
        ] {
            angles
                .def_type_at(
                    name,
                    &ends,
                    Params::from_pairs(&[("k", 70.0), ("theta0", 2.0)]),
                )
                .unwrap();
        }
        ff.def_style("dihedral", "opls", Params::new())
            .unwrap()
            .def_type_at(
                "CZ-CT-CT-CW",
                &["CZ", "CT", "CT", "CW"],
                Params::from_pairs(&[("k1", 1.0), ("k2", 0.0), ("k3", 0.5), ("k4", 0.0)]),
            )
            .unwrap();

        let estimator = Parmchk2Estimator::new(&ff, &meta);
        OPLSAATypifier::new(meta, ff)
            .with_strict(false)
            .with_estimator(estimator)
    }

    /// The carbon skeleton `C=C-C-C` (no hydrogens), its atoms added in chain
    /// order, or in reverse chain order when `reversed`. Dihedrals are
    /// enumerated `i-j-k-l` with `j < k` in atom order, so the one torsion
    /// comes out `za-tb-tc-yd` in the first and `yd-tc-tb-za` in the second.
    fn but_1_ene_skeleton(reversed: bool) -> Atomistic {
        let mut g = Atomistic::new();
        let mut order: Vec<usize> = (0..4).collect();
        if reversed {
            order.reverse();
        }
        let mut ids = [None; 4];
        for &k in &order {
            ids[k] = Some(g.add_atom(Atom::xyz("C", 1.4 * k as f64, 0.0, 0.0)));
        }
        let c = ids.map(|id| id.expect("every chain atom added"));
        let double = g.add_bond(c[0], c[1]).unwrap();
        g.set_bond_type(double, BondType::Double).unwrap();
        g.add_bond(c[1], c[2]).unwrap();
        g.add_bond(c[2], c[3]).unwrap();
        g
    }

    /// One torsion typed from both ends defines one type: the estimator must
    /// give `za-tb-tc-yd` and `yd-tc-tb-za` the same params, because
    /// `BondedTerm::type_name` gives them the same (canonical) name.
    ///
    /// Hand-derived: against `CZ-CT-CT-CW`, reading the row forward costs one
    /// element-priced substitution (`CW` for `CY`), reading it backwards two;
    /// both leave the inner `CT-CT` pair alone, so the two readings tie on the
    /// inner score. The estimate — `estimate_penalty` included — may not
    /// depend on which end the enumeration started from.
    #[test]
    fn typing_one_estimated_torsion_from_both_ends_defines_one_type() {
        let mut typing = Typing::new(one_torsion_estimating_typifier());
        let forward = typing
            .typify(&but_1_ene_skeleton(false))
            .expect("the chain-ordered skeleton types");
        let backward = typing
            .typify(&but_1_ene_skeleton(true))
            .expect("the same torsion read from the other end is no TypeConflict");

        let torsion_type = |g: &Atomistic| -> String {
            let (_, d) = g.dihedrals().next().expect("one dihedral");
            str_prop(&d.props, "type").expect("the torsion is typed")
        };
        assert_eq!(torsion_type(&forward), torsion_type(&backward));

        let dihedrals = typing
            .forcefield()
            .get_style("dihedral", "opls")
            .expect("dihedral/opls declared")
            .defs()
            .collect_type_params();
        assert_eq!(dihedrals.len(), 1, "one torsion, one type: {dihedrals:?}");
    }

    /// Exactly one heavy atom (the O) matches no def: strict typing is an `Err`
    /// naming that atom and no other.
    #[test]
    fn strict_typify_names_the_single_untyped_heavy_atom() {
        let err = Typing::new(c_h_only_typifier())
            .typify(&methanol())
            .expect_err("strict typing must refuse the untyped oxygen");

        let oxygen = atom_label(1, "O");
        assert!(err.contains(&oxygen), "err names {oxygen}: {err}");
        for (i, el) in [(0, "C"), (2, "H"), (3, "H"), (4, "H"), (5, "H")] {
            let label = atom_label(i, el);
            assert!(
                !err.contains(&label),
                "err must not name typed {label}: {err}"
            );
        }
    }
}
