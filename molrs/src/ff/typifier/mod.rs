//! Molecular typifiers.
//!
//! Typing is a template method. A [`Typifier`] implements only
//! `r#match` ([`Typifier`]): it reads a molecular graph and produces a
//! [`Match`] — positional per-atom / per-link [`Annotation`]s, the styles they
//! are defined under, and the pair rows among the atom types used. The base,
//! [`Typing`], runs the match: [`Match::write_onto`] stamps the annotations onto
//! a private copy of the input and defines the matched types in the output
//! force field the base owns. No implementor writes the output itself.
//!
//! Materializing a typed graph into a [`Frame`](molrs::store::frame::Frame) for
//! `PotentialCompiler::compile` is the graph's `to_frame` job; typifiers stay on
//! the graph boundary.

pub mod am1bcc;
pub mod atd;
pub mod cmap;
pub mod element;
pub mod estimate;
pub mod gaff;
pub mod mmff;
pub mod opls;
pub(crate) mod topology;
pub mod uff;

pub use am1bcc::{BCCAtomChargeTypifier, BCCCorrectionTable, BCCCorrector, BccParameterSet};
pub use atd::{AtdParameterSet, AtdTypifier};
pub use cmap::assign_cmaps;
pub use element::ElementTypifier;
pub use estimate::{
    BondedTerm, Estimate, ParameterInterpolator, Parmchk2Estimator, TypifierParameterContext,
};
pub use gaff::{GaffParameterSet, GaffTypifier};
pub use opls::OPLSAATypifier;
pub use uff::UFFTypifier;

use std::collections::HashSet;

use molrs::system::atomistic::Atomistic;
use molrs::system::molgraph::{KindId, NodeId, PropValue, RelationId};

use crate::ff::forcefield::{ForceField, Params, Style};

/// A graph typifier: what matching a molecule against a force field produces.
///
/// An implementor provides `r#match` and
/// [`library`](Self::library) and nothing else; it cannot type a molecule by
/// itself. [`Typing`] owns the implementor and the output force field and is
/// the only caller of `r#match`.
///
/// The raw identifier `r#match` gives the Rust and the Python hook one name,
/// `match`.
pub trait Typifier {
    /// Match `graph` and return what it assigns.
    ///
    /// `graph` is the base's private working copy of the input: an
    /// implementation may write intermediate results onto it (generated
    /// topology, perceived bond types), and those stay on the graph
    /// [`Typing::typify`] returns. Types and parameters are not stamped here;
    /// they travel in the [`Match`], positional against this same graph.
    fn r#match(&self, graph: &mut Atomistic) -> Result<Match, String>;

    /// The force field this typifier matches against. [`Typing::new`] seeds
    /// its output from it with [`ForceField::empty_like`].
    fn library(&self) -> &ForceField;
}

impl Typifier for Box<dyn Typifier + Send + Sync> {
    fn r#match(&self, graph: &mut Atomistic) -> Result<Match, String> {
        (**self).r#match(graph)
    }

    fn library(&self) -> &ForceField {
        (**self).library()
    }
}

/// The value a [`Match`] writes under one key of one graph element.
#[derive(Debug, Clone, PartialEq)]
pub enum Annotation {
    /// Stamp `key = value`; defines nothing.
    Value(PropValue),
    /// Stamp `key = name` and every param under its own key (numeric as
    /// [`PropValue::F64`], string as [`PropValue::Str`]), and define the type
    /// `(category, style, name)` with `params`. The category follows the
    /// [`Match`] vector the annotation sits in (`nodes` → `atom`, `bonds` →
    /// `bond`, …). `endpoints` are the type's endpoint atom types, given to
    /// [`Style::def_type`] as they are (none for an atom type); `name` is
    /// never read for them.
    Type {
        style: String,
        name: String,
        endpoints: Vec<String>,
        params: Params,
    },
}

/// What a [`Typifier`] assigns to one graph.
///
/// `nodes` is positional against the graph's atoms in row order
/// (`graph.atoms()`); `bonds`, `angles`, `dihedrals` and `impropers` are
/// positional against **their own kind's** rows (`MolGraph::relation_ids(kind)`,
/// the order `graph.bonds()` etc. yield) — an improper never shifts a dihedral
/// position. An empty vector means "nothing for this kind"; an empty
/// per-element list means "this element gets nothing".
///
/// `styles` are `(category, style, style params)` to declare, in output
/// order; `pairs` are `(style, name, endpoints, params)` pair rows.
#[derive(Debug, Clone, Default)]
pub struct Match {
    pub nodes: Vec<Vec<(String, Annotation)>>,
    pub bonds: Vec<Vec<(String, Annotation)>>,
    pub angles: Vec<Vec<(String, Annotation)>>,
    pub dihedrals: Vec<Vec<(String, Annotation)>>,
    pub impropers: Vec<Vec<(String, Annotation)>>,
    pub styles: Vec<(String, String, Params)>,
    pub pairs: Vec<(String, String, Vec<String>, Params)>,
}

/// One graph element a [`Match`] writes to.
#[derive(Debug, Clone, Copy)]
enum Element {
    Node(NodeId),
    Link(KindId, RelationId),
}

/// The keys one element receives, each written once.
struct Stamp {
    element: Element,
    /// `"{category} {position}"`, for error messages.
    label: String,
    writes: Vec<(String, PropValue)>,
}

impl Stamp {
    /// Record `key = value`. The same value twice is one write; a different
    /// value under a key already written is an error.
    fn write(&mut self, key: String, value: PropValue) -> Result<(), String> {
        match self.writes.iter().find(|(k, _)| *k == key) {
            Some((_, old)) if *old == value => Ok(()),
            Some((_, old)) => Err(format!(
                "{}: key '{key}' written twice with different values ({old:?}, {value:?})",
                self.label
            )),
            None => {
                self.writes.push((key, value));
                Ok(())
            }
        }
    }

    fn apply(self, graph: &mut Atomistic) -> Result<(), String> {
        for (key, value) in self.writes {
            match self.element {
                Element::Node(id) => graph.set_atom(id, &key, value),
                Element::Link(kind, id) => graph.set_relation_prop(kind, id, &key, value),
            }
            .map_err(|e| format!("{}: {e}", self.label))?;
        }
        Ok(())
    }
}

impl Match {
    /// Run this match: stamp it onto `graph` and define it in `forcefield`.
    ///
    /// The one execution path of typing, in three phases:
    ///
    /// 1. **Validate.** A non-empty positional vector whose length differs
    ///    from the graph's count of that kind is an error naming the kind and
    ///    both counts (positions are the kind's own rows, see [`Match`]).
    ///    Every style is checked against `forcefield` and the other styles,
    ///    and every [`Annotation::Type`] and pair row against `forcefield` and
    ///    the rest of the match, by the rule [`ForceField::def_style`] /
    ///    [`Style::def_type`] apply. A type or pair row whose style is neither
    ///    in `forcefield` nor in `styles` is an error, as is one element
    ///    writing one key twice with different values.
    /// 2. **Stamp** every annotation onto `graph`. A write the graph refuses
    ///    (a value contradicting the dtype of a declared key) is an error.
    /// 3. **Commit** through [`ForceField::def_style`] (in `styles` order) and
    ///    [`Style::def_type`] (types in element order — nodes, bonds,
    ///    angles, dihedrals, impropers — then pair rows). Phase 1 makes this
    ///    infallible.
    ///
    /// # Errors
    ///
    /// On `Err`, `forcefield` is unchanged, and `graph` may be partly stamped
    /// and must be discarded. The force field is never cloned.
    pub fn write_onto(
        self,
        graph: &mut Atomistic,
        forcefield: &mut ForceField,
    ) -> Result<(), String> {
        let Match {
            nodes,
            bonds,
            angles,
            dihedrals,
            impropers,
            styles,
            pairs,
        } = self;

        // Phase 1: validate into a batch force field of this match's
        // definitions alone, and a list of stamps.
        let mut batch = ForceField::new(&forcefield.name);
        for (category, name, params) in styles {
            forcefield
                .check_style(&category, &name, &params)
                .map_err(|e| e.to_string())?;
            batch
                .def_style(&category, &name, params)
                .map_err(|e| format!("match styles: {e}"))?;
        }

        let mut stamps = Vec::new();
        for (category, kind, rows) in [
            ("atom", None, nodes),
            ("bond", Some("bonds"), bonds),
            ("angle", Some("angles"), angles),
            ("dihedral", Some("dihedrals"), dihedrals),
            ("improper", Some("impropers"), impropers),
        ] {
            if rows.is_empty() {
                continue;
            }
            let elements: Vec<Element> = match kind {
                None => graph.node_ids().map(Element::Node).collect(),
                Some(kind) => {
                    let id = graph
                        .kind_id(kind)
                        .ok_or_else(|| format!("graph has no '{kind}' relation kind"))?;
                    graph
                        .relation_ids(id)
                        .map(|r| Element::Link(id, r))
                        .collect()
                }
            };
            if rows.len() != elements.len() {
                return Err(format!(
                    "match has {} {category} annotation rows, the graph has {} {category} rows",
                    rows.len(),
                    elements.len()
                ));
            }
            for (position, (element, annotations)) in elements.into_iter().zip(rows).enumerate() {
                let mut stamp = Stamp {
                    element,
                    label: format!("{category} {position}"),
                    writes: Vec::new(),
                };
                for (key, annotation) in annotations {
                    match annotation {
                        Annotation::Value(value) => stamp.write(key, value)?,
                        Annotation::Type {
                            style,
                            name,
                            endpoints,
                            params,
                        } => {
                            for (k, v) in params.iter() {
                                stamp.write(k.to_owned(), PropValue::F64(v))?;
                            }
                            for (k, v) in params.iter_strings() {
                                stamp.write(k.to_owned(), PropValue::Str(v.to_owned()))?;
                            }
                            // An array param (a cmap `grid`) has no property
                            // form; it stays on the type `name` defines below.
                            let target =
                                Self::batch_style(&mut batch, forcefield, category, &style)?;
                            let e: Vec<&str> = endpoints.iter().map(String::as_str).collect();
                            target
                                .def_type(&name, &e, params)
                                .map_err(|e| format!("{}: {e}", stamp.label))?;
                            stamp.write(key, PropValue::Str(name))?;
                        }
                    }
                }
                stamps.push(stamp);
            }
        }

        for (style, name, endpoints, params) in pairs {
            let e: Vec<&str> = endpoints.iter().map(String::as_str).collect();
            Self::batch_style(&mut batch, forcefield, "pair", &style)?
                .def_type(&name, &e, params)
                .map_err(|e| format!("match pairs: {e}"))?;
        }

        for style in batch.styles() {
            if let Some(existing) = forcefield.get_style(style.category(), style.name()) {
                for (name, endpoints, params) in style.type_rows() {
                    existing
                        .check_type(name, &endpoints, params)
                        .map_err(|e| e.to_string())?;
                }
            }
        }

        // Phase 2: stamp.
        for stamp in stamps {
            stamp.apply(graph)?;
        }

        // Phase 3: commit. Styles come first in the batch in `styles` order;
        // a style added to the batch only for a type is already declared in
        // `forcefield`, so `def_style` returns it in place.
        for style in batch.styles() {
            let target = forcefield
                .def_style(style.category(), style.name(), style.params().clone())
                .map_err(|e| format!("commit after validation: {e}"))?;
            for (name, endpoints, params) in style.type_rows() {
                target
                    .def_type(name, &endpoints, params.clone())
                    .map_err(|e| format!("commit after validation: {e}"))?;
            }
        }
        Ok(())
    }

    /// The batch's `(category, style)` style, declared into the batch from
    /// `forcefield` when only the force field declares it; an error when
    /// neither does.
    fn batch_style<'b>(
        batch: &'b mut ForceField,
        forcefield: &ForceField,
        category: &str,
        style: &str,
    ) -> Result<&'b mut Style, String> {
        let params = match batch
            .get_style(category, style)
            .or_else(|| forcefield.get_style(category, style))
        {
            Some(declared) => declared.params().clone(),
            None => {
                return Err(format!(
                    "{category} style '{style}' is declared neither in the force field nor in \
                     the match's styles"
                ));
            }
        };
        batch
            .def_style(category, style, params)
            .map_err(|e| format!("match: {e}"))
    }

    /// Declare every style of `library`, in library order and with its style
    /// params, whether or not the match defines a type under it.
    pub(crate) fn declare_styles_of(&mut self, library: &ForceField) {
        self.styles.extend(library.styles().iter().map(|s| {
            (
                s.category().to_owned(),
                s.name().to_owned(),
                s.params().clone(),
            )
        }));
    }

    /// Add every pair row of `library` whose endpoints are all atom types in
    /// `used`: the self rows of the types used and the cross rows between
    /// them. Rows are added in library order.
    pub(crate) fn add_pairs_among(&mut self, library: &ForceField, used: &HashSet<&str>) {
        for style in library.styles().iter().filter(|s| s.category() == "pair") {
            for (name, endpoints, params) in style.type_rows() {
                if endpoints.iter().all(|e| used.contains(e)) {
                    self.pairs.push((
                        style.name().to_owned(),
                        name.to_owned(),
                        endpoints.iter().map(|e| (*e).to_owned()).collect(),
                        params.clone(),
                    ));
                }
            }
        }
    }
}

/// The base of every typifier: one [`Typifier`] plus the output force field
/// its typing accumulates.
///
/// A trait cannot hold the output, so the base is this struct. The output
/// starts as `typifier.library().empty_like()` and [`typify`](Self::typify) is
/// its only writer; there is no mutable accessor.
#[derive(Debug)]
pub struct Typing<T: Typifier> {
    typifier: T,
    output: ForceField,
}

impl<T: Typifier> Typing<T> {
    /// Wrap `typifier`, with an output seeded by
    /// [`ForceField::empty_like`] of its library: the library's name and
    /// declared units and special_bonds, no styles or types.
    pub fn new(typifier: T) -> Self {
        let output = typifier.library().empty_like();
        Self { typifier, output }
    }

    /// Type `mol`: copy it, match the copy and write the match onto the copy
    /// and the output ([`Match::write_onto`]). Returns the typed copy.
    ///
    /// `mol` is never touched. On `Err` the output is unchanged.
    pub fn typify(&mut self, mol: &Atomistic) -> Result<Atomistic, String> {
        let mut graph = mol.clone();
        let m = self.typifier.r#match(&mut graph)?;
        m.write_onto(&mut graph, &mut self.output)?;
        Ok(graph)
    }

    /// The accumulated output: exactly the definitions typing has assigned.
    pub fn forcefield(&self) -> &ForceField {
        &self.output
    }

    /// The wrapped typifier's library.
    pub fn library(&self) -> &ForceField {
        self.typifier.library()
    }

    /// The wrapped typifier.
    pub fn typifier(&self) -> &T {
        &self.typifier
    }
}

#[cfg(test)]
mod tests {
    //! `Match::write_onto` and `Typing<T>` against hand-written stub typifiers.
    //! Every expectation is written by hand; no native typifier runs here.

    use indexmap::IndexMap;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use molrs::system::atomistic::Atomistic;
    use molrs::system::molgraph::{Atom, PropValue};

    use super::*;
    use crate::ff::forcefield::tests::assert_same_definitions;
    use crate::ff::forcefield::{ForceField, Params, SpecialBonds};

    // -- stubs and fixtures ----------------------------------------------------

    /// Returns the next scripted `Match` on each call; `Err` once the script
    /// is exhausted. The `Mutex` keeps the stub `Send + Sync`.
    struct ScriptedTypifier {
        library: ForceField,
        script: Mutex<VecDeque<Match>>,
    }

    impl ScriptedTypifier {
        fn new(library: ForceField, script: Vec<Match>) -> Self {
            Self {
                library,
                script: Mutex::new(script.into()),
            }
        }
    }

    impl Typifier for ScriptedTypifier {
        fn r#match(&self, _graph: &mut Atomistic) -> Result<Match, String> {
            self.script
                .lock()
                .expect("script lock")
                .pop_front()
                .ok_or_else(|| "script exhausted".to_owned())
        }

        fn library(&self) -> &ForceField {
            &self.library
        }
    }

    /// Writes an intermediate result onto the graph it is handed (as a
    /// perception step would) and matches nothing.
    struct PerceivingTypifier {
        library: ForceField,
    }

    impl Typifier for PerceivingTypifier {
        fn r#match(&self, graph: &mut Atomistic) -> Result<Match, String> {
            let ids: Vec<_> = graph.atoms().map(|(id, _)| id).collect();
            for id in ids {
                graph
                    .set_atom(id, "perceived", true)
                    .map_err(|e| e.to_string())?;
            }
            Ok(Match::default())
        }

        fn library(&self) -> &ForceField {
            &self.library
        }
    }

    const SB: SpecialBonds = SpecialBonds {
        lj: [0.0, 0.0, 0.5],
        coul: [0.0, 0.0, 0.8333],
    };

    /// Three atoms in a chain: `c0-c1-c2`, two bonds, one angle.
    fn chain3() -> Atomistic {
        let mut g = Atomistic::new();
        let a = g.add_atom_bare("C");
        let b = g.add_atom_bare("C");
        let c = g.add_atom_bare("O");
        g.add_bond(a, b).unwrap();
        g.add_bond(b, c).unwrap();
        g.add_angle(a, b, c).unwrap();
        g
    }

    /// Two atoms, no links.
    fn pair2() -> Atomistic {
        let mut g = Atomistic::new();
        g.add_atom_bare("C");
        g.add_atom_bare("H");
        g
    }

    fn value(key: &str, v: impl Into<PropValue>) -> (String, Annotation) {
        (key.to_owned(), Annotation::Value(v.into()))
    }

    fn ty(
        key: &str,
        style: &str,
        name: &str,
        endpoints: &[&str],
        params: Params,
    ) -> (String, Annotation) {
        (
            key.to_owned(),
            Annotation::Type {
                style: style.to_owned(),
                name: name.to_owned(),
                endpoints: endpoints.iter().map(|s| (*s).to_owned()).collect(),
                params,
            },
        )
    }

    fn style(category: &str, name: &str, params: Params) -> (String, String, Params) {
        (category.to_owned(), name.to_owned(), params)
    }

    fn pair_row(
        style: &str,
        name: &str,
        endpoints: &[&str],
        params: Params,
    ) -> (String, String, Vec<String>, Params) {
        (
            style.to_owned(),
            name.to_owned(),
            endpoints.iter().map(|s| (*s).to_owned()).collect(),
            params,
        )
    }

    fn atom_full(name: &str, mass: f64) -> (String, Annotation) {
        ty(
            "type",
            "full",
            name,
            &[],
            Params::from_pairs(&[("mass", mass)]),
        )
    }

    fn atom_full_style() -> (String, String, Params) {
        style("atom", "full", Params::new())
    }

    fn nth_atom(g: &Atomistic, i: usize) -> Atom {
        g.atoms().nth(i).expect("atom index").1
    }

    fn nth_bond_props(g: &Atomistic, i: usize) -> IndexMap<String, PropValue> {
        g.bonds().nth(i).expect("bond index").1.props
    }

    /// Type names of the `(category, style)` style, in definition order.
    fn type_names(ff: &ForceField, category: &str, style: &str) -> Vec<String> {
        ff.get_style(category, style)
            .unwrap_or_else(|| panic!("no {category}/{style} style"))
            .defs()
            .collect_type_params()
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    }

    fn style_keys(ff: &ForceField) -> Vec<(&str, &str)> {
        ff.styles()
            .iter()
            .map(|s| (s.category(), s.name()))
            .collect()
    }

    /// Every atom's and every link's property bag, in row order.
    type GraphProps = (Vec<Atom>, Vec<Vec<IndexMap<String, PropValue>>>);

    fn graph_props(g: &Atomistic) -> GraphProps {
        let atoms = g.atoms().map(|(_, a)| a).collect();
        let links = vec![
            g.bonds().map(|(_, r)| r.props).collect(),
            g.angles().map(|(_, r)| r.props).collect(),
            g.dihedrals().map(|(_, r)| r.props).collect(),
            g.impropers().map(|(_, r)| r.props).collect(),
        ];
        (atoms, links)
    }

    // -- Match::write_onto: stamps ---------------------------------------------

    #[test]
    fn write_onto_stamps_a_value_on_the_atom_at_its_position_only() {
        let mut g = chain3();
        let mut ff = ForceField::new("out");
        let m = Match {
            nodes: vec![
                vec![value("class", "CT")],
                vec![],
                vec![value("aromatic_flag", true), value("ring_count", 2)],
            ],
            ..Match::default()
        };

        m.write_onto(&mut g, &mut ff).unwrap();

        assert_eq!(
            nth_atom(&g, 0).get("class"),
            Some(&PropValue::Str("CT".into()))
        );
        assert_eq!(nth_atom(&g, 1).get("class"), None);
        assert_eq!(nth_atom(&g, 2).get("class"), None);
        assert_eq!(
            nth_atom(&g, 2).get("aromatic_flag"),
            Some(&PropValue::Bool(true))
        );
        assert_eq!(nth_atom(&g, 2).get("ring_count"), Some(&PropValue::Int(2)));
    }

    /// `Type` stamps `key = name` plus every param under its own key: numeric
    /// as `F64`, string as `Str`.
    #[test]
    fn write_onto_stamps_a_type_name_and_every_param_on_its_atom() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let mut params = Params::from_pairs(&[("mass", 12.011), ("charge", -0.18)]);
        params.set_str("provenance", "hand");
        let m = Match {
            nodes: vec![vec![ty("type", "full", "CT", &[], params)], vec![]],
            styles: vec![atom_full_style()],
            ..Match::default()
        };

        m.write_onto(&mut g, &mut ff).unwrap();

        let a0 = nth_atom(&g, 0);
        assert_eq!(a0.get("type"), Some(&PropValue::Str("CT".into())));
        assert_eq!(a0.get("mass"), Some(&PropValue::F64(12.011)));
        assert_eq!(a0.get("charge"), Some(&PropValue::F64(-0.18)));
        assert_eq!(a0.get("provenance"), Some(&PropValue::Str("hand".into())));
        let a1 = nth_atom(&g, 1);
        assert_eq!(a1.get("type"), None);
        assert_eq!(a1.get("mass"), None);
    }

    #[test]
    fn write_onto_stamps_a_bond_type_on_the_bond_at_its_position_only() {
        let mut g = chain3();
        let mut ff = ForceField::new("out");
        let m = Match {
            bonds: vec![
                vec![],
                vec![ty(
                    "type",
                    "harmonic",
                    "C-O",
                    &["C", "O"],
                    Params::from_pairs(&[("k", 320.0), ("r0", 1.41)]),
                )],
            ],
            styles: vec![style("bond", "harmonic", Params::new())],
            ..Match::default()
        };

        m.write_onto(&mut g, &mut ff).unwrap();

        assert!(!nth_bond_props(&g, 0).contains_key("type"));
        let b1 = nth_bond_props(&g, 1);
        assert_eq!(b1.get("type"), Some(&PropValue::Str("C-O".into())));
        assert_eq!(b1.get("k"), Some(&PropValue::F64(320.0)));
        assert_eq!(b1.get("r0"), Some(&PropValue::F64(1.41)));
    }

    /// Dihedral and improper vectors are positional against their own kind's
    /// rows: an improper never shifts a dihedral position, and vice versa.
    #[test]
    fn write_onto_stamps_each_link_kind_on_its_own_rows() {
        let mut g = Atomistic::new();
        let ids: Vec<_> = (0..4).map(|_| g.add_atom_bare("C")).collect();
        g.add_improper(ids[1], ids[0], ids[2], ids[3]).unwrap();
        g.add_dihedral(ids[0], ids[1], ids[2], ids[3]).unwrap();
        let mut ff = ForceField::new("out");
        let m = Match {
            dihedrals: vec![vec![value("tag", "dihedral-0")]],
            impropers: vec![vec![value("tag", "improper-0")]],
            ..Match::default()
        };

        m.write_onto(&mut g, &mut ff).unwrap();

        let dihedral = g.dihedrals().next().unwrap().1.props;
        let improper = g.impropers().next().unwrap().1.props;
        assert_eq!(
            dihedral.get("tag"),
            Some(&PropValue::Str("dihedral-0".into()))
        );
        assert_eq!(
            improper.get("tag"),
            Some(&PropValue::Str("improper-0".into()))
        );
    }

    /// An empty vector means "nothing for this kind", whatever the count.
    #[test]
    fn write_onto_accepts_an_empty_vector_for_a_kind_the_graph_has() {
        let mut g = chain3();
        let mut ff = ForceField::new("out");
        let m = Match {
            nodes: vec![vec![], vec![], vec![]],
            bonds: vec![],
            angles: vec![],
            ..Match::default()
        };

        assert_eq!(m.write_onto(&mut g, &mut ff), Ok(()));
    }

    // -- Match::write_onto: definitions ------------------------------------------

    /// The output defines the match's `Type`s (one row per distinct
    /// definition) and pair rows, and nothing else.
    #[test]
    fn write_onto_defines_exactly_the_match_types_and_pair_rows() {
        let mut g = chain3();
        let mut ff = ForceField::new("out");
        let lj = |eps: f64, sigma: f64| Params::from_pairs(&[("epsilon", eps), ("sigma", sigma)]);
        let m = Match {
            nodes: vec![
                vec![atom_full("CT", 12.011)],
                vec![atom_full("CT", 12.011)],
                vec![atom_full("OH", 15.999)],
            ],
            bonds: vec![
                vec![ty(
                    "type",
                    "harmonic",
                    "CT-CT",
                    &["CT", "CT"],
                    Params::from_pairs(&[("k", 268.0), ("r0", 1.529)]),
                )],
                vec![ty(
                    "type",
                    "harmonic",
                    "CT-OH",
                    &["CT", "OH"],
                    Params::from_pairs(&[("k", 320.0), ("r0", 1.41)]),
                )],
            ],
            styles: vec![
                atom_full_style(),
                style("bond", "harmonic", Params::new()),
                style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 10.0)])),
            ],
            pairs: vec![
                pair_row("lj/cut", "CT", &["CT"], lj(0.066, 3.5)),
                pair_row("lj/cut", "OH", &["OH"], lj(0.17, 3.12)),
            ],
            ..Match::default()
        };

        m.write_onto(&mut g, &mut ff).unwrap();

        assert_eq!(ff.styles().len(), 3);
        assert_eq!(type_names(&ff, "atom", "full"), vec!["CT", "OH"]);
        assert_eq!(type_names(&ff, "bond", "harmonic"), vec!["CT-CT", "CT-OH"]);
        assert_eq!(type_names(&ff, "pair", "lj/cut"), vec!["CT", "OH"]);
        let full = ff.get_style("atom", "full").unwrap();
        assert_eq!(
            full.get_atomtype("OH").unwrap().params,
            Params::from_pairs(&[("mass", 15.999)])
        );
        let bond = ff.get_style("bond", "harmonic").unwrap();
        assert_eq!(
            bond.get_bondtype("CT", "OH").unwrap().params,
            Params::from_pairs(&[("k", 320.0), ("r0", 1.41)])
        );
        let pair = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(
            pair.get_pairtype("CT", None).unwrap().params,
            lj(0.066, 3.5)
        );
    }

    /// The given endpoints are the ones defined, under the name as given:
    /// `1-6` on `2`, `7` holds `2`, `7` — the name is never read.
    #[test]
    fn write_onto_defines_the_given_endpoints_and_never_reads_the_name() {
        let mut g = chain3();
        let mut ff = ForceField::new("out");
        let k = || Params::from_pairs(&[("kb", 4.2)]);
        let m = Match {
            bonds: vec![
                vec![ty("type", "mmff_bond", "0_1_1", &["1", "1"], k())],
                vec![ty("type", "mmff_bond", "1-6", &["2", "7"], k())],
            ],
            styles: vec![style("bond", "mmff_bond", Params::new())],
            ..Match::default()
        };

        m.write_onto(&mut g, &mut ff).unwrap();

        let s = ff.get_style("bond", "mmff_bond").unwrap();
        assert_eq!(
            s.type_endpoints("0_1_1"),
            Some(vec!["1".to_owned(), "1".to_owned()])
        );
        assert_eq!(
            s.type_endpoints("1-6"),
            Some(vec!["2".to_owned(), "7".to_owned()])
        );
    }

    #[test]
    fn write_onto_declares_styles_in_the_order_of_styles() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let m = Match {
            styles: vec![
                style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 9.0)])),
                style("dihedral", "opls", Params::new()),
                atom_full_style(),
                style("bond", "harmonic", Params::new()),
            ],
            ..Match::default()
        };

        m.write_onto(&mut g, &mut ff).unwrap();

        assert_eq!(
            style_keys(&ff),
            vec![
                ("pair", "lj/cut"),
                ("dihedral", "opls"),
                ("atom", "full"),
                ("bond", "harmonic"),
            ]
        );
        assert_eq!(
            ff.get_style("pair", "lj/cut").unwrap().params(),
            &Params::from_pairs(&[("cutoff", 9.0)])
        );
    }

    #[test]
    fn write_onto_of_a_stamp_only_match_leaves_the_forcefield_empty() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let m = Match {
            nodes: vec![vec![value("type", "c3")], vec![value("type", "hc")]],
            ..Match::default()
        };

        m.write_onto(&mut g, &mut ff).unwrap();

        assert!(ff.styles().is_empty(), "{:?}", ff.styles());
        assert_eq!(
            nth_atom(&g, 0).get("type"),
            Some(&PropValue::Str("c3".into()))
        );
        assert_eq!(
            nth_atom(&g, 1).get("type"),
            Some(&PropValue::Str("hc".into()))
        );
    }

    /// A `Type` whose style the force field already declares needs no entry
    /// in `styles`.
    #[test]
    fn write_onto_accepts_a_type_whose_style_the_forcefield_already_declares() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        ff.def_style("atom", "full", Params::new()).unwrap();
        let m = Match {
            nodes: vec![vec![atom_full("CT", 12.011)], vec![]],
            ..Match::default()
        };

        m.write_onto(&mut g, &mut ff).unwrap();

        assert_eq!(type_names(&ff, "atom", "full"), vec!["CT"]);
    }

    /// Re-defining a type the force field holds with identical params is a
    /// no-op: still one row.
    #[test]
    fn write_onto_of_an_identical_existing_type_is_a_no_op() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_pairs(&[("mass", 12.011)]))
            .unwrap();
        let before = ff.clone();
        let m = Match {
            nodes: vec![vec![atom_full("CT", 12.011)], vec![]],
            styles: vec![atom_full_style()],
            ..Match::default()
        };

        m.write_onto(&mut g, &mut ff).unwrap();

        assert_same_definitions(&ff, &before);
    }

    // -- Match::write_onto: errors leave the force field unchanged ----------------

    /// A conflicting `Type` fails the whole match: the new style and the new
    /// type that precede it in the batch do not land either.
    #[test]
    fn write_onto_type_conflicting_with_the_forcefield_errs_and_changes_nothing() {
        let mut g = chain3();
        let mut ff = ForceField::new("out");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_pairs(&[("mass", 12.011)]))
            .unwrap();
        let before = ff.clone();
        let m = Match {
            nodes: vec![
                vec![atom_full("OH", 15.999)],
                vec![atom_full("CT", 12.0)],
                vec![],
            ],
            bonds: vec![],
            styles: vec![style("bond", "harmonic", Params::new()), atom_full_style()],
            ..Match::default()
        };

        let result = m.write_onto(&mut g, &mut ff);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(&ff, &before);
    }

    /// Two elements of one match defining one name differently is a
    /// conflict within the batch.
    #[test]
    fn write_onto_two_conflicting_types_within_one_match_err_and_change_nothing() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let before = ff.clone();
        let m = Match {
            nodes: vec![vec![atom_full("CT", 12.011)], vec![atom_full("CT", 12.0)]],
            styles: vec![atom_full_style()],
            ..Match::default()
        };

        let result = m.write_onto(&mut g, &mut ff);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn write_onto_style_conflicting_with_the_forcefield_errs_and_changes_nothing() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        ff.def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap();
        let before = ff.clone();
        let m = Match {
            styles: vec![
                atom_full_style(),
                style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 12.0)])),
            ],
            ..Match::default()
        };

        let result = m.write_onto(&mut g, &mut ff);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn write_onto_type_under_an_undeclared_style_errs_and_changes_nothing() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let before = ff.clone();
        let m = Match {
            nodes: vec![vec![atom_full("CT", 12.011)], vec![]],
            styles: vec![style("pair", "lj/cut", Params::new())],
            ..Match::default()
        };

        let result = m.write_onto(&mut g, &mut ff);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn write_onto_pair_row_under_an_undeclared_style_errs_and_changes_nothing() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let before = ff.clone();
        let m = Match {
            styles: vec![atom_full_style()],
            pairs: vec![pair_row(
                "lj/cut",
                "CT",
                &["CT"],
                Params::from_pairs(&[("epsilon", 0.066), ("sigma", 3.5)]),
            )],
            ..Match::default()
        };

        let result = m.write_onto(&mut g, &mut ff);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(&ff, &before);
    }

    /// A non-empty node vector must have one entry per atom; the error names
    /// both counts.
    #[test]
    fn write_onto_node_vector_of_the_wrong_length_errs_naming_both_counts() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let before = ff.clone();
        let m = Match {
            nodes: vec![
                vec![atom_full("CT", 12.011)],
                vec![],
                vec![],
                vec![],
                vec![],
            ],
            styles: vec![atom_full_style()],
            ..Match::default()
        };

        let err = m.write_onto(&mut g, &mut ff).unwrap_err();

        assert!(err.contains('5') && err.contains('2'), "{err}");
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn write_onto_bond_vector_of_the_wrong_length_errs_naming_kind_and_counts() {
        let mut g = chain3();
        let mut ff = ForceField::new("out");
        let before = ff.clone();
        let m = Match {
            bonds: vec![vec![], vec![], vec![], vec![]],
            ..Match::default()
        };

        let err = m.write_onto(&mut g, &mut ff).unwrap_err();

        assert!(err.contains("bond"), "{err}");
        assert!(err.contains('4') && err.contains('2'), "{err}");
        assert_same_definitions(&ff, &before);
    }

    /// A `Type` param and a `Value` both writing `charge` on one atom, with
    /// different values.
    #[test]
    fn write_onto_one_key_written_twice_with_different_values_errs() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let before = ff.clone();
        let m = Match {
            nodes: vec![
                vec![
                    ty(
                        "type",
                        "full",
                        "CT",
                        &[],
                        Params::from_pairs(&[("charge", 0.5)]),
                    ),
                    value("charge", -0.5),
                ],
                vec![],
            ],
            styles: vec![atom_full_style()],
            ..Match::default()
        };

        let result = m.write_onto(&mut g, &mut ff);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(&ff, &before);
    }

    /// Only *different* values collide; the same value written twice is one
    /// write.
    #[test]
    fn write_onto_one_key_written_twice_with_the_same_value_is_accepted() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let m = Match {
            nodes: vec![
                vec![
                    ty(
                        "type",
                        "full",
                        "CT",
                        &[],
                        Params::from_pairs(&[("charge", 0.5)]),
                    ),
                    value("charge", 0.5),
                ],
                vec![],
            ],
            styles: vec![atom_full_style()],
            ..Match::default()
        };

        m.write_onto(&mut g, &mut ff).unwrap();

        assert_eq!(nth_atom(&g, 0).get("charge"), Some(&PropValue::F64(0.5)));
        assert_eq!(type_names(&ff, "atom", "full"), vec!["CT"]);
    }

    /// `mass` is declared float by the Frame schema, so a string `mass` is
    /// refused at the stamp. The valid `Type` on the other atom is not
    /// defined: stamping precedes the commit.
    #[test]
    fn write_onto_stamp_the_graph_refuses_errs_and_changes_nothing() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let before = ff.clone();
        let m = Match {
            nodes: vec![vec![atom_full("CT", 12.011)], vec![value("mass", "heavy")]],
            styles: vec![atom_full_style()],
            ..Match::default()
        };

        let result = m.write_onto(&mut g, &mut ff);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(&ff, &before);
    }

    // -- Typing -----------------------------------------------------------------

    fn library() -> ForceField {
        let mut lib = ForceField::new("lib");
        lib.set_units("metal");
        lib.set_special_bonds(SB);
        lib.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_pairs(&[("mass", 12.011)]))
            .unwrap()
            .def_type("UNUSED", &[], Params::from_pairs(&[("mass", 1.0)]))
            .unwrap();
        lib
    }

    /// `CT` on atom 0, nothing on atom 1, under the declared `atom/full`.
    fn ct_match(mass: f64) -> Match {
        Match {
            nodes: vec![vec![atom_full("CT", mass)], vec![]],
            styles: vec![atom_full_style()],
            ..Match::default()
        }
    }

    #[test]
    fn typing_new_seeds_the_output_with_the_library_name_units_and_special_bonds() {
        let typing = Typing::new(ScriptedTypifier::new(library(), vec![]));

        let out = typing.forcefield();

        assert_eq!(out.name, "lib");
        assert_eq!(out.declared_units(), Some("metal"));
        assert_eq!(out.declared_special_bonds(), Some(&SB));
        assert!(out.styles().is_empty(), "{:?}", out.styles());
    }

    #[test]
    fn typing_library_and_typifier_return_the_wrapped_typifier_and_its_library() {
        let typing = Typing::new(ScriptedTypifier::new(library(), vec![]));

        assert_same_definitions(typing.library(), &library());
        assert!(std::ptr::eq(typing.library(), typing.typifier().library()));
    }

    #[test]
    fn typify_returns_a_stamped_copy_and_leaves_the_input_unchanged() {
        let mol = pair2();
        let before = graph_props(&mol);
        let mut typing = Typing::new(ScriptedTypifier::new(library(), vec![ct_match(12.011)]));

        let typed = typing.typify(&mol).unwrap();

        assert_eq!(graph_props(&mol), before);
        assert_eq!(
            nth_atom(&typed, 0).get("type"),
            Some(&PropValue::Str("CT".into()))
        );
        assert_eq!(
            nth_atom(&typed, 0).get("mass"),
            Some(&PropValue::F64(12.011))
        );
    }

    /// `match` receives the private working copy, and that copy is what
    /// `typify` returns: an intermediate result written by `match` is on the
    /// returned graph and not on the input.
    #[test]
    fn typify_hands_match_the_working_copy_it_returns() {
        let mol = pair2();
        let mut typing = Typing::new(PerceivingTypifier {
            library: ForceField::new("lib"),
        });

        let typed = typing.typify(&mol).unwrap();

        assert_eq!(
            nth_atom(&typed, 1).get("perceived"),
            Some(&PropValue::Bool(true))
        );
        assert_eq!(nth_atom(&mol, 1).get("perceived"), None);
    }

    /// The output holds the types the match assigned, not the library's
    /// unused `UNUSED` row.
    #[test]
    fn typify_defines_the_matched_types_in_the_output_only() {
        let mut typing = Typing::new(ScriptedTypifier::new(library(), vec![ct_match(12.011)]));

        typing.typify(&pair2()).unwrap();

        assert_eq!(type_names(typing.forcefield(), "atom", "full"), vec!["CT"]);
    }

    #[test]
    fn typify_of_a_stamp_only_match_leaves_the_output_empty() {
        let stamp_only = Match {
            nodes: vec![vec![value("type", "c3")], vec![value("type", "hc")]],
            ..Match::default()
        };
        let mut typing = Typing::new(ScriptedTypifier::new(library(), vec![stamp_only]));

        let typed = typing.typify(&pair2()).unwrap();

        assert!(typing.forcefield().styles().is_empty());
        assert_eq!(
            nth_atom(&typed, 1).get("type"),
            Some(&PropValue::Str("hc".into()))
        );
    }

    #[test]
    fn typify_of_an_identical_redefinition_across_two_calls_is_a_no_op() {
        let mut typing = Typing::new(ScriptedTypifier::new(
            library(),
            vec![ct_match(12.011), ct_match(12.011)],
        ));
        typing.typify(&pair2()).unwrap();
        let after_first = typing.forcefield().clone();

        typing.typify(&pair2()).unwrap();

        assert_same_definitions(typing.forcefield(), &after_first);
    }

    #[test]
    fn typify_of_a_conflicting_type_errs_and_leaves_output_and_input_unchanged() {
        let mut typing = Typing::new(ScriptedTypifier::new(
            library(),
            vec![ct_match(12.011), ct_match(12.0)],
        ));
        typing.typify(&pair2()).unwrap();
        let after_first = typing.forcefield().clone();
        let mol = pair2();
        let mol_before = graph_props(&mol);

        let result = typing.typify(&mol);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(typing.forcefield(), &after_first);
        assert_eq!(graph_props(&mol), mol_before);
    }

    #[test]
    fn typify_propagates_a_match_error_and_leaves_the_output_unchanged() {
        let mut typing = Typing::new(ScriptedTypifier::new(library(), vec![]));
        let before = typing.forcefield().clone();

        let result = typing.typify(&pair2());

        assert_eq!(result.err().as_deref(), Some("script exhausted"));
        assert_same_definitions(typing.forcefield(), &before);
    }

    /// `Box<dyn Typifier + Send + Sync>` is itself a `Typifier`, so one
    /// `Typing` type can hold any typifier chosen at run time.
    #[test]
    fn boxed_dyn_typifier_is_a_typifier() {
        let boxed: Box<dyn Typifier + Send + Sync> =
            Box::new(ScriptedTypifier::new(library(), vec![ct_match(12.011)]));
        let mut typing: Typing<Box<dyn Typifier + Send + Sync>> = Typing::new(boxed);

        typing.typify(&pair2()).unwrap();

        assert_eq!(typing.library().name, "lib");
        assert_eq!(type_names(typing.forcefield(), "atom", "full"), vec!["CT"]);
    }
}
