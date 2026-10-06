//! The registry's protocol: what it accepts, what it refuses and why, and
//! that the built-ins are registered through it.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::ff::forcefield::{ForceField, Params};
use crate::ff::ir::{
    Arity, CategorySpec, Coordinate, Dim, EndpointOrder, ExpressionForm, ExpressionKernel, IrError,
    Kernel, Mix, ParamCols, ParamKind, ParamSpec, Registry, RowSource, Sample, ScalarForm,
    SpecialClass, StyleSpec, Value, builtin_categories, builtin_styles,
};
use crate::ff::potential::bond::harmonic::bond_harmonic_ctor;
use crate::ff::potential::{KernelRegistry, PotentialCompiler, register_kernel};
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::types::{F, Idx};
use ndarray::Array1;

/// `scale · k (q − q0)²`, its derivative off by `wrong` (1 is right).
struct Harmonic {
    scale: F,
    wrong: F,
}

const RIGHT: Harmonic = Harmonic {
    scale: 1.0,
    wrong: 1.0,
};

impl ScalarForm for Harmonic {
    fn eval(&self, q: &[F], p: &ParamCols<'_>, e: &mut [F], de: &mut [F]) {
        let (k, q0) = (p.get("k").unwrap(), p.get("r0").unwrap());
        for t in 0..q.len() {
            let d = q[t] - q0[t];
            e[t] = self.scale * k[t] * d * d;
            de[t] = self.wrong * 2.0 * self.scale * k[t] * d;
        }
    }
}

/// `k q1 q2 / r` — or, `asymmetric`, `k q1 / r`.
struct Charges {
    asymmetric: bool,
}

impl ScalarForm for Charges {
    fn eval(&self, r: &[F], p: &ParamCols<'_>, e: &mut [F], de: &mut [F]) {
        let (k, q1, q2) = (
            p.get("k").unwrap(),
            p.get("q1").unwrap(),
            p.get("q2").unwrap(),
        );
        for t in 0..r.len() {
            let qq = if self.asymmetric {
                q1[t]
            } else {
                q1[t] * q2[t]
            };
            e[t] = k[t] * qq / r[t];
            de[t] = -e[t] / r[t];
        }
    }
}

fn harmonic_spec(category: &'static str, name: &'static str) -> StyleSpec {
    StyleSpec::new(category, name).params(vec![
        ParamSpec::new("k", "E/L^2".parse().unwrap()),
        ParamSpec::new("r0", Dim::LENGTH),
    ])
}

fn sample() -> Sample {
    Sample {
        params: vec![
            ("k".into(), Value::Num(300.0)),
            ("r0".into(), Value::Num(1.5)),
        ],
        q: (0.9, 2.5),
    }
}

fn scalar(form: Harmonic) -> Option<Kernel> {
    Some(Kernel::Scalar(Arc::new(form)))
}

/// An expression-engine stand-in: what `ff::ir::expr` hands the registry,
/// with the variables and form chosen by the test.
struct FakeExpression {
    source: String,
    variables: Vec<String>,
    form: Arc<dyn ScalarForm>,
}

impl ExpressionKernel for FakeExpression {
    fn source(&self) -> &str {
        &self.source
    }
    fn variables(&self) -> Vec<String> {
        self.variables.clone()
    }
    fn form(&self) -> ExpressionForm {
        ExpressionForm::Scalar(self.form.clone())
    }
}

fn expression(variables: &[&str], form: Harmonic) -> Option<Kernel> {
    Some(Kernel::Expression(Arc::new(FakeExpression {
        source: "k*(r-r0)^2".into(),
        variables: variables.iter().map(|v| v.to_string()).collect(),
        form: Arc::new(form),
    })))
}

#[test]
fn every_builtin_kernel_has_its_spec() {
    let kernels: BTreeSet<(String, String)> = KernelRegistry::builtin()
        .names()
        .map(|(c, n)| (c.to_owned(), n.to_owned()))
        .collect();
    let specs: BTreeSet<(String, String)> = builtin_styles()
        .into_iter()
        .map(|s| (s.category.into_owned(), s.name.into_owned()))
        .collect();
    let without_kernel: BTreeSet<(String, String)> = specs.difference(&kernels).cloned().collect();
    assert!(
        kernels.is_subset(&specs),
        "{:?}",
        kernels.difference(&specs)
    );
    let expected: BTreeSet<(String, String)> = [
        ("atom", "full"),
        ("constraint", "fixed"),
        ("dihedral", "rb"),
        ("drude", "harmonic"),
        ("virtual_site", "average2"),
        ("virtual_site", "average3"),
        ("virtual_site", "outofplane3"),
    ]
    .iter()
    .map(|(c, n)| (c.to_string(), n.to_string()))
    .collect();
    assert_eq!(without_kernel, expected);
    let r = Registry::builtin();
    assert_eq!(r.styles(None).count(), specs.len());
    assert_eq!(r.categories().count(), builtin_categories().len());
    assert!(
        r.styles(None)
            .all(|(s, _)| r.is_sealed(&s.category, &s.name))
    );
    // `dihedral rb` is pure protocol: a built-in priced by its expression.
    let (rb, kernel) = r.style("dihedral", "rb").unwrap();
    assert!(kernel.is_none() && rb.expression.is_some());
}

/// The built-in table states each style's per-type parameters in the order
/// a LAMMPS `*_coeff` line lists them, with the protocol's dimensions.
#[test]
fn builtin_params_are_lammps_coeff_order_with_dims() {
    let r = Registry::builtin();
    let spec = |c: &str, s: &str| r.style(c, s).unwrap().0.clone();
    let order = |c: &str, s: &str| -> Vec<String> {
        spec(c, s)
            .params
            .iter()
            .map(|p| p.name.to_string())
            .collect()
    };
    assert_eq!(order("bond", "harmonic"), ["k", "r0"]);
    assert_eq!(order("bond", "class2"), ["r0", "k2", "k3", "k4"]);
    assert_eq!(order("angle", "charmm"), ["k", "theta0", "k_ub", "r_ub"]);
    assert_eq!(
        order("dihedral", "charmm"),
        ["k", "periodicity", "phase", "w"]
    );
    assert_eq!(order("pair", "buck"), ["a", "rho", "c"]);
    assert_eq!(
        order("pair", "lj/charmm"),
        ["epsilon", "sigma", "epsilon14", "sigma14"]
    );
    let angle = spec("angle", "harmonic");
    assert_eq!(
        angle.param("theta0").unwrap().dim,
        Dim::ANGLE,
        "an angle value"
    );
    assert_eq!(
        angle.param("k").unwrap().dim.to_string(),
        "E/A^2",
        "per radian"
    );
    assert_eq!(spec("atom", "full").param("mass").unwrap().dim, Dim::MASS);
    let periodic = spec("dihedral", "periodic");
    assert!(periodic.unindexed_one_term && periodic.params.iter().all(|p| p.indexed));
    assert!(!spec("pair", "coul/charmm").force_is_gradient);
    assert_eq!(
        spec("pair", "lj/cut").param("epsilon").unwrap().mix,
        Mix::LjEpsilon {
            sigma: "sigma".into()
        }
    );
    assert_eq!(
        spec("angle", "harmonic").expression.as_deref(),
        Some("k*(theta-theta0*0.017453292519943295)^2")
    );
}

#[test]
fn a_builtin_is_sealed_and_an_identical_restatement_is_a_no_op() {
    let mut r = Registry::builtin();
    let (spec, kernel) = r.style("bond", "harmonic").unwrap();
    let (spec, kernel) = (spec.clone(), kernel.cloned());
    assert_eq!(r.register_style(spec, kernel), Ok(()));
    let err = r
        .register_style(
            StyleSpec::new("bond", "harmonic"),
            Some(Kernel::ctor(bond_harmonic_ctor)),
        )
        .unwrap_err();
    assert!(matches!(err, IrError::Sealed { .. }), "{err}");
    assert!(matches!(
        r.unregister_style("bond", "harmonic"),
        Err(IrError::Sealed { .. })
    ));
    // The global registry seals them too, through the old shim.
    assert!(matches!(
        register_kernel("bond", "harmonic", bond_harmonic_ctor),
        Err(IrError::Sealed { .. })
    ));
    // A built-in category likewise.
    let mut bond = r.category("bond").unwrap().clone();
    assert_eq!(r.register_category(bond.clone()), Ok(()));
    bond.order = EndpointOrder::Ordered;
    assert!(matches!(
        r.register_category(bond),
        Err(IrError::Sealed { .. })
    ));
}

#[test]
fn a_custom_style_conflicts_unless_restated_identically() {
    let mut r = Registry::builtin();
    let kernel = scalar(RIGHT);
    r.register_style(harmonic_spec("bond", "mine"), kernel.clone())
        .unwrap();
    assert_eq!(
        r.register_style(harmonic_spec("bond", "mine"), kernel),
        Ok(())
    );
    assert!(matches!(
        r.register_style(harmonic_spec("bond", "mine"), scalar(RIGHT)),
        Err(IrError::Conflict { .. })
    ));
    assert_eq!(r.unregister_style("bond", "mine"), Ok(()));
    assert!(matches!(
        r.unregister_style("bond", "mine"),
        Err(IrError::NoKernel { .. })
    ));
}

#[test]
fn custom_categories_follow_the_custom_rules() {
    let mut r = Registry::builtin();
    let ub = CategorySpec::custom(
        "urey_bradley",
        3,
        Coordinate::Compound,
        EndpointOrder::Reversible,
    );
    assert_eq!(ub.block, "urey_bradleys");
    assert_eq!(r.register_category(ub.clone()), Ok(()));
    assert_eq!(r.register_category(ub.clone()), Ok(()));
    let mut other = ub.clone();
    other.order = EndpointOrder::Ordered;
    assert!(matches!(
        r.register_category(other),
        Err(IrError::Conflict { .. })
    ));

    let bad = |c: CategorySpec| Registry::builtin().register_category(c).unwrap_err();
    assert!(matches!(
        bad(CategorySpec::custom(
            "Bad",
            2,
            Coordinate::Compound,
            EndpointOrder::Ordered
        )),
        IrError::BadName { .. }
    ));
    for arity in [1, 6] {
        assert!(matches!(
            bad(CategorySpec::custom(
                "x",
                arity,
                Coordinate::Compound,
                EndpointOrder::Ordered
            )),
            IrError::Arity { .. }
        ));
    }
    let mut elsewhere = CategorySpec::custom("x", 2, Coordinate::Distance, EndpointOrder::Ordered);
    elsewhere.block = "things".into();
    assert!(matches!(bad(elsewhere), IrError::BlockName { .. }));
    assert!(matches!(
        bad(CategorySpec::custom(
            "x",
            2,
            Coordinate::Angle,
            EndpointOrder::Ordered
        )),
        IrError::CoordinateMismatch { .. }
    ));
    assert!(matches!(
        bad(CategorySpec::new(
            "x",
            Arity::SelfOrPair,
            "xs",
            Coordinate::Distance,
            EndpointOrder::Unordered
        )),
        IrError::Arity { .. }
    ));
}

#[test]
fn nonconforming_styles_are_refused_by_name() {
    let mut r = Registry::builtin();

    let err = r
        .register_style(harmonic_spec("bend", "x"), scalar(RIGHT))
        .unwrap_err();
    assert!(matches!(err, IrError::UnknownCategory { .. }), "{err}");

    for reserved in [
        "name", "type", "style", "itom", "atomj", "desc", "theta", "chi", "p3",
    ] {
        let spec = StyleSpec::new("bond", "x").params(vec![ParamSpec::new(reserved, Dim::NONE)]);
        assert_eq!(
            r.register_style(spec, Some(Kernel::ctor(bond_harmonic_ctor))),
            Err(IrError::ReservedParam {
                style: "x".into(),
                param: reserved.into()
            }),
            "{reserved}"
        );
    }
    // A pair binds `q1`, `q2` and `<param>1`, `<param>2` itself.
    for reserved in ["q", "q2", "k1"] {
        let spec = StyleSpec::new("pair", "x").params(vec![
            ParamSpec::new("k", Dim::ENERGY),
            ParamSpec::new(reserved, Dim::NONE),
        ]);
        assert!(
            matches!(
                r.register_style(spec, Some(Kernel::ctor(bond_harmonic_ctor))),
                Err(IrError::ReservedParam { .. })
            ),
            "{reserved}"
        );
    }
    // A function name is no variable: a call always has its parenthesis.
    for function in ["exp", "delta", "select"] {
        let spec =
            StyleSpec::new("bond", function).params(vec![ParamSpec::new(function, Dim::NONE)]);
        r.register_style(spec, Some(Kernel::ctor(bond_harmonic_ctor)))
            .unwrap();
    }
    assert!(matches!(
        r.register_style(
            StyleSpec::new("bond", "x").params(vec![ParamSpec::new("2k", Dim::NONE)]),
            Some(Kernel::ctor(bond_harmonic_ctor))
        ),
        Err(IrError::BadName { .. })
    ));

    let twice = StyleSpec::new("bond", "x")
        .params(vec![ParamSpec::new("k", Dim::ENERGY)])
        .style_params(vec![ParamSpec::new("k", Dim::ENERGY)]);
    assert!(matches!(
        r.register_style(twice, Some(Kernel::ctor(bond_harmonic_ctor))),
        Err(IrError::DuplicateParam { .. })
    ));

    let per_degree =
        StyleSpec::new("bond", "x").params(vec![ParamSpec::new("k", "E*A".parse().unwrap())]);
    assert!(matches!(
        r.register_style(per_degree, Some(Kernel::ctor(bond_harmonic_ctor))),
        Err(IrError::Dim { .. })
    ));

    // A scalar form on a compound category; a compound form on a pair is
    // the mirror case.
    let err = r
        .register_style(harmonic_spec("cmap", "x"), scalar(RIGHT))
        .unwrap_err();
    assert!(matches!(err, IrError::CoordinateMismatch { .. }), "{err}");

    // Any kernel on a category that prices nothing.
    let err = r
        .register_style(harmonic_spec("constraint", "x"), scalar(RIGHT))
        .unwrap_err();
    assert!(matches!(err, IrError::CoordinateMismatch { .. }), "{err}");

    // Neither a kernel nor an expression.
    assert!(matches!(
        r.register_style(harmonic_spec("bond", "x"), None),
        Err(IrError::NoKernel { .. })
    ));

    // `cutoff` keeps its reserved meaning.
    let cutoff =
        harmonic_spec("pair", "x").style_params(vec![ParamSpec::new("cutoff", Dim::ENERGY)]);
    assert!(matches!(
        r.register_style(cutoff, scalar(RIGHT)),
        Err(IrError::ReservedParam { .. })
    ));

    // A neighbour-driven form on a bonded category.
    let typed = Kernel::Ctor {
        compiled: bond_harmonic_ctor,
        typed: Some((bond_harmonic_ctor, SpecialClass::Vdw)),
        rows: RowSource::CategoryBlock,
    };
    assert!(matches!(
        r.register_style(StyleSpec::new("bond", "x"), Some(typed)),
        Err(IrError::CoordinateMismatch { .. })
    ));

    // A mixing rule on a bond.
    let mixing = StyleSpec::new("bond", "x")
        .params(vec![ParamSpec::new("k", Dim::ENERGY).mix(Mix::Geometric)]);
    assert!(matches!(
        r.register_style(mixing, Some(Kernel::ctor(bond_harmonic_ctor))),
        Err(IrError::Malformed { .. })
    ));

    // Nothing is left behind by a refusal.
    assert!(r.style("bond", "x").is_none());
}

#[test]
fn a_form_is_checked_on_its_samples_at_registration() {
    let mut r = Registry::builtin();
    let half = Harmonic {
        scale: 1.0,
        wrong: 0.5,
    };
    let err = r
        .register_style(harmonic_spec("bond", "x").sample(sample()), scalar(half))
        .unwrap_err();
    assert!(matches!(err, IrError::Derivative { .. }), "{err}");
    r.register_style(harmonic_spec("bond", "x").sample(sample()), scalar(RIGHT))
        .unwrap();

    // A pair energy must not change when its two atoms are exchanged.
    let coulomb = |name: &'static str| {
        StyleSpec::new("pair", name)
            .params(vec![
                ParamSpec::new("k", "E*L/Q^2".parse().unwrap()).mix(Mix::Geometric),
            ])
            .sample(Sample {
                params: vec![("k".into(), Value::Num(332.0))],
                q: (1.0, 5.0),
            })
    };
    let err = r
        .register_style(
            coulomb("lopsided"),
            Some(Kernel::Scalar(Arc::new(Charges { asymmetric: true }))),
        )
        .unwrap_err();
    assert!(matches!(err, IrError::Asymmetric { .. }), "{err}");
    r.register_style(
        coulomb("coulomb"),
        Some(Kernel::Scalar(Arc::new(Charges { asymmetric: false }))),
    )
    .unwrap();
}

/// A bond of two atoms 1.6 apart, typed `A-A`, and a force field defining
/// `bond <name>` over it.
fn one_bond(name: &str) -> (ForceField, Frame) {
    let mut ff = ForceField::new("t");
    ff.def_style("bond", name, Params::new())
        .unwrap()
        .def_type(
            "A-A",
            &["A", "A"],
            Params::from_pairs(&[("k", 2.0), ("r0", 1.0)]),
        )
        .unwrap();
    let mut atoms = Block::new();
    for (key, v) in [("x", [0.0, 1.6]), ("y", [0.0, 0.0]), ("z", [0.0, 0.0])] {
        atoms
            .insert(key, Array1::from_vec(v.to_vec()).into_dyn())
            .unwrap();
    }
    let mut bonds = Block::new();
    bonds
        .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn())
        .unwrap();
    bonds
        .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn())
        .unwrap();
    bonds
        .insert("type", Array1::from_vec(vec!["A-A".to_string()]).into_dyn())
        .unwrap();
    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame.insert("bonds", bonds);
    (ff, frame)
}

/// Without samples the check runs at the first compile, on real terms, and
/// its verdict stands for every later compile.
#[test]
fn a_form_without_samples_is_checked_at_its_first_compile() {
    let mut r = Registry::builtin();
    let half = Harmonic {
        scale: 1.0,
        wrong: 0.5,
    };
    r.register_style(harmonic_spec("bond", "unchecked"), scalar(half))
        .unwrap();
    let (ff, frame) = one_bond("unchecked");
    for _ in 0..2 {
        let err = PotentialCompiler::with_registry(&ff, &r)
            .compile(&frame)
            .unwrap_err();
        assert!(err.contains("central difference"), "{err}");
    }
}

#[test]
fn an_expression_reads_only_what_its_style_declares() {
    let mut r = Registry::builtin();
    r.register_style(
        harmonic_spec("bond", "expr"),
        expression(&["r", "k", "r0"], RIGHT),
    )
    .unwrap();
    assert_eq!(
        r.register_style(
            harmonic_spec("bond", "x"),
            expression(&["theta", "k"], RIGHT)
        ),
        Err(IrError::UnboundVariable {
            style: "x".into(),
            name: "theta".into()
        })
    );
    // A pair binds the per-atom inputs as well; an improper `phi` and `chi`.
    r.register_style(
        harmonic_spec("pair", "expr"),
        expression(&["r", "k1", "r02", "q1"], RIGHT),
    )
    .unwrap();
    r.register_style(
        harmonic_spec("improper", "expr"),
        expression(&["phi", "chi", "k"], RIGHT),
    )
    .unwrap();
    // An indexed family by any member.
    let indexed = StyleSpec::new("dihedral", "series").params(vec![
        ParamSpec::new("k", Dim::ENERGY).indexed(),
        ParamSpec::new("r0", Dim::NONE),
    ]);
    r.register_style(indexed, expression(&["phi", "k1", "k7"], RIGHT))
        .unwrap();

    // The compiled expression is the spec's, byte for byte.
    let spec = harmonic_spec("bond", "y").expression("k*(r - r0)^2");
    assert!(matches!(
        r.register_style(spec, expression(&["r"], RIGHT)),
        Err(IrError::Malformed { .. })
    ));
}

/// The expression engine is reached through [`ExpressionCompiler`]: an
/// expression declared beside a native form must price what it prices, and
/// an expression-only style is priced by it.
///
/// [`ExpressionCompiler`]: crate::ff::ir::ExpressionCompiler
#[test]
fn an_expression_beside_a_native_form_must_agree_with_it() {
    fn agreeing(_: &CategorySpec, s: &StyleSpec) -> Result<Arc<dyn ExpressionKernel>, IrError> {
        Ok(Arc::new(FakeExpression {
            source: s.expression.clone().unwrap(),
            variables: vec!["r".into(), "k".into(), "r0".into()],
            form: Arc::new(RIGHT),
        }))
    }
    fn off_by_1e9(_: &CategorySpec, s: &StyleSpec) -> Result<Arc<dyn ExpressionKernel>, IrError> {
        Ok(Arc::new(FakeExpression {
            source: s.expression.clone().unwrap(),
            variables: vec!["r".into()],
            form: Arc::new(Harmonic {
                scale: 1.0 + 1e-9,
                wrong: 1.0,
            }),
        }))
    }
    let spec = || {
        harmonic_spec("bond", "native")
            .expression("k*(r-r0)^2")
            .sample(sample())
    };
    let mut r = Registry::builtin();
    r.set_expression_compiler(Some(agreeing));
    r.register_style(spec(), scalar(RIGHT)).unwrap();

    // Expression only: priced by the engine.
    r.register_style(
        harmonic_spec("bond", "by/expression").expression("k*(r-r0)^2"),
        None,
    )
    .unwrap();
    let (ff, frame) = one_bond("by/expression");
    let pots = PotentialCompiler::with_registry(&ff, &r)
        .compile(&frame)
        .unwrap();
    assert!((pots.calc_energy(&[0.0, 0.0, 0.0, 1.6, 0.0, 0.0]) - 2.0 * 0.36).abs() < 1e-12);

    let mut r = Registry::builtin();
    r.set_expression_compiler(Some(off_by_1e9));
    let err = r.register_style(spec(), scalar(RIGHT)).unwrap_err();
    assert!(matches!(err, IrError::Disagree { .. }), "{err}");

    // Without an engine nothing about the expression can be checked, and an
    // expression-only style is priced by nothing.
    let mut r = Registry::builtin();
    r.register_style(spec(), scalar(RIGHT)).unwrap();
    r.register_style(
        harmonic_spec("bond", "by/expression").expression("k*(r-r0)^2"),
        None,
    )
    .unwrap();
    let err = PotentialCompiler::with_registry(&ff, &r)
        .compile(&frame)
        .unwrap_err();
    assert!(err.contains("no kernel for bond `by/expression`"), "{err}");
}

/// A compile against a registry without the category names it.
#[test]
fn compiling_an_unregistered_category_is_an_error() {
    let (ff, _) = one_bond("harmonic");
    let empty = Registry::new();
    let err = PotentialCompiler::with_registry(&ff, &empty)
        .compile(&Frame::new())
        .unwrap_err();
    assert!(err.contains("category 'bond' is not registered"), "{err}");
}

/// A style registered in one registry is seen by a compile against it and
/// by nothing else.
#[test]
fn a_registry_of_ones_own_is_seen_by_its_compile_only() {
    let (ff, frame) = one_bond("only/here");
    let mut r = Registry::builtin();
    r.register_style(harmonic_spec("bond", "only/here"), scalar(RIGHT))
        .unwrap();
    let pots = PotentialCompiler::with_registry(&ff, &r)
        .compile(&frame)
        .unwrap();
    // k (r − r0)² = 2 · 0.6²
    assert!((pots.calc_energy(&[0.0, 0.0, 0.0, 1.6, 0.0, 0.0]) - 0.72).abs() < 1e-12);
    let err = PotentialCompiler::new(&ff).compile(&frame).unwrap_err();
    assert!(err.contains("no kernel"), "{err}");
}

/// A text style parameter is checked to be text, and an array parameter is
/// accepted by a bonded form (it reads it through `ParamCols::array`).
#[test]
fn text_and_array_params_are_declared_by_kind() {
    let mut r = Registry::builtin();
    let mixing = harmonic_spec("pair", "x").style_params(vec![ParamSpec::new("mixing", Dim::NONE)]);
    assert!(matches!(
        r.register_style(mixing, scalar(RIGHT)),
        Err(IrError::ReservedParam { .. })
    ));
    let table = harmonic_spec("bond", "tabled").params(vec![
        ParamSpec::new("k", "E/L^2".parse().unwrap()),
        ParamSpec::new("r0", Dim::LENGTH),
        ParamSpec::new("table", Dim::ENERGY).kind(ParamKind::Array { rank: 1 }),
    ]);
    r.register_style(table, scalar(RIGHT)).unwrap();
}
