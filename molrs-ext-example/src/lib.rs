//! A third party extends the molrs force-field IR, with nothing in molrs
//! rebuilt.
//!
//! This crate depends on molrs as any user's crate does — a standalone
//! workspace, molrs a path dependency — so it can reach nothing but molrs's
//! `pub` API. Through it, it adds:
//!
//! * [`LjSmoothLinear`]: LAMMPS's `pair_style lj/smooth/linear` (shifted
//!   force), a Tier-2 [`ScalarForm`] of `r`. Its spec declares the
//!   parameters' order and dimensions, so LAMMPS's positional codec writes
//!   it with nothing else written;
//! * a **new category**, `urey_bradley` (3 atoms, Frame block
//!   `urey_bradleys`), priced by a native [`CompoundForm`],
//!   [`UreyBradley`];
//! * `bond fene` (LAMMPS's Kremer–Grest FENE), by its expression alone
//!   ([`FENE`], Tier 1);
//! * a second new category, `bond_angle`: LAMMPS `angle_style class2`'s
//!   bond-angle cross term (`ba`), by an expression and by a native form
//!   ([`BondAngle`]) beside it, which the registry holds to agree.
//!
//! [`register`] registers them all into a [`Registry`]; `tests/proof.rs`
//! prices them against LAMMPS, persists them, and walks every refusal of
//! the protocol.

use std::sync::Arc;

use molrs::ff::ir::{
    CategorySpec, Coordinate, Dim, EndpointOrder, IrError, Kernel, LammpsForm, Mix, ParamSpec,
    Registry, Sample, SpecialClass, StyleSpec, Value,
};
use molrs::ff::potential::form_kernel::{CompoundForm, ParamColumns, ScalarForm};

/// The numeric column `name` of a batch; the registry checked at build
/// that every column a form states in its `inputs` is there.
fn col<'a>(p: &ParamColumns<'a>, name: &str) -> &'a [f64] {
    p.get(name).expect("the kernel supplies every stated input")
}

fn dim(spelling: &str) -> Dim {
    spelling.parse().expect("a dimension spelling")
}

/// A sample point of the registration checks (16 seeded points each):
/// these parameter values, the coordinate (or a compound term's bond
/// lengths) drawn from `q`.
fn sample(params: &[(&'static str, f64)], q: (f64, f64)) -> Sample {
    Sample {
        params: params
            .iter()
            .map(|&(name, v)| (name.into(), Value::Num(v)))
            .collect(),
        q,
    }
}

// ---------------------------------------------------------------------------
// pair lj/smooth/linear: a Tier-2 scalar form
// ---------------------------------------------------------------------------

/// LAMMPS `pair_style lj/smooth/linear`:
/// `E = φ(r) − φ(rc) − (r − rc) φ′(rc)` with `φ` the 12-6 Lennard-Jones,
/// so energy and force both vanish at the cutoff. Parameters arrive as
/// stored: the pair's mixed `epsilon`, `sigma`, and the style's `cutoff`.
pub struct LjSmoothLinear;

impl ScalarForm for LjSmoothLinear {
    fn eval(&self, r: &[f64], p: &ParamColumns<'_>, e: &mut [f64], de_dr: &mut [f64]) {
        let (eps, sigma, rc) = (col(p, "epsilon"), col(p, "sigma"), col(p, "cutoff"));
        for t in 0..r.len() {
            // φ and φ′ of the 12-6 at x.
            let lj = |x: f64| {
                let s6 = (sigma[t] / x).powi(6);
                let phi = 4.0 * eps[t] * (s6 * s6 - s6);
                (phi, 24.0 * eps[t] * (s6 - 2.0 * s6 * s6) / x)
            };
            let ((u, du), (uc, duc)) = (lj(r[t]), lj(rc[t]));
            e[t] = u - uc - (r[t] - rc[t]) * duc;
            de_dr[t] = du - duc;
        }
    }

    fn inputs(&self) -> Vec<String> {
        vec!["epsilon".into(), "sigma".into(), "cutoff".into()]
    }
}

/// The spec of `pair lj/smooth/linear`: LAMMPS's `pair_coeff` order
/// (`epsilon sigma`), Lorentz–Berthelot mixing by the style's `mixing`,
/// the `lj` special-bonds weights, LAMMPS's positional form.
pub fn lj_smooth_linear() -> StyleSpec {
    StyleSpec::new("pair", "lj/smooth/linear")
        .params(vec![
            ParamSpec::new("epsilon", Dim::ENERGY).mix(Mix::LjEpsilon {
                sigma: "sigma".into(),
            }),
            ParamSpec::new("sigma", Dim::LENGTH).mix(Mix::LjSigma {
                epsilon: "epsilon".into(),
            }),
        ])
        .style_params(vec![
            ParamSpec::new("cutoff", Dim::LENGTH),
            ParamSpec::text("mixing", &["arithmetic", "geometric", "sixthpower"])
                .default_value(Value::Text("arithmetic".into())),
        ])
        .special(SpecialClass::Vdw)
        .lammps(LammpsForm::positional())
        .sample(sample(
            &[("epsilon", 0.2), ("sigma", 3.1), ("cutoff", 8.0)],
            (2.9, 7.5),
        ))
}

// ---------------------------------------------------------------------------
// urey_bradley: a new category, a native compound form
// ---------------------------------------------------------------------------

/// The `urey_bradley` category: three atoms (an angle's), its own Frame
/// block `urey_bradleys`, priced as a function of the atoms' positions.
pub fn urey_bradley_category() -> CategorySpec {
    CategorySpec::custom(
        "urey_bradley",
        3,
        Coordinate::Compound,
        EndpointOrder::Reversible,
    )
}

/// The Urey–Bradley 1-3 spring `k_ub (r₁₃ − r_ub)²` — CHARMM's, LAMMPS
/// `angle_style charmm` with `K = 0` — as a function of the term's three
/// positions: `x[t·3 + 0..3]` are term `t`'s atoms in row order, and the
/// form writes `∂E/∂x` (not the force) in the same layout.
pub struct UreyBradley;

impl CompoundForm for UreyBradley {
    fn eval(
        &self,
        x: &[[f64; 3]],
        arity: usize,
        p: &ParamColumns<'_>,
        e: &mut [f64],
        grad: &mut [[f64; 3]],
    ) {
        let (k, r0) = (col(p, "k_ub"), col(p, "r_ub"));
        for t in 0..e.len() {
            let (a, c) = (x[t * arity], x[t * arity + 2]);
            let d = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            e[t] = k[t] * (r - r0[t]).powi(2);
            // dE/dr, along the unit vector from the first atom to the third.
            let g = 2.0 * k[t] * (r - r0[t]) / r;
            grad[t * arity] = [-g * d[0], -g * d[1], -g * d[2]];
            grad[t * arity + 1] = [0.0; 3];
            grad[t * arity + 2] = [g * d[0], g * d[1], g * d[2]];
        }
    }

    fn inputs(&self) -> Vec<String> {
        vec!["k_ub".into(), "r_ub".into()]
    }
}

/// The spec of `urey_bradley harmonic`: CHARMM's `k_ub r_ub`.
pub fn urey_bradley_harmonic() -> StyleSpec {
    StyleSpec::new("urey_bradley", "harmonic")
        .params(vec![
            ParamSpec::new("k_ub", dim("E/L^2")),
            ParamSpec::new("r_ub", Dim::LENGTH),
        ])
        .sample(sample(&[("k_ub", 20.0), ("r_ub", 2.4)], (1.0, 1.6)))
}

// ---------------------------------------------------------------------------
// bond fene: an expression style
// ---------------------------------------------------------------------------

/// LAMMPS `bond_style fene` (Kremer–Grest): the FENE spring and, inside
/// `2^(1/6) σ`, the WCA repulsion — a Lepton expression of `r`.
pub const FENE: &str = "-0.5*k*r0^2*log(1-(r/r0)^2)+step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)";

/// The spec of `bond fene`: LAMMPS's `bond_coeff` order `K R0 epsilon
/// sigma`, priced by [`FENE`] alone, written by LAMMPS's positional codec.
pub fn fene() -> StyleSpec {
    StyleSpec::new("bond", "fene")
        .params(vec![
            ParamSpec::new("k", dim("E/L^2")),
            ParamSpec::new("r0", Dim::LENGTH),
            ParamSpec::new("epsilon", Dim::ENERGY),
            ParamSpec::new("sigma", Dim::LENGTH),
        ])
        .expression(FENE)
        .lammps(LammpsForm::positional())
}

// ---------------------------------------------------------------------------
// bond_angle: class2's bond-angle cross term, by expression and natively
// ---------------------------------------------------------------------------

/// LAMMPS `angle_style class2`'s bond-angle term
/// `E = [n1 (r₁₂ − r1) + n2 (r₂₃ − r2)] (θ − θ0)`, `θ` the angle at the
/// middle atom, `θ0` in degrees.
pub const BOND_ANGLE: &str = "(n1*(distance(p1,p2)-r1)+n2*(distance(p2,p3)-r2))*(angle(p1,p2,p3)-theta0*0.017453292519943295)";

/// The `bond_angle` category: an angle's three atoms, block `bond_angles`.
pub fn bond_angle_category() -> CategorySpec {
    CategorySpec::custom(
        "bond_angle",
        3,
        Coordinate::Compound,
        EndpointOrder::Reversible,
    )
}

/// [`BOND_ANGLE`] written out by hand, with its gradient.
pub struct BondAngle;

impl CompoundForm for BondAngle {
    fn eval(
        &self,
        x: &[[f64; 3]],
        arity: usize,
        p: &ParamColumns<'_>,
        e: &mut [f64],
        grad: &mut [[f64; 3]],
    ) {
        let (n1, n2) = (col(p, "n1"), col(p, "n2"));
        let (r1, r2, theta0) = (col(p, "r1"), col(p, "r2"), col(p, "theta0"));
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        for t in 0..e.len() {
            let (p1, p2, p3) = (x[t * arity], x[t * arity + 1], x[t * arity + 2]);
            let (u, v) = (sub(p1, p2), sub(p3, p2));
            let (a, b) = (dot(u, u).sqrt(), dot(v, v).sqrt());
            let c = (dot(u, v) / (a * b)).clamp(-1.0, 1.0);
            let theta = c.acos();
            let stretch = n1[t] * (a - r1[t]) + n2[t] * (b - r2[t]);
            let bend = theta - theta0[t].to_radians();
            e[t] = stretch * bend;
            // ∂θ/∂x₁ = −(v/(ab) − c u/a²)/sin θ, likewise for x₃.
            let s = (1.0 - c * c).sqrt();
            let mut g1 = [0.0; 3];
            let mut g3 = [0.0; 3];
            for d in 0..3 {
                let dtheta1 = -(v[d] / (a * b) - c * u[d] / (a * a)) / s;
                let dtheta3 = -(u[d] / (a * b) - c * v[d] / (b * b)) / s;
                g1[d] = n1[t] * u[d] / a * bend + stretch * dtheta1;
                g3[d] = n2[t] * v[d] / b * bend + stretch * dtheta3;
            }
            grad[t * arity] = g1;
            grad[t * arity + 1] = [-g1[0] - g3[0], -g1[1] - g3[1], -g1[2] - g3[2]];
            grad[t * arity + 2] = g3;
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["n1", "n2", "r1", "r2", "theta0"]
            .map(String::from)
            .to_vec()
    }
}

/// The spec of `bond_angle class2`: LAMMPS's `angle_coeff … ba N1 N2 r1
/// r2` and the angle's `theta0`; its expression and [`BondAngle`] beside
/// it must agree (checked at registration on the sample).
pub fn bond_angle_class2() -> StyleSpec {
    let per_length_radian = dim("E/L/A");
    StyleSpec::new("bond_angle", "class2")
        .params(vec![
            ParamSpec::new("n1", per_length_radian),
            ParamSpec::new("n2", per_length_radian),
            ParamSpec::new("r1", Dim::LENGTH),
            ParamSpec::new("r2", Dim::LENGTH),
            ParamSpec::new("theta0", Dim::ANGLE),
        ])
        .expression(BOND_ANGLE)
        .sample(sample(
            &[
                ("n1", 10.0),
                ("n2", 8.0),
                ("r1", 1.5),
                ("r2", 1.45),
                ("theta0", 105.0),
            ],
            (1.2, 1.7),
        ))
}

/// Register everything this crate adds into `r` (a [`Registry::builtin`]
/// one): the two categories, the four styles and their kernels. Each
/// registration is checked — names, dimensions, the expression's variables,
/// each native form's derivative against a central difference of its own
/// energy, the bond-angle expression against its native form — and refused
/// with an [`IrError`] naming what does not conform.
pub fn register(r: &mut Registry) -> Result<(), IrError> {
    r.register_style(
        lj_smooth_linear(),
        Some(Kernel::Scalar(Arc::new(LjSmoothLinear))),
    )?;
    r.register_category(urey_bradley_category())?;
    r.register_style(
        urey_bradley_harmonic(),
        Some(Kernel::Compound(Arc::new(UreyBradley))),
    )?;
    r.register_style(fene(), None)?;
    r.register_category(bond_angle_category())?;
    r.register_style(
        bond_angle_class2(),
        Some(Kernel::Compound(Arc::new(BondAngle))),
    )
}

/// [`Registry::builtin`] with this crate's styles.
pub fn registry() -> Registry {
    let mut r = Registry::builtin();
    register(&mut r).expect("the example's styles conform");
    r
}
