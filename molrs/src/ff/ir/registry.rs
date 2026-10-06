//! The registry of the force-field IR: categories, styles and the kernels
//! that price them — one table, open to any caller that conforms.
//!
//! [`Registry::builtin`] holds molrs's own categories and styles, registered
//! through the same spec types a third party uses and then **sealed**: a
//! built-in cannot be overridden or removed ([`IrError::Sealed`]).
//! Re-registering anything identically is a no-op; registering something
//! else under a taken name is [`IrError::Conflict`], mirroring the force
//! field's own conflict rule.
//!
//! [`PotentialCompiler`](crate::ff::potential::PotentialCompiler) reads the
//! process-wide registry ([`register_style`], [`with_global`], …) unless it
//! is handed one ([`PotentialCompiler::with_registry`]), which is how a test
//! extends the IR without touching anything another test sees.
//!
//! [`PotentialCompiler::with_registry`]: crate::ff::potential::PotentialCompiler::with_registry

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, OnceLock, RwLock};

use crate::ff::forcefield::Params;
use crate::ff::ir::conformance::{self, PROBE_TERMS, Probe, form_id};
use crate::ff::ir::engine::{Engine, LammpsCodec, LammpsForm};
use crate::ff::ir::form::FormCodec;
use crate::ff::ir::{CategorySpec, IrError, StyleSpec, builtin_categories, builtin_styles};
use crate::ff::potential::KernelRegistry;
use crate::ff::potential::Member;
use crate::ff::potential::generic::{
    CompoundForm, CompoundTerms, ScalarBonded, ScalarForm, ScalarPair,
};
use molrs::store::frame::Frame;
use molrs::types::F;

// ---------------------------------------------------------------------------
// Registration vocabulary
// ---------------------------------------------------------------------------

/// Builds a molecule-bound [`Member`] from a style's params, its per-type
/// params (`(type_label, params)`), and a typed [`Frame`]. Every kernel
/// constructor in the crate matches this signature.
pub type KernelConstructor =
    fn(&Params, &[(&str, &Params)], &Frame) -> Result<Member, crate::ff::potential::CompileError>;

/// Which `Frame` block decides whether a style has any rows to act on.
///
/// `PotentialCompiler::compile` skips a style whose topology is absent — a bond style
/// with no bonds contributes nothing, and letting the kernel fault on the
/// missing block instead would be a worse way to say so. Which block that is,
/// is a property of the **kernel**, not of its category.
///
/// PME is the case that proves it: registered under `pair` because that is
/// where an electrostatic style belongs, it reads per-atom charges and
/// `exclusions` and never looks at `pairs`. Gated on `pairs`, it was skipped
/// outright for any system whose caller had not built a pair list — deleting
/// the entire long-range electrostatics, silently, to exactly zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RowSource {
    /// The category's own topology block: `bonds`, `angles`, `dihedrals`,
    /// `impropers` or `pairs`. Absent or empty means the style contributes
    /// nothing.
    #[default]
    CategoryBlock,
    /// The atoms, or rows the kernel finds for itself. Nothing gates it.
    Atoms,
}

/// Where a kernel's parameters come from — the question the empty-type-params
/// guard must ask before it rejects a style with no type rows.
///
/// A kernel constructor that binds its type-params as `_tp` (i.e. resolves
/// nothing from them) **is not a table-driven style**, and must say so by being
/// registered [`PerInstance`](ParamSource::PerInstance).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamSource {
    /// Parameters come from the style's type-definition rows (the `tp` slice).
    /// A style with no rows resolves nothing, and is an error.
    TypeRows,
    /// Parameters are resolved per interaction by the typifier and baked into
    /// [`Frame`] columns; `tp` is ignored and may legitimately be empty.
    PerInstance,
}

/// Which of a force field's special-bonds weight sets scales a pair style.
///
/// A force field may scale close van-der-Waals and electrostatic neighbours
/// differently — Amber uses `1/2` and `1/1.2` — and in molrs those are
/// separate kernels, so each has to say which set is its own. Declared at
/// registration rather than guessed from the style's name: a name is a label,
/// and this is a fact about the physics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialClass {
    /// Scaled by the force field's van-der-Waals weights.
    Vdw,
    /// Scaled by its electrostatic weights.
    Coulomb,
}

/// A style's energy written as an expression, compiled.
///
/// The seam between the registry and the expression engine
/// (`ff::ir::expr`, whose `Compiled` implements it): the registry never
/// parses an expression, it asks this trait. The registry checks it like
/// any other kernel — its variables against the style's declarations
/// ([`variables`](Self::variables)), its derivative against a central
/// difference ([`form`](Self::form)) — and builds it into the same generic
/// kernels a native form gets.
pub trait ExpressionKernel: Send + Sync + 'static {
    /// The expression exactly as declared (molrec: kept byte for byte).
    fn source(&self) -> &str;

    /// Every free name the expression reads that its own `;` definitions do
    /// not bind and that is not a point (`p1` … in `distance(p1, p2)`).
    /// Registration refuses ([`IrError::UnboundVariable`]) one that is not
    /// a variable of the category ([`Coordinate::variables`]), a declared
    /// numeric per-type or style parameter (an indexed family by any member
    /// `k1`, `k2`, …), or on a pair `q1`, `q2`, `<param>1`, `<param>2`.
    ///
    /// [`Coordinate::variables`]: crate::ff::ir::Coordinate::variables
    fn variables(&self) -> Vec<String>;

    /// The kernel the expression evaluates as: a [`ScalarForm`] of the
    /// category's coordinate, or a [`CompoundForm`] of the points (a
    /// `Compound` category, or any use of a point function). A pair form
    /// reads `x1`/`x2` (self rows) from columns of those names, which the
    /// pair kernel does not supply to Tier 2: an expression engine that
    /// binds them resolves them itself.
    fn form(&self) -> ExpressionForm;
}

/// What an [`ExpressionKernel`] evaluates as.
#[derive(Clone)]
pub enum ExpressionForm {
    Scalar(Arc<dyn ScalarForm>),
    Compound(Arc<dyn CompoundForm>),
}

/// Compiles a style's [`expression`](StyleSpec::expression) for its
/// category. The expression engine installs one
/// ([`Registry::set_expression_compiler`]); without one, an expression-only
/// style is priced by nothing ([`IrError::NoKernel`]) and an expression
/// beside a native kernel is not checked.
pub type ExpressionCompiler =
    fn(&CategorySpec, &StyleSpec) -> Result<Arc<dyn ExpressionKernel>, IrError>;

/// What prices a style: the three kernel tiers.
#[derive(Clone)]
pub enum Kernel {
    /// Tier 1: the energy as an expression, compiled.
    Expression(Arc<dyn ExpressionKernel>),
    /// Tier 2: a batch function of the category's coordinate, built into
    /// [`ScalarBonded`] (a bonded category) or [`ScalarPair`] (a pair
    /// category).
    Scalar(Arc<dyn ScalarForm>),
    /// Tier 2: a batch N-body function of positions, built into
    /// [`CompoundTerms`].
    Compound(Arc<dyn CompoundForm>),
    /// Tier 3: a constructor that builds the whole [`Member`] itself — every
    /// built-in kernel.
    Ctor {
        compiled: KernelConstructor,
        /// The neighbour-driven form of a pair style, and the special-bonds
        /// weights that scale it.
        typed: Option<(KernelConstructor, SpecialClass)>,
        /// Which block's emptiness means the style contributes nothing.
        rows: RowSource,
    },
}

impl Kernel {
    /// A Tier-3 constructor with no neighbour-driven form, gated on its
    /// category's block.
    pub fn ctor(compiled: KernelConstructor) -> Self {
        Kernel::Ctor {
            compiled,
            typed: None,
            rows: RowSource::CategoryBlock,
        }
    }

    /// The tier, for messages.
    pub(crate) fn tier(&self) -> &'static str {
        match self {
            Kernel::Expression(_) => "expression",
            Kernel::Scalar(_) => "scalar form",
            Kernel::Compound(_) => "compound form",
            Kernel::Ctor { .. } => "constructor",
        }
    }

    /// Whether `self` and `other` are the same kernel: the same object, or
    /// the same constructors with the same declarations.
    fn same(&self, other: &Kernel) -> bool {
        fn thin<T: ?Sized>(p: &Arc<T>) -> *const () {
            Arc::as_ptr(p) as *const ()
        }
        match (self, other) {
            (Kernel::Expression(a), Kernel::Expression(b)) => thin(a) == thin(b),
            (Kernel::Scalar(a), Kernel::Scalar(b)) => thin(a) == thin(b),
            (Kernel::Compound(a), Kernel::Compound(b)) => thin(a) == thin(b),
            (
                Kernel::Ctor {
                    compiled: a,
                    typed: ta,
                    rows: ra,
                },
                Kernel::Ctor {
                    compiled: b,
                    typed: tb,
                    rows: rb,
                },
            ) => {
                std::ptr::fn_addr_eq(*a, *b)
                    && ra == rb
                    && match (ta, tb) {
                        (None, None) => true,
                        (Some((a, ca)), Some((b, cb))) => std::ptr::fn_addr_eq(*a, *b) && ca == cb,
                        _ => false,
                    }
            }
            _ => false,
        }
    }
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Kernel::{}", self.tier())
    }
}

/// One registered style: its spec and its kernel (none for a style priced
/// by its expression alone, or in a category that prices nothing).
#[derive(Clone, Debug)]
pub(crate) struct StyleEntry {
    spec: StyleSpec,
    kernel: Option<Kernel>,
    sealed: bool,
    /// What the first-compile verdict is keyed on instead of the form's
    /// address: the registry expression an instance expression is checked
    /// against, whose compiled form is new at every compile.
    probe_id: Option<usize>,
}

/// What a style's form kernel is, and where it came from.
struct Form {
    form: ExpressionForm,
    /// A native form (Tier 2), which an expression beside it must agree
    /// with.
    native: bool,
}

impl StyleEntry {
    pub fn spec(&self) -> &StyleSpec {
        &self.spec
    }

    /// The entry an **unregistered** style that carries an `expression` is
    /// priced under: the compile fallback (protocol §4), through the
    /// installed expression engine.
    pub(crate) fn fallback(
        category: &CategorySpec,
        name: &str,
        style: &Params,
        tp: &[(&str, &Params)],
        expressions: Option<ExpressionCompiler>,
    ) -> Option<Result<StyleEntry, IrError>> {
        let expression = style.get_str("expression")?;
        let spec = crate::ff::ir::expression::fallback_spec(category, name, style, tp, expression);
        let Some(compile) = expressions else {
            return Some(Err(IrError::NoKernel {
                category: category.name.to_string(),
                style: name.to_owned(),
            }));
        };
        Some(compile(category, &spec).map(|x| StyleEntry {
            spec,
            kernel: Some(Kernel::Expression(x)),
            sealed: false,
            probe_id: None,
        }))
    }

    /// The entry a style instance carrying its own `expression` is priced
    /// under (protocol §4, D16).
    ///
    /// The registered kernel prices it, a registered kernel taking priority
    /// over an expression; an instance expression that differs from the
    /// registry's is checked for agreement with that kernel at first compile
    /// ([`IrError::Disagree`]), like an expression registered beside a native
    /// kernel. A registered style with neither kernel nor expression is
    /// priced by the instance's. A sealed built-in, and a Tier-3
    /// constructor (never sampled), is priced as registered, unchecked.
    pub(crate) fn with_instance_expression(
        &self,
        category: &CategorySpec,
        instance: Option<&str>,
        expressions: Option<ExpressionCompiler>,
    ) -> Result<std::borrow::Cow<'_, StyleEntry>, IrError> {
        use std::borrow::Cow;
        let Some(instance) = instance else {
            return Ok(Cow::Borrowed(self));
        };
        if self.sealed || self.spec.expression.as_deref() == Some(instance) {
            return Ok(Cow::Borrowed(self));
        }
        let native = |form: ExpressionForm| match form {
            ExpressionForm::Scalar(f) => Kernel::Scalar(f),
            ExpressionForm::Compound(f) => Kernel::Compound(f),
        };
        let keyed = |source: &str| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            source.hash(&mut h);
            Some(h.finish() as usize)
        };
        let (kernel, probe_id) = match (&self.kernel, &self.spec.expression) {
            (Some(Kernel::Ctor { .. }), _) => return Ok(Cow::Borrowed(self)),
            (Some(k @ (Kernel::Scalar(_) | Kernel::Compound(_))), _) => (Some(k.clone()), None),
            (Some(Kernel::Expression(x)), _) => (Some(native(x.form())), keyed(x.source())),
            (None, Some(registered)) => match expressions {
                Some(compile) => (
                    Some(native(compile(category, &self.spec)?.form())),
                    keyed(registered),
                ),
                None => return Ok(Cow::Borrowed(self)),
            },
            (None, None) => (None, None),
        };
        let mut spec = self.spec.clone();
        spec.expression = Some(instance.to_owned());
        spec.samples.clear();
        Ok(Cow::Owned(StyleEntry {
            spec,
            kernel,
            sealed: false,
            probe_id,
        }))
    }

    /// Where the style's rows come from.
    pub fn row_source(&self) -> RowSource {
        match self.kernel {
            Some(Kernel::Ctor { rows, .. }) => rows,
            _ => RowSource::CategoryBlock,
        }
    }

    /// The form a Tier-1 or Tier-2 kernel evaluates as; for an
    /// expression-only style, its expression compiled now.
    fn form(
        &self,
        category: &CategorySpec,
        expressions: Option<ExpressionCompiler>,
    ) -> Result<Form, IrError> {
        match &self.kernel {
            Some(Kernel::Expression(x)) => Ok(Form {
                form: x.form(),
                native: false,
            }),
            Some(Kernel::Scalar(f)) => Ok(Form {
                form: ExpressionForm::Scalar(f.clone()),
                native: true,
            }),
            Some(Kernel::Compound(f)) => Ok(Form {
                form: ExpressionForm::Compound(f.clone()),
                native: true,
            }),
            Some(Kernel::Ctor { .. }) => unreachable!("a constructor is no form"),
            None => match (expressions, &self.spec.expression) {
                (Some(compile), Some(_)) => Ok(Form {
                    form: compile(category, &self.spec)?.form(),
                    native: false,
                }),
                _ => Err(IrError::NoKernel {
                    category: self.spec.category.to_string(),
                    style: self.spec.name.to_string(),
                }),
            },
        }
    }

    /// Run the first-compile conformance check of a form kernel.
    fn first_compile(
        &self,
        category: &CategorySpec,
        form: &Form,
        expressions: Option<ExpressionCompiler>,
        probe: impl FnOnce() -> Probe,
    ) -> Result<(), IrError> {
        conformance::check_at_first_compile(
            category,
            &self.spec,
            &form.form,
            self.probe_id.unwrap_or_else(|| form_id(&form.form)),
            form.native,
            expressions,
            probe,
        )
    }

    /// Build the style's kernel for a **compiled** evaluation: its terms
    /// resolved against `frame` (a pair style against its `pairs` block).
    pub(crate) fn compiled(
        &self,
        category: &CategorySpec,
        params: &Params,
        tp: &[(&str, &Params)],
        frame: &Frame,
        expressions: Option<ExpressionCompiler>,
    ) -> Result<Member, crate::ff::potential::CompileError> {
        let (params, rows) = self.spec.gather(params, tp)?;
        let (params, tp) = (&params, &borrowed(&rows));
        if let Some(Kernel::Ctor { compiled, .. }) = &self.kernel {
            return compiled(params, tp, frame);
        }
        let form = self.form(category, expressions)?;
        let coords = frame_coords(frame);
        let spec = &self.spec;
        Ok(match &form.form {
            ExpressionForm::Scalar(f) if category.is_pair_driven() => {
                let k = ScalarPair::compiled(f.clone(), spec, params, tp, frame)?;
                self.first_compile(category, &form, expressions, || {
                    probe_on(&coords, |c| k.probe(c, PROBE_TERMS))
                })?;
                Member::pair(k)
            }
            ExpressionForm::Scalar(f) => {
                let k = ScalarBonded::build(f.clone(), category, spec, params, tp, frame)?;
                self.first_compile(category, &form, expressions, || {
                    probe_on(&coords, |c| k.probe(c, PROBE_TERMS))
                })?;
                Member::indexed(k)
            }
            ExpressionForm::Compound(f) => {
                let k = CompoundTerms::build(f.clone(), category, spec, params, tp, frame)?;
                self.first_compile(category, &form, expressions, || {
                    probe_on(&coords, |c| k.probe(c, PROBE_TERMS))
                })?;
                Member::indexed(k)
            }
        })
    }

    /// Build a pair style's kernel for a **neighbour-driven** evaluation,
    /// with the special-bonds weights that scale it; `None` when the style
    /// has no such form.
    pub(crate) fn typed(
        &self,
        category: &CategorySpec,
        params: &Params,
        tp: &[(&str, &Params)],
        frame: &Frame,
        expressions: Option<ExpressionCompiler>,
    ) -> Result<Option<(Member, SpecialClass)>, crate::ff::potential::CompileError> {
        let (params, rows) = self.spec.gather(params, tp)?;
        let (params, tp) = (&params, &borrowed(&rows));
        match &self.kernel {
            Some(Kernel::Ctor { typed: None, .. }) => return Ok(None),
            Some(Kernel::Ctor {
                typed: Some((ctor, class)),
                ..
            }) => return Ok(Some((ctor(params, tp, frame)?, *class))),
            _ => {}
        }
        let form = self.form(category, expressions)?;
        let ExpressionForm::Scalar(f) = &form.form else {
            return Ok(None);
        };
        let k = ScalarPair::typed(f.clone(), &self.spec, params, tp, frame)?;
        let coords = frame_coords(frame);
        self.first_compile(category, &form, expressions, || {
            probe_on(&coords, |c| k.probe(c, PROBE_TERMS))
        })?;
        Ok(Some((Member::pair(k), self.spec.special_class())))
    }
}

/// `rows` as the `(label, row)` slice a kernel takes.
fn borrowed(rows: &[(String, Params)]) -> Vec<(&str, &Params)> {
    rows.iter().map(|(l, r)| (l.as_str(), r)).collect()
}

/// The probe of a kernel's real terms at `coords`, or none (the check
/// waits for a compile whose frame has positions) when the frame has none.
fn probe_on(coords: &[F], probe: impl FnOnce(&[F]) -> Probe) -> Probe {
    if coords.is_empty() {
        Probe::default()
    } else {
        probe(coords)
    }
}

/// The atoms' coordinates of `frame`, flat; empty without them.
fn frame_coords(frame: &Frame) -> Vec<F> {
    let Some(atoms) = frame.get("atoms") else {
        return Vec::new();
    };
    let axis = |k: &str| atoms.get(k).and_then(|c| c.as_float());
    let (Some(x), Some(y), Some(z)) = (axis("x"), axis("y"), axis("z")) else {
        return Vec::new();
    };
    x.iter()
        .zip(y.iter())
        .zip(z.iter())
        .flat_map(|((x, y), z)| [*x, *y, *z])
        .collect()
}

#[derive(Clone, Debug)]
struct CategoryEntry {
    spec: CategorySpec,
    sealed: bool,
}

/// One registered form codec.
#[derive(Clone)]
struct FormEntry {
    codec: FormCodec,
    sealed: bool,
}

/// Categories and styles of the force-field IR, the kernels that price the
/// styles, and the form codecs that convert between them
/// ([`crate::ff::ir::form`]).
#[derive(Clone, Default)]
pub struct Registry {
    categories: BTreeMap<String, CategoryEntry>,
    styles: BTreeMap<(String, String), StyleEntry>,
    forms: BTreeMap<(String, String), FormEntry>,
    expressions: Option<ExpressionCompiler>,
}

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Registry")
            .field("categories", &self.categories.len())
            .field("styles", &self.styles.len())
            .field("forms", &self.forms.len())
            .finish()
    }
}

impl Registry {
    /// An empty registry: no categories, no styles.
    pub fn new() -> Self {
        Self::default()
    }

    /// Every built-in category and style, sealed.
    ///
    /// The kernels are [`KernelRegistry::builtin`]'s constructors, each with
    /// its spec from [`builtin_styles`]; a spec without a constructor is
    /// priced by its expression or prices nothing (`dihedral rb`, `drude
    /// harmonic`, `atom full`, …). A test holds the two tables to one set of
    /// names.
    pub fn builtin() -> Self {
        let mut r = Self::new();
        r.set_expression_compiler(Some(crate::ff::ir::expression::compile_expression));
        // The built-in categories are molrec's table, which the custom rules
        // (`register_category`) are not: `atom` and `virtual_site` name no
        // endpoints, a pair's block is its atoms.
        for c in builtin_categories() {
            r.categories.insert(
                c.name.to_string(),
                CategoryEntry {
                    spec: c,
                    sealed: true,
                },
            );
        }
        let ctors = KernelRegistry::builtin();
        for spec in builtin_styles() {
            let kernel = ctors.kernel(&spec.category, &spec.name);
            let (category, name) = (spec.category.clone(), spec.name.clone());
            r.register_style(spec, kernel)
                .unwrap_or_else(|e| panic!("built-in {category} `{name}`: {e}"));
        }
        for (category, style, codec) in crate::ff::ir::form::builtin_forms() {
            r.register_form(category, style, codec)
                .unwrap_or_else(|e| panic!("built-in form of {category} `{style}`: {e}"));
        }
        for s in r.styles.values_mut() {
            s.sealed = true;
        }
        for f in r.forms.values_mut() {
            f.sealed = true;
        }
        r
    }

    /// Register a category added at run time ([`CategorySpec::custom`]).
    ///
    /// Refuses a malformed name, an arity outside 2..=5, a block other than
    /// `<name>s`, a coordinate its arity cannot carry, and a name a
    /// different category holds (a built-in with [`IrError::Sealed`]); an
    /// identical re-registration is a no-op.
    pub fn register_category(&mut self, c: CategorySpec) -> Result<(), IrError> {
        if let Some(existing) = self.categories.get(c.name.as_ref()) {
            if existing.spec == c {
                return Ok(());
            }
            let category = c.name.into_owned();
            let style = String::new();
            return Err(if existing.sealed {
                IrError::Sealed { category, style }
            } else {
                IrError::Conflict { category, style }
            });
        }
        conformance::check_category(&c)?;
        self.categories.insert(
            c.name.to_string(),
            CategoryEntry {
                spec: c,
                sealed: false,
            },
        );
        Ok(())
    }

    /// Register a style and its kernel, after checking that both conform
    /// ([`conformance`]). `kernel` `None`
    /// registers a style priced by its `expression` alone (Tier 1 through
    /// the installed expression engine), or a style of a category that
    /// prices nothing.
    ///
    /// An identical re-registration (equal spec, the same kernel) is a
    /// no-op; anything else under a taken name is [`IrError::Sealed`] for a
    /// built-in and [`IrError::Conflict`] otherwise.
    pub fn register_style(
        &mut self,
        spec: StyleSpec,
        kernel: Option<Kernel>,
    ) -> Result<(), IrError> {
        let key = (spec.category.to_string(), spec.name.to_string());
        if let Some(existing) = self.styles.get(&key) {
            let same_kernel = match (&existing.kernel, &kernel) {
                (None, None) => true,
                (Some(a), Some(b)) => a.same(b),
                _ => false,
            };
            if existing.spec == spec && same_kernel {
                return Ok(());
            }
            let (category, style) = key;
            return Err(if existing.sealed {
                IrError::Sealed { category, style }
            } else {
                IrError::Conflict { category, style }
            });
        }
        let category = self
            .category(&spec.category)
            .ok_or_else(|| IrError::UnknownCategory {
                category: spec.category.to_string(),
            })?;
        conformance::check_style(category, &spec, kernel.as_ref(), self.expressions)?;
        spec.lammps.check_spec(&spec)?;
        self.styles.insert(
            key,
            StyleEntry {
                spec,
                kernel,
                sealed: false,
                probe_id: None,
            },
        );
        Ok(())
    }

    /// Remove a style registered at run time. A built-in is
    /// [`IrError::Sealed`]; a style that is not registered,
    /// [`IrError::NoKernel`].
    pub fn unregister_style(&mut self, category: &str, name: &str) -> Result<(), IrError> {
        let key = (category.to_owned(), name.to_owned());
        match self.styles.get(&key) {
            None => Err(IrError::NoKernel {
                category: key.0,
                style: key.1,
            }),
            Some(e) if e.sealed => Err(IrError::Sealed {
                category: key.0,
                style: key.1,
            }),
            Some(_) => {
                self.forms.remove(&key);
                self.styles.remove(&key);
                Ok(())
            }
        }
    }

    /// Register the form codec of a registered style: its family, and its
    /// exact maps to and from the family's canonical parameters
    /// (`ff-ir-02-protocol` §9).
    ///
    /// Refuses a style that is not registered ([`IrError::NoKernel`]), a
    /// second canonical style of one family ([`IrError::FormConflict`]), and
    /// a different codec under a style that has one ([`IrError::Sealed`]
    /// for a built-in, [`IrError::Conflict`] otherwise); the same codec again
    /// is a no-op.
    pub fn register_form(
        &mut self,
        category: &str,
        style: &str,
        codec: FormCodec,
    ) -> Result<(), IrError> {
        let key = (category.to_owned(), style.to_owned());
        if !self.styles.contains_key(&key) {
            return Err(IrError::NoKernel {
                category: key.0,
                style: key.1,
            });
        }
        if let Some(existing) = self.forms.get(&key) {
            if existing.codec.same(&codec) {
                return Ok(());
            }
            let (category, style) = key;
            return Err(if existing.sealed {
                IrError::Sealed { category, style }
            } else {
                IrError::Conflict { category, style }
            });
        }
        if codec.canonical
            && let Some(((c, s), _)) = self
                .forms
                .iter()
                .find(|(_, f)| f.codec.canonical && f.codec.family == codec.family)
        {
            return Err(IrError::FormConflict {
                family: codec.family.to_string(),
                reason: format!(
                    "{c} `{s}` is its canonical style already; {category} `{style}` cannot be \
                     a second one"
                ),
            });
        }
        self.forms.insert(
            key,
            FormEntry {
                codec,
                sealed: false,
            },
        );
        Ok(())
    }

    /// Give a registered style the engine form `form` of `engine`
    /// (`ff-ir-02-protocol` §8): a LAMMPS form for a style registered
    /// without one.
    ///
    /// Only LAMMPS takes a registered form: the OpenMM XML writer writes an
    /// expression style's `Custom*Force` from its expression, and GROMACS
    /// and AMBER hold the built-in styles only ([`IrError::NoEngineForm`]).
    /// A style that is not registered is [`IrError::NoKernel`]; the same
    /// form again is a no-op; another form under a style that has one is
    /// [`IrError::Sealed`] for a built-in and [`IrError::Conflict`]
    /// otherwise; a positional form the spec cannot have is
    /// [`IrError::NoEngineForm`].
    pub fn register_engine_form(
        &mut self,
        engine: Engine,
        category: &str,
        style: &str,
        form: LammpsForm,
    ) -> Result<(), IrError> {
        let key = (category.to_owned(), style.to_owned());
        let Some(entry) = self.styles.get_mut(&key) else {
            return Err(IrError::NoKernel {
                category: key.0,
                style: key.1,
            });
        };
        if engine != Engine::Lammps {
            return Err(engine.refuse(
                category,
                style,
                "only LAMMPS takes a registered form: the OpenMM XML writer writes an expression \
                 style's Custom*Force from its expression, and GROMACS and AMBER hold the \
                 built-in styles only",
            ));
        }
        if entry.spec.lammps == form {
            return Ok(());
        }
        if !matches!(entry.spec.lammps, LammpsForm::None) || entry.sealed {
            let (category, style) = key;
            return Err(if entry.sealed {
                IrError::Sealed { category, style }
            } else {
                IrError::Conflict { category, style }
            });
        }
        form.check_spec(&entry.spec)?;
        entry.spec.lammps = form;
        Ok(())
    }

    /// The style of `category` a LAMMPS file's style name `lammps` reads
    /// as, with its codec: the style of that name when its form writes the
    /// name, else the one style whose form does (`fourier` is `dihedral
    /// periodic`). `None` when no style, or more than one, claims it.
    pub fn lammps_style(
        &self,
        category: &str,
        lammps: &str,
    ) -> Option<(&StyleSpec, &dyn LammpsCodec)> {
        let writes =
            |e: &&StyleEntry| e.spec.lammps.lammps_name(&e.spec).as_deref() == Some(lammps);
        let found = match self.entry(category, lammps).filter(writes) {
            Some(e) => e,
            None => {
                let mut claims = self
                    .styles
                    .values()
                    .filter(|e| e.spec.category == category)
                    .filter(writes);
                let first = claims.next()?;
                if claims.next().is_some() {
                    return None;
                }
                first
            }
        };
        Some((&found.spec, found.spec.lammps.codec()?))
    }

    /// The form codec of `(category, style)`, if it registered one.
    pub fn form(&self, category: &str, style: &str) -> Option<&FormCodec> {
        self.forms
            .get(&(category.to_owned(), style.to_owned()))
            .map(|f| &f.codec)
    }

    /// Every registered form codec, by `(category, style)`.
    pub fn forms(&self) -> impl Iterator<Item = (&str, &str, &FormCodec)> + '_ {
        self.forms
            .iter()
            .map(|((c, s), f)| (c.as_str(), s.as_str(), &f.codec))
    }

    /// The canonical style `(category, style)` of `family`;
    /// [`IrError::FormConflict`] when it has none.
    pub fn canonical_form(&self, family: &str) -> Result<(&str, &str), IrError> {
        self.forms()
            .find(|(_, _, f)| f.canonical && f.family == family)
            .map(|(c, s, _)| (c, s))
            .ok_or_else(|| IrError::FormConflict {
                family: family.to_owned(),
                reason: "no style is its canonical one".into(),
            })
    }

    /// Install the expression engine: it prices expression-only styles and
    /// checks an expression declared beside a native kernel.
    pub fn set_expression_compiler(&mut self, compiler: Option<ExpressionCompiler>) {
        self.expressions = compiler;
    }

    /// The installed expression engine, if any.
    pub fn expression_compiler(&self) -> Option<ExpressionCompiler> {
        self.expressions
    }

    pub fn category(&self, name: &str) -> Option<&CategorySpec> {
        self.categories.get(name).map(|c| &c.spec)
    }

    /// Every category, by name.
    pub fn categories(&self) -> impl Iterator<Item = &CategorySpec> + '_ {
        self.categories.values().map(|c| &c.spec)
    }

    /// A registered style's spec and kernel.
    pub fn style(&self, category: &str, name: &str) -> Option<(&StyleSpec, Option<&Kernel>)> {
        self.entry(category, name)
            .map(|e| (&e.spec, e.kernel.as_ref()))
    }

    /// Every style, or those of `category`, by `(category, name)`.
    pub fn styles<'a>(
        &'a self,
        category: Option<&'a str>,
    ) -> impl Iterator<Item = (&'a StyleSpec, Option<&'a Kernel>)> + 'a {
        self.styles
            .values()
            .filter(move |e| category.is_none_or(|c| e.spec.category == c))
            .map(|e| (&e.spec, e.kernel.as_ref()))
    }

    /// Whether `(category, name)` is a sealed built-in.
    pub fn is_sealed(&self, category: &str, name: &str) -> bool {
        self.entry(category, name).is_some_and(|e| e.sealed)
    }

    pub(crate) fn entry(&self, category: &str, name: &str) -> Option<&StyleEntry> {
        self.styles.get(&(category.to_owned(), name.to_owned()))
    }

    /// The [`ParamSource`] of a registered style.
    pub fn param_source(&self, category: &str, name: &str) -> Option<ParamSource> {
        self.entry(category, name).map(|e| e.spec.source)
    }

    /// Where a registered style's rows come from.
    pub fn row_source(&self, category: &str, name: &str) -> Option<RowSource> {
        self.entry(category, name).map(StyleEntry::row_source)
    }
}

/// The process-wide registry, [`Registry::builtin`] on first use.
fn global() -> &'static RwLock<Registry> {
    static REGISTRY: OnceLock<RwLock<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Registry::builtin()))
}

/// [`Registry::register_category`] on the process-wide registry.
pub fn register_category(c: CategorySpec) -> Result<(), IrError> {
    global().write().unwrap().register_category(c)
}

/// [`Registry::register_style`] on the process-wide registry: the extension
/// point every compile reads, with nothing rebuilt.
pub fn register_style(spec: StyleSpec, kernel: Option<Kernel>) -> Result<(), IrError> {
    global().write().unwrap().register_style(spec, kernel)
}

/// [`Registry::register_form`] on the process-wide registry.
pub fn register_form(category: &str, style: &str, codec: FormCodec) -> Result<(), IrError> {
    global()
        .write()
        .unwrap()
        .register_form(category, style, codec)
}

/// [`Registry::register_engine_form`] on the process-wide registry.
pub fn register_engine_form(
    engine: Engine,
    category: &str,
    style: &str,
    form: LammpsForm,
) -> Result<(), IrError> {
    global()
        .write()
        .unwrap()
        .register_engine_form(engine, category, style, form)
}

/// The registry an engine reader or writer reads: the process-wide one
/// (the default), or a caller's own — a test's styles, seen by nothing
/// else.
#[derive(Clone, Debug, Default)]
pub enum RegistryRef {
    #[default]
    Global,
    Own(Arc<Registry>),
}

impl RegistryRef {
    /// Run `f` on the registry: a snapshot of the process-wide one (no lock
    /// is held while `f` runs, so `f` may read the process-wide registry
    /// itself), or the caller's.
    pub fn with<R>(&self, f: impl FnOnce(&Registry) -> R) -> R {
        match self {
            RegistryRef::Global => f(&with_global(Registry::clone)),
            RegistryRef::Own(r) => f(r),
        }
    }
}

/// [`Registry::unregister_style`] on the process-wide registry.
pub fn unregister_style(category: &str, name: &str) -> Result<(), IrError> {
    global().write().unwrap().unregister_style(category, name)
}

/// [`Registry::set_expression_compiler`] on the process-wide registry.
pub fn set_expression_compiler(compiler: Option<ExpressionCompiler>) {
    global().write().unwrap().set_expression_compiler(compiler);
}

/// Read the process-wide registry. The lock is held for the call, so `f`
/// must not register anything.
pub fn with_global<R>(f: impl FnOnce(&Registry) -> R) -> R {
    f(&global().read().unwrap())
}
