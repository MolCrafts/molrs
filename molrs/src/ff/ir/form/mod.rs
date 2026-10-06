//! Forms: converting a force field between styles of one physics
//! (`ff-ir-01` P3, on the protocol of `ff-ir-02` §9).
//!
//! A **form family** is a set of styles whose energies are one function
//! space parametrised differently — the torsions (Fourier series in φ), the
//! harmonic bonds and angles, the Lennard-Jones pairs. Each member registers a
//! [`FormCodec`] beside its kernel ([`Registry::register_form`]): its family,
//! and two exact maps between its own parameters and the family's
//! **canonical** style's:
//!
//! * `embed` — this style → canonical (exact: the same energy, constant
//!   included), refusing a row the canonical style cannot hold;
//! * `project` — canonical → this style, exact on its image, refusing with a
//!   [`Refusal`] that names the condition otherwise (a Fourier series with
//!   `bₙ ≠ 0` has no RB form);
//! * optionally `seed` — canonical → the nearest member of this style's
//!   image by a rule of thumb, where [`fit`](Registry::fit_form) starts.
//!
//! Three operations on a [`ForceField`] use them, one job each:
//!
//! * [`ForceField::canonical`] maps every style of a family **in the
//!   canonical style's category** onto the canonical style (styles of the
//!   family in another category — impropers in the torsion family — stay);
//! * [`ForceField::to_form`] converts one category's styles of a family to
//!   one style of it, through the canonical parameters, exactly or not at all
//!   ([`IrError::OutOfImage`] names the type and the condition);
//! * [`ForceField::fit_form`] is the projection that always answers: least
//!   squares under a declared [`Metric`], using nothing but each style's
//!   energy `E(q)` from the registry's kernels — so it works for an
//!   expression or a Python style as for a native one — and returns the
//!   [`Residual`] beside the parameters.
//!
//! The built-in families ([`builtin_forms`]):
//!
//! | family | canonical | members |
//! |---|---|---|
//! | `torsion` | `dihedral periodic` | `dihedral` `charmm` (w = 0), `opls`, `multi/harmonic`, `nharmonic`, `harmonic`, `class2`, `rb`; `improper` `cvff`, `periodic` |
//! | `bond` | `bond harmonic` | `bond class2` (k3 = k4 = 0) |
//! | `angle` | `angle harmonic` | `angle class2` (k3 = k4 = 0), `angle charmm` (k_ub = 0) |
//! | `lj` | `pair lj/cut` | `pair lj/class2` (= lj/cut n = 9, m = 6, σ' = (2/3)^⅓ σ) |
//!
//! No codec, by design: `improper harmonic` and `bond`/`pair` `morse` are no
//! member of any family's function space (fit them with `fit_form`);
//! `pair lj/charmm` always switches between `inner < cutoff` (its kernel
//! refuses `inner = cutoff`), so it is never exactly `lj/cut`.

mod builtin;
mod fit;
#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

pub use builtin::builtin_forms;
pub use fit::{Metric, Residual, TypeResidual};

use crate::ff::forcefield::{DefError, ForceField, Params, Style};
use crate::ff::ir::{IrError, Registry, StyleSpec, with_global};

/// One type's full parameter set: its style's parameters and its own row.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TypeParams {
    /// The style parameters (`cutoff`, `mixing`, …).
    pub style: Params,
    /// The type's row.
    pub row: Params,
}

impl TypeParams {
    pub fn new(style: Params, row: Params) -> Self {
        Self { style, row }
    }

    /// A row of a style without style parameters.
    pub fn row(row: Params) -> Self {
        Self {
            style: Params::new(),
            row,
        }
    }
}

/// Why a form map refused a row: the condition, in words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub reason: String,
}

impl Refusal {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for Refusal {}

/// A map between one style's parameters and its family's canonical ones.
pub type FormFn = Arc<dyn Fn(&TypeParams) -> Result<TypeParams, Refusal> + Send + Sync>;

/// A style's membership of a form family (`ff-ir-02-protocol` §9).
#[derive(Clone)]
pub struct FormCodec {
    /// The family, e.g. `torsion`.
    pub family: Cow<'static, str>,
    /// Whether this is the family's canonical style (exactly one is).
    pub canonical: bool,
    /// This style → the canonical style's parameters, exactly (the same
    /// energy, constant included), or a refusal naming the condition. On the
    /// canonical style itself: its canonical spelling.
    pub embed: FormFn,
    /// Canonical parameters → this style's, exact on its image, else a
    /// refusal naming the condition.
    pub project: FormFn,
    /// Canonical parameters → the nearest member of this style's image by a
    /// rule of thumb (the dominant order of a one-term torsion): the start of
    /// [`Registry::fit_form`]. `None`: the fit starts from the source row's
    /// same-named values.
    pub seed: Option<FormFn>,
}

impl FormCodec {
    /// A member of `family` (not its canonical style) with exact maps
    /// `embed` and `project`.
    pub fn new(
        family: impl Into<Cow<'static, str>>,
        embed: impl Fn(&TypeParams) -> Result<TypeParams, Refusal> + Send + Sync + 'static,
        project: impl Fn(&TypeParams) -> Result<TypeParams, Refusal> + Send + Sync + 'static,
    ) -> Self {
        Self {
            family: family.into(),
            canonical: false,
            embed: Arc::new(embed),
            project: Arc::new(project),
            seed: None,
        }
    }

    /// Mark this as its family's canonical style.
    pub fn as_canonical(mut self) -> Self {
        self.canonical = true;
        self
    }

    /// Give the fit a starting point.
    pub fn seed(
        mut self,
        seed: impl Fn(&TypeParams) -> Result<TypeParams, Refusal> + Send + Sync + 'static,
    ) -> Self {
        self.seed = Some(Arc::new(seed));
        self
    }

    /// Whether `self` and `other` are the same registration: the same family
    /// and role, the same map objects.
    pub(crate) fn same(&self, other: &FormCodec) -> bool {
        fn thin(f: &FormFn) -> *const () {
            Arc::as_ptr(f) as *const ()
        }
        self.family == other.family
            && self.canonical == other.canonical
            && thin(&self.embed) == thin(&other.embed)
            && thin(&self.project) == thin(&other.project)
            && match (&self.seed, &other.seed) {
                (None, None) => true,
                (Some(a), Some(b)) => thin(a) == thin(b),
                _ => false,
            }
    }
}

impl fmt::Debug for FormCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FormCodec")
            .field("family", &self.family)
            .field("canonical", &self.canonical)
            .field("seed", &self.seed.is_some())
            .finish()
    }
}

// ── the operations ──────────────────────────────────────────────────────────

/// `"<category> <style>"`, as messages name a style.
fn named(category: &str, style: &str) -> String {
    format!("{category} {style}")
}

/// Whether `key` is one of `spec`'s per-type parameters: declared, a member
/// `<name><m>` of a declared indexed family, or a `<name>` of an indexed
/// family the style also spells unindexed.
pub(crate) fn declared(spec: &StyleSpec, key: &str) -> bool {
    if spec.param(key).is_some() {
        return true;
    }
    let stem = key.trim_end_matches(|c: char| c.is_ascii_digit());
    stem.len() < key.len() && spec.param(stem).is_some_and(|p| p.indexed)
}

/// The output row of a conversion: `converted`, plus every key of the source
/// row that is no parameter of the source style (`id`, `desc`, `smarts`, …),
/// carried as it is.
fn carry_annotations(source: &Params, spec: Option<&StyleSpec>, mut converted: Params) -> Params {
    let keep = |key: &str| spec.is_some_and(|s| !declared(s, key));
    for (key, value) in source.iter() {
        if keep(key) && converted.get(key).is_none() {
            converted.set(key, value);
        }
    }
    for (key, value) in source.iter_strings() {
        if keep(key) && converted.get_str(key).is_none() {
            converted.set_str(key, value);
        }
    }
    for (key, value) in source.iter_arrays() {
        if keep(key) && converted.get_array(key).is_none() {
            converted.set_array(key, value.clone());
        }
    }
    converted
}

/// One converted row: its name, endpoints and parameters in the target style.
pub(crate) struct Converted {
    pub name: String,
    pub endpoints: Vec<String>,
    pub params: TypeParams,
}

/// Every row of `style` converted by `convert`, annotations carried.
pub(crate) fn convert_rows(
    registry: &Registry,
    style: &Style,
    mut convert: impl FnMut(&str, TypeParams) -> Result<TypeParams, IrError>,
) -> Result<Vec<Converted>, IrError> {
    let spec = registry
        .style(style.category(), style.name())
        .map(|(spec, _)| spec);
    style
        .type_rows()
        .into_iter()
        .map(|(name, endpoints, row)| {
            let mut out = convert(name, TypeParams::new(style.params().clone(), row.clone()))?;
            out.row = carry_annotations(row, spec, out.row);
            Ok(Converted {
                name: name.to_owned(),
                endpoints: endpoints.into_iter().map(str::to_owned).collect(),
                params: out,
            })
        })
        .collect()
}

/// `ff` with the styles `sources` of `category` replaced by `target`, holding
/// `rows` (the sources' rows converted) after the target's own rows when it
/// is not among the sources. The target takes the place of the first of
/// them; every other style keeps its place and its rows.
pub(crate) fn rewrite(
    ff: &ForceField,
    family: &str,
    category: &str,
    target: &str,
    sources: &[String],
    rows: Vec<Converted>,
) -> Result<ForceField, IrError> {
    let conflict = |reason: String| IrError::FormConflict {
        family: family.to_owned(),
        reason,
    };
    // One set of style parameters for the target.
    let existing = ff
        .get_style(category, target)
        .filter(|_| !sources.iter().any(|s| s == target));
    let mut style_params: Option<(String, Params)> =
        existing.map(|s| (named(category, target), s.params().clone()));
    for row in &rows {
        match &style_params {
            Some((from, p)) if *p != row.params.style => {
                return Err(conflict(format!(
                    "`{}` needs style parameters {:?} for the row '{}', and {from} {:?}",
                    named(category, target),
                    row.params.style,
                    row.name,
                    p
                )));
            }
            Some(_) => {}
            None => style_params = Some((format!("row '{}'", row.name), row.params.style.clone())),
        }
    }
    let style_params = style_params.map(|(_, p)| p).unwrap_or_default();

    let def = |e: DefError| match e {
        DefError::TypeConflict { name, .. } => conflict(format!(
            "two rows named '{name}' convert to different `{}` rows",
            named(category, target)
        )),
        DefError::PairConflict {
            itom, jtom, name, ..
        } => conflict(format!(
            "the pair {itom}-{jtom} of '{name}' converts to a `{}` row another type states \
             differently",
            named(category, target)
        )),
        other => conflict(other.to_string()),
    };
    let is_replaced = |s: &Style| {
        s.category() == category && (s.name() == target || sources.iter().any(|n| n == s.name()))
    };
    let mut out = ff.empty_like();
    let mut rows = Some(rows);
    for style in ff.styles() {
        if !is_replaced(style) {
            let copy = out
                .def_style(style.category(), style.name(), style.params().clone())
                .map_err(def)?;
            for (name, endpoints, params) in style.type_rows() {
                copy.def_type(name, &endpoints, params.clone())
                    .map_err(def)?;
            }
            continue;
        }
        let Some(rows) = rows.take() else {
            continue; // the target is already in place
        };
        let t = out
            .def_style(category, target, style_params.clone())
            .map_err(def)?;
        if let Some(own) = existing {
            for (name, endpoints, params) in own.type_rows() {
                t.def_type(name, &endpoints, params.clone()).map_err(def)?;
            }
        }
        for row in rows {
            let endpoints: Vec<&str> = row.endpoints.iter().map(String::as_str).collect();
            t.def_type(&row.name, &endpoints, row.params.row)
                .map_err(def)?;
        }
    }
    Ok(out)
}

/// The styles of `category` in `ff` that hold rows and register a codec of
/// `family`, by name, in their order.
fn members(registry: &Registry, ff: &ForceField, category: &str, family: &str) -> Vec<String> {
    ff.get_styles(category)
        .into_iter()
        .filter(|s| !s.type_rows().is_empty())
        .filter(|s| {
            registry
                .form(category, s.name())
                .is_some_and(|c| c.family == family)
        })
        .map(|s| s.name().to_owned())
        .collect()
}

impl Registry {
    /// [`ForceField::canonical`] against this registry.
    pub fn canonical(&self, ff: &ForceField) -> Result<ForceField, IrError> {
        // A family a style of `ff` belongs to must have its canonical style.
        for style in ff.styles() {
            if let Some(codec) = self.form(style.category(), style.name()) {
                self.canonical_form(&codec.family)?;
            }
        }
        let families: Vec<(String, String, String)> = self
            .forms()
            .filter(|(_, _, f)| f.canonical)
            .map(|(c, s, f)| (f.family.to_string(), c.to_owned(), s.to_owned()))
            .collect();
        let mut out = ff.clone();
        for (family, category, target) in families {
            let sources = members(self, &out, &category, &family);
            if sources.is_empty() {
                continue;
            }
            let mut rows = Vec::new();
            for source in &sources {
                let style = out.get_style(&category, source).expect("a member");
                let codec = self.form(&category, source).expect("a member");
                rows.extend(convert_rows(self, style, |name, tp| {
                    (codec.embed)(&tp).map_err(|e| IrError::OutOfImage {
                        from: named(&category, source),
                        to: named(&category, &target),
                        type_: name.to_owned(),
                        reason: e.reason,
                    })
                })?);
            }
            out = rewrite(&out, &family, &category, &target, &sources, rows)?;
        }
        Ok(out)
    }

    /// [`ForceField::to_form`] against this registry.
    pub fn to_form(
        &self,
        ff: &ForceField,
        category: &str,
        style: &str,
    ) -> Result<ForceField, IrError> {
        let codec = self.form(category, style).ok_or_else(|| IrError::NoForm {
            category: category.to_owned(),
            style: style.to_owned(),
        })?;
        let family = codec.family.to_string();
        let (cc, cs) = self.canonical_form(&family)?;
        let canonical = named(cc, cs);
        let sources: Vec<String> = members(self, ff, category, &family)
            .into_iter()
            .filter(|s| s != style)
            .collect();
        let mut rows = Vec::new();
        for source in &sources {
            let from = self.form(category, source).expect("a member");
            let style_of = ff.get_style(category, source).expect("a member");
            rows.extend(convert_rows(self, style_of, |name, tp| {
                let refused = |to: &str, e: Refusal| IrError::OutOfImage {
                    from: named(category, source),
                    to: to.to_owned(),
                    type_: name.to_owned(),
                    reason: e.reason,
                };
                let c = (from.embed)(&tp).map_err(|e| refused(&canonical, e))?;
                (codec.project)(&c).map_err(|e| refused(&named(category, style), e))
            })?);
        }
        if rows.is_empty() {
            return Ok(ff.clone());
        }
        rewrite(ff, &family, category, style, &sources, rows)
    }
}

impl ForceField {
    /// The force field in canonical form: every style of a form family in the
    /// family's canonical style's category mapped onto that style, exactly
    /// (the same energy, constant included), its rows in their canonical
    /// spelling — `dihedral opls`, `charmm`, `rb`, … → `dihedral periodic`,
    /// `bond class2` → `bond harmonic`, `pair lj/class2` → `pair lj/cut`.
    /// Styles of no family, and of a family whose canonical style is in
    /// another category (impropers), stay as they are. Idempotent.
    ///
    /// # Errors
    ///
    /// [`IrError::OutOfImage`] for a row the canonical style cannot hold
    /// (a charmm `w ≠ 0`, a class2 bond with `k3 ≠ 0`), naming the type and
    /// the condition; [`IrError::FormConflict`] for two rows of one name
    /// that become different rows, two styles that need different style
    /// parameters, or a family without a canonical style.
    pub fn canonical(&self) -> Result<ForceField, IrError> {
        with_global(Registry::clone).canonical(self)
    }

    /// The force field with every style of `category` in the form family of
    /// `style` converted to `style`, exactly, through the family's canonical
    /// parameters; the other styles stay.
    ///
    /// # Errors
    ///
    /// [`IrError::NoForm`] when `style` registers no form codec;
    /// [`IrError::OutOfImage`] naming the first row outside `style`'s image
    /// and the condition (`sin(2φ) coefficient … ≠ 0`, `the constant term …`);
    /// [`IrError::FormConflict`] as for [`canonical`](Self::canonical).
    pub fn to_form(&self, category: &str, style: &str) -> Result<ForceField, IrError> {
        with_global(Registry::clone).to_form(self, category, style)
    }

    /// The force field with every other style of `category` that holds rows
    /// fitted to `style` by least squares under `metric`, and the
    /// [`Residual`] of each fitted row. Generic: it evaluates every style
    /// through the registry's kernels alone. See [`Registry::fit_form`].
    pub fn fit_form(
        &self,
        category: &str,
        style: &str,
        metric: &Metric,
    ) -> Result<(ForceField, Residual), IrError> {
        with_global(Registry::clone).fit_form(self, category, style, metric)
    }
}
