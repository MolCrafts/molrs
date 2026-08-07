---
title: Radian default range for AngleDistribution / DihedralDistribution bindings
slug: distribution-angular-default-range
status: approved
created: 2026-08-06
---

# Radian default range for AngleDistribution / DihedralDistribution bindings

## Summary

The PyO3 constructors for `AngleDistribution` and `DihedralDistribution` declare their histogram
bounds in **degrees** (`min=0.0, max=180.0` and `min=-180.0, max=180.0`) while the underlying
observables emit **radians**. Every Python caller who accepts the default therefore bins radian
samples onto a degree axis: an exact 90° angle is recorded at 1.571 instead of 90, and — because
`DistributionResult::finalize` evaluates `sin(bin_center)` on numbers that are degrees, producing
negative weights — the solid-angle-corrected ADF is silently renormalized to **all zeros**. This
spec makes both constructors take optional bounds and, when the caller supplies neither, delegate
to `DistributionFunction::over_natural_range`, which reads the range the observable already owns
(`[0, π]`, `[−π, π]`). Supplying exactly one of `min`/`max` becomes an explicit `ValueError` rather
than a silent half-default. The `.pyi` stub is corrected in the same change, the stub gate is
generalized so declared defaults can never again drift from the runtime, and every new test is
chosen so that restoring the degree default makes it fail.

## Domain basis

**Angle θ (arity 3).** For a triple i–j–k with vertex j, `θ = arccos((r_ij·r_kj)/(|r_ij||r_kj|))`.
Cauchy–Schwarz bounds the argument to `[-1, 1]` and the principal `arccos` maps it onto `[0, π]`;
both endpoints are attained (collinear triples). No convention can exceed π — an unsigned angle
between two vectors has no preferred normal with which to sign it. `[0, π]` is therefore exact and
exhaustive, and is what `AngleObservable::natural_range` returns
(`molrs/src/compute/distribution/angle.rs:31`).

**Dihedral φ (arity 4).** The signed torsion is IUPAC-correct: IUPAC-IUB Commission on Biochemical
Nomenclature (1970) Rule 1.6, <https://iupac.qmul.ac.uk/misc/ppep1.html> — *"Angles are measured in
the range −180 < θ ≤ +180"*, with the eclipsed (synplanar) arrangement at 0; sign convention at
<https://iupac.qmul.ac.uk/stereo/TZ.html> (positive when the proximal bond rotates clockwise viewed
along the central bond). The `atan2` form in `dihedral.rs:86-94` was machine-checked against a
right-handed viewer frame: 0 disagreements over 20 000 random quadruples, max error 1.6e-14 rad.
The default **must stay signed** — folding to |φ| collapses g+ onto g− and destroys
chirality-sensitive conformer populations, and the fold is one-way.
`DihedralObservable::natural_range` returns `(−π, π)` bounds (`dihedral.rs:40`).

**Solid-angle Jacobian — angle only.** `dΩ = sin θ dθ dφ` gives `p(θ) = sin(θ)/2` for the isotropic
null, so the corrected ADF divides the histogram by `sin θ`. VOTCA CSG theory,
<https://www.votca.org/csg/theory.html>, verbatim: `P_r(r) = H_r(r)/(4πr²)`,
`P_θ(θ) = H_θ(θ)/sin θ`, `P_φ(φ) = H_φ(φ)`. For the dihedral the residual freedom at **fixed bond
geometry** is SO(2), whose invariant measure is `dφ` — no Jacobian. (That null is the correct one
here because quadruplets are read from the `dihedrals` topology block, not sampled as four
independent atoms.) Applying the sin correction to a dihedral yields negative densities; molrs
correctly returns `is_angular() == false` for it (`dihedral.rs:34-38`).

**Why the degree default is a correctness bug, not a cosmetic one.**
`mod.rs:251` evaluates `bin_centers[i].sin()` on the stored numbers. On a degree axis those are
degrees-as-radians: `sin(150.5) = −0.292`, `sin(179.5) = −0.416`. Negative weights drive
`renormalize_density` (`histogram1d.rs:204-210`) to a total ≤ 0, which returns `Array1::zeros`. The
corrected ADF is then identically zero with no error raised. Measured on θ ~ N(170°, 5°), 20 000
samples, `n_bins=180`: `density_sin_corrected` sums to 0.0000 with the default range and 57.2958
with `min=0.0, max=π`. The dihedral degree default is less severe — no sin correction is applied —
but occupies only `2π/360 = 1.75 %` of the axis, discarding 98.25 % of the resolution.

**Analytic golden used by the tests.** Three angle samples at θ = 30°, 90°, 90°, binned into 3 bins
over `[0, π]` (centers π/6, π/2, 5π/6; sines ½, 1, ½):
`density = [1/π, 2/π, 0]`, `w_i = density_i / sin(c_i) = [2/π, 2/π, 0]`,
`∫w dθ = (4/π)(π/3) = 4/3`, so `density_sin_corrected = [3/(2π), 3/(2π), 0]`. The correction is
therefore **flat where the raw density is 1:2** — a property no uncorrected array can satisfy.
Verified against the published 0.12 wheel with explicit radian bounds: `bin_edges` =
`[0, 1.047197551197, 2.094395102393, 3.14159265359]`, `counts` = `[1, 2, 0]`, `density` =
`[0.3183098861837907, 0.6366197723675814, 0]`, `density_sin_corrected` =
`[0.47746482927568606, 0.477464829275686, 0]`.

**Reference implementation citations** (already carried at
`molrs/src/compute/distribution/mod.rs:14-16`): Brehm & Kirchner, *J. Chem. Inf. Model.* **2011**,
51, 2007–2023, doi:10.1021/ci200217w; Brehm, Thomas, Gehrke, Kirchner, *J. Chem. Phys.* **2020**,
152, 164105, doi:10.1063/5.0005078.

**1/sin pole amplification** (documented, not fixed here): near θ → 0 and θ → π the divisor
vanishes, so the corrected density amplifies counting noise. At `n_bins=180`, bin 0 divides by
`sin(0.5°) = 0.00873` — a 115× gain. On an isotropic 200 k sample whose corrected answer is exactly
flat, interior bins scatter 2.4 % while bin 0 is +24 % and bin 179 +11 %. See
<https://github.com/votca/csg/issues/140>.

## Design

Two PyO3 constructors change shape; nothing in `molrs/` (the core crate) changes behaviour.

**`PyAngleDistribution::new` / `PyDihedralDistribution::new`**
(`molrs-python/src/compute/analysis.rs:105` and `:136`) become

```rust
#[pyo3(signature = (n_bins, min=None, max=None))]
fn new(n_bins: usize, min: Option<NpF>, max: Option<NpF>) -> PyResult<Self> {
    let inner = match (min, max) {
        (None, None) => DistributionFunction::over_natural_range(AngleObservable, n_bins),
        (Some(min), Some(max)) => DistributionFunction::new(AngleObservable, n_bins, min, max),
        _ => return Err(PyValueError::new_err(
            "AngleDistribution: supply both `min` and `max` (radians), or neither \
             to use the observable's natural range [0, pi]",
        )),
    }
    .map_err(py_value_err)?;
    Ok(Self { inner })
}
```

The literal `180.0` / `-180.0` disappears and **no `PI` literal replaces it**. Swapping the
constants would re-declare, in a second place, a range the observable already owns — that
duplication is precisely what produced this defect. The range constant lives at `angle.rs:31` /
`dihedral.rs:40` and reaches the binding only through `over_natural_range`.

**Half-supplied bounds are an error.** The new `Option` pair admits a state the old signature could
not reach: exactly one of `min`/`max` given. The Closest pattern (`PyMolGraph::add_atom`,
`molrs-python/src/core/system/molgraph.rs:551`, twin at `:1312`) silently falls through to the
"bare" constructor on a partial tuple. This spec deliberately diverges, per CLAUDE.md § Potential
System: *"a style that omits `coulomb` / `dielectric` / `coulomb14scale` is an `Err`, never a silent
default."* A caller who writes `AngleDistribution(180, min=0.0)` means a custom range and must be
told the other half is missing, not handed a silent `[0, π]`. Everything else about the pattern is
followed exactly: `#[pyo3(signature = ...)]` with `None` defaults, `Option<T>` parameters, a match
on the option tuple dispatching between two core constructors.

**`PyDistanceDistribution::new` (`:166`) is untouched.** `DistanceObservable` does not override
`natural_range` (`observable.rs:162` default `None`), so `over_natural_range` correctly `Err`s for
it; its bounds stay mandatory positional parameters. The fix applies to 2 of 3 classes and the
asymmetry is intentional, not an oversight.

**No WASM change.** `WasmAngleDistribution::new(n_bins)` / `WasmDihedralDistribution::new(n_bins)`
(`molrs-wasm/src/compute.rs:5015`, `:5045`) never accepted bounds and already route through
`over_natural_range` (`:5030`, `:5054`). `molrs-capi`, `molrs-cxxapi` and `molrs-ffi` expose no
distribution surface at all. The PyO3 layer was the only caller of `DistributionFunction::new` for
an angular observable.

**Docs carried in the same change.** Both pyclass rustdocs and both stub docstrings state that
bounds are **radians** (mirroring the existing "Å or radians" wording at `mod.rs:211`), name the
natural range used when bounds are omitted, and — on `AngleDistribution` — warn about the 1/sin
pole amplification near θ = 0 and θ = π. `CombinedDistribution`'s docstring (`analysis.rs:257-259`,
`_lib.pyi:2024-2027`) gains the same units sentence; its API does not change.

**Stub-default gate.** `TestStubMatchesRuntime` (`molrs-python/tests/test_molrec.py:103`) already
AST-parses `_lib.pyi` and proves every declared member exists at runtime. It is generalized to also
prove every declared **default value** matches the runtime. Discovery is an AST scan over all
classes in the stub that declare an `__init__` with defaults — deliberately **not** a hand-written
class list, per `.claude/notes/notes.md` 2026-07-14 lesson 1 ("where a list can be
directory-scanned, it MUST be"). Runtime classes resolve from `molrs._lib` (the module the stub
actually describes) rather than the curated `molrs` namespace, which does not re-export
`AngleDistribution`. Classes whose runtime type exposes no `__text_signature__` go in a small
frozen ledger with a stated reason; the test asserts each ledger entry still lacks a signature, so
the ledger is shrink-only and self-cleaning.

### Reuse decision

- `DistributionFunction::over_natural_range` (`mod.rs:83`) — **reuse**. Four existing production
  call sites (`molrs-wasm/src/compute.rs:5030`, `:5054`; `molrs/benches/compute/distribution.rs:54`,
  `:70`). This spec adds the two that were missing.
- `AngleObservable::natural_range` (`angle.rs:31`), `DihedralObservable::natural_range`
  (`dihedral.rs:40`) — **reuse**, transitively via `over_natural_range`. No new range constant is
  introduced anywhere.
- `PyMolGraph::add_atom` optional-arg → core-default dispatch (`molgraph.rs:551`) — **pattern**.
  Followed for signature shape and match structure; diverges only on the half-supplied case (Err,
  not silent fall-through), justified above.
- `WasmAngleDistribution` / `WasmDihedralDistribution` (`molrs-wasm/src/compute.rs:5015`, `:5045`) —
  **reuse as-is**; verified already correct, no edit, and `molrs-wasm/pkg/*.d.ts` stays untouched.
- `TestStubMatchesRuntime` (`test_molrec.py:103`) — **generalize**. Promoted from "declared members
  exist" to "declared members exist **and** declared defaults match", serving both the pre-existing
  `MolRec`/`Observables` case and this one.
- `AxisSpec::over_natural_range` for `CombinedDistribution` — **new — rejected.** It would have
  exactly one call site, contrary to CLAUDE.md § *Inline until the second real use*, and
  `CombinedDistribution`'s bounds are already mandatory so it carries no wrong default. Docstring
  only.
- Test-fixture helper — **new**. `molrs-python/tests/test_compute.py:20` has a private `_make_frame`
  that builds atoms-only frames; the new module needs frames carrying `angles` / `dihedrals`
  topology blocks, so it defines its own builders locally rather than exporting a second-use helper
  to `conftest.py`. Geometry follows the ideal-known-geometry pattern of
  `molrs/src/ff/potential/angle/harmonic.rs:133-141`; topology-block construction follows
  `molrs/src/compute/distribution/observable.rs:277`; test naming follows
  `molrs/src/compute/order/steinhardt.rs:518` (`wl_absent_by_default`).

## Files to create or modify

- `molrs-python/src/compute/analysis.rs`
- `molrs-python/python/molrs/_lib.pyi`
- `molrs-python/tests/test_distribution.py` (new)
- `molrs-python/tests/test_molrec.py`
- `regressions/distribution-angular-default-range.py` (new)

## Tasks

- [ ] Write failing unit tests for the radian defaults (`molrs-python/tests/test_distribution.py` → `TestAngleDistribution`, `TestDihedralDistribution`): bin-edge and bin-index goldens plus the sin-corrected `3/(2π)` golden
- [ ] Write failing unit tests for the bound contract in `molrs-python/tests/test_distribution.py`: half-supplied `min`/`max` raises `ValueError`, `DistanceDistribution` still requires both
- [ ] Implement `Option<NpF>` bounds dispatching to `DistributionFunction::over_natural_range` in `PyAngleDistribution::new` and `PyDihedralDistribution::new` (`molrs-python/src/compute/analysis.rs`)
- [ ] Update `AngleDistribution` / `DihedralDistribution` stubs to `min: Optional[float] = None, max: Optional[float] = None` in `molrs-python/python/molrs/_lib.pyi`
- [ ] Add docstring per rustdoc style with units: radians + natural range on both pyclasses and both stubs, 1/sin pole-amplification warning on `AngleDistribution`, radian units sentence on `CombinedDistribution` (`molrs-python/src/compute/analysis.rs`, `molrs-python/python/molrs/_lib.pyi`)
- [ ] Generalize `TestStubMatchesRuntime` in `molrs-python/tests/test_molrec.py` to compare every AST-discovered stub default against `inspect.signature` of the matching `molrs._lib` class
- [ ] Add regression example `regressions/distribution-angular-default-range.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Verify against the reverse gate: temporarily restore `min=0.0, max=180.0` / `min=-180.0, max=180.0` in `molrs-python/src/compute/analysis.rs`, rebuild, confirm the new tests go red, revert
- [ ] Run full check + test suite

## Testing strategy

**Every obvious assertion about this bug cannot fail.** Measured against the published 0.12 wheel
with the degree default, `n_bins=180`, 4 000 isotropic triplets: `compute` returns Ok;
`n_raw_samples == n_binned == 4000` (nothing skipped); `density` integrates to exactly 1.0;
`density.shape == (180,)`. A criterion asserting success, no dropped samples, normalization, or
shape is worthless here. Every test below therefore asserts on a **bin index**, a **bin edge**, or a
**corrected-density value**, and each carries a stated reason it bites.

Unit tests live in `molrs-python/tests/test_distribution.py` (flat layout, mirroring the existing
`tests/test_<area>.py` convention; there is no `tests/test_compute/` package). Endpoint columns are
inserted as `np.uint32` — molrs `U = u32`, and `AtomGroups::from_frame` reads them via `get_uint`
(`observable.rs:70-76`) — following `test_compute.py:106`. Frames carry `molrs.Box.cube(100.0)` so
minimum-image never wraps the sub-ångström fixtures. Per `.claude/notes/testing.md`, these are
**seam** tests: the defect is in the binding, so the binding is where it must be gated; no
multi-molecule numerical corpus is added.

**Angle fixture** — 5 atoms, vertex at index 1: `0=(1,0,0)`, `1=(0,0,0)`, `2=(0.8660254037844387,
0.5, 0)`, `3=(0,1,0)`, `4=(0,0,1)`; `angles` block `atomi=[0,0,0]`, `atomj=[1,1,1]`,
`atomk=[2,3,4]` → samples 30°, 90°, 90°.

1. *Happy path — the axis is radians.* `AngleDistribution(3).compute(frame)`:
   `bin_edges == [0.0, 1.0471975511965976, 2.0943951023931953, 3.141592653589793]` (atol 1e-12).
   **Bites** because the degree default gives `[0, 60, 120, 180]`.
2. *Happy path — samples land in the right bins.* Same result: `counts == [1.0, 2.0, 0.0]`
   (atol 1e-9) and `density == [0.3183098861837907, 0.6366197723675814, 0.0]` (= `[1/π, 2/π, 0]`).
   **Bites** because on the degree axis all three radian samples fall below the first bin center
   and clamp into bin 0, giving `counts == [3.0, 0.0, 0.0]`.
3. *Domain validation — the sin θ correction is alive and correct.*
   `density_sin_corrected == [0.4774648292756860, 0.4774648292756860, 0.0]` (= `3/(2π)`, atol
   1e-9), i.e. **flat where `density` is 1:2**. **Bites twice**: with the degree default the total
   weight is negative (`sin(30 rad) = −0.988`) so `renormalize_density` returns
   `[0.0, 0.0, 0.0]` — hard-verified for this exact fixture, not a probabilistic claim; and an
   implementation that returned the raw density unchanged would fail the flatness, so the test
   cannot pass on an uncorrected array.

**Dihedral fixture** — 4 atoms `0=(0,1,0)`, `1=(0,0,0)`, `2=(1,0,0)`, `3=(1,0,1)`; `dihedrals`
block `atomi=[0]`, `atomj=[1]`, `atomk=[2]`, `atoml=[3]`. `b1×b2 = (0,0,1)`, `b2×b3 = (0,−1,0)`, so
`φ = atan2(1, 0) = +π/2` **exactly** (no rounding).

4. *Happy path — signed radian axis.* `DihedralDistribution(6).compute(frame)`:
   `bin_edges[0] == -3.141592653589793`, `bin_edges[-1] == 3.141592653589793` (atol 1e-12), and
   `counts == [0, 0, 0, 0, 1.0, 0]` (atol 1e-9) — `+π/2` is exactly the center of bin 4.
   **Bites** because the degree axis clamps `1.5708` near the middle, splitting it as
   `counts[2] = 0.473820061220085`, `counts[3] = 0.526179938779915`, and reports
   `bin_edges[-1] == 180.0`.
5. *Reverse gate — a dihedral is never sin-corrected.* Same result: `angular is False` and
   `density_sin_corrected is None`. Asserts **absence**; it goes red if anyone later "unifies" the
   two classes by turning the ADF Jacobian on for torsions, which would produce negative densities
   (§ Domain basis).

**Edge cases.**

6. `AngleDistribution(6, min=0.0)` and `DihedralDistribution(6, max=3.14)` each raise `ValueError`
   whose message names both `min` and `max`. **Bites** on a silent-half-default regression, the
   exact failure mode of the Closest pattern this design diverges from.
7. `AngleDistribution(6, min=0.0, max=math.pi)` still honours explicit bounds
   (`bin_edges[-1] == π`), and `DistanceDistribution(6)` still raises `TypeError` for missing
   required arguments. **Bites** on an over-eager change that makes distance bounds optional —
   `over_natural_range` would then `Err` at runtime instead of failing at the call site.
8. `pytest -k stub` (`test_molrec.py::TestStubMatchesRuntime`) — the generalized gate reports
   `AngleDistribution` and `DihedralDistribution` among the classes it compared, and finds no
   default mismatch. **Bites** on the whole bug class: a stub that says `180.0` while the runtime
   says `None` is now a test failure, which is what would have caught this in the first place.

**Regression example** — `regressions/distribution-angular-default-range.py`: a minimal public-API
script (`import molrs`; `molrs.compute.distribution.AngleDistribution` /
`DihedralDistribution`) that rebuilds both fixtures, calls both classes with **default** bounds, and
asserts the hard-coded goldens `π = 3.141592653589793`, `counts == [1, 2, 0]`,
`density_sin_corrected == [0.4774648292756860, 0.4774648292756860, 0.0]`,
`counts == [0, 0, 0, 0, 1, 0]`, `density_sin_corrected is None`, plus the half-bound `ValueError`.
All goldens are analytic (derived in § Domain basis), so no oracle tool is involved and no
third-party package is imported at runtime.

**Reverse-gate proof.** Task 8 requires restoring the degree literals, rebuilding the extension, and
recording which tests go red. A gate that has never been red is indistinguishable from no gate
(`.claude/notes/notes.md` 2026-07-14, lesson 6). Expected red set: tests 1, 2, 3, 4 and the
regression script; expected still-green: tests 5, 6, 7 (they do not depend on the range).

## Out of scope

Recorded as follow-ups, named here so none of it is silent (CLAUDE.md § *Iron law — no silent
debt*). None is a blocker for this spec; each is a defect or debt found while drafting it.

- **`renormalize_density` silently returns zeros on a non-positive total**
  (`molrs/src/compute/distribution/histogram1d.rs:204-210`). This spec removes the only way to reach
  it from the Python API, but a Rust caller writing
  `DistributionFunction::new(AngleObservable, n, 0.0, 180.0)` still gets a silently-zeroed corrected
  density instead of an `Err`. Core behaviour change — separate spec.
- **`Observable::is_angular` conflates two facts** (`observable.rs:154`): "samples are in radians"
  and "needs the solid-angle Jacobian". A dihedral is angular yet must not be sin-corrected. Rename
  to `needs_solid_angle_correction` — a cross-crate refactor.
- **`dihedral.rs:25` rustdoc says `(−π, π]`** but IEEE `atan2(−0.0, x<0)` returns exactly `−π`, so
  the attainable set is closed `[−π, π]`. Harmless (`Histogram1d` bins both endpoints); a core
  rustdoc fix, with the IUPAC `(−π, π]` convention cited separately.
- **`mod.rs:252`'s `|sin| > 1e-12` guard is dead code** — it would need `n_bins > 1.6e12` to fire.
  A meaningful guard is on bin occupancy, not on `|sin|`. The pole amplification is documented in
  this spec; changing the guard is not.
- **`histogram1d.rs:184` CIC deposition clamps rather than wraps**, so a circular dihedral trans peak
  splits across bins 0 and n−1. Measured against a wrapped-CIC reference on von Mises(π, κ=8): max
  0.03 % deviation. Real, negligible; must not be allowed to grow this fix.
- **`CombinedDistribution` API** (`analysis.rs:270`, `combined.rs:59`) keeps mandatory bounds; no
  `AxisSpec::over_natural_range`. Docstring units sentence only, which **is** in scope.
- **`PyMolGraph::add_atom` (`molgraph.rs:551`, twin `:1312`) silently accepts a partial coordinate
  tuple** and falls through to `add_atom_bare` — the same half-supplied-tuple class this spec turns
  into an error for the distributions. Behaviour change to a widely-used constructor; separate spec.
- **TRAVIS-parity claim** (`mod.rs:8`, "Ported from the reference implementation"): both cited papers
  were paywalled and unfetchable during drafting, so whether TRAVIS's ADF default applies the sin
  correction is unconfirmed. Either verify with someone who has access or soften "ported" to
  "follows".
- **Moving `TestStubMatchesRuntime` out of `test_molrec.py`** into its own module now that it gates
  the whole stub rather than one class. Deferred to keep this diff to the defect.
- **All molpy changes.** `molpy/src/molpy/compute/distribution.py:61` and `:81` mirror the same wrong
  defaults with docstrings that say "Angle range in degrees";
  `molpy/tests/test_compute/test_distribution.py` asserts only `result.density.shape == (30,)` and
  its `test_dihedral_distribution_reads_dihedrals_from_frame` exercises the bug and passes. None of
  it can move until a molrs release carries this fix, per CLAUDE.md § *Release before molpy*.
