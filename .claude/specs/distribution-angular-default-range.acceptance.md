---
spec: distribution-angular-default-range
created: 2026-08-06
criteria:
  - id: ac-001
    summary: Angular constructors delegate to over_natural_range, declare no range literal
    type: code
    pass_when: |
      In molrs-python/src/compute/analysis.rs, PyAngleDistribution::new and
      PyDihedralDistribution::new each carry
      `#[pyo3(signature = (n_bins, min=None, max=None))]` with `Option<NpF>`
      parameters and call `DistributionFunction::over_natural_range` on the
      (None, None) branch; neither signature attribute nor either function body
      contains `180.0`, `-180.0`, `PI`, or `std::f64::consts`.
    status: pending
  - id: ac-002
    summary: DistanceDistribution keeps mandatory bounds
    type: code
    pass_when: |
      PyDistanceDistribution::new still declares
      `#[pyo3(signature = (n_bins, min, max))]` with non-Option `NpF`
      parameters and still calls `DistributionFunction::new`; it does not call
      over_natural_range (DistanceObservable::natural_range is None).
    status: pending
  - id: ac-003
    summary: A half-supplied bound raises ValueError naming both min and max
    type: runtime
    pass_when: |
      `AngleDistribution(6, min=0.0)` and `DihedralDistribution(6, max=3.14)`
      each raise ValueError whose message contains both "min" and "max"; and
      `AngleDistribution(6, min=0.0, max=math.pi).compute(f).bin_edges[-1]`
      equals math.pi, proving explicit bounds still work.
    status: pending
  - id: ac-004
    summary: Angle default axis is radians — asserted on bin edges and bin index
    type: runtime
    pass_when: |
      For the 30/90/90-degree three-triplet fixture in
      molrs-python/tests/test_distribution.py,
      `AngleDistribution(3).compute(frame)` returns
      bin_edges == [0.0, 1.0471975511965976, 2.0943951023931953,
      3.141592653589793] (atol 1e-12) and counts == [1.0, 2.0, 0.0]
      (atol 1e-9). Shape, n_binned and normalization assertions do NOT satisfy
      this criterion — they pass under the defect.
    status: pending
  - id: ac-005
    summary: sin-theta corrected ADF is non-zero and analytically correct
    type: runtime
    pass_when: |
      Same fixture: density == [0.3183098861837907, 0.6366197723675814, 0.0]
      (= [1/pi, 2/pi, 0]) and density_sin_corrected ==
      [0.4774648292756860, 0.4774648292756860, 0.0] (= 3/(2*pi)), atol 1e-9 —
      i.e. the corrected array is flat where the raw density is 1:2, and its
      sum is non-zero.
    status: pending
  - id: ac-006
    summary: Dihedral default spans [-pi, pi] and is never sin-corrected
    type: runtime
    pass_when: |
      For the +pi/2 single-quadruple fixture,
      `DihedralDistribution(6).compute(frame)` returns
      bin_edges[0] == -3.141592653589793 and bin_edges[-1] ==
      3.141592653589793 (atol 1e-12), counts == [0, 0, 0, 0, 1.0, 0]
      (atol 1e-9), `angular is False`, and `density_sin_corrected is None`.
    status: pending
  - id: ac-007
    summary: Stub declares optional radian bounds on both angular classes
    type: code
    pass_when: |
      In molrs-python/python/molrs/_lib.pyi, AngleDistribution.__init__ and
      DihedralDistribution.__init__ both read
      `min: Optional[float] = None, max: Optional[float] = None`; the literals
      180.0 and -180.0 appear nowhere in the AngleDistribution,
      DihedralDistribution or CombinedDistribution class blocks; and the
      docstring of each of those three classes contains the word "radians".
    status: pending
  - id: ac-008
    summary: 1/sin pole amplification documented where a Python user reads it
    type: code
    pass_when: |
      The PyAngleDistribution rustdoc in
      molrs-python/src/compute/analysis.rs and the AngleDistribution docstring
      in molrs-python/python/molrs/_lib.pyi each state that the sin-theta
      correction amplifies noise near theta = 0 and theta = pi, and each names
      the natural range [0, pi] used when bounds are omitted.
    status: pending
  - id: ac-009
    summary: Stub-default gate is AST-discovered, covers the angular classes, green
    type: runtime
    pass_when: |
      `TestStubMatchesRuntime` in molrs-python/tests/test_molrec.py contains a
      test that discovers, by AST scan of _lib.pyi with no hand-written class
      list, every class whose __init__ declares defaults, resolves each from
      `molrs._lib`, and asserts each stub default equals the corresponding
      `inspect.signature` default. The test is green and the set of classes it
      compared includes AngleDistribution and DihedralDistribution. Any class
      excluded for lacking __text_signature__ sits in a frozen ledger that the
      same test proves is shrink-only.
    status: pending
  - id: ac-010
    summary: Regression example reproduces the analytic goldens offline
    type: runtime
    pass_when: |
      `python regressions/distribution-angular-default-range.py` exits 0. It
      imports only `molrs` (no numpy-external, no RDKit/freud/AmberTools, no
      subprocess), calls AngleDistribution and DihedralDistribution with
      default bounds, and asserts the hard-coded values
      3.141592653589793, counts [1, 2, 0],
      density_sin_corrected [0.4774648292756860, 0.4774648292756860, 0.0],
      counts [0, 0, 0, 0, 1, 0], and `density_sin_corrected is None`
      (atol 1e-9).
    status: pending
  - id: ac-011
    summary: Reverse gate — restoring the degree default turns the new tests red
    type: runtime
    pass_when: |
      With `min=0.0, max=180.0` and `min=-180.0, max=180.0` temporarily
      restored in molrs-python/src/compute/analysis.rs and the extension
      rebuilt, `pytest molrs-python/tests/test_distribution.py` fails the
      assertions behind ac-004, ac-005 and ac-006, and
      `python regressions/distribution-angular-default-range.py` exits
      non-zero; reverting restores green. The red set is recorded in the
      /mol:impl close-out summary.
    status: pending
out_of_scope:
  - "renormalize_density returning zeros instead of Err on a non-positive total (histogram1d.rs:204)"
  - "Renaming Observable::is_angular to needs_solid_angle_correction (observable.rs:154)"
  - "Core rustdoc nit: dihedral attainable set is closed [-pi, pi], not (-pi, pi] (dihedral.rs:25)"
  - "Replacing the dead |sin| > 1e-12 guard with a bin-occupancy guard (mod.rs:252)"
  - "Circular CIC wrap for dihedrals (histogram1d.rs:184; measured 0.03%)"
  - "CombinedDistribution API change / AxisSpec::over_natural_range — docstring only"
  - "PyMolGraph::add_atom silently accepting a partial coordinate tuple (molgraph.rs:551, :1312)"
  - "Confirming or softening the TRAVIS 'ported from' claim (mod.rs:8)"
  - "Moving TestStubMatchesRuntime into its own test module"
  - "All molpy changes (distribution.py:61,:81 defaults and 'degrees' docstrings, shape-only tests) — blocked on a molrs release per CLAUDE.md 'Release before molpy'"
---

# Acceptance — distribution-angular-default-range

Done means: a Python caller who writes `AngleDistribution(180)` or
`DihedralDistribution(180)` gets a histogram whose axis is the observable's own
radian range, the solid-angle-corrected ADF is a real distribution rather than
an array of zeros, and the class of defect — a stub or a binding re-declaring a
constant the core already owns — is gated so it cannot recur silently. The bar
is deliberately narrow on *what* is asserted: this bug is invisible to success,
sample-count, normalization and shape checks, all of which pass while the axis
is wrong. Only bin edges, bin indices and corrected-density values count.

## AC-001 — Constructors delegate, they do not re-declare

The fix is *delegation*, not a constant swap. Replacing `180.0` with `PI` would
leave the range declared in two places, which is what produced the defect: the
binding said degrees while `angle.rs:31` said `[0, pi]`. The grep for `PI` and
`std::f64::consts` in these two functions is therefore part of the bar, not
pedantry.

## AC-002 — The asymmetry is deliberate

`DistanceObservable` returns `None` from `natural_range` (`observable.rs:162`),
so `over_natural_range` correctly errors for it. A reviewer seeing "2 of 3
classes changed" should find this criterion rather than assume an oversight.

## AC-003 — Half-supplied bounds

The `Option` pair admits a state the old signature could not reach. The Closest
pattern (`molgraph.rs:551`) falls through silently; this spec diverges per
CLAUDE.md's "a style that omits ... is an Err, never a silent default". The
message must name both parameters so the caller knows which half is missing.

## AC-004 / AC-005 — The two halves of the angle defect

AC-004 covers the visible half (wrong axis, wrong bin). AC-005 covers the
silent half the scientist escalated to CRITICAL: `mod.rs:251` calls `.sin()` on
degree-valued bin centers, negative weights drive `renormalize_density`'s total
below zero, and `histogram1d.rs:206` returns zeros with no error. For the exact
three-sample fixture used here the degree path yields
`density_sin_corrected == [0, 0, 0]` deterministically — hand-verified, not a
sampling claim. The `3/(2*pi)` golden is analytic:
`w_i = density_i / sin(c_i) = [2/pi, 2/pi, 0]`, normalized by `(4/pi)(pi/3) =
4/3`. Its flatness against a 1:2 raw density is what makes the assertion
impossible to satisfy with an uncorrected array. Both goldens were reproduced
against the published 0.12 wheel with explicit radian bounds before this spec
was persisted.

## AC-006 — Reverse gate on the dihedral

Asserts absence: `density_sin_corrected is None` and `angular is False`. The
signed torsion's residual freedom at fixed bond geometry is SO(2), whose
invariant measure is `dphi` — no Jacobian (VOTCA CSG theory). Applying the ADF
correction here produces negative densities, and folding to `|phi|` collapses
g+ onto g- irreversibly. This criterion exists so a future "consistency" change
across the three classes has to argue with a test.

## AC-007 / AC-008 — Documentation is part of the fix

The stub is the only thing most Python users and every type checker read. A
stub that still says `max: float = 180.0` against a runtime default of `None`
is the same defect in a second file. AC-008 additionally lands the pole-
amplification warning where the affected user is (the Python docstring), rather
than only in core rustdoc.

## AC-009 — The gate that makes this class self-detecting

`TestStubMatchesRuntime` already proves declared members exist; it did not
prove declared *defaults* match, which is exactly the gap this bug lived in.
Discovery must be an AST scan, not a class list — per the 2026-07-14 lesson
that a hand-picked fixture list let one molecule hide a missing MMFF
electrostatics term for a month. Runtime lookup goes through `molrs._lib`
because `molrs/__init__.py` does not re-export `AngleDistribution`.

## AC-010 — Regression example

`regressions/distribution-angular-default-range.py`, public API only. Every
golden is analytic (derived in the spec's Domain basis), so no third-party
oracle is involved at authoring time or at run time, per CLAUDE.md's
"no third-party scientific software in the default test gate".

## AC-011 — Proof the gates bite

A gate that has never been red is indistinguishable from no gate. Because this
defect is invisible to every naive assertion, the bite-proof is the only
evidence that the new tests measure the thing that was broken. The impl
close-out must name which tests went red and confirm the revert.
