//! Unified [`Compute`] trait — the single public entry point for any analysis.

use molrs::store::frame_access::FrameAccess;

use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;

/// Run an analysis over a sequence of frames and produce a finalized result.
///
/// A single frame is just a length-1 slice — single-frame, trajectory, and
/// dataset analyses are structurally identical. The `Args` GAT carries any
/// non-frame input the analysis needs (neighbor lists, upstream Compute
/// outputs, masses). Concrete `Compute` impls decide whether to iterate the
/// slice, take the first frame as a reference, or treat the whole sequence as
/// a matrix.
///
/// # Contract
///
/// - `&self` is an **immutable parameter bag** (bin count, cutoff, seed, …).
///   No hidden mutable state — two `compute` calls with identical `frames` +
///   `args` must produce identical [`Output`](Self::Output) values.
/// - `Output` is always [`'static`](::std::marker::Send) +
///   [`Send`] + [`Sync`] + [`Clone`]: it must be shareable across
///   downstream consumers.
/// - `Output: ComputeResult` — callers should invoke
///   [`finalize`](ComputeResult::finalize) after `compute` to obtain
///   the user-facing final form.
///
/// # Example (conceptual)
///
/// ```ignore
/// struct COM;
/// struct COMResult { /* ... */ }
/// impl ComputeResult for COMResult {}
///
/// impl Compute for COM {
///     type Args<'a> = &'a Vec<ClusterResult>;
///     type Output = Vec<COMResult>;
///     fn compute<'a, FA: FrameAccess + 'a>(
///         &self,
///         frames: &[&'a FA],
///         clusters: Self::Args<'a>,
///     ) -> Result<Self::Output, ComputeError> {
///         // one COMResult per frame, aligned by index
///         # unimplemented!()
///     }
/// }
/// ```
pub trait Compute {
    /// Non-frame inputs, with a borrow tied to the frame-slice lifetime.
    type Args<'a>;

    /// Finalized output type.
    type Output: ComputeResult + Clone + Send + Sync + 'static;

    /// Run the analysis. `frames` may be empty; implementations that need at
    /// least one frame return [`ComputeError::EmptyInput`].
    ///
    /// `FA: Sync` lets impls that parallelize across frames (e.g. MSD, COM,
    /// inertia) share `&FA` across rayon threads without an explicit clone.
    /// Both `Frame` and `FrameView` satisfy it; user-defined FrameAccess types
    /// need only `Sync`, not `Send`.
    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError>;
}

/// Fit / smooth / spectral-transform an **upstream compute result** into a
/// scalar, curve, or spectrum.
///
/// `Fit` is the companion of [`Compute`]: where a `Compute` consumes raw
/// frames and produces a raw observable (MSD curve, ACF, …), a `Fit` consumes
/// that observable — a single [`Array1<f64>`](ndarray::Array1) curve or a
/// metadata-carrying raw result — and produces a derived quantity (an OLS
/// slope, a running trapezoid integral, a plateau mean, a windowed FFT
/// spectrum). The split keeps "what the simulation measured" separate from
/// "how the analyst processes it": the same raw curve can feed many fits with
/// different windows / fit ranges without recomputing the observable.
///
/// # Contract
///
/// - `&self` is an **immutable parameter bag** (fit window, prefactor,
///   dimension `d`, …). Identical `input` + identical `&self` ⇒ identical
///   [`Output`](Self::Output); no hidden mutable state.
/// - `Input<'a>` is a [GAT](https://doc.rust-lang.org/reference/items/associated-items.html)
///   so a fit may borrow its upstream input (`&Array1<f64>`, a raw ACF, …)
///   without cloning.
/// - `Output: ComputeResult + Clone + Send + Sync + 'static` — same shareable
///   bound as [`Compute::Output`], so fit outputs slot into the same
///   downstream consumers.
/// - Reuses [`ComputeError`]; no new error enum. Degenerate fit windows use
///   [`ComputeError::OutOfRange`], shape mismatches use
///   [`ComputeError::DimensionMismatch`], empty/too-short input uses
///   [`ComputeError::EmptyInput`].
///
/// # Example (conceptual)
///
/// ```ignore
/// struct LinearFit { window: (f64, f64) }
/// struct LinearFitResult { /* slope, intercept, r2, … */ }
/// impl ComputeResult for LinearFitResult {}
///
/// impl Fit for LinearFit {
///     type Input<'a> = (&'a Array1<f64>, &'a Array1<f64>); // (x, y)
///     type Output = LinearFitResult;
///     fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
///         # unimplemented!()
///     }
/// }
/// ```
pub trait Fit {
    /// Upstream input — a raw curve or a metadata-carrying raw result,
    /// borrowed for `'a`. Never frames.
    type Input<'a>;

    /// Finalized fit / transform output.
    type Output: ComputeResult + Clone + Send + Sync + 'static;

    /// Run the fit / transform. Returns [`ComputeError`] on degenerate
    /// windows, shape mismatches, or input too short to cover the requested
    /// lag range.
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError>;
}

/// The conclusion of a [`Check`], alongside the diagnostics that produced it.
///
/// Every check output must be able to answer "did it pass" without the caller
/// knowing which check ran. The diagnostics (residual, integral vs expected,
/// per-pair RMS, …) stay on the concrete type.
pub trait Verdict {
    /// Whether the data satisfied the criterion.
    fn passed(&self) -> bool;
}

/// Judge an **upstream result** against physics and return a verdict.
///
/// The third stage after [`Compute`] (frames → raw observable) and [`Fit`]
/// (raw observable → derived quantity). A `Check` consumes what those produce
/// — a spectrum, a set of route results — and reports whether it is
/// physically self-consistent.
///
/// # A failed check is `Ok`, never `Err`
///
/// This is the whole reason `Check` is not just another [`Fit`]: the two have
/// the same *shape* but opposite error semantics.
///
/// - `Err(ComputeError)` — the check **could not be evaluated**: mismatched
///   array lengths, too few frequency points, a non-positive volume.
/// - `Ok(v)` with `v.passed() == false` — the check **ran** and the data is
///   inconsistent. That is a successful measurement of a negative result.
///
/// Collapsing the two makes "the check crashed" indistinguishable from "the
/// physics is wrong", which is precisely the answer the caller wanted. A
/// `Fit` that hits a degenerate window returns `Err` because there is no
/// answer; a `Check` that finds a violated sum rule has *the* answer.
///
/// # Examples
///
/// ```
/// use molrs::compute::{Check, ConductivitySumRule, Verdict};
/// use ndarray::Array1;
///
/// let omega = Array1::from_vec(vec![0.0, 1.0, 2.0]);
/// let sigma = Array1::from_vec(vec![1.0, 1.0, 1.0]);
/// let rule = ConductivitySumRule { current_sq_mean: 1.0, volume: 1.0, temperature: 300.0 };
///
/// // Runs fine, and reports a verdict — pass or fail is data, not an error.
/// let v = rule.check((&omega, &sigma)).unwrap();
/// let _: bool = v.passed();
/// ```
pub trait Check {
    /// Upstream input — a computed curve or a set of named results, borrowed
    /// for `'a`. Never frames.
    type Input<'a>;

    /// The verdict, carrying both the conclusion and its diagnostics.
    type Output: Verdict + ComputeResult + Clone + Send + Sync + 'static;

    /// Evaluate the criterion. `Err` only when the criterion **cannot be
    /// evaluated** — never to report that it was not satisfied.
    fn check<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::result::ComputeResult;
    use ndarray::Array1;

    /// Trivial in-crate `Fit` over `&Array1<f64>` — exercises the GAT input and
    /// the `ComputeResult` output bound (ac-001).
    struct Sum;

    #[derive(Clone)]
    struct SumResult {
        total: f64,
    }
    impl ComputeResult for SumResult {}

    impl Fit for Sum {
        type Input<'a> = &'a Array1<f64>;
        type Output = SumResult;
        fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
            if input.is_empty() {
                return Err(ComputeError::EmptyInput);
            }
            Ok(SumResult {
                total: input.iter().sum(),
            })
        }
    }

    #[test]
    fn trivial_fit_impl_compiles_and_runs() {
        let curve = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let out = Sum.fit(&curve).unwrap();
        assert!((out.total - 6.0).abs() < 1e-12);
    }

    #[test]
    fn trivial_fit_empty_errors() {
        let curve: Array1<f64> = Array1::from_vec(vec![]);
        assert!(matches!(Sum.fit(&curve), Err(ComputeError::EmptyInput)));
    }
}
