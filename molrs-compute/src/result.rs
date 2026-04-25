//! Traits carried by every [`Compute`](crate::Compute) output.
//!
//! - [`ComputeResult`] — "finalize into a fully-usable value". Multi-frame
//!   accumulations can return a not-yet-normalized intermediate value from
//!   `Compute::compute`; `ComputeResult::finalize` turns it into the
//!   user-facing final form. [`Graph::run`](crate::Graph) calls it once for
//!   every node before inserting into the [`Store`](crate::Store).
//! - [`DescriptorRow`] — "flatten into an `&[F]` row". Used by downstream
//!   matrix consumers such as PCA and k-means to treat any prior Compute
//!   output as a descriptor row without an extra conversion step.

use molrs::types::F;

/// Marker + finalization hook for Compute outputs.
///
/// `finalize` is called **once** by [`Graph::run`](crate::Graph) on the output
/// of every node before the value is moved into the [`Store`](crate::Store).
/// The default is a no-op, which suits outputs already in their final form
/// (cluster assignments, per-frame observables, etc.). Accumulating outputs
/// (notably RDF's raw pair histogram) override it to perform their
/// normalization step.
///
/// # Invariant
///
/// Implementations **must** be idempotent: calling `finalize` twice must
/// yield the same state as calling it once. Graph calls it exactly once, but
/// users may call it again on persisted results.
pub trait ComputeResult {
    /// Convert any accumulated intermediate state into the final user-facing form.
    fn finalize(&mut self) {}
}

/// Flatten a Compute output into a row of floats.
///
/// Consumed by downstream multi-frame analyses (PCA, k-means) that treat a
/// `Vec<T: DescriptorRow>` as a row-major matrix. Implementations must return
/// a **consistent** row length across calls on a given type — changing
/// dimension between rows is a programmer error that the caller checks at
/// matrix assembly time.
pub trait DescriptorRow {
    /// The flat row representation.
    fn as_row(&self) -> &[F];
}

/// Per-frame sequences automatically finalize by propagating to each element.
impl<T: ComputeResult + Clone + Send + Sync + 'static> ComputeResult for Vec<T> {
    fn finalize(&mut self) {
        for item in self.iter_mut() {
            item.finalize();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct NoopResult;
    impl ComputeResult for NoopResult {}

    #[derive(Clone)]
    struct Counter {
        raw: f64,
        norm: f64,
        finalized: bool,
    }

    impl ComputeResult for Counter {
        fn finalize(&mut self) {
            if !self.finalized {
                self.norm = self.raw / 10.0;
                self.finalized = true;
            }
        }
    }

    #[derive(Clone)]
    struct Row(Vec<F>);
    impl DescriptorRow for Row {
        fn as_row(&self) -> &[F] {
            &self.0
        }
    }

    #[test]
    fn default_finalize_is_noop() {
        let mut x = NoopResult;
        x.finalize();
        x.finalize();
    }

    #[test]
    fn finalize_is_idempotent() {
        let mut c = Counter {
            raw: 50.0,
            norm: 0.0,
            finalized: false,
        };
        c.finalize();
        assert!((c.norm - 5.0).abs() < 1e-12);
        c.finalize();
        assert!((c.norm - 5.0).abs() < 1e-12);
    }

    #[test]
    fn descriptor_row_as_row() {
        let r = Row(vec![1.0, 2.0, 3.0]);
        assert_eq!(r.as_row(), &[1.0, 2.0, 3.0]);
    }
}
