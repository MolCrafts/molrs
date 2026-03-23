use crate::Frame;
use crate::neighbors::NeighborList;

use super::error::ComputeError;
use super::reducer::Reducer;
use super::traits::{Compute, PairCompute};

/// Wraps a [`Compute`] with a [`Reducer`] to process trajectories.
pub struct Accumulator<C, R> {
    compute: C,
    reducer: R,
}

impl<C: Compute, R: Reducer<C::Output>> Accumulator<C, R> {
    pub fn new(compute: C, reducer: R) -> Self {
        Self { compute, reducer }
    }

    /// Compute on one frame and feed the result to the reducer.
    pub fn feed(&mut self, frame: &Frame) -> Result<(), ComputeError> {
        let output = self.compute.compute(frame)?;
        self.reducer.feed(output);
        Ok(())
    }

    /// Read the current accumulated result.
    pub fn result(&self) -> R::Output {
        self.reducer.result()
    }

    /// Reset the reducer to initial state.
    pub fn reset(&mut self) {
        self.reducer.reset();
    }

    /// Number of frames accumulated.
    pub fn count(&self) -> usize {
        self.reducer.count()
    }
}

/// Wraps a [`PairCompute`] with a [`Reducer`] to process trajectories.
pub struct PairAccumulator<C, R> {
    compute: C,
    reducer: R,
}

impl<C: PairCompute, R: Reducer<C::Output>> PairAccumulator<C, R> {
    pub fn new(compute: C, reducer: R) -> Self {
        Self { compute, reducer }
    }

    /// Compute on one frame+neighbors and feed the result to the reducer.
    pub fn feed(&mut self, frame: &Frame, neighbors: &NeighborList) -> Result<(), ComputeError> {
        let output = self.compute.compute(frame, neighbors)?;
        self.reducer.feed(output);
        Ok(())
    }

    /// Read the current accumulated result.
    pub fn result(&self) -> R::Output {
        self.reducer.result()
    }

    /// Reset the reducer to initial state.
    pub fn reset(&mut self) {
        self.reducer.reset();
    }

    /// Number of frames accumulated.
    pub fn count(&self) -> usize {
        self.reducer.count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Frame;
    use crate::compute::error::ComputeError;
    use crate::compute::reducer::SumReducer;
    use crate::compute::traits::Compute;

    /// Mock compute that always returns a fixed f64 value.
    struct MockCompute(f64);

    impl Compute for MockCompute {
        type Output = f64;

        fn compute(&self, _frame: &Frame) -> Result<f64, ComputeError> {
            Ok(self.0)
        }
    }

    // ---- 1. new has count 0 ----

    #[test]
    fn test_accumulator_new_and_count() {
        let acc = Accumulator::new(MockCompute(1.0), SumReducer::<f64>::new());
        assert_eq!(acc.count(), 0);
    }

    // ---- 2. feed and result ----

    #[test]
    fn test_accumulator_feed_and_result() {
        let mut acc = Accumulator::new(MockCompute(42.0), SumReducer::<f64>::new());
        let frame = Frame::new();
        acc.feed(&frame).unwrap();
        assert_eq!(acc.count(), 1);
        let result = acc.result().unwrap();
        assert!((result - 42.0).abs() < 1e-12);
    }

    // ---- 3. reset ----

    #[test]
    fn test_accumulator_reset() {
        let mut acc = Accumulator::new(MockCompute(10.0), SumReducer::<f64>::new());
        let frame = Frame::new();
        acc.feed(&frame).unwrap();
        assert_eq!(acc.count(), 1);
        acc.reset();
        assert_eq!(acc.count(), 0);
        assert!(acc.result().is_none());
    }

    // ---- 4. multiple feeds with SumReducer ----

    #[test]
    fn test_accumulator_multiple_feeds() {
        let mut acc = Accumulator::new(MockCompute(5.0), SumReducer::<f64>::new());
        let frame = Frame::new();
        acc.feed(&frame).unwrap();
        acc.feed(&frame).unwrap();
        acc.feed(&frame).unwrap();
        assert_eq!(acc.count(), 3);
        let result = acc.result().unwrap();
        assert!((result - 15.0).abs() < 1e-12);
    }
}
