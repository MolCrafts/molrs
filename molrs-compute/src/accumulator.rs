use molrs::frame_access::FrameAccess;

use super::error::ComputeError;
use super::reducer::Reducer;
use super::traits::Compute;

/// Wraps a [`Compute`] with a [`Reducer`] to process trajectories.
///
/// Feeds each frame's result to the reducer for accumulation (sum, concat, etc.).
pub struct Accumulator<C, R> {
    compute: C,
    reducer: R,
}

impl<C: Compute, R: Reducer<C::Output>> Accumulator<C, R> {
    pub fn new(compute: C, reducer: R) -> Self {
        Self { compute, reducer }
    }

    /// Compute on one frame and feed the result to the reducer.
    pub fn feed(
        &mut self,
        frame: &impl FrameAccess,
        args: C::Args<'_>,
    ) -> Result<(), ComputeError> {
        let output = self.compute.compute(frame, args)?;
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
    use crate::error::ComputeError;
    use crate::reducer::SumReducer;
    use crate::traits::Compute;
    use molrs::Frame;

    /// Mock compute that always returns a fixed f64 value.
    struct MockCompute(f64);

    impl Compute for MockCompute {
        type Args<'a> = ();
        type Output = f64;

        fn compute<FA: FrameAccess>(&self, _frame: &FA, _args: ()) -> Result<f64, ComputeError> {
            Ok(self.0)
        }
    }

    #[test]
    fn test_accumulator_new_and_count() {
        let acc = Accumulator::new(MockCompute(1.0), SumReducer::<f64>::new());
        assert_eq!(acc.count(), 0);
    }

    #[test]
    fn test_accumulator_feed_and_result() {
        let mut acc = Accumulator::new(MockCompute(42.0), SumReducer::<f64>::new());
        let frame = Frame::new();
        acc.feed(&frame, ()).unwrap();
        assert_eq!(acc.count(), 1);
        let result = acc.result().unwrap();
        assert!((result - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_accumulator_reset() {
        let mut acc = Accumulator::new(MockCompute(10.0), SumReducer::<f64>::new());
        let frame = Frame::new();
        acc.feed(&frame, ()).unwrap();
        assert_eq!(acc.count(), 1);
        acc.reset();
        assert_eq!(acc.count(), 0);
        assert!(acc.result().is_none());
    }

    #[test]
    fn test_accumulator_multiple_feeds() {
        let mut acc = Accumulator::new(MockCompute(5.0), SumReducer::<f64>::new());
        let frame = Frame::new();
        acc.feed(&frame, ()).unwrap();
        acc.feed(&frame, ()).unwrap();
        acc.feed(&frame, ()).unwrap();
        assert_eq!(acc.count(), 3);
        let result = acc.result().unwrap();
        assert!((result - 15.0).abs() < 1e-12);
    }
}
