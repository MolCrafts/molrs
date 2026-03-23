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
