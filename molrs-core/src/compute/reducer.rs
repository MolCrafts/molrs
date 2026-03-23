/// Strategy for combining per-frame results across a trajectory.
pub trait Reducer<T> {
    /// The type produced by finalization.
    type Output;

    /// Feed one per-frame result.
    fn feed(&mut self, value: T);

    /// Read the current accumulated result (snapshot, clones).
    fn result(&self) -> Self::Output;

    /// Reset to initial state.
    fn reset(&mut self);

    /// Number of items fed.
    fn count(&self) -> usize;
}

// ---------------------------------------------------------------------------
// SumReducer — element-wise accumulation via AddAssign
// ---------------------------------------------------------------------------

/// Element-wise sum of per-frame results (histograms, pair counts).
///
/// Requires `T: AddAssign + Clone`.
#[derive(Debug, Clone)]
pub struct SumReducer<T> {
    acc: Option<T>,
    count: usize,
}

impl<T> SumReducer<T> {
    pub fn new() -> Self {
        Self {
            acc: None,
            count: 0,
        }
    }

    /// Borrow the accumulated value without cloning.
    pub fn result_ref(&self) -> Option<&T> {
        self.acc.as_ref()
    }

    /// Consume the reducer and return the accumulated value.
    pub fn into_result(self) -> Option<T> {
        self.acc
    }
}

impl<T> Default for SumReducer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: std::ops::AddAssign + Clone> Reducer<T> for SumReducer<T> {
    type Output = Option<T>;

    fn feed(&mut self, value: T) {
        match &mut self.acc {
            Some(acc) => *acc += value,
            None => self.acc = Some(value),
        }
        self.count += 1;
    }

    fn result(&self) -> Self::Output {
        self.acc.clone()
    }

    fn reset(&mut self) {
        self.acc = None;
        self.count = 0;
    }

    fn count(&self) -> usize {
        self.count
    }
}

// ---------------------------------------------------------------------------
// ConcatReducer — collect per-frame results into a Vec
// ---------------------------------------------------------------------------

/// Collect per-frame results into a `Vec<T>`.
#[derive(Debug, Clone)]
pub struct ConcatReducer<T> {
    items: Vec<T>,
}

impl<T> ConcatReducer<T> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Borrow the collected items without cloning.
    pub fn result_ref(&self) -> &[T] {
        &self.items
    }

    /// Consume the reducer and return the collected items.
    pub fn into_result(self) -> Vec<T> {
        self.items
    }
}

impl<T> Default for ConcatReducer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Reducer<T> for ConcatReducer<T> {
    type Output = Vec<T>;

    fn feed(&mut self, value: T) {
        self.items.push(value);
    }

    fn result(&self) -> Self::Output {
        self.items.clone()
    }

    fn reset(&mut self) {
        self.items.clear();
    }

    fn count(&self) -> usize {
        self.items.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_reducer_accumulates() {
        let mut r = SumReducer::new();
        r.feed(1.0_f64);
        r.feed(2.0);
        r.feed(3.0);
        assert_eq!(r.count(), 3);
        assert!((r.result().unwrap() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn sum_reducer_empty_returns_none() {
        let r: SumReducer<f64> = SumReducer::new();
        assert!(r.result().is_none());
        assert_eq!(r.count(), 0);
    }

    #[test]
    fn sum_reducer_reset() {
        let mut r = SumReducer::new();
        r.feed(10.0_f64);
        r.reset();
        assert!(r.result().is_none());
        assert_eq!(r.count(), 0);
    }

    #[test]
    fn sum_reducer_borrow_and_consume() {
        let mut r = SumReducer::new();
        r.feed(5.0_f64);
        r.feed(3.0);
        assert!((r.result_ref().unwrap() - 8.0).abs() < 1e-12);
        assert!((r.into_result().unwrap() - 8.0).abs() < 1e-12);
    }

    #[test]
    fn concat_reducer_collects() {
        let mut r = ConcatReducer::new();
        r.feed(1);
        r.feed(2);
        r.feed(3);
        assert_eq!(r.count(), 3);
        assert_eq!(r.result(), vec![1, 2, 3]);
    }

    #[test]
    fn concat_reducer_reset() {
        let mut r = ConcatReducer::new();
        r.feed(42);
        r.reset();
        assert_eq!(r.count(), 0);
        assert!(r.result().is_empty());
    }

    #[test]
    fn concat_reducer_borrow_and_consume() {
        let mut r = ConcatReducer::new();
        r.feed(10);
        r.feed(20);
        assert_eq!(r.result_ref(), &[10, 20]);
        assert_eq!(r.into_result(), vec![10, 20]);
    }
}
