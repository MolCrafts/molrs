//! External input bag for [`Graph::run`](super::Graph).

use std::any::Any;
use std::collections::HashMap;

use super::node::{NodeId, Slot};

/// Binds values to input Slots declared with `Graph::input`.
///
/// Each `with` call moves a value of the Slot's type into the bag. `Graph::run`
/// consumes the bag, inserting every value into the `Store` before executing
/// the first node.
#[derive(Default)]
pub struct Inputs {
    values: HashMap<NodeId, Box<dyn Any + Send + Sync>>,
}

impl Inputs {
    /// Build an empty bag.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind `value` to `slot`. If the same slot is bound twice the last binding wins.
    pub fn with<T: Send + Sync + 'static>(mut self, slot: Slot<T>, value: T) -> Self {
        self.values.insert(slot.id(), Box::new(value));
        self
    }

    /// Number of bound slots.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether no slots have been bound.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Whether `id` has been bound.
    pub fn contains(&self, id: NodeId) -> bool {
        self.values.contains_key(&id)
    }

    /// Remove and return a bound value by id. Graph uses this at `run` start.
    pub(crate) fn take(&mut self, id: NodeId) -> Option<Box<dyn Any + Send + Sync>> {
        self.values.remove(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_empty() {
        let inp = Inputs::new();
        assert!(inp.is_empty());
        assert_eq!(inp.len(), 0);
    }

    #[test]
    fn with_stores_values() {
        let slot: Slot<i32> = Slot::new(NodeId::new(1));
        let inp = Inputs::new().with(slot, 42i32);
        assert_eq!(inp.len(), 1);
        assert!(inp.contains(NodeId::new(1)));
    }

    #[test]
    fn with_last_binding_wins() {
        let slot: Slot<i32> = Slot::new(NodeId::new(1));
        let mut inp = Inputs::new().with(slot, 1).with(slot, 2);
        let boxed = inp.take(NodeId::new(1)).unwrap();
        let val = *boxed.downcast::<i32>().unwrap();
        assert_eq!(val, 2);
    }

    #[test]
    fn take_removes_value() {
        let slot: Slot<String> = Slot::new(NodeId::new(5));
        let mut inp = Inputs::new().with(slot, "hello".to_string());
        assert!(inp.take(NodeId::new(5)).is_some());
        assert!(inp.take(NodeId::new(5)).is_none());
        assert!(inp.is_empty());
    }
}
