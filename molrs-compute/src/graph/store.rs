//! Result bag returned by [`Graph::run`](super::Graph).

use std::any::Any;
use std::collections::HashMap;

use super::node::{NodeId, Slot};

/// Typed bag of outputs produced by a single `Graph::run`.
///
/// Every node that ran has exactly one entry; each entry has been passed
/// through `ComputeResult::finalize` before insertion. Users retrieve outputs
/// with [`Store::get`] (borrow) or [`Store::take`] (consume).
pub struct Store {
    values: HashMap<NodeId, Box<dyn Any + Send + Sync>>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let keys: Vec<NodeId> = self.values.keys().copied().collect();
        f.debug_struct("Store").field("keys", &keys).finish()
    }
}

impl Store {
    pub(crate) fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Borrow the output of a node.
    ///
    /// # Panics
    ///
    /// Panics if the slot is not in the store or the stored type does not match `T`.
    /// Both conditions are unreachable for slots returned by the same `Graph`:
    /// `Graph::run` guarantees each slot is populated with the statically-typed
    /// value. Using a slot from a different `Graph` is a programmer error.
    pub fn get<T: 'static>(&self, slot: Slot<T>) -> &T {
        self.values
            .get(&slot.id())
            .and_then(|boxed| boxed.downcast_ref::<T>())
            .expect("Store::get: slot not in store or wrong type")
    }

    /// Consume the store to extract a single output by value.
    ///
    /// Useful when the caller wants ownership (e.g. for persistence or further
    /// processing). Subsequent `get` calls on other slots are impossible after
    /// `take` — it moves out of `self`.
    pub fn take<T: 'static>(mut self, slot: Slot<T>) -> T {
        let boxed = self
            .values
            .remove(&slot.id())
            .expect("Store::take: slot not in store");
        *boxed
            .downcast::<T>()
            .unwrap_or_else(|_| panic!("Store::take: wrong type for slot"))
    }

    /// Number of stored outputs.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the store holds no outputs.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub(crate) fn insert_any(&mut self, id: NodeId, value: Box<dyn Any + Send + Sync>) {
        self.values.insert(id, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_then_get_roundtrip() {
        let mut s = Store::new();
        let slot: Slot<i32> = Slot::new(NodeId::new(0));
        s.insert_any(NodeId::new(0), Box::new(42i32));
        assert_eq!(*s.get(slot), 42);
    }

    #[test]
    fn take_consumes_value() {
        let mut s = Store::new();
        let slot: Slot<String> = Slot::new(NodeId::new(1));
        s.insert_any(NodeId::new(1), Box::new("owned".to_string()));
        let out = s.take(slot);
        assert_eq!(out, "owned");
    }

    #[test]
    #[should_panic(expected = "not in store")]
    fn get_missing_panics() {
        let s = Store::new();
        let slot: Slot<u32> = Slot::new(NodeId::new(99));
        let _ = s.get(slot);
    }

    #[test]
    #[should_panic(expected = "not in store or wrong type")]
    fn get_wrong_type_panics() {
        let mut s = Store::new();
        s.insert_any(NodeId::new(0), Box::new(1i32));
        let slot: Slot<String> = Slot::new(NodeId::new(0));
        let _ = s.get(slot);
    }
}
