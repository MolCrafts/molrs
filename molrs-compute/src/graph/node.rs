//! Node identifiers and typed slot handles for the Graph DAG.

use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

/// Stable unique identifier for a node inside a single [`Graph`](super::Graph).
///
/// Assigned monotonically by `Graph::input` / `Graph::add`; two `NodeId`s from
/// different graphs are not comparable (the collision probability is harmless
/// but no runtime invariant depends on cross-graph identity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub(crate) u32);

impl NodeId {
    pub(crate) const fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub fn raw(&self) -> u32 {
        self.0
    }
}

/// Typed handle to a node's output in a [`Graph`](super::Graph).
///
/// Zero-sized phantom type carries the output type so `deps_fn` closures can
/// extract typed references from a [`Store`](super::Store) without runtime
/// downcasts on the caller side. `Slot` is `Copy`, so a single slot can be
/// captured by multiple closures.
pub struct Slot<T> {
    pub(crate) id: NodeId,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Slot<T> {
    pub(crate) fn new(id: NodeId) -> Self {
        Self {
            id,
            _marker: PhantomData,
        }
    }

    /// The underlying [`NodeId`].
    pub fn id(self) -> NodeId {
        self.id
    }
}

// Manual `Clone` / `Copy` impls: derive would add a `T: Clone/Copy` bound
// because of how the derive macro handles type parameters, but `PhantomData<fn() -> T>`
// is always `Copy` regardless of T.
impl<T> Clone for Slot<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Slot<T> {}

impl<T> PartialEq for Slot<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for Slot<T> {}

impl<T> Hash for Slot<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl<T> std::fmt::Debug for Slot<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Slot").field(&self.id).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_is_copy() {
        let a: Slot<u32> = Slot::new(NodeId::new(7));
        let b = a;
        let c = a;
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a.id().raw(), 7);
    }

    #[test]
    fn slot_hashes_to_node_id() {
        use std::collections::HashMap;
        let a: Slot<i32> = Slot::new(NodeId::new(3));
        let b: Slot<i32> = Slot::new(NodeId::new(3));
        let mut m = HashMap::new();
        m.insert(a, "hi");
        assert_eq!(m.get(&b), Some(&"hi"));
    }

    #[test]
    fn node_id_order_is_numeric() {
        assert!(NodeId::new(0) < NodeId::new(1));
        assert!(NodeId::new(42) < NodeId::new(43));
    }
}
