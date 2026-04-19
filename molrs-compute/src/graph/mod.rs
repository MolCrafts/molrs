//! Lightweight typed DAG for chaining [`Compute`] nodes with
//! shared intermediate results.
//!
//! A [`Graph`] is built declaratively: each `add` registers a `Compute` node
//! and its dependencies (extracted from a [`Store`] by a `deps_fn` closure),
//! and returns a typed [`Slot`] handle for downstream consumers. `Graph::run`
//! executes each node once in insertion order, finalizes each output, and
//! returns a [`Store`].
//!
//! **Key property**: in a diamond pattern like `Rg, Inertia → COM → Cluster`,
//! `Cluster` and `COM` each run exactly once, shared by all downstream
//! consumers via their `Slot` handles.
//!
//! # Execution order
//!
//! The typed API structurally guarantees that insertion order is a valid
//! topological order — a Slot can only be captured by a `deps_fn` after its
//! owning node has been registered, so every predecessor of node `i` lives in
//! `nodes[0..i]` (or in the input bag). Thus `Graph::run` iterates nodes in
//! registration order, no topo-sort at runtime. A standalone Kahn
//! implementation is retained privately for future parallel execution schemes.
//!
//! # Sub-modules
//!
//! - [`node`] — [`NodeId`] and typed [`Slot<T>`] handles
//! - [`inputs`] — [`Inputs`] bag for binding external values
//! - [`store`] — [`Store`] bag holding a run's outputs

pub mod inputs;
pub mod node;
pub mod store;
pub(crate) mod topo;

pub use inputs::Inputs;
pub use node::{NodeId, Slot};
pub use store::Store;

use std::any::Any;
use std::marker::PhantomData;

use molrs::frame_access::FrameAccess;

use crate::error::ComputeError;
use crate::result::ComputeResult;
use crate::traits::Compute;

/// Typed DAG of [`Compute`] nodes.
///
/// Generic over the frame type `FA` so the whole graph is monomorphized for a
/// single concrete frame at compile time. Compute impls are type-erased inside
/// the graph via an internal sealed trait — their concrete `Args` / `Output`
/// GATs are monomorphized into the wrapper but hidden behind `dyn`.
pub struct Graph<FA: FrameAccess + Sync + 'static> {
    nodes: Vec<Box<dyn ErasedCompute<FA>>>,
    input_ids: Vec<NodeId>,
    next_id: u32,
}

impl<FA: FrameAccess + Sync + 'static> Graph<FA> {
    /// Start a new, empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            input_ids: Vec::new(),
            next_id: 0,
        }
    }

    fn alloc_id(&mut self) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;
        id
    }

    /// Declare an external input of type `T`. The returned [`Slot`] is bound
    /// to a concrete value at `run` time via [`Inputs::with`].
    pub fn input<T: Send + Sync + 'static>(&mut self) -> Slot<T> {
        let id = self.alloc_id();
        self.input_ids.push(id);
        Slot::new(id)
    }

    /// Register a [`Compute`] node whose dependencies are extracted from the
    /// [`Store`] by `deps_fn`.
    ///
    /// `deps_fn` is a higher-rank closure so the store borrow passed to it
    /// can have any shorter lifetime than the graph itself. The returned
    /// [`Slot<C::Output>`] can be captured by later `add` calls to build a DAG.
    ///
    /// # Note on closures
    ///
    /// When rustc cannot infer the higher-rank bound from a bare `|store| …`
    /// closure (typical when `Args<'a>` is a tuple of references), annotate
    /// the parameter explicitly: `|store: &Store| …`. That makes the HRTB
    /// unambiguous for inference.
    pub fn add<C, DepsFn>(&mut self, compute: C, deps_fn: DepsFn) -> Slot<C::Output>
    where
        C: Compute + Send + Sync + 'static,
        DepsFn: for<'a> Fn(&'a Store) -> <C as Compute>::Args<'a> + Send + Sync + 'static,
    {
        let id = self.alloc_id();
        self.nodes.push(Box::new(Node::<C, DepsFn, FA> {
            id,
            compute,
            deps_fn,
            _phantom: PhantomData,
        }));
        Slot::new(id)
    }

    /// Execute every node once, in insertion order.
    ///
    /// Consumes `inputs`: every input slot declared by [`Graph::input`] must
    /// be bound, or the call returns [`ComputeError::MissingInput`]. Every
    /// node's output is passed through [`ComputeResult::finalize`] before it
    /// lands in the store.
    pub fn run(&self, frames: &[&FA], mut inputs: Inputs) -> Result<Store, ComputeError> {
        let mut store = Store::new();
        for &id in &self.input_ids {
            match inputs.take(id) {
                Some(value) => store.insert_any(id, value),
                None => return Err(ComputeError::MissingInput { slot: id }),
            }
        }
        for node in &self.nodes {
            node.run(frames, &mut store)?;
        }
        Ok(store)
    }
}

impl<FA: FrameAccess + Sync + 'static> Default for Graph<FA> {
    fn default() -> Self {
        Self::new()
    }
}

/// Type-erased Compute node held by [`Graph`].
///
/// Hides the concrete `C: Compute` behind a vtable so the graph can store
/// heterogeneous nodes in a single `Vec`. Each impl preserves the full GAT
/// chain internally, so calls through this trait remain fully typed.
trait ErasedCompute<FA: FrameAccess + Sync + 'static>: Send + Sync {
    fn run(&self, frames: &[&FA], store: &mut Store) -> Result<(), ComputeError>;
}

struct Node<C, DepsFn, FA>
where
    C: Compute + Send + Sync + 'static,
    DepsFn: for<'a> Fn(&'a Store) -> <C as Compute>::Args<'a> + Send + Sync + 'static,
    FA: FrameAccess + Sync + 'static,
{
    id: NodeId,
    compute: C,
    deps_fn: DepsFn,
    _phantom: PhantomData<fn() -> FA>,
}

impl<C, DepsFn, FA> ErasedCompute<FA> for Node<C, DepsFn, FA>
where
    C: Compute + Send + Sync + 'static,
    DepsFn: for<'a> Fn(&'a Store) -> <C as Compute>::Args<'a> + Send + Sync + 'static,
    FA: FrameAccess + Sync + 'static,
{
    fn run(&self, frames: &[&FA], store: &mut Store) -> Result<(), ComputeError> {
        let id = self.id;
        let output = {
            let store_ref: &Store = store;
            let args = (self.deps_fn)(store_ref);
            let mut out = self
                .compute
                .compute(frames, args)
                .map_err(|e| wrap_node_error(id, e))?;
            out.finalize();
            out
        };
        let boxed: Box<dyn Any + Send + Sync> = Box::new(output);
        store.insert_any(id, boxed);
        Ok(())
    }
}

fn wrap_node_error(node_id: NodeId, err: ComputeError) -> ComputeError {
    // Avoid nesting Node errors within Node errors: if an inner node has
    // already been tagged, preserve the original innermost source but carry
    // the outer node id.
    match err {
        ComputeError::Node { source, .. } => ComputeError::Node { node_id, source },
        other => ComputeError::Node {
            node_id,
            source: Box::new(other),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // --- Mock Compute impls (local to tests) ---

    /// Identity: emit a fixed value every time, no deps.
    #[derive(Clone)]
    struct Constant<T: Clone + Send + Sync + 'static>(T);

    impl<T: Clone + Send + Sync + 'static> Compute for Constant<T> {
        type Args<'a> = ();
        type Output = TestVal<T>;
        fn compute<'a, FA: FrameAccess + 'a>(
            &self,
            _frames: &[&'a FA],
            _args: (),
        ) -> Result<TestVal<T>, ComputeError> {
            Ok(TestVal(self.0.clone()))
        }
    }

    /// `Double` doubles the upstream `i64`.
    #[derive(Clone)]
    struct Double;
    impl Compute for Double {
        type Args<'a> = &'a TestVal<i64>;
        type Output = TestVal<i64>;
        fn compute<'a, FA: FrameAccess + 'a>(
            &self,
            _frames: &[&'a FA],
            upstream: &'a TestVal<i64>,
        ) -> Result<TestVal<i64>, ComputeError> {
            Ok(TestVal(upstream.0 * 2))
        }
    }

    /// Counting wrapper — tracks how many times `compute` is invoked.
    #[derive(Clone)]
    struct Counting {
        counter: Arc<AtomicUsize>,
    }
    impl Compute for Counting {
        type Args<'a> = ();
        type Output = TestVal<i64>;
        fn compute<'a, FA: FrameAccess + 'a>(
            &self,
            _frames: &[&'a FA],
            _args: (),
        ) -> Result<TestVal<i64>, ComputeError> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(TestVal(7))
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct TestVal<T: Clone + Send + Sync + 'static>(T);
    impl<T: Clone + Send + Sync + 'static> ComputeResult for TestVal<T> {}

    // --- Tests ---

    #[test]
    fn run_empty_graph_succeeds() {
        let g: Graph<Frame> = Graph::new();
        let frames: Vec<&Frame> = Vec::new();
        let store = g.run(&frames, Inputs::new()).unwrap();
        assert!(store.is_empty());
    }

    #[test]
    fn single_node_runs() {
        let mut g: Graph<Frame> = Graph::new();
        let s = g.add(Constant(42i64), |_store| ());
        let frame = Frame::new();
        let store = g.run(&[&frame], Inputs::new()).unwrap();
        assert_eq!(store.get(s), &TestVal(42i64));
    }

    #[test]
    fn linear_chain_runs_in_order() {
        let mut g: Graph<Frame> = Graph::new();
        let a = g.add(Constant(3i64), |_store| ());
        let b = g.add(Double, move |store| store.get(a));
        let c = g.add(Double, move |store| store.get(b));
        let frame = Frame::new();
        let store = g.run(&[&frame], Inputs::new()).unwrap();
        assert_eq!(store.get(c), &TestVal(12)); // 3 -> 6 -> 12
    }

    #[test]
    fn diamond_reuse_shared_node_runs_once() {
        // a (counting) feeds both b = 2a and c = 2a. Expect counter == 1.
        let counter = Arc::new(AtomicUsize::new(0));
        let mut g: Graph<Frame> = Graph::new();
        let a = g.add(
            Counting {
                counter: Arc::clone(&counter),
            },
            |_store| (),
        );
        let _b = g.add(Double, move |store| store.get(a));
        let _c = g.add(Double, move |store| store.get(a));
        let frame = Frame::new();
        let _store = g.run(&[&frame], Inputs::new()).unwrap();
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "shared node should execute exactly once per run"
        );
    }

    #[test]
    fn missing_input_returns_error() {
        let mut g: Graph<Frame> = Graph::new();
        let _input_slot = g.input::<i64>();
        let frame = Frame::new();
        let err = g.run(&[&frame], Inputs::new()).unwrap_err();
        assert!(matches!(err, ComputeError::MissingInput { .. }));
    }

    #[test]
    fn input_is_read_by_downstream() {
        let mut g: Graph<Frame> = Graph::new();
        let inp = g.input::<TestVal<i64>>();
        let out = g.add(Double, move |store| store.get(inp));
        let frame = Frame::new();
        let store = g
            .run(&[&frame], Inputs::new().with(inp, TestVal(21i64)))
            .unwrap();
        assert_eq!(store.get(out), &TestVal(42));
    }
}
