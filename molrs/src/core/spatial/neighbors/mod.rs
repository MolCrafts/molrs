//! Neighbor-list algorithms for pairwise distance queries.
//!
//! A **neighbor list** is the table of particle pairs that lie closer together
//! than a fixed **cutoff** distance. Almost every pairwise quantity in
//! molecular simulation — short-range forces, the radial distribution function,
//! bond-order parameters, clustering — is computed by walking such a table, and
//! building it by testing every pair costs `O(N²)` distance evaluations for `N`
//! particles. The algorithms here bucket space so that the cost falls to
//! `O(N)`.
//!
//! ## Minimum image, sign convention, units
//!
//! Under periodic boundary conditions the simulation box is tiled infinitely
//! through space, so two particles have infinitely many separations — one per
//! periodic copy ("image") of the box. The **minimum-image convention**
//! (**MIC**) keeps only the shortest of them, and that shortest separation is
//! the only one this module ever reports.
//!
//! For a pair `(i, j)` a search produces two quantities:
//!
//! ```text
//! disp    = MIC(r_j - r_i)   displacement vector, pointing from i to j
//! dist_sq = |disp|^2         its squared length, from that same image
//! ```
//!
//! `disp` is **not** normalized — it is not divided by the distance, so its
//! length *is* the pair distance rather than 1. Both quantities come from the
//! single call [`SimBox::shortest_vector_impl`], so they always describe the
//! same periodic image and can never disagree. Lengths carry the unit of the
//! input coordinates, which is Å (ångström) everywhere in molrs; `disp` is
//! therefore in Å and `dist_sq` in Å².
//!
//! ## Algorithms
//!
//! - [`LinkCell`] — O(N) **cell list**: space is cut into a grid of cells at
//!   least one cutoff wide, so a particle can only have neighbors in its own
//!   cell and the 26 cells touching it. Parallelized with rayon when the
//!   `rayon` feature is enabled (default). Recommended for production use.
//! - [`BruteForce`] — O(N²) all-pairs reference implementation, useful for
//!   correctness testing and small systems.
//! - [`AabbQuery`] — bounding-volume tree; additionally answers
//!   k-nearest-neighbor queries ([`AabbQuery::query_knn`]), which a fixed
//!   cutoff cannot express.
//!
//! All three implement [`NbListAlgo`] and can be used through the lower-level
//! API or wrapped in [`NeighborQuery`] for a freud-style workflow.
//!
//! ## Streaming a pair versus materializing a table
//!
//! A search can hand each pair to a callback as it is discovered
//! ([`NbListAlgo::build_index`] + [`NbListAlgo::visit_pairs`], carrying the
//! [`NeighborPair`] quantities), or write the pairs into the [`Neighbors`]
//! table. A streamed pair always carries both physical columns. A materialized
//! table keeps only the columns [`NeighborsStorage`] asked for, and reports a
//! column it never stored as `None` — never as a fabricated zero. Two further
//! conventions pin the table down:
//!
//! - A **self-query** searches one point set against itself and is *half-shell*:
//!   each unordered pair appears exactly once, with `i < j`. A **cross-query**
//!   searches a set of query points against a separate reference set; it is
//!   directed and has no ordering constraint.
//! - [`NeighborsStorage::FULL`] means *every column is present*. It does **not**
//!   mean a full-shell (bidirectional) pair list. Column policy and pair
//!   direction are independent choices.
//!
//! ## High-level API (freud-style)
//!
//! ```ignore
//! let nq = NeighborQuery::new(&simbox, points.view(), 3.0);
//! let nlist = nq.query(query_points.view());   // cross-query
//! let nlist = nq.query_self();                  // self-query (i < j)
//!
//! nlist.query_point_indices()   // &[u32]        — i of each pair
//! nlist.point_indices()         // &[u32]        — j of each pair
//! nlist.dist_sq()               // Option<&[F]>  — Å², None if not stored
//! nlist.disp()                  // Option<FNx3View> — Å, None if not stored
//! ```
//!
//! ## Lower-level API
//!
//! ```ignore
//! let mut lc = LinkCell::new().cutoff(3.0);
//! lc.build(points.view(), &simbox);
//! let result = lc.query();
//! ```
//!
//! ## Name mapping from freud
//!
//! The API mirrors [freud-analysis](https://freud.readthedocs.io/) closely
//! enough that freud's documentation transfers, with two deliberate
//! differences: molrs stores the *squared* distance, and molrs makes the
//! physical columns opt-in.
//!
//! | freud | molrs | difference |
//! |---|---|---|
//! | `NeighborList.query_point_indices` | [`Neighbors::query_point_indices()`] | — |
//! | `NeighborList.point_indices` | [`Neighbors::point_indices()`] | — |
//! | `NeighborList.distances` (r, Å) | [`Neighbors::dist_sq()`] (r², Å²) | molrs stores the square and never hides a square root inside an accessor; take `.sqrt()` at the call site |
//! | `NeighborList.vectors` | [`Neighbors::disp()`] | same unnormalized MIC vector `r_j - r_i` |
//! | (both always present) | [`NeighborsStorage`] | freud always carries distances and vectors; molrs returns `None` for a column the search was told not to store |
//! | `freud.locality.LinkCell` | [`LinkCell`] | — |
//! | `freud.locality.AABBQuery` | [`AabbQuery`] | — |
//! | `freud.locality.FilterSANN` / `FilterRAD` | [`filter_sann`] / [`filter_rad`] | — |

use crate::spatial::simbox::SimBox;
use crate::types::{F, FNx3View};
use ndarray::ArrayView2;

pub mod aabb;
pub mod bruteforce;
pub mod filter;
pub mod grid;
mod linkcell;
pub mod periodic_buffer;
mod query;

pub use aabb::AabbQuery;
pub use bruteforce::BruteForce;
pub use filter::{filter_rad, filter_sann};
pub use grid::CellGrid;
pub use linkcell::LinkCell;
pub use periodic_buffer::{PeriodicBufferResult, periodic_buffer};
pub use query::NeighborQuery;

// NeighborsStorage and NeighborPair are defined below next to Neighbors.

// ---------------------------------------------------------------------------
// QueryMode — which point sets a table indexes, and how large they are
// ---------------------------------------------------------------------------

/// Which point sets a [`Neighbors`] table indexes, plus the size of each.
///
/// A search is one of two shapes. Either one point set is searched against
/// itself (*self-query*), in which case there is a single population and a
/// single count; or a set of query points is searched against a separate
/// reference set (*cross-query*), in which case there are two populations and
/// two counts.
///
/// The counts live inside the mode rather than beside it, so the illegal state
/// "self-query whose two counts disagree" cannot be written down at all. The
/// counts matter downstream: number densities (and therefore radial
/// distribution function normalization) are computed from them, and a
/// half-shell self-query contributes each pair once where a full-shell list
/// would contribute it twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMode {
    /// Self-query: pairs within the same point set, half-shell.
    ///
    /// Both `i` and `j` index that one set, and every pair satisfies `i < j`,
    /// so each unordered pair is present exactly once and no pair has `i == j`.
    SelfQuery {
        /// Size of the single point set both `i` and `j` index.
        num_points: usize,
    },
    /// Cross-query: `i` indexes the query points, `j` indexes the reference
    /// points.
    ///
    /// Directed and full-shell: there is no `i < j` constraint, and every query
    /// point reports all of its reference neighbors. If the caller passes the
    /// same coordinates as both sets, a point is its own neighbor at distance
    /// zero and both orderings of a pair appear — that is the defined
    /// behavior, not an accident, and it is why a cross-query must never be
    /// relabelled as a self-query.
    CrossQuery {
        /// Size of the query point set (`i` indices).
        num_query_points: usize,
        /// Size of the reference point set (`j` indices).
        num_points: usize,
    },
}

// ---------------------------------------------------------------------------
// PairVisitor -- zero-allocation callback for on-demand pair traversal
// ---------------------------------------------------------------------------

/// Callback for on-demand pair traversal (zero-allocation alternative to
/// [`Neighbors`]).
///
/// Implement this trait to process pairs without storing them. Used by
/// [`NbListAlgo::visit_pairs`]. A visitor always receives the complete physics
/// of a pair — the same two quantities a [`NeighborPair`] carries — because
/// nothing has been dropped by a storage policy yet.
pub trait PairVisitor {
    /// Called for each pair `(i, j)` within the cutoff.
    ///
    /// * `i`, `j` — original particle indices (indices into the point set the
    ///   search was built from, not into any sorted internal order)
    /// * `dist_sq` — squared minimum-image distance, Å²
    /// * `diff` — minimum-image displacement `r_j - r_i`, Å, unnormalized.
    ///   This is the same quantity the table calls `disp`
    ///   ([`NeighborPair::disp`]); the parameter keeps the older name for
    ///   source compatibility of existing visitors.
    fn visit_pair(&mut self, i: u32, j: u32, dist_sq: F, diff: [F; 3]);
}

/// Blanket impl: any `FnMut(u32, u32, F, [F; 3])` is a `PairVisitor`.
impl<T: FnMut(u32, u32, F, [F; 3])> PairVisitor for T {
    #[inline(always)]
    fn visit_pair(&mut self, i: u32, j: u32, dist_sq: F, diff: [F; 3]) {
        self(i, j, dist_sq, diff)
    }
}

// ---------------------------------------------------------------------------
// NbListAlgo trait
// ---------------------------------------------------------------------------

/// Trait implemented by neighbor-list algorithms.
///
/// Each algorithm maintains internal state and caches pair results after
/// [`build`](NbListAlgo::build) or [`update`](NbListAlgo::update), so that
/// [`query`](NbListAlgo::query) returns a cheap reference.
///
/// Every implementor searches one point set against itself, so the cached
/// table is always a half-shell self-query: `mode()` is
/// [`QueryMode::SelfQuery`] and every pair satisfies `i < j`. Cross-queries
/// against a second point set go through [`NeighborQuery`] instead.
pub trait NbListAlgo {
    /// Build the neighbor list from scratch.
    ///
    /// `points` is an `N × 3` view of Cartesian coordinates (Å) and `bx` is the
    /// simulation box supplying the periodicity used for minimum-image
    /// distances.
    ///
    /// # Panics
    /// Panics if the cutoff is not set (it defaults to zero and must be given a
    /// positive value first) or `points` does not have exactly 3 columns.
    fn build(&mut self, points: FNx3View<'_>, bx: &SimBox);

    /// Rebuild the neighbor list with new positions.
    ///
    /// Implementations may reuse internal buffers for efficiency. There is no
    /// skin/Verlet shortcut here: `update` recomputes the pairs, it does not
    /// decide for you whether a rebuild was necessary.
    fn update(&mut self, points: FNx3View<'_>, bx: &SimBox);

    /// Return a reference to the cached pair results.
    ///
    /// Before the first [`build`](NbListAlgo::build) this is an empty table
    /// tagged `SelfQuery { num_points: 0 }`, not an error.
    fn query(&self) -> &Neighbors;

    /// Return a reference to the simulation box used in the last build/update.
    fn box_ref(&self) -> &SimBox;

    /// Build the spatial index only — NO pair pre-computation.
    ///
    /// After calling this, [`visit_pairs`](NbListAlgo::visit_pairs) can
    /// traverse pairs on-demand without allocating a [`Neighbors`] table.
    /// The default falls back to a full [`build`](NbListAlgo::build).
    fn build_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.build(points, bx);
    }

    /// Rebuild the spatial index only — NO pair pre-computation.
    ///
    /// Equivalent to [`update`](NbListAlgo::update) but skips pair enumeration.
    /// The default falls back to a full [`update`](NbListAlgo::update).
    fn update_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.update(points, bx);
    }

    /// Traverse pairs on-demand, calling the visitor for each pair within
    /// the cutoff. Zero allocation — no [`Neighbors`] table is built.
    ///
    /// The default replays the pre-stored pairs from
    /// [`query`](NbListAlgo::query). A visitor is handed real physics or
    /// nothing, so that table must carry both physical columns; algorithms
    /// that enumerate on the fly (`LinkCell`, `BruteForce`) override this and
    /// compute the values directly.
    ///
    /// # Panics
    /// Panics if the cached table omits `dist_sq` or `disp`.
    fn visit_pairs(&self, visitor: &mut dyn PairVisitor) {
        let result = self.query();
        let dist_sq = result
            .dist_sq()
            .expect("visit_pairs replays a cached table: it must store dist_sq");
        let disp = result
            .disp()
            .expect("visit_pairs replays a cached table: it must store disp");
        for k in 0..result.n_pairs() {
            visitor.visit_pair(
                result.query_point_indices()[k],
                result.point_indices()[k],
                dist_sq[k],
                [disp[[k, 0]], disp[[k, 1]], disp[[k, 2]]],
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Algorithm-generic wrapper (lower-level API)
// ---------------------------------------------------------------------------

/// Algorithm-generic neighbor list wrapper (lower-level API).
///
/// Thin facade that delegates every call to the underlying [`NbListAlgo`]
/// implementation. Construct via `NbList(algo)`.
#[derive(Debug, Clone)]
pub struct NbList<A: NbListAlgo>(pub A);

impl<A: NbListAlgo> NbList<A> {
    /// Build the neighbor list from scratch. See [`NbListAlgo::build`].
    pub fn build(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.0.build(points, bx)
    }

    /// Rebuild with new positions. See [`NbListAlgo::update`].
    pub fn update(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.0.update(points, bx)
    }

    /// Return cached pair results. See [`NbListAlgo::query`].
    pub fn query(&self) -> &Neighbors {
        self.0.query()
    }

    /// Return the simulation box. See [`NbListAlgo::box_ref`].
    pub fn box_ref(&self) -> &SimBox {
        self.0.box_ref()
    }

    /// Build spatial index only (no pair pre-computation). See [`NbListAlgo::build_index`].
    pub fn build_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.0.build_index(points, bx)
    }

    /// Rebuild spatial index only. See [`NbListAlgo::update_index`].
    pub fn update_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.0.update_index(points, bx)
    }

    /// Traverse pairs on-demand. See [`NbListAlgo::visit_pairs`].
    pub fn visit_pairs(&self, visitor: &mut dyn PairVisitor) {
        self.0.visit_pairs(visitor)
    }
}

// ---------------------------------------------------------------------------
// Neighbors — the materialized pair table
// ---------------------------------------------------------------------------

/// Which per-pair physical columns a materialized [`Neighbors`] table keeps.
///
/// The pair indices `(i, j)` are always stored — two `u32`s, 8 bytes per pair.
/// The two physical columns are optional:
///
/// - [`dist_sq`](NeighborsStorage::dist_sq) — squared minimum-image distance in
///   Å², one `f64` (8 B) per pair.
/// - [`disp`](NeighborsStorage::disp) — minimum-image displacement
///   `r_j - r_i` in Å, three `f64`s (24 B) per pair.
///
/// # `FULL` is about columns, not about direction
///
/// [`FULL`](Self::FULL) means **every column is present**. It says nothing
/// about pair direction: a self-query stays half-shell (each unordered pair
/// once, `i < j`) under `FULL`, and an indices-only cross-query stays directed
/// and bidirectional. Storage policy and pair direction are orthogonal, and the
/// word "full" here never means "both `(i, j)` and `(j, i)` are stored".
///
/// # Choosing a policy
///
/// Full storage costs 40 B/pair. An analysis that only needs distances — a
/// radial distribution function histogram built from a materialized table, for
/// instance — should use [`DIST_SQ`](Self::DIST_SQ) at 16 B/pair; one that only
/// needs directions, such as a bond-order parameter, should use
/// [`DISP`](Self::DISP) at 32 B/pair. When pair counts are large, prefer not
/// materializing at all: [`NbListAlgo::build_index`] plus
/// [`NbListAlgo::visit_pairs`] streams every pair with both quantities and
/// allocates nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighborsStorage {
    /// Store squared minimum-image distances (Å²).
    pub dist_sq: bool,
    /// Store minimum-image displacement vectors `r_j - r_i` as `(dx, dy, dz)`
    /// (Å, unnormalized).
    pub disp: bool,
}

/// The default is [`FULL`](NeighborsStorage::FULL): unless a caller says
/// otherwise, a materialized table carries both physical columns, so no
/// downstream consumer is surprised by a missing one.
impl Default for NeighborsStorage {
    fn default() -> Self {
        Self::FULL
    }
}

impl NeighborsStorage {
    /// Indices only — lightest materialization (8 B/pair).
    ///
    /// Enough for connectivity questions (clustering, percolation) that never
    /// look at a distance or a direction.
    pub const INDICES_ONLY: Self = Self {
        dist_sq: false,
        disp: false,
    };
    /// Indices + squared distance `d²` (16 B/pair).
    ///
    /// Enough for any purely radial analysis, such as a radial distribution
    /// function histogram.
    pub const DIST_SQ: Self = Self {
        dist_sq: true,
        disp: false,
    };
    /// Indices + minimum-image displacement (32 B/pair).
    ///
    /// Enough for direction-dependent analyses (bond-order parameters), which
    /// can recover `d²` from the displacement themselves.
    pub const DISP: Self = Self {
        dist_sq: false,
        disp: true,
    };
    /// Every column present (40 B/pair) — *not* a bidirectional pair list.
    pub const FULL: Self = Self {
        dist_sq: true,
        disp: true,
    };
}

/// One neighbor pair carrying its full physics — the value a search emits.
///
/// A pair is never short a column at enumeration time. Writing `r_i` and `r_j`
/// for the two particle positions and `MIC(·)` for the minimum-image convention
/// (the shortest separation over all periodic copies of the box), the two
/// physical fields are
///
/// ```text
/// disp    = MIC(r_j - r_i)
/// dist_sq = |disp|^2
/// ```
///
/// Three properties follow and are worth stating outright, because each has
/// been a source of silent errors elsewhere:
///
/// - `disp` is **not normalized**. It is not divided by the distance and is not
///   a unit direction; its length is the pair distance itself. A caller that
///   wants a direction divides by `dist_sq.sqrt()`.
/// - `disp` points **from `i` to `j`**, so swapping the two indices flips its
///   sign.
/// - `dist_sq` is taken from the *same* minimum image as `disp`, in the same
///   evaluation, so `dist_sq == |disp|²` holds to floating-point rounding and
///   the two can never describe different periodic copies.
///
/// Columns become optional only on the way into a [`Neighbors`] table; a value
/// of this type always has all of them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeighborPair {
    /// Query point index (`i`).
    pub i: u32,
    /// Reference point index (`j`).
    pub j: u32,
    /// Squared minimum-image distance (Å²).
    pub dist_sq: F,
    /// Minimum-image displacement `r_j - r_i` (Å), unnormalized.
    pub disp: [F; 3],
}

/// Materialized neighbor pair table (freud-style).
///
/// Column store of every pair within the cutoff: two index columns that are
/// always present, plus whichever physical columns [`NeighborsStorage`] asked
/// for. Row `k` of the table is one pair, and all columns are indexed by the
/// same `k`.
///
/// # Pair direction
///
/// [`mode()`](Neighbors::mode()) records which point sets the indices refer to
/// and how large those sets are:
///
/// - **Self-query** — both indices refer to the same point set, and the table
///   is half-shell: every pair satisfies `i < j`, so each unordered pair
///   appears exactly once and never with `i == j`.
/// - **Cross-query** — `i` indexes the query points, `j` indexes the reference
///   points. Directed, with no ordering constraint between the two.
///
/// # Optional columns: `None` is not "empty"
///
/// [`dist_sq()`](Neighbors::dist_sq()) and [`disp()`](Neighbors::disp()) return
/// `Option`s. `None` means *this table never stored that quantity*, and it is
/// deliberately distinct from a stored-but-empty column: a table with a hundred
/// pairs and no `disp` column returns `None`, never a hundred rows of zeros.
/// Zeros would be a physically meaningful displacement (two coincident
/// particles), so fabricating them would turn a missing column into a silent
/// wrong answer. If an analysis needs a column, rerun the search with that
/// column enabled — it cannot be reconstructed afterwards (see
/// [`repack`](Neighbors::repack)).
///
/// # Building one
///
/// [`from_pairs`](Neighbors::from_pairs) is the public constructor: it consumes
/// a stream of [`NeighborPair`] values and keeps the columns the policy asks
/// for. Search algorithms fill their own cached table in place and expose it
/// through [`NbListAlgo::query`].
#[derive(Debug, Clone)]
pub struct Neighbors {
    /// Which optional columns `push` writes; the rest stay empty.
    pub(crate) storage: NeighborsStorage,
    /// Query point index per pair (i).
    pub(crate) idx_i: Vec<u32>,
    /// Reference point index per pair (j).
    pub(crate) idx_j: Vec<u32>,
    /// Squared distances, one per pair (empty when `!storage.dist_sq`).
    pub(crate) dist_sq: Vec<F>,
    /// Flat MIC displacements `[dx0,dy0,dz0, …]` (empty when `!storage.disp`).
    pub(crate) disp_flat: Vec<F>,
    /// Query identity: self vs cross, plus the point-set sizes.
    pub(crate) mode: QueryMode,
}

impl Neighbors {
    /// Materialize a pair stream into a table, keeping the columns `storage`
    /// asks for.
    ///
    /// This is the only public way to build a table from pairs. Each
    /// [`NeighborPair`] arrives with both physical quantities; `storage` decides
    /// which of them survive into the table. A column the policy drops is simply
    /// not written — it is never zero-filled, and the corresponding accessor
    /// will report `None`. Rows keep the order in which the iterator yielded
    /// them, so a search that emits pairs in a deterministic order produces a
    /// deterministic table.
    ///
    /// `mode` states what the indices mean and how many points each set holds;
    /// it is not inferred from the pairs, because an empty stream carries no
    /// evidence either way.
    ///
    /// # Panics
    /// In debug builds, panics if `mode` is [`QueryMode::SelfQuery`] and a pair
    /// violates the half-shell contract `i < j`. Mislabelling a full-shell
    /// stream as a self-query would silently double every count downstream (a
    /// radial distribution function multiplies self-query pair counts by two to
    /// recover both orderings), so the contract is checked rather than trusted.
    /// The check is a `debug_assert!` and is compiled out of release builds; it
    /// is a guard against a programming error, not against untrusted input.
    /// [`QueryMode::CrossQuery`] imposes no ordering constraint and is never
    /// checked.
    pub fn from_pairs(
        pairs: impl IntoIterator<Item = NeighborPair>,
        storage: NeighborsStorage,
        mode: QueryMode,
    ) -> Self {
        let mut out = Self::empty(mode, storage);
        let half_shell = matches!(mode, QueryMode::SelfQuery { .. });
        for p in pairs {
            debug_assert!(
                !half_shell || p.i < p.j,
                "self-query pairs are half-shell: expected i < j, got i = {}, j = {}",
                p.i,
                p.j
            );
            out.push(p.i, p.j, p.dist_sq, p.disp);
        }
        out
    }

    /// An empty table with the given query identity and column policy.
    pub(crate) fn empty(mode: QueryMode, storage: NeighborsStorage) -> Self {
        Self {
            storage,
            idx_i: Vec::new(),
            idx_j: Vec::new(),
            dist_sq: Vec::new(),
            disp_flat: Vec::new(),
            mode,
        }
    }

    /// Clear all buffers for reuse (keeps storage policy and mode).
    pub(crate) fn clear(&mut self) {
        self.idx_i.clear();
        self.idx_j.clear();
        self.dist_sq.clear();
        self.disp_flat.clear();
    }

    /// Storage policy for optional columns — equivalently, which of
    /// [`dist_sq()`](Self::dist_sq()) and [`disp()`](Self::disp()) will return
    /// `Some` for this table.
    #[inline]
    pub fn storage(&self) -> NeighborsStorage {
        self.storage
    }

    /// Push a neighbor pair into the table.
    ///
    /// `d2` is the squared minimum-image distance (Å²) and `dr` the
    /// minimum-image displacement `r_j - r_i` (Å); each is retained only when
    /// the corresponding [`NeighborsStorage`] flag is set, and silently dropped
    /// otherwise. Callers are responsible for the mode's ordering contract —
    /// this is the crate-internal fast path that engines use to write straight
    /// into the table, so unlike `from_pairs` it does not check `i < j`.
    #[inline(always)]
    pub(crate) fn push(&mut self, i: u32, j: u32, d2: F, dr: [F; 3]) {
        self.idx_i.push(i);
        self.idx_j.push(j);
        if self.storage.dist_sq {
            self.dist_sq.push(d2);
        }
        if self.storage.disp {
            self.disp_flat.extend(dr);
        }
    }

    /// Iterate the materialized pairs without allocating.
    ///
    /// `callback` is invoked once per pair, in table order, with four
    /// arguments:
    ///
    /// 1. `i: u32` — the query point index of the pair.
    /// 2. `j: u32` — the reference point index of the pair.
    /// 3. `Option<F>` — the squared minimum-image distance in Å².
    /// 4. `Option<[F; 3]>` — the minimum-image displacement `r_j - r_i` in Å,
    ///    unnormalized.
    ///
    /// A column this table did not store arrives as `None` — never as a
    /// fabricated `0.0` or `[0, 0, 0]`, both of which are legal physical values
    /// (two coincident particles) and would therefore be indistinguishable from
    /// real data. The `Option`s are decided once by the table's storage policy,
    /// not per pair: either every call sees `Some` for a given argument or every
    /// call sees `None`. Only a [`NeighborPair`] stream straight out of a search
    /// is guaranteed to carry every physical quantity.
    ///
    /// A consumer that requires a column should reject `None` loudly rather than
    /// substitute a value — see [`filter_rad`] and [`filter_sann`] for the
    /// in-tree precedent.
    pub fn for_each_pair<C>(&self, mut callback: C)
    where
        C: FnMut(u32, u32, Option<F>, Option<[F; 3]>),
    {
        for k in 0..self.n_pairs() {
            let d2 = self.storage.dist_sq.then(|| self.dist_sq[k]);
            let dr = self.storage.disp.then(|| {
                let b = k * 3;
                [
                    self.disp_flat[b],
                    self.disp_flat[b + 1],
                    self.disp_flat[b + 2],
                ]
            });
            callback(self.idx_i[k], self.idx_j[k], d2, dr);
        }
    }

    /// Re-materialize this table under a leaner storage policy.
    ///
    /// **Downgrade only.** Columns kept by both the old and the new policy are
    /// copied verbatim; columns the new policy drops are discarded. Indices and
    /// [`mode()`](Self::mode()) are preserved, so the pair set and its ordering
    /// contract are unchanged; asking for the policy the table already has is a
    /// plain clone. The source table is left untouched.
    ///
    /// An upgrade is impossible by construction, not merely unimplemented: a
    /// column that was never stored cannot be recovered from the indices, since
    /// the table does not keep the coordinates or the box the distances came
    /// from. Rerun the search with that column enabled instead.
    ///
    /// # Panics
    /// Panics when `storage` asks for a column this table never stored, with a
    /// message naming the column. It never invents values, and in particular
    /// never fills the new column with zeros — a zero displacement or distance
    /// is physically meaningful (coincident particles) and would be
    /// indistinguishable from real data downstream.
    pub fn repack(&self, storage: NeighborsStorage) -> Self {
        if storage.dist_sq && !self.storage.dist_sq {
            panic!(
                "repack cannot add a dist_sq column: physical quantities cannot be \
                 fabricated from an indices-only / lean list — rerun the neighbor \
                 search with NeighborsStorage::dist_sq set"
            );
        }
        if storage.disp && !self.storage.disp {
            panic!(
                "repack cannot add a disp column: physical quantities cannot be \
                 fabricated from an indices-only / lean list — rerun the neighbor \
                 search with NeighborsStorage::disp set"
            );
        }
        Self {
            storage,
            idx_i: self.idx_i.clone(),
            idx_j: self.idx_j.clone(),
            dist_sq: if storage.dist_sq {
                self.dist_sq.clone()
            } else {
                Vec::new()
            },
            disp_flat: if storage.disp {
                self.disp_flat.clone()
            } else {
                Vec::new()
            },
            mode: self.mode,
        }
    }

    // -----------------------------------------------------------------------
    // Public accessors (freud-style names)
    // -----------------------------------------------------------------------

    /// Number of neighbor pairs — the number of rows every column shares.
    #[inline]
    pub fn n_pairs(&self) -> usize {
        self.idx_i.len()
    }

    /// Query identity: self vs cross, plus the point-set sizes.
    #[inline]
    pub fn mode(&self) -> QueryMode {
        self.mode
    }

    /// Number of reference points (the point set used to build the spatial index).
    #[inline]
    pub fn num_points(&self) -> usize {
        match self.mode {
            QueryMode::SelfQuery { num_points } => num_points,
            QueryMode::CrossQuery { num_points, .. } => num_points,
        }
    }

    /// Number of query points.
    /// - Self-query: same as `num_points`.
    /// - Cross-query: the number of points passed to `query()`.
    #[inline]
    pub fn num_query_points(&self) -> usize {
        match self.mode {
            QueryMode::SelfQuery { num_points } => num_points,
            QueryMode::CrossQuery {
                num_query_points, ..
            } => num_query_points,
        }
    }

    /// Query point indices (i), one per pair.
    ///
    /// - Self-query: indices into the single point set.
    /// - Cross-query: indices into `query_points`.
    #[inline]
    pub fn query_point_indices(&self) -> &[u32] {
        &self.idx_i
    }

    /// Reference point indices (j), one per pair.
    ///
    /// - Self-query: indices into the single point set (always > i, half-shell).
    /// - Cross-query: indices into the reference `points`.
    #[inline]
    pub fn point_indices(&self) -> &[u32] {
        &self.idx_j
    }

    /// Squared minimum-image distances (Å²), one per pair.
    ///
    /// `None` when [`NeighborsStorage::dist_sq`] was not set for this table —
    /// which is different from an empty slice, and must not be treated as
    /// "distances are zero". When `Some`, the slice has exactly
    /// [`n_pairs()`](Self::n_pairs) elements, aligned row-for-row with the index
    /// columns.
    ///
    /// The square root is deliberately left to the caller: most consumers
    /// compare against a squared cutoff and never need it, and an accessor that
    /// hid a per-pair `sqrt` would make that cost invisible.
    #[inline]
    pub fn dist_sq(&self) -> Option<&[F]> {
        self.storage.dist_sq.then_some(self.dist_sq.as_slice())
    }

    /// Minimum-image displacements `r_j - r_i` (Å) as an `n_pairs × 3` view.
    ///
    /// Row `k` holds `(dx, dy, dz)` for pair `k`, pointing from `i` to `j`, and
    /// is **not** normalized — its length is the pair distance, so a caller who
    /// wants a direction divides the row by that length. When `Some`, the view
    /// has exactly [`n_pairs()`](Self::n_pairs) rows.
    ///
    /// `None` when [`NeighborsStorage::disp`] was not set for this table. That
    /// is not the same as a zero vector, which would mean two coincident
    /// particles.
    #[inline]
    pub fn disp(&self) -> Option<FNx3View<'_>> {
        if !self.storage.disp {
            return None;
        }
        Some(
            ArrayView2::from_shape((self.n_pairs(), 3), &self.disp_flat)
                .expect("disp_flat shape mismatch"),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests — spec neighborlist-01-types: Neighbors / from_pairs / optional columns
// ---------------------------------------------------------------------------

#[cfg(test)]
mod from_pairs_tests {
    use super::*;
    use ndarray::array;

    /// Two hard-coded pairs shared by the column-flag tests.
    ///
    /// | k | i | j | `dist_sq` | `disp`        | ‖disp‖² |
    /// |---|---|---|-----------|---------------|---------|
    /// | 0 | 0 | 1 | 2.0       | (1.0,1.0,0.0) | 2.0     |
    /// | 1 | 1 | 3 | 9.0       | (0.0,0.0,3.0) | 9.0     |
    ///
    /// Both satisfy the half-shell contract `i < j`, so they are legal under
    /// `QueryMode::SelfQuery { num_points: 4 }`.
    fn two_pairs() -> [NeighborPair; 2] {
        [
            NeighborPair {
                i: 0,
                j: 1,
                dist_sq: 2.0,
                disp: [1.0, 1.0, 0.0],
            },
            NeighborPair {
                i: 1,
                j: 3,
                dist_sq: 9.0,
                disp: [0.0, 0.0, 3.0],
            },
        ]
    }

    /// AC-003: `FULL` materializes indices + both physical columns verbatim.
    #[test]
    fn from_pairs_full_roundtrip() {
        let nb = Neighbors::from_pairs(
            two_pairs(),
            NeighborsStorage::FULL,
            QueryMode::SelfQuery { num_points: 4 },
        );

        assert_eq!(nb.n_pairs(), 2);
        assert!(matches!(nb.mode(), QueryMode::SelfQuery { num_points: 4 }));
        assert_eq!(nb.query_point_indices(), &[0u32, 1u32][..]);
        assert_eq!(nb.point_indices(), &[1u32, 3u32][..]);

        let d2 = nb.dist_sq().expect("FULL storage must materialize dist_sq");
        assert_eq!(d2, &[2.0_f64, 9.0_f64][..]);

        let disp = nb.disp().expect("FULL storage must materialize disp");
        assert_eq!(disp.nrows(), 2);
        assert_eq!(disp.ncols(), 3);

        let expected: [[F; 3]; 2] = [[1.0, 1.0, 0.0], [0.0, 0.0, 3.0]];
        for k in 0..2 {
            for c in 0..3 {
                assert!(
                    (disp[[k, c]] - expected[k][c]).abs() <= 1e-15,
                    "disp[{k}][{c}] = {} != {}",
                    disp[[k, c]],
                    expected[k][c]
                );
            }
        }
    }

    /// AC-004: no column requested ⇒ `None`, never a fabricated zero column.
    #[test]
    fn from_pairs_indices_only_gives_none_columns() {
        let nb = Neighbors::from_pairs(
            two_pairs(),
            NeighborsStorage::INDICES_ONLY,
            QueryMode::SelfQuery { num_points: 4 },
        );

        assert_eq!(nb.n_pairs(), 2);
        assert!(
            nb.dist_sq().is_none(),
            "INDICES_ONLY must not fabricate a zero dist_sq column"
        );
        assert!(
            nb.disp().is_none(),
            "INDICES_ONLY must not fabricate a zero disp column"
        );
    }

    /// AC-005: `DIST_SQ` keeps d² only.
    #[test]
    fn from_pairs_dist_sq_only() {
        let nb = Neighbors::from_pairs(
            two_pairs(),
            NeighborsStorage::DIST_SQ,
            QueryMode::SelfQuery { num_points: 4 },
        );

        let d2 = nb.dist_sq().expect("DIST_SQ must materialize dist_sq");
        assert_eq!(d2.len(), nb.n_pairs());
        assert_eq!(d2, &[2.0_f64, 9.0_f64][..]);
        assert!(
            nb.disp().is_none(),
            "DIST_SQ must leave the disp column absent"
        );
    }

    /// AC-005: `DISP` keeps the MIC displacement column only.
    #[test]
    fn from_pairs_disp_only() {
        let nb = Neighbors::from_pairs(
            two_pairs(),
            NeighborsStorage::DISP,
            QueryMode::SelfQuery { num_points: 4 },
        );

        assert!(
            nb.dist_sq().is_none(),
            "DISP must leave the dist_sq column absent"
        );
        let disp = nb.disp().expect("DISP must materialize disp");
        assert_eq!(disp.nrows(), nb.n_pairs());
        assert_eq!(disp.ncols(), 3);
        assert!((disp[[0, 0]] - 1.0).abs() <= 1e-15);
        assert!((disp[[1, 2]] - 3.0).abs() <= 1e-15);
    }

    /// AC-006 + AC-007: a materialized self-query is half-shell (`i < j`) and
    /// its two physical columns agree, `d² == ‖disp‖²`, under PBC.
    ///
    /// Hard-coded orthorhombic box 10 × 8 × 6 Å, fully periodic, cutoff 1.5 Å:
    ///
    /// | pair | separation                    | d²   |
    /// |------|-------------------------------|------|
    /// | 0-1  | x-image, 0.9 Å                | 0.81 |
    /// | 0-2  | y-image, 0.8 Å                | 0.64 |
    /// | 1-2  | x- and y-image, (0.9, 0.8, 0) | 1.45 |
    ///
    /// Atom 3 sits at the box centre, beyond 1.5 Å of every other atom.
    #[test]
    fn bruteforce_self_half_shell_and_consistency() {
        let bx = SimBox::ortho(
            array![10.0, 8.0, 6.0],
            array![0.0, 0.0, 0.0],
            [true, true, true],
        )
        .expect("valid orthorhombic box");
        let pts = array![
            [0.5, 0.5, 0.5],
            [9.6, 0.5, 0.5],
            [0.5, 7.7, 0.5],
            [5.0, 4.0, 3.0],
        ];

        let mut bf = BruteForce::new(1.5);
        bf.build(pts.view(), &bx);
        let nb: &Neighbors = bf.query();

        assert_eq!(nb.n_pairs(), 3);
        assert!(matches!(nb.mode(), QueryMode::SelfQuery { .. }));

        let idx_i = nb.query_point_indices();
        let idx_j = nb.point_indices();
        let d2 = nb.dist_sq().expect("BruteForce materializes dist_sq");
        let disp = nb.disp().expect("BruteForce materializes disp");

        for k in 0..nb.n_pairs() {
            assert!(
                idx_i[k] < idx_j[k],
                "self-query pair {k} violates half-shell: {} !< {}",
                idx_i[k],
                idx_j[k]
            );
            let norm_sq = disp[[k, 0]] * disp[[k, 0]]
                + disp[[k, 1]] * disp[[k, 1]]
                + disp[[k, 2]] * disp[[k, 2]];
            assert!(
                (d2[k] - norm_sq).abs() <= 1e-12,
                "pair {k}: dist_sq {} != ||disp||^2 {}",
                d2[k],
                norm_sq
            );
        }
    }

    /// AC-009: dropping a column copies what survives, exactly.
    #[test]
    fn repack_downgrade_drops_columns() {
        let full = Neighbors::from_pairs(
            two_pairs(),
            NeighborsStorage::FULL,
            QueryMode::SelfQuery { num_points: 4 },
        );
        let lean = full.repack(NeighborsStorage::DIST_SQ);

        assert_eq!(lean.n_pairs(), 2);
        assert_eq!(lean.query_point_indices(), &[0u32, 1u32][..]);
        assert_eq!(lean.point_indices(), &[1u32, 3u32][..]);
        assert_eq!(
            lean.dist_sq().expect("downgrade to DIST_SQ keeps dist_sq"),
            &[2.0_f64, 9.0_f64][..]
        );
        assert!(
            lean.disp().is_none(),
            "downgrade must drop the disp column, not zero it"
        );
        // Source table is untouched by the downgrade.
        assert!(full.disp().is_some());
    }

    /// AC-009: an upgrade cannot invent physics — it fails loudly.
    #[test]
    #[should_panic]
    fn repack_upgrade_panics() {
        let lean = Neighbors::from_pairs(
            two_pairs(),
            NeighborsStorage::INDICES_ONLY,
            QueryMode::SelfQuery { num_points: 4 },
        );
        let _upgraded = lean.repack(NeighborsStorage::FULL);
    }

    /// Spec §from_pairs trust boundary: `SelfQuery` with `i > j` is a
    /// programming error and must trip the debug assertion.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn self_query_from_pairs_debug_asserts_half_shell() {
        let bad = [NeighborPair {
            i: 3,
            j: 1,
            dist_sq: 4.0,
            disp: [2.0, 0.0, 0.0],
        }];
        let _nb = Neighbors::from_pairs(
            bad,
            NeighborsStorage::FULL,
            QueryMode::SelfQuery { num_points: 4 },
        );
    }
}
