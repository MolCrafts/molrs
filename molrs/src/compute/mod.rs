//! Analysis compute modules for molrs molecular simulation.
//!
//! Trajectory analysis (RDF, MSD, transport, spectroscopy, clustering,
//! shape descriptors, PCA/k-means) built around a single unified [`Compute`]
//! trait. Every analysis is stateless — orchestrate from the caller.
//!
//! # Stateless `Compute` — orchestrate from the caller
//!
//! Each [`Compute`] impl is a pure function: `&self` is an immutable
//! parameter bag. Two `compute` calls with identical `frames` + `args`
//! always produce identical output. There is no hidden mutable state,
//! no DAG, no store — just the trait and per-category modules.
//!
//! For DAG orchestration (topological order, diamond reuse, external
//! input validation), use `molpy.compute.Workflow` on the Python side.
//! It composes `Compute` nodes via Python's stdlib `graphlib` and
//! calls each Rust kernel directly.
//!
//! # Unified trait
//!
//! Every analysis implements [`Compute`]. A single frame, a trajectory, and a
//! whole dataset are the same kind of input — a slice `&[&F]` — and each
//! impl decides how to interpret it:
//!
//! - **Whole-sequence accumulators** (RDF) iterate every frame.
//! - **Time-series analyses** (MSD) use `frames[0]` as a reference.
//! - **Per-frame analyses** (Cluster, COM) return one result per frame.
//! - **Matrix consumers** (PCA, k-means) take upstream per-frame outputs as
//!   rows (via [`DescriptorRow`]).
//!
//! # Neighbor tables: what `Args` carries, and which columns a kernel needs
//!
//! Most analyses here are *pair* analyses: they answer a question about every
//! pair of particles lying closer together than a fixed **cutoff** distance.
//! Finding those pairs is not their job. The search lives in
//! [`molrs::spatial::neighbors`], and what it
//! produces — a [`Neighbors`](molrs::spatial::neighbors::Neighbors) table, a
//! column store holding one row per pair — is handed to the kernel as its
//! `Args`:
//! `&Neighbors` for a single frame, `&Vec<Neighbors>` for a trajectory, in
//! which case there is **one table per frame, index-aligned with `frames`**. A
//! length mismatch is [`ComputeError::DimensionMismatch`], never a silent
//! truncation.
//!
//! ## Columns are opt-in, and a missing one is an error rather than a zero
//!
//! Every table stores each pair's two particle indices `(i, j)`. Its two
//! *physical* columns are stored only if the caller asked for them when the
//! table was materialized, by naming a
//! [`NeighborsStorage`](molrs::spatial::neighbors::NeighborsStorage) policy:
//!
//! - `dist_sq` — the squared pair distance `|r_j − r_i|²` in Å², taken under
//!   the **minimum-image convention**: with periodic boundaries the box is
//!   tiled through space, so a pair has one separation per periodic copy
//!   ("image") of the box, and the search reports only the shortest of them.
//! - `disp` — the displacement `r_j − r_i` itself in Å, from that same
//!   minimum image, pointing from `i` to `j` and **not** normalized. Its length
//!   is the pair distance, so a caller who wants a direction divides by that
//!   length.
//!
//! A column the table never stored reads back as `None`, never as a fabricated
//! zero: a zero displacement is a physically meaningful value (two coincident
//! particles) and would be indistinguishable from real data. So a kernel that
//! needs a column rejects `None` instead of substituting anything, and every
//! kernel rejects it the same way — [`ComputeError::BadShape`], whose
//! `expected` text names the missing column and the number of pairs it was
//! needed for. Nothing here indexes an absent column, which would be an
//! out-of-bounds panic on a native target and an `unreachable` trap under
//! WebAssembly.
//!
//! Which column a kernel needs follows from what it measures:
//!
//! | Needs | Kernels | Materialize the table with |
//! |---|---|---|
//! | `disp` — bond *directions* | [`Steinhardt`], [`Hexatic`], [`SolidLiquid`], [`ContinuousCoordination`] (the last three via [`compute_qlm`](order::compute_qlm)), every [`pmft`] kernel, [`BondOrder`], [`LocalDescriptors`], [`LocalBondProjection`], [`MatchEnv`] | `NeighborsStorage::DISP` or `FULL` |
//! | `dist_sq` — distances only | [`RDF`] when fed a materialized table, [`CorrelationFunction`], [`LocalDensity`] | `NeighborsStorage::DIST_SQ` or `FULL` |
//! | indices only — connectivity | [`Cluster`], [`AngularSeparationNeighbor`] | any policy, `INDICES_ONLY` included |
//!
//! `FULL` means *every column is present*. It never means a bidirectional pair
//! list — pair direction is an independent property, recorded by the table's
//! [`QueryMode`](molrs::spatial::neighbors::QueryMode). A **self-query**
//! searches one point set against itself and is half-shell: each unordered pair
//! appears exactly once, with `i < j`. A **cross-query** searches query points
//! against a separate reference set and is directed, with both orderings
//! present. Radial-distribution normalization is the one place that distinction
//! changes a number — a half-shell count is short by a factor of two — see
//! [`RdfMode`].
//!
//! Some entry points take no table at all, because they run a search of their
//! own per call: [`RDF::compute_self`], [`RDF::compute_frame`] and
//! [`RDF::compute_cross`] stream pairs straight out of the cell list without
//! ever materializing them, and [`HBonds`] and [`VanHove`] build a cross-query
//! internally, one per frame and one per time origin respectively.
//!
//! # Results
//!
//! Every Compute output implements [`ComputeResult`]. Accumulating outputs
//! (RDF) override [`finalize`](ComputeResult::finalize) to normalize; other
//! outputs use the default no-op.
//!
//! # Implementation modules (`compute/<name>/`)
//!
//! One folder per kernel family. **UI/catalog categories** (see
//! `molrs-wasm` `molrsComputeCatalog`, catalog v3) follow freud's top-level
//! modules and are not always 1:1 with folder names:
//! - g(r) lives in [`rdf`] but is catalogued under **density**
//!   (`freud.density.RDF`)
//! - Voronoi kernels are catalogued under **locality**
//!   (`freud.locality.Voronoi`)
//! - van Hove / pair persistence live under **transport** (with VACF/diffusion)
//! - static dielectric is under **spectroscopy**
//! - cluster radius-of-gyration is under **cluster** (freud.cluster.ClusterProperties)
//!
//! | Folder | Methods |
//! |----------|---------|
//! | [`rdf`] | pair distribution g(r) (+ streaming [`RDFAccumulator`]) |
//! | [`msd`] | mean squared displacement (+ streaming [`MSDAccumulator`]) |
//! | [`transport`] | VACF (+ streaming [`VACFAccumulator`]), Einstein/Green–Kubo diffusion & conductivity, Debye relaxation, Onsager |
//! | [`spectroscopy`] | IR / Raman / VCD / ROA / resonance-Raman raw correlators + spectral transforms, dielectric spectra |
//! | [`fitting`] | generic curve fits: [`LinearFit`], [`CumulativeTrapezoid`], [`Plateau`], [`DebyeFit`] |
//! | [`dynamics`] | van Hove G(r, t), pair persistence |
//! | [`dielectric`] | static dielectric constant from dipole fluctuations |
//! | [`cluster`] | connected-component clustering + per-cluster properties |
//! | [`shape`] | center of mass, cluster centers, gyration/inertia tensors, Rg |
//! | [`ml`] | PCA projection, k-means |
//! | [`density`] | correlation function, Gaussian/local density, spatial distribution, voxelization |
//! | [`order`] | Steinhardt, hexatic, nematic, cubatic, solid-liquid, … |
//! | [`environment`] | bond order, local descriptors, environment matching, … |
//! | [`diffraction`] | S(k) (Debye & direct), diffraction pattern |
//! | [`pmft`] | potentials of mean force and torque (R12/XY/XYT/XYZ) |
//! | [`distribution`] | distance/angle/dihedral distribution functions |
//! | [`hbond`] | hydrogen-bond detection, lifetimes, network components |
//! | [`voronoi`] | radical Voronoi cells, domains, voids (feature `voronoi`) |

pub mod cluster;
pub mod density;
pub mod dielectric;
pub mod diffraction;
pub mod distribution;
pub mod dynamics;
pub mod environment;
pub mod error;
pub mod fitting;
pub mod hbond;
pub mod ml;
pub mod msd;
pub mod order;
pub mod pmft;
pub mod rdf;
pub(crate) mod require;
pub mod result;
pub mod shape;
pub mod spectroscopy;
#[cfg(test)]
pub(crate) mod test_support;
pub mod traits;
pub mod transport;
pub mod util;
#[cfg(feature = "voronoi")]
pub mod voronoi;

// Re-exports
pub use cluster::{Cluster, ClusterProperties, ClusterPropertiesResult, ClusterResult};
pub use density::{
    CorrelationFunction, CorrelationFunctionResult, GaussianDensity, GaussianDensityResult,
    GridSpec, LocalDensity, LocalDensityResult, SpatialDistribution, SpatialDistributionResult,
    SphereVoxelization, SphereVoxelizationResult,
};
pub use dielectric::{
    StaticDielectricResult, compute_current_density, compute_dipole_moment, decompose_current,
    static_dielectric_constant, static_dielectric_constant_components,
};
pub use diffraction::{
    DiffractionPattern, DiffractionPatternResult, StaticStructureFactorDebye,
    StaticStructureFactorDebyeResult, StaticStructureFactorDirect,
    StaticStructureFactorDirectResult,
};
pub use distribution::{AxisSpec, CombinedDistribution, CombinedDistributionResult};
pub use dynamics::{
    Acf, AcfArgs, AcfResult, PersistResult, SurvivalMethod, VanHove, VanHoveResult,
    autocorrelation, pair_survival_tcf,
};
pub use environment::{
    AngularSeparationGlobal, AngularSeparationGlobalResult, AngularSeparationNeighbor,
    AngularSeparationNeighborResult, BondOrder, BondOrderResult, LocalBondProjection,
    LocalBondProjectionResult, LocalDescriptors, LocalDescriptorsResult, MatchEnv, MatchEnvResult,
};
pub use error::ComputeError;
pub use fitting::{
    CumulativeTrapezoid, CumulativeTrapezoidResult, LinearFit, LinearFitResult, Plateau,
    PlateauResult,
};
pub use hbond::{
    DistKind, HBond, HBondCriterion, HBonds, HBondsResult, LifetimeResult, NetworkResult,
    hbond_components, hbond_lifetimes, presence_from_hbonds,
};
pub use ml::{KMeans, KMeansResult, Pca2, PcaResult};
pub use msd::{MSD, MSDAccumulator, MSDResult, MSDTimeSeries, MsdMode};
pub use order::{
    ContinuousCoordination, ContinuousCoordinationResult, Cubatic, CubaticResult, Hexatic,
    HexaticResult, LegendreReorientation, LegendreReorientationResult, Nematic, NematicResult,
    RotationalAutocorrelation, RotationalAutocorrelationResult, SolidLiquid, SolidLiquidResult,
    Steinhardt, SteinhardtResult,
};
pub use pmft::{
    PMFTR12, PMFTR12Args, PMFTR12Result, PMFTXY, PMFTXYArgs, PMFTXYResult, PMFTXYT, PMFTXYTArgs,
    PMFTXYTResult, PMFTXYZ, PMFTXYZArgs, PMFTXYZResult,
};
pub use rdf::{RDF, RDFAccumulator, RDFResult, RdfMode};
/// Crate-internal: the column guards every neighbor-consuming kernel calls
/// before it reads `disp` (Å) or `dist_sq` (Å²). Deliberately not public API —
/// a caller outside the crate holds the table itself and asks it directly with
/// [`Neighbors::disp()`](molrs::spatial::neighbors::Neighbors::disp) /
/// [`Neighbors::dist_sq()`](molrs::spatial::neighbors::Neighbors::dist_sq),
/// which answer `Option` rather than [`ComputeError`].
pub(crate) use require::{require_disp, require_dist_sq};
pub use result::{ComputeResult, DescriptorRow};
pub use shape::{
    COMResult, CenterOfMass, ClusterCenters, ClusterCentersResult, GyrationTensor,
    GyrationTensorResult, InertiaTensor, InertiaTensorResult, RadiusOfGyration, RgResult,
};
pub use spectroscopy::{
    ConductivitySumRule, DielectricSpectrumResult, EinsteinHelfandSpectrum, GreenKuboSpectrum,
    IRFlux, IRFluxResult, IRSpectrum, KramersKronig, KramersKronigCheck, PowerSpectrum,
    RamanSpectrum, RamanSpectrumResult, RamanTensor, RamanTensorResult, ResonanceRamanSpectrum,
    ResonanceRamanTensor, RoaCrossResult, RoaCrossTensor, RoaSpectrum, RouteAgreement,
    RouteAgreementCheck, SpectrumResult, SumRuleCheck, VcdCrossFlux, VcdCrossResult, VcdSpectrum,
};
pub use traits::{Check, Compute, Fit, Verdict};
pub use transport::{
    DebyeFit, DebyeFitResult, DebyeRelaxation, DebyeRelaxationResult, EinsteinConductivity,
    EinsteinConductivityResult, EinsteinDiffusion, EinsteinDiffusionArgs, EwaldBoundary,
    GreenKuboConductivity, GreenKuboConductivityResult, GreenKuboDiffusion, OnsagerCorrelation,
    OnsagerResult, VACF, VACFAccumulator, VacfResult,
};
#[cfg(feature = "voronoi")]
pub use voronoi::{
    DensityGrid, DomainAnalysis, DomainResult, Face, MolecularMoments, RadicalVoronoi,
    VoidAnalysis, VoidResult, VoronoiCells, VoronoiIntegration, polarizability_finite_field,
};

// ---------------------------------------------------------------------------
// Tests — spec neighborlist-03-compute, task 1: the column requirement helpers
// ---------------------------------------------------------------------------

/// `require_disp` / `require_dist_sq`: the one place a compute kernel turns
/// "this table never stored that column" into a [`ComputeError::BadShape`].
///
/// A [`Neighbors`](molrs::spatial::neighbors::Neighbors) table reports an
/// absent column as `None`, never as a fabricated zero — so every kernel that
/// needs one has to reject `None` itself. These helpers are that rejection,
/// written once: a kernel calls one of them and gets either the column or an
/// error naming what was missing and how many pairs went unanswered.
///
/// The tests pin the **re-exported path** `crate::compute::require_*`; which
/// file inside `compute/` defines them is the implementer's choice.
///
/// The fixture is the two-pair table from the `neighbors` unit tests, so the
/// same hard-coded numbers describe the same pairs on both sides of the crate:
///
/// | k | i | j | `dist_sq` | `disp`        |
/// |---|---|---|-----------|---------------|
/// | 0 | 0 | 1 | 2.0       | (1.0,1.0,0.0) |
/// | 1 | 1 | 3 | 9.0       | (0.0,0.0,3.0) |
#[cfg(test)]
mod require_tests {
    use crate::compute::error::ComputeError;
    use crate::compute::{require_disp, require_dist_sq};
    use molrs::spatial::neighbors::{NeighborPair, Neighbors, NeighborsStorage, QueryMode};
    use molrs::types::F;

    /// Two hard-coded half-shell pairs (`i < j`), legal under
    /// `SelfQuery { num_points: 4 }`.
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

    fn table(storage: NeighborsStorage) -> Neighbors {
        Neighbors::from_pairs(two_pairs(), storage, QueryMode::SelfQuery { num_points: 4 })
    }

    /// Basics: on a `FULL` table the displacement column comes back as an
    /// `n_pairs × 3` view holding exactly the pushed vectors (Å, exact — these
    /// are copies, not arithmetic).
    #[test]
    fn require_disp_returns_the_full_table_column() {
        let nb = table(NeighborsStorage::FULL);
        let disp = require_disp(&nb).expect("FULL table has a disp column");

        assert_eq!(disp.nrows(), 2);
        assert_eq!(disp.ncols(), 3);
        let expected: [[F; 3]; 2] = [[1.0, 1.0, 0.0], [0.0, 0.0, 3.0]];
        for k in 0..2 {
            for c in 0..3 {
                assert!(
                    (disp[[k, c]] - expected[k][c]).abs() <= 1e-12,
                    "disp[{k}][{c}] = {} != {}",
                    disp[[k, c]],
                    expected[k][c]
                );
            }
        }
    }

    /// Basics: on a `FULL` table the squared-distance column comes back as a
    /// slice of `n_pairs` values, in table order (Å², exact copies).
    #[test]
    fn require_dist_sq_returns_the_full_table_column() {
        let nb = table(NeighborsStorage::FULL);
        let d2 = require_dist_sq(&nb).expect("FULL table has a dist_sq column");

        assert_eq!(d2.len(), nb.n_pairs());
        assert_eq!(d2, &[2.0_f64, 9.0_f64][..]);
    }

    /// Edge: an indices-only table that *does* have pairs is the case that
    /// would otherwise index an empty column — `BadShape`, and the `expected`
    /// text names the missing column and how many pairs it was needed for.
    #[test]
    fn require_disp_on_indices_only_is_bad_shape() {
        let nb = table(NeighborsStorage::INDICES_ONLY);
        assert_eq!(
            nb.n_pairs(),
            2,
            "the guard must be tested on a non-empty table"
        );

        let err = require_disp(&nb).expect_err("indices-only table has no disp column");
        let ComputeError::BadShape { expected, got } = err else {
            panic!("expected BadShape, got {err:?}");
        };
        assert!(
            expected.contains("disp"),
            "BadShape.expected must name the missing column: {expected:?}"
        );
        assert!(
            expected.contains('2'),
            "BadShape.expected must name the pair count (2): {expected:?}"
        );
        assert!(!got.is_empty(), "BadShape.got must describe the table");
    }

    /// Edge: same for the radial column — a lean list with pairs but no `d²`.
    #[test]
    fn require_dist_sq_on_indices_only_is_bad_shape() {
        let nb = table(NeighborsStorage::INDICES_ONLY);
        assert_eq!(
            nb.n_pairs(),
            2,
            "the guard must be tested on a non-empty table"
        );

        let err = require_dist_sq(&nb).expect_err("indices-only table has no dist_sq column");
        let ComputeError::BadShape { expected, got } = err else {
            panic!("expected BadShape, got {err:?}");
        };
        assert!(
            expected.contains("dist_sq"),
            "BadShape.expected must name the missing column: {expected:?}"
        );
        assert!(
            expected.contains('2'),
            "BadShape.expected must name the pair count (2): {expected:?}"
        );
        assert!(!got.is_empty(), "BadShape.got must describe the table");
    }
}
