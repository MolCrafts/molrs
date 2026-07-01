//! Analysis compute modules for molrs molecular simulation.
//!
//! Trajectory analysis (RDF, MSD, clustering, gyration/inertia, PCA,
//! k-means) built around a single unified [`Compute`] trait.
//! Every analysis is stateless — orchestrate from the caller.
//!
//! # Stateless `Compute` — orchestrate from the caller
//!
//! Each [`Compute`] impl is a pure function: `&self` is an immutable
//! parameter bag. Two `compute` calls with identical `frames` + `args`
//! always produce identical output. There is no hidden mutable state,
//! no DAG, no store — just the trait and per-analysis modules.
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
//! # Results
//!
//! Every Compute output implements [`ComputeResult`]. Accumulating outputs
//! (RDF) override [`finalize`](ComputeResult::finalize) to normalize; other
//! outputs use the default no-op.
//!
//! # Available analyses
//!
//! | Module | Args | Output |
//! |--------|------|--------|
//! | [`rdf`] | `&Vec<NeighborList>` | [`RDFResult`] |
//! | [`msd`] | `()` | [`MSDTimeSeries`] |
//! | [`cluster`] | `&Vec<NeighborList>` | `Vec<`[`ClusterResult`]`>` |
//! | [`cluster_centers`] | `&Vec<ClusterResult>` | `Vec<`[`ClusterCentersResult`]`>` |
//! | [`center_of_mass`] | `&Vec<ClusterResult>` | `Vec<`[`COMResult`]`>` |
//! | [`gyration_tensor`] | `(&Vec<ClusterResult>, &Vec<ClusterCentersResult>)` | `Vec<`[`GyrationTensorResult`]`>` |
//! | [`inertia_tensor`] | `(&Vec<ClusterResult>, &Vec<COMResult>)` | `Vec<`[`InertiaTensorResult`]`>` |
//! | [`radius_of_gyration`] | `(&Vec<ClusterResult>, &Vec<COMResult>)` | `Vec<`[`RgResult`]`>` |
//! | [`pca`] | `&Vec<T: DescriptorRow>` | [`PcaResult`] |
//! | [`kmeans`] | `&PcaResult` | [`KMeansResult`] |
//! | [`distribution`] | `&AtomGroups` | [`DistributionResult`](distribution::DistributionResult) |
//! | [`distribution::CombinedDistribution`] | `&[AtomGroups]` | [`CombinedDistributionResult`](distribution::CombinedDistributionResult) |

pub mod center_of_mass;
pub mod cluster;
pub mod cluster_centers;
pub mod density;
pub mod dielectric;
pub mod diffraction;
pub mod distribution;
pub mod environment;
pub mod error;
pub mod fit;
pub mod gyration_tensor;
pub mod hbond;
pub mod inertia_tensor;
pub mod jacf;
pub mod kmeans;
pub mod msd;
pub mod onsager;
pub mod order;
pub mod pca;
pub mod persist;
pub mod pmft;
pub mod radius_of_gyration;
pub mod rdf;
pub mod result;
pub mod spectra;
pub mod traits;
pub mod util;
pub mod validate;
pub mod van_hove;
#[cfg(feature = "voronoi")]
pub mod voronoi;

// Re-exports
pub use center_of_mass::{COMResult, CenterOfMass};
pub use cluster::{Cluster, ClusterProperties, ClusterPropertiesResult, ClusterResult};
pub use cluster_centers::{ClusterCenters, ClusterCentersResult};
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
pub use environment::{
    AngularSeparationGlobal, AngularSeparationGlobalResult, AngularSeparationNeighbor,
    AngularSeparationNeighborResult, BondOrder, BondOrderResult, LocalBondProjection,
    LocalBondProjectionResult, LocalDescriptors, LocalDescriptorsResult, MatchEnv, MatchEnvResult,
};
pub use error::ComputeError;
pub use fit::{
    DebyeFit, DebyeFitResult, DebyeRelaxation, DebyeRelaxationResult, DielectricSpectrumResult,
    EinsteinConductivity, EinsteinConductivityResult, EinsteinDiffusion, EinsteinDiffusionArgs,
    EinsteinHelfandSpectrum, EwaldBoundary, GreenKuboConductivity, GreenKuboConductivityResult,
    GreenKuboDiffusion, GreenKuboSpectrum, IRFlux, IRFluxResult, IRSpectrum, LinearFit,
    LinearFitResult, Plateau, PlateauResult, PowerSpectrum, RamanSpectrum, RamanTensor,
    RamanTensorResult, ResonanceRamanSpectrum, ResonanceRamanTensor, RoaCrossResult,
    RoaCrossTensor, RoaSpectrum, RunningIntegral, RunningIntegralResult, VACF, VacfResult,
    VcdCrossFlux, VcdCrossResult, VcdSpectrum,
};
pub use gyration_tensor::{GyrationTensor, GyrationTensorResult};
pub use hbond::{
    DistKind, HBond, HBondCriterion, HBonds, HBondsResult, LifetimeResult, NetworkResult,
    hbond_components, hbond_lifetimes, presence_from_hbonds,
};
pub use inertia_tensor::{InertiaTensor, InertiaTensorResult};
// `jacf` is now a documentation-only module: the Green–Kubo conductivity is the
// `GreenKuboConductivity` (raw ACF) + `fit::RunningIntegral` composition. Its
// former `JacfResult` / `green_kubo_conductivity` exports were removed.
pub use kmeans::{KMeans, KMeansResult};
pub use msd::{MSD, MSDResult, MSDTimeSeries, MsdMode};
pub use onsager::{OnsagerCorrelation, OnsagerResult};
pub use order::{
    ContinuousCoordination, ContinuousCoordinationResult, Cubatic, CubaticResult, Hexatic,
    HexaticResult, LegendreReorientation, LegendreReorientationResult, Nematic, NematicResult,
    RotationalAutocorrelation, RotationalAutocorrelationResult, SolidLiquid, SolidLiquidResult,
    Steinhardt, SteinhardtResult,
};
pub use pca::{Pca2, PcaResult};
pub use persist::{PersistResult, SurvivalMethod, pair_survival_tcf};
pub use pmft::{
    PMFTR12, PMFTR12Args, PMFTR12Result, PMFTXY, PMFTXYArgs, PMFTXYResult, PMFTXYT, PMFTXYTArgs,
    PMFTXYTResult, PMFTXYZ, PMFTXYZArgs, PMFTXYZResult,
};
pub use radius_of_gyration::{RadiusOfGyration, RgResult};
pub use rdf::{RDF, RDFResult};
pub use result::{ComputeResult, DescriptorRow};
pub use spectra::{RamanSpectrumResult, SpectrumResult};
pub use traits::{Compute, Fit};
pub use van_hove::{VanHove, VanHoveResult};
#[cfg(feature = "voronoi")]
pub use voronoi::{
    DensityGrid, DomainAnalysis, DomainResult, Face, MolecularMoments, RadicalVoronoi,
    VoidAnalysis, VoidResult, VoronoiCells, VoronoiIntegration, polarizability_finite_field,
};
