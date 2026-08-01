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
pub use dynamics::{PersistResult, SurvivalMethod, VanHove, VanHoveResult, pair_survival_tcf};
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
pub use rdf::{RDF, RDFAccumulator, RDFResult};
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
