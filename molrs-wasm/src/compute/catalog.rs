//! The compute catalog: everything a downstream UI needs to present,
//! configure and dispatch the analyses this module exports.

use super::js_value;
use molrs::core::keys;
use molrs::op::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

//
// Everything a caller needs to present, configure and dispatch an analysis
// lives here, so no consumer has to keep a parallel hand-written table that
// can drift out of sync with the bindings above.

/// Default value of a [`ParamSpec`], serialized untagged so JS sees a bare
/// `number`, `boolean` or `string`.
#[derive(Serialize, Clone, Copy)]
#[serde(untagged)]
enum ParamDefault {
    Num(F),
    Bool(bool),
    Text(&'static str),
}

/// One user-facing knob of an analysis.
///
/// These are **UI-level** parameters, not a literal mirror of the WASM
/// constructor: `diffraction.static_structure_factor` exposes `kMin`/`kMax`/`nK`
/// and the caller expands them into the `k_values` array the constructor wants.
/// `kind` tells the caller how to render and coerce the value:
///
/// | `kind` | JS value |
/// |--------|----------|
/// | `int`, `float` | `number` |
/// | `bool` | `boolean` |
/// | `select` | one of `options` |
/// | `intList`, `floatList` | comma-separated `string` → typed array |
#[derive(Serialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
struct ParamSpec {
    key: &'static str,
    label: &'static str,
    kind: &'static str,
    default: ParamDefault,
    /// `true` when the binding accepts `null` for this argument.
    optional: bool,
    /// `"ctor"` — a positional constructor argument, in declaration order,
    /// after any leading arguments the dispatch shape supplies itself. Every
    /// piece of configuration lives here: `compute` / `fit` take only data.
    /// `"call"` — the knob configures a *different* object the caller builds
    /// first (`NeighborList`'s cutoff, `Cluster`'s min size), never this one.
    slot: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    min: Option<F>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max: Option<F>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<&'static [&'static str]>,
}

const fn base(
    key: &'static str,
    label: &'static str,
    kind: &'static str,
    default: ParamDefault,
) -> ParamSpec {
    ParamSpec {
        key,
        label,
        kind,
        default,
        optional: false,
        slot: "ctor",
        min: None,
        max: None,
        unit: None,
        options: None,
    }
}

fn p_int(key: &'static str, label: &'static str, default: u32, min: F, max: F) -> ParamSpec {
    ParamSpec {
        min: Some(min),
        max: Some(max),
        ..base(key, label, "int", ParamDefault::Num(F::from(default)))
    }
}

fn p_float(
    key: &'static str,
    label: &'static str,
    default: F,
    unit: Option<&'static str>,
) -> ParamSpec {
    ParamSpec {
        unit,
        ..base(key, label, "float", ParamDefault::Num(default))
    }
}

fn p_bool(key: &'static str, label: &'static str, default: bool) -> ParamSpec {
    base(key, label, "bool", ParamDefault::Bool(default))
}

fn p_select(
    key: &'static str,
    label: &'static str,
    default: &'static str,
    options: &'static [&'static str],
) -> ParamSpec {
    ParamSpec {
        options: Some(options),
        ..base(key, label, "select", ParamDefault::Text(default))
    }
}

fn p_list(
    key: &'static str,
    label: &'static str,
    kind: &'static str,
    default: &'static str,
) -> ParamSpec {
    base(key, label, kind, ParamDefault::Text(default))
}

fn optional(spec: ParamSpec) -> ParamSpec {
    ParamSpec {
        optional: true,
        ..spec
    }
}

/// Mark a knob as configuring a helper object the caller builds, not this one.
fn call(spec: ParamSpec) -> ParamSpec {
    ParamSpec {
        slot: "call",
        ..spec
    }
}

/// The `cutoff` knob every neighbor-driven analysis needs to build its
/// `NeighborList` before `compute(frame, nlist)`.
fn p_cutoff(default: F) -> ParamSpec {
    call(p_float("cutoff", "Neighbor cutoff", default, Some("Å")))
}

/// A menu category, in the order a picker should present it.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CatalogCategory {
    id: &'static str,
    label: &'static str,
}

/// One analysis: what it is, how to call it, and what it needs.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ComputeCatalogEntry {
    id: &'static str,
    category: &'static str,
    label: &'static str,
    /// Class exported from this module. Always present — the catalog never
    /// names a binding that does not exist.
    wasm_export: &'static str,
    /// How to drive the binding. The caller dispatches on this, not on `id`.
    ///
    /// | `input_kind` | invocation |
    /// |--------------|-----------|
    /// | `frame` | `compute(frame)` |
    /// | `frameNeighbors` | `compute(frame, nlist)`, `nlist` from `cutoff` |
    /// | `frameClusters` | `compute(frame, clusterResult)` |
    /// | `frameGroups` | `compute(frame, atomIndexTuples)` |
    /// | `frameGroupSets` | `compute(frame, number[][])` |
    /// | `frameRadii` | `compute(frame, radii, …)` — Voronoi family |
    /// | `accumulate` | `feed(frame)` per frame, then `compute()` / `results()` |
    /// | `series` | `compute(…)` / `fit(…)` over raw arrays; no `Frame` |
    input_kind: &'static str,
    /// Shape of the payload, for picking a renderer.
    result_kind: &'static str,
    /// Per-atom or per-trajectory inputs needed **beyond positions**. A caller
    /// that cannot supply one of these should disable the entry and say which.
    requires: &'static [&'static str],
    params: Vec<ParamSpec>,
}

/// Menu categories — freud top-level analysis modules first, then molrs
/// extensions. Not 1:1 with every Rust `compute/` folder.
///
/// freud mappings:
/// - `density` — g(r) (`freud.density.Rdf`) + Local/Gaussian density, …
/// - `locality` — Voronoi (`freud.locality.Voronoi`); neighbor queries are infra
/// - `msd` / `cluster` / `order` / `environment` / `diffraction` / `pmft` — 1:1
///
/// molrs extensions (no freud top-level):
/// - `transport` — VACF/diffusion/conductivity **and** van Hove / pair persistence
/// - `spectroscopy` — IR/Raman/… **and** static dielectric constant
/// - `distribution` / `hbond` / `shape` / `fit` / `ml`
const CATEGORIES: [CatalogCategory; 15] = [
    // --- freud core ---------------------------------------------------------
    CatalogCategory {
        id: "density",
        label: "Density",
    },
    CatalogCategory {
        id: "locality",
        label: "Locality",
    },
    CatalogCategory {
        id: "msd",
        label: "Msd",
    },
    CatalogCategory {
        id: "cluster",
        label: "Cluster",
    },
    CatalogCategory {
        id: "order",
        label: "Order",
    },
    CatalogCategory {
        id: "environment",
        label: "Environment",
    },
    CatalogCategory {
        id: "diffraction",
        label: "Diffraction",
    },
    CatalogCategory {
        id: "pmft",
        label: "PMFT",
    },
    // --- molrs extensions ---------------------------------------------------
    CatalogCategory {
        id: "transport",
        label: "Transport",
    },
    CatalogCategory {
        id: "spectroscopy",
        label: "Spectroscopy",
    },
    CatalogCategory {
        id: "hbond",
        label: "Hydrogen Bonds",
    },
    CatalogCategory {
        id: "distribution",
        label: "Distribution",
    },
    CatalogCategory {
        id: "shape",
        label: "Shape",
    },
    CatalogCategory {
        id: "fit",
        label: "Fit",
    },
    CatalogCategory {
        id: "ml",
        label: "ML",
    },
];

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ComputeCatalog {
    /// Bump whenever an entry's `id`, `wasm_export`, `input_kind` or param keys change.
    version: u32,
    categories: &'static [CatalogCategory],
    analyses: Vec<ComputeCatalogEntry>,
}

// One catalog row per call; the positional form mirrors the table it fills.
#[allow(clippy::too_many_arguments)]
fn entry(
    id: &'static str,
    category: &'static str,
    label: &'static str,
    wasm_export: &'static str,
    input_kind: &'static str,
    result_kind: &'static str,
    requires: &'static [&'static str],
    params: Vec<ParamSpec>,
) -> ComputeCatalogEntry {
    ComputeCatalogEntry {
        id,
        category,
        label,
        wasm_export,
        input_kind,
        result_kind,
        requires,
        params,
    }
}

/// Describe every analysis this module exports.
///
/// Returns `{ version, categories, analyses }`. Consumers should group
/// `analyses` by `category` in `categories` order to build a menu.
#[wasm_bindgen(js_name = molrsComputeCatalog)]
pub fn molrs_compute_catalog() -> Result<JsValue, JsValue> {
    let analyses = vec![
        // --- density (freud.density: RDF + local/gaussian density, …) -------
        // The menu category is `density`, matching freud.density.RDF.
        entry(
            "density.radial_distribution",
            "density",
            "Radial distribution g(r)",
            "Rdf",
            "frameNeighbors",
            "lineSeries",
            &[],
            vec![
                p_cutoff(10.0),
                p_int("nBins", "Bins", 100, 1.0, 4096.0),
                p_float("rMax", "r max", 10.0, Some("Å")),
                optional(p_float("rMin", "r min", 0.0, Some("Å"))),
            ],
        ),
        // --- msd ------------------------------------------------------------
        entry(
            "msd.mean_squared_displacement",
            "msd",
            "Mean squared displacement",
            "Msd",
            "accumulate",
            "trajectorySeries",
            &[],
            vec![],
        ),
        // --- transport ------------------------------------------------------
        entry(
            "transport.vacf",
            "transport",
            "Vacf",
            "Vacf",
            "series",
            "lineSeries",
            &["velocity"],
            // `n_dof` is the column count of the velocity matrix (3 x atoms),
            // so the caller derives it from the data rather than asking.
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_int("resolution", "Max lag", 200, 1.0, 1e6),
            ],
        ),
        entry(
            "transport.einstein_diffusion",
            "transport",
            "Einstein diffusion",
            "EinsteinDiffusion",
            "accumulate",
            "lineSeries",
            &[],
            vec![p_float("dt", "Timestep", 1.0, Some("fs"))],
        ),
        entry(
            "transport.green_kubo_diffusion",
            "transport",
            "Green-Kubo diffusion",
            "GreenKuboDiffusion",
            "series",
            "lineSeries",
            &["velocity"],
            // `n_dof` is the column count of the velocity matrix (3 x atoms),
            // so the caller derives it from the data rather than asking.
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_int("resolution", "Max lag", 200, 1.0, 1e6),
            ],
        ),
        entry(
            "transport.conductivity",
            "transport",
            "Conductivity",
            "GreenKuboConductivity",
            "series",
            "lineSeries",
            &["charge", "velocity"],
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_int("maxLag", "Max lag", 200, 1.0, 1e6),
            ],
        ),
        entry(
            "transport.einstein_conductivity",
            "transport",
            "Einstein conductivity",
            "EinsteinConductivity",
            "series",
            "lineSeries",
            &["charge"],
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_int("maxLag", "Max lag", 200, 1.0, 1e6),
            ],
        ),
        entry(
            "transport.onsager_correlation",
            "transport",
            "Onsager correlation",
            "OnsagerCorrelation",
            "series",
            "lineSeries",
            &["charge", "velocity"],
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_int("maxLag", "Max lag", 200, 1.0, 1e6),
            ],
        ),
        // --- dynamics -------------------------------------------------------
        entry(
            "dynamics.van_hove_function",
            "transport",
            "Van Hove function",
            "VanHove",
            "accumulate",
            "matrix",
            &[],
            vec![
                p_int("nRBins", "r bins", 100, 1.0, 4096.0),
                p_float("rMax", "r max", 10.0, Some("Å")),
                p_list("lags", "Lags", "intList", "1,2,5,10"),
                optional(p_int("stride", "Stride", 1, 1.0, 1e6)),
            ],
        ),
        entry(
            "dynamics.pair_survival",
            "transport",
            "Pair survival",
            "PairSurvival",
            "series",
            "lineSeries",
            &["atomPairs"],
            vec![
                p_float("r0", "Birth radius r0", 3.0, Some("Å")),
                p_float("r1", "Break radius r1", 3.5, Some("Å")),
                p_select(
                    "method",
                    "Survival method",
                    "continuous",
                    &["continuous", "intermittent", "ssp"],
                ),
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_int("maxLag", "Max lag", 200, 1.0, 1e6),
                p_bool("excludeSelf", "Exclude self pairs", true),
            ],
        ),
        // --- spectroscopy ---------------------------------------------------
        entry(
            "spectroscopy.power_spectrum",
            "spectroscopy",
            "Power spectrum",
            "PowerSpectrum",
            "series",
            "lineSeries",
            &["velocity"],
            vec![
                p_float("dtFs", "Timestep", 1.0, Some("fs")),
                call(p_int("resolution", "Max lag", 200, 1.0, 1e6)),
            ],
        ),
        entry(
            "spectroscopy.ir_spectrum",
            "spectroscopy",
            "IR spectrum",
            "IrSpectrum",
            "series",
            "lineSeries",
            &["dipole"],
            vec![
                p_float("dtFs", "Timestep", 1.0, Some("fs")),
                call(p_int("resolution", "Max lag", 200, 1.0, 1e6)),
            ],
        ),
        entry(
            "spectroscopy.raman_spectrum",
            "spectroscopy",
            "Raman spectrum",
            "RamanSpectrum",
            "series",
            "lineSeries",
            &["polarizability"],
            vec![
                optional(p_float(
                    "incidentFrequencyCm1",
                    "Incident frequency",
                    0.0,
                    Some("cm^-1"),
                )),
                optional(p_float("temperatureK", "Temperature", 0.0, Some("K"))),
                optional(p_bool("averaged", "Orientation averaged", false)),
                p_float("dtFs", "Timestep", 1.0, Some("fs")),
                call(p_int("resolution", "Max lag", 200, 1.0, 1e6)),
            ],
        ),
        entry(
            "spectroscopy.vcd_spectrum",
            "spectroscopy",
            "VCD spectrum",
            "VcdSpectrum",
            "series",
            "lineSeries",
            &["dipole", "magneticDipole"],
            vec![
                p_float("dtFs", "Timestep", 1.0, Some("fs")),
                call(p_int("resolution", "Max lag", 200, 1.0, 1e6)),
            ],
        ),
        entry(
            "spectroscopy.roa_spectrum",
            "spectroscopy",
            "ROA spectrum",
            "RoaSpectrum",
            "series",
            "lineSeries",
            &["polarizability", "gTensor"],
            vec![
                optional(p_float(
                    "incidentFrequencyCm1",
                    "Incident frequency",
                    0.0,
                    Some("cm^-1"),
                )),
                optional(p_float("temperatureK", "Temperature", 0.0, Some("K"))),
                optional(p_bool("averaged", "Orientation averaged", false)),
                p_float("dtFs", "Timestep", 1.0, Some("fs")),
                call(p_int("resolution", "Max lag", 200, 1.0, 1e6)),
            ],
        ),
        entry(
            "spectroscopy.dielectric_spectrum",
            "spectroscopy",
            "Dielectric spectrum",
            "GreenKuboDielectricSpectrum",
            "series",
            "lineSeries",
            &["charge", "velocity"],
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_float("volume", "Box volume", 0.0, Some("Å³")),
                p_float("temperature", "Temperature", 300.0, Some("K")),
                optional(p_float("epsilonInf", "ε∞", 1.0, None)),
            ],
        ),
        // --- dielectric -----------------------------------------------------
        entry(
            "dielectric.static_dielectric_constant",
            "spectroscopy",
            "Static dielectric constant",
            "StaticDielectric",
            "series",
            "scalar",
            &["dipole"],
            vec![
                p_float("volume", "Box volume", 0.0, Some("Å³")),
                p_float("temperature", "Temperature", 300.0, Some("K")),
                optional(p_float("epsilonInf", "ε∞", 1.0, None)),
            ],
        ),
        // --- fit ------------------------------------------------------------
        entry(
            "fit.linear_fit",
            "fit",
            "Linear fit",
            "LinearFit",
            "series",
            "scalar",
            &["xySeries"],
            vec![
                p_float("startFrac", "Window start", 0.2, None),
                p_float("endFrac", "Window end", 0.8, None),
            ],
        ),
        entry(
            "fit.running_integral",
            "fit",
            "Running integral",
            "CumulativeTrapezoid",
            "series",
            "lineSeries",
            &["series"],
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                optional(p_int("nLags", "Lags", 200, 1.0, 1e6)),
            ],
        ),
        entry(
            "fit.plateau",
            "fit",
            "Plateau",
            "Plateau",
            "series",
            "scalar",
            &["series"],
            vec![
                p_float("startFrac", "Window start", 0.2, None),
                p_float("endFrac", "Window end", 0.8, None),
            ],
        ),
        entry(
            "fit.debye_fit",
            "fit",
            "Debye fit",
            "DebyeFit",
            "series",
            "scalar",
            &["series"],
            vec![p_float("dt", "Timestep", 1.0, Some("fs"))],
        ),
        // --- cluster --------------------------------------------------------
        entry(
            "cluster.connected_components",
            "cluster",
            "Cluster analysis",
            "Cluster",
            "frameNeighbors",
            "barSeries",
            &[],
            vec![
                p_cutoff(3.0),
                p_int("minClusterSize", "Min cluster size", 1, 1.0, 1e6),
            ],
        ),
        // --- shape ----------------------------------------------------------
        entry(
            "shape.cluster_properties",
            "cluster",
            "Radius of gyration",
            "RadiusOfGyration",
            "frameClusters",
            "table",
            &[],
            vec![
                p_cutoff(3.0),
                call(p_int("minClusterSize", "Min cluster size", 1, 1.0, 1e6)),
            ],
        ),
        entry(
            "shape.center_of_mass",
            "shape",
            "Center of mass",
            "CenterOfMass",
            "frameClusters",
            "table",
            &[],
            vec![
                p_cutoff(3.0),
                call(p_int("minClusterSize", "Min cluster size", 1, 1.0, 1e6)),
            ],
        ),
        entry(
            "shape.gyration_tensor",
            "shape",
            "Gyration tensor",
            "GyrationTensor",
            "frameClusters",
            "matrix",
            &[],
            vec![
                p_cutoff(3.0),
                call(p_int("minClusterSize", "Min cluster size", 1, 1.0, 1e6)),
            ],
        ),
        entry(
            "shape.inertia_tensor",
            "shape",
            "Inertia tensor",
            "InertiaTensor",
            "frameClusters",
            "matrix",
            &[],
            vec![
                p_cutoff(3.0),
                call(p_int("minClusterSize", "Min cluster size", 1, 1.0, 1e6)),
            ],
        ),
        // --- density --------------------------------------------------------
        entry(
            "density.correlation_function",
            "density",
            "Correlation function",
            "CorrelationFunction",
            "frameNeighbors",
            "lineSeries",
            &["scalarField"],
            vec![
                p_cutoff(10.0),
                p_int("nBins", "Bins", 100, 1.0, 4096.0),
                p_float("rMax", "r max", 10.0, Some("Å")),
                optional(p_float("rMin", "r min", 0.0, Some("Å"))),
            ],
        ),
        entry(
            "density.gaussian_density",
            "density",
            "Gaussian density",
            "GaussianDensity",
            "frame",
            "grid3",
            &[],
            vec![
                p_int("nx", "Grid x", 32, 1.0, 512.0),
                p_int("ny", "Grid y", 32, 1.0, 512.0),
                p_int("nz", "Grid z", 32, 1.0, 512.0),
                p_float("sigma", "Sigma", 1.0, Some("Å")),
                optional(p_float("rMax", "Cutoff", 5.0, Some("Å"))),
            ],
        ),
        entry(
            "density.local_density",
            "density",
            "Local density",
            "LocalDensity",
            "frameNeighbors",
            "lineSeries",
            &[],
            vec![
                p_cutoff(5.0),
                p_float("rMax", "r max", 5.0, Some("Å")),
                optional(p_float("diameter", "Particle diameter", 1.0, Some("Å"))),
            ],
        ),
        entry(
            "density.spatial_distribution",
            "density",
            "Spatial distribution",
            "SpatialDistribution",
            "accumulate",
            "grid3",
            &["referenceAtoms", "template", "targetAtoms"],
            vec![
                p_int("nx", "Grid x", 32, 1.0, 512.0),
                p_int("ny", "Grid y", 32, 1.0, 512.0),
                p_int("nz", "Grid z", 32, 1.0, 512.0),
                p_float("extentX", "Extent x", 10.0, Some("Å")),
                p_float("extentY", "Extent y", 10.0, Some("Å")),
                p_float("extentZ", "Extent z", 10.0, Some("Å")),
                optional(p_float("bulkDensity", "Bulk density", 0.0, Some("Å⁻³"))),
            ],
        ),
        entry(
            "density.sphere_voxelization",
            "density",
            "Sphere voxelization",
            "SphereVoxelization",
            "frame",
            "grid3",
            &[],
            vec![
                p_int("nx", "Grid x", 32, 1.0, 512.0),
                p_int("ny", "Grid y", 32, 1.0, 512.0),
                p_int("nz", "Grid z", 32, 1.0, 512.0),
                p_float("rMax", "Sphere radius", 2.0, Some("Å")),
            ],
        ),
        // --- order ----------------------------------------------------------
        entry(
            "order.steinhardt",
            "order",
            "Steinhardt",
            "Steinhardt",
            "frameNeighbors",
            "matrix",
            &[],
            vec![
                p_cutoff(3.0),
                p_list("lValues", "l values", "intList", "6"),
                optional(p_bool("average", "Averaged", false)),
                optional(p_bool("wl", "Compute w_l", false)),
                optional(p_bool("wlNormalize", "Normalize w_l", false)),
            ],
        ),
        entry(
            "order.hexatic",
            "order",
            "Hexatic",
            "Hexatic",
            "frameNeighbors",
            "lineSeries",
            &[],
            vec![p_cutoff(3.0), p_int("k", "Symmetry k", 6, 1.0, 32.0)],
        ),
        entry(
            "order.nematic",
            "order",
            "Nematic",
            "Nematic",
            "series",
            "scalar",
            &["orientation"],
            vec![],
        ),
        entry(
            "order.cubatic",
            "order",
            "Cubatic",
            "Cubatic",
            "series",
            "scalar",
            &["orientation"],
            vec![
                optional(p_int("seed", "Seed", 0, 0.0, 4.294e9)),
                optional(p_float("initialTemp", "Initial temperature", 5.0, None)),
                optional(p_float("coolingRate", "Cooling rate", 0.9, None)),
                optional(p_int("nSteps", "Steps", 100, 1.0, 1e6)),
                optional(p_int("nChains", "Chains", 10, 1.0, 1e4)),
            ],
        ),
        entry(
            "order.solid_liquid",
            "order",
            "Solid-liquid",
            "SolidLiquid",
            "frameNeighbors",
            "table",
            &[],
            vec![
                p_cutoff(3.0),
                p_int("l", "l", 6, 0.0, 32.0),
                optional(p_float("qThreshold", "q threshold", 0.7, None)),
                optional(p_int("nThreshold", "Neighbor threshold", 6, 0.0, 64.0)),
                optional(p_bool("normalizeQ", "Normalize q", true)),
            ],
        ),
        entry(
            "order.rotational_autocorrelation",
            "order",
            "Rotational autocorrelation",
            "RotationalAutocorrelation",
            "series",
            "lineSeries",
            &["orientation"],
            vec![p_int("l", "l", 2, 0.0, 32.0)],
        ),
        // --- environment ----------------------------------------------------
        entry(
            "environment.bond_order",
            "environment",
            "Bond order",
            "BondOrientationalOrder",
            "frameNeighbors",
            "matrix",
            &[],
            vec![
                p_cutoff(3.0),
                p_int("nTheta", "θ bins", 60, 1.0, 512.0),
                p_int("nPhi", "φ bins", 30, 1.0, 512.0),
            ],
        ),
        entry(
            "environment.local_descriptors",
            "environment",
            "Local descriptors",
            "LocalDescriptors",
            "frameNeighbors",
            "matrix",
            &[],
            vec![p_cutoff(3.0), p_int("lMax", "l max", 6, 0.0, 32.0)],
        ),
        entry(
            "environment.angular_separation",
            "environment",
            "Angular separation",
            "AngularSeparation",
            "series",
            "lineSeries",
            &["orientation"],
            vec![optional(p_bool(
                "equivalentOrientations",
                "Fold equivalent orientations",
                false,
            ))],
        ),
        entry(
            "environment.environment_matching",
            "environment",
            "Environment matching",
            "EnvironmentMatch",
            "frameNeighbors",
            "table",
            &[],
            vec![
                p_cutoff(3.0),
                p_float("rmsdThreshold", "RMSD threshold", 0.1, Some("Å")),
                optional(p_bool("registration", "Registration", false)),
                optional(p_int(
                    "maxNeighborsForRegistration",
                    "Max neighbors",
                    12,
                    1.0,
                    128.0,
                )),
            ],
        ),
        // --- diffraction ----------------------------------------------------
        entry(
            "diffraction.static_structure_factor",
            "diffraction",
            "Static structure factor S(k)",
            "StaticStructureFactorDebye",
            "frame",
            "lineSeries",
            &[],
            vec![
                p_float("kMin", "k min", 0.1, Some("Å⁻¹")),
                p_float("kMax", "k max", 10.0, Some("Å⁻¹")),
                p_int("nK", "k samples", 100, 1.0, 4096.0),
            ],
        ),
        entry(
            "diffraction.diffraction_pattern",
            "diffraction",
            "Diffraction pattern",
            "DiffractionPattern",
            "frame",
            "matrix",
            &[],
            vec![
                p_int("nGrid", "Grid", 512, 8.0, 4096.0),
                p_float("sigma", "Sigma", 1.0, None),
                optional(p_int("axis", "Zone axis", 2, 0.0, 2.0)),
            ],
        ),
        // --- distribution ---------------------------------------------------
        entry(
            "distribution.distance_distribution",
            "distribution",
            "Distance distribution",
            "DistanceDistribution",
            "frameGroups",
            "lineSeries",
            &["atomPairs"],
            vec![
                p_int("nBins", "Bins", 100, 1.0, 4096.0),
                p_float("min", "Min", 0.0, Some("Å")),
                p_float("max", "Max", 10.0, Some("Å")),
            ],
        ),
        entry(
            "distribution.angle_distribution",
            "distribution",
            "Angle distribution",
            "AngleDistribution",
            "frameGroups",
            "lineSeries",
            &["atomTriples"],
            vec![p_int("nBins", "Bins", 100, 1.0, 4096.0)],
        ),
        entry(
            "distribution.dihedral_distribution",
            "distribution",
            "Dihedral distribution",
            "DihedralDistribution",
            "frameGroups",
            "lineSeries",
            &["atomQuads"],
            vec![p_int("nBins", "Bins", 100, 1.0, 4096.0)],
        ),
        entry(
            "distribution.combined_distribution",
            "distribution",
            "Combined distribution",
            "CombinedDistribution",
            "frameGroupSets",
            "matrix",
            &["atomGroups"],
            vec![
                p_list("kinds", "Observables", "textList", "distance,angle"),
                p_list("bins", "Bins per axis", "intList", "50,50"),
                p_list("mins", "Axis minima", "floatList", "0,0"),
                p_list("maxs", "Axis maxima", "floatList", "10,3.14159265"),
                optional(p_list("sinWeight", "sin θ weighting", "intList", "0,1")),
            ],
        ),
        // --- pmft -----------------------------------------------------------
        entry(
            "pmft.pmft_r12",
            "pmft",
            "PMFT R12",
            "PmftR12",
            "frameNeighbors",
            "matrix",
            &["orientation"],
            vec![
                p_cutoff(5.0),
                p_float("rMax", "r max", 5.0, Some("Å")),
                p_int("nR", "r bins", 50, 1.0, 1024.0),
                p_int("nT1", "θ₁ bins", 36, 1.0, 1024.0),
                p_int("nT2", "θ₂ bins", 36, 1.0, 1024.0),
            ],
        ),
        entry(
            "pmft.pmft_xy",
            "pmft",
            "PMFT XY",
            "PmftXy",
            "frameNeighbors",
            "matrix",
            &[],
            vec![
                p_cutoff(5.0),
                p_float("xMax", "x max", 5.0, Some("Å")),
                p_float("yMax", "y max", 5.0, Some("Å")),
                p_int("nX", "x bins", 50, 1.0, 1024.0),
                p_int("nY", "y bins", 50, 1.0, 1024.0),
            ],
        ),
        entry(
            "pmft.pmft_xyt",
            "pmft",
            "PMFT XYT",
            "PmftXyt",
            "frameNeighbors",
            "matrix",
            &["orientation"],
            vec![
                p_cutoff(5.0),
                p_float("xMax", "x max", 5.0, Some("Å")),
                p_float("yMax", "y max", 5.0, Some("Å")),
                p_int("nX", "x bins", 50, 1.0, 1024.0),
                p_int("nY", "y bins", 50, 1.0, 1024.0),
                p_int("nT", "θ bins", 36, 1.0, 1024.0),
            ],
        ),
        entry(
            "pmft.pmft_xyz",
            "pmft",
            "PMFT XYZ",
            "PmftXyz",
            "frameNeighbors",
            "matrix",
            &[],
            vec![
                p_cutoff(5.0),
                p_float("xMax", "x max", 5.0, Some("Å")),
                p_float("yMax", "y max", 5.0, Some("Å")),
                p_float("zMax", "z max", 5.0, Some("Å")),
                p_int("nX", "x bins", 30, 1.0, 512.0),
                p_int("nY", "y bins", 30, 1.0, 512.0),
                p_int("nZ", "z bins", 30, 1.0, 512.0),
            ],
        ),
        // --- hbond ----------------------------------------------------------
        entry(
            "hbond.hydrogen_bond_detection",
            "hbond",
            "Hydrogen-bond detection",
            "HBonds",
            "accumulate",
            "table",
            &["donors", "acceptors"],
            vec![
                optional(p_float("distCutoff", "Distance cutoff", 3.5, Some("Å"))),
                optional(p_select(
                    "distKind",
                    "Distance criterion",
                    "donor_acceptor",
                    &["donor_acceptor", "hydrogen_acceptor"],
                )),
                optional(p_float("angleCutoff", "Angle cutoff", 150.0, Some("°"))),
            ],
        ),
        entry(
            "hbond.lifetime",
            "hbond",
            "Lifetime",
            "HBondLifetime",
            "series",
            "lineSeries",
            &["hbondPresence"],
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_int("maxLag", "Max lag", 200, 1.0, 1e6),
            ],
        ),
        entry(
            "hbond.network_components",
            "hbond",
            "Network components",
            "HBondNetwork",
            "series",
            "table",
            &["hbondEdges"],
            vec![],
        ),
        // --- locality (freud.locality: Voronoi; neighbor queries are infra) --
        entry(
            "locality.radical_voronoi",
            "locality",
            "Radical Voronoi",
            "RadicalVoronoi",
            "frameRadii",
            "table",
            &[],
            vec![p_bool("useAtomRadii", "Weight by covalent radii", true)],
        ),
        entry(
            "locality.voronoi_domain_analysis",
            "locality",
            "Domain analysis",
            "VoronoiDomainAnalysis",
            "frameRadii",
            "table",
            &["labels"],
            vec![
                p_bool("useAtomRadii", "Weight by covalent radii", true),
                call(p_select(
                    "labelBy",
                    "Label cells by",
                    "element",
                    &[keys::ELEMENT, keys::MOL_ID, keys::TYPE],
                )),
            ],
        ),
        entry(
            "locality.voronoi_void_analysis",
            "locality",
            "Void analysis",
            "VoronoiVoidAnalysis",
            "frameRadii",
            "table",
            &["voidMask"],
            vec![
                p_bool("useAtomRadii", "Weight by covalent radii", true),
                optional(p_float("boxVolume", "Box volume", 0.0, Some("Å³"))),
            ],
        ),
        // --- ml -------------------------------------------------------------
        entry(
            "ml.pca",
            "ml",
            "PCA",
            "Pca",
            "series",
            "custom",
            &["descriptorMatrix"],
            vec![],
        ),
        entry(
            "ml.kmeans",
            "ml",
            "k-means",
            "KMeans",
            "series",
            "custom",
            &["descriptorMatrix"],
            vec![
                p_int("k", "Clusters", 3, 1.0, 1024.0),
                p_int("maxIter", "Max iterations", 100, 1.0, 1e5),
                p_int("seed", "Seed", 0, 0.0, 4.294e9),
            ],
        ),
    ];

    js_value(&ComputeCatalog {
        // v3: freud core order + molrs extensions — dynamics→transport,
        // static dielectric→spectroscopy; cluster_properties→cluster.
        // v4: `wasmExport` names drop the `Wasm` prefix (`WasmVACF` → `VACF`).
        // v5: exports cased as words (`VACF` → `Vacf`, `PMFTXY` → `PmftXy`,
        // `MatchEnv` → `EnvironmentMatch`, `PairPersistence` → `PairSurvival`);
        // the `rdf.*` / `voronoi.*` id prefixes are gone (`density.*`,
        // `locality.*`), and `dynamics.pair_persistence` is
        // `dynamics.pair_survival`.
        version: 5,
        categories: &CATEGORIES,
        analyses,
    })
}
