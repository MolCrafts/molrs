//! Python bindings for `molrs::builder`: the structure generators
//! `CarbonTubeBuilder` and `GrapheneBuilder`, each building a fresh `Frame`,
//! and site-graph assembly — `Assembler(library, SitePlacer(),
//! AxisOrienter()).assemble(sites)`, which orients, places and links one world
//! `Fragment`.

use std::collections::HashMap;

use molrs::builder::{
    AssembleError, Assembler, AxisOrienter, GrowthPlacer, OrientError, PlaceError, Placer,
    SitePlacer,
};
use molrs::builder::{CarbonTubeBuilder, GrapheneBuilder};
use molrs::system::LinkManyError;
use molrs::system::MolGraph;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyMapping;

use crate::core::spatial::simbox::PyBox;
use crate::core::store::frame::PyFrame;
use crate::core::system::molgraph::{
    AnyGraph, GraphClass, PyCoarseGrain, center_error_message, link_error_message,
};

/// Exact single-wall carbon nanotube builder.
#[pyclass(module = "molrs.builder", name = "CarbonTubeBuilder", subclass)]
pub struct PyCarbonTubeBuilder {
    inner: CarbonTubeBuilder,
}

#[pymethods]
impl PyCarbonTubeBuilder {
    #[new]
    #[pyo3(signature = (n, m, *, length=None, cells=None, bond_length=1.42, periodic=false, vacuum=10.0))]
    fn new(
        n: u32,
        m: u32,
        length: Option<f64>,
        cells: Option<usize>,
        bond_length: f64,
        periodic: bool,
        vacuum: f64,
    ) -> PyResult<Self> {
        if length.is_some() && cells.is_some() {
            return Err(PyTypeError::new_err(
                "length and cells are mutually exclusive",
            ));
        }

        let mut inner = CarbonTubeBuilder::new(n, m)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .with_bond_length(bond_length)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .with_periodic(periodic)
            .with_vacuum(vacuum)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if let Some(cells) = cells {
            inner = inner
                .with_cells(cells)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        if let Some(length) = length {
            inner = inner
                .with_length(length)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        inner
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    /// Build a fresh frame containing atoms, bonds, and the simulation box.
    #[pyo3(signature = (*, atom_type=None, charge=0.0))]
    fn build(&self, atom_type: Option<String>, charge: f64) -> PyResult<PyFrame> {
        let mut builder = self
            .inner
            .clone()
            .with_charge(charge)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if let Some(atom_type) = atom_type {
            builder = builder
                .with_atom_type(atom_type)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        PyFrame::from_core_frame(
            builder
                .build()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )
    }

    /// Return the matching simulation cell, optionally overriding vacuum.
    #[pyo3(signature = (*, vacuum=None))]
    fn cell(&self, vacuum: Option<f64>) -> PyResult<PyBox> {
        let builder = match vacuum {
            Some(vacuum) => self
                .inner
                .clone()
                .with_vacuum(vacuum)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            None => self.inner.clone(),
        };
        Ok(PyBox {
            inner: builder
                .cell()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[getter]
    fn n(&self) -> u32 {
        self.inner.n()
    }

    #[getter]
    fn m(&self) -> u32 {
        self.inner.m()
    }

    #[getter]
    fn cells(&self) -> usize {
        self.inner.cells()
    }

    #[getter]
    fn bond_length(&self) -> f64 {
        self.inner.bond_length()
    }

    #[getter]
    fn periodic(&self) -> bool {
        self.inner.periodic()
    }
}

/// Rectangular graphene (honeycomb) sheet builder.
#[pyclass(module = "molrs.builder", name = "GrapheneBuilder", subclass)]
pub struct PyGrapheneBuilder {
    inner: GrapheneBuilder,
}

#[pymethods]
impl PyGrapheneBuilder {
    #[new]
    #[pyo3(signature = (nx, ny, *, bond_length=1.42, vacuum=10.0, periodic_xy=true))]
    fn new(nx: u32, ny: u32, bond_length: f64, vacuum: f64, periodic_xy: bool) -> PyResult<Self> {
        let inner = GrapheneBuilder::new(nx, ny)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .with_bond_length(bond_length)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .with_vacuum(vacuum)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .with_periodic_xy(periodic_xy);
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, atom_type=None, charge=0.0))]
    fn build(&self, atom_type: Option<String>, charge: f64) -> PyResult<PyFrame> {
        let mut builder = self
            .inner
            .clone()
            .with_charge(charge)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Some(atom_type) = atom_type {
            builder = builder
                .with_atom_type(atom_type)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        PyFrame::from_core_frame(
            builder
                .build()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        )
    }

    #[pyo3(signature = (*, vacuum=None))]
    fn cell(&self, vacuum: Option<f64>) -> PyResult<PyBox> {
        let builder = match vacuum {
            Some(v) => self
                .inner
                .clone()
                .with_vacuum(v)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            None => self.inner.clone(),
        };
        Ok(PyBox {
            inner: builder
                .cell()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    #[getter]
    fn nx(&self) -> u32 {
        self.inner.nx()
    }

    #[getter]
    fn ny(&self) -> u32 {
        self.inner.ny()
    }

    #[getter]
    fn bond_length(&self) -> f64 {
        self.inner.bond_length()
    }

    #[getter]
    fn periodic_xy(&self) -> bool {
        self.inner.periodic_xy()
    }
}

/// The translation-only placer — ``molrs.builder.SitePlacer``.
///
/// Each copy's centre of mass (Å, weights ``mass``) lands on its site;
/// rotation is the orienter's job. Takes no arguments.
///
/// Examples
/// --------
/// >>> assembler = molrs.builder.Assembler(
/// ...     library, molrs.builder.SitePlacer(), molrs.builder.AxisOrienter()
/// ... )
#[pyclass(module = "molrs.builder", name = "SitePlacer", frozen)]
pub struct PySitePlacer;

#[pymethods]
impl PySitePlacer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "SitePlacer()".to_owned()
    }
}

/// The growth placer — ``molrs.builder.GrowthPlacer``.
///
/// Grows each molecule copy by copy in the assembler's walk order: the first
/// copy keeps its template pose (its centre of mass on the site when the
/// site graph has positions), and every later copy is rotated and moved so
/// the anchor of its port toward its parent lands on the parent's leaving
/// handle, pointing back along that bond. Needs no site positions; bond
/// lengths and overlaps are left to a later minimisation. Takes no
/// arguments.
#[pyclass(module = "molrs.builder", name = "GrowthPlacer", frozen)]
pub struct PyGrowthPlacer;

#[pymethods]
impl PyGrowthPlacer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "GrowthPlacer()".to_owned()
    }
}

/// The axis orienter — ``molrs.builder.AxisOrienter``.
///
/// Turns each copy about its template's centre of mass. A chain site (only
/// ``<`` / ``>`` ports, a two-port template) matches the template's
/// backbone-to-centre direction to the site axis (``CoarseGrain.axes``) and
/// its two joining atoms to the site's bond line. Any other bonded site fits
/// the template's port directions to its bond directions. A site with no
/// bond is not turned. Takes no arguments.
#[pyclass(module = "molrs.builder", name = "AxisOrienter", frozen)]
pub struct PyAxisOrienter;

#[pymethods]
impl PyAxisOrienter {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "AxisOrienter()".to_owned()
    }
}

/// One placed, linked world from a site graph — ``molrs.builder.Assembler``.
///
/// Each bead of the site graph is one unit: a copy of
/// ``library[bead_type]``, turned by the orienter (when given) and given its
/// pose by the placer.
/// Each site bond joins one port of each end's copy (``<`` with ``>``, ``$``
/// with ``$``); the leaving groups are removed. Any topology works: chains,
/// branches, rings. Every atom gets ``frag_id`` (the site's ordinal) and
/// ``mol_id`` (its connected component's ordinal + 1).
///
/// Parameters
/// ----------
/// library : Mapping[str, Graph]
///     Name → template, copied at construction: any graph (``Graph``,
///     ``Atomistic``, ``CoarseGrain``), with ports where a site bonds. A
///     template without ports can only fill an unbonded site.
/// placer : SitePlacer | GrowthPlacer
///     ``SitePlacer`` moves each copy's centre of mass onto its site;
///     ``GrowthPlacer`` grows each molecule onto its parents' ports and needs
///     no site positions.
/// orienter : AxisOrienter, optional
///     Turns each copy about its centre of mass before it is placed; needs
///     site positions.
///
/// Raises
/// ------
/// TypeError
///     If ``library`` is not a mapping of ``str`` to graphs, ``placer`` is
///     not a :class:`SitePlacer` or :class:`GrowthPlacer`, or ``orienter``
///     is not an :class:`AxisOrienter`.
///
/// Examples
/// --------
/// >>> sites = molrs.perceive.Coarsener(cg).coarsen(groups, names)
/// >>> world = molrs.builder.Assembler(
/// ...     {"PMA": pma, "Li": li},
/// ...     molrs.builder.SitePlacer(),
/// ...     molrs.builder.AxisOrienter(),
/// ... ).assemble(sites, molrs.Atomistic)
#[pyclass(module = "molrs.builder", name = "Assembler", frozen)]
pub struct PyAssembler {
    inner: Assembler,
}

#[pymethods]
impl PyAssembler {
    #[new]
    #[pyo3(signature = (library, placer, orienter=None))]
    fn new(
        library: &Bound<'_, PyMapping>,
        placer: &Bound<'_, PyAny>,
        orienter: Option<&Bound<'_, PyAxisOrienter>>,
    ) -> PyResult<Self> {
        // Neither placer nor orienter carries state: the type is the whole
        // choice.
        let placer: Box<dyn Placer> = if placer.cast::<PySitePlacer>().is_ok() {
            Box::new(SitePlacer::new())
        } else if placer.cast::<PyGrowthPlacer>().is_ok() {
            Box::new(GrowthPlacer::new())
        } else {
            return Err(PyTypeError::new_err(format!(
                "placer must be a SitePlacer or a GrowthPlacer, not {}",
                placer.get_type().name()?
            )));
        };
        let orienter = orienter.map(|_| Box::new(AxisOrienter::new()) as _);
        let mut templates: HashMap<String, MolGraph> = HashMap::new();
        for item in library.items()?.iter() {
            let (name, value): (String, Bound<'_, PyAny>) = item.extract()?;
            let template = AnyGraph::of(&value)
                .map_err(|_| {
                    PyTypeError::new_err(format!(
                        "library['{name}'] must be a graph (Graph, Atomistic, CoarseGrain)"
                    ))
                })?
                .to_molgraph()?;
            templates.insert(name, template);
        }
        Ok(Self {
            inner: Assembler::new(templates, placer, orienter),
        })
    }

    /// Place and join one copy per site; return the world.
    ///
    /// The GIL is released while assembling.
    ///
    /// Parameters
    /// ----------
    /// sites : CoarseGrain
    ///     The site graph: each bead's ``bead_type`` names its template and
    ///     each bond joins two copies; its optional position (Å) and axis
    ///     are read by the placer and the orienter.
    /// cls : type, optional
    ///     The graph class to build the world as — ``Graph`` (the default),
    ///     ``Atomistic`` or ``CoarseGrain``.
    ///
    /// Returns
    /// -------
    /// Graph
    ///     The world, an instance of ``cls``; ports without a site bond stay
    ///     on it. Empty when ``sites`` is empty.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Naming the site at fault: a bead lacks its type or position, a name
    ///     is not in the library, no accepting port exists for every bond of
    ///     a site, a site cannot be oriented or placed, a join is refused, or
    ///     the world breaks ``cls``'s invariant.
    /// TypeError
    ///     If ``cls`` is not a graph class.
    #[pyo3(signature = (sites, cls=None))]
    fn assemble(
        &self,
        py: Python<'_>,
        sites: PyRef<'_, PyCoarseGrain>,
        cls: Option<&Bound<'_, pyo3::types::PyType>>,
    ) -> PyResult<Py<PyAny>> {
        let sites = sites.core().clone();
        let assembler = &self.inner;
        let world: MolGraph = py
            .detach(|| assembler.assemble(&sites))
            .map_err(|e| PyValueError::new_err(assemble_error_message(e)))?;
        GraphClass::of(py, cls)?.build(py, world)
    }

    fn __repr__(&self) -> String {
        "Assembler(...)".to_owned()
    }
}

/// The message of an [`AssembleError`] as Python sees it: node and port ids
/// as int handles, never `NodeId(..)` / `RelationId(..)`. Every variant is
/// matched by name; the wording of the outer sentence is the core's.
fn assemble_error_message(e: AssembleError) -> String {
    match e {
        AssembleError::Place { name, site, source } => {
            let reason = match source {
                PlaceError::Template(center) => format!(
                    "the template has no centre of mass: {}",
                    center_error_message(center)
                ),
                other @ (PlaceError::NoPosition
                | PlaceError::NonFinitePoint
                | PlaceError::Port(_)) => other.to_string(),
            };
            format!("site {site} ('{name}') cannot be placed: {reason}")
        }
        AssembleError::Orient { name, site, source } => {
            let reason = match source {
                OrientError::Center(center) => format!(
                    "the template has no centre of mass: {}",
                    center_error_message(center)
                ),
                other @ (OrientError::Template(_)
                | OrientError::NoAxis { .. }
                | OrientError::Frame { .. }) => other.to_string(),
            };
            format!("site {site} ('{name}') cannot be oriented: {reason}")
        }
        AssembleError::Link {
            site,
            partner,
            source,
        } => {
            let reason = match source {
                LinkManyError::Pair { pair, source } => {
                    format!("pair {pair} is refused: {}", link_error_message(source))
                }
                batch @ (LinkManyError::PortReused { .. }
                | LinkManyError::DuplicateBond { .. }
                | LinkManyError::BranchesOverlap { .. }) => batch.to_string(),
            };
            format!("site {site} cannot join site {partner}: {reason}")
        }
        e @ (AssembleError::TooManyUnits { .. }
        | AssembleError::Output(_)
        | AssembleError::Sites(_)
        | AssembleError::UnknownName { .. }
        | AssembleError::Template { .. }
        | AssembleError::Ports { .. }
        | AssembleError::Replicate { .. }) => e.to_string(),
    }
}
