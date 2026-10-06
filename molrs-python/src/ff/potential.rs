//! Python bindings for `molrs::ff::potential` — the kernels themselves.
//!
//! One class per kernel, constructed from explicit instances (atom indices and
//! one parameter row each, in the force field's convention: LAMMPS's, angles in
//! degrees) and **moved** into a `Potentials` collection by
//! ``Potentials.push``. They are the parts a compiled force field is made of,
//! exposed so a caller can assemble one by hand.
//!
//! ```text
//! pots = Potentials()
//! pots.push(BondHarmonic(atomi, atomj, k, r0))
//! pots.push(DihedralPeriodic(i, j, k, l, k_mn, periodicity_mn, phase_mn))
//! energy, forces = pots.calc_energy_forces(pos)
//! ```

use molrs::ff::potential::angle::AngleHarmonic;
use molrs::ff::potential::bond::BondHarmonic;
use molrs::ff::potential::dihedral::periodic::DihedralPeriodic;
use molrs::ff::potential::improper::{ImproperCvff, ImproperPeriodic};
use molrs::ff::potential::pair::{LJCut, PairCoulCut, PairPotential};
use molrs::ff::potential::{Member, Potential};
use molrs::types::F;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::spatial::neighborlist::{PyNeighbors, PyVerletSkin};
use crate::helpers::NpF;
use crate::md::check_nx3;

fn same_len(what: &str, lens: &[usize], n: usize) -> PyResult<()> {
    if lens.iter().any(|&len| len != n) {
        return Err(PyValueError::new_err(format!(
            "{what}: every array must have one entry per instance ({n})"
        )));
    }
    Ok(())
}

/// `(energy, forces)` of a kernel at `pos` `(N, 3)`.
fn evaluate<'py>(
    py: Python<'py>,
    kernel: &dyn Potential,
    pos: PyReadonlyArray2<'_, NpF>,
) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
    check_nx3(&pos, "pos")?;
    let view = pos.as_array();
    let n = view.nrows();
    let flat: Vec<F> = view.iter().copied().collect();
    let (energy, forces) = kernel.calc_energy_forces(&flat);
    let arr =
        Array2::from_shape_vec((n, 3), forces).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((energy, arr.into_pyarray(py)))
}

fn moved_err(what: &str) -> PyErr {
    PyValueError::new_err(format!(
        "this {what} has been moved into a Potentials; build it again"
    ))
}

/// A kernel class holding its Rust kernel until ``Potentials.push`` moves it.
macro_rules! owned_kernel {
    ($py:ident, $name:literal, $kernel:ty, $member:ident) => {
        impl $py {
            fn kernel(&self) -> PyResult<&$kernel> {
                self.inner.as_ref().ok_or_else(|| moved_err($name))
            }

            pub(crate) fn take_member(&mut self) -> PyResult<Member> {
                self.inner
                    .take()
                    .map(Member::$member)
                    .ok_or_else(|| moved_err($name))
            }
        }
    };
}

/// LAMMPS ``bond_style harmonic``: ``k (r − r0)²`` per bond.
#[pyclass(name = "BondHarmonic", module = "molrs.ff.potential", subclass)]
pub struct PyBondHarmonic {
    inner: Option<BondHarmonic>,
}
owned_kernel!(PyBondHarmonic, "BondHarmonic", BondHarmonic, indexed);

#[pymethods]
impl PyBondHarmonic {
    #[new]
    fn new(atomi: Vec<usize>, atomj: Vec<usize>, k: Vec<F>, r0: Vec<F>) -> PyResult<Self> {
        same_len(
            "BondHarmonic",
            &[atomj.len(), k.len(), r0.len()],
            atomi.len(),
        )?;
        Ok(Self {
            inner: Some(BondHarmonic::new(atomi, atomj, k, r0)),
        })
    }

    /// Returns ``(energy, forces)`` at ``pos`` ``(N, 3)``.
    fn calc_energy_forces<'py>(
        &self,
        py: Python<'py>,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        evaluate(py, self.kernel()?, pos)
    }
}

/// LAMMPS ``angle_style harmonic``: ``k (θ − theta0)²`` per angle, ``theta0`` in degrees.
#[pyclass(name = "AngleHarmonic", module = "molrs.ff.potential", subclass)]
pub struct PyAngleHarmonic {
    inner: Option<AngleHarmonic>,
}
owned_kernel!(PyAngleHarmonic, "AngleHarmonic", AngleHarmonic, indexed);

#[pymethods]
impl PyAngleHarmonic {
    #[new]
    fn new(
        atomi: Vec<usize>,
        atomj: Vec<usize>,
        atomk: Vec<usize>,
        k: Vec<F>,
        theta0: Vec<F>,
    ) -> PyResult<Self> {
        same_len(
            "AngleHarmonic",
            &[atomj.len(), atomk.len(), k.len(), theta0.len()],
            atomi.len(),
        )?;
        let theta0 = theta0.into_iter().map(F::to_radians).collect();
        Ok(Self {
            inner: Some(AngleHarmonic::new(atomi, atomj, atomk, k, theta0)),
        })
    }

    /// Returns ``(energy, forces)`` at ``pos`` ``(N, 3)``.
    fn calc_energy_forces<'py>(
        &self,
        py: Python<'py>,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        evaluate(py, self.kernel()?, pos)
    }
}

/// ``dihedral periodic`` (LAMMPS ``dihedral_style fourier``):
/// ``Σₘ kₘ [1 + cos(nₘ φ − γₘ)]`` per dihedral.
///
/// ``k``, ``periodicity`` and ``phase`` are ``(M, T)``: row ``idx`` is the
/// ``T``-term series of dihedral ``idx``, the phase in degrees. A term with
/// ``k = 0`` contributes nothing, so series of different lengths share one
/// ``T`` by padding.
#[pyclass(name = "DihedralPeriodic", module = "molrs.ff.potential", subclass)]
pub struct PyDihedralPeriodic {
    inner: Option<DihedralPeriodic>,
}
owned_kernel!(
    PyDihedralPeriodic,
    "DihedralPeriodic",
    DihedralPeriodic,
    indexed
);

#[pymethods]
impl PyDihedralPeriodic {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        atomi: Vec<usize>,
        atomj: Vec<usize>,
        atomk: Vec<usize>,
        atoml: Vec<usize>,
        k: PyReadonlyArray2<'_, NpF>,
        periodicity: PyReadonlyArray2<'_, NpF>,
        phase: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Self> {
        let n = atomi.len();
        same_len(
            "DihedralPeriodic",
            &[atomj.len(), atomk.len(), atoml.len()],
            n,
        )?;
        let (k, periodicity, phase) = (k.as_array(), periodicity.as_array(), phase.as_array());
        if k.nrows() != n || periodicity.shape() != k.shape() || phase.shape() != k.shape() {
            return Err(PyValueError::new_err(format!(
                "DihedralPeriodic: k, periodicity and phase must share one shape ({n}, T)"
            )));
        }
        let terms = (0..n)
            .map(|idx| {
                (0..k.ncols())
                    .map(|t| {
                        (
                            k[[idx, t]],
                            periodicity[[idx, t]],
                            phase[[idx, t]].to_radians(),
                        )
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            inner: Some(DihedralPeriodic::new(atomi, atomj, atomk, atoml, terms)),
        })
    }

    /// Returns ``(energy, forces)`` at ``pos`` ``(N, 3)``.
    fn calc_energy_forces<'py>(
        &self,
        py: Python<'py>,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        evaluate(py, self.kernel()?, pos)
    }
}

/// LAMMPS ``improper_style cvff``: ``k [1 + sign·cos(n φ)]``, ``φ`` the
/// dihedral of the stored order, the centre first.
#[pyclass(name = "ImproperCvff", module = "molrs.ff.potential", subclass)]
pub struct PyImproperCvff {
    inner: Option<ImproperCvff>,
}
owned_kernel!(PyImproperCvff, "ImproperCvff", ImproperCvff, indexed);

#[pymethods]
impl PyImproperCvff {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        atomi: Vec<usize>,
        atomj: Vec<usize>,
        atomk: Vec<usize>,
        atoml: Vec<usize>,
        k: Vec<F>,
        sign: Vec<F>,
        periodicity: Vec<F>,
    ) -> PyResult<Self> {
        same_len(
            "ImproperCvff",
            &[
                atomj.len(),
                atomk.len(),
                atoml.len(),
                k.len(),
                sign.len(),
                periodicity.len(),
            ],
            atomi.len(),
        )?;
        Ok(Self {
            inner: Some(ImproperCvff::new(
                atomi,
                atomj,
                atomk,
                atoml,
                k,
                sign,
                periodicity,
            )),
        })
    }

    /// Returns ``(energy, forces)`` at ``pos`` ``(N, 3)``.
    fn calc_energy_forces<'py>(
        &self,
        py: Python<'py>,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        evaluate(py, self.kernel()?, pos)
    }
}

/// ``improper periodic`` (AMBER / GAFF / OpenMM): ``k [1 + cos(n φ − γ)]``,
/// ``φ`` the dihedral of the stored (AMBER, centre third) order, ``γ`` in degrees.
#[pyclass(name = "ImproperPeriodic", module = "molrs.ff.potential", subclass)]
pub struct PyImproperPeriodic {
    inner: Option<ImproperPeriodic>,
}
owned_kernel!(
    PyImproperPeriodic,
    "ImproperPeriodic",
    ImproperPeriodic,
    indexed
);

#[pymethods]
impl PyImproperPeriodic {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        atomi: Vec<usize>,
        atomj: Vec<usize>,
        atomk: Vec<usize>,
        atoml: Vec<usize>,
        k: Vec<F>,
        periodicity: Vec<F>,
        phase: Vec<F>,
    ) -> PyResult<Self> {
        same_len(
            "ImproperPeriodic",
            &[
                atomj.len(),
                atomk.len(),
                atoml.len(),
                k.len(),
                periodicity.len(),
                phase.len(),
            ],
            atomi.len(),
        )?;
        let phase = phase.into_iter().map(F::to_radians).collect();
        Ok(Self {
            inner: Some(ImproperPeriodic::new(
                atomi,
                atomj,
                atomk,
                atoml,
                k,
                periodicity,
                phase,
            )),
        })
    }

    /// Returns ``(energy, forces)`` at ``pos`` ``(N, 3)``.
    fn calc_energy_forces<'py>(
        &self,
        py: Python<'py>,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        evaluate(py, self.kernel()?, pos)
    }
}

/// LAMMPS ``pair_style coul/cut`` over a fixed pair list:
/// ``coulomb · qᵢqⱼ / (dielectric · (r + delta))`` for ``r < cutoff``.
///
/// ``qiqj`` is the charge product of each pair with any 1-4 weight already
/// applied, as ``PotentialCompiler.compile`` builds it.
#[pyclass(name = "PairCoulCut", module = "molrs.ff.potential", subclass)]
pub struct PyPairCoulCut {
    inner: Option<PairCoulCut>,
}
owned_kernel!(PyPairCoulCut, "PairCoulCut", PairCoulCut, pair);

#[pymethods]
impl PyPairCoulCut {
    #[new]
    #[pyo3(signature = (atomi, atomj, qiqj, *, coulomb, dielectric=1.0, delta=0.0, cutoff=F::INFINITY))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        atomi: Vec<usize>,
        atomj: Vec<usize>,
        qiqj: Vec<F>,
        coulomb: F,
        dielectric: F,
        delta: F,
        cutoff: F,
    ) -> PyResult<Self> {
        same_len("PairCoulCut", &[atomj.len(), qiqj.len()], atomi.len())?;
        Ok(Self {
            inner: Some(PairCoulCut::new(
                atomi, atomj, qiqj, coulomb, dielectric, delta, cutoff,
            )),
        })
    }

    /// Returns ``(energy, forces)`` at ``pos`` ``(N, 3)``.
    fn calc_energy_forces<'py>(
        &self,
        py: Python<'py>,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        evaluate(py, self.kernel()?, pos)
    }
}

/// The member a kernel class moves into a collection, or `None` when `obj`
/// is not one of this module's kernels.
pub(crate) fn take_kernel(obj: &Bound<'_, PyAny>) -> Option<PyResult<Member>> {
    if let Ok(k) = obj.cast::<PyLJCut>() {
        return Some(Ok(Member::pair(k.borrow().inner.clone())));
    }
    if let Ok(k) = obj.cast::<PyBondHarmonic>() {
        return Some(k.borrow_mut().take_member());
    }
    if let Ok(k) = obj.cast::<PyAngleHarmonic>() {
        return Some(k.borrow_mut().take_member());
    }
    if let Ok(k) = obj.cast::<PyDihedralPeriodic>() {
        return Some(k.borrow_mut().take_member());
    }
    if let Ok(k) = obj.cast::<PyImproperCvff>() {
        return Some(k.borrow_mut().take_member());
    }
    if let Ok(k) = obj.cast::<PyImproperPeriodic>() {
        return Some(k.borrow_mut().take_member());
    }
    if let Ok(k) = obj.cast::<PyPairCoulCut>() {
        return Some(k.borrow_mut().take_member());
    }
    None
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLJCut>()?;
    m.add_class::<PyBondHarmonic>()?;
    m.add_class::<PyAngleHarmonic>()?;
    m.add_class::<PyDihedralPeriodic>()?;
    m.add_class::<PyImproperCvff>()?;
    m.add_class::<PyImproperPeriodic>()?;
    m.add_class::<PyPairCoulCut>()?;
    Ok(())
}

/// LAMMPS ``pair_style lj/cut``: cut Lennard-Jones / Mie pair kernel.
///
/// The constructor is the one-type kernel a neighbour loop feeds pairs to;
/// :meth:`compiled` is the same style over an explicit pair list with a
/// parameter row per pair.
#[pyclass(name = "LJCut", module = "molrs.ff.potential", subclass)]
pub struct PyLJCut {
    pub(crate) inner: LJCut,
}

#[pymethods]
impl PyLJCut {
    #[new]
    #[pyo3(signature = (epsilon, sigma, cutoff, *, n=12, m=6, shifted=true, smeared=false))]
    fn new(
        epsilon: F,
        sigma: F,
        cutoff: F,
        n: i32,
        m: i32,
        shifted: bool,
        smeared: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LJCut::new(epsilon, sigma, cutoff, n, m, shifted, smeared)
                .map_err(PyValueError::new_err)?,
        })
    }

    /// ``lj/cut`` over a fixed pair list, one ``(epsilon, sigma)`` per pair.
    ///
    /// Each row is priced ``4ε[(σ/r)¹² − (σ/r)⁶]`` with no cutoff and no
    /// shift — what ``PotentialCompiler.compile`` builds from a ``pairs``
    /// block, with any 1-4 weight already folded into ``epsilon``.
    #[staticmethod]
    fn compiled(
        atomi: Vec<usize>,
        atomj: Vec<usize>,
        epsilon: Vec<F>,
        sigma: Vec<F>,
    ) -> PyResult<Self> {
        same_len(
            "LJCut.compiled",
            &[atomj.len(), epsilon.len(), sigma.len()],
            atomi.len(),
        )?;
        Ok(Self {
            inner: LJCut::compiled(atomi, atomj, epsilon, sigma),
        })
    }

    #[getter]
    fn epsilon(&self) -> F {
        self.inner.epsilon()
    }
    #[getter]
    fn sigma(&self) -> F {
        self.inner.sigma()
    }
    #[getter]
    fn cutoff(&self) -> F {
        self.inner.cutoff()
    }
    #[getter]
    fn n(&self) -> i32 {
        self.inner.n()
    }
    #[getter]
    fn m(&self) -> i32 {
        self.inner.m()
    }
    #[getter]
    fn shifted(&self) -> bool {
        self.inner.shifted()
    }
    #[getter]
    fn smeared(&self) -> bool {
        self.inner.smeared()
    }

    fn pair_energy(&self, r2: F, disp: [F; 3]) -> Option<F> {
        self.inner.pair_energy(r2, disp)
    }
    fn pair_force(&self, r2: F, disp: [F; 3]) -> Option<[F; 3]> {
        self.inner.pair_force(r2, disp)
    }
    fn pair_eval(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        self.inner.pair_eval(r2, disp)
    }

    fn calc_energy_forces<'py>(
        &self,
        py: Python<'py>,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        check_nx3(&pos, "pos")?;
        let view = pos.as_array();
        let n = view.nrows();
        // A standard-layout `(N, 3)` array *is* the flat `[x0, y0, z0, …]` a
        // kernel wants; only a strided view is copied.
        let owned: Vec<F>;
        let flat: &[F] = match view.as_slice() {
            Some(slice) => slice,
            None => {
                owned = view.iter().copied().collect();
                &owned
            }
        };
        let (energy, forces) = Potential::calc_energy_forces(&self.inner, flat);
        let arr = Array2::from_shape_vec((n, 3), forces)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((energy, arr.into_pyarray(py)))
    }

    fn eval<'py>(
        &self,
        py: Python<'py>,
        neighbors: &mut PyVerletSkin,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        check_nx3(&pos, "pos")?;
        let nl = neighbors.get_mut()?;
        let (e, f) = self
            .inner
            .eval(nl, pos.as_array())
            .map_err(PyValueError::new_err)?;
        Ok((e, f.into_pyarray(py)))
    }

    fn eval_table<'py>(
        &self,
        py: Python<'py>,
        n_atoms: usize,
        neighbors: &PyNeighbors,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        let (e, f) = self
            .inner
            .eval_table(n_atoms, &neighbors.inner)
            .map_err(PyValueError::new_err)?;
        Ok((e, f.into_pyarray(py)))
    }

    #[pyo3(signature = (n_atoms, i, j, disp, dist_sq=None))]
    fn eval_pairs<'py>(
        &self,
        py: Python<'py>,
        n_atoms: usize,
        i: PyReadonlyArray1<'_, u32>,
        j: PyReadonlyArray1<'_, u32>,
        disp: PyReadonlyArray2<'_, NpF>,
        dist_sq: Option<PyReadonlyArray1<'_, NpF>>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        check_nx3(&disp, "disp")?;
        let d2 = match dist_sq.as_ref() {
            Some(a) => Some(a.as_slice()?),
            None => None,
        };
        let (e, f) = self
            .inner
            .eval_pairs(n_atoms, i.as_slice()?, j.as_slice()?, disp.as_array(), d2)
            .map_err(PyValueError::new_err)?;
        Ok((e, f.into_pyarray(py)))
    }
}
