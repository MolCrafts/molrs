# Transport Kernels

Technical reference for the electrolyte-transport compute kernels in
`molrs-compute`: the Onsager collective-displacement correlation, the
current-autocorrelation Green–Kubo conductivity, and the pair-survival
(persistence) correlation. They are ports of the
[*tame*](https://github.com/Roy-Kid/tame) trajectory-analysis recipes.

Like the [dielectric](trajectory-analysis.md) kernels, these are **array-based
free functions**, not `Compute`-trait implementations: the caller assembles the
per-frame collective quantities (summed displacements, currents, per-species
coordinates) in the host language and passes plain arrays in. The Rust layer
performs the windowed correlation / survival accounting. They are reached through
the `molrs.compute.transport` namespace from Python.

For the physical background and worked, from-principles derivations, see the
MolPy guides [Diffusion & Ionic Transport](https://molpy.molcrafts.org/compute/transport/)
and [Pair Persistence](https://molpy.molcrafts.org/compute/persistence/).

## Conventions

- Displacement over lag $\tau$: $\Delta\mathbf{P}(\tau)=\mathbf{P}(t+\tau)-\mathbf{P}(t)$.
- $\langle\cdots\rangle_t$ — average over all time origins $t$; for lag $\tau$
  there are $N_\text{frames}-\tau$ valid origins.
- Units (LAMMPS *real*): length Å, time ps, charge $e$, volume Å³, temperature K.
- All kernels take `max_correlation_time` in **frames** (clamped to
  $N_\text{frames}-1$) and return arrays of length `max_correlation_time + 1`.

---

## Onsager — `onsager_correlation`

Windowed cross-correlation of two species' collective coordinates:

$$
L_{ij}(\tau) = \big\langle\,\Delta\mathbf{P}_i(\tau)\cdot\Delta\mathbf{P}_j(\tau)\,\big\rangle_t
= \frac{1}{N-\tau}\sum_{t=0}^{N-1-\tau}\sum_{d}\big(P_i[t{+}\tau,d]-P_i[t,d]\big)\big(P_j[t{+}\tau,d]-P_j[t,d]\big).
$$

The collective coordinate $\mathbf{P}_s(t)=\sum_{a\in s}\mathbf{r}_a(t)$ must be
assembled and **unwrapped** by the caller. The diagonal ($i=j$) is the collective
MSD of species $i$. The Onsager phenomenological coefficient is the long-time
slope, $\Omega_{ij}=\lim_{\tau\to\infty}L_{ij}(\tau)/(6\,k_BTVN_A\,\tau)$ (taken
by the caller).

| Argument | Type | Meaning |
|----------|------|---------|
| `p_i`, `p_j` | `Array2<f64>` `(n_frames, 3)` | collective coordinates (unwrapped); same frame count |
| `dt` | `f64` | frame spacing (> 0); sets `lag_times` |
| `max_correlation_time` | `usize` | longest lag in frames |

**Returns** `OnsagerResult { lag_times, correlation }`, each length
`max_lag + 1`. No volume normalization is applied (faithful to *tame*).

**Errors** `DimensionMismatch` (non-`(_,3)` or mismatched frame counts),
`EmptyInput` (< 2 frames), `NonFinite`, `OutOfRange` (`dt ≤ 0`).

```python
from molrs.compute.transport import Onsager
res = Onsager.correlation(P_i, P_j, dt=0.01, max_correlation_time=2000)
res["lag_times"], res["correlation"]
```

---

## JACF — Green–Kubo conductivity

Green–Kubo DC ionic conductivity from the charge-current autocorrelation:

$$
C(\tau) = \big\langle\mathbf{J}(0)\cdot\mathbf{J}(\tau)\big\rangle_t,
\qquad
\sigma = \frac{1}{3\,V k_B T}\int_0^{\tau_\max} C(\tau)\,d\tau,
$$

with the collective charge current $\mathbf{J}(t)=\sum_a q_a\mathbf{v}_a(t)$
assembled by the caller. The autocorrelation is the unbiased windowed estimator
$C(\tau)=\tfrac{1}{N-\tau}\sum_t\mathbf{J}(t)\cdot\mathbf{J}(t+\tau)$; the running
integral is trapezoidal.

!!! note "API — compose the raw-compute classes"

    The bundled `transport.Jacf.green_kubo_conductivity` kernel was removed in
    `compute-fit-03-cleanup`. Build the conductivity from the raw-compute
    classes instead: `molrs.compute.transport.GreenKuboConductivity` produces the raw current ACF
    $C(\tau)$, `molrs.compute.fitting.CumulativeTrapezoid` forms $\int_0^\tau C\,d\tau'$, and a
    `1/(3·V·k_B·T)` MD→SI prefactor (folding in $e^2$, Å→m, ps→s — the same
    factors as the Einstein–Helfand route, with the Green–Kubo $1/3$ replacing
    the Einstein $1/6$) converts the result to S·m⁻¹. This mirrors the
    Einstein–Helfand route documented with the
    [dielectric kernels](trajectory-analysis.md).

---

## Persist — `pair_survival_tcf`

Pair-survival (residence-time) correlation between two species:

$$
C(\tau) = \Big\langle\,\frac{1}{N_i}\sum_i\sum_j S_{ij}(t,\,t+\tau)\,\Big\rangle_t,
$$

where the survival indicator $S_{ij}\in\{0,1\}$ depends on the method. A pair is
*born* at $t$ when its minimum-image distance is within the inner cutoff $r_0$;
survival to lag $\tau$ is judged against the outer cutoff $r_1\ge r_0$:

| `method` | $S_{ij}(t,t+\tau)=1$ iff |
|----------|--------------------------|
| `continuous` (`cr`, `rf`) | within $r_1$ at **every** frame in $[t, t+\tau]$ |
| `intermittent` (`imm`) | within $r_1$ at frame $t+\tau$ (gaps allowed) |
| `ssp` | born within $r_0$ **and** within $r_1$ at every frame since |

Distances use the orthorhombic minimum-image convention
$d \mathrel{-}= \mathrm{round}(d/L)\,L$ per axis (matching *tame*'s
`tpairsurvive`). $C(0)$ is the mean coordination number.

| Argument | Type | Meaning |
|----------|------|---------|
| `coords_i`, `coords_j` | `Array3<f64>` `(n_frames, n, 3)` | per-species coordinates (wrapped), same frame count |
| `box_lengths` | `Array2<f64>` `(n_frames, 3)` | per-frame orthorhombic edge lengths (≤ 0 disables an axis) |
| `r0`, `r1` | `f64` | inner/outer cutoff, Å (`r0 > 0`, `r1 ≥ r0`) |
| `method` | `&str` / `SurvivalMethod` | survival criterion |
| `dt` | `f64` | frame spacing, ps (> 0) |
| `max_correlation_time` | `usize` | longest lag in frames |
| `exclude_self` | `bool` | drop the `i == j` self-pair when both species are identical |

**Returns** `PersistResult { lag_times, correlation }`, each length
`max_lag + 1`.

**Errors** `DimensionMismatch` (wrong rank/shape or mismatched frame counts),
`EmptyInput` (< 2 frames or an empty species), `OutOfRange`
(`r0 ≤ 0`, `r1 < r0`, `dt ≤ 0`, or unknown `method`).

```python
from molrs.compute.transport import Persist
res = Persist.pair_survival_tcf(coords_i, coords_j, box_lengths,
                                r0=3.0, r1=4.0, method="ssp",
                                dt=0.01, max_correlation_time=3000,
                                exclude_self=False)
res["correlation"]   # C(tau); C(0) = mean coordination number
```

---

## References

- L. Onsager, *Phys. Rev.* **37**, 405 (1931) — phenomenological coefficients.
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids* — Green–Kubo current
  correlation and conductivity.
- D. C. Rapaport, *Mol. Phys.* **50**, 1151 (1983) — continuous vs intermittent
  survival.
- A. Luzar, *J. Chem. Phys.* **113**, 10663 (2000); D. Laage, J. T. Hynes,
  *J. Phys. Chem. B* **112**, 14230 (2008) — residence-time / stable-states
  definitions.
- H. Gudla, Y. Shao et al., *J. Phys. Chem. Lett.* **12**, 8460 (2021) — pairing
  diffusion via persistence-weighted correlations.
