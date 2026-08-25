"""ForceField + Frame MD — the one driver over the Rust integrators.

User-facing spelling: ``molpy.md.MD``. Frame topology is compiled per
:meth:`MD.run`; the Rust ``VelocityVerlet`` owns the neighbour loop (the
``VerletSkin`` rebuild policy and pair feeding) — Python never does pair
bookkeeping.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._lib import NeighborList, Potentials, VerletSkin
from .._lib import md as _md

_NEIGHBOR_DEFAULTS = {
    "cutoff": None,
    "skin": 2.0,
    "every": 1,
    "delay": 0,
    "check": True,
}


def _stack_xyz(atoms: object) -> NDArray[np.float64]:
    return np.stack(
        [np.asarray(atoms["x"], dtype=np.float64),
         np.asarray(atoms["y"], dtype=np.float64),
         np.asarray(atoms["z"], dtype=np.float64)],
        axis=1,
    )


def _stack_vel(atoms: object, shape: tuple[int, int]) -> NDArray[np.float64]:
    if all(name in atoms for name in ("vx", "vy", "vz")):
        return np.stack(
            [np.asarray(atoms["vx"], dtype=np.float64),
             np.asarray(atoms["vy"], dtype=np.float64),
             np.asarray(atoms["vz"], dtype=np.float64)],
            axis=1,
        )
    return np.zeros(shape, dtype=np.float64)


class MD:
    """Run a ``ForceField`` (or pre-built potential) over a ``Frame``::

        from molpy import md

        driver = md.MD()
        driver.set_forcefield(ff)
        driver.set_neighbors(cutoff=7.5, skin=2.0)
        state = driver.run(frame, 1000, dt=1.0, kb=molpy.UnitPreset("real").boltzmann())

    The driver is unit-agnostic: nothing here converts units.
    Force-field parameters must already be consistent with ``dt`` / ``mass``
    / velocities. ``thermo=N`` requires an explicit ``kb=``.

    Precision is ``MD(dtype=np.float64)`` only; float32 / mixed belong in
    the Rust integrator.

    After :meth:`run`, :attr:`num_edges` / :attr:`rebuild_count` /
    :attr:`ago` report the run's neighbour state (``None`` when the run had
    no neighbour list) and :attr:`thermo` holds the sampled observables.
    """

    def __init__(self, *, dtype=np.float64) -> None:
        dt = np.dtype(dtype)
        if dt != np.dtype(np.float64):
            raise ValueError(
                "MD(dtype=) currently accepts only numpy.float64; "
                "float32 / mixed belong in the Rust integrator"
            )
        self.dtype = dt
        self._forcefield: object | None = None
        self._potential: object | None = None
        self._skin: VerletSkin | None = None
        self._skin_used = False
        self._neighbor_config: dict | None = None
        self._integrator: _md.VelocityVerlet | None = None
        self.thermo: list[dict[str, float]] = []

    # -- configuration ------------------------------------------------------

    def set_forcefield(self, forcefield: object) -> MD:
        """Attach a ``ForceField``; each :meth:`run` compiles it per frame."""
        if not hasattr(forcefield, "to_potentials"):
            raise TypeError(
                f"set_forcefield expects a ForceField, got {type(forcefield).__name__}"
            )
        self._forcefield = forcefield
        self._potential = None
        return self

    def set_potential(self, potential: object) -> MD:
        """Attach a pre-built potential (advanced; replaces :meth:`set_forcefield`).

        Accepts a compiled ``Potentials`` collection, an ``LJCut``, or a
        ``Potential`` subclass instance. The caller owns units (apply
        units) — and, when skipping :meth:`set_neighbors`, neighbor correctness too:
        compiled ``Potentials`` evaluate exactly the topology (any ``pairs``
        block included) they were bound to; nothing is rebuilt as coordinates
        move. A ``Potentials`` collection is **moved** into the run's
        integrator — one run per attach.
        """
        if isinstance(potential, Potentials) and len(potential) == 0:
            raise ValueError(
                "Potentials is still deferred (len==0); compile with "
                "ff.to_potentials(frame) first"
            )
        self._potential = potential
        self._forcefield = None
        return self

    def set_neighbors(
        self,
        neighbors: VerletSkin | None = None,
        *,
        cutoff: float | None = None,
        skin: float | None = None,
        every: int | None = None,
        delay: int | None = None,
        check: bool | None = None,
    ) -> MD:
        """Configure the neighbour list the run's integrator will own.

        Two mutually exclusive forms:

        * **Prebuilt** — pass a ``VerletSkin`` for full control of the search
          engine and rebuild policy. It is **moved** into the next run's
          integrator (single-shot); its force ``cutoff`` must cover the pair
          style's own cutoff.
        * **Kwargs** — each run builds a fresh
          ``VerletSkin(NeighborList(cutoff + skin), cutoff, …)`` over the
          frame. ``cutoff`` is the force cutoff in Å (default: derived from
          the pair style's ``cutoff`` param); ``skin`` is the Verlet buffer
          in Å (default 2.0); ``every`` / ``delay`` / ``check`` mirror the
          ``VerletSkin`` rebuild policy (defaults 1 / 0 / True).

        Only consulted when the run has a nonbond term (force-field pair
        styles, or a :meth:`set_potential` potential that needs pairs fed);
        bonded-only force fields need no neighbour configuration.
        """
        kwargs_given = any(v is not None for v in (cutoff, skin, every, delay, check))
        if neighbors is not None:
            if kwargs_given:
                raise ValueError(
                    "set_neighbors takes a prebuilt VerletSkin OR kwargs, not both"
                )
            if not isinstance(neighbors, VerletSkin):
                raise TypeError(
                    f"neighbors must be a VerletSkin, got {type(neighbors).__name__}"
                )
            self._skin = neighbors
            self._skin_used = False
            self._neighbor_config = None
        else:
            if cutoff is not None and float(cutoff) <= 0.0:
                raise ValueError("cutoff must be > 0 Å")
            if skin is not None and float(skin) < 0.0:
                raise ValueError("skin must be >= 0 Å")
            config = dict(_NEIGHBOR_DEFAULTS)
            if cutoff is not None:
                config["cutoff"] = float(cutoff)
            if skin is not None:
                config["skin"] = float(skin)
            if every is not None:
                config["every"] = int(every)
            if delay is not None:
                config["delay"] = int(delay)
            if check is not None:
                config["check"] = bool(check)
            self._skin = None
            self._neighbor_config = config
        return self

    # -- run ----------------------------------------------------------------

    @property
    def num_edges(self) -> int | None:
        """Pair edges in the last run's list (``None`` without neighbors)."""
        return None if self._integrator is None else self._integrator.num_edges

    @property
    def rebuild_count(self) -> int | None:
        """Neighbour rebuilds during the last run (``None`` without neighbors)."""
        return None if self._integrator is None else self._integrator.rebuild_count

    @property
    def ago(self) -> int | None:
        """Updates since the last rebuild (``None`` without neighbors)."""
        return None if self._integrator is None else self._integrator.ago

    def _pair_kernels(self, frame: object, config: dict) -> tuple[list, float]:
        """``pair:lj/cut`` styles → ``LJCut`` kernels + the force cutoff (Å).

        Parameters are taken exactly as the force field states them — no unit
        conversion. Per style: exactly one ``(epsilon, sigma)`` set may be
        in use by this frame; the force cutoff is the first of
        ``set_neighbors(cutoff=…)`` > the style-level ``cutoff`` param > the
        per-type maximum > a prebuilt skin's own cutoff.
        """
        ff = self._forcefield
        pair_names = [
            cat_name.split(":", 1)[1]
            for cat_name in ff.style_names()
            if cat_name.split(":", 1)[0] == "pair"
        ]
        if not pair_names:
            return [], 0.0
        mini = ff.subset(frame)
        kernels: list = []
        cutoffs: list[float] = []
        for name in pair_names:
            if name != "lj/cut":
                raise NotImplementedError(
                    f"the MD driver derives nonbond kernels for pair style "
                    f"'lj/cut' only; got '{name}'. Precompile Potentials over "
                    "a consumer-built pairs block and use set_potential."
                )
            rows = mini.types("pair", name)
            if not rows:
                raise ValueError(
                    f"pair style '{name}' has no types used by this frame; "
                    "check the atoms block's 'type' column"
                )
            try:
                distinct = {
                    (float(params["epsilon"]), float(params["sigma"]))
                    for _, params in rows
                }
            except KeyError as exc:
                raise ValueError(
                    f"pair style '{name}' types must carry 'epsilon' and 'sigma'"
                ) from exc
            if len(distinct) != 1:
                raise NotImplementedError(
                    f"pair style '{name}' carries {len(distinct)} distinct "
                    "(epsilon, sigma) sets for this frame; the MD driver "
                    "builds one uniform kernel (single atom-type parameter "
                    "set only — no per-type mixing yet). Precompile "
                    "Potentials over a consumer-built pairs block and use "
                    "set_potential."
                )
            cutoff = config["cutoff"]
            if cutoff is None:
                cutoff = dict(ff.style_params("pair", name)).get("cutoff")
            if cutoff is None:
                type_cutoffs = [params.get("cutoff") for _, params in rows]
                if type_cutoffs and all(c is not None for c in type_cutoffs):
                    cutoff = max(type_cutoffs)
            if cutoff is None and self._skin is not None:
                cutoff = float(self._skin.cutoff)
            if cutoff is None:
                raise ValueError(
                    f"cannot derive a force cutoff for pair style '{name}': "
                    "no 'cutoff' param at style or type level. Call "
                    "set_neighbors(cutoff=<Å>, skin=<Å>) before run."
                )
            ((epsilon, sigma),) = distinct
            kernels.append(_md.LJCut(epsilon, sigma, float(cutoff)))
            cutoffs.append(float(cutoff))
        return kernels, (max(cutoffs) if cutoffs else 0.0)

    def _build_skin(
        self, frame: object, pos: NDArray[np.float64], force_cutoff: float | None
    ) -> VerletSkin:
        """One fresh ``VerletSkin`` for this run (prebuilt skins pass through)."""
        if self._skin is not None:
            skin = self._skin
            if force_cutoff is not None and force_cutoff > float(skin.cutoff) + 1e-12:
                raise ValueError(
                    f"prebuilt VerletSkin cutoff {skin.cutoff} Å is smaller "
                    f"than the pair-style force cutoff {force_cutoff} Å; "
                    "pairs would be silently missed"
                )
            # Moved into the integrator below: single-shot by construction.
            self._skin = None
            self._skin_used = True
            return skin
        if self._skin_used and self._neighbor_config is None:
            raise ValueError(
                "the prebuilt VerletSkin from set_neighbors was moved into a "
                "previous run's integrator; call set_neighbors again with a "
                "fresh VerletSkin or kwargs"
            )
        config = self._neighbor_config or dict(_NEIGHBOR_DEFAULTS)
        cutoff = config["cutoff"] if config["cutoff"] is not None else force_cutoff
        if not cutoff:
            raise ValueError(
                "cannot derive a force cutoff for the neighbour list. Call "
                "set_neighbors(cutoff=<Å>, skin=<Å>) before run."
            )
        box = getattr(frame, "box", None)
        if box is None:
            raise ValueError(
                "a neighbour-driven run needs periodic minimum images: set frame.box"
            )
        skin_width = float(config["skin"])
        return VerletSkin(
            NeighborList(float(cutoff) + skin_width),
            float(cutoff),
            pos,
            box,
            skin=skin_width,
            every=int(config["every"]),
            delay=int(config["delay"]),
            check=bool(config["check"]),
        )

    def _assemble(
        self, frame: object, dt: float, pos: NDArray[np.float64], mass: NDArray[np.float64]
    ) -> _md.VelocityVerlet:
        """Wire one run. This single step does exactly:

        1. **Compile the potential.** ``set_forcefield`` path: the non-pair
           styles compile once through ``to_potentials(frame)``
           (coordinate-independent topology); every ``pair:lj/cut`` style
           becomes an ``LJCut`` kernel (:meth:`_pair_kernels`) **pushed into
           the same collection**. ``set_potential`` path: adopt the attached
           potential as-is (caller owns units).
        2. **Build the neighbour state.** With a nonbond term: a fresh
           ``VerletSkin(NeighborList(rc + skin), rc, pos, frame.box, …)``
           from the :meth:`set_neighbors` kwargs (defaults otherwise), or
           the prebuilt skin (single-shot; cutoff checked against the force
           cutoff). Bonded-only runs carry no neighbour state.
        3. **Construct the integrator.** ``VelocityVerlet(dt,
           potential=…, neighbors=…, mass=…)`` — potential, skin and the
           loop's pair feeding are **moved** into Rust; nothing to bookkeep
           in Python.
        """
        if self._forcefield is not None:
            ff = self._forcefield
            config = self._neighbor_config or dict(_NEIGHBOR_DEFAULTS)
            kernels, force_cutoff = self._pair_kernels(frame, config)
            if kernels:
                if "bonds" in frame and frame["bonds"].nrows > 0:
                    raise ValueError(
                        "pair-style MD over a bonded topology needs "
                        "special_bonds exclusions, which the neighbour-driven "
                        "pair path does not apply yet: every pair within the "
                        "cutoff would interact, double-counting bonded "
                        "1-2/1-3/1-4 neighbours. Precompile Potentials over a "
                        "consumer-built pairs block and use set_potential for "
                        "molecular systems."
                    )
                mini = ff.subset(frame)
                for cat_name in ff.style_names():
                    category, _, name = cat_name.partition(":")
                    if category == "pair":
                        mini.remove_style("pair", name)
                pots = (
                    mini.to_potentials(frame) if mini.style_names() else Potentials()
                )
                for kernel in kernels:
                    pots.push(kernel)
                neighbors = self._build_skin(frame, pos, force_cutoff)
            else:
                pots = ff.to_potentials(frame)
                if len(pots) == 0:
                    raise ValueError(
                        "forcefield.to_potentials(frame) produced empty Potentials"
                    )
                neighbors = None
        elif self._potential is not None:
            pots = self._potential
            if isinstance(pots, Potentials):
                self._potential = None  # moved into the integrator below
            neighbors = None
            if self._skin is not None or self._neighbor_config is not None:
                neighbors = self._build_skin(frame, pos, None)
        else:
            raise RuntimeError("set_forcefield or set_potential before run")
        return _md.VelocityVerlet(
            float(dt), potential=pots, neighbors=neighbors, mass=mass
        )

    def run(
        self,
        frame: object,
        n_steps: int,
        *,
        dt: float,
        mass: NDArray[np.floating] | float | None = None,
        temperature: float | None = None,
        seed: int = 0,
        thermo: int | None = None,
        kb: float | None = None,
    ) -> _md.MDState:
        """Integrate ``n_steps`` NVE steps; write pos and vel back to ``frame``.

        Assembly is per call (:meth:`_assemble`) — a driver configured via
        :meth:`set_forcefield` runs again and again. ``mass=`` overrides
        ``frame["atoms"]["mass"]`` (a scalar broadcasts). ``temperature=``
        draws initial velocities through ``MaxwellBoltzmann(temperature,
        seed=seed)`` (LAMMPS ``velocity create``; this helper fixes MD units:
        K and amu → Å/fs); otherwise the frame's ``vx``/``vy``/``vz`` (or
        zeros) are used. ``thermo=N`` samples ``step`` / ``pe`` / ``ke`` /
        ``etotal`` / ``temp`` every N steps into :attr:`thermo` (the ``temp``
        column uses ``kb=``). Returns the final ``MDState``.
        """
        if self._forcefield is None and self._potential is None:
            raise RuntimeError("set_forcefield or set_potential before run")
        if (thermo is not None or temperature is not None) and kb is None:
            raise ValueError("MD.run(thermo=...) / temperature= requires an explicit kb=")
        atoms = frame["atoms"]
        pos = _stack_xyz(atoms)
        if mass is None:
            if "mass" not in atoms:
                raise ValueError(
                    "frame['atoms'] must carry a mass column, or pass mass="
                )
            mass_arr = np.asarray(atoms["mass"], dtype=np.float64)
        else:
            mass_arr = np.atleast_1d(np.asarray(mass, dtype=np.float64))
        if mass_arr.size == 1:
            mass_arr = np.full(pos.shape[0], float(mass_arr[0]))
        if temperature is not None:
            vel = _md.MaxwellBoltzmann(
                float(kb) * float(temperature), seed=int(seed)
            ).velocities(pos, mass_arr)
        else:
            vel = _stack_vel(atoms, pos.shape)

        integrator = self._assemble(frame, dt, pos, mass_arr)
        self._integrator = integrator
        state = integrator.initial(pos, vel)
        self.thermo = []
        if thermo is None:
            if int(n_steps) > 0:
                state = integrator.advance_n(state, int(n_steps))
        else:
            interval = int(thermo)
            if interval < 1:
                raise ValueError(f"thermo must be >= 1, got {thermo}")
            kb = float(kb)
            dof = max(1, 3 * pos.shape[0] - int(integrator.removed_dof))
            mass_col = mass_arr.reshape(-1, 1)

            def record(step: int, state: _md.MDState) -> None:
                ke = float(0.5 * (mass_col * state.vel * state.vel).sum())
                pe = float(state.energy)
                self.thermo.append(
                    {
                        "step": step,
                        "pe": pe,
                        "ke": ke,
                        "etotal": pe + ke,
                        "temp": 2.0 * ke / (dof * kb),
                    }
                )

            record(0, state)
            done = 0
            while done < int(n_steps):
                n = min(interval, int(n_steps) - done)
                state = integrator.advance_n(state, n)
                done += n
                record(done, state)

        atoms["x"] = state.pos[:, 0]
        atoms["y"] = state.pos[:, 1]
        atoms["z"] = state.pos[:, 2]
        atoms["vx"] = state.vel[:, 0]
        atoms["vy"] = state.vel[:, 1]
        atoms["vz"] = state.vel[:, 2]
        return state
