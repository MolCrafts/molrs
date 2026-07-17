# Third-Party Attributions & Licenses

molrs is distributed under the [BSD 3-Clause License](../LICENSE)
(Copyright © 2025, MolCrafts). It ports or adapts code and reference data from
the projects listed here. **This is the single, authoritative attribution file
for molrs.**

## Ported / adapted — permissive licenses

| Project | SPDX | Copyright | molrs modules | Upstream |
|---|---|---|---|---|
| **RDKit** | `BSD-3-Clause` | © 2006–2015 Rational Discovery LLC, Greg Landrum, Julie Penzotti and others | `core/chem/aromaticity.rs`, `core/chem/gasteiger.rs`, `optimize/lbfgs.rs`, `conformer/distgeom/**`, `conformer/etkdg/**`, `ff/mmff/**`, `ff/constants.rs` | [rdkit/rdkit](https://github.com/rdkit/rdkit) |
| **freud** | `BSD-3-Clause` | © 2010–2026 The Regents of the University of Michigan | `compute/density/**`, `compute/diffraction/**`, `compute/environment/**`, `compute/order/**`, `compute/pmft/**`, `compute/rdf/**`, `compute/msd/**`, `core/spatial/neighbors/{aabb,filter,periodic_buffer,mod,query}.rs`, `core/math/{wigner3j,spherical_harmonics,diagonalize,mod}.rs` | [glotzerlab/freud](https://github.com/glotzerlab/freud) |
| **voro++** | `BSD-3-Clause-LBNL` | © 2008 The Regents of the University of California, through Lawrence Berkeley National Laboratory (Chris Rycroft) | `compute/voronoi/{radical,cell,mod}.rs` (radical/Laguerre tessellation) | [chr1shr/voro](https://github.com/chr1shr/voro) |
| **tame** | `BSD-3-Clause` | © Yunqi Shao | `compute/jacf.rs`, `compute/onsager.rs`, `compute/persist.rs`, `molrs-python/src/transport.rs` — Green–Kubo / Onsager / pair-persistence recipes | [yqshao-archive/tame](https://github.com/yqshao-archive/tame) (archived) |

> `BSD-3-Clause-LBNL` adds a grant-back clause for enhancements; keep the LBNL
> copyright notice intact. It remains compatible with BSD-3 redistribution.
>
> `tame` declares `license = "BSD-3-Clause"` in its `pyproject.toml` (it ships no
> standalone `LICENSE` file). It is an archived third-party project by Yunqi Shao.

## Bundled data files

Parameter data is no longer bundled as *text*: since `chem-perceive-14` every
force-field table molrs ships is typed Rust under `molrs/src/ff/params/`, and the
XML it used to be parsed from is deleted. The origin and licence of the numbers
are unchanged by that — they travel with the numbers, so the rows below name the
tables that hold them now.

| File | Origin | License |
|---|---|---|
| `molrs/src/ff/params/oplsaa.rs` (was `molrs/data/oplsaa.xml`) | OPLS-AA atom types & typing definitions from **foyer** | `MIT` (foyer) |
| `molrs/src/ff/params/mmff.rs` (was `molrs/src/ff/mmff/tables.rs` + `molrs/data/mmff94{,s}.xml`; workspace-root `data/` copies remain) | ported from **RDKit** `Code/ForceField/MMFF/Params.cpp` (MMFF94/94s tables); the style skeleton previously routed through `scripts/mmff_to_xml.py` | `BSD-3-Clause` (RDKit); MMFF94 parameters © Merck / T. A. Halgren, *J. Comput. Chem.* **17**, 490 (1996) |
| `tests-data/` (fetched at build via `scripts/fetch-test-data.sh`) | **chemfiles** integration-test fixtures (mirrored at MolCrafts/tests-data) | `CC0-1.0` (public domain) |

## Formula / method references (papers, no code copied)

Standard results cited in doc-comments, implemented independently: van Hove
*Phys. Rev.* **95**, 249 (1954) (`compute/van_hove.rs`); Racah/Edmonds Wigner-3j
(`core/math/wigner3j.rs`); Press et al. *Numerical Recipes*
(`core/math/spherical_harmonics.rs`, `diagonalize.rs`); Tang–Toennies
DOI 10.1063/1.447150 and Thole DOI 10.1016/0301-0104(81)85176-2 as emitted by
**paduagroup/clandpol** (`ff/potential/pair/{tang_toennies,thole}.rs`); NIST
CODATA constants (`core/units/constants.rs`).

## Specifications implemented (cite, not license)

SMILES/SMARTS in `io/smiles/**` implement the **Daylight** theory manual
(© Daylight C.I.S. — proprietary documentation, cited only) and Weininger,
*J. Chem. Inf. Comput. Sci.* **28**, 31 (1988).
