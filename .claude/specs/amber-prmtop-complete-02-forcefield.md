---
title: AMBER prmtop force-field reader — registered pair split, uniform 1-4 divisors, full-ICO LJ decode
status: code-complete
created: 2026-09-04
depends_on: [amber-prmtop-complete-01-structure]
---

# AMBER prmtop force-field reader — registered pair split, uniform 1-4 divisors, full-ICO LJ decode

## Summary

Phase 02 of the `amber-prmtop-complete` chain makes the AMBER prmtop *force-field* reader (`molrs/src/ff/forcefield/readers/prmtop.rs`) produce a force field that molrs can actually compile and that states every constant it means. Today the reader declares the pair style `lj/cut/coul/long` with only `cutoff_lj`/`cutoff_coul`; that style name is not in the `KernelRegistry`, so `ForceField::to_potentials` cannot build the non-bonded terms of *any* prmtop-derived force field, and the declaration silently omits `coulomb` and `dielectric`. This phase replaces it with the sibling precedent's registered split — `lj/cut` + `coul/cut` — carrying explicit named constants that live in **one** crate-internal home (a new hand-maintained `ff::params::amber` module, shared with the GAFF typifier force field instead of a second private copy), tightens the two silent decodes that surround it (a first-non-zero SCEE/SCNB pick that hides non-uniform 1-4 divisors, and a diagonal-only LJ decode that never looks at the off-diagonal NBFIX entries), turns four currently-silent unsupported inputs into named errors (12-6-4, multi-term impropers, 10-12, and off-diagonal LJ pairs that deviate from Lorentz–Berthelot), and pins the LAMMPS `pair_coeff` text the production PEO pipeline consumes so the fix is provably invisible to that consumer. No new public symbol is added.

## Domain basis

**1-4 scaling (SCEE / SCNB).** AMBER stores 1-4 scaling as per-dihedral-type **divisors** in `SCEE_SCALE_FACTOR` / `SCNB_SCALE_FACTOR`: `E_14^coul = E^coul / SCEE`, `E_14^vdw = E^vdw / SCNB`. molrs `SpecialBonds` stores multiplicative weights, so `coul[2] = 1/SCEE`, `lj[2] = 1/SCNB`. When the sections are absent (pre-Amber-11 topologies) the format defaults are **SCEE = 1.2, SCNB = 2.0** — the GAFF / ff14SB values; GLYCAM uses 1.0 / 1.0, and mixed-force-field topologies therefore carry genuinely non-uniform columns which a scalar `SpecialBonds` cannot represent. Reference: <https://ambermd.org/FileFormats.php> (`SCEE_SCALE_FACTOR`, `SCNB_SCALE_FACTOR`).

**Dihedral pointer semantics.** In `DIHEDRALS_{INC,WITHOUT}_HYDROGEN` (5 integers per torsion), a **negative 3rd** pointer means the 1-4 pair is *suppressed* (no 1-4 interaction is computed for that torsion — ring closures, multi-term duplicates), and a **negative 4th** pointer marks an improper. A negative `PN` in `DIHEDRAL_PERIODICITY` means the *next* `PK/PN/PHASE` triple continues the same torsion (multi-term Fourier). Only type ids reached by **non-suppressed proper** torsions can contribute a 1-4 interaction, so only those constrain SCEE/SCNB uniformity.

**Coulomb constant (new in this phase).** AMBER's `CHARGE` section stores `q · 18.2223`; the structure reader de-scales it (`CHARGE_CONVERSION_FACTOR = 18.2223`, `molrs/src/io/data/prmtop.rs:48`). The electrostatic constant implied by that factor is therefore **k = 18.2223² = 332.05221729 kcal·Å·mol⁻¹·e⁻²**, which is what sander/pmemd effectively evaluate; reproducing AMBER energies exactly requires this number, not CODATA. molrs's CODATA constant `COULOMB_REAL = 332.06371` (`molrs/src/core/units/constants.rs:40`, used by `readers/lammps.rs`) differs by **3.46e-5 relative** ((332.06371 − 332.05221729)/332.0637 = 3.461e-5) — a documented, expected cross-engine offset, not an error. Dielectric is vacuum, `D = 1` (`ff::constants::VACUUM_DIELECTRIC`). References: <https://ambermd.org/FileFormats.php>, <https://ambermd.org/Questions/units.html>, ParmEd `parmed/constants.py` (`AMBER_ELECTROSTATIC = 18.2223`).

**LJ decode (ICO).** The non-bonded index is `idx = ICO[NTYPES·(IAC(i) − 1) + IAC(j)]` (1-based Fortran); the 0-based self entry is `(NTYPES + 1)·(IAC − 1)`. With `E = A/r¹² − B/r⁶`: `r_min = (2A/B)^{1/6}`, `ε = B²/(4A)`, `σ = 2^{−1/6} r_min`. A **negative** ICO entry indexes the 10-12 H-bond tables (`HBOND_ACOEF`/`HBOND_BCOEF`) — unsupported. Cross terms: AMBER stores every `(i,j)` pair explicitly, so a topology may carry off-diagonal entries that are *not* the Lorentz–Berthelot combination (`σ_ij = (σ_i + σ_j)/2`, `ε_ij = √(ε_i ε_j)`) — NBFIX / 1-4-specific pairs. Detection threshold is **relative 1e-6**; the f64 round-trip noise of `A/B → σ,ε → A/B` measured on the exact-LB fixture is ~7.5e-9, i.e. ~130× below the threshold, so the test discriminates cleanly. molrs has no kernel that reads a per-pair LJ override — `pair_lj_cut_ctor` keys `type_params` by a single atom-type name and always mixes (`ff/potential/pair/lj_cut.rs:342-387`) — so a deviating cross entry cannot be represented and is **refused**, not stored.

**12-6-4.** `LENNARD_JONES_CCOEF` marks the Li & Merz polarizable 12-6-4 model, `E = A/r¹² − B/r⁶ − C/r⁴` — a different functional form with no molrs kernel. Li & Merz, *J. Chem. Theory Comput.* **10**, 289 (2014), DOI [10.1021/ct400751u](https://doi.org/10.1021/ct400751u). Refuse rather than silently drop `C`.

**Bonded forms (unchanged this phase).** AMBER stores `E = RK(r − r₀)²` and `E = TK(θ − θ₀)²` without the ½, so molrs's half-k kernels take `k = 2·RK` / `k = 2·TK`; `θ₀` is already radians in the prmtop. Dihedral `PK` is already divided by `IDIVF` by tleap and is **not** doubled (`PK[1 + cos(nφ − δ)]`, `δ` radians) — matching the `fourier`/`periodic` kernels.

**Cutoffs.** A prmtop carries **no** non-bonded cutoff: `cut` lives in the mdin, not the topology. The reader therefore supplies documented defaults (9 Å LJ, 10 Å Coulomb — today's inline literals, unchanged) as named constants.

## Design

**Scope.** All code changes are inside `molrs/src/ff/` (`forcefield/readers/prmtop.rs`, `forcefield/gaff.rs`, new `params/amber.rs`, one `pub mod` line in `params/mod.rs`). The two `molrs/src/io/data/` edits are **rustdoc comments only** (zero code, zero symbol, zero behaviour) and therefore do not make this a cross-layer spec; they exist because the Iron law (CLAUDE.md:40-52) forbids leaving the debt and the false claim unnamed at the sites this spec's constants depend on. Two harness notes (`.claude/notes/notes.md`, `.claude/notes/release.md`) gain one entry each.

**No new public symbol.** `AmberPrmtopFfReader` (the type + its `ForceFieldReader` impl) and the lossy path-only alias `read_amber_prmtop_ff` keep their exact signatures — the alias mirrors `readers/gromacs.rs:70` and stays. The rejected `with_cutoffs` builder is **not** introduced: it would have no in-tree caller (`read_amber_prmtop_ff` is path-only, no FFI surface asks for it), which Shape check #3 (CLAUDE.md:83-89, "only one in-tree call site → do not extract") forbids. Cutoffs stay named constants; the "undeclared cutoff → `Err`" idea is deferred (Out of scope).

**Pair styles — registered split (resolves architect 🔴 1).** `readers/prmtop.rs:409-411` currently does `def_pairstyle("lj/cut/coul/long", [cutoff_lj, cutoff_coul])`. That name is not registered (`registry.rs:150-172` registers `lj/cut`, `coul/cut` (`ParamSource::PerInstance`), and PME as `coul/long/pme`), so `to_potentials` cannot build it; and it omits `coulomb`/`dielectric`, which CLAUDE.md:304 makes an `Err`-class defect, never a silent default. Replaced by the sibling precedent `readers/lammps.rs::build_pairs` (:474-495), inline in `build_forcefield` (one call site):

- `lj/cut` with `[("cutoff", DEFAULT_CUTOFF_LJ)]`, carrying the per-type `epsilon`/`sigma`.
- `coul/cut` with `[("coulomb", AMBER_COULOMB), ("dielectric", VACUUM_DIELECTRIC), ("cutoff", DEFAULT_CUTOFF_COUL)]`; charges come from the frame, so it carries no pair types.

**AMBER constants — one home (resolves architect pass-2 🔴 1).** `AMBER_COULOMB = 332.052_217_29` already exists as a *private* const in `molrs/src/ff/forcefield/gaff.rs:71`, documented as **measured** from AmberTools `sander` single-points; the same file also holds `AMBER_LJ_14 = 0.5` / `AMBER_COUL_14 = 1/1.2` (`:59,:62`), while `readers/prmtop.rs:38-41` spells the same physics as divisors `DEFAULT_SCEE = 1.2` / `DEFAULT_SCNB = 2.0`. Re-typing any of them in the reader would give one force-field constant two homes and two provenance stories (`ff/constants.rs:1-5`; MMFF precedent relocated its constant into `ff::params::mmff`). The home must be **hand-maintained**: `ff::params::gaff` (and `gaff2`) are *emitted* by `scripts/gen_param_tables.py` ("DO NOT HAND-EDIT", checksummed in `molrs/src/ff/params/MANIFEST.sha256`), so a hand-written const there would be deleted by the next regeneration and would falsify the manifest meanwhile; `ff::constants` admits only "properties of the universe" by its own header; `params/mmff.rs` is the precedent precisely because it is a hand-maintained, non-emitted sibling. Therefore this phase adds a **new hand-maintained module `molrs/src/ff/params/amber.rs`** (`pub mod amber;` in `params/mod.rs`; **not** listed in `MANIFEST.sha256`, same status as `mmff.rs`), whose module rustdoc says what it is: the AMBER *file-format* constants that are neither `gaff.dat` rows nor properties of the universe, shared by every consumer of an AMBER-family topology (GAFF/GAFF2 typifier force fields, ff14SB/GLYCAM prmtops). It holds three `pub(crate)` consts:

- `AMBER_COULOMB: f64 = 332.052_217_29` — rustdoc merges both provenance paragraphs: it is `18.2223²` (the charge factor AMBER writes into `CHARGE`, `molrs/src/io/data/prmtop.rs:48` — cited by path, since ff must not `use` io, `architecture-rules.md:26-34`) **and** it is the value AmberTools `sander` single-points recover (the measurement recorded in today's `forcefield/gaff.rs:64-70` rustdoc — whose cited generator `scripts/gen_gaff_energy_oracle.py` does **not** exist in the tree; the merged rustdoc states the measurement without the dangling pointer, and the missing oracle script is logged in `.claude/notes/notes.md` as debt); the 3.46e-5 relative offset from `COULOMB_REAL` is recorded as the expected cross-engine difference.
- `AMBER_SCEE: f64 = 1.2`, `AMBER_SCNB: f64 = 2.0` — the format's 1-4 **divisors**. Their rustdoc states **both roles** explicitly: (i) the 1-4 parameter of the GAFF/GAFF2 typifier force fields (`forcefield/gaff.rs` derives its weights as `1.0 / AMBER_SCNB`, `1.0 / AMBER_SCEE`), and (ii) the prmtop reader's fallback when `SCEE_SCALE_FACTOR` / `SCNB_SCALE_FACTOR` are absent (the format's documented pre-Amber-11 default). A prmtop that *carries* the sections never touches the fallback, so a GLYCAM file (1.0/1.0) reads correctly; a future change to the GAFF role must therefore consciously keep the format-default role, and the rustdoc says so.

`forcefield/gaff.rs` deletes its three private consts and imports the `params::amber` ones; `readers/prmtop.rs` deletes `DEFAULT_SCEE` / `DEFAULT_SCNB` and imports the same. Nothing is re-typed, and the generated tables are untouched.

Reader-local constants in `readers/prmtop.rs`, each with rustdoc:

- `DEFAULT_CUTOFF_LJ: f64 = 9.0`, `DEFAULT_CUTOFF_COUL: f64 = 10.0` — today's literals, unchanged values; rustdoc states that a prmtop carries no cutoff (it lives in the mdin) and that these are reader defaults, not file data.
- `VACUUM_DIELECTRIC` is **reused** from `crate::ff::constants` (same import the LAMMPS reader uses).

*Downstream contract (hard constraint).* `writers/lammps.rs::write_pair_section` (:560-575) routes a `{lj/cut, coul/cut}` pair through `is_split_lj_coulomb` → `write_combined_lj_coulomb` (:614-652), which emits `pair_coeff <t> <t> <eps> <sigma>` from the `lj/cut` style alone (:649-651) — the same `write_pair_coeffs` path the current single-style branch (:577-590) takes. The self-type ε/σ values are bit-identical (same closed form, same A/B, same operation order), the emitted self-type **set** is unchanged, and self types are still `def_pairtype`'d in first-appearance-over-atoms order, so the `pair_coeff` block is byte-identical to today's, and with `skip_pair_style = true` the include still contains only `pair_coeff` lines. One line *does* change when `skip_pair_style = false`: the header becomes `pair_style lj/cut/coul/cut 9 10` instead of today's `pair_style lj/cut/coul/long 10 10` (today's combined name has no `cutoff` key, so `style_cutoff` at :673-675 misses and `pair_style_cutoffs` falls back to `DEFAULT_PAIR_CUTOFF = 10.0`). The production PEO pipeline uses `skip_pair_style = true` and supplies its own `pair_style lj/cut/coul/long 12.0 12.0`, so it is unaffected; the change is pinned by test either way.

**`uniform_divisor` (replaces `scale_divisor`, :164-167).** Today "pick the first positive value" silently collapses a mixed-force-field topology onto one arbitrary divisor. New private fn:

```
fn uniform_divisor(values: &[f64], used_tids: &[i64], flag: &str, default: f64) -> Result<f64, String>
```

`used_tids` is the set of 1-based dihedral type ids reached by **non-suppressed proper** torsions (3rd pointer ≥ 0), including multi-term continuation ids. Empty `values` (section absent) → `default` (`AMBER_SCEE` / `AMBER_SCNB`). A used id out of range, or a value ≤ 0, is an `Err` naming `flag` and the id. Two used values differing by more than **1e-6 relative** is an `Err` naming `flag` and **both** values. The result feeds `SpecialBonds { lj: [0,0,1/SCNB], coul: [0,0,1/SCEE] }` through the existing `set_special_bonds`.

**`decode_lj_types` (replaces the inline diagonal decode, :409-445; resolves architect pass-2 🔴 2).** Private fn walking the **full** ICO matrix over `1..=NTYPES` in both indices, returning a named alias (matching the sibling naming at `io/data/prmtop_tables.rs:207`, architect 🟢 4):

```
/// `(type_name, sigma_A, epsilon_kcal_per_mol)`
type LjSelfRow = (String, f64, f64);
fn decode_lj_types(...) -> Result<Vec<LjSelfRow>, String>
```

- Self terms → `style.def_pairtype(t, None, &[("epsilon", ..), ("sigma", ..)])`, one per **atom-type name** (names sharing an LJ index get the same params, preserving today's line set), emitted in today's first-appearance order.
- Every off-diagonal entry is compared against the Lorentz–Berthelot prediction from the two self terms. A deviation > **1e-6 relative** in either σ or ε is an **`Err`** naming both atom types — `"off-diagonal (NBFIX) Lennard-Jones pairs are not supported: c3-hc deviates from Lorentz-Berthelot"`. Cross pair types are **never emitted**: no kernel would read them (`pair_lj_cut_ctor` mixes unconditionally, `ff/potential/pair/lj_cut.rs:342-387`) and `write_pair_coeffs_dedup` (`writers/lammps.rs:704-725`) would print them as `pair_coeff i j` lines, so storing them would be a value written to disk and silently dropped from every energy — the exact split the Iron law forbids. LB-consistent cross entries (every LEaP-written GAFF/ff14SB topology) pass, so an ordinary prmtop writes exactly today's include.
- Refusals, all `Result<_, String>` (the module's existing error type): `LENNARD_JONES_CCOEF` present → `Err` naming **12-6-4**; negative ICO entry or any non-zero `HBOND_ACOEF`/`HBOND_BCOEF` → the existing message `"10-12 interactions are not supported"` (unchanged spelling); NBFIX deviation → the message above.

**Multi-term improper refusal.** `:375-388` silently keeps only the first Fourier term of a multi-term improper (`terms.values().next()`), discarding parameters. That becomes an `Err` naming the improper's dihedral type id.

**Both new fns are private and single-call-site by design** — extraction is justified by the second clause of CLAUDE.md:65 ("or when a unit test must target that unit"): each is a pure table decode with several refusal branches that cannot be reached through `read_str` alone without fabricating whole topologies, exactly as `io/data/prmtop_tables.rs` splits its `decode_*` functions. No public surface grows.

**Found rot, fixed here (Iron law).** (a) `ff/potential/kspace/pme.rs` documents the exclusions block as columns `"i"`/`"j"` at **:10** and **:857** while `pme_ctor` reads `atomi`/`atomj` at **:902** — the repo-wide pairs schema (`docs/interop.md`). Doc-only fix, both lines. (b) `io/data/prmtop.rs:47` claims AMBER stores `charges × √(332.0636) ≈ 18.2223`; 18.2223 is Amber's *own* factor and `18.2223² = 332.05221729`, so the stated provenance is off by the 3.46e-5 this spec documents. Doc-only correction, one line — this spec's Coulomb constant depends on that claim being right.

**Reuse decision** (resolving every advisory candidate):

- `crate::ff::constants::VACUUM_DIELECTRIC` — **reuse**. Same constant, same meaning, already imported by `readers/lammps.rs:490`.
- `forcefield/gaff.rs::{AMBER_COULOMB, AMBER_LJ_14, AMBER_COUL_14}` (:59-71) and `readers/prmtop.rs::{DEFAULT_SCEE, DEFAULT_SCNB}` (:38-41) — **generalize**: consolidated into `pub(crate)` `AMBER_COULOMB` / `AMBER_SCEE` / `AMBER_SCNB` in the new hand-maintained `molrs/src/ff/params/amber.rs` (the `ff::params::mmff::MMFF_ELE_STYLE` precedent: a non-emitted sibling of the generated tables); both consumers import, neither re-types. The two provenance paragraphs (measured from `sander`; equals `18.2223²`) are merged into the one rustdoc. `params/gaff.rs` was considered and rejected because it is generated (`MANIFEST.sha256`).
- `readers/lammps.rs::build_pairs` (:474-495) — **pattern**, not reuse. Its shape (registered `lj/cut` + `coul/cut`, constants stated on the style) is copied verbatim; the body is 6 lines against a different input and lives behind a different `Option`-free cutoff story, so extracting a shared helper would be a premature abstraction over two readers (CLAUDE.md:65). New code reads like the LAMMPS reader.
- `io/data/prmtop_tables.rs::decode_nonbond_params` (:214) — **new**. `ff` must not depend on `io` (`architecture-rules.md:26-34`), and the io function is diagonal-only, per-**atom** (not per-type), and returns `(atom_1based, σ, ε)` — the wrong shape for a type-table walk that also needs cross terms.
- σ/ε closed form duplicated at `io/data/prmtop_tables.rs:214` and here (architect 🟡 3) — **new (accepted duplication, option (a))**. `core` is *not* forced open by the layer rule (both layers may import `core`), so the honest reason is the threshold: three lines, one use per layer, and `decode_nonbond_params`'s only consumer is a Python export. Below CLAUDE.md:65's "inline until the second real use" bar; revisit at the third use. A rustdoc `Debt:` note at **both** sites names the duplication and the revisit trigger. `core` is not touched in this phase.
- `io/data/prmtop.rs::CHARGE_CONVERSION_FACTOR` (:48) — **new**. Same number, unreachable direction (`ff` ↛ `io`); `params::amber::AMBER_COULOMB`'s rustdoc points at that line.
- `writers/lammps.rs` `write_pair_section` / `is_split_lj_coulomb` / `write_combined_lj_coulomb` — **reuse**, unmodified: the design targets its existing `{lj/cut, coul/cut}` branch rather than teaching the writer a new name.
- `ForceField::def_pairtype(itom, Some(jtom), ..)` (`ff/forcefield/mod.rs:339`) — **not used**: a cross pair type has no kernel reader today, so emitting it would be a value with no consumer (architect pass-2 🔴 2). NBFIX support, if ever wanted, is a kernel + writer + reader spec of its own (Out of scope).
- `ForceField::{def_pairstyle, def_pairtype(t, None, ..), set_special_bonds}`, `SpecialBonds` — **reuse**, unchanged calls.

## Files to create or modify

- `molrs/src/ff/forcefield/readers/prmtop.rs` — pair-style split, `uniform_divisor`, `decode_lj_types` + alias, NBFIX / multi-term-improper / 12-6-4 refusals, cutoff consts, rustdoc (incl. `Debt:` note), `#[cfg(test)]` tests
- `molrs/src/ff/params/amber.rs` (new, hand-maintained, not in `MANIFEST.sha256`) — `pub(crate)` `AMBER_COULOMB` / `AMBER_SCEE` / `AMBER_SCNB` with the merged provenance rustdoc and the two-roles note
- `molrs/src/ff/params/mod.rs` — `pub mod amber;` plus one sentence in the module's provenance paragraph (`:1-16`) naming `amber.rs` as a hand-maintained sibling like `mmff.rs` / `clpol.rs` / `uff.rs`
- `molrs/src/ff/forcefield/gaff.rs` — delete the private `AMBER_COULOMB` / `AMBER_LJ_14` / `AMBER_COUL_14`; import from `params::amber` (weights derived as `1.0 / AMBER_SCNB`, `1.0 / AMBER_SCEE`)
- `molrs/src/ff/potential/kspace/pme.rs` — rustdoc only, lines 10 and 857: `"i"`/`"j"` → `atomi`/`atomj` (owned by phase 01 if it lands first; whichever phase lands first does it, the other finds it done)
- `molrs/src/io/data/prmtop_tables.rs` — rustdoc only: `Debt:` note on `decode_nonbond_params` (:210-214) naming the duplicated σ/ε closed form
- `molrs/src/io/data/prmtop.rs` — rustdoc only, line 47: correct the charge-factor provenance (`18.2223`, `18.2223² = 332.05221729`, 3.46e-5 below CODATA)
- `.claude/notes/notes.md` — one dated entry recording (i) the σ/ε closed-form duplication debt and its revisit trigger (third use), (ii) the missing `scripts/gen_gaff_energy_oracle.py` that `forcefield/gaff.rs` cites for the `AMBER_COULOMB` measurement (the value stands; its generator is not in the tree), and (iii) the drift that `molrs/tests/architecture_gate.rs` exists while `CLAUDE.md` § Testing Rules and `.claude/notes/testing.md` say there is no `molrs/tests/` tree (found while writing this spec; not resolved here)
- `.claude/notes/release.md` — one bullet under the next untagged version: prmtop-derived force fields now declare `lj/cut` + `coul/cut` with explicit constants; a LAMMPS include written **with** its header changes from `pair_style lj/cut/coul/long 10 10` to `pair_style lj/cut/coul/cut 9 10` (`pair_coeff` lines unchanged); NBFIX / 12-6-4 / multi-term-improper / non-uniform-SCEE prmtops are now refused
- `regressions/amber-prmtop-complete-02-forcefield.py` (new)

## Tasks

- [ ] Write failing unit tests for the pair-style split and the one-home AMBER constants in `molrs/src/ff/forcefield/readers/prmtop.rs` `#[cfg(test)]` (registered `lj/cut` + `coul/cut`, `coulomb`/`dielectric`/`cutoff` present, no `lj/cut/coul/long`, `AMBER_COULOMB == 18.2223²`) plus the `pair_coeff` characterization pin through `LammpsFfWriter` with and without `skip_pair_style`
- [ ] Write failing unit tests for `uniform_divisor` and `decode_lj_types` in the same module (defaults 1.2/2.0, non-uniform SCEE `Err`, suppressed-torsion exemption, σ/ε goldens, LB-consistent cross entry → `Ok` with no cross pair type, NBFIX-deviating cross entry → `Err` naming both types, 12-6-4 / multi-term improper / negative ICO / non-zero HBOND refusals)
- [ ] Create the hand-maintained `molrs/src/ff/params/amber.rs` (`pub mod amber;` in `params/mod.rs`, not in `MANIFEST.sha256`) holding `pub(crate)` `AMBER_COULOMB` / `AMBER_SCEE` / `AMBER_SCNB` with the merged provenance rustdoc and the two-roles note, delete the private copies in `molrs/src/ff/forcefield/gaff.rs` and `readers/prmtop.rs`, and import them at both sites
- [ ] Implement `DEFAULT_CUTOFF_LJ` / `DEFAULT_CUTOFF_COUL` and the registered `lj/cut` + `coul/cut` pair split in `molrs/src/ff/forcefield/readers/prmtop.rs`, replacing `lj/cut/coul/long`
- [ ] Implement `uniform_divisor` in `molrs/src/ff/forcefield/readers/prmtop.rs`, replacing `scale_divisor`, and feed `SpecialBonds` from it
- [ ] Implement `decode_lj_types` (`LjSelfRow` alias, full-ICO walk, NBFIX refusal) in `molrs/src/ff/forcefield/readers/prmtop.rs`, replacing the inline diagonal decode, and add the 12-6-4 and multi-term-improper refusals
- [ ] Add rustdoc per rustdoc style with units for every new symbol in `molrs/src/ff/forcefield/readers/prmtop.rs`, the `Debt:` σ/ε duplication note there and in `molrs/src/io/data/prmtop_tables.rs`, the charge-factor provenance correction in `molrs/src/io/data/prmtop.rs` (doc comments only), the stale exclusions-column rustdoc in `molrs/src/ff/potential/kspace/pme.rs` (lines 10 and 857), the debt entry in `.claude/notes/notes.md`, and the release bullet in `.claude/notes/release.md`
- [ ] Add regression example `regressions/amber-prmtop-complete-02-forcefield.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Verify the GAFF `c3` σ/ε closed-form goldens and the 1.2/2.0 divisor case against the hard-coded values, with no AmberTools in the loop
- [ ] Run full check + test suite

## Testing strategy

Project rule (CLAUDE.md "Testing Rules (MANDATORY)", `.claude/notes/testing.md`): unit tests are `#[cfg(test)]` next to the code, each targeting one function, and **no AmberTools / ParmEd / OpenMM** at test time. All tests below live in `molrs/src/ff/forcefield/readers/prmtop.rs` beside the three existing ones, over one inline `GAFF_MINI` fixture (`NTYPES = 2`, types `c3`/`hc`, full `NONBONDED_PARM_INDEX`, one bond/angle/dihedral, `SCEE`/`SCNB` sections) and small per-case variants, plus one corpus anchor: the committed `tests-data/prmtop/LiTFSI.prmtop` (fetched by `scripts/fetch-test-data.sh` in CI/pre-push; CLAUDE.md: "use `tests-data/` only when the corpus file is the assertion" — a byte-identity pin of a real LEaP-written file is exactly that), following the repo's existing corpus-test pattern. Gate: `cargo test -p molcrafts-molrs --lib --features full,filesystem`, plus `cargo test --doc -p molcrafts-molrs --features full,filesystem` for the new rustdoc.

Happy path:

- `pair_styles_are_registered_lj_cut_and_coul_cut` — `get_style("pair","lj/cut")` and `("pair","coul/cut")` both resolve, `lj/cut/coul/long` is gone, `coul/cut.params` carries `coulomb = 332.05221729`, `dielectric = 1.0`, `cutoff = 10.0`, `lj/cut.params["cutoff"] = 9.0`.
- `special_bonds_are_reciprocal_divisors` — `lj[2] = 0.5`, `coul[2] = 1/1.2` for the SCEE 1.2 / SCNB 2.0 fixture.
- `pair_coeff_text_is_pinned` — **characterization test**: the `pair_coeff` block emitted by `LammpsFfWriter` for the fixture equals a hard-coded literal captured from the *pre-change* reader (so the test is green before and after); with `LammpsWriteOptions { skip_pair_style: true, .. }` the output contains no `pair_style` substring and still contains every pinned `pair_coeff` line; with default options the header line is exactly `pair_style lj/cut/coul/cut 9.000000 10.000000`.
- `litfsi_corpus_pair_coeff_text_is_pinned` — the same pin on the real `tests-data/prmtop/LiTFSI.prmtop` (NTYPES = 6, SCEE ≡ 1.2, SCNB ≡ 2.0, no CCOEF, no negative ICO, zero HBOND, every off-diagonal exactly Lorentz–Berthelot, no multi-term impropers — i.e. it passes every new refusal): the **seven** `pair_coeff` lines captured from the pre-change reader (seven distinct `AMBER_ATOM_TYPE` names — `f`, `c3`, `s6`, `o`, `ne`, `sy`, `Li+` — over `NTYPES = 6`; `sy` and `s6` share LJ index 3, and the reader emits one line per *name*, `readers/prmtop.rs:437-443`) are reproduced byte for byte, in first-appearance order, and `special_bonds()` is `lj_14 = 0.5`, `coul_14 = 1/1.2`.

Edge cases:

- `uniform_divisor_defaults_when_absent` → 1.2 / 2.0 with both sections removed.
- `uniform_divisor_rejects_mixed_scee` → `Err` whose message contains `SCEE_SCALE_FACTOR` and both values (1.2 and 1.0).
- `uniform_divisor_ignores_suppressed_torsions` → a torsion with a negative 3rd pointer referencing a divergent divisor does **not** error.
- `uniform_divisor_rejects_nonpositive` → a used type id with divisor 0.0 is an `Err` naming the flag and the id.
- `rejects_12_6_4` → fixture with `LENNARD_JONES_CCOEF` → `Err` containing `12-6-4`.
- `rejects_multiterm_improper` → improper whose type id chains a negative `PN` → `Err` naming that type id.
- `rejects_negative_ico` / `rejects_nonzero_hbond` → both `Err("10-12 interactions are not supported")`.
- `empty_prmtop_errors`, `multiterm_expansion`, `comment_lines_ignored` — existing tests kept green.

Domain validation (hard-coded expected values, hand-derived from published GAFF parameters — no oracle tool involved):

- `decode_lj_types_self_terms_match_closed_form` — fixture `LENNARD_JONES_ACOEF = 1043080.23`, `BCOEF = 675.612248` for `c3` (built from GAFF `R* = 1.9080 Å`, `ε = 0.1094 kcal/mol`: `A = ε·r_min¹²`, `B = 2ε·r_min⁶`, `r_min = 2R*`) → `ε = 0.109400 kcal/mol` and `σ = 3.3996695 Å`, both within **1e-6 relative**.
- `decode_lj_types_refuses_nbfix_cross_terms` — an exact-LB `c3`/`hc` off-diagonal entry yields `Ok` with **no** two-endpoint pair type (round-trip noise ~7.5e-9 ≪ 1e-6), while the same entry with ε scaled by 1.01 yields `Err` whose message contains `c3`, `hc` and `not supported`.
- `amber_coulomb_is_18_2223_squared` — `params::amber::AMBER_COULOMB` equals `18.2223_f64.powi(2)` within 1e-9, and `(COULOMB_REAL − AMBER_COULOMB)/COULOMB_REAL` equals `3.4610e-5` within 1e-8 (pins the documented cross-engine offset); `1.0 / AMBER_SCEE` and `1.0 / AMBER_SCNB` equal the `SpecialBonds` weights the GAFF typifier force field (`forcefield/gaff.rs`) declares — one definition, two consumers.

Regression example: `regressions/amber-prmtop-complete-02-forcefield.py`, public API only — `molrs.ff.read_amber_prmtop_ff_str(TEXT)` on the same inline GAFF-shaped prmtop, then `molrs.ff.write_lammps_forcefield_str(ff, skip_pair_style=True)`. Asserts, against hard-coded literals: `lj/cut` `c3` `σ = 3.3996695 Å` / `ε = 0.1094 kcal/mol` (rel 1e-6), `coul/cut` `coulomb = 332.05221729` / `dielectric = 1.0` / `cutoff = 10.0`, `lj/cut` `cutoff = 9.0`, that the written include contains the `pair_coeff c3 c3` line while containing no `pair_style` substring, and that an NBFIX-scaled variant of the text raises `ValueError` naming both types. Header comment records that the goldens are hand-derived from the closed form and published GAFF `R*`/`ε` — **no** AmberTools/ParmEd import or subprocess anywhere in the file.

## Out of scope

- prmtop **structure**/connectivity reading and charge de-scaling (`molrs/src/io/data/prmtop.rs`) — phase `amber-prmtop-complete-01`; only its rustdoc line 47 is touched here.
- `special_bonds` emission/parsing for the LAMMPS and GROMACS force-field writers/readers — phase `amber-prmtop-complete-03`.
- Python binding surface — phase `amber-prmtop-complete-04`. `molrs.ff.read_amber_prmtop_ff_str` and `molrs.ff.write_lammps_forcefield_str` already exist and are consumed unchanged by the regression script; `SpecialBonds` is deliberately **not** asserted from Python because it has no Python accessor yet.
- "Undeclared cutoff → `Err`" (deferred): a prmtop carries no cutoff, so refusing the file would break every current caller. The named constants + rustdoc are the minimum; revisit when a caller can supply the mdin cutoff.
- Moving the σ/ε closed form into `core::math` — accepted duplication with `Debt:` notes; revisit at the third use. `core` is not touched (single-layer rule).
- `coul/long/pme` selection, Ewald/PME parameters, or any `alpha`/grid inference from the prmtop.
- 12-6-4 (`LENNARD_JONES_CCOEF`), 10-12 H-bond, and NBFIX off-diagonal LJ support — all three are explicit refusals, not implementations. NBFIX would need a per-pair kernel path (`pair_lj_cut_ctor` mixes unconditionally), a writer guard so `pair_coeff i j` lines are emitted only when a kernel reads them, and a LAMMPS-reader counterpart — its own spec.
- CHAMBER-flavour prmtop sections (Urey–Bradley, CMAP), `LES` topologies, and polarizable (`IPOL`) topologies.
- Any AmberTools-generated fixture or oracle in the test gate.

## Follow-up (molpy)

- **Release order is an iron law** (CLAUDE.md "Release before molpy"): this changes what a prmtop-derived `ForceField` *is* (pair style names and declared constants), so molrs must reach `master` with a `vX.Y.Z` tag and be published before molpy bumps its minor pin.
- molpy's AMBER path must re-verify the PEO pipeline include after the bump: the `pair_coeff` block is pinned byte-identical and `skip_pair_style=True` output is unchanged, but a molpy caller that writes **with** the header now gets `pair_style lj/cut/coul/cut 9 10` instead of `lj/cut/coul/long 10 10`.
- molpy code that reads the `coulomb` constant off an AMBER force field now sees `332.05221729` (Amber's own) rather than a missing key; any molpy-side cross-check against a CODATA-based engine should expect the documented 3.46e-5 relative offset.
- `SpecialBonds` (1-4 weights) still has no Python accessor; molpy consumers waiting on it should track phases 03/04, not this spec.
