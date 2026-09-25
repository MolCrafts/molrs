---
title: GROMACS force-field I/O reads and writes force-field directives
slug: opls-gromacs-01-gromacs-io
status: code-complete
created: 2026-09-25
chain: opls-gromacs (01-gromacs-io → 02-table → 03-rules)
depends_on:
---

# GROMACS force-field I/O reads and writes force-field directives

## Summary

molrs cannot currently read GROMACS's own OPLS-AA files, and what it writes to GROMACS is wrong.

- **Reader.** It refuses combination rule 3, which OPLS-AA declares. It does not read `[ bondtypes ]`, `[ angletypes ]` or `[ dihedraltypes ]`. It maps dihedral function codes 2 and 3 wrongly (Ryckaert–Bellemans, code 3, is read as "harmonic"). It puts σ and ε on atom types, where the `lj/cut` kernel never looks. It also builds force-field types out of a molecule's `[ atoms ]` and `[ bonds ]` rows, which is topology.
- **Writer.** It invents per-atom `[ atoms ]` fields and writes an OPLS torsion as code 1 with zeros, which silently loses it. It never declares the combination rule.

After this link:

- Reader and writer both deal only in force-field directives. The reader turns `[ defaults ]`, `[ atomtypes ]` (split into `atom/full` and `pair/lj/cut`), the three bonded type sections, and `#include`, `#define` and `#ifdef` into molrs styles with correct kernel names, converting units at the boundary. Anything it does not model is refused by name, unless the caller has explicitly skipped that section.
- The writer produces those same sections, including OPLS torsions as code 3 through the exact Fourier→Ryckaert–Bellemans relation, and refuses what GROMACS cannot express.
- The RB↔Fourier conversions exist once, in `ff::forcefield::torsion`.
- The duplicate Rust free-function doors are deleted.

This is what lets link 02 read `share/top/oplsaa.ff` directly.

## Domain basis

Functional forms (GROMACS reference manual, § Bonded interactions and § Non-bonded interactions; Abraham et al., *SoftwareX* 1–2, 19–25 (2015), DOI 10.1016/j.softx.2015.06.001), each against the molrs kernel it maps to:

| GROMACS | Form | molrs style | Conversion at the boundary |
|---|---|---|---|
| bondtypes funct 1 | ½k_b(r−b₀)² | `bond/harmonic` (½k form) | r0 = b₀·10 (nm→Å); k = k_b/418.4 |
| bondtypes funct 3 | D[1−e^{−β(r−b₀)}]² | `bond/morse` (`D`, `alpha`, `r0`) | D/4.184; alpha = β/10; r0 = b₀·10 |
| angletypes funct 1 | ½k_θ(θ−θ₀)² | `angle/harmonic` (½k form) | theta0 deg→rad; k/4.184 |
| dihedraltypes funct 1 | k_φ[1+cos(nφ−φ_s)] | `dihedral/periodic` (`k`, `periodicity`, `phase`) | phase deg→rad; k/4.184 |
| dihedraltypes funct 2 | ½k_ξ(ξ−ξ₀)² | `improper/harmonic` (K(χ−χ₀)², χ = \|φ\|) | K = k_ξ/(2·4.184); accepted only for ξ₀ = 0, where signed and unsigned forms agree |
| dihedraltypes funct 3 | Σₙ Cₙ cosⁿψ, ψ = φ−180° (Ryckaert & Bellemans, *Faraday Discuss. Chem. Soc.* 66, 95 (1978), DOI 10.1039/DC9786600095) | `dihedral/opls` (`k1..k4`) | Fourier inversion below, then /4.184 |
| dihedraltypes funct 4 | k_φ[1+cos(nφ−φ_s)] | `improper/periodic` | as funct 1 |

OPLS Fourier form, V = ½[F₁(1+cosφ) + F₂(1−cos2φ) + F₃(1+cos3φ) + F₄(1−cos4φ)] (Jorgensen, Maxwell, Tirado-Rives, *JACS* 118, 11225 (1996), DOI 10.1021/ja9621760). Exact relations (the GROMACS manual's RB↔Fourier equations, Eqs. 200–201 in molrs's existing citation):

- RB → Fourier: F₁ = −2C₁ − 1.5C₃, F₂ = −C₂ − C₄, F₃ = −C₃/2, F₄ = −C₄/4.
- Fourier → RB: C₀ = F₂ + ½(F₁+F₃), C₁ = ½(−F₁+3F₃), C₂ = −F₂+4F₄, C₃ = −2F₃, C₄ = −4F₄, C₅ = 0.
- A row is representable iff C₅ = 0 and ΣCₙ = 0, because V_Fourier(180°) = 0 and V_RB(ψ=0) = ΣCₙ.

Combination rules: comb-rule 2 means σᵢⱼ = ½(σᵢ+σⱼ), εᵢⱼ = √(εᵢεⱼ), i.e. `Mixing::Arithmetic`. comb-rule 3 means σᵢⱼ = √(σᵢσⱼ), εᵢⱼ = √(εᵢεⱼ), i.e. `Mixing::Geometric`. Under both, `[ atomtypes ]` V/W are σ (nm) and ε (kJ/mol). Internal units are Å, kcal/mol, rad, e; the file uses nm, kJ/mol, degrees.

## Design

### 0. Chain rules (shared by links 01–03)

- **Operator rulings, 2026-09-25.**
  - GROMACS `share/top/oplsaa.ff` (tag v2026.3, commit `42105e4672205b4aa951962b8e3cdb4c27890da1`) is the authority for the OPLS-AA type set, charges, masses, classes (`bond_type`), LJ and bonded parameters.
  - molrs owns the typing rules: standard Daylight SMARTS, explicit bond orders where needed, aromaticity perceived before matching, no dialect flag.
  - opls_150/opls_178 follow GROMACS.
  - CL&P does **not** land in molrs in this chain. molpy deletes its typifiers, `ClpTypifier` included, in its own chain; CL&P is a future separate spec.
- **Verification.** Every task runs only `cargo mrs-test [-- <module filter>]`. There is no A/B harness, no second `CARGO_TARGET_DIR` or worktree, and no hand-typed feature string. There is no clippy, fmt check, doctest, rustdoc or binder build inside the chain. The full gate runs once, at the end of link 03, and discharges every link's full-gate criterion.
- **Rust first.** Links 01–02 change only `molrs/` and repo-root config and notes. The one binder change (molrs-python) is in link 03. Binders may not compile against the tree between 01 and 03; that is by design, and acceptable only because links 01 and 02 are never committed or pushed on their own — the chain commits once, at the end of link 03 (pre-push builds the binders).
- **Test inputs.** Goldens are hand-derived only. A GROMACS parameter line is the force-field definition being transcribed. A test may hand-copy one and hand-convert it (the notes.md 2026-09-25 "source-equivalence" pattern). No number captured from running any third-party program is used.
- **Constitution.** There is no `.claude/notes/law.md`. The governing rules are CLAUDE.md § Design preferences / § Testing Rules, `architecture-rules.md` (io and ff never name each other; ff never names md), and notes.md § Binding-surface symmetry.
- **Regression example.** There is no `regressions/` tree; each link's regression example is a rustdoc doctest, run by the chain-end gate.
- **Stage and language.** Stage is experimental: changed APIs change outright, with no shims. English only. The chain lands on the unreleased 0.15 tree; molpy follows in its own chain.

### 1. Torsion conversions — `molrs/src/ff/forcefield/torsion.rs` (new)

There is one home for both directions:

- **`rb_to_opls`.** It moves out of `readers/opls.rs:497`, where it is private, as `pub(crate) fn rb_to_opls(c: [f64; 6]) -> Result<[f64; 4], String>`.
  - Input and output are in kJ/mol; callers divide by 4.184.
  - It is `Err` (naming the coefficients) when |C₅| > `RB_TOL` or |ΣCₙ| > `RB_TOL`.
  - `RB_TOL = 1e-4` kJ/mol. Rustdoc gives the reason: GROMACS prints 5 decimals, so 6 rounded coefficients sum to at most 3e-5 kJ/mol.
  - This closes today's silent discard of C₀ and C₅.
- **`opls_to_rb`.** `pub(crate) fn opls_to_rb(f: [f64; 4]) -> [f64; 6]`, same units in and out. It **replaces** the private `opls_to_rb` at `writers/xml.rs:330` (used at `:208`). The XML writer keeps its own `c0..c5` pass-through branch and its kcal→kJ conversion at the call site, and calls `torsion::opls_to_rb` for the `k1..k4` branch. The GROMACS writer (§ 6) is its second caller.

These are free functions because the math has no owning type (CLAUDE.md § Prefer). They live in `ff::forcefield`, not `ff::potential`, for the reason `mixing.rs` records: a reader importing from a kernel once caused a cycle.

### 2. `Mixing` — `molrs/src/ff/forcefield/mixing.rs`, `molrs/src/ff/potential/pair/lj_cut.rs`

- `pub(crate) const Mixing::UNDECLARED: Mixing = Mixing::Arithmetic`: the rule an `lj/cut` style that declares none is evaluated under.
- `pub(crate) fn Mixing::name(self) -> &'static str`: the canonical spelling.
- `pair_lj_cut_ctor` (`lj_cut.rs:645-648`) reads `UNDECLARED` instead of its own literal.
- The unknown-rule message at `mixing.rs:39` has a run of stray spaces from a line continuation; it is fixed.

### 3. Reader: force-field directives only — `molrs/src/ff/forcefield/readers/gromacs.rs`

**Construction.** `GromacsTopFfReader`'s fields become private. Configuration is by builder: `with_include(bool)` (kept) and `with_skipped_directive(&str)` (new; the skipped set is private). The `include_dirs` capability is deleted: nothing in any sibling repo sets it and Python never exposed it (CLAUDE.md § Prefer); includes resolve relative to the including file, as the link-02 generator needs. The free function `read_gromacs_top_ff` (`:84`) and its re-export (`ff/mod.rs:14`) are deleted; callers construct the reader. The only in-repo caller is the molrs-python binder, which migrates in link 03.

**Sections read:**

- **`[ defaults ]`.** Requires nbfunc 1 and gen-pairs yes. comb-rule 2 or 3 sets the `pair/lj/cut` string param `mixing` to `arithmetic` or `geometric`. comb-rule 1 (C6/C12) and nbfunc 2 are `Err` naming the value. Special bonds are `lj [0,0,fudgeLJ]`, `coul [0,0,fudgeQQ]`.
- **`[ atomtypes ]`.** Column resolution is unchanged (`parse_atomtypes_row`).
  - `mass`, `charge`, string `ptype`, string `bond_type` and `atomic_number` (when present) go on `atom/full`.
  - σ (Å) and ε (kcal/mol) go on the `pair/lj/cut` self row (`def_type_at(name, &[name], …)`), as `ff/typifier/opls/embedded.rs:95-102` does.
  - When atom types are present, `pair/coul/cut` is defined with `coulomb = COULOMB_REAL` and `dielectric = VACUUM_DIELECTRIC`, as `OplsXmlReader` does.
- **`[ bondtypes ]` / `[ angletypes ]` / `[ dihedraltypes ]`.**
  - Keyed by the row's labels through `def_type_at` and the conflict rule; function codes and conversions as in Domain basis.
  - GROMACS `X` becomes the canonical empty-endpoint wildcard (`core/store/type_labels.rs:37-40`), which the OPLS matcher already scores as a wildcard (`assign.rs:82`).
  - The 2-name dihedraltypes form is `Err`.

**Refusals.** Each is `Err` naming the section, code and row:

- Refused function codes: bond 2, 4, 5+; angle 2+; dihedral 5, 8, 9 (multi-term), 10+; and dihedral 2 with ξ₀ ≠ 0.
- Refused sections: `[ pairtypes ]`, `[ nonbond_params ]`, `[ constrainttypes ]`, `[ cmaptypes ]`, `[ implicit_genborn_params ]`, and any unknown section, unless the caller skipped it.
- Molecule-level sections, `[ moleculetype ]` and everything that belongs to a molecule (`[ atoms ]`, `[ bonds ]`, `[ pairs ]`, `[ angles ]`, `[ dihedrals ]`, `[ exclusions ]`, `[ settles ]`, `[ system ]`, `[ molecules ]` …), are topology. They are `Err` with the plain-text hint "topology: read with io::data::top::read_top" (plain text in both the message and the rustdoc — an intra-doc link would name `io` from `ff`, architecture-rules.md, and break rustdoc without the `io` feature), unless the caller skipped them. The `[ atoms ]`→index resolution path (`:330-371`, `:428-691`) is deleted.

### 4. Reader: preprocessor

- `#include` behaves as today.
- `#define NAME [body]` and `#undef NAME` maintain a define set.
- `#ifdef` / `#ifndef` / `#else` / `#endif` are evaluated against that set, and nest.
- `#if` and `#elif` are `Err`.
- Macro bodies are never expanded. OPLS's six improper macros are applied by GROMACS through `.rtp` files; see Out of scope.

### 5. Molecule sections leave the force-field reader

At experimental stage, reader and writer are both directive-only, and `io::data::top` owns topology. Deleting the molecule path also removes the debt recorded in notes.md 2026-09-25 "GROMACS force-field files model a molecule, not directives": per-instance `TypeConflict`, merging `[ atoms ]` across molecule types, and the invented writer fields. Task 8 updates that entry.

### 6. Writer — `molrs/src/ff/forcefield/writers/gromacs.rs`

Whole-force-field serialization as directives only. `[ atoms ]`, `[ bonds ]`, `[ angles ]`, `[ dihedrals ]` and `[ pairs ]` are no longer written. The free functions `write_gromacs_top_ff` / `write_gromacs_top_ff_str` (`:317`, `:323`) and their re-exports (`ff/mod.rs:21`) are deleted; callers use `GromacsTopFfWriter::new().with_precision(p).write(ff, path)` / `.write_str(ff)`. The only in-repo callers are `molrs-python/src/ff/mod.rs:1988` and `:1999`, which migrate in link 03.

The writer produces:

- **`[ defaults ]`:** `1 <comb> yes <fudgeLJ> <fudgeQQ>`. comb comes from `lj/cut`'s `mixing`, or `Mixing::UNDECLARED` when absent: Arithmetic→2, Geometric→3, SixthPower→`Err`. Special bonds with a non-zero 1-2 or 1-3 weight are `Err`.
- **`[ atomtypes ]`:** mass and charge from `atom/full`, σ/ε from the `lj/cut` self row, in the 6/7/8-column form chosen by `bond_type` / `atomic_number`.
  - A type missing `mass`, `charge` or its `lj/cut` self row is `Err` naming the type and what is missing; the old `0.0` placeholders are gone.
  - A missing `ptype` writes `A`, because molrs `atom/full` types are real atoms; a declared `ptype` round-trips.
  - An explicit `lj/cut` cross row is `Err`.
- **`[ bondtypes ]` / `[ angletypes ]` / `[ dihedraltypes ]`:** the inverse of § 3. `dihedral/opls` is written as code 3 via `torsion::opls_to_rb`. A multi-term `dihedral/periodic` is `Err`. The empty wildcard is written as `X`.
- **Refusals:** any other style (e.g. `dihedral/charmm`, `multi/harmonic`) is `Err`. So is a non-wildcard endpoint label that is neither an atom-type name nor a `bond_type`.

### 7. Behaviour changes disclosed

- **The GROMACS reader no longer reads molecule sections.** A full `.top` needs its molecule sections skipped, or the caller reads `forcefield.itp`. σ/ε move to `pair/lj/cut`, styles carry kernel names, impropers are in the `improper` category, and unmodelled content fails loudly.
- **The writer output layout is new.**
- **`OplsXmlReader` refuses non-representable RB rows.**
- **The Rust free functions are deleted.**

molpy's `io/forcefield/top.py` consumes the Python `read_gromacs_top_ff` / `write_gromacs_top_ff` on `.top` files and follows in its own chain. The Python names are kept; link 03 adds `skip_directives`.

### Reuse decision

No librarian report was passed. Candidates come from the orchestrator's brief and the architect review.

- `generalize rb_to_opls` (`readers/opls.rs:497`) and `generalize opls_to_rb` (`writers/xml.rs:330`) into `torsion.rs`. One copy of each relation; three callers (the OPLS XML reader, the GROMACS reader and writer, the XML writer).
- `generalize Mixing` with `pub(crate)` `UNDECLARED` / `name`.
- `reuse` the `OplsXmlReader` non-bonded shape (`coul/cut` constants, `lj/cut` self rows, `mixing` string, `readers/opls.rs:195-260`) and the embedded `lj/cut` self-row construction (`embedded.rs:95-102`).
- `reuse parse_atomtypes_row` (`readers/gromacs.rs:381`) and the writer's `atomtypes_row` column logic.
- `reuse` the `TypeName` empty-endpoint wildcard and the conflict rule of `def_type_at`.
- `new — with_skipped_directive`: no existing option says "this section is deliberately not modelled"; its shape follows `with_include`. Its current consumers are the link-02 generator (`constrainttypes`) and the Python readers (link 03).

## Files to create or modify

- `molrs/src/ff/forcefield/torsion.rs` (new)
- `molrs/src/ff/forcefield/mod.rs`
- `molrs/src/ff/forcefield/mixing.rs`
- `molrs/src/ff/potential/pair/lj_cut.rs`
- `molrs/src/ff/forcefield/readers/opls.rs`
- `molrs/src/ff/forcefield/writers/xml.rs`
- `molrs/src/ff/forcefield/readers/gromacs.rs`
- `molrs/src/ff/forcefield/writers/gromacs.rs`
- `molrs/src/ff/mod.rs`
- `.claude/notes/notes.md`

## Tasks

- [x] Write failing unit tests for `rb_to_opls` / `opls_to_rb` in `molrs/src/ff/forcefield/torsion.rs` and for `Mixing::UNDECLARED` / `name` in `molrs/src/ff/forcefield/mixing.rs`; verify with `cargo mrs-test -- ff::forcefield::torsion ff::forcefield::mixing`
- [x] Generalize both conversions into `molrs/src/ff/forcefield/torsion.rs` (register it in `molrs/src/ff/forcefield/mod.rs`). Point `molrs/src/ff/forcefield/readers/opls.rs` and `molrs/src/ff/forcefield/writers/xml.rs` (keeping its c0..c5 branch and kcal→kJ at the call site) at it, deleting both private copies. Add `pub(crate)` `Mixing::UNDECLARED` / `name` and use them in `molrs/src/ff/potential/pair/lj_cut.rs`. Fix the `mixing.rs:39` message. Verify with `cargo mrs-test -- ff::forcefield ff::potential::pair::lj_cut`
- [x] Write failing reader tests in `molrs/src/ff/forcefield/readers/gromacs.rs` for § 3–4: defaults comb-rules, atomtypes split, the three bonded sections, wildcard, function-code map and refusals, molecule-section refusal with its `read_top` hint, preprocessor, `with_skipped_directive`. Delete the `[ atoms ]`-path tests
- [x] Implement the directive-only reader, preprocessor and private builder configuration in `molrs/src/ff/forcefield/readers/gromacs.rs`. Delete the molecule-section path and `read_gromacs_top_ff`, and its re-export in `molrs/src/ff/mod.rs`. Verify with `cargo mrs-test -- ff::forcefield::readers::gromacs`
- [x] Write failing writer tests in `molrs/src/ff/forcefield/writers/gromacs.rs` for § 6: defaults line; atomtypes from `atom/full` + `lj/cut`; missing mass, charge or self row is `Err`; bonded sections; `opls` as code 3; `X` wildcard; every refusal; read(write(ff)) round trip
- [x] Implement the directive writer in `molrs/src/ff/forcefield/writers/gromacs.rs`, deleting `write_gromacs_top_ff(_str)` and their re-export in `molrs/src/ff/mod.rs`; verify with `cargo mrs-test -- ff::forcefield::writers::gromacs`
- [x] Write a failing test in `molrs/src/ff/forcefield/readers/opls.rs` that an `<RBTorsionForce>` row with c5 ≠ 0 is `Err`, and one in `molrs/src/ff/forcefield/writers/xml.rs` that an `opls` torsion with k3 0.3 is written with `c0..c5` = 1.2552·(½, 1.5, 0, −2, 0, 0) kJ/mol; verify with `cargo mrs-test -- ff::forcefield::readers::opls ff::forcefield::writers::xml`
- [x] Rewrite the module rustdoc of `readers/gromacs.rs`, `writers/gromacs.rs` and `torsion.rs` per rustdoc style, with units, and add a doctest on `GromacsTopFfReader` (the regression example). In `.claude/notes/notes.md`:
  - update the 2026-09-25 GROMACS entry: molecule model removed, its debts gone;
  - add routed entries with path:line:
    - `OplsXmlReader` silent drops (`readers/opls.rs:137-146` skips ImproperTorsionForce / PeriodicImproperForce / Custom*Force / Residues; `:457` drops `<Improper>` children; `:335-360` invents placeholder atom types `type_="*"` / `class_` for bonded classes), `/mol:fix`;
    - `io/data/top.rs:271-273` reads both branches of `#ifdef`/`#else`, `/mol:fix` (io and ff may not share code);
    - private `KJ_PER_KCAL` / `NM_TO_ANGSTROM` copies (`readers/gromacs.rs:37-38`, `writers/gromacs.rs:37-38`, `readers/opls.rs:57-59`, `writers/xml.rs:25`) beside `molrs::units`, `/mol:refactor`;
    - OPLS improper assignment (GROMACS `.rtp` macros), `/mol:spec`;
    - the remaining Rust convenience free functions beside their reader/writer types — `read_amber_prmtop_ff`, `read_forcefield_xml(_str)`, `write_forcefield_xml(_str)` (`ff/mod.rs:13-26`, `writers/xml.rs:348`) — `/mol:refactor`
- [x] Run the full unit suite `cargo mrs-test`. The full gate and binders run once, in link 03

## Testing strategy

Unit tests live in `#[cfg(test)]` next to the code, with inline string fixtures and hand-derived expectations. Filters: `ff::forcefield::{torsion,mixing,readers::gromacs,writers::gromacs,readers::opls,writers::xml}`, `ff::potential::pair::lj_cut`.

- **`torsion.rs`:**
  - `rb_to_opls([0.62760, 1.88280, 0, −2.51040, 0, 0]) = [0, 0, 1.2552, 0]`;
  - `opls_to_rb` of that result returns the six inputs;
  - the round trip is the identity on four-term inputs;
  - ΣC = 1 is `Err`, and C₅ = 0.1 is `Err`.
- **`mixing.rs`:** `name` for every variant; `parse(name(m)) == m`; `UNDECLARED == Arithmetic`.
- **Reader:**
  - `1 3 yes 0.5 0.5` gives `"geometric"`; `1 2 …` gives `"arithmetic"`.
  - The opls_135 8-column row gives `atom/full` {12.011, −0.18, CT, 6, A} plus an `lj/cut` self row {σ 3.5, ε 0.066}.
  - `CT HC 1 0.10900 284512.0` gives (1.09, 680.0).
  - `HC CT HC 1 107.800 276.144` gives (107.8·π/180, 66.0).
  - Funct 3 HC-CT-CT-HC gives (0, 0, 0.3, 0).
  - `X CT CT X` gives `["", "CT", "CT", ""]`.
  - `CT CT CT CT 1 0.0 4.184 3` gives periodic (1.0, 3, 0).
  - `X X N H 4 180.0 10.46 2` gives `improper/periodic` (2.5, π).
  - `X X C O 2 0.0 167.36` gives `improper/harmonic` K 20.0.
  - Morse `CT CT 3 0.1529 400.0 20.0` gives (95.602294…, 2.0, 1.529).
  - Refusals, each an `Err` naming its cause: comb-rule 1, nbfunc 2, dihedral code 9, code 2 with ξ₀ ≠ 0, ΣC ≠ 0, `[ pairtypes ]`, `[ constrainttypes ]` (then `Ok` after `with_skipped_directive`), `[ atoms ]` (message names `read_top`), `#if`.
  - Preprocessor: a `#ifdef FOO` block is read iff `#define FOO` came first; `#ifndef` is the complement; `#else` takes the other branch.
- **Writer:**
  - A geometric FF gives `1 3 yes …`; an undeclared FF gives `1 2 …`.
  - The atomtypes row reproduces the opls_135 values.
  - `opls` k3 0.3 gives `3 0.6276 1.8828 0 -2.5104 0 0`; the empty endpoint is written as `X`; no `[ atoms ]` appears.
  - `Err` for: sixthpower; `dihedral/charmm`; a multi-term periodic; an explicit cross row; a 1-2 special-bond weight; an unresolvable endpoint; a type lacking a mass, a charge or an `lj/cut` self row.
  - read(write(ff)) equals ff for a force field with every supported style.
- **Regression example:** a doctest on `GromacsTopFfReader` asserts the `mixing` string and one converted bond `r0` against literals. It runs in the chain-end gate.

## Out of scope

- OPLS improper assignment (`.rtp` macro application): routed `/mol:spec` by task 8.
- Code 9, C6/C12, Buckingham, `[ pairtypes ]`, `[ nonbond_params ]`, constraints, CMAP: refused by name.
- Topology reading: owned by `io::data::top`. Its preprocessor defect is routed by task 8, not fixed here.
- `OplsXmlReader`'s silent drops and the duplicated unit constants: routed by task 8.
- GROMACS export of a *typing output*: typing stamps only numeric params, so `bond_type`/`ptype` are absent, and the writer's endpoint and atomtype checks refuse it. Routed `/mol:spec`.
- Binder migration (molrs-python calls the deleted free functions): link 03.
- CL&P: a future separate spec.
