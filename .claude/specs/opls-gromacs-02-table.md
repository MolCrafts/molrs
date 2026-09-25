---
title: The committed OPLS-AA table is generated from pinned GROMACS oplsaa.ff
slug: opls-gromacs-02-table
status: code-complete
created: 2026-09-25
chain: opls-gromacs (01-gromacs-io → 02-table → 03-rules)
depends_on: [opls-gromacs-01-gromacs-io]
---

# The committed OPLS-AA table is generated from pinned GROMACS oplsaa.ff

## Summary

`molrs/src/ff/params/oplsaa.rs` was emitted from foyer's `oplsaa.xml` by a generator that no longer exists, and has since been hand-edited.

- **Classes are foyer's, not GROMACS's.** 651 of 813 types carry their own name as class. As a result 162/300 bond rows, 506/932 angle rows and 636/1048 dihedral rows can never be matched.
- **Rows differ from GROMACS.** It has 4 foyer-only rows, lacks H-N-CT_2-C, and writes σ = 10 Å where GROMACS has 0.
- **Mixing disagrees.** The embedded `lj/cut` declares no mixing rule, so the kernel mixes arithmetically while an exported LAMMPS input (with no `pair_modify`) mixes geometrically. CT–HC σ comes out 2.958 Å in one and 3.000 Å in the other.

After this link:

- `cargo mrs-gen-opls --gromacs <oplsaa.ff>` regenerates the table byte for byte from the pinned GROMACS files, through link 01's reader.
- Classes are GROMACS `bond_type`, and the embedded `lj/cut` declares `geometric`.
- The LAMMPS writer names the kernel's rule whenever a style declares none.
- The header records where every number came from.
- The SMARTS typing rules move, unchanged, into their own molrs-owned table of `OplsRuleRow`s, which link 03 rewrites.

## Domain basis

- OPLS-AA: Jorgensen, Maxwell, Tirado-Rives, *JACS* 118, 11225 (1996), DOI 10.1021/ja9621760. Combining rule geometric in σ and ε; GROMACS `forcefield.itp` `[ defaults ] 1 3 yes 0.5 0.5`.
- **Authority:** GROMACS v2026.3, commit `42105e4672205b4aa951962b8e3cdb4c27890da1`, `share/top/oplsaa.ff/{forcefield.itp, ffnonbonded.itp, ffbonded.itp}`, LGPL-2.1-or-later (Abraham et al. 2015, DOI 10.1016/j.softx.2015.06.001).
  - `ffnonbonded.itp`: 813 atomtypes.
  - `ffbonded.itp`: 300 bondtypes (½k), 930 angletypes, 1048 dihedraltypes funct 3 (96 with `X`), 6 improper `#define` macros.
- Conversions are link 01's reader: σ×10, ε/4.184, b₀×10, k_b/418.4, θ₀ deg→rad, k_θ/4.184, RB→Fourier.

## Design

### 0. Chain rules

As in `opls-gromacs-01-gromacs-io` § 0.

### 1. Generator — `molrs/examples/gen_opls_params.rs` (new)

It is a Rust example. That way it dogfoods link 01's reader, "nothing parses parameter text at runtime" stays true of the library, and no second GROMACS parser exists.

**Cargo setup.**

- `molrs/Cargo.toml`:
  - `[[example]] name = "gen_opls_params"`, `required-features = ["ff"]`;
  - `exclude = ["examples/"]` in `[package]`, so the tool is not published; `cargo package` then drops the target with a warning — `.claude/notes/release.md`'s publish checklist records this warning as expected, so it is not mistaken for a packaging failure;
  - `sha2 = "0.10"` under `[dev-dependencies]` (it shares `digest 0.10`, already locked);
  - the targets comment changes to "one example target: the OPLS parameter generator, a tool, not a test; excluded from the package".
- `.cargo/config.toml` gains `mrs-gen-opls = ["run", "-p", "molcrafts-molrs", "--example", "gen_opls_params", "--features", "full,filesystem,stream", "--"]`. The alias already ends in `--`, so **the command is `cargo mrs-gen-opls --gromacs <dir>`**, and the header, CLAUDE.md and the criteria all spell it that way.

**Fingerprints, stated honestly.** The example uses build set 1's feature string, so it links the same library unit as `mrs-build`. It does add exactly one example unit under `target/debug/.fingerprint/molcrafts-molrs-*`, and `mrs-check` / `mrs-clippy --all-targets` add their check units for it.

- The legitimate-builds lists in `.cargo/config.toml` and `.claude/notes/build.md` § "Four legitimate builds" are amended to say so.
- Task 4 verifies no *library* fingerprint is added.
- `CLAUDE.md` § Build & Test Commands lists the command.

**Behaviour of `--gromacs <dir>`:**

1. Checks the SHA-256 of the three files against constants pinned in the generator; a mismatch is a hard error naming the file.
2. Reads `forcefield.itp` via `GromacsTopFfReader::new().with_include(true).with_skipped_directive("constrainttypes")`.
3. Requires `mixing == "geometric"` and special bonds `[0,0,0.5]`.
4. Walks `atom/full`, `pair/lj/cut`, `bond/harmonic`, `angle/harmonic` and `dihedral/opls` in insertion order. Any other style, or an atom type without an `lj/cut` self row, is a hard error.
5. Writes `molrs/src/ff/params/oplsaa.rs`: f64 via `{:?}`, `#[rustfmt::skip]` tables, and a header containing:
   - "DO NOT HAND-EDIT — regenerate with `cargo mrs-gen-opls --gromacs <path>`";
   - the tag, commit, the three paths and their SHA-256s;
   - the LGPL-2.1-or-later attribution ("derived from GROMACS share/top/oplsaa.ff, © the GROMACS development team");
   - the conversion table;
   - rows read per section and rows emitted (any difference is exactly the duplicates the conflict rule merged);
   - what is not encoded, with reasons: `[ constrainttypes ]` (virtual-site constraints for MNH3/MNH2/MCH3A/MCH3B; molrs has no constraint category), the 6 improper macros, and `at.num` (§ 2).

### 2. Table shape — `molrs/src/ff/params/mod.rs`, `molrs/src/ff/params/oplsaa.rs`

- `OplsAtomRow` becomes `{ name, class, mass, charge, sigma, epsilon }`. `class` is documented as the GROMACS `bond_type`.
- The bond, angle and dihedral row types are unchanged.
- Constants: `OPLSAA_MIXING = "geometric"`, `OPLSAA_LJ_14` / `OPLSAA_COULOMB_14` (from fudge), `OPLSAA_NAME = "OPLS-AA"`.
- σ is GROMACS's own value, including 0 for the ε = 0 types. Under geometric mixing, σᵢⱼ = εᵢⱼ = 0 whenever either type has ε = 0.
- Wildcards are the empty endpoint.
- **`at.num` is not carried.** GROMACS has opls_009 wrong (a united-atom CH₂ given 7), and nothing in molrs reads a per-type atomic number. The element of a class derives from mass when needed (notes.md 2026-09-25 estimator entry, routed `/mol:debug`).

### 3. Rules move — `molrs/src/ff/params/oplsaa_typing.rs` (new)

`pub struct OplsRuleRow { name: &'static str, def: &'static str, overrides: &'static [&'static str] }` goes in `params/mod.rs`. It is the **static, compile-time rule record**.

The doc comments of `OplsRuleRow` and `OplsTypeRow` (`ff/typifier/opls/meta.rs:32`) state the relationship in both directions. `OplsTypeRow` is the **runtime** typing record (owned strings, plus class, explicit priority and layer). `embedded::typing_meta` builds one from each `OplsRuleRow`, taking `class` from the matching `OplsAtomRow` and `layer` from the table (0); `read_opls_typing_xml_str` builds them from XML.

`OPLSAA_TYPING: &[OplsRuleRow]` holds the 157 current defs and overrides, byte-identical (a verbatim move). `priority` is always `None` in the shipped data and is dropped. The header says: molrs-owned, hand-maintained, never touched by `mrs-gen-opls`; link 03 makes it Daylight SMARTS. The move has to happen in this link, or regeneration would overwrite hand-owned rules.

### 4. Embedded assembly — `molrs/src/ff/typifier/opls/embedded.rs`

- `try_force_field`: `lj/cut` params `mixing = OPLSAA_MIXING`.
- `typing_meta()` `expect`s the fallible `try_typing_meta()`, which joins `OPLSAA_TYPING` to `OPLSAA_ATOMS` by name. It is `Err` when a rule names no atom row or an override names no rule.
- The `expect` message names `ff::typifier::opls::embedded::tests::typing_meta_joins_every_rule`, and the notes.md 2026-09-25 scoped-amendment list gains that line.

### 5. LAMMPS writer — `molrs/src/ff/forcefield/writers/lammps.rs`

`pair_modify_line` (`:781`, also used by the combined branch at `:579`) writes `pair_modify mix <Mixing::UNDECLARED.name()>` for an `lj/cut` style without `mixing`. Before, it wrote nothing, so LAMMPS mixed geometrically while the kernel mixed arithmetically, for every undeclared force field.

The principled fix is to make `mixing` mandatory on every `lj/cut`, the way `coul/cut` constants are. It is recorded in notes.md as a routed `/mol:spec` with path:line `ff/potential/pair/lj_cut.rs:644-648` (task 7).

### 6. Behaviour changes disclosed

- **OPLS energies change.** Geometric mixing now agrees with GROMACS and LAMMPS. The σ 10→0 rows carry ε = 0, so no energy changes from those. Previously unreachable bonded rows now match.
- **Typed atoms carry GROMACS class names.**
- **Row changes.** The foyer-only rows (angles O_3-C-O_3, OS-C_2-OS; dihedrals NZ-CZ-CA-X, CT-OS-C_2-OS) disappear; H-N-CT_2-C appears.
- **LAMMPS exports.** Undeclared-mixing exports gain `pair_modify mix arithmetic`.

### Reuse decision

No librarian report was passed. Candidates come from the orchestrator's brief.

- `reuse GromacsTopFfReader` (link 01) as the generator's only parser.
- `reuse` the `embedded.rs` builders (the `expect` plus named-test pattern).
- `generalize OplsAtomRow`: its typing fields split out into the new `OplsRuleRow`.
- `reuse pair_modify_line`, extended through `Mixing::UNDECLARED`.
- `new — gen_opls_params` example: `scripts/gen_param_tables.py` is not restored, because a second (Python) GROMACS parser would duplicate link 01's reader.

## Files to create or modify

- `molrs/src/ff/params/mod.rs`
- `molrs/src/ff/params/oplsaa.rs` (regenerated)
- `molrs/src/ff/params/oplsaa_typing.rs` (new)
- `molrs/src/ff/typifier/opls/embedded.rs`
- `molrs/src/ff/typifier/opls/meta.rs`
- `molrs/src/ff/typifier/opls/mod.rs`
- `molrs/src/ff/forcefield/writers/lammps.rs`
- `molrs/examples/gen_opls_params.rs` (new)
- `molrs/Cargo.toml`
- `.cargo/config.toml`
- `.claude/notes/build.md`
- `CLAUDE.md`
- `.claude/notes/notes.md`

## Tasks

- [x] Write failing unit tests:
  - in `molrs/src/ff/typifier/opls/embedded.rs`: `lj/cut` declares geometric; hand-converted source-equivalence pins; `typing_meta_joins_every_rule`; class from GROMACS `bond_type`;
  - in `molrs/src/ff/forcefield/writers/lammps.rs`: undeclared `lj/cut` → `pair_modify mix arithmetic`
- [x] Restructure the row types in `molrs/src/ff/params/mod.rs` (`OplsAtomRow` without typing fields; new `OplsRuleRow` with the doc comment relating it to `OplsTypeRow`, and the reciprocal sentence in `molrs/src/ff/typifier/opls/meta.rs`). Move the 157 defs/overrides verbatim into `molrs/src/ff/params/oplsaa_typing.rs` (`OPLSAA_TYPING`)
- [x] Add the generator `molrs/examples/gen_opls_params.rs`. Add its `[[example]]`, `exclude = ["examples/"]` and `sha2` dev-dependency in `molrs/Cargo.toml` (updating the targets comment), and the `mrs-gen-opls` alias in `.cargo/config.toml`. Amend the legitimate-builds text in `.cargo/config.toml` and `.claude/notes/build.md` (one example unit on build set 1), and list `cargo mrs-gen-opls --gromacs <dir>` in `CLAUDE.md` § Build & Test Commands
- [x] Regenerate `molrs/src/ff/params/oplsaa.rs` from GROMACS v2026.3 (`42105e46…`) with `cargo mrs-gen-opls --gromacs <dir>`. Stub the table as empty slices of the new row types first, so the example builds. Pin the three SHA-256s. Then:
  - run `cargo mrs-build`, record the `target/debug/.fingerprint/molcrafts-molrs-*` directories containing a `lib-molrs*` file, rerun the generator, and confirm that set is unchanged and the only new directory holds `example-gen_opls_params*`;
  - confirm `cargo package --list -p molcrafts-molrs` lists no `examples/` path;
  - if any funct-3 row is not representable, or any section is refused, stop and route to the operator with the offending rows
- [x] Implement `mixing = OPLSAA_MIXING` and the fallible `try_typing_meta` join in `molrs/src/ff/typifier/opls/embedded.rs`; verify with `cargo mrs-test -- ff::typifier::opls`
- [x] Implement the undeclared-mixing `pair_modify` line in `molrs/src/ff/forcefield/writers/lammps.rs`; verify with `cargo mrs-test -- ff::forcefield::writers::lammps`
- [x] Update rustdoc per rustdoc style: the `params/mod.rs` module doc (OPLS-AA from GROMACS; rules molrs-owned) and `OPLSAATypifier::oplsaa` in `molrs/src/ff/typifier/opls/mod.rs` (provenance, geometric mixing), with a doctest asserting the library's `lj/cut` `mixing` is `"geometric"` (the regression example). In `.claude/notes/notes.md`, add `typing_meta` to the scoped-amendment list and record the "mixing mandatory on every `lj/cut`" follow-up (`ff/potential/pair/lj_cut.rs:644-648`, routed `/mol:spec`)
- [x] Run the full unit suite `cargo mrs-test`. The full gate runs once, in link 03

## Testing strategy

Unit tests next to the code, hand-derived. Filters: `ff::typifier::opls`, `ff::forcefield::writers::lammps`.

- **Source-equivalence pins.** Each GROMACS line is hand-copied and hand-converted (the notes.md 2026-09-25 pattern):
  - opls_135 → class `CT`, 12.011, −0.18, σ 3.5, ε 0.066;
  - opls_150 → `C=`, −0.115, 3.55, 0.076;
  - opls_155 → σ 0;
  - bond CT-HC (1.09, 680.0); bond C=-C= (1.46, 770);
  - angle CM-C=-C= (124°·π/180, 140);
  - dihedral HC-CT-CT-HC (0, 0, 0.3, 0);
  - relative tolerance 1e-12.
- **Mixing:** the `lj/cut` param `mixing` equals `"geometric"`.
- **Join:** `try_typing_meta()` is `Ok`, and opls_135 → `CT`. Test-local tables with a dangling rule or override are `Err`.
- **LAMMPS writer:** undeclared → `pair_modify mix arithmetic`; geometric → `pair_modify mix geometric`.
- **Regeneration and fingerprints:** runtime criteria (ac-003, ac-004), not unit tests.
- **Regression example:** the doctest on `OPLSAATypifier::oplsaa`, run in the chain-end gate.

## Out of scope

- Typing-rule content: link 03.
- OPLS impropers: routed by link 01.
- `[ constrainttypes ]`: not encoded; listed in the header.
- Found debt, routed `/mol:fix` (recorded here, not touched): 17 other `ff/params/*.rs` headers cite the deleted `scripts/gen_param_tables.py` (e.g. `gaff.rs:3`, `bccparm.rs:3`, `ff/typifier/estimate/tables.rs:4`).
- Mandatory `mixing` on every `lj/cut`: recorded by task 7.
- The estimator's all-caps class bug: routed `/mol:debug` (notes.md 2026-09-25).
