---
title: AMBER prmtop completeness 03 — 1-4 weights declared in LAMMPS, GROMACS and XML
status: code-complete
created: 2026-09-04
depends_on: [amber-prmtop-complete-02-forcefield]
---

# AMBER prmtop completeness 03 — 1-4 weights declared in LAMMPS, GROMACS and XML

## Summary

A force field read from an AMBER prmtop carries its 1-4 scaling on `ForceField::special_bonds`
(`lj_14 = 1/SCNB`, `coul_14 = 1/SCEE`), but every text format molrs writes today throws those two
numbers away: the LAMMPS `*.ff` writer emits no `special_bonds` line, the GROMACS `.top` writer emits
no `[ defaults ]` section, and the molrs XML writer prints `coulomb14scale`/`lj14scale` off a pair
style that never holds them, falling back to a hard-coded `0.5`. Symmetrically, the LAMMPS reader
ignores any `special_bonds` line it is given and *substitutes* the AMBER weights unconditionally, so a
CHARMM or DREIDING include silently comes back as AMBER. This phase makes the 1-4 weights round-trip
data in all three formats: writers declare them, the LAMMPS reader parses them and refuses a `.ff`
text that does not declare them, and the GROMACS reader parses and validates `[ defaults ]`. After
this phase, `prmtop → LAMMPS/GROMACS/XML → re-read` preserves `0.5` and `1/1.2` to writer precision
instead of reproducing them by coincidence.

## Domain basis

All quantities here are **dimensionless multiplicative weights** applied to a nonbonded pair
interaction between atoms separated by 1, 2 or 3 bonds. `SpecialBonds { lj: [f64; 3], coul: [f64; 3] }`
(`molrs/src/ff/forcefield/mod.rs:659-697`) stores them in LAMMPS order `[1-2, 1-3, 1-4]`; molrs
realises 1-2/1-3 exclusion by omitting those pairs from the neighbour list, so only the `[2]` (1-4)
entry reaches a kernel, projected as `lj14scale` / `coulomb14scale` by `Style::to_potential`
(`molrs/src/ff/potential/mod.rs:347-354`).

- **AMBER**: `SCEE = 1.2` and `SCNB = 2.0` are *divisors*; the multiplicative weights are
  `coul_14 = 1/1.2 = 0.8333333333333334` and `lj_14 = 1/2.0 = 0.5`. Cornell et al., *J. Am. Chem.
  Soc.* **117**, 5179 (1995), doi:10.1021/ja00124a002; GAFF keeps the same pair: Wang et al.,
  *J. Comput. Chem.* **25**, 1157 (2004), doi:10.1002/jcc.20035. `AmberPrmtopFfReader` already
  computes exactly this (`readers/prmtop.rs:200-211`).
- **LAMMPS `special_bonds`** (docs.lammps.org/special_bonds.html; Thompson et al., *Comput. Phys.
  Commun.* **271**, 108171 (2022), doi:10.1016/j.cpc.2021.108171). Forms accepted here:
  - presets — `amber` → lj `[0, 0, 0.5]`, coul `[0, 0, 5/6]`; `charmm` → both `[0, 0, 0]` (1-4 is
    carried by the `charmm` pair style, not by weights); `dreiding` → both `[0, 0, 1.0]`;
    `fene` → both `[0, 1.0, 1.0]`;
  - `lj w12 w13 w14 coul w12 w13 w14`, or `lj …` alone, or `coul …` alone, or a bare triple
    `w12 w13 w14` applied to both kinds;
  - trailing `angle yes|no` / `dihedral yes|no` are tolerated (they change how LAMMPS *derives* the
    exclusion topology, not the weights — molrs derives 1-2/1-3/1-4 from `Topology`);
  - any other token is an error.
  The manual prints the `amber` Coulomb weight rounded as `0.8333`; molrs stores `5/6`, which is the
  same number AMBER's `1/SCEE` produces, so the prmtop → LAMMPS → prmtop identity is exact in store.
  **LAMMPS's own default when the command is absent is `lj/coul 0.0 0.0 0.0`** — full 1-4 exclusion,
  not AMBER's weights. That divergence is why an undeclared `.ff` cannot be given a default here.
- **GROMACS `[ defaults ]`** (GROMACS manual, *Topology file / topology file formats*; Abraham et al.,
  *SoftwareX* **1-2**, 19 (2015), doi:10.1016/j.softx.2015.06.001): columns
  `nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ`. `nbfunc = 1` is Lennard-Jones, `comb-rule = 2` is
  the σ/ε Lorentz-Berthelot form molrs stores, `gen-pairs = yes` generates 1-4 pairs from the normal
  LJ parameters scaled by `fudgeLJ`. `fudgeLJ`/`fudgeQQ` apply to **1-4 pairs only** — GROMACS has no
  1-2/1-3 weights (those are exclusions), so they map onto `lj[2]` / `coul[2]` and nothing else. An
  AMBER port writes `1 2 yes 0.5 0.8333`.
- **OpenMM ForceField XML**: `<NonbondedForce coulomb14scale="0.8333333333333334" lj14scale="0.5">`
  — multiplicative 1-4 weights with exactly the `SpecialBonds[2]` semantics (this is the literal
  header of OpenMM's shipped `amber14` XMLs). molrs's own XML reader path already consumes them
  (`readers/opls.rs:115-117,154-158`). OpenMM's own default when an attribute is **absent** is
  `1.0` (unscaled), so the `0.5` molrs currently substitutes (`readers/opls.rs:116-117`) matches
  neither OpenMM nor any force field — which is why § 8 makes the attributes required rather than
  swapping one default for another.
- **Cross-format identity checked by this phase**: `1/1.2` written at the writers' default precision 6
  prints `0.833333` in all three formats and reads back to within `1e-6` of `5/6`; `1/2.0` prints
  `0.500000` and reads back exactly. Precision 6 is the existing file-number convention of both
  writers (`fmt_num` / `fmt_f`); the loss is stated, bounded and tested rather than hidden.

## Design

**Entities touched** — no new public type, no new public function, no changed signature. The unit of
truth is the existing `ForceField::special_bonds()` / `set_special_bonds()` pair; this phase makes
four boundaries read and write it.

1. **`LammpsFfWriter::write_str`** (`writers/lammps.rs:182-198`) always emits, immediately after the
   `units` line (and still emitted when `skip_units` is set — LAMMPS accepts `special_bonds` at any
   point before the run):
   `special_bonds lj <w12> <w13> <w14> coul <w12> <w13> <w14>`, each number through the existing
   `fmt_num(v, opts.precision)`. Explicit numeric triples, never a preset name: a preset is a claim
   about a whole force field, the weights on the `ForceField` are the fact. The two `lines.push`
   calls stay **inline** in `write_str` (Shape check #3: one call site, no extraction).
2. **`LammpsFfReader`** (`readers/lammps.rs`): a private free fn
   `parse_special_bonds(rest: &[&str], where_) -> Result<SpecialBonds, String>` — a sibling of the
   existing `require_pair_style` / `require_kernel` per-keyword helpers, kept private and *not*
   promoted onto `SpecialBonds` (architect 🟢6). It replaces the no-op `"special_bonds"` arm at
   `readers/lammps.rs:205`. The `AMBER_LJ14` / `AMBER_COUL14` constants (`:50-53`) and their
   unconditional application (`:125-128`) are **deleted**.
3. **Absent declaration is an `Err`.** A `.ff` text with no `special_bonds` line fails with a message
   naming the missing declaration, in the same voice as this reader's existing
   `unknown LAMMPS keyword` contract. Rationale is CLAUDE.md:304: a style that omits
   `coulomb`/`dielectric`/`coulomb14scale` is an `Err`, never a silent default — and because
   `SpecialBonds` is *projected* into `coulomb14scale` (`potential/mod.rs:347-354`), an invented
   `[0,0,1]` would sail straight through the `required(style_params, "coulomb14scale")` gate at
   `pair/coul_cut.rs:192` and silently change the physics. LAMMPS's own default (0 0 0) differs from
   the AMBER weights molrs used to invent, so there is no defensible default to pick.
   **Behaviour break, named:** a line-less `.ff` that used to yield the AMBER weights is now an error.
   Documented in the module rustdoc (replacing the AMBER claim at `readers/lammps.rs:40-41`) and in
   Follow-up (molpy).
4. **`read_data_coeffs` keeps today's physics, explicitly.** LAMMPS DATA files carry no
   `special_bonds`, and `read_data_coeffs` (`readers/lammps.rs:96-113`) already synthesizes its own
   command text (`units …`, `pair_style lj/cut 10.0`, `bond_style harmonic`, …). It gains one more
   synthesized line — `special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333333333333334` — so data-file
   reads keep the 0.5 / 5/6 they have always produced. Its rustdoc states that this is the data-file
   default the reader has always assumed, exactly as it already assumes `pair_style lj/cut 10.0`.
   Letting the caller pass an explicit declaration instead is routed to a follow-up `/mol:fix`
   (Follow-up), not left unsaid.
5. **`GromacsTopFfWriter::write_str`** (`writers/gromacs.rs:46-48`) emits `[ defaults ]` as the first
   section (GROMACS requires it before any type table), with a `;` column header and the row
   `1  2  yes  <fudgeLJ>  <fudgeQQ>` where `fudgeLJ = ff.special_bonds().lj_14()` and
   `fudgeQQ = coul_14()`, both through `fmt_f`. Inline in `write_str`; no new symbol.
6. **`GromacsTopFfReader`** gains a private `parse_defaults_section` (sibling of the existing
   `parse_pair_section` / bonded section parsers) called from `build_forcefield`
   (`readers/gromacs.rs:254`). It sets `lj[2] = fudgeLJ`, `coul[2] = fudgeQQ` and errs on
   `nbfunc != 1` (molrs stores only LJ), `comb-rule != 2` (molrs stores σ/ε), and — symmetrically with
   the writer, architect 🟡5 — on `gen-pairs` other than `yes`, because neither side handles an
   explicit `[ pairtypes ]` table. Missing `fudgeLJ`/`fudgeQQ` columns are an `Err` naming the column,
   not the format's 1.0 default.
   **Absent `[ defaults ]` section** (architect pass-2 🔴): a molecule-only `.itp` fragment
   legitimately carries no `[ defaults ]` (GROMACS supplies it from the including `.top`; the
   existing reader test at `readers/gromacs.rs:545` is exactly such a fragment), so a blanket `Err`
   would delete a supported read. But leaving `SpecialBonds::default()` in place when the topology
   *does* declare 1-4 pairs would let `Style::to_potential` fabricate `coulomb14scale = 1.0` past the
   `required(...)` gate — the same gate-defeat § 3 forbids. The gate is therefore **`[ pairs ]`-keyed,
   in this phase**: `build_forcefield` already holds `sections.get("pairs")`; when a `[ pairs ]`
   section is present and `[ defaults ]` is absent, the reader returns `Err` naming the missing
   declaration; when neither is present (a fragment with no 1-4 pairs) the weights stay at the
   default and the rustdoc says so. The molrs writer always emits the section, so molrs-produced
   files always declare their weights.
7. **`XmlForceFieldWriter::write_str`** (`writers/xml.rs:285-291`) reads `coulomb14scale` /
   `lj14scale` off `style.params` with `unwrap_or(0.5)`, but those weights are owned by `ForceField`
   (`mod.rs:659-697`) and the XML *reader* already lands them there
   (`molrs/src/ff/forcefield/xml.rs:457-466` — the architect labelled this `readers/xml.rs:461-465`;
   the molrs-native XML reader lives at `forcefield/xml.rs`, there is no `readers/xml.rs`). The writer
   now emits `coulomb14scale = ff.special_bonds().coul_14()` and `lj14scale = lj_14()` with **no**
   `unwrap_or`. A write → `read_forcefield_xml_str` → same-`SpecialBonds` test closes the loop (the
   emitted `<NonbondedForce>` routes the read through the OpenMM-pack branch at
   `forcefield/xml.rs:72-85` → `OplsXmlReader`, which sets `SpecialBonds` at `readers/opls.rs:154-158`).
8. **`readers/opls.rs:116-117` — the reader half of the same hole, fixed here (architect pass-2 🔴).**
   That reader takes `coulomb14scale` / `lj14scale` off `<NonbondedForce>` with `unwrap_or(0.5)`, the
   identical silent default § 7 removes from the writer, on the very function the § 7 round-trip test
   routes through. It is two lines, the only in-repo fixture (`readers/opls.rs:517`) already declares
   both attributes, and no shipped XML data exists (`molrs/data/` was removed) — local and
   stage-allowed, so the Iron law's fix-now branch applies. Both attributes become **required**
   (`require_f64`, the file's existing helper), a missing one is an `Err` naming the attribute, and a
   unit test pins it. The previously proposed `/mol:fix` route is withdrawn.

**No public surface change, therefore no binder change.** `LammpsWriteOptions` gains **no**
`skip_special_bonds` field (architect 🔴1): the writer always emits the line, so the option would have
zero callers (CLAUDE.md:63-65, Shape check #3). That also keeps `molrs-python/src/ff/mod.rs` untouched
— its three call sites construct `LammpsWriteOptions` as **full struct literals** with no
`..Default::default()` (`molrs-python/src/ff/mod.rs:1723-1734`, `:1776`, `:1829`), so any new field
would have broken the binder build. **Confirmed: `molrs-python/src/ff/mod.rs` is not in Files**
(architect 🔴3).

### Architect findings resolved

- 🔴1 → no `skip_special_bonds`; line is unconditional; binder untouched (§ "No public surface change").
- 🔴2 → absent `.ff` declaration is an `Err` (§ 3); `read_data_coeffs` pushes an explicit line + rustdoc
  + `/mol:fix` follow-up (§ 4); Rust test at `readers/lammps.rs:875-881` rewritten to feed an explicit
  line; two molrs-python fixtures updated (test-only, listed in Files); rustdoc at `:40-41` replaced.
- 🔴3 → confirmed not needed; `molrs-python/src/ff/mod.rs` absent from Files.
- 🔴4 → `writers/xml.rs` folded into this phase with a round-trip test (§ 7); task count held at 10 by
  merging tests-per-format and impl-per-format.
- 🟡5 → `gen-pairs` parsed and required `yes`, writer emits `yes` (§ 6).
- 🟢6 → `parse_special_bonds` stays a private free fn (§ 2).

### Reuse decision

No `librarian_report` was supplied with this re-draft, so reuse was resolved directly against the
symbols read in-tree; there are no unresolved candidates.

- `reuse` `ForceField::special_bonds()` / `set_special_bonds()` / `SpecialBonds::{lj_14, coul_14}`
  (`mod.rs:687-726`) — the single owner of these weights; all four boundaries go through it.
- `reuse` `fmt_num` (`writers/lammps.rs:1133`) and `GromacsTopFfWriter::fmt_f`
  (`writers/gromacs.rs:40-42`) for number formatting — no new formatter.
- `reuse` the reader's existing `strip_comment` / `where_()` line-context closure / `parse_f64`
  helpers for the new `special_bonds` arm; error strings follow the existing
  `"{where_}: unknown LAMMPS keyword `{other}`"` shape so the new code reads like the old code.
- `reuse` `parse_sections_text` + the `sections: HashMap<String, Vec<String>>` contract in the GROMACS
  reader — `[ defaults ]` is already captured by the generic section scanner; only a consumer is new.
- `new — parse_special_bonds` — no existing helper parses a weights triple; it is a private free fn
  matching the file's one-helper-per-keyword pattern, not a new public symbol.
- `new — parse_defaults_section` — no existing section parser covers `[ defaults ]`; private, sibling
  of `parse_pair_section`.

## Files to create or modify

- `molrs/src/ff/forcefield/writers/lammps.rs` — emit the `special_bonds` line after `units`; module
  rustdoc; new/updated `#[cfg(test)]` tests. **Every inline `.ff` fixture in this test module gains
  a `special_bonds` line** — `MINI` (`:1150`) and the local `SRC` texts at `:1274`, `:1353`, `:1392` —
  because the reader half of the round trip now refuses an undeclared text; this is a wall of
  mechanical fixture edits, named here so no assertion is weakened to get past it.
  **No** new `LammpsWriteOptions` field.
- `molrs/src/ff/forcefield/readers/lammps.rs` — `parse_special_bonds`; replace the no-op arm at `:205`;
  delete `AMBER_LJ14`/`AMBER_COUL14` (`:50-53`) and their application (`:125-128`); `Err` on an absent
  declaration; explicit synthesized line in `read_data_coeffs` (`:96-113`) + its rustdoc; module
  rustdoc `:40-41` rewritten; `#[cfg(test)]` tests — the assertions at `:875-881` are rewritten and
  **all eleven inline `read_str` texts in the module gain a `special_bonds` line** (same reason).
- `molrs/src/ff/forcefield/writers/gromacs.rs` — emit `[ defaults ]` first; module rustdoc;
  `#[cfg(test)]` tests (module exists at `:355`).
- `molrs/src/ff/forcefield/readers/gromacs.rs` — `parse_defaults_section` + validation; module rustdoc
  states the absent-section behaviour; `#[cfg(test)]` tests (module exists at `:545`).
- `molrs/src/ff/forcefield/writers/xml.rs` — `<NonbondedForce>` 1-4 attributes from
  `ff.special_bonds()`, no `unwrap_or`; **new** `#[cfg(test)]` module (this file has none today).
- `molrs/src/ff/forcefield/readers/opls.rs` — `coulomb14scale` / `lj14scale` required via
  `require_f64` (`:116-117`), `Err` naming a missing attribute; one `#[cfg(test)]` test.
- `molrs-python/tests/test_forcefield_lammps_reader.py` — **test-only edit**: the two inline `.ff`
  fixtures (`_FF` at `:15-31`, the local `src` at `:70-78`) gain a `special_bonds` line so they still
  parse under the changed reader contract. No production binder code changes.
- `regressions/amber-prmtop-complete-03-lammps.py` (new)

## Tasks

- [ ] Write failing unit tests for the LAMMPS `special_bonds` contract in
      `molrs/src/ff/forcefield/readers/lammps.rs` and `writers/lammps.rs` `#[cfg(test)]` modules
      (presets, explicit/partial/bare triples, trailing `angle|dihedral`, unknown token, absent-line
      `Err`, `read_data_coeffs` default, writer line placement, round trip), and add the
      `special_bonds` line to every existing inline `.ff` fixture in both modules (eleven reader
      texts; writer `MINI` + the three local `SRC` texts) without weakening any assertion
- [ ] Implement the unconditional `special_bonds` line in `LammpsFfWriter::write_str`
      (`molrs/src/ff/forcefield/writers/lammps.rs`), emitted right after `units`, with the module
      rustdoc example updated — add no `LammpsWriteOptions` field
- [ ] Implement `parse_special_bonds` in `molrs/src/ff/forcefield/readers/lammps.rs`, delete
      `AMBER_LJ14`/`AMBER_COUL14` and their unconditional application, `Err` on an absent declaration,
      push the explicit data-file line in `read_data_coeffs`, and rewrite the module + `read_data_coeffs`
      rustdoc per `<doc.style>` (weights dimensionless)
- [ ] Write failing unit tests for the GROMACS `[ defaults ]` contract and the molrs-XML 1-4 round trip
      in `writers/gromacs.rs`, `readers/gromacs.rs`, and a new `#[cfg(test)]` module in `writers/xml.rs`
- [ ] Implement the `[ defaults ]` section in `GromacsTopFfWriter::write_str`
      (`molrs/src/ff/forcefield/writers/gromacs.rs`) as the first section, `1 2 yes fudgeLJ fudgeQQ`
- [ ] Implement `parse_defaults_section` in `molrs/src/ff/forcefield/readers/gromacs.rs` with
      `nbfunc != 1` / `comb-rule != 2` / `gen-pairs != yes` errors and the `[ pairs ]`-keyed
      missing-`[ defaults ]` `Err`, and document the absent-section behaviour in the module rustdoc
- [ ] Implement `<NonbondedForce coulomb14scale/lj14scale>` from `ff.special_bonds()` in
      `XmlForceFieldWriter::write_str` (`molrs/src/ff/forcefield/writers/xml.rs`), removing both
      `unwrap_or(0.5)` defaults, and make the same two attributes required in
      `molrs/src/ff/forcefield/readers/opls.rs` (`require_f64`, `Err` naming the attribute)
- [ ] Update the two inline `.ff` fixtures in `molrs-python/tests/test_forcefield_lammps_reader.py`
      to declare `special_bonds` (test-only; follows the changed reader contract)
- [ ] Add regression example `regressions/amber-prmtop-complete-03-lammps.py` (public API only;
      hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Per CLAUDE.md ("Prefer unit tests next to the code … there is **no** `molrs/tests/` integration-binary
tree"), every Rust test is an inline `#[cfg(test)]` module in the file under test; the project rule
overrides the generic `tests/` mirror layout. Green for one file =
`cargo test -p molcrafts-molrs --lib --features full,filesystem <test_name>`. All expected values are
hard-coded literals; no LAMMPS, GROMACS, AmberTools or OpenMM at test time.

**`readers/lammps.rs`** (one function under test: `LammpsFfReader::read_str` / `read_data_coeffs`)

- happy: `special_bonds amber` → `lj == [0.0, 0.0, 0.5]`, `coul_14()` within `1e-12` of `5/6`.
- happy: `special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333333333333334` → same values exactly.
- edge: `special_bonds 0.0 0.0 0.5` (bare triple) → both kinds `[0.0, 0.0, 0.5]`.
- edge: `special_bonds lj 0.0 0.0 0.5` alone → `coul` untouched at `[0.0, 0.0, 1.0]`.
- edge: presets `charmm` → both `[0,0,0]`; `dreiding` → both `[0,0,1.0]`; `fene` → both `[0,1.0,1.0]`.
- edge: `special_bonds amber angle yes dihedral no` parses; `special_bonds amber banana` → `Err`.
- edge: a `.ff` text without any `special_bonds` line → `Err` whose message contains `special_bonds`
  (replaces the AMBER assertions at `readers/lammps.rs:875-881`, which are rewritten to feed an
  explicit line).
- domain: `read_data_coeffs` on a `Pair Coeffs` fragment → `lj_14() == 0.5` and
  `coul_14()` within `1e-12` of `5/6` (data-file physics unchanged).

**`writers/lammps.rs`** (`LammpsFfWriter::write_str`)

- happy: output contains `special_bonds lj 0.000000 0.000000 0.500000 coul 0.000000 0.000000 0.833333`,
  and it appears **after** the `units` line and **before** the first `pair_coeff` / `bond_coeff` /
  `angle_coeff` / `dihedral_coeff` / `improper_coeff` line — the ordering LAMMPS needs (weights before
  coefficients are applied), not a fixed line index (LAMMPS accepts the command anywhere before the run).
- edge: with `skip_units: true` the `special_bonds` line is still present.
- domain: write → `LammpsFfReader::read_str` → `lj_14() == 0.5` and `|coul_14() − 5/6| ≤ 1e-6`
  (the precision-6 bound stated in Domain basis).

**`writers/gromacs.rs`** (`GromacsTopFfWriter::write_str`)

- happy: `[ defaults ]` is the first `[ … ]` header and its data row's whitespace-split tokens are
  exactly `["1", "2", "yes", "0.500000", "0.833333"]`.

**`readers/gromacs.rs`** (`GromacsTopFfReader::read_str`)

- happy: `[ defaults ]` with `1 2 yes 0.5 0.8333333333333334` → `lj[2] == 0.5`, `coul[2]` within
  `1e-12` of `5/6`; `lj[0..2]`/`coul[0..2]` stay `0.0`.
- edge: `nbfunc = 2` → `Err`; `comb-rule = 3` → `Err`; `gen-pairs = no` → `Err`; a row missing
  `fudgeQQ` → `Err` naming the column.
- edge: a fragment with neither `[ defaults ]` nor `[ pairs ]` still parses and keeps
  `SpecialBonds::default()` (`[0,0,1]`) — the documented, deliberate behaviour; the existing reader
  test at `:545` stays green.
- edge: a topology with a `[ pairs ]` section and no `[ defaults ]` → `Err` whose message contains
  `defaults` (the `[ pairs ]`-keyed gate).

**`readers/opls.rs`** (`OplsXmlReader`)

- edge: a `<NonbondedForce>` without `coulomb14scale` (or without `lj14scale`) → `Err` naming the
  missing attribute; the existing fixture at `:517`, which declares both, still parses.

**`writers/xml.rs`** (new module, `XmlForceFieldWriter::write_str`)

- happy: a `ForceField` with `special_bonds = {lj: [0,0,0.5], coul: [0,0,5/6]}` and one `lj/cut` pair
  type emits `<NonbondedForce coulomb14scale="0.833333" lj14scale="0.500000">`.
- scope note: `writers/xml.rs:285-291` emits one `<NonbondedForce>` per pair style, so a force field
  with two pair styles emits two blocks. After this phase both carry the identical force-field-level
  values (benign); a second assertion on a two-pair-style force field pins that the values agree.
  Collapsing to one block is a writer-shape change outside this phase.
- domain: write → `read_forcefield_xml_str` → `lj_14() == 0.5`, `|coul_14() − 5/6| ≤ 1e-6`; a force
  field whose `coul_14()` is `1.0` must **not** come back as `0.5` (proves the old `unwrap_or` is gone).

**Regression example** — `regressions/amber-prmtop-complete-03-lammps.py`, public API only, run with
`python regressions/amber-prmtop-complete-03-lammps.py`. It embeds a small hand-written prmtop literal
(`%FLAG POINTERS` / `MASS` / `AMBER_ATOM_TYPE` / `ATOM_TYPE_INDEX` / `SCEE_SCALE_FACTOR` /
`SCNB_SCALE_FACTOR` / LJ `ACOEF`/`BCOEF` / `NONBONDED_PARM_INDEX`; no AmberTools, no fixture fetch)
and asserts, against hard-coded goldens:

1. `molrs.ff.write_lammps_forcefield_str(molrs.ff.read_amber_prmtop_ff_str(PRMTOP))` contains the line
   `special_bonds lj 0.000000 0.000000 0.500000 coul 0.000000 0.000000 0.833333`, immediately after the
   `units real` block;
2. re-reading that text with `molrs.ff.read_lammps_forcefield_str` and re-writing reproduces the same
   `special_bonds` line byte-for-byte (the reader consumed the declaration, it did not default);
3. deleting the `special_bonds` line and re-reading raises `ValueError` mentioning `special_bonds`;
4. `molrs.ff.write_gromacs_top_ff_str(ff)` has `[ defaults ]` as its first section with tokens
   `["1", "2", "yes", "0.500000", "0.833333"]`, and the same tokens survive
   `read_gromacs_top_ff_str` → re-write;
5. the goldens are annotated with their provenance — AMBER `SCEE = 1.2` → `1/1.2 = 0.8333333333333334`
   (printed `0.833333` at precision 6, tolerance `1e-6`) and `SCNB = 2.0` → `0.5` exactly.

The Python `ForceField` exposes no `special_bonds` accessor and this phase adds none, so every
regression assertion is on emitted text and on text round trips — deliberately, to keep the binder
surface unchanged.

## Out of scope

- Adding a `special_bonds` accessor to the Python/WASM/C `ForceField` surface. Not needed by any
  assertion here, and adding it would put this phase on the binder seam (architect 🔴3).
- Preset **names** on write. The writer emits explicit numerics; recognising that a weight set equals
  `amber` and printing the preset back is cosmetic and lossy.
- 1-2 / 1-3 weight *application*. molrs still realises those by neighbour-list omission; the parsed
  `[0]`/`[1]` entries are stored (and now round-trip) but no kernel consumes them.
- GROMACS `[ pairtypes ]` / explicit 1-4 tables, and `gen-pairs no`. Rejected loudly instead
  (architect 🟡5).
- GROMACS `comb-rule 1|3` and `nbfunc 2` (Buckingham). Rejected loudly.
- `readers/prmtop.rs`'s `DEFAULT_SCEE = 1.2` / `DEFAULT_SCNB = 2.0` for an absent section
  (`readers/prmtop.rs:38-41`). Reviewed and left alone: unlike the LAMMPS case, these are the AMBER
  *format's own* documented defaults for its own files, not a foreign convention imported into
  another format. Recorded here so the decision is on the record.
- Anything in phases 01, 02, 04 of the `amber-prmtop-complete` chain.

## Follow-up (molpy)

- **Behaviour break to mirror downstream**: a LAMMPS `*.ff` text without a `special_bonds` line is now
  an error instead of yielding the AMBER weights. molpy code (and any user script) that hands molrs a
  bare GAFF include must add `special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333333333333334`.
  molrs-written includes always carry the line, so writer-produced files are unaffected. Ship molrs
  with a tag before molpy pins the new minor (CLAUDE.md, *Release before molpy*).
- **Second behaviour break, OpenMM-style XML**: a `<NonbondedForce>` that omits `coulomb14scale` or
  `lj14scale` is now an error instead of silently reading as `0.5`. OpenMM's shipped force-field
  XMLs always declare both, and molrs-written XML now always does too; a hand-written or trimmed
  XML must add them.
- `/mol:fix` — let `LammpsFfReader::read_data_coeffs` take an explicit `special_bonds` declaration from
  its caller instead of the synthesized AMBER default it has always assumed.
- Downstream 1-4 parity check: once molpy pins the new minor, confirm its GROMACS/LAMMPS export tests
  read the weights out of the file rather than re-deriving them.
