---
title: OPLS-AA typing rules are molrs-owned Daylight SMARTS with pairwise override ranking
slug: opls-gromacs-03-rules
status: code-complete
created: 2026-09-25
chain: opls-gromacs (01-gromacs-io → 02-table → 03-rules)
depends_on: [opls-gromacs-02-table]
---

# OPLS-AA typing rules are molrs-owned Daylight SMARTS with pairwise override ranking

## Summary

The shipped OPLS typing rules were written for foyer, whose matcher ignores bond order. In molrs's standard SMARTS an unmarked bond means single-or-aromatic, so every rule that crosses a `=` bond never matches.

- Alkene and carbonyl carbons stay untyped.
- Aromatic atoms are cased inconsistently. Chlorobenzene's Cl is typed as chloride (net −0.82), and pyridine, pyrimidine and pyrrole cannot be typed.
- `[Cl,C,H]` reads its `H` as a hydrogen count.
- The matcher never perceives aromaticity.
- A later dependency level overwrites a type that declares it overrides the newcomer.
- Overrides are collapsed into a single number.

After this link:

- The rules are molrs-owned Daylight SMARTS with explicit bonds, and opls_150/opls_178 are added.
- Aromaticity is perceived on a private copy of the graph.
- Ranking is pairwise: a type that overrides another, or sits on a higher layer, always wins, and no later level replaces it.
- Fifteen hand-derived molecules type exactly and are neutral.

As the chain's last link, this one also moves molrs-python to the settled GROMACS I/O API. Both readers gain `skip_directives`, so Python can read the pinned `oplsaa.ff`, which contains `[ constrainttypes ]`. The link then runs the chain's single full gate.

## Domain basis

- **SMARTS** (Daylight Theory Manual ch. 4, https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html):
  - an unmarked bond is single-or-aromatic;
  - `~` is any bond;
  - `H<n>` inside brackets is a total-H count, and a hydrogen atom is `[#1]`;
  - `X<n>` is total connections;
  - `r<n>` means the smallest ring containing the atom has size n. molrs follows RDKit/Daylight here, not foyer's any-chordless-cycle reading.
- **OPLS-AA types and charges:** Jorgensen et al. 1996, DOI 10.1021/ja9621760. The pinned GROMACS `ffnonbonded.itp` and `atomtypes.atp` give each type's chemistry:
  - opls_150 `C=` "diene =CH-CH=" (−0.115);
  - opls_178 `C=` "diene =CR-CR=" (0.000);
  - same LJ as opls_142 / opls_141.
- **Neutrality invariant:** each golden is neutral, so the charges of its types sum to 0 (|Σq| < 1e-9).

## Design

### 0. Chain rules

As in `opls-gromacs-01-gromacs-io` § 0. This link is the last one: it owns the only binder change (§ 7) and the chain-end full gate (§ 8).

### 1. Rule conventions — `molrs/src/ff/params/oplsaa_typing.rs`

Every rule in `OPLSAA_TYPING` follows these conventions, and the header states them:

1. **Explicit bonds.** Every bond is written `-`, `=`, `#` or `:`. `~` is used only where a type deliberately spans resonance forms (carboxylate O, nitro O, sulfonyl O), each with a trailing comment.
2. **Hydrogens.** Hydrogen atoms are `[#1]`; `H<n>` appears only as a count.
   - `[!H]` (opls_154, opls_467) becomes `[!#1]`.
   - `[Cl,C,H]` (opls_152/153, `oplsaa.rs:188-189`) becomes `[Cl,C,#1]`.
   - The bare leading `H` of foyer rules becomes `[#1]`.
3. **Aromatic case.** Aromatic atoms are lowercase, aliphatic uppercase. The case defects fixed this way include:
   - opls_264 `[Cl;X1]-[c;%opls_263]`, opls_719 `F-[c;%opls_718]`, opls_534 `[#1]-[c;%opls_531]`, opls_167 `[O;X2](-[#1])-[c;%opls_166]`;
   - opls_520/530 and their rings to `n` / `c` with `:` bonds;
   - pyrrole opls_542–547 to aromatic `n` / `c`.
4. **Monatomic ions.** Written `X0` (opls_401 `[Cl;X0]`, opls_406 `[Li;X0]`).
5. **Preconditions.** Explicit hydrogens; smallest-ring `r<n>`.
6. **Comments.** Each rule carries its GROMACS `.atp` description as a comment.
7. **Overrides.** Kept from the moved table. Changes, and the two new rules, are listed in the header.

Carried from link 02: opls_927 lost its override of opls_928 because opls_928 (GROMACS class `CZ`) has no rule. Decide here whether opls_928 gets a Daylight rule (and restore the override) or stays rule-less; record the choice in the header.

New rules:

- opls_150 `[C;X3;H1](=[C;X3])-[C;X3]=[C;X3]`, overrides opls_142.
- opls_178 `[C;X3;H0](=[C;X3])(-[#6])-[C;X3]=[C;X3]`, overrides opls_141.

Decision, open to the grill: `C=` is only for conjugated dienes, so methacrylate's α-carbon stays opls_141.

### 2. Pairwise dominance — `molrs/src/ff/typifier/opls/layered.rs`, `meta.rs`

A private `Dominance` relation is built once in `LayeredTypingEngine::build`:

- `dominates(a, b)` holds iff `layer(a) > layer(b)`, or the layers are equal and `(a, b)` is in the transitive closure of declared overrides.
- Build is `Err` for an overrides cycle (naming its members) and for an override naming a type absent from the typing metadata (naming both). The engine is built inside `r#match` (`typing.rs:88`, routed debt), so construction must check too: `OPLSAATypifier::from_xml_str`, already fallible, validates the overrides of its parsed metadata and is `Err` for a dangling override, so an invalid typifier cannot be constructed. The shipped table can never hit it, as link 02's join test proves.

It is used in two places:

- **Within a level.** Winners are W = {c ∈ C : no d ∈ C dominates c}. Within W the order is explicit `priority` (absent = 0), then `num_query_atoms`, then earlier sorted name. This is order-independent and deterministic.
- **Across levels** (`layered.rs:182-190`): the new level's winner n replaces the current assignment t iff `!dominates(t, n)`. Circular-group iteration uses the same merge.

`OplsTypingMeta::priorities` and `LAYER_PRIORITY_STRIDE` (`meta.rs:26`, `:105-129`; re-export `opls/mod.rs:50`) are deleted outright. `OplsTypeRow.priority` / `layer` stay, for the XML path.

### 3. Aromaticity on a private copy — `molrs/src/ff/typifier/opls/typing.rs`

`typify_atoms` runs the engine on `Perceive::new().find_aromaticity(mol)`, a clone with preserved `AtomId`s, and stamps against the caller's ids. Returned bond orders are the caller's: Kekulé stays Kekulé. The same applies to `from_xml_str` typifiers.

### 4. Context-label dependencies — `molrs/src/ff/typifier/opls/deps.rs`

`OplsDependencyAnalyzer::new` (`:72-76`) keeps a label as a dependency iff it names a type in the metadata. The `starts_with("opls_")` filter goes. In OPLS-style typing a context label is only ever a type name, and a hard-coded prefix silently drops the dependencies of any XML rule set whose types are not named `opls_*`.

### 5. Tests that named butadiene — `molrs/src/ff/typifier/opls/mod.rs`

`strict_typify_names_every_untyped_atom` and `non_strict_typify_accepts_a_partly_typed_molecule` move to hand-built methylsilane: C 0, Si 1, H on C 2..=4, H on Si 5..=7. By hand from the rules, no rule types C–Si, Si or H–Si, while `[#1]-[C;X4]` (opls_140) types atoms 2..=4. The strict error therefore names atoms 0, 1, 5, 6, 7 and none of 2..=4.

### 6. Behaviour changes disclosed (Rust)

- Shipped OPLS typing changes for molecules with multiple bonds, aromatic rings, alkyl chlorides or override-dependent atoms. Strict OPLS no longer refuses every C=C / C=O molecule.
- `from_xml_str` typifiers now perceive aromaticity, rank by dominance, and refuse dangling overrides. Foyer-dialect XML rule sets that wrote aromatic ring atoms uppercase stop matching them.

### 7. Python binder — `molrs-python/src/ff/mod.rs`, `molrs-python/python/molrs/_lib.pyi`

This is the chain's one binder change. From link 01 until here, molrs-python does not compile, because it calls the deleted Rust free functions.

- **Readers.** `read_gromacs_top_ff(path, include=False, *, skip_directives=())` and `read_gromacs_top_ff_str(text, include=False, *, skip_directives=())` (`:1958`, `:1970`) build `GromacsTopFfReader::new().with_include(include)` and call `.with_skipped_directive(name)` for each entry. This restores binding symmetry with the Rust reader: without it, Python cannot read the pinned `oplsaa.ff`.
- **Public wrappers.** Users call `molrs.ff.read_gromacs_top_ff` / `read_gromacs_top_ff_str`, exported (`molrs-python/python/molrs/ff/__init__.py:59-64`) from the pure-Python wrappers in `molrs-python/python/molrs/ff/forcefield.py:578-601`. Both wrappers gain keyword-only `skip_directives: Sequence[str] = ()` and pass it through to `_lib`; their four docstrings (`:579-584`, `:588`, and the write wrappers) are rewritten for the directive-only model, naming Python's topology reader (not the Rust `read_top`) for molecule sections.
- **Writers.** `write_gromacs_top_ff` / `write_gromacs_top_ff_str` (`:1988`, `:1999`) call `GromacsTopFfWriter::new().with_precision(p).write(…)` / `.write_str(…)`.
- **Docstrings** (`:1951-1956`) and the `.pyi` stubs (`_lib.pyi:3876`, `:3880`, `:3930`, `:3934`) describe the directive model: what is read, what is refused, that molecule sections need `read_top` or a skip, and what `skip_directives` does.
- The Python names are unchanged, so molpy's `io/forcefield/top.py` keeps importing; its behaviour follows in molpy's chain.

### 8. Chain-end gate

1. Bring the binders to the settled API. Only molrs-python changes; molrs-wasm, molrs-capi, molrs-ffi and molrs-cxxapi must build unchanged.
2. Commit. The pre-commit hook runs rustfmt and clippy.
3. Run `prek run --all-files --hook-stage pre-push`.

This discharges the full-gate criteria of links 01–03, which close together.

### Reuse decision

Candidates come from the brief's librarian placement notes and the architect review.

- `reuse BondPrimitive::Any` (`perceive/smarts/ast.rs:287`) and the standard bond primitives. No dialect flag; `parser.rs:236` / `:294` are untouched.
- `reuse` recursive `$(…)` (`parser.rs:609`) where a label would add a level.
- `reuse compile_def` (`layered.rs:220`). The bare-element fallback (`:225`) stays for XML inputs.
- `reuse MatchOptions::labels` (`perceive/smarts/mod.rs:91`).
- `reuse Perceive::find_aromaticity` (`perceive/builder.rs:130`).
- `generalize OplsDependencyAnalyzer::new` (`deps.rs:60-76`).
- `reuse GromacsTopFfReader::with_skipped_directive` / `GromacsTopFfWriter` (link 01) behind the Python functions. No Python-side parsing.
- `new — Dominance` (private): it replaces `priorities()` rather than sitting beside it.

## Files to create or modify

- `molrs/src/ff/params/oplsaa_typing.rs`
- `molrs/src/ff/params/mod.rs`
- `molrs/src/ff/typifier/opls/layered.rs`
- `molrs/src/ff/typifier/opls/meta.rs`
- `molrs/src/ff/typifier/opls/deps.rs`
- `molrs/src/ff/typifier/opls/typing.rs`
- `molrs/src/ff/typifier/opls/mod.rs`
- `molrs/src/ff/typifier/opls/embedded.rs`
- `molrs-python/src/ff/mod.rs`
- `molrs-python/python/molrs/_lib.pyi`
- `molrs-python/python/molrs/ff/forcefield.py`
- `molrs-python/tests/test_forcefield_gromacs_reader.py` (new)

## Tasks

- [x] Write failing engine tests:
  - dominance in `molrs/src/ff/typifier/opls/layered.rs`: override beats specificity; transitive; layer beats overrides; unrelated candidates rank by priority, size, name; a later level keeps a dominating type and replaces a non-dominating one; cycle and dangling override are build `Err`;
  - private-copy perception in `molrs/src/ff/typifier/opls/typing.rs`;
  - any-type-name label dependencies in `molrs/src/ff/typifier/opls/deps.rs`
- [x] Implement `Dominance` and the dominance merge in `molrs/src/ff/typifier/opls/layered.rs`, deleting `priorities()` / `LAYER_PRIORITY_STRIDE` from `molrs/src/ff/typifier/opls/meta.rs` and the re-export in `molrs/src/ff/typifier/opls/mod.rs`; verify with `cargo mrs-test -- ff::typifier::opls::layered ff::typifier::opls::meta`
- [x] Implement private-copy perception in `molrs/src/ff/typifier/opls/typing.rs` and the label generalization in `molrs/src/ff/typifier/opls/deps.rs`; verify with `cargo mrs-test -- ff::typifier::opls`
- [x] Write failing golden typing tests in `molrs/src/ff/typifier/opls/embedded.rs` for the 15 molecules of Testing strategy (every atom's type plus net charge 0), through `OPLSAATypifier::oplsaa().with_strict(false)`
- [x] Rewrite the rules in `molrs/src/ff/params/oplsaa_typing.rs` per § 1, add opls_150 / opls_178 with overrides, write the header, and update the `OplsRuleRow` doc in `molrs/src/ff/params/mod.rs` (Daylight, explicit H); verify with `cargo mrs-test -- ff::typifier::opls`
- [x] Move the two butadiene tests in `molrs/src/ff/typifier/opls/mod.rs` to methylsilane, with untyped set {0, 1, 5, 6, 7}; verify with `cargo mrs-test -- ff::typifier::opls`
- [x] Update rustdoc per rustdoc style: `typing.rs` "Conflict resolution" / "SMARTS reuse", the `meta.rs` module doc, the `opls/mod.rs` module doc and `oplsaa()`. Add a doctest on `OPLSAATypifier::oplsaa` that types hand-built ethanol and asserts the O is `"opls_154"` (the regression example)
- [x] Write a failing Python seam smoke test `molrs-python/tests/test_forcefield_gromacs_reader.py`: `read_gromacs_top_ff_str` on a string holding `[ defaults ]`, one atomtypes row and a `[ constrainttypes ]` section raises `ValueError` without `skip_directives`, and returns a `ForceField` with `skip_directives=["constrainttypes"]` — called through the public `molrs.ff.read_gromacs_top_ff_str`, never `_lib`
- [x] Implement the Python binder changes in `molrs-python/src/ff/mod.rs` (`skip_directives` on both readers; writers call `GromacsTopFfWriter`; directive-model docstrings), in the public wrappers in `molrs-python/python/molrs/ff/forcefield.py` (keyword-only `skip_directives` passed through; docstrings), and in `molrs-python/python/molrs/_lib.pyi`
- [ ] Run the chain-end full gate: confirm molrs-wasm, molrs-capi, molrs-ffi and molrs-cxxapi build unchanged against the settled API, commit (pre-commit rustfmt + clippy), then run `prek run --all-files --hook-stage pre-push`, discharging the full-gate criteria of links 01–03

## Testing strategy

Unit tests next to the code. Fixtures are hand-built graphs (`add_atom` / `add_bond` / `set_bond_type`, as `opls/mod.rs::butadiene`), or `io::smiles` plus explicit hydrogens (the test-only exception in architecture-rules.md). Every expectation is hand-derived. Filters: `ff::typifier::opls::{layered,typing,deps,embedded}`, `ff::typifier::opls`.

**Engine tests:**

- dominance cases per task 1;
- a Kekulé benzene with `[c;X3;r6]` / `[#1]-[c]` rules types all 12 atoms and keeps its input bond orders;
- `HX = [#1]-[#6;%CX]` sits at level 1.

**Goldens** (shipped rules; explicit H; every atom asserted; Σq = 0):

| Molecule | Expected types |
|---|---|
| N-methylacetamide | acetyl C opls_135, H opls_140; C=O C opls_235, O opls_236; N opls_238; HN opls_241; N-CH₃ C opls_242, H opls_140 |
| 1,3-butadiene | C1/C4 opls_143; C2/C3 opls_150; H opls_144 |
| ethanol | CH₃ opls_135; CH₂ opls_157; O opls_154; HO opls_155; HC opls_140 |
| benzene, aromatic input | C opls_145, H opls_146 |
| benzene, Kekulé input | C opls_145, H opls_146 |
| propylene carbonate | C=O opls_772; exo O opls_771; ring O opls_773; CH opls_775; CH₂ opls_774; CH₃ opls_776; H opls_777 / opls_778 / opls_779 |
| methyl methacrylate | =CH₂ opls_143 (H opls_144); =C< opls_141; C-CH₃ opls_135 (H opls_140); C=O C opls_465; =O opls_466; ester O opls_467; O-CH₃ opls_468 (H opls_469) |
| methyl formate | formyl H opls_279; C opls_465; =O opls_466; O opls_467; CH₃ opls_468; H opls_469 |
| benzonitrile | C≡ opls_261; N opls_262; ipso opls_260; ring opls_145/opls_146 |
| chlorobenzene | Cl opls_264; C-Cl opls_263; others opls_145/opls_146 |
| chloroethane | CH₃ opls_135 (H opls_140); CH₂Cl C opls_152 (H opls_153); Cl opls_151 |
| fluorobenzene | F opls_719; C-F opls_718; others opls_145/opls_146 |
| pyridine | N opls_520; Cα/Cβ/Cγ opls_521 / opls_522 / opls_523; H opls_524 / opls_525 / opls_526 |
| pyrimidine | N opls_530; C2 opls_531; C4/C6 opls_532; C5 opls_533; H2 opls_534; H4/H6 opls_535; H5 opls_536 |
| pyrrole | N opls_542; Cα opls_543; Cβ opls_544; HN opls_545; Hα opls_546; Hβ opls_547 |

(CH₃Cl types under opls_152/153 but, by OPLS construction, is +0.103. It is not a golden.)

**Python:** one seam smoke test (the skip keyword reaches the Rust reader, and the refusal maps to `ValueError`). No chemistry is re-derived.

**Regression example:** the ethanol doctest on `OPLSAATypifier::oplsaa`, run by this link's gate.

## Out of scope

- OPLS improper assignment: routed `/mol:spec` by link 01.
- United-atom typing rules (opls_001–opls_134 stay rule-less); chemistries not listed.
- CL&P: a future separate spec (operator ruling 2026-09-25). molpy deletes its typifiers, `ClpTypifier` included, in its own chain.
- Found debt, routed `/mol:refactor`: `typify_atoms` rebuilds and recompiles the whole `LayeredTypingEngine` on every match (`typing.rs:88`). Building it once changes `OPLSAATypifier::new` to fallible, a public-signature decision of its own.
- wasm/C: bind no GROMACS reader and no OPLSAA (notes.md 2026-09-25 class-set asymmetry). No change.
- molpy changes: its own chain, after molrs tags.
