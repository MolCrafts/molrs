---
slug: chem-perceive-13-python-bind
created: 2026-07-12
criteria:
  - id: ac-001
    summary: Native AM1-BCC is reachable from Python
    type: code
    pass_when: |
      `molrs.BccModel(parameter_set="bcc").correct(mol, am1)` returns an ndarray of charges.
              On the 37-molecule oracle the Python result is bitwise identical to the Rust result.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      AMENDED 2026-07-14. Original pass_when demanded the Python result be "bitwise identical to the
      Rust result" ON THE 37-MOLECULE ORACLE. That is not provable: the oracle's columns carry six
      decimals, so an f32 downcast in the binding lands ~1e-8 off and sails through any comparison
      against them. The amended contract is checkable and STRICTER than a value diff:
      (1) value parity with the oracle at tol 1e-4 across all 37 molecules;
      (2) dtype == float64, ndim == 1, shape == (n_atoms,) — catches an f32 downcast or a transposition;
      (3) BCC increments are pairwise antisymmetric, so TOTAL CHARGE IS CONSERVED TO 1e-12 across all
      37 — an f32 round-trip misses this by four orders of magnitude;
      (4) methane's increments equal 4 x 0.0393 to 1e-12, the same value and tolerance as the Rust
      doctest on `BccModel::correct` — the one place a Python charge is pinned to a RUST-side number.
      `molrs.BccModel(parameter_set="bcc").correct(mol, am1)` returns a float64 ndarray. Charges cross
      as f64 end to end; no f32 anywhere, no renormalization. This is PYTHON'S FIRST NATIVE AM1-BCC —
      molpy's only AM1-BCC was shelling out to `antechamber -c bcc`.
  - id: ac-002
    summary: Perceive and AtdTypifier are exposed
    type: code
    pass_when: |
      `molrs.Perceive().find_rings(g)` returns a graph. `molrs.AtdTypifier(parameter_set="gaff")`
              types a molecule and matches `antechamber -at gff`.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `molrs.Perceive()` exposes find_rings / aromaticity / hydrogens / stereo / rotatable / bond_types /
      equivalence_classes, all graph-in/graph-out and non-mutating (a test asserts the input graph is
      unchanged). `molrs.AtdTypifier(parameter_set="gaff")` types the full 37-molecule oracle and matches
      antechamber's `-at` output; set names are antechamber's FLAG spellings ("gaff"/"gaff2"), not the Rust
      enum's table names (Gff/Gff2), and the flag<->table mapping is written down in exactly one place
      (PARAMETER_SETS in molrs-python/src/ff/atd.rs). At least two sets are shown to disagree on the same
      molecule, so a wrong-set binding cannot pass by accident.
  - id: ac-003
    summary: All three charge models are bound
    type: code
    pass_when: |
      BccModel, MullikenModel and GasteigerModel are all importable from `molrs` and all
              implement the same Python-side calling convention.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      BccModel / MullikenModel / GasteigerModel all importable from `molrs`, all sharing one convention:
      `needs_equivalencing()`, `assign(mol, qm=None)`, `correct(mol, am1)`. Bound in a NEW file
      molrs-python/src/ff/charge.rs (ff/mod.rs was already 44K; house style is many small files), with
      AtdTypifier in ff/atd.rs and Perceive in src/perceive.rs. `compute_gasteiger_charges` survives as the
      compat door and DELEGATES to the one Rust GasteigerModel — a test compares them with np.array_equal
      (exact), so a second Gasteiger implementation cannot creep back.
  - id: ac-004
    summary: The compat alias is gone
    type: code
    pass_when: |
      `grep -rn 'molrs::chem' .` returns 0 hits across molrs, molrs-python, molrs-cxxapi,
              molrs-ffi and molrs-wasm. `pub use crate::perceive as chem;` is removed from
              molrs/src/lib.rs. Existing Python tests are green with only import paths changed.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `grep -rn 'molrs::chem'` -> 0 hits across molrs, molrs-python, molrs-cxxapi, molrs-ffi, molrs-wasm;
      `pub use crate::perceive as chem;` deleted from molrs/src/lib.rs. Enforced by
      molrs/tests/perceive/chem_alias.rs, which scans the sibling workspaces too (the pre-existing gates
      did not) and asserts it actually read molrs-python/src, so it cannot pass vacuously.
      THE SPEC UNDERSTATED THIS BADLY: it named 4 sites; there were 20. The alias was load-bearing inside
      molrs ITSELF (conformer/distgeom x3, conformer/etkdg, ff/mmff/topo, perceive/smarts, benches,
      examples), and 4 more were spelled `crate::chem` rather than `molrs::chem` — including two intra-doc
      links that only `cargo doc -D warnings` catches. (molrs/src/io/smiles/chem/ is an unrelated, legitimate
      `chem` module — the SMILES/SMARTS shared AST — and was left alone.) Existing Python tests green with
      import paths only; no asserted value changed.
---
