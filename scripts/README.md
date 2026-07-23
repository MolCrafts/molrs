# Scripts

## Offline table / oracle generators (not part of the test gate)

These scripts need a local AmberTools (and sometimes RDKit) install. They are
**developer regeneration tools**, never imported or executed by CI or
`cargo test` / `pytest`:

| Script | Output |
|--------|--------|
| `gen_param_tables.py` | Committed Rust tables under `molrs/src/ff/params/` |
| `gen_am1bcc_oracle.py` | `molrs-cxxapi/tests/antechamber_oracle.rs` (static numbers) |
| `gen_gaff_energy_oracle.py` | JSON printed for hand-checking energy constants |

Default tests consume only the **committed** outputs. They never call RDKit,
antechamber, sander, or any other third-party scientific package.

## Optional format corpus

```bash
bash scripts/fetch-test-data.sh
```

Clones a chemfiles-derived fixture tree into `tests-data/` (gitignored). This is
**optional** exploratory data — not required by the default CI / pre-commit
gate. Binding smoke tests build their own fixtures with molrs writers.
