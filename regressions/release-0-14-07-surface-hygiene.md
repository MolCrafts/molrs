# Surface hygiene 0.14 grep gates

| Predicate | Expected |
|---|---|
| `read_pdb("missing.pdb")` message contains `missing.pdb` | 1 |
| PyO3 type error on `Sphere` names `center` | 1 |
| `Block.view("xx")` message contains a real column name | 1 |
| glob `molrs-python/examples/*.py` each exit 0 | all |
| binder rustdoc `float32 view` claims | 0 |
| `Exposed as \`molrs.Potentials\`` (should be `molrs.ff.Potentials`) | 0 |
| `.gitignore` `target-aarch64/` | 1 |
| molpy `.gitignore` `benchmarks/md/` | 1 |
| architecture.md has `md` module row, no `FieldSpec`, no Frame `metadata` field | 1 |
| notes.md binder-surface-symmetry names 0.15 and four NeighborQuery consumers | 1 |
