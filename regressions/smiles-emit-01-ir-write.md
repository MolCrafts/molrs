# Regression — smiles-emit write path

Public Rust API (via lib tests as executable documentation):

```text
parse_smiles("CCO") → write_smiles → parse → write  (stable)
from_atomistic(to_atomistic(parse("c1ccccc1")), Default) → write_smiles  (parseable)
write_local_smarts(mol, center, Default) → SmartsPattern::parse → has_match
```

Runnable gate: `cargo test -p molcrafts-molrs --lib --features full,filesystem smiles::`

Python: `pytest molrs-python/tests/test_smiles_emit.py`
