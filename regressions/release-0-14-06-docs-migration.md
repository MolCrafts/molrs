# Docs migration 0.14 grep gates

Predicates (directory scan, no file-name list):

| Predicate | Expected hits |
|---|---|
| `site-src/**/*.md` python fences containing `import molrs` / `from molrs` / `molrs.` | 0 |
| `site-src` + `python/molrs/compute/**` lines matching `(dt\|lag\|analysis)` near `\bps\b` | 0 |
| `zensical.toml` contains `getting-started/migration-0-14.md` | 1 |
| `zensical.toml` contains `guides/md.md` | 1 |
| `migration-0-14.md` + `guides/md.md` contain `energy_to_md` / `preset_energy_to_md` / `kb_md` / `set_energy_scale` / `prec=` / `resolve_prec` | 0 |
| `.claude/notes/science.md` Time row is `fs` | 1 |

Run by `molrs-python/tests/test_docs_gates.py`.
