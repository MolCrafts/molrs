---
name: molrs-docs
description: Operate on the docs system (Zensical site, narrative guides, `:::` reference injections, docs.yml CI, READMEs, and `molrs-python/python/molrs/molrs.pyi`). Writes `docs/`, READMEs, and the `.pyi` stub. Distinct from `molrs-doc` (which is the rustdoc reference standard).
argument-hint: "<what to update — pyi | guide <slug> | reference | readme | ci>"
user-invocable: true
---

# molrs-docs

Read `CLAUDE.md` and `docs/doc-system-spec.md` before proposing any
change to the docs system. This skill is a thin workflow wrapper around
the `molrs-docs-engineer` agent, which owns the axis.

## Procedure

1. **Classify the ask** — one of:
   - `pyi` — sync `molrs-python/python/molrs/molrs.pyi` against the
     current `#[pyfunction]` / `#[pyclass]` set in
     `molrs-python/src/**`.
   - `guide <slug>` — author or update a narrative guide
     (`docs/guides/**`, `docs/getting-started/**`).
   - `reference` — update `::: module.Symbol` injections in
     `docs/reference/python.md` or `docs/reference/wasm.md`.
   - `readme` — overhaul `README.md` / `molrs-python/README.md` /
     `molrs-wasm/README.md`.
   - `ci` — edit `.github/workflows/docs.yml` (match flags to the
     publish workflows byte-for-byte where the spec requires it).
2. **Delegate** — spawn `molrs-docs-engineer` agent with the classified
   ask. The agent loads `docs/doc-system-spec.md`, applies the
   invariants, and returns a diff.
3. **Verify** — per the agent's output, either run the local check it
   names (e.g. the `.pyi` freshness guard, `zensical build --strict`)
   or defer to CI with that noted explicitly.

## Rules

- Never hand-write content that duplicates a `///` comment — inject with
  `::: module.Symbol`. Single source of truth for API prose is Rust
  `///`.
- Never commit generated artifacts (`pkg/molwasm.d.ts`, `site/`). The
  `.pyi` is the one hand-maintained exception and that's spelled out in
  the docs-system spec.
- Inactive crates (`molrs-ffi`, `molrs-capi`) must not surface on the
  site. If the agent proposes wiring one in, push back.

## Output

One line: `docs: <kind> updated, <N> files, verified | deferred to CI`.
