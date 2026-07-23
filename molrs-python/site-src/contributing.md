# Contributing to Documentation

The documentation system has one central invariant: public API prose starts in
Rust `///` comments. Zensical pages may explain workflows and concepts, but
reference pages should inject generated API docs or link to generated API docs
instead of copying signatures by hand.

For Python, keep `molrs-python/python/molrs/_lib.pyi` synchronized with the
PyO3 module exports. The freshness guard checks that every class and function
registered in `molrs-python/src/lib.rs` is declared in the stub.

For WASM, build declarations with the same `wasm-pack` flags used by npm
publishing. The generated `pkg/` directory is ignored and must not be committed.

Local documentation loop. The Zensical config lives at
`molrs-python/zensical.toml`, so build and serve from that directory (its
`[doc]` extra pulls in Zensical plus the mkdocstrings Python handler):

```bash
cd molrs-python
pip install -e ".[doc]"                 # zensical + mkdocstrings[python]
python ../scripts/check-python-stub-exports.py
maturin develop                         # build the molrs module so the API reference can introspect it
zensical build                          # reads ./zensical.toml, writes ./site
zensical serve -a localhost:8000        # live preview
```
