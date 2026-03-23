# Test Data

Test data is stored in `molrs-core/target/tests-data/` (not in the source tree).

## Fetch test data

```bash
bash scripts/fetch-test-data.sh
```

## Run tests

```bash
cargo test
```

## CI

Test data is cached in CI using `actions/cache@v4` with key based on the fetch script hash.
